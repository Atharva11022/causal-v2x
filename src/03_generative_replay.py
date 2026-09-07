"""Generate near-miss reactions using a stability-aware causal mask."""

import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_LABELS = [
    "clean",
    "dropout_10",
    "dropout_25",
    "dropout_50",
    "delay_1",
    "delay_2",
    "dropout_20_delay_1",
]
TARGET_COLS = ["ego_accel", "ego_speed"]
CONTEXT_COLS = [
    "lead_speed",
    "lead_accel",
    "gap_distance",
    "relative_speed",
    "lead_observed",
]
LATENT_DIM = 4
HIDDEN_DIM = 16
EPOCHS = 200
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_training_data(
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    events = pd.read_csv(DATA_DIR / f"near_miss_events_{label}.csv")
    required = {"scenario_id", "ego_accel", "ego_speed", *CONTEXT_COLS}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing generator columns: {sorted(missing)}")

    scenario_ids = events["scenario_id"].drop_duplicates().to_numpy()
    train_ids, val_ids = train_test_split(
        scenario_ids, test_size=0.2, random_state=SEED
    )
    train = events[events["scenario_id"].isin(train_ids)].copy()
    val = events[events["scenario_id"].isin(val_ids)].copy()

    x_scaler = StandardScaler().fit(train[TARGET_COLS])
    c_scaler = StandardScaler().fit(
        train[CONTEXT_COLS].fillna(train[CONTEXT_COLS].median())
    )

    def transform(frame: pd.DataFrame):
        context = frame[CONTEXT_COLS].fillna(train[CONTEXT_COLS].median())
        return (
            x_scaler.transform(frame[TARGET_COLS]).astype(np.float32),
            c_scaler.transform(context).astype(np.float32),
        )

    x_train, c_train = transform(train)
    x_val, c_val = transform(val)
    return x_train, c_train, x_val, c_val, x_scaler, c_scaler


def build_mask(label: str, robust: bool = False) -> np.ndarray:
    stability = pd.read_csv(OUT_DIR / "causal_edge_stability.csv")
    stable = (
        pd.read_csv(OUT_DIR / "causal_edges_robust.csv")
        if robust
        else stability[stability["degradation"].eq(label) & stability["stable"]]
    )
    mask = np.zeros((len(CONTEXT_COLS), len(TARGET_COLS)), dtype=np.float32)
    for _, edge in stable.iterrows():
        endpoints = {edge["source"], edge["target"]}
        for context_index, context_name in enumerate(CONTEXT_COLS):
            graph_name = context_name
            if graph_name in endpoints:
                for target_index, target_name in enumerate(TARGET_COLS):
                    if target_name in endpoints:
                        mask[context_index, target_index] = 1.0
    np.save(OUT_DIR / "causal_mask.npy", mask)
    source = "robust-all-conditions" if robust else label
    print(f"Mask from {source}: {int(mask.sum())}/{mask.size} context-target links")
    return mask


class MaskedLinear(nn.Module):
    def __init__(self, input_dim, output_dim, mask=None):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.register_buffer(
            "mask", torch.as_tensor(mask.T) if mask is not None else None
        )

    def forward(self, values):
        if self.mask is None:
            return self.linear(values)
        weights = self.linear.weight * self.mask
        return nn.functional.linear(values, weights, self.linear.bias)


class CVAE(nn.Module):
    def __init__(self, causal_mask=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2 + len(CONTEXT_COLS), HIDDEN_DIM), nn.ReLU()
        )
        self.mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.decoder_hidden = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.decoder_output = nn.Linear(HIDDEN_DIM, 2)
        self.context_output = MaskedLinear(len(CONTEXT_COLS), 2, causal_mask)

    def encode(self, x, context):
        hidden = self.encoder(torch.cat([x, context], dim=1))
        return self.mu(hidden), self.logvar(hidden)

    def decode(self, latent, context):
        generated = self.decoder_output(torch.relu(self.decoder_hidden(latent)))
        return generated + self.context_output(context)

    def forward(self, x, context):
        mu, logvar = self.encode(x, context)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(latent, context), mu, logvar


def loss_function(prediction, target, mu, logvar):
    reconstruction = nn.functional.mse_loss(prediction, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return reconstruction + 0.01 * kl, reconstruction


def train_model(mask, tag, x_train, c_train, x_val, c_val):
    model = CVAE(mask).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x_train_t = torch.tensor(x_train, device=DEVICE)
    c_train_t = torch.tensor(c_train, device=DEVICE)
    x_val_t = torch.tensor(x_val, device=DEVICE)
    c_val_t = torch.tensor(c_val, device=DEVICE)
    started = time.time()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        predicted, mu, logvar = model(x_train_t, c_train_t)
        total, reconstruction = loss_function(predicted, x_train_t, mu, logvar)
        total.backward()
        optimizer.step()

        if epoch == 0 or epoch == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                val_prediction, val_mu, val_logvar = model(x_val_t, c_val_t)
                _, val_reconstruction = loss_function(
                    val_prediction, x_val_t, val_mu, val_logvar
                )
            print(
                f"{tag} epoch={epoch} train_recon={reconstruction.item():.4f} val_recon={val_reconstruction.item():.4f}"
            )

    elapsed = time.time() - started
    torch.save(model.state_dict(), OUT_DIR / f"cvae_{tag}.pt")
    return model, elapsed


def generate(model, contexts, samples_per_context=5):
    model.eval()
    repeated = contexts.repeat_interleave(samples_per_context, dim=0)
    with torch.no_grad():
        latent = torch.randn(len(repeated), LATENT_DIM, device=DEVICE)
        generated = model.decode(latent, repeated)
    return generated.cpu().numpy(), repeated.cpu().numpy()


if __name__ == "__main__":
    performance = []
    for label in EXPERIMENT_LABELS:
        print(f"\n=== Generating under {label} ===")
        x_train, c_train, x_val, c_val, x_scaler, c_scaler = load_training_data(label)
        mask = build_mask(label)
        mask_t = torch.tensor(mask, device=DEVICE)
        masked_model, masked_time = train_model(
            mask_t, f"causal_masked_{label}", x_train, c_train, x_val, c_val
        )
        baseline_model, baseline_time = train_model(
            None, f"baseline_{label}", x_train, c_train, x_val, c_val
        )
        robust_model = None
        robust_time = 0.0
        robust_mask = None
        if label == "dropout_20_delay_1":
            robust_mask = build_mask(label, robust=True)
            robust_model, robust_time = train_model(
                torch.tensor(robust_mask, device=DEVICE),
                "robust_masked_dropout_20_delay_1",
                x_train,
                c_train,
                x_val,
                c_val,
            )

        c_val_t = torch.tensor(c_val, device=DEVICE)
        generated_masked, context_masked = generate(masked_model, c_val_t)
        generated_baseline, _ = generate(baseline_model, c_val_t)
        suffix = "" if label == "clean" else f"_{label}"
        np.save(
            OUT_DIR / f"gen_masked_X{suffix}.npy",
            x_scaler.inverse_transform(generated_masked),
        )
        np.save(
            OUT_DIR / f"gen_baseline_X{suffix}.npy",
            x_scaler.inverse_transform(generated_baseline),
        )
        np.save(
            OUT_DIR / f"gen_context{suffix}.npy",
            c_scaler.inverse_transform(context_masked),
        )
        np.save(OUT_DIR / f"causal_mask{suffix}.npy", mask)
        np.save(OUT_DIR / f"real_val_X{suffix}.npy", x_scaler.inverse_transform(x_val))
        np.save(OUT_DIR / f"real_val_C{suffix}.npy", c_scaler.inverse_transform(c_val))
        if robust_model is not None:
            generated_robust, _ = generate(robust_model, c_val_t)
            np.save(
                OUT_DIR / "gen_robust_X_dropout_20_delay_1.npy",
                x_scaler.inverse_transform(generated_robust),
            )
            np.save(
                OUT_DIR / "causal_mask_robust_dropout_20_delay_1.npy",
                robust_mask,
            )

        performance.append(
            {
                "degradation": label,
                "device": DEVICE,
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "epochs": EPOCHS,
                "random_seed": SEED,
                "causal_mask_links": int(mask.sum()),
                "causal_masked_train_time_sec": masked_time,
                "baseline_train_time_sec": baseline_time,
                "robust_masked_train_time_sec": robust_time,
            }
        )

    pd.DataFrame(performance).to_csv(OUT_DIR / "performance_metrics.csv", index=False)
    print("Generative replay complete for all degradation conditions.")
