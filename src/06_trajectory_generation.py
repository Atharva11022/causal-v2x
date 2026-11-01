"""Generate five-step near-miss reaction trajectories under degraded V2X."""

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
EXPERIMENT_LABELS = [
    "clean",
    "dropout_10",
    "dropout_25",
    "dropout_50",
    "delay_1",
    "delay_2",
    "dropout_20_delay_1",
]
CONTEXT_COLS = [
    "lead_speed",
    "lead_accel",
    "gap_distance",
    "relative_speed",
    "lead_observed",
]
TARGET_NAMES = ["ego_speed", "ego_accel"]
HORIZON = 5
LATENT_DIM = 8
HIDDEN_DIM = 32
EPOCHS = 200
SEED = 42
EVAL_SEED = 2024
SAMPLES_PER_CONTEXT = 32
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_window_data(label):
    windows = pd.read_csv(DATA_DIR / f"near_miss_windows_{label}.csv")
    target_cols = [
        f"future_{target}_{step}"
        for step in range(1, HORIZON + 1)
        for target in TARGET_NAMES
    ]
    required = {"scenario_id", *CONTEXT_COLS, *target_cols}
    missing = required.difference(windows.columns)
    if missing:
        raise ValueError(f"Missing trajectory-window columns: {sorted(missing)}")

    scenario_ids = windows["scenario_id"].drop_duplicates().to_numpy()
    train_ids, val_ids = train_test_split(
        scenario_ids, test_size=0.2, random_state=SEED
    )
    train = windows[windows["scenario_id"].isin(train_ids)].copy()
    val = windows[windows["scenario_id"].isin(val_ids)].copy()
    target_scaler = StandardScaler().fit(train[target_cols])
    context_scaler = StandardScaler().fit(
        train[CONTEXT_COLS].fillna(train[CONTEXT_COLS].median())
    )

    def transform(frame):
        context = frame[CONTEXT_COLS].fillna(train[CONTEXT_COLS].median())
        return (
            target_scaler.transform(frame[target_cols]).astype(np.float32),
            context_scaler.transform(context).astype(np.float32),
        )

    y_train, c_train = transform(train)
    y_val, c_val = transform(val)
    return y_train, c_train, y_val, c_val, target_scaler, context_scaler


def build_mask(label, robust=False):
    stability = pd.read_csv(OUT_DIR / "causal_edge_stability.csv")
    stable = (
        pd.read_csv(OUT_DIR / "causal_edges_robust.csv")
        if robust
        else stability[stability["degradation"].eq(label) & stability["stable"]]
    )
    mask = np.zeros((len(CONTEXT_COLS), len(TARGET_NAMES)), dtype=np.float32)
    for _, edge in stable.iterrows():
        endpoints = {edge["source"], edge["target"]}
        for context_index, context_name in enumerate(CONTEXT_COLS):
            if context_name not in endpoints:
                continue
            for target_index, target_name in enumerate(TARGET_NAMES):
                if target_name in endpoints:
                    mask[context_index, target_index] = 1.0
    return np.tile(mask, (1, HORIZON))


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
        return nn.functional.linear(
            values, self.linear.weight * self.mask, self.linear.bias
        )


class SequenceCVAE(nn.Module):
    def __init__(self, causal_mask=None):
        super().__init__()
        target_dim = HORIZON * len(TARGET_NAMES)
        self.encoder = nn.Sequential(
            nn.Linear(target_dim + len(CONTEXT_COLS), HIDDEN_DIM), nn.ReLU()
        )
        self.mu = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.decoder_hidden = nn.Linear(LATENT_DIM, HIDDEN_DIM)
        self.decoder_output = nn.Linear(HIDDEN_DIM, target_dim)
        self.context_output = MaskedLinear(len(CONTEXT_COLS), target_dim, causal_mask)

    def encode(self, target, context):
        hidden = self.encoder(torch.cat([target, context], dim=1))
        return self.mu(hidden), self.logvar(hidden)

    def decode(self, latent, context):
        generated = self.decoder_output(torch.relu(self.decoder_hidden(latent)))
        return generated + self.context_output(context)

    def forward(self, target, context):
        mu, logvar = self.encode(target, context)
        latent = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        return self.decode(latent, context), mu, logvar


def loss_function(prediction, target, mu, logvar):
    reconstruction = nn.functional.mse_loss(prediction, target)
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return reconstruction + 0.01 * kl, reconstruction


def train_model(mask, tag, y_train, c_train, y_val, c_val):
    model = SequenceCVAE(mask).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    y_train_t = torch.tensor(y_train, device=DEVICE)
    c_train_t = torch.tensor(c_train, device=DEVICE)
    y_val_t = torch.tensor(y_val, device=DEVICE)
    c_val_t = torch.tensor(c_val, device=DEVICE)
    started = time.time()
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        prediction, mu, logvar = model(y_train_t, c_train_t)
        total, reconstruction = loss_function(prediction, y_train_t, mu, logvar)
        total.backward()
        optimizer.step()
        if epoch in {0, EPOCHS - 1}:
            model.eval()
            with torch.no_grad():
                val_prediction, val_mu, val_logvar = model(y_val_t, c_val_t)
                _, val_reconstruction = loss_function(
                    val_prediction, y_val_t, val_mu, val_logvar
                )
            print(
                f"{tag} epoch={epoch} train_recon={reconstruction.item():.4f} "
                f"val_recon={val_reconstruction.item():.4f}"
            )
    torch.save(model.state_dict(), OUT_DIR / f"cvae_{tag}.pt")
    return model, time.time() - started


def generate(model, contexts, latent_bank):
    model.eval()
    repeated = contexts.repeat_interleave(SAMPLES_PER_CONTEXT, dim=0)
    with torch.no_grad():
        generated = model.decode(latent_bank, repeated)
    return generated.cpu().numpy()


def reshape_targets(values):
    return values.reshape(len(values), HORIZON, len(TARGET_NAMES))


def physical_metrics(generated):
    trajectories = reshape_targets(generated)
    speeds = trajectories[:, :, 0]
    accelerations = trajectories[:, :, 1]
    return {
        "negative_speed_rate": float(np.mean(speeds < 0.0)),
        "accel_limit_violation_rate": float(np.mean(np.abs(accelerations) > 15.0)),
        "jerk_limit_rate": float(
            np.mean(np.abs(np.diff(accelerations, axis=1) / 0.1) > 30.0)
        ),
    }


if __name__ == "__main__":
    performance = []
    for label in EXPERIMENT_LABELS:
        print(f"\n=== Sequence generation under {label} ===")
        y_train, c_train, y_val, c_val, target_scaler, context_scaler = (
            load_window_data(label)
        )
        mask = build_mask(label)
        masked_model, masked_time = train_model(
            torch.tensor(mask, device=DEVICE),
            f"trajectory_masked_{label}",
            y_train,
            c_train,
            y_val,
            c_val,
        )
        baseline_model, baseline_time = train_model(
            None,
            f"trajectory_baseline_{label}",
            y_train,
            c_train,
            y_val,
            c_val,
        )
        robust_model = None
        robust_time = 0.0
        robust_mask = None
        if label == "dropout_20_delay_1":
            robust_mask = build_mask(label, robust=True)
            robust_model, robust_time = train_model(
                torch.tensor(robust_mask, device=DEVICE),
                "trajectory_robust_dropout_20_delay_1",
                y_train,
                c_train,
                y_val,
                c_val,
            )

        contexts = torch.tensor(c_val, device=DEVICE)
        latent_generator = torch.Generator(device=DEVICE).manual_seed(EVAL_SEED)
        latent_bank = torch.randn(
            (len(c_val) * SAMPLES_PER_CONTEXT, LATENT_DIM),
            generator=latent_generator,
            device=DEVICE,
        )
        suffix = "" if label == "clean" else f"_{label}"
        real = target_scaler.inverse_transform(y_val)
        generated_models = {
            "trajectory_masked": generate(masked_model, contexts, latent_bank),
            "trajectory_baseline": generate(baseline_model, contexts, latent_bank),
        }
        if robust_model is not None:
            generated_models["trajectory_robust"] = generate(
                robust_model, contexts, latent_bank
            )
        np.save(OUT_DIR / f"trajectory_real_X{suffix}.npy", real)
        np.save(
            OUT_DIR / f"trajectory_context{suffix}.npy",
            context_scaler.inverse_transform(c_val),
        )
        for model_name, generated in generated_models.items():
            generated = target_scaler.inverse_transform(generated)
            np.save(OUT_DIR / f"{model_name}_X{suffix}.npy", generated)
            metrics = physical_metrics(generated)
            metrics.update(
                {
                    "degradation": label,
                    "model": model_name,
                    "mean_mae": float(
                        np.mean(np.abs(generated.mean(0) - real.mean(0)))
                    ),
                    "diversity_std": float(np.mean(generated.std(0))),
                }
            )
            performance.append(metrics)

        performance.append(
            {
                "degradation": label,
                "model": "training_metadata",
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "random_seed": SEED,
                "evaluation_seed": EVAL_SEED,
                "horizon": HORIZON,
                "samples_per_context": SAMPLES_PER_CONTEXT,
                "masked_train_time_sec": masked_time,
                "baseline_train_time_sec": baseline_time,
                "robust_train_time_sec": robust_time,
            }
        )

    pd.DataFrame(performance).to_csv(
        OUT_DIR / "trajectory_generation_metrics.csv", index=False
    )
    print("Trajectory generation complete for all degradation conditions.")
