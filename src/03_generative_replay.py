"""
Phase 5 — Generative replay of near-miss scenarios.
Run: python 04_generative_replay.py

Trains TWO small conditional VAEs on data/near_miss_events.csv:
  1) causal-masked CVAE  -> the ego "reaction" is generated only from
     context features that Phase 4's causal graph actually links to it.
  2) unconstrained CVAE  -> baseline, fully dense connections.

Saves both trained models and a batch of generated samples from each,
for Phase 6 evaluation.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

os.makedirs("outputs", exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

events = pd.read_csv("data/near_miss_events.csv")

TARGET_COLS  = ["v_Acc", "v_Vel"]                                           # ego "reaction"
CONTEXT_COLS = ["lead_vel", "lead_acc", "Space_Headway", "Time_Headway", "rel_speed_to_lead"]  # V2X-proxy context

X = events[TARGET_COLS].to_numpy(dtype=np.float32)
C = events[CONTEXT_COLS].to_numpy(dtype=np.float32)

x_scaler = StandardScaler().fit(X)
c_scaler = StandardScaler().fit(C)
X = x_scaler.transform(X)
C = c_scaler.transform(C)

X_train, X_val, C_train, C_val = train_test_split(X, C, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train, device=DEVICE)
C_train_t = torch.tensor(C_train, device=DEVICE)
X_val_t   = torch.tensor(X_val, device=DEVICE)
C_val_t   = torch.tensor(C_val, device=DEVICE)

print(f"Train events: {len(X_train)}, Val events: {len(X_val)}")

# ------------------------------------------------------------
# 2. BUILD THE CAUSAL MASK  (context_features -> target_features)
#    from Phase 4's causal_edges_v2x.csv
# ------------------------------------------------------------

# Map aggregated causal-graph node names -> frame-level feature names used here
TARGET_MAP  = {"mean_accel": "v_Acc", "mean_speed": "v_Vel"}
CONTEXT_MAP = {
    "mean_lead_vel": "lead_vel",
    "mean_lead_acc": "lead_acc",
    "space_headway_mean": "Space_Headway",
    "min_time_headway": "Time_Headway",
    "rel_speed_to_lead": "rel_speed_to_lead",
}

edges = pd.read_csv("outputs/causal_edges_v2x.csv")

mask = np.zeros((len(CONTEXT_COLS), len(TARGET_COLS)), dtype=np.float32)
n_edges_used = 0
for _, row in edges.iterrows():
    src, tgt = row["source"], row["target"]
    # check both directions since PC edges aren't always fully oriented
    for a, b in [(src, tgt), (tgt, src)]:
        if a in CONTEXT_MAP and b in TARGET_MAP:
            ci = CONTEXT_COLS.index(CONTEXT_MAP[a])
            ti = TARGET_COLS.index(TARGET_MAP[b])
            mask[ci, ti] = 1.0
            n_edges_used += 1

print(f"Causal mask built: {n_edges_used} context->target links active out of {mask.size} possible")
if n_edges_used == 0:
    print("WARNING: no causal edges mapped — check causal_edges_v2x.csv node names, "
          "or the constrained model will get zero context (degenerates to unconditional).")

mask_t = torch.tensor(mask, device=DEVICE)

# ------------------------------------------------------------
# 3. MODEL
# ------------------------------------------------------------

LATENT_DIM = 4
HIDDEN_DIM = 16
X_DIM = len(TARGET_COLS)
C_DIM = len(CONTEXT_COLS)

class MaskedLinear(nn.Module):
    """Linear layer where some input->output connections are forced to zero."""
    def __init__(self, in_dim, out_dim, mask=None):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.mask = mask  # shape (in_dim, out_dim) or None (=unconstrained)

    def forward(self, x):
        if self.mask is None:
            return self.linear(x)
        # zero out masked-off weights each forward pass
        w = self.linear.weight * self.mask.T
        return torch.nn.functional.linear(x, w, self.linear.bias)

class CVAE(nn.Module):
    def __init__(self, x_dim, c_dim, latent_dim, hidden_dim, causal_mask=None):
        super().__init__()
        # Encoder sees x + c (full info; only the DECODER's context path is masked,
        # since the mask represents which context vars are ALLOWED to drive the
        # generated reaction — training/inference use is via decode()).
        self.enc = nn.Sequential(
            nn.Linear(x_dim + c_dim, hidden_dim), nn.ReLU(),
        )
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)

        # context_to_output: mask shape is exactly (c_dim, x_dim) — a direct,
        # causally-gated contribution from each context feature to each target
        # feature, matching the causal graph edges one-to-one.
        self.context_to_output = MaskedLinear(c_dim, x_dim, mask=causal_mask)
        self.dec_hidden = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, x_dim)

    def encode(self, x, c):
        h = self.enc(torch.cat([x, c], dim=-1))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        h = torch.relu(self.dec_hidden(z))
        return self.dec_out(h) + self.context_to_output(c)

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparam(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar

def vae_loss(x_hat, x, mu, logvar):
    recon = nn.functional.mse_loss(x_hat, x, reduction="mean")
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + 0.01 * kld, recon, kld

# ------------------------------------------------------------
# 4. TRAIN
# ------------------------------------------------------------

def train_model(causal_mask, tag, epochs=300, lr=1e-3):
    model = CVAE(X_DIM, C_DIM, LATENT_DIM, HIDDEN_DIM, causal_mask=causal_mask).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        x_hat, mu, logvar = model(X_train_t, C_train_t)
        loss, recon, kld = vae_loss(x_hat, X_train_t, mu, logvar)
        loss.backward()
        opt.step()

        if epoch % 50 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                xv_hat, muv, logvarv = model(X_val_t, C_val_t)
                val_loss, val_recon, _ = vae_loss(xv_hat, X_val_t, muv, logvarv)
            print(f"[{tag}] epoch {epoch:4d}  train_recon={recon.item():.4f}  val_recon={val_recon.item():.4f}")

    torch.save(model.state_dict(), f"outputs/cvae_{tag}.pt")
    return model

print("\nTraining CAUSAL-MASKED CVAE ...")
model_masked = train_model(causal_mask=mask_t, tag="causal_masked")

print("\nTraining UNCONSTRAINED baseline CVAE ...")
model_baseline = train_model(causal_mask=None, tag="baseline")

# ------------------------------------------------------------
# 5. GENERATE SAMPLES (for Phase 6 evaluation)
# ------------------------------------------------------------

def generate(model, C_context, n_samples_per_context=5):
    model.eval()
    with torch.no_grad():
        C_rep = C_context.repeat_interleave(n_samples_per_context, dim=0)
        z = torch.randn(C_rep.shape[0], LATENT_DIM, device=DEVICE)
        x_gen = model.decode(z, C_rep)
    return x_gen.cpu().numpy(), C_rep.cpu().numpy()

gen_masked, ctx_masked = generate(model_masked, C_val_t)
gen_baseline, ctx_baseline = generate(model_baseline, C_val_t)

np.save("outputs/gen_masked_X.npy", x_scaler.inverse_transform(gen_masked))
np.save("outputs/gen_baseline_X.npy", x_scaler.inverse_transform(gen_baseline))
np.save("outputs/gen_context.npy", c_scaler.inverse_transform(ctx_masked))
np.save("outputs/real_val_X.npy", x_scaler.inverse_transform(X_val))
np.save("outputs/real_val_C.npy", c_scaler.inverse_transform(C_val))
np.save("outputs/causal_mask.npy", mask)

print("\nSaved generated samples and models to outputs/")
print("Phase 5 complete.")
