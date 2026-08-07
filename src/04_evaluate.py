"""
Phase 6 — Evaluation.
Run: python 05_evaluate.py

Compares the causal-masked CVAE vs. the unconstrained baseline CVAE
(from Phase 5) on:
  1) Reconstruction accuracy vs. real near-miss data
  2) Diversity of generated samples
  3) Causal consistency: does the generated ego reaction correlate
     with context features the causal graph says it should — and NOT
     correlate with ones it doesn't?
Saves outputs/evaluation_summary.csv and a comparison plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGET_COLS  = ["v_Acc", "v_Vel"]
CONTEXT_COLS = ["lead_vel", "lead_acc", "Space_Headway", "Time_Headway", "rel_speed_to_lead"]

real_X = np.load("outputs/real_val_X.npy")
real_C = np.load("outputs/real_val_C.npy")
gen_masked = np.load("outputs/gen_masked_X.npy")
gen_baseline = np.load("outputs/gen_baseline_X.npy")
context = np.load("outputs/gen_context.npy")
mask = np.load("outputs/causal_mask.npy")  # (context_dim, target_dim)

results = {}

# ------------------------------------------------------------
# 1. Reconstruction accuracy (distributional closeness to real data)
#    Compare mean/std of generated vs. real target features.
# ------------------------------------------------------------

for name, gen in [("causal_masked", gen_masked), ("baseline", gen_baseline)]:
    mae_mean = np.mean(np.abs(gen.mean(axis=0) - real_X.mean(axis=0)))
    results[f"{name}_dist_mean_MAE"] = mae_mean

# ------------------------------------------------------------
# 2. Diversity — variance of generated samples (too low = mode collapse)
# ------------------------------------------------------------

for name, gen in [("causal_masked", gen_masked), ("baseline", gen_baseline),
                   ("real_data", real_X)]:
    results[f"{name}_diversity_std"] = float(np.mean(gen.std(axis=0)))

# ------------------------------------------------------------
# 3. Causal consistency score
#    For each context feature, compute |correlation| with each target
#    feature in the GENERATED data, then compare against what the
#    causal graph says should be linked (mask==1) vs not (mask==0).
# ------------------------------------------------------------

def causal_consistency(gen_X, context, mask):
    n_c, n_t = mask.shape
    corr_matrix = np.zeros((n_c, n_t))
    for i in range(n_c):
        for j in range(n_t):
            if np.std(context[:, i]) > 1e-8 and np.std(gen_X[:, j]) > 1e-8:
                corr_matrix[i, j] = abs(np.corrcoef(context[:, i], gen_X[:, j])[0, 1])

    linked_corr = corr_matrix[mask == 1].mean() if (mask == 1).any() else np.nan
    unlinked_corr = corr_matrix[mask == 0].mean() if (mask == 0).any() else np.nan
    return linked_corr, unlinked_corr, corr_matrix

linked_m, unlinked_m, corr_m = causal_consistency(gen_masked, context, mask)
linked_b, unlinked_b, corr_b = causal_consistency(gen_baseline, context, mask)

results["causal_masked_linked_corr"] = linked_m
results["causal_masked_unlinked_corr"] = unlinked_m
results["baseline_linked_corr"] = linked_b
results["baseline_unlinked_corr"] = unlinked_b

# A good constrained model: high linked_corr, low unlinked_corr.
# A good "causal consistency gap" = linked_corr - unlinked_corr, should be
# larger for the causal-masked model than the baseline.
results["causal_masked_consistency_gap"] = linked_m - unlinked_m
results["baseline_consistency_gap"] = linked_b - unlinked_b

# ------------------------------------------------------------
# PRINT + SAVE
# ------------------------------------------------------------

print("\n=== Evaluation summary ===")
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if v == v else f"  {k}: n/a")

pd.DataFrame([results]).to_csv("outputs/evaluation_summary.csv", index=False)
print("\nSaved: outputs/evaluation_summary.csv")

# ------------------------------------------------------------
# PLOT: consistency gap comparison
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4.5))
models = ["Causal-masked", "Baseline"]
gaps = [results["causal_masked_consistency_gap"], results["baseline_consistency_gap"]]
ax.bar(models, gaps, color=["#4a90d9", "#d97a4a"])
ax.set_ylabel("Causal consistency gap\n(linked corr - unlinked corr)")
ax.set_title("Causal consistency: masked generator vs. baseline")
plt.tight_layout()
plt.savefig("outputs/causal_consistency_comparison.png", dpi=150)
print("Saved: outputs/causal_consistency_comparison.png")

print("\nPhase 6 complete. This plot + evaluation_summary.csv is your objective-8 evidence.")
