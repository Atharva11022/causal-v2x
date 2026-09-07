"""Evaluate reliable generation under degraded V2X communication."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path("outputs")
TARGET_COLS = ["ego_accel", "ego_speed"]
CONTEXT_COLS = [
    "lead_speed",
    "lead_accel",
    "gap_distance",
    "relative_speed",
    "lead_observed",
]
EXPERIMENT_LABELS = [
    "clean",
    "dropout_10",
    "dropout_25",
    "dropout_50",
    "delay_1",
    "delay_2",
    "dropout_20_delay_1",
]


def correlation_matrix(generated, context):
    matrix = np.zeros((context.shape[1], generated.shape[1]))
    for context_index in range(context.shape[1]):
        for target_index in range(generated.shape[1]):
            if (
                np.std(context[:, context_index]) > 1e-8
                and np.std(generated[:, target_index]) > 1e-8
            ):
                matrix[context_index, target_index] = abs(
                    np.corrcoef(context[:, context_index], generated[:, target_index])[
                        0, 1
                    ]
                )
    return matrix


def evaluate_generator(generated, context, mask):
    correlations = correlation_matrix(generated, context)
    linked = correlations[mask == 1].mean() if mask.any() else np.nan
    unlinked = correlations[mask == 0].mean() if (~mask.astype(bool)).any() else np.nan
    return float(linked), float(unlinked), float(linked - unlinked)


if __name__ == "__main__":
    rows = []
    for label in EXPERIMENT_LABELS:
        suffix = "" if label == "clean" else f"_{label}"
        real = np.load(OUT_DIR / f"real_val_X{suffix}.npy")
        context = np.load(OUT_DIR / f"gen_context{suffix}.npy")
        mask = np.load(OUT_DIR / f"causal_mask{suffix}.npy").astype(bool)
        for name in ["causal_masked", "baseline"]:
            file_name = "masked" if name == "causal_masked" else name
            generated = np.load(OUT_DIR / f"gen_{file_name}_X{suffix}.npy")
            linked, unlinked, gap = evaluate_generator(generated, context, mask)
            rows.append(
                {
                    "degradation": label,
                    "model": name,
                    "causal_mask_links": int(mask.sum())
                    if name == "causal_masked"
                    else np.nan,
                    "distribution_mean_mae": float(
                        np.mean(np.abs(generated.mean(axis=0) - real.mean(axis=0)))
                    ),
                    "diversity_std": float(np.mean(generated.std(axis=0))),
                    "linked_corr": linked,
                    "unlinked_corr": unlinked,
                    "consistency_gap": gap,
                }
            )
        if label == "dropout_20_delay_1":
            robust = np.load(OUT_DIR / "gen_robust_X_dropout_20_delay_1.npy")
            robust_mask = np.load(
                OUT_DIR / "causal_mask_robust_dropout_20_delay_1.npy"
            ).astype(bool)
            linked, unlinked, gap = evaluate_generator(robust, context, robust_mask)
            rows.append(
                {
                    "degradation": label,
                    "model": "robust_masked",
                    "causal_mask_links": int(robust_mask.sum()),
                    "distribution_mean_mae": float(
                        np.mean(np.abs(robust.mean(axis=0) - real.mean(axis=0)))
                    ),
                    "diversity_std": float(np.mean(robust.std(axis=0))),
                    "linked_corr": linked,
                    "unlinked_corr": unlinked,
                    "consistency_gap": gap,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "evaluation_summary.csv", index=False)
    print(summary.to_string(index=False))

    figure, axis = plt.subplots(figsize=(8, 4.5))
    for model, color in [("causal_masked", "#4a90d9"), ("baseline", "#d97a4a")]:
        subset = summary[summary["model"].eq(model)]
        axis.plot(
            subset["degradation"],
            subset["consistency_gap"],
            "o-",
            label=model,
            color=color,
        )
    axis.set_ylabel("Linked correlation - unlinked correlation")
    axis.set_title("Causal consistency under degraded V2X")
    axis.tick_params(axis="x", rotation=35)
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT_DIR / "causal_consistency_comparison.png", dpi=150)
    print("Saved evaluation_summary.csv and causal_consistency_comparison.png")
