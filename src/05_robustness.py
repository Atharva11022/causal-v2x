"""Run multi-seed and sparsity-matched controls for the primary condition."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
PRIMARY = "dropout_20_delay_1"
SEEDS = [42, 142, 242]
EVAL_SEED = 2024
SAMPLES_PER_CONTEXT = 32
MODEL_NAMES = ["causal_masked", "robust_masked", "random_masked", "baseline"]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def random_mask(shape, links, seed):
    generator = np.random.default_rng(seed)
    selected = generator.choice(np.prod(shape), size=links, replace=False)
    mask = np.zeros(np.prod(shape), dtype=np.float32)
    mask[selected] = 1.0
    return mask.reshape(shape)


if __name__ == "__main__":
    generator = load_module("generator", "src/03_generative_replay.py")
    evaluator = load_module("evaluator", "src/04_evaluate.py")
    x_train, c_train, x_val, c_val, x_scaler, c_scaler = generator.load_training_data(
        PRIMARY
    )
    causal = generator.build_mask(PRIMARY)
    robust = generator.build_mask(PRIMARY, robust=True)
    random_link_count = int(causal.sum())
    real = x_scaler.inverse_transform(x_val)
    contexts = torch.tensor(c_val, device=generator.DEVICE)
    evaluator_context = np.repeat(
        c_scaler.inverse_transform(c_val), SAMPLES_PER_CONTEXT, axis=0
    )
    rows = []

    for seed in SEEDS:
        for model_name, mask in [
            ("causal_masked", causal),
            ("robust_masked", robust),
            ("random_masked", random_mask(causal.shape, random_link_count, seed)),
            ("baseline", None),
        ]:
            np.random.seed(seed)
            torch.manual_seed(seed)
            mask_tensor = (
                None if mask is None else torch.tensor(mask, device=generator.DEVICE)
            )
            model, train_time = generator.train_model(
                mask_tensor,
                f"robustness_{model_name}_{seed}",
                x_train,
                c_train,
                x_val,
                c_val,
            )
            latent_generator = torch.Generator(device=generator.DEVICE).manual_seed(
                EVAL_SEED
            )
            latent_bank = torch.randn(
                (len(c_val) * SAMPLES_PER_CONTEXT, generator.LATENT_DIM),
                generator=latent_generator,
                device=generator.DEVICE,
            )
            generated, _ = generator.generate(model, contexts, latent_bank)
            generated = x_scaler.inverse_transform(generated)
            eval_mask = causal if model_name == "baseline" else mask
            linked, unlinked, gap = evaluator.evaluate_generator(
                generated, evaluator_context, eval_mask
            )
            rows.append(
                {
                    "condition": PRIMARY,
                    "seed": seed,
                    "model": model_name,
                    "mask_links": 0 if mask is None else int(mask.sum()),
                    "train_time_sec": train_time,
                    "distribution_mean_mae": float(
                        np.mean(np.abs(generated.mean(axis=0) - real.mean(axis=0)))
                    ),
                    "diversity_std": float(np.mean(generated.std(axis=0))),
                    "linked_corr": linked,
                    "unlinked_corr": unlinked,
                    "consistency_gap": gap,
                }
            )

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_DIR / "robustness_primary.csv", index=False)
    metric_names = [
        "distribution_mean_mae",
        "diversity_std",
        "linked_corr",
        "unlinked_corr",
        "consistency_gap",
    ]
    summary_rows = []
    for model_name, group in detail.groupby("model", sort=False):
        for metric in metric_names:
            values = group[metric].to_numpy()
            mean = float(values.mean())
            standard_error = float(values.std(ddof=1) / np.sqrt(len(values)))
            summary_rows.append(
                {
                    "condition": PRIMARY,
                    "model": model_name,
                    "metric": metric,
                    "mean": mean,
                    "std": float(values.std(ddof=1)),
                    "ci95_low": mean - 1.96 * standard_error,
                    "ci95_high": mean + 1.96 * standard_error,
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / "robustness_primary_summary.csv", index=False
    )
    print(detail.to_string(index=False))
    print(pd.DataFrame(summary_rows).to_string(index=False))
