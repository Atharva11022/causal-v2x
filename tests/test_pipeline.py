import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.preprocess = load_module("preprocess", "src/01_preprocess.py")
        cls.discovery = load_module("discovery", "src/02_causal_discovery.py")
        cls.raw = cls.preprocess.load_tfd(max_scenarios=2)
        cls.interactions = cls.preprocess.build_interactions(cls.raw)

    def test_tfd_subset_has_real_interactions(self):
        self.assertGreater(len(self.raw), 0)
        self.assertGreater(len(self.interactions), 0)
        self.assertTrue(
            {"ego_speed", "lead_speed", "gap_distance", "relative_speed"}.issubset(
                self.interactions.columns
            )
        )

    def test_near_miss_labels_are_clean_and_boolean(self):
        events = self.preprocess.extract_near_miss_events(self.interactions)
        self.assertGreater(len(events), 0)
        self.assertTrue(events["near_miss"].all())
        self.assertTrue(np.isfinite(events["ego_accel"]).all())

    def test_degradation_preserves_targets_and_changes_context(self):
        delayed = self.preprocess.simulate_degradation(
            self.interactions, delay_frames=2
        )
        dropped = self.preprocess.simulate_degradation(
            self.interactions, dropout_rate=0.25
        )
        np.testing.assert_array_equal(
            delayed["ego_accel"].to_numpy(), self.interactions["ego_accel"].to_numpy()
        )
        self.assertGreater(delayed["lead_speed"].isna().sum(), 0)
        self.assertGreater(dropped["lead_speed"].isna().sum(), 0)

    def test_delay_requires_timestamp_and_same_lead(self):
        frame = pd.DataFrame(
            [
                {
                    "scenario_id": "synthetic",
                    "ego_id": 1,
                    "lead_id": 10,
                    "timestamp": 0.0,
                    "lead_speed": 10.0,
                    "lead_accel": 1.0,
                    "gap_distance": 20.0,
                    "relative_speed": 2.0,
                    "ego_accel": 0.0,
                },
                {
                    "scenario_id": "synthetic",
                    "ego_id": 1,
                    "lead_id": 10,
                    "timestamp": 0.1,
                    "lead_speed": 11.0,
                    "lead_accel": 1.1,
                    "gap_distance": 19.0,
                    "relative_speed": 2.1,
                    "ego_accel": 0.0,
                },
                {
                    "scenario_id": "synthetic",
                    "ego_id": 1,
                    "lead_id": 20,
                    "timestamp": 0.2,
                    "lead_speed": 30.0,
                    "lead_accel": 3.0,
                    "gap_distance": 18.0,
                    "relative_speed": 3.0,
                    "ego_accel": 0.0,
                },
            ]
        )
        delayed = self.preprocess.simulate_degradation(frame, delay_frames=1)
        self.assertEqual(delayed.loc[1, "lead_speed"], 10.0)
        self.assertTrue(pd.isna(delayed.loc[2, "lead_speed"]))
        self.assertEqual(delayed.loc[2, "lead_observed"], 0.0)

    def test_scenario_bootstrap_returns_edges_with_stability(self):
        table = self.discovery.bootstrap_stability(
            self.discovery.load_condition("clean"),
            self.discovery.COOPERATIVE_COLS,
            "clean",
        )
        self.assertIn("stability", table.columns)
        if not table.empty:
            self.assertTrue(table["stability"].between(0, 1).all())

    def test_saved_primary_evaluation_has_three_models(self):
        summary = pd.read_csv(ROOT / "outputs/evaluation_summary.csv")
        primary = summary[summary["degradation"].eq("dropout_20_delay_1")]
        self.assertEqual(
            set(primary["model"]),
            {"causal_masked", "baseline", "robust_masked"},
        )

    def test_saved_outputs_cover_all_conditions_and_are_finite(self):
        expected_conditions = {
            "clean",
            "dropout_10",
            "dropout_25",
            "dropout_50",
            "delay_1",
            "delay_2",
            "dropout_20_delay_1",
        }
        stability = pd.read_csv(ROOT / "outputs/causal_edge_stability.csv")
        summary = pd.read_csv(ROOT / "outputs/evaluation_summary.csv")
        performance = pd.read_csv(ROOT / "outputs/performance_metrics.csv")

        self.assertEqual(set(stability["degradation"]), expected_conditions)
        self.assertEqual(set(summary["degradation"]), expected_conditions)
        self.assertEqual(set(performance["degradation"]), expected_conditions)
        self.assertEqual(set(performance["samples_per_context"]), {32})
        self.assertEqual(set(performance["evaluation_seed"]), {2024})
        metric_columns = [
            "distribution_mean_mae",
            "joint_energy_distance",
            "hard_decel_rate",
            "diversity_std",
            "linked_corr",
            "unlinked_corr",
            "consistency_gap",
        ]
        self.assertTrue(np.isfinite(summary[metric_columns].to_numpy()).all())
        self.assertTrue((ROOT / "outputs/causal_seed_sensitivity.csv").is_file())

    def test_primary_robustness_controls_are_complete(self):
        detail = pd.read_csv(ROOT / "outputs/robustness_primary.csv")
        summary = pd.read_csv(ROOT / "outputs/robustness_primary_summary.csv")
        self.assertEqual(set(detail["seed"]), {42, 142, 242})
        self.assertEqual(
            set(detail["model"]),
            {"causal_masked", "robust_masked", "random_masked", "baseline"},
        )
        self.assertEqual(len(detail), 12)
        self.assertEqual(
            set(summary["metric"]),
            {
                "distribution_mean_mae",
                "diversity_std",
                "linked_corr",
                "unlinked_corr",
                "consistency_gap",
            },
        )
        self.assertTrue(
            np.isfinite(
                detail[
                    [
                        "distribution_mean_mae",
                        "diversity_std",
                        "linked_corr",
                        "unlinked_corr",
                        "consistency_gap",
                    ]
                ].to_numpy()
            ).all()
        )

    def test_trajectory_generation_artifacts_are_complete(self):
        metrics = pd.read_csv(ROOT / "outputs/trajectory_generation_metrics.csv")
        generated = metrics[metrics["model"].ne("training_metadata")]
        self.assertEqual(len(generated), 15)
        self.assertEqual(
            set(generated["model"]),
            {"trajectory_masked", "trajectory_baseline", "trajectory_robust"},
        )
        physical_columns = [
            "negative_speed_rate",
            "accel_limit_violation_rate",
            "jerk_limit_rate",
            "mean_mae",
            "diversity_std",
        ]
        self.assertTrue(np.isfinite(generated[physical_columns].to_numpy()).all())
        for label in [
            "clean",
            "dropout_10",
            "dropout_25",
            "dropout_50",
            "delay_1",
            "delay_2",
            "dropout_20_delay_1",
        ]:
            self.assertGreater(
                len(pd.read_csv(ROOT / f"data/near_miss_windows_{label}.csv")), 0
            )


if __name__ == "__main__":
    unittest.main()
