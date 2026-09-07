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


if __name__ == "__main__":
    unittest.main()
