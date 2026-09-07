"""Preprocess official V2X-Seq-TFD cooperative trajectory data."""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_ROOT = Path("data/v2x_seq_tfd_example_extracted/V2X-Seq-TFD-Example")
SPLIT = "train"
OUT_DIR = Path("data")
MAX_SCENARIOS = None

TARGET_COLS = ["ego_accel", "ego_speed"]
CONTEXT_COLS = [
    "lead_speed",
    "lead_accel",
    "gap_distance",
    "relative_speed",
    "lead_observed",
]

DEGRADATION_LEVELS = [
    ("clean", 0.0, 0),
    ("dropout_10", 0.10, 0),
    ("dropout_25", 0.25, 0),
    ("dropout_50", 0.50, 0),
    ("delay_1", 0.0, 1),
    ("delay_2", 0.0, 2),
    ("dropout_20_delay_1", 0.20, 1),
]


def _trajectory_files(root: Path, split: str) -> list[Path]:
    directory = (
        root
        / "cooperative-vehicle-infrastructure"
        / "vehicle-trajectories"
        / split
        / "data"
    )
    files = sorted(directory.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No TFD trajectory files found at {directory}")
    return files


def load_tfd(
    root: Path = DATA_ROOT,
    split: str = SPLIT,
    max_scenarios: int | None = MAX_SCENARIOS,
) -> pd.DataFrame:
    """Load real cooperative vehicle tracks from V2X-Seq-TFD."""
    files = _trajectory_files(root, split)
    if max_scenarios is not None:
        files = files[:max_scenarios]

    frames = []
    required = {"timestamp", "id", "type", "x", "y", "length", "theta", "v_x", "v_y"}
    for path in files:
        frame = pd.read_csv(path)
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        frame["scenario_id"] = path.stem
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)
    data = data[data["type"].eq("VEHICLE")].copy()
    data = data.sort_values(["scenario_id", "id", "timestamp"])
    data["speed"] = np.hypot(data["v_x"], data["v_y"])
    grouped = data.groupby(["scenario_id", "id"], sort=False)
    dt = grouped["timestamp"].diff()
    data["accel"] = grouped["speed"].diff() / dt
    valid_motion = dt.between(0.05, 0.2) & data["accel"].abs().le(15.0)
    data.loc[~valid_motion, "accel"] = np.nan
    data["accel"] = data["accel"].fillna(0.0)
    return data.reset_index(drop=True)


def _nearest_forward_vehicle(frame: pd.DataFrame, ego: pd.Series):
    """Find the nearest observed vehicle ahead of the ego vehicle."""
    others = frame[frame["id"] != ego["id"]].copy()
    if others.empty:
        return None

    heading = np.array([np.cos(ego["theta"]), np.sin(ego["theta"])])
    delta = others[["x", "y"]].to_numpy() - np.array([ego["x"], ego["y"]])
    longitudinal = delta @ heading
    lateral = delta @ np.array([-heading[1], heading[0]])
    corridor = np.maximum(3.0, (ego["width"] + others["width"].to_numpy()) / 2 + 1.0)
    forward_and_near = (longitudinal > 0) & (np.abs(lateral) <= corridor)
    if not forward_and_near.any():
        return None

    candidates = others.loc[forward_and_near].copy()
    candidates["longitudinal_gap"] = longitudinal[forward_and_near]
    return candidates.sort_values("longitudinal_gap").iloc[0]


def build_interactions(data: pd.DataFrame) -> pd.DataFrame:
    """Build frame-level ego/leader interactions from cooperative tracks."""
    rows = []
    for (scenario_id, timestamp), frame in data.groupby(
        ["scenario_id", "timestamp"], sort=False
    ):
        for _, ego in frame.iterrows():
            lead = _nearest_forward_vehicle(frame, ego)
            if lead is None:
                continue

            heading = np.array([np.cos(ego["theta"]), np.sin(ego["theta"])])
            gap = max(float(lead["longitudinal_gap"] - ego["length"] / 2), 0.01)
            relative_speed = float(
                (ego[["v_x", "v_y"]].to_numpy() - lead[["v_x", "v_y"]].to_numpy())
                @ heading
            )
            ttc = gap / relative_speed if relative_speed > 0 else np.inf

            rows.append(
                {
                    "scenario_id": scenario_id,
                    "timestamp": timestamp,
                    "ego_id": ego["id"],
                    "lead_id": lead["id"],
                    "ego_speed": ego["speed"],
                    "ego_accel": ego["accel"],
                    "lead_speed": lead["speed"],
                    "lead_accel": lead["accel"],
                    "gap_distance": gap,
                    "relative_speed": relative_speed,
                    "ttc": ttc,
                    "lead_observed": 1.0,
                }
            )

    interactions = pd.DataFrame(rows)
    if interactions.empty:
        raise ValueError("No vehicle interactions could be constructed from TFD data")
    return interactions.sort_values(["scenario_id", "ego_id", "timestamp"]).reset_index(
        drop=True
    )


def extract_near_miss_events(
    interactions: pd.DataFrame,
    ttc_threshold: float = 3.0,
    hard_decel: float = -3.0,
) -> pd.DataFrame:
    """Label near misses using clean TTC and ego deceleration."""
    labeled = interactions.copy()
    labeled["near_miss"] = labeled["ttc"].lt(ttc_threshold) | labeled["ego_accel"].lt(
        hard_decel
    )
    return labeled[labeled["near_miss"]].copy()


def simulate_degradation(
    interactions: pd.DataFrame,
    dropout_rate: float = 0.0,
    delay_frames: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    """Degrade shared context on the full timeline, preserving clean targets."""
    degraded = interactions.copy()
    context_to_degrade = ["lead_speed", "lead_accel", "gap_distance", "relative_speed"]

    if delay_frames:
        degraded = degraded.sort_values(["scenario_id", "ego_id", "timestamp"])
        for column in context_to_degrade:
            degraded[column] = degraded.groupby(["scenario_id", "ego_id"], sort=False)[
                column
            ].shift(delay_frames)

    if dropout_rate:
        rng = np.random.default_rng(seed)
        for column in context_to_degrade:
            dropped = rng.random(len(degraded)) < dropout_rate
            degraded.loc[dropped, column] = np.nan
        degraded["lead_observed"] = (
            degraded[context_to_degrade].notna().all(axis=1).astype(float)
        )

    return degraded.reset_index(drop=True)


def build_ego_features(interactions: pd.DataFrame) -> pd.DataFrame:
    return (
        interactions.groupby(["scenario_id", "ego_id"], as_index=False)
        .agg(
            mean_speed=("ego_speed", "mean"),
            mean_accel=("ego_accel", "mean"),
            accel_std=("ego_accel", "std"),
            min_ttc=("ttc", "min"),
        )
        .fillna(0.0)
    )


def build_cooperative_features(interactions: pd.DataFrame) -> pd.DataFrame:
    return (
        interactions.groupby(["scenario_id", "ego_id"], as_index=False)
        .agg(
            mean_speed=("ego_speed", "mean"),
            mean_accel=("ego_accel", "mean"),
            accel_std=("ego_accel", "std"),
            mean_lead_speed=("lead_speed", "mean"),
            mean_lead_accel=("lead_accel", "mean"),
            mean_gap=("gap_distance", "mean"),
            mean_relative_speed=("relative_speed", "mean"),
            min_ttc=("ttc", "min"),
        )
        .fillna(0.0)
    )


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Loading V2X-Seq-TFD trajectories from {DATA_ROOT} ...")
    raw = load_tfd()
    print(
        f"  rows: {len(raw):,}; scenarios: {raw['scenario_id'].nunique():,}; "
        f"vehicles: {raw['id'].nunique():,}"
    )

    interactions = build_interactions(raw)
    events = extract_near_miss_events(interactions)
    print(f"  interaction rows: {len(interactions):,}")
    print(
        f"  clean near-miss events: {len(events):,} "
        f"({100 * len(events) / len(interactions):.2f}%)"
    )

    build_ego_features(interactions).to_csv(OUT_DIR / "ego_features.csv", index=False)
    build_cooperative_features(interactions).to_csv(
        OUT_DIR / "v2x_features.csv", index=False
    )
    events.to_csv(OUT_DIR / "near_miss_events.csv", index=False)

    for label, dropout_rate, delay_frames in DEGRADATION_LEVELS:
        degraded_interactions = simulate_degradation(
            interactions, dropout_rate, delay_frames
        )
        degraded_events = degraded_interactions.loc[events.index].copy()
        degraded_events["near_miss"] = events["near_miss"].to_numpy()
        degraded_events.to_csv(OUT_DIR / f"near_miss_events_{label}.csv", index=False)
        print(f"  saved {label}: {len(degraded_events):,} events")

    print("Preprocessing complete.")
