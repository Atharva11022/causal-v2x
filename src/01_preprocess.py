"""
Phase 3 — Preprocessing, feature engineering, near-miss mining.
Run: python 02_preprocess.py

Reads the raw NGSIM CSV, produces:
  data/ego_features.csv          (per-vehicle, ego-only features)
  data/v2x_features.csv          (per-vehicle, ego + neighbor/V2X-proxy features)
  data/near_miss_events.csv      (per-event snapshot features for the CVAE, Phase 5)
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/trajectories-0750am-0805am.txt"   # <-- your actual downloaded file
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

MAX_ROWS = 300_000   # keep this modest so everything stays fast on 8GB RAM

# Official NGSIM column order (the raw .txt has NO header row)
NGSIM_COLUMNS = [
    "Vehicle_ID", "Frame_ID", "Total_Frames", "Global_Time",
    "Local_X", "Local_Y", "Global_X", "Global_Y",
    "v_Length", "v_Width", "v_Class", "v_Vel", "v_Acc",
    "Lane_ID", "Preceding", "Following", "Space_Headway", "Time_Headway",
]

# ------------------------------------------------------------
# 1. LOAD
# ------------------------------------------------------------

def load_ngsim(path, max_rows=MAX_ROWS):
    print(f"Loading {path} ...")
    df = pd.read_csv(
        path,
        sep=r"\s+",          # whitespace-delimited, not comma
        header=None,
        names=NGSIM_COLUMNS,
        engine="python",
    )
    print(f"  raw rows: {len(df):,}")
    print(f"  unique vehicles: {df['Vehicle_ID'].nunique():,}")
    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).head(max_rows).copy()
    print(f"  using rows: {len(df):,}  (subset for speed/memory)")
    return df

# ------------------------------------------------------------
# 2. EGO-ONLY FEATURES  (what a single vehicle's own sensors would know)
# ------------------------------------------------------------

def build_ego_features(df):
    feats = df.groupby("Vehicle_ID").apply(lambda g: pd.Series({
        "mean_speed": g["v_Vel"].mean(),
        "mean_accel": g["v_Acc"].mean(),
        "accel_std": g["v_Acc"].std(),
        "lane_changes": g["Lane_ID"].diff().fillna(0).abs().gt(0).sum(),
        "min_time_headway": g["Time_Headway"].replace(0, np.nan).min(),
    })).reset_index()
    return feats.fillna(0)

# ------------------------------------------------------------
# 3. V2X-PROXY FEATURES  (adds preceding-vehicle context — only
#    knowable in reality via cooperative/V2X-shared observation)
# ------------------------------------------------------------

def build_v2x_features(df):
    ego = build_ego_features(df)

    lead = df[["Vehicle_ID", "Frame_ID", "v_Vel", "v_Acc"]].rename(
        columns={"Vehicle_ID": "Preceding", "v_Vel": "lead_vel", "v_Acc": "lead_acc"}
    )
    merged = df.merge(lead, on=["Preceding", "Frame_ID"], how="left")

    neighbor_feats = merged.groupby("Vehicle_ID").apply(lambda g: pd.Series({
        "mean_lead_vel": g["lead_vel"].mean(),
        "mean_lead_acc": g["lead_acc"].mean(),
        "rel_speed_to_lead": (g["v_Vel"] - g["lead_vel"]).mean(),
        "space_headway_mean": g["Space_Headway"].mean(),
    })).reset_index()

    return ego.merge(neighbor_feats, on="Vehicle_ID", how="left").fillna(0)

# ------------------------------------------------------------
# 4. NEAR-MISS EVENT EXTRACTION (frame-level snapshots, for the CVAE)
# ------------------------------------------------------------

def extract_near_miss_events(df, ttc_threshold=3.0, hard_decel_ft=-10.0):
    """
    Flags frames where the ego vehicle is in a genuine near-miss state,
    using TRUE time-to-collision (based on closing speed, not simple
    time-headway) and a properly-scaled hard-braking threshold.

    NOTE: NGSIM units are feet / feet-per-second / feet-per-second^2,
    NOT meters. hard_decel_ft=-10.0 ft/s^2 ~= -3.05 m/s^2, a genuinely
    hard braking event (roughly 0.3g), not routine deceleration.
    """
    lead = df[["Vehicle_ID", "Frame_ID", "v_Vel", "v_Acc"]].rename(
        columns={"Vehicle_ID": "Preceding", "v_Vel": "lead_vel", "v_Acc": "lead_acc"}
    )
    merged = df.merge(lead, on=["Preceding", "Frame_ID"], how="left")

    merged["rel_speed_to_lead"] = merged["v_Vel"] - merged["lead_vel"]

    # True TTC: only defined (and meaningful) when the ego vehicle is
    # actually CLOSING the gap (positive relative speed). Otherwise the
    # gap is stable or growing -> not a collision course -> TTC = infinity.
    closing = merged["rel_speed_to_lead"] > 1.0  # ft/s, ignore noise near zero
    merged["ttc"] = np.where(
        closing,
        merged["Space_Headway"] / merged["rel_speed_to_lead"].clip(lower=1e-3),
        np.inf,
    )

    merged["near_miss"] = (
        (merged["ttc"] < ttc_threshold) |
        (merged["v_Acc"] < hard_decel_ft)
    )

    events = merged[merged["near_miss"]].copy()

    events = events[[
        "Vehicle_ID", "Frame_ID",
        "v_Acc", "v_Vel",                      # target/reaction features
        "lead_vel", "lead_acc",                # context/V2X-proxy features
        "Space_Headway", "Time_Headway", "rel_speed_to_lead", "ttc",
    ]].dropna(subset=["v_Acc", "v_Vel", "lead_vel", "lead_acc", "Space_Headway", "Time_Headway"])

    print(f"  near-miss events extracted: {len(events):,}  "
          f"({100*len(events)/len(merged):.1f}% of frames)")
    return events

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------

if __name__ == "__main__":
    raw = load_ngsim(RAW_PATH)

    print("\nBuilding ego-only features ...")
    ego_df = build_ego_features(raw)
    ego_df.to_csv(f"{OUT_DIR}/ego_features.csv", index=False)
    print(f"  saved: {OUT_DIR}/ego_features.csv  ({len(ego_df)} vehicles)")

    print("\nBuilding V2X-proxy features ...")
    v2x_df = build_v2x_features(raw)
    v2x_df.to_csv(f"{OUT_DIR}/v2x_features.csv", index=False)
    print(f"  saved: {OUT_DIR}/v2x_features.csv  ({len(v2x_df)} vehicles)")

    print("\nExtracting near-miss events ...")
    events_df = extract_near_miss_events(raw)
    events_df.to_csv(f"{OUT_DIR}/near_miss_events.csv", index=False)
    print(f"  saved: {OUT_DIR}/near_miss_events.csv")

    print("\nPhase 3 complete.")
