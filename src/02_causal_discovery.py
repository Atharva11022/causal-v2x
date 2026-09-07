"""Bootstrap causal-edge stability under degraded V2X context."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc


DATA_DIR = Path("data")
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEGRADATION_LABELS = [
    "clean",
    "dropout_10",
    "dropout_25",
    "dropout_50",
    "delay_1",
    "delay_2",
    "dropout_20_delay_1",
]
EGO_COLS = ["ego_speed", "ego_accel"]
COOPERATIVE_COLS = EGO_COLS + [
    "lead_speed",
    "lead_accel",
    "gap_distance",
    "relative_speed",
]
N_BOOTSTRAPS = 20
BOOTSTRAP_FRACTION = 0.8
STABILITY_THRESHOLD = 0.70
SENSITIVITY_THRESHOLDS = [0.60, 0.70, 0.80]
ALPHA_SENSITIVITY = [0.01, 0.05, 0.10, 0.20]
SEED_BASES = [42, 142, 242]


def load_condition(label: str) -> pd.DataFrame:
    path = DATA_DIR / f"near_miss_events_{label}.csv"
    if label == "clean" and not path.exists():
        path = DATA_DIR / "near_miss_events.csv"
    frame = pd.read_csv(path)
    required = {"scenario_id", *COOPERATIVE_COLS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame[["scenario_id", *COOPERATIVE_COLS]].replace([np.inf, -np.inf], np.nan)


def prepare_matrix(frame: pd.DataFrame, columns: list[str], seed: int) -> np.ndarray:
    """Resample complete scenarios so temporal frames are not IID rows."""
    values = frame[columns].astype(float)
    values = values.fillna(values.median()).fillna(0.0)
    rng = np.random.default_rng(seed)
    scenario_ids = frame["scenario_id"].drop_duplicates().to_numpy()
    sample_size = max(2, int(len(scenario_ids) * BOOTSTRAP_FRACTION))
    selected = rng.choice(scenario_ids, size=sample_size, replace=True)
    matrix = pd.concat(
        [values.loc[frame["scenario_id"].eq(scenario_id)] for scenario_id in selected],
        ignore_index=True,
    )
    return matrix.to_numpy(dtype=float)


def edge_set(
    data: np.ndarray, columns: list[str], alpha: float = 0.05
) -> set[tuple[str, str]]:
    """Return canonical undirected adjacencies from the PC CPDAG."""
    if len(data) < 20 or np.any(np.nanstd(data, axis=0) < 1e-10):
        return set()

    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        graph = pc(data, alpha=alpha).G.graph

    edges = set()
    for left in range(len(columns)):
        for right in range(left + 1, len(columns)):
            if graph[left, right] != 0 or graph[right, left] != 0:
                edges.add(tuple(sorted((columns[left], columns[right]))))
    return edges


def bootstrap_stability(
    frame: pd.DataFrame, columns: list[str], label: str, seed_base: int = 42
) -> pd.DataFrame:
    counts: dict[tuple[str, str], int] = {}
    for bootstrap_id in range(N_BOOTSTRAPS):
        matrix = prepare_matrix(frame, columns, seed=seed_base + bootstrap_id)
        for edge in edge_set(matrix, columns):
            counts[edge] = counts.get(edge, 0) + 1

    return pd.DataFrame(
        [
            {
                "degradation": label,
                "source": source,
                "target": target,
                "support_count": counts[(source, target)],
                "n_bootstraps": N_BOOTSTRAPS,
                "stability": counts[(source, target)] / N_BOOTSTRAPS,
                "stable": counts[(source, target)] / N_BOOTSTRAPS
                >= STABILITY_THRESHOLD,
            }
            for source, target in sorted(counts)
        ]
    )


def plot_graph(
    edges: pd.DataFrame, title: str, path: Path, node_columns: list[str]
) -> None:
    graph = nx.Graph()
    graph.add_nodes_from(node_columns)
    graph.add_edges_from(edges[["source", "target"]].itertuples(index=False, name=None))
    figure, axis = plt.subplots(figsize=(10, 6))
    position = nx.spring_layout(graph, seed=42)
    nx.draw_networkx(
        graph,
        position,
        ax=axis,
        node_color="#cfe8ff",
        node_size=1500,
        font_size=8,
        edge_color="#356b8c",
    )
    axis.set_title(title)
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    stability_tables = []
    for label in DEGRADATION_LABELS:
        condition = load_condition(label)
        table = bootstrap_stability(condition, COOPERATIVE_COLS, label)
        stability_tables.append(table)
        stable_count = int(table["stable"].sum()) if not table.empty else 0
        print(
            f"{label}: rows={len(condition):,}, "
            f"candidate_edges={len(table)}, stable_edges={stable_count}"
        )

    stability = (
        pd.concat(stability_tables, ignore_index=True)
        if stability_tables
        else pd.DataFrame()
    )
    stability.to_csv(OUT_DIR / "causal_edge_stability.csv", index=False)

    clean = stability[stability["degradation"].eq("clean") & stability["stable"]].copy()
    clean.to_csv(OUT_DIR / "causal_edges_v2x.csv", index=False)
    plot_graph(
        clean,
        "Stable cooperative causal graph",
        OUT_DIR / "causal_graph_comparison.png",
        COOPERATIVE_COLS,
    )

    ego_condition = load_condition("clean")
    ego_stability = bootstrap_stability(ego_condition, EGO_COLS, "clean")
    ego_stability.to_csv(OUT_DIR / "causal_edge_stability_ego_only.csv", index=False)
    ego_clean = ego_stability[ego_stability["stable"]].copy()
    ego_clean.to_csv(OUT_DIR / "causal_edges_ego_only.csv", index=False)
    plot_graph(
        ego_clean,
        "Stable ego-only causal graph",
        OUT_DIR / "causal_graph_ego_only.png",
        EGO_COLS,
    )
    pd.DataFrame(
        [
            {
                "view": "ego_only",
                "stable_edges": len(ego_clean),
            },
            {
                "view": "cooperative",
                "stable_edges": len(clean),
            },
            {
                "view": "additional_cooperative_edges",
                "stable_edges": len(
                    set(map(tuple, clean[["source", "target"]].to_numpy()))
                    - set(map(tuple, ego_clean[["source", "target"]].to_numpy()))
                ),
            },
        ]
    ).to_csv(OUT_DIR / "causal_context_comparison.csv", index=False)

    alpha_rows = []
    for alpha in ALPHA_SENSITIVITY:
        counts = {}
        for bootstrap_id in range(N_BOOTSTRAPS):
            matrix = prepare_matrix(ego_condition, EGO_COLS, seed=42 + bootstrap_id)
            for edge in edge_set(matrix, EGO_COLS, alpha=alpha):
                counts[edge] = counts.get(edge, 0) + 1
        for source, target in sorted(counts):
            alpha_rows.append(
                {
                    "alpha": alpha,
                    "source": source,
                    "target": target,
                    "support_count": counts[(source, target)],
                    "n_bootstraps": N_BOOTSTRAPS,
                    "stability": counts[(source, target)] / N_BOOTSTRAPS,
                    "stable_at_0_70": counts[(source, target)] / N_BOOTSTRAPS
                    >= STABILITY_THRESHOLD,
                }
            )
    pd.DataFrame(alpha_rows).to_csv(
        OUT_DIR / "causal_ego_alpha_sensitivity.csv", index=False
    )

    seed_rows = []
    for label in ["clean", "dropout_20_delay_1"]:
        condition = load_condition(label)
        for seed_base in SEED_BASES:
            table = bootstrap_stability(
                condition, COOPERATIVE_COLS, label, seed_base=seed_base
            )
            seed_rows.append(
                {
                    "degradation": label,
                    "seed_base": seed_base,
                    "candidate_edges": len(table),
                    "stable_edges": int(table["stable"].sum())
                    if not table.empty
                    else 0,
                }
            )
    pd.DataFrame(seed_rows).to_csv(OUT_DIR / "causal_seed_sensitivity.csv", index=False)

    summary = stability.groupby("degradation", as_index=False).agg(
        candidate_edges=("source", "size"), stable_edges=("stable", "sum")
    )
    summary.to_csv(OUT_DIR / "causal_stability_summary.csv", index=False)

    threshold_rows = []
    for threshold in SENSITIVITY_THRESHOLDS:
        for label, condition in stability.groupby("degradation"):
            threshold_rows.append(
                {
                    "degradation": label,
                    "stability_threshold": threshold,
                    "candidate_edges": len(condition),
                    "stable_edges": int((condition["stability"] >= threshold).sum()),
                }
            )
    pd.DataFrame(threshold_rows).to_csv(
        OUT_DIR / "causal_threshold_sensitivity.csv", index=False
    )
    robust_edges = (
        stability[stability["stable"]]
        .groupby(["source", "target"], as_index=False)
        .agg(condition_count=("degradation", "nunique"))
    )
    robust_edges = robust_edges[
        robust_edges["condition_count"] == len(DEGRADATION_LABELS)
    ]
    robust_edges.to_csv(OUT_DIR / "causal_edges_robust.csv", index=False)
    print(
        f"Saved {len(stability):,} edge stability records and "
        f"{len(clean):,} clean stable edges; "
        f"{len(robust_edges):,} robust edges across all conditions."
    )
