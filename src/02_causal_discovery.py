"""
Phase 4 — Causal discovery.
Run: python 03_causal_discovery.py

Reads data/ego_features.csv and data/v2x_features.csv (from Phase 3),
runs the PC algorithm on each, compares the graphs, and saves:
  outputs/causal_graph_comparison.png
  outputs/causal_edges_v2x.csv        (edge list of the V2X-augmented graph)
This edge list is what Phase 5 uses to build the causal mask for the CVAE.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
import os

os.makedirs("outputs", exist_ok=True)

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

ego_df = pd.read_csv("data/ego_features.csv")
v2x_df = pd.read_csv("data/v2x_features.csv")

# ------------------------------------------------------------
# RUN PC ALGORITHM
# ------------------------------------------------------------

def run_pc(feature_df, exclude_cols=("Vehicle_ID",), alpha=0.05):
    cols = [c for c in feature_df.columns if c not in exclude_cols]
    data = feature_df[cols].to_numpy()
    cg = pc(data, alpha=alpha)
    return cg, cols

def graph_to_nx(cg, col_names):
    G = nx.DiGraph()
    G.add_nodes_from(col_names)
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            # directed edge i -> j
            if cg.G.graph[i, j] == 1 and cg.G.graph[j, i] == -1:
                G.add_edge(col_names[i], col_names[j])
            # undirected edge: keep as a plain (adjacency) edge too,
            # since PC often can't fully orient edges on small samples
            elif cg.G.graph[i, j] == -1 and cg.G.graph[j, i] == -1 and i < j:
                G.add_edge(col_names[i], col_names[j])
                G.add_edge(col_names[j], col_names[i])
    return G

print("Running PC on ego-only features ...")
cg_ego, cols_ego = run_pc(ego_df)
G_ego = graph_to_nx(cg_ego, cols_ego)

print("Running PC on ego+V2X-proxy features ...")
cg_v2x, cols_v2x = run_pc(v2x_df)
G_v2x = graph_to_nx(cg_v2x, cols_v2x)

# ------------------------------------------------------------
# COMPARE
# ------------------------------------------------------------

print(f"\nEgo-only graph:  {G_ego.number_of_edges()} edges")
print(f"Ego+V2X graph:   {G_v2x.number_of_edges()} edges")

new_edges = set(G_v2x.edges()) - set(G_ego.edges())
print(f"New edges recovered with V2X-proxy context: {len(new_edges)}")
for e in sorted(new_edges):
    print("  ", e)

# ------------------------------------------------------------
# PLOT
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

def plot_graph(G, title, ax):
    pos = nx.spring_layout(G, seed=42, k=1.2)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color="#cfe8ff",
             node_size=1600, font_size=7, arrows=True, arrowsize=12)
    ax.set_title(title, fontsize=11)

plot_graph(G_ego, "Ego-only causal graph", axes[0])
plot_graph(G_v2x, "Ego + V2X-proxy causal graph", axes[1])
plt.tight_layout()
plt.savefig("outputs/causal_graph_comparison.png", dpi=150)
print("\nSaved plot: outputs/causal_graph_comparison.png")

# ------------------------------------------------------------
# SAVE EDGE LIST (needed by Phase 5 for the causal mask)
# ------------------------------------------------------------

edge_df = pd.DataFrame(list(G_v2x.edges()), columns=["source", "target"])
edge_df.to_csv("outputs/causal_edges_v2x.csv", index=False)
print("Saved edge list: outputs/causal_edges_v2x.csv")

print("\nPhase 4 complete.")
