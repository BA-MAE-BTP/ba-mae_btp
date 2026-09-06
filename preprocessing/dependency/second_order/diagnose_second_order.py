import os
import numpy as np
import pandas as pd

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


# ============================================================
# STAGE 4B.1 — DIAGNOSE SECOND-ORDER GRAPH DENSITY
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "second_order_dependency_matrix.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "second_order_threshold_diagnostics.csv"
)

print("=" * 70)
print("STAGE 4B.1 — SECOND-ORDER GRAPH DIAGNOSTICS")
print("=" * 70)

# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

S_df = pd.read_csv(
    INPUT_FILE,
    index_col=0
)

features = list(S_df.index)
S = S_df.to_numpy(dtype=float)

n = len(features)

print(f"\nFeatures: {n}")
print(f"Matrix: {S.shape}")

# ------------------------------------------------------------
# Upper triangle
# ------------------------------------------------------------

upper = S[
    np.triu_indices(n, k=1)
]

print("\nGlobal second-order similarity:")
print(f"Min    : {upper.min():.4f}")
print(f"Median : {np.median(upper):.4f}")
print(f"Mean   : {upper.mean():.4f}")
print(f"Q75    : {np.percentile(upper, 75):.4f}")
print(f"Q90    : {np.percentile(upper, 90):.4f}")
print(f"Q95    : {np.percentile(upper, 95):.4f}")
print(f"Q99    : {np.percentile(upper, 99):.4f}")
print(f"Max    : {upper.max():.4f}")

# ------------------------------------------------------------
# Threshold diagnostics
# ------------------------------------------------------------

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.92,
    0.93
]

results = []

print("\nThreshold analysis:")
print("-" * 70)

for threshold in thresholds:

    # Keep only edges >= threshold
    adjacency = (
        S >= threshold
    ).astype(int)

    np.fill_diagonal(
        adjacency,
        0
    )

    graph = csr_matrix(
        adjacency
    )

    n_components, labels = connected_components(
        graph,
        directed=False
    )

    component_sizes = np.bincount(
        labels
    )

    component_sizes = sorted(
        component_sizes,
        reverse=True
    )

    largest = component_sizes[0]

    components_le_7 = sum(
        size <= 7
        for size in component_sizes
    )

    edges = int(
        np.sum(adjacency) // 2
    )

    possible_edges = (
        n * (n - 1) // 2
    )

    density = (
        edges / possible_edges
    )

    results.append(
        {
            "threshold": threshold,
            "edges": edges,
            "graph_density": density,
            "components": n_components,
            "largest_component": largest,
            "components_size_le_7": components_le_7
        }
    )

    print(
        f"threshold={threshold:.2f} | "
        f"edges={edges:3d} | "
        f"density={density:.3f} | "
        f"components={n_components:2d} | "
        f"largest={largest:2d} | "
        f"components<=7={components_le_7:2d}"
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("STAGE 4B.1 COMPLETE")
print("=" * 70)