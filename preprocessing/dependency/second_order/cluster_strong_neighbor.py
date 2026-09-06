import os
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ============================================================
# STAGE 4E — CLUSTER STRONG-NEIGHBOR SECOND-ORDER GRAPH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GRAPH_FILE = os.path.join(
    BASE_DIR,
    "second_order_strong_0.60.csv"
)

FIRST_ORDER_FILE = os.path.join(
    os.path.dirname(BASE_DIR),
    "alpha_optimization",
    "final_dependency_matrix.csv"
)

OUTPUT_RESULTS = os.path.join(
    BASE_DIR,
    "strong_neighbor_0.60_clustering_results.csv"
)

OUTPUT_BLOCKS = os.path.join(
    BASE_DIR,
    "strong_neighbor_0.60_candidate_blocks.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "strong_neighbor_0.60_summary.txt"
)


print("=" * 70)
print("STAGE 4E — CLUSTER STRONG-NEIGHBOR SECOND-ORDER GRAPH")
print("=" * 70)


# ------------------------------------------------------------
# Load
# ------------------------------------------------------------

print("\nLoading second-order graph...")

S_df = pd.read_csv(
    GRAPH_FILE,
    index_col=0
)

D_df = pd.read_csv(
    FIRST_ORDER_FILE,
    index_col=0
)

features = list(S_df.index)

S = S_df.to_numpy(dtype=float)

D = D_df.loc[
    features,
    features
].to_numpy(dtype=float)

n = len(features)

print(f"Features: {n}")
print(f"Graph shape: {S.shape}")


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

if S.shape != (39, 39):
    raise ValueError(
        f"Expected 39x39 graph, got {S.shape}"
    )

if not np.allclose(S, S.T, atol=1e-8):
    raise ValueError(
        "Second-order graph is not symmetric."
    )

if not np.allclose(D, D.T, atol=1e-8):
    raise ValueError(
        "First-order matrix is not symmetric."
    )

print("✓ Matrices validated")


# ------------------------------------------------------------
# Distance
# ------------------------------------------------------------

distance = 1.0 - S

distance = np.clip(
    distance,
    0.0,
    1.0
)

np.fill_diagonal(
    distance,
    0.0
)

condensed = squareform(
    distance,
    checks=True
)


# ------------------------------------------------------------
# Hierarchical clustering
# ------------------------------------------------------------

Z = linkage(
    condensed,
    method="average"
)


# ------------------------------------------------------------
# Evaluate k
# ------------------------------------------------------------

print("\nEvaluating k = 6 ... 20")
print("-" * 70)

results = []

for k in range(6, 21):

    labels = fcluster(
        Z,
        t=k,
        criterion="maxclust"
    )

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    min_size = int(
        counts.min()
    )

    max_size = int(
        counts.max()
    )

    mean_size = float(
        counts.mean()
    )

    if max_size <= 7:

        silhouette = silhouette_score(
            distance,
            labels,
            metric="precomputed"
        )

        valid = True

    else:

        silhouette = np.nan
        valid = False

    results.append(
        {
            "k": k,
            "silhouette": silhouette,
            "min_block_size": min_size,
            "max_block_size": max_size,
            "mean_block_size": mean_size,
            "valid": valid
        }
    )

    if valid:

        print(
            f"k={k:2d} | "
            f"silhouette={silhouette:.4f} | "
            f"min={min_size:2d} | "
            f"max={max_size:2d} | "
            f"mean={mean_size:.2f}"
        )

    else:

        print(
            f"k={k:2d} | "
            f"silhouette=INVALID | "
            f"min={min_size:2d} | "
            f"max={max_size:2d} | "
            f"mean={mean_size:.2f}"
        )


results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


# ------------------------------------------------------------
# Select best valid k
# ------------------------------------------------------------

valid = results_df[
    results_df["valid"] == True
].copy()

if valid.empty:

    raise RuntimeError(
        "No valid k satisfies maximum block size <= 7."
    )

best = valid.loc[
    valid["silhouette"].idxmax()
]

best_k = int(
    best["k"]
)

best_silhouette = float(
    best["silhouette"]
)

print("\n" + "=" * 70)
print("BEST STRONG-NEIGHBOR CONFIGURATION")
print("=" * 70)

print(
    f"k = {best_k}"
)

print(
    f"Silhouette = {best_silhouette:.4f}"
)


# ------------------------------------------------------------
# Final clustering
# ------------------------------------------------------------

labels = fcluster(
    Z,
    t=best_k,
    criterion="maxclust"
)


# ------------------------------------------------------------
# Build blocks
# ------------------------------------------------------------

blocks = []

for cluster_id in sorted(
    np.unique(labels)
):

    indices = np.where(
        labels == cluster_id
    )[0]

    block_features = [
        features[i]
        for i in indices
    ]

    size = len(indices)

    # ----------------------------------------
    # Strong-neighbor internal similarity
    # ----------------------------------------

    if size >= 2:

        internal_S = S[
            np.ix_(
                indices,
                indices
            )
        ]

        upper_S = internal_S[
            np.triu_indices(
                size,
                k=1
            )
        ]

        mean_second_order = float(
            np.mean(upper_S)
        )

        min_second_order = float(
            np.min(upper_S)
        )

    else:

        mean_second_order = np.nan
        min_second_order = np.nan


    # ----------------------------------------
    # Original first-order dependency
    # ----------------------------------------

    if size >= 2:

        internal_D = D[
            np.ix_(
                indices,
                indices
            )
        ]

        upper_D = internal_D[
            np.triu_indices(
                size,
                k=1
            )
        ]

        mean_first_order = float(
            np.mean(upper_D)
        )

        min_first_order = float(
            np.min(upper_D)
        )

    else:

        mean_first_order = np.nan
        min_first_order = np.nan


    blocks.append(
        {
            "block_id": int(cluster_id),
            "block_size": size,
            "features": " | ".join(
                block_features
            ),
            "mean_second_order": mean_second_order,
            "min_second_order": min_second_order,
            "mean_first_order": mean_first_order,
            "min_first_order": min_first_order
        }
    )


blocks_df = pd.DataFrame(
    blocks
)

blocks_df = blocks_df.sort_values(
    "block_size",
    ascending=False
).reset_index(
    drop=True
)

blocks_df["block_id"] = np.arange(
    1,
    len(blocks_df) + 1
)


blocks_df.to_csv(
    OUTPUT_BLOCKS,
    index=False
)


# ------------------------------------------------------------
# Aggregate first-order block quality
# ------------------------------------------------------------

within_D = []
between_D = []

for i in range(n):

    for j in range(i + 1, n):

        if labels[i] == labels[j]:

            within_D.append(
                D[i, j]
            )

        else:

            between_D.append(
                D[i, j]
            )


mean_within_D = float(
    np.mean(within_D)
)

mean_between_D = float(
    np.mean(between_D)
)

ratio_D = (
    mean_within_D /
    mean_between_D
)

max_cross_D = float(
    np.max(between_D)
)


# ------------------------------------------------------------
# Print blocks
# ------------------------------------------------------------

print("\nCandidate blocks:")
print("-" * 70)

for _, row in blocks_df.iterrows():

    print(
        f"Block {int(row['block_id']):2d} "
        f"({int(row['block_size'])}): "
        f"{row['features']}"
    )

    if row["block_size"] >= 2:

        print(
            f"    S² mean = "
            f"{row['mean_second_order']:.4f}"
        )

        print(
            f"    D mean  = "
            f"{row['mean_first_order']:.4f}"
        )


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

singleton_count = int(
    np.sum(
        blocks_df["block_size"] == 1
    )
)

largest_block = int(
    blocks_df["block_size"].max()
)

print("\n" + "=" * 70)
print("STAGE 4E COMPLETE")
print("=" * 70)

print(
    f"\nSelected k: {best_k}"
)

print(
    f"Silhouette: {best_silhouette:.4f}"
)

print(
    f"Blocks: {len(blocks_df)}"
)

print(
    f"Singletons: {singleton_count}"
)

print(
    f"Largest block: {largest_block}"
)

print(
    f"\nFirst-order mean within: "
    f"{mean_within_D:.4f}"
)

print(
    f"First-order mean between: "
    f"{mean_between_D:.4f}"
)

print(
    f"First-order within/between ratio: "
    f"{ratio_D:.4f}"
)

print(
    f"First-order max cross-block: "
    f"{max_cross_D:.4f}"
)


# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 4E — STRONG-NEIGHBOR SECOND-ORDER CLUSTERING\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Input graph: second_order_strong_0.60.csv\n"
    )

    f.write(
        "Clustering: average-linkage hierarchical clustering\n"
    )

    f.write(
        "Constraint: maximum block size <= 7\n\n"
    )

    f.write(
        f"Selected k: {best_k}\n"
    )

    f.write(
        f"Silhouette: {best_silhouette:.6f}\n"
    )

    f.write(
        f"Number of blocks: {len(blocks_df)}\n"
    )

    f.write(
        f"Singleton blocks: {singleton_count}\n"
    )

    f.write(
        f"Largest block: {largest_block}\n\n"
    )

    f.write(
        "FIRST-ORDER QUALITY OF SECOND-ORDER BLOCKS\n"
    )

    f.write(
        f"Mean within D: {mean_within_D:.6f}\n"
    )

    f.write(
        f"Mean between D: {mean_between_D:.6f}\n"
    )

    f.write(
        f"Within/between ratio: {ratio_D:.6f}\n"
    )

    f.write(
        f"Max cross-block D: {max_cross_D:.6f}\n\n"
    )

    f.write(
        "BLOCKS\n"
    )

    f.write("-" * 70 + "\n")

    for _, row in blocks_df.iterrows():

        f.write(
            f"Block {int(row['block_id'])}: "
            f"{row['features']}\n"
        )

        if row["block_size"] >= 2:

            f.write(
                f"  S² mean: "
                f"{row['mean_second_order']:.6f}\n"
            )

            f.write(
                f"  D mean: "
                f"{row['mean_first_order']:.6f}\n"
            )

print("\nSaved:")
print(OUTPUT_RESULTS)
print(OUTPUT_BLOCKS)
print(OUTPUT_SUMMARY)