import os
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ============================================================
# STAGE 4B — CLUSTER SECOND-ORDER DEPENDENCY GRAPH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "second_order_dependency_matrix.csv"
)

OUTPUT_BLOCKS = os.path.join(
    BASE_DIR,
    "second_order_candidate_blocks.csv"
)

OUTPUT_RESULTS = os.path.join(
    BASE_DIR,
    "second_order_clustering_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "second_order_clustering_summary.txt"
)

print("=" * 70)
print("STAGE 4B — CLUSTER SECOND-ORDER DEPENDENCY GRAPH")
print("=" * 70)

# ------------------------------------------------------------
# Load matrix
# ------------------------------------------------------------

print("\nLoading second-order dependency matrix...")

S_df = pd.read_csv(INPUT_FILE, index_col=0)

features = list(S_df.index)
S = S_df.to_numpy(dtype=float)

n = len(features)

print(f"Number of features: {n}")
print(f"Matrix shape: {S.shape}")

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

if S.shape != (39, 39):
    raise ValueError(f"Expected 39x39 matrix, got {S.shape}")

if list(S_df.columns) != features:
    raise ValueError(
        "Row and column feature ordering does not match."
    )

if not np.allclose(S, S.T, atol=1e-8):
    raise ValueError("Second-order matrix is not symmetric.")

if not np.allclose(np.diag(S), 1.0, atol=1e-8):
    raise ValueError("Diagonal is not 1.")

if np.isnan(S).any() or np.isinf(S).any():
    raise ValueError("Matrix contains NaN or Inf.")

print("✓ Matrix validated")

# ------------------------------------------------------------
# Convert similarity to distance
# ------------------------------------------------------------

print("\nConverting similarity to distance:")

print("distance = 1 - second_order_similarity")

distance = 1.0 - S

# Numerical cleanup
distance = np.clip(distance, 0.0, 1.0)

np.fill_diagonal(distance, 0.0)

# squareform expects a symmetric zero-diagonal matrix
condensed_distance = squareform(
    distance,
    checks=True
)

# ------------------------------------------------------------
# Evaluate k = 6 ... 20
# ------------------------------------------------------------

print("\nEvaluating cluster configurations...")
print("-" * 70)

results = []

for k in range(6, 21):

    # Average linkage on second-order distance
    Z = linkage(
        condensed_distance,
        method="average"
    )

    labels = fcluster(
        Z,
        t=k,
        criterion="maxclust"
    )

    unique_labels, counts = np.unique(
        labels,
        return_counts=True
    )

    min_size = int(counts.min())
    max_size = int(counts.max())
    mean_size = float(counts.mean())

    # --------------------------------------------------------
    # Constraint from BA-MAE proposal:
    # maximum block size <= 7
    # --------------------------------------------------------

    if max_size > 7:
        silhouette = np.nan
        valid = False
    else:

        # silhouette_score accepts precomputed distances
        silhouette = silhouette_score(
            distance,
            labels,
            metric="precomputed"
        )

        valid = True

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

    sil_text = (
        f"{silhouette:.4f}"
        if not np.isnan(silhouette)
        else "INVALID"
    )

    print(
        f"k={k:2d} | "
        f"silhouette={sil_text:>8} | "
        f"min={min_size:2d} | "
        f"max={max_size:2d} | "
        f"mean={mean_size:.2f}"
    )

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

# ------------------------------------------------------------
# Select best valid configuration
# ------------------------------------------------------------

valid_results = results_df[
    results_df["valid"] == True
].copy()

if valid_results.empty:
    raise RuntimeError(
        "No valid clustering configuration satisfies "
        "maximum block size <= 7."
    )

best_row = valid_results.loc[
    valid_results["silhouette"].idxmax()
]

best_k = int(best_row["k"])
best_silhouette = float(best_row["silhouette"])

print("\n" + "=" * 70)
print("BEST SECOND-ORDER CONFIGURATION")
print("=" * 70)

print(f"k = {best_k}")
print(f"Silhouette = {best_silhouette:.4f}")
print(
    f"Maximum block size = "
    f"{int(best_row['max_block_size'])}"
)

# ------------------------------------------------------------
# Build final candidate blocks
# ------------------------------------------------------------

Z = linkage(
    condensed_distance,
    method="average"
)

labels = fcluster(
    Z,
    t=best_k,
    criterion="maxclust"
)

blocks = []

for cluster_id in sorted(np.unique(labels)):

    indices = np.where(labels == cluster_id)[0]

    block_features = [
        features[i]
        for i in indices
    ]

    # Internal second-order similarities
    if len(indices) >= 2:

        internal_values = S[
            np.ix_(indices, indices)
        ]

        upper = internal_values[
            np.triu_indices(
                len(indices),
                k=1
            )
        ]

        mean_internal = float(
            np.mean(upper)
        )

        min_internal = float(
            np.min(upper)
        )

    else:

        mean_internal = np.nan
        min_internal = np.nan

    blocks.append(
        {
            "block_id": int(cluster_id),
            "block_size": len(block_features),
            "features": " | ".join(block_features),
            "mean_internal_second_order": mean_internal,
            "min_internal_second_order": min_internal
        }
    )

blocks_df = pd.DataFrame(blocks)

blocks_df = blocks_df.sort_values(
    "block_size",
    ascending=False
).reset_index(drop=True)

# Renumber blocks after sorting
blocks_df["block_id"] = np.arange(
    1,
    len(blocks_df) + 1
)

blocks_df.to_csv(
    OUTPUT_BLOCKS,
    index=False
)

# ------------------------------------------------------------
# Print blocks
# ------------------------------------------------------------

print("\nSecond-order candidate blocks:")
print("-" * 70)

for _, row in blocks_df.iterrows():

    print(
        f"Block {int(row['block_id']):2d} "
        f"({int(row['block_size'])}): "
        f"{row['features']}"
    )

    if row["block_size"] >= 2:

        print(
            f"    mean internal S² = "
            f"{row['mean_internal_second_order']:.4f}"
        )

        print(
            f"    min internal S²  = "
            f"{row['min_internal_second_order']:.4f}"
        )

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

singleton_count = int(
    (blocks_df["block_size"] == 1).sum()
)

largest_block = int(
    blocks_df["block_size"].max()
)

print("\n" + "=" * 70)
print("STAGE 4B COMPLETE")
print("=" * 70)

print(f"\nSelected k: {best_k}")
print(f"Silhouette: {best_silhouette:.4f}")
print(f"Number of blocks: {len(blocks_df)}")
print(f"Singleton blocks: {singleton_count}")
print(f"Largest block: {largest_block}")

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 4B — SECOND-ORDER DEPENDENCY CLUSTERING\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        "Graph: cosine similarity between dependency fingerprints\n"
    )
    f.write(
        "Distance: 1 - second-order similarity\n"
    )
    f.write(
        "Linkage: average\n"
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

    f.write("Candidate blocks:\n")
    f.write("-" * 70 + "\n")

    for _, row in blocks_df.iterrows():

        f.write(
            f"Block {int(row['block_id'])}: "
            f"{row['features']}\n"
        )

        if row["block_size"] >= 2:

            f.write(
                f"  Mean internal S²: "
                f"{row['mean_internal_second_order']:.6f}\n"
            )

            f.write(
                f"  Min internal S²: "
                f"{row['min_internal_second_order']:.6f}\n"
            )

        f.write("\n")