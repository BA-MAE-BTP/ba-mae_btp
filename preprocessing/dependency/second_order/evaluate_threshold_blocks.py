import os
import numpy as np
import pandas as pd

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


# ============================================================
# STAGE 4C — EVALUATE THRESHOLDED SECOND-ORDER BLOCKS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECOND_ORDER_FILE = os.path.join(
    BASE_DIR,
    "second_order_dependency_matrix.csv"
)

FIRST_ORDER_FILE = os.path.join(
    os.path.dirname(BASE_DIR),
    "alpha_optimization",
    "final_dependency_matrix.csv"
)

OUTPUT_RESULTS = os.path.join(
    BASE_DIR,
    "second_order_threshold_block_results.csv"
)

OUTPUT_SUMMARY = os.path.join(
    BASE_DIR,
    "second_order_threshold_block_summary.txt"
)

print("=" * 70)
print("STAGE 4C — THRESHOLDED SECOND-ORDER BLOCK EVALUATION")
print("=" * 70)


# ------------------------------------------------------------
# Load matrices
# ------------------------------------------------------------

print("\nLoading matrices...")

S_df = pd.read_csv(
    SECOND_ORDER_FILE,
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
print(f"Second-order shape: {S.shape}")
print(f"First-order shape:  {D.shape}")


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------

if n != 39:
    raise ValueError(
        f"Expected 39 features, got {n}"
    )

if not np.allclose(S, S.T, atol=1e-8):
    raise ValueError(
        "Second-order matrix is not symmetric."
    )

if not np.allclose(D, D.T, atol=1e-8):
    raise ValueError(
        "First-order matrix is not symmetric."
    )

print("✓ Matrices validated")


# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------

thresholds = [
    0.85,
    0.88,
    0.90,
    0.91,
    0.92,
    0.93
]

results = []
all_blocks = []


# ------------------------------------------------------------
# Evaluate each threshold
# ------------------------------------------------------------

for threshold in thresholds:

    print("\n" + "-" * 70)
    print(f"SECOND-ORDER THRESHOLD = {threshold:.2f}")
    print("-" * 70)

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

    largest_component = int(
        component_sizes.max()
    )

    singleton_count = int(
        np.sum(component_sizes == 1)
    )

    print(
        f"Components: {n_components}"
    )

    print(
        f"Largest component: "
        f"{largest_component}"
    )

    print(
        f"Singletons: "
        f"{singleton_count}"
    )

    # --------------------------------------------------------
    # Calculate block quality using FIRST-ORDER D
    # --------------------------------------------------------

    within_values = []
    between_values = []

    block_records = []

    for component_id in range(n_components):

        indices = np.where(
            labels == component_id
        )[0]

        block_features = [
            features[i]
            for i in indices
        ]

        size = len(indices)

        # Internal first-order dependencies
        if size >= 2:

            internal = D[
                np.ix_(
                    indices,
                    indices
                )
            ]

            upper = internal[
                np.triu_indices(
                    size,
                    k=1
                )
            ]

            mean_within = float(
                np.mean(upper)
            )

            min_within = float(
                np.min(upper)
            )

            within_values.extend(
                upper.tolist()
            )

        else:

            mean_within = np.nan
            min_within = np.nan

        block_records.append(
            {
                "threshold": threshold,
                "component": component_id + 1,
                "size": size,
                "features": " | ".join(
                    block_features
                ),
                "mean_first_order_within": mean_within,
                "min_first_order_within": min_within
            }
        )

    # --------------------------------------------------------
    # Between-block first-order dependency
    # --------------------------------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            if labels[i] != labels[j]:

                between_values.append(
                    D[i, j]
                )

    # --------------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------------

    mean_within = (
        float(np.mean(within_values))
        if within_values
        else np.nan
    )

    mean_between = (
        float(np.mean(between_values))
        if between_values
        else np.nan
    )

    ratio = (
        mean_within / mean_between
        if mean_between > 0
        else np.nan
    )

    # Cross-block maximum
    max_cross = (
        float(np.max(between_values))
        if between_values
        else np.nan
    )

    # Strong cross-block pairs
    strong_cross = sum(
        value >= 0.70
        for value in between_values
    )

    # Strong within-block pairs
    strong_within = sum(
        value >= 0.70
        for value in within_values
    )

    results.append(
        {
            "threshold": threshold,
            "number_of_blocks": n_components,
            "singleton_blocks": singleton_count,
            "largest_block": largest_component,
            "mean_block_size": n / n_components,
            "mean_first_order_within": mean_within,
            "mean_first_order_between": mean_between,
            "within_between_ratio": ratio,
            "max_cross_block_first_order": max_cross,
            "within_pairs_ge_0.70": strong_within,
            "cross_pairs_ge_0.70": strong_cross
        }
    )

    all_blocks.extend(
        block_records
    )

    print(
        f"Mean first-order within : "
        f"{mean_within:.4f}"
    )

    print(
        f"Mean first-order between: "
        f"{mean_between:.4f}"
    )

    print(
        f"Within/between ratio    : "
        f"{ratio:.4f}"
    )

    print(
        f"Max cross-block D       : "
        f"{max_cross:.4f}"
    )

    print(
        f"Within pairs >= 0.70    : "
        f"{strong_within}"
    )

    print(
        f"Cross pairs >= 0.70     : "
        f"{strong_cross}"
    )


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

results_df = pd.DataFrame(
    results
)

blocks_df = pd.DataFrame(
    all_blocks
)

results_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)

blocks_output = os.path.join(
    BASE_DIR,
    "second_order_threshold_blocks.csv"
)

blocks_df.to_csv(
    blocks_output,
    index=False
)


# ------------------------------------------------------------
# Print comparison
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("THRESHOLD COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
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
        "STAGE 4C — THRESHOLDED SECOND-ORDER BLOCK EVALUATION\n"
    )

    f.write("=" * 70 + "\n\n")

    f.write(
        "Second-order graph: cosine similarity between dependency fingerprints\n"
    )

    f.write(
        "Evaluation reference: original first-order dependency matrix D\n\n"
    )

    f.write(
        results_df.to_string(
            index=False
        )
    )

    f.write("\n\nBLOCK DETAILS\n")
    f.write("-" * 70 + "\n")

    for threshold in thresholds:

        f.write(
            f"\nThreshold = {threshold:.2f}\n"
        )

        subset = blocks_df[
            blocks_df["threshold"] == threshold
        ]

        for _, row in subset.iterrows():

            f.write(
                f"Block {int(row['component'])}: "
                f"{row['features']}\n"
            )

            if row["size"] >= 2:

                f.write(
                    f"  Size: {int(row['size'])}\n"
                )

                f.write(
                    f"  Mean first-order within: "
                    f"{row['mean_first_order_within']:.6f}\n"
                )

                f.write(
                    f"  Min first-order within: "
                    f"{row['min_first_order_within']:.6f}\n"
                )

print("\nSaved:")
print(OUTPUT_RESULTS)
print(blocks_output)
print(OUTPUT_SUMMARY)

print("\n" + "=" * 70)
print("STAGE 4C COMPLETE")
print("=" * 70)