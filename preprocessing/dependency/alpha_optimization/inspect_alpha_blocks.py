import os
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DEPENDENCY_DIR = os.path.dirname(
    BASE_DIR
)

SPEARMAN_FILE = os.path.join(
    DEPENDENCY_DIR,
    "spearman_matrix.csv"
)

NMI_FILE = os.path.join(
    DEPENDENCY_DIR,
    "normalized_mi_matrix.csv"
)

OUTPUT_DIR = BASE_DIR


# ============================================================
# ALPHAS TO INSPECT
# ============================================================

# These are the strongest candidates from Stage 3A.
#
# We include:
#   0.20 -> best silhouette
#   0.25 -> best within/between ratio
#   0.30 -> strong silhouette
#   0.35 -> strong within/between ratio
#   0.40 -> useful intermediate point

ALPHAS_TO_INSPECT = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
]


# ============================================================
# CLUSTERING SETTINGS
# ============================================================

MIN_K = 6
MAX_K = 20

MAX_BLOCK_SIZE = 7


# ============================================================
# LOAD MATRICES
# ============================================================

print("=" * 70)
print("STAGE 3B — ALPHA BLOCK INSPECTION")
print("=" * 70)

print("\nLoading matrices...")

spearman_df = pd.read_csv(
    SPEARMAN_FILE,
    index_col=0
).astype(float)

nmi_df = pd.read_csv(
    NMI_FILE,
    index_col=0
).astype(float)


# ============================================================
# VERIFY
# ============================================================

if not spearman_df.index.equals(
    nmi_df.index
):
    raise ValueError(
        "Spearman and NMI feature ordering differs."
    )

if not spearman_df.columns.equals(
    nmi_df.columns
):
    raise ValueError(
        "Spearman and NMI columns differ."
    )


features = list(
    spearman_df.index
)

print(
    f"Features: {len(features)}"
)


# ============================================================
# PROCESS EACH ALPHA
# ============================================================

for alpha in ALPHAS_TO_INSPECT:

    print("\n")
    print("=" * 70)
    print(
        f"ALPHA = {alpha:.2f}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Unified dependency
    # --------------------------------------------------------

    dependency = (
        alpha * nmi_df.values
        +
        (1.0 - alpha)
        * spearman_df.values
    )

    dependency = np.clip(
        dependency,
        0.0,
        1.0
    )

    np.fill_diagonal(
        dependency,
        1.0
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance = 1.0 - dependency

    condensed = squareform(
        distance,
        checks=False
    )

    # --------------------------------------------------------
    # Hierarchical clustering
    # --------------------------------------------------------

    Z = linkage(
        condensed,
        method="average"
    )

    # --------------------------------------------------------
    # Find best valid k using silhouette
    # --------------------------------------------------------

    best_k = None
    best_silhouette = -np.inf
    best_labels = None

    from sklearn.metrics import silhouette_score

    for k in range(
        MIN_K,
        MAX_K + 1
    ):

        labels = fcluster(
            Z,
            t=k,
            criterion="maxclust"
        )

        block_sizes = (
            pd.Series(labels)
            .value_counts()
        )

        # Reject blocks larger than 7 features.

        if block_sizes.max() > MAX_BLOCK_SIZE:
            continue

        if len(block_sizes) < 2:
            continue

        silhouette = silhouette_score(
            distance,
            labels,
            metric="precomputed"
        )

        if silhouette > best_silhouette:

            best_silhouette = silhouette
            best_k = k
            best_labels = labels

    if best_k is None:

        print(
            "No valid clustering found."
        )

        continue

    print(
        f"\nSelected k = {best_k}"
    )

    print(
        f"Silhouette = "
        f"{best_silhouette:.4f}"
    )

    # --------------------------------------------------------
    # Build blocks
    # --------------------------------------------------------

    blocks = {}

    for feature, label in zip(
        features,
        best_labels
    ):

        blocks.setdefault(
            int(label),
            []
        ).append(feature)

    # --------------------------------------------------------
    # Print blocks
    # --------------------------------------------------------

    print(
        f"\nNumber of blocks: "
        f"{len(blocks)}"
    )

    print("\nFeature blocks:\n")

    rows = []

    for block_id in sorted(
        blocks.keys()
    ):

        block_features = blocks[
            block_id
        ]

        print(
            f"Block {block_id} "
            f"({len(block_features)}):"
        )

        print(
            "    "
            + " | ".join(
                block_features
            )
        )

        # ----------------------------------------------------
        # Calculate mean within-block dependency
        # ----------------------------------------------------

        pair_values = []

        for i in range(
            len(block_features)
        ):

            for j in range(
                i + 1,
                len(block_features)
            ):

                f1 = block_features[i]
                f2 = block_features[j]

                i1 = features.index(f1)
                i2 = features.index(f2)

                pair_values.append(
                    dependency[i1, i2]
                )

        if pair_values:

            mean_dependency = np.mean(
                pair_values
            )

            min_dependency = np.min(
                pair_values
            )

            max_dependency = np.max(
                pair_values
            )

        else:

            mean_dependency = np.nan
            min_dependency = np.nan
            max_dependency = np.nan

        rows.append({

            "alpha": alpha,

            "block_id": block_id,

            "block_size":
                len(block_features),

            "features":
                " | ".join(
                    block_features
                ),

            "mean_within_dependency":
                mean_dependency,

            "min_within_dependency":
                min_dependency,

            "max_within_dependency":
                max_dependency

        })

        print(
            f"    Mean dependency: "
            f"{mean_dependency:.4f}"
            if not np.isnan(
                mean_dependency
            )
            else
            "    Singleton block"
        )

        print()

    # --------------------------------------------------------
    # Save blocks
    # --------------------------------------------------------

    output_file = os.path.join(
        OUTPUT_DIR,
        f"alpha_{alpha:.2f}_blocks.csv"
    )

    pd.DataFrame(rows).to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )


print("\n")
print("=" * 70)
print("STAGE 3B COMPLETE")
print("=" * 70)

print(
    "\nInspect the printed blocks before "
    "freezing the final alpha."
)