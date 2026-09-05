import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (
    linkage,
    dendrogram,
    fcluster
)

from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ============================================================
# BA-MAE — HIERARCHICAL FEATURE CLUSTERING
# Unified Dependency → Feature Blocks
# ============================================================

INPUT_PATH = (
    "preprocessing/dependency/"
    "unified_dependency_matrix.csv"
)

OUTPUT_DIR = "preprocessing/dependency"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Average linkage is used because it considers the
# average distance between members of two clusters.
LINKAGE_METHOD = "average"

# We want candidate blocks of approximately this size.
MIN_BLOCK_SIZE = 1
MAX_BLOCK_SIZE = 7

# We will examine multiple cluster counts instead of
# arbitrarily selecting one.
MIN_CLUSTERS = 6
MAX_CLUSTERS = 20


# ============================================================
# LOAD DEPENDENCY MATRIX
# ============================================================

def load_dependency_matrix():

    print("\n" + "=" * 70)
    print("STEP 1 — LOADING UNIFIED DEPENDENCY MATRIX")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH,
        index_col=0
    )

    # Make sure rows and columns match
    assert list(df.index) == list(df.columns), (
        "Dependency matrix row/column mismatch."
    )

    print(
        f"Features loaded: {len(df)}"
    )

    print(
        f"Matrix shape: {df.shape}"
    )

    return df


# ============================================================
# CONVERT DEPENDENCY → DISTANCE
# ============================================================

def dependency_to_distance(dependency):

    print("\n" + "=" * 70)
    print("STEP 2 — CONVERTING DEPENDENCY TO DISTANCE")
    print("=" * 70)

    # High dependency = small distance
    #
    # D = 1 - dependency

    distance = 1.0 - dependency

    # Numerical safety
    distance = distance.clip(
        lower=0.0,
        upper=1.0
    )

    # Force diagonal to zero
    np.fill_diagonal(
        distance.values,
        0.0
    )

    # Force exact symmetry
    distance = (
        distance + distance.T
    ) / 2

    print(
        "Distance = 1 − Unified Dependency"
    )

    print(
        f"Minimum distance: "
        f"{distance.values[np.triu_indices_from(distance, 1)].min():.4f}"
    )

    print(
        f"Maximum distance: "
        f"{distance.values[np.triu_indices_from(distance, 1)].max():.4f}"
    )

    return distance


# ============================================================
# HIERARCHICAL LINKAGE
# ============================================================

def perform_linkage(distance):

    print("\n" + "=" * 70)
    print("STEP 3 — HIERARCHICAL CLUSTERING")
    print("=" * 70)

    # scipy linkage expects condensed distance matrix
    condensed_distance = squareform(
        distance.values,
        checks=True
    )

    Z = linkage(
        condensed_distance,
        method=LINKAGE_METHOD
    )

    print(
        f"Linkage method: {LINKAGE_METHOD}"
    )

    print(
        "Hierarchical clustering completed."
    )

    return Z


# ============================================================
# DENDROGRAM
# ============================================================

def save_dendrogram(Z, features):

    print("\n" + "=" * 70)
    print("STEP 4 — GENERATING DENDROGRAM")
    print("=" * 70)

    plt.figure(
        figsize=(18, 10)
    )

    dendrogram(
        Z,
        labels=features,
        leaf_rotation=90,
        leaf_font_size=8,
        color_threshold=None
    )

    plt.title(
        "Hierarchical Clustering of Network Features"
    )

    plt.xlabel(
        "Features"
    )

    plt.ylabel(
        "Distance = 1 − Dependency"
    )

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "feature_dendrogram.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {path}"
    )


# ============================================================
# CLUSTER QUALITY
# ============================================================

def evaluate_cluster_counts(
    Z,
    distance,
    features
):

    print("\n" + "=" * 70)
    print("STEP 5 — EVALUATING CLUSTER COUNTS")
    print("=" * 70)

    condensed_distance = squareform(
        distance.values
    )

    results = []

    for k in range(
        MIN_CLUSTERS,
        MAX_CLUSTERS + 1
    ):

        labels = fcluster(
            Z,
            t=k,
            criterion="maxclust"
        )

        unique_labels = np.unique(
            labels
        )

        if len(unique_labels) < 2:
            continue

        # Silhouette using precomputed distances
        silhouette = silhouette_score(
            distance.values,
            labels,
            metric="precomputed"
        )

        sizes = pd.Series(
            labels
        ).value_counts()

        results.append({

            "Num_Clusters": k,

            "Silhouette_Score":
                silhouette,

            "Min_Block_Size":
                sizes.min(),

            "Max_Block_Size":
                sizes.max(),

            "Mean_Block_Size":
                sizes.mean()
        })

        print(
            f"k={k:2d} | "
            f"Silhouette={silhouette:.4f} | "
            f"Min={sizes.min():2d} | "
            f"Max={sizes.max():2d} | "
            f"Mean={sizes.mean():.2f}"
        )

    results_df = pd.DataFrame(
        results
    )

    path = os.path.join(
        OUTPUT_DIR,
        "cluster_evaluation.csv"
    )

    results_df.to_csv(
        path,
        index=False
    )

    print(
        f"\nSaved: {path}"
    )

    return results_df


# ============================================================
# GENERATE CANDIDATE BLOCKS
# ============================================================

def generate_candidate_blocks(
    Z,
    features,
    k
):

    labels = fcluster(
        Z,
        t=k,
        criterion="maxclust"
    )

    blocks = {}

    for feature, label in zip(
        features,
        labels
    ):

        blocks.setdefault(
            label,
            []
        ).append(feature)

    return blocks


# ============================================================
# BLOCK STATISTICS
# ============================================================

def calculate_block_statistics(
    blocks,
    dependency
):

    rows = []

    for block_id, features in blocks.items():

        # Within-block dependency
        values = []

        for i in range(len(features)):

            for j in range(
                i + 1,
                len(features)
            ):

                values.append(
                    dependency.loc[
                        features[i],
                        features[j]
                    ]
                )

        if len(values) > 0:

            mean_dependency = np.mean(
                values
            )

            min_dependency = np.min(
                values
            )

            max_dependency = np.max(
                values
            )

        else:

            mean_dependency = np.nan
            min_dependency = np.nan
            max_dependency = np.nan

        rows.append({

            "Block_ID": block_id,

            "Block_Size":
                len(features),

            "Mean_Within_Dependency":
                mean_dependency,

            "Min_Within_Dependency":
                min_dependency,

            "Max_Within_Dependency":
                max_dependency,

            "Features":
                " | ".join(features)
        })

    return pd.DataFrame(rows)


# ============================================================
# SELECT CANDIDATE K
# ============================================================

def choose_candidate_k(
    evaluation
):

    print("\n" + "=" * 70)
    print("STEP 6 — SELECTING CANDIDATE CLUSTERING")
    print("=" * 70)

    # Prefer solutions where the largest cluster
    # does not exceed the desired 5–7 feature range.
    valid = evaluation[
        evaluation["Max_Block_Size"]
        <= MAX_BLOCK_SIZE
    ].copy()

    if len(valid) == 0:

        print(
            "No solution has all blocks <= 7."
        )

        print(
            "We will retain the highest-silhouette "
            "solution for manual refinement."
        )

        best = evaluation.loc[
            evaluation["Silhouette_Score"].idxmax()
        ]

    else:

        # Among valid solutions, choose highest silhouette.
        best = valid.loc[
            valid["Silhouette_Score"].idxmax()
        ]

    k = int(
        best["Num_Clusters"]
    )

    print(
        f"Candidate number of clusters: {k}"
    )

    print(
        f"Silhouette score: "
        f"{best['Silhouette_Score']:.4f}"
    )

    print(
        f"Maximum block size: "
        f"{best['Max_Block_Size']}"
    )

    return k


# ============================================================
# SAVE BLOCKS
# ============================================================

def save_blocks(
    blocks,
    dependency,
    k
):

    print("\n" + "=" * 70)
    print("STEP 7 — SAVING CANDIDATE FEATURE BLOCKS")
    print("=" * 70)

    statistics = calculate_block_statistics(
        blocks,
        dependency
    )

    # Sort blocks by size
    statistics.sort_values(
        "Block_ID",
        inplace=True
    )

    path = os.path.join(
        OUTPUT_DIR,
        "candidate_feature_blocks.csv"
    )

    statistics.to_csv(
        path,
        index=False
    )

    print(
        f"Saved: {path}"
    )

    print("\nCandidate blocks:\n")

    for _, row in statistics.iterrows():

        print(
            f"Block {int(row['Block_ID'])} "
            f"({int(row['Block_Size'])} features)"
        )

        print(
            f"  Mean dependency: "
            f"{row['Mean_Within_Dependency']:.4f}"
            if not pd.isna(
                row["Mean_Within_Dependency"]
            )
            else
            "  Singleton block"
        )

        for feature in row[
            "Features"
        ].split(" | "):

            print(
                f"    - {feature}"
            )

        print()

    return statistics


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BA-MAE — HIERARCHICAL FEATURE BLOCK DISCOVERY")
    print("=" * 70)

    # 1. Load dependency matrix
    dependency = load_dependency_matrix()

    features = list(
        dependency.columns
    )

    # 2. Dependency → distance
    distance = dependency_to_distance(
        dependency
    )

    # 3. Hierarchical clustering
    Z = perform_linkage(
        distance
    )

    # 4. Dendrogram
    save_dendrogram(
        Z,
        features
    )

    # 5. Evaluate multiple cluster counts
    evaluation = evaluate_cluster_counts(
        Z,
        distance,
        features
    )

    # 6. Select candidate number of clusters
    k = choose_candidate_k(
        evaluation
    )

    # 7. Generate candidate blocks
    blocks = generate_candidate_blocks(
        Z,
        features,
        k
    )

    # 8. Save blocks
    save_blocks(
        blocks,
        dependency,
        k
    )

    print("\n" + "=" * 70)
    print("HIERARCHICAL CLUSTERING COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")

    print(
        "  - feature_dendrogram.png"
    )

    print(
        "  - cluster_evaluation.csv"
    )

    print(
        "  - candidate_feature_blocks.csv"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These are CANDIDATE blocks."
    )

    print(
        "Do not use them for MAE masking yet."
    )

    print(
        "They must be validated first."
    )


if __name__ == "__main__":
    main() 