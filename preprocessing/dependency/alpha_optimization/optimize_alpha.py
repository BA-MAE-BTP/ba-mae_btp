import os
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# BASE_DIR = ...\preprocessing\dependency\alpha_optimization
# DEPENDENCY_DIR = ...\preprocessing\dependency

DEPENDENCY_DIR = os.path.dirname(BASE_DIR)

SPEARMAN_FILE = os.path.join(
    DEPENDENCY_DIR,
    "spearman_matrix.csv"
)

MI_FILE = os.path.join(
    DEPENDENCY_DIR,
    "normalized_mi_matrix.csv"
)

OUTPUT_DIR = BASE_DIR

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

# Balanced range:
# Both MI and Spearman always contribute.

ALPHAS = np.round(
    np.arange(0.20, 0.801, 0.05),
    2
)

MIN_K = 6
MAX_K = 20

# Proposal requires blocks of at most 5–7 features.
MAX_BLOCK_SIZE = 7


# ============================================================
# LOAD MATRIX
# ============================================================

def load_matrix(path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nFile not found:\n{path}\n"
        )

    df = pd.read_csv(
        path,
        index_col=0
    )

    df = df.astype(float)

    return df


# ============================================================
# EVALUATE ONE ALPHA
# ============================================================

def evaluate_alpha(
    alpha,
    spearman_df,
    nmi_df
):

    # --------------------------------------------------------
    # Unified dependency
    #
    # D = alpha * NMI
    #     + (1-alpha) * |Spearman|
    # --------------------------------------------------------

    dependency = (
        alpha * nmi_df +
        (1.0 - alpha) * spearman_df
    )

    dependency = dependency.clip(
        lower=0.0,
        upper=1.0
    )

    np.fill_diagonal(
        dependency.values,
        1.0
    )

    # --------------------------------------------------------
    # Convert dependency to distance
    #
    # high dependency -> small distance
    # low dependency  -> large distance
    # --------------------------------------------------------

    distance = 1.0 - dependency

    condensed_distance = squareform(
        distance.values,
        checks=False
    )

    # --------------------------------------------------------
    # Hierarchical clustering
    # --------------------------------------------------------

    Z = linkage(
        condensed_distance,
        method="average"
    )

    candidates = []

    # --------------------------------------------------------
    # Test k = 6 ... 20
    # --------------------------------------------------------

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

        # Reject solutions containing
        # excessively large blocks.

        if block_sizes.max() > MAX_BLOCK_SIZE:
            continue

        # Need at least two clusters.

        if len(block_sizes) < 2:
            continue

        # ----------------------------------------------------
        # Silhouette score
        # ----------------------------------------------------

        silhouette = silhouette_score(
            distance.values,
            labels,
            metric="precomputed"
        )

        # ----------------------------------------------------
        # Within-block and between-block dependency
        # ----------------------------------------------------

        within_values = []
        between_values = []

        n_features = dependency.shape[0]

        for i in range(n_features):

            for j in range(
                i + 1,
                n_features
            ):

                if labels[i] == labels[j]:

                    within_values.append(
                        dependency.iloc[i, j]
                    )

                else:

                    between_values.append(
                        dependency.iloc[i, j]
                    )

        mean_within = (
            np.mean(within_values)
            if within_values
            else 0.0
        )

        mean_between = (
            np.mean(between_values)
            if between_values
            else 0.0
        )

        if mean_between > 0:

            within_between_ratio = (
                mean_within /
                mean_between
            )

        else:

            within_between_ratio = np.inf

        candidates.append({

            "alpha": alpha,

            "k": k,

            "silhouette": silhouette,

            "mean_within": mean_within,

            "mean_between": mean_between,

            "within_between_ratio":
                within_between_ratio,

            "min_block_size":
                block_sizes.min(),

            "max_block_size":
                block_sizes.max(),

            "mean_block_size":
                block_sizes.mean()

        })

    # --------------------------------------------------------
    # No valid clustering
    # --------------------------------------------------------

    if not candidates:
        return None

    # --------------------------------------------------------
    # Choose best k for this alpha
    #
    # Primary:
    #   silhouette
    #
    # Secondary:
    #   within/between ratio
    # --------------------------------------------------------

    candidates = sorted(
        candidates,
        key=lambda x: (
            x["silhouette"],
            x["within_between_ratio"]
        ),
        reverse=True
    )

    return candidates[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 3A — ALPHA OPTIMIZATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load matrices
    # --------------------------------------------------------

    print("\nLoading dependency matrices...")

    spearman_df = load_matrix(
        SPEARMAN_FILE
    )

    nmi_df = load_matrix(
        MI_FILE
    )

    # --------------------------------------------------------
    # Verify matrices
    # --------------------------------------------------------

    if not spearman_df.index.equals(
        nmi_df.index
    ):

        raise ValueError(
            "Spearman and NMI matrices have different feature ordering."
        )

    if not spearman_df.columns.equals(
        nmi_df.columns
    ):

        raise ValueError(
            "Spearman and NMI matrices have different columns."
        )

    if not spearman_df.index.equals(
        spearman_df.columns
    ):

        raise ValueError(
            "Spearman matrix is not square."
        )

    features = list(
        spearman_df.index
    )

    print(
        f"Features: {len(features)}"
    )

    print(
        f"Alpha range: "
        f"{ALPHAS[0]:.2f} "
        f"→ "
        f"{ALPHAS[-1]:.2f}"
    )

    print(
        f"Number of alpha values: "
        f"{len(ALPHAS)}"
    )

    print(
        f"Allowed k: "
        f"{MIN_K}–{MAX_K}"
    )

    print(
        f"Maximum block size: "
        f"{MAX_BLOCK_SIZE}"
    )

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    results = []

    for alpha in ALPHAS:

        print(
            f"\nTesting α = {alpha:.2f}"
        )

        result = evaluate_alpha(
            alpha,
            spearman_df,
            nmi_df
        )

        if result is None:

            print(
                "  No valid clustering."
            )

            continue

        results.append(result)

        print(
            f"  k={result['k']} | "
            f"silhouette="
            f"{result['silhouette']:.4f} | "
            f"within="
            f"{result['mean_within']:.4f} | "
            f"between="
            f"{result['mean_between']:.4f} | "
            f"ratio="
            f"{result['within_between_ratio']:.4f} | "
            f"max_block="
            f"{result['max_block_size']}"
        )

    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:

        raise RuntimeError(
            "No valid alpha/clustering results were produced."
        )

    # --------------------------------------------------------
    # Save raw results
    # --------------------------------------------------------

    raw_file = os.path.join(
        OUTPUT_DIR,
        "alpha_results_constrained.csv"
    )

    results_df.to_csv(
        raw_file,
        index=False
    )

    # --------------------------------------------------------
    # Rank alpha values
    # --------------------------------------------------------

    results_df["rank_silhouette"] = (
        results_df["silhouette"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    results_df["rank_ratio"] = (
        results_df["within_between_ratio"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # Equal importance for now.

    results_df["combined_rank"] = (
        results_df["rank_silhouette"] +
        results_df["rank_ratio"]
    )

    ranked_df = (
        results_df
        .sort_values(
            [
                "combined_rank",
                "silhouette",
                "within_between_ratio"
            ],
            ascending=[
                True,
                False,
                False
            ]
        )
    )

    ranked_file = os.path.join(
        OUTPUT_DIR,
        "alpha_ranked.csv"
    )

    ranked_df.to_csv(
        ranked_file,
        index=False
    )

    # --------------------------------------------------------
    # Print complete results
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ALL RESULTS")
    print("=" * 70)

    display_columns = [

        "alpha",
        "k",
        "silhouette",
        "mean_within",
        "mean_between",
        "within_between_ratio",
        "min_block_size",
        "max_block_size",
        "mean_block_size"

    ]

    print(
        results_df[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Print top candidates
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TOP ALPHA CANDIDATES")
    print("=" * 70)

    print(
        ranked_df[
            display_columns
        ]
        .head(5)
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"\n{raw_file}"
    )

    print(
        f"{ranked_file}"
    )

    print("\nStage 3A complete.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()