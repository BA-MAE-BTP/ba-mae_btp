import os
import numpy as np
import pandas as pd

from sklearn.metrics import adjusted_rand_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ALPHAS = [
    0.20,
    0.25,
    0.30,
    0.35,
    0.40
]


# ============================================================
# LOAD BLOCK FILE
# ============================================================

def load_labels(alpha):

    filename = os.path.join(
        BASE_DIR,
        f"alpha_{alpha:.2f}_blocks.csv"
    )

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nMissing file:\n{filename}"
        )

    df = pd.read_csv(filename)

    print(
        f"\nReading: "
        f"alpha_{alpha:.2f}_blocks.csv"
    )

    print(
        f"Columns found: "
        f"{list(df.columns)}"
    )

    # --------------------------------------------------------
    # The block files contain one row per block.
    #
    # The feature names are stored together in a single
    # "features" column, separated by "|".
    #
    # Therefore we reconstruct feature → block assignments.
    # --------------------------------------------------------

    if "features" in df.columns and "block_id" in df.columns:

        rows = []

        for _, row in df.iterrows():

            block_id = row["block_id"]

            feature_string = str(
                row["features"]
            )

            features = [
                x.strip()
                for x in feature_string.split("|")
            ]

            for feature in features:

                rows.append({
                    "feature": feature,
                    "block_id": block_id
                })

        assignment_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Fallback if file already has one feature per row.
    # --------------------------------------------------------

    elif (
        "feature" in df.columns
        and "block_id" in df.columns
    ):

        assignment_df = df[
            [
                "feature",
                "block_id"
            ]
        ].copy()

    else:

        raise ValueError(
            f"\nCould not understand:\n{filename}\n"
            f"Columns found: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Clean
    # --------------------------------------------------------

    assignment_df["feature"] = (
        assignment_df["feature"]
        .astype(str)
        .str.strip()
    )

    assignment_df["block_id"] = (
        assignment_df["block_id"]
        .astype(int)
    )

    # --------------------------------------------------------
    # Check duplicates
    # --------------------------------------------------------

    if assignment_df["feature"].duplicated().any():

        duplicates = (
            assignment_df[
                assignment_df["feature"].duplicated(
                    keep=False
                )
            ]
            ["feature"]
            .tolist()
        )

        raise ValueError(
            f"\nDuplicate feature assignments "
            f"found for alpha={alpha}: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Sort by feature name
    # --------------------------------------------------------

    assignment_df = (
        assignment_df
        .sort_values("feature")
        .reset_index(drop=True)
    )

    return (
        assignment_df["feature"].tolist(),
        assignment_df["block_id"].tolist()
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STAGE 3C — ALPHA STABILITY ANALYSIS")
    print("=" * 70)

    all_labels = {}

    # --------------------------------------------------------
    # Load each alpha
    # --------------------------------------------------------

    for alpha in ALPHAS:

        features, labels = load_labels(alpha)

        all_labels[alpha] = {
            "features": features,
            "labels": labels
        }

        print(
            f"  Features: {len(features)}"
        )

        print(
            f"  Blocks: {len(set(labels))}"
        )

    # --------------------------------------------------------
    # Verify identical feature sets
    # --------------------------------------------------------

    reference_features = all_labels[
        ALPHAS[0]
    ]["features"]

    reference_set = set(
        reference_features
    )

    for alpha in ALPHAS[1:]:

        current_features = all_labels[
            alpha
        ]["features"]

        current_set = set(
            current_features
        )

        if current_set != reference_set:

            missing = (
                reference_set -
                current_set
            )

            extra = (
                current_set -
                reference_set
            )

            raise ValueError(
                f"\nFeature mismatch for "
                f"alpha={alpha}\n"
                f"Missing: {missing}\n"
                f"Extra: {extra}"
            )

    print(
        "\nAll alpha solutions contain "
        f"the same {len(reference_features)} features."
    )

    # --------------------------------------------------------
    # Pairwise ARI
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("PAIRWISE ADJUSTED RAND INDEX")
    print("=" * 70)

    pairwise_results = []

    for i in range(len(ALPHAS)):

        for j in range(i + 1, len(ALPHAS)):

            alpha_a = ALPHAS[i]
            alpha_b = ALPHAS[j]

            labels_a = all_labels[
                alpha_a
            ]["labels"]

            labels_b = all_labels[
                alpha_b
            ]["labels"]

            ari = adjusted_rand_score(
                labels_a,
                labels_b
            )

            pairwise_results.append({

                "alpha_a": alpha_a,

                "alpha_b": alpha_b,

                "ARI": ari

            })

            print(
                f"α {alpha_a:.2f} vs "
                f"α {alpha_b:.2f}"
                f" → ARI = {ari:.4f}"
            )

    pairwise_df = pd.DataFrame(
        pairwise_results
    )

    # --------------------------------------------------------
    # Calculate mean stability per alpha
    # --------------------------------------------------------

    stability_results = []

    for alpha in ALPHAS:

        scores = []

        for _, row in pairwise_df.iterrows():

            if (
                row["alpha_a"] == alpha
                or
                row["alpha_b"] == alpha
            ):

                scores.append(
                    row["ARI"]
                )

        stability_results.append({

            "alpha": alpha,

            "mean_ARI":
                np.mean(scores),

            "min_ARI":
                np.min(scores),

            "max_ARI":
                np.max(scores),

            "num_comparisons":
                len(scores)

        })

    stability_df = pd.DataFrame(
        stability_results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pairwise_file = os.path.join(
        BASE_DIR,
        "alpha_pairwise_ARI.csv"
    )

    stability_file = os.path.join(
        BASE_DIR,
        "alpha_stability_summary.csv"
    )

    pairwise_df.to_csv(
        pairwise_file,
        index=False
    )

    stability_df.to_csv(
        stability_file,
        index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("ALPHA STABILITY SUMMARY")
    print("=" * 70)

    print(
        stability_df.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )

    # --------------------------------------------------------
    # Most stable
    # --------------------------------------------------------

    most_stable = (
        stability_df
        .sort_values(
            "mean_ARI",
            ascending=False
        )
        .iloc[0]
    )

    print("\n")
    print("=" * 70)
    print("MOST STABLE ALPHA")
    print("=" * 70)

    print(
        f"α = "
        f"{most_stable['alpha']:.2f}"
    )

    print(
        f"Mean ARI = "
        f"{most_stable['mean_ARI']:.4f}"
    )

    print(
        f"Minimum ARI = "
        f"{most_stable['min_ARI']:.4f}"
    )

    print(
        f"Maximum ARI = "
        f"{most_stable['max_ARI']:.4f}"
    )

    print("\n")
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(pairwise_file)
    print(stability_file)

    print("\nStage 3C complete.")


if __name__ == "__main__":
    main()