import os
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STAGE 4D — STRONG-NEIGHBOR SECOND-ORDER GRAPH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "second_order_dependency_matrix.csv"
)

# IMPORTANT:
# We actually need the FIRST-ORDER matrix here.
FIRST_ORDER_FILE = os.path.join(
    os.path.dirname(BASE_DIR),
    "alpha_optimization",
    "final_dependency_matrix.csv"
)

print("=" * 70)
print("STAGE 4D — STRONG-NEIGHBOR SECOND-ORDER GRAPH")
print("=" * 70)


# ------------------------------------------------------------
# Load first-order dependency matrix
# ------------------------------------------------------------

print("\nLoading first-order dependency matrix...")

D_df = pd.read_csv(
    FIRST_ORDER_FILE,
    index_col=0
)

features = list(D_df.index)
D = D_df.to_numpy(dtype=float)

n = len(features)

if D.shape != (39, 39):
    raise ValueError(
        f"Expected 39x39 matrix, got {D.shape}"
    )

print(f"Features: {n}")
print("✓ First-order matrix loaded")


# ------------------------------------------------------------
# Thresholds
# ------------------------------------------------------------

thresholds = [
    0.50,
    0.60,
    0.70
]


for threshold in thresholds:

    print("\n" + "=" * 70)
    print(
        f"STRONG-NEIGHBOR THRESHOLD = {threshold:.2f}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Build sparse dependency fingerprints
    # --------------------------------------------------------

    profiles = D.copy()

    # Remove self-dependency
    np.fill_diagonal(
        profiles,
        0.0
    )

    # Keep only strong direct dependencies
    profiles[
        profiles < threshold
    ] = 0.0

    # --------------------------------------------------------
    # Count strong neighbors
    # --------------------------------------------------------

    neighbor_counts = np.sum(
        profiles > 0,
        axis=1
    )

    print("\nStrong-neighbor counts:")

    print(
        f"Min    : {neighbor_counts.min():.0f}"
    )

    print(
        f"Median : {np.median(neighbor_counts):.0f}"
    )

    print(
        f"Mean   : {neighbor_counts.mean():.2f}"
    )

    print(
        f"Max    : {neighbor_counts.max():.0f}"
    )

    # --------------------------------------------------------
    # Calculate second-order similarity
    # --------------------------------------------------------

    print(
        "\nComputing sparse fingerprint similarity..."
    )

    S2 = cosine_similarity(
        profiles
    )

    S2 = np.clip(
        S2,
        0.0,
        1.0
    )

    S2 = (
        S2 + S2.T
    ) / 2

    np.fill_diagonal(
        S2,
        1.0
    )

    # --------------------------------------------------------
    # Save matrix
    # --------------------------------------------------------

    output_matrix = os.path.join(
        BASE_DIR,
        f"second_order_strong_{threshold:.2f}.csv"
    )

    S2_df = pd.DataFrame(
        S2,
        index=features,
        columns=features
    )

    S2_df.to_csv(
        output_matrix
    )

    print(
        f"\nSaved: {output_matrix}"
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    upper = S2[
        np.triu_indices(
            n,
            k=1
        )
    ]

    print("\nSecond-order similarity:")

    print(
        f"Min    : {upper.min():.4f}"
    )

    print(
        f"Median : {np.median(upper):.4f}"
    )

    print(
        f"Q75    : {np.percentile(upper, 75):.4f}"
    )

    print(
        f"Q90    : {np.percentile(upper, 90):.4f}"
    )

    print(
        f"Q95    : {np.percentile(upper, 95):.4f}"
    )

    print(
        f"Max    : {upper.max():.4f}"
    )

    # --------------------------------------------------------
    # Top relationships
    # --------------------------------------------------------

    pairs = []

    for i in range(n):

        for j in range(i + 1, n):

            pairs.append(
                (
                    features[i],
                    features[j],
                    S2[i, j]
                )
            )

    pairs_df = pd.DataFrame(
        pairs,
        columns=[
            "feature_1",
            "feature_2",
            "second_order_similarity"
        ]
    )

    pairs_df = pairs_df.sort_values(
        "second_order_similarity",
        ascending=False
    )

    print(
        "\nTop 15 strong-neighbor second-order relationships:"
    )

    for _, row in pairs_df.head(15).iterrows():

        print(
            f"{row['feature_1']} <-> "
            f"{row['feature_2']} : "
            f"{row['second_order_similarity']:.4f}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_file = os.path.join(
        BASE_DIR,
        f"second_order_strong_{threshold:.2f}_summary.txt"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "STAGE 4D — STRONG-NEIGHBOR SECOND-ORDER GRAPH\n"
        )

        f.write("=" * 70 + "\n\n")

        f.write(
            f"Strong dependency threshold: {threshold:.2f}\n"
        )

        f.write(
            "Base dependency:\n"
        )

        f.write(
            "D = 0.35*NMI + 0.65*abs(Spearman)\n\n"
        )

        f.write(
            "Fingerprint construction:\n"
        )

        f.write(
            "1. Set diagonal to zero.\n"
        )

        f.write(
            "2. Keep only D >= threshold.\n"
        )

        f.write(
            "3. Compute cosine similarity between sparse fingerprints.\n\n"
        )

        f.write(
            "Strong-neighbor counts:\n"
        )

        f.write(
            f"Min: {neighbor_counts.min():.0f}\n"
        )

        f.write(
            f"Median: {np.median(neighbor_counts):.0f}\n"
        )

        f.write(
            f"Mean: {neighbor_counts.mean():.2f}\n"
        )

        f.write(
            f"Max: {neighbor_counts.max():.0f}\n\n"
        )

        f.write(
            "Second-order similarity statistics:\n"
        )

        f.write(
            f"Min: {upper.min():.6f}\n"
        )

        f.write(
            f"Median: {np.median(upper):.6f}\n"
        )

        f.write(
            f"Q75: {np.percentile(upper, 75):.6f}\n"
        )

        f.write(
            f"Q90: {np.percentile(upper, 90):.6f}\n"
        )

        f.write(
            f"Q95: {np.percentile(upper, 95):.6f}\n"
        )

        f.write(
            f"Max: {upper.max():.6f}\n\n"
        )

        f.write(
            "Top 15 relationships:\n"
        )

        for rank, (_, row) in enumerate(
            pairs_df.head(15).iterrows(),
            start=1
        ):

            f.write(
                f"{rank}. "
                f"{row['feature_1']} <-> "
                f"{row['feature_2']} : "
                f"{row['second_order_similarity']:.6f}\n"
            )

    print(
        f"Saved: {summary_file}"
    )


print("\n" + "=" * 70)
print("STAGE 4D COMPLETE")
print("=" * 70)