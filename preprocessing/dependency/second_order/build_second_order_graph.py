import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# STAGE 4A — BUILD SECOND-ORDER DEPENDENCY GRAPH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPENDENCY_DIR = os.path.dirname(BASE_DIR)

INPUT_FILE = os.path.join(
    DEPENDENCY_DIR,
    "alpha_optimization",
    "final_dependency_matrix.csv"
)

OUTPUT_DIR = BASE_DIR

OUTPUT_MATRIX = os.path.join(
    OUTPUT_DIR,
    "second_order_dependency_matrix.csv"
)

OUTPUT_SUMMARY = os.path.join(
    OUTPUT_DIR,
    "second_order_dependency_summary.txt"
)

print("=" * 70)
print("STAGE 4A — BUILD SECOND-ORDER DEPENDENCY GRAPH")
print("=" * 70)

print("\nLoading first-order dependency matrix...")
print(INPUT_FILE)

D_df = pd.read_csv(INPUT_FILE, index_col=0)

features = list(D_df.index)
D = D_df.to_numpy(dtype=float)

# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

print("\nValidation:")

print(f"Number of features: {len(features)}")
print(f"Matrix shape: {D.shape}")

if D.shape != (39, 39):
    raise ValueError(
        f"Expected 39x39 matrix, got {D.shape}"
    )

if list(D_df.columns) != features:
    raise ValueError(
        "Row and column feature ordering does not match."
    )

if not np.allclose(D, D.T, atol=1e-8):
    raise ValueError("Dependency matrix is not symmetric.")

if not np.allclose(np.diag(D), 1.0, atol=1e-8):
    raise ValueError("Dependency matrix diagonal is not 1.")

if np.isnan(D).any() or np.isinf(D).any():
    raise ValueError("Dependency matrix contains NaN or Inf.")

print("✓ 39 features")
print("✓ Matrix is symmetric")
print("✓ Diagonal = 1")
print("✓ No NaN/Inf")

# ------------------------------------------------------------
# Remove self-dependency from fingerprints
# ------------------------------------------------------------

print("\nRemoving self-dependency from feature fingerprints...")

D_profile = D.copy()

# Self-dependency D[i,i] = 1 would artificially increase
# similarity between feature fingerprints.
np.fill_diagonal(D_profile, 0.0)

print("✓ Diagonal set to 0 for second-order profiles")

# ------------------------------------------------------------
# Construct second-order similarity
# ------------------------------------------------------------

print("\nComputing second-order dependency similarity...")

# Each row is now a dependency fingerprint.
#
# S2[i,j] = cosine similarity between
# dependency fingerprint of feature i and feature j.

S2 = cosine_similarity(D_profile)

# Numerical cleanup
S2 = np.clip(S2, 0.0, 1.0)

# Force exact symmetry and diagonal
S2 = (S2 + S2.T) / 2
np.fill_diagonal(S2, 1.0)

S2_df = pd.DataFrame(
    S2,
    index=features,
    columns=features
)

# ------------------------------------------------------------
# Save matrix
# ------------------------------------------------------------

S2_df.to_csv(OUTPUT_MATRIX)

print(f"\nSaved:")
print(OUTPUT_MATRIX)

# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

upper = S2[np.triu_indices(len(features), k=1)]

print("\nSecond-order similarity statistics:")
print(f"Min:    {upper.min():.4f}")
print(f"Q25:    {np.percentile(upper, 25):.4f}")
print(f"Median: {np.percentile(upper, 50):.4f}")
print(f"Q75:    {np.percentile(upper, 75):.4f}")
print(f"Q90:    {np.percentile(upper, 90):.4f}")
print(f"Q95:    {np.percentile(upper, 95):.4f}")
print(f"Q99:    {np.percentile(upper, 99):.4f}")
print(f"Max:    {upper.max():.4f}")

# ------------------------------------------------------------
# Top second-order relationships
# ------------------------------------------------------------

pairs = []

for i in range(len(features)):
    for j in range(i + 1, len(features)):
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

print("\nTop 20 second-order relationships:")

for idx, row in pairs_df.head(20).iterrows():
    print(
        f"{row['feature_1']} <-> "
        f"{row['feature_2']} : "
        f"{row['second_order_similarity']:.4f}"
    )

# ------------------------------------------------------------
# Save summary
# ------------------------------------------------------------

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:

    f.write("STAGE 4A — SECOND-ORDER DEPENDENCY GRAPH\n")
    f.write("=" * 70 + "\n\n")

    f.write("Base first-order dependency:\n")
    f.write("D = 0.35*NMI + 0.65*abs(Spearman)\n\n")

    f.write("Second-order construction:\n")
    f.write(
        "1. Set diagonal of D to zero.\n"
        "2. Treat each row as a dependency fingerprint.\n"
        "3. Compute cosine similarity between fingerprints.\n"
        "4. Use resulting similarity matrix as second-order graph.\n\n"
    )

    f.write("Matrix statistics:\n")
    f.write(f"Features: {len(features)}\n")
    f.write(f"Min: {upper.min():.6f}\n")
    f.write(f"Median: {np.percentile(upper, 50):.6f}\n")
    f.write(f"Q75: {np.percentile(upper, 75):.6f}\n")
    f.write(f"Q90: {np.percentile(upper, 90):.6f}\n")
    f.write(f"Q95: {np.percentile(upper, 95):.6f}\n")
    f.write(f"Q99: {np.percentile(upper, 99):.6f}\n")
    f.write(f"Max: {upper.max():.6f}\n\n")

    f.write("Top 20 second-order relationships:\n")

    for rank, (_, row) in enumerate(
        pairs_df.head(20).iterrows(),
        start=1
    ):
        f.write(
            f"{rank}. "
            f"{row['feature_1']} <-> "
            f"{row['feature_2']} : "
            f"{row['second_order_similarity']:.6f}\n"
        )

print("\n" + "=" * 70)
print("STAGE 4A COMPLETE")
print("=" * 70)