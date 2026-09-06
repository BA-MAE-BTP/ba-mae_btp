import os
import numpy as np
import pandas as pd


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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "final_dependency_matrix.csv"
)


# ============================================================
# FINAL ALPHA
# ============================================================

ALPHA = 0.35


# ============================================================
# LOAD MATRICES
# ============================================================

print("=" * 70)
print("STAGE 3D — BUILD FINAL DEPENDENCY MATRIX")
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
# VALIDATION
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
    f"Number of features: {len(features)}"
)

print(
    f"\nFinal α = {ALPHA:.2f}"
)

print(
    f"MI contribution = {ALPHA:.2f}"
)

print(
    f"Spearman contribution = "
    f"{1 - ALPHA:.2f}"
)


# ============================================================
# BUILD FINAL MATRIX
# ============================================================

final_dependency = (
    ALPHA * nmi_df.values
    +
    (1.0 - ALPHA)
    * spearman_df.values
)

# Numerical safety
final_dependency = np.clip(
    final_dependency,
    0.0,
    1.0
)

# Perfect self-dependency
np.fill_diagonal(
    final_dependency,
    1.0
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

final_df = pd.DataFrame(
    final_dependency,
    index=features,
    columns=features
)


# ============================================================
# SAVE
# ============================================================

final_df.to_csv(
    OUTPUT_FILE
)


# ============================================================
# BASIC VALIDATION
# ============================================================

print("\nValidation:")

print(
    f"Shape: {final_df.shape}"
)

print(
    f"Minimum: "
    f"{final_df.values.min():.6f}"
)

print(
    f"Maximum: "
    f"{final_df.values.max():.6f}"
)

print(
    f"Mean: "
    f"{final_df.values.mean():.6f}"
)

print(
    f"Diagonal mean: "
    f"{np.diag(final_df.values).mean():.6f}"
)

# Check symmetry
is_symmetric = np.allclose(
    final_df.values,
    final_df.values.T,
    atol=1e-10
)

print(
    f"Symmetric: {is_symmetric}"
)

# Check diagonal
diagonal_correct = np.allclose(
    np.diag(final_df.values),
    1.0
)

print(
    f"Diagonal = 1: "
    f"{diagonal_correct}"
)


# ============================================================
# TOP DEPENDENCY PAIRS
# ============================================================

print("\nTop dependency pairs:")

pairs = []

for i in range(
    len(features)
):

    for j in range(
        i + 1,
        len(features)
    ):

        pairs.append({

            "feature_1":
                features[i],

            "feature_2":
                features[j],

            "dependency":
                final_dependency[i, j]

        })


pairs_df = pd.DataFrame(
    pairs
)

pairs_df = pairs_df.sort_values(
    "dependency",
    ascending=False
)

print(
    pairs_df.head(15).to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# SAVE PAIR SCORES
# ============================================================

pairs_output = os.path.join(
    BASE_DIR,
    "final_dependency_pair_scores.csv"
)

pairs_df.to_csv(
    pairs_output,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary_file = os.path.join(
    BASE_DIR,
    "final_dependency_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL DEPENDENCY MATRIX\n"
    )

    f.write(
        "=======================\n\n"
    )

    f.write(
        f"Alpha = {ALPHA}\n"
    )

    f.write(
        f"NMI weight = {ALPHA}\n"
    )

    f.write(
        f"Spearman weight = "
        f"{1 - ALPHA}\n\n"
    )

    f.write(
        f"Number of features = "
        f"{len(features)}\n"
    )

    f.write(
        f"Matrix shape = "
        f"{final_df.shape}\n"
    )

    f.write(
        f"Minimum dependency = "
        f"{final_df.values.min()}\n"
    )

    f.write(
        f"Maximum dependency = "
        f"{final_df.values.max()}\n"
    )

    f.write(
        f"Mean dependency = "
        f"{final_df.values.mean()}\n"
    )

    f.write(
        f"Symmetric = "
        f"{is_symmetric}\n"
    )

    f.write(
        f"Diagonal = 1 = "
        f"{diagonal_correct}\n\n"
    )

    f.write(
        "Top dependency pairs:\n"
    )

    f.write(
        pairs_df.head(20).to_string(
            index=False
        )
    )


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 3D COMPLETE")
print("=" * 70)

print(
    f"\nSaved:"
)

print(
    OUTPUT_FILE
)

print(
    pairs_output
)

print(
    summary_file
)

print(
    "\nFINAL α IS NOW FROZEN AT 0.35."
)