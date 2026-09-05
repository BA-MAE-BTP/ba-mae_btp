import pandas as pd
import os

# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/splits/train.csv"
VAL_FILE = "data/splits/validation.csv"
TEST_FILE = "data/splits/test.csv"

CHUNK_SIZE = 200_000

FILES = {
    "Train": TRAIN_FILE,
    "Validation": VAL_FILE,
    "Test": TEST_FILE
}

# ============================================================
# FUNCTION: ANALYZE ONE SPLIT
# ============================================================

def analyze_split(name, filepath):

    print("\n" + "=" * 70)
    print(f"{name.upper()} SPLIT")
    print("=" * 70)

    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None

    total_rows = 0

    attack_counts = {}
    dataset_counts = {}
    label_counts = {}

    column_names = None

    for chunk in pd.read_csv(
        filepath,
        chunksize=CHUNK_SIZE
    ):

        if column_names is None:
            column_names = list(chunk.columns)

        total_rows += len(chunk)

        # Attack distribution
        for attack, count in chunk["Attack"].value_counts().items():
            attack_counts[attack] = (
                attack_counts.get(attack, 0) + count
            )

        # Dataset distribution
        for dataset, count in chunk["Dataset"].value_counts().items():
            dataset_counts[dataset] = (
                dataset_counts.get(dataset, 0) + count
            )

        # Label distribution
        for label, count in chunk["Label"].value_counts().items():
            label_counts[label] = (
                label_counts.get(label, 0) + count
            )

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Total columns: {len(column_names)}")

    print("\nColumns:")
    print(column_names)

    print("\nLabel distribution:")
    for label, count in sorted(label_counts.items()):
        print(
            f"  {label}: {count:,} "
            f"({count / total_rows * 100:.4f}%)"
        )

    print("\nAttack distribution:")
    for attack, count in sorted(
        attack_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"  {attack:30s} "
            f"{count:12,} "
            f"({count / total_rows * 100:7.4f}%)"
        )

    print("\nSource Dataset distribution:")
    for dataset, count in sorted(
        dataset_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"  {dataset:30s} "
            f"{count:12,} "
            f"({count / total_rows * 100:7.4f}%)"
        )

    return {
        "rows": total_rows,
        "attacks": attack_counts,
        "datasets": dataset_counts,
        "labels": label_counts,
        "columns": column_names
    }


# ============================================================
# ANALYZE ALL THREE
# ============================================================

results = {}

for name, filepath in FILES.items():
    results[name] = analyze_split(name, filepath)


# ============================================================
# COMPARE ATTACK DISTRIBUTIONS
# ============================================================

print("\n\n" + "=" * 70)
print("ATTACK DISTRIBUTION COMPARISON")
print("=" * 70)

all_attacks = sorted(
    set(results["Train"]["attacks"])
    | set(results["Validation"]["attacks"])
    | set(results["Test"]["attacks"])
)

print(
    f"\n{'Attack':30s}"
    f"{'Train %':>12s}"
    f"{'Val %':>12s}"
    f"{'Test %':>12s}"
)

print("-" * 70)

for attack in all_attacks:

    percentages = []

    for split in ["Train", "Validation", "Test"]:

        count = results[split]["attacks"].get(attack, 0)
        total = results[split]["rows"]

        percentage = count / total * 100
        percentages.append(percentage)

    print(
        f"{attack:30s}"
        f"{percentages[0]:11.4f}"
        f"{percentages[1]:11.4f}"
        f"{percentages[2]:11.4f}"
    )


# ============================================================
# CHECK FEATURE COUNT
# ============================================================

print("\n\n" + "=" * 70)
print("FEATURE CHECK")
print("=" * 70)

columns = results["Train"]["columns"]

metadata = ["Label", "Attack", "Dataset"]

feature_columns = [
    c for c in columns
    if c not in metadata
]

print(f"\nTotal columns: {len(columns)}")
print(f"Model features: {len(feature_columns)}")
print(f"Metadata columns: {len(metadata)}")

print("\nModel features:")

for i, feature in enumerate(feature_columns, 1):
    print(f"{i:2d}. {feature}")

if len(feature_columns) == 39:
    print("\n✓ CORRECT: 39 usable model features")
else:
    print(
        f"\n⚠ WARNING: Expected 39 features, "
        f"found {len(feature_columns)}"
    )

print("\nVerification complete.")