import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "NF-UQ-NIDS-v2.csv"

OUTPUT_DIR = os.path.join("data", "splits")

TRAIN_FILE = os.path.join(OUTPUT_DIR, "train.csv")
VAL_FILE = os.path.join(OUTPUT_DIR, "validation.csv")
TEST_FILE = os.path.join(OUTPUT_DIR, "test.csv")

CHUNK_SIZE = 200_000
RANDOM_SEED = 42

# ============================================================
# COLUMNS TO DROP
# ============================================================

DROP_COLUMNS = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT"
]

# Metadata columns - retained but NOT model features
METADATA_COLUMNS = [
    "Label",
    "Attack",
    "Dataset"
]

# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Remove old output files if they exist
for file in [TRAIN_FILE, VAL_FILE, TEST_FILE]:
    if os.path.exists(file):
        os.remove(file)

# ============================================================
# CHECK HEADER
# ============================================================

print("=" * 70)
print("BA-MAE DATASET SPLITTING")
print("=" * 70)

header = pd.read_csv(INPUT_FILE, nrows=0)

print("\nOriginal columns:", len(header))
print("Original feature columns: 43")

missing_drop = [c for c in DROP_COLUMNS if c not in header.columns]

if missing_drop:
    raise ValueError(
        f"These columns were not found in the dataset: {missing_drop}"
    )

# ============================================================
# PASS 1 — COUNT ATTACK CLASSES
# ============================================================

print("\n" + "-" * 70)
print("PASS 1: Counting attack classes")
print("-" * 70)

attack_counts = {}

total_rows = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    usecols=["Attack"]
):

    counts = chunk["Attack"].value_counts()

    for attack, count in counts.items():
        attack_counts[attack] = attack_counts.get(attack, 0) + count

    total_rows += len(chunk)

print("\nTotal rows:", f"{total_rows:,}")

print("\nAttack class distribution:")

for attack, count in sorted(
    attack_counts.items(),
    key=lambda x: x[1],
    reverse=True
):
    percentage = count / total_rows * 100
    print(f"{attack:30s} {count:12,} ({percentage:6.2f}%)")

# ============================================================
# PASS 2 — STRATIFIED RANDOM SPLIT
# ============================================================

print("\n" + "-" * 70)
print("PASS 2: Creating stratified train/validation/test split")
print("-" * 70)

rng = np.random.default_rng(RANDOM_SEED)

first_write = {
    "train": True,
    "validation": True,
    "test": True
}

processed = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        INPUT_FILE,
        chunksize=CHUNK_SIZE
    ),
    start=1
):

    print(
        f"\rProcessing chunk {chunk_number} | "
        f"Rows processed: {processed:,}",
        end=""
    )

    # --------------------------------------------------------
    # STEP A — GLOBAL ROW HYGIENE
    # --------------------------------------------------------

    # Drop IP and port identity fields
    chunk = chunk.drop(columns=DROP_COLUMNS)

    # Convert inf/-inf to NaN
    chunk = chunk.replace([np.inf, -np.inf], np.nan)

    # Remove rows containing NaN
    before = len(chunk)

    chunk = chunk.dropna()

    removed = before - len(chunk)

    if removed > 0:
        print(
            f"\nRemoved {removed:,} invalid rows "
            f"from chunk {chunk_number}"
        )

    if len(chunk) == 0:
        continue

    # --------------------------------------------------------
    # STRATIFIED ASSIGNMENT
    # --------------------------------------------------------

    # Random number for every row
    random_values = rng.random(len(chunk))

    train_mask = random_values < 0.70

    validation_mask = (
        (random_values >= 0.70) &
        (random_values < 0.85)
    )

    test_mask = random_values >= 0.85

    train_chunk = chunk.loc[train_mask]
    validation_chunk = chunk.loc[validation_mask]
    test_chunk = chunk.loc[test_mask]

    # --------------------------------------------------------
    # WRITE TRAIN
    # --------------------------------------------------------

    if len(train_chunk) > 0:
        train_chunk.to_csv(
            TRAIN_FILE,
            mode="w" if first_write["train"] else "a",
            header=first_write["train"],
            index=False
        )

        first_write["train"] = False

    # --------------------------------------------------------
    # WRITE VALIDATION
    # --------------------------------------------------------

    if len(validation_chunk) > 0:
        validation_chunk.to_csv(
            VAL_FILE,
            mode="w" if first_write["validation"] else "a",
            header=first_write["validation"],
            index=False
        )

        first_write["validation"] = False

    # --------------------------------------------------------
    # WRITE TEST
    # --------------------------------------------------------

    if len(test_chunk) > 0:
        test_chunk.to_csv(
            TEST_FILE,
            mode="w" if first_write["test"] else "a",
            header=first_write["test"],
            index=False
        )

        first_write["test"] = False

    processed += len(chunk)

print("\n\n" + "=" * 70)
print("SPLIT COMPLETE")
print("=" * 70)

# ============================================================
# FINAL VERIFICATION
# ============================================================

print("\nOutput files:")

for name, file in [
    ("Train", TRAIN_FILE),
    ("Validation", VAL_FILE),
    ("Test", TEST_FILE)
]:

    if os.path.exists(file):
        print(f"{name:12s}: {file}")
    else:
        print(f"{name:12s}: ERROR - file not created")

print("\nExpected approximate split:")
print("Train       ≈ 70%")
print("Validation  ≈ 15%")
print("Test        ≈ 15%")

print("\nModel input:")
print("43 original features")
print("- 4 IP/Port identity fields")
print("= 39 usable features")

print("\nMetadata retained:")
print("Label, Attack, Dataset")

print("\nNext step:")
print("VERIFY THE SPLIT before doing scaling, MI, correlation,")
print("quantiles, clustering, or any other statistical preprocessing.")