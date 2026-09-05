import pandas as pd
import numpy as np
from collections import Counter
import os
import time

FILE = "NF-UQ-NIDS-v2.csv"
CHUNK_SIZE = 100_000

print("=" * 70)
print("NF-UQ-NIDS-v2 DATA AUDIT")
print("=" * 70)

print(f"\nFile: {FILE}")
print(f"Size: {os.path.getsize(FILE) / (1024**3):.2f} GB")

# ---------------------------------------------------------
# Read header
# ---------------------------------------------------------

header = pd.read_csv(FILE, nrows=0)
columns = list(header.columns)

print(f"\nNumber of columns: {len(columns)}")
print("\nColumns:")
for i, col in enumerate(columns, 1):
    print(f"{i:2}. {col}")

# ---------------------------------------------------------
# Counters
# ---------------------------------------------------------

total_rows = 0
missing_counts = Counter()
label_counts = Counter()
attack_counts = Counter()
dataset_counts = Counter()

duplicate_count = 0

numeric_stats = {}

start_time = time.time()

# ---------------------------------------------------------
# Process file in chunks
# ---------------------------------------------------------

for chunk_number, chunk in enumerate(
    pd.read_csv(FILE, chunksize=CHUNK_SIZE, low_memory=False)
):

    total_rows += len(chunk)

    # Missing values
    missing = chunk.isna().sum()

    for col, count in missing.items():
        missing_counts[col] += int(count)

    # Label distribution
    if "Label" in chunk.columns:
        label_counts.update(
            chunk["Label"].astype(str).value_counts().to_dict()
        )

    # Attack distribution
    if "Attack" in chunk.columns:
        attack_counts.update(
            chunk["Attack"].astype(str).value_counts().to_dict()
        )

    # Dataset distribution
    if "Dataset" in chunk.columns:
        dataset_counts.update(
            chunk["Dataset"].astype(str).value_counts().to_dict()
        )

    # Duplicate rows within this chunk
    duplicate_count += int(chunk.duplicated().sum())

    # Numerical statistics
    numeric_columns = chunk.select_dtypes(
        include=[np.number]
    ).columns

    for col in numeric_columns:

        if col not in numeric_stats:
            numeric_stats[col] = {
                "count": 0,
                "sum": 0.0,
                "sum_sq": 0.0,
                "min": np.inf,
                "max": -np.inf,
            }

        series = chunk[col].dropna().astype(float)

        if len(series) == 0:
            continue

        numeric_stats[col]["count"] += len(series)
        numeric_stats[col]["sum"] += series.sum()
        numeric_stats[col]["sum_sq"] += (series ** 2).sum()
        numeric_stats[col]["min"] = min(
            numeric_stats[col]["min"], series.min()
        )
        numeric_stats[col]["max"] = max(
            numeric_stats[col]["max"], series.max()
        )

    if chunk_number % 10 == 0:
        elapsed = time.time() - start_time
        print(
            f"Processed {total_rows:,} rows | "
            f"Elapsed: {elapsed/60:.1f} min"
        )

# ---------------------------------------------------------
# Print results
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT RESULTS")
print("=" * 70)

print(f"\nTotal rows: {total_rows:,}")
print(f"Total columns: {len(columns)}")

# ---------------------------------------------------------
# Missing values
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("MISSING VALUES")
print("-" * 70)

for col in columns:
    print(f"{col}: {missing_counts[col]:,}")

# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("LABEL DISTRIBUTION")
print("-" * 70)

for value, count in label_counts.most_common():
    percentage = count / total_rows * 100
    print(f"{value}: {count:,} ({percentage:.2f}%)")

# ---------------------------------------------------------
# Attacks
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("ATTACK DISTRIBUTION")
print("-" * 70)

for value, count in attack_counts.most_common():
    percentage = count / total_rows * 100
    print(f"{value}: {count:,} ({percentage:.2f}%)")

# ---------------------------------------------------------
# Dataset sources
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("SOURCE DATASET DISTRIBUTION")
print("-" * 70)

for value, count in dataset_counts.most_common():
    percentage = count / total_rows * 100
    print(f"{value}: {count:,} ({percentage:.2f}%)")

# ---------------------------------------------------------
# Numerical statistics
# ---------------------------------------------------------

print("\n" + "-" * 70)
print("NUMERICAL FEATURE STATISTICS")
print("-" * 70)

for col, stats in numeric_stats.items():

    count = stats["count"]

    if count == 0:
        continue

    mean = stats["sum"] / count

    variance = (
        stats["sum_sq"] / count
        - mean ** 2
    )

    variance = max(variance, 0)
    std = np.sqrt(variance)

    print(f"\n{col}")
    print(f"  Count : {count:,}")
    print(f"  Mean  : {mean:.6g}")
    print(f"  Std   : {std:.6g}")
    print(f"  Min   : {stats['min']:.6g}")
    print(f"  Max   : {stats['max']:.6g}")

# ---------------------------------------------------------

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)

print(f"Total processing time: {(time.time()-start_time)/60:.2f} minutes")