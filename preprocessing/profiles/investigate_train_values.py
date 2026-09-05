import pandas as pd
import numpy as np


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/splits/train.csv"

CHUNK_SIZE = 200_000

FEATURES_TO_CHECK = [
    "FLOW_DURATION_MILLISECONDS",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "DNS_TTL_ANSWER",
]


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = {
    "FLOW_DURATION_MILLISECONDS": [
        4_000_000,
        4_200_000,
        4_290_000,
        4_294_000,
        4_294_900,
        4_294_967,
    ],

    "SRC_TO_DST_SECOND_BYTES": [
        1e6,
        1e9,
        1e12,
        1e20,
        1e100,
    ],

    "DST_TO_SRC_SECOND_BYTES": [
        1e6,
        1e9,
        1e12,
        1e20,
        1e100,
    ],

    "DNS_TTL_ANSWER": [
        1_000,
        10_000,
        100_000,
        1_000_000,
        1_000_000_000,
    ],
}


# ============================================================
# STORAGE
# ============================================================

counts = {
    feature: {
        threshold: 0
        for threshold in THRESHOLDS[feature]
    }
    for feature in FEATURES_TO_CHECK
}

total_rows = 0

largest_values = {
    feature: []
    for feature in FEATURES_TO_CHECK
}


# ============================================================
# READ TRAINING DATA
# ============================================================

print("=" * 70)
print("BA-MAE — TRAINING VALUE INVESTIGATION")
print("=" * 70)

print()
print("Reading training split...")
print()


for chunk in pd.read_csv(
    TRAIN_FILE,
    usecols=FEATURES_TO_CHECK,
    chunksize=CHUNK_SIZE
):

    total_rows += len(chunk)

    for feature in FEATURES_TO_CHECK:

        values = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        # ----------------------------------------------------
        # Threshold counts
        # ----------------------------------------------------

        for threshold in THRESHOLDS[feature]:

            counts[feature][threshold] += int(
                np.sum(values > threshold)
            )

        # ----------------------------------------------------
        # Keep largest values seen in each chunk
        # ----------------------------------------------------

        top_values = np.sort(values)[-10:]

        largest_values[feature].extend(
            top_values.tolist()
        )

        # Keep list manageable
        largest_values[feature] = sorted(
            largest_values[feature]
        )[-100:]


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("RESULTS")
print("=" * 70)

print(
    f"Training rows scanned: {total_rows:,}"
)

print()


for feature in FEATURES_TO_CHECK:

    print()
    print("-" * 70)
    print(feature)
    print("-" * 70)

    for threshold, count in counts[feature].items():

        percentage = (
            count / total_rows
        ) * 100

        print(
            f"> {threshold:,.6g} : "
            f"{count:,} rows "
            f"({percentage:.6f}%)"
        )

    print()
    print("Largest values observed:")

    top = sorted(
        largest_values[feature],
        reverse=True
    )[:20]

    for value in top:

        print(
            f"  {value:.12g}"
        )


print()
print("=" * 70)
print("DONE")
print("=" * 70)