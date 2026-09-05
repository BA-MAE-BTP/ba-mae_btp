import pandas as pd
import numpy as np


TRAIN_FILE = "data/splits/train.csv"
CHUNK_SIZE = 200_000


# Features needed to understand the suspicious values
FEATURES = [
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "DNS_TTL_ANSWER",
    "Dataset",
    "Label",
    "Attack",
]


TARGETS = {
    "SRC_TO_DST_SECOND_BYTES": 1e20,
    "DST_TO_SRC_SECOND_BYTES": 1e20,
    "DNS_TTL_ANSWER": 1e9,
}


examples = {
    feature: []
    for feature in TARGETS
}


duration_values = []


print("=" * 70)
print("BA-MAE — DEEP EXTREME VALUE CHECK")
print("=" * 70)

print("\nScanning training data...\n")


for chunk in pd.read_csv(
    TRAIN_FILE,
    usecols=FEATURES,
    chunksize=CHUNK_SIZE
):

    # --------------------------------------------------------
    # Check suspicious values
    # --------------------------------------------------------

    for feature, threshold in TARGETS.items():

        values = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        )

        mask = values > threshold

        if mask.any():

            selected = chunk.loc[
                mask,
                FEATURES
            ].copy()

            examples[feature].append(selected)

    # --------------------------------------------------------
    # Duration distribution
    # --------------------------------------------------------

    duration = pd.to_numeric(
        chunk["FLOW_DURATION_MILLISECONDS"],
        errors="coerce"
    )

    duration_values.extend(
        duration.sample(
            min(1000, len(duration)),
            random_state=42
        ).tolist()
    )


# ============================================================
# PRINT EXTREME EXAMPLES
# ============================================================

for feature in TARGETS:

    print("\n")
    print("=" * 70)
    print(f"EXTREME VALUES: {feature}")
    print("=" * 70)

    if not examples[feature]:

        print("No examples found.")
        continue

    combined = pd.concat(
        examples[feature],
        ignore_index=True
    )

    combined = combined.sort_values(
        feature,
        ascending=False
    )

    print(
        combined.head(15).to_string(
            index=False
        )
    )


# ============================================================
# DURATION CHECK
# ============================================================

print("\n")
print("=" * 70)
print("FLOW DURATION CHECK")
print("=" * 70)

duration_values = np.asarray(
    duration_values,
    dtype=np.float64
)

print(
    f"Sample size: {len(duration_values):,}"
)

for threshold in [
    1000,
    10_000,
    100_000,
    1_000_000,
    2_000_000,
    4_000_000,
    4_200_000,
    4_290_000,
    4_294_000,
    4_294_900,
    4_294_967,
]:

    count = np.sum(
        duration_values > threshold
    )

    percentage = (
        count / len(duration_values)
    ) * 100

    print(
        f"> {threshold:,} ms : "
        f"{count:,} "
        f"({percentage:.3f}%)"
    )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)