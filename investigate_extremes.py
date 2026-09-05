import pandas as pd
import numpy as np

FILE = "NF-UQ-NIDS-v2.csv"
CHUNK_SIZE = 200_000

# Values that deserve investigation
CONDITIONS = {
    "SRC_TO_DST_SECOND_BYTES": 1e100,
    "DST_TO_SRC_SECOND_BYTES": 1e100,
    "DNS_TTL_ANSWER": 1e9,
}

# Columns needed to understand what is happening
SHOW_COLS = [
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_SRC_PORT",
    "L4_DST_PORT",
    "PROTOCOL",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "DNS_TTL_ANSWER",
    "Label",
    "Attack",
    "Dataset",
]

counts = {col: 0 for col in CONDITIONS}
dataset_counts = {
    col: {}
    for col in CONDITIONS
}

examples = {
    col: []
    for col in CONDITIONS
}

for chunk in pd.read_csv(
    FILE,
    chunksize=CHUNK_SIZE,
    usecols=SHOW_COLS
):

    for col, threshold in CONDITIONS.items():

        x = pd.to_numeric(
            chunk[col],
            errors="coerce"
        )

        mask = x > threshold

        if mask.any():

            selected = chunk.loc[mask].copy()

            counts[col] += len(selected)

            # Count source datasets
            for dataset, n in selected["Dataset"].value_counts().items():
                dataset_counts[col][dataset] = (
                    dataset_counts[col].get(dataset, 0) + int(n)
                )

            # Keep only first few examples
            if len(examples[col]) < 10:
                examples[col].extend(
                    selected.head(10).to_dict("records")
                )

print("\n" + "=" * 80)
print("EXTREME VALUE INVESTIGATION")
print("=" * 80)

for col in CONDITIONS:

    print("\n" + "-" * 80)
    print(col)
    print("-" * 80)

    print("Total suspicious values:", counts[col])

    print("\nBy source Dataset:")

    for dataset, count in sorted(
        dataset_counts[col].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {dataset}: {count:,}")

    print("\nExample rows:")

    for i, row in enumerate(examples[col][:10], 1):

        print(f"\nExample {i}:")

        for field in SHOW_COLS:
            print(f"  {field}: {row[field]}")