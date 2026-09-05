import pandas as pd
import numpy as np

FILE = "NF-UQ-NIDS-v2.csv"
CHUNK_SIZE = 200_000

FEATURES = [
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "DNS_TTL_ANSWER",
    "DNS_QUERY_TYPE",
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
]

# Track important thresholds
THRESHOLDS = {
    "SRC_TO_DST_SECOND_BYTES": [1e6, 1e9, 1e12, 1e20, 1e100],
    "DST_TO_SRC_SECOND_BYTES": [1e6, 1e9, 1e12, 1e20, 1e100],
    "DNS_TTL_ANSWER": [1e3, 1e4, 1e5, 1e6, 1e9],
    "DNS_QUERY_TYPE": [100, 1000, 10000, 50000],
    "IN_BYTES": [1e4, 1e6, 1e8],
    "OUT_BYTES": [1e4, 1e6, 1e8],
    "IN_PKTS": [100, 1000, 10000, 100000],
    "OUT_PKTS": [100, 1000, 10000, 100000],
}

counts = {f: {t: 0 for t in THRESHOLDS[f]} for f in FEATURES}
totals = {f: 0 for f in FEATURES}

# Keep a few extreme examples
examples = {f: [] for f in FEATURES}

for chunk in pd.read_csv(FILE, chunksize=CHUNK_SIZE):

    for f in FEATURES:
        x = pd.to_numeric(chunk[f], errors="coerce").dropna()

        totals[f] += len(x)

        for t in THRESHOLDS[f]:
            counts[f][t] += (x > t).sum()

        # Store top 5 values from each chunk
        top = x.nlargest(5).tolist()
        examples[f].extend(top)

for f in FEATURES:

    examples[f] = sorted(examples[f], reverse=True)[:10]

    print("\n" + "=" * 70)
    print(f)

    print(f"Total values: {totals[f]:,}")

    for t, count in counts[f].items():
        percentage = 100 * count / totals[f]
        print(
            f"> {t:.2e}: "
            f"{count:,} "
            f"({percentage:.6f}%)"
        )

    print("Largest observed values:")
    for value in examples[f]:
        print(f"  {value:.10e}")