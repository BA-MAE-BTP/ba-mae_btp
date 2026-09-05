import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/splits/train.csv"

OUTPUT_DIR = "preprocessing/profiles"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "train_feature_profile.csv"
)

CHUNK_SIZE = 200_000

# Maximum number of sampled values stored per feature
MAX_SAMPLES = 100_000

# Samples taken from each chunk
SAMPLE_PER_CHUNK = 1_000


# ============================================================
# 39 MODEL FEATURES
# ============================================================
# The four IP/port identity/network-address fields were removed
# before splitting.
#
# Label, Attack and Dataset are metadata and are NOT model inputs.

FEATURES = [
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
]


# ============================================================
# SETUP
# ============================================================

if not os.path.exists(TRAIN_FILE):
    raise FileNotFoundError(
        f"\nTraining file not found:\n{TRAIN_FILE}\n\n"
        "Make sure the script is being run from the ba-mae project root."
    )

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# INITIALIZE STATISTICS
# ============================================================

stats = {}

samples = {
    feature: []
    for feature in FEATURES
}

rng = np.random.default_rng(42)

total_rows = 0
chunk_number = 0


# ============================================================
# READ TRAINING DATA
# ============================================================

print("=" * 70)
print("BA-MAE — TRAINING FEATURE PROFILING")
print("=" * 70)

print(f"Training file : {TRAIN_FILE}")
print(f"Chunk size    : {CHUNK_SIZE:,}")
print(f"Features      : {len(FEATURES)}")
print()

print("Reading TRAINING split only...")
print()


for chunk in pd.read_csv(
    TRAIN_FILE,
    usecols=FEATURES,
    chunksize=CHUNK_SIZE
):

    chunk_number += 1
    total_rows += len(chunk)

    if chunk_number % 10 == 0:

        print(
            f"Processed {total_rows:,} training rows..."
        )

    for feature in FEATURES:

        values = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        # ----------------------------------------------------
        # Identify finite values
        # ----------------------------------------------------

        finite_mask = np.isfinite(values)

        finite_values = values[finite_mask]

        # ----------------------------------------------------
        # Initialize statistics
        # ----------------------------------------------------

        if feature not in stats:

            stats[feature] = {
                "count": 0,
                "missing": 0,
                "nonfinite": 0,
                "negative": 0,
                "min": np.inf,
                "max": -np.inf,
            }

        s = stats[feature]

        # ----------------------------------------------------
        # Missing values
        # ----------------------------------------------------

        s["missing"] += int(
            np.isnan(values).sum()
        )

        # ----------------------------------------------------
        # Non-finite values
        # ----------------------------------------------------

        s["nonfinite"] += int(
            (~finite_mask).sum()
        )

        if len(finite_values) == 0:
            continue

        # ----------------------------------------------------
        # Negative values
        # ----------------------------------------------------

        s["negative"] += int(
            (finite_values < 0).sum()
        )

        # ----------------------------------------------------
        # Count
        # ----------------------------------------------------

        s["count"] += len(finite_values)

        # ----------------------------------------------------
        # Min / Max
        # ----------------------------------------------------

        current_min = float(
            np.min(finite_values)
        )

        current_max = float(
            np.max(finite_values)
        )

        s["min"] = min(
            s["min"],
            current_min
        )

        s["max"] = max(
            s["max"],
            current_max
        )

        # ----------------------------------------------------
        # Sampling
        # ----------------------------------------------------

        if len(finite_values) > SAMPLE_PER_CHUNK:

            indices = rng.choice(
                len(finite_values),
                size=SAMPLE_PER_CHUNK,
                replace=False
            )

            sampled = finite_values[indices]

        else:

            sampled = finite_values

        samples[feature].extend(
            sampled.tolist()
        )

        # ----------------------------------------------------
        # Keep memory bounded
        # ----------------------------------------------------

        if len(samples[feature]) > MAX_SAMPLES:

            indices = rng.choice(
                len(samples[feature]),
                size=MAX_SAMPLES,
                replace=False
            )

            samples[feature] = [
                samples[feature][i]
                for i in indices
            ]


# ============================================================
# BUILD PROFILE
# ============================================================

print()
print("=" * 70)
print("BUILDING TRAINING PROFILE")
print("=" * 70)

results = []


for feature in FEATURES:

    s = stats[feature]

    sample = np.asarray(
        samples[feature],
        dtype=np.float64
    )

    row = {
        "feature": feature,
        "count": s["count"],
        "missing": s["missing"],
        "nonfinite": s["nonfinite"],
        "negative": s["negative"],
        "min": s["min"],
        "max": s["max"],
    }

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    if len(sample) > 0:

        row["q01"] = np.percentile(sample, 1)
        row["q05"] = np.percentile(sample, 5)
        row["q25"] = np.percentile(sample, 25)
        row["q50"] = np.percentile(sample, 50)
        row["q75"] = np.percentile(sample, 75)
        row["q90"] = np.percentile(sample, 90)
        row["q95"] = np.percentile(sample, 95)
        row["q97_5"] = np.percentile(sample, 97.5)
        row["q99"] = np.percentile(sample, 99)
        row["q99_9"] = np.percentile(sample, 99.9)

        # ----------------------------------------------------
        # Mean / standard deviation
        #
        # Only calculate if values are numerically safe.
        # Astronomically large values can cause variance overflow.
        # ----------------------------------------------------

        if np.max(np.abs(sample)) < 1e12:

            row["mean"] = np.mean(sample)
            row["std"] = np.std(sample)

        else:

            row["mean"] = np.nan
            row["std"] = np.nan

    else:

        for column in [
            "q01",
            "q05",
            "q25",
            "q50",
            "q75",
            "q90",
            "q95",
            "q97_5",
            "q99",
            "q99_9",
            "mean",
            "std",
        ]:

            row[column] = np.nan

    results.append(row)


# ============================================================
# SAVE
# ============================================================

profile_df = pd.DataFrame(results)

profile_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("PROFILE COMPLETE")
print("=" * 70)

print(
    f"Training rows processed : {total_rows:,}"
)

print(
    f"Features profiled       : {len(FEATURES)}"
)

print(
    f"Output file             : {OUTPUT_FILE}"
)

print()


# ============================================================
# PROBLEM VALUES
# ============================================================

print("Missing / non-finite values:")

problem_features = profile_df[
    (profile_df["missing"] > 0) |
    (profile_df["nonfinite"] > 0)
]

if problem_features.empty:

    print("  None")

else:

    print(
        problem_features[
            [
                "feature",
                "missing",
                "nonfinite"
            ]
        ].to_string(index=False)
    )


print()

print("Negative values:")

negative_features = profile_df[
    profile_df["negative"] > 0
]

if negative_features.empty:

    print("  None")

else:

    print(
        negative_features[
            [
                "feature",
                "negative"
            ]
        ].to_string(index=False)
    )


# ============================================================
# FEATURE DISTRIBUTION TABLE
# ============================================================

print()
print("Feature distribution:")

print()

print(
    profile_df[
        [
            "feature",
            "min",
            "q25",
            "q50",
            "q75",
            "q90",
            "q95",
            "q97_5",
            "q99",
            "q99_9",
            "max"
        ]
    ].to_string(index=False)
)


print()
print("=" * 70)
print("DONE")
print("=" * 70)