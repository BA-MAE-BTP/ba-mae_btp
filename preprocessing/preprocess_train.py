import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler


# ============================================================
# CONFIG
# ============================================================

TRAIN_FILE = "data/splits/train.csv"

OUTPUT_DIR = "data/processed"

TRAIN_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "train_processed.csv"
)

SCALER_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "robust_scaler.pkl"
)

PROFILE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "preprocessing_profile.csv"
)

TAIL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "tail_thresholds.txt"
)

CHUNK_SIZE = 200_000


# ============================================================
# 39 MODEL FEATURES
# ============================================================

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
# FEATURES WITH LOG1P TRANSFORMATION
# ============================================================

LOG_FEATURES = [
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "DURATION_IN",
    "DURATION_OUT",
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
]


# ============================================================
# SETUP
# ============================================================

if not os.path.exists(TRAIN_FILE):

    raise FileNotFoundError(
        f"Training file not found: {TRAIN_FILE}"
    )

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PASS 1
# FIND TRAINING MEDIANS
# ============================================================

print("=" * 70)
print("BA-MAE — TRAIN-ONLY PREPROCESSING")
print("=" * 70)

print()
print("PASS 1: Finding training medians...")
print()


median_samples = {
    feature: []
    for feature in FEATURES
}

rng = np.random.default_rng(42)

rows_seen = 0


for chunk in pd.read_csv(
    TRAIN_FILE,
    usecols=FEATURES,
    chunksize=CHUNK_SIZE
):

    rows_seen += len(chunk)

    for feature in FEATURES:

        values = pd.to_numeric(
            chunk[feature],
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        finite = values[np.isfinite(values)]

        if len(finite) == 0:
            continue

        # Keep at most 2,000 representative values/chunk
        n = min(2000, len(finite))

        if len(finite) > n:

            idx = rng.choice(
                len(finite),
                size=n,
                replace=False
            )

            finite = finite[idx]

        median_samples[feature].extend(
            finite.tolist()
        )

        # Keep memory bounded
        if len(median_samples[feature]) > 100_000:

            idx = rng.choice(
                len(median_samples[feature]),
                size=100_000,
                replace=False
            )

            median_samples[feature] = [
                median_samples[feature][i]
                for i in idx
            ]


medians = {}

for feature in FEATURES:

    values = np.asarray(
        median_samples[feature],
        dtype=np.float64
    )

    medians[feature] = float(
        np.median(values)
    )


print(
    f"Rows scanned: {rows_seen:,}"
)

print("Training medians calculated.")
print()


# ============================================================
# ARTIFACT RULES
# ============================================================

# These thresholds are deliberately ONLY for clearly
# impossible/overflow-like values identified during investigation.

ARTIFACT_THRESHOLDS = {

    "SRC_TO_DST_SECOND_BYTES": 1e20,

    "DST_TO_SRC_SECOND_BYTES": 1e20,

    "DNS_TTL_ANSWER": 1e9,
}


# ============================================================
# PASS 2
# TRANSFORM + SCALE
# ============================================================

print("=" * 70)
print("PASS 2: TRANSFORMING AND SCALING")
print("=" * 70)

print()

scaler = RobustScaler()

# We fit the scaler incrementally using a bounded sample
# from the training data.

scaler_samples = []

rows_seen = 0

for chunk in pd.read_csv(
    TRAIN_FILE,
    usecols=FEATURES,
    chunksize=CHUNK_SIZE
):

    rows_seen += len(chunk)

    data = chunk.astype(np.float64)

    # --------------------------------------------------------
    # Replace invalid values
    # --------------------------------------------------------

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # Replace identified artifacts with NaN
    # --------------------------------------------------------

    for feature, threshold in ARTIFACT_THRESHOLDS.items():

        mask = data[feature] > threshold

        data.loc[
            mask,
            feature
        ] = np.nan

    # --------------------------------------------------------
    # Median imputation using TRAIN medians
    # --------------------------------------------------------

    for feature in FEATURES:

        data[feature] = data[feature].fillna(
            medians[feature]
        )

    # --------------------------------------------------------
    # log1p
    # --------------------------------------------------------

    for feature in LOG_FEATURES:

        data[feature] = np.log1p(
            data[feature]
        )

    # --------------------------------------------------------
    # Keep a bounded scaler sample
    # --------------------------------------------------------

    sample_n = min(
        2000,
        len(data)
    )

    if len(data) > sample_n:

        sample = data.sample(
            n=sample_n,
            random_state=42
        )

    else:

        sample = data

    scaler_samples.append(
        sample.to_numpy(dtype=np.float64)
    )

    if rows_seen % 2_000_000 < CHUNK_SIZE:

        print(
            f"Prepared {rows_seen:,} rows..."
        )


# Combine bounded sample

scaler_data = np.vstack(
    scaler_samples
)

print()
print(
    f"Rows used to fit scaler: "
    f"{len(scaler_data):,}"
)

print()

# ------------------------------------------------------------
# FIT SCALER — TRAIN ONLY
# ------------------------------------------------------------

scaler.fit(
    scaler_data
)


# ============================================================
# SAVE SCALER
# ============================================================

with open(
    SCALER_OUTPUT,
    "wb"
) as f:

    pickle.dump(
        scaler,
        f
    )


print(
    f"Scaler saved: {SCALER_OUTPUT}"
)


# ============================================================
# FLOW-SIZE TAIL THRESHOLDS
# ============================================================

print()
print("=" * 70)
print("CALCULATING TRAINING FLOW-SIZE TAIL")
print("=" * 70)

flow_size_samples = []

rows_seen = 0

for chunk in pd.read_csv(
    TRAIN_FILE,
    usecols=[
        "IN_BYTES",
        "OUT_BYTES"
    ],
    chunksize=CHUNK_SIZE
):

    rows_seen += len(chunk)

    flow_size = (
        chunk["IN_BYTES"].astype(np.float64)
        +
        chunk["OUT_BYTES"].astype(np.float64)
    )

    n = min(
        5000,
        len(flow_size)
    )

    if len(flow_size) > n:

        flow_size = flow_size.sample(
            n=n,
            random_state=42
        )

    flow_size_samples.extend(
        flow_size.tolist()
    )

    if len(flow_size_samples) > 500_000:

        idx = rng.choice(
            len(flow_size_samples),
            size=500_000,
            replace=False
        )

        flow_size_samples = [
            flow_size_samples[i]
            for i in idx
        ]


flow_size = np.asarray(
    flow_size_samples,
    dtype=np.float64
)

q90 = np.percentile(
    flow_size,
    90
)

q95 = np.percentile(
    flow_size,
    95
)

q975 = np.percentile(
    flow_size,
    97.5
)

q99 = np.percentile(
    flow_size,
    99
)


print()
print(f"Q90   = {q90:,.4f} bytes")
print(f"Q95   = {q95:,.4f} bytes")
print(f"Q97.5 = {q975:,.4f} bytes")
print(f"Q99   = {q99:,.4f} bytes")


# ============================================================
# SAVE TAIL THRESHOLDS
# ============================================================

with open(
    TAIL_OUTPUT,
    "w"
) as f:

    f.write(
        "Flow-size definition: IN_BYTES + OUT_BYTES\n"
    )

    f.write(
        "Quantiles computed from TRAINING data only.\n\n"
    )

    f.write(
        f"Q90={q90}\n"
    )

    f.write(
        f"Q95={q95}\n"
    )

    f.write(
        f"Q97.5={q975}\n"
    )

    f.write(
        f"Q99={q99}\n"
    )


# ============================================================
# SAVE PREPROCESSING PROFILE
# ============================================================

profile = pd.DataFrame({
    "feature": FEATURES,
    "training_median": [
        medians[f]
        for f in FEATURES
    ],
    "log1p": [
        f in LOG_FEATURES
        for f in FEATURES
    ],
    "artifact_threshold": [
        ARTIFACT_THRESHOLDS.get(
            f,
            np.nan
        )
        for f in FEATURES
    ]
})

profile.to_csv(
    PROFILE_OUTPUT,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING SETUP COMPLETE")
print("=" * 70)

print()
print("Created:")

print(
    f"  {SCALER_OUTPUT}"
)

print(
    f"  {PROFILE_OUTPUT}"
)

print(
    f"  {TAIL_OUTPUT}"
)

print()
print("IMPORTANT:")
print("No validation or test data was used.")
print("No rows were deleted.")
print("Legitimate tail values were retained.")
print()
print("=" * 70)