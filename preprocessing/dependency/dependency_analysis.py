import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# BA-MAE — Dependency Discovery
# MI + Spearman → Unified Dependency Matrix
# ============================================================

# -----------------------------
# Paths
# -----------------------------
TRAIN_PATH = "data/splits/train.csv"
OUTPUT_DIR = "preprocessing/dependency"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Configuration
# -----------------------------
CHUNK_SIZE = 200_000
SAMPLE_SIZE = 100_000
SAMPLE_PER_CHUNK = 1_000
RANDOM_STATE = 42

# Maximum number of bins for empirical MI
MAX_BINS = 32

# Weighting in unified dependency score
ALPHA = 0.5       # MI weight
BETA = 0.5        # Spearman weight


# ============================================================
# 39 MODEL FEATURES
# ============================================================

MODEL_FEATURES = [
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
# FEATURES WITH KNOWN EXTREME ARTIFACTS
# ============================================================

ARTIFACT_THRESHOLDS = {
    "SRC_TO_DST_SECOND_BYTES": 1e20,
    "DST_TO_SRC_SECOND_BYTES": 1e20,
    "DNS_TTL_ANSWER": 1e9,
}


# ============================================================
# FEATURES TO LOG-TRANSFORM
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
# STEP 1 — SAMPLE TRAINING DATA
# ============================================================

def load_training_sample():

    print("\n" + "=" * 70)
    print("STEP 1 — LOADING TRAINING SAMPLE")
    print("=" * 70)

    samples = []

    total_rows = 0

    reader = pd.read_csv(
        TRAIN_PATH,
        usecols=MODEL_FEATURES,
        chunksize=CHUNK_SIZE,
        low_memory=False
    )

    for chunk_number, chunk in enumerate(reader, start=1):

        total_rows += len(chunk)

        # Random sample from every chunk
        n = min(SAMPLE_PER_CHUNK, len(chunk))

        sampled = chunk.sample(
            n=n,
            random_state=RANDOM_STATE + chunk_number
        )

        samples.append(sampled)

        if chunk_number % 25 == 0:
            print(
                f"Processed ~{total_rows:,} rows | "
                f"chunks: {chunk_number}"
            )

    df = pd.concat(samples, ignore_index=True)

    # Final random downsample
    if len(df) > SAMPLE_SIZE:
        df = df.sample(
            n=SAMPLE_SIZE,
            random_state=RANDOM_STATE
        ).reset_index(drop=True)

    print(f"\nTotal training rows scanned: {total_rows:,}")
    print(f"Dependency-analysis sample: {len(df):,}")
    print(f"Number of features: {len(MODEL_FEATURES)}")

    return df


# ============================================================
# STEP 2 — CLEAN SAMPLE
# ============================================================

def clean_sample(df):

    print("\n" + "=" * 70)
    print("STEP 2 — CLEANING DEPENDENCY SAMPLE")
    print("=" * 70)

    df = df.copy()

    # Convert everything to numeric
    for col in MODEL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace known extreme artifacts with NaN
    for col, threshold in ARTIFACT_THRESHOLDS.items():

        mask = df[col] > threshold

        count = mask.sum()

        if count > 0:
            print(
                f"{col}: replacing {count:,} "
                f"extreme artifact values"
            )

            df.loc[mask, col] = np.nan

    # Replace infinities
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Median imputation using the dependency sample
    for col in MODEL_FEATURES:

        median_value = df[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        df[col] = df[col].fillna(median_value)

    # Log transform heavy-tailed variables
    for col in LOG_FEATURES:

        # Ensure non-negative before log1p
        df[col] = df[col].clip(lower=0)

        df[col] = np.log1p(df[col])

    print("\nCleaning complete.")

    return df


# ============================================================
# STEP 3 — SPEARMAN CORRELATION
# ============================================================

def calculate_spearman(df):

    print("\n" + "=" * 70)
    print("STEP 3 — CALCULATING SPEARMAN CORRELATION")
    print("=" * 70)

    spearman = df[MODEL_FEATURES].corr(method="spearman")

    output_path = os.path.join(
        OUTPUT_DIR,
        "spearman_matrix.csv"
    )

    spearman.to_csv(output_path)

    print(f"Saved: {output_path}")

    return spearman


# ============================================================
# STEP 4 — DISCRETIZATION FOR EMPIRICAL MI
# ============================================================

def discretize_feature(series):

    series = series.to_numpy()

    unique_values = np.unique(series)

    # If feature has relatively few unique values,
    # preserve its natural discrete structure.
    if len(unique_values) <= MAX_BINS:

        _, encoded = np.unique(
            series,
            return_inverse=True
        )

        return encoded

    # Otherwise use quantile bins.
    try:

        binned = pd.qcut(
            series,
            q=MAX_BINS,
            labels=False,
            duplicates="drop"
        )

        return np.asarray(binned)

    except Exception:

        # Fallback to equal-width bins
        binned = pd.cut(
            series,
            bins=MAX_BINS,
            labels=False,
            duplicates="drop"
        )

        return np.asarray(binned)


# ============================================================
# STEP 5 — EMPIRICAL MUTUAL INFORMATION
# ============================================================

def entropy(values):

    values = np.asarray(values)

    _, counts = np.unique(
        values,
        return_counts=True
    )

    probabilities = counts / counts.sum()

    return -np.sum(
        probabilities *
        np.log(probabilities + 1e-12)
    )


def mutual_information(x, y):

    x = np.asarray(x)
    y = np.asarray(y)

    # Joint distribution
    joint = pd.crosstab(
        pd.Series(x),
        pd.Series(y),
        normalize=True
    ).to_numpy()

    # Marginals
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)

    mi = 0.0

    for i in range(len(px)):

        for j in range(len(py)):

            if joint[i, j] > 0:

                mi += joint[i, j] * np.log(
                    joint[i, j] /
                    (px[i] * py[j])
                )

    return mi


# ============================================================
# STEP 6 — NORMALIZED MI
# ============================================================

def calculate_mi(df):

    print("\n" + "=" * 70)
    print("STEP 4 — CALCULATING MUTUAL INFORMATION")
    print("=" * 70)

    n_features = len(MODEL_FEATURES)

    # Discretize every feature once
    print("Discretizing features...")

    discretized = {}

    for col in MODEL_FEATURES:

        discretized[col] = discretize_feature(
            df[col]
        )

    mi_matrix = np.zeros(
        (n_features, n_features)
    )

    nmi_matrix = np.zeros(
        (n_features, n_features)
    )

    # Entropy of each feature
    entropies = {}

    for col in MODEL_FEATURES:

        entropies[col] = entropy(
            discretized[col]
        )

    total_pairs = (
        n_features * (n_features - 1) // 2
    )

    pair_number = 0

    for i in range(n_features):

        for j in range(i + 1, n_features):

            feature_i = MODEL_FEATURES[i]
            feature_j = MODEL_FEATURES[j]

            mi = mutual_information(
                discretized[feature_i],
                discretized[feature_j]
            )

            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi

            # Symmetric NMI
            denominator = np.sqrt(
                entropies[feature_i] *
                entropies[feature_j]
            )

            if denominator > 0:

                nmi = mi / denominator

            else:

                nmi = 0.0

            # Numerical safety
            nmi = min(max(nmi, 0.0), 1.0)

            nmi_matrix[i, j] = nmi
            nmi_matrix[j, i] = nmi

            pair_number += 1

            if pair_number % 100 == 0:

                print(
                    f"MI pairs completed: "
                    f"{pair_number}/{total_pairs}"
                )

    # Diagonal
    for i in range(n_features):

        mi_matrix[i, i] = entropies[
            MODEL_FEATURES[i]
        ]

        nmi_matrix[i, i] = 1.0

    mi_df = pd.DataFrame(
        mi_matrix,
        index=MODEL_FEATURES,
        columns=MODEL_FEATURES
    )

    nmi_df = pd.DataFrame(
        nmi_matrix,
        index=MODEL_FEATURES,
        columns=MODEL_FEATURES
    )

    mi_path = os.path.join(
        OUTPUT_DIR,
        "mi_matrix.csv"
    )

    nmi_path = os.path.join(
        OUTPUT_DIR,
        "normalized_mi_matrix.csv"
    )

    mi_df.to_csv(mi_path)
    nmi_df.to_csv(nmi_path)

    print(f"Saved: {mi_path}")
    print(f"Saved: {nmi_path}")

    return mi_df, nmi_df


# ============================================================
# STEP 7 — UNIFIED DEPENDENCY SCORE
# ============================================================

def calculate_unified_dependency(
    nmi,
    spearman
):

    print("\n" + "=" * 70)
    print("STEP 5 — BUILDING UNIFIED DEPENDENCY MATRIX")
    print("=" * 70)

    # Absolute Spearman because both positive and
    # negative monotonic relationships represent dependency.
    abs_spearman = spearman.abs()

    unified = (
        ALPHA * nmi +
        BETA * abs_spearman
    ).copy()

    # Force diagonal to 1
    for i in range(len(unified)):
        unified.iloc[i, i] = 1.0
    unified_path = os.path.join(
        OUTPUT_DIR,
        "unified_dependency_matrix.csv"
    )

    unified.to_csv(unified_path)

    print(
        f"Saved: {unified_path}"
    )

    return unified


# ============================================================
# STEP 8 — SUMMARY STATISTICS
# ============================================================

def generate_summary(
    nmi,
    spearman,
    unified
):

    print("\n" + "=" * 70)
    print("STEP 6 — DEPENDENCY SUMMARY")
    print("=" * 70)

    pairs = []

    for i in range(len(MODEL_FEATURES)):

        for j in range(i + 1, len(MODEL_FEATURES)):

            f1 = MODEL_FEATURES[i]
            f2 = MODEL_FEATURES[j]

            pairs.append({
                "Feature_1": f1,
                "Feature_2": f2,
                "NMI": nmi.loc[f1, f2],
                "Spearman": spearman.loc[f1, f2],
                "Abs_Spearman": abs(
                    spearman.loc[f1, f2]
                ),
                "Unified_Dependency": unified.loc[
                    f1, f2
                ]
            })

    pairs_df = pd.DataFrame(pairs)

    pairs_df.sort_values(
        "Unified_Dependency",
        ascending=False,
        inplace=True
    )

    pairs_path = os.path.join(
        OUTPUT_DIR,
        "dependency_pair_scores.csv"
    )

    pairs_df.to_csv(
        pairs_path,
        index=False
    )

    print(f"\nSaved: {pairs_path}")

    scores = pairs_df[
        "Unified_Dependency"
    ]

    print("\nUnified dependency distribution:")

    for q in [0.50, 0.75, 0.90, 0.95, 0.99]:

        print(
            f"Q{int(q * 100):02d}: "
            f"{scores.quantile(q):.4f}"
        )

    print("\nNumber of feature pairs above threshold:")

    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:

        count = (
            scores >= threshold
        ).sum()

        print(
            f">= {threshold:.1f}: "
            f"{count} pairs"
        )

    print("\nTop 20 dependent feature pairs:\n")

    print(
        pairs_df.head(20).to_string(
            index=False
        )
    )

    # Save text summary
    summary_path = os.path.join(
        OUTPUT_DIR,
        "dependency_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BA-MAE DEPENDENCY DISCOVERY SUMMARY\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Training sample size: "
            f"{SAMPLE_SIZE:,}\n"
        )

        f.write(
            f"Number of model features: "
            f"{len(MODEL_FEATURES)}\n"
        )

        f.write(
            f"MI weight (alpha): {ALPHA}\n"
        )

        f.write(
            f"Spearman weight (beta): {BETA}\n\n"
        )

        f.write(
            "Unified Dependency Distribution\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        for q in [
            0.50,
            0.75,
            0.90,
            0.95,
            0.99
        ]:

            f.write(
                f"Q{int(q * 100):02d}: "
                f"{scores.quantile(q):.6f}\n"
            )

        f.write("\nThreshold Counts\n")
        f.write("-" * 40 + "\n")

        for threshold in [
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8
        ]:

            count = (
                scores >= threshold
            ).sum()

            f.write(
                f">= {threshold:.1f}: "
                f"{count} pairs\n"
            )

        f.write(
            "\nTop 20 Feature Pairs\n"
        )

        f.write(
            "-" * 40 + "\n"
        )

        f.write(
            pairs_df.head(20).to_string(
                index=False
            )
        )

    print(
        f"\nSaved: {summary_path}"
    )

    return pairs_df


# ============================================================
# STEP 9 — HEATMAP
# ============================================================

def save_heatmap(
    matrix,
    title,
    filename,
    vmin=None,
    vmax=None
):

    plt.figure(
        figsize=(16, 14)
    )

    plt.imshow(
        matrix.values,
        aspect="auto",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )

    plt.colorbar()

    plt.xticks(
        range(len(MODEL_FEATURES)),
        MODEL_FEATURES,
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        range(len(MODEL_FEATURES)),
        MODEL_FEATURES,
        fontsize=7
    )

    plt.title(title)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BA-MAE DEPENDENCY DISCOVERY")
    print("MI + SPEARMAN → UNIFIED DEPENDENCY")
    print("=" * 70)

    # 1. Load training-only sample
    df = load_training_sample()

    # 2. Clean
    df = clean_sample(df)

    # 3. Spearman
    spearman = calculate_spearman(df)

    # 4. MI + NMI
    mi, nmi = calculate_mi(df)

    # 5. Unified dependency
    unified = calculate_unified_dependency(
        nmi,
        spearman
    )

    # 6. Summary
    generate_summary(
        nmi,
        spearman,
        unified
    )

    # 7. Heatmaps
    print("\n" + "=" * 70)
    print("STEP 7 — GENERATING HEATMAPS")
    print("=" * 70)

    save_heatmap(
        nmi,
        "Normalized Mutual Information",
        "mi_heatmap.png",
        vmin=0,
        vmax=1
    )

    save_heatmap(
        spearman,
        "Spearman Correlation",
        "spearman_heatmap.png",
        vmin=-1,
        vmax=1
    )

    save_heatmap(
        unified,
        "Unified Feature Dependency",
        "unified_dependency_heatmap.png",
        vmin=0,
        vmax=1
    )

    print("\n" + "=" * 70)
    print("DEPENDENCY ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nGenerated files:")
    print("  - mi_matrix.csv")
    print("  - normalized_mi_matrix.csv")
    print("  - spearman_matrix.csv")
    print("  - unified_dependency_matrix.csv")
    print("  - dependency_pair_scores.csv")
    print("  - dependency_summary.txt")
    print("  - mi_heatmap.png")
    print("  - spearman_heatmap.png")
    print("  - unified_dependency_heatmap.png")

    print("\nIMPORTANT:")
    print("No validation or test data was used.")
    print("Dependency discovery uses TRAIN ONLY.")


if __name__ == "__main__":
    main()