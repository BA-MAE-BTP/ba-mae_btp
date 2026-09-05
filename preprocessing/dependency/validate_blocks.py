import os
import numpy as np
import pandas as pd


# ============================================================
# BA-MAE — FEATURE BLOCK VALIDATION
#
# Validates candidate hierarchical clusters using:
# 1. Within-block dependency
# 2. Between-block dependency
# 3. Internal pairwise consistency
# 4. Weakest relationships inside each block
#
# No raw dataset is scanned here.
# ============================================================

DEPENDENCY_PATH = (
    "preprocessing/dependency/"
    "unified_dependency_matrix.csv"
)

BLOCK_PATH = (
    "preprocessing/dependency/"
    "candidate_feature_blocks.csv"
)

OUTPUT_DIR = "preprocessing/dependency"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("STEP 1 — LOADING DEPENDENCY MATRIX AND CANDIDATE BLOCKS")
    print("=" * 70)

    dependency = pd.read_csv(
        DEPENDENCY_PATH,
        index_col=0
    )

    blocks_df = pd.read_csv(
        BLOCK_PATH
    )

    print(
        f"Dependency matrix: {dependency.shape}"
    )

    print(
        f"Candidate blocks: {len(blocks_df)}"
    )

    return dependency, blocks_df


# ============================================================
# CONVERT BLOCK TABLE INTO DICTIONARY
# ============================================================

def get_blocks(blocks_df):

    blocks = {}

    for _, row in blocks_df.iterrows():

        block_id = int(row["Block_ID"])

        features = [
            x.strip()
            for x in row["Features"].split("|")
        ]

        blocks[block_id] = features

    return blocks


# ============================================================
# WITHIN-BLOCK ANALYSIS
# ============================================================

def analyze_within_blocks(
    dependency,
    blocks
):

    print("\n" + "=" * 70)
    print("STEP 2 — WITHIN-BLOCK DEPENDENCY")
    print("=" * 70)

    rows = []

    for block_id, features in blocks.items():

        pair_values = []

        pair_names = []

        for i in range(len(features)):

            for j in range(
                i + 1,
                len(features)
            ):

                f1 = features[i]
                f2 = features[j]

                value = dependency.loc[
                    f1,
                    f2
                ]

                pair_values.append(value)

                pair_names.append(
                    (f1, f2)
                )

        if len(pair_values) == 0:

            mean_dep = np.nan
            min_dep = np.nan
            max_dep = np.nan
            std_dep = np.nan
            weak_pair = "SINGLETON"

        else:

            mean_dep = np.mean(
                pair_values
            )

            min_dep = np.min(
                pair_values
            )

            max_dep = np.max(
                pair_values
            )

            std_dep = np.std(
                pair_values
            )

            weakest_index = np.argmin(
                pair_values
            )

            weak_pair = (
                f"{pair_names[weakest_index][0]} "
                f"<-> "
                f"{pair_names[weakest_index][1]}"
            )

        rows.append({

            "Block_ID": block_id,

            "Block_Size": len(features),

            "Mean_Within_Dependency":
                mean_dep,

            "Min_Within_Dependency":
                min_dep,

            "Max_Within_Dependency":
                max_dep,

            "Std_Within_Dependency":
                std_dep,

            "Weakest_Internal_Pair":
                weak_pair,

            "Features":
                " | ".join(features)
        })

    result = pd.DataFrame(rows)

    return result


# ============================================================
# BETWEEN-BLOCK ANALYSIS
# ============================================================

def analyze_between_blocks(
    dependency,
    blocks
):

    print("\n" + "=" * 70)
    print("STEP 3 — BETWEEN-BLOCK DEPENDENCY")
    print("=" * 70)

    block_ids = sorted(
        blocks.keys()
    )

    pair_values = []

    rows = []

    for i in range(len(block_ids)):

        for j in range(
            i + 1,
            len(block_ids)
        ):

            block_a = block_ids[i]
            block_b = block_ids[j]

            values = []

            for f1 in blocks[block_a]:

                for f2 in blocks[block_b]:

                    values.append(
                        dependency.loc[
                            f1,
                            f2
                        ]
                    )

            mean_value = np.mean(
                values
            )

            rows.append({

                "Block_A":
                    block_a,

                "Block_B":
                    block_b,

                "Mean_Between_Dependency":
                    mean_value,

                "Max_Between_Dependency":
                    np.max(values),

                "Min_Between_Dependency":
                    np.min(values),

                "Num_Cross_Pairs":
                    len(values)
            })

            pair_values.extend(
                values
            )

    result = pd.DataFrame(rows)

    overall_mean = np.mean(
        pair_values
    )

    overall_max = np.max(
        pair_values
    )

    print(
        f"Overall mean between-block dependency: "
        f"{overall_mean:.4f}"
    )

    print(
        f"Maximum cross-block dependency: "
        f"{overall_max:.4f}"
    )

    return result, overall_mean


# ============================================================
# OVERALL VALIDATION
# ============================================================

def calculate_overall_within(
    dependency,
    blocks
):

    values = []

    for features in blocks.values():

        for i in range(len(features)):

            for j in range(
                i + 1,
                len(features)
            ):

                values.append(
                    dependency.loc[
                        features[i],
                        features[j]
                    ]
                )

    if len(values) == 0:

        return np.nan

    return np.mean(values)


# ============================================================
# FIND SUSPICIOUS BLOCKS
# ============================================================

def identify_suspicious_blocks(
    within_df
):

    print("\n" + "=" * 70)
    print("STEP 4 — IDENTIFYING BLOCKS FOR REVIEW")
    print("=" * 70)

    suspicious = []

    for _, row in within_df.iterrows():

        # Singleton blocks are valid by design
        if row["Block_Size"] == 1:
            continue

        # Flag blocks containing a particularly
        # weak internal relationship.
        if row["Min_Within_Dependency"] < 0.40:

            suspicious.append({

                "Block_ID":
                    int(row["Block_ID"]),

                "Reason":
                    "Weak internal dependency",

                "Minimum_Dependency":
                    row["Min_Within_Dependency"],

                "Weakest_Pair":
                    row["Weakest_Internal_Pair"],

                "Features":
                    row["Features"]
            })

    suspicious_df = pd.DataFrame(
        suspicious
    )

    if len(suspicious_df) == 0:

        print(
            "No blocks have an internal "
            "dependency below 0.40."
        )

    else:

        print(
            "\nBlocks requiring review:"
        )

        print(
            suspicious_df.to_string(
                index=False
            )
        )

    return suspicious_df


# ============================================================
# SAVE VALIDATION REPORT
# ============================================================

def save_outputs(
    within_df,
    between_df,
    suspicious_df,
    overall_within,
    overall_between
):

    print("\n" + "=" * 70)
    print("STEP 5 — SAVING VALIDATION RESULTS")
    print("=" * 70)

    within_path = os.path.join(
        OUTPUT_DIR,
        "within_block_validation.csv"
    )

    between_path = os.path.join(
        OUTPUT_DIR,
        "between_block_validation.csv"
    )

    suspicious_path = os.path.join(
        OUTPUT_DIR,
        "blocks_requiring_review.csv"
    )

    within_df.to_csv(
        within_path,
        index=False
    )

    between_df.to_csv(
        between_path,
        index=False
    )

    suspicious_df.to_csv(
        suspicious_path,
        index=False
    )

    print(
        f"Saved: {within_path}"
    )

    print(
        f"Saved: {between_path}"
    )

    print(
        f"Saved: {suspicious_path}"
    )

    # Overall summary
    summary_path = os.path.join(
        OUTPUT_DIR,
        "block_validation_summary.txt"
    )

    ratio = (
        overall_within /
        overall_between
        if overall_between > 0
        else np.nan
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "BA-MAE FEATURE BLOCK VALIDATION\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Mean within-block dependency: "
            f"{overall_within:.6f}\n"
        )

        f.write(
            f"Mean between-block dependency: "
            f"{overall_between:.6f}\n"
        )

        f.write(
            f"Within/Between ratio: "
            f"{ratio:.6f}\n\n"
        )

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "Higher within-block dependency combined "
            "with lower between-block dependency "
            "indicates better block separation.\n"
        )

    print(
        f"Saved: {summary_path}"
    )

    return ratio


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("BA-MAE — FEATURE BLOCK VALIDATION")
    print("=" * 70)

    # 1. Load
    dependency, blocks_df = load_data()

    # 2. Convert to dictionary
    blocks = get_blocks(
        blocks_df
    )

    # 3. Within-block analysis
    within_df = analyze_within_blocks(
        dependency,
        blocks
    )

    # 4. Between-block analysis
    between_df, overall_between = (
        analyze_between_blocks(
            dependency,
            blocks
        )
    )

    # 5. Overall within dependency
    overall_within = (
        calculate_overall_within(
            dependency,
            blocks
        )
    )

    print("\n" + "=" * 70)
    print("OVERALL BLOCK SEPARATION")
    print("=" * 70)

    print(
        f"Mean within-block dependency: "
        f"{overall_within:.4f}"
    )

    print(
        f"Mean between-block dependency: "
        f"{overall_between:.4f}"
    )

    if overall_between > 0:

        print(
            f"Within/Between ratio: "
            f"{overall_within / overall_between:.2f}"
        )

    # 6. Suspicious blocks
    suspicious_df = (
        identify_suspicious_blocks(
            within_df
        )
    )

    # 7. Save
    save_outputs(
        within_df,
        between_df,
        suspicious_df,
        overall_within,
        overall_between
    )

    # 8. Print complete block table
    print("\n" + "=" * 70)
    print("BLOCK VALIDATION TABLE")
    print("=" * 70)

    print(
        within_df.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("BLOCK VALIDATION COMPLETE")
    print("=" * 70)

    print(
        "\nIMPORTANT:"
    )

    print(
        "Candidate blocks have NOT been frozen yet."
    )

    print(
        "Review the validation results before "
        "creating the final feature_groups.csv."
    )


if __name__ == "__main__":
    main()