import pandas as pd
import numpy as np

DEPENDENCY_PATH = "preprocessing/dependency/unified_dependency_matrix.csv"
BLOCK_PATH = "preprocessing/dependency/refined_candidate_blocks.csv"
OUTPUT_PATH = "preprocessing/dependency/refined_block_validation.csv"


def main():

    print("=" * 70)
    print("VALIDATING REFINED CANDIDATE BLOCKS")
    print("=" * 70)

    dependency = pd.read_csv(
        DEPENDENCY_PATH,
        index_col=0
    )

    blocks_df = pd.read_csv(BLOCK_PATH)

    blocks = {}

    for _, row in blocks_df.iterrows():
        block_id = int(row["Block_ID"])

        features = [
            x.strip()
            for x in row["Features"].split("|")
        ]

        blocks[block_id] = features

    within_scores = []
    between_scores = []

    # ---------------------------------------------------------
    # WITHIN-BLOCK DEPENDENCY
    # ---------------------------------------------------------

    for block_id, features in blocks.items():

        if len(features) < 2:
            continue

        for i in range(len(features)):
            for j in range(i + 1, len(features)):

                f1 = features[i]
                f2 = features[j]

                score = dependency.loc[f1, f2]

                within_scores.append({
                    "Block_ID": block_id,
                    "Feature_1": f1,
                    "Feature_2": f2,
                    "Dependency": score
                })

    # ---------------------------------------------------------
    # BETWEEN-BLOCK DEPENDENCY
    # ---------------------------------------------------------

    block_ids = list(blocks.keys())

    for i in range(len(block_ids)):

        b1 = block_ids[i]

        for j in range(i + 1, len(block_ids)):

            b2 = block_ids[j]

            for f1 in blocks[b1]:
                for f2 in blocks[b2]:

                    score = dependency.loc[f1, f2]

                    between_scores.append({
                        "Block_1": b1,
                        "Block_2": b2,
                        "Feature_1": f1,
                        "Feature_2": f2,
                        "Dependency": score
                    })

    within_df = pd.DataFrame(within_scores)
    between_df = pd.DataFrame(between_scores)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    within_mean = within_df["Dependency"].mean()
    within_min = within_df["Dependency"].min()
    within_max = within_df["Dependency"].max()

    between_mean = between_df["Dependency"].mean()
    between_min = between_df["Dependency"].min()
    between_max = between_df["Dependency"].max()

    ratio = within_mean / between_mean

    print()
    print("REFINED BLOCK SUMMARY")
    print("-" * 70)

    print(f"Number of blocks          : {len(blocks)}")
    print(f"Within-block pairs        : {len(within_df)}")
    print(f"Between-block pairs       : {len(between_df)}")

    print()
    print(f"Mean within dependency    : {within_mean:.4f}")
    print(f"Minimum within dependency : {within_min:.4f}")
    print(f"Maximum within dependency : {within_max:.4f}")

    print()
    print(f"Mean between dependency   : {between_mean:.4f}")
    print(f"Minimum between dependency: {between_min:.4f}")
    print(f"Maximum between dependency: {between_max:.4f}")

    print()
    print(f"Within / Between ratio    : {ratio:.2f}")

    # ---------------------------------------------------------
    # WEAK INTERNAL PAIRS
    # ---------------------------------------------------------

    weak = within_df[
        within_df["Dependency"] < 0.50
    ].sort_values("Dependency")

    print()
    print("WEAK WITHIN-BLOCK PAIRS (< 0.50)")
    print("-" * 70)

    if len(weak) == 0:
        print("None")
    else:
        print(weak.to_string(index=False))

    # ---------------------------------------------------------
    # STRONG CROSS-BLOCK PAIRS
    # ---------------------------------------------------------

    strong_cross = between_df[
        between_df["Dependency"] >= 0.70
    ].sort_values(
        "Dependency",
        ascending=False
    )

    print()
    print("STRONG CROSS-BLOCK PAIRS (>= 0.70)")
    print("-" * 70)

    print(
        strong_cross.head(20).to_string(index=False)
    )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------

    within_df.to_csv(
        "preprocessing/dependency/refined_within_block_pairs.csv",
        index=False
    )

    between_df.to_csv(
        "preprocessing/dependency/refined_between_block_pairs.csv",
        index=False
    )

    summary = pd.DataFrame([{
        "Num_Blocks": len(blocks),
        "Within_Pairs": len(within_df),
        "Between_Pairs": len(between_df),
        "Mean_Within": within_mean,
        "Min_Within": within_min,
        "Max_Within": within_max,
        "Mean_Between": between_mean,
        "Min_Between": between_min,
        "Max_Between": between_max,
        "Within_Between_Ratio": ratio
    }])

    summary.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()