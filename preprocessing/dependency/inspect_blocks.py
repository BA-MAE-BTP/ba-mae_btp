import pandas as pd
import numpy as np
import os

DEPENDENCY_PATH = (
    "preprocessing/dependency/"
    "unified_dependency_matrix.csv"
)

BLOCK_PATH = (
    "preprocessing/dependency/"
    "candidate_feature_blocks.csv"
)

OUTPUT_PATH = (
    "preprocessing/dependency/"
    "block_internal_pairs.csv"
)


def main():

    dependency = pd.read_csv(
        DEPENDENCY_PATH,
        index_col=0
    )

    blocks_df = pd.read_csv(
        BLOCK_PATH
    )

    all_pairs = []

    for _, row in blocks_df.iterrows():

        block_id = int(row["Block_ID"])

        features = [
            x.strip()
            for x in row["Features"].split("|")
        ]

        if len(features) < 2:
            continue

        for i in range(len(features)):

            for j in range(i + 1, len(features)):

                f1 = features[i]
                f2 = features[j]

                score = dependency.loc[f1, f2]

                all_pairs.append({
                    "Block_ID": block_id,
                    "Feature_1": f1,
                    "Feature_2": f2,
                    "Dependency": score
                })

    pairs_df = pd.DataFrame(all_pairs)

    pairs_df.sort_values(
        ["Block_ID", "Dependency"],
        ascending=[True, False],
        inplace=True
    )

    pairs_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("INTERNAL FEATURE-PAIR ANALYSIS")
    print("=" * 70)

    for block_id in sorted(
        pairs_df["Block_ID"].unique()
    ):

        block = pairs_df[
            pairs_df["Block_ID"] == block_id
        ]

        print(
            f"\nBLOCK {block_id}"
        )

        print(
            "-" * 60
        )

        for _, r in block.iterrows():

            print(
                f"{r['Feature_1']:<35} "
                f"<-> "
                f"{r['Feature_2']:<35} "
                f"{r['Dependency']:.4f}"
            )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Total internal feature pairs: "
        f"{len(pairs_df)}"
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()