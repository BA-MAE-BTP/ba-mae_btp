import pandas as pd

DEPENDENCY_PATH = "preprocessing/dependency/unified_dependency_matrix.csv"
BLOCK_PATH = "preprocessing/dependency/refined_candidate_blocks.csv"


def main():

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

    print("=" * 75)
    print("CROSS-BLOCK DEPENDENCY INSPECTION")
    print("=" * 75)

    pairs = []

    block_ids = list(blocks.keys())

    for i in range(len(block_ids)):

        b1 = block_ids[i]

        for j in range(i + 1, len(block_ids)):

            b2 = block_ids[j]

            for f1 in blocks[b1]:

                for f2 in blocks[b2]:

                    score = dependency.loc[f1, f2]

                    pairs.append({
                        "Block_1": b1,
                        "Block_2": b2,
                        "Feature_1": f1,
                        "Feature_2": f2,
                        "Dependency": score
                    })

    df = pd.DataFrame(pairs)

    df = df.sort_values(
        "Dependency",
        ascending=False
    )

    print()
    print("TOP 30 CROSS-BLOCK PAIRS")
    print("-" * 75)

    print(
        df.head(30).to_string(index=False)
    )

    print()
    print("=" * 75)
    print("CROSS-BLOCK PAIRS >= 0.60")
    print("=" * 75)

    strong = df[df["Dependency"] >= 0.60]

    print(
        strong.to_string(index=False)
    )

    print()
    print("=" * 75)
    print("CROSS-BLOCK PAIRS >= 0.70")
    print("=" * 75)

    very_strong = df[df["Dependency"] >= 0.70]

    print(
        very_strong.to_string(index=False)
    )

    df.to_csv(
        "preprocessing/dependency/all_cross_block_pairs.csv",
        index=False
    )

    print()
    print("Saved:")
    print("preprocessing/dependency/all_cross_block_pairs.csv")


if __name__ == "__main__":
    main()