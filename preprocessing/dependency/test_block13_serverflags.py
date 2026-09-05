import pandas as pd
import numpy as np

DEPENDENCY_PATH = "preprocessing/dependency/unified_dependency_matrix.csv"
BLOCK_PATH = "preprocessing/dependency/refined_candidate_blocks.csv"


def calculate(blocks, dependency):

    within = []

    for block_id, features in blocks.items():

        if len(features) < 2:
            continue

        for i in range(len(features)):
            for j in range(i + 1, len(features)):

                score = dependency.loc[
                    features[i],
                    features[j]
                ]

                within.append(score)

    between = []

    block_ids = list(blocks.keys())

    for i in range(len(block_ids)):

        for j in range(i + 1, len(block_ids)):

            b1 = block_ids[i]
            b2 = block_ids[j]

            for f1 in blocks[b1]:
                for f2 in blocks[b2]:

                    between.append(
                        dependency.loc[f1, f2]
                    )

    within_mean = np.mean(within)
    between_mean = np.mean(between)

    return (
        within_mean,
        between_mean,
        within_mean / between_mean
    )


def main():

    dependency = pd.read_csv(
        DEPENDENCY_PATH,
        index_col=0
    )

    df = pd.read_csv(BLOCK_PATH)

    blocks = {}

    for _, row in df.iterrows():

        block_id = int(row["Block_ID"])

        features = [
            x.strip()
            for x in row["Features"].split("|")
        ]

        blocks[block_id] = features

    # ---------------------------------------------------------
    # CURRENT CONFIGURATION
    # ---------------------------------------------------------

    current = {
        k: list(v)
        for k, v in blocks.items()
    }

    # ---------------------------------------------------------
    # ALTERNATIVE CONFIGURATION
    # Move SERVER_TCP_FLAGS from Block 15 → Block 13
    # ---------------------------------------------------------

    alternative = {
        k: list(v)
        for k, v in blocks.items()
    }

    alternative[15].remove("SERVER_TCP_FLAGS")
    alternative[13].append("SERVER_TCP_FLAGS")

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

    current_results = calculate(
        current,
        dependency
    )

    alternative_results = calculate(
        alternative,
        dependency
    )

    print("=" * 75)
    print("BLOCK 13 / BLOCK 15 ALTERNATIVE TEST")
    print("=" * 75)

    print()
    print("CURRENT CONFIGURATION")
    print("-" * 75)

    print(
        f"Mean within dependency : {current_results[0]:.4f}"
    )

    print(
        f"Mean between dependency: {current_results[1]:.4f}"
    )

    print(
        f"Within/Between ratio   : {current_results[2]:.2f}"
    )

    print()
    print("ALTERNATIVE")
    print("-" * 75)

    print(
        "SERVER_TCP_FLAGS moved "
        "from Block 15 → Block 13"
    )

    print(
        f"Mean within dependency : {alternative_results[0]:.4f}"
    )

    print(
        f"Mean between dependency: {alternative_results[1]:.4f}"
    )

    print(
        f"Within/Between ratio   : {alternative_results[2]:.2f}"
    )

    print()
    print("BLOCK 13 AFTER MOVE")
    print("-" * 75)

    for f in alternative[13]:
        print(" -", f)

    print()
    print("BLOCK 15 AFTER MOVE")
    print("-" * 75)

    for f in alternative[15]:
        print(" -", f)


if __name__ == "__main__":
    main()