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
    "refined_candidate_blocks.csv"
)


def main():

    dependency = pd.read_csv(
        DEPENDENCY_PATH,
        index_col=0
    )

    blocks_df = pd.read_csv(
        BLOCK_PATH
    )

    refined = []

    new_id = 1

    for _, row in blocks_df.iterrows():

        block_id = int(row["Block_ID"])

        features = [
            x.strip()
            for x in row["Features"].split("|")
        ]

        # ----------------------------------------------------
        # Block 2:
        # Split retransmission pair and ICMP pair
        # ----------------------------------------------------

        if block_id == 2:

            refined.append({
                "Block_ID": new_id,
                "Original_Block": 2,
                "Features":
                    "RETRANSMITTED_IN_BYTES | "
                    "RETRANSMITTED_IN_PKTS"
            })

            new_id += 1

            refined.append({
                "Block_ID": new_id,
                "Original_Block": 2,
                "Features":
                    "ICMP_TYPE | "
                    "ICMP_IPV4_TYPE"
            })

            new_id += 1

        # ----------------------------------------------------
        # Block 4:
        # Separate MIN_IP_PKT_LEN from DNS group
        # ----------------------------------------------------

        elif block_id == 4:

            refined.append({
                "Block_ID": new_id,
                "Original_Block": 4,
                "Features":
                    "DNS_QUERY_ID | "
                    "DNS_QUERY_TYPE | "
                    "DNS_TTL_ANSWER"
            })

            new_id += 1

            refined.append({
                "Block_ID": new_id,
                "Original_Block": 4,
                "Features":
                    "MIN_IP_PKT_LEN"
            })

            new_id += 1

        # ----------------------------------------------------
        # Everything else remains unchanged
        # ----------------------------------------------------

        else:

            refined.append({
                "Block_ID": new_id,
                "Original_Block": block_id,
                "Features":
                    " | ".join(features)
            })

            new_id += 1

    refined_df = pd.DataFrame(
        refined
    )

    refined_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 70)
    print("REFINED CANDIDATE BLOCKS")
    print("=" * 70)

    for _, row in refined_df.iterrows():

        features = [
            x.strip()
            for x in row["Features"].split("|")
        ]

        print(
            f"\nBlock {int(row['Block_ID'])} "
            f"(from original Block "
            f"{int(row['Original_Block'])})"
        )

        for feature in features:
            print(
                f"  - {feature}"
            )

    print("\n" + "=" * 70)

    print(
        f"Original blocks: "
        f"{len(blocks_df)}"
    )

    print(
        f"Refined blocks: "
        f"{len(refined_df)}"
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()