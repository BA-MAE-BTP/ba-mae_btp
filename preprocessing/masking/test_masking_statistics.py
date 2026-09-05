import numpy as np

from block_masker import (
    load_blocks,
    validate_blocks,
    create_block_mask,
    select_blocks
)


def main():

    print("=" * 70)
    print("BLOCK MASKING STATISTICAL TEST")
    print("=" * 70)

    blocks = load_blocks()

    validate_blocks(
        blocks,
        expected_feature_count=39
    )

    feature_names = [
        feature
        for features in blocks.values()
        for feature in features
    ]

    rng = np.random.default_rng(42)

    masking_ratio = 0.25
    num_trials = 1000

    ratios = []
    block_counts = []
    violations = 0

    for _ in range(num_trials):

        selected_blocks = select_blocks(
            blocks,
            masking_ratio,
            rng
        )

        mask = create_block_mask(
            feature_names,
            blocks,
            selected_blocks
        )

        ratios.append(mask.mean())
        block_counts.append(len(selected_blocks))

        # Check that every block is either
        # completely masked or completely visible
        for block_id, features in blocks.items():

            indices = [
                feature_names.index(feature)
                for feature in features
            ]

            values = mask[indices]

            if not (
                np.all(values)
                or np.all(~values)
            ):
                violations += 1

    ratios = np.array(ratios)
    block_counts = np.array(block_counts)

    print()
    print("TRIALS")
    print("-" * 70)
    print(f"Number of trials       : {num_trials}")

    print()
    print("MASKING RATIO")
    print("-" * 70)
    print(f"Target ratio           : {masking_ratio:.4f}")
    print(f"Mean actual ratio      : {ratios.mean():.4f}")
    print(f"Minimum actual ratio   : {ratios.min():.4f}")
    print(f"Maximum actual ratio   : {ratios.max():.4f}")
    print(f"Std. deviation         : {ratios.std():.4f}")

    print()
    print("SELECTED BLOCKS")
    print("-" * 70)
    print(f"Mean blocks selected   : {block_counts.mean():.2f}")
    print(f"Minimum blocks         : {block_counts.min()}")
    print(f"Maximum blocks         : {block_counts.max()}")

    print()
    print("BLOCK INTEGRITY")
    print("-" * 70)
    print(f"Partial-block violations: {violations}")

    print()
    print("=" * 70)

    if violations == 0:
        print("✓ BLOCK INTEGRITY PASSED")
    else:
        print("✗ BLOCK INTEGRITY FAILED")

    print("=" * 70)


if __name__ == "__main__":
    main()
    