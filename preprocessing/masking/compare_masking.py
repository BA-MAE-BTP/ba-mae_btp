import numpy as np

from block_masker import (
    load_blocks,
    validate_blocks,
    create_block_mask,
    select_blocks
)


def random_feature_mask(
    num_features,
    masking_ratio,
    rng
):
    """
    Standard random feature-level masking.
    """

    num_mask = int(
        round(
            num_features *
            masking_ratio
        )
    )

    indices = rng.choice(
        num_features,
        size=num_mask,
        replace=False
    )

    mask = np.zeros(
        num_features,
        dtype=bool
    )

    mask[indices] = True

    return mask


def main():

    print("=" * 70)
    print("RANDOM VS BLOCK-AWARE MASKING")
    print("=" * 70)

    blocks = load_blocks()

    validate_blocks(
        blocks,
        expected_feature_count=39
    )

    feature_names = [
        feature
        for block_features in blocks.values()
        for feature in block_features
    ]

    rng = np.random.default_rng(42)

    masking_ratio = 0.25

    # --------------------------------------------------------
    # RANDOM FEATURE MASKING
    # --------------------------------------------------------

    random_mask = random_feature_mask(
        len(feature_names),
        masking_ratio,
        rng
    )

    # --------------------------------------------------------
    # BLOCK MASKING
    # --------------------------------------------------------

    selected_blocks = select_blocks(
        blocks,
        masking_ratio,
        rng
    )

    block_mask = create_block_mask(
        feature_names,
        blocks,
        selected_blocks
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    print()
    print("RANDOM MASKING")
    print("-" * 70)

    print(
        f"Target masking ratio : "
        f"{masking_ratio:.2f}"
    )

    print(
        f"Masked features      : "
        f"{random_mask.sum()}"
    )

    print(
        f"Actual ratio         : "
        f"{random_mask.mean():.4f}"
    )

    print()
    print("BLOCK-AWARE MASKING")
    print("-" * 70)

    print(
        f"Target masking ratio : "
        f"{masking_ratio:.2f}"
    )

    print(
        f"Selected blocks      : "
        f"{selected_blocks}"
    )

    print(
        f"Masked features      : "
        f"{block_mask.sum()}"
    )

    print(
        f"Actual ratio         : "
        f"{block_mask.mean():.4f}"
    )

    # --------------------------------------------------------
    # VERIFY BLOCK INTEGRITY
    # --------------------------------------------------------

    violations = []

    for block_id, features in blocks.items():

        indices = [
            feature_names.index(feature)
            for feature in features
        ]

        values = block_mask[indices]

        # Either entire block masked
        # or entire block visible

        if not (
            np.all(values)
            or
            np.all(~values)
        ):

            violations.append(
                block_id
            )

    print()
    print("BLOCK INTEGRITY")
    print("-" * 70)

    if violations:

        print(
            "✗ Partial masking detected in blocks:",
            violations
        )

    else:

        print(
            "✓ No partial block masking detected."
        )

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()