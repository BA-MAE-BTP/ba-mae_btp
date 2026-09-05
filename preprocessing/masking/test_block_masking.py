import numpy as np

from block_masker import (
    load_blocks,
    validate_blocks,
    create_feature_to_block_mapping,
    create_block_mask,
    apply_mask
)


def main():

    print("=" * 70)
    print("TESTING BLOCK-AWARE MASKING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD BLOCKS
    # --------------------------------------------------------

    blocks = load_blocks()

    validate_blocks(
        blocks,
        expected_feature_count=39
    )

    feature_to_block = create_feature_to_block_mapping(
        blocks
    )

    feature_names = list(
        feature_to_block.keys()
    )

    print()
    print(f"Features loaded: {len(feature_names)}")

    # --------------------------------------------------------
    # CREATE SYNTHETIC DATA
    # --------------------------------------------------------

    X = np.arange(
        39,
        dtype=float
    ).reshape(1, 39)

    print()
    print("Original sample:")
    print(X)

    # --------------------------------------------------------
    # TEST BLOCK 13
    # --------------------------------------------------------

    selected_blocks = [13]

    mask = create_block_mask(
        feature_names,
        blocks,
        selected_blocks
    )

    X_masked = apply_mask(
        X,
        mask,
        mask_value=0.0
    )

    print()
    print("Selected block:")
    print(selected_blocks)

    print()
    print("Masked features:")

    for i, feature in enumerate(feature_names):

        if mask[i]:

            print(
                f"  ✓ {feature} "
                f"(Block {feature_to_block[feature]})"
            )

    # --------------------------------------------------------
    # VERIFY WHOLE BLOCK WAS MASKED
    # --------------------------------------------------------

    expected_features = set(
        blocks[13]
    )

    actual_features = {
        feature_names[i]
        for i in range(len(feature_names))
        if mask[i]
    }

    if actual_features != expected_features:

        raise AssertionError(
            "Block 13 was not masked completely."
        )

    # --------------------------------------------------------
    # VERIFY NO OTHER BLOCK WAS MASKED
    # --------------------------------------------------------

    for feature in feature_names:

        belongs_to_selected = (
            feature_to_block[feature]
            in selected_blocks
        )

        index = feature_names.index(
            feature
        )

        is_masked = bool(
            mask[index]
        )

        if belongs_to_selected != is_masked:

            raise AssertionError(
                f"Incorrect masking for feature: "
                f"{feature}"
            )

    # --------------------------------------------------------
    # VERIFY MASKED VALUES
    # --------------------------------------------------------

    for feature in expected_features:

        index = feature_names.index(
            feature
        )

        if X_masked[0, index] != 0.0:

            raise AssertionError(
                f"{feature} was not replaced correctly."
            )

    print()
    print("=" * 70)
    print("✓ BLOCK MASKING TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()