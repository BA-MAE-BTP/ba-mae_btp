import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BLOCK_PATH = "preprocessing/dependency/refined_candidate_blocks.csv"


# ============================================================
# LOAD BLOCKS
# ============================================================

def load_blocks(block_path=BLOCK_PATH):
    """
    Load the frozen dependency blocks.

    Returns
    -------
    blocks : dict
        {block_id: [feature1, feature2, ...]}
    """

    df = pd.read_csv(block_path)

    blocks = {}

    for _, row in df.iterrows():

        block_id = int(row["Block_ID"])

        features = [
            x.strip()
            for x in row["Features"].split("|")
        ]

        blocks[block_id] = features

    return blocks


# ============================================================
# FEATURE → BLOCK MAPPING
# ============================================================

def create_feature_to_block_mapping(blocks):
    """
    Create a mapping from each feature to its block.

    Example:
        TCP_FLAGS -> 13
        CLIENT_TCP_FLAGS -> 13
    """

    feature_to_block = {}

    for block_id, features in blocks.items():

        for feature in features:

            if feature in feature_to_block:

                raise ValueError(
                    f"Feature '{feature}' appears in "
                    f"multiple blocks: "
                    f"{feature_to_block[feature]} and {block_id}"
                )

            feature_to_block[feature] = block_id

    return feature_to_block


# ============================================================
# VALIDATE BLOCK STRUCTURE
# ============================================================

def validate_blocks(blocks, expected_feature_count=39):
    """
    Validate that the block configuration is usable.
    """

    all_features = []

    for block_id, features in blocks.items():

        if len(features) == 0:

            raise ValueError(
                f"Block {block_id} contains no features."
            )

        all_features.extend(features)

    # Check duplicate features
    if len(all_features) != len(set(all_features)):

        duplicates = [
            feature
            for feature in set(all_features)
            if all_features.count(feature) > 1
        ]

        raise ValueError(
            f"Duplicate features detected: {duplicates}"
        )

    # Check feature count
    if len(all_features) != expected_feature_count:

        raise ValueError(
            f"Expected {expected_feature_count} features, "
            f"but found {len(all_features)}."
        )

    return True


# ============================================================
# SELECT BLOCKS
# ============================================================

def select_blocks(
    blocks,
    masking_ratio=0.25,
    rng=None
):
    """
    Select complete blocks for masking.

    The selection is performed at BLOCK level,
    never at individual-feature level.

    masking_ratio is approximate because blocks
    have different sizes.
    """

    if rng is None:

        rng = np.random.default_rng(42)

    block_ids = list(blocks.keys())

    rng.shuffle(block_ids)

    selected_blocks = []

    total_features = sum(
        len(features)
        for features in blocks.values()
    )

    target_features = masking_ratio * total_features

    masked_features = 0

    for block_id in block_ids:

        # Don't exceed the target unnecessarily
        if masked_features >= target_features:

            break

        selected_blocks.append(block_id)

        masked_features += len(
            blocks[block_id]
        )

    return selected_blocks


# ============================================================
# CREATE MASK
# ============================================================

def create_block_mask(
    feature_names,
    blocks,
    selected_blocks
):
    """
    Create a boolean feature mask.

    True  = feature is masked
    False = feature remains visible
    """

    mask = np.zeros(
        len(feature_names),
        dtype=bool
    )

    feature_to_index = {
        feature: i
        for i, feature in enumerate(feature_names)
    }

    for block_id in selected_blocks:

        if block_id not in blocks:

            raise ValueError(
                f"Unknown block ID: {block_id}"
            )

        for feature in blocks[block_id]:

            if feature not in feature_to_index:

                raise ValueError(
                    f"Feature '{feature}' from "
                    f"Block {block_id} not found "
                    f"in input features."
                )

            index = feature_to_index[feature]

            mask[index] = True

    return mask


# ============================================================
# APPLY MASK
# ============================================================

def apply_mask(
    X,
    mask,
    mask_value=0.0
):
    """
    Apply the feature mask to a NumPy array.

    X shape:
        (n_samples, n_features)

    mask shape:
        (n_features,)
    """

    X_masked = X.copy()

    X_masked[:, mask] = mask_value

    return X_masked


# ============================================================
# SINGLE-SAMPLE HELPER
# ============================================================

def mask_sample(
    sample,
    feature_names,
    blocks,
    masking_ratio=0.25,
    rng=None,
    mask_value=0.0
):
    """
    Mask one feature vector.

    Returns
    -------
    masked_sample
    mask
    selected_blocks
    """

    if sample.ndim != 1:

        raise ValueError(
            "sample must be a 1D NumPy array."
        )

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

    masked_sample = sample.copy()

    masked_sample[mask] = mask_value

    return (
        masked_sample,
        mask,
        selected_blocks
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("BLOCK-AWARE MASKER")
    print("=" * 70)

    blocks = load_blocks()

    validate_blocks(
        blocks,
        expected_feature_count=39
    )

    feature_to_block = create_feature_to_block_mapping(
        blocks
    )

    print()
    print(f"Number of blocks  : {len(blocks)}")
    print(f"Number of features: {len(feature_to_block)}")

    print()
    print("Block structure:")
    print("-" * 70)

    for block_id, features in blocks.items():

        print(
            f"Block {block_id:2d} "
            f"(size={len(features)}): "
            + " | ".join(features)
        )

    print()
    print("✓ Block configuration successfully loaded.")
    print("✓ All 39 features belong to exactly one block.")