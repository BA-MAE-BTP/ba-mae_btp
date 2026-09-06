import os
import numpy as np
import pandas as pd


# ============================================================
# BA-MAE BLOCK-AWARE MASKER
# ============================================================
#
# 39 model features
# 20 dependency blocks
# Target masking ratio = 25%
#
# IMPORTANT:
# A dependency block is ALWAYS masked completely or
# left completely visible.
#
# The selector chooses complete blocks whose total number
# of features is closest to the requested masking count.
#
# ============================================================


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DEPENDENCY_FILE = os.path.join(
    os.path.dirname(BASE_DIR),
    "dependency",
    "refined_candidate_blocks.csv"
)


# ============================================================
# LOAD BLOCKS
# ============================================================

def load_blocks(
    dependency_file=DEPENDENCY_FILE
):
    """
    Load dependency blocks.

    Expected CSV:

        Block_ID
        Original_Block
        Features

    Features contains pipe-separated feature names.

    Returns
    -------
    dict

        {
            block_id: [feature1, feature2, ...]
        }
    """

    if not os.path.exists(
        dependency_file
    ):
        raise FileNotFoundError(
            "\nDependency file not found:\n"
            f"{dependency_file}"
        )

    df = pd.read_csv(
        dependency_file
    )

    # --------------------------------------------------------
    # Find Features column
    # --------------------------------------------------------

    feature_column = None

    for column in [
        "Features",
        "features",
        "FEATURES",
        "Feature",
        "feature"
    ]:
        if column in df.columns:
            feature_column = column
            break

    if feature_column is None:
        raise ValueError(
            "\nCould not find Features column.\n"
            f"Columns found: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Find block ID column
    # --------------------------------------------------------

    block_id_column = None

    for column in [
        "Block_ID",
        "block_id",
        "BlockID",
        "block"
    ]:
        if column in df.columns:
            block_id_column = column
            break

    # If Block_ID is unavailable, use row numbers
    if block_id_column is None:
        block_ids = range(
            1,
            len(df) + 1
        )
    else:
        block_ids = df[
            block_id_column
        ].tolist()

    # --------------------------------------------------------
    # Parse blocks
    # --------------------------------------------------------

    blocks = {}

    for block_id, value in zip(
        block_ids,
        df[feature_column]
    ):

        if pd.isna(value):

            raise ValueError(
                f"Empty feature list "
                f"for block {block_id}."
            )

        features = [
            feature.strip()
            for feature in str(
                value
            ).split("|")
            if feature.strip()
        ]

        if not features:

            raise ValueError(
                f"Block {block_id} "
                "contains no features."
            )

        blocks[int(block_id)] = features

    return blocks


# ============================================================
# VALIDATE BLOCKS
# ============================================================

def validate_blocks(
    blocks,
    expected_feature_count=39
):
    """
    Validate dependency blocks.

    Checks:

    1. Blocks exist
    2. No block is empty
    3. No feature occurs more than once
    4. Exactly 39 model features exist

    Returns
    -------
    True
        If validation passes.
    """

    if not isinstance(
        blocks,
        dict
    ):
        raise TypeError(
            "blocks must be a dictionary."
        )

    if len(blocks) == 0:
        raise ValueError(
            "No dependency blocks found."
        )

    all_features = []

    # --------------------------------------------------------
    # Check individual blocks
    # --------------------------------------------------------

    for block_id, features in blocks.items():

        if not isinstance(
            features,
            list
        ):
            raise TypeError(
                f"Block {block_id} "
                "must contain a list of features."
            )

        if len(features) == 0:
            raise ValueError(
                f"Block {block_id} is empty."
            )

        all_features.extend(
            features
        )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    feature_counts = {}

    for feature in all_features:

        feature_counts[feature] = (
            feature_counts.get(
                feature,
                0
            ) + 1
        )

    duplicates = [
        feature
        for feature, count
        in feature_counts.items()
        if count > 1
    ]

    if duplicates:

        raise ValueError(
            "\nDuplicate features detected:\n"
            f"{sorted(duplicates)}"
        )

    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    actual_count = len(
        all_features
    )

    if actual_count != expected_feature_count:

        raise ValueError(
            "\nIncorrect feature count.\n"
            f"Expected: {expected_feature_count}\n"
            f"Found   : {actual_count}"
        )

    print(
        f"✓ Dependency blocks validated: "
        f"{len(blocks)} blocks, "
        f"{actual_count} features"
    )

    return True


# ============================================================
# SELECT BLOCKS
# ============================================================

def select_blocks(
    blocks,
    masking_ratio,
    rng
):
    """
    Select complete dependency blocks whose combined
    feature count is closest to the desired masking ratio.

    Example:

        39 features
        masking ratio = 0.25

        target = 39 * 0.25
              = 9.75

        closest integer = 10

        actual ratio = 10 / 39
                     = 0.2564

    Parameters
    ----------
    blocks : dict
        Dependency blocks.

    masking_ratio : float
        Desired masking ratio.

    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    list
        Selected block IDs.
    """

    if not (
        0.0 < masking_ratio < 1.0
    ):
        raise ValueError(
            "masking_ratio must be "
            "between 0 and 1."
        )

    # --------------------------------------------------------
    # Total features
    # --------------------------------------------------------

    total_features = sum(
        len(features)
        for features in blocks.values()
    )

    if total_features <= 1:
        raise ValueError(
            "Not enough features for masking."
        )

    # --------------------------------------------------------
    # Desired number of masked features
    # --------------------------------------------------------

    target_count = round(
        total_features
        * masking_ratio
    )

    target_count = max(
        1,
        target_count
    )

    target_count = min(
        total_features - 1,
        target_count
    )

    # --------------------------------------------------------
    # Random block order
    #
    # This preserves randomness between trials while the
    # dynamic-programming selection keeps the final count
    # as close as possible to the target.
    # --------------------------------------------------------

    block_ids = list(
        blocks.keys()
    )

    block_ids = np.array(
        block_ids
    )

    rng.shuffle(
        block_ids
    )

    # --------------------------------------------------------
    # Dynamic programming
    #
    # reachable[count] = list of block IDs producing
    # exactly 'count' masked features.
    # --------------------------------------------------------

    reachable = {
        0: []
    }

    for block_id in block_ids:

        block_size = len(
            blocks[int(block_id)]
        )

        current_states = list(
            reachable.items()
        )

        for current_count, selected in current_states:

            new_count = (
                current_count
                + block_size
            )

            # Never mask all features
            if new_count >= total_features:
                continue

            # Keep one valid combination for this count
            if new_count not in reachable:

                reachable[
                    new_count
                ] = (
                    selected
                    + [int(block_id)]
                )

    # --------------------------------------------------------
    # Find closest achievable count
    # --------------------------------------------------------

    if len(reachable) <= 1:

        raise RuntimeError(
            "Could not construct a valid "
            "whole-block masking combination."
        )

    best_count = min(
        reachable.keys(),
        key=lambda count: (
            abs(
                count
                - target_count
            ),
            count
        )
    )

    selected_blocks = reachable[
        best_count
    ]

    return selected_blocks


# ============================================================
# CREATE FEATURE MASK
# ============================================================

def create_block_mask(
    feature_names,
    blocks,
    selected_blocks
):
    """
    Create a boolean feature mask.

    True  = masked
    False = visible

    Every selected dependency block is masked completely.
    """

    feature_to_index = {
        feature: index
        for index, feature
        in enumerate(feature_names)
    }

    mask = np.zeros(
        len(feature_names),
        dtype=bool
    )

    # --------------------------------------------------------
    # Validate feature names
    # --------------------------------------------------------

    for feature in feature_names:

        if feature not in feature_to_index:
            raise ValueError(
                f"Invalid feature: {feature}"
            )

    # --------------------------------------------------------
    # Mask selected blocks
    # --------------------------------------------------------

    for block_id in selected_blocks:

        if block_id not in blocks:

            raise ValueError(
                f"Unknown block ID: "
                f"{block_id}"
            )

        for feature in blocks[
            block_id
        ]:

            if feature not in feature_to_index:

                raise ValueError(
                    f"Feature '{feature}' "
                    "from dependency block "
                    "is missing from feature_names."
                )

            index = feature_to_index[
                feature
            ]

            mask[index] = True

    # --------------------------------------------------------
    # Final integrity validation
    # --------------------------------------------------------

    for block_id, features in blocks.items():

        indices = [
            feature_to_index[
                feature
            ]
            for feature in features
        ]

        values = mask[
            indices
        ]

        # A block must be either:
        #
        # [False, False, False]
        #
        # or:
        #
        # [True, True, True]

        if not (
            np.all(values)
            or
            np.all(~values)
        ):

            raise RuntimeError(
                "\nPartial dependency block "
                "masking detected.\n"
                f"Block: {block_id}\n"
                f"Features: {features}"
            )

    return mask


# ============================================================
# OPTIONAL CLASS INTERFACE
# ============================================================

class BlockMasker:

    """
    Object-oriented interface for BA-MAE masking.

    This is kept compatible with the standalone functions
    above so both training code and statistical tests can
    use the same masking implementation.
    """

    def __init__(
        self,
        feature_names,
        mask_ratio=0.25,
        mask_value=0.0,
        random_state=42
    ):

        self.feature_names = list(
            feature_names
        )

        self.mask_ratio = float(
            mask_ratio
        )

        self.mask_value = float(
            mask_value
        )

        self.rng = np.random.default_rng(
            random_state
        )

        self.blocks = load_blocks()

        validate_blocks(
            self.blocks,
            expected_feature_count=39
        )

        block_features = [
            feature
            for features in self.blocks.values()
            for feature in features
        ]

        if set(
            self.feature_names
        ) != set(
            block_features
        ):

            missing = (
                set(block_features)
                - set(self.feature_names)
            )

            extra = (
                set(self.feature_names)
                - set(block_features)
            )

            raise ValueError(
                "\nFeature mismatch.\n"
                f"Missing: {sorted(missing)}\n"
                f"Extra: {sorted(extra)}"
            )

    def generate_mask(self):

        selected_blocks = select_blocks(
            self.blocks,
            self.mask_ratio,
            self.rng
        )

        mask = create_block_mask(
            self.feature_names,
            self.blocks,
            selected_blocks
        )

        return (
            mask,
            selected_blocks
        )

    def mask_batch(
        self,
        X
    ):

        X = np.asarray(
            X
        )

        if X.ndim != 2:

            raise ValueError(
                "X must be 2-dimensional."
            )

        if X.shape[1] != len(
            self.feature_names
        ):

            raise ValueError(
                "Feature dimension mismatch."
            )

        mask, selected_blocks = (
            self.generate_mask()
        )

        X_masked = X.copy()

        X_masked[
            :,
            mask
        ] = self.mask_value

        return (
            X_masked,
            mask,
            selected_blocks
        )


# ============================================================
# SELF TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "BA-MAE BLOCK MASKER — SELF TEST"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    blocks = load_blocks()

    print(
        f"\nDependency file:"
    )

    print(
        DEPENDENCY_FILE
    )

    print(
        "\nBlocks loaded:"
    )

    print(
        len(blocks)
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_blocks(
        blocks,
        expected_feature_count=39
    )

    # --------------------------------------------------------
    # Feature list
    # --------------------------------------------------------

    feature_names = [
        feature
        for features in blocks.values()
        for feature in features
    ]

    print(
        f"Feature count: "
        f"{len(feature_names)}"
    )

    # --------------------------------------------------------
    # Display blocks
    # --------------------------------------------------------

    print(
        "\nDependency blocks:"
    )

    for block_id, features in blocks.items():

        print(
            f"Block {block_id:2d} "
            f"({len(features)}): "
            f"{' | '.join(features)}"
        )

    # --------------------------------------------------------
    # Generate mask
    # --------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    selected_blocks = select_blocks(
        blocks,
        0.25,
        rng
    )

    mask = create_block_mask(
        feature_names,
        blocks,
        selected_blocks
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    masked_count = int(
        mask.sum()
    )

    actual_ratio = float(
        mask.mean()
    )

    target_ratio = 0.25

    print(
        "\n" + "-" * 70
    )

    print(
        f"Target ratio       : "
        f"{target_ratio:.4f}"
    )

    print(
        f"Target feature count: "
        f"{round(39 * target_ratio)}"
    )

    print(
        f"Actual feature count: "
        f"{masked_count}"
    )

    print(
        f"Actual ratio        : "
        f"{actual_ratio:.4f}"
    )

    print(
        f"Ratio error         : "
        f"{abs(actual_ratio - target_ratio):.4f}"
    )

    # --------------------------------------------------------
    # Selected blocks
    # --------------------------------------------------------

    print(
        "\nSelected blocks:"
    )

    for block_id in selected_blocks:

        print(
            f"  Block {block_id}: "
            f"{' | '.join(blocks[block_id])}"
        )

    # --------------------------------------------------------
    # Integrity
    # --------------------------------------------------------

    violations = 0

    feature_to_index = {
        feature: index
        for index, feature
        in enumerate(feature_names)
    }

    for block_id, features in blocks.items():

        indices = [
            feature_to_index[
                feature
            ]
            for feature in features
        ]

        values = mask[
            indices
        ]

        if not (
            np.all(values)
            or
            np.all(~values)
        ):

            violations += 1

    print(
        "\nBlock integrity:"
    )

    print(
        f"Partial-block violations: "
        f"{violations}"
    )

    if violations != 0:

        raise RuntimeError(
            "Block integrity test FAILED."
        )

    print(
        "✓ BLOCK INTEGRITY PASSED"
    )

    # --------------------------------------------------------
    # Test actual masking
    # --------------------------------------------------------

    X = np.random.randn(
        5,
        39
    )

    X_masked = X.copy()

    X_masked[
        :,
        mask
    ] = 0.0

    if not np.all(
        X_masked[:, mask] == 0.0
    ):

        raise RuntimeError(
            "Mask application FAILED."
        )

    print(
        "✓ MASK APPLICATION PASSED"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "✓ BLOCK MASKER SELF-TEST PASSED"
    )

    print(
        "=" * 70
    )