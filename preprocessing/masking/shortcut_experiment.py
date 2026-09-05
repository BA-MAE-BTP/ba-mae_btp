import pandas as pd
import numpy as np

from block_masker import (
    load_blocks,
    validate_blocks,
    create_block_mask,
    select_blocks
)


# ============================================================
# CONFIGURATION
# ============================================================

DEPENDENCY_PATH = (
    "preprocessing/dependency/"
    "unified_dependency_matrix.csv"
)

NUM_TRIALS = 10000
MASKING_RATIO = 0.25
DEPENDENCY_THRESHOLD = 0.70
RANDOM_SEED = 42


# ============================================================
# RANDOM FEATURE MASKING
# ============================================================

def random_feature_mask(
    num_features,
    masking_ratio,
    rng
):

    num_mask = int(
        round(num_features * masking_ratio)
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


# ============================================================
# BLOCK-AWARE MASKING
# ============================================================

def block_feature_mask(
    feature_names,
    blocks,
    masking_ratio,
    rng
):

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

    return mask


# ============================================================
# SHORTCUT ANALYSIS
# ============================================================

def analyze_shortcuts(
    mask,
    feature_names,
    dependency,
    threshold
):

    shortcut_pairs = 0
    masked_features = 0

    masked_features_with_visible_partner = 0

    for i, feature in enumerate(feature_names):

        if not mask[i]:
            continue

        masked_features += 1

        # Check whether this masked feature has
        # a strongly dependent visible feature.
        for j, other_feature in enumerate(feature_names):

            if i == j:
                continue

            if mask[j]:
                continue

            score = dependency.loc[
                feature,
                other_feature
            ]

            if score >= threshold:

                shortcut_pairs += 1

                masked_features_with_visible_partner += 1

                break

    if masked_features == 0:

        shortcut_rate = 0.0

    else:

        shortcut_rate = (
            masked_features_with_visible_partner
            / masked_features
        )

    return (
        shortcut_pairs,
        masked_features,
        masked_features_with_visible_partner,
        shortcut_rate
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("SHORTCUT EXPERIMENT")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD DEPENDENCY MATRIX
    # --------------------------------------------------------

    dependency = pd.read_csv(
        DEPENDENCY_PATH,
        index_col=0
    )

    # --------------------------------------------------------
    # LOAD BLOCKS
    # --------------------------------------------------------

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

    print()
    print(f"Features              : {len(feature_names)}")
    print(f"Blocks                : {len(blocks)}")
    print(f"Trials                : {NUM_TRIALS}")
    print(f"Masking ratio         : {MASKING_RATIO:.2f}")
    print(
        f"Dependency threshold  : "
        f"{DEPENDENCY_THRESHOLD:.2f}"
    )

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    random_rates = []
    block_rates = []

    random_shortcut_pairs = []
    block_shortcut_pairs = []

    # --------------------------------------------------------
    # RUN EXPERIMENT
    # --------------------------------------------------------

    for _ in range(NUM_TRIALS):

        # Random masking
        random_mask = random_feature_mask(
            len(feature_names),
            MASKING_RATIO,
            rng
        )

        result_random = analyze_shortcuts(
            random_mask,
            feature_names,
            dependency,
            DEPENDENCY_THRESHOLD
        )

        random_shortcut_pairs.append(
            result_random[0]
        )

        random_rates.append(
            result_random[3]
        )

        # Block-aware masking
        block_mask = block_feature_mask(
            feature_names,
            blocks,
            MASKING_RATIO,
            rng
        )

        result_block = analyze_shortcuts(
            block_mask,
            feature_names,
            dependency,
            DEPENDENCY_THRESHOLD
        )

        block_shortcut_pairs.append(
            result_block[0]
        )

        block_rates.append(
            result_block[3]
        )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    random_rates = np.array(
        random_rates
    )

    block_rates = np.array(
        block_rates
    )

    random_shortcut_pairs = np.array(
        random_shortcut_pairs
    )

    block_shortcut_pairs = np.array(
        block_shortcut_pairs
    )

    print()
    print("=" * 75)
    print("RESULTS")
    print("=" * 75)

    print()
    print("RANDOM FEATURE MASKING")
    print("-" * 75)

    print(
        f"Mean shortcut rate : "
        f"{random_rates.mean():.4f}"
    )

    print(
        f"Std. shortcut rate  : "
        f"{random_rates.std():.4f}"
    )

    print(
        f"Min shortcut rate   : "
        f"{random_rates.min():.4f}"
    )

    print(
        f"Max shortcut rate   : "
        f"{random_rates.max():.4f}"
    )

    print(
        f"Mean shortcut pairs : "
        f"{random_shortcut_pairs.mean():.2f}"
    )

    print()
    print("BLOCK-AWARE MASKING")
    print("-" * 75)

    print(
        f"Mean shortcut rate : "
        f"{block_rates.mean():.4f}"
    )

    print(
        f"Std. shortcut rate  : "
        f"{block_rates.std():.4f}"
    )

    print(
        f"Min shortcut rate   : "
        f"{block_rates.min():.4f}"
    )

    print(
        f"Max shortcut rate   : "
        f"{block_rates.max():.4f}"
    )

    print(
        f"Mean shortcut pairs : "
        f"{block_shortcut_pairs.mean():.2f}"
    )

    # --------------------------------------------------------
    # IMPROVEMENT
    # --------------------------------------------------------

    random_mean = random_rates.mean()
    block_mean = block_rates.mean()

    reduction = (
        (random_mean - block_mean)
        / random_mean
        * 100
        if random_mean > 0
        else 0
    )

    print()
    print("=" * 75)
    print("SHORTCUT REDUCTION")
    print("=" * 75)

    print(
        f"Random shortcut rate : "
        f"{random_mean:.4f}"
    )

    print(
        f"Block shortcut rate  : "
        f"{block_mean:.4f}"
    )

    print(
        f"Reduction            : "
        f"{reduction:.2f}%"
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame({
        "random_shortcut_rate": random_rates,
        "block_shortcut_rate": block_rates,
        "random_shortcut_pairs": random_shortcut_pairs,
        "block_shortcut_pairs": block_shortcut_pairs
    })

    output_path = (
        "preprocessing/masking/"
        "shortcut_experiment_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
    