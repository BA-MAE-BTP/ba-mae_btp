# ============================================================
# BLOCK-AWARE MASKING CONFIGURATION
# ============================================================

# Path to frozen dependency blocks
BLOCK_PATH = (
    "preprocessing/dependency/"
    "refined_candidate_blocks.csv"
)

# Initial masking ratio for the baseline
BASE_MASKING_RATIO = 0.25

# Value used to replace masked features
MASK_VALUE = 0.0

# Random seed for reproducibility
RANDOM_SEED = 42

# Number of model features
NUM_FEATURES = 39