"""
Centralized Configuration for Psoriasis Segmentation System
All paths, hyperparameters, and constants are defined here.
"""

import os
import torch

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")
MASK_DIR = os.path.join(DATA_DIR, "masks")

# Model directories
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Legacy model path (for backward compatibility)
LEGACY_MODEL_PATH = os.path.join(PROJECT_ROOT, "unet_model.pth")

# Default model path
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "unet_model.pth")

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

# Image settings
IMG_SIZE = 256
IMG_CHANNELS = 3

# Dataset split ratios (train/val/test)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42

# Data augmentation settings
USE_AUGMENTATION = True
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": True,
    "rotation_range": 15,  # degrees
    "brightness_range": 0.1,
    "contrast_range": 0.1,
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Training hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 30  # Increased for larger dataset (400 images)
NUM_WORKERS = 0  # Set to 0 for Windows compatibility

# Optimizer settings
OPTIMIZER = "adam"
WEIGHT_DECAY = 1e-5

# Loss function
LOSS_FUNCTION = "bce_with_logits"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# U-Net architecture
UNET_CONFIG = {
    "in_channels": IMG_CHANNELS,
    "out_channels": 1,
    "init_features": 64,
}

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================

# Segmentation threshold
SEGMENTATION_THRESHOLD = 0.5

# ============================================================================
# EXTERNAL DATASET PATHS (Optional - for dataset preparation)
# ============================================================================

# These paths are used by dataset_prepare.py to copy data from external sources
# Modify these to match your dataset location
EXTERNAL_SOURCE_IMAGES = r"F:\archive\ISIC2018_Task1-2_Training_Input"
EXTERNAL_SOURCE_MASKS = r"F:\archive\ISIC2018_Task1_Training_GroundTruth"

# Number of samples to prepare (set to None for all available)
# 400 images provides good balance: 280 train / 60 val / 60 test (70/15/15 split)
SAMPLE_SIZE = 400  # Change to None to use full dataset

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

# Streamlit app settings
STREAMLIT_CONFIG = {
    "title": "Psoriasis Lesion Segmentation",
    "upload_types": ["jpg", "png", "jpeg"],
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_model_path(model_name=None):
    """
    Get the path to a model file.

    Args:
        model_name: Name of the model file. If None, returns default path.
                   If the name exists in legacy location, returns that path.

    Returns:
        Path to the model file
    """
    if model_name is None:
        # Check if legacy model exists
        if os.path.exists(LEGACY_MODEL_PATH):
            return LEGACY_MODEL_PATH
        return DEFAULT_MODEL_PATH

    # Check legacy location first for backward compatibility
    legacy_path = os.path.join(PROJECT_ROOT, model_name)
    if os.path.exists(legacy_path):
        return legacy_path

    # Otherwise use models directory
    return os.path.join(MODEL_DIR, model_name)


def print_config():
    """Print current configuration settings."""
    print("=" * 70)
    print("PSORIASIS SEGMENTATION SYSTEM - CONFIGURATION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"\nData Directory: {DATA_DIR}")
    print(f"  - Images: {IMAGE_DIR}")
    print(f"  - Masks: {MASK_DIR}")
    print(f"\nModel Directory: {MODEL_DIR}")
    print(f"  - Default Model: {get_model_path()}")
    print(f"\nDataset Configuration:")
    print(f"  - Image Size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - Train/Val/Test Split: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"  - Augmentation: {'Enabled' if USE_AUGMENTATION else 'Disabled'}")
    print(f"\nTraining Configuration:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Learning Rate: {LEARNING_RATE}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Device: {DEVICE}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()
