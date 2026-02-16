"""
Enhanced Dataset Preparation Script
Copies images and masks from external ISIC2018 dataset to project data directory.
Features: validation, progress reporting, reproducible selection.
"""

import os
import shutil
import random

import config

# Source paths from config
SOURCE_IMAGES = config.EXTERNAL_SOURCE_IMAGES
SOURCE_MASKS = config.EXTERNAL_SOURCE_MASKS

# Target paths from config
TARGET_IMAGES = config.IMAGE_DIR
TARGET_MASKS = config.MASK_DIR

# Number of samples to prepare
SAMPLE_SIZE = config.SAMPLE_SIZE


def prepare_dataset():
    """
    Prepare dataset by copying images and masks from external source.

    Features:
    - Validates source directories exist
    - Uses seeded random selection for reproducibility
    - Verifies all selected images have corresponding masks
    - Reports progress during copying
    - Provides summary statistics
    """
    print("=" * 70)
    print("DATASET PREPARATION")
    print("=" * 70)

    # Validate source directories
    if not os.path.exists(SOURCE_IMAGES):
        raise ValueError(f"Source images directory not found: {SOURCE_IMAGES}")
    if not os.path.exists(SOURCE_MASKS):
        raise ValueError(f"Source masks directory not found: {SOURCE_MASKS}")

    print(f"\nSource Images: {SOURCE_IMAGES}")
    print(f"Source Masks: {SOURCE_MASKS}")
    print(f"Target Images: {TARGET_IMAGES}")
    print(f"Target Masks: {TARGET_MASKS}")

    # Create target directories
    os.makedirs(TARGET_IMAGES, exist_ok=True)
    os.makedirs(TARGET_MASKS, exist_ok=True)

    # Get all image files
    print("\nScanning source directory...")
    image_files = [f for f in os.listdir(SOURCE_IMAGES) if f.endswith(".jpg")]
    print(f"Found {len(image_files)} total images in source")

    if len(image_files) == 0:
        raise ValueError("No .jpg images found in source directory")

    if SAMPLE_SIZE > len(image_files):
        raise ValueError(
            f"Requested sample size ({SAMPLE_SIZE}) exceeds available images ({len(image_files)})"
        )

    # Set random seed for reproducibility
    random.seed(config.RANDOM_SEED)

    # Select random sample
    print(f"\nSelecting {SAMPLE_SIZE} random images (seed={config.RANDOM_SEED})...")
    selected = random.sample(image_files, SAMPLE_SIZE)

    # Validate that all selected images have corresponding masks
    print("Validating mask availability...")
    valid_pairs = []
    missing_masks = []

    for img_name in selected:
        base = img_name.replace(".jpg", "")
        mask_name = base + "_segmentation.png"
        src_mask = os.path.join(SOURCE_MASKS, mask_name)

        if os.path.exists(src_mask):
            valid_pairs.append((img_name, mask_name))
        else:
            missing_masks.append(img_name)

    if missing_masks:
        print(f"\nWarning: {len(missing_masks)} images have no corresponding masks:")
        for img in missing_masks[:5]:  # Show first 5
            print(f"  - {img}")
        if len(missing_masks) > 5:
            print(f"  ... and {len(missing_masks) - 5} more")

    print(f"\nValid image-mask pairs: {len(valid_pairs)}")

    # Copy files with progress reporting
    print(f"\nCopying {len(valid_pairs)} image-mask pairs...")
    copied_images = 0
    copied_masks = 0

    for i, (img_name, mask_name) in enumerate(valid_pairs, 1):
        src_img = os.path.join(SOURCE_IMAGES, img_name)
        src_mask = os.path.join(SOURCE_MASKS, mask_name)

        dst_img = os.path.join(TARGET_IMAGES, img_name)
        dst_mask = os.path.join(TARGET_MASKS, mask_name)

        # Copy files
        shutil.copy(src_img, dst_img)
        shutil.copy(src_mask, dst_mask)

        copied_images += 1
        copied_masks += 1

        # Progress reporting every 50 files
        if i % 50 == 0 or i == len(valid_pairs):
            print(f"  Progress: {i}/{len(valid_pairs)} pairs copied")

    # Summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Images copied: {copied_images}")
    print(f"Masks copied: {copied_masks}")
    print(f"Target directory: {TARGET_IMAGES}")
    print(f"\nWith {config.TRAIN_RATIO}/{config.VAL_RATIO}/{config.TEST_RATIO} split:")
    print(f"  - Training: ~{int(copied_images * config.TRAIN_RATIO)} images")
    print(f"  - Validation: ~{int(copied_images * config.VAL_RATIO)} images")
    print(f"  - Test: ~{int(copied_images * config.TEST_RATIO)} images")
    print("=" * 70)


if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)
