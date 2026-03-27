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
    - Clears existing data before copying (fresh start)
    - Validates source directories exist
    - Uses seeded random selection for reproducibility
    - Supports SAMPLE_SIZE=None for using all available images
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

    # Clear existing data for a fresh start
    for target_dir in [TARGET_IMAGES, TARGET_MASKS]:
        if os.path.exists(target_dir):
            existing = [f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))]
            if existing:
                print(f"\nClearing {len(existing)} existing files from {target_dir}...")
                for f in existing:
                    os.remove(os.path.join(target_dir, f))

    # Create target directories
    os.makedirs(TARGET_IMAGES, exist_ok=True)
    os.makedirs(TARGET_MASKS, exist_ok=True)

    # Get all image files
    print("\nScanning source directory...")
    image_files = [f for f in os.listdir(SOURCE_IMAGES) if f.endswith(".jpg")]
    print(f"Found {len(image_files)} total images in source")

    if len(image_files) == 0:
        raise ValueError("No .jpg images found in source directory")

    # First, find all valid image-mask pairs
    print("Finding all valid image-mask pairs...")
    all_valid_pairs = []
    for img_name in image_files:
        base = img_name.replace(".jpg", "")
        mask_name = base + "_segmentation.png"
        src_mask = os.path.join(SOURCE_MASKS, mask_name)
        if os.path.exists(src_mask):
            all_valid_pairs.append((img_name, mask_name))

    print(f"Found {len(all_valid_pairs)} images with matching masks")

    # Determine how many to use
    if SAMPLE_SIZE is None:
        selected_pairs = all_valid_pairs
        print(f"\nUsing ALL {len(selected_pairs)} valid image-mask pairs")
    else:
        if SAMPLE_SIZE > len(all_valid_pairs):
            print(f"\nWarning: Requested {SAMPLE_SIZE} but only {len(all_valid_pairs)} valid pairs available.")
            print(f"Using all {len(all_valid_pairs)} valid pairs instead.")
            selected_pairs = all_valid_pairs
        else:
            # Set random seed for reproducibility
            random.seed(config.RANDOM_SEED)
            selected_pairs = random.sample(all_valid_pairs, SAMPLE_SIZE)
            print(f"\nSelected {SAMPLE_SIZE} random image-mask pairs (seed={config.RANDOM_SEED})")

    # Copy files with progress reporting
    print(f"\nCopying {len(selected_pairs)} image-mask pairs...")
    copied = 0

    for i, (img_name, mask_name) in enumerate(selected_pairs, 1):
        src_img = os.path.join(SOURCE_IMAGES, img_name)
        src_mask = os.path.join(SOURCE_MASKS, mask_name)

        dst_img = os.path.join(TARGET_IMAGES, img_name)
        dst_mask = os.path.join(TARGET_MASKS, mask_name)

        shutil.copy(src_img, dst_img)
        shutil.copy(src_mask, dst_mask)
        copied += 1

        # Progress reporting every 100 files
        if i % 100 == 0 or i == len(selected_pairs):
            print(f"  Progress: {i}/{len(selected_pairs)} pairs copied")

    # Summary
    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Images copied: {copied}")
    print(f"Masks copied: {copied}")
    print(f"Target directory: {TARGET_IMAGES}")
    print(f"\nWith {config.TRAIN_RATIO}/{config.VAL_RATIO}/{config.TEST_RATIO} split:")
    print(f"  - Training: ~{int(copied * config.TRAIN_RATIO)} images")
    print(f"  - Validation: ~{int(copied * config.VAL_RATIO)} images")
    print(f"  - Test: ~{int(copied * config.TEST_RATIO)} images")
    print("=" * 70)


if __name__ == "__main__":
    try:
        prepare_dataset()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        exit(1)
