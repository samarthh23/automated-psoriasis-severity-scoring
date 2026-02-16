"""
Enhanced Dataset Module for Psoriasis Segmentation
Supports train/val/test splits, data augmentation, and efficient loading.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, random_split

# Try to import albumentations (optional dependency)
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    print(
        "Warning: albumentations not installed. Using SimplePsoriasisDataset instead."
    )
    print("Install with: pip install albumentations")

import config


class PsoriasisDataset(Dataset):
    """
    Enhanced dataset class for psoriasis lesion segmentation.

    Features:
    - Automatic train/val/test splitting
    - Optional data augmentation
    - Proper error handling
    - Efficient loading
    """

    def __init__(self, split="train", use_augmentation=None, transform=None):
        """
        Initialize the dataset.

        Args:
            split: One of "train", "val", "test", or "all"
            use_augmentation: Whether to use data augmentation (default: config.USE_AUGMENTATION for train)
            transform: Custom albumentations transform (optional)
        """
        self.split = split
        self.img_size = config.IMG_SIZE

        # Set augmentation
        if use_augmentation is None:
            self.use_augmentation = config.USE_AUGMENTATION and (split == "train")
        else:
            self.use_augmentation = use_augmentation

        # Load all image filenames
        if not os.path.exists(config.IMAGE_DIR):
            raise ValueError(f"Image directory not found: {config.IMAGE_DIR}")

        all_images = sorted(
            [
                f
                for f in os.listdir(config.IMAGE_DIR)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if len(all_images) == 0:
            raise ValueError(f"No images found in {config.IMAGE_DIR}")

        # Split dataset
        self.image_files = self._split_dataset(all_images, split)

        print(f"Loaded {len(self.image_files)} images for {split} split")

        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_transforms()

    def _split_dataset(self, all_images, split):
        """
        Split dataset into train/val/test based on config ratios.

        Args:
            all_images: List of all image filenames
            split: Desired split ("train", "val", "test", or "all")

        Returns:
            List of image filenames for the requested split
        """
        if split == "all":
            return all_images

        # Set random seed for reproducibility
        np.random.seed(config.RANDOM_SEED)

        # Shuffle images
        indices = np.random.permutation(len(all_images))

        # Calculate split sizes
        n_total = len(all_images)
        n_train = int(n_total * config.TRAIN_RATIO)
        n_val = int(n_total * config.VAL_RATIO)

        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        # Return appropriate split
        if split == "train":
            return [all_images[i] for i in train_indices]
        elif split == "val":
            return [all_images[i] for i in val_indices]
        elif split == "test":
            return [all_images[i] for i in test_indices]
        else:
            raise ValueError(
                f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'all'"
            )

    def _get_transforms(self):
        """
        Get albumentations transforms based on configuration.

        Returns:
            Albumentations Compose transform
        """
        if self.use_augmentation:
            aug_config = config.AUGMENTATION_CONFIG

            transform_list = [
                A.Resize(self.img_size, self.img_size),
            ]

            # Add augmentations
            if aug_config.get("horizontal_flip", False):
                transform_list.append(A.HorizontalFlip(p=0.5))

            if aug_config.get("vertical_flip", False):
                transform_list.append(A.VerticalFlip(p=0.5))

            if aug_config.get("rotation_range", 0) > 0:
                rotation_limit = aug_config["rotation_range"]
                transform_list.append(A.Rotate(limit=rotation_limit, p=0.5))

            if aug_config.get("brightness_range", 0) > 0:
                brightness_limit = aug_config["brightness_range"]
                transform_list.append(
                    A.RandomBrightnessContrast(
                        brightness_limit=brightness_limit,
                        contrast_limit=aug_config.get("contrast_range", 0),
                        p=0.5,
                    )
                )

            # Normalize
            transform_list.append(
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            )

        else:
            # No augmentation - just resize and normalize
            transform_list = [
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]

        return A.Compose(transform_list)

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        img_name = self.image_files[idx]

        # Construct paths
        img_path = os.path.join(config.IMAGE_DIR, img_name)

        # Get corresponding mask name
        base_name = (
            img_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        )
        mask_name = base_name + "_segmentation.png"
        mask_path = os.path.join(config.MASK_DIR, mask_name)

        # Load image
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError(f"Failed to load image: {img_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise ValueError(f"Error loading image {img_path}: {str(e)}")

        # Load mask
        try:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {mask_path}")
        except Exception as e:
            raise ValueError(f"Error loading mask {mask_path}: {str(e)}")

        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        image = transformed["image"]
        mask = transformed["mask"]

        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float() / 255.0

        return image, mask


class SimplePsoriasisDataset(Dataset):
    """
    Simplified dataset class for backward compatibility.
    Uses simple normalization without albumentations.
    """

    def __init__(self, split="train"):
        """
        Initialize the simple dataset.

        Args:
            split: One of "train", "val", "test", or "all"
        """
        self.split = split
        self.img_size = config.IMG_SIZE

        # Load all image filenames
        if not os.path.exists(config.IMAGE_DIR):
            raise ValueError(f"Image directory not found: {config.IMAGE_DIR}")

        all_images = sorted(
            [
                f
                for f in os.listdir(config.IMAGE_DIR)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        if len(all_images) == 0:
            raise ValueError(f"No images found in {config.IMAGE_DIR}")

        # Split dataset
        self.image_files = self._split_dataset(all_images, split)

        print(f"Loaded {len(self.image_files)} images for {split} split (simple mode)")

    def _split_dataset(self, all_images, split):
        """Split dataset into train/val/test."""
        if split == "all":
            return all_images

        np.random.seed(config.RANDOM_SEED)
        indices = np.random.permutation(len(all_images))

        n_total = len(all_images)
        n_train = int(n_total * config.TRAIN_RATIO)
        n_val = int(n_total * config.VAL_RATIO)

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        if split == "train":
            return [all_images[i] for i in train_indices]
        elif split == "val":
            return [all_images[i] for i in val_indices]
        elif split == "test":
            return [all_images[i] for i in test_indices]
        else:
            raise ValueError(f"Invalid split: {split}")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        img_name = self.image_files[idx]

        # Construct paths
        img_path = os.path.join(config.IMAGE_DIR, img_name)
        base_name = (
            img_name.replace(".jpg", "").replace(".jpeg", "").replace(".png", "")
        )
        mask_name = base_name + "_segmentation.png"
        mask_path = os.path.join(config.MASK_DIR, mask_name)

        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = np.transpose(img, (2, 0, 1))

        # Load and preprocess mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = mask / 255.0
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(img, dtype=torch.float32), torch.tensor(
            mask, dtype=torch.float32
        )


def get_dataloaders(batch_size=None, num_workers=None, use_simple=False):
    """
    Create train, validation, and test dataloaders.

    Args:
        batch_size: Batch size (default: config.BATCH_SIZE)
        num_workers: Number of workers (default: config.NUM_WORKERS)
        use_simple: Use SimplePsoriasisDataset instead of PsoriasisDataset

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    if num_workers is None:
        num_workers = config.NUM_WORKERS

    # Choose dataset class
    DatasetClass = SimplePsoriasisDataset if use_simple else PsoriasisDataset

    # Create datasets
    train_dataset = DatasetClass(split="train")
    val_dataset = DatasetClass(split="val")
    test_dataset = DatasetClass(split="test")

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == "cuda" else False,
    )

    return train_loader, val_loader, test_loader


def get_dataset_stats():
    """
    Get statistics about the dataset.

    Returns:
        Dictionary with dataset statistics
    """
    try:
        all_images = sorted(
            [
                f
                for f in os.listdir(config.IMAGE_DIR)
                if f.endswith((".jpg", ".jpeg", ".png"))
            ]
        )

        n_total = len(all_images)
        n_train = int(n_total * config.TRAIN_RATIO)
        n_val = int(n_total * config.VAL_RATIO)
        n_test = n_total - n_train - n_val

        return {
            "total": n_total,
            "train": n_train,
            "val": n_val,
            "test": n_test,
            "split_ratios": f"{config.TRAIN_RATIO}/{config.VAL_RATIO}/{config.TEST_RATIO}",
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    # Test the dataset
    print("=" * 70)
    print("DATASET MODULE TEST")
    print("=" * 70)

    # Print dataset statistics
    stats = get_dataset_stats()
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Test loading datasets
    print("\nTesting dataset loading...")
    try:
        # Try with albumentations first
        try:
            train_ds = PsoriasisDataset(split="train")
            val_ds = PsoriasisDataset(split="val")
            test_ds = PsoriasisDataset(split="test")
            print("✓ PsoriasisDataset (with augmentation) loaded successfully")
        except ImportError:
            print("⚠ Albumentations not installed, using SimplePsoriasisDataset")
            train_ds = SimplePsoriasisDataset(split="train")
            val_ds = SimplePsoriasisDataset(split="val")
            test_ds = SimplePsoriasisDataset(split="test")

        # Test loading a sample
        if len(train_ds) > 0:
            img, mask = train_ds[0]
            print(f"\nSample from train set:")
            print(f"  Image shape: {img.shape}")
            print(f"  Mask shape: {mask.shape}")
            print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")
            print(f"  Mask range: [{mask.min():.3f}, {mask.max():.3f}]")

        print("\n✓ All tests passed!")

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback

        traceback.print_exc()

    print("=" * 70)
