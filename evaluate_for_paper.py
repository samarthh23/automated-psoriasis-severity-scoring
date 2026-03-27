"""
Evaluation Script for Implementation Paper
Computes segmentation metrics, generates sample outputs, and measures inference time.

Run: python evaluate_for_paper.py
Output: logs/paper_metrics.txt + logs/paper_samples/
"""

import os
import sys
import time
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import config
from segmentation_model import UNet
from dataset import SimplePsoriasisDataset, get_dataset_stats

# Try importing Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False
    print("Warning: pytorch_grad_cam not installed. Grad-CAM outputs will be skipped.")

# ============================================================================
# CONFIGURATION
# ============================================================================
THRESHOLD = config.SEGMENTATION_THRESHOLD  # 0.5
NUM_SAMPLES = 5  # Number of sample outputs to generate
OUTPUT_DIR = os.path.join(config.LOG_DIR, "paper_samples")
METRICS_FILE = os.path.join(config.LOG_DIR, "paper_metrics.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Force CPU for consistent inference time measurement
DEVICE = "cpu"


# ============================================================================
# METRIC FUNCTIONS
# ============================================================================
def dice_coefficient(pred_mask, gt_mask):
    """Compute Dice coefficient between predicted and ground truth binary masks."""
    intersection = np.sum(pred_mask * gt_mask)
    if np.sum(pred_mask) + np.sum(gt_mask) == 0:
        return 1.0  # Both empty = perfect match
    return (2.0 * intersection) / (np.sum(pred_mask) + np.sum(gt_mask))


def iou_score(pred_mask, gt_mask):
    """Compute Intersection over Union between predicted and ground truth binary masks."""
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask) - intersection
    if union == 0:
        return 1.0  # Both empty = perfect match
    return intersection / union


# ============================================================================
# LOAD MODEL
# ============================================================================
print("=" * 70)
print("PSORIASIS SEGMENTATION - PAPER EVALUATION")
print("=" * 70)

MODEL_PATH = config.get_model_path()
print(f"\nLoading model from: {MODEL_PATH}")
print(f"Device: {DEVICE}")

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
model.eval()
print("Model loaded successfully.\n")


# ============================================================================
# 1. DATASET INFO
# ============================================================================
print("=" * 70)
print("SECTION 1: DATASET INFORMATION")
print("=" * 70)

stats = get_dataset_stats()
total = stats["total"]
train_count = stats["train"]
val_count = stats["val"]
test_count = stats["test"]

dataset_info = f"""
Total images: {total}
Train: {train_count}
Validation: {val_count}
Test: {test_count}
Split ratio: {stats['split_ratios']}
Image size: {config.IMG_SIZE}x{config.IMG_SIZE}
Dataset: ISIC 2018 Task 1 (subset)
"""
print(dataset_info)


# ============================================================================
# 2. SEGMENTATION METRICS ON TEST SET
# ============================================================================
print("=" * 70)
print("SECTION 2: SEGMENTATION METRICS (Test Set)")
print("=" * 70)

test_dataset = SimplePsoriasisDataset(split="test")
print(f"Evaluating on {len(test_dataset)} test images...\n")

dice_scores = []
iou_scores = []
inference_times = []

for i in range(len(test_dataset)):
    img_tensor, mask_tensor = test_dataset[i]

    # Measure inference time
    start_time = time.time()
    with torch.no_grad():
        pred_logits = model(img_tensor.unsqueeze(0).to(DEVICE))[0, 0].cpu().numpy()
        pred_prob = torch.sigmoid(torch.from_numpy(pred_logits)).numpy()
    elapsed = time.time() - start_time
    inference_times.append(elapsed)

    # Binary masks
    pred_binary = (pred_prob > THRESHOLD).astype(np.float32)
    gt_binary = (mask_tensor[0].numpy() > 0.5).astype(np.float32)

    # Compute metrics
    dice = dice_coefficient(pred_binary, gt_binary)
    iou = iou_score(pred_binary, gt_binary)
    dice_scores.append(dice)
    iou_scores.append(iou)

    # Progress
    if (i + 1) % 10 == 0 or i == len(test_dataset) - 1:
        print(f"  Processed {i+1}/{len(test_dataset)} images...")

# Compute summary statistics
avg_dice = np.mean(dice_scores)
std_dice = np.std(dice_scores)
min_dice = np.min(dice_scores)
max_dice = np.max(dice_scores)

avg_iou = np.mean(iou_scores)
std_iou = np.std(iou_scores)
min_iou = np.min(iou_scores)
max_iou = np.max(iou_scores)

metrics_text = f"""
Test Images: {len(test_dataset)}
Threshold: {THRESHOLD}

--- Dice Coefficient ---
Average Dice: {avg_dice:.4f}
Std Dice:     {std_dice:.4f}
Best Dice:    {max_dice:.4f}
Worst Dice:   {min_dice:.4f}

--- IoU (Jaccard Index) ---
Average IoU:  {avg_iou:.4f}
Std IoU:      {std_iou:.4f}
Best IoU:     {max_iou:.4f}
Worst IoU:    {min_iou:.4f}
"""
print(metrics_text)


# ============================================================================
# 3. INFERENCE TIME
# ============================================================================
print("=" * 70)
print("SECTION 3: INFERENCE TIME")
print("=" * 70)

avg_time = np.mean(inference_times)
min_time = np.min(inference_times)
max_time = np.max(inference_times)

time_text = f"""
Device: CPU
Average inference time: {avg_time:.4f} sec/image
Min inference time:     {min_time:.4f} sec/image
Max inference time:     {max_time:.4f} sec/image
"""
print(time_text)


# ============================================================================
# 4. SAMPLE OUTPUTS
# ============================================================================
print("=" * 70)
print("SECTION 4: SAMPLE OUTPUTS")
print("=" * 70)

# Pick sample indices (evenly spaced for variety)
np.random.seed(config.RANDOM_SEED)
sample_indices = np.random.choice(len(test_dataset), size=min(NUM_SAMPLES, len(test_dataset)), replace=False)
sample_indices.sort()

print(f"Generating {len(sample_indices)} sample outputs...")

for idx_num, idx in enumerate(sample_indices):
    img_tensor, mask_tensor = test_dataset[idx]
    img_name = test_dataset.image_files[idx]

    # Run inference
    with torch.no_grad():
        pred_logits = model(img_tensor.unsqueeze(0).to(DEVICE))[0, 0].cpu().numpy()
        pred_prob = torch.sigmoid(torch.from_numpy(pred_logits)).numpy()

    pred_binary = (pred_prob > THRESHOLD).astype(np.uint8)
    gt_binary = (mask_tensor[0].numpy() > 0.5).astype(np.uint8)

    # Recover displayable image (undo /255 normalization)
    img_display = np.transpose(img_tensor.numpy(), (1, 2, 0))
    img_display = np.clip(img_display, 0, 1)

    # Create overlay
    overlay = img_display.copy()
    overlay[pred_binary == 1] = [1.0, 0.0, 0.0]  # Red overlay on predictions
    blended = cv2.addWeighted(
        (img_display * 255).astype(np.uint8), 0.6,
        (overlay * 255).astype(np.uint8), 0.4, 0
    )

    # Compute per-image metrics
    d = dice_coefficient(pred_binary.astype(np.float32), gt_binary.astype(np.float32))
    j = iou_score(pred_binary.astype(np.float32), gt_binary.astype(np.float32))

    # Determine number of columns
    n_cols = 5 if GRADCAM_AVAILABLE else 4

    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    fig.suptitle(f"Sample {idx_num+1}: {img_name}  |  Dice: {d:.4f}  |  IoU: {j:.4f}", fontsize=12, fontweight='bold')

    axes[0].imshow(img_display)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(gt_binary, cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    axes[2].imshow(pred_binary, cmap="gray")
    axes[2].set_title("Predicted Mask")
    axes[2].axis("off")

    axes[3].imshow(blended)
    axes[3].set_title("Overlay")
    axes[3].axis("off")

    # Grad-CAM
    if GRADCAM_AVAILABLE:
        target_layers = [model.conv1.conv[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        def segmentation_target(output):
            return output.mean()

        input_tensor = img_tensor.unsqueeze(0).to(DEVICE)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[segmentation_target])[0]
        heatmap = show_cam_on_image(img_display.astype(np.float32), grayscale_cam, use_rgb=True)

        axes[4].imshow(heatmap)
        axes[4].set_title("Grad-CAM")
        axes[4].axis("off")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"sample_{idx_num+1}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

print(f"\nAll sample outputs saved to: {OUTPUT_DIR}")


# ============================================================================
# 5. TRAINING LOSS REFERENCE
# ============================================================================
print("\n" + "=" * 70)
print("SECTION 5: TRAINING LOSS")
print("=" * 70)

training_curves_path = os.path.join(config.LOG_DIR, "training_curves.png")
if os.path.exists(training_curves_path):
    print(f"\nTraining curves plot available at: {training_curves_path}")
    print("Refer to this image for final training and validation loss values.")
else:
    print("\nNo training curves found. Run training script to generate them.")

loss_text = """
Note: The training script (train_with_validation.py) printed the final
train/val loss at the end of training. If you have those values, add them here.
You can also refer to: logs/training_curves.png
"""
print(loss_text)


# ============================================================================
# 6. SAVE ALL METRICS TO FILE
# ============================================================================
print("=" * 70)
print("SAVING RESULTS")
print("=" * 70)

with open(METRICS_FILE, "w") as f:
    f.write("=" * 70 + "\n")
    f.write("PSORIASIS SEGMENTATION - PAPER EVALUATION METRICS\n")
    f.write("=" * 70 + "\n\n")

    f.write("DATASET INFORMATION\n")
    f.write("-" * 40 + "\n")
    f.write(dataset_info + "\n")

    f.write("SEGMENTATION METRICS (Test Set)\n")
    f.write("-" * 40 + "\n")
    f.write(metrics_text + "\n")

    f.write("INFERENCE TIME\n")
    f.write("-" * 40 + "\n")
    f.write(time_text + "\n")

    f.write("SAMPLE OUTPUTS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Sample images saved to: {OUTPUT_DIR}\n")
    f.write(f"Number of samples: {len(sample_indices)}\n\n")

    f.write("TRAINING LOSS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Training curves: {training_curves_path}\n")
    f.write("Add final train/val loss values from training output here.\n\n")

    f.write("MODEL INFORMATION\n")
    f.write("-" * 40 + "\n")
    f.write(f"Architecture: U-Net\n")
    f.write(f"Input size: {config.IMG_SIZE}x{config.IMG_SIZE}x3\n")
    f.write(f"Output: 1-channel binary mask\n")
    f.write(f"Loss function: {config.LOSS_FUNCTION}\n")
    f.write(f"Optimizer: Adam (lr={config.LEARNING_RATE})\n")
    f.write(f"Epochs: {config.EPOCHS}\n")
    f.write(f"Batch size: {config.BATCH_SIZE}\n")
    f.write(f"Threshold: {THRESHOLD}\n")

    # Per-image results table
    f.write("\n\nPER-IMAGE RESULTS (Test Set)\n")
    f.write("-" * 40 + "\n")
    f.write(f"{'Image #':<10}{'Dice':<12}{'IoU':<12}{'Time (s)':<12}\n")
    f.write("-" * 46 + "\n")
    for i in range(len(test_dataset)):
        f.write(f"{i+1:<10}{dice_scores[i]:<12.4f}{iou_scores[i]:<12.4f}{inference_times[i]:<12.4f}\n")

print(f"\nAll metrics saved to: {METRICS_FILE}")
print("\n" + "=" * 70)
print("EVALUATION COMPLETE!")
print("=" * 70)
