"""
Improved Training Script with Train/Val Split
Uses Dice+BCE combined loss, cosine annealing, and deeper U-Net
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import config
from segmentation_model import UNet
from dataset import SimplePsoriasisDataset, get_dataset_stats

# Use augmented dataset if albumentations is available
try:
    from dataset import PsoriasisDataset, ALBUMENTATIONS_AVAILABLE
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False


# ============================================================================
# DICE + BCE COMBINED LOSS (improves segmentation metrics significantly)
# ============================================================================
class DiceBCELoss(torch.nn.Module):
    """Combined Dice Loss and BCE Loss for better segmentation training."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        # BCE component
        bce_loss = self.bce(predictions, targets)

        # Dice component (apply sigmoid first)
        probs = torch.sigmoid(predictions)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        # Combined loss (equal weighting)
        return bce_loss + dice_loss


# ============================================================================
# DICE METRIC (for monitoring during training)
# ============================================================================
def compute_dice(predictions, targets, threshold=0.5):
    """Compute Dice coefficient for a batch."""
    probs = torch.sigmoid(predictions)
    pred_binary = (probs > threshold).float()
    intersection = (pred_binary * targets).sum()
    union = pred_binary.sum() + targets.sum()
    if union == 0:
        return 1.0
    return (2.0 * intersection / union).item()


# ============================================================================
# SETUP
# ============================================================================
print("=" * 70)
print("PSORIASIS SEGMENTATION - TRAINING WITH VALIDATION")
print("=" * 70)

# Print dataset statistics
stats = get_dataset_stats()
print("\nDataset Statistics:")
for key, value in stats.items():
    print(f"  {key}: {value}")

# Create datasets
print("\nLoading datasets...")
if ALBUMENTATIONS_AVAILABLE:
    print("Using PsoriasisDataset with augmentation")
    train_dataset = PsoriasisDataset(split="train")
    val_dataset = PsoriasisDataset(split="val", use_augmentation=False)
else:
    print("Albumentations not available, using SimplePsoriasisDataset (no augmentation)")
    train_dataset = SimplePsoriasisDataset(split="train")
    val_dataset = SimplePsoriasisDataset(split="val")

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# Initialize model
print(f"\nInitializing model on device: {config.DEVICE}")
model = UNet().to(config.DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
)

# Choose loss function
if config.LOSS_FUNCTION == "dice_bce":
    loss_fn = DiceBCELoss()
    print("Loss function: Dice + BCE Combined Loss")
else:
    loss_fn = torch.nn.BCEWithLogitsLoss()
    print("Loss function: BCEWithLogitsLoss")

# Cosine Annealing scheduler (smoother LR decay, better convergence)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.EPOCHS, eta_min=1e-6
)

# Training history
train_losses = []
val_losses = []
train_dice_scores = []
val_dice_scores = []
best_val_loss = float("inf")
best_val_dice = 0.0

# Early stopping parameters
patience = 10
patience_counter = 0

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(config.EPOCHS):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0
    train_dice = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_dice += compute_dice(preds, masks)

    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    train_losses.append(avg_train_loss)
    train_dice_scores.append(avg_train_dice)

    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0
    val_dice = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            val_loss += loss.item()
            val_dice += compute_dice(preds, masks)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    val_losses.append(avg_val_loss)
    val_dice_scores.append(avg_val_dice)

    # Step scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']

    # Print progress
    print(
        f"Epoch {epoch + 1}/{config.EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - "
        f"Train Dice: {avg_train_dice:.4f} - Val Dice: {avg_val_dice:.4f} - "
        f"LR: {current_lr:.6f}"
    )

    # Save best model (based on val dice for better metric optimization)
    if avg_val_dice > best_val_dice:
        best_val_dice = avg_val_dice
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_path = config.get_model_path("best_unet_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"  → Best model saved! (Val Dice: {best_val_dice:.4f})")
    else:
        patience_counter += 1
        if patience_counter > 0 and patience_counter % 3 == 0:
            print(f"  → No improvement (patience: {patience_counter}/{patience})")

    # Early stopping check
    if patience_counter >= patience:
        print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
        print(f"Best validation Dice: {best_val_dice:.4f}")
        break

# Save final model
final_model_path = config.get_model_path("unet_model.pth")
torch.save(model.state_dict(), final_model_path)

# Also save to legacy path for app.py compatibility
legacy_path = os.path.join(config.PROJECT_ROOT, "unet_model.pth")
torch.save(model.state_dict(), legacy_path)

print(f"\nFinal model saved to: {final_model_path}")
print(f"Legacy model saved to: {legacy_path}")
print(f"Best model saved to: {config.get_model_path('best_unet_model.pth')}")

# ========== PLOT TRAINING CURVES ==========
print("\nGenerating training curves...")
epochs_range = range(1, len(train_losses) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(epochs_range, train_losses, label="Train Loss", marker="o", markersize=3)
ax1.plot(epochs_range, val_losses, label="Val Loss", marker="s", markersize=3)
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(True)

# Dice plot
ax2.plot(epochs_range, train_dice_scores, label="Train Dice", marker="o", markersize=3)
ax2.plot(epochs_range, val_dice_scores, label="Val Dice", marker="s", markersize=3)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Dice Coefficient")
ax2.set_title("Training and Validation Dice")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plot_path = os.path.join(config.LOG_DIR, "training_curves.png")
plt.savefig(plot_path, dpi=150)
print(f"Training curves saved to: {plot_path}")
plt.close()

# ========== VISUAL TEST ==========
print("\n" + "=" * 70)
print("VISUAL TEST")
print("=" * 70)

# Load best model for visual test
model.load_state_dict(torch.load(best_model_path, map_location=config.DEVICE, weights_only=False))
model.eval()

img, mask = val_dataset[0]

with torch.no_grad():
    pred_logits = model(img.unsqueeze(0).to(config.DEVICE)).cpu().numpy()[0, 0]
    pred = torch.sigmoid(torch.from_numpy(pred_logits)).numpy()

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
plt.title("Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask[0], cmap="gray")
plt.title("Ground Truth")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred, cmap="gray")
plt.title("Prediction")
plt.axis("off")

test_plot_path = os.path.join(config.LOG_DIR, "test_prediction.png")
plt.savefig(test_plot_path, dpi=150)
print(f"\nTest prediction saved to: {test_plot_path}")
plt.close()

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best Validation Dice: {best_val_dice:.4f}")
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Val Loss: {val_losses[-1]:.4f}")
print(f"Final Train Dice: {train_dice_scores[-1]:.4f}")
print(f"Final Val Dice: {val_dice_scores[-1]:.4f}")
print("=" * 70)
