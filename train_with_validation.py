"""
Improved Training Script with Train/Val Split
Uses config.py and dataset.py modules
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config
from segmentation_model import UNet
from dataset import SimplePsoriasisDataset, get_dataset_stats

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
optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Learning rate scheduler for better convergence
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)

# Training history
train_losses = []
val_losses = []
best_val_loss = float("inf")

# Early stopping parameters
patience = 7
patience_counter = 0

print("\n" + "=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(config.EPOCHS):
    # ========== TRAINING ==========
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)

        preds = model(imgs)
        loss = loss_fn(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ========== VALIDATION ==========
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(config.DEVICE), masks.to(config.DEVICE)

            preds = model(imgs)
            loss = loss_fn(preds, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    # Print progress
    print(
        f"Epoch {epoch + 1}/{config.EPOCHS} - "
        f"Train Loss: {avg_train_loss:.4f} - "
        f"Val Loss: {avg_val_loss:.4f}"
    )

    # Learning rate scheduling
    scheduler.step(avg_val_loss)

    # Save best model and early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_path = config.get_model_path("best_unet_model.pth")
        torch.save(model.state_dict(), best_model_path)
        print(f"  → Best model saved! (Val Loss: {best_val_loss:.4f})")
    else:
        patience_counter += 1
        print(f"  → No improvement (patience: {patience_counter}/{patience})")

    # Early stopping check
    if patience_counter >= patience:
        print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
        print(f"Best validation loss: {best_val_loss:.4f}")
        break

# Save final model
final_model_path = config.get_model_path("unet_model.pth")
torch.save(model.state_dict(), final_model_path)
print(f"\nFinal model saved to: {final_model_path}")
print(f"Best model saved to: {config.get_model_path('best_unet_model.pth')}")

# Plot training curves
print("\nGenerating training curves...")
plt.figure(figsize=(10, 5))

plt.plot(range(1, config.EPOCHS + 1), train_losses, label="Train Loss", marker="o")
plt.plot(range(1, config.EPOCHS + 1), val_losses, label="Val Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)

# Save plot
plot_path = os.path.join(config.LOG_DIR, "training_curves.png")
plt.savefig(plot_path)
print(f"Training curves saved to: {plot_path}")
plt.show()

# ========== VISUAL TEST ==========
print("\n" + "=" * 70)
print("VISUAL TEST")
print("=" * 70)

model.eval()
img, mask = val_dataset[0]

with torch.no_grad():
    pred_logits = model(img.unsqueeze(0).to(config.DEVICE)).cpu().numpy()[0, 0]
    # Apply sigmoid to convert logits to probabilities (same as inference)
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

# Save test visualization
test_plot_path = os.path.join(config.LOG_DIR, "test_prediction.png")
plt.savefig(test_plot_path)
print(f"\nTest prediction saved to: {test_plot_path}")
plt.show()

print("\n" + "=" * 70)
print("TRAINING COMPLETE!")
print("=" * 70)
print(f"Best Validation Loss: {best_val_loss:.4f}")
print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Val Loss: {val_losses[-1]:.4f}")
print("=" * 70)
