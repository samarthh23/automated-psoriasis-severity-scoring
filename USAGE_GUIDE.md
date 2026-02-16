# Quick Usage Guide - Getting Outputs

## 🚀 Option 1: Run the Streamlit Demo (Easiest)

This is the **fastest way** to get segmentation results with a visual interface.

### Step 1: Start the app

```bash
streamlit run app.py
```

### Step 2: Upload an image

- Click "Browse files" button
- Select a skin image (JPG/PNG)
- Wait for processing

### Step 3: View results

You'll see:

- **Original Image**
- **Segmentation Overlay** (red = lesion)
- **Severity Score** (percentage of affected area)
- **Grad-CAM Heatmap** (explainability)

---

## 🔬 Option 2: Use Python Script for Batch Processing

### Quick Inference Script

Create a file `quick_inference.py`:

```python
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

import config
from segmentation_model import UNet

# Load model
model = UNet().to(config.DEVICE)
model.load_state_dict(torch.load(config.get_model_path(), map_location=config.DEVICE))
model.eval()

# Load and preprocess image
image_path = "data/images/ISIC_0000000.jpg"  # Change this to your image
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

# Prepare for model
input_img = img / 255.0
tensor = np.transpose(input_img, (2, 0, 1))
tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

# Get prediction
with torch.no_grad():
    pred = model(tensor)[0, 0].cpu().numpy()

# Create binary mask
mask = (pred > config.SEGMENTATION_THRESHOLD).astype(np.uint8)

# Calculate severity
severity = (np.sum(mask) / mask.size) * 100

# Create overlay
overlay = img.copy()
overlay[mask == 1] = [255, 0, 0]  # Red for lesion

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 4, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.subplot(1, 4, 3)
plt.imshow(overlay)
plt.title(f"Overlay\nSeverity: {severity:.2f}%")
plt.axis("off")

plt.subplot(1, 4, 4)
plt.imshow(pred, cmap="hot")
plt.title("Prediction Heatmap")
plt.axis("off")

plt.tight_layout()
plt.savefig("output_result.png")
plt.show()

print(f"✓ Severity Score: {severity:.2f}%")
print(f"✓ Results saved to: output_result.png")
```

### Run it:

```bash
python quick_inference.py
```

---

## 📊 Option 3: Use the Severity Scorer Class

```python
from severity_model import SeverityScorer

# Initialize scorer
scorer = SeverityScorer()

# Analyze an image
image_path = "data/images/ISIC_0000000.jpg"
image, mask, severity = scorer.analyze(image_path)

print(f"Severity Score: {severity}%")

# Visualize
import matplotlib.pyplot as plt

overlay = image.copy()
overlay[mask == 1] = [255, 0, 0]

plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")

plt.subplot(1, 3, 3)
plt.imshow(overlay)
plt.title(f"Severity: {severity}%")

plt.show()
```

---

## 🎓 Option 4: Train a New Model with Validation

### Step 1: Install dependencies (if not done)

```bash
pip install -r requirements.txt
```

### Step 2: Run training

```bash
python train_with_validation.py
```

### What you'll get:

- Training progress printed to console
- Best model saved to `models/best_unet_model.pth`
- Final model saved to `models/unet_model.pth`
- Training curves saved to `logs/training_curves.png`
- Test prediction saved to `logs/test_prediction.png`

### Output example:

```
======================================================================
PSORIASIS SEGMENTATION - TRAINING WITH VALIDATION
======================================================================

Dataset Statistics:
  total: 80
  train: 56
  val: 12
  test: 12
  split_ratios: 0.7/0.15/0.15

Loading datasets...
Loaded 56 images for train split (simple mode)
Loaded 12 images for val split (simple mode)

Train batches: 14
Val batches: 3

Initializing model on device: cuda

======================================================================
TRAINING
======================================================================
Epoch 1/15 - Train Loss: 0.5234 - Val Loss: 0.4891
  → Best model saved! (Val Loss: 0.4891)
Epoch 2/15 - Train Loss: 0.4567 - Val Loss: 0.4523
  → Best model saved! (Val Loss: 0.4523)
...
```

---

## 🔍 Option 5: Test on Dataset Splits

```python
from dataset import SimplePsoriasisDataset
import torch
from segmentation_model import UNet
import config

# Load model
model = UNet().to(config.DEVICE)
model.load_state_dict(torch.load(config.get_model_path(), map_location=config.DEVICE))
model.eval()

# Load test dataset
test_dataset = SimplePsoriasisDataset(split="test")

print(f"Testing on {len(test_dataset)} images...")

# Test on first image
img, mask_true = test_dataset[0]

# Get prediction
with torch.no_grad():
    pred = model(img.unsqueeze(0).to(config.DEVICE))
    pred = pred[0, 0].cpu().numpy()

# Calculate metrics
pred_binary = (pred > 0.5).astype(float)
mask_true_np = mask_true[0].numpy()

# Intersection over Union (IoU)
intersection = (pred_binary * mask_true_np).sum()
union = ((pred_binary + mask_true_np) > 0).sum()
iou = intersection / union if union > 0 else 0

print(f"IoU Score: {iou:.4f}")
```

---

## 📁 Batch Processing Multiple Images

```python
import os
import cv2
import torch
import numpy as np
from pathlib import Path

import config
from segmentation_model import UNet

# Load model
model = UNet().to(config.DEVICE)
model.load_state_dict(torch.load(config.get_model_path(), map_location=config.DEVICE))
model.eval()

# Process all images in a folder
input_folder = "data/images"
output_folder = "output_results"
os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

print(f"Processing {len(image_files)} images...")

results = []

for img_file in image_files:
    # Load image
    img_path = os.path.join(input_folder, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))

    # Preprocess
    input_img = img / 255.0
    tensor = np.transpose(input_img, (2, 0, 1))
    tensor = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).to(config.DEVICE)

    # Predict
    with torch.no_grad():
        pred = model(tensor)[0, 0].cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8)
    severity = (np.sum(mask) / mask.size) * 100

    # Save result
    overlay = img.copy()
    overlay[mask == 1] = [255, 0, 0]

    output_path = os.path.join(output_folder, f"result_{img_file}")
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    results.append({
        'filename': img_file,
        'severity': severity
    })

    print(f"✓ {img_file}: {severity:.2f}%")

# Save summary
import json
with open(os.path.join(output_folder, "summary.json"), 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ All results saved to: {output_folder}/")
```

---

## 🎯 Quick Commands Summary

| Task                   | Command                           |
| ---------------------- | --------------------------------- |
| **Run Demo**           | `streamlit run app.py`            |
| **Train Model**        | `python train_with_validation.py` |
| **Test Dataset**       | `python dataset.py`               |
| **View Config**        | `python config.py`                |
| **Check Severity**     | `python severity_model.py`        |
| **Test Preprocessing** | `python preprocessing.py`         |

---

## 📝 Expected Output Files

After running different scripts, you'll find:

```
project/
├── logs/
│   ├── training_curves.png      # Training/validation loss plot
│   └── test_prediction.png      # Sample prediction
│
├── models/
│   ├── unet_model.pth           # Final trained model
│   └── best_unet_model.pth      # Best model (lowest val loss)
│
└── output_results/              # Batch processing results
    ├── result_image1.jpg
    ├── result_image2.jpg
    └── summary.json
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'albumentations'"

**Solution**: Use SimplePsoriasisDataset or install albumentations:

```bash
pip install albumentations
```

### Issue: "CUDA out of memory"

**Solution**: Edit `config.py`:

```python
BATCH_SIZE = 2  # Reduce from 4
```

### Issue: "Model file not found"

**Solution**: Train a model first:

```bash
python train_with_validation.py
```

---

## ✅ Verification

Test everything works:

```bash
# Test imports
python -c "import config; import dataset; print('✓ All imports OK')"

# Test dataset
python -c "from dataset import get_dataset_stats; print(get_dataset_stats())"

# Test model loading
python -c "from segmentation_model import UNet; import torch; m = UNet(); print('✓ Model OK')"
```

---

Need help? Check the [README.md](README.md) or [walkthrough.md](../brain/walkthrough.md) for more details!
