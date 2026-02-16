# Psoriasis Lesion Segmentation System

Automated psoriasis lesion segmentation and severity scoring using deep learning with explainable AI.

## Features

- 🎯 **Automated Segmentation**: U-Net based lesion detection
- 📊 **Severity Scoring**: Quantitative assessment of lesion coverage
- 🔍 **Explainability**: Grad-CAM visualizations
- 🖥️ **Web Interface**: Easy-to-use Streamlit demo
- 📈 **Train/Val Split**: Proper evaluation with 70/15/15 split

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Run the Demo

```bash
streamlit run app.py
```

Upload a skin image and get:

- Segmentation mask overlay
- Severity score (% of affected area)
- Grad-CAM explainability heatmap

## Training

### Train with Validation Split

```bash
python train_with_validation.py
```

This will:

- Split dataset into train (70%), val (15%), test (15%)
- Train for 15 epochs
- Save best model based on validation loss
- Generate training curves

### Legacy Training (All Data)

```bash
python train_segmentation.py
```

## Project Structure

```
project/
├── config.py                    # Centralized configuration
├── dataset.py                   # Enhanced dataset with train/val/test split
├── segmentation_model.py        # U-Net architecture
├── train_with_validation.py     # Training with validation
├── app.py                       # Streamlit web interface
├── severity_model.py            # Severity scoring module
├── explainability.py            # Grad-CAM visualization
├── requirements.txt             # Dependencies
│
├── data/
│   ├── images/                  # Input images
│   └── masks/                   # Ground truth masks
│
├── models/                      # Saved models
│   ├── unet_model.pth          # Final model
│   └── best_unet_model.pth     # Best validation model
│
└── logs/                        # Training logs and visualizations
    ├── training_curves.png
    └── test_prediction.png
```

## Configuration

All settings are centralized in `config.py`:

```python
# Dataset configuration
IMG_SIZE = 256
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Training configuration
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
EPOCHS = 15

# Augmentation
USE_AUGMENTATION = True
```

## Dataset Module

### Simple Usage (No Augmentation)

```python
from dataset import SimplePsoriasisDataset

train_ds = SimplePsoriasisDataset(split="train")
val_ds = SimplePsoriasisDataset(split="val")
test_ds = SimplePsoriasisDataset(split="test")
```

### Advanced Usage (With Augmentation)

```python
from dataset import PsoriasisDataset

# Requires: pip install albumentations
train_ds = PsoriasisDataset(split="train", use_augmentation=True)
val_ds = PsoriasisDataset(split="val", use_augmentation=False)
```

### Helper Functions

```python
from dataset import get_dataloaders, get_dataset_stats

# Get statistics
stats = get_dataset_stats()
print(stats)  # {'total': 80, 'train': 56, 'val': 12, 'test': 12}

# Get dataloaders
train_loader, val_loader, test_loader = get_dataloaders(batch_size=4)
```

## Model Architecture

**U-Net** with:

- 2-level encoder-decoder
- Skip connections
- Binary segmentation output

## Dataset

Currently using 80 images from ISIC2018 dataset:

- Train: 56 images (70%)
- Validation: 12 images (15%)
- Test: 12 images (15%)

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- Streamlit
- grad-cam
- albumentations (optional, for augmentation)

See `requirements.txt` for complete list.

## License

Academic/Research Use

## Authors

Samarth Hegde
