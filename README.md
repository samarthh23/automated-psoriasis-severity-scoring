# Automated Psoriasis Lesion Segmentation and Severity Scoring

A comprehensive deep learning-based system for the automated segmentation of psoriasis lesions from skin images. This project provides not only precise lesion masks but also quantitative severity scoring and model explainability using Grad-CAM, all wrapped in a professional clinical-grade Streamlit web interface.

## 🎯 Key Features

- **Automated Segmentation**: Deep 3-level U-Net architecture for robust lesion detection across different skin tones and lighting conditions.
- **Quantitative Severity Scoring**: Calculates the exact percentage of the skin area affected by lesions.
- **Explainable AI (XAI)**: Integrated Grad-CAM visualization to highlight the regions the model focuses on, improving clinical trust.
- **Clinical Web Interface**: A beautifully designed, dark-themed Streamlit application for interactive analysis, threshold adjustment, and visual comparisons.
- **High Performance**: Optimized using a combined Dice + BCE loss function, achieving an Average Dice Coefficient of **~0.79** and an Average IoU of **~0.70**.

## 📊 Dataset Details

The model is trained on a carefully curated subset of the **ISIC 2018 Task 1 (Lesion Boundary Segmentation)** dataset. 

- **Total Images**: 1,200 images (expanded from an initial 400 for better generalization)
- **Data Split**:
  - **Training Set**: ~840 images (70%)
  - **Validation Set**: ~180 images (15%)
  - **Test Set**: ~180 images (15%)
- **Preprocessing**: Images are resized to 256×256 pixels and normalized to a [0, 1] range.
- **Data Augmentation**: To prevent overfitting and simulate real-world dermatology scenarios, the following augmentations are applied during training (via `albumentations`):
  - Horizontal & Vertical Flips (p=0.5)
  - Random Rotation (±20°)
  - Brightness/Contrast Adjustment (±0.15)
  - Elastic Transform (p=0.3) - *Crucial for learning deformable lesion boundaries.*
  - Hue/Saturation Jitter (p=0.3) - *Ensures robustness across diverse skin tones.*

## 🧠 Model Architecture & Training Strategy

### Architecture: Enhanced U-Net
The project utilizes an upgraded U-Net architecture designed specifically for medical image segmentation:
- **Encoder**: 3-level depth (64 → 128 → 256 channels) to capture both granular and diffuse lesion features.
- **Bottleneck**: 512 channels with `Dropout2d(0.3)` to prevent overfitting.
- **Normalization**: `BatchNorm2d` after every convolutional layer for stable gradient flow.
- **Bias**: Convolutional biases are removed in favor of BatchNorm.

### Training Pipeline
- **Loss Function**: `DiceBCELoss` (Combination of Dice Loss and Binary Cross-Entropy). *Dice loss directly optimizes the segmentation overlap metric, effectively handling the class imbalance between background skin and lesion pixels.*
- **Optimizer**: Adam (`lr=1e-3`, `weight_decay=1e-5`).
- **Learning Rate Scheduler**: Cosine Annealing (`T_max=50`, `eta_min=1e-6`) for smooth convergence.
- **Epochs**: 50
- **Early Stopping**: Based on **Validation Dice** (patience = 10 epochs).

## 📈 Performance Metrics

Evaluated on the 180-image Test Set:

| Metric | Score | Note |
|--------|-------|------|
| **Average Dice Coefficient** | **0.7917** | Primary measure of overlap between prediction and ground truth. |
| **Average IoU (Jaccard)** | **0.6980** | Stringent metric penalizing both false positives and false negatives. |
| **Inference Time (CPU)** | ~0.33 sec | Fast enough for real-time clinical usage. |

## 📁 Project Structure

```text
project/
├── config.py                    # Centralized hyperparameter & path configurations
├── dataset.py                   # Data loaders, augmentations, and train/val/test splitting
├── segmentation_model.py        # 3-level U-Net architecture definition
├── severity_model.py            # Logic for calculating the % severity score
├── explainability.py            # Grad-CAM heatmap generation module
├── app.py                       # The Streamlit web interface
├── train_with_validation.py     # Main training script (with Dice/Loss tracking)
├── evaluate_for_paper.py        # Script to generate final metrics and sample output images
├── preprocessing.py             # Utility for basic image/mask visual inspection
├── requirements.txt             # Python dependencies
│
├── data/                        # Local dataset storage (not tracked in Git)
│   ├── images/                  
│   └── masks/                   
│
├── models/                      # Saved PyTorch model checkpoints
│   ├── best_unet_model.pth      # Model with the highest Validation Dice
│   └── unet_model.pth           # Final epoch model
│
└── logs/                        # Generated charts, metrics, and visual samples
    ├── paper_metrics.txt
    ├── training_curves.png
    └── paper_samples/
```

## 🚀 Quick Start Guide

### 1. Installation

Ensure you have Python 3.8+ installed.

```bash
git clone https://github.com/samarthh23/automated-psoriasis-severity-scoring.git
cd automated-psoriasis-severity-scoring
pip install -r requirements.txt
```

### 2. Running the Clinical Interface

Start the Streamlit application:

```bash
streamlit run app.py
```

Upload a skin image in the browser to view:
- The predicted lesion mask
- Overlaid and blended visual comparisons
- The computed Severity Score (%)
- Grad-CAM attention heatmaps

### 3. Model Training

To train the model from scratch on your local dataset:

1. Ensure your data is placed in `data/images/` and `data/masks/`.
2. Run the training script:

```bash
python train_with_validation.py
```
This script handles the automatic dataset splitting, applies the augmentations, trains the model, saves the best checkpoints to the `models/` directory, and plots the training curves.

### 4. Evaluation & Metrics Generation

To generate quantitative reports and visual samples for documentation:

```bash
python evaluate_for_paper.py
```
Outputs will be saved in the `logs/` directory.

## 📦 Deployment (Streamlit Cloud)

The application is optimized for deployment on Streamlit Community Cloud. See `DEPLOYMENT.md` for specific instructions regarding Git LFS and model weight handling.

## 📝 License

For Academic and Research Use.

## 👨‍💻 Author

**Samarth Hegde**
