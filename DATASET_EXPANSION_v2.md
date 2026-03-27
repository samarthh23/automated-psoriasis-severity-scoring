# Dataset Expansion & Model Improvements — v2.0

This document summarizes all changes made to expand the dataset and improve segmentation metrics.

---

## 1. Dataset Expansion: 400 → 1200 Images

| Metric | Before (v1) | After (v2) |
|---|---|---|
| **Total images** | 400 | **1200** |
| **Training set** | ~280 | **~840** |
| **Validation set** | ~60 | **~180** |
| **Test set** | ~60 | **~180** |
| **Source** | ISIC2018 Task 1 (random subset) | ISIC2018 Task 1 (larger random subset) |
| **Split ratio** | 70/15/15 | 70/15/15 (unchanged) |

> **Why 1200?** The ISIC2018 dataset has 2594 training images with masks. 1200 is a balanced choice — 3x more data than before while keeping training time manageable. More data reduces overfitting and improves generalization.

---

## 2. Model Architecture Improvements

### Before: Shallow U-Net (2-level)
```
Encoder: 3→64 → 64→128 → Bottleneck: 128→256
No BatchNorm, No Dropout
```

### After: Deeper U-Net (3-level) with BatchNorm + Dropout
```
Encoder: 3→64 → 64→128 → 128→256 → Bottleneck: 256→512
BatchNorm after every conv layer
Dropout2d(0.3) at bottleneck
```

| Feature | Before | After |
|---|---|---|
| **Encoder levels** | 2 | **3** |
| **Bottleneck channels** | 256 | **512** |
| **BatchNorm** | ❌ None | ✅ After every conv |
| **Dropout** | ❌ None | ✅ Dropout2d(0.3) |
| **Conv bias** | Yes | **No** (BatchNorm handles it) |

> **Impact**: BatchNorm stabilizes gradient flow, enabling faster convergence. The deeper architecture captures multi-scale features (small and large lesions). Dropout prevents overfitting on the training set.

---

## 3. Loss Function: BCE → Dice+BCE Combined

### Before
- `BCEWithLogitsLoss` — optimizes pixel-wise classification accuracy

### After
- `DiceBCELoss` = BCE Loss + Dice Loss (equal weighting)

```python
# Dice Loss directly optimizes the Dice coefficient metric
dice_loss = 1 - (2 * intersection + smooth) / (union + smooth)
combined_loss = bce_loss + dice_loss
```

> **Impact**: Standard BCE treats each pixel independently and doesn't care about the overlap between prediction and ground truth. **Dice loss directly optimizes the Dice/F1 metric**, which is the primary evaluation metric for segmentation. This leads to significantly better Dice and IoU scores.

---

## 4. Training Strategy Improvements

| Setting | Before | After |
|---|---|---|
| **Epochs** | 30 | **50** |
| **LR Scheduler** | ReduceLROnPlateau | **Cosine Annealing** (η_min=1e-6) |
| **Early stopping criterion** | Val Loss | **Val Dice** (directly metric-aligned) |
| **Early stopping patience** | 7 | **10** |
| **Metric tracking** | Loss only | **Loss + Dice per epoch** |
| **Training curves** | Loss only | **Loss + Dice (side-by-side)** |

> **Cosine Annealing** provides smoother learning rate decay compared to step-based reduction, typically yielding better final performance. Early stopping on **Val Dice** ensures we save the model that has the best segmentation quality, not just the lowest loss.

---

## 5. Data Augmentation Improvements

| Augmentation | Before | After |
|---|---|---|
| Horizontal Flip | ✅ p=0.5 | ✅ p=0.5 |
| Vertical Flip | ✅ p=0.5 | ✅ p=0.5 |
| Rotation | ±15° | **±20°** |
| Brightness/Contrast | ±0.1 | **±0.15** |
| **Elastic Transform** | ❌ | ✅ **p=0.3** (NEW) |
| **Hue/Saturation Jitter** | ❌ | ✅ **p=0.3** (NEW) |

> **Elastic Transform** teaches the model to handle deformable lesion shapes. **Hue/Saturation Jitter** makes the model robust to different skin tones and lighting conditions — critical for real-world dermatology images.

---

## 6. Expected Effects on Metrics

| Metric | Expected Improvement | Reason |
|---|---|---|
| **Dice Coefficient** | ↑ Significant | Dice loss directly optimizes this metric + more data |
| **IoU (Jaccard)** | ↑ Significant | Correlated with Dice improvement |
| **Generalization** | ↑ Better | 3x more training data + stronger augmentation |
| **Robustness** | ↑ Better | Elastic transform + color jitter + BatchNorm |
| **Overfitting** | ↓ Reduced | Dropout + more data + augmentation |

### Training Progress (live)
Training is currently running on **NVIDIA RTX 3050 GPU (CUDA)**:
- Epoch 8/50 — Best Val Dice: **0.688** (and climbing)
- Training will complete automatically and save the best model

---

## 7. Files Changed

| File | Change Summary |
|---|---|
| `config.py` | SAMPLE_SIZE=1200, EPOCHS=50, stronger augmentation, dice_bce loss |
| `segmentation_model.py` | 3-level UNet + BatchNorm + Dropout |
| `dataset.py` | Added elastic transform + color jitter augmentation |
| `dataset_prepare.py` | Clears old data before copy, handles SAMPLE_SIZE=None |
| `train_with_validation.py` | DiceBCE loss, cosine annealing, Dice tracking, dual plot |
| `evaluate_for_paper.py` | Updated loss function reference |

---

## 8. How to Know When Training Completes

The training terminal will print a final summary block when it finishes:

```
======================================================================
TRAINING COMPLETE!
======================================================================
Best Validation Dice: X.XXXX
Best Validation Loss: X.XXXX
Final Train Loss: X.XXXX
Final Val Loss: X.XXXX
Final Train Dice: X.XXXX
Final Val Dice: X.XXXX
======================================================================
```

**Other signs it's done:**
- The terminal will return to the prompt (`PS C:\...>`)
- Training curves plot will be saved at: `logs/training_curves.png`
- Test prediction sample will be saved at: `logs/test_prediction.png`
- Model files will be at: `models/unet_model.pth` and `unet_model.pth`

**After training completes**, run:
```bash
python evaluate_for_paper.py
```
This will generate updated Dice/IoU metrics on the test set and save results to `logs/paper_metrics.txt`.
