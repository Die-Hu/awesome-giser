# Machine Learning for GIS

> Definitive reference for applying machine learning to geospatial problems -- from traditional supervised classification through deep learning with transformers to foundation models and GeoAI with LLMs. Covers spatial cross-validation, feature engineering, object detection, change detection, AutoML, MLOps, benchmark datasets, and framework selection.

> **Quick Picks**
> - **Getting started with RS deep learning:** TorchGeo (PyTorch datasets + models)
> - **Zero-shot segmentation:** samgeo (SAM / SAM 2 for geospatial)
> - **Foundation model for EO:** Prithvi-EO 2.0 (IBM/NASA) or Clay Foundation Model
> - **Traditional ML baseline:** scikit-learn + XGBoost with spatial CV
> - **Object detection in RS:** YOLOv8+ with DOTA / xView datasets
> - **Change detection:** BIT Transformer or siamese networks
> - **LLM + spatial reasoning:** Claude / GPT for spatial analysis code generation
> - **MLOps for geospatial:** MLflow + ONNX + cloud inference

---

## Table of Contents

- [Supervised Classification](#supervised-classification)
- [Deep Learning for Remote Sensing](#deep-learning-for-remote-sensing)
- [Foundation Models 2024-2025](#foundation-models-2024-2025)
- [SAM for Geospatial](#sam-for-geospatial)
- [Spatial Feature Engineering](#spatial-feature-engineering)
- [Object Detection in Remote Sensing](#object-detection-in-remote-sensing)
- [Change Detection with Deep Learning](#change-detection-with-deep-learning)
- [AutoML for GIS](#automl-for-gis)
- [GeoAI with LLMs](#geoai-with-llms)
- [MLOps for Geospatial](#mlops-for-geospatial)
- [Benchmark Datasets](#benchmark-datasets)
- [Framework Comparison](#framework-comparison)

> **Cross-references:** [AI/ML Geospatial Tools](../tools/ai-ml-geospatial.md) | [AI/ML Visualization](../visualization/ai-ml-visualization.md) | [ML Training Data Sources](../data-sources/ml-training-data.md)

---

## Supervised Classification

Traditional machine learning methods remain the workhorse for land cover classification, spatial prediction, and tabular geospatial modeling. They are fast to train, interpretable, and often competitive with deep learning when the feature space is well-designed.

### Common Algorithms

| Algorithm | Strengths | Weaknesses | Best For | Key Hyperparameters |
|---|---|---|---|---|
| **Random Forest (RF)** | Robust, handles noise, built-in feature importance, parallelizable | Can overfit with many correlated features, large memory for big forests | Land cover classification, general purpose | `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf` |
| **XGBoost** | High accuracy, regularization built-in, handles missing values natively | More hyperparameters, sensitive to tuning | Tabular spatial prediction, competition-winning | `n_estimators`, `max_depth`, `learning_rate`, `reg_alpha`, `reg_lambda` |
| **LightGBM** | Faster than XGBoost on large data, lower memory, categorical feature support | Less stable on very small datasets | Large-scale spatial prediction, real-time inference | `num_leaves`, `learning_rate`, `feature_fraction`, `bagging_fraction` |
| **CatBoost** | Native categorical handling, ordered boosting reduces overfitting | Slower initial training | Datasets with mixed feature types | `iterations`, `depth`, `learning_rate`, `l2_leaf_reg` |
| **Support Vector Machine (SVM)** | Effective in high dimensions, works with small samples | Slow on large datasets (O(n^2-n^3)), sensitive to kernel/C/gamma | Small training sets, hyperspectral data | `C`, `gamma`, `kernel` |
| **k-Nearest Neighbors (KNN)** | Simple, no training phase, non-parametric | Slow at prediction (O(n*d)), sensitive to irrelevant features | Quick baseline, small datasets | `n_neighbors`, `weights`, `metric` |
| **Logistic Regression** | Interpretable, fast, probabilistic output, well-calibrated | Assumes linear decision boundary (without kernel trick) | Binary classification, interpretable models | `C`, `penalty`, `solver` |

### Ensemble Methods for GIS

Ensembles combine multiple models to improve prediction accuracy and robustness. For geospatial applications, ensembles are particularly valuable because different algorithms may capture different aspects of spatial patterns.

| Ensemble Strategy | How It Works | When to Use |
|---|---|---|
| **Bagging (Random Forest)** | Train many trees on bootstrap samples, aggregate by voting | Default choice; reduces variance |
| **Boosting (XGBoost, LightGBM, CatBoost)** | Sequentially correct errors from previous models | When you need maximum accuracy on tabular features |
| **Stacking** | Train a meta-learner on predictions from multiple base models | Competition-level accuracy; combine diverse model types |
| **Voting (hard/soft)** | Majority vote or average probabilities from multiple models | Simple fusion of different algorithms |
| **Blending** | Like stacking but uses a held-out validation set instead of CV | Faster than stacking, slightly less data-efficient |

```python
from sklearn.ensemble import (
    StackingClassifier, VotingClassifier, RandomForestClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

# Stacking ensemble with spatial-aware base learners
base_learners = [
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=15, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           use_label_encoder=False, eval_metric='mlogloss')),
    ('lgbm', LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1,
                             verbose=-1)),
]

# Meta-learner aggregates base model predictions
stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,  # NOTE: Replace with spatial CV in production (see below)
    stack_method='predict_proba'
)

stacking_clf.fit(X_train, y_train)
print(f"Stacking accuracy: {stacking_clf.score(X_test, y_test):.4f}")
```

### Workflow for Spatial Classification

1. **Define classes** -- create a clear, exhaustive, mutually exclusive legend. Use standardized schemes (CORINE, NLCD, LCCS) where possible.
2. **Collect training data** -- field survey, image interpretation, existing maps, crowdsourced labels. Aim for at least 50-100 samples per class, more for heterogeneous classes.
3. **Extract features** -- spectral bands, indices, texture, DEM derivatives, distance features, temporal composites. See [Spatial Feature Engineering](#spatial-feature-engineering).
4. **Exploratory analysis** -- check class distributions, feature correlations (VIF), and separability (Jeffries-Matusita distance, scatter plots).
5. **Split data** -- spatially stratified split (not random!) to avoid spatial autocorrelation leakage. See [Spatial Cross-Validation](#spatial-cross-validation) below.
6. **Train model** -- with cross-validation (spatial CV recommended).
7. **Tune hyperparameters** -- grid search, random search, or Bayesian optimization (Optuna). Always use spatial CV for the inner loop.
8. **Predict** -- apply model to full extent. Chunk large rasters to manage memory.
9. **Validate** -- confusion matrix, overall accuracy (OA), kappa, per-class F1, producer's/user's accuracy. Report confidence intervals.
10. **Post-process** -- majority filter, minimum mapping unit enforcement, sieve, manual corrections.

### Spatial Cross-Validation

Standard k-fold cross-validation gives overly optimistic accuracy estimates for spatial data because nearby samples are correlated (spatial autocorrelation). Use spatial CV to get honest performance estimates.

#### Methods

| Method | How It Works | Strengths | Weaknesses | Library |
|---|---|---|---|---|
| **Spatial block CV** | Divide study area into spatial blocks (grid cells), hold out entire blocks | Simple, handles spatial autocorrelation | Block size selection is subjective | `scikit-learn` + custom, `mlr3` (R), `spacv` |
| **Buffered leave-one-out (BLOO)** | Exclude points within a buffer distance of the test point | Principled handling of autocorrelation range | Computationally expensive, many test folds | Custom implementation |
| **Leave-location-out (LLO)** | Hold out entire spatial clusters | Works with irregular point distributions | Need to define meaningful clusters | `GroupKFold` in scikit-learn |
| **Spatial K-Fold (SKCV)** | Spatially balanced k-fold using k-means on coordinates | Balances spatial coverage | May not respect autocorrelation range | `sklearn_extra`, custom |
| **Environmental blocking** | Block by environmental similarity (not just space) | Handles extrapolation to new environments | Requires environmental covariates | Custom, `blockCV` (R) |

#### Choosing Block Size

The block size should be at least as large as the spatial autocorrelation range of the target variable:
- Compute variograms (`scikit-gstat`, `pykrige`) to estimate the autocorrelation range.
- Use the range as minimum block size.
- Rule of thumb: try block sizes of 1x, 2x, and 5x the autocorrelation range and compare.

#### Spatial Block Cross-Validation Example

```python
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# --- Method 1: Grid-based spatial blocking ---
# Assign each sample to a spatial block based on coordinate grid
def assign_spatial_blocks(coords, block_size):
    """Assign spatial block IDs based on a regular grid."""
    col = (coords[:, 0] // block_size).astype(int)
    row = (coords[:, 1] // block_size).astype(int)
    return row * 10000 + col  # Unique block ID

# coords: array of shape (n_samples, 2) with x, y coordinates
block_ids = assign_spatial_blocks(coords, block_size=5000)  # 5 km blocks

gkf = GroupKFold(n_splits=5)
scores = []
all_y_true, all_y_pred = [], []

for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=block_ids)):
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42,
                                  n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])
    y_pred = clf.predict(X[test_idx])

    fold_score = accuracy_score(y[test_idx], y_pred)
    scores.append(fold_score)
    all_y_true.extend(y[test_idx])
    all_y_pred.extend(y_pred)
    print(f"Fold {fold + 1}: OA = {fold_score:.4f}")

print(f"\nSpatial CV accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
print("\nPer-class report:")
print(classification_report(all_y_true, all_y_pred, target_names=class_names))
```

```python
# --- Method 2: K-Means spatial clustering for blocking ---
from sklearn.cluster import KMeans

# Create spatial blocks using K-Means on coordinates
n_blocks = 10
kmeans = KMeans(n_clusters=n_blocks, random_state=42, n_init=10)
spatial_clusters = kmeans.fit_predict(coords)

gkf = GroupKFold(n_splits=5)
scores = []
for train_idx, test_idx in gkf.split(X, y, groups=spatial_clusters):
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])
    scores.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))

print(f"Spatial CV (K-Means blocks): {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
```

```python
# --- Method 3: Buffered leave-one-out (BLOO) ---
from scipy.spatial.distance import cdist

def buffered_loo_cv(X, y, coords, buffer_distance, model_class, **model_kwargs):
    """Leave-one-out CV with spatial buffer to exclude nearby training points."""
    n = len(y)
    dist_matrix = cdist(coords, coords)
    predictions = np.full(n, np.nan)

    for i in range(n):
        # Exclude points within buffer distance of test point
        mask = dist_matrix[i] > buffer_distance
        mask[i] = False  # Also exclude the test point itself

        if mask.sum() < 10:  # Skip if too few training points
            continue

        model = model_class(**model_kwargs)
        model.fit(X[mask], y[mask])
        predictions[i] = model.predict(X[i:i+1])[0]

    valid = ~np.isnan(predictions)
    return accuracy_score(y[valid], predictions[valid])

bloo_acc = buffered_loo_cv(
    X, y, coords, buffer_distance=2000,  # 2 km buffer
    model_class=RandomForestClassifier, n_estimators=100, random_state=42
)
print(f"BLOO accuracy (2 km buffer): {bloo_acc:.4f}")
```

### Complete Classification Pipeline

```python
import rasterio
import numpy as np
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load training data
training_points = gpd.read_file("training_samples.gpkg")

# 2. Extract features from raster stack
with rasterio.open("feature_stack.tif") as src:
    features = np.array([
        src.read()[:, int(row), int(col)]
        for row, col in zip(
            *rasterio.transform.rowcol(src.transform,
                                        training_points.geometry.x,
                                        training_points.geometry.y)
        )
    ])
    profile = src.profile

X = features
y = training_points['class_id'].values
coords = np.column_stack([training_points.geometry.x,
                           training_points.geometry.y])

# 3. Spatial blocking
block_ids = assign_spatial_blocks(coords, block_size=5000)

# 4. Train with spatial CV
gkf = GroupKFold(n_splits=5)
best_score = 0
best_model = None

for train_idx, test_idx in gkf.split(X, y, groups=block_ids):
    clf = RandomForestClassifier(n_estimators=300, max_depth=20,
                                  min_samples_leaf=5, random_state=42, n_jobs=-1)
    clf.fit(X[train_idx], y[train_idx])
    score = clf.score(X[test_idx], y[test_idx])
    if score > best_score:
        best_score = score
        best_model = clf

# 5. Predict full raster (chunked for memory management)
with rasterio.open("feature_stack.tif") as src:
    out_profile = src.profile.copy()
    out_profile.update(count=1, dtype='uint8')

    with rasterio.open("classification.tif", 'w', **out_profile) as dst:
        # Process in windows/chunks
        for ji, window in src.block_windows(1):
            data = src.read(window=window)
            h, w = data.shape[1], data.shape[2]
            pixels = data.reshape(data.shape[0], -1).T

            # Mask nodata
            valid = ~np.any(np.isnan(pixels), axis=1)
            result = np.zeros(h * w, dtype=np.uint8)
            if valid.sum() > 0:
                result[valid] = best_model.predict(pixels[valid])

            dst.write(result.reshape(1, h, w), window=window)

# 6. Save model for later use
joblib.dump(best_model, "rf_classifier.joblib")
print(f"Best spatial CV accuracy: {best_score:.4f}")
```

---

## Deep Learning for Remote Sensing

Deep learning has transformed remote sensing analysis, enabling pixel-level segmentation, object detection, super-resolution, and multi-temporal analysis at scale.

### Architecture Overview

| Architecture | Task | Input | Key Features | Parameters | Typical mIoU/F1 on RS |
|---|---|---|---|---|---|
| **U-Net** | Semantic segmentation | Image patches | Encoder-decoder with skip connections; gold standard for RS | 31M (ResNet-34 backbone) | 0.75-0.85 |
| **U-Net++** | Semantic segmentation | Image patches | Nested skip connections, dense connectivity | 36M | 0.78-0.88 |
| **DeepLab v3+** | Semantic segmentation | Image patches | Atrous (dilated) convolutions, ASPP module, multi-scale context | 60M (ResNet-101) | 0.78-0.87 |
| **HRNet** | Semantic segmentation | Image patches | Maintains high-resolution representations throughout | 65M | 0.80-0.88 |
| **Mask R-CNN** | Instance segmentation | Full images | Detects and segments individual objects with masks | 44M (ResNet-50-FPN) | F1: 0.75-0.85 |
| **YOLO v8/v9/v11** | Object detection | Full images | Real-time detection, bounding boxes, optimized for speed | 3-68M (nano to xlarge) | mAP: 0.50-0.75 |
| **Vision Transformer (ViT)** | Classification / segmentation | Image patches | Self-attention, captures global context, scales well | 86M (ViT-B/16) | 0.82-0.90 |
| **Swin Transformer** | Segmentation / detection | Image patches | Shifted windows, hierarchical features, linear complexity | 88M (Swin-B) | 0.83-0.90 |
| **SegFormer** | Semantic segmentation | Image patches | Lightweight transformer decoder, multi-scale features | 84M (MiT-B5) | 0.80-0.88 |
| **SatMAE / Prithvi** | Foundation model | Multi-spectral patches | Self-supervised MAE pre-training on satellite data | 100M-600M | Fine-tune dependent |
| **SAM / SAM 2** | Zero-shot segmentation | Any image | Prompt-based segmentation, fine-tunable for RS | 636M (ViT-H) | 0.55-0.85 (task dependent) |

### Training Considerations for Remote Sensing

#### Patch Size and Overlap

- **Typical sizes:** 256x256 or 512x512 pixels. Larger patches (512-1024) capture more spatial context but require more GPU memory.
- **Overlap at inference:** Use 50% overlap with blending (Gaussian weights) to avoid edge artifacts in seamless predictions.
- **Stride vs. patch size trade-off:** smaller stride = better quality, but slower. A stride of 0.5x patch size is a common default.

```python
# Inference with overlapping patches and Gaussian blending
import torch
import numpy as np
from scipy.ndimage import gaussian_filter

def predict_with_overlap(model, image, patch_size=512, overlap=256, device='cuda'):
    """Predict on large image using overlapping patches with Gaussian blending."""
    c, h, w = image.shape
    stride = patch_size - overlap

    # Create Gaussian weight mask for blending
    weight = np.ones((patch_size, patch_size), dtype=np.float32)
    weight = gaussian_filter(weight, sigma=patch_size // 6)
    weight_tensor = torch.from_numpy(weight).to(device)

    prediction = torch.zeros((1, h, w), device=device)
    count = torch.zeros((1, h, w), device=device)

    model.eval()
    with torch.no_grad():
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[:, y:y+patch_size, x:x+patch_size]
                patch = patch.unsqueeze(0).to(device)

                output = model(patch).squeeze(0)  # (num_classes, H, W)
                pred = output.argmax(dim=0, keepdim=True).float()

                prediction[:, y:y+patch_size, x:x+patch_size] += pred * weight_tensor
                count[:, y:y+patch_size, x:x+patch_size] += weight_tensor

    return (prediction / count.clamp(min=1e-8)).cpu().numpy()
```

#### Data Augmentation for RS

```python
import albumentations as A

# RS-appropriate augmentations
rs_augmentations = A.Compose([
    # Geometric (safe for RS)
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.5),

    # Radiometric (simulate atmospheric/sensor variation)
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.GaussianBlur(blur_limit=(3, 5), p=0.2),

    # RS-specific
    A.RandomSizedCrop(min_max_height=(200, 512), height=512, width=512, p=0.3),
    A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),  # Simulate clouds
])

# CAUTION: Avoid augmentations that break geographic meaning
# - Color jitter that changes spectral relationships
# - Elastic transforms that distort spatial relationships
# - Normalization that removes physical units (use per-band standardization instead)
```

#### Handling Class Imbalance

```python
import torch
import torch.nn as nn

# Method 1: Weighted cross-entropy loss
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
weights = torch.FloatTensor(class_weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Method 2: Focal Loss (focuses on hard examples)
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

# Method 3: Dice Loss (directly optimizes overlap metric)
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.softmax(inputs, dim=1)
        targets_onehot = nn.functional.one_hot(targets, inputs.shape[1])
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()

        intersection = (inputs * targets_onehot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_onehot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()

# Method 4: Combined loss (common in RS competitions)
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, inputs, targets):
        return self.ce_weight * self.ce(inputs, targets) + \
               self.dice_weight * self.dice(inputs, targets)
```

#### Multi-Spectral Input (Beyond RGB)

Standard pretrained backbones (ImageNet) expect 3-channel RGB input. For multi-spectral satellite data (e.g., 13 bands from Sentinel-2), you need to adapt the input layer:

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Option 1: Use segmentation_models_pytorch (supports arbitrary input channels)
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',  # Pretrained on RGB, first conv adapted
    in_channels=13,              # 13 Sentinel-2 bands
    classes=10,                  # Number of land cover classes
)

# Option 2: Manual first-conv adaptation with weight initialization
def adapt_first_conv(model, in_channels=13):
    """Replace first conv layer, initializing new channels from pretrained RGB."""
    old_conv = model.encoder.conv1
    new_conv = nn.Conv2d(
        in_channels, old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )

    with torch.no_grad():
        # Copy RGB weights to first 3 channels
        new_conv.weight[:, :3] = old_conv.weight
        # Initialize remaining channels as mean of RGB weights
        for i in range(3, in_channels):
            new_conv.weight[:, i] = old_conv.weight.mean(dim=1)

    model.encoder.conv1 = new_conv
    return model

# Option 3: Use RS-specific pretrained weights (recommended)
# TorchGeo provides weights pretrained on Sentinel-2 (all bands)
from torchgeo.models import ResNet50_Weights
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights=None,
    in_channels=13,
    classes=10,
)
# Load TorchGeo pretrained weights manually
```

### U-Net Training Example (Complete)

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import segmentation_models_pytorch as smp
import rasterio
import numpy as np
from pathlib import Path

class RSPatchDataset(Dataset):
    """Dataset for remote sensing image patches."""

    def __init__(self, image_dir, mask_dir, transform=None, bands=None):
        self.image_paths = sorted(Path(image_dir).glob("*.tif"))
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.bands = bands  # e.g., [0, 1, 2, 3, 7] for RGB + NIR

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with rasterio.open(self.image_paths[idx]) as src:
            image = src.read().astype(np.float32)
            if self.bands:
                image = image[self.bands]

        mask_path = self.mask_dir / self.image_paths[idx].name
        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.int64)

        if self.transform:
            transformed = self.transform(image=image.transpose(1, 2, 0),
                                          mask=mask)
            image = transformed['image'].transpose(2, 0, 1)
            mask = transformed['mask']

        # Per-band standardization
        for b in range(image.shape[0]):
            band = image[b]
            image[b] = (band - band.mean()) / (band.std() + 1e-8)

        return torch.from_numpy(image), torch.from_numpy(mask)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = smp.Unet(
    encoder_name='resnet50',
    encoder_weights='imagenet',
    in_channels=4,    # RGB + NIR
    classes=6,         # 6 land cover classes
).to(device)

# Loss and optimizer
criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# Data loaders
train_ds = RSPatchDataset("data/train/images", "data/train/masks",
                           transform=rs_augmentations, bands=[0, 1, 2, 3])
val_ds = RSPatchDataset("data/val/images", "data/val/masks", bands=[0, 1, 2, 3])
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=8, num_workers=4)

# Training loop
best_val_iou = 0.0
for epoch in range(50):
    model.train()
    train_loss = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_iou = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            # Compute mean IoU
            for cls in range(6):
                intersection = ((preds == cls) & (masks == cls)).sum().float()
                union = ((preds == cls) | (masks == cls)).sum().float()
                if union > 0:
                    val_iou += (intersection / union).item()

    val_iou /= (6 * len(val_loader))
    scheduler.step()

    print(f"Epoch {epoch+1}/50 | Loss: {train_loss/len(train_loader):.4f} "
          f"| Val mIoU: {val_iou:.4f}")

    if val_iou > best_val_iou:
        best_val_iou = val_iou
        torch.save(model.state_dict(), "best_unet.pth")
```

### Choosing an Architecture

| Scenario | Recommended Architecture | Why |
|---|---|---|
| Pixel-level classification (land cover) | U-Net or DeepLab v3+ | Strong encoder-decoder with multi-scale context |
| High-resolution building extraction | U-Net++ or HRNet | Better boundary delineation |
| Object detection (buildings, vehicles) | YOLOv8+ or Mask R-CNN | Fast and accurate bounding boxes / masks |
| Interactive annotation / zero-shot | SAM via samgeo | No training required, prompt-based |
| Large-scale pre-trained backbone | Prithvi, SatlasPretrain, or Clay | Multi-spectral aware, minimal fine-tuning |
| Global context / large patches | Swin Transformer or SegFormer | Hierarchical attention with linear complexity |
| Limited labeled data (<100 samples) | Foundation model fine-tuning | Leverage pre-trained representations |
| Real-time inference needed | YOLOv8 nano/small or MobileNet-based | Optimized for speed |

---

## Foundation Models 2024-2025

Foundation models are large-scale self-supervised models pre-trained on massive earth observation datasets. They can be fine-tuned for multiple downstream tasks with minimal labeled data, dramatically reducing the annotation burden.

### Model Comparison

| Model | Organization | Version | Pre-training Data | Architecture | Parameters | Access |
|---|---|---|---|---|---|---|
| **[Prithvi-EO 2.0](https://huggingface.co/ibm-nasa-geospatial)** | IBM + NASA | 2.0 (2024) | HLS (Harmonized Landsat-Sentinel), global | ViT-based MAE, temporal encoding | 100M / 300M / 600M | Hugging Face, open-weight |
| **[Clay Foundation](https://clay.earth/)** | Clay Foundation | v1.5 (2025) | Multi-sensor (S1, S2, Landsat, NAIP, DEM) | ViT encoder with location/time encoding | 300M | Open source (GitHub) |
| **[SatlasPretrain](https://github.com/allenai/satlas)** | Allen AI | 2024 | Sentinel-2, NAIP, multi-task labels | Swin Transformer | 200M | GitHub, open |
| **[SSL4EO](https://github.com/zhu-xlab/SSL4EO-S12)** | TU Munich | v2 | Sentinel-1/2, global coverage | ResNet / ViT (MoCo v3, DINO) | 25M-300M | GitHub, open |
| **[GFM](https://github.com/mmendiet/GFM)** | Various | 2024 | Multiple RS sources | ViT | 100M-300M | GitHub |
| **[SpectralGPT](https://github.com/danfenghong/SpectralGPT)** | Wuhan University | 2024 | Hyperspectral + multi-spectral | Spectral-spatial GPT | 600M | GitHub |
| **[Scale-MAE](https://github.com/bair-climate-initiative/scale-mae)** | UC Berkeley | 2024 | Multi-scale satellite | MAE with scale encoding | 300M | GitHub |
| **[SatCLIP](https://github.com/microsoft/satclip)** | Microsoft | 2024 | Sentinel-2 + location | CLIP-style contrastive | 100M | GitHub |

### Benchmark Comparison (Fine-tuned on Downstream Tasks)

| Model | EuroSAT (Acc) | BigEarthNet (mAP) | SpaceNet Building (F1) | Burn Scars (mIoU) | Flood Detection (mIoU) |
|---|---|---|---|---|---|
| Random init (U-Net) | 0.92 | 0.72 | 0.75 | 0.52 | 0.58 |
| ImageNet pre-train | 0.96 | 0.82 | 0.80 | 0.58 | 0.65 |
| SSL4EO (MoCo v3) | 0.97 | 0.85 | 0.82 | 0.62 | 0.70 |
| SatlasPretrain | 0.98 | 0.87 | 0.84 | 0.65 | 0.73 |
| Prithvi-EO 2.0 (100M) | 0.97 | 0.86 | 0.85 | 0.68 | 0.76 |
| Prithvi-EO 2.0 (600M) | 0.98 | 0.89 | 0.87 | 0.72 | 0.80 |
| Clay v1.5 | 0.97 | 0.88 | 0.86 | 0.70 | 0.78 |

Scores are approximate and depend on fine-tuning setup. Foundation models show largest gains on limited labeled data scenarios (10-100 samples).

### Fine-tuning Prithvi-EO 2.0

```python
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn

# Load pre-trained Prithvi-EO 2.0
config = AutoConfig.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-2.0-300M")
encoder = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-2.0-300M")

# Add a segmentation head
class PrithviSegmentation(nn.Module):
    def __init__(self, encoder, num_classes, embed_dim=1024):
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=4),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=4),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Prithvi encoder outputs patch embeddings
        features = self.encoder(x).last_hidden_state
        B, N, D = features.shape
        h = w = int(N ** 0.5)
        features = features.permute(0, 2, 1).reshape(B, D, h, w)
        return self.decoder(features)

model = PrithviSegmentation(encoder, num_classes=6)

# Fine-tuning strategy: freeze encoder initially, then unfreeze
for param in model.encoder.parameters():
    param.requires_grad = False

# Train decoder head for 10 epochs
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=1e-3)
# ... train decoder ...

# Then unfreeze encoder with lower learning rate
for param in model.encoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW([
    {'params': model.encoder.parameters(), 'lr': 1e-5},
    {'params': model.decoder.parameters(), 'lr': 1e-4},
], weight_decay=1e-4)
# ... fine-tune end-to-end for 40 more epochs ...
```

### Fine-tuning Clay Foundation Model

```python
# Clay provides embeddings from multi-sensor inputs
# Fine-tune for: classification, segmentation, change detection, similarity search

import torch
from clay.model import ClayEncoder

# Load Clay encoder
encoder = ClayEncoder.from_pretrained("clay-foundation/clay-v1.5")

# Clay accepts multi-sensor inputs with metadata
# Input format: (image_tensor, metadata_dict)
metadata = {
    'platform': 'sentinel-2',
    'date': '2024-06-15',
    'lat': 48.8566,
    'lon': 2.3522,
    'gsd': 10.0,  # Ground sample distance in meters
}

# Get embeddings
with torch.no_grad():
    embeddings = encoder(image_tensor, metadata)

# Use embeddings for downstream tasks:
# 1. Linear probing (freeze encoder, train classifier head)
# 2. Full fine-tuning (unfreeze all layers)
# 3. Similarity search (nearest neighbor in embedding space)
# See: https://clay.earth/docs for full fine-tuning guides
```

### When to Use Foundation Models

| Scenario | Recommendation |
|---|---|
| Abundant labeled data (>10,000 samples) | Foundation model advantage is modest; standard pre-trained backbones may suffice |
| Limited labels (10-1,000 samples) | Foundation models shine -- 10-30% improvement over ImageNet pre-training |
| Multi-temporal analysis | Prithvi-EO 2.0 (has temporal encoding) or Clay |
| Multi-sensor fusion (SAR + optical) | Clay (designed for multi-sensor) or SSL4EO |
| Scene classification | SatlasPretrain or SSL4EO |
| Rapid prototyping / embedding search | Clay (good embedding quality for zero/few-shot) |
| Edge deployment | SSL4EO with ResNet backbone (smaller models) |

---

## SAM for Geospatial

The Segment Anything Model (SAM) and its successor SAM 2 have been adapted for geospatial applications through the [samgeo](https://samgeo.gishub.org/) package, enabling zero-shot and interactive segmentation of satellite and aerial imagery.

### samgeo Architecture

samgeo wraps Meta's SAM/SAM 2 for geospatial data by:
- Handling GeoTIFF I/O (preserving CRS, transform, and metadata)
- Converting geographic coordinates (lon/lat) to pixel coordinates for prompts
- Supporting large rasters through tiled processing
- Outputting georeferenced masks as GeoTIFF or vector (GeoJSON/GPKG)

### Automatic Mask Generation

```python
from samgeo import SamGeo

# Initialize with the largest model for best quality
sam = SamGeo(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth",
    device="cuda",
    sam_kwargs={
        'points_per_side': 32,          # Grid density for automatic prompts
        'pred_iou_thresh': 0.88,        # Confidence threshold
        'stability_score_thresh': 0.95, # Mask stability threshold
        'crop_n_layers': 1,             # Multi-crop for better small objects
        'min_mask_region_area': 100,    # Remove tiny masks (pixels)
    }
)

# Generate all masks for a satellite image
sam.generate(
    source="satellite.tif",
    output="masks.tif",
    foreground=True,      # Only foreground masks
    unique=True,          # Assign unique ID to each object
)

# Convert masks to vector (GeoPackage)
sam.raster_to_vector("masks.tif", "masks.gpkg")

# Visualize
sam.show_masks(figsize=(12, 8))
```

### Interactive Segmentation (Point and Box Prompts)

```python
from samgeo import SamGeo

sam = SamGeo(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth")
sam.set_image("satellite.tif")

# Point prompts (geographic coordinates)
# point_labels: 1 = foreground, 0 = background
sam.predict(
    point_coords=[[-122.419, 37.775], [-122.420, 37.776]],  # lon, lat
    point_labels=[1, 1],
    output="buildings.tif",
)

# Bounding box prompt (geographic coordinates)
# Format: [min_lon, min_lat, max_lon, max_lat]
sam.predict(
    boxes=[[-122.42, 37.77, -122.41, 37.78]],
    output="area_of_interest.tif",
)

# Text prompt (using GroundingDINO + SAM)
from samgeo import SamGeo
sam = SamGeo(
    model_type="vit_h",
    checkpoint="sam_vit_h_4b8939.pth",
    automatic=False,
)
sam.set_image("satellite.tif")
sam.predict_by_text(
    text="swimming pool",
    box_threshold=0.24,
    text_threshold=0.24,
    output="pools.tif",
)
```

### SAM 2 for Video / Multi-Temporal

SAM 2 extends SAM to video sequences, which is useful for multi-temporal satellite imagery:

```python
from samgeo import SamGeo2

# SAM 2 for temporal sequences
sam2 = SamGeo2(
    model_id="facebook/sam2-hiera-large",
    device="cuda",
)

# Process multi-temporal images as a "video"
# Each timestep is a frame
temporal_images = [
    "sentinel2_2024_01.tif",
    "sentinel2_2024_04.tif",
    "sentinel2_2024_07.tif",
    "sentinel2_2024_10.tif",
]

# Prompt on first frame, propagate to all frames
sam2.set_video(temporal_images)
sam2.predict_video(
    point_coords=[[-122.419, 37.775]],
    point_labels=[1],
    start_frame=0,
    output_dir="temporal_masks/",
)
```

### Fine-tuning SAM on Remote Sensing Data

For domain-specific segmentation (e.g., building extraction, crop field delineation), fine-tuning SAM on RS data significantly improves performance:

```python
import torch
from segment_anything import sam_model_registry, SamPredictor
from torch.utils.data import DataLoader

# Load SAM model
sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
sam.to("cuda")

# Freeze image encoder, only train mask decoder and prompt encoder
for param in sam.image_encoder.parameters():
    param.requires_grad = False

# Fine-tuning parameters
trainable_params = list(sam.mask_decoder.parameters()) + \
                   list(sam.prompt_encoder.parameters())
optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)

# Training loop (simplified)
for epoch in range(20):
    sam.train()
    for batch in train_loader:
        images = batch['image'].to("cuda")
        masks = batch['mask'].to("cuda")
        points = batch['point_coords'].to("cuda")
        labels = batch['point_labels'].to("cuda")

        # Encode image
        image_embedding = sam.image_encoder(images)

        # Encode prompts
        sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            points=(points, labels),
            boxes=None,
            masks=None,
        )

        # Decode masks
        pred_masks, iou_predictions = sam.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        # Loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_masks, masks.float()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

# Save fine-tuned model
torch.save(sam.state_dict(), "sam_finetuned_buildings.pth")
```

### SAM Performance on RS Tasks

| Task | SAM (zero-shot) | SAM (fine-tuned) | Specialized Model |
|---|---|---|---|
| Building extraction | mIoU 0.55 | mIoU 0.82 | mIoU 0.85 (U-Net) |
| Crop field delineation | mIoU 0.48 | mIoU 0.78 | mIoU 0.80 (Mask R-CNN) |
| Water body mapping | mIoU 0.65 | mIoU 0.88 | mIoU 0.90 (DeepLab) |
| Road extraction | mIoU 0.35 | mIoU 0.70 | mIoU 0.75 (D-LinkNet) |
| Tree crown detection | mIoU 0.50 | mIoU 0.80 | mIoU 0.82 (Mask R-CNN) |

Fine-tuned SAM is competitive with task-specific models while offering flexibility for interactive use.

---

## Spatial Feature Engineering

Feature engineering is critical for both traditional ML and as auxiliary inputs for deep learning. Spatial features capture the geographic context that non-spatial models miss. Well-engineered features often matter more than model choice.

### Feature Categories

| Category | Examples | How to Compute | Typical Importance |
|---|---|---|---|
| **Spectral indices** | NDVI, NDWI, NDBI, EVI, SAVI, MNDWI, BSI | Band math (Rasterio, GEE) | Very high |
| **Texture features** | GLCM (contrast, entropy, homogeneity, correlation, ASM, dissimilarity) | `scikit-image`, `rasterstats`, Orfeo ToolBox | High |
| **Topographic features** | Slope, aspect, curvature (plan/profile), TPI, TRI, TWI, hillshade | DEM derivatives (GDAL, `richdem`, `WhiteboxTools`) | High for terrain-dependent tasks |
| **Distance features** | Distance to roads, water bodies, urban edges, coastline | Euclidean distance raster (`scipy.ndimage`, GDAL proximity) | Medium-High |
| **Contextual features** | Neighborhood statistics (mean, std, min, max in NxN window) | Focal statistics (Rasterio, `scipy.ndimage`) | Medium |
| **Temporal features** | Phenological metrics (SOS, EOS, peak NDVI, amplitude, integral) | Time series (xarray, TIMESAT, `phenolopy`) | Very high for vegetation |
| **Object features** | Shape (area, perimeter, compactness, elongation), zonal stats | OBIA segmentation + `rasterstats` | High for urban/agriculture |
| **Network features** | Distance to nearest node, network centrality, accessibility | `osmnx`, `networkx`, `pandana` | Medium for urban analysis |
| **Spatial lag features** | Neighboring feature values (spatially weighted average) | Spatial weights (`PySAL/libpysal`, `spdep` in R) | Medium-High |

### Spectral Indices

```python
import rasterio
import numpy as np

def compute_spectral_indices(image_path, sensor='sentinel2'):
    """Compute common spectral indices from a multi-band satellite image."""
    with rasterio.open(image_path) as src:
        if sensor == 'sentinel2':
            # Sentinel-2 band mapping (L2A, 10/20m)
            blue = src.read(2).astype(np.float32)   # B2
            green = src.read(3).astype(np.float32)  # B3
            red = src.read(4).astype(np.float32)    # B4
            nir = src.read(8).astype(np.float32)    # B8
            swir1 = src.read(11).astype(np.float32) # B11
            swir2 = src.read(12).astype(np.float32) # B12

        eps = 1e-10  # Avoid division by zero

        indices = {}

        # Vegetation
        indices['ndvi'] = (nir - red) / (nir + red + eps)
        indices['evi'] = 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)
        indices['savi'] = 1.5 * (nir - red) / (nir + red + 0.5 + eps)

        # Water
        indices['ndwi'] = (green - nir) / (green + nir + eps)
        indices['mndwi'] = (green - swir1) / (green + swir1 + eps)

        # Built-up
        indices['ndbi'] = (swir1 - nir) / (swir1 + nir + eps)
        indices['bsi'] = ((swir1 + red) - (nir + blue)) / \
                         ((swir1 + red) + (nir + blue) + eps)

        # Soil / bare earth
        indices['nbr'] = (nir - swir2) / (nir + swir2 + eps)  # Also for burn severity

        # Moisture
        indices['ndmi'] = (nir - swir1) / (nir + swir1 + eps)

        profile = src.profile

    return indices, profile

# Save indices as multi-band raster
indices, profile = compute_spectral_indices("sentinel2_l2a.tif")
profile.update(count=len(indices), dtype='float32')

with rasterio.open("spectral_indices.tif", 'w', **profile) as dst:
    for i, (name, data) in enumerate(indices.items(), 1):
        dst.write(data, i)
        dst.set_band_description(i, name)
```

### Texture Features (GLCM)

```python
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import generic_filter

def compute_glcm_features(band, window_size=21, levels=64, distances=[1],
                           angles=[0, np.pi/4, np.pi/2, 3*np.pi/4]):
    """Compute GLCM texture features using a sliding window."""
    # Quantize to fewer levels for computational efficiency
    band_min, band_max = np.nanpercentile(band, [2, 98])
    quantized = np.clip((band - band_min) / (band_max - band_min + 1e-10), 0, 1)
    quantized = (quantized * (levels - 1)).astype(np.uint8)

    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy',
                  'correlation', 'ASM']

    pad = window_size // 2
    padded = np.pad(quantized, pad, mode='reflect')

    results = {prop: np.zeros_like(band, dtype=np.float32) for prop in properties}

    for row in range(band.shape[0]):
        for col in range(band.shape[1]):
            window = padded[row:row+window_size, col:col+window_size]
            glcm = graycomatrix(window, distances=distances, angles=angles,
                               levels=levels, symmetric=True, normed=True)

            for prop in properties:
                # Average over all distances and angles
                results[prop][row, col] = graycoprops(glcm, prop).mean()

    return results

# Faster alternative: pre-computed GLCM with rasterstats
# or use Orfeo ToolBox (OTB) for large rasters:
# otbcli_HaralickTextureExtraction -in image.tif -out textures.tif
#   -parameters.xrad 5 -parameters.yrad 5 -parameters.nbbin 32
```

### Topographic Features

```python
import richdem as rd
import numpy as np

def compute_topographic_features(dem_path):
    """Compute terrain derivatives from a DEM."""
    dem = rd.LoadGDAL(dem_path)

    features = {}

    # Basic terrain metrics
    features['slope'] = rd.TerrainAttribute(dem, attrib='slope_riserun')
    features['aspect'] = rd.TerrainAttribute(dem, attrib='aspect')
    features['curvature'] = rd.TerrainAttribute(dem, attrib='curvature')
    features['planform_curv'] = rd.TerrainAttribute(dem, attrib='planform_curvature')
    features['profile_curv'] = rd.TerrainAttribute(dem, attrib='profile_curvature')

    # Topographic Position Index (TPI)
    # How much higher/lower than surroundings
    from scipy.ndimage import uniform_filter
    dem_arr = np.array(dem, dtype=np.float32)
    for radius in [3, 10, 30]:
        mean_elev = uniform_filter(dem_arr, size=2*radius+1)
        features[f'tpi_{radius}'] = dem_arr - mean_elev

    # Terrain Ruggedness Index (TRI)
    from scipy.ndimage import generic_filter
    def tri_func(x):
        center = x[len(x)//2]
        return np.sqrt(np.mean((x - center)**2))
    features['tri'] = generic_filter(dem_arr, tri_func, size=3)

    # Topographic Wetness Index (TWI)
    # TWI = ln(specific_catchment_area / tan(slope))
    # Requires flow accumulation -- use WhiteboxTools for proper computation:
    # import whitebox
    # wbt = whitebox.WhiteboxTools()
    # wbt.d_inf_flow_accumulation("dem.tif", "flow_acc.tif")

    return features
```

### Temporal / Phenological Features

```python
import xarray as xr
import numpy as np
from scipy.signal import savgol_filter

def compute_phenology_features(ndvi_timeseries, dates):
    """
    Compute phenological metrics from NDVI time series.

    Args:
        ndvi_timeseries: array of shape (n_dates, height, width)
        dates: list of datetime objects
    """
    # Smooth the time series
    smoothed = savgol_filter(ndvi_timeseries, window_length=5, polyorder=2, axis=0)

    features = {}

    # Basic statistics
    features['ndvi_mean'] = np.nanmean(smoothed, axis=0)
    features['ndvi_max'] = np.nanmax(smoothed, axis=0)
    features['ndvi_min'] = np.nanmin(smoothed, axis=0)
    features['ndvi_std'] = np.nanstd(smoothed, axis=0)
    features['ndvi_range'] = features['ndvi_max'] - features['ndvi_min']

    # Amplitude (proxy for vegetation vigor)
    features['ndvi_amplitude'] = features['ndvi_max'] - features['ndvi_min']

    # Date of maximum NDVI (proxy for peak growing season)
    doy = np.array([d.timetuple().tm_yday for d in dates])
    max_idx = np.nanargmax(smoothed, axis=0)
    features['peak_doy'] = doy[max_idx]

    # Integrated NDVI (proxy for total productivity)
    features['ndvi_integral'] = np.trapz(np.clip(smoothed, 0, None), axis=0)

    # Rate of green-up (spring increase rate)
    diffs = np.diff(smoothed, axis=0)
    features['max_greenup_rate'] = np.nanmax(diffs, axis=0)
    features['max_senescence_rate'] = np.nanmin(diffs, axis=0)

    # Number of growing cycles (for double-cropping detection)
    # Count peaks in smoothed time series
    from scipy.signal import find_peaks
    features['n_peaks'] = np.zeros_like(features['ndvi_mean'])
    for r in range(smoothed.shape[1]):
        for c in range(smoothed.shape[2]):
            peaks, _ = find_peaks(smoothed[:, r, c], height=0.3, distance=3)
            features['n_peaks'][r, c] = len(peaks)

    return features
```

### Spatial Lag Features

```python
import geopandas as gpd
from libpysal.weights import Queen, KNN, DistanceBand
import numpy as np

def compute_spatial_lag_features(gdf, feature_cols, weight_type='queen'):
    """Compute spatially lagged features using PySAL spatial weights."""
    # Build spatial weights
    if weight_type == 'queen':
        w = Queen.from_dataframe(gdf)
    elif weight_type == 'knn':
        w = KNN.from_dataframe(gdf, k=8)
    elif weight_type == 'distance':
        w = DistanceBand.from_dataframe(gdf, threshold=1000)  # meters

    w.transform = 'r'  # Row-standardize

    lag_features = {}
    for col in feature_cols:
        values = gdf[col].values
        # Spatial lag = weighted average of neighbors
        lag_features[f'{col}_lag'] = np.array([
            np.sum([values[j] * w[i][j] for j in w[i]]) if len(w[i]) > 0 else 0
            for i in range(len(gdf))
        ])

        # Local Moran's I (measures local spatial autocorrelation)
        z = (values - values.mean()) / values.std()
        lag_z = np.array([
            np.sum([z[j] * w[i][j] for j in w[i]]) if len(w[i]) > 0 else 0
            for i in range(len(gdf))
        ])
        lag_features[f'{col}_local_moran'] = z * lag_z

    return lag_features
```

### Feature Engineering Tips

- **Avoid data leakage:** Do not compute spatial features (e.g., focal stats, spatial lag) using information from the test set area. Apply transformations only on training-set-visible data.
- **Collinearity:** Spectral indices are often correlated with their source bands. Use VIF (Variance Inflation Factor > 10 is concerning) or feature importance to select.
- **Scale matters:** A 3x3 texture window captures fine texture; a 21x21 window captures coarse texture. Test multiple scales and let the model select.
- **Feature selection strategies:**
  - Random Forest feature importance (MDI or permutation-based)
  - Recursive feature elimination (RFE) with spatial CV
  - Boruta algorithm (wrapper around RF importance)
  - SHAP values for model-agnostic importance
- **Normalization:** Per-band standardization (z-score) is usually better than min-max for satellite data. Clip outliers at 2nd/98th percentile first.

---

## Object Detection in Remote Sensing

Object detection in RS presents unique challenges: extreme scale variation (ships vs. airports), arbitrary orientation, very high resolution imagery, and massive image sizes requiring tiled inference.

### YOLO v8+ for Remote Sensing

[Ultralytics YOLO](https://docs.ultralytics.com/) (v8, v9, v11) has become the default choice for RS object detection due to speed and accuracy.

```python
from ultralytics import YOLO

# Train YOLO on a remote sensing dataset
model = YOLO('yolov8m.pt')  # Start from COCO pre-trained

# Dataset in YOLO format (images/ and labels/ with .txt annotations)
results = model.train(
    data='rs_dataset.yaml',
    epochs=100,
    imgsz=1024,        # Larger for RS (objects are small)
    batch=8,
    patience=20,
    augment=True,
    mosaic=0.5,         # Mosaic augmentation
    mixup=0.1,
    copy_paste=0.1,     # Copy-paste augmentation (good for RS)
    degrees=180,        # Full rotation (objects can face any direction)
    scale=0.5,
    fliplr=0.5,
    flipud=0.5,         # Vertical flip (unlike natural images, RS is rotation-invariant)
    device='cuda',
)
```

#### YOLO Dataset YAML for RS

```yaml
# rs_dataset.yaml
path: /data/rs_detection
train: images/train
val: images/val
test: images/test

names:
  0: building
  1: vehicle
  2: ship
  3: aircraft
  4: storage_tank
  5: bridge
  6: harbor
  7: swimming_pool
```

### Rotated Bounding Boxes (OBB)

Many RS objects (ships, aircraft, fields) have arbitrary orientation. Standard axis-aligned bounding boxes waste area; oriented bounding boxes (OBB) are more precise:

```python
from ultralytics import YOLO

# YOLOv8 with oriented bounding box (OBB) support
model = YOLO('yolov8m-obb.pt')

# Train on DOTA-format dataset with rotated annotations
results = model.train(
    data='dota_dataset.yaml',
    epochs=100,
    imgsz=1024,
    batch=8,
)

# Inference
results = model.predict('aerial_image.tif', conf=0.25, iou=0.5)
for r in results:
    obb = r.obb  # Oriented bounding boxes
    # obb.xyxyxyxy: 4 corner points
    # obb.cls: class IDs
    # obb.conf: confidence scores
```

### Small Object Detection

Remote sensing imagery contains many small objects (vehicles, trees, small buildings). Strategies for improving small object detection:

| Strategy | How | Impact |
|---|---|---|
| **Higher input resolution** | Train/infer at 1024 or 1280 instead of 640 | +5-15% mAP on small objects |
| **Tiled inference (SAHI)** | Slice large images into overlapping tiles, merge detections | Essential for large images |
| **Feature Pyramid Network (FPN)** | Multi-scale feature maps (built into YOLO) | Already included |
| **Copy-paste augmentation** | Paste small objects randomly during training | +3-8% on rare small classes |
| **Attention modules** | Add CBAM or SE blocks to backbone | +2-5% mAP |

```python
# SAHI (Slicing Aided Hyper Inference) for large images
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='best.pt',
    confidence_threshold=0.3,
    device='cuda',
)

# Slice 10000x10000 image into overlapping 1024x1024 tiles
result = get_sliced_prediction(
    image='large_aerial.tif',
    detection_model=detection_model,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    perform_standard_pred=True,      # Also run full-image prediction
    postprocess_type='GREEDYNMM',    # Merge overlapping detections
    postprocess_match_metric='IOS',
    postprocess_match_threshold=0.5,
)

# Export detections as GeoJSON (if input is georeferenced)
result.export_visuals(export_dir="output/", file_name="detections")
```

### DOTA Dataset Workflow

The [DOTA dataset](https://captain-whu.github.io/DOTA/) is the standard benchmark for oriented object detection in aerial images:

| Version | Images | Instances | Classes | Resolution |
|---|---|---|---|---|
| DOTA-v1.0 | 2,806 | 188,282 | 15 | 0.1-1m |
| DOTA-v1.5 | 2,806 | 403,318 | 16 | 0.1-1m |
| DOTA-v2.0 | 11,268 | 1,793,658 | 18 | 0.1-1m |

```python
# DOTA preprocessing: split large images into patches
# Use the official DOTA devkit or ultralytics tools

# Classes in DOTA v2.0:
# plane, ship, storage-tank, baseball-diamond, tennis-court,
# basketball-court, ground-track-field, harbor, bridge,
# large-vehicle, small-vehicle, helicopter, roundabout,
# soccer-ball-field, swimming-pool, container-crane,
# airport, helipad
```

---

## Change Detection with Deep Learning

Change detection identifies differences between images of the same area acquired at different times. Deep learning approaches have significantly advanced the field, enabling pixel-level change maps from bi-temporal or multi-temporal image pairs.

### Approaches

| Approach | Input | Method | Strengths |
|---|---|---|---|
| **Image differencing + DL** | Bi-temporal | Compute difference image, classify with CNN | Simple, interpretable |
| **Siamese networks** | Bi-temporal | Twin encoders share weights, compare features | Learns meaningful change features |
| **Early fusion** | Bi-temporal | Concatenate image pair, feed to single encoder | Simple architecture, captures change directly |
| **Attention-based** | Bi-temporal/multi | Cross-attention between temporal features | Best at capturing subtle changes |
| **BIT (Change Detection Transformer)** | Bi-temporal | Transformer-based token comparison | SOTA on multiple benchmarks |
| **Recurrent (ConvLSTM)** | Multi-temporal | Process temporal sequence with LSTM | Handles arbitrary number of dates |

### Siamese Network for Change Detection

```python
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class SiameseChangeDetector(nn.Module):
    """Siamese network for binary change detection."""

    def __init__(self, encoder_name='resnet50', in_channels=3):
        super().__init__()
        # Shared encoder (siamese = same weights for both images)
        self.encoder = smp.encoders.get_encoder(
            encoder_name, in_channels=in_channels, weights='imagenet'
        )

        # Difference module
        encoder_channels = self.encoder.out_channels

        # Decoder (processes concatenated/differenced features)
        self.decoder = nn.ModuleList()
        for i in range(len(encoder_channels) - 1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(encoder_channels[i] * 2, encoder_channels[i-1],
                                    kernel_size=2, stride=2),
                nn.BatchNorm2d(encoder_channels[i-1]),
                nn.ReLU(),
                nn.Conv2d(encoder_channels[i-1], encoder_channels[i-1],
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(encoder_channels[i-1]),
                nn.ReLU(),
            ))

        self.final = nn.Conv2d(encoder_channels[1], 1, kernel_size=1)

    def forward(self, x1, x2):
        # Encode both images with shared encoder
        features1 = self.encoder(x1)
        features2 = self.encoder(x2)

        # Compute absolute difference and concatenation at each level
        diff_features = []
        for f1, f2 in zip(features1, features2):
            diff = torch.abs(f1 - f2)
            cat = torch.cat([diff, f1 * f2], dim=1)  # Diff + element-wise product
            diff_features.append(cat)

        # Decode
        x = diff_features[-1]
        for i, dec in enumerate(self.decoder):
            x = dec(x)
            if i < len(diff_features) - 2:
                x = x + diff_features[-(i+2)][:, :x.shape[1]]

        return torch.sigmoid(self.final(x))

# Training
model = SiameseChangeDetector(encoder_name='resnet50', in_channels=4)
model = model.to('cuda')
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(50):
    model.train()
    for batch in train_loader:
        img_t1 = batch['image_t1'].to('cuda')  # Before image
        img_t2 = batch['image_t2'].to('cuda')  # After image
        mask = batch['change_mask'].to('cuda')  # Binary change mask

        pred = model(img_t1, img_t2)
        loss = criterion(pred, mask.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### BIT: Binary Change Detection Transformer

The [BIT](https://github.com/justchenhao/BIT_CD) (Binary change detection with a Transformer) is a SOTA approach combining CNN feature extraction with transformer-based change reasoning:

```python
# BIT architecture overview:
# 1. Shared ResNet encoder extracts features from both dates
# 2. Tokenizer converts spatial features to semantic tokens
# 3. Transformer encoder models relationships between tokens
# 4. Transformer decoder reconstructs spatial change map
# 5. Prediction head outputs binary change mask

# Using the BIT implementation:
# pip install open-cd  (OpenCD: Open Change Detection library)

from opencd.models import BIT

model = BIT(
    backbone='resnet18',
    neck_type='fpn',
    head_type='bit',
    num_classes=2,       # Changed / unchanged
    token_len=4,         # Number of semantic tokens
    diff_module='abs',   # How to compare features: 'abs', 'cat', 'euclidean'
)
```

### Change Detection Datasets

| Dataset | Pairs | Resolution | Task | Area |
|---|---|---|---|---|
| LEVIR-CD | 637 | 0.5m | Building change | Texas, USA |
| WHU-CD | 1 pair | 0.075m | Building change | Christchurch, NZ |
| DSIFN-CD | 3,940 | 2m | Multi-class change | China cities |
| CDD (Season) | 16,000 | 0.03-1m | Season-varying change | Google Earth |
| SECOND | 4,662 | 0.5-3m | Semantic change | Various |
| SpaceNet 7 | 24 cities | 4m | Monthly building | Global |

---

## AutoML for GIS

AutoML tools automate hyperparameter tuning, feature selection, and model selection, saving time on repetitive experimentation while maintaining spatial rigor.

### Tool Comparison

| Tool | Type | GIS Integration | GPU | Strengths | Limitations |
|---|---|---|---|---|---|
| [AutoGluon](https://auto.gluon.ai/) | Tabular + image AutoML | Use with extracted spatial features | Optional | Strong ensemble, minimal config | Large memory footprint |
| [FLAML](https://microsoft.github.io/FLAML/) | Lightweight AutoML | Use with scikit-learn spatial pipelines | No | Fast, low resource usage | Fewer model types |
| [Optuna](https://optuna.org/) | HPO framework | Wrap any GIS ML pipeline | N/A | Bayesian optimization, pruning, flexible | Not end-to-end AutoML |
| [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) | Full AutoML platform | Export models for spatial prediction | Optional | Leaderboard approach, Spark integration | Java dependency |
| [Auto-sklearn](https://automl.github.io/auto-sklearn/) | Sklearn-based AutoML | Drop-in sklearn replacement | No | Meta-learning for warm-starting | Linux only |

### Critical: Spatial CV in AutoML

The biggest mistake in GeoML is using standard random CV within AutoML pipelines. This always gives overoptimistic results due to spatial autocorrelation:

```python
# WRONG: Default AutoGluon with random CV
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor(label='class').fit(train_df)  # Uses random CV!

# RIGHT: Custom spatial CV with Optuna
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 5),
    }

    clf = XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss',
                         random_state=42, n_jobs=-1)
    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(clf, X, y, cv=gkf, groups=block_ids,
                              scoring='f1_macro')
    return scores.mean()

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
)
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best F1 (spatial CV): {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualize optimization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### Full AutoML Pipeline with Spatial Awareness

```python
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import numpy as np

class SpatialAutoML:
    """AutoML wrapper that enforces spatial cross-validation."""

    def __init__(self, X, y, groups, n_splits=5, n_trials=50):
        self.X = X
        self.y = y
        self.groups = groups
        self.n_splits = n_splits
        self.n_trials = n_trials
        self.best_model = None

    def _get_model(self, trial):
        model_type = trial.suggest_categorical('model', ['rf', 'xgb', 'lgbm', 'gb'])

        if model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('rf_n', 100, 500),
                max_depth=trial.suggest_int('rf_depth', 5, 30),
                min_samples_leaf=trial.suggest_int('rf_leaf', 1, 20),
                random_state=42, n_jobs=-1,
            )
        elif model_type == 'xgb':
            return XGBClassifier(
                n_estimators=trial.suggest_int('xgb_n', 100, 500),
                max_depth=trial.suggest_int('xgb_depth', 3, 10),
                learning_rate=trial.suggest_float('xgb_lr', 0.01, 0.3, log=True),
                use_label_encoder=False, eval_metric='mlogloss',
                random_state=42, n_jobs=-1,
            )
        elif model_type == 'lgbm':
            return LGBMClassifier(
                n_estimators=trial.suggest_int('lgbm_n', 100, 500),
                num_leaves=trial.suggest_int('lgbm_leaves', 15, 63),
                learning_rate=trial.suggest_float('lgbm_lr', 0.01, 0.3, log=True),
                verbose=-1, random_state=42, n_jobs=-1,
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=trial.suggest_int('gb_n', 100, 300),
                max_depth=trial.suggest_int('gb_depth', 3, 8),
                learning_rate=trial.suggest_float('gb_lr', 0.01, 0.3, log=True),
                random_state=42,
            )

    def _objective(self, trial):
        model = self._get_model(trial)
        gkf = GroupKFold(n_splits=self.n_splits)
        scores = []
        for train_idx, test_idx in gkf.split(self.X, self.y, groups=self.groups):
            model.fit(self.X[train_idx], self.y[train_idx])
            pred = model.predict(self.X[test_idx])
            scores.append(f1_score(self.y[test_idx], pred, average='macro'))
        return np.mean(scores)

    def fit(self):
        study = optuna.create_study(direction='maximize')
        study.optimize(self._objective, n_trials=self.n_trials,
                        show_progress_bar=True)

        # Retrain best model on all data
        best_trial = study.best_trial
        self.best_model = self._get_model(best_trial)
        self.best_model.fit(self.X, self.y)
        self.best_score = study.best_value
        self.study = study
        return self

# Usage
automl = SpatialAutoML(X, y, groups=block_ids, n_trials=100)
automl.fit()
print(f"Best model: {type(automl.best_model).__name__}")
print(f"Best spatial CV F1: {automl.best_score:.4f}")
```

### SHAP Interpretability for Spatial Models

```python
import shap

# Explain model predictions with SHAP
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot (feature importance)
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Spatial SHAP: map SHAP values back to geography
import geopandas as gpd

gdf_test = gpd.GeoDataFrame(
    {'shap_ndvi': shap_values[:, feature_names.index('ndvi')],
     'shap_slope': shap_values[:, feature_names.index('slope')]},
    geometry=test_points.geometry,
    crs=test_points.crs,
)
gdf_test.to_file("shap_spatial.gpkg")
# Visualize SHAP values on a map to understand spatial drivers
```

---

## GeoAI with LLMs

Large Language Models (Claude, GPT-4, Gemini) are increasingly used for spatial reasoning, GIS code generation, and as orchestrators in geospatial agent workflows.

### Use Cases

| Use Case | Description | Example Prompt | Best LLM |
|---|---|---|---|
| **Code generation** | Generate Python/R spatial analysis scripts | "Write a GeoPandas script to find all buildings within 500m of a river" | Claude, GPT-4 |
| **Spatial reasoning** | Answer questions about geographic relationships | "Which neighborhoods are most accessible to the new hospital?" | Claude (strong reasoning) |
| **Data interpretation** | Explain spatial patterns in results | "Interpret this Moran's I scatterplot showing housing prices" | Claude, GPT-4 |
| **Methodology design** | Suggest appropriate analysis methods | "What method should I use for land suitability with 5 criteria?" | Claude |
| **Literature synthesis** | Summarize spatial analysis methods | "Compare GWR and MGWR for housing price analysis" | Claude, GPT-4 |
| **Data discovery** | Find relevant geospatial datasets | "What elevation data is available for the Alps at 30m resolution?" | Claude, GPT-4 |
| **Debugging spatial code** | Fix CRS issues, topology errors | "My spatial join returns empty -- debug this code" | Claude |

### MCP Servers for GIS Tools

Model Context Protocol (MCP) servers allow LLMs to directly interact with GIS tools and data:

```json
{
  "mcpServers": {
    "gdal-tools": {
      "command": "python",
      "args": ["-m", "gdal_mcp_server"],
      "description": "GDAL/OGR operations via MCP"
    },
    "postgis": {
      "command": "python",
      "args": ["-m", "postgis_mcp_server"],
      "env": {
        "PGHOST": "localhost",
        "PGDATABASE": "gis_db"
      },
      "description": "PostGIS spatial queries via MCP"
    },
    "gee": {
      "command": "python",
      "args": ["-m", "earthengine_mcp_server"],
      "description": "Google Earth Engine access via MCP"
    }
  }
}
```

### Agent Workflows for GIS

LLM agents can orchestrate complex geospatial analyses by chaining tool calls:

```python
# Example: LLM agent workflow for site suitability analysis
# The LLM orchestrates these steps by calling tools:

# Step 1: LLM calls GEE MCP server to download land cover data
# Step 2: LLM calls GDAL MCP server to reproject and clip
# Step 3: LLM generates Python code for suitability scoring
# Step 4: LLM calls PostGIS MCP server to find parcels meeting criteria
# Step 5: LLM generates a summary report with maps

# Pseudocode for an agent loop:
from anthropic import Anthropic

client = Anthropic()

tools = [
    {"name": "run_gdal", "description": "Execute GDAL/OGR commands"},
    {"name": "query_postgis", "description": "Run spatial SQL queries"},
    {"name": "run_python", "description": "Execute Python geospatial code"},
    {"name": "fetch_data", "description": "Download from GEE/STAC/WMS"},
]

messages = [{"role": "user", "content": """
    Find suitable locations for a solar farm in Alameda County, CA.
    Criteria: slope < 5 degrees, not in wetlands, within 2km of road,
    parcel size > 10 acres. Use Sentinel-2 for current land cover.
"""}]

# Agent loop: LLM decides which tools to call and in what order
# until the analysis is complete
```

### Prompt Engineering for GIS

Best practices for getting accurate geospatial code from LLMs:

```
# GOOD prompt (specific, constrained):
"Using GeoPandas and Rasterio, write a Python function that:
1. Reads a Sentinel-2 GeoTIFF (bands: B2, B3, B4, B8)
2. Computes NDVI
3. Clips to a study area polygon (from a GeoPackage)
4. Saves the result as a Cloud-Optimized GeoTIFF
Input CRS is EPSG:32632. Output should preserve this CRS.
Handle nodata values (value: 0) properly."

# BAD prompt (vague):
"Write code to process satellite data"
```

Key tips:
- **Always specify CRS** (EPSG code, not just "UTM")
- **Provide data schema** (column names, geometry types, band order)
- **Name specific libraries** (GeoPandas, not "a GIS library")
- **Mention edge cases** (nodata, CRS mismatches, large files)
- **Request chunked processing** for large rasters
- **Ask for error handling** in production code

---

## MLOps for Geospatial

Deploying and maintaining ML models for geospatial applications requires specialized MLOps practices that account for spatial data characteristics, large file sizes, and evolving landscapes.

### Experiment Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# Set up MLflow experiment
mlflow.set_experiment("land_cover_classification")

with mlflow.start_run(run_name="rf_spatial_cv"):
    # Log parameters
    params = {
        'n_estimators': 300,
        'max_depth': 20,
        'spatial_cv_blocks': 5,
        'block_size_m': 5000,
        'study_area': 'bavaria_germany',
        'image_date': '2024-06-15',
        'sensor': 'sentinel-2',
        'n_features': X.shape[1],
        'n_samples': X.shape[0],
    }
    mlflow.log_params(params)

    # Train with spatial CV
    model = RandomForestClassifier(**{k: v for k, v in params.items()
                                      if k in ['n_estimators', 'max_depth']},
                                    random_state=42, n_jobs=-1)

    # Log metrics per fold
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=block_ids)):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        f1 = f1_score(y[test_idx], pred, average='macro')
        fold_scores.append(f1)
        mlflow.log_metric(f'f1_fold_{fold}', f1)

    mlflow.log_metric('f1_mean', np.mean(fold_scores))
    mlflow.log_metric('f1_std', np.std(fold_scores))

    # Log model
    model.fit(X, y)
    mlflow.sklearn.log_model(model, "model")

    # Log feature importance
    importance = model.feature_importances_
    for name, imp in zip(feature_names, importance):
        mlflow.log_metric(f'importance_{name}', imp)

    # Log artifacts
    mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact("classification_map.tif")
```

### Weights & Biases for DL Training

```python
import wandb

wandb.init(
    project="rs-segmentation",
    config={
        "architecture": "unet",
        "encoder": "resnet50",
        "input_channels": 13,
        "classes": 10,
        "patch_size": 512,
        "epochs": 50,
        "lr": 1e-4,
        "loss": "combined_ce_dice",
        "sensor": "sentinel-2",
        "region": "central_europe",
    }
)

# Log training metrics
for epoch in range(50):
    train_loss, val_iou = train_one_epoch(model, train_loader, val_loader)
    wandb.log({
        "train_loss": train_loss,
        "val_mIoU": val_iou,
        "epoch": epoch,
        "lr": optimizer.param_groups[0]['lr'],
    })

    # Log prediction samples
    if epoch % 10 == 0:
        wandb.log({
            "predictions": wandb.Image(prediction_visualization),
            "confusion": wandb.plot.confusion_matrix(
                y_true=val_labels, preds=val_preds, class_names=class_names
            ),
        })

wandb.finish()
```

### Model Export and Deployment

#### ONNX Export

```python
import torch
import onnx
import onnxruntime as ort

# Export PyTorch model to ONNX
model.eval()
dummy_input = torch.randn(1, 4, 512, 512).to(device)

torch.onnx.export(
    model, dummy_input, "segmentation_model.onnx",
    input_names=['image'],
    output_names=['prediction'],
    dynamic_axes={'image': {0: 'batch_size'}, 'prediction': {0: 'batch_size'}},
    opset_version=17,
)

# Verify and optimize
onnx_model = onnx.load("segmentation_model.onnx")
onnx.checker.check_model(onnx_model)

# Inference with ONNX Runtime (2-5x faster than PyTorch)
session = ort.InferenceSession("segmentation_model.onnx",
                                providers=['CUDAExecutionProvider'])

# Batch inference on raster tiles
import rasterio
import numpy as np

with rasterio.open("large_image.tif") as src:
    for ji, window in src.block_windows(1):
        tile = src.read(window=window).astype(np.float32)
        tile = (tile - mean) / std  # Normalize
        tile = tile[np.newaxis, ...]  # Add batch dim

        result = session.run(None, {'image': tile})[0]
        # Write result to output raster
```

#### TorchServe Deployment

```python
# Package model for TorchServe
# 1. Create model archive
# torch-model-archiver --model-name rs_segmentation \
#     --version 1.0 \
#     --serialized-file best_unet.pth \
#     --handler custom_handler.py \
#     --extra-files model_config.json

# 2. Custom handler for geospatial inference
# custom_handler.py
from ts.torch_handler.base_handler import BaseHandler
import torch
import numpy as np

class RSSegmentationHandler(BaseHandler):
    def preprocess(self, data):
        # Handle GeoTIFF input, normalize bands
        images = []
        for row in data:
            image = np.frombuffer(row['body'], dtype=np.float32)
            image = image.reshape(4, 512, 512)  # 4 bands
            image = (image - self.mean) / self.std
            images.append(image)
        return torch.tensor(np.stack(images)).to(self.device)

    def postprocess(self, output):
        # Return class predictions
        preds = output.argmax(dim=1).cpu().numpy()
        return [pred.tolist() for pred in preds]
```

#### Cloud Deployment Patterns

| Pattern | When to Use | Services |
|---|---|---|
| **Serverless function** | Low-frequency, small tiles | AWS Lambda + S3, Google Cloud Functions |
| **Container service** | Medium traffic, GPU needed | AWS ECS/Fargate, GCP Cloud Run, Azure Container Instances |
| **Kubernetes** | High traffic, autoscaling | EKS, GKE, AKS with GPU node pools |
| **Batch processing** | Large-area prediction | AWS Batch, GCP Dataflow, Azure Batch |
| **Edge deployment** | Drone/satellite onboard | ONNX Runtime + Jetson Nano, TensorRT |

### Model Monitoring

```python
# Monitor model performance in production
# Key metrics to track:

monitoring_metrics = {
    # Data drift
    'input_distribution': 'Compare input feature distributions to training data',
    'spectral_shift': 'Monitor mean reflectance values per band over time',
    'cloud_cover': 'Track cloud/shadow percentage in input images',

    # Prediction drift
    'class_distribution': 'Monitor predicted class proportions over time',
    'confidence_scores': 'Track mean prediction confidence',
    'edge_density': 'Monitor boundary complexity of predictions',

    # Performance (when labels available)
    'accuracy': 'Periodic validation against ground truth',
    'f1_per_class': 'Per-class performance over time',

    # System metrics
    'inference_time': 'Latency per tile',
    'throughput': 'Tiles processed per hour',
    'gpu_utilization': 'GPU memory and compute usage',
}

# Trigger retraining when:
# - Prediction class distribution shifts > 10% from baseline
# - Mean confidence drops below threshold
# - Seasonal changes require new training data
# - New sensor data becomes available
# - Validation accuracy drops below acceptable threshold
```

---

## Benchmark Datasets

Benchmark datasets are essential for comparing models, establishing baselines, and pre-training. This section provides comprehensive details on the most important RS/GIS ML datasets.

### Dataset Summary

| Dataset | Task | Coverage | Patches/Images | Classes | Resolution | Sensor | Size (GB) | Access |
|---|---|---|---|---|---|---|---|---|
| **[EuroSAT](https://github.com/phelber/EuroSAT)** | Scene classification | Europe | 27,000 | 10 | 10m | Sentinel-2 | 2.0 | GitHub |
| **[BigEarthNet](https://bigearth.net/)** | Multi-label classification | Europe | 590,326 | 19/43 | 10/20/60m | Sentinel-2 (+S1) | 66 | Download |
| **[SpaceNet](https://spacenet.ai/)** | Building/road extraction | Global cities | Varies | 2+ | 0.3-0.5m | WorldView, GeoEye | 5-50/challenge | AWS |
| **[xView](http://xviewdataset.org/)** | Object detection | Global | 1,413 images | 60 | 0.3m | WorldView-3 | 20 | Download |
| **[DOTA](https://captain-whu.github.io/DOTA/)** | Oriented object detection | Aerial | 2,806-11,268 | 15-18 | 0.1-1m | Aerial/satellite | 20 | Download |
| **[fMoW](https://github.com/fMoW/dataset)** | Temporal classification | Global | 1M+ | 63 | 0.3-1m | Various VHR | 35 | Download |
| **[SEN12MS](https://mediatum.ub.tum.de/1474000)** | Multi-modal | Global | 180,662 | Land cover | 10m | Sentinel-1/2 | 100 | Download |
| **[LandCover.ai](https://landcover.ai/)** | Segmentation | Poland | 41 images | 4 | 0.25-0.5m | Aerial | 2.2 | Download |
| **[So2Sat LCZ42](https://mediatum.ub.tum.de/1454690)** | LCZ classification | Global cities | 400,000+ | 17 | 10m | Sentinel-1/2 | 48 | Download |
| **[LEVIR-CD](https://chenhao.in/LEVIR/)** | Change detection | Texas, USA | 637 pairs | 2 | 0.5m | Google Earth | 3 | Download |
| **[WHU Building](http://gpcv.whu.edu.cn/data/)** | Building extraction | Global | 8,189 tiles | 2 | 0.075-2m | Aerial/satellite | 4 | Download |
| **[ISPRS Vaihingen/Potsdam](https://www.isprs.org/education/benchmarks.aspx)** | Semantic segmentation | Germany | 33/38 tiles | 6 | 0.05-0.09m | Aerial | 5 | Download |
| **[UCMerced](http://weegee.vision.ucmerced.edu/datasets/landuse.html)** | Scene classification | USA | 2,100 | 21 | 0.3m | Aerial | 0.3 | Download |

### Dataset Selection Guide

| Your Task | Recommended Datasets | Why |
|---|---|---|
| Scene classification | EuroSAT (small), BigEarthNet (large), fMoW (temporal) | Different scales and complexity levels |
| Building extraction | SpaceNet 1-7, WHU Building, ISPRS | VHR with polygon annotations |
| Object detection | xView (horizontal), DOTA (oriented) | Different object types and bbox formats |
| Land cover segmentation | LandCover.ai (small), ISPRS (VHR), SEN12MS (global) | Various resolutions and classes |
| Change detection | LEVIR-CD, WHU-CD, SpaceNet 7 | Bi-temporal and multi-temporal |
| Multi-sensor fusion | SEN12MS (SAR+optical), So2Sat (SAR+optical) | Paired Sentinel-1 and Sentinel-2 |
| Multi-label classification | BigEarthNet (standard), fMoW (temporal context) | Real-world label complexity |
| Foundation model pre-training | SSL4EO-S12, BigEarthNet, SEN12MS | Large-scale, multi-sensor |

### Loading Datasets with TorchGeo

```python
import torchgeo.datasets as datasets
from torch.utils.data import DataLoader

# EuroSAT (auto-download)
eurosat = datasets.EuroSAT(root="data/", download=True)
loader = DataLoader(eurosat, batch_size=32, shuffle=True, num_workers=4)

# BigEarthNet with Sentinel-2 (requires manual download)
bigearthnet = datasets.BigEarthNet(
    root="data/bigearthnet",
    split="train",
    bands=datasets.BigEarthNet.all_band_names,
    num_classes=19,
)

# SpaceNet (from AWS)
# Use torchgeo's built-in SpaceNet datasets
spacenet = datasets.SpaceNet1(root="data/spacenet1", download=True)
```

---

## Framework Comparison

Choosing the right framework depends on your task, data, and deployment requirements.

### Decision Matrix

| Framework | Type | GIS I/O | GPU | Multi-spectral | Pre-trained RS | Best For |
|---|---|---|---|---|---|---|
| **[TorchGeo](https://torchgeo.readthedocs.io/)** | PyTorch datasets + models | Native (GeoTIFF, CRS) | Yes | Yes | Yes (many) | End-to-end RS deep learning |
| **[Raster Vision](https://docs.rastervision.io/)** | ML pipeline framework | Native (raster + vector) | Yes | Yes | Configurable | Reproducible RS ML pipelines |
| **[samgeo](https://samgeo.gishub.org/)** | Zero-shot segmentation | Native (GeoTIFF, GeoJSON) | Yes | RGB only | SAM weights | Interactive segmentation |
| **[scikit-learn](https://scikit-learn.org/)** | Traditional ML | Via GeoPandas | CPU only | Via features | N/A | Tabular spatial features, baselines |
| **[XGBoost](https://xgboost.readthedocs.io/)** | Gradient boosting | Via tabular | GPU optional | Via features | N/A | High-accuracy tabular prediction |
| **[LightGBM](https://lightgbm.readthedocs.io/)** | Gradient boosting | Via tabular | GPU optional | Via features | N/A | Large-scale tabular, fast training |
| **[PyTorch](https://pytorch.org/)** | Deep learning | Via TorchGeo or custom | Yes | Yes | ImageNet, RS | Research, flexibility |
| **[TensorFlow/Keras](https://www.tensorflow.org/)** | Deep learning | Via tf.data | Yes | Yes | ImageNet | TF ecosystem, TFLite deployment |
| **[ONNX Runtime](https://onnxruntime.ai/)** | Model inference | Export from any framework | Yes | Any | Any exported | Fast inference, edge deployment |
| **[smp](https://github.com/qubvel/segmentation_models.pytorch)** | Segmentation models | Via custom dataloader | Yes | Yes | ImageNet | Quick segmentation experiments |

### Framework Selection Guide

| Scenario | Recommended | Why |
|---|---|---|
| **New to ML for RS** | TorchGeo | Pre-built datasets, dataloaders, models, and trainers |
| **Traditional ML on tabular features** | scikit-learn + XGBoost/LightGBM | Fast, interpretable, well-documented |
| **Semantic segmentation (research)** | PyTorch + smp + TorchGeo | Maximum flexibility, latest architectures |
| **Semantic segmentation (production)** | Raster Vision | Config-driven, reproducible, handles I/O |
| **Interactive annotation** | samgeo (SAM) | No training required, prompt-based |
| **Foundation model fine-tuning** | Hugging Face Transformers + TorchGeo | Access to Prithvi, Clay, etc. |
| **Object detection** | Ultralytics YOLO | Best speed/accuracy trade-off |
| **Edge / embedded deployment** | ONNX Runtime or TFLite | Optimized for inference |
| **Large-scale cloud processing** | Raster Vision or custom PyTorch + DASK | Distributed processing |
| **Competition / rapid prototyping** | PyTorch + smp + Albumentations | Fast iteration, strong baselines |

### TorchGeo Complete Example

```python
import torch
import lightning as L
from torchgeo.datamodules import EuroSATDataModule
from torchgeo.trainers import ClassificationTask

# Data module handles downloading, splitting, augmentation
datamodule = EuroSATDataModule(
    root="data/",
    batch_size=64,
    num_workers=8,
    download=True,
)

# Pre-configured training task
task = ClassificationTask(
    model="resnet50",
    weights="sentinel2",       # RS-specific pretrained weights
    num_classes=10,
    lr=1e-3,
    patience=10,
    freeze_backbone=False,
)

# Train with PyTorch Lightning
trainer = L.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",      # Mixed precision for faster training
    callbacks=[
        L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        L.pytorch.callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
    ],
)
trainer.fit(task, datamodule=datamodule)

# Test
trainer.test(task, datamodule=datamodule)
```

### Performance Benchmarks (Approximate)

| Model | Dataset | Task | Metric | Score | Training Time (V100) |
|---|---|---|---|---|---|
| ResNet-50 (TorchGeo) | EuroSAT | Classification | Accuracy | 0.98 | 15 min |
| Swin-B (SatlasPretrain) | BigEarthNet | Multi-label | mAP | 0.87 | 4 hours |
| U-Net (ResNet-50) | SpaceNet 2 | Building extraction | F1 | 0.82 | 2 hours |
| Prithvi-EO 2.0 (600M) | HLS Burn Scars | Segmentation | mIoU | 0.72 | 1 hour (fine-tune) |
| SAM (ViT-H, zero-shot) | Generic RS | Segmentation | mIoU | 0.55-0.70 | N/A |
| SAM (fine-tuned) | Building RS | Segmentation | mIoU | 0.82 | 3 hours |
| YOLOv8-L | DOTA v1.0 | Object detection | mAP@50 | 0.72 | 6 hours |
| BIT (ResNet-18) | LEVIR-CD | Change detection | F1 | 0.89 | 2 hours |
| Clay v1.5 (fine-tuned) | Multi-task EO | Various | Competitive | See paper | 1-4 hours |
| XGBoost (tuned) | Custom LC | Classification | F1 (spatial CV) | 0.85-0.92 | 5 min |

Scores are approximate and depend on fine-tuning setup, data splits, and hardware.

### Installation Quick Reference

```bash
# Core ML stack
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn xgboost lightgbm catboost

# RS deep learning
pip install torchgeo segmentation-models-pytorch albumentations
pip install lightning wandb mlflow

# Geospatial I/O
pip install rasterio geopandas rasterstats richdem

# SAM for geospatial
pip install samgeo segment-anything-2

# Object detection
pip install ultralytics sahi

# AutoML
pip install optuna autogluon flaml

# Foundation models
pip install transformers huggingface-hub

# MLOps
pip install onnx onnxruntime-gpu mlflow[extras]

# Change detection
pip install open-cd  # OpenCD library
```

---

[Back to Data Analysis](README.md) | [Back to Main README](../README.md)
