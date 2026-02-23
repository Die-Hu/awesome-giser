# Machine Learning for GIS

> Practical guidance on applying machine learning to geospatial problems, from traditional supervised classification to deep learning with transformers. Covers frameworks, pre-trained models, spatial feature engineering, and integration patterns.

> **Quick Picks**
> - **Getting started with RS deep learning:** TorchGeo (PyTorch datasets + models)
> - **Zero-shot segmentation:** samgeo (SAM for geospatial)
> - **Foundation model for EO:** Prithvi (IBM/NASA) or Clay Foundation Model
> - **Traditional ML baseline:** scikit-learn + XGBoost with spatial CV
> - **LLM + spatial reasoning:** Claude / GPT-4 for spatial analysis code generation

---

## Table of Contents

- [Supervised Classification](#supervised-classification)
- [Deep Learning for Remote Sensing](#deep-learning-for-remote-sensing)
- [Spatial Feature Engineering](#spatial-feature-engineering)
- [AutoML for GIS](#automl-for-gis)
- [Pre-trained Models & Datasets](#pre-trained-models--datasets)
- [Foundation Models for Earth Observation](#foundation-models-for-earth-observation)
- [GeoAI with LLMs](#geoai-with-llms)
- [Frameworks Comparison](#frameworks-comparison)

---

## Supervised Classification

Traditional machine learning methods for land cover classification, object detection, and spatial prediction.

### Common Algorithms

| Algorithm | Strengths | Weaknesses | Best For |
|---|---|---|---|
| **Random Forest** | Robust, handles noise, feature importance | Can overfit with many correlated features | Land cover classification, general purpose |
| **Gradient Boosting (XGBoost / LightGBM)** | High accuracy, handles imbalanced data | Slower to train, more hyperparameters | Tabular spatial prediction, competition-winning |
| **Support Vector Machine (SVM)** | Works in high dimensions, effective with small samples | Slow on large datasets, sensitive to parameters | Small training sets, hyperspectral data |
| **k-Nearest Neighbors (KNN)** | Simple, no training phase | Slow at prediction, sensitive to irrelevant features | Quick baseline, small datasets |
| **Logistic Regression** | Interpretable, fast, probabilistic output | Assumes linear decision boundary | Binary classification, interpretable models |

### Workflow for Spatial Classification

1. **Define classes** — create a clear, exhaustive, mutually exclusive legend
2. **Collect training data** — field survey, image interpretation, existing maps
3. **Extract features** — spectral bands, indices, texture, DEM derivatives, distance features
4. **Split data** — spatially stratified split (not random!) to avoid spatial autocorrelation leakage
5. **Train model** — with cross-validation (spatial CV recommended)
6. **Tune hyperparameters** — grid search or Bayesian optimization
7. **Predict** — apply model to full extent
8. **Validate** — confusion matrix, overall accuracy, kappa, per-class metrics
9. **Post-process** — majority filter, minimum mapping unit, manual corrections

### Important: Spatial Cross-Validation

Standard k-fold cross-validation can give overly optimistic accuracy estimates for spatial data because nearby samples are correlated. Use spatial CV instead:

| Method | How It Works | Library |
|---|---|---|
| Spatial block CV | Divide study area into spatial blocks, hold out entire blocks | `scikit-learn` + custom, `mlr3` (R) |
| Buffered leave-one-out | Exclude points within a buffer distance of the test point | Custom implementation |
| Leave-location-out | Hold out entire spatial clusters | `GroupKFold` in scikit-learn |

### Spatial Block Cross-Validation Example

```python
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Assign each sample to a spatial block (e.g., grid cell ID)
# block_ids can be computed by dividing coordinates into a grid
block_ids = (X[:, 0] // grid_size).astype(int) * 1000 + \
            (X[:, 1] // grid_size).astype(int)

gkf = GroupKFold(n_splits=5)
scores = []
for train_idx, test_idx in gkf.split(X, y, groups=block_ids):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X[train_idx], y[train_idx])
    scores.append(accuracy_score(y[test_idx], clf.predict(X[test_idx])))

print(f"Spatial CV accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
```

---

## Deep Learning for Remote Sensing

Deep learning has transformed remote sensing analysis, enabling pixel-level segmentation, object detection, and super-resolution.

### Architectures

| Architecture | Task | Input | Key Features |
|---|---|---|---|
| **U-Net** | Semantic segmentation | Image patches | Encoder-decoder with skip connections; standard for RS |
| **DeepLab v3+** | Semantic segmentation | Image patches | Atrous convolutions, multi-scale context |
| **Mask R-CNN** | Instance segmentation | Full images | Detects and segments individual objects |
| **YOLO (v8+)** | Object detection | Full images | Real-time detection, bounding boxes |
| **Vision Transformer (ViT)** | Classification / segmentation | Image patches | Self-attention, captures global context |
| **Swin Transformer** | Segmentation / detection | Image patches | Shifted windows, hierarchical features |
| **SatMAE / Prithvi** | Foundation model (pre-trained) | Multi-spectral patches | Self-supervised pre-training on satellite data |
| **SAM / SAM 2** | Zero-shot segmentation | Any image | Prompt-based segmentation, fine-tunable for RS via samgeo |

### Training Considerations for Remote Sensing

- **Patch size:** Typically 256x256 or 512x512 pixels. Larger patches capture more context but require more GPU memory.
- **Data augmentation:** Random flips, rotations, brightness/contrast adjustment. Be careful with geometric augmentations that break geographic orientation.
- **Class imbalance:** Common in RS (e.g., 90% non-building, 10% building). Use weighted loss, focal loss, or oversampling.
- **Multi-spectral input:** Standard pretrained backbones expect 3-channel RGB. Adapt input convolution for more bands or use RS-specific pretrained models.
- **Transfer learning:** Start from ImageNet weights (for RGB) or RS-specific weights (SatlasPretrain, SSL4EO) for faster convergence.

### SAM for Geospatial (samgeo)

The [samgeo](https://samgeo.gishub.org/) package adapts Meta's Segment Anything Model for geospatial applications:

```python
from samgeo import SamGeo

sam = SamGeo(model_type="vit_h", checkpoint="sam_vit_h_4b8939.pth")

# Automatic mask generation for a satellite image
sam.generate("satellite.tif", output="masks.tif")

# Interactive: prompt with points or bounding boxes
sam.predict("satellite.tif", point_coords=[[lon, lat]],
            point_labels=[1], output="building.tif")
```

### Choosing an Architecture

- **Pixel-level classification (land cover):** U-Net or DeepLab v3+
- **Object detection (buildings, vehicles):** YOLO v8+ or Mask R-CNN
- **Interactive annotation / zero-shot:** SAM via samgeo
- **Multi-task pre-trained backbone:** Prithvi, SatlasPretrain, or Clay
- **Global context / large patches:** Vision Transformer or Swin Transformer

---

## Spatial Feature Engineering

Feature engineering is critical for both traditional ML and as auxiliary inputs for deep learning. Spatial features capture the geographic context that non-spatial models miss.

### Feature Categories

| Category | Examples | How to Compute |
|---|---|---|
| **Spectral indices** | NDVI, NDWI, NDBI, EVI, SAVI | Band math (Rasterio, GEE) |
| **Texture features** | GLCM (contrast, entropy, homogeneity, correlation) | `scikit-image`, `rasterstats`, Orfeo ToolBox |
| **Topographic features** | Slope, aspect, curvature, TPI, TRI, TWI | DEM derivatives (GDAL, `richdem`, `WhiteboxTools`) |
| **Distance features** | Distance to roads, water, urban edges | Euclidean distance raster (`scipy.ndimage`, GDAL) |
| **Contextual features** | Neighborhood statistics (mean, std in NxN window) | Focal statistics (Rasterio, `scipy.ndimage`) |
| **Temporal features** | Phenological metrics (date of green-up, max NDVI, amplitude) | Time series analysis (xarray, TIMESAT) |
| **Object features** | Shape (area, perimeter, compactness), zonal stats per object | OBIA segmentation + `rasterstats` |
| **Network features** | Distance to nearest node, network centrality | `osmnx`, `networkx` |
| **Spatial lag** | Neighboring feature values (weighted average) | Spatial weights (`PySAL`, `spdep`) |

### Tips

- **Avoid data leakage:** Do not compute spatial features using information from the test set area.
- **Collinearity:** Spectral indices are often correlated with their source bands. Use feature importance or VIF to select.
- **Scale matters:** A 3x3 texture window captures different information than a 21x21 window. Test multiple scales.
- **Feature selection:** Use Random Forest feature importance, permutation importance, or recursive feature elimination.

### Example: Computing NDVI and Texture Features

```python
import rasterio
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# NDVI from Sentinel-2 (Band 4 = Red, Band 8 = NIR)
with rasterio.open("sentinel2.tif") as src:
    red = src.read(4).astype(float)
    nir = src.read(8).astype(float)
    ndvi = (nir - red) / (nir + red + 1e-10)

# GLCM texture features
gray = ((ndvi - ndvi.min()) / (ndvi.max() - ndvi.min()) * 255).astype(np.uint8)
glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True)
contrast = graycoprops(glcm, 'contrast')[0, 0]
homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
```

---

## AutoML for GIS

AutoML tools can automate hyperparameter tuning, feature selection, and model selection, saving time on repetitive experimentation.

| Tool | Type | GIS Integration | Notes |
|---|---|---|---|
| [AutoGluon](https://auto.gluon.ai/) | Tabular + image AutoML | Use with extracted spatial features | Strong for tabular data, ensemble-based |
| [FLAML](https://microsoft.github.io/FLAML/) | Lightweight AutoML | Use with scikit-learn spatial pipelines | Fast, low resource usage |
| [Optuna](https://optuna.org/) | Hyperparameter optimization | Wrap any GIS ML pipeline | Bayesian optimization, pruning |
| [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) | Full AutoML platform | Export models for spatial prediction | Leaderboard approach |
| [Auto-sklearn](https://automl.github.io/auto-sklearn/) | Sklearn-based AutoML | Drop-in replacement for sklearn | Meta-learning for warm-starting |

### Considerations for GIS

- Always use **spatial cross-validation** in AutoML pipelines; default random CV will overestimate accuracy.
- Ensure **spatial features** are included as inputs; AutoML cannot invent spatial context from coordinates alone.
- **Interpretability:** Use SHAP or LIME to understand which features drive predictions spatially.
- **Prediction at scale:** AutoML may produce ensemble models that are slower at inference; consider model distillation for large-area prediction.

### Example: Optuna + Spatial CV Pipeline

```python
import optuna
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    clf = GradientBoostingClassifier(**params, random_state=42)
    gkf = GroupKFold(n_splits=5)
    scores = cross_val_score(clf, X, y, cv=gkf, groups=block_ids, scoring='f1_macro')
    return scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
```

---

## Pre-trained Models & Datasets

Pre-trained models and benchmark datasets accelerate development and provide strong baselines.

### Pre-trained Models

| Model | Task | Data | Access |
|---|---|---|---|
| [TorchGeo models](https://torchgeo.readthedocs.io/) | Classification, segmentation | Multi-sensor | PyTorch, pip install |
| [Prithvi (IBM/NASA)](https://huggingface.co/ibm-nasa-geospatial) | Foundation model for EO | HLS (Landsat + Sentinel-2) | Hugging Face |
| [SatlasPretrain](https://github.com/allenai/satlas) | Multi-task RS backbone | Sentinel-2, NAIP | GitHub |
| [SSL4EO](https://github.com/zhu-xlab/SSL4EO-S12) | Self-supervised RS backbone | Sentinel-1/2 | GitHub |
| [Segment Anything (SAM)](https://segment-anything.com/) | Zero-shot segmentation | General images | Meta, Hugging Face |
| [Clay Foundation](https://clay.earth/) | EO foundation model | Multi-sensor | Open source |
| [Raster Vision](https://docs.rastervision.io/) | ML pipeline for RS | Configurable | pip install |

### Benchmark Performance (Approximate)

| Model | Dataset | Task | Metric | Score |
|---|---|---|---|---|
| Prithvi-100M | HLS Burn Scars | Segmentation | mIoU | 0.65 |
| SatlasPretrain (Swin-B) | BigEarthNet | Multi-label | mAP | 0.87 |
| TorchGeo (ResNet-50) | EuroSAT | Classification | Accuracy | 0.98 |
| U-Net (ImageNet init) | SpaceNet 2 | Building extraction | F1 | 0.82 |
| SAM (ViT-H, zero-shot) | Generic RS | Segmentation | mIoU | 0.55-0.70 |
| Clay v1 | Multi-task EO | Various | Competitive | See paper |

Scores are approximate and depend on fine-tuning and evaluation setup.

---

## Foundation Models for Earth Observation

Foundation models are large-scale pre-trained models that can be fine-tuned for multiple downstream tasks with minimal labeled data.

| Model | Organization | Pre-training Data | Architecture | Access |
|---|---|---|---|---|
| [Prithvi](https://huggingface.co/ibm-nasa-geospatial) | IBM + NASA | HLS (Harmonized Landsat-Sentinel) | ViT-based MAE | Hugging Face, open-weight |
| [Clay](https://clay.earth/) | Clay Foundation | Multi-sensor EO (S1, S2, Landsat, NAIP, DEM) | ViT encoder | Open source (GitHub) |
| [SatlasPretrain](https://github.com/allenai/satlas) | Allen AI | Sentinel-2, NAIP | Swin Transformer | GitHub, open |
| [SSL4EO](https://github.com/zhu-xlab/SSL4EO-S12) | TU Munich | Sentinel-1/2 | ResNet/ViT | GitHub, open |
| [GFM (General Foundation Model)](https://github.com/mmendiet/GFM) | Various | Multiple RS sources | ViT | GitHub |

### Using Prithvi for Fine-tuning

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-100M")
# Add a task-specific head (e.g., segmentation decoder)
# Fine-tune on your labeled dataset
```

### Using Clay Foundation Model

```python
# Clay uses a ViT encoder pre-trained on multi-sensor data
# Fine-tune for: classification, segmentation, change detection, similarity search
# See: https://clay.earth/docs for API and fine-tuning guides
```

---

## GeoAI with LLMs

Large Language Models (Claude, GPT-4, etc.) are increasingly used for spatial reasoning and GIS code generation.

### Use Cases

| Use Case | Description | Example |
|---|---|---|
| **Code generation** | Generate Python/R spatial analysis scripts from natural language | "Write a GeoPandas script to find all buildings within 500m of a river" |
| **Spatial reasoning** | Answer questions about geographic relationships | "Which neighborhoods are most accessible to the new hospital?" |
| **Data interpretation** | Explain spatial patterns in results | "Interpret this Moran's I scatterplot" |
| **Methodology design** | Suggest appropriate analysis methods | "What method should I use for land suitability with 5 criteria?" |
| **Literature synthesis** | Summarize spatial analysis methods from papers | "Compare GWR and MGWR for housing price analysis" |

### Best Practices for GIS + LLMs

- **Be specific about CRS, units, and coordinate systems** in prompts
- **Provide data schema** (column names, geometry types) for code generation
- **Validate generated code** -- LLMs can produce plausible but incorrect spatial logic
- **Use for prototyping**, then verify with domain expertise
- **Combine with tool use** -- LLMs can call spatial analysis APIs for grounded answers

### Benchmark Datasets

| Dataset | Task | Coverage | Size | Access |
|---|---|---|---|---|
| [EuroSAT](https://github.com/phelber/EuroSAT) | Scene classification (10 classes) | Europe | 27,000 patches | GitHub |
| [BigEarthNet](https://bigearth.net/) | Multi-label classification | Europe | 590,326 patches | Download |
| [SpaceNet](https://spacenet.ai/) | Building/road extraction | Global cities | Varies by challenge | AWS |
| [LandCover.ai](https://landcover.ai/) | Segmentation (4 classes) | Poland | 41 images | Download |
| [xView](http://xviewdataset.org/) | Object detection (60 classes) | Global | 1 million objects | Download |
| [So2Sat LCZ42](https://mediatum.ub.tum.de/1454690) | Local Climate Zone classification | Global cities | 400,000+ patches | Download |
| [DOTA](https://captain-whu.github.io/DOTA/) | Oriented object detection | Aerial images | 188,282 objects | Download |
| [fMoW (Functional Map of the World)](https://github.com/fMoW/dataset) | Temporal classification (63 classes) | Global | 1M+ images | Download |
| [SEN12MS](https://mediatum.ub.tum.de/1474000) | Multi-modal (SAR + optical) | Global | 180,662 patches | Download |

### Dataset Details

| Dataset | Size (GB) | Resolution | Sensor |
|---|---|---|---|
| EuroSAT | 2.0 | 10m | Sentinel-2 |
| BigEarthNet | 66 | 10/20/60m | Sentinel-2 |
| SpaceNet | 5-50 per challenge | 0.3-0.5m | WorldView, GeoEye |
| LandCover.ai | 2.2 | 0.25-0.5m | Aerial |
| xView | 20 | 0.3m | WorldView-3 |
| So2Sat LCZ42 | 48 | 10m (S2), 10m (S1) | Sentinel-1/2 |
| DOTA | 20 | Various (0.1-1m) | Aerial / satellite |
| fMoW | 35 | 0.3-1m | Various VHR |
| SEN12MS | 100 | 10m | Sentinel-1/2 |

---

## Frameworks Comparison

| Framework | Type | GIS Integration | GPU Support | Best For |
|---|---|---|---|---|
| [TorchGeo](https://torchgeo.readthedocs.io/) | PyTorch datasets + models for RS | Native (reads GeoTIFFs, handles CRS) | Yes | End-to-end RS deep learning |
| [Raster Vision](https://docs.rastervision.io/) | ML pipeline framework | Native (raster + vector I/O) | Yes | Configurable RS ML pipelines |
| [Segment Anything (SAM)](https://segment-anything.com/) | Zero-shot segmentation | Via GeoJSON prompts | Yes | Interactive segmentation, fine-tuning |
| [scikit-learn](https://scikit-learn.org/) | Traditional ML | Via GeoPandas features | CPU only | Tabular spatial features, baselines |
| [XGBoost](https://xgboost.readthedocs.io/) | Gradient boosting | Via tabular features | GPU optional | High-accuracy tabular prediction |
| [TensorFlow / Keras](https://www.tensorflow.org/) | Deep learning | Via tf.data pipeline | Yes | When TF ecosystem is preferred |
| [PyTorch](https://pytorch.org/) | Deep learning | Via TorchGeo or custom | Yes | Research, flexibility |
| [ONNX Runtime](https://onnxruntime.ai/) | Model inference | Export from any framework | Yes | Fast inference, edge deployment |

### Choosing a Framework

- **New to ML for RS:** Start with TorchGeo (pre-built datasets, dataloaders, and models).
- **Traditional ML on tabular features:** scikit-learn + XGBoost.
- **Production pipeline:** Raster Vision (config-driven, reproducible).
- **Interactive annotation:** SAM with GIS prompts.
- **Edge / embedded deployment:** Export to ONNX, run with ONNX Runtime.

### TorchGeo Latest Features

TorchGeo (v0.6+) includes:
- **Pre-trained weights** for ResNet, Swin, and ViT on RS data
- **Built-in datasets** with automatic download (EuroSAT, BigEarthNet, SpaceNet, etc.)
- **Geo-aware data loading** that handles CRS, resolution, and spatial indexing
- **Integration** with PyTorch Lightning for training loops
- **Non-geo datasets** for scene classification benchmarks

```python
import torchgeo.datasets as datasets
from torchgeo.trainers import ClassificationTask

dataset = datasets.EuroSAT(root="data/", download=True)
task = ClassificationTask(model="resnet50", weights="sentinel2", num_classes=10)
```

---

[Back to Data Analysis](README.md) · [Back to Main README](../README.md)
