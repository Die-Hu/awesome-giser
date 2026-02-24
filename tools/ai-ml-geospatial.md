# AI & Machine Learning for Geospatial

> **Quick Picks**
> - **SOTA**: TorchGeo + Prithvi-EO-2.0 — the current best-practice stack for serious EO deep learning
> - **Free Best**: [segment-geospatial](https://github.com/opengeos/segment-geospatial) — SAM adapted for geo, zero training required
> - **Fastest Setup**: Google Earth Engine built-in classifiers — train and classify at planetary scale in a browser

---

## Table of Contents

1. [Deep Learning Frameworks for EO](#deep-learning-frameworks-for-eo)
2. [Foundation Models for Earth Observation](#foundation-models-for-earth-observation)
3. [Traditional ML for Geospatial](#traditional-ml-for-geospatial)
4. [Object Detection in Imagery](#object-detection-in-imagery)
5. [Spatial Statistics & ML](#spatial-statistics--ml)
6. [Point Cloud ML](#point-cloud-ml)
7. [Geospatial NLP & Geocoding ML](#geospatial-nlp--geocoding-ml)
8. [AutoML & No-Code Geo ML](#automl--no-code-geo-ml)
9. [Model Deployment & Serving](#model-deployment--serving)
10. [Benchmarks & Competitions](#benchmarks--competitions)
11. [Advanced Dark Arts](#advanced-dark-arts)

---

## Deep Learning Frameworks for EO

### TorchGeo

[TorchGeo](https://github.com/microsoft/torchgeo) is Microsoft's PyTorch library built specifically for geospatial data. It solves the hardest part of geo deep learning: getting arbitrarily-CRS rasters and labels into clean, spatially-aligned mini-batches.

**Why it matters over vanilla PyTorch + Rasterio:**
- Handles CRS reprojection transparently at dataset level
- Samplers produce geographically correct patches without overlap artifacts
- 50+ benchmark datasets with automatic download and MD5 verification
- Lightning DataModules let you swap datasets with one line change

**Key components:**

| Component | Description |
|-----------|-------------|
| `RasterDataset` | Base class for any GeoTIFF stack |
| `VectorDataset` | Base class for shapefiles/GeoJSONs as label masks |
| `IntersectionDataset` | Automatically aligns two datasets by spatial extent and CRS |
| `UnionDataset` | Merge non-overlapping datasets of the same type |
| `RandomGeoSampler` | Spatially random patch sampling |
| `GridGeoSampler` | Exhaustive grid sampling (for inference) |
| `RandomBlockSampler` | Block-stratified sampling (avoids spatial autocorrelation) |

**Complete TorchGeo training loop — EuroSAT land cover classification:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchgeo.datasets import EuroSAT
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask
import torchvision.models as tvm
import lightning.pytorch as pl
from torchvision.transforms import v2

# --- Data ---
# EuroSAT has 13 Sentinel-2 bands; ResNet18 expects 3 channels.
# TorchGeo lets you specify which bands to load.
dataset_train = EuroSAT(
    root="data/eurosat",
    split="train",
    bands=EuroSAT.all_band_names,   # all 13 bands
    transforms=None,
    download=True,
)
dataset_val = EuroSAT(root="data/eurosat", split="val",
                      bands=EuroSAT.all_band_names)

# --- Model: swap first conv layer to accept 13-band input ---
weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
model = tvm.resnet18()
model.conv1 = nn.Conv2d(13, 64, kernel_size=7, stride=2, padding=3, bias=False)
# Load pretrained weights (trained on Sentinel-2, not ImageNet)
model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
model.fc = nn.Linear(512, EuroSAT.num_classes)  # 10 land cover classes

# --- Lightning Trainer ---
task = ClassificationTask(
    model=model,
    loss="ce",
    lr=1e-3,
    patience=5,
    freeze_backbone=False,
)

datamodule = pl.LightningDataModule.from_datasets(
    train_dataset=dataset_train,
    val_dataset=dataset_val,
    batch_size=64,
    num_workers=8,
)

trainer = pl.Trainer(
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    precision="16-mixed",            # AMP — free perf on modern GPUs
    log_every_n_steps=10,
)
trainer.fit(task, datamodule=datamodule)
```

**TRICK: `IntersectionDataset` — align imagery with labels, no pre-processing needed.**

```python
from torchgeo.datasets import Landsat8, LandCoverAI
from torchgeo.samplers import RandomGeoSampler

# Imagery and labels in completely different CRS and resolution
imagery = Landsat8(root="data/landsat", bands=["B4","B3","B2","B5","B6","B7"])
labels  = LandCoverAI(root="data/landcoverai")

# IntersectionDataset reprojects on-the-fly so every sample is aligned
dataset = imagery & labels  # shorthand for IntersectionDataset(imagery, labels)

sampler = RandomGeoSampler(dataset, size=256, length=1000)
loader  = DataLoader(dataset, sampler=sampler, batch_size=16,
                     collate_fn=dataset.collate_fn, num_workers=4)

for batch in loader:
    images = batch["image"]   # (B, C, 256, 256) float32
    masks  = batch["mask"]    # (B, 256, 256)    int64
    # ... train step
```

**TRICK: Multi-temporal stacking for change detection without manual alignment.**

```python
from torchgeo.datasets import Sentinel2
from torchgeo.samplers import RandomGeoSampler
import torchvision.transforms.functional as TF

# Two Sentinel-2 scenes at different dates
s2_before = Sentinel2(root="data/before", bands=["B02","B03","B04","B08"])
s2_after  = Sentinel2(root="data/after",  bands=["B02","B03","B04","B08"])

# Stack them into a single dataset via intersection
bi_temporal = s2_before & s2_after   # spatial intersection ensures same footprint
# Each sample yields {"image": (8, H, W)} — first 4 bands before, last 4 after
# Feed to a Siamese or difference-based change detection network
```

---

### Raster Vision

[Raster Vision](https://docs.rastervision.io/) (Azavea) is a full-pipeline abstraction from raw GeoTIFF to evaluated model bundle. It handles chipping, augmentation, distributed training, and geo-referenced prediction output.

**Pipeline stages:**

```
ANALYZE → CHIP → TRAIN → PREDICT → EVALUATE → BUNDLE
```

Each stage can be rerun independently. The entire pipeline is defined in Python config:

```python
from rastervision.core.rv_pipeline import SemanticSegmentationConfig
from rastervision.pytorch_backend import PyTorchSemanticSegmentationConfig
from rastervision.pytorch_learner import (
    SemanticSegmentationLearnerConfig,
    SolverConfig,
    UnetBackboneConfig,
)
from rastervision.core.data import (
    ClassConfig,
    SemanticSegmentationLabelSourceConfig,
    RasterioSourceConfig,
    SceneConfig,
    DatasetConfig,
)

class BuildingSegConfig:
    def get_config(self) -> SemanticSegmentationConfig:
        class_config = ClassConfig(
            names=["background", "building"],
            colors=["black", "white"],
        )

        scenes = [
            SceneConfig(
                id="scene_01",
                raster_source=RasterioSourceConfig(
                    uris=["s3://my-bucket/imagery/scene_01.tif"],
                    channel_order=[0, 1, 2],    # RGB
                ),
                label_source=SemanticSegmentationLabelSourceConfig(
                    raster_source=RasterioSourceConfig(
                        uris=["s3://my-bucket/labels/scene_01_mask.tif"]
                    )
                ),
            )
        ]

        return SemanticSegmentationConfig(
            root_uri="s3://my-bucket/rv-output/",
            dataset=DatasetConfig(
                class_config=class_config,
                train_scenes=scenes,
                validation_scenes=scenes,
            ),
            backend=PyTorchSemanticSegmentationConfig(
                model=UnetBackboneConfig(backbone="resnet50"),
                solver=SolverConfig(
                    lr=1e-4, num_epochs=20, batch_sz=8,
                    external_loss_def=None,
                    ignore_class_index=None,
                ),
                augmentors=["RandomFlip", "RandomRotate90"],
            ),
            chip_sz=300,
            chip_options={"stride": 150},          # 50% overlap during chipping
        )
```

**TRICK: `SemanticSegmentationSlidingWindowGeoDataset` for inference on multi-GB GeoTIFFs without manual tiling.**

```python
from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset

predict_ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
    image_uri="very_large_city.tif",
    image_raster_source_config=RasterioSourceConfig(channel_order=[0,1,2]),
    class_config=class_config,
    size=512,
    stride=256,    # 50% overlap → soft-vote at seams
)
# Pass to a DataLoader and run model.predict_step on each batch
# Raster Vision stitches predictions back into a geo-referenced output GeoTIFF
```

---

### segment-geospatial (SAM for Geo)

[segment-geospatial](https://github.com/opengeos/segment-geospatial) wraps Meta's Segment Anything Model with a geospatial-aware API. Zero training required. Point click, text prompt, or fully automatic.

**Install:**
```bash
pip install segment-geospatial
```

**Text-prompted segmentation of satellite imagery:**

```python
from samgeo import SamGeo
from samgeo.text_sam import LangSAM

# --- Automatic everything segmentation ---
sam = SamGeo(
    model_type="vit_h",
    automatic=True,
    device="cuda",
    sam_kwargs={
        "points_per_side": 32,
        "pred_iou_thresh": 0.86,
        "stability_score_thresh": 0.92,
        "crop_n_layers": 1,
        "crop_n_points_downscale_factor": 2,
        "min_mask_region_area": 100,
    },
)

# Download a basemap tile as GeoTIFF, then segment everything in it
sam.tms_to_geotiff(
    output="city_patch.tif",
    bbox=[-87.6298, 41.8781, -87.6098, 41.8981],  # Chicago downtown
    zoom=17,
    source="Satellite",
)
sam.generate("city_patch.tif", output="all_segments.tif")
sam.tiff_to_vector("all_segments.tif", "all_segments.gpkg")

# --- Text-prompted: "find buildings" ---
lang_sam = LangSAM(model_type="vit_h")
lang_sam.predict(
    image="city_patch.tif",
    text_prompt="building",
    box_threshold=0.24,
    text_threshold=0.24,
    output="buildings.tif",
)
lang_sam.raster_to_vector("buildings.tif", "buildings.gpkg")

# --- Interactive point prompt ---
sam_interactive = SamGeo(model_type="vit_h", automatic=False, device="cuda")
sam_interactive.set_image("city_patch.tif")
point_coords = [[longitude, latitude]]          # click location
point_labels = [1]                              # 1=foreground, 0=background
sam_interactive.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    point_crs="EPSG:4326",
    output="clicked_segment.tif",
)
```

**TRICK: Use with high-res basemaps for building extraction without any labeled training data.**

```python
# No training data, no GPU cluster, no labeled dataset.
# Download Bing/Google aerial tiles → SAM → polygons → profit.

import leafmap
m = leafmap.Map(center=[37.7749, -122.4194], zoom=17)
m.add_basemap("SATELLITE")

# Export visible extent as GeoTIFF
m.layer_to_image("SATELLITE", output="sf_patch.tif")

sam = SamGeo(model_type="vit_h", automatic=True, device="cuda")
sam.generate("sf_patch.tif", output="sf_segments.tif")
sam.tiff_to_vector("sf_segments.tif", "sf_buildings.gpkg")

# Now filter by area/shape to isolate building footprints
import geopandas as gpd
gdf = gpd.read_file("sf_buildings.gpkg")
buildings = gdf[(gdf.area > 50) & (gdf.area < 10000)]   # tune thresholds
buildings.to_file("sf_buildings_filtered.gpkg")
```

---

### MMSegmentation / MMDetection for Geo

[OpenMMLab](https://github.com/open-mmlab) provides config-driven training with a huge model zoo. Particularly powerful for remote sensing object detection with oriented bounding boxes.

```python
# Train UNet on a custom satellite segmentation dataset
# configs/custom_unet_satellite.py

_base_ = [
    "../_base_/models/fcn_unet_s5-d16.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_40k.py",
]

model = dict(
    decode_head=dict(num_classes=6),     # e.g., water/forest/urban/crop/barren/cloud
    auxiliary_head=dict(num_classes=6),
)

dataset_type = "CustomDataset"
data_root = "data/my_satellite/"

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type="RandomCrop", crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type="RandomFlip", prob=0.5),
    dict(type="RandomFlip", prob=0.5, direction="vertical"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path="img_dir/train", seg_map_path="ann_dir/train"),
        pipeline=train_pipeline,
    ),
)
```

---

## Foundation Models for Earth Observation

### Prithvi-EO 2.0 (NASA / IBM)

[Prithvi-EO 2.0](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0) is a 600M parameter Vision Transformer pre-trained on the NASA Harmonized Landsat-Sentinel-2 (HLS) dataset. It understands multi-temporal sequences and 6-band multi-spectral imagery natively.

**Model capabilities:**
- Input: multi-temporal sequences of 6-band (Blue, Green, Red, NIR, SWIR1, SWIR2) 224x224 patches
- Pre-trained task: masked image modeling (MAE-style)
- Fine-tune for: flood mapping, crop type classification, wildfire burn scar detection, building segmentation

**Fine-tuning Prithvi for flood mapping:**

```python
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig
from torch.utils.data import DataLoader
import numpy as np

# Load pretrained Prithvi backbone
config = AutoConfig.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-2.0")
backbone = AutoModel.from_pretrained("ibm-nasa-geospatial/Prithvi-EO-2.0")

class PrithviFloodMapper(nn.Module):
    """Prithvi backbone + UNet-style decoder for binary flood segmentation."""

    def __init__(self, backbone, num_classes=2, embed_dim=768):
        super().__init__()
        self.backbone = backbone

        # Freeze early layers, fine-tune last 4 transformer blocks
        for name, param in self.backbone.named_parameters():
            block_num = None
            for part in name.split("."):
                if part.isdigit():
                    block_num = int(part)
                    break
            if block_num is not None and block_num < 8:   # freeze blocks 0-7
                param.requires_grad = False

        # Simple linear decoder head (replace with UPerNet for better results)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64,  32,  kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W) — T=time steps, C=6 bands
        B, T, C, H, W = x.shape
        outputs = self.backbone(pixel_values=x, output_hidden_states=True)
        # Take last hidden state, reshape from patch tokens to spatial grid
        hidden = outputs.last_hidden_state           # (B, N_patches, embed_dim)
        h = w = int(hidden.shape[1] ** 0.5)
        hidden = hidden.reshape(B, h, w, -1).permute(0, 3, 1, 2)  # (B, D, h, w)
        return self.decoder(hidden)                  # (B, num_classes, H, W)


model = PrithviFloodMapper(backbone).cuda()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-4, weight_decay=0.05,
)
criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]).cuda())  # flood rare

# Training loop
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for batch in loader:
        images = batch["image"].cuda()   # (B, T, 6, 224, 224)
        masks  = batch["mask"].cuda()    # (B, 224, 224)
        optimizer.zero_grad()
        logits = model(images)           # (B, 2, 224, 224)
        # Upsample if needed
        logits = nn.functional.interpolate(logits, size=masks.shape[-2:],
                                           mode="bilinear", align_corners=False)
        loss = criterion(logits, masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)
```

---

### Clay Foundation Model

[Clay](https://github.com/Clay-foundation/model) is an open-source EO foundation model trained on Sentinel-1, Sentinel-2, Landsat, and NAIP imagery. Designed for embedding-based retrieval and zero-shot transfer.

```python
# Generate Clay embeddings for STAC-based imagery
import torch
from clay.model import Clay

model = Clay.from_pretrained("made-with-clay/Clay")
model.eval()

# Clay accepts normalized datacube with STAC metadata
datacube = {
    "pixels": torch.randn(1, 6, 256, 256),   # (B, bands, H, W)
    "time":   torch.tensor([[2024, 6, 15]]),  # year, month, day
    "latlon": torch.tensor([[37.77, -122.41]]),
    "gsd":    torch.tensor([10.0]),           # ground sampling distance in meters
    "waves":  torch.tensor([[0.49, 0.56, 0.665, 0.842, 1.61, 2.19]]),  # band wavelengths µm
}

with torch.no_grad():
    embeddings = model.encode(datacube)["cls_token"]   # (B, 768)

# Use for similarity search, clustering, or as feature input to downstream model
```

---

### SatMAE

[SatMAE](https://github.com/sustainlab-group/SatMAE) uses masked autoencoding with temporal and spectral positional encodings — particularly effective for multi-temporal change detection pre-training.

```python
from satmae import SatMAEConfig, SatMAEModel
import torch

config = SatMAEConfig(
    img_size=224,
    patch_size=16,
    in_chans=10,             # e.g., Sentinel-2 10m bands
    temporal_encode=True,
    mask_ratio=0.75,
)
model = SatMAEModel(config)

# Pre-train
imgs    = torch.randn(4, 3, 10, 224, 224)   # (B, T, C, H, W)
timestamps = torch.randint(0, 365, (4, 3))  # day-of-year for each time step
loss, pred, mask = model(imgs, timestamps)
```

---

## Traditional ML for Geospatial

### Google Earth Engine ML

GEE classifiers run server-side on petabytes of data. No downloads, no GPU required.

```javascript
// GEE: Random Forest land cover classification
var composite = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
  .filterDate("2023-06-01", "2023-09-01")
  .filterBounds(geometry)
  .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
  .median()
  .select(["B2","B3","B4","B5","B6","B7","B8","B8A","B11","B12"]);

// Add spectral indices as features
var ndvi = composite.normalizedDifference(["B8","B4"]).rename("NDVI");
var ndwi = composite.normalizedDifference(["B3","B8"]).rename("NDWI");
var ndbi = composite.normalizedDifference(["B11","B8"]).rename("NDBI");
var stack = composite.addBands([ndvi, ndwi, ndbi]);

// Sample training points from labeled polygons
var training = stack.sampleRegions({
  collection: trainingPolygons,
  properties: ["landcover"],
  scale: 10,
  tileScale: 4,
});

// Train Random Forest
var classifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 200,
  variablesPerSplit: 4,
  minLeafPopulation: 10,
  seed: 42,
}).train({
  features: training,
  classProperty: "landcover",
  inputProperties: stack.bandNames(),
});

// Classify and export
var classified = stack.classify(classifier);

// Accuracy assessment with confusion matrix
var validation = stack.sampleRegions({
  collection: validationPolygons,
  properties: ["landcover"],
  scale: 10,
});
var validated = validation.classify(classifier);
var matrix = validated.errorMatrix("landcover", "classification");
print("Overall accuracy:", matrix.accuracy());
print("Kappa:", matrix.kappa());

Export.image.toDrive({
  image: classified,
  description: "landcover_rf",
  scale: 10,
  maxPixels: 1e13,
});
```

**TRICK: Gradient Tree Boost for higher accuracy on imbalanced classes.**

```javascript
var gtb = ee.Classifier.smileGradientTreeBoost({
  numberOfTrees: 100,
  shrinkage: 0.05,         // learning rate
  samplingRate: 0.7,
  maxNodes: 15,
  loss: "LeastAbsoluteDeviation",
  seed: 42,
}).train({
  features: training,
  classProperty: "landcover",
  inputProperties: stack.bandNames(),
});
```

---

### scikit-learn for Raster Classification

Full workflow: GeoTIFF → scikit-learn → classified GeoTIFF with preserved georeferencing.

```python
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report
import geopandas as gpd
from rasterio.features import geometry_mask

# --- 1. Read multi-band imagery ---
with rasterio.open("sentinel2_composite.tif") as src:
    bands = src.read()             # (n_bands, H, W)
    profile = src.profile.copy()
    transform = src.transform
    crs = src.crs

n_bands, H, W = bands.shape

# --- 2. Read labeled polygons and rasterize ---
gdf = gpd.read_file("training_polygons.gpkg").to_crs(crs)

from rasterio.features import rasterize
from shapely.geometry import mapping

label_raster = rasterize(
    shapes=((mapping(geom), cls) for geom, cls in
            zip(gdf.geometry, gdf["class_id"])),
    out_shape=(H, W),
    transform=transform,
    fill=0,         # 0 = no data
    dtype=np.uint8,
)

# --- 3. Flatten to pixel table, filter labeled pixels ---
X = bands.reshape(n_bands, -1).T               # (H*W, n_bands)
y = label_raster.ravel()                       # (H*W,)
mask = y > 0
X_labeled = X[mask]
y_labeled = y[mask]

# --- TRICK: Scale each band separately (crucial for mixed units) ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_labeled)

# --- 4. Train / val split preserving class balance ---
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(X_scaled, y_labeled))

X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
y_train, y_val = y_labeled[train_idx], y_labeled[val_idx]

rf = RandomForestClassifier(
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=5,
    n_jobs=-1,
    oob_score=True,
    random_state=42,
    class_weight="balanced",    # handle class imbalance automatically
)
rf.fit(X_train, y_train)

print(f"OOB score: {rf.oob_score_:.4f}")
print(classification_report(y_val, rf.predict(X_val)))

# --- 5. Predict entire image ---
X_all_scaled = scaler.transform(X)             # (H*W, n_bands)
pred_flat = rf.predict(X_all_scaled)           # (H*W,)
pred_raster = pred_flat.reshape(H, W).astype(np.uint8)

# --- 6. Write classified GeoTIFF with original georeferencing ---
profile.update(count=1, dtype=rasterio.uint8, nodata=0)
with rasterio.open("classified_output.tif", "w", **profile) as dst:
    dst.write(pred_raster, 1)

print("Done. Classified GeoTIFF written.")
```

---

### XGBoost / LightGBM for Spatial Prediction

Ideal for tabular spatial features: terrain derivatives, spectral indices, proximity variables.

```python
import pandas as pd
import geopandas as gpd
import numpy as np
import lightgbm as lgb
import h3
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_squared_error

# --- Feature engineering for spatial ML ---
gdf = gpd.read_file("soil_samples.gpkg")     # point dataset

# Coordinates as raw features (let the model find spatial pattern)
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y

# TRICK: Add H3 cell ID as categorical feature for spatial context
# This gives the model a discrete "neighborhood" token
gdf["h3_7"]  = gdf.apply(lambda r: h3.geo_to_h3(r.lat, r.lon, 7),  axis=1)
gdf["h3_5"]  = gdf.apply(lambda r: h3.geo_to_h3(r.lat, r.lon, 5),  axis=1)
gdf["h3_3"]  = gdf.apply(lambda r: h3.geo_to_h3(r.lat, r.lon, 3),  axis=1)

# Encode H3 IDs
for col in ["h3_7", "h3_5", "h3_3"]:
    gdf[col] = pd.Categorical(gdf[col]).codes

# Terrain derivatives (pre-extracted with richdem/GDAL)
feature_cols = [
    "elevation", "slope", "aspect", "curvature", "twi",
    "ndvi_mean", "ndvi_std", "evi_mean",
    "dist_to_river", "dist_to_road",
    "clay_pct", "sand_pct",
    "lon", "lat",
    "h3_7", "h3_5", "h3_3",
]
X = gdf[feature_cols]
y = gdf["soil_organic_carbon"]

# --- Spatial cross-validation: split by H3 cell to avoid spatial leakage ---
# Group by H3 level-5 cell so nearby points never leak across train/val
groups = gdf["h3_5"]
gss = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

oof_preds = np.zeros(len(gdf))
for fold, (train_idx, val_idx) in enumerate(gss.split(X, y, groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_train, label=y_train,
                             categorical_feature=["h3_7","h3_5","h3_3"])
    val_data   = lgb.Dataset(X_val,   label=y_val,
                             categorical_feature=["h3_7","h3_5","h3_3"])

    params = {
        "objective":      "regression",
        "metric":         "rmse",
        "num_leaves":     63,
        "learning_rate":  0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":   5,
        "min_child_samples": 20,
        "n_jobs":         -1,
        "verbose":        -1,
    }

    model = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    fold_rmse = np.sqrt(mean_squared_error(y_val, oof_preds[val_idx]))
    print(f"Fold {fold}: RMSE = {fold_rmse:.4f}")

overall_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"OOF RMSE: {overall_rmse:.4f}")
```

---

## Object Detection in Imagery

### YOLO for Geospatial

YOLOv8/v9 adapted for high-resolution satellite and aerial imagery. The key challenge is that objects (vehicles, ships, buildings) are tiny relative to the image.

**SAHI (Slicing Aided Hyper Inference) — the essential trick for small objects:**

```python
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
import geopandas as gpd
from shapely.geometry import box
import rasterio
from rasterio.transform import rowcol, xy

# --- Train YOLOv8 on xView or DOTA ---
model_yolo = YOLO("yolov8m.pt")    # medium model, good balance
results = model_yolo.train(
    data="xview.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.001,
    lrf=0.01,
    mosaic=1.0,
    flipud=0.5,         # aerial: vertical flip is valid augmentation
    fliplr=0.5,
    degrees=90,         # aerial: any rotation is valid
    scale=0.5,
    device=0,
    amp=True,
)

# --- SAHI inference on large GeoTIFF ---
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",
    model_path="runs/detect/train/weights/best.pt",
    confidence_threshold=0.35,
    device="cuda:0",
)

# Sliced prediction: 640x640 slices with 20% overlap
result = get_sliced_prediction(
    "large_airport.tif",
    detection_model,
    slice_height=640,
    slice_width=640,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    perform_standard_pred=True,     # also run full-image pass
    postprocess_type="NMM",         # Non-Maximum Merging at stitch seams
    postprocess_match_threshold=0.5,
)

# --- Convert SAHI pixel detections → geographic coordinates ---
with rasterio.open("large_airport.tif") as src:
    transform = src.transform
    crs = src.crs

geoms = []
scores = []
labels = []
for obj in result.object_prediction_list:
    x1, y1, x2, y2 = (obj.bbox.minx, obj.bbox.miny,
                       obj.bbox.maxx, obj.bbox.maxy)
    # Pixel → geographic (top-left anchor)
    lon1, lat1 = xy(transform, y1, x1)
    lon2, lat2 = xy(transform, y2, x2)
    geoms.append(box(lon1, lat2, lon2, lat1))
    scores.append(obj.score.value)
    labels.append(obj.category.name)

gdf = gpd.GeoDataFrame({"label": labels, "score": scores},
                        geometry=geoms, crs=crs)
gdf.to_file("detected_objects.gpkg")
print(f"Detected {len(gdf)} objects")
```

**Oriented Bounding Boxes (OBB) for rotated objects (DOTA dataset):**

```python
# YOLOv8-OBB for oriented detection — critical for ships, aircraft in satellite imagery
model_obb = YOLO("yolov8m-obb.pt")
model_obb.train(
    data="DOTAv1.yaml",
    epochs=100,
    imgsz=1024,
    batch=8,
    device=0,
)
# Predictions include rotation angle: x_center, y_center, width, height, angle
```

---

### Detectron2

```python
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances

# Register your geo dataset (convert labels to COCO format first)
register_coco_instances(
    "buildings_train", {},
    "data/buildings/annotations/train.json",
    "data/buildings/images/train/",
)
register_coco_instances(
    "buildings_val", {},
    "data/buildings/annotations/val.json",
    "data/buildings/images/val/",
)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("buildings_train",)
cfg.DATASETS.TEST  = ("buildings_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.SOLVER.STEPS = (7000, 9000)
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1                  # just "building"
cfg.INPUT.MIN_SIZE_TRAIN = (512, 768, 1024)          # multi-scale training

trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

---

## Spatial Statistics & ML

### PySAL

[PySAL](https://pysal.org/) is the Python Spatial Analysis Library — essential for spatial weights, autocorrelation analysis, and spatial regression.

```python
import libpysal as lps
import esda
import spreg
import geopandas as gpd
import matplotlib.pyplot as plt

gdf = gpd.read_file("census_tracts.gpkg")

# --- Spatial weights matrix ---
w = lps.weights.Queen.from_dataframe(gdf)
w.transform = "R"    # row-standardize

# --- Global Moran's I: is the variable spatially autocorrelated? ---
moran = esda.Moran(gdf["income"], w)
print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")

# --- Local Moran's I (LISA): find hot-spots and cold-spots ---
moran_loc = esda.Moran_Local(gdf["income"], w, seed=42)
gdf["lisa_q"]  = moran_loc.q          # 1=HH, 2=LH, 3=LL, 4=HL
gdf["lisa_p"]  = moran_loc.p_sim
gdf["lisa_sig"] = gdf["lisa_p"] < 0.05

# Classify significant clusters
conditions = [
    (gdf["lisa_sig"]) & (gdf["lisa_q"] == 1),   # High-High (hot spot)
    (gdf["lisa_sig"]) & (gdf["lisa_q"] == 3),   # Low-Low  (cold spot)
    (gdf["lisa_sig"]) & (gdf["lisa_q"] == 2),   # Low-High (spatial outlier)
    (gdf["lisa_sig"]) & (gdf["lisa_q"] == 4),   # High-Low (spatial outlier)
]
choices = ["HH Hot Spot", "LL Cold Spot", "LH Outlier", "HL Outlier"]
import numpy as np
gdf["cluster"] = np.select(conditions, choices, default="Not Significant")

# --- Getis-Ord Gi* for hot-spot detection ---
g_star = esda.G_Local(gdf["income"], w, star=True, seed=42)
gdf["gi_star"] = g_star.Zs
gdf["gi_p"]    = g_star.p_sim

# --- Spatial Lag Model (accounts for spatial autocorrelation in dependent var) ---
y = gdf[["income"]].values
X = gdf[["education", "unemployment", "age_median"]].values

ols = spreg.OLS(y, X, name_y="income",
                name_x=["education","unemployment","age_median"])
print(ols.summary)

# If Lagrange Multiplier tests suggest spatial lag dependence:
lag_model = spreg.ML_Lag(y, X, w=w, name_y="income",
                          name_x=["education","unemployment","age_median"])
print(lag_model.summary)
```

---

### scikit-mobility

```python
import skmob
from skmob.preprocessing import filtering, compression, detection, clustering
from skmob.measures import individual as ind_measures
from skmob.models import EPR, Gravity

# Load GPS trajectory data
tdf = skmob.TrajDataFrame.from_file(
    "gps_traces.csv",
    latitude="lat", longitude="lon",
    datetime="timestamp", user_id="uid",
)

# Preprocess
tdf_filtered    = filtering.filter(tdf, max_speed_kmh=200)
tdf_compressed  = compression.compress(tdf_filtered, spatial_radius_km=0.2)
stdf            = detection.stay_locations(tdf_compressed,
                                           minutes_for_a_stop=20,
                                           spatial_radius_km=0.2)
cstdf           = clustering.cluster(stdf, cluster_radius_km=0.1, min_samples=1)

# Mobility metrics
radius_gyr = ind_measures.radius_of_gyration(tdf_filtered)
n_locs     = ind_measures.number_of_locations(cstdf)
home_loc   = ind_measures.home_location(cstdf)

# Gravity model for flow generation
gravity = Gravity(gravity_type="singly constrained")
fdf = gravity.generate(
    tessellation,                  # GeoDataFrame of zones
    tile_id_column="tile_ID",
    tot_outflows_column="tot_outflow",
    relevance_column="population",
    out_format="flows",
)
```

---

## Point Cloud ML

**RandLA-Net** — efficient semantic segmentation of massive outdoor point clouds.

```python
# RandLA-Net: handles 1M+ points via random sampling with local feature aggregation
# pip install open3d-ml

import open3d.ml.torch as ml3d
import open3d as o3d
import numpy as np

# --- Load point cloud ---
pcd = o3d.io.read_point_cloud("lidar_scan.las")   # or .laz via laspy
points = np.asarray(pcd.points)                    # (N, 3)

# Add intensity and return number as features if available
feat = np.hstack([points, np.ones((len(points), 1))])   # (N, 4) xyz + intensity

# --- Dataset: Toronto-3D, SemanticKITTI, or Semantic3D available in Open3D-ML ---
cfg_file = ml3d.utils.Config.load_from_file(
    "ml3d/configs/randlanet_toronto3d.yml")
model = ml3d.models.RandLANet(**cfg_file.model)

dataset = ml3d.datasets.Toronto3D(dataset_path="data/Toronto3D/",
                                   **cfg_file.dataset)
pipeline = ml3d.pipelines.SemanticSegmentation(
    model, dataset=dataset, device="gpu", **cfg_file.pipeline)

# Train
pipeline.run_train()

# Inference on new scan
results = pipeline.run_inference({"point": points, "feat": feat})
labels  = results["predict_labels"]    # per-point class: ground/vegetation/building/etc.

# Export classified LAS
import laspy
las = laspy.read("lidar_scan.las")
las.classification = labels.astype(np.uint8)
las.write("classified_scan.las")
```

**Applications of point cloud ML:**
- Building vs vegetation vs ground classification (ALS preprocessing)
- Powerline corridor inspection: detect wire catenary and tower structure
- Urban tree inventory: individual tree segmentation and DBH estimation
- Road surface defect detection from mobile LiDAR

---

## Geospatial NLP & Geocoding ML

### mordecai3 — state-of-the-art geoparsing

[mordecai3](https://github.com/openeventdata/mordecai3) uses transformer-based NER + country/city disambiguation to extract and geocode place names from unstructured text.

```python
from mordecai3 import Geoparser

geo = Geoparser()

text = """
Flooding in the Sind province of Pakistan forced over 300,000
people to flee to Sukkur and Larkana. Relief camps were set up
near the banks of the Indus River.
"""

result = geo.geoparse_doc(text)
for ent in result["geolocations"]:
    print(f"  '{ent['name']}' → ({ent['lat']:.4f}, {ent['lon']:.4f}) "
          f"[{ent['country_code3']}] confidence={ent['score']:.2f}")

# Output:
# 'Sind province' → (25.8943, 68.5247) [PAK] confidence=0.91
# 'Pakistan'      → (30.3753, 69.3451) [PAK] confidence=0.98
# 'Sukkur'        → (27.7052, 68.8574) [PAK] confidence=0.95
# 'Larkana'       → (27.5560, 68.2098) [PAK] confidence=0.93
# 'Indus River'   → (25.3960, 68.1839) [PAK] confidence=0.87
```

**TRICK: Fine-tune SpaCy NER for humanitarian report location extraction.**

```python
import spacy
from spacy.tokens import DocBin
from spacy.training import Example
import json

# Start from pretrained transformer model
nlp = spacy.load("en_core_web_trf")
ner = nlp.get_pipe("ner")

# Add custom entity types for humanitarian domain
for label in ["DISASTER_AREA", "CAMP", "CROSSING", "RIVER"]:
    ner.add_label(label)

# Training data: annotated humanitarian situation reports (OCHA ReliefWeb)
# Format: {"text": "...", "entities": [(start, end, label), ...]}
train_data = json.load(open("humanitarian_ner_train.json"))

examples = []
for item in train_data:
    doc = nlp.make_doc(item["text"])
    ents = [doc.char_span(s, e, label=l) for s, e, l in item["entities"]
            if doc.char_span(s, e, label=l) is not None]
    doc.ents = ents
    examples.append(Example.from_dict(doc, {"entities": item["entities"]}))

# Fine-tune only the NER component
optimizer = nlp.resume_training()
nlp.select_pipes(enable=["ner"])

import random
for epoch in range(30):
    random.shuffle(examples)
    losses = {}
    for batch in spacy.util.minibatch(examples, size=8):
        nlp.update(batch, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch}: NER loss = {losses['ner']:.4f}")

nlp.to_disk("humanitarian_ner_model")
```

---

## AutoML & No-Code Geo ML

### ArcGIS Deep Learning

- Built into ArcGIS Pro 3.x — no code required
- Tools: Classify Pixels Using Deep Learning, Detect Objects Using Deep Learning, Classify Point Cloud Using Deep Learning
- ESRI Model Zoo: pre-trained models for building footprints, roads, trees, cars
- Custom training: use `Export Training Data for Deep Learning` → train in Pro or Jupyter → deploy back

### GEE + Vertex AI Pipeline

```python
# Export training chips from GEE, train on Google Cloud, deploy back

# Step 1: GEE — export labeled chips to Cloud Storage
import ee
ee.Initialize(project="my-gcp-project")

task = ee.batch.Export.image.toCloudStorage(
    image=labeled_stack,
    description="training_chips",
    bucket="my-gee-exports",
    fileNamePrefix="chips/scene_001",
    region=aoi,
    scale=10,
    fileFormat="GeoTIFF",
    maxPixels=1e9,
)
task.start()

# Step 2: Vertex AI — train with custom container
from google.cloud import aiplatform

aiplatform.init(project="my-gcp-project", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="prithvi-flood-finetune",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
    requirements=["torchgeo", "transformers"],
)

model = job.run(
    args=["--data-gcs-path", "gs://my-gee-exports/chips/",
          "--epochs", "50", "--lr", "1e-4"],
    replica_count=1,
    machine_type="n1-standard-8",
    accelerator_type="NVIDIA_TESLA_A100",
    accelerator_count=1,
)
```

### Amazon SageMaker Geospatial

```python
import boto3

client = boto3.client("sagemaker-geospatial", region_name="us-west-2")

# Land Cover Segmentation — no model training required
response = client.start_earth_observation_job(
    Name="crop-classification-job",
    InputConfig={
        "RasterDataCollectionQuery": {
            "RasterDataCollectionArn": "arn:aws:sagemaker-geospatial:::raster-data-collection/public/sentinel-2-l2a",
            "AreaOfInterest": {
                "AreaOfInterestGeometry": {
                    "PolygonGeometry": {
                        "Coordinates": [[[78.0, 28.0],[79.0, 28.0],
                                         [79.0, 29.0],[78.0, 29.0],[78.0, 28.0]]]
                    }
                }
            },
            "TimeRangeFilter": {
                "StartTime": "2023-06-01T00:00:00Z",
                "EndTime": "2023-09-01T00:00:00Z",
            },
            "PropertyFilters": {
                "Properties": [{"Property": {"EoCloudCover": {"LowerBound": 0, "UpperBound": 20}}}]
            },
        }
    },
    JobConfig={
        "LandCoverSegmentationConfig": {}    # built-in EO model, no training
    },
)
print(response["Arn"])
```

---

## Model Deployment & Serving

### TorchServe for Geo Models

```python
# custom_handler.py — TorchServe handler for GeoTIFF input
import torch
import numpy as np
import rasterio
import io
from ts.torch_handler.base_handler import BaseHandler

class GeoTIFFHandler(BaseHandler):
    def preprocess(self, data):
        """Accept GeoTIFF bytes, return normalized tensor."""
        tensors = []
        for row in data:
            body = row.get("body") or row.get("data")
            with rasterio.open(io.BytesIO(body)) as src:
                arr = src.read().astype(np.float32)    # (C, H, W)
            # Normalize to [0, 1] per band using dataset statistics
            means = np.array([0.485, 0.456, 0.406, 0.35, 0.30, 0.25])
            stds  = np.array([0.229, 0.224, 0.225, 0.20, 0.18, 0.15])
            arr = (arr - means[:, None, None]) / stds[:, None, None]
            tensors.append(torch.from_numpy(arr))
        return torch.stack(tensors)                    # (B, C, H, W)

    def postprocess(self, data):
        """Return class probabilities as JSON."""
        probs = torch.softmax(data, dim=1).cpu().numpy()
        return [{"probabilities": p.tolist()} for p in probs]
```

### ONNX Export for Edge Deployment

```python
import torch
import torch.onnx
from torchgeo.models import resnet50

model = resnet50(weights=None, in_chans=6, num_classes=10)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

dummy_input = torch.randn(1, 6, 224, 224)

torch.onnx.export(
    model, dummy_input, "geo_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=17,
)

# Verify and optimize with ONNX Runtime
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("geo_classifier.onnx",
                               providers=["CUDAExecutionProvider",
                                          "CPUExecutionProvider"])
ort_inputs = {"input": dummy_input.numpy()}
ort_outs   = session.run(None, ort_inputs)
print("ONNX output shape:", ort_outs[0].shape)
```

### TiTiler + ML: On-the-fly Raster Inference

```python
# Extend TiTiler with a custom endpoint that runs model inference per tile
from fastapi import FastAPI
from titiler.core.factory import TilerFactory
from titiler.core.dependencies import RescalingParams
import torch
import numpy as np

app = FastAPI()
cog = TilerFactory()
app.include_router(cog.router, prefix="/cog", tags=["Cloud Optimized GeoTIFF"])

# Load model once at startup
model = torch.jit.load("geo_model_scripted.pt").cuda().eval()

@app.get("/ml-tile/{z}/{x}/{y}.png")
async def ml_tile(z: int, x: int, y: int, url: str):
    """Fetch tile, run model inference, return classified tile as PNG."""
    from titiler.core.utils import render
    from morecantile import tms

    tile = tms.get("WebMercatorQuad").xy_bounds(x, y, z)
    # ... fetch raster data for tile bounds
    # ... run model
    # ... encode predictions as colored PNG
    pass
```

### Annotation Tools

- **Label Studio** — web-based annotation with geo support (Leaflet/MapLibre integration), supports polygon, bounding box, semantic segmentation for GeoTIFF/COG
- **CVAT** — Computer Vision Annotation Tool by Intel, excellent for aerial/satellite image labeling with rectangle, polygon, polyline tools
- **Roboflow** — managed annotation + augmentation + export to YOLO/COCO formats (good for satellite object detection datasets)
- **Labelbox** — enterprise geo annotation with QGIS plugin for round-trip export

---

## Benchmarks & Competitions

### Standardized Evaluation

| Benchmark | Domain | Metrics | Notes |
|-----------|--------|---------|-------|
| [GEO-Bench](https://github.com/ServiceNow/geo-bench) | Foundation model eval | mIoU, F1, OA | 6 segmentation + 6 classification tasks |
| [OpenEarthMap](https://open-earth-map.org/) | Land cover segmentation | mIoU | 8 classes, 97 regions |
| [BigEarthNet](https://bigearth.net/) | Multi-label classification | F1 | Sentinel-2, 19 or 43 classes |
| [GeoAI Challenge](https://www.geoai.org/) | Various | Task-specific | Annual challenges |

### Competitions

| Competition | Task | Dataset |
|-------------|------|---------|
| [SpaceNet](https://spacenet.ai/challenges/) | Building footprints, roads, flood | DigitalGlobe VHR |
| [IEEE GRSS Data Fusion](http://www.grss-ieee.org/community/technical-committees/data-fusion/) | Multi-source RS | Annual variety |
| [Kaggle Planet: Understanding Amazon](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space) | Multi-label land use | Sentinel/Planet |
| [Kaggle RSNA](https://www.kaggle.com/) | Various geo challenges | Ongoing |
| [DrivenData Competitions](https://www.drivendata.org/) | Disaster response ML | Various |

---

## Advanced Dark Arts

These techniques separate production EO models from academic demos. Most published papers do not describe them, but practitioners rely on them.

### 1. Pseudo-labeling: Weak Labels at Scale

Use GEE classifiers (fast, scalable) to generate noisy labels, then train a DL model to refine them.

```python
# GEE: generate weak labels for 10,000 km² using RF classifier
# (see GEE section above for RF training)
# Export weak-labeled raster as GeoTIFF

# Python: train DL model on weak labels, iterate
import torch
import torch.nn.functional as F

def pseudo_label_loss(logits, weak_labels, confidence_threshold=0.85):
    """
    Only backprop on pixels where model is confident (confident self-training).
    Ignores uncertain predictions and noisy weak labels simultaneously.
    """
    probs      = F.softmax(logits, dim=1)
    max_probs  = probs.max(dim=1).values         # (B, H, W)
    pseudo_lbl = probs.argmax(dim=1)             # (B, H, W)

    # Confidence mask: high-confidence model predictions only
    conf_mask = max_probs > confidence_threshold

    # Also keep pixels where weak label and model agree
    agree_mask = (pseudo_lbl == weak_labels)

    final_mask = conf_mask | agree_mask           # union of both

    if final_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    loss = F.cross_entropy(logits, weak_labels, reduction="none")
    return (loss * final_mask.float()).sum() / final_mask.float().sum()


# Iterative training: generate labels → train → re-label → retrain
for iteration in range(3):
    print(f"=== Pseudo-label iteration {iteration} ===")
    # Train for N epochs on current labels
    for epoch in range(10):
        for batch in loader:
            logits = model(batch["image"].cuda())
            loss = pseudo_label_loss(logits, batch["weak_mask"].cuda())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Re-generate labels using updated model
    regenerate_pseudo_labels(model, unlabeled_loader, output_dir=f"labels_v{iteration+1}")
```

### 2. Domain Adaptation: Sentinel-2 → VHR

```python
import torch
import torch.nn as nn

class DomainAdaptationLoss(nn.Module):
    """
    Gradient Reversal Layer for adversarial domain adaptation.
    Pre-train backbone on Sentinel-2 (abundant, free).
    Adapt to VHR commercial imagery (scarce, expensive).
    """
    def __init__(self, lambda_grl=1.0):
        super().__init__()
        self.lambda_grl = lambda_grl
        self.domain_classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2),    # 0=Sentinel-2, 1=VHR
        )

    def gradient_reversal(self, x):
        """Reverse gradients during backward pass."""
        return GradientReversalLayer.apply(x, self.lambda_grl)

    def forward(self, features_s2, features_vhr):
        # Reverse gradients so backbone learns domain-invariant features
        feat_s2_rev  = self.gradient_reversal(features_s2)
        feat_vhr_rev = self.gradient_reversal(features_vhr)

        domain_pred_s2  = self.domain_classifier(feat_s2_rev)
        domain_pred_vhr = self.domain_classifier(feat_vhr_rev)

        labels_s2  = torch.zeros(len(features_s2),  dtype=torch.long)
        labels_vhr = torch.ones( len(features_vhr), dtype=torch.long)

        loss_s2  = nn.CrossEntropyLoss()(domain_pred_s2,  labels_s2.cuda())
        loss_vhr = nn.CrossEntropyLoss()(domain_pred_vhr, labels_vhr.cuda())
        return (loss_s2 + loss_vhr) / 2.0

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None
```

### 3. Test-Time Augmentation (TTA)

```python
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def tta_predict(model, image_tensor, n_augments=8):
    """
    Flip/rotate patches during inference → average predictions.
    Consistently improves mIoU by 1-3 points on segmentation tasks.
    """
    model.eval()
    augmentations = [
        lambda x: x,                              # original
        lambda x: TF.hflip(x),                   # horizontal flip
        lambda x: TF.vflip(x),                   # vertical flip
        lambda x: TF.hflip(TF.vflip(x)),         # both flips
        lambda x: torch.rot90(x, 1, [-2,-1]),     # 90°
        lambda x: torch.rot90(x, 2, [-2,-1]),     # 180°
        lambda x: torch.rot90(x, 3, [-2,-1]),     # 270°
        lambda x: TF.hflip(torch.rot90(x, 1, [-2,-1])),   # flip + 90°
    ]
    inv_augmentations = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.vflip(TF.hflip(x)),
        lambda x: torch.rot90(x, -1, [-2,-1]),
        lambda x: torch.rot90(x, -2, [-2,-1]),
        lambda x: torch.rot90(x, -3, [-2,-1]),
        lambda x: torch.rot90(x, -1, [-2,-1]),   # then hflip in practice
    ]

    preds = []
    with torch.no_grad():
        for aug, inv_aug in zip(augmentations[:n_augments],
                                inv_augmentations[:n_augments]):
            aug_img  = aug(image_tensor)
            logits   = model(aug_img.unsqueeze(0))
            probs    = F.softmax(logits, dim=1).squeeze(0)
            preds.append(inv_aug(probs))

    return torch.stack(preds).mean(0)             # averaged probability map
```

### 4. Sliding Window with 50% Overlap + Soft Voting

```python
import torch
import torch.nn.functional as F
import numpy as np

def sliding_window_inference(model, image_np, window=512, stride=256, num_classes=10):
    """
    Predict on large image via overlapping windows with Gaussian weight map.
    Gaussian weighting upweights center pixels (more reliable) over edge pixels.
    Eliminates stitching artifacts completely.
    """
    C, H, W = image_np.shape
    output      = torch.zeros(num_classes, H, W)
    weight_map  = torch.zeros(1, H, W)

    # Gaussian weight: center of each tile gets higher weight
    def gaussian_kernel(size, sigma=None):
        if sigma is None:
            sigma = size / 6
        k = torch.arange(size).float() - size // 2
        gauss = torch.exp(-k**2 / (2*sigma**2))
        gauss2d = gauss[:, None] * gauss[None, :]
        return (gauss2d / gauss2d.max()).unsqueeze(0)   # (1, size, size)

    kernel = gaussian_kernel(window)                    # (1, window, window)

    model.eval()
    with torch.no_grad():
        for y in range(0, H - window + 1, stride):
            for x in range(0, W - window + 1, stride):
                patch = image_np[:, y:y+window, x:x+window]
                tensor = torch.from_numpy(patch).unsqueeze(0).float().cuda()
                logits = model(tensor).cpu().squeeze(0)            # (C, window, window)
                probs  = F.softmax(logits, dim=0)
                output[:, y:y+window, x:x+window]     += probs * kernel
                weight_map[:, y:y+window, x:x+window] += kernel

    # Normalize by accumulated weights
    output = output / (weight_map + 1e-8)
    return output.argmax(0).numpy().astype(np.uint8)   # (H, W)
```

### 5. Multi-Scale Inference

```python
def multi_scale_inference(model, image_tensor, scales=[0.75, 1.0, 1.25, 1.5],
                          num_classes=10):
    """
    Predict at multiple zoom levels, combine for objects of different sizes.
    Critical for data with wide range of object scales (e.g., GSD varies).
    """
    H, W    = image_tensor.shape[-2:]
    avg_probs = torch.zeros(num_classes, H, W)

    model.eval()
    with torch.no_grad():
        for scale in scales:
            new_H = int(H * scale)
            new_W = int(W * scale)

            scaled = F.interpolate(image_tensor.unsqueeze(0), size=(new_H, new_W),
                                   mode="bilinear", align_corners=False)
            logits = model(scaled.cuda()).cpu().squeeze(0)
            probs  = F.softmax(logits, dim=0)

            # Resize predictions back to original resolution
            probs_orig = F.interpolate(probs.unsqueeze(0), size=(H, W),
                                       mode="bilinear", align_corners=False).squeeze(0)
            avg_probs += probs_orig

    return (avg_probs / len(scales)).argmax(0)
```

### 6. SAM as Superpixel Pre-processor

```python
# Use SAM segments as superpixels → classify at segment level → much faster
# and produces cleaner boundaries than per-pixel classification

from samgeo import SamGeo
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier

sam = SamGeo(model_type="vit_b", automatic=True, device="cuda")   # vit_b is fastest
sam.generate("scene.tif", output="segments.tif")

# Read original imagery and SAM segment IDs
with rasterio.open("scene.tif") as src:
    bands = src.read().astype(np.float32)       # (C, H, W)
with rasterio.open("segments.tif") as src:
    seg_ids = src.read(1)                       # (H, W) int32 segment labels

# Compute per-segment statistics (mean, std, percentiles)
unique_ids = np.unique(seg_ids[seg_ids > 0])
seg_features, seg_y = [], []

for sid in unique_ids:
    mask = seg_ids == sid
    feat = []
    for b in range(bands.shape[0]):
        band_vals = bands[b][mask]
        feat.extend([band_vals.mean(), band_vals.std(),
                     np.percentile(band_vals, 25), np.percentile(band_vals, 75)])
    # Additional features: segment area, perimeter ratio
    feat.append(mask.sum())                     # area in pixels
    seg_features.append(feat)

X_seg = np.array(seg_features)

# Train RF on labeled segments (label by sampling training polygons)
rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
rf.fit(X_seg_train, y_train)

# Predict all segments → map back to pixels
seg_preds = rf.predict(X_seg)
pixel_pred = np.zeros_like(seg_ids)
for sid, pred in zip(unique_ids, seg_preds):
    pixel_pred[seg_ids == sid] = pred
```

### 7. Combining SAR + Optical (Early/Late Fusion)

```python
class EarlyFusionModel(nn.Module):
    """Concatenate SAR and optical bands at input. Simple, often works well."""
    def __init__(self, n_optical=6, n_sar=2, num_classes=10):
        super().__init__()
        # Standard UNet or similar, with expanded input channels
        self.encoder = get_encoder("resnet50", in_channels=n_optical + n_sar,
                                   weights=None)
        self.decoder = get_decoder(self.encoder, num_classes=num_classes)

    def forward(self, optical, sar):
        x = torch.cat([optical, sar], dim=1)
        return self.decoder(self.encoder(x))


class LateFusionModel(nn.Module):
    """Separate encoders, fuse at feature level. Better when modalities differ a lot."""
    def __init__(self, n_optical=6, n_sar=2, num_classes=10):
        super().__init__()
        self.optical_enc = get_encoder("resnet50", in_channels=n_optical)
        self.sar_enc     = get_encoder("resnet34", in_channels=n_sar)
        feat_dim = self.optical_enc.out_channels[-1] + self.sar_enc.out_channels[-1]
        self.fusion      = nn.Sequential(
            nn.Conv2d(feat_dim, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        self.decoder = get_decoder_from_dim(512, num_classes=num_classes)

    def forward(self, optical, sar):
        opt_feat = self.optical_enc(optical)[-1]    # (B, D1, h, w)
        sar_feat = self.sar_enc(sar)[-1]            # (B, D2, h, w)
        # Resize SAR features to match optical if GSD differs
        sar_feat = F.interpolate(sar_feat, size=opt_feat.shape[-2:], mode="bilinear")
        fused    = self.fusion(torch.cat([opt_feat, sar_feat], dim=1))
        return self.decoder(fused)
```

### 8. Active Learning Loop

```python
import torch
import numpy as np
from scipy.stats import entropy as scipy_entropy

def uncertainty_sampling(model, unlabeled_loader, n_select=500, strategy="entropy"):
    """
    Select the most uncertain pixels/patches for human labeling.
    Iterate: train → score → label → retrain. 5-10 cycles beats a large labeled set.
    """
    model.eval()
    uncertainties = []
    indices       = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(unlabeled_loader):
            imgs = batch["image"].cuda()
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()    # (B, C, H, W)

            for i, prob_map in enumerate(probs):
                if strategy == "entropy":
                    # Mean entropy across all spatial positions
                    ent = scipy_entropy(prob_map, axis=0).mean()   # scalar
                elif strategy == "least_confident":
                    ent = 1.0 - prob_map.max(axis=0).mean()
                elif strategy == "margin":
                    sorted_probs = np.sort(prob_map, axis=0)[::-1]
                    ent = (sorted_probs[0] - sorted_probs[1]).mean()
                    ent = -ent   # higher = more uncertain

                uncertainties.append(ent)
                indices.append((batch_idx, i))

    # Sort by descending uncertainty, select top-N for annotation
    ranked   = sorted(zip(uncertainties, indices), reverse=True)
    selected = [idx for _, idx in ranked[:n_select]]
    return selected

# Active learning workflow
labeled_indices   = initial_labeled_set
unlabeled_indices = set(range(len(full_dataset))) - set(labeled_indices)

for al_round in range(10):
    print(f"=== Active Learning Round {al_round} | Labeled: {len(labeled_indices)} ===")

    # Train on current labeled set
    train_subset  = Subset(full_dataset, list(labeled_indices))
    train_loader  = DataLoader(train_subset, batch_size=16, shuffle=True)
    train_model(model, train_loader, epochs=20)

    # Score unlabeled data
    unlabeled_subset  = Subset(full_dataset, list(unlabeled_indices))
    unlabeled_loader  = DataLoader(unlabeled_subset, batch_size=16)
    to_label          = uncertainty_sampling(model, unlabeled_loader, n_select=100)

    # Simulate oracle labeling (in practice: send to annotation tool)
    newly_labeled = [list(unlabeled_indices)[i] for _, i in to_label]
    labeled_indices   = labeled_indices   + newly_labeled
    unlabeled_indices = unlabeled_indices - set(newly_labeled)
```

---

## Quick Reference

### Library Install Commands

```bash
# Core geo deep learning
pip install torchgeo lightning
pip install rastervision-pytorch-backend
pip install segment-geospatial

# Foundation models
pip install transformers accelerate
# Clay: pip install git+https://github.com/Clay-foundation/model.git

# Traditional ML + spatial
pip install scikit-learn xgboost lightgbm
pip install pysal libpysal esda spreg skmob
pip install h3 shapely geopandas rasterio

# Object detection
pip install ultralytics sahi
# pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Point cloud
pip install open3d laspy

# Geoparsing
pip install mordecai3 spacy
python -m spacy download en_core_web_trf

# Deployment
pip install onnx onnxruntime-gpu torchserve torch-model-archiver
```

### Decision Tree: Which Tool to Use?

```
Do you have labeled training data?
├── No →  Use segment-geospatial (SAM) or GEE unsupervised clustering
└── Yes →
    How large is your imagery footprint?
    ├── Continental/Global → Google Earth Engine classifiers (no downloads needed)
    └── Regional/Local →
        Image-based or tabular features?
        ├── Tabular (points, polygons) → LightGBM + spatial features + H3
        └── Image-based →
            Do you have GPU?
            ├── No → scikit-learn RF on raster bands (works for <10GB)
            └── Yes →
                How much labeled data?
                ├── <1000 samples → Fine-tune Prithvi-EO-2.0 or Clay embeddings
                ├── 1000–50000 → TorchGeo + pretrained backbone
                └── >50000 → Train from scratch with TorchGeo or MMSeg
```

### Key Papers

| Paper | Year | Contribution |
|-------|------|--------------|
| [Prithvi-EO 2.0](https://arxiv.org/abs/2412.02732) | 2024 | NASA/IBM 600M EO foundation model |
| [SatMAE](https://arxiv.org/abs/2207.08051) | 2022 | Temporal masked autoencoder for satellite |
| [TorchGeo](https://arxiv.org/abs/2111.08872) | 2022 | PyTorch library for geospatial |
| [segment-geospatial](https://arxiv.org/abs/2306.06081) | 2023 | SAM for geospatial data |
| [RandLA-Net](https://arxiv.org/abs/1911.11236) | 2020 | Efficient 3D point cloud segmentation |
| [SAHI](https://arxiv.org/abs/2202.06934) | 2022 | Slicing Aided Hyper Inference |
| [GEO-Bench](https://arxiv.org/abs/2306.03831) | 2023 | Foundation model geo benchmark |
