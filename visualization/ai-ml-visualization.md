# AI/ML Interpretation & Visualization

> SOTA tools for model explainability, neural network visualization, training monitoring, and AI-powered geospatial analysis dashboards.

**Last updated:** 2026-02-25

---

## Quick Picks

| Use Case | Recommendation | Why |
|----------|---------------|-----|
| **SOTA** | SHAP + folium/pydeck | Spatial feature importance maps -- see which features matter WHERE on the map |
| **Free Best** | TensorBoard | Training monitoring with geospatial plugins, image grids, confusion matrices |
| **Fastest Setup** | Weights & Biases | ML experiment tracking with built-in visualization, zero config |

---

## Table of Contents

1. [Model Explainability Visualization](#1-model-explainability-visualization)
2. [Neural Network Architecture Visualization](#2-neural-network-architecture-visualization)
3. [Training Monitoring Dashboards](#3-training-monitoring-dashboards)
4. [Uncertainty & Confidence Visualization](#4-uncertainty--confidence-visualization)
5. [Geospatial ML Result Visualization](#5-geospatial-ml-result-visualization)
6. [Interactive ML Dashboards](#6-interactive-ml-dashboards)
7. [AutoML & Hyperparameter Visualization](#7-automl--hyperparameter-visualization)
8. [Foundation Model Visualization](#8-foundation-model-visualization)
9. [Tool Comparison Matrix](#9-tool-comparison-matrix)
10. [Dark Arts Tips](#10-dark-arts-tips)

---

## 1. Model Explainability Visualization

Understanding **why** a model makes a prediction is just as important as the prediction itself -- especially in geospatial contexts where stakeholders need to know which environmental, demographic, or spectral features drive decisions across different regions.

### SHAP (SHapley Additive exPlanations)

- **Repository:** [slundberg/shap](https://github.com/slundberg/shap)
- **License:** MIT
- **Language:** Python
- **Install:** `pip install shap`

SHAP provides a unified framework for feature importance based on game-theoretic Shapley values. It is the gold standard for model-agnostic explainability.

**Core Plot Types:**

| Plot Type | Purpose | Geospatial Application |
|-----------|---------|----------------------|
| Summary plot | Global feature importance ranking | Which bands/features matter most across the study area |
| Dependence plot | Feature value vs. SHAP value relationship | How NDVI values affect predictions differently by region |
| Force plot | Single prediction explanation | Why was this specific parcel classified as urban? |
| Waterfall plot | Cumulative feature contribution | Step-by-step explanation for a zoning decision |
| Beeswarm plot | Distribution of SHAP values | Feature impact spread across all spatial units |

**Spatial SHAP: Mapping Feature Importance Geographically**

The real power for GIS practitioners is mapping SHAP values spatially. Instead of asking "which features matter?" you ask "which features matter WHERE?"

```python
import shap
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# --- Spatial SHAP for Land Cover Classification ---

# Load spatial data with features
gdf = gpd.read_file("training_parcels.geojson")

# Features: spectral bands, indices, terrain, proximity
feature_cols = [
    "ndvi_mean", "ndwi_mean", "ndbi_mean",
    "elevation", "slope", "aspect",
    "dist_to_road", "dist_to_water", "population_density",
    "b2_mean", "b3_mean", "b4_mean", "b8_mean"
]
X = gdf[feature_cols].values
y = gdf["land_cover_class"].values

# Train model
model = GradientBoostingClassifier(n_estimators=200, max_depth=6)
model.fit(X, y)

# Compute SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# --- Global summary plot ---
shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig("shap_summary_land_cover.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Spatial mapping of SHAP values ---
# Map the SHAP value for "ndvi_mean" across all parcels
# This shows WHERE NDVI matters most for the classification
ndvi_feature_idx = feature_cols.index("ndvi_mean")

# For multi-class, shap_values is a list; pick a target class
target_class = 0  # e.g., "Forest"
gdf["shap_ndvi_forest"] = shap_values[target_class][:, ndvi_feature_idx]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Left: raw NDVI values
gdf.plot(column="ndvi_mean", cmap="YlGn", legend=True, ax=axes[0])
axes[0].set_title("NDVI Mean Values")

# Right: SHAP values for NDVI (Forest class)
gdf.plot(column="shap_ndvi_forest", cmap="RdBu_r", legend=True, ax=axes[1])
axes[1].set_title("SHAP Value for NDVI (Forest Classification)")

plt.tight_layout()
plt.savefig("spatial_shap_ndvi.png", dpi=150, bbox_inches="tight")
```

**Integration with GeoPandas for Choropleth SHAP Maps:**

```python
import folium
from branca.colormap import LinearColormap

# Create an interactive map of SHAP values per spatial unit
m = folium.Map(location=[gdf.geometry.centroid.y.mean(),
                          gdf.geometry.centroid.x.mean()], zoom_start=10)

# Colormap for SHAP values
colormap = LinearColormap(
    colors=["blue", "white", "red"],
    vmin=gdf["shap_ndvi_forest"].min(),
    vmax=gdf["shap_ndvi_forest"].max(),
    caption="SHAP Value: NDVI contribution to Forest classification"
)

folium.GeoJson(
    gdf[["geometry", "shap_ndvi_forest"]].to_json(),
    style_function=lambda feature: {
        "fillColor": colormap(feature["properties"]["shap_ndvi_forest"]),
        "color": "black",
        "weight": 0.5,
        "fillOpacity": 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=["shap_ndvi_forest"],
                                   aliases=["SHAP (NDVI):"])
).add_to(m)

colormap.add_to(m)
m.save("shap_spatial_map.html")
```

### LIME (Local Interpretable Model-agnostic Explanations)

- **Repository:** [marcotcr/lime](https://github.com/marcotcr/lime)
- **License:** BSD-2-Clause
- **Install:** `pip install lime`

**Superpixel Explanations for Satellite Image Classification:**

LIME segments satellite images into superpixels and perturbs them to understand which regions drive the prediction. This is especially useful for CNN-based classifiers on remote sensing imagery.

```python
from lime import lime_image
from skimage.segmentation import quickshift
import numpy as np

# Assume `model` is a trained CNN for satellite image classification
# and `predict_fn` returns class probabilities for a batch of images

explainer = lime_image.LimeImageExplainer()

# Explain a single satellite image tile
explanation = explainer.explain_instance(
    satellite_tile,              # shape: (H, W, C)
    predict_fn,                  # callable returning (N, num_classes)
    top_labels=3,
    hide_color=0,
    num_samples=1000,
    segmentation_fn=lambda img: quickshift(img[:, :, :3], ratio=0.5)
)

# Get explanation mask for top predicted class
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=False
)
```

**Tabular LIME for Spatial Regression:**

```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_cols,
    class_names=["Low Risk", "Medium Risk", "High Risk"],
    mode="classification"
)

# Explain a single spatial unit prediction
explanation = explainer.explain_instance(
    X_test[idx],
    model.predict_proba,
    num_features=8
)
explanation.show_in_notebook()
```

### Captum (PyTorch Model Interpretability)

- **Repository:** [pytorch/captum](https://github.com/pytorch/captum)
- **License:** BSD-3-Clause
- **Install:** `pip install captum`

Captum provides gradient-based attribution methods tightly integrated with PyTorch, making it ideal for deep learning models applied to remote sensing.

**Key Methods for Geospatial CNNs:**

| Method | What It Shows | Best For |
|--------|--------------|----------|
| Integrated Gradients | Pixel-level attribution | Which pixels drove the classification |
| GradCAM | Class activation heatmap | Which spatial regions the model focuses on |
| DeepLIFT | Neuron contribution | Feature importance in spectral bands |
| Feature Ablation | Impact of removing features | Band importance for multispectral models |
| Layer Conductance | Layer-specific contributions | Understanding hierarchical feature extraction |

```python
from captum.attr import IntegratedGradients, GradCam
import torch

# For a remote sensing CNN
model.eval()
ig = IntegratedGradients(model)

# Compute attributions for a satellite image
input_tensor = torch.tensor(satellite_tile).unsqueeze(0).float()
input_tensor.requires_grad = True

attributions = ig.attribute(input_tensor, target=target_class, n_steps=200)

# Visualize attributions overlaid on the satellite image
from captum.attr import visualization as viz
viz.visualize_image_attr_multiple(
    attributions.squeeze().permute(1, 2, 0).detach().numpy(),
    satellite_tile.transpose(1, 2, 0),
    methods=["blended_heat_map", "original_image"],
    signs=["absolute_value", "absolute_value"],
    titles=["Attribution Heatmap", "Original Image"]
)
```

### ELI5: Permutation Importance

- **Repository:** [TeamHG-Memex/eli5](https://github.com/TeamHG-Memex/eli5)
- **Install:** `pip install eli5`

Quick permutation importance visualization -- shuffle a feature, measure the drop in performance. Simple, model-agnostic, and effective.

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(model, random_state=42).fit(X_test, y_test)
eli5.show_weights(perm, feature_names=feature_cols)
```

### Additional Explainability Libraries

| Library | Strengths | Geospatial Relevance |
|---------|-----------|---------------------|
| [Alibi](https://github.com/SeldonIO/alibi) | Counterfactual explanations, anchor explanations | "What would need to change for this parcel to be classified differently?" |
| [InterpretML](https://github.com/interpretml/interpret) | Glass-box models (EBM), unified API | Inherently interpretable models for spatial regression |
| [DiCE](https://github.com/interpretml/DiCE) | Diverse counterfactual explanations | Generate alternative scenarios for spatial predictions |
| [OmniXAI](https://github.com/salesforce/OmniXAI) | Unified explainability for vision, NLP, tabular | Multi-modal satellite + text explanations |

---

## 2. Neural Network Architecture Visualization

Visualizing the architecture itself -- layers, connections, parameter counts -- is essential for understanding model complexity and communicating designs to stakeholders.

### Model Architecture Diagrams

**Netron:**

- **Website:** [netron.app](https://netron.app)
- **Formats:** ONNX, TensorFlow, PyTorch, Keras, CoreML, and more
- **Usage:** Open any model file in the browser or desktop app; get an interactive, zoomable architecture diagram

```bash
# Install CLI
pip install netron

# Visualize a model
netron model.onnx
```

**TorchViz (PyTorch):**

```python
from torchviz import make_dot
import torch

# Forward pass to build computation graph
x = torch.randn(1, 13, 256, 256)  # 13-band satellite image
output = model(x)

# Generate architecture diagram
dot = make_dot(output, params=dict(model.named_parameters()),
               show_attrs=True, show_saved=True)
dot.render("model_architecture", format="png")
```

**Keras plot_model:**

```python
from tensorflow.keras.utils import plot_model

plot_model(
    model,
    to_file="model_architecture.png",
    show_shapes=True,
    show_layer_names=True,
    show_layer_activations=True,
    expand_nested=True,
    dpi=150
)
```

### Feature Maps / Activation Visualization

Visualizing intermediate layer outputs reveals what the network "sees" at different depths. For satellite imagery CNNs, early layers detect edges and spectral contrasts; deeper layers detect spatial patterns like road networks or building footprints.

```python
import torch
import matplotlib.pyplot as plt

# Hook to capture activations
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks on layers of interest
model.encoder.layer1.register_forward_hook(hook_fn("layer1"))
model.encoder.layer2.register_forward_hook(hook_fn("layer2"))
model.encoder.layer3.register_forward_hook(hook_fn("layer3"))

# Forward pass
with torch.no_grad():
    output = model(input_tensor)

# Plot activation maps for a layer
def plot_activations(act, layer_name, num_maps=16):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < min(num_maps, act.shape[1]):
            ax.imshow(act[0, i].cpu().numpy(), cmap="viridis")
            ax.set_title(f"Filter {i}")
        ax.axis("off")
    fig.suptitle(f"Activation Maps: {layer_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"activations_{layer_name}.png", dpi=150)

plot_activations(activations["layer1"], "Layer 1 (low-level features)")
plot_activations(activations["layer3"], "Layer 3 (high-level features)")
```

### Attention Maps for Geospatial Transformers

Vision Transformers (ViT) and geospatial foundation models (Prithvi-EO, SatMAE, Clay) use self-attention. Visualizing attention maps shows which image patches the model considers most relevant.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention(model, input_tensor, layer_idx=-1, head_idx=0):
    """
    Extract and visualize attention from a Vision Transformer.
    Works with ViT, Prithvi-EO, SatMAE-style architectures.
    """
    # Hook to capture attention weights
    attention_weights = []

    def attention_hook(module, input, output):
        # output is (attn_output, attn_weights) for many ViT implementations
        if isinstance(output, tuple) and len(output) > 1:
            attention_weights.append(output[1].detach())

    # Register hook on the target attention layer
    handle = model.blocks[layer_idx].attn.register_forward_hook(attention_hook)

    with torch.no_grad():
        _ = model(input_tensor)

    handle.remove()

    if not attention_weights:
        raise ValueError("No attention weights captured. Check model architecture.")

    # Shape: (batch, num_heads, seq_len, seq_len)
    attn = attention_weights[0][0, head_idx]  # first sample, selected head

    # Remove CLS token attention, reshape to spatial grid
    # For a 224x224 image with patch_size=16: 14x14 = 196 patches
    patch_size = 16
    num_patches_side = int(np.sqrt(attn.shape[-1] - 1))

    # Attention from CLS token to all patches
    cls_attention = attn[0, 1:]  # CLS attends to all patches
    cls_attention = cls_attention.reshape(num_patches_side, num_patches_side)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(input_tensor[0, :3].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input Image (RGB)")
    im = axes[1].imshow(cls_attention.cpu().numpy(), cmap="hot",
                         interpolation="bilinear")
    axes[1].set_title(f"Attention Map (Layer {layer_idx}, Head {head_idx})")
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig("attention_map.png", dpi=150)
```

### GradCAM / Grad-CAM++ for Satellite Imagery

Class Activation Maps highlight which spatial regions contribute most to a specific class prediction. Overlaying these on satellite imagery produces intuitive visual explanations.

```python
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np

# Target layer: last convolutional layer
target_layers = [model.backbone.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)

# Generate CAM for target class
grayscale_cam = cam(input_tensor=input_tensor, targets=None)
grayscale_cam = grayscale_cam[0, :]

# Normalize RGB for overlay
rgb_img = satellite_tile[:3].transpose(1, 2, 0)
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

# Overlay CAM on satellite image
visualization = show_cam_on_image(rgb_img.astype(np.float32),
                                   grayscale_cam, use_rgb=True)
```

### Semantic Segmentation Output Visualization

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Class definitions for land cover segmentation
class_names = ["Water", "Forest", "Urban", "Agriculture", "Barren"]
class_colors = ["#0077be", "#228b22", "#dc143c", "#ffd700", "#d2b48c"]
cmap = plt.cm.colors.ListedColormap(class_colors)

def plot_segmentation_results(image, prediction, ground_truth, confidence):
    """Plot segmentation prediction alongside ground truth and confidence."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # Original image
    axes[0].imshow(image[:3].transpose(1, 2, 0))
    axes[0].set_title("Satellite Image (RGB)")

    # Predicted segmentation
    axes[1].imshow(prediction, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[1].set_title("Predicted Segmentation")

    # Ground truth
    axes[2].imshow(ground_truth, cmap=cmap, vmin=0, vmax=len(class_names)-1)
    axes[2].set_title("Ground Truth")

    # Confidence map (max softmax probability)
    im = axes[3].imshow(confidence, cmap="RdYlGn", vmin=0.5, vmax=1.0)
    axes[3].set_title("Prediction Confidence")
    plt.colorbar(im, ax=axes[3])

    # Legend
    patches = [mpatches.Patch(color=c, label=n)
               for c, n in zip(class_colors, class_names)]
    fig.legend(handles=patches, loc="lower center", ncol=len(class_names))

    plt.tight_layout()
    plt.savefig("segmentation_results.png", dpi=150, bbox_inches="tight")
```

> **Cross-reference:** For geospatial AI/ML tools and frameworks, see [AI/ML Geospatial Tools](../tools/ai-ml-geospatial.md). For training data sources, see [ML Training Data](../data-sources/ml-training-data.md).

---

## 3. Training Monitoring Dashboards

Tracking training progress is non-negotiable for reproducible geospatial ML. These tools visualize loss curves, learning rate schedules, predictions over time, and enable experiment comparison.

### TensorBoard

- **Website:** [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard)
- **License:** Apache 2.0
- **Install:** `pip install tensorboard`

TensorBoard is the de facto standard for training visualization. It supports scalars, images, histograms, embeddings, and custom plugins.

**Key Features for Geospatial ML:**

| Feature | How It Helps |
|---------|-------------|
| Scalar logging | Loss curves, accuracy, IoU, F1 over epochs |
| Image logging | Log prediction maps every N epochs -- watch the model learn |
| Histogram | Weight distributions, gradient flow analysis |
| PR Curves | Precision-recall per class for imbalanced land cover datasets |
| Confusion matrix | Class-level performance breakdown |
| Hyperparameter tuning | Compare runs with different configurations |
| Profiler | GPU utilization, data pipeline bottleneck detection |

**TorchGeo + TensorBoard for Land Cover Training:**

```python
import torch
from torch.utils.tensorboard import SummaryWriter
from torchgeo.datasets import EuroSAT
from torchgeo.datamodules import EuroSATDataModule
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

writer = SummaryWriter("runs/land_cover_experiment_001")

# Log training configuration
writer.add_text("config", """
- Model: ResNet50-UNet
- Dataset: EuroSAT
- Learning Rate: 1e-4
- Batch Size: 32
- Augmentations: RandomFlip, RandomRotation, ColorJitter
""")

def log_training_step(writer, epoch, step, loss, lr, metrics):
    """Log standard training metrics."""
    global_step = epoch * steps_per_epoch + step
    writer.add_scalar("train/loss", loss, global_step)
    writer.add_scalar("train/learning_rate", lr, global_step)
    for name, value in metrics.items():
        writer.add_scalar(f"train/{name}", value, global_step)

def log_validation_results(writer, epoch, val_metrics, predictions, targets, images):
    """Log validation metrics and prediction visualizations."""
    # Scalar metrics
    for name, value in val_metrics.items():
        writer.add_scalar(f"val/{name}", value, epoch)

    # Log prediction vs ground truth images
    fig = create_prediction_grid(images[:4], predictions[:4], targets[:4])
    writer.add_figure("val/predictions", fig, epoch)
    plt.close(fig)

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(targets.flatten(), predictions.flatten())
    fig_cm, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax)
    writer.add_figure("val/confusion_matrix", fig_cm, epoch)
    plt.close(fig_cm)

def log_geospatial_predictions(writer, epoch, model, test_tiles, tile_coords):
    """Log predicted maps as a geospatial plugin visualization."""
    import folium
    from folium.raster_layers import ImageOverlay

    m = folium.Map(location=[tile_coords[0][0], tile_coords[0][1]],
                   zoom_start=12)

    for tile, (lat, lon, bounds) in zip(test_tiles, tile_coords):
        with torch.no_grad():
            pred = model(tile.unsqueeze(0)).argmax(dim=1)[0]

        # Convert prediction to colored image
        pred_colored = class_colors_array[pred.cpu().numpy()]

        ImageOverlay(
            image=pred_colored,
            bounds=bounds,
            opacity=0.6,
            name=f"Epoch {epoch} Prediction"
        ).add_to(m)

    # Save as HTML artifact
    m.save(f"runs/predictions_epoch_{epoch:03d}.html")

# Training loop integration
for epoch in range(num_epochs):
    model.train()
    for step, batch in enumerate(train_loader):
        loss = train_step(batch)
        log_training_step(writer, epoch, step, loss.item(),
                         optimizer.param_groups[0]["lr"], {})

    # Validation
    val_metrics, preds, targets, imgs = validate(model, val_loader)
    log_validation_results(writer, epoch, val_metrics, preds, targets, imgs)

    # Log geospatial predictions every 5 epochs
    if epoch % 5 == 0:
        log_geospatial_predictions(writer, epoch, model,
                                    test_tiles, tile_coords)

writer.close()
```

### Weights & Biases (W&B)

- **Website:** [wandb.ai](https://wandb.ai)
- **License:** Proprietary (free tier available)
- **Install:** `pip install wandb`

W&B provides experiment tracking, hyperparameter sweeps, artifact versioning, and team collaboration. Its custom panel system is exceptionally flexible.

**Key Geospatial Features:**

```python
import wandb
import numpy as np

wandb.init(
    project="land-cover-classification",
    config={
        "model": "UNet-ResNet50",
        "dataset": "EuroSAT",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 100,
        "bands": ["B2", "B3", "B4", "B8"],
        "spatial_resolution": "10m",
        "study_area": "Central Europe",
    }
)

# Log GeoJSON predictions as W&B artifacts
def log_geojson_predictions(epoch, gdf_predictions):
    """Log geospatial predictions as a versioned artifact."""
    artifact = wandb.Artifact(
        f"predictions-epoch-{epoch}",
        type="predictions",
        metadata={"epoch": epoch, "crs": "EPSG:4326"}
    )

    geojson_path = f"predictions_epoch_{epoch}.geojson"
    gdf_predictions.to_file(geojson_path, driver="GeoJSON")
    artifact.add_file(geojson_path)
    wandb.log_artifact(artifact)

# Log prediction images with bounding boxes
def log_detection_results(images, predictions, ground_truths):
    """Log object detection results with bounding boxes."""
    wandb_images = []
    for img, pred, gt in zip(images, predictions, ground_truths):
        boxes_pred = [
            {"position": {"minX": b[0], "minY": b[1],
                          "maxX": b[2], "maxY": b[3]},
             "class_id": int(b[4]),
             "scores": {"confidence": float(b[5])},
             "box_caption": f"{class_names[int(b[4])]} {b[5]:.2f}"}
            for b in pred
        ]
        boxes_gt = [
            {"position": {"minX": b[0], "minY": b[1],
                          "maxX": b[2], "maxY": b[3]},
             "class_id": int(b[4]),
             "box_caption": class_names[int(b[4])]}
            for b in gt
        ]

        wandb_images.append(wandb.Image(
            img,
            boxes={
                "predictions": {"box_data": boxes_pred,
                                "class_labels": dict(enumerate(class_names))},
                "ground_truth": {"box_data": boxes_gt,
                                 "class_labels": dict(enumerate(class_names))}
            }
        ))

    wandb.log({"detection_results": wandb_images})

# Log training metrics
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_metrics = validate(model, val_loader)

    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "val/accuracy": val_metrics["accuracy"],
        "val/mean_iou": val_metrics["mean_iou"],
        "val/f1_macro": val_metrics["f1_macro"],
    })

wandb.finish()
```

### MLflow

- **Website:** [mlflow.org](https://mlflow.org)
- **License:** Apache 2.0
- **Install:** `pip install mlflow`

MLflow focuses on the full ML lifecycle: experiment tracking, model registry, and deployment. It integrates well with on-premise infrastructure -- important for organizations that cannot use cloud-based tracking.

```python
import mlflow
import mlflow.pytorch

mlflow.set_experiment("geospatial-land-cover")

with mlflow.start_run(run_name="unet_resnet50_v3"):
    # Log parameters
    mlflow.log_params({
        "model": "UNet-ResNet50",
        "lr": 1e-4,
        "epochs": 100,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
    })

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader)
        val_metrics = validate(model, val_loader)

        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_iou": val_metrics["mean_iou"],
        }, step=epoch)

    # Log model artifact
    mlflow.pytorch.log_model(model, "model")

    # Log prediction visualizations
    mlflow.log_artifact("segmentation_results.png")
    mlflow.log_artifact("confusion_matrix.png")
```

### Neptune.ai

- **Website:** [neptune.ai](https://neptune.ai)
- **License:** Proprietary (free tier)
- **Install:** `pip install neptune`

Rich media logging with native support for images, interactive charts, and large-scale experiment comparison.

### ClearML

- **Website:** [clear.ml](https://clear.ml)
- **License:** Apache 2.0
- **Install:** `pip install clearml`

End-to-end ML pipeline monitoring with built-in data versioning, experiment management, and orchestration. Supports on-premise deployment.

**Comparison of Training Monitoring Tools:**

| Feature | TensorBoard | W&B | MLflow | Neptune | ClearML |
|---------|------------|-----|--------|---------|---------|
| Cost | Free | Free tier + paid | Free (OSS) | Free tier + paid | Free (OSS) + paid |
| Self-hosted | Yes | Enterprise only | Yes | No | Yes |
| Image logging | Yes | Yes (rich) | Yes (artifacts) | Yes | Yes |
| Hyperparameter sweeps | Limited | Built-in | No (use Optuna) | No | Built-in |
| Model registry | No | Yes | Yes | Yes | Yes |
| Team collaboration | Limited | Excellent | Good | Good | Good |
| GeoJSON support | Plugin needed | Custom panels | Artifacts | Custom | Artifacts |
| Setup effort | Minimal | Minimal | Moderate | Minimal | Moderate |

---

## 4. Uncertainty & Confidence Visualization

In geospatial ML, knowing WHERE the model is uncertain is as valuable as the predictions themselves. Uncertainty maps guide field validation, active learning sampling, and stakeholder trust.

### Prediction Confidence Maps

Convert softmax probabilities into spatial heatmaps to see where the model is confident versus uncertain.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds

def generate_confidence_map(model, image_tensor, output_path):
    """
    Generate a prediction confidence map from a segmentation model.
    Confidence = max softmax probability per pixel.
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0))  # (1, C, H, W)
        probabilities = torch.softmax(logits, dim=1)  # (1, C, H, W)

        # Max probability per pixel = confidence
        confidence = probabilities.max(dim=1)[0][0]  # (H, W)

        # Predicted class
        prediction = probabilities.argmax(dim=1)[0]  # (H, W)

    confidence_np = confidence.cpu().numpy()
    prediction_np = prediction.cpu().numpy()

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image_tensor[:3].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input Image")

    axes[1].imshow(prediction_np, cmap="tab10")
    axes[1].set_title("Predicted Classes")

    im = axes[2].imshow(confidence_np, cmap="RdYlGn", vmin=0.3, vmax=1.0)
    axes[2].set_title("Prediction Confidence")
    plt.colorbar(im, ax=axes[2], label="Max Probability")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")

    return confidence_np, prediction_np
```

### Monte Carlo Dropout: Uncertainty Estimation

Monte Carlo (MC) Dropout runs multiple forward passes with dropout enabled at inference time. The variance across predictions estimates epistemic uncertainty -- the model's uncertainty due to limited training data.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def enable_dropout(model):
    """Enable dropout layers during inference for MC Dropout."""
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()

def mc_dropout_uncertainty(model, image_tensor, n_forward=50):
    """
    Estimate prediction uncertainty via Monte Carlo Dropout.

    Returns:
        mean_prediction: averaged class probabilities (H, W, C)
        uncertainty: entropy-based uncertainty map (H, W)
        prediction: final predicted class map (H, W)
    """
    model.eval()
    enable_dropout(model)

    predictions = []
    for _ in range(n_forward):
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)
            predictions.append(probs.cpu().numpy())

    # Stack: (n_forward, 1, C, H, W) -> (n_forward, C, H, W)
    predictions = np.concatenate(predictions, axis=0)

    # Mean prediction
    mean_pred = predictions.mean(axis=0)[0]  # (C, H, W)

    # Predictive entropy as uncertainty measure
    # H = -sum(p * log(p))
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=0)  # (H, W)

    # Mutual information (epistemic uncertainty)
    # MI = H[E[p]] - E[H[p]]
    individual_entropy = -np.sum(
        predictions * np.log(predictions + 1e-10), axis=2
    )  # (n_forward, H, W)
    expected_entropy = individual_entropy.mean(axis=0)[0]  # (H, W)
    mutual_info = entropy - expected_entropy  # (H, W)

    final_class = mean_pred.argmax(axis=0)  # (H, W)

    return mean_pred, entropy, mutual_info, final_class

# --- Visualize uncertainty geographically ---
def plot_uncertainty_map(image, prediction, entropy, mutual_info, bounds, crs):
    """Plot uncertainty maps side by side."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    axes[0].imshow(image[:3].transpose(1, 2, 0))
    axes[0].set_title("Input Image (RGB)")

    axes[1].imshow(prediction, cmap="tab10")
    axes[1].set_title("MC Dropout Mean Prediction")

    im2 = axes[2].imshow(entropy, cmap="magma")
    axes[2].set_title("Predictive Entropy (Total Uncertainty)")
    plt.colorbar(im2, ax=axes[2])

    im3 = axes[3].imshow(mutual_info, cmap="inferno")
    axes[3].set_title("Mutual Information (Epistemic Uncertainty)")
    plt.colorbar(im3, ax=axes[3])

    plt.suptitle("Monte Carlo Dropout Uncertainty Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig("mc_dropout_uncertainty.png", dpi=150, bbox_inches="tight")
```

### Ensemble Disagreement Maps

When you have multiple trained models (different seeds, architectures, or hyperparameters), their disagreement reveals regions of high uncertainty.

```python
import numpy as np
import matplotlib.pyplot as plt

def ensemble_disagreement(models, image_tensor):
    """
    Compute ensemble disagreement from multiple models.
    High disagreement = models predict different classes = uncertain region.
    """
    all_predictions = []
    all_probabilities = []

    for model in models:
        model.eval()
        with torch.no_grad():
            logits = model(image_tensor.unsqueeze(0))
            probs = torch.softmax(logits, dim=1)[0]  # (C, H, W)
            pred = probs.argmax(dim=0)                # (H, W)

            all_predictions.append(pred.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())

    predictions = np.stack(all_predictions)      # (n_models, H, W)
    probabilities = np.stack(all_probabilities)  # (n_models, C, H, W)

    # --- Disagreement metrics ---

    # 1. Vote entropy: how spread are the class votes?
    n_models = len(models)
    n_classes = probabilities.shape[1]
    vote_counts = np.zeros((n_classes, *predictions.shape[1:]))
    for c in range(n_classes):
        vote_counts[c] = (predictions == c).sum(axis=0)
    vote_fractions = vote_counts / n_models
    vote_entropy = -np.sum(
        vote_fractions * np.log(vote_fractions + 1e-10), axis=0
    )

    # 2. Probability variance: average variance across classes
    prob_variance = probabilities.var(axis=0).mean(axis=0)  # (H, W)

    # 3. Agreement ratio: fraction of models agreeing with majority
    from scipy import stats
    majority_vote = stats.mode(predictions, axis=0)[0][0]
    agreement = (predictions == majority_vote).mean(axis=0)

    return {
        "vote_entropy": vote_entropy,
        "prob_variance": prob_variance,
        "agreement_ratio": agreement,
        "majority_vote": majority_vote,
        "individual_predictions": predictions
    }
```

### Calibration Plots for Spatial Predictions

A well-calibrated model should have predicted probabilities that match actual frequencies. Calibration plots (reliability diagrams) reveal whether your model is overconfident or underconfident.

```python
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

def plot_calibration_per_region(y_true_by_region, y_prob_by_region, region_names):
    """
    Plot calibration curves for different geographic regions.
    Reveals if model confidence is well-calibrated across space.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for y_true, y_prob, name in zip(y_true_by_region,
                                     y_prob_by_region,
                                     region_names):
        fraction_positive, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )
        ax.plot(mean_predicted, fraction_positive, "s-", label=name)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves by Geographic Region")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("calibration_by_region.png", dpi=150)
```

### Error Spatial Distribution

Map where the model makes errors to identify geographic patterns in failure -- does it struggle in mountainous regions? At land cover boundaries? In urban-rural transition zones?

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

def plot_error_heatmap(gdf, y_true_col, y_pred_col, geometry_col="geometry"):
    """
    Create a geographic error heatmap showing where the model fails.
    """
    gdf = gdf.copy()
    gdf["correct"] = (gdf[y_true_col] == gdf[y_pred_col]).astype(int)
    gdf["error"] = 1 - gdf["correct"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Error map (binary: correct vs incorrect)
    gdf.plot(column="error", cmap="RdYlGn_r", legend=True, ax=axes[0],
             legend_kwds={"label": "Error (0=Correct, 1=Error)"})
    axes[0].set_title("Prediction Error Map")

    # Error rate by grid cell (spatial aggregation)
    # Create a grid and compute error rate per cell
    from shapely.geometry import box
    bounds = gdf.total_bounds
    cell_size = 0.01  # degrees
    grid_cells = []
    for x in np.arange(bounds[0], bounds[2], cell_size):
        for y in np.arange(bounds[1], bounds[3], cell_size):
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))

    grid = gpd.GeoDataFrame(geometry=grid_cells, crs=gdf.crs)
    joined = gpd.sjoin(gdf, grid, how="right", predicate="within")
    error_rate = joined.groupby(joined.index)["error"].mean()
    grid["error_rate"] = error_rate

    grid.dropna(subset=["error_rate"]).plot(
        column="error_rate", cmap="Reds", legend=True, ax=axes[1],
        legend_kwds={"label": "Error Rate"}
    )
    axes[1].set_title("Error Rate by Grid Cell")

    # Confusion by class and location
    from sklearn.metrics import classification_report
    report = classification_report(gdf[y_true_col], gdf[y_pred_col],
                                    output_dict=True)
    classes = [k for k in report if k not in
               ["accuracy", "macro avg", "weighted avg"]]
    f1_scores = [report[c]["f1-score"] for c in classes]

    axes[2].barh(classes, f1_scores, color="steelblue")
    axes[2].set_xlabel("F1 Score")
    axes[2].set_title("Per-Class Performance")
    axes[2].set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("error_spatial_distribution.png", dpi=150, bbox_inches="tight")
```

---

## 5. Geospatial ML Result Visualization

This section covers how to visualize the outputs of common geospatial ML tasks: classification, object detection, change detection, anomaly detection, and clustering.

### Classification Result Maps

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

def plot_classification_with_confidence(
    image_path, prediction_array, confidence_array,
    class_names, class_colors, output_path
):
    """
    Plot classification results with confidence overlay on satellite imagery.
    """
    with rasterio.open(image_path) as src:
        rgb = np.stack([src.read(4), src.read(3), src.read(2)], axis=-1)
        rgb = np.clip(rgb / 3000, 0, 1)  # Scale for display
        transform = src.transform
        crs = src.crs

    cmap = ListedColormap(class_colors)
    norm = BoundaryNorm(range(len(class_names) + 1), cmap.N)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Satellite Image (RGB)")

    # Classification
    im1 = axes[0, 1].imshow(prediction_array, cmap=cmap, norm=norm)
    axes[0, 1].set_title("Land Cover Classification")

    # Confidence
    im2 = axes[1, 0].imshow(confidence_array, cmap="RdYlGn", vmin=0.4, vmax=1.0)
    axes[1, 0].set_title("Prediction Confidence")
    plt.colorbar(im2, ax=axes[1, 0])

    # Classification masked by low confidence
    threshold = 0.7
    masked_pred = np.where(confidence_array >= threshold,
                           prediction_array, -1)
    cmap_masked = ListedColormap(["lightgray"] + class_colors)
    norm_masked = BoundaryNorm(range(-1, len(class_names) + 1), cmap_masked.N)
    axes[1, 1].imshow(masked_pred, cmap=cmap_masked, norm=norm_masked)
    axes[1, 1].set_title(f"High-Confidence Predictions (>= {threshold})")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
```

### Object Detection on Satellite Imagery

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_detections(image, detections, class_names, class_colors,
                    score_threshold=0.5):
    """
    Plot object detection results (bounding boxes + instance masks)
    on satellite imagery.

    detections: list of dicts with keys:
        bbox (x1, y1, x2, y2), class_id, score, mask (optional)
    """
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)

    for det in detections:
        if det["score"] < score_threshold:
            continue

        x1, y1, x2, y2 = det["bbox"]
        cls_id = det["class_id"]
        score = det["score"]
        color = class_colors[cls_id]

        # Bounding box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)

        # Label
        ax.text(x1, y1 - 5,
                f"{class_names[cls_id]}: {score:.2f}",
                color="white", fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                         facecolor=color, alpha=0.8))

        # Instance mask overlay (if available)
        if "mask" in det and det["mask"] is not None:
            mask = det["mask"]
            colored_mask = np.zeros((*mask.shape, 4))
            rgb = plt.cm.colors.to_rgba(color)
            colored_mask[mask > 0.5] = (*rgb[:3], 0.4)
            ax.imshow(colored_mask)

    ax.set_title(f"Object Detection Results (threshold={score_threshold})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig("detection_results.png", dpi=150, bbox_inches="tight")
```

### Change Detection Results

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_change_detection(image_before, image_after, change_prob,
                           change_binary, output_path):
    """
    Visualize change detection results: before/after + change probability + binary change.
    """
    fig = plt.figure(figsize=(20, 10))
    gs = GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

    # Before
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax1.imshow(image_before)
    ax1.set_title("Before (T1)")

    # After
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.imshow(image_after)
    ax2.set_title("After (T2)")

    # Change probability map
    ax3 = fig.add_subplot(gs[1, 0:2])
    im = ax3.imshow(change_prob, cmap="hot", vmin=0, vmax=1)
    ax3.set_title("Change Probability")
    plt.colorbar(im, ax=ax3, label="P(change)")

    # Binary change map
    ax4 = fig.add_subplot(gs[1, 2:4])
    change_overlay = np.stack([
        change_binary.astype(float),
        np.zeros_like(change_binary, dtype=float),
        np.zeros_like(change_binary, dtype=float),
        change_binary.astype(float) * 0.6
    ], axis=-1)
    ax4.imshow(image_after)
    ax4.imshow(change_overlay)
    ax4.set_title("Detected Changes (overlay)")

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
```

### Anomaly Detection: Spatial Outlier Visualization

```python
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

def spatial_anomaly_detection(gdf, feature_cols, contamination=0.05):
    """
    Run Isolation Forest for spatial anomaly detection and map results.
    """
    X = gdf[feature_cols].values

    iso_forest = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=200
    )

    gdf = gdf.copy()
    gdf["anomaly_score"] = iso_forest.decision_function(X)
    gdf["is_anomaly"] = iso_forest.predict(X)  # -1 = anomaly, 1 = normal

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Anomaly score map (continuous)
    gdf.plot(column="anomaly_score", cmap="RdBu", legend=True, ax=axes[0],
             legend_kwds={"label": "Anomaly Score (lower = more anomalous)"})
    axes[0].set_title("Anomaly Score Map")

    # Binary anomaly map
    colors = {1: "steelblue", -1: "red"}
    gdf.plot(color=gdf["is_anomaly"].map(colors), ax=axes[1])
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue",
               markersize=10, label="Normal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=10, label="Anomaly"),
    ]
    axes[1].legend(handles=legend_elements)
    axes[1].set_title(f"Detected Anomalies ({contamination*100:.0f}% contamination)")

    plt.tight_layout()
    plt.savefig("anomaly_detection_map.png", dpi=150, bbox_inches="tight")

    return gdf
```

### Clustering Results on Map

```python
import hdbscan
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def plot_clustering_results(gdf, feature_cols, min_cluster_size=50):
    """
    HDBSCAN clustering with spatial visualization and quality metrics.
    """
    X = gdf[feature_cols].values

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric="euclidean"
    )
    labels = clusterer.fit_predict(X)

    gdf = gdf.copy()
    gdf["cluster"] = labels
    gdf["cluster_prob"] = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    # Silhouette score (excluding noise)
    mask = labels != -1
    if mask.sum() > 0 and n_clusters > 1:
        sil_score = silhouette_score(X[mask], labels[mask])
    else:
        sil_score = 0.0

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # Cluster map
    gdf[gdf["cluster"] != -1].plot(
        column="cluster", cmap="tab20", categorical=True,
        legend=True, ax=axes[0], markersize=5
    )
    gdf[gdf["cluster"] == -1].plot(
        color="lightgray", ax=axes[0], markersize=2, alpha=0.5
    )
    axes[0].set_title(f"HDBSCAN Clusters (n={n_clusters}, noise={n_noise})")

    # Cluster membership probability
    gdf.plot(column="cluster_prob", cmap="YlOrRd", legend=True, ax=axes[1],
             legend_kwds={"label": "Membership Probability"})
    axes[1].set_title("Cluster Membership Probability")

    # Cluster size distribution
    cluster_sizes = gdf[gdf["cluster"] != -1]["cluster"].value_counts().sort_index()
    axes[2].bar(cluster_sizes.index.astype(str), cluster_sizes.values,
                color="steelblue")
    axes[2].set_xlabel("Cluster ID")
    axes[2].set_ylabel("Size")
    axes[2].set_title(f"Cluster Sizes (Silhouette: {sil_score:.3f})")

    plt.tight_layout()
    plt.savefig("clustering_results.png", dpi=150, bbox_inches="tight")

    return gdf
```

> **SOTA Tools:** [segment-geospatial](https://github.com/opengeos/segment-geospatial) (SAM for geospatial), [TorchGeo](https://github.com/microsoft/torchgeo), [Raster Vision](https://github.com/azavea/raster-vision)
>
> **Cross-reference:** [AI/ML Geospatial Tools](../tools/ai-ml-geospatial.md)

---

## 6. Interactive ML Dashboards

Move beyond static plots. Interactive dashboards let stakeholders explore model predictions, adjust thresholds, and compare model outputs spatially.

### Streamlit for ML Inference Dashboards

- **Website:** [streamlit.io](https://streamlit.io)
- **License:** Apache 2.0
- **Install:** `pip install streamlit`

```python
# app.py -- Streamlit ML Inference Dashboard
import streamlit as st
import torch
import numpy as np
import folium
from streamlit_folium import st_folium
import rasterio
from PIL import Image

st.set_page_config(page_title="GeoAI Inference Dashboard", layout="wide")
st.title("Land Cover Classification Dashboard")

# --- Sidebar: Model Selection ---
st.sidebar.header("Model Configuration")
model_name = st.sidebar.selectbox(
    "Select Model",
    ["UNet-ResNet50", "DeepLabV3-ResNet101", "SegFormer-B3"]
)
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.0, 1.0, 0.7, 0.05
)
compare_mode = st.sidebar.checkbox("Compare Two Models")

# --- Main Area ---
uploaded_file = st.file_uploader(
    "Upload Satellite Image (GeoTIFF)", type=["tif", "tiff"]
)

if uploaded_file is not None:
    # Load and display image
    with rasterio.open(uploaded_file) as src:
        image = src.read()
        bounds = src.bounds
        crs = src.crs

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Image")
        rgb = np.stack([image[3], image[2], image[1]], axis=-1)
        rgb = np.clip(rgb / 3000, 0, 1)
        st.image(rgb, caption="RGB Composite")

    # Run inference
    with st.spinner("Running inference..."):
        prediction, confidence = run_inference(model_name, image)

    with col2:
        st.subheader("Prediction")
        # Apply confidence threshold
        masked = np.where(confidence >= confidence_threshold,
                          prediction, -1)
        st.image(colorize_prediction(masked),
                 caption=f"Classification (confidence >= {confidence_threshold})")

    # Interactive map output
    st.subheader("Results on Map")
    m = folium.Map(
        location=[(bounds.bottom + bounds.top) / 2,
                   (bounds.left + bounds.right) / 2],
        zoom_start=13
    )

    folium.raster_layers.ImageOverlay(
        image=colorize_prediction(prediction),
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.6,
        name="Prediction"
    ).add_to(m)

    folium.LayerControl().add_to(m)
    st_folium(m, width=800, height=500)

    # Metrics
    st.subheader("Prediction Statistics")
    unique, counts = np.unique(prediction, return_counts=True)
    for cls_id, count in zip(unique, counts):
        pct = count / prediction.size * 100
        st.metric(class_names[cls_id], f"{pct:.1f}%")

    # --- Compare mode ---
    if compare_mode:
        st.subheader("Model Comparison")
        model_b = st.selectbox("Second Model",
                               [m for m in ["UNet-ResNet50",
                                "DeepLabV3-ResNet101", "SegFormer-B3"]
                                if m != model_name])
        pred_b, conf_b = run_inference(model_b, image)

        col_a, col_b, col_diff = st.columns(3)
        with col_a:
            st.image(colorize_prediction(prediction),
                     caption=model_name)
        with col_b:
            st.image(colorize_prediction(pred_b),
                     caption=model_b)
        with col_diff:
            disagreement = (prediction != pred_b).astype(float)
            st.image(disagreement, caption="Disagreement Map",
                     clamp=True)
```

### Gradio: Quick ML Demo Interfaces

- **Website:** [gradio.app](https://gradio.app)
- **License:** Apache 2.0
- **Install:** `pip install gradio`

```python
import gradio as gr
import numpy as np

def classify_satellite_image(image, model_choice, threshold):
    """Run classification and return results."""
    prediction, confidence = run_inference(model_choice, image)
    colored = colorize_prediction(prediction)
    conf_map = confidence_to_heatmap(confidence)
    stats = compute_class_stats(prediction)
    return colored, conf_map, stats

demo = gr.Interface(
    fn=classify_satellite_image,
    inputs=[
        gr.Image(label="Satellite Image"),
        gr.Dropdown(["UNet", "DeepLabV3", "SegFormer"], label="Model"),
        gr.Slider(0.0, 1.0, 0.7, label="Confidence Threshold"),
    ],
    outputs=[
        gr.Image(label="Classification Result"),
        gr.Image(label="Confidence Map"),
        gr.JSON(label="Class Statistics"),
    ],
    title="Satellite Image Classification",
    description="Upload a satellite image to classify land cover types.",
)

demo.launch(share=True)
```

### Panel + HoloViews: Linked ML and Spatial Views

- **Website:** [panel.holoviz.org](https://panel.holoviz.org)
- **Install:** `pip install panel holoviews geoviews`

Panel with HoloViews enables linked brushing between ML metric plots and spatial views. Select a region on the map, and the accuracy chart updates. Click a point on the scatter plot, and the map highlights it.

```python
import panel as pn
import holoviews as hv
import geoviews as gv
import geopandas as gpd

pn.extension()
hv.extension("bokeh")

gdf = gpd.read_file("predictions.geojson")

# Linked spatial + metric views
points = gv.Points(gdf, kdims=["longitude", "latitude"],
                   vdims=["prediction", "confidence", "error"])

# Map view with tile background
map_view = gv.tile_sources.CartoDark() * points.opts(
    color="confidence", cmap="RdYlGn", size=5,
    tools=["hover", "lasso_select", "box_select"],
    width=600, height=400
)

# Linked histogram of confidence
confidence_hist = hv.operation.histogram(
    points, dimension="confidence", num_bins=30
).opts(width=400, height=300, title="Confidence Distribution")

# Error scatter
error_scatter = hv.Scatter(
    gdf, kdims=["confidence"], vdims=["error"]
).opts(width=400, height=300, title="Confidence vs Error",
       tools=["hover"])

dashboard = pn.Column(
    pn.pane.Markdown("# ML Results Explorer"),
    pn.Row(map_view, pn.Column(confidence_hist, error_scatter))
)

dashboard.servable()
```

### Evidently AI: ML Monitoring Over Geography

- **Website:** [evidentlyai.com](https://evidentlyai.com)
- **License:** Apache 2.0
- **Install:** `pip install evidently`

Evidently detects data drift, concept drift, and model performance degradation. For geospatial ML, this means monitoring whether model accuracy degrades in certain regions or as input data characteristics shift over time.

```python
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import (
    DataDriftPreset,
    ClassificationPreset,
    TargetDriftPreset,
)

column_mapping = ColumnMapping(
    target="land_cover_true",
    prediction="land_cover_pred",
    numerical_features=["ndvi", "elevation", "slope", "population_density"],
    categorical_features=["region", "climate_zone"],
)

report = Report(metrics=[
    DataDriftPreset(),
    ClassificationPreset(),
    TargetDriftPreset(),
])

report.run(reference_data=df_reference, current_data=df_current,
           column_mapping=column_mapping)
report.save_html("model_monitoring_report.html")
```

> **Cross-reference:** [Dashboards](dashboards.md) for more general-purpose dashboard tools.

---

## 7. AutoML & Hyperparameter Visualization

### Optuna: Hyperparameter Optimization Visualization

- **Website:** [optuna.org](https://optuna.org)
- **License:** MIT
- **Install:** `pip install optuna optuna-dashboard`

Optuna provides built-in visualization for hyperparameter search results -- critical for understanding which configurations work best for your geospatial model.

```python
import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_contour,
    plot_parallel_coordinate,
    plot_slice,
)

def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    n_layers = trial.suggest_int("n_layers", 3, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])

    model = build_model(n_layers=n_layers, dropout=dropout)
    val_iou = train_and_evaluate(model, lr=lr, batch_size=batch_size,
                                  optimizer_name=optimizer)
    return val_iou

study = optuna.create_study(direction="maximize",
                             study_name="land_cover_hpo")
study.optimize(objective, n_trials=200)

# --- Visualization ---
# Optimization history
fig1 = plot_optimization_history(study)
fig1.write_image("optuna_history.png")

# Hyperparameter importance
fig2 = plot_param_importances(study)
fig2.write_image("optuna_importance.png")

# Contour plot: interaction between two parameters
fig3 = plot_contour(study, params=["lr", "dropout"])
fig3.write_image("optuna_contour.png")

# Parallel coordinates: all parameters at once
fig4 = plot_parallel_coordinate(study)
fig4.write_image("optuna_parallel.png")

# Slice plot: each parameter independently
fig5 = plot_slice(study)
fig5.write_image("optuna_slice.png")

# Best parameters
print(f"Best trial: {study.best_trial.value:.4f}")
print(f"Best params: {study.best_trial.params}")
```

### Feature Importance Ranking

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance

def plot_feature_importance_comparison(model, X_test, y_test, feature_names):
    """
    Compare multiple feature importance methods side by side.
    """
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    # 1. Built-in importance (tree-based models)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        sorted_idx = np.argsort(importances)
        axes[0].barh(range(len(sorted_idx)),
                     importances[sorted_idx], color="steelblue")
        axes[0].set_yticks(range(len(sorted_idx)))
        axes[0].set_yticklabels(np.array(feature_names)[sorted_idx])
        axes[0].set_title("Built-in Feature Importance (Gini/Gain)")

    # 2. Permutation importance
    perm_result = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=42
    )
    sorted_idx = perm_result.importances_mean.argsort()
    axes[1].boxplot(
        perm_result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(feature_names)[sorted_idx]
    )
    axes[1].set_title("Permutation Importance")

    # 3. Drop-column importance
    from sklearn.model_selection import cross_val_score
    baseline = cross_val_score(model, X_test, y_test, cv=5).mean()
    drop_importances = []
    for i in range(X_test.shape[1]):
        X_drop = np.delete(X_test, i, axis=1)
        # Retrain with one fewer feature
        score = cross_val_score(model, X_drop, y_test, cv=5).mean()
        drop_importances.append(baseline - score)

    sorted_idx = np.argsort(drop_importances)
    axes[2].barh(range(len(sorted_idx)),
                 np.array(drop_importances)[sorted_idx], color="coral")
    axes[2].set_yticks(range(len(sorted_idx)))
    axes[2].set_yticklabels(np.array(feature_names)[sorted_idx])
    axes[2].set_title("Drop-Column Importance")

    plt.tight_layout()
    plt.savefig("feature_importance_comparison.png", dpi=150, bbox_inches="tight")
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(model, X, y, cv=5):
    """
    Plot learning curves: performance as a function of training set size.
    Diagnoses underfitting vs overfitting.
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring="f1_macro",
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="blue")
    ax.fill_between(train_sizes, val_mean - val_std,
                     val_mean + val_std, alpha=0.1, color="orange")
    ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training Score")
    ax.plot(train_sizes, val_mean, "o-", color="orange", label="Validation Score")

    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("F1 Score (Macro)")
    ax.set_title("Learning Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Add gap annotation
    gap = train_mean[-1] - val_mean[-1]
    ax.annotate(
        f"Gap: {gap:.3f}\n{'Overfitting' if gap > 0.05 else 'Good fit'}",
        xy=(train_sizes[-1], (train_mean[-1] + val_mean[-1]) / 2),
        fontsize=10, ha="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow")
    )

    plt.tight_layout()
    plt.savefig("learning_curves.png", dpi=150, bbox_inches="tight")
```

---

## 8. Foundation Model Visualization

Foundation models for Earth observation (Prithvi-EO, SatMAE, Clay, IBM NASA Geospatial) are pre-trained on massive satellite imagery datasets. Visualizing their internals and outputs requires specialized techniques.

### Vision Transformer Attention Visualization

See which image patches SAM, Prithvi-EO, or SatMAE attends to when making predictions.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_foundation_model_attention(model, image_tensor, patch_size=16):
    """
    Visualize multi-head self-attention from a geospatial foundation model.
    Shows attention for all heads across multiple layers.
    """
    attention_maps = []

    def attention_hook(module, input, output):
        if isinstance(output, tuple):
            attention_maps.append(output[1].detach())

    # Register hooks on all attention layers
    hooks = []
    for block in model.blocks:
        h = block.attn.register_forward_hook(attention_hook)
        hooks.append(h)

    with torch.no_grad():
        _ = model(image_tensor.unsqueeze(0))

    for h in hooks:
        h.remove()

    n_layers = len(attention_maps)
    n_heads = attention_maps[0].shape[1]
    grid_size = int(np.sqrt(attention_maps[0].shape[-1] - 1))

    # Plot attention rollout (multiply attention matrices through layers)
    # This approximates how information flows from input to output
    rollout = torch.eye(attention_maps[0].shape[-1])
    for attn in attention_maps:
        attn_mean = attn[0].mean(dim=0)  # average over heads
        attn_mean = attn_mean + torch.eye(attn_mean.shape[-1])
        attn_mean = attn_mean / attn_mean.sum(dim=-1, keepdim=True)
        rollout = torch.matmul(attn_mean, rollout)

    # CLS token attention to patches
    cls_rollout = rollout[0, 1:].reshape(grid_size, grid_size)

    fig, axes = plt.subplots(2, min(n_heads, 6) + 1, figsize=(24, 8))

    # Row 1: Per-head attention (last layer)
    last_attn = attention_maps[-1][0]  # (n_heads, seq, seq)
    for h in range(min(n_heads, 6)):
        head_attn = last_attn[h, 0, 1:].reshape(grid_size, grid_size)
        axes[0, h].imshow(head_attn.cpu().numpy(), cmap="hot")
        axes[0, h].set_title(f"Head {h}")
        axes[0, h].axis("off")

    # Attention rollout
    axes[0, -1].imshow(cls_rollout.cpu().numpy(), cmap="hot")
    axes[0, -1].set_title("Attention Rollout")
    axes[0, -1].axis("off")

    # Row 2: Layer-by-layer average attention
    for l in range(min(n_layers, 6)):
        layer_attn = attention_maps[l][0].mean(dim=0)[0, 1:]
        layer_attn = layer_attn.reshape(grid_size, grid_size)
        axes[1, l].imshow(layer_attn.cpu().numpy(), cmap="hot")
        axes[1, l].set_title(f"Layer {l}")
        axes[1, l].axis("off")

    axes[1, -1].imshow(image_tensor[:3].permute(1, 2, 0).cpu().numpy())
    axes[1, -1].set_title("Original Image")
    axes[1, -1].axis("off")

    plt.suptitle("Foundation Model Attention Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig("foundation_attention.png", dpi=150, bbox_inches="tight")
```

### Embedding Space Visualization

Use t-SNE or UMAP to project high-dimensional satellite image embeddings into 2D. Color by land cover class to see how well the foundation model separates different terrain types.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap

def visualize_embedding_space(embeddings, labels, class_names, class_colors,
                                method="umap"):
    """
    Project high-dimensional embeddings to 2D and visualize.

    embeddings: (N, D) array of model embeddings
    labels: (N,) array of class indices
    """
    if method == "umap":
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                             metric="cosine", random_state=42)
        coords = reducer.fit_transform(embeddings)
    elif method == "tsne":
        reducer = TSNE(n_components=2, perplexity=30,
                       random_state=42, n_iter=1000)
        coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(12, 10))
    for cls_idx, (name, color) in enumerate(zip(class_names, class_colors)):
        mask = labels == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=name, s=10, alpha=0.6)

    ax.legend(markerscale=3, fontsize=10)
    ax.set_title(f"Satellite Image Embedding Space ({method.upper()})")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"embedding_space_{method}.png", dpi=150, bbox_inches="tight")
```

### Zero-Shot Classification Confidence

For CLIP-like geospatial models, visualize how confident the model is for different text prompts across an image.

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def zero_shot_confidence_map(model, image_tensor, text_prompts, tokenizer,
                              patch_size=16):
    """
    Generate per-patch confidence maps for zero-shot classification prompts.
    Works with CLIP-based geospatial models.
    """
    H, W = image_tensor.shape[1], image_tensor.shape[2]
    grid_h, grid_w = H // patch_size, W // patch_size

    # Encode text prompts
    text_tokens = tokenizer(text_prompts)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Extract patch-level image features
    with torch.no_grad():
        image_features = model.encode_image_patches(
            image_tensor.unsqueeze(0)
        )  # (1, n_patches, D)
        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

    # Compute similarity for each patch and prompt
    similarity = (image_features[0] @ text_features.T).cpu().numpy()
    # shape: (n_patches, n_prompts)

    # Reshape to spatial grid
    confidence_maps = similarity.reshape(grid_h, grid_w, len(text_prompts))

    # Plot
    n_prompts = len(text_prompts)
    fig, axes = plt.subplots(1, n_prompts + 1, figsize=(5 * (n_prompts + 1), 5))

    # Original image
    axes[0].imshow(image_tensor[:3].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    for i, prompt in enumerate(text_prompts):
        im = axes[i + 1].imshow(confidence_maps[:, :, i], cmap="hot",
                                 interpolation="bilinear")
        axes[i + 1].set_title(f'"{prompt}"')
        axes[i + 1].axis("off")
        plt.colorbar(im, ax=axes[i + 1])

    plt.suptitle("Zero-Shot Classification Confidence Maps", fontsize=14)
    plt.tight_layout()
    plt.savefig("zero_shot_confidence.png", dpi=150, bbox_inches="tight")
```

### Prompt Engineering Visualization for SAM

Visualize how different point prompts and bounding box prompts change SAM segmentation results.

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_sam_prompt_sensitivity(sam_model, image, prompt_configs):
    """
    Show how different prompts produce different SAM segmentation masks.

    prompt_configs: list of dicts, each with:
        - "points": list of (x, y) coords
        - "labels": list of 0/1 (background/foreground)
        - "box": optional [x1, y1, x2, y2]
        - "description": string
    """
    n = len(prompt_configs)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(6 * ((n + 1) // 2), 12))
    axes = axes.flatten()

    for idx, config in enumerate(prompt_configs):
        masks = sam_model.predict(
            image=image,
            point_coords=np.array(config.get("points", [])),
            point_labels=np.array(config.get("labels", [])),
            box=np.array(config.get("box")) if config.get("box") else None,
        )

        axes[idx].imshow(image)
        # Overlay best mask
        mask_overlay = np.zeros((*masks[0].shape, 4))
        mask_overlay[masks[0] > 0] = [1, 0, 0, 0.4]
        axes[idx].imshow(mask_overlay)

        # Plot prompt points
        if "points" in config:
            for (px, py), label in zip(config["points"], config["labels"]):
                color = "green" if label == 1 else "red"
                axes[idx].plot(px, py, "o", color=color, markersize=10)

        # Plot prompt box
        if "box" in config and config["box"]:
            x1, y1, x2, y2 = config["box"]
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor="blue",
                                  facecolor="none")
            axes[idx].add_patch(rect)

        axes[idx].set_title(config.get("description", f"Prompt {idx}"))
        axes[idx].axis("off")

    plt.suptitle("SAM Prompt Sensitivity Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig("sam_prompt_sensitivity.png", dpi=150, bbox_inches="tight")
```

---

## 9. Tool Comparison Matrix

| Tool | Type | Language | Geospatial Support | Interactivity | Cost | Best For |
|------|------|----------|-------------------|---------------|------|----------|
| **SHAP** | Explainability | Python | Via GeoPandas/folium mapping | Static + notebook | Free (MIT) | Feature importance, spatial SHAP maps |
| **LIME** | Explainability | Python | Image superpixels on satellite imagery | Static + notebook | Free (BSD) | Local explanations, image classification |
| **Captum** | Explainability | Python (PyTorch) | Via custom overlays | Static + notebook | Free (BSD) | Deep learning attribution, GradCAM |
| **TensorBoard** | Training Monitor | Python | Plugin-based | Interactive (browser) | Free (Apache 2.0) | Training curves, image logging, profiling |
| **W&B** | Experiment Tracking | Python | Custom panels, artifacts | Interactive (cloud) | Free tier + paid | Team collaboration, sweeps, artifact versioning |
| **MLflow** | ML Lifecycle | Python | Artifacts | Interactive (self-hosted) | Free (Apache 2.0) | Model registry, on-premise deployments |
| **Gradio** | ML Demo | Python | Via map widgets | Interactive (web) | Free (Apache 2.0) | Quick prototyping, public demos |
| **Streamlit** | Dashboard | Python | Via streamlit-folium, pydeck | Interactive (web) | Free (Apache 2.0) | Full inference dashboards, internal tools |
| **Evidently** | ML Monitoring | Python | By region/feature | Interactive (reports) | Free (Apache 2.0) | Data drift, model performance monitoring |
| **Optuna** | HPO | Python | N/A | Interactive (dashboard) | Free (MIT) | Hyperparameter optimization, parameter importance |

**Selection Guide:**

| Scenario | Recommended Stack |
|----------|------------------|
| Research paper, need to explain spatial model | SHAP + matplotlib + GeoPandas |
| Training a deep learning model for remote sensing | TensorBoard or W&B + TorchGeo |
| Building a demo for stakeholders | Streamlit + folium + SHAP |
| Monitoring production geospatial ML pipeline | Evidently + MLflow + W&B |
| Hyperparameter tuning for satellite image classifier | Optuna + W&B |
| Explaining a CNN prediction on satellite imagery | Captum (GradCAM) + rasterio overlay |
| Quick one-off model demo | Gradio |

---

## 10. Dark Arts Tips

**1. SHAP is slow on large geospatial datasets -- use sampling wisely.**

SHAP's KernelExplainer is computationally expensive. For large spatial datasets (100k+ samples), use `shap.TreeExplainer` for tree-based models (it is exact and fast). For neural networks, use `shap.DeepExplainer` or `shap.GradientExplainer` instead of `KernelExplainer`. As a fallback, subsample your data spatially (stratified by region) to maintain geographic representativeness.

```python
# Fast: TreeExplainer for XGBoost/LightGBM/Random Forest
explainer = shap.TreeExplainer(xgb_model)  # O(TLD) per sample

# Slow but model-agnostic: KernelExplainer -- subsample first
background = shap.sample(X_train, 100)  # use 100 background samples
explainer = shap.KernelExplainer(model.predict_proba, background)
# Only explain a subset
shap_values = explainer.shap_values(X_test[:500])
```

**2. TensorBoard image logging eats disk space -- log strategically.**

Logging full-resolution prediction maps every epoch will balloon your runs directory. Log images every 5-10 epochs, at reduced resolution, and for a fixed set of validation tiles so you can visually compare progress.

**3. W&B offline mode for air-gapped geospatial workstations.**

Many GIS labs and government agencies operate on restricted networks. W&B supports offline mode:

```bash
WANDB_MODE=offline python train.py
# Later, when online:
wandb sync ./wandb/offline-run-*
```

**4. Combine GradCAM with georeference information.**

GradCAM outputs a heatmap in pixel coordinates. To make it useful for GIS, warp it back to geographic coordinates using the original raster's transform:

```python
import rasterio
from rasterio.transform import from_bounds

# Write GradCAM as a GeoTIFF with the same CRS/transform as the input
with rasterio.open(input_raster_path) as src:
    profile = src.profile.copy()
    profile.update(count=1, dtype="float32")

    with rasterio.open("gradcam_geo.tif", "w", **profile) as dst:
        dst.write(gradcam_heatmap.astype(np.float32), 1)
```

Now you can load `gradcam_geo.tif` in QGIS and overlay it on other spatial layers.

**5. Use Optuna's pruning for expensive geospatial training.**

Satellite image models are expensive to train. Optuna can prune unpromising trials early, saving GPU hours:

```python
from optuna.pruners import MedianPruner

study = optuna.create_study(
    direction="maximize",
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)
```

Report intermediate values in your training loop with `trial.report(val_iou, epoch)` and call `trial.should_prune()` to stop bad trials.

**6. Spatial cross-validation changes everything about your metrics.**

Standard random train/test splits leak spatial autocorrelation. Use spatial cross-validation (e.g., `sklearn.model_selection.GroupKFold` with spatial blocks) and visualize the difference. Your honest accuracy will almost always be lower -- and that is the number you should report.

**7. Uncertainty maps are your best friend for active learning.**

Use MC Dropout or ensemble uncertainty maps to identify the most uncertain regions. Sample new training data from those regions. This is far more efficient than random sampling and can cut annotation costs by 50-70%.

**8. Log predictions as GeoJSON artifacts, not just images.**

Images in TensorBoard/W&B are useful for quick inspection, but GeoJSON predictions can be loaded into QGIS, kepler.gl, or any GIS software for proper spatial analysis. Always log both.

**9. Attention maps lie (sometimes).**

Attention weights in transformers do not always indicate true feature importance. They show where the model looks, not necessarily what drives the decision. Use attention visualization alongside gradient-based methods (Integrated Gradients, GradCAM) for a more complete picture.

**10. Batch your SHAP computations for large rasters.**

When computing SHAP values for every pixel in a large raster, process in tiles with overlap to avoid edge artifacts. Stitch the results back together for a seamless spatial SHAP map.

```python
def compute_shap_tiled(model, raster, tile_size=256, overlap=32):
    """Compute SHAP values tile by tile for large rasters."""
    H, W = raster.shape[1], raster.shape[2]
    shap_map = np.zeros((raster.shape[0], H, W))

    for y in range(0, H, tile_size - overlap):
        for x in range(0, W, tile_size - overlap):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            tile = raster[:, y:y_end, x:x_end]

            # Compute SHAP for this tile
            tile_shap = compute_shap_for_tile(model, tile)

            # Write interior (skip overlap on edges)
            y_start_write = y + overlap // 2 if y > 0 else y
            x_start_write = x + overlap // 2 if x > 0 else x
            y_off = y_start_write - y
            x_off = x_start_write - x

            shap_map[:, y_start_write:y_end, x_start_write:x_end] = \
                tile_shap[:, y_off:, x_off:]

    return shap_map
```

---

## Further Reading

- Molnar, C. (2022). *Interpretable Machine Learning* -- [christophm.github.io/interpretable-ml-book](https://christophm.github.io/interpretable-ml-book/)
- Lundberg, S. & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions* (SHAP paper)
- Ribeiro, M. et al. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME paper)
- TorchGeo documentation: [torchgeo.readthedocs.io](https://torchgeo.readthedocs.io)
- Raster Vision documentation: [docs.rastervision.io](https://docs.rastervision.io)
- segment-geospatial: [samgeo.gishub.org](https://samgeo.gishub.org)

---

[Back to Visualization](README.md) | [Back to Main README](../README.md)
