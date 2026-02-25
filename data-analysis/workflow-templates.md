# Workflow Templates

> Ten end-to-end geospatial analysis workflow templates for GIS professionals. Each includes a clear objective, required data with sources, step-by-step methodology, recommended tools across Python/R/Desktop GIS, production-ready code snippets, text-based flowcharts, expected outputs, and cross-references to companion resources. Adapt these to your specific study area and data availability.

> **Quick Picks**
> | Need | Template | Core Method |
> |---|---|---|
> | Planning a new facility | [Site Suitability Analysis](#1-site-suitability-analysis) | AHP + weighted overlay |
> | Monitoring land change | [Land Use Change Detection](#2-land-use-change-detection) | Post-classification comparison |
> | Measuring service coverage | [Accessibility Analysis](#3-accessibility-analysis) | Isochrones + 2SFCA |
> | Environmental review | [Environmental Impact Assessment](#4-environmental-impact-assessment) | Buffer + overlay analysis |
> | Urban climate study | [Urban Heat Island Analysis](#5-urban-heat-island-analysis) | LST + Gi* hot spots |
> | Flood hazard mapping | [Flood Risk Assessment](#6-flood-risk-assessment) | DEM hydrology + inundation modeling |
> | Crop & drought monitoring | [Agricultural Monitoring](#7-agricultural-monitoring) | NDVI time series + crop classification |
> | Population mapping | [Population Estimation](#8-population-estimation) | Dasymetric mapping + building footprints |
> | Logistics & equity | [Network Analysis & Routing](#9-network-analysis--routing) | Graph-based shortest path + service areas |
> | Pollution & health | [Air Quality Assessment](#10-air-quality-assessment) | LUR + exposure modeling |

> **Cross-references:** Data sources are catalogued in [`data-sources/`](../data-sources/); tool installation guides in [`tools/`](../tools/); cartographic guidance in [`visualization/`](../visualization/).

---

## Table of Contents

1. [Site Suitability Analysis](#1-site-suitability-analysis)
2. [Land Use Change Detection](#2-land-use-change-detection)
3. [Accessibility Analysis](#3-accessibility-analysis)
4. [Environmental Impact Assessment](#4-environmental-impact-assessment)
5. [Urban Heat Island Analysis](#5-urban-heat-island-analysis)
6. [Flood Risk Assessment](#6-flood-risk-assessment)
7. [Agricultural Monitoring](#7-agricultural-monitoring)
8. [Population Estimation](#8-population-estimation)
9. [Network Analysis & Routing](#9-network-analysis--routing)
10. [Air Quality Assessment](#10-air-quality-assessment)

---

## 1. Site Suitability Analysis

### Objective

Identify the most suitable locations for a facility (e.g., solar farm, school, warehouse, wind turbine) based on multiple weighted criteria using the Analytic Hierarchy Process (AHP) and weighted overlay analysis. The workflow produces a classified suitability raster with sensitivity analysis to validate robustness of the ranking.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| DEM (Digital Elevation Model) | Slope, aspect, elevation constraints | SRTM, ASTER GDEM, national LiDAR | GeoTIFF, 30m |
| Land use / land cover | Exclude unsuitable areas (water, forest, urban) | CORINE, NLCD, ESA WorldCover | GeoTIFF, 10-30m |
| Road network | Proximity to transport | OpenStreetMap, national road database | Shapefile / GeoPackage |
| Utility infrastructure | Proximity to power lines, water supply | Local utility provider | Shapefile |
| Protected areas | Exclusion zones | WDPA, national environment agency | Shapefile / GeoPackage |
| Soil type / geology | Foundation suitability, permeability | National soil survey, SoilGrids | Shapefile / GeoTIFF |
| Population / demand data | Proximity to demand centers | Census, WorldPop | CSV / GeoTIFF |
| Solar irradiance (for solar) | Energy potential | Global Solar Atlas, PVGIS | GeoTIFF |

**Data Portals:** [SRTM via USGS EarthExplorer](https://earthexplorer.usgs.gov/) | [ESA WorldCover](https://worldcover2021.esa.int/) | [CORINE (Copernicus)](https://land.copernicus.eu/pan-european/corine-land-cover) | [NLCD (MRLC)](https://www.mrlc.gov/) | [OpenStreetMap (Geofabrik)](https://download.geofabrik.de/) | [WDPA](https://www.protectedplanet.net/) | [WorldPop](https://www.worldpop.org/) | [SoilGrids](https://soilgrids.org/) | [Global Solar Atlas](https://globalsolaratlas.info/)

> See also: [`data-sources/elevation.md`](../data-sources/elevation.md), [`data-sources/land-cover.md`](../data-sources/land-cover.md)

### Steps

1. **Define criteria** -- List all factors; classify each as a constraint (binary: suitable/unsuitable) or a factor (continuous: more/less suitable). Document rationale for every criterion.
2. **Acquire and preprocess data** -- Download, clip to study area, unify CRS (recommend a projected CRS such as UTM), resample all rasters to a common resolution (e.g., 30 m).
3. **Create constraint layers** -- Binary masks (1 = suitable, 0 = excluded). Examples: slope < 15 degrees, not in a protected area, not on water bodies, outside flood zones.
4. **Create factor layers** -- Normalize each factor to a 0-1 scale using min-max or fuzzy membership. Examples: distance to road (closer = higher score), solar irradiance (higher = higher score).
5. **Assign weights via AHP** -- Construct a pairwise comparison matrix, compute the priority vector (eigenvector method), check consistency ratio (CR < 0.10). Weights must sum to 1.0.
6. **Weighted overlay** -- Multiply each normalized factor by its weight, sum all factors, then apply the combined constraint mask.
7. **Classify suitability** -- Divide the result into classes (e.g., High / Moderate / Low / Unsuitable) using natural breaks (Jenks), quantiles, or domain thresholds.
8. **Sensitivity analysis** -- Vary each weight by +/- 10-20 % (one-at-a-time or Monte Carlo) and check whether the top-ranked areas remain stable. Report Spearman rank correlation between original and perturbed results.
9. **Validate** -- Compare results with known suitable sites, field verification, or independent expert review. Compute agreement statistics.
10. **Report** -- Map of suitability classes, area statistics, methodology description, AHP weight table, sensitivity results.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Data preprocessing | `rasterio`, `geopandas`, `pyproj` | `sf`, `terra` | QGIS Processing, ArcGIS |
| Distance calculation | `scipy.ndimage.distance_transform_edt` | `terra::distance()` | QGIS Proximity (raster) |
| Normalization | `numpy` min-max / fuzzy | `terra::classify()` | Raster Calculator |
| AHP weights | `ahpy` package | `ahpsurvey` package | Manual / Excel |
| Weighted overlay | `numpy` array math | `terra` weighted sum | Weighted Overlay tool |
| Classification | `numpy.digitize`, `jenkspy` | `classInt::classIntervals()` | Reclassify |
| Sensitivity analysis | Custom loop + `scipy.stats.spearmanr` | `sensitivity` package | Manual or scripted |
| Visualization | `matplotlib`, `contextily` | `tmap`, `ggplot2` | QGIS Print Layout |

### Code: AHP Weighted Overlay in Python

```python
"""
Site Suitability Analysis — AHP + Weighted Overlay
Requirements: pip install rasterio numpy scipy ahpy jenkspy geopandas matplotlib contextily
"""
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.ndimage import distance_transform_edt
import ahpy
import jenkspy
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
STUDY_CRS = "EPSG:32633"          # UTM zone 33N — adjust for your area
CELL_SIZE = 30                     # metres
NODATA = -9999

CRITERIA = {
    "slope":       {"type": "factor",     "weight_name": "slope",    "goal": "minimize"},
    "dist_road":   {"type": "factor",     "weight_name": "dist_road","goal": "minimize"},
    "solar":       {"type": "factor",     "weight_name": "solar",    "goal": "maximize"},
    "dist_grid":   {"type": "factor",     "weight_name": "dist_grid","goal": "minimize"},
    "protected":   {"type": "constraint", "exclude_value": 1},
    "water":       {"type": "constraint", "exclude_value": 1},
    "slope_steep":  {"type": "constraint", "threshold": 15},  # degrees
}

# -------------------------------------------------------------------
# 2. Helper functions
# -------------------------------------------------------------------
def read_raster(path):
    """Read a single-band raster and return (array, profile)."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float64)
        profile = src.profile.copy()
    return arr, profile

def normalize_minmax(arr, goal="maximize", nodata=NODATA):
    """Min-max normalize to 0-1. goal='minimize' inverts the scale."""
    mask = arr == nodata
    valid = arr[~mask]
    lo, hi = valid.min(), valid.max()
    if hi == lo:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    if goal == "minimize":
        norm = 1.0 - norm
    norm[mask] = nodata
    return norm

def euclidean_distance_raster(binary_arr, cell_size, nodata=NODATA):
    """Compute Euclidean distance from feature pixels (value=1)."""
    target = (binary_arr == 1)
    dist = distance_transform_edt(~target) * cell_size  # metres
    return dist

# -------------------------------------------------------------------
# 3. AHP pairwise comparison and weight derivation
# -------------------------------------------------------------------
# Saaty scale: 1=equal, 3=moderate, 5=strong, 7=very strong, 9=extreme
comparisons = {
    ("solar", "slope"):     3,
    ("solar", "dist_road"): 2,
    ("solar", "dist_grid"): 2,
    ("slope", "dist_road"): 1/2,
    ("slope", "dist_grid"): 1,
    ("dist_road", "dist_grid"): 2,
}

ahp_model = ahpy.Compare(
    name="SiteSuitability",
    comparisons=comparisons,
    precision=4
)

print("AHP Weights:", ahp_model.target_weights)
print("Consistency Ratio:", round(ahp_model.consistency_ratio, 4))
assert ahp_model.consistency_ratio < 0.10, "CR too high — revise comparisons"

weights = ahp_model.target_weights
# Example output: {'solar': 0.42, 'slope': 0.17, 'dist_road': 0.25, 'dist_grid': 0.16}

# -------------------------------------------------------------------
# 4. Load and preprocess layers (all already reprojected & resampled)
# -------------------------------------------------------------------
slope_arr, profile = read_raster("data/slope_degrees.tif")
solar_arr, _       = read_raster("data/solar_irradiance.tif")
roads_binary, _    = read_raster("data/roads_rasterized.tif")   # 1 = road pixel
grid_binary, _     = read_raster("data/grid_rasterized.tif")    # 1 = power line pixel
protected_arr, _   = read_raster("data/protected_areas.tif")    # 1 = protected
water_arr, _       = read_raster("data/water_bodies.tif")       # 1 = water

# Distance rasters
dist_road = euclidean_distance_raster(roads_binary, CELL_SIZE)
dist_grid = euclidean_distance_raster(grid_binary, CELL_SIZE)

# -------------------------------------------------------------------
# 5. Normalize factors to 0-1
# -------------------------------------------------------------------
norm_slope    = normalize_minmax(slope_arr, goal="minimize")
norm_solar    = normalize_minmax(solar_arr, goal="maximize")
norm_dist_road = normalize_minmax(dist_road, goal="minimize")
norm_dist_grid = normalize_minmax(dist_grid, goal="minimize")

# -------------------------------------------------------------------
# 6. Create combined constraint mask (1=suitable, 0=excluded)
# -------------------------------------------------------------------
constraint_mask = np.ones_like(slope_arr, dtype=np.uint8)
constraint_mask[protected_arr == 1] = 0
constraint_mask[water_arr == 1] = 0
constraint_mask[slope_arr > 15] = 0  # too steep

print(f"Excluded {(constraint_mask == 0).sum()} pixels "
      f"({(constraint_mask == 0).mean() * 100:.1f}% of study area)")

# -------------------------------------------------------------------
# 7. Weighted overlay
# -------------------------------------------------------------------
suitability = (
    weights["slope"]    * norm_slope +
    weights["solar"]    * norm_solar +
    weights["dist_road"] * norm_dist_road +
    weights["dist_grid"] * norm_dist_grid
)

# Apply constraint mask
suitability[constraint_mask == 0] = NODATA

# -------------------------------------------------------------------
# 8. Classify using Jenks natural breaks
# -------------------------------------------------------------------
valid_pixels = suitability[suitability != NODATA]
breaks = jenkspy.jenks_breaks(valid_pixels.tolist(), n_classes=4)
# Classes: 1=Low, 2=Moderate, 3=High, 4=Very High
classified = np.full_like(suitability, NODATA, dtype=np.float64)
classified[(suitability >= breaks[0]) & (suitability < breaks[1])] = 1
classified[(suitability >= breaks[1]) & (suitability < breaks[2])] = 2
classified[(suitability >= breaks[2]) & (suitability < breaks[3])] = 3
classified[(suitability >= breaks[3]) & (suitability <= breaks[4])] = 4
classified[constraint_mask == 0] = NODATA

# -------------------------------------------------------------------
# 9. Save classified raster
# -------------------------------------------------------------------
out_profile = profile.copy()
out_profile.update(dtype="float64", nodata=NODATA)
with rasterio.open("output/suitability_classified.tif", "w", **out_profile) as dst:
    dst.write(classified, 1)

# Area statistics
pixel_area_ha = (CELL_SIZE ** 2) / 10_000
labels = {1: "Low", 2: "Moderate", 3: "High", 4: "Very High"}
print("\n--- Suitability Area Statistics ---")
for cls, name in labels.items():
    count = (classified == cls).sum()
    area = count * pixel_area_ha
    print(f"  {name:12s}: {count:>8,} px  |  {area:>10,.1f} ha")

# -------------------------------------------------------------------
# 10. Sensitivity analysis — one-at-a-time (OAT) weight perturbation
# -------------------------------------------------------------------
from scipy.stats import spearmanr

original_flat = suitability[suitability != NODATA]
perturbation = 0.15  # +/- 15%

print("\n--- Sensitivity Analysis (OAT, +/-15%) ---")
for factor_name in weights:
    for direction in [-1, 1]:
        perturbed_weights = weights.copy()
        delta = perturbed_weights[factor_name] * perturbation * direction
        perturbed_weights[factor_name] += delta
        # Redistribute delta proportionally among other factors
        others = [k for k in perturbed_weights if k != factor_name]
        other_sum = sum(perturbed_weights[k] for k in others)
        for k in others:
            perturbed_weights[k] -= delta * (perturbed_weights[k] / other_sum)

        suit_p = (
            perturbed_weights["slope"]     * norm_slope +
            perturbed_weights["solar"]     * norm_solar +
            perturbed_weights["dist_road"] * norm_dist_road +
            perturbed_weights["dist_grid"] * norm_dist_grid
        )
        suit_p[constraint_mask == 0] = NODATA
        perturbed_flat = suit_p[suit_p != NODATA]

        rho, pval = spearmanr(original_flat, perturbed_flat)
        sign = "+" if direction == 1 else "-"
        print(f"  {factor_name:12s} {sign}{int(perturbation*100)}%: "
              f"Spearman rho = {rho:.6f}  (p = {pval:.2e})")
```

### Code: AHP Weighted Overlay in QGIS Processing (PyQGIS)

```python
"""
QGIS Processing script for site suitability weighted overlay.
Run from the QGIS Python console or as a Processing script.
"""
from qgis.core import QgsRasterLayer, QgsProject
import processing

# Paths to preprocessed layers (all same CRS and resolution)
layers = {
    "slope":    "/data/slope_norm.tif",
    "solar":    "/data/solar_norm.tif",
    "dist_road": "/data/dist_road_norm.tif",
    "dist_grid": "/data/dist_grid_norm.tif",
    "constraint": "/data/constraint_mask.tif",
}

# AHP-derived weights
w = {"slope": 0.17, "solar": 0.42, "dist_road": 0.25, "dist_grid": 0.16}

# Weighted overlay via raster calculator
expression = (
    f'("{layers["slope"]}@1" * {w["slope"]} + '
    f'"{layers["solar"]}@1" * {w["solar"]} + '
    f'"{layers["dist_road"]}@1" * {w["dist_road"]} + '
    f'"{layers["dist_grid"]}@1" * {w["dist_grid"]}) * '
    f'"{layers["constraint"]}@1"'
)

result = processing.run("qgis:rastercalculator", {
    "EXPRESSION": expression,
    "LAYERS": [layers["slope"]],
    "CELLSIZE": 30,
    "EXTENT": None,   # auto from layers
    "CRS": "EPSG:32633",
    "OUTPUT": "/output/suitability_raw.tif"
})

# Reclassify into 4 classes using equal intervals
processing.run("native:reclassifybytable", {
    "INPUT_RASTER": result["OUTPUT"],
    "RASTER_BAND": 1,
    "TABLE": [0, 0.25, 1,  0.25, 0.50, 2,  0.50, 0.75, 3,  0.75, 1.0, 4],
    "NO_DATA": -9999,
    "RANGE_BOUNDARIES": 0,   # min <= x < max
    "OUTPUT": "/output/suitability_classified.tif"
})
print("Suitability analysis complete.")
```

### Expected Output

- **Suitability raster** (classified GeoTIFF, 4 classes)
- **Suitability map** (publication-ready with legend, scale bar, north arrow)
- **Statistics table**: area per suitability class, top-N candidate sites with coordinates
- **AHP weight table**: pairwise comparison matrix, priority vector, consistency ratio
- **Sensitivity analysis summary**: Spearman rho for each weight perturbation scenario
- **Methodology report**: criteria justification, data sources, limitations

### Flowchart

```
[Define Criteria & Weights]
         |
         v
[Acquire Data] ---> [Clip & Reproject to Common CRS] ---> [Resample to Common Resolution]
         |
         +---------------+---------------+---------------+
         |               |               |               |
         v               v               v               v
   [Constraint      [Factor          [Factor         [Factor
    Layers]          Layer 1]         Layer 2]        Layer N]
   (binary 0/1)     (raw values)     (raw values)    (raw values)
         |               |               |               |
         |               v               v               v
         |          [Normalize 0-1] [Normalize 0-1] [Normalize 0-1]
         |               |               |               |
         |               +-------+-------+-------+------+
         |                       |
         |                       v
         |              [AHP Pairwise Comparison]
         |              [Consistency Check CR<0.10]
         |                       |
         |                       v
         |              [Weighted Sum = Sum(wi * fi)]
         |                       |
         +------------>+<--------+
                       |
                       v
              [Apply Constraint Mask]
                       |
                       v
              [Classify (Jenks / Quantile)]
                       |
                       v
         [Sensitivity Analysis (OAT / Monte Carlo)]
                       |
                       v
              [Validate & Report]
```

---

## 2. Land Use Change Detection

### Objective

Detect, quantify, and map changes in land use / land cover (LULC) between two or more time periods using satellite imagery and supervised classification. The workflow produces bias-adjusted area estimates with confidence intervals following best practices from Olofsson et al. (2014).

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Satellite imagery (T1) | Baseline land cover | Sentinel-2, Landsat 8/9, NAIP | L2A surface reflectance |
| Satellite imagery (T2) | Current land cover | Same sensor as T1 (consistency) | L2A surface reflectance |
| Training samples (T1 & T2) | Supervised classification | Field data, photo-interpretation, existing maps | GeoPackage / Shapefile |
| Validation samples | Independent accuracy assessment | Stratified random sample | GeoPackage |
| Administrative boundaries | Reporting units | National boundary dataset | Shapefile |
| Reference change data | Cross-validation | High-resolution imagery (Google Earth), field surveys | Various |

**Data Portals:** [Copernicus Open Access Hub](https://scihub.copernicus.eu/) | [USGS EarthExplorer](https://earthexplorer.usgs.gov/) | [Google Earth Engine Data Catalog](https://developers.google.com/earth-engine/datasets) | [Planetary Computer](https://planetarycomputer.microsoft.com/)

> See also: [`data-sources/satellite-imagery.md`](../data-sources/satellite-imagery.md)

### Steps

1. **Image selection** -- Choose images from the same season (to minimize phenological differences), same sensor, cloud cover < 10%. Prefer anniversary dates.
2. **Preprocessing** -- Atmospheric correction (to surface reflectance via Sen2Cor or LEDAPS), cloud/shadow masking (s2cloudless, Fmask), co-registration check.
3. **Feature engineering** -- Compute spectral indices (NDVI, NDWI, NDBI, SAVI), texture metrics (GLCM), and optionally terrain derivatives.
4. **Training data preparation** -- Stratified sampling, minimum 50 samples per class, spatial autocorrelation-aware splits.
5. **Classification (T1)** -- Train a Random Forest or Gradient Boosting classifier, apply to T1 image.
6. **Classification (T2)** -- Train classifier for T2 (can reuse model if spectral consistency is confirmed).
7. **Accuracy assessment** -- Stratified random validation sample, error matrix, overall accuracy, producer's/user's accuracy, kappa (or preferably quantity/allocation disagreement).
8. **Post-classification comparison** -- Cross-tabulate T1 and T2 maps to produce a change matrix.
9. **Bias-adjusted area estimation** -- Apply Olofsson et al. (2014) method: stratified estimation with confidence intervals based on the error matrix.
10. **Change map** -- Visualize "from-to" transitions (e.g., Forest to Urban, Cropland to Forest).
11. **Annual rate of change** -- Compute annualized rates using the compound formula.
12. **Report** -- Maps (T1, T2, change), change matrix, area estimates with 95% CI, methodology.

### Tools

| Step | Python | R | Desktop GIS / Cloud |
|---|---|---|---|
| Image download | `pystac-client`, `planetary_computer`, `sentinelsat` | `sen2r`, `getSpatialData`, `rstac` | Copernicus Hub, EarthExplorer |
| Preprocessing | `rasterio`, `s2cloudless`, `sen2cor` | `terra`, `sen2r` | SNAP, QGIS SCP |
| Feature engineering | `rasterio`, `numpy`, `scikit-image` (GLCM) | `terra`, `glcm` package | QGIS Raster Calculator |
| Classification | `scikit-learn`, `xgboost`, `lightgbm` | `caret`, `ranger`, `xgboost` | QGIS SCP, GEE |
| Accuracy assessment | `sklearn.metrics`, custom | `caret::confusionMatrix()` | Manual / Excel |
| Area estimation | Custom (Olofsson method) | `mapaccuracy` package | Manual |
| Visualization | `matplotlib`, `rasterio.plot` | `tmap`, `ggplot2` | QGIS, ArcGIS |

### Code: Classification and Change Detection in Python

```python
"""
Land Use Change Detection — Random Forest + Post-Classification Comparison
Requirements: pip install rasterio numpy scikit-learn geopandas matplotlib
"""
import numpy as np
import rasterio
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -------------------------------------------------------------------
# 1. Load imagery and training data
# -------------------------------------------------------------------
def load_image_stack(path):
    """Load multi-band image as (rows, cols, bands) array."""
    with rasterio.open(path) as src:
        img = src.read()  # (bands, rows, cols)
        profile = src.profile.copy()
    return np.moveaxis(img, 0, -1), profile  # -> (rows, cols, bands)

def extract_training_pixels(image, training_gdf, label_col, profile):
    """Extract pixel values at training point locations."""
    from rasterio.transform import rowcol
    transform = profile["transform"]
    X, y = [], []
    for _, row in training_gdf.iterrows():
        r, c = rowcol(transform, row.geometry.x, row.geometry.y)
        if 0 <= r < image.shape[0] and 0 <= c < image.shape[1]:
            X.append(image[r, c, :])
            y.append(row[label_col])
    return np.array(X), np.array(y)

# Load images
img_t1, profile_t1 = load_image_stack("data/sentinel2_t1_stack.tif")
img_t2, profile_t2 = load_image_stack("data/sentinel2_t2_stack.tif")

# Add spectral indices as extra bands
def add_indices(img):
    """Append NDVI, NDWI, NDBI as extra bands. Assumes B2,B3,B4,B8,B11,B12 order."""
    red, nir, swir = img[:,:,2], img[:,:,3], img[:,:,4]
    green = img[:,:,1]
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndwi = (green - nir) / (green + nir + 1e-10)
    ndbi = (swir - nir) / (swir + nir + 1e-10)
    return np.dstack([img, ndvi, ndwi, ndbi])

img_t1 = add_indices(img_t1)
img_t2 = add_indices(img_t2)

# Load training samples
train_t1 = gpd.read_file("data/training_t1.gpkg")
train_t2 = gpd.read_file("data/training_t2.gpkg")

X_t1, y_t1 = extract_training_pixels(img_t1, train_t1, "lulc_class", profile_t1)
X_t2, y_t2 = extract_training_pixels(img_t2, train_t2, "lulc_class", profile_t2)

# -------------------------------------------------------------------
# 2. Train and validate Random Forest classifiers
# -------------------------------------------------------------------
def train_and_validate(X, y, n_trees=500):
    """Train RF with stratified 5-fold cross-validation."""
    clf = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oa_scores = []
    for train_idx, val_idx in skf.split(X, y):
        clf.fit(X[train_idx], y[train_idx])
        oa_scores.append(clf.score(X[val_idx], y[val_idx]))
    print(f"  Cross-val OA: {np.mean(oa_scores):.3f} +/- {np.std(oa_scores):.3f}")
    # Final model on all data
    clf.fit(X, y)
    return clf

print("Training T1 classifier...")
clf_t1 = train_and_validate(X_t1, y_t1)

print("Training T2 classifier...")
clf_t2 = train_and_validate(X_t2, y_t2)

# -------------------------------------------------------------------
# 3. Classify full images
# -------------------------------------------------------------------
def classify_image(image, clf):
    """Apply classifier to entire image, returning classified 2D array."""
    rows, cols, bands = image.shape
    flat = image.reshape(-1, bands)
    predicted = clf.predict(flat)
    return predicted.reshape(rows, cols)

classified_t1 = classify_image(img_t1, clf_t1)
classified_t2 = classify_image(img_t2, clf_t2)

# -------------------------------------------------------------------
# 4. Accuracy assessment
# -------------------------------------------------------------------
val_points = gpd.read_file("data/validation_samples.gpkg")
# Assumed fields: geometry (point), lulc_t1, lulc_t2

def accuracy_report(classified, val_gdf, label_col, profile):
    """Generate confusion matrix and classification report."""
    from rasterio.transform import rowcol
    transform = profile["transform"]
    y_true, y_pred = [], []
    for _, row in val_gdf.iterrows():
        r, c = rowcol(transform, row.geometry.x, row.geometry.y)
        if 0 <= r < classified.shape[0] and 0 <= c < classified.shape[1]:
            y_true.append(row[label_col])
            y_pred.append(classified[r, c])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))
    return cm, report, y_true, y_pred

print("\n=== T1 Accuracy ===")
cm_t1, report_t1, _, _ = accuracy_report(classified_t1, val_points, "lulc_t1", profile_t1)

print("\n=== T2 Accuracy ===")
cm_t2, report_t2, _, _ = accuracy_report(classified_t2, val_points, "lulc_t2", profile_t2)

# -------------------------------------------------------------------
# 5. Post-classification comparison — change matrix
# -------------------------------------------------------------------
classes = sorted(np.unique(np.concatenate([classified_t1.ravel(), classified_t2.ravel()])))
n_classes = len(classes)

change_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
for i, c1 in enumerate(classes):
    for j, c2 in enumerate(classes):
        change_matrix[i, j] = np.sum((classified_t1 == c1) & (classified_t2 == c2))

class_names = ["Water", "Forest", "Cropland", "Urban", "Bare"]  # adjust to your classes
print("\n=== Change Matrix (pixel counts) ===")
header = f"{'':>12s}" + "".join(f"{n:>12s}" for n in class_names)
print(header)
for i, name in enumerate(class_names):
    row_str = f"{name:>12s}" + "".join(f"{change_matrix[i,j]:>12,}" for j in range(n_classes))
    print(row_str)

# -------------------------------------------------------------------
# 6. Bias-adjusted area estimation (Olofsson et al., 2014)
# -------------------------------------------------------------------
def olofsson_area_estimation(error_matrix, map_areas, class_names, confidence=1.96):
    """
    Bias-adjusted area estimation with confidence intervals.
    error_matrix: n x n array (rows = map class, cols = reference class)
    map_areas: array of mapped area per class (e.g., in hectares)
    """
    n = len(class_names)
    # Proportional error matrix
    nj = error_matrix.sum(axis=1)  # row totals (sample size per stratum)
    Wj = map_areas / map_areas.sum()  # area weights per stratum

    # Estimated area proportions
    p_hat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            p_hat[i, j] = Wj[i] * error_matrix[i, j] / nj[i] if nj[i] > 0 else 0

    # Estimated area per reference class
    area_total = map_areas.sum()
    A_hat = np.array([p_hat[:, j].sum() * area_total for j in range(n)])

    # Variance of estimated area proportion
    V_hat = np.zeros(n)
    for j in range(n):
        var_sum = 0
        for i in range(n):
            if nj[i] > 1:
                pij = error_matrix[i, j] / nj[i] if nj[i] > 0 else 0
                var_sum += Wj[i]**2 * pij * (1 - pij) / (nj[i] - 1)
        V_hat[j] = var_sum

    SE_hat = np.sqrt(V_hat) * area_total
    CI_lower = A_hat - confidence * SE_hat
    CI_upper = A_hat + confidence * SE_hat

    print("\n=== Bias-Adjusted Area Estimates (95% CI) ===")
    for j, name in enumerate(class_names):
        print(f"  {name:>12s}: {A_hat[j]:>12,.1f} ha  "
              f"[{CI_lower[j]:>12,.1f} — {CI_upper[j]:>12,.1f}]")

    return A_hat, SE_hat, CI_lower, CI_upper

# Compute mapped areas (T2)
pixel_area_ha = (30 ** 2) / 10_000  # 30m pixels to hectares
map_areas_t2 = np.array([(classified_t2 == c).sum() * pixel_area_ha for c in classes])

A_hat, SE_hat, CI_lo, CI_hi = olofsson_area_estimation(
    cm_t2, map_areas_t2, class_names
)
```

### Code: Change Detection in Google Earth Engine (JavaScript)

```javascript
// ============================================================
// Land Use Change Detection — Google Earth Engine
// ============================================================

// 1. Define study area and time periods
var roi = ee.FeatureCollection('FAO/GAUL/2015/level1')
    .filter(ee.Filter.eq('ADM1_NAME', 'Your Region'));

var t1_start = '2015-06-01', t1_end = '2015-09-30';
var t2_start = '2023-06-01', t2_end = '2023-09-30';

// 2. Load Sentinel-2 composites
function getComposite(start, end) {
    return ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(roi)
        .filterDate(start, end)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        .median()
        .clip(roi);
}

var img_t1 = getComposite(t1_start, t1_end);
var img_t2 = getComposite(t2_start, t2_end);

// 3. Add spectral indices
function addIndices(img) {
    var ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var ndwi = img.normalizedDifference(['B3', 'B8']).rename('NDWI');
    var ndbi = img.normalizedDifference(['B11', 'B8']).rename('NDBI');
    return img.addBands([ndvi, ndwi, ndbi]);
}

img_t1 = addIndices(img_t1);
img_t2 = addIndices(img_t2);

// 4. Classification bands
var bands = ['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12','NDVI','NDWI','NDBI'];

// 5. Load training data (upload as assets)
var training_t1 = ee.FeatureCollection('users/yourname/training_t1');
var training_t2 = ee.FeatureCollection('users/yourname/training_t2');

// 6. Train classifiers
var trainData_t1 = img_t1.select(bands).sampleRegions({
    collection: training_t1, properties: ['lulc'], scale: 10
});
var clf_t1 = ee.Classifier.smileRandomForest(500).train(trainData_t1, 'lulc', bands);

var trainData_t2 = img_t2.select(bands).sampleRegions({
    collection: training_t2, properties: ['lulc'], scale: 10
});
var clf_t2 = ee.Classifier.smileRandomForest(500).train(trainData_t2, 'lulc', bands);

// 7. Classify
var classified_t1 = img_t1.select(bands).classify(clf_t1).rename('lulc_t1');
var classified_t2 = img_t2.select(bands).classify(clf_t2).rename('lulc_t2');

// 8. Change detection — encode from-to as (T1 * 100 + T2)
var change = classified_t1.multiply(100).add(classified_t2).rename('change');

// 9. Visualize
var palette = ['blue','green','yellow','red','gray'];
Map.centerObject(roi, 10);
Map.addLayer(classified_t1, {min:0, max:4, palette:palette}, 'LULC T1');
Map.addLayer(classified_t2, {min:0, max:4, palette:palette}, 'LULC T2');
Map.addLayer(change, {min:0, max:404}, 'Change');

// 10. Area statistics
var areaT2 = ee.Image.pixelArea().divide(10000)  // hectares
    .addBands(classified_t2)
    .reduceRegion({
        reducer: ee.Reducer.sum().group({groupField: 1}),
        geometry: roi, scale: 10, maxPixels: 1e13
    });
print('Area by class (ha):', areaT2);
```

### Expected Output

- **Classified maps** for T1 and T2 (GeoTIFF or GEE asset)
- **Change detection map** (from-to class transitions, color-coded)
- **Change matrix** (cross-tabulation table with pixel counts and areas)
- **Bias-adjusted area estimates** per class with 95% confidence intervals
- **Accuracy assessment reports** (confusion matrices, OA, PA, UA) for each date
- **Annual rate of change** per transition type
- **Methodology report** referencing Olofsson et al. (2014) best practices

### Flowchart

```
[Image T1 (Surface Reflectance)]         [Image T2 (Surface Reflectance)]
         |                                         |
         v                                         v
[Cloud Mask + Preprocessing]             [Cloud Mask + Preprocessing]
         |                                         |
         v                                         v
[Add Spectral Indices]                   [Add Spectral Indices]
  (NDVI, NDWI, NDBI)                      (NDVI, NDWI, NDBI)
         |                                         |
         v                                         v
[Train RF Classifier]                    [Train RF Classifier]
  (training samples T1)                    (training samples T2)
         |                                         |
         v                                         v
[Classify T1]                            [Classify T2]
         |                                         |
         v                                         v
[Accuracy Assessment]                    [Accuracy Assessment]
  (confusion matrix, OA, PA, UA)           (confusion matrix, OA, PA, UA)
         |                                         |
         +-----------------+-----------------------+
                           |
                           v
             [Post-Classification Comparison]
             [Cross-tabulate T1 x T2 classes]
                           |
                           v
                    [Change Matrix]
                           |
                           v
          [Bias-Adjusted Area Estimation]
          [Olofsson et al. 2014 method]
          [95% Confidence Intervals]
                           |
                           v
               [Change Map & Report]
```

---

## 3. Accessibility Analysis

### Objective

Measure how accessible a service or facility is to the surrounding population using travel time, distance, or generalized cost as the metric. Produce isochrone maps, origin-destination travel time matrices, and equity-informed accessibility indicators including the Two-Step Floating Catchment Area (2SFCA) method.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Road network | Travel routes and speeds | OpenStreetMap (via OSMnx), HERE, TomTom | Graph / Shapefile |
| Facility locations | Destinations (hospitals, schools, etc.) | Government open data, POI databases | GeoPackage / CSV |
| Population data | Demand / catchment | WorldPop, census blocks, GHSL | Raster / Shapefile |
| Administrative boundaries | Reporting units | National boundary dataset | Shapefile |
| Public transit GTFS (optional) | Multi-modal analysis | Local transit authority | GTFS .zip |
| Elevation data (optional) | Walking/cycling impedance | SRTM, national LiDAR | GeoTIFF |

**Data Portals:** [OpenStreetMap (Geofabrik)](https://download.geofabrik.de/) | [Transitland GTFS](https://www.transit.land/) | [OpenMobilityData](https://transitfeeds.com/) | [WorldPop](https://www.worldpop.org/) | [GHSL](https://ghsl.jrc.ec.europa.eu/)

> See also: [`data-sources/transport.md`](../data-sources/transport.md), [`tools/network-analysis.md`](../tools/network-analysis.md)

### Steps

1. **Define the service and metric** -- What facility type? What travel mode(s)? What metric (time, distance, generalized cost)? What threshold (e.g., 30-minute drive)?
2. **Build the network** -- Download or prepare the road/transit network as a routable graph. For multi-modal, integrate GTFS data.
3. **Assign travel speeds** -- Based on road type (`highway` tag in OSM), surface quality, speed limits, congestion factors. For walking, use ~5 km/h; cycling ~15 km/h.
4. **Calculate isochrones** -- From each facility, compute travel-time zones (e.g., 5, 10, 15, 30, 60 minutes).
5. **Calculate OD matrix** -- Origin-destination travel times from population centroids to all facilities within the threshold.
6. **Compute accessibility indicators:**
   - *Nearest facility travel time* -- Minimum travel time to any facility
   - *Cumulative opportunities* -- Count of facilities within threshold time
   - *Gravity-based* -- Sum of facilities weighted by distance decay (e.g., Gaussian, power)
   - *Two-Step Floating Catchment Area (2SFCA)* -- Supply-demand ratio accounting for facility capacity
   - *Enhanced 2SFCA (E2SFCA)* -- Adds distance decay weights within catchment sub-zones
7. **Map results** -- Choropleth of accessibility score by census area, isochrone overlays, bivariate maps (accessibility vs. deprivation).
8. **Identify gaps** -- Areas with low accessibility and high population/deprivation (underserved).
9. **Scenario analysis** -- Test adding a new facility or transit route; compare before/after accessibility.
10. **Report** -- Accessibility maps, statistics per administrative unit, equity analysis, scenario comparison.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Network download | `osmnx` | `osmdata`, `osmextract` | OSM download plugin |
| Network graph | `osmnx`, `networkx` | `sfnetworks`, `dodgr`, `cppRouting` | pgRouting |
| Isochrones | `osmnx`, Valhalla, `openrouteservice` | `osrm`, `opentripplanner` | ORS Tools plugin |
| OD matrix | `r5py`, `pandana`, `openrouteservice` | `r5r`, `accessibility` | OD Cost Matrix (ArcGIS) |
| 2SFCA | `access` package, custom | `SpatialAcc`, `accessibility` | Custom script |
| Population overlay | `geopandas`, `rasterstats` | `sf`, `terra`, `exactextractr` | Zonal Statistics |
| Multi-modal routing | `r5py` (with GTFS) | `r5r` (with GTFS) | OpenTripPlanner |
| Visualization | `matplotlib`, `folium`, `contextily` | `tmap`, `ggplot2`, `mapview` | QGIS Print Layout |

### Code: Isochrone and 2SFCA Analysis in Python

```python
"""
Accessibility Analysis — Isochrones + E2SFCA
Requirements: pip install osmnx networkx geopandas r5py shapely matplotlib
"""
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from shapely.ops import unary_union
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Download and prepare road network
# -------------------------------------------------------------------
PLACE = "Nairobi, Kenya"
NETWORK_TYPE = "drive"  # or "walk", "bike"

G = ox.graph_from_place(PLACE, network_type=NETWORK_TYPE)
G = ox.add_edge_speeds(G)        # infer speeds from OSM highway tags
G = ox.add_edge_travel_times(G)  # compute travel time per edge (seconds)

print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# -------------------------------------------------------------------
# 2. Load facility locations (hospitals)
# -------------------------------------------------------------------
hospitals = gpd.read_file("data/hospitals_nairobi.gpkg")
# Snap each hospital to the nearest network node
hospital_nodes = []
for _, row in hospitals.iterrows():
    nearest = ox.nearest_nodes(G, row.geometry.x, row.geometry.y)
    hospital_nodes.append(nearest)
hospitals["node_id"] = hospital_nodes

# -------------------------------------------------------------------
# 3. Compute isochrones from each facility
# -------------------------------------------------------------------
def isochrone_polygon(G, center_node, trip_time_minutes, edge_weight="travel_time"):
    """Generate isochrone polygon for a given travel time threshold."""
    trip_time_sec = trip_time_minutes * 60
    subgraph = nx.ego_graph(G, center_node, radius=trip_time_sec, distance=edge_weight)
    node_points = [Point(data["x"], data["y"]) for node, data in subgraph.nodes(data=True)]
    return gpd.GeoSeries(node_points).unary_union.convex_hull

thresholds = [5, 10, 15, 30]  # minutes
iso_records = []

for idx, row in hospitals.iterrows():
    for t in thresholds:
        poly = isochrone_polygon(G, row["node_id"], t)
        iso_records.append({
            "hospital_id": row.get("id", idx),
            "hospital_name": row.get("name", f"Hospital {idx}"),
            "minutes": t,
            "geometry": poly
        })

isochrones = gpd.GeoDataFrame(iso_records, crs=hospitals.crs)
isochrones.to_file("output/isochrones.gpkg", driver="GPKG")
print(f"Generated {len(isochrones)} isochrone polygons")

# -------------------------------------------------------------------
# 4. Calculate OD travel time matrix
# -------------------------------------------------------------------
# Population centroids (e.g., census blocks)
pop_centroids = gpd.read_file("data/population_centroids.gpkg")
pop_centroids["node_id"] = [
    ox.nearest_nodes(G, row.geometry.x, row.geometry.y)
    for _, row in pop_centroids.iterrows()
]

# Compute shortest path travel times from all origins to all hospital nodes
# Using Dijkstra from each hospital (reverse graph for many-to-few)
G_rev = G.reverse()  # reverse so we compute FROM hospitals
travel_times = {}  # {(origin_node, hospital_node): seconds}

for h_idx, h_node in enumerate(hospitals["node_id"]):
    lengths = nx.single_source_dijkstra_path_length(
        G_rev, h_node, cutoff=30 * 60, weight="travel_time"
    )
    for o_idx, o_node in enumerate(pop_centroids["node_id"]):
        if o_node in lengths:
            key = (o_idx, h_idx)
            travel_times[key] = lengths[o_node] / 60  # convert to minutes

# Build OD matrix as DataFrame
od_records = []
for (o, h), tt in travel_times.items():
    od_records.append({"origin_idx": o, "hospital_idx": h, "travel_min": tt})
od_df = pd.DataFrame(od_records)

# Nearest facility travel time
nearest_tt = od_df.groupby("origin_idx")["travel_min"].min().reset_index()
nearest_tt.columns = ["origin_idx", "nearest_min"]
pop_centroids = pop_centroids.merge(nearest_tt, left_index=True, right_on="origin_idx", how="left")

# -------------------------------------------------------------------
# 5. Enhanced Two-Step Floating Catchment Area (E2SFCA)
# -------------------------------------------------------------------
CATCHMENT_MIN = 30  # minutes
SUB_ZONES = [0, 10, 20, 30]  # sub-zone boundaries
DECAY_WEIGHTS = [1.0, 0.68, 0.22]  # Gaussian-like decay for each sub-zone

def e2sfca(od_df, pop_centroids, hospitals, sub_zones, weights, catchment):
    """
    Enhanced 2SFCA:
    Step 1: For each facility j, compute supply-to-demand ratio Rj
    Step 2: For each population i, sum Rj of all reachable facilities
    """
    # Step 1: Compute Rj for each hospital
    R = {}
    for h_idx in range(len(hospitals)):
        h_od = od_df[od_df["hospital_idx"] == h_idx].copy()
        weighted_demand = 0
        for k in range(len(sub_zones) - 1):
            zone_mask = (h_od["travel_min"] >= sub_zones[k]) & (h_od["travel_min"] < sub_zones[k + 1])
            origins_in_zone = h_od.loc[zone_mask, "origin_idx"].values
            pop_in_zone = pop_centroids.loc[
                pop_centroids.index.isin(origins_in_zone), "population"
            ].sum()
            weighted_demand += pop_in_zone * weights[k]

        capacity = hospitals.iloc[h_idx].get("beds", 100)  # fallback to 100
        R[h_idx] = capacity / weighted_demand if weighted_demand > 0 else 0

    # Step 2: Compute accessibility Ai for each population location
    A = np.zeros(len(pop_centroids))
    for o_idx in range(len(pop_centroids)):
        o_od = od_df[od_df["origin_idx"] == o_idx]
        for _, row in o_od.iterrows():
            h_idx = int(row["hospital_idx"])
            tt = row["travel_min"]
            # Determine sub-zone weight
            w = 0
            for k in range(len(sub_zones) - 1):
                if sub_zones[k] <= tt < sub_zones[k + 1]:
                    w = weights[k]
                    break
            A[o_idx] += R[h_idx] * w

    return A

pop_centroids["e2sfca_score"] = e2sfca(
    od_df, pop_centroids, hospitals, SUB_ZONES, DECAY_WEIGHTS, CATCHMENT_MIN
)

# -------------------------------------------------------------------
# 6. Identify underserved areas
# -------------------------------------------------------------------
pop_centroids["underserved"] = (
    (pop_centroids["nearest_min"] > 15) &
    (pop_centroids["e2sfca_score"] < pop_centroids["e2sfca_score"].quantile(0.25))
)

n_underserved = pop_centroids["underserved"].sum()
pop_underserved = pop_centroids.loc[pop_centroids["underserved"], "population"].sum()
print(f"\nUnderserved areas: {n_underserved} zones, "
      f"{pop_underserved:,.0f} people (>{15} min + low E2SFCA)")

# -------------------------------------------------------------------
# 7. Scenario analysis — add a new facility
# -------------------------------------------------------------------
# Candidate location for new hospital
candidate = Point(36.82, -1.28)  # lon, lat
candidate_node = ox.nearest_nodes(G, candidate.x, candidate.y)

# Recompute travel times including new facility
new_lengths = nx.single_source_dijkstra_path_length(
    G_rev, candidate_node, cutoff=30 * 60, weight="travel_time"
)
new_od = []
h_new = len(hospitals)
for o_idx, o_node in enumerate(pop_centroids["node_id"]):
    if o_node in new_lengths:
        new_od.append({"origin_idx": o_idx, "hospital_idx": h_new,
                       "travel_min": new_lengths[o_node] / 60})

od_scenario = pd.concat([od_df, pd.DataFrame(new_od)], ignore_index=True)
hospitals_scenario = pd.concat([hospitals, gpd.GeoDataFrame(
    [{"name": "Proposed Hospital", "beds": 150, "geometry": candidate, "node_id": candidate_node}],
    crs=hospitals.crs
)], ignore_index=True)

pop_centroids["e2sfca_scenario"] = e2sfca(
    od_scenario, pop_centroids, hospitals_scenario, SUB_ZONES, DECAY_WEIGHTS, CATCHMENT_MIN
)

improvement = pop_centroids["e2sfca_scenario"] - pop_centroids["e2sfca_score"]
print(f"\nScenario impact: mean E2SFCA improvement = {improvement.mean():.4f}")
print(f"  Max improvement: {improvement.max():.4f}")
print(f"  Areas with >50% improvement: {(improvement > pop_centroids['e2sfca_score'] * 0.5).sum()}")

# Save results
pop_centroids.to_file("output/accessibility_results.gpkg", driver="GPKG")
```

### Code: Multi-Modal Accessibility with r5py

```python
"""
Multi-modal accessibility using r5py (R5 routing engine).
Requires: pip install r5py geopandas
Also requires Java 21+ and will auto-download the R5 JAR.
"""
import r5py
import geopandas as gpd
import pandas as pd
import datetime

# Build transport network (road + GTFS)
transport_network = r5py.TransportNetwork(
    osm_pbf="data/nairobi.osm.pbf",
    gtfs=["data/nairobi_gtfs.zip"]  # public transit schedule
)

# Origins: population centroids
origins = gpd.read_file("data/population_centroids.gpkg")
origins = origins.to_crs("EPSG:4326")

# Destinations: hospitals
destinations = gpd.read_file("data/hospitals_nairobi.gpkg")
destinations = destinations.to_crs("EPSG:4326")

# Compute travel time matrix (transit + walking)
travel_time_matrix = r5py.TravelTimeMatrixComputer(
    transport_network,
    origins=origins,
    destinations=destinations,
    transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
    departure=datetime.datetime(2024, 7, 15, 8, 0),  # Monday 8 AM
    departure_time_window=datetime.timedelta(hours=1),  # average over 1 hour
    max_time=datetime.timedelta(minutes=60),
).compute_travel_times()

print(f"OD pairs computed: {len(travel_time_matrix):,}")
print(travel_time_matrix.head(10))

# Nearest facility by transit
nearest = travel_time_matrix.groupby("from_id")["travel_time"].min().reset_index()
nearest.columns = ["id", "nearest_transit_min"]
origins = origins.merge(nearest, on="id", how="left")

origins.to_file("output/transit_accessibility.gpkg", driver="GPKG")
```

### Expected Output

- **Isochrone maps** per facility (5, 10, 15, 30 minute polygons)
- **Travel time raster** or polygon layer (nearest facility time)
- **E2SFCA accessibility score** per census unit (continuous, higher = better access)
- **Underserved area map** (high population + low accessibility)
- **Scenario comparison** (current vs. proposed, showing improvement)
- **Equity analysis** (accessibility by income quintile / deprivation index)
- **Statistics table**: population within each isochrone, mean/median travel time per district

### Flowchart

```
[Facility Locations]     [Road Network / GTFS]     [Population Data]
        |                        |                        |
        v                        v                        |
  [Geocode/Clean]         [Build Routable Graph            |
                           + Assign Travel Speeds]         |
        |                        |                        |
        +----------+-------------+                        |
                   |                                      |
                   v                                      |
        [Isochrones per Facility]                         |
        [5, 10, 15, 30 min zones]                         |
                   |                                      |
                   v                                      |
        [OD Travel Time Matrix]                           |
        [Origins = pop centroids]                         |
        [Destinations = facilities]                       |
                   |                                      |
                   +------------------+-------------------+
                                      |
                                      v
                       [Accessibility Indicators]
                       [Nearest | Cumulative | E2SFCA]
                                      |
                       +--------------+--------------+
                       |              |              |
                       v              v              v
                  [Gap/Equity    [Scenario       [Bivariate
                   Analysis]     Analysis]        Mapping]
                       |              |              |
                       +--------------+--------------+
                                      |
                                      v
                               [Map & Report]
```

---

## 4. Environmental Impact Assessment

### Objective

Assess the potential environmental impact of a proposed development (e.g., road, dam, mine, wind farm) on the surrounding landscape, habitats, water resources, and communities. Produce a spatially-explicit EIA with quantitative impact matrices, species assessments via GBIF, viewshed analysis, watershed delineation, and cumulative impact scoring.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Project footprint | Area directly affected | Project design documents (CAD/GIS) | Shapefile / DXF |
| Buffer zones | Area indirectly affected | Regulatory guidelines (500m, 1km, 5km) | Generated |
| Land cover / habitat | Habitat loss assessment | Sentinel-2 classification, local habitat map | GeoTIFF / Shapefile |
| Protected areas / sensitive sites | Regulatory compliance | WDPA, Ramsar, IBA, national parks | Shapefile |
| Species occurrence data | Biodiversity impact | GBIF, eBird, iNaturalist, national databases | CSV / API |
| Hydrology (rivers, watersheds) | Water impact | HydroSHEDS, national hydrography, OpenStreetMap | Shapefile / GeoTIFF |
| Air quality / noise baseline | Pollution impact | Monitoring stations, CAMS, noise models | CSV / Raster |
| Population / settlements | Social impact | Census, OpenStreetMap buildings | Shapefile / CSV |
| Soil / geology | Erosion, contamination risk | National soil survey, SoilGrids | Shapefile / GeoTIFF |
| DEM | Visibility, runoff, slope stability | SRTM, LiDAR | GeoTIFF |

**Data Portals:** [WDPA](https://www.protectedplanet.net/) | [GBIF](https://www.gbif.org/) | [HydroSHEDS](https://www.hydrosheds.org/) | [eBird](https://ebird.org/data) | [CAMS](https://atmosphere.copernicus.eu/) | [SoilGrids](https://soilgrids.org/)

> See also: [`data-sources/biodiversity.md`](../data-sources/biodiversity.md), [`data-sources/hydrology.md`](../data-sources/hydrology.md)

### Steps

1. **Define the project footprint** -- Import or digitize the proposed development boundary, including all infrastructure (access roads, staging areas, transmission lines).
2. **Create buffer zones** -- Generate buffers at regulatory distances (e.g., 500 m, 1 km, 5 km). Use dissolve to merge overlapping buffers.
3. **Baseline mapping** -- Map current land cover, habitats, water bodies, settlements within the full impact zone.
4. **Overlay analysis** -- Intersect the project footprint and each buffer zone with every environmental layer. Compute affected areas.
5. **Habitat impact** -- Calculate area of each habitat type directly lost (within footprint) and indirectly disturbed (within buffers). Flag any critical habitats (IUCN Key Biodiversity Areas).
6. **Species impact via GBIF** -- Query species occurrence records within the impact zone, cross-reference IUCN Red List status, flag protected/endangered species.
7. **Hydrological impact** -- Delineate watersheds upstream and downstream of the project, identify affected water bodies, calculate catchment areas.
8. **Visual impact** -- Viewshed analysis from the project site (and key observation points) using the DEM. Calculate zone of theoretical visibility (ZTV).
9. **Noise / air quality impact** -- Buffer-based noise contours, dispersion modeling for air pollutants (if applicable).
10. **Cumulative impact** -- Overlay with existing developments, planned projects, and historical land use change in the area.
11. **Impact scoring matrix** -- Score each impact (magnitude, extent, duration, reversibility, probability) to prioritize.
12. **Mitigation mapping** -- Identify areas for habitat restoration, buffer planting, wildlife corridors, offsets.
13. **Report** -- Impact matrices, maps, quantitative summaries, mitigation recommendations.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Buffer / overlay | `geopandas`, `shapely` | `sf` | QGIS, ArcGIS |
| Habitat mapping | `scikit-learn` + `rasterio` | `terra` + `caret` | QGIS SCP |
| Species query | `pygbif`, `requests` | `rgbif` | GBIF web interface |
| Viewshed | `WhiteboxTools`, `GDAL` | `terra::viewshed()`, `whitebox` | QGIS Sketcher, ArcGIS |
| Watershed delineation | `pysheds`, `WhiteboxTools` | `whitebox`, `terra` | QGIS, ArcGIS Hydrology |
| Noise modeling | `noisemodelling` (Java), custom | Custom | CadnaA, NoiseModelling |
| Impact matrix | `pandas`, `numpy` | `data.frame` | Excel |
| Reporting | `jinja2` + `matplotlib` | R Markdown / Quarto | ArcGIS Layouts |

### Code: EIA Overlay and GBIF Species Query in Python

```python
"""
Environmental Impact Assessment — Buffer Overlay + GBIF Species Query
Requirements: pip install geopandas shapely pygbif whiteboxtools rasterio matplotlib
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
from pygbif import occurrences, species
import rasterio
from whitebox import WhiteboxTools
import matplotlib.pyplot as plt

wbt = WhiteboxTools()
wbt.set_verbose_mode(False)

# -------------------------------------------------------------------
# 1. Load project footprint and create buffer zones
# -------------------------------------------------------------------
project = gpd.read_file("data/project_footprint.gpkg")
project = project.to_crs("EPSG:32637")  # UTM for metre-based buffers

buffer_distances = [500, 1000, 5000]  # metres
buffers = {}
for dist in buffer_distances:
    buffers[dist] = project.buffer(dist).unary_union

# Combined impact zone (outermost buffer)
impact_zone = gpd.GeoDataFrame(
    geometry=[buffers[5000]], crs=project.crs
)

print(f"Project area:     {project.area.sum() / 1e6:.2f} km2")
for dist in buffer_distances:
    area = buffers[dist].area / 1e6
    print(f"Buffer {dist:>5d}m area: {area:.2f} km2")

# -------------------------------------------------------------------
# 2. Habitat impact analysis
# -------------------------------------------------------------------
habitat = gpd.read_file("data/habitat_map.gpkg").to_crs(project.crs)

# Direct impact (within project footprint)
direct = gpd.overlay(habitat, project, how="intersection")
direct["area_ha"] = direct.geometry.area / 10_000

print("\n=== Direct Habitat Impact (within footprint) ===")
direct_summary = direct.groupby("habitat_type")["area_ha"].sum().sort_values(ascending=False)
print(direct_summary.to_string())

# Indirect impact (within buffers, excluding footprint)
for dist in buffer_distances:
    buffer_gdf = gpd.GeoDataFrame(geometry=[buffers[dist]], crs=project.crs)
    ring = gpd.overlay(buffer_gdf, project, how="difference")  # buffer minus footprint
    indirect = gpd.overlay(habitat, ring, how="intersection")
    indirect["area_ha"] = indirect.geometry.area / 10_000
    print(f"\n=== Indirect Impact ({dist}m buffer) ===")
    print(indirect.groupby("habitat_type")["area_ha"].sum().sort_values(ascending=False).to_string())

# -------------------------------------------------------------------
# 3. Species query via GBIF
# -------------------------------------------------------------------
# Convert impact zone to WGS84 for GBIF query
iz_wgs84 = impact_zone.to_crs("EPSG:4326")
bounds = iz_wgs84.total_bounds  # [minx, miny, maxx, maxy]

# Query GBIF for occurrences within the bounding box
gbif_results = occurrences.search(
    decimalLatitude=f"{bounds[1]},{bounds[3]}",
    decimalLongitude=f"{bounds[0]},{bounds[2]}",
    hasCoordinate=True,
    limit=10000
)

records = gbif_results["results"]
if records:
    species_df = pd.DataFrame([{
        "species": r.get("species", "Unknown"),
        "kingdom": r.get("kingdom", ""),
        "iucnRedListCategory": r.get("iucnRedListCategory", "NE"),
        "lat": r.get("decimalLatitude"),
        "lon": r.get("decimalLongitude"),
        "year": r.get("year"),
        "basisOfRecord": r.get("basisOfRecord"),
    } for r in records])

    # Filter to points actually within impact zone
    geometry = [Point(xy) for xy in zip(species_df["lon"], species_df["lat"])]
    species_gdf = gpd.GeoDataFrame(species_df, geometry=geometry, crs="EPSG:4326")
    species_gdf = species_gdf.to_crs(project.crs)
    species_in_zone = gpd.sjoin(species_gdf, impact_zone, predicate="within")

    # Flag threatened species
    threatened_cats = ["CR", "EN", "VU"]
    threatened = species_in_zone[species_in_zone["iucnRedListCategory"].isin(threatened_cats)]

    print(f"\n=== Species within Impact Zone ===")
    print(f"Total occurrence records: {len(species_in_zone):,}")
    print(f"Unique species: {species_in_zone['species'].nunique()}")
    print(f"Threatened species (CR/EN/VU): {threatened['species'].nunique()}")
    if len(threatened) > 0:
        print("\nThreatened species list:")
        for _, row in threatened.drop_duplicates("species").iterrows():
            print(f"  [{row['iucnRedListCategory']}] {row['species']}")

# -------------------------------------------------------------------
# 4. Viewshed analysis (zone of theoretical visibility)
# -------------------------------------------------------------------
wbt.viewshed(
    dem="data/dem_30m.tif",
    stations="data/project_centroid.shp",  # observation point(s)
    output="output/viewshed.tif",
    height=50.0  # structure height in metres (e.g., wind turbine)
)

# Calculate visible area
with rasterio.open("output/viewshed.tif") as src:
    vs = src.read(1)
    pixel_area = abs(src.res[0] * src.res[1])  # m2
    visible_area_km2 = (vs == 1).sum() * pixel_area / 1e6
    print(f"\nViewshed: {visible_area_km2:.2f} km2 visible from project site")

# -------------------------------------------------------------------
# 5. Watershed delineation (downstream impact)
# -------------------------------------------------------------------
wbt.fill_depressions("data/dem_30m.tif", "temp/dem_filled.tif")
wbt.d8_pointer("temp/dem_filled.tif", "temp/d8_pointer.tif")
wbt.d8_flow_accumulation("temp/dem_filled.tif", "temp/flow_acc.tif")

# Delineate watershed from project pour point
wbt.watershed(
    d8_pntr="temp/d8_pointer.tif",
    pour_pts="data/project_pour_point.shp",
    output="output/downstream_watershed.tif"
)

with rasterio.open("output/downstream_watershed.tif") as src:
    ws = src.read(1)
    ws_area_km2 = (ws == 1).sum() * pixel_area / 1e6
    print(f"Downstream watershed area: {ws_area_km2:.2f} km2")

# -------------------------------------------------------------------
# 6. Cumulative impact scoring matrix
# -------------------------------------------------------------------
impact_categories = [
    "Habitat Loss", "Species Disturbance", "Water Quality",
    "Visual Impact", "Noise", "Air Quality", "Social Displacement"
]
scores = {
    "magnitude":    [4, 3, 3, 2, 2, 1, 3],   # 1-5
    "extent_km":    [2, 5, 10, 15, 2, 5, 3],
    "duration_yr":  [50, 20, 50, 50, 5, 5, 10],
    "reversibility": [2, 3, 3, 4, 5, 5, 3],   # 1=irreversible, 5=fully reversible
    "probability":  [5, 4, 3, 3, 5, 2, 4],    # 1-5
}

impact_df = pd.DataFrame(scores, index=impact_categories)
# Composite score: higher = more significant (invert reversibility)
impact_df["composite"] = (
    impact_df["magnitude"] *
    impact_df["probability"] *
    (6 - impact_df["reversibility"])  # invert so irreversible scores higher
)
impact_df = impact_df.sort_values("composite", ascending=False)

print("\n=== Impact Scoring Matrix ===")
print(impact_df.to_string())
print(f"\nHighest impact: {impact_df.index[0]} (score={impact_df['composite'].iloc[0]})")
```

### Expected Output

- **Baseline environment maps** (habitat types, water bodies, settlements within impact zone)
- **Impact zone maps** (footprint + 500 m / 1 km / 5 km buffers)
- **Habitat impact table** (area per type, direct and indirect, by buffer distance)
- **Species impact report** (species list with IUCN status, count of threatened species)
- **Viewshed map** (zone of theoretical visibility from project site)
- **Hydrological impact map** (affected watersheds, downstream areas)
- **Impact scoring matrix** (composite scores ranking all impact categories)
- **Mitigation plan map** (restoration areas, wildlife corridors, offsets)
- **Full EIA report** with maps, tables, quantitative summaries, and recommendations

### Flowchart

```
[Project Footprint (CAD/GIS)]
         |
         v
[Create Buffer Zones: 500m, 1km, 5km]
         |
         +----------+----------+----------+----------+----------+
         |          |          |          |          |          |
         v          v          v          v          v          v
   [Habitat    [Species   [Hydrology  [Visual    [Noise/    [Social
    Impact]     Query]     Analysis]   Impact]    Air Qual]   Impact]
   [overlay]   [GBIF API] [watershed] [viewshed] [buffers]  [overlay]
   [area calc] [IUCN flag] [pour pts] [DEM+ht]   [contours] [pop count]
         |          |          |          |          |          |
         +----------+----------+----------+----------+----------+
                                   |
                                   v
                      [Impact Scoring Matrix]
                      [magnitude x probability x (6-reversibility)]
                                   |
                                   v
                      [Cumulative Assessment]
                      [existing + planned developments]
                                   |
                                   v
                      [Mitigation Planning]
                      [restoration, corridors, offsets]
                                   |
                                   v
                           [EIA Report]
```

---

## 5. Urban Heat Island Analysis

### Objective

Map and analyze the urban heat island (UHI) effect by comparing land surface temperatures (LST) across urban and rural areas. Identify land cover and urban form factors that contribute to elevated temperatures, locate statistically significant thermal hot spots using Getis-Ord Gi*, and produce vulnerability maps overlaying heat exposure with socioeconomic indicators.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Thermal imagery | Land surface temperature | Landsat 8/9 Band 10, MODIS LST, ECOSTRESS | GeoTIFF, L2 |
| Multispectral imagery | NDVI, NDBI, albedo | Sentinel-2, Landsat | L2A surface reflectance |
| Land cover map | Correlate LST with land cover | CORINE, NLCD, ESA WorldCover, local classification | GeoTIFF |
| Building footprints | Urban density, impervious fraction | OpenStreetMap, Microsoft/Google footprints | GeoJSON |
| Street trees / green space | Cooling effect analysis | Local urban forestry, OpenStreetMap | Shapefile |
| DEM | Elevation correction | SRTM, LiDAR | GeoTIFF |
| Weather station data | Validation, air temperature reference | NOAA ISD, national met service | CSV |
| Census / socioeconomic data | Vulnerability mapping | Census, deprivation indices | Shapefile / CSV |

**Data Portals:** [USGS EarthExplorer (Landsat)](https://earthexplorer.usgs.gov/) | [ECOSTRESS](https://ecostress.jpl.nasa.gov/) | [ESA WorldCover](https://worldcover2021.esa.int/) | [Microsoft Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | [NOAA ISD](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database)

> See also: [`data-sources/satellite-imagery.md`](../data-sources/satellite-imagery.md), [`visualization/thematic-maps.md`](../visualization/thematic-maps.md)

### Steps

1. **Acquire thermal imagery** -- Select cloud-free summer daytime (and optionally nighttime) scenes. Landsat 8/9 provides 100 m thermal data resampled to 30 m.
2. **Calculate LST** -- Convert thermal band DN to at-sensor radiance, to brightness temperature, then to LST using NDVI-based emissivity correction.
3. **Calculate urban indices** -- NDVI (vegetation), NDBI (built-up), MNDWI (water), ISA (impervious surface area), albedo.
4. **Define urban-rural zones** -- Delineate urban core, suburban ring, and rural reference areas (using land cover or morphological criteria).
5. **UHI intensity** -- Calculate the LST difference between urban and rural zones (mean UHI intensity, by zone, by pixel).
6. **Correlation analysis** -- Correlate LST with NDVI, NDBI, ISA, building density, distance to green space (OLS regression and spatial regression).
7. **Hot spot analysis (Gi*)** -- Apply Getis-Ord Gi* statistic to identify statistically significant thermal hot/cold spots at census-unit or grid level.
8. **Vulnerability mapping** -- Overlay thermal hot spots with vulnerable populations (elderly, children, low-income, chronic disease, no A/C).
9. **Cooling potential** -- Model the LST reduction from adding green space, cool roofs, or increasing albedo using empirical coefficients.
10. **Report** -- LST maps, UHI intensity, correlation plots, hot spot maps, vulnerability maps, cooling scenario results.

### Tools

| Step | Python | R | Desktop GIS / Cloud |
|---|---|---|---|
| LST calculation | `rasterio`, `numpy` | `terra` | QGIS Raster Calculator, GEE |
| Index calculation | `rasterio`, `numpy` | `terra` | QGIS Raster Calculator |
| Hot spot analysis | `pysal` (`esda.Moran`, `esda.G_Local`) | `spdep` (`localG`) | ArcGIS Hot Spot, GeoDa |
| Correlation/regression | `scipy`, `statsmodels`, `pysal.spreg` | `stats`, `spatialreg`, `spdep` | GeoDa |
| Visualization | `matplotlib`, `seaborn`, `folium` | `tmap`, `ggplot2` | QGIS, ArcGIS |
| Vulnerability overlay | `geopandas`, `rasterstats` | `sf`, `exactextractr` | QGIS |

### Code: LST Calculation and UHI Analysis in Python

```python
"""
Urban Heat Island Analysis — LST from Landsat + Gi* Hot Spots
Requirements: pip install rasterio numpy scipy geopandas pysal libpysal esda matplotlib seaborn
"""
import numpy as np
import rasterio
import geopandas as gpd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# 1. Calculate Land Surface Temperature from Landsat 8/9
# -------------------------------------------------------------------
def landsat_lst(thermal_path, ndvi_path, output_path=None):
    """
    Calculate LST from Landsat 8/9 Band 10 (Level 2 surface temperature product).
    For Level 1: requires DN -> radiance -> brightness temp -> emissivity correction.
    This version uses the NDVI-based emissivity method.
    """
    # Load thermal band (Band 10, already in Kelvin*10 for L2, or DN for L1)
    with rasterio.open(thermal_path) as src:
        thermal = src.read(1).astype(np.float64)
        profile = src.profile.copy()

    with rasterio.open(ndvi_path) as src:
        ndvi = src.read(1).astype(np.float64)

    # --- For Level 1 data: DN to radiance to brightness temperature ---
    # Radiance: L = ML * DN + AL  (from metadata)
    ML = 3.3420E-04  # RADIANCE_MULT_BAND_10
    AL = 0.10000     # RADIANCE_ADD_BAND_10
    radiance = ML * thermal + AL

    # Brightness temperature (Kelvin)
    K1 = 774.8853    # K1_CONSTANT_BAND_10
    K2 = 1321.0789   # K2_CONSTANT_BAND_10
    BT = K2 / np.log((K1 / radiance) + 1)

    # --- Emissivity from NDVI (Sobrino et al., 2004) ---
    # Proportion of vegetation (Pv)
    ndvi_soil = 0.2
    ndvi_veg = 0.5
    Pv = np.clip((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil), 0, 1) ** 2

    # Emissivity
    emissivity = np.where(
        ndvi < ndvi_soil,
        0.973,                          # bare soil
        np.where(
            ndvi > ndvi_veg,
            0.99,                       # full vegetation
            0.004 * Pv + 0.986          # mixed
        )
    )

    # --- LST (Artis & Carnahan, 1982) ---
    wavelength = 10.9e-6  # Band 10 central wavelength (metres)
    rho = 1.438e-2        # h*c/sigma (m*K)
    LST = BT / (1 + (wavelength * BT / rho) * np.log(emissivity))

    # Convert to Celsius
    LST_C = LST - 273.15

    # Mask invalid pixels
    LST_C[thermal == 0] = np.nan

    if output_path:
        out_profile = profile.copy()
        out_profile.update(dtype="float64", nodata=np.nan)
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(LST_C, 1)
        print(f"LST saved to {output_path}")

    return LST_C, profile

LST, profile = landsat_lst(
    "data/landsat8_B10.tif",
    "data/ndvi.tif",
    "output/lst_celsius.tif"
)

print(f"LST range: {np.nanmin(LST):.1f} - {np.nanmax(LST):.1f} C")
print(f"LST mean:  {np.nanmean(LST):.1f} C")

# -------------------------------------------------------------------
# 2. Calculate spectral indices
# -------------------------------------------------------------------
def calc_index(band_a_path, band_b_path):
    """Normalized difference: (A - B) / (A + B)."""
    with rasterio.open(band_a_path) as src:
        a = src.read(1).astype(np.float64)
    with rasterio.open(band_b_path) as src:
        b = src.read(1).astype(np.float64)
    return (a - b) / (a + b + 1e-10)

ndvi = calc_index("data/landsat8_B5.tif", "data/landsat8_B4.tif")  # NIR - Red
ndbi = calc_index("data/landsat8_B6.tif", "data/landsat8_B5.tif")  # SWIR1 - NIR

# -------------------------------------------------------------------
# 3. UHI intensity by zone
# -------------------------------------------------------------------
zones = gpd.read_file("data/urban_rural_zones.gpkg")  # columns: zone_type (urban/suburban/rural)
# Zonal statistics using rasterstats
from rasterstats import zonal_stats

zone_stats = zonal_stats(zones, "output/lst_celsius.tif", stats=["mean", "median", "std", "count"])
for i, row in zones.iterrows():
    zones.loc[i, "lst_mean"] = zone_stats[i]["mean"]
    zones.loc[i, "lst_median"] = zone_stats[i]["median"]
    zones.loc[i, "lst_std"] = zone_stats[i]["std"]

# UHI intensity = urban mean LST - rural mean LST
urban_mean = zones.loc[zones["zone_type"] == "urban", "lst_mean"].values[0]
rural_mean = zones.loc[zones["zone_type"] == "rural", "lst_mean"].values[0]
uhi_intensity = urban_mean - rural_mean
print(f"\nUHI Intensity: {uhi_intensity:.2f} C (urban - rural)")
print(f"  Urban mean LST:  {urban_mean:.2f} C")
print(f"  Rural mean LST:  {rural_mean:.2f} C")

# -------------------------------------------------------------------
# 4. Correlation analysis: LST vs NDVI, NDBI
# -------------------------------------------------------------------
# Flatten and sample (avoid memory issues)
mask = ~np.isnan(LST) & ~np.isnan(ndvi) & ~np.isnan(ndbi)
n_sample = min(50000, mask.sum())
idx = np.random.choice(np.where(mask.ravel())[0], size=n_sample, replace=False)

lst_sample = LST.ravel()[idx]
ndvi_sample = ndvi.ravel()[idx]
ndbi_sample = ndbi.ravel()[idx]

# Pearson correlation
r_ndvi, p_ndvi = stats.pearsonr(lst_sample, ndvi_sample)
r_ndbi, p_ndbi = stats.pearsonr(lst_sample, ndbi_sample)

print(f"\nCorrelation: LST vs NDVI: r={r_ndvi:.3f} (p={p_ndvi:.2e})")
print(f"Correlation: LST vs NDBI: r={r_ndbi:.3f} (p={p_ndbi:.2e})")

# Scatter plots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(ndvi_sample, lst_sample, alpha=0.05, s=1)
axes[0].set_xlabel("NDVI"); axes[0].set_ylabel("LST (C)")
axes[0].set_title(f"LST vs NDVI (r={r_ndvi:.3f})")
axes[1].scatter(ndbi_sample, lst_sample, alpha=0.05, s=1, color="red")
axes[1].set_xlabel("NDBI"); axes[1].set_ylabel("LST (C)")
axes[1].set_title(f"LST vs NDBI (r={r_ndbi:.3f})")
plt.tight_layout()
plt.savefig("output/lst_correlation.png", dpi=150)

# -------------------------------------------------------------------
# 5. Hot spot analysis: Getis-Ord Gi*
# -------------------------------------------------------------------
from libpysal.weights import Queen
from esda.getisord import G_Local

# Aggregate LST to census tracts
tracts = gpd.read_file("data/census_tracts.gpkg")
tract_stats = zonal_stats(tracts, "output/lst_celsius.tif", stats=["mean"])
tracts["lst_mean"] = [s["mean"] for s in tract_stats]
tracts = tracts.dropna(subset=["lst_mean"])

# Spatial weights (Queen contiguity)
w = Queen.from_dataframe(tracts, use_index=True)
w.transform = "R"  # row-standardize

# Gi* statistic
gi_star = G_Local(tracts["lst_mean"].values, w, star=True, permutations=999)

tracts["gi_z"] = gi_star.Zs
tracts["gi_p"] = gi_star.p_sim

# Classify hot/cold spots
tracts["hotspot_class"] = "Not Significant"
tracts.loc[(tracts["gi_z"] > 1.96) & (tracts["gi_p"] < 0.05), "hotspot_class"] = "Hot Spot (95%)"
tracts.loc[(tracts["gi_z"] > 2.58) & (tracts["gi_p"] < 0.01), "hotspot_class"] = "Hot Spot (99%)"
tracts.loc[(tracts["gi_z"] < -1.96) & (tracts["gi_p"] < 0.05), "hotspot_class"] = "Cold Spot (95%)"
tracts.loc[(tracts["gi_z"] < -2.58) & (tracts["gi_p"] < 0.01), "hotspot_class"] = "Cold Spot (99%)"

print("\n=== Gi* Hot Spot Classification ===")
print(tracts["hotspot_class"].value_counts().to_string())

tracts.to_file("output/lst_hotspots.gpkg", driver="GPKG")

# -------------------------------------------------------------------
# 6. Vulnerability mapping (bivariate: heat x deprivation)
# -------------------------------------------------------------------
# Merge socioeconomic data
socio = gpd.read_file("data/socioeconomic.gpkg")  # fields: tract_id, deprivation_index, pct_elderly
tracts = tracts.merge(socio[["tract_id", "deprivation_index", "pct_elderly"]], on="tract_id")

# Bivariate classification: high heat + high deprivation = most vulnerable
tracts["heat_class"] = pd.qcut(tracts["lst_mean"], q=3, labels=["Low", "Med", "High"])
tracts["deprivation_class"] = pd.qcut(tracts["deprivation_index"], q=3, labels=["Low", "Med", "High"])
tracts["vulnerability"] = tracts["heat_class"].astype(str) + " / " + tracts["deprivation_class"].astype(str)

most_vulnerable = tracts[(tracts["heat_class"] == "High") & (tracts["deprivation_class"] == "High")]
print(f"\nMost vulnerable tracts: {len(most_vulnerable)} "
      f"({len(most_vulnerable)/len(tracts)*100:.1f}% of all tracts)")

tracts.to_file("output/vulnerability_map.gpkg", driver="GPKG")

# -------------------------------------------------------------------
# 7. Cooling potential estimation
# -------------------------------------------------------------------
# Empirical: 10% increase in NDVI -> ~1.5 C reduction in LST (varies by city)
# Reference: Estoque et al. (2017), Sci Total Environ
COOLING_COEFF = -15.0  # dLST/dNDVI (C per unit NDVI) from regression slope

# Scenario: plant trees in hot spot areas (increase NDVI by 0.1)
hotspot_tracts = tracts[tracts["hotspot_class"].str.contains("Hot Spot")]
ndvi_increase = 0.10
cooling_effect = ndvi_increase * COOLING_COEFF

print(f"\n=== Cooling Scenario ===")
print(f"Target: {len(hotspot_tracts)} hot spot tracts")
print(f"NDVI increase: +{ndvi_increase:.2f}")
print(f"Estimated LST reduction: {cooling_effect:.1f} C")
print(f"Post-intervention mean LST: {hotspot_tracts['lst_mean'].mean() + cooling_effect:.1f} C")
```

### Expected Output

- **Land surface temperature map** (continuous LST in Celsius, GeoTIFF)
- **UHI intensity map** (urban-rural LST difference by zone)
- **Correlation scatter plots** (LST vs. NDVI, LST vs. NDBI with regression lines)
- **Thermal hot spot map** (Gi* classified: hot spot 99%, hot spot 95%, not significant, cold spot)
- **Vulnerability map** (bivariate: heat exposure x socioeconomic deprivation)
- **Cooling scenario results** (estimated LST reduction from greening interventions)
- **Statistics summary**: UHI intensity, correlation coefficients, count of hot/cold spot tracts
- **Analysis report** with maps, charts, statistics, and policy recommendations

### Flowchart

```
[Thermal Imagery (Band 10)]        [Multispectral Imagery]        [Census / Socioeconomic]
         |                                  |                              |
         v                                  v                              |
[DN -> Radiance -> BT]              [Calculate NDVI, NDBI,                |
[Emissivity from NDVI]               MNDWI, Albedo, ISA]                 |
[BT -> LST (Celsius)]                      |                              |
         |                                  |                              |
         +-----------------+----------------+                              |
                           |                                               |
                           v                                               |
                [Urban-Rural Zone Definition]                              |
                [Zonal LST Statistics]                                     |
                           |                                               |
                           v                                               |
                [UHI Intensity = LST_urban - LST_rural]                   |
                           |                                               |
              +------------+------------+                                  |
              |            |            |                                  |
              v            v            v                                  |
       [Correlation  [Gi* Hot Spot   [Aggregate LST                       |
        Analysis]     Analysis]       to Census Tracts]                   |
       [OLS + Spatial [Queen weights] [Zonal stats]                       |
        Regression]   [p<0.05 flag]         |                             |
              |            |                +-----------------------------+
              |            |                |
              |            |                v
              |            |        [Vulnerability Mapping]
              |            |        [Bivariate: Heat x Deprivation]
              |            |                |
              +------------+----------------+
                           |
                           v
                [Cooling Potential Scenarios]
                [NDVI increase -> LST reduction]
                           |
                           v
                    [Report & Maps]
```

---

## 6. Flood Risk Assessment

### Objective

Assess flood risk by combining hydrological modeling (flow accumulation, flood inundation extent) with exposure and vulnerability analysis. The workflow covers DEM-based terrain preprocessing, rainfall-runoff estimation, simplified inundation mapping, and risk classification by overlaying flood hazard with population and asset exposure.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| DEM (high-resolution preferred) | Flow direction, accumulation, floodplain delineation | LiDAR, SRTM, Copernicus DEM, ALOS | GeoTIFF, 5-30m |
| River network | Stream channels, order, connectivity | HydroSHEDS, OpenStreetMap, national hydrography | Shapefile |
| Rainfall / precipitation | Extreme event inputs (IDF curves, design storms) | NOAA Atlas 14, ERA5, CHIRPS, national met office | CSV / NetCDF |
| Land cover | Surface roughness (Manning's n) | ESA WorldCover, NLCD, CORINE | GeoTIFF |
| Soil / infiltration data | Runoff coefficient, curve number | SoilGrids, SSURGO, national soil survey | Shapefile / GeoTIFF |
| Historical flood extents | Calibration / validation | Sentinel-1 SAR flood maps, FEMA floodplains, JRC Global Surface Water | Shapefile / GeoTIFF |
| Building footprints | Exposure: buildings at risk | OpenStreetMap, Microsoft footprints | GeoJSON |
| Population data | Exposure: people at risk | WorldPop, census, GHSL | GeoTIFF / Shapefile |
| Critical infrastructure | Exposure: hospitals, schools, power | Government data, OpenStreetMap | Shapefile |

**Data Portals:** [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) | [HydroSHEDS](https://www.hydrosheds.org/) | [ERA5 (CDS)](https://cds.climate.copernicus.eu/) | [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | [JRC Global Surface Water](https://global-surface-water.appspot.com/) | [NOAA Atlas 14](https://hdsc.nws.noaa.gov/pfds/) | [FEMA Flood Maps](https://www.fema.gov/flood-maps)

> See also: [`data-sources/elevation.md`](../data-sources/elevation.md), [`data-sources/hydrology.md`](../data-sources/hydrology.md)

### Steps

1. **DEM preprocessing** -- Fill sinks/depressions, breach channels (prefer breaching over filling for LiDAR). Validate against known stream locations.
2. **Flow direction and accumulation** -- Compute D8 flow direction, flow accumulation, and extract stream network from accumulation threshold.
3. **Watershed delineation** -- Define catchments for the study area at pour points (gauge stations or confluences).
4. **Design storm** -- Select return period (e.g., 100-year, 500-year), obtain rainfall depth/intensity from IDF curves or ERA5 extremes.
5. **Runoff estimation** -- Apply SCS Curve Number method or rational method using land cover and soil data to estimate peak discharge.
6. **Flood inundation mapping** -- Use HAND (Height Above Nearest Drainage) for rapid inundation estimation, or couple with HEC-RAS / LISFLOOD-FP for detailed 1D/2D modeling.
7. **Flood hazard classification** -- Classify inundation into hazard zones by depth (e.g., <0.5 m low, 0.5-1.5 m moderate, >1.5 m high) and velocity if available.
8. **Exposure analysis** -- Overlay flood hazard with buildings, population, critical infrastructure. Count/sum exposed assets per hazard zone.
9. **Vulnerability scoring** -- Assign vulnerability weights based on building type, population age structure, infrastructure criticality.
10. **Risk mapping** -- Combine hazard, exposure, and vulnerability into a composite risk index. Classify into risk zones.
11. **Scenario comparison** -- Compare multiple return periods (10yr, 50yr, 100yr, 500yr) and/or climate change projections.
12. **Report** -- Hazard maps, exposure tables, risk maps, scenario comparison, recommendations for flood risk management.

### Tools

| Step | Python | R | Desktop GIS / Modeling |
|---|---|---|---|
| DEM preprocessing | `WhiteboxTools`, `richdem`, `pysheds` | `whitebox`, `terra` | QGIS, SAGA, TauDEM |
| Flow analysis | `WhiteboxTools`, `pysheds` | `whitebox`, `terra` | QGIS, ArcGIS Hydrology |
| HAND calculation | `WhiteboxTools` (`elevation_above_stream`) | `whitebox` | Custom script |
| Runoff estimation | `numpy`, custom SCS-CN | Custom | HEC-HMS |
| 2D flood modeling | `LISFLOOD-FP` (subprocess), `Anuga` | `FloodR` | HEC-RAS, LISFLOOD-FP, TUFLOW |
| Exposure overlay | `geopandas`, `rasterstats` | `sf`, `exactextractr` | QGIS, ArcGIS |
| Visualization | `matplotlib`, `folium`, `contextily` | `tmap`, `ggplot2` | QGIS, ArcGIS |

### Code: DEM Hydrology and HAND-Based Flood Mapping in Python

```python
"""
Flood Risk Assessment — DEM Processing + HAND Inundation + Exposure Analysis
Requirements: pip install whiteboxtools rasterio numpy geopandas rasterstats matplotlib
"""
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from rasterstats import zonal_stats
from whitebox import WhiteboxTools
import matplotlib.pyplot as plt

wbt = WhiteboxTools()
wbt.set_verbose_mode(True)

# -------------------------------------------------------------------
# 1. DEM preprocessing — breach depressions (preferred for LiDAR)
# -------------------------------------------------------------------
wbt.breach_depressions_least_cost(
    dem="data/dem_5m.tif",
    output="temp/dem_breached.tif",
    dist=50,       # max breach distance (pixels)
    fill=True      # fill remaining depressions after breaching
)

# -------------------------------------------------------------------
# 2. Flow direction and accumulation
# -------------------------------------------------------------------
wbt.d8_pointer(
    dem="temp/dem_breached.tif",
    output="temp/d8_pointer.tif"
)

wbt.d8_flow_accumulation(
    i="temp/dem_breached.tif",
    output="temp/flow_acc.tif",
    out_type="cells"
)

# -------------------------------------------------------------------
# 3. Extract stream network (threshold = 10,000 cells for 5m DEM)
# -------------------------------------------------------------------
wbt.extract_streams(
    flow_accum="temp/flow_acc.tif",
    output="temp/streams.tif",
    threshold=10000
)

# Raster streams to vector
wbt.raster_streams_to_vector(
    streams="temp/streams.tif",
    d8_pntr="temp/d8_pointer.tif",
    output="output/streams.shp"
)

# -------------------------------------------------------------------
# 4. Watershed delineation
# -------------------------------------------------------------------
wbt.watershed(
    d8_pntr="temp/d8_pointer.tif",
    pour_pts="data/pour_points.shp",
    output="temp/watersheds.tif"
)

# -------------------------------------------------------------------
# 5. HAND (Height Above Nearest Drainage)
# -------------------------------------------------------------------
wbt.elevation_above_stream(
    dem="temp/dem_breached.tif",
    streams="temp/streams.tif",
    output="output/hand.tif"
)

with rasterio.open("output/hand.tif") as src:
    hand = src.read(1).astype(np.float64)
    profile = src.profile.copy()
    print(f"HAND range: {np.nanmin(hand):.1f} - {np.nanmax(hand):.1f} m")

# -------------------------------------------------------------------
# 6. Flood inundation mapping using HAND threshold
# -------------------------------------------------------------------
# Relate flood stage (water level above stream) to return period
# These values come from hydrological modeling or gauge data
flood_scenarios = {
    "10yr":  2.0,   # metres above stream
    "50yr":  3.5,
    "100yr": 4.5,
    "500yr": 6.0,
}

for scenario, stage in flood_scenarios.items():
    inundation = np.where(hand <= stage, hand, np.nan)  # depth = stage - HAND
    depth = np.where(hand <= stage, stage - hand, np.nan)

    # Classify hazard by depth
    hazard = np.full_like(depth, np.nan)
    hazard[(depth >= 0) & (depth < 0.5)]   = 1  # Low
    hazard[(depth >= 0.5) & (depth < 1.5)] = 2  # Moderate
    hazard[(depth >= 1.5) & (depth < 3.0)] = 3  # High
    hazard[depth >= 3.0]                    = 4  # Very High

    # Save depth raster
    out_profile = profile.copy()
    out_profile.update(dtype="float64", nodata=np.nan)
    with rasterio.open(f"output/flood_depth_{scenario}.tif", "w", **out_profile) as dst:
        dst.write(depth, 1)
    with rasterio.open(f"output/flood_hazard_{scenario}.tif", "w", **out_profile) as dst:
        dst.write(hazard, 1)

    # Area statistics
    pixel_area_m2 = abs(profile["transform"].a * profile["transform"].e)
    for level, name in [(1, "Low"), (2, "Moderate"), (3, "High"), (4, "Very High")]:
        count = np.nansum(hazard == level)
        area_km2 = count * pixel_area_m2 / 1e6
        print(f"  {scenario} - {name:>10s}: {area_km2:>8.2f} km2")

# -------------------------------------------------------------------
# 7. SCS Curve Number runoff estimation
# -------------------------------------------------------------------
def scs_runoff(P_mm, CN):
    """
    SCS Curve Number method for rainfall-runoff estimation.
    P_mm: total rainfall depth (mm)
    CN: Curve Number (0-100)
    Returns: direct runoff depth (mm)
    """
    S = 25400 / CN - 254  # potential maximum retention (mm)
    Ia = 0.2 * S           # initial abstraction (mm)
    Q = np.where(P_mm > Ia, (P_mm - Ia)**2 / (P_mm - Ia + S), 0)
    return Q

# Load curve number raster (from land cover + soil hydrological group)
with rasterio.open("data/curve_number.tif") as src:
    cn_arr = src.read(1).astype(np.float64)
    cn_profile = src.profile.copy()

# Design storm: 100-year, 24-hour rainfall = 150 mm (from IDF curves)
P_100yr = 150.0  # mm
Q_100yr = scs_runoff(P_100yr, cn_arr)

print(f"\n=== SCS-CN Runoff Estimation (100-yr storm) ===")
print(f"Rainfall: {P_100yr:.0f} mm")
print(f"Mean CN: {np.nanmean(cn_arr):.1f}")
print(f"Mean runoff: {np.nanmean(Q_100yr):.1f} mm")
print(f"Runoff coefficient: {np.nanmean(Q_100yr)/P_100yr:.2f}")

# -------------------------------------------------------------------
# 8. Exposure analysis — buildings and population in flood zone
# -------------------------------------------------------------------
buildings = gpd.read_file("data/building_footprints.gpkg")
buildings = buildings.to_crs(profile["crs"])

population = gpd.read_file("data/population_grid.gpkg")
population = population.to_crs(profile["crs"])

print("\n=== Exposure Analysis ===")
for scenario, stage in flood_scenarios.items():
    # Create flood extent polygon
    with rasterio.open(f"output/flood_depth_{scenario}.tif") as src:
        depth_arr = src.read(1)
        transform = src.transform
        crs = src.crs

    flood_mask = (~np.isnan(depth_arr)).astype(np.uint8)
    from rasterio.features import shapes
    flood_polys = [
        shape for shape, val in shapes(flood_mask, transform=transform) if val == 1
    ]
    if flood_polys:
        from shapely.geometry import shape as shapely_shape
        from shapely.ops import unary_union
        flood_extent = gpd.GeoDataFrame(
            geometry=[unary_union([shapely_shape(p) for p in flood_polys])],
            crs=crs
        )

        # Buildings in flood zone
        buildings_flooded = gpd.sjoin(buildings, flood_extent, predicate="intersects")
        n_buildings = len(buildings_flooded)

        # Population in flood zone (zonal sum)
        pop_flooded = gpd.sjoin(population, flood_extent, predicate="intersects")
        pop_total = pop_flooded["population"].sum() if "population" in pop_flooded.columns else 0

        print(f"  {scenario}: {n_buildings:>6,} buildings | {pop_total:>10,.0f} people exposed")

# -------------------------------------------------------------------
# 9. Risk classification (Hazard x Exposure x Vulnerability)
# -------------------------------------------------------------------
# Simplified grid-based risk: aggregate to hex grid or census units
admin_units = gpd.read_file("data/admin_units.gpkg").to_crs(profile["crs"])

# Hazard score: mean flood depth in 100yr scenario
hazard_stats = zonal_stats(
    admin_units, "output/flood_depth_100yr.tif",
    stats=["mean", "max", "count"],
    nodata=np.nan
)
admin_units["hazard_mean_depth"] = [s["mean"] if s["mean"] else 0 for s in hazard_stats]
admin_units["hazard_score"] = np.clip(admin_units["hazard_mean_depth"] / 3.0, 0, 1)

# Exposure score: normalized building count
building_counts = gpd.sjoin(buildings, admin_units, predicate="within").groupby("index_right").size()
admin_units["building_count"] = admin_units.index.map(building_counts).fillna(0)
admin_units["exposure_score"] = admin_units["building_count"] / admin_units["building_count"].max()

# Risk = Hazard x Exposure (simplified)
admin_units["risk_score"] = admin_units["hazard_score"] * admin_units["exposure_score"]
admin_units["risk_class"] = pd.cut(
    admin_units["risk_score"],
    bins=[0, 0.1, 0.3, 0.6, 1.0],
    labels=["Low", "Moderate", "High", "Very High"],
    include_lowest=True
)

print("\n=== Risk Classification (100yr scenario) ===")
print(admin_units["risk_class"].value_counts().sort_index().to_string())

admin_units.to_file("output/flood_risk_map.gpkg", driver="GPKG")
```

### Expected Output

- **HAND raster** (height above nearest drainage, continuous metres)
- **Flood inundation maps** per return period (10yr, 50yr, 100yr, 500yr depth rasters)
- **Flood hazard classification** (Low / Moderate / High / Very High by depth)
- **Stream network** (extracted from DEM, vector format)
- **Watershed boundaries** (delineated catchments)
- **Exposure summary** (buildings and population per scenario)
- **Risk map** (composite hazard x exposure, classified by admin unit)
- **Scenario comparison table** (area flooded, people exposed per return period)
- **Report** with maps, tables, and flood risk management recommendations

### Flowchart

```
[DEM (5-30m)]
      |
      v
[Breach/Fill Depressions]
      |
      v
[D8 Flow Direction] ---> [Flow Accumulation] ---> [Extract Streams]
      |                                                    |
      v                                                    v
[Watershed Delineation]                          [HAND Calculation]
      |                                          [Elev Above Stream]
      |                                                    |
      v                                                    v
[SCS Curve Number]                            [Flood Stage Thresholds]
[Rainfall-Runoff]                             [10yr, 50yr, 100yr, 500yr]
[Peak Discharge]                                           |
      |                                                    v
      +------------------------------------>  [Inundation Depth Maps]
                                                           |
                                                           v
                                              [Hazard Classification]
                                              [by depth & velocity]
                                                           |
                                    +-----------+----------+-----------+
                                    |           |                      |
                                    v           v                      v
                              [Building    [Population          [Infrastructure
                               Exposure]    Exposure]            Exposure]
                                    |           |                      |
                                    +-----------+----------+-----------+
                                                |
                                                v
                                    [Risk = Hazard x Exposure]
                                    [Classify per admin unit]
                                                |
                                                v
                                    [Scenario Comparison & Report]
```

---

## 7. Agricultural Monitoring

### Objective

Monitor crop growth, classify agricultural land use, estimate yield, and detect drought conditions using satellite-derived vegetation indices time series. The workflow combines NDVI/EVI temporal profiles, supervised crop type classification, yield regression modeling, and drought indices (SPI, VCI) for precision agriculture and food security applications.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Satellite imagery (time series) | Vegetation index time series | Sentinel-2 (10 m), MODIS (250 m), Landsat | Multi-temporal GeoTIFF |
| Crop type ground truth | Training/validation for classification | Government crop surveys, farm records | GeoPackage / Shapefile |
| Field boundaries | Parcel-level analysis | National cadaster, LPIS, digitized boundaries | Shapefile |
| Weather / climate data | Temperature, precipitation for yield models | ERA5, CHIRPS, local met stations | CSV / NetCDF |
| Yield statistics | Calibration of yield models | National agricultural statistics, FAOSTAT | CSV |
| Soil data | Soil quality, water-holding capacity | SoilGrids, SSURGO, national survey | GeoTIFF / Shapefile |
| Irrigation data (optional) | Irrigated vs. rainfed classification | Government records, MODIS irrigation maps | Shapefile / GeoTIFF |

**Data Portals:** [Copernicus Open Access Hub (Sentinel-2)](https://scihub.copernicus.eu/) | [MODIS (LP DAAC)](https://lpdaac.usgs.gov/) | [CHIRPS](https://www.chc.ucsb.edu/data/chirps) | [ERA5 (CDS)](https://cds.climate.copernicus.eu/) | [FAOSTAT](https://www.fao.org/faostat/) | [SoilGrids](https://soilgrids.org/) | [CropScape (USDA)](https://nassgeodata.gmu.edu/CropScape/)

> See also: [`data-sources/satellite-imagery.md`](../data-sources/satellite-imagery.md), [`data-sources/climate.md`](../data-sources/climate.md)

### Steps

1. **Image collection** -- Assemble dense time series (every 5-10 days) covering the full growing season. Cloud-mask and gap-fill.
2. **Vegetation index time series** -- Compute NDVI, EVI, SAVI for each date. Create pixel-level temporal profiles.
3. **Time series smoothing** -- Apply Savitzky-Golay filter or HANTS (Harmonic ANalysis of Time Series) to remove noise and fill gaps.
4. **Phenological metrics** -- Extract start/end/peak of season (SOS, EOS, POS), maximum NDVI, integrated NDVI, growing season length from smoothed profiles.
5. **Crop type classification** -- Use phenological features + spectral bands as inputs to Random Forest / temporal CNN classifier. Validate with independent test set.
6. **Yield estimation** -- Build regression model (NDVI_max or integrated NDVI vs. reported yield at district level). Validate with leave-one-year-out cross-validation.
7. **Drought monitoring -- VCI** -- Vegetation Condition Index: VCI = (NDVI - NDVI_min) / (NDVI_max - NDVI_min) * 100, using long-term min/max per pixel per dekad.
8. **Drought monitoring -- SPI** -- Standardized Precipitation Index from rainfall time series (gamma distribution fitting). Classify drought severity.
9. **Anomaly detection** -- Compare current season NDVI to historical mean. Flag pixels with z-score < -2 as severe negative anomaly.
10. **Precision agriculture** -- Within-field variability mapping using coefficient of variation of NDVI, management zone delineation (k-means clustering).
11. **Report** -- Crop maps, yield estimates, drought severity maps, anomaly alerts, management zone maps.

### Tools

| Step | Python | R | Desktop GIS / Cloud |
|---|---|---|---|
| Image collection | `pystac-client`, `planetary_computer`, GEE API | `rstac`, `gdalcubes` | Google Earth Engine |
| NDVI time series | `numpy`, `xarray`, `rasterio` | `terra`, `sits` | GEE |
| Smoothing | `scipy.signal.savgol_filter`, `timesat` | `signal::sgolayfilt()`, `phenofit` | TIMESAT |
| Phenology | `phenolopy`, custom | `phenofit`, `greenbrown` | TIMESAT |
| Classification | `scikit-learn`, `tslearn` | `sits`, `caret` | GEE, QGIS SCP |
| Yield regression | `scikit-learn`, `statsmodels` | `lm`, `caret` | Excel, custom |
| Drought indices | `climate_indices` (SPI), custom VCI | `SPEI` package, custom | CDT (WMO) |
| Visualization | `matplotlib`, `seaborn`, `xarray.plot` | `ggplot2`, `tmap` | QGIS, ArcGIS |

### Code: NDVI Time Series, Crop Classification, and Drought Monitoring in Python

```python
"""
Agricultural Monitoring — NDVI Time Series + Crop Classification + Drought Indices
Requirements: pip install numpy rasterio geopandas scikit-learn scipy xarray matplotlib
"""
import numpy as np
import rasterio
import geopandas as gpd
import xarray as xr
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Build NDVI time series cube from multi-date images
# -------------------------------------------------------------------
import glob
import os
from datetime import datetime

ndvi_files = sorted(glob.glob("data/ndvi_timeseries/ndvi_*.tif"))
dates = [datetime.strptime(os.path.basename(f).split("_")[1].split(".")[0], "%Y%m%d")
         for f in ndvi_files]

# Stack into 3D array (time, rows, cols)
with rasterio.open(ndvi_files[0]) as src:
    profile = src.profile.copy()
    rows, cols = src.height, src.width

ndvi_cube = np.zeros((len(ndvi_files), rows, cols), dtype=np.float32)
for i, f in enumerate(ndvi_files):
    with rasterio.open(f) as src:
        ndvi_cube[i] = src.read(1)

# Replace nodata with NaN
ndvi_cube[ndvi_cube < -1] = np.nan
print(f"NDVI cube: {ndvi_cube.shape} (dates={len(dates)}, rows={rows}, cols={cols})")

# -------------------------------------------------------------------
# 2. Savitzky-Golay smoothing of NDVI time series
# -------------------------------------------------------------------
def smooth_timeseries(cube, window=7, polyorder=2):
    """Apply Savitzky-Golay filter along time axis, handling NaN."""
    smoothed = np.copy(cube)
    for r in range(cube.shape[1]):
        for c in range(cube.shape[2]):
            ts = cube[:, r, c]
            valid = ~np.isnan(ts)
            if valid.sum() >= window:
                # Interpolate gaps first
                ts_interp = np.interp(
                    np.arange(len(ts)),
                    np.where(valid)[0],
                    ts[valid]
                )
                smoothed[:, r, c] = savgol_filter(ts_interp, window, polyorder)
            else:
                smoothed[:, r, c] = ts
    return smoothed

# For large rasters, vectorize or process in blocks
# Here we demonstrate on a subset
subset = ndvi_cube[:, :500, :500]
smoothed = smooth_timeseries(subset)

# -------------------------------------------------------------------
# 3. Extract phenological metrics
# -------------------------------------------------------------------
def extract_phenology(ts, dates, ndvi_threshold=0.3):
    """
    Extract phenological metrics from a smoothed NDVI time series.
    Returns: SOS (start of season), EOS (end of season), POS (peak),
             NDVI_max, NDVI_integrated (area under curve).
    """
    doy = np.array([d.timetuple().tm_yday for d in dates])
    metrics = {}

    if np.all(np.isnan(ts)):
        return {k: np.nan for k in ["SOS", "EOS", "POS", "NDVI_max", "NDVI_int", "GSL"]}

    # Peak of season
    peak_idx = np.nanargmax(ts)
    metrics["POS"] = doy[peak_idx]
    metrics["NDVI_max"] = np.nanmax(ts)

    # SOS: first date NDVI crosses threshold (ascending)
    above = ts > ndvi_threshold
    sos_candidates = np.where(above[:peak_idx])[0]
    metrics["SOS"] = doy[sos_candidates[0]] if len(sos_candidates) > 0 else np.nan

    # EOS: last date NDVI crosses threshold (descending)
    eos_candidates = np.where(above[peak_idx:])[0]
    if len(eos_candidates) > 0:
        metrics["EOS"] = doy[peak_idx + eos_candidates[-1]]
    else:
        metrics["EOS"] = np.nan

    # Growing season length
    metrics["GSL"] = metrics["EOS"] - metrics["SOS"] if not np.isnan(metrics["EOS"]) else np.nan

    # Integrated NDVI (sum of NDVI above threshold during growing season)
    if not np.isnan(metrics["SOS"]):
        in_season = (doy >= metrics["SOS"]) & (doy <= (metrics["EOS"] if not np.isnan(metrics["EOS"]) else 365))
        metrics["NDVI_int"] = np.nansum(ts[in_season])
    else:
        metrics["NDVI_int"] = np.nan

    return metrics

# Example: extract phenology for a single pixel
sample_ts = smoothed[:, 250, 250]
pheno = extract_phenology(sample_ts, dates)
print("\n=== Phenological Metrics (sample pixel) ===")
for k, v in pheno.items():
    print(f"  {k}: {v}")

# -------------------------------------------------------------------
# 4. Crop type classification using phenological features
# -------------------------------------------------------------------
# Load field boundaries with crop type labels
fields = gpd.read_file("data/crop_fields.gpkg")  # columns: geometry, crop_type

# Extract mean NDVI time series per field
from rasterstats import zonal_stats

field_features = []
for date_idx in range(len(ndvi_files)):
    stats = zonal_stats(
        fields, ndvi_files[date_idx], stats=["mean"], nodata=-9999
    )
    fields[f"ndvi_{date_idx:03d}"] = [s["mean"] for s in stats]

# Build feature matrix: NDVI at each date + phenological metrics
feature_cols = [f"ndvi_{i:03d}" for i in range(len(ndvi_files))]
X = fields[feature_cols].values
y = fields["crop_type"].values

# Handle NaN
X = np.nan_to_num(X, nan=0.0)

# Stratified 5-fold cross-validation
clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                              class_weight="balanced", n_jobs=-1, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true, all_y_pred = [], []
for train_idx, test_idx in skf.split(X, y):
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    all_y_true.extend(y[test_idx])
    all_y_pred.extend(pred)

print("\n=== Crop Classification Report ===")
print(classification_report(all_y_true, all_y_pred))

# Train final model on all data
clf.fit(X, y)

# Feature importance: which dates matter most?
importances = clf.feature_importances_
top_dates = np.argsort(importances)[-5:]
print("Top 5 most important dates (indices):", top_dates)
print("Dates:", [dates[i].strftime("%Y-%m-%d") for i in top_dates])

# -------------------------------------------------------------------
# 5. Yield estimation (NDVI_max vs district-level yield)
# -------------------------------------------------------------------
yield_data = gpd.read_file("data/district_yield.gpkg")  # columns: district, year, yield_ton_ha

# Compute mean NDVI_max per district per year
district_ndvi = zonal_stats(
    yield_data, "output/ndvi_max_2024.tif", stats=["mean"]
)
yield_data["ndvi_max_mean"] = [s["mean"] for s in district_ndvi]
yield_data = yield_data.dropna(subset=["ndvi_max_mean", "yield_ton_ha"])

# Simple linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

X_yield = yield_data["ndvi_max_mean"].values.reshape(-1, 1)
y_yield = yield_data["yield_ton_ha"].values

reg = LinearRegression().fit(X_yield, y_yield)
y_pred = reg.predict(X_yield)

r2 = r2_score(y_yield, y_pred)
mae = mean_absolute_error(y_yield, y_pred)
print(f"\n=== Yield Model ===")
print(f"  yield = {reg.coef_[0]:.2f} * NDVI_max + {reg.intercept_:.2f}")
print(f"  R2 = {r2:.3f}, MAE = {mae:.2f} ton/ha")

# -------------------------------------------------------------------
# 6. Drought monitoring — VCI and SPI
# -------------------------------------------------------------------
def vegetation_condition_index(ndvi_current, ndvi_min_hist, ndvi_max_hist):
    """
    VCI = (NDVI_current - NDVI_min) / (NDVI_max - NDVI_min) * 100
    VCI < 35: severe drought, 35-50: moderate, >50: normal
    """
    denom = ndvi_max_hist - ndvi_min_hist
    denom[denom == 0] = np.nan
    vci = (ndvi_current - ndvi_min_hist) / denom * 100
    return np.clip(vci, 0, 100)

# Load historical NDVI statistics (precomputed from multi-year archive)
with rasterio.open("data/ndvi_min_historical.tif") as src:
    ndvi_min_hist = src.read(1).astype(np.float64)
with rasterio.open("data/ndvi_max_historical.tif") as src:
    ndvi_max_hist = src.read(1).astype(np.float64)
with rasterio.open("data/ndvi_current.tif") as src:
    ndvi_current = src.read(1).astype(np.float64)
    vci_profile = src.profile.copy()

vci = vegetation_condition_index(ndvi_current, ndvi_min_hist, ndvi_max_hist)

# Classify drought severity
drought_class = np.full_like(vci, np.nan)
drought_class[vci < 10]                    = 1  # Extreme drought
drought_class[(vci >= 10) & (vci < 20)]    = 2  # Severe drought
drought_class[(vci >= 20) & (vci < 35)]    = 3  # Moderate drought
drought_class[(vci >= 35) & (vci < 50)]    = 4  # Mild stress
drought_class[vci >= 50]                    = 5  # Normal/Good

# Save VCI
vci_profile.update(dtype="float64", nodata=np.nan)
with rasterio.open("output/vci.tif", "w", **vci_profile) as dst:
    dst.write(vci, 1)

print("\n=== VCI Drought Classification ===")
labels = {1: "Extreme", 2: "Severe", 3: "Moderate", 4: "Mild", 5: "Normal"}
for cls, name in labels.items():
    pct = np.nansum(drought_class == cls) / np.nansum(~np.isnan(drought_class)) * 100
    print(f"  {name:>12s}: {pct:>6.1f}%")

# -------------------------------------------------------------------
# 7. SPI (Standardized Precipitation Index) — 3-month
# -------------------------------------------------------------------
import pandas as pd
from scipy.stats import gamma as gamma_dist

def compute_spi(precip_series, window=3):
    """
    Compute SPI from a monthly precipitation time series.
    precip_series: pandas Series with DatetimeIndex
    window: accumulation period in months
    """
    # Rolling sum
    precip_roll = precip_series.rolling(window=window, min_periods=window).sum()

    # Fit gamma distribution to each calendar month
    spi = pd.Series(index=precip_series.index, dtype=np.float64)
    for month in range(1, 13):
        month_data = precip_roll[precip_roll.index.month == month].dropna()
        if len(month_data) < 10:
            continue
        # Gamma fit
        shape, loc, scale = gamma_dist.fit(month_data[month_data > 0], floc=0)
        # CDF -> SPI (inverse normal)
        from scipy.stats import norm
        cdf = gamma_dist.cdf(month_data, shape, loc=loc, scale=scale)
        # Handle zeros
        q = (month_data == 0).sum() / len(month_data)
        cdf_adjusted = q + (1 - q) * cdf
        cdf_adjusted = np.clip(cdf_adjusted, 0.001, 0.999)
        spi_values = norm.ppf(cdf_adjusted)
        spi.loc[month_data.index] = spi_values

    return spi

# Load monthly precipitation for a station/pixel
precip = pd.read_csv("data/monthly_precip.csv", parse_dates=["date"], index_col="date")
spi_3 = compute_spi(precip["precip_mm"], window=3)

print("\n=== SPI-3 Classification ===")
print(f"  Current SPI-3: {spi_3.iloc[-1]:.2f}")
if spi_3.iloc[-1] < -2:
    print("  Classification: Extreme Drought")
elif spi_3.iloc[-1] < -1.5:
    print("  Classification: Severe Drought")
elif spi_3.iloc[-1] < -1:
    print("  Classification: Moderate Drought")
else:
    print("  Classification: Normal or Wet")
```

### Expected Output

- **NDVI time series** (smoothed temporal profiles per field/pixel)
- **Phenological metrics maps** (SOS, EOS, POS, NDVI_max, growing season length)
- **Crop type classification map** (per-field crop labels with accuracy report)
- **Yield estimation map** (predicted yield per district with confidence interval)
- **VCI drought map** (Vegetation Condition Index classified by severity)
- **SPI time series** (3-month SPI with drought classification)
- **NDVI anomaly map** (current season vs. historical mean z-score)
- **Management zone map** (within-field variability, k-means clusters)
- **Report** with crop statistics, drought alerts, and yield forecasts

### Flowchart

```
[Satellite Time Series (Sentinel-2 / MODIS)]
                    |
                    v
         [Cloud Mask + Gap Fill]
                    |
                    v
         [NDVI / EVI Computation]
                    |
                    v
         [Savitzky-Golay Smoothing]
                    |
        +-----------+-----------+-----------+
        |           |           |           |
        v           v           v           v
 [Phenological  [Crop Type   [Yield      [Drought
  Metrics]       Classific.] Estimation]  Indices]
 [SOS,EOS,POS]  [RF on       [NDVI_max   [VCI from
  NDVI_max,      temporal     vs yield    historical
  NDVI_int]      profiles]    regression] min/max]
        |           |           |           |
        |           |           |      +----+----+
        |           |           |      |         |
        |           |           |      v         v
        |           |           |    [VCI]     [SPI]
        |           |           |    [map]     [precip]
        |           |           |      |         |
        +-----------+-----------+------+---------+
                           |
                           v
              [Anomaly Detection & Alerts]
              [Z-score vs historical mean]
                           |
                           v
              [Management Zone Mapping]
              [Within-field k-means]
                           |
                           v
                    [Report & Dashboard]
```

---

## 8. Population Estimation

### Objective

Estimate population distribution at fine spatial resolution using dasymetric mapping, building footprint analysis, and ancillary satellite-derived data. The workflow disaggregates census totals to grid cells weighted by land cover, building density, nighttime lights, and other proxies, producing gridded population surfaces suitable for disaster response, infrastructure planning, and accessibility analysis.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Census population (admin units) | Total population per unit | National census bureau | Shapefile / CSV |
| Building footprints | Building density, residential area | OpenStreetMap, Microsoft, Google Open Buildings | GeoJSON / GeoPackage |
| Land cover | Residential vs. non-residential mask | ESA WorldCover, CORINE, NLCD | GeoTIFF |
| Nighttime lights | Urban intensity proxy | VIIRS DNB, DMSP-OLS | GeoTIFF |
| WorldPop / GHSL (reference) | Comparison / validation | WorldPop, GHSL-POP | GeoTIFF |
| DEM | Exclude steep/uninhabitable terrain | SRTM, Copernicus DEM | GeoTIFF |
| Road network | Settlement proximity | OpenStreetMap | Shapefile |
| Administrative boundaries | Reporting units, aggregation | National boundary dataset | Shapefile |

**Data Portals:** [WorldPop](https://www.worldpop.org/) | [GHSL](https://ghsl.jrc.ec.europa.eu/) | [Google Open Buildings](https://sites.research.google/open-buildings/) | [Microsoft Building Footprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | [VIIRS Nighttime Lights](https://eogdata.mines.edu/products/vnl/) | [ESA WorldCover](https://worldcover2021.esa.int/)

> See also: [`data-sources/population.md`](../data-sources/population.md), [`data-sources/land-cover.md`](../data-sources/land-cover.md)

### Steps

1. **Prepare census data** -- Obtain population totals per administrative unit (e.g., district, ward). Standardize fields, join to boundary geometries.
2. **Create population exclusion mask** -- Identify uninhabitable areas: water, steep slopes (>30 degrees), protected forests, industrial zones. These receive zero population.
3. **Building footprint analysis** -- Count buildings and total building area per grid cell (e.g., 100 m resolution). Classify residential vs. non-residential (by area, shape, OSM tags).
4. **Ancillary weight layers** -- Compute additional proxies: nighttime light intensity, distance to roads, land cover class weights (e.g., urban residential=1.0, suburban=0.7, rural=0.3, non-residential=0).
5. **Dasymetric disaggregation** -- Distribute census population to grid cells proportionally to combined weights. The weight of cell _i_ within admin unit _k_ is: w_i = f(buildings_i, lights_i, landcover_i). Population of cell _i_ = Pop_k * w_i / sum(w_j for j in k).
6. **Validation** -- Compare with WorldPop/GHSL at admin-unit level (R-squared, RMSE). Cross-validate using leave-one-unit-out.
7. **Change detection** -- Compare population grids from two census years. Compute growth rates, identify rapidly growing areas.
8. **Population projections** -- Apply growth rates or regression model to project future population (5, 10, 20 years).
9. **Uncertainty mapping** -- Bootstrap disaggregation weights, compute confidence intervals per grid cell.
10. **Report** -- Population density maps, comparison with existing products, growth maps, projection maps.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Census data prep | `geopandas`, `pandas` | `sf`, `dplyr` | QGIS, ArcGIS |
| Building analysis | `geopandas`, `shapely` | `sf` | QGIS |
| Rasterization | `rasterio`, `geocube` | `terra::rasterize()` | QGIS Rasterize |
| Dasymetric mapping | Custom (`numpy` + `rasterio`) | `populR`, custom | Areal interpolation tools |
| Validation | `sklearn.metrics` | `Metrics`, `yardstick` | Excel / custom |
| Visualization | `matplotlib`, `contextily` | `tmap`, `ggplot2` | QGIS Print Layout |

### Code: Dasymetric Population Disaggregation in Python

```python
"""
Population Estimation — Dasymetric Mapping with Building Footprints
Requirements: pip install rasterio numpy geopandas geocube rasterstats matplotlib
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd
from rasterstats import zonal_stats
from shapely.geometry import box
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Configuration
# -------------------------------------------------------------------
GRID_RES = 100  # metres (output grid resolution)
TARGET_CRS = "EPSG:32637"
NODATA = -9999

# -------------------------------------------------------------------
# 2. Load census data and administrative boundaries
# -------------------------------------------------------------------
admin = gpd.read_file("data/admin_boundaries.gpkg").to_crs(TARGET_CRS)
# admin must have columns: admin_id, population (census total)

print(f"Admin units: {len(admin)}")
print(f"Total population: {admin['population'].sum():,.0f}")

# -------------------------------------------------------------------
# 3. Build raster grid covering study area
# -------------------------------------------------------------------
bounds = admin.total_bounds  # [minx, miny, maxx, maxy]
width = int(np.ceil((bounds[2] - bounds[0]) / GRID_RES))
height = int(np.ceil((bounds[3] - bounds[1]) / GRID_RES))
transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)

grid_profile = {
    "driver": "GTiff", "dtype": "float64", "nodata": NODATA,
    "width": width, "height": height, "count": 1,
    "crs": TARGET_CRS, "transform": transform
}
print(f"Output grid: {width} x {height} pixels ({GRID_RES}m)")

# -------------------------------------------------------------------
# 4. Building footprint density layer
# -------------------------------------------------------------------
buildings = gpd.read_file("data/building_footprints.gpkg").to_crs(TARGET_CRS)

# Compute building area
buildings["bldg_area"] = buildings.geometry.area

# Rasterize: total building area per grid cell
# Create cell polygons and compute zonal building area
# Alternative: rasterize building centroids with area as burn value
building_centroids = buildings.copy()
building_centroids["geometry"] = buildings.geometry.centroid

# Rasterize building area to grid
shapes_iter = ((geom, area) for geom, area in
               zip(building_centroids.geometry, building_centroids["bldg_area"]))
bldg_raster = rasterize(
    shapes=[(geom, 1) for geom in buildings.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    merge_alg=rasterio.enums.MergeAlg.add,
    dtype="float64"
)

# Count buildings per cell (using centroid rasterization)
bldg_count = rasterize(
    shapes=[(geom, 1) for geom in building_centroids.geometry],
    out_shape=(height, width),
    transform=transform,
    fill=0,
    merge_alg=rasterio.enums.MergeAlg.add,
    dtype="float64"
)

print(f"Building density: max {bldg_count.max():.0f} buildings/cell")

# -------------------------------------------------------------------
# 5. Land cover weight layer
# -------------------------------------------------------------------
with rasterio.open("data/landcover.tif") as src:
    lc = src.read(1)
    # Resample to target grid if needed (simplified: assume same extent)

# Assign population weights to land cover classes
# ESA WorldCover classes: 10=tree, 20=shrub, 30=grassland, 40=cropland,
# 50=built-up, 60=bare, 80=water, 90=wetland, 95=mangrove, 100=ice
lc_weights = {
    10: 0.05, 20: 0.05, 30: 0.1, 40: 0.2,
    50: 1.0,  60: 0.0,  80: 0.0, 90: 0.0, 95: 0.0, 100: 0.0
}
lc_weight_raster = np.zeros_like(lc, dtype=np.float64)
for cls, w in lc_weights.items():
    lc_weight_raster[lc == cls] = w

# -------------------------------------------------------------------
# 6. Nighttime lights weight layer
# -------------------------------------------------------------------
with rasterio.open("data/viirs_nightlights.tif") as src:
    ntl = src.read(1).astype(np.float64)
    ntl[ntl < 0] = 0

# Normalize to 0-1
ntl_norm = ntl / (ntl.max() + 1e-10)

# -------------------------------------------------------------------
# 7. Combined weight layer
# -------------------------------------------------------------------
# Weight = building_density * 0.5 + lc_weight * 0.3 + nightlights * 0.2
combined_weight = (
    0.5 * (bldg_count / (bldg_count.max() + 1e-10)) +
    0.3 * lc_weight_raster +
    0.2 * ntl_norm
)

# Exclusion mask (water, steep slopes)
with rasterio.open("data/slope.tif") as src:
    slope = src.read(1)
exclusion = (lc_weight_raster == 0) | (slope > 30)
combined_weight[exclusion] = 0

# -------------------------------------------------------------------
# 8. Dasymetric disaggregation
# -------------------------------------------------------------------
# Rasterize admin unit IDs
admin_raster = rasterize(
    shapes=((geom, admin_id) for geom, admin_id in
            zip(admin.geometry, admin["admin_id"])),
    out_shape=(height, width),
    transform=transform,
    fill=0,
    dtype="int32"
)

population_grid = np.zeros((height, width), dtype=np.float64)

for _, row in admin.iterrows():
    admin_id = row["admin_id"]
    pop_total = row["population"]

    mask = admin_raster == admin_id
    weights_in_unit = combined_weight[mask]
    weight_sum = weights_in_unit.sum()

    if weight_sum > 0:
        population_grid[mask] = pop_total * (weights_in_unit / weight_sum)
    else:
        # Uniform distribution if no weight information
        n_cells = mask.sum()
        population_grid[mask] = pop_total / n_cells if n_cells > 0 else 0

# -------------------------------------------------------------------
# 9. Save gridded population
# -------------------------------------------------------------------
with rasterio.open("output/population_grid.tif", "w", **grid_profile) as dst:
    dst.write(population_grid, 1)

print(f"\n=== Population Grid Summary ===")
print(f"Total pop (grid):  {population_grid.sum():>12,.0f}")
print(f"Total pop (census): {admin['population'].sum():>12,.0f}")
print(f"Max density:        {population_grid.max():>12,.0f} per cell")
print(f"Non-zero cells:     {(population_grid > 0).sum():>12,}")

# -------------------------------------------------------------------
# 10. Validation — compare with WorldPop
# -------------------------------------------------------------------
worldpop_stats = zonal_stats(
    admin, "data/worldpop_100m.tif", stats=["sum"]
)
admin["worldpop_sum"] = [s["sum"] if s["sum"] else 0 for s in worldpop_stats]

our_stats = zonal_stats(admin, "output/population_grid.tif", stats=["sum"])
admin["our_pop_sum"] = [s["sum"] if s["sum"] else 0 for s in our_stats]

from sklearn.metrics import r2_score, mean_absolute_error
mask_valid = (admin["worldpop_sum"] > 0) & (admin["our_pop_sum"] > 0)
r2 = r2_score(admin.loc[mask_valid, "population"], admin.loc[mask_valid, "our_pop_sum"])
mae = mean_absolute_error(admin.loc[mask_valid, "population"], admin.loc[mask_valid, "our_pop_sum"])

print(f"\n=== Validation vs Census ===")
print(f"  R2:  {r2:.4f}")
print(f"  MAE: {mae:,.0f} people per admin unit")

# Compare with WorldPop
r2_wp = r2_score(admin.loc[mask_valid, "population"], admin.loc[mask_valid, "worldpop_sum"])
print(f"\n=== WorldPop vs Census ===")
print(f"  R2:  {r2_wp:.4f}")
print(f"  Our product {'outperforms' if r2 > r2_wp else 'underperforms'} WorldPop")

# -------------------------------------------------------------------
# 11. Population change detection (two census years)
# -------------------------------------------------------------------
# Assume we have population grids for 2010 and 2020
with rasterio.open("output/population_grid_2010.tif") as src:
    pop_2010 = src.read(1)

growth_rate = np.where(
    pop_2010 > 10,  # avoid division by small numbers
    (population_grid - pop_2010) / pop_2010 * 100,
    np.nan
)

print(f"\n=== Population Change 2010-2020 ===")
print(f"  Mean growth rate: {np.nanmean(growth_rate):.1f}%")
print(f"  Cells with >50% growth: {np.nansum(growth_rate > 50):,}")
print(f"  Cells with decline: {np.nansum(growth_rate < 0):,}")
```

### Expected Output

- **Gridded population surface** (100 m resolution GeoTIFF, people per cell)
- **Population density map** (people per km2)
- **Building density raster** (buildings/area per grid cell)
- **Combined weight layer** (visualization of disaggregation weights)
- **Validation report** (R2, MAE vs. census totals, comparison with WorldPop/GHSL)
- **Population change map** (growth rate between two census years)
- **Population projection maps** (5, 10, 20 year horizons)
- **Uncertainty map** (confidence interval per grid cell from bootstrap)

### Flowchart

```
[Census Data (Admin Units)]     [Building Footprints]     [Ancillary Layers]
[Population totals per unit]    [Count + Area per cell]   [Land Cover, NTL, Slope]
         |                              |                          |
         v                              v                          v
[Admin Boundary Rasterization]  [Building Density Raster]  [Weight Layers (0-1)]
         |                              |                          |
         |                              +------------+-------------+
         |                                           |
         |                                           v
         |                               [Combined Weight Layer]
         |                               [w = 0.5*bldg + 0.3*lc + 0.2*ntl]
         |                                           |
         |                                           v
         |                               [Exclusion Mask (water, steep)]
         |                                           |
         +-------------------------------------------+
                           |
                           v
              [Dasymetric Disaggregation]
              [Pop_cell = Pop_admin * w_cell / sum(w)]
                           |
                           v
              [Gridded Population Surface]
                           |
              +------------+------------+
              |            |            |
              v            v            v
       [Validation   [Change       [Population
        vs Census     Detection]    Projection]
        & WorldPop]  [2 timepoints] [Growth rates]
              |            |            |
              +------------+------------+
                           |
                           v
                    [Report & Maps]
```

---

## 9. Network Analysis & Routing

### Objective

Build routable graph networks from OpenStreetMap data, compute shortest paths and service areas, optimize fleet routing, and perform spatial equity analysis of service coverage. The workflow uses OSMnx for graph construction, NetworkX/igraph for analysis, and optionally pgRouting for database-backed routing at scale.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Road network | Graph construction | OpenStreetMap (via OSMnx or Geofabrik PBF) | Graph / PBF |
| Facility locations | Service points (depots, stores, hospitals) | Government data, POI databases | GeoPackage / CSV |
| Demand points | Customer/patient locations | Synthetic from population, address databases | GeoPackage / CSV |
| Speed / travel time data | Edge weights | OSM speed limits, HERE/TomTom traffic data | Attributes on edges |
| Population / census | Equity analysis | Census, WorldPop | Shapefile / GeoTIFF |
| Administrative boundaries | Reporting units | National boundary dataset | Shapefile |
| Elevation data (optional) | Grade-adjusted travel for cycling/walking | SRTM, LiDAR | GeoTIFF |

**Data Portals:** [OpenStreetMap (Geofabrik)](https://download.geofabrik.de/) | [OpenRouteService](https://openrouteservice.org/) | [Valhalla](https://github.com/valhalla/valhalla) | [OSRM](http://project-osrm.org/) | [HERE Traffic](https://developer.here.com/)

> See also: [`tools/network-analysis.md`](../tools/network-analysis.md), [`data-sources/transport.md`](../data-sources/transport.md)

### Steps

1. **Download and build graph** -- Use OSMnx to download the road network for the study area and construct a NetworkX directed graph with travel time attributes.
2. **Graph simplification** -- Simplify (merge degree-2 nodes), project to local UTM, add edge speeds and travel times.
3. **Shortest path analysis** -- Compute shortest path (Dijkstra) between origin-destination pairs, extract route geometry, distance, and travel time.
4. **Service area / isochrone** -- Compute the set of reachable nodes within a travel time threshold from each facility (ego graph).
5. **Closest facility assignment** -- Assign each demand point to its nearest facility by travel time (not Euclidean distance).
6. **Fleet routing (VRP)** -- Solve the Vehicle Routing Problem for a set of demand points and depot using OR-Tools or heuristic solvers. Minimize total travel time subject to vehicle capacity constraints.
7. **Service coverage equity** -- Compute travel time to nearest facility per census unit. Disaggregate by income, race, or deprivation quintile. Test for statistically significant differences (Kruskal-Wallis).
8. **Network centrality** -- Compute betweenness centrality to identify critical road segments. Simulate disruptions.
9. **pgRouting (optional)** -- Load network into PostGIS for database-backed routing at scale with pgRouting extension.
10. **Report** -- Route maps, service area maps, equity analysis, VRP solution, network vulnerability assessment.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Graph construction | `osmnx` | `osmdata`, `sfnetworks` | QGIS OSM download |
| Shortest path | `networkx`, `igraph` | `cppRouting`, `dodgr` | pgRouting, ArcGIS NA |
| Service areas | `osmnx`, `networkx.ego_graph` | `sfnetworks` | ORS Tools, ArcGIS NA |
| VRP solver | `ortools` (Google OR-Tools) | `ompr`, `TSP` | ArcGIS VRP solver |
| Equity analysis | `geopandas`, `scipy.stats` | `sf`, `ineq`, `stats` | QGIS, custom |
| pgRouting | `psycopg2` + SQL | `DBI` + SQL | pgAdmin, QGIS |
| Visualization | `osmnx.plot`, `folium`, `matplotlib` | `tmap`, `leaflet` | QGIS Print Layout |

### Code: Network Analysis, Routing, and Equity in Python

```python
"""
Network Analysis & Routing — OSMnx + OR-Tools VRP + Equity Analysis
Requirements: pip install osmnx networkx geopandas ortools scipy matplotlib folium
"""
import osmnx as ox
import networkx as nx
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, LineString
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Download and build road network graph
# -------------------------------------------------------------------
PLACE = "Portland, Oregon, USA"
NETWORK_TYPE = "drive"

G = ox.graph_from_place(PLACE, network_type=NETWORK_TYPE)
G = ox.project_graph(G)           # project to local UTM
G = ox.add_edge_speeds(G)         # infer speeds from highway tags
G = ox.add_edge_travel_times(G)   # travel_time in seconds

print(f"Network: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# Basic stats
stats_dict = ox.basic_stats(G)
print(f"  Total edge length: {stats_dict['edge_length_total']/1000:.0f} km")
print(f"  Average node degree: {stats_dict['streets_per_node_avg']:.2f}")

# -------------------------------------------------------------------
# 2. Shortest path between two points
# -------------------------------------------------------------------
origin = (45.523, -122.676)  # lat, lon
destination = (45.510, -122.650)

orig_node = ox.nearest_nodes(G, origin[1], origin[0])
dest_node = ox.nearest_nodes(G, destination[1], destination[0])

route = ox.shortest_path(G, orig_node, dest_node, weight="travel_time")
route_length = nx.path_weight(G, route, weight="length")
route_time = nx.path_weight(G, route, weight="travel_time")

print(f"\nShortest path: {route_length/1000:.2f} km, {route_time/60:.1f} min")

# Extract route as GeoDataFrame
route_edges = ox.routing.route_to_gdf(G, route)
route_edges.to_file("output/shortest_route.gpkg", driver="GPKG")

# -------------------------------------------------------------------
# 3. Service areas (multiple thresholds from multiple facilities)
# -------------------------------------------------------------------
facilities = gpd.read_file("data/fire_stations.gpkg")
facilities = facilities.to_crs(G.graph["crs"])
facility_nodes = [ox.nearest_nodes(G, row.geometry.x, row.geometry.y)
                  for _, row in facilities.iterrows()]

thresholds_min = [4, 8, 12]  # minutes

service_areas = []
for f_idx, f_node in enumerate(facility_nodes):
    for t in thresholds_min:
        subgraph = nx.ego_graph(G, f_node, radius=t * 60, distance="travel_time")
        node_points = [Point(G.nodes[n]["x"], G.nodes[n]["y"])
                       for n in subgraph.nodes()]
        if node_points:
            hull = gpd.GeoSeries(node_points, crs=G.graph["crs"]).unary_union.convex_hull
            service_areas.append({
                "facility_id": f_idx,
                "threshold_min": t,
                "geometry": hull
            })

service_gdf = gpd.GeoDataFrame(service_areas, crs=G.graph["crs"])
service_gdf.to_file("output/service_areas.gpkg", driver="GPKG")
print(f"\nService areas computed: {len(service_gdf)} polygons")

# -------------------------------------------------------------------
# 4. Closest facility assignment for all census block centroids
# -------------------------------------------------------------------
census = gpd.read_file("data/census_blocks.gpkg")
census = census.to_crs(G.graph["crs"])
census["centroid"] = census.geometry.centroid
census_nodes = [ox.nearest_nodes(G, c.x, c.y) for c in census["centroid"]]
census["node_id"] = census_nodes

# Compute travel time from each facility to all reachable nodes
facility_tt = {}
G_rev = G.reverse()
for f_idx, f_node in enumerate(facility_nodes):
    lengths = nx.single_source_dijkstra_path_length(
        G_rev, f_node, cutoff=30 * 60, weight="travel_time"
    )
    facility_tt[f_idx] = lengths

# Assign nearest facility and travel time
census["nearest_facility"] = -1
census["travel_time_min"] = np.inf

for c_idx, c_node in enumerate(census["node_id"]):
    for f_idx, lengths in facility_tt.items():
        if c_node in lengths:
            tt_min = lengths[c_node] / 60
            if tt_min < census.loc[c_idx, "travel_time_min"]:
                census.loc[c_idx, "travel_time_min"] = tt_min
                census.loc[c_idx, "nearest_facility"] = f_idx

census.loc[census["travel_time_min"] == np.inf, "travel_time_min"] = np.nan

print(f"\n=== Travel Time to Nearest Fire Station ===")
print(f"  Mean: {census['travel_time_min'].mean():.1f} min")
print(f"  Median: {census['travel_time_min'].median():.1f} min")
print(f"  95th percentile: {census['travel_time_min'].quantile(0.95):.1f} min")
print(f"  Beyond 8 min: {(census['travel_time_min'] > 8).sum()} blocks "
      f"({(census['travel_time_min'] > 8).mean()*100:.1f}%)")

# -------------------------------------------------------------------
# 5. Vehicle Routing Problem (VRP) with Google OR-Tools
# -------------------------------------------------------------------
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

# Setup: 1 depot, N demand points, K vehicles
depot = facilities.iloc[0]
demand_points = gpd.read_file("data/delivery_points.gpkg").to_crs(G.graph["crs"])

# Build list: depot (index 0) + demand points
all_points = [depot] + [row for _, row in demand_points.iterrows()]
all_nodes = [ox.nearest_nodes(G, p.geometry.x, p.geometry.y) for p in all_points]

# Compute travel time matrix
n = len(all_nodes)
tt_matrix = np.full((n, n), 999999, dtype=int)
for i in range(n):
    lengths = nx.single_source_dijkstra_path_length(
        G, all_nodes[i], cutoff=60 * 60, weight="travel_time"
    )
    for j in range(n):
        if all_nodes[j] in lengths:
            tt_matrix[i, j] = int(lengths[all_nodes[j]])

# OR-Tools VRP
NUM_VEHICLES = 3
VEHICLE_CAPACITY = 15  # max stops per vehicle

manager = pywrapcp.RoutingIndexManager(n, NUM_VEHICLES, 0)  # 0 = depot
routing = pywrapcp.RoutingModel(manager)

def time_callback(from_index, to_index):
    from_node = manager.IndexToNode(from_index)
    to_node = manager.IndexToNode(to_index)
    return tt_matrix[from_node][to_node]

transit_callback_index = routing.RegisterTransitCallback(time_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

# Capacity constraint
def demand_callback(from_index):
    node = manager.IndexToNode(from_index)
    return 0 if node == 0 else 1  # each stop = 1 unit of demand

demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
routing.AddDimensionWithVehicleCapacity(
    demand_callback_index, 0, [VEHICLE_CAPACITY] * NUM_VEHICLES, True, "Capacity"
)

# Solve
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
)
search_parameters.local_search_metaheuristic = (
    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
)
search_parameters.time_limit.seconds = 30

solution = routing.SolveWithParameters(search_parameters)

if solution:
    print(f"\n=== VRP Solution ===")
    total_time = 0
    for v in range(NUM_VEHICLES):
        index = routing.Start(v)
        route_nodes = []
        route_time = 0
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            route_nodes.append(node)
            next_index = solution.Value(routing.NextVar(index))
            route_time += routing.GetArcCostForVehicle(index, next_index, v)
            index = next_index
        route_nodes.append(manager.IndexToNode(index))
        total_time += route_time
        print(f"  Vehicle {v}: {len(route_nodes)-2} stops, "
              f"{route_time/60:.0f} min, route: {route_nodes}")
    print(f"  Total time: {total_time/60:.0f} min")

# -------------------------------------------------------------------
# 6. Service coverage equity analysis
# -------------------------------------------------------------------
# Load socioeconomic data
census = census.merge(
    gpd.read_file("data/socioeconomic.gpkg")[["block_id", "median_income", "pct_minority"]],
    on="block_id", how="left"
)

# Income quintiles
census["income_quintile"] = pd.qcut(
    census["median_income"], q=5,
    labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
)

# Travel time by income quintile
print("\n=== Equity Analysis: Travel Time by Income Quintile ===")
equity_summary = census.groupby("income_quintile")["travel_time_min"].agg(
    ["mean", "median", "std", "count"]
).round(2)
print(equity_summary.to_string())

# Kruskal-Wallis test (non-parametric ANOVA)
groups = [group["travel_time_min"].dropna().values
          for _, group in census.groupby("income_quintile")]
H_stat, p_value = stats.kruskal(*groups)
print(f"\nKruskal-Wallis test: H={H_stat:.2f}, p={p_value:.4f}")
if p_value < 0.05:
    print("  Significant difference in travel time across income groups.")
else:
    print("  No significant difference across income groups.")

# -------------------------------------------------------------------
# 7. Network centrality (betweenness) — identify critical links
# -------------------------------------------------------------------
# Compute on simplified graph for performance
G_simple = ox.simplify_graph(ox.project_graph(
    ox.graph_from_place(PLACE, network_type=NETWORK_TYPE)
))

# Edge betweenness centrality (sample k nodes for speed)
bc = nx.edge_betweenness_centrality(G_simple, k=200, weight="travel_time")

# Add to edges
for (u, v), centrality in bc.items():
    if G_simple.has_edge(u, v):
        G_simple[u][v][0]["betweenness"] = centrality

# Top 10 most critical edges
top_edges = sorted(bc.items(), key=lambda x: x[1], reverse=True)[:10]
print("\n=== Top 10 Most Critical Road Segments ===")
for (u, v), cent in top_edges:
    name = G_simple[u][v][0].get("name", "unnamed")
    length = G_simple[u][v][0].get("length", 0)
    print(f"  {name}: betweenness={cent:.6f}, length={length:.0f}m")

census.to_file("output/network_analysis_results.gpkg", driver="GPKG")
```

### Code: pgRouting Setup (SQL)

```sql
-- pgRouting: Load OSM network and compute shortest paths in PostGIS
-- Prerequisite: osm2pgrouting to load OSM into PostGIS with topology

-- 1. Shortest path (Dijkstra)
SELECT seq, node, edge, cost, agg_cost, ST_AsText(the_geom) as geom
FROM pgr_dijkstra(
    'SELECT id, source, target, cost_s AS cost, reverse_cost_s AS reverse_cost
     FROM ways',
    (SELECT id FROM ways_vertices_pgr ORDER BY ST_Distance(the_geom,
     ST_SetSRID(ST_MakePoint(-122.676, 45.523), 4326)) LIMIT 1),  -- origin
    (SELECT id FROM ways_vertices_pgr ORDER BY ST_Distance(the_geom,
     ST_SetSRID(ST_MakePoint(-122.650, 45.510), 4326)) LIMIT 1),  -- destination
    directed := true
) AS route
JOIN ways ON route.edge = ways.gid;

-- 2. Service area (driving distance / isochrone)
SELECT node, agg_cost,
       ST_Buffer(v.the_geom, 50) as geom  -- buffer for visualization
FROM pgr_drivingDistance(
    'SELECT id, source, target, cost_s AS cost FROM ways',
    (SELECT id FROM ways_vertices_pgr ORDER BY ST_Distance(the_geom,
     ST_SetSRID(ST_MakePoint(-122.676, 45.523), 4326)) LIMIT 1),
    480,  -- 8 minutes (in seconds)
    directed := true
) AS dd
JOIN ways_vertices_pgr AS v ON dd.node = v.id;

-- 3. Many-to-many shortest path matrix
SELECT start_vid, end_vid, agg_cost
FROM pgr_dijkstraCostMatrix(
    'SELECT id, source, target, cost_s AS cost, reverse_cost_s AS reverse_cost FROM ways',
    ARRAY[1001, 1002, 1003, 1004, 1005],  -- facility vertex IDs
    directed := true
);
```

### Expected Output

- **Shortest path routes** (geometry + distance + travel time)
- **Service area polygons** (4, 8, 12 minute isochrones per facility)
- **Closest facility map** (each census block colored by assigned facility)
- **VRP solution** (optimized routes per vehicle, total travel time, route maps)
- **Equity analysis table** (travel time statistics by income/demographic quintile)
- **Kruskal-Wallis test result** (statistical significance of equity gaps)
- **Network centrality map** (betweenness centrality, critical segments highlighted)
- **Report** with route maps, coverage statistics, equity findings, VRP results

### Flowchart

```
[OpenStreetMap Data (PBF/API)]
              |
              v
[Build Graph (OSMnx / osm2pgrouting)]
[Add speeds, travel times, topology]
              |
   +----------+----------+----------+----------+
   |          |          |          |          |
   v          v          v          v          v
[Shortest  [Service   [Closest   [VRP /     [Network
 Path]      Area]      Facility]  Fleet      Centrality]
[Dijkstra] [Ego       [Reverse   Routing]   [Betweenness]
[A* / BFS]  Graph]     Dijkstra] [OR-Tools] [Critical
   |          |        [Assign    [Capacity   segments]
   |          |         to pop]    + time]       |
   |          |          |          |            |
   +----------+----------+----------+           |
                    |                            |
                    v                            |
         [Population / Census Overlay]           |
         [Travel time per census unit]           |
                    |                            |
                    v                            |
         [Equity Analysis]                       |
         [By income, race, deprivation]          |
         [Kruskal-Wallis / Gini]                 |
                    |                            |
                    +----------------------------+
                    |
                    v
             [Report & Maps]
```

---

## 10. Air Quality Assessment

### Objective

Assess spatial patterns of air pollution using monitoring station data, land use regression (LUR) modeling, population exposure estimation, and environmental justice analysis. The workflow interpolates sparse sensor measurements into continuous surfaces, builds a LUR model to explain pollution variability by land use predictors, estimates population-weighted exposure, and evaluates whether pollution burdens are equitably distributed across demographic groups.

### Data Required

| Data | Purpose | Example Source | Format |
|---|---|---|---|
| Air quality monitoring data | PM2.5, NO2, O3 measurements | EPA AQS, EEA AirBase, OpenAQ, local networks | CSV / API |
| Low-cost sensor data (optional) | High-density supplementary data | PurpleAir, Clarity, community networks | CSV / API |
| Land cover | LUR predictor (green space, industrial) | ESA WorldCover, NLCD, CORINE | GeoTIFF |
| Road network | Traffic proximity predictor | OpenStreetMap | Shapefile |
| Population data | Exposure estimation | Census, WorldPop | Shapefile / GeoTIFF |
| Building footprints / land use | Urban density predictor | OpenStreetMap, zoning maps | Shapefile |
| Elevation / terrain | Dispersion modeling | SRTM, LiDAR | GeoTIFF |
| Meteorology (optional) | Wind, temperature, mixing height | ERA5, local weather stations | CSV / NetCDF |
| Socioeconomic / demographic data | Environmental justice analysis | Census (income, race, age) | Shapefile / CSV |

**Data Portals:** [OpenAQ](https://openaq.org/) | [EPA AQS](https://aqs.epa.gov/aqsweb/airdata/download_files.html) | [EEA AirBase](https://www.eea.europa.eu/data-and-maps/data/airbase-the-european-air-quality-database-8) | [PurpleAir](https://www2.purpleair.com/) | [CAMS Global (Copernicus)](https://ads.atmosphere.copernicus.eu/) | [ERA5](https://cds.climate.copernicus.eu/)

> See also: [`data-sources/air-quality.md`](../data-sources/air-quality.md), [`data-sources/climate.md`](../data-sources/climate.md)

### Steps

1. **Data acquisition and QA/QC** -- Download monitoring data, remove invalid readings, handle duplicates, flag outliers, apply calibration corrections for low-cost sensors.
2. **Exploratory analysis** -- Summary statistics, temporal patterns (diurnal, seasonal), spatial distribution of stations, correlation between pollutants.
3. **Spatial interpolation** -- Create continuous pollution surfaces using IDW (Inverse Distance Weighting), ordinary kriging, or universal kriging. Evaluate with leave-one-out cross-validation (LOOCV).
4. **LUR predictor extraction** -- For each monitoring site, compute buffer-based predictors: traffic intensity within 100/300/500/1000 m, road length by type, % green space, % industrial land, population density, elevation, distance to major sources.
5. **Land Use Regression model** -- Build stepwise or regularized (LASSO/Ridge) regression model. Predictors selected by statistical significance and VIF < 5. Validate with LOOCV R-squared and RMSE.
6. **Apply LUR to grid** -- Predict pollution at every grid cell using the LUR model and gridded predictor layers. Produce high-resolution (100 m) pollution surface.
7. **Population exposure estimation** -- Overlay pollution surface with population grid. Compute population-weighted mean exposure (PWE): PWE = sum(Pop_i * Conc_i) / sum(Pop_i).
8. **Health impact assessment (optional)** -- Apply concentration-response functions (e.g., WHO, GBD) to estimate attributable mortality/morbidity from PM2.5 or NO2 exposure.
9. **Environmental justice analysis** -- Compare pollution exposure across income quintiles, racial/ethnic groups, deprivation indices. Statistical tests (Kruskal-Wallis, Mann-Whitney) and Lorenz curve / Gini coefficient.
10. **Temporal trends** -- Analyze annual trends, assess whether pollution is improving/worsening and whether equity gaps are narrowing.
11. **Report** -- Pollution maps, LUR model summary, exposure statistics, EJ findings, policy recommendations.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Data download | `openaq` (API), `requests`, `pandas` | `ropenaq`, `httr` | Manual download |
| Interpolation | `pykrige`, `scipy.interpolate`, `verde` | `gstat`, `automap` | QGIS, ArcGIS |
| LUR modeling | `scikit-learn` (LASSO, Ridge, RF), `statsmodels` | `lm`, `glmnet`, `caret` | Custom / R |
| Buffer predictors | `geopandas`, `shapely`, `rasterstats` | `sf`, `exactextractr` | QGIS buffers |
| Population overlay | `rasterstats`, `geopandas` | `exactextractr`, `sf` | Zonal Statistics |
| Health impact | Custom (relative risk functions) | `HRAPIE`, custom | BenMAP (EPA) |
| EJ analysis | `scipy.stats`, `numpy`, `matplotlib` | `ineq`, `stats` | Custom / Excel |
| Visualization | `matplotlib`, `seaborn`, `folium` | `tmap`, `ggplot2` | QGIS, ArcGIS |

### Code: Kriging, LUR Model, and Environmental Justice Analysis in Python

```python
"""
Air Quality Assessment — Kriging + LUR + Population Exposure + EJ Analysis
Requirements: pip install numpy pandas geopandas scikit-learn pykrige rasterstats scipy matplotlib
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from pykrige.ok import OrdinaryKriging
from rasterstats import zonal_stats
from scipy import stats
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# 1. Load and QA/QC monitoring data
# -------------------------------------------------------------------
monitors = gpd.read_file("data/air_quality_monitors.gpkg")
# Expected columns: station_id, geometry (point), pm25_annual (ug/m3), no2_annual (ug/m3)

# Basic QA: remove invalid readings
monitors = monitors[(monitors["pm25_annual"] > 0) & (monitors["pm25_annual"] < 200)]
monitors = monitors.dropna(subset=["pm25_annual"])
monitors = monitors.to_crs("EPSG:32610")  # UTM zone 10N

print(f"Monitoring stations after QA: {len(monitors)}")
print(f"PM2.5 range: {monitors['pm25_annual'].min():.1f} - "
      f"{monitors['pm25_annual'].max():.1f} ug/m3")
print(f"PM2.5 mean:  {monitors['pm25_annual'].mean():.1f} ug/m3")

# -------------------------------------------------------------------
# 2. Ordinary Kriging interpolation
# -------------------------------------------------------------------
x = monitors.geometry.x.values
y = monitors.geometry.y.values
z = monitors["pm25_annual"].values

# Create prediction grid
grid_res = 100  # metres
bounds = monitors.total_bounds
grid_x = np.arange(bounds[0], bounds[2], grid_res)
grid_y = np.arange(bounds[1], bounds[3], grid_res)

# Fit variogram and krige
OK = OrdinaryKriging(
    x, y, z,
    variogram_model="spherical",
    verbose=False,
    enable_plotting=False,
    nlags=20
)

z_pred, z_var = OK.execute("grid", grid_x, grid_y)
z_pred = z_pred.data  # (ny, nx) array

print(f"\nKriging grid: {z_pred.shape}")
print(f"Predicted PM2.5 range: {np.nanmin(z_pred):.1f} - {np.nanmax(z_pred):.1f}")

# LOOCV for kriging
loo = LeaveOneOut()
kriging_errors = []
for train_idx, test_idx in loo.split(z):
    ok_cv = OrdinaryKriging(
        x[train_idx], y[train_idx], z[train_idx],
        variogram_model="spherical", verbose=False, enable_plotting=False
    )
    z_cv, _ = ok_cv.execute("points", x[test_idx], y[test_idx])
    kriging_errors.append((z[test_idx[0]], z_cv.data.flatten()[0]))

kriging_errors = np.array(kriging_errors)
kriging_r2 = r2_score(kriging_errors[:, 0], kriging_errors[:, 1])
kriging_rmse = np.sqrt(mean_squared_error(kriging_errors[:, 0], kriging_errors[:, 1]))
print(f"Kriging LOOCV: R2={kriging_r2:.3f}, RMSE={kriging_rmse:.2f} ug/m3")

# -------------------------------------------------------------------
# 3. Extract LUR predictors at monitoring sites
# -------------------------------------------------------------------
roads = gpd.read_file("data/roads.gpkg").to_crs(monitors.crs)
landcover = "data/landcover.tif"

# Buffer-based predictor extraction
buffer_radii = [100, 300, 500, 1000]  # metres

for radius in buffer_radii:
    # Road length within buffer
    buffered = monitors.copy()
    buffered["geometry"] = monitors.buffer(radius)
    clipped_roads = gpd.overlay(roads, buffered, how="intersection")
    road_lengths = clipped_roads.groupby("station_id").geometry.apply(
        lambda x: x.length.sum()
    )
    monitors[f"road_len_{radius}m"] = monitors["station_id"].map(road_lengths).fillna(0)

    # Major road length (highway types)
    major = clipped_roads[clipped_roads["highway"].isin(
        ["motorway", "trunk", "primary", "secondary"]
    )]
    major_lengths = major.groupby("station_id").geometry.apply(lambda x: x.length.sum())
    monitors[f"major_road_{radius}m"] = monitors["station_id"].map(major_lengths).fillna(0)

    # Green space fraction
    lc_stats = zonal_stats(
        buffered, landcover, categorical=True, nodata=0
    )
    for i, s in enumerate(lc_stats):
        total = sum(s.values()) if s else 1
        green_classes = [10, 20, 30]  # tree, shrub, grassland
        green_count = sum(s.get(c, 0) for c in green_classes)
        monitors.loc[monitors.index[i], f"green_frac_{radius}m"] = green_count / total if total > 0 else 0

    # Population density
    pop_stats = zonal_stats(buffered, "data/population_100m.tif", stats=["sum"])
    buffer_area_km2 = np.pi * (radius / 1000) ** 2
    monitors[f"pop_density_{radius}m"] = [
        (s["sum"] / buffer_area_km2) if s["sum"] else 0 for s in pop_stats
    ]

# Elevation
elev_stats = zonal_stats(monitors, "data/dem.tif", stats=["mean"])
monitors["elevation"] = [s["mean"] for s in elev_stats]

# Distance to nearest industrial area
industrial = gpd.read_file("data/industrial_zones.gpkg").to_crs(monitors.crs)
monitors["dist_industrial"] = monitors.geometry.apply(
    lambda pt: industrial.distance(pt).min()
)

print(f"\nLUR predictors extracted: {len([c for c in monitors.columns if c not in ['geometry', 'station_id', 'pm25_annual', 'no2_annual']])} variables")

# -------------------------------------------------------------------
# 4. Build Land Use Regression model
# -------------------------------------------------------------------
predictor_cols = [c for c in monitors.columns if any(
    c.startswith(p) for p in ["road_len", "major_road", "green_frac",
                               "pop_density", "elevation", "dist_industrial"]
)]

X = monitors[predictor_cols].values
y_pm25 = monitors["pm25_annual"].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# LASSO with cross-validation for automatic feature selection
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y_pm25)

# Selected features (non-zero coefficients)
selected = np.array(predictor_cols)[lasso.coef_ != 0]
print(f"\n=== LUR Model (LASSO) ===")
print(f"Selected predictors ({len(selected)}):")
for feat, coef in sorted(zip(predictor_cols, lasso.coef_), key=lambda x: abs(x[1]), reverse=True):
    if coef != 0:
        print(f"  {feat:>25s}: {coef:>8.4f}")
print(f"Alpha: {lasso.alpha_:.6f}")

# LOOCV evaluation
loo = LeaveOneOut()
y_pred_loo = np.zeros_like(y_pm25)
for train_idx, test_idx in loo.split(X_scaled):
    lasso_cv = LassoCV(cv=3, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled[train_idx], y_pm25[train_idx])
    y_pred_loo[test_idx] = lasso_cv.predict(X_scaled[test_idx])

lur_r2 = r2_score(y_pm25, y_pred_loo)
lur_rmse = np.sqrt(mean_squared_error(y_pm25, y_pred_loo))
print(f"\nLUR LOOCV: R2={lur_r2:.3f}, RMSE={lur_rmse:.2f} ug/m3")
print(f"vs Kriging: R2={kriging_r2:.3f}, RMSE={kriging_rmse:.2f} ug/m3")

# -------------------------------------------------------------------
# 5. Apply LUR to prediction grid (100m resolution)
# -------------------------------------------------------------------
# Load gridded predictor layers (pre-computed at 100m resolution)
# For brevity, assume we have a stacked predictor raster
import rasterio

with rasterio.open("data/lur_predictors_stack.tif") as src:
    predictor_stack = src.read()  # (n_bands, rows, cols)
    pred_profile = src.profile.copy()
    n_bands, n_rows, n_cols = predictor_stack.shape

# Reshape, scale, predict
X_grid = predictor_stack.reshape(n_bands, -1).T  # (pixels, bands)
X_grid_scaled = scaler.transform(X_grid)
pm25_predicted = lasso.predict(X_grid_scaled).reshape(n_rows, n_cols)

# Clip to reasonable range
pm25_predicted = np.clip(pm25_predicted, 0, 100)

# Save prediction surface
out_profile = pred_profile.copy()
out_profile.update(count=1, dtype="float64", nodata=-9999)
with rasterio.open("output/pm25_lur_surface.tif", "w", **out_profile) as dst:
    dst.write(pm25_predicted, 1)

print(f"\nLUR surface saved. Range: {pm25_predicted.min():.1f} - {pm25_predicted.max():.1f} ug/m3")

# -------------------------------------------------------------------
# 6. Population exposure estimation
# -------------------------------------------------------------------
census_tracts = gpd.read_file("data/census_tracts.gpkg").to_crs(monitors.crs)

# Zonal mean PM2.5 per tract
pm25_stats = zonal_stats(census_tracts, "output/pm25_lur_surface.tif", stats=["mean"])
census_tracts["pm25_mean"] = [s["mean"] if s["mean"] else np.nan for s in pm25_stats]

# Population-weighted exposure
pop_stats = zonal_stats(census_tracts, "data/population_100m.tif", stats=["sum"])
census_tracts["population"] = [s["sum"] if s["sum"] else 0 for s in pop_stats]

total_pop = census_tracts["population"].sum()
pwe = (census_tracts["pm25_mean"] * census_tracts["population"]).sum() / total_pop

print(f"\n=== Population Exposure ===")
print(f"Population-weighted mean PM2.5: {pwe:.1f} ug/m3")
print(f"Unweighted spatial mean PM2.5:  {census_tracts['pm25_mean'].mean():.1f} ug/m3")
print(f"WHO guideline (annual PM2.5):    5.0 ug/m3")
print(f"People above WHO guideline:     "
      f"{census_tracts.loc[census_tracts['pm25_mean'] > 5, 'population'].sum():,.0f} "
      f"({census_tracts.loc[census_tracts['pm25_mean'] > 5, 'population'].sum()/total_pop*100:.1f}%)")

# -------------------------------------------------------------------
# 7. Health impact assessment (simplified)
# -------------------------------------------------------------------
# Attributable fraction using concentration-response from GBD / WHO
# All-cause mortality RR per 10 ug/m3 PM2.5 increase: ~1.08 (WHO 2021)
BASELINE_MORTALITY_RATE = 8.0 / 1000  # per year (crude example)
RR_PER_10 = 1.08
COUNTERFACTUAL = 5.0  # WHO guideline as counterfactual

census_tracts["excess_pm25"] = np.maximum(census_tracts["pm25_mean"] - COUNTERFACTUAL, 0)
census_tracts["rr"] = RR_PER_10 ** (census_tracts["excess_pm25"] / 10)
census_tracts["af"] = 1 - 1 / census_tracts["rr"]  # attributable fraction
census_tracts["attributable_deaths"] = (
    census_tracts["af"] * BASELINE_MORTALITY_RATE * census_tracts["population"]
)

total_deaths = census_tracts["attributable_deaths"].sum()
print(f"\n=== Health Impact (simplified) ===")
print(f"Estimated attributable deaths from PM2.5: {total_deaths:.0f} per year")
print(f"  (Using RR={RR_PER_10} per 10 ug/m3, counterfactual={COUNTERFACTUAL} ug/m3)")

# -------------------------------------------------------------------
# 8. Environmental justice analysis
# -------------------------------------------------------------------
# Load demographic data
demo = gpd.read_file("data/demographics.gpkg")
census_tracts = census_tracts.merge(
    demo[["tract_id", "median_income", "pct_minority", "pct_children", "deprivation_index"]],
    on="tract_id", how="left"
)

# PM2.5 by income quintile
census_tracts["income_quintile"] = pd.qcut(
    census_tracts["median_income"], q=5,
    labels=["Q1 (lowest)", "Q2", "Q3", "Q4", "Q5 (highest)"]
)

print("\n=== Environmental Justice: PM2.5 by Income Quintile ===")
ej_summary = census_tracts.groupby("income_quintile").agg(
    pm25_mean=("pm25_mean", "mean"),
    pm25_pop_weighted=("pm25_mean", lambda x: np.average(
        x, weights=census_tracts.loc[x.index, "population"]
    )),
    population=("population", "sum"),
    attr_deaths=("attributable_deaths", "sum")
).round(2)
print(ej_summary.to_string())

# Statistical test: do lower-income areas have higher PM2.5?
groups = [group["pm25_mean"].dropna().values
          for _, group in census_tracts.groupby("income_quintile")]
H_stat, p_value = stats.kruskal(*groups)
print(f"\nKruskal-Wallis: H={H_stat:.2f}, p={p_value:.4f}")

# Spearman correlation: income vs PM2.5
rho, p_rho = stats.spearmanr(
    census_tracts["median_income"].dropna(),
    census_tracts.loc[census_tracts["median_income"].notna(), "pm25_mean"]
)
print(f"Spearman (income vs PM2.5): rho={rho:.3f}, p={p_rho:.4f}")

# PM2.5 by minority percentage (above/below median)
median_minority = census_tracts["pct_minority"].median()
high_minority = census_tracts[census_tracts["pct_minority"] >= median_minority]
low_minority = census_tracts[census_tracts["pct_minority"] < median_minority]

U_stat, p_mw = stats.mannwhitneyu(
    high_minority["pm25_mean"].dropna(),
    low_minority["pm25_mean"].dropna(),
    alternative="greater"
)
print(f"\nMann-Whitney (high vs low minority % PM2.5): U={U_stat:.0f}, p={p_mw:.4f}")
print(f"  High minority mean PM2.5: {high_minority['pm25_mean'].mean():.1f} ug/m3")
print(f"  Low minority mean PM2.5:  {low_minority['pm25_mean'].mean():.1f} ug/m3")

# -------------------------------------------------------------------
# 9. Lorenz curve and Gini coefficient for pollution equity
# -------------------------------------------------------------------
def lorenz_gini(values, weights=None):
    """Compute Lorenz curve and Gini coefficient."""
    if weights is None:
        weights = np.ones_like(values)
    sorted_idx = np.argsort(values)
    sorted_vals = values[sorted_idx]
    sorted_weights = weights[sorted_idx]

    cum_weights = np.cumsum(sorted_weights) / sorted_weights.sum()
    cum_pollution = np.cumsum(sorted_vals * sorted_weights) / (sorted_vals * sorted_weights).sum()

    # Gini = 1 - 2 * area under Lorenz curve
    gini = 1 - 2 * np.trapz(cum_pollution, cum_weights)
    return cum_weights, cum_pollution, gini

valid = census_tracts.dropna(subset=["pm25_mean", "population"])
cum_pop, cum_pm25, gini = lorenz_gini(
    valid["pm25_mean"].values,
    valid["population"].values
)

print(f"\n=== Pollution Equity ===")
print(f"Gini coefficient (PM2.5 exposure): {gini:.3f}")
print(f"  0 = perfectly equal, 1 = perfectly unequal")

census_tracts.to_file("output/air_quality_ej_results.gpkg", driver="GPKG")
```

### Expected Output

- **Kriging interpolation surface** (continuous PM2.5 / NO2 map from monitoring data)
- **LUR model summary** (selected predictors, coefficients, LOOCV R2 and RMSE)
- **LUR prediction surface** (100 m resolution PM2.5 map)
- **Population exposure statistics** (population-weighted mean, % above WHO guideline)
- **Health impact estimate** (attributable deaths from excess PM2.5 exposure)
- **Environmental justice analysis** (PM2.5 by income/minority quintile, statistical tests)
- **Lorenz curve and Gini coefficient** (pollution equity metric)
- **Trend analysis** (annual PM2.5 trends, equity gap trends)
- **Report** with maps, model diagnostics, exposure tables, EJ findings, policy recommendations

### Flowchart

```
[Monitoring Station Data (PM2.5, NO2)]
              |
              +------------------+
              |                  |
              v                  v
[Kriging Interpolation]   [LUR Predictor Extraction]
[Variogram fitting]       [Road length, green space,
[LOOCV validation]         pop density, elevation
              |             per buffer radius]
              |                  |
              |                  v
              |           [LASSO / Stepwise LUR]
              |           [Feature selection]
              |           [LOOCV validation]
              |                  |
              |                  v
              |           [Apply LUR to Grid]
              |           [100m PM2.5 surface]
              |                  |
              +------------------+
                       |
                       v
            [Pollution Surface (best model)]
                       |
          +------------+------------+
          |            |            |
          v            v            v
   [Population    [Health       [Environmental
    Exposure]      Impact]       Justice]
   [Pop-weighted  [Conc-Response [By income, race]
    mean PM2.5]    function]     [Kruskal-Wallis]
   [% above WHO]  [Attributable  [Lorenz / Gini]
                    deaths]           |
          |            |              |
          +------------+--------------+
                       |
                       v
              [Trend Analysis]
              [Annual changes, equity gaps]
                       |
                       v
                [Report & Maps]
```

---

[Back to Data Analysis](README.md) | [Back to Main README](../README.md)
