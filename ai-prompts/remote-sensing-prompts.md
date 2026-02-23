# Remote Sensing Prompts

> Prompt templates for satellite image classification, spectral index calculation, change detection, preprocessing, and Google Earth Engine code generation.

> **Quick Picks**
> - 🏆 **Most Useful**: [Supervised Classification Workflow](#prompt-1--supervised-classification-workflow) — end-to-end classification with accuracy assessment
> - 🚀 **Time Saver**: [GEE Time-Series Animation](#prompt-4--gee-time-series-animation) — produce animated GIFs of change over time in minutes
> - 🆕 **Cutting Edge**: [SAM for Geospatial Segmentation](#prompt-4--segment-anything-model-sam-for-geospatial) — zero-shot feature extraction using foundation models

---

## Table of Contents

- [Image Classification](#image-classification)
- [NDVI & Index Calculation](#ndvi--index-calculation)
- [Change Detection](#change-detection)
- [Cloud Masking & Preprocessing](#cloud-masking--preprocessing)
- [Google Earth Engine](#google-earth-engine)
- [Foundation Models & Deep Learning](#foundation-models--deep-learning)

---

## Image Classification

### Prompt 1 — Supervised Classification Workflow

**Context:** You have a multispectral satellite image and ground truth training samples, and you want to classify land cover types.

**Template:**

```
I have a [sensor, e.g., Sentinel-2 / Landsat 8] multispectral image covering [study area].
Bands available: [list bands, e.g., B2 Blue, B3 Green, B4 Red, B8 NIR, B11 SWIR1, B12 SWIR2].
I also have training samples in a [format, e.g., GeoPackage] with a "class" column containing
these land cover types: [list classes, e.g., Water, Forest, Cropland, Urban, Bare Soil].

Write a Python script using Rasterio (1.3+) and Scikit-learn (1.3+) to:
1. Load the image bands and stack them into a multiband array
2. Extract pixel values at training sample locations
3. Split into train/test (70/30) with stratification
4. Train a Random Forest classifier with hyperparameter tuning (GridSearchCV)
5. Classify the full image
6. Generate accuracy metrics: overall accuracy, kappa, per-class F1, confusion matrix
7. Export the classified raster as a GeoTIFF with a color table
8. Export the confusion matrix as a CSV

Include comments explaining each step for a GIS analyst learning ML.
```

**Variables to customize:**
- Sensor and band list
- Training sample format and class names
- Classifier (Random Forest, SVM, XGBoost)
- Train/test split ratio

**Expected output format:** Complete Python script with accuracy report and classified GeoTIFF output.

---

### Prompt 2 — Unsupervised Classification (Clustering)

**Context:** You do not have training data and want to explore natural spectral clusters in an image.

**Template:**

```
I have a [sensor] image with bands [list bands]. I want to perform unsupervised classification
to identify [N, e.g., 5-10] spectral clusters.

Write a Python script that:
1. Loads and stacks the bands
2. Masks NoData and cloud pixels (using [mask band/file] if available)
3. Samples [percentage]% of pixels for efficiency
4. Runs K-Means clustering with the elbow method to suggest optimal K
5. Classifies all pixels using the chosen K
6. Exports the clustered raster
7. Creates a quick visualization with distinct colors per cluster

Suggest how I should manually interpret and label each cluster afterward.
```

**Variables to customize:**
- Sensor and bands
- Number of clusters or range to test
- Masking approach
- Sampling percentage

**Expected output format:** Python script plus guidance on cluster interpretation.

---

### Prompt 3 — Deep Learning Segmentation Setup

**Context:** You want to use a pre-trained deep learning model (e.g., U-Net) for land cover segmentation.

**Template:**

```
I want to perform semantic segmentation on [sensor] imagery for [task, e.g., building footprint extraction].
Image resolution: [X] meters. Study area size: approximately [W] x [H] km.

Using TorchGeo (0.5+) and PyTorch (2.0+) (or Raster Vision 0.21+), write a Python workflow that:
1. Tiles the input image into [size, e.g., 256x256] patches with [overlap]% overlap
2. Loads a pre-trained [model, e.g., U-Net with ResNet50 backbone] from TorchGeo
3. Runs inference on all patches (GPU if available, CPU fallback)
4. Merges predictions back into a single raster, handling overlap via averaging
5. Applies a confidence threshold of [X] and converts to vector polygons
6. Exports both the prediction raster and the vector result

List the hardware requirements and expected processing time.
```

**Variables to customize:**
- Sensor, resolution, study area size
- Task (buildings, roads, trees, etc.)
- Model architecture and backbone
- Tile size and overlap

**Expected output format:** Python script with environment setup instructions and hardware guidance.

---

## NDVI & Index Calculation

### Prompt 1 — Multi-Index Calculation

**Context:** You need to calculate several spectral indices from a single satellite image for environmental monitoring.

**Template:**

```
I have a [sensor, e.g., Sentinel-2 Level-2A] image. Write a Python script using Rasterio (1.3+) and NumPy (1.24+) to calculate
the following indices and save each as a separate single-band GeoTIFF:

1. NDVI (Normalized Difference Vegetation Index): (NIR - Red) / (NIR + Red)
2. NDWI (Normalized Difference Water Index): (Green - NIR) / (Green + NIR)
3. NDBI (Normalized Difference Built-up Index): (SWIR1 - NIR) / (SWIR1 + NIR)
4. EVI (Enhanced Vegetation Index): 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)

Band mapping for [sensor]:
- Blue: [band], Green: [band], Red: [band], NIR: [band], SWIR1: [band]

Requirements:
- Handle division by zero (output NoData)
- Preserve the original CRS and transform
- Set output dtype to float32
- Add descriptive band names in metadata
- Create a composite false-color visualization (NIR-Red-Green) as a PNG
```

**Variables to customize:**
- Sensor and band mapping
- Which indices to calculate
- Output format preferences

**Expected output format:** Python script producing individual index GeoTIFFs and a composite PNG.

---

### Prompt 2 — Time Series NDVI Analysis

**Context:** You have a collection of images over time and want to analyze vegetation phenology.

**Template:**

```
I have [N] [sensor] images from [date range] covering [study area].
They are stored as individual GeoTIFFs in [directory].

Write a Python script to:
1. Calculate NDVI for each image
2. Stack all NDVI rasters into a time series (handle slight alignment differences)
3. For each pixel, compute: mean, max, min, standard deviation, date of max NDVI
4. Generate a time series plot for [N] sample points at coordinates [list]
5. Export the summary statistics as multi-band GeoTIFF
6. Identify pixels where NDVI dropped below [threshold] (potential deforestation/drought)

Use xarray (2024.1+) and rioxarray (0.15+) for efficient handling.
```

**Variables to customize:**
- Number of images, date range, sensor
- Sample point coordinates
- Threshold for anomaly detection

**Expected output format:** Python script with time series plots and multi-band summary raster.

---

## Change Detection

### Prompt 1 — Bi-Temporal Change Detection

**Context:** You have two images from different dates and want to identify areas of change.

**Template:**

```
I have two [sensor] images of [study area]:
- Before: [date], file: [path]
- After: [date], file: [path]

Both are atmospherically corrected (Level-2A) with CRS [EPSG code].

Write a Python script to perform change detection using:
1. Image differencing (NDVI difference)
2. Change Vector Analysis (CVA) using [bands]
3. Threshold determination using Otsu's method
4. Classification of change into: [categories, e.g., vegetation loss, vegetation gain, urban expansion, no change]
5. Accuracy assessment if reference data is available at [path]

Output:
- Change magnitude raster
- Change classification raster
- Change statistics table (area per category in hectares)
- Before/after/change RGB composite visualization
```

**Variables to customize:**
- Sensor and dates
- Change categories of interest
- Threshold method (Otsu, manual, statistical)
- Reference data availability

**Expected output format:** Python script with classified change raster and area statistics.

---

### Prompt 2 — Multi-Temporal Trend Analysis

**Context:** You want to detect long-term trends over many years (e.g., deforestation, urbanization).

**Template:**

```
I have annual [index, e.g., NDVI / NDBI] composites from [start year] to [end year]
for [study area], stored as [format] files.

Write a Python script that:
1. Loads all annual composites into an xarray DataArray with a time dimension
2. For each pixel, fits a linear regression (value ~ year)
3. Extracts the slope, intercept, R-squared, and p-value per pixel
4. Classifies significant trends (p < 0.05): increasing, decreasing, stable
5. Calculates total absolute change over the period
6. Exports trend classification raster and slope raster
7. Creates a map showing significant trends with appropriate color scheme

Handle NoData values and ensure at least [N] valid observations per pixel for reliable trend fitting.
```

**Variables to customize:**
- Index type, date range
- Significance threshold
- Minimum valid observation count

**Expected output format:** Python script with trend rasters and visualization.

---

## Cloud Masking & Preprocessing

### Prompt 1 — Sentinel-2 Cloud Masking

**Context:** You downloaded Sentinel-2 imagery and need to apply cloud and shadow masking before analysis.

**Template:**

```
I have a Sentinel-2 Level-2A product downloaded from [source, e.g., Copernicus Data Space].
The .SAFE directory is at [path].

Write a Python script to:
1. Read the SCL (Scene Classification Layer) band
2. Create a cloud mask that excludes: cloud high probability, cloud medium probability,
   thin cirrus, cloud shadows, and snow/ice (SCL classes 3, 8, 9, 10, 11)
3. Apply the mask to all 10m and 20m bands
4. Resample 20m bands to 10m using bilinear interpolation
5. Stack all masked bands into a single multi-band GeoTIFF
6. Report the percentage of valid (non-masked) pixels

If the cloud cover exceeds [threshold]%, warn the user and suggest downloading a different date.
```

**Variables to customize:**
- Product path
- SCL classes to mask
- Resampling method
- Cloud cover threshold

**Expected output format:** Python script producing a cloud-free multi-band stack.

---

### Prompt 2 — Landsat Collection 2 Preprocessing

**Context:** You need to preprocess Landsat Collection 2 Level-2 data including scaling, masking, and clipping.

**Template:**

```
I have Landsat [8/9] Collection 2 Level-2 Science Products for [study area].
Files are in [directory] with the standard USGS naming convention.

Write a Python script to:
1. Parse the metadata (MTL.json) for acquisition date, cloud cover, sun elevation
2. Apply the Collection 2 scale factors (multiply by 0.0000275, add -0.2 for SR bands)
3. Apply QA_PIXEL bit mask to remove cloud, cloud shadow, cirrus, snow
4. Clip to study area boundary from [boundary file]
5. Calculate surface reflectance for bands [list]
6. Export as a clipped, masked, properly scaled GeoTIFF

Handle the thermal bands separately (scale factor: 0.00341802, add 149.0 for Kelvin).
```

**Variables to customize:**
- Landsat version (8 or 9)
- Directory path
- Boundary file
- Bands of interest

**Expected output format:** Python script with metadata parsing and scaled/masked output.

---

## Google Earth Engine

### Prompt 1 — NDVI Time Series in GEE

**Context:** You want to generate a monthly NDVI time series for a region using Google Earth Engine's JavaScript API.

**Template:**

```
Write a Google Earth Engine JavaScript script for the Code Editor to:

1. Define a study area using [method: coordinates / asset / drawn geometry]
   - If coordinates: var roi = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax]);
   - If asset: var roi = ee.FeatureCollection('[asset path]');
   Coordinates: [lon, lat] or Asset ID: [path]

2. Filter the [Sentinel-2 SR (COPERNICUS/S2_SR_HARMONIZED) / Landsat 8/9 C2 SR] collection:
   - Date range: [start date] to [end date]
   - Bounds: .filterBounds(roi)
   - Cloud filter: .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', [threshold, e.g., 20]))

3. Apply cloud masking:
   - For Sentinel-2: use the SCL band — mask classes 3 (cloud shadow), 8, 9 (cloud), 10 (cirrus), 11 (snow)
   - Write a reusable maskS2clouds(image) function that returns image.updateMask(mask)

4. Calculate NDVI for each image using .map():
   - Sentinel-2: NDVI = (B8 - B4) / (B8 + B4)
   - Add as a band named 'NDVI' using .addBands() and .rename()

5. Create monthly median composites:
   - Use ee.List.sequence(1, 12) and .map() to generate monthly composites
   - Each composite = collection.filter(ee.Filter.calendarRange(month, month, 'month')).median()
   - Combine into an ee.ImageCollection

6. Generate a time series chart:
   - Use ui.Chart.image.seriesByRegion() or ui.Chart.image.series()
   - X-axis: date, Y-axis: mean NDVI
   - Set chart options: title, axis labels, line color, legend

7. Export monthly composites to Google Drive:
   - Use Export.image.toDrive() in a loop
   - Set scale: [resolution, e.g., 10] meters, CRS: [EPSG code], region: roi
   - File name pattern: 'NDVI_[YYYY]_[MM]'

Add to the Map: roi boundary (red outline), latest NDVI composite (palette: ['brown','yellow','green']), chart to console.
Set Map.centerObject(roi, [zoom level]).
```

**Variables to customize:**
- Study area definition method
- Sensor and date range
- Composite period (monthly, seasonal, annual)
- Export destination (Drive, Asset, Cloud Storage)

**Expected output format:** GEE JavaScript code ready to paste into the Code Editor.

---

### Prompt 2 — Land Cover Classification in GEE

**Context:** You want to classify land cover using GEE's built-in classifiers and training data.

**Template:**

```
Write a Google Earth Engine JavaScript script to:
1. Load [sensor, e.g., Sentinel-2] imagery for [year] over [study area]
2. Create a cloud-free annual median composite
3. Calculate and add spectral indices: NDVI, NDWI, NDBI
4. Use training points from [source: GEE asset / manually defined FeatureCollection]
   Classes: [list classes with numeric codes]
5. Sample the composite at training point locations
6. Train a [classifier: smileRandomForest / smileCart / libsvm] with [N] trees
7. Classify the composite
8. Assess accuracy using a [split: random 70/30 / separate validation set]
9. Print the confusion matrix, overall accuracy, and kappa
10. Export the classified image to Drive at [resolution] meters

Visualize with this palette: [list hex colors per class].
```

**Variables to customize:**
- Sensor, year, study area
- Training data source and class definitions
- Classifier and parameters
- Export resolution
- Color palette

**Expected output format:** GEE JavaScript code with classification, accuracy assessment, and export.

---

### Prompt 3 — Automated Flood Mapping in GEE

**Context:** You need a rapid flood mapping script using SAR data (Sentinel-1) for disaster response.

**Template:**

```
Write a Google Earth Engine JavaScript script for rapid flood mapping:

1. Define the area of interest: [coordinates or country/region name]
2. Define the flood event date: [date]
3. Load Sentinel-1 GRD (VV polarization, IW mode, descending orbit)
4. Create a pre-event composite: [date - 30 days] to [date - 5 days]
5. Create a during-event image: [date] to [date + 5 days]
6. Apply speckle filtering (focal median, radius 50m)
7. Detect floods using change detection: VV_before - VV_during > [threshold, e.g., 3 dB]
8. Mask out permanent water using JRC Global Surface Water (occurrence > 50%)
9. Mask out terrain shadow using SRTM DEM (slope > 15 degrees)
10. Calculate flooded area in km²
11. Export the flood extent as a vector (GeoJSON) and raster

Add layers to the Map: pre-event, during-event, flood extent (blue), permanent water (cyan).
```

**Variables to customize:**
- Area of interest
- Event date
- dB threshold
- Permanent water and slope thresholds

**Expected output format:** GEE JavaScript code for rapid deployment during flood events.

---

### Prompt 4 — GEE Time-Series Animation

**Context:** You want to create an animated GIF showing how a landscape changes over time (urban growth, deforestation, seasonal vegetation cycles) directly from Google Earth Engine.

**Template:**

```
Write a Google Earth Engine JavaScript script that generates a time-series animation (video/GIF):

Study area: [coordinates or asset ID]
Time period: [start year] to [end year]
Temporal step: [monthly / seasonal / annual]
Sensor: [Sentinel-2 / Landsat 8/9 / MODIS]

Script should:
1. Define the study area and center the map
2. Build an ImageCollection with one composite per time step:
   - For monthly: median composite per calendar month across the date range
   - For annual: median composite per year
   - Apply cloud masking before compositing
3. For each composite, create a visualization image:
   - Option A: True color RGB (e.g., B4/B3/B2 for Sentinel-2, scaled to [min, max])
   - Option B: False color (NIR/Red/Green) or single-index (NDVI with palette)
4. Add text annotations to each frame:
   - Use ee.Image.paint() or text overlay showing the date/year
   - Add a title bar at the top: "[Study Area] — [Variable] [Date]"
5. Export as video using Export.video.toDrive():
   - framesPerSecond: [fps, e.g., 4]
   - dimensions: [width, e.g., 1920]
   - region: roi
   - CRS: EPSG:3857 (for web display) or [EPSG code]
6. Also display the animation inline using ui.Thumbnail():
   - var animation = ui.Thumbnail({
       image: collection,
       params: {dimensions: 512, framesPerSecond: 2, region: roi},
       style: {position: 'bottom-right'}
     });
   - Add to the map or a side panel

Include a helper function to add timestamp text to each frame.
```

**Variables to customize:**
- Study area, time period, sensor
- Visualization type (true color, false color, index)
- Frame rate, dimensions, export settings
- Text annotation content

**Expected output format:** GEE JavaScript code that produces an inline animation preview and exports video to Google Drive.

---

### Prompt 5 — Sentinel-2 Super-Resolution with Deep Learning

**Context:** You want to enhance the spatial resolution of Sentinel-2 20m bands (e.g., Red Edge, SWIR) to 10m using deep learning super-resolution models, or pan-sharpen using the 10m bands.

**Template:**

```
Write a Python workflow to perform super-resolution on Sentinel-2 imagery, enhancing 20m bands to 10m.

Input: Sentinel-2 Level-2A product at [path]
Bands to super-resolve: B5 (Red Edge 1, 20m), B6 (Red Edge 2, 20m), B7 (Red Edge 3, 20m),
                         B8A (NIR narrow, 20m), B11 (SWIR1, 20m), B12 (SWIR2, 20m)
Reference 10m bands: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)

Approach — implement TWO methods and compare:

Method 1: Classical pan-sharpening (Brovey or Gram-Schmidt)
- Use Rasterio (1.3+) and NumPy (1.24+)
- Upsample 20m bands to 10m using bilinear interpolation
- Apply pan-sharpening using B8 (10m NIR) as the panchromatic reference
- Preserve spectral fidelity as much as possible

Method 2: Deep learning super-resolution
- Use the opensr-test or SR4RS framework (or a pre-trained ESPCN/EDSR model from TorchGeo)
- Load the pre-trained model weights (provide download instructions)
- Tile the image into patches, run inference, merge with overlap handling
- GPU recommended; include CPU fallback with batch size of 1

For both methods:
1. Read input bands using Rasterio, handle CRS and transform
2. Produce super-resolved output at 10m for all target bands
3. Stack all bands (original 10m + super-resolved) into a single multi-band GeoTIFF
4. Validate by:
   - Comparing spectral statistics (mean, std) of original vs super-resolved
   - Computing PSNR and SSIM against the original (downsampled from predicted)
   - Visual comparison: side-by-side PNG of a sample region
5. Export the final 10m all-band composite

List required packages with versions and estimated processing time per tile.
```

**Variables to customize:**
- Input product path
- Bands to super-resolve
- Super-resolution model choice
- Validation metrics
- Hardware (GPU/CPU)

**Expected output format:** Python script with both approaches, validation metrics, and comparison visualizations.

---

## Foundation Models & Deep Learning

### Prompt 4 — Segment Anything Model (SAM) for Geospatial

**Context:** You want to use Meta's Segment Anything Model (SAM) or its geospatial variants (Geo-SAM, SAM-Geo) for zero-shot or prompted segmentation of features in satellite/aerial imagery without training a custom model.

**Template:**

```
Write a Python workflow to segment [target features, e.g., "building footprints" / "agricultural fields" /
"water bodies" / "solar panels"] from [imagery type, e.g., "high-resolution aerial orthophoto at 0.3m" /
"Sentinel-2 10m RGB"] using the Segment Anything Model.

Image details:
- Path: [path to GeoTIFF]
- Resolution: [X] meters
- Bands: [list, e.g., RGB or RGB+NIR]
- CRS: [EPSG code]
- Approximate area: [W x H km]

Use the samgeo library (segment-geospatial 0.11+) which wraps SAM for geospatial data:

Part 1 — Automatic mask generation (no prompts):
1. Install: pip install segment-geospatial
2. Load the image and initialize SamGeo with model_type="vit_h" (or "vit_l" for faster inference)
3. Download the SAM checkpoint weights automatically
4. Run sam.generate(source=[image], output=[output.tif]) to generate all masks
5. Convert masks to vector polygons: sam.tiff_to_gpkg(output=[output.gpkg])
6. Filter polygons by:
   - Minimum area: [threshold] m² (remove tiny fragments)
   - Maximum area: [threshold] m² (remove background)
   - Compactness ratio (Polsby-Popper or convexity) to filter by shape

Part 2 — Prompted segmentation (point or box prompts):
1. Load the image in SamGeo
2. Provide point prompts from a GeoDataFrame of [source, e.g., "building centroids"]:
   - point_coords = [[lon1, lat1], [lon2, lat2], ...]
   - point_labels = [1, 1, ...] (1 = foreground)
3. Or provide bounding box prompts from [source, e.g., "detection results"]:
   - boxes = [[xmin, ymin, xmax, ymax], ...]
4. Run segmentation and export results as GeoPackage

Part 3 — Post-processing:
1. Merge overlapping polygons
2. Smooth boundaries (Douglas-Peucker or Visvalingam simplification, tolerance: [value] m)
3. Remove holes smaller than [threshold] m²
4. Calculate attributes: area_m2, perimeter_m, compactness
5. Optionally classify segments using mean spectral values per segment (object-based classification)
6. Export final result as GeoPackage with CRS preserved

Include hardware requirements, expected processing time, and memory usage estimates.
For very large images (>10,000 x 10,000 px), implement a tiling strategy with overlap.
```

**Variables to customize:**
- Target features and imagery type
- SAM model variant (vit_h, vit_l, vit_b, or SAM2)
- Point/box prompt sources
- Filtering thresholds (area, compactness)
- Post-processing parameters

**Expected output format:** Python script with automatic and prompted segmentation, post-processing, and vector export.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
