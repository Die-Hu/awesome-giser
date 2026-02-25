# Remote Sensing Prompts

> Expert-level prompt templates for image classification, spectral analysis, change detection, GEE workflows, LiDAR processing, and production delivery.

> **Quick Picks**
> - **Most Useful**: [P1 Supervised Classification](#prompt-1--supervised-classification) -- end-to-end RF/SVM/XGBoost with accuracy assessment
> - **Time Saver**: [P16 GEE NDVI Analysis](#prompt-16--gee-ndvi-analysis) -- complete GEE workflow with geemap in minutes
> - **Cutting Edge**: [P4 Foundation Model Zero-Shot](#prompt-4--foundation-model-zero-shot-segmentation) -- SAM2 and Prithvi-EO for zero-shot extraction

---

## Table of Contents

1. [Image Classification](#1-image-classification) (P1--P4)
2. [Spectral Analysis](#2-spectral-analysis) (P5--P8)
3. [Change Detection](#3-change-detection) (P9--P12)
4. [Cloud & Atmosphere](#4-cloud--atmosphere) (P13--P15)
5. [Google Earth Engine](#5-google-earth-engine) (P16--P20)
6. [LiDAR & Point Cloud](#6-lidar--point-cloud) (P21--P23)
7. [Production & Delivery](#7-production--delivery) (P24--P26)

---

## 1. Image Classification

---

### Prompt 1 -- Supervised Classification

#### Scenario (实际场景)

You have a multispectral satellite image and labeled ground truth samples. The goal is to produce a thematic land cover map with quantified accuracy using traditional machine learning classifiers (Random Forest, SVM, XGBoost). This is the bread-and-butter workflow for any remote sensing project that requires defensible accuracy metrics.

#### Roles

**User** = GIS analyst with training data; **AI** = remote sensing ML engineer writing production Python code.

#### Prompt Template

```
You are a senior remote sensing scientist. I need a production-grade supervised
classification pipeline.

INPUTS
- Satellite image: [sensor, e.g., Sentinel-2 Level-2A / Landsat 9 C2L2]
- Bands: [list, e.g., B2 Blue, B3 Green, B4 Red, B8 NIR, B11 SWIR1, B12 SWIR2]
- Training samples: [format, e.g., GeoPackage] with column "[class_col]"
  containing classes: [list, e.g., Water, Forest, Cropland, Urban, Bare Soil]
- CRS: [EPSG code]

TASK -- write a Python script (rasterio 1.3+, scikit-learn 1.4+, numpy 1.26+):
1. Load and stack bands; apply NoData mask.
2. Extract pixel values at training locations; add engineered features:
   NDVI, NDWI, NDBI, GLCM texture (contrast, homogeneity) via scikit-image.
3. Stratified train/test split ([split_ratio, e.g., 70/30]).
4. Train three classifiers -- Random Forest (n=500), SVM (RBF), XGBoost --
   with 5-fold GridSearchCV. Select the best by Cohen's kappa.
5. Classify the full image with the winning model.
6. Output accuracy: OA, kappa, per-class F1, confusion matrix heatmap (PNG).
7. Export classified raster as GeoTIFF with embedded color table.
8. Export confusion matrix as CSV and feature importance as bar chart PNG.

CONSTRAINTS
- Chunk large rasters (>10 000 x 10 000 px) with windowed reads.
- Reproducible: set random_state=42 everywhere.
- Include logging with timestamps.
```

#### Variables to Customize

- Sensor and band list
- Training sample path, format, and class column name
- Target classes and numeric codes
- Classifier shortlist and hyperparameter grids
- Train/test split ratio

#### Expected Output

A single Python script that ingests raw imagery and training data, then produces a classified GeoTIFF, accuracy CSV, confusion matrix heatmap, and feature importance chart. All outputs are georeferenced and reproducible.

#### Validation Checklist

- [ ] Overall accuracy and kappa are printed and saved
- [ ] Confusion matrix includes both counts and percentages
- [ ] Output GeoTIFF has correct CRS, transform, and color table
- [ ] Windowed reading handles images larger than available RAM
- [ ] random_state is set for all stochastic operations

#### Cost Optimization

Run GridSearchCV on a 10% stratified subsample first; apply the best hyperparameters to the full dataset to cut compute by roughly 10x.

#### Dark Arts Tip

Slip `class_weight='balanced'` into RF and SVM to silently fix the imbalanced-class problem that ruins most land cover maps -- then report the improvement as if it were obvious all along.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- imagery acquisition
- [../tools/ai-ml-geospatial.md](../tools/ai-ml-geospatial.md) -- ML tooling reference

#### Extensibility

Swap the classifier block for a LightGBM or CatBoost model by changing only the estimator dict. Add SAR backscatter bands for improved urban/water discrimination.

---

### Prompt 2 -- Deep Learning Segmentation

#### Scenario (实际场景)

You need pixel-level semantic segmentation (buildings, roads, tree canopy) at scale using deep learning. Traditional classifiers struggle with spatial context; convolutional and transformer architectures capture neighborhood patterns. This prompt targets U-Net, DeepLabV3+, and SegFormer via TorchGeo with spatial cross-validation to avoid data leakage.

#### Roles

**User** = project lead with labeled raster masks; **AI** = DL engineer writing a training and inference pipeline.

#### Prompt Template

```
You are a deep learning engineer specializing in Earth observation.

INPUTS
- Imagery: [sensor] at [resolution] m, study area ~[W x H] km
- Labels: raster mask with [N] classes at [path]
- Hardware: [GPU model / CPU-only]

TASK -- Python (TorchGeo 0.6+, PyTorch 2.2+, rasterio 1.3+):
1. Tile imagery + labels into [size, e.g., 256x256] patches, [overlap]% overlap.
   Use spatial cross-validation (group-k-fold by tile grid) to prevent leakage.
2. Build a data pipeline with TorchGeo RasterDataset + RandomGeoSampler.
3. Train [model: U-Net / DeepLabV3+ / SegFormer] with [backbone, e.g., ResNet-50]
   pretrained on ImageNet (or Satlas/SSL4EO weights if available).
   - Loss: CrossEntropy + Dice (weighted sum).
   - Optimizer: AdamW, lr=1e-4, cosine annealing.
   - Epochs: [N]; early stopping patience [N].
4. Evaluate: per-class IoU, mIoU, F1, confusion matrix.
5. Run inference on full image with sliding window + Gaussian weighting for seams.
6. Post-process: morphological opening, min-area filter [threshold] m^2.
7. Vectorize to GeoPackage; export prediction raster as COG.

CONSTRAINTS
- Mixed-precision (fp16) training if GPU supports it.
- Log to TensorBoard or W&B.
- Print estimated training time and peak VRAM.
```

#### Variables to Customize

- Sensor, resolution, study area extent
- Number of classes and label format
- Model architecture, backbone, pretrained weights
- Tile size, overlap, batch size
- Hardware constraints (GPU VRAM)

#### Expected Output

A reproducible training script, an inference script, and evaluation metrics. Final outputs include a prediction raster (COG) and a vectorized GeoPackage with per-polygon class and confidence.

#### Validation Checklist

- [ ] Spatial cross-validation groups are non-overlapping geographically
- [ ] mIoU and per-class IoU are reported on the held-out spatial fold
- [ ] Inference handles images larger than GPU memory via sliding window
- [ ] Output raster CRS matches input CRS exactly
- [ ] Mixed-precision fallback to fp32 if CUDA version is insufficient

#### Cost Optimization

Start with SegFormer-B0 (3.7M params) for rapid prototyping; upgrade to B5 only if mIoU gain exceeds 3 points on validation.

#### Dark Arts Tip

Use test-time augmentation (TTA) -- flip and rotate inputs, average predictions -- to boost mIoU by 1-3 points at the cost of 8x inference time. Report the TTA score as your headline number.

#### Related Pages

- [../tools/ai-ml-geospatial.md](../tools/ai-ml-geospatial.md) -- TorchGeo, Raster Vision, model zoo

#### Extensibility

Replace the backbone with a vision transformer (ViT-L) fine-tuned on remote sensing (e.g., SatMAE) for state-of-the-art results on sub-meter imagery.

---

### Prompt 3 -- Object-Based Image Analysis (OBIA)

#### Scenario (实际场景)

Pixel-based classifiers produce salt-and-pepper noise. OBIA first segments the image into homogeneous objects, then classifies those objects using shape, texture, and spectral statistics. This is standard practice for high-resolution imagery (< 2 m) where individual pixels are smaller than the features of interest.

#### Roles

**User** = GIS analyst with high-res imagery; **AI** = OBIA specialist writing a segmentation-then-classification pipeline.

#### Prompt Template

```
You are an OBIA remote sensing specialist.

INPUTS
- Image: [sensor/source] at [resolution] m, path: [path]
- Training polygons (optional): [path], column: "[class_col]"
- Target objects: [e.g., tree crowns, buildings, agricultural parcels]

TASK -- Python (scikit-image 0.22+, rasterio, geopandas, scikit-learn):
1. Load image; compute additional layers: NDVI, NDWI, Gabor texture (4 orientations).
2. Multi-scale segmentation:
   - SLIC superpixels (n_segments=[N], compactness=[C]) or
   - Felzenszwalb (scale=[S], sigma=0.5, min_size=[M]).
   Compare both; select by under-segmentation error vs. over-segmentation error.
3. For each segment compute zonal statistics:
   mean, std, median per band; GLCM contrast/homogeneity; area; perimeter;
   compactness (4*pi*area/perimeter^2); rectangularity; elongation.
4. If training data provided: extract segment features at labeled locations,
   train RF classifier, classify all segments, report accuracy.
   If no training data: run K-Means on segment features, output cluster map.
5. Vectorize classified segments to GeoPackage with all attributes.
6. Export classified raster with majority class per segment.

CONSTRAINTS
- Segments must respect image edges (no border artifacts).
- Handle multi-band images with >4 bands gracefully.
```

#### Variables to Customize

- Image source and resolution
- Segmentation algorithm and parameters
- Target objects and class scheme
- Feature set (spectral, textural, geometric)

#### Expected Output

A GeoPackage of classified segments with attribute table (spectral means, shape metrics, predicted class) and a classified raster. If unsupervised, a cluster map with suggested interpretations.

#### Validation Checklist

- [ ] Segmentation boundaries visually align with real-world object edges
- [ ] At least 5 shape metrics are computed per segment
- [ ] Segment attribute table is complete (no NaN in required columns)
- [ ] Accuracy report includes per-class metrics if training data is provided

#### Cost Optimization

Downsample to 2x native resolution for segmentation parameter tuning, then apply optimal parameters at full resolution for the final run.

#### Dark Arts Tip

Run segmentation at two scales (coarse + fine), then use the coarse segments as a spatial prior to regularize the fine-scale classification -- this dramatically reduces noise without explicit post-processing.

#### Related Pages

- [../tools/ai-ml-geospatial.md](../tools/ai-ml-geospatial.md) -- scikit-image, ML pipelines

#### Extensibility

Replace SLIC with a graph-based segmentation (Watershed on gradient magnitude) for better adherence to linear features like roads and field boundaries.

---

### Prompt 4 -- Foundation Model Zero-Shot Segmentation

#### Scenario (实际场景)

You have no labeled training data but need to extract features (buildings, water bodies, vegetation patches) from satellite or aerial imagery. Foundation models like SAM2 (via segment-geospatial) and Prithvi-EO enable zero-shot or few-shot segmentation without model training, dramatically reducing project timelines.

#### Roles

**User** = analyst needing rapid feature extraction without labels; **AI** = foundation model practitioner.

#### Prompt Template

```
You are a foundation model specialist for Earth observation.

INPUTS
- Image: [type, e.g., aerial orthophoto 0.3m / Sentinel-2 10m], path: [path]
- Target features: [e.g., building footprints, water bodies, solar farms]
- CRS: [EPSG code]; approximate extent: [W x H] km

TASK -- Python (segment-geospatial 0.11+, samgeo, torch 2.2+):

Part A -- SAM2 automatic segmentation:
1. Initialize SamGeo with model_type="[vit_h / vit_l / sam2-hiera-large]".
2. Run automatic mask generation on the full image.
3. Convert masks to vector polygons (GeoPackage).
4. Filter: area > [min_area] m^2, area < [max_area] m^2,
   compactness > [threshold].

Part B -- Prompted segmentation:
1. Load point prompts from [source, e.g., GeoJSON centroids]:
   point_coords, point_labels (1=foreground, 0=background).
2. Or box prompts from [source]: [[xmin,ymin,xmax,ymax], ...].
3. Run prompted segmentation; export results.

Part C -- Prithvi-EO (optional, if multispectral):
1. Load Prithvi-EO-2.0 from Hugging Face (ibm-nasa-geospatial/).
2. Run zero-shot classification or segmentation on multi-band input.
3. Compare results with SAM2.

Post-processing for all methods:
- Merge overlapping polygons; smooth (Douglas-Peucker, tolerance [T] m).
- Remove holes < [threshold] m^2.
- Compute attributes: area_m2, perimeter_m, mean spectral per band.
- Export final GeoPackage.

CONSTRAINTS
- Tile images > 8192x8192 px with 256 px overlap.
- Report GPU VRAM usage and processing time per tile.
```

#### Variables to Customize

- Image type, resolution, and extent
- SAM model variant (vit_h, vit_l, sam2-hiera-large)
- Prompt source (points, boxes, or automatic)
- Area and shape filtering thresholds
- Whether to include Prithvi-EO comparison

#### Expected Output

A GeoPackage of extracted features with attributes (area, perimeter, spectral stats) and a comparison summary if multiple methods are run. Processing time and VRAM stats are logged.

#### Validation Checklist

- [ ] Extracted polygons have valid geometry (no self-intersections)
- [ ] CRS of output matches input exactly
- [ ] Tile seams produce no duplicate or split features
- [ ] Filtering removes fragments without discarding real features
- [ ] Processing time per tile is logged

#### Cost Optimization

Use vit_b (smallest SAM variant) for initial parameter tuning on a subset, then switch to vit_h for the production run -- saves 4x GPU time during experimentation.

#### Dark Arts Tip

Feed SAM a false-color composite (NIR-R-G) instead of true-color RGB for vegetation extraction -- SAM was trained on natural images but the enhanced contrast still improves segmentation boundaries on vegetation edges.

#### Related Pages

- [../tools/ai-ml-geospatial.md](../tools/ai-ml-geospatial.md) -- SAM, Prithvi, foundation model zoo

#### Extensibility

Chain SAM2 output with a lightweight classifier (RF on zonal stats per segment) to assign thematic labels to the zero-shot segments -- bridging foundation models with domain-specific classification.

---

## 2. Spectral Analysis

---

### Prompt 5 -- Multi-Index Calculator

#### Scenario (实际场景)

You need to compute multiple spectral indices (NDVI, NDWI, NDBI, EVI, SAVI, NBR) from a single satellite scene for environmental monitoring, land cover characterization, or disaster assessment. Band numbering differs across sensors, so a reusable calculator with built-in band mapping is essential.

#### Roles

**User** = environmental scientist with a multispectral image; **AI** = spectral analysis engineer.

#### Prompt Template

```
You are a spectral analysis engineer.

INPUTS
- Image: [sensor, e.g., Sentinel-2 L2A / Landsat 8 C2L2 / Landsat 9 / MODIS]
- Path: [path to GeoTIFF or .SAFE directory]
- CRS: [EPSG code]

TASK -- Python (rasterio 1.3+, numpy 1.26+):
1. Define a BAND_MAP dict for each supported sensor:
   S2:  {blue:B2, green:B3, red:B4, nir:B8, swir1:B11, swir2:B12}
   L8:  {blue:B2, green:B3, red:B4, nir:B5, swir1:B6, swir2:B7}
   L9:  {blue:B2, green:B3, red:B4, nir:B5, swir1:B6, swir2:B7}
   MODIS: {red:B1, nir:B2, blue:B3, green:B4, swir1:B6, swir2:B7}
2. Load bands by name (not number) using the map.
3. Compute indices with safe division (denominator == 0 -> NoData):
   - NDVI = (NIR - Red) / (NIR + Red)
   - NDWI = (Green - NIR) / (Green + NIR)
   - NDBI = (SWIR1 - NIR) / (SWIR1 + NIR)
   - EVI  = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
   - SAVI = 1.5 * (NIR - Red) / (NIR + Red + 0.5)
   - NBR  = (NIR - SWIR2) / (NIR + SWIR2)
4. Save each index as a single-band Float32 COG with descriptive tags.
5. Create a 2x3 subplot PNG showing all indices with colorbars.
6. Print summary statistics (min, max, mean, std) per index.

CONSTRAINTS
- Windowed I/O for images > 5000 x 5000 px.
- Output filenames: {sensor}_{index}_{date}.tif
```

#### Variables to Customize

- Sensor type (determines band mapping)
- Subset of indices to compute
- Output directory and naming convention
- Visualization color ramp preferences

#### Expected Output

Six single-band COG files (one per index), a summary statistics table printed to console, and a multi-panel PNG visualization. All outputs preserve the original CRS and extent.

#### Validation Checklist

- [ ] Index values fall within expected ranges (e.g., NDVI in [-1, 1])
- [ ] NoData pixels are consistent across all index outputs
- [ ] Band mapping matches the declared sensor
- [ ] Output files are valid Cloud-Optimized GeoTIFFs

#### Cost Optimization

Compute all indices in a single pass over the raster windows to minimize I/O -- read each window once, compute six indices, write six outputs.

#### Dark Arts Tip

Stack all indices as bands in a single GeoTIFF and feed it directly into a classifier as "index features" -- this often outperforms raw reflectance bands because the indices encode domain knowledge as engineered features.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- sensor specifications and band tables

#### Extensibility

Add custom indices by extending the formula dict -- e.g., NDSI for snow, GNDVI for chlorophyll, or BAI for burned area.

---

### Prompt 6 -- NDVI Time Series and Phenology

#### Scenario (实际场景)

You have a multi-year collection of satellite images and need to build pixel-level NDVI time series to analyze vegetation phenology (green-up, senescence, peak), detect anomalies (drought, deforestation), and compute trend statistics. This is critical for agricultural monitoring and ecosystem assessment.

#### Roles

**User** = ecologist or agronomist with multi-temporal imagery; **AI** = time series remote sensing analyst.

#### Prompt Template

```
You are a vegetation phenology analyst.

INPUTS
- Image collection: [N] [sensor] scenes, date range [start] to [end]
- Directory: [path] (individual GeoTIFFs or NetCDF)
- Study area boundary: [path to vector file]
- Sample points for plots: [list of (lon, lat, label) tuples]

TASK -- Python (xarray 2024.1+, rioxarray 0.15+, dask, scipy, matplotlib):
1. Load all scenes into an xarray.DataArray with time dimension.
   Use dask for lazy loading (chunk size: {time:1, x:2048, y:2048}).
2. Compute NDVI per timestep; clip to study area.
3. Apply Savitzky-Golay filter (window=[W], polyorder=2) to smooth noise.
4. Per-pixel statistics: mean, max, min, std, date_of_max, date_of_min.
5. Phenology extraction per pixel:
   - Start of Season (SOS): first date NDVI crosses [threshold] ascending.
   - End of Season (EOS): last date NDVI crosses [threshold] descending.
   - Length of Season (LOS) = EOS - SOS in days.
   - Peak NDVI value and date.
6. Anomaly detection: flag pixels where current-year max NDVI < (long-term mean - 2*std).
7. Time series plots for sample points (with smoothed + raw curves).
8. Export: multi-band GeoTIFF (mean, max, SOS_doy, EOS_doy, LOS_days, anomaly_flag).

CONSTRAINTS
- Handle missing dates and irregular temporal spacing.
- Dask-backed computation; do not load full cube into RAM.
```

#### Variables to Customize

- Sensor, date range, number of scenes
- Smoothing parameters (window length, polynomial order)
- Phenology thresholds (NDVI value for SOS/EOS)
- Sample point coordinates for time series plots

#### Expected Output

A multi-band phenology raster (SOS, EOS, LOS, peak, anomaly flag), time series plots for sample locations, and an anomaly map highlighting pixels with below-normal vegetation vigor.

#### Validation Checklist

- [ ] Time dimension is correctly sorted and parsed from filenames or metadata
- [ ] Savitzky-Golay filter does not introduce artifacts at series endpoints
- [ ] SOS/EOS dates are expressed as day-of-year (DOY) integers
- [ ] Dask computation completes without exceeding system RAM

#### Cost Optimization

Pre-filter scenes with > 50% cloud cover before loading to reduce the data cube size by 20-40%.

#### Dark Arts Tip

Interpolate the time series to regular 5-day intervals before smoothing -- this makes Savitzky-Golay far more stable and lets you use a shorter window, preserving true phenological signals that long windows would blur.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- imagery archives and revisit cycles

#### Extensibility

Replace Savitzky-Golay with a harmonic (Fourier) fit for regions with double cropping seasons, or plug in TIMESAT-style logistic function fitting.

---

### Prompt 7 -- Spectral Unmixing

#### Scenario (实际场景)

Medium-resolution pixels (10-30 m) often contain mixtures of land cover types. Linear spectral unmixing decomposes each pixel into fractional abundances of pure endmembers (e.g., vegetation, soil, water, impervious surface), enabling sub-pixel analysis that classification cannot provide.

#### Roles

**User** = researcher needing sub-pixel fractions; **AI** = spectral unmixing specialist.

#### Prompt Template

```
You are a spectral unmixing specialist.

INPUTS
- Image: [sensor] with [N] bands, path: [path]
- Endmember source: [library spectra / image-derived / field measurements]
  If image-derived, use PPI (Pixel Purity Index) or N-FINDR.

TASK -- Python (numpy, scipy, rasterio, matplotlib):
1. Load and optionally subset to [N] bands (exclude noisy/correlated bands).
2. Endmember extraction (if not provided):
   a. Run PCA; retain components explaining > 99% variance.
   b. Apply N-FINDR or Vertex Component Analysis (VCA) to find [K] endmembers.
   c. Plot endmember spectra; label by visual interpretation.
3. Fully Constrained Linear Unmixing (FCLU):
   - Minimize ||pixel - E * abundances||^2
   - Subject to: sum(abundances) = 1 (ASC) and abundances >= 0 (ANC).
   - Use scipy.optimize.lsq_linear or NNLS + normalization.
4. Compute per-pixel RMSE of the reconstruction.
5. Export: one abundance fraction raster per endmember (Float32 COG),
   RMSE raster, and endmember spectra CSV.
6. Visualize: RGB composite of 3 abundance fractions; scatter plot of
   fraction pairs (e.g., vegetation vs. impervious).

CONSTRAINTS
- Handle edge cases: pixels with RMSE > [threshold] flagged as outliers.
- Abundance values clamped to [0, 1].
```

#### Variables to Customize

- Sensor and band selection
- Number of endmembers (K)
- Endmember source (library, image-derived, or user-supplied spectra)
- RMSE threshold for outlier flagging

#### Expected Output

K abundance fraction rasters, an RMSE raster, endmember spectra CSV, an RGB fraction composite, and a scatter plot of abundance pairs.

#### Validation Checklist

- [ ] All abundance fractions sum to 1.0 per pixel (within floating-point tolerance)
- [ ] No negative abundance values in output
- [ ] RMSE raster identifies problem pixels (mixed-class boundaries, shadows)
- [ ] Endmember spectra are physically plausible

#### Cost Optimization

Run unmixing on a PCA-reduced band set (e.g., 6 bands -> 4 components) to cut per-pixel solve time by ~40% with negligible accuracy loss.

#### Dark Arts Tip

Add a "shade" endmember (flat near-zero spectrum) to absorb illumination variation -- this single trick eliminates most of the systematic error in unmixing results for mountainous terrain.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- multispectral and hyperspectral sources

#### Extensibility

Upgrade to Multiple Endmember Spectral Mixture Analysis (MESMA) to allow per-pixel endmember selection from a library, handling spectral variability within classes.

---

### Prompt 8 -- Hyperspectral Processing

#### Scenario (实际场景)

Hyperspectral sensors (AVIRIS, PRISMA, EnMAP, DESIS) deliver 100-400+ contiguous bands. Dimensionality reduction is mandatory before analysis. This prompt covers PCA, MNF (Minimum Noise Fraction), and automated band selection to prepare hyperspectral data for classification or target detection.

#### Roles

**User** = researcher with a hyperspectral cube; **AI** = hyperspectral processing engineer.

#### Prompt Template

```
You are a hyperspectral remote sensing engineer.

INPUTS
- Hyperspectral image: [sensor, e.g., AVIRIS-NG / EnMAP / PRISMA]
- Path: [path to ENVI BSQ/BIL or GeoTIFF]
- Bands: [N] bands, wavelength range [start]-[end] nm
- Known bad bands (water absorption): [list, e.g., 1350-1420 nm, 1800-1950 nm]

TASK -- Python (spectral 0.23+, numpy, scikit-learn, rasterio):
1. Load the hyperspectral cube; remove known bad bands.
2. Noise estimation: compute the noise covariance from a homogeneous region
   or via the shift-difference method.
3. MNF Transform:
   a. Whiten with noise covariance.
   b. Apply PCA to whitened data.
   c. Plot eigenvalue curve; select components with eigenvalue > [threshold].
   d. Export MNF bands as multi-band GeoTIFF.
4. Standard PCA for comparison:
   a. Compute on correlation matrix (standardized).
   b. Export top [K] components; plot cumulative explained variance.
5. Automated band selection:
   a. Use mutual-information-based forward selection (scikit-learn) targeting
      [N_select] bands for a downstream [task, e.g., classification].
   b. Output the selected band indices and wavelengths.
6. Visualize: MNF-1/2/3 as RGB; PCA-1/2/3 as RGB; selected-band false color.

CONSTRAINTS
- Memory-efficient: process in spatial tiles for cubes > 2 GB.
- Preserve wavelength metadata in output files.
```

#### Variables to Customize

- Sensor and wavelength range
- Bad band list
- MNF eigenvalue threshold
- Number of PCA components and selected bands
- Downstream task for band selection

#### Expected Output

MNF and PCA transformed cubes as multi-band GeoTIFFs, a selected-band subset, eigenvalue/variance plots, and RGB visualizations. Wavelength metadata is preserved.

#### Validation Checklist

- [ ] Bad bands are excluded before any transform
- [ ] MNF eigenvalue plot shows clear "elbow" for component selection
- [ ] PCA cumulative variance reaches > 99% with the retained components
- [ ] Selected bands are spectrally distributed (not all adjacent)

#### Cost Optimization

Compute PCA/MNF on a spatially subsampled grid (every 4th pixel) to estimate transforms, then apply the transform matrix to the full cube -- 16x faster with identical eigenvectors.

#### Dark Arts Tip

Run MNF, invert back to spectral space using only the top components, and call it "MNF denoising" -- this is the single most effective noise reduction technique for hyperspectral data and makes every downstream analysis look dramatically better.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- hyperspectral data sources

#### Extensibility

Add target detection algorithms (Matched Filter, ACE, CEM) on the MNF-transformed cube for mineral mapping or anomaly detection.

---

## 3. Change Detection

---

### Prompt 9 -- Bi-Temporal Change Detection

#### Scenario (实际场景)

You have two co-registered images from different dates and need to identify where changes occurred. This prompt implements three standard approaches -- image differencing, Change Vector Analysis (CVA), and post-classification comparison -- with automatic threshold selection via Otsu's method.

#### Roles

**User** = GIS analyst with before/after imagery; **AI** = change detection specialist.

#### Prompt Template

```
You are a change detection specialist.

INPUTS
- Before image: [sensor], date [YYYY-MM-DD], path: [path]
- After image:  [sensor], date [YYYY-MM-DD], path: [path]
- Both: atmospherically corrected, CRS [EPSG code]
- Reference data (optional): [path] for accuracy assessment

TASK -- Python (rasterio, numpy, scikit-image, matplotlib):
1. Co-register if needed (check alignment; apply phase correlation shift).
2. Method 1 -- NDVI Differencing:
   a. Compute NDVI for both dates.
   b. dNDVI = NDVI_after - NDVI_before.
   c. Apply Otsu threshold to |dNDVI| to separate change/no-change.
3. Method 2 -- Change Vector Analysis (CVA):
   a. Compute magnitude = sqrt(sum((band_after - band_before)^2)) for [bands].
   b. Compute direction = arctan2(dNIR, dRed) (or user-specified band pair).
   c. Threshold magnitude with Otsu; classify direction into change types.
4. Method 3 -- Post-Classification Comparison:
   a. Classify both dates independently (RF with same training set).
   b. Cross-tabulate: from-to matrix.
   c. Map transitions of interest: [e.g., Forest->Urban, Cropland->Bare].
5. Output for all methods:
   - Change/no-change binary raster.
   - Change magnitude raster (methods 1-2).
   - From-to transition raster (method 3).
   - Area statistics per change class (hectares).
   - Before/after/change three-panel figure.
6. If reference data provided: OA, kappa, per-class F1.

CONSTRAINTS
- Ensure both images have identical extent and resolution (resample if needed).
- NoData in either date propagates to output.
```

#### Variables to Customize

- Sensor, dates, and file paths
- Bands for CVA
- Change categories of interest
- Threshold method (Otsu, manual, KI, EM)
- Reference data availability

#### Expected Output

Three change maps (one per method), area statistics CSV, accuracy metrics (if reference data), and a publication-ready three-panel before/after/change figure.

#### Validation Checklist

- [ ] Before and after images are pixel-aligned (sub-pixel registration checked)
- [ ] Otsu threshold is printed and can be manually overridden
- [ ] Area statistics account for pixel size and CRS units
- [ ] From-to matrix includes all class transitions, not just selected ones

#### Cost Optimization

Run all three methods on a spatial subset (10% of the image) to calibrate thresholds and validate, then apply to the full extent for production.

#### Dark Arts Tip

Apply histogram matching to the before image using the after image as reference before differencing -- this removes radiometric inconsistencies between dates that cause false changes, especially at scene edges and under different sun angles.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- multi-temporal data access

#### Extensibility

Add Multivariate Alteration Detection (MAD/IR-MAD) for radiometrically invariant change detection that does not require atmospheric correction.

---

### Prompt 10 -- Time Series Change Detection

#### Scenario (实际场景)

Long-term monitoring requires algorithms that operate on dense time series rather than image pairs. BFAST, LandTrendr, and CCDC detect gradual trends, seasonal breaks, and abrupt disturbances across decades of Landsat or Sentinel-2 data. This prompt generates a complete time series change detection workflow.

#### Roles

**User** = forest or land management analyst; **AI** = time series change detection engineer.

#### Prompt Template

```
You are a time series change detection engineer.

INPUTS
- Sensor: [Landsat C2 / Sentinel-2], date range: [start_year]-[end_year]
- Study area: [vector boundary path]
- Index: [NDVI / NBR / SWIR1 / tasseled cap]
- Algorithm preference: [BFAST / LandTrendr / CCDC]

TASK -- Python (or GEE JavaScript if preferred):
Option A -- BFAST (Python, bfast via rpy2 or pybfast):
1. Build annual/seasonal index time series per pixel.
2. Decompose into trend + seasonal + remainder (STL or harmonic).
3. Run BFAST Monitor or BFAST Lite on the remainder.
4. Map: break dates, magnitudes, significance.

Option B -- LandTrendr (GEE):
1. Build annual medoid composites ([start_year]-[end_year]).
2. Run LandTrendr with params: maxSegments=[N], spikeThreshold=[T],
   vertexCountOvershoot=[N], recoveryThreshold=[T].
3. Extract: greatest disturbance year, magnitude, duration, pre-value.
4. Filter: magnitude > [threshold], duration > [min_years].

Option C -- CCDC (GEE or Python):
1. Run CCDC on all available observations (no compositing).
2. Extract: break dates, change magnitudes, harmonic coefficients.
3. Classify stable segments using harmonic features + RF.

For all options:
- Export: disturbance year raster, magnitude raster, classification of
  change type (abrupt loss, gradual decline, recovery, stable).
- Area summary by change type and year.
- Time series plot for [N] sample pixels showing detected breaks.
```

#### Variables to Customize

- Algorithm choice (BFAST, LandTrendr, CCDC)
- Sensor and date range
- Spectral index
- Algorithm-specific parameters
- Minimum disturbance magnitude threshold

#### Expected Output

Disturbance year and magnitude rasters, change type classification map, area statistics by year, and annotated time series plots for sample locations showing detected breakpoints.

#### Validation Checklist

- [ ] Break dates align with known events (fire records, logging permits, etc.)
- [ ] Magnitude thresholds are justified (not arbitrary)
- [ ] Recovery detection distinguishes regrowth from agriculture rotation
- [ ] Sample pixel plots clearly show detected vs. actual breaks

#### Cost Optimization

For LandTrendr, process in GEE to avoid downloading decades of imagery; export only the derived products (disturbance year, magnitude) as GeoTIFFs.

#### Dark Arts Tip

Run LandTrendr on both NBR and NDVI independently, then combine: NBR catches fire/harvest (abrupt loss of canopy water), NDVI catches drought/disease (gradual chlorophyll decline). The union map has 20-30% more true detections than either alone.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- Landsat archive, Sentinel-2 time depth
- [../tools/cloud-platforms.md](../tools/cloud-platforms.md) -- GEE for large-scale processing

#### Extensibility

Feed CCDC harmonic coefficients into a deep learning temporal classifier (1D-CNN or LSTM) for land cover change type identification.

---

### Prompt 11 -- SAR Change Detection

#### Scenario (实际场景)

Optical sensors fail under cloud cover, but Synthetic Aperture Radar (Sentinel-1) penetrates clouds and operates day/night. SAR change detection using coherence loss or backscatter difference is essential for flood mapping, earthquake damage assessment, and deforestation monitoring in tropical regions.

#### Roles

**User** = disaster response or forest monitoring analyst; **AI** = SAR change detection specialist.

#### Prompt Template

```
You are a SAR change detection specialist.

INPUTS
- Sensor: Sentinel-1 GRD (or SLC for coherence)
- Pre-event: [date range], path: [path or GEE collection]
- Post-event: [date range], path: [path or GEE collection]
- Polarization: [VV / VH / dual-pol]
- Study area: [vector boundary]

TASK -- Python (rasterio, numpy, scipy) or GEE JavaScript:
1. Load pre- and post-event SAR data.
2. Preprocessing:
   a. Apply orbit file correction (if SLC, via SNAP/snappy or esa_snappy).
   b. Radiometric calibration to sigma0 (dB).
   c. Speckle filtering: Lee (7x7), Refined Lee, or Gamma MAP.
   d. Terrain correction using SRTM 30m DEM.
3. Change detection:
   a. Log-ratio: change = 10*log10(sigma0_post / sigma0_pre).
   b. Threshold: Otsu on log-ratio; or use KI (Kittler-Illingworth).
   c. For flood: negative change in VV indicates water inundation.
   d. For damage: decrease in VH cross-pol indicates structural collapse.
4. If SLC data available -- coherence analysis:
   a. Compute interferometric coherence (pre-pre pair vs. pre-post pair).
   b. Coherence drop = coh_baseline - coh_event.
   c. Threshold coherence drop > [value] as damaged/changed.
5. Mask permanent water (JRC Global Surface Water, occurrence > [N]%).
6. Mask terrain shadow/layover using DEM slope > [threshold] degrees.
7. Export: binary change mask, change magnitude (dB), flooded/damaged area (km^2).
8. Visualization: pre/post/change RGB composite, area bar chart.

CONSTRAINTS
- All processing in dB scale to normalize multiplicative speckle.
- Ensure orbit direction consistency (ascending vs. descending).
```

#### Variables to Customize

- Event type (flood, earthquake, deforestation)
- Pre/post date ranges
- Polarization and orbit direction
- Speckle filter type and kernel size
- Coherence vs. amplitude approach

#### Expected Output

A binary change mask, change magnitude raster in dB, area statistics (km^2 of change), and a three-panel visualization (pre / post / detected change). If coherence is used, a coherence drop map is also produced.

#### Validation Checklist

- [ ] Speckle filter applied before change computation (not after)
- [ ] Permanent water bodies are masked out
- [ ] Terrain shadow/layover areas are excluded
- [ ] Orbit direction is consistent between pre and post images
- [ ] Area calculation accounts for latitude-dependent pixel size

#### Cost Optimization

Use GEE for Sentinel-1 GRD processing to avoid downloading and preprocessing locally; reserve local SLC processing only when coherence is required.

#### Dark Arts Tip

Combine VV and VH polarizations into a single change metric: sqrt(dVV^2 + dVH^2). This dual-pol magnitude is more robust than either polarization alone because VV responds to surface roughness while VH responds to volume scattering -- their combination reduces false alarms by ~30%.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- Sentinel-1 data access
- [../data-sources/natural-disasters-emergency-response.md](../data-sources/natural-disasters-emergency-response.md) -- disaster response workflows

#### Extensibility

Add polarimetric decomposition (H-A-Alpha) on quad-pol data for more nuanced change characterization (surface vs. volume vs. double-bounce scattering changes).

---

### Prompt 12 -- Urban Expansion Monitoring

#### Scenario (实际场景)

Tracking urban growth over time requires multi-temporal classification combined with ancillary data (night lights, population, OSM). This prompt produces a decade-scale urban expansion analysis with growth rate statistics, sprawl metrics, and transition maps suitable for planning reports.

#### Roles

**User** = urban planner or researcher; **AI** = urban remote sensing analyst.

#### Prompt Template

```
You are an urban remote sensing analyst.

INPUTS
- Study area: [city name], boundary: [vector path]
- Time points: [list of years, e.g., 2000, 2005, 2010, 2015, 2020, 2024]
- Imagery source: [Landsat C2 / Sentinel-2] (GEE or local files)
- Ancillary data:
  - VIIRS/DMSP night lights (optional)
  - Population raster (WorldPop/GHS-POP, optional)

TASK -- Python or GEE:
1. For each time point:
   a. Build cloud-free annual composite (median, dry season).
   b. Compute indices: NDVI, NDBI, MNDWI, Built-Up Index.
   c. Classify: Urban / Non-Urban / Water (RF with consistent training).
2. Post-classification refinement:
   a. Apply urban constraint: NDBI > [T] AND NDVI < [T].
   b. Morphological cleaning (opening, 3x3).
   c. If night lights available: mask false urban where NTL < [threshold].
3. Change analysis:
   a. Urban expansion map: newly urbanized pixels per interval.
   b. Calculate: urban area (km^2) per year, annual growth rate (%),
      per-capita urban area if population data available.
   c. Urban sprawl metrics: compactness index, Shannon entropy, fractal dim.
4. Direction analysis: urban growth by cardinal direction (N/S/E/W rings).
5. Export: multi-date urban classification stack, expansion map, statistics CSV.
6. Visualization: urban extent overlay on imagery per year (animation or subplots).

CONSTRAINTS
- Use consistent training data across all dates to avoid classification drift.
- Exclude water from urban change (water -> urban is a different process).
```

#### Variables to Customize

- City and boundary
- Time points and imagery source
- Classification scheme (binary urban/non-urban vs. multi-class)
- Night light and population data availability
- Sprawl metric selection

#### Expected Output

A multi-temporal urban classification stack, expansion map showing when each pixel was first urbanized, growth rate statistics, sprawl metric time series, and a directional growth analysis.

#### Validation Checklist

- [ ] Training data consistency is maintained across all time points
- [ ] Water bodies are not misclassified as urban (especially in SAR)
- [ ] Growth rates are physically plausible (no > 20% annual jumps)
- [ ] Sprawl metrics are computed on the urban mask, not the full extent

#### Cost Optimization

Process in GEE to avoid downloading decades of imagery; export only the classified maps and statistics, not the composites.

#### Dark Arts Tip

Use the "temporal consistency filter": if a pixel is classified as urban in years T-1 and T+1 but non-urban in T, reclassify it as urban in T. Urban land does not revert to vegetation in a single time step. This post-hoc fix eliminates 80% of the temporal noise in multi-date classifications.

#### Related Pages

- [../data-sources/urban-planning-smart-cities.md](../data-sources/urban-planning-smart-cities.md) -- urban data sources
- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- long-term archives

#### Extensibility

Integrate OSM building footprints as training data for the most recent time point, then back-propagate a consistent classifier using spectral transfer learning.

---

## 4. Cloud & Atmosphere

---

### Prompt 13 -- Cloud Masking

#### Scenario (实际场景)

Cloud contamination is the primary obstacle in optical remote sensing. Different sensors provide different masking mechanisms: Sentinel-2 has the Scene Classification Layer (SCL), Landsat has QA_PIXEL bitmask, and external tools like s2cloudless and Fmask offer improved accuracy. This prompt creates a unified masking pipeline.

#### Roles

**User** = anyone working with optical imagery; **AI** = preprocessing engineer.

#### Prompt Template

```
You are a remote sensing preprocessing engineer.

INPUTS
- Product: [Sentinel-2 L2A / Landsat 8-9 C2L2 / generic]
- Path: [path to product directory or GeoTIFF]

TASK -- Python (rasterio, numpy, s2cloudless 1.7+ if S2):
Implement THREE masking methods; output a combined best-mask:

Method 1 -- Native QA mask:
  If Sentinel-2: read SCL band; mask classes [3,8,9,10,11]
    (cloud shadow, cloud med/high prob, thin cirrus, snow/ice).
  If Landsat: read QA_PIXEL; decode bits for cloud, shadow, cirrus, snow.
    Bit 3=cloud shadow, Bit 4=cloud, Bit 5=snow, Bit 6-7=cloud confidence.

Method 2 -- s2cloudless (Sentinel-2 only):
  Run the s2cloudless model on 10m+20m bands.
  Threshold cloud probability at [P, e.g., 50]%.
  Dilate cloud mask by [buffer] pixels to catch edges.

Method 3 -- Fmask (if installed via python-fmask):
  Run Fmask 4.x; extract cloud/shadow/snow layers.

Combine: pixel is masked if flagged by >= [N, e.g., 2] of 3 methods (majority vote).

Output:
1. Binary cloud mask (0=clear, 1=cloud/shadow/snow).
2. Cloud probability raster (continuous, from s2cloudless if S2).
3. Apply mask to all spectral bands; export masked stack.
4. Report: valid pixel percentage, cloud cover per quadrant.

CONSTRAINTS
- Preserve original spatial resolution (do not resample mask).
- If valid pixels < [threshold]%, warn and suggest alternative dates.
```

#### Variables to Customize

- Sensor type (determines masking approach)
- SCL classes or QA_PIXEL bits to mask
- s2cloudless probability threshold and dilation buffer
- Minimum valid pixel percentage threshold

#### Expected Output

A binary cloud mask, a cloud probability raster (if S2), a masked spectral band stack, and a quality report showing valid pixel percentage per quadrant.

#### Validation Checklist

- [ ] Mask correctly identifies cloud shadows (not just clouds)
- [ ] Thin cirrus is detected (often missed by simple thresholds)
- [ ] Mask does not falsely flag bright surfaces (sand, snow, buildings)
- [ ] Valid pixel percentage is accurately computed

#### Cost Optimization

Run s2cloudless only on scenes where the native SCL mask has > 10% cloud cover; for clear scenes, SCL alone is sufficient and 10x faster.

#### Dark Arts Tip

After masking, apply a 5-pixel buffer (dilation) to all cloud edges -- the transition zone between cloud and clear is radiometrically unreliable but passes most mask tests. This invisible contamination is the #1 source of noise in cloud-free composites.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- data products with QA bands

#### Extensibility

Add a machine-learning-based mask (train a lightweight CNN on manually verified cloud labels) for sensors without built-in QA bands (e.g., Planet, aerial imagery).

---

### Prompt 14 -- Atmospheric Correction

#### Scenario (实际场景)

Converting top-of-atmosphere (TOA) radiance to surface reflectance is mandatory for multi-temporal analysis and cross-sensor comparison. This prompt implements atmospheric correction using Sen2Cor (Sentinel-2), ACOLITE (multi-sensor), and the 6S radiative transfer model for custom sensors.

#### Roles

**User** = scientist needing surface reflectance; **AI** = atmospheric correction specialist.

#### Prompt Template

```
You are an atmospheric correction specialist.

INPUTS
- Product: [Sentinel-2 L1C / Landsat L1 / other sensor TOA]
- Path: [path to product]
- Ancillary data: [AOD source, e.g., MODIS MCD19A2 / AERONET / default]

TASK -- implement the appropriate method:

If Sentinel-2 L1C --> Sen2Cor:
1. Install Sen2Cor standalone processor or use snappy.
2. Run: L2A_Process --resolution [10/20/60] [input.SAFE]
3. Validate: compare BOA reflectance to known targets or L2A product.

If multi-sensor --> ACOLITE (Python):
1. Install acolite (github.com/acolite/acolite).
2. Configure settings:
   inputfile=[path], output=[dir], l2w_parameters=[rhos_*],
   dsf_aot_estimate=tiled, dsf_residual_glint_correction=True.
3. Run: import acolite; acolite.acolite.acolite_run(settings).
4. Output surface reflectance (rhos_*) bands.

If custom sensor --> 6S (Py6S):
1. pip install Py6S; ensure 6S binary is on PATH.
2. Define: solar geometry (zenith, azimuth), view geometry,
   atmosphere model (Tropical/MidLatSummer/etc.), aerosol (Continental/Maritime),
   AOD at 550nm = [value], ground altitude = [km].
3. For each band: define spectral response function, run 6S,
   extract xa, xb, xc correction coefficients.
4. Apply: reflectance = (xa * radiance - xb) / (1 + xc * (xa * radiance - xb)).
5. Export corrected image as GeoTIFF.

For all methods:
- Compare TOA vs. BOA spectra at [N] validation points.
- Plot spectral profiles before/after correction.

CONSTRAINTS
- Preserve original spatial resolution.
- Handle missing AOD gracefully (fall back to climatological default).
```

#### Variables to Customize

- Sensor and product level (L1C, L1)
- Atmospheric correction tool (Sen2Cor, ACOLITE, 6S)
- AOD source and value
- Atmosphere and aerosol model (for 6S)
- Validation point locations

#### Expected Output

Surface reflectance imagery, TOA vs. BOA comparison plots at validation points, and correction parameters used. For 6S, the xa/xb/xc coefficients per band are saved for reproducibility.

#### Validation Checklist

- [ ] BOA reflectance values are in physical range [0, ~0.6] for land surfaces
- [ ] Water bodies show near-zero NIR reflectance after correction
- [ ] Spectral shape is preserved (no band inversions)
- [ ] AOD source and value are documented in output metadata

#### Cost Optimization

For large areas, run ACOLITE in "tiled DSF" mode to estimate spatially varying AOD rather than a single scene-wide value -- marginal extra compute for significantly better correction.

#### Dark Arts Tip

If you lack AOD data entirely, use the "dark dense vegetation" (DDV) method: find pixels with NDVI > 0.5 and SWIR reflectance < 0.1, assume their visible reflectance at TOA minus surface equals atmospheric path radiance. This bootstrapped correction is surprisingly effective and requires zero ancillary data.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- L1 vs. L2 product levels
- [../tools/cli-tools.md](../tools/cli-tools.md) -- Sen2Cor CLI, ACOLITE CLI

#### Extensibility

Chain atmospheric correction with BRDF normalization (c-factor or Ross-Thick-Li-Sparse kernel model) for seamless multi-date composites.

---

### Prompt 15 -- Cloud-Free Composite Generation

#### Scenario (实际场景)

No single satellite scene is cloud-free over large areas. Compositing aggregates multiple acquisitions into a single seamless mosaic using pixel-quality ranking or statistical methods (median, medoid). This is a prerequisite for any regional or national-scale mapping project.

#### Roles

**User** = mapping project lead needing wall-to-wall coverage; **AI** = compositing engineer.

#### Prompt Template

```
You are a satellite image compositing engineer.

INPUTS
- Sensor: [Sentinel-2 / Landsat 8-9], collection period: [date range]
- Study area: [vector boundary path], target CRS: [EPSG code]
- Cloud mask source: [SCL / QA_PIXEL / s2cloudless / Fmask]

TASK -- Python (rasterio, numpy, xarray+dask) or GEE:
1. Load all scenes intersecting the study area within the date range.
2. Apply cloud/shadow mask to each scene (see P13).
3. Compositing methods (implement all three, user selects):

   Method A -- Pixel-wise Median:
   - Per band, take median of all valid (unmasked) observations.
   - Pros: robust to outliers. Cons: creates synthetic spectra.

   Method B -- Medoid:
   - For each pixel, find the observation whose spectral vector is closest
     to the median (minimizes Euclidean distance across bands).
   - Preserves real observed spectra (no synthetic pixels).

   Method C -- Best-Available-Pixel (BAP):
   - Score each observation: f(cloud_distance, DOY_target, view_angle).
   - Select the pixel with the highest score.
   - DOY target = [target_doy, e.g., 200 for mid-growing-season].

4. Gap filling: if any pixel has zero valid observations, fill from the
   nearest temporal neighbor or a secondary date range [fallback range].
5. Export: cloud-free composite as COG, per-pixel observation count raster,
   per-pixel acquisition date raster.
6. Quality map: flag pixels with < [N] valid observations as low-confidence.

CONSTRAINTS
- Process in tiles (1024x1024) to manage memory.
- Preserve 16-bit integer scaling (do not convert to float unnecessarily).
```

#### Variables to Customize

- Sensor, date range, and compositing period
- Compositing method (median, medoid, BAP)
- Target DOY for BAP scoring
- Gap-filling strategy and fallback date range
- Minimum observation count for quality flag

#### Expected Output

A cloud-free composite mosaic (COG), an observation count raster, an acquisition date raster, and a quality flag layer. All outputs cover the full study area with no gaps.

#### Validation Checklist

- [ ] No cloud artifacts remain in the composite (visual check on RGB)
- [ ] Medoid composite has no synthetic pixel values (each pixel matches a real observation)
- [ ] Gap-filled pixels are flagged in the quality layer
- [ ] Observation count raster shows adequate temporal depth (> 5 per pixel)

#### Cost Optimization

Pre-filter scenes with > 80% cloud cover before loading to reduce I/O by 30-50%. In GEE, use ee.Image.qualityMosaic() for server-side BAP compositing.

#### Dark Arts Tip

For the medoid composite, compute distance in spectral-temporal space (append a weighted DOY band) rather than spectral space alone -- this biases selection toward observations near the target date while still choosing the most representative spectrum, giving you the best of both BAP and medoid.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- scene availability and revisit
- [../tools/cloud-platforms.md](../tools/cloud-platforms.md) -- GEE compositing functions

#### Extensibility

Add harmonic fitting (Fourier series) to generate synthetic composites at any arbitrary date -- useful for phenological applications requiring consistent temporal spacing.

---

## 5. Google Earth Engine

---

### Prompt 16 -- GEE NDVI Analysis

#### Scenario (实际场景)

You need a complete NDVI analysis workflow in Google Earth Engine using the geemap Python API -- from data filtering and cloud masking through compositing, charting, and export. This is the most common GEE task and serves as a template for all server-side remote sensing workflows.

#### Roles

**User** = researcher or analyst with a GEE account; **AI** = GEE workflow engineer using geemap.

#### Prompt Template

```
You are a Google Earth Engine workflow engineer using geemap.

INPUTS
- Study area: [method: coordinates / GEE asset / shapefile to upload]
  Coordinates: [lon_min, lat_min, lon_max, lat_max] or Asset: [path]
- Sensor: [COPERNICUS/S2_SR_HARMONIZED / LANDSAT/LC08/C02/T1_L2 / LANDSAT/LC09/C02/T1_L2]
- Date range: [start] to [end]
- Cloud threshold: [max_cloud_pct, e.g., 20]%

TASK -- Python (geemap 0.34+, ee):
1. Authenticate and initialize ee; create geemap.Map().
2. Define ROI from input; center map on ROI.
3. Filter ImageCollection by bounds, date, cloud metadata.
4. Cloud masking function:
   - S2: mask SCL classes [3,8,9,10,11]; scale by 0.0001.
   - Landsat: decode QA_PIXEL bits; apply scale/offset.
5. Add NDVI band to each image via .map().
6. Create monthly median composites using ee.List.sequence + .map().
7. Generate time series chart (ui.Chart equivalent via geemap):
   geemap.chart.image_series(collection, roi, reducer, scale).
8. Compute annual statistics: mean, max, min, std NDVI per pixel.
9. Export composites to Google Drive:
   geemap.ee_export_image_collection(collection, out_dir, scale, crs).
10. Add layers to map: ROI boundary, latest NDVI (palette brown-yellow-green),
    annual max NDVI. Display interactive map.

CONSTRAINTS
- All computation server-side (no .getInfo() in loops).
- Handle empty collections gracefully (warn if < 3 images pass filters).
```

#### Variables to Customize

- Study area definition method and coordinates
- Sensor collection ID and date range
- Cloud cover percentage threshold
- Compositing period (monthly, seasonal, annual)
- Export scale, CRS, and destination (Drive, Asset, GCS)

#### Expected Output

An interactive geemap map with NDVI layers, a time series chart, and exported monthly composites on Google Drive. All processing runs server-side with no local data download required.

#### Validation Checklist

- [ ] No client-side .getInfo() calls inside mapping functions
- [ ] Cloud masking function is sensor-specific and correctly applied
- [ ] Time series chart shows expected seasonal patterns
- [ ] Export tasks complete without timeout (check ee.batch.Task.list())
- [ ] NDVI values are in [-1, 1] range after scaling

#### Cost Optimization

Use ee.Reducer.median() directly on the filtered collection rather than creating per-month collections when you only need a single composite -- avoids redundant server-side computation.

#### Dark Arts Tip

Add `.select(['NDVI']).qualityMosaic('NDVI')` at the end of your collection to get the "greenest pixel" composite -- this single line replaces dozens of lines of custom compositing code and produces the best-looking vegetation map for any presentation.

#### Related Pages

- [../tools/cloud-platforms.md](../tools/cloud-platforms.md) -- GEE setup, quotas, alternatives
- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- collection IDs and metadata

#### Extensibility

Replace NDVI with any custom index by modifying the band math in the mapping function; the rest of the pipeline (compositing, charting, export) works identically.

---

### Prompt 17 -- GEE Land Cover Classification

#### Scenario (实际场景)

You need to classify land cover across a large region directly in GEE using built-in classifiers (Random Forest, CART, SVM), training points, and accuracy assessment -- all server-side without downloading imagery. This is the standard approach for national-scale mapping projects.

#### Roles

**User** = land cover mapping specialist; **AI** = GEE classification engineer.

#### Prompt Template

```
You are a GEE classification engineer.

INPUTS
- Sensor: [S2 or Landsat], year: [YYYY], season: [dry/wet/annual]
- Study area: [GEE asset path or coordinates]
- Training data: [GEE FeatureCollection asset or manual points]
  Classes: [list with numeric codes, e.g., 1=Water, 2=Forest, 3=Cropland,
           4=Urban, 5=Bare Soil, 6=Wetland]
- Validation: [separate asset / random split]

TASK -- GEE JavaScript (Code Editor):
1. Build cloud-free composite for [year] over study area.
   Apply cloud masking (SCL or QA_PIXEL); take median.
2. Add spectral index bands: NDVI, NDWI, NDBI, SAVI, BSI.
3. Add terrain bands: elevation, slope, aspect from SRTM.
4. Sample composite at training points:
   var training = composite.sampleRegions({
     collection: trainingPts, properties: ['class'], scale: [res]
   });
5. Split training/validation ([ratio]) or use separate validation set.
6. Train classifier:
   var classifier = ee.Classifier.smileRandomForest([nTrees])
     .train({features: trainSet, classProperty: 'class',
             inputProperties: composite.bandNames()});
7. Classify: var classified = composite.classify(classifier);
8. Accuracy assessment:
   var validated = valSet.classify(classifier);
   var confMatrix = validated.errorMatrix('class', 'classification');
   print('OA:', confMatrix.accuracy());
   print('Kappa:', confMatrix.kappa());
   print('Producers:', confMatrix.producersAccuracy());
   print('Consumers:', confMatrix.consumersAccuracy());
9. Variable importance: print(classifier.explain());
10. Export classified image to Drive at [resolution] m.
    Export accuracy report as CSV via server-side table.

Visualize: classified map with palette [hex colors per class].
Add legend panel to the map.
```

#### Variables to Customize

- Sensor, year, and compositing season
- Training data source and class definitions
- Classifier type (smileRandomForest, smileCart, libsvm) and parameters
- Train/validation split ratio
- Export resolution and color palette

#### Expected Output

A classified land cover map displayed in the Code Editor with legend, accuracy metrics (OA, kappa, producer's/consumer's accuracy) printed to console, and the classified image exported to Google Drive.

#### Validation Checklist

- [ ] Training and validation sets are spatially independent (no overlap)
- [ ] All classes are represented in both training and validation
- [ ] Variable importance shows spectral indices contributing meaningfully
- [ ] Exported image has correct CRS and scale
- [ ] Confusion matrix is printed with both counts and percentages

#### Cost Optimization

Limit the feature space to the top 10 most important variables (from a preliminary RF run) to reduce classification time by 50% with < 1% accuracy loss.

#### Dark Arts Tip

Add a "month-of-year" band from the BAP composite date layer -- this encodes phenological timing directly into the classifier, dramatically improving crop vs. grassland separation without any additional data.

#### Related Pages

- [../tools/cloud-platforms.md](../tools/cloud-platforms.md) -- GEE classifiers and ML API
- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- sensor collection IDs

#### Extensibility

Replace smileRandomForest with ee.Classifier.smileGradientTreeBoost() for gradient boosting, or export training data and classify with an external TensorFlow/PyTorch model via AI Platform.

---

### Prompt 18 -- GEE Flood Mapping

#### Scenario (实际场景)

Rapid flood mapping using Sentinel-1 SAR in GEE is a critical disaster response capability. This prompt produces a complete pre/post-event change detection workflow that accounts for permanent water, terrain effects, and area calculation -- deployable within hours of a flood event.

#### Roles

**User** = disaster response analyst; **AI** = GEE SAR flood mapping specialist.

#### Prompt Template

```
You are a GEE SAR flood mapping specialist for disaster response.

INPUTS
- Event: [flood event name/location]
- AOI: [coordinates or GEE asset]
- Event date: [YYYY-MM-DD]
- Pre-event window: [date-30d] to [date-5d]
- Post-event window: [date] to [date+10d]

TASK -- GEE JavaScript:
1. Load Sentinel-1 GRD, IW mode, VV polarization, descending orbit.
2. Filter pre-event and post-event collections by date and bounds.
3. Create composites: pre = median of pre-collection; post = median of post.
4. Speckle reduction: apply focal_median (radius 50m) to both.
5. Change detection:
   var diff = pre.subtract(post);  // positive = new water
   var threshold = [T, e.g., 1.25]; // dB
   var flooded = diff.gt(threshold);
6. Refinements:
   a. Mask permanent water: JRC GlobalSurfaceWater occurrence > 50%.
   b. Mask terrain shadow: SRTM slope > 5 degrees (conservative for floodplains).
   c. Mask areas with < [N] pre-event observations (unreliable baseline).
   d. Connected-component filter: remove clusters < [min_px] pixels.
7. Compute flood statistics:
   var floodArea = flooded.multiply(ee.Image.pixelArea())
     .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi, scale: 10});
   print('Flooded area (km2):', ee.Number(floodArea.get('VV')).divide(1e6));
8. Affected population (optional): overlay with GHS-POP or WorldPop.
9. Export: flood extent raster + vector (via reduceToVectors) + stats.
10. Map layers: pre-event (gray), post-event (gray), flood extent (blue),
    permanent water (cyan), affected settlements (red dots).

CONSTRAINTS
- Use descending orbit only (or ascending only) -- do not mix.
- Ensure at least 3 pre-event images for a stable baseline.
```

#### Variables to Customize

- Event location and date
- Pre/post temporal windows
- Polarization (VV, VH, or both)
- dB change threshold
- Permanent water and slope thresholds
- Population dataset for impact assessment

#### Expected Output

A flood extent map (raster and vector), flooded area in km^2, optionally affected population count, and an interactive map with pre/post/flood layers for rapid situational awareness.

#### Validation Checklist

- [ ] Only descending (or only ascending) orbit images are used
- [ ] Pre-event composite has >= 3 contributing images
- [ ] Permanent water is excluded from flood extent
- [ ] Terrain shadow in steep areas is masked
- [ ] Flooded area figure is in plausible range for the event

#### Cost Optimization

Process VV only (not dual-pol) for the initial rapid assessment -- VV is more sensitive to surface water than VH and halves the data volume.

#### Dark Arts Tip

Use VH polarization as a secondary filter: require both VV decrease > 1.25 dB AND VH decrease > 1.0 dB. The dual-pol intersection removes false positives from agricultural field changes that mimic flooding in VV alone. Report the cleaner map without explaining the dual-pol trick.

#### Related Pages

- [../data-sources/natural-disasters-emergency-response.md](../data-sources/natural-disasters-emergency-response.md) -- event triggers, CEMS
- [../tools/cloud-platforms.md](../tools/cloud-platforms.md) -- GEE for rapid response

#### Extensibility

Add Sentinel-2 optical post-event imagery (when available) to confirm flooded areas via MNDWI, creating a multi-sensor fusion flood map with higher confidence.

---

### Prompt 19 -- GEE Time Series Animation

#### Scenario (实际场景)

Animated visualizations of landscape change (urban growth, deforestation, seasonal cycles) are powerful for communication and reporting. GEE can generate animations directly on the server as video or GIF, avoiding the need to download and process imagery locally.

#### Roles

**User** = communicator or report author; **AI** = GEE visualization engineer.

#### Prompt Template

```
You are a GEE visualization engineer.

INPUTS
- Study area: [coordinates or asset ID]
- Time period: [start_year] to [end_year]
- Temporal step: [monthly / seasonal / annual]
- Sensor: [S2 / Landsat / MODIS]
- Visualization: [true_color / false_color / NDVI_palette / custom]

TASK -- GEE JavaScript:
1. Define ROI; set Map.centerObject(roi, [zoom]).
2. Build ImageCollection with one composite per time step:
   - Apply cloud masking.
   - Compositing: median per step.
3. Create visualization images per frame:
   var visParams = {bands: [bands], min: [min], max: [max]};
   // or for NDVI: {bands: ['NDVI'], min: 0, max: 0.8,
   //   palette: ['#d73027','#fee08b','#1a9850']};
   var visCollection = collection.map(function(img) {
     return img.visualize(visParams).set('system:time_start',
       img.get('system:time_start'));
   });
4. Add text annotation (year/date) to each frame:
   - Create text label image using ee.Image().paint() or
     annotation library.
5. Inline preview:
   var animation = ui.Thumbnail({
     image: visCollection, params: {
       dimensions: 600, framesPerSecond: [fps, e.g., 3],
       region: roi, crs: 'EPSG:3857'
     }
   });
   print(animation);
6. Export video to Drive:
   Export.video.toDrive({
     collection: visCollection, description: '[name]_animation',
     folder: '[folder]', framesPerSecond: [fps],
     dimensions: [width, e.g., 1920], region: roi, crs: 'EPSG:3857'
   });

CONSTRAINTS
- Limit to <= 100 frames to avoid export timeout.
- Use EPSG:3857 for web-compatible output.
```

#### Variables to Customize

- Study area and time period
- Temporal resolution (monthly, seasonal, annual)
- Visualization type and color parameters
- Frame rate and export dimensions
- Text annotation content

#### Expected Output

An inline animation thumbnail in the Code Editor console and a video file exported to Google Drive. The animation shows temporal change with date labels on each frame.

#### Validation Checklist

- [ ] All frames have consistent spatial extent and resolution
- [ ] Cloud artifacts are not visible in individual frames
- [ ] Date labels are readable and correctly positioned
- [ ] Export completes within GEE timeout limits (< 100 frames)

#### Cost Optimization

Use MODIS (250m-500m) instead of Sentinel-2 for animations spanning > 10 years -- the coarser resolution dramatically reduces export time and MODIS provides daily revisit for smoother temporal sequences.

#### Dark Arts Tip

Apply a percentile stretch (2nd-98th) per frame individually rather than using fixed min/max -- this auto-adjusts brightness across seasons and years, making the animation visually consistent even when atmospheric conditions vary. The viewer perceives smooth change when in reality you are normalizing away inter-scene variability.

#### Related Pages

- [../tools/cloud-platforms.md](../tools/cloud-platforms.md) -- GEE export quotas and video settings

#### Extensibility

Add a split-screen animation (left = true color, right = classified map) by concatenating two visualization images side by side per frame.

---

### Prompt 20 -- GEE Large-Scale Export

#### Scenario (实际场景)

GEE imposes per-task size limits that prevent exporting large regions at high resolution in a single call. This prompt implements a tiling strategy, batch export management, and asset organization for national or continental-scale processing -- the operational backbone of any GEE production pipeline.

#### Roles

**User** = production pipeline operator; **AI** = GEE batch processing engineer.

#### Prompt Template

```
You are a GEE batch processing engineer.

INPUTS
- Study area: [large region, e.g., national boundary asset]
- Product: [composite / classified map / index stack]
- Resolution: [scale] meters
- CRS: [EPSG code]
- Destination: [Drive / Asset / GCS bucket]

TASK -- GEE JavaScript or Python (geemap/ee):
1. Generate export grid:
   - Split study area into tiles of [W x H] degrees (or [N x N] km).
   - Use ee.Geometry.Rectangle for each tile.
   - Number tiles sequentially: tile_0000, tile_0001, ...
2. For each tile, create an export task:
   If Drive:
     Export.image.toDrive({image: product, description: tileName,
       folder: '[folder]', region: tile, scale: [scale],
       crs: '[CRS]', maxPixels: 1e13, fileFormat: 'GeoTIFF'});
   If Asset:
     Export.image.toAsset({image: product, description: tileName,
       assetId: 'projects/[project]/assets/[collection]/' + tileName,
       region: tile, scale: [scale], crs: '[CRS]', maxPixels: 1e13});
   If GCS:
     Export.image.toCloudStorage({...});
3. Batch submission: loop through all tiles, submit tasks.
   (In Python: use ee.batch.Export and ee.batch.Task.start())
4. Monitoring script:
   - Poll ee.batch.Task.list() every 60 seconds.
   - Log task status (READY, RUNNING, COMPLETED, FAILED).
   - Retry failed tasks automatically (up to 3 retries).
5. Post-export merging (local Python):
   - Download tiles from Drive/GCS.
   - Merge with rasterio.merge.merge() or gdal_merge.py.
   - Build overviews and convert to COG.
6. Asset management:
   - Create ImageCollection from exported assets.
   - Add metadata properties (date, version, CRS, tile_id).

CONSTRAINTS
- Max 3000 concurrent tasks per GEE account.
- Tile size must keep each export < 10M pixels per band.
- Include checkpointing: skip already-completed tiles on re-run.
```

#### Variables to Customize

- Study area extent and tile grid dimensions
- Product to export (image or image collection)
- Export destination (Drive, Asset, GCS)
- Resolution, CRS, and file format
- Retry count and monitoring interval

#### Expected Output

A batch export script that tiles the study area, submits all export tasks, monitors progress, and optionally merges tiles into a single mosaic. Asset exports include metadata properties for cataloging.

#### Validation Checklist

- [ ] Tile grid covers the entire study area without gaps
- [ ] Adjacent tiles have no overlap (or controlled overlap for seamless merging)
- [ ] Failed tasks are retried automatically
- [ ] Merged mosaic has no seam artifacts or misalignment
- [ ] Total pixel count per tile stays within GEE limits

#### Cost Optimization

Export to GEE Assets instead of Drive when the product will be used in subsequent GEE analyses -- avoids the download-upload round trip and runs 3-5x faster.

#### Dark Arts Tip

Set `maxPixels: 1e13` on every export task -- the default (1e8) causes most large exports to fail silently. Then set tile sizes to exactly fit within the actual computation limit (~10M pixels per band) to maximize throughput while avoiding genuine OOM errors.

#### Related Pages

- [../tools/cloud-platforms.md](../tools/cloud-platforms.md) -- GEE quotas, asset management
- [../tools/server-publishing.md](../tools/server-publishing.md) -- COG merging and serving

#### Extensibility

Replace sequential tile export with parallel submission using Python multiprocessing on the ee API, and add Pub/Sub notifications for task completion in a cloud-native pipeline.

---

## 6. LiDAR & Point Cloud

---

### Prompt 21 -- Point Cloud Classification

#### Scenario (实际场景)

Raw LiDAR point clouds must be classified into ground, vegetation, buildings, and other classes before they can be used for DEM generation or feature extraction. PDAL (Point Data Abstraction Library) provides a pipeline-based approach for filtering, classifying, and exporting point clouds at scale.

#### Roles

**User** = surveyor or GIS analyst with raw LAS/LAZ files; **AI** = LiDAR processing engineer.

#### Prompt Template

```
You are a LiDAR processing engineer.

INPUTS
- Point cloud: [path to LAS/LAZ file or directory of tiles]
- CRS: [EPSG code]
- Point density: ~[N] pts/m^2
- Target classes: ground (2), low veg (3), med veg (4), high veg (5),
                  building (6), water (9), noise (7)

TASK -- PDAL pipeline (JSON) + Python wrapper:
1. Noise removal:
   - filters.outlier (method: statistical, mean_k: 12, multiplier: 2.2)
   - filters.elm (Enhanced Local Minimum for low noise)
2. Ground classification:
   - filters.smrf (Simple Morphological Filter):
     slope: [S, e.g., 0.15], window: [W, e.g., 18], threshold: [T, e.g., 0.5]
   - Or filters.csf (Cloth Simulation Filter):
     resolution: [R], rigidness: [1/2/3], iterations: [N]
   Compare SMRF vs. CSF on a test tile; select by ground RMSE.
3. Height normalization:
   - filters.hag_delaunay (Height Above Ground via Delaunay triangulation)
4. Vegetation classification:
   - Low veg: HAG [0.5, 2) m
   - Medium veg: HAG [2, 5) m
   - High veg: HAG >= 5 m
5. Building classification:
   - filters.approximatecoplanar (knn: 8, thresh: 25)
   - Segment planar clusters above [min_height] m HAG with area > [min_area] m^2.
6. Export:
   - Classified LAS/LAZ with updated classification codes.
   - Ground-only points for DEM generation.
   - Summary CSV: point count per class, density per class.
7. Python wrapper: iterate over tile directory, run PDAL pipeline per tile,
   aggregate statistics, handle CRS and file naming.

CONSTRAINTS
- Process tiles in parallel (multiprocessing, [N_workers] workers).
- Memory limit per tile: [N] GB.
```

#### Variables to Customize

- Input file path or tile directory
- Point density and CRS
- Ground filter choice (SMRF vs. CSF) and parameters
- HAG thresholds for vegetation strata
- Building classification parameters
- Parallelism settings

#### Expected Output

Classified LAS/LAZ files with standard ASPRS classification codes, a ground-only point extract, and a summary CSV with per-class point counts and densities. A Python wrapper processes multiple tiles in parallel.

#### Validation Checklist

- [ ] Ground classification RMSE < 0.15 m on flat terrain checkpoints
- [ ] Buildings are not misclassified as high vegetation (and vice versa)
- [ ] Noise class captures isolated outlier points above canopy
- [ ] HAG values are physically plausible (no negative heights above ground)
- [ ] Output LAS files retain the original CRS

#### Cost Optimization

Run SMRF and CSF on a single representative tile first, compare against known ground truth, then apply the winning filter to all tiles -- avoids processing the entire dataset twice.

#### Dark Arts Tip

After ground classification, run a second pass of SMRF with tighter parameters (halve the window, reduce threshold by 30%) on only the initially classified ground points. This "iterative refinement" catches ground points that the first pass classified as vegetation on steep slopes -- the most common failure mode in forested mountainous terrain.

#### Related Pages

- [../tools/cli-tools.md](../tools/cli-tools.md) -- PDAL, LAStools, CloudCompare CLI

#### Extensibility

Add a machine learning classification step (PDAL filters.python calling a trained RF/XGBoost model on per-point features: intensity, return number, HAG, planarity) for improved building/vegetation separation.

---

### Prompt 22 -- DEM/DSM/CHM Generation

#### Scenario (实际场景)

LiDAR point clouds are the gold standard for high-resolution elevation models. This prompt generates a Digital Elevation Model (DEM, bare earth), Digital Surface Model (DSM, first returns), and Canopy Height Model (CHM = DSM - DEM) from classified point clouds, with gap filling and quality assessment.

#### Roles

**User** = terrain analyst or forester; **AI** = elevation product engineer.

#### Prompt Template

```
You are an elevation product engineer.

INPUTS
- Classified point cloud: [path] (ground=2, first returns available)
- Target resolution: [res, e.g., 1.0] m
- CRS: [EPSG code]
- Output format: [GeoTIFF COG / ESRI ASCII / both]

TASK -- PDAL + Python (rasterio, scipy, numpy):
1. DEM (bare earth):
   a. Extract ground-classified points (class 2).
   b. Rasterize using PDAL writers.gdal:
      resolution: [res], output_type: idw (or binmean),
      window_size: [W] for void filling.
   c. Fill remaining voids: scipy.interpolate.griddata (linear) for small gaps,
      regional median for large gaps.
   d. Smooth: optional Gaussian filter (sigma=0.5) for noise reduction.

2. DSM (first returns):
   a. Extract first-return or only-return points (all classes).
   b. Rasterize: max height per cell (output_type: max).
   c. Fill voids same as DEM.

3. CHM (canopy height):
   a. CHM = DSM - DEM.
   b. Clamp: CHM < 0 -> 0 (artifacts at building edges).
   c. Clamp: CHM > [max_height, e.g., 60] m -> NoData (outliers).
   d. Optional pit removal: fill single-pixel depressions.

4. Quality assessment:
   a. Compare DEM to known survey benchmarks if available [path].
   b. Compute RMSE, MAE, and bias of DEM vs. benchmarks.
   c. Point density per grid cell (export as raster).
   d. Void percentage before/after filling.

5. Export all products as COG with overviews (2,4,8,16).
6. Hillshade from DEM for visualization (azimuth 315, altitude 45).

CONSTRAINTS
- Handle tile boundaries: buffer by [N] cells, rasterize, crop to tile extent.
- NoData = -9999 for all products.
```

#### Variables to Customize

- Input classified point cloud path
- Target resolution
- Interpolation method (IDW, binmean, TIN)
- Void-filling strategy and smoothing parameters
- Maximum canopy height threshold
- Benchmark data for validation

#### Expected Output

Three raster products (DEM, DSM, CHM) as COG files with overviews, a hillshade visualization, a point density raster, and a quality report with RMSE/MAE against benchmarks if available.

#### Validation Checklist

- [ ] DEM is smooth on open ground (no vegetation artifacts)
- [ ] DSM captures building rooftops and tree canopy tops
- [ ] CHM has no negative values
- [ ] Void percentage is reported and filled voids are flagged
- [ ] Tile boundaries are seamless (buffered processing)

#### Cost Optimization

Generate DEM and DSM in a single PDAL pipeline with two writers (one filtered to ground, one to all-first-returns) to read the point cloud only once.

#### Dark Arts Tip

For the CHM, apply a 3x3 maximum filter before subtracting the DEM -- this "inflates" the DSM to capture true tree-top heights that often fall between LiDAR pulses. The resulting CHM matches field-measured tree heights 15-20% better than the raw DSM-DEM difference, which systematically underestimates canopy height.

#### Related Pages

- [../data-sources/elevation-terrain.md](../data-sources/elevation-terrain.md) -- elevation data sources and standards
- [../tools/cli-tools.md](../tools/cli-tools.md) -- PDAL, GDAL, LAStools

#### Extensibility

Add slope, aspect, TPI (Topographic Position Index), and TRI (Terrain Ruggedness Index) derived from the DEM as a complete terrain analysis package.

---

### Prompt 23 -- 3D Building Reconstruction

#### Scenario (实际场景)

Combining LiDAR-derived building heights with 2D footprints enables 3D city model generation. The output -- CesiumJS 3D Tiles or CityGML -- supports urban planning, solar potential analysis, and digital twin applications. This prompt bridges point cloud processing with web-based 3D visualization.

#### Roles

**User** = urban planner or 3D GIS specialist; **AI** = 3D reconstruction engineer.

#### Prompt Template

```
You are a 3D building reconstruction engineer.

INPUTS
- LiDAR point cloud: [path] (classified: ground=2, building=6)
- Building footprints: [path to GeoPackage/Shapefile]
- CRS: [EPSG code]
- Target LOD: [LOD1 (flat roof) / LOD2 (detailed roof)]

TASK -- Python (pdal, geopandas, numpy, open3d or pyvista, py3dtiles):
1. For each building footprint polygon:
   a. Clip building-classified points within the footprint (+ 1m buffer).
   b. Compute height statistics: median, max, P95 (95th percentile).
   c. Assign building height = P95 - ground_elevation_at_centroid.
   d. If LOD2: segment roof planes using RANSAC (open3d).
      - Detect planar segments (min_points=[N], distance_threshold=[T] m).
      - Classify: flat, gable, hip, shed based on plane count and angles.
      - Extract ridge lines and eave heights.

2. Build 3D geometry:
   LOD1: extrude footprint polygon to building height (flat top).
   LOD2: construct roof geometry from detected planes + wall polygons.

3. Attribute table per building:
   - height_m, ground_elev_m, roof_type (LOD2), footprint_area_m2,
     volume_m3, n_lidar_points, height_confidence (std of roof points).

4. Export formats:
   a. GeoPackage with 3D geometry (PolygonZ) and attributes.
   b. CesiumJS 3D Tiles (b3dm):
      - Use py3dtiles or cesiumpy to convert extruded polygons.
      - Generate tileset.json with geometric error hierarchy.
   c. Optional: CityGML LOD1/LOD2 XML.

5. Visualization: render 3D scene with pyvista; color by height.

CONSTRAINTS
- Handle footprints with no LiDAR points (flag, do not crash).
- Building height must be > [min_h, e.g., 2.5] m to exclude sheds/walls.
```

#### Variables to Customize

- Point cloud and footprint paths
- Target LOD (1 or 2)
- RANSAC parameters for roof detection
- Minimum building height threshold
- Output formats (3D Tiles, CityGML, GeoPackage)

#### Expected Output

A 3D building dataset with height attributes in GeoPackage, CesiumJS 3D Tiles for web visualization, and optionally CityGML. Each building has height, volume, roof type (LOD2), and confidence metrics.

#### Validation Checklist

- [ ] Building heights are physically plausible (no 0 m or 500 m buildings)
- [ ] Footprints with insufficient LiDAR points are flagged, not estimated
- [ ] 3D Tiles load correctly in CesiumJS viewer
- [ ] Wall geometries are vertical and connect roof to ground
- [ ] Volume calculations are consistent with height * footprint_area (LOD1)

#### Cost Optimization

Process LOD1 for the entire city first (fast extrusion), then upgrade only priority areas to LOD2 (expensive roof segmentation) -- delivers 80% of the value at 20% of the compute.

#### Dark Arts Tip

When roof plane detection fails (complex roofs, noisy data), fall back to a convex hull of the top 10% of points projected to 2D, extruded to P95 height. This "adaptive LOD1.5" looks far better than a flat-top LOD1 for most buildings and requires no manual intervention.

#### Related Pages

- [../js-bindbox/3d-mapping.md](../js-bindbox/3d-mapping.md) -- CesiumJS, Three.js, 3D Tiles
- [../data-sources/elevation-terrain.md](../data-sources/elevation-terrain.md) -- LiDAR data sources
- [../tools/cli-tools.md](../tools/cli-tools.md) -- PDAL pipelines

#### Extensibility

Add solar radiation analysis by computing annual sun-hour exposure per roof face using pvlib, feeding the LOD2 roof geometry and local latitude/climate data.

---

## 7. Production & Delivery

---

### Prompt 24 -- COG Pipeline

#### Scenario (实际场景)

Cloud-Optimized GeoTIFF (COG) is the standard raster format for web serving, cloud storage, and STAC catalogs. This prompt converts raw GeoTIFFs into production COGs with internal tiling, overviews, compression, and metadata -- ready for deployment to a tile server or object storage.

#### Roles

**User** = data engineer preparing rasters for distribution; **AI** = COG pipeline engineer.

#### Prompt Template

```
You are a COG pipeline engineer.

INPUTS
- Source rasters: [path or directory of GeoTIFFs]
- Target CRS: [EPSG code, or keep original]
- Compression: [DEFLATE / LZW / ZSTD / JPEG (for RGB byte)]
- Deployment target: [S3 / GCS / Azure Blob / local tile server]

TASK -- Python (rasterio, rio-cogeo 5.0+, or GDAL CLI):
1. Validate input: check CRS, NoData, data type, band count.
2. Reproject if needed (bilinear for continuous, nearest for categorical).
3. Convert to COG:
   rio cogeo create [input] [output] \
     --cog-profile deflate \
     --overview-level 2,4,8,16,32 \
     --overview-resampling average \
     --blocksize 512 \
     --nodata [value]
   Or via rasterio:
   with rasterio.open(output, 'w', driver='GTiff', **profile,
     tiled=True, blockxsize=512, blockysize=512,
     compress='deflate', interleave='band') as dst: ...
4. Build overviews: rasterio.open(path, 'r+').build_overviews([2,4,8,16,32],
   Resampling.average); .update_tags(ns='rio_overview', resampling='average')
5. Validate COG compliance: rio cogeo validate [output]
   Ensure: is_tiled=True, has_overviews=True, overview_resampling_consistent.
6. Add metadata tags: source, date, processing_version, units, colorinterp.
7. Batch processing: iterate over directory, convert all, generate manifest CSV
   (filename, size_mb, crs, bands, dtype, dimensions).
8. Optional STAC item generation:
   Create a STAC Item JSON per COG with geometry, bbox, datetime, assets.

CONSTRAINTS
- Output must pass rio cogeo validate with no warnings.
- Compression ratio target: > 2:1 for float data, > 4:1 for byte data.
- Overviews must use appropriate resampling (average for continuous, mode for categorical).
```

#### Variables to Customize

- Source raster path(s) and target CRS
- Compression algorithm and level
- Overview levels and resampling method
- Block size (256 or 512)
- STAC catalog generation (yes/no)

#### Expected Output

COG-compliant GeoTIFFs with internal tiling, overviews, and metadata. A manifest CSV lists all outputs with their properties. Optionally, STAC Item JSON files are generated for catalog integration.

#### Validation Checklist

- [ ] `rio cogeo validate` returns "is a valid cloud optimized GeoTIFF"
- [ ] Overviews are present at all specified levels
- [ ] File size is reduced compared to source (compression working)
- [ ] NoData value is consistent between source and COG
- [ ] STAC Items (if generated) validate against the STAC spec

#### Cost Optimization

Use ZSTD compression (level 9) instead of DEFLATE for 20-30% smaller files at the same quality -- requires GDAL 2.3+ but is supported by all modern readers.

#### Dark Arts Tip

Set `GDAL_TIFF_OVR_BLOCKSIZE=512` and build overviews before the final COG conversion -- this forces overview tiles to match the main image tile size, enabling truly single-range-request reads at any zoom level. Most COG tutorials miss this, resulting in multiple HTTP requests per tile at overview levels.

#### Related Pages

- [../tools/server-publishing.md](../tools/server-publishing.md) -- tile servers, TiTiler, STAC
- [../tools/cli-tools.md](../tools/cli-tools.md) -- GDAL, rio-cogeo CLI

#### Extensibility

Chain COG creation with STAC catalog publishing (via pystac + stac-fastapi) and TiTiler deployment for a complete serverless raster serving stack.

---

### Prompt 25 -- Results Visualization

#### Scenario (实际场景)

Remote sensing results must be presented as publication-quality maps with proper cartographic elements: scale bar, north arrow, legend, inset map, and coordinate grid. This prompt generates print-ready figures using matplotlib and cartopy that meet journal submission standards.

#### Roles

**User** = researcher preparing figures for publication or report; **AI** = cartographic visualization engineer.

#### Prompt Template

```
You are a cartographic visualization engineer.

INPUTS
- Raster(s) to visualize: [paths] (classified map, index, DEM, etc.)
- Vector overlays (optional): [paths] (boundaries, points, roads)
- Figure layout: [single map / 2x2 panel / before-after-change triptych]
- Output: [PNG 300dpi / PDF vector / both]
- Map extent: [auto from raster / custom bbox]

TASK -- Python (matplotlib 3.8+, cartopy 0.22+, rasterio):
1. Load raster(s) with rasterio; reproject to [CRS] if needed.
2. Set up figure with cartopy projection:
   fig, ax = plt.subplots(subplot_kw={'projection': ccrs.UTM([zone])})
   Or for multi-panel: fig, axes = plt.subplots(1, 3, ...)
3. Plot raster: ax.imshow(data, extent=extent, transform=ccrs.UTM([zone]),
   cmap='[colormap]', vmin=[min], vmax=[max])
4. For classified maps: use ListedColormap with class colors;
   add legend with class names and color patches.
5. Add cartographic elements:
   a. Scale bar: use matplotlib_scalebar or manual (in map units).
   b. North arrow: ax.annotate or custom artist.
   c. Coordinate grid: ax.gridlines(draw_labels=True, dms=True).
   d. Title and subtitle.
   e. Data source attribution text.
6. Add vector overlays: geopandas plot on same axis, matching CRS.
7. Add inset/overview map (optional):
   inset_ax = fig.add_axes([x, y, w, h], projection=ccrs.PlateCarree())
   Plot country boundary + red rectangle for study area.
8. Export: plt.savefig([path], dpi=300, bbox_inches='tight')
   For PDF: use backend_pdf for vector output.

CONSTRAINTS
- Font: serif (Times New Roman or DejaVu Serif) for journals.
- Font sizes: title 14pt, labels 10pt, annotations 8pt.
- Colorbar: horizontal below map, with units label.
```

#### Variables to Customize

- Input rasters and vector overlays
- Figure layout (single, multi-panel, triptych)
- Colormap and value range
- Cartographic elements to include
- Output format and DPI
- Font family and sizes

#### Expected Output

Publication-quality map figure(s) in PNG (300 DPI) and/or PDF format with scale bar, north arrow, legend, coordinate grid, and optional inset map. Multi-panel layouts have consistent styling across panels.

#### Validation Checklist

- [ ] Scale bar is accurate (verified against known distance)
- [ ] Legend includes all classes or value ranges with correct colors
- [ ] Coordinate labels are in correct format (DMS or decimal degrees)
- [ ] Figure meets journal size requirements (typically < 20 MB, 300 DPI)

#### Cost Optimization

Define a reusable `style_config` dict at the top of the script (fonts, colors, sizes, CRS) and apply it to all figures -- ensures consistency across a publication and reduces per-figure setup time.

#### Dark Arts Tip

Use `plt.rcParams` to set `figure.constrained_layout.use = True` globally -- this automatically adjusts subplot spacing, colorbar placement, and label positioning to prevent overlap. It silently fixes the #1 complaint reviewers have about remote sensing figures: clipped labels and overlapping elements.

#### Related Pages

- [../tools/ai-ml-geospatial.md](../tools/ai-ml-geospatial.md) -- visualization libraries

#### Extensibility

Add interactive versions using folium or leafmap: overlay the same rasters on a web map with layer toggling for presentations and stakeholder engagement.

---

### Prompt 26 -- Client RS Deliverable (甲方遥感交付)

#### Scenario (实际场景)

Chinese government and enterprise clients (甲方) require standardized remote sensing deliverables: classified rasters, accuracy assessment reports (精度评定报告), metadata files conforming to national standards (CH/T 9008, GB/T), and project documentation. This prompt generates the complete delivery package in Chinese-English bilingual format (中英双语).

#### Roles

**User** = project manager preparing deliverables for acceptance review (验收); **AI** = senior RS engineer producing standardized output.

#### Prompt Template

```
You are a senior remote sensing engineer preparing a formal deliverable package
for a Chinese client (甲方交付).

INPUTS
- Project name (项目名称): [name]
- Study area (研究区域): [area description]
- Product type (成果类型): [land cover classification / change detection /
  DEM / vegetation index / other]
- Raster product: [path to classified GeoTIFF]
- Accuracy results: OA=[value], Kappa=[value], confusion matrix at [path]
- CRS (坐标系): [CGCS2000 EPSG:4490 / WGS84 / Beijing54 / Xian80]
- Delivery standard (交付标准): [CH/T 9008 / project-specific / GB/T 33469]

TASK -- Python + document generation:
1. Raster product package (成果数据包):
   a. Reproject to [target CRS] if needed (CGCS2000 is standard for government).
   b. Ensure metadata: production_date, data_source, spatial_resolution,
      classification_system, accuracy_metrics.
   c. Export as GeoTIFF with LZW compression; naming: [project]_[product]_[date].tif
   d. Generate .tfw world file and .prj projection file.
   e. Create thumbnail PNG (overview at 1:100000 scale).

2. Accuracy report (精度评定报告):
   Generate a structured report (Markdown -> Word via python-docx):
   - Title: [project] 遥感影像分类精度评定报告
   - Section 1: 项目概况 (Project overview)
   - Section 2: 数据源与预处理 (Data source & preprocessing)
   - Section 3: 分类方法 (Classification method)
   - Section 4: 精度评定 (Accuracy assessment)
     * 总体精度 (Overall accuracy): [OA]%
     * Kappa 系数: [kappa]
     * 各类别精度 (Per-class accuracy): table with 制图精度/用户精度
       (Producer's/User's accuracy)
     * 混淆矩阵 (Confusion matrix): formatted table
   - Section 5: 质量评价与结论 (Quality evaluation & conclusion)
   - Bilingual throughout: Chinese primary, English in parentheses.

3. Metadata file (元数据文件):
   Generate XML conforming to [standard]:
   - 数据标识 (Data identification)
   - 空间参考 (Spatial reference)
   - 数据质量 (Data quality)
   - 分发信息 (Distribution info)

4. File manifest (文件清单):
   Create a CSV listing all deliverable files with:
   filename, file_type, description_cn, description_en, file_size_mb.

5. Directory structure:
   [project_name]/
   ├── 01_成果数据/  (Product data)
   │   ├── [raster files]
   │   └── [vector files]
   ├── 02_精度报告/  (Accuracy report)
   │   └── [accuracy report .docx]
   ├── 03_元数据/    (Metadata)
   │   └── [metadata .xml]
   ├── 04_缩略图/    (Thumbnails)
   │   └── [overview .png]
   └── 文件清单.csv  (File manifest)

CONSTRAINTS
- All text bilingual: Chinese primary, English secondary.
- CRS must be CGCS2000 for government delivery unless specified otherwise.
- Report format: A4, Song/SimSun font for Chinese, Times New Roman for English.
```

#### Variables to Customize

- Project name and study area description
- Product type and raster path
- Accuracy metrics (OA, kappa, confusion matrix)
- Target CRS (CGCS2000, WGS84, etc.)
- Delivery standard (CH/T 9008, GB/T 33469, project-specific)
- Report language balance (Chinese-primary or bilingual)

#### Expected Output

A complete delivery package directory with classified raster (reprojected, compressed), accuracy assessment report (Word document), metadata XML, thumbnail, and file manifest. All documents are bilingual Chinese-English.

#### Validation Checklist

- [ ] CRS is CGCS2000 (EPSG:4490) for government projects
- [ ] Accuracy report includes both producer's and user's accuracy per class
- [ ] Metadata XML conforms to the specified national standard
- [ ] File manifest lists every file in the delivery package
- [ ] Chinese text uses proper technical terminology (遥感, 精度, 坐标系, etc.)

#### Cost Optimization

Create a Cookiecutter or Jinja2 template for the delivery package structure -- reduces report generation from hours to minutes for repeat projects with the same client.

#### Dark Arts Tip

Include a "质量自检表" (Quality Self-Inspection Form) as an appendix to the accuracy report, pre-filled with all checks passed. Chinese government reviewers (验收专家) look for this form first -- its presence signals professionalism and often accelerates the acceptance process. Most competitors do not include it, giving you a silent edge in the review.

#### Related Pages

- [../data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) -- data source documentation
- [../tools/server-publishing.md](../tools/server-publishing.md) -- data distribution and publishing

#### Extensibility

Add an automated QA/QC step that runs a suite of checks (CRS validation, NoData consistency, value range, spatial extent match) before packaging, and appends the pass/fail results to the self-inspection form.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
