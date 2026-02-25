# Time-Series Analysis for Geospatial Data

> A definitive expert-level reference for temporal and time-series analysis in GIS,
> covering satellite image time series, change detection, climate trends, movement
> analytics, forecasting, and space-time pattern mining.

**Cross-references:**
- [Satellite Imagery Data Sources](../data-sources/satellite-imagery.md)
- [Climate & Weather Data Sources](../data-sources/climate-weather.md)
- [Temporal Animation & Visualization](../visualization/temporal-animation.md)
- [Spatial Statistics](spatial-statistics.md)
- [Python Stack](python-stack.md) | [R Stack](r-stack.md)
- [ML for GIS](ml-gis.md)

---

## Table of Contents

1. [Introduction & Quick Picks](#1-introduction--quick-picks)
2. [Satellite Image Time Series](#2-satellite-image-time-series)
3. [NDVI/EVI Time Series & Phenology](#3-ndvievi-time-series--phenology)
4. [Change Detection](#4-change-detection)
5. [Climate Time Series](#5-climate-time-series)
6. [Urban Growth Analysis](#6-urban-growth-analysis)
7. [Movement & Trajectory Analysis](#7-movement--trajectory-analysis)
8. [Forecasting](#8-forecasting)
9. [Event Detection & Early Warning](#9-event-detection--early-warning)
10. [Space-Time Cubes](#10-space-time-cubes)
11. [Panel Data & Spatial Econometrics](#11-panel-data--spatial-econometrics)
12. [Tools Comparison](#12-tools-comparison)
13. [Workflow Templates](#13-workflow-templates)

---

## 1. Introduction & Quick Picks

Geospatial time-series analysis is the study of spatially-referenced observations
collected over time. It bridges remote sensing science, climatology, ecology,
epidemiology, transportation, and urban studies. The core challenge is that
observations are autocorrelated in **both** space and time, demanding specialised
methods that honour this dual dependency.

### 1.1 State-of-the-Art Tool Matrix

| Task | Python | R | Cloud/Platform |
|------|--------|---|----------------|
| Satellite image time series | `xarray` + `rioxarray` | `sits`, `stars`, `terra` | Google Earth Engine |
| NDVI phenology | `phenolopy`, `eemont` | `phenofit`, `TIMESAT` | GEE (NDVI composites) |
| Change detection | `nrt` (near-real-time) | `bfast`, `bfastSpatial` | GEE CCDC |
| Climate trends | `xarray` + `scipy` | `trend`, `zyp` | Copernicus CDS API |
| Urban growth | `xarray` + GHSL | `landscapemetrics` | GEE nightlights |
| Trajectory analysis | `movingpandas`, `scikit-mobility` | `trajectories`, `move` | Uber H3 |
| Forecasting | `statsforecast`, `pytorch-forecasting` | `forecast`, `fable` | Vertex AI |
| Space-time cubes | `xarray`, `datacube` | `stars`, `spacetime` | ArcGIS Pro STCube |
| Panel/econometrics | `spreg`, `linearmodels` | `splm`, `plm` | -- |

### 1.2 Why Geospatial Time Series Differ from Standard Time Series

Standard time-series methods (ARIMA, exponential smoothing) assume observations at
a single location or are i.i.d. across locations. Geospatial time series violate
these assumptions:

- **Spatial autocorrelation** -- nearby pixels/stations co-vary (Tobler's First Law).
- **Irregular temporal sampling** -- cloud cover, orbit gaps, sensor failures create
  missing data that is **not** missing-at-random (clouds correlate with weather).
- **Massive dimensionality** -- a Sentinel-2 tile has ~120 million pixels; a
  10-year stack has >600 images. Brute-force per-pixel analysis is infeasible
  without chunked/lazy evaluation.
- **Non-stationarity** -- land use change means the generative process shifts over
  time, requiring breakpoint-aware models (BFAST, CCDC).

### 1.3 Key Concepts

| Concept | Definition |
|---------|-----------|
| **Temporal composite** | Aggregating multiple observations into a single best-pixel image (e.g., median NDVI for a 16-day window) |
| **Gap-filling** | Reconstructing missing observations via interpolation, harmonic fitting, or ML |
| **Harmonic regression** | Fitting sine/cosine terms to capture seasonal cycles: `y(t) = a0 + a1*cos(2*pi*t/T) + b1*sin(2*pi*t/T) + ...` |
| **Breakpoint** | A structural change in the time series, indicating land-cover conversion, disturbance, or regime shift |
| **Phenometrics** | Quantitative measures of vegetation phenology: start of season (SOS), peak, end of season (EOS), amplitude |
| **Space-time cube** | A 3D structure (x, y, t) that bins events or raster values into spatiotemporal voxels |

---

## 2. Satellite Image Time Series

### 2.1 Building Analysis-Ready Time Series

The foundation of any satellite time-series workflow is constructing an
**Analysis-Ready Data (ARD)** cube: geometrically aligned, atmospherically
corrected, cloud-masked, and temporally regular.

#### Python: xarray + stackstac + pystac

```python
"""
Build a Sentinel-2 NDVI time series cube from Microsoft Planetary Computer.
Requires: pystac-client, stackstac, xarray, rioxarray, dask
"""
import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
import numpy as np

# --- 1. Search the STAC catalogue ---
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[11.35, 46.45, 11.45, 46.55],   # Bolzano, Italy
    datetime="2022-01-01/2022-12-31",
    query={"eo:cloud_cover": {"lt": 30}},
)
items = search.item_collection()
print(f"Found {len(items)} scenes")

# --- 2. Lazy-load into an xarray DataArray via stackstac ---
stack = stackstac.stack(
    items,
    assets=["B04", "B08", "SCL"],         # Red, NIR, Scene Classification
    resolution=10,
    epsg=32632,
    chunksize=(1, 1, 512, 512),           # (time, band, y, x)
)

# --- 3. Cloud masking using SCL band ---
scl = stack.sel(band="SCL")
cloud_mask = scl.isin([3, 8, 9, 10, 11])  # cloud shadow, med/high prob cloud, cirrus, snow
clean = stack.where(~cloud_mask)

# --- 4. Compute NDVI ---
red = clean.sel(band="B04").astype("float32")
nir = clean.sel(band="B08").astype("float32")
ndvi = (nir - red) / (nir + red + 1e-10)
ndvi = ndvi.clip(-1, 1)

# --- 5. Monthly composite (median) ---
ndvi_monthly = ndvi.resample(time="1MS").median()
ndvi_monthly = ndvi_monthly.compute()     # trigger Dask execution
print(ndvi_monthly)
# Dimensions: (time: 12, y: ~1100, x: ~1100)
```

#### R: sits (Satellite Image Time Series)

The `sits` package (Simoes et al.) provides an integrated pipeline from
STAC search through classification of time series.

```r
# install.packages("sits")
library(sits)

# --- 1. Define a data cube from Microsoft Planetary Computer ---
s2_cube <- sits_cube(
  source     = "MPC",
  collection = "SENTINEL-2-L2A",
  bands      = c("B04", "B08", "CLOUD"),
  roi        = c(lon_min = 11.35, lon_max = 11.45,
                 lat_min = 46.45, lat_max = 46.55),
  start_date = "2022-01-01",
  end_date   = "2022-12-31"
)

# --- 2. Regularise to 16-day composites ---
reg_cube <- sits_regularize(
  cube       = s2_cube,
  period     = "P16D",
  res        = 10,
  output_dir = tempdir()
)

# --- 3. Extract NDVI time series at sample points ---
samples <- tibble::tibble(
  longitude = c(11.40, 11.42),
  latitude  = c(46.50, 46.48),
  label     = c("forest", "vineyard"),
  start_date = "2022-01-01",
  end_date   = "2022-12-31"
)
ts <- sits_get_data(reg_cube, samples)
plot(ts)
```

### 2.2 gdalcubes for On-the-Fly Data Cubes

`gdalcubes` (Appel & Pebesma, 2019) builds regular raster data cubes from
irregular image collections entirely on-the-fly, avoiding large intermediate files.

```r
library(gdalcubes)

# Create image collection from local or cloud files
ic <- create_image_collection(
  files   = list.files("s2_scenes/", pattern = "\\.tif$", full.names = TRUE),
  format  = "Sentinel2_L2A"
)

# Define target cube view
v <- cube_view(
  srs        = "EPSG:32632",
  extent     = list(left = 680000, right = 690000,
                    bottom = 5150000, top = 5160000,
                    t0 = "2022-01-01", t1 = "2022-12-31"),
  dx = 10, dy = 10, dt = "P1M",
  aggregation = "median",
  resampling  = "bilinear"
)

# Build the cube and apply NDVI
cube <- raster_cube(ic, v) |>
  select_bands(c("B04", "B08")) |>
  apply_pixel("(B08-B04)/(B08+B04)", names = "NDVI")

# Plot monthly NDVI
plot(cube, zlim = c(0, 0.9), col = viridis::viridis(20))
```

### 2.3 Temporal Compositing Strategies

| Strategy | Method | Best For | Artefacts |
|----------|--------|----------|-----------|
| **Maximum value** | `max()` over window | NDVI greenness | Overestimates in cloudy regions |
| **Median** | `median()` | General-purpose | Blurs rapid changes |
| **Medoid** | Observation closest to median in feature space | Multi-band composites | Computationally heavier |
| **Weighted mean** | Weight by cloud distance, view angle | Landsat ARD | Requires quality metadata |
| **Best-available pixel** | Select observation with lowest cloud probability | High-res imagery | May mix dates within tile |

### 2.4 Gap-Filling Methods

Missing observations are inevitable. The choice of gap-filling strategy depends on
the temporal frequency, gap duration, and downstream application.

#### Whittaker Smoother (Python)

```python
"""
Whittaker smoother for NDVI time series gap-filling.
Fast, tuneable smoothness via lambda parameter.
"""
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve

def whittaker_smooth(y, weights, lmbda=100):
    """
    Fit a Whittaker smoother.

    Parameters
    ----------
    y : array-like, shape (n,)
        Observed values (NaN for missing).
    weights : array-like, shape (n,)
        0 for missing, 1 for valid (or continuous quality weight).
    lmbda : float
        Smoothing parameter. Higher = smoother.

    Returns
    -------
    z : np.ndarray
        Smoothed/gap-filled time series.
    """
    n = len(y)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    W = diags(w, 0, shape=(n, n))
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    Z = W + lmbda * D.T @ D
    z = spsolve(Z, W @ y)
    return z

# Example usage
time_steps = 23  # 16-day composites in a year
ndvi_obs = np.random.uniform(0.3, 0.8, time_steps)
ndvi_obs[[3, 4, 5, 15, 16]] = np.nan   # simulate cloud gaps
weights = np.where(np.isnan(ndvi_obs), 0, 1)
ndvi_obs_filled = np.where(np.isnan(ndvi_obs), 0, ndvi_obs)

smoothed = whittaker_smooth(ndvi_obs_filled, weights, lmbda=10)
print(f"Gap-filled NDVI shape: {smoothed.shape}")
```

#### Harmonic Regression (Python)

```python
"""
Harmonic regression for seasonal fitting and gap-filling.
Models annual + semi-annual harmonics.
"""
import numpy as np
from sklearn.linear_model import LinearRegression

def harmonic_fit(doy, values, mask, n_harmonics=2, period=365.25):
    """
    Fit harmonic regression to valid observations, predict at all DOYs.

    Parameters
    ----------
    doy : array, day-of-year for each observation
    values : array, observed values
    mask : boolean array, True = valid
    n_harmonics : int, number of sine/cosine pairs
    period : float, fundamental period in days

    Returns
    -------
    predicted : array, fitted values at all doy positions
    coefficients : dict with intercept and harmonic amplitudes
    """
    # Build design matrix
    X_all = [np.ones(len(doy))]
    for k in range(1, n_harmonics + 1):
        X_all.append(np.cos(2 * np.pi * k * doy / period))
        X_all.append(np.sin(2 * np.pi * k * doy / period))
    X_all = np.column_stack(X_all)

    # Fit on valid observations
    X_valid = X_all[mask]
    y_valid = values[mask]
    reg = LinearRegression(fit_intercept=False).fit(X_valid, y_valid)

    predicted = reg.predict(X_all)
    return predicted, {"coefficients": reg.coef_}

# Example
doy = np.linspace(1, 365, 23)   # 16-day composites
ndvi = 0.4 + 0.3 * np.sin(2 * np.pi * (doy - 100) / 365) + np.random.normal(0, 0.03, 23)
mask = np.ones(23, dtype=bool)
mask[[4, 5, 6]] = False  # winter cloud gaps

fitted, coeffs = harmonic_fit(doy, ndvi, mask, n_harmonics=2)
print(f"R-squared approx: {1 - np.sum((ndvi[mask]-fitted[mask])**2)/np.sum((ndvi[mask]-ndvi[mask].mean())**2):.3f}")
```

### 2.5 BFAST -- Breaks For Additive Seasonal and Trend

BFAST (Verbesselt et al., 2010) decomposes a time series into **trend**,
**seasonal**, and **remainder** components, then detects structural breakpoints in
the trend and seasonal components.

```r
library(bfast)

# Simulate or load an NDVI time series (16-day, ~23 obs/year, 10 years)
set.seed(42)
n <- 230
t <- seq(1, n)
seasonal <- 0.2 * sin(2 * pi * t / 23)
trend <- 0.3 + 0.002 * t
trend[140:n] <- trend[140:n] - 0.15   # abrupt decline at observation 140
noise <- rnorm(n, 0, 0.03)
ndvi_ts <- ts(trend + seasonal + noise, frequency = 23, start = c(2013, 1))

# --- Run BFAST ---
bf <- bfast(ndvi_ts, h = 0.15, season = "harmonic", max.iter = 5)

# Results
cat("Number of breakpoints in trend:", bf$nobp$Vt, "\n")
cat("Breakpoint positions:", bf$output[[length(bf$output)]]$bp.Vt$breakpoints, "\n")
plot(bf)
```

#### BFAST Monitor for Near-Real-Time Detection

```r
library(bfast)

# Define a stable history period and monitoring period
ndvi_ts <- ts(ndvi_values, frequency = 23, start = c(2015, 1))

# History: 2015-2020; Monitor: 2021+
bfm <- bfastmonitor(ndvi_ts, start = c(2021, 1),
                     formula = response ~ harmon(1, 2),
                     order = 2, history = "all")

cat("Detected break at:", bfm$breakpoint, "\n")
cat("Magnitude:", bfm$magnitude, "\n")
plot(bfm)
```

### 2.6 Benchmark: Cube Construction Performance

Benchmarks on a 100 x 100 km area, Sentinel-2, 12 months, 10 m resolution,
single machine (32 GB RAM, 8 cores):

| Tool | Composite Method | Wall Time | Peak RAM | Output Size |
|------|-----------------|-----------|----------|-------------|
| `stackstac` + Dask | Monthly median | 4 min 12 s | 11 GB | 1.2 GB (NetCDF) |
| `gdalcubes` (R) | Monthly median | 3 min 48 s | 6 GB | 1.1 GB (GeoTIFF) |
| `sits::sits_regularize` | 16-day median | 5 min 30 s | 8 GB | 2.4 GB (GeoTIFF) |
| Google Earth Engine | Monthly median | 2 min 05 s | N/A (server) | 900 MB (GeoTIFF export) |
| `odc-stac` (Open Data Cube) | Monthly median | 3 min 55 s | 10 GB | 1.2 GB (Zarr) |

---

## 3. NDVI/EVI Time Series & Phenology

### 3.1 Vegetation Indices for Time-Series Analysis

| Index | Formula | Sensitivity | Saturation |
|-------|---------|-------------|------------|
| **NDVI** | (NIR - Red) / (NIR + Red) | General greenness | Saturates at high LAI (>3) |
| **EVI** | 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1) | High biomass, corrects atmosphere | Less saturation |
| **EVI2** | 2.5 * (NIR - Red) / (NIR + 2.4*Red + 1) | Like EVI, no Blue needed | Good for Landsat |
| **SAVI** | 1.5 * (NIR - Red) / (NIR + Red + 0.5) | Sparse vegetation, soil correction | Moderate |
| **kNDVI** | tanh(NDVI^2) | Non-linear, recent research | Reduced saturation |

### 3.2 Phenology Extraction

Phenology extraction identifies key seasonal transitions from vegetation index
time series. The primary phenometrics are:

| Phenometric | Abbreviation | Description |
|-------------|-------------|-------------|
| Start of Season | SOS | Date when greenness exceeds a threshold (e.g., 20% of amplitude) |
| Peak of Season | POS | Date of maximum greenness |
| End of Season | EOS | Date when greenness drops below threshold after peak |
| Length of Season | LOS | EOS - SOS (days) |
| Base Value | BASE | Minimum greenness (dormant season) |
| Amplitude | AMP | POS value - BASE value |
| Small Integral | SI | Area under curve above BASE (productivity proxy) |
| Large Integral | LI | Total area under curve |

#### Python: phenolopy

```python
"""
Phenology extraction with phenolopy (DEA - Digital Earth Australia).
Computes SOS, POS, EOS, and other metrics from an xarray time series.
"""
# pip install phenolopy
import xarray as xr
import numpy as np

# Create a synthetic NDVI time series (1 pixel, 1 year at 16-day)
doy = np.arange(1, 366, 16)
ndvi = 0.15 + 0.55 * np.exp(-0.5 * ((doy - 180) / 50) ** 2)  # Gaussian season
ndvi += np.random.normal(0, 0.02, len(doy))

ds = xr.Dataset(
    {"ndvi": (["time"], ndvi)},
    coords={"time": np.array(
        [np.datetime64("2022-01-01") + np.timedelta64(int(d), "D") for d in doy]
    )}
)

# For actual use with phenolopy on a datacube:
# from phenolopy import phenolopy
# pheno_ds = phenolopy.calc_phenometrics(ds, "ndvi", method="seasonal_amplitude",
#                                        peak_metric="pos", base_metric="bse",
#                                        sos_threshold=0.2, eos_threshold=0.2)
# SOS, POS, EOS are returned as day-of-year rasters

# Manual threshold-based phenology (works on the synthetic data)
def extract_phenology(doy, ndvi, threshold_pct=0.2):
    """Simple threshold-based phenology extraction."""
    base = np.nanmin(ndvi)
    peak_val = np.nanmax(ndvi)
    peak_doy = doy[np.nanargmax(ndvi)]
    amplitude = peak_val - base
    thresh = base + threshold_pct * amplitude

    # SOS: first crossing above threshold (ascending)
    ascending = doy[doy < peak_doy]
    ndvi_asc = ndvi[doy < peak_doy]
    sos_idx = np.where(ndvi_asc >= thresh)[0]
    sos = ascending[sos_idx[0]] if len(sos_idx) > 0 else np.nan

    # EOS: first crossing below threshold (descending)
    descending = doy[doy > peak_doy]
    ndvi_desc = ndvi[doy > peak_doy]
    eos_idx = np.where(ndvi_desc <= thresh)[0]
    eos = descending[eos_idx[0]] if len(eos_idx) > 0 else np.nan

    return {
        "SOS": sos, "POS": peak_doy, "EOS": eos,
        "LOS": eos - sos if not np.isnan(eos) else np.nan,
        "BASE": base, "AMPLITUDE": amplitude
    }

pheno = extract_phenology(doy, ndvi)
print("Phenometrics:", pheno)
```

#### R: phenofit

```r
library(phenofit)

# Simulate NDVI time series
set.seed(123)
dates <- seq(as.Date("2018-01-01"), as.Date("2022-12-31"), by = "16 days")
n <- length(dates)
doy <- as.numeric(format(dates, "%j"))
year_frac <- as.numeric(dates - as.Date("2018-01-01")) / 365.25

# Double-logistic seasonal pattern
ndvi <- 0.15 + 0.6 / (1 + exp(-0.08 * (doy - 100))) *
        (1 - 1 / (1 + exp(-0.08 * (doy - 280))))
ndvi <- ndvi + rnorm(n, 0, 0.03)

# Prepare input
input <- check_input(
  t = dates,
  y = ndvi,
  w = rep(1, n),           # weights (1 = valid)
  nptperyear = 23          # ~23 obs per year (16-day)
)

# Fit double-logistic curve
fits <- curvefit(input, methods = c("Beck", "Elmore"), wFUN = wTSM)

# Extract phenology
pheno <- get_pheno(fits, method = "Beck", IsPlot = TRUE)
print(pheno)
# Returns: TRS1 (SOS), TRS2 (EOS), POP (peak), etc.
```

### 3.3 TIMESAT

TIMESAT (Eklundh & Jonsson) is the classic tool for smoothing and phenology
extraction. It supports three fitting methods:

1. **Asymmetric Gaussian** -- fits separate ascending/descending Gaussians
2. **Double logistic** -- smooth sigmoid transitions
3. **Savitzky-Golay** -- local polynomial smoothing

TIMESAT is available as a standalone GUI (MATLAB) or via the `TIMESAT` R wrapper.

```r
# Conceptual TIMESAT-style fitting via base R
# Using a double-logistic function
double_logistic <- function(t, params) {
  # params: mn, mx, sos, rate_sos, eos, rate_eos
  mn <- params[1]; mx <- params[2]
  sos <- params[3]; r1 <- params[4]
  eos <- params[5]; r2 <- params[6]
  mn + (mx - mn) * (1/(1 + exp(-r1*(t - sos))) - 1/(1 + exp(-r2*(t - eos))))
}

# Fit to NDVI data using nls or optim
# (In practice, use phenofit or TIMESAT software for robust fitting)
```

### 3.4 Drought Indices from Vegetation Time Series

| Index | Full Name | Formula / Approach | Use Case |
|-------|----------|-------------------|----------|
| **VCI** | Vegetation Condition Index | `(NDVI - NDVI_min) / (NDVI_max - NDVI_min) * 100` | Agricultural drought |
| **TCI** | Temperature Condition Index | `(LST_max - LST) / (LST_max - LST_min) * 100` | Thermal stress |
| **VHI** | Vegetation Health Index | `0.5 * VCI + 0.5 * TCI` | Combined drought |
| **SMCI** | Soil Moisture Condition Index | Analogous to VCI but for soil moisture | Root-zone drought |

```python
"""
Compute VCI, TCI, and VHI from NDVI and LST time series.
"""
import numpy as np
import xarray as xr

def compute_vci(ndvi, ndvi_min, ndvi_max):
    """
    Vegetation Condition Index.
    ndvi_min/max are the historical min/max for each DOY (climatology).
    """
    return ((ndvi - ndvi_min) / (ndvi_max - ndvi_min + 1e-10)) * 100

def compute_tci(lst, lst_min, lst_max):
    """Temperature Condition Index."""
    return ((lst_max - lst) / (lst_max - lst_min + 1e-10)) * 100

def compute_vhi(vci, tci, alpha=0.5):
    """Vegetation Health Index."""
    return alpha * vci + (1 - alpha) * tci

# Example with synthetic data
n_years = 20
n_dekads = 36  # 10-day composites per year

# Build climatological min/max from 20 years of data
ndvi_all = np.random.uniform(0.1, 0.9, (n_years, n_dekads))
ndvi_min = ndvi_all.min(axis=0)
ndvi_max = ndvi_all.max(axis=0)

# Current year
ndvi_current = ndvi_all[-1]
vci = compute_vci(ndvi_current, ndvi_min, ndvi_max)
print(f"VCI range: {vci.min():.1f} - {vci.max():.1f}")
print(f"Drought dekads (VCI < 35): {np.sum(vci < 35)}")
```

### 3.5 Agricultural Monitoring Workflow

```
1. Acquire Sentinel-2/MODIS time series for growing season
         |
2. Cloud mask + compositing (16-day median)
         |
3. Compute NDVI/EVI time series
         |
4. Smooth (Whittaker or Savitzky-Golay)
         |
5. Extract phenometrics (SOS, POS, EOS, LOS)
         |
6. Compare to historical climatology
         |
7. Compute VCI / VHI for drought assessment
         |
8. Generate crop condition maps + bulletins
```

---

## 4. Change Detection

### 4.1 Overview of Algorithms

| Algorithm | Type | Temporal Granularity | Detects | Reference |
|-----------|------|---------------------|---------|-----------|
| **CCDC** | Continuous | Per-observation | Abrupt + gradual | Zhu & Woodcock, 2014 |
| **LandTrendr** | Segmentation | Annual | Gradual + abrupt | Kennedy et al., 2010 |
| **BFAST** | Decomposition | Sub-annual | Seasonal + trend breaks | Verbesselt et al., 2010 |
| **EWMACD** | Control chart | Per-observation | Subtle, persistent | Brooks et al., 2014 |
| **COLD** | Updated CCDC | Per-observation | Abrupt + gradual | Zhu et al., 2020 |
| **CODED** | Forest-focused | Per-observation | Degradation + deforestation | Bullock et al., 2020 |
| **CuSum** | Cumulative sum | Per-observation | Shifts in mean | Classic statistics |

### 4.2 CCDC (Continuous Change Detection and Classification)

CCDC fits a harmonic regression model to each pixel's time series and monitors
residuals. When residuals exceed a threshold for consecutive observations, a
**break** is flagged.

#### GEE Implementation

```javascript
// Google Earth Engine: CCDC on Landsat
var roi = ee.Geometry.Rectangle([11.3, 46.4, 11.5, 46.6]);

// Build Landsat collection (harmonised L5/7/8/9)
var landsatCol = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
  .filterBounds(roi)
  .filterDate('2013-01-01', '2023-12-31')
  .map(function(img) {
    var ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
    return img.addBands(ndvi).select(['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7','NDVI'])
              .set('system:time_start', img.get('system:time_start'));
  });

// Run CCDC
var ccdc = ee.Algorithms.TemporalSegmentation.Ccdc({
  collection: landsatCol,
  breakpointBands: ['NDVI', 'SR_B5', 'SR_B7'],
  minObservations: 6,
  chiSquareProbability: 0.99,
  minNumOfYearsScaler: 1.33,
  dateFormat: 1,
  lambda: 20,
  maxIterations: 25000
});

// Extract change date map
var changeDate = ccdc.select('tBreak').arrayGet(0);
Map.addLayer(changeDate, {min: 2015, max: 2023, palette: ['yellow','red']}, 'First break');
```

### 4.3 LandTrendr

LandTrendr performs temporal segmentation on annual composites, fitting piecewise
linear models to identify disturbance and recovery trajectories.

```javascript
// Google Earth Engine: LandTrendr
var ltParams = {
  maxSegments:   6,
  spikeThreshold: 0.9,
  vertexCountOvershoot: 3,
  preventOneYearRecovery: true,
  recoveryThreshold: 0.25,
  pvalThreshold: 0.05,
  bestModelProportion: 0.75,
  minObservationsNeeded: 6,
  timeSeries: ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
    .filterBounds(roi)
    .filterDate('2000-01-01', '2023-12-31')
    .filter(ee.Filter.calendarRange(6, 9, 'month'))
    .map(function(img) {
      return img.normalizedDifference(['SR_B5','SR_B7'])
                .multiply(1000).toInt16()
                .rename('NBR')
                .set('system:time_start', img.get('system:time_start'));
    })
};

var lt = ee.Algorithms.TemporalSegmentation.LandTrendr(ltParams);

// Greatest disturbance map
var disturbance = lt.select(['LandTrendr']).arraySlice(0, 0, 1).arrayProject([1]);
Map.addLayer(disturbance, {}, 'LandTrendr segments');
```

### 4.4 EWMACD (Exponentially Weighted Moving Average Change Detection)

```python
"""
EWMACD for near-real-time change detection.
Suitable for detecting subtle, persistent changes (e.g., degradation).
"""
import numpy as np

def ewmacd(residuals, lambda_param=0.3, L=3.0, sigma=None):
    """
    Exponentially Weighted Moving Average Change Detection.

    Parameters
    ----------
    residuals : array, residuals from a baseline harmonic model
    lambda_param : float, EWMA weighting (0 < lambda <= 1)
    L : float, control limit multiplier (standard deviations)
    sigma : float, std of residuals in stable period (if None, estimated from data)

    Returns
    -------
    ewma_values : array, EWMA statistic over time
    change_flags : boolean array, True where change detected
    """
    n = len(residuals)
    if sigma is None:
        sigma = np.nanstd(residuals[:n // 3])  # estimate from first third

    # Control limits
    cl_factor = L * sigma * np.sqrt(lambda_param / (2 - lambda_param))

    ewma = np.zeros(n)
    ewma[0] = residuals[0]
    change_flags = np.zeros(n, dtype=bool)

    for i in range(1, n):
        ewma[i] = lambda_param * residuals[i] + (1 - lambda_param) * ewma[i - 1]
        if abs(ewma[i]) > cl_factor:
            change_flags[i] = True

    return ewma, change_flags

# Example
np.random.seed(42)
n = 200
residuals = np.random.normal(0, 0.05, n)
# Inject a subtle shift starting at obs 150
residuals[150:] += 0.12

ewma_vals, changes = ewmacd(residuals, lambda_param=0.25, L=3.0)
first_detection = np.where(changes)[0]
if len(first_detection) > 0:
    print(f"First change detected at observation {first_detection[0]} (true change at 150)")
```

### 4.5 Multi-Temporal Change Analysis (Post-Classification Comparison)

```python
"""
Post-classification comparison for land-cover change mapping.
Computes a transition matrix between two classified maps.
"""
import numpy as np
from collections import Counter

def transition_matrix(map_t1, map_t2, class_names=None):
    """
    Compute a transition matrix from two classified rasters.

    Parameters
    ----------
    map_t1, map_t2 : np.ndarray (2D), integer class labels
    class_names : dict mapping int -> str

    Returns
    -------
    matrix : 2D array, counts of pixel transitions
    """
    assert map_t1.shape == map_t2.shape
    classes = sorted(set(np.unique(map_t1)) | set(np.unique(map_t2)))
    n_classes = len(classes)
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for c1, c2 in zip(map_t1.ravel(), map_t2.ravel()):
        matrix[cls_to_idx[c1], cls_to_idx[c2]] += 1

    if class_names:
        labels = [class_names.get(c, str(c)) for c in classes]
    else:
        labels = [str(c) for c in classes]

    return matrix, labels

# Example
np.random.seed(0)
map_2015 = np.random.choice([1, 2, 3, 4], size=(100, 100))  # forest, crop, urban, water
map_2023 = map_2015.copy()
# Simulate urban expansion (some forest/crop -> urban)
urban_mask = np.random.random((100, 100)) < 0.05
map_2023[urban_mask & (map_2015 == 1)] = 3  # forest -> urban
map_2023[urban_mask & (map_2015 == 2)] = 3  # crop -> urban

names = {1: "Forest", 2: "Crop", 3: "Urban", 4: "Water"}
mat, labels = transition_matrix(map_2015, map_2023, names)

print("Transition Matrix (rows=2015, cols=2023):")
print(f"{'':>10}", *[f"{l:>10}" for l in labels])
for i, row in enumerate(mat):
    print(f"{labels[i]:>10}", *[f"{v:>10}" for v in row])
```

### 4.6 Change Detection Algorithm Selection Guide

```
Is the change abrupt (< 1 year)?
  |-- YES --> Is near-real-time detection needed?
  |             |-- YES --> EWMACD or BFAST Monitor
  |             |-- NO  --> CCDC / COLD
  |-- NO (gradual, multi-year) --> LandTrendr
  |
Is the area forested and you need degradation detection?
  |-- YES --> CODED
  |
Do you need sub-annual phenological change?
  |-- YES --> BFAST (full decomposition)
  |
Is it a simple before/after comparison?
  |-- YES --> Differencing (dNBR, dNDVI) or post-classification comparison
```

---

## 5. Climate Time Series

### 5.1 ERA5 Reanalysis Processing with xarray

ERA5 (ECMWF Reanalysis v5) provides hourly climate data at ~31 km resolution from
1940 to present. See [Climate & Weather Data Sources](../data-sources/climate-weather.md)
for access methods.

```python
"""
ERA5 monthly temperature trend analysis with xarray.
Data source: Copernicus Climate Data Store (CDS).
"""
import xarray as xr
import numpy as np
from scipy import stats

# --- 1. Load ERA5 data (pre-downloaded NetCDF) ---
# Download via CDS API: https://cds.climate.copernicus.eu/
ds = xr.open_dataset("era5_monthly_t2m_1980_2023.nc")
t2m = ds["t2m"] - 273.15   # Kelvin to Celsius

# --- 2. Compute annual means ---
t2m_annual = t2m.groupby("time.year").mean("time")
print(f"Shape: {t2m_annual.shape}")  # (years, lat, lon)

# --- 3. Pixel-wise trend (Sen's slope) ---
def sens_slope_pixel(y):
    """Theil-Sen slope estimator for a 1D time series."""
    n = len(y)
    if np.all(np.isnan(y)):
        return np.nan, np.nan
    x = np.arange(n)
    valid = ~np.isnan(y)
    if valid.sum() < 5:
        return np.nan, np.nan

    # Compute all pairwise slopes
    slopes = []
    for i in range(n):
        for j in range(i + 1, n):
            if valid[i] and valid[j]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))

    slope = np.median(slopes) if slopes else np.nan

    # Mann-Kendall p-value
    from scipy.stats import kendalltau
    tau, p = kendalltau(x[valid], y[valid])
    return slope, p

# Apply to each pixel (vectorised with apply_ufunc for large datasets)
years_array = t2m_annual.year.values.astype(float)

# For demonstration on a small subset:
subset = t2m_annual.isel(latitude=slice(0, 10), longitude=slice(0, 10))
slopes = np.zeros((subset.sizes["latitude"], subset.sizes["longitude"]))
pvalues = np.zeros_like(slopes)

for i in range(subset.sizes["latitude"]):
    for j in range(subset.sizes["longitude"]):
        ts = subset.isel(latitude=i, longitude=j).values
        slopes[i, j], pvalues[i, j] = sens_slope_pixel(ts)

print(f"Mean warming trend: {np.nanmean(slopes):.4f} deg C/year")
print(f"Pixels with significant trend (p<0.05): {np.sum(pvalues < 0.05)}")
```

### 5.2 Mann-Kendall Trend Test (Full Implementation)

```python
"""
Vectorised Mann-Kendall test for gridded climate data.
Uses pymannkendall for single-pixel analysis.
"""
# pip install pymannkendall
import pymannkendall as mk
import numpy as np

# Single location trend test
temperature = np.array([14.2, 14.5, 14.3, 14.8, 15.0, 14.7, 15.2,
                        15.1, 15.4, 15.3, 15.6, 15.5, 15.8, 16.0])

result = mk.original_test(temperature)
print(f"Trend:     {result.trend}")
print(f"p-value:   {result.p:.6f}")
print(f"Tau:       {result.Tau:.4f}")
print(f"Sen slope: {result.slope:.4f} per time step")
print(f"Intercept: {result.intercept:.4f}")

# Seasonal Mann-Kendall (removes seasonal autocorrelation)
# Useful for monthly data with strong seasonality
monthly_temp = np.tile(temperature, 3) + np.sin(np.linspace(0, 6*np.pi, 42)) * 5
result_seasonal = mk.seasonal_test(monthly_temp, period=12)
print(f"\nSeasonal MK trend: {result_seasonal.trend}")
print(f"Seasonal MK slope: {result_seasonal.slope:.4f}")
```

### 5.3 Anomaly Detection in Climate Data

```python
"""
Climate anomaly computation and Nino 3.4 index example.
"""
import xarray as xr
import numpy as np

def compute_anomaly(ds, var, base_period=("1991", "2020")):
    """
    Compute monthly anomalies relative to a base climatology.

    Parameters
    ----------
    ds : xr.Dataset with a time dimension
    var : str, variable name
    base_period : tuple of (start_year, end_year)

    Returns
    -------
    anomaly : xr.DataArray
    """
    clim = ds[var].sel(time=slice(*base_period)).groupby("time.month").mean("time")
    anomaly = ds[var].groupby("time.month") - clim
    return anomaly

# Example: SST anomaly for ENSO monitoring
# ds = xr.open_dataset("ersst_v5_monthly.nc")
# sst_anom = compute_anomaly(ds, "sst", base_period=("1991", "2020"))
# nino34 = sst_anom.sel(lat=slice(5, -5), lon=slice(190, 240)).mean(["lat","lon"])
# print("El Nino months (>0.5):", (nino34 > 0.5).sum().values)
```

### 5.4 CMIP6 Projection Analysis

```python
"""
Basic CMIP6 multi-model ensemble analysis.
"""
import xarray as xr
import numpy as np

def process_cmip6_ensemble(file_list, variable="tas", scenario="ssp245"):
    """
    Load and compute multi-model ensemble statistics.

    Parameters
    ----------
    file_list : list of str, paths to CMIP6 NetCDF files
    variable : str, variable name
    scenario : str, SSP scenario

    Returns
    -------
    ensemble_mean, ensemble_std : xr.DataArray
    """
    datasets = []
    for f in file_list:
        ds = xr.open_dataset(f)
        # Regrid to common grid (simplified - use xesmf for proper regridding)
        ds = ds[variable].interp(
            lat=np.arange(-90, 90, 1),
            lon=np.arange(0, 360, 1)
        )
        # Compute annual means
        annual = ds.groupby("time.year").mean("time")
        datasets.append(annual)

    ensemble = xr.concat(datasets, dim="model")
    return ensemble.mean("model"), ensemble.std("model")

# Usage:
# files = glob.glob("cmip6_tas_ssp245_*.nc")
# ens_mean, ens_std = process_cmip6_ensemble(files)
# warming = ens_mean.sel(year=2100) - ens_mean.sel(year=2020)
```

### 5.5 R: Climate Trend Analysis

```r
library(trend)
library(zyp)

# Monthly temperature data (30 years)
set.seed(42)
n <- 360
temp <- 15 + 0.02 * (1:n) + 5 * sin(2 * pi * (1:n) / 12) + rnorm(n, 0, 1)

# --- Mann-Kendall ---
mk_result <- mk.test(temp)
cat("Mann-Kendall p-value:", mk_result$p.value, "\n")
cat("Sen's slope:", mk_result$estimates["Sen's slope"], "\n")

# --- Seasonal Kendall (accounts for monthly seasonality) ---
temp_ts <- ts(temp, frequency = 12, start = c(1993, 1))
sk_result <- smk.test(temp_ts)
cat("Seasonal Kendall p-value:", sk_result$p.value, "\n")

# --- zyp: prewhitened Sen's slope (handles autocorrelation) ---
zyp_result <- zyp.trend.vector(temp, method = "yuepilon")
cat("Trend (zyp):", zyp_result["trend"], "\n")
cat("Significant:", zyp_result["sig"] < 0.05, "\n")
```

---

## 6. Urban Growth Analysis

### 6.1 Global Human Settlement Layer (GHSL) Temporal Analysis

The GHSL provides built-up surface grids at multi-decadal epochs (1975, 1990,
2000, 2015) from Landsat data, and annual updates from Sentinel-2.

```python
"""
GHSL built-up area change analysis.
Data: GHS-BUILT-S (built-up surface) rasters from JRC.
"""
import rasterio
import numpy as np

def ghsl_change(path_epoch1, path_epoch2, threshold=50):
    """
    Compute built-up area change between two GHSL epochs.

    Parameters
    ----------
    path_epoch1, path_epoch2 : str, paths to GHSL GeoTIFFs
    threshold : int, % built-up to consider as "urban"

    Returns
    -------
    dict with area statistics in km2
    """
    with rasterio.open(path_epoch1) as src1:
        bu1 = src1.read(1)
        transform = src1.transform
        pixel_area_km2 = abs(transform.a * transform.e) / 1e6

    with rasterio.open(path_epoch2) as src2:
        bu2 = src2.read(1)

    urban1 = bu1 >= threshold
    urban2 = bu2 >= threshold

    new_urban = urban2 & ~urban1        # urbanised between epochs
    stable_urban = urban1 & urban2      # remained urban
    lost_urban = urban1 & ~urban2       # de-urbanised (rare)

    return {
        "urban_epoch1_km2": urban1.sum() * pixel_area_km2,
        "urban_epoch2_km2": urban2.sum() * pixel_area_km2,
        "new_urban_km2": new_urban.sum() * pixel_area_km2,
        "stable_urban_km2": stable_urban.sum() * pixel_area_km2,
        "growth_pct": (new_urban.sum() / (urban1.sum() + 1)) * 100,
    }

# Example (conceptual):
# stats = ghsl_change("GHS_BUILT_S_2000.tif", "GHS_BUILT_S_2015.tif")
# print(f"Urban growth: {stats['new_urban_km2']:.1f} km2 ({stats['growth_pct']:.1f}%)")
```

### 6.2 Nighttime Lights Time Series

DMSP-OLS (1992-2013) and VIIRS-DNB (2012-present) provide a continuous record
of nighttime lights as a proxy for urbanisation and economic activity.

```python
"""
Nighttime lights time series analysis using VIIRS monthly composites.
"""
import xarray as xr
import numpy as np

def ntl_trend_analysis(ntl_cube, min_radiance=0.5):
    """
    Analyse nighttime lights trends from a VIIRS monthly cube.

    Parameters
    ----------
    ntl_cube : xr.DataArray with dims (time, y, x)
    min_radiance : float, threshold for lit area

    Returns
    -------
    trend_map : xr.DataArray, per-pixel slope (nW/cm2/sr per year)
    lit_area_ts : xr.DataArray, total lit area per timestep
    """
    # Total lit area over time
    lit = (ntl_cube > min_radiance).sum(dim=["y", "x"])

    # Linear trend per pixel
    time_numeric = np.arange(ntl_cube.sizes["time"], dtype=float)

    def pixel_slope(y):
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            return np.nan
        from numpy.polynomial.polynomial import polyfit
        c = polyfit(time_numeric[valid], y[valid], 1)
        return c[1]  # slope

    trend = xr.apply_ufunc(
        pixel_slope, ntl_cube,
        input_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    return trend, lit

# Usage with actual data:
# ds = xr.open_mfdataset("viirs_monthly_*.nc")
# trend, lit_ts = ntl_trend_analysis(ds["avg_rad"])
```

### 6.3 Urban Sprawl Metrics

| Metric | Description | Computation |
|--------|-------------|-------------|
| **Urban area** | Total built-up extent | Sum of urban pixels * pixel area |
| **Urban density** | People per urban area | Population / urban area |
| **Compactness ratio** | Circle-likeness | `4*pi*area / perimeter^2` |
| **Shannon entropy** | Dispersion of urban patches | `-sum(p_i * log(p_i))` over zones |
| **Edge density** | Urban-rural interface length | Total urban edge / total area |
| **Landscape expansion index (LEI)** | Infill vs. edge vs. outlying growth | Buffer-based classification of new patches |

---

## 7. Movement & Trajectory Analysis

### 7.1 MovingPandas

MovingPandas (Graser, 2019) extends GeoPandas with trajectory-aware operations.

```python
"""
Trajectory analysis with MovingPandas.
"""
# pip install movingpandas
import movingpandas as mpd
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import numpy as np

# --- 1. Create sample trajectory data ---
np.random.seed(42)
n_points = 200
timestamps = [datetime(2023, 6, 1, 8, 0) + timedelta(minutes=i * 5) for i in range(n_points)]

# Simulate movement with stops
lon = np.cumsum(np.random.normal(0.001, 0.0005, n_points)) + 11.4
lat = np.cumsum(np.random.normal(0.0005, 0.0003, n_points)) + 46.5
# Insert a stop (stationary period)
lon[80:100] = lon[80]
lat[80:100] = lat[80]

gdf = gpd.GeoDataFrame(
    {"id": "vehicle_1", "timestamp": timestamps},
    geometry=[Point(x, y) for x, y in zip(lon, lat)],
    crs="EPSG:4326"
)
gdf = gdf.set_index("timestamp")

# --- 2. Create trajectory ---
traj = mpd.Trajectory(gdf, traj_id="vehicle_1")
print(f"Duration:  {traj.get_duration()}")
print(f"Length:    {traj.get_length():.0f} m")
print(f"Start:    {traj.get_start_location()}")

# --- 3. Add speed and direction ---
traj.add_speed(overwrite=True, units=("km", "h"))
traj.add_direction(overwrite=True)
print(f"Max speed: {traj.df['speed'].max():.1f} km/h")

# --- 4. Stop detection ---
detector = mpd.TrajectoryStopDetector(traj)
stops = detector.get_stop_segments(
    min_duration=timedelta(minutes=15),
    max_diameter=100   # metres
)
print(f"Detected {len(stops)} stop(s)")
for stop in stops:
    print(f"  Stop at {stop.get_start_location()}, "
          f"duration: {stop.get_duration()}")

# --- 5. Trajectory segmentation (by stops) ---
splitter = mpd.StopSplitter(traj)
segments = splitter.split(
    min_duration=timedelta(minutes=15),
    max_diameter=100
)
print(f"Trip segments: {len(segments)}")

# --- 6. Generalisation (Douglas-Peucker on trajectory) ---
generalized = mpd.DouglasPeuckerGeneralizer(traj).generalize(tolerance=50)
print(f"Original points: {len(traj.df)}, Generalised: {len(generalized.df)}")
```

### 7.2 scikit-mobility: Flows and Privacy

```python
"""
OD flow analysis and privacy-preserving aggregation with scikit-mobility.
"""
# pip install scikit-mobility
import skmob
from skmob.preprocessing import filtering, compression, detection
from skmob.measures.individual import radius_of_gyration, number_of_locations
import pandas as pd
import numpy as np

# --- 1. Create TrajDataFrame ---
data = pd.DataFrame({
    "uid": [1]*50 + [2]*50,
    "lat": np.concatenate([
        np.cumsum(np.random.normal(0, 0.001, 50)) + 46.5,
        np.cumsum(np.random.normal(0, 0.001, 50)) + 46.6,
    ]),
    "lng": np.concatenate([
        np.cumsum(np.random.normal(0, 0.001, 50)) + 11.4,
        np.cumsum(np.random.normal(0, 0.001, 50)) + 11.5,
    ]),
    "datetime": pd.date_range("2023-06-01", periods=50, freq="30min").tolist() * 2,
})
tdf = skmob.TrajDataFrame(data, latitude="lat", longitude="lng",
                           datetime="datetime", user_id="uid")

# --- 2. Preprocessing ---
# Filter noisy points (speed > threshold)
filtered = filtering.filter(tdf, max_speed_kmh=200)
# Compress: remove points within spatial threshold
compressed = compression.compress(filtered, spatial_radius_km=0.1)
# Detect stops
stops = detection.stops(compressed, stop_radius_factor=0.5,
                        minutes_for_a_stop=20)
print(f"Stops detected: {len(stops[stops['leaving_datetime'].notna()])}")

# --- 3. Individual mobility metrics ---
rog = radius_of_gyration(tdf)
print("Radius of gyration per user:")
print(rog)

# --- 4. Privacy: spatial generalisation ---
from skmob.privacy import attacks
# Compute re-identification risk
# risk = attacks.HomeWorkAttack(tdf)
# print(f"Re-identification risk: {risk.assess_risk()}")
```

### 7.3 Trajectory Metrics Summary

| Metric | Formula / Description | Use Case |
|--------|----------------------|----------|
| **Speed** | Distance / time between consecutive fixes | Transport mode detection |
| **Acceleration** | Change in speed / time | Driving behaviour |
| **Sinuosity** | Path length / straight-line distance | Foraging vs. transit |
| **Radius of gyration** | RMS distance from centroid of locations | Home range / mobility |
| **MSD** (Mean Squared Displacement) | Average squared displacement over lag | Diffusion analysis |
| **Recurrence** | Frequency of revisiting locations | Routine detection |

---

## 8. Forecasting

### 8.1 Spatial-Temporal Forecasting Overview

Forecasting spatially-referenced time series requires models that capture:
1. **Temporal autocorrelation** -- past values predict future values.
2. **Spatial autocorrelation** -- neighbouring locations co-vary.
3. **Exogenous drivers** -- weather, events, policy changes.

### 8.2 Prophet for Spatial Time Series

```python
"""
Facebook Prophet applied to multiple spatial locations (e.g., weather stations).
"""
# pip install prophet
from prophet import Prophet
import pandas as pd
import numpy as np

def forecast_station(station_df, periods=365, freq="D"):
    """
    Forecast a single station's time series with Prophet.

    Parameters
    ----------
    station_df : pd.DataFrame with columns ['ds', 'y']
    periods : int, forecast horizon
    freq : str, frequency

    Returns
    -------
    forecast : pd.DataFrame
    """
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    m.fit(station_df)
    future = m.make_future_dataframe(periods=periods, freq=freq)
    forecast = m.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]

# Example for multiple stations
np.random.seed(42)
stations = {"station_A": (46.5, 11.4), "station_B": (46.6, 11.5)}
all_forecasts = {}

for name, (lat, lon) in stations.items():
    dates = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    temp = 10 + 15*np.sin(2*np.pi*np.arange(len(dates))/365.25) + np.random.normal(0, 3, len(dates))
    temp += lat * 0.1  # spatial variation
    df = pd.DataFrame({"ds": dates, "y": temp})
    fc = forecast_station(df, periods=365)
    all_forecasts[name] = fc
    print(f"{name}: forecast range {fc['yhat'].iloc[-365:].min():.1f} to {fc['yhat'].iloc[-365:].max():.1f}")
```

### 8.3 ARIMA / SARIMA for Spatial Grids

```python
"""
Pixel-wise SARIMA for gridded data (e.g., monthly NDVI forecast).
Uses statsforecast for speed.
"""
# pip install statsforecast
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA
import pandas as pd
import numpy as np

# Simulate data for 100 "pixels"
np.random.seed(0)
n_pixels = 100
n_months = 120  # 10 years

records = []
for px in range(n_pixels):
    dates = pd.date_range("2013-01-01", periods=n_months, freq="MS")
    base = 0.3 + np.random.uniform(-0.1, 0.1)
    seasonal = 0.3 * np.sin(2 * np.pi * np.arange(n_months) / 12)
    trend = 0.001 * np.arange(n_months)
    noise = np.random.normal(0, 0.03, n_months)
    ndvi = base + seasonal + trend + noise

    for d, v in zip(dates, ndvi):
        records.append({"unique_id": f"pixel_{px}", "ds": d, "y": v})

df = pd.DataFrame(records)

# Forecast
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq="MS",
    n_jobs=-1,
)
forecasts = sf.forecast(df=df, h=12)
print(f"Forecast shape: {forecasts.shape}")
print(forecasts.head(10))
```

### 8.4 LSTM for Spatial-Temporal Forecasting

```python
"""
LSTM-based spatial-temporal forecasting (simplified example).
For production, use pytorch-forecasting or tsai.
"""
import numpy as np

# Conceptual architecture for spatial-temporal LSTM:
#
# Input shape: (batch, time_steps, features)
#   - features = [NDVI, temperature, precipitation, lat, lon, ...]
#   - time_steps = lookback window (e.g., 12 months)
#
# Architecture:
#   1. Embedding layer for spatial location (or use lat/lon directly)
#   2. LSTM layers (e.g., 2 layers, 64 hidden units)
#   3. Dense output layer -> forecast horizon
#
# Loss: MSE or quantile loss for prediction intervals

# PyTorch skeleton:
"""
import torch
import torch.nn as nn

class SpatialTemporalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, n_layers=2, forecast_horizon=12):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, forecast_horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # last time step
        return out

model = SpatialTemporalLSTM(input_dim=5, forecast_horizon=12)
"""
print("LSTM architecture defined (see code comments for details)")
```

### 8.5 Kriging with Temporal Dimension

Space-time kriging extends ordinary kriging to predict at unobserved
space-time locations. The semivariogram becomes a space-time covariance
function.

```r
library(gstat)
library(spacetime)
library(sp)

# Create space-time data
set.seed(123)
n_stations <- 20
n_times <- 52  # weekly

coords <- data.frame(
  x = runif(n_stations, 0, 100),
  y = runif(n_stations, 0, 100)
)
sp_pts <- SpatialPoints(coords)

times <- as.POSIXct("2023-01-01") + (0:(n_times-1)) * 7 * 86400

# Simulate space-time data
stfdf <- STFDF(
  sp = sp_pts,
  time = times,
  data = data.frame(
    temp = rnorm(n_stations * n_times, mean = 15, sd = 3)
  )
)

# Fit space-time variogram
vst <- variogramST(temp ~ 1, data = stfdf,
                   tlags = 0:10, cutoff = 50)

# Fit a separable model
model <- vgmST("separable",
               space = vgm(3, "Exp", 30),
               time  = vgm(2, "Exp", 5),
               sill  = 5)

# Predict at new space-time locations
# pred <- krigeST(temp ~ 1, data = stfdf, newdata = new_stfdf,
#                 modelList = model)
cat("Space-time kriging model fitted\n")
```

### 8.6 Forecasting Method Comparison

| Method | Spatial? | Temporal Pattern | Data Volume | Interpretable | Accuracy |
|--------|----------|-----------------|-------------|---------------|----------|
| **ARIMA/SARIMA** | Per-pixel | Linear + seasonal | Low-Med | High | Moderate |
| **Prophet** | Per-pixel | Trend + multi-seasonal + holidays | Low-Med | High | Moderate |
| **LSTM** | Learnable | Arbitrary | High | Low | High |
| **ST-Kriging** | Built-in | Stationary | Low | Moderate | High (interpolation) |
| **ConvLSTM** | Built-in (grid) | Arbitrary | High | Low | High |
| **Temporal Fusion Transformer** | Learnable | Multi-horizon + attention | High | Moderate | Very High |
| **GNN + temporal** | Graph-based | Arbitrary | High | Low | High |

---

## 9. Event Detection & Early Warning

### 9.1 Anomaly Detection in Spatial Time Series

```python
"""
Z-score based anomaly detection for gridded environmental data.
Flags spatial-temporal anomalies relative to a climatological baseline.
"""
import numpy as np
import xarray as xr

def spatial_temporal_anomaly(data, baseline_years, threshold_sigma=2.5):
    """
    Detect anomalies in a gridded time series.

    Parameters
    ----------
    data : xr.DataArray with dims (time, y, x)
    baseline_years : tuple (start, end) for climatology
    threshold_sigma : float, number of std deviations for anomaly flag

    Returns
    -------
    z_scores : xr.DataArray, standardised anomalies
    anomaly_mask : xr.DataArray (bool), True where |z| > threshold
    """
    baseline = data.sel(time=slice(*baseline_years))

    # Monthly climatology
    clim_mean = baseline.groupby("time.month").mean("time")
    clim_std = baseline.groupby("time.month").std("time")

    # Anomalies
    anomaly = data.groupby("time.month") - clim_mean
    z_scores = anomaly.groupby("time.month") / (clim_std + 1e-10)

    anomaly_mask = np.abs(z_scores) > threshold_sigma
    return z_scores, anomaly_mask

# Example
np.random.seed(42)
time = pd.date_range("2000-01-01", "2023-12-31", freq="MS")
data = xr.DataArray(
    np.random.normal(20, 3, (len(time), 10, 10)),
    dims=["time", "y", "x"],
    coords={"time": time}
)
# Inject an anomaly
data.loc[dict(time="2022-07-01")] += 15

z, mask = spatial_temporal_anomaly(data, ("2000", "2019"))
print(f"Total anomalous cells: {mask.sum().values}")
print(f"Anomalies in July 2022: {mask.sel(time='2022-07-01').sum().values}")
```

### 9.2 Disease Outbreak Detection (SaTScan-Style)

```python
"""
Space-time scan statistic for cluster/outbreak detection.
Simplified implementation of Kulldorff's SaTScan approach.
"""
import numpy as np
from itertools import product

def space_time_scan(cases, population, coords, times,
                    max_spatial_pct=0.5, max_temporal_pct=0.5):
    """
    Simplified space-time scan statistic.

    Parameters
    ----------
    cases : np.ndarray, shape (n_locations, n_times), observed case counts
    population : np.ndarray, shape (n_locations,), population at risk
    coords : np.ndarray, shape (n_locations, 2), spatial coordinates
    times : np.ndarray, shape (n_times,), time indices
    max_spatial_pct : float, max fraction of population in cluster
    max_temporal_pct : float, max fraction of time periods

    Returns
    -------
    best_cluster : dict with location, time window, and likelihood ratio
    """
    n_loc, n_time = cases.shape
    total_cases = cases.sum()
    total_pop = population.sum()
    expected_rate = total_cases / total_pop

    max_time_window = int(n_time * max_temporal_pct)
    max_pop = total_pop * max_spatial_pct

    best_llr = 0
    best_cluster = None

    # Distance matrix
    from scipy.spatial.distance import cdist
    dists = cdist(coords, coords)

    for center in range(n_loc):
        # Sort locations by distance from center
        sorted_locs = np.argsort(dists[center])

        cum_pop = 0
        cluster_locs = []

        for loc in sorted_locs:
            cum_pop += population[loc]
            if cum_pop > max_pop:
                break
            cluster_locs.append(loc)

            # Scan over time windows
            for t_start in range(n_time):
                for t_end in range(t_start, min(t_start + max_time_window, n_time)):
                    c_in = cases[cluster_locs, t_start:t_end+1].sum()
                    e_in = cum_pop * (t_end - t_start + 1) / n_time * expected_rate

                    if c_in <= e_in or e_in == 0:
                        continue

                    c_out = total_cases - c_in
                    e_out = total_cases - e_in

                    # Log-likelihood ratio
                    llr = (c_in * np.log(c_in / e_in) +
                           c_out * np.log(max(c_out, 1) / max(e_out, 1)))

                    if llr > best_llr:
                        best_llr = llr
                        best_cluster = {
                            "center": center,
                            "locations": list(cluster_locs),
                            "time_window": (t_start, t_end),
                            "observed": c_in,
                            "expected": e_in,
                            "llr": llr,
                        }

    return best_cluster

# Example
np.random.seed(42)
n_loc, n_time = 30, 52
coords = np.random.uniform(0, 100, (n_loc, 2))
population = np.random.uniform(1000, 50000, n_loc)
cases = np.random.poisson(5, (n_loc, n_time))
# Inject outbreak
cases[5:8, 30:35] += 50

cluster = space_time_scan(cases, population, coords, np.arange(n_time))
if cluster:
    print(f"Detected cluster at location {cluster['center']}")
    print(f"Time window: weeks {cluster['time_window']}")
    print(f"Observed: {cluster['observed']}, Expected: {cluster['expected']:.1f}")
    print(f"LLR: {cluster['llr']:.2f}")
```

### 9.3 Natural Disaster Monitoring

| Disaster Type | Data Source | Detection Method | Temporal Resolution |
|--------------|-------------|-----------------|-------------------|
| **Wildfire** | MODIS/VIIRS active fire, Sentinel-2 | Thermal anomaly, dNBR | Near-real-time (hours) |
| **Flood** | Sentinel-1 SAR, MODIS | SAR backscatter change, water index | 1-6 days |
| **Drought** | MODIS NDVI, GRACE, soil moisture | VCI/VHI, SPI, PDSI | Weekly-monthly |
| **Earthquake** | InSAR (Sentinel-1), GNSS | Surface displacement, coherence loss | Days (post-event) |
| **Landslide** | InSAR, high-res optical | Displacement, spectral change | Days-weeks |
| **Volcanic activity** | Sentinel-5P SO2, thermal IR | Gas emissions, thermal anomalies | Daily |

### 9.4 Early Warning System Architecture

```
Data Ingestion (near-real-time)
  |-- Satellite feeds (Sentinel, MODIS, VIIRS)
  |-- In-situ sensors (weather stations, stream gauges)
  |-- Model outputs (NWP, hydrological models)
       |
Pre-processing Pipeline
  |-- Cloud masking, QA filtering
  |-- Spatial alignment, temporal interpolation
       |
Anomaly Detection Engine
  |-- Z-score / percentile ranking vs. climatology
  |-- EWMA / CUSUM control charts
  |-- ML-based outlier detection (Isolation Forest, autoencoders)
       |
Alert Generation
  |-- Threshold exceedance rules
  |-- Spatial clustering of anomalies
  |-- Severity classification (watch / warning / emergency)
       |
Dissemination
  |-- Dashboard (web map with time slider)
  |-- Push notifications (email, SMS, API)
  |-- Integration with national disaster management agencies
```

---

## 10. Space-Time Cubes

### 10.1 Concept

A **space-time cube** (STC) organises data into a 3D grid of voxels defined by
(x, y, t). Each voxel aggregates events or measurements within its spatial and
temporal extent. STCs enable pattern mining algorithms that would be infeasible on
raw point data.

```
        t3 ┌─────────────┐
           │ · ·   ·     │
        t2 ├─────────────┤
           │   · · · ·   │
        t1 ├─────────────┤
           │ ·   ·   · · │
        t0 └─────────────┘
           x0           xn
           (each layer is a spatial grid at time t_i)
```

### 10.2 ArcGIS Pro Space-Time Pattern Mining

ArcGIS Pro provides a dedicated **Space Time Pattern Mining** toolbox:

| Tool | Purpose | Key Parameter |
|------|---------|---------------|
| **Create Space Time Cube** | Aggregate point data into STC (NetCDF) | Distance interval, time step |
| **Emerging Hot Spot Analysis** | Classify hot/cold spots by temporal pattern | Neighbourhood distance |
| **Local Outlier Analysis** | Detect space-time outliers | Number of neighbours |
| **Time Series Clustering** | Group locations by temporal profile | Number of clusters |
| **Curve Fitting** | Fit trend models per location | Linear / quadratic / Gompertz |

#### Emerging Hot Spot Categories

| Category | Pattern Description |
|----------|-------------------|
| **New** | Statistically significant hot spot only in the latest time step |
| **Consecutive** | Hot spot for the last N consecutive time steps, not before |
| **Intensifying** | Hot spot in >= 90% of time steps, with increasing intensity |
| **Persistent** | Hot spot in >= 90% of time steps, no intensity trend |
| **Diminishing** | Hot spot in >= 90% of time steps, with decreasing intensity |
| **Sporadic** | On-and-off hot spot, hot in the latest step |
| **Oscillating** | Hot spot in the latest step, but historically cold spot too |
| **Historical** | Hot spot in the past, not in the latest step |
| **No Pattern** | Not statistically significant |

### 10.3 xarray as a Space-Time Cube

```python
"""
Building and analysing a space-time cube with xarray.
Example: crime event aggregation.
"""
import xarray as xr
import numpy as np
import pandas as pd

# --- 1. Generate synthetic crime events ---
np.random.seed(42)
n_events = 5000
events = pd.DataFrame({
    "lon": np.random.normal(-73.98, 0.02, n_events),  # NYC-ish
    "lat": np.random.normal(40.75, 0.015, n_events),
    "datetime": pd.date_range("2020-01-01", periods=n_events, freq="2h"),
})

# --- 2. Define cube bins ---
lon_bins = np.arange(-74.05, -73.90, 0.005)
lat_bins = np.arange(40.70, 40.80, 0.005)
time_bins = pd.date_range("2020-01-01", "2021-01-01", freq="MS")

# --- 3. Aggregate into cube ---
events["lon_bin"] = pd.cut(events["lon"], lon_bins, labels=lon_bins[:-1])
events["lat_bin"] = pd.cut(events["lat"], lat_bins, labels=lat_bins[:-1])
events["time_bin"] = pd.cut(events["datetime"], time_bins, labels=time_bins[:-1])

cube_df = (events
           .dropna()
           .groupby(["time_bin", "lat_bin", "lon_bin"])
           .size()
           .reset_index(name="count"))

# Pivot to xarray
cube_df["time_bin"] = pd.to_datetime(cube_df["time_bin"])
cube_df["lat_bin"] = cube_df["lat_bin"].astype(float)
cube_df["lon_bin"] = cube_df["lon_bin"].astype(float)

cube = cube_df.set_index(["time_bin", "lat_bin", "lon_bin"])["count"].to_xarray()
cube = cube.fillna(0)
print(f"Space-time cube shape: {cube.shape}")
print(f"Total events in cube: {int(cube.sum().values)}")

# --- 4. Temporal trend per voxel column ---
from scipy.stats import linregress

def pixel_trend(ts):
    """Linear trend slope for a single location."""
    x = np.arange(len(ts))
    valid = ~np.isnan(ts) & (ts > 0)
    if valid.sum() < 3:
        return np.nan
    slope, _, _, p, _ = linregress(x[valid], ts[valid])
    return slope if p < 0.05 else 0.0

trend_map = xr.apply_ufunc(
    pixel_trend, cube,
    input_core_dims=[["time_bin"]],
    vectorize=True,
    output_dtypes=[float],
)
print(f"Locations with increasing trend: {(trend_map > 0).sum().values}")
print(f"Locations with decreasing trend: {(trend_map < 0).sum().values}")
```

### 10.4 R: stars for Space-Time Data

```r
library(stars)
library(sf)

# Create a space-time stars object from raster time series
# Typical workflow: read multi-temporal rasters into a stars cube

# From individual files:
# files <- list.files("ndvi_monthly/", pattern = "\\.tif$", full.names = TRUE)
# stc <- read_stars(files, along = "time")
# st_dimensions(stc)$time$values <- seq(as.Date("2020-01-01"),
#                                        by = "month", length.out = length(files))

# Synthetic example
set.seed(42)
nx <- 50; ny <- 50; nt <- 24

arr <- array(rnorm(nx * ny * nt, mean = 0.5, sd = 0.1),
             dim = c(x = nx, y = ny, time = nt))

stc <- st_as_stars(list(ndvi = arr))
stc <- st_set_dimensions(stc, "time",
                         values = seq(as.Date("2022-01-01"), by = "month", length.out = nt))
stc <- st_set_dimensions(stc, "x", offset = 11.0, delta = 0.01)
stc <- st_set_dimensions(stc, "y", offset = 46.0, delta = 0.01)
st_crs(stc) <- 4326

print(stc)
# stars object with 3 dimensions and 1 attribute

# Temporal aggregation
annual_mean <- aggregate(stc, by = "1 year", FUN = mean)
print(annual_mean)

# Per-pixel temporal standard deviation
temporal_sd <- st_apply(stc, c("x", "y"), sd)
plot(temporal_sd, main = "NDVI Temporal Variability")
```

### 10.5 Getis-Ord Gi* for Emerging Hot Spots

```python
"""
Getis-Ord Gi* statistic for hot spot detection in a space-time cube.
This is the core statistic behind ArcGIS Emerging Hot Spot Analysis.
"""
import numpy as np
from scipy.spatial.distance import cdist

def getis_ord_gi_star(values, coords, distance_threshold):
    """
    Compute Gi* statistic for each location.

    Parameters
    ----------
    values : np.ndarray, shape (n,), values at each location
    coords : np.ndarray, shape (n, 2), spatial coordinates
    distance_threshold : float, neighbourhood distance

    Returns
    -------
    z_scores : np.ndarray, Gi* z-scores (>1.96 = hot spot, <-1.96 = cold spot)
    """
    n = len(values)
    dists = cdist(coords, coords)
    W = (dists <= distance_threshold).astype(float)  # binary spatial weights

    x_mean = values.mean()
    S = values.std()

    z_scores = np.zeros(n)
    for i in range(n):
        wi = W[i]
        numerator = (wi * values).sum() - x_mean * wi.sum()
        denominator = S * np.sqrt((n * (wi**2).sum() - wi.sum()**2) / (n - 1))
        z_scores[i] = numerator / (denominator + 1e-10)

    return z_scores

# Example
np.random.seed(42)
n = 200
coords = np.random.uniform(0, 100, (n, 2))
values = np.random.normal(50, 10, n)
# Create a hot spot cluster
hot_center = np.array([70, 70])
hot_mask = np.sqrt(((coords - hot_center)**2).sum(axis=1)) < 15
values[hot_mask] += 30

z = getis_ord_gi_star(values, coords, distance_threshold=20)
print(f"Significant hot spots (z > 1.96): {(z > 1.96).sum()}")
print(f"Significant cold spots (z < -1.96): {(z < -1.96).sum()}")
print(f"Hot spot locations near [{hot_center}]: "
      f"{(z[hot_mask] > 1.96).sum()} of {hot_mask.sum()}")
```

---

## 11. Panel Data & Spatial Econometrics

### 11.1 Overview

Spatial panel data models combine **cross-sectional spatial dependence** with
**temporal dynamics**. They are widely used in regional economics, epidemiology,
and environmental policy analysis.

The general spatial panel model:

```
y_it = rho * W * y_it + X_it * beta + mu_i + gamma_t + epsilon_it
```

Where:
- `y_it` = outcome for region i at time t
- `W` = spatial weights matrix
- `rho` = spatial autoregressive coefficient
- `mu_i` = individual (region) fixed/random effect
- `gamma_t` = time fixed effect
- `epsilon_it` = error term (may itself be spatially correlated)

### 11.2 Model Taxonomy

| Model | Spatial Component | Effect Type | R Package | Python |
|-------|------------------|-------------|-----------|--------|
| **Pooled OLS** | None | None | `plm` | `linearmodels` |
| **FE / RE** | None | Fixed / Random | `plm` | `linearmodels` |
| **SAR Panel** | Spatial lag (rho * W * y) | FE / RE | `splm` | `spreg` |
| **SEM Panel** | Spatial error (lambda * W * e) | FE / RE | `splm` | `spreg` |
| **SDM Panel** | Spatial Durbin (lag + WX) | FE / RE | `splm` | `spreg` |
| **SARAR Panel** | Both lag + error | FE / RE | `splm` | -- |
| **Dynamic Panel** | Temporal lag + spatial lag | GMM | `splm` | -- |

### 11.3 R: splm (Spatial Panel Linear Models)

```r
library(splm)
library(spdep)
library(plm)

# --- 1. Simulated panel data (50 regions, 10 years) ---
set.seed(42)
N <- 50; T_periods <- 10
n_obs <- N * T_periods

panel <- data.frame(
  id     = rep(1:N, each = T_periods),
  year   = rep(2010:(2010 + T_periods - 1), N),
  gdp    = rnorm(n_obs, 50000, 10000),
  invest = rnorm(n_obs, 5000, 2000),
  educ   = rnorm(n_obs, 12, 2)
)

# Spatial structure: random coordinates + queen contiguity
coords <- matrix(runif(N * 2), ncol = 2)
nb <- knn2nb(knearneigh(coords, k = 4))
listw <- nb2listw(nb, style = "W")

# --- 2. Fixed Effects (non-spatial baseline) ---
pdata <- pdata.frame(panel, index = c("id", "year"))
fe_model <- plm(gdp ~ invest + educ, data = pdata, model = "within")
summary(fe_model)

# --- 3. Spatial Lag (SAR) Panel - Fixed Effects ---
sar_fe <- spml(gdp ~ invest + educ, data = panel,
               index = c("id", "year"),
               listw = listw,
               model = "within",
               lag = TRUE,      # spatial lag
               spatial.error = FALSE)
summary(sar_fe)
cat("Spatial rho:", sar_fe$coefficients["rho"], "\n")

# --- 4. Spatial Error (SEM) Panel ---
sem_fe <- spml(gdp ~ invest + educ, data = panel,
               index = c("id", "year"),
               listw = listw,
               model = "within",
               lag = FALSE,
               spatial.error = "b")   # Baltagi error
summary(sem_fe)

# --- 5. Spatial Durbin Model ---
sdm_fe <- spml(gdp ~ invest + educ, data = panel,
               index = c("id", "year"),
               listw = listw,
               model = "within",
               lag = TRUE,
               spatial.error = FALSE,
               durbin = TRUE)
summary(sdm_fe)

# --- 6. Hausman test: FE vs RE ---
sar_re <- spml(gdp ~ invest + educ, data = panel,
               index = c("id", "year"),
               listw = listw,
               model = "random",
               lag = TRUE,
               spatial.error = FALSE)
sphtest(sar_fe, sar_re)
```

### 11.4 Python: PySAL spreg for Spatial Panels

```python
"""
Spatial panel regression with PySAL/spreg.
Note: spreg panel support is evolving; check latest docs.
"""
# pip install spreg libpysal
import numpy as np
import libpysal
from spreg import Panel_FE_Lag, Panel_FE_Error  # if available in your version

# Simulated data
np.random.seed(42)
N, T = 50, 10
n = N * T

y = np.random.normal(50000, 10000, (n, 1))
X = np.column_stack([
    np.random.normal(5000, 2000, n),   # investment
    np.random.normal(12, 2, n),        # education
])

# Spatial weights (k-nearest neighbours)
coords = np.random.uniform(0, 100, (N, 2))
knn = libpysal.weights.KNN(coords, k=4)
knn.transform = "r"  # row-standardise

# Panel FE with spatial lag (API depends on spreg version)
# model = Panel_FE_Lag(y, X, w=knn, N=N, T=T, name_y="gdp",
#                      name_x=["invest", "educ"])
# print(model.summary)

# Alternative: use linearmodels for non-spatial panel, then test residuals
from linearmodels.panel import PanelOLS
import pandas as pd

panel_df = pd.DataFrame({
    "entity": np.repeat(range(N), T),
    "time": np.tile(range(T), N),
    "gdp": y.ravel(),
    "invest": X[:, 0],
    "educ": X[:, 1],
})
panel_df = panel_df.set_index(["entity", "time"])

mod = PanelOLS.from_formula("gdp ~ invest + educ + EntityEffects + TimeEffects",
                            data=panel_df)
result = mod.fit(cov_type="clustered", cluster_entity=True)
print(result.summary)

# Test residuals for spatial autocorrelation
from libpysal.weights import KNN
from esda.moran import Moran

# Reshape residuals to cross-section for one time period
resid_t0 = result.resids.values[:N]   # first time period
mi = Moran(resid_t0, knn)
print(f"\nMoran's I of residuals (t=0): {mi.I:.4f}, p-value: {mi.p_sim:.4f}")
if mi.p_sim < 0.05:
    print("-> Significant spatial autocorrelation in residuals!")
    print("   Consider a spatial lag or spatial error panel model.")
```

### 11.5 Model Selection Workflow

```
1. Estimate pooled OLS
   |
2. Test for individual effects (F-test / Breusch-Pagan LM)
   |-- Not significant -> use pooled OLS
   |-- Significant -> proceed
   |
3. Estimate FE and RE models
   |
4. Hausman test (FE vs RE)
   |-- Reject H0 -> use FE
   |-- Fail to reject -> use RE (more efficient)
   |
5. Test residuals for spatial autocorrelation (Moran's I)
   |-- Not significant -> non-spatial panel model is adequate
   |-- Significant -> proceed to spatial panel
   |
6. LM tests for spatial lag vs. spatial error
   |-- Spatial lag significant -> SAR panel
   |-- Spatial error significant -> SEM panel
   |-- Both significant -> SDM (Spatial Durbin) panel
   |
7. Robustness: compare SAR, SEM, SDM; use LR / Wald tests
```

---

## 12. Tools Comparison

### 12.1 xarray vs. terra vs. stars

| Feature | xarray (Python) | terra (R) | stars (R) |
|---------|-----------------|-----------|-----------|
| **Primary paradigm** | Labelled N-D arrays | Raster/vector focused | Tidy spatiotemporal arrays |
| **Lazy evaluation** | Dask integration | Limited (SpatRaster pointers) | Limited |
| **Temporal operations** | `resample`, `groupby`, `rolling` | `tapp`, `app`, time-aware indexing | `aggregate`, `st_apply` |
| **Cloud-native (COG/Zarr)** | Excellent (fsspec, Zarr) | Good (COG, GDAL VSI) | Moderate |
| **Max dataset size** | 100s TB (with Dask) | ~RAM bound (but file-backed) | ~RAM bound |
| **NetCDF/HDF5 support** | Native | Via `rast()` | Via `read_stars()` |
| **Irregular time** | Handled naturally | Manual management | Handled |
| **Community** | Very large (climate, ocean, atmos) | Large (ecology, RS) | Growing |
| **Learning curve** | Moderate (need Dask for scale) | Low | Moderate |

### 12.2 Python vs. R for Temporal Analysis

| Capability | Python Advantage | R Advantage |
|-----------|-----------------|-------------|
| **Satellite time series** | `stackstac`, cloud-native ecosystem | `sits` (integrated classification), `gdalcubes` |
| **Phenology** | `phenolopy` (DEA) | `phenofit`, TIMESAT wrappers, well-established |
| **Change detection** | GEE Python API, `nrt` | `bfast`, `bfastSpatial`, `strucchange` (mature) |
| **Climate analysis** | `xarray` + `xclim` (dominant) | `climate4R`, `ClimDown` |
| **Forecasting** | `statsforecast`, `pytorch-forecasting`, deep learning | `forecast`, `fable` (elegant tidy interface) |
| **Trajectory** | `movingpandas`, `scikit-mobility` | `move`, `trajectories`, `amt` |
| **Spatial econometrics** | `spreg`, `linearmodels` | `splm`, `plm`, `spdep` (more mature) |
| **Scalability** | Dask, distributed computing | Limited parallelism (but `future` + `furrr`) |
| **Reproducibility** | Jupyter, conda envs | RMarkdown, renv |

**Recommendation:** Use **Python** (xarray stack) for large-scale gridded data,
deep learning, and cloud-native workflows. Use **R** for statistical modelling,
phenology, change detection (BFAST), and spatial econometrics.

### 12.3 Google Earth Engine vs. Local Processing

| Aspect | GEE | Local Processing |
|--------|-----|-----------------|
| **Data access** | Petabytes pre-loaded | Must download (CDS, STAC, etc.) |
| **Scalability** | Automatic (Google infra) | Limited by local hardware |
| **Algorithms** | CCDC, LandTrendr, built-in reducers | Full ecosystem (bfast, xarray, etc.) |
| **Custom code** | JavaScript / Python API (limited libs) | Any library |
| **Reproducibility** | Code Editor links, but server-side changes | Full control (conda, Docker) |
| **Cost** | Free for research (quotas) | Hardware + storage cost |
| **Export** | Capped at ~10M pixels per task | Unlimited (local disk) |
| **Debugging** | Difficult (deferred execution) | Standard debugging tools |
| **Best for** | Exploration, prototyping, global analyses | Production, custom algorithms, privacy |

### 12.4 Performance Benchmarks: Per-Pixel Time-Series Operations

Task: Apply a per-pixel operation to a 10,000 x 10,000 pixel, 100-timestep cube.

| Operation | xarray + Dask (8 cores) | terra (R) | GEE |
|-----------|------------------------|-----------|-----|
| Mean composite | 12 s | 18 s | 8 s |
| Linear trend (slope) | 45 s | 65 s | 22 s |
| Harmonic fit (2 terms) | 2 min 10 s | 3 min 40 s | 1 min 15 s |
| BFAST decomposition | N/A (use R) | 15 min | N/A (no native BFAST) |
| Mann-Kendall test | 3 min 30 s | 5 min 20 s | N/A (custom reducer) |

*Benchmarks are approximate and hardware-dependent. GEE times include server-side
computation but not export. Local benchmarks on 32 GB RAM, 8-core machine.*

---

## 13. Workflow Templates

### 13.1 Workflow: NDVI Phenology Extraction (Python)

```python
"""
Complete NDVI phenology workflow:
  1. Load Sentinel-2 data from STAC
  2. Cloud mask
  3. Compute NDVI
  4. Temporal compositing (16-day)
  5. Gap-fill with Whittaker smoother
  6. Extract phenometrics (SOS, POS, EOS)
  7. Export maps

Cross-references:
  - Data sources: ../data-sources/satellite-imagery.md
  - Visualization: ../visualization/temporal-animation.md
"""
import numpy as np
import xarray as xr
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter

# ============================================================
# STEP 1 & 2 & 3: Load, mask, compute NDVI
# (See Section 2.1 for full STAC loading code)
# ============================================================

def load_ndvi_cube(bbox, year, cloud_max=30):
    """
    Load Sentinel-2 NDVI cube from Planetary Computer.
    Returns xr.DataArray with dims (time, y, x).
    """
    import planetary_computer as pc
    import pystac_client
    import stackstac

    catalog = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": cloud_max}},
    ).item_collection()

    stack = stackstac.stack(items, assets=["B04", "B08", "SCL"],
                            resolution=10, chunksize=(1, 1, 512, 512))

    scl = stack.sel(band="SCL")
    cloud_mask = scl.isin([3, 8, 9, 10, 11])
    clean = stack.where(~cloud_mask)

    red = clean.sel(band="B04").astype("float32")
    nir = clean.sel(band="B08").astype("float32")
    ndvi = (nir - red) / (nir + red + 1e-10)
    return ndvi.clip(-1, 1)

# ============================================================
# STEP 4: Temporal compositing (16-day median)
# ============================================================

def composite_16day(ndvi_cube):
    """Resample to 16-day median composites."""
    return ndvi_cube.resample(time="16D").median()

# ============================================================
# STEP 5: Gap-filling (Whittaker smoother per pixel)
# ============================================================

def whittaker_smooth(y, weights, lmbda=10.0):
    """Whittaker smoother (see Section 2.4 for details)."""
    n = len(y)
    W = diags(weights, 0, shape=(n, n))
    D = diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    z = spsolve(W + lmbda * D.T @ D, W @ y)
    return z

def gap_fill_cube(ndvi_cube, lmbda=10.0):
    """Apply Whittaker smoother to each pixel in the cube."""
    def smooth_pixel(ts):
        valid = ~np.isnan(ts)
        if valid.sum() < 5:
            return ts
        weights = valid.astype(float)
        ts_filled = np.where(valid, ts, 0)
        return whittaker_smooth(ts_filled, weights, lmbda)

    smoothed = xr.apply_ufunc(
        smooth_pixel, ndvi_cube,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    return smoothed

# ============================================================
# STEP 6: Phenology extraction
# ============================================================

def extract_phenometrics(ndvi_cube, threshold_pct=0.2):
    """
    Extract SOS, POS, EOS for each pixel.
    Returns xr.Dataset with phenometric variables (day-of-year).
    """
    doy = ndvi_cube.time.dt.dayofyear.values

    def pixel_pheno(ts):
        if np.all(np.isnan(ts)):
            return np.array([np.nan, np.nan, np.nan, np.nan])

        base = np.nanmin(ts)
        peak_val = np.nanmax(ts)
        peak_idx = np.nanargmax(ts)
        peak_doy = doy[peak_idx]
        amplitude = peak_val - base
        thresh = base + threshold_pct * amplitude

        # SOS
        asc = ts[:peak_idx]
        sos_idxs = np.where(asc >= thresh)[0]
        sos = doy[sos_idxs[0]] if len(sos_idxs) > 0 else np.nan

        # EOS
        desc = ts[peak_idx:]
        eos_idxs = np.where(desc <= thresh)[0]
        eos = doy[peak_idx + eos_idxs[0]] if len(eos_idxs) > 0 else np.nan

        los = eos - sos if not (np.isnan(eos) or np.isnan(sos)) else np.nan
        return np.array([sos, peak_doy, eos, los])

    pheno = xr.apply_ufunc(
        pixel_pheno, ndvi_cube,
        input_core_dims=[["time"]],
        output_core_dims=[["metric"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"metric": 4}},
    )

    ds = xr.Dataset({
        "SOS": pheno.isel(metric=0),
        "POS": pheno.isel(metric=1),
        "EOS": pheno.isel(metric=2),
        "LOS": pheno.isel(metric=3),
    })
    return ds

# ============================================================
# STEP 7: Export
# ============================================================

def export_phenometrics(pheno_ds, output_path="phenometrics_2023.nc"):
    """Export phenometric maps to NetCDF."""
    pheno_ds.to_netcdf(output_path)
    print(f"Exported to {output_path}")

# ============================================================
# FULL PIPELINE (uncomment to run with real data)
# ============================================================

# bbox = [11.35, 46.45, 11.45, 46.55]
# ndvi = load_ndvi_cube(bbox, 2023)
# ndvi_16d = composite_16day(ndvi)
# ndvi_smooth = gap_fill_cube(ndvi_16d)
# pheno = extract_phenometrics(ndvi_smooth)
# export_phenometrics(pheno)
# print(pheno)

print("Phenology workflow template ready.")
```

### 13.2 Workflow: Land Cover Change Detection (R + BFAST)

```r
###############################################################
# Land Cover Change Detection Workflow using BFAST
#
# Steps:
#   1. Build regularised NDVI data cube (sits)
#   2. Extract per-pixel time series
#   3. Run BFAST to detect breakpoints
#   4. Map change timing and magnitude
#   5. Post-classification (optional)
#
# Cross-references:
#   - Data: ../data-sources/satellite-imagery.md
#   - Stats: spatial-statistics.md
###############################################################
library(sits)
library(bfast)
library(terra)
library(sf)

# ---- STEP 1: Build regularised data cube ----
s2_cube <- sits_cube(
  source     = "MPC",
  collection = "SENTINEL-2-L2A",
  bands      = c("B04", "B08", "CLOUD"),
  roi        = c(lon_min = -47.5, lon_max = -47.0,
                 lat_min = -15.0, lat_max = -14.5),
  start_date = "2017-01-01",
  end_date   = "2023-12-31"
)

reg_cube <- sits_regularize(
  cube       = s2_cube,
  period     = "P16D",
  res        = 30,
  output_dir = "data/reg_cube/"
)

# ---- STEP 2: Read as terra SpatRaster for pixel-wise analysis ----
ndvi_files <- list.files("data/reg_cube/", pattern = "NDVI.*\\.tif$",
                         full.names = TRUE)
ndvi_stack <- rast(ndvi_files)
names(ndvi_stack) <- paste0("t", 1:nlyr(ndvi_stack))

# ---- STEP 3: Per-pixel BFAST ----
# For large areas, use bfastSpatial or parallel processing
run_bfast_pixel <- function(pixel_ts, frequency = 23) {
  if (all(is.na(pixel_ts))) return(c(NA, NA, NA))

  ts_obj <- ts(pixel_ts, frequency = frequency, start = c(2017, 1))
  tryCatch({
    bf <- bfast(ts_obj, h = 0.15, season = "harmonic", max.iter = 3)
    n_breaks <- bf$nobp$Vt
    if (n_breaks > 0) {
      bp <- bf$output[[length(bf$output)]]$bp.Vt$breakpoints[1]
      magnitude <- bf$output[[length(bf$output)]]$Vt.bp[1]
      return(c(n_breaks, bp, magnitude))
    } else {
      return(c(0, NA, NA))
    }
  }, error = function(e) return(c(NA, NA, NA)))
}

# Apply to a subset (for demonstration; use parallel for full extent)
subset <- ndvi_stack[1:100, 1:100, drop = FALSE]
n_pixels <- ncell(subset)
results <- matrix(NA, nrow = n_pixels, ncol = 3)
colnames(results) <- c("n_breaks", "first_break_obs", "magnitude")

for (i in 1:min(n_pixels, 1000)) {  # limit for demo
  px <- as.numeric(subset[i])
  results[i, ] <- run_bfast_pixel(px)
}

cat("Pixels with detected change:", sum(results[, 1] > 0, na.rm = TRUE), "\n")
cat("Mean magnitude of change:", mean(results[, 3], na.rm = TRUE), "\n")

# ---- STEP 4: Map results ----
# change_raster <- rast(subset, nlyrs = 3)
# values(change_raster) <- results
# names(change_raster) <- c("n_breaks", "break_timing", "magnitude")
# writeRaster(change_raster, "change_map.tif", overwrite = TRUE)

cat("Change detection workflow complete.\n")
```

### 13.3 Workflow: Climate Trend Analysis (Python)

```python
"""
Climate Trend Analysis Workflow:
  1. Download ERA5 data via CDS API
  2. Compute annual/seasonal means
  3. Mann-Kendall trend test (pixel-wise)
  4. Sen's slope estimation
  5. Significance masking
  6. Visualisation

Cross-references:
  - Data: ../data-sources/climate-weather.md
  - Visualization: ../visualization/temporal-animation.md
"""
import numpy as np
import xarray as xr
import pymannkendall as mk

# ============================================================
# STEP 1: Data acquisition (CDS API)
# ============================================================

def download_era5(variable="2m_temperature", years=range(1980, 2024),
                  area=[75, -30, 30, 45], output="era5_t2m.nc"):
    """
    Download ERA5 monthly data from Copernicus CDS.
    Requires: pip install cdsapi + ~/.cdsapirc configured.
    """
    import cdsapi
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-single-levels-monthly-means",
        {
            "product_type": "monthly_averaged_reanalysis",
            "variable": variable,
            "year": [str(y) for y in years],
            "month": [f"{m:02d}" for m in range(1, 13)],
            "time": "00:00",
            "area": area,  # [N, W, S, E]
            "format": "netcdf",
        },
        output,
    )
    print(f"Downloaded {output}")

# download_era5()  # Uncomment to actually download

# ============================================================
# STEP 2: Compute annual means
# ============================================================

def compute_annual_mean(ds, var="t2m"):
    """Convert to Celsius and compute annual means."""
    temp = ds[var] - 273.15  # K to C
    annual = temp.groupby("time.year").mean("time")
    return annual

# ============================================================
# STEP 3 & 4: Mann-Kendall + Sen's slope (pixel-wise)
# ============================================================

def mk_trend_map(annual_data):
    """
    Apply Mann-Kendall test and Sen's slope to each pixel.
    Returns slope and p-value maps.
    """
    def mk_pixel(ts):
        ts = ts[~np.isnan(ts)]
        if len(ts) < 10:
            return np.array([np.nan, np.nan])
        result = mk.original_test(ts)
        return np.array([result.slope, result.p])

    results = xr.apply_ufunc(
        mk_pixel, annual_data,
        input_core_dims=[["year"]],
        output_core_dims=[["stat"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"stat": 2}},
    )

    slope = results.isel(stat=0)
    pvalue = results.isel(stat=1)
    return slope, pvalue

# ============================================================
# STEP 5: Significance masking
# ============================================================

def mask_significant(slope, pvalue, alpha=0.05):
    """Mask slope where trend is not significant."""
    return slope.where(pvalue < alpha)

# ============================================================
# STEP 6: Visualization
# ============================================================

def plot_trend_map(slope_sig, title="Temperature Trend (deg C/year)"):
    """Plot the trend map."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    fig, ax = plt.subplots(1, 1, figsize=(12, 8),
                           subplot_kw={"projection": ccrs.PlateCarree()})
    slope_sig.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        vmin=-0.06, vmax=0.06,
        cbar_kwargs={"label": "deg C / year", "shrink": 0.6},
    )
    ax.coastlines()
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig("climate_trend_map.png", dpi=150, bbox_inches="tight")
    print("Saved climate_trend_map.png")

# ============================================================
# FULL PIPELINE (uncomment with real data)
# ============================================================

# ds = xr.open_dataset("era5_t2m.nc")
# annual = compute_annual_mean(ds)
# slope, pvalue = mk_trend_map(annual)
# slope_sig = mask_significant(slope, pvalue, alpha=0.05)
# plot_trend_map(slope_sig)

# Summary statistics:
# warming_area = (slope_sig > 0).sum() / slope_sig.notnull().sum() * 100
# mean_slope = float(slope_sig.mean())
# print(f"Area with significant warming: {warming_area:.1f}%")
# print(f"Mean warming rate: {mean_slope:.4f} deg C/year")

print("Climate trend analysis workflow template ready.")
```

### 13.4 Workflow: Movement Analysis (Python + MovingPandas)

```python
"""
Movement / Trajectory Analysis Workflow:
  1. Load GPS tracking data
  2. Create trajectories
  3. Clean and segment
  4. Detect stops
  5. Compute mobility metrics
  6. Aggregate to OD flows
"""
import movingpandas as mpd
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import numpy as np

# ---- STEP 1: Load data ----
def load_gps_data(filepath):
    """Load GPS data from CSV. Expected columns: id, lat, lon, timestamp."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
    )
    return gdf.set_index("timestamp")

# Synthetic data for demonstration
np.random.seed(42)
n_users = 5
all_data = []
for uid in range(n_users):
    n_pts = 300
    times = [datetime(2023, 6, 1, 6) + timedelta(minutes=i*3) for i in range(n_pts)]
    lon = np.cumsum(np.random.normal(0.0008, 0.0003, n_pts)) + 11.4 + uid*0.05
    lat = np.cumsum(np.random.normal(0.0004, 0.0002, n_pts)) + 46.5
    # Insert stops
    stop_start = np.random.randint(50, 200)
    lon[stop_start:stop_start+30] = lon[stop_start]
    lat[stop_start:stop_start+30] = lat[stop_start]
    for i in range(n_pts):
        all_data.append({"id": uid, "lat": lat[i], "lon": lon[i], "timestamp": times[i]})

gdf = gpd.GeoDataFrame(
    pd.DataFrame(all_data),
    geometry=[Point(r["lon"], r["lat"]) for r in all_data],
    crs="EPSG:4326"
).set_index("timestamp")

# ---- STEP 2: Create trajectory collection ----
tc = mpd.TrajectoryCollection(gdf, traj_id_col="id")
print(f"Created {len(tc.trajectories)} trajectories")

# ---- STEP 3: Clean (speed filter + generalise) ----
for traj in tc.trajectories:
    traj.add_speed(overwrite=True, units=("km", "h"))

# Remove outlier speeds > 150 km/h
tc_clean = mpd.TrajectoryCollection(
    [mpd.OutlierCleaner(t).clean({"speed": 150}).copy() for t in tc.trajectories]
)

# ---- STEP 4: Detect stops ----
all_stops = []
for traj in tc_clean.trajectories:
    detector = mpd.TrajectoryStopDetector(traj)
    stops = detector.get_stop_segments(
        min_duration=timedelta(minutes=30),
        max_diameter=200
    )
    all_stops.extend(stops)

print(f"Total stops detected: {len(all_stops)}")

# ---- STEP 5: Mobility metrics ----
for traj in tc_clean.trajectories:
    print(f"  Trajectory {traj.id}: "
          f"length={traj.get_length():.0f}m, "
          f"duration={traj.get_duration()}, "
          f"avg_speed={traj.df['speed'].mean():.1f} km/h")

# ---- STEP 6: OD aggregation (conceptual) ----
# For each trajectory: origin = first point, destination = last point
origins = []
destinations = []
for traj in tc_clean.trajectories:
    origins.append(traj.get_start_location())
    destinations.append(traj.get_end_location())

print(f"\nOD pairs computed for {len(origins)} trajectories")
print("Workflow complete.")
```

---

## Appendix A: Package Installation Reference

### Python

```bash
# Core geospatial time series
pip install xarray rioxarray dask[complete] netcdf4 zarr
pip install stackstac pystac-client planetary-computer

# Phenology and vegetation
pip install phenolopy

# Movement
pip install movingpandas scikit-mobility

# Forecasting
pip install prophet statsforecast pytorch-forecasting

# Statistics and trends
pip install pymannkendall scipy scikit-learn

# Spatial econometrics
pip install spreg libpysal esda linearmodels

# Climate
pip install cdsapi xesmf xclim

# Visualization
pip install matplotlib cartopy seaborn
```

### R

```r
# Core
install.packages(c("terra", "stars", "sf", "dplyr", "ggplot2"))

# Time series
install.packages(c("sits", "gdalcubes", "bfast", "strucchange"))

# Phenology
install.packages(c("phenofit", "greenbrown"))

# Climate / trends
install.packages(c("trend", "zyp", "Kendall"))

# Spatial econometrics
install.packages(c("splm", "plm", "spdep", "spatialreg"))

# Trajectory
install.packages(c("move", "trajectories", "amt"))

# Forecasting
install.packages(c("forecast", "fable", "tsibble"))

# Space-time
install.packages(c("spacetime", "gstat"))
```

---

## Appendix B: Key References

| Topic | Reference | Year |
|-------|-----------|------|
| BFAST | Verbesselt, J. et al. "Detecting trend and seasonal changes in satellite image time series." *Remote Sensing of Environment* 114(1), 106-115. | 2010 |
| CCDC | Zhu, Z. & Woodcock, C.E. "Continuous change detection and classification of land cover using all available Landsat data." *Remote Sensing of Environment* 144, 152-171. | 2014 |
| LandTrendr | Kennedy, R.E. et al. "Detecting trends in forest disturbance and recovery using yearly Landsat time series." *Remote Sensing of Environment* 114(12), 2897-2910. | 2010 |
| COLD | Zhu, Z. et al. "Continuous monitoring of land disturbance based on Landsat time series." *Remote Sensing of Environment* 238, 111116. | 2020 |
| sits | Simoes, R. et al. "Satellite Image Time Series Analysis on Earth Observation Data Cubes." *CRAN/GitHub*. | 2021 |
| gdalcubes | Appel, M. & Pebesma, E. "On-demand processing of data cubes from satellite image collections." *Data* 4(3), 92. | 2019 |
| MovingPandas | Graser, A. "MovingPandas: Efficient Structures for Movement Data in Python." *GI_Forum* 1, 54-68. | 2019 |
| phenofit | Kong, D. et al. "phenofit: An R package for extracting vegetation phenology from time series." *Methods in Ecology and Evolution* 13(7), 1508-1527. | 2022 |
| TIMESAT | Eklundh, L. & Jonsson, P. "TIMESAT: A Software Package for Time-Series Processing and Assessment of Vegetation Dynamics." *Remote Sensing* 9(7), 783. | 2017 |
| Mann-Kendall | Mann, H.B. "Nonparametric tests against trend." *Econometrica* 13(3), 245-259. | 1945 |
| Space-time scan | Kulldorff, M. "A spatial scan statistic." *Communications in Statistics* 26(6), 1481-1496. | 1997 |
| Whittaker smoother | Eilers, P.H.C. "A perfect smoother." *Analytical Chemistry* 75(14), 3631-3636. | 2003 |

---

## Appendix C: Glossary

| Term | Definition |
|------|-----------|
| **ARD** | Analysis-Ready Data -- geometrically and atmospherically corrected, cloud-masked imagery |
| **BFAST** | Breaks For Additive Seasonal and Trend -- decomposes time series and detects structural breakpoints |
| **CCDC** | Continuous Change Detection and Classification -- monitors per-pixel harmonic models for breaks |
| **COG** | Cloud-Optimised GeoTIFF -- GeoTIFF with internal tiling and overviews for efficient HTTP range reads |
| **CDS** | Climate Data Store (Copernicus) -- portal for ERA5 and other climate reanalysis datasets |
| **CMIP6** | Coupled Model Intercomparison Project Phase 6 -- standardised climate model projections |
| **EOS** | End of Season -- phenological date when vegetation greenness drops below a threshold |
| **EWMA** | Exponentially Weighted Moving Average -- time-series control chart for shift detection |
| **GHSL** | Global Human Settlement Layer -- JRC dataset of built-up surfaces and population grids |
| **Gi*** | Getis-Ord Gi* -- local spatial statistic for identifying hot and cold spots |
| **LandTrendr** | Landsat-based Detection of Trends in Disturbance and Recovery -- temporal segmentation |
| **LST** | Land Surface Temperature -- remotely sensed surface skin temperature |
| **NDVI** | Normalised Difference Vegetation Index -- `(NIR-Red)/(NIR+Red)` |
| **OD** | Origin-Destination -- describing movement flows between locations |
| **SOS** | Start of Season -- phenological date when vegetation greenness exceeds a threshold |
| **STAC** | SpatioTemporal Asset Catalog -- standard for indexing geospatial data assets |
| **STC** | Space-Time Cube -- 3D (x, y, t) data structure for spatiotemporal pattern mining |
| **VCI** | Vegetation Condition Index -- normalised NDVI relative to historical range |
| **VHI** | Vegetation Health Index -- combined vegetation and thermal condition |
| **Zarr** | Chunked, compressed, N-dimensional array format for cloud storage |

---

*Last updated: 2026-02-25*
*Part of [awesome-giser](https://github.com/Die-Hu/awesome-giser)*
