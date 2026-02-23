# Spatial Statistics

> A guide to spatial statistical methods commonly used in GIS, from exploratory analysis to advanced spatial regression. Covers concepts, tools, and when to use each method.

> **Quick Picks**
> - **GUI for learning:** GeoDa (free, interactive, excellent tutorials)
> - **Python:** PySAL (esda + spreg + mgwr)
> - **R:** spdep + spatialreg + spatstat
> - **Spatial autocorrelation:** Moran's I (global) + LISA (local)
> - **Interpolation:** Ordinary Kriging via gstat (R) or pykrige (Python)

---

## Table of Contents

- [Exploratory Spatial Data Analysis (ESDA)](#exploratory-spatial-data-analysis-esda)
- [Point Pattern Analysis](#point-pattern-analysis)
- [Spatial Autocorrelation](#spatial-autocorrelation)
- [Geostatistics](#geostatistics)
- [Spatial Regression](#spatial-regression)
- [Tools Comparison](#tools-comparison)

---

## Exploratory Spatial Data Analysis (ESDA)

ESDA is the spatial equivalent of EDA (Exploratory Data Analysis). It helps you understand the spatial distribution of your data before applying formal models.

### Key Techniques

| Technique | What It Reveals | When to Use |
|---|---|---|
| Choropleth mapping | Spatial distribution of a variable | First look at any area-based data |
| Box maps | Identify spatial outliers | Find unusually high/low values in their spatial context |
| Cartogram | Magnitude relative to geography | Communicate population-weighted patterns |
| Spatial histogram | Frequency distribution by location | Compare distributions across sub-regions |
| Moran scatterplot | Spatial autocorrelation structure | Visualize the relationship between a value and its neighbors |

### Workflow

1. Map the raw variable (choropleth)
2. Examine the histogram and descriptive statistics
3. Construct spatial weights (see Spatial Autocorrelation section)
4. Create a Moran scatterplot to assess spatial dependence
5. Map the spatial lag to compare with the original variable
6. Identify outliers in both attribute and geographic space

### ESDA Example in PySAL

```python
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local

gdf = gpd.read_file("census_tracts.gpkg")
w = Queen.from_dataframe(gdf)
w.transform = 'r'  # Row-standardize weights

# Global Moran's I
mi = Moran(gdf["income"], w)
print(f"Moran's I: {mi.I:.4f}, p-value: {mi.p_sim:.4f}")

# Local Moran's I (LISA)
lisa = Moran_Local(gdf["income"], w)
gdf["lisa_cluster"] = lisa.q  # 1=HH, 2=LH, 3=LL, 4=HL
gdf["lisa_sig"] = lisa.p_sim < 0.05
```

---

## Point Pattern Analysis

Point pattern analysis examines whether a set of events (points) exhibits clustering, dispersion, or randomness, and if so, at what spatial scales.

### Methods

| Method | Question It Answers | Assumptions | Tool |
|---|---|---|---|
| **Quadrat analysis** | Is the density uniform across the study area? | Homogeneous study area | `spatstat` (R), `pointpats` (Python) |
| **Nearest Neighbor Index (NNI)** | Are points closer together (or farther apart) than expected under CSR? | Complete Spatial Randomness (CSR) as null | GeoDa, `pointpats`, `spatstat` |
| **Ripley's K-function** | At what distances does clustering (or dispersion) occur? | Isotropic, stationary process | `spatstat`, `pointpats` |
| **L-function** | Stabilized K-function (easier to interpret) | Same as K-function | `spatstat`, `pointpats` |
| **G-function** | Nearest neighbor distance distribution | CSR null | `spatstat`, `pointpats` |
| **Kernel Density Estimation (KDE)** | What is the intensity surface? | Bandwidth selection matters | QGIS, ArcGIS, `scipy`, `spatstat` |
| **DBSCAN** | What are the spatial clusters? | Minimum points and distance parameters | `scikit-learn`, `hdbscan` |

### Key Concepts

- **Complete Spatial Randomness (CSR):** The null hypothesis for most point pattern tests. Points are independent and uniformly distributed.
- **First-order effects:** Variation in intensity (density) across the study area.
- **Second-order effects:** Interactions between points (attraction = clustering, repulsion = dispersion).
- **Edge correction:** Points near the boundary of the study area can bias results. Use border, guard area, or toroidal corrections.

### Point Pattern Analysis Example in R

```r
library(spatstat)

# Create a point pattern from coordinates
pp <- ppp(x, y, window = owin(c(xmin, xmax), c(ymin, ymax)))

# Test for CSR using Clark-Evans nearest neighbor index
clarkevans.test(pp, correction = "guard")

# Ripley's K-function with Monte Carlo envelope
k_env <- envelope(pp, Kest, nsim = 99)
plot(k_env, main = "Ripley's K-function")

# Kernel Density Estimation
dens <- density(pp, sigma = bw.diggle)
plot(dens, main = "Kernel Density Estimate")
```

---

## Spatial Autocorrelation

Spatial autocorrelation measures the degree to which nearby observations have similar (or dissimilar) values. It is one of the most fundamental concepts in spatial statistics.

### Global Measures

| Measure | What It Tests | Range | Interpretation |
|---|---|---|---|
| **Moran's I** | Overall spatial clustering of values | -1 to +1 | +1 = perfect clustering, 0 = random, -1 = perfect dispersion |
| **Geary's C** | Similar to Moran's I, more sensitive to local variation | 0 to 2+ | <1 = positive autocorrelation, 1 = random, >1 = negative |
| **Getis-Ord General G** | Clustering of high or low values | depends on context | High G = hot spot clustering, low G = cold spot clustering |

### Local Measures (LISA)

| Measure | What It Identifies | Output |
|---|---|---|
| **Local Moran's I** | Local clusters and spatial outliers | HH (hot spot), LL (cold spot), HL (high outlier), LH (low outlier) |
| **Getis-Ord Gi\*** | Statistically significant hot and cold spots | Z-score and p-value per feature |
| **Local Geary** | Local spatial associations (multivariate capable) | Cluster classification |

### Spatial Weights

The results of spatial autocorrelation tests depend on how you define "neighbors." Common spatial weights:

| Weight Type | Definition | Best For |
|---|---|---|
| Queen contiguity | Shares a boundary or vertex | Regular polygons (census tracts, grid cells) |
| Rook contiguity | Shares a boundary (not vertex) | Regular grids |
| K-Nearest Neighbors (KNN) | K closest features by centroid distance | Irregular polygons, points |
| Distance band | All features within a distance threshold | Points, when a meaningful distance exists |
| Inverse distance | Weight decreases with distance | Continuous spatial processes |
| Kernel | Smooth distance decay function | Regression, smoothing |

### Spatial Weights Comparison

- **Queen contiguity**: Most common for census-type polygons. A polygon's neighbors include all units sharing a boundary or corner vertex. Typically produces 5-8 neighbors per unit.
- **Rook contiguity**: More restrictive -- only shared edges count. Produces fewer neighbors (4-6 typical). Best for regular grids.
- **KNN (k=5 or k=8)**: Fixed number of neighbors based on centroid distance. Works for irregular geometries and point data. Ensures every observation has the same number of neighbors.
- **Distance band**: All features within a threshold distance. Choose the threshold to ensure every unit has at least one neighbor (use `libpysal.weights.min_threshold_distance()`).

---

## Geostatistics

Geostatistics models spatial continuity using variograms and performs spatial prediction (interpolation) via kriging.

### Key Concepts

| Concept | Description |
|---|---|
| **Regionalized variable** | A variable that varies continuously across space (e.g., soil pH, temperature) |
| **Variogram (semivariogram)** | A function describing how spatial similarity decreases with distance |
| **Nugget** | Variogram value at distance 0 (measurement error + micro-scale variation) |
| **Sill** | The variogram plateau (total variance) |
| **Range** | The distance at which the variogram reaches the sill (spatial correlation distance) |

### Kriging Methods

| Method | When to Use | Assumptions |
|---|---|---|
| **Ordinary Kriging** | Interpolate a single variable, unknown mean | Second-order stationarity |
| **Simple Kriging** | Known and constant mean | Strict stationarity |
| **Universal Kriging** | Variable has a spatial trend | Trend must be modeled (polynomial, external drift) |
| **Indicator Kriging** | Estimate probability of exceeding a threshold | Binary transformation of data |
| **Co-Kriging** | Multiple correlated variables | Cross-variogram model |
| **Regression Kriging** | Combine regression with kriging of residuals | Predictors available at all locations |

### Variogram Models

| Model | Shape | Use When |
|---|---|---|
| Spherical | S-shaped curve | Most common; general purpose |
| Exponential | Asymptotic approach to sill | Processes with gradual correlation decay |
| Gaussian | Smooth near origin | Very smooth spatial processes |
| Matern | Flexible smoothness parameter | When you want to control smoothness |

### Kriging Example in R

```r
library(gstat)
library(sf)

# Compute empirical variogram
v <- variogram(zinc ~ 1, data = meuse_sf)

# Fit a variogram model
v_fit <- fit.variogram(v, vgm(psill = 1, model = "Sph", range = 900, nugget = 0.1))
plot(v, v_fit, main = "Fitted Variogram")

# Ordinary Kriging
kriged <- krige(zinc ~ 1, meuse_sf, newdata = grid_sf, model = v_fit)
plot(kriged["var1.pred"], main = "Kriging Prediction")
```

### Variogram Anatomy

```
semivariance
    |
    |          ........ sill (total variance)
    |        .
    |      .   <-- range (correlation distance)
    |    .
    |  .
    |.
    +-- nugget (measurement error + micro-scale variation)
    +---------------------------------- distance (lag)
```

---

## Spatial Regression

When your data exhibits spatial autocorrelation, standard OLS regression violates the independence assumption. Spatial regression models explicitly account for spatial structure.

### When to Use Spatial Regression

1. Run OLS regression
2. Test residuals for spatial autocorrelation (Moran's I on residuals)
3. If significant spatial autocorrelation exists, choose a spatial model
4. Use Lagrange Multiplier (LM) diagnostics to choose between lag and error models

### Model Types

| Model | Abbreviation | What It Models | When to Use |
|---|---|---|---|
| **Spatial Lag (SAR)** | SLM | Spatial dependence in the dependent variable | Outcome in one area affects neighbors (diffusion) |
| **Spatial Error (SEM)** | SEM | Spatial autocorrelation in the error term | Omitted variables have spatial structure |
| **Spatial Durbin (SDM)** | SDM | Lag of both dependent and independent variables | Both spillover and local effects matter |
| **GWR** | GWR | Spatially varying coefficients | Relationships vary across space |
| **MGWR** | MGWR | Multi-scale GWR (different bandwidth per variable) | Different processes operate at different scales |

### Diagnostic Workflow

```
OLS Regression
    |
    +--> Test residuals (Moran's I)
           |
           +--> NOT significant --> Use OLS (no spatial model needed)
           |
           +--> Significant --> Run LM diagnostics
                                   |
                                   +--> LM-Lag significant, LM-Error not --> Spatial Lag (SLM)
                                   |
                                   +--> LM-Error significant, LM-Lag not --> Spatial Error (SEM)
                                   |
                                   +--> Both significant --> Compare Robust LM tests
                                   |                           |
                                   |                           +--> Robust LM-Lag significant --> SLM
                                   |                           +--> Robust LM-Error significant --> SEM
                                   |                           +--> Both significant --> SDM
                                   |
                                   +--> Suspect varying relationships --> GWR / MGWR
```

### Spatial Regression Example in PySAL

```python
from spreg import ML_Lag, ML_Error, OLS
from libpysal.weights import Queen
import numpy as np

w = Queen.from_dataframe(gdf)
w.transform = 'r'

y = gdf[["house_price"]].values
X = gdf[["sqft", "distance_cbd", "green_pct"]].values

# Step 1: Run OLS and check residual autocorrelation
ols = OLS(y, X, w=w, spat_diag=True, name_y="price", name_x=["sqft","dist","green"])
# Check LM-Lag and LM-Error p-values in ols.summary

# Step 2: Based on diagnostics, fit spatial lag or error model
lag = ML_Lag(y, X, w=w, name_y="price", name_x=["sqft","dist","green"])
print(lag.summary)
```

---

## Tools Comparison

Side-by-side comparison of the three main tools for spatial statistics.

| Capability | Python (PySAL) | R (spdep / spatialreg) | GeoDa (GUI) |
|---|---|---|---|
| **Spatial weights** | `libpysal.weights` | `spdep::nb2listw()` | Weights Manager GUI |
| **Global Moran's I** | `esda.Moran()` | `spdep::moran.test()` | Space > Univariate Moran's I |
| **LISA** | `esda.Moran_Local()` | `spdep::localmoran()` | Space > Univariate Local Moran's I |
| **Getis-Ord Gi\*** | `esda.G_Local()` | `spdep::localG()` | Space > Getis-Ord Statistics |
| **Spatial Lag Model** | `spreg.ML_Lag()` | `spatialreg::lagsarlm()` | Regression > Spatial Lag |
| **Spatial Error Model** | `spreg.ML_Error()` | `spatialreg::errorsarlm()` | Regression > Spatial Error |
| **GWR** | `mgwr.GWR()` | `GWmodel::gwr.basic()` | Not built-in (use GWR4 software) |
| **Kriging** | `pykrige` / `gstools` | `gstat::krige()` | Not built-in |
| **Point patterns** | `pointpats` | `spatstat` | Limited |
| **Variograms** | `gstools.Variogram()` | `gstat::variogram()` | Not built-in |
| **Learning curve** | Moderate (need Python skills) | Moderate (need R skills) | Low (GUI-based) |
| **Reproducibility** | High (scripts) | High (scripts) | Low (manual steps) |
| **Scalability** | Good (Dask integration) | Moderate | Limited (desktop) |

### Performance Notes

For large datasets (100K+ features):
- **PySAL**: Supports sparse weights matrices; scales well with KNN weights. Dask integration planned.
- **R spdep**: Memory-intensive for large weight matrices; consider `spdep::nb2listw(style="W")` with sparse representation.
- **GeoDa**: Limited to available RAM; works well up to ~50K polygons on modern machines.
- **Alternative for big data**: Use DuckDB Spatial for aggregation, then PySAL/R for statistical modeling on summarized data.

### Recommendation

- **Beginners or exploratory work:** Start with GeoDa for its intuitive GUI, then graduate to PySAL or R.
- **Python workflows:** PySAL integrates seamlessly with GeoPandas and Scikit-learn.
- **R workflows:** spdep/spatialreg integrate with the tidyverse and ggplot2.
- **Publication:** R or Python for reproducible scripts; GeoDa for quick exploration.

---

[Back to Data Analysis](README.md) · [Back to Main README](../README.md)
