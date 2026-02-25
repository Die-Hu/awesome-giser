# Spatial Statistics

> The definitive reference for spatial statistical methods in GIS -- from exploratory analysis and point pattern analysis through geostatistics, spatial regression, Bayesian modeling, and spatial econometrics. Every section includes theory, interpretation guidance, and production-ready code in Python and R.

> **Quick Picks**
> - **GUI for learning:** GeoDa (free, interactive, excellent tutorials)
> - **Python ecosystem:** PySAL (esda + spreg + mgwr + pointpats + spaghetti)
> - **R ecosystem:** spdep + spatialreg + spatstat + gstat + R-INLA + GWmodel
> - **Spatial autocorrelation:** Moran's I (global) + LISA (local) + Getis-Ord Gi* (hot spots)
> - **Interpolation:** Ordinary Kriging via gstat (R) or pykrige / gstools (Python)
> - **Bayesian spatial:** R-INLA (fast approximate) or brms/Stan (full MCMC)

> **Cross-references:** [Python Stack](python-stack.md) | [R Stack](r-stack.md) | [Thematic Maps](../visualization/thematic-maps.md) | [Scientific Visualization](../visualization/scientific-visualization.md) | [ML for GIS](ml-gis.md)

---

## Table of Contents

- [Exploratory Spatial Data Analysis (ESDA)](#exploratory-spatial-data-analysis-esda)
- [Point Pattern Analysis](#point-pattern-analysis)
- [Spatial Autocorrelation](#spatial-autocorrelation)
- [Spatial Weights](#spatial-weights)
- [Geostatistics](#geostatistics)
- [Spatial Regression](#spatial-regression)
- [GWR and MGWR](#gwr-and-mgwr)
- [Bayesian Spatial Models](#bayesian-spatial-models)
- [Spatial Clustering and Regionalization](#spatial-clustering-and-regionalization)
- [Hot Spot Analysis](#hot-spot-analysis)
- [Spatial Econometrics](#spatial-econometrics)
- [Tool Comparison](#tool-comparison)

---

## Exploratory Spatial Data Analysis (ESDA)

ESDA is the spatial equivalent of EDA (Exploratory Data Analysis). It reveals spatial patterns, outliers, and structure in your data before you commit to formal modeling. Skipping ESDA is the single most common mistake in spatial analysis -- you cannot choose appropriate models without understanding your data's spatial structure first.

### Key Techniques

| Technique | What It Reveals | When to Use | Tool |
|---|---|---|---|
| **Choropleth mapping** | Spatial distribution of a variable | First look at any area-based data | GeoDa, QGIS, GeoPandas |
| **Box maps** | Spatial outliers (values beyond 1.5 IQR in geographic context) | Find unusually high/low values in their spatial context | GeoDa |
| **Standard deviation maps** | Values above/below the mean by sigma bands | Compare deviation from global mean across space | GeoDa, custom code |
| **Percentile maps** | Distribution-free spatial ranking | Non-normal data, ordinal comparisons | GeoDa, custom code |
| **Cartogram** | Magnitude relative to geography | Communicate population-weighted patterns | QGIS (cartogram3 plugin), `cartopy` |
| **Spatial histogram** | Frequency distribution by location | Compare distributions across sub-regions | Custom code |
| **Moran scatterplot** | Spatial autocorrelation structure | Visualize relationship between a value and its spatial lag | GeoDa, PySAL, spdep |
| **LISA cluster map** | Local clusters and outliers | Identify statistically significant local patterns | GeoDa, PySAL, spdep |
| **Conditional maps** | Bivariate spatial distributions | Explore two variables simultaneously across space | GeoDa |

### Choropleth Classification Schemes

The choice of classification scheme dramatically changes how your map tells its story. Different schemes can make the same data look clustered or dispersed.

| Scheme | Logic | Best For | Pitfall |
|---|---|---|---|
| **Equal interval** | Equal-width bins across the data range | Uniformly distributed data | Misleading when data is skewed |
| **Quantiles** | Equal number of observations per bin | General purpose, comparable maps | Can split natural groups, hide gaps |
| **Natural breaks (Jenks)** | Minimize within-class variance | Naturally grouped data | Not comparable across datasets |
| **Standard deviation** | Bins based on sigma from mean | Identifying deviations from the mean | Assumes normality |
| **Head/tail breaks** | Iterative mean splitting for heavy-tailed data | Power-law distributed data (city sizes, income) | Not widely implemented |
| **Custom breaks** | Domain-specific thresholds | Regulatory or policy thresholds | Requires domain knowledge |

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify

gdf = gpd.read_file("census_tracts.gpkg")

# Compare classification schemes side-by-side
schemes = {
    'Equal Interval': mapclassify.EqualInterval(gdf['income'], k=5),
    'Quantiles': mapclassify.Quantiles(gdf['income'], k=5),
    'Natural Breaks': mapclassify.NaturalBreaks(gdf['income'], k=5),
    'Std Dev': mapclassify.StdMean(gdf['income']),
    'Head/Tail Breaks': mapclassify.HeadTailBreaks(gdf['income']),
}

fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for ax, (name, classifier) in zip(axes, schemes.items()):
    gdf['class'] = classifier.yb
    gdf.plot(column='class', ax=ax, legend=True, cmap='YlOrRd')
    ax.set_title(name)
    ax.axis('off')
plt.tight_layout()
plt.savefig("classification_comparison.png", dpi=150, bbox_inches='tight')
```

### Moran Scatterplot Deep Dive

The Moran scatterplot is one of the most informative single graphics in spatial analysis. It plots the standardized value of each observation (z) against its spatially lagged value (Wz). The four quadrants correspond to the four types of local spatial association:

```
        Spatial Lag (Wz)
              |
     LH       |       HH
  (Low value,  | (High value,
   High lag)   |  High lag)
              |
  ------------+------------ Value (z)
              |
     LL       |       HL
  (Low value,  | (High value,
   Low lag)    |  Low lag)
              |
```

- **HH (High-High):** Spatial clusters of high values (hot spots)
- **LL (Low-Low):** Spatial clusters of low values (cold spots)
- **HL (High-Low):** Spatial outliers -- high values surrounded by low values
- **LH (Low-High):** Spatial outliers -- low values surrounded by high values
- **Slope of the regression line** through the scatterplot equals Moran's I

### ESDA Workflow

```
1. Map the raw variable (choropleth)
   |
2. Try multiple classification schemes -- do patterns change?
   |
3. Examine the histogram and descriptive statistics
   |   - Skewness? Heavy tails? Multimodal?
   |   - Consider transformation (log, Box-Cox)
   |
4. Construct spatial weights (see Spatial Weights section)
   |   - Start with Queen contiguity for polygons
   |   - Use KNN for points or irregular geometries
   |
5. Compute Global Moran's I
   |   - Is there significant spatial autocorrelation?
   |
6. Create a Moran scatterplot
   |   - What is the dominant pattern (clustering vs. dispersion)?
   |   - Are there spatial outliers?
   |
7. Compute LISA (Local Moran's I)
   |   - Map significant clusters (HH, LL) and outliers (HL, LH)
   |   - Use significance filter (p < 0.05 with Bonferroni or FDR correction)
   |
8. Cross-validate with Getis-Ord Gi* for hot/cold spot confirmation
   |
9. Investigate outliers and clusters -- what processes generate them?
```

### Complete ESDA Example in PySAL

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from libpysal.weights import Queen
from esda.moran import Moran, Moran_Local
from splot.esda import (
    moran_scatterplot,
    lisa_cluster,
    plot_moran_simulation,
    plot_local_autocorrelation,
)

# --- Load and prepare data ---
gdf = gpd.read_file("census_tracts.gpkg")
y = gdf["median_income"]

# --- Spatial weights ---
w = Queen.from_dataframe(gdf)
w.transform = 'r'  # Row-standardize

# --- Global Moran's I ---
mi = Moran(y, w, permutations=999)
print(f"Moran's I:     {mi.I:.4f}")
print(f"Expected I:    {mi.EI:.4f}")
print(f"p-value (sim): {mi.p_sim:.4f}")
print(f"z-score:       {mi.z_sim:.4f}")

# Interpretation guide:
# I > EI and p < 0.05 --> significant positive spatial autocorrelation (clustering)
# I < EI and p < 0.05 --> significant negative spatial autocorrelation (dispersion)
# p >= 0.05 --> cannot reject CSR (spatial randomness)

# --- Moran scatterplot ---
fig, ax = moran_scatterplot(mi, aspect_equal=False)
ax.set_title(f"Moran Scatterplot (I = {mi.I:.3f}, p = {mi.p_sim:.3f})")
plt.tight_layout()
plt.savefig("moran_scatterplot.png", dpi=150)

# --- Reference distribution from permutations ---
fig, ax = plot_moran_simulation(mi)
plt.savefig("moran_permutation_dist.png", dpi=150)

# --- Local Moran's I (LISA) ---
lisa = Moran_Local(y, w, permutations=999)

# LISA attributes:
#   lisa.Is      - local Moran's I for each observation
#   lisa.q       - quadrant (1=HH, 2=LH, 3=LL, 4=HL)
#   lisa.p_sim   - pseudo p-value from permutations
#   lisa.z_sim   - z-score

gdf["lisa_I"] = lisa.Is
gdf["lisa_q"] = lisa.q
gdf["lisa_p"] = lisa.p_sim

# Apply FDR correction for multiple testing
from statsmodels.stats.multitest import multipletests
reject, pvals_corrected, _, _ = multipletests(lisa.p_sim, method='fdr_bh')
gdf["lisa_sig_fdr"] = reject

# --- LISA cluster map ---
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
lisa_cluster(lisa, gdf, ax=ax, p=0.05)
ax.set_title("LISA Cluster Map (p < 0.05)")
plt.savefig("lisa_clusters.png", dpi=150, bbox_inches='tight')

# --- Summary statistics for LISA ---
sig_mask = gdf["lisa_sig_fdr"]
cluster_labels = {1: "HH (Hot Spot)", 2: "LH (Low-High Outlier)",
                  3: "LL (Cold Spot)", 4: "HL (High-Low Outlier)"}
print("\nSignificant LISA clusters (FDR-corrected):")
for q_val, label in cluster_labels.items():
    count = ((gdf["lisa_q"] == q_val) & sig_mask).sum()
    print(f"  {label}: {count} features")
```

### ESDA in GeoDa

GeoDa provides the most intuitive interface for ESDA. Key operations:

1. **File > Open:** Load shapefile, GeoPackage, or GeoJSON
2. **Map > Quantile Map / Natural Breaks Map:** Quick choropleth exploration
3. **Space > Univariate Moran's I:** Compute global Moran's I with permutation inference
4. **Space > Univariate Local Moran's I:** Generate LISA cluster map and significance map
5. **Tools > Weights Manager:** Create and manage spatial weight matrices

GeoDa's linked views are particularly powerful: brushing observations in one view highlights them in all views simultaneously (scatter plot, map, histogram, box plot).

---

## Point Pattern Analysis

Point pattern analysis examines whether a set of events (points) exhibits clustering, dispersion, or randomness, and if so, at what spatial scales. It is fundamental to epidemiology, ecology, crime analysis, retail site selection, and natural hazard assessment.

### Fundamental Concepts

- **Complete Spatial Randomness (CSR):** The null hypothesis for most point pattern tests. Under CSR, points follow a homogeneous Poisson process -- they are independent and uniformly distributed across the study area. The number of points in any sub-region follows a Poisson distribution.
- **First-order effects (intensity):** Variation in the density/intensity of points across the study area. Driven by underlying environmental covariates (e.g., crime clusters near bars, trees cluster on fertile soil).
- **Second-order effects (interaction):** Interactions between points themselves (attraction = clustering, repulsion = regularity/inhibition). Independent of location.
- **Stationarity:** The process generating points does not change across the study area.
- **Isotropy:** The process does not depend on direction.
- **Edge correction:** Points near the boundary can bias results because their neighbors may lie outside the study window. Methods: border correction, Ripley's isotropic correction, toroidal (wrap-around), guard area.

### First-Order Analysis

#### Quadrat Analysis

Divides the study area into equal-sized cells and counts points per cell. Tests whether counts follow a Poisson distribution (CSR).

```python
# Python - pointpats
from pointpats import QStatistic, PointPattern
import numpy as np

# Create point pattern from coordinates
pp = PointPattern(np.column_stack([x_coords, y_coords]))

# Quadrat test with 5x5 grid
q = QStatistic(pp, shape="rectangle", nx=5, ny=5)
print(f"Chi-squared: {q.chi2:.2f}")
print(f"p-value:     {q.chi2_pvalue:.4f}")
# p < 0.05 --> reject CSR, points are not randomly distributed
```

```r
# R - spatstat
library(spatstat)

# Create point pattern object
pp <- ppp(x, y, window = owin(c(xmin, xmax), c(ymin, ymax)))

# Quadrat test
qt <- quadrat.test(pp, nx = 5, ny = 5)
print(qt)
plot(qt)  # Shows observed vs expected counts per cell
```

**Limitations:** Results depend on quadrat size and placement. Too large and you miss local patterns; too small and most cells are empty.

#### Kernel Density Estimation (KDE)

KDE produces a smooth intensity surface from point data. It is the most common method for visualizing point pattern intensity.

```
Intensity at location s = (1/n) * sum_i K((s - s_i) / h) / h^2
```

where K is the kernel function, h is the bandwidth, and s_i are the point locations.

**Bandwidth selection is the most critical decision in KDE.** Too small creates noisy spikes; too large over-smooths and hides real patterns.

| Bandwidth Method | Logic | Implementation |
|---|---|---|
| **Rule of thumb** | Based on data spread (Silverman's rule) | `scipy.stats.gaussian_kde` (default) |
| **Diggle's** | Minimizes MSE for intensity estimation | `spatstat::bw.diggle()` |
| **Scott's** | Optimal for Gaussian data | `scipy.stats.gaussian_kde` |
| **Likelihood cross-validation** | Maximizes out-of-sample likelihood | `spatstat::bw.ppl()` |
| **Least-squares cross-validation** | Minimizes integrated squared error | `spatstat::bw.CvL()` |
| **Adaptive bandwidth** | Bandwidth varies with local density | `spatstat::adaptive.density()` |

```python
# Python - KDE with multiple bandwidth methods
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity

coords = np.column_stack([x_coords, y_coords])

# scipy gaussian_kde (Scott's rule by default)
kde_scipy = gaussian_kde(coords.T)

# sklearn with explicit bandwidth
kde_sk = KernelDensity(bandwidth=500, kernel='gaussian', metric='haversine')
kde_sk.fit(np.radians(coords))  # For geographic coordinates

# Evaluate on a grid
x_grid = np.linspace(coords[:, 0].min(), coords[:, 0].max(), 200)
y_grid = np.linspace(coords[:, 1].min(), coords[:, 1].max(), 200)
xx, yy = np.meshgrid(x_grid, y_grid)
grid_points = np.column_stack([xx.ravel(), yy.ravel()])

density = kde_scipy(grid_points.T).reshape(xx.shape)
```

```r
# R - spatstat KDE with bandwidth selection
library(spatstat)

pp <- ppp(x, y, window = owin(c(xmin, xmax), c(ymin, ymax)))

# Compare bandwidth selectors
bw_diggle <- bw.diggle(pp)
bw_ppl <- bw.ppl(pp)
bw_scott <- bw.scott(pp)

cat("Diggle bandwidth:", bw_diggle, "\n")
cat("Likelihood CV:   ", bw_ppl, "\n")
cat("Scott bandwidth: ", bw_scott[1], "\n")

# Compute KDE with different bandwidths
par(mfrow = c(1, 3))
plot(density(pp, sigma = bw_diggle), main = paste("Diggle (bw =", round(bw_diggle), ")"))
plot(density(pp, sigma = bw_ppl), main = paste("Likelihood CV (bw =", round(bw_ppl), ")"))
plot(density(pp, sigma = bw_scott), main = paste("Scott (bw =", round(bw_scott[1]), ")"))

# Adaptive KDE (variable bandwidth)
ada <- adaptive.density(pp, method = "kernel")
plot(ada, main = "Adaptive KDE")
```

### Second-Order Analysis

#### Nearest Neighbor Index (NNI)

The Clark-Evans nearest neighbor index compares the observed mean nearest neighbor distance to the expected distance under CSR.

```
NNI = d_observed / d_expected
```

- NNI < 1: Clustered (points closer than expected)
- NNI = 1: Random (CSR)
- NNI > 1: Dispersed/regular (points farther apart than expected)

```python
# Python - pointpats
from pointpats import PointPattern

pp = PointPattern(np.column_stack([x_coords, y_coords]))
print(f"Mean NN distance: {pp.mean_nnd:.2f}")
print(f"Min NN distance:  {pp.min_nnd:.2f}")

# Clark-Evans test
from pointpats.centrography import nni
nni_val = nni(np.column_stack([x_coords, y_coords]))
print(f"NNI: {nni_val:.4f}")
```

```r
# R - spatstat
library(spatstat)

pp <- ppp(x, y, window = owin(c(xmin, xmax), c(ymin, ymax)))

# Clark-Evans test with guard-area edge correction
ce <- clarkevans.test(pp, correction = "guard", alternative = "clustered")
print(ce)
# R < 1 and p < 0.05 --> significant clustering
```

#### Ripley's K, L, and G Functions

These are the workhorses of second-order point pattern analysis. They characterize spatial dependence at multiple distance scales simultaneously.

**K-function:** K(d) estimates the expected number of additional points within distance d of a typical point, divided by the overall intensity.

```
Under CSR: K(d) = pi * d^2  (area of a circle)
K(d) > pi * d^2  --> clustering at distance d
K(d) < pi * d^2  --> regularity at distance d
```

**L-function:** A variance-stabilized transformation of K that is easier to interpret:

```
L(d) = sqrt(K(d) / pi)
Under CSR: L(d) = d, so L(d) - d = 0
L(d) - d > 0 --> clustering
L(d) - d < 0 --> regularity
```

**G-function (nearest neighbor distance distribution):** G(d) is the proportion of nearest-neighbor distances that are less than or equal to d.

```
Under CSR: G(d) = 1 - exp(-lambda * pi * d^2)
G(d) above CSR --> clustering (points have close neighbors)
G(d) below CSR --> regularity (points avoid each other)
```

**F-function (empty space function):** F(d) is the distribution of distances from an arbitrary point to the nearest event. The complement of G.

```r
# R - complete point pattern analysis with spatstat
library(spatstat)

pp <- ppp(x, y, window = owin(c(xmin, xmax), c(ymin, ymax)))

# Summary statistics
summary(pp)

# Ripley's K-function with Monte Carlo envelope (99 simulations)
k_env <- envelope(pp, Kest, nsim = 99, correction = "isotropic")
plot(k_env, main = "Ripley's K-function",
     xlab = "Distance (d)", ylab = "K(d)")

# L-function (variance-stabilized K)
l_env <- envelope(pp, Lest, nsim = 99, correction = "isotropic")
plot(l_env, . - r ~ r, main = "L(d) - d",
     xlab = "Distance (d)", ylab = "L(d) - d")
# Values above the envelope --> significant clustering
# Values below the envelope --> significant regularity

# G-function (nearest-neighbor distance distribution)
g_env <- envelope(pp, Gest, nsim = 99, correction = "km")
plot(g_env, main = "G-function (NN distance CDF)")

# F-function (empty space function)
f_env <- envelope(pp, Fest, nsim = 99, correction = "km")
plot(f_env, main = "F-function (empty space CDF)")

# J-function (J = (1-G)/(1-F); J < 1 = clustering, J > 1 = regularity)
j_env <- envelope(pp, Jest, nsim = 99)
plot(j_env, main = "J-function")
```

```python
# Python - pointpats K/L/G functions
from pointpats import PointPattern, PoissonPointProcess
from pointpats import k_function, l_function, g_function, f_function
import numpy as np
import matplotlib.pyplot as plt

pp = PointPattern(np.column_stack([x_coords, y_coords]))

# K-function
d, k_obs = k_function(pp, intervals=50)

# Monte Carlo simulation envelope for CSR
n_sim = 99
k_sims = np.zeros((n_sim, len(d)))
for i in range(n_sim):
    csr = PoissonPointProcess(pp.window, pp.n, 1, asPP=True)
    _, k_sims[i] = k_function(csr.realizations[0], intervals=50)

k_upper = np.percentile(k_sims, 97.5, axis=0)
k_lower = np.percentile(k_sims, 2.5, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(d, k_obs, 'b-', label='Observed K(d)', linewidth=2)
plt.fill_between(d, k_lower, k_upper, alpha=0.3, color='gray',
                 label='95% CSR envelope')
plt.plot(d, np.pi * d**2, 'r--', label='Theoretical CSR')
plt.xlabel('Distance (d)')
plt.ylabel('K(d)')
plt.legend()
plt.title("Ripley's K-function with CSR Envelope")
plt.savefig("k_function.png", dpi=150, bbox_inches='tight')
```

### Spatial Point Clustering

#### DBSCAN and HDBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) identifies clusters of arbitrary shape based on point density. HDBSCAN extends it with hierarchical clustering and automatic parameter selection.

```python
# Python - DBSCAN and HDBSCAN for spatial clustering
import numpy as np
from sklearn.cluster import DBSCAN
import hdbscan
import geopandas as gpd
from shapely.geometry import Point

coords = np.column_stack([x_coords, y_coords])

# --- DBSCAN ---
# eps: maximum distance between two points to be considered neighbors
# min_samples: minimum points to form a dense region
# For geographic coordinates, use haversine metric with radians
db = DBSCAN(eps=0.01, min_samples=5, metric='haversine')
labels_db = db.fit_predict(np.radians(coords))

n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()
print(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points")

# --- HDBSCAN (preferred -- fewer parameters, more robust) ---
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5,
                        metric='haversine')
labels_hdb = hdb.fit_predict(np.radians(coords))
print(f"HDBSCAN: {labels_hdb.max() + 1} clusters")
print(f"Cluster persistence: {hdb.cluster_persistence_}")

# --- Parameter selection for DBSCAN ---
# Use k-nearest neighbor distance plot to choose eps
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, metric='haversine')
nn.fit(np.radians(coords))
distances, _ = nn.kneighbors(np.radians(coords))
k_dist = np.sort(distances[:, -1])

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
plt.plot(k_dist)
plt.xlabel('Points (sorted by distance)')
plt.ylabel('5-NN distance')
plt.title('k-distance plot (look for elbow)')
plt.savefig("k_distance_plot.png", dpi=150)
# The "elbow" in this plot suggests an appropriate eps value
```

### Marked Point Patterns

When points carry additional information (marks) -- such as tree diameter, earthquake magnitude, or crime type -- you can analyze mark-correlation structure.

```r
# R - marked point pattern analysis
library(spatstat)

# Create marked point pattern
pp_marked <- ppp(x, y, window = owin(c(xmin, xmax), c(ymin, ymax)),
                 marks = dbh)  # dbh = diameter at breast height

# Mark correlation function (Stoyan's k_mm)
# Tests whether marks of nearby points are correlated
mc <- markcorr(pp_marked, correction = "isotropic")
plot(mc, main = "Mark Correlation Function")
# k_mm(r) > 1: nearby points have similar marks
# k_mm(r) < 1: nearby points have dissimilar marks

# Mark variogram (for continuous marks)
mv <- markvar(pp_marked, correction = "isotropic")
plot(mv, main = "Mark Variogram")

# For categorical marks (e.g., species type)
pp_multi <- ppp(x, y, window = win,
                marks = factor(species))
# Cross-type K-function
k_cross <- Kcross(pp_multi, i = "oak", j = "pine", correction = "isotropic")
plot(k_cross, main = "Cross K-function: Oak vs Pine")
```

---

## Spatial Autocorrelation

Spatial autocorrelation measures the degree to which nearby observations have similar (or dissimilar) values. It is one of the most fundamental concepts in spatial statistics -- Tobler's First Law of Geography states that "everything is related to everything else, but near things are more related than distant things."

### Why It Matters

1. **Violates independence assumption:** Most classical statistics assume i.i.d. observations. Spatial autocorrelation means your effective sample size is smaller than the actual sample size.
2. **Model specification:** Significant spatial autocorrelation in regression residuals signals model misspecification -- you need spatial regression models.
3. **Pattern detection:** Identifies meaningful spatial clusters and outliers that may point to underlying processes.

### Global Measures

#### Moran's I

The most widely used measure of global spatial autocorrelation. It is essentially a spatial correlation coefficient.

```
I = (n / S0) * (sum_i sum_j w_ij * (x_i - x_bar)(x_j - x_bar)) / (sum_i (x_i - x_bar)^2)
```

where n is the number of observations, w_ij is the spatial weight between i and j, S0 is the sum of all weights, and x_bar is the mean.

| Value | Interpretation |
|---|---|
| I close to +1 | Strong positive spatial autocorrelation (similar values cluster) |
| I close to E(I) = -1/(n-1) | Spatial randomness (no autocorrelation) |
| I close to -1 | Strong negative spatial autocorrelation (checkerboard pattern) |

**Inference:** Two approaches:
- **Analytical (normal approximation):** Fast but assumes normality. Can use randomization assumption (permutation of values) or normality assumption.
- **Permutation-based (conditional randomization):** Randomly permute values across locations many times (e.g., 999), compute I each time, compare observed I to the reference distribution. More robust, no distributional assumptions.

```python
# Python - Global Moran's I with PySAL
from libpysal.weights import Queen
from esda.moran import Moran
import geopandas as gpd
import numpy as np

gdf = gpd.read_file("census_tracts.gpkg")
y = gdf["median_income"].values

w = Queen.from_dataframe(gdf)
w.transform = 'r'

# Permutation-based inference (recommended)
mi = Moran(y, w, permutations=9999)
print(f"Moran's I:           {mi.I:.4f}")
print(f"Expected I:          {mi.EI:.4f}")
print(f"Variance (norm):     {mi.VI_norm:.6f}")
print(f"z-score (norm):      {mi.z_norm:.4f}")
print(f"p-value (norm):      {mi.p_norm:.4f}")
print(f"p-value (perm):      {mi.p_sim:.4f}")
print(f"# permutations:     {mi.permutations}")

# Effect size interpretation:
# |I| < 0.10  --> very weak
# 0.10-0.30   --> weak to moderate
# 0.30-0.50   --> moderate to strong
# > 0.50      --> strong
```

```r
# R - Global Moran's I with spdep
library(spdep)
library(sf)

shp <- st_read("census_tracts.gpkg")
nb <- poly2nb(shp, queen = TRUE)
lw <- nb2listw(nb, style = "W")

# Analytical test (normal approximation under randomization)
mt <- moran.test(shp$median_income, lw, randomisation = TRUE)
print(mt)

# Permutation test (Monte Carlo)
mc <- moran.mc(shp$median_income, lw, nsim = 9999)
print(mc)
plot(mc)  # Histogram of simulated I values with observed I marked
```

#### Geary's C

Geary's C focuses on squared differences between neighboring values (rather than cross-products like Moran's I). It is more sensitive to local spatial autocorrelation.

```
C = ((n-1) / (2 * S0)) * (sum_i sum_j w_ij * (x_i - x_j)^2) / (sum_i (x_i - x_bar)^2)
```

| Value | Interpretation |
|---|---|
| C close to 0 | Strong positive spatial autocorrelation |
| C = 1 | Spatial randomness |
| C > 1 | Negative spatial autocorrelation |

Note the inverse relationship with Moran's I: Moran's I and Geary's C are related as C approximately equals 1 - I for large samples, but they are not perfectly inversely related because they measure different aspects of spatial association.

```python
# Python
from esda.geary import Geary

gc = Geary(y, w, permutations=9999)
print(f"Geary's C:     {gc.C:.4f}")
print(f"Expected C:    {gc.EC:.4f}")
print(f"p-value (sim): {gc.p_sim:.4f}")
```

```r
# R
gt <- geary.test(shp$median_income, lw, randomisation = TRUE)
print(gt)
```

#### Getis-Ord General G

Tests whether high values (or low values) tend to cluster together. Unlike Moran's I and Geary's C, the General G statistic specifically detects clustering of high or low values.

```
G = (sum_i sum_j w_ij * x_i * x_j) / (sum_i sum_j x_i * x_j), for i != j
```

| Result | Interpretation |
|---|---|
| G > E(G), significant | Clustering of HIGH values |
| G < E(G), significant | Clustering of LOW values |
| Not significant | No spatial clustering of extreme values |

**Important:** General G requires all values to be positive (no negatives or zeros). It uses binary (0/1) spatial weights, typically distance-band weights.

```python
# Python
from esda.getisord import G

# Note: General G requires binary weights (not row-standardized)
from libpysal.weights import DistanceBand
w_binary = DistanceBand.from_dataframe(gdf, threshold=5000, binary=True)

g = G(y, w_binary, permutations=9999)
print(f"General G:     {g.G:.6f}")
print(f"Expected G:    {g.EG:.6f}")
print(f"z-score:       {g.z_sim:.4f}")
print(f"p-value:       {g.p_sim:.4f}")
```

### Local Measures (LISA)

Local Indicators of Spatial Association (LISA) decompose global measures into contributions from each observation, identifying WHERE spatial clusters and outliers are located.

#### Local Moran's I

The most commonly used LISA statistic. For each observation i:

```
I_i = z_i * sum_j(w_ij * z_j)
```

where z_i and z_j are standardized values.

Each observation is classified into one of four quadrants based on its value and the weighted average of its neighbors:

| Quadrant | Local I | Description | Map Color (convention) |
|---|---|---|---|
| HH | Positive | High value surrounded by high values | Red |
| LL | Positive | Low value surrounded by low values | Blue |
| HL | Negative | High value surrounded by low values | Pink/Light red |
| LH | Negative | Low value surrounded by high values | Light blue |

**Critical: Multiple testing correction.** When you test hundreds or thousands of features simultaneously, some will appear significant by chance alone. Always apply correction:

| Correction | Approach | Strictness |
|---|---|---|
| **Bonferroni** | Divide alpha by n | Very conservative (high Type II error) |
| **FDR (Benjamini-Hochberg)** | Controls false discovery rate | Moderate -- recommended for LISA |
| **Conditional permutation** | Built into PySAL/GeoDa | Default approach, moderate |

```python
# Python - Local Moran's I with full interpretation
from esda.moran import Moran_Local
from statsmodels.stats.multitest import multipletests
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

gdf = gpd.read_file("census_tracts.gpkg")
y = gdf["median_income"].values

w = Queen.from_dataframe(gdf)
w.transform = 'r'

lisa = Moran_Local(y, w, permutations=9999)

# Apply FDR correction
reject, pvals_corrected, _, _ = multipletests(lisa.p_sim, alpha=0.05,
                                                method='fdr_bh')

# Build classification
gdf["lisa_q"] = lisa.q       # 1=HH, 2=LH, 3=LL, 4=HL
gdf["lisa_p"] = lisa.p_sim
gdf["lisa_sig"] = reject
gdf["lisa_label"] = "Not Significant"
gdf.loc[(gdf["lisa_q"] == 1) & gdf["lisa_sig"], "lisa_label"] = "HH (Hot Spot)"
gdf.loc[(gdf["lisa_q"] == 2) & gdf["lisa_sig"], "lisa_label"] = "LH (Low-High)"
gdf.loc[(gdf["lisa_q"] == 3) & gdf["lisa_sig"], "lisa_label"] = "LL (Cold Spot)"
gdf.loc[(gdf["lisa_q"] == 4) & gdf["lisa_sig"], "lisa_label"] = "HL (High-Low)"

# Plot
colors = {"Not Significant": "#d3d3d3", "HH (Hot Spot)": "#e31a1c",
          "LL (Cold Spot)": "#1f78b4", "HL (High-Low)": "#fb9a99",
          "LH (Low-High)": "#a6cee3"}
fig, ax = plt.subplots(figsize=(12, 10))
for label, color in colors.items():
    subset = gdf[gdf["lisa_label"] == label]
    if len(subset) > 0:
        subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.3,
                    label=f"{label} (n={len(subset)})")
ax.legend(loc='lower right', fontsize=10)
ax.set_title("LISA Cluster Map (FDR-corrected, p < 0.05)")
ax.axis('off')
plt.savefig("lisa_map.png", dpi=200, bbox_inches='tight')
```

```r
# R - Local Moran's I with spdep
library(spdep)
library(sf)
library(ggplot2)

shp <- st_read("census_tracts.gpkg")
nb <- poly2nb(shp, queen = TRUE)
lw <- nb2listw(nb, style = "W")

# Local Moran's I
local_m <- localmoran(shp$median_income, lw, alternative = "two.sided")

# Extract results
shp$Ii <- local_m[, "Ii"]
shp$z_score <- local_m[, "Z.Ii"]
shp$p_value <- local_m[, "Pr(z != E(Ii))"]

# FDR correction
shp$p_fdr <- p.adjust(shp$p_value, method = "fdr")

# Classify quadrants
y_std <- scale(shp$median_income)
wy_std <- lag.listw(lw, y_std)

shp$quadrant <- NA
shp$quadrant[y_std > 0 & wy_std > 0] <- "HH"
shp$quadrant[y_std < 0 & wy_std < 0] <- "LL"
shp$quadrant[y_std > 0 & wy_std < 0] <- "HL"
shp$quadrant[y_std < 0 & wy_std > 0] <- "LH"
shp$quadrant[shp$p_fdr >= 0.05] <- "Not Significant"

# Plot
ggplot(shp) +
  geom_sf(aes(fill = quadrant), color = "grey30", linewidth = 0.1) +
  scale_fill_manual(values = c("HH" = "#e31a1c", "LL" = "#1f78b4",
                                "HL" = "#fb9a99", "LH" = "#a6cee3",
                                "Not Significant" = "#d3d3d3")) +
  labs(title = "LISA Cluster Map", fill = "Cluster Type") +
  theme_minimal()
```

#### Getis-Ord Gi*

Identifies statistically significant spatial clusters of high values (hot spots) and low values (cold spots). Unlike Local Moran's I, Gi* does not identify spatial outliers -- it focuses purely on the intensity of clustering.

```
Gi*(d) = (sum_j w_ij * x_j - X_bar * sum_j w_ij) /
         (S * sqrt((n * sum_j w_ij^2 - (sum_j w_ij)^2) / (n-1)))
```

The result is a z-score: positive z = hot spot, negative z = cold spot.

```python
# Python
from esda.getisord import G_Local

# Gi* includes self-neighbors (star version)
gi_star = G_Local(y, w, star=True, permutations=9999)

gdf["gi_z"] = gi_star.Zs
gdf["gi_p"] = gi_star.p_sim

# Classify
gdf["hotspot"] = "Not Significant"
gdf.loc[(gdf["gi_z"] > 1.96) & (gdf["gi_p"] < 0.05), "hotspot"] = "Hot Spot (95%)"
gdf.loc[(gdf["gi_z"] > 2.58) & (gdf["gi_p"] < 0.01), "hotspot"] = "Hot Spot (99%)"
gdf.loc[(gdf["gi_z"] < -1.96) & (gdf["gi_p"] < 0.05), "hotspot"] = "Cold Spot (95%)"
gdf.loc[(gdf["gi_z"] < -2.58) & (gdf["gi_p"] < 0.01), "hotspot"] = "Cold Spot (99%)"
```

```r
# R
gi <- localG(shp$median_income, lw)
shp$gi_z <- as.numeric(gi)
shp$gi_p <- 2 * pnorm(-abs(shp$gi_z))
```

#### Local Geary

A local version of Geary's C. It measures the squared differences between an observation and its neighbors, making it sensitive to negative spatial autocorrelation and useful for multivariate analysis.

```python
# Python - Multivariate Local Geary
from esda.geary_local_mv import Geary_Local_MV
import numpy as np

# Multivariate local Geary with multiple variables
variables = gdf[["income", "education", "housing_value"]].values
mlg = Geary_Local_MV(connectivity=w, data=variables, permutations=999)

gdf["local_geary"] = mlg.localG
gdf["local_geary_p"] = mlg.p_sim
```

---

## Spatial Weights

Spatial weights formalize the concept of "spatial neighborhood" -- they define which observations are neighbors and how strongly they are connected. The choice of spatial weights directly affects every subsequent spatial statistical analysis, so it deserves careful consideration.

### Weight Types

#### Contiguity-Based Weights

| Type | Definition | Typical # Neighbors | Best For |
|---|---|---|---|
| **Queen** | Shares any boundary or vertex | 5-8 for regular polygons | Census tracts, administrative units |
| **Rook** | Shares an edge only (not vertex) | 4-6 for regular polygons | Regular grids, raster-like data |
| **Bishop** | Shares a vertex only | Rare | Almost never used in practice |

```python
# Python - contiguity weights
from libpysal.weights import Queen, Rook

w_queen = Queen.from_dataframe(gdf)
w_rook = Rook.from_dataframe(gdf)

# Inspect connectivity
print(f"Queen: mean neighbors = {w_queen.mean_neighbors:.1f}")
print(f"Rook:  mean neighbors = {w_rook.mean_neighbors:.1f}")

# Check for islands (features with no neighbors)
if w_queen.islands:
    print(f"WARNING: {len(w_queen.islands)} island(s) detected: {w_queen.islands}")
    # Islands break many spatial statistics -- you must address them
    # Options: use KNN weights, snap geometries, manually assign neighbors
```

```r
# R - contiguity weights
library(spdep)
library(sf)

shp <- st_read("census_tracts.gpkg")

# Queen contiguity
nb_queen <- poly2nb(shp, queen = TRUE)
summary(nb_queen)

# Rook contiguity
nb_rook <- poly2nb(shp, queen = FALSE)

# Convert to listw object (row-standardized)
lw_queen <- nb2listw(nb_queen, style = "W", zero.policy = TRUE)
lw_rook <- nb2listw(nb_rook, style = "W", zero.policy = TRUE)

# Visualize the neighbor graph
plot(st_geometry(shp), border = "grey")
plot(nb_queen, st_coordinates(st_centroid(shp)), add = TRUE, col = "red")
```

#### Distance-Based Weights

| Type | Definition | Parameters | Best For |
|---|---|---|---|
| **KNN** | K closest features by centroid distance | k (# neighbors) | Irregular polygons, points, ensures connectivity |
| **Distance band** | All features within threshold distance | threshold distance | When a meaningful distance exists |
| **Inverse distance** | Weight = 1/d^alpha | alpha (distance decay power) | Continuous spatial processes |
| **Kernel** | Weight from kernel function (Gaussian, bi-square, etc.) | bandwidth, kernel type | Smoothing, GWR |

```python
# Python - distance-based weights
from libpysal.weights import KNN, DistanceBand, Kernel
from libpysal.weights.util import min_threshold_distance

# KNN (k=5)
w_knn = KNN.from_dataframe(gdf, k=5)
print(f"KNN(5): each feature has exactly {w_knn.max_neighbors} neighbors")

# Distance band
# First find the minimum threshold that ensures all features have >= 1 neighbor
min_thresh = min_threshold_distance(
    np.array(gdf.geometry.centroid.apply(lambda p: [p.x, p.y]).tolist())
)
print(f"Minimum threshold for full connectivity: {min_thresh:.1f}")

w_dist = DistanceBand.from_dataframe(gdf, threshold=min_thresh * 1.5, binary=False)
print(f"Distance band: mean neighbors = {w_dist.mean_neighbors:.1f}")

# Kernel weights (fixed Gaussian bandwidth)
w_kernel = Kernel.from_dataframe(gdf, bandwidth=5000, function='gaussian', fixed=True)
```

```r
# R - distance-based weights
library(spdep)

coords <- st_coordinates(st_centroid(shp))

# KNN (k = 5)
knn5 <- knearneigh(coords, k = 5)
nb_knn <- knn2nb(knn5)
lw_knn <- nb2listw(nb_knn, style = "W")

# Distance band
# Find minimum distance to ensure all have >= 1 neighbor
dists <- nbdists(knn2nb(knearneigh(coords, k = 1)), coords)
min_d <- max(unlist(dists))
cat("Min threshold for connectivity:", min_d, "\n")

nb_dist <- dnearneigh(coords, d1 = 0, d2 = min_d * 1.5)
lw_dist <- nb2listw(nb_dist, style = "W")

# Inverse distance weights
dlist <- nbdists(nb_dist, coords)
idw <- lapply(dlist, function(x) 1 / x)
lw_idw <- nb2listw(nb_dist, glist = idw, style = "W")
```

#### Higher-Order Weights

Higher-order weights capture spatial effects that extend beyond immediate neighbors.

```python
# Python - higher-order weights
from libpysal.weights import Queen, higher_order

w1 = Queen.from_dataframe(gdf)

# Second-order: neighbors of neighbors (but not self or first-order)
w2 = higher_order(w1, k=2)

# Cumulative: includes first AND second order
from libpysal.weights import W
w_cumulative = w1.__or__(w2)  # Union of first and second order
```

### Weight Standardization

| Style | Definition | When to Use |
|---|---|---|
| **Binary (B)** | w_ij = 1 if neighbors, 0 otherwise | Getis-Ord G statistic |
| **Row-standardized (W)** | Weights in each row sum to 1 | Most common; Moran's I, spatial regression |
| **Globally standardized (C)** | All weights sum to 1 | Comparing across study areas |
| **Variance-stabilizing (S)** | Minimizes variance of spatial lag | Advanced applications |

```python
# Row standardization (most common)
w.transform = 'r'  # 'r' = row-standardize, 'b' = binary, 'o' = original
```

### Spatial Weight Diagnostics

Always verify your weights before proceeding to analysis.

```python
# Python - weight diagnostics
w = Queen.from_dataframe(gdf)

# Basic summary
print(f"Number of observations:  {w.n}")
print(f"Number of nonzero links: {w.nonzero}")
print(f"Min neighbors:           {w.min_neighbors}")
print(f"Max neighbors:           {w.max_neighbors}")
print(f"Mean neighbors:          {w.mean_neighbors:.2f}")
print(f"Median neighbors:        {w.median:.0f}")
print(f"Islands:                 {w.islands}")
print(f"Histogram of cardinalities:")
for k, v in sorted(w.histogram):
    print(f"  {k} neighbors: {v} features")

# Connectivity plot
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 10))
gdf.boundary.plot(ax=ax, color='grey', linewidth=0.5)
# Plot neighbor links
for i, neighbors in w.neighbors.items():
    xi, yi = gdf.geometry.iloc[i].centroid.coords[0]
    for j in neighbors:
        xj, yj = gdf.geometry.iloc[j].centroid.coords[0]
        ax.plot([xi, xj], [yi, yj], 'r-', linewidth=0.3, alpha=0.5)
ax.set_title("Spatial Weight Graph (Queen Contiguity)")
ax.axis('off')
plt.savefig("weight_graph.png", dpi=150, bbox_inches='tight')
```

### Sensitivity Analysis

**Always test whether your results are robust to the choice of spatial weights.** If results change dramatically with different weight specifications, your conclusions are fragile.

```python
# Python - sensitivity to weight specification
from libpysal.weights import Queen, Rook, KNN, DistanceBand
from esda.moran import Moran

y = gdf["median_income"].values

weights = {
    "Queen": Queen.from_dataframe(gdf),
    "Rook": Rook.from_dataframe(gdf),
    "KNN(4)": KNN.from_dataframe(gdf, k=4),
    "KNN(6)": KNN.from_dataframe(gdf, k=6),
    "KNN(8)": KNN.from_dataframe(gdf, k=8),
    "KNN(10)": KNN.from_dataframe(gdf, k=10),
}

print(f"{'Weight':<12} {'Moran I':>10} {'p-value':>10}")
print("-" * 35)
for name, w in weights.items():
    w.transform = 'r'
    mi = Moran(y, w, permutations=999)
    print(f"{name:<12} {mi.I:>10.4f} {mi.p_sim:>10.4f}")
```

---

## Geostatistics

Geostatistics models spatial continuity using variograms and performs spatial prediction (interpolation) via kriging. Originally developed for mining applications by Matheron and Krige, it is now used across environmental science, soil science, hydrology, atmospheric science, and public health.

### Fundamental Assumptions

| Assumption | Description | When It Breaks |
|---|---|---|
| **Stationarity** | Statistical properties do not change across space | Strong trends, domain boundaries |
| **Intrinsic stationarity** | Variance of differences depends only on separation distance | Weaker than strict stationarity; sufficient for variograms |
| **Isotropy** | Spatial correlation is the same in all directions | Anisotropic processes (e.g., river valleys, wind patterns) |
| **Ergodicity** | Spatial averages equal ensemble averages | Very small study areas, non-repeating processes |

### The Variogram

The variogram (technically the semivariogram) is the cornerstone of geostatistics. It describes how spatial similarity decreases with distance.

```
gamma(h) = (1 / 2n(h)) * sum_i (Z(s_i) - Z(s_i + h))^2
```

where h is the lag distance, n(h) is the number of pairs at distance h, and Z(s) is the value at location s.

#### Variogram Anatomy

```
semivariance
    |
    |          ............................  sill (C0 + C1 = total variance)
    |        ..
    |      ..   <-- practical range (~95% of sill for exponential)
    |    ..
    |  ..
    | .
    |.
    +-- nugget (C0: measurement error + micro-scale variation)
    +----------------------------------------- distance (lag h)
```

| Component | Symbol | Meaning | Diagnostic |
|---|---|---|---|
| **Nugget** | C0 | Variance at zero distance (noise + micro-scale) | Large nugget = noisy data or fine-scale variation |
| **Partial sill** | C1 | Spatially structured variance | C1 / (C0 + C1) = proportion of variance that is spatially structured |
| **Sill** | C0 + C1 | Total variance (plateau) | Should approximately equal the sample variance |
| **Range** | a | Distance at which spatial correlation effectively disappears | Determines the "reach" of spatial prediction |

#### Variogram Models

| Model | Formula | Shape | Use When |
|---|---|---|---|
| **Spherical** | gamma(h) = C0 + C1[1.5(h/a) - 0.5(h/a)^3] for h <= a | S-curve, reaches sill exactly at range | Most common; general purpose |
| **Exponential** | gamma(h) = C0 + C1[1 - exp(-h/a)] | Asymptotic approach | Gradual correlation decay |
| **Gaussian** | gamma(h) = C0 + C1[1 - exp(-(h/a)^2)] | Very smooth near origin | Very smooth spatial processes |
| **Matern** | gamma(h) = C0 + C1[1 - (2^(1-nu)/Gamma(nu))(h/a)^nu K_nu(h/a)] | Flexible smoothness via nu | When you want to control smoothness |
| **Power** | gamma(h) = C0 + b*h^alpha | No sill (unbounded) | Fractal processes, intrinsic stationarity only |
| **Hole effect** | gamma(h) = C0 + C1[1 - sin(h/a)/(h/a)] | Oscillating | Periodic/cyclic spatial processes |

#### Computing and Fitting Variograms

```r
# R - complete variogram workflow with gstat
library(gstat)
library(sf)

# Load data (using meuse dataset as example)
data(meuse, package = "sp")
meuse_sf <- st_as_sf(meuse, coords = c("x", "y"), crs = 28992)

# --- Empirical variogram ---
v_emp <- variogram(log(zinc) ~ 1, data = meuse_sf)
print(v_emp)
plot(v_emp, main = "Empirical Variogram")

# --- Directional variogram (check for anisotropy) ---
v_dir <- variogram(log(zinc) ~ 1, data = meuse_sf,
                    alpha = c(0, 45, 90, 135))  # 4 directions
plot(v_dir, main = "Directional Variograms")
# If variograms differ by direction --> anisotropy is present

# --- Variogram cloud (all pairs) ---
v_cloud <- variogram(log(zinc) ~ 1, data = meuse_sf, cloud = TRUE)
plot(v_cloud, main = "Variogram Cloud")

# --- Fit multiple models and compare ---
# Spherical
v_sph <- fit.variogram(v_emp, vgm(psill = 0.5, "Sph", range = 900, nugget = 0.1))
# Exponential
v_exp <- fit.variogram(v_emp, vgm(psill = 0.5, "Exp", range = 900, nugget = 0.1))
# Gaussian
v_gau <- fit.variogram(v_emp, vgm(psill = 0.5, "Gau", range = 900, nugget = 0.1))
# Matern (nu = 1.5)
v_mat <- fit.variogram(v_emp, vgm(psill = 0.5, "Mat", range = 900,
                                    nugget = 0.1, kappa = 1.5))

# Compare fits visually
par(mfrow = c(2, 2))
plot(v_emp, v_sph, main = paste("Spherical (SSE =", round(attr(v_sph, "SSErr"), 2), ")"))
plot(v_emp, v_exp, main = paste("Exponential (SSE =", round(attr(v_exp, "SSErr"), 2), ")"))
plot(v_emp, v_gau, main = paste("Gaussian (SSE =", round(attr(v_gau, "SSErr"), 2), ")"))
plot(v_emp, v_mat, main = paste("Matern (SSE =", round(attr(v_mat, "SSErr"), 2), ")"))

# Use the model with lowest SSE (sum of squared errors)
cat("SSE Spherical:  ", attr(v_sph, "SSErr"), "\n")
cat("SSE Exponential:", attr(v_exp, "SSErr"), "\n")
cat("SSE Gaussian:   ", attr(v_gau, "SSErr"), "\n")
cat("SSE Matern:     ", attr(v_mat, "SSErr"), "\n")
```

```python
# Python - variogram with gstools
import gstools as gs
import numpy as np

# Empirical variogram
bin_center, gamma = gs.vario_estimate(
    (x_coords, y_coords), values,
    bin_edges=np.linspace(0, max_dist, 30)
)

# Fit models
models = {
    "Spherical": gs.Spherical(dim=2),
    "Exponential": gs.Exponential(dim=2),
    "Gaussian": gs.Gaussian(dim=2),
    "Matern": gs.Matern(dim=2, nu=1.5),
}

for name, model in models.items():
    model.fit_variogram(bin_center, gamma, nugget=True)
    print(f"{name:12s}: var={model.var:.3f}, len_scale={model.len_scale:.1f}, "
          f"nugget={model.nugget:.3f}")
    ax = model.plot(x_max=max_dist)
    ax.scatter(bin_center, gamma, color='k', label='Empirical')
    ax.set_title(name)
```

### Kriging Methods

| Method | Trend/Mean | Key Assumption | When to Use |
|---|---|---|---|
| **Simple Kriging (SK)** | Known constant mean | Mean is known a priori | Rare in practice; used in co-simulation |
| **Ordinary Kriging (OK)** | Unknown constant mean | Intrinsic stationarity | Default choice for most interpolation |
| **Universal Kriging (UK)** | Unknown polynomial trend | Non-stationary mean | Spatial trends in the data |
| **Regression Kriging (RK)** | Regression on covariates | Covariates explain trend | Auxiliary variables available at all locations |
| **Indicator Kriging (IK)** | Binary indicator variable | Threshold exceedance | Probability mapping (e.g., contamination > threshold) |
| **Co-Kriging (CK)** | Cross-correlated variables | Joint second-order stationarity | Sparse primary + dense secondary variable |
| **Block Kriging** | Mean over a block area | Same as OK | Predicting area-averaged values |

#### Ordinary Kriging -- Complete Workflow

```r
# R - Ordinary Kriging with gstat
library(gstat)
library(sf)
library(stars)

data(meuse, package = "sp")
data(meuse.grid, package = "sp")

meuse_sf <- st_as_sf(meuse, coords = c("x", "y"), crs = 28992)
grid_sf <- st_as_sf(meuse.grid, coords = c("x", "y"), crs = 28992)

# Step 1: Compute and fit variogram
v_emp <- variogram(log(zinc) ~ 1, data = meuse_sf)
v_fit <- fit.variogram(v_emp, vgm(psill = 0.59, "Sph", range = 897, nugget = 0.05))

# Step 2: Ordinary Kriging
ok_result <- krige(
  log(zinc) ~ 1,
  locations = meuse_sf,
  newdata = grid_sf,
  model = v_fit
)

# Step 3: Back-transform (if log-transformed)
ok_result$zinc_pred <- exp(ok_result$var1.pred)
ok_result$zinc_se <- sqrt(ok_result$var1.var)

# Step 4: Plot predictions and uncertainty
par(mfrow = c(1, 2))
plot(ok_result["zinc_pred"], main = "Kriging Prediction")
plot(ok_result["zinc_se"], main = "Kriging Std Error")
```

```python
# Python - Ordinary Kriging with pykrige
from pykrige.ok import OrdinaryKriging
import numpy as np

ok = OrdinaryKriging(
    x_coords, y_coords, values,
    variogram_model='spherical',
    variogram_parameters={'sill': 0.59, 'range': 897, 'nugget': 0.05},
    verbose=True,
    enable_plotting=False,
)

# Create prediction grid
grid_x = np.linspace(x_coords.min(), x_coords.max(), 100)
grid_y = np.linspace(y_coords.min(), y_coords.max(), 100)

z_pred, z_var = ok.execute('grid', grid_x, grid_y)

# z_pred: predicted values
# z_var: kriging variance (uncertainty)
```

#### Universal Kriging (Kriging with a Trend)

```r
# R - Universal Kriging with coordinate trend
# Appropriate when you see a spatial trend in your data
v_uk <- variogram(log(zinc) ~ x + y, data = meuse_sf)
v_uk_fit <- fit.variogram(v_uk, vgm(0.5, "Sph", 900, 0.1))

uk_result <- krige(
  log(zinc) ~ x + y,
  locations = meuse_sf,
  newdata = grid_sf,
  model = v_uk_fit
)

# Compare OK vs UK predictions
cor(ok_result$var1.pred, uk_result$var1.pred)
```

#### Regression Kriging

Regression kriging is a two-step approach: (1) fit a regression model with covariates, (2) krige the residuals.

```r
# R - Regression Kriging
library(gstat)
library(sf)

# Step 1: Regression model
lm_fit <- lm(log(zinc) ~ dist + ffreq + soil, data = meuse)

# Step 2: Variogram of residuals
meuse_sf$residuals <- residuals(lm_fit)
v_res <- variogram(residuals ~ 1, data = meuse_sf)
v_res_fit <- fit.variogram(v_res, vgm(0.2, "Sph", 600, 0.05))

# Step 3: Krige residuals
res_krige <- krige(residuals ~ 1, meuse_sf, grid_sf, model = v_res_fit)

# Step 4: Predict trend at grid locations
grid_sf$trend <- predict(lm_fit, newdata = grid_sf)

# Step 5: Combine
grid_sf$rk_pred <- grid_sf$trend + res_krige$var1.pred
```

### Cross-Validation

Cross-validation is essential to evaluate kriging performance and compare models.

```r
# R - Leave-one-out cross-validation
library(gstat)

# LOO-CV for ordinary kriging
cv_ok <- krige.cv(log(zinc) ~ 1, meuse_sf, model = v_fit, nfold = nrow(meuse_sf))

# Performance metrics
cat("ME  (should be ~0):", mean(cv_ok$residual), "\n")
cat("RMSE:              ", sqrt(mean(cv_ok$residual^2)), "\n")
cat("MAE:               ", mean(abs(cv_ok$residual)), "\n")
cat("R-squared:         ", 1 - var(cv_ok$residual) / var(log(meuse_sf$zinc)), "\n")
cat("Mean Std Error:    ", mean(sqrt(cv_ok$var1.var)), "\n")
cat("MSDR (should be ~1):", mean(cv_ok$residual^2 / cv_ok$var1.var), "\n")
# MSDR (Mean Squared Deviation Ratio) close to 1 means
# the kriging variance correctly estimates prediction uncertainty
```

```python
# Python - cross-validation with pykrige
from sklearn.model_selection import LeaveOneOut
import numpy as np

loo = LeaveOneOut()
residuals = []

for train_idx, test_idx in loo.split(x_coords):
    ok_cv = OrdinaryKriging(
        x_coords[train_idx], y_coords[train_idx], values[train_idx],
        variogram_model='spherical',
        variogram_parameters={'sill': 0.59, 'range': 897, 'nugget': 0.05},
    )
    z_pred, _ = ok_cv.execute('points',
                               x_coords[test_idx], y_coords[test_idx])
    residuals.append(values[test_idx] - z_pred.flatten())

residuals = np.array(residuals).flatten()
print(f"RMSE: {np.sqrt(np.mean(residuals**2)):.4f}")
print(f"MAE:  {np.mean(np.abs(residuals)):.4f}")
```

---

## Spatial Regression

When your data exhibits spatial autocorrelation, standard OLS regression violates the independence assumption, leading to biased standard errors, unreliable p-values, and potentially biased coefficient estimates. Spatial regression models explicitly account for spatial structure.

### The OLS-to-Spatial-Model Decision Workflow

```
Step 1: Fit OLS model
        y = X*beta + epsilon
        |
Step 2: Test residuals for spatial autocorrelation
        - Moran's I on OLS residuals
        |
        +---> NOT significant (p >= 0.05)
        |     --> OLS is fine, no spatial model needed
        |
        +---> SIGNIFICANT (p < 0.05)
              --> Spatial dependence exists, need spatial model
              |
Step 3: Run Lagrange Multiplier (LM) diagnostics
        |
        +---> LM-Lag significant, LM-Error NOT
        |     --> Spatial Lag Model (SLM / SAR)
        |
        +---> LM-Error significant, LM-Lag NOT
        |     --> Spatial Error Model (SEM)
        |
        +---> BOTH significant
        |     --> Compare Robust LM tests
        |     |
        |     +---> Robust LM-Lag significant --> SLM
        |     +---> Robust LM-Error significant --> SEM
        |     +---> Both Robust significant --> Spatial Durbin Model (SDM)
        |
        +---> Neither significant (rare)
              --> Consider GWR or different weights specification
              |
Step 4: Check if coefficients vary spatially
        - Fit GWR/MGWR
        - Map local coefficients
        - Test for spatial non-stationarity
```

### Model Specifications

| Model | Equation | Key Parameter | Interpretation |
|---|---|---|---|
| **OLS** | y = Xb + e | -- | No spatial effects |
| **SAR (Spatial Lag)** | y = rho*Wy + Xb + e | rho (spatial autoregressive) | Neighbor's y affects own y (diffusion/spillover) |
| **SEM (Spatial Error)** | y = Xb + u; u = lambda*Wu + e | lambda (spatial error) | Correlated errors due to omitted spatial variables |
| **SDM (Spatial Durbin)** | y = rho*Wy + Xb + WXg + e | rho, gamma | Both spatial lag and spatially lagged covariates |
| **SDEM (Spatial Durbin Error)** | y = Xb + WXg + u; u = lambda*Wu + e | lambda, gamma | Error autocorrelation + lagged covariates |
| **SARAR (SAC)** | y = rho*Wy + Xb + u; u = lambda*Wu + e | rho, lambda | Both lag and error autocorrelation |
| **SLX** | y = Xb + WXg + e | gamma | Only spatially lagged covariates (no autoregressive) |

### Complete Spatial Regression Workflow in Python

```python
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen
from spreg import OLS, ML_Lag, ML_Error, GM_Lag, GM_Error

# --- Data preparation ---
gdf = gpd.read_file("census_tracts.gpkg")
w = Queen.from_dataframe(gdf)
w.transform = 'r'

y = gdf[["house_price"]].values
X = gdf[["sqft", "distance_cbd", "green_pct", "school_quality"]].values
var_names = ["sqft", "dist_cbd", "green_pct", "school_quality"]

# --- Step 1: OLS with spatial diagnostics ---
ols = OLS(y, X, w=w, spat_diag=True,
          name_y="house_price", name_x=var_names,
          name_w="queen", name_ds="census_tracts")
print(ols.summary)

# Key diagnostics to check in OLS output:
# 1. Moran's I on residuals (p-value)
# 2. LM (lag) test statistic and p-value
# 3. LM (error) test statistic and p-value
# 4. Robust LM (lag) and Robust LM (error)

# --- Step 2: Based on diagnostics, fit appropriate model ---

# Maximum Likelihood Spatial Lag Model
ml_lag = ML_Lag(y, X, w=w, name_y="house_price", name_x=var_names)
print(ml_lag.summary)
print(f"Spatial autoregressive parameter (rho): {ml_lag.rho:.4f}")
print(f"Log-likelihood: {ml_lag.logll:.2f}")
print(f"AIC: {ml_lag.aic:.2f}")

# Maximum Likelihood Spatial Error Model
ml_err = ML_Error(y, X, w=w, name_y="house_price", name_x=var_names)
print(ml_err.summary)
print(f"Spatial error parameter (lambda): {ml_err.lam:.4f}")
print(f"Log-likelihood: {ml_err.logll:.2f}")
print(f"AIC: {ml_err.aic:.2f}")

# --- Compare models ---
print("\nModel Comparison:")
print(f"{'Model':<20} {'Log-LL':>10} {'AIC':>10}")
print("-" * 42)
print(f"{'OLS':<20} {ols.logll:>10.2f} {ols.aic:>10.2f}")
print(f"{'ML Spatial Lag':<20} {ml_lag.logll:>10.2f} {ml_lag.aic:>10.2f}")
print(f"{'ML Spatial Error':<20} {ml_err.logll:>10.2f} {ml_err.aic:>10.2f}")
# Lower AIC = better model fit
```

### Complete Spatial Regression Workflow in R

```r
library(spdep)
library(spatialreg)
library(sf)

shp <- st_read("census_tracts.gpkg")
nb <- poly2nb(shp, queen = TRUE)
lw <- nb2listw(nb, style = "W")

# --- Step 1: OLS ---
ols <- lm(house_price ~ sqft + distance_cbd + green_pct + school_quality,
           data = shp)
summary(ols)

# --- Step 2: Moran's I test on residuals ---
moran_res <- lm.morantest(ols, lw)
print(moran_res)

# --- Step 3: Lagrange Multiplier diagnostics ---
lm_tests <- lm.LMtests(ols, lw,
                         test = c("LMerr", "LMlag", "RLMerr", "RLMlag", "SARMA"))
print(lm_tests)

# --- Step 4: Fit spatial models based on diagnostics ---

# Spatial Lag Model (SAR)
sar <- lagsarlm(house_price ~ sqft + distance_cbd + green_pct + school_quality,
                 data = shp, listw = lw)
summary(sar)
# rho = spatial autoregressive coefficient

# Spatial Error Model (SEM)
sem <- errorsarlm(house_price ~ sqft + distance_cbd + green_pct + school_quality,
                   data = shp, listw = lw)
summary(sem)
# lambda = spatial error coefficient

# Spatial Durbin Model (SDM)
sdm <- lagsarlm(house_price ~ sqft + distance_cbd + green_pct + school_quality,
                 data = shp, listw = lw, type = "mixed")
summary(sdm)

# SARAR (SAC) model
sac <- sacsarlm(house_price ~ sqft + distance_cbd + green_pct + school_quality,
                 data = shp, listw = lw)
summary(sac)

# --- Step 5: Model comparison ---
AIC(ols, sar, sem, sdm, sac)

# --- Step 6: Impact measures for SAR/SDM (direct + indirect effects) ---
impacts_sar <- impacts(sar, listw = lw, R = 999)
summary(impacts_sar, zstats = TRUE, short = TRUE)
# Direct effect:   effect of x_i on y_i
# Indirect effect:  effect of x_i on y_j (spillover to neighbors)
# Total effect:     direct + indirect
```

### Estimation Methods

| Method | Description | Pros | Cons |
|---|---|---|---|
| **Maximum Likelihood (ML)** | Maximize the log-likelihood | Asymptotically efficient, direct LR tests | Assumes normality, computationally intensive for large n |
| **GMM (Method of Moments)** | Match theoretical and sample moments | No normality assumption, faster | Less efficient than ML |
| **Bayesian MCMC** | Full posterior distribution | Uncertainty quantification, priors | Slow, complex |
| **2SLS (Instrumental Variables)** | Instrument spatial lag with W^2X, W^3X, etc. | Handles endogeneity | May be inefficient |

```python
# Python - GMM estimation (faster, no normality assumption)
from spreg import GM_Lag, GM_Error, GM_Combo

# GMM Spatial Lag
gm_lag = GM_Lag(y, X, w=w, name_y="house_price", name_x=var_names)
print(gm_lag.summary)

# GMM Spatial Error (with heteroskedasticity-robust)
gm_err = GM_Error(y, X, w=w, name_y="house_price", name_x=var_names)
print(gm_err.summary)

# GMM SARAR (Combo: lag + error)
gm_combo = GM_Combo(y, X, w=w, name_y="house_price", name_x=var_names)
print(gm_combo.summary)
```

---

## GWR and MGWR

Geographically Weighted Regression (GWR) allows regression coefficients to vary across space, capturing spatial non-stationarity -- the idea that relationships between variables may differ from place to place.

### When to Use GWR

- **Residuals from a global model show spatial patterns** (clustered positive/negative residuals)
- **Theory suggests relationships vary spatially** (e.g., housing price determinants differ between urban and suburban areas)
- **You want to map local coefficient surfaces** to explore spatial heterogeneity

### GWR Specification

For each location i, GWR fits a local regression:

```
y_i = beta_0(u_i, v_i) + sum_k beta_k(u_i, v_i) * x_ik + epsilon_i
```

where (u_i, v_i) are the coordinates and each beta_k varies as a function of location. The local model at each point is weighted by a kernel function centered on that point -- nearby observations get more weight.

### Bandwidth Selection

The bandwidth controls how many nearby observations contribute to each local regression. It is the most critical parameter in GWR.

| Method | Description | Implementation |
|---|---|---|
| **AICc minimization** | Minimize corrected Akaike Information Criterion | Default in mgwr, GWmodel |
| **CV (cross-validation)** | Minimize prediction error | Alternative option |
| **Fixed bandwidth** | Same distance for all locations | Use when study area is roughly homogeneous |
| **Adaptive bandwidth** | Fixed number of nearest neighbors | Use when density varies across space (recommended) |

### GWR in Python (mgwr)

```python
import geopandas as gpd
import numpy as np
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW

gdf = gpd.read_file("census_tracts.gpkg")

# Prepare coordinates and variables
coords = np.column_stack([
    gdf.geometry.centroid.x,
    gdf.geometry.centroid.y
])
y = gdf[["house_price"]].values
X = gdf[["sqft", "distance_cbd", "green_pct"]].values

# --- Standard GWR ---

# Step 1: Select optimal bandwidth (adaptive, using AICc)
bw_selector = Sel_BW(coords, y, X, kernel='bisquare', fixed=False)
optimal_bw = bw_selector.search(criterion='AICc')
print(f"Optimal bandwidth (# neighbors): {optimal_bw}")

# Step 2: Fit GWR model
gwr_model = GWR(coords, y, X, bw=optimal_bw, kernel='bisquare', fixed=False)
gwr_results = gwr_model.fit()

print(gwr_results.summary())

# Step 3: Extract and map local coefficients
gdf["intercept"] = gwr_results.params[:, 0]
gdf["beta_sqft"] = gwr_results.params[:, 1]
gdf["beta_cbd"] = gwr_results.params[:, 2]
gdf["beta_green"] = gwr_results.params[:, 3]

# Local R-squared
gdf["local_r2"] = gwr_results.localR2

# Local t-values for significance testing
# gwr_results.filter_tvals() applies multiple testing correction
t_critical = gwr_results.critical_tval()  # Adjusted critical t-value
gdf["sqft_tval"] = gwr_results.tvalues[:, 1]
gdf["sqft_sig"] = np.abs(gdf["sqft_tval"]) > t_critical

# Step 4: Map coefficients
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (col, title) in enumerate(zip(
    ["beta_sqft", "beta_cbd", "beta_green"],
    ["Sqft Effect", "CBD Distance Effect", "Green Space Effect"]
)):
    gdf.plot(column=col, ax=axes[i], legend=True, cmap='RdBu_r',
             edgecolor='black', linewidth=0.2)
    axes[i].set_title(title)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig("gwr_coefficients.png", dpi=200, bbox_inches='tight')
```

### Multi-Scale GWR (MGWR)

MGWR extends GWR by allowing each covariate to have its own bandwidth. This is crucial because different processes operate at different spatial scales.

```python
# --- Multi-Scale GWR (MGWR) ---

# Step 1: Select variable-specific bandwidths
mgwr_selector = Sel_BW(coords, y, X, multi=True, kernel='bisquare', fixed=False)
mgwr_bws = mgwr_selector.search()
print(f"MGWR bandwidths: {mgwr_bws}")
# Example output: [150, 45, 200, 80]
# Interpretation: intercept operates at scale 150, sqft at 45 (very local),
# CBD distance at 200 (regional), green space at 80 (neighborhood)

# Step 2: Fit MGWR
mgwr_model = MGWR(coords, y, X, selector=mgwr_selector,
                    kernel='bisquare', fixed=False)
mgwr_results = mgwr_model.fit()
print(mgwr_results.summary())

# Step 3: Compare GWR vs MGWR
print(f"\nGWR  AICc: {gwr_results.aicc:.2f}")
print(f"MGWR AICc: {mgwr_results.aicc:.2f}")
# Lower AICc = better fit

# Step 4: Bandwidth interpretation
var_names = ["Intercept", "Sqft", "CBD Distance", "Green Space"]
for name, bw in zip(var_names, mgwr_bws):
    if bw > 0.8 * len(gdf):
        scale = "global (nearly constant across space)"
    elif bw > 0.3 * len(gdf):
        scale = "regional"
    else:
        scale = "local (varies substantially across space)"
    print(f"  {name}: bandwidth = {bw:.0f} --> {scale}")
```

### GWR in R (GWmodel)

```r
library(GWmodel)
library(sf)
library(sp)

shp <- st_read("census_tracts.gpkg")
sp_data <- as(shp, "Spatial")  # GWmodel uses sp objects

# --- Bandwidth selection ---
bw_gwr <- bw.gwr(house_price ~ sqft + distance_cbd + green_pct,
                   data = sp_data,
                   approach = "AICc",
                   kernel = "bisquare",
                   adaptive = TRUE)
cat("Optimal bandwidth:", bw_gwr, "neighbors\n")

# --- Fit GWR ---
gwr_result <- gwr.basic(house_price ~ sqft + distance_cbd + green_pct,
                          data = sp_data,
                          bw = bw_gwr,
                          kernel = "bisquare",
                          adaptive = TRUE)
print(gwr_result)

# Local coefficients are in gwr_result$SDF
# Map them
spplot(gwr_result$SDF, "sqft",
       main = "GWR: Local coefficient for Sqft")

# --- Multi-scale GWR (MGWR equivalent in R) ---
# GWmodel provides gwr.multiscale()
mgwr_result <- gwr.multiscale(
  house_price ~ sqft + distance_cbd + green_pct,
  data = sp_data,
  kernel = "bisquare",
  adaptive = TRUE,
  criterion = "dCVR",  # or "CVR"
  max.iterations = 1000
)
print(mgwr_result)
```

### GWR Interpretation Pitfalls

1. **Multicollinearity:** Local multicollinearity can be severe even when global VIF is acceptable. Check local condition numbers -- values > 30 indicate problematic multicollinearity.
2. **Multiple testing:** Mapping all t-values without correction inflates false positives. Use the adjusted critical t-value from `gwr_results.critical_tval()`.
3. **Effective number of parameters:** GWR has many more effective parameters than OLS. Always compare models using AICc (not AIC or R-squared).
4. **Boundary effects:** Coefficient estimates are less reliable near the edges of the study area where fewer observations contribute.
5. **Not causal:** Spatial variation in coefficients does not imply spatially varying causal effects -- it may reflect omitted variable bias that varies spatially.

---

## Bayesian Spatial Models

Bayesian approaches to spatial statistics offer several advantages: natural uncertainty quantification, incorporation of prior knowledge, and flexible model specification. Two dominant frameworks exist: INLA (fast approximate Bayesian inference) and MCMC (exact but slower).

### When to Use Bayesian Spatial Models

- **Disease mapping and epidemiology:** Smoothing unstable rates in areas with small populations
- **Ecological modeling:** Species distribution with spatial random effects
- **Small area estimation:** Borrowing strength from neighbors
- **Complex hierarchical structures:** Spatial + temporal + group-level effects
- **Uncertainty quantification:** Full posterior distributions rather than point estimates

### R-INLA (Integrated Nested Laplace Approximation)

INLA provides fast, deterministic approximate Bayesian inference for latent Gaussian models. It is orders of magnitude faster than MCMC for spatial models.

```r
# R - Disease mapping with BYM model (Besag-York-Mollie)
# The gold standard for spatial disease mapping
library(INLA)
library(sf)
library(spdep)

shp <- st_read("health_districts.gpkg")

# Prepare adjacency structure for INLA
nb <- poly2nb(shp, queen = TRUE)
nb2INLA("adj_graph.txt", nb)
adj_graph <- inla.read.graph("adj_graph.txt")

# Data: observed counts, expected counts, covariate
shp$id <- 1:nrow(shp)

# BYM model: y_i ~ Poisson(E_i * exp(eta_i))
# eta_i = beta_0 + beta_1*x_i + u_i + v_i
# u_i = spatially structured random effect (ICAR / Besag)
# v_i = unstructured random effect (iid)

formula_bym <- observed ~ 1 + poverty_rate +
  f(id, model = "bym2", graph = adj_graph,
    hyper = list(
      phi = list(prior = "pc", param = c(0.5, 2/3)),  # mixing parameter
      prec = list(prior = "pc.prec", param = c(1, 0.01))  # precision
    ))

result_bym <- inla(
  formula_bym,
  family = "poisson",
  data = as.data.frame(shp),
  E = shp$expected,  # expected counts (offset)
  control.compute = list(dic = TRUE, waic = TRUE, cpo = TRUE),
  control.predictor = list(compute = TRUE)
)

summary(result_bym)

# Extract results
shp$rr <- result_bym$summary.fitted.values$mean           # Relative Risk
shp$rr_lower <- result_bym$summary.fitted.values$`0.025quant`
shp$rr_upper <- result_bym$summary.fitted.values$`0.975quant`

# Exceedance probability: P(RR > 1)
shp$prob_excess <- 1 - result_bym$summary.fitted.values$`0.5quant`
# Or more precisely using the marginals:
for (i in 1:nrow(shp)) {
  shp$prob_excess[i] <- 1 - inla.pmarginal(
    1, result_bym$marginals.fitted.values[[i]])
}

# Plot relative risk map
library(ggplot2)
ggplot(shp) +
  geom_sf(aes(fill = rr), color = "grey50", linewidth = 0.1) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                        midpoint = 1, name = "Relative\nRisk") +
  labs(title = "Disease Mapping: Smoothed Relative Risk (BYM2 Model)") +
  theme_minimal()
```

### Bayesian Spatial Regression with brms (Stan backend)

```r
# R - Bayesian spatial regression with brms
library(brms)
library(sf)

shp <- st_read("census_tracts.gpkg")

# Prepare spatial structure
coords <- st_coordinates(st_centroid(shp))
shp$x <- coords[, 1]
shp$y <- coords[, 2]

# CAR (Conditional Autoregressive) spatial model
# Requires adjacency matrix
library(spdep)
nb <- poly2nb(shp, queen = TRUE)
W <- nb2mat(nb, style = "B")  # Binary adjacency matrix

# Fit Bayesian spatial model
fit_car <- brm(
  house_price ~ sqft + distance_cbd + green_pct + car(W, gr = id, type = "icar"),
  data = shp,
  data2 = list(W = W),
  family = gaussian(),
  chains = 4, iter = 4000, warmup = 1000,
  cores = 4,
  control = list(adapt_delta = 0.95)
)

summary(fit_car)
plot(fit_car)  # Trace plots and posterior densities

# Posterior predictive checks
pp_check(fit_car, ndraws = 100)

# Extract spatial random effects
spatial_effects <- ranef(fit_car)$id
shp$spatial_effect <- spatial_effects[, "Estimate", "Intercept"]
```

### Model Comparison

| Framework | Speed | Flexibility | Complexity | Best For |
|---|---|---|---|---|
| **R-INLA** | Very fast (seconds to minutes) | Latent Gaussian models only | Moderate | Disease mapping, spatial smoothing, routine Bayesian spatial |
| **Stan/brms** | Slow (minutes to hours) | Any model | High | Custom models, non-Gaussian, complex hierarchical |
| **NIMBLE** | Moderate | Very flexible | High | Custom MCMC, ecology |
| **PyMC** | Moderate | Very flexible | Moderate | Python users, general Bayesian |

---

## Spatial Clustering and Regionalization

Spatial clustering groups observations into regions that are both internally homogeneous (similar attributes) and spatially contiguous. This is distinct from standard clustering (e.g., k-means) because it respects geographic adjacency.

### Methods

| Method | Approach | Key Parameters | Implementation |
|---|---|---|---|
| **SKATER** | Minimum spanning tree pruning | k (number of clusters), spatial weights | PySAL (`spopt`), `spdep` (R) |
| **REDCAP** | Region-building using dendrogram | k, linkage method | PySAL (`spopt`) |
| **Max-p** | Maximize number of regions subject to threshold | threshold (min attribute sum per region) | PySAL (`spopt`), GeoDa |
| **AZP** | Automatic Zoning Procedure | k, objective function | PySAL (`spopt`) |
| **Ward + contiguity** | Hierarchical clustering with spatial constraint | k, spatial weights | `scikit-learn` (connectivity), `ClustGeo` (R) |
| **K-means + contiguity** | K-means with adjacency constraint | k | Custom implementation |

### Spatially Constrained Clustering in Python

```python
import geopandas as gpd
import numpy as np
from libpysal.weights import Queen
from spopt.region import Skater, MaxPHeuristic
from sklearn.preprocessing import StandardScaler

gdf = gpd.read_file("census_tracts.gpkg")

# Variables for clustering (standardize first)
attrs = ["median_income", "pct_college", "median_age", "pct_owner_occupied"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(gdf[attrs].values)

# Spatial weights
w = Queen.from_dataframe(gdf)

# --- SKATER (Spatial 'K'luster Analysis by Tree Edge Removal) ---
skater = Skater(gdf, w, attrs, n_clusters=8, floor=10)
skater.solve()

gdf["skater_cluster"] = skater.labels_
print(f"SKATER: {len(set(skater.labels_))} clusters")

# --- Max-p Regionalization ---
# Maximize the number of regions such that each region meets
# a minimum threshold (e.g., minimum population of 50,000)
maxp = MaxPHeuristic(gdf, w, attrs,
                      threshold_name="population",
                      threshold=50000,
                      top_n=10)
maxp.solve()
gdf["maxp_cluster"] = maxp.labels_
print(f"Max-p: {len(set(maxp.labels_))} regions")
```

### Spatially Constrained Clustering in R

```r
# R - spatially constrained hierarchical clustering
library(ClustGeo)
library(sf)
library(spdep)

shp <- st_read("census_tracts.gpkg")

# Attribute dissimilarity matrix
attrs <- scale(st_drop_geometry(shp[, c("median_income", "pct_college",
                                          "median_age", "pct_owner")]))
D0 <- dist(attrs)

# Geographic dissimilarity matrix (from adjacency)
nb <- poly2nb(shp, queen = TRUE)
coords <- st_coordinates(st_centroid(shp))
D1 <- as.dist(as.matrix(dist(coords)))

# Choose mixing parameter alpha (balances attribute vs spatial homogeneity)
# Use choicealpha() to find optimal alpha
cr <- choicealpha(D0, D1, range.alpha = seq(0, 1, 0.05), K = 8, graph = TRUE)
# Look for the "elbow" where geographic homogeneity increases
# without too much loss of attribute homogeneity

# Fit with chosen alpha
tree <- hclustgeo(D0, D1, alpha = 0.3)
shp$cluster <- cutree(tree, k = 8)

# Plot
library(ggplot2)
ggplot(shp) +
  geom_sf(aes(fill = factor(cluster)), color = "white", linewidth = 0.2) +
  scale_fill_brewer(palette = "Set2", name = "Region") +
  labs(title = "Spatially Constrained Clusters (ClustGeo, alpha=0.3)") +
  theme_minimal()
```

### Geodemographic Classification

A specific application of spatial clustering for creating neighborhood typologies (e.g., census-based socioeconomic profiles).

```python
# Python - geodemographic classification workflow
import geopandas as gpd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from libpysal.weights import Queen
from spopt.region import Skater

gdf = gpd.read_file("census_tracts.gpkg")

# Select geodemographic variables
vars_demo = [
    "median_income", "pct_college", "median_age",
    "pct_owner_occupied", "pct_minority", "unemployment_rate",
    "median_home_value", "pct_single_family"
]

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(gdf[vars_demo].values)

# Find optimal k using silhouette score
sil_scores = {}
for k in range(4, 15):
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = kmeans.fit_predict(X)
    sil_scores[k] = silhouette_score(X, labels)
    print(f"k={k}: silhouette={sil_scores[k]:.3f}")

optimal_k = max(sil_scores, key=sil_scores.get)
print(f"\nOptimal k: {optimal_k}")

# Spatially constrained version
w = Queen.from_dataframe(gdf)
skater = Skater(gdf, w, vars_demo, n_clusters=optimal_k, floor=5)
skater.solve()
gdf["geodemo_type"] = skater.labels_

# Profile each type
profiles = gdf.groupby("geodemo_type")[vars_demo].mean()
print("\nGeodemo Type Profiles:")
print(profiles.round(2))
```

---

## Hot Spot Analysis

Hot spot analysis identifies statistically significant spatial clusters of high values (hot spots) and low values (cold spots). It is one of the most widely used spatial statistical techniques in crime analysis, epidemiology, retail analytics, and urban planning.

### Getis-Ord Gi* -- Detailed Guide

The Gi* statistic (pronounced "G-i-star") measures the concentration of high or low values around each feature relative to the global mean.

**Interpretation of z-scores:**

| z-score | Confidence | Interpretation |
|---|---|---|
| > +2.58 | 99% | Very significant hot spot |
| +1.96 to +2.58 | 95% | Significant hot spot |
| +1.65 to +1.96 | 90% | Marginal hot spot |
| -1.65 to +1.65 | -- | Not significant |
| -1.96 to -1.65 | 90% | Marginal cold spot |
| -2.58 to -1.96 | 95% | Significant cold spot |
| < -2.58 | 99% | Very significant cold spot |

### Common Pitfalls

1. **Edge effects:** Features at the boundary of the study area have fewer neighbors, which can bias Gi* values. Buffer your study area if possible.
2. **Weight specification sensitivity:** Results change with different distance thresholds. Run sensitivity analysis.
3. **Multiple testing:** When testing many features, some will appear significant by chance. Apply FDR correction.
4. **Spatial scale:** Hot spots at one scale may disappear at another. Test multiple distance thresholds.
5. **Confounding with population:** High crime counts in dense areas may simply reflect population density, not elevated risk. Use rates or normalize.

### Emerging Hot Spot Analysis (Space-Time)

Emerging hot spot analysis extends Gi* to space-time data, classifying each location into temporal trend categories.

```python
# Python - space-time hot spot analysis concept
import geopandas as gpd
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from libpysal.weights import DistanceBand
from scipy.stats import theilslopes

# Assume panel data: locations x time periods
gdf = gpd.read_file("census_tracts.gpkg")
time_periods = ["crime_2018", "crime_2019", "crime_2020",
                "crime_2021", "crime_2022"]

w = DistanceBand.from_dataframe(gdf, threshold=5000, binary=False)
w.transform = 'r'

# Compute Gi* for each time period
gi_results = {}
for t in time_periods:
    gi = G_Local(gdf[t].values, w, star=True, permutations=999)
    gi_results[t] = gi.Zs

gi_df = pd.DataFrame(gi_results, index=gdf.index)

# Classify temporal trends using Mann-Kendall or Theil-Sen slope
def classify_hotspot(z_series):
    """Classify emerging hot spot type based on Gi* z-score time series."""
    sig_hot = (z_series > 1.96).sum()
    sig_cold = (z_series < -1.96).sum()
    n = len(z_series)
    latest = z_series.iloc[-1]
    slope = theilslopes(z_series.values, np.arange(n))[0]

    if sig_hot == n:
        if slope > 0.1:
            return "Intensifying Hot Spot"
        elif slope < -0.1:
            return "Diminishing Hot Spot"
        else:
            return "Persistent Hot Spot"
    elif sig_hot > 0 and latest > 1.96 and z_series.iloc[0] < 1.96:
        return "New Hot Spot"
    elif sig_hot > 0 and latest < 1.96 and z_series.iloc[-2] > 1.96:
        return "Historical Hot Spot"
    elif sig_cold == n:
        return "Persistent Cold Spot"
    elif sig_cold > 0 and latest < -1.96 and z_series.iloc[0] > -1.96:
        return "New Cold Spot"
    elif slope > 0.05 and sig_hot >= n // 2:
        return "Sporadic Hot Spot"
    else:
        return "No Pattern"

gdf["hotspot_type"] = gi_df.apply(classify_hotspot, axis=1)
print(gdf["hotspot_type"].value_counts())
```

### Optimized Hot Spot Analysis

When you do not know the appropriate distance threshold, you can search for the scale that maximizes the clustering signal.

```python
# Python - optimized distance threshold for hot spot analysis
from esda.getisord import G_Local
from libpysal.weights import DistanceBand
import numpy as np

y = gdf["crime_rate"].values

# Search across multiple thresholds
thresholds = np.arange(500, 10001, 500)
results = []

for d in thresholds:
    w = DistanceBand.from_dataframe(gdf, threshold=d, binary=False)
    w.transform = 'r'
    gi = G_Local(y, w, star=True, permutations=99)
    n_hot = (gi.Zs > 1.96).sum()
    n_cold = (gi.Zs < -1.96).sum()
    mean_abs_z = np.mean(np.abs(gi.Zs))
    results.append({
        'threshold': d,
        'n_hot': n_hot,
        'n_cold': n_cold,
        'mean_abs_z': mean_abs_z,
        'pct_significant': (n_hot + n_cold) / len(y) * 100
    })

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Choose the threshold that maximizes mean absolute z-score
# (strongest overall clustering signal)
best_thresh = results_df.loc[results_df['mean_abs_z'].idxmax(), 'threshold']
print(f"\nOptimal threshold: {best_thresh}")
```

---

## Spatial Econometrics

Spatial econometrics extends standard econometric models to account for spatial dependence and spatial heterogeneity. It is particularly relevant for panel data (multiple regions observed over multiple time periods).

### Panel Spatial Models

| Model | Specification | Use Case |
|---|---|---|
| **Spatial Lag Panel** | y_it = rho*W*y_it + X_it*b + mu_i + e_it | Spatial spillovers in panel data |
| **Spatial Error Panel** | y_it = X_it*b + mu_i + u_it; u_it = lambda*W*u_it + e_it | Spatially correlated errors in panel |
| **Spatial Durbin Panel** | y_it = rho*W*y_it + X_it*b + W*X_it*g + mu_i + e_it | Spillovers in both y and X |
| **Fixed effects** | mu_i = area-specific intercepts | Control for unobserved heterogeneity |
| **Random effects** | mu_i ~ N(0, sigma^2) | Assume effects uncorrelated with X |

### Direct and Indirect Effects

In spatial lag models, a change in x_i affects not only y_i (direct effect) but also y_j for neighbors (indirect effect / spillover). The total effect is the sum of direct and indirect effects.

```
For SAR: y = (I - rho*W)^(-1) * (X*beta + epsilon)

Direct effect of x_k:   mean diagonal of (I - rho*W)^(-1) * beta_k
Indirect effect of x_k: mean off-diagonal of (I - rho*W)^(-1) * beta_k
Total effect:            direct + indirect
```

**This is why you should NOT directly interpret SAR coefficients as marginal effects.** The coefficient beta_k in a spatial lag model is NOT the marginal effect of x_k on y. You must compute the proper impact measures.

### Spatial Panel Models in R

```r
# R - spatial panel data with splm
library(splm)
library(spdep)
library(plm)

# Panel data: regions x time periods
panel_data <- read.csv("regional_panel.csv")
# Columns: region_id, year, gdp_pc, investment, education, ...

# Spatial weights (same for all time periods)
shp <- st_read("regions.gpkg")
nb <- poly2nb(shp, queen = TRUE)
lw <- nb2listw(nb, style = "W")

# Convert to pdata.frame
pdata <- pdata.frame(panel_data, index = c("region_id", "year"))

# --- Test for spatial effects in panel ---
# Baltagi, Song & Koh (2003) joint test
bsk_test <- bsktest(gdp_pc ~ investment + education,
                     data = pdata, listw = lw, test = "LM1")
print(bsk_test)

# --- Fixed-effects spatial lag panel ---
fe_sar <- spml(gdp_pc ~ investment + education,
                data = pdata, listw = lw,
                model = "within",   # fixed effects
                spatial.error = "none",
                lag = TRUE)
summary(fe_sar)

# --- Fixed-effects spatial error panel ---
fe_sem <- spml(gdp_pc ~ investment + education,
                data = pdata, listw = lw,
                model = "within",
                spatial.error = "b",  # Baltagi error
                lag = FALSE)
summary(fe_sem)

# --- Random-effects spatial lag panel ---
re_sar <- spml(gdp_pc ~ investment + education,
                data = pdata, listw = lw,
                model = "random",
                spatial.error = "none",
                lag = TRUE)
summary(re_sar)

# --- Compute impacts (direct/indirect effects) ---
W_mat <- as(as_dgRMatrix_listw(lw), "CsparseMatrix")
impacts_fe <- impacts(fe_sar, listw = lw, time = length(unique(panel_data$year)))
summary(impacts_fe, zstats = TRUE)
```

### Spatial Panel Models in Python

```python
# Python - spatial panel models with spreg
from spreg import Panel_FE_Lag, Panel_FE_Error
import numpy as np
import pandas as pd
from libpysal.weights import Queen

# Load panel data
panel = pd.read_csv("regional_panel.csv")
gdf = gpd.read_file("regions.gpkg")
w = Queen.from_dataframe(gdf)
w.transform = 'r'

# Reshape for spreg panel format
# spreg expects data stacked by time, ordered by cross-section within each time
panel = panel.sort_values(["year", "region_id"])

y = panel[["gdp_pc"]].values
X = panel[["investment", "education"]].values

n_regions = len(gdf)
n_time = panel["year"].nunique()

# Fixed-effects spatial lag panel
fe_lag = Panel_FE_Lag(y, X, w, n_regions, n_time,
                       name_y="gdp_pc",
                       name_x=["investment", "education"])
print(fe_lag.summary)
```

### Common Pitfalls in Spatial Econometrics

1. **Endogeneity:** The spatial lag Wy is endogenous by construction. Use ML, GMM, or 2SLS -- never OLS.
2. **Weight matrix exogeneity:** Spatial weights must be exogenous (not determined by the dependent variable). Distance and contiguity weights are generally safe; trade-flow weights may be problematic.
3. **Interpreting coefficients:** In SAR/SDM models, coefficients are NOT marginal effects. Always compute direct/indirect/total effects via impacts().
4. **Model selection:** The LM test framework (Anselin 1988) is designed for cross-sectional data. For panels, use the Baltagi-Song-Koh tests or compare models via information criteria.
5. **Computational burden:** ML estimation with large panels is computationally intensive. Consider GMM or Bayesian approaches for very large panels.

---

## Tool Comparison

### Feature Matrix

| Capability | PySAL (Python) | spdep/spatialreg (R) | GeoDa (GUI) | ArcGIS Pro | QGIS |
|---|---|---|---|---|---|
| **Spatial weights** | libpysal.weights | spdep::nb2listw | Weights Manager | Generate Spatial Weights | -- |
| **Global Moran's I** | esda.Moran | spdep::moran.test | Space menu | Spatial Autocorrelation | -- |
| **LISA (Local Moran)** | esda.Moran_Local | spdep::localmoran | Space menu | Cluster & Outlier Analysis | -- |
| **Getis-Ord Gi*** | esda.G_Local | spdep::localG | Space menu | Hot Spot Analysis | -- |
| **Local Geary** | esda.Geary_Local | spdep::localC | -- | -- | -- |
| **OLS + diagnostics** | spreg.OLS | lm + lm.LMtests | Regression menu | OLS | -- |
| **Spatial Lag (SAR)** | spreg.ML_Lag | spatialreg::lagsarlm | Regression menu | -- | -- |
| **Spatial Error (SEM)** | spreg.ML_Error | spatialreg::errorsarlm | Regression menu | -- | -- |
| **Spatial Durbin** | spreg.ML_Lag (SDM) | spatialreg::lagsarlm(type="mixed") | -- | -- | -- |
| **GWR** | mgwr.GWR | GWmodel::gwr.basic | -- (use GWR4) | GWR tool | -- |
| **MGWR** | mgwr.MGWR | GWmodel::gwr.multiscale | -- | -- | -- |
| **Kriging** | pykrige, gstools | gstat::krige | -- | Kriging | -- |
| **Variograms** | gstools.Variogram | gstat::variogram | -- | Semivariogram | -- |
| **Point patterns** | pointpats | spatstat | Limited | -- | -- |
| **Bayesian spatial** | PyMC | R-INLA, brms | -- | -- | -- |
| **Spatial panel** | spreg (Panel_FE_*) | splm | -- | -- | -- |
| **Clustering (constrained)** | spopt (Skater, MaxP) | spdep::skater | Space menu | -- | -- |
| **KDE** | scipy, sklearn | spatstat::density | -- | Kernel Density | Heatmap |

### Performance Benchmarks (Approximate)

| Operation | n = 1K | n = 10K | n = 50K | n = 100K |
|---|---|---|---|---|
| **Queen weights (PySAL)** | <1s | 2s | 15s | 40s |
| **Moran's I (PySAL, 999 perm)** | <1s | 3s | 20s | 60s |
| **LISA (PySAL, 999 perm)** | <1s | 5s | 30s | 90s |
| **ML Spatial Lag (PySAL)** | <1s | 5s | 45s | 3min |
| **OK Kriging (gstat, R)** | <1s | 10s | 5min | 20min+ |
| **GWR (mgwr, Python)** | 2s | 30s | 10min | 30min+ |
| **INLA BYM (R-INLA)** | 2s | 10s | 1min | 5min |

### Learning Curve and Recommendations

| Profile | Recommended Tool | Reasoning |
|---|---|---|
| **Beginner / Exploration** | GeoDa | Free GUI, linked views, excellent tutorials by Luc Anselin |
| **Python data scientist** | PySAL + GeoPandas | Seamless integration with pandas/sklearn/matplotlib |
| **R statistician** | spdep + spatialreg + gstat | Deep integration with tidyverse, ggplot2, extensive documentation |
| **Bayesian modeler** | R-INLA (fast) or brms/Stan (flexible) | INLA for speed, Stan for custom models |
| **GIS analyst** | ArcGIS Pro + PySAL | ArcGIS for data management, PySAL for statistics |
| **Large datasets** | PySAL (sparse weights) + DuckDB Spatial | Aggregation in DuckDB, modeling in PySAL |
| **Publication** | R or Python scripts | Full reproducibility |
| **Teaching** | GeoDa + PySAL notebooks | Visual + code combination |

### Ecosystem Comparison

| Aspect | PySAL (Python) | R Spatial | GeoDa |
|---|---|---|---|
| **Maintainer** | Center for Geospatial Sciences, UC Riverside | Community + R Spatial Task View | Luc Anselin's lab |
| **License** | BSD 3-Clause | Various (GPL, MIT) | GPLv3 |
| **Documentation** | Good (growing) | Excellent | Excellent tutorials |
| **Textbook support** | "Geographic Data Science with Python" | "Applied Spatial Data Analysis with R" | "GeoDa Workbook" |
| **Active development** | Very active | Active | Moderate |
| **Integration** | GeoPandas, scikit-learn, Dask | sf, terra, tidyverse, ggplot2 | Standalone |
| **Reproducibility** | High (scripts/notebooks) | High (scripts/Rmarkdown) | Low (manual GUI steps) |
| **Scalability** | Good (sparse matrices, planned Dask) | Moderate (memory-bound) | Limited (desktop) |
| **Community** | Growing rapidly | Large, established | Moderate |

### Migration Guide

Moving between tools? Here are the key equivalences:

```
# Spatial Weights
PySAL:  Queen.from_dataframe(gdf)
R:      poly2nb(shp, queen=TRUE) |> nb2listw(style="W")
GeoDa:  Tools > Weights Manager > Queen Contiguity

# Global Moran's I
PySAL:  Moran(y, w, permutations=999)
R:      moran.test(y, lw) / moran.mc(y, lw, nsim=999)
GeoDa:  Space > Univariate Moran's I

# LISA
PySAL:  Moran_Local(y, w, permutations=999)
R:      localmoran(y, lw)
GeoDa:  Space > Univariate Local Moran's I

# Spatial Lag Model
PySAL:  ML_Lag(y, X, w=w)
R:      lagsarlm(y ~ x1 + x2, data, listw=lw)
GeoDa:  Regression > Classical > Spatial Lag

# Spatial Error Model
PySAL:  ML_Error(y, X, w=w)
R:      errorsarlm(y ~ x1 + x2, data, listw=lw)
GeoDa:  Regression > Classical > Spatial Error
```

---

## Key References

### Foundational Textbooks

| Book | Author(s) | Focus |
|---|---|---|
| *Spatial Econometrics: Methods and Models* | Anselin (1988) | Spatial regression theory |
| *Geographic Data Science with Python* | Rey, Arribas-Bel, Wolf (2023) | Modern Python spatial analysis |
| *Applied Spatial Data Analysis with R* | Bivand, Pebesma, Gomez-Rubio (2013) | R-based spatial statistics |
| *Statistics for Spatial Data* | Cressie (1993) | Geostatistics theory |
| *Model-based Geostatistics* | Diggle & Ribeiro (2007) | Bayesian geostatistics |
| *Statistical Analysis of Spatial and Spatio-Temporal Point Patterns* | Diggle (2013) | Point pattern analysis |
| *Geographically Weighted Regression* | Fotheringham, Brunsdon, Charlton (2002) | GWR theory |
| *Spatial Analysis Along Networks* | Okabe & Sugihara (2012) | Network spatial analysis |
| *Bayesian Disease Mapping* | Lawson (2018) | Bayesian spatial epidemiology |

### Key Software Documentation

- **PySAL:** https://pysal.org/
- **spdep/spatialreg:** https://r-spatial.github.io/spdep/ and https://r-spatial.github.io/spatialreg/
- **spatstat:** https://spatstat.org/
- **gstat:** https://r-spatial.github.io/gstat/
- **R-INLA:** https://www.r-inla.org/
- **mgwr:** https://mgwr.readthedocs.io/
- **GeoDa:** https://geodacenter.github.io/
- **GWmodel:** https://cran.r-project.org/package=GWmodel

---

[Back to Data Analysis](README.md) | [Back to Main README](../README.md)
