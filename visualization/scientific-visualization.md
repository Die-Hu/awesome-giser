# Scientific & Statistical Visualization

> SOTA tools for biology, ecology, social science, statistics, and cross-disciplinary data visualization that complement geospatial analysis.

> **Quick Picks**
> - **SOTA**: [Observable Plot](https://observablehq.com/plot/) + [D3.js](https://d3js.org) -- grammar of graphics with native geographic projections
> - **Free Best**: [R ggplot2](https://ggplot2.tidyverse.org) + [sf](https://r-spatial.github.io/sf/) -- publication-quality statistical maps, the academic gold standard
> - **Fastest Setup**: [Datawrapper](https://www.datawrapper.de) -- upload a CSV, get a publication-ready chart in minutes

---

## Table of Contents

| Section | Description |
|---------|-------------|
| [1. Statistical Visualization](#1-statistical-visualization) | Distributions, correlations, regression diagnostics, spatial statistics |
| [2. Ecological Visualization](#2-ecological-visualization) | Species distribution, biodiversity, habitat connectivity, phenology |
| [3. Biological & Medical Visualization](#3-biological--medical-visualization) | Epidemiology, environmental health, genomic geography, facility access |
| [4. Social Science Visualization](#4-social-science-visualization) | Demographics, inequality, survey data, electoral mapping, historical GIS |
| [5. Climate & Environmental Visualization](#5-climate--environmental-visualization) | Climate stripes, wind flow, sea level rise, air quality, wildfire |
| [6. Multi-dimensional Data Visualization](#6-multi-dimensional-data-visualization) | Parallel coordinates, Sankey diagrams, treemaps, alluvial plots |
| [7. Interactive Exploration Tools](#7-interactive-exploration-tools) | Crossfilter, faceting, drill-down, brushing and linking |
| [8. Publication-Quality Output](#8-publication-quality-output) | Python, R, LaTeX, export formats, reproducibility pipelines |
| [9. Tool Comparison Matrix](#9-tool-comparison-matrix) | Comprehensive feature comparison across 14 tools |

---

## 1. Statistical Visualization

Statistical charts are the backbone of quantitative spatial analysis. Every GIS professional encounters data that needs to be summarized, compared, or diagnosed before it ever lands on a map.

### Distribution Plots

Understanding the shape of your data is the first step in any analysis. Choose the plot type based on what aspect of distribution matters most.

| Plot Type | What It Shows | When to Use | Watch Out |
|-----------|--------------|-------------|-----------|
| **Histogram** | Frequency distribution with binned counts | Quick overview of a single variable | Bin width choice dramatically changes interpretation |
| **Density plot** | Smoothed probability distribution (KDE) | Comparing distributions, continuous data | Bandwidth selection introduces bias; can show impossible values |
| **Violin plot** | Mirrored density + box plot hybrid | Comparing distributions across groups | Unfamiliar to general audiences; explain in legend |
| **Ridgeline plot** | Stacked density plots with overlap | Comparing many distributions (e.g., monthly temps) | Order matters; overlap can obscure small distributions |
| **Raincloud plot** | Density + box plot + raw data points | Maximum information density | Complex; best for technical audiences |
| **ECDF** | Empirical cumulative distribution | Comparing to theoretical distributions; percentile queries | Less intuitive than histograms for non-statisticians |

#### Ridgeline Plot Example (Python, plotnine)

```python
from plotnine import *
import pandas as pd

# Monthly temperature distributions across weather stations
df = pd.read_csv("station_temperatures.csv")

(
    ggplot(df, aes(x="temperature", y="month", fill="month"))
    + geom_density_ridges(alpha=0.7, scale=1.5)
    + scale_fill_brewer(type="qual", palette="Set3")
    + labs(
        title="Monthly Temperature Distributions",
        x="Temperature (C)",
        y=""
    )
    + theme_minimal()
    + theme(legend_position="none")
)
```

#### Raincloud Plot Example (R, ggdist)

```r
library(ggplot2)
library(ggdist)

ggplot(station_data, aes(x = region, y = rainfall, fill = region)) +
  stat_halfeye(adjust = 0.5, width = 0.6, .width = 0, justification = -0.2) +
  geom_boxplot(width = 0.15, outlier.shape = NA) +
  geom_jitter(width = 0.05, alpha = 0.3, size = 0.8) +
  coord_flip() +
  labs(title = "Rainfall Distribution by Region",
       x = NULL, y = "Annual Rainfall (mm)") +
  theme_minimal() +
  theme(legend.position = "none")
```

### Correlation Visualization

Correlation plots reveal relationships between variables -- critical when deciding which covariates to include in spatial models.

| Plot Type | Variables | Best For | Tool Recommendation |
|-----------|-----------|----------|---------------------|
| **Scatterplot matrix (SPLOM)** | 3--8 continuous | Exploring all pairwise relationships | `GGally::ggpairs()` in R, `plotly.express.scatter_matrix()` |
| **Correlogram** | 5--50 continuous | Quick overview of correlation structure | `corrplot` in R, `seaborn.heatmap()` |
| **Bubble plot** | 3 continuous (x, y, size) | Adding a third dimension to scatter | Observable Plot, ggplot2 |
| **Hexbin plot** | 2 continuous (large n) | Overplotted scatter replacement | `matplotlib.hexbin()`, `ggplot2::geom_hex()` |

#### Correlogram Example (R, corrplot)

```r
library(corrplot)

# Correlation matrix for environmental variables
env_vars <- soil_data[, c("ph", "organic_matter", "clay_pct",
                           "elevation", "slope", "ndvi", "rainfall")]
cor_matrix <- cor(env_vars, use = "pairwise.complete.obs")

corrplot(cor_matrix,
         method = "ellipse",
         type = "upper",
         order = "hclust",
         tl.col = "black",
         tl.srt = 45,
         addCoef.col = "black",
         number.cex = 0.7,
         col = colorRampPalette(c("#2166AC", "white", "#B2182B"))(200))
```

### Regression Diagnostics

Before trusting any spatial model, diagnose it visually. These plots catch problems that summary statistics miss.

| Plot | What It Reveals | Red Flags |
|------|----------------|-----------|
| **Residuals vs. Fitted** | Non-linearity, heteroscedasticity | Funnel shape, curved pattern |
| **Q-Q plot** | Departures from normality | S-curve (heavy tails), offset (skew) |
| **Scale-Location** | Homoscedasticity check | Upward trend in spread |
| **Cook's Distance** | Influential observations | Points exceeding 4/n threshold |
| **Residuals vs. Leverage** | Outliers with high influence | Points in upper/lower right corners |
| **Partial regression** | Individual predictor contribution | Non-linear relationships that need transformation |
| **Added variable plot** | Effect of adding a predictor | Whether a new variable improves the model |

#### Diagnostic Plot Grid (Python, statsmodels)

```python
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Fit OLS model for housing prices with spatial covariates
model = sm.OLS.from_formula(
    "log_price ~ distance_cbd + elevation + noise_db + green_pct",
    data=parcels
).fit()

fig = sm.graphics.plot_regress_exog(model, "distance_cbd", fig=plt.figure(figsize=(12, 8)))
plt.tight_layout()
plt.savefig("regression_diagnostics.pdf", dpi=300)
```

### Spatial Statistics Visualization

These plots bridge the gap between statistics and geography. They are the visual language of spatial autocorrelation and geostatistics.

| Plot Type | Purpose | Key Interpretation |
|-----------|---------|-------------------|
| **Moran scatterplot** | Visualize spatial autocorrelation | Slope = Moran's I; quadrants show HH, LL, HL, LH |
| **LISA cluster map** | Local spatial association | Hot spots (HH), cold spots (LL), spatial outliers (HL, LH) |
| **Variogram (semivariogram)** | Model spatial dependence structure | Nugget, sill, range define kriging parameters |
| **Kriging surface** | Interpolated prediction surface | Smooth surface with prediction uncertainty |
| **Variogram cloud** | Raw semivariance at all lags | Detect anisotropy, outliers in spatial structure |
| **Covariogram** | Spatial covariance vs. distance | Alternative to variogram; decays rather than increases |

#### Moran Scatterplot (Python, PySAL/esda)

```python
import geopandas as gpd
from libpysal.weights import Queen
from esda.moran import Moran
from splot.esda import moran_scatterplot
import matplotlib.pyplot as plt

gdf = gpd.read_file("census_tracts.gpkg")
w = Queen.from_dataframe(gdf)
w.transform = "r"

moran = Moran(gdf["median_income"], w)

fig, ax = moran_scatterplot(moran, aspect_equal=True)
ax.set_title(f"Moran's I = {moran.I:.3f} (p = {moran.p_sim:.4f})")
plt.tight_layout()
plt.savefig("moran_scatterplot.png", dpi=300, bbox_inches="tight")
```

#### LISA Cluster Map (Python, PySAL/splot)

```python
from esda.moran import Moran_Local
from splot.esda import lisa_cluster

lisa = Moran_Local(gdf["median_income"], w)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
lisa_cluster(lisa, gdf, p=0.05, ax=ax)
ax.set_title("LISA Cluster Map: Median Income")
ax.set_axis_off()
plt.savefig("lisa_clusters.png", dpi=300, bbox_inches="tight")
```

#### Variogram Fitting (R, gstat)

```r
library(gstat)
library(sf)

soil <- st_read("soil_samples.gpkg")

# Compute empirical variogram
v <- variogram(zinc ~ 1, data = soil, cutoff = 1500)

# Fit theoretical variogram model
v_fit <- fit.variogram(v, vgm(psill = 1, model = "Sph", range = 900, nugget = 0.1))

plot(v, v_fit,
     main = "Empirical vs. Fitted Variogram (Spherical)",
     xlab = "Distance (m)",
     ylab = "Semivariance")
```

### SOTA Tools for Statistical Visualization

| Tool | Language | Strengths | Limitations |
|------|----------|-----------|-------------|
| [Observable Plot](https://observablehq.com/plot/) | JS | Grammar of graphics, geographic projections, reactive | Requires Observable runtime or npm setup |
| [Vega-Lite](https://vega.github.io/vega-lite/) | JSON/JS | Declarative spec, excellent interaction grammar | Verbose for complex plots |
| [Altair](https://altair-viz.github.io/) | Python | Pythonic Vega-Lite wrapper, concise API | Large dataset performance; exports as Vega spec |
| [ggplot2](https://ggplot2.tidyverse.org/) | R | The reference implementation of grammar of graphics | Static output; interactivity needs plotly/ggiraph |
| [Plotly](https://plotly.com/) | Python/R/JS | Interactive by default, 3D support | Opinionated styling; large bundle size |
| [seaborn](https://seaborn.pydata.org/) | Python | Statistical plots with minimal code | Less flexible than matplotlib for customization |

---

## 2. Ecological Visualization

Ecological data has unique properties: it is often sparse, spatially biased, temporally variable, and involves complex community structures. The visualization tools below are designed for these challenges.

### Species Distribution Maps

| Map Type | Data Requirement | Method | Tool |
|----------|-----------------|--------|------|
| **Presence/absence** | Binary occurrence records | Point map with classification | QGIS, R (sf + ggplot2) |
| **Abundance map** | Count or density data | Graduated symbols, choropleth | ggplot2, tmap |
| **Range map** | Occurrence polygons or modeled extent | Convex hull, alpha shapes, SDM output | R (rangeBuilder, ENMTools) |
| **SDM probability surface** | Environmental predictors + occurrences | MaxEnt, random forest, GAM | R (dismo, sdm, biomod2), Python (scikit-learn) |
| **Occupancy map** | Repeat survey detection/non-detection | Occupancy model output | R (unmarked) |

#### Species Distribution Model Output (R, ggplot2 + terra)

```r
library(ggplot2)
library(terra)
library(tidyterra)

pred_raster <- rast("maxent_prediction.tif")
occurrences <- read.csv("species_occurrences.csv")

ggplot() +
  geom_spatraster(data = pred_raster) +
  scale_fill_viridis_c(
    name = "Habitat\nSuitability",
    option = "magma",
    na.value = "transparent"
  ) +
  geom_point(
    data = occurrences,
    aes(x = longitude, y = latitude),
    color = "cyan", size = 1.2, alpha = 0.6
  ) +
  labs(title = "Predicted Habitat Suitability (MaxEnt)",
       subtitle = "Occurrence points overlaid") +
  theme_minimal() +
  coord_sf(crs = 4326)
```

### Biodiversity Index Mapping

Biodiversity indices summarize community composition into single values that can be mapped spatially.

| Index | Formula Essence | Interpretation | Range |
|-------|----------------|----------------|-------|
| **Shannon (H')** | -sum(p_i * ln(p_i)) | Higher = more diverse + even | 0 to ln(S) |
| **Simpson (1-D)** | 1 - sum(p_i^2) | Probability two random individuals differ | 0 to 1 |
| **Species Richness (S)** | Count of species | Raw count, no abundance weighting | 0 to infinity |
| **Pielou's Evenness (J)** | H' / ln(S) | How evenly distributed are abundances | 0 to 1 |
| **Chao1** | S + (f1^2 / 2*f2) | Estimated true richness including unseen | S to infinity |

#### Biodiversity Map (R, vegan + sf)

```r
library(vegan)
library(sf)
library(ggplot2)

# Community matrix: rows = sites, columns = species
community <- read.csv("community_matrix.csv", row.names = 1)
sites <- st_read("sample_sites.gpkg")

# Calculate diversity indices
sites$shannon <- diversity(community, index = "shannon")
sites$simpson <- diversity(community, index = "simpson")
sites$richness <- specnumber(community)

ggplot(sites) +
  geom_sf(aes(color = shannon), size = 3) +
  scale_color_viridis_c(option = "plasma", name = "Shannon H'") +
  labs(title = "Shannon Diversity Index by Sample Site") +
  theme_minimal()
```

### Habitat Connectivity Visualization

Habitat connectivity analysis identifies corridors and barriers for wildlife movement. The results require specialized visualization techniques.

| Visualization | What It Shows | Tool |
|--------------|---------------|------|
| **Resistance surface** | Cost of movement through each cell | QGIS raster styling, R (terra) |
| **Least-cost path** | Optimal corridor between patches | Circuitscape, R (gdistance) |
| **Current flow map** | Cumulative current density (circuit theory) | Circuitscape, Omniscape |
| **Corridor pinch points** | Narrow passages where connectivity is vulnerable | Linkage Mapper, Circuitscape |
| **Core area / buffer** | Habitat patches with edge effects | FRAGSTATS, R (landscapemetrics) |

#### Circuit Theory Connectivity (Python, visualization of Circuitscape output)

```python
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

with rasterio.open("circuitscape_curmap.tif") as src:
    current = src.read(1)
    extent = [src.bounds.left, src.bounds.right,
              src.bounds.bottom, src.bounds.top]

current[current == src.nodata] = np.nan

fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(current, extent=extent, cmap="inferno",
               norm=LogNorm(vmin=np.nanpercentile(current, 5),
                            vmax=np.nanpercentile(current, 99)))
plt.colorbar(im, ax=ax, label="Current Density", shrink=0.7)
ax.set_title("Habitat Connectivity: Circuit Theory Current Flow")
ax.set_xlabel("Easting (m)")
ax.set_ylabel("Northing (m)")
plt.tight_layout()
plt.savefig("connectivity_current.png", dpi=300)
```

### Phenology Visualization

Phenology -- the timing of biological events -- is inherently temporal and spatial. Visualizations must capture both dimensions.

| Plot Type | X-axis | Y-axis | Use Case |
|-----------|--------|--------|----------|
| **Seasonal NDVI curve** | Day of year | NDVI | Crop growth monitoring, green-up timing |
| **Growing degree day map** | Space (map) | Color = GDD accumulation | Predict pest emergence, planting windows |
| **Phenophase calendar** | Month | Species/location | Compare flowering, leafout, migration timing |
| **Space-time cube** | Lon, Lat | Time (z-axis) | Migration tracking, phenological wave |

#### NDVI Time Series (Python, matplotlib)

```python
import matplotlib.pyplot as plt
import pandas as pd

ndvi = pd.read_csv("ndvi_timeseries.csv", parse_dates=["date"])
ndvi["doy"] = ndvi["date"].dt.dayofyear

fig, ax = plt.subplots(figsize=(12, 5))
for year, group in ndvi.groupby(ndvi["date"].dt.year):
    ax.plot(group["doy"], group["ndvi"], label=str(year), alpha=0.7)

ax.set_xlabel("Day of Year")
ax.set_ylabel("NDVI")
ax.set_title("Seasonal NDVI Curve (2018-2025)")
ax.legend(loc="upper right", ncol=2)
ax.set_xlim(1, 365)
ax.set_ylim(0, 1)
ax.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="Green-up threshold")
plt.tight_layout()
plt.savefig("ndvi_phenology.png", dpi=300)
```

### Population Dynamics

| Plot Type | What It Shows | When to Use |
|-----------|---------------|-------------|
| **Time series** | Abundance over time | Basic population monitoring |
| **Phase portrait** | N(t) vs. dN/dt | Detect equilibria, cycles, chaos |
| **Age structure pyramid** | Age/size class distribution | Assess population viability |
| **Leslie matrix projection** | Projected population trajectory | Management scenario comparison |
| **Mark-recapture estimate** | Population size with confidence | Abundance estimation from capture data |

### SOTA Tools for Ecological Visualization

| Tool | Language | Specialization |
|------|----------|---------------|
| [vegan](https://cran.r-project.org/package=vegan) | R | Community ecology ordination, diversity |
| [adehabitatHR](https://cran.r-project.org/package=adehabitatHR) | R | Home range estimation, utilization distributions |
| [landscapemetrics](https://r-spatialecology.github.io/landscapemetrics/) | R | Landscape pattern analysis (FRAGSTATS equivalent) |
| [scikit-bio](http://scikit-bio.org/) | Python | Diversity metrics, distance matrices, ordination |
| [moveVis](https://movevis.org/) | R | Animated movement trajectory visualization |
| [Circuitscape](https://circuitscape.org/) | Julia | Circuit-theory connectivity modeling |

> **Cross-reference:** [Agriculture & Food Security](../data-sources/agriculture-food-security.md) | [Climate & Weather](../data-sources/climate-weather.md)

---

## 3. Biological & Medical Visualization

Health geography and medical mapping require careful treatment of sensitive data, small-number problems, and the ecological fallacy. The visualizations below are tailored to these constraints.

### Epidemiological Maps

| Map Type | Data | Method | Caveat |
|----------|------|--------|--------|
| **Disease incidence choropleth** | Case counts + population | Rate per 100k, age-standardized | Small-number instability in rural areas |
| **Smoothed rate surface** | Point-level cases | Kernel density, Bayesian smoothing | Smoothing can obscure real clusters |
| **SIR model output** | Compartmental model results | Animated chloropleth over time | Model assumptions drive visual output |
| **Contact tracing network** | Transmission links | Force-directed graph layout | Privacy concerns; anonymize |
| **Excess mortality map** | Observed vs. expected deaths | Standardized mortality ratio (SMR) | Baseline period selection matters |
| **Space-time scan** | Cases with location + time | SaTScan cluster detection | Multiple testing; report p-values |

#### SaTScan Cluster Visualization (R, sf + ggplot2)

```r
library(sf)
library(ggplot2)

counties <- st_read("counties.gpkg")
clusters <- read.csv("satscan_clusters.csv")

# SaTScan outputs cluster center, radius, and significance
cluster_circles <- st_buffer(
  st_as_sf(clusters, coords = c("lon", "lat"), crs = 4326),
  dist = clusters$radius_km * 1000  # convert to meters
)

ggplot() +
  geom_sf(data = counties, fill = "gray95", color = "gray60") +
  geom_sf(data = cluster_circles, aes(fill = relative_risk),
          alpha = 0.4, color = "red") +
  scale_fill_gradient(low = "yellow", high = "red",
                      name = "Relative\nRisk") +
  labs(title = "Spatial Clusters of Disease Incidence (SaTScan)",
       subtitle = "Statistically significant clusters, p < 0.05") +
  theme_minimal() +
  theme(axis.text = element_blank(), axis.ticks = element_blank())
```

### Environmental Health Visualization

| Visualization | Data Source | Method |
|--------------|------------|--------|
| **Exposure surface** | Monitoring stations, dispersion models | IDW/kriging interpolation of pollutant concentrations |
| **Dose-response curve** | Epidemiological studies | Logistic or Cox regression with confidence bands |
| **Health risk map** | Exposure x vulnerability | Raster algebra combining hazard, exposure, sensitivity |
| **Buffer analysis** | Point sources (factories, highways) | Distance-based exposure zones with health outcome overlay |
| **Environmental justice map** | Demographics + pollution | Bivariate choropleth (race/income vs. exposure) |

#### Bivariate Choropleth for Environmental Justice (R)

```r
library(biscale)
library(ggplot2)
library(cowplot)
library(sf)

tracts <- st_read("census_tracts.gpkg")

# Create bivariate classes
bi_data <- bi_class(tracts,
                    x = pct_minority,
                    y = pollution_index,
                    style = "quantile", dim = 3)

map <- ggplot() +
  geom_sf(data = bi_data, aes(fill = bi_class),
          color = "white", size = 0.1, show.legend = FALSE) +
  bi_scale_fill(pal = "DkViolet", dim = 3) +
  labs(title = "Environmental Justice: Minority Population vs. Pollution") +
  theme_void()

legend <- bi_legend(pal = "DkViolet", dim = 3,
                    xlab = "% Minority (higher)",
                    ylab = "Pollution (higher)",
                    size = 8)

final <- ggdraw() +
  draw_plot(map, 0, 0, 1, 1) +
  draw_plot(legend, 0.7, 0.1, 0.25, 0.25)

ggsave("env_justice_bivariate.png", final, width = 12, height = 10, dpi = 300)
```

### Genomic Geography

| Visualization | Purpose | Tool |
|--------------|---------|------|
| **Allele frequency map** | Geographic distribution of genetic variants | R (poppr, adegenet) + ggplot2 |
| **Phylogeographic tree on map** | Overlay phylogeny on geography | R (phytools), Python (toytree) |
| **Isolation-by-distance plot** | Genetic distance vs. geographic distance | Mantel test visualization |
| **Admixture map** | Pie charts of ancestry proportions on map | R (pophelper, mapmixture) |
| **Landscape genetics resistance** | Gene flow barriers overlaid on terrain | ResistanceGA, Circuitscape |

### Medical Facility Access

| Analysis | Method | Output |
|----------|--------|--------|
| **Isochrone maps** | Network-based travel time polygons | Concentric travel-time zones from facilities |
| **Two-step floating catchment** | Supply/demand ratio within travel threshold | Access score per demand location |
| **Voronoi service areas** | Nearest facility assignment | Polygonal service territories |
| **Gravity model** | Distance-decay weighted access | Continuous accessibility surface |

#### Isochrone Access Map (Python, osmnx + networkx)

```python
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Download road network
G = ox.graph_from_place("Portland, Oregon", network_type="drive")
G = ox.speed.add_edge_speeds(G)
G = ox.speed.add_edge_travel_times(G)

# Hospital location
hospital = (45.5122, -122.6587)
center_node = ox.nearest_nodes(G, hospital[1], hospital[0])

# Calculate isochrones at 5, 10, 15, 20 minutes
trip_times = [5, 10, 15, 20]
colors = ["#d73027", "#fc8d59", "#fee08b", "#d9ef8b"]

fig, ax = plt.subplots(figsize=(12, 12))

for tt, color in zip(sorted(trip_times, reverse=True), colors):
    subgraph = nx.ego_graph(G, center_node, radius=tt * 60,
                            distance="travel_time")
    nodes = ox.graph_to_gdfs(subgraph, edges=False)
    hull = nodes.unary_union.convex_hull
    gpd.GeoSeries([hull]).plot(ax=ax, color=color, alpha=0.5,
                               label=f"{tt} min")

ax.plot(hospital[1], hospital[0], "r+", markersize=15, markeredgewidth=3)
ax.legend(title="Travel Time")
ax.set_title("Hospital Accessibility Isochrones")
plt.tight_layout()
plt.savefig("isochrone_access.png", dpi=300)
```

### SOTA Tools for Biological & Medical Visualization

| Tool | Language | Specialization |
|------|----------|---------------|
| [spatstat](https://spatstat.org/) | R | Spatial point pattern analysis |
| [SaTScan](https://www.satscan.org/) | Standalone | Space-time cluster detection |
| [GeoDa](https://geodacenter.github.io/) | Standalone | Exploratory spatial data analysis, LISA |
| [EpiMap](https://epimap.org/) | Web | Quick epidemiological mapping |
| [rsatscan](https://cran.r-project.org/package=rsatscan) | R | R interface to SaTScan |
| [surveillance](https://cran.r-project.org/package=surveillance) | R | Outbreak detection algorithms |

> **Cross-reference:** [Healthcare & Public Health](../data-sources/healthcare-public-health.md)

---

## 4. Social Science Visualization

Social science data is messy, politically sensitive, and often aggregated to administrative units that introduce the modifiable areal unit problem (MAUP). Visualization must honestly represent uncertainty and avoid misleading readers.

### Demographic Visualization

| Visualization | What It Shows | Dark Art Tip |
|--------------|---------------|-------------|
| **Population pyramid** | Age-sex structure | Compare two pyramids side-by-side (e.g., 2000 vs. 2025) to show demographic transition |
| **Migration flow map** | Origin-destination movement | Use logarithmic line width; raw counts create visual chaos |
| **Urbanization curve** | Urban population share over time | Add vertical lines for policy events (land reform, industrialization) |
| **Dot density map** | Population distribution at sub-unit level | One dot = N people; randomize within polygon to avoid centroids |
| **Dasymetric map** | Population redistributed to habitable land | Ancillary data (land cover, building footprints) improves choropleth |

#### Population Pyramid (Python, matplotlib)

```python
import matplotlib.pyplot as plt
import pandas as pd

pop = pd.read_csv("population_by_age_sex.csv")
age_groups = pop["age_group"].unique()

fig, ax = plt.subplots(figsize=(10, 8))

ax.barh(age_groups, -pop[pop["sex"] == "Male"]["population"],
        color="#4393c3", label="Male", height=0.8)
ax.barh(age_groups, pop[pop["sex"] == "Female"]["population"],
        color="#d6604d", label="Female", height=0.8)

max_val = pop["population"].max()
ax.set_xlim(-max_val * 1.1, max_val * 1.1)
ax.set_xlabel("Population")
ax.set_title("Population Pyramid (2025)")
ax.legend(loc="upper right")

# Format x-axis to show absolute values
ticks = ax.get_xticks()
ax.set_xticklabels([f"{abs(int(t)):,}" for t in ticks])

plt.tight_layout()
plt.savefig("population_pyramid.png", dpi=300)
```

### Inequality Mapping

| Visualization | Metric | Interpretation |
|--------------|--------|----------------|
| **Gini coefficient map** | Choropleth of Gini by region | 0 = perfect equality, 1 = perfect inequality |
| **Lorenz curve** | Cumulative share of income vs. population | Distance from 45-degree line = inequality |
| **Deprivation index map** | Composite index (education, income, health) | Identify areas needing intervention |
| **Bivariate map** | Two variables simultaneously | Reveal spatial coincidence of disadvantages |
| **Dot density by income** | Dots colored by income bracket | Visualize segregation without aggregation bias |

#### Lorenz Curve with Gini (Python)

```python
import numpy as np
import matplotlib.pyplot as plt

def lorenz_curve(incomes):
    sorted_inc = np.sort(incomes)
    cumulative = np.cumsum(sorted_inc) / np.sum(sorted_inc)
    cumulative = np.insert(cumulative, 0, 0)
    population = np.linspace(0, 1, len(cumulative))
    gini = 1 - 2 * np.trapz(cumulative, population)
    return population, cumulative, gini

fig, ax = plt.subplots(figsize=(8, 8))

for region, data in regions.items():
    pop, cum, gini = lorenz_curve(data["income"])
    ax.plot(pop, cum, label=f"{region} (Gini={gini:.3f})", linewidth=2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect equality")
ax.set_xlabel("Cumulative Share of Population")
ax.set_ylabel("Cumulative Share of Income")
ax.set_title("Lorenz Curves by Region")
ax.legend()
ax.set_aspect("equal")
plt.tight_layout()
plt.savefig("lorenz_curves.png", dpi=300)
```

### Survey Data Visualization

| Plot Type | Use Case | Key Consideration |
|-----------|----------|-------------------|
| **Likert scale plot** | Summarize agree/disagree responses | Center on neutral; use diverging colors |
| **Confidence interval map** | Map estimates with uncertainty | Width of CI matters more than point estimate in small samples |
| **Margin of error visualization** | Show reliability of survey estimates | Transparent fill or hatching for high-MOE areas |
| **Small area estimation map** | Modeled estimates for sub-survey geographies | Clearly label as "modeled" not "observed" |

#### Likert Scale Plot (R, ggplot2 + likert)

```r
library(likert)
library(ggplot2)

survey <- read.csv("survey_responses.csv")
survey_likert <- likert(survey[, 3:8])  # columns with Likert responses

plot(survey_likert, ordered = TRUE, wrap = 40) +
  labs(title = "Community Attitudes Toward Development",
       subtitle = "n = 1,247 respondents across 5 districts") +
  theme(text = element_text(size = 12))
```

### Electoral and Political Visualization

| Visualization | Description | Pitfall to Avoid |
|--------------|-------------|------------------|
| **Swing map** | Change in vote share between elections | Land area bias; rural areas dominate visually |
| **Cartogram** | Distort geometry by a variable (population) | Disorienting; pair with a reference map |
| **Proportional symbol** | Circles sized by votes on a base map | Overlapping symbols in dense areas |
| **Hexagonal cartogram** | Equal-area hex per unit (e.g., constituency) | Loses geographic relationships |
| **Vote share waffle chart** | Grid of squares showing proportions | Better than pie charts for part-to-whole |

### Historical GIS Visualization

| Technique | Tool | Example |
|-----------|------|---------|
| **Time-slider map** | QGIS Temporal Controller, Kepler.gl | Boundary changes over centuries |
| **Animated boundary morph** | D3.js transitions, Mapshaper | Show administrative reorganization |
| **Georeferenced overlay** | QGIS Sketcher, Leaflet side-by-side | Historical map atop modern basemap |
| **Timeline-linked map** | TimelineJS + Leaflet | Narrative with synchronized map views |

### SOTA Tools for Social Science Visualization

| Tool | Strength |
|------|----------|
| [D3.js](https://d3js.org) | Maximum control over custom visualizations |
| [Observable](https://observablehq.com) | Reactive notebooks with D3, ideal for exploratory work |
| [R ggplot2 + sf](https://ggplot2.tidyverse.org) | Publication-quality static maps and charts in one pipeline |
| [QGIS Temporal Controller](https://qgis.org) | Animate temporal data without code |
| [Flourish](https://flourish.studio) | No-code animated maps and charts for journalism |

> **Cross-reference:** [Urban Planning & Smart Cities](../data-sources/urban-planning-smart-cities.md) | [Administrative Boundaries](../data-sources/administrative-boundaries.md)

---

## 5. Climate & Environmental Visualization

Climate and environmental data often comes as massive raster time series, multi-model ensembles, or real-time sensor streams. The visualizations below are designed to communicate complex atmospheric and environmental phenomena to both scientists and the public.

### Climate Stripes (Ed Hawkins Warming Stripes)

The warming stripes visualization -- created by Ed Hawkins at the University of Reading -- strips away axes, labels, and gridlines to show temperature anomaly as a sequence of colored bars. It has become one of the most iconic scientific visualizations of the 21st century.

#### Climate Stripes (Python, matplotlib)

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np

temps = pd.read_csv("global_temperature_anomaly.csv")

# Ed Hawkins color palette
cmap = mcolors.LinearSegmentedColormap.from_list(
    "hawkins",
    ["#08306b", "#2171b5", "#6baed6", "#c6dbef",
     "#fee0d2", "#fc9272", "#de2d26", "#67000d"]
)

norm = mcolors.Normalize(vmin=temps["anomaly"].min(),
                         vmax=temps["anomaly"].max())

fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(temps["year"], 1, width=1.0,
       color=[cmap(norm(a)) for a in temps["anomaly"]])
ax.set_xlim(temps["year"].min() - 0.5, temps["year"].max() + 0.5)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor("white")
plt.tight_layout()
plt.savefig("climate_stripes.png", dpi=300, bbox_inches="tight",
            facecolor="white")
```

### Wind Flow Visualization

| Technique | Description | Tool |
|-----------|-------------|------|
| **Streamlines** | Lines tangent to wind field at each point | matplotlib streamplot, D3 |
| **Particle animation** | Animated particles advected by wind field | earth.nullschool.net approach, deck.gl WindLayer |
| **Wind barbs** | Standard meteorological notation | matplotlib barbs, MetPy |
| **Wind rose** | Directional frequency/speed histogram | R (openair), Python (windrose) |
| **Divergence/convergence** | Scalar field derived from vector field | ParaView, matplotlib contourf |

#### Wind Rose (Python, windrose)

```python
from windrose import WindroseAxes
import matplotlib.pyplot as plt
import pandas as pd

wind = pd.read_csv("weather_station.csv")

ax = WindroseAxes.from_ax()
ax.bar(wind["direction"], wind["speed"],
       normed=True, opening=0.8,
       edgecolor="white", linewidth=0.5,
       cmap=plt.cm.viridis)
ax.set_legend(title="Wind Speed (m/s)")
plt.title("Wind Rose: Station Alpha (2024)")
plt.savefig("wind_rose.png", dpi=300, bbox_inches="tight")
```

#### Animated Particle Wind (JavaScript, deck.gl)

```javascript
import { Deck } from "@deck.gl/core";
import { WindLayer } from "deck.gl-particle";

const deckgl = new Deck({
  initialViewState: {
    longitude: 0,
    latitude: 30,
    zoom: 2,
  },
  layers: [
    new WindLayer({
      id: "wind",
      image: "wind_data.png",       // R/G encoded u/v components
      bounds: [-180, -90, 180, 90],
      numParticles: 4096,
      maxAge: 60,
      speedFactor: 2.0,
      color: [255, 255, 255],
      width: 1.5,
      opacity: 0.8,
    }),
  ],
});
```

### Sea Level Rise Visualization

| Approach | Method | Audience |
|----------|--------|----------|
| **Bathtub model** | DEM < threshold = inundated | Quick screening, public communication |
| **Scenario comparison** | Side-by-side or slider for RCP/SSP scenarios | Policy makers |
| **Probabilistic inundation** | Monte Carlo over DEM uncertainty + SLR range | Engineers, planners |
| **Time animation** | Decade-by-decade inundation expansion | Public engagement |
| **3D perspective** | Flood depth draped on 3D terrain | Stakeholder meetings |

#### Sea Level Rise Bathtub Model (Python, rasterio)

```python
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

with rasterio.open("coastal_dem.tif") as src:
    dem = src.read(1)
    extent = [src.bounds.left, src.bounds.right,
              src.bounds.bottom, src.bounds.top]

scenarios = [0.5, 1.0, 2.0]  # meters of sea level rise
colors = ["#ffffb2", "#fd8d3c", "#bd0026"]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, slr, color in zip(axes, scenarios, colors):
    inundated = np.where((dem > 0) & (dem <= slr), 1, 0)
    ax.imshow(dem, extent=extent, cmap="terrain", vmin=-5, vmax=50)
    ax.imshow(np.ma.masked_where(inundated == 0, inundated),
              extent=extent, cmap=ListedColormap([color]),
              alpha=0.6)
    ax.set_title(f"+{slr}m Sea Level Rise")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

plt.suptitle("Coastal Inundation Scenarios", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("sea_level_rise_scenarios.png", dpi=300)
```

### Air Quality Visualization

| Visualization | Data Source | Method |
|--------------|-------------|--------|
| **Interpolated pollution surface** | Monitoring stations | IDW, kriging, or land-use regression |
| **AQI dashboard** | Real-time sensor API | Streamlit, Grafana, Observable |
| **Concentration time series** | Hourly sensor readings | Line chart with EPA threshold bands |
| **Source apportionment** | Chemical speciation data | Stacked bar or pie charts by source category |
| **Plume dispersion** | Gaussian plume model output | Contour overlay on basemap |

#### AQI Time Series with Threshold Bands (Python)

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

aqi = pd.read_csv("aqi_hourly.csv", parse_dates=["timestamp"])

fig, ax = plt.subplots(figsize=(14, 5))

# EPA AQI category bands
bands = [
    (0, 50, "#00e400", "Good"),
    (50, 100, "#ffff00", "Moderate"),
    (100, 150, "#ff7e00", "Unhealthy (Sensitive)"),
    (150, 200, "#ff0000", "Unhealthy"),
    (200, 300, "#8f3f97", "Very Unhealthy"),
    (300, 500, "#7e0023", "Hazardous"),
]

for lo, hi, color, label in bands:
    ax.axhspan(lo, hi, color=color, alpha=0.15)

ax.plot(aqi["timestamp"], aqi["pm25_aqi"], color="black",
        linewidth=0.8, alpha=0.8)
ax.set_ylabel("PM2.5 AQI")
ax.set_xlabel("Date")
ax.set_title("Hourly PM2.5 AQI with EPA Category Bands")
ax.set_ylim(0, min(500, aqi["pm25_aqi"].max() * 1.1))

handles = [mpatches.Patch(color=c, alpha=0.3, label=l)
           for _, _, c, l in bands if l != "Hazardous"]
ax.legend(handles=handles, loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("aqi_timeseries.png", dpi=300)
```

### Wildfire Visualization

| Visualization | Source Data | Tool |
|--------------|-------------|------|
| **Active fire map** | FIRMS/VIIRS hotspots | Kepler.gl, QGIS, deck.gl ScatterplotLayer |
| **Fire perimeter progression** | NIFC/MTBS perimeters | Time-slider map, D3 transition |
| **Burn severity** | dNBR from Landsat/Sentinel | Classified raster (low/moderate/high) |
| **Smoke dispersion** | HRRR-Smoke, BlueSky | Animated contour or particle layer |
| **Fire weather dashboard** | RAWS stations, gridded forecasts | Streamlit with wind rose + fire danger gauge |
| **Evacuation zone map** | Fire perimeter + road network | Network analysis, isochrone overlay |

#### Fire Perimeter Progression (Python, geopandas)

```python
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

perimeters = gpd.read_file("fire_perimeters_daily.gpkg")
perimeters["day_num"] = (perimeters["date"] - perimeters["date"].min()).dt.days

fig, ax = plt.subplots(figsize=(12, 10))

norm = Normalize(vmin=0, vmax=perimeters["day_num"].max())
cmap = plt.cm.YlOrRd

for _, row in perimeters.iterrows():
    gpd.GeoSeries([row.geometry]).plot(
        ax=ax, color=cmap(norm(row["day_num"])),
        edgecolor="black", linewidth=0.5, alpha=0.6
    )

sm = ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Days Since Ignition", shrink=0.7)
ax.set_title("Fire Perimeter Progression")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("fire_progression.png", dpi=300)
```

### SOTA Tools for Climate & Environmental Visualization

| Tool | Type | Best For |
|------|------|----------|
| [earth.nullschool.net](https://earth.nullschool.net) | Web | Global wind, ocean current, chemistry animation |
| [Windy.com](https://www.windy.com) | Web | Multi-layer weather visualization |
| [ParaView](https://www.paraview.org/) | Desktop | 3D volumetric scientific visualization |
| [deck.gl WindLayer](https://deck.gl) | JS | Custom particle-based wind animation |
| [MetPy](https://unidata.github.io/MetPy/) | Python | Meteorological calculations + plotting |
| [openair](https://davidcarslaw.github.io/openair/) | R | Air quality data analysis and visualization |
| [xarray](https://xarray.pydata.org/) + matplotlib | Python | NetCDF climate model output |

> **Cross-reference:** [Climate & Weather](../data-sources/climate-weather.md) | [Natural Disasters & Emergency Response](../data-sources/natural-disasters-emergency-response.md)

---

## 6. Multi-dimensional Data Visualization

When your data has more than three dimensions -- and spatial data almost always does -- you need visualization techniques that go beyond the x-y-color paradigm.

### Parallel Coordinates

Parallel coordinates plot each variable on a separate vertical axis, connecting each observation's values with a polyline. They are especially useful for exploring multivariate spatial data (e.g., comparing census tracts across 10+ socioeconomic indicators).

#### Parallel Coordinates (Python, Plotly)

```python
import plotly.express as px
import pandas as pd

tracts = pd.read_csv("census_tracts_indicators.csv")

fig = px.parallel_coordinates(
    tracts,
    dimensions=["median_income", "pct_college", "unemployment",
                "commute_time", "housing_cost", "green_space_pct"],
    color="cluster_label",
    color_continuous_scale=px.colors.diverging.Tealrose,
    labels={
        "median_income": "Income ($)",
        "pct_college": "College (%)",
        "unemployment": "Unemployment (%)",
        "commute_time": "Commute (min)",
        "housing_cost": "Housing ($)",
        "green_space_pct": "Green Space (%)",
    },
    title="Census Tract Profiles by Cluster"
)
fig.update_layout(width=1000, height=500)
fig.write_html("parallel_coordinates.html")
```

### Radar / Spider Charts

Radar charts compare multiple indicators across a small number of entities (e.g., comparing sustainability scores across 5 cities on 8 dimensions).

| When to Use | When NOT to Use |
|-------------|-----------------|
| 3--8 variables, 2--5 entities | More than 8 axes (becomes unreadable) |
| Variables on similar scales | Variables with very different units/scales |
| Showing "profile shape" matters | Precise value comparison needed |

#### Radar Chart (Python, matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

categories = ["Air Quality", "Water", "Green Space", "Transit",
              "Energy Efficiency", "Waste Management"]
N = len(categories)

cities = {
    "Portland": [0.85, 0.78, 0.92, 0.71, 0.80, 0.75],
    "Phoenix":  [0.45, 0.40, 0.35, 0.50, 0.60, 0.55],
    "Denver":   [0.70, 0.65, 0.75, 0.60, 0.72, 0.68],
}

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for city, values in cities.items():
    values += values[:1]
    ax.plot(angles, values, linewidth=2, label=city)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 1)
ax.set_title("City Sustainability Profiles", pad=20, fontsize=14)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("radar_sustainability.png", dpi=300, bbox_inches="tight")
```

### Treemaps and Sunbursts

Treemaps encode hierarchical data as nested rectangles, where area represents quantity. Sunbursts use concentric rings instead. Both are excellent for visualizing hierarchical land use classification, budget allocation, or administrative breakdowns.

| Variant | Best For | Tool |
|---------|----------|------|
| **Treemap** | Flat or 2-level hierarchy, comparing sizes | Plotly, D3, ECharts, Observable Plot |
| **Sunburst** | Deep hierarchy (3+ levels), drilling down | Plotly, D3, ECharts |
| **Icicle** | Same as sunburst but in rectangular form | Plotly, D3 |
| **Circle packing** | Aesthetic hierarchy, less precise than treemap | D3, Observable |

#### Treemap of Land Use (Python, Plotly)

```python
import plotly.express as px
import pandas as pd

land = pd.DataFrame({
    "category": ["Urban", "Urban", "Urban",
                  "Agriculture", "Agriculture",
                  "Forest", "Forest", "Water"],
    "subcategory": ["Residential", "Commercial", "Industrial",
                    "Cropland", "Pasture",
                    "Deciduous", "Conifer", "Lakes"],
    "area_km2": [450, 120, 85, 1200, 800, 2100, 1400, 350],
})

fig = px.treemap(
    land, path=["category", "subcategory"], values="area_km2",
    color="area_km2",
    color_continuous_scale="Greens",
    title="Land Use Composition (km2)"
)
fig.update_layout(width=800, height=600)
fig.write_html("land_use_treemap.html")
```

### Sankey Diagrams

Sankey diagrams show flows between nodes. They are indispensable for visualizing:

- **Land use change**: agricultural to urban conversion flows between two time periods
- **Material flows**: resource extraction to consumption to waste
- **Energy flows**: source to conversion to end use
- **Migration flows**: origin regions to destination regions
- **Water budgets**: precipitation to runoff, infiltration, evapotranspiration

#### Land Use Change Sankey (Python, Plotly)

```python
import plotly.graph_objects as go

labels = ["Forest 2000", "Agriculture 2000", "Urban 2000",
          "Forest 2020", "Agriculture 2020", "Urban 2020"]

fig = go.Figure(go.Sankey(
    node=dict(
        pad=15, thickness=20,
        label=labels,
        color=["#2d6a4f", "#a7c957", "#d62828",
               "#2d6a4f", "#a7c957", "#d62828"]
    ),
    link=dict(
        source=[0, 0, 0, 1, 1, 1, 2, 2, 2],
        target=[3, 4, 5, 3, 4, 5, 3, 4, 5],
        value= [850, 200, 50, 30, 680, 90, 0, 5, 295],
        color=["rgba(45,106,79,0.3)", "rgba(167,201,87,0.3)",
               "rgba(214,40,40,0.3)", "rgba(45,106,79,0.3)",
               "rgba(167,201,87,0.3)", "rgba(214,40,40,0.3)",
               "rgba(45,106,79,0.3)", "rgba(167,201,87,0.3)",
               "rgba(214,40,40,0.3)"]
    )
))
fig.update_layout(title="Land Use Change 2000-2020 (km2)", width=900, height=500)
fig.write_html("land_use_sankey.html")
```

### Alluvial Diagrams

Alluvial diagrams are a special case of Sankey: they show how categorical memberships change over time. For example, tracking how census tracts transition between deprivation quintiles across decades.

#### Alluvial Diagram (R, ggalluvial)

```r
library(ggplot2)
library(ggalluvial)

transitions <- read.csv("tract_deprivation_transitions.csv")

ggplot(transitions,
       aes(x = decade, stratum = quintile, alluvium = tract_id,
           fill = quintile, label = quintile)) +
  geom_flow(stat = "alluvium", alpha = 0.5) +
  geom_stratum(width = 0.3) +
  geom_text(stat = "stratum", size = 3) +
  scale_fill_brewer(type = "div", palette = "RdYlGn", direction = -1) +
  labs(title = "Deprivation Quintile Transitions by Census Tract",
       x = "Decade", y = "Number of Tracts") +
  theme_minimal()
```

### SOTA Tools for Multi-dimensional Visualization

| Tool | Language | Best For |
|------|----------|----------|
| [D3.js](https://d3js.org) | JS | Maximum flexibility; any custom diagram |
| [ECharts](https://echarts.apache.org/) | JS | Rich chart gallery with good defaults |
| [Plotly](https://plotly.com/) | Python/R/JS | Interactive parallel coords, Sankey, treemap |
| [Observable Plot](https://observablehq.com/plot/) | JS | Faceted marks with concise API |
| [Nivo](https://nivo.rocks/) | React | Beautiful React chart components (Sankey, sunburst, radar) |
| [RAWGraphs](https://rawgraphs.io/) | Web | No-code; paste data, get alluvial/treemap/bumpchart |

---

## 7. Interactive Exploration Tools

Static plots answer pre-defined questions. Interactive exploration lets users discover their own questions. The tools below create linked, filterable, and drillable interfaces that connect charts to maps.

### Crossfilter (Linked Brushing)

Crossfilter enables selecting a range on one chart and instantly filtering all other charts. This is the foundation of interactive geospatial dashboards.

| Implementation | Stack | Complexity |
|---------------|-------|------------|
| [dc.js](https://dc-js.github.io/dc.js/) | D3 + Crossfilter.js | Medium -- requires JS, great for dashboards |
| [Vega-Lite selections](https://vega.github.io/vega-lite/) | Vega-Lite JSON | Low -- declarative, linked views in JSON |
| [Observable Plot](https://observablehq.com/plot/) | Observable runtime | Low -- reactive by design |
| [Altair selections](https://altair-viz.github.io/) | Python (Vega-Lite) | Low -- Pythonic interactive grammar |
| [Plotly Dash](https://dash.plotly.com/) | Python | Medium -- callbacks for linked filtering |

#### Linked Selection (Vega-Lite JSON, conceptual)

```json
{
  "hconcat": [
    {
      "mark": "point",
      "encoding": {
        "x": {"field": "income", "type": "quantitative"},
        "y": {"field": "life_expectancy", "type": "quantitative"},
        "color": {
          "condition": {"param": "brush", "field": "region", "type": "nominal"},
          "value": "lightgray"
        }
      },
      "params": [{"name": "brush", "select": "interval"}]
    },
    {
      "mark": "geoshape",
      "encoding": {
        "color": {
          "condition": {"param": "brush", "field": "region", "type": "nominal"},
          "value": "lightgray"
        }
      }
    }
  ]
}
```

### Faceting (Small Multiples)

Small multiples repeat the same chart structure for each category, sharing axes so comparisons are easy. Edward Tufte calls this "the best design solution for a wide range of problems."

| Tool | Faceting Syntax | Max Practical Facets |
|------|----------------|---------------------|
| ggplot2 | `facet_wrap(~var)`, `facet_grid(row~col)` | 20--30 |
| Observable Plot | `fx`, `fy` channels | 15--20 |
| Vega-Lite | `facet` encoding | 15--20 |
| Altair | `.facet()` method | 15--20 |
| seaborn | `FacetGrid`, `catplot` | 15--20 |

#### Small Multiples Map (R, ggplot2 + sf)

```r
library(ggplot2)
library(sf)

counties <- st_read("counties_with_data.gpkg")

ggplot(counties) +
  geom_sf(aes(fill = unemployment_rate), color = "white", size = 0.1) +
  scale_fill_viridis_c(option = "inferno", name = "Unemployment (%)") +
  facet_wrap(~year, ncol = 4) +
  labs(title = "Unemployment Rate by County (2016-2025)") +
  theme_void() +
  theme(
    strip.text = element_text(size = 11, face = "bold"),
    legend.position = "bottom"
  )
```

### Drill-down Interaction

Drill-down allows clicking a region on a map to reveal a detailed view -- a chart, a sub-map, or a data table. This pattern is standard in dashboards.

| Pattern | Implementation | Use Case |
|---------|---------------|----------|
| **Map -> Chart** | Click state -> show county bar chart | Regional comparison |
| **Map -> Sub-map** | Click country -> zoom to provinces | Hierarchical geography |
| **Map -> Table** | Click polygon -> show attribute table | Data inspection |
| **Chart -> Map** | Click bar -> highlight region on map | Reverse drill-down |

#### Drill-down Dashboard (Python, Plotly Dash, conceptual structure)

```python
from dash import Dash, dcc, html, Input, Output, callback
import plotly.express as px
import geopandas as gpd
import pandas as pd

app = Dash(__name__)

states = gpd.read_file("states.geojson")
county_data = pd.read_csv("county_data.csv")

app.layout = html.Div([
    dcc.Graph(id="state-map"),
    dcc.Graph(id="county-detail"),
])

@callback(
    Output("county-detail", "figure"),
    Input("state-map", "clickData")
)
def drill_down(click_data):
    if click_data is None:
        return px.bar(title="Click a state to see county details")
    state_fips = click_data["points"][0]["location"]
    filtered = county_data[county_data["state_fips"] == state_fips]
    return px.bar(filtered, x="county_name", y="population",
                  title=f"Counties in State {state_fips}")

if __name__ == "__main__":
    app.run(debug=True)
```

### Brushing and Linking

Brushing (selecting a subset on one view) and linking (highlighting that subset on all other views) is the most powerful interactive analysis pattern.

| Tool | Brushing Support | Linking Mechanism |
|------|-----------------|-------------------|
| dc.js | Range brush on any dimension | Crossfilter library handles joins |
| Vega-Lite | `interval` and `point` selections | `condition` encoding propagates selection |
| Kepler.gl | Time filter, spatial filter | Built-in linked time + space |
| Observable Plot | Reactive inputs | Observable runtime reactivity |
| Plotly Dash | Range slider, lasso select | Callback functions |

> **Dark Art:** The most effective linked view combines a scatterplot (for statistical relationships), a map (for spatial patterns), and a histogram (for distribution context). Brushing on any one view should filter the other two. This "triad" pattern instantly reveals whether statistical outliers are also spatial outliers.

### SOTA Tools for Interactive Exploration

| Tool | Learning Curve | Geo Support | Deployment |
|------|---------------|-------------|------------|
| [dc.js](https://dc-js.github.io/dc.js/) | Moderate | Via Leaflet plugin | Static HTML |
| [Vega-Lite](https://vega.github.io/vega-lite/) | Low | GeoShape mark | Embed anywhere |
| [Observable](https://observablehq.com) | Low | Full D3 geo | Observable cloud or self-host |
| [Kepler.gl](https://kepler.gl) | Very Low | Native | React app or hosted |
| [Plotly Dash](https://dash.plotly.com/) | Moderate | Via Mapbox/MapLibre | Python server |
| [Streamlit](https://streamlit.io/) | Very Low | Via pydeck, folium | Python server |
| [Panel](https://panel.holoviz.org/) | Moderate | Via hvPlot, GeoViews | Python server |

> **Cross-reference:** [Charting Integration](../js-bindbox/charting-integration.md)

---

## 8. Publication-Quality Output

Academic journals, government reports, and NGO publications demand high-quality figures with precise control over typography, color, and layout. This section covers the tools and workflows that produce camera-ready output.

### Python Stack

| Tool | Role | Strength |
|------|------|----------|
| [matplotlib](https://matplotlib.org/) + [cartopy](https://scitools.org.uk/cartopy/) | Base plotting + geographic projections | The scientific standard; total control |
| [seaborn](https://seaborn.pydata.org/) | Statistical plots on top of matplotlib | Beautiful defaults, concise API |
| [plotnine](https://plotnine.org/) | ggplot2 grammar for Python | Familiar to R users; clean syntax |
| [matplotlib + pgf backend](https://matplotlib.org/stable/users/explain/text/pgf.html) | LaTeX-native output | Matches document fonts exactly |

#### Publication Figure (Python, matplotlib + cartopy)

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np

fig, ax = plt.subplots(
    figsize=(7, 5),  # journal column width
    subplot_kw={"projection": ccrs.PlateCarree()}
)

ax.set_extent([-130, -65, 24, 50])
ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="gray")
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.8, linestyle="--")

# Plot data
scatter = ax.scatter(
    lons, lats, c=values, s=30,
    cmap="RdYlBu_r", transform=ccrs.PlateCarree(),
    edgecolors="black", linewidths=0.3, zorder=5
)

cbar = plt.colorbar(scatter, ax=ax, orientation="horizontal",
                     pad=0.05, shrink=0.7, label="Temperature Anomaly (K)")

ax.set_title("Surface Temperature Anomaly (2025)", fontsize=11, pad=10)

plt.savefig("fig01_temp_anomaly.pdf", dpi=300, bbox_inches="tight")
plt.savefig("fig01_temp_anomaly.png", dpi=300, bbox_inches="tight")
```

### R Stack (The Gold Standard)

| Tool | Role | Why It Dominates |
|------|------|-----------------|
| [ggplot2](https://ggplot2.tidyverse.org/) | Grammar of graphics | Layered, declarative, endlessly extensible |
| [sf](https://r-spatial.github.io/sf/) | Spatial data handling | Integrates seamlessly with ggplot2 via `geom_sf()` |
| [tmap](https://r-tmap.github.io/tmap/) | Thematic mapping | Switch between static and interactive with one argument |
| [patchwork](https://patchwork.data-imaginist.com/) | Multi-panel layouts | Combine ggplots with `+` and `/` operators |
| [ggspatial](https://paleolimbot.github.io/ggspatial/) | Map furniture | North arrow, scale bar, basemap tiles |

#### Multi-panel Publication Figure (R)

```r
library(ggplot2)
library(sf)
library(patchwork)
library(ggspatial)

counties <- st_read("counties.gpkg")

p1 <- ggplot(counties) +
  geom_sf(aes(fill = income_2020), color = "white", size = 0.1) +
  scale_fill_viridis_c(option = "plasma", name = "Median\nIncome ($)") +
  annotation_scale(location = "bl") +
  annotation_north_arrow(location = "tr", style = north_arrow_minimal()) +
  labs(title = "(a) Median Household Income, 2020") +
  theme_void()

p2 <- ggplot(counties, aes(x = income_2020, y = life_expectancy)) +
  geom_point(alpha = 0.4, size = 1) +
  geom_smooth(method = "lm", color = "firebrick") +
  labs(title = "(b) Income vs. Life Expectancy",
       x = "Median Income ($)", y = "Life Expectancy (years)") +
  theme_minimal()

p3 <- ggplot(counties, aes(x = income_2020)) +
  geom_histogram(bins = 40, fill = "steelblue", color = "white") +
  labs(title = "(c) Income Distribution",
       x = "Median Income ($)", y = "Count") +
  theme_minimal()

combined <- p1 / (p2 | p3) +
  plot_annotation(
    title = "Figure 1. Spatial and Statistical Distribution of Income",
    theme = theme(plot.title = element_text(size = 13, face = "bold"))
  )

ggsave("figure_01.pdf", combined, width = 10, height = 12, dpi = 300)
ggsave("figure_01.png", combined, width = 10, height = 12, dpi = 300)
```

### LaTeX Integration

| Method | How It Works | When to Use |
|--------|-------------|-------------|
| **pgf backend (matplotlib)** | Render text with LaTeX engine | When figure text must match document font |
| **TikZ / pgfplots** | Write plots directly in LaTeX | Simple plots with perfect integration |
| **R tikzDevice** | ggplot2 output as TikZ code | R workflow with LaTeX documents |
| **SVG include** | Export SVG, include in LaTeX | When PDF export has issues |
| **Quarto** | Render R/Python chunks to PDF | Full reproducible document |

#### matplotlib with pgf Backend

```python
import matplotlib
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False,
    "font.size": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(3.5, 2.5))  # single column width
ax.plot(x, y, label=r"$R^2 = 0.87$")
ax.set_xlabel(r"Distance to CBD (km)")
ax.set_ylabel(r"Log Price (\$/m$^2$)")
ax.legend()
plt.savefig("figure_regression.pgf", bbox_inches="tight")
```

### Export Format Guide

| Format | Use Case | DPI | Notes |
|--------|----------|-----|-------|
| **PDF** | Journal submission, print | Vector | Preferred for line art, maps |
| **PNG** | Web, presentations | 300 | Use for raster-heavy maps (satellite imagery) |
| **SVG** | Web, editable graphics | Vector | Editable in Inkscape/Illustrator |
| **EPS** | Legacy journal requirements | Vector | Being replaced by PDF |
| **TIFF** | Some journals require it | 300--600 | Lossless; large files |
| **PGF** | LaTeX integration | Vector | Renders text with document engine |

### Reproducibility Pipelines

| Tool | Language | Output Formats | Key Feature |
|------|----------|---------------|-------------|
| [Quarto](https://quarto.org/) | R/Python/Julia | PDF, HTML, Word, slides | Multi-language, cross-format |
| [R Markdown](https://rmarkdown.rstudio.com/) | R | PDF, HTML, Word | Mature ecosystem, knitr engine |
| [Jupyter](https://jupyter.org/) | Python/R/Julia | HTML, PDF (via nbconvert) | Interactive, widely used |
| [Observable](https://observablehq.com) | JS | HTML | Reactive, shareable |
| [Typst](https://typst.app/) | Typst | PDF | Modern LaTeX alternative, fast compilation |

> **Dark Art:** For maximum reproducibility, pin every package version. In R, use `renv::snapshot()`. In Python, use `pip freeze > requirements.txt` or `conda env export`. Include the lockfile in your repository. Future you will be grateful.

> **Cross-reference:** [Academic Writing Prompts](../ai-prompts/academic-writing-prompts.md)

---

## 9. Tool Comparison Matrix

The table below compares 14 tools across dimensions that matter for GIS professionals producing scientific visualizations.

### Feature Matrix

| Tool | Language | Geo Native | Interactive | Publication Quality | Learning Curve | Free |
|------|----------|-----------|-------------|-------------------|---------------|------|
| **Observable Plot** | JS | Yes (projections) | Yes | Good | Low | Yes |
| **D3.js** | JS | Yes (full geo) | Yes | Excellent | High | Yes |
| **Vega-Lite** | JSON/JS | Yes (GeoShape) | Yes | Good | Low | Yes |
| **Altair** | Python | Via Vega-Lite | Yes | Good | Low | Yes |
| **ggplot2** | R | Via sf/stars | No (static) | Excellent | Medium | Yes |
| **tmap** | R | Yes (native) | Yes (tmap_mode) | Excellent | Low | Yes |
| **matplotlib** | Python | Via cartopy | No (static) | Excellent | Medium | Yes |
| **seaborn** | Python | No | No (static) | Good | Low | Yes |
| **Plotly** | Py/R/JS | Yes (Mapbox) | Yes | Good | Low | Yes (OSS) |
| **ECharts** | JS | Yes (GeoJSON) | Yes | Good | Medium | Yes |
| **Nivo** | React | No | Yes | Good | Medium | Yes |
| **Datawrapper** | Web | Yes (choropleth) | Limited | Excellent | Very Low | Freemium |
| **RAWGraphs** | Web | No | No | Good | Very Low | Yes |
| **Flourish** | Web | Yes | Yes | Good | Very Low | Freemium |

### Use Case Recommendations

| Use Case | Recommended Tool | Why |
|----------|-----------------|-----|
| **Journal figure with map + chart** | R ggplot2 + sf + patchwork | One pipeline, PDF output, total layout control |
| **Exploratory spatial data analysis** | GeoDa or Observable + D3 | LISA, Moran, linked views |
| **Web dashboard with linked map** | Plotly Dash or Observable Framework | Interactivity, server or static deployment |
| **Quick chart for a report** | Datawrapper or RAWGraphs | No code, fast iteration, clean defaults |
| **Epidemiological cluster analysis** | R (spatstat + SaTScan) + ggplot2 | Statistical rigor, reproducible output |
| **Climate model output** | Python xarray + matplotlib + cartopy | Handles NetCDF natively, standard in climate science |
| **Land use change flows** | Plotly Sankey or D3 | Interactive flow diagrams |
| **Animated migration flows** | D3.js or Flourish | Smooth transitions, particle animation |
| **React web application** | Nivo + deck.gl | Component-based, modern stack |
| **Reproducible academic paper** | Quarto + R/Python | Multi-language, multi-format, version-controlled |

### Performance at Scale

| Tool | 1K points | 100K points | 1M points | 10M points |
|------|-----------|-------------|-----------|------------|
| **Observable Plot** | Instant | Fast | Slow (SVG limit) | Use Canvas |
| **D3.js** | Instant | Fast (Canvas) | Feasible (Canvas) | WebGL needed |
| **deck.gl** | Instant | Instant | Fast | Feasible (GPU) |
| **ggplot2** | Instant | Fast | Slow (rendering) | Aggregate first |
| **matplotlib** | Instant | Fast | Slow | Aggregate first |
| **Plotly** | Instant | Moderate | Slow (WebGL mode helps) | Aggregate first |
| **Datawrapper** | Instant | Limit ~10K | Not supported | Not supported |

### Decision Flowchart (Text)

```
START: What kind of visualization?
|
+-- Statistical chart (no map)?
|   +-- Quick exploration? --> seaborn / Altair
|   +-- Publication? --> ggplot2 / matplotlib
|   +-- Interactive web? --> Observable Plot / Plotly
|
+-- Map + chart combo?
|   +-- R workflow? --> ggplot2 + sf + patchwork
|   +-- Python workflow? --> matplotlib + cartopy + seaborn
|   +-- Web deployment? --> Observable / Plotly Dash / Streamlit
|
+-- Specialized scientific?
|   +-- Ecology? --> R vegan + ggplot2
|   +-- Epidemiology? --> R spatstat + SaTScan
|   +-- Climate? --> Python xarray + cartopy
|   +-- Social science? --> D3.js / Observable (custom)
|
+-- Multi-dimensional?
|   +-- Sankey/alluvial? --> Plotly / D3
|   +-- Parallel coordinates? --> Plotly / Vega-Lite
|   +-- Treemap/sunburst? --> Plotly / ECharts
|
+-- No code needed?
    +-- Chart? --> Datawrapper / RAWGraphs
    +-- Map? --> Datawrapper / Flourish
    +-- Dashboard? --> Flourish / Kepler.gl
```

---

## Dark Arts: Tips for Scientific Visualization

These are hard-won lessons that separate good visualizations from great ones.

### Color

1. **Never use rainbow colormaps.** Viridis, magma, inferno, and plasma are perceptually uniform. Rainbow (jet) creates false boundaries and is unreadable to colorblind viewers.

2. **Diverging palettes need a meaningful midpoint.** Do not use diverging schemes for sequential data. The midpoint must represent something real (zero anomaly, national average, break-even point).

3. **Test with a colorblind simulator.** 8% of men have some form of color vision deficiency. Use [Viz Palette](https://projects.susielu.com/viz-palette) or the `colorblindcheck` R package.

4. **Print in grayscale?** Add a secondary visual variable (pattern, line style, label) so the chart works without color.

### Layout

5. **Align baselines.** When combining charts, shared axes should start at the same position. Use `patchwork` (R) or `gridspec` (matplotlib) for precise alignment.

6. **Small multiples beat animation.** Animation is flashy but makes comparison hard. Small multiples let the eye scan back and forth.

7. **Map insets for context.** When showing a small study area, add an inset map showing where it sits in the larger region.

### Statistical Honesty

8. **Show uncertainty.** Confidence intervals, prediction bands, and credible intervals are not optional for scientific figures. A point estimate without uncertainty is a half-truth.

9. **Avoid dual y-axes.** They are almost always misleading. Use facets or indexed values instead.

10. **Label directly.** Legends force the reader to look back and forth. Direct labels (on the line, on the bar) reduce cognitive load.

11. **Aspect ratio matters.** Banking to 45 degrees (Cleveland & McGill, 1988) makes slopes easier to perceive. Do not stretch or compress time series arbitrarily.

12. **Report sample size.** Every statistical visualization should state n somewhere. A beautiful chart based on n=12 means something different than n=12,000.

---

## Further Reading

| Resource | Type | URL |
|----------|------|-----|
| Wilke, *Fundamentals of Data Visualization* | Book (free online) | [clauswilke.com/dataviz](https://clauswilke.com/dataviz/) |
| Healy, *Data Visualization: A Practical Introduction* | Book | [socviz.co](https://socviz.co/) |
| Lovelace et al., *Geocomputation with R* | Book (free online) | [r.geocompx.org](https://r.geocompx.org/) |
| Observable Plot documentation | Reference | [observablehq.com/plot](https://observablehq.com/plot/) |
| From Data to Viz | Decision tool | [data-to-viz.com](https://www.data-to-viz.com/) |
| The Grammar of Graphics (Wilkinson) | Foundational theory | ISBN 978-0-387-24544-7 |
| ColorBrewer 2.0 | Color tool | [colorbrewer2.org](https://colorbrewer2.org/) |

---

[Back to Visualization](README.md) | [Back to Main README](../README.md)
