# Data Analysis

> Geospatial analysis tools, libraries, workflows, and statistical methods for Python, R, SQL, and cloud-native platforms -- the definitive expert-level reference.

> **Quick Picks**
> - **Python Core**: [GeoPandas](https://geopandas.org) + [Shapely 2.0](https://shapely.readthedocs.io) + [Rasterio](https://rasterio.readthedocs.io) -- the foundation
> - **R Core**: [sf](https://r-spatial.github.io/sf/) + [terra](https://rspatial.github.io/terra/) -- modern R spatial stack
> - **Spatial SQL**: [PostGIS](https://postgis.net) (production) or [DuckDB Spatial](https://duckdb.org/docs/extensions/spatial.html) (analytics)
> - **Cloud-Native**: [GeoParquet](https://geoparquet.org) + [STAC](https://stacspec.org) + [xarray](https://docs.xarray.dev)
> - **ML/DL**: [TorchGeo](https://torchgeo.readthedocs.io) + [samgeo](https://samgeo.gishub.org) -- foundation models for Earth observation

## Table of Contents

### Programming Stacks
| Page | Lines | Description |
|------|-------|-------------|
| [Python Geospatial Stack](python-stack.md) | 4,315 | GeoPandas 1.0+, Shapely 2.0, Rasterio, PySAL, xarray, DuckDB, visualization (Folium/Plotly/Lonboard/Datashader), 22 cookbook recipes, performance optimization, testing & CI |
| [R Geospatial Stack](r-stack.md) | 3,004 | sf/terra/stars, tmap v4, ggplot2+geom_sf, rayshader, spdep/spatialreg/spatstat, tidymodels spatial CV, sits (satellite time series), 17 cookbook recipes, Python↔R comparison |
| [Geospatial SQL](geospatial-sql.md) | 3,617 | PostGIS deep dive, DuckDB Spatial, SpatiaLite, geometry operations, spatial joins, pgRouting, raster SQL, advanced patterns, performance tuning, Docker deployment, 24 SQL recipes |

### Analysis & Statistics
| Page | Lines | Description |
|------|-------|-------------|
| [Spatial Statistics](spatial-statistics.md) | 2,456 | ESDA, point patterns (K/L/G functions), Moran's I & LISA, geostatistics (variograms, kriging), spatial regression (SAR/SEM/SDM), GWR/MGWR, Bayesian spatial (INLA), clustering & regionalization |
| [Machine Learning for GIS](ml-gis.md) | 2,201 | RF/XGBoost/LightGBM, deep learning (U-Net, YOLO, ViT), foundation models (Prithvi, Clay, SatlasPretrain), SAM for geospatial, spatial feature engineering, AutoML, GeoAI with LLMs, MLOps |
| [Time Series Analysis](time-series-analysis.md) | 2,801 | Satellite image time series, NDVI phenology, change detection (CCDC, LandTrendr, BFAST), climate analysis, urban growth, trajectory analysis (MovingPandas), forecasting, space-time cubes |

### Infrastructure & Workflows
| Page | Lines | Description |
|------|-------|-------------|
| [Cloud-Native Analytics](cloud-native-analytics.md) | 2,363 | GeoParquet/GeoArrow, DuckDB Spatial, COG, STAC ecosystem, xarray+Zarr, Apache Sedona, Dask-GeoPandas, GEE, Planetary Computer, Overture Maps, pipeline patterns, benchmarks |
| [Workflow Templates](workflow-templates.md) | 3,613 | 10 end-to-end workflows: site suitability, change detection, accessibility, EIA, UHI, flood risk, agricultural monitoring, population estimation, network analysis/routing, air quality assessment |

---

**Total: 24,370+ lines across 8 expert-level pages.**

## What Makes This Different

- **Full Stack Coverage**: Python + R + SQL + cloud platforms -- not just one language
- **Production Code**: Every section has runnable code examples, not pseudocode
- **SOTA Methods**: Foundation models (Prithvi, Clay), GeoParquet, DuckDB WASM, spatial AutoML
- **Statistical Rigor**: Spatial CV, LM diagnostics, Bayesian spatial modeling, uncertainty quantification
- **10 Workflow Templates**: Complete analysis pipelines with data sources, tools, code, and expected outputs
- **Performance Benchmarks**: Format comparisons, library benchmarks, optimization strategies
- **Cloud-Native**: STAC, COG, Zarr, GeoArrow -- modern data infrastructure patterns

## How to Choose

| Goal | Start Here |
|------|-----------|
| Set up Python for GIS | [Python Stack](python-stack.md) → Environment Setup |
| Set up R for GIS | [R Stack](r-stack.md) → Environment Setup |
| Spatial SQL queries | [Geospatial SQL](geospatial-sql.md) → PostGIS or DuckDB |
| Moran's I / LISA / hot spots | [Spatial Statistics](spatial-statistics.md) → Spatial Autocorrelation |
| Kriging / interpolation | [Spatial Statistics](spatial-statistics.md) → Geostatistics |
| Spatial regression (GWR) | [Spatial Statistics](spatial-statistics.md) → GWR & MGWR |
| Land cover classification (ML) | [ML for GIS](ml-gis.md) → Supervised Classification |
| Deep learning on imagery | [ML for GIS](ml-gis.md) → Foundation Models |
| SAM segmentation | [ML for GIS](ml-gis.md) → SAM for Geospatial |
| NDVI time series / phenology | [Time Series](time-series-analysis.md) → NDVI/EVI |
| Change detection | [Time Series](time-series-analysis.md) → Change Detection |
| GeoParquet / DuckDB | [Cloud-Native](cloud-native-analytics.md) → GeoParquet |
| STAC + xarray workflow | [Cloud-Native](cloud-native-analytics.md) → STAC Ecosystem |
| Site suitability analysis | [Workflows](workflow-templates.md) → Site Suitability |
| Flood risk assessment | [Workflows](workflow-templates.md) → Flood Risk |
| Network routing / isochrones | [Workflows](workflow-templates.md) → Network Analysis |

## Cross-References

- **[Tools](../tools/)** -- Desktop GIS, CLI tools, servers, databases that complement these analysis libraries
- **[JS Bindbox](../js-bindbox/)** -- JavaScript libraries for web-based visualization of analysis results
- **[Visualization](../visualization/)** -- Thematic maps, dashboards, 3D viz for presenting analysis outputs
- **[Data Sources](../data-sources/)** -- Where to find the geospatial data these analyses consume
- **[AI Prompts](../ai-prompts/)** -- LLM prompts for generating analysis code and interpreting results
