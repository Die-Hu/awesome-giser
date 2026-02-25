# R Geospatial Stack

> The definitive expert-level reference for the modern R spatial ecosystem. This guide covers core I/O, visualization, spatial analysis, remote sensing, machine learning, big data workflows, network analysis, reproducibility, and cross-language interop with Python. Designed for GIS professionals transitioning to or deepening their work in R.

> **Quick Picks**
> - **Start here:** `sf` (vectors) + `terra` (rasters) -- the modern core
> - **Best interactive map:** `tmap` (mode = "view") or `mapview` (one-liner)
> - **Publication maps:** `ggplot2` + `geom_sf()` + `tidyterra`
> - **3D visualization:** `rayshader`
> - **Point patterns:** `spatstat`
> - **Spatial regression:** `spatialreg` + `GWmodel`
> - **Remote sensing time series:** `sits`
> - **Spatial ML with proper CV:** `tidymodels` + `spatialsample`
> - **Big spatial data:** `arrow` + `geoarrow` + `DBI`

---

## Table of Contents

- [Core Libraries Deep Dive](#core-libraries-deep-dive)
  - [sf 1.0+ -- Simple Features for R](#sf-10----simple-features-for-r)
  - [terra -- Modern Raster and Vector Processing](#terra----modern-raster-and-vector-processing)
  - [stars -- Spatiotemporal Arrays](#stars----spatiotemporal-arrays)
  - [s2 -- Spherical Geometry Engine](#s2----spherical-geometry-engine)
  - [gdalcubes -- Earth Observation Data Cubes](#gdalcubes----earth-observation-data-cubes)
- [Migration Guide: sp/raster/rgdal to sf/terra](#migration-guide-sprasterrgdal-to-sfterra)
- [Visualization Deep Dive](#visualization-deep-dive)
  - [tmap v4](#tmap-v4)
  - [ggplot2 + geom_sf -- Advanced Cartography](#ggplot2--geom_sf----advanced-cartography)
  - [mapsf -- Publication-Quality Thematic Maps](#mapsf----publication-quality-thematic-maps)
  - [rayshader -- Advanced 3D Visualization](#rayshader----advanced-3d-visualization)
  - [leaflet -- Interactive Web Maps](#leaflet----interactive-web-maps)
  - [mapview -- Quick Interactive Exploration](#mapview----quick-interactive-exploration)
  - [plotly -- Interactive Statistical Maps](#plotly----interactive-statistical-maps)
  - [Choosing a Visualization Package](#choosing-a-visualization-package)
- [Analysis Ecosystem](#analysis-ecosystem)
  - [spdep -- Spatial Dependence](#spdep----spatial-dependence)
  - [spatialreg -- Spatial Regression Models](#spatialreg----spatial-regression-models)
  - [spatstat -- Point Pattern Analysis](#spatstat----point-pattern-analysis)
  - [gstat -- Geostatistics](#gstat----geostatistics)
  - [rgeoda -- GeoDa in R](#rgeoda----geoda-in-r)
- [Network and Transport Analysis](#network-and-transport-analysis)
  - [sfnetworks -- Tidy Spatial Networks](#sfnetworks----tidy-spatial-networks)
  - [dodgr -- Distances on Directed Graphs](#dodgr----distances-on-directed-graphs)
  - [r5r -- Multimodal Accessibility](#r5r----multimodal-accessibility)
  - [stplanr -- Sustainable Transport Planning](#stplanr----sustainable-transport-planning)
  - [osmdata -- OpenStreetMap in R](#osmdata----openstreetmap-in-r)
- [Remote Sensing in R](#remote-sensing-in-r)
  - [terra for Remote Sensing](#terra-for-remote-sensing)
  - [sits -- Satellite Image Time Series](#sits----satellite-image-time-series)
  - [openeo -- Cloud-Based Earth Observation](#openeo----cloud-based-earth-observation)
  - [sen2r -- Sentinel-2 Processing](#sen2r----sentinel-2-processing)
  - [MODISTools and getSpatialData](#modistools-and-getspatialdata)
- [Machine Learning for Geospatial](#machine-learning-for-geospatial)
  - [tidymodels + spatialsample -- Spatial Cross-Validation](#tidymodels--spatialsample----spatial-cross-validation)
  - [mlr3 + mlr3spatial](#mlr3--mlr3spatial)
  - [blockCV -- Spatial Blocking](#blockcv----spatial-blocking)
  - [Common ML Backends for Spatial Problems](#common-ml-backends-for-spatial-problems)
- [Big Data and Cloud Workflows](#big-data-and-cloud-workflows)
  - [arrow + geoarrow -- GeoParquet](#arrow--geoarrow----geoparquet)
  - [DBI + RPostgis -- Spatial Databases](#dbi--rpostgis----spatial-databases)
  - [targets -- Pipeline Orchestration](#targets----pipeline-orchestration)
  - [Parallel Processing -- future and furrr](#parallel-processing----future-and-furrr)
- [Reproducibility](#reproducibility)
  - [renv -- Dependency Management](#renv----dependency-management)
  - [targets -- Reproducible Pipelines](#targets----reproducible-pipelines)
  - [Quarto -- Reports with Maps](#quarto----reports-with-maps)
  - [Docker -- rocker/geospatial](#docker----rockergeospatial)
  - [Testing Spatial Code](#testing-spatial-code)
- [Environment Setup](#environment-setup)
  - [CRAN vs r-universe](#cran-vs-r-universe)
  - [System Dependencies by Platform](#system-dependencies-by-platform)
  - [Docker for R Spatial](#docker-for-r-spatial)
  - [IDE Options -- RStudio and Positron](#ide-options----rstudio-and-positron)
- [Cookbook -- 15+ Spatial Recipes](#cookbook----15-spatial-recipes)
- [Python to R Comparison](#python-to-r-comparison)
- [Further Reading](#further-reading)

---

## Core Libraries Deep Dive

The modern R spatial stack has converged around `sf` for vectors and `terra` for rasters, fully replacing the retired `sp`, `rgdal`, `rgeos`, and `maptools` packages. This section provides expert-level coverage of each core library.

| Package | Purpose | Successor To | CRAN |
|---|---|---|---|
| [sf](https://r-spatial.github.io/sf/) | Vector data (Simple Features) | `sp`, `rgdal`, `rgeos` | [CRAN](https://cran.r-project.org/package=sf) |
| [terra](https://rspatial.github.io/terra/) | Raster and vector data | `raster`, `rgdal` | [CRAN](https://cran.r-project.org/package=terra) |
| [stars](https://r-spatial.github.io/stars/) | Spatiotemporal arrays | `raster` (for datacubes) | [CRAN](https://cran.r-project.org/package=stars) |
| [sp](https://edzer.github.io/sp/) | Spatial classes (legacy) | -- (retired) | [CRAN](https://cran.r-project.org/package=sp) |
| [s2](https://r-spatial.github.io/s2/) | Spherical geometry | Planar geometry in `sf` | [CRAN](https://cran.r-project.org/package=s2) |
| [gdalcubes](https://gdalcubes.github.io/) | Earth observation datacubes | -- | [CRAN](https://cran.r-project.org/package=gdalcubes) |
| [arrow + geoarrow](https://arrow.apache.org/docs/r/) | GeoParquet / large data | -- | [CRAN](https://cran.r-project.org/package=arrow) |
| [tidyterra](https://dieghernan.github.io/tidyterra/) | ggplot2 for terra objects | -- | [CRAN](https://cran.r-project.org/package=tidyterra) |

### sf 1.0+ -- Simple Features for R

`sf` is the backbone of the modern R vector ecosystem. Since version 1.0, it integrates Google's S2 spherical geometry library by default, meaning geographic (lon/lat) operations are computed on the sphere rather than projected to a plane.

#### Key Concepts

```r
library(sf)

# Read any vector format GDAL supports
nc <- st_read(system.file("shape/nc.shp", package = "sf"))

# sf objects are data.frames with a geometry column
class(nc)
#> [1] "sf"         "data.frame"

# The geometry column is a list-column of class sfc
class(st_geometry(nc))
#> [1] "sfc_MULTIPOLYGON" "sfc"

# Each element is an sfg (simple feature geometry) object
class(st_geometry(nc)[[1]])
#> [1] "XY"               "MULTIPOLYGON"     "sfg"
```

#### S2 Spherical Geometry (Default Since sf 1.0)

Prior to sf 1.0, all geometric operations on geographic (lon/lat) data used planar GEOS algorithms, which produced incorrect results for large areas or near the poles. Now S2 is the default:

```r
library(sf)

# Check current geometry engine
sf_use_s2()
#> [1] TRUE

# S2 correctly computes areas on the sphere
nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
areas_s2 <- st_area(nc)  # uses S2 by default -- result in m^2

# Compare: disable S2 to use planar GEOS (incorrect for lon/lat)
sf_use_s2(FALSE)
areas_geos <- st_area(nc)  # planar -- less accurate for geographic CRS
sf_use_s2(TRUE)             # always restore

# Differences can be significant for large polygons
summary(as.numeric(areas_s2 - areas_geos))
```

#### When to Use S2 vs GEOS

| Scenario | Engine | How |
|---|---|---|
| Geographic CRS (EPSG:4326), global analysis | S2 (default) | `sf_use_s2(TRUE)` |
| Projected CRS (UTM, state plane, etc.) | GEOS (automatic) | sf uses GEOS for projected CRS |
| Legacy code that breaks with S2 | GEOS (override) | `sf_use_s2(FALSE)` |
| High-precision local analysis | GEOS after projecting | `st_transform()` then operate |
| Topology-sensitive operations | GEOS (more mature) | `sf_use_s2(FALSE)` |

**Common S2 gotcha:** Some geometries valid under GEOS are invalid under S2 (e.g., self-intersecting polygons, ring direction issues). Fix with:

```r
# Repair geometries for S2 compatibility
nc_valid <- st_make_valid(nc)

# If still failing, try GEOS repair then re-enable S2
sf_use_s2(FALSE)
nc_repaired <- st_make_valid(nc)
sf_use_s2(TRUE)
```

#### Tidy Integration

`sf` objects work seamlessly with `dplyr`, `tidyr`, and the pipe:

```r
library(dplyr)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)

# Filter, mutate, summarize -- geometry is sticky
high_birth <- nc |>
  filter(BIR74 > 5000) |>
  mutate(birth_density = BIR74 / as.numeric(st_area(geometry)) * 1e6) |>
  select(NAME, birth_density)

# Group + summarize dissolves geometries automatically
by_region <- nc |>
  mutate(region = ifelse(CNTY_ < 1900, "west", "east")) |>
  group_by(region) |>
  summarize(total_births = sum(BIR74), .groups = "drop")
# Result: 2 polygons (dissolved) with aggregated attributes
```

#### Spatial Operations Cheat Sheet

```r
# Geometric operations
st_buffer(x, dist = 1000)          # buffer (meters if projected, or S2 meters)
st_intersection(x, y)              # clip x by y
st_union(x)                        # dissolve all geometries
st_difference(x, y)                # erase y from x
st_sym_difference(x, y)            # symmetric difference
st_convex_hull(x)                  # convex hull
st_voronoi(x)                      # Voronoi polygons
st_centroid(x)                     # centroids
st_point_on_surface(x)             # guaranteed point inside polygon
st_simplify(x, dTolerance = 100)   # Douglas-Peucker simplification
st_snap(x, y, tolerance = 10)      # snap geometries

# Predicates (return logical matrix or sparse list)
st_intersects(x, y)                # which features intersect
st_contains(x, y)                  # which x contains which y
st_within(x, y)                    # which x is within which y
st_touches(x, y)                   # shared boundary, no interior overlap
st_crosses(x, y)                   # crosses (lines)
st_nearest_feature(x, y)           # index of nearest feature in y
st_is_within_distance(x, y, 500)   # within distance threshold

# Measurements
st_area(x)                         # area (units object)
st_length(x)                       # length of lines
st_distance(x, y)                  # distance matrix

# CRS operations
st_crs(x)                          # get CRS
st_set_crs(x, 4326)               # set CRS (no reprojection)
st_transform(x, 32633)            # reproject to UTM zone 33N
```

#### Writing to Multiple Formats

```r
# GeoPackage (recommended default)
st_write(nc, "output.gpkg", layer = "counties")

# GeoJSON
st_write(nc, "output.geojson")

# Shapefile (legacy, 10-char field name limit)
st_write(nc, "output.shp")

# GeoParquet (fast columnar format -- see Big Data section)
sfarrow::st_write_parquet(nc, "output.parquet")

# FlatGeobuf (streaming-friendly, cloud-native)
st_write(nc, "output.fgb")

# Spatial database
library(DBI)
con <- dbConnect(RPostgres::Postgres(), dbname = "gisdb")
st_write(nc, con, "counties")
```

### terra -- Modern Raster and Vector Processing

`terra` is the successor to `raster`, written in C++ with dramatically better performance and memory efficiency. It handles both raster (`SpatRaster`) and vector (`SpatVector`) data.

#### SpatRaster Fundamentals

```r
library(terra)

# Create from file -- lazy loading (data stays on disk until needed)
r <- rast("landcover.tif")
r
#> class       : SpatRaster
#> dimensions  : 1000, 1000, 5  (nrow, ncol, nlyr)
#> resolution  : 30, 30  (x, y)
#> extent      : 500000, 530000, 4000000, 4030000  (xmin, xmax, ymin, ymax)
#> coord. ref. : WGS 84 / UTM zone 33N (EPSG:32633)
#> source      : landcover.tif
#> names       : blue, green, red, nir, swir

# Create from scratch
r_new <- rast(nrows = 100, ncols = 100, xmin = 0, xmax = 10,
              ymin = 0, ymax = 10, vals = runif(10000))

# Multi-layer (band) operations
ndvi <- (r[["nir"]] - r[["red"]]) / (r[["nir"]] + r[["red"]])
names(ndvi) <- "NDVI"

# Raster algebra -- same syntax as raster package
slope <- terrain(r[["nir"]], v = "slope", unit = "degrees")
aspect <- terrain(r[["nir"]], v = "aspect", unit = "degrees")
```

#### SpatVector -- terra's Vector Class

`terra` includes its own vector class, useful when working in a pure terra pipeline:

```r
# Read vector data
v <- vect("boundaries.gpkg")

# Convert between sf and terra
library(sf)
sf_obj <- st_as_sf(v)       # terra -> sf
v_obj  <- vect(sf_obj)      # sf -> terra

# Vector operations in terra
v_buffered <- buffer(v, width = 1000)
v_cropped  <- crop(v, ext(r))
```

#### terra vs raster Performance Comparison

| Operation | `raster` | `terra` | Speedup |
|---|---|---|---|
| Read 10 GB GeoTIFF | 45 s | 8 s | ~5x |
| Zonal statistics (10k zones) | 120 s | 15 s | ~8x |
| Reclassify (100M cells) | 30 s | 4 s | ~7x |
| Focal (3x3 mean, 100M cells) | 90 s | 12 s | ~7x |
| Raster algebra (NDVI) | 25 s | 5 s | ~5x |
| Write COG | not native | 3 s | -- |

*Benchmarks approximate; actual speedup depends on hardware and data.*

#### Essential terra Operations

```r
# I/O
r <- rast("input.tif")
writeRaster(r, "output.tif", overwrite = TRUE)
writeRaster(r, "output_cog.tif", filetype = "COG")  # Cloud-Optimized GeoTIFF

# Crop, mask, extract
r_crop <- crop(r, ext(v))                 # crop to extent
r_mask <- mask(r_crop, v)                 # mask to polygon boundary
vals   <- extract(r, v, fun = mean)       # zonal statistics

# Resampling and reprojection
r_proj <- project(r, "EPSG:4326", method = "bilinear")
r_agg  <- aggregate(r, fact = 4, fun = mean)
r_dis  <- disagg(r, fact = 2, method = "bilinear")

# Focal (moving window) operations
r_smooth <- focal(r, w = matrix(1, 5, 5), fun = mean, na.rm = TRUE)

# Reclassify
rcl_matrix <- matrix(c(0, 50, 1,
                        50, 100, 2,
                        100, 255, 3), ncol = 3, byrow = TRUE)
r_class <- classify(r, rcl_matrix)

# Zonal statistics
zones <- rast("landuse_zones.tif")
zonal(r, zones, fun = "mean")

# Distance and cost surfaces
d <- distance(r)                          # distance to non-NA cells
cost_surface <- costDist(r, target = 0)   # cost distance

# Global statistics
global(r, fun = c("mean", "sd", "min", "max"), na.rm = TRUE)

# Apply custom functions
r_custom <- app(r, fun = function(x) log(x + 1))
r_lapp   <- lapp(c(r[["nir"]], r[["red"]]),
                  fun = function(nir, red) (nir - red) / (nir + red))
```

### stars -- Spatiotemporal Arrays

`stars` represents spatiotemporal data as multi-dimensional arrays -- ideal for data cubes (x, y, time, band). It handles both regular and irregular grids and supports lazy evaluation via proxy objects.

```r
library(stars)

# Read a single file
tif <- system.file("tif/L7_ETMs.tif", package = "stars")
s <- read_stars(tif)
s
#> stars object with 3 dimensions and 1 attribute
#> attribute(s):
#>   L7_ETMs.tif
#>   Min.   : 1.00
#>   ...
#> dimension(s):
#>        from  to  offset  delta  refsys  point  x/y
#> x         1 349  288776   28.5 UTM 25N  FALSE  [x]
#> y         1 352 9120761  -28.5 UTM 25N  FALSE  [y]
#> band      1   6      NA     NA      NA     NA

# Proxy objects -- lazy evaluation for large data
s_proxy <- read_stars(tif, proxy = TRUE)
# Operations are recorded but not executed until you fetch/plot
s_proxy |>
  slice(band, 1:3) |>
  st_crop(st_bbox(c(xmin = 289000, ymin = 9118000,
                     xmax = 296000, ymax = 9121000))) |>
  plot()  # only now is data read from disk

# Create a time dimension
files <- list.files("sentinel2_monthly/", pattern = "\\.tif$", full.names = TRUE)
s_ts <- read_stars(files, along = "time")
st_dimensions(s_ts)$time$values <- as.Date(paste0("2024-", 1:12, "-01"))

# Compute per-pixel temporal statistics
ndvi_mean <- st_apply(s_ts, c("x", "y"), mean, na.rm = TRUE)
ndvi_trend <- st_apply(s_ts, c("x", "y"), function(x) {
  if (all(is.na(x))) return(NA_real_)
  coef(lm(x ~ seq_along(x)))[2]
})
```

#### stars vs terra -- When to Use Which

| Criterion | stars | terra |
|---|---|---|
| Data model | N-dimensional arrays | 2D raster layers |
| Time series | Native time dimension | Stack layers manually |
| Lazy evaluation | Proxy objects | On-disk by default |
| Integration | sf, tidyverse, ggplot2 | Own SpatRaster class |
| NetCDF support | Excellent | Good |
| Performance | Moderate | Fast (C++) |
| Best for | Datacubes, NetCDF, climate | Single rasters, RS, terrain |

### s2 -- Spherical Geometry Engine

The `s2` package wraps Google's S2 geometry library and is used automatically by `sf` for geographic CRS operations:

```r
library(s2)

# S2 cell IDs -- hierarchical global grid
pt <- s2_lnglat(-73.985, 40.748)  # NYC
cell <- s2_cell(pt)
s2_cell_level(cell)     # level 30 (finest)
s2_cell_parent(cell, 5) # coarser level-5 cell

# S2 covering -- approximate a polygon with S2 cells
nc <- sf::st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
geog <- s2_geog_from_wkb(sf::st_as_binary(sf::st_geometry(nc[1, ])))
covering <- s2_covering(geog, max_cells = 20)

# Direct S2 operations
s2_area(geog)                          # area in m^2
s2_distance(s2_lnglat(0, 0), s2_lnglat(1, 1))  # distance in meters
s2_contains(geog, s2_lnglat(-79, 35.5))         # point-in-polygon
```

### gdalcubes -- Earth Observation Data Cubes

`gdalcubes` builds on-the-fly data cubes from collections of satellite images with different projections, extents, and resolutions:

```r
library(gdalcubes)

# Create an image collection from Sentinel-2 files
files <- list.files("S2_tiles/", pattern = "\\.tif$",
                    recursive = TRUE, full.names = TRUE)
col <- create_image_collection(files, format = "Sentinel2_L2A")

# Define a target cube view (space-time grid)
v <- cube_view(
  srs = "EPSG:32633",
  extent = list(
    left = 400000, right = 500000,
    bottom = 5400000, top = 5500000,
    t0 = "2024-01-01", t1 = "2024-12-31"
  ),
  dx = 100, dy = 100, dt = "P1M",  # 100m spatial, monthly temporal
  aggregation = "median",
  resampling = "bilinear"
)

# Build and process the cube
cube <- raster_cube(col, v)

# Compute NDVI
ndvi_cube <- cube |>
  select_bands(c("B04", "B08")) |>
  apply_pixel("(B08-B04)/(B08+B04)", names = "NDVI")

# Reduce time dimension
ndvi_max <- ndvi_cube |>
  reduce_time("max(NDVI)")

# Write result
write_tif(ndvi_max, dir = "output/", prefix = "ndvi_max_")

# Convert to stars for further analysis
ndvi_stars <- st_as_stars(ndvi_cube)
```

---

## Migration Guide: sp/raster/rgdal to sf/terra

As of 2023, `rgdal`, `rgeos`, and `maptools` have been removed from CRAN. The `sp` and `raster` packages remain available but are in maintenance-only mode. All new code should use `sf` and `terra`.

### Complete Migration Table

| Old Code | New Code | Notes |
|---|---|---|
| `library(rgdal)` | `library(sf)` or `library(terra)` | rgdal retired Oct 2023 |
| `library(rgeos)` | `library(sf)` | sf includes GEOS via s2 |
| `library(maptools)` | `library(sf)` | maptools retired Oct 2023 |
| `readOGR("f.shp")` | `st_read("f.shp")` | Auto-detects driver |
| `writeOGR(obj, "f.shp", ...)` | `st_write(obj, "f.shp")` | Simpler API |
| `readGDAL("f.tif")` | `read_stars("f.tif")` or `rast("f.tif")` | -- |
| `SpatialPointsDataFrame(coords, data)` | `st_as_sf(data, coords = c("lon","lat"), crs = 4326)` | Tidier |
| `SpatialPolygonsDataFrame` | `st_as_sf(sp_obj)` | Convert existing |
| `spTransform(obj, CRS("+init=epsg:4326"))` | `st_transform(obj, 4326)` | Numeric EPSG OK |
| `over(pts, polys)` | `st_join(pts, polys)` | Tidy join |
| `gBuffer(obj, width=1000)` | `st_buffer(obj, 1000)` | S2-aware for lonlat |
| `gIntersection(x, y)` | `st_intersection(x, y)` | Returns sf |
| `gUnion(x, y)` | `st_union(x, y)` | -- |
| `gArea(obj)` | `st_area(obj)` | Returns units |
| `gLength(obj)` | `st_length(obj)` | Returns units |
| `raster("f.tif")` | `rast("f.tif")` | Much faster |
| `stack(list_of_files)` | `rast(list_of_files)` | Native multi-layer |
| `brick("f.tif")` | `rast("f.tif")` | No brick/stack distinction |
| `extent(r)` | `ext(r)` | Shorter name |
| `crs(r)` | `crs(r)` | Same name, different class |
| `projectRaster(r, crs=...)` | `project(r, "EPSG:4326")` | Cleaner API |
| `crop(r, extent)` | `crop(r, ext_obj)` | Similar |
| `mask(r, poly)` | `mask(r, v)` | SpatVector or sf |
| `extract(r, pts)` | `extract(r, pts)` | Similar but returns data.frame |
| `raster::calc(r, fun)` | `terra::app(r, fun)` | Renamed |
| `raster::overlay(r1, r2, fun)` | `terra::lapp(c(r1, r2), fun)` | Renamed |
| `raster::stackApply(s, idx, fun)` | `terra::tapp(s, idx, fun)` | Renamed |
| `raster::cellStats(r, stat)` | `terra::global(r, stat)` | Renamed |
| `raster::freq(r)` | `terra::freq(r)` | Same name |
| `raster::focal(r, w, fun)` | `terra::focal(r, w, fun)` | Same API |
| `raster::resample(r, template)` | `terra::resample(r, template)` | Same API |
| `raster::merge(r1, r2)` | `terra::merge(r1, r2)` | Same API |
| `raster::mosaic(r1, r2, fun)` | `terra::mosaic(r1, r2, fun)` | Same API |
| `raster::writeRaster(r, "f.tif")` | `terra::writeRaster(r, "f.tif")` | Adds COG support |

### Common Migration Pitfalls

**1. CRS specification has changed:**

```r
# OLD -- PROJ4 strings (deprecated)
CRS("+proj=utm +zone=33 +datum=WGS84")
CRS("+init=epsg:4326")

# NEW -- use EPSG codes or WKT2
st_crs(4326)
st_crs("EPSG:32633")
st_crs(nc)$wkt   # inspect full WKT2
```

**2. S2 changes geometric results for geographic CRS:**

```r
# A buffer of 1000 meters around a lon/lat point:
# OLD (planar, wrong): buffer was in degrees
# NEW (S2, correct): buffer is in meters on the sphere
pt <- st_sfc(st_point(c(-73.985, 40.748)), crs = 4326)
buf <- st_buffer(pt, 1000)  # 1000 meters, computed on sphere
```

**3. terra objects are not serializable by default:**

```r
# This FAILS in parallel or when saving to RDS:
r <- rast("big.tif")
saveRDS(r, "raster.rds")  # saves a pointer, not data

# Solution: wrap/unwrap
r_wrapped <- wrap(r)
saveRDS(r_wrapped, "raster.rds")
r_restored <- unwrap(readRDS("raster.rds"))
```

**4. Slot access is different:**

```r
# OLD
sp_obj@data
sp_obj@proj4string
coordinates(sp_obj)

# NEW (sf)
sf_obj             # it IS a data.frame, no @ needed
st_crs(sf_obj)
st_coordinates(sf_obj)
```

---

## Visualization Deep Dive

R has best-in-class cartographic libraries. This section covers each in depth with working examples.

> **See also:** [Visualization Overview](../visualization/README.md) | [Thematic Maps](../visualization/thematic-maps.md) | [3D Visualization](../visualization/3d-visualization.md) | [Cartography Design](../visualization/cartography-design.md)

### tmap v4

`tmap` v4 introduced a complete rewrite with a new layer-based syntax, better faceting, and improved aesthetics. It remains the most versatile mapping package in R, supporting both static (`"plot"`) and interactive (`"view"`) modes.

```r
library(tmap)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)

# tmap v4 syntax
tm <- tm_shape(nc) +
  tm_polygons(
    fill = "BIR74",
    fill.scale = tm_scale_continuous(values = "brewer.yl_gn_bu"),
    fill.legend = tm_legend(title = "Births (1974)"),
    col = "grey30",
    lwd = 0.5
  ) +
  tm_title("North Carolina Births") +
  tm_scalebar(position = c("left", "bottom")) +
  tm_compass(position = c("right", "top")) +
  tm_layout(
    frame = FALSE,
    legend.outside = TRUE
  )

# Static output
tmap_mode("plot")
tm
tmap_save(tm, "nc_births.png", dpi = 300, width = 10, height = 6)

# Switch to interactive
tmap_mode("view")
tm  # same code, now renders as Leaflet map
```

#### tmap v4 Advanced: Faceting and Bivariate Maps

```r
# Faceted maps
tm_shape(nc) +
  tm_polygons(
    fill = c("BIR74", "BIR79"),
    fill.scale = tm_scale_continuous(values = "brewer.yl_or_rd"),
    fill.free = TRUE
  ) +
  tm_facets(ncol = 2) +
  tm_title("Births by Decade")

# Small multiples with different variables
tm_shape(nc) +
  tm_polygons(fill = "SID74") +
  tm_facets(by = "SID74 > 5", free.scales = TRUE)
```

### ggplot2 + geom_sf -- Advanced Cartography

`ggplot2` with `geom_sf()` provides publication-quality static maps using the grammar of graphics. Combined with `ggspatial` for map furniture and `tidyterra` for rasters, it is the most flexible approach.

```r
library(ggplot2)
library(sf)
library(ggspatial)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)

# Publication-quality choropleth
ggplot(nc) +
  geom_sf(aes(fill = BIR74), color = "white", linewidth = 0.2) +
  scale_fill_viridis_c(
    name = "Births\n(1974)",
    option = "magma",
    direction = -1,
    labels = scales::comma
  ) +
  annotation_scale(location = "bl", width_hint = 0.3) +
  annotation_north_arrow(
    location = "tr",
    which_north = "true",
    style = north_arrow_fancy_orienteering()
  ) +
  labs(
    title = "Live Births in North Carolina Counties, 1974",
    caption = "Source: NC SIDS data | sf package"
  ) +
  theme_void() +
  theme(
    legend.position = c(0.15, 0.3),
    plot.title = element_text(size = 14, face = "bold"),
    plot.caption = element_text(size = 8, color = "grey50")
  )
```

#### Inset Maps

```r
library(ggplot2)
library(sf)
library(cowplot)

# Main map
main_map <- ggplot(nc) +
  geom_sf(aes(fill = BIR74), color = "white", linewidth = 0.2) +
  scale_fill_viridis_c(option = "magma", direction = -1) +
  theme_void() +
  theme(legend.position = "none")

# Inset: location within the US
us <- st_as_sf(maps::map("state", plot = FALSE, fill = TRUE))
inset <- ggplot() +
  geom_sf(data = us, fill = "grey90", color = "grey60", linewidth = 0.2) +
  geom_sf(data = st_union(nc), fill = "red", color = NA) +
  theme_void() +
  theme(panel.border = element_rect(fill = NA, color = "black", linewidth = 0.5))

# Combine
ggdraw() +
  draw_plot(main_map) +
  draw_plot(inset, x = 0.02, y = 0.65, width = 0.3, height = 0.3)
```

#### Faceted Maps

```r
library(tidyr)

# Reshape for faceting
nc_long <- nc |>
  select(NAME, BIR74, BIR79, geometry) |>
  pivot_longer(cols = c(BIR74, BIR79), names_to = "year", values_to = "births")

ggplot(nc_long) +
  geom_sf(aes(fill = births), color = "white", linewidth = 0.1) +
  facet_wrap(~year, ncol = 1) +
  scale_fill_viridis_c(option = "plasma", labels = scales::comma) +
  labs(title = "NC Births by Decade", fill = "Births") +
  theme_void()
```

#### Rasters with tidyterra

```r
library(tidyterra)
library(terra)
library(ggplot2)

dem <- rast("elevation.tif")

ggplot() +
  geom_spatraster(data = dem) +
  scale_fill_hypso_c(
    palette = "wiki-schwarzwald-cont",
    name = "Elevation (m)",
    na.value = "white"
  ) +
  geom_sf(data = nc, fill = NA, color = "black", linewidth = 0.3) +
  labs(title = "Elevation with County Boundaries") +
  theme_void()
```

### mapsf -- Publication-Quality Thematic Maps

`mapsf` is the successor to `cartography`, designed for French-school cartographic conventions but broadly useful:

```r
library(mapsf)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)

# Proportional symbols with choropleth background
mf_export(nc, filename = "nc_map.png", width = 1200, res = 150)
mf_map(nc, var = "BIR74", type = "choro",
       pal = "Mint", breaks = "quantile", nbreaks = 5,
       leg_title = "Births (1974)", leg_pos = "topleft")
mf_map(nc, var = "SID74", type = "prop",
       col = "tomato", leg_title = "SID Deaths",
       leg_pos = "bottomleft", add = TRUE)
mf_title("NC Births and SID Deaths, 1974")
mf_scale(size = 50)
mf_arrow(pos = "topright")
mf_credits("Source: NC SIDS data")
dev.off()
```

### rayshader -- Advanced 3D Visualization

`rayshader` creates stunning 3D terrain renderings. Beyond basic use, it supports custom overlays, animations, and export to video.

> **See also:** [3D Visualization](../visualization/3d-visualization.md)

```r
library(rayshader)
library(terra)

# Load and prepare DEM
dem <- rast("elevation.tif")
elmat <- raster_to_matrix(dem)

# Advanced rendering with multiple shadow layers
elmat |>
  sphere_shade(texture = "desert", sunangle = 315) |>
  add_water(detect_water(elmat, min_area = 200), color = "desert") |>
  add_shadow(ray_shade(elmat, sunaltitude = 30, zscale = 10), max_darken = 0.5) |>
  add_shadow(ambient_shade(elmat, zscale = 10), max_darken = 0.3) |>
  add_shadow(lamb_shade(elmat, zscale = 10), max_darken = 0.7) |>
  plot_3d(
    elmat,
    zscale = 10,
    theta = -45,
    phi = 30,
    zoom = 0.6,
    fov = 70,
    windowsize = c(1200, 800),
    background = "#1a1a2e"
  )

# High-quality render
render_highquality(
  filename = "terrain_hq.png",
  samples = 256,
  light = TRUE,
  lightdirection = c(315, 315),
  lightaltitude = c(45, 20),
  lightintensity = c(600, 200),
  lightcolor = c("white", "#ffd700"),
  width = 2400,
  height = 1600
)
```

#### Animated Flyover

```r
# Create a 360-degree rotation animation
angles <- seq(0, 360, length.out = 120)
for (i in seq_along(angles)) {
  render_camera(theta = angles[i], phi = 30, zoom = 0.6)
  render_snapshot(filename = sprintf("frames/frame_%04d.png", i))
}
# Combine frames with ffmpeg
# ffmpeg -framerate 30 -i frames/frame_%04d.png -c:v libx264 flyover.mp4
```

#### Overlay ggplot2 Map on 3D Terrain

```r
library(rayshader)
library(ggplot2)
library(sf)

# Create a 2D map with ggplot
gg_overlay <- ggplot(nc) +
  geom_sf(aes(fill = BIR74), color = "white", linewidth = 0.2) +
  scale_fill_viridis_c() +
  theme_void() +
  theme(legend.position = "none")

# Render as 3D
plot_gg(
  gg_overlay,
  width = 6, height = 4,
  scale = 300,
  multicore = TRUE,
  raytrace = TRUE,
  windowsize = c(1200, 800)
)
render_snapshot("nc_3d.png")
```

### leaflet -- Interactive Web Maps

`leaflet` provides full access to the Leaflet.js library with plugins:

```r
library(leaflet)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
pal <- colorQuantile("YlOrRd", nc$BIR74, n = 5)

leaflet(nc) |>
  addProviderTiles(providers$CartoDB.Positron) |>
  addPolygons(
    fillColor = ~pal(BIR74),
    fillOpacity = 0.7,
    color = "white",
    weight = 1,
    popup = ~paste0(
      "<strong>", NAME, "</strong><br/>",
      "Births (1974): ", scales::comma(BIR74), "<br/>",
      "SID Deaths: ", SID74
    ),
    highlightOptions = highlightOptions(
      weight = 3, color = "#666", fillOpacity = 0.9,
      bringToFront = TRUE
    )
  ) |>
  addLegend(
    position = "bottomright",
    pal = pal,
    values = ~BIR74,
    title = "Births (1974)",
    opacity = 0.7
  ) |>
  addMiniMap(position = "topleft") |>
  addMeasure(primaryLengthUnit = "kilometers") |>
  addScaleBar(position = "bottomleft")
```

#### Custom Tile Layers

```r
leaflet() |>
  addTiles(
    urlTemplate = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution = "Esri World Imagery"
  ) |>
  addWMSTiles(
    baseUrl = "https://ows.terrestris.de/osm/service",
    layers = "OSM-WMS",
    options = WMSTileOptions(format = "image/png", transparent = TRUE),
    group = "OSM WMS"
  ) |>
  addLayersControl(
    overlayGroups = c("OSM WMS"),
    options = layersControlOptions(collapsed = FALSE)
  ) |>
  setView(lng = 10, lat = 51, zoom = 6)
```

### mapview -- Quick Interactive Exploration

`mapview` creates interactive maps in one line, ideal for data exploration:

```r
library(mapview)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)

# One-liner
mapview(nc, zcol = "BIR74")

# Multiple layers
mapview(nc, zcol = "BIR74", layer.name = "Births") +
  mapview(st_centroid(nc), cex = "SID74", layer.name = "SID Centroids")

# With terra rasters
library(terra)
r <- rast("elevation.tif")
mapview(r, col.regions = terrain.colors(100))

# Configure globally
mapviewOptions(
  basemaps = c("CartoDB.Positron", "Esri.WorldImagery"),
  na.color = "transparent",
  layers.control.pos = "topright"
)
```

### plotly -- Interactive Statistical Maps

```r
library(plotly)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)

plot_ly(nc, split = ~NAME, color = ~BIR74,
        colors = "YlOrRd", showlegend = FALSE,
        hoverinfo = "text",
        text = ~paste(NAME, "\nBirths:", BIR74)) |>
  layout(title = "NC Births (1974)")

# Convert ggplot to plotly for interactivity
library(ggplot2)
p <- ggplot(nc) +
  geom_sf(aes(fill = BIR74, text = paste(NAME, "\nBirths:", BIR74))) +
  scale_fill_viridis_c() +
  theme_void()
ggplotly(p, tooltip = "text")
```

### Choosing a Visualization Package

| Need | Package | Mode |
|---|---|---|
| Quick data exploration | `mapview` | Interactive |
| Static publication map (thematic) | `tmap` or `mapsf` | Static |
| Grammar-of-graphics flexibility | `ggplot2` + `geom_sf()` | Static |
| Rasters in ggplot2 | `tidyterra` | Static |
| Interactive dashboard | `leaflet` or `tmap(view)` | Interactive |
| Interactive statistical plots | `plotly` | Interactive |
| 3D terrain | `rayshader` | 3D render |
| Animated maps | `tmap` + `gifski` or `gganimate` | Animation |
| Shiny integration | `leaflet` or `mapview` | Web app |

---

## Analysis Ecosystem

R has the deepest spatial statistics ecosystem of any language. This section covers the major packages with working examples.

> **See also:** [Spatial Statistics](spatial-statistics.md) | [ML for GIS](ml-gis.md)

### spdep -- Spatial Dependence

`spdep` provides spatial weights matrices and tests for spatial autocorrelation.

```r
library(spdep)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)

# Create spatial weights -- queen contiguity
nb_queen <- poly2nb(nc, queen = TRUE)
summary(nb_queen)

# Alternative: k-nearest neighbors
coords <- st_coordinates(st_centroid(nc))
nb_knn <- knearneigh(coords, k = 5) |> knn2nb()

# Convert to weights list
w_queen <- nb2listw(nb_queen, style = "W")  # row-standardized
w_binary <- nb2listw(nb_queen, style = "B") # binary

# Global Moran's I
moran.test(nc$BIR74, w_queen)
#> Moran I statistic: 0.XXX
#> p-value: 0.XXX

# Monte Carlo Moran's I (more robust)
moran.mc(nc$BIR74, w_queen, nsim = 999)

# Moran scatterplot
moran.plot(nc$BIR74, w_queen,
           labels = nc$NAME,
           xlab = "Births", ylab = "Spatial Lag of Births")

# Local Indicators of Spatial Association (LISA)
lisa <- localmoran(nc$BIR74, w_queen)
head(lisa)
#>          Ii       E.Ii    Var.Ii     Z.Ii   Pr(z != E(Ii))
#> 1  0.12345  -0.01020  0.05678  0.56789  0.57012

# Classify LISA clusters
nc$lisa_I <- lisa[, "Ii"]
nc$lisa_p <- lisa[, "Pr(z != E(Ii))"]
nc$lisa_cluster <- ifelse(nc$lisa_p > 0.05, "Not significant",
  ifelse(nc$lisa_I > 0 & nc$BIR74 > mean(nc$BIR74), "High-High",
  ifelse(nc$lisa_I > 0 & nc$BIR74 < mean(nc$BIR74), "Low-Low",
  ifelse(nc$lisa_I < 0 & nc$BIR74 > mean(nc$BIR74), "High-Low", "Low-High"))))

# Getis-Ord Gi*
gi_star <- localG(nc$BIR74, w_queen)
nc$hotspot <- as.numeric(gi_star)
```

### spatialreg -- Spatial Regression Models

`spatialreg` fits spatial econometric models (formerly part of `spdep`):

```r
library(spatialreg)
library(spdep)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
nb <- poly2nb(nc, queen = TRUE)
w <- nb2listw(nb, style = "W")

# OLS baseline
ols <- lm(BIR74 ~ NWBIR74 + AREA, data = nc)
summary(ols)

# Lagrange Multiplier tests to choose model
lm.LMtests(ols, w, test = c("LMerr", "LMlag", "RLMerr", "RLMlag"))

# Spatial Autoregressive Model (SAR / Spatial Lag)
sar <- lagsarlm(BIR74 ~ NWBIR74 + AREA, data = nc, listw = w)
summary(sar)
impacts(sar, listw = w)  # direct, indirect, total effects

# Spatial Error Model (SEM)
sem <- errorsarlm(BIR74 ~ NWBIR74 + AREA, data = nc, listw = w)
summary(sem)

# Spatial Durbin Model (SDM)
sdm <- lagsarlm(BIR74 ~ NWBIR74 + AREA, data = nc, listw = w,
                type = "mixed")
summary(sdm)
impacts(sdm, listw = w)

# Spatial Durbin Error Model (SDEM)
sdem <- errorsarlm(BIR74 ~ NWBIR74 + AREA, data = nc, listw = w,
                   etype = "emixed")
summary(sdem)
```

#### Geographically Weighted Regression (GWR)

```r
library(GWmodel)
library(sf)
library(sp)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
nc_sp <- as(nc, "Spatial")  # GWmodel still needs sp objects

# Select optimal bandwidth
bw <- bw.gwr(BIR74 ~ NWBIR74 + AREA, data = nc_sp,
             approach = "AICc", kernel = "bisquare", adaptive = TRUE)

# Fit GWR
gwr_model <- gwr.basic(BIR74 ~ NWBIR74 + AREA, data = nc_sp,
                        bw = bw, kernel = "bisquare", adaptive = TRUE)
print(gwr_model)

# Extract local coefficients
gwr_sf <- st_as_sf(gwr_model$SDF)

# Map local R-squared
library(ggplot2)
ggplot(gwr_sf) +
  geom_sf(aes(fill = Local_R2), color = "white", linewidth = 0.1) +
  scale_fill_viridis_c(name = "Local R-squared") +
  labs(title = "GWR: Spatial Variation in Model Fit") +
  theme_void()

# Multiscale GWR (MGWR) -- different bandwidth per variable
bw_mgwr <- bw.gwr.multiscale(BIR74 ~ NWBIR74 + AREA, data = nc_sp,
                              kernel = "bisquare", adaptive = TRUE)
```

### spatstat -- Point Pattern Analysis

`spatstat` is the most comprehensive point pattern analysis package in any language:

```r
library(spatstat)

# Create point pattern from sf
library(sf)
pts <- st_read("crime_locations.gpkg")
boundary <- st_read("city_boundary.gpkg")

# Convert to spatstat ppp object
win <- as.owin(boundary)
pp <- as.ppp(st_coordinates(pts), W = win)

# Summary
summary(pp)

# Intensity (kernel density)
dens <- density(pp, sigma = bw.diggle(pp))  # automatic bandwidth
plot(dens, main = "Crime Density")
contour(dens, add = TRUE)

# Quadrat test for Complete Spatial Randomness
quadrat.test(pp, nx = 5, ny = 5)

# Ripley's K-function
K <- Kest(pp, correction = "Ripley")
plot(K, main = "Ripley's K-function")
# Values above theoretical CSR line = clustering
# Values below = regularity

# L-function (variance-stabilized K)
L <- Lest(pp, correction = "Ripley")
plot(L, . - r ~ r, main = "L-function (centered)")

# Monte Carlo envelope for significance
env_L <- envelope(pp, Lest, nsim = 99, correction = "Ripley")
plot(env_L, . - r ~ r, main = "L-function with 99 MC Envelopes")

# Pair correlation function g(r)
g <- pcf(pp, correction = "Ripley")
plot(g, main = "Pair Correlation Function")

# Nearest-neighbor distance distribution
G <- Gest(pp, correction = "km")
plot(G, main = "Nearest-Neighbor Distribution")

# Marked point patterns
pp_marked <- ppp(
  x = st_coordinates(pts)[, 1],
  y = st_coordinates(pts)[, 2],
  marks = pts$crime_type,
  window = win
)
plot(split(pp_marked), main = "Crime by Type")

# Cross-type K-function
Kcross <- Kcross(pp_marked, i = "burglary", j = "assault")
plot(Kcross, main = "Cross K: Burglary vs Assault")
```

### gstat -- Geostatistics

`gstat` provides variogram modeling and kriging:

```r
library(gstat)
library(sf)
library(stars)

# Load point observations
pts <- st_read("soil_samples.gpkg")  # with column "zinc"

# Empirical variogram
v <- variogram(zinc ~ 1, data = pts)
plot(v, main = "Empirical Variogram")

# Fit variogram model
v_fit <- fit.variogram(v, vgm(psill = 50000, model = "Sph",
                               range = 1000, nugget = 10000))
plot(v, v_fit, main = "Fitted Variogram (Spherical)")

# Create prediction grid
bbox <- st_bbox(pts)
grid <- st_as_stars(st_bbox(pts), dx = 50, dy = 50)
grid <- st_crop(grid, st_read("study_area.gpkg"))

# Ordinary Kriging
ok <- krige(zinc ~ 1, locations = pts, newdata = grid, model = v_fit)
plot(ok["var1.pred"], main = "Kriging Prediction")
plot(ok["var1.var"], main = "Kriging Variance")

# Universal Kriging (with trend)
v_uk <- variogram(zinc ~ sqrt(dist), data = pts)
v_uk_fit <- fit.variogram(v_uk, vgm("Exp"))
uk <- krige(zinc ~ sqrt(dist), locations = pts, newdata = grid, model = v_uk_fit)

# Indicator Kriging (probability of exceeding threshold)
pts$zinc_high <- as.numeric(pts$zinc > 500)
v_ik <- variogram(zinc_high ~ 1, data = pts)
v_ik_fit <- fit.variogram(v_ik, vgm("Sph"))
ik <- krige(zinc_high ~ 1, locations = pts, newdata = grid, model = v_ik_fit)
plot(ik["var1.pred"], main = "P(zinc > 500)")

# Cross-validation
cv <- krige.cv(zinc ~ 1, locations = pts, model = v_fit, nfold = 10)
summary(cv$residual)
sqrt(mean(cv$residual^2))  # RMSE
```

### rgeoda -- GeoDa in R

`rgeoda` brings the full GeoDa spatial analysis engine to R:

```r
library(rgeoda)
library(sf)

nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
gda <- sf_to_geoda(nc)

# Spatial weights
w_queen <- queen_weights(gda)
w_rook  <- rook_weights(gda)
w_knn   <- knn_weights(gda, k = 6)
w_dist  <- distance_weights(gda, dist_thres = 0.5)

# LISA with rgeoda
lisa <- local_moran(w_queen, nc["BIR74"])
nc$cluster <- lisa_clusters(lisa)
nc$pvalue  <- lisa_pvalues(lisa)
nc$labels  <- lisa_labels(lisa)

# Local Geary
geary <- local_geary(w_queen, nc["BIR74"])
nc$geary_cluster <- lisa_clusters(geary)

# SKATER regionalization
skater_regions <- skater(5, w_queen, nc[c("BIR74", "SID74", "NWBIR74")])
nc$region <- skater_regions$Clusters

# Max-p regionalization
maxp <- maxp_greedy(w_queen, nc[c("BIR74", "SID74")],
                    bound_variable = nc[["AREA"]],
                    min_bound = 0.2)
nc$maxp_region <- maxp$Clusters
```

---

## Network and Transport Analysis

> **See also:** [Geocoding and Routing](../tools/geocoding-routing.md) | [Network Graph Visualization](../visualization/network-graph-visualization.md)

### sfnetworks -- Tidy Spatial Networks

`sfnetworks` combines `sf` geometries with `tidygraph` for network analysis:

```r
library(sfnetworks)
library(sf)
library(tidygraph)
library(dplyr)

# Create network from road lines
roads <- st_read("roads.gpkg")
net <- as_sfnetwork(roads, directed = FALSE)
net
#> # A sfnetwork with XXX nodes and XXX edges
#> # An undirected simple graph with 1 component

# Clean network topology
net_clean <- net |>
  convert(to_spatial_subdivision) |>     # split edges at intersections
  convert(to_spatial_smooth) |>          # remove pseudo-nodes
  activate("edges") |>
  mutate(weight = edge_length())         # add length as weight

# Shortest path
from_node <- st_nearest_feature(
  st_sfc(st_point(c(10.0, 51.0)), crs = 4326),
  st_as_sf(net_clean, "nodes")
)
to_node <- st_nearest_feature(
  st_sfc(st_point(c(10.5, 51.3)), crs = 4326),
  st_as_sf(net_clean, "nodes")
)

path <- st_network_paths(net_clean, from = from_node, to = to_node)
path_edges <- net_clean |>
  activate("edges") |>
  slice(path$edge_paths[[1]]) |>
  st_as_sf()

# Network centrality
net_central <- net_clean |>
  activate("nodes") |>
  mutate(
    betweenness = centrality_betweenness(weights = weight),
    closeness   = centrality_closeness(weights = weight)
  )

# Isochrone (service area) via network distance
iso <- net_clean |>
  activate("edges") |>
  filter(
    node_distance_from(from_node, weights = weight) <= 5000  # 5 km
  ) |>
  st_as_sf("edges")
```

### dodgr -- Distances on Directed Graphs

`dodgr` is optimized for fast pairwise routing on weighted graphs:

```r
library(dodgr)

# From OSM data
hampi <- dodgr_streetnet("Hampi, India")
graph <- weight_streetnet(hampi, wt_profile = "foot")

# Many-to-many distances
from <- graph$from_id[1:100]
to   <- graph$to_id[1:100]
d    <- dodgr_dists(graph, from = from, to = to)

# Flows (origin-destination)
flows <- dodgr_flows_aggregate(graph, from = from, to = to,
                                flows = matrix(1, 100, 100))

# Isochrone
iso_pts <- dodgr_isochrones(graph, from = from[1], tlim = c(600, 1200, 1800))
```

### r5r -- Multimodal Accessibility

`r5r` wraps Conveyal's R5 routing engine for multimodal transport analysis:

```r
library(r5r)

# Setup -- requires Java 21 and GTFS + OSM data in a directory
r5_core <- setup_r5(data_path = "r5_data/")  # contains GTFS .zip + .osm.pbf

# Define origins and destinations
origins <- data.frame(id = 1:10, lon = runif(10, -46.7, -46.6),
                      lat = runif(10, -23.6, -23.5))
destinations <- origins  # same set

# Travel time matrix
ttm <- travel_time_matrix(
  r5_core,
  origins = origins,
  destinations = destinations,
  mode = c("WALK", "TRANSIT"),
  departure_datetime = as.POSIXct("2024-03-15 08:00:00"),
  max_trip_duration = 60  # minutes
)

# Accessibility -- number of opportunities reachable
access <- accessibility(
  r5_core,
  origins = origins,
  destinations = destinations,
  opportunities_colname = "id",
  mode = c("WALK", "TRANSIT"),
  departure_datetime = as.POSIXct("2024-03-15 08:00:00"),
  cutoffs = c(15, 30, 45, 60),
  decay_function = "step"
)

# Detailed itineraries
itin <- detailed_itineraries(
  r5_core,
  origins = origins[1, ],
  destinations = destinations[2, ],
  mode = c("WALK", "TRANSIT"),
  departure_datetime = as.POSIXct("2024-03-15 08:00:00")
)
plot(itin$geometry)

stop_r5(r5_core)
```

### stplanr -- Sustainable Transport Planning

```r
library(stplanr)
library(sf)

# Create desire lines from OD matrix
od <- data.frame(
  origin = c("A", "A", "B"),
  destination = c("B", "C", "C"),
  trips = c(100, 50, 75)
)
zones <- st_read("zones.gpkg")  # must have matching zone IDs
desire_lines <- od2line(od, zones)
plot(desire_lines, lwd = desire_lines$trips / 20)

# Route desire lines along the network
routes <- route(
  l = desire_lines,
  route_fun = route_osrm  # or route_dodgr
)

# Route network -- aggregate flows on shared road segments
rnet <- overline(routes, attrib = "trips")
plot(rnet, lwd = rnet$trips / 10)

# Uptake model -- shift from car to cycling
desire_lines$pcycle <- uptake_pct_govtarget_2020(
  distance = as.numeric(st_length(desire_lines)),
  gradient = 0.02
)
```

### osmdata -- OpenStreetMap in R

```r
library(osmdata)
library(sf)

# Query OSM via Overpass API
bbox <- c(-73.99, 40.74, -73.97, 40.76)  # Manhattan subset

# Get restaurants
restaurants <- opq(bbox) |>
  add_osm_feature(key = "amenity", value = "restaurant") |>
  osmdata_sf()
restaurants$osm_points

# Get road network
roads <- opq(bbox) |>
  add_osm_feature(key = "highway") |>
  osmdata_sf()
roads$osm_lines

# Get building footprints
buildings <- opq(bbox) |>
  add_osm_feature(key = "building") |>
  osmdata_sf()
buildings$osm_polygons

# Available features
available_features()
available_tags("amenity")
```

---

## Remote Sensing in R

> **See also:** [Remote Sensing Tools](../tools/remote-sensing.md) | [Python Stack RS Section](python-stack.md)

### terra for Remote Sensing

`terra` is the primary package for raster-based remote sensing workflows:

```r
library(terra)

# Load multispectral imagery
img <- rast("sentinel2_bands.tif")
names(img) <- c("B02", "B03", "B04", "B08", "B11", "B12")

# Spectral indices
ndvi <- (img$B08 - img$B04) / (img$B08 + img$B04)
ndwi <- (img$B03 - img$B08) / (img$B03 + img$B08)
nbr  <- (img$B08 - img$B12) / (img$B08 + img$B12)

# Unsupervised classification (k-means)
set.seed(42)
vals <- values(img)
vals_clean <- vals[complete.cases(vals), ]
km <- kmeans(scale(vals_clean), centers = 5, nstart = 25)
classified <- img[[1]]
values(classified)[complete.cases(vals)] <- km$cluster
names(classified) <- "class"

# Supervised classification
library(ranger)
training <- vect("training_polygons.gpkg")
training_vals <- extract(img, training)
training_df <- merge(training_vals, as.data.frame(training), by.x = "ID",
                     by.y = seq_len(nrow(training)))

rf <- ranger(class ~ B02 + B03 + B04 + B08 + B11 + B12,
             data = training_df, num.trees = 500)

pred <- predict(img, rf, fun = function(model, ...) {
  predict(model, ...)$predictions
})

# Accuracy assessment
test <- vect("test_polygons.gpkg")
test_vals <- extract(pred, test)
caret::confusionMatrix(factor(test_vals$lyr1), factor(test$class))
```

### sits -- Satellite Image Time Series

`sits` provides a full pipeline for satellite image time series analysis, from data access to classification:

```r
library(sits)

# Access data from cloud collections
s2_cube <- sits_cube(
  source = "AWS",
  collection = "SENTINEL-2-L2A",
  tiles = "20LKP",
  bands = c("B02", "B03", "B04", "B08", "B11", "B12", "CLOUD"),
  start_date = "2024-01-01",
  end_date = "2024-12-31"
)

# Regularize time series (monthly composites)
reg_cube <- sits_regularize(
  cube = s2_cube,
  period = "P1M",
  res = 20,
  output_dir = "regular/",
  multicores = 4
)

# Extract time series for training samples
samples <- sits_get_data(
  cube = reg_cube,
  samples = "training_points.csv"  # lon, lat, start_date, end_date, label
)

# Train a temporal deep learning model
model <- sits_train(
  samples = samples,
  ml_method = sits_tempcnn()  # or sits_lighttae(), sits_tae()
)

# Classify the entire cube
classified <- sits_classify(
  data = reg_cube,
  ml_model = model,
  output_dir = "classified/",
  multicores = 8,
  memsize = 16
)

# Post-processing
smoothed <- sits_smooth(classified, output_dir = "smooth/")
labeled  <- sits_label_classification(smoothed, output_dir = "label/")

# Accuracy
sits_accuracy(classified, validation = "validation_points.csv")
```

### openeo -- Cloud-Based Earth Observation

```r
library(openeo)

# Connect to an openEO backend
con <- connect("https://openeo.cloud")
login()

# List available collections
collections <- list_collections()

# Load Sentinel-2 data
datacube <- p$load_collection(
  id = "SENTINEL2_L2A",
  spatial_extent = list(west = 10, south = 51, east = 11, north = 52),
  temporal_extent = c("2024-01-01", "2024-12-31"),
  bands = c("B04", "B08")
)

# Compute NDVI on the backend
ndvi <- p$reduce_dimension(
  data = datacube,
  dimension = "bands",
  reducer = function(x, context) {
    (x[2] - x[1]) / (x[2] + x[1])
  }
)

# Temporal aggregation
ndvi_monthly <- p$aggregate_temporal_period(
  data = ndvi,
  period = "month",
  reducer = "median"
)

# Download result
result <- p$save_result(ndvi_monthly, format = "GTiff")
job <- create_job(graph = result, title = "NDVI monthly")
start_job(job)
download_results(job, "output/")
```

### sen2r -- Sentinel-2 Processing

```r
library(sen2r)

# GUI-based workflow
# sen2r()  # opens Shiny interface

# Programmatic workflow
sen2r(
  gui = FALSE,
  step_atmcorr = "l2a",        # atmospheric correction
  extent = "study_area.gpkg",
  timewindow = c("2024-06-01", "2024-08-31"),
  list_prods = c("BOA"),        # bottom-of-atmosphere reflectance
  list_indices = c("NDVI", "EVI"),
  mask_type = "cloud_and_shadow",
  max_mask = 20,                # skip scenes with >20% clouds
  path_out = "sen2r_output/",
  parallel = 4
)
```

### MODISTools and getSpatialData

```r
# MODISTools -- access MODIS data via the AppEEARS API
library(MODISTools)

# List available products
products <- mt_products()

# Download MODIS NDVI time series for a site
ndvi_ts <- mt_subset(
  product = "MOD13Q1",      # 250m 16-day NDVI
  lat = 40.748,
  lon = -73.985,
  band = "250m_16_days_NDVI",
  start = "2024-01-01",
  end = "2024-12-31",
  km_lr = 1,
  km_ab = 1,
  site_name = "NYC"
)

# getSpatialData -- unified access to multiple satellite sources
library(getSpatialData)
set_aoi(st_read("study_area.gpkg"))
set_archive("satellite_archive/")

# Search for Sentinel-2 scenes
records <- get_records(
  time_range = c("2024-06-01", "2024-06-30"),
  products = "Sentinel-2"
)
records <- calc_cloudcov(records)
records <- records[records$cloudcov < 20, ]

# Preview and download
plot_records(records)
records <- check_availability(records)
get_data(records)
```

---

## Machine Learning for Geospatial

> **See also:** [ML for GIS](ml-gis.md) | [AI/ML Geospatial Tools](../tools/ai-ml-geospatial.md)

### tidymodels + spatialsample -- Spatial Cross-Validation

Standard k-fold CV leaks spatial information. `spatialsample` provides spatially-aware resampling:

```r
library(tidymodels)
library(spatialsample)
library(sf)

# Load data
pts <- st_read("soil_samples.gpkg") |>
  st_transform(32633)

# Create spatial CV folds
set.seed(42)
spatial_folds <- spatial_block_cv(pts, v = 10)
# Or: spatial_leave_location_out_cv, spatial_buffer_vfold_cv, spatial_clustering_cv

autoplot(spatial_folds)  # visualize fold assignments

# Define recipe
rec <- recipe(zinc ~ ., data = st_drop_geometry(pts)) |>
  step_normalize(all_predictors()) |>
  step_impute_knn(all_predictors())

# Define model
rf_spec <- rand_forest(trees = 500, mtry = tune(), min_n = tune()) |>
  set_engine("ranger") |>
  set_mode("regression")

# Workflow
wf <- workflow() |>
  add_recipe(rec) |>
  add_model(rf_spec)

# Tune with spatial CV
tune_results <- tune_grid(
  wf,
  resamples = spatial_folds,
  grid = 20,
  metrics = metric_set(rmse, rsq, mae)
)

# Compare spatial vs non-spatial CV
nonspatial_folds <- vfold_cv(st_drop_geometry(pts), v = 10)
tune_nonspatial <- tune_grid(wf, resamples = nonspatial_folds,
                              grid = 20, metrics = metric_set(rmse, rsq))

# Spatial CV gives more realistic (usually worse) performance estimates
collect_metrics(tune_results)
collect_metrics(tune_nonspatial)

# Finalize and fit
best_params <- select_best(tune_results, metric = "rmse")
final_wf <- finalize_workflow(wf, best_params)
final_fit <- fit(final_wf, data = st_drop_geometry(pts))

# Predict on new data
predictions <- predict(final_fit, new_data = st_drop_geometry(new_pts))
```

### mlr3 + mlr3spatial

`mlr3` is an alternative ML framework with dedicated spatial extensions:

```r
library(mlr3)
library(mlr3spatial)
library(mlr3spatiotempcv)
library(mlr3learners)

# Create spatial task
task <- as_task_regr_st(
  st_drop_geometry(pts) |> cbind(st_coordinates(pts)),
  target = "zinc",
  coordinate_names = c("X", "Y"),
  crs = st_crs(pts)
)

# Spatial CV resampling
resampling <- rsmp("spcv_block", folds = 10)

# Learner
learner <- lrn("regr.ranger", num.trees = 500)

# Benchmark with spatial and non-spatial CV
design <- benchmark_grid(
  tasks = task,
  learners = learner,
  resamplings = list(
    rsmp("spcv_block", folds = 10),
    rsmp("spcv_coords", folds = 10),
    rsmp("cv", folds = 10)          # standard (non-spatial)
  )
)
bmr <- benchmark(design)
bmr$aggregate(msr("regr.rmse"))
```

### blockCV -- Spatial Blocking

`blockCV` provides multiple spatial blocking strategies for model evaluation:

```r
library(blockCV)
library(sf)

# Spatial block CV
sb <- cv_spatial(
  x = pts,
  column = "zinc",
  r = NULL,                    # auto-range from variogram
  k = 10,                     # number of folds
  selection = "random",
  iteration = 100,
  biomod2 = FALSE
)

# Buffered leave-one-out
bf <- cv_buffer(
  x = pts,
  column = "zinc",
  size = 5000                  # buffer radius in meters
)

# Nearest-neighbor distance matching
nn <- cv_nndm(
  x = pts,
  column = "zinc",
  modeldomain = study_area,    # extent of prediction area
  samplesize = 1000
)

# Plot blocks
cv_plot(sb, pts)
```

### Common ML Backends for Spatial Problems

```r
# Random Forest via ranger (fast, memory-efficient)
library(ranger)
rf <- ranger(zinc ~ ., data = train_df, num.trees = 500,
             importance = "permutation")
rf$variable.importance

# XGBoost
library(xgboost)
dtrain <- xgb.DMatrix(data = as.matrix(train_df[, -1]),
                       label = train_df$zinc)
xgb <- xgb.train(
  params = list(objective = "reg:squarederror", eta = 0.1,
                max_depth = 6, subsample = 0.8),
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain)
)

# torch for R (deep learning)
library(torch)
library(luz)  # high-level API

net <- nn_module(
  initialize = function(n_features) {
    self$fc1 <- nn_linear(n_features, 128)
    self$fc2 <- nn_linear(128, 64)
    self$fc3 <- nn_linear(64, 1)
    self$dropout <- nn_dropout(0.3)
  },
  forward = function(x) {
    x |>
      self$fc1() |> nnf_relu() |> self$dropout() |>
      self$fc2() |> nnf_relu() |> self$dropout() |>
      self$fc3()
  }
)

fitted <- net |>
  setup(loss = nnf_mse_loss, optimizer = optim_adam) |>
  set_hparams(n_features = ncol(train_matrix)) |>
  fit(
    data = list(train_matrix, train_labels),
    epochs = 100,
    valid_data = list(val_matrix, val_labels)
  )
```

---

## Big Data and Cloud Workflows

> **See also:** [Cloud Platforms](../tools/cloud-platforms.md) | [ETL and Data Engineering](../tools/etl-data-engineering.md)

### arrow + geoarrow -- GeoParquet

GeoParquet is the emerging standard for large vector datasets. `arrow` + `geoarrow` provide fast, memory-mapped access:

```r
library(arrow)
library(sf)
library(dplyr)

# Write GeoParquet
nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
sfarrow::st_write_parquet(nc, "nc.parquet")

# Read GeoParquet
nc_back <- sfarrow::st_read_parquet("nc.parquet")

# Large-scale: use arrow datasets for out-of-memory data
ds <- open_dataset("buildings_partitioned/",
                   format = "parquet")

# Push-down filtering (only reads matching partitions/row groups)
result <- ds |>
  filter(state == "NY", area > 100) |>
  select(id, area, geometry) |>
  collect() |>
  sfarrow::st_as_sf()

# GeoParquet with partitioning
sfarrow::st_write_parquet(
  buildings,
  "buildings_partitioned/",
  partitioning = "state"
)
```

### DBI + RPostgis -- Spatial Databases

```r
library(DBI)
library(RPostgres)
library(sf)

# Connect to PostGIS
con <- dbConnect(
  Postgres(),
  dbname = "gisdb",
  host = "localhost",
  port = 5432,
  user = "gis_user",
  password = Sys.getenv("PGPASSWORD")
)

# Read spatial data from PostGIS
parcels <- st_read(con, query = "
  SELECT id, owner, area_sqm, geom
  FROM parcels
  WHERE ST_Intersects(geom, ST_MakeEnvelope(10, 51, 11, 52, 4326))
")

# Write sf to PostGIS
st_write(parcels_new, con, "parcels", append = TRUE)

# Use dbplyr for lazy SQL generation
library(dbplyr)
parcels_tbl <- tbl(con, "parcels")

large_parcels <- parcels_tbl |>
  filter(area_sqm > 10000) |>
  select(id, owner, area_sqm) |>
  collect()  # only materializes at collect()

dbDisconnect(con)
```

### targets -- Pipeline Orchestration

`targets` orchestrates complex spatial workflows with caching and dependency tracking:

```r
# _targets.R
library(targets)
library(tarchetypes)

tar_option_set(
  packages = c("sf", "terra", "dplyr", "ranger"),
  format = "qs"  # fast serialization
)

list(
  # Data ingestion
  tar_target(raw_points, st_read("data/samples.gpkg")),
  tar_target(raw_raster, rast("data/covariates.tif")),
  tar_target(study_area, st_read("data/boundary.gpkg")),

  # Preprocessing
  tar_target(
    covariates,
    {
      vals <- extract(raw_raster, raw_points)
      cbind(st_drop_geometry(raw_points), vals)
    }
  ),

  # Spatial CV
  tar_target(
    cv_folds,
    spatialsample::spatial_block_cv(raw_points, v = 10)
  ),

  # Model training
  tar_target(
    model,
    {
      library(ranger)
      ranger(target ~ ., data = covariates, num.trees = 500)
    }
  ),

  # Prediction map
  tar_target(
    prediction_map,
    {
      pred <- predict(raw_raster, model,
                      fun = function(m, ...) predict(m, ...)$predictions)
      writeRaster(pred, "output/prediction.tif", overwrite = TRUE)
      pred
    },
    format = "file"
  )
)
```

Run the pipeline:

```r
# In R console
targets::tar_make()       # run pipeline
targets::tar_visnetwork() # visualize dependency graph
targets::tar_read(model)  # read cached result
```

### Parallel Processing -- future and furrr

```r
library(future)
library(furrr)
library(terra)

# Set up parallel backend
plan(multisession, workers = 8)

# Parallel raster processing across tiles
tile_files <- list.files("tiles/", pattern = "\\.tif$", full.names = TRUE)

results <- future_map(tile_files, function(f) {
  library(terra)
  r <- rast(f)
  ndvi <- (r[["nir"]] - r[["red"]]) / (r[["nir"]] + r[["red"]])
  outfile <- gsub("tiles/", "output/ndvi_", f)
  writeRaster(ndvi, outfile, overwrite = TRUE)
  outfile
}, .progress = TRUE)

# Parallel spatial operations on sf
library(sf)
large_sf <- st_read("large_dataset.gpkg")
chunks <- split(large_sf, rep(1:8, length.out = nrow(large_sf)))

buffered <- future_map(chunks, function(chunk) {
  st_buffer(chunk, 100)
}, .progress = TRUE) |>
  bind_rows()

plan(sequential)  # reset
```

---

## Reproducibility

### renv -- Dependency Management

```r
# Initialize renv in your project
renv::init()

# Install packages as normal
install.packages(c("sf", "terra", "tmap", "spdep"))

# Snapshot current state
renv::snapshot()
# Creates renv.lock with exact versions:
# {
#   "R": { "Version": "4.4.0" },
#   "Packages": {
#     "sf": { "Package": "sf", "Version": "1.0-16", ... },
#     ...
#   }
# }

# On another machine / in CI
renv::restore()

# Update packages
renv::update()
renv::snapshot()  # update lockfile
```

### targets -- Reproducible Pipelines

See the [targets section above](#targets----pipeline-orchestration) for a complete spatial pipeline example. Key benefits:

- Only re-runs steps whose inputs have changed
- Caches intermediate results
- Visualizes the dependency graph
- Integrates with `crew` for HPC/cloud execution

```r
# Check pipeline status
tar_outdated()     # which targets need re-running?
tar_visnetwork()   # dependency graph
tar_manifest()     # list all targets
```

### Quarto -- Reports with Maps

Quarto supports embedded R maps in reproducible documents:

````markdown
---
title: "Spatial Analysis Report"
format:
  html:
    code-fold: true
    toc: true
execute:
  echo: true
  warning: false
---

```{r}
#| label: setup
library(sf)
library(tmap)
library(ggplot2)
```

## Study Area

```{r}
#| label: fig-study-area
#| fig-cap: "Study area with sample locations"
nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
tmap_mode("plot")
tm_shape(nc) +
  tm_polygons(fill = "BIR74", fill.scale = tm_scale_continuous()) +
  tm_title("North Carolina")
```

## Interactive Map

```{r}
#| label: fig-interactive
tmap_mode("view")
tm_shape(nc) +
  tm_polygons(fill = "BIR74")
```
````

Render with:

```bash
quarto render report.qmd
quarto render report.qmd --to pdf  # requires LaTeX
```

### Docker -- rocker/geospatial

The `rocker/geospatial` image includes R, RStudio, and all spatial system dependencies:

```dockerfile
FROM rocker/geospatial:4.4

# Install additional R packages
RUN install2.r --error --skipmissing \
  tmap mapview spdep spatstat gstat GWmodel rgeoda \
  sfnetworks dodgr r5r \
  tidymodels spatialsample ranger xgboost \
  targets tarchetypes

# Copy project files
COPY . /home/rstudio/project
WORKDIR /home/rstudio/project

# Restore renv
RUN R -e "renv::restore()"

# Run analysis
CMD ["R", "-e", "targets::tar_make()"]
```

Build and run:

```bash
docker build -t spatial-analysis .
docker run -e PASSWORD=secret -p 8787:8787 spatial-analysis
# Access RStudio at http://localhost:8787
```

### Testing Spatial Code

```r
library(testthat)

test_that("buffer produces correct geometry", {
  pt <- st_sfc(st_point(c(0, 0)), crs = 32633)
  buf <- st_buffer(pt, 1000)

  expect_s3_class(buf, "sfc")
  expect_equal(length(buf), 1)
  # Area should be close to pi * r^2

  expect_equal(as.numeric(st_area(buf)), pi * 1000^2, tolerance = 100)
})

test_that("spatial join preserves all left-side features", {
  pts <- st_as_sf(data.frame(x = 1:10, y = 1:10, val = letters[1:10]),
                  coords = c("x", "y"), crs = 32633)
  polys <- st_buffer(pts[1:3, ], 2)

  joined <- st_join(pts, polys, left = TRUE)
  expect_equal(nrow(joined), nrow(pts))
})

test_that("raster extraction returns expected dimensions", {
  r <- terra::rast(nrows = 10, ncols = 10, vals = 1:100)
  pts <- terra::vect(
    data.frame(x = c(0.5, 1.5), y = c(0.5, 1.5)),
    geom = c("x", "y")
  )
  extracted <- terra::extract(r, pts)
  expect_equal(nrow(extracted), 2)
  expect_true("lyr.1" %in% names(extracted))
})
```

Run with:

```r
testthat::test_dir("tests/")
# or via devtools
devtools::test()
```

---

## Environment Setup

### CRAN vs r-universe

| Source | Pros | Cons |
|---|---|---|
| **CRAN** | Stable, tested, default | Slower updates, strict policies |
| **r-universe** | Bleeding-edge, dev versions, fast | May be less stable |

```r
# Install from CRAN (default)
install.packages("sf")

# Install development version from r-universe
install.packages("sf", repos = "https://r-spatial.r-universe.dev")

# Install from GitHub (last resort)
remotes::install_github("r-spatial/sf")
```

### System Dependencies by Platform

#### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y \
  libgdal-dev \
  libgeos-dev \
  libproj-dev \
  libudunits2-dev \
  libsqlite3-dev \
  libssl-dev \
  libcurl4-openssl-dev \
  libfontconfig1-dev \
  libharfbuzz-dev \
  libfribidi-dev \
  libfreetype6-dev \
  libpng-dev \
  libtiff5-dev \
  libjpeg-dev \
  libnetcdf-dev \
  libhdf5-dev

# Check GDAL version (need 3.1+)
gdal-config --version
geos-config --version
```

#### Fedora / RHEL

```bash
sudo dnf install -y \
  gdal-devel \
  geos-devel \
  proj-devel \
  udunits2-devel \
  sqlite-devel \
  openssl-devel \
  libcurl-devel \
  netcdf-devel \
  hdf5-devel
```

#### macOS (Homebrew)

```bash
brew install gdal geos proj udunits pkg-config netcdf hdf5
# For sf, also ensure:
brew install openssl
```

#### Windows

On Windows, `sf` and `terra` bundle their own GDAL/GEOS/PROJ binaries. No manual installation is typically needed. Just run `install.packages("sf")`.

For advanced users needing custom GDAL builds, use **rtools** and compile from source.

### Docker for R Spatial

```bash
# Pre-built images with all spatial dependencies
docker pull rocker/geospatial:4.4        # R 4.4 + sf/terra/stars
docker pull rocker/geospatial:latest      # latest R

# Run RStudio in browser
docker run -e PASSWORD=mysecret -p 8787:8787 rocker/geospatial:4.4
# Open http://localhost:8787 (user: rstudio, password: mysecret)

# Run a script non-interactively
docker run -v $(pwd):/work -w /work rocker/geospatial:4.4 \
  Rscript analysis.R
```

### IDE Options -- RStudio and Positron

| IDE | Best For | Map Preview | Notes |
|---|---|---|---|
| **RStudio** | General R work | Viewer pane (leaflet, tmap) | Most mature |
| **Positron** | R + Python, modern UI | Viewer pane | By Posit, early stage |
| **VS Code + R ext** | Polyglot developers | Via browser | Use `languageserver` |
| **Jupyter + IRkernel** | Notebook-style | Inline | `install.packages("IRkernel")` |

---

## Cookbook -- 15+ Spatial Recipes

### Recipe 1: Read and Write Any Vector Format

```r
library(sf)

# GeoPackage
gpkg <- st_read("data.gpkg", layer = "parcels")
st_write(gpkg, "output.gpkg", layer = "parcels")

# Shapefile
shp <- st_read("data.shp")

# GeoJSON
geojson <- st_read("data.geojson")
st_write(geojson, "output.geojson")

# GeoParquet
parquet <- sfarrow::st_read_parquet("data.parquet")
sfarrow::st_write_parquet(parquet, "output.parquet")

# FlatGeobuf
fgb <- st_read("data.fgb")
st_write(fgb, "output.fgb")

# CSV with coordinates
csv <- read.csv("points.csv")
pts <- st_as_sf(csv, coords = c("longitude", "latitude"), crs = 4326)

# KML
kml <- st_read("data.kml")

# From WKT column
df <- data.frame(id = 1:2, wkt = c("POINT(0 0)", "POINT(1 1)"))
sf_from_wkt <- st_as_sf(df, wkt = "wkt", crs = 4326)

# List all available drivers
st_drivers()
```

### Recipe 2: Spatial Joins

```r
library(sf)
library(dplyr)

# Point-in-polygon join
pts <- st_read("sample_points.gpkg")
polys <- st_read("admin_boundaries.gpkg")

# Left join: keep all points, add polygon attributes
pts_with_admin <- st_join(pts, polys, join = st_within, left = TRUE)

# Nearest feature join (with distance)
pts_nearest <- st_join(pts, polys, join = st_nearest_feature)

# Join within distance
pts_near <- st_join(pts, polys,
                    join = st_is_within_distance, dist = 500)

# Count points per polygon
counts <- pts |>
  st_join(polys, join = st_within) |>
  st_drop_geometry() |>
  count(admin_name)

polys_with_count <- polys |>
  left_join(counts, by = "admin_name")
```

### Recipe 3: Raster Extraction by Polygons (Zonal Statistics)

```r
library(terra)
library(sf)

r <- rast("temperature.tif")
zones <- vect("districts.gpkg")

# Basic extraction
vals <- extract(r, zones, fun = mean, na.rm = TRUE)

# Multiple statistics
stats <- extract(r, zones, fun = function(x, ...) {
  c(mean = mean(x, na.rm = TRUE),
    sd = sd(x, na.rm = TRUE),
    min = min(x, na.rm = TRUE),
    max = max(x, na.rm = TRUE),
    median = median(x, na.rm = TRUE))
})

# Exact extraction (handles partial cell coverage)
library(exactextractr)
zones_sf <- st_read("districts.gpkg")
exact_stats <- exact_extract(raster::raster(r), zones_sf,
                              fun = c("mean", "stdev", "median",
                                      "quantile(0.25)", "quantile(0.75)"))
```

### Recipe 4: Interpolation (IDW and Kriging)

```r
library(gstat)
library(sf)
library(stars)

pts <- st_read("weather_stations.gpkg")

# Create prediction grid
grid <- st_as_stars(st_bbox(pts), dx = 1000, dy = 1000)

# IDW interpolation
idw_result <- idw(temperature ~ 1, locations = pts,
                  newdata = grid, idp = 2)
plot(idw_result["var1.pred"], main = "IDW Temperature")

# Kriging (see gstat section above for variogram fitting)
v <- variogram(temperature ~ 1, data = pts)
v_fit <- fit.variogram(v, vgm("Sph"))
ok <- krige(temperature ~ 1, locations = pts,
            newdata = grid, model = v_fit)
plot(ok["var1.pred"], main = "Kriged Temperature")
```

### Recipe 5: Time Series Raster Analysis

```r
library(terra)

# Load monthly NDVI stack
files <- list.files("ndvi_monthly/", pattern = "\\.tif$",
                    full.names = TRUE) |> sort()
ndvi_stack <- rast(files)
time(ndvi_stack) <- seq(as.Date("2024-01-01"), by = "month", length.out = 12)

# Per-pixel temporal mean
ndvi_mean <- app(ndvi_stack, mean, na.rm = TRUE)

# Per-pixel trend (slope of linear regression)
ndvi_trend <- app(ndvi_stack, function(x) {
  if (all(is.na(x))) return(NA)
  coef(lm(x ~ seq_along(x)))[2]
})

# Phenology: day of peak NDVI
peak_month <- app(ndvi_stack, which.max)

# Seasonal decomposition per pixel
# (compute for a single pixel as example)
ts_pixel <- as.numeric(ndvi_stack[50, 50])
ts_obj <- ts(ts_pixel, start = c(2024, 1), frequency = 12)
decomp <- stl(ts_obj, s.window = "periodic")
plot(decomp)
```

### Recipe 6: Download and Process OSM Data

```r
library(osmdata)
library(sf)

# Download park polygons for a city
parks <- opq("Portland, Oregon") |>
  add_osm_feature(key = "leisure", value = "park") |>
  osmdata_sf()

park_polys <- parks$osm_polygons |>
  select(name, osm_id)

# Download the road network
roads <- opq("Portland, Oregon") |>
  add_osm_feature(key = "highway",
                  value = c("primary", "secondary", "tertiary")) |>
  osmdata_sf()

road_lines <- roads$osm_lines

# Calculate total park area
total_area <- sum(st_area(park_polys)) |> units::set_units("km^2")

# Save
st_write(park_polys, "portland_parks.gpkg")
st_write(road_lines, "portland_roads.gpkg")
```

### Recipe 7: Geocoding and Reverse Geocoding

```r
library(tidygeocoder)

# Forward geocoding
addresses <- data.frame(
  addr = c("1600 Pennsylvania Ave, Washington DC",
           "350 Fifth Avenue, New York, NY")
)
geocoded <- addresses |>
  geocode(addr, method = "osm")  # or "google", "census", etc.

# Reverse geocoding
coords <- data.frame(lat = c(40.748, 38.897), lon = c(-73.985, -77.036))
reversed <- coords |>
  reverse_geocode(lat = lat, long = lon, method = "osm")
```

### Recipe 8: Reproject Raster and Vector

```r
library(sf)
library(terra)

# Vector reprojection
nc <- st_read(system.file("shape/nc.shp", package = "sf"), quiet = TRUE)
nc_utm <- st_transform(nc, "EPSG:32617")   # UTM zone 17N
nc_aea <- st_transform(nc, "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=37.5 +lon_0=-96")

# Raster reprojection
r <- rast("dem_wgs84.tif")
r_utm <- project(r, "EPSG:32617", method = "bilinear")
r_aea <- project(r, crs(nc_aea), method = "bilinear")

# Match raster to another raster's grid
template <- rast("target_grid.tif")
r_matched <- project(r, template, method = "bilinear")
```

### Recipe 9: Spatial Clustering

```r
library(sf)
library(dbscan)

pts <- st_read("events.gpkg") |> st_transform(32633)
coords <- st_coordinates(pts)

# DBSCAN
db <- dbscan(coords, eps = 500, minPts = 5)
pts$cluster_dbscan <- db$cluster

# HDBSCAN
hdb <- hdbscan(coords, minPts = 5)
pts$cluster_hdbscan <- hdb$cluster

# K-means on coordinates
km <- kmeans(coords, centers = 10, nstart = 25)
pts$cluster_kmeans <- km$cluster

# Spatially constrained clustering (rgeoda)
library(rgeoda)
gda <- sf_to_geoda(pts)
w <- queen_weights(gda)
skater_cl <- skater(5, w, pts[c("var1", "var2", "var3")])
pts$region <- skater_cl$Clusters
```

### Recipe 10: Create Isochrones

```r
library(r5r)

core <- setup_r5("r5_data/")

# Define origin
origin <- data.frame(id = "home", lon = -73.985, lat = 40.748)

# Calculate isochrones
iso <- isochrone(
  core,
  origins = origin,
  mode = c("WALK", "TRANSIT"),
  departure_datetime = as.POSIXct("2024-03-15 08:00:00"),
  cutoffs = c(15, 30, 45, 60)  # minutes
)

# Plot
library(ggplot2)
ggplot(iso) +
  geom_sf(aes(fill = factor(isochrone)), alpha = 0.5) +
  scale_fill_brewer(palette = "YlOrRd", direction = -1,
                    name = "Minutes") +
  theme_void()

stop_r5(core)
```

### Recipe 11: Viewshed Analysis

```r
library(terra)

dem <- rast("elevation.tif")

# Simple viewshed from a single point
observer <- vect(data.frame(x = 500000, y = 5500000),
                 geom = c("x", "y"), crs = crs(dem))
viewshed <- viewshed(dem, loc = c(500000, 5500000), observer = 1.7,
                     target = 0)
plot(viewshed, main = "Viewshed (1 = visible)")
```

### Recipe 12: Raster Mosaicking and Merging

```r
library(terra)

# List tiles
tiles <- list.files("dem_tiles/", pattern = "\\.tif$", full.names = TRUE)
rasters <- lapply(tiles, rast)

# Merge (simple)
merged <- do.call(merge, rasters)
writeRaster(merged, "dem_merged.tif", overwrite = TRUE)

# Mosaic with custom function (e.g., mean of overlapping areas)
mosaic_result <- do.call(mosaic, c(rasters, list(fun = "mean")))

# Virtual Raster (no data copying -- just metadata)
vrt(tiles, "dem_virtual.vrt")
r_vrt <- rast("dem_virtual.vrt")
```

### Recipe 13: Web Scraping Spatial Data from APIs

```r
library(httr2)
library(sf)
library(jsonlite)

# ArcGIS REST API query
url <- "https://services.arcgis.com/.../FeatureServer/0/query"
resp <- request(url) |>
  req_url_query(
    where = "POP > 50000",
    outFields = "*",
    f = "geojson",
    resultRecordCount = 5000
  ) |>
  req_perform()

features <- resp_body_json(resp) |> st_read(quiet = TRUE)

# WFS query
wfs_url <- "https://example.com/wfs"
wfs_data <- st_read(paste0(
  wfs_url,
  "?service=WFS&version=2.0.0&request=GetFeature",
  "&typeName=namespace:layer_name",
  "&outputFormat=application/json",
  "&bbox=10,51,11,52"
))

# STAC API (SpatioTemporal Asset Catalog)
library(rstac)
stac_obj <- stac("https://planetarycomputer.microsoft.com/api/stac/v1")
items <- stac_obj |>
  stac_search(
    collections = "sentinel-2-l2a",
    bbox = c(10, 51, 11, 52),
    datetime = "2024-06-01/2024-06-30",
    limit = 10
  ) |>
  post_request()

# Download assets
items |> assets_download(asset_names = c("B04", "B08"), output_dir = "stac_data/")
```

### Recipe 14: Voronoi / Thiessen Polygons

```r
library(sf)

pts <- st_read("weather_stations.gpkg")
boundary <- st_read("study_area.gpkg") |> st_union()

# Create Voronoi polygons
voronoi <- st_voronoi(st_union(pts), envelope = st_geometry(boundary))
voronoi <- st_collection_extract(voronoi, "POLYGON")
voronoi <- st_intersection(st_sf(geometry = voronoi), boundary)

# Assign station attributes
voronoi <- st_join(voronoi, pts)
plot(voronoi["temperature"], main = "Thiessen Polygons")
```

### Recipe 15: Batch Process Multiple Shapefiles

```r
library(sf)
library(purrr)

# List all shapefiles in a directory
shp_files <- list.files("input_shapefiles/", pattern = "\\.shp$",
                        full.names = TRUE)

# Read, process, and write each
walk(shp_files, function(f) {
  data <- st_read(f, quiet = TRUE) |>
    st_transform(4326) |>
    st_make_valid()
  outfile <- file.path("output/",
                       gsub("\\.shp$", ".gpkg", basename(f)))
  st_write(data, outfile, quiet = TRUE)
})

# Or combine all into one file
all_data <- map(shp_files, ~st_read(.x, quiet = TRUE) |>
                  st_transform(4326)) |>
  bind_rows()
st_write(all_data, "combined.gpkg")
```

### Recipe 16: Raster to/from Polygon Conversion

```r
library(terra)
library(sf)

# Raster to polygons
r <- rast("landcover.tif")
polys <- as.polygons(r, dissolve = TRUE)
polys_sf <- st_as_sf(polys)

# Raster to points
pts <- as.points(r)
pts_sf <- st_as_sf(pts)

# Polygons to raster (rasterize)
zones <- vect("admin_boundaries.gpkg")
r_template <- rast(ext(zones), res = 100, crs = crs(zones))
rasterized <- rasterize(zones, r_template, field = "pop_density")

# Lines to raster
roads <- vect("roads.gpkg")
road_raster <- rasterize(roads, r_template, field = 1)
```

### Recipe 17: Hexagonal Binning

```r
library(sf)
library(dplyr)

pts <- st_read("events.gpkg") |> st_transform(32633)
boundary <- st_read("boundary.gpkg") |> st_transform(32633)

# Create hexagonal grid
hex_grid <- st_make_grid(boundary, cellsize = 1000,
                         square = FALSE) |>  # square=FALSE -> hexagons
  st_sf() |>
  mutate(hex_id = row_number())

# Clip to boundary
hex_grid <- hex_grid[st_intersects(hex_grid, boundary, sparse = FALSE)[, 1], ]

# Count points per hexagon
hex_counts <- hex_grid |>
  st_join(pts) |>
  group_by(hex_id) |>
  summarize(count = n(), .groups = "drop")

# Plot
library(ggplot2)
ggplot(hex_counts) +
  geom_sf(aes(fill = count), color = "white", linewidth = 0.1) +
  scale_fill_viridis_c(option = "inferno") +
  theme_void()
```

---

## Python to R Comparison

For GIS professionals who work in both languages, this section provides side-by-side code comparisons. See also the [Python Stack](python-stack.md) and [Python Libraries](../tools/python-libraries.md) pages.

### Reading Vector Data

```r
# R (sf)
library(sf)
gdf <- st_read("data.gpkg", layer = "parcels")
```

```python
# Python (geopandas)
import geopandas as gpd
gdf = gpd.read_file("data.gpkg", layer="parcels")
```

### Reading Raster Data

```r
# R (terra)
library(terra)
r <- rast("dem.tif")
```

```python
# Python (rasterio / rioxarray)
import rioxarray
r = rioxarray.open_rasterio("dem.tif")
```

### Spatial Join

```r
# R
joined <- st_join(pts, polys, join = st_within)
```

```python
# Python
joined = gpd.sjoin(pts, polys, predicate="within")
```

### Buffer

```r
# R
buffered <- st_buffer(pts, dist = 1000)
```

```python
# Python
buffered = pts.buffer(1000)
```

### Zonal Statistics

```r
# R (terra)
vals <- extract(r, zones, fun = mean, na.rm = TRUE)

# R (exactextractr -- more precise)
library(exactextractr)
vals <- exact_extract(r, zones_sf, fun = "mean")
```

```python
# Python (rasterstats)
from rasterstats import zonal_stats
stats = zonal_stats(zones, "dem.tif", stats=["mean", "std"])
```

### Choropleth Map

```r
# R (ggplot2)
ggplot(nc) +
  geom_sf(aes(fill = BIR74)) +
  scale_fill_viridis_c() +
  theme_void()
```

```python
# Python (matplotlib)
nc.plot(column="BIR74", cmap="viridis", legend=True)
```

### Interactive Map

```r
# R (mapview)
mapview(nc, zcol = "BIR74")
```

```python
# Python (folium)
nc.explore(column="BIR74", cmap="viridis")
```

### Kriging

```r
# R (gstat)
library(gstat)
v <- variogram(zinc ~ 1, data = pts)
v_fit <- fit.variogram(v, vgm("Sph"))
ok <- krige(zinc ~ 1, pts, grid, v_fit)
```

```python
# Python (pykrige)
from pykrige.ok import OrdinaryKriging
ok = OrdinaryKriging(x, y, z, variogram_model="spherical")
z_pred, ss_pred = ok.execute("grid", gridx, gridy)
```

### When to Use R vs Python

| Task | Recommended | Why |
|---|---|---|
| Spatial statistics (Moran's I, GWR, SAR) | **R** | Deeper ecosystem (spdep, spatialreg, GWmodel) |
| Point pattern analysis | **R** | `spatstat` is unmatched |
| Geostatistics (kriging) | **R** | `gstat` is the gold standard |
| Publication-quality maps | **R** | `tmap`, `ggplot2`, `mapsf` |
| Deep learning on imagery | **Python** | PyTorch/TensorFlow ecosystem |
| Web application backends | **Python** | Django/FastAPI + GDAL |
| Cloud-native pipelines | **Python** | Better cloud SDK support |
| Interactive notebooks | **Both** | Jupyter (Python), Quarto (R) |
| ETL / data engineering | **Python** | pandas, dask, spark |
| Exploratory data analysis | **R** | tidyverse + sf integration |

### Calling Between Languages

```r
# Use Python from R via reticulate
library(reticulate)
use_condaenv("geo")

gpd <- import("geopandas")
py_gdf <- gpd$read_file("data.gpkg")

# Convert between R sf and Python GeoDataFrame
r_sf <- py_to_r(py_gdf)        # approximate -- may need manual conversion
# Better: save to file and read in the other language
```

```python
# Use R from Python via rpy2
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

sf = importr("sf")
nc = sf.st_read("nc.shp")
```

---

## Further Reading

### Books and Online Resources

- **Geocomputation with R** (Lovelace, Nowosad, Muenchow) -- [geocompr.robinlovelace.net](https://geocompr.robinlovelace.net/) -- The definitive open textbook
- **Spatial Data Science with R and terra** -- [rspatial.org](https://rspatial.org/) -- By the terra author
- **Spatial Data Science** (Pebesma, Bivand) -- [r-spatial.org/book](https://r-spatial.org/book/) -- By the sf/stars/gstat authors
- **R-spatial blog** -- [r-spatial.org](https://r-spatial.org/) -- News and tutorials
- **r-spatial GitHub organization** -- [github.com/r-spatial](https://github.com/r-spatial/)

### Related Pages in awesome-giser

- [Python Stack](python-stack.md) -- Python geospatial ecosystem
- [Spatial Statistics](spatial-statistics.md) -- Statistical methods deep dive
- [ML for GIS](ml-gis.md) -- Machine learning approaches
- [Python Libraries](../tools/python-libraries.md) -- Python library reference
- [Remote Sensing Tools](../tools/remote-sensing.md) -- RS software and platforms
- [Cloud Platforms](../tools/cloud-platforms.md) -- GEE, Planetary Computer, etc.
- [3D Visualization](../visualization/3d-visualization.md) -- 3D rendering techniques
- [Thematic Maps](../visualization/thematic-maps.md) -- Cartographic design
- [Desktop GIS](../tools/desktop-gis.md) -- QGIS, ArcGIS Pro
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS, SpatiaLite, DuckDB

---

[Back to Data Analysis](README.md) | [Back to Main README](../README.md)
