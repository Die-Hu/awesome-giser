# R Geospatial Stack

> A curated guide to the modern R spatial ecosystem, organized by purpose: core I/O, visualization, and analysis. Where applicable, legacy packages and their modern successors are noted.

> **Quick Picks**
> - **Start here:** sf (vectors) + terra (rasters) -- the modern core
> - **Best interactive map:** tmap (mode = "view") or mapview (one-liner)
> - **Publication maps:** ggplot2 + geom_sf() + tidyterra
> - **3D visualization:** rayshader
> - **Point patterns:** spatstat
> - **Spatial regression:** spatialreg + GWmodel

---

## Table of Contents

- [Core Libraries](#core-libraries)
- [Visualization](#visualization)
- [Analysis](#analysis)
- [Environment Setup](#environment-setup)

---

## Core Libraries

The modern R spatial stack has converged around `sf` for vectors and `terra` for rasters, replacing the older `sp`, `rgdal`, and `raster` packages.

| Package | Purpose | Successor To | CRAN |
|---|---|---|---|
| [sf](https://r-spatial.github.io/sf/) | Vector data (Simple Features) | `sp`, `rgdal`, `rgeos` | [CRAN](https://cran.r-project.org/package=sf) |
| [terra](https://rspatial.github.io/terra/) | Raster and vector data | `raster`, `rgdal` | [CRAN](https://cran.r-project.org/package=terra) |
| [stars](https://r-spatial.github.io/stars/) | Spatiotemporal arrays | `raster` (for datacubes) | [CRAN](https://cran.r-project.org/package=stars) |
| [sp](https://edzer.github.io/sp/) | Spatial classes (legacy) | — (being retired) | [CRAN](https://cran.r-project.org/package=sp) |
| [s2](https://r-spatial.github.io/s2/) | Spherical geometry | Planar geometry in `sf` | [CRAN](https://cran.r-project.org/package=s2) |
| [gdalcubes](https://gdalcubes.github.io/) | Earth observation datacubes | — | [CRAN](https://cran.r-project.org/package=gdalcubes) |
| [arrow + geoarrow](https://arrow.apache.org/docs/r/) | GeoParquet / large data | — | [CRAN](https://cran.r-project.org/package=arrow) |
| [tidyterra](https://dieghernan.github.io/tidyterra/) | ggplot2 for terra objects | — | [CRAN](https://cran.r-project.org/package=tidyterra) |

### Migration Guide: sp/raster to sf/terra

| Old Package | New Package | Key Differences |
|---|---|---|
| `sp::SpatialPointsDataFrame` | `sf::st_as_sf()` | sf uses tibble-like data frames with geometry column |
| `rgdal::readOGR()` | `sf::st_read()` | sf auto-detects format, returns sf data frame |
| `rgdal::writeOGR()` | `sf::st_write()` | sf supports GeoPackage, GeoJSON, Parquet, etc. |
| `raster::raster()` | `terra::rast()` | terra is faster, uses C++ backend, lower memory |
| `raster::stack()` | `terra::rast(c(...))` | terra handles multi-band natively |
| `rgeos::gBuffer()` | `sf::st_buffer()` | sf uses GEOS via s2 for spherical geometry |
| `sp::over()` | `sf::st_join()` | sf uses tidyverse-style joins |

### Migration Note

As of 2024, `rgdal`, `rgeos`, and `maptools` have been retired from CRAN. If you have code using these packages, migrate to `sf` (for vectors) and `terra` (for rasters). The `sf` package provides drop-in replacements for most operations.

---

## Visualization

R has excellent cartographic libraries, from grammar-of-graphics maps to interactive web maps.

| Package | Purpose | Successor To | CRAN |
|---|---|---|---|
| [tmap](https://r-tmap.github.io/tmap/) | Thematic maps (static + interactive) | — | [CRAN](https://cran.r-project.org/package=tmap) |
| [ggplot2](https://ggplot2.tidyverse.org/) + `geom_sf()` | Grammar-of-graphics maps | `ggmap`, `spplot` | [CRAN](https://cran.r-project.org/package=ggplot2) |
| [leaflet](https://rstudio.github.io/leaflet/) | Interactive web maps | — | [CRAN](https://cran.r-project.org/package=leaflet) |
| [mapview](https://r-spatial.github.io/mapview/) | Quick interactive viewing | — | [CRAN](https://cran.r-project.org/package=mapview) |
| [mapsf](https://riatelab.github.io/mapsf/) | Publication-quality thematic maps | `cartography` | [CRAN](https://cran.r-project.org/package=mapsf) |
| [rayshader](https://www.rayshader.com/) | 3D maps and elevation | — | [CRAN](https://cran.r-project.org/package=rayshader) |
| [plotly](https://plotly-r.com/) | Interactive charts + maps | — | [CRAN](https://cran.r-project.org/package=plotly) |
| [tidyterra](https://dieghernan.github.io/tidyterra/) | ggplot2 + terra integration | — | [CRAN](https://cran.r-project.org/package=tidyterra) |

### tidyterra: ggplot2 for Rasters

tidyterra bridges terra SpatRaster objects and ggplot2, enabling grammar-of-graphics for raster data:

```r
library(tidyterra)
library(terra)
library(ggplot2)

dem <- rast("elevation.tif")
ggplot() +
  geom_spatraster(data = dem) +
  scale_fill_hypso_c() +
  theme_minimal()
```

### rayshader: 3D Terrain Visualization

rayshader converts elevation data into 3D rendered maps with hillshading, water detection, and camera control:

```r
library(rayshader)
library(terra)

dem <- rast("elevation.tif") |> as.matrix()
dem |>
  sphere_shade(texture = "desert") |>
  add_shadow(ray_shade(dem)) |>
  plot_3d(dem, zscale = 10, theta = -45, phi = 30)
render_snapshot("3d_map.png")
```

### Choosing a Visualization Package

- **Quick exploration:** `mapview` (one-liner interactive map)
- **Publication-quality static maps:** `tmap` (mode = "plot") or `mapsf`
- **Grammar-of-graphics integration:** `ggplot2 + geom_sf()`
- **Interactive dashboards:** `leaflet` or `tmap` (mode = "view")
- **3D terrain visualization:** `rayshader`

---

## Analysis

R has a deep ecosystem for spatial statistics, point pattern analysis, and geostatistics.

| Package | Purpose | Successor To | CRAN |
|---|---|---|---|
| [spdep](https://r-spatial.github.io/spdep/) | Spatial dependence & autocorrelation | — | [CRAN](https://cran.r-project.org/package=spdep) |
| [spatstat](https://spatstat.org/) | Point pattern analysis | — | [CRAN](https://cran.r-project.org/package=spatstat) |
| [gstat](https://r-spatial.github.io/gstat/) | Geostatistics (kriging, variograms) | — | [CRAN](https://cran.r-project.org/package=gstat) |
| [spatialreg](https://r-spatial.github.io/spatialreg/) | Spatial regression models | Part of old `spdep` | [CRAN](https://cran.r-project.org/package=spatialreg) |
| [GWmodel](https://cran.r-project.org/package=GWmodel) | Geographically Weighted models | — | [CRAN](https://cran.r-project.org/package=GWmodel) |
| [rgeoda](https://geodacenter.github.io/rgeoda/) | GeoDa spatial analysis in R | — | [CRAN](https://cran.r-project.org/package=rgeoda) |
| [osmdata](https://docs.ropensci.org/osmdata/) | Download OpenStreetMap data | — | [CRAN](https://cran.r-project.org/package=osmdata) |
| [tidygeocoder](https://jessecambon.github.io/tidygeocoder/) | Geocoding | `ggmap::geocode` | [CRAN](https://cran.r-project.org/package=tidygeocoder) |
| [accessibility](https://ipeagit.github.io/accessibility/) | Transport accessibility metrics | — | [CRAN](https://cran.r-project.org/package=accessibility) |
| [stplanr](https://docs.ropensci.org/stplanr/) | Transport planning | — | [CRAN](https://cran.r-project.org/package=stplanr) |
| [sfnetworks](https://luukvdmeer.github.io/sfnetworks/) | Tidy spatial networks | — | [CRAN](https://cran.r-project.org/package=sfnetworks) |

### Which R Spatial Analysis Package Should I Use?

- **Spatial autocorrelation (Moran's I, LISA):** spdep
- **Spatial regression (lag, error, Durbin):** spatialreg
- **Geographically weighted regression:** GWmodel
- **Point pattern analysis (K-function, density):** spatstat
- **Kriging and variograms:** gstat
- **GeoDa-style analysis in R:** rgeoda
- **Network analysis (routing, centrality):** sfnetworks, dodgr
- **Transport accessibility (isochrones, 2SFCA):** r5r, accessibility
- **Download OSM data:** osmdata
- **Geocoding:** tidygeocoder

---

## Environment Setup

### Installing Core Packages

```r
# Install core spatial packages
install.packages(c("sf", "terra", "stars", "tmap", "ggplot2", "leaflet", "mapview"))

# Install analysis packages
install.packages(c("spdep", "spatstat", "gstat", "spatialreg", "GWmodel"))

# Install data access packages
install.packages(c("osmdata", "tidygeocoder", "gdalcubes"))
```

### System Dependencies

On Linux, you need system libraries before installing R spatial packages:

```bash
# Ubuntu/Debian
sudo apt-get install \
  libgdal-dev libgeos-dev libproj-dev \
  libudunits2-dev libsqlite3-dev
```

On macOS with Homebrew:

```bash
brew install gdal geos proj
```

### Docker

```dockerfile
FROM rocker/geospatial:4.3
RUN install2.r --error \
  tmap mapview spdep spatstat gstat GWmodel rgeoda
```

The `rocker/geospatial` image comes with `sf`, `terra`, `stars`, and system dependencies pre-installed.

### Reproducible R Environments with renv

```r
# Initialize renv in your project
renv::init()

# Install packages as normal
install.packages(c("sf", "terra", "tmap", "spdep"))

# Take a snapshot of current packages
renv::snapshot()

# Restore on another machine
renv::restore()
```

This creates a `renv.lock` file that pins exact package versions for reproducibility.

---

[Back to Data Analysis](README.md) · [Back to Main README](../README.md)
