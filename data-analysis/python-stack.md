# Python Geospatial Stack

> A curated guide to the Python libraries every GIS professional should know, organized by purpose: core I/O, visualization, analysis, and cloud-scale processing.

> **Quick Picks**
> - **Start here:** GeoPandas + Shapely + Rasterio (core I/O)
> - **Best viz in notebooks:** Lonboard (fast, GPU-accelerated) or Folium (classic interactive)
> - **Local analytics without PostGIS:** DuckDB Spatial + GeoParquet
> - **Cloud-native raster:** xarray + rioxarray + stackstac
> - **Trajectory data:** MovingPandas

---

## Table of Contents

- [Core Libraries](#core-libraries)
- [Visualization](#visualization)
- [Analysis](#analysis)
- [Cloud & Big Data](#cloud--big-data)
- [Environment Setup](#environment-setup)
- [Common Recipes](#common-recipes)

---

## Core Libraries

The foundation of any Python geospatial workflow. These libraries handle reading, writing, manipulating, and transforming spatial data.

| Library | Purpose | Key Features | Install |
|---|---|---|---|
| [GeoPandas](https://geopandas.org/) | Vector data analysis | DataFrame API for spatial data, spatial joins, overlay, dissolve | `conda install geopandas` |
| [Shapely](https://shapely.readthedocs.io/) | Geometry operations | Buffer, intersection, union, convex hull, STRtree spatial index | `pip install shapely` |
| [Fiona](https://fiona.readthedocs.io/) | Vector file I/O | Read/write Shapefile, GeoJSON, GPKG, GML; OGR wrapper | `conda install fiona` |
| [Rasterio](https://rasterio.readthedocs.io/) | Raster file I/O | Read/write GeoTIFF, windowed reading, masking, reprojection | `conda install rasterio` |
| [PyProj](https://pyproj4.github.io/pyproj/) | CRS transformations | PROJ bindings, CRS objects, coordinate transformations | `pip install pyproj` |
| [GDAL/OGR (osgeo)](https://gdal.org/api/python.html) | Low-level spatial I/O | Full GDAL/OGR API, format support, performance | `conda install gdal` |
| [GeoAlchemy2](https://geoalchemy-2.readthedocs.io/) | PostGIS + SQLAlchemy | Spatial queries via ORM, WKT/WKB, spatial indexes | `pip install geoalchemy2` |
| [GeoArrow](https://geoarrow.org/) + [GeoParquet](https://geoparquet.org/) | Columnar spatial data | Zero-copy exchange, fast I/O, interop with Arrow ecosystem | `pip install geoarrow-pyarrow` |

### Version Compatibility (Python 3.10 / 3.11 / 3.12)

All major libraries (GeoPandas 1.0+, Shapely 2.0+, Rasterio 1.3+, Fiona 1.9+, PyProj 3.6+) support Python 3.10 through 3.12. Shapely 2.0 dropped the `speedups` module and made GEOS the default backend. GeoPandas 1.0 requires Shapely 2.0+. Use `conda-forge` channel for best compatibility.

---

## Visualization

Libraries for creating static maps, interactive web maps, and 3D visualizations.

| Library | Purpose | Key Features | Install |
|---|---|---|---|
| [Matplotlib](https://matplotlib.org/) + GeoPandas | Static maps | Publication-quality, full control, subplot support | `pip install matplotlib` |
| [Folium](https://python-visualization.github.io/folium/) | Interactive web maps | Leaflet.js wrapper, choropleth, markers, heatmaps | `pip install folium` |
| [Plotly](https://plotly.com/python/) | Interactive charts + maps | Mapbox integration, scatter_mapbox, choropleth_mapbox | `pip install plotly` |
| [Pydeck](https://deckgl.readthedocs.io/) | 3D web maps | deck.gl bindings, large datasets, GPU rendering | `pip install pydeck` |
| [Contextily](https://contextily.readthedocs.io/) | Basemap tiles | Add basemaps to matplotlib/GeoPandas plots | `pip install contextily` |
| [Datashader](https://datashader.org/) | Big data rasterization | Render millions of points, dynamic re-aggregation | `conda install datashader` |
| [Lonboard](https://developmentseed.org/lonboard/) | Fast web maps in notebooks | GPU-accelerated, millions of features in Jupyter | `pip install lonboard` |

### Choosing a Visualization Library

- **Quick exploration in Jupyter:** Lonboard (fastest for large datasets) or Folium (simpler API)
- **Publication-quality static maps:** Matplotlib + GeoPandas + Contextily
- **Interactive dashboards:** Plotly or Pydeck
- **Millions of points:** Datashader (server-side) or Lonboard (client-side WebGL)

---

## Analysis

Libraries for spatial statistics, geostatistics, and spatial modeling.

| Library | Purpose | Key Features | Install |
|---|---|---|---|
| [PySAL](https://pysal.org/) | Spatial analysis suite | Weights, autocorrelation, clustering, regression, inequality | `conda install pysal` |
| [Scikit-learn](https://scikit-learn.org/) | Machine learning | Classification, regression, clustering (combine with spatial features) | `pip install scikit-learn` |
| [SciPy (spatial)](https://docs.scipy.org/doc/scipy/reference/spatial.html) | Spatial algorithms | KDTree, Voronoi, Delaunay, convex hull, distance matrices | `pip install scipy` |
| [MovingPandas](https://movingpandas.github.io/movingpandas/) | Trajectory analysis | Movement data, speed, direction, splitting, aggregation | `conda install movingpandas` |
| [xarray-spatial](https://xarray-spatial.readthedocs.io/) | Raster analysis | Focal, zonal, surface, proximity, pathfinding on xarray | `pip install xarray-spatial` |
| [NetworkX](https://networkx.org/) + [OSMnx](https://osmnx.readthedocs.io/) | Network analysis | Graph-based routing, centrality, isochrones, OSM download | `pip install osmnx` |
| [Tobler](https://pysal.org/tobler/) | Areal interpolation | Dasymetric mapping, pycnophylactic, model-based | `pip install tobler` |
| [pointpats](https://pysal.org/pointpats/) | Point pattern analysis | Quadrat, nearest neighbor, K-function, G-function | `pip install pointpats` |

### Which Analysis Library Should I Use?

- **Spatial joins, overlay, dissolve:** GeoPandas
- **Autocorrelation, clustering, regression:** PySAL
- **Point pattern analysis:** pointpats (Python) or spatstat (R)
- **Network routing and accessibility:** OSMnx + NetworkX
- **Trajectory / movement data:** MovingPandas
- **Raster focal, zonal, surface analysis:** xarray-spatial
- **Areal interpolation (census tracts, zones):** Tobler
- **General ML on spatial features:** scikit-learn + XGBoost

---

## Cloud & Big Data

Libraries for handling large-scale geospatial data, cloud-optimized formats, and distributed computing.

| Library | Purpose | Key Features | Install |
|---|---|---|---|
| [xarray](https://docs.xarray.dev/) | N-dimensional arrays | NetCDF, Zarr, labeled dimensions, lazy loading | `conda install xarray` |
| [rioxarray](https://corteva.github.io/rioxarray/) | xarray + Rasterio | CRS-aware xarray, clip, reproject, to_raster | `pip install rioxarray` |
| [Dask-GeoPandas](https://dask-geopandas.readthedocs.io/) | Parallel GeoPandas | Distributed spatial operations, out-of-core processing | `pip install dask-geopandas` |
| [stackstac](https://stackstac.readthedocs.io/) | STAC + xarray | Load STAC items as xarray DataArrays, lazy cloud access | `pip install stackstac` |
| [Planetary Computer SDK](https://planetarycomputer.microsoft.com/) | Microsoft data catalog | Access Sentinel, Landsat, NAIP via STAC + Dask | `pip install planetary-computer` |
| [Google Earth Engine (ee)](https://earthengine.google.com/) | Cloud geoprocessing | Server-side computation, petabytes of imagery | `pip install earthengine-api` |
| [DuckDB (spatial)](https://duckdb.org/docs/extensions/spatial.html) | Analytical SQL | Fast local analytics, spatial extension, Parquet/GeoParquet | `pip install duckdb` |

### DuckDB Spatial: Local Analytics Without PostGIS

DuckDB Spatial is a game-changer for local geospatial analysis. It provides PostGIS-like SQL spatial functions on local files -- no database server needed. Read GeoParquet, Shapefile, GeoJSON, and FlatGeobuf directly.

```python
import duckdb

con = duckdb.connect()
con.install_extension("spatial")
con.load_extension("spatial")

# Read GeoParquet and run spatial SQL
con.sql("""
    SELECT name, ST_Area(geom) as area
    FROM 'buildings.parquet'
    WHERE ST_Within(geom, ST_GeomFromText('POLYGON((...))'))
""")
```

### GeoArrow / GeoParquet Ecosystem

GeoParquet is the emerging standard for columnar geospatial data. It offers 10-100x faster I/O than Shapefile and integrates with the Arrow ecosystem for zero-copy data exchange between Python, R, DuckDB, and WASM.

```python
import geopandas as gpd

# Read/write GeoParquet (built into GeoPandas 1.0+)
gdf = gpd.read_parquet("data.parquet")
gdf.to_parquet("output.parquet")
```

### Cloud Cost Comparison

| Platform | Free Tier | Pay-As-You-Go | Best For |
|---|---|---|---|
| Google Earth Engine | Free for research | Enterprise pricing | Petabyte-scale RS analysis |
| Microsoft Planetary Computer | Free (hub access) | Azure compute costs | STAC-based data access |
| AWS (S3 + Lambda) | 5 GB S3, 1M Lambda | ~$0.023/GB storage | Custom pipelines |
| Local (DuckDB + GeoParquet) | Free | Hardware cost only | Privacy-sensitive, offline |

---

## Environment Setup

### Recommended: Conda (Mamba)

Conda handles the C/C++ dependencies (GDAL, GEOS, PROJ) that pip often struggles with.

```bash
# Install mamba (faster conda)
conda install -n base -c conda-forge mamba

# Create a geospatial environment
mamba create -n geo python=3.11 \
  geopandas rasterio fiona shapely pyproj \
  matplotlib folium plotly pydeck contextily \
  pysal scikit-learn scipy xarray rioxarray \
  jupyterlab ipywidgets

# Activate
conda activate geo
```

### Alternative: Pip with System GDAL

If you prefer pip, install system GDAL first:

```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin libgdal-dev
export GDAL_VERSION=$(gdal-config --version)

# Then pip install
pip install GDAL==$GDAL_VERSION
pip install geopandas rasterio fiona shapely pyproj
```

### Docker

For reproducible environments, use a pre-built geospatial Docker image:

```dockerfile
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.0
RUN pip install geopandas rasterio pysal scikit-learn folium
```

### Docker Compose with JupyterLab + PostGIS

```yaml
version: '3.8'
services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: gisdb
      POSTGRES_USER: gis
      POSTGRES_PASSWORD: gispass
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

  jupyter:
    image: quay.io/jupyter/scipy-notebook:latest
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
    environment:
      JUPYTER_ENABLE_LAB: "yes"
    command: >
      bash -c "pip install geopandas rasterio fiona shapely pyproj
      folium contextily pysal scikit-learn duckdb lonboard &&
      start-notebook.sh --NotebookApp.token=''"
    depends_on:
      - postgis

volumes:
  pgdata:
```

---

## Common Recipes

Quick reference for frequently needed operations. Each recipe is a self-contained code snippet.

### Read a Shapefile and Reproject

```python
import geopandas as gpd

gdf = gpd.read_file("input.shp")
gdf_utm = gdf.to_crs(epsg=32633)  # Reproject to UTM Zone 33N
gdf_utm.to_file("output.gpkg", driver="GPKG")
```

### Spatial Join (Points in Polygons)

```python
points = gpd.read_file("points.geojson")
polygons = gpd.read_file("zones.gpkg")
joined = gpd.sjoin(points, polygons, how="inner", predicate="within")
```

### Buffer and Dissolve

```python
gdf["geometry"] = gdf.geometry.buffer(500)  # 500m buffer (projected CRS)
dissolved = gdf.dissolve(by="category")
```

### Zonal Statistics (Raster + Vector)

```python
from rasterstats import zonal_stats
stats = zonal_stats("zones.geojson", "elevation.tif",
                    stats=["mean", "min", "max", "std"])
```

### Read a Cloud-Optimized GeoTIFF (COG) from S3

```python
import rioxarray
ds = rioxarray.open_rasterio(
    "https://example.com/data/cog.tif",
    overview_level=2  # Read at reduced resolution
)
```

### Create a Choropleth with Folium

```python
import folium
m = folium.Map(location=[40, -100], zoom_start=4)
folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    data=gdf, columns=["id", "population"],
    key_on="feature.properties.id",
    fill_color="YlOrRd"
).add_to(m)
```

### Parallel Processing with Dask-GeoPandas

```python
import dask_geopandas
ddf = dask_geopandas.from_geopandas(gdf, npartitions=8)
result = ddf.geometry.buffer(100).compute()
```

---

[Back to Data Analysis](README.md) · [Back to Main README](../README.md)
