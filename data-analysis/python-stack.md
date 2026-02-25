# Python Geospatial Stack

> The definitive guide to the Python geospatial ecosystem for GIS professionals. Covers every layer from low-level I/O to cloud-scale analytics, with production-tested code, benchmarks, and architectural guidance.

> **Quick Picks**
> - **Start here:** GeoPandas 1.0 + Shapely 2.0 + Rasterio (core I/O)
> - **Fastest notebook viz:** Lonboard (GPU-accelerated) or Folium (classic interactive)
> - **Local analytics without PostGIS:** DuckDB Spatial + GeoParquet
> - **Cloud-native raster:** xarray + rioxarray + stackstac
> - **Trajectory data:** MovingPandas
> - **Columnar exchange format:** GeoArrow + GeoParquet (10-100x faster than Shapefile)
> - **Spatial statistics:** PySAL (esda + spreg + mgwr)

> **Cross-references:** [Python Libraries Catalog](../tools/python-libraries.md) | [Visualization Guide](../visualization/README.md) | [Data Sources](../data-sources/README.md) | [JS Bindbox](../js-bindbox/)

---

## Table of Contents

- [1. Core Libraries Deep Dive](#1-core-libraries-deep-dive)
  - [GeoPandas 1.0+](#geopandas-10)
  - [Shapely 2.0](#shapely-20)
  - [Fiona vs pyogrio](#fiona-vs-pyogrio)
  - [Rasterio](#rasterio)
  - [PyProj 3.x](#pyproj-3x)
  - [GeoAlchemy2](#geoalchemy2)
- [2. GeoArrow & GeoParquet Ecosystem](#2-geoarrow--geoparquet-ecosystem)
  - [Format Comparison Benchmarks](#format-comparison-benchmarks)
  - [GeoArrow Python APIs](#geoarrow-python-apis)
  - [Lonboard: GPU Visualization via GeoArrow](#lonboard-gpu-visualization-via-geoarrow)
- [3. Visualization Deep Dive](#3-visualization-deep-dive)
  - [Matplotlib + Cartopy](#matplotlib--cartopy)
  - [Folium Advanced](#folium-advanced)
  - [Plotly Mapbox](#plotly-mapbox)
  - [Pydeck Layers](#pydeck-layers)
  - [Lonboard GPU Maps](#lonboard-gpu-maps)
  - [Datashader Big Data](#datashader-big-data)
  - [hvPlot + GeoViews](#hvplot--geoviews)
- [4. Analysis Libraries](#4-analysis-libraries)
  - [PySAL Complete Ecosystem](#pysal-complete-ecosystem)
  - [scikit-learn Spatial Patterns](#scikit-learn-spatial-patterns)
  - [SciPy Spatial](#scipy-spatial)
  - [NetworkX + OSMnx](#networkx--osmnx)
  - [MovingPandas](#movingpandas)
- [5. Cloud & Big Data](#5-cloud--big-data)
  - [STAC Workflow: xarray + rioxarray + stackstac](#stac-workflow)
  - [Dask-GeoPandas](#dask-geopandas)
  - [DuckDB Spatial](#duckdb-spatial)
  - [Google Earth Engine](#google-earth-engine)
  - [Planetary Computer SDK](#planetary-computer-sdk)
  - [Apache Sedona](#apache-sedona)
- [6. Raster Analysis Stack](#6-raster-analysis-stack)
  - [Rasterio Recipes](#rasterio-recipes)
  - [xarray-spatial](#xarray-spatial)
  - [rasterstats](#rasterstats)
  - [WhiteboxTools](#whiteboxtools)
  - [richdem](#richdem)
  - [GDAL Python Bindings](#gdal-python-bindings)
- [7. Vector Processing Recipes](#7-vector-processing-recipes)
- [8. Environment Setup](#8-environment-setup)
- [9. Performance Optimization](#9-performance-optimization)
- [10. Testing & CI](#10-testing--ci)
- [11. Integration Patterns](#11-integration-patterns)
- [12. Cookbook](#12-cookbook)

---

## 1. Core Libraries Deep Dive

The foundation of every Python geospatial workflow. These libraries handle reading, writing, manipulating, and transforming spatial data. Understanding their architecture, version-specific features, and performance characteristics is essential.

### GeoPandas 1.0+

GeoPandas extends pandas with spatial data types and operations. Version 1.0 (released 2024) brought significant changes.

**What changed in GeoPandas 1.0:**

| Feature | Pre-1.0 | 1.0+ |
|---|---|---|
| Default I/O engine | Fiona | pyogrio (2-5x faster) |
| Required Shapely | 1.x or 2.x | 2.0+ only |
| `geometry` dtype | Custom | Arrow-backed (optional) |
| GeoParquet support | Experimental | First-class |
| `.explore()` | Folium | Folium (improved) |
| Deprecations removed | Many legacy paths | Clean API |

**Core API patterns (GeoPandas 1.0):**

```python
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

# --- Reading data ---
# GeoParquet (fastest for large datasets)
gdf = gpd.read_parquet("buildings.parquet")

# Shapefile with bbox filter (reads only intersecting features)
gdf = gpd.read_file("land_use.shp", bbox=(10.0, 45.0, 12.0, 47.0))

# GeoPackage with SQL filter (pyogrio engine, default in 1.0)
gdf = gpd.read_file(
    "census.gpkg",
    layer="tracts",
    where="population > 10000",
    columns=["geoid", "population", "geometry"],
)

# FlatGeobuf from HTTP (streaming, spatial filter on server)
gdf = gpd.read_file(
    "https://example.com/data/buildings.fgb",
    bbox=(13.3, 52.4, 13.5, 52.6),
)

# PostGIS via SQLAlchemy
from sqlalchemy import create_engine
engine = create_engine("postgresql://user:pass@localhost:5432/gisdb")
gdf = gpd.read_postgis(
    "SELECT * FROM parcels WHERE ST_Area(geom) > 1000",
    engine,
    geom_col="geom",
)

# --- Writing data ---
gdf.to_parquet("output.parquet")          # GeoParquet (recommended)
gdf.to_file("output.gpkg", driver="GPKG") # GeoPackage
gdf.to_file("output.fgb", driver="FlatGeobuf")  # FlatGeobuf
gdf.to_postgis("table_name", engine, if_exists="replace")

# --- Spatial operations ---
# Spatial join: points in polygons
points = gpd.read_file("points.geojson")
zones = gpd.read_file("zones.gpkg")
joined = gpd.sjoin(points, zones, how="inner", predicate="within")

# Nearest join (new in 0.10+, improved in 1.0)
nearest = gpd.sjoin_nearest(
    points, zones, how="left", max_distance=1000, distance_col="dist_m"
)

# Overlay operations
intersection = gpd.overlay(gdf1, gdf2, how="intersection")
union = gpd.overlay(gdf1, gdf2, how="union")
difference = gpd.overlay(gdf1, gdf2, how="difference")
symmetric_diff = gpd.overlay(gdf1, gdf2, how="symmetric_difference")

# Dissolve with aggregation
dissolved = gdf.dissolve(
    by="district",
    aggfunc={
        "population": "sum",
        "area_km2": "sum",
        "name": "first",
    },
)

# Clip to boundary
clipped = gpd.clip(gdf, boundary_polygon)

# CRS transformation
gdf_utm = gdf.to_crs(epsg=32633)       # To UTM Zone 33N
gdf_wgs84 = gdf.to_crs(epsg=4326)      # Back to WGS84
gdf_equal = gdf.to_crs("+proj=moll")    # Mollweide (equal-area)

# --- Vectorized geometry operations (Shapely 2.0 backend) ---
gdf["area_m2"] = gdf.geometry.area
gdf["centroid"] = gdf.geometry.centroid
gdf["buffered"] = gdf.geometry.buffer(100)
gdf["simplified"] = gdf.geometry.simplify(tolerance=10)
gdf["length_m"] = gdf.geometry.length
gdf["bounds"] = gdf.geometry.bounds
gdf["is_valid"] = gdf.geometry.is_valid
```

**Performance tip: pyogrio engine**

```python
# GeoPandas 1.0 uses pyogrio by default. Explicit selection:
gdf = gpd.read_file("data.gpkg", engine="pyogrio")  # Default, fastest
gdf = gpd.read_file("data.gpkg", engine="fiona")    # Legacy fallback

# pyogrio direct access (bypasses GeoDataFrame construction)
import pyogrio
# Read raw arrays -- fastest possible vector I/O
geometry, fields = pyogrio.read(
    "buildings.gpkg",
    columns=["id", "height"],
    bbox=(13.3, 52.4, 13.5, 52.6),
    max_features=100000,
)

# Read to Arrow table (zero-copy to DuckDB, Polars, etc.)
table = pyogrio.read_arrow("buildings.gpkg")
```

### Shapely 2.0

Shapely 2.0 was a ground-up rewrite. The key change: all operations are now **vectorized** -- they accept arrays of geometries and dispatch to GEOS in bulk, avoiding Python-loop overhead.

**Key Shapely 2.0 features:**

```python
import numpy as np
from shapely import (
    Point, LineString, Polygon, MultiPoint,
    box, points, linestrings, polygons,
    area, length, distance, buffer, intersection, union_all,
    contains, within, intersects, crosses,
    STRtree, get_coordinates, set_coordinates,
    make_valid, is_valid, prepare,
)

# --- Vectorized construction ---
# Create 1 million points from coordinate arrays (fast)
coords = np.random.uniform(low=[0, 0], high=[100, 100], size=(1_000_000, 2))
pts = points(coords)  # Returns ndarray of Point geometries

# Create polygons from coordinate arrays
rings = np.array([
    [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]],
    [[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]],
])
polys = polygons(rings)

# --- Vectorized operations ---
# Buffer all geometries at once (no Python loop)
buffered = buffer(pts, distance=10.0)

# Pairwise distance between two geometry arrays
dists = distance(pts[:1000], pts[1000:2000])

# Element-wise predicates
mask = contains(polys_array, pts_array)
mask = intersects(line_array, poly_array)

# Aggregate operations
merged = union_all(polys)  # Replaces cascaded_union

# --- STRtree spatial index ---
tree = STRtree(polygons_array)

# Query: which polygons intersect each point?
indices = tree.query(pts, predicate="intersects")
# Returns (input_idx, tree_idx) pairs

# Nearest neighbor query
nearest_idx = tree.nearest(pts)

# Query within distance
idx = tree.query(pts, predicate="dwithin", distance=100.0)

# Bulk nearest with distances
nearest_idx, nearest_dist = tree.nearest(pts)  # k=1 by default

# --- Coordinate manipulation ---
coords = get_coordinates(geometry_array)  # Extract all coordinates
new_geoms = set_coordinates(geometry_array, new_coords)  # Replace coordinates

# --- Validation and repair ---
valid_mask = is_valid(geometry_array)
repaired = make_valid(geometry_array)  # Fix self-intersections, etc.

# --- Performance: prepare geometries ---
# Pre-compute spatial index for repeated predicate tests
prepared = prepare(large_polygon)
mask = contains(large_polygon, pts)  # Automatically uses prepared geometry
```

**Shapely 2.0 vs 1.x performance comparison:**

| Operation | Shapely 1.x (loop) | Shapely 2.0 (vectorized) | Speedup |
|---|---|---|---|
| Buffer 100k points | 12.4 s | 0.3 s | ~40x |
| Distance 100k pairs | 8.1 s | 0.15 s | ~54x |
| Contains 100k tests | 5.2 s | 0.08 s | ~65x |
| STRtree query 100k | 3.8 s | 0.2 s | ~19x |
| union_all 10k polys | 6.5 s | 0.9 s | ~7x |

### Fiona vs pyogrio

Both libraries wrap GDAL/OGR for vector file I/O. pyogrio is the newer, faster alternative and is now the default engine in GeoPandas 1.0.

| Feature | Fiona 1.9+ | pyogrio 0.7+ |
|---|---|---|
| API style | Pythonic iterator | Bulk read/write |
| Memory model | Feature-by-feature | Columnar (Arrow) |
| Read speed (1M features) | ~18 s | ~4 s |
| Write speed (1M features) | ~22 s | ~6 s |
| Arrow support | No | Yes (zero-copy) |
| SQL `WHERE` filter | Via OGR SQL | Via OGR SQL |
| Spatial filter (bbox) | Yes | Yes |
| Schema introspection | Yes (rich) | Yes (basic) |
| Streaming / chunked | Yes (native) | Via `read_bounds` |
| Format support | All GDAL/OGR | All GDAL/OGR |

```python
# Fiona: feature-by-feature (good for streaming, memory-constrained)
import fiona

with fiona.open("buildings.gpkg") as src:
    print(src.crs, src.schema)
    for feature in src.filter(bbox=(13.3, 52.4, 13.5, 52.6)):
        props = feature["properties"]
        geom = shape(feature["geometry"])

# pyogrio: bulk read (good for analysis, fastest path to GeoDataFrame)
import pyogrio

# Read to GeoDataFrame
gdf = pyogrio.read_dataframe(
    "buildings.gpkg",
    columns=["id", "height", "type"],
    bbox=(13.3, 52.4, 13.5, 52.6),
    fid_as_index=True,
)

# Read to Arrow Table (zero-copy interop with DuckDB, Polars)
table = pyogrio.read_arrow(
    "buildings.gpkg",
    bbox=(13.3, 52.4, 13.5, 52.6),
)

# Write from GeoDataFrame
pyogrio.write_dataframe(gdf, "output.fgb", driver="FlatGeobuf")
```

### Rasterio

The Python interface to GDAL for raster I/O. Provides Pythonic access to GeoTIFF, COG, NetCDF, and 100+ other raster formats.

```python
import rasterio
from rasterio.windows import Window, from_bounds
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.enums import Resampling
import numpy as np

# --- Basic read ---
with rasterio.open("dem.tif") as src:
    print(f"CRS: {src.crs}")
    print(f"Bounds: {src.bounds}")
    print(f"Resolution: {src.res}")
    print(f"Shape: {src.height} x {src.width}, {src.count} bands")
    print(f"Data type: {src.dtypes}")
    print(f"NoData: {src.nodata}")

    data = src.read(1)                       # Read band 1
    data = src.read([1, 2, 3])               # Read RGB bands
    data = src.read(1, masked=True)          # Returns masked array

# --- Windowed reading (memory-efficient for large rasters) ---
with rasterio.open("large_raster.tif") as src:
    # Read a 512x512 pixel window
    window = Window(col_off=1000, row_off=2000, width=512, height=512)
    data = src.read(1, window=window)

    # Read window by geographic bounds
    window = from_bounds(
        left=13.3, bottom=52.4, right=13.5, top=52.6,
        transform=src.transform,
    )
    data = src.read(1, window=window)

    # Iterate over windows for processing large rasters
    for ji, window in src.block_windows(1):
        block = src.read(1, window=window)
        # process block...

# --- Cloud-Optimized GeoTIFF (COG) from HTTP/S3 ---
with rasterio.open(
    "https://example.com/cog.tif",
    overview_level=2,                         # Read at reduced resolution
) as src:
    data = src.read(1)

# Read COG from S3 with GDAL environment config
with rasterio.Env(
    AWS_NO_SIGN_REQUEST="YES",
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    VSI_CACHE=True,
    VSI_CACHE_SIZE=5000000,
):
    with rasterio.open("s3://bucket/cog.tif") as src:
        data = src.read(1, window=Window(0, 0, 256, 256))

# --- Write a GeoTIFF ---
profile = {
    "driver": "GTiff",
    "dtype": "float32",
    "width": 1000,
    "height": 1000,
    "count": 1,
    "crs": "EPSG:32633",
    "transform": transform_from_bounds(10, 45, 12, 47, 1000, 1000),
    "compress": "deflate",
    "tiled": True,
    "blockxsize": 256,
    "blockysize": 256,
}

with rasterio.open("output.tif", "w", **profile) as dst:
    dst.write(data, 1)
    dst.update_tags(AUTHOR="GISer", DATE="2025-01-15")

# --- Write a Cloud-Optimized GeoTIFF (COG) ---
from rasterio.shutil import copy as rio_copy

with rasterio.open("input.tif") as src:
    rio_copy(
        src, "output_cog.tif",
        driver="GTiff",
        tiled=True,
        compress="deflate",
        copy_src_overviews=True,
        overview_resampling="average",
    )

# --- Reproject a raster ---
with rasterio.open("input.tif") as src:
    transform, width, height = calculate_default_transform(
        src.crs, "EPSG:4326", src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update(crs="EPSG:4326", transform=transform, width=width, height=height)

    with rasterio.open("reprojected.tif", "w", **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
```

### PyProj 3.x

PROJ bindings for coordinate reference system transformations. PyProj 3.x uses the PROJ 9.x C library and supports datum transformations with high accuracy.

```python
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

# --- CRS objects ---
crs_wgs84 = CRS.from_epsg(4326)
crs_utm33 = CRS.from_epsg(32633)
crs_custom = CRS.from_proj4("+proj=laea +lat_0=52 +lon_0=10 +datum=WGS84")
crs_wkt = CRS.from_wkt(wkt_string)

# Inspect CRS
print(crs_utm33.name)                # "WGS 84 / UTM zone 33N"
print(crs_utm33.is_projected)        # True
print(crs_utm33.axis_info)           # Axis details
print(crs_utm33.area_of_use)         # Geographic bounds
print(crs_utm33.to_epsg())           # 32633

# --- Coordinate transformations ---
# Always create Transformer once, reuse many times
transformer = Transformer.from_crs(
    "EPSG:4326", "EPSG:32633", always_xy=True  # lon/lat order
)

# Transform single coordinate
x, y = transformer.transform(13.4, 52.5)

# Transform arrays (vectorized, fast)
import numpy as np
lons = np.array([13.4, 13.5, 13.6])
lats = np.array([52.5, 52.6, 52.7])
xs, ys = transformer.transform(lons, lats)

# --- Find the right UTM zone automatically ---
utm_info = query_utm_crs_info(
    datum_name="WGS 84",
    area_of_interest=AreaOfInterest(
        west_lon_degree=13.0, south_lat_degree=52.0,
        east_lon_degree=14.0, north_lat_degree=53.0,
    ),
)
utm_crs = CRS.from_authority(utm_info[0].auth_name, utm_info[0].code)
print(utm_crs)  # EPSG:32633

# --- Geodesic calculations ---
from pyproj import Geod

geod = Geod(ellps="WGS84")
# Forward azimuth, back azimuth, distance between two points
az12, az21, dist = geod.inv(lon1, lat1, lon2, lat2)

# Geodesic line (great circle) with intermediate points
lonlats = geod.npts(lon1, lat1, lon2, lat2, npts=100)

# Area of a polygon on the ellipsoid
poly_area, poly_perimeter = geod.geometry_area_perimeter(shapely_polygon)
```

### GeoAlchemy2

SQLAlchemy extension for PostGIS. Enables spatial queries through Python's ORM.

```python
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.orm import declarative_base, Session
from geoalchemy2 import Geometry, WKTElement
from geoalchemy2.shape import from_shape, to_shape
from shapely.geometry import Point, Polygon

Base = declarative_base()
engine = create_engine("postgresql://user:pass@localhost:5432/gisdb")

# --- Define a spatial model ---
class Building(Base):
    __tablename__ = "buildings"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    height = Column(Integer)
    geom = Column(Geometry("POLYGON", srid=4326, spatial_index=True))

Base.metadata.create_all(engine)

# --- Insert spatial data ---
with Session(engine) as session:
    building = Building(
        name="HQ",
        height=42,
        geom=from_shape(Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]), srid=4326),
    )
    session.add(building)
    session.commit()

# --- Spatial queries ---
with Session(engine) as session:
    # Find buildings within 1km of a point
    point = WKTElement("POINT(13.4 52.5)", srid=4326)
    nearby = session.query(Building).filter(
        func.ST_DWithin(
            func.ST_Transform(Building.geom, 32633),
            func.ST_Transform(point, 32633),
            1000,  # meters
        )
    ).all()

    # Area calculation
    areas = session.query(
        Building.name,
        func.ST_Area(func.ST_Transform(Building.geom, 32633)).label("area_m2"),
    ).all()

    # Spatial join
    result = session.query(Building, Zone).filter(
        func.ST_Within(Building.geom, Zone.geom)
    ).all()

    # Convert result to Shapely
    for b in nearby:
        shapely_geom = to_shape(b.geom)
```

---

## 2. GeoArrow & GeoParquet Ecosystem

GeoParquet and GeoArrow are the modern columnar formats for geospatial data. They provide dramatically faster I/O, smaller file sizes, and zero-copy interoperability across the Arrow ecosystem (Python, R, Rust, DuckDB, WASM).

See also: [Data Sources - Vector Data](../data-sources/vector-data.md) for datasets already available in GeoParquet.

### Format Comparison Benchmarks

**Test dataset: 2.3 million building footprints, mixed polygon complexity**

| Format | File Size | Write Time | Read Time | Spatial Filter Read |
|---|---|---|---|---|
| Shapefile (.shp) | 890 MB | 48 s | 32 s | 32 s (full scan) |
| GeoJSON (.geojson) | 1.4 GB | 62 s | 55 s | 55 s (full scan) |
| GeoPackage (.gpkg) | 720 MB | 35 s | 18 s | 2.1 s (spatial index) |
| FlatGeobuf (.fgb) | 650 MB | 22 s | 12 s | 0.8 s (built-in index) |
| GeoParquet (.parquet) | 380 MB | 8 s | 3.5 s | 1.2 s (row group pruning) |
| GeoParquet (zstd) | 280 MB | 10 s | 4.0 s | 1.4 s (row group pruning) |

**Key takeaways:**
- GeoParquet is 3-8x faster than Shapefile for read/write
- GeoParquet files are 2-5x smaller than Shapefile
- FlatGeobuf has the fastest spatial filter performance (HTTP range requests)
- GeoPackage is a good middle ground with broad tool support

### GeoArrow Python APIs

```python
# --- geoarrow-pyarrow: Arrow-native geometry arrays ---
import pyarrow as pa
import geoarrow.pyarrow as ga
import geopandas as gpd

# Convert GeoDataFrame to Arrow table with GeoArrow encoding
gdf = gpd.read_file("buildings.gpkg")
table = ga.to_geoarrow(pa.Table.from_pandas(gdf))

# Write GeoParquet with GeoArrow encoding
import pyarrow.parquet as pq
pq.write_table(table, "buildings.parquet")

# --- geoarrow-pandas: GeoArrow-backed pandas extension ---
import geoarrow.pandas as gap

# Read GeoParquet with Arrow-backed geometry column
gdf = gpd.read_parquet("buildings.parquet")

# The geometry column uses Arrow memory layout internally
# This enables zero-copy exchange with DuckDB, Polars, etc.

# --- GeoPandas 1.0 native GeoParquet ---
gdf = gpd.read_parquet("buildings.parquet")

# Write with specific compression and row group size
gdf.to_parquet(
    "buildings_optimized.parquet",
    compression="zstd",
    row_group_size=65536,
)

# --- DuckDB reads GeoParquet natively ---
import duckdb

con = duckdb.connect()
con.install_extension("spatial")
con.load_extension("spatial")

# Direct GeoParquet query (no Python conversion needed)
result = con.sql("""
    SELECT type, COUNT(*) as cnt, SUM(ST_Area(geometry)) as total_area
    FROM read_parquet('buildings.parquet')
    WHERE ST_Within(
        geometry,
        ST_GeomFromText('POLYGON((13.3 52.4, 13.5 52.4, 13.5 52.6, 13.3 52.6, 13.3 52.4))')
    )
    GROUP BY type
    ORDER BY cnt DESC
""").fetchdf()
```

### Lonboard: GPU Visualization via GeoArrow

Lonboard uses GeoArrow internally for zero-copy GPU rendering of millions of features in Jupyter notebooks.

```python
from lonboard import Map, ScatterplotLayer, PathLayer, SolidPolygonLayer
import geopandas as gpd

gdf = gpd.read_parquet("buildings.parquet")

# Render 2M+ polygons in <1 second
layer = SolidPolygonLayer.from_geopandas(
    gdf,
    get_fill_color=[255, 100, 100, 180],
    get_line_color=[0, 0, 0, 255],
    get_line_width=1,
)
m = Map(layers=[layer])
m  # Display in Jupyter
```

---

## 3. Visualization Deep Dive

Choosing the right visualization library depends on your audience (print vs web), data size, and interactivity needs.

See also: [Visualization Guide](../visualization/README.md) | [Thematic Maps](../visualization/thematic-maps.md) | [Large-Scale Visualization](../visualization/large-scale-visualization.md)

**Decision matrix:**

| Scenario | Library | Why |
|---|---|---|
| Publication-quality static map | Matplotlib + Cartopy | Full typographic control, journal-ready |
| Quick interactive exploration | Folium or `.explore()` | Simple API, basemaps included |
| Dashboard with hover/click | Plotly or Pydeck | Rich interactivity, widget integration |
| 1M+ features in notebook | Lonboard | GPU-accelerated, GeoArrow backend |
| 10M+ points server-side | Datashader | Server-side rasterization |
| Linked views, multi-panel | hvPlot + GeoViews | HoloViz ecosystem |
| 3D terrain / buildings | Pydeck | deck.gl 3D layers |
| Animated time series | GeoViews + Panel | Temporal widgets |

### Matplotlib + Cartopy

For publication-quality maps with precise control over projections, labels, legends, and styling.

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import BoundaryNorm
from matplotlib import cm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas as gpd
import numpy as np

# --- Basic map with Cartopy projection ---
fig, ax = plt.subplots(
    figsize=(12, 8),
    subplot_kw={"projection": ccrs.LambertConformal(central_longitude=10)},
)

# Add natural features
ax.add_feature(cfeature.LAND, facecolor="#f0e6d2")
ax.add_feature(cfeature.OCEAN, facecolor="#c6e2ff")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linestyle="--", linewidth=0.3)
ax.add_feature(cfeature.RIVERS, edgecolor="#4a90d9", linewidth=0.3)
ax.add_feature(cfeature.LAKES, facecolor="#c6e2ff", edgecolor="#4a90d9")

# Plot GeoDataFrame on projected axes
gdf = gpd.read_file("countries.gpkg")
gdf.plot(
    ax=ax,
    column="gdp_per_capita",
    cmap="YlOrRd",
    legend=True,
    legend_kwds={
        "label": "GDP per capita (USD)",
        "orientation": "horizontal",
        "shrink": 0.6,
        "pad": 0.05,
    },
    edgecolor="black",
    linewidth=0.3,
    transform=ccrs.PlateCarree(),  # Source CRS of the data
)

# Gridlines with labels
gl = ax.gridlines(draw_labels=True, linewidth=0.2, alpha=0.5)
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

ax.set_extent([-15, 45, 35, 72], crs=ccrs.PlateCarree())
ax.set_title("GDP per Capita in Europe", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("europe_gdp.png", dpi=300, bbox_inches="tight")

# --- Multi-panel map comparison ---
fig, axes = plt.subplots(
    1, 3, figsize=(18, 6),
    subplot_kw={"projection": ccrs.PlateCarree()},
)

for ax, (year, color) in zip(axes, [("2000", "Blues"), ("2010", "Greens"), ("2020", "Reds")]):
    gdf_year = gdf[gdf["year"] == int(year)]
    gdf_year.plot(ax=ax, column="value", cmap=color, transform=ccrs.PlateCarree())
    ax.coastlines()
    ax.set_title(year)

plt.suptitle("Land Use Change 2000-2020", fontsize=16)
plt.tight_layout()
plt.savefig("landuse_comparison.png", dpi=300)

# --- Inset map ---
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={"projection": ccrs.PlateCarree()})
gdf.plot(ax=ax, color="lightblue", edgecolor="gray", transform=ccrs.PlateCarree())
ax.set_extent([5, 18, 47, 55])

# Add inset showing location in world
inset_ax = fig.add_axes([0.62, 0.65, 0.25, 0.25], projection=ccrs.PlateCarree())
inset_ax.set_global()
inset_ax.add_feature(cfeature.LAND, facecolor="lightgray")
inset_ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
inset_ax.plot([5, 18, 18, 5, 5], [47, 47, 55, 55, 47], color="red", linewidth=2,
              transform=ccrs.PlateCarree())
```

### Folium Advanced

Folium wraps Leaflet.js for rich interactive maps. Beyond basic markers and choropleths, it supports heatmaps, time-based visualization, draw controls, and custom plugins.

```python
import folium
from folium.plugins import (
    HeatMap, HeatMapWithTime, MarkerCluster, FastMarkerCluster,
    Draw, MeasureControl, Fullscreen, LocateControl,
    MiniMap, GroupedLayerControl, FloatImage,
    TimestampedGeoJson,
)
import geopandas as gpd
import json

# --- Choropleth with custom styling ---
gdf = gpd.read_file("districts.gpkg")
m = folium.Map(location=[52.5, 13.4], zoom_start=10, tiles="CartoDB positron")

choropleth = folium.Choropleth(
    geo_data=gdf.to_json(),
    data=gdf,
    columns=["district_id", "population_density"],
    key_on="feature.properties.district_id",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.3,
    legend_name="Population Density (people/km2)",
    nan_fill_color="white",
    bins=[0, 500, 1000, 2000, 5000, 10000, 20000],
)
choropleth.add_to(m)

# Add tooltips to choropleth
choropleth.geojson.add_child(
    folium.features.GeoJsonTooltip(
        fields=["name", "population_density", "area_km2"],
        aliases=["District:", "Density:", "Area (km2):"],
        style="font-size: 12px;",
    )
)

# --- Heatmap ---
heat_data = [[row.geometry.y, row.geometry.x, row["intensity"]]
             for _, row in points_gdf.iterrows()]
HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)

# --- Marker cluster for thousands of points ---
marker_cluster = MarkerCluster().add_to(m)
for _, row in poi_gdf.iterrows():
    folium.Marker(
        location=[row.geometry.y, row.geometry.x],
        popup=folium.Popup(f"<b>{row['name']}</b><br>{row['type']}", max_width=200),
        icon=folium.Icon(color="blue", icon="info-sign"),
    ).add_to(marker_cluster)

# --- Fast marker cluster (for 100k+ points) ---
callback = """
function(row) {
    var marker = L.marker(new L.LatLng(row[0], row[1]));
    marker.bindPopup(row[2]);
    return marker;
}
"""
FastMarkerCluster(
    data=[[r.geometry.y, r.geometry.x, r["name"]] for _, r in gdf.iterrows()],
    callback=callback,
).add_to(m)

# --- Layer control with multiple layers ---
fg_parks = folium.FeatureGroup(name="Parks", show=True)
fg_buildings = folium.FeatureGroup(name="Buildings", show=False)

folium.GeoJson(parks_gdf, style_function=lambda x: {"color": "green"}).add_to(fg_parks)
folium.GeoJson(buildings_gdf, style_function=lambda x: {"color": "gray"}).add_to(fg_buildings)

fg_parks.add_to(m)
fg_buildings.add_to(m)
folium.LayerControl().add_to(m)

# --- Interactive tools ---
Draw(export=True).add_to(m)
MeasureControl(primary_length_unit="meters").add_to(m)
Fullscreen().add_to(m)
MiniMap().add_to(m)

m.save("interactive_map.html")
```

### Plotly Mapbox

Plotly provides interactive maps via Mapbox GL JS. Excellent for dashboards and data exploration.

```python
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import pandas as pd

# --- Scatter map ---
df = pd.read_csv("stations.csv")  # lat, lon, value columns
fig = px.scatter_mapbox(
    df, lat="lat", lon="lon", color="value", size="magnitude",
    color_continuous_scale="Viridis",
    mapbox_style="carto-positron",
    zoom=5, center={"lat": 52, "lon": 10},
    hover_data=["station_name", "value"],
    title="Weather Stations",
)
fig.update_layout(height=600)
fig.show()

# --- Choropleth map from GeoDataFrame ---
gdf = gpd.read_file("counties.gpkg")
fig = px.choropleth_mapbox(
    gdf, geojson=gdf.geometry.__geo_interface__,
    locations=gdf.index, color="unemployment_rate",
    color_continuous_scale="RdYlBu_r",
    mapbox_style="carto-positron",
    center={"lat": 51, "lon": 10}, zoom=5,
    opacity=0.6,
    labels={"unemployment_rate": "Unemployment (%)"},
)
fig.show()

# --- Density mapbox (kernel density) ---
fig = px.density_mapbox(
    df, lat="lat", lon="lon", z="value",
    radius=20, zoom=8,
    mapbox_style="carto-darkmatter",
    color_continuous_scale="Hot",
)
fig.show()

# --- Line map (trajectories) ---
fig = px.line_mapbox(
    trajectory_df, lat="lat", lon="lon", color="vehicle_id",
    mapbox_style="open-street-map",
    zoom=10,
)
fig.show()

# --- Multi-trace with graph_objects ---
fig = go.Figure()
fig.add_trace(go.Scattermapbox(
    lat=df["lat"], lon=df["lon"],
    mode="markers+text",
    marker=dict(size=12, color=df["value"], colorscale="Viridis", showscale=True),
    text=df["label"],
    textposition="top center",
))
fig.update_layout(
    mapbox=dict(style="carto-positron", center=dict(lat=52, lon=13), zoom=10),
    height=600,
)
fig.show()
```

### Pydeck Layers

Pydeck wraps deck.gl for GPU-accelerated 3D web maps. It supports a rich set of layer types.

```python
import pydeck as pdk
import geopandas as gpd
import pandas as pd

# --- ScatterplotLayer ---
scatter = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["lon", "lat"],
    get_radius=100,
    get_fill_color=[255, 140, 0, 180],
    pickable=True,
)

# --- HexagonLayer (3D aggregation) ---
hexagon = pdk.Layer(
    "HexagonLayer",
    data=df,
    get_position=["lon", "lat"],
    radius=200,
    elevation_scale=4,
    elevation_range=[0, 1000],
    extruded=True,
    coverage=0.8,
)

# --- GeoJsonLayer ---
geojson_layer = pdk.Layer(
    "GeoJsonLayer",
    data=gdf.__geo_interface__,
    opacity=0.6,
    stroked=True,
    filled=True,
    extruded=True,
    wireframe=True,
    get_elevation="properties.height",
    get_fill_color="[255, 255, properties.height * 2, 180]",
    get_line_color=[0, 0, 0],
    pickable=True,
)

# --- PathLayer (trajectories) ---
path_layer = pdk.Layer(
    "PathLayer",
    data=paths_df,
    get_path="coordinates",
    get_color="[255, 0, 0, 200]",
    width_min_pixels=2,
    get_width=5,
)

# --- ColumnLayer (3D bars on map) ---
column = pdk.Layer(
    "ColumnLayer",
    data=df,
    get_position=["lon", "lat"],
    get_elevation="value",
    elevation_scale=100,
    radius=200,
    get_fill_color=["value * 2", "255 - value", 0, 200],
    pickable=True,
)

# --- TripsLayer (animated trajectories) ---
trips = pdk.Layer(
    "TripsLayer",
    data=trip_data,
    get_path="waypoints",
    get_timestamps="timestamps",
    get_color=[253, 128, 93],
    width_min_pixels=5,
    trail_length=200,
    current_time=500,
)

# --- Combine layers and render ---
view = pdk.ViewState(latitude=52.5, longitude=13.4, zoom=11, pitch=45, bearing=0)
deck = pdk.Deck(
    layers=[geojson_layer, scatter],
    initial_view_state=view,
    tooltip={"text": "{name}\nHeight: {height}m"},
    map_style="mapbox://styles/mapbox/dark-v10",
)
deck.to_html("pydeck_map.html")
```

### Lonboard GPU Maps

Lonboard renders millions of features in Jupyter via WebGL using GeoArrow data. It is the fastest option for large vector datasets in notebooks.

```python
from lonboard import Map, ScatterplotLayer, PathLayer, SolidPolygonLayer, BitmapLayer
from lonboard.colormap import apply_continuous_cmap
from matplotlib import colormaps
import geopandas as gpd
import numpy as np

gdf = gpd.read_parquet("buildings.parquet")  # 2M features

# --- Color by attribute ---
values = gdf["height"].values
normalized = (values - values.min()) / (values.max() - values.min())
colors = apply_continuous_cmap(normalized, colormaps["viridis"])

layer = SolidPolygonLayer.from_geopandas(
    gdf,
    get_fill_color=colors,
    get_line_color=[0, 0, 0, 50],
)

m = Map(layers=[layer], _height=600)
m  # Display in Jupyter -- renders 2M polygons in <1s

# --- Scatterplot for large point datasets ---
points_gdf = gpd.read_parquet("taxi_pickups.parquet")  # 5M points
scatter = ScatterplotLayer.from_geopandas(
    points_gdf,
    get_radius=10,
    get_fill_color=[255, 0, 0, 100],
    radius_min_pixels=1,
)
Map(layers=[scatter])

# --- Multiple layers ---
buildings = SolidPolygonLayer.from_geopandas(buildings_gdf, get_fill_color=[200, 200, 200, 180])
roads = PathLayer.from_geopandas(roads_gdf, get_color=[50, 50, 50, 255], get_width=2)
Map(layers=[buildings, roads])

# --- Dynamic updates (widget integration) ---
import ipywidgets as widgets

slider = widgets.FloatSlider(min=0, max=50, step=1, description="Min height:")

def update(change):
    mask = gdf["height"] >= change["new"]
    layer.from_geopandas(gdf[mask])

slider.observe(update, names="value")
widgets.VBox([slider, m])
```

### Datashader Big Data

Datashader performs server-side rasterization of millions to billions of data points. It renders aggregated pixels instead of individual features.

```python
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
import pandas as pd
import colorcet as cc

# --- Render 100M GPS points ---
df = pd.read_parquet("gps_traces.parquet")  # columns: x, y (projected coords)

canvas = ds.Canvas(plot_width=1200, plot_height=800)
agg = canvas.points(df, "x", "y")
img = tf.shade(agg, cmap=cc.fire, how="log")
export_image(img, "gps_heatmap")

# --- Aggregate by category ---
agg = canvas.points(df, "x", "y", agg=ds.count_cat("vehicle_type"))
img = tf.shade(agg, color_key={"car": "blue", "truck": "red", "bus": "green"})

# --- Line rendering (road network) ---
agg = canvas.line(roads_df, "x", "y", agg=ds.count())
img = tf.shade(agg, cmap=cc.blues)

# --- With Matplotlib + contextily ---
import matplotlib.pyplot as plt
import contextily as ctx

fig, ax = plt.subplots(figsize=(12, 8))
agg = canvas.points(df, "x", "y")
img = tf.shade(agg, cmap=cc.fire, how="log")
ax.imshow(img.to_pil(), extent=[df.x.min(), df.x.max(), df.y.min(), df.y.max()])
ctx.add_basemap(ax, crs="EPSG:3857", source=ctx.providers.CartoDB.DarkMatter)
plt.savefig("datashader_basemap.png", dpi=200)

# --- With HoloViews for interactive panning/zooming ---
import holoviews as hv
from holoviews.operation.datashader import datashade, dynspread
hv.extension("bokeh")

points = hv.Points(df, kdims=["x", "y"])
shaded = datashade(points, cmap=cc.fire).opts(width=1000, height=600)
dynspread(shaded)  # Spread pixels for visibility at low zoom
```

### hvPlot + GeoViews

HoloViz ecosystem for interactive, linked, multi-panel geospatial visualization.

```python
import hvplot.pandas
import geopandas as gpd
import geoviews as gv
import holoviews as hv
from holoviews import opts
hv.extension("bokeh")

gdf = gpd.read_file("cities.gpkg")

# --- Quick interactive map with hvPlot ---
gdf.hvplot.points(
    "longitude", "latitude",
    geo=True, tiles="CartoDB positron",
    color="population", cmap="viridis",
    size=dim("population") / 10000,
    hover_cols=["name", "country", "population"],
    width=800, height=500,
    title="World Cities by Population",
)

# --- Choropleth ---
gdf.hvplot(
    geo=True, tiles="CartoDB positron",
    c="gdp_per_capita", cmap="YlOrRd",
    hover_cols=["name", "gdp_per_capita"],
    width=900, height=500,
)

# --- Linked views ---
map_view = gdf.hvplot.points(
    "lon", "lat", geo=True, tiles=True,
    color="category", width=500, height=400,
)
hist = gdf.hvplot.hist("value", bins=30, width=400, height=400)
scatter = gdf.hvplot.scatter("area", "population", color="category", width=400, height=400)

# Linked selection: brush on one plot filters all others
(map_view + hist + scatter).cols(2).opts(shared_axes=False)

# --- GeoViews: advanced geospatial plotting ---
import geoviews.feature as gf

coastlines = gf.coastline.opts(line_width=0.5)
borders = gf.borders.opts(line_width=0.3, line_dash="dashed")
land = gf.land.opts(fill_color="#f0e6d2")
ocean = gf.ocean.opts(fill_color="#c6e2ff")

points = gv.Points(gdf, kdims=["longitude", "latitude"], vdims=["name", "value"])
styled = points.opts(
    color="value", cmap="hot", size=8,
    tools=["hover"], width=900, height=500,
)

(ocean * land * coastlines * borders * styled)
```

---

## 4. Analysis Libraries

### PySAL Complete Ecosystem

PySAL (Python Spatial Analysis Library) is a modular ecosystem of packages for spatial data science. Each sub-package focuses on a specific analytical domain.

```
pysal (meta-package)
  |-- lib          # Core: weights, spatial lag, spatial graphs
  |-- explore
  |     |-- esda      # Exploratory spatial data analysis (Moran's I, LISA, G*)
  |     |-- pointpats # Point pattern analysis (K, G, F functions)
  |     |-- inequality# Spatial inequality measures
  |     |-- segregation# Segregation indices
  |     |-- spaghetti # Network-constrained spatial analysis
  |-- model
  |     |-- spreg     # Spatial regression (OLS, 2SLS, ML, GM)
  |     |-- mgwr      # Multi-scale GWR
  |     |-- spglm     # Spatial GLMs
  |     |-- spint     # Spatial interaction models
  |     |-- spopt     # Spatial optimization (facility location, regionalization)
  |     |-- tobler    # Areal interpolation
  |     |-- access    # Spatial access / accessibility measures
  |-- viz
        |-- splot     # Static visualizations for PySAL analytics
        |-- mapclassify # Classification schemes for choropleth maps
```

```python
# --- Spatial weights ---
from libpysal.weights import Queen, Rook, KNN, DistanceBand, Kernel
import geopandas as gpd

gdf = gpd.read_file("census_tracts.gpkg")

w_queen = Queen.from_dataframe(gdf)        # Queen contiguity
w_rook = Rook.from_dataframe(gdf)          # Rook contiguity
w_knn = KNN.from_dataframe(gdf, k=5)      # 5 nearest neighbors
w_dist = DistanceBand.from_dataframe(gdf, threshold=5000)  # 5km band
w_kernel = Kernel.from_dataframe(gdf, bandwidth=10000)     # Kernel weights

# Row-standardize
w_queen.transform = "r"

# --- Exploratory Spatial Data Analysis (esda) ---
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local
from splot.esda import moran_scatterplot, lisa_cluster

# Global Moran's I
moran = Moran(gdf["income"], w_queen)
print(f"Moran's I: {moran.I:.4f}, p-value: {moran.p_sim:.4f}")

# Local Moran's I (LISA)
lisa = Moran_Local(gdf["income"], w_queen)
gdf["lisa_q"] = lisa.q            # Quadrant (1=HH, 2=LH, 3=LL, 4=HL)
gdf["lisa_p"] = lisa.p_sim        # Pseudo p-value
gdf["lisa_sig"] = lisa.p_sim < 0.05  # Significant clusters

# Getis-Ord Gi* (hot/cold spots)
g_local = G_Local(gdf["income"], w_queen, star=True)
gdf["g_z"] = g_local.Zs
gdf["hotspot"] = (g_local.Zs > 1.96).astype(int) - (g_local.Zs < -1.96).astype(int)

# Visualization
fig, ax = moran_scatterplot(moran, aspect_equal=True)
fig, ax = lisa_cluster(lisa, gdf, p=0.05, figsize=(10, 8))

# --- Spatial regression (spreg) ---
from spreg import OLS, ML_Lag, ML_Error, GM_Lag, GM_Error, TSLS

# OLS with diagnostics for spatial dependence
ols = OLS(
    gdf[["income"]].values,
    gdf[["education", "employment", "density"]].values,
    w=w_queen,
    name_y="income",
    name_x=["education", "employment", "density"],
    spat_diag=True,   # LM tests for spatial lag/error
    moran=True,        # Moran's I on residuals
)
print(ols.summary)

# Spatial lag model (ML estimation)
lag = ML_Lag(
    gdf[["income"]].values,
    gdf[["education", "employment", "density"]].values,
    w=w_queen,
    name_y="income",
    name_x=["education", "employment", "density"],
)
print(f"Rho (spatial lag): {lag.rho:.4f}")
print(lag.summary)

# Spatial error model
error = ML_Error(
    gdf[["income"]].values,
    gdf[["education", "employment", "density"]].values,
    w=w_queen,
)
print(f"Lambda (spatial error): {error.lam:.4f}")

# --- Multi-scale GWR (mgwr) ---
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import numpy as np

coords = np.column_stack([gdf.geometry.centroid.x, gdf.geometry.centroid.y])
y = gdf["income"].values.reshape(-1, 1)
X = gdf[["education", "employment", "density"]].values

# GWR: single bandwidth
bw = Sel_BW(coords, y, X).search()
gwr = GWR(coords, y, X, bw).fit()
print(gwr.summary())

# MGWR: variable-specific bandwidths
selector = Sel_BW(coords, y, X, multi=True)
bw_mgwr = selector.search()
mgwr = MGWR(coords, y, X, selector).fit()
print(mgwr.summary())

# Map local coefficients
gdf["beta_education"] = mgwr.params[:, 1]
gdf.plot(column="beta_education", cmap="RdBu", legend=True)

# --- Segregation analysis ---
from segregation.singlegroup import Dissimilarity, Gini, Entropy, Isolation
from segregation.multigroup import MultiDissimilarity, MultiInfoTheory

dissim = Dissimilarity(gdf, "minority_pop", "total_pop")
print(f"Dissimilarity index: {dissim.statistic:.4f}")

gini = Gini(gdf, "minority_pop", "total_pop")
entropy = Entropy(gdf, "minority_pop", "total_pop")

# --- Accessibility (access) ---
from access import Access, weights as access_weights

A = Access(
    demand_df=gdf,
    demand_index="tract_id",
    demand_value="population",
    supply_df=hospitals_gdf,
    supply_index="hospital_id",
    supply_value="beds",
    cost_df=travel_time_df,
    cost_origin="tract_id",
    cost_dest="hospital_id",
    cost_name="travel_time_min",
)

# Two-step floating catchment area (2SFCA)
A.two_stage_fca(name="2sfca", max_cost=30)

# Enhanced 2SFCA with distance decay
A.enhanced_two_stage_fca(name="e2sfca", max_cost=30)

# --- Point pattern analysis (pointpats) ---
from pointpats import PointPattern, PoissonPointProcess
from pointpats import k_test, g_test, f_test, j_test

pp = PointPattern(gdf[["x", "y"]].values)
print(f"N points: {pp.n}, Area: {pp.mbb_area:.2f}")
print(f"Mean NN distance: {pp.mean_nnd:.4f}")
print(f"Min NN distance: {pp.min_nnd:.4f}")

# Complete Spatial Randomness tests
k_result = k_test(pp.points, support=(0, pp.max_nnd, 50))
g_result = g_test(pp.points)

# --- Map classification (mapclassify) ---
from mapclassify import (
    NaturalBreaks, FisherJenks, Quantiles, EqualInterval,
    MaximumBreaks, StdMean, BoxPlot, HeadTailBreaks,
)

classifier = NaturalBreaks(gdf["income"], k=5)
gdf["income_class"] = classifier.yb
print(classifier)  # Shows break points, counts per class

# Use with GeoPandas plot
gdf.plot(column="income", scheme="naturalbreaks", k=5, cmap="YlOrRd", legend=True)
gdf.plot(column="income", scheme="quantiles", k=5, cmap="Blues", legend=True)

# --- Areal interpolation (tobler) ---
from tobler.area_weighted import area_interpolate
from tobler.dasymetric import masked_area_interpolate

# Area-weighted interpolation (source zones -> target zones)
interpolated = area_interpolate(
    source_gdf, target_gdf,
    extensive_variables=["population", "housing_units"],
    intensive_variables=["median_income"],
)

# Dasymetric interpolation (using ancillary land use data)
dasy = masked_area_interpolate(
    source_gdf, target_gdf,
    extensive_variables=["population"],
    raster="land_use.tif",  # Binary mask: 1=developed, 0=undeveloped
)

# --- Spatial optimization (spopt) ---
from spopt.locate import PMedian, MCLP, LSCP
from spopt.region import MaxP, Skater, WardSpatial

# P-Median: minimize total demand-weighted distance
pmedian = PMedian.from_geodataframe(
    gdf_demand, gdf_supply,
    "demand", p_facilities=5,
    predefined_facility_col=None,
)
pmedian.solve()
print(f"Objective: {pmedian.problem.objective.value()}")
```

### scikit-learn Spatial Patterns

scikit-learn is the standard ML library. Combined with spatial features and proper spatial cross-validation, it excels at geospatial prediction tasks.

```python
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import geopandas as gpd

gdf = gpd.read_file("properties.gpkg")

# --- DBSCAN spatial clustering ---
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
db = DBSCAN(eps=500, min_samples=10, metric="euclidean")  # eps in CRS units
gdf["cluster"] = db.fit_predict(coords)

# DBSCAN with haversine distance (for lat/lon)
coords_rad = np.radians(coords)
db = DBSCAN(eps=0.5 / 6371, min_samples=10, metric="haversine")  # 500m
gdf["cluster"] = db.fit_predict(coords_rad)

# --- HDBSCAN (density-based, no eps needed) ---
hdb = HDBSCAN(min_cluster_size=15, min_samples=5)
gdf["cluster"] = hdb.fit_predict(coords)

# --- Spatial feature engineering ---
gdf["x"] = gdf.geometry.centroid.x
gdf["y"] = gdf.geometry.centroid.y
gdf["area"] = gdf.geometry.area
gdf["perimeter"] = gdf.geometry.length
gdf["compactness"] = 4 * np.pi * gdf["area"] / (gdf["perimeter"] ** 2)

# Distance to nearest POI
from scipy.spatial import cKDTree
poi_coords = np.column_stack([poi_gdf.geometry.x, poi_gdf.geometry.y])
tree = cKDTree(poi_coords)
dists, _ = tree.query(coords)
gdf["dist_to_poi"] = dists

# --- Random Forest with spatial features ---
features = ["area", "dist_to_poi", "x", "y", "elevation", "slope"]
X = gdf[features].values
y = gdf["price"].values

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)),
])

# IMPORTANT: spatial cross-validation (not random CV)
from sklearn.model_selection import GroupKFold
gdf["spatial_group"] = KMeans(n_clusters=5, random_state=42).fit_predict(coords)
cv = GroupKFold(n_splits=5)
scores = cross_val_score(pipe, X, y, cv=cv, groups=gdf["spatial_group"], scoring="r2")
print(f"Spatial CV R2: {scores.mean():.3f} +/- {scores.std():.3f}")
```

### SciPy Spatial

Low-level spatial data structures and algorithms.

```python
from scipy.spatial import (
    cKDTree, KDTree, Voronoi, Delaunay,
    ConvexHull, distance_matrix,
)
from scipy.spatial.distance import cdist
import numpy as np

coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

# --- KDTree for fast nearest neighbor ---
tree = cKDTree(coords)

# K nearest neighbors
distances, indices = tree.query(query_point, k=5)

# All points within radius
indices = tree.query_ball_point(query_point, r=1000)

# Sparse distance matrix (memory-efficient)
sparse_dist = tree.sparse_distance_matrix(tree, max_distance=5000)

# --- Voronoi tessellation ---
vor = Voronoi(coords)
# vor.vertices: Voronoi vertices
# vor.regions: polygons as vertex indices
# vor.ridge_vertices: edges

# Convert to GeoDataFrame
from shapely.geometry import Polygon
from shapely.ops import voronoi_diagram
voronoi_polys = voronoi_diagram(gdf.geometry.unary_union)

# --- Delaunay triangulation ---
tri = Delaunay(coords)
# tri.simplices: triangle vertex indices

# --- Convex hull ---
hull = ConvexHull(coords)
hull_polygon = Polygon(coords[hull.vertices])

# --- Distance matrix ---
dist_matrix = cdist(coords_a, coords_b, metric="euclidean")
```

### NetworkX + OSMnx

Graph-based spatial analysis. OSMnx downloads, models, and analyzes street networks from OpenStreetMap.

```python
import osmnx as ox
import networkx as nx

# --- Download street network ---
G = ox.graph_from_place("Berlin, Germany", network_type="drive")
G = ox.graph_from_bbox(52.55, 52.45, 13.5, 13.3, network_type="walk")
G = ox.graph_from_point((52.52, 13.405), dist=2000, network_type="bike")

# --- Basic graph stats ---
stats = ox.basic_stats(G)
print(f"Nodes: {stats['n']}, Edges: {stats['m']}")
print(f"Total street length: {stats['street_length_total']:.0f} m")
print(f"Average circuity: {stats['circuity_avg']:.3f}")

# --- Shortest path ---
orig = ox.nearest_nodes(G, 13.38, 52.52)
dest = ox.nearest_nodes(G, 13.45, 52.50)
route = nx.shortest_path(G, orig, dest, weight="length")
route_length = nx.shortest_path_length(G, orig, dest, weight="length")

# Visualize route
fig, ax = ox.plot_graph_route(G, route, route_linewidth=3, node_size=0)

# --- Isochrones (travel time polygons) ---
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

center = ox.nearest_nodes(G, 13.405, 52.52)
travel_times = nx.single_source_dijkstra_path_length(G, center, weight="travel_time")

# Extract nodes reachable within 10 minutes
subgraph = G.subgraph([n for n, t in travel_times.items() if t <= 600])

# --- Network centrality ---
bc = nx.betweenness_centrality(G, weight="length")
cc = nx.closeness_centrality(G, distance="length")

# --- Download other OSM data ---
buildings = ox.features_from_place("Berlin, Germany", tags={"building": True})
parks = ox.features_from_place("Berlin, Germany", tags={"leisure": "park"})
shops = ox.features_from_place("Berlin, Germany", tags={"shop": True})

# --- Convert to GeoDataFrame ---
nodes, edges = ox.graph_to_gdfs(G)
edges.to_parquet("berlin_streets.parquet")
```

### MovingPandas

Trajectory analysis for movement data (GPS tracks, vessel AIS, animal telemetry).

```python
import movingpandas as mpd
import geopandas as gpd
import pandas as pd
from datetime import timedelta

# --- Create trajectory from GPS data ---
df = pd.read_csv("gps_tracks.csv", parse_dates=["timestamp"])
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326"
)

traj = mpd.Trajectory(gdf, traj_id="vehicle_001", t="timestamp")

# Create collection from multiple trajectories
tc = mpd.TrajectoryCollection(gdf, traj_id_col="vehicle_id", t="timestamp")
print(f"Number of trajectories: {len(tc.trajectories)}")

# --- Trajectory properties ---
print(f"Duration: {traj.get_duration()}")
print(f"Length: {traj.get_length():.0f} m")
print(f"Start: {traj.get_start_location()}")
print(f"End: {traj.get_end_location()}")

# Speed and direction
traj.add_speed(overwrite=True, units=("km", "h"))
traj.add_direction(overwrite=True)
traj.add_acceleration(overwrite=True)

# --- Splitting ---
# Split by observation gaps
split = mpd.ObservationGapSplitter(traj).split(gap=timedelta(minutes=15))

# Split by stop detection
stops = mpd.StopSplitter(traj).split(
    max_diameter=100,  # meters
    min_duration=timedelta(minutes=5),
)

# Split by speed
segments = mpd.SpeedSplitter(traj).split(speed=5, duration=timedelta(minutes=2))

# --- Generalization ---
simplified = mpd.DouglasPeuckerGeneralizer(traj).generalize(tolerance=50)
smoothed = mpd.MinTimeDeltaGeneralizer(traj).generalize(tolerance=timedelta(seconds=30))

# --- Aggregation ---
# Flow map from trajectory collection
from movingpandas.trajectory_utils import convert_to_flow_df
flows = tc.get_flow(origin_col="origin", destination_col="destination")

# --- Visualization ---
traj.plot(column="speed", cmap="RdYlGn_r", legend=True)
tc.explore(column="speed", cmap="hot", tiles="CartoDB positron")
```

---

## 5. Cloud & Big Data

### STAC Workflow

The SpatioTemporal Asset Catalog (STAC) standard enables searching and loading cloud-hosted Earth observation data. The typical Python workflow: pystac-client (search) -> stackstac (load) -> xarray + rioxarray (analyze).

```python
from pystac_client import Client
import stackstac
import xarray as xr
import rioxarray

# --- Search for Sentinel-2 imagery ---
catalog = Client.open("https://earth-search.aws.element84.com/v1")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[13.0, 52.3, 13.8, 52.7],  # Berlin area
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 20}},
    max_items=50,
)

items = search.item_collection()
print(f"Found {len(items)} scenes")

# --- Load as xarray DataArray ---
stack = stackstac.stack(
    items,
    assets=["red", "green", "blue", "nir"],
    resolution=10,           # 10m per pixel
    bounds_latlon=[13.2, 52.4, 13.6, 52.6],
    epsg=32633,
    dtype="float32",
    fill_value=0,
    chunksize=(1, 1, 512, 512),  # Dask chunks: (time, band, y, x)
)
print(stack)  # Lazy -- no data loaded yet

# --- Compute NDVI ---
red = stack.sel(band="red")
nir = stack.sel(band="nir")
ndvi = (nir - red) / (nir + red)

# Temporal composite: median NDVI for summer 2024
ndvi_median = ndvi.median(dim="time").compute()  # Triggers actual loading

# --- Save result ---
ndvi_median.rio.to_raster("ndvi_summer_2024.tif", driver="COG")

# --- With Planetary Computer ---
import planetary_computer as pc

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,  # Auto-sign URLs
)

search = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=[13.0, 52.3, 13.8, 52.7],
    datetime="2024-01-01/2024-12-31",
    query={"eo:cloud_cover": {"lt": 15}},
)
items = search.item_collection()

# Load Landsat with stackstac
stack = stackstac.stack(items, assets=["red", "green", "blue", "nir08"])
```

### Dask-GeoPandas

Parallel and out-of-core processing for vector data using Dask's distributed computing model.

```python
import dask_geopandas
import geopandas as gpd
import dask.dataframe as dd

# --- From GeoDataFrame ---
gdf = gpd.read_parquet("buildings.parquet")
ddf = dask_geopandas.from_geopandas(gdf, npartitions=16)

# --- From GeoParquet (lazy, partitioned) ---
ddf = dask_geopandas.read_parquet("buildings_partitioned/")

# --- Operations (lazy until .compute()) ---
# Buffer
buffered = ddf.geometry.buffer(100)

# Spatial properties
ddf["area"] = ddf.geometry.area
ddf["centroid"] = ddf.geometry.centroid

# Filter
large_buildings = ddf[ddf.geometry.area > 500]

# Dissolve
dissolved = ddf.dissolve(by="district", aggfunc="sum")

# Spatial join (requires spatial partitioning)
ddf = ddf.spatial_shuffle()  # Hilbert curve spatial partitioning
result = dask_geopandas.sjoin(ddf, zones_ddf)

# --- Compute (triggers execution) ---
result_gdf = result.compute()

# --- Write partitioned GeoParquet ---
ddf.to_parquet("output_partitioned/", write_index=False)
```

### DuckDB Spatial

DuckDB with the spatial extension provides PostGIS-like SQL on local files. No server needed. Reads GeoParquet, Shapefile, GeoJSON, and FlatGeobuf natively.

```python
import duckdb

con = duckdb.connect()
con.install_extension("spatial")
con.load_extension("spatial")
con.install_extension("httpfs")
con.load_extension("httpfs")

# --- Read and query GeoParquet ---
result = con.sql("""
    SELECT
        type,
        COUNT(*) as count,
        SUM(ST_Area(ST_Transform(geometry, 'EPSG:4326', 'EPSG:32633'))) as total_area_m2,
        AVG(height) as avg_height
    FROM read_parquet('buildings.parquet')
    WHERE height > 10
    GROUP BY type
    ORDER BY count DESC
""").fetchdf()

# --- Spatial join ---
con.sql("""
    SELECT b.name, z.zone_name, b.height
    FROM read_parquet('buildings.parquet') b
    JOIN read_parquet('zones.parquet') z
      ON ST_Within(b.geometry, z.geometry)
    WHERE b.height > 20
""")

# --- Read from HTTP (remote GeoParquet) ---
con.sql("""
    SELECT *
    FROM read_parquet('https://data.source.coop/file.parquet')
    WHERE ST_Within(
        geometry,
        ST_GeomFromText('POLYGON((13.3 52.4, 13.5 52.4, 13.5 52.6, 13.3 52.6, 13.3 52.4))')
    )
""")

# --- Export to GeoParquet ---
con.sql("""
    COPY (
        SELECT * FROM read_parquet('input.parquet')
        WHERE ST_Area(geometry) > 100
    ) TO 'filtered.parquet' (FORMAT PARQUET)
""")

# --- Interop with GeoPandas ---
gdf = gpd.read_parquet("data.parquet")
con.register("my_table", gdf)
result = con.sql("SELECT * FROM my_table WHERE population > 10000").fetchdf()

# Convert back
result_gdf = gpd.GeoDataFrame(result, geometry="geometry")
```

### Google Earth Engine

Server-side computation on petabytes of Earth observation data.

```python
import ee
import geemap

# Authenticate and initialize
ee.Authenticate()
ee.Initialize(project="my-project-id")

# --- Load and filter imagery ---
s2 = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(ee.Geometry.Point(13.4, 52.5))
    .filterDate("2024-06-01", "2024-08-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
)

print(f"Images found: {s2.size().getInfo()}")

# --- Compute NDVI composite ---
def add_ndvi(image):
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return image.addBands(ndvi)

ndvi_composite = s2.map(add_ndvi).select("NDVI").median()

# --- Export to Drive ---
task = ee.batch.Export.image.toDrive(
    image=ndvi_composite,
    description="Berlin_NDVI_2024",
    folder="ee_exports",
    region=ee.Geometry.Rectangle([13.0, 52.3, 13.8, 52.7]),
    scale=10,
    crs="EPSG:32633",
    maxPixels=1e9,
)
task.start()

# --- Interactive map with geemap ---
Map = geemap.Map(center=[52.5, 13.4], zoom=10)
Map.addLayer(ndvi_composite, {"min": 0, "max": 0.8, "palette": ["red", "yellow", "green"]}, "NDVI")
Map.addLayerControl()
Map

# --- Zonal statistics ---
zones = ee.FeatureCollection("users/myuser/districts")
stats = ndvi_composite.reduceRegions(
    collection=zones,
    reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(), sharedInputs=True),
    scale=10,
)
# Export to pandas
stats_df = geemap.ee_to_df(stats)

# --- Land cover classification ---
training = ndvi_composite.sampleRegions(
    collection=training_points,
    properties=["class"],
    scale=10,
)
classifier = ee.Classifier.smileRandomForest(100).train(
    features=training, classProperty="class", inputProperties=["NDVI"]
)
classified = ndvi_composite.classify(classifier)
```

### Planetary Computer SDK

Microsoft's cloud platform for geospatial data with STAC-based catalog access.

```python
import planetary_computer as pc
from pystac_client import Client
import stackstac

catalog = Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace,
)

# Available collections
collections = catalog.get_collections()
for c in collections:
    print(f"{c.id}: {c.title}")

# Search Sentinel-2
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[13.0, 52.3, 13.8, 52.7],
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 10}},
)

items = search.item_collection()
stack = stackstac.stack(items, assets=["B04", "B03", "B02", "B08"], resolution=10)

# Process with xarray
rgb = stack.sel(band=["B04", "B03", "B02"]).median(dim="time").compute()
```

### Apache Sedona

Distributed spatial computing on Spark clusters. For datasets too large for a single machine.

```python
from sedona.spark import SedonaContext

sedona = SedonaContext.builder().getOrCreate()

# Read spatial data
df = sedona.read.format("geoparquet").load("s3://bucket/buildings.parquet")

# Spatial SQL
df.createOrReplaceTempView("buildings")
result = sedona.sql("""
    SELECT zone_id, COUNT(*) as building_count,
           AVG(ST_Area(geometry)) as avg_area
    FROM buildings
    WHERE ST_Within(geometry, ST_GeomFromWKT('POLYGON((...))'))
    GROUP BY zone_id
""")

# Spatial join at scale (billions of features)
points_df.createOrReplaceTempView("points")
zones_df.createOrReplaceTempView("zones")
joined = sedona.sql("""
    SELECT p.*, z.zone_name
    FROM points p, zones z
    WHERE ST_Contains(z.geometry, p.geometry)
""")
```

---

## 6. Raster Analysis Stack

### Rasterio Recipes

Production-tested recipes for common raster operations.

```python
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import Window, from_bounds
from rasterio.features import rasterize, shapes
from rasterio.enums import Resampling
import numpy as np
from shapely.geometry import mapping, shape
import geopandas as gpd

# --- Mosaic multiple rasters ---
src_files = ["tile_1.tif", "tile_2.tif", "tile_3.tif"]
src_datasets = [rasterio.open(f) for f in src_files]
mosaic, out_transform = merge(src_datasets, method="first")  # or "max", "min", "last"

out_meta = src_datasets[0].meta.copy()
out_meta.update({
    "height": mosaic.shape[1],
    "width": mosaic.shape[2],
    "transform": out_transform,
})

with rasterio.open("mosaic.tif", "w", **out_meta) as dst:
    dst.write(mosaic)

for ds in src_datasets:
    ds.close()

# --- Clip raster by vector ---
gdf = gpd.read_file("aoi.geojson")
geometries = [mapping(geom) for geom in gdf.geometry]

with rasterio.open("dem.tif") as src:
    clipped, clipped_transform = mask(
        src, geometries, crop=True, nodata=-9999, all_touched=True
    )
    out_meta = src.meta.copy()
    out_meta.update({
        "height": clipped.shape[1],
        "width": clipped.shape[2],
        "transform": clipped_transform,
        "nodata": -9999,
    })

with rasterio.open("dem_clipped.tif", "w", **out_meta) as dst:
    dst.write(clipped)

# --- Rasterize vector features ---
gdf = gpd.read_file("land_use.gpkg")

with rasterio.open("reference.tif") as ref:
    out_shape = (ref.height, ref.width)
    out_transform = ref.transform

rasterized = rasterize(
    [(mapping(geom), value) for geom, value in zip(gdf.geometry, gdf["class_id"])],
    out_shape=out_shape,
    transform=out_transform,
    fill=0,
    dtype="uint8",
)

# --- Vectorize raster to polygons ---
with rasterio.open("classified.tif") as src:
    data = src.read(1)
    mask_data = data != src.nodata

    results = list(shapes(data, mask=mask_data, transform=src.transform))
    geoms = [shape(geom) for geom, value in results]
    values = [value for geom, value in results]

    gdf = gpd.GeoDataFrame({"class": values, "geometry": geoms}, crs=src.crs)

# --- Hillshade ---
with rasterio.open("dem.tif") as src:
    dem = src.read(1).astype(float)
    cellsize = src.res[0]

dx = np.gradient(dem, cellsize, axis=1)
dy = np.gradient(dem, cellsize, axis=0)
slope = np.arctan(np.sqrt(dx**2 + dy**2))
aspect = np.arctan2(-dy, dx)

azimuth = np.radians(315)
altitude = np.radians(45)
hillshade = (
    np.sin(altitude) * np.cos(slope) +
    np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
)
hillshade = np.clip(hillshade * 255, 0, 255).astype(np.uint8)

# --- Windowed processing for large rasters ---
with rasterio.open("large_raster.tif") as src:
    profile = src.profile.copy()

    with rasterio.open("processed.tif", "w", **profile) as dst:
        for ji, window in src.block_windows(1):
            data = src.read(1, window=window)
            # Process block
            processed = np.where(data > 0, np.log1p(data), 0)
            dst.write(processed, 1, window=window)
```

### xarray-spatial

GPU-accelerated raster analysis on xarray DataArrays.

```python
import xarray as xr
import rioxarray
from xrspatial import slope, aspect, curvature, hillshade
from xrspatial import focal, zonal_stats
from xrspatial import proximity, viewshed
from xrspatial.classify import natural_breaks, equal_interval, quantile

# --- Load DEM ---
dem = rioxarray.open_rasterio("dem.tif").squeeze()

# --- Terrain analysis ---
slp = slope(dem)                   # Slope in degrees
asp = aspect(dem)                  # Aspect in degrees
curv = curvature(dem)              # Plan curvature
hs = hillshade(dem, azimuth=315, angle_altitude=45)

# --- Focal statistics ---
from xrspatial.focal import mean as focal_mean, apply as focal_apply
import numpy as np

smoothed = focal_mean(dem, kernel=np.ones((5, 5)))

# Custom focal kernel
gaussian = np.array([[1,2,1],[2,4,2],[1,2,1]]) / 16
filtered = focal_apply(dem, kernel=gaussian)

# --- Zonal statistics ---
zones = rioxarray.open_rasterio("zones.tif").squeeze()
zs = zonal_stats(zones, dem, stats_funcs=["mean", "std", "min", "max", "count"])

# --- Proximity (distance to features) ---
target = xr.where(dem < 100, 1, 0)  # Low elevation areas
dist = proximity(target)

# --- Classification ---
classified = natural_breaks(dem, k=5)
equal = equal_interval(dem, k=10)
quant = quantile(dem, k=4)
```

### rasterstats

Zonal statistics: summarize raster values within vector zones.

```python
from rasterstats import zonal_stats, point_query
import geopandas as gpd

gdf = gpd.read_file("parcels.gpkg")

# --- Zonal statistics ---
stats = zonal_stats(
    gdf,
    "elevation.tif",
    stats=["count", "min", "max", "mean", "median", "std", "sum", "range"],
    geojson_out=True,
)

# Add stats to GeoDataFrame
stats_df = gpd.GeoDataFrame.from_features(stats)

# --- With percentiles and custom stats ---
stats = zonal_stats(
    gdf, "temperature.tif",
    stats=["mean", "percentile_10", "percentile_90"],
    add_stats={"cv": lambda x: np.std(x) / np.mean(x) if np.mean(x) != 0 else 0},
)

# --- Categorical raster (land cover) ---
cat_stats = zonal_stats(gdf, "landcover.tif", categorical=True)
# Returns: [{"1": 450, "2": 120, "3": 30, ...}, ...]

# --- Point query ---
points_gdf = gpd.read_file("sample_points.gpkg")
values = point_query(points_gdf, "dem.tif", interpolate="bilinear")
points_gdf["elevation"] = values
```

### WhiteboxTools

Comprehensive open-source geospatial analysis platform with 500+ tools for terrain, hydrology, and image analysis.

```python
from whitebox import WhiteboxTools

wbt = WhiteboxTools()
wbt.set_verbose_mode(False)
wbt.set_working_dir("/path/to/data")

# --- Hydrological analysis ---
wbt.fill_depressions("dem.tif", "dem_filled.tif")
wbt.d8_pointer("dem_filled.tif", "flow_dir.tif")
wbt.d8_flow_accumulation("flow_dir.tif", "flow_acc.tif")
wbt.extract_streams("flow_acc.tif", "streams.tif", threshold=1000)
wbt.watershed("flow_dir.tif", "pour_points.shp", "watersheds.tif")

# --- Terrain analysis ---
wbt.slope("dem.tif", "slope.tif", units="degrees")
wbt.aspect("dem.tif", "aspect.tif")
wbt.ruggedness_index("dem.tif", "tri.tif")
wbt.wetness_index("dem.tif", "twi.tif")
wbt.relative_topographic_position("dem.tif", "rtp.tif", filter_size=11)

# --- LiDAR processing ---
wbt.lidar_info("point_cloud.las")
wbt.lidar_ground_point_filter("points.las", "ground.las", radius=2.0)
wbt.lidar_tin_gridding("ground.las", "dsm.tif", resolution=1.0)
```

### richdem

High-performance terrain analysis.

```python
import richdem as rd
import numpy as np

dem = rd.LoadGDAL("dem.tif")

# Terrain attributes
slope = rd.TerrainAttribute(dem, attrib="slope_riserun")
aspect = rd.TerrainAttribute(dem, attrib="aspect")
curvature = rd.TerrainAttribute(dem, attrib="curvature")
planc = rd.TerrainAttribute(dem, attrib="planform_curvature")
profc = rd.TerrainAttribute(dem, attrib="profile_curvature")

# Depression filling
filled = rd.FillDepressions(dem, epsilon=True)

# Flow accumulation
accum = rd.FlowAccumulation(dem, method="D8")

rd.SaveGDAL("slope.tif", slope)
```

### GDAL Python Bindings

Low-level access to the full GDAL/OGR API. Use when rasterio/fiona cannot handle a specific operation.

```python
from osgeo import gdal, ogr, osr
import numpy as np

gdal.UseExceptions()  # Enable exceptions instead of error codes

# --- Read raster ---
ds = gdal.Open("dem.tif")
band = ds.GetRasterBand(1)
data = band.ReadAsArray()
nodata = band.GetNoDataValue()
transform = ds.GetGeoTransform()
projection = ds.GetProjection()
ds = None  # Close

# --- Create raster ---
driver = gdal.GetDriverByName("GTiff")
out_ds = driver.Create(
    "output.tif", xsize=1000, ysize=1000, bands=1, eType=gdal.GDT_Float32,
    options=["COMPRESS=DEFLATE", "TILED=YES"],
)
out_ds.SetGeoTransform(transform)
out_ds.SetProjection(projection)
out_ds.GetRasterBand(1).WriteArray(data)
out_ds.GetRasterBand(1).SetNoDataValue(-9999)
out_ds.FlushCache()
out_ds = None

# --- Virtual raster (VRT) ---
vrt_options = gdal.BuildVRTOptions(resampleAlg="bilinear", addAlpha=False)
vrt = gdal.BuildVRT("mosaic.vrt", ["tile1.tif", "tile2.tif", "tile3.tif"], options=vrt_options)
vrt = None

# --- Translate (format conversion) ---
gdal.Translate("output.tif", "input.nc", format="GTiff", creationOptions=["COMPRESS=LZW"])

# --- Warp (reproject and resample) ---
gdal.Warp(
    "reprojected.tif", "input.tif",
    dstSRS="EPSG:4326",
    xRes=0.001, yRes=0.001,
    resampleAlg="bilinear",
    creationOptions=["COMPRESS=DEFLATE"],
)

# --- Polygonize raster to vector ---
src_ds = gdal.Open("classified.tif")
src_band = src_ds.GetRasterBand(1)

drv = ogr.GetDriverByName("GPKG")
dst_ds = drv.CreateDataSource("polygons.gpkg")
srs = osr.SpatialReference()
srs.ImportFromWkt(src_ds.GetProjection())
dst_layer = dst_ds.CreateLayer("polygons", srs, ogr.wkbPolygon)
dst_layer.CreateField(ogr.FieldDefn("class_id", ogr.OFTInteger))

gdal.Polygonize(src_band, None, dst_layer, 0, [])
dst_ds = None
src_ds = None

# --- OGR: read and query vector data ---
ds = ogr.Open("data.gpkg")
layer = ds.GetLayer("buildings")
layer.SetSpatialFilterRect(13.3, 52.4, 13.5, 52.6)
layer.SetAttributeFilter("height > 20")

for feature in layer:
    geom = feature.GetGeometryRef()
    name = feature.GetField("name")
    print(f"{name}: {geom.GetArea():.2f} m2")

ds = None
```

---

## 7. Vector Processing Recipes

Comprehensive recipes for vector data manipulation, from spatial joins to geocoding.

### Spatial Joins

```python
import geopandas as gpd
import numpy as np

points = gpd.read_file("sensors.gpkg")
polygons = gpd.read_file("districts.gpkg")

# --- Point-in-polygon ---
joined = gpd.sjoin(points, polygons, how="inner", predicate="within")
# Result: each point gets the attributes of the polygon it falls within

# Left join (keep all points, even those not in any polygon)
joined = gpd.sjoin(points, polygons, how="left", predicate="within")

# --- Nearest join (distance-based) ---
nearest = gpd.sjoin_nearest(
    points, polygons,
    how="left",
    max_distance=5000,       # Maximum search distance (CRS units)
    distance_col="dist_m",   # Add distance column
)

# --- Within distance (all pairs within threshold) ---
from shapely import STRtree, buffer

# Buffer approach
buffered_points = points.copy()
buffered_points["geometry"] = points.geometry.buffer(1000)
within_dist = gpd.sjoin(buffered_points, polygons, how="inner", predicate="intersects")

# STRtree approach (more memory efficient)
tree = STRtree(polygons.geometry.values)
idx = tree.query(points.geometry.values, predicate="dwithin", distance=1000)
# idx is (points_idx, polygons_idx) pairs

# --- Line intersection ---
roads = gpd.read_file("roads.gpkg")
zones = gpd.read_file("zones.gpkg")
road_zone = gpd.sjoin(roads, zones, how="inner", predicate="intersects")

# --- Spatial join with aggregation ---
# Count points per polygon
joined = gpd.sjoin(points, polygons, how="right", predicate="contains")
counts = joined.groupby(joined.index).agg(
    point_count=("index_left", "count"),
    avg_value=("value", "mean"),
    max_value=("value", "max"),
).reset_index()
polygons = polygons.join(counts)
```

### Overlay Operations

```python
import geopandas as gpd

parcels = gpd.read_file("parcels.gpkg")
flood_zone = gpd.read_file("flood_zone.gpkg")

# --- Intersection (clip features to overlay boundary) ---
intersected = gpd.overlay(parcels, flood_zone, how="intersection")
# Each parcel is split where it overlaps flood zones
intersected["flood_area"] = intersected.geometry.area

# --- Union (combine both layers, split at boundaries) ---
unioned = gpd.overlay(parcels, flood_zone, how="union")

# --- Difference (subtract overlay from input) ---
outside_flood = gpd.overlay(parcels, flood_zone, how="difference")

# --- Symmetric difference (non-overlapping areas from both) ---
sym_diff = gpd.overlay(parcels, flood_zone, how="symmetric_difference")

# --- Identity (like intersection but keeps all of input) ---
identity = gpd.overlay(parcels, flood_zone, how="identity")

# --- Percentage of each parcel in flood zone ---
parcels["total_area"] = parcels.geometry.area
intersected = gpd.overlay(parcels, flood_zone, how="intersection")
intersected["flood_area"] = intersected.geometry.area
pct = intersected.groupby("parcel_id")["flood_area"].sum().reset_index()
parcels = parcels.merge(pct, on="parcel_id", how="left")
parcels["flood_pct"] = (parcels["flood_area"] / parcels["total_area"] * 100).fillna(0)
```

### Dissolve and Aggregation

```python
import geopandas as gpd
import numpy as np

gdf = gpd.read_file("census_blocks.gpkg")

# --- Basic dissolve ---
# Merge all geometries by district, sum numeric columns
dissolved = gdf.dissolve(by="district", aggfunc="sum")

# --- Multi-function aggregation ---
dissolved = gdf.dissolve(
    by="district",
    aggfunc={
        "population": "sum",
        "income": "mean",
        "area_km2": "sum",
        "name": "first",
        "density": lambda x: x.sum() / x.count(),
    },
)

# --- Dissolve all (single geometry) ---
boundary = gdf.dissolve()  # All features merged into one

# --- Dissolve with buffer to merge nearby features ---
gdf["geometry"] = gdf.geometry.buffer(50)
merged = gdf.dissolve()
# Optionally re-buffer inward
merged["geometry"] = merged.geometry.buffer(-50)

# --- Group-by without dissolve (keep original geometries) ---
summary = gdf.groupby("district").agg(
    total_pop=("population", "sum"),
    avg_income=("income", "mean"),
    block_count=("population", "count"),
).reset_index()
```

### Geocoding

```python
from geopy.geocoders import Nominatim, GoogleV3, Photon
from geopy.extra.rate_limiter import RateLimiter
import geopandas as gpd
import pandas as pd

# --- Forward geocoding (address -> coordinates) ---
geolocator = Nominatim(user_agent="my_gis_app", timeout=10)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

addresses = pd.DataFrame({
    "address": [
        "Brandenburger Tor, Berlin",
        "Eiffel Tower, Paris",
        "Colosseum, Rome",
    ]
})

addresses["location"] = addresses["address"].apply(geocode)
addresses["latitude"] = addresses["location"].apply(lambda x: x.latitude if x else None)
addresses["longitude"] = addresses["location"].apply(lambda x: x.longitude if x else None)

gdf = gpd.GeoDataFrame(
    addresses,
    geometry=gpd.points_from_xy(addresses.longitude, addresses.latitude),
    crs="EPSG:4326",
)

# --- Reverse geocoding (coordinates -> address) ---
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

location = reverse((52.5163, 13.3777))
print(location.address)
print(location.raw)  # Full response with structured fields

# --- Batch geocoding with error handling ---
def safe_geocode(address):
    try:
        location = geocode(address)
        if location:
            return pd.Series({"lat": location.latitude, "lon": location.longitude, "found": True})
    except Exception as e:
        pass
    return pd.Series({"lat": None, "lon": None, "found": False})

results = addresses["address"].apply(safe_geocode)
addresses = pd.concat([addresses, results], axis=1)
print(f"Geocoded: {addresses['found'].sum()} / {len(addresses)}")

# --- Using Google Geocoder (requires API key) ---
google = GoogleV3(api_key="YOUR_API_KEY")
location = google.geocode("1600 Amphitheatre Parkway, Mountain View, CA")
```

### Coordinate Transformations

```python
import geopandas as gpd
from pyproj import Transformer, CRS
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info
import numpy as np

gdf = gpd.read_file("data.geojson")  # WGS84

# --- Reproject entire GeoDataFrame ---
gdf_utm = gdf.to_crs(epsg=32633)
gdf_mercator = gdf.to_crs(epsg=3857)
gdf_equal_area = gdf.to_crs("+proj=aea +lat_1=29.5 +lat_2=45.5 +lon_0=-96")

# --- Auto-detect UTM zone ---
centroid = gdf.geometry.unary_union.centroid
utm_info = query_utm_crs_info(
    datum_name="WGS 84",
    area_of_interest=AreaOfInterest(
        west_lon_degree=centroid.x - 1,
        south_lat_degree=centroid.y - 1,
        east_lon_degree=centroid.x + 1,
        north_lat_degree=centroid.y + 1,
    ),
)
utm_crs = CRS.from_authority(utm_info[0].auth_name, utm_info[0].code)
gdf_utm = gdf.to_crs(utm_crs)

# --- Transform individual coordinates ---
transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)

# Single point
x, y = transformer.transform(13.4, 52.5)

# Array of points
lons = np.array([13.4, 13.5, 13.6])
lats = np.array([52.5, 52.6, 52.7])
xs, ys = transformer.transform(lons, lats)

# --- Web Mercator pixel coordinates ---
def lonlat_to_tile(lon, lat, zoom):
    """Convert lon/lat to tile x,y at given zoom level."""
    n = 2 ** zoom
    x = int((lon + 180) / 360 * n)
    y = int((1 - np.log(np.tan(np.radians(lat)) + 1/np.cos(np.radians(lat))) / np.pi) / 2 * n)
    return x, y

# --- Ensure correct area/distance calculations ---
# WRONG: calculating area in WGS84 (degrees, not meters)
# area_wrong = gdf.geometry.area  # Returns degrees squared!

# CORRECT: project to equal-area CRS first
gdf_ea = gdf.to_crs("+proj=aea +lat_1=29.5 +lat_2=45.5")
gdf["area_m2"] = gdf_ea.geometry.area

# Or use PyProj geodesic for accurate area on the ellipsoid
from pyproj import Geod
geod = Geod(ellps="WGS84")
gdf["area_m2_geodesic"] = gdf.geometry.apply(
    lambda g: abs(geod.geometry_area_perimeter(g)[0])
)
```

### Format Conversion Patterns

```python
import geopandas as gpd
import json

gdf = gpd.read_file("input.shp")

# --- Common format conversions ---
gdf.to_file("output.gpkg", driver="GPKG")
gdf.to_file("output.geojson", driver="GeoJSON")
gdf.to_file("output.fgb", driver="FlatGeobuf")
gdf.to_parquet("output.parquet")

# --- GeoJSON string (for APIs) ---
geojson_str = gdf.to_json()
geojson_dict = json.loads(geojson_str)

# --- To/from WKT ---
gdf["wkt"] = gdf.geometry.to_wkt()
from shapely import from_wkt
gdf["geometry"] = gdf["wkt"].apply(from_wkt)

# --- To/from WKB ---
gdf["wkb"] = gdf.geometry.to_wkb()
from shapely import from_wkb
gdf["geometry"] = gdf["wkb"].apply(from_wkb)

# --- CSV with geometry ---
# Write
gdf["lon"] = gdf.geometry.x
gdf["lat"] = gdf.geometry.y
gdf.drop(columns="geometry").to_csv("output.csv", index=False)

# Read
import pandas as pd
df = pd.read_csv("input.csv")
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

# --- KML ---
gdf.to_file("output.kml", driver="KML")

# --- Shapefile to GeoParquet batch conversion ---
from pathlib import Path

shp_dir = Path("shapefiles/")
for shp in shp_dir.glob("*.shp"):
    gdf = gpd.read_file(shp)
    gdf.to_parquet(shp.with_suffix(".parquet"))
    print(f"Converted {shp.name}")
```

---

## 8. Environment Setup

### conda / mamba

Conda handles the notoriously difficult C/C++ geospatial dependencies (GDAL, GEOS, PROJ, HDF5, NetCDF). Mamba is a drop-in replacement that resolves dependencies 10-50x faster.

```bash
# Install miniforge (includes mamba by default)
# https://github.com/conda-forge/miniforge
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
bash Miniforge3-$(uname)-$(uname -m).sh

# Create a comprehensive geospatial environment
mamba create -n geo python=3.12 \
  # Core I/O
  geopandas=1.0 shapely=2.0 rasterio fiona pyogrio pyproj gdal \
  # Formats
  h5py netcdf4 zarr \
  # Visualization
  matplotlib cartopy folium plotly pydeck contextily datashader lonboard \
  holoviews geoviews hvplot panel bokeh \
  # Analysis
  pysal scikit-learn scipy xgboost lightgbm \
  osmnx networkx movingpandas \
  # Raster
  xarray rioxarray xarray-spatial rasterstats richdem \
  # Cloud
  dask dask-geopandas distributed \
  stackstac pystac-client planetary-computer \
  # Database
  sqlalchemy geoalchemy2 psycopg2 duckdb \
  # Jupyter
  jupyterlab ipywidgets ipyleaflet \
  # Utils
  tqdm requests aiohttp pyarrow

mamba activate geo

# Verify installation
python -c "
import geopandas; print(f'GeoPandas {geopandas.__version__}')
import shapely; print(f'Shapely {shapely.__version__}')
import rasterio; print(f'Rasterio {rasterio.__version__}')
from osgeo import gdal; print(f'GDAL {gdal.__version__}')
"

# Export environment for reproducibility
mamba env export --no-builds > environment.yml

# Recreate on another machine
mamba env create -f environment.yml
```

### pip + System GDAL

When conda is not an option (CI, Docker, corporate environments).

```bash
# --- Ubuntu / Debian ---
sudo apt-get update && sudo apt-get install -y \
  gdal-bin libgdal-dev \
  libgeos-dev libproj-dev \
  libspatialindex-dev \
  python3-dev python3-venv

export GDAL_VERSION=$(gdal-config --version)
python3 -m venv .venv && source .venv/bin/activate

pip install "GDAL==${GDAL_VERSION}"
pip install geopandas rasterio fiona shapely pyproj pyogrio
pip install folium plotly pydeck contextily lonboard
pip install pysal scikit-learn scipy osmnx movingpandas
pip install xarray rioxarray rasterstats duckdb

# --- macOS (Homebrew) ---
brew install gdal geos proj spatialindex
export GDAL_VERSION=$(gdal-config --version)

python3 -m venv .venv && source .venv/bin/activate
pip install "GDAL==${GDAL_VERSION}"
pip install geopandas rasterio fiona shapely pyproj

# --- Windows ---
# Recommended: Use conda/mamba. If pip is required:
# 1. Install OSGeo4W (https://trac.osgeo.org/osgeo4w/)
# 2. Use the OSGeo4W shell
# 3. pip install from wheels: https://www.lfd.uci.edu/~gohlke/pythonlibs/
# Or use pipwin:
pip install pipwin
pipwin install gdal fiona rasterio shapely
pip install geopandas pyproj
```

### Docker

```dockerfile
# --- Minimal GDAL image ---
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.0
RUN apt-get update && apt-get install -y python3-pip python3-venv && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    geopandas rasterio fiona shapely pyproj pyogrio \
    folium plotly pydeck lonboard duckdb \
    pysal scikit-learn xarray rioxarray rasterstats
WORKDIR /data

# --- JupyterLab geospatial image ---
FROM quay.io/jupyter/scipy-notebook:python-3.12
USER root
RUN apt-get update && apt-get install -y libgdal-dev gdal-bin && rm -rf /var/lib/apt/lists/*
USER ${NB_UID}
RUN pip install --no-cache-dir \
    geopandas rasterio fiona shapely pyproj pyogrio \
    folium contextily lonboard pydeck plotly \
    pysal osmnx movingpandas \
    xarray rioxarray stackstac dask-geopandas duckdb \
    geoalchemy2 sqlalchemy psycopg2-binary \
    jupyterlab-git
```

### Docker Compose with JupyterLab + PostGIS

```yaml
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
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gis -d gisdb"]
      interval: 5s
      timeout: 5s
      retries: 5

  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/home/jovyan/data
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      DATABASE_URL: "postgresql://gis:gispass@postgis:5432/gisdb"
    depends_on:
      postgis:
        condition: service_healthy

volumes:
  pgdata:
```

### Dev Container

For VS Code dev containers or GitHub Codespaces.

```json
// .devcontainer/devcontainer.json
{
  "name": "GeoSpatial Python",
  "image": "ghcr.io/osgeo/gdal:ubuntu-small-3.9.0",
  "features": {
    "ghcr.io/devcontainers/features/python:1": {
      "version": "3.12",
      "installTools": true
    }
  },
  "postCreateCommand": "pip install geopandas rasterio fiona shapely pyproj pyogrio folium plotly lonboard pysal duckdb xarray rioxarray jupyterlab",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-toolsai.jupyter",
        "ms-python.vscode-pylance"
      ]
    }
  },
  "forwardPorts": [8888]
}
```

### uv (Fast Python Package Manager)

uv is the fastest Python package installer (10-100x faster than pip). Works well for pure-Python geo packages but may struggle with C extensions.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project with uv
uv init geo-project && cd geo-project
uv venv --python 3.12

# Install packages (blazing fast)
uv pip install geopandas shapely pyproj duckdb folium plotly

# For packages with C deps (GDAL, rasterio), still need system libs
sudo apt-get install -y libgdal-dev
uv pip install rasterio fiona

# Sync from requirements file
uv pip install -r requirements.txt

# Lock dependencies
uv pip compile requirements.in -o requirements.txt
```

### pixi (conda-forge Package Manager)

pixi is a fast, modern package manager that uses conda-forge packages. It handles the C/C++ dependency problem elegantly.

```bash
# Install pixi
curl -fsSL https://pixi.sh/install.sh | bash

# Initialize project
pixi init geo-project && cd geo-project

# Add dependencies (resolves C++ deps automatically)
pixi add python=3.12 geopandas rasterio fiona shapely pyproj
pixi add folium plotly lonboard duckdb
pixi add pysal scikit-learn xarray rioxarray
pixi add jupyterlab

# Run commands in the environment
pixi run python script.py
pixi run jupyter lab

# pixi.toml is the project config (like package.json for geo Python)
```

---

## 9. Performance Optimization

### Vectorized Shapely 2.0 Operations

The single biggest performance win: replace Python loops with Shapely 2.0 vectorized operations.

```python
import numpy as np
from shapely import buffer, distance, contains, intersects, area, STRtree
import geopandas as gpd
import time

gdf = gpd.read_parquet("buildings.parquet")  # 500k features

# --- WRONG: Python loop (slow) ---
# areas = [geom.area for geom in gdf.geometry]  # ~8 seconds

# --- RIGHT: Vectorized (fast) ---
# areas = gdf.geometry.area  # ~0.05 seconds (160x faster)

# --- WRONG: Loop with apply ---
# gdf["buffered"] = gdf.geometry.apply(lambda g: g.buffer(100))  # ~12 seconds

# --- RIGHT: Vectorized buffer ---
# gdf["buffered"] = gdf.geometry.buffer(100)  # ~0.3 seconds (40x faster)

# --- Benchmark common operations ---
geoms = gdf.geometry.values  # numpy array of Shapely geometries
n = len(geoms)

t0 = time.perf_counter()
areas = area(geoms)
t1 = time.perf_counter()
print(f"Area ({n} features): {t1-t0:.3f}s")

t0 = time.perf_counter()
buffered = buffer(geoms, 100)
t1 = time.perf_counter()
print(f"Buffer ({n} features): {t1-t0:.3f}s")

t0 = time.perf_counter()
dists = distance(geoms[:n//2], geoms[n//2:n//2 + n//2])
t1 = time.perf_counter()
print(f"Distance ({n//2} pairs): {t1-t0:.3f}s")
```

### Spatial Indexing

Spatial indexes are essential for any operation that involves searching or filtering by location.

```python
from shapely import STRtree
import geopandas as gpd
import numpy as np

buildings = gpd.read_parquet("buildings.parquet")  # 2M features
points = gpd.read_parquet("sensors.parquet")       # 50k features

# --- STRtree (Shapely 2.0) ---
tree = STRtree(buildings.geometry.values)

# Query: which buildings intersect each point's buffer?
buffers = points.geometry.buffer(500)
left_idx, right_idx = tree.query(buffers.values, predicate="intersects")
# left_idx: index into buffers, right_idx: index into buildings

# Nearest neighbor for each point
nearest_idx = tree.nearest(points.geometry.values)
nearest_building = buildings.iloc[nearest_idx]

# --- GeoPandas sindex (R-tree) ---
# Automatically used in sjoin, but can be accessed directly
sindex = buildings.sindex

# Query by bounds
possible_matches_idx = list(sindex.intersection((13.3, 52.4, 13.5, 52.6)))
possible_matches = buildings.iloc[possible_matches_idx]

# --- Performance comparison ---
# Without spatial index: O(n * m) -- for 2M x 50k = 100 billion comparisons
# With spatial index: O(n * log(m)) -- for 50k * ~21 = ~1 million comparisons
# Speedup: ~100,000x for this example
```

### Chunked Processing

For datasets too large to fit in memory.

```python
import geopandas as gpd
import pyogrio
import pandas as pd
from pathlib import Path

# --- Read in chunks with pyogrio ---
total_features = pyogrio.read_info("huge_dataset.gpkg")["features"]
chunk_size = 100_000
results = []

for offset in range(0, total_features, chunk_size):
    chunk = pyogrio.read_dataframe(
        "huge_dataset.gpkg",
        skip_features=offset,
        max_features=chunk_size,
    )
    # Process chunk
    chunk["area"] = chunk.geometry.area
    summary = chunk.groupby("type")["area"].sum()
    results.append(summary)
    del chunk  # Free memory

final = pd.concat(results).groupby(level=0).sum()

# --- Write in chunks ---
for i, chunk_gdf in enumerate(np.array_split(huge_gdf, 20)):
    mode = "w" if i == 0 else "a"
    chunk_gdf.to_file("output.gpkg", driver="GPKG", mode=mode)

# --- Raster chunk processing ---
import rasterio
import numpy as np

with rasterio.open("huge_raster.tif") as src:
    profile = src.profile.copy()

    with rasterio.open("output.tif", "w", **profile) as dst:
        for ji, window in src.block_windows(1):
            data = src.read(window=window)
            # Process
            result = np.where(data > 0, np.log1p(data), data)
            dst.write(result, window=window)
```

### Parallel Processing with Dask

```python
import dask_geopandas
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import geopandas as gpd

# --- Start local Dask cluster ---
cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit="4GB")
client = Client(cluster)
print(client.dashboard_link)  # Monitor at http://localhost:8787

# --- Parallel vector processing ---
gdf = gpd.read_parquet("buildings.parquet")
ddf = dask_geopandas.from_geopandas(gdf, npartitions=16)

# All operations are lazy
ddf["area"] = ddf.geometry.area
ddf["centroid_x"] = ddf.geometry.centroid.x
large = ddf[ddf["area"] > 500]

# Compute triggers parallel execution
result = large.compute()

# --- Spatial partitioning (required for spatial joins) ---
ddf = ddf.spatial_shuffle()  # Hilbert-curve spatial partitioning
result = dask_geopandas.sjoin(ddf, zones_ddf).compute()

# --- Parallel raster processing with Dask arrays ---
import dask.array as da
import xarray as xr
import rioxarray

# Open raster as lazy Dask array
ds = xr.open_dataset("large_raster.tif", engine="rasterio", chunks={"x": 512, "y": 512})
data = ds["band_data"]

# Compute (parallelized across chunks)
mean_val = data.mean().compute()
result = (data - data.mean()) / data.std()
result.rio.to_raster("normalized.tif")

# --- Close cluster ---
client.close()
cluster.close()
```

### Memory Profiling

```python
import tracemalloc
import geopandas as gpd
import sys

# --- Basic memory measurement ---
gdf = gpd.read_parquet("data.parquet")
print(f"Memory usage: {gdf.memory_usage(deep=True).sum() / 1e6:.1f} MB")

# Per-column breakdown
for col in gdf.columns:
    size_mb = gdf[col].memory_usage(deep=True) / 1e6
    print(f"  {col}: {size_mb:.1f} MB")

# --- tracemalloc for detailed profiling ---
tracemalloc.start()

gdf = gpd.read_file("large.gpkg")
gdf["area"] = gdf.geometry.area
result = gdf.dissolve(by="category")

snapshot = tracemalloc.take_snapshot()
stats = snapshot.statistics("lineno")
for stat in stats[:10]:
    print(stat)

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.1f} MB, Peak: {peak / 1e6:.1f} MB")
tracemalloc.stop()

# --- Reduce memory: downcast dtypes ---
gdf["population"] = gdf["population"].astype("int32")  # From int64
gdf["area_km2"] = gdf["area_km2"].astype("float32")    # From float64
gdf["type"] = gdf["type"].astype("category")            # String -> category

# --- Simplify geometries to reduce memory ---
gdf["geometry"] = gdf.geometry.simplify(tolerance=1.0)
```

### Benchmark Cheatsheet

| Task | Naive approach | Optimized approach | Speedup |
|---|---|---|---|
| Buffer 1M points | Python loop | Shapely 2.0 vectorized | 40-60x |
| Point-in-polygon 1M x 10k | Nested loop | STRtree + predicate | 1000x+ |
| Read 1M features (Shapefile) | `gpd.read_file()` Fiona | `gpd.read_file()` pyogrio | 3-5x |
| Read 1M features (GeoParquet) | -- | `gpd.read_parquet()` | 8-15x vs Shapefile |
| Spatial join 1M x 100k | gpd.sjoin (naive) | Dask-GeoPandas (16 cores) | 10-14x |
| Zonal stats 10k zones | rasterstats (serial) | rasterstats + multiprocessing | 4-8x |
| Distance matrix 10k x 10k | scipy cdist | cKDTree.sparse_distance_matrix | 5-10x |
| Format: SHP -> GeoParquet | gpd.read + write | pyogrio.read_arrow + write | 6-10x |

---

## 10. Testing & CI

### pytest Patterns for Geospatial Code

```python
# tests/conftest.py
import pytest
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString, box
import numpy as np

@pytest.fixture
def sample_points():
    """Create a GeoDataFrame of sample points."""
    return gpd.GeoDataFrame(
        {"name": ["A", "B", "C"], "value": [10, 20, 30]},
        geometry=[Point(0, 0), Point(1, 1), Point(2, 2)],
        crs="EPSG:4326",
    )

@pytest.fixture
def sample_polygons():
    """Create a GeoDataFrame of sample polygons."""
    return gpd.GeoDataFrame(
        {"zone": ["X", "Y"]},
        geometry=[
            box(0, 0, 1.5, 1.5),
            box(1.5, 1.5, 3, 3),
        ],
        crs="EPSG:4326",
    )

@pytest.fixture
def sample_raster(tmp_path):
    """Create a temporary test raster."""
    import rasterio
    from rasterio.transform import from_bounds

    path = tmp_path / "test.tif"
    data = np.random.rand(100, 100).astype(np.float32)
    transform = from_bounds(0, 0, 1, 1, 100, 100)

    with rasterio.open(
        path, "w", driver="GTiff",
        height=100, width=100, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    return path

# tests/test_spatial_ops.py
import geopandas as gpd
from geopandas.testing import assert_geodataframe_equal
from shapely.geometry import Point
import numpy as np
import pytest

def test_spatial_join(sample_points, sample_polygons):
    result = gpd.sjoin(sample_points, sample_polygons, predicate="within")
    assert len(result) == 2  # Points A and B in zone X, C in zone Y
    assert "zone" in result.columns

def test_buffer(sample_points):
    buffered = sample_points.to_crs(epsg=32633).geometry.buffer(1000)
    assert all(buffered.area > 0)
    assert all(buffered.geom_type == "Polygon")

def test_crs_preservation(sample_points):
    reprojected = sample_points.to_crs(epsg=32633)
    assert reprojected.crs.to_epsg() == 32633
    back = reprojected.to_crs(epsg=4326)
    assert back.crs.to_epsg() == 4326
    # Check coordinates are approximately preserved
    np.testing.assert_allclose(
        back.geometry.x.values, sample_points.geometry.x.values, atol=1e-6
    )

def test_geodataframe_equality(sample_points):
    copy = sample_points.copy()
    assert_geodataframe_equal(sample_points, copy)

def test_geometry_validity(sample_polygons):
    assert all(sample_polygons.geometry.is_valid)
    assert all(~sample_polygons.geometry.is_empty)

def test_zonal_stats(sample_polygons, sample_raster):
    from rasterstats import zonal_stats
    stats = zonal_stats(sample_polygons, str(sample_raster), stats=["mean", "count"])
    assert len(stats) == len(sample_polygons)
    for s in stats:
        assert "mean" in s
        assert s["count"] > 0

@pytest.mark.parametrize("crs", [4326, 32633, 3857])
def test_multiple_crs(sample_points, crs):
    result = sample_points.to_crs(epsg=crs)
    assert result.crs.to_epsg() == crs
```

### CI with GDAL

```yaml
# .github/workflows/geo-tests.yml
name: Geospatial Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Install GDAL system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install Python dependencies
        run: |
          pip install --upgrade pip
          pip install "GDAL==$(gdal-config --version)"
          pip install geopandas rasterio fiona shapely pyproj
          pip install pytest pytest-cov rasterstats

      - name: Run tests
        run: pytest tests/ -v --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: coverage.xml

  # Alternative: conda-based CI
  test-conda:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: conda-incubator/setup-miniconda@v3
        with:
          miniforge-variant: Mambaforge
          activate-environment: test
          environment-file: environment.yml
      - name: Run tests
        shell: bash -l {0}
        run: pytest tests/ -v
```

### Property-Based Testing

```python
from hypothesis import given, settings
from hypothesis.strategies import floats, integers, tuples
from shapely.geometry import Point, Polygon, box
from shapely import is_valid, area, buffer
import numpy as np

# Test that buffering always produces valid geometry
@given(
    x=floats(min_value=-180, max_value=180),
    y=floats(min_value=-90, max_value=90),
    radius=floats(min_value=0.001, max_value=10),
)
@settings(max_examples=200)
def test_buffer_always_valid(x, y, radius):
    pt = Point(x, y)
    buffered = pt.buffer(radius)
    assert buffered.is_valid
    assert buffered.area > 0
    assert buffered.contains(pt)

# Test that area is always non-negative
@given(
    x=floats(min_value=0, max_value=100),
    y=floats(min_value=0, max_value=100),
    w=floats(min_value=0.1, max_value=50),
    h=floats(min_value=0.1, max_value=50),
)
def test_box_area_positive(x, y, w, h):
    b = box(x, y, x + w, y + h)
    assert b.area > 0
    assert np.isclose(b.area, w * h)

# Test spatial join invariants
@given(n_points=integers(min_value=1, max_value=100))
def test_sjoin_inner_subset(n_points):
    import geopandas as gpd
    pts = gpd.GeoDataFrame(
        geometry=[Point(np.random.rand(), np.random.rand()) for _ in range(n_points)],
        crs="EPSG:4326",
    )
    poly = gpd.GeoDataFrame(geometry=[box(0, 0, 0.5, 0.5)], crs="EPSG:4326")
    result = gpd.sjoin(pts, poly, predicate="within")
    assert len(result) <= len(pts)
```

---

## 11. Integration Patterns

### PostGIS <-> GeoPandas

```python
from sqlalchemy import create_engine, text
import geopandas as gpd

engine = create_engine("postgresql://user:pass@localhost:5432/gisdb")

# --- Read from PostGIS ---
gdf = gpd.read_postgis(
    "SELECT * FROM buildings WHERE height > 20",
    engine,
    geom_col="geom",
    crs="EPSG:4326",
)

# Read with parameterized query
gdf = gpd.read_postgis(
    text("SELECT * FROM buildings WHERE type = :btype"),
    engine,
    geom_col="geom",
    params={"btype": "residential"},
)

# --- Write to PostGIS ---
gdf.to_postgis("buildings_processed", engine, if_exists="replace", index=False)

# Append to existing table
gdf.to_postgis("buildings_processed", engine, if_exists="append", index=False)

# --- Execute raw SQL ---
with engine.connect() as conn:
    conn.execute(text("CREATE INDEX idx_buildings_geom ON buildings_processed USING GIST (geometry)"))
    conn.commit()

# --- Bulk operations ---
# For large datasets, use COPY instead of INSERT
from io import StringIO
import csv

def bulk_insert_postgis(gdf, table_name, engine):
    """Fast bulk insert using COPY."""
    buf = StringIO()
    gdf_copy = gdf.copy()
    gdf_copy["geometry"] = gdf_copy["geometry"].to_wkt()
    gdf_copy.to_csv(buf, index=False, quoting=csv.QUOTE_NONNUMERIC)
    buf.seek(0)

    with engine.raw_connection() as conn:
        with conn.cursor() as cur:
            cur.copy_expert(
                f"COPY {table_name} FROM STDIN WITH CSV HEADER",
                buf,
            )
        conn.commit()
```

### DuckDB <-> GeoPandas

```python
import duckdb
import geopandas as gpd

con = duckdb.connect()
con.install_extension("spatial")
con.load_extension("spatial")

# --- GeoDataFrame -> DuckDB ---
gdf = gpd.read_parquet("buildings.parquet")
con.register("buildings", gdf)

result = con.sql("""
    SELECT type, COUNT(*) as cnt, AVG(height) as avg_height
    FROM buildings
    WHERE height > 10
    GROUP BY type
    ORDER BY cnt DESC
""").fetchdf()

# --- DuckDB -> GeoDataFrame ---
result_gdf = con.sql("""
    SELECT *, ST_Area(geometry) as area
    FROM read_parquet('buildings.parquet')
    WHERE ST_Within(geometry, ST_GeomFromText('POLYGON((...))'))
""").fetchdf()
result_gdf = gpd.GeoDataFrame(result_gdf, geometry="geometry")

# --- DuckDB as processing engine (faster than pandas for aggregations) ---
stats = con.sql("""
    SELECT
        district,
        COUNT(*) as building_count,
        SUM(ST_Area(geometry)) as total_area,
        AVG(height) as avg_height,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY height) as median_height
    FROM read_parquet('buildings.parquet')
    GROUP BY district
    ORDER BY building_count DESC
""").fetchdf()
```

### Jupyter Widgets

```python
import ipywidgets as widgets
from IPython.display import display
import geopandas as gpd
import folium

gdf = gpd.read_file("data.gpkg")
categories = gdf["type"].unique().tolist()

# --- Interactive filter widget ---
category_select = widgets.SelectMultiple(
    options=categories,
    value=categories[:3],
    description="Types:",
)
buffer_slider = widgets.IntSlider(
    value=100, min=0, max=1000, step=50, description="Buffer (m):"
)
output = widgets.Output()

def update_map(change):
    with output:
        output.clear_output(wait=True)
        filtered = gdf[gdf["type"].isin(category_select.value)]
        if buffer_slider.value > 0:
            filtered = filtered.copy()
            filtered["geometry"] = filtered.to_crs(epsg=32633).buffer(buffer_slider.value).to_crs(epsg=4326)
        m = filtered.explore(tiles="CartoDB positron")
        display(m)

category_select.observe(update_map, names="value")
buffer_slider.observe(update_map, names="value")

display(widgets.VBox([category_select, buffer_slider, output]))
update_map(None)
```

### Streamlit Geo App

```python
# app.py -- run with: streamlit run app.py
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px

st.set_page_config(page_title="GIS Explorer", layout="wide")
st.title("Spatial Data Explorer")

# --- File upload ---
uploaded = st.file_uploader("Upload a GeoJSON or GPKG file", type=["geojson", "gpkg"])
if uploaded:
    gdf = gpd.read_file(uploaded)
else:
    gdf = gpd.read_file("sample_data.gpkg")

# --- Sidebar filters ---
st.sidebar.header("Filters")
numeric_cols = gdf.select_dtypes(include="number").columns.tolist()
selected_col = st.sidebar.selectbox("Color by:", numeric_cols)
min_val, max_val = float(gdf[selected_col].min()), float(gdf[selected_col].max())
range_filter = st.sidebar.slider("Value range:", min_val, max_val, (min_val, max_val))

filtered = gdf[(gdf[selected_col] >= range_filter[0]) & (gdf[selected_col] <= range_filter[1])]
st.sidebar.metric("Features shown", len(filtered), delta=len(filtered) - len(gdf))

# --- Map ---
col1, col2 = st.columns([2, 1])

with col1:
    m = filtered.explore(
        column=selected_col,
        cmap="YlOrRd",
        tiles="CartoDB positron",
        legend=True,
    )
    st_folium(m, width=700, height=500)

with col2:
    fig = px.histogram(filtered, x=selected_col, nbins=30, title=f"Distribution of {selected_col}")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(filtered.drop(columns="geometry").describe())

# --- Download ---
st.download_button(
    "Download filtered data (GeoJSON)",
    filtered.to_json(),
    file_name="filtered.geojson",
    mime="application/json",
)
```

### Dash Geo App

```python
# app.py -- run with: python app.py
import dash
from dash import html, dcc, Input, Output
import dash_leaflet as dl
import geopandas as gpd
import plotly.express as px
import json

app = dash.Dash(__name__)
gdf = gpd.read_file("districts.gpkg")

app.layout = html.Div([
    html.H1("District Explorer"),
    html.Div([
        html.Div([
            dcc.Dropdown(
                id="color-column",
                options=[{"label": c, "value": c} for c in gdf.select_dtypes("number").columns],
                value="population",
            ),
            dl.Map(
                id="map",
                children=[dl.TileLayer(), dl.GeoJSON(id="geojson")],
                center=[52.5, 13.4], zoom=10,
                style={"height": "500px"},
            ),
        ], style={"width": "60%", "display": "inline-block"}),
        html.Div([
            dcc.Graph(id="histogram"),
            dcc.Graph(id="scatter"),
        ], style={"width": "38%", "display": "inline-block", "verticalAlign": "top"}),
    ]),
])

@app.callback(
    Output("geojson", "data"),
    Output("histogram", "figure"),
    Input("color-column", "value"),
)
def update(col):
    geojson = json.loads(gdf.to_json())
    hist = px.histogram(gdf, x=col, title=f"{col} Distribution")
    return geojson, hist

if __name__ == "__main__":
    app.run(debug=True)
```

---

## 12. Cookbook

A collection of self-contained recipes for real-world geospatial tasks.

### Recipe 1: Batch Format Conversion

```python
"""Convert all Shapefiles in a directory to GeoParquet."""
from pathlib import Path
import geopandas as gpd

input_dir = Path("shapefiles/")
output_dir = Path("parquet/")
output_dir.mkdir(exist_ok=True)

for shp in input_dir.glob("*.shp"):
    gdf = gpd.read_file(shp)
    out_path = output_dir / shp.with_suffix(".parquet").name
    gdf.to_parquet(out_path, compression="zstd")
    print(f"{shp.name} -> {out_path.name} ({len(gdf)} features)")
```

### Recipe 2: Download OSM Data for Any City

```python
"""Download buildings, roads, and POIs from OpenStreetMap."""
import osmnx as ox

place = "Munich, Germany"

buildings = ox.features_from_place(place, tags={"building": True})
roads = ox.graph_from_place(place, network_type="drive")
parks = ox.features_from_place(place, tags={"leisure": "park"})
restaurants = ox.features_from_place(place, tags={"amenity": "restaurant"})

buildings.to_parquet("munich_buildings.parquet")
nodes, edges = ox.graph_to_gdfs(roads)
edges.to_parquet("munich_roads.parquet")
parks.to_parquet("munich_parks.parquet")
restaurants.to_parquet("munich_restaurants.parquet")
```

### Recipe 3: Calculate Distance Matrix Between Points

```python
"""Compute pairwise distances between all locations."""
import geopandas as gpd
import numpy as np
from scipy.spatial.distance import cdist
from pyproj import Geod

gdf = gpd.read_file("locations.gpkg")

# Euclidean distance (for projected CRS)
gdf_proj = gdf.to_crs(epsg=32633)
coords = np.column_stack([gdf_proj.geometry.x, gdf_proj.geometry.y])
dist_matrix = cdist(coords, coords, metric="euclidean")

# Geodesic distance (accurate, for lat/lon)
geod = Geod(ellps="WGS84")
n = len(gdf)
dist_geo = np.zeros((n, n))
for i in range(n):
    for j in range(i+1, n):
        _, _, dist = geod.inv(
            gdf.geometry.x.iloc[i], gdf.geometry.y.iloc[i],
            gdf.geometry.x.iloc[j], gdf.geometry.y.iloc[j],
        )
        dist_geo[i, j] = dist_geo[j, i] = dist

# For large n, use vectorized approach
from itertools import combinations
pairs = list(combinations(range(n), 2))
lons1 = [gdf.geometry.x.iloc[i] for i, j in pairs]
lats1 = [gdf.geometry.y.iloc[i] for i, j in pairs]
lons2 = [gdf.geometry.x.iloc[j] for i, j in pairs]
lats2 = [gdf.geometry.y.iloc[j] for i, j in pairs]
_, _, dists = geod.inv(lons1, lats1, lons2, lats2)
```

### Recipe 4: Scrape and Map Real-Time Earthquake Data

```python
"""Fetch earthquake data from USGS API and create an interactive map."""
import requests
import geopandas as gpd
import pandas as pd
import folium

url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_week.geojson"
response = requests.get(url)
data = response.json()

features = []
for f in data["features"]:
    features.append({
        "magnitude": f["properties"]["mag"],
        "place": f["properties"]["place"],
        "time": pd.to_datetime(f["properties"]["time"], unit="ms"),
        "longitude": f["geometry"]["coordinates"][0],
        "latitude": f["geometry"]["coordinates"][1],
        "depth_km": f["geometry"]["coordinates"][2],
    })

df = pd.DataFrame(features)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
for _, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row.latitude, row.longitude],
        radius=row.magnitude * 3,
        color="red" if row.magnitude >= 5 else "orange",
        fill=True,
        popup=f"M{row.magnitude} - {row.place}<br>{row.time}",
    ).add_to(m)

m.save("earthquakes.html")
```

### Recipe 5: Sentinel-2 NDVI Time Series

```python
"""Compute monthly NDVI for an area of interest using STAC."""
from pystac_client import Client
import stackstac
import xarray as xr
import matplotlib.pyplot as plt

catalog = Client.open("https://earth-search.aws.element84.com/v1")

search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[11.5, 48.0, 11.7, 48.2],  # Munich area
    datetime="2024-01-01/2024-12-31",
    query={"eo:cloud_cover": {"lt": 20}},
)

items = search.item_collection()
stack = stackstac.stack(items, assets=["red", "nir"], resolution=20, epsg=32632)

red = stack.sel(band="red").astype(float)
nir = stack.sel(band="nir").astype(float)
ndvi = (nir - red) / (nir + red)

# Monthly median NDVI
monthly = ndvi.resample(time="1ME").median()
monthly_mean = monthly.mean(dim=["x", "y"]).compute()

# Plot time series
fig, ax = plt.subplots(figsize=(12, 4))
monthly_mean.plot(ax=ax, marker="o")
ax.set_ylabel("Mean NDVI")
ax.set_title("Monthly NDVI - Munich Area (2024)")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ndvi_timeseries.png", dpi=150)
```

### Recipe 6: Isochrone Map (Travel Time Polygons)

```python
"""Generate 5, 10, 15 minute walk isochrones from a point."""
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
import geopandas as gpd

G = ox.graph_from_point((52.52, 13.405), dist=3000, network_type="walk")
G = ox.add_edge_speeds(G)
G = ox.add_edge_travel_times(G)

center = ox.nearest_nodes(G, 13.405, 52.52)

isochrones = []
for minutes in [5, 10, 15]:
    seconds = minutes * 60
    subgraph = nx.ego_graph(G, center, radius=seconds, distance="travel_time")
    nodes = ox.graph_to_gdfs(subgraph, edges=False)
    hull = nodes.geometry.unary_union.convex_hull
    isochrones.append({"minutes": minutes, "geometry": hull})

iso_gdf = gpd.GeoDataFrame(isochrones, crs="EPSG:4326")
iso_gdf.explore(column="minutes", cmap="RdYlGn_r", tiles="CartoDB positron")
```

### Recipe 7: Batch Reproject and Compress Rasters

```python
"""Reproject and compress all GeoTIFFs in a directory to COG format."""
from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

input_dir = Path("raw_tiffs/")
output_dir = Path("cog_output/")
output_dir.mkdir(exist_ok=True)
target_crs = "EPSG:4326"

for tif in input_dir.glob("*.tif"):
    with rasterio.open(tif) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            crs=target_crs, transform=transform,
            width=width, height=height,
            driver="GTiff", compress="deflate",
            tiled=True, blockxsize=256, blockysize=256,
        )

        out_path = output_dir / tif.name
        with rasterio.open(out_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear,
                )

    # Add overviews for COG
    with rasterio.open(out_path, "r+") as dst:
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling="average")

    print(f"Converted: {tif.name}")
```

### Recipe 8: Voronoi Polygons from Point Data

```python
"""Create Voronoi polygons from a set of points, clipped to a boundary."""
import geopandas as gpd
from shapely.ops import voronoi_diagram
from shapely.geometry import MultiPoint

points = gpd.read_file("weather_stations.gpkg")
boundary = gpd.read_file("country_boundary.gpkg").geometry.unary_union

# Create Voronoi diagram
multipoint = MultiPoint(points.geometry.tolist())
voronoi = voronoi_diagram(multipoint, envelope=boundary)

# Convert to GeoDataFrame and clip
voronoi_gdf = gpd.GeoDataFrame(geometry=list(voronoi.geoms), crs=points.crs)
voronoi_gdf = gpd.clip(voronoi_gdf, boundary)

# Assign point attributes to Voronoi polygons via spatial join
voronoi_gdf = gpd.sjoin(voronoi_gdf, points, predicate="contains")
voronoi_gdf.to_parquet("voronoi_zones.parquet")
```

### Recipe 9: Heatmap from CSV Coordinates

```python
"""Create a heatmap from a CSV file with lat/lon columns."""
import pandas as pd
import folium
from folium.plugins import HeatMap

df = pd.read_csv("incidents.csv")  # columns: lat, lon, severity
df = df.dropna(subset=["lat", "lon"])

m = folium.Map(
    location=[df.lat.mean(), df.lon.mean()],
    zoom_start=12,
    tiles="CartoDB positron",
)

heat_data = df[["lat", "lon", "severity"]].values.tolist()
HeatMap(heat_data, radius=20, blur=15, max_zoom=15).add_to(m)
m.save("heatmap.html")
```

### Recipe 10: Land Use Change Detection

```python
"""Compare two classified rasters and compute change statistics."""
import rasterio
import numpy as np
import pandas as pd

with rasterio.open("landuse_2015.tif") as src1, rasterio.open("landuse_2020.tif") as src2:
    lc2015 = src1.read(1)
    lc2020 = src2.read(1)
    profile = src1.profile.copy()

# Change detection
change = np.where(lc2015 != lc2020, 1, 0)
change_type = lc2015 * 100 + lc2020  # Encode from-to transitions

# Transition matrix
classes = {1: "Urban", 2: "Forest", 3: "Agriculture", 4: "Water"}
transitions = {}
for from_cls in classes:
    for to_cls in classes:
        if from_cls != to_cls:
            count = np.sum((lc2015 == from_cls) & (lc2020 == to_cls))
            if count > 0:
                transitions[f"{classes[from_cls]} -> {classes[to_cls]}"] = count

transition_df = pd.DataFrame(
    list(transitions.items()), columns=["Transition", "Pixel Count"]
).sort_values("Pixel Count", ascending=False)
print(transition_df)

# Save change raster
profile.update(dtype="uint8")
with rasterio.open("change_2015_2020.tif", "w", **profile) as dst:
    dst.write(change.astype(np.uint8), 1)
```

### Recipe 11: Geocode Addresses and Find Nearest Facility

```python
"""Geocode a list of addresses and find the nearest hospital to each."""
import geopandas as gpd
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

addresses = pd.read_csv("customer_addresses.csv")
hospitals = gpd.read_file("hospitals.gpkg")

# Geocode
geolocator = Nominatim(user_agent="facility_finder")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

addresses["location"] = addresses["address"].apply(geocode)
addresses = addresses.dropna(subset=["location"])
addresses["lat"] = addresses["location"].apply(lambda x: x.latitude)
addresses["lon"] = addresses["location"].apply(lambda x: x.longitude)

customers = gpd.GeoDataFrame(
    addresses, geometry=gpd.points_from_xy(addresses.lon, addresses.lat), crs="EPSG:4326"
)

# Nearest join
nearest = gpd.sjoin_nearest(
    customers.to_crs(epsg=32633),
    hospitals.to_crs(epsg=32633),
    how="left",
    distance_col="distance_m",
)

nearest["distance_km"] = nearest["distance_m"] / 1000
print(nearest[["address", "hospital_name", "distance_km"]].head(10))
```

### Recipe 12: Watershed Delineation

```python
"""Delineate watersheds from a DEM using WhiteboxTools."""
from whitebox import WhiteboxTools
import rasterio
import geopandas as gpd

wbt = WhiteboxTools()
wbt.set_working_dir("/path/to/data")

# Hydrological conditioning
wbt.fill_depressions("dem.tif", "dem_filled.tif")
wbt.d8_pointer("dem_filled.tif", "flow_dir.tif")
wbt.d8_flow_accumulation("flow_dir.tif", "flow_acc.tif")

# Extract stream network
wbt.extract_streams("flow_acc.tif", "streams.tif", threshold=5000)

# Snap pour points to stream network
wbt.snap_pour_points("pour_points.shp", "flow_acc.tif", "snapped_points.shp", snap_dist=100)

# Delineate watersheds
wbt.watershed("flow_dir.tif", "snapped_points.shp", "watersheds.tif")

# Vectorize watersheds
from rasterio.features import shapes
from shapely.geometry import shape

with rasterio.open("watersheds.tif") as src:
    data = src.read(1)
    mask = data != src.nodata
    results = list(shapes(data, mask=mask, transform=src.transform))

gdf = gpd.GeoDataFrame(
    [{"watershed_id": int(v), "geometry": shape(g)} for g, v in results],
    crs=src.crs,
)
gdf.to_parquet("watersheds.parquet")
```

### Recipe 13: API Integration (Overpass / Nominatim)

```python
"""Query OpenStreetMap Overpass API for features in an area."""
import requests
import geopandas as gpd
from shapely.geometry import shape
import json

# Overpass API: find all hospitals in Berlin
overpass_url = "https://overpass-api.de/api/interpreter"
query = """
[out:json][timeout:60];
area["name"="Berlin"]["admin_level"="4"]->.searchArea;
(
  node["amenity"="hospital"](area.searchArea);
  way["amenity"="hospital"](area.searchArea);
  relation["amenity"="hospital"](area.searchArea);
);
out center;
"""

response = requests.post(overpass_url, data={"data": query})
data = response.json()

# Parse results
features = []
for element in data["elements"]:
    if "center" in element:
        lat, lon = element["center"]["lat"], element["center"]["lon"]
    elif "lat" in element:
        lat, lon = element["lat"], element["lon"]
    else:
        continue

    features.append({
        "name": element.get("tags", {}).get("name", "Unknown"),
        "type": element.get("tags", {}).get("healthcare", "hospital"),
        "latitude": lat,
        "longitude": lon,
    })

import pandas as pd
df = pd.DataFrame(features)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")
gdf.to_parquet("berlin_hospitals.parquet")
print(f"Found {len(gdf)} hospitals")
```

### Recipe 14: Merge Multiple GeoPackage Layers

```python
"""Merge all layers from multiple GeoPackage files into one."""
import geopandas as gpd
import fiona
from pathlib import Path

input_files = list(Path("regions/").glob("*.gpkg"))
merged_layers = {}

for gpkg_path in input_files:
    layers = fiona.listlayers(str(gpkg_path))
    for layer_name in layers:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
        gdf["source_file"] = gpkg_path.stem
        if layer_name in merged_layers:
            merged_layers[layer_name] = gpd.GeoDataFrame(
                pd.concat([merged_layers[layer_name], gdf], ignore_index=True)
            )
        else:
            merged_layers[layer_name] = gdf

# Write merged layers to single GeoPackage
for layer_name, gdf in merged_layers.items():
    gdf.to_file("merged.gpkg", layer=layer_name, driver="GPKG")
    print(f"Layer '{layer_name}': {len(gdf)} features")
```

### Recipe 15: Spatial Clustering and Labeling

```python
"""Cluster point data spatially and label each cluster."""
import geopandas as gpd
import numpy as np
from sklearn.cluster import HDBSCAN
from shapely.geometry import MultiPoint

gdf = gpd.read_file("events.gpkg").to_crs(epsg=32633)
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

# Cluster
clusterer = HDBSCAN(min_cluster_size=10, min_samples=3)
gdf["cluster"] = clusterer.fit_predict(coords)

# Label clusters by centroid location
clusters = gdf[gdf["cluster"] >= 0].dissolve(by="cluster")
clusters["centroid"] = clusters.geometry.centroid
clusters["label"] = [f"Cluster {i} (n={len(gdf[gdf.cluster==i])})" for i in clusters.index]

# Convex hulls for cluster boundaries
cluster_hulls = gdf[gdf["cluster"] >= 0].groupby("cluster").apply(
    lambda g: g.geometry.unary_union.convex_hull
).reset_index(name="geometry")
cluster_gdf = gpd.GeoDataFrame(cluster_hulls, geometry="geometry", crs=gdf.crs)
cluster_gdf.to_parquet("cluster_boundaries.parquet")
```

### Recipe 16: Raster Sampling at Point Locations

```python
"""Extract raster values at point locations (multi-band)."""
import rasterio
import geopandas as gpd
import numpy as np

points = gpd.read_file("sample_points.gpkg")

with rasterio.open("multispectral.tif") as src:
    # Ensure same CRS
    if points.crs != src.crs:
        points = points.to_crs(src.crs)

    coords = [(p.x, p.y) for p in points.geometry]
    values = list(src.sample(coords))

    for band_idx in range(src.count):
        band_name = src.descriptions[band_idx] or f"band_{band_idx+1}"
        points[band_name] = [v[band_idx] for v in values]

points.to_parquet("points_with_raster_values.parquet")
```

### Recipe 17: Create Hexagonal Grid

```python
"""Generate a hexagonal grid covering an area of interest."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon

def create_hex_grid(bounds, hex_size):
    """Create hexagonal grid covering the given bounds."""
    xmin, ymin, xmax, ymax = bounds
    cols = int((xmax - xmin) / (hex_size * 1.5)) + 2
    rows = int((ymax - ymin) / (hex_size * np.sqrt(3))) + 2

    hexagons = []
    for col in range(cols):
        for row in range(rows):
            x = xmin + col * hex_size * 1.5
            y = ymin + row * hex_size * np.sqrt(3)
            if col % 2 == 1:
                y += hex_size * np.sqrt(3) / 2

            hex_coords = []
            for angle in range(6):
                angle_rad = np.radians(60 * angle + 30)
                hx = x + hex_size * np.cos(angle_rad)
                hy = y + hex_size * np.sin(angle_rad)
                hex_coords.append((hx, hy))
            hex_coords.append(hex_coords[0])
            hexagons.append(Polygon(hex_coords))

    return hexagons

aoi = gpd.read_file("aoi.gpkg").to_crs(epsg=32633)
hexagons = create_hex_grid(aoi.total_bounds, hex_size=500)  # 500m hexagons

hex_gdf = gpd.GeoDataFrame(geometry=hexagons, crs="EPSG:32633")
hex_gdf = gpd.clip(hex_gdf, aoi)
hex_gdf["hex_id"] = range(len(hex_gdf))
hex_gdf.to_parquet("hex_grid_500m.parquet")
```

### Recipe 18: Temporal Join (Events to Time Windows)

```python
"""Join spatial events to temporal windows (e.g., hourly bins)."""
import geopandas as gpd
import pandas as pd

events = gpd.read_file("traffic_incidents.gpkg")
events["timestamp"] = pd.to_datetime(events["timestamp"])

# Temporal binning
events["hour"] = events["timestamp"].dt.floor("h")
events["day_of_week"] = events["timestamp"].dt.day_name()

# Aggregate by grid cell and hour
events_proj = events.to_crs(epsg=32633)
events_proj["grid_x"] = (events_proj.geometry.x // 1000).astype(int) * 1000
events_proj["grid_y"] = (events_proj.geometry.y // 1000).astype(int) * 1000

hourly_grid = events_proj.groupby(["grid_x", "grid_y", "hour"]).agg(
    count=("timestamp", "count"),
    avg_severity=("severity", "mean"),
).reset_index()

# Peak hour analysis
peak_hours = hourly_grid.groupby("hour")["count"].sum().sort_values(ascending=False)
print("Top 5 peak hours:")
print(peak_hours.head())
```

### Recipe 19: Multi-Criteria Site Selection

```python
"""Find optimal locations based on multiple spatial criteria."""
import geopandas as gpd
import numpy as np
from shapely import STRtree

parcels = gpd.read_file("parcels.gpkg").to_crs(epsg=32633)
roads = gpd.read_file("major_roads.gpkg").to_crs(epsg=32633)
schools = gpd.read_file("schools.gpkg").to_crs(epsg=32633)
flood_zones = gpd.read_file("flood_zones.gpkg").to_crs(epsg=32633)

# Criterion 1: Parcel area > 2000 m2
parcels["area_m2"] = parcels.geometry.area
c1 = parcels["area_m2"] > 2000

# Criterion 2: Within 500m of a major road
road_tree = STRtree(roads.geometry.values)
road_idx = road_tree.query(parcels.geometry.values, predicate="dwithin", distance=500)
near_road_parcels = set(road_idx[0])
c2 = parcels.index.isin(near_road_parcels)

# Criterion 3: More than 200m from schools (noise)
school_tree = STRtree(schools.geometry.values)
school_idx = school_tree.query(parcels.geometry.values, predicate="dwithin", distance=200)
near_school_parcels = set(school_idx[0])
c3 = ~parcels.index.isin(near_school_parcels)

# Criterion 4: Not in flood zone
flood_union = flood_zones.geometry.unary_union
c4 = ~parcels.geometry.intersects(flood_union)

# Combine criteria
suitable = parcels[c1 & c2 & c3 & c4].copy()
suitable["score"] = (
    0.4 * (suitable["area_m2"] / suitable["area_m2"].max()) +
    0.3 * 1.0 +  # Near road (binary, already filtered)
    0.3 * 1.0    # Not in flood zone (binary, already filtered)
)

suitable = suitable.sort_values("score", ascending=False)
print(f"Found {len(suitable)} suitable parcels")
suitable.to_parquet("suitable_sites.parquet")
```

### Recipe 20: Export Map to Multiple Formats

```python
"""Create a publication map and export to PNG, SVG, and PDF."""
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx

gdf = gpd.read_file("results.gpkg").to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(
    ax=ax, column="value", cmap="YlOrRd",
    legend=True, edgecolor="black", linewidth=0.3,
    legend_kwds={"label": "Value", "shrink": 0.6},
)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
ax.set_axis_off()
ax.set_title("Analysis Results", fontsize=16, fontweight="bold", pad=20)

# Export to multiple formats
fig.savefig("map.png", dpi=300, bbox_inches="tight", facecolor="white")
fig.savefig("map.svg", bbox_inches="tight", facecolor="white")
fig.savefig("map.pdf", bbox_inches="tight", facecolor="white")
plt.close()
print("Exported: map.png, map.svg, map.pdf")
```

### Recipe 21: Validate and Fix Geometries

```python
"""Check and repair invalid geometries in a dataset."""
import geopandas as gpd
from shapely import is_valid, make_valid, get_coordinate_dimension
from shapely.validation import explain_validity
import numpy as np

gdf = gpd.read_file("messy_data.gpkg")

# Check validity
gdf["is_valid"] = gdf.geometry.is_valid
invalid = gdf[~gdf["is_valid"]]
print(f"Invalid geometries: {len(invalid)} / {len(gdf)}")

# Explain why they're invalid
for idx, row in invalid.head(5).iterrows():
    print(f"  Row {idx}: {explain_validity(row.geometry)}")

# Fix all invalid geometries
gdf["geometry"] = make_valid(gdf.geometry.values)

# Remove empty geometries
gdf = gdf[~gdf.geometry.is_empty]

# Remove null geometries
gdf = gdf[gdf.geometry.notna()]

# Verify
assert gdf.geometry.is_valid.all(), "Still have invalid geometries!"
print(f"All {len(gdf)} geometries are valid")

gdf.to_parquet("clean_data.parquet")
```

### Recipe 22: Build a Spatial Database from Scratch

```python
"""Set up a PostGIS database and load spatial data."""
from sqlalchemy import create_engine, text
import geopandas as gpd

engine = create_engine("postgresql://gis:gispass@localhost:5432/gisdb")

# Enable PostGIS extension
with engine.connect() as conn:
    conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    conn.commit()

# Load data
datasets = {
    "buildings": "buildings.parquet",
    "roads": "roads.parquet",
    "parcels": "parcels.gpkg",
    "dem_points": "elevation_points.geojson",
}

for table_name, filepath in datasets.items():
    if filepath.endswith(".parquet"):
        gdf = gpd.read_parquet(filepath)
    else:
        gdf = gpd.read_file(filepath)

    gdf.to_postgis(table_name, engine, if_exists="replace", index=True)

    # Create spatial index
    with engine.connect() as conn:
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_geom ON {table_name} USING GIST (geometry)"))
        conn.commit()

    print(f"Loaded {table_name}: {len(gdf)} features")

# Verify
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT table_name, pg_size_pretty(pg_total_relation_size(table_name::regclass))
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """))
    for row in result:
        print(f"  {row[0]}: {row[1]}")
```

---

## Version Compatibility Matrix

| Library | Min Python | Latest Version | GDAL Required | Notes |
|---|---|---|---|---|
| GeoPandas | 3.10 | 1.0+ | Via Fiona/pyogrio | Requires Shapely 2.0+ |
| Shapely | 3.8 | 2.0+ | No (uses GEOS) | Vectorized API |
| Rasterio | 3.9 | 1.4+ | Yes (3.5+) | Wheels include GDAL |
| Fiona | 3.8 | 1.10+ | Yes (3.1+) | Legacy engine |
| pyogrio | 3.9 | 0.9+ | Yes (3.4+) | Default in GeoPandas 1.0 |
| PyProj | 3.10 | 3.7+ | No (uses PROJ) | |
| Folium | 3.7 | 0.17+ | No | |
| Lonboard | 3.9 | 0.10+ | No | Requires Jupyter |
| PySAL | 3.10 | 24.1+ | No | Meta-package |
| DuckDB | 3.8 | 1.1+ | No | Spatial extension |
| xarray | 3.10 | 2024.10+ | No | |
| rioxarray | 3.10 | 0.17+ | Via Rasterio | |

---

**Further Reading:**

- [Python Libraries Catalog](../tools/python-libraries.md) - Comprehensive listing of all Python geospatial packages
- [Spatial Databases](../tools/spatial-databases.md) - PostGIS, SpatiaLite, DuckDB Spatial setup and administration
- [Visualization Guide](../visualization/README.md) - Map design principles and library comparisons
- [Data Sources](../data-sources/README.md) - Where to find geospatial data
- [Satellite Imagery](../data-sources/satellite-imagery.md) - Remote sensing data access
- [ML for GIS](../tools/ai-ml-geospatial.md) - Machine learning in geospatial contexts
- [R Stack](r-stack.md) - Complementary R geospatial ecosystem
- [Spatial Statistics](spatial-statistics.md) - Statistical methods for spatial data

---

[Back to Data Analysis](README.md) | [Back to Main README](../README.md)
