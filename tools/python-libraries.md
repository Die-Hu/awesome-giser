# Python Geospatial Libraries

The Python geospatial ecosystem is one of the richest in data science. This guide covers the full stack — from low-level geometry engines to high-level visualization — with real code, benchmark notes, and the tricks that separate intermediate users from experts.

> **Quick Picks**
> - :trophy: **SOTA**: [GeoPandas 1.x](https://geopandas.org) + [Shapely 2.0](https://shapely.readthedocs.io) --- the vectorized combo that replaced the old slow stack; Shapely 2.0's C-level array ops make the classic "loop over geometries" anti-pattern obsolete
> - :moneybag: **Free Best**: [leafmap](https://leafmap.org) --- one-liner interactive maps with COG/STAC/GEE support and zero boilerplate; replaces hours of folium configuration
> - :zap: **Fastest Setup**: `pip install geopandas` then `gdf = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))` --- you are plotting a world map in under 60 seconds

---

## Core Stack (the "Big 5")

These five libraries are the foundation of 95% of Python GIS work. Master them before everything else.

---

### GeoPandas

GeoPandas extends Pandas DataFrames with a geometry column and spatial operations. Since v0.14 it ships with Shapely 2.0 as its geometry engine, unlocking vectorized C-level operations that make the old "apply a lambda over each geometry" pattern unnecessary.

- **Install**: `pip install geopandas` or `conda install -c conda-forge geopandas`
- **Links**: [geopandas.org](https://geopandas.org) | [API Docs](https://geopandas.org/en/stable/docs/reference.html)

#### Quick Start

```python
import geopandas as gpd

# Read any OGR-supported format
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
print(world.crs)           # EPSG:4326
print(world.geometry.type.value_counts())

# Basic spatial operations
world['area_km2'] = world.to_crs(epsg=6933).area / 1e6  # equal-area projection
world.plot(column='area_km2', cmap='OrRd', legend=True)
```

#### Reading and Writing All Formats

```python
import geopandas as gpd

# Vector formats via GDAL/OGR
gdf = gpd.read_file("data.gpkg", layer="roads")           # GeoPackage layer
gdf = gpd.read_file("data.geojson")                        # GeoJSON
gdf = gpd.read_file("data.shp")                            # Shapefile (legacy)
gdf = gpd.read_file("data.fgb")                            # FlatGeobuf (streaming-friendly)
gdf = gpd.read_file("data.gpkg", bbox=(xmin, ymin, xmax, ymax))   # spatial filter at read time
gdf = gpd.read_file("data.gpkg", where="population > 1000000")    # attribute filter at read time

# GeoParquet — fastest for large analytical datasets
gdf = gpd.read_parquet("data.parquet")                     # requires pyarrow
gdf.to_parquet("output.parquet", compression="snappy")

# Writing
gdf.to_file("output.gpkg", layer="result", driver="GPKG")
gdf.to_file("output.geojson", driver="GeoJSON")
gdf[["id", "geometry"]].to_file("out.fgb", driver="FlatGeobuf")  # include only needed cols
```

**Performance note**: `read_parquet()` is 5-20x faster than `read_file()` for GeoPackage on large datasets because it avoids row-by-row OGR deserialization. For analytical workflows, convert your source data to GeoParquet once and read that.

#### Spatial Joins

```python
import geopandas as gpd

cities = gpd.read_file("cities.gpkg")
countries = gpd.read_file("countries.gpkg")

# Standard spatial join (predicate: intersects, contains, within, crosses, touches)
joined = gpd.sjoin(cities, countries, how="left", predicate="within")

# Nearest-neighbor join — critical for address matching, snapping, fuzzy joins
# max_distance is in CRS units (use projected CRS for meaningful meters)
cities_proj = cities.to_crs(epsg=3857)
pois_proj = pois.to_crs(epsg=3857)

nearest = gpd.sjoin_nearest(
    cities_proj,
    pois_proj,
    how="left",
    max_distance=500,          # within 500 meters only
    distance_col="dist_m"      # adds a column with the actual distance
)

# Many-to-many: find all points within polygons (not just the first match)
result = gpd.sjoin(points, polygons, how="inner", predicate="within")
```

#### Overlay Operations

```python
import geopandas as gpd

a = gpd.read_file("parcels.gpkg")
b = gpd.read_file("flood_zones.gpkg")

# Geometric set operations — keeps attributes from both inputs
intersection = gpd.overlay(a, b, how="intersection")   # A AND B
union        = gpd.overlay(a, b, how="union")          # A OR B (with splits)
difference   = gpd.overlay(a, b, how="difference")     # A NOT B
sym_diff     = gpd.overlay(a, b, how="symmetric_difference")  # XOR

# Dissolve — aggregate geometries by attribute (like GROUP BY for geometry)
provinces = counties.dissolve(by="state_fips", aggfunc={
    "population": "sum",
    "area_km2":   "sum",
    "name":       "first"
})
```

#### CRS Handling

```python
import geopandas as gpd

gdf = gpd.read_file("data.shp")

# Check CRS
print(gdf.crs)                         # CRS object
print(gdf.crs.to_epsg())               # integer EPSG code
print(gdf.crs.is_geographic)           # True if lat/lon
print(gdf.crs.is_projected)            # True if meters/feet

# Reproject (returns new GeoDataFrame, does not modify in place)
gdf_web = gdf.to_crs(epsg=3857)        # Web Mercator
gdf_utm = gdf.to_crs("EPSG:32632")    # UTM Zone 32N
gdf_laea = gdf.to_crs(epsg=3035)      # ETRS89 LAEA (equal-area Europe)

# Set CRS without reprojecting (fix missing/wrong CRS metadata)
gdf = gdf.set_crs(epsg=4326)          # assign
gdf = gdf.set_crs(epsg=4326, allow_override=True)  # force override wrong CRS
```

#### ADVANCED: Manual R-tree Queries with `.sindex`

The spatial index on a GeoDataFrame is exposed directly. Use it when you need custom spatial query logic that `sjoin` does not cover:

```python
import geopandas as gpd
import numpy as np

buildings = gpd.read_file("buildings.gpkg")
query_geom = shapely.geometry.box(xmin, ymin, xmax, ymax)

# .sindex gives you the underlying STRtree
# query() returns integer positions (iloc-style)
candidate_idx = buildings.sindex.query(query_geom, predicate="intersects")
candidates = buildings.iloc[candidate_idx]

# Batch query: for each row in 'parcels', find all buildings that intersect it
parcels = gpd.read_file("parcels.gpkg")
# Returns two arrays: [input_index, tree_index]
left_idx, right_idx = buildings.sindex.query(
    parcels.geometry,
    predicate="intersects"
)
pairs = list(zip(parcels.index[left_idx], buildings.index[right_idx]))
```

#### ADVANCED: Vectorized Shapely 2.0 for 100x Speedup

```python
import geopandas as gpd
import shapely
import numpy as np

gdf = gpd.read_file("large_dataset.gpkg")

# SLOW — old pattern, iterates Python objects one by one
buffers = gdf.geometry.apply(lambda g: g.buffer(100))

# FAST — Shapely 2.0 operates on the entire geometry array at C level
buffers = shapely.buffer(gdf.geometry.values, 100)       # returns numpy array
# Or directly on the GeoSeries:
buffers = gdf.geometry.buffer(100)                       # GeoPandas uses Shapely 2.0 internally

# Vectorized distance between two aligned GeoSeries
dist = shapely.distance(gdf_a.geometry.values, gdf_b.geometry.values)

# Area of all geometries (no loop needed)
areas = shapely.area(gdf.geometry.values)

# Batch WKT parsing — much faster than a list comprehension
wkt_series = pd.Series(["POINT(0 0)", "POINT(1 1)", "LINESTRING(0 0, 1 1)"])
geoms = shapely.from_wkt(wkt_series.values)   # numpy array of geometries
```

#### ADVANCED: Chunked Reading of Huge Files

```python
import geopandas as gpd
import pandas as pd

# Fiona-backed chunked reading for files too large for RAM
# Read in chunks and process incrementally
chunks = []
with fiona.open("huge_dataset.gpkg") as src:
    chunk_size = 50_000
    features = []
    for i, feature in enumerate(src):
        features.append(feature)
        if len(features) == chunk_size:
            chunk_gdf = gpd.GeoDataFrame.from_features(features, crs=src.crs)
            # process chunk here (filter, reproject, etc.)
            result = chunk_gdf[chunk_gdf.geometry.area > 1000]
            chunks.append(result)
            features = []
    if features:
        chunks.append(gpd.GeoDataFrame.from_features(features, crs=src.crs))

final = pd.concat(chunks, ignore_index=True)

# Alternative: use pyogrio (fastest OGR Python binding) with skip_features/max_features
import pyogrio
gdf = pyogrio.read_dataframe("huge.gpkg", skip_features=100_000, max_features=50_000)
```

#### ADVANCED: GeoArrow Integration

```python
import geopandas as gpd
import pyarrow as pa

# GeoParquet uses GeoArrow encoding under the hood
# Round-trip through Arrow for zero-copy interop with DuckDB, Polars, etc.
gdf = gpd.read_file("parcels.gpkg")

# Export to Arrow table (geometry as WKB binary column)
table = gdf.to_arrow()   # requires geopandas >= 1.0 + pyarrow

# Or use pa.Table directly for DuckDB
import duckdb
duckdb.execute("INSTALL spatial; LOAD spatial;")
# Register the Arrow table and query spatially
duckdb.register("parcels_arrow", table)
result = duckdb.execute("""
    SELECT id, ST_Area(geom) AS area
    FROM parcels_arrow
    WHERE ST_Area(geom) > 10000
""").fetchdf()
```

---

### Shapely 2.0

Shapely is the geometry engine that GeoPandas uses internally. Version 2.0 (2022) was a ground-up rewrite: all operations now work on NumPy arrays of geometries at C level, making Python loops over geometries effectively obsolete for performance-sensitive work.

- **Install**: `pip install shapely` (Shapely 2.0+ is default since late 2022)
- **Links**: [shapely.readthedocs.io](https://shapely.readthedocs.io)

#### Quick Start

```python
import shapely
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely import wkt, wkb
import numpy as np

# Create geometries
pt = Point(13.405, 52.520)          # Berlin
line = LineString([(0, 0), (1, 1), (2, 0)])
poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# Basic predicates and measurements
print(poly.contains(pt))
print(poly.area, poly.length, poly.bounds)
```

#### Vectorized Operations with STRtree

```python
import shapely
import numpy as np
from shapely.geometry import Point

# Build an array of geometries
rng = np.random.default_rng(42)
points = shapely.points(rng.uniform(0, 100, 10_000), rng.uniform(0, 100, 10_000))

# STRtree for fast spatial queries on arrays
tree = shapely.STRtree(points)

# Query: which points are within 5 units of a query geometry?
query_circle = Point(50, 50).buffer(5)
result_idx = tree.query(query_circle, predicate="within")
nearby_points = points[result_idx]

# Batch nearest: for each of N query points, find the nearest in the tree
query_points = shapely.points([10, 20, 30], [10, 20, 30])
nearest_idx = tree.nearest(query_points)      # returns array of indices
distances = shapely.distance(query_points, points[nearest_idx])
```

#### Prepared Geometries

Prepared geometries pre-compute internal spatial index structures. Use them when the same geometry is tested against many others:

```python
import shapely

large_polygon = shapely.from_wkt("POLYGON((...))")  # complex polygon

# Prepare once, query many times — 5-10x faster for contains/intersects
prepared = shapely.prepare(large_polygon)
# shapely automatically uses the prepared version for predicates now
results = shapely.contains(large_polygon, points_array)  # uses prepared internally

# For maximum control, prepare manually and use the predicate functions:
shapely.prepare(large_polygon)
mask = shapely.contains(large_polygon, points_array)    # C-level batch test
```

#### Union Operations

```python
import shapely
from shapely.ops import unary_union

polygons = list(gdf.geometry)   # list of Shapely geometries

# SLOW — iterative union (quadratic complexity)
result = polygons[0]
for p in polygons[1:]:
    result = result.union(p)

# FAST — unary_union uses a tree-based merge strategy
result = unary_union(polygons)                    # shapely.ops version

# FASTER for Shapely 2.0 — operates on geometry array directly
import shapely
result = shapely.union_all(shapely.from_shapely(polygons))   # Shapely 2.0 array API
# or on a numpy array:
geom_array = gdf.geometry.values   # numpy array of geometries
result = shapely.union_all(geom_array)
```

#### Geometry Creation from WKT / WKB / GeoJSON

```python
import shapely
import json

# WKT
geom = shapely.from_wkt("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
geom_array = shapely.from_wkt(["POINT(0 0)", "POINT(1 1)", None])  # handles nulls

# WKB (hex or bytes)
wkb_hex = geom.wkb_hex
geom_back = shapely.from_wkb(wkb_hex)
geom_back = shapely.from_wkb(bytes.fromhex(wkb_hex))

# GeoJSON
geojson_dict = {"type": "Point", "coordinates": [13.405, 52.520]}
geom = shapely.from_geojson(json.dumps(geojson_dict))

# Export
print(shapely.to_wkt(geom))
print(shapely.to_wkb(geom).hex())
print(shapely.to_geojson(geom))
```

#### Validation and Fixing Topology Issues

```python
import shapely

# Check validity
geom = shapely.from_wkt("POLYGON((0 0, 1 1, 1 0, 0 1, 0 0))")  # self-intersecting bowtie
print(shapely.is_valid(geom))           # False
print(shapely.is_valid_reason(geom))    # "Self-intersection[...]"

# Fix invalid geometries
fixed = shapely.make_valid(geom)        # returns valid geometry (may split into multi)
print(shapely.is_valid(fixed))          # True

# Batch fix an entire GeoDataFrame column
import geopandas as gpd
gdf = gpd.read_file("messy_data.gpkg")
invalid_mask = ~shapely.is_valid(gdf.geometry.values)
if invalid_mask.any():
    gdf.loc[invalid_mask, "geometry"] = shapely.make_valid(
        gdf.geometry.values[invalid_mask]
    )
```

#### TRICK: `set_precision()` to Fix Topology Issues

Floating-point imprecision in coordinates causes countless "topology exception" errors. `set_precision()` snaps all coordinates to a grid, resolving these issues:

```python
import shapely

# Coordinates with floating-point noise causing topology failures
polygon_a = shapely.from_wkt("POLYGON((0 0, 1.0000000001 0, 1 1, 0 1, 0 0))")
polygon_b = shapely.from_wkt("POLYGON((1 0, 2 0, 2 1, 1.0000000001 1, 1 0))")

# These may raise TopologyException:
# result = shapely.intersection(polygon_a, polygon_b)

# Fix: snap to a grid first (1e-6 meter precision is usually fine)
a_snapped = shapely.set_precision(polygon_a, grid_size=1e-6)
b_snapped = shapely.set_precision(polygon_b, grid_size=1e-6)
result = shapely.intersection(a_snapped, b_snapped)   # works cleanly

# Apply to an entire array
clean = shapely.set_precision(gdf.geometry.values, grid_size=0.001)  # 1mm grid
```

#### Geodesic Measurements with `shapely.measurement`

```python
import shapely

# Note: Shapely works in Cartesian space. For geodesic (true Earth) measurements,
# use pyproj.Geod (see pyproj section). However, shapely has basic measurement funcs:
geom = shapely.from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
print(shapely.area(geom))        # 100 (CRS units squared)
print(shapely.length(geom))      # 40
print(shapely.hausdorff_distance(geom_a, geom_b))   # shape similarity metric
print(shapely.frechet_distance(line_a, line_b))      # trajectory similarity
```

---

### Rasterio

Rasterio is the standard Python library for reading and writing raster data, built on top of GDAL. It exposes a clean, Pythonic API with NumPy arrays and context managers.

- **Install**: `pip install rasterio` or `conda install -c conda-forge rasterio`
- **Links**: [rasterio.readthedocs.io](https://rasterio.readthedocs.io)

#### Quick Start

```python
import rasterio
import numpy as np

with rasterio.open("dem.tif") as src:
    print(src.crs, src.transform, src.count, src.dtypes)
    # Read all bands
    data = src.read()                    # shape: (bands, rows, cols)
    band1 = src.read(1)                  # shape: (rows, cols)
    print(src.meta)                      # dict of profile metadata
```

#### Windowed Reading (Memory-Efficient for Giant Rasters)

```python
import rasterio
from rasterio.windows import Window

with rasterio.open("giant_landsat.tif") as src:
    # Read a specific pixel window
    window = Window(col_off=1000, row_off=500, width=256, height=256)
    data = src.read(window=window)

    # Iterate over the full raster in tiles
    for ji, window in src.block_windows(1):   # use dataset's native block size
        data = src.read(window=window)
        # process tile...
        result = process(data)
        # write result to output (must open dst with same profile)
```

#### Masking Rasters with Vector Data

```python
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import mapping

shapes = gpd.read_file("watershed.gpkg")
geometry_list = [mapping(g) for g in shapes.geometry]

with rasterio.open("dem.tif") as src:
    out_image, out_transform = mask(
        src,
        geometry_list,
        crop=True,           # crop to geometry bounding box
        nodata=-9999,
        all_touched=False    # True to include edge pixels
    )
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": -9999
    })

with rasterio.open("clipped_dem.tif", "w", **out_meta) as dst:
    dst.write(out_image)
```

#### Virtual Warping with `WarpedVRT`

Reproject on-the-fly without writing a new file to disk. This is critical for reading data from different projections into a common analysis grid:

```python
import rasterio
from rasterio.vrt import WarpedVRT
import rasterio.crs

with rasterio.open("utm_data.tif") as src:
    # Create a virtual warped view in WGS84
    with WarpedVRT(
        src,
        crs=rasterio.crs.CRS.from_epsg(4326),
        resampling=rasterio.enums.Resampling.bilinear
    ) as vrt:
        data = vrt.read()          # reads and reprojects on the fly
        print(vrt.crs)             # EPSG:4326
        print(vrt.transform)       # new geographic transform
        # Can save to file:
        from rasterio.shutil import copy
        copy(vrt, "reprojected.tif", driver="GTiff")
```

#### Concurrent Reads with GDAL Thread Settings

```python
import rasterio

# Set GDAL environment for concurrent reads — important for cloud data
with rasterio.Env(
    GDAL_NUM_THREADS="ALL_CPUS",
    GDAL_CACHEMAX=512,                    # MB of GDAL cache
    VSI_CACHE=True,
    VSI_CACHE_SIZE=10_000_000,            # bytes
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MULTIPLEX="YES",
    GDAL_HTTP_VERSION="2"                 # HTTP/2 for S3
):
    with rasterio.open("s3://my-bucket/large.tif") as src:
        data = src.read()

# For reading many files in parallel using concurrent.futures:
from concurrent.futures import ThreadPoolExecutor
import rasterio

def read_tile(path):
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(path) as src:
            return src.read(1)

paths = ["tile_001.tif", "tile_002.tif", "tile_003.tif"]
with ThreadPoolExecutor(max_workers=4) as executor:
    tiles = list(executor.map(read_tile, paths))
```

#### COG (Cloud-Optimized GeoTIFF) Creation

```python
import rasterio
from rasterio.shutil import copy

# Method 1: Write a standard GeoTIFF, then COG-ify it
with rasterio.open("input.tif") as src:
    copy(
        src,
        "output_cog.tif",
        driver="GTiff",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="deflate",
        predictor=2,              # horizontal differencing for better compression
        overview_resampling="average",
        copy_src_overviews=True,  # embed overviews (required for true COG)
    )

# Method 2: Use rio-cogeo (dedicated tool)
# pip install rio-cogeo
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
cog_translate("input.tif", "output.tif", cog_profiles.get("deflate"))
```

#### `MemoryFile` for In-Memory Rasters

Avoid writing intermediate files to disk. Critical for serverless/API workflows:

```python
import rasterio
from rasterio.io import MemoryFile
import numpy as np

# Write to memory
data = np.random.randint(0, 255, (1, 256, 256), dtype=np.uint8)
profile = {
    "driver": "GTiff", "dtype": "uint8",
    "width": 256, "height": 256, "count": 1,
    "crs": "EPSG:4326",
    "transform": rasterio.transform.from_bounds(-180, -90, 180, 90, 256, 256)
}

with MemoryFile() as memfile:
    with memfile.open(**profile) as dst:
        dst.write(data)
    # memfile.read() gives raw bytes — send as HTTP response, pass to another library, etc.
    raw_bytes = memfile.read()

# Read from bytes (e.g., from a network response or database BLOB)
with MemoryFile(raw_bytes) as memfile:
    with memfile.open() as src:
        data = src.read()
```

#### Reading from S3/HTTP via `/vsicurl/`

```python
import rasterio
import rasterio.env

# GDAL virtual filesystem paths work natively in rasterio
# /vsicurl/ — any HTTP/HTTPS URL
url = "https://opendata.example.com/dem_10m.tif"
with rasterio.open(f"/vsicurl/{url}") as src:
    print(src.meta)
    data = src.read(1, window=src.window(*src.bounds))  # read full raster

# /vsis3/ — S3 (set AWS credentials via env vars or IAM role)
with rasterio.open("s3://sentinel-cogs/sentinel-s2-l2a-cogs/2024/S2B.tif") as src:
    print(src.profile)

# /vsigz/ — gzip'd files
with rasterio.open("/vsigz//vsicurl/https://example.com/data.tif.gz") as src:
    data = src.read()
```

#### ADVANCED: Vector-to-Raster with `rasterize()`

```python
import rasterio
from rasterio.transform import from_bounds
from rasterio.features import rasterize
import geopandas as gpd

buildings = gpd.read_file("buildings.gpkg").to_crs(epsg=3857)

# Define output raster properties
resolution = 10   # 10 meters per pixel
xmin, ymin, xmax, ymax = buildings.total_bounds
width  = int((xmax - xmin) / resolution)
height = int((ymax - ymin) / resolution)
transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

# Burn geometry values into a raster array
burned = rasterize(
    shapes=((geom, value) for geom, value in zip(buildings.geometry, buildings["height"])),
    out_shape=(height, width),
    transform=transform,
    fill=0,           # background value
    dtype="float32",
    all_touched=False
)

with rasterio.open("buildings_height.tif", "w", driver="GTiff",
                   height=height, width=width, count=1,
                   dtype="float32", crs="EPSG:3857", transform=transform) as dst:
    dst.write(burned, 1)
```

#### ADVANCED: Raster-to-Vector with `shapes()`

```python
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

with rasterio.open("classified_landcover.tif") as src:
    image = src.read(1)
    transform = src.transform
    crs = src.crs

# Extract polygons for each unique pixel value
features = []
for geom_dict, value in shapes(image, transform=transform):
    features.append({
        "geometry": shape(geom_dict),
        "class_id": int(value)
    })

gdf = gpd.GeoDataFrame(features, crs=crs)
gdf = gdf[gdf["class_id"] != 0]   # remove nodata class
gdf.to_file("vectorized_landcover.gpkg", driver="GPKG")
```

---

### Fiona

Fiona is the direct OGR binding for reading and writing vector data. It is lower-level than GeoPandas and is useful when you need streaming iteration, fine-grained schema control, or want to avoid loading a full dataset into memory.

- **Install**: `pip install fiona` (installed automatically with geopandas)
- **Links**: [fiona.readthedocs.io](https://fiona.readthedocs.io)
- **Note**: pyogrio is now preferred as a faster alternative for bulk reads, but Fiona remains useful for streaming and schema inspection.

#### Quick Start

```python
import fiona

with fiona.open("data.gpkg") as src:
    print(src.crs)
    print(src.schema)             # {'geometry': 'Polygon', 'properties': {...}}
    print(src.driver)
    print(len(src))

    for feature in src:
        geom = feature["geometry"]
        props = feature["properties"]
        print(props["name"], geom["type"])
```

#### Filtering with `where` and `bbox`

```python
import fiona

# Attribute filter — pushed down to OGR, no Python iteration
with fiona.open("cities.gpkg", where="population > 1000000") as src:
    large_cities = list(src)

# Spatial filter — bounding box pushed to OGR spatial index
bbox = (-0.5, 51.3, 0.5, 51.7)   # London area (west, south, east, north)
with fiona.open("buildings.gpkg", bbox=bbox) as src:
    local_buildings = list(src)

# Combine both filters
with fiona.open("data.gpkg", where="type='park'", bbox=bbox) as src:
    parks_in_area = list(src)
```

#### Layer Management

```python
import fiona

# List all layers in a GeoPackage
layers = fiona.listlayers("atlas.gpkg")
print(layers)   # ['roads', 'buildings', 'hydrology']

# Read a specific layer
with fiona.open("atlas.gpkg", layer="hydrology") as src:
    rivers = list(src)

# Write a new layer to an existing GeoPackage
schema = {
    "geometry": "Point",
    "properties": {"name": "str", "elevation": "float"}
}
with fiona.open("atlas.gpkg", "w", driver="GPKG", layer="peaks",
                schema=schema, crs="EPSG:4326") as dst:
    dst.write({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [8.85, 45.83]},
        "properties": {"name": "Mont Blanc", "elevation": 4808.0}
    })
```

#### TRICK: Single-Geometry Reprojection Without Loading Full Dataset

When you only need to reproject one or a few geometries (e.g., a query bounding box), avoid loading a full GeoDataFrame:

```python
import fiona
import fiona.transform

# Reproject a single geometry dict from EPSG:4326 to EPSG:3857
geom = {"type": "Point", "coordinates": [13.405, 52.520]}
reprojected = fiona.transform.transform_geom(
    src_crs="EPSG:4326",
    dst_crs="EPSG:3857",
    geom=geom
)
print(reprojected)   # {"type": "Point", "coordinates": [1491800.44, 6894500.37]}

# Useful: convert a bounding box to the dataset's native CRS for spatial filtering
with fiona.open("utm_data.gpkg") as src:
    dataset_crs = src.crs
    query_bbox_wgs84 = (-0.5, 51.3, 0.5, 51.7)
    # Transform the bbox polygon to dataset CRS for proper spatial filter
    bbox_geom = {
        "type": "Polygon",
        "coordinates": [[[-0.5, 51.3], [0.5, 51.3], [0.5, 51.7], [-0.5, 51.7], [-0.5, 51.3]]]
    }
    bbox_native = fiona.transform.transform_geom("EPSG:4326", dataset_crs, bbox_geom)
```

---

### pyproj

pyproj is the Python binding for the PROJ library, handling coordinate reference system (CRS) definitions and transformations. It is the authoritative source of truth for CRS metadata in the Python geospatial stack.

- **Install**: `pip install pyproj` (installed automatically with geopandas)
- **Links**: [pyproj4.github.io](https://pyproj4.github.io/pyproj/)

#### Quick Start

```python
from pyproj import Transformer, CRS, Geod

# Transform coordinates (use Transformer, not the deprecated pyproj.transform())
transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
x, y = transformer.transform(13.405, 52.520)   # Berlin: lon, lat order with always_xy=True
print(x, y)   # 1491804.0, 6894805.0
```

#### `Transformer.from_crs()` — Always Use This

```python
from pyproj import Transformer
import numpy as np

# always_xy=True ensures (lon, lat) input regardless of CRS axis order
# This avoids the classic "swapped coordinates" bug with EPSG:4326 (which is lat/lon order)
t = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)

# Transform arrays of coordinates (vectorized, fast)
lons = np.array([13.405, 8.682, 2.349])
lats = np.array([52.520, 50.110, 48.853])
xs, ys = t.transform(lons, lats)

# Inverse transform
lons_back, lats_back = t.transform(xs, ys, direction="INVERSE")

# Round-trip many points in one call
x_arr, y_arr, z_arr = t.transform(lons, lats, np.zeros_like(lons))
```

#### `Geod` for Geodesic Distance and Area

```python
from pyproj import Geod

# Create geodesic calculator on the WGS84 ellipsoid
geod = Geod(ellps="WGS84")

# Distance between two points (returns forward azimuth, back azimuth, distance in meters)
az12, az21, dist_m = geod.inv(
    lons1=13.405, lats1=52.520,   # Berlin (always lon, lat)
    lons2=2.349,  lats2=48.853    # Paris
)
print(f"Berlin → Paris: {dist_m/1000:.1f} km")   # ~878 km

# Geodesic area of a polygon (returns area in m² and perimeter in m)
from shapely.geometry import Polygon
poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
lons, lats = poly.exterior.coords.xy
area, perimeter = geod.polygon_area_perimeter(lons, lats)
print(f"Area: {abs(area)/1e6:.2f} km²")  # abs() because sign indicates winding direction

# Generate points along a geodesic (great circle route)
n_pts = 100
r = geod.inv_intermediate(
    13.405, 52.520,   # Berlin
    -73.935, 40.730,  # New York
    npts=n_pts
)
route_lons, route_lats = r.lons, r.lats
```

#### CRS Object Inspection

```python
from pyproj import CRS

crs = CRS.from_epsg(32632)
print(crs.name)                    # "WGS 84 / UTM zone 32N"
print(crs.to_authority())          # ('EPSG', '32632')
print(crs.is_projected)            # True
print(crs.is_geographic)           # False
print(crs.axis_info)               # [Axis(name=Easting,...), Axis(name=Northing,...)]
print(crs.datum)                   # WGS 84
print(crs.to_wkt())                # full WKT2 string
print(crs.to_proj4())              # PROJ4 string (legacy)
print(crs.to_dict())               # dict representation

# Compare CRS objects
crs1 = CRS.from_epsg(4326)
crs2 = CRS.from_string("+proj=longlat +datum=WGS84")
print(crs1.equals(crs2))           # True
```

#### TRICK: Search for CRS by Area of Use

```python
from pyproj.database import query_crs_info
from pyproj import CRS

# Find all UTM zones that cover a specific area
results = query_crs_info(
    auth_name="EPSG",
    pj_types="PROJECTED_CRS",
    area_of_interest=(12.0, 50.0, 15.0, 55.0)   # (west, south, east, north)
)

for info in results[:5]:
    print(info.code, info.name)
# 32632 WGS 84 / UTM zone 32N
# 25832 ETRS89 / UTM zone 32N
# etc.

# Find the "best" CRS for an area (uses PROJ's area-of-use database)
from pyproj.aoi import AreaOfInterest
from pyproj.database import query_utm_crs_info

utm_info = query_utm_crs_info(
    datum_name="WGS 84",
    area_of_interest=AreaOfInterest(
        west_lon_degree=13.0,
        south_lat_degree=52.0,
        east_lon_degree=14.0,
        north_lat_degree=53.0
    )
)
print(utm_info[0].code)   # 32632
```

#### Pipeline Transformations

```python
from pyproj import Transformer

# PROJ pipeline for custom multi-step transformations
# Example: EPSG:25832 (ETRS89 UTM 32N) → custom oblique Mercator
pipeline = "+proj=pipeline +step +inv +proj=utm +zone=32 +ellps=GRS80 +step +proj=omerc +lat_0=52 +lonc=12 +alpha=0 +ellps=GRS80"
t = Transformer.from_pipeline(pipeline)
```

---

## Data Analysis & Manipulation

### Polars + Geo

Polars is a DataFrame library written in Rust with a lazy evaluation engine. For large-scale tabular processing with geometry columns, it is significantly faster than Pandas for filtering, grouping, and aggregation.

- **Install**: `pip install polars geoarrow-pandas`
- **Links**: [pola.rs](https://www.pola.rs) | [GeoPolars](https://github.com/geopolars/geopolars)

#### Why Polars for Geo Data

```python
import polars as pl
import geopandas as gpd

# Read a large CSV with coordinates — Polars is much faster than Pandas
df = pl.read_csv("10M_gps_points.csv")

# Fast filtering and computation on attributes (no geometry yet)
filtered = df.filter(
    (pl.col("speed") > 10) &
    (pl.col("timestamp").is_between(start, end))
).with_columns([
    (pl.col("x") * 2 - pl.col("y")).alias("derived")
])

# Convert to GeoPandas for spatial operations
import pandas as pd
pdf = filtered.to_pandas()
gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf["x"], pdf["y"]), crs="EPSG:4326")
```

#### Converting Between Polars and GeoPandas

```python
import polars as pl
import geopandas as gpd
import shapely

# GeoPandas → Polars (geometry as WKB bytes)
gdf = gpd.read_file("parcels.gpkg")
wkb_series = gdf.geometry.apply(lambda g: g.wkb)   # bytes
pl_df = pl.from_pandas(gdf.drop(columns="geometry")).with_columns(
    pl.Series("geometry_wkb", wkb_series)
)

# Polars → GeoPandas (decode WKB)
pdf = pl_df.to_pandas()
geoms = shapely.from_wkb(pdf["geometry_wkb"].values)
gdf_back = gpd.GeoDataFrame(pdf, geometry=geoms, crs="EPSG:4326")
```

---

### xarray + rioxarray

xarray provides labeled N-dimensional arrays, ideal for raster time series, climate data, and multi-band satellite imagery. rioxarray adds CRS, reprojection, and clipping operations via a `.rio` accessor.

- **Install**: `pip install xarray rioxarray`
- **Links**: [xarray.pydata.org](https://xarray.pydata.org) | [corteva.github.io/rioxarray](https://corteva.github.io/rioxarray/)

#### Quick Start

```python
import xarray as xr
import rioxarray

# Open a raster as xarray DataArray
da = rioxarray.open_rasterio("landsat_band4.tif")
print(da.dims)    # ('band', 'y', 'x')
print(da.rio.crs)
print(da.rio.transform())

# Open NetCDF/GRIB with xarray (climate model output, ERA5, etc.)
ds = xr.open_dataset("era5_temperature_2024.nc")
print(ds)         # shows all variables, coordinates, and dimensions
temp = ds["t2m"]  # select a variable → DataArray
```

#### Clipping to a Geometry

```python
import xarray as xr
import rioxarray
import geopandas as gpd

da = rioxarray.open_rasterio("sentinel2_b4.tif")
watershed = gpd.read_file("watershed.gpkg")

# Clip raster to vector geometry
clipped = da.rio.clip(
    watershed.geometry.values,
    crs=watershed.crs,
    drop=True,          # remove empty rows/columns after clip
    invert=False        # True to mask OUT the geometry
)
clipped.rio.to_raster("clipped_b4.tif")
```

#### Reprojection

```python
import rioxarray

da = rioxarray.open_rasterio("utm_data.tif")

# Reproject to WGS84
da_wgs84 = da.rio.reproject("EPSG:4326")

# Reproject to match another raster (align grids)
target = rioxarray.open_rasterio("reference_grid.tif")
da_aligned = da.rio.reproject_match(target)
```

#### Time-Series Analysis on Raster Stacks

```python
import xarray as xr
import rioxarray
import numpy as np

# Open a stack of monthly NDVI GeoTIFFs as a time series
import pandas as pd
files = sorted(glob.glob("ndvi_*.tif"))
dates = pd.date_range("2024-01-01", periods=len(files), freq="MS")

da = xr.concat(
    [rioxarray.open_rasterio(f).squeeze("band") for f in files],
    dim=pd.Index(dates, name="time")
)
print(da.dims)   # ('time', 'y', 'x')

# Temporal statistics
mean_ndvi = da.mean(dim="time")
max_ndvi  = da.max(dim="time")
trend     = da.polyfit(dim="time", deg=1)  # linear trend per pixel
```

#### TRICK: Lazy Loading Terabytes with Dask

```python
import xarray as xr
import rioxarray

# Open many NetCDF files lazily — Dask chunks data, never loads all at once
ds = xr.open_mfdataset(
    "era5_data_*.nc",
    parallel=True,          # use Dask for parallel I/O
    combine="by_coords",    # merge on coordinate values
    chunks={"time": 10, "latitude": 100, "longitude": 100}  # Dask chunk size
)

# Computation is lazy — nothing computed yet
result = ds["t2m"].mean(dim="time")  # builds a task graph

# Trigger computation
import dask.distributed
client = dask.distributed.Client()   # start local Dask cluster
computed = result.compute()          # now actually runs in parallel
```

---

### Dask-GeoPandas

Parallel/distributed spatial operations on GeoDataFrames that don't fit in memory. Dask-GeoPandas partitions a GeoDataFrame across cores or a cluster and applies GeoPandas operations to each partition.

- **Install**: `pip install dask-geopandas`
- **Links**: [dask-geopandas.readthedocs.io](https://dask-geopandas.readthedocs.io)

```python
import dask_geopandas as dask_gpd
import geopandas as gpd

# Convert a large GeoDataFrame to Dask-GeoPandas
gdf = gpd.read_file("100M_rows.parquet")  # or read in chunks
ddf = dask_gpd.from_geopandas(gdf, npartitions=8)

# Or read directly (lazy)
ddf = dask_gpd.read_parquet("100M_rows.parquet")

# Spatial partitioning by geometry (improves sjoin/overlay performance)
ddf_partitioned = ddf.spatial_shuffle(by="hilbert")

# Apply GeoPandas operations — same API, runs in parallel
buffered = ddf.buffer(100)
reprojected = ddf.to_crs(epsg=3857)
result = ddf.dissolve(by="region", aggfunc="sum")

# Spatial join in parallel
ddf_result = dask_gpd.sjoin(ddf_points, ddf_polygons, predicate="within")

# Trigger computation
gdf_result = result.compute()
```

---

### MovingPandas

Trajectory analysis for time-stamped movement data: GPS tracks, animal telemetry, vessel AIS, vehicle telematics.

- **Install**: `pip install movingpandas`
- **Links**: [movingpandas.org](https://movingpandas.org)

```python
import movingpandas as mpd
import geopandas as gpd
import pandas as pd

# Create a trajectory from a GeoDataFrame with a datetime index
gdf = gpd.read_file("gps_tracks.gpkg")
gdf["t"] = pd.to_datetime(gdf["timestamp"])
gdf = gdf.set_index("t")

tc = mpd.TrajectoryCollection(gdf, traj_id_col="vessel_id")
print(len(tc))   # number of trajectories

traj = tc.trajectories[0]
print(traj.get_length())       # total distance in CRS units
print(traj.get_duration())     # timedelta
print(traj.get_sampling_interval())

# Speed and direction
traj.add_speed()
traj.add_direction()

# Split trajectory where there are long time gaps
split = mpd.ObservationGapSplitter(traj).split(gap=pd.Timedelta("1H"))

# Stop detection
stops = mpd.TrajectoryStopDetector(traj).get_stop_segments(
    min_duration=pd.Timedelta("5min"),
    max_diameter=50  # in CRS units
)

# Simplify by removing redundant points (Douglas-Peucker)
simplified = traj.generalize(mode="douglas-peucker", tolerance=10)
```

---

### PySAL

PySAL (Python Spatial Analysis Library) is a family of packages for spatial statistics, spatial econometrics, and exploratory spatial data analysis.

- **Install**: `pip install pysal` (installs all subpackages) or individual packages
- **Links**: [pysal.org](https://pysal.org)

```python
import geopandas as gpd
import libpysal
from libpysal.weights import Queen, KNN
import esda
import splot

gdf = gpd.read_file("census_tracts.gpkg")
y = gdf["income_per_capita"].values

# Build spatial weights matrix
w = Queen.from_dataframe(gdf)   # queen contiguity (shared edge or vertex)
w.transform = "r"               # row-standardize

# Global Moran's I — is there spatial autocorrelation?
moran = esda.Moran(y, w)
print(f"Moran's I = {moran.I:.4f}, p-value = {moran.p_sim:.4f}")

# Local Moran's I — LISA clusters
lisa = esda.Moran_Local(y, w)
gdf["lisa_cluster"] = lisa.q      # 1=HH, 2=LH, 3=LL, 4=HL
gdf["lisa_pvalue"]  = lisa.p_sim

significant = gdf[gdf["lisa_pvalue"] < 0.05]

# Getis-Ord G* hotspot analysis
from esda import G_Local
g_local = G_Local(y, w, star=True)
gdf["hotspot_z"] = g_local.Zs

# Spatial weights: K-nearest neighbors instead of contiguity
wknn = KNN.from_dataframe(gdf, k=5)
```

---

## Visualization

### folium

folium creates Leaflet.js maps from Python and renders them in Jupyter notebooks or as standalone HTML files.

- **Install**: `pip install folium`
- **Links**: [python-visualization.github.io/folium](https://python-visualization.github.io/folium/)

```python
import folium
import geopandas as gpd

m = folium.Map(location=[52.52, 13.405], zoom_start=10, tiles="CartoDB positron")

# GeoJSON overlay from a GeoDataFrame
gdf = gpd.read_file("districts.gpkg").to_crs(epsg=4326)

# Choropleth map
folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    data=gdf,
    columns=["district_id", "population"],
    key_on="feature.properties.district_id",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Population"
).add_to(m)

# Marker cluster for point data
from folium.plugins import MarkerCluster, HeatMap, TimestampedGeoJson

cluster = MarkerCluster().add_to(m)
for _, row in points_gdf.iterrows():
    folium.Marker([row.geometry.y, row.geometry.x], tooltip=row["name"]).add_to(cluster)

# Heatmap
heat_data = [[row.geometry.y, row.geometry.x, row["weight"]] for _, row in points_gdf.iterrows()]
HeatMap(heat_data, radius=15).add_to(m)

m.save("map.html")
m   # displays inline in Jupyter
```

---

### leafmap

One-liner interactive maps with 40+ basemaps, Google Earth Engine integration, COG/STAC visualization, and split-map comparison. The fastest way to inspect geospatial data in Jupyter.

- **Install**: `pip install leafmap`
- **Links**: [leafmap.org](https://leafmap.org)

```python
import leafmap

# One-liner world map
m = leafmap.Map(center=[20, 0], zoom=2)
m

# Add a vector file
m.add_vector("countries.gpkg", layer_name="Countries")

# Add a COG from URL (streams tiles, no download)
cog_url = "https://opendata.example.com/sentinel2/mosaic.tif"
m.add_cog_layer(cog_url, name="Sentinel-2")

# Add STAC item
stac_url = "https://earth-search.aws.element84.com/v1"
m.add_stac_layer(
    url=stac_url,
    collection="sentinel-2-l2a",
    item="S2B_10TET_20240815_0_L2A",
    assets=["red", "green", "blue"],
    name="Sentinel-2 RGB"
)

# Split-map comparison (before/after)
m = leafmap.Map()
m.split_map(
    left_layer="TERRAIN",
    right_layer="OpenTopoMap"
)

# Add basemap
m.add_basemap("Esri.WorldImagery")

# Google Earth Engine (requires ee auth)
import ee; ee.Initialize()
m.add_ee_layer(
    ee.ImageCollection("COPERNICUS/S2_SR").first(),
    {"bands": ["B4","B3","B2"], "max": 3000},
    "Sentinel-2"
)
```

---

### pydeck

GPU-accelerated visualization using Uber's Deck.gl, rendered in Jupyter or as standalone HTML. Handles millions of features smoothly.

- **Install**: `pip install pydeck`
- **Links**: [pydeck.gl](https://pydeck.gl)

```python
import pydeck as pdk
import pandas as pd

df = pd.read_csv("trips.csv")   # columns: start_lon, start_lat, end_lon, end_lat, count

# ScatterplotLayer — millions of points at 60fps
scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    get_position=["lon", "lat"],
    get_color=[255, 0, 0, 150],
    get_radius=100,
    pickable=True
)

# ArcLayer — origin/destination flows
arc_layer = pdk.Layer(
    "ArcLayer",
    data=df,
    get_source_position=["start_lon", "start_lat"],
    get_target_position=["end_lon", "end_lat"],
    get_source_color=[0, 128, 200],
    get_target_color=[200, 0, 80],
    width_scale=2,
    width_min_pixels=2
)

# H3HexagonLayer — aggregated hex grid
h3_layer = pdk.Layer(
    "H3HexagonLayer",
    data=h3_df,
    get_hexagon="h3_index",
    get_fill_color="[255, (1-count/max_count)*255, 0]",
    get_elevation="count",
    elevation_scale=10,
    extruded=True
)

# TripLayer — animated trajectories
trip_layer = pdk.Layer(
    "TripsLayer",
    data=trips_df,
    get_path="path",           # list of [lon, lat, timestamp]
    get_color=[253, 128, 93],
    width_min_pixels=3,
    trail_length=200,
    current_time=current_timestamp
)

r = pdk.Deck(
    layers=[arc_layer],
    initial_view_state=pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=9, pitch=45),
    tooltip={"text": "{count} trips"}
)
r.to_html("output.html")
r   # inline in Jupyter
```

---

### contextily

Adds basemap tiles to existing matplotlib and GeoPandas plots. One line of code turns a coordinate-bare matplotlib figure into a map.

- **Install**: `pip install contextily`
- **Links**: [contextily.readthedocs.io](https://contextily.readthedocs.io)

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

gdf = gpd.read_file("buildings.gpkg").to_crs(epsg=3857)  # Web Mercator required for most basemaps

fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax, alpha=0.5, edgecolor="black")

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
# Other providers:
# ctx.providers.OpenStreetMap.Mapnik
# ctx.providers.Esri.WorldImagery
# ctx.providers.Stamen.TonerLite
# ctx.providers.OpenTopoMap

# Set zoom level manually (auto-detection can be slow for large areas)
ctx.add_basemap(ax, zoom=14, source=ctx.providers.CartoDB.Positron)

plt.tight_layout()
plt.savefig("map_with_basemap.png", dpi=150)
```

---

### matplotlib + cartopy

Publication-quality maps with proper cartographic projections. The standard for scientific papers and reports.

- **Install**: `pip install cartopy`
- **Links**: [scitools.org.uk/cartopy](https://scitools.org.uk/cartopy/)

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd

fig, ax = plt.subplots(
    figsize=(12, 8),
    subplot_kw={"projection": ccrs.Robinson()}   # Robinson projection
)

# Add natural earth features
ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
ax.add_feature(cfeature.LAND, facecolor="wheat")
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
ax.gridlines(draw_labels=True, dms=True)

# Plot a GeoDataFrame
gdf = gpd.read_file("country_stats.gpkg")
gdf.plot(
    ax=ax,
    column="gdp_per_capita",
    cmap="viridis",
    transform=ccrs.PlateCarree(),   # data is in WGS84
    legend=True
)

ax.set_global()
plt.title("GDP per Capita")
plt.savefig("world_map.pdf", bbox_inches="tight")
```

---

### lonboard

NEW (2023): Fast GeoArrow-based map widget for Jupyter. Renders millions of features using WebGL with native GeoArrow memory layout — no JSON serialization bottleneck.

- **Install**: `pip install lonboard`
- **Links**: [developmentseed.org/lonboard](https://developmentseed.org/lonboard/)

```python
import geopandas as gpd
from lonboard import Map, ScatterplotLayer, PathLayer, SolidPolygonLayer
from lonboard.colormap import apply_continuous_cmap
import palettable

gdf = gpd.read_file("10M_points.gpkg")   # 10 million points — no problem

# Color by attribute
normalized = (gdf["value"] - gdf["value"].min()) / (gdf["value"].max() - gdf["value"].min())
colors = apply_continuous_cmap(normalized.values, palettable.matplotlib.Viridis_256)

layer = ScatterplotLayer.from_geopandas(
    gdf,
    get_fill_color=colors,
    get_radius=50,
    radius_units="meters",
    opacity=0.8
)

m = Map(layer)
m   # renders millions of points instantly in Jupyter
```

---

## Remote Sensing in Python

### geemap

Earth Engine Python API wrapper with 500+ functions, interactive mapping, and export workflows. The fastest way to use Google Earth Engine in Python.

- **Install**: `pip install geemap`
- **Links**: [geemap.org](https://geemap.org)

```python
import ee
import geemap

ee.Initialize(project="your-project-id")

m = geemap.Map(center=[20, 0], zoom=2)

# Load and display a Sentinel-2 mosaic
s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterDate("2024-01-01", "2024-12-31")
      .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
      .median())

vis = {"bands": ["B4", "B3", "B2"], "min": 0, "max": 3000, "gamma": 1.4}
m.addLayer(s2, vis, "Sentinel-2 2024")

# NDVI time series for a point
ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
point = ee.Geometry.Point([13.405, 52.520])
ts = geemap.ee_to_df(
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate("2022-01-01", "2024-12-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
    .map(lambda img: img.normalizedDifference(["B8","B4"])
                        .set("date", img.date().format())),
    region=point, scale=10
)

# Export to GCS
geemap.ee_export_image_to_gcs(ndvi, "my-bucket", "ndvi_berlin.tif", scale=10, region=point.buffer(5000))
m
```

---

### stackstac + pystac-client

STAC-native lazy loading of cloud imagery as xarray DataArrays. The modern way to access Sentinel-2, Landsat, MODIS, etc. from cloud catalogs without downloading data.

- **Install**: `pip install stackstac pystac-client`
- **Links**: [stackstac.readthedocs.io](https://stackstac.readthedocs.io)

```python
import pystac_client
import stackstac
import numpy as np

# Search a public STAC catalog
catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")

items = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[12.0, 51.5, 15.0, 53.5],
    datetime="2024-06-01/2024-09-30",
    query={"eo:cloud_cover": {"lt": 20}}
).item_collection()

print(f"Found {len(items)} scenes")

# Stack items into a lazy xarray DataArray (Dask-backed)
stack = stackstac.stack(
    items,
    assets=["B04", "B08"],         # Red, NIR
    resolution=10,
    bounds_latlon=[12.0, 51.5, 15.0, 53.5],
    epsg=32632
)
print(stack)   # (time, band, y, x) — lazy, not yet loaded

# Compute NDVI
red = stack.sel(band="B04").astype("float32")
nir = stack.sel(band="B08").astype("float32")
ndvi = (nir - red) / (nir + red)

# Temporal median composite
ndvi_median = ndvi.median(dim="time").compute()   # triggers Dask computation
ndvi_median.rio.to_raster("ndvi_median_2024.tif")
```

---

### odc-stac

Open Data Cube's STAC loader, optimized for large-scale cloud analysis with configurable resolution, CRS, and chunking.

- **Install**: `pip install odc-stac`
- **Links**: [odc-stac.readthedocs.io](https://odc-stac.readthedocs.io)

```python
import odc.stac
import pystac_client

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
items = catalog.search(collections=["sentinel-2-l2a"], bbox=[...]).item_collection()

ds = odc.stac.load(
    items,
    bands=["red", "green", "blue", "nir"],
    resolution=20,
    crs="EPSG:4326",
    chunks={"time": 1, "y": 2048, "x": 2048},  # Dask chunks
    groupby="solar_day"    # one output timestep per calendar day
)
print(ds)   # xarray Dataset with all bands as variables
```

---

## Geocoding & Routing

### geopy

Geocoding with 20+ provider backends. Essential for address lookup and batch geocoding workflows.

- **Install**: `pip install geopy`
- **Links**: [geopy.readthedocs.io](https://geopy.readthedocs.io)

```python
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Single geocode
geolocator = Nominatim(user_agent="myapp/1.0 (myemail@example.com)")
location = geolocator.geocode("Brandenburg Gate, Berlin")
print(location.latitude, location.longitude)

# Batch geocoding with rate limiting (respect Nominatim's 1 req/sec limit)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)

addresses = pd.DataFrame({"address": ["Eiffel Tower, Paris", "Big Ben, London", "Colosseum, Rome"]})
addresses["location"] = addresses["address"].apply(geocode)
addresses["point"] = addresses["location"].apply(
    lambda loc: Point(loc.longitude, loc.latitude) if loc else None
)
gdf = gpd.GeoDataFrame(addresses, geometry="point", crs="EPSG:4326")

# Reverse geocoding
location = geolocator.reverse("52.520, 13.405")
print(location.address)

# Use Google Maps geocoder for higher quality (requires API key)
from geopy.geocoders import GoogleV3
geolocator = GoogleV3(api_key="YOUR_KEY")
```

---

### osmnx

Download, model, analyze, and visualize street networks from OpenStreetMap. Also retrieves building footprints, POIs, and any OSM features.

- **Install**: `pip install osmnx`
- **Links**: [osmnx.readthedocs.io](https://osmnx.readthedocs.io)

```python
import osmnx as ox
import networkx as nx

# Download street network
G = ox.graph_from_place("Berlin, Germany", network_type="drive")
print(G.number_of_nodes(), G.number_of_edges())

# Shortest path routing
orig = ox.nearest_nodes(G, X=13.405, Y=52.520)   # Brandenburg Gate
dest = ox.nearest_nodes(G, X=13.377, Y=52.510)   # Potsdamer Platz

route = nx.shortest_path(G, orig, dest, weight="length")
route_gdf = ox.utils_graph.route_to_gdf(G, route)
print(f"Distance: {route_gdf['length'].sum():.0f} m")

# Isochrone: reachable area within N minutes walking
G_walk = ox.graph_from_point((52.520, 13.405), dist=2000, network_type="walk")
center_node = ox.nearest_nodes(G_walk, X=13.405, Y=52.520)

# Build isochrone polygon
trip_times = [5, 10, 15]   # minutes
meters_per_min = 4.5 * 1000 / 60   # ~4.5 km/h walking speed

for trip_time in trip_times:
    subgraph = nx.ego_graph(G_walk, center_node, radius=trip_time * meters_per_min, distance="length")
    node_points = [ox.utils_graph.get_node_attributes(G_walk, n) for n in subgraph.nodes()]
    # ... create convex hull from node_points

# Download building footprints
buildings = ox.features_from_place("Berlin Mitte, Germany", tags={"building": True})
print(buildings.shape)

# TRICK: custom network analysis with any OSM tags
G_bike = ox.graph_from_polygon(your_polygon, network_type="bike")
G_bike = ox.add_edge_speeds(G_bike)
G_bike = ox.add_edge_travel_times(G_bike)
route_time = nx.shortest_path(G_bike, orig, dest, weight="travel_time")
```

---

### routingpy

Unified Python API for routing engines: OSRM, Valhalla, GraphHopper, Google, HERE, Mapbox, OpenRouteService, and more.

- **Install**: `pip install routingpy`
- **Links**: [routingpy.readthedocs.io](https://routingpy.readthedocs.io)

```python
import routingpy as rp

# OSRM (free, self-hostable)
osrm = rp.OSRM(base_url="https://router.project-osrm.org")
route = osrm.directions(
    locations=[[13.405, 52.520], [2.349, 48.853]],  # [lon, lat]
    profile="car"
)
print(f"Distance: {route.distance/1000:.1f} km, Duration: {route.duration/3600:.1f} h")
geometry = route.geometry   # list of [lon, lat] coordinate pairs

# Valhalla (free, self-hostable, supports isochrones natively)
valhalla = rp.Valhalla(base_url="https://valhalla1.openstreetmap.de")
isochrones = valhalla.isochrones(
    locations=[[13.405, 52.520]],
    profile="pedestrian",
    intervals=[300, 600, 900]  # 5, 10, 15 minutes in seconds
)
for iso in isochrones:
    print(iso.interval, iso.geometry)   # polygon for each time band

# OpenRouteService (free tier available)
ors = rp.ORS(api_key="YOUR_ORS_KEY")
matrix = ors.matrix(
    locations=[[13.405, 52.520], [2.349, 48.853], [4.899, 52.374]],
    profile="driving-car",
    metrics=["distance", "duration"]
)
print(matrix.durations)   # N×N matrix of travel times
```

---

## Specialty Libraries

### rasterstats and exactextract

Zonal statistics: summarize raster values within polygon zones.

- **Install**: `pip install rasterstats exactextract`
- **Links**: [pythonhosted.org/rasterstats](https://pythonhosted.org/rasterstats/) | [github.com/isciences/exactextract](https://github.com/isciences/exactextract)

```python
# rasterstats — pure Python, easy to use
from rasterstats import zonal_stats
import geopandas as gpd

zones = gpd.read_file("watersheds.gpkg")

stats = zonal_stats(
    zones,
    "dem.tif",
    stats=["min", "max", "mean", "median", "std", "count"],
    all_touched=False,
    nodata=-9999
)
zones["mean_elev"] = [s["mean"] for s in stats]

# exactextract — C++ backend, 10-100x faster for large datasets
import exactextract

result = exactextract.exact_extract(
    rast="dem.tif",
    vec=zones,
    ops=["mean", "stdev", "max", "count"],
    output="pandas"
)
print(result.head())
```

---

### h3-py

Uber's H3 hexagonal discrete global grid system. Efficient spatial aggregation, indexing, and neighborhood analysis.

- **Install**: `pip install h3`
- **Links**: [h3geo.org](https://h3geo.org)

```python
import h3
import geopandas as gpd
from shapely.geometry import Polygon

# Index a point to H3 cell
lat, lon = 52.520, 13.405
cell = h3.latlng_to_cell(lat, lon, res=9)   # resolution 9 ≈ 150m hexagons
print(cell)   # "891f1d48587ffff"

# Get cell boundary as polygon
boundary = h3.cell_to_boundary(cell)   # list of (lat, lon) tuples
polygon = Polygon([(lon, lat) for lat, lon in boundary])

# Ring of neighbors
neighbors = h3.grid_disk(cell, k=2)   # all cells within 2 rings

# Fill a polygon with H3 cells
polygon_coords = [[-0.5, 51.3], [0.5, 51.3], [0.5, 51.7], [-0.5, 51.7], [-0.5, 51.3]]
cells = h3.polygon_to_cells({"type":"Polygon","coordinates":[polygon_coords]}, res=8)

# Aggregate points to H3 hexagons
import pandas as pd
df = pd.read_csv("gps_points.csv")
df["h3_cell"] = df.apply(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], res=9), axis=1)
hex_counts = df.groupby("h3_cell").size().reset_index(name="count")
```

---

### WhiteboxTools Python Frontend

500+ geomorphometric, hydrological, and remote sensing tools via Python wrapper.

- **Install**: `pip install whitebox`
- **Links**: [whiteboxgeo.com](https://www.whiteboxgeo.com)

```python
import whitebox

wbt = whitebox.WhiteboxTools()
wbt.verbose = False

# Hydrological analysis workflow
wbt.breach_depressions_least_cost("dem.tif", "dem_conditioned.tif", dist=5)
wbt.d8_flow_direction("dem_conditioned.tif", "flow_dir.tif")
wbt.d8_flow_accumulation("flow_dir.tif", "flow_accum.tif")
wbt.extract_streams("flow_accum.tif", "streams.tif", threshold=1000)
wbt.watershed("flow_dir.tif", "pour_points.shp", "watersheds.tif")
wbt.raster_to_vector_polygons("watersheds.tif", "watersheds.shp")

# Terrain analysis
wbt.slope("dem.tif", "slope.tif", units="degrees")
wbt.aspect("dem.tif", "aspect.tif")
wbt.hillshade("dem.tif", "hillshade.tif", azimuth=315, altitude=30)
wbt.topographic_wetness_index("flow_accum.tif", "slope.tif", "twi.tif")
```

---

### DuckDB Spatial

DuckDB is an in-process analytical database that can replace GeoPandas for SQL-native workflows on large datasets.

- **Install**: `pip install duckdb`
- **Links**: [duckdb.org/docs/extensions/spatial](https://duckdb.org/docs/extensions/spatial.html)

```python
import duckdb

con = duckdb.connect()
con.execute("INSTALL spatial; LOAD spatial;")

# Query a GeoParquet file directly — no Python loading needed
result = con.execute("""
    SELECT
        country_name,
        ST_Area(geometry) / 1e6 AS area_km2,
        ST_Centroid(geometry) AS centroid
    FROM read_parquet('countries.parquet')
    WHERE ST_Intersects(geometry, ST_GeomFromText('POLYGON((10 45, 25 45, 25 55, 10 55, 10 45))'))
    ORDER BY area_km2 DESC
    LIMIT 10
""").df()

# Spatial join in SQL
joined = con.execute("""
    SELECT c.name, p.population
    FROM read_parquet('cities.parquet') p
    JOIN read_parquet('countries.parquet') c
      ON ST_Within(p.geometry, c.geometry)
""").df()

# Convert DuckDB result to GeoPandas
import geopandas as gpd
import shapely

df = con.execute("SELECT *, ST_AsWKB(geometry) AS wkb FROM read_parquet('data.parquet')").df()
geoms = shapely.from_wkb(df["wkb"].values)
gdf = gpd.GeoDataFrame(df.drop(columns="wkb"), geometry=geoms, crs="EPSG:4326")
```

---

### verde

Spatial data interpolation, gridding, and kriging. For converting sparse point observations into continuous raster grids.

- **Install**: `pip install verde`
- **Links**: [verde.fatiando.org](https://verde.fatiando.org)

```python
import verde as vd
import numpy as np

# Synthetic scattered temperature observations
rng = np.random.default_rng(42)
coords = (rng.uniform(-5, 5, 200), rng.uniform(-5, 5, 200))  # (lon, lat)
temperature = 25 - np.sqrt(coords[0]**2 + coords[1]**2) + rng.normal(0, 0.5, 200)

# Biharmonic spline interpolation
spline = vd.ScipyGridder(method="cubic")
spline.fit(coords, temperature)

# Predict on a regular grid
region = (-5, 5, -5, 5)   # (west, east, south, north)
grid_coords = vd.grid_coordinates(region, spacing=0.1)
predictions = spline.predict(grid_coords)

# Convert to xarray
grid = vd.make_xarray_grid(
    grid_coords, predictions, data_names=["temperature"], dims=("latitude", "longitude")
)
print(grid)
```

---

## Advanced Tricks & Dark Arts

These are the techniques that are rarely documented but solve real performance and scale problems.

---

### Memory-Mapped GeoParquet with PyArrow for Zero-Copy Reads

Standard `gpd.read_parquet()` copies data into Python memory. For read-only analytical queries, use PyArrow memory mapping to avoid the copy entirely:

```python
import pyarrow.parquet as pq
import pyarrow as pa
import geopandas as gpd
import shapely

# Memory-map the file — no data loaded into RAM yet
mmap_table = pq.read_table(
    "large_parcels.parquet",
    memory_map=True           # zero-copy reads from OS page cache
)

# Apply a filter using PyArrow compute (stays in Arrow, no Python)
import pyarrow.compute as pc
filtered_table = mmap_table.filter(
    pc.greater(pc.field("area"), 10000)
)

# Convert only the filtered subset to GeoPandas
pdf = filtered_table.to_pandas()
geoms = shapely.from_wkb(pdf["geometry"].values)
gdf = gpd.GeoDataFrame(pdf, geometry=geoms, crs="EPSG:4326")

# For repeated queries, keep the PyArrow table around and filter it —
# much cheaper than re-reading the file
```

---

### Using `multiprocessing.Pool` with Rasterio

Rasterio's thread-safety requirements mean you must create a new `rasterio.Env()` in each worker process. Forgetting this causes silent failures or crashes:

```python
import rasterio
from multiprocessing import Pool
import numpy as np

def process_tile(args):
    filepath, window = args
    # CRITICAL: each worker process needs its own rasterio.Env()
    with rasterio.Env(GDAL_CACHEMAX=256, GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(filepath) as src:
            data = src.read(window=window)
    return compute_stats(data)

def compute_stats(data):
    return {"mean": float(np.mean(data)), "std": float(np.std(data))}

if __name__ == "__main__":
    filepath = "large_raster.tif"
    with rasterio.open(filepath) as src:
        windows = [w for _, w in src.block_windows(1)]

    args = [(filepath, w) for w in windows]

    with Pool(processes=8) as pool:
        results = pool.map(process_tile, args)

    print(f"Processed {len(results)} tiles")
```

---

### Vectorized Shapely 2.0 vs. Looping: Benchmark

The performance difference between looping over geometries and using Shapely 2.0's array operations is dramatic:

```python
import shapely
import numpy as np
import geopandas as gpd
import time

gdf = gpd.read_file("1M_points.gpkg")

# SLOW: Python loop (Shapely 1.x style)
start = time.perf_counter()
buffers_slow = [geom.buffer(100) for geom in gdf.geometry]
print(f"Loop: {time.perf_counter() - start:.2f}s")    # ~45 seconds for 1M points

# FAST: Shapely 2.0 vectorized (operates on C array, no Python loop)
start = time.perf_counter()
buffers_fast = shapely.buffer(gdf.geometry.values, 100)
print(f"Vectorized: {time.perf_counter() - start:.2f}s")  # ~0.4 seconds → ~100x faster

# FAST: GeoPandas uses Shapely 2.0 internally now
start = time.perf_counter()
buffers_gpd = gdf.geometry.buffer(100)
print(f"GeoPandas: {time.perf_counter() - start:.2f}s")   # ~0.4 seconds

# Vectorized distance matrix (N points vs M reference points)
pts_a = gdf.geometry.values[:10_000]
pts_b = gdf.geometry.values[:10_000]

# STRtree for nearest-neighbor distance (not O(N²))
tree = shapely.STRtree(pts_b)
nearest_idx = tree.nearest(pts_a)
distances = shapely.distance(pts_a, pts_b[nearest_idx])
```

---

### GDAL `/vsimem/` for In-Memory File Operations

GDAL's virtual filesystem `/vsimem/` lets you create files that exist only in memory. This is critical for chaining GDAL operations without disk I/O:

```python
import rasterio
from rasterio.io import MemoryFile
import subprocess
import numpy as np

# Write a raster to GDAL's /vsimem/ (via rasterio MemoryFile)
profile = {"driver":"GTiff","dtype":"float32","width":512,"height":512,"count":1,"crs":"EPSG:4326",
           "transform": rasterio.transform.from_bounds(-180,-90,180,90,512,512)}
data = np.random.random((1,512,512)).astype("float32")

with MemoryFile() as memfile:
    with memfile.open(**profile) as dst:
        dst.write(data)
    bytes_data = memfile.read()

# Pass to another rasterio operation without touching disk
with MemoryFile(bytes_data) as memfile:
    with memfile.open() as src:
        reprojected_data = src.read()

# Use /vsimem/ path directly in GDAL command line calls
import gdal   # or osgeo.gdal
mem_path = "/vsimem/temp_raster.tif"
gdal.FileFromMemBuffer(mem_path, bytes_data)
ds = gdal.Open(mem_path)
# ... process with GDAL Python bindings
gdal.Unlink(mem_path)   # clean up
```

---

### `rio-tiler` for Custom Tile Servers from COGs in 10 Lines

```python
from rio_tiler.io import COGReader
from rio_tiler.models import ImageData

cog_url = "s3://my-bucket/large_ortho.tif"

# Serve XYZ tiles from a COG — this is the core of every Python tile server
def get_tile(z: int, x: int, y: int) -> bytes:
    with COGReader(cog_url) as cog:
        img: ImageData = cog.tile(x, y, z, tilesize=256)
        # Apply color ramp, band math, etc.
        img.rescale(in_range=[(0, 3000)])
        return img.render(img_format="PNG")

# With FastAPI:
from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

@app.get("/tiles/{z}/{x}/{y}.png")
def tile_endpoint(z: int, x: int, y: int):
    png_bytes = get_tile(z, x, y)
    return Response(content=png_bytes, media_type="image/png")

# Run: uvicorn main:app --host 0.0.0.0 --port 8000
# Add to Leaflet: L.tileLayer("http://localhost:8000/tiles/{z}/{x}/{y}.png")
```

---

### Combining `planetary_computer.sign()` with `stackstac` for Lazy Cloud Processing

Microsoft Planetary Computer hosts petabytes of satellite data. Assets require token signing before access:

```python
import planetary_computer as pc
import pystac_client
import stackstac
import numpy as np

# Authenticate with Planetary Computer
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=pc.sign_inplace   # automatically sign all asset URLs
)

# Search for Landsat Collection 2 Level-2
items = catalog.search(
    collections=["landsat-c2-l2"],
    bbox=[12.0, 51.5, 15.0, 53.5],
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 15}, "platform": {"in": ["landsat-8", "landsat-9"]}}
).item_collection()

print(f"Found {len(items)} scenes")

# Stack into lazy xarray (all URLs are pre-signed)
stack = stackstac.stack(
    items,
    assets=["red", "nir08", "qa_pixel"],
    resolution=30,
    epsg=32632,
    bounds_latlon=[12.0, 51.5, 15.0, 53.5],
    chunksize=2048
)

# Apply cloud mask from QA_PIXEL band
qa = stack.sel(band="qa_pixel")
cloud_mask = (qa & 0b0000000000001000).astype(bool)   # bit 3 = cloud

# Compute cloud-free NDVI median composite — lazy until .compute()
red = stack.sel(band="red").where(~cloud_mask).astype("float32")
nir = stack.sel(band="nir08").where(~cloud_mask).astype("float32")
ndvi = (nir - red) / (nir + red)
ndvi_median = ndvi.median(dim="time")   # still lazy

# Trigger computation (uses Dask, downloads only needed chunks)
result = ndvi_median.compute()
result.rio.to_raster("landsat_ndvi_median_summer2024.tif", driver="GTiff")
```

---

### `geopandas.sjoin_nearest()` with `max_distance` for Fuzzy Spatial Matching

The classic address-matching and record-linkage trick. Match two datasets that do not have exact geometric overlap:

```python
import geopandas as gpd

# Example: match building footprints to address points with slight coordinate offsets
buildings = gpd.read_file("buildings.gpkg").to_crs(epsg=3857)
addresses = gpd.read_file("addresses.gpkg").to_crs(epsg=3857)

# Nearest match within 25 meters — handles GPS drift, digitizing errors
matched = gpd.sjoin_nearest(
    addresses,
    buildings,
    how="left",
    max_distance=25,        # meters (since CRS is EPSG:3857)
    distance_col="match_distance_m"
)

# Rows with NaN in match_distance_m had no building within 25m
unmatched = matched[matched["match_distance_m"].isna()]
print(f"Matched: {matched['match_distance_m'].notna().sum()}, Unmatched: {len(unmatched)}")

# For de-duplication: after joining, keep only the closest match per address
matched = matched.sort_values("match_distance_m").drop_duplicates(subset="address_id", keep="first")
```

---

### Memory-Efficient Geometry Column Downcasting

```python
import geopandas as gpd
import shapely
import numpy as np

gdf = gpd.read_file("large_polygon_dataset.gpkg")

# WKB size check — identify bloated geometries
wkb_sizes = shapely.to_wkb(gdf.geometry.values)
sizes = np.array([len(w) for w in wkb_sizes])
print(f"Total geometry memory: {sizes.sum()/1e6:.1f} MB")
print(f"Top 5 largest geometries:\n{np.sort(sizes)[-5:]}")

# Simplify over-detailed geometries (e.g., from country borders with 1mm precision)
# tolerance in CRS units
simplified = shapely.simplify(gdf.geometry.values, tolerance=10, preserve_topology=True)
wkb_simplified = np.array([len(w) for w in shapely.to_wkb(simplified)])
print(f"After simplification: {wkb_simplified.sum()/1e6:.1f} MB  ({100*(1-wkb_simplified.sum()/sizes.sum()):.0f}% reduction)")

gdf.geometry = simplified
```

---

### Environment Setup Reference

```bash
# Recommended: conda for binary dependencies (GDAL, PROJ, GEOS)
conda create -n geo python=3.12
conda activate geo
conda install -c conda-forge geopandas rasterio fiona pyproj shapely

# Then pip for pure-Python libraries
pip install leafmap stackstac pystac-client planetary-computer osmnx routingpy
pip install geemap dask-geopandas movingpandas pysal
pip install exactextract rasterstats whitebox h3 verde lonboard pydeck

# Alternative: use the all-in-one spatial stack
conda install -c conda-forge --file https://raw.githubusercontent.com/opengeos/geospatial/main/requirements.txt
```

---

*Last updated: February 2026. Libraries evolve rapidly — always pin versions in production and check changelogs when upgrading.*
