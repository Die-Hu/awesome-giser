# Spatial Databases

Database engines with spatial extensions for storing, querying, and analyzing geospatial data at scale.

> **Quick Picks**
> - :trophy: **SOTA**: [DuckDB Spatial](https://duckdb.org/docs/extensions/spatial.html) --- zero-config, reads Parquet/GeoJSON/Shapefile directly, blazing-fast analytics
> - :moneybag: **Free Best**: [PostGIS 3.4+](https://postgis.net) --- the gold standard for production spatial databases with full OGC support
> - :zap: **Fastest Setup**: [DuckDB](https://duckdb.org) --- `pip install duckdb`, load spatial extension, query files in seconds with no server

## Comparison Table

| Database | Type | Spatial Extension | License | Latest Version | Best For | Max Dataset Size |
|----------|------|------------------|---------|---------------|----------|-----------------|
| PostgreSQL + PostGIS | Server-based RDBMS | PostGIS | Open Source | PostGIS 3.4 / PG 16 | Production GIS, complex queries, enterprise | Terabytes+ |
| SQLite + SpatiaLite | File-based RDBMS | SpatiaLite | Open Source | 5.1 | Lightweight, embedded, mobile | ~140 TB (theoretical) |
| GeoPackage | File-based (SQLite) | Built-in | OGC Standard | 1.4 | Data exchange, offline, field work | Gigabytes |
| DuckDB + Spatial | Embedded OLAP | duckdb_spatial | MIT | 1.1+ | Analytics, large scans, Parquet/Arrow | Memory-bound (100 GB+) |
| H3 | Spatial indexing | N/A (index system) | Apache 2.0 | 4.x | Hexagonal tiling, aggregation, analytics | N/A |
| FlatGeobuf | File format (streaming) | N/A | BSD-2 | 3.x | Web-friendly vector streaming, spatial filtering | Gigabytes |
| GeoParquet | File format (columnar) | N/A | Apache 2.0 | 1.1 | Cloud-native analytics, data lake integration | Terabytes+ |

## PostGIS

The gold standard for spatial databases. Extends PostgreSQL with geometry types, spatial indexing, and hundreds of spatial functions. PostGIS **3.4** adds performance improvements for ST_Intersects, better GIST index support, and improved GeoJSON output.

- **Current Version**: 3.4 (on PostgreSQL 16)
- **Install**: `apt install postgresql-16-postgis-3` or Docker: `docker run postgis/postgis:16-3.4`
- **Links**: [postgis.net](https://postgis.net) | [PostGIS docs](https://postgis.net/docs/)

### Key Features

- Full OGC Simple Features and SQL/MM Spatial support
- Raster data support via PostGIS Raster
- Topology support for shared geometry editing
- 3D geometry and geography types (Z and M coordinates)
- Point cloud support via pgPointCloud
- Network routing via pgRouting
- Integration with QGIS, GeoServer, MapServer, and virtually all GIS tools
- Parallel query support for spatial operations (PostgreSQL 16+)

### Quick Start

```sql
-- Enable PostGIS extension
CREATE EXTENSION postgis;

-- Create a spatial table
CREATE TABLE cities (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    population INTEGER,
    geom GEOMETRY(Point, 4326)
);

-- Create spatial index (GiST)
CREATE INDEX idx_cities_geom ON cities USING GIST (geom);

-- Insert a point
INSERT INTO cities (name, population, geom)
VALUES ('Beijing', 21540000, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326));

-- Spatial query: find cities within 50km of a point
SELECT name, population,
       ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)::geography) AS dist_m
FROM cities
WHERE ST_DWithin(
    geom::geography,
    ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)::geography,
    50000
);
```

### Advanced PostGIS Recipes

```sql
-- Spatial join: count points in polygons
SELECT p.name, COUNT(pt.id) AS num_pois
FROM polygons p
LEFT JOIN points pt ON ST_Contains(p.geom, pt.geom)
GROUP BY p.name;

-- Buffer and dissolve
SELECT ST_Union(ST_Buffer(geom::geography, 500)::geometry) AS buffer_zone
FROM fire_stations;

-- Nearest-neighbor search using KNN operator (<->)
SELECT name, geom
FROM restaurants
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)
LIMIT 5;

-- Generate H3 indexes (requires h3-pg extension)
CREATE EXTENSION h3;
SELECT h3_lat_lng_to_cell(ST_Y(geom), ST_X(geom), 9) AS h3_index
FROM cities;

-- Import GeoJSON directly (PostGIS 3.4+)
INSERT INTO my_table (name, geom)
SELECT properties->>'name',
       ST_SetSRID(ST_GeomFromGeoJSON(geometry::text), 4326)
FROM json_populate_recordset(NULL::json_row,
     (SELECT content FROM http_get('https://example.com/data.geojson'))::json->'features');
```

## SpatiaLite

Spatial extension for SQLite providing a lightweight, file-based spatial database. Ideal for embedding spatial queries in applications without running a database server.

- **Current Version**: 5.1
- **Install**: Included with most GDAL installations; or `apt install libsqlite3-mod-spatialite`
- **Links**: [gaia-gis.it/fossil/libspatialite](https://www.gaia-gis.it/fossil/libspatialite/index)

### Key Features

- No server required -- single file database
- Full OGC Simple Features support
- Compatible with QGIS, GDAL/OGR
- Ideal for embedded applications, prototyping, and mobile apps
- R-tree spatial index support

### Quick Start

```sql
-- Load SpatiaLite extension
SELECT load_extension('mod_spatialite');
SELECT InitSpatialMetaData();

-- Create spatial table
CREATE TABLE pois (id INTEGER PRIMARY KEY, name TEXT);
SELECT AddGeometryColumn('pois', 'geom', 4326, 'POINT', 'XY');
SELECT CreateSpatialIndex('pois', 'geom');

-- Insert a point
INSERT INTO pois (name, geom)
VALUES ('Park', MakePoint(116.4, 39.9, 4326));

-- Spatial query
SELECT name FROM pois
WHERE ST_Within(geom, BuildMbr(116.0, 39.5, 117.0, 40.5, 4326));
```

## GeoPackage

OGC standard file-based format built on SQLite. The modern replacement for Shapefiles, widely adopted by national mapping agencies and international organizations.

- **Spec Version**: 1.4
- **Links**: [geopackage.org](https://www.geopackage.org) | [OGC GeoPackage Standard](https://www.ogc.org/standard/geopackage/)

### Key Features

- Single file for vector, raster, and attributes
- No field name length limits (unlike Shapefile's 10-char limit)
- Supports large datasets and multiple layers
- Built-in spatial indexing (R-tree)
- Widely supported by QGIS, GDAL, ArcGIS, and most GIS tools
- Can be read/written with standard SQLite libraries

### Advantages Over Shapefile

| Feature | Shapefile | GeoPackage |
|---------|-----------|------------|
| Files per dataset | 3-12+ (.shp, .dbf, .shx, .prj, .cpg, ...) | 1 (.gpkg) |
| Field name length | 10 chars | Unlimited |
| File size limit | 2 GB | None (practical) |
| NULL geometry | No | Yes |
| Multiple geometry types | No | Yes |
| Multiple layers | No | Yes |
| Date/time fields | Limited | Full SQL types |
| Encoding | Depends on .cpg | UTF-8 always |

## DuckDB Spatial

**The game-changer for GIS analytics.** DuckDB is an in-process analytical database (like SQLite but column-oriented) with a spatial extension that lets you run SQL queries on GeoParquet, GeoJSON, Shapefiles, and FlatGeobuf files directly --- no ETL, no server, no configuration.

- **Current Version**: DuckDB 1.1+ with `duckdb_spatial` 0.11+
- **Install**: `pip install duckdb` (Python) or `brew install duckdb` (CLI) or download from [duckdb.org](https://duckdb.org)
- **Links**: [DuckDB Spatial docs](https://duckdb.org/docs/extensions/spatial.html) | [github.com/duckdb/duckdb_spatial](https://github.com/duckdb/duckdb_spatial)

### Why DuckDB Spatial Is a Game-Changer

1. **Zero configuration** -- no database server, no setup. Install, load the extension, and start querying.
2. **Reads everything directly** -- GeoParquet, GeoJSON, Shapefile, FlatGeobuf, CSV, and more. No import step.
3. **SQL-native** -- full SQL with spatial functions. If you know SQL, you know DuckDB Spatial.
4. **Blazing fast** -- columnar storage, vectorized execution, automatic parallelism. 10-100x faster than traditional tools for analytical queries.
5. **GeoParquet first-class support** -- native read/write of the emerging cloud-native geospatial format.
6. **Runs anywhere** -- Python, R, Node.js, CLI, Jupyter notebooks, or embedded in your application.
7. **Memory-efficient** -- streams data from disk; can query datasets larger than RAM.

### Quick Start

```sql
-- Install and load spatial extension
INSTALL spatial;
LOAD spatial;

-- Read GeoParquet directly (no import needed!)
SELECT * FROM read_parquet('buildings.parquet')
WHERE ST_Within(geometry, ST_GeomFromText('POLYGON((116 39, 117 39, 117 40, 116 40, 116 39))'))
LIMIT 10;

-- Read GeoJSON directly
SELECT * FROM ST_Read('cities.geojson');

-- Read Shapefile directly
SELECT * FROM ST_Read('parcels.shp');

-- Read FlatGeobuf directly
SELECT * FROM ST_Read('buildings.fgb');

-- Read remote GeoParquet from HTTP/S3
SELECT * FROM read_parquet('https://data.source.coop/vida/google-microsoft-open-buildings/geoparquet/by_country/country_iso=CHN/*.parquet');
```

### DuckDB Spatial Recipes

```sql
INSTALL spatial; LOAD spatial;

-- Spatial join: points in polygons
SELECT p.name AS district, COUNT(*) AS num_buildings
FROM ST_Read('districts.geojson') p
JOIN ST_Read('buildings.fgb') b
  ON ST_Contains(p.geom, b.geom)
GROUP BY p.name
ORDER BY num_buildings DESC;

-- Buffer analysis
SELECT name, ST_Buffer(geom, 0.01) AS buffer_geom
FROM ST_Read('schools.geojson');

-- Aggregate to H3 hexagons (with h3 extension)
INSTALL h3 FROM community; LOAD h3;
SELECT h3_latlng_to_cell(ST_Y(geom), ST_X(geom), 7) AS h3_id,
       COUNT(*) AS cnt,
       AVG(price) AS avg_price
FROM ST_Read('properties.geojson')
GROUP BY h3_id;

-- Convert between formats
COPY (SELECT * FROM ST_Read('input.shp'))
TO 'output.parquet' (FORMAT PARQUET);

-- Query Overture Maps data (GeoParquet on S3)
SELECT names.primary AS name, geometry
FROM read_parquet('s3://overturemaps-us-west-2/release/2024-09-18.0/theme=places/type=place/*')
WHERE bbox.xmin BETWEEN 116 AND 117
  AND bbox.ymin BETWEEN 39 AND 40;

-- Compute area and length
SELECT name,
       ST_Area(geom) AS area_sq_deg,
       ST_Perimeter(geom) AS perimeter_deg
FROM ST_Read('parcels.gpkg');
```

### When to Use DuckDB vs PostGIS

| Scenario | DuckDB Spatial | PostGIS |
|----------|---------------|---------|
| Ad-hoc file analysis | Best choice -- no setup | Overkill |
| GeoParquet analytics | Native, fast | Requires import |
| Multi-user production system | Not designed for this | Best choice |
| Concurrent writes | Single writer | Full MVCC |
| Complex spatial joins at scale | Excellent for batch | Excellent with indexes |
| Web application backend | Not ideal | Standard choice |
| Notebook/data science workflow | Excellent | Good (needs server) |
| Streaming/real-time data | Not designed for this | Works with LISTEN/NOTIFY |

## H3

Uber's hierarchical hexagonal spatial indexing system for efficient geospatial aggregation and analysis. H3 divides the globe into hexagonal cells at 16 resolution levels.

- **Current Version**: 4.x
- **Install**: `pip install h3` (Python), `npm install h3-js` (JavaScript)
- **Links**: [h3geo.org](https://h3geo.org) | [github.com/uber/h3](https://github.com/uber/h3)

### Key Features

- 16 resolution levels (0-15) from ~4.3 million km2 to ~0.9 m2
- Uniform hexagonal tiling avoids edge/corner distortions of square grids
- Efficient neighbor traversal and hierarchical parent/child relationships
- Bindings for Python, JavaScript, Java, Go, Rust, PostgreSQL (h3-pg), and DuckDB

### Use Cases

- Ride-sharing demand aggregation (Uber's original use case)
- Spatial joins at scale (convert both datasets to H3, then join on index)
- Point aggregation and density analysis
- Multi-resolution spatial analytics
- Visualization of large point datasets (aggregate millions of points to hex grids)

```python
# Python example
import h3

# Convert lat/lng to H3 index at resolution 9
h3_index = h3.latlng_to_cell(39.9, 116.4, 9)
print(h3_index)  # e.g., '891f0c4a4c7ffff'

# Get hexagon boundary as lat/lng pairs
boundary = h3.cell_to_boundary(h3_index)

# Get neighboring hexagons (ring of distance 1)
neighbors = h3.grid_ring(h3_index, 1)

# Get parent cell at coarser resolution
parent = h3.cell_to_parent(h3_index, 7)

# Fill a polygon with H3 cells
polygon = h3.LatLngPoly(
    [(39.8, 116.3), (40.0, 116.3), (40.0, 116.5), (39.8, 116.5)]
)
cells = h3.polygon_to_cells(polygon, 9)
print(f"Polygon contains {len(cells)} H3 cells at resolution 9")
```

```sql
-- H3 in PostgreSQL (h3-pg extension)
CREATE EXTENSION h3;
CREATE EXTENSION h3_postgis;  -- for PostGIS integration

-- Index points to H3 cells
SELECT h3_lat_lng_to_cell(geom, 9) AS h3_index,
       COUNT(*) AS point_count
FROM my_points
GROUP BY h3_index;

-- Get hex boundary as PostGIS geometry
SELECT h3_cell_to_boundary_geometry(h3_lat_lng_to_cell(geom, 7)) AS hex_geom
FROM my_points;
```

## Modern Geospatial File Formats

### GeoParquet

The emerging standard for cloud-native geospatial data. GeoParquet stores vector data in Apache Parquet format with standardized geometry encoding and spatial metadata.

- **Spec Version**: 1.1
- **Links**: [geoparquet.org](https://geoparquet.org) | [github.com/opengeospatial/geoparquet](https://github.com/opengeospatial/geoparquet)
- **Why it matters**: Columnar storage enables fast analytical queries; Parquet is the lingua franca of data lakes; GeoParquet bridges GIS and the modern data ecosystem (Spark, DuckDB, Polars, BigQuery).

```python
# Read GeoParquet with GeoPandas
import geopandas as gpd
gdf = gpd.read_parquet("buildings.parquet")

# Write GeoParquet
gdf.to_parquet("output.parquet")
```

### FlatGeobuf

A performant binary encoding for geographic data, optimized for streaming and spatial filtering via HTTP range requests.

- **Links**: [flatgeobuf.org](https://flatgeobuf.org) | [github.com/flatgeobuf/flatgeobuf](https://github.com/flatgeobuf/flatgeobuf)
- **Why it matters**: Spatial index baked into the file allows the browser to fetch only the features it needs (HTTP range requests). Excellent for web mapping with large datasets.

```bash
# Create FlatGeobuf from GeoJSON
ogr2ogr -f FlatGeobuf output.fgb input.geojson

# Read in DuckDB
SELECT * FROM ST_Read('output.fgb') WHERE ST_Intersects(geom, ST_MakeEnvelope(116, 39, 117, 40));
```
