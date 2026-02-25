# Geospatial SQL

> A definitive reference for writing spatial SQL across PostGIS, DuckDB Spatial, and SpatiaLite. Covers geometry operations, spatial joins, raster analytics, network analysis, performance tuning, and 20+ production-ready recipes for GIS professionals.

> **Quick Picks**
> - **Full-featured spatial database:** PostGIS on PostgreSQL (industry standard, 300+ spatial functions)
> - **Fast analytics without a server:** DuckDB Spatial (reads Parquet/Shapefile/GeoJSON directly, embedded)
> - **Mobile / offline / embedded:** SpatiaLite (SQLite + spatial, single-file database)
> - **Network routing:** pgRouting on PostGIS (shortest path, isochrones, TSP)
> - **Cloud-managed PostGIS:** Supabase, AWS RDS, Google Cloud SQL, Azure Database for PostgreSQL

---

## Table of Contents

- [PostGIS vs DuckDB Spatial vs SpatiaLite](#postgis-vs-duckdb-spatial-vs-spatialite)
- [PostGIS Deep Dive](#postgis-deep-dive)
- [DuckDB Spatial](#duckdb-spatial)
- [SpatiaLite](#spatialite)
- [Geometry Operations](#geometry-operations)
- [Spatial Relationships](#spatial-relationships)
- [Spatial Joins](#spatial-joins)
- [Aggregation and Analytics](#aggregation-and-analytics)
- [Raster in SQL](#raster-in-sql)
- [Network Analysis in SQL](#network-analysis-in-sql)
- [Advanced Patterns](#advanced-patterns)
- [Migration and ETL](#migration-and-etl)
- [Performance Optimization](#performance-optimization)
- [Docker and Deployment](#docker-and-deployment)
- [Cookbook](#cookbook)
- [Cross-References](#cross-references)

---

## PostGIS vs DuckDB Spatial vs SpatiaLite

Choosing the right spatial database depends on your workload, team size, and deployment constraints.

| Feature | PostGIS | DuckDB Spatial | SpatiaLite |
|---|---|---|---|
| **Engine** | PostgreSQL extension | Embedded analytical DB | SQLite extension |
| **Architecture** | Client-server | In-process / embedded | In-process / embedded |
| **Geometry model** | OGC SFS, ISO SQL/MM | GEOS-backed GEOMETRY | OGC SFS via GEOS/PROJ |
| **Spatial functions** | 300+ ST_ functions | ~80 ST_ functions | 150+ spatial functions |
| **Raster support** | PostGIS Raster (built-in) | None (use external tools) | RasterLite2 (separate) |
| **Network analysis** | pgRouting extension | None | VirtualNetwork |
| **Spatial index** | GiST, SP-GiST | R-tree (automatic) | R*-tree via SpatialIndex |
| **Max dataset size** | TB+ (production proven) | 100s of GB (RAM-dependent) | ~100 GB practical limit |
| **File format I/O** | via ogr_fdw / COPY | Native Parquet, Shapefile, GeoJSON, CSV | via VirtualShape, GPKG |
| **Cloud-native** | RDS, Cloud SQL, Supabase | WASM, httpfs for remote Parquet | N/A |
| **Best for** | Production apps, complex analytics, multi-user | Ad-hoc analytics, data pipelines, prototyping | Mobile apps, field data collection, offline GIS |
| **License** | GPL v2 | MIT | MPL tri-license |

### When to Use Which

**Choose PostGIS when:**
- You need multi-user concurrent access with ACID transactions
- Your application requires raster processing or network routing in SQL
- You are serving tiles or features via pg_tileserv / pg_featureserv
- You need the full power of PostgreSQL (CTEs, window functions, JSONB, full-text search)
- Data exceeds available RAM and you need disk-based indexing

**Choose DuckDB Spatial when:**
- You are exploring data interactively or building ETL pipelines
- Your data lives in Parquet / GeoParquet / CSV / Shapefile and you want zero-setup SQL
- You need to query remote files via HTTP without downloading them first
- You want to embed spatial SQL in a Python / R / Node.js application
- You value simplicity: no server, no configuration, no connection strings

**Choose SpatiaLite when:**
- You are building a mobile or desktop application needing an embedded spatial DB
- You need offline spatial queries (field data collection, disconnected environments)
- You want GeoPackage interoperability (GPKG is SQLite-based)
- Your dataset fits in a single file and you do not need concurrency

---

## PostGIS Deep Dive

PostGIS extends PostgreSQL with spatial types, indexes, and hundreds of functions. It is the most mature and widely deployed open-source spatial database.

### Installation

```bash
# Ubuntu / Debian
sudo apt-get install postgresql-16-postgis-3 postgresql-16-postgis-3-scripts

# macOS (Homebrew)
brew install postgis

# Fedora / RHEL
sudo dnf install postgis34_16

# From source (when you need specific GEOS/PROJ/GDAL versions)
# Requires: PostgreSQL dev headers, GEOS >= 3.11, PROJ >= 9.0, GDAL >= 3.6
wget https://download.osgeo.org/postgis/source/postgis-3.4.2.tar.gz
tar xzf postgis-3.4.2.tar.gz
cd postgis-3.4.2
./configure --with-pgconfig=/usr/bin/pg_config
make && sudo make install
```

After installation, enable the extension in your database:

```sql
-- Create extensions (run once per database)
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS postgis_raster;      -- raster support
CREATE EXTENSION IF NOT EXISTS postgis_topology;     -- topology support
CREATE EXTENSION IF NOT EXISTS postgis_sfcgal;       -- advanced 3D operations
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;        -- required by postgis_tiger_geocoder
CREATE EXTENSION IF NOT EXISTS postgis_tiger_geocoder; -- US geocoding

-- Verify installation
SELECT PostGIS_Full_Version();
-- POSTGIS="3.4.2" [EXTENSION] PGSQL="160" GEOS="3.12.1" PROJ="9.3.1" GDAL="3.8.3" ...
```

### Geometry Types

PostGIS supports two spatial type families:

| Type | Storage | Use Case |
|---|---|---|
| `geometry` | Planar (Cartesian) coordinates | Projected data, local analysis, measurements in meters |
| `geography` | Geodetic (lon/lat on spheroid) | Global data, great-circle distances, cross-meridian queries |

```sql
-- Creating tables with geometry columns
CREATE TABLE cities (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    population INTEGER,
    geom geometry(Point, 4326)       -- WGS84 point
);

CREATE TABLE parcels (
    id SERIAL PRIMARY KEY,
    parcel_id TEXT UNIQUE,
    area_sqm DOUBLE PRECISION,
    geom geometry(MultiPolygon, 32650) -- UTM Zone 50N multipolygon
);

CREATE TABLE roads (
    id SERIAL PRIMARY KEY,
    road_name TEXT,
    road_class TEXT,
    geom geometry(LineString, 4326)
);

-- Geography type for global distance calculations
CREATE TABLE airports (
    id SERIAL PRIMARY KEY,
    iata_code CHAR(3),
    name TEXT,
    geog geography(Point, 4326)
);

-- Insert with WKT
INSERT INTO cities (name, population, geom)
VALUES ('Beijing', 21540000, ST_GeomFromText('POINT(116.4074 39.9042)', 4326));

-- Insert with GeoJSON
INSERT INTO cities (name, population, geom)
VALUES ('Shanghai', 24870000,
    ST_GeomFromGeoJSON('{"type":"Point","coordinates":[121.4737,31.2304]}')
);

-- Insert with WKB (common in programmatic access)
INSERT INTO cities (name, population, geom)
VALUES ('Tokyo', 13960000,
    ST_GeomFromWKB('\x0101000000...', 4326)
);

-- Insert geography (note: always lon/lat order)
INSERT INTO airports (iata_code, name, geog)
VALUES ('PEK', 'Beijing Capital International',
    ST_GeogFromText('POINT(116.5844 40.0799)')
);
```

### Supported Geometry Sub-Types

```sql
-- All OGC geometry sub-types
geometry(Point, 4326)
geometry(LineString, 4326)
geometry(Polygon, 4326)
geometry(MultiPoint, 4326)
geometry(MultiLineString, 4326)
geometry(MultiPolygon, 4326)
geometry(GeometryCollection, 4326)

-- 3D variants (Z coordinate)
geometry(PointZ, 4326)
geometry(LineStringZ, 4326)
geometry(PolygonZ, 4326)

-- M variants (measure coordinate, e.g., for linear referencing)
geometry(PointM, 4326)
geometry(LineStringM, 4326)

-- 3DM (Z + M)
geometry(PointZM, 4326)

-- Curved geometries (ISO SQL/MM)
geometry(CircularString, 4326)
geometry(CompoundCurve, 4326)
geometry(CurvePolygon, 4326)

-- Triangle and TIN (for 3D surfaces)
geometry(Triangle, 4326)
geometry(TIN, 4326)
```

### SRID Management

The Spatial Reference Identifier (SRID) ties a geometry to a coordinate reference system. PostGIS ships with the `spatial_ref_sys` table containing thousands of CRS definitions.

```sql
-- Look up an SRID
SELECT srid, auth_name, auth_srid, srtext
FROM spatial_ref_sys
WHERE auth_name = 'EPSG' AND auth_srid = 4326;

-- Search by name
SELECT srid, auth_name || ':' || auth_srid AS code, srtext
FROM spatial_ref_sys
WHERE srtext ILIKE '%UTM zone 50N%' AND auth_name = 'EPSG';

-- Add a custom CRS (e.g., a local projection)
INSERT INTO spatial_ref_sys (srid, auth_name, auth_srid, srtext, proj4text)
VALUES (
    900001,
    'CUSTOM', 900001,
    'PROJCS["My Local Grid", ...]',
    '+proj=tmerc +lat_0=0 +lon_0=117 +k=1 +x_0=500000 +y_0=0 +datum=WGS84 +units=m'
);

-- Transform geometry between SRIDs
SELECT ST_Transform(geom, 32650) FROM cities;

-- Reproject an entire table
ALTER TABLE cities
    ALTER COLUMN geom TYPE geometry(Point, 32650)
    USING ST_Transform(geom, 32650);

-- Check SRID of a geometry
SELECT ST_SRID(geom) FROM cities LIMIT 1;

-- Set SRID without transforming coordinates (metadata only)
SELECT ST_SetSRID(geom, 4326) FROM cities;

-- Common SRID reference:
-- 4326  = WGS 84 (lon/lat, GPS coordinates)
-- 3857  = Web Mercator (web mapping tiles)
-- 32601-32660 = WGS 84 / UTM zones 1N-60N
-- 32701-32760 = WGS 84 / UTM zones 1S-60S
-- 4490  = CGCS2000 (China Geodetic Coordinate System 2000)
-- 4547  = CGCS2000 / 3-degree Gauss-Kruger CM 117E
-- 2154  = RGF93 / Lambert-93 (France)
-- 27700 = OSGB 1936 / British National Grid
-- 2193  = NZGD2000 / New Zealand Transverse Mercator
```

### Spatial Indexing

Spatial indexes are essential for performant queries. PostGIS supports GiST (Generalized Search Tree) and SP-GiST (Space-Partitioned GiST).

```sql
-- GiST index (default, recommended for most use cases)
CREATE INDEX idx_cities_geom ON cities USING GIST (geom);

-- GiST index on geography columns
CREATE INDEX idx_airports_geog ON airports USING GIST (geog);

-- SP-GiST index (better for skewed/clustered point data)
CREATE INDEX idx_cities_geom_spgist ON cities USING SPGIST (geom);

-- BRIN index (Block Range Index, for very large sorted/clustered datasets)
CREATE INDEX idx_cities_geom_brin ON cities USING BRIN (geom)
    WITH (pages_per_range = 128);

-- Partial index (index only a subset of rows)
CREATE INDEX idx_large_parcels ON parcels USING GIST (geom)
    WHERE area_sqm > 10000;

-- Multi-column index (spatial + attribute)
CREATE INDEX idx_roads_class_geom ON roads USING GIST (geom)
    WHERE road_class IN ('motorway', 'trunk', 'primary');

-- Index maintenance
REINDEX INDEX idx_cities_geom;
VACUUM ANALYZE cities;

-- Check index usage
EXPLAIN (ANALYZE, BUFFERS) SELECT *
FROM cities
WHERE ST_Intersects(geom, ST_MakeEnvelope(116, 39, 117, 41, 4326));
```

**GiST vs SP-GiST vs BRIN:**

| Index Type | Build Speed | Query Speed | Size | Best For |
|---|---|---|---|---|
| GiST | Medium | Fast | Medium | General spatial queries, overlapping geometries |
| SP-GiST | Fast | Fast (point data) | Small | Non-overlapping point data, quad-tree partitioning |
| BRIN | Very fast | Moderate | Very small | Huge tables with spatially clustered/sorted data |

### ST_ Functions Reference

PostGIS provides 300+ functions. Here are the most important ones grouped by category.

#### Constructors

```sql
-- From WKT
ST_GeomFromText('POINT(116.4 39.9)', 4326)
ST_GeomFromText('LINESTRING(0 0, 1 1, 2 2)', 4326)
ST_GeomFromText('POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))', 4326)

-- Shorthand constructors
ST_Point(116.4, 39.9)                        -- no SRID
ST_SetSRID(ST_Point(116.4, 39.9), 4326)      -- with SRID
ST_MakePoint(116.4, 39.9)                    -- 2D point
ST_MakePoint(116.4, 39.9, 100.0)             -- 3D point (with Z)
ST_MakeLine(geom1, geom2)                    -- line from two points
ST_MakePolygon(ST_GeomFromText('LINESTRING(0 0, 1 0, 1 1, 0 1, 0 0)'))
ST_MakeEnvelope(xmin, ymin, xmax, ymax, 4326)  -- bounding box polygon
ST_MakeValid(geom)                           -- fix invalid geometries

-- From GeoJSON
ST_GeomFromGeoJSON('{"type":"Point","coordinates":[116.4,39.9]}')

-- From WKB / EWKB
ST_GeomFromWKB(wkb_bytea, 4326)
ST_GeomFromEWKB(ewkb_bytea)

-- From Encoded Polyline (Google format)
ST_LineFromEncodedPolyline('_p~iF~ps|U_ulLnnqC_mqNvxq`@')

-- Generating geometries
ST_GeneratePoints(polygon_geom, 100)          -- 100 random points in polygon
ST_OffsetCurve(line_geom, 10.0)              -- parallel offset line
ST_Reverse(line_geom)                         -- reverse vertex order
```

#### Measurement Functions

```sql
-- Length and distance
ST_Length(line_geom)               -- length in CRS units (use projected CRS for meters)
ST_Length(geog_line)               -- length in meters (geography type)
ST_3DLength(lineZ_geom)           -- 3D length including elevation changes
ST_Perimeter(polygon_geom)        -- polygon perimeter
ST_Distance(geom1, geom2)         -- minimum distance between geometries
ST_3DDistance(geom1, geom2)       -- 3D distance
ST_MaxDistance(geom1, geom2)      -- maximum distance
ST_HausdorffDistance(g1, g2)      -- Hausdorff distance (shape similarity)
ST_FrechetDistance(g1, g2)        -- Frechet distance (curve similarity)

-- Area
ST_Area(polygon_geom)             -- area in CRS units squared
ST_Area(geog_polygon)             -- area in square meters (geography)

-- Angles and azimuth
ST_Azimuth(point1, point2)        -- angle from north in radians
ST_Angle(p1, p2, p3)              -- angle at p2 between p1-p2-p3

-- Centroid and center
ST_Centroid(geom)                 -- geometric centroid (may be outside polygon)
ST_PointOnSurface(geom)           -- guaranteed point inside the geometry
ST_GeometricMedian(multipoint)    -- geometric median (minimizes total distance)

-- Bounding box
ST_Envelope(geom)                 -- bounding box as polygon
ST_Expand(geom, 100)              -- expand bounding box by 100 units
Box2D(geom)                       -- 2D bounding box
Box3D(geom)                       -- 3D bounding box

-- Coordinate extraction
ST_X(point_geom)                  -- X coordinate
ST_Y(point_geom)                  -- Y coordinate
ST_Z(point_geom)                  -- Z coordinate
ST_M(point_geom)                  -- M coordinate
ST_XMin(geom), ST_XMax(geom)     -- bounding box extents
ST_NPoints(geom)                  -- number of vertices
ST_NRings(polygon_geom)          -- number of rings
ST_NumGeometries(multi_geom)     -- number of sub-geometries
ST_GeometryN(multi_geom, 1)     -- extract Nth sub-geometry (1-indexed)
```

#### Relationship Functions (Predicates)

```sql
-- Topological relationships (return boolean)
ST_Intersects(geom1, geom2)       -- geometries share any space
ST_Disjoint(geom1, geom2)         -- geometries share no space
ST_Contains(geom1, geom2)         -- geom1 completely contains geom2
ST_Within(geom1, geom2)           -- geom1 is completely within geom2
ST_Covers(geom1, geom2)           -- like Contains but handles boundary
ST_CoveredBy(geom1, geom2)       -- like Within but handles boundary
ST_Touches(geom1, geom2)          -- geometries touch at boundary only
ST_Crosses(geom1, geom2)          -- geometries cross (line/polygon)
ST_Overlaps(geom1, geom2)         -- geometries overlap (partial)
ST_Equals(geom1, geom2)           -- geometries are topologically equal

-- Distance-based
ST_DWithin(geom1, geom2, 1000)    -- within 1000 CRS units
ST_DWithin(geog1, geog2, 1000)    -- within 1000 meters (geography)
ST_DFullyWithin(g1, g2, 100)      -- every point within distance

-- DE-9IM (Dimensionally Extended 9-Intersection Model)
ST_Relate(geom1, geom2)           -- returns DE-9IM matrix string
ST_Relate(geom1, geom2, 'T*F**FFF*')  -- test against pattern
-- The 9 chars represent: II IB IE BI BB BE EI EB EE
-- (I=Interior, B=Boundary, E=Exterior)
-- Values: T(intersects), F(no intersection), 0(point), 1(line), 2(area), *(any)

-- Named spatial relationships as DE-9IM patterns:
-- Contains:    'T*****FF*'
-- Within:      'T*F**F***'
-- Touches:     'FT*******' OR 'F**T*****' OR 'F***T****'
-- Crosses:     'T*T******' (line/line)
-- Overlaps:    'T*T***T**' (polygon/polygon)
```

#### Processing Functions

```sql
-- Buffer
ST_Buffer(geom, 100)                          -- 100 unit buffer
ST_Buffer(geom, 100, 'quad_segs=8')           -- more segments = smoother
ST_Buffer(geom, 100, 'endcap=flat join=mitre') -- flat endcaps, mitre joins
ST_Buffer(geog, 1000)                          -- 1000 meter buffer (geography)

-- Set operations
ST_Intersection(geom1, geom2)                  -- shared area
ST_Union(geom1, geom2)                         -- combined area
ST_Difference(geom1, geom2)                    -- geom1 minus geom2
ST_SymDifference(geom1, geom2)                -- XOR (either but not both)

-- Simplification
ST_Simplify(geom, tolerance)                   -- Douglas-Peucker
ST_SimplifyPreserveTopology(geom, tolerance)   -- topology-safe
ST_SimplifyVW(geom, tolerance)                 -- Visvalingam-Whyatt
ST_ChaikinSmoothing(geom, 3)                   -- smooth line/polygon

-- Convex hull and concave hull
ST_ConvexHull(geom)                            -- convex hull
ST_ConcaveHull(geom, 0.3)                      -- concave hull (target ratio)

-- Voronoi and Delaunay
ST_VoronoiPolygons(multipoint_geom)            -- Voronoi diagram
ST_VoronoiLines(multipoint_geom)               -- Voronoi edges
ST_DelaunayTriangles(multipoint_geom)           -- Delaunay triangulation

-- Line operations
ST_LineMerge(multilinestring)                   -- merge connected lines
ST_LineSubstring(line, 0.0, 0.5)                -- first half of line
ST_LineInterpolatePoint(line, 0.5)              -- point at 50% of line
ST_LineLocatePoint(line, point)                 -- fraction along line
ST_LocateAlong(lineM, measure_value)            -- point at M value
ST_AddMeasure(line, 0, 100)                     -- add M values 0-100
ST_Split(geom, blade_geom)                      -- split geometry with blade
ST_Snap(geom1, geom2, tolerance)                -- snap vertices
ST_SharedPaths(line1, line2)                     -- find shared segments

-- Polygon operations
ST_Subdivide(geom, 256)                         -- subdivide into max 256 vertices
ST_ClipByBox2D(geom, box2d)                     -- fast bbox clip
ST_BuildArea(linestring_geom)                   -- build polygon from closed line
ST_Polygonize(geom_array)                       -- build polygons from lines
ST_Node(linestring_geom)                        -- node a linestring

-- Affine transformations
ST_Translate(geom, dx, dy)                      -- move geometry
ST_Scale(geom, sx, sy)                          -- scale geometry
ST_Rotate(geom, radians)                        -- rotate around centroid
ST_Rotate(geom, radians, ST_Centroid(geom))     -- rotate around point
ST_TransScale(geom, dx, dy, sx, sy)             -- translate + scale
ST_Affine(geom, a, b, c, d, e, f, g, h, i, x, y, z) -- general affine
```

### PostGIS Performance Tuning

```sql
-- PostgreSQL configuration for spatial workloads (postgresql.conf)
-- Increase shared_buffers for caching spatial index pages
-- shared_buffers = '4GB'              -- 25% of system RAM
-- effective_cache_size = '12GB'       -- 75% of system RAM
-- work_mem = '256MB'                  -- for complex spatial operations
-- maintenance_work_mem = '1GB'        -- for index creation, VACUUM
-- max_parallel_workers_per_gather = 4 -- parallel query execution
-- random_page_cost = 1.1              -- if using SSD
-- jit = off                           -- JIT can hurt spatial queries

-- Table statistics for spatial columns
ALTER TABLE parcels ALTER COLUMN geom SET STATISTICS 1000;
ANALYZE parcels;

-- CLUSTER table on spatial index (physically reorders rows)
CLUSTER parcels USING idx_parcels_geom;

-- TOAST settings for large geometries
ALTER TABLE parcels ALTER COLUMN geom SET STORAGE MAIN;

-- Check table and index sizes
SELECT
    pg_size_pretty(pg_total_relation_size('parcels')) AS total_size,
    pg_size_pretty(pg_relation_size('parcels')) AS table_size,
    pg_size_pretty(pg_indexes_size('parcels')) AS index_size;
```

---

## DuckDB Spatial

DuckDB is an embedded analytical database that can query spatial data directly from files without importing. The `spatial` extension adds GEOMETRY types and ST_ functions powered by GEOS.

### Extension Setup

```sql
-- Install and load the spatial extension
INSTALL spatial;
LOAD spatial;

-- Also install httpfs for remote file access
INSTALL httpfs;
LOAD httpfs;

-- Install H3 for hexagonal aggregation
INSTALL h3;
LOAD h3;

-- Verify
SELECT ST_Point(116.4, 39.9);
```

### Reading Spatial File Formats

DuckDB Spatial can read virtually any geospatial format via its GDAL integration.

```sql
-- Read Shapefile
SELECT * FROM ST_Read('buildings.shp');

-- Read GeoJSON
SELECT * FROM ST_Read('parcels.geojson');

-- Read GeoPackage (specific layer)
SELECT * FROM ST_Read('data.gpkg', layer='buildings');

-- Read GeoParquet (native, extremely fast)
SELECT * FROM read_parquet('buildings.parquet');
-- or with spatial filter pushdown:
SELECT * FROM read_parquet('buildings.parquet')
WHERE ST_Intersects(geometry, ST_MakeEnvelope(116, 39, 117, 41));

-- Read FlatGeobuf
SELECT * FROM ST_Read('data.fgb');

-- Read KML
SELECT * FROM ST_Read('data.kml');

-- Read CSV with coordinates and create geometry
SELECT *, ST_Point(longitude, latitude) AS geom
FROM read_csv('stations.csv');

-- Read multiple files with glob
SELECT * FROM ST_Read('tiles/*.shp');

-- Read from URL via httpfs
SELECT * FROM read_parquet('https://data.example.com/buildings.parquet');
SELECT * FROM read_parquet('s3://my-bucket/geo/parcels.parquet');

-- Configure S3 credentials
SET s3_region = 'us-east-1';
SET s3_access_key_id = 'AKIA...';
SET s3_secret_access_key = '...';
SELECT * FROM read_parquet('s3://my-bucket/data.parquet');

-- Write spatial data out
COPY (SELECT * FROM my_table)
TO 'output.parquet' (FORMAT PARQUET);

COPY (SELECT * FROM my_table)
TO 'output.geojson'
WITH (FORMAT GDAL, DRIVER 'GeoJSON');

COPY (SELECT * FROM my_table)
TO 'output.shp'
WITH (FORMAT GDAL, DRIVER 'ESRI Shapefile');

COPY (SELECT * FROM my_table)
TO 'output.gpkg'
WITH (FORMAT GDAL, DRIVER 'GPKG');
```

### Spatial SQL in DuckDB

```sql
-- Create table with geometry
CREATE TABLE pois AS
SELECT
    name,
    category,
    ST_Point(lon, lat) AS geom
FROM read_csv('pois.csv');

-- Spatial filter
SELECT name, category
FROM pois
WHERE ST_Within(
    geom,
    ST_GeomFromText('POLYGON((116 39, 117 39, 117 41, 116 41, 116 39))')
);

-- Buffer and intersect
SELECT
    a.name AS poi_name,
    b.name AS park_name
FROM pois a, ST_Read('parks.shp') b
WHERE ST_Intersects(
    ST_Buffer(a.geom, 0.01),
    b.geom
);

-- Distance calculation
SELECT
    a.name,
    b.name AS nearest_station,
    ST_Distance(a.geom, b.geom) AS dist
FROM pois a, ST_Read('stations.shp') b
ORDER BY a.name, dist
LIMIT 10;

-- Aggregation
SELECT
    category,
    COUNT(*) AS count,
    ST_Extent(geom) AS bbox
FROM pois
GROUP BY category;
```

### H3 Hexagonal Extension

```sql
LOAD h3;
LOAD spatial;

-- Convert point to H3 cell index at resolution 9
SELECT
    h3_latlng_to_cell(ST_Y(geom), ST_X(geom), 9) AS h3_index,
    COUNT(*) AS count
FROM pois
GROUP BY 1
ORDER BY count DESC;

-- Get H3 cell boundary as geometry
SELECT
    h3_index,
    count,
    h3_cell_to_boundary_wkt(h3_index) AS hex_boundary
FROM (
    SELECT
        h3_latlng_to_cell(ST_Y(geom), ST_X(geom), 9) AS h3_index,
        COUNT(*) AS count
    FROM pois
    GROUP BY 1
);

-- H3 k-ring neighborhood
SELECT h3_grid_disk(h3_latlng_to_cell(39.9, 116.4, 9), 2) AS neighbors;

-- Compact / uncompact for multi-resolution
SELECT h3_compact(ARRAY_AGG(h3_index)) FROM h3_cells;
```

### DuckDB WASM for Browser

DuckDB compiles to WebAssembly, enabling spatial SQL entirely in the browser.

```javascript
// In a web application
import * as duckdb from '@duckdb/duckdb-wasm';

const db = await duckdb.AsyncDuckDB.create();
await db.open({});
const conn = await db.connect();

// Load spatial extension (WASM build)
await conn.query("INSTALL spatial; LOAD spatial;");

// Query GeoJSON fetched via HTTP
await conn.query(`
    CREATE TABLE parks AS
    SELECT * FROM ST_Read('https://example.com/parks.geojson');
`);

const result = await conn.query(`
    SELECT name, ST_Area(geom) AS area
    FROM parks
    ORDER BY area DESC
    LIMIT 10;
`);
```

---

## SpatiaLite

SpatiaLite extends SQLite with spatial capabilities. It is ideal for embedded, mobile, and offline scenarios.

### Setup and Basic Usage

```bash
# Ubuntu / Debian
sudo apt-get install spatialite-bin libsqlite3-mod-spatialite

# macOS
brew install spatialite-tools

# Open a SpatiaLite database
spatialite my_database.sqlite
```

```sql
-- Initialize spatial metadata tables
SELECT InitSpatialMetadata(1);

-- Create a spatial table
CREATE TABLE buildings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    building_type TEXT
);
SELECT AddGeometryColumn('buildings', 'geom', 4326, 'POLYGON', 'XY');
SELECT CreateSpatialIndex('buildings', 'geom');

-- Insert data
INSERT INTO buildings (name, building_type, geom)
VALUES ('City Hall', 'government',
    GeomFromText('POLYGON((116.39 39.91, 116.40 39.91, 116.40 39.92, 116.39 39.92, 116.39 39.91))', 4326));

-- Spatial query
SELECT name, building_type
FROM buildings
WHERE ST_Intersects(geom,
    BuildMbr(116.38, 39.90, 116.41, 39.93, 4326))
AND ROWID IN (
    SELECT ROWID FROM SpatialIndex
    WHERE f_table_name = 'buildings'
    AND f_geometry_column = 'geom'
    AND search_frame = BuildMbr(116.38, 39.90, 116.41, 39.93, 4326)
);
```

### GeoPackage Interoperability

GeoPackage (.gpkg) is built on SQLite, so SpatiaLite can read and write GPKG files directly.

```sql
-- Open a GeoPackage as a SpatiaLite database
-- (from command line)
-- spatialite my_data.gpkg

-- Query GeoPackage tables
SELECT table_name, data_type, srs_id
FROM gpkg_contents;

-- Read features
SELECT fid, name, geom
FROM my_layer
WHERE ST_Intersects(geom, BuildMbr(116, 39, 117, 41));

-- Export SpatiaLite table to GeoPackage
SELECT ExportGeoPackage('buildings', 'geom', 'output.gpkg', 'buildings_layer');
```

### Mobile and Offline Use Cases

SpatiaLite is commonly used in mobile GIS applications:

```
# Android: use org.spatialite:spatialite-android
# iOS: compile libspatialite as a static library
# Qt: load mod_spatialite as SQLite extension
# Python: sqlite3 module + load_extension('mod_spatialite')
```

```python
# Python example: SpatiaLite in an offline field app
import sqlite3

conn = sqlite3.connect('field_data.sqlite')
conn.enable_load_extension(True)
conn.load_extension('mod_spatialite')
conn.execute("SELECT InitSpatialMetadata(1)")

# Create collection table
conn.execute("""
    CREATE TABLE IF NOT EXISTS field_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        observer TEXT,
        observation_time DATETIME DEFAULT CURRENT_TIMESTAMP,
        notes TEXT
    )
""")
conn.execute("""
    SELECT AddGeometryColumn('field_observations', 'geom', 4326, 'POINT', 'XY')
""")
conn.execute("""
    SELECT CreateSpatialIndex('field_observations', 'geom')
""")

# Insert observation from GPS
lon, lat = 116.4074, 39.9042
conn.execute("""
    INSERT INTO field_observations (observer, notes, geom)
    VALUES (?, ?, MakePoint(?, ?, 4326))
""", ('Zhang Wei', 'Erosion observed at riverbank', lon, lat))

conn.commit()
```

---

## Geometry Operations

Detailed reference for the core geometry manipulation operations available across all three databases.

### Buffer Operations

```sql
-- PostGIS: Buffer with various styles
-- Round buffer (default)
SELECT ST_Buffer(geom, 100) FROM parcels;

-- Flat end caps on line buffers
SELECT ST_Buffer(geom, 50, 'endcap=flat') FROM roads;

-- Square end caps
SELECT ST_Buffer(geom, 50, 'endcap=square') FROM roads;

-- Mitre join (sharp corners)
SELECT ST_Buffer(geom, 100, 'join=mitre mitre_limit=2.0') FROM parcels;

-- Single-sided buffer (one side of a line)
SELECT ST_Buffer(geom, 50, 'side=left') FROM roads;
SELECT ST_Buffer(geom, 50, 'side=right') FROM roads;

-- Variable-width buffer (PostGIS 3.2+)
-- No built-in function, but achievable with segment-by-segment approach:
WITH segments AS (
    SELECT
        id,
        n,
        ST_MakeLine(
            ST_PointN(geom, n),
            ST_PointN(geom, n + 1)
        ) AS seg,
        -- Width varies based on position along line
        10 + 40 * (n::float / ST_NPoints(geom)) AS width
    FROM roads, generate_series(1, ST_NPoints(geom) - 1) AS n
)
SELECT id, ST_Union(ST_Buffer(seg, width, 'endcap=flat')) AS variable_buffer
FROM segments
GROUP BY id;

-- Negative buffer (shrink polygon)
SELECT ST_Buffer(geom, -10) FROM parcels
WHERE ST_Area(geom) > 400;  -- avoid collapsing small polygons

-- DuckDB Spatial: Buffer
SELECT ST_Buffer(geom, 100) FROM parcels;

-- SpatiaLite: Buffer
SELECT Buffer(geom, 100) FROM parcels;
```

### Set Operations (Intersection, Union, Difference)

```sql
-- Intersection: find the shared area between two geometries
SELECT
    a.id AS parcel_id,
    b.id AS flood_zone_id,
    ST_Intersection(a.geom, b.geom) AS intersection_geom,
    ST_Area(ST_Intersection(a.geom, b.geom)) AS overlap_area,
    ST_Area(ST_Intersection(a.geom, b.geom)) / ST_Area(a.geom) * 100 AS pct_in_flood
FROM parcels a
JOIN flood_zones b ON ST_Intersects(a.geom, b.geom);

-- Union: merge two geometries into one
SELECT ST_Union(a.geom, b.geom) AS merged
FROM neighborhoods a, neighborhoods b
WHERE a.name = 'Dongcheng' AND b.name = 'Xicheng';

-- Difference: subtract one geometry from another
SELECT ST_Difference(a.geom, b.geom) AS remaining
FROM parcels a, easements b
WHERE ST_Intersects(a.geom, b.geom) AND a.id = 42;

-- Symmetric difference: area in either but not both
SELECT ST_SymDifference(a.geom, b.geom) AS sym_diff
FROM zoning_2020 a, zoning_2025 b
WHERE a.zone_id = b.zone_id;

-- Cascaded union of all geometries in a table
SELECT ST_Union(geom) AS dissolved
FROM provinces
WHERE country = 'China';

-- Subdivide large geometries for faster processing
-- (splits polygons with > max_vertices into smaller pieces)
SELECT id, ST_Subdivide(geom, 256) AS sub_geom
FROM large_polygons;

-- Then use the subdivided version for faster spatial joins:
CREATE TABLE parcels_subdivided AS
SELECT id, ST_Subdivide(geom, 256) AS geom FROM parcels;
CREATE INDEX ON parcels_subdivided USING GIST (geom);
```

### Convex Hull and Concave Hull

```sql
-- Convex hull of a set of points
SELECT ST_ConvexHull(ST_Collect(geom)) AS hull
FROM gps_tracks
WHERE trip_id = 42;

-- Concave hull (tighter boundary, PostGIS 3.3+)
-- target_pct: 1.0 = convex hull, 0.0 = tightest possible
SELECT ST_ConcaveHull(ST_Collect(geom), 0.3) AS concave
FROM gps_tracks
WHERE trip_id = 42;

-- Alpha shape (via SFCGAL extension)
SELECT ST_AlphaShape(ST_Collect(geom), 100.0) AS alpha
FROM sample_points;

-- Optimal alpha shape
SELECT ST_OptimalAlphaShape(ST_Collect(geom)) AS optimal_alpha
FROM sample_points;
```

### Voronoi and Delaunay

```sql
-- Voronoi polygons from a set of points
SELECT
    (ST_Dump(
        ST_VoronoiPolygons(ST_Collect(geom))
    )).geom AS voronoi_cell
FROM weather_stations;

-- Voronoi clipped to a boundary
WITH stations AS (
    SELECT ST_Collect(geom) AS points FROM weather_stations
),
boundary AS (
    SELECT ST_Union(geom) AS outline FROM provinces WHERE name = 'Guangdong'
)
SELECT
    (ST_Dump(
        ST_Intersection(
            ST_VoronoiPolygons(stations.points, 0, boundary.outline),
            boundary.outline
        )
    )).geom AS clipped_voronoi
FROM stations, boundary;

-- Delaunay triangulation
SELECT
    (ST_Dump(
        ST_DelaunayTriangles(ST_Collect(geom))
    )).geom AS triangle
FROM elevation_points;

-- Delaunay as edges (lines) rather than triangles
SELECT
    (ST_Dump(
        ST_DelaunayTriangles(ST_Collect(geom), 0, 1)  -- flags=1 returns edges
    )).geom AS edge
FROM elevation_points;
```

### Precision Handling

```sql
-- Snap coordinates to a grid to avoid floating-point artifacts
SELECT ST_SnapToGrid(geom, 0.001) FROM parcels;   -- 0.001 degree grid
SELECT ST_SnapToGrid(geom, 0.01, 0.01, 0, 0) FROM parcels;  -- custom origin

-- Reduce coordinate precision
SELECT ST_QuantizeCoordinates(geom, 6) FROM parcels;  -- 6 decimal places

-- Fix invalid geometries
SELECT ST_MakeValid(geom) FROM parcels WHERE NOT ST_IsValid(geom);

-- Check validity with reason
SELECT id, ST_IsValidReason(geom)
FROM parcels
WHERE NOT ST_IsValid(geom);

-- Validate and report detail
SELECT id, (ST_IsValidDetail(geom)).*
FROM parcels
WHERE NOT ST_IsValid(geom);

-- Set precision model (PostGIS 3.3+, uses fixed-precision overlay via GEOS)
SELECT ST_ReducePrecision(geom, 0.001) FROM parcels;
```

---

## Spatial Relationships

Understanding spatial relationships is fundamental to spatial SQL. This section covers predicates, the DE-9IM model, and practical usage patterns.

### Topological Predicates

```sql
-- ST_Intersects: do geometries share any space?
-- This is the most commonly used predicate.
SELECT a.name, b.name
FROM buildings a, flood_zones b
WHERE ST_Intersects(a.geom, b.geom);

-- ST_Contains: does A completely contain B?
-- (B must be entirely inside A's interior)
SELECT d.name AS district, COUNT(s.id) AS num_schools
FROM districts d
JOIN schools s ON ST_Contains(d.geom, s.geom)
GROUP BY d.name;

-- ST_Within: is A completely within B?
-- (inverse of ST_Contains)
SELECT s.name, d.name AS district
FROM schools s
JOIN districts d ON ST_Within(s.geom, d.geom);

-- ST_Covers / ST_CoveredBy: like Contains/Within but handles boundary cases
-- ST_Covers(A, B) is true if no point of B is outside A
-- Preferred over Contains/Within for most practical use cases
SELECT d.name, COUNT(p.id) AS num_pois
FROM districts d
JOIN pois p ON ST_Covers(d.geom, p.geom)
GROUP BY d.name;

-- ST_Touches: do geometries share boundary but not interior?
SELECT a.name, b.name
FROM parcels a, parcels b
WHERE a.id < b.id AND ST_Touches(a.geom, b.geom);

-- ST_Crosses: do geometries cross each other?
-- (e.g., a road crossing a river)
SELECT r.name AS road, w.name AS river
FROM roads r, waterways w
WHERE ST_Crosses(r.geom, w.geom);

-- ST_Overlaps: do geometries overlap partially?
-- (same dimension, share some but not all interior space)
SELECT a.name, b.name,
    ST_Area(ST_Intersection(a.geom, b.geom)) AS overlap_area
FROM zones a, zones b
WHERE a.id < b.id AND ST_Overlaps(a.geom, b.geom);

-- ST_Equals: are geometries topologically identical?
-- (same shape, potentially different vertex order)
SELECT a.id, b.id
FROM dataset_a a, dataset_b b
WHERE ST_Equals(a.geom, b.geom);
```

### The DE-9IM Model

The Dimensionally Extended 9-Intersection Model provides a complete framework for describing spatial relationships.

```sql
-- Get the DE-9IM matrix for two geometries
SELECT ST_Relate(
    ST_GeomFromText('POLYGON((0 0, 2 0, 2 2, 0 2, 0 0))'),
    ST_GeomFromText('POLYGON((1 1, 3 1, 3 3, 1 3, 1 1))')
);
-- Returns: '212101212' (these polygons overlap)

-- Test against a specific pattern
-- Example: "A contains B, and B touches A's boundary"
SELECT ST_Relate(a.geom, b.geom, 'T*F*FF***') AS a_contains_b
FROM table_a a, table_b b;

-- Custom relationship: "geometries share exactly a line segment"
SELECT a.id, b.id
FROM polygons a, polygons b
WHERE a.id < b.id
AND ST_Relate(a.geom, b.geom, '****1****');

-- DE-9IM matrix positions:
--          Interior  Boundary  Exterior
-- Interior   [0]       [1]       [2]
-- Boundary   [3]       [4]       [5]
-- Exterior   [6]       [7]       [8]
--
-- Values:
--   T = intersection exists (dim >= 0)
--   F = no intersection (dim = -1)
--   0 = intersection is a point
--   1 = intersection is a line
--   2 = intersection is an area
--   * = don't care
```

### Practical Spatial Predicate Patterns

```sql
-- Find features that are "almost" within a polygon
-- (within, touching, or within a small tolerance)
SELECT p.name
FROM pois p, districts d
WHERE d.name = 'Haidian'
AND ST_DWithin(p.geom, d.geom, 0.0001);  -- ~11 meters tolerance

-- Find all adjacent polygons (share an edge, not just a point)
SELECT a.id, b.id
FROM parcels a, parcels b
WHERE a.id < b.id
AND ST_Relate(a.geom, b.geom, 'F***1****');  -- boundary intersection is a line

-- Find donut holes: polygons that have interior rings
SELECT id FROM parcels
WHERE ST_NRings(geom) > 1;

-- Find features that straddle a boundary
SELECT b.name
FROM buildings b, district_boundaries db
WHERE ST_Crosses(db.geom, b.geom)
   OR (ST_Intersects(db.geom, b.geom)
       AND NOT ST_Contains(db.geom, b.geom)
       AND NOT ST_Within(b.geom, db.geom));
```

---

## Spatial Joins

Spatial joins are the workhorse of spatial SQL. They combine data from two tables based on spatial relationships.

### Point-in-Polygon Join

```sql
-- Basic point-in-polygon: assign each point to a polygon
SELECT
    p.id AS poi_id,
    p.name AS poi_name,
    d.name AS district_name,
    d.population
FROM pois p
JOIN districts d ON ST_Within(p.geom, d.geom);

-- With index hint (PostGIS uses GiST automatically, but explicit && helps)
SELECT
    p.id, p.name, d.name AS district
FROM pois p
JOIN districts d ON p.geom && d.geom  -- bounding box filter (uses index)
    AND ST_Within(p.geom, d.geom);    -- exact test

-- Count points per polygon
SELECT
    d.name,
    COUNT(p.id) AS num_pois,
    ARRAY_AGG(p.name ORDER BY p.name) AS poi_names
FROM districts d
LEFT JOIN pois p ON ST_Within(p.geom, d.geom)
GROUP BY d.name
ORDER BY num_pois DESC;

-- Point-in-polygon with attributes from both tables
SELECT
    d.name AS district,
    p.category,
    COUNT(*) AS count,
    SUM(p.revenue) AS total_revenue
FROM pois p
JOIN districts d ON ST_Contains(d.geom, p.geom)
GROUP BY d.name, p.category
ORDER BY d.name, count DESC;
```

### Nearest Neighbor (KNN) Joins

PostGIS supports efficient K-nearest neighbor queries using the `<->` and `<#>` operators with GiST indexes.

```sql
-- Find the single nearest hospital to each school
SELECT DISTINCT ON (s.id)
    s.name AS school,
    h.name AS nearest_hospital,
    ST_Distance(s.geom, h.geom) AS distance_m
FROM schools s
CROSS JOIN LATERAL (
    SELECT name, geom
    FROM hospitals
    ORDER BY s.geom <-> geom   -- <-> is the KNN distance operator
    LIMIT 1
) h;

-- Find the 3 nearest restaurants to each hotel
SELECT
    h.name AS hotel,
    r.name AS restaurant,
    r.cuisine,
    ST_Distance(h.geom::geography, r.geom::geography) AS distance_m,
    r.rank
FROM hotels h
CROSS JOIN LATERAL (
    SELECT
        name, cuisine, geom,
        ROW_NUMBER() OVER () AS rank
    FROM restaurants
    ORDER BY h.geom <-> geom
    LIMIT 3
) r;

-- KNN with distance filter (nearest within 5km)
SELECT DISTINCT ON (s.id)
    s.name AS school,
    h.name AS nearest_hospital,
    ST_Distance(s.geom::geography, h.geom::geography) AS dist_m
FROM schools s
CROSS JOIN LATERAL (
    SELECT name, geom
    FROM hospitals
    WHERE ST_DWithin(s.geom::geography, geom::geography, 5000)
    ORDER BY s.geom <-> geom
    LIMIT 1
) h;

-- <-> vs <#>:
-- <-> : centroid-based distance (index-assisted, then refines)
-- <#> : bounding-box distance (faster initial filter)
-- Use <-> for point data; <#> when geometries have varying extents

-- DuckDB: nearest neighbor (no <-> operator, use subquery)
SELECT
    s.name,
    (SELECT r.name FROM restaurants r ORDER BY ST_Distance(s.geom, r.geom) LIMIT 1) AS nearest
FROM schools s;
```

### Within-Distance Joins

```sql
-- Find all parks within 1km of each school
SELECT
    s.name AS school,
    p.name AS park,
    ST_Distance(s.geom::geography, p.geom::geography) AS distance_m
FROM schools s
JOIN parks p ON ST_DWithin(s.geom::geography, p.geom::geography, 1000);

-- Efficient within-distance with projected coordinates
-- (when both tables use the same projected CRS in meters)
SELECT
    s.name, p.name,
    ST_Distance(s.geom, p.geom) AS distance_m
FROM schools s
JOIN parks p ON ST_DWithin(s.geom, p.geom, 1000)  -- 1000 meters
WHERE s.geom && ST_Expand(p.geom, 1000);           -- explicit bbox filter

-- Count features within distance bands
SELECT
    s.name AS school,
    COUNT(*) FILTER (WHERE dist <= 500) AS within_500m,
    COUNT(*) FILTER (WHERE dist <= 1000) AS within_1km,
    COUNT(*) FILTER (WHERE dist <= 2000) AS within_2km
FROM schools s
CROSS JOIN LATERAL (
    SELECT ST_Distance(s.geom::geography, r.geom::geography) AS dist
    FROM restaurants r
    WHERE ST_DWithin(s.geom::geography, r.geom::geography, 2000)
) sub
GROUP BY s.name;
```

### Spatial Join Performance Tips

```sql
-- 1. ALWAYS have spatial indexes on both tables
CREATE INDEX idx_pois_geom ON pois USING GIST (geom);
CREATE INDEX idx_districts_geom ON districts USING GIST (geom);

-- 2. Use && (bounding box) before exact predicates
-- PostGIS often does this automatically, but being explicit helps the planner
WHERE a.geom && b.geom AND ST_Intersects(a.geom, b.geom)

-- 3. Subdivide large polygons for faster joins
CREATE TABLE districts_sub AS
SELECT id, name, ST_Subdivide(geom, 256) AS geom FROM districts;
CREATE INDEX ON districts_sub USING GIST (geom);

-- Now join against subdivided version
SELECT d.name, COUNT(p.id)
FROM districts_sub d
JOIN pois p ON ST_Within(p.geom, d.geom)
GROUP BY d.name;

-- 4. Use ST_Covers/ST_CoveredBy instead of ST_Contains/ST_Within
-- (avoids edge cases with boundary points)

-- 5. For geography joins, consider transforming to a local projected CRS
-- (geography operations are slower due to spheroidal calculations)
SELECT
    a.name, b.name,
    ST_Distance(
        ST_Transform(a.geom, 32650),
        ST_Transform(b.geom, 32650)
    ) AS distance_m
FROM table_a a, table_b b
WHERE ST_DWithin(
    ST_Transform(a.geom, 32650),
    ST_Transform(b.geom, 32650),
    1000
);

-- 6. EXPLAIN ANALYZE to check index usage
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT d.name, COUNT(p.id)
FROM districts d
JOIN pois p ON ST_Contains(d.geom, p.geom)
GROUP BY d.name;
```

---

## Aggregation and Analytics

Spatial aggregation combines geometries and attributes for summary analysis, reporting, and visualization.

### ST_Collect and ST_Union Aggregate

```sql
-- ST_Collect: merge geometries into a GeometryCollection (fast, no dissolve)
SELECT
    province,
    ST_Collect(geom) AS all_cities_multipoint
FROM cities
GROUP BY province;

-- ST_Union: merge and dissolve boundaries (slower, topologically clean)
SELECT
    region,
    ST_Union(geom) AS dissolved_region
FROM provinces
GROUP BY region;

-- Dissolve with attribute aggregation
SELECT
    land_use,
    ST_Union(geom) AS dissolved,
    SUM(area_ha) AS total_area_ha,
    COUNT(*) AS num_parcels
FROM parcels
GROUP BY land_use;

-- Cascaded union (for very large sets, more memory-efficient)
-- PostGIS handles this automatically via ST_Union aggregate
-- For manual control:
SELECT ST_MemUnion(geom) FROM parcels WHERE land_use = 'residential';
```

### GROUP BY with Spatial Functions

```sql
-- Area statistics by district
SELECT
    d.name,
    COUNT(b.id) AS num_buildings,
    SUM(ST_Area(b.geom)) AS total_building_area,
    AVG(ST_Area(b.geom)) AS avg_building_area,
    SUM(ST_Area(b.geom)) / ST_Area(d.geom) * 100 AS building_coverage_pct
FROM districts d
JOIN buildings b ON ST_Contains(d.geom, b.geom)
GROUP BY d.name, d.geom;

-- Geometry statistics
SELECT
    road_class,
    COUNT(*) AS num_segments,
    SUM(ST_Length(geom::geography)) / 1000 AS total_km,
    AVG(ST_Length(geom::geography)) AS avg_segment_m,
    MIN(ST_Length(geom::geography)) AS min_segment_m,
    MAX(ST_Length(geom::geography)) AS max_segment_m
FROM roads
GROUP BY road_class
ORDER BY total_km DESC;

-- Centroid of grouped points (weighted by population)
SELECT
    province,
    ST_Centroid(ST_Collect(geom)) AS simple_centroid,
    -- Population-weighted centroid
    ST_SetSRID(
        ST_MakePoint(
            SUM(ST_X(geom) * population) / SUM(population),
            SUM(ST_Y(geom) * population) / SUM(population)
        ),
        4326
    ) AS weighted_centroid
FROM cities
GROUP BY province;
```

### Window Functions with Spatial Data

```sql
-- Running distance along a GPS track
SELECT
    timestamp,
    ST_Distance(
        geom::geography,
        LAG(geom) OVER (ORDER BY timestamp)::geography
    ) AS segment_distance_m,
    SUM(ST_Distance(
        geom::geography,
        LAG(geom) OVER (ORDER BY timestamp)::geography
    )) OVER (ORDER BY timestamp) AS cumulative_distance_m
FROM gps_points
WHERE trip_id = 42;

-- Rank POIs by distance from a reference point
SELECT
    name,
    category,
    ST_Distance(geom::geography, ref.geom::geography) AS distance_m,
    RANK() OVER (
        PARTITION BY category
        ORDER BY ST_Distance(geom::geography, ref.geom::geography)
    ) AS rank_in_category
FROM pois,
    (SELECT ST_SetSRID(ST_MakePoint(116.4074, 39.9042), 4326) AS geom) ref;

-- Moving window: average speed over last 5 GPS points
SELECT
    timestamp,
    AVG(
        ST_Distance(geom::geography, LAG(geom) OVER w::geography)
        / NULLIF(EXTRACT(EPOCH FROM timestamp - LAG(timestamp) OVER w), 0)
    ) OVER (ORDER BY timestamp ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS avg_speed_mps
FROM gps_points
WHERE trip_id = 42
WINDOW w AS (ORDER BY timestamp);

-- Detect clusters: points that have >= 5 neighbors within 100m
SELECT
    id, name, geom,
    (SELECT COUNT(*)
     FROM pois b
     WHERE b.id != a.id
     AND ST_DWithin(a.geom::geography, b.geom::geography, 100)
    ) AS neighbor_count
FROM pois a
HAVING neighbor_count >= 5;
```

### H3 Hexagonal Aggregation

```sql
-- PostGIS with h3-pg extension
CREATE EXTENSION IF NOT EXISTS h3;
CREATE EXTENSION IF NOT EXISTS h3_postgis;

-- Aggregate points to H3 hexagons
SELECT
    h3_lat_lng_to_cell(geom, 9) AS h3_index,
    COUNT(*) AS count,
    AVG(value) AS avg_value,
    h3_cell_to_boundary_geometry(h3_lat_lng_to_cell(geom, 9)) AS hex_geom
FROM observations
GROUP BY 1;

-- Multi-resolution aggregation
WITH h3_res AS (
    SELECT
        h3_lat_lng_to_cell(geom, 7) AS h3_7,
        h3_lat_lng_to_cell(geom, 9) AS h3_9,
        value
    FROM observations
)
SELECT
    h3_7,
    COUNT(DISTINCT h3_9) AS num_fine_cells,
    COUNT(*) AS num_points,
    AVG(value) AS avg_value
FROM h3_res
GROUP BY h3_7;

-- DuckDB H3 aggregation (see DuckDB section above)
-- Compact representation for variable-resolution coverage
SELECT h3_compact(LIST(h3_index)) AS compact_cells
FROM (
    SELECT h3_latlng_to_cell(ST_Y(geom), ST_X(geom), 9) AS h3_index
    FROM pois
);
```

---

## Raster in SQL

PostGIS Raster enables raster data storage and analysis directly in the database, including raster-vector interactions.

### Loading Raster Data

```bash
# Load a GeoTIFF into PostGIS
raster2pgsql -s 4326 -I -C -M elevation.tif public.elevation | psql -d mydb

# Options:
# -s SRID        : assign SRID
# -I             : create spatial index
# -C             : apply raster constraints
# -M             : vacuum analyze after load
# -t 256x256     : tile size (default: full raster as one row)
# -l 2,4,8       : create overview pyramid levels
# -N -9999       : set nodata value

# Load with tiling and overviews
raster2pgsql -s 32650 -I -C -M -t 256x256 -l 2,4,8 dem.tif public.dem | psql -d mydb
```

### Raster Queries

```sql
-- Get raster metadata
SELECT
    rid,
    ST_Width(rast) AS width,
    ST_Height(rast) AS height,
    ST_NumBands(rast) AS bands,
    ST_ScaleX(rast) AS pixel_size_x,
    ST_ScaleY(rast) AS pixel_size_y,
    ST_SRID(rast) AS srid,
    ST_Envelope(rast) AS extent
FROM elevation
LIMIT 1;

-- Get pixel value at a point
SELECT ST_Value(rast, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)) AS elevation
FROM elevation
WHERE ST_Intersects(rast, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326));

-- Get pixel values along a transect line
SELECT
    (ST_DumpAsPolygons(
        ST_Clip(rast, line.geom)
    )).*
FROM elevation,
    (SELECT ST_SetSRID(
        ST_MakeLine(ST_MakePoint(116.3, 39.8), ST_MakePoint(116.5, 40.0)),
        4326
    ) AS geom) line
WHERE ST_Intersects(rast, line.geom);

-- Sample raster values at point locations
SELECT
    p.id,
    p.name,
    ST_Value(r.rast, p.geom) AS elevation
FROM sample_points p
JOIN elevation r ON ST_Intersects(r.rast, p.geom);
```

### Zonal Statistics via SQL

```sql
-- Zonal statistics: statistics of raster values within polygon zones
SELECT
    d.name AS district,
    (ST_SummaryStatsAgg(ST_Clip(r.rast, d.geom), 1, TRUE)).*
FROM districts d
JOIN elevation r ON ST_Intersects(r.rast, d.geom)
GROUP BY d.name;
-- Returns: count, sum, mean, stddev, min, max for each district

-- Manual zonal statistics with more control
SELECT
    d.name,
    COUNT(val) AS pixel_count,
    AVG(val) AS mean_elevation,
    MIN(val) AS min_elevation,
    MAX(val) AS max_elevation,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY val) AS median_elevation,
    STDDEV(val) AS stddev_elevation
FROM districts d
JOIN elevation r ON ST_Intersects(r.rast, d.geom)
CROSS JOIN LATERAL (
    SELECT (ST_PixelAsPoints(ST_Clip(r.rast, d.geom, TRUE), 1)).val
) pixels
WHERE val IS NOT NULL
GROUP BY d.name;

-- Histogram of raster values within a polygon
SELECT
    width_bucket(val, 0, 5000, 50) AS bucket,
    COUNT(*) AS freq,
    MIN(val) AS range_min,
    MAX(val) AS range_max
FROM elevation r,
    (SELECT geom FROM districts WHERE name = 'Haidian') d
CROSS JOIN LATERAL (
    SELECT (ST_PixelAsPoints(ST_Clip(r.rast, d.geom, TRUE), 1)).val
) pixels
WHERE ST_Intersects(r.rast, d.geom)
AND val IS NOT NULL
GROUP BY bucket
ORDER BY bucket;
```

### Map Algebra

```sql
-- Single raster map algebra: classify elevation
SELECT ST_MapAlgebra(
    rast, 1,                          -- input raster, band
    'CASE
        WHEN [rast] < 200 THEN 1      -- lowland
        WHEN [rast] < 500 THEN 2      -- hills
        WHEN [rast] < 1500 THEN 3     -- mountains
        ELSE 4                         -- high mountains
    END'::text,
    '32BF'::text                       -- output pixel type
) AS classified
FROM elevation;

-- Two-raster map algebra: NDVI calculation
-- NDVI = (NIR - Red) / (NIR + Red)
SELECT ST_MapAlgebra(
    nir.rast, 1,
    red.rast, 1,
    '([rast1] - [rast2]) / NULLIF(([rast1] + [rast2]), 0)'::text,
    '32BF'::text
) AS ndvi
FROM nir_band nir, red_band red
WHERE ST_Intersects(nir.rast, red.rast);

-- Slope and aspect from DEM
SELECT
    ST_Slope(rast, 1, '32BF') AS slope,
    ST_Aspect(rast, 1, '32BF') AS aspect
FROM elevation;

-- Hillshade
SELECT ST_HillShade(rast, 1, '32BF', 315, 45) AS hillshade
FROM elevation;

-- Raster to vector: contour lines
SELECT
    (ST_Contour(rast, 1, ARRAY[100, 200, 500, 1000, 2000, 3000])).*
FROM elevation;

-- Raster to polygons
SELECT (ST_DumpAsPolygons(rast)).*
FROM classified_landcover;
```

---

## Network Analysis in SQL

pgRouting extends PostGIS with graph-based routing algorithms for road networks, utilities, and any topological network.

### Installation and Setup

```bash
# Ubuntu / Debian
sudo apt-get install postgresql-16-pgrouting

# macOS
brew install pgrouting
```

```sql
CREATE EXTENSION IF NOT EXISTS pgrouting;

-- Create a routable network table
CREATE TABLE road_network (
    id SERIAL PRIMARY KEY,
    source INTEGER,
    target INTEGER,
    cost DOUBLE PRECISION,
    reverse_cost DOUBLE PRECISION,
    geom geometry(LineString, 4326)
);

-- Load road data (e.g., from OpenStreetMap via osm2pgrouting)
-- or build topology from existing line data:
SELECT pgr_createTopology(
    'road_network',    -- table
    0.00001,           -- tolerance (in CRS units)
    'geom',            -- geometry column
    'id'               -- id column
);

-- Verify topology
SELECT pgr_analyzeGraph('road_network', 0.00001, 'geom', 'id');
```

### Shortest Path (Dijkstra)

```sql
-- Single shortest path
SELECT seq, node, edge, cost, agg_cost, geom
FROM pgr_dijkstra(
    'SELECT id, source, target, cost, reverse_cost FROM road_network',
    1,          -- start node
    100,        -- end node
    directed := true
) AS route
JOIN road_network ON route.edge = road_network.id;

-- Get the route as a single LineString
SELECT ST_MakeLine(geom ORDER BY seq) AS route_line
FROM pgr_dijkstra(
    'SELECT id, source, target, cost, reverse_cost FROM road_network',
    1, 100, directed := true
) AS route
JOIN road_network ON route.edge = road_network.id;

-- Multiple shortest paths: many-to-many
SELECT *
FROM pgr_dijkstra(
    'SELECT id, source, target, cost, reverse_cost FROM road_network',
    ARRAY[1, 5, 10],     -- start nodes
    ARRAY[100, 200],      -- end nodes
    directed := true
);

-- A* (faster with coordinates heuristic)
SELECT seq, node, edge, cost, agg_cost
FROM pgr_aStar(
    'SELECT id, source, target, cost, reverse_cost,
            ST_X(ST_StartPoint(geom)) AS x1, ST_Y(ST_StartPoint(geom)) AS y1,
            ST_X(ST_EndPoint(geom)) AS x2, ST_Y(ST_EndPoint(geom)) AS y2
     FROM road_network',
    1, 100, directed := true
);
```

### Isochrones (Service Areas)

```sql
-- Driving distance: all nodes reachable within a cost threshold
SELECT node, edge, cost, agg_cost
FROM pgr_drivingDistance(
    'SELECT id, source, target, cost, reverse_cost FROM road_network',
    1,              -- start node
    600,            -- max cost (e.g., 600 seconds = 10 minutes)
    directed := true
);

-- Build isochrone polygon from reachable nodes
WITH reachable AS (
    SELECT node, agg_cost
    FROM pgr_drivingDistance(
        'SELECT id, source, target, cost, reverse_cost FROM road_network',
        1, 600, directed := true
    )
),
node_geoms AS (
    SELECT r.agg_cost, v.the_geom AS geom
    FROM reachable r
    JOIN road_network_vertices_pgr v ON r.node = v.id
)
SELECT
    band,
    ST_ConcaveHull(ST_Collect(geom), 0.3) AS isochrone
FROM (
    SELECT geom,
        CASE
            WHEN agg_cost <= 300 THEN '5 min'
            WHEN agg_cost <= 600 THEN '10 min'
        END AS band
    FROM node_geoms
) sub
GROUP BY band;
```

### Traveling Salesman Problem (TSP)

```sql
-- Optimal order to visit a set of nodes
SELECT seq, node, cost, agg_cost
FROM pgr_TSP(
    $$
    SELECT * FROM pgr_dijkstraCostMatrix(
        'SELECT id, source, target, cost, reverse_cost FROM road_network',
        ARRAY[1, 5, 10, 20, 50, 100],
        directed := true
    )
    $$,
    start_id := 1,
    end_id := 1    -- return to start (optional)
);
```

### Turn Restrictions

```sql
-- Create turn restrictions table
CREATE TABLE turn_restrictions (
    id SERIAL PRIMARY KEY,
    via_path INTEGER[],    -- array of edge IDs
    cost DOUBLE PRECISION  -- high cost = restriction
);

-- No left turn from edge 5 via node to edge 12
INSERT INTO turn_restrictions (via_path, cost)
VALUES (ARRAY[5, 12], 1000000);

-- Route with turn restrictions
SELECT *
FROM pgr_trsp(
    'SELECT id, source, target, cost, reverse_cost FROM road_network',
    'SELECT via_path, cost FROM turn_restrictions',
    1, 100, directed := true
);
```

---

## Advanced Patterns

Production spatial SQL often requires combining spatial operations with advanced PostgreSQL features.

### Materialized Views for Spatial Analytics

```sql
-- Pre-compute district statistics as a materialized view
CREATE MATERIALIZED VIEW district_stats AS
SELECT
    d.id,
    d.name,
    d.geom,
    COUNT(b.id) AS num_buildings,
    SUM(ST_Area(b.geom)) AS total_building_area,
    (SELECT COUNT(*) FROM pois p WHERE ST_Contains(d.geom, p.geom)) AS num_pois,
    (SELECT SUM(ST_Length(r.geom::geography))
     FROM roads r
     WHERE ST_Intersects(d.geom, r.geom)) / 1000 AS road_km
FROM districts d
LEFT JOIN buildings b ON ST_Contains(d.geom, b.geom)
GROUP BY d.id, d.name, d.geom
WITH DATA;

-- Create spatial index on the materialized view
CREATE INDEX idx_district_stats_geom ON district_stats USING GIST (geom);

-- Refresh when source data changes
REFRESH MATERIALIZED VIEW CONCURRENTLY district_stats;
-- (CONCURRENTLY requires a unique index)
CREATE UNIQUE INDEX idx_district_stats_id ON district_stats (id);
```

### Spatial Triggers

```sql
-- Auto-populate geometry from lat/lon columns
CREATE OR REPLACE FUNCTION update_geom()
RETURNS TRIGGER AS $$
BEGIN
    NEW.geom := ST_SetSRID(ST_MakePoint(NEW.longitude, NEW.latitude), 4326);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_update_geom
    BEFORE INSERT OR UPDATE OF longitude, latitude
    ON observations
    FOR EACH ROW
    EXECUTE FUNCTION update_geom();

-- Validate geometry on insert/update
CREATE OR REPLACE FUNCTION validate_geometry()
RETURNS TRIGGER AS $$
BEGIN
    IF NOT ST_IsValid(NEW.geom) THEN
        NEW.geom := ST_MakeValid(NEW.geom);
        RAISE NOTICE 'Geometry for row % was invalid and has been repaired', NEW.id;
    END IF;

    IF ST_IsEmpty(NEW.geom) THEN
        RAISE EXCEPTION 'Empty geometry not allowed for row %', NEW.id;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_validate_geom
    BEFORE INSERT OR UPDATE OF geom
    ON parcels
    FOR EACH ROW
    EXECUTE FUNCTION validate_geometry();

-- Log spatial changes (audit trail)
CREATE TABLE spatial_audit (
    id SERIAL PRIMARY KEY,
    table_name TEXT,
    row_id INTEGER,
    operation TEXT,
    old_geom geometry,
    new_geom geometry,
    changed_at TIMESTAMPTZ DEFAULT NOW(),
    changed_by TEXT DEFAULT current_user
);

CREATE OR REPLACE FUNCTION audit_spatial_change()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND NOT ST_Equals(OLD.geom, NEW.geom) THEN
        INSERT INTO spatial_audit (table_name, row_id, operation, old_geom, new_geom)
        VALUES (TG_TABLE_NAME, OLD.id, 'UPDATE', OLD.geom, NEW.geom);
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO spatial_audit (table_name, row_id, operation, old_geom)
        VALUES (TG_TABLE_NAME, OLD.id, 'DELETE', OLD.geom);
    END IF;
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_audit_parcels
    AFTER UPDATE OR DELETE ON parcels
    FOR EACH ROW
    EXECUTE FUNCTION audit_spatial_change();
```

### Temporal + Spatial Queries

```sql
-- Find all events within 1km and 1 hour of each other
SELECT
    a.id AS event_a,
    b.id AS event_b,
    ST_Distance(a.geom::geography, b.geom::geography) AS distance_m,
    ABS(EXTRACT(EPOCH FROM a.event_time - b.event_time)) / 60 AS time_diff_min
FROM events a
JOIN events b ON a.id < b.id
    AND ST_DWithin(a.geom::geography, b.geom::geography, 1000)
    AND ABS(EXTRACT(EPOCH FROM a.event_time - b.event_time)) <= 3600;

-- Temporal range with spatial filter: find features valid at a specific time
-- Using PostgreSQL range types
CREATE TABLE land_parcels (
    id SERIAL PRIMARY KEY,
    parcel_no TEXT,
    valid_range TSTZRANGE,
    geom geometry(Polygon, 4326)
);

SELECT parcel_no, geom
FROM land_parcels
WHERE valid_range @> '2025-06-15'::timestamptz
AND ST_Intersects(geom, ST_MakeEnvelope(116, 39, 117, 41, 4326));

-- Movement analysis: speed between consecutive GPS points
SELECT
    id,
    timestamp,
    ST_Distance(geom::geography, prev_geom::geography) /
        NULLIF(EXTRACT(EPOCH FROM timestamp - prev_ts), 0) * 3.6 AS speed_kmh
FROM (
    SELECT
        id, timestamp, geom,
        LAG(geom) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS prev_geom,
        LAG(timestamp) OVER (PARTITION BY vehicle_id ORDER BY timestamp) AS prev_ts
    FROM gps_tracks
) sub
WHERE prev_geom IS NOT NULL;
```

### Recursive CTEs for Network Traversal

```sql
-- Trace upstream in a river network
WITH RECURSIVE upstream AS (
    -- Start from a specific river segment
    SELECT id, from_node, to_node, geom, 0 AS depth, stream_order
    FROM river_segments
    WHERE id = 42

    UNION ALL

    -- Recursively find upstream segments
    SELECT r.id, r.from_node, r.to_node, r.geom, u.depth + 1, r.stream_order
    FROM river_segments r
    JOIN upstream u ON r.to_node = u.from_node
    WHERE u.depth < 50  -- limit recursion depth
)
SELECT
    ST_Union(geom) AS upstream_network,
    SUM(ST_Length(geom::geography)) / 1000 AS total_km,
    MAX(stream_order) AS max_stream_order
FROM upstream;

-- Find all connected parcels (adjacency graph)
WITH RECURSIVE connected AS (
    SELECT id, geom, id AS cluster_id, 0 AS depth
    FROM parcels
    WHERE id = 1

    UNION ALL

    SELECT p.id, p.geom, c.cluster_id, c.depth + 1
    FROM parcels p
    JOIN connected c ON ST_Touches(p.geom, c.geom)
    WHERE c.depth < 20
    AND p.id NOT IN (SELECT id FROM connected)
)
SELECT cluster_id, COUNT(*) AS num_parcels, ST_Union(geom) AS cluster_geom
FROM connected
GROUP BY cluster_id;
```

### LATERAL Joins for Spatial Analysis

```sql
-- For each district, find the 5 largest buildings
SELECT d.name, b.*
FROM districts d
CROSS JOIN LATERAL (
    SELECT id, name, ST_Area(geom) AS area
    FROM buildings
    WHERE ST_Contains(d.geom, geom)
    ORDER BY ST_Area(geom) DESC
    LIMIT 5
) b;

-- For each road segment, find the nearest parallel road
SELECT
    r1.id,
    r1.road_name,
    nearest.*
FROM roads r1
CROSS JOIN LATERAL (
    SELECT
        r2.id AS neighbor_id,
        r2.road_name AS neighbor_name,
        ST_Distance(r1.geom, r2.geom) AS distance,
        -- Check if roughly parallel (similar azimuth)
        ABS(
            ST_Azimuth(ST_StartPoint(r1.geom), ST_EndPoint(r1.geom)) -
            ST_Azimuth(ST_StartPoint(r2.geom), ST_EndPoint(r2.geom))
        ) AS azimuth_diff
    FROM roads r2
    WHERE r2.id != r1.id
    AND ST_DWithin(r1.geom, r2.geom, 100)
    ORDER BY ST_Distance(r1.geom, r2.geom)
    LIMIT 1
) nearest
WHERE nearest.azimuth_diff < 0.1;  -- roughly parallel (< ~6 degrees)
```

---

## Migration and ETL

Moving spatial data between formats, databases, and services is a routine GIS task. These tools and patterns cover the most common scenarios.

### ogr2ogr to PostGIS

`ogr2ogr` (part of GDAL) is the Swiss Army knife for spatial data conversion and loading.

```bash
# Shapefile to PostGIS
ogr2ogr -f "PostgreSQL" \
    PG:"host=localhost dbname=gisdb user=gis password=secret" \
    buildings.shp \
    -nln public.buildings \
    -nlt PROMOTE_TO_MULTI \
    -lco GEOMETRY_NAME=geom \
    -lco FID=id \
    -lco SPATIAL_INDEX=GIST \
    -s_srs EPSG:4326 \
    -t_srs EPSG:4326

# GeoJSON to PostGIS
ogr2ogr -f "PostgreSQL" \
    PG:"host=localhost dbname=gisdb" \
    parcels.geojson \
    -nln parcels \
    -append           # append to existing table (vs -overwrite)

# GeoPackage to PostGIS (specific layer)
ogr2ogr -f "PostgreSQL" \
    PG:"dbname=gisdb" \
    data.gpkg \
    -sql "SELECT * FROM buildings WHERE type = 'residential'" \
    -nln residential_buildings

# PostGIS to GeoJSON
ogr2ogr -f "GeoJSON" output.geojson \
    PG:"dbname=gisdb" \
    -sql "SELECT id, name, geom FROM buildings WHERE ST_Area(geom) > 100"

# PostGIS to GeoParquet
ogr2ogr -f "Parquet" buildings.parquet \
    PG:"dbname=gisdb" \
    -sql "SELECT * FROM buildings"

# PostGIS to Shapefile
ogr2ogr -f "ESRI Shapefile" output_dir/ \
    PG:"dbname=gisdb" \
    buildings

# CSV with coordinates to PostGIS
ogr2ogr -f "PostgreSQL" PG:"dbname=gisdb" \
    stations.csv \
    -oo X_POSSIBLE_NAMES=longitude \
    -oo Y_POSSIBLE_NAMES=latitude \
    -a_srs EPSG:4326 \
    -nln stations

# Reproject during load
ogr2ogr -f "PostgreSQL" PG:"dbname=gisdb" \
    buildings.shp \
    -s_srs EPSG:32650 \
    -t_srs EPSG:4326 \
    -nln buildings_wgs84

# Batch load all shapefiles in a directory
for f in data/*.shp; do
    table=$(basename "$f" .shp | tr '[:upper:]' '[:lower:]')
    ogr2ogr -f "PostgreSQL" PG:"dbname=gisdb" \
        "$f" -nln "$table" -overwrite \
        -lco GEOMETRY_NAME=geom -lco SPATIAL_INDEX=GIST
done

# Load from a WFS service
ogr2ogr -f "PostgreSQL" PG:"dbname=gisdb" \
    WFS:"https://example.com/geoserver/wfs" \
    "namespace:layer_name" \
    -nln wfs_data

# Load from a zipped shapefile
ogr2ogr -f "PostgreSQL" PG:"dbname=gisdb" \
    /vsizip/buildings.zip/buildings.shp \
    -nln buildings

# Load from S3
ogr2ogr -f "PostgreSQL" PG:"dbname=gisdb" \
    /vsis3/my-bucket/data/parcels.parquet \
    -nln parcels

# Performance: use PG COPY mode for faster loading
ogr2ogr -f "PostgreSQL" PG:"dbname=gisdb" \
    large_dataset.gpkg \
    -nln large_data \
    --config PG_USE_COPY YES
```

### COPY with Spatial Data

PostgreSQL's COPY command is the fastest way to bulk-load data.

```sql
-- Export to CSV with WKT geometry
COPY (
    SELECT id, name, ST_AsText(geom) AS geom_wkt
    FROM buildings
) TO '/tmp/buildings.csv' WITH CSV HEADER;

-- Import from CSV with WKT
CREATE TEMP TABLE buildings_import (
    id INTEGER,
    name TEXT,
    geom_wkt TEXT
);
COPY buildings_import FROM '/tmp/buildings.csv' WITH CSV HEADER;
INSERT INTO buildings (id, name, geom)
SELECT id, name, ST_GeomFromText(geom_wkt, 4326)
FROM buildings_import;

-- Export to CSV with WKB (more efficient for programmatic use)
COPY (
    SELECT id, name, ST_AsEWKB(geom)::text AS geom_ewkb
    FROM buildings
) TO '/tmp/buildings_wkb.csv' WITH CSV HEADER;

-- Using \copy from psql (client-side, no server file access needed)
\copy (SELECT id, name, ST_AsGeoJSON(geom) FROM buildings) TO 'buildings.csv' CSV HEADER
```

### pg_dump and Restore

```bash
# Dump a spatial database
pg_dump -Fc -d gisdb -f gisdb.dump

# Dump specific tables
pg_dump -Fc -d gisdb -t buildings -t parcels -t roads -f spatial_tables.dump

# Dump with schema only (no data)
pg_dump -Fc -d gisdb --schema-only -f gisdb_schema.dump

# Restore to a new database
createdb gisdb_new
psql -d gisdb_new -c "CREATE EXTENSION postgis;"
pg_restore -d gisdb_new gisdb.dump

# Restore specific tables
pg_restore -d gisdb_new -t buildings -t parcels spatial_tables.dump

# Plain SQL dump (readable, but larger)
pg_dump -d gisdb --inserts -f gisdb.sql
```

### Format Conversion Patterns

```sql
-- PostGIS: export as GeoJSON feature collection
SELECT json_build_object(
    'type', 'FeatureCollection',
    'features', json_agg(
        json_build_object(
            'type', 'Feature',
            'geometry', ST_AsGeoJSON(geom)::json,
            'properties', json_build_object(
                'id', id,
                'name', name,
                'population', population
            )
        )
    )
) AS geojson
FROM cities;

-- Export as KML
SELECT
    xmlelement(name "kml",
        xmlattributes('http://www.opengis.net/kml/2.2' AS xmlns),
        xmlelement(name "Document",
            xmlagg(
                xmlelement(name "Placemark",
                    xmlelement(name "name", name),
                    xmlelement(name "description", 'Pop: ' || population),
                    ST_AsKML(geom)::xml
                )
            )
        )
    )
FROM cities;

-- Convert between geometry representations
SELECT
    ST_AsText(geom) AS wkt,              -- Well-Known Text
    ST_AsBinary(geom) AS wkb,            -- Well-Known Binary
    ST_AsEWKT(geom) AS ewkt,             -- Extended WKT (includes SRID)
    ST_AsEWKB(geom) AS ewkb,             -- Extended WKB
    ST_AsGeoJSON(geom) AS geojson,       -- GeoJSON geometry
    ST_AsGML(geom) AS gml,               -- Geography Markup Language
    ST_AsKML(geom) AS kml,               -- KML
    ST_AsSVG(geom) AS svg,               -- SVG path
    ST_AsEncodedPolyline(geom) AS polyline, -- Google Encoded Polyline
    ST_AsMVTGeom(geom, bounds) AS mvt,   -- Mapbox Vector Tile geometry
    ST_AsGeobuf(row) AS geobuf           -- Geobuf (Protocol Buffers)
FROM cities;

-- DuckDB: convert between formats
SELECT
    ST_AsText(geom) AS wkt,
    ST_AsWKB(geom) AS wkb,
    ST_AsGeoJSON(geom) AS geojson
FROM my_table;
```

### Shapefile, GeoJSON, and GeoParquet Import/Export

```sql
-- PostGIS: load Shapefile via shp2pgsql
-- (alternative to ogr2ogr, PostGIS-specific)
-- shp2pgsql -s 4326 -I buildings.shp public.buildings | psql -d gisdb

-- PostGIS: Foreign Data Wrapper for direct file access
CREATE EXTENSION IF NOT EXISTS ogr_fdw;

CREATE SERVER ogr_server
    FOREIGN DATA WRAPPER ogr_fdw
    OPTIONS (datasource '/data/spatial_files');

IMPORT FOREIGN SCHEMA ogr_all
    FROM SERVER ogr_server
    INTO public;

-- Now query shapefiles directly as tables
SELECT * FROM buildings_shp WHERE population > 10000;

-- DuckDB: seamless format conversion pipeline
-- Read GeoJSON, transform, write GeoParquet
COPY (
    SELECT
        id, name, category,
        ST_Transform(geom, 'EPSG:4326', 'EPSG:32650') AS geom
    FROM ST_Read('input.geojson')
    WHERE category IN ('hospital', 'school', 'park')
) TO 'filtered_pois.parquet' (FORMAT PARQUET);

-- DuckDB: GeoParquet to GeoJSON
COPY (SELECT * FROM read_parquet('data.parquet'))
TO 'output.geojson' WITH (FORMAT GDAL, DRIVER 'GeoJSON');

-- DuckDB: Shapefile to GeoPackage
COPY (SELECT * FROM ST_Read('input.shp'))
TO 'output.gpkg' WITH (FORMAT GDAL, DRIVER 'GPKG');
```

---

## Performance Optimization

Spatial queries can be computationally expensive. Proper optimization makes the difference between seconds and hours.

### EXPLAIN ANALYZE for Spatial Queries

```sql
-- Always start with EXPLAIN ANALYZE to understand query plans
EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)
SELECT d.name, COUNT(p.id)
FROM districts d
JOIN pois p ON ST_Contains(d.geom, p.geom)
GROUP BY d.name;

-- What to look for in the output:
-- 1. "Index Scan using idx_pois_geom" = good (index is being used)
-- 2. "Seq Scan on pois" = bad (full table scan, missing index)
-- 3. "Bitmap Index Scan" = acceptable (index used, then heap scan)
-- 4. "Recheck Cond" = normal for GiST (bbox match, then exact test)
-- 5. "Buffers: shared hit" = data from cache
-- 6. "Buffers: shared read" = data from disk

-- Force index usage (if planner chooses wrong)
SET enable_seqscan = off;
EXPLAIN ANALYZE SELECT ...;
SET enable_seqscan = on;  -- reset

-- Check if spatial index exists
SELECT
    schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE indexdef ILIKE '%gist%'
ORDER BY tablename;
```

### Index Strategies

```sql
-- Strategy 1: GiST index on every geometry column
CREATE INDEX idx_buildings_geom ON buildings USING GIST (geom);
CREATE INDEX idx_roads_geom ON roads USING GIST (geom);
CREATE INDEX idx_parcels_geom ON parcels USING GIST (geom);

-- Strategy 2: Partial indexes for frequently filtered subsets
CREATE INDEX idx_major_roads_geom ON roads USING GIST (geom)
    WHERE road_class IN ('motorway', 'trunk', 'primary');

CREATE INDEX idx_active_parcels ON parcels USING GIST (geom)
    WHERE status = 'active';

-- Strategy 3: Expression indexes
CREATE INDEX idx_buildings_centroid ON buildings USING GIST (ST_Centroid(geom));
CREATE INDEX idx_buildings_geog ON buildings USING GIST ((geom::geography));

-- Strategy 4: Include columns for index-only scans (PostgreSQL 12+)
-- Note: not supported for GiST, but useful with B-tree for attribute filters
CREATE INDEX idx_buildings_type ON buildings (building_type)
    INCLUDE (name, height);

-- Strategy 5: BRIN for huge, spatially-ordered tables
-- Best when data is physically ordered by location (e.g., loaded by tile)
CREATE INDEX idx_lidar_geom_brin ON lidar_points USING BRIN (geom)
    WITH (pages_per_range = 128);

-- Compare index sizes
SELECT
    indexrelname AS index_name,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE relname = 'buildings'
ORDER BY pg_relation_size(indexrelid) DESC;
```

### Clustering (CLUSTER)

```sql
-- Physically reorder table rows to match spatial index order
-- This dramatically improves range queries by reducing disk I/O

-- Step 1: Create spatial index
CREATE INDEX idx_buildings_geom ON buildings USING GIST (geom);

-- Step 2: Cluster the table
CLUSTER buildings USING idx_buildings_geom;

-- Step 3: Analyze
ANALYZE buildings;

-- Note: CLUSTER locks the table exclusively and rewrites it completely.
-- For large tables, schedule during maintenance windows.
-- New inserts go to the end (not spatially ordered), so re-cluster periodically.

-- Alternative: pg_repack for online reclustering
-- (no exclusive lock, but requires the pg_repack extension)
-- pg_repack -d gisdb -t buildings --order-by "geom USING <<<"
```

### Partitioning

```sql
-- Range partitioning by spatial region
-- Useful for very large datasets that span multiple regions

CREATE TABLE observations (
    id BIGSERIAL,
    timestamp TIMESTAMPTZ,
    value DOUBLE PRECISION,
    geom geometry(Point, 4326)
) PARTITION BY LIST (
    -- Partition by grid cell
    (floor(ST_X(geom))::int || '_' || floor(ST_Y(geom))::int)
);

-- Simpler approach: partition by a region column
CREATE TABLE sensor_data (
    id BIGSERIAL,
    region TEXT NOT NULL,
    timestamp TIMESTAMPTZ,
    value DOUBLE PRECISION,
    geom geometry(Point, 4326)
) PARTITION BY LIST (region);

CREATE TABLE sensor_data_north PARTITION OF sensor_data FOR VALUES IN ('north');
CREATE TABLE sensor_data_south PARTITION OF sensor_data FOR VALUES IN ('south');
CREATE TABLE sensor_data_east PARTITION OF sensor_data FOR VALUES IN ('east');
CREATE TABLE sensor_data_west PARTITION OF sensor_data FOR VALUES IN ('west');

-- Each partition gets its own spatial index
CREATE INDEX ON sensor_data_north USING GIST (geom);
CREATE INDEX ON sensor_data_south USING GIST (geom);
CREATE INDEX ON sensor_data_east USING GIST (geom);
CREATE INDEX ON sensor_data_west USING GIST (geom);

-- Hash partitioning for uniform distribution
CREATE TABLE points (
    id BIGSERIAL,
    geom geometry(Point, 4326),
    properties JSONB
) PARTITION BY HASH (id);

CREATE TABLE points_p0 PARTITION OF points FOR VALUES WITH (MODULUS 4, REMAINDER 0);
CREATE TABLE points_p1 PARTITION OF points FOR VALUES WITH (MODULUS 4, REMAINDER 1);
CREATE TABLE points_p2 PARTITION OF points FOR VALUES WITH (MODULUS 4, REMAINDER 2);
CREATE TABLE points_p3 PARTITION OF points FOR VALUES WITH (MODULUS 4, REMAINDER 3);

-- Time-based partitioning (common for sensor/IoT data)
CREATE TABLE gps_tracks (
    id BIGSERIAL,
    vehicle_id INTEGER,
    timestamp TIMESTAMPTZ NOT NULL,
    geom geometry(Point, 4326)
) PARTITION BY RANGE (timestamp);

CREATE TABLE gps_tracks_2025_01 PARTITION OF gps_tracks
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE gps_tracks_2025_02 PARTITION OF gps_tracks
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
-- ... create partitions for each month

-- Automatic partition creation (pg_partman extension)
CREATE EXTENSION pg_partman;
SELECT partman.create_parent(
    p_parent_table := 'public.gps_tracks',
    p_control := 'timestamp',
    p_type := 'native',
    p_interval := '1 month',
    p_premake := 3
);
```

### Connection Pooling and Concurrency

```bash
# PgBouncer configuration for spatial workloads
# pgbouncer.ini
[databases]
gisdb = host=127.0.0.1 port=5432 dbname=gisdb

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction       # transaction pooling for web apps
max_client_conn = 200
default_pool_size = 25
reserve_pool_size = 5
server_lifetime = 3600
server_idle_timeout = 600
```

```sql
-- Monitor active spatial queries
SELECT
    pid,
    state,
    query_start,
    NOW() - query_start AS duration,
    LEFT(query, 100) AS query_preview
FROM pg_stat_activity
WHERE query ILIKE '%st_%'
AND state = 'active'
ORDER BY query_start;

-- Kill long-running spatial queries
SELECT pg_cancel_backend(pid)
FROM pg_stat_activity
WHERE NOW() - query_start > INTERVAL '5 minutes'
AND query ILIKE '%st_%'
AND state = 'active';

-- Set statement timeout for spatial queries
SET statement_timeout = '120s';  -- 2 minute limit
```

### Query Optimization Patterns

```sql
-- Pattern 1: Use ST_Subdivide for large polygon joins
-- Before (slow):
SELECT d.name, COUNT(p.id)
FROM districts d JOIN pois p ON ST_Contains(d.geom, p.geom)
GROUP BY d.name;

-- After (fast):
WITH district_sub AS (
    SELECT id, name, ST_Subdivide(geom, 256) AS geom
    FROM districts
)
SELECT name, COUNT(p.id)
FROM district_sub d JOIN pois p ON ST_Contains(d.geom, p.geom)
GROUP BY name;

-- Pattern 2: Two-stage filter (bbox then exact)
SELECT *
FROM buildings
WHERE geom && ST_MakeEnvelope(116, 39, 117, 41, 4326)  -- fast bbox
AND ST_Intersects(geom, search_polygon);                 -- exact

-- Pattern 3: Avoid function calls in WHERE for indexed columns
-- Bad (prevents index use):
WHERE ST_X(geom) BETWEEN 116 AND 117

-- Good (uses spatial index):
WHERE geom && ST_MakeEnvelope(116, 39, 117, 41, 4326)

-- Pattern 4: Materialize intermediate results for repeated use
WITH search_area AS MATERIALIZED (
    SELECT ST_Buffer(
        ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)::geography,
        5000
    )::geometry AS geom
)
SELECT b.name, b.height
FROM buildings b, search_area s
WHERE ST_Intersects(b.geom, s.geom);

-- Pattern 5: Use ST_DWithin instead of ST_Distance < threshold
-- Bad (computes distance for every pair):
WHERE ST_Distance(a.geom::geography, b.geom::geography) < 1000

-- Good (uses spatial index):
WHERE ST_DWithin(a.geom::geography, b.geom::geography, 1000)

-- Pattern 6: Reduce geometry precision before expensive operations
SELECT ST_Intersection(
    ST_SnapToGrid(a.geom, 0.0001),
    ST_SnapToGrid(b.geom, 0.0001)
) FROM ...;

-- Pattern 7: Use geography only when needed
-- Compute distance in projected CRS (faster than geography)
SELECT ST_Distance(
    ST_Transform(a.geom, 32650),
    ST_Transform(b.geom, 32650)
) AS distance_m
FROM ...;

-- Pattern 8: Parallel query execution
SET max_parallel_workers_per_gather = 4;
SET parallel_tuple_cost = 0.001;
SET parallel_setup_cost = 100;
-- PostGIS functions are parallel-safe as of PostGIS 3.0
```

---

## Docker and Deployment

### PostGIS Docker

```bash
# Official PostGIS Docker image
docker run -d \
    --name postgis \
    -e POSTGRES_PASSWORD=mysecret \
    -e POSTGRES_DB=gisdb \
    -p 5432:5432 \
    -v pgdata:/var/lib/postgresql/data \
    postgis/postgis:16-3.4

# With specific locale and extensions
docker run -d \
    --name postgis \
    -e POSTGRES_PASSWORD=mysecret \
    -e POSTGRES_DB=gisdb \
    -e POSTGRES_INITDB_ARGS="--locale=en_US.UTF-8" \
    -p 5432:5432 \
    -v pgdata:/var/lib/postgresql/data \
    postgis/postgis:16-3.4

# Connect
psql -h localhost -U postgres -d gisdb

# Docker Compose example
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: gisdb
      POSTGRES_USER: gis
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gis -d gisdb"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G

  pg_tileserv:
    image: pramsey/pg_tileserv:latest
    environment:
      DATABASE_URL: postgres://gis:${POSTGRES_PASSWORD}@postgis:5432/gisdb
    ports:
      - "7800:7800"
    depends_on:
      postgis:
        condition: service_healthy

  pg_featureserv:
    image: pramsey/pg_featureserv:latest
    environment:
      DATABASE_URL: postgres://gis:${POSTGRES_PASSWORD}@postgis:5432/gisdb
    ports:
      - "9000:9000"
    depends_on:
      postgis:
        condition: service_healthy

volumes:
  pgdata:
```

### pg_tileserv (Vector Tile Server)

pg_tileserv serves PostGIS tables and functions as Mapbox Vector Tiles (MVT) automatically.

```bash
# Run standalone
export DATABASE_URL=postgres://gis:secret@localhost:5432/gisdb
pg_tileserv

# Access tiles at:
# http://localhost:7800/{schema}.{table}/{z}/{x}/{y}.pbf
# http://localhost:7800/public.buildings/14/13520/6375.pbf
```

```sql
-- Create a function that pg_tileserv will auto-expose as a tile layer
CREATE OR REPLACE FUNCTION public.buildings_by_type(
    z integer, x integer, y integer,
    building_type text DEFAULT 'residential'
)
RETURNS bytea AS $$
DECLARE
    bounds geometry;
    mvt bytea;
BEGIN
    -- Convert tile coordinates to bounds
    SELECT ST_TileEnvelope(z, x, y) INTO bounds;

    SELECT ST_AsMVT(tile, 'buildings_by_type', 4096, 'geom') INTO mvt
    FROM (
        SELECT
            id,
            name,
            height,
            ST_AsMVTGeom(
                ST_Transform(geom, 3857),
                bounds,
                4096, 256, true
            ) AS geom
        FROM buildings
        WHERE geom && ST_Transform(bounds, 4326)
        AND building_type = buildings_by_type.building_type
    ) AS tile;

    RETURN mvt;
END;
$$ LANGUAGE plpgsql
STABLE
PARALLEL SAFE;

COMMENT ON FUNCTION public.buildings_by_type IS
    'Buildings filtered by type, served as vector tiles';
```

### pg_featureserv (OGC Features API)

```bash
# pg_featureserv exposes PostGIS tables as OGC API - Features endpoints
export DATABASE_URL=postgres://gis:secret@localhost:5432/gisdb
pg_featureserv

# Access features at:
# http://localhost:9000/collections
# http://localhost:9000/collections/public.buildings/items
# http://localhost:9000/collections/public.buildings/items?limit=100&bbox=116,39,117,41
# http://localhost:9000/collections/public.buildings/items?building_type=school
```

```sql
-- Create a function for pg_featureserv
CREATE OR REPLACE FUNCTION public.nearest_pois(
    lon float DEFAULT 116.4,
    lat float DEFAULT 39.9,
    num integer DEFAULT 10,
    category text DEFAULT NULL
)
RETURNS TABLE (
    id integer,
    name text,
    category text,
    distance_m float,
    geom geometry
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        p.id, p.name, p.category,
        ST_Distance(p.geom::geography, ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography) AS distance_m,
        p.geom
    FROM pois p
    WHERE (nearest_pois.category IS NULL OR p.category = nearest_pois.category)
    ORDER BY p.geom <-> ST_SetSRID(ST_MakePoint(lon, lat), 4326)
    LIMIT num;
END;
$$ LANGUAGE plpgsql
STABLE
PARALLEL SAFE;
```

### Cloud Deployment

#### AWS RDS PostGIS

```bash
# Create RDS instance with PostGIS
aws rds create-db-instance \
    --db-instance-identifier spatial-db \
    --db-instance-class db.r6g.xlarge \
    --engine postgres \
    --engine-version 16.2 \
    --master-username gis \
    --master-user-password "${DB_PASSWORD}" \
    --allocated-storage 100 \
    --storage-type gp3 \
    --db-name gisdb \
    --vpc-security-group-ids sg-xxxxx \
    --publicly-accessible
```

```sql
-- Enable PostGIS on RDS
CREATE EXTENSION postgis;
CREATE EXTENSION postgis_raster;
CREATE EXTENSION postgis_topology;
-- Note: pgRouting is available on RDS as of PostgreSQL 13+
CREATE EXTENSION pgrouting;

-- RDS-specific: grant rds_superuser for raster imports
GRANT rds_superuser TO gis;
```

#### Google Cloud SQL

```bash
# Create Cloud SQL instance
gcloud sql instances create spatial-db \
    --database-version=POSTGRES_16 \
    --tier=db-custom-4-16384 \
    --region=asia-east1 \
    --storage-size=100GB \
    --storage-type=SSD \
    --database-flags=max_parallel_workers_per_gather=4

# Enable PostGIS
gcloud sql databases create gisdb --instance=spatial-db
gcloud sql connect spatial-db --user=postgres
# Then: CREATE EXTENSION postgis;
```

#### Supabase

```sql
-- Supabase has PostGIS pre-installed
-- Enable via SQL Editor or Dashboard:
CREATE EXTENSION IF NOT EXISTS postgis WITH SCHEMA extensions;

-- Create spatial table with Row Level Security
CREATE TABLE public.field_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    description TEXT,
    geom geometry(Point, 4326),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE public.field_data ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can read own data"
    ON public.field_data FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own data"
    ON public.field_data FOR INSERT
    WITH CHECK (auth.uid() = user_id);

-- Spatial index
CREATE INDEX idx_field_data_geom ON public.field_data USING GIST (geom);

-- Query via Supabase client (PostGIS functions work in RPC calls)
-- supabase.rpc('nearby_places', { lon: 116.4, lat: 39.9, radius_m: 1000 })
CREATE OR REPLACE FUNCTION nearby_places(lon float, lat float, radius_m float)
RETURNS SETOF public.field_data AS $$
    SELECT *
    FROM public.field_data
    WHERE ST_DWithin(
        geom::geography,
        ST_SetSRID(ST_MakePoint(lon, lat), 4326)::geography,
        radius_m
    )
    ORDER BY geom <-> ST_SetSRID(ST_MakePoint(lon, lat), 4326)
$$ LANGUAGE sql STABLE;
```

---

## Cookbook

Production-ready spatial SQL recipes for common GIS tasks. Each recipe includes the problem statement, SQL solution, and performance notes.

### Recipe 1: Find All Features Within a Polygon

```sql
-- Problem: Find all POIs within a hand-drawn polygon
-- Performance: uses spatial index via ST_Intersects

SELECT p.id, p.name, p.category
FROM pois p
WHERE ST_Intersects(
    p.geom,
    ST_GeomFromGeoJSON('{
        "type": "Polygon",
        "coordinates": [[[116.3, 39.8], [116.5, 39.8],
                          [116.5, 40.0], [116.3, 40.0], [116.3, 39.8]]]
    }')
);
```

### Recipe 2: Calculate Distance Matrix

```sql
-- Problem: Distance between every pair of cities (small set)
SELECT
    a.name AS from_city,
    b.name AS to_city,
    ROUND(ST_Distance(a.geom::geography, b.geom::geography) / 1000) AS distance_km
FROM cities a
CROSS JOIN cities b
WHERE a.id < b.id
ORDER BY distance_km;
```

### Recipe 3: Geocode Addresses to Nearest Road

```sql
-- Problem: Snap address points to the nearest road segment
SELECT DISTINCT ON (a.id)
    a.id,
    a.address,
    r.road_name,
    ST_ClosestPoint(r.geom, a.geom) AS snapped_point,
    ST_Distance(a.geom::geography, r.geom::geography) AS snap_distance_m
FROM addresses a
CROSS JOIN LATERAL (
    SELECT road_name, geom
    FROM roads
    ORDER BY a.geom <-> geom
    LIMIT 5
) r
ORDER BY a.id, ST_Distance(a.geom, r.geom);
```

### Recipe 4: Generate a Regular Grid

```sql
-- Problem: Create a hexagonal or square grid covering an area
-- Square grid
WITH bounds AS (
    SELECT ST_SetSRID(ST_MakeEnvelope(116, 39, 117, 41), 4326) AS geom
),
grid AS (
    SELECT
        (ST_SquareGrid(0.01, geom)).*  -- 0.01 degree cells
    FROM bounds
)
SELECT ROW_NUMBER() OVER () AS id, geom
FROM grid;

-- Hexagonal grid (PostGIS 3.1+)
WITH bounds AS (
    SELECT ST_SetSRID(ST_MakeEnvelope(116, 39, 117, 41), 4326) AS geom
)
SELECT
    ROW_NUMBER() OVER () AS id,
    (ST_HexagonalGrid(0.01, geom)).*
FROM bounds;
```

### Recipe 5: Detect Overlapping Polygons

```sql
-- Problem: Find all pairs of parcels that overlap (data quality check)
SELECT
    a.id AS parcel_a,
    b.id AS parcel_b,
    ST_Area(ST_Intersection(a.geom, b.geom)) AS overlap_area,
    ST_Area(ST_Intersection(a.geom, b.geom)) / LEAST(ST_Area(a.geom), ST_Area(b.geom)) * 100
        AS overlap_pct
FROM parcels a
JOIN parcels b ON a.id < b.id
    AND a.geom && b.geom
    AND ST_Overlaps(a.geom, b.geom)
ORDER BY overlap_area DESC;
```

### Recipe 6: Build Service Areas (Isochrone without Network)

```sql
-- Problem: Approximate service area using buffer on road network
-- (when pgRouting is not available)
WITH road_buffer AS (
    SELECT ST_Union(ST_Buffer(geom::geography, 50)::geometry) AS service_area
    FROM roads
    WHERE ST_DWithin(
        geom::geography,
        ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)::geography,
        5000  -- 5km radius
    )
)
SELECT
    ST_ConcaveHull(service_area, 0.5) AS isochrone_approx
FROM road_buffer;
```

### Recipe 7: Line of Sight Analysis

```sql
-- Problem: Check if two points can "see" each other over terrain
-- Sample elevation along a line between two points
WITH sight_line AS (
    SELECT ST_SetSRID(
        ST_MakeLine(ST_MakePoint(116.3, 39.9), ST_MakePoint(116.5, 40.0)),
        4326
    ) AS geom
),
sample_points AS (
    SELECT
        n,
        ST_LineInterpolatePoint(sl.geom, n::float / 100) AS pt,
        n::float / 100 AS fraction
    FROM sight_line sl, generate_series(0, 100) n
),
elevations AS (
    SELECT
        sp.n,
        sp.fraction,
        ST_Value(r.rast, sp.pt) AS ground_elevation
    FROM sample_points sp
    JOIN elevation r ON ST_Intersects(r.rast, sp.pt)
),
line_of_sight AS (
    SELECT
        n, fraction, ground_elevation,
        -- Linear interpolation of sight line height
        (SELECT ground_elevation + 1.8 FROM elevations WHERE n = 0) +
        fraction * (
            (SELECT ground_elevation + 1.8 FROM elevations WHERE n = 100) -
            (SELECT ground_elevation + 1.8 FROM elevations WHERE n = 0)
        ) AS sight_height
    FROM elevations
)
SELECT
    BOOL_AND(ground_elevation < sight_height) AS is_visible,
    COUNT(*) FILTER (WHERE ground_elevation >= sight_height) AS num_obstructions
FROM line_of_sight
WHERE n > 0 AND n < 100;
```

### Recipe 8: Spatial Clustering (DBSCAN)

```sql
-- Problem: Find clusters of points using DBSCAN algorithm
-- PostGIS 3.2+ has ST_ClusterDBSCAN
SELECT
    id,
    name,
    geom,
    ST_ClusterDBSCAN(geom, eps := 0.001, minpoints := 5)
        OVER () AS cluster_id
FROM pois;

-- Get cluster summaries
WITH clustered AS (
    SELECT
        id, name, geom,
        ST_ClusterDBSCAN(geom, eps := 0.001, minpoints := 5)
            OVER () AS cluster_id
    FROM pois
)
SELECT
    cluster_id,
    COUNT(*) AS num_points,
    ST_ConvexHull(ST_Collect(geom)) AS cluster_hull,
    ST_Centroid(ST_Collect(geom)) AS cluster_center
FROM clustered
WHERE cluster_id IS NOT NULL
GROUP BY cluster_id
ORDER BY num_points DESC;

-- K-means clustering
SELECT
    id, name, geom,
    ST_ClusterKMeans(geom, 10) OVER () AS cluster_id
FROM pois;
```

### Recipe 9: Geocoding Pipeline (Batch)

```sql
-- Problem: Match addresses to known locations by proximity and name
WITH address_candidates AS (
    SELECT
        a.id AS address_id,
        a.address_text,
        r.id AS ref_id,
        r.name AS ref_name,
        ST_Distance(a.geom::geography, r.geom::geography) AS dist_m,
        similarity(a.address_text, r.name) AS name_sim,
        ROW_NUMBER() OVER (
            PARTITION BY a.id
            ORDER BY ST_Distance(a.geom, r.geom) * (1 - similarity(a.address_text, r.name))
        ) AS rank
    FROM addresses a
    CROSS JOIN LATERAL (
        SELECT id, name, geom
        FROM reference_locations
        WHERE ST_DWithin(a.geom::geography, geom::geography, 500)
        ORDER BY a.geom <-> geom
        LIMIT 10
    ) r
)
SELECT address_id, address_text, ref_name, dist_m, name_sim
FROM address_candidates
WHERE rank = 1;
```

### Recipe 10: Create Heatmap Grid

```sql
-- Problem: Generate a density heatmap grid from point data
WITH grid AS (
    SELECT (ST_SquareGrid(0.005, ST_SetSRID(ST_MakeEnvelope(116, 39, 117, 41), 4326))).*
),
density AS (
    SELECT
        g.geom AS cell_geom,
        COUNT(p.id) AS point_count,
        COALESCE(AVG(p.value), 0) AS avg_value
    FROM grid g
    LEFT JOIN pois p ON ST_Intersects(g.geom, p.geom)
    GROUP BY g.geom
)
SELECT
    cell_geom AS geom,
    point_count,
    avg_value,
    -- Normalize to 0-1 range for visualization
    point_count::float / NULLIF(MAX(point_count) OVER (), 0) AS density_normalized
FROM density
WHERE point_count > 0;
```

### Recipe 11: Watershed Delineation (Raster)

```sql
-- Problem: Delineate watershed boundary from DEM
-- Step 1: Fill sinks
WITH filled AS (
    SELECT ST_MapAlgebra(
        rast, 1,
        'CASE WHEN [rast] < (
            SELECT MIN(val) FROM (VALUES
                ([rast.x-1,y-1]), ([rast.x,y-1]), ([rast.x+1,y-1]),
                ([rast.x-1,y]),                    ([rast.x+1,y]),
                ([rast.x-1,y+1]), ([rast.x,y+1]), ([rast.x+1,y+1])
            ) AS t(val)
        ) THEN [rast] + 0.01 ELSE [rast] END'
    ) AS rast
    FROM elevation
)
-- Note: for production watershed delineation, use GRASS GIS r.watershed
-- or WhiteboxTools via command-line, then load results into PostGIS
SELECT * FROM filled;
```

### Recipe 12: Viewshed from DEM

```sql
-- Problem: Calculate visible area from a viewpoint
-- Note: PostGIS doesn't have a native viewshed function.
-- Use GRASS GIS r.viewshed or GDAL gdal_viewshed, then import.

-- After importing viewshed raster:
SELECT
    (ST_DumpAsPolygons(rast, 1)).*
FROM viewshed_result
WHERE val = 1;  -- visible pixels
```

### Recipe 13: Temporal Movement Analysis

```sql
-- Problem: Analyze vehicle movement patterns
WITH trajectory AS (
    SELECT
        vehicle_id,
        timestamp,
        geom,
        LAG(geom) OVER w AS prev_geom,
        LAG(timestamp) OVER w AS prev_ts,
        LEAD(geom) OVER w AS next_geom
    FROM gps_tracks
    WINDOW w AS (PARTITION BY vehicle_id ORDER BY timestamp)
),
segments AS (
    SELECT
        vehicle_id,
        timestamp,
        geom,
        ST_MakeLine(prev_geom, geom) AS segment,
        ST_Distance(prev_geom::geography, geom::geography) AS dist_m,
        EXTRACT(EPOCH FROM timestamp - prev_ts) AS dt_sec
    FROM trajectory
    WHERE prev_geom IS NOT NULL
)
SELECT
    vehicle_id,
    DATE(timestamp) AS trip_date,
    COUNT(*) AS num_points,
    SUM(dist_m) / 1000 AS total_km,
    MAX(dist_m / NULLIF(dt_sec, 0)) * 3.6 AS max_speed_kmh,
    AVG(dist_m / NULLIF(dt_sec, 0)) * 3.6 AS avg_speed_kmh,
    ST_MakeLine(geom ORDER BY timestamp) AS trajectory_line
FROM segments
GROUP BY vehicle_id, DATE(timestamp);
```

### Recipe 14: Find Gaps and Overlaps in Polygon Coverage

```sql
-- Problem: Quality check - find gaps and overlaps in a parcel dataset
-- Gaps: areas within the boundary but not covered by any parcel
WITH boundary AS (
    SELECT ST_Union(geom) AS geom FROM parcels
),
coverage AS (
    SELECT ST_Union(geom) AS geom FROM parcels
)
SELECT
    ST_Difference(
        ST_ConvexHull(b.geom),  -- or use a known boundary polygon
        c.geom
    ) AS gaps
FROM boundary b, coverage c;

-- Overlaps: areas covered by more than one parcel
SELECT
    a.id AS parcel_a,
    b.id AS parcel_b,
    ST_Intersection(a.geom, b.geom) AS overlap_geom,
    ST_Area(ST_Intersection(a.geom, b.geom)) AS overlap_area
FROM parcels a
JOIN parcels b ON a.id < b.id
    AND ST_Intersects(a.geom, b.geom)
    AND NOT ST_Touches(a.geom, b.geom);  -- exclude shared edges
```

### Recipe 15: Proportional Split (Areal Interpolation)

```sql
-- Problem: Distribute population from census tracts to grid cells
-- based on area overlap proportion
SELECT
    g.id AS grid_id,
    SUM(
        c.population * ST_Area(ST_Intersection(g.geom, c.geom)) / ST_Area(c.geom)
    ) AS estimated_population
FROM grid_cells g
JOIN census_tracts c ON ST_Intersects(g.geom, c.geom)
GROUP BY g.id;
```

### Recipe 16: Longest Line Within Polygon (Interior Diameter)

```sql
-- Problem: Find the longest straight line that fits inside a polygon
-- Approximate using the longest axis of the minimum bounding rectangle
SELECT
    id,
    ST_Length(
        ST_LongestLine(
            ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(geom)), 1),
            ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(geom)), 3)
        )
    ) AS approx_diameter,
    ST_LongestLine(
        ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(geom)), 1),
        ST_PointN(ST_ExteriorRing(ST_OrientedEnvelope(geom)), 3)
    ) AS diameter_line
FROM parcels;
```

### Recipe 17: Extract Road Intersections

```sql
-- Problem: Find all road intersections (nodes where roads meet)
SELECT
    (ST_DumpPoints(
        ST_Intersection(a.geom, b.geom)
    )).geom AS intersection_point,
    a.road_name AS road_a,
    b.road_name AS road_b
FROM roads a
JOIN roads b ON a.id < b.id
    AND ST_Intersects(a.geom, b.geom)
    AND NOT ST_Touches(a.geom, b.geom);  -- exclude end-point touches
```

### Recipe 18: Building Shadow Calculation

```sql
-- Problem: Calculate building shadow footprint given sun azimuth and altitude
-- sun_azimuth: degrees from north, sun_altitude: degrees above horizon
WITH params AS (
    SELECT
        radians(225) AS azimuth,    -- SW sun
        radians(30) AS altitude      -- 30 degrees above horizon
),
shadows AS (
    SELECT
        b.id,
        b.name,
        b.height,
        -- Shadow length = building height / tan(sun altitude)
        b.height / tan(p.altitude) AS shadow_length,
        -- Shadow direction (opposite of sun azimuth)
        ST_Translate(
            b.geom,
            -sin(p.azimuth) * (b.height / tan(p.altitude)),
            -cos(p.azimuth) * (b.height / tan(p.altitude))
        ) AS shadow_footprint
    FROM buildings b, params p
    WHERE b.height > 0
)
SELECT
    id, name, height, shadow_length,
    ST_ConvexHull(ST_Union(geom, shadow_footprint)) AS shadow_polygon
FROM shadows
CROSS JOIN LATERAL (SELECT geom FROM buildings WHERE buildings.id = shadows.id) orig;
```

### Recipe 19: Multi-Criteria Site Selection

```sql
-- Problem: Find optimal locations for a new facility
-- Criteria: within 500m of transit, >1km from competitors,
--           in commercial zone, parcel > 500 sqm
WITH candidates AS (
    SELECT
        p.id,
        p.geom,
        p.area_sqm,
        -- Distance to nearest transit stop
        (SELECT MIN(ST_Distance(p.geom::geography, t.geom::geography))
         FROM transit_stops t) AS transit_dist,
        -- Distance to nearest competitor
        (SELECT MIN(ST_Distance(p.geom::geography, c.geom::geography))
         FROM competitors c) AS competitor_dist,
        -- Is in commercial zone?
        EXISTS (
            SELECT 1 FROM zoning z
            WHERE z.zone_type = 'commercial'
            AND ST_Contains(z.geom, p.geom)
        ) AS in_commercial
    FROM parcels p
    WHERE p.area_sqm >= 500
)
SELECT
    id,
    area_sqm,
    transit_dist,
    competitor_dist,
    -- Composite score (lower is better)
    (transit_dist / 500.0) +
    (1.0 - LEAST(competitor_dist / 2000.0, 1.0)) +
    CASE WHEN in_commercial THEN 0 ELSE 1 END AS score
FROM candidates
WHERE transit_dist <= 500
AND competitor_dist >= 1000
AND in_commercial
ORDER BY score
LIMIT 20;
```

### Recipe 20: Generate Contour Lines from Point Samples

```sql
-- Problem: Create contour lines from scattered elevation measurements
-- Step 1: Create a Delaunay TIN
WITH tin AS (
    SELECT ST_DelaunayTriangles(ST_Collect(geom)) AS geom
    FROM elevation_samples
),
-- Step 2: For each triangle, interpolate contour crossing points
-- Note: full contour generation is complex in pure SQL.
-- Recommended approach: use ST_MapAlgebra on rasterized surface
rasterized AS (
    SELECT ST_AsRaster(
        geom,
        0.001,    -- pixel size X
        0.001,    -- pixel size Y
        '32BF'    -- pixel type
    ) AS rast
    FROM tin
)
-- Step 3: Generate contours from raster
SELECT (ST_Contour(rast, 1,
    ARRAY[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
)).*
FROM rasterized;
```

### Recipe 21: Polygon Smoothing and Generalization

```sql
-- Problem: Smooth jagged polygon boundaries for cartographic display
-- Chaikin smoothing (corner-cutting algorithm)
SELECT
    id,
    ST_ChaikinSmoothing(geom, 3) AS smooth_geom  -- 3 iterations
FROM parcels;

-- Scale-dependent simplification
SELECT
    id,
    CASE
        WHEN :zoom < 8 THEN ST_Simplify(geom, 0.01)
        WHEN :zoom < 12 THEN ST_Simplify(geom, 0.001)
        WHEN :zoom < 15 THEN ST_Simplify(geom, 0.0001)
        ELSE geom
    END AS display_geom
FROM parcels;

-- Topology-preserving simplification (no self-intersections)
SELECT
    id,
    ST_SimplifyPreserveTopology(geom, 0.001) AS simplified
FROM parcels;

-- Visvalingam-Whyatt (better for natural features)
SELECT
    id,
    ST_SimplifyVW(geom, 0.0001) AS simplified_vw
FROM coastlines;
```

### Recipe 22: Spatial Data Quality Report

```sql
-- Problem: Generate a comprehensive quality report for a spatial dataset
SELECT
    'Total features' AS metric, COUNT(*)::text AS value FROM parcels
UNION ALL
SELECT
    'Invalid geometries', COUNT(*)::text FROM parcels WHERE NOT ST_IsValid(geom)
UNION ALL
SELECT
    'Empty geometries', COUNT(*)::text FROM parcels WHERE ST_IsEmpty(geom)
UNION ALL
SELECT
    'Null geometries', COUNT(*)::text FROM parcels WHERE geom IS NULL
UNION ALL
SELECT
    'Duplicate geometries',
    COUNT(*)::text
FROM (
    SELECT geom, COUNT(*) AS cnt
    FROM parcels
    GROUP BY ST_AsBinary(geom)
    HAVING COUNT(*) > 1
) dupes
UNION ALL
SELECT
    'Self-intersecting polygons',
    COUNT(*)::text
FROM parcels
WHERE NOT ST_IsSimple(geom)
UNION ALL
SELECT
    'Min area (sqm)',
    ROUND(MIN(ST_Area(geom::geography))::numeric, 2)::text
FROM parcels
UNION ALL
SELECT
    'Max area (sqm)',
    ROUND(MAX(ST_Area(geom::geography))::numeric, 2)::text
FROM parcels
UNION ALL
SELECT
    'Total extent',
    ST_AsText(ST_Extent(geom))
FROM parcels
UNION ALL
SELECT
    'SRID',
    ST_SRID(geom)::text
FROM parcels LIMIT 1;
```

### Recipe 23: Dynamic MVT Tile Generation

```sql
-- Problem: Generate Mapbox Vector Tiles on the fly from PostGIS
-- This function can be called from any tile server or API endpoint
CREATE OR REPLACE FUNCTION get_tile(z int, x int, y int)
RETURNS bytea AS $$
DECLARE
    bounds geometry;
    mvt bytea;
BEGIN
    bounds := ST_TileEnvelope(z, x, y);

    WITH layers AS (
        -- Buildings layer (visible at zoom >= 14)
        SELECT
            ST_AsMVTGeom(
                ST_Transform(geom, 3857), bounds, 4096, 64, true
            ) AS geom,
            id, name, height, building_type,
            'buildings' AS layer
        FROM buildings
        WHERE z >= 14
        AND geom && ST_Transform(bounds, 4326)

        UNION ALL

        -- Roads layer (visible at zoom >= 10)
        SELECT
            ST_AsMVTGeom(
                ST_Transform(geom, 3857), bounds, 4096, 64, true
            ) AS geom,
            id, road_name AS name, 0 AS height, road_class AS building_type,
            'roads' AS layer
        FROM roads
        WHERE z >= 10
        AND geom && ST_Transform(bounds, 4326)

        UNION ALL

        -- POIs layer (visible at zoom >= 12)
        SELECT
            ST_AsMVTGeom(
                ST_Transform(geom, 3857), bounds, 4096, 64, true
            ) AS geom,
            id, name, 0 AS height, category AS building_type,
            'pois' AS layer
        FROM pois
        WHERE z >= 12
        AND geom && ST_Transform(bounds, 4326)
    )
    SELECT STRING_AGG(tile, ''::bytea) INTO mvt
    FROM (
        SELECT ST_AsMVT(t, layer, 4096, 'geom') AS tile
        FROM layers t
        GROUP BY layer
    ) sub;

    RETURN mvt;
END;
$$ LANGUAGE plpgsql
STABLE
PARALLEL SAFE;
```

### Recipe 24: Spatial Autocorrelation (Moran's I)

```sql
-- Problem: Calculate Global Moran's I for a spatial variable
-- (measures spatial clustering of values)
WITH data AS (
    SELECT
        id, geom,
        value,
        AVG(value) OVER () AS mean_val,
        COUNT(*) OVER () AS n
    FROM observations
),
weights AS (
    -- Queen contiguity weights (neighbors share edge or vertex)
    SELECT
        a.id AS id_i,
        b.id AS id_j,
        1.0 AS w_ij
    FROM data a
    JOIN data b ON a.id != b.id
        AND ST_Touches(a.geom, b.geom)
),
components AS (
    SELECT
        SUM(w.w_ij * (d1.value - d1.mean_val) * (d2.value - d2.mean_val)) AS numerator,
        SUM((d1.value - d1.mean_val)^2) AS denominator,
        SUM(w.w_ij) AS sum_w,
        d1.n AS n
    FROM weights w
    JOIN data d1 ON w.id_i = d1.id
    JOIN data d2 ON w.id_j = d2.id
    GROUP BY d1.n
)
SELECT
    (n / sum_w) * (numerator / denominator) AS morans_i
FROM components;
-- Moran's I near +1 = clustered, near 0 = random, near -1 = dispersed
```

---

## Cross-References

Related guides in this repository:

| Guide | Relevance |
|---|---|
| [tools/spatial-databases.md](../tools/spatial-databases.md) | Database software overview, installation guides, comparison tables |
| [tools/server-publishing.md](../tools/server-publishing.md) | GeoServer, MapServer, pg_tileserv, pg_featureserv, QGIS Server deployment |
| [data-analysis/python-stack.md](python-stack.md) | GeoPandas, GeoAlchemy2, DuckDB Python bindings for spatial SQL integration |
| [data-analysis/spatial-statistics.md](spatial-statistics.md) | Statistical methods (point patterns, interpolation, regression) that complement SQL analytics |
| [data-analysis/r-stack.md](r-stack.md) | R sf package with DBI/RPostgis for spatial SQL from R |
| [data-sources/README.md](../data-sources/README.md) | Where to find open spatial datasets to load into your spatial database |

---

> **Further Reading**
> - [PostGIS Official Documentation](https://postgis.net/documentation/) - complete function reference
> - [PostGIS Introduction Tutorial](https://postgis.net/workshops/postgis-intro/) - the definitive PostGIS workshop
> - [DuckDB Spatial Documentation](https://duckdb.org/docs/extensions/spatial.html) - extension reference
> - [pgRouting Documentation](https://docs.pgrouting.org/) - network analysis functions
> - [SpatiaLite Cookbook](https://www.gaia-gis.it/fossil/libspatialite/wiki?name=SpatiaLite+Cookbook) - SpatiaLite tutorials
> - [Crunchy Data PostGIS Guides](https://www.crunchydata.com/developers/tutorials) - practical PostGIS patterns
> - [Modern SQL for GIS](https://blog.crunchydata.com/blog/spatial-analytics-with-modern-sql) - advanced SQL patterns for spatial work
