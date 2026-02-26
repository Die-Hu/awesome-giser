# Performance Optimization — 2025 Complete Guide

Geospatial applications often deal with large datasets, complex geometries, and real-time rendering demands. This guide covers optimization strategies across the entire stack — from database queries to browser rendering.

> **Quick Picks**
> - **Zero-infra tile serving:** PMTiles on CDN (Cloudflare R2 for free egress)
> - **Vector tile generation:** Tippecanoe 2.x (the standard tool)
> - **Cloud raster access:** Cloud-Optimized GeoTIFF with HTTP range requests
> - **Client-side geo processing:** WebAssembly (geos-wasm, DuckDB-WASM)
> - **Fastest rendering:** MapLibre GL JS 4.x or deck.gl 9.x (WebGL/WebGPU)

---

## Vector Tile Pipeline

### Tippecanoe 2.x — Master Recipes

```bash
# Recipe 1: Buildings — zoom-dependent detail
tippecanoe -o buildings.pmtiles \
  --minimum-zoom=10 --maximum-zoom=15 \
  --simplification=10 \
  --drop-smallest-as-needed \
  --include=name,height,type,area \
  --layer=buildings \
  buildings.geojson

# Recipe 2: Roads — importance-based filtering
tippecanoe -o roads.pmtiles \
  --minimum-zoom=4 --maximum-zoom=14 \
  -j '{ "*": { "minzoom_by_attribute": { "highway": { "motorway": 4, "trunk": 6, "primary": 8, "secondary": 10, "tertiary": 12, "residential": 13 } } } }' \
  --coalesce-smallest-as-needed \
  roads.geojson

# Recipe 3: Points — clustering at low zooms
tippecanoe -o poi.pmtiles \
  --minimum-zoom=2 --maximum-zoom=14 \
  --cluster-distance=50 \
  --cluster-maxzoom=12 \
  --accumulate-attribute=count:sum \
  --include=name,category,count \
  poi.geojson

# Recipe 4: Admin boundaries — topology-preserving
tippecanoe -o admin.pmtiles \
  --minimum-zoom=0 --maximum-zoom=12 \
  --simplification=10 \
  --detect-shared-borders \
  --no-tiny-polygon-reduction \
  --include=name,admin_level,iso_code \
  admin_boundaries.geojson

# Recipe 5: Water polygons — aggressive simplification
tippecanoe -o water.pmtiles \
  --minimum-zoom=0 --maximum-zoom=14 \
  --simplification=20 \
  --drop-densest-as-needed \
  --buffer=8 \
  water.geojson

# Recipe 6: Multi-layer basemap
tippecanoe -o basemap.pmtiles \
  --named-layer=buildings:buildings.fgb \
  --named-layer=roads:roads.fgb \
  --named-layer=water:water.fgb \
  --named-layer=landuse:landuse.fgb \
  --named-layer=poi:poi.fgb \
  --minimum-zoom=0 --maximum-zoom=14 \
  --read-parallel

# Recipe 7: FlatGeobuf input (2-5x faster than GeoJSON)
ogr2ogr -f FlatGeobuf buildings.fgb buildings.geojson
tippecanoe -o buildings.pmtiles --read-parallel buildings.fgb

# Recipe 8: Tile statistics and size limits
tippecanoe -o output.pmtiles \
  --minimum-zoom=0 --maximum-zoom=14 \
  --maximum-tile-bytes=500000 \
  --maximum-tile-features=200000 \
  --stats \
  input.geojson 2>&1 | tee stats.log

# Recipe 9: Overzooming (serve z14 data at z15-22)
tippecanoe -o data.pmtiles \
  --minimum-zoom=0 --maximum-zoom=14 \
  --extra-detail=12 \
  input.geojson
# MapLibre: set "maxzoom": 14 in source, tiles overzoom client-side

# Recipe 10: Large dataset with progress reporting
tippecanoe -o output.pmtiles \
  --minimum-zoom=0 --maximum-zoom=14 \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  --force \
  --progress-interval=10 \
  huge-dataset.fgb
```

### PMTiles Architecture

```
PMTiles file structure:
┌─────────────────────────────────┐
│  Header (127 bytes)             │  ← Fixed metadata
├─────────────────────────────────┤
│  Root Directory (variable)      │  ← Top-level tile index
├─────────────────────────────────┤
│  Leaf Directories (variable)    │  ← Tile offset/length entries
├─────────────────────────────────┤
│  Tile Data (bulk)               │  ← Actual MVT/PNG/WebP data
├─────────────────────────────────┤
│  Metadata JSON                  │  ← TileJSON-compatible
└─────────────────────────────────┘

HTTP range request flow:
1. Client reads header (0-127 bytes) → gets root directory offset
2. Client reads root directory → finds leaf directory for z/x/y
3. Client reads leaf directory → finds tile offset + length
4. Client reads tile data → renders
Total: 2-3 HTTP requests per tile (first load), 1 after caching directories
```

```javascript
// PMTiles + MapLibre — Complete setup
import { Protocol } from 'pmtiles';
import maplibregl from 'maplibre-gl';

const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    glyphs: 'https://fonts.example.com/{fontstack}/{range}.pbf',
    sources: {
      basemap: {
        type: 'vector',
        url: 'pmtiles://https://r2.example.com/basemap.pmtiles',
        maxzoom: 14, // Enable client-side overzooming beyond z14
      },
      satellite: {
        type: 'raster',
        url: 'pmtiles://https://r2.example.com/satellite.pmtiles',
        tileSize: 512,
      },
    },
    layers: [
      { id: 'water', type: 'fill', source: 'basemap', 'source-layer': 'water',
        paint: { 'fill-color': '#a0d0f0' } },
      { id: 'landuse', type: 'fill', source: 'basemap', 'source-layer': 'landuse',
        paint: { 'fill-color': ['match', ['get', 'type'],
          'park', '#c8e6c9', 'forest', '#a5d6a7', '#e8e8e8'] } },
      { id: 'roads', type: 'line', source: 'basemap', 'source-layer': 'roads',
        paint: { 'line-color': '#999', 'line-width': ['interpolate', ['linear'], ['zoom'],
          8, 0.5, 14, 3] } },
      { id: 'buildings', type: 'fill', source: 'basemap', 'source-layer': 'buildings',
        minzoom: 13,
        paint: { 'fill-color': '#ddd', 'fill-outline-color': '#bbb' } },
    ],
  },
});
```

### Martin Performance Tuning

```yaml
# martin-config.yaml — Optimized for production
listen_addresses: '0.0.0.0:3000'
worker_processes: 8            # Match CPU cores
keep_alive: 75
cache_size_mb: 1024            # 1 GB tile cache

postgres:
  connection_string: 'postgresql://user:pass@localhost:5432/gisdb?
    application_name=martin&
    connect_timeout=10&
    keepalives=1&
    keepalives_idle=30'
  pool_size: 30                # 2x expected concurrent tile requests
  auto_publish:
    tables:
      from_schemas: ['public']
```

### Pre-Generated vs On-the-Fly vs Hybrid

| Approach | Latency | Storage | Update Speed | Best For |
|----------|---------|---------|-------------|----------|
| Pre-generated (PMTiles) | 5-20ms | High | Slow (regenerate) | Static data, CDN serving |
| On-the-fly (Martin) | 50-500ms | None | Instant | Dynamic data, PostGIS |
| Hybrid (Martin + cache) | 5-50ms | Medium | Minutes (cache TTL) | Production apps |

```nginx
# Hybrid approach: Nginx cache in front of Martin
proxy_cache_path /var/cache/tiles levels=1:2 keys_zone=tiles:100m max_size=50g inactive=30d;

location /tiles/ {
    proxy_pass http://martin:3000/;
    proxy_cache tiles;
    proxy_cache_valid 200 24h;
    proxy_cache_valid 404 1m;
    proxy_cache_use_stale error timeout updating;
    proxy_cache_lock on;
    add_header X-Cache-Status $upstream_cache_status;
    add_header Cache-Control "public, max-age=86400";
    add_header Access-Control-Allow-Origin "*";
}
```

---

## Raster Optimization

### Cloud Optimized GeoTIFF (COG) Deep Dive

```
COG internal structure:
┌─────────────────────────────────────────┐
│  IFD 0: Full resolution (overview 0)    │
│  ┌─────┬─────┬─────┬─────┬─────┐       │
│  │Tile │Tile │Tile │Tile │Tile │       │
│  │ 0,0 │ 1,0 │ 2,0 │ 3,0 │ ... │       │
│  ├─────┼─────┼─────┼─────┼─────┤       │
│  │Tile │Tile │Tile │Tile │     │       │
│  │ 0,1 │ 1,1 │ 2,1 │ 3,1 │ ... │       │
│  └─────┴─────┴─────┴─────┴─────┘       │
├─────────────────────────────────────────┤
│  IFD 1: Overview 1 (2x smaller)         │
│  ┌─────┬─────┬─────┐                    │
│  │     │     │     │                    │
│  └─────┴─────┴─────┘                    │
├─────────────────────────────────────────┤
│  IFD 2: Overview 2 (4x smaller)         │
│  ┌─────┐                                │
│  │     │                                │
│  └─────┘                                │
└─────────────────────────────────────────┘

HTTP range request: read specific tiles by byte offset
→ No need to download entire file
→ Ideal for S3/GCS/R2 object storage
```

```bash
# Create optimized COG
gdal_translate input.tif output_cog.tif \
  -of COG \
  -co COMPRESS=ZSTD \
  -co PREDICTOR=2 \
  -co OVERVIEW_RESAMPLING=AVERAGE \
  -co BLOCKSIZE=512 \
  -co NUM_THREADS=ALL_CPUS \
  -co BIGTIFF=IF_SAFER

# Validate COG
python -c "
from osgeo import gdal
ds = gdal.Open('output_cog.tif')
md = ds.GetMetadata('IMAGE_STRUCTURE')
print('Layout:', md.get('LAYOUT', 'Not COG'))
print('Compression:', md.get('COMPRESSION', 'None'))
"

# Add overviews if missing
gdaladdo -r average output_cog.tif 2 4 8 16 32

# Inspect COG structure
gdalinfo output_cog.tif -json | python -m json.tool
```

### Compression Benchmarks (1 GB uncompressed)

| Compression | Size | Read Speed | Ratio | Best For |
|-------------|------|-----------|-------|----------|
| None | 1000 MB | Fastest | 1.0x | Local SSD |
| LZW | 450 MB | Fast | 2.2x | General |
| ZSTD | 380 MB | Fast | 2.6x | Modern (best ratio+speed) |
| DEFLATE | 420 MB | Medium | 2.4x | Max compatibility |
| JPEG (lossy) | 120 MB | Fast | 8.3x | RGB imagery |
| LERC | 350 MB | Fast | 2.9x | Floating-point elevation |
| WEBP (lossy) | 100 MB | Fast | 10.0x | Visual tiles |

### TiTiler Optimization

```python
# Optimized TiTiler with caching
from titiler.core.factory import TilerFactory
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from starlette_cramjam.middleware import CompressionMiddleware
import aiocache

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
app.add_middleware(CompressionMiddleware)

# Configure caching
cache = aiocache.Cache(aiocache.Cache.REDIS, endpoint="localhost", port=6379)

cog = TilerFactory(
    router_prefix="/cog",
)
app.include_router(cog.router, prefix="/cog")

# Environment variables for optimization
# GDAL_CACHEMAX=512           # 512 MB GDAL block cache
# GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR  # Skip directory listing
# CPL_VSIL_CURL_ALLOWED_EXTENSIONS=.tif,.tiff  # Limit file extensions
# VSI_CACHE=TRUE              # Enable VSI caching
# VSI_CACHE_SIZE=536870912    # 512 MB VSI cache
```

---

## Database Performance

### PostGIS Tuning — postgresql.conf

```ini
# Memory (for 16 GB RAM server)
shared_buffers = 4GB           # 25% of RAM
effective_cache_size = 12GB    # 75% of RAM
work_mem = 256MB               # Per-operation memory (spatial ops are memory-hungry)
maintenance_work_mem = 2GB     # For VACUUM, CREATE INDEX

# Parallelism
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
parallel_tuple_cost = 0.01
parallel_setup_cost = 100

# WAL (write-ahead log)
wal_buffers = 64MB
checkpoint_completion_target = 0.9
max_wal_size = 4GB

# Planner
random_page_cost = 1.1         # SSD storage (default 4.0 for HDD)
effective_io_concurrency = 200  # SSD
default_statistics_target = 500 # Better stats for spatial data

# PostGIS specific
postgis.gdal_enabled_drivers = 'GTiff PNG JPEG'
postgis.enable_outdb_rasters = true
```

### Index Strategies

```sql
-- 1. Standard GiST (default, best for general queries)
CREATE INDEX idx_geom ON features USING GIST (geom);

-- 2. Partial index (filter first, then spatial)
CREATE INDEX idx_buildings_geom ON features USING GIST (geom)
  WHERE type = 'building';

-- 3. Covering index (include commonly selected columns)
CREATE INDEX idx_geom_covering ON features USING GIST (geom)
  INCLUDE (name, category);

-- 4. BRIN index (for large time-series with spatial locality)
-- 100x smaller than GiST, but slower queries
CREATE INDEX idx_brin ON sensor_readings USING BRIN (geom, timestamp)
  WITH (pages_per_range = 128);

-- 5. SP-GiST (for point data, quadtree-like)
CREATE INDEX idx_spgist ON points USING SPGIST (geom);

-- When to use which:
-- GiST:    General purpose, polygons, lines, mixed geometry
-- SP-GiST: Points only, kNN queries
-- BRIN:    Very large tables (>100M rows), data ordered spatially/temporally
```

### Query Optimization — EXPLAIN ANALYZE Examples

```sql
-- BAD: Full table scan (no spatial filter)
EXPLAIN ANALYZE
SELECT * FROM buildings WHERE name LIKE '%school%';
-- Seq Scan on buildings  (cost=0..15000  rows=50  actual time=0.2..120ms)

-- GOOD: Spatial filter first, then attribute filter
EXPLAIN ANALYZE
SELECT * FROM buildings
WHERE geom && ST_MakeEnvelope(116.3, 39.8, 116.5, 40.0, 4326)
  AND name LIKE '%school%';
-- Bitmap Index Scan on idx_geom  (cost=0..50  actual time=0.1..5ms)

-- BAD: ST_Distance < threshold (no index usage)
EXPLAIN ANALYZE
SELECT * FROM poi
WHERE ST_Distance(geom, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)) < 0.01;
-- Seq Scan (no index!)

-- GOOD: ST_DWithin (uses spatial index)
EXPLAIN ANALYZE
SELECT * FROM poi
WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326), 0.01);
-- Index Scan using idx_poi_geom

-- GOOD: kNN with <-> operator (uses GiST index)
EXPLAIN ANALYZE
SELECT *, geom <-> ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326) AS dist
FROM poi
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)
LIMIT 10;
-- Index Scan using idx_poi_geom (KNN)

-- Tile query optimization (what Martin does internally)
EXPLAIN ANALYZE
SELECT ST_AsMVT(q, 'buildings', 4096, 'geom') FROM (
  SELECT id, name, height,
    ST_AsMVTGeom(ST_Transform(geom, 3857), ST_TileEnvelope(14, 13527, 6197), 4096, 64, true) AS geom
  FROM buildings
  WHERE geom && ST_Transform(ST_TileEnvelope(14, 13527, 6197), 4326)
) q;

-- Simplify at lower zoom levels
SELECT ST_AsMVT(q, 'buildings', 4096, 'geom') FROM (
  SELECT id, name,
    ST_AsMVTGeom(
      ST_SimplifyPreserveTopology(ST_Transform(geom, 3857), 10),
      bounds, 4096, 64, true
    ) AS geom
  FROM buildings
  WHERE geom && ST_Transform(bounds, 4326)
    AND ST_Area(geom) > 0.000001  -- Skip tiny buildings at low zoom
) q;
```

### Partitioning Strategies

```sql
-- Time-based partitioning (sensor data)
CREATE TABLE observations (
    id BIGSERIAL,
    sensor_id INT,
    value FLOAT,
    geom GEOMETRY(Point, 4326),
    observed_at TIMESTAMPTZ NOT NULL
) PARTITION BY RANGE (observed_at);

-- Monthly partitions
CREATE TABLE obs_2024_01 PARTITION OF observations
    FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
-- ... create for each month

-- Auto-create partitions with pg_partman
CREATE EXTENSION pg_partman;
SELECT partman.create_parent('public.observations', 'observed_at', 'native', 'monthly');

-- Each partition gets its own spatial index
-- Query planner prunes partitions based on WHERE clause
```

### DuckDB Spatial — When to Use Instead of PostGIS

```sql
-- DuckDB for analytical queries on GeoParquet
INSTALL spatial; LOAD spatial;

-- Load GeoParquet (no import needed!)
SELECT COUNT(*), AVG(height), ST_Area(ST_Union_Agg(geom)) / 1e6 as total_km2
FROM read_parquet('s3://data/buildings.parquet')
WHERE category = 'commercial';

-- Performance comparison (1M buildings, analytical query):
-- PostGIS: ~5 seconds
-- DuckDB:  ~0.8 seconds (columnar + vectorized execution)

-- DuckDB shines for:
-- 1. Analytical aggregations on large datasets
-- 2. Reading directly from GeoParquet/CSV without import
-- 3. Client-side analysis via DuckDB-WASM
-- 4. One-off queries without database setup
```

---

## Frontend Rendering

### WebGL Pipeline — How MapLibre Renders

```
Tile request → Decode MVT → Parse geometry → Build buffers → GPU upload → Render

1. Tile fetched (HTTP/PMTiles)
2. Protobuf decoded to geometry arrays
3. Geometry tessellated (polygons → triangles)
4. Vertex/index buffers built
5. Uploaded to GPU memory (WebGL buffers)
6. Rendered with shaders per frame
7. Labels placed (collision detection)
```

### deck.gl Optimization

```javascript
// 1. Binary data instead of JSON (10x faster parsing)
const layer = new ScatterplotLayer({
  id: 'points',
  data: {
    length: 1000000,
    attributes: {
      getPosition: new Float32Array(buffer), // [lng, lat, ...]
      getFillColor: new Uint8Array(colorBuffer),
    },
  },
  getRadius: 50,
});

// 2. GPU data filtering (filter without re-uploading data)
import { DataFilterExtension } from '@deck.gl/extensions';

const layer = new ScatterplotLayer({
  data: points,
  getFilterValue: d => d.timestamp,
  filterRange: [startTime, endTime],
  extensions: [new DataFilterExtension({ filterSize: 1 })],
  updateTriggers: { getFilterValue: [startTime, endTime] },
});

// 3. Layer update optimization
// GOOD: Update only changed props
layer.clone({ data: newData });

// BAD: Create new layer instance every render
// const layer = new ScatterplotLayer({ ... }); // in render function
```

### Web Workers for Spatial Computation

```javascript
// spatialWorker.js
import * as turf from '@turf/turf';

self.onmessage = (e) => {
  const { type, payload } = e.data;

  switch (type) {
    case 'buffer': {
      const result = turf.buffer(payload.geojson, payload.distance, {
        units: payload.units || 'kilometers',
      });
      self.postMessage({ type: 'buffer-result', payload: result });
      break;
    }
    case 'simplify': {
      const result = turf.simplify(payload.geojson, {
        tolerance: payload.tolerance,
        highQuality: true,
      });
      self.postMessage({ type: 'simplify-result', payload: result });
      break;
    }
    case 'cluster': {
      const result = turf.clustersKmeans(payload.points, {
        numberOfClusters: payload.k,
      });
      self.postMessage({ type: 'cluster-result', payload: result });
      break;
    }
  }
};

// main.js — Worker pool pattern
class SpatialWorkerPool {
  constructor(size = navigator.hardwareConcurrency || 4) {
    this.workers = Array.from({ length: size },
      () => new Worker(new URL('./spatialWorker.js', import.meta.url), { type: 'module' })
    );
    this.queue = [];
    this.available = [...this.workers];
  }

  async execute(type, payload) {
    return new Promise((resolve) => {
      const run = (worker) => {
        worker.onmessage = (e) => {
          resolve(e.data.payload);
          this.available.push(worker);
          if (this.queue.length > 0) {
            const next = this.queue.shift();
            next();
          }
        };
        worker.postMessage({ type, payload });
      };

      if (this.available.length > 0) {
        run(this.available.pop());
      } else {
        this.queue.push(() => run(this.available.pop()));
      }
    });
  }
}

const pool = new SpatialWorkerPool();
const buffered = await pool.execute('buffer', { geojson: feature, distance: 5 });
```

### DuckDB-WASM — Client-Side Spatial SQL

```javascript
import * as duckdb from '@duckdb/duckdb-wasm';

// Initialize DuckDB in browser
const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);
const worker = new Worker(bundle.mainWorker);
const logger = new duckdb.ConsoleLogger();
const db = new duckdb.AsyncDuckDB(logger, worker);
await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

const conn = await db.connect();
await conn.query('INSTALL spatial; LOAD spatial;');

// Load GeoParquet from URL
await conn.query(`
  CREATE TABLE buildings AS
  SELECT * FROM read_parquet('https://data.example.com/buildings.parquet');
`);

// Spatial query in browser!
const result = await conn.query(`
  SELECT category, COUNT(*) as count, AVG(height) as avg_height
  FROM buildings
  WHERE ST_Within(geom, ST_GeomFromText('POLYGON((116.3 39.8, 116.5 39.8, 116.5 40.0, 116.3 40.0, 116.3 39.8))'))
  GROUP BY category
  ORDER BY count DESC
`);
```

### FlatGeobuf — Streaming Vector Data

```javascript
import { deserialize } from 'flatgeobuf/lib/mjs/geojson.js';

// Stream features within bbox (only fetches needed bytes)
async function loadFeaturesInBBox(url, bbox) {
  const iter = deserialize(url, {
    minX: bbox[0], minY: bbox[1], maxX: bbox[2], maxY: bbox[3],
  });

  const features = [];
  for await (const feature of iter) {
    features.push(feature);
  }

  return { type: 'FeatureCollection', features };
}

// Usage with MapLibre
const geojson = await loadFeaturesInBBox(
  'https://data.example.com/buildings.fgb',
  [116.3, 39.8, 116.5, 40.0]
);
map.getSource('buildings')?.setData(geojson);
```

---

## Network & Caching

### CDN Configuration for Tiles

```nginx
# Cloudflare: Set page rules or Transform Rules
# Match: /tiles/*
# Cache Level: Cache Everything
# Edge Cache TTL: 1 month
# Browser Cache TTL: 1 day

# AWS CloudFront: Cache policy for tiles
# TTL: min=86400, default=2592000, max=31536000
# Accept-Encoding: gzip, br
# Query strings: None (ignore for tile caching)
```

### Service Worker Tile Caching — Workbox

```javascript
// sw.js — Service Worker with Workbox
import { registerRoute } from 'workbox-routing';
import { CacheFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';

// Cache basemap tiles (rarely change)
registerRoute(
  ({ url }) => url.pathname.match(/\/tiles\/.*\/\d+\/\d+\/\d+/),
  new CacheFirst({
    cacheName: 'map-tiles',
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({
        maxEntries: 10000,
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
      }),
    ],
  })
);

// Cache API responses (change periodically)
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/features'),
  new StaleWhileRevalidate({
    cacheName: 'feature-api',
    plugins: [
      new CacheableResponsePlugin({ statuses: [200] }),
      new ExpirationPlugin({
        maxEntries: 500,
        maxAgeSeconds: 5 * 60, // 5 minutes
      }),
    ],
  })
);

// Cache PMTiles files (download for offline use)
registerRoute(
  ({ url }) => url.pathname.endsWith('.pmtiles'),
  new CacheFirst({
    cacheName: 'pmtiles-cache',
    plugins: [
      new CacheableResponsePlugin({ statuses: [200, 206] }),
      new ExpirationPlugin({ maxEntries: 20 }),
    ],
  })
);

// Cache fonts and sprites
registerRoute(
  ({ url }) => url.pathname.match(/\/(font|sprite)\//),
  new CacheFirst({
    cacheName: 'map-assets',
    plugins: [
      new ExpirationPlugin({ maxAgeSeconds: 365 * 24 * 60 * 60 }),
    ],
  })
);
```

### Compression Comparison (Vector Tiles)

| Compression | Size (avg MVT tile) | Decode Speed | Support |
|-------------|-------------------|-------------|---------|
| None | 50 KB | Fastest | Universal |
| gzip | 18 KB | Fast | Universal |
| Brotli | 14 KB | Fast | Modern browsers |
| zstd | 15 KB | Fastest decode | CDN + modern |

```nginx
# Nginx: Brotli + gzip for tiles
brotli on;
brotli_types application/x-protobuf application/vnd.mapbox-vector-tile;
brotli_comp_level 6;

gzip on;
gzip_types application/x-protobuf application/vnd.mapbox-vector-tile application/json;
gzip_comp_level 6;
```

### Full Nginx Tile Proxy Configuration

```nginx
upstream martin {
    server martin:3000;
    keepalive 32;
}

proxy_cache_path /var/cache/tiles
    levels=1:2
    keys_zone=tiles:100m
    max_size=50g
    inactive=30d
    use_temp_path=off;

server {
    listen 80;
    server_name tiles.example.com;

    # Tile endpoint
    location /tiles/ {
        proxy_pass http://martin/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        # Caching
        proxy_cache tiles;
        proxy_cache_valid 200 24h;
        proxy_cache_valid 204 1m;
        proxy_cache_valid 404 1m;
        proxy_cache_use_stale error timeout updating;
        proxy_cache_lock on;
        proxy_cache_lock_timeout 5s;

        # Headers
        add_header X-Cache-Status $upstream_cache_status;
        add_header Cache-Control "public, max-age=86400, stale-while-revalidate=604800";
        add_header Access-Control-Allow-Origin "*";
        add_header Vary "Accept-Encoding";

        # Compression
        gzip on;
        gzip_types application/x-protobuf application/vnd.mapbox-vector-tile;
    }

    # Health check
    location /health {
        proxy_pass http://martin/health;
        access_log off;
    }
}
```

---

## Benchmarking & Monitoring

### Load Testing Tile Server — k6

```javascript
// k6-tile-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const tileLatency = new Trend('tile_latency');
const tileFail = new Rate('tile_fail_rate');

export const options = {
  stages: [
    { duration: '30s', target: 50 },
    { duration: '2m', target: 200 },
    { duration: '30s', target: 0 },
  ],
};

const TILE_SERVER = 'http://localhost:3000';

export default function () {
  // Random tile at zoom 12-14
  const z = Math.floor(Math.random() * 3) + 12;
  const x = Math.floor(Math.random() * Math.pow(2, z));
  const y = Math.floor(Math.random() * Math.pow(2, z));

  const res = http.get(`${TILE_SERVER}/buildings/${z}/${x}/${y}`);

  tileLatency.add(res.timings.duration);
  tileFail.add(res.status !== 200);

  check(res, {
    'status is 200': (r) => r.status === 200,
    'latency < 500ms': (r) => r.timings.duration < 500,
    'size < 500KB': (r) => r.body.length < 500000,
  });

  sleep(0.1);
}
```

```bash
# Run load test
k6 run k6-tile-load-test.js

# Expected output:
# tile_latency: avg=45ms p95=120ms p99=250ms
# tile_fail_rate: 0.01%
# http_reqs: 15000/s
```

### Key Performance Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Tile response time (cached) | < 50ms | CDN/Nginx cache |
| Tile response time (uncached) | < 500ms | Martin/TiTiler |
| Time to first tile | < 1s | Browser DevTools |
| Feature API response | < 500ms | Application monitoring |
| Map interaction FPS | > 30fps | Chrome DevTools |
| Vector tile size | < 500KB | tippecanoe --stats |
| Initial page load (LCP) | < 2.5s | Lighthouse |
| PostGIS query time | < 200ms | pg_stat_statements |

### Before/After Optimization Examples

| Scenario | Before | After | Improvement |
|----------|--------|-------|-------------|
| 1M buildings (GeoJSON) | 2.1 GB, 15s load | 120 MB PMTiles, instant | 94% smaller, 100x faster |
| PostGIS bbox query | 800ms (seq scan) | 15ms (GiST index) | 53x faster |
| Raster tile from GeoTIFF | 2s (full read) | 50ms (COG range req) | 40x faster |
| Feature API (no cache) | 500ms | 5ms (Redis cache) | 100x faster |
| Frontend with 100K markers | 2fps, crashed | 60fps (deck.gl) | 30x faster |
