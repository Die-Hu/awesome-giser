# Performance Optimization -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**Biggest single win:** Convert GeoJSON to vector tiles with Tippecanoe -> serve via PMTiles on CDN. **Database:** Tune PostGIS (shared_buffers, random_page_cost, GiST indexes). **Client rendering at scale:** deck.gl with binary data mode. **Client analytics:** DuckDB-WASM for analytical dashboards (not interactive maps).

---

## Tier 1 -- Production First Choices

---

### Tippecanoe -- Vector Tile Generation

The industry standard for converting GeoJSON/FlatGeobuf to vector tiles (PMTiles/MBTiles). Created by Mapbox, maintained by Felt. Deterministic output, excellent CI/CD integration.

**Why Tier 1:** Production-readiness 5/5. The single most impactful performance tool in the GIS stack. No alternative comes close.

```bash
# Recipe 1: Buildings -- zoom-dependent detail
tippecanoe -o buildings.pmtiles \
  --minimum-zoom=10 --maximum-zoom=15 \
  --simplification=10 \
  --drop-smallest-as-needed \
  --include=name,height,type,area \
  --layer=buildings \
  buildings.geojson

# Recipe 2: Roads -- importance-based filtering
tippecanoe -o roads.pmtiles \
  --minimum-zoom=4 --maximum-zoom=14 \
  -j '{ "*": { "minzoom_by_attribute": { "highway": { "motorway": 4, "trunk": 6, "primary": 8, "secondary": 10, "tertiary": 12, "residential": 13 } } } }' \
  --coalesce-smallest-as-needed \
  roads.geojson

# Recipe 3: Points -- clustering at low zooms
tippecanoe -o poi.pmtiles \
  --minimum-zoom=2 --maximum-zoom=14 \
  --cluster-distance=50 \
  --cluster-maxzoom=12 \
  --accumulate-attribute=count:sum \
  --include=name,category,count \
  poi.geojson

# Recipe 4: Admin boundaries -- topology-preserving
tippecanoe -o admin.pmtiles \
  --minimum-zoom=0 --maximum-zoom=12 \
  --simplification=10 \
  --detect-shared-borders \
  --no-tiny-polygon-reduction \
  --include=name,admin_level,iso_code \
  admin_boundaries.geojson

# Recipe 5: Multi-layer basemap from FlatGeobuf (2-5x faster)
tippecanoe -o basemap.pmtiles \
  --named-layer=buildings:buildings.fgb \
  --named-layer=roads:roads.fgb \
  --named-layer=water:water.fgb \
  --named-layer=landuse:landuse.fgb \
  --named-layer=poi:poi.fgb \
  --minimum-zoom=0 --maximum-zoom=14 \
  --read-parallel

# Recipe 6: Overzooming (serve z14 data at z15-22)
tippecanoe -o data.pmtiles \
  --minimum-zoom=0 --maximum-zoom=14 \
  --extra-detail=12 \
  input.geojson
# MapLibre: set "maxzoom": 14 in source, tiles overzoom client-side
```

**Caveats:**
- **Memory usage scales with data.** Processing a 10GB GeoJSON file can consume 20-40GB RAM. CI/CD runners with 7GB RAM will OOM. Must pre-split data or use FlatGeobuf input.
- **Processing time for large datasets.** 100M features can take 2-6 hours. No checkpointing -- if it crashes at 95%, you start over.
- **No incremental updates.** Cannot add features to an existing PMTiles file. Must regenerate the entire tileset.
- **Drop strategies are opaque.** `--drop-densest-as-needed` makes lossy decisions about what to keep. Always verify output visually.
- **GeoJSON input is slow.** Converting to FlatGeobuf first (`ogr2ogr -f FlatGeobuf`) then using `--read-parallel` is 2-5x faster.

---

### PMTiles -- Serverless Tile Archive

Single-file tile format served via HTTP range requests. No tile server needed -- upload to any CDN.

**Why Tier 1:** Production-readiness 4/5. Eliminates tile server infrastructure entirely. Cloudflare R2 ($0 egress) + PMTiles is the lowest-cost tile serving architecture possible.

```
PMTiles file structure:
┌─────────────────────────────────┐
│  Header (127 bytes)             │  <- Fixed metadata
├─────────────────────────────────┤
│  Root Directory (variable)      │  <- Top-level tile index
├─────────────────────────────────┤
│  Leaf Directories (variable)    │  <- Tile offset/length entries
├─────────────────────────────────┤
│  Tile Data (bulk)               │  <- Actual MVT/PNG/WebP data
├─────────────────────────────────┤
│  Metadata JSON                  │  <- TileJSON-compatible
└─────────────────────────────────┘

HTTP range request flow:
1. Client reads header (0-127 bytes) -> gets root directory offset
2. Client reads root directory -> finds leaf directory for z/x/y
3. Client reads leaf directory -> finds tile offset + length
4. Client reads tile data -> renders
Total: 2-3 HTTP requests per tile (first load), 1 after caching directories
```

```javascript
// PMTiles + MapLibre -- Complete setup
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
    },
    layers: [
      { id: 'water', type: 'fill', source: 'basemap', 'source-layer': 'water',
        paint: { 'fill-color': '#a0d0f0' } },
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

**Caveats:**
- **Static data only.** Any data change requires regeneration + re-upload.
- **Range request support required.** Most CDNs support this, but some corporate proxies strip Range headers.
- **Initial load latency.** First tile requires 2-3 range requests (header -> directory -> tile data). On slow connections, this adds 200-500ms.
- **No dynamic filtering.** Unlike Martin, you cannot pass query parameters to filter tile content at request time.
- **Anti-pattern: Not setting maxzoom on the MapLibre source.** Without maxzoom, the client requests tiles at zoom levels that don't exist.

---

### Cloud-Optimized GeoTIFF (COG) -- Raster Optimization

The standard format for serving raster data from object storage. HTTP range requests enable reading only the needed pixels.

**Why Tier 1:** Production-readiness 5/5 -- USGS, NASA, ESA all use COG. The standard for earth observation and raster data pipelines.

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

# Add overviews if missing
gdaladdo -r average output_cog.tif 2 4 8 16 32
```

#### Compression Benchmarks (1 GB uncompressed)

| Compression | Size | Read Speed | Ratio | Best For |
|-------------|------|-----------|-------|----------|
| None | 1000 MB | Fastest | 1.0x | Local SSD |
| LZW | 450 MB | Fast | 2.2x | General |
| ZSTD | 380 MB | Fast | 2.6x | Modern (best ratio+speed) |
| DEFLATE | 420 MB | Medium | 2.4x | Max compatibility |
| JPEG (lossy) | 120 MB | Fast | 8.3x | RGB imagery |
| LERC | 350 MB | Fast | 2.9x | Floating-point elevation |
| WEBP (lossy) | 100 MB | Fast | 10.0x | Visual tiles |

**Caveats:**
- **Conversion is mandatory.** Existing GeoTIFFs must be converted with `gdal_translate -of COG`. Large datasets (100GB+) take hours.
- **Missing overviews.** Forgetting `gdaladdo` means the client downloads full-resolution data even for zoomed-out views. This is the #1 performance mistake with COGs.
- **CORS on object storage.** Must configure CORS on S3/R2 to allow Range requests from browser. Missing CORS headers is the #1 "why doesn't my COG load" issue.

---

### PostGIS Tuning -- Database Performance

PostgreSQL defaults are configured for a 1990s workstation. Proper tuning is the single highest-impact performance optimization for dynamic GIS applications.

**Why Tier 1:** Production-readiness 5/5. PostGIS tuning is mandatory for any enterprise deployment.

```ini
# postgresql.conf -- Optimized for 16 GB RAM server
# Memory
shared_buffers = 4GB           # 25% of RAM
effective_cache_size = 12GB    # 75% of RAM
work_mem = 64MB                # Safe starting value; 256MB only if concurrent queries are low
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

# Planner -- CRITICAL for spatial
random_page_cost = 1.1         # SSD storage (default 4.0 for HDD!)
effective_io_concurrency = 200  # SSD
default_statistics_target = 500 # Better stats for spatial data

# PostGIS specific
postgis.gdal_enabled_drivers = 'GTiff PNG JPEG'
postgis.enable_outdb_rasters = true
```

**Note on work_mem:** Setting `work_mem=256MB` means each sort/hash operation gets 256MB. 50 concurrent queries * 256MB = 12.5GB -- can OOM the server. Start with 64MB and increase only after monitoring actual usage with `pg_stat_statements`.

#### Index Strategies

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

#### Query Optimization -- EXPLAIN ANALYZE Examples

```sql
-- BAD: ST_Distance < threshold (no index usage!)
SELECT * FROM poi
WHERE ST_Distance(geom, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)) < 0.01;
-- Seq Scan (no index!)

-- GOOD: ST_DWithin (uses spatial index)
SELECT * FROM poi
WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326), 0.01);
-- Index Scan using idx_poi_geom

-- GOOD: kNN with <-> operator (uses GiST index)
SELECT *, geom <-> ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326) AS dist
FROM poi
ORDER BY geom <-> ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326)
LIMIT 10;
-- Index Scan using idx_poi_geom (KNN)
```

#### Partitioning for Time-Series Spatial Data

```sql
CREATE TABLE observations (
    id BIGSERIAL,
    sensor_id INT,
    value FLOAT,
    geom GEOMETRY(Point, 4326),
    observed_at TIMESTAMPTZ NOT NULL
) PARTITION BY RANGE (observed_at);

-- Auto-create partitions with pg_partman
CREATE EXTENSION pg_partman;
SELECT partman.create_parent('public.observations', 'observed_at', 'native', 'monthly');
```

**Caveats:**
- **Wrong defaults.** `shared_buffers=128MB`, `work_mem=4MB` are useless for spatial. Must tune to at least 25% RAM for shared_buffers.
- **VACUUM neglect.** Large spatial tables with frequent updates need aggressive autovacuum settings. Default thresholds are too high.
- **Index bloat.** GiST indexes on frequently updated tables grow unbounded. Must schedule `REINDEX CONCURRENTLY` periodically.
- **`random_page_cost` misconfiguration.** Default 4.0 assumes HDD. On SSD, must set to 1.1-1.5 or PostgreSQL will avoid index scans.
- **Anti-pattern: Creating indexes before bulk load.** Building indexes during insert is 10x slower. Create GiST indexes after bulk inserts.

---

### Martin Performance Tuning -- Dynamic Tile Server

Optimized configuration for production Martin deployments.

**Why Tier 1:** The hybrid approach (Martin + Nginx cache + CDN) is the production standard. Target: >90% cache hit rate, <50ms p95 latency for cached tiles.

```yaml
# martin-config.yaml -- Optimized for production
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

#### Pre-Generated vs On-the-Fly vs Hybrid

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

**Caveats:**
- **No built-in cache persistence.** The in-memory cache is lost on restart. Nginx/Varnish/CDN in front is mandatory.
- **Connection pool exhaustion at scale.** Default `pool_size: 20` is too low for high-concurrency.
- **Complex tile queries can OOM PostGIS.** Badly written function sources that do full-table scans at low zoom levels can consume gigabytes of RAM.

---

### DuckDB-WASM -- Client-Side Spatial SQL

SQL-powered spatial analysis running entirely in the browser. Query GeoParquet files directly from URLs.

**Why Tier 1 (analytical dashboards only):** Game-changing for dashboards where users run ad-hoc spatial queries on pre-published datasets. Eliminates the need for a backend analytics API. Production-readiness 3/5 -- rapidly maturing. **Not a replacement for PostGIS for transactional workloads or interactive map applications.**

```javascript
import * as duckdb from '@duckdb/duckdb-wasm';

const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);
const worker = new Worker(bundle.mainWorker);
const logger = new duckdb.ConsoleLogger();
const db = new duckdb.AsyncDuckDB(logger, worker);
await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

const conn = await db.connect();
await conn.query('INSTALL spatial; LOAD spatial;');

// Load GeoParquet from URL (no import needed!)
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

**Performance note:** Documented comparisons showing DuckDB-WASM outperforming PostGIS (e.g., ~0.8s vs ~5s for analytical queries) reflect **warm query** performance only. The full picture includes:
- **Cold start:** 1-3 seconds for WASM compilation + spatial extension loading on first use
- **WASM download:** 4-8MB initial bundle (must be lazy-loaded)
- **First query total time:** 3-6 seconds including cold start
- **Browser memory limit:** ~2-4GB WASM memory. Large GeoParquet files can exceed this -- the tab crashes with no graceful error.

DuckDB-WASM excels for analytical dashboards where users run multiple queries after the initial load. It is not suitable for interactive map applications where sub-second first-load is required.

**Caveats:**
- **WASM bundle size.** 4-8MB WASM module. Must be lazy-loaded; including in initial bundle destroys load time.
- **Spatial extension maturity.** The spatial extension is newer than the core. Some PostGIS functions are missing.
- **Threading model.** Uses Web Workers internally. On Safari, SharedArrayBuffer requires Cross-Origin-Isolation headers (COOP/COEP), which can break third-party scripts.
- **npm supply chain note.** DuckDB-WASM experienced a supply chain compromise in September 2025 (since resolved). Lock specific versions and use integrity hashes.

---

### deck.gl Optimization -- Large-Scale Rendering

GPU-accelerated techniques for rendering 10M+ features at 60fps.

**Why Tier 1:** The only option for >10K features at 60fps. Used by Uber, Foursquare, Google. WebGPU migration is in progress.

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
```

**Caveats:**
- **Binary data mode is necessary for scale.** JSON/GeoJSON tops out at ~100K before frame drops.
- **`updateTriggers` is non-negotiable.** deck.gl uses shallow comparison. The #1 source of "my data doesn't render" bugs.
- **GPU memory leaks.** Must call `deck.finalize()` on unmount.
- **Anti-pattern: Using deck.gl for <1K features.** Use MapLibre's built-in layers.

---

### Web Workers for Spatial Computation

Offload CPU-intensive spatial operations to background threads to keep the UI responsive.

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
        tolerance: payload.tolerance, highQuality: true,
      });
      self.postMessage({ type: 'simplify-result', payload: result });
      break;
    }
  }
};

// main.js -- Worker pool pattern
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
```

**Caveats:**
- **Data serialization overhead.** Transferring large GeoJSON between main thread and worker involves JSON serialization. For 10MB GeoJSON, the transfer alone takes 50-100ms.
- **Bundler configuration.** Vite, Webpack, and Rollup all have different worker loading mechanisms.
- **SharedArrayBuffer restrictions.** Cross-origin isolation headers (COOP/COEP) are required and can break third-party scripts.

---

### FlatGeobuf -- Streaming Vector Data

Binary format with built-in spatial index. HTTP range requests enable fetching only features within a bounding box -- without downloading the entire file.

**Why Tier 1:** Production-readiness 4/5. Excellent for medium-sized datasets (10K-1M features) that need spatial filtering but don't justify a full tile pipeline. Adopted by QGIS, GDAL.

```javascript
import { deserialize } from 'flatgeobuf/lib/mjs/geojson.js';

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
```

**Caveats:**
- **Write support is limited.** Browser library is read-only. Must create .fgb files server-side (GDAL, GeoPandas).
- **Spatial filter requires HTTP Range requests.** The key feature requires Range request support from the server.
- **No attribute indexing.** Can only spatially filter by bbox. Attribute queries require downloading all features.
- **Anti-pattern: Using FlatGeobuf for <1000 features.** GeoJSON is simpler and nearly as fast.

---

## Network & Caching

### Nginx Tile Cache -- Full Production Configuration

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

    location /tiles/ {
        proxy_pass http://martin/;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        proxy_cache tiles;
        proxy_cache_valid 200 24h;
        proxy_cache_valid 204 1m;
        proxy_cache_valid 404 1m;
        proxy_cache_use_stale error timeout updating;
        proxy_cache_lock on;
        proxy_cache_lock_timeout 5s;

        add_header X-Cache-Status $upstream_cache_status;
        add_header Cache-Control "public, max-age=86400, stale-while-revalidate=604800";
        add_header Access-Control-Allow-Origin "*";
        add_header Vary "Accept-Encoding";

        gzip on;
        gzip_types application/x-protobuf application/vnd.mapbox-vector-tile;
    }

    location /health {
        proxy_pass http://martin/health;
        access_log off;
    }
}
```

### Compression Comparison (Vector Tiles)

| Compression | Size (avg MVT tile) | Decode Speed | Support |
|-------------|-------------------|-------------|---------|
| None | 50 KB | Fastest | Universal |
| gzip | 18 KB | Fast | Universal |
| Brotli | 14 KB | Fast | Modern browsers |
| zstd | 15 KB | Fastest decode | CDN + modern |

---

## Benchmarking & Monitoring

### Load Testing Tile Server -- k6

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
