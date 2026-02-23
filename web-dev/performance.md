# Performance Optimization

Geospatial applications often deal with large datasets, complex geometries, and real-time rendering demands. This guide covers optimization strategies across the entire stack -- from database queries to browser rendering.

> **Quick Picks**
> - **Zero-infra tile serving:** PMTiles on CDN (Cloudflare R2 for free egress)
> - **Vector tile generation:** Tippecanoe (the standard tool)
> - **Cloud raster access:** Cloud-Optimized GeoTIFF with HTTP range requests
> - **Client-side geo processing:** WebAssembly (GEOS via geos-wasm, Proj via proj4js)
> - **Fastest rendering:** MapLibre GL JS or Deck.gl (WebGL-based)

---

## Vector Tile Optimization

Vector tiles are the most efficient way to deliver large feature datasets to the browser. Proper optimization can reduce tile sizes by 10x or more.

### Tippecanoe

Mapbox's tippecanoe is the standard tool for generating optimized vector tilesets.

Key strategies:

- **Simplification**: Reduce vertex count at lower zoom levels (`--simplification`)
- **Feature dropping**: Remove less important features at lower zooms (`--drop-densest-as-needed`)
- **Attribute filtering**: Include only needed properties (`--include`)
- **Zoom range**: Set appropriate min/max zoom per layer (`--minimum-zoom`, `--maximum-zoom`)
- **Tile size limits**: Cap tile sizes to prevent slow rendering (`--maximum-tile-bytes`)

```bash
tippecanoe -o output.pmtiles \
  --minimum-zoom=2 --maximum-zoom=14 \
  --simplification=10 \
  --drop-densest-as-needed \
  --include=name,type,population \
  input.geojson
```

### Geometry Simplification

| Algorithm | Use Case | Library |
|-----------|----------|---------|
| Douglas-Peucker | General simplification | GEOS, Shapely, Turf |
| Visvalingam-Whyatt | Area-based, better visual quality | Topojson, Mapshaper |
| Topology-preserving | Shared boundaries (admin areas) | Mapshaper, PostGIS |

### PMTiles: Zero-Infrastructure Tile Serving

[PMTiles](https://protomaps.com/docs/pmtiles) is a single-file tile archive that supports HTTP range requests, eliminating the need for a tile server.

```javascript
// MapLibre + PMTiles protocol
import { Protocol } from 'pmtiles';
import maplibregl from 'maplibre-gl';

const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({
  style: {
    sources: {
      buildings: {
        type: 'vector',
        url: 'pmtiles://https://r2.example.com/buildings.pmtiles'
      }
    },
    layers: [{ id: 'buildings', type: 'fill', source: 'buildings', 'source-layer': 'buildings' }]
  }
});
```

**Deployment:** Upload `.pmtiles` file to S3/R2/GCS with CORS headers. No server needed.

### Before/After File Size Examples

| Dataset | GeoJSON | Shapefile | PMTiles (z2-14) | Reduction |
|---|---|---|---|---|
| US counties (3K features) | 45 MB | 12 MB | 3 MB | 93% |
| NYC buildings (1M features) | 2.1 GB | 800 MB | 120 MB | 94% |
| Global admin boundaries | 350 MB | 150 MB | 25 MB | 93% |

---

## Raster Optimization

### Cloud Optimized GeoTIFF (COG)

COGs allow efficient range-request access to raster data on cloud storage without downloading entire files.

```bash
gdal_translate input.tif output_cog.tif \
  -of COG \
  -co COMPRESS=ZSTD \
  -co OVERVIEW_RESAMPLING=AVERAGE \
  -co BLOCKSIZE=512
```

### Key Raster Optimization Techniques

| Technique | Purpose | Tool |
|-----------|---------|------|
| Overviews (pyramids) | Fast display at lower zooms | gdaladdo |
| Compression (ZSTD, LZW, JPEG) | Reduce file size | gdal_translate |
| Tiling (internal tiles) | Enable partial reads | gdal_translate (-co TILED=YES) |
| COG conversion | Cloud-native access | gdal_translate (-of COG) |
| Band reduction | Remove unused bands | gdal_translate (-b) |
| Bit depth optimization | Reduce precision where possible | gdal_translate (-ot) |

### COG Reading Pattern (Client-Side)

Cloud-Optimized GeoTIFFs enable reading specific regions via HTTP range requests without downloading the entire file.

```javascript
import GeoTIFF from 'geotiff';

// Read a specific bounding box from a remote COG
const tiff = await GeoTIFF.fromUrl('https://s3.example.com/data.tif');
const image = await tiff.getImage();
const data = await image.readRasters({
  bbox: [-122.5, 37.7, -122.3, 37.9],
  width: 256,
  height: 256
});
```

### Compression Comparison (1 GB uncompressed GeoTIFF)

| Compression | File Size | Read Speed | Best For |
|---|---|---|---|
| None | 1000 MB | Fastest | Local SSD |
| LZW | 450 MB | Fast | General purpose |
| ZSTD | 380 MB | Fast | Modern systems (best ratio + speed) |
| DEFLATE | 420 MB | Moderate | Maximum compatibility |
| JPEG (lossy) | 120 MB | Fast | Visual imagery (RGB) |
| LERC | 350 MB | Fast | Floating-point elevation data |

---

## Spatial Indexing Strategies

Spatial indexes dramatically speed up queries involving location-based filtering, nearest-neighbor searches, and spatial joins.

### Index Types

| Index Type | Best For | Implementation |
|-----------|----------|---------------|
| R-tree (GiST) | General spatial queries, bounding box | PostGIS (default), SQLite |
| SP-GiST | Point data, quadtree-like partitioning | PostGIS |
| BRIN | Very large tables with spatial locality | PostGIS |
| H3 | Hexagonal grid aggregation, analytics | H3 library, DuckDB |
| S2 | Global spherical indexing, BigQuery | Google S2, BigQuery |
| Geohash | Simple prefix-based proximity | Redis, Elasticsearch |

### PostGIS Index Tips

```sql
-- Standard spatial index
CREATE INDEX idx_geom ON my_table USING GIST (geom);

-- Partial index for specific feature types
CREATE INDEX idx_buildings ON my_table USING GIST (geom)
  WHERE type = 'building';

-- BRIN for large time-series spatial data
CREATE INDEX idx_brin ON sensor_data USING BRIN (geom, timestamp);
```

### HTTP Range Requests for Spatial Data

Range requests let clients fetch only the bytes they need from large files. Key formats that support range requests:

| Format | Type | Range Request Support | Use Case |
|---|---|---|---|
| COG (Cloud-Optimized GeoTIFF) | Raster | Native (internal tiles + overviews) | Remote sensing, elevation |
| PMTiles | Vector/Raster tiles | Native (directory + tile data) | Map tiles from CDN |
| GeoParquet | Vector | Via Parquet row groups | Large vector datasets |
| FlatGeobuf | Vector | Via spatial index + HTTP ranges | Streaming vector features |

### WebAssembly for Client-Side Geo Processing

WASM enables running spatial operations in the browser at near-native speed.

| Library | Purpose | Size | Performance |
|---|---|---|---|
| [geos-wasm](https://github.com/nicklockwood/geos-wasm) | GEOS operations (buffer, intersect, union) | ~500 KB | Near-native |
| [proj4js](https://github.com/proj4js/proj4js) | CRS transformations | ~100 KB | Fast |
| [Turf.js](https://turfjs.org/) | Spatial analysis (JS, not WASM) | Tree-shakeable | Good for simple ops |
| [loaders.gl](https://loaders.gl/) | Parse spatial formats (Shapefile, GeoPackage) | Modular | Good |
| [DuckDB-WASM](https://duckdb.org/docs/api/wasm/) | SQL analytics in browser | ~10 MB | Near-native |

---

## Frontend Rendering Optimization

### WebGL Rendering

WebGL-based renderers (MapLibre, Deck.gl) can handle hundreds of thousands of features compared to SVG/DOM renderers (Leaflet).

| Technique | Description | Library |
|-----------|-------------|---------|
| GPU-based rendering | Offload geometry to GPU | MapLibre, Deck.gl |
| Instanced rendering | Reuse geometry for repeated features | Deck.gl |
| Binary data transfer | Skip JSON parsing with binary formats | Deck.gl (binary) |

### Web Workers

Offload heavy computation to background threads to keep the UI responsive.

- Geometry operations (simplification, buffering)
- Data parsing (CSV, GeoJSON)
- Spatial queries (point-in-polygon, nearest neighbor)

### Virtualization

Only render features visible in the current viewport. Most map libraries do this automatically for tiles, but custom layers may need manual viewport culling.

### Web Worker Example

```javascript
// worker.js
import * as turf from '@turf/turf';
self.onmessage = (e) => {
  const { geojson, distance } = e.data;
  const buffered = turf.buffer(geojson, distance, { units: 'kilometers' });
  self.postMessage(buffered);
};

// main.js
const worker = new Worker('worker.js', { type: 'module' });
worker.postMessage({ geojson: featureCollection, distance: 5 });
worker.onmessage = (e) => map.getSource('buffer').setData(e.data);
```

---

## Caching Strategies

| Strategy | Layer | Tool | Best For |
|----------|-------|------|----------|
| Tile cache | Server | GeoWebCache, Varnish, nginx | Raster/vector tile reuse |
| CDN | Network | CloudFront, Cloudflare, Fastly | Global tile distribution |
| Service Worker | Browser | Workbox, custom SW | Offline maps, repeat visits |
| HTTP caching | Browser | Cache-Control headers | Static tiles, infrequent updates |
| In-memory cache | Server | Redis, Memcached | Feature API responses |
| Browser storage | Browser | IndexedDB, Cache API | Offline feature storage |

### Cache Invalidation

- Use content-based hashes in tile URLs for cache busting
- Set appropriate `Cache-Control` / `ETag` headers
- Consider tile expiry strategies for frequently updated data

### Nginx Tile Cache Configuration

```nginx
proxy_cache_path /var/cache/tiles levels=1:2 keys_zone=tiles:10m max_size=10g inactive=30d;

location /tiles/ {
    proxy_pass http://martin:3000/;
    proxy_cache tiles;
    proxy_cache_valid 200 30d;
    add_header X-Cache-Status $upstream_cache_status;
    add_header Cache-Control "public, max-age=2592000";
}
```

---

## Database Optimization

### PostGIS Performance

#### Indexing

```sql
-- Ensure spatial index exists
CREATE INDEX IF NOT EXISTS idx_geom ON features USING GIST (geom);

-- Analyze table statistics for query planner
ANALYZE features;

-- Cluster table by spatial index for locality
CLUSTER features USING idx_geom;
```

#### Query Optimization

- Use `ST_Intersects` with bounding box filters (`&&` operator)
- Prefer `ST_DWithin` over `ST_Distance < X` (index-friendly)
- Use `ST_Simplify` in queries serving lower zoom levels
- Limit returned columns and rows

#### Connection Pooling

Use PgBouncer or built-in connection pools (SQLAlchemy, asyncpg) to avoid connection overhead.

| Tool | Type | Best For |
|------|------|----------|
| PgBouncer | External pooler | High-concurrency apps |
| pgpool-II | External pooler + load balancer | Read replicas |
| SQLAlchemy pool | Application-level | Python apps |
| asyncpg pool | Application-level (async) | FastAPI / async Python |

### EXPLAIN ANALYZE Examples

```sql
-- Bbox query (uses spatial index efficiently)
EXPLAIN ANALYZE
SELECT id, name, ST_AsGeoJSON(geom)
FROM buildings
WHERE geom && ST_MakeEnvelope(-122.5, 37.7, -122.3, 37.9, 4326);

-- Distance query (index-friendly version)
EXPLAIN ANALYZE
SELECT id, name FROM buildings
WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint(-122.4, 37.8), 4326), 0.01);
-- Use ST_DWithin instead of ST_Distance < X for index usage
```

---

## Benchmarking Tools and Metrics

| Tool | Measures | Use Case |
|------|----------|----------|
| pgbench + custom SQL | Database query throughput | PostGIS query performance |
| k6 / wrk | HTTP request throughput | Tile server load testing |
| Lighthouse | Frontend performance score | Overall web performance |
| Chrome DevTools (Performance) | Rendering frame rate, JS execution | Map rendering performance |
| tippecanoe --stats | Tile size statistics | Vector tile optimization |
| gdalinfo | Raster file structure | COG verification |

### Key Metrics to Track

- **Tile response time**: < 200ms for cached, < 1s for generated
- **Time to first tile**: How quickly the initial map loads
- **Frame rate**: Target 60fps during map interactions
- **Feature query time**: < 500ms for typical spatial queries
- **Tile size**: < 500KB per vector tile, < 256KB ideal

