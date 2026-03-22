# Performance Optimization

> Data validated: 2026-03-21
>
> For enterprise-scale performance tuning (PostGIS deep tuning, k6 load testing, monitoring) -> [web-dev/performance.md](../web-dev/performance.md)

## 30-Second Decision

| Problem | Solution | Effort |
|---------|----------|--------|
| Map renders slow with many features | Vector tiles (PMTiles + MapLibre) | 30 min |
| GeoJSON too large to load | FlatGeobuf or PMTiles | 30 min |
| Spatial operations block the UI | Web Workers | 1 hour |
| Need millions of points at 60fps | deck.gl binary data mode | 2 hours |
| Need SQL analytics in browser | DuckDB-WASM + GeoParquet (dashboards only, not interactive maps) | 2 hours |
| Database queries slow | PostGIS tuning | 1-2 hours |
| Tile serving slow | CDN/caching in front | 1 hour |

---

## Rendering Benchmarks

Mid-range desktop (Intel i7, GTX 1060, 16GB RAM, Chrome). Mobile degrades 2-5x.

| Library | Renderer | 1K | 10K | 100K | 1M | 10M |
|---------|----------|-----|------|-------|-----|------|
| Leaflet (SVG) | SVG DOM | 60fps | 30fps | 5fps | crash | crash |
| Leaflet (Canvas) | Canvas 2D | 60fps | 60fps | 30fps | 5fps | crash |
| MapLibre GL JS | WebGL | 60fps | 60fps | 60fps | 50fps* | 30fps* |
| deck.gl | WebGL2 | 60fps | 60fps | 60fps | 60fps | 50fps |
| CesiumJS (Entity) | WebGL | 60fps | 30fps | 5fps | crash | crash |
| CesiumJS (Primitive) | WebGL | 60fps | 60fps | 50fps | 30fps | N/A |

*via vector tiles, not raw GeoJSON

---

## Solutions (easiest first)

### 1. Vector Tiles (PMTiles + MapLibre)

The single biggest performance improvement for most geo apps. Pre-tiled data, only visible tiles loaded.

**Effort:** 30 min | **Impact:** 10x-100x for large datasets

```bash
tippecanoe -o data.pmtiles -z14 -Z0 --drop-densest-as-needed input.geojson
aws s3 cp data.pmtiles s3://my-bucket/
```

```javascript
import { Protocol } from 'pmtiles';
maplibregl.addProtocol('pmtiles', new Protocol().tile);
map.addSource('data', {
  type: 'vector',
  url: 'pmtiles://https://cdn.example.com/data.pmtiles',
  maxzoom: 14  // ALWAYS set this
});
```

**Key caveats:** Static data only. tippecanoe drop strategies are opaque -- test visually. Always set `maxzoom`.

---

### 2. Web Workers for Heavy Spatial Operations

Move CPU-intensive work off the main thread.

**Effort:** 1 hour | **Impact:** Eliminates UI freezes

```javascript
// main.js
const worker = new Worker('spatial-worker.js');
worker.postMessage({ operation: 'buffer', geojson: largeGeoJSON, distance: 500, units: 'meters' });
worker.onmessage = (e) => map.getSource('buffered').setData(e.data);
```

```javascript
// spatial-worker.js
import buffer from '@turf/buffer';
self.onmessage = (e) => {
  const { geojson, distance, units } = e.data;
  self.postMessage(buffer(geojson, distance, { units }));
};
```

**Key caveats:** Structured cloning overhead for large data (use Transferable objects). Create worker once, reuse.

---

### 3. FlatGeobuf for Spatial-Filtered Loading

Only download the data you need. Bbox-filtered HTTP range requests from static hosting.

**Effort:** 30 min | **Impact:** 10x reduction in data transfer

```javascript
import { deserialize } from 'flatgeobuf/lib/mjs/geojson';
const bounds = map.getBounds();
const bbox = { minX: bounds.getWest(), minY: bounds.getSouth(),
               maxX: bounds.getEast(), maxY: bounds.getNorth() };
const features = [];
for await (const feature of deserialize('https://cdn.example.com/data.fgb', bbox))
  features.push(feature);
map.getSource('data').setData({ type: 'FeatureCollection', features });
```

---

### 4. Spatial Indexing (Flatbush / rbush)

Stop scanning all features. O(n) -> O(log n).

**Effort:** 30 min | **Impact:** 100x-1000x faster spatial queries for >10K features

Flatbush = 8 bytes/item (static). rbush = 80 bytes/item (dynamic insert/delete). See [spatial-analysis.md](spatial-analysis.md) for code examples.

---

### 5. deck.gl Binary Data Mode

10M+ points at 60fps using typed arrays instead of JSON objects.

**Effort:** 2 hours | **Impact:** 10x-100x over JSON mode at >100K features

```javascript
const positions = new Float32Array(points.length * 2);
points.forEach((p, i) => { positions[i*2] = p.lng; positions[i*2+1] = p.lat; });
const layer = new ScatterplotLayer({
  data: { length: points.length, attributes: { getPosition: { value: positions, size: 2 } } },
  getRadius: 100
});
```

**Key caveats:** `updateTriggers` are critical -- shallow comparison. Must call `deck.finalize()` on cleanup.

---

### 6. PostGIS Quick Tuning

Default PostgreSQL settings are designed for a 1990s workstation.

```sql
-- Essential tuning (postgresql.conf)
-- shared_buffers = 25% of RAM
-- work_mem = 64-256MB (careful: per-sort, not per-query)
-- effective_cache_size = 75% of RAM
-- random_page_cost = 1.1 (for SSD -- default 4.0 assumes HDD!)

-- CRITICAL: spatial indexes
CREATE INDEX CONCURRENTLY idx_geom ON features USING GIST (geom);

-- Use ST_DWithin, NOT ST_Distance for indexed queries
SELECT * FROM points WHERE ST_DWithin(geom, target, 1000);
```

**Key caveats:** `work_mem=256MB` * 50 concurrent queries = 12.5GB. Balance per-query perf vs total memory.

For deep PostGIS tuning, index strategies, VACUUM -> [web-dev/performance.md](../web-dev/performance.md)

---

### 7. CDN Caching for Tile Servers

Every tile server (Martin, pg_tileserv) MUST have caching in front for production.

```nginx
proxy_cache_path /data/tile-cache levels=1:2 keys_zone=tiles:100m max_size=10g;
location /tiles/ {
    proxy_pass http://martin:3000/;
    proxy_cache tiles;
    proxy_cache_valid 200 1d;
    add_header X-Cache-Status $upstream_cache_status;
}
```

For full deployment and caching architecture -> [web-dev/deployment.md](../web-dev/deployment.md)

---

## Performance Decision Flowchart

```
What's slow?
+-- Map rendering laggy
|   +-- >10K features? -> Vector tiles (PMTiles + MapLibre)
|   +-- >100K points? -> deck.gl binary mode
|   +-- Leaflet hitting limits? -> Switch to MapLibre GL JS
+-- Data loading slow
|   +-- GeoJSON >10MB? -> FlatGeobuf or vector tiles
|   +-- Need SQL analytics? -> DuckDB-WASM + GeoParquet (dashboards only)
+-- Spatial ops freeze UI -> Web Workers
+-- Tile serving slow
|   +-- No caching? -> Add Nginx/CDN
|   +-- DB queries slow? -> PostGIS tuning
|   +-- Could be static? -> PMTiles
+-- Everything slow on mobile
    +-- Bundle too large? -> Code split, lazy load map
    +-- WebGL issues? -> Reduce layers, lower maxzoom
```
