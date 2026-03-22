# Spatial Analysis in the Browser

> Data validated: 2026-03-21

## 30-Second Decision

| I need... | Use this | Startup time |
|-----------|----------|-------------|
| Buffer, intersect, point-in-polygon (90% of cases) | `@turf/*` individual packages | 2 min |
| Coordinate transformation (EPSG codes) | `proj4` | 5 min |
| Fast spatial search (static data) | `flatbush` | 5 min |
| Fast spatial search (dynamic data) | `rbush` | 5 min |
| Point-only nearest-neighbor | `kdbush` | 5 min |
| Hexagonal binning and analytics | `h3-js` | 15 min |
| Topologically correct polygon operations | `jsts` | 30 min |
| Raster / COG in browser | `geotiff` | 30 min |
| Full GDAL in browser | `gdal3.js` (avoid if possible) | Hours |

---

## Detailed Guide (by startup time)

### 1. Turf.js

The standard library for client-side spatial analysis. 100+ operations, modular packages, GeoJSON in/out.

**Quick facts:** Individual packages 3-20KB | Easy | Full TypeScript | ~934K npm/week (@turf/turf)

```javascript
// Install ONLY what you need (critical for bundle size)
import booleanPointInPolygon from '@turf/boolean-point-in-polygon';
import buffer from '@turf/buffer';
import { point } from '@turf/helpers';

const pt = point([116.4, 39.9]);
const buffered = buffer(pt, 500, { units: 'meters' });
const isInside = booleanPointInPolygon(pt, somePolygon);
```

```javascript
// ANTI-PATTERN: Don't import the full bundle!
// import * as turf from '@turf/turf';  // 180KB+ -- DON'T
import buffer from '@turf/buffer';       // ~20KB -- DO THIS
```

**Small project:** Default choice. Modular (3KB per function), GeoJSON-native, no build complexity, no WASM.

**Key caveats:**
- Single-threaded: heavy ops on >1K features block the main thread. Use Web Workers.
- Not topologically correct -- can produce self-intersecting geometry. Use JSTS for that.
- No spatial index -- linear scan on all features. Pair with Flatbush/rbush for large data.
- Planar approximation: `buffer()` inaccurate for large areas or near poles.
- **Anti-pattern:** Using `@turf/turf` (full bundle, 180KB) instead of individual packages.

---

### 2. Proj4js

Coordinate transformation between CRS. Thousands of CRS definitions, microsecond transforms.

**Quick facts:** ~30KB gzip | 10+ years stable | 5/5 production-readiness

```javascript
import proj4 from 'proj4';
proj4.defs('EPSG:4490', '+proj=longlat +ellps=GRS80 +no_defs');
const [lng, lat] = proj4('EPSG:4490', 'EPSG:4326', [116.4, 39.9]);
```

**Small project:** Only needed if your data comes in a non-WGS84 CRS (government data, Chinese mapping, military).

**Key caveats:**
- CRS definitions must be registered manually (get from epsg.io)
- Some datum shifts (GCJ02 to WGS84) require grid files not included
- **Anti-pattern:** Transforming coordinates on every frame. Pre-transform once.

---

### 3. Flatbush

The fastest static spatial index for JavaScript. 8 bytes per item, 1M features indexed in ~100ms.

**Quick facts:** ~5KB | Easy | By mourner (Mapbox/MapLibre maintainer)

```javascript
import Flatbush from 'flatbush';
const index = new Flatbush(features.length);
for (const f of features) {
  const [minX, minY, maxX, maxY] = getBbox(f);
  index.add(minX, minY, maxX, maxY);
}
index.finish();
const results = index.search(minX, minY, maxX, maxY); // array of indices
```

**Small project:** Only needed for >10K features with spatial queries. For <10K, Turf's linear scan is fast enough.

**Key caveats:**
- Static only -- cannot add/remove after build. Use rbush for dynamic data.
- Bbox queries only -- post-filter with Turf for precise spatial operations.

---

### 4. rbush

Dynamic spatial index. Supports insert/delete for real-time data like vehicle tracking.

**Quick facts:** ~6KB | ~80 bytes per item (10x Flatbush, but dynamic) | By mourner

```javascript
import RBush from 'rbush';
const tree = new RBush();
tree.load(items.map(item => ({
  minX: item.lng - 0.01, minY: item.lat - 0.01,
  maxX: item.lng + 0.01, maxY: item.lat + 0.01,
  ...item
})));
tree.insert(newItem);
tree.remove(oldItem);
const results = tree.search({ minX, minY, maxX, maxY });
```

**Small project:** Use over Flatbush when data changes at runtime. If data is static, Flatbush is 10x more memory-efficient.

---

### 5. kdbush

Point-only spatial index. Faster than Flatbush for pure point datasets, supports k-nearest-neighbor.

**Quick facts:** ~3KB | Points only | Static

---

### 6. H3-js

Uber's hexagonal hierarchical spatial index. 16 resolution levels, excellent for aggregation and binning.

**Quick facts:** ~500KB (WASM) | Medium learning curve | Uber/Foursquare/Facebook production use

```javascript
import { latLngToCell, cellToBoundary } from 'h3-js';
const h3Index = latLngToCell(39.9, 116.4, 7); // ~5km2 cells
const boundary = cellToBoundary(h3Index);
```

**Small project:** Specialized -- only needed for hexagonal heatmaps, ride-sharing analytics, or hierarchical spatial aggregation.

**Key caveats:**
- WASM adds ~500KB to bundle
- Resolution choice matters enormously (wrong = too coarse or memory explosion)
- BigInt serialization: `JSON.stringify` throws on BigInt, must convert to string
- **Anti-pattern:** Using H3 as replacement for spatial indexing. It's for aggregation, not search.

---

### 7. JSTS (JavaScript Topology Suite)

Port of JTS -- the gold standard for computational geometry correctness.

**Quick facts:** ~200-400KB gzip | Hard (Java-style API) | 4/5 production-readiness

```javascript
import { GeoJSONReader, GeoJSONWriter } from 'jsts/org/locationtech/jts/io';
import { BufferOp } from 'jsts/org/locationtech/jts/operation/buffer';

const reader = new GeoJSONReader();
const writer = new GeoJSONWriter();
const geom = reader.read(myGeoJSON);
const buffered = BufferOp.bufferOp(geom, 0.01);
const result = writer.write(buffered);
```

**Small project:** You almost certainly don't need JSTS. Use only when Turf produces invalid geometry or you need guaranteed topological correctness (land subdivision, utility routing).

**Key caveats:**
- Not GeoJSON-native -- requires conversion to/from JSTS objects
- API mirrors Java naming conventions (`GeometryFactory`, `PrecisionModel`)
- Slower than Turf for simple operations

---

### 8. geotiff.js

Read GeoTIFFs and Cloud-Optimized GeoTIFFs (COGs) in the browser via HTTP range requests.

**Quick facts:** Moderate bundle | 4/5 production-readiness | Used by many earth observation apps

```javascript
import { fromUrl } from 'geotiff';
const tiff = await fromUrl('https://example.com/data.tif');
const image = await tiff.getImage();
const data = await image.readRasters({ window: [0, 0, 256, 256] });
```

**Small project:** Only needed for raster data (satellite imagery, elevation, weather). Most web map projects use vector data.

**Key caveats:**
- CPU-intensive decoding -- use Web Workers
- CORS required on S3 for Range requests (#1 "why doesn't my COG load" issue)
- No reprojection -- reads data as-is

---

## Low Priority / Legacy

### loam (GDAL WASM) -- DEPRECATED

**Deprecated:** Last update November 2023. Do not use for new projects.

### gdal3.js (GDAL WASM)

More actively maintained (~415 stars, updated Feb 2026). Viable if you absolutely need GDAL in the browser.

**Key caveats:** 10MB+ WASM binary, 5-10 second cold start, browser memory limits, not all GDAL features work. For anything beyond simple format conversion, process rasters server-side with real GDAL.

---

## Decision Flowchart

```
Do you need spatial operations (buffer, intersect, etc.)?
+-- YES -> Turf.js (covers 90% of cases)
|         +-- Need topological correctness? -> Add JSTS
+-- NO -> Do you need spatial search on >10K features?
    +-- YES -> Static data? -> Flatbush / kdbush (points)
    |          Dynamic data? -> rbush
    +-- NO -> Need coordinate transformation? -> proj4js
             Need hexagonal binning? -> h3-js
             Need raster data? -> geotiff.js
             None of the above? -> You don't need a spatial analysis library
```
