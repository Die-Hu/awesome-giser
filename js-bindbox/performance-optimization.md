# Performance & Optimization Guide

**Quick Picks**: MapLibre GL JS + deck.gl for rendering (WebGL2, GPU-native), DuckDB WASM for in-browser analytical queries, Web Workers for heavy spatial operations that would block the main thread. These three together cover ~90% of geo web app performance problems.

---

## Rendering Performance Benchmarks

### Feature Count Limits by Library

Real-world approximate benchmarks measured on a mid-range desktop (Intel i7, NVIDIA GTX 1060, 16 GB RAM, Chrome 120). Numbers degrade on mobile — see the Mobile section.

| Library | Renderer | 1K features | 10K | 100K | 1M | 10M |
|---------|----------|-------------|-----|------|-----|-----|
| Leaflet (SVG) | SVG DOM | 60fps | 30fps | 5fps | crash | crash |
| Leaflet (Canvas) | Canvas 2D | 60fps | 60fps | 30fps | 5fps | crash |
| MapLibre GL JS | WebGL | 60fps | 60fps | 60fps | 50fps* | 30fps* |
| OpenLayers (Canvas) | Canvas 2D | 60fps | 60fps | 40fps | 10fps | crash |
| OpenLayers (WebGL) | WebGL | 60fps | 60fps | 60fps | 55fps | 40fps |
| deck.gl | WebGL2 | 60fps | 60fps | 60fps | 60fps | 50fps |
| CesiumJS (Entity) | WebGL | 60fps | 30fps | 5fps | crash | crash |
| CesiumJS (Primitive) | WebGL | 60fps | 60fps | 50fps | 30fps | N/A |

*via vector tiles, not raw GeoJSON

### Why These Numbers Matter

Understanding the rendering pipeline explains why these limits exist:

**Data → Buffer → GPU → Screen**

1. **Data stage**: raw GeoJSON/coordinates are parsed in JS into arrays
2. **Buffer stage**: coordinate data is packed into typed arrays (Float32Array, Uint16Array) and uploaded to GPU memory as vertex buffer objects (VBOs)
3. **GPU stage**: vertex shader transforms coordinates, fragment shader fills pixels
4. **Screen**: the composited frame is displayed

**Why SVG dies at 10K features**: Each feature becomes a DOM element. At 10K points you have 10K live DOM nodes. Every pan/zoom triggers layout reflow across all nodes. The browser's rendering engine is not designed for this — DOM diffing, style recalculation, and paint invalidation all scale O(n) or worse. At 30K+ features the main thread is perpetually blocked.

**Why Canvas 2D struggles at 100K features**: Canvas avoids DOM overhead but every frame requires iterating all features in JS and issuing one `drawImage()`/`fillRect()` call per feature. At 100K features you're making 100K JS → native bridge calls per 16ms frame budget. JS-to-Canvas bridge calls are not free — each one involves type checking and bounds validation. The main thread still does all the work.

**Why WebGL wins at 1M+ features**: All geometry is uploaded once to the GPU as a single buffer. Per-frame rendering is a single GPU draw call (`gl.drawArrays(GL_POINTS, 0, 1000000)`). The GPU's thousands of shader cores process all vertices in parallel. The main thread only issues a handful of WebGL API calls per frame — the heavy lifting is entirely on the GPU.

**Why vector tiles make 1M+ features at 60fps possible**: Vector tiles implement Level of Detail (LOD) — at zoom 10 you see simplified country outlines, at zoom 15 you see individual buildings. Only features in the current viewport tile are loaded. GPU batching means all features from one tile are rendered in a single draw call. The client never holds 1M raw geometries in memory — it holds the tiles covering the current viewport, which might be 10K-50K features max.

### Memory Benchmarks

| Data | In-memory size | GeoJSON parse time | Notes |
|------|---------------|-------------------|-------|
| 10K points (GeoJSON) | ~5 MB | ~50ms | Fine for main thread |
| 100K points (GeoJSON) | ~50 MB | ~500ms | Main thread blocks visibly |
| 1M points (GeoJSON) | ~500 MB | ~5s | Tab may crash on mobile |
| 10K polygons (avg 50 vertices) | ~30 MB | ~200ms | Each polygon is heavier than a point |
| Overture buildings GeoParquet (city) | ~200 MB raw, ~50 MB Parquet | Streaming | Use DuckDB WASM |
| GeoParquet 1M rows (columnar, zstd) | ~80 MB on disk | Streaming | Only read needed columns |
| 100K H3 cells (resolution 9) | ~3 MB (uint64 IDs) | — | Use H3 binary encoding |

**Parse time caveat**: JSON.parse is synchronous and single-threaded. A 500ms parse on the main thread means 500ms of frozen UI — no panning, no clicking, no scrolling. Always parse large GeoJSON in a Web Worker.

---

## Data Loading Strategies

### GeoJSON

Choose a strategy based on data size:

**Small data (<1 MB)**: Fetch directly and add to map source. Simple, zero dependencies.

```js
const res = await fetch('/api/features.geojson');
const geojson = await res.json();
map.addSource('my-source', { type: 'geojson', data: geojson });
```

**Medium data (1-50 MB)**: Use a streaming JSON parser to avoid blocking the main thread during parse. Libraries: `oboe.js` (streaming SAX-style) or `clarinet`. Better: parse in a Web Worker (see Workers section).

```js
// oboe.js progressive loading
import oboe from 'oboe';

oboe('/api/large-features.geojson')
  .node('features.*', (feature) => {
    // process each feature as it arrives — don't block UI
    batchBuffer.push(feature);
    if (batchBuffer.length >= 500) flushBatch();
  })
  .done(() => flushBatch());
```

**Large data (50 MB+)**: Do NOT use GeoJSON. Switch to vector tiles (PMTiles) for display or GeoParquet + DuckDB WASM for analysis. GeoJSON at this scale will crash mobile browsers and create a poor experience on desktop.

### Vector Tiles (MVT / PMTiles)

Vector tiles are the correct format for any dataset you need to display interactively at scale.

**Static data workflow** (pre-tiled, best performance):

```bash
# 1. Convert GeoJSON to PMTiles with tippecanoe
tippecanoe \
  --output=buildings.pmtiles \
  --layer=buildings \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  --maximum-zoom=16 \
  --minimum-zoom=8 \
  buildings.geojson

# 2. Serve the PMTiles file (S3, Cloudflare R2, or any static host)
# 3. Reference it in MapLibre
```

```js
// MapLibre with PMTiles
import maplibregl from 'maplibre-gl';
import { Protocol } from 'pmtiles';

const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({ /* ... */ });
map.addSource('buildings', {
  type: 'vector',
  url: 'pmtiles://https://example.com/buildings.pmtiles'
});
map.addLayer({
  id: 'buildings-fill',
  type: 'fill',
  source: 'buildings',
  'source-layer': 'buildings',
  paint: { 'fill-color': '#aaa' }
});
```

**Live data workflow** (dynamic tiles from PostGIS):

```
PostGIS → Martin tile server → MapLibre GL JS
```

Martin generates MVT tiles on-demand from PostGIS. Suitable for data that changes frequently. Add a Redis cache in front of Martin for high traffic.

**Why vector tiles win**:
- Only tiles covering the current viewport are requested (often 10-20 tiles for a city view)
- Each tile is GPU-native: uploaded once as a buffer, rendered as a batch
- Automatic LOD: simplified geometries at low zoom, full detail at high zoom
- Cacheable: tiles are immutable once generated

### FlatGeobuf

FlatGeobuf is a binary format that supports HTTP range requests for spatial filtering. Unlike GeoJSON, you can fetch only the features intersecting a bounding box without downloading the whole file.

**When to use FlatGeobuf vs PMTiles**:
- FlatGeobuf: single static file on any HTTP server, load-by-bounding-box queries, no pre-tiling step needed, good for datasets up to ~500 MB
- PMTiles: pre-tiled (tippecanoe step required), faster rendering (GPU batching by tile), better for larger datasets or frequently accessed data

```js
// FlatGeobuf spatial filter — load only features in current viewport
import { deserialize } from 'flatgeobuf';

async function loadFeaturesInViewport(map) {
  const bounds = map.getBounds();
  const bbox = {
    minX: bounds.getWest(),
    minY: bounds.getSouth(),
    maxX: bounds.getEast(),
    maxY: bounds.getNorth()
  };

  const response = await fetch('https://example.com/data.fgb');
  const fc = { type: 'FeatureCollection', features: [] };

  // deserialize is async-iterable — streams features from range requests
  for await (const feature of deserialize(response.body, bbox)) {
    fc.features.push(feature);
  }

  // Update map source with only viewport features
  map.getSource('my-source').setData(fc);
}

map.on('moveend', () => loadFeaturesInViewport(map));
```

```js
// Same pattern with Leaflet
import { deserialize } from 'flatgeobuf';
import L from 'leaflet';

let currentLayer = null;

async function loadFlatGeobuf(map) {
  const bounds = map.getBounds();
  const bbox = {
    minX: bounds.getWest(), minY: bounds.getSouth(),
    maxX: bounds.getEast(), maxY: bounds.getNorth()
  };

  const response = await fetch('https://example.com/data.fgb');
  const features = [];
  for await (const f of deserialize(response.body, bbox)) {
    features.push(f);
  }

  if (currentLayer) map.removeLayer(currentLayer);
  currentLayer = L.geoJSON({ type: 'FeatureCollection', features }).addTo(map);
}
```

### GeoParquet + DuckDB WASM

DuckDB WASM brings a full analytical SQL engine to the browser. Combined with the `duckdb-spatial` extension, you can run spatial queries against GeoParquet files hosted on any static server — no backend required.

**Performance**: DuckDB WASM can query 100M row GeoParquet files in ~2 seconds using HTTP range requests and columnar scan. You only download the columns and row groups you need.

```js
import * as duckdb from '@duckdb/duckdb-wasm';

async function initDuckDB() {
  const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
  const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);
  const worker = new Worker(bundle.mainWorker);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);
  return db;
}

async function queryGeoParquet(db, viewportWKT) {
  const conn = await db.connect();

  // Load spatial extension once
  await conn.query("LOAD spatial;");

  // Query only features within viewport — DuckDB uses column pruning and
  // row group filtering to avoid downloading irrelevant data
  const result = await conn.query(`
    SELECT
      id,
      name,
      ST_AsGeoJSON(geometry) AS geom_json,
      population
    FROM read_parquet('https://example.com/cities.parquet')
    WHERE ST_Within(
      geometry,
      ST_GeomFromText('${viewportWKT}')
    )
    AND population > 10000
    ORDER BY population DESC
    LIMIT 5000
  `);

  // Convert Arrow result to GeoJSON features
  const features = result.toArray().map(row => ({
    type: 'Feature',
    geometry: JSON.parse(row.geom_json),
    properties: { id: row.id, name: row.name, population: Number(row.population) }
  }));

  await conn.close();
  return { type: 'FeatureCollection', features };
}
```

**Aggregation before rendering** (the real power move):

```js
// Aggregate 10M points into H3 hexagons in DuckDB before rendering
const result = await conn.query(`
  LOAD spatial;
  SELECT
    h3_latlng_to_cell(ST_Y(geometry), ST_X(geometry), 7) AS h3_cell,
    COUNT(*) AS count,
    AVG(value) AS avg_value
  FROM read_parquet('https://example.com/observations.parquet')
  WHERE ST_Within(geometry, ST_GeomFromText('${viewportWKT}'))
  GROUP BY h3_cell
`);
// Now render a few thousand hexagons instead of 10M points
```

### Apache Arrow / GeoArrow

Arrow IPC format enables zero-copy data transfer between WASM modules and JavaScript typed arrays. deck.gl supports GeoArrow tables natively — you can load binary geometry directly into the GPU without an intermediate GeoJSON step.

```js
import * as arrow from 'apache-arrow';
import { ScatterplotLayer } from '@deck.gl/layers';

// Fetch Arrow IPC file (binary, columnar — much faster than GeoJSON)
const response = await fetch('https://example.com/points.arrow');
const buffer = await response.arrayBuffer();
const table = arrow.tableFromIPC(buffer);

// deck.gl can consume Arrow tables directly — geometry never becomes JS objects
const layer = new ScatterplotLayer({
  id: 'points',
  data: table,
  getPosition: { object: table.getChild('geometry') }, // GeoArrow geometry column
  getRadius: 5,
  getFillColor: [255, 140, 0]
});
```

**lonboard path (Python → browser)**:

```python
# Python: convert GeoPandas → Arrow IPC → browser via lonboard
import geopandas as gpd
import lonboard

gdf = gpd.read_file('my_data.gpkg')
# lonboard serializes GeoDataFrame to Arrow IPC and renders via deck.gl
map = lonboard.Map(layers=[lonboard.ScatterplotLayer.from_geopandas(gdf)])
map  # Jupyter display
```

### Cloud-Optimized GeoTIFF (COG)

COGs support HTTP range requests — you can load only the portion of a raster covering the viewport.

```js
import { fromUrl } from 'geotiff';

async function loadCOGTile(cogUrl, bbox, width, height) {
  const tiff = await fromUrl(cogUrl);
  const image = await tiff.getImage();

  // Read only the window covering bbox — range request, not full file
  const window = image.getBoundingBox(); // [west, south, east, north]
  const data = await image.readRasters({
    bbox,          // viewport bounding box
    width,         // output pixel width
    height,        // output pixel height
    interleave: true
  });

  return data;
}

// Band math in browser (NDVI from Sentinel-2 COG)
function computeNDVI(redBand, nirBand) {
  return redBand.map((r, i) => {
    const nir = nirBand[i];
    return (nir - r) / (nir + r + 1e-10);
  });
}
```

For production tile serving from COG, use **TiTiler** (FastAPI + rio-tiler). It serves dynamic raster tiles from any COG URL:

```
https://titiler.xyz/cog/tiles/{z}/{x}/{y}?url=https://example.com/raster.tif
```

---

## Rendering Optimization Techniques

### Vector Tile Optimization

**tippecanoe flags for production tiles**:

```bash
tippecanoe \
  --output=output.pmtiles \
  --drop-densest-as-needed \      # auto-drop features at low zoom by density
  --extend-zooms-if-still-dropping \  # add zoom levels until features fit
  --coalesce-fraction \           # merge small polygons into neighbors
  --simplification=4 \            # Douglas-Peucker simplification factor
  --minimum-zoom=6 \
  --maximum-zoom=15 \
  --layer=mydata \
  input.geojson
```

**Tile size**: 512px tiles mean 4x fewer tile requests than 256px tiles for the same viewport (MapLibre default is 512px). Don't change this unless you have a specific reason.

**Source maxzoom**: Set `maxzoom` on your source. MapLibre will use the max-zoom tile for higher zoom levels (overzooming) instead of requesting new tiles. This prevents unnecessary requests when your data doesn't benefit from higher zoom.

```js
map.addSource('my-source', {
  type: 'vector',
  url: 'pmtiles://https://example.com/data.pmtiles',
  maxzoom: 14  // stop requesting new tiles above zoom 14
});
```

**Filter expressions**: Apply filters in the layer style, not in data pre-processing. MapLibre evaluates filters on the GPU — no JS overhead.

```js
map.addLayer({
  id: 'major-roads',
  type: 'line',
  source: 'roads',
  'source-layer': 'roads',
  filter: ['>=', ['get', 'road_class'], 3],  // GPU-evaluated, free
  paint: { 'line-color': '#555' }
});
```

### MapLibre GL JS Optimization

**Feature state for hover/selection** — the single biggest win for interactive maps:

```js
// WRONG: re-creating GeoJSON source data on every hover (triggers full re-render)
map.getSource('features').setData(updatedGeojson);

// RIGHT: setFeatureState — only updates paint properties, no geometry re-upload
map.on('mousemove', 'features-layer', (e) => {
  if (e.features.length > 0) {
    if (hoveredId !== null) {
      map.setFeatureState({ source: 'features', id: hoveredId }, { hover: false });
    }
    hoveredId = e.features[0].id;
    map.setFeatureState({ source: 'features', id: hoveredId }, { hover: true });
  }
});

// Paint property driven by feature state — evaluated per-feature on GPU
map.addLayer({
  id: 'features-layer',
  type: 'fill',
  source: 'features',
  paint: {
    'fill-color': [
      'case',
      ['boolean', ['feature-state', 'hover'], false],
      '#f00',  // hovered: red
      '#00f'   // default: blue
    ]
  }
});
```

**Efficient click detection**:

```js
map.on('click', (e) => {
  // queryRenderedFeatures is GPU-assisted — fast even with 100K features loaded
  const features = map.queryRenderedFeatures(e.point, {
    layers: ['my-layer'],
    filter: ['==', ['get', 'type'], 'building']  // filter reduces candidates
  });
  if (features.length > 0) handleFeatureClick(features[0]);
});
```

**Clustering on the source** (GPU-side, no JS overhead):

```js
map.addSource('points', {
  type: 'geojson',
  data: '/api/points.geojson',
  cluster: true,
  clusterMaxZoom: 14,
  clusterRadius: 50,
  clusterProperties: {
    // aggregate properties across cluster members
    'sum_count': ['+', ['get', 'count']]
  }
});
```

**Layer management best practices**:

```js
// Prefer toggling visibility over adding/removing layers
// Adding/removing triggers shader recompilation; toggling visibility does not
map.setLayoutProperty('my-layer', 'visibility', 'none');  // fast
map.setLayoutProperty('my-layer', 'visibility', 'visible');  // fast

// vs (slow on first add after remove — shader recompile)
map.removeLayer('my-layer');
map.addLayer(layerDef);

// Merge multiple layers from the same source using expressions
// 6 layers → 1 layer with conditional paint = fewer draw calls
map.addLayer({
  id: 'roads-combined',
  type: 'line',
  source: 'roads',
  'source-layer': 'roads',
  paint: {
    'line-width': [
      'match', ['get', 'class'],
      'motorway', 4,
      'primary', 2,
      1  // default
    ],
    'line-color': [
      'match', ['get', 'class'],
      'motorway', '#f00',
      'primary', '#fa0',
      '#ccc'
    ]
  }
});
```

**Other MapLibre tips**:

```js
// Instant tile transitions (no fade) — useful for real-time data
map.addSource('live', {
  type: 'raster',
  tiles: ['https://tiles.example.com/{z}/{x}/{y}.png'],
  tileSize: 256
});
map.addLayer({
  id: 'live-raster',
  type: 'raster',
  source: 'live',
  paint: { 'raster-fade-duration': 0 }  // no cross-fade animation
});

// Preload a target area before animating there
map.once('idle', () => {
  map.flyTo({ center: [lng, lat], zoom: 15 });
});
```

### Leaflet Optimization

**Always use `preferCanvas: true`** when you have more than 500 markers:

```js
// Initialize map with Canvas renderer (10x more markers than SVG)
const map = L.map('map', { preferCanvas: true });

// Per-layer Canvas override
const canvasRenderer = L.canvas({ padding: 0.5 });
const layer = L.geoJSON(data, { renderer: canvasRenderer });
```

**Chunked marker addition** — prevent frame drops when adding thousands of markers:

```js
function addMarkersInChunks(markers, chunkSize = 500) {
  const group = L.featureGroup().addTo(map);
  let index = 0;

  function addChunk() {
    const end = Math.min(index + chunkSize, markers.length);
    for (; index < end; index++) {
      L.circleMarker(markers[index].latlng, {
        radius: 5,
        renderer: L.canvas()
      }).addTo(group);
    }
    if (index < markers.length) {
      // Yield to browser between chunks — keeps UI responsive
      requestIdleCallback(addChunk);
    }
  }

  addChunk();
  return group;
}
```

**Marker clustering** with Leaflet.markercluster:

```js
import 'leaflet.markercluster';

const mcg = L.markerClusterGroup({
  chunkedLoading: true,       // don't block main thread
  chunkInterval: 200,         // ms between chunks
  chunkDelay: 50,             // ms delay between chunk batches
  disableClusteringAtZoom: 16 // show individual markers at high zoom
});

markers.forEach(m => mcg.addLayer(L.marker(m.latlng)));
map.addLayer(mcg);
```

### deck.gl Optimization

**Disable picking on non-interactive layers** — the picking pass (off-screen framebuffer render) is expensive:

```js
new ScatterplotLayer({
  id: 'background-points',
  data: backgroundData,
  pickable: false,  // skip picking pass for this layer — saves ~2ms per frame
  getPosition: d => d.coords,
  getRadius: 3
});
```

**updateTriggers to prevent unnecessary re-creation**:

```js
// Without updateTriggers: deck.gl can't tell if accessor functions changed
// — it re-creates the layer every render (re-uploads all GPU buffers)

// With updateTriggers: deck.gl only re-creates when the trigger values change
new ScatterplotLayer({
  id: 'points',
  data: points,
  getFillColor: d => colorScale(d.value),
  updateTriggers: {
    getFillColor: [colorScale.domain(), colorScale.range()]  // re-run only when these change
  }
});
```

**GPU-side filtering with DataFilterExtension** — filter millions of features without copying data:

```js
import { DataFilterExtension } from '@deck.gl/extensions';

new ScatterplotLayer({
  id: 'filtered-points',
  data: allPoints,  // all 1M points stay in GPU memory
  extensions: [new DataFilterExtension({ filterSize: 1 })],
  getFilterValue: d => d.timestamp,           // value to filter on
  filterRange: [startTime, endTime],          // filter range — updated without data copy
  getPosition: d => [d.lng, d.lat],
  getRadius: 5,
  pickable: true
});
```

**Pre-aggregate with H3 before rendering**:

```js
import * as h3 from 'h3-js';
import { H3HexagonLayer } from '@deck.gl/geo-layers';

// Aggregate 1M points into H3 cells (do this in a Worker)
function aggregateToH3(points, resolution) {
  const counts = new Map();
  for (const pt of points) {
    const cell = h3.latLngToCell(pt.lat, pt.lng, resolution);
    counts.set(cell, (counts.get(cell) ?? 0) + 1);
  }
  return Array.from(counts, ([cell, count]) => ({ cell, count }));
}

// Render a few thousand hexagons instead of 1M points
const hexData = aggregateToH3(rawPoints, 7);  // ~500 hexagons for a city

new H3HexagonLayer({
  id: 'hex-layer',
  data: hexData,
  getHexagon: d => d.cell,
  getElevation: d => d.count,
  getFillColor: d => colorScale(d.count),
  extruded: true
});
```

**Binary data for maximum throughput**:

```js
// Slow: JS objects with accessor functions
const layer = new ScatterplotLayer({
  data: points,  // [{lng, lat, r, g, b}, ...]
  getPosition: d => [d.lng, d.lat],
  getFillColor: d => [d.r, d.g, d.b, 255]
});

// Fast: typed arrays — no accessor function overhead, direct buffer upload
const positions = new Float32Array(points.flatMap(p => [p.lng, p.lat, 0]));
const colors = new Uint8Array(points.flatMap(p => [p.r, p.g, p.b, 255]));

const layer = new ScatterplotLayer({
  data: { length: points.length },
  getPosition: { value: positions, size: 3 },
  getFillColor: { value: colors, size: 4 }
});
```

### CesiumJS Optimization

**Request render mode** — stop re-rendering when the scene is static (huge battery savings):

```js
const viewer = new Cesium.Viewer('cesiumContainer', {
  requestRenderMode: true,
  maximumRenderTimeChange: Infinity  // only render when explicitly requested
});

// Manually request a render when data changes
viewer.scene.requestRender();

// Or set a moderate value for semi-static scenes
const viewer2 = new Cesium.Viewer('cesiumContainer', {
  requestRenderMode: true,
  maximumRenderTimeChange: 0.1  // re-render if scene changes by more than 0.1s
});
```

**Primitive API vs Entity API** (10x performance difference):

```js
// SLOW: Entity API — high-level, convenient, not for 10K+ features
for (const building of buildings) {
  viewer.entities.add({
    position: Cesium.Cartesian3.fromDegrees(building.lng, building.lat),
    box: {
      dimensions: new Cesium.Cartesian3(building.width, building.depth, building.height),
      material: Cesium.Color.GRAY
    }
  });
}

// FAST: Primitive API — low-level, batched GPU upload
const geometryInstances = buildings.map(b =>
  new Cesium.GeometryInstance({
    geometry: new Cesium.BoxGeometry({
      vertexFormat: Cesium.VertexFormat.DEFAULT,
      dimensions: new Cesium.Cartesian3(b.width, b.depth, b.height)
    }),
    modelMatrix: Cesium.Matrix4.multiplyByTranslation(
      Cesium.Transforms.eastNorthUpToFixedFrame(
        Cesium.Cartesian3.fromDegrees(b.lng, b.lat)
      ),
      new Cesium.Cartesian3(0, 0, b.height / 2),
      new Cesium.Matrix4()
    )
  })
);

viewer.scene.primitives.add(
  new Cesium.Primitive({
    geometryInstances,  // all buildings in ONE draw call
    appearance: new Cesium.PerInstanceColorAppearance()
  })
);
```

**3D Tiles LOD tuning**:

```js
const tileset = await Cesium.Cesium3DTileset.fromUrl(url, {
  maximumMemoryUsage: 512,     // MB — evict tiles when GPU memory exceeds this
  maximumScreenSpaceError: 8,  // lower = higher quality, higher = faster
  skipLevelOfDetail: true,     // jump zoom levels for faster traversal
  immediatelyLoadDesiredLevelOfDetail: false,
  loadSiblings: false          // don't load adjacent tiles speculatively
});
viewer.scene.primitives.add(tileset);
```

---

## Web Workers for Heavy Operations

### Architecture Pattern

```
Main Thread (60fps UI budget: 16.67ms per frame)
├── Map rendering (MapLibre/Leaflet/deck.gl)
├── User interactions (pan, zoom, click)
├── Lightweight state management
└── Postmessage coordination

Worker Thread(s)  (no UI budget constraint)
├── Turf.js spatial operations (buffer, intersect, union)
├── DuckDB WASM queries
├── geotiff.js raster decoding and band math
├── H3 aggregation (millions of points)
├── Data format conversion (CSV → GeoJSON, WKT parsing)
├── Custom analytics and statistics
└── GeoParquet parsing
```

**Golden rule**: if a JS operation takes more than 5ms, run it in a Worker. The main thread has a 16.67ms budget for a full frame (map render + event handling + layout). Any heavy computation steals from that budget and drops frames.

### Using comlink for Ergonomic Worker RPC

```js
// worker.js
import { expose } from 'comlink';
import * as turf from '@turf/turf';

const api = {
  async bufferFeatures(geojson, radiusKm) {
    // This runs in the worker thread — no main thread blocking
    return turf.buffer(geojson, radiusKm, { units: 'kilometers' });
  },

  async dissolveByProperty(geojson, property) {
    return turf.dissolve(geojson, { propertyName: property });
  },

  async spatialJoin(targetGeojson, joinGeojson) {
    // Heavy operation — may take 500ms+, fine in worker
    return turf.tag(targetGeojson, joinGeojson, 'zone', 'zone_id');
  }
};

expose(api);
```

```js
// main.js
import { wrap } from 'comlink';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const geoWorker = wrap(worker);

// Usage: looks like a normal async function call
// but actually runs in the worker thread
async function runBufferAnalysis() {
  const result = await geoWorker.bufferFeatures(inputGeojson, 5);
  map.getSource('buffer-result').setData(result);
}
```

### Transferable Objects (Zero-Copy)

Transferring a 50 MB `ArrayBuffer` from worker to main thread normally copies all 50 MB. Using `Transferable`, you transfer ownership instead — zero copy, near-instant.

```js
// worker.js
self.onmessage = async (e) => {
  const { rasterUrl, bbox } = e.data;

  // Decode raster in worker
  const tiff = await fromUrl(rasterUrl);
  const image = await tiff.getImage();
  const [band] = await image.readRasters({ bbox });

  const buffer = band.buffer;  // ArrayBuffer

  // Transfer ownership — no copy, buffer is now invalid in worker
  self.postMessage({ buffer, width: image.getWidth(), height: image.getHeight() }, [buffer]);
};
```

```js
// main.js
worker.onmessage = (e) => {
  const { buffer, width, height } = e.data;
  const pixels = new Float32Array(buffer);  // instant — no copy
  renderRaster(pixels, width, height);
};
```

### SharedArrayBuffer for Parallel Processing

```js
// Parallel raster processing with SharedArrayBuffer
// Requires: Cross-Origin-Opener-Policy: same-origin
//           Cross-Origin-Embedder-Policy: require-corp

const TILE_SIZE = 256;
const shared = new SharedArrayBuffer(TILE_SIZE * TILE_SIZE * 4); // Float32
const arr = new Float32Array(shared);

// Spin up 4 workers for parallel band processing
const workers = Array.from({ length: 4 }, () =>
  new Worker(new URL('./raster-worker.js', import.meta.url))
);

// Each worker processes a quarter of the raster
workers.forEach((w, i) => {
  w.postMessage({ shared, offset: i * (TILE_SIZE * TILE_SIZE), chunk: TILE_SIZE * TILE_SIZE / 4 });
});
```

### Worker Pool Pattern

```js
class WorkerPool {
  constructor(workerUrl, size = navigator.hardwareConcurrency) {
    this.workers = Array.from({ length: size }, () =>
      new Worker(workerUrl, { type: 'module' })
    );
    this.queue = [];
    this.idle = [...this.workers];
  }

  run(task) {
    return new Promise((resolve, reject) => {
      const execute = (worker) => {
        worker.onmessage = (e) => {
          resolve(e.data);
          if (this.queue.length > 0) {
            execute(worker, this.queue.shift());
          } else {
            this.idle.push(worker);
          }
        };
        worker.onerror = reject;
        worker.postMessage(task);
      };

      if (this.idle.length > 0) {
        execute(this.idle.pop());
      } else {
        this.queue.push(task);
      }
    });
  }
}

// Usage
const pool = new WorkerPool(new URL('./tile-decoder.js', import.meta.url));
const decodedTiles = await Promise.all(tileUrls.map(url => pool.run({ url })));
```

---

## Bundle Size Optimization

### Library Sizes (gzip)

| Library | Full bundle | Tree-shaken min | CDN |
|---------|-------------|-----------------|-----|
| Leaflet | 42 KB | 42 KB (not tree-shakable) | Yes |
| MapLibre GL JS | 220 KB | ~200 KB | Yes |
| OpenLayers | 160 KB | ~60 KB (with tree-shaking) | No |
| deck.gl (core + layers) | ~300 KB | ~100 KB (few layers) | No |
| CesiumJS | ~3 MB | ~2 MB | Kind of |
| Turf.js | ~180 KB | ~3-20 KB per module | No |
| D3 (d3-geo only) | ~30 KB | ~15 KB | Yes |
| ECharts | ~300 KB | ~100 KB (basic) | Yes |
| H3-js | ~80 KB | ~80 KB (WASM) | No |
| DuckDB WASM | ~4 MB | N/A | No |

### Strategies

**Dynamic import** — load map only on map pages:

```js
// Pages without a map pay zero cost for MapLibre
async function initMap(container) {
  const [{ default: maplibregl }, { Protocol }] = await Promise.all([
    import('maplibre-gl'),
    import('pmtiles')
  ]);
  // now initialize map
}
```

**Turf.js tree-shaking** — import only what you use:

```js
// WRONG: imports entire Turf bundle (~180 KB gzip)
import * as turf from '@turf/turf';

// RIGHT: import only needed modules (~3-20 KB each)
import { buffer } from '@turf/buffer';
import { intersect } from '@turf/intersect';
import { area } from '@turf/area';
```

**Vite chunk splitting for geo dependencies**:

```js
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'maplibre': ['maplibre-gl'],
          'deck': ['@deck.gl/core', '@deck.gl/layers', '@deck.gl/geo-layers'],
          'duckdb': ['@duckdb/duckdb-wasm'],
          'geo-utils': ['@turf/buffer', '@turf/intersect', 'h3-js']
        }
      }
    }
  }
};
```

**Module preload for critical map chunks**:

```html
<!-- Preload MapLibre chunk before user navigates to map page -->
<link rel="modulepreload" href="/assets/maplibre-abc123.js">
<link rel="modulepreload" href="/assets/maplibre-abc123.css" as="style">
```

**CDN for development, bundled for production**:

```js
// Development: use CDN for fast iteration (no local install)
// <script src="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js"></script>

// Production: bundle for reliability + tree-shaking
import maplibregl from 'maplibre-gl';
```

---

## Mobile Performance

### Constraints

Mobile devices have significantly different performance characteristics:

- **GPU**: Mobile GPUs (Adreno 740, Apple A17 GPU) are 5-10x slower than desktop discrete GPUs for sustained workloads. Sustained performance also throttles due to heat — performance at minute 5 can be 30-50% of performance at minute 0
- **Memory**: Mobile browser tabs are killed at 2-4 GB RAM. Aggressive GC pauses are common
- **WebGL limits**: Mobile GPUs have stricter `MAX_TEXTURE_SIZE`, `MAX_VERTEX_ATTRIBS`, and `MAX_UNIFORM_VECTORS` limits
- **Battery**: Continuous 60fps WebGL rendering can drain a phone battery in 1-2 hours
- **Network**: 4G/5G has high variance latency; tile requests that take 50ms on broadband may take 300ms+ on mobile

### Mobile-Specific Optimizations

**Detect device capability and adapt**:

```js
const isLowEnd =
  navigator.hardwareConcurrency <= 2 ||
  navigator.deviceMemory <= 2 ||
  /Android [4-7]/.test(navigator.userAgent);

const mapConfig = isLowEnd
  ? {
      maxZoom: 16,          // fewer zoom levels = fewer tile variants to cache
      tileSize: 256,        // smaller tiles = less GPU memory per tile
      antialias: false,     // disable MSAA (saves ~20% GPU load)
      fadeDuration: 0       // instant tile transitions (no alpha blending)
    }
  : {
      maxZoom: 22,
      tileSize: 512,
      antialias: true,
      fadeDuration: 300
    };

const map = new maplibregl.Map({ container: 'map', ...mapConfig });
```

**Disable terrain on low-end devices**:

```js
if (!isLowEnd) {
  map.setTerrain({ source: 'terrain-dem', exaggeration: 1.5 });
}
```

**Prevent accidental scroll capture**:

```js
// MapLibre: require two fingers to pan the map on touch devices
const map = new maplibregl.Map({
  cooperativeGestures: true  // shows overlay message when single finger is used
});
```

**Reduce tile requests at high zoom**:

```js
map.addSource('detailed', {
  type: 'vector',
  url: 'pmtiles://https://example.com/data.pmtiles',
  maxzoom: 14,       // don't request tiles above zoom 14
  minzoom: 10        // don't render below zoom 10
});
```

**Battery-conscious rendering** (CesiumJS):

```js
// Only render when scene changes — critical for battery life on mobile
const viewer = new Cesium.Viewer('cesiumContainer', {
  requestRenderMode: true,
  maximumRenderTimeChange: 0.5
});

// Re-render on user interaction only
document.getElementById('cesiumContainer').addEventListener('touchstart', () => {
  viewer.scene.requestRender();
});
```

**Viewport meta for correct touch targets**:

```html
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
```

---

## Monitoring & Profiling

### Chrome DevTools

**Performance tab workflow for map apps**:

1. Open DevTools → Performance tab
2. Click Record
3. Pan/zoom the map 3-5 times, click a few features
4. Stop recording
5. Look for: long tasks (red triangles), frame drops below 60fps, main thread blocking

Key things to find:
- `JSONparse` / `JSON.parse` taking >50ms → move to Worker
- `drawElements` / `drawArrays` taking >5ms → too many draw calls or too much geometry
- `XHR` or `fetch` blocking on main thread → should be async already, check for sync usage
- GC events causing frame drops → memory pressure, check for GeoJSON leaks

**Memory tab for GeoJSON leaks**:

```
1. DevTools → Memory → Take heap snapshot (before loading data)
2. Load and interact with your map data
3. Take another snapshot
4. Use "Comparison" view — look for GeoJSON objects that weren't in snapshot 1
5. Common leak: event listeners holding references to large feature arrays
```

**Checking WebGL limits**:

Navigate to `chrome://gpu` for:
- WebGL renderer (integrated vs discrete GPU)
- Max texture size (1024 on low-end mobile vs 16384 on desktop)
- Whether WebGL2 is available (required for deck.gl)

**Tile waterfall analysis**:

DevTools Network tab → filter by `Fetch/XHR` → check:
- Tile request timing (TTFB should be <100ms for cached tiles)
- Concurrent tile requests (HTTP/2 allows many, HTTP/1.1 limited to 6 per domain)
- Tile response sizes (MVT should be <100KB per tile, warn if >500KB)

### Map-Specific Debug Tools

**MapLibre built-in debug**:

```js
// Show tile boundaries and IDs
map.showTileBoundaries = true;

// Show collision boxes for symbol layers (labels, icons)
map.showCollisionBoxes = true;

// Show padding boxes
map.showPadding = true;

// Log render stats to console
map.on('render', () => {
  const stats = map.painter.renderStats;
  if (stats) console.log('Draw calls:', stats.numDrawCalls);
});
```

**CesiumJS Inspector**:

```js
// Add Cesium Inspector widget (comprehensive scene stats)
viewer.extend(Cesium.viewerCesiumInspectorMixin);

// Or just show FPS counter
viewer.scene.debugShowFramesPerSecond = true;
```

**deck.gl stats widget**:

```js
import { log } from '@deck.gl/core';

// Enable debug mode (shows layer stats)
const deck = new Deck({
  _debug: true,
  layers: [...]
});

// Access internal stats
deck.stats.get('CPU Time').lastTiming;
deck.stats.get('GPU Time').lastTiming;
```

### Custom Performance Metrics

Track these metrics for your geo app:

```js
class MapPerformanceMonitor {
  constructor(map) {
    this.map = map;
    this.metrics = {};
    this.setupListeners();
  }

  setupListeners() {
    // Time to first tile rendered
    const firstTileStart = performance.now();
    this.map.once('data', (e) => {
      if (e.dataType === 'tile') {
        this.metrics.timeToFirstTile = performance.now() - firstTileStart;
        console.log(`First tile: ${this.metrics.timeToFirstTile.toFixed(0)}ms`);
      }
    });

    // Time to interactive (all visible tiles loaded)
    const loadStart = performance.now();
    this.map.once('idle', () => {
      this.metrics.timeToInteractive = performance.now() - loadStart;
      console.log(`Time to interactive: ${this.metrics.timeToInteractive.toFixed(0)}ms`);
    });

    // Track FPS during interaction
    let lastTime = performance.now();
    let frameCount = 0;
    this.map.on('render', () => {
      frameCount++;
      const now = performance.now();
      if (now - lastTime >= 1000) {
        this.metrics.fps = frameCount;
        frameCount = 0;
        lastTime = now;
      }
    });
  }

  // Frame budget analysis: 16.67ms for 60fps
  measureOperation(name, fn) {
    const start = performance.now();
    const result = fn();
    const elapsed = performance.now() - start;
    if (elapsed > 16.67) {
      console.warn(`${name} took ${elapsed.toFixed(1)}ms — exceeds frame budget`);
    }
    return result;
  }
}

const monitor = new MapPerformanceMonitor(map);
```

---

## Advanced Dark Arts

These are non-obvious techniques used in production high-performance geo apps.

**Viewport-aware data loading** — never load features outside the viewport:

```js
function getExpandedBounds(map, bufferRatio = 0.2) {
  const bounds = map.getBounds();
  const latSpan = bounds.getNorth() - bounds.getSouth();
  const lngSpan = bounds.getEast() - bounds.getWest();
  return {
    minLng: bounds.getWest() - lngSpan * bufferRatio,
    maxLng: bounds.getEast() + lngSpan * bufferRatio,
    minLat: bounds.getSouth() - latSpan * bufferRatio,
    maxLat: bounds.getNorth() + latSpan * bufferRatio
  };
}

// Load data only within viewport + 20% buffer
const bbox = getExpandedBounds(map);
const features = await queryFeaturesInBbox(bbox);
```

**Request deduplication with AbortController** — cancel stale requests when viewport changes:

```js
let abortController = null;

map.on('moveend', async () => {
  // Cancel previous in-flight request
  if (abortController) {
    abortController.abort();
  }
  abortController = new AbortController();

  try {
    const data = await fetch('/api/features', {
      signal: abortController.signal,
      body: JSON.stringify(getExpandedBounds(map))
    });
    map.getSource('features').setData(await data.json());
  } catch (e) {
    if (e.name !== 'AbortError') throw e;  // ignore expected cancellation
  }
});
```

**Tile prefetching from mouse velocity**:

```js
let lastMousePos = null;
let lastMouseTime = null;

map.on('mousemove', (e) => {
  const now = performance.now();
  if (lastMousePos && lastMouseTime) {
    const dt = now - lastMouseTime;
    const dx = e.point.x - lastMousePos.x;
    const dy = e.point.y - lastMousePos.y;
    const velocity = { x: dx / dt, y: dy / dt };

    // Predict where the user is panning — prefetch those tiles
    if (Math.abs(velocity.x) > 0.5 || Math.abs(velocity.y) > 0.5) {
      prefetchTilesInDirection(map, velocity);
    }
  }
  lastMousePos = e.point;
  lastMouseTime = now;
});
```

**GPU texture atlas for icon-heavy maps** — pack all marker icons into one atlas texture to eliminate texture bind overhead:

```js
// Generate sprite atlas programmatically
const canvas = document.createElement('canvas');
canvas.width = 512; canvas.height = 512;
const ctx = canvas.getContext('2d');

const iconLayout = {};
let x = 0, y = 0, rowHeight = 0;
const PADDING = 2;

for (const [name, img] of Object.entries(iconImages)) {
  if (x + img.width > 512) { x = 0; y += rowHeight + PADDING; rowHeight = 0; }
  ctx.drawImage(img, x, y);
  iconLayout[name] = { x, y, width: img.width, height: img.height, pixelRatio: 1 };
  x += img.width + PADDING;
  rowHeight = Math.max(rowHeight, img.height);
}

// Add sprite to MapLibre — all icons from one texture = one draw call
map.addSprite('custom-icons', canvas.toDataURL(), iconLayout);
```

**WebGL context recovery**:

```js
// MapLibre handles this automatically, but here's the manual approach
// for custom WebGL code
canvas.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();  // MUST preventDefault to allow recovery
  console.warn('WebGL context lost — pausing renders');
  cancelAnimationFrame(rafHandle);
}, false);

canvas.addEventListener('webglcontextrestored', () => {
  console.log('WebGL context restored — reinitializing');
  // Re-upload all GPU buffers, re-compile shaders
  reinitializeWebGL();
  renderLoop();
}, false);
```

**OffscreenCanvas for Leaflet Canvas renderer in a Worker** (experimental):

```js
// Transfer canvas to worker (Chrome 69+)
const canvas = document.getElementById('map-canvas');
const offscreen = canvas.transferControlToOffscreen();

worker.postMessage({ canvas: offscreen }, [offscreen]);

// worker.js
self.onmessage = (e) => {
  const { canvas } = e.data;
  // Leaflet Canvas renderer can draw to OffscreenCanvas
  // This moves all canvas 2D drawing off the main thread
  initLeafletWithOffscreenCanvas(canvas);
};
```

**Incremental rendering with requestIdleCallback**:

```js
function renderLayersIncrementally(layers) {
  const visible = layers.filter(l => isInViewport(l));
  const background = layers.filter(l => !isInViewport(l));

  // Render visible layers immediately
  visible.forEach(l => map.setLayoutProperty(l.id, 'visibility', 'visible'));

  // Render background layers during idle time
  requestIdleCallback((deadline) => {
    let i = 0;
    while (deadline.timeRemaining() > 0 && i < background.length) {
      map.setLayoutProperty(background[i].id, 'visibility', 'visible');
      i++;
    }
    if (i < background.length) {
      // Didn't finish — schedule another idle callback for the rest
      requestIdleCallback(() => renderLayersIncrementally(background.slice(i)));
    }
  });
}
```

**HTTP/2 for tile servers** — HTTP/1.1 allows only 6 parallel connections per domain. HTTP/2 multiplexes hundreds of streams over one connection. Domain sharding (old HTTP/1.1 trick) is counterproductive with HTTP/2. Verify your tile server supports HTTP/2:

```bash
curl -I --http2 https://tiles.example.com/tiles/0/0/0.pbf | grep -i "http/"
# HTTP/2 200  <-- good
# HTTP/1.1 200 <-- configure nginx/caddy for HTTP/2
```

**Brotli tile compression** — MVT tiles compress 15-25% better with Brotli vs gzip. Configure in nginx:

```nginx
# nginx.conf
brotli on;
brotli_types application/vnd.mapbox-vector-tile application/x-protobuf;
brotli_comp_level 6;
```

**ETag caching for instant reloads** — tile servers must send proper ETag headers so browsers serve cached tiles from memory on revisit:

```python
# FastAPI tile server with proper caching
from fastapi import Response
import hashlib

@app.get("/tiles/{z}/{x}/{y}.pbf")
async def get_tile(z: int, x: int, y: int, response: Response):
    tile_data = generate_tile(z, x, y)
    etag = hashlib.md5(tile_data).hexdigest()

    response.headers["ETag"] = f'"{etag}"'
    response.headers["Cache-Control"] = "public, max-age=3600"
    response.headers["Content-Type"] = "application/vnd.mapbox-vector-tile"
    response.headers["Content-Encoding"] = "gzip"

    return Response(content=tile_data)
```

---

## Decision Matrix: When to Use What

| Scenario | Recommended Approach |
|----------|---------------------|
| <10K static points | Leaflet + Canvas renderer or MapLibre GeoJSON |
| 10K-100K points, interactive | MapLibre GeoJSON source with clustering |
| 100K-1M points, display only | PMTiles via tippecanoe → MapLibre |
| 1M+ points | deck.gl ScatterplotLayer + binary data OR H3 aggregation |
| Large polygons (countries, land use) | PMTiles → MapLibre (pre-simplified per zoom) |
| Real-time streaming features | MapLibre GeoJSON source + setData() at <2fps |
| Analytical queries on large data | DuckDB WASM + GeoParquet (aggregate before rendering) |
| Raster visualization | COG + geotiff.js OR TiTiler dynamic tiles |
| 3D buildings | CesiumJS 3D Tiles OR deck.gl PolygonLayer extruded |
| Heavy spatial ops (buffer, union) | Turf.js in Web Worker via comlink |
| Cross-filter dashboards | DuckDB WASM (sub-second queries on 100M rows) |

---

## Quick Benchmark Checklist

Run these checks before shipping a geo app to production:

- [ ] Open Chrome DevTools Performance tab — no main thread tasks >50ms during pan/zoom
- [ ] Memory tab — no growth in retained heap after loading/unloading data multiple times
- [ ] Network tab — tile requests <100ms TTFB, no redundant duplicate requests
- [ ] Lighthouse → Performance score — TBT (Total Blocking Time) <300ms
- [ ] Test on a mid-range Android phone (Pixel 6a class) — target 30fps minimum
- [ ] `chrome://gpu` confirms WebGL2 is available (required for deck.gl)
- [ ] Check bundle size: `npx source-map-explorer dist/assets/*.js`
- [ ] Verify HTTP/2 on tile server: `curl -I --http2 https://your-tile-server/...`
- [ ] Check tile cache headers: `Cache-Control: public`, ETag present
- [ ] Web Worker offloading: no GeoJSON operations >16ms on main thread

---

## References

- [MapLibre GL JS Performance](https://maplibre.org/maplibre-gl-js/docs/examples/optimize-style-layers/) — official optimization examples
- [deck.gl Performance Tips](https://deck.gl/docs/developer-guide/performance) — binary data, updateTriggers, layer management
- [tippecanoe documentation](https://github.com/felt/tippecanoe) — all tile simplification flags
- [DuckDB WASM spatial](https://duckdb.org/docs/extensions/spatial.html) — spatial SQL extension
- [FlatGeobuf spec](https://flatgeobuf.org/) — format details and HTTP range request API
- [PMTiles spec](https://github.com/protomaps/PMTiles) — single-file tile archive format
- [GeoArrow spec](https://geoarrow.org/) — binary geometry columnar format
- [Web Workers API](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API) — MDN reference
- [comlink](https://github.com/GoogleChromeLabs/comlink) — Worker RPC library from Google Chrome Labs
- [CesiumJS Performance Guide](https://cesium.com/learn/cesiumjs/ref-doc/module-Cesium.html) — requestRenderMode and Primitive API
