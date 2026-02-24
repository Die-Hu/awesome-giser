# Browser-Side Data Formats & Loading

**Quick Picks**
- **SOTA**: DuckDB WASM + GeoParquet — SQL analytics on columnar geo data, directly in the browser
- **Free Best**: FlatGeobuf streaming — spatial index range requests, no backend, no pre-tiling
- **Fastest Setup**: GeoJSON fetch — one line, works everywhere, great for prototypes

---

## Format Comparison Matrix

| Format | Size (10K polygons) | Load time | Spatial filter | Streaming | Browser support | Best for |
|--------|---------------------|-----------|----------------|-----------|-----------------|----------|
| GeoJSON | ~5 MB | Parse-all | No (client-side) | No | Native JSON | Small data, prototypes |
| TopoJSON | ~1 MB | Parse-all | No | No | topojson-client | Shared boundaries, smaller transfer |
| FlatGeobuf | ~2 MB | Streaming | Yes (server-side range) | Yes | flatgeobuf.js | Medium data, spatial filter |
| GeoParquet | ~1 MB | Columnar | Yes (via DuckDB) | Partial | duckdb-wasm | Analytics, large data |
| PMTiles (MVT) | ~3 MB (all zooms) | Tile-by-tile | Yes (tile-based) | Yes | pmtiles.js | Map rendering |
| COG (GeoTIFF) | varies | Range-request | Yes (window) | Yes | geotiff.js | Raster data |
| CSV/TSV | ~2 MB | Parse-all | No | With Papa Parse streaming | Built-in | Point data |
| GPX/KML | ~3 MB | Parse-all | No | No | togeojson/omnivore | GPS tracks, simple features |
| Shapefile | ~4 MB + .dbf + .shx | Parse-all | No | No | shapefile.js | Legacy data |
| Protocol Buffers (MVT) | ~0.5 MB/tile | Tile decode | Yes (tile extent) | Yes | Built into MapLibre | Vector tile rendering |

---

## GeoJSON -- The Universal Format

GeoJSON is the lingua franca of browser geospatial. Every library reads it, every API speaks it, and the browser parses it natively. The limitation is scale: JSON is verbose, and parsing 50 MB of text on the main thread blocks the UI.

### Loading Patterns

```
Small (<1 MB):   fetch() → map.addSource()
Medium (1-10 MB): fetch() → Web Worker parse → map.addSource()
Large (10-50 MB): stream with oboe.js, progressive rendering
Huge (50+ MB):   DON'T USE GEOJSON. Convert to PMTiles or GeoParquet
```

### Simple Fetch (small data)

```javascript
const response = await fetch('https://example.com/data.geojson');
const geojson = await response.json();

map.addSource('my-data', { type: 'geojson', data: geojson });
map.addLayer({
  id: 'my-layer',
  type: 'fill',
  source: 'my-data',
  paint: { 'fill-color': '#088', 'fill-opacity': 0.6 }
});
```

### Web Worker Parse (medium data)

Moving JSON.parse() off the main thread keeps the map responsive during heavy loads.

**main.js**
```javascript
const worker = new Worker('./geojson-worker.js');

worker.onmessage = (e) => {
  if (e.data.type === 'geojson-ready') {
    map.getSource('big-data').setData(e.data.geojson);
  }
};

const response = await fetch('big-dataset.geojson');
const buffer = await response.arrayBuffer();
// Transfer ownership — zero-copy
worker.postMessage({ type: 'parse', buffer }, [buffer]);
```

**geojson-worker.js**
```javascript
self.onmessage = (e) => {
  if (e.data.type === 'parse') {
    const text = new TextDecoder().decode(e.data.buffer);
    const geojson = JSON.parse(text);
    self.postMessage({ type: 'geojson-ready', geojson });
  }
};
```

### Streaming GeoJSON with oboe.js

`oboe.js` is a SAX-style streaming JSON parser. Features are emitted as they arrive over the network — the map populates before the download completes.

```javascript
import oboe from 'oboe';

const features = [];

oboe('https://example.com/large.geojson')
  .node('features.*', (feature) => {
    features.push(feature);

    // Update map every 500 features
    if (features.length % 500 === 0) {
      map.getSource('stream-source').setData({
        type: 'FeatureCollection',
        features: [...features]
      });
    }

    // Return oboe.drop to free memory after processing
    return oboe.drop;
  })
  .done(() => {
    // Final update with all features
    map.getSource('stream-source').setData({
      type: 'FeatureCollection',
      features
    });
  })
  .fail((err) => console.error('Stream error:', err));
```

### NDJSON (Newline-Delimited JSON) Streaming

One feature per line — trivially streamable with the browser Streams API. No library needed.

```javascript
async function streamNDJSON(url, onFeature) {
  const response = await fetch(url);
  const reader = response.body
    .pipeThrough(new TextDecoderStream())
    .getReader();

  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += value;
    const lines = buffer.split('\n');
    buffer = lines.pop(); // Keep incomplete last line

    for (const line of lines) {
      if (line.trim()) {
        onFeature(JSON.parse(line));
      }
    }
  }
}

const features = [];
await streamNDJSON('data.ndjson', (feature) => {
  features.push(feature);
  if (features.length % 200 === 0) {
    map.getSource('ndjson-source').setData({
      type: 'FeatureCollection', features: [...features]
    });
  }
});
```

### Performance Tips

- Always use `map.getSource(id).setData()` instead of removing/re-adding the source — re-adding causes full style recompilation.
- Use `generateId: true` on the GeoJSON source when you need `setFeatureState()` for hover/selection states.
- Compress with gzip/brotli on the server — GeoJSON compresses 5-10x, bringing a 5 MB file down to ~500 KB transfer.
- Prefer TopoJSON for boundary data — shared arcs eliminate ~80% of coordinate duplication.
- For point clouds, consider clustering at the source level: `cluster: true, clusterMaxZoom: 14, clusterRadius: 50`.

### TopoJSON

TopoJSON encodes topology: shared boundaries are stored once as arcs, referenced by multiple polygons. A world countries file shrinks from ~700 KB GeoJSON to ~110 KB TopoJSON.

```javascript
import { feature, merge } from 'topojson-client';

const response = await fetch('countries.topojson');
const topology = await response.json();

// Extract a single layer as GeoJSON
const countries = feature(topology, topology.objects.countries);

// Dissolve polygons (merge shared arcs) — e.g., continent outlines
const continents = merge(topology, topology.objects.countries.geometries);

map.addSource('countries', { type: 'geojson', data: countries });
```

**Creating TopoJSON from GeoJSON (Node.js)**:
```javascript
import { topology } from 'topojson-server';
import { simplify, presimplify } from 'topojson-simplify';
import fs from 'fs';

const geojson = JSON.parse(fs.readFileSync('input.geojson'));
let topo = topology({ layer: geojson });
topo = presimplify(topo);
topo = simplify(topo, 0.0001); // Visvalingam weight threshold

fs.writeFileSync('output.topojson', JSON.stringify(topo));
```

---

## FlatGeobuf -- Spatial Streaming

FlatGeobuf is a binary format using FlatBuffers with a packed Hilbert R-tree spatial index embedded in the file header. This enables the browser to issue HTTP range requests for only the features within a given bounding box — the server does no work, it just serves byte ranges.

### How It Works

1. Client reads the file header (a few KB) to get the spatial index
2. Client walks the R-tree to find which byte ranges contain features intersecting the query bbox
3. Client issues 1-3 HTTP range requests for those byte ranges
4. FlatBuffers zero-copy decode — no full parse, just memory mapping

### Implementation with MapLibre GL JS

```javascript
import { deserialize } from 'flatgeobuf/lib/mjs/geojson.js';

async function loadFGB(map, url) {
  const bounds = map.getBounds();
  const bbox = {
    minX: bounds.getWest(),
    minY: bounds.getSouth(),
    maxX: bounds.getEast(),
    maxY: bounds.getNorth()
  };

  const features = [];
  for await (const feature of deserialize(url, bbox)) {
    features.push(feature);
  }

  const source = map.getSource('fgb-source');
  if (source) {
    source.setData({ type: 'FeatureCollection', features });
  } else {
    map.addSource('fgb-source', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features }
    });
    map.addLayer({
      id: 'fgb-layer',
      type: 'fill',
      source: 'fgb-source',
      paint: { 'fill-color': '#e55', 'fill-opacity': 0.5 }
    });
  }
}

// Initial load and reload on map move
map.on('load', () => loadFGB(map, 'https://cdn.example.com/data.fgb'));
map.on('moveend', () => loadFGB(map, 'https://cdn.example.com/data.fgb'));
```

### Implementation with Leaflet

```javascript
import L from 'leaflet';
import { deserialize } from 'flatgeobuf/lib/mjs/geojson.js';

let fgbLayer = null;

async function loadFGB(map) {
  const bounds = map.getBounds();
  const bbox = {
    minX: bounds.getWest(),
    minY: bounds.getSouth(),
    maxX: bounds.getEast(),
    maxY: bounds.getNorth()
  };

  const features = [];
  for await (const feature of deserialize('data.fgb', bbox)) {
    features.push(feature);
  }

  // Remove old layer
  if (fgbLayer) map.removeLayer(fgbLayer);

  fgbLayer = L.geoJSON({ type: 'FeatureCollection', features }, {
    style: (feature) => ({
      color: '#e55',
      weight: 1,
      fillOpacity: 0.5
    }),
    onEachFeature: (feature, layer) => {
      layer.bindPopup(JSON.stringify(feature.properties, null, 2));
    }
  }).addTo(map);
}

map.on('load moveend', () => loadFGB(map));
```

### Debouncing Moves

Avoid hammering the server on fast panning by debouncing:

```javascript
function debounce(fn, ms) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

map.on('moveend', debounce(() => loadFGB(map, url), 300));
```

### Performance Benchmarks

- First feature: ~50 ms (index lookup + first range request)
- 10K features in viewport: ~200 ms
- Same data as GeoJSON fetch: ~2000 ms (full download + parse)
- Memory: only viewport features in memory at any time

### Creating FlatGeobuf

```bash
# From GeoJSON (GDAL required, EPSG:4326 for correct lat/lon spatial index)
ogr2ogr -f FlatGeobuf output.fgb input.geojson

# From PostGIS
ogr2ogr -f FlatGeobuf output.fgb \
  "PG:host=localhost dbname=mydb user=postgres" \
  -sql "SELECT geom, name, population FROM cities"

# Reproject to WGS84 if needed
ogr2ogr -f FlatGeobuf -t_srs EPSG:4326 output.fgb input_in_utm.shp
```

**Important**: The spatial index works in the coordinate space of the file. For lat/lon bounding box queries from the browser, the file must be in EPSG:4326.

### FlatGeobuf vs PMTiles Decision Guide

| Aspect | FlatGeobuf | PMTiles |
|--------|------------|---------|
| Pre-processing | ogr2ogr (seconds) | tippecanoe (minutes to hours) |
| Spatial query | Exact R-tree | Approximate tile grid |
| Rendering approach | Client-side GeoJSON renderer | GPU vector tile renderer |
| Feature count comfort zone | ~100K | Millions |
| Styling capability | Limited by GeoJSON renderer | Full MapLibre style spec |
| Simplification at zoom | Manual or none | Automatic per zoom level |
| Update workflow | Replace file (instant) | Rebuild tiles |
| Attribute access | Full properties per feature | Style-spec expressions |
| Server requirements | Static file host (S3, CDN) | Static file host (S3, CDN) |

---

## GeoParquet + DuckDB WASM -- SQL Analytics in the Browser

DuckDB compiled to WebAssembly brings the full DuckDB analytical engine to the browser. Combined with the spatial extension and httpfs, it enables SQL queries over remote GeoParquet files with columnar pushdown — reading only the columns and row groups actually needed.

### Architecture

```
Browser
  └── Web Worker
        └── DuckDB WASM (~4 MB)
              ├── spatial extension (ST_* functions)
              ├── httpfs extension (read remote Parquet/S3)
              └── Persistent in-memory database
```

### Installation

```bash
npm install @duckdb/duckdb-wasm apache-arrow
```

### Initialization

```javascript
import * as duckdb from '@duckdb/duckdb-wasm';

async function initDuckDB() {
  const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
  const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);

  const worker_url = URL.createObjectURL(
    new Blob([`importScripts("${bundle.mainWorker}");`], { type: 'text/javascript' })
  );

  const worker = new Worker(worker_url);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

  URL.revokeObjectURL(worker_url);

  const conn = await db.connect();
  await conn.query('INSTALL spatial; LOAD spatial;');
  await conn.query('INSTALL httpfs; LOAD httpfs;');

  return conn;
}

const conn = await initDuckDB();
```

### Querying Remote GeoParquet (Overture Maps)

```javascript
// Query Overture Maps buildings from S3 with bbox filter
const result = await conn.query(`
  SELECT
    names.primary AS name,
    ST_AsGeoJSON(geometry) AS geojson,
    height,
    num_floors
  FROM read_parquet(
    's3://overturemaps-us-west-2/release/2024-11-13.0/theme=buildings/type=building/*',
    hive_partitioning=1
  )
  WHERE bbox.xmin BETWEEN 116.30 AND 116.50
    AND bbox.ymin BETWEEN 39.80 AND 40.00
    AND height IS NOT NULL
  LIMIT 5000
`);

const features = result.toArray().map(row => ({
  type: 'Feature',
  geometry: JSON.parse(row.geojson),
  properties: {
    name: row.name,
    height: row.height,
    num_floors: row.num_floors
  }
}));

map.getSource('buildings').setData({ type: 'FeatureCollection', features });
```

### Spatial Aggregation in the Browser

DuckDB WASM can aggregate millions of rows in the browser. This pattern replaces server-side tile aggregation for exploratory dashboards.

```javascript
// H3 hexagonal aggregation (requires h3 extension)
await conn.query('INSTALL h3 FROM community; LOAD h3;');

const result = await conn.query(`
  SELECT
    h3_latlng_to_cell(ST_Y(geometry), ST_X(geometry), 7) AS hex_id,
    COUNT(*) AS count,
    AVG(value) AS avg_value
  FROM read_parquet('data.parquet')
  WHERE ST_Within(
    geometry,
    ST_GeomFromText('POLYGON((116.0 39.5, 117.0 39.5, 117.0 40.5, 116.0 40.5, 116.0 39.5))')
  )
  GROUP BY hex_id
  ORDER BY count DESC
`);
```

### Loading Local Files

DuckDB WASM can load files directly from the browser via `registerFileBuffer`:

```javascript
// From file input
document.getElementById('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  const buffer = await file.arrayBuffer();

  await db.registerFileBuffer(file.name, new Uint8Array(buffer));

  const result = await conn.query(`
    SELECT ST_AsGeoJSON(geometry) AS geojson, *
    FROM read_parquet('${file.name}')
    LIMIT 10000
  `);

  // Process result...
});
```

### Performance Reference

| Operation | Time |
|-----------|------|
| Initial WASM load + extensions | ~2–4 s (cached on revisit) |
| Query 1M row Parquet with spatial filter | ~500 ms |
| COUNT/AVG aggregation on 10M rows (local) | ~200 ms |
| Read remote Parquet with column pushdown | ~300 ms + network |
| H3 aggregation, 500K points | ~800 ms |

### When to Use DuckDB WASM

- Exploratory dashboards where users filter/aggregate/pivot data on the fly
- Offline-capable analytics (cache Parquet files in OPFS or IndexedDB)
- Reducing server costs by pushing queries to the client
- Rapid prototyping without a data backend
- Cross-tab DuckDB sharing via Shared Worker (see Advanced section)

### When NOT to Use DuckDB WASM

- Initial page load must be fast — 2-4s WASM initialization is unavoidable on first visit
- Data exceeds what fits in browser memory (~500 MB practical limit)
- Users are on low-end devices (WASM execution is CPU-intensive)

---

## Apache Arrow / GeoArrow -- Zero-Copy Data

### What Arrow Enables

Apache Arrow defines a binary columnar memory format. The key insight: an Arrow Table in memory looks the same in JavaScript, C++ WASM modules, and GPU buffers. This means:

- Pass data between WASM modules (DuckDB → deck.gl) without copying
- Share between Web Workers via `SharedArrayBuffer`
- Send to GPU without intermediate serialization

```bash
npm install apache-arrow
```

### Reading Arrow IPC Files

```javascript
import { tableFromIPC, RecordBatchReader } from 'apache-arrow';

// Synchronous read (full file)
const response = await fetch('data.arrow');
const table = tableFromIPC(await response.arrayBuffer());

console.log(table.schema.fields.map(f => f.name));
console.log(table.numRows, 'rows');

// Streaming IPC (file with multiple record batches)
const response2 = await fetch('data.arrows'); // .arrows = streaming format
const reader = await RecordBatchReader.from(response2.body);
for await (const batch of reader) {
  console.log('Batch:', batch.numRows, 'rows');
}
```

### GeoArrow with deck.gl

GeoArrow stores geometry as an Arrow extension type (WKB, WKT, or native struct). deck.gl reads GeoArrow natively — no GeoJSON serialization round-trip.

```javascript
import { GeoArrowScatterplotLayer } from '@deck.gl/layers';
import { GeoArrowArcLayer } from '@deck.gl/layers';
import { tableFromIPC } from 'apache-arrow';

const response = await fetch('points.arrow');
const table = tableFromIPC(await response.arrayBuffer());

// deck.gl reads geometry column directly
const layer = new GeoArrowScatterplotLayer({
  id: 'points',
  data: table,
  getPosition: table.getChild('geometry'), // GeoArrow geometry column
  getFillColor: [255, 100, 0, 200],
  getRadius: 50,
  radiusUnits: 'meters',
  pickable: true
});
```

**Memory comparison for 1M points:**

| Approach | Memory usage |
|----------|-------------|
| GeoJSON array | ~800 MB |
| Arrow Table (GeoArrow) | ~80 MB |
| Shared via SharedArrayBuffer | 80 MB (shared, not duplicated) |

### DuckDB WASM → Arrow → deck.gl Pipeline

```javascript
// Query DuckDB, get Arrow result, pass directly to deck.gl
const arrowResult = await conn.query(`
  SELECT ST_X(geometry) AS lon, ST_Y(geometry) AS lat, value
  FROM parquet_scan('data.parquet')
  WHERE bbox_filter
`);

// arrowResult is already an Apache Arrow Table
const layer = new GeoArrowScatterplotLayer({
  data: arrowResult,
  getPosition: arrowResult.getChild('geometry'),
  ...
});
```

---

## Cloud-Optimized GeoTIFF (COG) in the Browser

A Cloud-Optimized GeoTIFF reorganizes the internal structure of a GeoTIFF so that the spatial index and tile blocks are accessible via HTTP range requests. The browser fetches only the tiles needed for the current viewport and zoom level.

### geotiff.js Deep Dive

```bash
npm install geotiff
```

**Reading a Remote COG:**

```javascript
import { fromUrl, fromArrayBuffer } from 'geotiff';

const tiff = await fromUrl('https://example.com/elevation.tif');
const image = await tiff.getImage();

// Inspect metadata
console.log('Width:', image.getWidth());
console.log('Height:', image.getHeight());
console.log('Bands:', image.getSamplesPerPixel());
console.log('Bbox:', image.getBoundingBox());
console.log('CRS EPSG:', image.geoKeys.ProjectedCSTypeGeoKey);

// Read a spatial window (in pixel coordinates)
const [x0, y0, x1, y1] = [0, 0, 256, 256];
const data = await image.readRasters({
  window: [x0, y0, x1, y1],
  width: 256,
  height: 256,
  resampleMethod: 'bilinear',
  fillValue: -9999
});

// data is an array of TypedArrays, one per band
// data[0] = Float32Array of elevation values
const elevations = data[0];
```

**Reading by Geographic Bbox:**

```javascript
import { fromUrl } from 'geotiff';
import { toArrayBuffer } from 'geotiff/src/utils';

async function readCOGByBbox(url, [west, south, east, north]) {
  const tiff = await fromUrl(url);
  const image = await tiff.getImage();

  const [imgLeft, imgBottom, imgRight, imgTop] = image.getBoundingBox();
  const imgWidth = image.getWidth();
  const imgHeight = image.getHeight();

  // Convert geographic bbox to pixel window
  const x0 = Math.floor((west - imgLeft) / (imgRight - imgLeft) * imgWidth);
  const y0 = Math.floor((imgTop - north) / (imgTop - imgBottom) * imgHeight);
  const x1 = Math.ceil((east - imgLeft) / (imgRight - imgLeft) * imgWidth);
  const y1 = Math.ceil((imgTop - south) / (imgTop - imgBottom) * imgHeight);

  return image.readRasters({
    window: [x0, y0, x1, y1],
    width: 256,
    height: 256
  });
}
```

### Rendering COG as a Custom MapLibre Protocol

```javascript
import maplibregl from 'maplibre-gl';
import { fromUrl, Pool } from 'geotiff';

const pool = new Pool(); // Worker pool for parallel tile decoding

maplibregl.addProtocol('cog', async (params, abortController) => {
  const url = params.url.replace('cog://', '');
  const tiff = await fromUrl(url);
  const image = await tiff.getImage();

  // Parse tile coordinates from params
  const { z, x, y } = parseTileCoords(params.url);
  const bbox = tileToBbox(z, x, y); // returns [west, south, east, north]

  const [west, south, east, north] = bbox;
  const [imgLeft, imgBottom, imgRight, imgTop] = image.getBoundingBox();

  const scaleX = image.getWidth() / (imgRight - imgLeft);
  const scaleY = image.getHeight() / (imgTop - imgBottom);

  const window = [
    Math.max(0, Math.floor((west - imgLeft) * scaleX)),
    Math.max(0, Math.floor((imgTop - north) * scaleY)),
    Math.min(image.getWidth(), Math.ceil((east - imgLeft) * scaleX)),
    Math.min(image.getHeight(), Math.ceil((imgTop - south) * scaleY))
  ];

  const rasters = await image.readRasters({ window, width: 256, height: 256, pool });

  // Convert to RGBA canvas
  const canvas = new OffscreenCanvas(256, 256);
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(256, 256);

  // Normalize and colorize elevation data
  const [min, max] = [-100, 8000];
  for (let i = 0; i < rasters[0].length; i++) {
    const val = rasters[0][i];
    const norm = Math.max(0, Math.min(1, (val - min) / (max - min)));
    const [r, g, b] = elevationColormap(norm);
    imageData.data[i * 4 + 0] = r;
    imageData.data[i * 4 + 1] = g;
    imageData.data[i * 4 + 2] = b;
    imageData.data[i * 4 + 3] = val === -9999 ? 0 : 255;
  }

  ctx.putImageData(imageData, 0, 0);
  const blob = await canvas.convertToBlob();
  return { data: await blob.arrayBuffer() };
});
```

### Band Math in Browser (NDVI Example)

```javascript
import { fromUrl } from 'geotiff';

async function computeNDVI(nirUrl, redUrl, bbox) {
  const [nirTiff, redTiff] = await Promise.all([
    fromUrl(nirUrl),
    fromUrl(redUrl)
  ]);

  const [nirImg, redImg] = await Promise.all([
    nirTiff.getImage(),
    redTiff.getImage()
  ]);

  const window = bboxToPixelWindow(nirImg, bbox);

  const [nirData, redData] = await Promise.all([
    nirImg.readRasters({ window, width: 512, height: 512 }),
    redImg.readRasters({ window, width: 512, height: 512 })
  ]);

  const nir = nirData[0];
  const red = redData[0];
  const ndvi = new Float32Array(nir.length);

  for (let i = 0; i < ndvi.length; i++) {
    const n = nir[i], r = red[i];
    // Guard against nodata and division by zero
    ndvi[i] = (n === -9999 || r === -9999) ? NaN : (n - r) / (n + r + 1e-6);
  }

  return ndvi; // Float32Array, range [-1, 1]
}
```

---

## STAC (SpatioTemporal Asset Catalog) Browser Client

STAC is the standard catalog format for geospatial imagery. `stac-js` lets you search STAC catalogs from the browser and retrieve signed URLs for COG assets.

```bash
npm install stac-js
```

```javascript
import { StacApi } from 'stac-js';

const api = new StacApi('https://earth-search.aws.element84.com/v1');

// Search for low-cloud Sentinel-2 over Beijing in summer 2024
const results = await api.search({
  collections: ['sentinel-2-l2a'],
  bbox: [116.0, 39.5, 117.0, 40.5],
  datetime: '2024-06-01/2024-08-31',
  query: { 'eo:cloud_cover': { lt: 20 } },
  sortby: [{ field: 'eo:cloud_cover', direction: 'asc' }],
  limit: 10
});

const items = await results.items();

for (const item of items) {
  console.log('Scene:', item.id);
  console.log('Date:', item.properties.datetime);
  console.log('Cloud cover:', item.properties['eo:cloud_cover'], '%');

  // Get asset URLs (signed for S3 if auth is configured)
  const nir = item.assets['nir']?.href;
  const red = item.assets['red']?.href;
  const tci = item.assets['visual']?.href; // True color image (COG)

  if (tci) {
    // Load TCI as COG
    const tiff = await fromUrl(tci);
    // ... render on map
  }
}
```

---

## Shapefile in the Browser

### shapefile.js (mbostock)

Parses `.shp` + `.dbf` in the browser. Note: does not handle `.prj` — assumes EPSG:4326 or you must project manually.

```bash
npm install shapefile
```

```javascript
import * as shapefile from 'shapefile';

// From URL
const source = await shapefile.open(
  'https://example.com/data.shp',
  'https://example.com/data.dbf',
  { encoding: 'UTF-8' }
);

const features = [];
while (true) {
  const result = await source.read();
  if (result.done) break;
  features.push(result.value);
}

map.getSource('shp').setData({ type: 'FeatureCollection', features });
```

**From file input (user upload):**

```javascript
async function loadShapefileUpload(shpFile, dbfFile) {
  const [shpBuf, dbfBuf] = await Promise.all([
    shpFile.arrayBuffer(),
    dbfFile.arrayBuffer()
  ]);

  const source = await shapefile.open(shpBuf, dbfBuf);
  const features = [];

  while (true) {
    const result = await source.read();
    if (result.done) break;
    features.push(result.value);
  }

  return { type: 'FeatureCollection', features };
}
```

### shp-write (Export to Shapefile)

```bash
npm install shp-write
```

```javascript
import shpwrite from 'shp-write';

// Download GeoJSON as Shapefile ZIP
function downloadAsShapefile(geojson, filename = 'export') {
  const options = {
    folder: filename,
    filename: filename,
    outputType: 'blob',
    compression: 'DEFLATE',
    types: {
      point: 'points',
      polygon: 'polygons',
      polyline: 'lines'
    }
  };

  shpwrite.download(geojson, options);
}
```

---

## CSV / TSV Point Data

### Papa Parse (Gold Standard)

```bash
npm install papaparse
```

```javascript
import Papa from 'papaparse';

// Streaming parse (handles huge files without OOM)
Papa.parse('https://example.com/data.csv', {
  download: true,
  header: true,
  dynamicTyping: true, // auto-convert numbers
  skipEmptyLines: true,
  step: (row) => {
    // Called for each row — no memory accumulation
    const { longitude, latitude, value, name } = row.data;
    if (isFinite(longitude) && isFinite(latitude)) {
      features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [longitude, latitude] },
        properties: { value, name }
      });
    }
  },
  complete: () => {
    map.getSource('csv-points').setData({ type: 'FeatureCollection', features });
  },
  error: (err) => console.error('Parse error:', err)
});
```

**From file input:**

```javascript
document.getElementById('csv-upload').addEventListener('change', (e) => {
  Papa.parse(e.target.files[0], {
    header: true,
    dynamicTyping: true,
    complete: (results) => {
      const features = results.data
        .filter(row => isFinite(row.lon) && isFinite(row.lat))
        .map(row => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [row.lon, row.lat] },
          properties: row
        }));
      // Use features...
    }
  });
});
```

---

## GPX / KML

### togeojson

```bash
npm install @mapbox/togeojson
```

```javascript
import { gpx, kml } from '@mapbox/togeojson';

// GPX
const gpxResponse = await fetch('track.gpx');
const gpxText = await gpxResponse.text();
const gpxDom = new DOMParser().parseFromString(gpxText, 'text/xml');
const gpxGeoJSON = gpx(gpxDom);

// KML
const kmlResponse = await fetch('features.kml');
const kmlText = await kmlResponse.text();
const kmlDom = new DOMParser().parseFromString(kmlText, 'text/xml');
const kmlGeoJSON = kml(kmlDom);

map.addSource('track', { type: 'geojson', data: gpxGeoJSON });
map.addLayer({
  id: 'track-line',
  type: 'line',
  source: 'track',
  paint: { 'line-color': '#f00', 'line-width': 3 }
});
```

---

## Protocol Buffers / Vector Tiles (MVT)

### Manual MVT Decode

MapLibre handles MVT internally. Manual decode is useful for feature extraction and custom analysis.

```bash
npm install @mapbox/vector-tile pbf
```

```javascript
import { VectorTile } from '@mapbox/vector-tile';
import Pbf from 'pbf';

async function decodeMVTTile(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const tile = new VectorTile(new Pbf(buffer));

  const features = [];
  for (const layerName of Object.keys(tile.layers)) {
    const layer = tile.layers[layerName];
    for (let i = 0; i < layer.length; i++) {
      const feature = layer.feature(i);
      features.push({
        type: 'Feature',
        geometry: feature.toGeoJSON(tileX, tileY, tileZ),
        properties: feature.properties,
        layer: layerName
      });
    }
  }

  return features;
}
```

---

## WASM Geospatial Tools

### DuckDB WASM

Covered in depth above. Package: `@duckdb/duckdb-wasm`. The flagship WASM GIS tool for analytical workflows.

### GDAL WASM (gdal3.js / loam)

Full GDAL compiled to WebAssembly (~10 MB). Enables format conversion, reprojection, and raster math — server-side GDAL workflows moved entirely to the browser.

```bash
npm install gdal3.js
```

```javascript
import Gdal from 'gdal3.js';

const gdal = await Gdal();

// Convert Shapefile → GeoJSON in browser
async function convertToGeoJSON(shpFile) {
  const shpPath = `/vsimem/input.shp`;
  const outPath = `/vsimem/output.geojson`;

  await gdal.open(shpPath, await shpFile.arrayBuffer());

  gdal.ogr2ogr([
    '-f', 'GeoJSON',
    '-t_srs', 'EPSG:4326',
    outPath,
    shpPath
  ]);

  const result = gdal.getFileBytes(outPath);
  return JSON.parse(new TextDecoder().decode(result));
}

// Reproject on the fly
async function reproject(geojsonBuffer, fromEPSG, toEPSG) {
  await gdal.open('/vsimem/in.geojson', geojsonBuffer);
  gdal.ogr2ogr([
    '-f', 'GeoJSON',
    '-s_srs', `EPSG:${fromEPSG}`,
    '-t_srs', `EPSG:${toEPSG}`,
    '/vsimem/out.geojson',
    '/vsimem/in.geojson'
  ]);
  return gdal.getFileBytes('/vsimem/out.geojson');
}
```

**Use cases**: Client-side format conversion tools, user-facing "upload any geo format" workflows, raster analysis without server round-trips.

### PROJ WASM

The authoritative PROJ library compiled to WASM. More accurate than `proj4js` for complex datum transformations (grid-based shifts, NADCON, etc).

```javascript
import proj4 from 'proj4'; // proj4js: good for simple cases

// For complex/authoritative transforms, use proj-wasm
import initProj from 'proj-wasm';

const proj = await initProj();
const result = proj.transform('EPSG:4326', 'EPSG:32650', [116.4, 39.9]);
```

### Turf.js (Pure JS, No WASM)

Not WASM, but deserves mention as the standard spatial analysis library:

```bash
npm install @turf/turf
```

```javascript
import * as turf from '@turf/turf';

// Buffer analysis
const buffered = turf.buffer(point, 500, { units: 'meters' });

// Spatial join
const joined = turf.pointsWithinPolygon(points, polygon);

// Voronoi diagram
const voronoi = turf.voronoi(points, { bbox: [-180, -90, 180, 90] });

// Area calculation
const area = turf.area(polygon); // square meters

// Nearest neighbor
const nearest = turf.nearestPoint(targetPoint, featureCollection);
```

### Pyodide (Python Geospatial Stack)

Full CPython in WASM. Enables Shapely, GeoPandas, Fiona, Rasterio — the entire Python GIS ecosystem in the browser.

```javascript
import { loadPyodide } from 'pyodide';

const pyodide = await loadPyodide();
await pyodide.loadPackage(['shapely', 'geopandas']);

const result = pyodide.runPython(`
  import geopandas as gpd
  import json

  gdf = gpd.read_file('data.geojson')
  buffered = gdf.buffer(0.001)
  dissolved = buffered.dissolve()
  dissolved.to_json()
`);
```

**Caveats**: Pyodide is ~40 MB download, startup is ~5s. Use for power-user tools, not primary UX.

---

## Loading Strategy Decision Tree

```
How big is your data?
├── < 1 MB
│   └── GeoJSON, just fetch() it. Done.
│
├── 1–10 MB
│   ├── Need spatial filter?
│   │   └── FlatGeobuf (viewport-based range requests)
│   └── No filter needed?
│       └── GeoJSON with Web Worker parse
│
├── 10–100 MB
│   ├── Map rendering primary use case?
│   │   └── PMTiles (vector tiles, GPU rendering, tippecanoe preprocessing)
│   ├── Analysis primary use case?
│   │   └── GeoParquet + DuckDB WASM
│   └── Both?
│       └── Dual format: PMTiles for rendering + DuckDB WASM for queries
│
├── 100 MB – 1 GB
│   ├── Static data that changes infrequently?
│   │   └── PMTiles on CDN (S3 + CloudFront)
│   └── Dynamic or frequently queried?
│       └── Server-side PostGIS + Martin tile server + REST API
│
└── > 1 GB
    └── Always server-side. PostGIS + Martin/TiTiler, stream results.
        Client-side is not viable at this scale.
```

**Format selection by use case:**

| Use case | Recommended | Why |
|----------|-------------|-----|
| Prototype / demo | GeoJSON fetch | Zero setup |
| Admin boundaries | TopoJSON | Shared arcs, 5x smaller |
| User-uploaded data | shapefile.js / GDAL WASM | Accept anything |
| Viewport-based filter | FlatGeobuf | R-tree range requests |
| Large-scale rendering | PMTiles (MVT) | GPU tile rendering |
| Analytics dashboard | DuckDB WASM + GeoParquet | SQL in browser |
| Satellite imagery | COG + geotiff.js | Range-request tiles |
| GPS tracks | togeojson (GPX/KML) | Direct format support |
| Point clouds (millions) | GeoArrow + deck.gl | Zero-copy GPU path |
| Raster analysis | COG + geotiff.js or GDAL WASM | Band math in browser |

---

## Advanced Patterns

### Dual Format: PMTiles + FlatGeobuf

PMTiles for rendering (GPU vector tiles, fast visual update), FlatGeobuf for detail-on-demand (exact properties, full geometry on click). Avoids encoding all attributes into vector tiles.

```javascript
// PMTiles: fast map rendering
map.addSource('polygons-tiles', {
  type: 'vector',
  url: 'pmtiles://https://cdn.example.com/data.pmtiles'
});

// On click, fetch exact feature from FlatGeobuf
map.on('click', 'polygon-fill', async (e) => {
  const clickedId = e.features[0].properties.id;
  const bbox = featureBboxFromId(clickedId); // compute tight bbox

  const features = [];
  for await (const feature of deserialize(fgbUrl, bbox)) {
    if (feature.properties.id === clickedId) {
      features.push(feature);
      break; // Found it
    }
  }

  showDetailPanel(features[0]);
});
```

### Predictive Prefetch

Prefetch FlatGeobuf tiles outside the current viewport based on pan direction, reducing latency when the user moves the map.

```javascript
const PREFETCH_FACTOR = 1.5; // Load 1.5x the viewport area

map.on('moveend', () => {
  const bounds = map.getBounds();
  const center = map.getCenter();
  const panDir = getPanDirection(prevCenter, center); // returns {dx, dy}

  // Expand bbox in pan direction
  const extendedBbox = extendBbox(bounds, panDir, PREFETCH_FACTOR);

  // Load in background
  loadFGBInBackground(fgbUrl, extendedBbox);
  prevCenter = center;
});
```

### IndexedDB Tile Cache for PMTiles

```javascript
const DB_NAME = 'tile-cache';
const STORE = 'tiles';

async function openTileCache() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = (e) => e.target.result.createObjectStore(STORE);
    req.onsuccess = (e) => resolve(e.target.result);
    req.onerror = reject;
  });
}

async function getCachedTile(db, key) {
  return new Promise((resolve) => {
    const tx = db.transaction(STORE, 'readonly');
    const req = tx.objectStore(STORE).get(key);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => resolve(null);
  });
}

async function setCachedTile(db, key, data) {
  return new Promise((resolve) => {
    const tx = db.transaction(STORE, 'readwrite');
    tx.objectStore(STORE).put(data, key);
    tx.oncomplete = resolve;
  });
}
```

### Service Worker for Offline Tile Serving

```javascript
// service-worker.js
const TILE_CACHE = 'tile-cache-v1';

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Intercept tile requests
  if (url.pathname.match(/\/tiles\/\d+\/\d+\/\d+\.(pbf|png|webp)/)) {
    event.respondWith(
      caches.open(TILE_CACHE).then(async (cache) => {
        const cached = await cache.match(event.request);
        if (cached) return cached;

        const response = await fetch(event.request);
        if (response.ok) {
          cache.put(event.request, response.clone());
        }
        return response;
      })
    );
  }
});
```

### Shared Worker: Single DuckDB WASM Across Tabs

One DuckDB WASM instance shared across all browser tabs. Eliminates repeated 4 MB downloads and WASM initialization per tab.

```javascript
// shared-duckdb-worker.js
importScripts('@duckdb/duckdb-wasm/dist/duckdb-browser-coi.worker.js');

let db, conn;

self.onconnect = async (event) => {
  const port = event.ports[0];

  if (!db) {
    db = await initDuckDB();
    conn = await db.connect();
    await conn.query('INSTALL spatial; LOAD spatial;');
  }

  port.onmessage = async (e) => {
    const { id, sql } = e.data;
    try {
      const result = await conn.query(sql);
      port.postMessage({ id, result: result.toArray() });
    } catch (err) {
      port.postMessage({ id, error: err.message });
    }
  };
};

// In application code:
const worker = new SharedWorker('./shared-duckdb-worker.js');
worker.port.start();

function queryDuckDB(sql) {
  return new Promise((resolve, reject) => {
    const id = crypto.randomUUID();
    worker.port.onmessage = (e) => {
      if (e.data.id === id) {
        e.data.error ? reject(new Error(e.data.error)) : resolve(e.data.result);
      }
    };
    worker.port.postMessage({ id, sql });
  });
}
```

### Zero-Copy ArrayBuffer Transfer Between Workers

When transferring large binary data between workers, transfer ownership rather than copying. This is O(1) regardless of size.

```javascript
// COPY (slow): 100 MB takes ~100 ms to copy
worker.postMessage({ data: largeArrayBuffer });

// TRANSFER (fast): 100 MB takes <1 ms — memory moved, not copied
worker.postMessage({ data: largeArrayBuffer }, [largeArrayBuffer]);
// Note: largeArrayBuffer is now detached (length = 0) in sender
```

### Streaming Decompression in Browser

Decompress gzip on-the-fly without buffering the full response:

```javascript
async function fetchGzipped(url) {
  const response = await fetch(url);

  const decompressedStream = response.body
    .pipeThrough(new DecompressionStream('gzip'))
    .pipeThrough(new TextDecoderStream());

  const reader = decompressedStream.getReader();
  let text = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    text += value;
  }

  return JSON.parse(text);
}

// Or collect into a single string more efficiently
async function fetchGzippedEfficient(url) {
  const response = await fetch(url);
  const decompressed = response.body.pipeThrough(new DecompressionStream('gzip'));
  const text = await new Response(decompressed).text();
  return JSON.parse(text);
}
```

### Blob URL for Large Generated Datasets

Avoid passing large GeoJSON objects through postMessage. Create a Blob URL instead:

```javascript
// In a Web Worker — generate large GeoJSON
const geojson = generateLargeGeoJSON(); // 50 MB object
const blob = new Blob([JSON.stringify(geojson)], { type: 'application/json' });
const blobUrl = URL.createObjectURL(blob);

self.postMessage({ blobUrl }); // Transfer the tiny URL string, not the data

// In main thread
worker.onmessage = (e) => {
  map.addSource('generated', { type: 'geojson', data: e.data.blobUrl });
  // MapLibre fetches directly from the Blob URL
  // Remember to revoke when done:
  // URL.revokeObjectURL(e.data.blobUrl);
};
```

### OPFS (Origin Private File System) for Large Persistent Data

Cache large GeoParquet or PMTiles files in the browser's OPFS — persistent storage with file-system semantics, accessible from Web Workers.

```javascript
async function cacheParquetFile(url, filename) {
  const root = await navigator.storage.getDirectory();
  const fileHandle = await root.getFileHandle(filename, { create: true });

  // Check if already cached
  const file = await fileHandle.getFile();
  if (file.size > 0) {
    console.log('Serving from OPFS cache');
    return fileHandle;
  }

  // Download and cache
  const response = await fetch(url);
  const writable = await fileHandle.createWritable();
  await response.body.pipeTo(writable);

  console.log('Cached to OPFS:', filename, 'bytes:', file.size);
  return fileHandle;
}
```

---

## Library Reference

| Library | npm package | Size | Purpose |
|---------|-------------|------|---------|
| FlatGeobuf | `flatgeobuf` | 120 KB | Binary geo streaming |
| DuckDB WASM | `@duckdb/duckdb-wasm` | 4 MB WASM | SQL analytics |
| geotiff.js | `geotiff` | 200 KB | COG raster reading |
| topojson-client | `topojson-client` | 20 KB | TopoJSON decode/dissolve |
| topojson-server | `topojson-server` | 40 KB | TopoJSON creation |
| shapefile.js | `shapefile` | 30 KB | Shapefile parse |
| shp-write | `shp-write` | 50 KB | Shapefile export |
| Papa Parse | `papaparse` | 80 KB | CSV/TSV streaming |
| togeojson | `@mapbox/togeojson` | 30 KB | GPX/KML parse |
| oboe.js | `oboe` | 30 KB | Streaming JSON parse |
| Apache Arrow | `apache-arrow` | 800 KB | Columnar data in JS |
| Turf.js | `@turf/turf` | 1 MB | Spatial analysis |
| gdal3.js | `gdal3.js` | 10 MB WASM | Full GDAL in browser |
| stac-js | `stac-js` | 50 KB | STAC catalog search |
| @mapbox/vector-tile | `@mapbox/vector-tile` | 30 KB | MVT decode |
| pmtiles | `pmtiles` | 50 KB | PMTiles protocol |
