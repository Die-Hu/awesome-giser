# Data Formats & Loading

> Data validated: 2026-03-21

## 30-Second Decision

| I need... | Use this | Startup time |
|-----------|----------|-------------|
| Fastest setup, small data (<1MB) | GeoJSON | 0 min (it's just JSON) |
| Boundary data, smaller transfer | TopoJSON | 10 min |
| Medium data with spatial filter, no server | FlatGeobuf | 15 min |
| GPS tracks, KML import | GPX/KML via @tmcw/togeojson | 5 min |
| CSV point data | Papa Parse | 5 min |
| Legacy Shapefile import | shapefile.js (trusted sources only) | 10 min |
| SQL analytics on geo data in browser | GeoParquet + DuckDB-WASM | 1 hour |
| Raster imagery in browser | COG + geotiff.js | 30 min |

---

## Format Comparison

| Format | Size (10K polygons) | Spatial filter | Streaming | Best for |
|--------|---------------------|----------------|-----------|----------|
| GeoJSON | ~5 MB | No | No | Small data, prototypes |
| TopoJSON | ~1 MB | No | No | Shared boundaries |
| FlatGeobuf | ~2 MB | Yes (range) | Yes | Medium data, spatial filter |
| GeoParquet | ~1 MB | Yes (via DuckDB) | Partial | Analytics, large data |
| CSV/TSV | ~2 MB | No | With Papa Parse | Point data |
| GPX/KML | ~3 MB | No | No | GPS tracks |
| Shapefile | ~4 MB + .dbf + .shx | No | No | Legacy import |
| COG | varies | Yes (window) | Yes | Raster data |

---

## Detailed Guide (by startup time)

### 1. GeoJSON -- The Universal Format

It's just JSON. Every library reads it, every API speaks it, the browser parses it natively.

```
Loading strategy by size:
  <1 MB:    fetch() -> map.addSource()           -- just do it
  1-10 MB:  fetch() -> Web Worker parse -> map    -- offload parsing
  >10 MB:   DON'T USE GEOJSON. Use FlatGeobuf or vector tiles.
```

```javascript
const response = await fetch('/data/points.geojson');
const geojson = await response.json();
map.addSource('points', { type: 'geojson', data: geojson });
```

**Small project:** Default choice. Zero setup, human-readable, every map library supports it natively.

**Key caveats:**
- 5-10x larger than binary alternatives
- No streaming, no spatial filtering
- Main thread parsing blocks UI for >10MB
- **Anti-pattern:** Serving 50MB+ GeoJSON from an API. Use vector tiles or FlatGeobuf.

---

### 2. TopoJSON

GeoJSON with shared boundary compression. 60-80% smaller for admin regions, parcels, census tracts.

```javascript
import { feature } from 'topojson-client';
const topology = await fetch('/data/us-states.topojson').then(r => r.json());
const states = feature(topology, topology.objects.states); // -> GeoJSON
map.addSource('states', { type: 'geojson', data: states });
```

**Small project:** Good for boundary data (choropleth maps). Combined with D3, standard for static cartographic visualization.

**Key caveats:**
- Only smaller when features share boundaries; minimal savings for point data
- Requires conversion step (`topojson-server`)
- Most map libraries expect GeoJSON -- must convert back with `topojson-client`

---

### 3. CSV/TSV + Papa Parse

Simplest path for point data. Export from Excel/Google Sheets, load directly.

```javascript
import Papa from 'papaparse';
Papa.parse('/data/sensors.csv', {
  download: true, header: true, dynamicTyping: true,
  step: (row) => addPoint(row.data.latitude, row.data.longitude)
});
```

**Key caveats:** Points only (practically). No spatial indexing. Coordinate column names vary ("lat", "latitude", "y").

---

### 4. GPX / KML via togeojson

Import GPS tracks and Google Earth data. One-liner conversion to GeoJSON.

```javascript
import { gpx } from '@tmcw/togeojson';
const gpxText = await fetch('/track.gpx').then(r => r.text());
const dom = new DOMParser().parseFromString(gpxText, 'text/xml');
const geojson = gpx(dom);
map.addSource('track', { type: 'geojson', data: geojson });
```

**Key caveats:** XML parsing slow for large files. Complex KML features (NetworkLinks, 3D models) not fully supported.

---

### 5. FlatGeobuf

Streaming binary format with built-in spatial index. HTTP range requests enable bbox-filtered loading from static hosting.

**Quick facts:** ~7.8K npm/week | 4/5 production-readiness | Adopted by QGIS, GDAL

```javascript
import { deserialize } from 'flatgeobuf/lib/mjs/geojson';

const bbox = { minX: 116.0, minY: 39.5, maxX: 117.0, maxY: 40.5 };
const iter = deserialize('https://cdn.example.com/data.fgb', bbox);
const features = [];
for await (const feature of iter) features.push(feature);
map.addSource('filtered', { type: 'geojson', data: { type: 'FeatureCollection', features } });
```

**Small project:** Worth using when you have 1K-100K features too large for GeoJSON on static hosting (S3/CDN). Zero backend required.

**Key caveats:**
- Requires HTTP Range request support (test your hosting)
- Read-only in browser; must create .fgb files server-side (GDAL, ogr2ogr)
- No attribute-only queries -- bbox spatial filter only
- Static file: must regenerate for data changes

---

### 6. Shapefile (Legacy Import)

Legacy format still common in government data portals. Use for import, not as application format.

**Note:** shapefile.js has not had a new release in ~8 years. It works for reading Shapefiles, but **use only for trusted data sources** -- no security patches means potential risk with untrusted user uploads.

```javascript
import { read } from 'shapefile';
const source = await read('/data/parcels.shp', '/data/parcels.dbf');
const features = [];
let result;
while (!(result = await source.read()).done) features.push(result.value);
```

**Key caveats:**
- Multi-file format (.shp + .dbf + .shx minimum); missing any file = broken data
- Encoding issues with non-Latin characters (check .cpg file)
- 2GB file size limit
- **Anti-pattern:** Using Shapefile as your application format. Convert to GeoJSON or FlatGeobuf immediately.

---

### 7. GeoParquet + DuckDB-WASM

SQL analytics on geospatial data directly in the browser. Columnar storage, predicate pushdown, aggregation.

**Quick facts:** DuckDB-WASM 4-8MB (must lazy-load) | Production-readiness: 3/5

**Recommended for:** Analysis dashboards, data exploration, aggregation queries without a backend.

**Not recommended for:** Interactive map applications -- the 4-8MB WASM download and 1-3 second cold start make it inappropriate for instant-interaction UIs.

```javascript
import * as duckdb from '@duckdb/duckdb-wasm';
const db = await duckdb.AsyncDuckDB.create();
await db.open({});
const conn = await db.connect();
await conn.query("INSTALL spatial; LOAD spatial;");

const result = await conn.query(`
  SELECT zone, COUNT(*) as count, SUM(ST_Area(geometry)) as total_area
  FROM 'parcels.parquet'
  WHERE ST_Within(geometry, ST_GeomFromText('POLYGON((...))'))
  GROUP BY zone
`);
```

**Key caveats:**
- WASM bundle 4-8MB -- must lazy-load, never include in initial bundle
- Cold start 1-3 seconds -- not for instant-interaction apps
- Browser memory ~2-4GB; large queries can crash with no graceful error
- npm supply chain incident (September 2025, fixed within 4 hours) -- pin versions, use lockfiles
- Spatial extension still evolving; not all PostGIS functions available
- **Anti-pattern:** Using DuckDB-WASM as a PostGIS replacement. It's analytical, not transactional.

---

## Low Priority

### Apache Arrow / GeoArrow

Zero-copy columnar data transfer between DuckDB-WASM and deck.gl. Impressive but ecosystem is immature. Do not use for small projects -- GeoJSON or FlatGeobuf is simpler for every case.

---

## Decision Flowchart

```
How big is your dataset?
+-- <1 MB -> GeoJSON (just do it)
+-- 1-10 MB -> Shared boundaries? -> TopoJSON (60-80% smaller)
|              Point data in CSV? -> Papa Parse
|              Otherwise -> GeoJSON with Web Worker parsing
+-- 10-100 MB -> Need spatial filtering? -> FlatGeobuf
|                Otherwise -> Vector tiles (PMTiles)
+-- >100 MB -> Need SQL analytics? -> GeoParquet + DuckDB-WASM
               Otherwise -> Vector tiles (PMTiles)
```
