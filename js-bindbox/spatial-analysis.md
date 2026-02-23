# Spatial Analysis in JavaScript

> Client-side and server-side JavaScript libraries for performing geospatial analysis, coordinate transformations, and spatial indexing.

> **Quick Picks**
> - **SOTA**: [Turf.js](https://turfjs.org) -- modular, tree-shakable, covers 90% of browser-side spatial ops
> - **Free Best**: [H3-js](https://h3geo.org) -- Uber's hexagonal spatial index, ideal for aggregation and analytics
> - **Fastest Setup**: [Turf.js](https://turfjs.org) -- `npm install @turf/turf` and start analyzing in one line

## Turf.js

The most popular geospatial analysis library for JavaScript, providing a modular set of spatial operations.

- **Website:** [turfjs.org](https://turfjs.org)
- **Key Strengths:** Modular (tree-shakable), GeoJSON native, comprehensive operations
- **Best For:** Client-side spatial analysis, feature manipulation, measurement

### Install

```bash
# Full library
npm install @turf/turf

# Individual modules (recommended for bundle size)
npm install @turf/buffer @turf/intersect @turf/centroid
```

### Operations Catalog

| Category | Operations | Docs |
|----------|-----------|------|
| **Measurement** | distance, area, length, along, midpoint, center, centroid, bbox | [turfjs.org/docs/#measurement](https://turfjs.org/docs/#measurement) |
| **Transformation** | buffer, union, intersect, difference, dissolve, simplify, bezierSpline | [turfjs.org/docs/#transformation](https://turfjs.org/docs/#transformation) |
| **Classification** | pointsWithinPolygon, tag, nearest, tin, voronoi | [turfjs.org/docs/#classification](https://turfjs.org/docs/#classification) |
| **Aggregation** | collect, clustersDbscan, clustersKmeans | [turfjs.org/docs/#aggregation](https://turfjs.org/docs/#aggregation) |
| **Interpolation** | interpolate, isobands, isolines, planepoint, tin | [turfjs.org/docs/#interpolation](https://turfjs.org/docs/#interpolation) |
| **Grids** | hexGrid, pointGrid, squareGrid, triangleGrid | [turfjs.org/docs/#grids](https://turfjs.org/docs/#grids) |
| **Boolean** | booleanContains, booleanCrosses, booleanOverlap, booleanWithin | [turfjs.org/docs/#booleans](https://turfjs.org/docs/#booleans) |
| **Joins** | tag, spatialJoin (via pointsWithinPolygon) | [turfjs.org/docs/#joins](https://turfjs.org/docs/#joins) |

### Common Operation Examples

```javascript
import * as turf from '@turf/turf';

// Buffer a point by 5 km
const point = turf.point([116.4, 39.9]);
const buffered = turf.buffer(point, 5, { units: 'kilometers' });

// Calculate area of a polygon (sq meters)
const area = turf.area(polygon);

// Points within polygon
const within = turf.pointsWithinPolygon(pointCollection, polygon);

// Distance between two points (km)
const dist = turf.distance(
  turf.point([116.4, 39.9]),
  turf.point([121.5, 31.2]),
  { units: 'kilometers' }
);

// Hex grid over a bounding box
const bbox = [116.0, 39.5, 117.0, 40.5];
const hexgrid = turf.hexGrid(bbox, 2, { units: 'kilometers' });
```

---

## JSTS

A JavaScript port of the Java Topology Suite (JTS), offering robust computational geometry operations.

- **Repository:** [github.com/bjornharrtell/jsts](https://github.com/bjornharrtell/jsts)
- **Key Strengths:** Topological correctness, full JTS API, precision model
- **Best For:** Complex polygon operations, topology validation, precision-sensitive analysis

### Install

```bash
npm install jsts
```

### Usage & Comparison with Turf.js

```javascript
import { GeoJSONReader, GeoJSONWriter } from 'jsts/org/locationtech/jts/io';
import { BufferOp } from 'jsts/org/locationtech/jts/operation/buffer';

const reader = new GeoJSONReader();
const writer = new GeoJSONWriter();

// Read GeoJSON, buffer, write back
const geom = reader.read(geojsonPolygon);
const buffered = BufferOp.bufferOp(geom, 0.01); // degrees
const result = writer.write(buffered);
```

| Aspect | Turf.js | JSTS |
|--------|---------|------|
| API style | Functional, GeoJSON in/out | OOP, JTS-compatible |
| Topological correctness | Good for most cases | Guaranteed (precision model) |
| Bundle size | Tree-shakable, small per-module | ~200 KB full |
| Best for | 90% of spatial ops | Edge cases, validation, topology |

> **Rule of thumb:** Use Turf.js first. Reach for JSTS when you hit topological edge cases (self-intersecting polygons, precision errors in boolean ops).

---

## Proj4js

JavaScript library for coordinate transformations between different map projections.

- **Repository:** [github.com/proj4js/proj4js](https://github.com/proj4js/proj4js)
- **Key Strengths:** Supports thousands of CRS definitions, EPSG codes, custom projections
- **Best For:** Reprojecting data between coordinate systems in the browser

### Install

```bash
npm install proj4
```

```javascript
import proj4 from 'proj4';

// WGS84 to Web Mercator
const result = proj4('EPSG:4326', 'EPSG:3857', [116.4, 39.9]);
console.log(result); // [12958175.4, 4852834.1]

// Register a custom projection (e.g., CGCS2000 / China zone)
proj4.defs('EPSG:4547',
  '+proj=tmerc +lat_0=0 +lon_0=114 +k=1 +x_0=500000 +y_0=0 ' +
  '+ellps=GRS80 +units=m +no_defs'
);

const cgcs = proj4('EPSG:4326', 'EPSG:4547', [114.0, 30.5]);
```

> **Tip:** Look up proj4 strings at [epsg.io](https://epsg.io) -- search by EPSG code and copy the Proj4js definition.

---

## geotiff.js

Read and parse GeoTIFF / Cloud Optimized GeoTIFF (COG) files directly in the browser.

- **Repository:** [github.com/geotiffjs/geotiff.js](https://github.com/geotiffjs/geotiff.js)
- **Key Strengths:** COG support with HTTP range requests, multi-band, Web Worker compatible
- **Best For:** Loading raster data (elevation, satellite imagery, analysis results) client-side

### Install

```bash
npm install geotiff
```

```javascript
import { fromUrl } from 'geotiff';

// Read a Cloud Optimized GeoTIFF from a URL
const tiff = await fromUrl('https://example.com/data/elevation.tif');
const image = await tiff.getImage();
const data = await image.readRasters();

console.log('Width:', image.getWidth());
console.log('Height:', image.getHeight());
console.log('Bands:', data.length);
console.log('Pixel [0][0]:', data[0][0]); // first band, first pixel
```

---

## loam (GDAL in WebAssembly)

GDAL compiled to WebAssembly for running raster/vector geospatial operations in the browser.

- **Repository:** [github.com/azavea/loam](https://github.com/azavea/loam)
- **Key Strengths:** Real GDAL in the browser, supports 140+ raster/vector formats, reprojection, warping
- **Best For:** Format conversion, reprojection, and raster analysis without a server

### Install

```bash
npm install loam
```

```javascript
import loam from 'loam';

await loam.initialize();

// Open a GeoTIFF and reproject it
const dataset = await loam.open(file);
const info = await dataset.convert(['-t_srs', 'EPSG:4326']);
const reprojected = await info.bytes();
```

> **Caveat:** loam loads a ~10 MB WASM binary. Best suited for tools/apps where server-side processing is not available.

---

## H3-js (Hexagonal Spatial Index)

JavaScript bindings for Uber's H3 hierarchical hexagonal spatial indexing system.

- **Website:** [h3geo.org](https://h3geo.org)
- **Repository:** [github.com/uber/h3-js](https://github.com/uber/h3-js)
- **Key Strengths:** Hierarchical resolution (0-15), uniform area hexagons, fast neighbor/ring operations
- **Best For:** Spatial aggregation, analytics, grid-based analysis, joining point data to areas

### Install

```bash
npm install h3-js
```

```javascript
import { latLngToCell, cellToBoundary, gridDisk, cellToLatLng } from 'h3-js';

// Convert a lat/lng to an H3 cell at resolution 7
const h3Index = latLngToCell(39.9, 116.4, 7);
console.log(h3Index); // e.g., '872830828ffffff'

// Get the hexagon boundary as [lat, lng] pairs
const boundary = cellToBoundary(h3Index);

// Get all hexagons within 2 rings
const neighbors = gridDisk(h3Index, 2);

// Aggregate points into hex bins
const hexCounts = {};
points.forEach(p => {
  const cell = latLngToCell(p.lat, p.lng, 7);
  hexCounts[cell] = (hexCounts[cell] || 0) + 1;
});
```

| H3 Resolution | Avg Hex Area | Use Case |
|---------------|-------------|----------|
| 3 | ~12,400 km^2 | Country/continent analytics |
| 5 | ~253 km^2 | Regional aggregation |
| 7 | ~5.16 km^2 | City-level analysis |
| 9 | ~0.105 km^2 | Neighborhood level |
| 11 | ~0.002 km^2 | Street level |

---

## Flatbush / RBush

High-performance spatial indexing libraries for fast point and rectangle queries.

### Flatbush

- **Type:** Static spatial index (R-tree)
- **Best For:** Indexing large static datasets for fast spatial queries
- **Performance:** Extremely fast for bulk-loaded data

### RBush

- **Type:** Dynamic spatial index (R-tree)
- **Best For:** Data that changes frequently, real-time applications
- **Performance:** Supports insert/remove operations

### Install

```bash
npm install flatbush rbush
```

```javascript
import Flatbush from 'flatbush';

// Index 1 million bounding boxes
const index = new Flatbush(points.length);
points.forEach(p => index.add(p.minX, p.minY, p.maxX, p.maxY));
index.finish();

// Query: find all items intersecting a bbox
const results = index.search(116.0, 39.5, 117.0, 40.5);
// results = array of indices into original data
```

| Library | Type | Insert/Remove | Bulk Load | Memory | Best For |
|---------|------|---------------|-----------|--------|----------|
| Flatbush | Static R-tree | No | Yes | Low | Large static datasets |
| RBush | Dynamic R-tree | Yes | Yes | Medium | Dynamic, real-time data |

---

## Common Operation Recipes

| Recipe | Libraries Used | Description |
|--------|---------------|-------------|
| Point-in-polygon query | Turf.js + RBush | Fast spatial filtering of points |
| Buffer & dissolve | Turf.js | Service area generation |
| Reproject & analyze | Proj4js + Turf.js | Analyze data in projected CRS |
| Topology validation | JSTS | Validate and repair polygon geometry |
| Nearest neighbor search | Flatbush + Turf.js | Find closest features efficiently |
| Hex binning & aggregation | H3-js | Aggregate point data to uniform hexagons |
| Raster value extraction | geotiff.js + Turf.js | Sample elevation/raster at point locations |

### Recipe: Point-in-Polygon with Spatial Index

```javascript
import RBush from 'rbush';
import * as turf from '@turf/turf';

// Build spatial index on polygons
const tree = new RBush();
polygons.features.forEach((f, i) => {
  const [minX, minY, maxX, maxY] = turf.bbox(f);
  tree.insert({ minX, minY, maxX, maxY, index: i });
});

// For each point, find candidate polygons via bbox, then exact test
points.features.forEach(pt => {
  const [x, y] = pt.geometry.coordinates;
  const candidates = tree.search({ minX: x, minY: y, maxX: x, maxY: y });
  for (const c of candidates) {
    if (turf.booleanPointInPolygon(pt, polygons.features[c.index])) {
      pt.properties.region = polygons.features[c.index].properties.name;
      break;
    }
  }
});
```
