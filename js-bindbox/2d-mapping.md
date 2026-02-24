# 2D Mapping Libraries -- Deep Dive

> A senior web GIS engineer's definitive reference for Leaflet, MapLibre GL JS, OpenLayers, and MapTiler SDK. Production-tested numbers, real tradeoffs, and patterns that actually ship.

---

## Quick Picks

| Goal | Pick | Why |
|------|------|-----|
| **Get something on screen in 10 min** | MapTiler SDK | Pre-configured MapLibre + free basemaps, one API key |
| **Open-source vector tiles, full control** | MapLibre GL JS | WebGL, style spec, no vendor lock-in |
| **Simple raster map, 700+ plugins** | Leaflet | Dead-simple API, massive ecosystem, 40 KB gzip |
| **Enterprise OGC (WMS/WFS/WMTS), projections** | OpenLayers | Built-in everything, no plugins needed |
| **10K–500K features interactive** | MapLibre GL JS | WebGL rendering, MVT, consistent 60fps |
| **Offline PWA, PMTiles, zero-server** | MapLibre + PMTiles protocol | `addProtocol()` + `pmtiles://` prefix |
| **Chinese projections (CGCS2000, GCJ02)** | OpenLayers or Leaflet + proj4 | Both support custom CRS; MapLibre is EPSG:3857 only |

---

## Library Deep Dives

---

## Leaflet

Lightweight, mobile-friendly interactive mapping library. First released in 2011 by Vladimir Agafonkin, still the most-starred JS mapping library on GitHub (~42K stars). The simplicity is the feature.

### Performance Profile

**Rendering engine**

Leaflet defaults to SVG for vector layers (polygons, polylines) and positions HTML `<img>` elements for markers. Each feature becomes a DOM node — fast to interact with (native browser hit-testing), but costly at scale. The Canvas renderer swaps the SVG backend for a single `<canvas>` element, dramatically reducing DOM pressure.

```javascript
// Activate Canvas renderer globally (must do before creating layers)
const map = L.map('map', {
  renderer: L.canvas()  // or L.svg() (default)
});

// Or per-layer
const layer = L.geoJSON(data, {
  renderer: L.canvas({ padding: 0.5 })
});
```

**Feature count limits (empirical)**

| Renderer | Comfortable | Starts to stutter | Hard ceiling |
|----------|-------------|-------------------|--------------|
| SVG (default) | ~2,000 features | ~5,000 | ~10,000 (DOM bloat) |
| Canvas | ~20,000 features | ~50,000 | ~100,000 (CPU raster) |
| Marker HTML | ~500 markers | ~1,500 | ~3,000 (layout thrash) |
| markercluster | ~100,000 points | ~500,000 | ~1M (cluster calc) |

These numbers assume complex polygons with styling on a mid-range laptop. Simple point geometries can push 2–3x higher.

**Memory footprint**

Leaflet stores feature data as JavaScript objects in the layer tree. Approximate heap usage:

- 10K GeoJSON polygons (moderate complexity): ~30–60 MB
- 100K point features (markers): ~200–400 MB (mostly DOM nodes)
- 100K point features (Canvas renderer): ~40–80 MB
- Tile layers: minimal (browser image cache handles tiles)

**FPS benchmarks**

- SVG, < 2K features: 60fps pan/zoom on all modern devices
- Canvas, 20K features: 55–60fps; drops to 30–40fps at 50K during rapid pan
- DOM markers, > 1K: visible jank on mobile at 30–60fps
- Tile loading: purely network-bound; WebWorkers not used

**GPU vs CPU**

Leaflet is entirely CPU-bound. There is no WebGL path. All rendering happens on the main thread — heavy Canvas redraws block user input. Mitigation strategies: requestAnimationFrame throttling, off-screen canvas, or switching to MapLibre for large datasets.

**Tile loading**

- Default: 4 concurrent tile requests per subdomain (a/b/c subdomains = up to 12 parallel)
- `maxNativeZoom`/`minNativeZoom` control tile reuse
- `keepBuffer` tiles cached in DOM (`default: 2` rows/cols around viewport)
- No tile request cancellation in Leaflet core (pending requests continue even after zoom change)
- Use `updateWhenIdle: true` to only redraw on drag-end, not during pan

```javascript
L.tileLayer(url, {
  maxNativeZoom: 18,
  maxZoom: 22,          // over-zooms tile at 22 instead of 404
  keepBuffer: 4,        // extra tiles around viewport
  updateWhenIdle: true, // skip mid-pan redraws
  updateWhenZooming: false
});
```

### Development Experience

**TypeScript support**

- `@types/leaflet` package (DefinitelyTyped) — not bundled, install separately
- Types are community-maintained, sometimes lag behind releases
- Styling properties are typed as `string | number` (loose), not as discriminated unions
- Plugin types vary wildly: some have `@types/`, most require manual declaration

```bash
npm install leaflet @types/leaflet
```

**Bundle size**

| Asset | Raw | Gzip |
|-------|-----|------|
| leaflet.js | ~143 KB | ~40 KB |
| leaflet.css | ~7 KB | ~2 KB |

Leaflet is not tree-shakable in the traditional sense — it uses a monolithic `L` namespace. However, the library is already compact. The real bundle bloat comes from plugins (markercluster adds ~30 KB gzip, Leaflet.draw ~25 KB).

```javascript
// You cannot selectively import — it's always the full bundle
import L from 'leaflet';
// NOT: import { map, tileLayer } from 'leaflet' -- won't tree-shake
```

**Learning curve**

- Time to first working map: 5 minutes (literally)
- Time to production-ready app with custom markers, popups, clustering: 4–8 hours
- Time to master custom layers, Canvas renderer, custom CRS: 2–4 days

**Debugging**

- Browser DevTools: inspect SVG elements directly, CSS debugging works naturally
- `map.getLayers()` and `layer.getLatLngs()` in console
- No dedicated DevTools extension
- Source maps included in npm package
- `L.stamp(layer)` gives internal ID for debugging layer references

**Documentation quality**

- Official docs: excellent for core API, comprehensive reference
- Community: very high Stack Overflow density (30K+ tagged questions)
- Tutorials: abundant (OSM wiki, official tutorials, blog posts)
- Plugin docs: extremely variable quality

**Breaking changes**

- Major versions: 0.7 → 1.0 (2016) was the big one, significant API rewrites
- 1.x series (2016–present): stable, minimal breaking changes
- 2.0 was announced and stalled; the library is in maintenance mode
- Verdict: extremely stable API, minimal migration risk

### Plugin & Extension Ecosystem

Leaflet's 700+ plugin ecosystem is its crown jewel. Here are the essential plugins organized by category.

**Clustering & Density**

| Plugin | npm | Notes |
|--------|-----|-------|
| [Leaflet.markercluster](https://github.com/Leaflet/Leaflet.markercluster) | `leaflet.markercluster` | The standard; animated, spider-view, custom icons |
| [Leaflet.markercluster.layersupport](https://github.com/ghybs/Leaflet.MarkerCluster.LayerSupport) | same | Integrates clusters with layer control |
| [Leaflet.heat](https://github.com/Leaflet/Leaflet.heat) | `leaflet.heat` | Canvas heatmap, simpleheat under the hood |
| [heatmap.js](https://github.com/pa7/heatmap.js) | `heatmap.js` | Higher quality heatmap, configurable |

```javascript
// markercluster with custom cluster icon
import L from 'leaflet';
import 'leaflet.markercluster';

const clusterGroup = L.markerClusterGroup({
  maxClusterRadius: 80,
  spiderfyOnMaxZoom: true,
  showCoverageOnHover: true,
  iconCreateFunction(cluster) {
    const count = cluster.getChildCount();
    const size = count > 100 ? 'large' : count > 10 ? 'medium' : 'small';
    return L.divIcon({
      html: `<div><span>${count}</span></div>`,
      className: `marker-cluster marker-cluster-${size}`,
      iconSize: L.point(40, 40)
    });
  }
});
```

**Drawing & Editing**

| Plugin | npm | Notes |
|--------|-----|-------|
| [Leaflet-Geoman](https://github.com/geoman-io/leaflet-geoman) | `@geoman-io/leaflet-geoman-free` | Best-in-class, snapping, cut, rotate, GeoJSON I/O |
| [Leaflet.draw](https://github.com/Leaflet/Leaflet.draw) | `leaflet-draw` | Older, less maintained, still widely deployed |
| [Leaflet.Editable](https://github.com/Leaflet/Leaflet.Editable) | `leaflet-editable` | Minimal, hooks-based editing API |

```javascript
// Geoman — production drawing setup
import '@geoman-io/leaflet-geoman-free';
import '@geoman-io/leaflet-geoman-free/dist/leaflet-geoman.css';

map.pm.addControls({
  position: 'topleft',
  drawCircle: false,
});

map.pm.setGlobalOptions({
  snappable: true,
  snapDistance: 20,
});

map.on('pm:create', ({ layer }) => {
  const geojson = layer.toGeoJSON();
  console.log('Created:', JSON.stringify(geojson));
  map.removeLayer(layer); // remove temp layer, handle in your state
});
```

**Data Loading & Formats**

| Plugin | npm | Notes |
|--------|-----|-------|
| [leaflet-omnivore](https://github.com/mapbox/leaflet-omnivore) | `@mapbox/leaflet-omnivore` | CSV, KML, GPX, TopoJSON, WKT, polyline |
| [Leaflet.FileLayer](https://github.com/makinacorpus/Leaflet.FileLayer) | `leaflet-filelayer` | Drag-and-drop local files |
| [leaflet-ajax](https://github.com/calvinmetcalf/leaflet-ajax) | `leaflet-ajax` | Async GeoJSON loading with events |
| [leaflet.vectorgrid](https://github.com/Leaflet/Leaflet.VectorGrid) | `leaflet.vectorgrid` | MVT/PBF vector tiles in Leaflet |

**Layer Control & Management**

| Plugin | npm | Notes |
|--------|-----|-------|
| [Leaflet.groupedlayercontrol](https://github.com/ismyrnow/leaflet-groupedlayercontrol) | `leaflet-groupedlayercontrol` | Group layers in the control panel |
| [Leaflet.opacity](https://github.com/ptma/Leaflet.Opacity) | `leaflet.opacity` | Per-layer opacity slider |
| [leaflet-sidebar-v2](https://github.com/nickpeihl/leaflet-sidebar-v2) | `leaflet-sidebar-v2` | Responsive sidebar panel |

**Geocoding & Search**

| Plugin | npm | Notes |
|--------|-----|-------|
| [leaflet-control-geocoder](https://github.com/perliedman/leaflet-control-geocoder) | `leaflet-control-geocoder` | Multi-provider (Nominatim, Google, Bing, Mapbox) |
| [Leaflet.geosearch](https://github.com/smeijer/leaflet-geosearch) | `leaflet-geosearch` | Modern, provider-based, framework integrations |

**Routing**

| Plugin | npm | Notes |
|--------|-----|-------|
| [Leaflet Routing Machine](https://github.com/perliedman/leaflet-routing-machine) | `leaflet-routing-machine` | OSRM/GraphHopper/Mapbox routing with turn-by-turn |
| [lrm-graphhopper](https://github.com/perliedman/lrm-graphhopper) | `lrm-graphhopper` | GraphHopper backend for LRM |

**Measurement & Analysis**

| Plugin | npm | Notes |
|--------|-----|-------|
| [Leaflet.measure](https://github.com/ljagis/leaflet-measure) | `leaflet-measure` | Area and distance measurement UI |
| [Leaflet.Terminator](https://github.com/joergdietrich/Leaflet.Terminator) | `@joergdietrich/leaflet.terminator` | Day/night terminator overlay |

**Animation & Advanced Visualization**

| Plugin | npm | Notes |
|--------|-----|-------|
| [Leaflet.Canvas-Markers](https://github.com/eJuke/Leaflet.Canvas-Markers) | — | Canvas-rendered custom markers, faster than SVG |
| [Leaflet.TrackDrawer](https://github.com/adoroszlai/leaflet-track-drawer) | `leaflet-track-drawer` | Animated track/route drawing |
| [leaflet-ant-path](https://github.com/rubenspgcavalcante/leaflet-ant-path) | `leaflet-ant-path` | Animated ant-path polylines |

**Printing & Export**

| Plugin | npm | Notes |
|--------|-----|-------|
| [Leaflet.easyPrint](https://github.com/rowanwins/leaflet-easyPrint) | `leaflet-easyprint` | Print/export to PNG |
| [leaflet-image](https://github.com/mapbox/leaflet-image) | `leaflet-image` | Export map to canvas/image |

### Advanced Patterns & Dark Arts

**Canvas renderer for 50K+ markers**

```javascript
const renderer = L.canvas({ padding: 0.5, tolerance: 5 });

// Custom Canvas CircleMarker subclass for minimal per-feature overhead
const FastPoint = L.CircleMarker.extend({
  options: { renderer, radius: 4, weight: 1, color: '#fff', fillColor: '#e31a1c', fillOpacity: 0.8 }
});

// Batch-add with requestAnimationFrame chunking
function addFeaturesChunked(features, chunkSize = 1000) {
  let i = 0;
  function addChunk() {
    const end = Math.min(i + chunkSize, features.length);
    for (; i < end; i++) {
      const [lng, lat] = features[i].geometry.coordinates;
      new FastPoint([lat, lng]).addTo(map);
    }
    if (i < features.length) requestAnimationFrame(addChunk);
  }
  requestAnimationFrame(addChunk);
}
```

**Custom `L.GridLayer` for non-standard tile sources**

```javascript
// Build a custom grid layer that fetches from a proprietary tile server
const CustomTileLayer = L.GridLayer.extend({
  createTile(coords, done) {
    const img = document.createElement('img');
    const { x, y, z } = coords;
    img.onload = () => done(null, img);
    img.onerror = (e) => done(e);
    img.src = `https://your-tileserver.com/tiles/${z}/${x}/${y}.png?token=${YOUR_TOKEN}`;
    img.crossOrigin = 'anonymous';
    return img;
  }
});
```

**`L.Util.throttle` for expensive layer updates**

```javascript
// Throttle a filter function that re-renders a GeoJSON layer
const updateFilter = L.Util.throttle((value) => {
  geojsonLayer.setStyle(feature => ({
    opacity: feature.properties.value > value ? 1 : 0.1
  }));
}, 100); // max once per 100ms

rangeInput.addEventListener('input', e => updateFilter(+e.target.value));
```

**CSS-based marker animation with `L.DomUtil.setTransform()`**

```javascript
// Smooth marker animation without re-creating DOM elements
function animateMarker(marker, targetLatLng, durationMs = 500) {
  const startLatLng = marker.getLatLng();
  const startTime = performance.now();

  function frame(now) {
    const t = Math.min((now - startTime) / durationMs, 1);
    const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // ease-in-out
    const lat = startLatLng.lat + (targetLatLng.lat - startLatLng.lat) * eased;
    const lng = startLatLng.lng + (targetLatLng.lng - startLatLng.lng) * eased;
    marker.setLatLng([lat, lng]);
    if (t < 1) requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
}
```

**Custom CRS for Chinese Projections (CGCS2000 / EPSG:4547)**

```javascript
import L from 'leaflet';
import proj4 from 'proj4';
import { proj4Leaflet } from 'proj4leaflet';

// CGCS2000 / 3-degree Gauss-Kruger zone 38 (EPSG:4547)
proj4.defs('EPSG:4547',
  '+proj=tmerc +lat_0=0 +lon_0=114 +k=1 +x_0=38500000 +y_0=0 +ellps=GRS80 +units=m +no_defs'
);

const crs = new L.Proj.CRS('EPSG:4547',
  '+proj=tmerc +lat_0=0 +lon_0=114 +k=1 +x_0=38500000 +y_0=0 +ellps=GRS80 +units=m +no_defs',
  {
    resolutions: [256, 128, 64, 32, 16, 8, 4, 2, 1, 0.5],
    origin: [0, 0],
    bounds: L.bounds([0, 0], [38999999, 5000000])
  }
);

const map = L.map('map', { crs });
```

**Chunked GeoJSON loading with progress**

```javascript
async function loadGeoJSONChunked(url, onProgress) {
  const res = await fetch(url);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let loaded = 0;
  const total = +res.headers.get('content-length') || 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    loaded += value.length;
    buffer += decoder.decode(value, { stream: true });
    onProgress(total ? loaded / total : 0);
  }

  return JSON.parse(buffer);
}

loadGeoJSONChunked('/large-dataset.geojson', (pct) => {
  progressBar.style.width = `${pct * 100}%`;
}).then(geojson => {
  L.geoJSON(geojson).addTo(map);
});
```

---

## MapLibre GL JS

Open-source WebGL vector tile renderer, forked from Mapbox GL JS after Mapbox changed its license in December 2020. Governed by the MapLibre Organization (Linux Foundation project). This is the community's preferred open alternative for high-performance vector maps.

### Performance Profile

**Rendering engine**

MapLibre renders everything in WebGL via a tightly optimized pipeline. Tiles are decoded in Web Workers, triangulated, and uploaded to GPU buffers. The main thread only orchestrates; all rasterization is GPU-side.

Key GPU pipeline stages:
1. **Tile decode** (Worker thread): PBF binary → JavaScript feature arrays
2. **Bucket fill** (Worker thread): feature arrays → geometry buffers
3. **GPU upload** (main thread): `gl.bufferData()` → VRAM
4. **Render** (main thread / GPU): draw calls via WebGL

**Feature count limits**

| Source type | Comfortable | 60fps limit | Notes |
|-------------|-------------|-------------|-------|
| GeoJSON (client-side) | 100K features | ~500K simple points | Depends on complexity |
| MVT (vector tiles) | Unlimited | Tile-limited | Only visible tile features loaded |
| Raster tiles | Unlimited | Bandwidth-limited | Pure GPU blit |
| `fill-extrusion` (3D) | 50K buildings | ~200K | GPU-bound |

The MVT approach fundamentally changes the mental model: you only ever render the features visible in the current viewport, so large global datasets remain performant. A 50M-feature planet extract streams fine as vector tiles.

**Memory footprint**

- Empty map + style: ~25 MB heap
- 10K GeoJSON polygons loaded: ~80–120 MB
- 100K point GeoJSON: ~150–250 MB
- MVT source (whole world): ~40–80 MB (only cached tiles kept, LRU eviction)
- GPU VRAM: ~50–200 MB for a typical city view

**FPS benchmarks**

- Raster basemap, standard zoom/pan: 60fps (capped by `requestAnimationFrame`)
- Vector basemap with labels: 60fps (GPU-limited, not CPU)
- GeoJSON with 50K features + data-driven styling: 55–60fps during pan, brief GPU stall on zoom
- `fill-extrusion` buildings, city zoom: 45–60fps (GPU-bound on integrated graphics)
- Pitch + bearing animation (3D view): 60fps on discrete GPU, 30–45fps on integrated

**GPU vs CPU**

| Operation | Where it runs |
|-----------|---------------|
| Tile fetch + decode | Worker threads (CPU, parallel) |
| Geometry triangulation | Worker threads (CPU) |
| Style evaluation (expressions) | Worker threads (CPU) |
| Label placement, collision detection | Worker threads (CPU) |
| All drawing/rendering | GPU via WebGL |
| `queryRenderedFeatures()` | GPU readback → CPU (expensive) |
| `querySourceFeatures()` | CPU (in-memory feature index) |

**Tile loading**

- 8–16 concurrent tile requests (configurable via `config.MAX_PARALLELS_IMAGE_REQUESTS`)
- Tile requests are cancelled when viewport changes (AbortController)
- LRU tile cache: default 200 tiles, configurable via `map.setMaxTileCacheSize()`
- Tiles fetched in priority order: visible center-out
- `transformRequest` callback fires for every network request (auth headers, URL rewriting)

```javascript
const map = new maplibregl.Map({
  // ...
  transformRequest: (url, resourceType) => {
    if (resourceType === 'Tile' && url.includes('private-tiles.example.com')) {
      return {
        url,
        headers: { 'Authorization': `Bearer ${getToken()}` }
      };
    }
  }
});
```

### Development Experience

**TypeScript support**

MapLibre GL JS has first-class TypeScript support — types are bundled in the package (no `@types/` needed since v3). The style specification types (`StyleSpecification`, `LayerSpecification`, etc.) are auto-generated from the spec JSON, providing accurate type-checking on layer paint/layout properties.

```typescript
import maplibregl, { StyleSpecification, GeoJSONSourceSpecification } from 'maplibre-gl';

const style: StyleSpecification = {
  version: 8,
  sources: {
    mydata: {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    } satisfies GeoJSONSourceSpecification
  },
  layers: []
};
```

**Bundle size**

| Asset | Raw | Gzip | Brotli |
|-------|-----|------|--------|
| maplibre-gl.js | ~880 KB | ~230 KB | ~200 KB |
| maplibre-gl.css | ~15 KB | ~4 KB | — |

MapLibre is not tree-shakable — the full WebGL engine must be present. The worker bundle is loaded separately (`maplibre-gl-csp-worker.js`). Using dynamic `import()` defers the ~880 KB parse cost:

```javascript
// Lazy-load the entire map library
let maplibre = null;
async function initMap(container) {
  if (!maplibre) maplibre = await import('maplibre-gl');
  await import('maplibre-gl/dist/maplibre-gl.css');
  return new maplibre.default.Map({ container, /* ... */ });
}
```

**Learning curve**

- Time to first map: 10 minutes
- Understanding style spec (sources/layers model): 2–4 hours
- Expression engine proficiency: 1–2 days
- Custom WebGL layers, protocols, advanced interactivity: 1 week
- Full production app: 1–3 days for experienced devs familiar with the spec

**Debugging**

- [MapLibre GL JS DevTools](https://github.com/maplibre/maplibre-gl-js-devtools) browser extension: inspect layers, sources, paint properties
- `map.showCollisionBoxes = true` — visualize label collision debug boxes
- `map.showTileBoundaries = true` — overlay tile grid
- `map.showPadding = true` — visualize camera padding
- `map.repaint = true` — force continuous repaint (measure FPS)
- `map.getStyle()` in console dumps the full current style JSON

**Documentation quality**

- Official API docs: excellent, auto-generated with TypeDoc
- Style spec reference: best-in-class, very detailed
- Community: growing fast, MapLibre Slack, GitHub Discussions
- Stack Overflow: moderate density (shared with Mapbox GL JS questions, often compatible)

**Breaking changes**

- v2 → v3 (2023): dropped IE11, Web Workers required, some API changes
- v3 → v4 (2024): globe mode, terrain improvements, some camera API changes
- Migration guides published with each major release
- v1 was the direct Mapbox GL JS fork with identical API — most Mapbox GL code migrates with just an import swap

### Plugin & Extension Ecosystem

**Custom Protocols (addProtocol)**

The most powerful MapLibre extension mechanism. Intercept any `scheme://` URL in tile/source requests:

```javascript
// PMTiles protocol — serve a full tile archive from a single file/URL
import { Protocol } from 'pmtiles';

const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile.bind(protocol));

const map = new maplibregl.Map({
  style: {
    version: 8,
    sources: {
      protomaps: {
        type: 'vector',
        url: 'pmtiles://https://example.com/world.pmtiles'
      }
    },
    layers: [/* ... */]
  }
});
```

```javascript
// Custom protocol to decode a proprietary binary format
maplibregl.addProtocol('custom', async (params, abortController) => {
  const url = params.url.replace('custom://', 'https://api.example.com/');
  const res = await fetch(url, { signal: abortController.signal });
  const binary = await res.arrayBuffer();
  // decode to MVT or image
  return { data: decodeToPBF(binary) };
});
```

**Key MapLibre Add-ons**

| Package | Purpose | Notes |
|---------|---------|-------|
| [@maplibre/maplibre-gl-geocoder](https://github.com/maplibre/maplibre-gl-geocoder) | Search/geocoding control | Nominatim, custom providers |
| [@mapbox/mapbox-gl-draw](https://github.com/mapbox/mapbox-gl-draw) | Drawing tools | Works with MapLibre (same API) |
| [terra-draw](https://github.com/JamesLMilner/terra-draw) | Modern drawing, framework-agnostic | Supports MapLibre, Leaflet, OL |
| [maplibre-gl-compare](https://github.com/maplibre/maplibre-gl-compare) | Side-by-side map comparison | Swipe control |
| [maplibre-gl-inspect](https://github.com/acalcutt/maplibre-gl-inspect) | Layer/feature inspector panel | Dev tool |
| [maplibre-contour](https://github.com/onthegomap/maplibre-contour) | Dynamic contour lines | From DEM raster |
| [deck.gl](https://deck.gl) | Advanced WebGL overlays | Integrates as custom layer |
| [pmtiles](https://github.com/protomaps/PMTiles) | Single-file tile archive | Protocol + viewer |

### Advanced Patterns & Dark Arts

**`queryRenderedFeatures()` vs `querySourceFeatures()`**

```javascript
// queryRenderedFeatures: only features CURRENTLY VISIBLE on screen (rendered)
// Use for: hover effects, click detection, what-you-see queries
// Cost: triggers GPU readback — expensive if called every mousemove
map.on('mousemove', 'parks-layer', (e) => {
  // throttle this in production
  const features = map.queryRenderedFeatures(e.point, { layers: ['parks-layer'] });
  if (features.length) showTooltip(features[0].properties);
});

// querySourceFeatures: ALL features in the source (entire dataset, not just visible)
// Use for: statistics, search, total counts, export
// Cost: CPU scan of in-memory feature index — fast but CPU-bound
const allParks = map.querySourceFeatures('parks-source', {
  sourceLayer: 'parks', // required for vector tile sources
  filter: ['>', ['get', 'area'], 10000]
});
console.log(`Total parks > 1ha: ${allParks.length}`);
```

**Expression engine deep dive**

MapLibre expressions are Lisp-like arrays evaluated per-feature on the GPU side. Mastering them unlocks zero-JS data-driven styling.

```javascript
// match: categorical lookup table (fast)
'fill-color': [
  'match', ['get', 'land_use'],
  'residential', '#f4e3c8',
  'commercial',  '#ffd580',
  'industrial',  '#c8d8f4',
  'park',        '#a8d8a8',
  '#cccccc'  // default
]

// interpolate: smooth gradient between stops
'circle-radius': [
  'interpolate', ['exponential', 1.5],
  ['zoom'],
  5,  2,   // at zoom 5: radius 2
  10, 8,   // at zoom 10: radius 8
  15, 20   // at zoom 15: radius 20
]

// step: discrete threshold jumps
'line-width': [
  'step', ['get', 'road_class'],
  1,           // default
  'motorway', 6,
  'trunk', 4,
  'primary', 3
]

// case: if/else logic
'icon-image': [
  'case',
  ['==', ['get', 'type'], 'restaurant'], 'restaurant-icon',
  ['==', ['get', 'type'], 'cafe'],       'cafe-icon',
  ['>', ['get', 'rating'], 4.5],         'star-icon',
  'default-poi-icon'
]

// coalesce: first non-null value
'text-field': [
  'coalesce',
  ['get', 'name_en'],
  ['get', 'name_zh'],
  ['get', 'name'],
  'Unknown'
]

// let/var: named variables for reuse (avoids repeating complex expressions)
'circle-color': [
  'let', 'norm', ['/', ['get', 'value'], ['get', 'max_value']],
  ['interpolate', ['linear'], ['var', 'norm'],
    0, '#2166ac',
    0.5, '#f7f7f7',
    1, '#d73027'
  ]
]
```

**`setFeatureState()` for hover/selection without re-rendering source**

```javascript
// Feature state is maintained by MapLibre, not stored in source data
// Changing feature state triggers re-evaluation of expressions that reference it
// WITHOUT re-uploading geometry to GPU — much faster than removing/re-adding layers

let hoveredId = null;

map.on('mousemove', 'countries', (e) => {
  if (e.features.length > 0) {
    if (hoveredId !== null) {
      map.setFeatureState({ source: 'countries', id: hoveredId }, { hover: false });
    }
    hoveredId = e.features[0].id;
    map.setFeatureState({ source: 'countries', id: hoveredId }, { hover: true });
  }
});

map.on('mouseleave', 'countries', () => {
  if (hoveredId !== null) {
    map.setFeatureState({ source: 'countries', id: hoveredId }, { hover: false });
  }
  hoveredId = null;
});

// In layer paint, reference the feature state:
'fill-color': [
  'case',
  ['boolean', ['feature-state', 'hover'], false],
  '#ff6600',   // hover color
  '#0066cc'    // default color
]

// NOTE: requires 'generateId: true' in GeoJSON source, or integer 'id' in MVT
```

**Custom WebGL layer for particle effects**

```javascript
// Implement a wind particle animation as a custom WebGL layer
const windParticleLayer = {
  id: 'wind-particles',
  type: 'custom',
  renderingMode: '2d',

  onAdd(map, gl) {
    this.map = map;
    // initialize WebGL program, buffers, textures
    const vsSource = `/* vertex shader GLSL */`;
    const fsSource = `/* fragment shader GLSL */`;
    this.program = initShaderProgram(gl, vsSource, fsSource);
    this.buffer = gl.createBuffer();
    // ... setup particles
  },

  render(gl, matrix) {
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uMatrix, false, matrix);
    // ... draw particles
    this.map.triggerRepaint(); // request next frame
  },

  onRemove(map, gl) {
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.buffer);
  }
};

map.on('load', () => map.addLayer(windParticleLayer));
```

**`cooperativeGestures` for scroll hijack prevention**

```javascript
// Prevent the map from capturing scroll events when embedded in a page
// User must use two fingers (touch) or Ctrl+scroll (desktop)
const map = new maplibregl.Map({
  cooperativeGestures: true,  // shows "Use Ctrl + scroll to zoom the map" hint
  // or configure the hint text:
  locale: {
    'ScrollZoomBlocker.CtrlMessage': 'Use Ctrl + scroll to zoom',
    'ScrollZoomBlocker.CmdMessage': 'Use ⌘ + scroll to zoom',
    'TouchPanBlocker.Message': 'Use two fingers to move the map'
  }
});
```

**Style diffing for minimal repaint**

```javascript
// MapLibre's setStyle() has a diff option that only re-renders changed layers
// Instead of replacing the whole style (which causes a full reload):
map.setStyle(newStyle, { diff: true });

// For dynamic re-styling, prefer modifying specific properties:
// These are instant (no GPU re-upload):
map.setPaintProperty('buildings', 'fill-color', '#ff0000');
map.setLayoutProperty('labels', 'visibility', 'none');
map.setFilter('points', ['>', ['get', 'score'], threshold]);

// Adding/removing layers is also efficient:
map.addLayer({ id: 'highlight', type: 'fill', source: 'countries', /* ... */ }, 'labels');
```

---

## OpenLayers

Full-featured enterprise GIS mapping library. Started at MetaCarta in 2006, now maintained by a community under OpenLayers, Inc. The reference implementation for OGC web service standards in JavaScript. No shortcuts taken — full EPSG projection database, every OGC protocol, feature editing with topology, custom renderers.

### Performance Profile

**Rendering engine**

OpenLayers has multiple rendering backends, switchable per-layer:

| Renderer | Use case | Performance |
|----------|----------|-------------|
| Canvas 2D | Default vector layers | Good to 50K features |
| WebGL (ol/layer/WebGLPoints) | Millions of points | GPU-accelerated |
| WebGL (ol/renderer/webgl) | Experimental full WebGL | Very fast |
| Image/Tile | Raster layers | Standard |

```javascript
import WebGLPointsLayer from 'ol/layer/WebGLPoints';
import VectorSource from 'ol/source/Vector';

// Render 1M points with WebGL
const webglLayer = new WebGLPointsLayer({
  source: vectorSource,
  style: {
    'circle-radius': 4,
    'circle-fill-color': ['match', ['get', 'category'],
      'A', '#ff0000',
      'B', '#00ff00',
      '#0000ff'
    ]
  }
});
```

**Feature count limits**

| Renderer | Comfortable | 60fps limit |
|----------|-------------|-------------|
| Canvas 2D (default) | 10K features | ~50K |
| WebGLPoints | 500K+ | ~2M simple points |
| Tile/Image (WMS) | Unlimited | Bandwidth-only |
| VectorTile (MVT) | Unlimited | Tile-limited |

**Memory footprint**

OL stores features in `ol/source/Vector` with full attribute/geometry objects. Higher per-feature overhead than MapLibre (no GPU-side-only storage).

- 10K polygon features: ~50–100 MB
- 100K point features: ~200–400 MB
- WebGL point layer with 1M points: ~500 MB–1 GB (geometry in Float32Array buffers)

**FPS benchmarks**

- Raster tile basemap: 60fps
- Canvas 2D, 10K features: 60fps during pan/zoom
- Canvas 2D, 50K features: 20–40fps (CPU-bound rasterization)
- WebGLPoints, 500K: 60fps on modern GPU
- Feature editing (modify interaction): 60fps for polygons up to ~1K vertices

**GPU vs CPU**

| Operation | Thread |
|-----------|--------|
| WMS/WMTS tile fetch | Network + main |
| Vector source fetch + parse | Main thread (no workers by default) |
| Canvas 2D rendering | Main thread CPU |
| WebGL rendering | GPU |
| Hit detection (forEachFeatureAtPixel) | Main thread CPU |
| Coordinate transforms | Main thread CPU |

**Tile loading**

- Uses browser's native fetch; no custom parallelism control
- WMTS strategy: follows tile matrix set definitions
- Vector tile loading: concurrent by tile; parsed on main thread
- Image tile loading: standard 6-connection-per-host browser limit

### Development Experience

**TypeScript support**

OpenLayers has first-class TypeScript support since OL 7 — types bundled in the package. The API is well-typed but the class hierarchy is deep (100+ classes). Generic types for features and sources reduce casting.

```typescript
import Feature from 'ol/Feature';
import Point from 'ol/geom/Point';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';

const source = new VectorSource<Feature<Point>>({
  features: [new Feature({ geometry: new Point([0, 0]), name: 'Origin' })]
});
```

**Bundle size**

| Import style | Gzip size |
|--------------|-----------|
| Full `import 'ol'` | ~230 KB |
| Selective imports (`ol/Map`, etc.) | ~60–160 KB |
| Just map + OSM tile layer | ~80 KB |

OpenLayers is highly tree-shakable — use selective imports:

```javascript
// Good: selective, tree-shakable
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';

// Bad: imports everything
import * as ol from 'ol';
```

**Learning curve**

- Time to first map: 15 minutes
- Understanding layer/source/view separation: 2 hours
- OGC WMS/WFS integration: 1–2 hours (built-in, well-documented)
- Custom interactions, editing, projections: 1–3 days
- Full production enterprise app: 1–2 weeks

**Debugging**

- No dedicated DevTools extension
- `layer.getSource().getFeatures()` in console
- `map.forEachFeatureAtPixel()` for hit-testing debug
- `map.getView().getCenter()`, `getZoom()`, `getProjection()` for state inspection
- Rendering pipeline is opaque — use browser Performance tab for Canvas 2D profiling

**Documentation quality**

- Official API docs: comprehensive, auto-generated, all parameters documented
- Workshop tutorials: excellent (openlayers.org/workshop)
- Examples: 200+ in the official examples gallery
- Stack Overflow: moderate density, mostly specific OGC questions

**Breaking changes**

- Major version releases happen ~yearly (OL 6, 7, 8, 9, 10)
- Breaking changes are documented in changelogs; common: removed APIs get deprecation warnings first
- OL 6→7: ES modules refactor, projection import changes
- OL 9→10: WebGL renderer changes

### Plugin & Extension Ecosystem

OpenLayers' built-in capabilities reduce plugin dependency significantly:

**What's built-in that other libraries need plugins for:**
- WMS / WMTS / WFS support (built-in sources)
- Arbitrary projection support (EPSG via proj4js)
- Feature editing (Modify, Draw, Snap, Select interactions)
- 30+ format parsers (GeoJSON, KML, GML, GPX, WKT, TopoJSON, MVT, EsriJSON, CSV, IGC, OSM...)
- Heatmap layer (`ol/layer/Heatmap`)
- Graticule (lat/lon grid overlay)
- Cluster source (`ol/source/Cluster`)

**`ol-ext` — 60+ additional extensions**

```bash
npm install ol-ext
```

Covers: profile/elevation charts, legend, print manager, swipe control, animation, flow lines, hex bins, canvasFilter, storymap, D3 integration, and much more.

### Advanced Patterns & Dark Arts

**WMS GetFeatureInfo popup pattern**

```javascript
import TileLayer from 'ol/layer/Tile';
import TileWMS from 'ol/source/TileWMS';
import Overlay from 'ol/Overlay';

const wmsSource = new TileWMS({
  url: 'https://geoserver.example.com/wfs',
  params: { LAYERS: 'regions', TILED: true },
  serverType: 'geoserver',
  crossOrigin: 'anonymous'
});

const popup = new Overlay({
  element: document.getElementById('popup'),
  autoPan: true
});
map.addOverlay(popup);

map.on('singleclick', async (evt) => {
  const viewResolution = map.getView().getResolution();
  const url = wmsSource.getFeatureInfoUrl(
    evt.coordinate,
    viewResolution,
    'EPSG:3857',
    { 'INFO_FORMAT': 'application/json', 'FEATURE_COUNT': 5 }
  );

  if (url) {
    const res = await fetch(url);
    const data = await res.json();
    if (data.features.length > 0) {
      popup.setPosition(evt.coordinate);
      document.getElementById('popup-content').innerHTML =
        formatFeatureTable(data.features[0].properties);
    }
  }
});
```

**Feature editing with snapping and topology**

```javascript
import { Draw, Modify, Snap, Select } from 'ol/interaction';
import { click } from 'ol/events/condition';

// Select interaction
const select = new Select({ condition: click });
map.addInteraction(select);

// Modify selected features
const modify = new Modify({ features: select.getFeatures() });
map.addInteraction(modify);

// Draw new features
const draw = new Draw({
  source: vectorSource,
  type: 'Polygon',
  freehand: false
});
map.addInteraction(draw);

// Snap to existing features (topology preservation)
const snap = new Snap({
  source: vectorSource,
  pixelTolerance: 15
});
map.addInteraction(snap); // Must be added LAST

draw.on('drawend', (evt) => {
  const feature = evt.feature;
  const wktFormat = new WKT();
  console.log('WKT:', wktFormat.writeFeature(feature));
});
```

**WebGL point rendering for millions of features**

```javascript
import WebGLPointsLayer from 'ol/layer/WebGLPoints';
import VectorSource from 'ol/source/Vector';
import GeoJSON from 'ol/format/GeoJSON';

// Load 1M points from GeoJSON
const source = new VectorSource({
  url: '/million-points.geojson',
  format: new GeoJSON()
});

const webglLayer = new WebGLPointsLayer({
  source,
  style: {
    variables: { filterMinScore: 0 },
    filter: ['>=', ['get', 'score'], ['var', 'filterMinScore']],
    'circle-radius': [
      'interpolate', ['linear'], ['get', 'score'],
      0, 3,
      100, 8
    ],
    'circle-fill-color': [
      'interpolate', ['linear'], ['get', 'score'],
      0, '#2166ac',
      50, '#f7f7f7',
      100, '#d73027'
    ],
    'circle-opacity': 0.8
  }
});

// Update WebGL filter variable without re-rendering source
filterSlider.addEventListener('input', (e) => {
  webglLayer.updateStyleVariables({ filterMinScore: +e.target.value });
});
```

**Custom interaction for measurement**

```javascript
import Interaction from 'ol/interaction/Interaction';
import { getLength } from 'ol/sphere';

class MeasureInteraction extends Interaction {
  constructor(options = {}) {
    super(options);
    this.measuring = false;
    this.sketch = null;
  }

  handleEvent(evt) {
    if (evt.type === 'click') {
      // start/continue measurement
    }
    if (evt.type === 'dblclick') {
      this.finishMeasure();
      return false; // prevent zoom
    }
    return true;
  }
}
```

---

## MapTiler SDK

A batteries-included wrapper around MapLibre GL JS maintained by MapTiler AG. Adds: typed `MapStyle` enum for built-in basemaps, geocoding control, geolocation, language utilities, terrain out-of-the-box, and TypeScript-first design. Shares 100% API compatibility with MapLibre.

### Performance Profile

MapTiler SDK has the same WebGL rendering engine as MapLibre GL JS — performance characteristics are identical. The SDK adds ~10 KB gzip overhead for extra utilities. Tile infrastructure is MapTiler's CDN (globally distributed, fast cold-start).

### Development Experience

**TypeScript support**: First-class, bundled types, stricter than raw MapLibre. `MapStyle` is a typed enum, not magic strings.

**Bundle size**: ~240 KB gzip (MapLibre ~230 KB + ~10 KB SDK extras).

**Key DX advantages over raw MapLibre:**

```typescript
import * as maptilersdk from '@maptiler/sdk';
import '@maptiler/sdk/dist/maptiler-sdk.css';

maptilersdk.config.apiKey = 'YOUR_KEY';
maptilersdk.config.primaryLanguage = maptilersdk.Language.CHINESE_SIMPLIFIED;

const map = new maptilersdk.Map({
  container: 'map',
  style: maptilersdk.MapStyle.DATAVIZ.DARK,  // typed enum, no magic strings
  center: [116.4, 39.9],
  zoom: 12,
  terrain: true,  // one-line 3D terrain
  terrainExaggeration: 1.5
});

// Built-in geocoding
const geocoder = new maptilersdk.GeocodingControl({
  apiKey: maptilersdk.config.apiKey,
  country: ['cn'],
  language: ['zh']
});
map.addControl(geocoder, 'top-left');
```

**When to use MapTiler SDK vs raw MapLibre:**

| Use MapTiler SDK | Use raw MapLibre |
|-----------------|-----------------|
| Using MapTiler basemaps | Using your own tile server |
| Need geocoding without setup | Need full vendor freedom |
| Want typed `MapStyle` enum | Need minimal bundle |
| Rapid prototyping | OSS-only dependency policy |

---

## Cross-Platform & Migration

### Leaflet → MapLibre: Detailed Migration Guide

The conceptual shift: Leaflet has a layer-per-thing model (each `L.Polygon` is a layer). MapLibre has a source+layer model (a GeoJSON source feeds multiple style layers). This changes how you think about adding data.

**Map initialization**

```javascript
// BEFORE: Leaflet
const map = L.map('map', { center: [39.9, 116.4], zoom: 10 });
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);

// AFTER: MapLibre
const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      osm: { type: 'raster', tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'], tileSize: 256 }
    },
    layers: [{ id: 'osm', type: 'raster', source: 'osm' }]
  },
  center: [116.4, 39.9],  // NOTE: [lng, lat] — longitude first in MapLibre
  zoom: 10
});
```

**Coordinate order warning**: Leaflet uses `[lat, lng]`. MapLibre uses `[lng, lat]` (GeoJSON order). This is the #1 migration bug.

**Markers & Popups**

```javascript
// BEFORE: Leaflet
L.marker([39.9, 116.4])
  .bindPopup('<b>Beijing</b><p>Population: 21M</p>')
  .addTo(map);

// AFTER: MapLibre (HTML marker)
new maplibregl.Marker()
  .setLngLat([116.4, 39.9])  // lng, lat order
  .setPopup(new maplibregl.Popup().setHTML('<b>Beijing</b><p>Population: 21M</p>'))
  .addTo(map);

// AFTER: MapLibre (GeoJSON + symbol layer — better for many markers)
map.on('load', () => {
  map.addSource('cities', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [{
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [116.4, 39.9] },
      properties: { name: 'Beijing', population: 21000000 }
    }]}
  });
  map.addLayer({ id: 'city-markers', type: 'circle', source: 'cities', paint: { 'circle-radius': 8 } });
  map.addLayer({ id: 'city-labels', type: 'symbol', source: 'cities', layout: { 'text-field': ['get', 'name'] } });

  map.on('click', 'city-markers', (e) => {
    new maplibregl.Popup()
      .setLngLat(e.lngLat)
      .setHTML(`<b>${e.features[0].properties.name}</b>`)
      .addTo(map);
  });
});
```

**GeoJSON layers**

```javascript
// BEFORE: Leaflet
const layer = L.geoJSON(data, {
  style: feature => ({ color: colorScale(feature.properties.value), weight: 2 }),
  onEachFeature: (feature, layer) => layer.bindPopup(feature.properties.name)
}).addTo(map);

// AFTER: MapLibre
map.on('load', () => {
  map.addSource('data', { type: 'geojson', data });
  map.addLayer({
    id: 'data-fill',
    type: 'fill',
    source: 'data',
    paint: {
      'fill-color': ['interpolate', ['linear'], ['get', 'value'], 0, '#f7fbff', 100, '#08306b'],
      'fill-opacity': 0.7
    }
  });
  map.addLayer({
    id: 'data-outline',
    type: 'line',
    source: 'data',
    paint: { 'line-color': '#ffffff', 'line-width': 2 }
  });
  // Click popup via event listener on layer id
  map.on('click', 'data-fill', (e) => {
    new maplibregl.Popup()
      .setLngLat(e.lngLat)
      .setHTML(e.features[0].properties.name)
      .addTo(map);
  });
});
```

**Event handling**

```javascript
// BEFORE: Leaflet
map.on('click', (e) => console.log(e.latlng));
layer.on('mouseover', (e) => highlight(e.target));

// AFTER: MapLibre
map.on('click', (e) => console.log(e.lngLat));
map.on('mousemove', 'layer-id', (e) => highlight(e.features[0]));
map.on('mouseleave', 'layer-id', () => clearHighlight());
```

**Layer control (show/hide)**

```javascript
// BEFORE: Leaflet
map.addLayer(overlay);
map.removeLayer(overlay);

// AFTER: MapLibre (prefer visibility toggle, avoids re-adding)
map.setLayoutProperty('layer-id', 'visibility', 'visible');
map.setLayoutProperty('layer-id', 'visibility', 'none');
```

### Mapbox GL JS → MapLibre: Exact Migration Steps

MapLibre was forked from Mapbox GL JS v1.13. The fork point is the cleanest migration:

```javascript
// BEFORE: Mapbox GL JS
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
mapboxgl.accessToken = 'pk.eyJ1...';
const map = new mapboxgl.Map({ style: 'mapbox://styles/mapbox/streets-v12' });

// AFTER: MapLibre GL JS (mostly a find-and-replace)
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
// No access token needed
const map = new maplibregl.Map({ style: 'https://demotiles.maplibre.org/style.json' });
```

**What breaks / changes:**

| Feature | Mapbox GL JS | MapLibre |
|---------|-------------|----------|
| Style URLs | `mapbox://styles/...` | Must use full HTTPS URL or own style JSON |
| Sprite URLs | `mapbox://sprites/...` | Must host own sprites |
| Access token | Required | Not required |
| Mapbox Terrain | `mapbox://mapbox.terrain-rgb` | Use MapTiler/Maptbox terrain alternatives |
| Globe mode | v2+ | Available in MapLibre v4+ |
| Sky layer | v2+ | Available in MapLibre |
| `mapbox-gl-draw` | Official plugin | Community-maintained, works with MapLibre |

**Mapbox v2+ features that needed workarounds** in early MapLibre forks are now mostly available natively: globe projection, fog, sky layer, terrain exaggeration, cooperative gestures, advanced camera animations.

### OpenLayers → MapLibre: When and What You Lose

**Migrate when:**
- You need vector tiles (not just WMS) and performance for 50K+ features
- You need custom map styling (data-driven colors, thematic maps)
- Your WMS dependency is gone or replaceable

**What you lose moving to MapLibre:**
- Built-in arbitrary projection support (MapLibre only does EPSG:3857/4326)
- Native WMS/WMTS/WFS sources (MapLibre has raster source for WMS, but no GetFeatureInfo)
- Built-in format parsers (KML, GML, GPX — need external libraries)
- Built-in feature editing (need mapbox-gl-draw or terra-draw)
- Robust feature topology tools

```javascript
// OpenLayers WMS → MapLibre raster source (limited: no GetFeatureInfo)
// BEFORE: OpenLayers
new TileLayer({ source: new TileWMS({ url: '...', params: { LAYERS: 'regions' } }) });

// AFTER: MapLibre (raster WMS, no click queries)
map.addSource('wms', {
  type: 'raster',
  tiles: ['https://geoserver.example.com/wfs?SERVICE=WMS&REQUEST=GetMap&LAYERS=regions&' +
          'BBOX={bbox-epsg-3857}&WIDTH=256&HEIGHT=256&FORMAT=image/png&SRS=EPSG:3857'],
  tileSize: 256
});
```

### Framework Wrapper Comparison

| Library | Leaflet | MapLibre | OpenLayers | Notes |
|---------|---------|----------|------------|-------|
| React | `react-leaflet` v4 | `react-map-gl` v7 / `react-maplibre` | `ol-react` | react-map-gl now officially supports MapLibre |
| Vue | `vue2-leaflet` / `vue-leaflet` | `vue-maplibre-gl` | — | Vue 3 wrappers available |
| Svelte | `svelte-leaflet` | `svelte-maplibre` | — | Svelte wrappers are thin |
| Angular | `ngx-leaflet` | — (raw service) | — | OL is common in Angular enterprise |
| Solid | — | `solid-map-gl` | — | Port of react-map-gl |

```bash
# React + MapLibre (preferred stack 2024+)
npm install react-map-gl maplibre-gl

# React + Leaflet
npm install react-leaflet leaflet @types/leaflet
```

```jsx
// react-map-gl with MapLibre
import Map from 'react-map-gl/maplibre';

export function MyMap() {
  return (
    <Map
      initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 10 }}
      style={{ width: '100%', height: '400px' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    >
      <Marker longitude={116.4} latitude={39.9} anchor="bottom">
        <img src="/pin.png" />
      </Marker>
    </Map>
  );
}
```

---

## Comprehensive Comparison Matrix

| Dimension | Leaflet | MapLibre GL JS | OpenLayers | MapTiler SDK |
|-----------|---------|----------------|------------|--------------|
| **Rendering** | Canvas 2D / SVG / DOM | WebGL | Canvas 2D / WebGL | WebGL (same as MapLibre) |
| **Vector tiles (MVT)** | Plugin (vectorgrid) | Native | Native | Native |
| **Raster tiles** | Native | Native | Native | Native |
| **WMS** | Basic (raster) | Raster only, no GFI | Full (tile, image, cascade) | Raster only |
| **WFS** | Via omnivore plugin | No | Full (bbox strategy, etc.) | No |
| **WMTS** | Plugin | Raster source | Full native | Raster source |
| **3D terrain** | No | Yes (v1.7+) | Experimental | Yes (easy, 1 line) |
| **Building extrusion** | No | Yes (fill-extrusion) | No | Yes |
| **Arbitrary projections** | Plugin (proj4leaflet) | No (3857/4326 only) | Yes, proj4js built-in | No |
| **Feature editing** | Plugin (geoman, draw) | Plugin (gl-draw) | Built-in interactions | Plugin (gl-draw) |
| **Touch / mobile** | Excellent | Very good | Good | Very good |
| **Accessibility (a11y)** | Basic | Basic | Partial | Basic |
| **RTL text** | No (plugin needed) | Plugin (`maplibre-gl-rtl-text`) | No | Included |
| **Offline / PWA** | Service worker caching | PMTiles protocol | Limited | PMTiles via MapLibre |
| **SSR (server-side)** | Partial (jsdom) | No (WebGL required) | Partial | No |
| **TypeScript** | @types/leaflet | Bundled native | Bundled native | Bundled native |
| **Tree-shakable** | No | No | Yes | No |
| **Gzip size** | ~40 KB | ~230 KB | ~160 KB (selective) | ~240 KB |
| **Community size** | Very large (~42K stars) | Large, fast-growing (~10K stars) | Large (~11K stars) | Small (MapLibre base) |
| **Corporate backing** | Community + Leaflet Inc. | MapLibre Org (Linux Foundation) | OpenLayers Inc. | MapTiler AG |
| **License** | BSD-2 | BSD-2 | BSD-2 | BSD-2 |
| **Breaking changes** | Rare (stable since 2016) | Annual major releases | Annual major releases | Follows MapLibre |
| **Plugin ecosystem** | 700+ | Growing (~50 quality) | ~60 (ol-ext) + built-ins | Same as MapLibre |
| **Data-driven styling** | Manual JS callbacks | Expression engine (GPU) | Style functions | Expression engine (GPU) |
| **Custom layers** | L.GridLayer / L.Layer | Custom WebGL layer | Custom renderer | Custom WebGL layer |
| **Tile request cancel** | No | Yes (AbortController) | Partial | Yes |
| **Label collision** | No | Yes (GPU) | Yes (Canvas) | Yes (GPU) |

---

## Bundle Size Optimization

### Tree-shaking Guide

**OpenLayers** (best tree-shaking)

```javascript
// Import only what you use — each class is its own module
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import GeoJSON from 'ol/format/GeoJSON';
// Result: ~80–120 KB gzip depending on features used
```

**MapLibre GL JS** (not tree-shakable — but lazy-loadable)

```javascript
// vite.config.js — ensure MapLibre stays in its own chunk
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'maplibre': ['maplibre-gl']
        }
      }
    }
  }
};
```

**Leaflet** (not tree-shakable, but small by default)

```javascript
// Avoid importing plugins at app start — dynamic import them
async function addClusteringSupport(map) {
  const { default: L } = await import('leaflet');
  await import('leaflet.markercluster');
  await import('leaflet.markercluster/dist/MarkerCluster.css');
  // now use L.markerClusterGroup()
}
```

### Lazy Loading Strategies

```javascript
// Pattern 1: Defer map initialization until user interaction
let mapModule = null;

document.getElementById('show-map-btn').addEventListener('click', async () => {
  if (!mapModule) {
    // Show loading indicator
    document.getElementById('map-loading').hidden = false;
    // Dynamic import — downloads ~230 KB only when needed
    mapModule = await import('maplibre-gl');
    await import('maplibre-gl/dist/maplibre-gl.css');
    document.getElementById('map-loading').hidden = true;
  }
  initializeMap(mapModule.default);
});

// Pattern 2: IntersectionObserver — load map when it enters viewport
const observer = new IntersectionObserver(async ([entry]) => {
  if (entry.isIntersecting) {
    observer.disconnect();
    const maplibre = await import('maplibre-gl');
    // initialize map
  }
}, { rootMargin: '200px' });

observer.observe(document.getElementById('map'));
```

### Code-Splitting with Vite

```javascript
// Dedicated map page/route — map library only loads on /map route
// React Router + Vite
const MapPage = lazy(() => import('./pages/MapPage'));  // dynamic import

// Inside MapPage.tsx — import is deferred until component renders
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
```

```javascript
// Webpack — explicit split point
// webpack.config.js
module.exports = {
  optimization: {
    splitChunks: {
      cacheGroups: {
        maplibre: {
          test: /[\\/]node_modules[\\/]maplibre-gl[\\/]/,
          name: 'vendor-maplibre',
          chunks: 'async',  // only in async chunks
          priority: 10
        }
      }
    }
  }
};
```

### Removing Unused Locales and Assets

```javascript
// Leaflet: the CSS includes icon URLs — override to avoid 404s in webpack
// webpack/vite projects often need this:
import L from 'leaflet';
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow
});
```

---

## Decision Matrix by Use Case

| Use Case | Recommended | Runner-up | Reason | Migration path if needs grow |
|----------|-------------|-----------|--------|------------------------------|
| Quick prototype / proof of concept | Leaflet | MapTiler SDK | Fastest to first result, simple API | → MapLibre when you need custom styling |
| Production app with MapTiler basemaps | MapTiler SDK | MapLibre | Pre-configured, typed styles, geocoding included | Drop to MapLibre if you switch tile provider |
| Custom vector tile basemap with full styling control | MapLibre GL JS | MapTiler SDK | Open-source, style spec, no vendor lock-in | Add MapTiler SDK wrapper if you need extras |
| Enterprise OGC integration (WMS, WFS, WMTS) | OpenLayers | Leaflet + plugins | Built-in WMS/WFS/WMTS, no plugins needed | — (no better option for deep OGC) |
| Chinese government data (CGCS2000, GCJ02, BD09) | OpenLayers | Leaflet + proj4leaflet | Full proj4 support, custom CRS trivial | — |
| 100K+ point features, interactive | MapLibre GL JS | OL WebGLPoints | WebGL rendering, consistent 60fps | — |
| 1M+ point features, visualization | OpenLayers (WebGLPoints) | MapLibre + clustering | ol/layer/WebGLPoints handles millions | → MapLibre if you also need vector basemap |
| Offline PWA, no tile server | MapLibre + PMTiles | Leaflet + Service Worker | PMTiles = single-file tile archive, no server | — |
| Real-time moving features (vehicles, ships) | MapLibre (`setFeatureState`) | Leaflet Canvas | GPU state updates, no re-render | — |
| Feature editing (digitizing, field data collection) | OpenLayers | Leaflet + Geoman | Built-in snap/modify/draw interactions | → QGIS if desktop editing needed |
| 3D terrain + building extrusion | MapLibre GL JS | MapTiler SDK | fill-extrusion, terrain-rgb, sky layer | — |
| Mobile web, low-bandwidth | Leaflet | MapLibre | 40 KB gzip, SVG renders without WebGL | → MapLibre when performance needed |
| Government/intranet, no CDN | OpenLayers | Leaflet | Both fully self-hostable, no external deps | — |
| Server-side rendering (Next.js SSR) | Leaflet (partial) | OpenLayers | No WebGL requirement; can render to canvas server-side | Add client-side hydration |
| Integrating with deck.gl overlays | MapLibre GL JS | — | deck.gl MapboxLayer works with MapLibre | — |
| Streaming real-time tile updates | MapLibre (source.setData) | OL Vector | Efficient JSON diff, no full reload | — |
| Map comparison (before/after swipe) | MapLibre + compare plugin | Leaflet + sync plugin | maplibre-gl-compare, smooth WebGL compositing | — |

---

## Common Gotchas & Production Notes

**Leaflet**
- Always set `height` on the map container element (not just percentage — it needs a pixel reference)
- `map.invalidateSize()` required after the container becomes visible (e.g., inside a tab or modal)
- Marker icons break in webpack/vite without manual path override (see bundle section above)
- Z-index of map controls defaults to 1000 — can clash with modal libraries

**MapLibre GL JS**
- `map.on('load')` fires when style is loaded; `map.on('idle')` fires when tiles finish loading
- GeoJSON sources require `generateId: true` for `setFeatureState()` to work
- Never mutate the data object passed to `addSource` — call `source.setData(newData)` instead
- `map.remove()` must be called on component unmount to free WebGL context (browsers limit contexts)
- WebGL context loss (GPU reset, tab backgrounding) fires `webglcontextlost` — handle recovery
- `transformRequest` runs for every resource (tiles, sprites, glyphs) — keep it synchronous and fast

**OpenLayers**
- Coordinate order is `[x, y]` = `[longitude, latitude]` — same as GeoJSON, opposite of Leaflet
- `fromLonLat([116.4, 39.9])` converts to the map projection (EPSG:3857 by default)
- WMS bbox parameter format differs between WMS 1.1 and 1.3 (axis order swap for EPSG:4326)
- `ol/proj/proj4` must be registered before using custom EPSG codes
- Feature IDs in OL are separate from properties — use `feature.getId()` not `feature.get('id')`

**MapTiler SDK**
- API key is required and must be valid for tile requests to succeed (unlike raw MapLibre)
- Free tier has usage limits — monitor your dashboard for production apps
- `maptilersdk.config.apiKey` is global — in multi-map apps, set it once before any map creation
- MapTiler SDK version is tied to a specific MapLibre version — check compatibility before upgrading
