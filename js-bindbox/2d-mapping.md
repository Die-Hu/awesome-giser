# 2D Mapping Libraries

> Data validated: 2026-03-21

## 30-Second Decision

| I need... | Use this | Startup time |
|-----------|----------|-------------|
| Fastest map on screen, <5K features | Leaflet | 5 min |
| Free basemaps, zero config | MapTiler SDK | 10 min |
| Vector tiles, full control, no vendor lock-in | MapLibre GL JS | 15 min |
| OGC (WMS/WFS), custom projections | OpenLayers | 1-2 hours |

---

## Detailed Guide (by startup time)

### 1. Leaflet

The fastest path to a working map. 40KB gzipped, CDN-ready, 5 lines of code.

**Quick facts:** ~40KB gzip | Easy learning curve | ~44K GitHub stars | ~2.8M npm/week

```javascript
const map = L.map('map').setView([39.9, 116.4], 10);
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap'
}).addTo(map);
```

**Performance limits:**

| Renderer | 1K | 5K | 10K | 50K |
|----------|-----|-----|------|------|
| SVG (default) | 60fps | 40fps | 15fps | crash |
| Canvas | 60fps | 60fps | 50fps | 20fps |

Activate Canvas renderer for better perf: `L.map('map', { renderer: L.canvas() })`.

**Small project strengths:**
- Smallest bundle of any full mapping library (40KB gzip)
- CDN usage = zero build step, just `<script>` tags
- 700+ plugins (markers, geocoding, routing, drawing)
- Works on every browser, no WebGL needed

**Key caveats:**
- DOM-based rendering caps at ~5K interactive features -- architectural ceiling
- No native vector tile support; plugins for this are poorly maintained
- Plugin quality is wildly inconsistent; many abandoned 3+ years ago
- TypeScript support has gaps (`@types/leaflet`)
- EPSG:3857 only by default (custom CRS via proj4leaflet is poorly documented)
- **When to migrate:** If you need >5K features or vector tiles, move to MapLibre GL JS

---

### 2. MapTiler SDK

Pre-configured MapLibre GL JS with free basemaps, geocoding, and sensible defaults. One API key = production-looking map.

```javascript
import * as maptilersdk from '@maptiler/sdk';
maptilersdk.config.apiKey = 'YOUR_API_KEY';
const map = new maptilersdk.Map({
  container: 'map',
  style: maptilersdk.MapStyle.STREETS,
  center: [116.4, 39.9],
  zoom: 10
});
```

**Small project strengths:**
- Beautiful basemaps with zero effort (streets, satellite, topo, dark)
- Built-in geocoding search bar
- Same MapLibre power underneath

**Key caveats:**
- Free tier = 100K tile requests/month; each page load = 20-40 requests
- Vendor coupling to MapTiler API/pricing
- For enterprise: use raw MapLibre GL JS + self-hosted tiles

---

### 3. MapLibre GL JS

The production standard for modern web mapping. Open-source (BSD), WebGL rendering, vector tiles, no vendor lock-in.

**Quick facts:** ~200KB gzip | Medium learning curve | ~10K GitHub stars | ~2.1M npm/week | Used by AWS, Meta, Microsoft

```javascript
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [116.4, 39.9],
  zoom: 10
});
```

```javascript
// PMTiles integration (zero-server tile hosting)
import { Protocol } from 'pmtiles';
maplibregl.addProtocol('pmtiles', new Protocol().tile);
```

**Performance limits:**

| Data Type | 1K | 10K | 100K | 1M |
|-----------|-----|------|-------|-----|
| Points | 60fps | 60fps | 60fps | 50fps* |
| Lines | 60fps | 60fps | 55fps | 40fps* |
| Polygons | 60fps | 60fps | 50fps | 35fps* |

*via vector tiles, not raw GeoJSON

**Small project strengths:**
- Scales from prototype to production without rewriting
- PMTiles = host tiles on any CDN, zero server infra
- First-class React/Vue/Svelte wrappers

**Key caveats:**
- EPSG:3857 only -- cannot render in arbitrary projections. Wrong choice for CGCS2000/GCJ02
- WebGL required, no fallback
- Style spec complexity: expression syntax is not intuitive, typos cause silent failures
- WebGL context limit (8-16 per page) -- multiple map instances cause black screens
- **Anti-pattern:** Loading raw GeoJSON for large datasets. Use vector tiles for >50K features.
- **Anti-pattern:** Not setting `maxzoom` on PMTiles sources -- causes 404s

---

### 4. OpenLayers

Built-in support for WMS, WFS, WMTS, GML, custom projections, reprojection, and every OGC standard. Use when you need OGC compliance or non-Mercator projections.

**Quick facts:** 150-300KB gzip | Hard learning curve | ~12K GitHub stars | 20+ years of development

```javascript
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';

const map = new Map({
  target: 'map',
  layers: [new TileLayer({ source: new OSM() })],
  view: new View({ center: [0, 0], zoom: 2 })
});
```

**Small project strengths:**
- Only choice for custom projections or OGC services natively, without plugins
- WebGL renderer handles large datasets

**Key caveats:**
- Usually overkill for small projects unless you specifically need custom CRS or WMS/WFS
- Massive API surface: Map -> View -> Layer -> Source -> Feature -> Geometry -> Style
- Documentation is reference-heavy, tutorial-light
- Fewer community resources than Leaflet or MapLibre
- **Anti-pattern:** Using OpenLayers when you don't need OGC/custom projections. MapLibre does the same in 1/3 the code.

---

## Decision Flowchart

```
Do you need custom projections or OGC (WMS/WFS)?
+-- YES -> OpenLayers
+-- NO -> Do you need >5K interactive features?
    +-- YES -> MapLibre GL JS
    +-- NO -> Do you want beautiful basemaps with zero effort?
        +-- YES -> MapTiler SDK (watch the pricing)
        +-- NO -> Leaflet (simplest, fastest, smallest)
```

---

## Head-to-Head Comparison

| Criterion | Leaflet | MapTiler SDK | MapLibre GL JS | OpenLayers |
|-----------|---------|-------------|----------------|------------|
| Bundle size (gzip) | **40KB** | ~220KB | ~200KB | 150-300KB |
| Time to first map | **5 min** | 10 min | 15 min | 1-2 hours |
| Max features (interactive) | ~5K | Same as MapLibre | **100K+** | ~50K (WebGL) |
| Vector tile support | Plugin (flaky) | Native | **Native** | Native |
| Custom projections | Plugin | No | **No** | **Native** |
| OGC support | No | No | No | **Full** |
| Learning curve | **Easy** | Easy | Medium | Hard |
| Vendor lock-in | None | MapTiler | **None** | None |
| Mobile performance | Good | Great | **Great** | Good |
