# 2D Mapping Libraries

> Core JavaScript libraries for rendering interactive 2D web maps.

> **Quick Picks**
> - **SOTA**: [MapLibre GL JS](https://maplibre.org) -- open-source vector tiles with WebGL, massive community momentum
> - **Free Best**: [Leaflet](https://leafletjs.com) -- zero cost, 700+ plugins, dead-simple API
> - **Fastest Setup**: [MapTiler SDK](https://docs.maptiler.com/sdk-js/) -- pre-configured MapLibre wrapper with built-in basemaps

## Leaflet

Lightweight, mobile-friendly interactive mapping library with a massive plugin ecosystem.

- **Website:** [leafletjs.com](https://leafletjs.com)
- **Key Strengths:** Simple API, extensive plugins (700+), excellent documentation
- **Best For:** Quick prototypes, standard web maps, mobile-first projects

### Install

```bash
npm install leaflet
```

**CDN:**
```html
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9/dist/leaflet.css" />
<script src="https://unpkg.com/leaflet@1.9/dist/leaflet.js"></script>
```

### Quick Start

```javascript
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';

const map = L.map('map').setView([39.9, 116.4], 10);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

L.marker([39.9, 116.4]).addTo(map).bindPopup('Beijing');
```

### Top Plugins

| Plugin | Install | Purpose |
|--------|---------|---------|
| [Leaflet.markercluster](https://github.com/Leaflet/Leaflet.markercluster) | `npm i leaflet.markercluster` | Animated marker clustering for dense point data |
| [Leaflet.heat](https://github.com/Leaflet/Leaflet.heat) | `npm i leaflet.heat` | Lightweight Canvas-based heatmap layer |
| [Leaflet.draw](https://github.com/Leaflet/Leaflet.draw) | `npm i leaflet-draw` | Drawing and editing vectors (polygons, circles, rectangles) |
| [Leaflet-Geoman](https://github.com/geoman-io/leaflet-geoman) | `npm i @geoman-io/leaflet-geoman-free` | Modern alternative to Leaflet.draw with snapping |
| [leaflet-omnivore](https://github.com/mapbox/leaflet-omnivore) | `npm i @mapbox/leaflet-omnivore` | Load CSV, KML, GPX, TopoJSON, WKT |

```javascript
// Marker clustering example
import 'leaflet.markercluster';
import 'leaflet.markercluster/dist/MarkerCluster.css';

const markers = L.markerClusterGroup();
data.features.forEach(f => {
  const [lng, lat] = f.geometry.coordinates;
  markers.addLayer(L.marker([lat, lng]));
});
map.addLayer(markers);
```

```javascript
// Heatmap example
import 'leaflet.heat';

const heat = L.heatLayer(
  points.map(p => [p.lat, p.lng, p.intensity]),
  { radius: 25, blur: 15, maxZoom: 17 }
).addTo(map);
```

---

## MapLibre GL JS

Open-source fork of Mapbox GL JS for rendering vector tile maps with WebGL.

- **Website:** [maplibre.org](https://maplibre.org)
- **Key Strengths:** Vector tiles, smooth zooming/rotation, style spec, WebGL rendering
- **Best For:** Vector tile maps, custom styling, high-performance rendering

### Install

```bash
npm install maplibre-gl
```

**CDN:**
```html
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl/dist/maplibre-gl.css" />
<script src="https://unpkg.com/maplibre-gl/dist/maplibre-gl.js"></script>
```

### Quick Start

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

### Data-Driven Styling

```javascript
// Choropleth via data-driven fill color
map.on('load', () => {
  map.addSource('population', {
    type: 'geojson',
    data: '/data/regions.geojson'
  });

  map.addLayer({
    id: 'population-fill',
    type: 'fill',
    source: 'population',
    paint: {
      'fill-color': [
        'interpolate', ['linear'],
        ['get', 'density'],
        0,    '#f7fbff',
        50,   '#6baed6',
        200,  '#2171b5',
        1000, '#08306b'
      ],
      'fill-opacity': 0.75
    }
  });

  // Proportional circles from a property
  map.addLayer({
    id: 'city-circles',
    type: 'circle',
    source: 'cities',
    paint: {
      'circle-radius': [
        'interpolate', ['linear'],
        ['get', 'population'],
        10000,  4,
        100000, 12,
        1000000, 24
      ],
      'circle-color': '#e31a1c',
      'circle-opacity': 0.7
    }
  });
});
```

---

## MapTiler SDK

A batteries-included wrapper around MapLibre GL JS with pre-configured basemaps, geocoding, and utilities.

- **Website:** [docs.maptiler.com/sdk-js](https://docs.maptiler.com/sdk-js/)
- **Key Strengths:** Built on MapLibre, free tier basemaps, geocoding/geolocation included, TypeScript-first
- **Best For:** Fastest path from zero to styled vector map

### Install

```bash
npm install @maptiler/sdk
```

### Quick Start

```javascript
import * as maptilersdk from '@maptiler/sdk';
import '@maptiler/sdk/dist/maptiler-sdk.css';

maptilersdk.config.apiKey = 'YOUR_MAPTILER_KEY';

const map = new maptilersdk.Map({
  container: 'map',
  style: maptilersdk.MapStyle.STREETS,
  center: [116.4, 39.9],
  zoom: 10
});
```

> MapTiler SDK is fully compatible with the MapLibre GL JS API -- any MapLibre plugin or code pattern works directly.

---

## OpenLayers

Full-featured, enterprise-grade mapping library with comprehensive format and projection support.

- **Website:** [openlayers.org](https://openlayers.org)
- **Key Strengths:** WMS/WFS/WMTS support, projection handling, feature-rich API
- **Best For:** Enterprise GIS, OGC services integration, complex cartographic requirements

### Install

```bash
npm install ol
```

### Quick Start

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

### OGC Service Integration

```javascript
import TileWMS from 'ol/source/TileWMS';

// WMS layer
const wmsLayer = new TileLayer({
  source: new TileWMS({
    url: 'https://example.com/geoserver/wms',
    params: { LAYERS: 'topp:states', TILED: true },
    serverType: 'geoserver'
  })
});
map.addLayer(wmsLayer);
```

```javascript
import VectorSource from 'ol/source/Vector';
import GeoJSON from 'ol/format/GeoJSON';
import { bbox as bboxStrategy } from 'ol/loadingstrategy';

// WFS vector source
const wfsSource = new VectorSource({
  format: new GeoJSON(),
  url: (extent) => {
    return `https://example.com/geoserver/wfs?service=WFS&version=2.0.0` +
      `&request=GetFeature&typeName=topp:states&outputFormat=application/json` +
      `&srsname=EPSG:3857&bbox=${extent.join(',')},EPSG:3857`;
  },
  strategy: bboxStrategy
});
```

---

## Comparison Table

| Library | Size (gzip) | Rendering | Vector Tiles | Learning Curve | Best For |
|---------|-------------|-----------|--------------|----------------|----------|
| Leaflet | ~40 KB | Canvas/SVG | Via plugins | Low | Simple maps, prototyping |
| MapLibre GL JS | ~220 KB | WebGL | Native | Medium | Vector maps, custom styles |
| MapTiler SDK | ~230 KB | WebGL | Native | Low | Quick vector maps |
| OpenLayers | ~160 KB | Canvas/WebGL | Native | High | Enterprise, OGC services |

### Rendering Performance Notes

- **Leaflet** handles ~10k features comfortably in SVG mode; use Canvas renderer (`preferCanvas: true`) for larger datasets.
- **MapLibre GL JS** renders 100k+ features via vector tiles with consistent 60fps on modern GPUs.
- **OpenLayers** WebGL renderer (via `ol/layer/WebGLPoints`) can handle millions of points.

---

## Decision Flowchart

```
Start
  |
  +-- Need vector tiles natively?
  |     +-- Yes --> MapLibre GL JS (or MapTiler SDK for faster setup)
  |     +-- No  |
  |             v
  +-- Need OGC services (WMS/WFS/WMTS)?
  |     +-- Yes --> OpenLayers
  |     +-- No  |
  |             v
  +-- Need simplicity & fast setup?
  |     +-- Yes --> Leaflet
  |     +-- No  |
  |             v
  +-- Need everything (projections, formats, editing)?
  |     +-- Yes --> OpenLayers
  |
  +-- Still unsure? --> Start with Leaflet, migrate as needs grow
```

## Migration Tips

| From | To | Key Changes |
|------|----|-------------|
| Leaflet -> MapLibre | Replace `L.tileLayer` with style JSON; markers become GeoJSON sources + symbol layers |
| Leaflet -> OpenLayers | Replace `L.map` with `new Map()`; coordinate order is `[x, y]` (lon, lat) in OL |
| Mapbox GL JS -> MapLibre | Drop-in replacement -- change the import from `mapbox-gl` to `maplibre-gl` and remove access token |
