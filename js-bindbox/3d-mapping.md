# 3D Mapping & Globe

> JavaScript libraries for 3D geospatial visualization, globe rendering, and immersive mapping experiences.

> **Quick Picks**
> - **SOTA**: [CesiumJS](https://cesium.com) -- 3D Tiles 1.1, Google Photorealistic Tiles, full globe with time-dynamic data
> - **Free Best**: [deck.gl](https://deck.gl) -- GPU-accelerated layers for millions of points, open source
> - **Fastest Setup**: [Globe.gl](https://globe.gl) -- three lines of code to an interactive 3D globe

## CesiumJS

The leading open-source library for 3D globes and maps, with native support for 3D Tiles, terrain, and geospatial data formats.

- **Website:** [cesium.com](https://cesium.com)
- **Key Strengths:** True 3D globe, 3D Tiles 1.1, terrain rendering, time-dynamic data
- **Best For:** Digital twins, flight simulation, urban planning, satellite imagery

### Install

```bash
npm install cesium
```

### Quick Start

```javascript
import { Viewer, Ion, createWorldTerrainAsync } from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

// Set your Cesium Ion access token
Ion.defaultAccessToken = 'YOUR_CESIUM_ION_TOKEN';

const viewer = new Viewer('cesiumContainer', {
  terrain: await createWorldTerrainAsync()
});

// Fly to a location
viewer.camera.flyTo({
  destination: Cesium.Cartesian3.fromDegrees(116.4, 39.9, 15000)
});
```

### Cesium Ion & 3D Tiles 1.1

[Cesium Ion](https://ion.cesium.com/) is the cloud platform for hosting, tiling, and streaming 3D content:

- **3D Tiles Next (1.1):** Adds glTF content (replaces b3dm/i3dm), metadata per-feature/per-vertex, implicit tiling for faster traversal, and S2 globe tiling
- **Asset Pipeline:** Upload CityGML, LAS/LAZ, GeoTIFF, KML and Ion converts to optimized 3D Tiles
- **Curated Content:** Cesium World Terrain, Cesium OSM Buildings, Bing Maps imagery

```javascript
import { Cesium3DTileset } from 'cesium';

// Load 3D Tiles from Cesium Ion
const tileset = await Cesium3DTileset.fromIonAssetId(96188); // OSM Buildings
viewer.scene.primitives.add(tileset);
```

### Google Photorealistic 3D Tiles via Cesium

```javascript
// Google Photorealistic 3D Tiles (requires Google Maps API key)
const googleTileset = await Cesium3DTileset.fromUrl(
  `https://tile.googleapis.com/v1/3dtiles/root.json?key=YOUR_GOOGLE_KEY`
);
viewer.scene.primitives.add(googleTileset);

// Or via Cesium Ion (asset ID 2275207)
const googleViaIon = await Cesium3DTileset.fromIonAssetId(2275207);
viewer.scene.primitives.add(googleViaIon);
```

---

## deck.gl

A WebGL/WebGPU-powered framework for visual exploratory data analysis of large datasets, built by Vis.gl (OpenJS Foundation).

- **Website:** [deck.gl](https://deck.gl)
- **Key Strengths:** GPU-accelerated rendering, millions of data points, layer-based architecture
- **Best For:** Large-scale data visualization, trip/trajectory analysis, scientific data

### Install

```bash
npm install deck.gl
```

### Quick Start

```javascript
import { Deck } from '@deck.gl/core';
import { GeoJsonLayer } from '@deck.gl/layers';
import { Map } from 'maplibre-gl';

const deck = new Deck({
  initialViewState: {
    longitude: 116.4, latitude: 39.9, zoom: 10, pitch: 45
  },
  controller: true,
  layers: [
    new GeoJsonLayer({
      id: 'buildings',
      data: '/data/buildings.geojson',
      extruded: true,
      getElevation: d => d.properties.height,
      getFillColor: [160, 160, 180],
      opacity: 0.8
    })
  ]
});
```

### deck.gl 9.x Features

- **WebGPU support:** Opt-in WebGPU backend for next-gen GPU performance
- **Globe view:** Built-in globe projection (no Cesium required for simple globe use cases)
- **Arrow/Parquet support:** Load large datasets directly from Apache Arrow and GeoParquet files
- **Improved TypeScript:** Full type coverage across all layers
- **MapLibre integration:** `@deck.gl/maplibre` module for seamless overlay

```javascript
import { COORDINATE_SYSTEM } from '@deck.gl/core';
import { _GlobeView as GlobeView } from '@deck.gl/core';

// deck.gl 9.x globe view
const deckInstance = new Deck({
  views: new GlobeView(),
  initialViewState: { longitude: 0, latitude: 20, zoom: 1 },
  layers: [/* ... */]
});
```

---

## Three.js + Geo

Using Three.js with geographic data for custom 3D visualizations beyond traditional mapping.

- **Website:** [threejs.org](https://threejs.org)
- **Key Strengths:** Full 3D rendering control, custom shaders, rich ecosystem
- **Best For:** Custom 3D effects, artistic geo-viz, AR/VR geospatial applications

### three-geo Example

[three-geo](https://github.com/w3reality/three-geo) builds Three.js terrain meshes from real-world elevation data:

```bash
npm install three three-geo
```

```javascript
import * as THREE from 'three';
import ThreeGeo from 'three-geo';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, innerWidth / innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

// Build terrain mesh around Mt. Fuji
const tgeo = new ThreeGeo({
  tokenMapbox: 'YOUR_MAPBOX_TOKEN'
});

const terrain = await tgeo.getTerrainRgb(
  [35.3606, 138.7274],  // lat, lng of Mt. Fuji
  13.0,                   // radius in km
  12                      // zoom level
);
scene.add(terrain);

camera.position.set(0, 8, 8);
camera.lookAt(0, 0, 0);

function animate() {
  requestAnimationFrame(animate);
  terrain.rotation.z += 0.002;
  renderer.render(scene, camera);
}
animate();
```

---

## Globe.gl

A lightweight, high-level wrapper around Three.js for creating interactive 3D globes with minimal code.

- **Website:** [globe.gl](https://globe.gl)
- **Repository:** [github.com/vasturiano/globe.gl](https://github.com/vasturiano/globe.gl)
- **Key Strengths:** Declarative API, arcs/points/polygons/hexbins built-in, Three.js underneath
- **Best For:** Quick 3D globe prototypes, data storytelling, network/flow visualization

### Install

```bash
npm install globe.gl
```

### Quick Start

```javascript
import Globe from 'globe.gl';

const globe = Globe()
  .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
  .pointsData(cities)
  .pointLat('lat')
  .pointLng('lng')
  .pointAltitude('population_norm')
  .pointColor(() => '#ff6b6b')
  (document.getElementById('globe'));
```

> Globe.gl is ideal when you need a 3D globe in minutes rather than hours. For complex applications, graduate to CesiumJS or deck.gl.

---

## MapLibre 3D Features

MapLibre GL JS includes built-in 3D capabilities for terrain, building extrusions, and sky/atmosphere effects.

- **Key Features:** Terrain exaggeration, 3D building extrusions, hillshade layers, sky layer
- **Best For:** Adding 3D context to 2D vector maps without a full 3D engine

### Terrain & 3D Extrusion

```javascript
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [11.39, 47.27],
  zoom: 12,
  pitch: 60
});

map.on('load', () => {
  // Enable 3D terrain
  map.addSource('terrain', {
    type: 'raster-dem',
    url: 'https://demotiles.maplibre.org/terrain-tiles/tiles.json',
    tileSize: 256
  });
  map.setTerrain({ source: 'terrain', exaggeration: 1.5 });

  // Add sky atmosphere
  map.addLayer({
    id: 'sky',
    type: 'sky',
    paint: {
      'sky-type': 'atmosphere',
      'sky-atmosphere-sun': [0.0, 90.0],
      'sky-atmosphere-sun-intensity': 15
    }
  });

  // 3D building extrusions
  map.addLayer({
    id: '3d-buildings',
    source: 'openmaptiles',
    'source-layer': 'building',
    type: 'fill-extrusion',
    paint: {
      'fill-extrusion-color': '#aaa',
      'fill-extrusion-height': ['get', 'render_height'],
      'fill-extrusion-base': ['get', 'render_min_height'],
      'fill-extrusion-opacity': 0.6
    }
  });
});
```

---

## Comparison Table

| Library | 3D Tiles | Terrain | Performance | Bundle Size | Best For |
|---------|----------|---------|-------------|-------------|----------|
| CesiumJS | 1.1 Native | Native | High | ~3 MB | Digital twins, globe apps |
| deck.gl 9.x | Via loaders | Via layers | Very High (GPU/WebGPU) | ~300 KB+ | Big data viz, analytics |
| Three.js + geo | Manual | three-geo | High | ~150 KB | Custom 3D, creative viz |
| Globe.gl | No | No | Medium | ~200 KB | Quick globe prototypes |
| MapLibre 3D | No | Native | High | ~220 KB | 3D-enhanced 2D maps |

## Use Case Guide

| Use Case | Recommended Library | Reason |
|----------|-------------------|--------|
| City-scale digital twin | CesiumJS | 3D Tiles 1.1 + terrain + time support |
| Google Photorealistic 3D | CesiumJS | Official integration via Cesium Ion |
| Visualize millions of GPS points | deck.gl | GPU-accelerated point rendering |
| Interactive 3D terrain map | MapLibre GL JS | Lightweight terrain with vector data |
| Custom 3D data sculpture | Three.js | Full creative control |
| Flight / satellite tracking | CesiumJS | True globe with time-dynamic entities |
| Transport flow analysis | deck.gl TripsLayer | Animated trajectory rendering |
| Quick globe prototype | Globe.gl | Three lines of code to interactive globe |
| AR/VR geospatial | Three.js + WebXR | Access to WebXR Device API |

## Integration Patterns

### deck.gl on MapLibre

```javascript
import { MapboxOverlay } from '@deck.gl/maplibre';
import { ScatterplotLayer } from '@deck.gl/layers';

const overlay = new MapboxOverlay({
  layers: [
    new ScatterplotLayer({
      data: points,
      getPosition: d => [d.lng, d.lat],
      getRadius: d => Math.sqrt(d.population),
      getFillColor: [255, 140, 0],
      radiusScale: 10
    })
  ]
});
map.addControl(overlay);
```

### CesiumJS + deck.gl (via @deck.gl/carto or custom)

For combining Cesium's globe and 3D Tiles with deck.gl's analytical layers, use the `@deck.gl/carto` module or render deck.gl layers as Cesium primitives.
