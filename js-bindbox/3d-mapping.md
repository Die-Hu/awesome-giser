# 3D Mapping & Globe

> Data validated: 2026-03-21

## 30-Second Decision

| I need... | Use this | Startup time |
|-----------|----------|-------------|
| Extruded buildings + terrain, fast | MapLibre 3D (2.5D) | 15 min |
| Quick globe prototype / landing page | Globe.gl | 10 min |
| Millions of analytical points on map | deck.gl | 1-2 hours |
| True 3D globe, digital twin, 3D Tiles | CesiumJS | Half a day |
| Custom shaders, AR/VR, particle effects | Three.js + geo libs | Days |

---

## Detailed Guide (by startup time)

### 1. MapLibre 3D (2.5D)

The quickest path to "3D" on a web map. Built-in terrain and building extrusion -- zero additional dependencies.

```javascript
const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [116.4, 39.9], zoom: 15, pitch: 60, bearing: -17
});

map.on('load', () => {
  map.addSource('terrain', {
    type: 'raster-dem',
    url: 'https://your-terrain-tiles.com/tiles.json'
  });
  map.setTerrain({ source: 'terrain', exaggeration: 1.5 });

  map.addLayer({
    id: '3d-buildings', source: 'composite', 'source-layer': 'building',
    type: 'fill-extrusion',
    paint: {
      'fill-extrusion-color': '#aaa',
      'fill-extrusion-height': ['get', 'height'],
      'fill-extrusion-opacity': 0.6
    }
  });
});
```

**Small project:** Best 3D option -- zero extra deps, same bundle as 2D map, works on mobile.

**Key caveats:**
- Not true 3D -- no 3D models, point clouds, 3D Tiles, or globe rendering
- Terrain is GPU-intensive on mobile
- Limited camera control (no orbit, no underground viewing)
- **Anti-pattern:** Promising "3D visualization" and delivering 2.5D. Set expectations early.

---

### 2. Globe.gl

Three lines of code to get a globe. Perfect for landing pages and quick demos.

**Quick facts:** Moderate bundle (includes Three.js) | Easy API | Production-readiness: 2/5

```javascript
import Globe from 'globe.gl';
Globe()
  .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
  .pointsData([{ lat: 39.9, lng: 116.4, size: 0.5, color: 'red' }])
  (document.getElementById('globe'));
```

**Small project:** Fastest way to an impressive globe. Great for portfolio pieces, demos, conference talks.

**Key caveats:**
- Not a GIS tool -- no projections, no tile sources, no spatial queries
- Performance ceiling at ~10K features, no LOD or tiling
- Small community, slower bug fixes
- **Anti-pattern:** Using Globe.gl for analytical GIS. Use deck.gl or CesiumJS instead.

---

### 3. deck.gl

GPU-accelerated visualization for massive datasets. 10M+ points at 60fps with binary data mode.

**Quick facts:** ~200KB+ gzip | Medium-Hard learning curve | Uber/Foursquare/Google production use

```javascript
import { Deck } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';

const deck = new Deck({
  initialViewState: { longitude: 116.4, latitude: 39.9, zoom: 10 },
  controller: true,
  layers: [
    new ScatterplotLayer({
      data: points,
      getPosition: d => d.position,
      getRadius: 100,
      getFillColor: d => d.color
    })
  ]
});
```

**Performance:**

| Data Mode | 1K | 10K | 100K | 1M | 10M |
|-----------|-----|------|-------|-----|------|
| JSON/GeoJSON | 60fps | 60fps | 60fps | 30fps | crash |
| Binary (Float32Array) | 60fps | 60fps | 60fps | 60fps | 50fps |

**Small project:** Overkill for <1K features. Shines when you have thousands to millions of data points.

**Key caveats:**
- Not a map library -- needs MapLibre or Google Maps underneath
- GPU-or-nothing: no fallback if WebGL unavailable
- Binary data mode requires thinking in typed arrays
- Must call `deck.finalize()` to avoid GPU memory leaks
- **Anti-pattern:** Using deck.gl for simple maps with <1K features. Use MapLibre's built-in layers.

---

### 4. CesiumJS

The only JS library for true 3D globe rendering. WGS84 ellipsoid, 3D Tiles 1.1, CZML, terrain, point clouds.

**Quick facts:** 3-5MB gzip (massive) | Hard learning curve | ~15K GitHub stars | NASA, US Air Force

```javascript
import { Viewer, Cartesian3, Ion } from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';
Ion.defaultAccessToken = 'YOUR_TOKEN';

const viewer = new Viewer('cesiumContainer', {
  terrain: Cesium.Terrain.fromWorldTerrain()
});
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(116.4, 39.9, 15000)
});
```

**Small project:** **Not recommended** unless you specifically need true globe, 3D Tiles, or time-dynamic CZML. Setup cost is high: large bundle, Cesium ion account, static asset hosting, webpack/vite config.

**Key caveats:**
- 3-5MB bundle, 10-30s load on 3G -- must lazy-load
- Entity API caps at ~1K objects; production must use Primitive API
- 500MB-2GB RAM for 3D Tiles + terrain; mobile devices crash
- Build configuration differs per bundler (Webpack/Vite/Next.js plugins)
- **Anti-pattern:** Using CesiumJS for 2D maps -- massive overkill, use MapLibre.

---

### 5. Three.js + Geo Libraries

Maximum rendering flexibility, minimum geo convenience. Custom shaders, AR/VR, particle systems.

**Small project:** Almost never the right choice unless you need WebXR + geo, custom shaders, or advanced visual effects. You must build coordinate systems, tile loading, LOD, and camera constraints from scratch.

**Key caveats:**
- No geospatial data model -- Three.js knows nothing about coordinates or projections
- 32-bit floats cause jitter at global scale; must implement camera-relative rendering
- Must manually dispose geometries, materials, textures to avoid GPU memory leaks
- **Anti-pattern:** Choosing Three.js when CesiumJS or deck.gl would work

---

## Comparison Table

| Criterion | MapLibre 3D | Globe.gl | deck.gl | CesiumJS | Three.js |
|-----------|-------------|---------|---------|----------|----------|
| Setup time | **15 min** | **10 min** | 1-2 hours | Half day | Days |
| Bundle size | **0 extra** | Moderate | ~200KB | 3-5MB | ~160KB + custom |
| True globe | No | Yes | No | **Yes** | DIY |
| 3D Tiles | No | No | No | **Yes** | No |
| Max features | 100K+ (tiles) | ~10K | **10M+** | 1M+ (Primitives) | Depends |
| Mobile-friendly | **Yes** | OK | OK | **No** | Depends |
| Learning curve | **Medium** | **Easy** | Medium-Hard | Hard | Hard |
