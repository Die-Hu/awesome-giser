# 3D Mapping & Globe -- Deep Dive

> A senior engineer's definitive reference for building production-grade 3D geospatial applications on the web. This covers architecture internals, real-world performance numbers, advanced patterns, and the trade-offs that actually matter when you are shipping digital twins, flight trackers, and urban planning tools.

---

## Quick Picks

| Goal | Library | Why |
|------|---------|-----|
| Digital twin / city-scale 3D | **CesiumJS** | 3D Tiles 1.1, true globe, time-dynamic, terrain, point clouds |
| Millions of analytical points | **deck.gl** | GPU-accelerated WebGL/WebGPU, 10M+ points at 60fps |
| Extruded buildings + terrain, fast | **MapLibre 3D** | Best mobile 2.5D, vector tile native, lighter bundle |
| Quick globe prototype | **Globe.gl** | Three lines of code, Three.js underneath, escape hatch available |
| Custom shaders / AR / VR | **Three.js + geo libs** | Full rendering control, WebXR, particle systems |

---

## CesiumJS -- The Full Story

CesiumJS is the only JavaScript library purpose-built for true 3D globe rendering with a first-class geospatial data model. Everything in Cesium -- coordinates, time, camera, rendering -- is designed around the WGS84 ellipsoid. That is its superpower and its complexity.

**Website:** [cesium.com](https://cesium.com) | **Repo:** [github.com/CesiumGS/cesium](https://github.com/CesiumGS/cesium) | **License:** Apache 2.0

```bash
npm install cesium
```

---

### Architecture & Rendering Pipeline

#### Scene Graph and Primitive Batching

Cesium's rendering model has three layers of abstraction:

1. **Entity API** -- High-level, time-aware, uses CZML/GeoJSON property bags. Each entity owns a visualizer that translates it to a Primitive internally. Convenient but carries overhead for 1000+ objects.
2. **Primitive API** -- Lower-level, direct GPU submission. `Primitive`, `GroundPrimitive`, `Cesium3DTileset`, `Model`. This is what you use when performance matters.
3. **Scene internals** -- `DrawCommand` queue, `RenderState`, `ShaderProgram`. You rarely touch this unless writing custom materials.

The scene renders via a `PassState` pipeline:
- **Globe pass** -- Terrain tiles, imagery compositing
- **Ground primitives pass** -- Clamped-to-ground geometry
- **Opaque pass** -- Solid primitives (buildings, models)
- **Translucent pass** -- Alpha-blended geometry (sorted back-to-front)
- **Overlay pass** -- Billboards, labels, HUD elements
- **Post-process pass** -- FXAA, bloom, ambient occlusion, custom stages

Primitive batching: Cesium batches `GeometryInstance` objects into a single `Primitive` and uploads a single draw call. Batch 10,000 simple geometries into one `Primitive` and you get one draw call instead of 10,000.

```javascript
// BAD: 10,000 separate entities = 10,000 draw calls
for (const city of cities) {
  viewer.entities.add({ position: city.pos, point: { pixelSize: 8 } });
}

// GOOD: one PointPrimitiveCollection = one draw call
const points = viewer.scene.primitives.add(new Cesium.PointPrimitiveCollection());
for (const city of cities) {
  points.add({ position: city.pos, pixelSize: 8, color: Cesium.Color.RED });
}
```

#### 3D Tiles Traversal and LOD Management

3D Tiles is a spatial data structure (implicitly a B-tree or quadtree/octree of tiles) where each tile has:
- A `boundingVolume` (box, sphere, or region)
- A `geometricError` -- the error in meters if this tile is rendered instead of its children
- Content: `b3dm` (batched 3D model), `i3dm` (instanced), `pnts` (point cloud), `cmpt` (composite), or in 1.1: `glTF`

The traversal algorithm at every frame:
1. Start at the root tile
2. If `screenSpaceError(tile) <= maximumScreenSpaceError`, render this tile and stop recursion
3. Otherwise, request children and recurse

`maximumScreenSpaceError` (default 16) is your primary LOD knob. Lower values = sharper tiles = more bandwidth and memory. For distant aerial views, 32 is fine. For walkthrough-level detail, try 4-8.

```javascript
const tileset = await Cesium.Cesium3DTileset.fromIonAssetId(96188);
tileset.maximumScreenSpaceError = 8;       // sharper
tileset.maximumMemoryUsage = 512;          // MB, tile cache ceiling
tileset.cullWithChildrenBounds = true;     // skip off-screen subtrees
viewer.scene.primitives.add(tileset);
```

#### Terrain Rendering: Quantized Mesh vs Heightmap

Cesium supports two terrain formats:

**Heightmap** (legacy): Regular grid of elevation values. Simple but wasteful -- flat areas waste memory on dense grids.

**Quantized-mesh** (preferred): Irregular triangulated mesh (TIN) with vertices encoded as 16-bit integers. Concentrates triangles where terrain is complex. Supports water mask, vertex normals for lighting, and metadata extensions.

```javascript
// Cesium World Terrain (quantized-mesh, ion-hosted)
const terrain = await Cesium.createWorldTerrainAsync({
  requestVertexNormals: true,    // smooth lighting
  requestWaterMask: true         // ocean/lake reflections
});

// Custom terrain provider (self-hosted quantized-mesh)
const customTerrain = new Cesium.CesiumTerrainProvider({
  url: 'https://your-terrain-server/tilesets/terrain',
  requestVertexNormals: true
});
```

Terrain exaggeration (useful for subtle topography like Netherlands or the Midwest):

```javascript
viewer.scene.verticalExaggeration = 3.0;          // Cesium 1.x API
// or on globe:
viewer.scene.globe.terrainExaggeration = 2.5;
```

#### Imagery Layers: Tile Loading, Caching, and Compositing

The `ImageryLayerCollection` composites multiple tile providers via GPU texture blending. Layers are rendered in stack order. Each layer has `alpha`, `brightness`, `contrast`, `hue`, `saturation`, `gamma` properties adjustable at runtime.

Tile loading pipeline:
1. Tiles are requested in a priority queue (closest to camera first)
2. Each tile fetches from the imagery provider URL template
3. Responses are decoded (PNG/JPEG) and uploaded to a WebGL `Texture2D`
4. At render time, each terrain tile samples the relevant imagery textures
5. Cesium maintains an LRU cache of loaded imagery tiles (default 100 MB)

```javascript
// Layer stack: satellite base + semi-transparent overlay
const layers = viewer.imageryLayers;

layers.addImageryProvider(
  await Cesium.IonImageryProvider.fromAssetId(3)  // Bing Aerial
);
layers.addImageryProvider(
  new Cesium.UrlTemplateImageryProvider({
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    credit: 'OpenStreetMap contributors'
  })
).alpha = 0.4;
```

---

### Performance Profile

#### 3D Tiles: Real Numbers

| Scenario | Bandwidth/frame | GPU Memory | Notes |
|----------|----------------|------------|-------|
| City overview (100K tris) | ~50 KB/s | ~80 MB | Initial load burst, then idle |
| Street-level walkthrough (1M tris) | ~500 KB/s | ~400 MB | Continuous loading as camera moves |
| Dense point cloud (10M points) | ~2 MB/s | ~1.2 GB | PNTS tiles, needs chunking |
| OSM Buildings, full city | ~100 KB/s steady | ~200 MB | Highly dependent on tile cache |

Rule of thumb: if `tileset.totalMemoryUsageInBytes` climbs past 800 MB in Chrome, expect tab crashes on integrated GPU systems.

#### Entity API vs Primitive API

| API | 100 objects | 1,000 objects | 10,000 objects | Use When |
|-----|-------------|---------------|----------------|----------|
| Entity API | 60 fps | 45-55 fps | 10-20 fps | Prototyping, time-dynamic data, CZML playback |
| Primitive API | 60 fps | 60 fps | 50-60 fps | Production visualizations |
| PointPrimitiveCollection | 60 fps | 60 fps | 60 fps | Points only, single draw call |
| BillboardCollection | 60 fps | 60 fps | 55-60 fps | Icons, markers |

The Entity API creates a `PropertyBag` for each entity and evaluates it every frame -- even if nothing changes. The Primitive API submits static geometry once. When your feature count exceeds a few hundred and performance matters, switch to primitives.

#### Memory Management

Always dispose tilesets and primitives when removing scenes:

```javascript
// Correct disposal
viewer.scene.primitives.remove(tileset);
tileset.destroy();

// Dispose imagery layer
viewer.imageryLayers.remove(layer, true); // true = destroy the provider

// Clear all entities (faster than entity-by-entity)
viewer.entities.removeAll();

// Nuclear option for full scene teardown
viewer.destroy();
```

WebGL context limits: Chrome allows ~16 WebGL contexts per process. Each `Viewer` creates one. In SPAs that mount/unmount map components, always call `viewer.destroy()` before unmounting or you will exhaust contexts.

#### Mobile 3D

| Platform | Reality |
|----------|---------|
| iOS Safari (iPhone 14+) | WebGL 2 works, 60fps for simple scenes, 3D Tiles limited to ~200 MB before crash |
| Android Chrome (mid-range) | WebGL 2 works, heavier scenes drop to 30fps |
| iOS WKWebView | WebGL 1 only in some configurations, avoidable with newer WKWebView settings |
| Android WebView | Inconsistent WebGL 2 support pre-Android 10 |

For mobile, always set:
```javascript
viewer.scene.requestRenderMode = true;         // only render when dirty
viewer.scene.maximumRenderTimeChange = Infinity;
tileset.maximumScreenSpaceError = 32;          // coarser tiles
tileset.maximumMemoryUsage = 128;              // tight budget
viewer.scene.fog.enabled = true;               // cull distant tiles with fog
viewer.scene.fog.density = 0.0002;
```

#### 60fps Budget Breakdown

At 60fps you have 16.6ms per frame. Typical breakdown for a CesiumJS scene with a loaded tileset:

| Task | Budget |
|------|--------|
| Tile traversal + culling | 1-3 ms |
| Tile decode (worker threads) | off-thread |
| Terrain tile selection | 0.5 ms |
| Draw command generation | 1-2 ms |
| WebGL draw calls (GPU) | 4-8 ms |
| Post-processing (FXAA + bloom) | 1-3 ms |
| Event handling / picking | 0.5-2 ms |
| Available for app logic | 2-5 ms |

Bloom and SSAO are expensive (3-5 ms each). Disable them on mobile.

---

### Deep Feature Coverage

#### 3D Tiles 1.1: What Changed

3D Tiles 1.1 (finalized 2022) brings major improvements:

- **glTF content**: Tiles use `.glb` directly instead of the old `b3dm`/`i3dm` wrappers. This means standard glTF tooling works directly on 3D Tiles content.
- **Implicit tiling**: Describes tile hierarchy mathematically (quadtree/octree + availability bitstream) instead of explicit `tileset.json` trees. A 1M-tile dataset that required a 40 MB JSON file now needs 4 KB of metadata.
- **Metadata (3DTILES_metadata)**: Per-tileset, per-group, per-tile, per-feature, and per-vertex metadata. Enables GPU-accelerated feature querying without picking.
- **S2 cells**: Alternative bounding volume using S2 geometry for globe-covering tilesets. Better than geographic region boxes near poles.

```javascript
// Access per-feature metadata (3D Tiles 1.1)
tileset.featureIdLabel = 'featureId_0';

tileset.style = new Cesium.Cesium3DTileStyle({
  color: {
    conditions: [
      ['${buildingType} === "residential"', 'color("lightblue")'],
      ['${buildingType} === "commercial"', 'color("orange")'],
      [true, 'color("white")']
    ]
  },
  show: '${height} > 10'
});
```

#### CZML: Time-Dynamic Data Format

CZML is a JSON-based format for describing time-dynamic scenes. Think of it as a stream of property packets that Cesium interpolates.

Complete CZML document for satellite tracking:

```json
[
  {
    "id": "document",
    "name": "ISS Tracking",
    "version": "1.0",
    "clock": {
      "interval": "2024-01-01T00:00:00Z/2024-01-01T01:30:00Z",
      "currentTime": "2024-01-01T00:00:00Z",
      "multiplier": 60,
      "range": "LOOP_STOP",
      "step": "SYSTEM_CLOCK_MULTIPLIER"
    }
  },
  {
    "id": "ISS",
    "name": "International Space Station",
    "availability": "2024-01-01T00:00:00Z/2024-01-01T01:30:00Z",
    "position": {
      "interpolationAlgorithm": "LAGRANGE",
      "interpolationDegree": 5,
      "referenceFrame": "INERTIAL",
      "epoch": "2024-01-01T00:00:00Z",
      "cartesian": [
        0,    -5972877.9, 2504683.2, 3226269.5,
        60,   -6212563.9, 1941155.4, 3361736.9,
        120,  -6376994.7, 1360278.5, 3457027.9
      ]
    },
    "model": {
      "gltf": "https://assets.cesium.com/iss.glb",
      "scale": 10.0,
      "minimumPixelSize": 32
    },
    "path": {
      "material": { "solidColor": { "color": { "rgba": [255, 255, 0, 128] } } },
      "width": 2,
      "leadTime": 0,
      "trailTime": 3600
    },
    "label": {
      "text": "ISS",
      "font": "14pt Lucida Console",
      "fillColor": { "rgba": [255, 255, 255, 255] },
      "pixelOffset": { "cartesian2": [0, -24] }
    }
  }
]
```

```javascript
// Load CZML into viewer
const czmlSource = new Cesium.CzmlDataSource();
await czmlSource.load('/data/iss-orbit.czml');
viewer.dataSources.add(czmlSource);
viewer.clock.shouldAnimate = true;
```

#### Entity API Reference

| Entity Type | Key Properties | Use Case |
|-------------|---------------|----------|
| `billboard` | image, scale, pixelOffset, alignedAxis | Icons, markers with fixed screen size |
| `label` | text, font, fillColor, outlineColor, pixelOffset | Text annotations |
| `polyline` | positions, width, material, clampToGround | Routes, trajectories |
| `polygon` | hierarchy, height, extrudedHeight, material | Zones, building footprints |
| `ellipsoid` | radii, material, outline | Sensor volumes, uncertainty spheres |
| `model` | uri (glTF), scale, minimumPixelSize, silhouette | Vehicles, aircraft, characters |
| `path` | leadTime, trailTime, width, material | Moving entity trails |
| `wall` | positions, maximumHeights, material | Fences, boundaries, airspace |
| `corridor` | positions, width, height, cornerType | Roads, pipelines |
| `point` | pixelSize, color, outlineColor | Simple dot markers |

#### Custom Shaders and Materials

Cesium's "Fabric" material system lets you write GLSL snippets:

```javascript
// Custom fabric material: animated scanline effect
const scanlineMaterial = new Cesium.Material({
  fabric: {
    type: 'Scanline',
    uniforms: {
      color: new Cesium.Color(0.0, 1.0, 0.5, 0.8),
      speed: 2.0,
      lineCount: 20.0
    },
    source: `
      czm_material czm_getMaterial(czm_materialInput materialInput) {
        czm_material material = czm_getDefaultMaterial(materialInput);
        float time = czm_frameNumber / 60.0 * speed;
        float stripe = mod(materialInput.st.t * lineCount + time, 1.0);
        material.diffuse = color.rgb;
        material.alpha = color.a * step(0.5, stripe);
        return material;
      }
    `
  }
});

polygon.material = scanlineMaterial;
```

Post-process pipeline (bloom + FXAA):

```javascript
const stages = viewer.scene.postProcessStages;

// Built-in FXAA (always on by default)
stages.fxaa.enabled = true;

// Built-in bloom
const bloom = stages.bloom;
bloom.enabled = true;
bloom.uniforms.glowOnly = false;
bloom.uniforms.contrast = 128;
bloom.uniforms.brightness = -0.3;
bloom.uniforms.delta = 1.0;
bloom.uniforms.sigma = 3.78;
bloom.uniforms.stepSize = 1.0;

// Custom post-process stage
const customStage = new Cesium.PostProcessStage({
  fragmentShader: `
    uniform sampler2D colorTexture;
    in vec2 v_textureCoordinates;
    void main() {
      vec4 color = texture(colorTexture, v_textureCoordinates);
      // Convert to night-vision green
      float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
      out_FragColor = vec4(0.0, luma, 0.0, color.a);
    }
  `
});
stages.add(customStage);
```

#### Time System

Cesium's time system is built on **Julian Date** (continuous real number counting days since noon Jan 1, 4713 BC). This avoids the complexity of leap seconds and calendar arithmetic.

```javascript
const now = Cesium.JulianDate.now();
const future = Cesium.JulianDate.addHours(now, 2, new Cesium.JulianDate());

// Configure the clock
viewer.clock.startTime = Cesium.JulianDate.fromIso8601('2024-01-01T00:00:00Z');
viewer.clock.stopTime  = Cesium.JulianDate.fromIso8601('2024-01-02T00:00:00Z');
viewer.clock.currentTime = viewer.clock.startTime.clone();
viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
viewer.clock.multiplier = 3600;  // 1 hour per second of wall clock
viewer.clock.shouldAnimate = true;

// React to time changes
viewer.clock.onTick.addEventListener((clock) => {
  const iso = Cesium.JulianDate.toIso8601(clock.currentTime);
  document.getElementById('time-display').textContent = iso;
});

// Time-dynamic property (SampledProperty)
const altitudeProp = new Cesium.SampledProperty(Number);
altitudeProp.interpolationAlgorithm = Cesium.LinearApproximation;
altitudeProp.addSample(startTime, 0);
altitudeProp.addSample(midTime, 5000);
altitudeProp.addSample(endTime, 0);

entity.point = { pixelSize: altitudeProp };  // pixelSize varies with time
```

#### Camera Control

```javascript
const camera = viewer.camera;

// Fly to position
camera.flyTo({
  destination: Cesium.Cartesian3.fromDegrees(116.4, 39.9, 2000),
  orientation: { heading: Cesium.Math.toRadians(45), pitch: Cesium.Math.toRadians(-30), roll: 0 },
  duration: 3.0,
  easingFunction: Cesium.EasingFunction.CUBIC_IN_OUT
});

// Look at a target
camera.lookAt(
  Cesium.Cartesian3.fromDegrees(116.4, 39.9, 0),
  new Cesium.HeadingPitchRange(Cesium.Math.toRadians(0), Cesium.Math.toRadians(-45), 5000)
);

// Set view instantly (no animation)
camera.setView({
  destination: Cesium.Cartesian3.fromDegrees(116.4, 39.9, 10000),
  orientation: { heading: 0, pitch: -Cesium.Math.PI_OVER_TWO, roll: 0 }
});

// React to camera movement (debounced!)
let moveEndTimeout;
camera.changed.addEventListener(() => {
  clearTimeout(moveEndTimeout);
  moveEndTimeout = setTimeout(() => {
    const pos = camera.positionCartographic;
    console.log('Camera at:', {
      lon: Cesium.Math.toDegrees(pos.longitude),
      lat: Cesium.Math.toDegrees(pos.latitude),
      alt: pos.height
    });
  }, 300);
});
```

---

### Cesium Ion vs Self-Hosted

#### What Cesium Ion Provides

- **Cesium World Terrain** -- Global quantized-mesh terrain at 1-arc-second resolution, with bathymetry
- **Cesium OSM Buildings** -- 350M+ building footprints extruded from OpenStreetMap as 3D Tiles
- **Asset pipeline** -- Upload and convert LAS, CityGML, GeoTIFF, FBX, OBJ, KML to 3D Tiles
- **Token auth** -- `Ion.defaultAccessToken` gates access; never expose tokens in client-side bundles without domain restrictions
- **CDN hosting** -- Globally distributed tile delivery

#### Self-Hosting the Stack

| Component | Tool | Notes |
|-----------|------|-------|
| Point clouds (.las/.laz) to 3D Tiles | `py3dtiles`, `PDAL` | Open source, slow for 1B+ points |
| CityGML/IFC to 3D Tiles | `FME`, `3d-tiles-tools` | FME is commercial but fast |
| Terrain generation | `ctb-quantized-mesh` (Cesium Terrain Builder) | From GeoTIFF DEM |
| Terrain hosting | `cesium-terrain-server` | Go-based tile server |
| 3D Tiles post-processing | `3d-tiles-tools` (official) | Merge, reformat, validate tilesets |

Cost analysis: Ion free tier is 5 GB storage, 200K tile requests/month. A city-scale digital twin with 50 GB of point cloud data and 500K daily tile requests costs ~$400/month on Ion. Self-hosting on AWS S3 + CloudFront costs ~$50-80/month for the same traffic once you have the tiling pipeline.

#### Google Photorealistic 3D Tiles

Google's Photorealistic 3D Tiles (part of Google Maps Platform) provides mesh-textured photogrammetry coverage of 2500+ cities worldwide.

```javascript
// Direct Google Maps API (no Ion)
const googleTileset = await Cesium.Cesium3DTileset.fromUrl(
  `https://tile.googleapis.com/v1/3dtiles/root.json?key=${GOOGLE_MAPS_API_KEY}`,
  {
    showCreditsOnScreen: true  // Required by terms of service
  }
);
viewer.scene.primitives.add(googleTileset);

// Disable Cesium's default globe so Google tiles fill the world
viewer.scene.globe.show = false;
```

**Terms of service gotchas:**
- Must display Google logo (handled by `showCreditsOnScreen: true`)
- Cannot store/cache tiles beyond browser cache
- Cannot combine with competing basemap services per policy
- Pricing: $0.006 per 1000 tile requests; a single user session loading a city can generate 2000+ tile requests

---

### Integration Patterns

#### Resium (React)

[Resium](https://resium.reearth.io/) wraps CesiumJS components as React components. Key patterns:

```jsx
import { Viewer, Entity, CameraFlyTo, Cesium3DTileset } from 'resium';
import { Cartesian3, Ion } from 'cesium';

Ion.defaultAccessToken = 'YOUR_TOKEN';

function MapComponent() {
  const [tilesetRef, setTilesetRef] = useState(null);

  return (
    <Viewer full>
      <CameraFlyTo
        destination={Cartesian3.fromDegrees(116.4, 39.9, 5000)}
        once={true}
      />
      <Cesium3DTileset
        url="https://your-tileset/tileset.json"
        onReady={(tileset) => {
          setTilesetRef(tileset);
          // Adjust height offset if tileset floats above terrain
          const boundingSphere = tileset.boundingSphere;
          viewer.zoomTo(tileset);
        }}
      />
      <Entity
        position={Cartesian3.fromDegrees(116.4, 39.9, 100)}
        point={{ pixelSize: 10 }}
      />
    </Viewer>
  );
}
```

**Resium performance pitfalls:**
- Every React re-render re-evaluates Resium component props. Wrap stable objects (`Cartesian3`, `Color`) in `useMemo` or define them outside the component.
- The `Viewer` component creates a new CesiumJS viewer on mount. In strict mode, React mounts twice -- always handle the second mount or you get double viewers.
- Use the `ref` prop to access the raw Cesium viewer instance for operations not exposed by Resium.

#### vue-cesium

```javascript
// main.js
import { createApp } from 'vue';
import VueCesium from 'vue-cesium';
import 'vue-cesium/dist/index.css';

createApp(App).use(VueCesium, {
  cesiumPath: 'https://unpkg.com/cesium/Build/Cesium/Cesium.js',
  accessToken: 'YOUR_TOKEN'
}).mount('#app');
```

```vue
<template>
  <vc-viewer @ready="onViewerReady">
    <vc-layer-imagery>
      <vc-imagery-provider-bing map-style="AerialWithLabels" />
    </vc-layer-imagery>
    <vc-terrain-provider-cesium />
    <vc-primitive-tileset :url="tilesetUrl" @ready="onTilesetReady" />
  </vc-viewer>
</template>
```

#### deck.gl Analytical Layers on Cesium Globe

```javascript
import { CesiumWidget } from 'cesium';
import { Deck } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';

const cesiumWidget = new CesiumWidget('cesiumContainer');
const scene = cesiumWidget.scene;

// Create an offscreen deck.gl canvas synced to Cesium's camera
const deckCanvas = document.createElement('canvas');
deckCanvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none';
document.getElementById('cesiumContainer').appendChild(deckCanvas);

const deck = new Deck({
  canvas: deckCanvas,
  width: '100%',
  height: '100%',
  controller: false,  // Cesium handles interaction
  layers: [
    new ScatterplotLayer({
      data: heatmapPoints,
      getPosition: d => [d.lon, d.lat, d.alt || 0],
      getRadius: d => d.magnitude * 100,
      getFillColor: d => [255, d.magnitude * 25, 0, 180],
      radiusUnits: 'meters',
      coordinateSystem: COORDINATE_SYSTEM.LNGLAT
    })
  ]
});

// Sync deck.gl viewport to Cesium camera every frame
scene.postRender.addEventListener(() => {
  const camera = scene.camera;
  const { clientWidth: width, clientHeight: height } = scene.canvas;
  const pos = camera.positionCartographic;

  deck.setProps({
    viewState: {
      longitude: Cesium.Math.toDegrees(pos.longitude),
      latitude: Cesium.Math.toDegrees(pos.latitude),
      zoom: altitudeToZoom(pos.height),
      bearing: Cesium.Math.toDegrees(-camera.heading),
      pitch: 90 - Cesium.Math.toDegrees(-camera.pitch)
    }
  });
  deck.redraw();
});
```

---

### Development Experience

#### Bundle Size

CesiumJS is ~3 MB gzip (11 MB raw). This is unavoidable -- it is a full geospatial runtime. Mitigation:

```javascript
// 1. Dynamic import to defer loading
const { Viewer } = await import('cesium');

// 2. Vite config: don't bundle Cesium, use CDN
// vite.config.js
export default {
  build: {
    rollupOptions: {
      external: ['cesium'],
      output: {
        globals: { cesium: 'Cesium' }
      }
    }
  }
}

// 3. cesium-webpack-plugin for production builds
// Copies Cesium's static assets, sets CESIUM_BASE_URL
```

#### TypeScript

`@types/cesium` ships with the main package since Cesium 1.95. Quality is good but watch for:

```typescript
// Common pain point: Cartesian3 vs Cartographic vs degrees
// Cesium.Cartesian3 = ECEF XYZ meters (NOT lat/lon!)
// Cesium.Cartographic = lon/lat/height in RADIANS
// Cesium.Cartesian3.fromDegrees() converts for you

const pos: Cesium.Cartesian3 = Cesium.Cartesian3.fromDegrees(116.4, 39.9, 0);
const cart: Cesium.Cartographic = Cesium.Cartographic.fromDegrees(116.4, 39.9, 0);

// SampledProperty generic type issue -- cast needed
const prop = new Cesium.SampledProperty(Cesium.Cartesian3) as Cesium.SampledProperty<Cesium.Cartesian3>;
```

#### Debugging Tools

```javascript
// Frame rate overlay
viewer.scene.debugShowFramesPerSecond = true;

// 3D Tiles Inspector widget
viewer.extend(Cesium.viewerCesium3DTilesInspectorMixin);

// Scene Inspector
viewer.extend(Cesium.viewerCesiumInspectorMixin);

// Visualize tile bounding volumes
tileset.debugShowBoundingVolume = true;
tileset.debugShowContentBoundingVolume = true;
tileset.debugShowViewerRequestVolume = true;
tileset.debugColorTiles = true;  // Random color per tile = LOD visualization

// Show draw call count
viewer.scene.debugShowCommands = true;
```

---

## deck.gl -- GPU-Powered Analytics

deck.gl is a framework for visual exploratory data analysis of large datasets, maintained by the Vis.gl steering committee under the OpenJS Foundation. It runs on WebGL 2 with an optional WebGPU backend.

**Website:** [deck.gl](https://deck.gl) | **Repo:** [github.com/visgl/deck.gl](https://github.com/visgl/deck.gl) | **License:** MIT

```bash
npm install deck.gl
# or individual modules:
npm install @deck.gl/core @deck.gl/layers @deck.gl/geo-layers @deck.gl/aggregation-layers
```

---

### Architecture

#### Layer Lifecycle

Every deck.gl layer goes through a deterministic lifecycle:

1. **`initializeState()`** -- Allocate GPU buffers, create `Model` objects (WebGL programs)
2. **`updateState({ props, oldProps, changeFlags })`** -- Called when props change. Diff old/new props, update only changed GPU attributes.
3. **`draw({ uniforms })`** -- Submit draw calls. Called every frame only if layer is visible.
4. **`finalizeState()`** -- Delete GPU resources. Called on unmount.

The key insight: `updateState` is **not** called every frame, only when props change. If your data does not change, the GPU buffers are not re-uploaded. This is why deck.gl can sustain 60fps with 10M static points.

#### Attribute Management

deck.gl's `AttributeManager` automatically manages TypedArrays and GPU buffer uploads:

```javascript
// deck.gl packs your accessor functions into Float32Arrays for the GPU
// getPosition: d => [d.x, d.y] is called ONCE per data update, not per frame
// Results are packed into a Float32Array and uploaded to GPU VRAM
// The shader reads directly from VRAM every frame at minimal cost

new ScatterplotLayer({
  data: myData,  // Array of 1M objects
  getPosition: d => [d.lon, d.lat],  // Called 1M times on data change, then cached
  getRadius: d => d.value * 10,      // Same: cached as Float32Array
  updateTriggers: {
    // IMPORTANT: Force attribute re-computation when derived values change
    getFillColor: [selectedId, colorScale]  // Re-run getFillColor when these change
  }
})
```

#### WebGPU Backend

As of deck.gl 9.x, opt-in WebGPU support via `@luma.gl/webgpu`:

```javascript
import { createDevice } from '@luma.gl/core';
import { webgpuAdapter } from '@luma.gl/webgpu';

const device = await createDevice({ type: 'webgpu', adapters: [webgpuAdapter] });

const deck = new Deck({
  device,  // WebGPU device instead of WebGL
  layers: [/* same layers, transparent to app code */]
});
```

WebGPU differences: compute shaders available (enables GPU aggregation without fragment shader tricks), better multi-threading, more predictable memory management. Not yet supported in Firefox (as of early 2026).

---

### Performance Profile

#### Real Benchmarks (Chrome, M2 MacBook Pro, 2024)

| Layer | Data Size | FPS | GPU Memory | Notes |
|-------|-----------|-----|------------|-------|
| ScatterplotLayer | 100K points | 60 | 8 MB | Trivial |
| ScatterplotLayer | 1M points | 60 | 40 MB | Still smooth |
| ScatterplotLayer | 10M points | 45-55 | 400 MB | Depends on radius computation |
| GeoJsonLayer (polygons) | 50K features | 55 | 80 MB | Triangulation cost on update |
| GeoJsonLayer (extruded) | 10K buildings | 60 | 30 MB | Less geometry than expected |
| HexagonLayer | 1M points | 58 | 200 MB | CPU aggregation on update |
| H3HexagonLayer | 100K pre-agg H3 | 60 | 20 MB | Pre-aggregated = no CPU cost |
| TripsLayer | 100K trips | 50 | 150 MB | Animated, updates uniforms per frame |

**H3HexagonLayer vs HexagonLayer:** If you pre-aggregate to H3 hex IDs server-side, H3HexagonLayer is 10x faster on data updates because there is no CPU aggregation step. Always do aggregation on the server/worker for production.

#### Tips for Peak Performance

```javascript
// 1. Use Float64Array for positions (avoids precision jitter at high zoom)
const positions = new Float64Array(data.flatMap(d => [d.lon, d.lat]));
new ScatterplotLayer({
  data: { length: positions.length / 2 },
  getPosition: (_, { index }) => positions.subarray(index * 2, index * 2 + 2),
  // ...
});

// 2. Minimize layer object recreation
// BAD: creates a new layer object every render (triggers updateState)
const layers = data.map(d => new ScatterplotLayer({ id: d.id, data: d.points }));

// GOOD: stable layer references, only re-run accessors when updateTriggers change
const layer = useMemo(() => new ScatterplotLayer({
  id: 'my-scatter',
  data: stableData,
  updateTriggers: { getFillColor: colorMode }
}), [stableData, colorMode]);

// 3. Binary data format for large datasets
const binaryData = {
  length: 1000000,
  attributes: {
    getPosition: { value: new Float32Array(positions), size: 2 },
    getRadius:   { value: new Float32Array(radii), size: 1 },
    getFillColor: { value: new Uint8Array(colors), size: 4, normalized: true }
  }
};
new ScatterplotLayer({ data: binaryData });
```

---

### Layer Catalog Deep Dive

| Layer | Data Type | GPU Acc. | Use Case | Max Features |
|-------|-----------|----------|----------|--------------|
| **ScatterplotLayer** | Points | Yes | Locations, events | 10M+ |
| **IconLayer** | Points | Yes | Categorized markers | 500K |
| **TextLayer** | Points | Yes (SDF) | Labels at scale | 100K |
| **LineLayer** | Line segments | Yes | Connections, flows | 5M |
| **ArcLayer** | OD pairs | Yes | Flight routes, OD flows | 1M |
| **PathLayer** | Polylines | Yes | Routes, trajectories | 200K |
| **SolidPolygonLayer** | Polygons | Yes | Zones, land use | 100K |
| **GeoJsonLayer** | GeoJSON mixed | Partial | General vector data | 50K |
| **ColumnLayer** | Points | Yes | 3D bar chart on map | 500K |
| **GridCellLayer** | Grid cells | Yes | Regular grid viz | 1M |
| **HexagonLayer** | Points (aggr.) | CPU aggr. | Density heatmap | 1M input |
| **ContourLayer** | Points (aggr.) | CPU aggr. | Isoline / isofill | 500K input |
| **HeatmapLayer** | Points | GPU | Smooth density | 1M |
| **ScreenGridLayer** | Points | GPU | Viewport grid aggr. | 5M |
| **GPUGridLayer** | Points | GPU | GPU-side aggregation | 5M |
| **MVTLayer** | Vector tiles | Yes | Tiled vector data | Unlimited |
| **TileLayer** | Raster tiles | Yes | Tiled imagery/data | Unlimited |
| **Tile3DLayer** | 3D Tiles | Yes | Point clouds, meshes | Unlimited (streamed) |
| **TerrainLayer** | DEM + texture | Yes | 3D terrain drape | 1 tile per viewport |
| **H3HexagonLayer** | H3 hex IDs | Yes | Pre-aggregated hex | 1M |
| **H3ClusterLayer** | H3 hex IDs | Yes | Hex clusters | 500K |
| **S2Layer** | S2 cell tokens | Yes | S2 geometry | 500K |
| **TripsLayer** | Trip timestamps | Yes | Animated movement | 100K trips |
| **GreatCircleLayer** | OD pairs | Yes | Globe arc paths | 500K |
| **SimpleMeshLayer** | Points + mesh | Yes | 3D object placement | 100K instances |
| **ScenegraphLayer** | Points + glTF | Yes | Animated 3D models | 10K |

---

### Integration with MapLibre

```javascript
import { MapboxOverlay } from '@deck.gl/maplibre';
import { ScatterplotLayer, ArcLayer } from '@deck.gl/layers';
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [116.4, 39.9], zoom: 10
});

// Interleaved mode: deck.gl layers render between MapLibre layers
// Use this for layers that need to respect z-order with map features
const overlay = new MapboxOverlay({
  interleaved: true,   // false = overlaid on top of everything
  layers: [
    new ScatterplotLayer({
      id: 'scatter-interleaved',
      data: points,
      getPosition: d => [d.lon, d.lat],
      getRadius: 50,
      getFillColor: [255, 0, 0],
      radiusUnits: 'meters',
      beforeId: 'waterway-label'  // Insert before this MapLibre layer
    })
  ]
});

map.addControl(overlay);

// Update layers dynamically
overlay.setProps({
  layers: [
    new ArcLayer({
      id: 'arcs',
      data: odPairs,
      getSourcePosition: d => d.origin,
      getTargetPosition: d => d.destination,
      getWidth: 2,
      getSourceColor: [0, 128, 255],
      getTargetColor: [255, 128, 0]
    })
  ]
});
```

**Interleaved vs Overlaid:**
- `interleaved: true` -- deck.gl shares the WebGL context with MapLibre. Requires MapLibre GL JS v3+ or mapbox-gl v2.6+. Deck.gl layers can sit between map layers (e.g., above roads but below labels).
- `interleaved: false` (default) -- Separate canvas on top. Simpler, more compatible, but deck.gl always renders above everything in MapLibre.

---

### pydeck (Python)

pydeck exposes every deck.gl layer as a Python API with Jupyter widget support:

```python
import pydeck as pdk

layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,  # Pandas DataFrame
    get_position="[lon, lat]",
    get_radius="value * 100",
    get_fill_color=[255, 0, 0, 160],
    pickable=True
)

view_state = pdk.ViewState(latitude=39.9, longitude=116.4, zoom=10, pitch=45)

r = pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "{name}: {value}"}
)

# Jupyter display
r  # auto-renders in notebook

# Export standalone HTML
r.to_html("output.html")
```

---

## Three.js + Geo

Three.js is a general-purpose 3D rendering library. It is not a GIS engine -- you bring your own coordinate system, projection, and data loading. This gives maximum creative freedom and also maximum complexity.

**Website:** [threejs.org](https://threejs.org) | **License:** MIT

---

### When to Choose Three.js Over Cesium/deck.gl

- Custom visual effects: particle systems, custom GLSL shaders, multi-pass rendering, volumetric fog
- Non-standard projections or coordinate spaces (e.g., interior building navigation with local coordinates)
- AR/VR via WebXR API (Cesium has limited WebXR support; Three.js is mature here)
- Game-like interactions: physics, collision detection, raycasting with complex meshes
- Artistic geo-visualization where accuracy is secondary to aesthetics
- Embedding geo data into an existing Three.js scene (product configurator with a globe, etc.)

---

### Geo Libraries in the Three.js Ecosystem

| Library | Purpose | Maturity |
|---------|---------|---------|
| **three-geo** | Terrain mesh from Mapbox DEM tiles | Maintained |
| **three-geojson-geometry** | GeoJSON → BufferGeometry (lines, polygons) | Stable |
| **three-globe** (vasturiano) | Globe.gl's Three.js core, reusable | Active |
| **iTowns** (IGN France) | Research-grade 3D viewer: 3D Tiles, WFS, point clouds | Active |
| **proj4js** | Coordinate projection transforms | Widely used |

**iTowns** is particularly powerful -- it handles 3D Tiles, WFS, WMS, elevation layers, and point clouds on top of Three.js. If you want Three.js control with CesiumJS-like geospatial capability, iTowns is the answer.

---

### three-geo Terrain Example

```bash
npm install three three-geo
```

```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import ThreeGeo from 'three-geo';

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.1, 1000);
camera.position.set(0, 12, 12);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.4));
const sun = new THREE.DirectionalLight(0xffffff, 1.2);
sun.position.set(5, 10, 5);
sun.castShadow = true;
scene.add(sun);

// Build terrain
const tgeo = new ThreeGeo({ tokenMapbox: 'YOUR_MAPBOX_TOKEN' });
const terrain = await tgeo.getTerrainRgb(
  [35.3606, 138.7274],  // Mt. Fuji
  14.0, 12
);
terrain.receiveShadow = true;
scene.add(terrain);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
```

---

### Performance

```javascript
// InstancedMesh for 100K identical objects (trees, buildings, vehicles)
const geometry = new THREE.CylinderGeometry(0.5, 0.5, 2, 6);  // hexagonal building
const material = new THREE.MeshLambertMaterial({ color: 0x8888aa });
const mesh = new THREE.InstancedMesh(geometry, material, 100000);

const matrix = new THREE.Matrix4();
const position = new THREE.Vector3();
const quaternion = new THREE.Quaternion();
const scale = new THREE.Vector3();

for (let i = 0; i < 100000; i++) {
  // Convert lat/lon to local Cartesian
  position.set(
    (lonArray[i] - centerLon) * metersPerDegreeLon,
    heightArray[i] / 2,
    -(latArray[i] - centerLat) * metersPerDegreeLat
  );
  scale.set(1, heightArray[i], 1);
  matrix.compose(position, quaternion, scale);
  mesh.setMatrixAt(i, matrix);
}
mesh.instanceMatrix.needsUpdate = true;
scene.add(mesh);

// Level of Detail
import { LOD } from 'three';
const lod = new LOD();
lod.addLevel(highDetailMesh, 0);     // < 100m
lod.addLevel(medDetailMesh, 100);    // 100-500m
lod.addLevel(lowDetailMesh, 500);    // > 500m
scene.add(lod);

// Memory cleanup -- ALWAYS do this
function disposeScene(scene) {
  scene.traverse(obj => {
    if (obj.geometry) obj.geometry.dispose();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose());
      else obj.material.dispose();
    }
    if (obj.texture) obj.texture.dispose();
  });
}
```

---

## MapLibre GL JS 3D Features

MapLibre can do a lot in 3D without needing CesiumJS. For 2.5D use cases (extruded buildings, terrain, sky), it is faster on mobile and much lighter to bundle.

---

### What MapLibre Can Do Without CesiumJS

```javascript
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [11.39, 47.27], zoom: 12, pitch: 60, bearing: -20,
  projection: 'globe'  // v4+: Globe projection!
});

map.on('load', () => {
  // 3D Terrain
  map.addSource('terrain', {
    type: 'raster-dem',
    url: 'https://demotiles.maplibre.org/terrain-tiles/tiles.json',
    tileSize: 256
  });
  map.setTerrain({ source: 'terrain', exaggeration: 2.0 });

  // Sky atmosphere
  map.addLayer({
    id: 'sky', type: 'sky',
    paint: {
      'sky-type': 'atmosphere',
      'sky-atmosphere-sun': [0.0, 90.0],
      'sky-atmosphere-sun-intensity': 15
    }
  });

  // Building extrusions from vector tiles
  map.addLayer({
    id: '3d-buildings',
    source: 'openmaptiles', 'source-layer': 'building',
    type: 'fill-extrusion',
    filter: ['==', 'extrude', 'true'],
    paint: {
      'fill-extrusion-color': [
        'interpolate', ['linear'], ['get', 'render_height'],
        0, '#aaa', 200, '#334'
      ],
      'fill-extrusion-height': ['get', 'render_height'],
      'fill-extrusion-base': ['get', 'render_min_height'],
      'fill-extrusion-opacity': 0.75
    }
  });
});
```

### Custom 3D Layer (Raw WebGL Injection)

```javascript
// Inject raw WebGL into MapLibre's render pipeline
// Classic use case: wind particle visualization
const windLayer = {
  id: 'wind-particles',
  type: 'custom',
  renderingMode: '3d',

  onAdd(map, gl) {
    this.program = createShaderProgram(gl, vertexSrc, fragmentSrc);
    this.buffer = initParticleBuffer(gl, 10000);
    this.map = map;
  },

  render(gl, matrix) {
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(
      gl.getUniformLocation(this.program, 'u_matrix'),
      false, matrix
    );
    // Update particle positions, draw
    updateParticles(gl, this.buffer);
    gl.drawArrays(gl.POINTS, 0, 10000);
    this.map.triggerRepaint();  // Continuous animation
  },

  onRemove(map, gl) {
    gl.deleteProgram(this.program);
    gl.deleteBuffer(this.buffer);
  }
};

map.addLayer(windLayer);
```

---

### MapLibre 3D vs CesiumJS: When to Use Which

| Feature | MapLibre 3D | CesiumJS |
|---------|-------------|---------|
| Building extrusions | Excellent | Good (via 3D Tiles) |
| Terrain | Excellent (raster-dem) | Excellent (quantized-mesh) |
| True globe | v4+ globe projection | True globe (WGS84 ellipsoid) |
| Time-dynamic data | Manual | Native (CZML, Clock) |
| 3D Tiles | No | Native |
| Point clouds | No | Native (pnts, las) |
| Mobile performance | Excellent | Moderate |
| Bundle size | ~220 KB gzip | ~3 MB gzip |

MapLibre 3D is the right choice for 2.5D dashboards where you need terrain context and building extrusion but not true 3D models or time animation. Cesium is the right choice when you are doing digital twins, simulation, or need to load 3D Tiles.

---

## Globe.gl / react-globe.gl

Globe.gl is a high-level wrapper around Three.js designed specifically for interactive globes with minimal code.

**Website:** [globe.gl](https://globe.gl) | **Repo:** [github.com/vasturiano/globe.gl](https://github.com/vasturiano/globe.gl)

```bash
npm install globe.gl         # vanilla
npm install react-globe.gl   # React
```

### Quick Globe with Multiple Layer Types

```javascript
import Globe from 'globe.gl';

const globe = Globe()
  .globeImageUrl('//unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
  .bumpImageUrl('//unpkg.com/three-globe/example/img/earth-topology.png')
  .backgroundImageUrl('//unpkg.com/three-globe/example/img/night-sky.png')
  // Points
  .pointsData(earthquakes)
  .pointLat('lat').pointLng('lng')
  .pointAltitude(d => d.magnitude / 10)
  .pointColor(d => magnitudeColor(d.magnitude))
  // Arcs (flight routes)
  .arcsData(routes)
  .arcStartLat('originLat').arcStartLng('originLng')
  .arcEndLat('destLat').arcEndLng('destLng')
  .arcColor(() => ['rgba(0,255,255,0.5)', 'rgba(255,0,255,0.5)'])
  .arcDashLength(0.4).arcDashGap(0.2).arcDashAnimateTime(2000)
  // Hexbin
  .hexBinPointsData(events)
  .hexBinPointLat('lat').hexBinPointLng('lng')
  .hexBinPointWeight('count')
  .hexAltitude(d => d.sumWeight / 5000)
  .hexTopColor(d => weightColor(d.sumWeight))
  (document.getElementById('globe'));
```

### Limitations

- No terrain (flat sphere)
- No tile streaming (all data must be loaded upfront)
- No true 3D model loading at scale
- Performance degrades at ~10K points (Three.js InstancedMesh is not used internally for points)
- No time system
- No coordinate system beyond WGS84 lat/lon

### When to Graduate to CesiumJS

Graduate when you need: 3D Tiles, photogrammetry, terrain-clamped data, time-dynamic simulation, or >10K interactive features.

### Three.js Escape Hatch

```javascript
// Access the underlying Three.js scene for full customization
const threeScene = globe.scene();
const threeCamera = globe.camera();
const threeRenderer = globe.renderer();

// Add custom Three.js objects directly
const geometry = new THREE.TorusGeometry(101, 1, 16, 100);
const material = new THREE.MeshBasicMaterial({ color: 0xff6600, wireframe: true });
threeScene.add(new THREE.Mesh(geometry, material));

// Custom animation loop (careful not to conflict with globe's internal loop)
globe.onZoom(() => {
  myCustomMesh.rotation.y += 0.01;
});
```

---

## Comprehensive Comparison Matrix

| Dimension | CesiumJS | deck.gl | Three.js | MapLibre 3D | Globe.gl |
|-----------|----------|---------|----------|-------------|----------|
| True WGS84 globe | Yes | Via GlobeView | Manual | v4+ approx. | Yes (sphere) |
| Terrain (quantized-mesh) | Yes | TerrainLayer | three-geo | raster-dem | No |
| 3D Tiles 1.1 | Native | Tile3DLayer | iTowns | No | No |
| Vector tiles (MVT) | Limited | MVTLayer | No native | Native | No |
| Time animation | Native CZML | Manual | Manual | Manual | Manual |
| Point clouds (LAS) | Native | Yes | Manual | No | No |
| WebGPU | No (2025) | Opt-in v9+ | Via WebGPURenderer | No | No |
| Mobile performance | Moderate | Good | Variable | Excellent | Moderate |
| Bundle size (gzip) | ~3 MB | ~300 KB | ~150 KB | ~220 KB | ~200 KB |
| TypeScript | Full | Full | Full | Full | Partial |
| React wrapper | Resium | Native | R3F | react-map-gl | react-globe.gl |
| Learning curve | Steep | Moderate | Steep | Low-Med | Very Low |
| Corporate backing | Cesium (NVIDIA) | Foursquare/OpenJS | No (community) | MapLibre Org | No (Vasturiano) |
| License | Apache 2.0 | MIT | MIT | BSD-3 | MIT |
| Max points (60fps) | ~100K entities | 10M+ (GPU) | 1M+ (instanced) | ~500K extruded | ~10K |

---

## Cross-Platform & Migration

### CesiumJS → deck.gl

**You gain:**
- 10x-50x higher point throughput (GPU attribute management)
- Smaller bundle (~300 KB vs 3 MB)
- Better aggregation layers (H3, GPU grid, heatmap)
- Easier React/Vue integration

**You lose:**
- True globe with WGS84 ellipsoid (deck.gl GlobeView is approximate)
- Native 3D Tiles support (Tile3DLayer exists but is less mature)
- CZML and time system
- Terrain (only via TerrainLayer, not quantized-mesh quality)
- Camera frustum culling aware of globe curvature

**When to do it:** Your app is analytics-focused (dashboards, heatmaps, flow viz) and does not need photogrammetry or time-dynamic simulation.

### deck.gl + MapLibre → CesiumJS

**Trigger points:**
- Client demands Google Photorealistic 3D Tiles or Cesium OSM Buildings
- Need to display LiDAR point clouds (PNTS / LAZ)
- Need time-dynamic simulation with CZML playback
- Terrain accuracy matters (quantized-mesh vs raster-dem)
- Project needs to display CityGML/IFC assets as 3D Tiles

**Migration path:** Keep deck.gl for analytical overlays (the `CesiumWidget` + deck.gl canvas sync pattern described above), move geospatial foundation to CesiumJS.

### Desktop (Unity/Unreal) → Web

| Concern | CesiumJS | Cesium for Unity/Unreal |
|---------|----------|------------------------|
| 3D Tiles streaming | Yes | Yes (same protocol) |
| Photorealistic tiles | Yes | Yes |
| Physics engine | No | Yes (native) |
| VR/AR headsets | WebXR (limited) | Full native XR |
| Deployment | Zero install | App download |
| Performance | WebGL GPU limits | Full GPU |
| Collaboration/sharing | URL shareable | Not without server |

For enterprise digital twin products where users need to install a client, Cesium for Unreal offers superior rendering quality. For web-first tools where sharing a URL matters, CesiumJS is the choice.

---

## Performance Optimization Cheat Sheet

| Library | Optimization | Impact |
|---------|-------------|--------|
| CesiumJS | `requestRenderMode = true` | Eliminates idle GPU usage |
| CesiumJS | Primitives over Entities for > 500 features | 5-10x draw call reduction |
| CesiumJS | `PointPrimitiveCollection` over point entities | 100x for point markers |
| CesiumJS | `maximumMemoryUsage = 256` on mobile | Prevents tab crashes |
| CesiumJS | `tileset.debugColorTiles = true` → diagnose LOD issues | Optimization insight |
| CesiumJS | Worker-based tile decode (already built-in) | Keep decode off main thread |
| CesiumJS | `scene.fog.enabled = true` | Culls distant tiles |
| deck.gl | Binary data format (`Float32Array` attributes) | 2-3x update speed |
| deck.gl | Pre-aggregate to H3 server-side → H3HexagonLayer | 10x vs HexagonLayer |
| deck.gl | `updateTriggers` scoped to changed props only | Avoid full attribute re-upload |
| deck.gl | `_subLayerProps` to configure GeoJsonLayer internals | Avoid composite layer overhead |
| deck.gl | WebGPU device (v9+) | 20-50% throughput on supported browsers |
| Three.js | `InstancedMesh` for repeated geometry | Reduces draw calls 100x |
| Three.js | `dispose()` geometry/material/texture on remove | Prevents GPU memory leak |
| Three.js | Offscreen canvas + Worker for heavy scenes | Keeps main thread free |
| MapLibre | `map.setMaxZoom(18)` to limit tile requests | Reduces bandwidth |
| MapLibre | `raster-dem` tile size 512 instead of 256 | 4x fewer terrain requests |
| MapLibre | `fill-extrusion` base/height from tile attributes | Zero CPU geometry |

---

## Advanced Dark Arts

### CesiumJS: Request Render Mode

The single highest-impact optimization for apps that are not continuously animating:

```javascript
viewer.scene.requestRenderMode = true;
viewer.scene.maximumRenderTimeChange = Infinity;  // Only render on explicit request

// Request a render when your data changes
viewer.scene.requestRender();

// When doing continuous animation (e.g., time playback), disable temporarily
viewer.scene.requestRenderMode = false;
```

This can reduce GPU usage from 100% to near 0% when the user is not interacting.

### CesiumJS: Ground Primitive for Terrain Draping

```javascript
// Entity polygons draped on terrain (easy but slow)
viewer.entities.add({
  polygon: {
    hierarchy: positions,
    material: Cesium.Color.RED.withAlpha(0.5),
    classificationType: Cesium.ClassificationType.TERRAIN
  }
});

// GroundPrimitive (fast, production-grade)
const geometry = new Cesium.PolygonGeometry({
  polygonHierarchy: new Cesium.PolygonHierarchy(
    Cesium.Cartesian3.fromDegreesArray(coords)
  )
});

const groundPrim = new Cesium.GroundPrimitive({
  geometryInstances: new Cesium.GeometryInstance({
    geometry,
    attributes: {
      color: Cesium.ColorGeometryInstanceAttribute.fromColor(
        Cesium.Color.RED.withAlpha(0.5)
      )
    }
  }),
  classificationType: Cesium.ClassificationType.TERRAIN
});

viewer.scene.groundPrimitives.add(groundPrim);
```

### deck.gl: Extensions for Custom Shader Injection

```javascript
import { DataFilterExtension } from '@deck.gl/extensions';
import { CollisionFilterExtension } from '@deck.gl/extensions';

// Data filter: GPU-side filter without layer rebuild
new ScatterplotLayer({
  data: allPoints,
  extensions: [new DataFilterExtension({ filterSize: 2 })],
  getFilterValue: d => [d.year, d.magnitude],
  filterRange: [[2020, 2024], [5.0, 10.0]],   // Dynamically filter by year & magnitude
  // Changing filterRange does NOT re-upload attribute buffers -- pure uniform change
  getFillColor: [255, 0, 0]
});

// Collision filter: GPU-based label collision detection
new TextLayer({
  data: cities,
  extensions: [new CollisionFilterExtension()],
  collisionEnabled: true,
  getCollisionPriority: d => d.population,
  getText: d => d.name,
  getSize: 14
});
```

### deck.gl: _subLayerProps for Composite Layer Internals

```javascript
// GeoJsonLayer renders several sub-layers internally.
// _subLayerProps lets you pass props directly to those sub-layers.
new GeoJsonLayer({
  data: geojson,
  filled: true,
  stroked: true,
  extruded: true,
  _subLayerProps: {
    // Target the polygon fill sub-layer
    'polygons-fill': {
      material: { ambient: 0.3, diffuse: 0.8, shininess: 64 }
    },
    // Target the line extrusion sub-layer
    'polygons-stroke': {
      widthMinPixels: 1,
      getColor: [255, 255, 0]
    },
    // Target the point sub-layer
    'points': {
      radiusMinPixels: 4
    }
  }
});
```

### Three.js: InstancedBufferGeometry for Million-Point Terrain Scatter

```javascript
import * as THREE from 'three';

// Base geometry (a single tree/rock/bush)
const baseGeometry = new THREE.ConeGeometry(0.3, 1.0, 5);

// Per-instance data
const count = 1000000;
const offsets = new Float32Array(count * 3);
const scales = new Float32Array(count);
const colors = new Float32Array(count * 3);

for (let i = 0; i < count; i++) {
  // Scatter on terrain (pre-computed positions from DEM sampling)
  offsets[i * 3]     = terrainPoints[i].x;
  offsets[i * 3 + 1] = terrainPoints[i].y;
  offsets[i * 3 + 2] = terrainPoints[i].z;
  scales[i] = 0.5 + Math.random() * 1.5;
  // Color variation
  const g = 0.3 + Math.random() * 0.4;
  colors[i * 3] = 0.1; colors[i * 3 + 1] = g; colors[i * 3 + 2] = 0.05;
}

const instancedGeometry = new THREE.InstancedBufferGeometry();
instancedGeometry.index = baseGeometry.index;
instancedGeometry.attributes = { ...baseGeometry.attributes };

instancedGeometry.setAttribute('offset',
  new THREE.InstancedBufferAttribute(offsets, 3));
instancedGeometry.setAttribute('instanceScale',
  new THREE.InstancedBufferAttribute(scales, 1));
instancedGeometry.setAttribute('instanceColor',
  new THREE.InstancedBufferAttribute(colors, 3));

const material = new THREE.RawShaderMaterial({
  vertexShader: `
    precision highp float;
    attribute vec3 position;
    attribute vec3 normal;
    attribute vec3 offset;
    attribute float instanceScale;
    attribute vec3 instanceColor;
    uniform mat4 modelViewMatrix;
    uniform mat4 projectionMatrix;
    varying vec3 vColor;
    varying vec3 vNormal;
    void main() {
      vColor = instanceColor;
      vNormal = normal;
      vec3 transformed = position * instanceScale + offset;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(transformed, 1.0);
    }
  `,
  fragmentShader: `
    precision highp float;
    varying vec3 vColor;
    varying vec3 vNormal;
    void main() {
      float light = dot(normalize(vNormal), normalize(vec3(1.0, 2.0, 1.0))) * 0.5 + 0.5;
      gl_FragColor = vec4(vColor * light, 1.0);
    }
  `
});

scene.add(new THREE.Mesh(instancedGeometry, material));
```

### MapLibre: Custom Layer Wind Particles

```javascript
// Wind particle visualization as a MapLibre custom layer
const particleLayer = {
  id: 'wind-particles',
  type: 'custom',

  onAdd(map, gl) {
    // Compile shaders
    const vs = `
      attribute vec2 a_pos;
      uniform mat4 u_matrix;
      uniform float u_size;
      void main() {
        gl_Position = u_matrix * vec4(a_pos, 0.0, 1.0);
        gl_PointSize = u_size;
      }
    `;
    const fs = `
      precision mediump float;
      uniform vec4 u_color;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        if (d > 0.5) discard;
        gl_FragColor = u_color * (1.0 - d * 2.0);
      }
    `;

    this.program = createProgram(gl, vs, fs);
    this.particles = initParticles(5000, windBounds);
    this.buffer = gl.createBuffer();
    this.map = map;
  },

  render(gl, matrix) {
    // Update particle positions along wind vectors
    advectParticles(this.particles, windField, 0.5);

    // Convert lon/lat to tile coordinates
    const positions = toMercatorCoords(this.particles);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(positions), gl.DYNAMIC_DRAW);

    gl.useProgram(this.program);
    gl.uniformMatrix4fv(gl.getUniformLocation(this.program, 'u_matrix'), false, matrix);
    gl.uniform1f(gl.getUniformLocation(this.program, 'u_size'), 3.0);
    gl.uniform4fv(gl.getUniformLocation(this.program, 'u_color'), [0.0, 0.8, 1.0, 0.7]);

    const posAttr = gl.getAttribLocation(this.program, 'a_pos');
    gl.enableVertexAttribArray(posAttr);
    gl.vertexAttribPointer(posAttr, 2, gl.FLOAT, false, 0, 0);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.drawArrays(gl.POINTS, 0, this.particles.length);

    this.map.triggerRepaint();  // Continuous animation loop
  }
};

map.addLayer(particleLayer, 'waterway-label');
```

---

## Install Summary

```bash
# CesiumJS
npm install cesium

# deck.gl (modular)
npm install @deck.gl/core @deck.gl/layers @deck.gl/geo-layers @deck.gl/aggregation-layers @deck.gl/extensions

# deck.gl MapLibre integration
npm install @deck.gl/maplibre

# Three.js + geo helpers
npm install three three-geo three-geojson-geometry

# MapLibre GL JS
npm install maplibre-gl

# Globe.gl
npm install globe.gl

# React wrappers
npm install resium                # CesiumJS + React
npm install react-globe.gl        # Globe.gl + React
npm install react-map-gl          # MapLibre + React (deck.gl ready)

# Python deck.gl
pip install pydeck
```
