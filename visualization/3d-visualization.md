# 3D Visualization

> Techniques and tools for rendering geospatial data in three dimensions -- terrain, buildings, point clouds, volumetric data, digital twins, and immersive experiences.

> **Quick Picks**
> - **SOTA**: [CesiumJS](https://cesium.com) + 3D Tiles 1.1 -- full globe with terrain, buildings, point clouds, and Google Photorealistic Tiles
> - **Free Best**: [deck.gl](https://deck.gl) -- GPU-accelerated TerrainLayer, PointCloudLayer, and custom 3D layers
> - **Fastest Setup**: [Potree](https://potree.github.io) -- drop LiDAR data in and view billions of points immediately
> - **Game Engine**: [Cesium for Unreal](https://cesium.com/platform/cesium-for-unreal/) -- photorealistic globe inside Unreal Engine 5
> - **Emerging**: [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) -- real-time novel view synthesis from photographs

---

## Table of Contents

1. [3D Terrain Rendering](#3d-terrain-rendering)
2. [CesiumJS + 3D Tiles Ecosystem](#cesiumjs--3d-tiles-ecosystem)
3. [Urban Digital Twins](#urban-digital-twins)
4. [Point Cloud Visualization](#point-cloud-visualization)
5. [Volumetric and Scientific 3D](#volumetric-and-scientific-3d)
6. [Photogrammetry and Reality Capture](#photogrammetry-and-reality-capture)
7. [Game Engines for Geo](#game-engines-for-geo)
8. [WebXR and Immersive Geo](#webxr-and-immersive-geo)
9. [Performance and Hardware](#performance-and-hardware)
10. [Tool Comparison Matrix](#tool-comparison-matrix)

---

## 3D Terrain Rendering

Visualizing elevation data as 3D surfaces for topographic analysis, flight simulation, viewshed computation, and cartographic context.

> Cross-reference: [Elevation and Terrain Data Sources](../data-sources/elevation-terrain.md) for detailed acquisition guides.

### DEM Sources

| Source | Resolution | Coverage | Format | Access |
|--------|-----------|----------|--------|--------|
| **Copernicus GLO-30** | 30 m | Global | GeoTIFF, COG | Free (ESA Copernicus) |
| **SRTM v3** | 30 m / 90 m | 60N-56S | GeoTIFF, HGT | Free (NASA EarthData) |
| **ALOS World 3D** | 30 m | Global | GeoTIFF | Free (JAXA) |
| **FABDEM** | 30 m (forest/building removed) | Global | GeoTIFF | Free (University of Bristol) |
| **LiDAR-derived** | 0.5-2 m | Local/regional | LAS, LAZ, GeoTIFF | Varies (USGS 3DEP, national programs) |
| **TanDEM-X** | 12 m / 30 m / 90 m | Global | GeoTIFF | Commercial (DLR) / Free at 90m |
| **ASTER GDEM v3** | 30 m | 83N-83S | GeoTIFF | Free (NASA/METI) |

**Dark Arts: DEM Selection Strategy**

- For global web terrain, Copernicus GLO-30 is the current best free option -- it supersedes SRTM in both coverage and accuracy at polar latitudes.
- FABDEM removes forest canopy and building artifacts from Copernicus, making it ideal for bare-earth hydrological modeling.
- For sub-meter terrain, nothing replaces airborne LiDAR. Check USGS 3DEP (USA), Environment Agency (UK), or your national mapping agency.
- When serving terrain tiles, convert to Cloud Optimized GeoTIFF (COG) first -- this enables HTTP range requests and eliminates the need for a tile server.

### Hillshade Algorithms

Hillshading is the foundation of terrain perception in both 2D and 3D contexts. The choice of algorithm affects realism and computational cost.

| Algorithm | Method | Quality | Speed | Best For |
|-----------|--------|---------|-------|----------|
| **Horn (1981)** | 8-neighbor weighted gradient | Good | Fast | Standard hillshading, GDAL default |
| **Zevenbergen-Thorne** | 4-neighbor gradient | Moderate | Fastest | Quick previews |
| **Blinn (multi-directional)** | Multiple light azimuths blended | Excellent | Moderate | Cartographic-quality terrain |
| **MDOW** (Multi-Directional Oblique Weighted) | 6 directions, weighted blend | Excellent | Moderate | Swiss-style cartography |
| **Ambient Occlusion** | Sky visibility per cell | Photorealistic | Slow | 3D rendering, crevice detail |

```bash
# Standard Horn hillshade with GDAL
gdaldem hillshade dem.tif hillshade.tif -z 2.0 -az 315 -alt 45

# Multi-directional hillshade (GDAL 2.2+)
gdaldem hillshade dem.tif hillshade_multi.tif -z 2.0 -multidirectional

# Generate slope for hypsometric tinting
gdaldem slope dem.tif slope.tif -p
gdaldem color-relief dem.tif color_ramp.txt hypsometric.tif
```

### Vertical Exaggeration and Hypsometric Tinting

Vertical exaggeration amplifies the Z-axis relative to X/Y, making subtle terrain features visible. A factor of 1.5-2.0 is standard for web maps; 3.0-5.0 is useful for flat terrain like the Netherlands or coastal plains.

Hypsometric tinting maps elevation to color ramps. Classic schemes include green-to-brown-to-white (vegetation to rock to snow) or continuous perceptual palettes (viridis, cividis) for scientific accuracy.

```javascript
// MapLibre: exaggeration is set on the terrain source
map.setTerrain({ source: 'dem-source', exaggeration: 2.0 });

// CesiumJS: vertical exaggeration on the globe
viewer.scene.verticalExaggeration = 2.0;
// Optional: set the relative height from which exaggeration is measured
viewer.scene.verticalExaggerationRelativeHeight = 0.0;
```

### Cross-Section / Profile Visualization

Elevation profiles are essential for trail planning, pipeline routing, and geological analysis.

```javascript
// Cesium: sample terrain heights along a polyline
import { sampleTerrainMostDetailed, Cartographic } from 'cesium';

const positions = [
  Cartographic.fromDegrees(86.9, 27.9),
  Cartographic.fromDegrees(86.95, 28.0),
  // ... more points along profile line
];

const updatedPositions = await sampleTerrainMostDetailed(
  viewer.terrainProvider,
  positions
);

// updatedPositions now contain .height values
// Plot with D3.js, Chart.js, or any charting library
const profileData = updatedPositions.map((pos, i) => ({
  distance: i * segmentLength,  // compute cumulative distance
  elevation: pos.height
}));
```

### CesiumJS Terrain -- Full Setup

```javascript
import {
  Viewer, createWorldTerrainAsync, Cartesian3,
  CesiumTerrainProvider, IonResource, EllipsoidTerrainProvider
} from 'cesium';

// Option 1: Cesium World Terrain (hosted on Cesium Ion)
const viewer = new Viewer('cesiumContainer', {
  terrain: await createWorldTerrainAsync({
    requestVertexNormals: true,   // enable lighting
    requestWaterMask: true        // enable water effects
  })
});

// Option 2: Custom quantized-mesh terrain server
const customTerrain = await CesiumTerrainProvider.fromUrl(
  'https://your-server.com/terrain-tiles',
  {
    requestVertexNormals: true,
    credit: 'Custom Terrain Provider'
  }
);
viewer.terrainProvider = customTerrain;

// Option 3: No terrain (flat ellipsoid) -- useful for 2.5D building scenes
viewer.terrainProvider = new EllipsoidTerrainProvider();

// Fly to a viewpoint
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(86.925, 27.988, 30000),  // Everest
  orientation: { heading: 0.0, pitch: -0.5, roll: 0.0 }
});
```

### MapLibre 3D Terrain

```javascript
const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [11.39085, 47.27574],  // Alps
  zoom: 12,
  pitch: 60,
  bearing: -17
});

map.on('load', () => {
  // Add DEM source
  map.addSource('terrain', {
    type: 'raster-dem',
    url: 'https://demotiles.maplibre.org/terrain-tiles/tiles.json',
    tileSize: 256
  });

  // Enable 3D terrain
  map.setTerrain({ source: 'terrain', exaggeration: 1.5 });

  // Add hillshade layer for visual depth
  map.addLayer({
    id: 'hillshade',
    type: 'hillshade',
    source: 'terrain',
    paint: {
      'hillshade-illumination-direction': 315,
      'hillshade-shadow-color': '#473B24',
      'hillshade-highlight-color': '#FFFFFF',
      'hillshade-exaggeration': 0.5
    }
  });

  // Add sky layer for atmosphere
  map.addLayer({
    id: 'sky',
    type: 'sky',
    paint: {
      'sky-type': 'atmosphere',
      'sky-atmosphere-sun': [0.0, 90.0],
      'sky-atmosphere-sun-intensity': 15
    }
  });
});
```

### deck.gl TerrainLayer

```javascript
import { Deck } from '@deck.gl/core';
import { TerrainLayer } from '@deck.gl/geo-layers';

const ELEVATION_DECODER = {
  rScaler: 256,
  gScaler: 1,
  bScaler: 1 / 256,
  offset: -32768
};

const terrain = new TerrainLayer({
  id: 'terrain',
  minZoom: 0,
  maxZoom: 15,
  elevationDecoder: ELEVATION_DECODER,
  elevationData: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  texture: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  meshMaxError: 4.0,        // higher = fewer triangles = faster
  wireframe: false,
  color: [255, 255, 255],   // modulate texture color
  operation: 'terrain+draw'
});

new Deck({
  initialViewState: { longitude: 86.9, latitude: 27.9, zoom: 10, pitch: 55 },
  controller: true,
  layers: [terrain]
});
```

### Three.js Custom Terrain Shader

For complete control over terrain rendering, Three.js with custom shaders is the most flexible option.

```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Load DEM as a displacement map
const textureLoader = new THREE.TextureLoader();
const demTexture = textureLoader.load('dem_heightmap.png');
const satelliteTexture = textureLoader.load('satellite_imagery.jpg');

// Custom shader with hypsometric tinting and hillshade
const terrainMaterial = new THREE.ShaderMaterial({
  uniforms: {
    displacementMap: { value: demTexture },
    diffuseMap: { value: satelliteTexture },
    displacementScale: { value: 500.0 },   // vertical exaggeration
    sunDirection: { value: new THREE.Vector3(1, 1, 1).normalize() }
  },
  vertexShader: `
    uniform sampler2D displacementMap;
    uniform float displacementScale;
    varying vec2 vUv;
    varying vec3 vNormal;
    varying vec3 vPosition;
    void main() {
      vUv = uv;
      float height = texture2D(displacementMap, uv).r * displacementScale;
      vec3 displaced = position + normal * height;
      vPosition = displaced;
      vNormal = normal;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(displaced, 1.0);
    }
  `,
  fragmentShader: `
    uniform sampler2D diffuseMap;
    uniform vec3 sunDirection;
    varying vec2 vUv;
    varying vec3 vNormal;
    void main() {
      vec3 color = texture2D(diffuseMap, vUv).rgb;
      float light = max(dot(normalize(vNormal), sunDirection), 0.15);
      gl_FragColor = vec4(color * light, 1.0);
    }
  `
});

const geometry = new THREE.PlaneGeometry(1000, 1000, 512, 512);
const mesh = new THREE.Mesh(geometry, terrainMaterial);
mesh.rotation.x = -Math.PI / 2;
scene.add(mesh);
```

---

## CesiumJS + 3D Tiles Ecosystem

CesiumJS is the most comprehensive platform for 3D geospatial visualization. Together with the 3D Tiles open standard (OGC Community Standard), it provides a complete pipeline for streaming and rendering massive heterogeneous 3D geospatial content on the web.

> Cross-reference: [3D Mapping Libraries](../js-bindbox/3d-mapping.md) for broader library comparisons.

### 3D Tiles 1.1 Specification

3D Tiles 1.1 is a major evolution that unifies all content types under glTF and adds rich metadata capabilities.

**Key improvements over 1.0:**

| Feature | 3D Tiles 1.0 | 3D Tiles 1.1 |
|---------|-------------|-------------|
| Content format | b3dm, i3dm, pnts, cmpt | glTF (replaces all legacy formats) |
| Metadata | Batch table (JSON) | EXT_structural_metadata (typed, per-vertex/texel) |
| Tiling scheme | Explicit tileset.json | Implicit tiling (octree/quadtree templates) |
| Multiple contents | Not supported | Multiple contents per tile |
| Metadata granularity | Per-feature | Per-tile, per-feature, per-vertex, per-texel |

### Content Types Reference

| Type | Extension | Content | Use Case | Status in 1.1 |
|------|-----------|---------|----------|---------------|
| Batched 3D Model | .b3dm | Textured meshes (buildings) | Urban models, BIM | Legacy (use glTF) |
| Instanced 3D Model | .i3dm | Repeated models (trees, lights) | Vegetation, infrastructure | Legacy (use glTF + EXT_mesh_gpu_instancing) |
| Point Cloud | .pnts | Colored points | LiDAR, photogrammetry | Legacy (use glTF) |
| Composite | .cmpt | Mix of above | Complex scenes | Legacy (use multiple contents) |
| **glTF** | .glb/.gltf | All content via glTF | **All use cases (1.1 standard)** | Current |

### Implicit Tiling

Implicit tiling eliminates the need for enormous tileset.json files by defining a template URL pattern and a tile availability bitstream.

```json
{
  "root": {
    "boundingVolume": {
      "region": [-1.318, 0.698, -1.319, 0.699, 0, 100]
    },
    "geometricError": 5000,
    "refine": "ADD",
    "content": {
      "uri": "tiles/{level}/{x}/{y}.glb"
    },
    "implicitTiling": {
      "subdivisionScheme": "QUADTREE",
      "availableLevels": 15,
      "subtrees": {
        "uri": "subtrees/{level}/{x}/{y}.json"
      },
      "subtreeLevels": 4
    }
  }
}
```

### Google Photorealistic 3D Tiles

Google's Photorealistic 3D Tiles provide photogrammetric mesh coverage of thousands of cities worldwide.

```javascript
import {
  Viewer, Cesium3DTileset, createGooglePhotorealistic3DTileset,
  Cartesian3, ScreenSpaceEventHandler, ScreenSpaceEventType,
  defined
} from 'cesium';

// Method 1: Via Cesium Ion (recommended -- handles attribution)
const viewer = new Viewer('cesiumContainer', {
  globe: false  // disable default globe when using Google 3D Tiles
});
const googleTileset = await Cesium3DTileset.fromIonAssetId(2275207);
viewer.scene.primitives.add(googleTileset);

// Method 2: Direct Google Maps API key
const directTileset = await Cesium3DTileset.fromUrl(
  `https://tile.googleapis.com/v1/3dtiles/root.json?key=${GOOGLE_API_KEY}`
);

// Fly to a city
viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(-74.006, 40.7128, 800),
  orientation: { heading: 0.3, pitch: -0.4, roll: 0 }
});
```

### Cesium ion Asset Management

Cesium ion is the cloud platform for tiling, hosting, and streaming 3D geospatial content.

```javascript
// Upload and tile assets programmatically via REST API
const response = await fetch('https://api.cesium.com/v1/assets', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${ION_ACCESS_TOKEN}`
  },
  body: JSON.stringify({
    name: 'My City Model',
    type: '3DTILES',
    options: {
      sourceType: 'CITYGML',
      geometricErrorScaling: 1.0
    }
  })
});

// The response contains upload instructions and the asset ID
const { assetMetadata, uploadLocation } = await response.json();
```

### Custom Styling with Cesium3DTileStyle

The Cesium3DTileStyle engine supports a rich expression language for data-driven visual encoding.

```javascript
import { Cesium3DTileStyle, Color } from 'cesium';

// Color by building height with transparency
tileset.style = new Cesium3DTileStyle({
  color: {
    conditions: [
      ['${height} > 200', "color('#FF0000', 0.9)"],   // Skyscrapers: red
      ['${height} > 100', "color('#FF8800', 0.8)"],   // Tall: orange
      ['${height} > 50',  "color('#FFCC00', 0.7)"],   // Medium: yellow
      ['${height} > 20',  "color('#88CC00', 0.6)"],   // Low-rise: green
      ['true',            "color('#CCCCCC', 0.5)"]     // Default: gray
    ]
  },
  show: '${height} > 5',     // hide features below 5m
  meta: {
    description: "'Building: ' + ${name} + ', Height: ' + ${height} + 'm'"
  }
});

// Dynamic style update based on time
function updateStyleByYear(year) {
  tileset.style = new Cesium3DTileStyle({
    show: `\${yearBuilt} <= ${year}`,
    color: {
      conditions: [
        [`\${yearBuilt} > ${year - 5}`, "color('#00FF00')"],  // Recent: green
        [`\${yearBuilt} > ${year - 20}`, "color('#FFFF00')"], // Aging: yellow
        ['true', "color('#FF8800')"]                           // Old: orange
      ]
    }
  });
}
```

### Performance Tuning for 3D Tiles

```javascript
// Tile loading budget
tileset.maximumScreenSpaceError = 16;     // default 16; lower = higher quality
tileset.maximumMemoryUsage = 512;         // MB of GPU memory for tiles
tileset.preloadWhenHidden = false;        // don't load tiles for off-screen cameras
tileset.preferLeaves = false;             // load intermediate tiles for progressive refinement
tileset.skipLevelOfDetail = true;         // skip loading intermediate levels

// Scene-level performance
viewer.scene.requestRenderMode = true;    // only render when scene changes
viewer.scene.maximumRenderTimeChange = 0.0; // render every frame when animating
viewer.scene.fog.enabled = true;          // reduce far-tile loading
viewer.scene.fog.density = 2.0e-4;

// Monitor performance
tileset.tileLoad.addEventListener((tile) => {
  console.log(`Loaded tile: ${tile.content.url}, triangles: ${tile.content.trianglesLength}`);
});

console.log('Tiles loaded:', tileset.tilesLoaded);
console.log('GPU memory used:', tileset.totalMemoryUsageInBytes / 1e6, 'MB');
```

**Dark Arts: 3D Tiles Performance**

- `maximumScreenSpaceError` is your primary quality/performance knob. Values of 2-4 produce near-pixel-perfect rendering but crush the GPU. Values of 16-32 are good for overviews.
- Set `tileset.maximumMemoryUsage` to about 60-70% of available GPU VRAM. CesiumJS evicts least-recently-used tiles when this budget is exceeded.
- For time-critical applications (VR, 60fps requirement), enable `skipLevelOfDetail` and increase `maximumScreenSpaceError` until you hit your frame budget.
- When displaying Google Photorealistic 3D Tiles alongside your own data, disable `globe.show` to avoid z-fighting with the terrain.
- Use `tileset.debugShowBoundingVolume = true` and `tileset.debugShowContentBoundingVolume = true` during development to visualize the tile hierarchy.

---

## Urban Digital Twins

Urban digital twins combine 3D city models with real-time sensor data, simulation results, and analytical overlays to create living representations of cities.

### City Model Standards

| Standard | Description | LOD Levels | Strengths |
|----------|-------------|------------|-----------|
| **CityGML 2.0/3.0** | OGC XML-based city model exchange | LOD 0-4 | Rich semantics, multi-resolution |
| **CityJSON** | JSON encoding of CityGML | LOD 0-3 | Compact, web-friendly |
| **3D Tiles** | OGC streaming format | N/A (continuous LOD) | Massive scale, web streaming |
| **IFC (BIM)** | Building Information Model | N/A | Interior detail, building lifecycle |
| **OpenStreetMap 3D** | Community-contributed heights | Simple extrusion | Global coverage, free |

### CityGML LOD Levels

| LOD | Geometry | Typical Use | Example |
|-----|----------|-------------|---------|
| LOD 0 | 2.5D footprint (ground + roof height) | Regional overview | Building footprints with height |
| LOD 1 | Block model (prismatic extrusion) | City-scale planning | Box-shaped buildings |
| LOD 2 | Roof structures, facade detail | Solar analysis, shadow studies | Pitched roofs, dormers |
| LOD 3 | Architectural detail (windows, doors, balconies) | Detailed urban planning | Facade textures |
| LOD 4 | Interior rooms, furniture | Indoor navigation, facility management | BIM integration |

### Real-Time IoT Sensor Overlay

```javascript
// CesiumJS: overlay real-time sensor data on 3D buildings
import { Cesium3DTileset, Cesium3DTileStyle, Entity, Color, CallbackProperty } from 'cesium';

// Load city model
const cityModel = await Cesium3DTileset.fromIonAssetId(YOUR_CITY_ASSET_ID);
viewer.scene.primitives.add(cityModel);

// WebSocket connection for real-time sensor data
const sensorData = new Map();
const ws = new WebSocket('wss://iot-api.your-city.com/sensors');

ws.onmessage = (event) => {
  const reading = JSON.parse(event.data);
  sensorData.set(reading.buildingId, reading);

  // Update building color based on energy consumption
  cityModel.style = new Cesium3DTileStyle({
    color: {
      evaluateColor: (feature) => {
        const sensor = sensorData.get(feature.getProperty('id'));
        if (!sensor) return Color.GRAY;
        const kwh = sensor.energyConsumption;
        // Green (low) to Red (high) gradient
        const t = Math.min(kwh / 1000, 1.0);
        return Color.fromHsl(0.33 * (1 - t), 0.8, 0.5, 0.8);
      }
    }
  });
};

// Add sensor point entities
function addSensorMarker(sensor) {
  viewer.entities.add({
    position: Cartesian3.fromDegrees(sensor.lon, sensor.lat, sensor.elevation),
    point: {
      pixelSize: 10,
      color: new CallbackProperty(() => {
        const data = sensorData.get(sensor.id);
        return data && data.alert ? Color.RED : Color.GREEN;
      }, false)
    },
    label: {
      text: new CallbackProperty(() => {
        const data = sensorData.get(sensor.id);
        return data ? `${data.temperature.toFixed(1)}C` : 'N/A';
      }, false),
      font: '12px sans-serif',
      verticalOrigin: 1  // BOTTOM
    }
  });
}
```

### Shadow Analysis

Shadow analysis computes the shadow footprint of buildings across time for solar exposure and urban comfort studies.

```javascript
// CesiumJS shadow analysis using built-in shadows
viewer.scene.globe.enableLighting = true;
viewer.shadows = true;
viewer.terrainShadows = Cesium.ShadowMode.ENABLED;

// Animate through a day to observe shadow patterns
const startTime = Cesium.JulianDate.fromIso8601('2025-06-21T06:00:00Z');
const endTime = Cesium.JulianDate.fromIso8601('2025-06-21T20:00:00Z');
viewer.clock.startTime = startTime;
viewer.clock.stopTime = endTime;
viewer.clock.currentTime = startTime;
viewer.clock.multiplier = 600;  // 10 minutes per second
viewer.clock.shouldAnimate = true;

// Custom shadow accumulation: render shadow maps at intervals
// and accumulate into a heat map texture
async function computeShadowAccumulation(extent, resolution, dateRange, interval) {
  const shadowGrid = new Float32Array(resolution * resolution);
  let time = dateRange.start.clone();

  while (Cesium.JulianDate.lessThan(time, dateRange.end)) {
    viewer.clock.currentTime = time;
    viewer.scene.render();  // force render at this time

    // Sample shadow state across the grid
    for (let row = 0; row < resolution; row++) {
      for (let col = 0; col < resolution; col++) {
        const lon = extent.west + (col / resolution) * (extent.east - extent.west);
        const lat = extent.south + (row / resolution) * (extent.north - extent.south);
        // Check if point is in shadow using scene.pickFromRay
        const position = Cartesian3.fromDegrees(lon, lat, 1.5); // pedestrian height
        const direction = Cesium.Simon1994PlanetaryPositions.computeSunPositionInEarthInertialFrame(time);
        // ... ray-cast logic
      }
    }
    Cesium.JulianDate.addMinutes(time, interval, time);
  }
  return shadowGrid;
}
```

### Indoor Mapping / BIM-GIS Integration

Integrating Building Information Models (IFC) with GIS context bridges the gap between architectural and geospatial domains.

**Pipeline: IFC to 3D Tiles**

```bash
# Convert IFC to glTF using IfcOpenShell
python3 -c "
import ifcopenshell
import ifcopenshell.geom

settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

ifc = ifcopenshell.open('building.ifc')
# Export geometry to glTF using ifcopenshell + pythreejs or trimesh
"

# Alternative: use open-source IFC to 3D Tiles pipeline
# https://github.com/nicokant/IFC-to-3DTiles
npx ifc-to-3dtiles --input building.ifc --output ./tiles/
```

### Digital Twin Platforms

| Platform | License | Strengths | BIM Support | IoT Support |
|----------|---------|-----------|-------------|-------------|
| **CesiumJS** (custom) | Open source | Full control, globe context | Via 3D Tiles | Custom WebSocket |
| **iTwin.js** (Bentley) | Open source | Native IFC/BIM, iModel.js | Excellent | iTwin IoT integrations |
| **ArcGIS Urban** | Commercial | Planning workflows, zoning | CityGML, BIM | ArcGIS Velocity |
| **Cesium for Unreal** | Free | Photorealistic rendering | Via 3D Tiles | Blueprint scripting |
| **Cesium for Unity** | Free | Cross-platform, AR/VR | Via 3D Tiles | C# scripting |
| **PLATEAU** (Japan MLIT) | Open data | Nationwide LOD2 city models | CityGML 2.0 | Demonstration projects |

### 3D Building Extrusion in MapLibre

For lightweight urban visualization without full 3D Tiles infrastructure.

```javascript
map.addLayer({
  id: '3d-buildings',
  source: 'openmaptiles',
  'source-layer': 'building',
  type: 'fill-extrusion',
  minzoom: 14,
  paint: {
    'fill-extrusion-color': [
      'interpolate', ['linear'], ['get', 'render_height'],
      0, '#e0e0e0',
      50, '#a0a0a0',
      100, '#606060',
      200, '#404040'
    ],
    'fill-extrusion-height': ['get', 'render_height'],
    'fill-extrusion-base': ['get', 'render_min_height'],
    'fill-extrusion-opacity': 0.85
  }
});

// Add ambient lighting for depth perception
map.setLight({
  anchor: 'viewport',
  color: '#ffffff',
  intensity: 0.4,
  position: [1.15, 210, 30]
});
```

---

## Point Cloud Visualization

Rendering large LiDAR, photogrammetry, or structured-light point clouds in the browser or desktop for inspection, measurement, and analysis.

### Potree -- The SOTA for Web Point Clouds

[Potree](https://potree.github.io) is the leading open-source web viewer for massive point clouds, capable of handling billions of points through hierarchical octree-based level-of-detail streaming.

- **Repository:** [github.com/potree/potree](https://github.com/potree/potree)
- **Capacity:** Billions of points (tested with 20+ billion)
- **Input:** LAS, LAZ, E57 (convert with PotreeConverter 2.x)
- **Features:** Measurement tools, cross-sections, classification coloring, elevation profiles, volume selection, annotations, EDL shading

```bash
# Install PotreeConverter 2.x
git clone https://github.com/potree/PotreeConverter.git
cd PotreeConverter && mkdir build && cd build
cmake .. && make -j$(nproc)

# Convert LAS to Potree octree format
./PotreeConverter input.las -o output_potree --generate-page index

# Convert multiple files with classification preservation
./PotreeConverter *.laz -o output_potree --generate-page index \
  --encoding DEFAULT --method poisson

# Serve the output directory
npx serve output_potree
```

### Potree Configuration and Customization

```javascript
// Advanced Potree viewer setup
const viewer = new Potree.Viewer(document.getElementById('potree_render_area'));

viewer.setEDLEnabled(true);           // Eye Dome Lighting for depth perception
viewer.setEDLRadius(1.4);
viewer.setEDLStrength(0.4);
viewer.setPointBudget(2_000_000);     // max visible points
viewer.setMinNodeSize(50);            // minimum pixel size of a node before it loads children
viewer.setBackground('gradient');      // 'skybox', 'gradient', 'black', 'white'

// Load point cloud
Potree.loadPointCloud('pointclouds/metadata.json', 'My LiDAR', (e) => {
  const pointcloud = e.pointcloud;
  viewer.scene.addPointCloud(pointcloud);

  // Color by classification
  const material = pointcloud.material;
  material.activeAttributeName = 'classification';
  material.size = 1;
  material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
  material.shape = Potree.PointShape.CIRCLE;

  // Custom classification colors (ASPRS standard)
  material.classificationScheme.set(2,  new THREE.Vector4(0.5, 0.3, 0.1, 1));  // Ground
  material.classificationScheme.set(3,  new THREE.Vector4(0.0, 0.6, 0.0, 1));  // Low Vegetation
  material.classificationScheme.set(4,  new THREE.Vector4(0.0, 0.8, 0.0, 1));  // Medium Vegetation
  material.classificationScheme.set(5,  new THREE.Vector4(0.0, 1.0, 0.0, 1));  // High Vegetation
  material.classificationScheme.set(6,  new THREE.Vector4(0.8, 0.0, 0.0, 1));  // Building
  material.classificationScheme.set(9,  new THREE.Vector4(0.0, 0.0, 0.8, 1));  // Water

  // Zoom to full extent
  viewer.fitToScreen();
});

// Add measurement tools
const measure = viewer.measuringTool;
const distanceMeasure = measure.startInsertion({ showDistances: true, closed: false });

// Cross-section tool
const profile = viewer.profileTool;
const profileWindow = viewer.profileWindowController;
```

### CesiumJS Point Clouds via 3D Tiles

```javascript
// Load point cloud as 3D Tiles (pnts or glTF with points)
const pointCloudTileset = await Cesium3DTileset.fromUrl(
  'https://your-server.com/pointcloud/tileset.json'
);
viewer.scene.primitives.add(pointCloudTileset);

// Style by classification
pointCloudTileset.style = new Cesium3DTileStyle({
  color: {
    conditions: [
      ['${classification} === 2', "color('#8B4513')"],  // Ground
      ['${classification} === 6', "color('#FF0000')"],  // Building
      ['${classification} === 5', "color('#00FF00')"],  // High Vegetation
      ['true', "color('#CCCCCC')"]
    ]
  },
  pointSize: '${classification} === 6 ? 3.0 : 2.0'
});

// Point cloud attenuation (size by distance) for better depth perception
pointCloudTileset.pointCloudShading.attenuation = true;
pointCloudTileset.pointCloudShading.maximumAttenuation = 4;
pointCloudTileset.pointCloudShading.eyeDomeLighting = true;
pointCloudTileset.pointCloudShading.eyeDomeLightingStrength = 1.5;
pointCloudTileset.pointCloudShading.eyeDomeLightingRadius = 1.0;
```

### deck.gl PointCloudLayer

```javascript
import { Deck, COORDINATE_SYSTEM } from '@deck.gl/core';
import { PointCloudLayer } from '@deck.gl/layers';

const pointCloud = new PointCloudLayer({
  id: 'lidar-points',
  data: '/data/pointcloud.bin',
  loaders: [LASLoader],                    // from @loaders.gl/las
  getPosition: d => [d.x, d.y, d.z],
  getColor: d => {
    // Color by intensity with classification override
    if (d.classification === 6) return [255, 0, 0];    // Building
    if (d.classification === 2) return [139, 69, 19];   // Ground
    const i = d.intensity / 65535 * 255;
    return [i, i, i];
  },
  getNormal: [0, 0, 1],
  pointSize: 2,
  coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
  coordinateOrigin: [-122.4, 37.78],
  sizeUnits: 'pixels',
  material: { ambient: 0.5, diffuse: 0.6, shininess: 32 },
  pickable: true,
  onHover: ({ object }) => {
    if (object) {
      tooltip.innerHTML = `Z: ${object.z.toFixed(2)}m, Class: ${object.classification}`;
    }
  }
});
```

### Three.js Point Cloud with Custom LOD

```javascript
import * as THREE from 'three';
import { LASLoader } from 'three/addons/loaders/LASLoader.js';

const loader = new LASLoader();
loader.load('scan.las', (geometry) => {
  // geometry.attributes: position, color, intensity, classification

  // Create point material with size attenuation
  const material = new THREE.PointsMaterial({
    size: 0.05,
    vertexColors: true,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.9,
    depthWrite: true
  });

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  // Manual LOD: subsample for far distances
  const lodGroup = new THREE.LOD();
  lodGroup.addLevel(points, 0);           // full resolution close up
  lodGroup.addLevel(subsample(geometry, 0.5), 50);   // 50% at 50m
  lodGroup.addLevel(subsample(geometry, 0.1), 200);  // 10% at 200m
  scene.add(lodGroup);
});

function subsample(geometry, fraction) {
  const count = Math.floor(geometry.attributes.position.count * fraction);
  const indices = new Uint32Array(count);
  const step = Math.floor(1 / fraction);
  for (let i = 0; i < count; i++) indices[i] = i * step;

  const subGeo = new THREE.BufferGeometry();
  // ... copy subsampled attributes
  return new THREE.Points(subGeo, material.clone());
}
```

### Eye Dome Lighting (EDL)

Eye Dome Lighting is a screen-space shading technique that enhances depth perception in point clouds without requiring surface normals. It is the single most impactful rendering technique for point cloud visualization.

**How it works:** For each pixel, compare its depth to neighboring pixels. Pixels surrounded by deeper neighbors appear brighter; pixels at depth discontinuities appear darker. This creates a subtle outline/shadow effect that makes structures visible without explicit lighting.

| Implementation | EDL Support | Configuration |
|----------------|-------------|---------------|
| Potree | Built-in | `viewer.setEDLEnabled(true)` |
| CesiumJS 3D Tiles | Built-in | `tileset.pointCloudShading.eyeDomeLighting = true` |
| Three.js | Custom post-processing | Requires depth buffer + custom shader pass |
| deck.gl | Custom effect | Requires post-processing extension |

### Tools Comparison

| Tool | Format Support | Max Points | Web-Based | EDL | Measurement | Best For |
|------|---------------|------------|-----------|-----|-------------|----------|
| **Potree** | LAS, LAZ, Potree | Billions | Yes | Yes | Yes | Large LiDAR datasets |
| **CesiumJS** | 3D Tiles (pnts/glTF) | Billions | Yes | Yes | Custom | Globe-context point clouds |
| **Three.js** | LAS, PLY, PCD | Millions | Yes | Custom | Custom | Custom point cloud apps |
| **deck.gl** | Binary, LAS (loaders.gl) | Millions | Yes | Custom | Custom | Analytical overlays |
| **CloudCompare** | LAS, LAZ, E57, PLY, PCD | Billions | No | Yes | Yes | Processing, classification, QA |
| **PDAL** | All formats | Unlimited | No (CLI) | N/A | N/A | Pipeline processing |
| **Entwine / EPT** | EPT → COPC | Billions | Server | N/A | N/A | Cloud-native point cloud serving |

### Point Cloud Optimization

- **Level of detail:** Use octree-based LOD (Potree, 3D Tiles, COPC) for progressive streaming. This is non-negotiable for anything over 10M points.
- **Compression:** LAZ (compressed LAS) reduces file size 5-10x. COPC (Cloud Optimized Point Cloud) adds spatial indexing to LAZ.
- **Tiling:** For static datasets, COPC is the current SOTA -- a single file with built-in spatial index, served via HTTP range requests. No tile server needed.
- **Point budget:** Set a visible point budget (1-5M for web, 10-50M for desktop) and let the LOD system handle the rest. More points does not mean better visualization above a threshold.
- **Adaptive point size:** Scale point size inversely with density so that sparse areas fill gaps and dense areas avoid overlap.
- **EDL:** Always enable Eye Dome Lighting. It is essentially free computationally and dramatically improves readability.

**Dark Arts: Point Cloud Performance**

- COPC (Cloud Optimized Point Cloud) is to LiDAR what COG is to rasters. A single `.copc.laz` file replaces an entire tile pyramid. Use `pdal translate input.laz output.copc.laz` to convert.
- For web delivery, the combination of COPC + HTTP range requests + a CDN eliminates the need for any specialized tile server. Just upload the file to S3/GCS and serve it.
- When working with classified point clouds, filter to only the classes you need before visualization. Rendering ground + buildings is often sufficient; including noise and overlap points wastes budget.
- The biggest performance mistake is not setting a point budget. Trying to render 100M points simultaneously will crash any browser. Let the octree do its job.

---

## Volumetric and Scientific 3D

Visualizing 3D continuous fields -- atmospheric data, ocean currents, subsurface geology, and other scientific phenomena that exist throughout a volume rather than on surfaces.

> Cross-reference: [Scientific Visualization](scientific-visualization.md) for 2D scientific vis techniques (color maps, contours, flow fields).

### Rendering Techniques

| Technique | Description | Performance | Best For |
|-----------|-------------|-------------|----------|
| **Isosurfaces** | Extract polygon mesh at a threshold value | Fast (after extraction) | Temperature boundaries, geological horizons |
| **Volume rendering** | Ray-cast through 3D scalar field | GPU-intensive | Medical imaging, atmospheric density |
| **Slice planes** | 2D cross-sections through 3D data | Fast | Geological profiles, debugging |
| **Particle tracing** | Animate particles through vector fields | Moderate | Wind, ocean currents |
| **Streamlines/tubes** | Trace integral curves through vector fields | Moderate | Steady-state flow visualization |
| **Voxel rendering** | Render individual volume elements | Variable | Discrete 3D data (geology, mining) |

### Atmospheric Data Visualization

```javascript
// Three.js: 3D wind field particle visualization
import * as THREE from 'three';

class WindFieldParticles {
  constructor(scene, windData) {
    this.particleCount = 100000;
    this.windData = windData;  // 3D grid: { u, v, w } at each grid point

    const positions = new Float32Array(this.particleCount * 3);
    const colors = new Float32Array(this.particleCount * 3);

    // Initialize particles randomly within the domain
    for (let i = 0; i < this.particleCount; i++) {
      positions[i * 3]     = Math.random() * windData.width;
      positions[i * 3 + 1] = Math.random() * windData.height;
      positions[i * 3 + 2] = Math.random() * windData.depth;
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
      size: 0.5,
      vertexColors: true,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false
    });

    this.points = new THREE.Points(this.geometry, material);
    scene.add(this.points);
  }

  update(dt) {
    const positions = this.geometry.attributes.position.array;
    const colors = this.geometry.attributes.color.array;

    for (let i = 0; i < this.particleCount; i++) {
      const idx = i * 3;
      // Trilinear interpolation of wind vector at particle position
      const wind = this.sampleWind(positions[idx], positions[idx + 1], positions[idx + 2]);

      // Advect particle
      positions[idx]     += wind.u * dt;
      positions[idx + 1] += wind.v * dt;
      positions[idx + 2] += wind.w * dt;

      // Color by wind speed
      const speed = Math.sqrt(wind.u ** 2 + wind.v ** 2 + wind.w ** 2);
      const t = Math.min(speed / 20.0, 1.0);
      colors[idx]     = t;        // R
      colors[idx + 1] = 0.3;     // G
      colors[idx + 2] = 1.0 - t; // B

      // Reset particles that leave the domain
      if (this.outOfBounds(positions[idx], positions[idx + 1], positions[idx + 2])) {
        this.resetParticle(positions, idx);
      }
    }

    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
  }
}
```

### Subsurface Geology Visualization

```javascript
// Three.js: voxel model for geological layers
function createGeologyVoxels(voxelData, cellSize) {
  // voxelData: 3D array of rock-type codes
  const rockColors = {
    1: 0x8B4513,  // Sandstone
    2: 0x696969,  // Shale
    3: 0xD2B48C,  // Limestone
    4: 0x2F4F4F,  // Basalt
    5: 0xFFD700   // Ore body
  };

  const group = new THREE.Group();
  const geometries = {};  // group by rock type for instanced rendering

  for (let z = 0; z < voxelData.depth; z++) {
    for (let y = 0; y < voxelData.height; y++) {
      for (let x = 0; x < voxelData.width; x++) {
        const rockType = voxelData.get(x, y, z);
        if (rockType === 0) continue;  // skip air/empty

        if (!geometries[rockType]) geometries[rockType] = [];
        geometries[rockType].push(new THREE.Vector3(
          x * cellSize, z * cellSize, y * cellSize  // swap Y/Z for geo convention
        ));
      }
    }
  }

  // Use InstancedMesh for performance
  const boxGeo = new THREE.BoxGeometry(cellSize * 0.95, cellSize * 0.95, cellSize * 0.95);
  for (const [type, positions] of Object.entries(geometries)) {
    const material = new THREE.MeshLambertMaterial({ color: rockColors[type], transparent: true, opacity: 0.8 });
    const mesh = new THREE.InstancedMesh(boxGeo, material, positions.length);

    const matrix = new THREE.Matrix4();
    positions.forEach((pos, i) => {
      matrix.setPosition(pos);
      mesh.setMatrixAt(i, matrix);
    });

    group.add(mesh);
  }

  return group;
}
```

### Borehole Visualization

```javascript
// Three.js: render borehole logs as 3D columns
function createBorehole(boreholeData) {
  // boreholeData: { x, y, intervals: [{ from, to, lithology, color }] }
  const group = new THREE.Group();
  const radius = 2.0;  // visual radius in meters

  for (const interval of boreholeData.intervals) {
    const height = interval.to - interval.from;
    const geometry = new THREE.CylinderGeometry(radius, radius, height, 16);
    const material = new THREE.MeshLambertMaterial({ color: interval.color });
    const cylinder = new THREE.Mesh(geometry, material);

    cylinder.position.set(
      boreholeData.x,
      -(interval.from + height / 2),  // negative depth
      boreholeData.y
    );

    cylinder.userData = {
      lithology: interval.lithology,
      from: interval.from,
      to: interval.to
    };

    group.add(cylinder);
  }

  return group;
}
```

### Scientific Visualization Tools

| Tool | Type | Volume Rendering | Iso-surfaces | Web Export | Best For |
|------|------|-----------------|--------------|-----------|----------|
| **ParaView** | Desktop + web (Glance) | Yes | Yes | ParaView Glance, trame | Large scientific datasets |
| **VisIt** | Desktop | Yes | Yes | Limited | HPC simulation output |
| **CesiumJS Voxels** | Web (experimental) | Partial | No | Native | Geo-referenced 3D grids |
| **Three.js** | Web | Custom shaders | Marching cubes | Native | Custom web visualizations |
| **deck.gl** | Web | Custom layers | Custom | Native | Analytical geo overlays |
| **VTK.js** | Web | Yes | Yes | Native | Medical/scientific web viz |
| **Matplotlib 3D** | Python | Voxels only | Yes | Static export | Quick prototyping |

---

## Photogrammetry and Reality Capture

Transforming photographs and sensor data into 3D models, point clouds, and meshes for geospatial analysis and visualization.

### Structure from Motion (SfM) Pipeline

The standard photogrammetry pipeline transforms overlapping photographs into georeferenced 3D models.

```
Photographs (drone/ground)
    |
    v
Feature Detection (SIFT/ORB)
    |
    v
Feature Matching + Bundle Adjustment
    |
    v
Sparse Point Cloud (tie points)
    |
    v
Dense Matching (MVS)
    |
    v
Dense Point Cloud (millions-billions of points)
    |
    v
Mesh Generation (Poisson/Delaunay)
    |
    v
Texture Mapping
    |
    v
Orthophoto + DSM/DTM
    |
    v
3D Tiles (for web streaming)
```

### Tools Comparison

| Tool | License | SfM | Dense Cloud | Meshing | 3D Tiles Export | Accuracy |
|------|---------|-----|-------------|---------|-----------------|----------|
| **OpenDroneMap** | AGPL | Yes | Yes | Yes | Via entwine/COPC | Good |
| **Agisoft Metashape** | Commercial | Yes | Yes | Yes | Yes (native) | Excellent |
| **RealityCapture** | Commercial | Yes | Yes | Yes | Yes (via plugin) | Excellent |
| **COLMAP** | BSD | Yes | Yes | Yes | No (manual) | Excellent |
| **Meshroom (AliceVision)** | MPL | Yes | Yes | Yes | No | Good |
| **Pix4D** | Commercial | Yes | Yes | Yes | Yes | Excellent |

### OpenDroneMap Pipeline

```bash
# Process drone imagery with WebODM (Docker-based)
docker run -ti -p 3000:3000 opendronemap/webodm

# Or use ODM directly from command line
docker run -ti --rm \
  -v /path/to/images:/datasets/code \
  opendronemap/odm \
  --project-path /datasets \
  --dsm --dtm \
  --mesh-octree-depth 12 \
  --feature-quality high \
  --pc-quality high \
  --georeferencing-mode gcp  # if using ground control points

# Output structure:
# /datasets/code/odm_dem/dsm.tif          - Digital Surface Model
# /datasets/code/odm_dem/dtm.tif          - Digital Terrain Model
# /datasets/code/odm_orthophoto/odm_orthophoto.tif - Orthophoto
# /datasets/code/odm_texturing/model.obj   - Textured mesh
# /datasets/code/odm_georeferencing/odm_georeferenced_model.laz - Point cloud
```

### Converting to 3D Tiles

```bash
# Point cloud to 3D Tiles via py3dtiles
pip install py3dtiles
py3dtiles convert pointcloud.laz --out ./tiles/ --srs_in EPSG:32632 --srs_out EPSG:4978

# Mesh to 3D Tiles via Cesium ion REST API (easiest path)
# Upload OBJ/glTF/FBX and ion will tile it automatically

# Or use obj2gltf + 3d-tiles-tools
npx obj2gltf -i model.obj -o model.glb
# Then tile with cesium 3d-tiles-tools or Cesium ion
```

### 3D Gaussian Splatting (Emerging SOTA)

3D Gaussian Splatting (3DGS) is a breakthrough technique (2023) that represents scenes as collections of 3D Gaussian primitives, enabling real-time novel view synthesis at quality rivaling NeRF but at 100-1000x faster rendering speed.

**Why GISers should care:** 3DGS produces photorealistic 3D representations from photographs that can be rendered in real-time in a web browser. For urban modeling, heritage documentation, and site inspection, it offers a compelling alternative to traditional photogrammetry meshes.

```bash
# Train a 3DGS model from COLMAP output
# 1. Run COLMAP SfM first
colmap automatic_reconstructor \
  --workspace_path ./workspace \
  --image_path ./images

# 2. Train 3D Gaussian Splatting
# (requires CUDA GPU)
git clone https://github.com/graphdeco-inria/gaussian-splatting.git
cd gaussian-splatting
python train.py \
  -s ./workspace \
  --iterations 30000 \
  --densify_until_iter 15000

# 3. View in browser with gsplat.js or other web viewers
# Export to .ply or .splat format for web rendering
```

**Web viewers for 3DGS:**

| Viewer | License | Features | URL |
|--------|---------|----------|-----|
| **gsplat.js** | MIT | Three.js integration, progressive loading | github.com/huggingface/gsplat.js |
| **antimatter15/splat** | MIT | Pure WebGL, lightweight | github.com/antimatter15/splat |
| **PlayCanvas SuperSplat** | MIT | Editor + viewer, compression | github.com/playcanvas/supersplat |
| **Luma AI** | Commercial | Cloud processing + hosting | lumalabs.ai |

### Mapillary / Street-Level Integration

```javascript
// Display Mapillary street-level imagery alongside 3D models in CesiumJS
const mapillaryImageLayer = new Cesium.ImageryLayer(
  new Cesium.UrlTemplateImageryProvider({
    url: 'https://tiles.mapillary.com/maps/vtp/mly1_public/2/{z}/{x}/{y}?access_token=' + MAPILLARY_TOKEN,
    minimumLevel: 0,
    maximumLevel: 14
  })
);

// Link 3D model click to nearest Mapillary image
viewer.screenSpaceEventHandler.setInputAction((click) => {
  const pickedPosition = viewer.scene.pickPosition(click.position);
  if (pickedPosition) {
    const carto = Cesium.Cartographic.fromCartesian(pickedPosition);
    const lon = Cesium.Math.toDegrees(carto.longitude);
    const lat = Cesium.Math.toDegrees(carto.latitude);

    // Query Mapillary API for nearest image
    fetch(`https://graph.mapillary.com/images?access_token=${MAPILLARY_TOKEN}&fields=id,geometry&bbox=${lon-0.001},${lat-0.001},${lon+0.001},${lat+0.001}&limit=1`)
      .then(r => r.json())
      .then(data => {
        if (data.data.length > 0) {
          window.open(`https://www.mapillary.com/app/?pKey=${data.data[0].id}`);
        }
      });
  }
}, Cesium.ScreenSpaceEventType.LEFT_DOUBLE_CLICK);
```

---

## Game Engines for Geo

Game engines bring physically-based rendering (PBR), advanced lighting, physics simulation, and massive scene management to geospatial visualization. The integration of Cesium's globe and 3D Tiles into Unreal Engine and Unity has opened new possibilities for photorealistic geospatial experiences.

> Cross-reference: [3D Visualization Tools](../tools/3d-visualization.md) for desktop and CLI tool comparisons.

### Unreal Engine + Cesium for Unreal

Cesium for Unreal brings the full Cesium 3D globe -- terrain, 3D Tiles, Google Photorealistic Tiles -- into Unreal Engine 5 with Nanite, Lumen, and the full UE rendering pipeline.

**Key capabilities:**
- Stream Cesium World Terrain, Google Photorealistic 3D Tiles, and custom 3D Tiles in UE5
- Nanite virtualized geometry for massive mesh detail without LOD authoring
- Lumen global illumination for realistic lighting on geospatial content
- Full Blueprint and C++ scripting for interactive applications
- VR/AR output via UE5's XR framework

```cpp
// Unreal C++: programmatic camera flight to a geo location
#include "CesiumGeoreference.h"
#include "CesiumGlobeAnchorComponent.h"

void AMyPawn::FlyToLocation(double Longitude, double Latitude, double Height)
{
    ACesiumGeoreference* Georeference = ACesiumGeoreference::GetDefaultGeoreference(this);
    if (Georeference)
    {
        // Convert WGS84 to Unreal world coordinates
        FVector UnrealPosition = Georeference->TransformLongitudeLatitudeHeightPositionToUnreal(
            FVector(Longitude, Latitude, Height)
        );

        // Smooth interpolation to target
        FLatentActionInfo LatentInfo;
        UKismetSystemLibrary::MoveComponentTo(
            GetRootComponent(),
            UnrealPosition,
            FRotator(-30, 0, 0),  // pitch down 30 degrees
            false, false, 3.0f,   // 3 second flight
            false, EMoveComponentAction::Move, LatentInfo
        );
    }
}
```

**Blueprint setup (common workflow):**

1. Add `CesiumGeoreference` actor to level
2. Add `Cesium3DTileset` actor, set Ion Asset ID (e.g., 2275207 for Google 3D Tiles)
3. Add `CesiumSunSky` for accurate sun position based on geolocation and time
4. Attach `CesiumGlobeAnchorComponent` to any actor to pin it to WGS84 coordinates
5. Configure `CesiumCameraManager` for tile loading based on player camera

### Unity + Cesium for Unity

Cesium for Unity provides the same globe and 3D Tiles capabilities within Unity, targeting mobile, desktop, WebGL, and XR platforms.

```csharp
// Unity C#: place an object at a geographic location
using CesiumForUnity;
using UnityEngine;

public class GeoPlacement : MonoBehaviour
{
    public CesiumGeoreference georeference;
    public double longitude = -74.006;
    public double latitude = 40.7128;
    public double height = 50.0;

    void Start()
    {
        // Add globe anchor to pin this object to WGS84 coordinates
        var anchor = gameObject.AddComponent<CesiumGlobeAnchor>();
        anchor.longitudeLatitudeHeight = new double3(longitude, latitude, height);
    }

    // Animate object along a geographic path
    public void AnimateAlongPath(double3[] waypoints, float duration)
    {
        StartCoroutine(FollowPath(waypoints, duration));
    }

    private IEnumerator FollowPath(double3[] waypoints, float totalDuration)
    {
        var anchor = GetComponent<CesiumGlobeAnchor>();
        float segmentDuration = totalDuration / (waypoints.Length - 1);

        for (int i = 0; i < waypoints.Length - 1; i++)
        {
            float elapsed = 0;
            while (elapsed < segmentDuration)
            {
                double t = elapsed / segmentDuration;
                double3 current = math.lerp(waypoints[i], waypoints[i + 1], t);
                anchor.longitudeLatitudeHeight = current;
                elapsed += Time.deltaTime;
                yield return null;
            }
        }
    }
}
```

### Blender GIS

Blender GIS enables cinematic-quality geospatial renders by importing terrain, satellite imagery, and vector data directly into Blender's 3D environment.

```python
# Blender Python: import terrain DEM and drape satellite imagery
import bpy
import bmesh

# Assumes BlenderGIS addon is installed and configured
# Import DEM as mesh via BlenderGIS addon
# Addon menu: GIS > Import > Georef Raster > select DEM.tif

# Programmatic approach: create terrain from heightmap
def create_terrain_from_heightmap(filepath, scale_xy=1.0, scale_z=0.1):
    """Create terrain mesh from a grayscale heightmap image."""
    img = bpy.data.images.load(filepath)
    width, height = img.size

    # Create plane mesh with subdivisions matching image resolution
    bpy.ops.mesh.primitive_grid_add(
        x_subdivisions=min(width, 500),
        y_subdivisions=min(height, 500),
        size=width * scale_xy
    )
    terrain = bpy.context.active_object
    terrain.name = "Terrain"

    # Apply displacement modifier
    disp_mod = terrain.modifiers.new(name="DEM", type='DISPLACE')
    tex = bpy.data.textures.new("HeightMap", type='IMAGE')
    tex.image = img
    disp_mod.texture = tex
    disp_mod.strength = scale_z
    disp_mod.direction = 'Z'

    # Add satellite imagery as texture
    mat = bpy.data.materials.new(name="SatelliteMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear defaults, set up image texture
    for node in nodes:
        nodes.remove(node)
    output = nodes.new('ShaderNodeOutputMaterial')
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    img_tex = nodes.new('ShaderNodeTexImage')
    img_tex.image = bpy.data.images.load("satellite.tif")

    links.new(img_tex.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    terrain.data.materials.append(mat)
    return terrain
```

### rayshader (R)

rayshader produces stunning 3D terrain renderings with ray-traced shadows, water detection, and ambient occlusion -- all from R.

```r
library(rayshader)
library(elevatr)
library(sf)
library(terra)

# Define area of interest (Yosemite Valley)
bbox <- st_bbox(c(xmin = -119.65, ymin = 37.71, xmax = -119.53, ymax = 37.76),
                crs = 4326)
aoi <- st_as_sfc(bbox)

# Get elevation data
dem <- get_elev_raster(aoi, z = 12, src = "aws")
dem_matrix <- raster_to_matrix(dem)

# Create 3D scene with multiple layers
dem_matrix %>%
  height_shade(texture = grDevices::colorRampPalette(
    c("#1B4332", "#52B788", "#D8F3DC", "#8B7355", "#FFFFFF")
  )(256)) %>%
  add_shadow(ray_shade(dem_matrix, zscale = 3, lambert = TRUE), 0.5) %>%
  add_shadow(ambient_shade(dem_matrix, zscale = 3), 0.3) %>%
  add_water(detect_water(dem_matrix), color = "#2176AE") %>%
  plot_3d(
    dem_matrix,
    zscale = 10,
    fov = 0,
    theta = -45,
    phi = 30,
    zoom = 0.7,
    windowsize = c(1200, 800),
    background = "#1a1a2e"
  )

# Render high-quality image with ray tracing
render_highquality(
  filename = "yosemite_3d.png",
  samples = 256,
  parallel = TRUE,
  light = TRUE,
  lightdirection = c(315, 315),
  lightaltitude = c(45, 30),
  lightintensity = c(600, 200),
  lightcolor = c("#FFFFFF", "#FFE4B5"),
  width = 3840,
  height = 2160
)

# Create animation orbiting the scene
render_movie(
  filename = "yosemite_orbit.mp4",
  type = "orbit",
  frames = 360,
  fps = 30,
  phi = 30,
  zoom = 0.7
)
```

### Game Engine Comparison for Geo

| Feature | Unreal + Cesium | Unity + Cesium | Blender GIS | rayshader (R) |
|---------|----------------|---------------|-------------|---------------|
| **Rendering** | PBR + Nanite + Lumen | PBR + URP/HDRP | Cycles/Eevee | Ray tracing |
| **Globe** | Full WGS84 globe | Full WGS84 globe | Local projection | Local DEM |
| **3D Tiles** | Native streaming | Native streaming | Import only | No |
| **Real-time** | Yes (60+ fps) | Yes (60+ fps) | Viewport only | No (offline) |
| **VR/AR** | Full XR support | Full XR support | VR viewport | No |
| **Scripting** | Blueprint / C++ | C# | Python | R |
| **Web export** | Pixel Streaming | WebGL build | No | Static images |
| **Best for** | Cinematic geo, simulations | Mobile geo apps, AR | Cinematic stills, data art | Publication figures |

---

## WebXR and Immersive Geo

Virtual Reality (VR) and Augmented Reality (AR) bring geospatial data into immersive, embodied experiences for planning, training, field work, and public engagement.

### WebXR API + Three.js

```javascript
import * as THREE from 'three';
import { VRButton } from 'three/addons/webxr/VRButton.js';
import { XRControllerModelFactory } from 'three/addons/webxr/XRControllerModelFactory.js';

// Enable WebXR on the renderer
renderer.xr.enabled = true;
document.body.appendChild(VRButton.createButton(renderer));

// Scale terrain to room-scale VR (1:10000)
const terrainGroup = new THREE.Group();
terrainGroup.scale.set(0.0001, 0.0001, 0.0001);  // 10km terrain fits in 1m
scene.add(terrainGroup);

// Add terrain mesh (loaded from DEM)
terrainGroup.add(terrainMesh);  // from earlier Three.js terrain example

// Controller-based interaction: point at terrain to query elevation
const controller = renderer.xr.getController(0);
controller.addEventListener('selectstart', () => {
  const raycaster = new THREE.Raycaster();
  raycaster.set(
    controller.position,
    new THREE.Vector3(0, 0, -1).applyQuaternion(controller.quaternion)
  );

  const intersects = raycaster.intersectObject(terrainMesh);
  if (intersects.length > 0) {
    const point = intersects[0].point;
    // Convert back to geographic coordinates
    const geo = sceneToGeo(point);
    showLabel(`Elevation: ${geo.z.toFixed(1)}m\nLat: ${geo.lat.toFixed(5)}\nLon: ${geo.lon.toFixed(5)}`);
  }
});
scene.add(controller);

// Teleportation for VR navigation across large terrain
const teleportMarker = new THREE.Mesh(
  new THREE.RingGeometry(0.15, 0.2, 32).rotateX(-Math.PI / 2),
  new THREE.MeshBasicMaterial({ color: 0x00ff00, transparent: true, opacity: 0.5 })
);

// Render loop must use setAnimationLoop for WebXR
renderer.setAnimationLoop(() => {
  renderer.render(scene, camera);
});
```

### A-Frame for Quick Geo VR Prototypes

A-Frame provides a declarative HTML approach to VR, making it easy to create quick geospatial VR prototypes without deep Three.js knowledge.

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://aframe.io/releases/1.5.0/aframe.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/aframe-terrain-model-component@0.3.2/dist/aframe-terrain-model-component.min.js"></script>
</head>
<body>
  <a-scene>
    <!-- Sky and environment -->
    <a-sky color="#87CEEB"></a-sky>
    <a-light type="directional" position="1 2 1" intensity="0.8"></a-light>
    <a-light type="ambient" intensity="0.4"></a-light>

    <!-- Terrain from DEM heightmap -->
    <a-entity
      terrain-model="
        DEM: url(dem_heightmap.png);
        texture: url(satellite.jpg);
        planeWidth: 100;
        planeHeight: 100;
        segmentsWidth: 256;
        segmentsHeight: 256;
        zPosition: 0.5;
      "
      position="0 -2 -20"
      rotation="-90 0 0"
      scale="0.1 0.1 0.1">
    </a-entity>

    <!-- GIS data points as 3D markers -->
    <a-entity id="poi-markers"></a-entity>

    <!-- Camera with VR-compatible controls -->
    <a-entity camera look-controls wasd-controls="acceleration: 20" position="0 5 0">
      <a-cursor></a-cursor>
    </a-entity>
  </a-scene>

  <script>
    // Add GIS points of interest as 3D markers
    const pois = [
      { name: "Weather Station A", x: -5, y: 3, z: -15, temp: 22.3 },
      { name: "Weather Station B", x: 8, y: 2, z: -25, temp: 19.8 },
    ];

    const container = document.getElementById('poi-markers');
    pois.forEach(poi => {
      const marker = document.createElement('a-entity');
      marker.setAttribute('position', `${poi.x} ${poi.y} ${poi.z}`);
      marker.innerHTML = `
        <a-sphere radius="0.2" color="${poi.temp > 20 ? '#FF4444' : '#4444FF'}"></a-sphere>
        <a-text value="${poi.name}\n${poi.temp}C" position="0 0.4 0"
                align="center" width="3" color="#FFFFFF"></a-text>
      `;
      container.appendChild(marker);
    });
  </script>
</body>
</html>
```

### AR Field Tools

Augmented Reality overlays GIS data on the real-world camera view, enabling field workers to see underground utilities, planned constructions, or analytical results in situ.

**Key technologies:**

| Technology | Platform | Geo Registration | Best For |
|------------|----------|-----------------|----------|
| **WebXR (AR mode)** | Mobile browsers | GPS + IMU | Quick prototypes, web-based |
| **ARCore (Google)** | Android | Visual + GPS | Production mobile apps |
| **ARKit (Apple)** | iOS | Visual + GPS + LiDAR | High-accuracy iOS apps |
| **Cesium for Unity AR** | iOS/Android | WGS84 globe anchor | Globe-scale AR |
| **ArcGIS Runtime AR** | iOS/Android | Built-in geo-tracking | Enterprise field apps |
| **Vuforia + geo** | Cross-platform | Marker + GPS hybrid | Industrial AR |

```javascript
// WebXR AR session with GIS overlay
async function startARSession() {
  const session = await navigator.xr.requestSession('immersive-ar', {
    requiredFeatures: ['hit-test', 'dom-overlay'],
    optionalFeatures: ['geolocation'],
    domOverlay: { root: document.getElementById('overlay') }
  });

  // Use device GPS for coarse geo-registration
  navigator.geolocation.watchPosition((pos) => {
    const userLon = pos.coords.longitude;
    const userLat = pos.coords.latitude;

    // Query nearby GIS features
    fetch(`/api/features?lon=${userLon}&lat=${userLat}&radius=100`)
      .then(r => r.json())
      .then(features => {
        features.forEach(f => {
          // Convert geographic offset to local AR coordinates
          const dx = (f.longitude - userLon) * 111320 * Math.cos(userLat * Math.PI / 180);
          const dy = (f.latitude - userLat) * 110540;
          placeARMarker(dx, f.elevation - pos.coords.altitude, -dy, f.properties);
        });
      });
  }, null, { enableHighAccuracy: true });
}
```

### Mixed Reality for Urban Planning

Mixed Reality (MR) enables stakeholders to visualize proposed developments at full scale, overlaid on the actual site. This is increasingly used for public engagement in urban planning.

**Workflow:**

1. Create 3D model of proposed development (SketchUp, Rhino, BIM software)
2. Export as glTF / 3D Tiles
3. Geo-register model using CesiumGlobeAnchor (Unity/Unreal) or manual WGS84 placement
4. Deploy to HoloLens 2 (indoor) or tablet AR (outdoor)
5. Stakeholders walk through the site seeing the proposed building at real scale

**Dark Arts: Immersive Geo**

- GPS accuracy is 3-10m on consumer devices. For precision AR field work, use RTK GNSS or visual localization (scanning known features). Do not rely on raw GPS for sub-meter AR placement.
- WebXR AR mode is supported on Chrome Android and Safari iOS 15+, but feature support varies. Test on target devices early.
- For large terrain in VR, scale everything to room-scale (1:10000 or 1:50000) rather than rendering at 1:1. Users lose orientation in 1:1 terrain that extends beyond their visible range.
- The most effective public engagement tool is a simple tablet AR app where citizens can point their device at a construction site and see the proposed building. Forget about headsets for public audiences.

---

## Performance and Hardware

### GPU Requirements by Use Case

| Use Case | Minimum GPU | Minimum VRAM | Minimum RAM | Recommended |
|----------|-------------|-------------|-------------|-------------|
| MapLibre 3D terrain | Integrated (Intel/Apple) | Shared | 4 GB | Any modern device |
| CesiumJS + OSM buildings | Dedicated (2 GB VRAM) | 2 GB | 8 GB | GTX 1060 / M1 |
| CesiumJS + Google 3D Tiles | Dedicated (4 GB VRAM) | 4 GB | 16 GB | RTX 3060 / M1 Pro |
| Potree (100M points) | Dedicated (2 GB VRAM) | 2 GB | 8 GB | GTX 1060 / M1 |
| Potree (1B+ points) | Dedicated (4 GB VRAM) | 4 GB | 16 GB | RTX 3060 / M1 Pro |
| deck.gl (1M+ features) | Dedicated (2 GB VRAM) | 2 GB | 8 GB | GTX 1060 / M1 |
| Volume rendering | Dedicated (4 GB+ VRAM) | 4 GB | 16 GB | RTX 3070+ |
| Unreal + Cesium (Nanite) | Dedicated (8 GB VRAM) | 8 GB | 32 GB | RTX 3080 / M2 Max |
| 3D Gaussian Splatting (training) | CUDA GPU (8 GB+ VRAM) | 8 GB | 32 GB | RTX 3090 / A100 |
| 3D Gaussian Splatting (viewing) | Dedicated (2 GB VRAM) | 2 GB | 8 GB | GTX 1060 / M1 |

### Memory Management for Massive Scenes

```javascript
// CesiumJS: monitor and manage memory
const scene = viewer.scene;

// Set tile cache size (MB)
scene.globe.tileCacheSize = 100;

// Monitor memory pressure
setInterval(() => {
  const stats = {
    tilesLoaded: tileset.tilesLoaded,
    tilesVisible: tileset.statistics?.numberOfTilesVisible,
    gpuMemoryMB: tileset.totalMemoryUsageInBytes / 1e6,
    commandsExecuted: scene.debugShowFramesPerSecond
  };

  // If GPU memory exceeds budget, increase screen space error
  if (stats.gpuMemoryMB > 800) {
    tileset.maximumScreenSpaceError = Math.min(
      tileset.maximumScreenSpaceError + 2,
      32
    );
    console.warn(`GPU memory high (${stats.gpuMemoryMB.toFixed(0)}MB), ` +
                 `relaxing SSE to ${tileset.maximumScreenSpaceError}`);
  }
}, 5000);

// Dispose resources when switching scenes
function cleanupScene() {
  viewer.scene.primitives.removeAll();
  viewer.entities.removeAll();
  viewer.dataSources.removeAll();

  // Force garbage collection of GPU resources
  viewer.scene.globe.tileCacheSize = 0;
  viewer.scene.renderForSpecs();
  viewer.scene.globe.tileCacheSize = 100;
}
```

### Level of Detail (LOD) Strategies

| Strategy | Mechanism | Used By | Pros | Cons |
|----------|-----------|---------|------|------|
| **Screen Space Error** | Refine tiles until pixel error < threshold | CesiumJS, 3D Tiles | Smooth, view-dependent | Memory for tile hierarchy |
| **Octree LOD** | 8-children subdivision of 3D space | Potree, 3D Tiles (points) | Handles billions of points | Complex tree management |
| **Quadtree LOD** | 4-children subdivision of 2D surface | Terrain tiles, imagery | Simple, efficient for surfaces | Only for 2D-ish data |
| **Distance-based** | Swap models at distance thresholds | Three.js LOD, game engines | Simple to implement | Visible pop-in |
| **Continuous LOD** | Geomorphing between detail levels | Terrain rendering | No pop-in | Complex implementation |
| **HLOD** | Pre-combined lower-res parent tiles | Unreal Engine, Nanite | Fast rendering | Large storage |

### Tile Caching and Prefetching

```javascript
// Service Worker for offline 3D tile caching
// sw.js
const TILE_CACHE = 'geo-tiles-v1';
const MAX_CACHE_SIZE = 500 * 1024 * 1024;  // 500 MB

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Cache terrain tiles, imagery, and 3D tiles
  if (url.pathname.match(/\.(terrain|b3dm|glb|pnts|png|jpg|json)$/)) {
    event.respondWith(
      caches.open(TILE_CACHE).then(cache =>
        cache.match(event.request).then(cached => {
          if (cached) return cached;

          return fetch(event.request).then(response => {
            if (response.ok) {
              cache.put(event.request, response.clone());
              trimCache(cache, MAX_CACHE_SIZE);
            }
            return response;
          });
        })
      )
    );
  }
});

async function trimCache(cache, maxSize) {
  const keys = await cache.keys();
  let totalSize = 0;

  // Estimate size and evict oldest entries
  for (const key of keys) {
    const response = await cache.match(key);
    const blob = await response.blob();
    totalSize += blob.size;
  }

  if (totalSize > maxSize) {
    // Remove oldest 20% of entries
    const removeCount = Math.ceil(keys.length * 0.2);
    for (let i = 0; i < removeCount; i++) {
      await cache.delete(keys[i]);
    }
  }
}
```

### Mobile 3D Considerations

Mobile devices face unique constraints for 3D geospatial rendering. Key guidelines:

- **Point budget:** Reduce to 500K-1M visible points (vs 2-5M on desktop)
- **Tile quality:** Increase `maximumScreenSpaceError` to 24-32 on mobile
- **Texture size:** Limit to 2048x2048 per tile; prefer compressed textures (ASTC on mobile, BC7 on desktop)
- **Draw calls:** Keep under 100 for smooth 30fps on mid-range mobile GPUs
- **Touch interaction:** Implement pinch-zoom, two-finger rotate. Do not rely on hover events.
- **Battery:** Use `requestRenderMode = true` and reduce `targetFrameRate` to 30fps to save battery
- **Network:** Assume 4G (10-50 Mbps). Prefetch tiles along likely navigation paths. Use lower-resolution fallback tiles.

```javascript
// Detect mobile and apply appropriate settings
function applyMobileSettings(viewer) {
  const isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent);

  if (isMobile) {
    viewer.scene.requestRenderMode = true;
    viewer.targetFrameRate = 30;
    viewer.scene.fog.density = 4.0e-4;  // more aggressive fog = fewer far tiles

    if (tileset) {
      tileset.maximumScreenSpaceError = 24;
      tileset.maximumMemoryUsage = 128;  // MB
      tileset.preloadWhenHidden = false;
      tileset.skipLevelOfDetail = true;
    }

    // Reduce resolution on high-DPI mobile screens
    viewer.resolutionScale = 0.75;  // render at 75% resolution
  }
}
```

**Dark Arts: Performance**

- The single biggest performance win for any 3D geospatial web app is `requestRenderMode = true`. Most geo apps are static 99% of the time -- rendering 60fps while the user reads a tooltip is pure waste.
- On CesiumJS, `viewer.scene.fog` is not just cosmetic. The fog distance determines the far plane for tile loading. Increasing fog density dramatically reduces the number of tiles loaded and rendered.
- For deck.gl, use binary data formats (Arrow, FlatBuffers) instead of GeoJSON. Parsing 1M features from GeoJSON takes seconds; from binary, milliseconds.
- WebGL context loss is the #1 crash cause on mobile. Handle the `webglcontextlost` event and reload gracefully. Never assume the GL context is permanent.
- Chrome on Android limits WebGL to 128MB of buffer memory by default. If you hit mysterious rendering failures on Android, this is likely the cause.

---

## Tool Comparison Matrix

### Comprehensive Feature Matrix

| Capability | CesiumJS | deck.gl | MapLibre GL | Three.js | Potree | Unreal+Cesium | Unity+Cesium | Blender GIS | QGIS 3D | ArcGIS Scene | ParaView |
|------------|----------|---------|-------------|----------|--------|--------------|-------------|-------------|---------|-------------|----------|
| **3D Terrain** | World Terrain + custom | TerrainLayer | raster-dem | Custom shader | No | Cesium terrain | Cesium terrain | DEM import | DEM | Elevation layers | VTK terrain |
| **3D Buildings** | 3D Tiles (b3dm/glTF) | Extrusion + custom | fill-extrusion | Custom | No | 3D Tiles | 3D Tiles | Import | CityGML/3D | 3D Tiles + BIM | Import |
| **Point Clouds** | 3D Tiles (pnts/glTF) | PointCloudLayer | No | Custom loaders | Octree LOD (billions) | 3D Tiles | 3D Tiles | Import | LAS/LAZ | Scene Layer | VTK readers |
| **Volumetric** | Voxel (experimental) | Custom layers | No | Custom shaders | No | UE volumes | Custom | Volumetrics | No | Voxel Layer | Full support |
| **Imagery drape** | Yes (multiple providers) | TileLayer | Raster sources | Custom UV | No | Cesium imagery | Cesium imagery | UV mapping | WMS/WMTS | Basemaps | VTK textures |
| **Globe** | Full WGS84 | Flat (WebMercator) | Flat (globe experimental) | Custom | No | Full WGS84 | Full WGS84 | No | No | Full | No |
| **Temporal** | Clock + CZML | FilterExtension | No (manual) | Custom | No | Blueprint | C# | Keyframes | Temporal manager | Time Slider | Time series |
| **VR/AR** | No (use game engines) | No | No | WebXR | No | Full XR | Full XR | VR viewport | No | No | Cave/VR |
| **Scripting** | JavaScript | JavaScript | JavaScript | JavaScript | JavaScript | Blueprint/C++ | C# | Python | Python/C++ | Python/JS | Python/C++ |
| **License** | Apache 2.0 | MIT | BSD | MIT | Apache 2.0 | Free (plugin) | Free (plugin) | GPL + addon | GPL | Commercial | BSD |
| **Offline** | Yes (self-host) | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes | Yes (Runtime) | Yes |

### When to Use What

| Scenario | Recommended Tool | Why |
|----------|-----------------|-----|
| Full 3D globe with multiple data types | **CesiumJS** | Only tool with native globe + 3D Tiles + terrain + imagery |
| Large-scale data analytics in 3D | **deck.gl** | GPU-accelerated layers, great for millions of features |
| Lightweight 3D map (terrain + buildings) | **MapLibre GL JS** | Small bundle, fast, easy to add to existing map |
| Custom 3D visualization / art | **Three.js** | Maximum flexibility, largest ecosystem |
| LiDAR point cloud inspection | **Potree** | Purpose-built, handles billions, measurement tools |
| Cinematic geospatial experience | **Unreal + Cesium** | Nanite + Lumen + full globe = unmatched visual quality |
| Cross-platform mobile geo app | **Unity + Cesium** | Build once, deploy to iOS/Android/WebGL |
| Publication-quality 3D figures | **rayshader (R)** or **Blender GIS** | Ray-traced rendering, fine aesthetic control |
| Desktop GIS with 3D view | **QGIS 3D View** | Integrated with full GIS toolbox |
| Enterprise 3D GIS platform | **ArcGIS Scene Viewer** | Full Esri ecosystem, enterprise support |
| Scientific volume data | **ParaView** | Best-in-class volume rendering, isosurfaces |
| Photorealistic city exploration | **CesiumJS + Google 3D Tiles** | Global photogrammetric mesh, free tier available |

### Decision Flowchart

```
Need 3D geospatial visualization?
|
+-- Need a full globe?
|   |
|   +-- Yes --> Need photorealistic rendering?
|   |           |
|   |           +-- Yes (game engine) --> Unreal + Cesium / Unity + Cesium
|   |           +-- Yes (web) --> CesiumJS + Google Photorealistic 3D Tiles
|   |           +-- No --> CesiumJS (standard)
|   |
|   +-- No --> What is the primary data type?
|       |
|       +-- Point clouds --> How many points?
|       |   |
|       |   +-- Billions --> Potree or CesiumJS + 3D Tiles
|       |   +-- Millions --> deck.gl PointCloudLayer or Three.js
|       |
|       +-- Buildings/urban --> Full 3D model or simple extrusion?
|       |   |
|       |   +-- Full 3D (3D Tiles) --> CesiumJS
|       |   +-- Simple extrusion --> MapLibre fill-extrusion
|       |
|       +-- Terrain only --> Need interactivity?
|       |   |
|       |   +-- Yes (web) --> MapLibre 3D terrain or deck.gl TerrainLayer
|       |   +-- No (static render) --> rayshader or Blender GIS
|       |
|       +-- Volumetric/scientific --> ParaView or Three.js custom shaders
|       |
|       +-- Custom/artistic --> Three.js (maximum flexibility)
```

---

## Further Reading and Cross-References

- [Elevation and Terrain Data Sources](../data-sources/elevation-terrain.md) -- DEM acquisition, processing, and serving
- [3D Mapping Libraries](../js-bindbox/3d-mapping.md) -- JavaScript library comparisons for 3D maps
- [3D Visualization Tools](../tools/3d-visualization.md) -- Desktop and CLI tools for 3D geo processing
- [Scientific Visualization](scientific-visualization.md) -- Color maps, contours, flow fields, and scientific data rendering
- [WebGL and GPU Computing](../js-bindbox/webgl-gpu.md) -- Low-level WebGL techniques for geospatial
- [LiDAR Processing](../data-sources/lidar-processing.md) -- Point cloud acquisition, classification, and processing pipelines

---

[Back to Visualization](README.md) | [Back to Main README](../README.md)
