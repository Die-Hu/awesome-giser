# 3D Visualization

> Techniques and tools for rendering geospatial data in three dimensions -- terrain, buildings, point clouds, and volumetric data.

> **Quick Picks**
> - **SOTA**: [CesiumJS](https://cesium.com) + 3D Tiles 1.1 -- full globe with terrain, buildings, point clouds, and Google Photorealistic Tiles
> - **Free Best**: [deck.gl](https://deck.gl) -- GPU-accelerated TerrainLayer, PointCloudLayer, and custom 3D layers
> - **Fastest Setup**: [Potree](https://potree.github.io) -- drop LiDAR data in and view billions of points immediately

## 3D Terrain Rendering

Visualizing elevation data as 3D surfaces for topographic analysis and context.

### Approaches

- **DEM-based rendering:** Use Digital Elevation Models to drape imagery over terrain
- **Terrain tiles:** Stream terrain data progressively (Cesium World Terrain, Mapbox Terrain)
- **Hillshade + 3D extrusion:** Combine shading effects with vertical exaggeration

### Data Sources

| Source | Resolution | Coverage | Format |
|--------|-----------|----------|--------|
| SRTM | 30m / 90m | Global (60N-56S) | GeoTIFF |
| Copernicus DEM | 30m | Global | GeoTIFF, COG |
| ALOS World 3D | 30m | Global | GeoTIFF |
| LiDAR-derived | <1m | Local/regional | LAS, LAZ, GeoTIFF |

### CesiumJS Terrain Example

```javascript
import { Viewer, createWorldTerrainAsync, Cartesian3 } from 'cesium';

const viewer = new Viewer('cesiumContainer', {
  terrain: await createWorldTerrainAsync()
});

viewer.camera.flyTo({
  destination: Cartesian3.fromDegrees(86.925, 27.988, 30000),  // Everest
  orientation: { heading: 0, pitch: -0.5, roll: 0 }
});
```

### MapLibre 3D Terrain

```javascript
map.on('load', () => {
  map.addSource('terrain', {
    type: 'raster-dem',
    url: 'https://demotiles.maplibre.org/terrain-tiles/tiles.json',
    tileSize: 256
  });
  map.setTerrain({ source: 'terrain', exaggeration: 1.5 });
  map.addLayer({
    id: 'hillshade',
    type: 'hillshade',
    source: 'terrain',
    paint: { 'hillshade-shadow-color': '#473B24' }
  });
});
```

### deck.gl TerrainLayer

```javascript
import { TerrainLayer } from '@deck.gl/geo-layers';

const terrain = new TerrainLayer({
  elevationDecoder: { rScaler: 256, gScaler: 1, bScaler: 1 / 256, offset: -32768 },
  elevationData: 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png',
  texture: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
  meshMaxError: 4.0
});
```

---

## CesiumJS + 3D Tiles Ecosystem

CesiumJS is the most comprehensive platform for 3D geospatial, with native support for the entire 3D Tiles ecosystem.

### 3D Tiles Content Types

| Type | Extension | Content | Use Case |
|------|-----------|---------|----------|
| Batched 3D Model | .b3dm | Textured meshes (buildings) | Urban models, BIM |
| Instanced 3D Model | .i3dm | Repeated models (trees, lights) | Vegetation, infrastructure |
| Point Cloud | .pnts | Colored points | LiDAR, photogrammetry |
| Composite | .cmpt | Mix of above | Complex scenes |
| glTF (3D Tiles 1.1) | .glb/.gltf | All content via glTF | Next-gen replacement for b3dm/i3dm |

### Loading 3D Tiles

```javascript
import { Cesium3DTileset, Cesium3DTileStyle } from 'cesium';

// Load from Cesium Ion
const buildings = await Cesium3DTileset.fromIonAssetId(96188);
viewer.scene.primitives.add(buildings);

// Style by property
buildings.style = new Cesium3DTileStyle({
  color: {
    conditions: [
      ["${height} > 100", "color('#ff0000')"],
      ["${height} > 50", "color('#ff8800')"],
      ["true", "color('#ffffff')"]
    ]
  }
});

// Google Photorealistic 3D Tiles
const google3D = await Cesium3DTileset.fromIonAssetId(2275207);
viewer.scene.primitives.add(google3D);
```

---

## Building / Urban Visualization

Rendering 3D buildings and urban models for planning, simulation, and analysis.

### Standards

- **3D Tiles 1.1:** OGC standard for streaming massive 3D datasets (buildings, terrain, point clouds)
- **CityGML:** OGC standard for city models with semantic information (LOD 1-4)
- **GeoJSON + height:** Simple extrusion from 2D footprints with height attributes

### Data Sources

- OpenStreetMap buildings (height tags)
- Google Photorealistic 3D Tiles (via Cesium Ion or direct API)
- Municipal open data (CityGML, 3D Tiles)
- Cesium OSM Buildings (global, free via Ion asset 96188)

### 3D Building Extrusion in MapLibre

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
      0, '#e0e0e0', 50, '#a0a0a0', 100, '#606060'
    ],
    'fill-extrusion-height': ['get', 'render_height'],
    'fill-extrusion-base': ['get', 'render_min_height'],
    'fill-extrusion-opacity': 0.7
  }
});
```

---

## Point Cloud Visualization

Rendering large LiDAR or photogrammetry point clouds in the browser.

### Potree -- The SOTA for Web Point Clouds

[Potree](https://potree.github.io) is the leading open-source web viewer for massive point clouds.

- **Repository:** [github.com/potree/potree](https://github.com/potree/potree)
- **Capacity:** Billions of points via hierarchical level-of-detail
- **Input:** LAS, LAZ, E57 (convert with PotreeConverter)
- **Features:** Measurement tools, cross-sections, classification coloring, elevation profiles

```bash
# Convert LAS to Potree format
PotreeConverter input.las -o output_potree --generate-page index

# Serve the output directory with any HTTP server
npx serve output_potree
```

### deck.gl PointCloudLayer

```javascript
import { PointCloudLayer } from '@deck.gl/layers';

const pointCloud = new PointCloudLayer({
  id: 'lidar',
  data: '/data/pointcloud.bin',
  getPosition: d => [d.x, d.y, d.z],
  getColor: d => [d.r, d.g, d.b],
  getNormal: [0, 0, 1],
  pointSize: 2,
  coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
  coordinateOrigin: [-122.4, 37.78]
});
```

### Tools Comparison

| Tool | Format Support | Max Points | Web-Based | Best For |
|------|---------------|------------|-----------|----------|
| Potree | LAS, LAZ, Potree | Billions | Yes | Large LiDAR datasets |
| CesiumJS | 3D Tiles (pnts) | Billions | Yes | Globe-context point clouds |
| Three.js + loaders | LAS, PLY | Millions | Yes | Custom point cloud viz |
| deck.gl PointCloudLayer | Binary | Millions | Yes | Analytical overlays |
| CloudCompare | LAS, LAZ, E57 | Billions | No (desktop) | Processing & classification |

### Point Cloud Optimization Tips

- **Level of detail:** Use octree-based LOD (Potree, 3D Tiles) for streaming
- **Compression:** LAZ (compressed LAS) reduces file size 5-10x
- **Tiling:** Split large datasets into spatial tiles for parallel loading
- **Color encoding:** Use classification or elevation-based coloring instead of RGB for smaller files
- **Budget:** Set a point budget (e.g., 2M visible points) to maintain frame rate

---

## Volumetric Data

Visualizing 3D continuous fields such as atmospheric data, ocean currents, or subsurface geology.

### Approaches

- **Isosurfaces:** Extract surfaces at specific values from 3D grids
- **Volume rendering:** Ray-casting through 3D scalar fields
- **Slice planes:** Cross-sections through volumetric datasets
- **Particle tracing:** Animate particles through 3D vector fields

### Tools

- CesiumJS (voxel support via 3D Tiles, experimental)
- ParaView (scientific visualization, web export via ParaView Glance)
- deck.gl (custom layers with WebGL shaders)
- Three.js (custom shaders, volume rendering)

---

## Tools & Platforms Comparison

| Tool/Platform | Terrain | Buildings | Point Clouds | Volumetric | Web | Best For |
|---------------|---------|-----------|--------------|------------|-----|----------|
| CesiumJS | Yes | Yes (3D Tiles) | Yes | Partial | Yes | Full 3D globe platform |
| deck.gl | TerrainLayer | Extrusion | PointCloudLayer | Custom | Yes | Data-driven 3D layers |
| MapLibre GL JS | Yes | Extrusion | No | No | Yes | Lightweight 3D maps |
| Three.js | Manual | Manual | Yes | Yes | Yes | Custom 3D applications |
| Potree | No | No | Yes (billions) | No | Yes | Dedicated point cloud viewer |
| QGIS 3D View | Yes | Yes | Yes | No | No | Desktop 3D analysis |
| ArcGIS Scene | Yes | Yes | Yes | Yes | Yes | Enterprise 3D GIS |
| Unity/Unreal | Yes | Yes | Yes | Yes | Export | Gaming-grade 3D geo |

### Performance & Hardware Guidelines

| Use Case | Min GPU | Min RAM | Recommended |
|----------|---------|---------|-------------|
| MapLibre 3D terrain | Integrated | 4 GB | Any modern device |
| CesiumJS + 3D Tiles | Dedicated (2 GB VRAM) | 8 GB | GTX 1060+ / M1+ |
| Potree (1B+ points) | Dedicated (4 GB VRAM) | 16 GB | RTX 3060+ / M1 Pro+ |
| deck.gl (1M+ features) | Dedicated (2 GB VRAM) | 8 GB | GTX 1060+ / M1+ |
| Volume rendering | Dedicated (4 GB+ VRAM) | 16 GB | RTX 3070+ |
