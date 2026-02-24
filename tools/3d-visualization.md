# 3D GIS, Visualization & Cartography

> **Quick Picks**
> - **SOTA**: CesiumJS + 3D Tiles (OGC standard, production-grade 3D streaming)
> - **Free Best**: QGIS 3D + Blender GIS (photorealistic renders, no license cost)
> - **Fastest Setup**: kepler.gl (drop CSV → instant 3D visualization)
> - **Best for Python**: pydeck (deck.gl layers in Jupyter, HTML export)
> - **Best for Print**: QGIS Print Layout with Atlas generation
> - **Best Terrain Renders**: rayshader (R) or Blender GIS

---

## 3D Web Globes & Viewers

### CesiumJS

CesiumJS is the de facto open-source standard for browser-based 3D geospatial visualization. It powers everything from NASA mission planning to real-estate digital twins. The library handles ellipsoidal Earth geometry, accurate sun/moon lighting, atmospheric scattering, and time-dynamic simulation out of the box.

**Install:**

```bash
npm install cesium
# or CDN
<script src="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Cesium.js"></script>
```

**Minimal globe viewer:**

```html
<!DOCTYPE html>
<html>
<head>
  <link href="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
  <script src="https://cesium.com/downloads/cesiumjs/releases/1.114/Build/Cesium/Cesium.js"></script>
  <style>
    html, body, #cesiumContainer { width: 100%; height: 100%; margin: 0; padding: 0; overflow: hidden; }
  </style>
</head>
<body>
  <div id="cesiumContainer"></div>
  <script>
    // Get a free token at cesium.com/ion/tokens
    Cesium.Ion.defaultAccessToken = 'YOUR_TOKEN_HERE';

    const viewer = new Cesium.Viewer('cesiumContainer', {
      terrain: Cesium.Terrain.fromWorldTerrain(),  // Cesium World Terrain
      timeline: true,
      animation: true,
    });

    // Add OSM Buildings as 3D Tiles (free via Cesium Ion)
    const osmBuildings = await Cesium.createOsmBuildingsAsync();
    viewer.scene.primitives.add(osmBuildings);

    // Fly to a location
    viewer.camera.flyTo({
      destination: Cesium.Cartesian3.fromDegrees(-87.6298, 41.8781, 3000), // Chicago
      orientation: { pitch: Cesium.Math.toRadians(-45) }
    });
  </script>
</body>
</html>
```

#### 3D Tiles: Streaming Massive Datasets

3D Tiles is an OGC standard for tiling and streaming heterogeneous 3D content — buildings, point clouds, photogrammetry meshes, vector features. Think "WMS/WMTS but for 3D."

```javascript
// Load a custom 3D Tiles tileset
const tileset = await Cesium.Cesium3DTileset.fromUrl(
  'https://your-server.com/tileset.json'
);
viewer.scene.primitives.add(tileset);
viewer.zoomTo(tileset);

// Data-driven styling: color buildings by height
tileset.style = new Cesium.Cesium3DTileStyle({
  color: {
    conditions: [
      ['${height} >= 100', 'color("red")'],
      ['${height} >= 50',  'color("orange")'],
      ['${height} >= 20',  'color("yellow")'],
      ['true',             'color("white")']
    ]
  },
  // Show only buildings taller than 5m
  show: '${height} > 5'
});

// Style by attribute (e.g., energy use kWh/m2)
tileset.style = new Cesium.Cesium3DTileStyle({
  color: `color('hsl(${240 - (${energy_use} / 300 * 240)}, 100%, 50%)')`
});
```

#### CZML: Time-Dynamic Visualization

CZML (Cesium Language) is a JSON format for describing time-dynamic scenes. Use it for satellite orbits, vehicle tracks, hurricane paths, or any moving object.

```javascript
const czml = [
  { id: "document", name: "CZML Demo", version: "1.0" },
  {
    id: "satellite-ISS",
    name: "ISS",
    availability: "2024-01-01T00:00:00Z/2024-01-01T01:30:00Z",
    billboard: {
      image: "data:image/png;base64,...",
      scale: 0.5
    },
    label: { text: "ISS", font: "14pt sans-serif" },
    position: {
      referenceFrame: "INERTIAL",
      interpolationAlgorithm: "LAGRANGE",
      interpolationDegree: 5,
      epoch: "2024-01-01T00:00:00Z",
      cartographicDegrees: [
        // [time_seconds, lon, lat, alt_meters, ...]
        0,    -80.0,  28.5, 408000,
        900,  -60.0,  35.0, 408000,
        1800, -40.0,  40.0, 408000
      ]
    },
    path: {
      resolution: 120,
      material: { solidColor: { color: { rgba: [255, 255, 0, 255] } } },
      width: 2
    }
  }
];

const dataSource = await Cesium.CzmlDataSource.load(czml);
viewer.dataSources.add(dataSource);
viewer.clock.shouldAnimate = true;
```

#### Entity API: Points, Lines, Polygons, Models

```javascript
// Add a glTF model
viewer.entities.add({
  name: 'Wind Turbine',
  position: Cesium.Cartesian3.fromDegrees(-103.0, 40.0),
  model: {
    uri: './turbine.glb',
    scale: 1.0,
    minimumPixelSize: 128,
    maximumScale: 20000
  }
});

// Draw an extruded polygon (building footprint)
viewer.entities.add({
  polygon: {
    hierarchy: Cesium.Cartesian3.fromDegreesArray([
      -87.63, 41.88,
      -87.62, 41.88,
      -87.62, 41.87,
      -87.63, 41.87
    ]),
    extrudedHeight: 150,
    height: 0,
    material: Cesium.Color.BLUE.withAlpha(0.6),
    outline: true,
    outlineColor: Cesium.Color.WHITE
  }
});

// Animated polyline (dashed, moving)
viewer.entities.add({
  polyline: {
    positions: Cesium.Cartesian3.fromDegreesArray([-87.6, 41.8, -87.5, 41.9]),
    width: 3,
    material: new Cesium.PolylineDashMaterialProperty({
      color: Cesium.Color.YELLOW,
      dashLength: 16
    })
  }
});
```

#### Terrain & Imagery

```javascript
// Custom terrain from Maptiler (free tier available)
const viewer = new Cesium.Viewer('cesiumContainer', {
  terrain: new Cesium.Terrain(
    Cesium.CesiumTerrainProvider.fromUrl('https://api.maptiler.com/tiles/terrain-quantized-mesh/?key=YOUR_KEY')
  )
});

// Add WMS imagery layer
viewer.imageryLayers.addImageryProvider(
  new Cesium.WebMapServiceImageryProvider({
    url: 'https://ows.terrestris.de/osm/service',
    layers: 'OSM-WMS',
    parameters: { transparent: true, format: 'image/png' }
  })
);

// Exaggerate terrain for dramatic effect
viewer.scene.verticalExaggeration = 2.5;
```

#### Cesium Ion: Free Cloud Hosting

Cesium Ion provides 5 GB free storage for tiling and hosting 3D data:
- Upload: LAS/LAZ point clouds, IFC, FBX, OBJ, GeoTIFF, KML/KMZ, GeoJSON, Shapefile
- Ion auto-converts to 3D Tiles or terrain quantized mesh
- API for programmatic upload: `cesium.com/docs/rest-api/`

```bash
# Upload via CLI (cesium-ion-cli npm package)
npm install -g @cesium/ion-cli
ion auth login
ion asset upload --name "MyBuildings" --type "3DTILES" ./buildings.shp
```

#### Framework Integrations

```bash
# React
npm install resium cesium
```

```jsx
// React component with Resium
import { Viewer, Entity, PointGraphics, CzmlDataSource } from 'resium';
import * as Cesium from 'cesium';

function GlobeApp() {
  return (
    <Viewer full terrain={Cesium.Terrain.fromWorldTerrain()}>
      <Entity
        name="Tokyo"
        position={Cesium.Cartesian3.fromDegrees(139.767, 35.681, 100)}
      >
        <PointGraphics pixelSize={10} color={Cesium.Color.RED} />
      </Entity>
    </Viewer>
  );
}
```

```bash
# Vue
npm install vue-cesium
# Angular
npm install angular-cesium
```

---

### iTowns (IGN France)

iTowns is a Three.js-based 3D geospatial viewer developed by IGN France. It excels at European geospatial data formats and oblique imagery — something CesiumJS does not natively support.

**Install:**

```bash
npm install itowns
```

**Basic setup:**

```javascript
import * as itowns from 'itowns';

// Create view
const extent = new itowns.Extent('EPSG:4326', -180, 180, -90, 90);
const view = new itowns.GlobeView(document.getElementById('viewerDiv'), extent);

// Add OSM basemap
itowns.Fetcher.json('./layers/JSONLayers/Ortho.json').then(config => {
  view.addLayer(new itowns.ColorLayer('Ortho', { source: new itowns.WMTSSource(config) }));
});

// Add 3D Tiles (point cloud)
const pointCloudLayer = new itowns.OGC3DTilesLayer('pointcloud', {
  source: new itowns.OGC3DTilesSource({
    url: 'http://server/tileset.json',
  }),
});
view.addLayer(pointCloudLayer);

// Oblique imagery (iTowns-specific feature)
const obliqueLayer = new itowns.OrientedImageLayer('oblique', {
  source: new itowns.OrientedImageSource({ url: './oblique-catalog.json' })
});
view.addLayer(obliqueLayer);
```

**Key advantage over CesiumJS**: Native COPC (Cloud Optimized Point Cloud) streaming and oblique image display for photogrammetric surveys.

---

### deck.gl 3D Layers

deck.gl runs on WebGL2/WebGPU and handles hundreds of millions of data points. Its 3D layers are ideal when you need to visualize data (not photorealistic scenes).

**Install:**

```bash
npm install deck.gl @deck.gl/core @deck.gl/layers @deck.gl/geo-layers
```

**3D buildings from 3D Tiles:**

```javascript
import { DeckGL } from 'deck.gl';
import { Tile3DLayer } from '@deck.gl/geo-layers';
import { CesiumIonLoader } from '@loaders.gl/3d-tiles';

const App = () => (
  <DeckGL
    initialViewState={{ longitude: 13.4, latitude: 52.5, zoom: 14, pitch: 60 }}
    controller={true}
  >
    <Tile3DLayer
      id="3d-tiles"
      data="https://assets.cesium.com/43978/tileset.json"  // OSM Buildings on Ion
      loader={CesiumIonLoader}
      loadOptions={{ 'cesium-ion': { accessToken: 'YOUR_TOKEN' } }}
      getPointColor={[0, 128, 200]}
      pointSize={2}
    />
  </DeckGL>
);
```

**LiDAR point cloud visualization:**

```javascript
import { PointCloudLayer } from '@deck.gl/layers';
import { LASLoader } from '@loaders.gl/las';

new PointCloudLayer({
  id: 'lidar',
  data: './scan.las',
  loaders: [LASLoader],
  pointSize: 2,
  getPosition: d => d.position,
  getColor: d => {
    // Color by intensity
    const i = d.intensity / 65535;
    return [i * 255, i * 200, 100, 255];
  },
  coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
  coordinateOrigin: [-117.75, 32.89]
});
```

**H3 hexbin 3D visualization:**

```javascript
import { H3HexagonLayer } from '@deck.gl/geo-layers';
import * as h3 from 'h3-js';

// Aggregate point data into H3 hexagons
const hexData = aggregateToH3(rawPoints, resolution=8);

new H3HexagonLayer({
  id: 'h3-hex',
  data: hexData,
  getHexagon: d => d.hex,
  getElevation: d => d.count * 50,     // 3D extrusion
  getFillColor: d => colorScale(d.count),
  elevationScale: 1,
  extruded: true,
  coverage: 0.9
});
```

**Pseudo-3D buildings without 3D Tiles (fast trick):**

```javascript
import { MVTLayer } from '@deck.gl/geo-layers';

// Uses Mapbox/MapTiler vector tiles with building height attribute
new MVTLayer({
  id: 'buildings-3d',
  data: 'https://api.maptiler.com/tiles/v3/{z}/{x}/{y}.pbf?key=YOUR_KEY',
  extruded: true,
  getElevation: f => (f.properties.render_height || f.properties.height || 5),
  getFillColor: f => {
    const h = f.properties.render_height || 0;
    return [
      Math.min(255, 100 + h * 0.5),
      Math.min(255, 150 + h * 0.3),
      200, 200
    ];
  }
});
```

---

### Mapbox/MapLibre Globe (v4+)

MapLibre GL JS v4 introduced a genuine 3D globe projection with atmosphere and fog — no CesiumJS needed for globe views in data visualization contexts.

```javascript
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  projection: 'globe',          // Enable 3D globe
  center: [0, 20],
  zoom: 1.5
});

map.on('style.load', () => {
  // Add atmosphere
  map.setFog({
    color: 'rgb(186, 210, 235)',
    'high-color': 'rgb(36, 92, 223)',
    'horizon-blend': 0.02,
    'space-color': 'rgb(11, 11, 25)',
    'star-intensity': 0.6
  });

  // Add 3D terrain
  map.addSource('terrain', {
    type: 'raster-dem',
    url: 'https://demotiles.maplibre.org/terrain-tiles/tiles.json'
  });
  map.setTerrain({ source: 'terrain', exaggeration: 1.5 });

  // 3D buildings extrusion
  map.addLayer({
    id: 'buildings',
    source: 'composite',
    'source-layer': 'building',
    type: 'fill-extrusion',
    paint: {
      'fill-extrusion-height': ['get', 'height'],
      'fill-extrusion-base': ['get', 'min_height'],
      'fill-extrusion-color': [
        'interpolate', ['linear'], ['get', 'height'],
        0,   '#1a1a2e',
        50,  '#16213e',
        100, '#0f3460',
        200, '#533483'
      ],
      'fill-extrusion-opacity': 0.9
    }
  });
});
```

---

## Desktop 3D GIS

### QGIS 3D View

QGIS 3D is built on Qt3D and provides a capable in-app 3D viewer directly tied to your project data — no external software required.

**Enable 3D view:** `View → 3D Map Views → New 3D Map View`

**Configure terrain:**

```python
# PyQGIS: configure 3D terrain from a DEM layer
from qgis.core import (Qgs3DMapSettings, QgsDemTerrainGenerator,
                        QgsRasterLayer, QgsProject)

settings = Qgs3DMapSettings()

# Set DEM terrain
dem_layer = QgsRasterLayer('/data/srtm_30m.tif', 'DEM')
QgsProject.instance().addMapLayer(dem_layer)

terrain_gen = QgsDemTerrainGenerator()
terrain_gen.setLayer(dem_layer)
terrain_gen.setResolution(16)
settings.setTerrainGenerator(terrain_gen)
settings.setTerrainVerticalScale(1.5)   # Vertical exaggeration
settings.setTerrainShadingEnabled(True)
```

**Extrude OSM buildings by `building:levels`:**

1. Load OSM buildings polygon layer
2. Open `Layer Properties → 3D View`
3. Set `Height`: `levels * 3` (expression: `coalesce("building:levels", 1) * 3`)
4. Set `Altitude clamping`: Terrain
5. Enable `Cast shadows`

```python
# PyQGIS: set 3D renderer for buildings layer
from qgis.core import (QgsVectorLayer3DRenderer, QgsPolygon3DSymbol,
                        QgsPhongMaterialSettings)
from qgis._3d import QgsVectorLayer3DRenderer, QgsPolygon3DSymbol

symbol = QgsPolygon3DSymbol()
symbol.setHeight(0)
symbol.setExtrusionHeight(0)  # Will be overridden by data-defined

# Data-defined extrusion height
symbol.dataDefinedProperties().setProperty(
    QgsAbstract3DSymbol.PropertyHeight,
    QgsProperty.fromExpression("coalesce(\"building:levels\", 1) * 3")
)

mat = QgsPhongMaterialSettings()
mat.setAmbient(QColor(180, 180, 220))
mat.setDiffuse(QColor(200, 200, 255))
mat.setSpecular(QColor(255, 255, 255))
mat.setShininess(32)
symbol.setMaterialSettings(mat)

renderer = QgsVectorLayer3DRenderer()
renderer.setSymbol(symbol)
buildings_layer.setRenderer3D(renderer)
```

**3D flythrough animation export:**

```python
# QGIS 3D animation: export frames for ffmpeg
# Use QGIS GUI: 3D Map View → Animations → Add keyframes → Export
# Then in terminal:
import subprocess
subprocess.run([
    'ffmpeg', '-r', '30', '-i', '/tmp/frame_%04d.png',
    '-vcodec', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p',
    'flythrough.mp4'
])
```

**Point cloud rendering (LAS/LAZ/COPC):**

- Drag-drop `.laz` or `.copc.laz` into QGIS
- Render modes: `Elevation`, `RGB`, `Classification`, `Intensity`, `Return Number`
- 3D view renders point clouds natively with octree streaming

```python
# Style a point cloud by classification
from qgis.core import QgsPointCloudLayer, QgsPointCloudClassifiedRenderer

layer = QgsPointCloudLayer('/data/lidar.copc.laz', 'LiDAR', 'pdal')

renderer = QgsPointCloudClassifiedRenderer()
renderer.setAttribute('Classification')
# Default ASPRS classification colors are applied automatically
layer.setRenderer(renderer)
QgsProject.instance().addMapLayer(layer)
```

---

### ArcGIS Pro 3D

ArcGIS Pro provides the most feature-complete desktop 3D GIS experience, including subsurface voxel visualization that no open-source tool matches.

**Scene types:**
- **Local scene**: projected CRS, flat Earth, precise engineering applications
- **Global scene**: geographic CRS, spherical Earth, continental/global scale

**Voxel layers for subsurface geology:**

```python
import arcpy

# Create voxel layer from netCDF (e.g., soil moisture, air quality)
arcpy.md.MakeNetCDFVoxelLayer(
    in_netcdf_file='soil_moisture_3d.nc',
    variable='sm',
    x_dimension='lon',
    y_dimension='lat',
    z_dimension='depth',
    out_voxel_layer='SoilMoisture3D',
    band_dimension='time',
    dimension_values=[['time', '2024-06-01']]
)
```

**Reality mapping workflow:**

1. Collect drone imagery with RTK GPS
2. Process in ArcGIS Reality for Desktop → orthoimage + DSM + 3D mesh
3. Publish 3D mesh as Scene Layer Package (`.slpk`) to ArcGIS Online
4. Load in Scene Viewer or ArcGIS Pro

**CityEngine integration:**

```python
# Export QGIS/ArcGIS features to CityEngine CGA rules
# Shapefile with height attributes → procedural 3D city model
# In CityEngine:
# 1. Import shapefile with building footprints
# 2. Apply rule: Lot --> extrude(height) comp(f) { top: Mass. | side: Mass. }
# 3. Export as SLPK, FBX, or GLTF
```

---

### Blender GIS

Blender is the most powerful tool for creating photorealistic 3D terrain renders, architectural visualizations, and animated flythrough videos. The BlenderGIS addon bridges the gap between GIS data and Blender's rendering engine.

**Install BlenderGIS:**

1. Download from `github.com/domlysz/BlenderGIS`
2. Blender → Edit → Preferences → Add-ons → Install from file
3. Enable `BlenderGIS`

**Import terrain workflow:**

```python
# Run in Blender Python console (Scripting workspace)
import bpy

# 1. Import DEM as terrain mesh
# Via BlenderGIS menu: GIS → Import → Georaster as DEM
# Or programmatically:
bpy.ops.importgis.georaster(
    filepath='/data/srtm_n40e013.tif',
    importMode='DEM',
    subdivision='mesh',
    rastCRS='EPSG:4326'
)

# 2. Import satellite imagery as texture
bpy.ops.importgis.georaster(
    filepath='/data/satellite_rgb.tif',
    importMode='PLANE',
    rastCRS='EPSG:32633'
)

# 3. Import building footprints as 3D objects
bpy.ops.importgis.shapefile(
    filepath='/data/buildings.shp',
    fieldElevName='',       # Ground elevation attribute
    fieldExtrudeName='height',  # Extrusion attribute
    extrusionAxis='Z',
    separateObjects=False
)
```

**Cinematic terrain render with satellite texture:**

```python
import bpy
import math

# Set up camera for aerial view
cam = bpy.data.cameras.new('Camera')
cam_obj = bpy.data.objects.new('Camera', cam)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Position above terrain
cam_obj.location = (1000, -2000, 3000)
cam_obj.rotation_euler = (math.radians(60), 0, math.radians(0))
cam.lens = 50

# Set up Cycles renderer for photorealism
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.samples = 128
bpy.context.scene.render.resolution_x = 3840
bpy.context.scene.render.resolution_y = 2160

# Add atmospheric haze using volumetric world shader
world = bpy.data.worlds['World']
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs[0].default_value = (0.8, 0.9, 1.0, 1.0)  # Sky blue
bg.inputs[1].default_value = 1.5

# Render
bpy.ops.render.render(write_still=True)
```

**blosm addon (OSM buildings with auto-texturing):**

```bash
# Install blosm from blendermarket.com or github.com/vvoovv/blosm
# In Blender: N panel → blosm tab
# 1. Set bounding box by map coordinates
# 2. Click "Get from OpenStreetMap"
# 3. Select "3D buildings" option
# 4. Import → auto-generates LOD2 city
```

**Geometry nodes for vegetation placement:**

```python
# Procedural tree placement on forest land cover
# 1. Import land cover raster (CORINE, NLCD)
# 2. Use geometry nodes to scatter tree objects on forest class polygons

import bpy

# Get the terrain mesh
terrain = bpy.data.objects['DEM']
terrain.select_set(True)

# Add geometry nodes modifier
geo_nodes = terrain.modifiers.new('GeoNodes', 'NODES')

# The node tree (normally built in GUI, but scriptable):
# Input: terrain mesh
# [Mesh to Points] → sample land cover attribute
# [Delete Geometry] where class != 'forest'
# [Instance on Points] with tree collection
# [Random rotation/scale] for natural look
# Output: instanced trees
```

**Blender export for web:**

```python
# Export scene as glTF for CesiumJS/Three.js
bpy.ops.export_scene.gltf(
    filepath='/output/terrain_scene.glb',
    export_format='GLB',
    export_draco_mesh_compression_enable=True,   # Compress geometry
    export_draco_mesh_compression_level=6,
    export_materials='EXPORT',
    export_apply=True
)
```

---

## 3D Data Formats & Standards

### 3D Tiles (OGC Standard)

3D Tiles 1.1 introduced implicit tiling (no per-tile manifest files) and metadata (per-feature properties accessible in shaders). It is now an official OGC Community Standard.

**Tileset structure:**

```json
{
  "asset": { "version": "1.1" },
  "geometricError": 512,
  "root": {
    "boundingVolume": {
      "region": [-2.0, 0.5, -1.9, 0.6, 0, 500]
    },
    "geometricError": 256,
    "implicitTiling": {
      "subdivisionScheme": "QUADTREE",
      "subtreeLevels": 4,
      "availableLevels": 12,
      "subtrees": { "uri": "subtrees/{level}/{x}/{y}.subtree" }
    },
    "content": { "uri": "content/{level}/{x}/{y}.glb" }
  }
}
```

**Convert CityGML to 3D Tiles with py3dtiles:**

```bash
pip install py3dtiles

# Convert LAS point cloud to 3D Tiles
py3dtiles convert input.las --out ./tiles/ --srs_in 2154 --srs_out 4978

# The output tileset.json can be served statically and loaded in CesiumJS
```

**Convert CityGML with citygml-tools:**

```bash
# Download: github.com/citygml4j/citygml-tools
./citygml-tools to-cityjson ./building.gml
./citygml-tools upgrade ./building.gml   # Upgrade to CityGML 3.0

# FME (commercial) for production pipelines:
# CityGML Reader → 3D Tiles Writer
# Supports LOD selection, attribute mapping, texture baking
```

**3d-tiles-tools (Cesium official):**

```bash
npm install -g 3d-tiles-tools

# Convert tileset formats
npx 3d-tiles-tools convert -i ./input/ -o ./output/
# Merge multiple tilesets
npx 3d-tiles-tools merge -i tileset1/ tileset2/ -o merged/
# Upgrade to 3D Tiles 1.1
npx 3d-tiles-tools upgrade -i v10/ -o v11/
# Validate tileset structure
npx 3d-tiles-tools validate -i ./tileset.json
# Pipeline: reformat B3DM to GLB
npx 3d-tiles-tools pipeline -i pipeline.json
```

---

### CityGML / CityJSON

CityGML is the OGC standard for semantic 3D city models. CityJSON is its JSON serialization — 10x more compact and developer-friendly.

**CityJSON with cjio:**

```bash
pip install cjio

# Inspect a file
cjio mybuildings.city.json info

# Reproject to WGS84
cjio mybuildings.city.json reproject --epsg 4326 save output.city.json

# Extract specific LOD
cjio mybuildings.city.json lod_filter --lod 2 save lod2.city.json

# Remove textures to reduce file size
cjio mybuildings.city.json textures_remove save notex.city.json

# Convert to OBJ for Blender import
cjio mybuildings.city.json export --format obj
```

**CityJSON in Python:**

```python
import cjio.cityjson as cityjson
import json

# Load CityJSON
with open('rotterdam.city.json') as f:
    cm = cityjson.load(f)

# Iterate buildings
for co_id, co in cm.cityobjects.items():
    if co.type == 'Building':
        geoms = co.geometry
        for geom in geoms:
            if geom.lod == '2':
                # Access geometry vertices
                vertices = [cm.vertices[v] for v in geom.get_vertices()]
                height = co.attributes.get('measuredHeight', 0)
                year = co.attributes.get('yearOfConstruction', 'unknown')
                print(f"Building {co_id}: {height}m, built {year}")

# Merge two CityJSON files
cm2 = cityjson.load(open('amsterdam.city.json'))
cm.merge([cm2])
cm.remove_duplicate_vertices()
cityjson.save(cm, 'nl_merged.city.json')
```

**Free CityGML/CityJSON Data Sources:**

| Country/City | LOD | URL |
|---|---|---|
| Germany (NRW) | LOD2 | `opengeodata.nrw.de` |
| Netherlands | LOD1.2, LOD2.2, LOD3.3 | `3dbag.nl` |
| Singapore | LOD1, LOD2 | `data.gov.sg` |
| New York City | LOD1 | `github.com/CityOfNewYork/nyc-geo-metadata` |
| Helsinki | LOD2 | `hri.fi` |
| Zürich | LOD2 | `data.stadt-zuerich.ch` |

---

### IFC (BIM Integration)

IFC (Industry Foundation Classes) is the open standard for Building Information Models. Integrating BIM with GIS enables digital twin applications spanning infrastructure, urban planning, and facility management.

**IfcOpenShell: Python IFC parsing:**

```bash
pip install ifcopenshell
```

```python
import ifcopenshell
import ifcopenshell.geom
import numpy as np

# Open IFC file
ifc = ifcopenshell.open('office_building.ifc')

# Extract all walls
walls = ifc.by_type('IfcWall')
for wall in walls:
    name = wall.Name
    # Get material
    for rel in wall.IsDefinedBy:
        if rel.is_a('IfcRelDefinesByProperties'):
            props = rel.RelatingPropertyDefinition
            if props.is_a('IfcPropertySet'):
                for prop in props.HasProperties:
                    print(f"  {prop.Name}: {prop.NominalValue}")

# Extract 3D geometry with triangulation
settings = ifcopenshell.geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

for product in ifc.by_type('IfcProduct'):
    if product.Representation:
        try:
            shape = ifcopenshell.geom.create_shape(settings, product)
            verts = shape.geometry.verts   # Flat float array [x,y,z,x,y,z,...]
            faces = shape.geometry.faces   # Flat int array [i,j,k,i,j,k,...]
            # Convert to numpy for further processing
            verts_np = np.array(verts).reshape(-1, 3)
            faces_np = np.array(faces).reshape(-1, 3)
        except:
            pass

# IFC to GeoJSON (georeferenced building footprint)
# Using IfcOpenShell + shapely
from shapely.geometry import Polygon, mapping
import json

features = []
for slab in ifc.by_type('IfcSlab'):
    if slab.PredefinedType == 'FLOOR':
        shape = ifcopenshell.geom.create_shape(settings, slab)
        v = np.array(shape.geometry.verts).reshape(-1, 3)
        # Get XY convex hull as footprint
        from shapely.geometry import MultiPoint
        footprint = MultiPoint(v[:, :2]).convex_hull
        features.append({
            'type': 'Feature',
            'geometry': mapping(footprint),
            'properties': {'name': slab.Name, 'level': slab.ObjectType}
        })

with open('bim_footprints.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': features}, f)
```

**IFC to 3D Tiles pipeline:**

```bash
# FME (commercial): IFC Reader → CityGML Writer → 3D Tiles Writer
# Open source alternative: IFCjs → Three.js → export GLTF → py3dtiles

# Using xeokit-bim-viewer (open source BIM web viewer):
npm install @xeokit/xeokit-sdk

# Convert IFC to xeokit's XKT format:
npm install -g @xeokit/xeokit-convert
xeokit-convert -s building.ifc -o building.xkt
```

---

## Data-Driven Visualization

### kepler.gl

kepler.gl is the fastest path from raw geospatial data to a stunning interactive visualization. No coding required for basic use; full React integration available for advanced apps.

**Web app (zero install):**

Visit `kepler.gl` → drag and drop CSV, GeoJSON, or JSON → instant visualization.

**Jupyter widget:**

```bash
pip install keplergl
```

```python
from keplergl import KeplerGl
import pandas as pd

# Load GPS trajectory data
df = pd.read_csv('uber_rides_nyc.csv')
# Expected columns: pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude

# Create map
m = KeplerGl(height=600)
m.add_data(data=df, name='NYC Rides')

# Display in Jupyter
m

# Save to HTML
m.save_to_html(file_name='nyc_rides.html')
```

**Configuration for 3D hexbin:**

```python
config = {
    'version': 'v1',
    'config': {
        'visState': {
            'layers': [{
                'id': 'hex_layer',
                'type': 'hexagon',
                'config': {
                    'dataId': 'NYC Rides',
                    'label': 'Ride Density',
                    'color': [255, 153, 31],
                    'columns': {
                        'lat': 'pickup_latitude',
                        'lng': 'pickup_longitude'
                    },
                    'visConfig': {
                        'opacity': 0.8,
                        'worldUnitSize': 0.3,   # km
                        'resolution': 8,
                        'colorRange': {
                            'name': 'Global Warming',
                            'type': 'sequential',
                            'category': 'Uber',
                            'colors': ['#5A1846','#900C3F','#C70039','#E3611C','#F1920E','#FFC300']
                        },
                        'coverage': 1,
                        'sizeRange': [0, 500],
                        'percentile': [0, 100],
                        'elevationPercentile': [0, 100],
                        'elevationScale': 5,
                        'enableElevationZoomFactor': True,
                        'colorAggregation': 'count',
                        'sizeAggregation': 'count',
                        'enable3d': True       # Enable 3D extrusion!
                    }
                }
            }]
        }
    }
}

m = KeplerGl(height=600, config=config)
m.add_data(data=df, name='NYC Rides')
m
```

**React integration:**

```bash
npm install kepler.gl react-redux redux
```

```jsx
import KeplerGl from 'kepler.gl';
import { addDataToMap } from 'kepler.gl/actions';
import { Provider, useDispatch } from 'react-redux';

function MapApp() {
  const dispatch = useDispatch();

  useEffect(() => {
    dispatch(addDataToMap({
      datasets: [{
        info: { id: 'trips', label: 'Trip Data' },
        data: { fields: [...], rows: [...] }
      }],
      config: { mapStyle: { styleType: 'dark' } }
    }));
  }, []);

  return (
    <KeplerGl
      id="map"
      mapboxApiAccessToken="YOUR_TOKEN"
      width={window.innerWidth}
      height={window.innerHeight}
    />
  );
}
```

---

### pydeck

pydeck is the Python interface to deck.gl. Every deck.gl layer is available, and outputs can be displayed in Jupyter, Streamlit, or exported as standalone HTML.

```bash
pip install pydeck
```

```python
import pydeck as pdk
import pandas as pd
import numpy as np

# Load earthquake data
df = pd.read_csv('earthquakes.csv')  # lon, lat, magnitude, depth columns

# Column layer: 3D cylinders extruded by magnitude
column_layer = pdk.Layer(
    'ColumnLayer',
    data=df,
    get_position=['lon', 'lat'],
    get_elevation='magnitude * 50000',
    elevation_scale=1,
    radius=20000,
    get_fill_color=[
        'magnitude > 7 ? 255 : magnitude > 6 ? 200 : 100',
        '50',
        '255 - magnitude * 30',
        200
    ],
    pickable=True,
    auto_highlight=True,
)

# Scatterplot layer for depth visualization
scatter_layer = pdk.Layer(
    'ScatterplotLayer',
    data=df,
    get_position=['lon', 'lat', '-depth * 1000'],  # depth as negative Z
    get_radius='magnitude * 5000',
    get_fill_color=[255, 165, 0, 150],
    pickable=True,
)

view_state = pdk.ViewState(
    longitude=130, latitude=35,
    zoom=4, pitch=50, bearing=30
)

r = pdk.Deck(
    layers=[column_layer, scatter_layer],
    initial_view_state=view_state,
    tooltip={
        'html': '<b>Magnitude:</b> {magnitude}<br><b>Depth:</b> {depth} km',
        'style': {'backgroundColor': 'steelblue', 'color': 'white'}
    },
    map_style='mapbox://styles/mapbox/dark-v10',
    api_keys={'mapbox': 'YOUR_TOKEN'}
)

# Jupyter display
r

# Export standalone HTML
r.to_html('earthquakes.html')
```

**Trip animation (GPS traces):**

```python
import pydeck as pdk

# GPS trip data: each row is a trip with timestamps
# Format: [{"path": [[lon,lat,ts], [lon,lat,ts]...], "color": [r,g,b]}]
trips_layer = pdk.Layer(
    'TripsLayer',
    data=trips_data,
    get_path='path',
    get_timestamps='timestamps',
    get_color='color',
    opacity=0.8,
    width_min_pixels=2,
    rounded=True,
    trail_length=600,     # Seconds of trail
    current_time=0,       # Controlled by animation
)
```

---

### Streamlit + pydeck (Rapid Dashboard)

```bash
pip install streamlit pydeck geopandas
```

```python
# app.py
import streamlit as st
import pydeck as pdk
import geopandas as gpd
import pandas as pd

st.set_page_config(layout="wide", page_title="3D City Explorer")
st.title("3D Building Explorer")

# Sidebar controls
city = st.sidebar.selectbox("City", ["New York", "London", "Tokyo"])
height_multiplier = st.sidebar.slider("Height Scale", 1.0, 10.0, 1.0)
color_by = st.sidebar.radio("Color By", ["Height", "Year Built", "Area"])

# Load buildings GeoJSON
@st.cache_data
def load_data(city):
    return gpd.read_file(f"data/{city.lower().replace(' ', '_')}_buildings.geojson")

gdf = load_data(city)
df = pd.DataFrame(gdf.drop(columns='geometry'))
df['lon'] = gdf.geometry.centroid.x
df['lat'] = gdf.geometry.centroid.y
df['height'] = gdf.geometry.area * 0.01  # Simplified

layer = pdk.Layer(
    'ColumnLayer',
    data=df,
    get_position=['lon', 'lat'],
    get_elevation=f'height * {height_multiplier}',
    radius=30,
    get_fill_color=[100, 150, 255, 200],
    pickable=True,
    auto_highlight=True,
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(longitude=df.lon.mean(), latitude=df.lat.mean(), zoom=13, pitch=45),
    tooltip={'text': 'Height: {height}m\nYear: {year_built}'}
))
```

```bash
streamlit run app.py
```

---

## Cartography & Print Maps

### QGIS Print Layout

QGIS Print Layout is a professional map composition engine supporting publication-quality output for scientific papers, planning documents, and wall maps.

**Atlas generation — automating a map series:**

```python
# PyQGIS: Generate an atlas map series programmatically
from qgis.core import (QgsProject, QgsPrintLayout, QgsLayoutItemMap,
                        QgsLayoutAtlas, QgsLayoutExporter, QgsRectangle)
from qgis.utils import iface
import os

project = QgsProject.instance()
layout_manager = project.layoutManager()

# Get existing layout (create in QGIS GUI first)
layout = layout_manager.layoutByName('Municipal Atlas')
atlas = layout.atlas()

# Configure atlas: iterate over municipality polygons
coverage_layer = project.mapLayersByName('Municipalities')[0]
atlas.setCoverageLayer(coverage_layer)
atlas.setEnabled(True)
atlas.setHideCoverage(False)
atlas.setFilterFeatures(False)
atlas.setSortFeatures(True)
atlas.setSortKeyAttributeName('municipality_name')

# Export all atlas pages to PDF
exporter = QgsLayoutExporter(layout)
settings = QgsLayoutExporter.PdfExportSettings()
settings.dpi = 300
settings.exportMetadata = True

output_dir = '/output/atlas_maps/'
os.makedirs(output_dir, exist_ok=True)

# Export individual PDFs per feature
for i in range(atlas.count()):
    atlas.seekTo(i)
    feat_name = atlas.currentFeature()['municipality_name']
    filename = os.path.join(output_dir, f"{feat_name}.pdf")
    exporter.exportToPdf(filename, settings)
    print(f"Exported: {filename}")

print(f"Generated {atlas.count()} maps")
```

**Dynamic expressions in Print Layout:**

```
# Title that changes per atlas feature
[%  'Population Map: ' || "municipality_name" %]

# Dynamic scale text
[% '1:' || format_number(@map_scale, 0) %]

# Date and attribution
[% 'Created: ' || format_date(now(), 'MMMM yyyy') || ' | Source: OSM, Statistics Bureau' %]

# Filter statistic to current atlas feature
[% with_variable('muni_id',  @atlas_featureid,
    aggregate(
        layer:='Population',
        aggregate:='sum',
        expression:="pop_2023",
        filter:= "municipality_id" = @muni_id
    )
) %]
```

**Label placement for dense maps:**

```python
# Configure Mapnik-quality labeling engine in QGIS
from qgis.core import (QgsPalLayerSettings, QgsVectorLayerSimpleLabeling,
                        QgsLabelLineSettings, QgsLabelObstacleSettings)

pal = QgsPalLayerSettings()
pal.fieldName = 'name'
pal.enabled = True

# Curved labels for roads
line_settings = QgsLabelLineSettings()
line_settings.setPlacementFlags(QgsLabelLineSettings.OnLine | QgsLabelLineSettings.AboveLine)
pal.setLineSettings(line_settings)

# Font settings
text_format = pal.format()
font = text_format.font()
font.setFamily('Noto Sans')
font.setPointSize(8)
font.setBold(False)
text_format.setFont(font)
text_format.setColor(QColor(30, 30, 30))

# Buffer/halo
buffer = text_format.buffer()
buffer.setEnabled(True)
buffer.setSize(0.8)
buffer.setColor(QColor(255, 255, 255, 200))
text_format.setBuffer(buffer)

pal.setFormat(text_format)

labeling = QgsVectorLayerSimpleLabeling(pal)
roads_layer.setLabeling(labeling)
roads_layer.setLabelsEnabled(True)
```

---

### MapLibre/Mapbox Style Specification

The MapLibre style specification is a JSON document defining the entire look of a vector tile map: sources, layers, colors, icons, fonts, and expressions.

**Maputnik: visual style editor:**

```bash
# Run locally
npx maputnik serve
# Access at http://localhost:8000
# Open existing style or start from a template
```

**Style with data-driven expressions:**

```json
{
  "version": 8,
  "name": "Urban Heat Island",
  "sources": {
    "census": {
      "type": "geojson",
      "data": "https://api.example.com/census-tracts.geojson"
    }
  },
  "layers": [
    {
      "id": "heat-choropleth",
      "type": "fill",
      "source": "census",
      "paint": {
        "fill-color": [
          "interpolate",
          ["linear"],
          ["get", "mean_summer_temp_c"],
          25, "#313695",
          28, "#4575b4",
          31, "#74add1",
          34, "#fdae61",
          37, "#f46d43",
          40, "#d73027",
          43, "#a50026"
        ],
        "fill-opacity": 0.75,
        "fill-outline-color": "rgba(255,255,255,0.2)"
      }
    },
    {
      "id": "heat-labels",
      "type": "symbol",
      "source": "census",
      "layout": {
        "text-field": ["concat", ["get", "neighborhood"], "\n", ["to-string", ["round", ["get", "mean_summer_temp_c"]]], "°C"],
        "text-font": ["Open Sans Bold"],
        "text-size": ["interpolate", ["linear"], ["zoom"], 11, 9, 15, 14]
      },
      "paint": {
        "text-color": "#ffffff",
        "text-halo-color": "rgba(0,0,0,0.5)",
        "text-halo-width": 1
      }
    }
  ]
}
```

**Protomaps: free, self-hostable basemap:**

```bash
# Download pre-built PMTiles basemap (planet = ~100GB, city = ~100MB)
wget https://build.protomaps.com/20240101.pmtiles

# Serve with any static file server
python -m http.server 8080
```

```javascript
import maplibregl from 'maplibre-gl';
import { Protocol } from 'pmtiles';

// Register PMTiles protocol
const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    glyphs: 'https://protomaps.github.io/basemaps-assets/fonts/{fontstack}/{range}.pbf',
    sprite: 'https://protomaps.github.io/basemaps-assets/sprites/v4/light',
    sources: {
      protomaps: {
        type: 'vector',
        url: 'pmtiles:///data/20240101.pmtiles',
        attribution: '<a href="https://protomaps.com">Protomaps</a> © <a href="https://openstreetmap.org">OpenStreetMap</a>'
      }
    },
    layers: [] // Add Protomaps layer definitions
  },
  center: [0, 20], zoom: 2
});
```

---

### ColorBrewer & Scientific Color Maps

**ColorBrewer in Python:**

```python
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# Sequential ColorBrewer (for population density, elevation)
cmap_sequential = plt.cm.YlOrRd     # Yellow-Orange-Red
cmap_diverging = plt.cm.RdBu_r      # Red-Blue diverging (temperature anomaly)
cmap_qualitative = plt.cm.Set2      # Categorical (land cover types)

# cmocean for oceanographic data
import cmocean
cmap_thermal = cmocean.cm.thermal   # Temperature
cmap_haline  = cmocean.cm.haline    # Salinity
cmap_topo    = cmocean.cm.topo      # Topography (diverging at sea level)
cmap_speed   = cmocean.cm.speed     # Current speed

# Scientific colour maps (Crameri) - perceptually uniform, colorblind-safe
import cmcrameri.cm as cmc
cmap_batlow  = cmc.batlow           # General purpose, colorblind-safe
cmap_vik     = cmc.vik              # Diverging (sea level change, anomalies)
cmap_romaO   = cmc.romaO            # Cyclic (wind direction, aspect)

# NEVER use this for scientific data:
bad_cmap = plt.cm.jet    # Perceptually non-uniform, misleading
bad_cmap = plt.cm.rainbow  # Same problem

# Correct temperature anomaly map
fig, ax = plt.subplots(figsize=(12, 6))
temp_anomaly = np.random.randn(90, 180) * 2  # Synthetic global data

im = ax.imshow(
    temp_anomaly,
    cmap=cmc.vik,           # Diverging, perceptually uniform
    vmin=-4, vmax=4,        # Symmetric around 0
    origin='upper',
    extent=[-180, 180, -90, 90]
)
plt.colorbar(im, ax=ax, label='Temperature Anomaly (°C)', shrink=0.6)
ax.set_title('Surface Temperature Anomaly vs 1981-2010 Baseline')
plt.savefig('temp_anomaly.png', dpi=300, bbox_inches='tight')
```

**ColorBrewer palettes for MapLibre:**

```javascript
// Sequential: 9-class YlOrRd (from colorbrewer2.org)
const ylOrRd9 = [
  '#ffffcc','#ffeda0','#fed976','#feb24c',
  '#fd8d3c','#fc4e2a','#e31a1c','#bd0026','#800026'
];

// Build interpolation expression for population density
const choroplethColor = [
  'step',
  ['get', 'pop_density'],
  ylOrRd9[0],
  10,  ylOrRd9[1],
  50,  ylOrRd9[2],
  100, ylOrRd9[3],
  200, ylOrRd9[4],
  500, ylOrRd9[5],
  1000,ylOrRd9[6],
  2000,ylOrRd9[7],
  5000,ylOrRd9[8]
];
```

---

## Animation & Time Series

### QGIS Temporal Controller

```python
# PyQGIS: configure a layer for temporal animation
from qgis.core import (QgsVectorLayerTemporalProperties,
                        QgsProject, QgsTemporalNavigationObject)
from PyQt5.QtCore import QDateTime

layer = QgsProject.instance().mapLayersByName('GPS_Tracks')[0]

# Configure temporal properties
temporal_props = layer.temporalProperties()
temporal_props.setIsActive(True)
temporal_props.setMode(
    QgsVectorLayerTemporalProperties.ModeFeatureDateTimeStartAndEndFromFields
)
temporal_props.setStartField('timestamp_start')
temporal_props.setEndField('timestamp_end')

# Access temporal controller from QGIS GUI
from qgis.utils import iface
controller = iface.mapCanvas().temporalController()
controller.setFrameDuration(QgsInterval(1, QgsUnitTypes.TemporalHours))
controller.setTemporalRangeCumulative(False)

# Export animation frames
from qgis.core import QgsMapRendererParallelJob, QgsMapSettings

# Use GUI: Temporal Controller panel → Export animation → Set output dir
```

**Google Earth Engine Timelapse with geemap:**

```bash
pip install geemap earthengine-api
earthengine authenticate
```

```python
import geemap
import ee

ee.Initialize()

# Landsat timelapse (1984-present, 3 lines!)
Map = geemap.Map()
Map.add_landsat_ts_gif(
    roi=ee.Geometry.BBox(-87.8, 41.6, -87.4, 42.0),  # Chicago
    start_year=1990,
    end_year=2023,
    start_date='06-01',
    end_date='09-01',
    bands=['Red', 'Green', 'Blue'],
    vis_params={'min': 0, 'max': 0.4},
    dimensions=800,
    frames_per_second=3,
    out_gif='chicago_timelapse.gif',
    title='Chicago 1990-2023',
    progress_bar_color='blue',
    add_text=True,
    font_size=30
)
```

**Sentinel Hub timelapse (Python):**

```python
from sentinelhub import (SHConfig, DataCollection, SentinelHubRequest,
                          BBox, CRS, MimeType, bbox_to_dimensions)
import datetime

config = SHConfig()
config.sh_client_id = 'YOUR_CLIENT_ID'
config.sh_client_secret = 'YOUR_CLIENT_SECRET'

aoi_bbox = BBox(bbox=[-87.7, 41.7, -87.5, 41.9], crs=CRS.WGS84)
size = bbox_to_dimensions(aoi_bbox, resolution=10)

# True color image for each month of 2023
images = []
for month in range(1, 13):
    start = datetime.date(2023, month, 1)
    end = datetime.date(2023, month, 28)

    request = SentinelHubRequest(
        evalscript="""
        //VERSION=3
        function setup() { return { input: ["B04","B03","B02"], output: { bands: 3 } }; }
        function evaluatePixel(s) { return [3.5*s.B04, 3.5*s.B03, 3.5*s.B02]; }
        """,
        input_data=[SentinelHubRequest.input_data(
            data_collection=DataCollection.SENTINEL2_L2A,
            time_interval=(start.isoformat(), end.isoformat()),
            mosaicking_order='leastCC'   # Least cloud cover
        )],
        responses=[SentinelHubRequest.output_response('default', MimeType.PNG)],
        bbox=aoi_bbox, size=size, config=config
    )
    images.append(request.get_data()[0])

# Export as GIF
import imageio
imageio.mimsave('sentinel_timelapse.gif', images, fps=2)
```

---

### Scrollytelling Maps

**Mapbox Storytelling template:**

```bash
git clone https://github.com/mapbox/storytelling
cd storytelling
# Edit config.js with your story chapters
npx serve .
```

```javascript
// config.js structure
const config = {
    style: 'mapbox://styles/mapbox/satellite-streets-v12',
    accessToken: 'YOUR_TOKEN',
    showMarkers: false,
    markerColor: '#3FB1CE',
    theme: 'dark',
    chapters: [
        {
            id: 'chapter-1-deforestation',
            alignment: 'left',
            hidden: false,
            title: 'Amazon Deforestation 2000-2023',
            image: './images/amazon_comparison.jpg',
            description: `The Amazon has lost 17% of its original forest cover...
                          Each red pixel represents one hectare deforested since 2000.`,
            location: {
                center: [-60.0, -3.0],
                zoom: 5,
                pitch: 0,
                bearing: 0
            },
            mapAnimation: 'flyTo',
            callback: () => {
                // Show deforestation layer
                map.setLayoutProperty('deforestation-layer', 'visibility', 'visible');
            },
            onChapterExit: [
                { layer: 'deforestation-layer', opacity: 0 }
            ]
        },
        {
            id: 'chapter-2-hotspot',
            title: 'Deforestation Hotspot: Pará State',
            location: {
                center: [-52.0, -5.0],
                zoom: 9,
                pitch: 45,
                bearing: -20
            },
            // ...
        }
    ]
};
```

**Custom scrollytelling with scrollama + MapLibre:**

```javascript
import scrollama from 'scrollama';
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [0, 0], zoom: 2,
  interactive: false  // Scroll controls, not mouse
});

const scroller = scrollama();

scroller
  .setup({
    step: '.scroll-section',
    offset: 0.5,
    progress: true
  })
  .onStepEnter(response => {
    const chapter = response.element.dataset.chapter;

    const flyConfigs = {
      'intro':       { center: [0, 20],      zoom: 2,  pitch: 0  },
      'europe':      { center: [10, 50],     zoom: 4,  pitch: 30 },
      'germany':     { center: [10.45, 51.1],zoom: 7,  pitch: 45 },
      'berlin':      { center: [13.4, 52.5], zoom: 12, pitch: 60, bearing: -20 }
    };

    if (flyConfigs[chapter]) {
      map.flyTo({ ...flyConfigs[chapter], duration: 2000, essential: true });
    }
  });
```

---

## Advanced Dark Arts

### rayshader (R): Raytrace Terrain Renders

rayshader creates stunning photorealistic terrain renders from DEMs using raytracing in R.

```r
install.packages(c("rayshader", "rayrender", "elevatr", "sf"))
library(rayshader)
library(elevatr)
library(sf)

# Get elevation data for Grand Canyon
bbox_sf <- st_as_sf(data.frame(
  x = c(-113.3, -111.8),
  y = c(35.9, 36.8)
), coords = c("x", "y"), crs = 4326)

elev_data <- get_elev_raster(bbox_sf, z = 11, src = "aws")
elev_mat <- raster_to_matrix(elev_data)

# Create hillshade map
elev_mat |>
  sphere_shade(texture = "desert") |>
  add_water(detect_water(elev_mat), color = "desert") |>
  add_shadow(ray_shade(elev_mat, zscale = 30), 0.5) |>
  add_shadow(ambient_shade(elev_mat), 0) |>
  plot_map()

# Render interactive 3D view
elev_mat |>
  sphere_shade(texture = "imhof1") |>
  add_shadow(ray_shade(elev_mat, zscale = 30), 0.5) |>
  plot_3d(
    elev_mat,
    zscale = 30,
    fov = 0,
    theta = 135,
    zoom = 0.75,
    phi = 45,
    windowsize = c(1200, 800),
    water = TRUE,
    waterdepth = 0,
    wateralpha = 0.5,
    watercolor = "#88b4d4",
    waterlinecolor = "white",
    waterlinealpha = 0.5
  )

# Render high-quality image
render_highquality(
  filename = "grand_canyon_3d.png",
  samples = 256,        # Ray samples (higher = better quality, slower)
  width = 3840,
  height = 2160,
  lightdirection = 315, # NW light direction
  lightaltitude = 35,
  lightintensity = 600,
  interactive = FALSE
)

# Animated rotation
for (angle in seq(0, 360, by = 5)) {
  render_camera(theta = angle, zoom = 0.75, phi = 45)
  render_snapshot(filename = sprintf("frame_%04d.png", angle / 5))
}
# Combine with ffmpeg
system("ffmpeg -r 20 -i frame_%04d.png -c:v libx264 -crf 18 canyon_rotation.mp4")
```

**Add satellite imagery overlay to rayshader:**

```r
library(terra)
library(rayshader)

# Load RGB satellite GeoTIFF (e.g., from Sentinel-2)
rgb_rast <- rast("sentinel2_rgb.tif")
rgb_array <- as.array(rgb_rast) / 255.0   # Normalize to 0-1

# Use as texture
elev_mat |>
  add_overlay(rgb_array, alphalayer = 1) |>
  add_shadow(ray_shade(elev_mat, zscale = 20), 0.4) |>
  plot_3d(elev_mat, zscale = 20, phi = 40, theta = -30, zoom = 0.8)
```

---

### Three.js Custom WebGL Globe

```javascript
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

// Earth sphere
const earthGeometry = new THREE.SphereGeometry(1, 64, 64);
const earthTexture = new THREE.TextureLoader().load('./earth_daymap_8k.jpg');
const earthNormal  = new THREE.TextureLoader().load('./earth_normal_8k.jpg');
const earthSpec    = new THREE.TextureLoader().load('./earth_specular.jpg');
const earthMaterial = new THREE.MeshPhongMaterial({
  map: earthTexture,
  normalMap: earthNormal,
  normalScale: new THREE.Vector2(0.5, 0.5),
  specularMap: earthSpec,
  specular: new THREE.Color(0x333333),
  shininess: 25
});
const earth = new THREE.Mesh(earthGeometry, earthMaterial);
scene.add(earth);

// Atmosphere (additive blending glow)
const atmosphereShader = {
  vertexShader: `
    varying vec3 vNormal;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    varying vec3 vNormal;
    void main() {
      float intensity = pow(0.7 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
      gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity;
    }
  `
};
const atmosphere = new THREE.Mesh(
  new THREE.SphereGeometry(1.1, 64, 64),
  new THREE.ShaderMaterial({
    ...atmosphereShader,
    blending: THREE.AdditiveBlending,
    side: THREE.BackSide,
    transparent: true
  })
);
scene.add(atmosphere);

// Convert lat/lon to 3D position on sphere
function latLonToVector3(lat, lon, radius = 1.02) {
  const phi   = (90 - lat) * (Math.PI / 180);
  const theta = (lon + 180) * (Math.PI / 180);
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
     radius * Math.cos(phi),
     radius * Math.sin(phi) * Math.sin(theta)
  );
}

// Add arcs between cities
function addArc(from, to, color = 0xffff00) {
  const start = latLonToVector3(from.lat, from.lon);
  const end   = latLonToVector3(to.lat, to.lon);
  const mid   = new THREE.Vector3()
    .addVectors(start, end)
    .normalize()
    .multiplyScalar(1.4);  // Arc height

  const curve = new THREE.QuadraticBezierCurve3(start, mid, end);
  const points = curve.getPoints(50);
  const geometry = new THREE.BufferGeometry().setFromPoints(points);
  const material = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.8 });
  scene.add(new THREE.Line(geometry, material));
}

addArc({ lat: 40.7, lon: -74.0 }, { lat: 51.5, lon: -0.1 });   // NYC - London
addArc({ lat: 35.7, lon: 139.7 }, { lat: 22.3, lon: 114.2 });  // Tokyo - HK

// Lighting
scene.add(new THREE.AmbientLight(0x333333));
const sunLight = new THREE.DirectionalLight(0xffffff, 1.5);
sunLight.position.set(5, 3, 5);
scene.add(sunLight);

camera.position.z = 2.5;
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
```

---

### MapLibre Custom WebGL Layer (Wind Particles)

```javascript
// Inject raw WebGL into MapLibre for particle effects
const windParticleLayer = {
  id: 'wind-particles',
  type: 'custom',

  onAdd(map, gl) {
    this.map = map;
    // Initialize particle positions (random across viewport)
    const numParticles = 65536;
    const positions = new Float32Array(numParticles * 2);
    for (let i = 0; i < numParticles * 2; i++) {
      positions[i] = Math.random();  // Normalized 0-1
    }

    // Load wind data texture (U,V components encoded as RGBA PNG)
    this.windTexture = gl.createTexture();
    const windImage = new Image();
    windImage.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, this.windTexture);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, windImage);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    };
    windImage.src = './wind_global.png';

    // Compile shaders
    const vertexShader = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertexShader, `
      precision mediump float;
      attribute vec2 a_pos;
      varying vec2 v_particle_pos;
      void main() {
        v_particle_pos = a_pos;
        gl_PointSize = 1.5;
        gl_Position = vec4(a_pos * 2.0 - 1.0, 0.0, 1.0);
      }
    `);
    gl.compileShader(vertexShader);

    this.particleBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.particleBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, positions, gl.DYNAMIC_DRAW);
  },

  render(gl, matrix) {
    // Update particle positions using wind field texture
    // (full implementation involves transform feedback or ping-pong framebuffers)
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    // ... draw particles
    this.map.triggerRepaint();
  }
};

map.on('load', () => {
  map.addLayer(windParticleLayer);
});
```

---

### Plotly 3D Terrain in Jupyter

```python
import plotly.graph_objects as go
import numpy as np
import rasterio
from rasterio.enums import Resampling

# Load DEM
with rasterio.open('/data/srtm_mount_rainier.tif') as src:
    # Resample to manageable size
    data = src.read(
        1,
        out_shape=(500, 500),
        resampling=Resampling.bilinear
    ).astype(float)

    # Get extent for axis labels
    bounds = src.bounds
    x = np.linspace(bounds.left, bounds.right, 500)
    y = np.linspace(bounds.bottom, bounds.top, 500)

# Replace nodata with NaN
data[data < -1000] = np.nan

# Create interactive 3D surface
fig = go.Figure(data=[
    go.Surface(
        z=data,
        x=x,
        y=y,
        colorscale='earth',
        colorbar=dict(title='Elevation (m)'),
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            specular=0.2,
            roughness=0.5,
            fresnel=0.2
        ),
        lightposition=dict(x=100, y=200, z=1000),
        contours=dict(
            z=dict(show=True, usecolormap=True,
                   highlightcolor='limegreen', project_z=True)
        )
    )
])

fig.update_layout(
    title='Mount Rainier — SRTM 30m DEM',
    scene=dict(
        xaxis_title='Longitude',
        yaxis_title='Latitude',
        zaxis_title='Elevation (m)',
        aspectratio=dict(x=1, y=1, z=0.3),  # Vertical exaggeration
        camera=dict(
            eye=dict(x=1.5, y=-1.5, z=1.0)
        )
    ),
    width=1200,
    height=700
)

fig.show()
# fig.write_html('rainier_3d.html')  # Export standalone
```

---

### CSS Blend Modes on Map Containers

```css
/* Multiply blend: makes basemap visible through a colored overlay */
.overlay-layer {
  mix-blend-mode: multiply;
  opacity: 0.7;
}

/* Screen blend: bright overlay colors pop against dark basemap */
.heatmap-overlay {
  mix-blend-mode: screen;
  opacity: 0.8;
}

/* Overlay blend: contrast-boosting effect */
.satellite-overlay {
  mix-blend-mode: overlay;
  opacity: 0.6;
}
```

```html
<!-- Stack a choropleth over a satellite basemap with multiply blend -->
<div style="position:relative; width:100%; height:100vh;">
  <!-- Base satellite map -->
  <div id="base-map" style="position:absolute; inset:0; z-index:1;"></div>

  <!-- Data canvas overlay with blend mode -->
  <canvas id="data-canvas"
    style="position:absolute; inset:0; z-index:2; mix-blend-mode:multiply; opacity:0.75;">
  </canvas>
</div>

<script>
// Render choropleth on canvas with transparency
const canvas = document.getElementById('data-canvas');
const ctx = canvas.getContext('2d');

// Draw colored polygons matching the map's geographic extent
// The multiply blend mode darkens the satellite imagery proportionally
// to the data value — creates an integrated look without covering base detail
</script>
```

---

### Cesium + CZML Satellite Orbit from TLE

```bash
npm install satellite.js
```

```javascript
import * as satellite from 'satellite.js';
import * as Cesium from 'cesium';

// ISS TLE (update regularly from celestrak.org)
const tleLine1 = '1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9993';
const tleLine2 = '2 25544  51.6412  47.6091 0006969 232.4348 127.5983 15.50095670432416';

const satrec = satellite.twoline2satrec(tleLine1, tleLine2);

const startTime = Cesium.JulianDate.now();
const stopTime  = Cesium.JulianDate.addHours(startTime, 3, new Cesium.JulianDate());

// Generate position samples every 30 seconds
const positionProperty = new Cesium.SampledPositionProperty();
const numSamples = 360; // 3 hours × 60 min / 30 sec

for (let i = 0; i < numSamples; i++) {
  const time = Cesium.JulianDate.addSeconds(startTime, i * 30, new Cesium.JulianDate());
  const jsDate = Cesium.JulianDate.toDate(time);

  const positionAndVelocity = satellite.propagate(satrec, jsDate);
  const gmst = satellite.gstime(jsDate);
  const geo  = satellite.eciToGeodetic(positionAndVelocity.position, gmst);

  const cartesian = Cesium.Cartesian3.fromRadians(
    geo.longitude,
    geo.latitude,
    geo.height * 1000  // Convert km to meters
  );

  positionProperty.addSample(time, cartesian);
}

// Add ISS entity to Cesium
viewer.entities.add({
  id: 'ISS',
  name: 'International Space Station',
  availability: new Cesium.TimeIntervalCollection([
    new Cesium.TimeInterval({ start: startTime, stop: stopTime })
  ]),
  position: positionProperty,
  orientation: new Cesium.VelocityOrientationProperty(positionProperty),
  model: {
    uri: './iss.glb',
    minimumPixelSize: 128,
    maximumScale: 20000
  },
  path: {
    resolution: 120,
    material: new Cesium.PolylineGlowMaterialProperty({
      glowPower: 0.1,
      color: Cesium.Color.YELLOW
    }),
    width: 2,
    leadTime: 0,
    trailTime: 5400  // 90-minute orbit trail
  },
  label: {
    text: 'ISS',
    font: '12pt sans-serif',
    fillColor: Cesium.Color.WHITE,
    outlineColor: Cesium.Color.BLACK,
    outlineWidth: 2,
    style: Cesium.LabelStyle.FILL_AND_OUTLINE,
    pixelOffset: new Cesium.Cartesian2(0, -20)
  }
});

viewer.clock.startTime = startTime.clone();
viewer.clock.stopTime  = stopTime.clone();
viewer.clock.currentTime = startTime.clone();
viewer.clock.clockRange = Cesium.ClockRange.LOOP_STOP;
viewer.clock.multiplier = 30;  // 30x real-time
viewer.timeline.zoomTo(startTime, stopTime);
```

---

## Useful Commands & One-Liners

```bash
# Convert Shapefile to 3D Tiles via ogr2ogr + cesium-ion-cli
ogr2ogr -f GeoJSON buildings.geojson buildings.shp
ion asset upload --name "Buildings" --type "3DTILES" buildings.geojson

# Convert LAS to COPC (Cloud Optimized Point Cloud) for streaming
pdal translate input.las output.copc.laz --writer copc

# Reproject and export terrain DEM for Cesium quantized mesh
gdal_translate -of GTiff input.tif /vsistdout/ | \
  ctb-tile -o ./terrain/ -f Mesh -C -N

# Generate 3D Tiles from CityJSON
pip install cjio py3dtiles
cjio city.json export --format obj | py3dtiles convert /dev/stdin --out tiles/

# Extract OSM buildings for a city as GeoJSON
curl "https://overpass-api.de/api/interpreter?data=[out:json];area[name='Munich']->.a;(way[building](area.a);relation[building](area.a););out+geom;" \
  | python -c "import sys,json; d=json.load(sys.stdin); print(json.dumps({'type':'FeatureCollection','features':[{'type':'Feature','properties':f['tags'],'geometry':{'type':'Polygon','coordinates':[[[n['lon'],n['lat']] for n in f['geometry']]]}} for f in d['elements'] if 'geometry' in f]}))" \
  > munich_buildings.geojson

# Quick rayshader render from R command line
Rscript -e "
library(rayshader); library(elevatr); library(sf)
bbox <- st_as_sf(data.frame(x=c(11.3,11.9), y=c(47.9,48.3)), coords=c('x','y'), crs=4326)
elev <- raster_to_matrix(get_elev_raster(bbox, z=10, src='aws'))
elev |> sphere_shade('desert') |> add_shadow(ray_shade(elev, zscale=30), 0.5) |>
  save_png('terrain.png', width=3840, height=2160)
"
```

---

## Tool Comparison Matrix

| Tool | 3D Buildings | Terrain | Point Cloud | Animation | Print Quality | Ease |
|------|-------------|---------|------------|-----------|--------------|------|
| CesiumJS | Excellent | Excellent | Good | Excellent | Poor | Medium |
| MapLibre v4 | Good | Good | No | Good | Poor | Easy |
| deck.gl | Good | Good | Excellent | Good | Poor | Medium |
| QGIS 3D | Good | Good | Excellent | Medium | Excellent | Easy |
| Blender GIS | Excellent | Excellent | Good | Excellent | Excellent | Hard |
| kepler.gl | Good | No | No | Excellent | Poor | Very Easy |
| ArcGIS Pro | Excellent | Excellent | Excellent | Good | Excellent | Medium |
| rayshader (R) | No | Excellent | No | Good | Excellent | Medium |
| iTowns | Good | Good | Excellent | Good | Poor | Hard |

---

## Resources & Data Sources

- **3D Tiles Samples** (Cesium): `github.com/CesiumGS/3d-tiles-samples`
- **3D BAG** (Netherlands LOD2-3 buildings): `3dbag.nl`
- **PDOK 3D** (Dutch cadastral 3D): `pdok.nl/3d-basisvoorziening`
- **OpenTopography** (global LiDAR): `opentopography.org`
- **USGS 3DEP** (US LiDAR): `apps.nationalmap.gov/3dep-lidar`
- **Copernicus DEM** (global 30m GLO-30): `spacedata.copernicus.eu`
- **TanDEM-X** (global 12m DEM): `eoc.dlr.de`
- **OSM Buildings** (global OSM 3D): `osmbuildings.org`
- **Sketchfab** (3D model repository): `sketchfab.com`
- **CesiumJS Documentation**: `cesium.com/docs`
- **3D Tiles Spec**: `github.com/CesiumGS/3d-tiles`
- **CityJSON Spec**: `cityjson.org`
- **ColorBrewer**: `colorbrewer2.org`
- **Scientific Colour Maps (Crameri)**: `fabiocrameri.ch/colourmaps`
- **cmocean**: `matplotlib.org/cmocean`
- **rayshader docs**: `rayshader.com`
