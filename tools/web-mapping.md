# Web Mapping & Visualization Libraries

> **Quick Picks**
> - **SOTA (Open Source)**: MapLibre GL JS + Deck.gl
> - **Free Best All-Rounder**: Leaflet
> - **Fastest Setup (CDN, zero build)**: Leaflet CDN
> - **Massive Data / GPU**: Deck.gl
> - **3D Globe**: CesiumJS
> - **Client-Side Analysis**: Turf.js

---

## MapLibre GL JS (the SOTA open-source choice)

### History & Why It Matters

Mapbox GL JS went commercial (BSL license) in v2. MapLibre GL JS is the community fork of the last open-source v1.x release. It is now maintained by the Linux Foundation with active contributors from AWS, Microsoft, Meta, and dozens of GIS companies. It has since surpassed the original in features: globe view (v4+), custom shaders, full WebGL2 support, terrain, and a thriving plugin ecosystem — all under the BSD-2-Clause license.

**Core capabilities:**
- WebGL-accelerated vector and raster tile rendering
- Runtime style modification (change colors, filters, layouts without reloading)
- 3D terrain via `exaggeration` on DEM sources
- Globe projection (v4+)
- Expression-based data-driven styling
- Custom WebGL layers (`CustomLayerInterface`)
- PMTiles and any custom protocol via `addProtocol()`

### Quick Start

```html
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.css" />
  <script src="https://unpkg.com/maplibre-gl@4/dist/maplibre-gl.js"></script>
  <style>
    body { margin: 0; }
    #map { width: 100vw; height: 100vh; }
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    const map = new maplibregl.Map({
      container: 'map',
      // Free basemap from Protomaps (no API key needed)
      style: 'https://api.protomaps.com/styles/v2/light.json?key=YOUR_FREE_KEY',
      center: [116.4, 39.9],
      zoom: 10
    });

    // Or use OpenFreeMap (zero signup)
    // style: 'https://tiles.openfreemap.org/styles/liberty'
  </script>
</body>
</html>
```

### PMTiles Integration: Zero-Server Vector Tiles

PMTiles is a single-file archive format for tiles. Host it on S3, R2, or GitHub Pages and serve vector (or raster) tiles with zero tile server infrastructure. The `pmtiles` protocol handler plugs directly into MapLibre.

```html
<script src="https://unpkg.com/pmtiles@3/dist/pmtiles.js"></script>
<script>
  // Register the pmtiles:// protocol BEFORE creating the map
  const protocol = new pmtiles.Protocol();
  maplibregl.addProtocol('pmtiles', protocol.tile.bind(protocol));

  const map = new maplibregl.Map({
    container: 'map',
    style: {
      version: 8,
      glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
      sources: {
        // Point directly at a .pmtiles file — no tile server!
        'my-data': {
          type: 'vector',
          url: 'pmtiles://https://r2.yourdomain.com/data/buildings.pmtiles'
        },
        // You can also use a PMTiles file from GitHub Releases:
        // url: 'pmtiles://https://github.com/user/repo/releases/download/v1/data.pmtiles'
      },
      layers: [
        {
          id: 'buildings-fill',
          type: 'fill',
          source: 'my-data',
          'source-layer': 'buildings',  // layer name inside the PMTiles file
          paint: {
            'fill-color': '#e8c87a',
            'fill-opacity': 0.8
          }
        }
      ]
    },
    center: [0, 0],
    zoom: 2
  });
</script>
```

To create a PMTiles file from your own data:
```bash
# Convert GeoJSON/Shapefile to PMTiles using tippecanoe
tippecanoe -o output.pmtiles -z14 --drop-densest-as-needed input.geojson

# Or use the go-pmtiles CLI to merge/inspect
pmtiles show output.pmtiles
pmtiles extract output.pmtiles subset.pmtiles --bbox="-74.1,40.6,-73.8,40.9"
```

### 3D Buildings with fill-extrusion

```javascript
map.on('load', () => {
  map.addSource('openmaptiles', {
    type: 'vector',
    url: 'https://api.maptiler.com/tiles/v3/tiles.json?key=YOUR_KEY'
    // Or use a self-hosted tile source
  });

  map.addLayer({
    id: '3d-buildings',
    source: 'openmaptiles',
    'source-layer': 'building',
    type: 'fill-extrusion',
    minzoom: 14,
    paint: {
      // Use building height attribute for realistic extrusion
      'fill-extrusion-height': [
        'interpolate', ['linear'], ['zoom'],
        14, 0,  // no height below zoom 14
        16, ['coalesce', ['get', 'render_height'], 10]
      ],
      'fill-extrusion-base': ['coalesce', ['get', 'render_min_height'], 0],
      'fill-extrusion-color': [
        'interpolate', ['linear'],
        ['coalesce', ['get', 'render_height'], 0],
        0,   '#aaaaaa',
        50,  '#cccccc',
        200, '#ffffff'
      ],
      'fill-extrusion-opacity': 0.85
    }
  });
});
```

### 3D Terrain

```javascript
map.on('load', () => {
  // Add DEM source (MapTiler, Maptiler Cloud, or your own COG terrain)
  map.addSource('terrain-dem', {
    type: 'raster-dem',
    url: 'https://api.maptiler.com/tiles/terrain-rgb-v2/tiles.json?key=YOUR_KEY',
    tileSize: 256
  });

  // Enable 3D terrain
  map.setTerrain({ source: 'terrain-dem', exaggeration: 1.5 });

  // Add sky layer for atmosphere effect
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

### Performance Tricks

**Clustering 100K+ points with source-level clustering:**

```javascript
map.addSource('earthquakes', {
  type: 'geojson',
  data: 'https://example.com/earthquakes.geojson',
  cluster: true,
  clusterMaxZoom: 14,
  clusterRadius: 50,
  // Custom cluster properties for aggregation
  clusterProperties: {
    'max_mag': ['max', ['get', 'magnitude']],
    'sum_count': ['+', 1]
  }
});

map.addLayer({
  id: 'clusters',
  type: 'circle',
  source: 'earthquakes',
  filter: ['has', 'point_count'],
  paint: {
    'circle-color': [
      'step', ['get', 'point_count'],
      '#51bbd6', 100,
      '#f1f075', 750,
      '#f28cb1'
    ],
    'circle-radius': ['step', ['get', 'point_count'], 20, 100, 30, 750, 40]
  }
});

map.addLayer({
  id: 'unclustered-point',
  type: 'circle',
  source: 'earthquakes',
  filter: ['!', ['has', 'point_count']],
  paint: {
    'circle-color': '#11b4da',
    'circle-radius': 5
  }
});
```

**Click interaction with queryRenderedFeatures:**

```javascript
map.on('click', 'unclustered-point', (e) => {
  const features = map.queryRenderedFeatures(e.point, {
    layers: ['unclustered-point']
  });
  if (!features.length) return;

  const { magnitude, place } = features[0].properties;
  new maplibregl.Popup()
    .setLngLat(e.lngLat)
    .setHTML(`<strong>M${magnitude}</strong><br>${place}`)
    .addTo(map);
});

// Change cursor to pointer on hover
map.on('mouseenter', 'unclustered-point', () => {
  map.getCanvas().style.cursor = 'pointer';
});
map.on('mouseleave', 'unclustered-point', () => {
  map.getCanvas().style.cursor = '';
});
```

**Dynamic filter expressions (no data reload):**

```javascript
// Filter by magnitude range without re-fetching data
function filterByMagnitude(min, max) {
  map.setFilter('unclustered-point', [
    'all',
    ['>=', ['get', 'magnitude'], min],
    ['<=', ['get', 'magnitude'], max]
  ]);
}

// Use with a range slider
document.getElementById('mag-slider').addEventListener('input', (e) => {
  filterByMagnitude(parseFloat(e.target.value), 9.0);
});
```

### TRICK: transformRequest for private tile auth

```javascript
const map = new maplibregl.Map({
  container: 'map',
  style: '...',
  transformRequest: (url, resourceType) => {
    // Add auth headers to your private tile server only
    if (url.startsWith('https://tiles.yourcompany.com')) {
      return {
        url: url,
        headers: {
          'Authorization': `Bearer ${getAuthToken()}`,
          'X-Tenant-ID': getCurrentTenant()
        }
      };
    }
    // Optional: rewrite URLs (e.g. proxy through your API gateway)
    if (url.includes('api.maptiler.com')) {
      return { url: url.replace('PLACEHOLDER', YOUR_REAL_KEY) };
    }
  }
});
```

### TRICK: addProtocol() for COG (Cloud Optimized GeoTIFF)

```javascript
import * as GeoTIFF from 'geotiff';

// Register a custom cog:// protocol to load raster tiles from a COG
maplibregl.addProtocol('cog', async (params, abortController) => {
  const url = params.url.replace('cog://', 'https://');
  // Parse tile coords from params
  const { x, y, z } = params;

  // Use geotiff.js to read a specific window from the COG
  const tiff = await GeoTIFF.fromUrl(url);
  const image = await tiff.getImage(z); // overview level
  const data = await image.readRasters({ window: [x*256, y*256, (x+1)*256, (y+1)*256] });

  // Return as ArrayBuffer
  return { data: data[0].buffer };
});

map.addSource('cog-source', {
  type: 'raster',
  tiles: ['cog://https://storage.googleapis.com/your-bucket/dem.tif/{z}/{x}/{y}'],
  tileSize: 256
});
```

### Plugins Ecosystem

| Plugin | Purpose |
|--------|---------|
| `@maplibre/maplibre-gl-geocoder` | Search/geocoding control |
| `maplibre-gl-directions` | Routing with turn-by-turn |
| `@mapbox/mapbox-gl-draw` (MapLibre fork) | Draw/edit features on the map |
| `maplibre-gl-compare` | Side-by-side swipe comparison |
| `maplibre-gl-inspect` | Debug layer/feature inspector |
| `terra-draw` | Modern drawing library, framework-agnostic |
| `maplibre-contour` | Real-time contour lines from terrain RGB |
| `maplibre-gl-opacity` | Layer opacity slider control |

### Free Basemap Sources (No API Key)

```javascript
// OpenFreeMap (100% free, self-hostable)
style: 'https://tiles.openfreemap.org/styles/liberty'

// Protomaps (free key, generous limits)
style: 'https://api.protomaps.com/styles/v2/light.json?key=FREE_KEY'

// Stadia Maps (free tier, no CC required)
style: 'https://tiles.stadiamaps.com/styles/alidade_smooth.json'

// Versatiles
style: 'https://tiles.versatiles.org/assets/styles/colorful.json'
```

---

## Leaflet

### Why Leaflet Still Dominates in 2025

- **42KB** gzipped — loads in milliseconds on 3G
- **Zero build step** — paste a `<script>` tag and you're mapping
- **700+ plugins** — almost any capability you need has a plugin
- **Stable API** — code written in 2015 still works today
- **Accessibility-friendly** — keyboard navigation, ARIA labels out of the box

Choose Leaflet when: prototyping fast, building low-complexity maps, targeting low-bandwidth users, or when you just need markers + popups without WebGL.

### Quick Start (CDN, zero build)

```html
<!DOCTYPE html>
<html>
<head>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9/dist/leaflet.js"></script>
  <style>
    body { margin: 0; }
    #map { width: 100vw; height: 100vh; }
  </style>
</head>
<body>
  <div id="map"></div>
  <script>
    const map = L.map('map').setView([39.9, 116.4], 10);

    // OpenStreetMap tiles (completely free)
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
      maxZoom: 19
    }).addTo(map);

    // GeoJSON layer with popup
    fetch('https://example.com/data.geojson')
      .then(r => r.json())
      .then(data => {
        L.geoJSON(data, {
          onEachFeature: (feature, layer) => {
            layer.bindPopup(`<b>${feature.properties.name}</b>`);
          },
          style: (feature) => ({
            color: feature.properties.type === 'park' ? 'green' : 'blue',
            weight: 2,
            fillOpacity: 0.4
          })
        }).addTo(map);
      });
  </script>
</body>
</html>
```

### leaflet-providers: 100+ Basemaps in One Plugin

```javascript
// npm install leaflet-providers
// or <script src="https://unpkg.com/leaflet-providers/leaflet-providers.js"></script>

// Stadia Maps (modern, clean)
L.tileLayer.provider('Stadia.AlidadeSmooth').addTo(map);

// CartoDB Dark
L.tileLayer.provider('CartoDB.DarkMatter').addTo(map);

// Esri World Imagery (satellite)
L.tileLayer.provider('Esri.WorldImagery').addTo(map);

// OpenTopoMap
L.tileLayer.provider('OpenTopoMap').addTo(map);

// Layer control with multiple basemaps
const basemaps = {
  'Streets': L.tileLayer.provider('Stadia.AlidadeSmooth'),
  'Satellite': L.tileLayer.provider('Esri.WorldImagery'),
  'Topo': L.tileLayer.provider('OpenTopoMap')
};
L.control.layers(basemaps).addTo(map);
```

### Marker Clustering with Leaflet.markercluster

```javascript
// <script src="https://unpkg.com/leaflet.markercluster/dist/leaflet.markercluster.js"></script>

const markers = L.markerClusterGroup({
  maxClusterRadius: 80,       // pixels — how close before grouping
  disableClusteringAtZoom: 16, // stop clustering at zoom 16
  spiderfyOnMaxZoom: true,
  showCoverageOnHover: false,
  iconCreateFunction: (cluster) => {
    const count = cluster.getChildCount();
    const size = count < 10 ? 'small' : count < 100 ? 'medium' : 'large';
    return L.divIcon({
      html: `<div><span>${count}</span></div>`,
      className: `marker-cluster marker-cluster-${size}`,
      iconSize: L.point(40, 40)
    });
  }
});

// Add 50,000 markers efficiently
data.features.forEach(f => {
  const [lng, lat] = f.geometry.coordinates;
  markers.addLayer(L.marker([lat, lng]).bindPopup(f.properties.name));
});

map.addLayer(markers);
```

### FlatGeobuf in Leaflet: Load Only Visible Features via HTTP Range Requests

FlatGeobuf is a binary format with a spatial index. You can fetch only features within the current viewport with a single HTTP range request — no server-side query needed.

```html
<script src="https://unpkg.com/flatgeobuf/dist/flatgeobuf-geojson.min.js"></script>
<script>
  let currentLayer = null;

  async function loadFGB() {
    // Remove previous layer
    if (currentLayer) map.removeLayer(currentLayer);

    // Get current viewport bounding box
    const bounds = map.getBounds();
    const bbox = {
      minX: bounds.getWest(),
      minY: bounds.getSouth(),
      maxX: bounds.getEast(),
      maxY: bounds.getNorth()
    };

    // deserialize() makes an HTTP range request using the spatial index
    // Only features within bbox are downloaded — even for a 1GB file!
    const iter = flatgeobuf.deserialize(
      'https://example.com/large-dataset.fgb',
      bbox
    );

    const features = [];
    for await (const feature of iter) {
      features.push(feature);
    }

    currentLayer = L.geoJSON({ type: 'FeatureCollection', features }, {
      style: { color: '#e74c3c', weight: 1, fillOpacity: 0.3 }
    }).addTo(map);
  }

  // Load on map move (with debounce)
  let timer;
  map.on('moveend', () => {
    clearTimeout(timer);
    timer = setTimeout(loadFGB, 300);
  });

  // Initial load
  map.on('load', loadFGB);
  loadFGB();
</script>
```

### TRICK: L.canvas() Renderer for 50K+ Markers

The default SVG renderer creates a DOM element per marker — it collapses at ~10K features. The Canvas renderer draws everything to a `<canvas>` element.

```javascript
// Create a canvas renderer — do this ONCE
const renderer = L.canvas({ padding: 0.5 });

// Pass it to circle markers instead of regular markers
// (Canvas only works with vector layers, not icon markers)
data.features.forEach(f => {
  const [lng, lat] = f.geometry.coordinates;
  L.circleMarker([lat, lng], {
    renderer: renderer,  // <-- the key
    radius: 4,
    color: '#e74c3c',
    weight: 1,
    fillOpacity: 0.7
  }).addTo(map);
});

// For 50K+ points this is roughly 5-10x faster than SVG
// For 200K+ points, use Canvas + requestAnimationFrame batching:
function addInBatches(features, batchSize = 1000) {
  let i = 0;
  function addBatch() {
    const end = Math.min(i + batchSize, features.length);
    for (; i < end; i++) {
      const [lng, lat] = features[i].geometry.coordinates;
      L.circleMarker([lat, lng], { renderer }).addTo(map);
    }
    if (i < features.length) requestAnimationFrame(addBatch);
  }
  requestAnimationFrame(addBatch);
}
```

### Leaflet.VectorGrid: MVT Vector Tiles in Leaflet

```javascript
// npm install leaflet.vectorgrid
// or CDN: unpkg.com/leaflet.vectorgrid/dist/Leaflet.VectorGrid.bundled.js

const vectorGrid = L.vectorGrid.protobuf(
  'https://your-tile-server/{z}/{x}/{y}.pbf',
  {
    vectorTileLayerStyles: {
      // Style each source layer by name
      'roads': {
        weight: 1,
        fillColor: '#555',
        fill: true,
        fillOpacity: 0.5
      },
      'buildings': (properties, zoom) => ({
        // Data-driven style function
        fillColor: properties.building_type === 'commercial' ? '#e8c87a' : '#cccccc',
        fillOpacity: 0.7,
        stroke: false,
        fill: true
      })
    },
    interactive: true,
    getFeatureId: (f) => f.properties.osm_id
  }
);

vectorGrid.on('click', (e) => {
  L.popup()
    .setContent(`OSM ID: ${e.layer.properties.osm_id}`)
    .setLatLng(e.latlng)
    .openOn(map);
});

vectorGrid.addTo(map);
```

### TRICK: Subdomain Rotation for Parallel Tile Loading

```javascript
// Without subdomains: browser limits to 6 parallel requests per hostname
// With subdomains: 3 hostnames × 6 = 18 parallel tile requests

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  subdomains: ['a', 'b', 'c'],  // rotated automatically by Leaflet
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

// For your own tile server on multiple hosts:
L.tileLayer('https://tiles{s}.yourdomain.com/{z}/{x}/{y}.png', {
  subdomains: ['1', '2', '3', '4'],
  maxNativeZoom: 16,
  maxZoom: 21  // overzoom beyond native zoom
}).addTo(map);
```

### Key Plugins Reference

| Plugin | Install / CDN | Use Case |
|--------|--------------|----------|
| `Leaflet.markercluster` | `npm i leaflet.markercluster` | Cluster thousands of markers |
| `Leaflet.draw` | `npm i leaflet-draw` | Draw shapes on the map |
| `Leaflet.heat` | CDN unpkg | Heatmap layer from point data |
| `Leaflet-omnivore` | CDN unpkg | Load KML, GPX, WKT, CSV, TopoJSON |
| `Leaflet.VectorGrid` | `npm i leaflet.vectorgrid` | MVT vector tile rendering |
| `Leaflet-Geoman` | `npm i @geoman-io/leaflet-geoman-free` | Advanced drawing + editing + snapping |
| `leaflet-providers` | `npm i leaflet-providers` | 100+ basemap providers |
| `Leaflet.Control.Geocoder` | `npm i leaflet-control-geocoder` | Nominatim/Photon search |
| `Leaflet.Routing.Machine` | `npm i leaflet-routing-machine` | Turn-by-turn routing UI |

---

## OpenLayers

### When to Choose OpenLayers

OpenLayers is the "enterprise GIS" choice in the browser. Choose it when:
- You need **WMS, WFS, WMTS, WCS** (OGC standards) out of the box
- You need **on-the-fly reprojection** (display EPSG:4326 data on a EPSG:3857 map, or Chinese CGCS2000 / GCJ02 coordinates)
- You need **server-side feature editing** with transactional WFS (WFS-T)
- You are integrating with **GeoServer, MapServer, or QGIS Server**
- You need **snapping + validation** for desktop-GIS-quality digitizing

### Quick Start

```javascript
import Map from 'ol/Map';
import View from 'ol/View';
import TileLayer from 'ol/layer/Tile';
import OSM from 'ol/source/OSM';
import VectorLayer from 'ol/layer/Vector';
import VectorSource from 'ol/source/Vector';
import GeoJSON from 'ol/format/GeoJSON';
import { fromLonLat } from 'ol/proj';

const map = new Map({
  target: 'map',
  layers: [
    new TileLayer({ source: new OSM() }),
    new VectorLayer({
      source: new VectorSource({
        url: '/api/features.geojson',
        format: new GeoJSON()
      })
    })
  ],
  view: new View({
    center: fromLonLat([116.4, 39.9]),
    zoom: 10
  })
});
```

### WMS / WFS Integration with GeoServer

```javascript
import TileWMS from 'ol/source/TileWMS';
import WFS from 'ol/format/WFS';
import { bbox as bboxStrategy } from 'ol/loadingstrategy';

// WMS raster layer
const wmsLayer = new TileLayer({
  source: new TileWMS({
    url: 'https://geoserver.yourorg.com/geoserver/workspace/wms',
    params: {
      LAYERS: 'workspace:layer_name',
      TILED: true,
      CQL_FILTER: "category = 'park'",  // server-side filter
    },
    serverType: 'geoserver',
    crossOrigin: 'anonymous'
  })
});

// WFS vector layer with bbox loading strategy (only loads visible area)
const wfsSource = new VectorSource({
  format: new WFS(),
  url: (extent) => {
    return 'https://geoserver.yourorg.com/geoserver/wfs?' +
      'service=WFS&version=2.0.0&request=GetFeature' +
      '&typeName=workspace:features' +
      '&outputFormat=application/json' +
      `&bbox=${extent.join(',')},EPSG:3857`;
  },
  strategy: bboxStrategy  // only fetch current viewport
});

// GetFeatureInfo on WMS click (like desktop GIS)
map.on('singleclick', async (evt) => {
  const viewResolution = map.getView().getResolution();
  const url = wmsLayer.getSource().getFeatureInfoUrl(
    evt.coordinate,
    viewResolution,
    'EPSG:3857',
    { INFO_FORMAT: 'application/json' }
  );
  if (url) {
    const response = await fetch(url);
    const data = await response.json();
    console.log('Feature info:', data.features[0]?.properties);
  }
});
```

### On-the-Fly Reprojection (unique to OpenLayers)

```javascript
import proj4 from 'proj4';
import { register } from 'ol/proj/proj4';

// Register Chinese CGCS2000 projection
proj4.defs('EPSG:4490', '+proj=longlat +ellps=GRS80 +no_defs');
register(proj4);

// Now you can display CGCS2000 data on a Web Mercator map seamlessly
const cgcsSource = new VectorSource({
  url: '/data/china-admin.geojson',
  format: new GeoJSON({
    dataProjection: 'EPSG:4490',      // data is in CGCS2000
    featureProjection: 'EPSG:3857'   // map is in Web Mercator
  })
});
```

---

## Deck.gl

### GPU-Accelerated Visualization for Millions of Points

Deck.gl uses WebGL2 to render data directly on the GPU. Where Leaflet starts to struggle at 10K markers, Deck.gl handles 10 million+ points at 60fps.

### Quick Start (MapLibre + Deck.gl overlay)

```javascript
import maplibregl from 'maplibre-gl';
import { Deck } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://tiles.openfreemap.org/styles/liberty',
  center: [-74.0, 40.7],
  zoom: 11
});

// Deck.gl overlays on top of MapLibre using the same canvas
const deck = new Deck({
  canvas: 'deck-canvas',  // a separate canvas layered over the map
  width: '100%',
  height: '100%',
  initialViewState: { longitude: -74.0, latitude: 40.7, zoom: 11 },
  controller: false,  // MapLibre handles pan/zoom
  onViewStateChange: ({ viewState }) => {
    map.jumpTo({
      center: [viewState.longitude, viewState.latitude],
      zoom: viewState.zoom,
      bearing: viewState.bearing,
      pitch: viewState.pitch
    });
  },
  layers: [
    new ScatterplotLayer({
      id: 'scatterplot',
      data: 'https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/scatterplot/manhattan.json',
      getPosition: d => [d[0], d[1]],
      getColor: d => d[2] === 1 ? [255, 0, 128] : [0, 128, 255],
      getRadius: 50,
      pickable: true,
      onClick: (info) => console.log('Clicked:', info.object)
    })
  ]
});

// Sync deck.gl when map moves
map.on('move', () => {
  const { lng, lat } = map.getCenter();
  deck.setProps({
    viewState: {
      longitude: lng,
      latitude: lat,
      zoom: map.getZoom(),
      bearing: map.getBearing(),
      pitch: map.getPitch()
    }
  });
});
```

### Layer Catalog with Examples

**HexagonLayer — aggregate points into hexbin heatmap:**

```javascript
import { HexagonLayer } from '@deck.gl/aggregation-layers';

new HexagonLayer({
  id: 'hexagon',
  data: points,
  getPosition: d => d.coordinates,
  radius: 500,           // hexagon radius in meters
  elevationScale: 4,     // 3D extrusion scale
  extruded: true,
  pickable: true,
  colorRange: [
    [1, 152, 189],
    [73, 227, 206],
    [216, 254, 181],
    [254, 237, 177],
    [254, 173, 84],
    [209, 55, 78]
  ]
});
```

**H3HexagonLayer — TRICK: Stunning 3D hex visualizations:**

```javascript
import { H3HexagonLayer } from '@deck.gl/geo-layers';
import * as h3 from 'h3-js';

// Pre-aggregate your data into H3 cells
const h3Data = {};
rawPoints.forEach(point => {
  const cellId = h3.latLngToCell(point.lat, point.lng, 7); // resolution 7
  h3Data[cellId] = (h3Data[cellId] || 0) + 1;
});

const h3Layer = new H3HexagonLayer({
  id: 'h3-hexagons',
  data: Object.entries(h3Data).map(([hex, count]) => ({ hex, count })),
  getHexagon: d => d.hex,
  getFillColor: d => [255, (1 - d.count / maxCount) * 255, 0],
  getElevation: d => d.count * 100,
  elevationScale: 1,
  extruded: true,
  pickable: true
});
```

**TripsLayer — animated vehicle traces:**

```javascript
import { TripsLayer } from '@deck.gl/geo-layers';

let currentTime = 0;
const animationDuration = 1800; // seconds of data

function animate() {
  currentTime = (currentTime + 1) % animationDuration;

  deck.setProps({
    layers: [
      new TripsLayer({
        id: 'trips',
        data: tripsData,
        getPath: d => d.waypoints.map(p => p.coordinates),
        getTimestamps: d => d.waypoints.map(p => p.timestamp),
        getColor: d => d.vendor === 0 ? [253, 128, 93] : [23, 184, 190],
        opacity: 0.8,
        widthMinPixels: 2,
        trailLength: 180,
        currentTime: currentTime
      })
    ]
  });

  requestAnimationFrame(animate);
}
animate();
```

**MVTLayer — vector tiles with GPU rendering:**

```javascript
import { MVTLayer } from '@deck.gl/geo-layers';

new MVTLayer({
  id: 'mvt-layer',
  data: 'https://tiles.example.com/{z}/{x}/{y}.pbf',
  minZoom: 0,
  maxZoom: 14,
  getLineColor: [0, 0, 0],
  getFillColor: f => {
    switch (f.properties.class) {
      case 'park': return [100, 200, 100];
      case 'water': return [100, 150, 250];
      default: return [200, 200, 200];
    }
  },
  pickable: true,
  autoHighlight: true,
  highlightColor: [255, 200, 0, 100]
});
```

### pydeck: Deck.gl in Python/Jupyter

```python
import pydeck as pdk
import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/scatterplot/manhattan.json',
                 header=None, names=['lng', 'lat', 'type'])

layer = pdk.Layer(
    'ScatterplotLayer',
    data=df,
    get_position='[lng, lat]',
    get_color='[type * 255, 0, 128 - type * 128, 160]',
    get_radius=50,
    pickable=True
)

view_state = pdk.ViewState(latitude=40.7, longitude=-74.0, zoom=11)

r = pdk.Deck(layers=[layer], initial_view_state=view_state,
             map_style='https://tiles.openfreemap.org/styles/liberty')
r.to_html('output.html')
# In Jupyter: r.show()
```

---

## CesiumJS & 3D Globes

### Quick Start

```html
<link href="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Widgets/widgets.css" rel="stylesheet">
<script src="https://cesium.com/downloads/cesiumjs/releases/1.117/Build/Cesium/Cesium.js"></script>

<div id="cesiumContainer" style="width:100%;height:100vh;"></div>
<script>
  // Free Ion token for basic use (sign up at cesium.com)
  Cesium.Ion.defaultAccessToken = 'YOUR_TOKEN';

  const viewer = new Cesium.Viewer('cesiumContainer', {
    terrainProvider: await Cesium.CesiumTerrainProvider.fromIonAssetId(1),
    timeline: false,
    animation: false
  });

  // Add Cesium OSM Buildings (3D Tiles)
  const osmBuildings = await Cesium.createOsmBuildingsAsync();
  viewer.scene.primitives.add(osmBuildings);

  // Fly to a location
  viewer.camera.flyTo({
    destination: Cesium.Cartesian3.fromDegrees(116.4, 39.9, 2000),
    orientation: { heading: 0, pitch: Cesium.Math.toRadians(-45), roll: 0 }
  });

  // Load local GeoJSON
  const dataSource = await Cesium.GeoJsonDataSource.load('/data/features.geojson', {
    stroke: Cesium.Color.RED,
    fill: Cesium.Color.RED.withAlpha(0.4),
    strokeWidth: 2
  });
  viewer.dataSources.add(dataSource);
  viewer.zoomTo(dataSource);
</script>
```

### 3D Tiles from Custom Sources

```javascript
// Load Photogrammetry / Point Cloud / BIM as 3D Tiles
const tileset = await Cesium.Cesium3DTileset.fromUrl(
  'https://your-server.com/3dtiles/tileset.json'
);
viewer.scene.primitives.add(tileset);
viewer.zoomTo(tileset);

// Style 3D Tiles by attribute
tileset.style = new Cesium.Cesium3DTileStyle({
  color: {
    conditions: [
      ["${height} >= 100", "color('blue')"],
      ["${height} >= 50",  "color('green')"],
      ["true",             "color('white')"]
    ]
  },
  show: "${height} > 10"  // hide very short features
});
```

### Time-Dynamic Data (CZML)

```javascript
// CZML: JSON format for time-dynamic entities (vehicles, satellites, sensor sweeps)
const czml = [
  { id: 'document', name: 'Demo', version: '1.0',
    clock: { interval: '2024-01-01T00:00:00Z/2024-01-01T01:00:00Z',
             currentTime: '2024-01-01T00:00:00Z', multiplier: 10 }
  },
  {
    id: 'vehicle',
    availability: '2024-01-01T00:00:00Z/2024-01-01T01:00:00Z',
    position: {
      interpolationAlgorithm: 'LAGRANGE',
      epoch: '2024-01-01T00:00:00Z',
      cartographicDegrees: [
        0,    116.4, 39.9, 100,   // time=0s:  lon, lat, height
        3600, 121.5, 31.2, 100    // time=3600s
      ]
    },
    model: { gltf: '/models/airplane.glb', scale: 100 }
  }
];

const czmlDataSource = await Cesium.CzmlDataSource.load(czml);
viewer.dataSources.add(czmlDataSource);
viewer.clock.shouldAnimate = true;
```

---

## Turf.js — The PostGIS of the Browser

### Quick Start

```javascript
import * as turf from '@turf/turf';

// Buffer a point by 5km
const point = turf.point([116.4, 39.9]);
const buffered = turf.buffer(point, 5, { units: 'kilometers' });

// Spatial join: tag each point with which polygon it falls in
const points = turf.featureCollection([
  turf.point([116.4, 39.9], { name: 'Point A' }),
  turf.point([116.5, 40.0], { name: 'Point B' })
]);

const polygons = turf.featureCollection([
  turf.polygon([[[116.3,39.8],[116.6,39.8],[116.6,40.1],[116.3,40.1],[116.3,39.8]]],
    { district: 'Chaoyang' })
]);

// Client-side spatial join — no server roundtrip!
points.features.forEach(pt => {
  polygons.features.forEach(poly => {
    if (turf.booleanPointInPolygon(pt, poly)) {
      pt.properties.district = poly.properties.district;
    }
  });
});

// Voronoi diagram from points
const voronoi = turf.voronoi(points, { bbox: [116.0, 39.5, 117.0, 40.5] });

// Nearest point
const nearest = turf.nearestPoint(point, points);

// Area and length
const area = turf.area(buffered);  // square meters
const perimeter = turf.length(buffered, { units: 'kilometers' });

console.log(`Area: ${(area / 1e6).toFixed(2)} km², Perimeter: ${perimeter.toFixed(2)} km`);
```

### TRICK: Fast Hit-Testing vs queryRenderedFeatures

```javascript
// queryRenderedFeatures only hits RENDERED features (clipped, zoomed out features missing)
// turf.booleanPointInPolygon works on raw GeoJSON — always accurate

// Store your polygon data globally
let adminBoundaries = null;
fetch('/data/admin-boundaries.geojson')
  .then(r => r.json())
  .then(data => { adminBoundaries = data; });

map.on('click', (e) => {
  if (!adminBoundaries) return;

  const clickedPoint = turf.point([e.lngLat.lng, e.lngLat.lat]);

  // Find which admin boundary was clicked (O(n) but usually fast for <10K polygons)
  const hit = adminBoundaries.features.find(f =>
    turf.booleanPointInPolygon(clickedPoint, f)
  );

  if (hit) {
    console.log('Clicked admin:', hit.properties.name);
  }
});
```

### Key Functions Reference

```javascript
// Geometry operations
turf.buffer(feature, radius, { units: 'kilometers' })
turf.intersect(poly1, poly2)
turf.union(poly1, poly2)
turf.difference(poly1, poly2)
turf.dissolve(featureCollection, { propertyName: 'admin' })
turf.simplify(feature, { tolerance: 0.001, highQuality: false })
turf.convex(featureCollection)
turf.concave(featureCollection, { maxEdge: 1, units: 'kilometers' })

// Measurement
turf.area(polygon)             // m²
turf.length(lineString, { units: 'kilometers' })
turf.distance(point1, point2, { units: 'miles' })
turf.bearing(point1, point2)   // degrees
turf.midpoint(point1, point2)

// Analysis
turf.centroid(feature)
turf.centerOfMass(feature)
turf.voronoi(points, { bbox })
turf.tin(points, 'elevation')  // Triangulated Irregular Network
turf.interpolate(points, 0.5, { gridType: 'hex', property: 'value', units: 'kilometers' })
turf.isobands(grid, breaks, { zProperty: 'value' })  // contour bands
turf.nearestPoint(target, points)
turf.pointsWithinPolygon(points, polygon)
turf.tag(points, polygons, 'field', 'tagged_as')  // spatial join

// Boolean predicates
turf.booleanPointInPolygon(point, polygon)
turf.booleanContains(polygon, feature)
turf.booleanOverlap(feature1, feature2)
turf.booleanIntersects(feature1, feature2)

// Navigation
turf.along(lineString, 5, { units: 'kilometers' })  // point at distance along line
turf.lineSlice(start, stop, lineString)
turf.nearestPointOnLine(lineString, point)
turf.destination(point, distance, bearing)
```

---

## Kepler.gl

### Visual Analytics Without Code

```javascript
import KeplerGl from '@kepler.gl/components';
import { Provider } from 'react-redux';
import { createStore, combineReducers } from 'redux';
import keplerGlReducer from '@kepler.gl/reducers';

const store = createStore(
  combineReducers({ keplerGl: keplerGlReducer })
);

function App() {
  return (
    <Provider store={store}>
      <KeplerGl
        id="map"
        mapboxApiAccessToken="NOT_NEEDED_FOR_MAPLIBRE"
        width={window.innerWidth}
        height={window.innerHeight}
      />
    </Provider>
  );
}
```

**Use Kepler.gl when:**
- Non-technical stakeholders need to explore large datasets visually
- You want instant heatmaps, arc layers, hexbins, trip animations without writing layer code
- Sharing self-contained HTML snapshots of analysis with clients
- Exploratory data analysis before committing to a custom visualization

**Standalone demo (no install):** https://kepler.gl — drag and drop a CSV or GeoJSON.

---

## Observable Plot + D3-geo

### Custom Cartography with D3 Projections

```javascript
import * as d3 from 'd3';
import * as topojson from 'topojson-client';

const width = 960, height = 600;
const svg = d3.select('#map').append('svg')
  .attr('width', width).attr('height', height);

// Choose a projection: conicEqualArea for China, albers for US, etc.
const projection = d3.geoConicEqualArea()
  .parallels([25, 47])        // standard parallels for China
  .rotate([-105, 0])
  .fitSize([width, height], chinaGeoJSON);

const path = d3.geoPath(projection);

// Choropleth: color by attribute
const colorScale = d3.scaleSequential()
  .domain(d3.extent(chinaGeoJSON.features, d => d.properties.gdp))
  .interpolator(d3.interpolateYlOrRd);

svg.selectAll('.province')
  .data(chinaGeoJSON.features)
  .join('path')
    .attr('class', 'province')
    .attr('d', path)
    .attr('fill', d => colorScale(d.properties.gdp))
    .attr('stroke', 'white')
    .attr('stroke-width', 0.5)
  .append('title')
    .text(d => `${d.properties.name}: ${d.properties.gdp}`);

// Animated transitions
function updateYear(year) {
  svg.selectAll('.province')
    .transition()
    .duration(500)
    .attr('fill', d => colorScale(d.properties[`gdp_${year}`]));
}
```

### Observable Plot for Geo

```javascript
import * as Plot from '@observablehq/plot';

// Dot map
Plot.plot({
  projection: 'mercator',
  marks: [
    Plot.geo(worldGeoJSON, { fill: '#eee', stroke: '#999' }),
    Plot.dot(cities, {
      x: 'longitude',
      y: 'latitude',
      r: d => Math.sqrt(d.population / Math.PI),
      fill: 'continent',
      opacity: 0.7,
      tip: true
    })
  ]
});
```

---

## React / Vue / Svelte Map Components

### react-map-gl (MapLibre + React)

```jsx
import Map, { Source, Layer, Popup, NavigationControl } from 'react-map-gl/maplibre';
import { useState } from 'react';

const buildingLayer = {
  id: '3d-buildings',
  type: 'fill-extrusion',
  source: 'composite',
  'source-layer': 'building',
  paint: {
    'fill-extrusion-color': '#aaa',
    'fill-extrusion-height': ['get', 'height'],
    'fill-extrusion-base': ['get', 'min_height'],
    'fill-extrusion-opacity': 0.8
  }
};

export default function MapComponent({ geojsonData }) {
  const [popupInfo, setPopupInfo] = useState(null);

  return (
    <Map
      initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 12 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://tiles.openfreemap.org/styles/liberty"
      onClick={(e) => {
        const feature = e.features?.[0];
        if (feature) setPopupInfo({ lngLat: e.lngLat, props: feature.properties });
      }}
      interactiveLayerIds={['my-layer']}
    >
      <NavigationControl position="top-right" />

      <Source id="my-source" type="geojson" data={geojsonData}>
        <Layer
          id="my-layer"
          type="circle"
          paint={{ 'circle-color': '#e74c3c', 'circle-radius': 6 }}
        />
      </Source>

      {popupInfo && (
        <Popup
          longitude={popupInfo.lngLat.lng}
          latitude={popupInfo.lngLat.lat}
          onClose={() => setPopupInfo(null)}
        >
          <div>{JSON.stringify(popupInfo.props)}</div>
        </Popup>
      )}
    </Map>
  );
}
```

### svelte-maplibre

```svelte
<script>
  import { MapLibre, GeoJSON, CircleLayer, Popup } from 'svelte-maplibre';

  let clickedFeature = null;
</script>

<MapLibre
  style="https://tiles.openfreemap.org/styles/liberty"
  class="map"
  center={[116.4, 39.9]}
  zoom={10}
>
  <GeoJSON id="points" data="/api/points.geojson">
    <CircleLayer
      id="point-circles"
      paint={{ 'circle-color': '#e74c3c', 'circle-radius': 8 }}
      on:click={(e) => (clickedFeature = e.detail.features[0])}
    />
  </GeoJSON>

  {#if clickedFeature}
    <Popup
      lngLat={clickedFeature.geometry.coordinates}
      on:close={() => (clickedFeature = null)}
    >
      <p>{clickedFeature.properties.name}</p>
    </Popup>
  {/if}
</MapLibre>

<style>
  .map { width: 100%; height: 100vh; }
</style>
```

### vue-maplibre-gl

```vue
<template>
  <mgl-map
    :mapStyle="mapStyle"
    :center="[116.4, 39.9]"
    :zoom="10"
    @map:load="onMapLoad"
  >
    <mgl-navigation-control position="top-right" />
    <mgl-geo-json-source source-id="points" :data="geojsonData">
      <mgl-circle-layer
        layer-id="point-layer"
        :paint="{ 'circle-color': '#e74c3c', 'circle-radius': 6 }"
      />
    </mgl-geo-json-source>
  </mgl-map>
</template>

<script setup>
import { MglMap, MglNavigationControl, MglGeoJsonSource, MglCircleLayer }
  from 'vue-maplibre-gl';

const mapStyle = 'https://tiles.openfreemap.org/styles/liberty';
const geojsonData = ref(null);

const onMapLoad = async (event) => {
  const map = event.map;
  const response = await fetch('/api/points.geojson');
  geojsonData.value = await response.json();
};
</script>
```

---

## Map Design & Basemaps

### Free Basemap Tile Sources (Production-Ready)

| Provider | URL / Style | Notes |
|----------|-------------|-------|
| OpenFreeMap | `https://tiles.openfreemap.org/styles/liberty` | 100% free, self-hostable, no key |
| Protomaps | `https://api.protomaps.com/styles/v2/light.json?key=FREE` | PMTiles-based, generous free tier |
| Stadia Maps | `https://tiles.stadiamaps.com/styles/alidade_smooth.json` | Free tier, no CC |
| Versatiles | `https://tiles.versatiles.org/assets/styles/colorful.json` | Free, German CDN |
| Carto Basemaps | `https://basemaps.cartocdn.com/gl/positron-gl-style/style.json` | Free, widely used |

### Maputnik: Open-Source Style Editor

Maputnik is the open-source alternative to Mapbox Studio for editing MapLibre GL styles visually.

```bash
# Run locally with Docker
docker run -p 8888:8888 maputnik/editor

# Or use the online version: https://maputnik.github.io

# Edit your local style.json file and live-preview changes
```

### TRICK: Self-Host PMTiles Basemap on Cloudflare R2 (Zero Egress Cost)

Cloudflare R2 has **zero egress fees** (unlike AWS S3 which charges per GB transferred). This makes it ideal for hosting PMTiles basemaps.

```bash
# 1. Download a pre-built PMTiles basemap (Protomaps planet files)
#    https://maps.protomaps.com/builds/ — full planet ~110GB, Europe ~15GB

# 2. Upload to R2 using Wrangler CLI
npm install -g wrangler
wrangler r2 bucket create map-tiles
wrangler r2 object put map-tiles/basemap.pmtiles --file=./basemap.pmtiles

# 3. Enable R2 public access in Cloudflare dashboard
#    Bucket Settings → Public Access → Allow

# 4. Use in MapLibre:
```

```javascript
import { Protocol } from 'pmtiles';

const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile.bind(protocol));

// Your R2 bucket's public URL
const BASEMAP_URL = 'pmtiles://https://pub-xxxx.r2.dev/basemap.pmtiles';

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      'basemap': { type: 'vector', url: BASEMAP_URL }
    },
    layers: [/* your layer styles */]
  }
});
```

### Managing Fonts (Glyphs) and Sprites

```bash
# Generate font glyphs from TTF/OTF files for self-hosted vector tile styling
npm install -g fontnik

# Convert fonts to PBF glyph format
mkdir -p fonts/NotoSansCJK
node_modules/.bin/fontnik fill \
  --font fonts/NotoSansCJK.ttf \
  --output fonts/NotoSansCJK

# Generate sprite sheets from SVG icons
npm install -g spritezero-cli
spritezero sprite ./icons/*.svg --output ./sprites/sprite
# Creates sprite.json + sprite.png + sprite@2x.json + sprite@2x.png
```

```javascript
// Reference in style
{
  "version": 8,
  "glyphs": "https://your-cdn.com/fonts/{fontstack}/{range}.pbf",
  "sprite": "https://your-cdn.com/sprites/sprite",
  "sources": { ... },
  "layers": [
    {
      "id": "labels",
      "type": "symbol",
      "layout": {
        "text-field": ["get", "name"],
        "text-font": ["NotoSansCJK Regular", "Open Sans Regular"],
        "icon-image": "marker-icon"  // references sprite sheet
      }
    }
  ]
}
```

---

## Advanced Dark Arts

### WebWorker for Off-Main-Thread GeoJSON Processing

Heavy GeoJSON processing blocks the UI thread and freezes the map. Push it to a Worker.

```javascript
// geo-worker.js
self.onmessage = async ({ data: { type, payload } }) => {
  if (type === 'PROCESS_GEOJSON') {
    // Import Turf in the worker (it's just JS)
    importScripts('https://unpkg.com/@turf/turf/turf.min.js');

    // Heavy processing: buffer 10K features
    const buffered = {
      type: 'FeatureCollection',
      features: payload.features.map(f =>
        turf.buffer(f, 0.5, { units: 'kilometers' })
      )
    };

    self.postMessage({ type: 'RESULT', payload: buffered });
  }
};
```

```javascript
// main.js
const worker = new Worker('/geo-worker.js');

worker.postMessage({ type: 'PROCESS_GEOJSON', payload: rawGeoJSON });

worker.onmessage = ({ data: { type, payload } }) => {
  if (type === 'RESULT') {
    map.getSource('buffered').setData(payload); // update map, no freeze
  }
};
```

### SharedArrayBuffer for Large Dataset Transfer

```javascript
// Avoid copying huge typed arrays between worker and main thread
// Use SharedArrayBuffer for zero-copy transfer

// Main thread: allocate shared memory
const coordsBuffer = new SharedArrayBuffer(numPoints * 2 * 8); // Float64
const coords = new Float64Array(coordsBuffer);

// Fill with coordinates from fetch
const raw = await (await fetch('/api/points')).arrayBuffer();
new Float64Array(raw).forEach((v, i) => coords[i] = v);

// Pass the SAME buffer to worker (no copy!)
worker.postMessage({ type: 'ANALYZE', buffer: coordsBuffer });

// Worker can read/write to the same memory
// REQUIRES cross-origin isolation headers:
// Cross-Origin-Opener-Policy: same-origin
// Cross-Origin-Embedder-Policy: require-corp
```

### MapLibre addImage() with Dynamic SVG Markers

```javascript
// Generate a custom color marker as SVG at runtime
function createColorMarker(color, label) {
  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="40" height="50" viewBox="0 0 40 50">
      <path d="M20 0 C9 0 0 9 0 20 C0 35 20 50 20 50 C20 50 40 35 40 20 C40 9 31 0 20 0Z"
            fill="${color}" stroke="white" stroke-width="2"/>
      <text x="20" y="25" text-anchor="middle" fill="white"
            font-size="14" font-weight="bold" font-family="Arial">${label}</text>
    </svg>`;
  const blob = new Blob([svg], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  return new Promise((resolve) => {
    const img = new Image(40, 50);
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.src = url;
  });
}

map.on('load', async () => {
  // Add different colored markers for different POI categories
  const categories = { restaurant: '#e74c3c', hotel: '#3498db', park: '#27ae60' };

  for (const [category, color] of Object.entries(categories)) {
    const img = await createColorMarker(color, category[0].toUpperCase());
    map.addImage(`marker-${category}`, img, { pixelRatio: 2 });
  }

  map.addLayer({
    id: 'poi-symbols',
    type: 'symbol',
    source: 'pois',
    layout: {
      'icon-image': ['concat', 'marker-', ['get', 'category']],
      'icon-allow-overlap': true,
      'icon-size': 0.8
    }
  });
});
```

### Smooth Animated Data with requestAnimationFrame

```javascript
// Real-time vehicle tracking at 60fps
class VehicleTracker {
  constructor(map) {
    this.map = map;
    this.vehicles = new Map(); // id -> { current, target, progress }
    this.animating = false;

    map.addSource('vehicles', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] }
    });

    map.addLayer({
      id: 'vehicles',
      type: 'symbol',
      source: 'vehicles',
      layout: {
        'icon-image': 'vehicle-icon',
        'icon-rotate': ['get', 'bearing'],
        'icon-rotation-alignment': 'map'
      }
    });
  }

  updatePositions(newPositions) {
    // newPositions: [{ id, lng, lat, bearing }]
    newPositions.forEach(pos => {
      const existing = this.vehicles.get(pos.id);
      if (existing) {
        existing.from = existing.current;
        existing.to = pos;
        existing.progress = 0;
      } else {
        this.vehicles.set(pos.id, { current: pos, from: pos, to: pos, progress: 1 });
      }
    });
    if (!this.animating) this.startAnimation();
  }

  startAnimation() {
    this.animating = true;
    const DURATION = 1000; // ms
    let lastTime = performance.now();

    const frame = (now) => {
      const delta = now - lastTime;
      lastTime = now;

      this.vehicles.forEach((v) => {
        v.progress = Math.min(1, v.progress + delta / DURATION);
        const t = v.progress;
        // Ease in-out interpolation
        const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;
        v.current = {
          lng: v.from.lng + (v.to.lng - v.from.lng) * ease,
          lat: v.from.lat + (v.to.lat - v.from.lat) * ease,
          bearing: v.to.bearing
        };
      });

      this.map.getSource('vehicles').setData({
        type: 'FeatureCollection',
        features: [...this.vehicles.entries()].map(([id, v]) => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [v.current.lng, v.current.lat] },
          properties: { id, bearing: v.current.bearing }
        }))
      });

      if ([...this.vehicles.values()].some(v => v.progress < 1)) {
        requestAnimationFrame(frame);
      } else {
        this.animating = false;
      }
    };

    requestAnimationFrame(frame);
  }
}
```

### Client-Side Reprojection with proj4js

```javascript
import proj4 from 'proj4';

// Register Chinese GCJ-02 (Mars Coordinates) — no official EPSG, use custom
// Note: GCJ-02 is non-linear obfuscation on WGS84, requires dedicated algorithm
// For standard projections:
proj4.defs('EPSG:4490', '+proj=longlat +ellps=GRS80 +no_defs'); // CGCS2000
proj4.defs('EPSG:32650', '+proj=utm +zone=50 +datum=WGS84 +units=m +no_defs'); // UTM 50N

// Convert UTM coords to WGS84 for display
function utmToWgs84(easting, northing, zone = 50) {
  const utmProj = `+proj=utm +zone=${zone} +datum=WGS84 +units=m +no_defs`;
  return proj4(utmProj, 'EPSG:4326', [easting, northing]);
}

const [lng, lat] = utmToWgs84(456789, 4419000);
console.log(`${lng}, ${lat}`); // ~116.4, 39.9

// Batch reproject a GeoJSON from UTM to WGS84
function reprojectGeoJSON(geojson, fromProj, toProj = 'EPSG:4326') {
  return {
    ...geojson,
    features: geojson.features.map(f => ({
      ...f,
      geometry: reprojectGeometry(f.geometry, fromProj, toProj)
    }))
  };
}

function reprojectGeometry(geom, from, to) {
  const reproject = (coord) => proj4(from, to, coord);
  switch (geom.type) {
    case 'Point': return { ...geom, coordinates: reproject(geom.coordinates) };
    case 'LineString': return { ...geom, coordinates: geom.coordinates.map(reproject) };
    case 'Polygon': return { ...geom, coordinates: geom.coordinates.map(r => r.map(reproject)) };
    case 'MultiPolygon':
      return { ...geom, coordinates: geom.coordinates.map(p => p.map(r => r.map(reproject))) };
    default: return geom;
  }
}
```

### Generating Contours Client-Side with d3-contour

```javascript
import { contourDensity, contours } from 'd3-contour';
import { geoPath, geoMercator } from 'd3-geo';
import * as GeoTIFF from 'geotiff';

// Generate contour lines from a raster DEM loaded via GeoTIFF.js
async function generateContours(cogUrl, bbox, thresholds) {
  const tiff = await GeoTIFF.fromUrl(cogUrl);
  const image = await tiff.getImage();

  // Read raster data for bbox
  const data = await image.readRasters({
    bbox: bbox,
    resX: 0.001, resY: 0.001  // ~100m resolution
  });

  const width = data.width;
  const height = data.height;
  const values = Array.from(data[0]); // elevation band

  // Generate contour polygons using D3
  const contourGen = contours()
    .size([width, height])
    .thresholds(thresholds); // e.g. [100, 200, 300, 400, 500]

  const contourFeatures = contourGen(values);

  // Transform from pixel coords to geographic coords
  const [minX, minY, maxX, maxY] = bbox;
  const contourGeoJSON = {
    type: 'FeatureCollection',
    features: contourFeatures.map(c => ({
      type: 'Feature',
      geometry: {
        ...c,
        coordinates: c.coordinates.map(ring =>
          ring.map(polygon =>
            polygon.map(([px, py]) => [
              minX + (px / width) * (maxX - minX),
              maxY - (py / height) * (maxY - minY)
            ])
          )
        )
      },
      properties: { elevation: c.value }
    }))
  };

  return contourGeoJSON;
}

// Add to MapLibre
const contours = await generateContours(
  'https://storage.googleapis.com/dem-tiles/srtm30m.cog.tif',
  [116.0, 39.5, 117.0, 40.5],
  [50, 100, 150, 200, 300, 500]
);

map.addSource('contours', { type: 'geojson', data: contours });
map.addLayer({
  id: 'contour-lines',
  type: 'line',
  source: 'contours',
  paint: {
    'line-color': ['interpolate', ['linear'], ['get', 'elevation'],
      50, '#a8e063', 200, '#f7971e', 500, '#c0392b'],
    'line-width': 1,
    'line-opacity': 0.8
  }
});
```

### CORS Proxy for WMS/WMTS without CORS Headers

Many government and institutional WMS endpoints don't set CORS headers. A lightweight proxy solves this.

```javascript
// Option 1: Use a public CORS proxy (dev only, never in production)
const PROXY = 'https://corsproxy.io/?';
const wmsUrl = PROXY + encodeURIComponent('http://geoserver.gov.example.com/wms?...');

// Option 2: Self-hosted Cloudflare Worker proxy (production-safe, free tier)
// Create a Cloudflare Worker:
```

```javascript
// cloudflare-worker.js (deploy to Cloudflare Workers)
export default {
  async fetch(request) {
    const url = new URL(request.url);
    const targetUrl = url.searchParams.get('url');

    if (!targetUrl) return new Response('Missing url param', { status: 400 });

    // Allowlist trusted sources only
    const allowed = ['geoserver.gov.example.com', 'wms.naturalearth.com'];
    const targetHost = new URL(targetUrl).hostname;
    if (!allowed.some(h => targetHost.endsWith(h))) {
      return new Response('Blocked', { status: 403 });
    }

    const response = await fetch(targetUrl, {
      headers: { 'User-Agent': 'GIS-Proxy/1.0' }
    });

    return new Response(response.body, {
      status: response.status,
      headers: {
        ...Object.fromEntries(response.headers),
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET'
      }
    });
  }
};
```

```javascript
// Use in MapLibre transformRequest:
const map = new maplibregl.Map({
  transformRequest: (url) => {
    if (url.includes('geoserver.gov.example.com')) {
      return {
        url: `https://your-proxy.workers.dev/?url=${encodeURIComponent(url)}`
      };
    }
  }
});
```

### Protobuf Vector Tile Decoding in Custom Workers

```javascript
// custom-tile-worker.js
// Decode MVT tiles manually for custom processing before rendering

importScripts('https://unpkg.com/pbf/dist/pbf.js');
importScripts('https://unpkg.com/@mapbox/vector-tile/dist/vector-tile.js');

self.onmessage = async ({ data: { tileUrl, z, x, y } }) => {
  const response = await fetch(tileUrl);
  const buffer = await response.arrayBuffer();

  const tile = new VectorTile.VectorTile(new Pbf(buffer));

  // Access specific layers within the MVT
  const roadsLayer = tile.layers['roads'];
  if (roadsLayer) {
    const features = [];
    for (let i = 0; i < roadsLayer.length; i++) {
      const feature = roadsLayer.feature(i);
      features.push({
        geometry: feature.toGeoJSON(x, y, z),
        properties: feature.properties
      });
    }
    self.postMessage({ type: 'TILE_DECODED', features, z, x, y });
  }
};
```

---

## Library Selection Guide

```
Requirement                              Recommended Library
─────────────────────────────────────────────────────────────────────
Fastest prototype, zero build step       Leaflet (CDN)
Production vector tiles, WebGL           MapLibre GL JS
3D terrain + buildings                   MapLibre GL JS
Globe view                               MapLibre GL JS v4 / CesiumJS
Millions of data points                  Deck.gl
Time-animated data (vehicles, flights)   Deck.gl TripsLayer
OGC WMS/WFS/WMTS integration             OpenLayers
On-the-fly reprojection                  OpenLayers
3D globe + time-dynamic simulation       CesiumJS
3D Tiles (photogrammetry, BIM, i3s)      CesiumJS
Client-side spatial analysis             Turf.js
Exploratory visual analytics             Kepler.gl
Custom cartographic projections          D3-geo
React application                        react-map-gl + MapLibre
Vue application                          vue-maplibre-gl
Svelte application                       svelte-maplibre
Python/Jupyter notebooks                 pydeck / Folium / ipyleaflet
Zero hosting cost for vector tiles       MapLibre + PMTiles + Cloudflare R2
```

---

## Performance Cheatsheet

| Problem | Solution |
|---------|---------|
| 10K+ markers freezing DOM | Leaflet: `L.canvas()` renderer |
| 100K+ markers | MapLibre `cluster` source |
| 1M+ points | Deck.gl ScatterplotLayer (GPU) |
| Large GeoJSON file | FlatGeobuf with bbox HTTP range requests |
| Heavy computation blocking UI | WebWorker |
| No tile server for vector tiles | PMTiles on S3/R2/GitHub Pages |
| Private tiles need auth | MapLibre `transformRequest` |
| CORS missing on WMS | Cloudflare Worker CORS proxy |
| Custom tile format | MapLibre `addProtocol()` |
| Smooth 60fps animation | `requestAnimationFrame` loop + `source.setData()` |
| Spatial query accuracy | `turf.booleanPointInPolygon()` over `queryRenderedFeatures()` |
| Slow reprojection | Pre-project server-side; client `proj4js` for small datasets |
