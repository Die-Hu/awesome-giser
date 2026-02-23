# Frontend Integration

Integrating interactive maps into modern frontend frameworks requires understanding both the map library lifecycle and the framework's rendering model. This guide covers React, Vue, Svelte, and vanilla JavaScript approaches with practical component patterns.

> **Quick Picks**
> - **React + WebGL maps:** react-map-gl + MapLibre GL JS (most popular 2025 combo)
> - **React + large data:** @deck.gl/react overlay on MapLibre
> - **Vue 3:** vue-maplibre-gl for WebGL, vue-leaflet for SVG
> - **Svelte:** svelte-maplibre (excellent reactive integration)
> - **3D globe:** Resium (CesiumJS for React)

---

## React + Map Libraries

### react-leaflet

The most popular React wrapper for Leaflet. Provides declarative components that map closely to Leaflet's API.

```jsx
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';

function MyMap() {
  return (
    <MapContainer center={[51.505, -0.09]} zoom={13}>
      <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
      <Marker position={[51.505, -0.09]}>
        <Popup>Hello world</Popup>
      </Marker>
    </MapContainer>
  );
}
```

- **Pros**: Mature, large plugin ecosystem, declarative
- **Cons**: Leaflet is SVG/DOM-based, limited performance with many features

### react-map-gl (Mapbox GL / MapLibre)

Uber's React wrapper for Mapbox GL JS / MapLibre GL JS. Best for WebGL-rendered maps.

```jsx
import Map, { Marker, NavigationControl } from 'react-map-gl/maplibre';

function MyMap() {
  return (
    <Map initialViewState={{ longitude: -122.4, latitude: 37.8, zoom: 14 }}>
      <NavigationControl />
      <Marker longitude={-122.4} latitude={37.8} />
    </Map>
  );
}
```

- **Pros**: WebGL performance, 3D support, vector tiles native
- **Cons**: Steeper learning curve, style spec required

### Resium (Cesium for React)

React components for CesiumJS, the leading 3D globe library.

- **Pros**: True 3D globe, terrain, 3D Tiles support
- **Cons**: Large bundle size, complex setup

```jsx
import { Viewer, Entity, CameraFlyTo } from 'resium';
import { Cartesian3 } from 'cesium';

function Globe() {
  return (
    <Viewer full>
      <Entity
        position={Cartesian3.fromDegrees(-122.4, 37.8, 100)}
        point={{ pixelSize: 10, color: Cesium.Color.RED }}
      />
      <CameraFlyTo
        destination={Cartesian3.fromDegrees(-122.4, 37.8, 10000)}
        duration={3}
      />
    </Viewer>
  );
}
```

### Next.js + MapLibre (Most Popular 2025 Pattern)

Next.js requires dynamic imports for map components since MapLibre uses the DOM.

```jsx
// components/Map.jsx
'use client';
import Map, { Source, Layer, NavigationControl } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

export default function GISMap({ geojsonData }) {
  return (
    <Map
      initialViewState={{ longitude: -122.4, latitude: 37.8, zoom: 12 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    >
      <NavigationControl position="top-right" />
      {geojsonData && (
        <Source type="geojson" data={geojsonData}>
          <Layer type="fill" paint={{ 'fill-color': '#088', 'fill-opacity': 0.6 }} />
        </Source>
      )}
    </Map>
  );
}
```

### Deck.gl + React Pattern

deck.gl renders millions of features using GPU acceleration, overlaid on MapLibre.

```jsx
import { DeckGL } from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import Map from 'react-map-gl/maplibre';

function BigDataMap({ data }) {
  const layers = [
    new ScatterplotLayer({
      id: 'scatter',
      data,
      getPosition: d => [d.lng, d.lat],
      getRadius: 50,
      getFillColor: [255, 140, 0],
    }),
  ];

  return (
    <DeckGL
      initialViewState={{ longitude: -122.4, latitude: 37.8, zoom: 10 }}
      controller
      layers={layers}
    >
      <Map mapStyle="https://demotiles.maplibre.org/style.json" />
    </DeckGL>
  );
}
```

---

## Vue + Map Libraries

### vue-leaflet

Vue 3 components for Leaflet with composition API support.

```vue
<template>
  <l-map :zoom="13" :center="[51.505, -0.09]">
    <l-tile-layer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
    <l-marker :lat-lng="[51.505, -0.09]">
      <l-popup>Hello world</l-popup>
    </l-marker>
  </l-map>
</template>
```

### vue-maplibre-gl

Vue 3 wrapper for MapLibre GL JS.

- **Pros**: WebGL performance, good Vue 3 integration
- **Cons**: Smaller community than react-map-gl

```vue
<template>
  <MglMap :mapStyle="styleUrl" :center="[116.4, 39.9]" :zoom="10">
    <MglNavigationControl position="top-right" />
    <MglGeoJsonSource source-id="buildings" :data="geojson">
      <MglFillLayer layer-id="buildings-fill" :paint="{ 'fill-color': '#088' }" />
    </MglGeoJsonSource>
  </MglMap>
</template>

<script setup>
import { MglMap, MglNavigationControl, MglGeoJsonSource, MglFillLayer } from 'vue-maplibre-gl';
const styleUrl = 'https://demotiles.maplibre.org/style.json';
</script>
```

---

## Svelte + Maps

Svelte's reactive model works particularly well with map libraries since map state updates can be handled through stores.

### svelte-maplibre

```svelte
<script>
  import { MapLibre, Marker, NavigationControl } from 'svelte-maplibre';
</script>

<MapLibre center={[-122.4, 37.8]} zoom={14}>
  <NavigationControl />
  <Marker lngLat={[-122.4, 37.8]} />
</MapLibre>
```

### Svelte + MapLibre (Full Example)

```svelte
<script>
  import { MapLibre, Marker, NavigationControl, GeoJSON, FillLayer } from 'svelte-maplibre';

  let geojsonData = null;
  async function loadData() {
    const res = await fetch('/api/zones.geojson');
    geojsonData = await res.json();
  }
  loadData();
</script>

<MapLibre
  style="https://demotiles.maplibre.org/style.json"
  center={[-122.4, 37.8]}
  zoom={12}
  class="w-full h-screen"
>
  <NavigationControl />
  {#if geojsonData}
    <GeoJSON data={geojsonData}>
      <FillLayer paint={{ 'fill-color': '#088', 'fill-opacity': 0.5 }} />
    </GeoJSON>
  {/if}
</MapLibre>
```

### Direct Leaflet Integration (Svelte Action)

```svelte
<script>
  import L from 'leaflet';

  function leafletMap(node) {
    const map = L.map(node).setView([51.505, -0.09], 13);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    return { destroy() { map.remove(); } };
  }
</script>

<div use:leafletMap style="width:100%;height:400px" />
```

---

## State Management with Geospatial Data

### Challenges

- Map state (center, zoom, bounds) changes frequently and must stay in sync with the UI
- Feature collections can be large (thousands of features) and expensive to diff
- Selected features, filters, and layer visibility must be coordinated

### Strategies

| Strategy | When To Use |
|----------|------------|
| Local component state | Simple maps with few interactions |
| Context / provide-inject | Sharing map instance between sibling components |
| Global store (Zustand, Pinia, Svelte stores) | Complex apps with multiple views depending on map state |
| URL state (query params) | Shareable map views, bookmarkable states |
| Server state (React Query, SWR) | Feature data fetched from APIs |

### Tips

- Keep the map instance ref accessible but avoid putting it in reactive state
- Use `useMemo` / `computed` for derived spatial calculations (e.g., features within bounds)
- Debounce map move events before updating global state
- Use GeoJSON as the interchange format between state and map

### URL State Example (React)

```jsx
import { useSearchParams } from 'next/navigation';

function useMapState() {
  const [params, setParams] = useSearchParams();
  const center = [parseFloat(params.get('lat') || 37.8), parseFloat(params.get('lng') || -122.4)];
  const zoom = parseInt(params.get('z') || 12);

  const updateState = (viewState) => {
    setParams({ lat: viewState.latitude.toFixed(4), lng: viewState.longitude.toFixed(4), z: Math.round(viewState.zoom) });
  };
  return { center, zoom, updateState };
}
```

---

## Component Patterns for Map UIs

### Common Patterns

#### Map Container

The root component that initializes the map and provides context to children.

```
┌─────────────────────────────────────┐
│  MapContainer                       │
│  ┌──────────────────────────────┐   │
│  │         Map Canvas           │   │
│  │                              │   │
│  │  ┌────────┐    ┌─────────┐  │   │
│  │  │Controls│    │ Popups  │  │   │
│  │  └────────┘    └─────────┘  │   │
│  └──────────────────────────────┘   │
│  ┌──────────────────────────────┐   │
│  │       Sidebar / Panel        │   │
│  └──────────────────────────────┘   │
└─────────────────────────────────────┘
```

#### Layer Control

A panel or toggle group that manages layer visibility.

#### Draw Tools

Toolbar components for creating/editing geometries (polygons, lines, points).

#### Geocoding Search

An autocomplete input that converts addresses to coordinates and vice versa.

#### Feature Inspector

A panel that displays attributes of a clicked or selected feature.


---

## Framework Comparison

| Feature | React | Vue | Svelte | Vanilla JS |
|---------|-------|-----|--------|------------|
| Leaflet wrapper | react-leaflet | vue-leaflet | svelte-leaflet | Direct API |
| MapLibre wrapper | react-map-gl | vue-maplibre-gl | svelte-maplibre | Direct API |
| Cesium wrapper | resium | vue-cesium | — | Direct API |
| Deck.gl integration | @deck.gl/react | — | — | @deck.gl/core |
| Bundle size overhead | Medium | Medium | Low | None |
| Learning curve | Medium | Low | Low | Varies |
| Ecosystem size | Large | Medium | Small | Large |
| SSR support | Next.js (dynamic) | Nuxt (client-only) | SvelteKit (client-only) | N/A |

**Wrapper Library Links:**
- [react-leaflet](https://react-leaflet.js.org/) | [react-map-gl](https://visgl.github.io/react-map-gl/) | [resium](https://resium.reearth.io/) | [@deck.gl/react](https://deck.gl/docs/get-started/using-with-react)
- [vue-leaflet](https://github.com/vue-leaflet/vue-leaflet) | [vue-maplibre-gl](https://indoorequal.github.io/vue-maplibre-gl/) | [vue-cesium](https://zouyaoji.top/vue-cesium/)
- [svelte-maplibre](https://github.com/dimfeld/svelte-maplibre)

---

## Common Integration Patterns

| Pattern | Description | Libraries |
|---------|-------------|-----------|
| Map Container | Root wrapper that manages map lifecycle | All |
| Layer Control | Toggle layer visibility from UI | react-leaflet, MapLibre |
| Draw Tools | Create/edit geometries on the map | Leaflet Draw, MapLibre Draw |
| Geocoding | Search bar with address autocomplete | Nominatim, Pelias, Mapbox |
| Feature Popup | Display feature info on click/hover | All |
| Synchronized Maps | Two maps that move together (compare view) | All (custom) |
| Minimap | Small overview map showing current extent | Leaflet plugin, custom |
