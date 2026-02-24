# Framework Integration -- React, Vue, Svelte & SSR

> **Quick Picks**
> - **SOTA (most features/ecosystem):** react-map-gl
> - **Free Best (DX + bundle):** svelte-maplibre
> - **Fastest Setup:** vue-maplibre-gl

A production map app is not just a map library — it is a map library embedded inside a component model, a state system, a routing layer, and a build pipeline. Every major framework has a different answer to the questions "who owns the DOM?", "how do you clean up side effects?", and "what triggers a re-render?". This guide covers the canonical approach for each framework, real production patterns, common traps, and how to escape them.

---

## React

React is the most common choice for geo frontends. The ecosystem is the largest, the wrappers are battle-tested, and the component model maps reasonably well onto map layers and sources.

### react-map-gl (Uber / Vis.gl)

react-map-gl is the canonical React wrapper for MapLibre GL JS (and Mapbox GL JS v1/v2). It ships two modes: **controlled** (you own viewport state) and **uncontrolled** (the map owns it). Most production apps start uncontrolled and move to controlled once URL sync or deep linking is required.

#### Installation

```bash
npm install react-map-gl maplibre-gl
```

#### Controlled vs Uncontrolled

```tsx
// Uncontrolled: map manages its own viewport
import Map from 'react-map-gl/maplibre';

function SimpleMap() {
  return (
    <Map
      initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 10 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    />
  );
}
```

```tsx
// Controlled: you own every aspect of viewport state
import { useState, useCallback } from 'react';
import Map, { ViewState } from 'react-map-gl/maplibre';

function ControlledMap() {
  const [viewState, setViewState] = useState<ViewState>({
    longitude: 116.4,
    latitude: 39.9,
    zoom: 10,
    bearing: 0,
    pitch: 0,
    padding: { top: 0, bottom: 0, left: 0, right: 0 },
  });

  return (
    <Map
      {...viewState}
      onMove={evt => setViewState(evt.viewState)}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    />
  );
}
```

Use controlled mode when:
- You need to sync the map viewport to URL params
- Multiple components need to read or set the viewport
- You want deterministic snapshot testing

Use uncontrolled mode when:
- The map is a standalone feature with no external viewport consumers
- You want the fastest possible setup

#### `<Source>` + `<Layer>` for Declarative Layer Management

react-map-gl renders sources and layers as React children. The library syncs them to the underlying MapLibre instance.

```tsx
import Map, { Source, Layer, LayerProps } from 'react-map-gl/maplibre';

const fillLayer: LayerProps = {
  id: 'districts-fill',
  type: 'fill',
  paint: {
    'fill-color': ['get', 'color'],
    'fill-opacity': 0.6,
  },
};

const lineLayer: LayerProps = {
  id: 'districts-line',
  type: 'line',
  paint: {
    'line-color': '#ffffff',
    'line-width': 1,
  },
};

function DistrictMap({ geojson }: { geojson: GeoJSON.FeatureCollection }) {
  return (
    <Map
      initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 9 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    >
      <Source id="districts" type="geojson" data={geojson}>
        <Layer {...fillLayer} />
        <Layer {...lineLayer} />
      </Source>
    </Map>
  );
}
```

Layer ordering matters: layers render in the order they appear in JSX. If you need to insert a layer below map labels, use the `beforeId` prop on `<Layer>`.

```tsx
<Layer {...fillLayer} beforeId="road-label" />
```

#### `useMap()` Hook for Imperative Access

When you need the raw MapLibre map instance — to call `flyTo`, `fitBounds`, `queryRenderedFeatures`, or add custom controls — use `useMap()` inside any descendant of `<Map>`.

```tsx
import Map, { useMap } from 'react-map-gl/maplibre';

function FlyToButton() {
  const { current: map } = useMap();

  const flyToShanghai = () => {
    map?.flyTo({ center: [121.47, 31.23], zoom: 12, duration: 2000 });
  };

  return <button onClick={flyToShanghai}>Shanghai</button>;
}

function App() {
  return (
    <Map
      id="main"
      initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 6 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    >
      <FlyToButton />
    </Map>
  );
}
```

If you have multiple maps on screen, assign each an `id` prop and pass it to `useMap('main')` to select the right instance.

#### `useControl()` for Custom Controls

```tsx
import { useControl } from 'react-map-gl/maplibre';
import maplibregl from 'maplibre-gl';

class MyControl implements maplibregl.IControl {
  private _container: HTMLDivElement | undefined;

  onAdd(map: maplibregl.Map): HTMLElement {
    this._container = document.createElement('div');
    this._container.className = 'maplibregl-ctrl maplibregl-ctrl-group';
    this._container.innerHTML = '<button>X</button>';
    return this._container;
  }

  onRemove(): void {
    this._container?.remove();
  }
}

function CustomControl() {
  useControl(() => new MyControl(), { position: 'top-right' });
  return null;
}
```

#### Event Handling

```tsx
import Map, { MapMouseEvent, MapLayerMouseEvent } from 'react-map-gl/maplibre';

function InteractiveMap() {
  const handleMapClick = useCallback((evt: MapMouseEvent) => {
    console.log('Clicked at', evt.lngLat);
  }, []);

  const handleLayerClick = useCallback((evt: MapLayerMouseEvent) => {
    const feature = evt.features?.[0];
    if (feature) {
      console.log('Feature:', feature.properties);
    }
  }, []);

  return (
    <Map
      initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 10 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
      onClick={handleMapClick}
      interactiveLayerIds={['districts-fill']}
      onMouseEnter={handleLayerClick}
    />
  );
}
```

Always wrap event callbacks in `useCallback` to prevent unnecessary re-renders. Passing a new function reference on every render causes react-map-gl to re-bind the event listener each time.

#### Performance Pitfalls

**Problem 1: Re-creating layer style objects on every render**

```tsx
// BAD: new object on every render, map diff thinks layer changed
function BadMap() {
  return (
    <Source id="points" type="geojson" data={data}>
      <Layer
        id="points"
        type="circle"
        paint={{ 'circle-radius': 6, 'circle-color': '#ff0000' }}
      />
    </Source>
  );
}

// GOOD: stable reference outside component or memoized
const circleLayer: LayerProps = {
  id: 'points',
  type: 'circle',
  paint: { 'circle-radius': 6, 'circle-color': '#ff0000' },
};

function GoodMap() {
  return (
    <Source id="points" type="geojson" data={data}>
      <Layer {...circleLayer} />
    </Source>
  );
}
```

**Problem 2: Storing large GeoJSON in React state**

```tsx
// BAD: 50MB GeoJSON in useState triggers React re-render machinery
const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection | null>(null);

// GOOD: use a ref for the data, update the map source directly
const mapRef = useRef<MapRef>(null);

useEffect(() => {
  fetch('/api/features')
    .then(r => r.json())
    .then(data => {
      const source = mapRef.current?.getSource('my-source') as maplibregl.GeoJSONSource;
      source?.setData(data);
    });
}, []);
```

**Problem 3: Inline onClick creating new references**

```tsx
// BAD
<Map onClick={(e) => handleClick(e)} />

// GOOD
const handleClick = useCallback((e) => { /* ... */ }, [/* deps */]);
<Map onClick={handleClick} />
```

#### Complete Production Example: Choropleth Map with Popup

```tsx
import { useState, useCallback, useMemo } from 'react';
import Map, { Source, Layer, Popup, LayerProps, MapLayerMouseEvent } from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

interface District {
  name: string;
  value: number;
}

const fillLayerStyle: LayerProps = {
  id: 'districts-fill',
  type: 'fill',
  paint: {
    'fill-color': [
      'interpolate',
      ['linear'],
      ['get', 'value'],
      0, '#f7fcb9',
      50, '#addd8e',
      100, '#31a354',
    ],
    'fill-opacity': 0.75,
  },
};

const lineLayerStyle: LayerProps = {
  id: 'districts-line',
  type: 'line',
  paint: { 'line-color': '#555', 'line-width': 0.5 },
};

const hoverLayerStyle: LayerProps = {
  id: 'districts-hover',
  type: 'fill',
  paint: { 'fill-color': '#000', 'fill-opacity': 0.15 },
  filter: ['==', ['get', 'id'], ''],
};

export function ChoroplethMap({ geojson }: { geojson: GeoJSON.FeatureCollection }) {
  const [popup, setPopup] = useState<{ lng: number; lat: number; data: District } | null>(null);
  const [hoverId, setHoverId] = useState<string>('');

  const hoverLayer = useMemo<LayerProps>(
    () => ({ ...hoverLayerStyle, filter: ['==', ['get', 'id'], hoverId] }),
    [hoverId]
  );

  const onLayerClick = useCallback((evt: MapLayerMouseEvent) => {
    const feature = evt.features?.[0];
    if (!feature) return;
    setPopup({
      lng: evt.lngLat.lng,
      lat: evt.lngLat.lat,
      data: { name: feature.properties?.name, value: feature.properties?.value },
    });
  }, []);

  const onMouseMove = useCallback((evt: MapLayerMouseEvent) => {
    const feature = evt.features?.[0];
    setHoverId(feature?.properties?.id ?? '');
  }, []);

  const onMouseLeave = useCallback(() => setHoverId(''), []);

  return (
    <Map
      initialViewState={{ longitude: 104, latitude: 35, zoom: 4 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
      interactiveLayerIds={['districts-fill']}
      onClick={onLayerClick}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      cursor={hoverId ? 'pointer' : 'default'}
    >
      <Source id="districts" type="geojson" data={geojson}>
        <Layer {...fillLayerStyle} />
        <Layer {...hoverLayer} />
        <Layer {...lineLayerStyle} />
      </Source>

      {popup && (
        <Popup
          longitude={popup.lng}
          latitude={popup.lat}
          onClose={() => setPopup(null)}
          closeOnClick={false}
          anchor="bottom"
        >
          <div>
            <strong>{popup.data.name}</strong>
            <br />
            Value: {popup.data.value}
          </div>
        </Popup>
      )}
    </Map>
  );
}
```

---

### @deck.gl/react

Deck.gl excels at rendering millions of data points with GPU instancing. The React wrapper is `@deck.gl/react`.

```bash
npm install @deck.gl/react @deck.gl/layers @deck.gl/geo-layers deck.gl
```

```tsx
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import Map from 'react-map-gl/maplibre';

const INITIAL_VIEW = {
  longitude: 116.4,
  latitude: 39.9,
  zoom: 11,
  pitch: 45,
  bearing: 0,
};

interface Point {
  coordinates: [number, number];
  value: number;
}

function DeckMap({ points }: { points: Point[] }) {
  const layers = useMemo(
    () => [
      new ScatterplotLayer<Point>({
        id: 'scatterplot',
        data: points,
        getPosition: d => d.coordinates,
        getRadius: d => Math.sqrt(d.value) * 10,
        getFillColor: [255, 140, 0, 200],
        pickable: true,
        updateTriggers: { getRadius: points },
      }),
    ],
    [points]
  );

  const getTooltip = useCallback(
    ({ object }: { object: Point | null }) =>
      object ? { html: `<div>Value: ${object.value}</div>` } : null,
    []
  );

  return (
    <DeckGL
      initialViewState={INITIAL_VIEW}
      controller={true}
      layers={layers}
      getTooltip={getTooltip}
    >
      <Map mapStyle="https://demotiles.maplibre.org/style.json" />
    </DeckGL>
  );
}
```

Key patterns:
- `layers` array: deck.gl diffs layer instances by `id`. Provide stable layer instances via `useMemo`.
- `updateTriggers`: tells deck.gl which accessor functions need to be re-evaluated when data changes without object identity change.
- `getTooltip`: render tooltip for the hovered object. Wrap in `useCallback`.
- Interleaved rendering: place `<Map>` as a child of `<DeckGL>` to composite deck.gl layers with MapLibre layers using the same WebGL context.

---

### Resium (CesiumJS for React)

Resium wraps CesiumJS with a component-per-entity pattern. CesiumJS is 1–2 MB of JavaScript; use dynamic import and code splitting aggressively.

```bash
npm install resium cesium
```

```tsx
import { Viewer, Entity, Cesium3DTileset, ImageryLayer } from 'resium';
import { Cartesian3, Color, IonResource } from 'cesium';

function CesiumScene() {
  return (
    <Viewer full>
      <Entity
        name="Beijing"
        position={Cartesian3.fromDegrees(116.4, 39.9, 0)}
        point={{ pixelSize: 10, color: Color.RED }}
        description="Capital of China"
      />
      <Cesium3DTileset url={IonResource.fromAssetId(96188)} />
    </Viewer>
  );
}
```

For Next.js, always lazy-load:

```tsx
import dynamic from 'next/dynamic';
const CesiumScene = dynamic(() => import('./CesiumScene'), { ssr: false });
```

---

### State Management with React Map Apps

#### Zustand (Recommended)

Zustand is the best fit for map state: minimal boilerplate, no context provider, works seamlessly with react-map-gl controlled mode.

```bash
npm install zustand
```

```ts
// store/mapStore.ts
import { create } from 'zustand';
import { ViewState } from 'react-map-gl/maplibre';

interface Feature {
  id: string;
  name: string;
  properties: Record<string, unknown>;
}

interface MapStore {
  viewState: ViewState;
  setViewState: (vs: ViewState) => void;
  selectedFeature: Feature | null;
  selectFeature: (f: Feature | null) => void;
  layerVisibility: Record<string, boolean>;
  toggleLayer: (id: string) => void;
}

export const useMapStore = create<MapStore>((set) => ({
  viewState: {
    longitude: 116.4, latitude: 39.9, zoom: 10,
    bearing: 0, pitch: 0,
    padding: { top: 0, bottom: 0, left: 0, right: 0 },
  },
  setViewState: (viewState) => set({ viewState }),
  selectedFeature: null,
  selectFeature: (selectedFeature) => set({ selectedFeature }),
  layerVisibility: { parcels: true, buildings: true, transit: false },
  toggleLayer: (id) =>
    set((s) => ({
      layerVisibility: { ...s.layerVisibility, [id]: !s.layerVisibility[id] },
    })),
}));
```

```tsx
// components/MapView.tsx
import Map from 'react-map-gl/maplibre';
import { useMapStore } from '../store/mapStore';

export function MapView() {
  const { viewState, setViewState } = useMapStore();

  return (
    <Map
      {...viewState}
      onMove={evt => setViewState(evt.viewState)}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    />
  );
}

// components/Sidebar.tsx -- reads the same store, no prop drilling
import { useMapStore } from '../store/mapStore';

export function Sidebar() {
  const { selectedFeature, layerVisibility, toggleLayer } = useMapStore();

  return (
    <div>
      {selectedFeature && <div>{selectedFeature.name}</div>}
      {Object.entries(layerVisibility).map(([id, visible]) => (
        <label key={id}>
          <input type="checkbox" checked={visible} onChange={() => toggleLayer(id)} />
          {id}
        </label>
      ))}
    </div>
  );
}
```

#### Jotai (Atomic State)

Jotai works best for fine-grained layer state where different components subscribe to different atoms.

```ts
import { atom, useAtom } from 'jotai';

export const parcelVisibleAtom = atom(true);
export const buildingVisibleAtom = atom(true);
export const selectedFeatureAtom = atom<GeoJSON.Feature | null>(null);

// Derived atom: only show layer controls when there is data
export const hasDataAtom = atom(
  (get) => get(parcelVisibleAtom) || get(buildingVisibleAtom)
);
```

---

### Next.js and Remix

#### The SSR Challenge

MapLibre GL JS accesses `window`, `document`, and WebGL on import. Server-side rendering will throw `ReferenceError: window is not defined`.

**Next.js solution: `dynamic` import with `ssr: false`**

```tsx
// app/map/page.tsx (App Router)
import dynamic from 'next/dynamic';

const MapView = dynamic(() => import('@/components/MapView'), {
  ssr: false,
  loading: () => <div style={{ height: '100vh' }} className="animate-pulse bg-gray-200" />,
});

export default function MapPage() {
  return <MapView />;
}
```

**Remix solution: `useEffect` + client-only state**

```tsx
// routes/map.tsx
import { useEffect, useState } from 'react';

export default function MapRoute() {
  const [MapComponent, setMapComponent] = useState<React.ComponentType | null>(null);

  useEffect(() => {
    import('../components/MapView').then(mod => setMapComponent(() => mod.MapView));
  }, []);

  if (!MapComponent) return <div>Loading map…</div>;
  return <MapComponent />;
}
```

#### URL State Sync (Next.js App Router)

```tsx
'use client';
import { useRouter, useSearchParams } from 'next/navigation';
import { useCallback, useEffect } from 'react';
import Map, { ViewState } from 'react-map-gl/maplibre';

function parseViewState(params: URLSearchParams): Partial<ViewState> {
  return {
    longitude: parseFloat(params.get('lng') ?? '116.4'),
    latitude: parseFloat(params.get('lat') ?? '39.9'),
    zoom: parseFloat(params.get('z') ?? '10'),
    bearing: parseFloat(params.get('b') ?? '0'),
    pitch: parseFloat(params.get('p') ?? '0'),
  };
}

export function URLSyncedMap() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const viewState = parseViewState(searchParams);

  const onMoveEnd = useCallback(
    (evt: { viewState: ViewState }) => {
      const { longitude, latitude, zoom, bearing, pitch } = evt.viewState;
      const params = new URLSearchParams({
        lng: longitude.toFixed(6),
        lat: latitude.toFixed(6),
        z: zoom.toFixed(2),
        b: bearing.toFixed(1),
        p: pitch.toFixed(1),
      });
      router.replace(`?${params.toString()}`, { scroll: false });
    },
    [router]
  );

  return (
    <Map
      {...viewState}
      onMoveEnd={onMoveEnd}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    />
  );
}
```

#### Server-side Spatial Queries (Next.js API Routes)

```ts
// app/api/features/route.ts
import { NextRequest, NextResponse } from 'next/server';
import { db } from '@/lib/db'; // Drizzle or Prisma client

export async function GET(req: NextRequest) {
  const { searchParams } = req.nextUrl;
  const bbox = searchParams.get('bbox'); // "minLng,minLat,maxLng,maxLat"
  if (!bbox) return NextResponse.json({ error: 'bbox required' }, { status: 400 });

  const [minLng, minLat, maxLng, maxLat] = bbox.split(',').map(Number);

  // PostGIS query via Drizzle raw SQL
  const features = await db.execute<{ geojson: string }>(
    `SELECT jsonb_build_object(
       'type', 'Feature',
       'id', id,
       'geometry', ST_AsGeoJSON(geom)::json,
       'properties', properties
     ) AS geojson
     FROM parcels
     WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326)
     LIMIT 5000`,
    [minLng, minLat, maxLng, maxLat]
  );

  const geojson = {
    type: 'FeatureCollection',
    features: features.rows.map(r => JSON.parse(r.geojson)),
  };

  return NextResponse.json(geojson, {
    headers: { 'Cache-Control': 's-maxage=60, stale-while-revalidate=300' },
  });
}
```

---

## Vue 3

Vue 3's Composition API and reactivity system integrate naturally with map libraries. The `ref`/`reactive` model maps onto map state elegantly.

### vue-maplibre-gl

The most complete Vue 3 wrapper for MapLibre GL JS. Uses a provide/inject pattern internally so child components can access the map instance.

```bash
npm install vue-maplibre-gl maplibre-gl
```

#### Basic Usage

```vue
<template>
  <MglMap
    :map-style="mapStyle"
    :center="[116.4, 39.9]"
    :zoom="10"
    @map:load="onMapLoad"
    style="width: 100%; height: 100vh"
  >
    <MglNavigationControl position="top-right" />
    <MglGeolocateControl position="top-right" />
    <MglScaleControl position="bottom-left" />

    <MglGeojsonSource source-id="districts" :data="geojson">
      <MglFillLayer
        layer-id="districts-fill"
        :paint="{ 'fill-color': '#088', 'fill-opacity': 0.6 }"
        @click="onLayerClick"
      />
      <MglLineLayer
        layer-id="districts-line"
        :paint="{ 'line-color': '#fff', 'line-width': 1 }"
      />
    </MglGeojsonSource>

    <MglMarker :coordinates="[116.4, 39.9]">
      <template #marker>
        <div class="custom-marker">★</div>
      </template>
    </MglMarker>
  </MglMap>
</template>

<script setup lang="ts">
import {
  MglMap, MglNavigationControl, MglGeolocateControl, MglScaleControl,
  MglGeojsonSource, MglFillLayer, MglLineLayer, MglMarker,
} from 'vue-maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const mapStyle = 'https://demotiles.maplibre.org/style.json';
const geojson = ref<GeoJSON.FeatureCollection | null>(null);

async function onMapLoad() {
  const response = await fetch('/api/districts');
  geojson.value = await response.json();
}

function onLayerClick(evt: any) {
  const feature = evt.features?.[0];
  if (feature) console.log(feature.properties);
}
</script>
```

#### `useMap()` Composable

```vue
<script setup lang="ts">
import { useMap } from 'vue-maplibre-gl';

const { map } = useMap(); // must be called inside a child of <MglMap>

function flyToBeijing() {
  map.value?.flyTo({ center: [116.4, 39.9], zoom: 12 });
}
</script>
```

#### Reactive Viewport with `v-model`

```vue
<template>
  <MglMap v-model:center="center" v-model:zoom="zoom" ... />
  <div>Center: {{ center.join(', ') }} | Zoom: {{ zoom.toFixed(1) }}</div>
</template>

<script setup lang="ts">
import { ref } from 'vue';
const center = ref<[number, number]>([116.4, 39.9]);
const zoom = ref(10);
</script>
```

#### Slot-based Popups with Vue Components Inside

One of vue-maplibre-gl's best features: popups can contain real Vue components.

```vue
<template>
  <MglMap ...>
    <MglGeojsonSource source-id="points" :data="points">
      <MglCircleLayer layer-id="points-circle" :paint="circlePaint" @click="onPointClick" />
    </MglGeojsonSource>

    <MglPopup v-if="selectedPoint" :coordinates="selectedPoint.coords" @close="selectedPoint = null">
      <FeatureCard :feature="selectedPoint.feature" />
    </MglPopup>
  </MglMap>
</template>

<script setup lang="ts">
import { ref } from 'vue';
import FeatureCard from './FeatureCard.vue';

interface SelectedPoint {
  coords: [number, number];
  feature: GeoJSON.Feature;
}

const selectedPoint = ref<SelectedPoint | null>(null);

function onPointClick(evt: any) {
  const feature = evt.features?.[0];
  if (!feature) return;
  selectedPoint.value = {
    coords: [evt.lngLat.lng, evt.lngLat.lat],
    feature,
  };
}
</script>
```

#### Complete Production Example: Vue 3 Heatmap

```vue
<template>
  <div class="map-wrapper">
    <MglMap
      :map-style="style"
      :center="[104, 35]"
      :zoom="4"
      style="width: 100%; height: 100vh"
    >
      <MglGeojsonSource source-id="crimes" :data="data">
        <MglHeatmapLayer
          layer-id="crimes-heat"
          :paint="heatmapPaint"
          :max-zoom="15"
        />
        <MglCircleLayer
          layer-id="crimes-point"
          :paint="circlePaint"
          :min-zoom="14"
        />
      </MglGeojsonSource>
    </MglMap>

    <div class="controls">
      <label>
        Radius
        <input type="range" min="5" max="50" v-model.number="radius" />
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue';

const radius = ref(20);

const heatmapPaint = computed(() => ({
  'heatmap-radius': radius.value,
  'heatmap-opacity': 0.85,
  'heatmap-color': [
    'interpolate', ['linear'], ['heatmap-density'],
    0, 'rgba(33,102,172,0)',
    0.2, 'rgb(103,169,207)',
    0.5, 'rgb(253,219,199)',
    0.8, 'rgb(239,138,98)',
    1, 'rgb(178,24,43)',
  ],
}));

const circlePaint = {
  'circle-radius': 4,
  'circle-color': '#ff5500',
  'circle-opacity': 0.8,
};

const data = ref<GeoJSON.FeatureCollection>({ type: 'FeatureCollection', features: [] });
const style = 'https://demotiles.maplibre.org/style.json';

onMounted(async () => {
  data.value = await fetch('/api/points').then(r => r.json());
});
</script>
```

---

### Nuxt

#### Client-only Map Initialization

```vue
<!-- pages/map.vue -->
<template>
  <div>
    <ClientOnly>
      <MapView />
      <template #fallback>
        <div class="h-screen animate-pulse bg-gray-200" />
      </template>
    </ClientOnly>
  </div>
</template>
```

Or with a Nuxt plugin:

```ts
// plugins/maplibre.client.ts  (the .client suffix = only runs in browser)
import maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

export default defineNuxtPlugin(() => {
  return { provide: { maplibre: maplibregl } };
});
```

#### Nuxt Server Routes for Geo Queries

```ts
// server/api/bbox.get.ts
import { defineEventHandler, getQuery } from 'h3';

export default defineEventHandler(async (event) => {
  const { minLng, minLat, maxLng, maxLat } = getQuery(event);
  // Query PostGIS here
  return { type: 'FeatureCollection', features: [] };
});
```

---

## Svelte / SvelteKit

Svelte has the best developer experience for map applications. No virtual DOM means updates are minimal and predictable. The reactivity model aligns perfectly with map state.

### svelte-maplibre

```bash
npm install svelte-maplibre maplibre-gl
```

#### Core Components

svelte-maplibre provides ergonomic, idiomatic Svelte components. Every component that needs the map instance gets it automatically via Svelte context.

```svelte
<script>
  import { MapLibre, GeoJSON, FillLayer, LineLayer, CircleLayer, Marker, Popup } from 'svelte-maplibre';
</script>

<MapLibre
  style="https://demotiles.maplibre.org/style.json"
  center={[116.4, 39.9]}
  zoom={10}
  class="w-full h-screen"
>
  <GeoJSON id="districts" data={geojson}>
    <FillLayer
      id="districts-fill"
      paint={{ 'fill-color': '#088', 'fill-opacity': 0.6 }}
    />
    <LineLayer
      id="districts-line"
      paint={{ 'line-color': '#fff', 'line-width': 1 }}
    />
  </GeoJSON>
</MapLibre>
```

#### `bind:map` for Imperative Access

```svelte
<script>
  import { MapLibre } from 'svelte-maplibre';
  import maplibregl from 'maplibre-gl';

  let map;

  function flyToShanghai() {
    map?.flyTo({ center: [121.47, 31.23], zoom: 12 });
  }
</script>

<MapLibre bind:map style="..." center={[116.4, 39.9]} zoom={10} class="h-screen" />
<button on:click={flyToShanghai}>Shanghai</button>
```

#### Reactive Statements for Derived Geo Data

```svelte
<script>
  import { MapLibre, GeoJSON, CircleLayer } from 'svelte-maplibre';

  export let rawData = [];
  export let minValue = 0;

  // $: re-runs whenever rawData or minValue changes
  $: filteredGeoJSON = {
    type: 'FeatureCollection',
    features: rawData
      .filter(d => d.value >= minValue)
      .map(d => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [d.lng, d.lat] },
        properties: { value: d.value },
      })),
  };
</script>

<MapLibre style="..." center={[116.4, 39.9]} zoom={8} class="h-screen">
  <GeoJSON id="points" data={filteredGeoJSON}>
    <CircleLayer
      id="points-circle"
      paint={{ 'circle-radius': 6, 'circle-color': '#f90' }}
    />
  </GeoJSON>
</MapLibre>
```

#### Svelte Stores for Cross-Component Map State

```ts
// stores/mapState.ts
import { writable, derived } from 'svelte/store';

export const center = writable<[number, number]>([116.4, 39.9]);
export const zoom = writable(10);
export const selectedFeature = writable<GeoJSON.Feature | null>(null);
export const layerVisibility = writable<Record<string, boolean>>({
  parcels: true,
  buildings: true,
});

// Derived store: bbox from center + zoom (approximate)
export const bbox = derived([center, zoom], ([$center, $zoom]) => {
  const degrees = 360 / Math.pow(2, $zoom);
  return [
    $center[0] - degrees,
    $center[1] - degrees / 2,
    $center[0] + degrees,
    $center[1] + degrees / 2,
  ] as [number, number, number, number];
});
```

```svelte
<!-- MapView.svelte -->
<script>
  import { MapLibre } from 'svelte-maplibre';
  import { center, zoom } from '../stores/mapState';
</script>

<MapLibre
  style="..."
  bind:center={$center}
  bind:zoom={$zoom}
  class="h-screen"
/>
```

#### Complete Production Example: SvelteKit Choropleth

```svelte
<!-- src/routes/map/+page.svelte -->
<script lang="ts">
  import { MapLibre, GeoJSON, FillLayer, LineLayer } from 'svelte-maplibre';
  import type { PageData } from './$types';

  export let data: PageData; // loaded server-side

  let hoveredId: string | null = null;
  let popupInfo: { lng: number; lat: number; name: string; value: number } | null = null;

  $: hoverFilter = ['==', ['get', 'id'], hoveredId ?? ''];

  function handleClick(e: CustomEvent) {
    const feature = e.detail.features?.[0];
    if (!feature) { popupInfo = null; return; }
    popupInfo = {
      lng: e.detail.lngLat.lng,
      lat: e.detail.lngLat.lat,
      name: feature.properties.name,
      value: feature.properties.value,
    };
  }

  function handleMouseMove(e: CustomEvent) {
    hoveredId = e.detail.features?.[0]?.properties?.id ?? null;
  }
</script>

<MapLibre
  style="https://demotiles.maplibre.org/style.json"
  center={[104, 35]}
  zoom={4}
  class="w-full h-screen"
>
  <GeoJSON id="districts" data={data.geojson}>
    <FillLayer
      id="districts-fill"
      paint={{
        'fill-color': ['interpolate', ['linear'], ['get', 'value'],
          0, '#f7fcb9', 100, '#31a354'],
        'fill-opacity': 0.75,
      }}
      on:click={handleClick}
      on:mousemove={handleMouseMove}
      on:mouseleave={() => hoveredId = null}
    />
    <FillLayer
      id="districts-hover"
      paint={{ 'fill-color': '#000', 'fill-opacity': 0.15 }}
      filter={hoverFilter}
    />
    <LineLayer
      id="districts-line"
      paint={{ 'line-color': '#555', 'line-width': 0.5 }}
    />
  </GeoJSON>
</MapLibre>

{#if popupInfo}
  <div class="popup" style="left: {popupInfo.lng}px; top: {popupInfo.lat}px">
    <strong>{popupInfo.name}</strong>: {popupInfo.value}
    <button on:click={() => popupInfo = null}>×</button>
  </div>
{/if}
```

```ts
// src/routes/map/+page.server.ts
import type { PageServerLoad } from './$types';
import { db } from '$lib/db';

export const load: PageServerLoad = async () => {
  const features = await db.query.districts.findMany();
  return {
    geojson: {
      type: 'FeatureCollection',
      features: features.map(f => ({
        type: 'Feature',
        id: f.id,
        geometry: JSON.parse(f.geom),
        properties: { id: f.id, name: f.name, value: f.value },
      })),
    },
  };
};
```

#### Why Svelte Wins for Maps

- **No VDOM overhead**: Svelte compiles to direct DOM mutations. Map state updates do not bubble through a reconciler.
- **`$:` reactive statements**: trivially express "re-compute this layer filter when the slider changes."
- **Smaller bundle**: Svelte adds ~2 KB runtime vs ~44 KB for React or ~33 KB for Vue.
- **Stores**: a clean primitive for sharing map state across components without context providers.
- **SvelteKit `+page.server.ts`**: run spatial queries at the server, return pre-processed GeoJSON, no client-side fetch needed.

#### Svelte 5 Runes

```svelte
<script lang="ts">
  import { MapLibre, GeoJSON, FillLayer } from 'svelte-maplibre';

  let rawFeatures = $state<GeoJSON.Feature[]>([]);
  let minValue = $state(0);

  // $derived replaces $: computed values
  const filtered = $derived({
    type: 'FeatureCollection' as const,
    features: rawFeatures.filter(f => (f.properties?.value ?? 0) >= minValue),
  });

  $effect(() => {
    fetch('/api/features').then(r => r.json()).then(d => { rawFeatures = d.features; });
  });
</script>

<MapLibre style="..." center={[116.4, 39.9]} zoom={10} class="h-screen">
  <GeoJSON id="filtered" data={filtered}>
    <FillLayer id="fill" paint={{ 'fill-color': '#f90', 'fill-opacity': 0.7 }} />
  </GeoJSON>
</MapLibre>

<input type="range" min="0" max="100" bind:value={minValue} />
```

---

## Angular

Angular's DI system, Zone.js change detection, and lifecycle hooks require specific patterns when integrating map libraries.

### Custom MapLibre in Angular (Manual Wrapping)

No official Angular wrapper for MapLibre GL JS with modern Angular support exists. The recommended approach is a thin service + directive pattern.

```ts
// map.service.ts
import { Injectable, OnDestroy } from '@angular/core';
import maplibregl from 'maplibre-gl';
import { BehaviorSubject } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class MapService implements OnDestroy {
  private map: maplibregl.Map | null = null;
  readonly map$ = new BehaviorSubject<maplibregl.Map | null>(null);

  init(container: HTMLElement, options: maplibregl.MapOptions): maplibregl.Map {
    this.map = new maplibregl.Map({ container, ...options });
    this.map$.next(this.map);
    return this.map;
  }

  ngOnDestroy(): void {
    this.map?.remove();
    this.map = null;
  }
}
```

```ts
// map.component.ts
import {
  Component, ElementRef, ViewChild, AfterViewInit,
  OnDestroy, NgZone, Input,
} from '@angular/core';
import maplibregl from 'maplibre-gl';
import { MapService } from './map.service';

@Component({
  selector: 'app-map',
  template: '<div #mapContainer style="width:100%;height:100vh;"></div>',
  standalone: true,
})
export class MapComponent implements AfterViewInit, OnDestroy {
  @ViewChild('mapContainer', { static: true }) container!: ElementRef<HTMLDivElement>;
  @Input() center: [number, number] = [116.4, 39.9];
  @Input() zoom = 10;

  constructor(private mapService: MapService, private ngZone: NgZone) {}

  ngAfterViewInit(): void {
    // Critical: run map initialization outside Angular's zone
    // Map events fire constantly (mousemove, render) — inside zone they
    // would trigger change detection on every frame.
    this.ngZone.runOutsideAngular(() => {
      this.mapService.init(this.container.nativeElement, {
        style: 'https://demotiles.maplibre.org/style.json',
        center: this.center,
        zoom: this.zoom,
      });
    });
  }

  ngOnDestroy(): void {
    this.mapService.ngOnDestroy();
  }
}
```

**Zone.js rule**: always call `ngZone.runOutsideAngular()` when setting up map event listeners. Only call `ngZone.run()` when you need to update Angular component state (e.g., on feature click to update a sidebar).

```ts
// Inside map initialization, outside Angular zone
this.map.on('click', 'my-layer', (evt) => {
  const feature = evt.features?.[0];
  if (!feature) return;
  // Re-enter zone to update Angular state
  this.ngZone.run(() => {
    this.selectedFeature = feature;
  });
});
```

### @terrestris/ol-angular

For enterprise applications that require WMS/WFS/WMTS, OpenLayers in Angular is the right choice.

```bash
npm install @terrestris/ol-angular ol
```

```html
<!-- map.component.html -->
<lot-map [zoom]="zoom" [center]="center">
  <lot-tile-layer [source]="osmSource"></lot-tile-layer>
  <lot-vector-layer [source]="wfsSource" [style]="featureStyle"></lot-vector-layer>
  <lot-scaleline-control></lot-scaleline-control>
</lot-map>
```

When to prefer OpenLayers over MapLibre in Angular:
- Hard requirement for WMS GetFeatureInfo
- WMTS tiling with custom projections (EPSG:3857, EPSG:4326, custom CRS)
- OGC-compliant filtering (CQL, OGC Filter)
- Military/government environments where Mapbox tiles are blocked

---

## Solid.js / Qwik and Other Frameworks

### Solid.js

No official wrapper exists. Use a ref + `createEffect` for initialization:

```tsx
import { onMount, onCleanup, createSignal, createEffect } from 'solid-js';
import maplibregl from 'maplibre-gl';

function SolidMap() {
  let container: HTMLDivElement | undefined;
  let map: maplibregl.Map | undefined;
  const [center, setCenter] = createSignal<[number, number]>([116.4, 39.9]);

  onMount(() => {
    map = new maplibregl.Map({
      container: container!,
      style: 'https://demotiles.maplibre.org/style.json',
      center: center(),
      zoom: 10,
    });
  });

  createEffect(() => {
    map?.setCenter(center());
  });

  onCleanup(() => {
    map?.remove();
  });

  return <div ref={container} style={{ width: '100%', height: '100vh' }} />;
}
```

Solid's fine-grained reactivity is ideal for maps: only the signals that change cause `createEffect` to re-run, with zero VDOM overhead.

### Qwik

Qwik's resumability model requires careful handling of non-serializable objects like a map instance.

```tsx
import { component$, useVisibleTask$, useSignal } from '@builder.io/qwik';

export const QwikMap = component$(() => {
  const containerRef = useSignal<HTMLDivElement>();

  // useVisibleTask$ runs only in the browser, after the component is visible
  useVisibleTask$(({ cleanup }) => {
    if (!containerRef.value) return;
    const { default: maplibregl } = await import('maplibre-gl');
    const map = new maplibregl.Map({
      container: containerRef.value,
      style: 'https://demotiles.maplibre.org/style.json',
      center: [116.4, 39.9],
      zoom: 10,
    });
    cleanup(() => map.remove());
  });

  return <div ref={containerRef} style={{ width: '100%', height: '100vh' }} />;
});
```

Qwik's lazy hydration means the map library is only downloaded when the map becomes visible — a significant performance win for pages with maps below the fold.

---

## Cross-Framework Patterns

### Map Component Architecture

A well-structured map application separates concerns into discrete layers:

```
MapContainer (framework component)
├── MapProvider (context/store for map instance)
├── SourceManager (data sources, auto-cleanup)
├── LayerManager (ordered layers, visibility)
├── InteractionManager (click, hover, draw)
├── ControlsManager (zoom, geolocate, scale)
└── PopupManager (framework-rendered popups on map)
```

This pattern works in every framework. The `MapProvider` is a React context / Vue provide-inject / Svelte `setContext` that makes the map instance available to all descendants without prop drilling.

### Shared Challenges and Solutions

#### 1. Client-only Rendering

Every framework needs to prevent map libraries from running on the server.

| Framework | Solution |
|-----------|----------|
| Next.js | `dynamic(() => import('./Map'), { ssr: false })` |
| Nuxt | `<ClientOnly>` component or `.client.ts` plugin |
| SvelteKit | `onMount` / `browser` import guard |
| Remix | `useEffect` state flag |
| Angular | maps only in browser, no SSR issue by default |
| Astro | `client:only="react"` directive |

```ts
// SvelteKit: guard against SSR
import { browser } from '$app/environment';

if (browser) {
  const maplibregl = await import('maplibre-gl');
  // initialize map
}
```

#### 2. Cleanup on Unmount

Always call `map.remove()`. Failure to do so leaks WebGL contexts (browsers limit these to ~16), event listeners, and Worker threads.

```ts
// React
useEffect(() => {
  const map = new maplibregl.Map({ ... });
  return () => map.remove(); // cleanup
}, []);

// Vue
onUnmounted(() => map.value?.remove());

// Svelte
onDestroy(() => map?.remove());

// Angular
ngOnDestroy() { this.map?.remove(); }

// Solid
onCleanup(() => map?.remove());
```

#### 3. Re-render Optimization

Map state changes should not propagate full component re-renders. Key rules:

- Keep the map instance in a `ref` / non-reactive variable, not in state
- Use shallow refs in Vue (`shallowRef`) for the map instance to avoid Vue's deep proxy wrapping a 100 MB GeoJSON object
- In React, update the map source directly with `source.setData()` rather than putting data in React state
- Debounce `onMove` handlers that sync to external state

```vue
<script setup>
// WRONG: deep reactivity on map instance causes Vue to proxy all 400+ methods
const map = ref<maplibregl.Map | null>(null);

// CORRECT: shallow ref prevents Vue from instrumenting the map object
const map = shallowRef<maplibregl.Map | null>(null);
</script>
```

#### 4. Popups with Framework Components

Framework-rendered popups (with reactivity, event handlers, component logic) need to be mounted into DOM nodes that are then passed to the map's popup system.

```tsx
// React: render a React component into a detached DOM node for a map popup
import { createRoot } from 'react-dom/client';
import maplibregl from 'maplibre-gl';

function addReactPopup(map: maplibregl.Map, lngLat: maplibregl.LngLatLike, feature: GeoJSON.Feature) {
  const container = document.createElement('div');
  const root = createRoot(container);
  root.render(<FeaturePopup feature={feature} />);

  const popup = new maplibregl.Popup()
    .setLngLat(lngLat)
    .setDOMContent(container)
    .addTo(map);

  popup.on('close', () => root.unmount()); // cleanup!
  return popup;
}
```

```ts
// Vue: use createApp + mount into detached node
import { createApp } from 'vue';
import FeaturePopup from './FeaturePopup.vue';
import maplibregl from 'maplibre-gl';

function addVuePopup(map: maplibregl.Map, lngLat: [number, number], props: object) {
  const container = document.createElement('div');
  const app = createApp(FeaturePopup, props);
  app.mount(container);

  const popup = new maplibregl.Popup()
    .setLngLat(lngLat)
    .setDOMContent(container)
    .addTo(map);

  popup.on('close', () => app.unmount());
}
```

#### 5. Coordinate State and URL Sync

Shareable map links are a hard requirement for most geo applications. The pattern is:

```
URL params ←→ application state ←→ map viewport
```

Encode center, zoom, bearing, and pitch in the URL hash. Update the URL on `moveend` (not `move` — too frequent). Parse the URL on load to set `initialViewState`.

```ts
// Universal URL state encoder/decoder
export function encodeViewState(vs: { center: [number, number]; zoom: number; bearing: number; pitch: number }): string {
  const { center, zoom, bearing, pitch } = vs;
  return `${center[1].toFixed(5)},${center[0].toFixed(5)},${zoom.toFixed(2)}z,${bearing.toFixed(0)}b,${pitch.toFixed(0)}p`;
}

export function decodeViewState(hash: string) {
  const [lat, lng, z, b, p] = hash.split(',');
  return {
    center: [parseFloat(lng), parseFloat(lat)] as [number, number],
    zoom: parseFloat(z),
    bearing: parseFloat(b),
    pitch: parseFloat(p),
  };
}

// Usage: update URL on moveend
map.on('moveend', () => {
  const hash = encodeViewState({
    center: map.getCenter().toArray() as [number, number],
    zoom: map.getZoom(),
    bearing: map.getBearing(),
    pitch: map.getPitch(),
  });
  window.history.replaceState(null, '', `#${hash}`);
});
```

#### 6. Responsive Maps

```ts
// Use ResizeObserver, not window resize event
const observer = new ResizeObserver(() => map.resize());
observer.observe(containerElement);
// cleanup:
observer.disconnect();
```

---

### Performance Anti-Patterns

**Anti-pattern 1: Re-creating layer objects on every render**

```tsx
// BAD — new object reference every render tells react-map-gl the layer changed
<Layer paint={{ 'fill-color': '#088', 'fill-opacity': 0.6 }} />

// GOOD — stable reference
const paint = useMemo(() => ({ 'fill-color': '#088', 'fill-opacity': 0.6 }), []);
<Layer paint={paint} />
```

**Anti-pattern 2: Updating map state in tight loops**

```ts
// BAD — fires change detection / store updates at 60 fps
map.on('mousemove', evt => store.setMousePosition(evt.lngLat));

// GOOD — batch with rAF
let pending: maplibregl.LngLat | null = null;
map.on('mousemove', evt => { pending = evt.lngLat; });
requestAnimationFrame(function tick() {
  if (pending) { store.setMousePosition(pending); pending = null; }
  requestAnimationFrame(tick);
});
```

**Anti-pattern 3: Large GeoJSON in framework state**

```tsx
// BAD — 10 MB GeoJSON triggers React reconciler, Vue deep reactive proxy, etc.
const [geojson, setGeojson] = useState<GeoJSON.FeatureCollection>();
// ...
setGeojson(await fetchHugeDataset()); // React re-renders entire tree

// GOOD — bypass framework state, update map source directly
const mapRef = useRef<MapRef>(null);
useEffect(() => {
  fetchHugeDataset().then(data => {
    const source = mapRef.current?.getSource('my-source') as maplibregl.GeoJSONSource;
    source?.setData(data); // map updates without React reconciliation
  });
}, []);
```

**Anti-pattern 4: Not cleaning up on unmount**

See section 2 above. Every framework. Every time. No exceptions.

---

### Testing Geo Components

#### Unit / Integration: Vitest + happy-dom

```ts
// __tests__/MapView.test.ts
import { vi, describe, it, expect } from 'vitest';

// Mock MapLibre GL JS — it requires WebGL which happy-dom doesn't support
vi.mock('maplibre-gl', () => {
  const Map = vi.fn().mockImplementation(() => ({
    on: vi.fn(),
    off: vi.fn(),
    remove: vi.fn(),
    flyTo: vi.fn(),
    getSource: vi.fn(),
    addSource: vi.fn(),
    addLayer: vi.fn(),
    removeLayer: vi.fn(),
    removeSource: vi.fn(),
    getZoom: vi.fn().mockReturnValue(10),
    getCenter: vi.fn().mockReturnValue({ lng: 116.4, lat: 39.9, toArray: () => [116.4, 39.9] }),
  }));
  return { default: { Map } };
});

// Now test your component normally
import { render, fireEvent } from '@testing-library/react';
import { MapView } from '../src/components/MapView';

describe('MapView', () => {
  it('calls flyTo when FlyToButton is clicked', async () => {
    const { getByText } = render(<MapView />);
    fireEvent.click(getByText('Shanghai'));
    // assert via store / callback / spy
  });
});
```

#### Visual Regression: Playwright

```ts
// tests/map.spec.ts
import { test, expect } from '@playwright/test';

test('map renders choropleth layer', async ({ page }) => {
  await page.goto('/map');
  await page.waitForFunction(() =>
    document.querySelector('canvas')?.getContext('webgl') !== null
  );
  // Wait for tiles to load
  await page.waitForTimeout(2000);
  await expect(page).toHaveScreenshot('choropleth.png', { maxDiffPixels: 100 });
});

test('popup appears on district click', async ({ page }) => {
  await page.goto('/map');
  await page.waitForSelector('canvas');
  // Click at approximate screen coordinates for a known district
  await page.click('canvas', { position: { x: 400, y: 300 } });
  await expect(page.locator('.maplibregl-popup')).toBeVisible();
});
```

#### Storybook with Mock Data

```tsx
// MapView.stories.tsx
import type { Meta, StoryObj } from '@storybook/react';
import { ChoroplethMap } from './ChoroplethMap';
import mockGeojson from '../mocks/districts.json';

const meta: Meta<typeof ChoroplethMap> = {
  title: 'Maps/ChoroplethMap',
  component: ChoroplethMap,
  parameters: { layout: 'fullscreen' },
};
export default meta;

type Story = StoryObj<typeof ChoroplethMap>;

export const Default: Story = {
  args: { geojson: mockGeojson },
};

export const Empty: Story = {
  args: { geojson: { type: 'FeatureCollection', features: [] } },
};
```

---

## Comprehensive Framework Comparison

| Aspect | React + react-map-gl | Vue 3 + vue-maplibre-gl | Svelte + svelte-maplibre | Angular |
|--------|---------------------|------------------------|--------------------------|---------|
| DX quality | Good | Good | **Best** | Moderate |
| Bundle overhead | +44 KB (React) | +33 KB (Vue) | **+2 KB** (Svelte) | +80 KB (Angular) |
| SSR support | Next.js `dynamic` | Nuxt `<ClientOnly>` | SvelteKit `browser` | Angular Universal |
| TypeScript | Full | Full | Full | Full |
| State sync | Zustand / Jotai | Pinia / composables | Stores | Services / NgRx |
| Ecosystem | **Largest** | Growing | Small but quality | Smallest |
| Learning curve | Medium | Low–Medium | **Low** | High |
| Job market | Largest | Large | Small | Enterprise |
| WebGL integration | Good (deck.gl) | Moderate | Moderate | Weak |
| Enterprise WMS/WFS | via OpenLayers | via OpenLayers | via OpenLayers | **@terrestris/ol-angular** |
| 3D / Globe | Resium | vue-cesium | Manual | cesium-angular |

**Decision guide:**

- Building a new greenfield product with a small team → **svelte-maplibre**
- Existing React codebase, need rich data viz → **react-map-gl + deck.gl**
- Existing Vue 3 / Nuxt app → **vue-maplibre-gl**
- Enterprise Angular with WMS/WFS requirements → **@terrestris/ol-angular**
- 3D globe / satellite / building visualization → **Resium** (React) or **vue-cesium**
- Maximum performance with millions of points → **deck.gl** (React or vanilla)

---

## Advanced Dark Arts

### React: `useImperativeHandle` to Expose Map Methods

Avoid prop-drilling the map instance by exposing it via ref:

```tsx
import { forwardRef, useImperativeHandle, useRef } from 'react';
import Map, { MapRef } from 'react-map-gl/maplibre';

export interface PublicMapAPI {
  flyTo(center: [number, number], zoom?: number): void;
  fitBounds(bbox: [number, number, number, number]): void;
}

export const MapView = forwardRef<PublicMapAPI>((props, ref) => {
  const mapRef = useRef<MapRef>(null);

  useImperativeHandle(ref, () => ({
    flyTo(center, zoom = 12) {
      mapRef.current?.flyTo({ center, zoom });
    },
    fitBounds(bbox) {
      mapRef.current?.fitBounds(bbox, { padding: 40 });
    },
  }));

  return (
    <Map ref={mapRef}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://demotiles.maplibre.org/style.json"
    />
  );
});

// Parent component:
function App() {
  const mapAPI = useRef<PublicMapAPI>(null);
  return (
    <>
      <MapView ref={mapAPI} />
      <button onClick={() => mapAPI.current?.flyTo([121.47, 31.23])}>Shanghai</button>
    </>
  );
}
```

### Vue: `shallowRef` for Map Instance

Never use `ref()` for a MapLibre map instance. Vue's deep reactivity will try to proxy all methods and properties of the map, creating massive overhead and breaking internal WeakMap-based state.

```ts
import { shallowRef } from 'vue';
import maplibregl from 'maplibre-gl';

// CORRECT
const map = shallowRef<maplibregl.Map | null>(null);

// ALSO CORRECT for large datasets
const hugeGeoJSON = shallowRef<GeoJSON.FeatureCollection | null>(null);
// Vue won't deep-proxy 100 MB of feature data
```

### Svelte 5: `$effect` for Reactive Derived Layers

```svelte
<script lang="ts">
  import maplibregl from 'maplibre-gl';

  let map: maplibregl.Map | undefined;
  let minValue = $state(0);
  let rawFeatures = $state<GeoJSON.Feature[]>([]);

  // Re-runs whenever minValue or rawFeatures change
  $effect(() => {
    const source = map?.getSource('filtered') as maplibregl.GeoJSONSource;
    if (!source) return;
    source.setData({
      type: 'FeatureCollection',
      features: rawFeatures.filter(f => (f.properties?.value ?? 0) >= minValue),
    });
  });
</script>
```

### Lazy Loading Heavy Map Libraries

MapLibre GL JS is ~220 KB gzip. CesiumJS is ~2 MB. Never bundle them in your initial chunk.

```ts
// React / Next.js
const MapView = dynamic(() => import('./MapView'), { ssr: false });

// Vue / Nuxt (Vite-native lazy)
const MapView = defineAsyncComponent(() => import('./MapView.vue'));

// SvelteKit
// +page.svelte
{#if browser}
  {#await import('./MapView.svelte') then { default: MapView }}
    <svelte:component this={MapView} />
  {/await}
{/if}
```

### Web Worker for Spatial Operations

Heavy operations — simplification, clustering, spatial joins, coordinate transformations — should never block the main thread.

```ts
// workers/geo.worker.ts
import * as turf from '@turf/turf';

self.onmessage = (evt: MessageEvent) => {
  const { type, payload } = evt.data;

  if (type === 'cluster') {
    const clustered = turf.clustersKmeans(payload.geojson, { numberOfClusters: payload.k });
    self.postMessage({ type: 'cluster:result', payload: clustered });
  }

  if (type === 'simplify') {
    const simplified = turf.simplify(payload.geojson, {
      tolerance: payload.tolerance,
      highQuality: true,
    });
    self.postMessage({ type: 'simplify:result', payload: simplified });
  }
};

// Usage in component (framework-agnostic):
const worker = new Worker(new URL('./workers/geo.worker.ts', import.meta.url), { type: 'module' });

worker.postMessage({ type: 'simplify', payload: { geojson: hugeGeoJSON, tolerance: 0.001 } });

worker.onmessage = (evt) => {
  if (evt.data.type === 'simplify:result') {
    // update map source with simplified data
    source.setData(evt.data.payload);
  }
};
```

### Next.js Edge Runtime for Lightweight Geo Queries

Edge Runtime runs on Cloudflare Workers / Vercel Edge. No Node.js APIs, but sub-10ms response times globally. Useful for simple bbox queries and tile proxying.

```ts
// app/api/tiles/[z]/[x]/[y]/route.ts
export const runtime = 'edge';

export async function GET(
  request: Request,
  { params }: { params: { z: string; x: string; y: string } }
) {
  const { z, x, y } = params;
  // Proxy to your tile server or Cloudflare R2 bucket
  const tileUrl = `https://your-tile-server.com/tiles/${z}/${x}/${y}.pbf`;
  const upstream = await fetch(tileUrl);

  return new Response(upstream.body, {
    headers: {
      'Content-Type': 'application/x-protobuf',
      'Content-Encoding': 'gzip',
      'Cache-Control': 'public, max-age=86400',
      'Access-Control-Allow-Origin': '*',
    },
  });
}
```

### URL State for Shareable Map Links (Complete Implementation)

```ts
// lib/mapUrlState.ts
export interface MapViewState {
  center: [number, number];
  zoom: number;
  bearing: number;
  pitch: number;
}

const DEFAULT: MapViewState = {
  center: [116.4, 39.9],
  zoom: 10,
  bearing: 0,
  pitch: 0,
};

export function readFromURL(): MapViewState {
  if (typeof window === 'undefined') return DEFAULT;
  const hash = window.location.hash.slice(1);
  if (!hash) return DEFAULT;
  try {
    const parts = hash.split('/');
    if (parts.length < 3) return DEFAULT;
    return {
      zoom: parseFloat(parts[0]),
      center: [parseFloat(parts[2]), parseFloat(parts[1])],
      bearing: parseFloat(parts[3] ?? '0'),
      pitch: parseFloat(parts[4] ?? '0'),
    };
  } catch {
    return DEFAULT;
  }
}

export function writeToURL(vs: MapViewState): void {
  const { center, zoom, bearing, pitch } = vs;
  const hash = [
    zoom.toFixed(2),
    center[1].toFixed(6),
    center[0].toFixed(6),
    bearing.toFixed(1),
    pitch.toFixed(1),
  ].join('/');
  window.history.replaceState(null, '', `#${hash}`);
}

// URL format: #zoom/lat/lng/bearing/pitch
// Example:    #10.00/39.900000/116.400000/0.0/0.0
```

---

## Checklist for Production Map Apps

Before shipping a map feature to production, verify:

- [ ] Map container has explicit dimensions (width + height, not just `flex: 1` without a parent height)
- [ ] `map.remove()` called on component unmount in all code paths
- [ ] Map library dynamically imported — not in the initial bundle
- [ ] Event callbacks wrapped in `useCallback` / `useMemo` / stable references
- [ ] Large GeoJSON updated via `source.setData()`, not framework state
- [ ] `ngZone.runOutsideAngular()` wrapping map init (Angular only)
- [ ] `shallowRef` used for map instance and large datasets (Vue only)
- [ ] `ResizeObserver` calling `map.resize()` on container size change
- [ ] URL state sync implemented if map position needs to be shareable or bookmarkable
- [ ] Popup DOM nodes properly unmounted when popup closes
- [ ] Web Worker used for any spatial operation taking > 16 ms
- [ ] Visual regression tests for key map states
- [ ] Tile requests cached with appropriate `Cache-Control` headers
- [ ] Attribution displayed (required by MapLibre / OpenStreetMap licenses)
