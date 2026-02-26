# Frontend Integration — 2025 Complete Guide

Integrating interactive maps into modern frontend frameworks requires understanding both the map library lifecycle and the framework's rendering model. This guide covers React, Vue, Svelte, and vanilla JavaScript approaches with practical component patterns, state management, accessibility, and testing.

> **Quick Picks**
> - **React + WebGL maps:** react-map-gl v7 + MapLibre GL JS 4.x
> - **React + large data:** @deck.gl/react 9.x overlay on MapLibre
> - **React + 3D globe:** Resium (CesiumJS for React)
> - **Vue 3:** vue-maplibre-gl for WebGL, vue-leaflet for SVG
> - **Svelte:** svelte-maplibre (excellent reactive integration)
> - **Next.js 15:** App Router + dynamic imports + react-map-gl
> - **Framework-agnostic:** Web Components with Lit + MapLibre

---

## React Ecosystem

### react-map-gl v7 + MapLibre GL JS 4.x

The most popular combination in 2025. react-map-gl provides controlled/uncontrolled map components with hooks API.

```jsx
// app/components/MapView.jsx
'use client';
import { useCallback, useState, useMemo } from 'react';
import Map, {
  Source, Layer, Marker, Popup,
  NavigationControl, ScaleControl, GeolocateControl,
  FullscreenControl, AttributionControl
} from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

const INITIAL_VIEW = {
  longitude: 116.4,
  latitude: 39.9,
  zoom: 10,
  pitch: 0,
  bearing: 0,
};

export default function MapView({ geojsonData, onFeatureClick }) {
  const [viewState, setViewState] = useState(INITIAL_VIEW);
  const [hoverInfo, setHoverInfo] = useState(null);
  const [selectedFeature, setSelectedFeature] = useState(null);

  const onHover = useCallback((event) => {
    const feature = event.features?.[0];
    setHoverInfo(feature ? { feature, x: event.point.x, y: event.point.y } : null);
  }, []);

  const onClick = useCallback((event) => {
    const feature = event.features?.[0];
    if (feature) {
      setSelectedFeature(feature);
      onFeatureClick?.(feature);
    }
  }, [onFeatureClick]);

  const fillLayer = useMemo(() => ({
    id: 'data-fill',
    type: 'fill',
    paint: {
      'fill-color': [
        'interpolate', ['linear'],
        ['get', 'value'],
        0, '#f7fbff',
        50, '#6baed6',
        100, '#08306b'
      ],
      'fill-opacity': [
        'case',
        ['boolean', ['feature-state', 'hover'], false],
        0.9, 0.6
      ]
    }
  }), []);

  const lineLayer = useMemo(() => ({
    id: 'data-line',
    type: 'line',
    paint: {
      'line-color': '#333',
      'line-width': [
        'case',
        ['boolean', ['feature-state', 'hover'], false],
        2, 0.5
      ]
    }
  }), []);

  return (
    <Map
      {...viewState}
      onMove={(evt) => setViewState(evt.viewState)}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
      interactiveLayerIds={['data-fill']}
      onMouseMove={onHover}
      onClick={onClick}
      cursor={hoverInfo ? 'pointer' : 'grab'}
      attributionControl={false}
    >
      <NavigationControl position="top-right" />
      <ScaleControl position="bottom-left" />
      <GeolocateControl position="top-right" />
      <FullscreenControl position="top-right" />
      <AttributionControl position="bottom-right" compact />

      {geojsonData && (
        <Source id="spatial-data" type="geojson" data={geojsonData}>
          <Layer {...fillLayer} />
          <Layer {...lineLayer} />
        </Source>
      )}

      {selectedFeature && (
        <Popup
          longitude={selectedFeature.geometry.coordinates[0]}
          latitude={selectedFeature.geometry.coordinates[1]}
          onClose={() => setSelectedFeature(null)}
          closeOnClick={false}
          maxWidth="320px"
        >
          <div className="p-2">
            <h3 className="font-bold">{selectedFeature.properties.name}</h3>
            <p>Value: {selectedFeature.properties.value}</p>
          </div>
        </Popup>
      )}
    </Map>
  );
}
```

### useMap Hook — Accessing Map Instance

```jsx
import { useMap } from 'react-map-gl/maplibre';

function FlyToButton({ longitude, latitude }) {
  const { current: map } = useMap();

  const handleClick = () => {
    map?.flyTo({
      center: [longitude, latitude],
      zoom: 14,
      duration: 2000,
      essential: true,
    });
  };

  return <button onClick={handleClick}>Fly to location</button>;
}
```

### useControl Hook — Custom Controls

```jsx
import { useControl } from 'react-map-gl/maplibre';
import MaplibreGeocoder from '@maplibre/maplibre-gl-geocoder';

function GeocoderControl({ position = 'top-left' }) {
  useControl(
    () => {
      const geocoder = new MaplibreGeocoder(
        {
          forwardGeocode: async (config) => {
            const response = await fetch(
              `https://nominatim.openstreetmap.org/search?q=${config.query}&format=geojson&limit=5`
            );
            const data = await response.json();
            return { features: data.features };
          },
        },
        { maplibregl }
      );
      return geocoder;
    },
    { position }
  );

  return null;
}
```

### Next.js 15 App Router Integration

```jsx
// app/map/page.tsx
import dynamic from 'next/dynamic';
import { Suspense } from 'react';

const MapView = dynamic(() => import('@/components/MapView'), {
  ssr: false,
  loading: () => (
    <div className="w-full h-screen flex items-center justify-center bg-gray-100">
      <div className="animate-pulse text-gray-500">Loading map...</div>
    </div>
  ),
});

export default function MapPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <MapView />
    </Suspense>
  );
}
```

```ts
// app/api/features/route.ts — Server-side spatial API
import { NextResponse } from 'next/server';
import { Pool } from 'pg';

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const bbox = searchParams.get('bbox')?.split(',').map(Number);

  if (!bbox || bbox.length !== 4) {
    return NextResponse.json({ error: 'Invalid bbox' }, { status: 400 });
  }

  const result = await pool.query(
    `SELECT id, name, value,
            ST_AsGeoJSON(geom)::json AS geometry
     FROM features
     WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326)
     LIMIT 5000`,
    bbox
  );

  const geojson = {
    type: 'FeatureCollection',
    features: result.rows.map((row) => ({
      type: 'Feature',
      properties: { id: row.id, name: row.name, value: row.value },
      geometry: row.geometry,
    })),
  };

  return NextResponse.json(geojson, {
    headers: { 'Cache-Control': 'public, max-age=300, stale-while-revalidate=600' },
  });
}
```

### deck.gl 9.x + React — GPU-Accelerated Visualization

```jsx
import { DeckGL } from '@deck.gl/react';
import { Map } from 'react-map-gl/maplibre';
import { ScatterplotLayer, ArcLayer } from '@deck.gl/layers';
import { DataFilterExtension } from '@deck.gl/extensions';
import { useState, useMemo, useCallback } from 'react';

function BigDataMap({ pointData, flowData }) {
  const [timeFilter, setTimeFilter] = useState([0, 24]);

  const layers = useMemo(() => [
    new ScatterplotLayer({
      id: 'points',
      data: pointData,
      getPosition: d => [d.lng, d.lat],
      getRadius: d => Math.sqrt(d.value) * 10,
      getFillColor: d => {
        const t = d.value / 100;
        return [255 * t, 255 * (1 - t), 0, 180];
      },
      radiusMinPixels: 2,
      radiusMaxPixels: 50,
      pickable: true,
      extensions: [new DataFilterExtension({ filterSize: 1 })],
      getFilterValue: d => d.hour,
      filterRange: timeFilter,
      updateTriggers: { getFilterValue: timeFilter },
    }),

    new ArcLayer({
      id: 'flows',
      data: flowData,
      getSourcePosition: d => d.from,
      getTargetPosition: d => d.to,
      getSourceColor: [0, 128, 255],
      getTargetColor: [255, 0, 128],
      getWidth: d => Math.log2(d.count + 1),
      pickable: true,
    }),
  ], [pointData, flowData, timeFilter]);

  const getTooltip = useCallback(({ object }) => {
    if (!object) return null;
    return {
      html: `<div><b>${object.name || 'Point'}</b><br/>Value: ${object.value}</div>`,
      style: { backgroundColor: '#fff', fontSize: '12px' },
    };
  }, []);

  return (
    <div className="relative w-full h-screen">
      <DeckGL
        initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 10, pitch: 45 }}
        controller
        layers={layers}
        getTooltip={getTooltip}
      >
        <Map mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" />
      </DeckGL>

      <div className="absolute bottom-8 left-8 bg-white p-4 rounded-lg shadow-lg">
        <label className="block text-sm font-medium">
          Time: {timeFilter[0]}:00 - {timeFilter[1]}:00
        </label>
        <input
          type="range" min={0} max={24} step={1}
          value={timeFilter[1]}
          onChange={(e) => setTimeFilter([timeFilter[0], Number(e.target.value)])}
          className="w-64"
        />
      </div>
    </div>
  );
}
```

### deck.gl Binary Data — Skip JSON Parsing

```jsx
async function loadBinaryData(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const float32 = new Float32Array(buffer);
  const count = float32.length / 3;

  return {
    length: count,
    attributes: {
      getPosition: { value: float32, size: 3, stride: 12, offset: 0 },
    },
  };
}

// Usage with ScatterplotLayer
const layer = new ScatterplotLayer({
  id: 'binary-points',
  data: await loadBinaryData('/api/points.bin'),
  getRadius: 50,
  getFillColor: [255, 140, 0],
});
```

### Resium — CesiumJS for React

```jsx
import { Viewer, Entity, CameraFlyTo, Cesium3DTileset, ImageryLayer } from 'resium';
import {
  Cartesian3, Color, Ion, UrlTemplateImageryProvider,
  Cesium3DTileStyle, HeadingPitchRange
} from 'cesium';
import 'cesium/Build/Cesium/Widgets/widgets.css';

Ion.defaultAccessToken = process.env.NEXT_PUBLIC_CESIUM_TOKEN;

function Globe3D({ buildings3dUrl }) {
  return (
    <Viewer
      full
      timeline={false}
      animation={false}
      homeButton={false}
      baseLayerPicker={false}
    >
      <ImageryLayer
        imageryProvider={new UrlTemplateImageryProvider({
          url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
          maximumLevel: 19,
        })}
      />

      {buildings3dUrl && (
        <Cesium3DTileset
          url={buildings3dUrl}
          style={new Cesium3DTileStyle({
            color: {
              conditions: [
                ['${height} > 100', 'color("red", 0.8)'],
                ['${height} > 50', 'color("orange", 0.8)'],
                ['true', 'color("white", 0.8)'],
              ],
            },
          })}
        />
      )}

      <Entity
        name="Beijing"
        position={Cartesian3.fromDegrees(116.4, 39.9, 100)}
        point={{ pixelSize: 12, color: Color.RED, outlineColor: Color.WHITE, outlineWidth: 2 }}
      />

      <CameraFlyTo
        destination={Cartesian3.fromDegrees(116.4, 39.9, 15000)}
        duration={3}
      />
    </Viewer>
  );
}
```

### React State Management — Zustand Store for Map

```jsx
// stores/mapStore.js
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

export const useMapStore = create(
  subscribeWithSelector((set, get) => ({
    viewState: { longitude: 116.4, latitude: 39.9, zoom: 10, pitch: 0, bearing: 0 },
    setViewState: (vs) => set({ viewState: vs }),

    layers: {
      buildings: { visible: true, opacity: 0.8 },
      roads: { visible: true, opacity: 1.0 },
      poi: { visible: false, opacity: 0.6 },
    },
    toggleLayer: (id) =>
      set((state) => ({
        layers: { ...state.layers, [id]: { ...state.layers[id], visible: !state.layers[id].visible } },
      })),
    setLayerOpacity: (id, opacity) =>
      set((state) => ({
        layers: { ...state.layers, [id]: { ...state.layers[id], opacity } },
      })),

    selectedFeatures: [],
    setSelectedFeatures: (features) => set({ selectedFeatures: features }),
    clearSelection: () => set({ selectedFeatures: [] }),

    drawMode: null,
    drawnFeatures: [],
    setDrawMode: (mode) => set({ drawMode: mode }),
    addDrawnFeature: (feature) =>
      set((state) => ({ drawnFeatures: [...state.drawnFeatures, feature] })),
  }))
);
```

```jsx
// Using the store
function LayerControl() {
  const layers = useMapStore((s) => s.layers);
  const toggleLayer = useMapStore((s) => s.toggleLayer);
  const setLayerOpacity = useMapStore((s) => s.setLayerOpacity);

  return (
    <div className="bg-white p-3 rounded-lg shadow">
      {Object.entries(layers).map(([id, config]) => (
        <div key={id} className="flex items-center gap-2 mb-2">
          <input type="checkbox" checked={config.visible} onChange={() => toggleLayer(id)} />
          <span className="flex-1 capitalize">{id}</span>
          <input
            type="range" min={0} max={1} step={0.1}
            value={config.opacity}
            onChange={(e) => setLayerOpacity(id, Number(e.target.value))}
            className="w-20"
          />
        </div>
      ))}
    </div>
  );
}
```

### React Query for Spatial Data

```jsx
import { useQuery } from '@tanstack/react-query';
import { useMapStore } from '@/stores/mapStore';
import { useMemo } from 'react';

function useSpatialFeatures(layerId) {
  const viewState = useMapStore((s) => s.viewState);

  const bbox = useMemo(() => {
    const { longitude, latitude, zoom } = viewState;
    const range = 180 / Math.pow(2, zoom);
    return [longitude - range, latitude - range, longitude + range, latitude + range].join(',');
  }, [Math.round(viewState.longitude * 100), Math.round(viewState.latitude * 100), Math.round(viewState.zoom)]);

  return useQuery({
    queryKey: ['features', layerId, bbox],
    queryFn: async () => {
      const res = await fetch(`/api/features/${layerId}?bbox=${bbox}`);
      if (!res.ok) throw new Error('Failed to fetch');
      return res.json();
    },
    staleTime: 5 * 60 * 1000,
    placeholderData: (prev) => prev,
    refetchOnWindowFocus: false,
  });
}
```

### URL State Synchronization

```jsx
// hooks/useMapUrlSync.js
import { useSearchParams, useRouter } from 'next/navigation';
import { useCallback, useEffect } from 'react';

export function useMapUrlSync(viewState, setViewState) {
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const lat = searchParams.get('lat');
    const lng = searchParams.get('lng');
    const z = searchParams.get('z');
    if (lat && lng && z) {
      setViewState({
        latitude: parseFloat(lat), longitude: parseFloat(lng),
        zoom: parseFloat(z), pitch: 0, bearing: 0,
      });
    }
  }, []);

  const syncToUrl = useCallback(
    debounce((vs) => {
      const params = new URLSearchParams({
        lat: vs.latitude.toFixed(5), lng: vs.longitude.toFixed(5), z: vs.zoom.toFixed(1),
      });
      router.replace(`?${params.toString()}`, { scroll: false });
    }, 500),
    [router]
  );

  return syncToUrl;
}

function debounce(fn, ms) {
  let timer;
  return (...args) => { clearTimeout(timer); timer = setTimeout(() => fn(...args), ms); };
}
```

### DrawTools Component

```jsx
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import { useControl } from 'react-map-gl/maplibre';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';

function DrawControl({ position = 'top-left', onUpdate, onDelete }) {
  useControl(
    () => new MapboxDraw({
      displayControlsDefault: false,
      controls: { polygon: true, line_string: true, point: true, trash: true },
      defaultMode: 'simple_select',
    }),
    ({ map }) => {
      map.on('draw.create', onUpdate);
      map.on('draw.update', onUpdate);
      map.on('draw.delete', onDelete);
    },
    ({ map }) => {
      map.off('draw.create', onUpdate);
      map.off('draw.update', onUpdate);
      map.off('draw.delete', onDelete);
    },
    { position }
  );
  return null;
}

function DrawableMap() {
  const [features, setFeatures] = useState({ type: 'FeatureCollection', features: [] });

  const onUpdate = useCallback((e) => {
    setFeatures((prev) => ({ ...prev, features: [...prev.features, ...e.features] }));
  }, []);

  const onDelete = useCallback((e) => {
    const ids = new Set(e.features.map((f) => f.id));
    setFeatures((prev) => ({ ...prev, features: prev.features.filter((f) => !ids.has(f.id)) }));
  }, []);

  return (
    <Map mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json">
      <DrawControl onUpdate={onUpdate} onDelete={onDelete} />
    </Map>
  );
}
```

---

## Vue 3 Ecosystem

### vue-maplibre-gl — Composables + MapLibre

```vue
<template>
  <MglMap
    :mapStyle="styleUrl"
    :center="[116.4, 39.9]"
    :zoom="10"
    @map:load="onMapLoad"
    @map:click="onMapClick"
    style="width: 100%; height: 100vh"
  >
    <MglNavigationControl position="top-right" />
    <MglScaleControl position="bottom-left" />
    <MglGeolocateControl position="top-right" />

    <MglGeoJsonSource source-id="zones" :data="geojsonData">
      <MglFillLayer
        layer-id="zones-fill"
        :paint="{
          'fill-color': ['interpolate', ['linear'], ['get', 'value'], 0, '#f7fbff', 100, '#08306b'],
          'fill-opacity': 0.7
        }"
      />
      <MglLineLayer
        layer-id="zones-line"
        :paint="{ 'line-color': '#333', 'line-width': 1 }"
      />
    </MglGeoJsonSource>

    <MglMarker
      v-for="poi in pois"
      :key="poi.id"
      :lngLat="[poi.lng, poi.lat]"
      :color="poi.color"
      @click="selectPoi(poi)"
    />

    <MglPopup
      v-if="selectedPoi"
      :lngLat="[selectedPoi.lng, selectedPoi.lat]"
      :closeOnClick="false"
      @close="selectedPoi = null"
    >
      <div class="p-2">
        <h3 class="font-bold">{{ selectedPoi.name }}</h3>
        <p>{{ selectedPoi.description }}</p>
      </div>
    </MglPopup>
  </MglMap>
</template>

<script setup>
import { ref, onMounted } from 'vue';
import {
  MglMap, MglNavigationControl, MglScaleControl, MglGeolocateControl,
  MglGeoJsonSource, MglFillLayer, MglLineLayer, MglMarker, MglPopup
} from 'vue-maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';

const styleUrl = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';
const geojsonData = ref(null);
const pois = ref([]);
const selectedPoi = ref(null);
let mapInstance = null;

const onMapLoad = (map) => { mapInstance = map; loadData(); };

const onMapClick = (e) => {
  const features = mapInstance.queryRenderedFeatures(e.point, { layers: ['zones-fill'] });
  if (features.length > 0) console.log('Clicked:', features[0].properties);
};

const selectPoi = (poi) => { selectedPoi.value = poi; };

async function loadData() {
  const res = await fetch('/api/zones.geojson');
  geojsonData.value = await res.json();
}
</script>
```

### Vue 3 + Pinia Map Store

```js
// stores/map.js
import { defineStore } from 'pinia';
import { ref, computed } from 'vue';

export const useMapStore = defineStore('map', () => {
  const center = ref([116.4, 39.9]);
  const zoom = ref(10);
  const pitch = ref(0);
  const bearing = ref(0);

  const layerVisibility = ref({ buildings: true, roads: true, poi: false });
  const toggleLayer = (id) => { layerVisibility.value[id] = !layerVisibility.value[id]; };

  const selectedFeatures = ref([]);
  const hasSelection = computed(() => selectedFeatures.value.length > 0);
  const clearSelection = () => { selectedFeatures.value = []; };

  const bbox = computed(() => {
    const range = 180 / Math.pow(2, zoom.value);
    return [center.value[0] - range, center.value[1] - range,
            center.value[0] + range, center.value[1] + range];
  });

  return { center, zoom, pitch, bearing, layerVisibility, toggleLayer,
           selectedFeatures, hasSelection, clearSelection, bbox };
});
```

### Nuxt 3 Integration

```vue
<!-- pages/map.vue -->
<template>
  <ClientOnly>
    <MapView :data="data" />
    <template #fallback>
      <div class="w-full h-screen flex items-center justify-center">Loading map...</div>
    </template>
  </ClientOnly>
</template>

<script setup>
const { data } = await useFetch('/api/features', { query: { bbox: '116.0,39.5,117.0,40.5' } });
</script>
```

```ts
// server/api/features.ts
import pg from 'pg';
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

export default defineEventHandler(async (event) => {
  const query = getQuery(event);
  const bbox = (query.bbox as string)?.split(',').map(Number);
  if (!bbox || bbox.length !== 4) throw createError({ statusCode: 400, message: 'Invalid bbox' });

  const result = await pool.query(
    `SELECT jsonb_build_object(
       'type', 'FeatureCollection',
       'features', COALESCE(jsonb_agg(jsonb_build_object(
         'type', 'Feature',
         'properties', jsonb_build_object('id', id, 'name', name),
         'geometry', ST_AsGeoJSON(geom)::jsonb
       )), '[]'::jsonb)) AS geojson
     FROM features WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326) LIMIT 5000`,
    bbox
  );
  return result.rows[0].geojson;
});
```

---

## Svelte / SvelteKit

### svelte-maplibre — Store-Based Reactivity

```svelte
<script>
  import {
    MapLibre, NavigationControl, ScaleControl, GeolocateControl,
    GeoJSON, FillLayer, LineLayer, Popup
  } from 'svelte-maplibre';
  import { writable } from 'svelte/store';

  let geojsonData = null;
  let selectedFeature = null;

  const layers = writable({ zones: true, buildings: true, roads: false });

  function handleClick(e) {
    if (e.detail.features?.length > 0) selectedFeature = e.detail.features[0];
  }

  async function loadData() {
    const res = await fetch('/api/zones.geojson');
    geojsonData = await res.json();
  }
  loadData();
</script>

<div class="relative w-full h-screen">
  <MapLibre
    style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
    center={[116.4, 39.9]}
    zoom={10}
    class="w-full h-full"
  >
    <NavigationControl position="top-right" />
    <ScaleControl position="bottom-left" />
    <GeolocateControl position="top-right" />

    {#if geojsonData && $layers.zones}
      <GeoJSON data={geojsonData} id="zones">
        <FillLayer
          paint={{
            'fill-color': ['interpolate', ['linear'], ['get', 'value'], 0, '#f7fbff', 100, '#08306b'],
            'fill-opacity': 0.6
          }}
          on:click={handleClick}
        />
        <LineLayer paint={{ 'line-color': '#333', 'line-width': 0.5 }} />
      </GeoJSON>
    {/if}

    {#if selectedFeature}
      <Popup lngLat={selectedFeature.geometry.coordinates[0]} on:close={() => selectedFeature = null}>
        <div class="p-2">
          <h3 class="font-bold">{selectedFeature.properties.name}</h3>
          <p>Value: {selectedFeature.properties.value}</p>
        </div>
      </Popup>
    {/if}
  </MapLibre>

  <div class="absolute top-4 left-4 bg-white p-3 rounded-lg shadow">
    {#each Object.entries($layers) as [id, visible]}
      <label class="flex items-center gap-2 mb-1">
        <input type="checkbox" bind:checked={$layers[id]} />
        <span class="capitalize">{id}</span>
      </label>
    {/each}
  </div>
</div>
```

### SvelteKit Load Functions

```ts
// src/routes/map/+page.server.ts
import type { PageServerLoad } from './$types';
import pg from 'pg';

const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

export const load: PageServerLoad = async ({ url }) => {
  const bbox = url.searchParams.get('bbox')?.split(',').map(Number) ?? [116.0, 39.5, 117.0, 40.5];
  const result = await pool.query(
    `SELECT jsonb_build_object('type', 'FeatureCollection',
       'features', COALESCE(jsonb_agg(jsonb_build_object(
         'type', 'Feature',
         'properties', to_jsonb(t.*) - 'geom',
         'geometry', ST_AsGeoJSON(geom)::jsonb)), '[]'::jsonb)) AS geojson
     FROM features t WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326) LIMIT 5000`, bbox);
  return { geojson: result.rows[0].geojson };
};
```

### Raw MapLibre with Svelte Action

```svelte
<script>
  import maplibregl from 'maplibre-gl';
  import 'maplibre-gl/dist/maplibre-gl.css';

  function mapAction(node, { center, zoom, style }) {
    const map = new maplibregl.Map({ container: node, style, center, zoom });
    map.addControl(new maplibregl.NavigationControl(), 'top-right');

    map.on('load', () => {
      map.addSource('data', { type: 'geojson', data: { type: 'FeatureCollection', features: [] } });
      map.addLayer({
        id: 'data-fill', type: 'fill', source: 'data',
        paint: { 'fill-color': '#088', 'fill-opacity': 0.5 },
      });
    });

    return {
      update({ data }) {
        if (map.isStyleLoaded() && data) map.getSource('data')?.setData(data);
      },
      destroy() { map.remove(); },
    };
  }
</script>

<div
  use:mapAction={{ center: [116.4, 39.9], zoom: 10,
    style: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json' }}
  class="w-full h-screen"
/>
```

---

## Web Components — Framework-Agnostic

### Lit + MapLibre

```js
import { LitElement, html, css } from 'lit';
import maplibregl from 'maplibre-gl';

class GisMap extends LitElement {
  static properties = {
    center: { type: Array },
    zoom: { type: Number },
    styleUrl: { type: String, attribute: 'style-url' },
    geojsonUrl: { type: String, attribute: 'geojson-url' },
  };

  static styles = css`
    :host { display: block; width: 100%; height: 100%; }
    #map { width: 100%; height: 100%; }
  `;

  constructor() {
    super();
    this.center = [116.4, 39.9];
    this.zoom = 10;
    this.styleUrl = 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json';
  }

  firstUpdated() {
    this.map = new maplibregl.Map({
      container: this.shadowRoot.getElementById('map'),
      style: this.styleUrl,
      center: this.center,
      zoom: this.zoom,
    });

    this.map.addControl(new maplibregl.NavigationControl());

    if (this.geojsonUrl) {
      this.map.on('load', () => this.loadGeoJSON());
    }
  }

  async loadGeoJSON() {
    const res = await fetch(this.geojsonUrl);
    const data = await res.json();
    this.map.addSource('data', { type: 'geojson', data });
    this.map.addLayer({
      id: 'data-fill', type: 'fill', source: 'data',
      paint: { 'fill-color': '#088', 'fill-opacity': 0.5 },
    });
    this.map.on('click', 'data-fill', (e) => {
      this.dispatchEvent(new CustomEvent('feature-click', {
        detail: e.features[0], bubbles: true, composed: true,
      }));
    });
  }

  disconnectedCallback() { super.disconnectedCallback(); this.map?.remove(); }

  render() {
    return html`
      <link rel="stylesheet" href="https://unpkg.com/maplibre-gl/dist/maplibre-gl.css" />
      <div id="map"></div>
    `;
  }
}
customElements.define('gis-map', GisMap);
```

```html
<!-- Usage in any HTML page or framework -->
<gis-map center="[116.4, 39.9]" zoom="10" geojson-url="/api/zones.geojson"></gis-map>
```

---

## Cross-Framework Patterns

### State Management Matrix

| State Type | Local | Global | URL | Server |
|-----------|-------|--------|-----|--------|
| View state (center/zoom/pitch) | Simple maps | Multi-panel | Shareable views | N/A |
| Layer visibility | Single map | Dashboard | Permalink | N/A |
| Selected features | Single component | Inspector panels | Permalink | N/A |
| Feature data | N/A | Cache (SWR/Query) | N/A | PostGIS/API |
| Drawn geometries | Simple draw | Collab edit | Export | Save to DB |
| Filters/search | Simple apps | Complex dashboards | Shareable URLs | N/A |

### Map Lifecycle Rules

```
Mount:    Create map → add controls → load data → bind events
Resize:   Window/container resize → map.resize()
Update:   New data → update sources (NOT recreate map)
Unmount:  Remove event listeners → map.remove()
```

1. **Never put map instance in reactive state** (React state, Vue ref, Svelte store)
2. **Use refs/actions** to hold map instance
3. **Debounce move events** before updating global state
4. **Call map.resize()** when container dimensions change

### Responsive Map Layout

```css
.map-container {
  width: 100%;
  height: 100dvh;
}

@media (min-width: 768px) {
  .map-layout {
    display: grid;
    grid-template-columns: 360px 1fr;
    height: 100dvh;
  }
  .sidebar { overflow-y: auto; }
}

@media (max-width: 768px) {
  .maplibregl-ctrl-top-right { top: 60px; }
  .maplibregl-ctrl-bottom-left { bottom: 80px; }
}
```

### Accessibility for Maps

```jsx
function AccessibleMap() {
  const { current: map } = useMap();
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const features = useMapStore((s) => s.visibleFeatures);

  const handleKeyDown = (e) => {
    switch (e.key) {
      case 'ArrowRight': setFocusedIndex((i) => Math.min(i + 1, features.length - 1)); break;
      case 'ArrowLeft': setFocusedIndex((i) => Math.max(i - 1, 0)); break;
      case 'Enter':
        if (focusedIndex >= 0) {
          map?.flyTo({ center: features[focusedIndex].geometry.coordinates, zoom: 14 });
        }
        break;
    }
  };

  return (
    <div role="application" aria-label="Interactive map" tabIndex={0} onKeyDown={handleKeyDown}>
      <Map /* ... */ />
      <div className="sr-only" aria-live="polite">
        {focusedIndex >= 0 &&
          `Feature ${focusedIndex + 1} of ${features.length}: ${features[focusedIndex].properties.name}`}
      </div>
    </div>
  );
}
```

### Testing Map Components

```js
// __tests__/MapView.test.jsx
import { render, screen } from '@testing-library/react';
import { vi, describe, it, expect } from 'vitest';

vi.mock('maplibre-gl', () => ({
  default: {
    Map: vi.fn().mockImplementation(() => ({
      on: vi.fn(), off: vi.fn(), remove: vi.fn(), resize: vi.fn(),
      addControl: vi.fn(), addSource: vi.fn(), addLayer: vi.fn(),
      getSource: vi.fn(() => ({ setData: vi.fn() })),
      flyTo: vi.fn(), getCanvas: vi.fn(() => ({ style: {} })),
      isStyleLoaded: vi.fn(() => true),
    })),
    NavigationControl: vi.fn(),
    ScaleControl: vi.fn(),
  },
}));

vi.mock('react-map-gl/maplibre', () => ({
  default: ({ children, ...props }) => <div data-testid="map" {...props}>{children}</div>,
  Source: ({ children }) => <div data-testid="source">{children}</div>,
  Layer: (props) => <div data-testid={`layer-${props.id}`} />,
  NavigationControl: () => <div data-testid="nav-control" />,
  useMap: () => ({ current: { flyTo: vi.fn() } }),
}));

describe('MapView', () => {
  it('renders map with controls', () => {
    render(<MapView />);
    expect(screen.getByTestId('map')).toBeDefined();
    expect(screen.getByTestId('nav-control')).toBeDefined();
  });

  it('renders GeoJSON data when provided', () => {
    const geojson = { type: 'FeatureCollection', features: [
      { type: 'Feature', properties: { name: 'Test' }, geometry: { type: 'Point', coordinates: [116.4, 39.9] } }
    ]};
    render(<MapView geojsonData={geojson} />);
    expect(screen.getByTestId('source')).toBeDefined();
  });
});
```

### TypeScript Patterns for GeoJSON

```ts
// types/geo.ts
import type { Feature, FeatureCollection, Point, Polygon, LineString } from 'geojson';

interface BuildingProperties {
  id: number;
  name: string;
  height: number;
  type: 'residential' | 'commercial' | 'industrial';
  yearBuilt: number;
}

type BuildingFeature = Feature<Polygon, BuildingProperties>;
type BuildingCollection = FeatureCollection<Polygon, BuildingProperties>;

type LayerConfig =
  | { type: 'fill'; paint: { 'fill-color': string; 'fill-opacity': number } }
  | { type: 'line'; paint: { 'line-color': string; 'line-width': number } }
  | { type: 'circle'; paint: { 'circle-radius': number; 'circle-color': string } }
  | { type: 'symbol'; layout: { 'icon-image': string; 'text-field': string } };

type BBox = [minLng: number, minLat: number, maxLng: number, maxLat: number];

interface SpatialQueryParams {
  bbox: BBox;
  limit?: number;
  offset?: number;
}

async function fetchFeatures<P>(
  endpoint: string, params: SpatialQueryParams
): Promise<FeatureCollection<Point | Polygon | LineString, P>> {
  const qs = new URLSearchParams({
    bbox: params.bbox.join(','),
    limit: String(params.limit ?? 1000),
  }).toString();
  const res = await fetch(`${endpoint}?${qs}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}
```

---

## Framework Comparison (2025)

| Feature | React | Vue 3 | Svelte | Solid | Vanilla |
|---------|-------|-------|--------|-------|---------|
| MapLibre wrapper | react-map-gl v7 | vue-maplibre-gl | svelte-maplibre | -- | Direct API |
| Leaflet wrapper | react-leaflet v4 | vue-leaflet | -- | -- | Direct API |
| CesiumJS wrapper | resium | vue-cesium | -- | -- | Direct API |
| deck.gl | @deck.gl/react | -- | -- | -- | @deck.gl/core |
| SSR framework | Next.js 15 | Nuxt 3 | SvelteKit | SolidStart | -- |
| Bundle overhead | ~45 KB | ~35 KB | ~5 KB | ~8 KB | 0 KB |
| Reactivity model | Re-render | Proxy | Compile-time | Signals | Manual |
| Best for | Large teams | Medium teams | Small-medium | Performance | Max control |

### Library Links

- **React:** [react-map-gl](https://visgl.github.io/react-map-gl/) | [react-leaflet](https://react-leaflet.js.org/) | [resium](https://resium.reearth.io/) | [deck.gl](https://deck.gl/)
- **Vue:** [vue-maplibre-gl](https://indoorequal.github.io/vue-maplibre-gl/) | [vue-leaflet](https://github.com/vue-leaflet/vue-leaflet) | [vue-cesium](https://zouyaoji.top/vue-cesium/)
- **Svelte:** [svelte-maplibre](https://github.com/dimfeld/svelte-maplibre)

---

## Performance Tips

| Tip | Impact | Framework |
|-----|--------|-----------|
| Memoize layer definitions | Prevents style re-parsing | React (useMemo) |
| Use `useDeferredValue` for large GeoJSON | Keeps UI responsive | React 18+ |
| Debounce viewState updates to global store | Reduces re-renders | All |
| Use `shallowRef` for GeoJSON data | Avoids deep reactivity | Vue 3 |
| Keep map instance outside reactive state | Prevents unnecessary tracking | All |
| Lazy-load map component | Reduces initial bundle | All (dynamic import) |
| Use vector tiles instead of GeoJSON for >10K features | GPU-native rendering | All |
| Binary data transfer with deck.gl | Skip JSON parsing | React + deck.gl |
| Web Worker for heavy spatial computations | Non-blocking UI | All |
| Service Worker for tile caching | Offline + faster reload | All (PWA) |
