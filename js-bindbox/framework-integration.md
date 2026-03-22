# Framework Integration

> Data validated: 2026-03-21

## 30-Second Decision

| Framework | Wrapper | Startup time | Ecosystem |
|-----------|---------|-------------|-----------|
| Svelte | svelte-maplibre | 10 min | Small but excellent DX |
| React | react-map-gl | 15 min | Largest, most battle-tested |
| Vue | vue-maplibre-gl | 15 min | Medium, fragmented |
| None / vanilla | MapLibre GL JS directly | 10 min | N/A |
| React + 3D Globe | Resium (CesiumJS) | Hours | Small, **single maintainer** |

---

## Detailed Guide (fastest DX first)

### 1. svelte-maplibre (Svelte/SvelteKit)

Best developer experience for geo apps. Svelte's reactivity maps naturally onto MapLibre's imperative API.

**Quick facts:** Zero framework runtime overhead | ~486 GitHub stars | Production-readiness: 3/5

```svelte
<script>
  import { MapLibre, GeoJSON, FillLayer } from 'svelte-maplibre';
</script>

<MapLibre
  style="https://demotiles.maplibre.org/style.json"
  center={[116.4, 39.9]}
  zoom={10}
>
  <GeoJSON data={geojsonData}>
    <FillLayer paint={{ 'fill-color': '#088', 'fill-opacity': 0.6 }} />
  </GeoJSON>
</MapLibre>
```

**Small project:** Best DX choice -- least boilerplate, Svelte reactivity "just works" with map state, no runtime overhead.

**Key caveats:**
- Smallest ecosystem (fewest components, examples, SO answers)
- Single maintainer (dimfeld) -- bus factor risk
- Svelte market share limits hiring pool
- SvelteKit SSR: must use `onMount` or `{#if browser}` for map init
- Svelte 5 runes migration may require wrapper updates

---

### 2. react-map-gl (React/Next.js)

The canonical React wrapper for MapLibre. Largest ecosystem, most battle-tested, maintained by Uber/vis.gl.

**Quick facts:** ~30KB on top of MapLibre | ~8.4K GitHub stars | ~1.13M npm/week | 5/5 production-readiness

```tsx
// Uncontrolled mode (simplest -- start here)
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

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
// Next.js App Router: MUST disable SSR
import dynamic from 'next/dynamic';
const MapComponent = dynamic(() => import('./Map'), { ssr: false });
// Or add 'use client' directive at top of the map component file
```

**Small project:** Safe choice -- largest ecosystem, most tutorials, start with uncontrolled mode (5 lines) and add complexity as needed.

**Key caveats:**
- Controlled mode (`onMove` + `useState`) causes frequent re-renders. Use uncontrolled mode unless you need URL sync.
- Map instance access requires refs (`useMap()` hook), bypassing React's declarative model
- Next.js App Router: must use `dynamic(() => ..., { ssr: false })` or `'use client'`
- Must clean up map in `useEffect` cleanup or WebGL contexts leak (limit: 8-16)
- **Anti-pattern:** Creating Source/Layer components dynamically in render. Stabilize keys.
- **Anti-pattern:** Using `useState` for viewport when not needed. Use `initialViewState`.

---

### 3. vue-maplibre-gl (Vue 3/Nuxt 3)

Vue wrapper for MapLibre. Vue reactivity works well with map state, but ecosystem is fragmented.

```vue
<template>
  <MglMap
    :mapStyle="'https://demotiles.maplibre.org/style.json'"
    :center="[116.4, 39.9]"
    :zoom="10"
  >
    <MglGeoJsonSource :data="geojsonData">
      <MglFillLayer :paint="{ 'fill-color': '#088', 'fill-opacity': 0.6 }" />
    </MglGeoJsonSource>
  </MglMap>
</template>

<script setup>
import { MglMap, MglGeoJsonSource, MglFillLayer } from 'vue-maplibre-gl';
</script>
```

**Small project:** Works well for Vue 3 projects. Clean component syntax. Nuxt 3 SSR requires `<ClientOnly>`.

**Key caveats:**
- Fragmented ecosystem: multiple competing wrappers (v-mapbox, vue-maplibre-gl, etc.)
- Smaller than react-map-gl ecosystem
- Vue reactivity can over-trigger map updates; use `shallowRef()` for large GeoJSON
- **Alternative:** Consider MapLibre GL JS directly with Vue 3 Composition API for maximum control

---

### 4. Vanilla JS / Web Components

Sometimes no framework is the right framework. For simple map widgets, embeds, or prototypes.

```javascript
import maplibregl from 'maplibre-gl';
const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [116.4, 39.9], zoom: 10
});
map.on('load', () => {
  map.addSource('data', { type: 'geojson', data: geojson });
  map.addLayer({ id: 'fill', type: 'fill', source: 'data',
    paint: { 'fill-color': '#088', 'fill-opacity': 0.6 } });
});
```

---

## Low Priority

### Resium (CesiumJS for React)

React wrapper for CesiumJS. Does NOT simplify Cesium's complexity -- just changes syntax to JSX.

**Single maintainer risk:** Maintained by one developer (reearth). If they step away, the wrapper is orphaned and CesiumJS upgrades (security patches, WebGL fixes) cannot be applied through Resium. For production-critical 3D globe apps, consider using CesiumJS directly with a custom React wrapper as a fallback plan.

**Key caveats:**
- CesiumJS (~4MB) + React + Resium = very heavy bundle
- Cesium's lifecycle doesn't fit React's render model; memory leaks common
- Build config nightmare (static asset copying per bundler)
- **Anti-pattern:** Using Resium without deep CesiumJS knowledge

---

## State Management for Geo Apps

### Zustand (React)

Lightweight, minimal boilerplate, works well with map viewport state.

```typescript
import { create } from 'zustand';
interface GeoStore {
  viewport: { longitude: number; latitude: number; zoom: number };
  selectedFeature: GeoJSON.Feature | null;
  setViewport: (v: Partial<GeoStore['viewport']>) => void;
  selectFeature: (f: GeoJSON.Feature | null) => void;
}
const useGeoStore = create<GeoStore>((set) => ({
  viewport: { longitude: 116.4, latitude: 39.9, zoom: 10 },
  selectedFeature: null,
  setViewport: (v) => set((state) => ({ viewport: { ...state.viewport, ...v } })),
  selectFeature: (f) => set({ selectedFeature: f })
}));
```

### Pinia (Vue)

Canonical Vue state management. Works well with geo state patterns.

---

## SSR Cheat Sheet

| Framework | Pattern |
|-----------|---------|
| **Next.js (Pages)** | `dynamic(() => import('./Map'), { ssr: false })` |
| **Next.js (App)** | `'use client'` directive on map component |
| **Nuxt 3** | `<ClientOnly><MapComponent /></ClientOnly>` |
| **SvelteKit** | `{#if browser}` or `onMount` with dynamic import |

For enterprise frontend architecture patterns -> [web-dev/frontend-integration.md](../web-dev/frontend-integration.md)
