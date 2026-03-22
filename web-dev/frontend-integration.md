# Frontend Integration -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**React:** react-map-gl + MapLibre GL JS. **Vue 3:** vue-maplibre-gl. **Svelte:** svelte-maplibre. **Large data (10M+ features):** deck.gl overlay on MapLibre. **3D globe:** evaluate MapLibre Globe view first; only use Resium/CesiumJS if you need terrain, 3D Tiles, or sub-meter accuracy. **Resium has single-maintainer risk -- plan a fallback.**

---

## Tier 1 -- Production First Choices

---

### react-map-gl + MapLibre GL JS

The most popular combination for production web mapping. Maintained by Uber's vis.gl team. Controlled/uncontrolled map components with hooks API.

**Why Tier 1:** Uber-maintained, 5/5 production-readiness (Uber, Foursquare, many production apps), TypeScript support, Next.js compatible, largest ecosystem of examples and community components. Controlled mode enables URL-synced viewports and multi-map coordination.

```jsx
'use client';
import { useCallback, useState } from 'react';
import Map, {
  Source, Layer, NavigationControl, ScaleControl
} from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';

export default function MapView({ geojsonData, onFeatureClick }) {
  const [viewState, setViewState] = useState({
    longitude: 116.4, latitude: 39.9, zoom: 10,
  });

  const onClick = useCallback((event) => {
    const feature = event.features?.[0];
    if (feature) onFeatureClick?.(feature);
  }, [onFeatureClick]);

  return (
    <Map
      {...viewState}
      onMove={(evt) => setViewState(evt.viewState)}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
      interactiveLayerIds={['buildings-fill']}
      onClick={onClick}
    >
      <NavigationControl position="top-right" />
      <ScaleControl />
      {geojsonData && (
        <Source id="buildings" type="geojson" data={geojsonData}>
          <Layer
            id="buildings-fill"
            type="fill"
            paint={{ 'fill-color': '#088', 'fill-opacity': 0.6 }}
          />
        </Source>
      )}
    </Map>
  );
}
```

**Caveats:**
- **SSR incompatibility.** MapLibre GL JS requires DOM/WebGL. Must use `dynamic(() => import(...), { ssr: false })` in Next.js or `'use client'` boundary. Forgetting this causes build failures.
- **Bundle size.** MapLibre GL JS is ~200KB gzipped. With react-map-gl, total map bundle is ~250KB. Code splitting is essential.
- **Memory leaks.** Failing to clean up map instances (missing `useEffect` cleanup) causes WebGL context leaks. After 8-16 leaked contexts, the browser refuses to create new ones -- black screens with no error message.
- **Controlled mode re-render storms.** In controlled mode, every `onMove` callback triggers a React state update + re-render. On older devices, this creates visible jank. Use `useCallback` and consider uncontrolled mode (just `initialViewState`).
- **Source/Layer ordering.** React children order determines layer paint order. Reordering components changes visual stacking.
- **Anti-pattern: Creating Source/Layer components dynamically in render.** Causes MapLibre to constantly add/remove sources, causing flicker. Stabilize component keys and use conditional visibility instead.
- **Anti-pattern: Using `useState` for map viewport when not needed.** Use uncontrolled mode (just `initialViewState`) unless you need URL sync or external viewport control.

---

### deck.gl -- Large-Scale Data Visualization

GPU-accelerated visualization for massive datasets. 10M+ points at 60fps. **Use deck.gl when MapLibre's built-in layers aren't enough for your data volume.**

**Why Tier 1:** The only option for enterprise analytics dashboards with >10K features at 60fps. Used by Uber, Foursquare, Google. WebGPU migration is in progress.

```jsx
import { DeckGL } from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { MapboxOverlay } from '@deck.gl/mapbox';

// Binary data mode for maximum performance (10x faster than JSON)
const layer = new ScatterplotLayer({
  id: 'points',
  data: {
    length: 1000000,
    attributes: {
      getPosition: new Float32Array(positionBuffer),
      getFillColor: new Uint8Array(colorBuffer),
    },
  },
  getRadius: 50,
  // CRITICAL: set updateTriggers when data content changes
  // but the array reference stays the same
  updateTriggers: { getPosition: [dataVersion] },
});
```

**Caveats:**
- **Not a map library.** deck.gl renders data layers, not basemaps. You need MapLibre underneath. This dual-library architecture adds complexity.
- **WebGL debugging is painful.** When something goes wrong (black screen, wrong colors), error messages are unhelpful.
- **Binary data mode is necessary for scale.** To render 1M+ points at 60fps, you MUST use typed arrays (Float32Array). JSON/GeoJSON tops out at ~100K before frame drops.
- **Layer lifecycle in React is tricky.** Creating layer instances inside `useMemo` with correct dependency arrays is error-prone.
- **Bundle size.** `@deck.gl/core` + layers is 200KB+ gzipped.
- **Memory leaks.** GPU buffers are not automatically freed when React components unmount. Must call `deck.finalize()`.
- **Anti-pattern: Using deck.gl for simple maps with <1K features.** The setup overhead is not justified. Use MapLibre's built-in layers.
- **Anti-pattern: Ignoring `updateTriggers`.** deck.gl uses shallow comparison. If your data array reference doesn't change but contents do, the layer won't update.

---

### vue-maplibre-gl -- Vue 3 Integration

The standard MapLibre wrapper for Vue 3 and Nuxt 3 projects.

**Why Tier 1:** If your enterprise is committed to Vue, this is the right choice. Production-readiness is 3/5 -- smaller community than React alternative but functionally complete.

**Caveats:**
- **Smaller ecosystem than react-map-gl.** Fewer examples, fewer community components.
- **Vue reactivity can over-trigger map updates.** Deep reactive objects passed as layer paint/layout props cause unnecessary style recalculations. Use `shallowRef()` for large GeoJSON objects.
- **Nuxt 3 SSR requires client-only wrapper.** Same SSR issues as Next.js.
- **Anti-pattern: Using reactive GeoJSON sources with large datasets.** Vue's reactivity proxy adds overhead. Use `shallowRef()` for large GeoJSON objects.

---

### svelte-maplibre -- Svelte Integration

Excellent reactive integration with MapLibre for Svelte and SvelteKit.

**Why Tier 1:** Best DX of the three framework options. Svelte's reactivity model is a natural fit for map state.

**Caveats:**
- **Smallest ecosystem.** Fewest third-party components and examples of the three framework options.
- **Svelte's market share limits hiring.** Finding developers experienced with both Svelte and GIS is harder than React + GIS.
- **SvelteKit SSR.** Must use `onMount` for map initialization. Server-side rendering will fail silently.
- **Breaking changes.** Svelte 5 runes migration may require wrapper updates. A separate `svelte-maplibre-gl` (Svelte 5 native) has emerged from MIERUNE, potentially fragmenting the ecosystem.

---

## Tier 2 -- Specialized Use Cases

---

### Resium -- CesiumJS for React

React wrapper around CesiumJS for 3D globe applications. **Only use when you need true 3D globe rendering** (digital twins, flight trackers, urban planning).

**WARNING: Single-maintainer risk.** Resium is maintained by one developer (reearth). If they step away, the wrapper is orphaned. For mission-critical 3D globe applications, evaluate whether your team can maintain the CesiumJS integration directly without Resium as a dependency.

**Why Tier 2:** CesiumJS is the only production-grade 3D globe engine in the browser. Resium wraps it for React. But the single-maintainer risk and 5MB+ bundle size are significant enterprise concerns. MapLibre Globe view (available in recent versions) may cover simpler 3D globe needs without CesiumJS.

**Caveats:**
- **Build configuration nightmare.** CesiumJS requires copying static assets (Workers, CSS) to the public directory. Every bundler (Webpack, Vite, Next.js) has different plugins/configs for this. This is the #1 pain point.
- **Bundle size explosion.** CesiumJS alone is ~3-5MB gzipped. Not suitable for apps where initial load time matters.
- **Memory management.** CesiumJS is notoriously memory-hungry. Entities must be explicitly destroyed.
- **Thin wrapper.** Resium doesn't abstract CesiumJS complexity -- it just wraps it in React components. You still need deep CesiumJS knowledge.
- **Anti-pattern: Using CesiumJS for 2D maps.** If you don't need a globe or 3D terrain, MapLibre is the right choice.

---

## SSR Considerations (Next.js / Nuxt / SvelteKit)

All map libraries require DOM/WebGL and cannot render server-side. This creates universal challenges:

- **Hydration mismatch.** Server renders empty div, client renders map. Causes React/Vue hydration warnings.
- **Bundle splitting.** Map libraries are 200-300KB. Without dynamic imports, they bloat the initial server-rendered page.
- **API keys in source.** MapTiler/Mapbox API keys in client-side code are visible. Use domain-restricted keys.
- **SEO.** Map content is not indexable. Provide text/table alternatives for SEO-critical pages.

```jsx
// Next.js App Router -- correct pattern
const MapView = dynamic(() => import('./MapView'), { ssr: false });
```

**Caveats:**
- **`'use client'` boundary placement matters.** Place it on the map component itself, not on the page. This keeps server components for non-map content.
- **WebGL context limits apply across all frameworks.** 8-16 max contexts per page. Multiple map instances in iframes multiply this risk.
