# JS Bindbox

> JavaScript libraries and frameworks for web-based geospatial development -- the definitive expert-level reference.

**Bindbox** = **Bind**ing tool**box** / sand**box** for JavaScript geospatial libraries. Every page is written as a deep-dive by domain experts, covering not just what each library does, but how it performs at scale, how to migrate between libraries, and the unconventional "dark arts" tricks that save hours in production.

## Table of Contents

### Core Mapping Libraries
| Page | Lines | Description |
|------|-------|-------------|
| [2D Mapping -- Deep Dive](2d-mapping.md) | 1,598 | Leaflet, MapLibre GL JS, OpenLayers, MapTiler SDK -- performance profiles, plugin ecosystems, migration guides, expression engine, Canvas/WebGL internals |
| [3D Mapping & Globe -- Deep Dive](3d-mapping.md) | 1,634 | CesiumJS (3D Tiles, CZML, terrain), deck.gl (GPU layers, WebGPU), Three.js + geo, Globe.gl, MapLibre 3D -- rendering pipelines, memory management, LOD strategies |
| [Tile Servers & Services](tile-servers.md) | 297 | PMTiles, Martin, TileServer GL, pg_tileserv, t-rex -- format comparison, cloud tile services, deployment |

### Analysis & Data
| Page | Lines | Description |
|------|-------|-------------|
| [Spatial Analysis -- Deep Dive](spatial-analysis.md) | 1,720 | Turf.js (complete guide + benchmarks), JSTS, Proj4js, geotiff.js (COG), H3-js, Flatbush/RBush/kdbush, loam (GDAL WASM), Web Workers |
| [Data Formats & Loading](data-formats-loading.md) | 1,451 | GeoJSON streaming, FlatGeobuf spatial filter, GeoParquet + DuckDB WASM, Apache Arrow/GeoArrow, COG in browser, STAC client, Shapefile/CSV/GPX/KML, WASM tools |
| [Charting & Geo-Viz -- Deep Dive](charting-integration.md) | 1,314 | D3-geo (projections, Canvas), ECharts 5.x, Observable Plot, Plotly.js, Vega-Lite, AntV L7, synchronized multi-view patterns (brushing, crossfilter, state management) |

### Engineering & Production
| Page | Lines | Description |
|------|-------|-------------|
| [Framework Integration](framework-integration.md) | 700+ | React (react-map-gl, Next.js), Vue 3 (vue-maplibre-gl, Nuxt), Svelte (svelte-maplibre, SvelteKit), Angular -- state management, SSR, testing, URL sync |
| [Performance & Optimization](performance-optimization.md) | 1,476 | Rendering benchmarks (FPS by feature count), memory profiles, bundle size optimization, Web Workers, mobile perf, monitoring, tile prefetching |
| [Real-Time, Offline & Advanced](realtime-offline-advanced.md) | 1,869 | WebSocket/SSE/MQTT live tracking, Service Worker tile caching, PWA, offline-first, collaborative editing (Yjs), testing (Playwright), a11y, security, deployment |

---

## What Makes This Different

- **Performance benchmarks**: Real numbers -- FPS at 1K/10K/100K/1M features for every library
- **Development experience**: TypeScript quality, debugging tools, learning curves, bundle sizes
- **Migration guides**: Detailed before/after code for Leaflet→MapLibre, Mapbox→MapLibre, OpenLayers→MapLibre
- **Plugin ecosystems**: 20+ Leaflet plugins, MapLibre protocols, OpenLayers extensions, deck.gl layers
- **Framework patterns**: Production React/Vue/Svelte/Angular code, not toy examples
- **Advanced Dark Arts**: Every page includes expert tricks, performance hacks, and unconventional techniques
- **Cross-platform**: Mobile optimization, offline/PWA, SSR considerations, accessibility

## How to Choose (Quick Reference)

| Goal | Start Here |
|------|-----------|
| Simple interactive map | [2D Mapping](2d-mapping.md) → Leaflet |
| High-performance vector tiles | [2D Mapping](2d-mapping.md) → MapLibre GL JS |
| 3D digital twin / globe | [3D Mapping](3d-mapping.md) → CesiumJS |
| Millions of points (GPU) | [3D Mapping](3d-mapping.md) → deck.gl |
| Client-side spatial analysis | [Spatial Analysis](spatial-analysis.md) → Turf.js + Flatbush |
| Browser SQL on geo data | [Data Formats](data-formats-loading.md) → DuckDB WASM |
| Geo dashboard with charts | [Charting](charting-integration.md) → ECharts or D3 + MapLibre |
| React/Vue/Svelte integration | [Framework Integration](framework-integration.md) |
| Optimize slow map app | [Performance](performance-optimization.md) |
| Live vehicle tracking | [Real-Time](realtime-offline-advanced.md) → WebSocket |
| Offline field app | [Real-Time](realtime-offline-advanced.md) → PWA + PMTiles |
| Serverless tile hosting | [Tile Servers](tile-servers.md) → PMTiles on CDN |

## Cross-Module Links

- **[Tools](../tools/)** -- Desktop GIS, Python libraries, servers, databases, AI/ML
- **[Data Sources](../data-sources/)** -- Where to find the geospatial data these libraries consume
