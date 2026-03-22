# JS Bindbox -- JavaScript GIS Quick Decision Guide

> Data validated: 2026-03-21
>
> **Bindbox** = **Bind**ing tool**box** for JavaScript geospatial libraries. Every tool is ranked by time-to-result for small projects. Need enterprise architecture? Go to [Web Dev Guide](../web-dev/).

---

## I Need...

### Show a map on screen

| Tool | Time | Best for |
|------|------|----------|
| Leaflet | 5 min | Simplest setup, <5K features, no build step |
| MapLibre GL JS | 15 min | Vector tiles, 100K+ features, production standard |

-> [Full guide: 2d-mapping.md](2d-mapping.md)

### Add spatial analysis

| Tool | Time | Best for |
|------|------|----------|
| Turf.js | 2 min | Buffer, intersect, point-in-polygon (90% of cases) |
| H3-js | 10 min | Hexagonal binning and analytics |

-> [Full guide: spatial-analysis.md](spatial-analysis.md)

### Add charts or a dashboard

| Tool | Time | Best for |
|------|------|----------|
| Observable Plot | 5 min | Quick choropleth, smallest bundle |
| ECharts | 30 min | Map + charts in one library |

-> [Full guide: charting-integration.md](charting-integration.md)

### Add 3D effects

| Tool | Time | Best for |
|------|------|----------|
| MapLibre 3D | 15 min | Extruded buildings + terrain, zero extra deps |
| Globe.gl | 15 min | Quick globe prototype / landing page |

-> [Full guide: 3d-mapping.md](3d-mapping.md)

### Load geodata

| Tool | Time | Best for |
|------|------|----------|
| GeoJSON | 0 min | <1MB, it's just JSON |
| FlatGeobuf | 5 min | Spatial filter, no server, 1K-1M features |
| PMTiles | 10 min | Large static data, CDN-hosted tiles |

-> [Full guide: data-formats-loading.md](data-formats-loading.md)

### Add a framework wrapper

| Framework | Wrapper | Time |
|-----------|---------|------|
| React | react-map-gl | 15 min |
| Vue | vue-maplibre-gl | 15 min |
| Svelte | svelte-maplibre | 10 min |

-> [Full guide: framework-integration.md](framework-integration.md)

### Deploy

| Stack | Cost |
|-------|------|
| PMTiles on CDN + static frontend | $0-5/mo |
| Docker Compose (PostGIS + Martin + Caddy) | $20-40/mo |

-> [Full guide: ../web-dev/deployment.md](../web-dev/deployment.md)

### Need enterprise architecture?

-> Go to [Web Dev Guide](../web-dev/) -- tiered recommendations, Kubernetes, CI/CD, testing, security.

---

## Unified Tool Index

All tools mentioned across js-bindbox, one line each.

### Mapping

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| Leaflet | Lightest 2D map library, 40KB, CDN-ready | Active | [2d-mapping](2d-mapping.md) |
| MapLibre GL JS | Production WebGL map engine, vector tiles, 100K+ features | Active | [2d-mapping](2d-mapping.md) |
| MapTiler SDK | MapLibre wrapper with free basemaps and geocoding | Active | [2d-mapping](2d-mapping.md) |
| OpenLayers | OGC standard support (WMS/WFS), custom projections | Active | [2d-mapping](2d-mapping.md) |

### 3D & Globe

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| MapLibre 3D | Extruded buildings + terrain, zero extra deps | Active | [3d-mapping](3d-mapping.md) |
| Globe.gl | 3-line globe for prototypes and landing pages | Active | [3d-mapping](3d-mapping.md) |
| deck.gl | GPU-accelerated 10M+ point visualization | Active | [3d-mapping](3d-mapping.md) |
| CesiumJS | True 3D globe, 3D Tiles, digital twins | Active | [3d-mapping](3d-mapping.md) |
| Three.js + geo | Custom shaders, AR/VR, max rendering control | Active | [3d-mapping](3d-mapping.md) |

### Spatial Analysis

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| Turf.js | Client-side spatial operations, modular, 3KB per function | Active | [spatial-analysis](spatial-analysis.md) |
| Proj4js | Coordinate transformation between CRS | Stable | [spatial-analysis](spatial-analysis.md) |
| Flatbush | Fastest static spatial index, 8 bytes/item | Stable | [spatial-analysis](spatial-analysis.md) |
| rbush | Dynamic spatial index, insert/delete at runtime | Stable | [spatial-analysis](spatial-analysis.md) |
| kdbush | Point-only spatial index with nearest-neighbor | Stable | [spatial-analysis](spatial-analysis.md) |
| H3-js | Uber hexagonal grid, analytics and binning | Active | [spatial-analysis](spatial-analysis.md) |
| JSTS | Topologically correct geometry operations | Stable | [spatial-analysis](spatial-analysis.md) |
| geotiff.js | Read COG/GeoTIFF in the browser | Active | [spatial-analysis](spatial-analysis.md) |
| loam | GDAL WASM wrapper | **Deprecated** | [spatial-analysis](spatial-analysis.md) |
| gdal3.js | GDAL WASM, niche browser-side raster processing | Active (niche) | [spatial-analysis](spatial-analysis.md) |

### Data Formats

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| GeoJSON | Universal geo format, zero deps | Standard | [data-formats](data-formats-loading.md) |
| TopoJSON | Shared-boundary compression for admin regions | Stable | [data-formats](data-formats-loading.md) |
| FlatGeobuf | Streaming binary with built-in spatial index | Active | [data-formats](data-formats-loading.md) |
| Papa Parse | CSV streaming parser | Stable | [data-formats](data-formats-loading.md) |
| @tmcw/togeojson | GPX/KML to GeoJSON converter | Active | [data-formats](data-formats-loading.md) |
| shapefile.js | Legacy Shapefile reader (trusted sources only) | Stable (8yr no release) | [data-formats](data-formats-loading.md) |
| DuckDB-WASM | SQL analytics on GeoParquet in browser (analysis dashboards, not interactive maps) | Active | [data-formats](data-formats-loading.md) |
| GeoArrow | Zero-copy columnar geo data (immature ecosystem) | Experimental | [data-formats](data-formats-loading.md) |

### Charts

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| Observable Plot | Fastest path to a choropleth, D3-based | Active | [charting](charting-integration.md) |
| ECharts | All-in-one map + charts dashboard library | Active | [charting](charting-integration.md) |
| Plotly.js | Python/R interop, interactive charts | Active | [charting](charting-integration.md) |
| Vega-Lite | Grammar-of-graphics, academic contexts | Active | [charting](charting-integration.md) |
| D3-geo | Full custom cartographic control, 150+ projections | Stable | [charting](charting-integration.md) |
| AntV L7 | Chinese market geo-visualization (AMap basemap) | Active | [charting](charting-integration.md) |
| Highcharts Maps | Enterprise commercial charting with map support | Active (commercial) | [charting](charting-integration.md) |

### Framework Wrappers

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| react-map-gl | Canonical React MapLibre wrapper (Uber/vis.gl) | Active | [framework](framework-integration.md) |
| vue-maplibre-gl | Vue 3 MapLibre wrapper | Active | [framework](framework-integration.md) |
| svelte-maplibre | Svelte MapLibre wrapper, best DX | Active | [framework](framework-integration.md) |
| Resium | React CesiumJS wrapper (**single maintainer risk**) | Active (risk) | [framework](framework-integration.md) |

### Tile Serving

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| PMTiles | Serverless tiles via CDN, zero ops | Active | [tile-servers](tile-servers.md) |
| Martin | Production PostGIS tile server (Rust, fastest) | Active (pre-1.0) | [tile-servers](tile-servers.md) |
| pg_tileserv | Zero-config PostGIS tiles (prototype only) | Active | [tile-servers](tile-servers.md) |
| TileServer GL | Vector-to-raster WMTS rendering | Active (niche) | [tile-servers](tile-servers.md) |
| t-rex | Rust tile server | **Deprecated** | [tile-servers](tile-servers.md) |
| GeoServer | Full OGC compliance (WMS/WFS/WMTS) | Active | [tile-servers](tile-servers.md) |

### Real-Time & Offline

| Tool | One-line description | Status | Guide |
|------|---------------------|--------|-------|
| SSE | Browser-native one-way streaming, zero deps | Standard | [realtime](realtime-offline-advanced.md) |
| Socket.io | Bi-directional real-time, largest ecosystem | Active | [realtime](realtime-offline-advanced.md) |
| Supabase Realtime | Managed real-time with PostGIS integration | Active | [realtime](realtime-offline-advanced.md) |
| MQTT (mqtt.js) | IoT sensor protocol (production: use wss://) | Active | [realtime](realtime-offline-advanced.md) |
| Dexie.js | IndexedDB wrapper for offline feature storage | Active | [realtime](realtime-offline-advanced.md) |
| Yjs | CRDT collaborative editing (requires y-redis + compression for production) | Active | [realtime](realtime-offline-advanced.md) |

### Performance

Summary of optimization techniques -> [performance-optimization.md](performance-optimization.md). For enterprise performance tuning -> [web-dev/performance.md](../web-dev/performance.md).

---

## General Pitfalls

Read these before you start. They apply across the entire JS GIS stack.

- **EPSG:3857 lock-in.** MapLibre GL JS only supports Web Mercator. Need CGCS2000, GCJ-02, or custom CRS? Use OpenLayers or Leaflet + proj4.

- **Bundle size adds up fast.** MapLibre (~200KB) + Turf.js full (~180KB) + a charting lib = 500KB+ before your app code. Use modular imports (`@turf/boolean-point-in-polygon` is 3KB vs `@turf/turf` at 180KB). Tree-shake everything.

- **WebGL context limits.** Browsers cap WebGL contexts at 8-16. MapLibre + deck.gl + CesiumJS on one page will hit it.

- **SSR breaks everything WebGL.** MapLibre, deck.gl, CesiumJS crash in Node.js. Use dynamic imports or client-only wrappers.

- **Memory leaks in SPAs.** Always call `map.remove()` / `viewer.destroy()` on component unmount.

- **GeoJSON does not scale.** Works for <10K features. Beyond that, switch to vector tiles (PMTiles) or FlatGeobuf.

- **Coordinate order confusion.** GeoJSON = `[longitude, latitude]`. Leaflet = `[latitude, longitude]`. Silent bugs.

---

## Cross-References

- **[Web Dev Guide](../web-dev/)** -- Full-stack architecture, backend services, deployment, testing, security. Enterprise-first.
