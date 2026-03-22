# Web GIS Development -- Enterprise Reference Guide

> Data validated: 2026-03-21

## 30-Second Decision Summary

| Practice Area | Best Default Choice |
|---|---|
| Architecture & Tech Stack | PostGIS + Martin + MapLibre + Next.js |
| Data Pipeline & Tile Serving | Martin (vector) + TiTiler (raster) + Tippecanoe for generation |
| Map Rendering & Frontend | react-map-gl + MapLibre GL JS for React; deck.gl for 10M+ features |
| Deployment & Ops | Docker Compose for small teams; Kubernetes for enterprise |
| Performance & Scaling | Tippecanoe -> PMTiles on CDN; PostGIS tuning; deck.gl binary mode |
| Real-Time & Collaboration | Socket.io + Redis adapter; PostGIS LISTEN/NOTIFY for simple cases |
| Testing & Security | Vitest + Testcontainers + Playwright; Zod for GeoJSON validation |
| PWA & Offline | Workbox 7 + IndexedDB (Dexie.js) + rbush |

---

## Choose Your Path

| | Large Project | Small Project |
|---|---|---|
| **Team size** | 5+ engineers | 1-3 engineers |
| **Timeline** | Months | Days to weeks |
| **Priorities** | Scalability, maintainability, compliance | Speed to result, low ops burden |
| **Start here** | [Large Project Path](#large-project-path) (this guide) | [JS Bindbox](../js-bindbox/) (small-project-first) |

---

## Large Project Path

Each guide below is ordered by enterprise suitability -- Tier 1 tools are the proven defaults, lower tiers serve specific or niche needs.

| I need to... | Read this | Top Pick |
|---|---|---|
| Design the architecture | [Full-Stack Architecture](fullstack-architecture.md) | PostGIS + Martin + MapLibre + Next.js |
| Set up tile servers & APIs | [Backend Services](backend-services.md) | Martin (vector) / TiTiler (raster) |
| Add maps to React/Vue/Svelte | [Frontend Integration](frontend-integration.md) | react-map-gl + MapLibre GL JS |
| Deploy to production | [Deployment](deployment.md) | Kubernetes + Prometheus/Grafana |
| Handle large datasets efficiently | [Performance Optimization](performance.md) | Tippecanoe + PMTiles + deck.gl |
| Build real-time features | [Real-Time & Collaboration](realtime-and-collaboration.md) | Socket.io + Redis adapter |
| Test and secure my app | [Testing & Security](testing-and-security.md) | Vitest + Testcontainers + Playwright |
| Make it work offline | [PWA & Offline Maps](pwa-and-offline.md) | Workbox 7 + PMTiles offline |

---

## Small Project Path

Weekend project, MVP, hackathon, or portfolio piece? Jump to the **[JS Bindbox](../js-bindbox/)** -- it is organized small-project-first with setup times for every tool.

| I need... | Go to | Top Pick | Time to Result |
|---|---|---|---|
| A 2D map on screen | [JS Bindbox: 2D Mapping](../js-bindbox/2d-mapping.md) | Leaflet | 5 minutes |
| A 3D globe or digital twin | [JS Bindbox: 3D Mapping](../js-bindbox/3d-mapping.md) | Globe.gl (quick) / CesiumJS (serious) | 15 min / hours |
| Tile serving with zero ops | [JS Bindbox: Tile Servers](../js-bindbox/tile-servers.md) | PMTiles on CDN | 10 minutes |
| Spatial analysis in the browser | [JS Bindbox: Spatial Analysis](../js-bindbox/spatial-analysis.md) | Turf.js | 2 minutes |
| Charts next to my map | [JS Bindbox: Charting](../js-bindbox/charting-integration.md) | Observable Plot | 5 minutes |
| Load geodata in the browser | [JS Bindbox: Data Formats](../js-bindbox/data-formats-loading.md) | GeoJSON (just JSON) | 0 minutes |
| Integrate with React/Vue/Svelte | [JS Bindbox: Frameworks](../js-bindbox/framework-integration.md) | svelte-maplibre (best DX) | 10 minutes |
| Fix my slow map | [JS Bindbox: Performance](../js-bindbox/performance-optimization.md) | Switch to vector tiles | 30 minutes |
| Real-time / offline features | [JS Bindbox: Real-Time & Offline](../js-bindbox/realtime-offline-advanced.md) | SSE (simple) / Socket.io (full) | 15-30 minutes |

---

## Recommended Stacks

| Stack | Components | Best For | Monthly Cost | Min Team |
|---|---|---|---|---|
| **Modern Full-Stack** | PostGIS + Martin + Next.js + MapLibre | Production apps | $20-30/mo | 2 |
| **Serverless** | PMTiles (R2) + Vite + MapLibre | Static data portals | $0-5/mo | 1 |
| **Rapid Prototype** | Supabase + SvelteKit + MapLibre | MVPs, hackathons | $0-25/mo | 1 |
| **Enterprise** | PostGIS + GeoServer + Spring Boot | OGC compliance | $40-80/mo | 3+ |
| **Analytics** | DuckDB + GeoParquet + Observable | Dashboards | $0-5/mo | 1 |
| **Real-Time IoT** | TimescaleDB + MQTT + deck.gl | Sensor networks | $50-100/mo | 2+ |

---

## General Pitfalls

These apply across the entire web GIS stack. Read before you start.

- **EPSG:3857 lock-in.** MapLibre GL JS only supports EPSG:3857 (Web Mercator). If you need custom projections (CGCS2000, GCJ-02, military grids), you must use OpenLayers or Leaflet + proj4. This is the single most common "wrong choice" in GIS web dev.

- **Bundle size surprise.** CesiumJS is 5MB+ gzipped. deck.gl adds ~300KB. MapLibre is ~200KB. Leaflet is ~40KB. If your app is mobile-first or bandwidth-constrained, measure before you commit.

- **WebGL context limits.** Browsers allow 8-16 WebGL contexts per page. If you use MapLibre + deck.gl + CesiumJS on the same page, you will hit the limit. Multiple map instances also count. Plan your architecture around this.

- **Tile server is not optional at scale.** Serving raw GeoJSON works under 10K features. Beyond that, you need vector tiles (MVT) or your app will be slow. Budget time for Tippecanoe + tile server setup.

- **PostGIS index discipline.** Every spatial query without a GiST index does a full table scan. `CREATE INDEX idx ON table USING gist(geom);` is non-negotiable. Add `EXPLAIN ANALYZE` to your review checklist.

- **CORS for tiles.** If your tile server and frontend are on different origins, tiles will fail silently. Set `Access-Control-Allow-Origin` headers on your tile server or CDN from day one.

- **GeoJSON injection.** Never pass user-supplied GeoJSON directly to `ST_GeomFromGeoJSON()` without validation. Use Zod schemas or equivalent to validate geometry type, coordinate bounds, and ring closure before any database operation.

- **SSR and WebGL.** MapLibre, deck.gl, and CesiumJS all require the DOM and WebGL. They will crash in Node.js SSR. Use dynamic imports (`next/dynamic` with `ssr: false`, Nuxt `<ClientOnly>`, SvelteKit `{#if browser}`) for all map components.

- **Memory leaks in SPAs.** Map instances must be explicitly destroyed when unmounting. `map.remove()` in Leaflet/MapLibre, `viewer.destroy()` in CesiumJS. Missing cleanup causes progressive memory leaks in React/Vue/Svelte apps.

---

## Cross-References

- **[JS Bindbox](../js-bindbox/)** -- JavaScript library deep-dives with code examples, benchmarks, migration guides. Organized small-project-first.
