# Web GIS Development — 2025 Complete Guide

Everything you need to build modern web-based GIS applications — from architecture patterns to production deployment, real-time collaboration, offline maps, and security.

---

## Table of Contents

### Architecture & Full-Stack
- [Full-Stack Architecture](fullstack-architecture.md) — Architecture patterns (monolith/microservices/serverless), 6 reference stacks, API design (REST/GraphQL/gRPC/OGC), data pipelines, infrastructure patterns, cost modeling

### Frontend
- [Frontend Integration](frontend-integration.md) — React (react-map-gl, deck.gl, Resium), Vue 3 (vue-maplibre-gl), Svelte (svelte-maplibre), Web Components (Lit), state management (Zustand/Pinia), URL sync, accessibility, testing, TypeScript patterns

### Backend
- [Backend Services](backend-services.md) — Tile servers (Martin, pg_tileserv, TiTiler, GeoServer), API frameworks (FastAPI, Django/GeoDjango, Express, Hono), OGC/STAC APIs, geocoding (Nominatim, Pelias), routing (OSRM, Valhalla, pgRouting), background processing

### Performance
- [Performance Optimization](performance.md) — Vector tile pipeline (Tippecanoe, PMTiles), raster optimization (COG, compression), PostGIS tuning (indexes, partitioning, query optimization), frontend rendering (WebGL, deck.gl, Web Workers, WASM), caching (CDN, Service Worker, Nginx)

### Deployment
- [Deployment](deployment.md) — Docker (production Dockerfiles, Compose), Kubernetes (StatefulSet, HPA, Ingress), serverless (Lambda, Cloudflare Workers, Supabase), CI/CD (GitHub Actions), monitoring (Prometheus/Grafana/OpenTelemetry), disaster recovery

### Real-Time
- [Real-Time & Collaboration](realtime-and-collaboration.md) — WebSocket (Socket.io fleet tracking), SSE, MQTT (IoT sensor networks), collaborative editing (Yjs CRDT), Supabase Realtime, streaming (PostGIS LISTEN/NOTIFY, Debezium CDC, GraphQL subscriptions), geofencing

### Testing & Security
- [Testing & Security](testing-and-security.md) — Unit testing (Vitest, Turf.js, Zod validation), integration testing (Testcontainers + PostGIS), E2E testing (Playwright map page objects), spatial RBAC, JWT spatial claims, SQL injection prevention, GeoJSON validation, CORS, rate limiting, GDPR

### Offline & PWA
- [PWA & Offline Maps](pwa-and-offline.md) — Service Worker (Workbox 7), offline tile storage (IndexedDB, PMTiles offline), offline vector data (Dexie.js, rbush), data synchronization (conflict resolution, Background Sync), field survey app, Geolocation API, storage management

---

## Key Highlights

- **8 expert deep-dive guides** covering the complete web GIS stack
- **Production-ready code** — every example is copy-paste ready
- **2025 SOTA tools** — MapLibre GL JS 4.x, deck.gl 9.x, Martin 0.15+, PMTiles, COG, DuckDB-WASM
- **Full-stack coverage** — from PostGIS tuning to WebGL rendering to Kubernetes deployment
- **Real-world patterns** — fleet tracking, field surveys, collaborative editing, IoT dashboards
- **Security-first** — spatial RBAC, input validation, rate limiting, GDPR compliance

---

## Quick Navigation

| I want to... | Read this |
|--------------|-----------|
| Start a new web GIS project | [Full-Stack Architecture](fullstack-architecture.md) |
| Add a map to React/Vue/Svelte | [Frontend Integration](frontend-integration.md) |
| Set up tile serving or APIs | [Backend Services](backend-services.md) |
| Handle large datasets | [Performance Optimization](performance.md) |
| Deploy to production | [Deployment](deployment.md) |
| Build real-time features | [Real-Time & Collaboration](realtime-and-collaboration.md) |
| Test and secure my app | [Testing & Security](testing-and-security.md) |
| Make my app work offline | [PWA & Offline Maps](pwa-and-offline.md) |

---

## Recommended Stacks (2025)

| Stack | Components | Best For | Cost |
|-------|-----------|----------|------|
| Modern Full-Stack | PostGIS + Martin + Next.js 15 + MapLibre | Production apps | $20-30/mo |
| Serverless | PMTiles (R2) + Vite + MapLibre | Static data portals | $0-5/mo |
| Rapid Prototype | Supabase + SvelteKit + MapLibre | MVPs, hackathons | $0-25/mo |
| Enterprise | PostGIS + GeoServer + Spring Boot | OGC compliance | $40-80/mo |
| Analytics | DuckDB + GeoParquet + Observable | Dashboards | $0-5/mo |
| Real-Time IoT | TimescaleDB + MQTT + deck.gl | Sensor networks | $50-100/mo |
