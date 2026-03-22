# Tile Servers & Hosting

> Data validated: 2026-03-21

## 30-Second Decision

| I need... | Use this | Startup time |
|-----------|----------|-------------|
| Zero ops, upload and done | PMTiles on CDN | 10 min |
| Free tiles, no self-hosting | MapTiler / Stadia Maps | 5 min (API key) |
| Live data from PostGIS | Martin | 30 min (Docker) |
| Full OGC enterprise | GeoServer | Days |

---

## Detailed Guide (by startup time)

### 1. PMTiles (Protomaps) -- The Serverless Revolution

No tile server needed. Upload a file. Done. Single-file archive serving tiles via HTTP range requests from any static hosting (S3, CloudFlare R2, GitHub Pages).

**Quick facts:** Zero ops | CDN-level scalability | Cost = storage + bandwidth only

```bash
# Generate PMTiles from GeoJSON
tippecanoe -o output.pmtiles -z14 -Z0 --drop-densest-as-needed input.geojson
# Upload to S3/R2
aws s3 cp output.pmtiles s3://my-bucket/tiles/
```

```javascript
import { Protocol } from 'pmtiles';
maplibregl.addProtocol('pmtiles', new Protocol().tile);

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      mydata: { type: 'vector', url: 'pmtiles://https://your-cdn.com/data.pmtiles' }
    },
    layers: [{
      id: 'my-layer', type: 'fill', source: 'mydata',
      'source-layer': 'your_layer_name',
      paint: { 'fill-color': '#088', 'fill-opacity': 0.6 }
    }]
  }
});
```

| Traditional Tiles | PMTiles |
|-------------------|---------|
| Tile server + database | Single `.pmtiles` file |
| Scaling = more servers | Scaling = CDN handles it |
| Ops: Docker, monitoring | Upload file, done |
| Cost: server + bandwidth | Cost: storage + bandwidth only |

**Small project:** Best choice for static/infrequently-updated data. Zero infrastructure, CDN performance from day one.

**Key caveats:**
- Static data only -- must regenerate entire file for data changes
- Range request support required (some corporate proxies strip them)
- No dynamic filtering (unlike Martin)
- Initial load requires 2-3 range requests (200-500ms on slow connections)
- Always set `maxzoom` on MapLibre source to match tippecanoe `-z` value
- **Anti-pattern:** Using PMTiles for frequently updated data. Use Martin + PostGIS instead.

---

### 2. Cloud Tile Services (MapTiler, Stadia Maps, Mapbox)

API key and go. Zero self-hosting, professionally maintained basemaps.

| Provider | Free Tier | Vector | Satellite | Geocoding |
|----------|-----------|--------|-----------|-----------|
| MapTiler | 100K req/mo | Yes | Yes | Yes |
| Stadia Maps | 200K credits/mo | Yes | No | Yes (Pelias) |
| Mapbox | 200K req/mo | Yes | Yes | Yes |
| Protomaps (DIY) | Unlimited (self-host) | Yes | No | No |

**Small project:** Great for quick setup. Beautiful basemaps, free tiers generous for development.

**Key caveats:**
- Cost at scale: production traffic at 100K+ req/day gets expensive ($200-5000/mo)
- Vendor lock-in (Mapbox GL JS v2+ is proprietary)
- Provider downtime = your map is offline
- **Anti-pattern:** Committing to Mapbox when MapLibre + self-hosted tiles works

---

### 3. Martin (MapLibre)

Production standard for live PostGIS tile serving. Blazing-fast Rust server, vector tiles directly from PostGIS queries.

**Quick facts:** Rust | PostGIS + MBTiles + PMTiles sources | 4/5 production-readiness (pre-1.0 but used by Felt)

```bash
docker run -p 3000:3000 \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  ghcr.io/maplibre/martin
```

**Small project:** Worth the setup only if you need dynamic/live data from PostGIS. For static data, PMTiles is simpler.

**Key caveats:**
- Requires PostGIS -- adds operational complexity
- No built-in caching: MUST put Nginx/CDN in front for production
- Pre-1.0 API: config format changes between versions
- Connection pool exhaustion at high concurrency (tune `pool_size`)
- No built-in authentication -- handle at reverse proxy layer
- **Anti-pattern:** Running Martin without a caching proxy

For detailed Martin production configuration -> [web-dev/backend-services.md](../web-dev/backend-services.md)

---

## Low Priority / Legacy

### pg_tileserv

Zero-config PostGIS tile serving for prototyping. Point at database, auto-discovers all tables.

**Quick facts:** Go | 15 min setup | Production-readiness: 3/5

**Use for:** Quick prototyping when you already have PostGIS data. **Migrate to Martin for production** -- Martin surpasses pg_tileserv in every metric.

**Key caveats:**
- Auto-discovery exposes ALL PostGIS tables -- security risk
- No caching, no composite sources, no sprite/font serving
- Less actively maintained than Martin

### TileServer GL

Vector-to-raster rendering and WMTS. Unique ability to render vector tiles as raster images server-side.

**Quick facts:** Node.js | MBTiles only | Production-readiness: 3/5

Niche use: legacy WMTS clients needing raster output from vector styles. Node.js single-threaded performance limits throughput.

### t-rex -- DEPRECATED

**Deprecated:** Project is no longer maintained. Functionality has been absorbed by Martin. Do not use for new projects.

### GeoServer

Full OGC compliance (WMS, WFS, WCS, WMTS). Government agencies with OGC interoperability requirements often have no alternative. For modern web apps it is massive overkill.

**Key caveats:** Java monolith (2-4GB JVM), XML configuration, startup in minutes, performance 10-50x slower than Martin for vector tiles, has had CVEs. Only use when OGC compliance is a hard requirement.

For enterprise tile serving architecture -> [web-dev/backend-services.md](../web-dev/backend-services.md)

---

## PMTiles vs Martin -- When to Use Which

| Aspect | PMTiles on CDN | Martin + PostGIS |
|--------|---------------|------------------|
| Setup | Upload file to S3/R2 | Docker + PostGIS |
| Dynamic data | No (rebuild file) | Yes (live queries) |
| Scalability | CDN-level (infinite) | Horizontal scaling needed |
| Cost | Storage + bandwidth | Compute + DB + bandwidth |
| **Best when** | **Data changes rarely** | **Data changes often** |
