# Full-Stack Architecture -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**Default stack:** PostGIS + Martin + MapLibre + Next.js. Start with a modular monolith; extract microservices only when you have proven scaling bottlenecks. For static-only data, skip the server entirely: PMTiles + COG on CDN.

---

## Tier 1 -- Production First Choices

---

### Architecture Patterns

#### Microservices -- The Enterprise Default

For large teams (5+ engineers), separate concerns into independent services. Each team owns their service end-to-end.

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Ingest  │  │  Tile    │  │  API     │  │ Frontend │
│  Service │  │  Server  │  │  Server  │  │  App     │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │              │
     └─────────────┼─────────────┘              │
                   │                            │
         ┌────────▼────────┐                    │
         │  PostGIS / S3   │◀───────────────────┘
         └─────────────────┘
```

**Caveats:**
- **Network overhead.** Each service boundary adds latency. A tile request that could be 5ms locally becomes 20ms+ across service boundaries.
- **Distributed debugging.** When a tile is wrong, is it the ingest service, the database, or the tile server? OpenTelemetry tracing is essential.
- **Data consistency.** PostGIS schema changes must be coordinated across services. Database migrations become a cross-team concern.
- **Operational burden.** Each service needs its own CI/CD, monitoring, alerting, and on-call rotation.
- **Anti-pattern: Microservices for a small team.** If 2-3 engineers maintain everything, a modular monolith is simpler with the same benefits.

**Why Tier 1:** The right architecture when teams need independent deployment cycles, services have different scaling profiles, and organizational boundaries map to service boundaries.

#### Modular Monolith -- Best Balance for Medium Teams

Single deployment unit, internally organized as modules. Each module can be extracted to a service later if needed.

```
┌─────────────────────────────────────────┐
│            Monolith Application          │
│  ┌──────┐ ┌──────┐ ┌───────┐ ┌───────┐ │
│  │Ingest│ │Store │ │ Serve │ │Render │ │
│  │Module│ │Module│ │Module │ │Module │ │
│  └──────┘ └──────┘ └───────┘ └───────┘ │
│            ┌──────────┐                  │
│            │ PostGIS  │                  │
│            └──────────┘                  │
└─────────────────────────────────────────┘
```

**Caveats:**
- **Module boundaries erode.** Without discipline, modules start importing each other's internals. Enforce boundaries with lint rules or package boundaries.
- **Single deployment means coordinated releases.** A bug in the ingest module delays the tile server fix.
- **Scaling is all-or-nothing.** Cannot scale the tile serving independently of the ingest service.

**Why Tier 1:** The pragmatic choice for medium teams (3-8 engineers). Organizational benefits of service separation without the operational overhead of distributed systems.

---

### Reference Stack: PostGIS + Martin + Next.js + MapLibre

The most common modern GIS web application stack. Proven at scale.

```
┌─────────────────────────────────┐
│           CDN / Edge            │
│  (Cloudflare, CloudFront)       │
│  - Static assets, tile cache    │
├─────────────────────────────────┤
│        Reverse Proxy            │
│  (Caddy / Nginx)               │
│  - HTTPS, routing, compression  │
├─────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐    │
│  │ Next.js  │  │  Martin  │    │
│  │ Frontend │  │  Tiles   │    │
│  │ + API    │  │          │    │
│  └────┬─────┘  └────┬─────┘    │
│       │              │          │
│       └──────┬───────┘          │
│         ┌────▼────┐             │
│         │ PostGIS │             │
│         └─────────┘             │
└─────────────────────────────────┘
```

**Caveats:**
- **PostGIS is the single point of failure.** If the database goes down, both tiles and API are dead. Use replicas for reads and automated failover.
- **Martin and Next.js compete for PostGIS connections.** Connection pool sizing must account for both workloads. Use PgBouncer for connection pooling.
- **No built-in circuit breaker.** If PostGIS times out, Martin returns 500s immediately with no graceful degradation. Plan a fallback (e.g., serve from CDN cache on backend failure).
- **Cache invalidation across layers.** When PostGIS data changes, Martin's in-memory cache and CDN cache may serve stale tiles. Plan your cache TTL strategy.

**Why Tier 1:** Scales to enterprise with K8s, managed PostGIS, CDN, and Prometheus/Grafana. The most well-trodden path with the most community support. Same stack works at small scale with Docker Compose on a single VPS.

---

### Zero-Infrastructure Stack: PMTiles + COG + Static Frontend

No servers to manage. Data as static files on object storage.

```
┌─────────────────────────────────┐
│           CDN / R2              │
│  ┌──────────┐  ┌──────────┐    │
│  │ PMTiles  │  │   COG    │    │
│  │ (vector) │  │ (raster) │    │
│  └──────────┘  └──────────┘    │
└─────────────────────────────────┘
              ▲
              │ HTTP Range Requests
┌─────────────┴────────────────┐
│  Static Frontend (Vercel)    │
│  MapLibre + PMTiles Protocol │
└──────────────────────────────┘
```

**Caveats:**
- **Static data only.** Any data update requires regenerating files and re-uploading. Not suitable for dynamic data.
- **No server-side spatial queries.** All filtering/analysis must happen client-side.
- **Range request support required.** Some corporate proxies strip HTTP Range headers.
- **File size limits.** Very large PMTiles files (10GB+) can cause issues with some CDN providers.

**Why Tier 1:** The cheapest stack possible. PMTiles on Cloudflare R2 + frontend on Vercel = $0-5/mo with global CDN distribution. Perfect for basemaps, reference layers, and analytical datasets that change infrequently.

---

## API Design for Spatial Data

### GeoJSON Feature API -- Best Practices

```python
# FastAPI spatial endpoint with proper validation and pagination
from fastapi import FastAPI, Query, HTTPException

@app.get("/api/features")
async def get_features(
    bbox: str = Query(..., description="minLng,minLat,maxLng,maxLat"),
    limit: int = Query(1000, le=5000),
    offset: int = Query(0),
):
    coords = [float(c) for c in bbox.split(",")]
    if len(coords) != 4:
        raise HTTPException(400, "Invalid bbox")
    # Validate ranges
    if not (-180 <= coords[0] <= 180 and -180 <= coords[2] <= 180):
        raise HTTPException(400, "Invalid longitude")
    if not (-90 <= coords[1] <= 90 and -90 <= coords[3] <= 90):
        raise HTTPException(400, "Invalid latitude")

    # GOOD: Uses spatial index via ST_Intersects with envelope
    result = await db.fetch("""
        SELECT id, name, category, ST_AsGeoJSON(geom)::json as geometry
        FROM features
        WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326)
        LIMIT $5 OFFSET $6
    """, *coords, limit, offset)

    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"id": r["id"], "name": r["name"], "category": r["category"]},
                "geometry": r["geometry"],
            }
            for r in result
        ],
    }
```

**Caveats:**
- **Always limit response size.** Without `LIMIT`, a bbox covering the entire world returns millions of features and crashes the client.
- **Validate bbox coordinates.** Unvalidated bbox strings are SQL injection vectors.
- **Use `&&` (bbox overlap) before `ST_Intersects`.** The `&&` operator uses the GiST index for initial filtering. `ST_Intersects` without `&&` can miss the index.
- **Anti-pattern: Returning full geometry when only properties are needed.** Large polygons in GeoJSON bloat response size. Offer `fields` parameter to control output.
- **Anti-pattern: GeoJSON for >10K features.** GeoJSON is verbose. For large datasets, return tile URLs or use FlatGeobuf/GeoParquet format.

---

## Data Pipeline Patterns

### ETL Pipeline: Ingest -> PostGIS -> Martin -> CDN

```
Raw Data (Shapefile, GeoJSON, CSV)
    │
    ▼ ogr2ogr / Python script
PostGIS (clean, indexed, validated)
    │
    ├──▶ Martin (live vector tiles)
    │       │
    │       ▼ Nginx cache → CDN
    │
    └──▶ Tippecanoe (static PMTiles)
            │
            ▼ Upload to R2/S3 → CDN
```

**Caveats:**
- **Data validation is critical.** Invalid geometries in PostGIS cause Martin to generate corrupt tiles. Always run `ST_IsValid()` checks and `ST_MakeValid()` fixes during ingest.
- **SRID consistency.** All data must be in EPSG:4326 for MapLibre consumption. Martin transforms on-the-fly but this adds CPU cost. Pre-transform during ingest.
- **Index creation after bulk load.** Create GiST indexes after bulk inserts, not before. Building indexes during insert is 10x slower.

---

## Decision Framework

| Scenario | Recommended Stack |
|----------|-------------------|
| Enterprise GIS with OGC requirements | PostGIS + GeoServer + OpenLayers |
| Modern web app, dynamic data | PostGIS + Martin + MapLibre + Next.js |
| Static data visualization | PMTiles + MapLibre + Vercel |
| Real-time fleet tracking | PostGIS + Martin + Socket.io + MapLibre |
| Raster/satellite imagery | PostGIS + TiTiler + COG on S3 + MapLibre |
| Offline field survey | PostGIS + Martin + PWA + IndexedDB |
| Analytics dashboard | DuckDB + GeoParquet + deck.gl |
| Collaborative editing | PostGIS + Yjs + MapLibre Draw |
| Rapid prototype | Supabase + MapLibre + SvelteKit |

**Caveats:**
- **Don't over-architect early.** Start with the simplest stack (PostGIS + Martin + MapLibre) and add complexity only when you hit specific scaling or feature bottlenecks.
- **Stack migration is expensive.** Choosing GeoServer when you need modern web mapping, or choosing Supabase when you need enterprise control, creates costly migration projects later.
- **Anti-pattern: Choosing technology based on what's trendy.** Choose based on your team's expertise, your data's characteristics, and your users' needs.
