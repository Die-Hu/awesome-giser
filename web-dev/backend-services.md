# Backend Services -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**Vector tiles from PostGIS:** Martin. **Feature API:** FastAPI + GeoAlchemy2. **Raster tiles from COG:** TiTiler. **Tile generation pipeline:** Tippecanoe. **Routing:** OSRM (speed) or Valhalla (features). **OGC compliance (only if required):** GeoServer.

---

## Tier 1 -- Production First Choices

---

### Martin -- Rust Vector Tile Server

The fastest open-source vector tile server. Serves MVT tiles directly from PostGIS tables, functions, MBTiles, and PMTiles files. **Use Martin unless you have a specific reason not to.**

**Why Tier 1:** Rust memory safety, single Docker image, Prometheus `/metrics` endpoint, health checks, composite sources for multi-layer tiles, fastest throughput of any open-source tile server. Used by Felt and the MapLibre ecosystem. Pre-1.0 API is the only real concern.

```yaml
# martin-config.yaml -- Production configuration
listen_addresses: '0.0.0.0:3000'
worker_processes: 8
keep_alive: 75
cache_size_mb: 512

postgres:
  connection_string: 'postgresql://user:pass@localhost:5432/gisdb'
  pool_size: 20
  auto_publish:
    tables:
      from_schemas: ['public', 'spatial']
      id_columns: ['gid', 'id']
    functions:
      from_schemas: ['public']

composite_sources:
  basemap:
    - source: buildings
    - source: roads
    - source: water

sprites:
  paths:
    - /data/sprites
fonts:
  paths:
    - /data/fonts

pmtiles:
  paths:
    - /data/tiles/
  sources:
    boundaries: /data/tiles/boundaries.pmtiles

health:
  path: /health
```

```bash
# Docker run with config
docker run -d --name martin \
  -p 3000:3000 \
  -v $(pwd)/martin-config.yaml:/config.yaml \
  -v $(pwd)/data:/data \
  ghcr.io/maplibre/martin --config /config.yaml
```

**Caveats:**
- **Pre-1.0 API.** Config format has changed between versions (0.11 -> 0.13 -> 0.15). Upgrades require reading changelogs carefully; YAML keys get renamed.
- **No built-in tile cache persistence.** The in-memory cache (`cache_size_mb`) is lost on restart. You MUST put Nginx/Varnish/CDN in front for production. Without this, every pod restart means a thundering herd of cache misses hitting PostGIS.
- **Connection pool exhaustion at scale.** Default `pool_size: 20` is too low for high-concurrency. Each tile request can hold a connection for 50-500ms. At 200 concurrent users, you will see `connection pool timeout` errors. Must tune `pool_size` to 2x expected concurrent tile requests.
- **PostGIS dependency is tight.** If PostGIS goes down, Martin returns 500s immediately. No circuit breaker, no graceful degradation.
- **Complex tile queries can OOM PostGIS.** Martin passes tile queries to PostGIS. A badly written function source that does full-table scans at low zoom levels can consume gigabytes of RAM on the database side.
- **No authentication.** Zero built-in auth. You must handle this at the reverse proxy layer (Nginx JWT, Caddy, etc.).
- **Anti-pattern: Running Martin without a caching proxy.** Every tile request hits PostGIS. At 100 concurrent users scrolling a map, that's thousands of DB queries per second.
- **Anti-pattern: Using `SELECT *` in function sources.** Only select columns you need in the tile. Extra columns bloat tile size and waste bandwidth.

---

### FastAPI + GeoAlchemy2 -- Python Spatial API

The best combination for building typed, async, documented GIS APIs. FastAPI gives you auto-generated OpenAPI docs; GeoAlchemy2 maps PostGIS functions to SQLAlchemy ORM.

**Why Tier 1:** Used by Microsoft, Netflix, Uber; GeoAlchemy2 is mature; async support enables high-throughput APIs; auto-generated OpenAPI docs reduce onboarding friction; OpenTelemetry + Prometheus integration is straightforward.

```python
# Spatial API endpoint
from fastapi import FastAPI
from geoalchemy2 import Geometry
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.asyncio import AsyncSession

app = FastAPI()

class Building(Base):
    __tablename__ = 'buildings'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    height = Column(Float)
    geom = Column(Geometry('Polygon', srid=4326))

@app.get("/features")
async def get_features(bbox: str, db: AsyncSession = Depends(get_db)):
    coords = [float(c) for c in bbox.split(",")]
    result = await db.execute(
        select(Building).where(
            Building.geom.ST_Intersects(
                func.ST_MakeEnvelope(*coords, 4326)
            )
        ).limit(5000)
    )
    return build_feature_collection(result.scalars().all())
```

**Caveats:**
- **GDAL/GEOS system dependency.** GeoAlchemy2 needs libgeos. Docker images must include `libgdal-dev`. Build times increase.
- **Async + SQLAlchemy pitfalls.** Using async with GeoAlchemy2 requires `asyncpg` + careful session management. Mixing sync and async code causes "already attached to a different loop" errors.
- **Python GIL limits true concurrency.** CPU-bound spatial operations don't parallelize within a single process. Use multiple uvicorn workers.
- **No spatial query validation.** Easy to accidentally write queries that bypass spatial indexes (e.g., using `ST_Distance < threshold` instead of `ST_DWithin`). PostGIS EXPLAIN ANALYZE is mandatory during development.
- **Memory leaks with large GeoJSON responses.** Serializing 10K+ features into GeoJSON on each request allocates significant memory. Pagination is critical but not built-in for spatial queries.
- **Anti-pattern: Using synchronous SQLAlchemy calls in async FastAPI endpoints.** Blocks the event loop. Use async drivers (asyncpg) or `run_in_executor`.

---

### TiTiler -- Dynamic Raster Tile Server

The standard for serving raster tiles from Cloud-Optimized GeoTIFFs. Band math, colormaps, mosaics, STAC integration. **If you serve raster imagery, this is the tool.**

**Why Tier 1:** NASA/USGS/Development Seed production-verified. FastAPI base makes it extensible. The GDAL dependency is the main operational burden but Docker solves it.

```python
# TiTiler with optimized settings
# Environment variables for production
# GDAL_CACHEMAX=512
# GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
# VSI_CACHE=TRUE
# VSI_CACHE_SIZE=536870912
```

**Caveats:**
- **GDAL dependency hell.** GDAL is notoriously difficult to install correctly. Version mismatches between Python bindings and system GDAL cause cryptic segfaults. Always use the official Docker image.
- **Memory hungry.** Each COG read operation allocates significant memory. Under load, Python processes can consume 2-4GB each. Set `GDAL_CACHEMAX` carefully.
- **Cold start on Lambda.** First invocation takes 3-8 seconds due to GDAL initialization. Provisioned concurrency ($$$) is the workaround.
- **No built-in tile cache.** Like Martin, every request re-reads from source. CDN/Redis caching is mandatory for production.
- **COG creation is a prerequisite.** Your data MUST be in Cloud-Optimized GeoTIFF format. Converting existing GeoTIFFs to COG adds a pipeline step.
- **Anti-pattern: Serving non-COG rasters through TiTiler.** If the source GeoTIFF is not Cloud-Optimized, TiTiler must read the entire file for each tile request. Always convert to COG first.

---

### Tippecanoe -- Vector Tile Generation CLI

The industry standard for converting GeoJSON/FlatGeobuf to PMTiles/MBTiles. Created by Mapbox, maintained by Felt. **Every vector tile pipeline should use this.**

**Why Tier 1:** Production-readiness 5/5. Deterministic output makes it CI/CD-friendly. No alternative comes close for quality and control over tile generation.

```bash
# Enterprise recipe: Multi-layer basemap from FlatGeobuf (2-5x faster than GeoJSON)
tippecanoe -o basemap.pmtiles \
  --named-layer=buildings:buildings.fgb \
  --named-layer=roads:roads.fgb \
  --named-layer=water:water.fgb \
  --minimum-zoom=0 --maximum-zoom=14 \
  --read-parallel \
  --maximum-tile-bytes=500000

# Recipe: Buildings with zoom-dependent detail
tippecanoe -o buildings.pmtiles \
  --minimum-zoom=10 --maximum-zoom=15 \
  --simplification=10 \
  --drop-smallest-as-needed \
  --include=name,height,type,area \
  --layer=buildings \
  buildings.geojson
```

**Caveats:**
- **Memory usage scales with data.** Processing a 10GB GeoJSON file can consume 20-40GB RAM. CI/CD runners with 7GB RAM will OOM. Pre-split data or use FlatGeobuf input (uses less memory).
- **Processing time for large datasets.** 100M features can take 2-6 hours. No checkpointing -- if it crashes at 95%, you start over.
- **No incremental updates.** Cannot add features to an existing PMTiles file. Must regenerate the entire tileset.
- **Drop strategies are opaque.** `--drop-densest-as-needed` makes lossy decisions about what to keep. Important features may be dropped at certain zoom levels with no warning. Always verify output visually.
- **GeoJSON input is slow.** Converting to FlatGeobuf first (`ogr2ogr -f FlatGeobuf`) then using `--read-parallel` is 2-5x faster.
- **CLI-only.** No library API. Must shell out from application code.

---

### GeoServer -- Full OGC Compliance

The gold standard for OGC compliance. WMS, WFS, WCS, WPS, WMTS, CSW -- every OGC standard is supported. **Only use GeoServer when OGC interoperability is a hard requirement** (government agencies, defense, inter-agency data sharing). For modern web apps, Martin + PostGIS is faster and simpler.

**Why Tier 1 (OGC-limited):** 20+ year history, used by governments worldwide. JMX metrics and Prometheus exporter plugins available. If you don't need OGC, use Martin instead -- it's 10-50x faster for the same data.

**Caveats:**
- **Memory monster.** Default heap is 256MB which is far too low. Production needs 2-8GB depending on concurrent WMS requests. GC pauses cause request timeouts.
- **Configuration drift.** The `data_dir` contains XML files as the single source of truth. Git-tracking the data_dir is essential but tricky (binary SLD files, generated caches).
- **Admin GUI is a trap.** Easy to click through, impossible to reproduce. All production config should go through the REST API or scripted setup.
- **Startup time.** 30-60 seconds cold start. Not suitable for auto-scaling scenarios.
- **Plugin compatibility hell.** GeoServer plugins must match the exact version. A 2.24 plugin on 2.25 can cause silent failures or ClassCastExceptions at runtime.
- **Security.** Default admin password. The web admin panel has had CVEs. Must be behind a firewall.
- **Anti-pattern: Using GeoServer for modern web apps that don't need OGC.** If you're building a React + MapLibre app, GeoServer adds massive complexity for no benefit. Use Martin + PostGIS.

---

### OSRM -- Fastest Open-Source Routing Engine

Pre-processed routing graph enables microsecond queries. **Use OSRM when speed is the priority and you don't need complex routing constraints.**

**Why Tier 1:** Used by Mapbox and logistics companies worldwide. Production-readiness 5/5. Fastest open-source router.

**Caveats:**
- **Pre-processing is mandatory and slow.** Planet-scale takes 12+ hours and 128GB+ RAM.
- **Memory at runtime.** Full planet routing graph needs 40-80GB RAM. Country-level extracts are 2-8GB.
- **No live updates.** Changing the road network requires full re-preprocessing.
- **Limited routing profiles.** Car, bicycle, foot. Custom profiles require modifying Lua profiles.
- **Anti-pattern: Using OSRM for complex routing constraints.** If you need truck height/weight restrictions, time-dependent routing, or multi-modal routing, use Valhalla instead.

---

### Valhalla -- Feature-Rich Routing Engine

More features than OSRM: isochrones, time-dependent routing, truck routing, multi-modal transit, live traffic integration. **Use Valhalla when you need routing capabilities beyond simple A-to-B.**

**Why Tier 1:** Better than OSRM for enterprise use when you need isochrones (delivery zone calculation), truck routing, or time-dependent routing. The feature richness justifies the 2-5x slower response times.

**Caveats:**
- **Slower than OSRM for simple A-to-B routing.** ~2-5x slower for basic car routing.
- **Complex build and deployment.** C++ dependencies, data processing pipeline, configuration files.
- **Memory-intensive.** Planet-scale Valhalla graphs consume 50-100GB RAM.
- **Anti-pattern: Using Valhalla for simple directions.** If you only need point-to-point car routing, OSRM is faster and simpler.

---

## Tier 2 -- Specific Use Cases

---

### Django + GeoDjango -- Full-Featured Python Web Framework

Use GeoDjango when you need Django's admin UI, ORM, auth, and middleware AND spatial capabilities. **If you don't need the Django ecosystem, FastAPI + GeoAlchemy2 is leaner.**

**Why Tier 2:** Django powers Instagram, Pinterest; GeoDjango is used by government agencies. The admin UI is uniquely valuable for spatial data management. But the synchronous default limits throughput compared to FastAPI.

**Caveats:**
- **Synchronous by default.** Django's sync-first model is a bottleneck for I/O-heavy GIS APIs. ASGI + async views help but GeoDjango's spatial functions are still sync.
- **ORM overhead for spatial queries.** GeoDjango generates correct but sometimes suboptimal SQL. Complex queries often need raw SQL.
- **GDAL dependency.** GeoDjango requires GDAL/GEOS system libraries, which are painful to install.
- **Anti-pattern: Using Django for a purely API-driven GIS backend.** If you don't need Django's admin/ORM/middleware, FastAPI is leaner and faster.

---

### Express.js / Hono + PostGIS -- Node.js Spatial Backend

Use when your team is JavaScript/TypeScript end-to-end and you want the same language on frontend and backend. **No spatial ORM exists in Node.js -- you write raw SQL.**

**Why Tier 2:** Express is production-proven (4/5). The lack of spatial ORM is the main gap. Good for teams already committed to a JS/TS stack. Hono is newer (3/5) but excellent DX and performance.

**Caveats:**
- **No spatial ORM.** Unlike GeoAlchemy2 or GeoDjango, there's no ORM for PostGIS in Node.js. You write raw SQL with parameterized ST_ functions.
- **CPU-bound operations block event loop.** Heavy geometry processing blocks all other requests. Use worker threads.
- **JSON serialization of large GeoJSON.** `JSON.stringify` on 50MB GeoJSON blocks the event loop for seconds. Must stream responses.
- **Anti-pattern: Writing spatial SQL with string concatenation.** SQL injection vulnerability. Always use parameterized queries.

---

### pygeoapi -- Lightweight OGC API Server

Python-based OGC API Features and Tiles server. Much simpler than GeoServer when you need modern OGC API compliance without the Java burden.

**Why Tier 2:** Good middle ground when you need OGC compliance but want to avoid GeoServer's operational weight. Python ecosystem makes it easier to integrate with data science workflows.

**Caveats:**
- **Performance ceiling is lower than Martin/GeoServer.** Python single-threaded performance limits throughput.
- **Less feature-complete than GeoServer.** Newer OGC API standards are supported, but legacy WMS/WFS interoperability is thinner.
- **Anti-pattern: Using pygeoapi for high-throughput tile serving.** It's designed for feature/metadata APIs, not serving millions of tile requests per day.

---

### Nominatim -- OSM Geocoding

Powers OpenStreetMap's geocoding. The standard for self-hosted geocoding from OSM data.

**Why Tier 2:** 5/5 production-readiness -- powers OSM's geocoding. The infrastructure requirements (1TB+ SSD, 64GB RAM, days-long import) are the main barrier.

**Caveats:**
- **Self-hosting requires 800GB+ disk for planet data.** Initial import takes 2-4 days on good hardware. 64GB RAM recommended.
- **Geocoding quality depends on OSM data quality.** Rural areas and developing countries have sparse coverage.
- **No fuzzy matching by default.** "Bejing" won't find "Beijing" without custom configuration.
- **Rate limiting your own instance.** Single instance handles ~20-50 queries/second. Need multiple replicas for production traffic.
- **Anti-pattern: Hitting the public Nominatim API from production apps.** Violates their usage policy. Self-host or use a commercial geocoding API.

---

## Tier 3 / Legacy -- Specialized Needs Only

---

### pg_tileserv -- Zero-Config PostGIS Tiles

Point it at a database, it auto-discovers tables and serves tiles. **Development/exploration tool only. Use Martin for production.**

**Why Tier 3:** Martin surpasses it in every metric: performance, features, observability. Auto-discovery is useful for exploration but is a security risk in production (exposes all PostGIS tables as tile endpoints by default). No caching, no composite sources, no Prometheus metrics.

---

### pg_featureserv -- PostGIS Feature API

Zero-config GeoJSON feature serving from PostGIS. **Development/exploration tool only. Use FastAPI + PostGIS for production.**

**Why Tier 3:** Functionality fully covered by FastAPI + PostGIS or Express + PostGIS. Limited query capabilities. Less maintained than Martin. Useful only for quickly exposing PostGIS data without writing code during prototyping.

---

### Pelias -- Multi-Source Geocoding

Community-maintained multi-source geocoder (OSM + OpenAddresses + GeoNames). Originally built by Mapzen (defunct). **Consider hosted Geocode.earth instead of self-hosting.**

**Why Tier 3:** 5-7 microservices (API, placeholder, PIP, interpolation, libpostal, Elasticsearch) that must all be healthy. Elasticsearch alone wants 4-8GB heap. Total stack needs 16-32GB RAM. Community maintenance is slow since Mapzen's closure. The operational cost is only justified for multi-source global geocoding at scale.

---

### TileServer GL -- Raster Rendering from Vector Tiles

Unique ability to render vector tiles as raster images server-side (WMTS output). **Only use when you need raster tile output from vector styles** (legacy clients, print maps, static map images).

**Why Tier 3:** Niche tool. Raster rendering is CPU-intensive and slow. Node.js single-threaded. MBTiles only as input. Headless GL dependency is fragile. Not a general-purpose tile server.

---

### t-rex -- Rust Vector Tile Server

**Deprecated.** Project is no longer maintained. Functionality fully covered by Martin. Included only for historical reference -- migrate to Martin.

---

### pgRouting -- SQL-Based Routing

Routing extension for PostGIS. **Only use when your routing network is already in PostGIS and OSRM/Valhalla overhead isn't justified.** 10-100x slower than OSRM/Valhalla for customer-facing APIs. Useful for internal analytical routing queries within PostGIS workflows.

---

## Cloud Tile Services (Mapbox, MapTiler, Stadia Maps)

| Provider | Large-project fit | Small-project fit | Key Risk |
|----------|------------------|------------------|----------|
| **Mapbox** | 4/5 | 4/5 | Cost at scale, proprietary GL JS v2+ |
| **MapTiler** | 3/5 | 5/5 | Free tier limits, vendor coupling |
| **Stadia Maps** | 2/5 | 4/5 | Smaller ecosystem |

**Caveats (all cloud tile services):**
- **Cost at scale is the primary risk.** Free tiers are generous for development. Production traffic at 100K+ requests/day can cost $200-5000/month.
- **Vendor lock-in.** Mapbox GL JS v2+ is proprietary. Building on Mapbox's SDK makes migration expensive.
- **Data sovereignty concerns.** Tile requests go through the provider's infrastructure. May violate data residency requirements for government/defense projects.
- **Anti-pattern: Using Mapbox for a project that could use MapLibre + self-hosted tiles.** Evaluate total cost of ownership.
