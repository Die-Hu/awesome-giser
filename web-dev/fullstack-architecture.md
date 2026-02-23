# Full-Stack GIS Architecture

Designing a geospatial web application requires careful consideration of how data flows from ingestion to rendering. This guide covers architecture patterns, common technology stacks, API design strategies, and reference architectures to help you build scalable GIS applications.

> **Quick Picks**
> - **Modern full-stack:** PostGIS + Martin + MapLibre + React/Next.js
> - **Zero-infrastructure:** S3/R2 COG + PMTiles + static frontend
> - **Enterprise/OGC:** PostGIS + GeoServer + Leaflet
> - **Fastest prototyping:** DuckDB + PMTiles + Deck.gl
> - **Standards-based API:** OGC API Features/Tiles (replacing WMS/WFS)

---

## Architecture Patterns

### Monolith vs Microservices for GIS

Choosing between a monolithic and microservices architecture depends on team size, data complexity, and scalability requirements.

**Monolithic Architecture**

A single application handles data ingestion, processing, serving, and rendering. Suitable for smaller teams and simpler use cases.

```
┌─────────────────────────────────────────┐
│            Monolith Application          │
│  ┌──────┐ ┌──────┐ ┌───────┐ ┌───────┐ │
│  │Ingest│ │Store │ │ Serve │ │Render │ │
│  └──────┘ └──────┘ └───────┘ └───────┘ │
│            ┌──────────┐                  │
│            │ PostGIS  │                  │
│            └──────────┘                  │
└─────────────────────────────────────────┘
```

**Microservices Architecture**

Each concern is a separate service with its own scaling profile. Suitable for large teams and complex workloads.

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  Ingest  │  │  Tile    │  │  API     │  │ Frontend │
│  Service │  │  Server  │  │  Gateway │  │  App     │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │              │
     ▼             ▼             ▼              ▼
┌──────────────────────────────────────────────────────┐
│                    PostGIS / Object Store             │
└──────────────────────────────────────────────────────┘
```

### Decision Criteria

Choose **monolith** when: small team (<5), single data source, simple read-heavy workload, quick time-to-market.

Choose **microservices** when: separate scaling for tiles vs API, multiple data sources (PostGIS + S3 + external APIs), team can own individual services, need independent deployment.

---

## Common Stacks

### Stack 1: PostGIS + GeoServer + Leaflet

The traditional open-source GIS stack. Well-documented, widely used, and standards-compliant.

- **Database**: PostgreSQL / PostGIS
- **Tile Server**: GeoServer (WMS, WFS, WMTS)
- **Frontend**: Leaflet with plugins
- **Best for**: Enterprise GIS, OGC-compliant systems

### Stack 2: PostGIS + FastAPI + MapLibre

A modern Python-based stack optimized for vector tiles and developer experience.

- **Database**: PostgreSQL / PostGIS
- **API / Tiles**: FastAPI + pg_tileserv or Martin
- **Frontend**: MapLibre GL JS
- **Best for**: Modern web apps, vector tile workflows

### Stack 3: DuckDB + PMTiles + Deck.gl

A serverless-friendly stack for large-scale data visualization.

- **Database**: DuckDB (with spatial extension) or flat files (GeoParquet)
- **Tiles**: Pre-generated PMTiles on CDN
- **Frontend**: Deck.gl or MapLibre GL JS
- **Best for**: Data visualization, static hosting, analytics dashboards

### Stack 4: PostGIS + Django/GeoDjango + OpenLayers

A full-featured Python web framework with built-in GIS support.

- **Database**: PostgreSQL / PostGIS
- **API**: Django + GeoDjango (admin, ORM, REST framework)
- **Frontend**: OpenLayers
- **Best for**: Data management apps, admin-heavy workflows

### Stack 5: PostGIS + Martin + MapLibre + Next.js (Modern Full-Stack)

The most popular 2025 stack for production web GIS applications.

- **Database**: PostgreSQL / PostGIS
- **Tile Server**: Martin (Rust, blazing fast vector tiles)
- **Frontend**: Next.js + MapLibre GL JS + react-map-gl
- **API**: Next.js API routes or separate FastAPI service
- **Best for**: Production apps, SSR-friendly, modern DX

### Stack 6: S3 COG + PMTiles + Static Hosting (Serverless)

Zero-infrastructure pattern for data visualization and analysis dashboards.

- **Raster**: Cloud-Optimized GeoTIFF on S3/R2, served via TiTiler or direct range requests
- **Vector**: PMTiles on CDN (Cloudflare R2 for free egress)
- **Frontend**: Static site (Vite + MapLibre or Deck.gl), deployed to Vercel/Netlify
- **Best for**: Public data portals, cost-sensitive projects, static datasets

### OGC API Pattern (Modern Standards)

The modern replacement for WMS/WFS/WMTS. RESTful, JSON-based, OpenAPI-documented.

- **Server**: pygeoapi, ldproxy, or GeoServer 2.24+
- **Standards**: OGC API - Features (GeoJSON), OGC API - Tiles (MVT/PNG)
- **Frontend**: Any map library (MapLibre, Leaflet, OpenLayers)
- **Best for**: Government, INSPIRE compliance, interoperable systems

---

## Stack Comparison

| Stack | Complexity | Scalability | Cost | Best For |
|-------|-----------|-------------|------|----------|
| PostGIS + GeoServer + Leaflet | Medium | Medium | Low | Enterprise / OGC compliance |
| PostGIS + FastAPI + MapLibre | Medium | High | Low | Modern vector tile apps |
| DuckDB + PMTiles + Deck.gl | Low | High | Very Low | Serverless / static hosting |
| PostGIS + Django + OpenLayers | Medium | Medium | Low | Admin-heavy / data management |
| PostGIS + Martin + MapLibre + Next.js | Medium | High | Low | Modern production apps |
| S3 COG + PMTiles + Static | Very Low | Very High | Very Low | Serverless / static data |
| pygeoapi + PostGIS | Medium | Medium | Low | OGC API compliance |

### Monthly Cost Estimates (Small-Medium Scale)

| Stack | Hosting | Database | CDN | Total/month |
|---|---|---|---|---|
| PostGIS + Martin + Next.js | $20 (VPS) | Included | $0-10 | ~$20-30 |
| DuckDB + PMTiles + Vercel | $0 (Vercel free) | N/A | $0 (R2) | ~$0-5 |
| PostGIS + GeoServer | $40 (needs more RAM) | Included | $0-10 | ~$40-50 |
| Managed (AWS RDS + ECS) | $50-100 | $30-50 | $10-30 | ~$90-180 |

---

## Reference Architecture

### Typical Web GIS Application

```
                    ┌─────────────┐
                    │   Browser   │
                    │  (MapLibre/ │
                    │   Leaflet)  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   CDN /     │
                    │   Nginx     │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌─────▼─────┐  ┌──▼───────┐
       │  Tile   │  │  Feature  │  │  Search  │
       │  Server │  │  API      │  │  API     │
       │ (Martin)│  │ (FastAPI) │  │(Pelias)  │
       └────┬────┘  └─────┬─────┘  └──┬───────┘
            │             │            │
            └─────────────┼────────────┘
                          │
                   ┌──────▼──────┐
                   │   PostGIS   │
                   └─────────────┘
```

---

## API Design for Geospatial

### REST vs GraphQL vs gRPC

| Approach | Strengths | Weaknesses | Best For |
|----------|-----------|------------|----------|
| REST | Simple, cacheable, well-tooled | Over-fetching, many endpoints | Public APIs, tile serving |
| GraphQL | Flexible queries, single endpoint | Complex spatial queries, caching | Feature-rich apps |
| gRPC | High performance, streaming | Browser support, debugging | Service-to-service |
| OGC API | Standards-based, interoperable | Less flexible | Government / enterprise |
### Example Endpoints

**REST:**
```
GET /api/features?bbox=-122.5,37.7,-122.3,37.8&limit=100
GET /api/tiles/{z}/{x}/{y}.mvt
```

**OGC API - Features:**
```
GET /collections/buildings/items?bbox=-122.5,37.7,-122.3,37.8&limit=100
GET /collections/buildings/items/{featureId}
```

**OGC API - Tiles:**
```
GET /collections/buildings/tiles/WebMercatorQuad/{z}/{x}/{y}?f=mvt
```

### OGC APIs

Modern OGC APIs are RESTful and JSON-based, replacing legacy XML services:

- **OGC API - Features**: Replaces WFS, serves GeoJSON
- **OGC API - Tiles**: Replaces WMTS, serves vector/raster tiles
- **OGC API - Maps**: Replaces WMS, serves rendered map images
- **OGC API - Processes**: Replaces WPS, executes server-side operations

---

## Data Flow

A typical geospatial data pipeline follows five stages:

```
Ingest  ──▶  Validate  ──▶  Store  ──▶  Serve  ──▶  Render
  │            │              │           │            │
  │            │              │           │            │
Shapefile   Schema check   PostGIS    Tile server   MapLibre
GeoJSON     CRS transform  S3/COG     REST API     Leaflet
GeoPackage  Topology fix   GeoParquet OGC API      Deck.gl
CSV/GPS     Deduplication  DuckDB     GraphQL      OpenLayers
```

### Stage Details

1. **Ingest**: Accept data from uploads, APIs, sensors, or ETL pipelines
2. **Validate**: Check schema, transform CRS, fix topology, deduplicate
3. **Store**: Write to PostGIS, object storage (COG/PMTiles), or file formats (GeoParquet)
4. **Serve**: Expose via tile server, feature API, or OGC-compliant service
5. **Render**: Display in browser using WebGL-based or SVG-based map libraries

### Ingest Example (Python)

```python
# Upload a Shapefile, validate, and store in PostGIS
import geopandas as gpd
from sqlalchemy import create_engine

gdf = gpd.read_file("upload.shp")
gdf = gdf.to_crs(epsg=4326)  # Normalize CRS
assert gdf.geometry.is_valid.all(), "Invalid geometries found"
engine = create_engine("postgresql://user:pass@localhost/gisdb")
gdf.to_postgis("features", engine, if_exists="append")
```
