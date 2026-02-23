# Backend Services

Geospatial backend services handle everything from tile generation and feature APIs to geocoding and routing. This guide covers the major open-source options, OGC-compliant services, and how to choose the right tools for your project.

> **Quick Picks**
> - **Vector tiles from PostGIS:** Martin (Rust, fastest) or pg_tileserv (Go, zero-config)
> - **Feature API from PostGIS:** pg_featureserv or FastAPI + GeoAlchemy2
> - **Raster tiles from COG:** TiTiler (Python, dynamic)
> - **Full OGC compliance:** GeoServer or pygeoapi
> - **Tile generation pipeline:** Tippecanoe (GeoJSON to PMTiles/MBTiles)

---

## Tile Servers

Tile servers generate and serve map tiles (raster or vector) from geospatial data sources. They are the backbone of most web mapping applications.

### GeoServer

The most feature-rich open-source geospatial server. Supports WMS, WFS, WMTS, WCS, and OGC API standards.

- **Language**: Java
- **Data Sources**: PostGIS, Shapefile, GeoTIFF, GeoPackage, and many more
- **Strengths**: Full OGC compliance, SLD styling, built-in admin UI
- **Weaknesses**: Heavy resource usage, Java dependency, complex configuration

### MapServer

A lightweight C-based map server with a long history in the GIS community.

- **Language**: C
- **Data Sources**: PostGIS, Shapefile, GeoTIFF, OGR/GDAL sources
- **Strengths**: Fast rendering, low memory footprint, Mapfile configuration
- **Weaknesses**: Less intuitive configuration, smaller community than GeoServer

### Martin

A fast, lightweight PostGIS vector tile server written in Rust.

- **Language**: Rust
- **Data Sources**: PostGIS, MBTiles, PMTiles
- **Strengths**: Extremely fast, low resource usage, simple setup
- **Weaknesses**: Vector tiles only, no raster support

### pg_tileserv

Serves vector tiles directly from PostGIS with minimal configuration.

- **Language**: Go
- **Data Sources**: PostGIS only
- **Strengths**: Zero-config, auto-discovers tables/functions, lightweight
- **Weaknesses**: PostGIS only, limited styling options

### Tile Server Comparison

| Server | Language | Vector Tiles | Raster Tiles | OGC Support | Admin UI | Best For |
|--------|----------|-------------|-------------|-------------|----------|----------|
| GeoServer | Java | Yes | Yes | Full | Yes | Enterprise / OGC compliance |
| MapServer | C | Limited | Yes | WMS/WFS | No | High-performance raster |
| Martin | Rust | Yes | No | No | No | Fast vector tile serving |
| pg_tileserv | Go | Yes | No | No | No | Quick PostGIS prototyping |
| [TiTiler](https://developmentseed.org/titiler/) | Python | No | Yes (dynamic from COG) | No | No | Dynamic raster tile serving from COGs |
| [Terracotta](https://terracotta-python.readthedocs.io/) | Python | No | Yes | No | No | Lightweight raster tiles |
| [pg_featureserv](https://access.crunchydata.com/documentation/pg_featureserv/) | Go | GeoJSON features | No | Partial (OGC API Features) | No | RESTful feature serving from PostGIS |

### pg_featureserv

Serves GeoJSON features directly from PostGIS tables and functions. Companion to pg_tileserv.

- **Language**: Go
- **Data Sources**: PostGIS only
- **Strengths**: Auto-discovers tables/functions, filtering, pagination, OGC API Features compatible
- **Weaknesses**: PostGIS only, read-only

### TiTiler: Dynamic Raster Tiles from COGs

TiTiler dynamically generates raster tiles from Cloud-Optimized GeoTIFFs, enabling serverless raster tile serving.

- **Language**: Python (FastAPI)
- **Data Sources**: COG files on S3, local, or any HTTP endpoint
- **Strengths**: Dynamic band math, colormaps, statistics endpoints, STAC integration
- **Weaknesses**: Requires compute for each request (no pre-rendering)

### Tippecanoe: Vector Tile Generation Pipeline

[Tippecanoe](https://github.com/felt/tippecanoe) converts GeoJSON/FlatGeobuf/CSV into optimized vector tilesets (PMTiles or MBTiles).

```bash
# Generate PMTiles with automatic zoom levels and simplification
tippecanoe -o buildings.pmtiles \
  --minimum-zoom=2 --maximum-zoom=14 \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  buildings.geojson
```

---

## API Frameworks

### FastAPI + GIS (Python)

FastAPI combined with libraries like Shapely, GeoPandas, and asyncpg for high-performance geospatial APIs.

```python
from fastapi import FastAPI
from geojson_pydantic import FeatureCollection

app = FastAPI()

@app.get("/features", response_model=FeatureCollection)
async def get_features(bbox: str = None):
    # Query PostGIS and return GeoJSON
    ...
```

- **Strengths**: Async, auto-generated docs, type safety, Python GIS ecosystem
- **Best for**: Modern Python GIS APIs

### Express + Turf.js (Node.js)

Express.js with Turf.js for server-side geospatial operations.

- **Strengths**: JavaScript full-stack, large npm ecosystem
- **Best for**: JavaScript teams, real-time apps with Socket.io

### Django + GeoDjango (Python)

Django's built-in GIS framework with ORM support for spatial queries.

- **Strengths**: Admin interface, ORM spatial lookups, mature ecosystem
- **Best for**: Data management apps, CRUD-heavy workflows

### FastAPI + SQLAlchemy + GeoAlchemy2 Pattern

A modern async Python API with PostGIS spatial queries via ORM.

```python
from fastapi import FastAPI, Query
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsGeoJSON, ST_MakeEnvelope, ST_Intersects
import json

app = FastAPI()
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/gisdb")

class Base(DeclarativeBase):
    pass

class Building(Base):
    __tablename__ = "buildings"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    geom = mapped_column(Geometry("POLYGON", srid=4326))

@app.get("/buildings")
async def get_buildings(
    bbox: str = Query(..., description="minx,miny,maxx,maxy")
):
    coords = [float(c) for c in bbox.split(",")]
    async with AsyncSession(engine) as session:
        result = await session.execute(
            select(Building.id, Building.name, ST_AsGeoJSON(Building.geom))
            .where(ST_Intersects(Building.geom, ST_MakeEnvelope(*coords, 4326)))
            .limit(1000)
        )
        return {"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"id": r[0], "name": r[1]},
             "geometry": json.loads(r[2])} for r in result
        ]}
```

### Express + Turf.js Example

```javascript
const express = require('express');
const turf = require('@turf/turf');

const app = express();

app.post('/api/buffer', express.json(), (req, res) => {
  const { geojson, distance } = req.body;
  const buffered = turf.buffer(geojson, distance, { units: 'kilometers' });
  res.json(buffered);
});

app.listen(3000);
```

---

## OGC Services

### Legacy Services

| Service | Full Name | Purpose |
|---------|-----------|---------|
| WMS | Web Map Service | Rendered map images |
| WFS | Web Feature Service | Vector feature access (GML/GeoJSON) |
| WMTS | Web Map Tile Service | Pre-rendered tile access |
| WCS | Web Coverage Service | Raster data access |
| WPS | Web Processing Service | Server-side geoprocessing |

### Modern OGC APIs

The OGC is transitioning from XML-based services to RESTful JSON APIs:

| OGC API | Replaces | Format | Status |
|---------|----------|--------|--------|
| OGC API - Features | WFS | GeoJSON | Approved |
| OGC API - Tiles | WMTS | MVT / PNG | Approved |
| OGC API - Maps | WMS | PNG / JPEG | Draft |
| OGC API - Processes | WPS | JSON | Approved |
| OGC API - Coverages | WCS | COG / GeoTIFF | Draft |
| OGC API - Records | CSW | JSON | Draft |

**Specifications:** [ogcapi.ogc.org](https://ogcapi.ogc.org/)

**Implementations:** [pygeoapi](https://pygeoapi.io/) (Python), [ldproxy](https://github.com/interactive-instruments/ldproxy) (Java), [GeoServer 2.24+](https://geoserver.org/) (OGC API module), [QGIS Server](https://qgis.org/en/site/about/features.html)

---

## Geocoding Services

Geocoding converts addresses to coordinates (forward) and coordinates to addresses (reverse).

| Service | Type | Self-Hosted | API Limit | Best For |
|---------|------|-------------|-----------|----------|
| Nominatim | Forward/Reverse | Yes | Rate-limited | OpenStreetMap-based geocoding |
| Pelias | Forward/Reverse | Yes | Unlimited (self-hosted) | Custom deployments |
| Photon | Forward/Reverse | Yes | Unlimited (self-hosted) | Fast autocomplete |
| Mapbox Geocoding | Forward/Reverse | No | Pay-per-request | Commercial apps |
| Google Geocoding | Forward/Reverse | No | Pay-per-request | High accuracy, global coverage |
**Self-hosting Nominatim:** `docker run -it -e PBF_URL=https://download.geofabrik.de/europe-latest.osm.pbf mediagis/nominatim:4.4`

**Self-hosting Pelias:** See [pelias/docker](https://github.com/pelias/docker) for Docker Compose setup with OSM + OpenAddresses data.

---

## Routing Engines

Routing engines calculate paths between points on a road network.

| Engine | Language | Algorithm | Features | Best For |
|--------|----------|-----------|----------|----------|
| OSRM | C++ | Contraction Hierarchies | Fast, car/bike/foot profiles | High-performance routing |
| Valhalla | C++ | Bidirectional A* | Turn-by-turn, isochrones, map matching | Full-featured routing |
| GraphHopper | Java | CH + A* | Multiple profiles, elevation | Flexible routing with elevation |
| pgRouting | SQL (C) | Dijkstra, A*, etc. | SQL-based, PostGIS integration | Database-integrated routing |
**OSRM quick start:** `docker run -t -v "${PWD}:/data" ghcr.io/project-osrm/osrm-backend osrm-extract -p /opt/car.lua /data/map.osm.pbf && docker run -t -p 5000:5000 ghcr.io/project-osrm/osrm-backend osrm-routed /data/map.osrm`

**Route request:** `GET http://localhost:5000/route/v1/driving/13.388860,52.517037;13.397634,52.529407?overview=full`

---

## Service Comparison

| Service | Type | Language | OGC Support | Self-Hosted | Best For |
|---------|------|----------|-------------|-------------|----------|
| GeoServer | Tile/Feature Server | Java | Full | Yes | Enterprise OGC compliance |
| Martin | Tile Server | Rust | No | Yes | Fast vector tiles |
| pg_tileserv | Tile Server | Go | No | Yes | Quick PostGIS prototyping |
| FastAPI | API Framework | Python | Via plugins | Yes | Modern Python APIs |
| GeoDjango | API Framework | Python | Via plugins | Yes | Admin-heavy apps |
| Nominatim | Geocoding | C++ | No | Yes | OSM-based geocoding |
| OSRM | Routing | C++ | No | Yes | High-performance routing |
| Valhalla | Routing | C++ | No | Yes | Full-featured routing |
### Resource Requirements

| Service | Min RAM | CPU | Disk | Setup Complexity |
|---|---|---|---|---|
| GeoServer | 2 GB | 2 cores | 1 GB+ | Medium (Java, config) |
| Martin | 128 MB | 1 core | Minimal | Low (single binary) |
| pg_tileserv | 64 MB | 1 core | Minimal | Very Low (auto-discover) |
| pg_featureserv | 64 MB | 1 core | Minimal | Very Low (auto-discover) |
| TiTiler | 512 MB | 1 core | Minimal | Low (pip install) |
| Nominatim | 2 GB+ | 2 cores | 50 GB+ (planet) | High (data import) |
| OSRM | 4 GB+ | 2 cores | 10 GB+ | Medium (data preprocessing) |
