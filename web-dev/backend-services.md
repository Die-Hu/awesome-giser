# Backend Services — 2025 Complete Guide

Geospatial backend services handle everything from tile generation and feature APIs to geocoding and routing. This guide covers the major open-source options, OGC-compliant services, message queues, and production deployment patterns.

> **Quick Picks**
> - **Vector tiles from PostGIS:** Martin 0.15+ (Rust, fastest) or pg_tileserv (Go, zero-config)
> - **Feature API from PostGIS:** pg_featureserv or FastAPI + GeoAlchemy2
> - **Raster tiles from COG:** TiTiler 0.18+ (Python, dynamic)
> - **Full OGC compliance:** GeoServer 2.25 or pygeoapi
> - **Tile generation pipeline:** Tippecanoe 2.x (GeoJSON to PMTiles/MBTiles)
> - **Geocoding:** Nominatim (OSM) or Pelias (multi-source)
> - **Routing:** OSRM (fastest) or Valhalla (most features)

---

## Tile Servers Deep Dive

### Martin 0.15+ — Rust Vector Tile Server

Martin is the fastest open-source vector tile server, written in Rust. It serves MVT tiles directly from PostGIS tables, functions, MBTiles, and PMTiles files.

```yaml
# martin-config.yaml — Production configuration
listen_addresses: '0.0.0.0:3000'
worker_processes: 8
keep_alive: 75
cache_size_mb: 512

# Connection pool
postgres:
  connection_string: 'postgresql://user:pass@localhost:5432/gisdb'
  pool_size: 20
  auto_publish:
    tables:
      from_schemas: ['public', 'spatial']
      id_columns: ['gid', 'id']
    functions:
      from_schemas: ['public']

# Composite sources
composite_sources:
  basemap:
    - source: buildings
    - source: roads
    - source: water

# Sprite and font support
sprites:
  paths:
    - /data/sprites
fonts:
  paths:
    - /data/fonts

# PMTiles file sources
pmtiles:
  paths:
    - /data/tiles/
  sources:
    boundaries: /data/tiles/boundaries.pmtiles

# Health check
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

# Available endpoints:
# GET /catalog — list all sources
# GET /{source}/{z}/{x}/{y} — get tile
# GET /health — health check
# GET /sprite/{name}.json — sprite JSON
# GET /sprite/{name}.png — sprite image
# GET /font/{name}/{start}-{end}.pbf — font glyphs
```

#### Martin Function Sources — Dynamic Tiles

```sql
-- Create a PostGIS function for dynamic tile generation
CREATE OR REPLACE FUNCTION public.buildings_by_type(
    z integer, x integer, y integer,
    building_type text DEFAULT 'residential'
)
RETURNS bytea AS $$
DECLARE
    result bytea;
    bounds geometry;
BEGIN
    bounds := ST_TileEnvelope(z, x, y);

    SELECT ST_AsMVT(q, 'buildings', 4096, 'geom') INTO result
    FROM (
        SELECT
            id,
            name,
            height,
            type,
            ST_AsMVTGeom(
                ST_Transform(geom, 3857),
                bounds,
                4096, 64, true
            ) AS geom
        FROM buildings
        WHERE geom && ST_Transform(bounds, 4326)
          AND type = building_type
          AND (z >= 14 OR height > 50)
          AND (z >= 12 OR height > 100)
    ) q;

    RETURN result;
END;
$$ LANGUAGE plpgsql STABLE PARALLEL SAFE;

-- Martin auto-discovers this as:
-- GET /function_source/buildings_by_type/{z}/{x}/{y}?building_type=commercial
```

### pg_tileserv — Zero-Config PostGIS Tiles

```bash
# Simplest possible tile server
export DATABASE_URL=postgresql://user:pass@localhost/gisdb
docker run -d --name pg_tileserv \
  -e DATABASE_URL \
  -p 7800:7800 \
  pramsey/pg_tileserv

# Auto-discovers all PostGIS tables and functions
# GET /index.json — catalog
# GET /public.buildings/{z}/{x}/{y}.pbf — table tiles
# GET /public.buildings_by_type/{z}/{x}/{y}.pbf?type=residential — function tiles
```

### TiTiler 0.18+ — Dynamic Raster Tiles from COG

```python
# Custom TiTiler application with STAC integration
from titiler.core.factory import TilerFactory
from titiler.extensions import stacExtension
from titiler.mosaic.factory import MosaicTilerFactory
from fastapi import FastAPI, Query
import uvicorn

app = FastAPI(title="GIS Raster Tile Server")

# COG endpoint: /cog/tiles/{z}/{x}/{y}.png
cog = TilerFactory(router_prefix="/cog")
app.include_router(cog.router, prefix="/cog")

# Mosaic endpoint: /mosaic/tiles/{z}/{x}/{y}.png
mosaic = MosaicTilerFactory(router_prefix="/mosaic")
app.include_router(mosaic.router, prefix="/mosaic")

# Custom NDVI endpoint
@app.get("/ndvi/{z}/{x}/{y}.png")
async def ndvi_tile(
    z: int, x: int, y: int,
    url: str = Query(..., description="COG URL"),
):
    from rio_tiler.io import Reader
    from rio_tiler.colormap import cmap as colormap

    with Reader(url) as src:
        img = src.tile(x, y, z, expression="(b5-b4)/(b5+b4)")
        img = img.rescale((-1, 1), (0, 255))
        img = img.apply_colormap(colormap.get("rdylgn"))
        return img.render(img_format="PNG")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

```bash
# TiTiler endpoints:
# GET /cog/tiles/{z}/{x}/{y}.png?url=https://s3.example.com/data.tif
# GET /cog/info?url=https://s3.example.com/data.tif
# GET /cog/statistics?url=https://s3.example.com/data.tif
# GET /cog/preview.png?url=https://s3.example.com/data.tif&rescale=0,3000&colormap_name=viridis
# GET /cog/point/{lng}/{lat}?url=https://s3.example.com/data.tif
```

### Tippecanoe 2.x — Advanced Tile Generation

```bash
# Recipe 1: Buildings with zoom-dependent detail
tippecanoe -o buildings.pmtiles \
  --minimum-zoom=10 --maximum-zoom=15 \
  --simplification=10 \
  --drop-smallest-as-needed \
  --include=name,height,type,area \
  --layer=buildings \
  buildings.geojson

# Recipe 2: Roads with importance-based filtering
tippecanoe -o roads.pmtiles \
  --minimum-zoom=4 --maximum-zoom=14 \
  -j '{ "roads": { "minzoom": { "*": 4, "motorway": 4, "trunk": 6, "primary": 8, "secondary": 10, "tertiary": 12, "residential": 13 } } }' \
  --coalesce-smallest-as-needed \
  roads.geojson

# Recipe 3: POI with clustering at low zooms
tippecanoe -o poi.pmtiles \
  --minimum-zoom=2 --maximum-zoom=14 \
  --cluster-distance=50 \
  --cluster-maxzoom=12 \
  --accumulate-attribute=count:sum \
  --include=name,category,count \
  poi.geojson

# Recipe 4: Multiple layers merged into single tileset
tippecanoe -o basemap.pmtiles \
  --named-layer=buildings:buildings.geojson \
  --named-layer=roads:roads.geojson \
  --named-layer=water:water.geojson \
  --named-layer=landuse:landuse.geojson \
  --minimum-zoom=2 --maximum-zoom=14

# Recipe 5: Convert from FlatGeobuf (faster than GeoJSON)
tippecanoe -o output.pmtiles \
  --minimum-zoom=0 --maximum-zoom=14 \
  --read-parallel \
  input.fgb

# Recipe 6: Generate tile statistics
tippecanoe -o output.pmtiles \
  --stats \
  --minimum-zoom=0 --maximum-zoom=14 \
  input.geojson 2>&1 | tee tile-stats.log
```

### GeoServer 2.25 — Enterprise OGC Server

```bash
# Docker with plugins
docker run -d --name geoserver \
  -p 8080:8080 \
  -e GEOSERVER_ADMIN_PASSWORD=mysecret \
  -v geoserver_data:/opt/geoserver/data_dir \
  kartoza/geoserver:2.25.0

# REST API for automation
# Create workspace
curl -u admin:mysecret -X POST http://localhost:8080/geoserver/rest/workspaces \
  -H "Content-Type: application/json" \
  -d '{"workspace":{"name":"gis"}}'

# Create PostGIS datastore
curl -u admin:mysecret -X POST http://localhost:8080/geoserver/rest/workspaces/gis/datastores \
  -H "Content-Type: application/json" \
  -d '{
    "dataStore": {
      "name": "postgis",
      "connectionParameters": {
        "entry": [
          {"@key": "host", "$": "postgis"},
          {"@key": "port", "$": "5432"},
          {"@key": "database", "$": "gisdb"},
          {"@key": "user", "$": "gisuser"},
          {"@key": "passwd", "$": "password"},
          {"@key": "dbtype", "$": "postgis"}
        ]
      }
    }
  }'

# Publish a layer
curl -u admin:mysecret -X POST \
  http://localhost:8080/geoserver/rest/workspaces/gis/datastores/postgis/featuretypes \
  -H "Content-Type: application/json" \
  -d '{"featureType":{"name":"buildings","nativeName":"buildings","srs":"EPSG:4326"}}'
```

### Tile Server Comparison

| Server | Language | Vector | Raster | OGC | Speed | Memory | Best For |
|--------|----------|--------|--------|-----|-------|--------|----------|
| Martin | Rust | MVT | No | No | Fastest | 128 MB | Production vector tiles |
| pg_tileserv | Go | MVT | No | No | Fast | 64 MB | Quick prototyping |
| TiTiler | Python | No | Dynamic COG | No | Medium | 512 MB | Cloud raster serving |
| GeoServer | Java | MVT+PNG | WMS/WMTS | Full | Medium | 2 GB+ | Enterprise OGC |
| MapServer | C | Limited | WMS | WMS/WFS | Fast | 256 MB | Legacy raster serving |
| Terracotta | Python | No | Static | No | Fast | 256 MB | Simple raster tiles |

---

## API Frameworks

### FastAPI + GeoAlchemy2 — Complete CRUD API

```python
# app/main.py — Production-ready spatial API
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import select, func, text
from geoalchemy2 import Geometry
from geoalchemy2.functions import ST_AsGeoJSON, ST_MakeEnvelope, ST_Intersects
from geoalchemy2.functions import ST_Buffer, ST_Area, ST_Centroid, ST_Transform
from pydantic import BaseModel, Field
from typing import Optional
import json

app = FastAPI(title="Spatial Feature API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost/gisdb",
    pool_size=20,
    max_overflow=10,
    pool_timeout=30,
)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

class Feature(Base):
    __tablename__ = "features"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str]
    category: Mapped[Optional[str]]
    value: Mapped[Optional[float]]
    geom = mapped_column(Geometry("GEOMETRY", srid=4326))

async def get_db():
    async with SessionLocal() as session:
        yield session

# --- Endpoints ---

@app.get("/features")
async def list_features(
    bbox: str = Query(None, description="minx,miny,maxx,maxy", example="-122.5,37.7,-122.3,37.9"),
    category: str = Query(None),
    limit: int = Query(1000, le=10000),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
):
    query = select(
        Feature.id, Feature.name, Feature.category, Feature.value,
        ST_AsGeoJSON(Feature.geom).label('geometry')
    )

    if bbox:
        coords = [float(c) for c in bbox.split(",")]
        query = query.where(ST_Intersects(Feature.geom, ST_MakeEnvelope(*coords, 4326)))

    if category:
        query = query.where(Feature.category == category)

    query = query.limit(limit).offset(offset)
    result = await db.execute(query)

    features = [{
        "type": "Feature",
        "properties": {"id": r.id, "name": r.name, "category": r.category, "value": r.value},
        "geometry": json.loads(r.geometry),
    } for r in result]

    return {"type": "FeatureCollection", "features": features}

@app.get("/features/{feature_id}")
async def get_feature(feature_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Feature.id, Feature.name, Feature.category, Feature.value,
               ST_AsGeoJSON(Feature.geom).label('geometry'))
        .where(Feature.id == feature_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Feature not found")
    return {
        "type": "Feature",
        "properties": {"id": row.id, "name": row.name, "category": row.category, "value": row.value},
        "geometry": json.loads(row.geometry),
    }

@app.get("/features/nearest")
async def nearest_features(
    lng: float = Query(...), lat: float = Query(...),
    radius_m: float = Query(1000), limit: int = Query(10),
    db: AsyncSession = Depends(get_db),
):
    point = func.ST_SetSRID(func.ST_MakePoint(lng, lat), 4326)
    distance = func.ST_Distance(func.ST_Transform(Feature.geom, 3857), func.ST_Transform(point, 3857))

    result = await db.execute(
        select(Feature.id, Feature.name, ST_AsGeoJSON(Feature.geom).label('geometry'),
               distance.label('distance_m'))
        .where(func.ST_DWithin(
            func.ST_Transform(Feature.geom, 3857),
            func.ST_Transform(point, 3857),
            radius_m
        ))
        .order_by(distance)
        .limit(limit)
    )

    return {"type": "FeatureCollection", "features": [
        {"type": "Feature",
         "properties": {"id": r.id, "name": r.name, "distance_m": round(r.distance_m, 1)},
         "geometry": json.loads(r.geometry)}
        for r in result
    ]}

@app.post("/features/buffer")
async def buffer_feature(
    feature_id: int = Query(...),
    distance_m: float = Query(..., description="Buffer distance in meters"),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(ST_AsGeoJSON(
            ST_Transform(
                ST_Buffer(ST_Transform(Feature.geom, 3857), distance_m),
                4326
            )
        ).label('geometry'))
        .where(Feature.id == feature_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Feature not found")
    return {"type": "Feature", "geometry": json.loads(row.geometry), "properties": {}}

@app.get("/statistics")
async def spatial_statistics(
    bbox: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    coords = [float(c) for c in bbox.split(",")]
    envelope = ST_MakeEnvelope(*coords, 4326)

    result = await db.execute(
        select(
            func.count(Feature.id).label('count'),
            func.avg(Feature.value).label('avg_value'),
            func.sum(func.ST_Area(func.ST_Transform(Feature.geom, 3857))).label('total_area_m2'),
            ST_AsGeoJSON(func.ST_Centroid(func.ST_Collect(Feature.geom))).label('centroid'),
        )
        .where(ST_Intersects(Feature.geom, envelope))
    )
    row = result.first()
    return {
        "count": row.count,
        "avg_value": round(row.avg_value, 2) if row.avg_value else None,
        "total_area_m2": round(row.total_area_m2, 2) if row.total_area_m2 else None,
        "centroid": json.loads(row.centroid) if row.centroid else None,
    }
```

### Django + GeoDjango — Admin + Spatial ORM

```python
# models.py
from django.contrib.gis.db import models

class Feature(models.Model):
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=100, blank=True)
    value = models.FloatField(null=True)
    geom = models.GeometryField(srid=4326)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['category']),]

    def __str__(self):
        return self.name

# admin.py
from django.contrib.gis import admin

@admin.register(Feature)
class FeatureAdmin(admin.GISModelAdmin):
    list_display = ['name', 'category', 'value', 'created_at']
    list_filter = ['category']
    search_fields = ['name']
    default_lon = 116.4
    default_lat = 39.9
    default_zoom = 10

# serializers.py
from rest_framework_gis.serializers import GeoFeatureModelSerializer

class FeatureSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = Feature
        geo_field = 'geom'
        fields = ['id', 'name', 'category', 'value']

# views.py
from rest_framework_gis.filters import InBBoxFilter, DistanceToPointFilter
from rest_framework import viewsets

class FeatureViewSet(viewsets.ModelViewSet):
    queryset = Feature.objects.all()
    serializer_class = FeatureSerializer
    bbox_filter_field = 'geom'
    filter_backends = [InBBoxFilter]

    def get_queryset(self):
        qs = super().get_queryset()
        category = self.request.query_params.get('category')
        if category:
            qs = qs.filter(category=category)
        return qs

# Spatial ORM queries
from django.contrib.gis.geos import Point, Polygon
from django.contrib.gis.measure import D

# Find features within 5km of a point
point = Point(116.4, 39.9, srid=4326)
nearby = Feature.objects.filter(geom__distance_lte=(point, D(km=5)))

# Find features within a bbox
bbox = Polygon.from_bbox((116.0, 39.5, 117.0, 40.5))
in_bbox = Feature.objects.filter(geom__within=bbox)

# Spatial aggregation
from django.contrib.gis.db.models.functions import Area, Centroid
stats = Feature.objects.filter(category='park').aggregate(
    total_area=Area('geom'),
    center=Centroid('geom'),
)
```

### Express + TypeScript + PostGIS

```typescript
// src/index.ts
import express from 'express';
import { Pool } from 'pg';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// GeoJSON Features endpoint
app.get('/api/features', async (req, res) => {
  const { bbox, category, limit = '1000' } = req.query;
  const params: (string | number)[] = [];
  let where = 'WHERE 1=1';

  if (bbox) {
    const [minx, miny, maxx, maxy] = (bbox as string).split(',').map(Number);
    params.push(minx, miny, maxx, maxy);
    where += ` AND geom && ST_MakeEnvelope($${params.length-3}, $${params.length-2}, $${params.length-1}, $${params.length}, 4326)`;
  }

  if (category) {
    params.push(category as string);
    where += ` AND category = $${params.length}`;
  }

  params.push(Number(limit));

  const result = await pool.query(`
    SELECT jsonb_build_object(
      'type', 'FeatureCollection',
      'features', COALESCE(jsonb_agg(jsonb_build_object(
        'type', 'Feature',
        'properties', jsonb_build_object('id', id, 'name', name, 'category', category, 'value', value),
        'geometry', ST_AsGeoJSON(geom)::jsonb
      )), '[]'::jsonb)
    ) AS geojson
    FROM (SELECT * FROM features ${where} LIMIT $${params.length}) sub
  `, params);

  res.json(result.rows[0].geojson);
});

// Bulk GeoJSON import
app.post('/api/features/import', async (req, res) => {
  const { features } = req.body;
  if (!features?.length) return res.status(400).json({ error: 'No features' });

  const client = await pool.connect();
  try {
    await client.query('BEGIN');
    let count = 0;
    for (const f of features) {
      await client.query(
        `INSERT INTO features (name, category, value, geom)
         VALUES ($1, $2, $3, ST_SetSRID(ST_GeomFromGeoJSON($4), 4326))`,
        [f.properties.name, f.properties.category, f.properties.value, JSON.stringify(f.geometry)]
      );
      count++;
    }
    await client.query('COMMIT');
    res.json({ imported: count });
  } catch (err) {
    await client.query('ROLLBACK');
    res.status(500).json({ error: (err as Error).message });
  } finally {
    client.release();
  }
});

app.listen(3000, () => console.log('API running on :3000'));
```

### Hono + Drizzle + PostGIS — Edge-Ready

```typescript
// src/index.ts — Cloudflare Workers or Node.js
import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import { sql } from 'drizzle-orm';

const app = new Hono();
app.use('/*', cors());

const client = postgres(process.env.DATABASE_URL!);
const db = drizzle(client);

app.get('/api/features', async (c) => {
  const bbox = c.req.query('bbox');
  const limit = Number(c.req.query('limit') ?? 1000);

  let query = sql`
    SELECT jsonb_build_object(
      'type', 'FeatureCollection',
      'features', COALESCE(jsonb_agg(jsonb_build_object(
        'type', 'Feature',
        'properties', jsonb_build_object('id', id, 'name', name),
        'geometry', ST_AsGeoJSON(geom)::jsonb
      )), '[]'::jsonb)
    ) AS geojson
    FROM features
  `;

  if (bbox) {
    const [minx, miny, maxx, maxy] = bbox.split(',').map(Number);
    query = sql`
      SELECT jsonb_build_object(
        'type', 'FeatureCollection',
        'features', COALESCE(jsonb_agg(jsonb_build_object(
          'type', 'Feature',
          'properties', jsonb_build_object('id', id, 'name', name),
          'geometry', ST_AsGeoJSON(geom)::jsonb
        )), '[]'::jsonb)
      ) AS geojson
      FROM (
        SELECT * FROM features
        WHERE geom && ST_MakeEnvelope(${minx}, ${miny}, ${maxx}, ${maxy}, 4326)
        LIMIT ${limit}
      ) sub
    `;
  }

  const result = await db.execute(query);
  return c.json(result[0].geojson);
});

export default app;
```

---

## OGC Services & pygeoapi

### pygeoapi Configuration

```yaml
# pygeoapi-config.yml
server:
  bind:
    host: 0.0.0.0
    port: 5000
  url: https://api.example.com
  mimetype: application/json
  encoding: utf-8
  language: en-US

metadata:
  identification:
    title: GIS Feature API
    description: OGC API Features for spatial data

resources:
  buildings:
    type: collection
    title: Buildings
    description: Building footprints
    keywords: [buildings, footprints]
    crs:
      - http://www.opengis.net/def/crs/OGC/1.3/CRS84
    providers:
      - type: feature
        name: PostgreSQL
        data:
          host: postgis
          dbname: gisdb
          user: gisuser
          password: ${DB_PASSWORD}
          search_path: [public]
        id_field: id
        table: buildings
        geom_field: geom

  elevation:
    type: collection
    title: Elevation
    description: Digital Elevation Model
    providers:
      - type: coverage
        name: rasterio
        data: /data/elevation.tif
        format:
          name: GTiff
          mimetype: image/tiff

  buffer:
    type: process
    processor:
      name: buffer-process
```

```bash
# Run pygeoapi with Docker
docker run -d --name pygeoapi \
  -p 5000:80 \
  -v $(pwd)/pygeoapi-config.yml:/pygeoapi/local.config.yml \
  geopython/pygeoapi:latest

# OGC API endpoints:
# GET /                              — landing page
# GET /collections                   — list collections
# GET /collections/buildings/items   — list features (GeoJSON)
# GET /collections/buildings/items?bbox=-122.5,37.7,-122.3,37.9
# GET /collections/buildings/items/{id}
# GET /processes                     — list processes
# POST /processes/buffer/execution   — execute buffer
```

### STAC API — stac-fastapi

```python
# STAC API for raster data discovery
from stac_fastapi.api import app
from stac_fastapi.pgstac.core import CoreCrudClient
from stac_fastapi.pgstac.db import close_db_connection, connect_to_db

# Endpoints:
# GET /collections — STAC collections
# GET /collections/{id}/items — STAC items
# POST /search — STAC search with spatial/temporal filters
# Example search:
# POST /search
# {
#   "bbox": [-122.5, 37.7, -122.3, 37.9],
#   "datetime": "2024-01-01/2024-12-31",
#   "collections": ["sentinel-2"],
#   "limit": 10
# }
```

---

## Geocoding Services

### Nominatim — Docker Setup

```bash
# Import a region (e.g., China)
docker run -it --rm \
  -e PBF_URL=https://download.geofabrik.de/asia/china-latest.osm.pbf \
  -e REPLICATION_URL=https://download.geofabrik.de/asia/china-updates/ \
  -e NOMINATIM_PASSWORD=mysecret \
  -v nominatim-data:/var/lib/postgresql/14/main \
  -p 8080:8080 \
  mediagis/nominatim:4.4

# API usage:
# Forward geocode
curl "http://localhost:8080/search?q=Beijing+Tsinghua+University&format=geojson&limit=5"

# Reverse geocode
curl "http://localhost:8080/reverse?lat=39.999&lon=116.326&format=geojson"

# Structured search
curl "http://localhost:8080/search?city=Beijing&street=Zhongguancun&format=geojson"
```

### Pelias — Multi-Source Geocoder

```yaml
# pelias/docker-compose.yml
version: '3'
services:
  api:
    image: pelias/api:latest
    ports: ["4000:4000"]
    depends_on: [elasticsearch, placeholder, pip, interpolation]

  elasticsearch:
    image: pelias/elasticsearch:7.17.14
    volumes: ["elasticsearch:/usr/share/elasticsearch/data"]

  placeholder:
    image: pelias/placeholder:latest
    volumes: ["pelias-data:/data"]

  pip:
    image: pelias/pip-service:latest
    volumes: ["pelias-data:/data"]

  interpolation:
    image: pelias/interpolation:latest
    volumes: ["pelias-data:/data"]

  # Data importers
  openstreetmap:
    image: pelias/openstreetmap:latest
    volumes: ["pelias-data:/data"]
    command: ["./bin/start"]

  openaddresses:
    image: pelias/openaddresses:latest
    volumes: ["pelias-data:/data"]

volumes:
  elasticsearch:
  pelias-data:
```

### Photon — Fast Autocomplete

```bash
# Photon from Nominatim export
docker run -d --name photon \
  -p 2322:2322 \
  komoot/photon \
  -nominatim-import \
  -host nominatim \
  -port 5432 \
  -database nominatim \
  -user nominatim \
  -password mysecret

# Autocomplete search
curl "http://localhost:2322/api?q=Tsinghua&limit=5&lat=39.9&lon=116.3"
```

### Geocoding Service Comparison

| Service | Speed | Accuracy | Self-Hosted | Data Source | Best For |
|---------|-------|----------|-------------|-------------|----------|
| Nominatim | Medium | Good | Yes | OSM | General geocoding |
| Pelias | Fast | Good | Yes | OSM + OpenAddresses | Custom deployments |
| Photon | Very Fast | Good | Yes | OSM | Autocomplete |
| Mapbox | Fast | Excellent | No | Proprietary | Commercial apps |
| Google | Fast | Excellent | No | Proprietary | High accuracy |

---

## Routing Engines

### OSRM — Fastest Open-Source Routing

```bash
# Download and preprocess OSM data
docker run -t -v "${PWD}/osrm:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-extract -p /opt/car.lua /data/china-latest.osm.pbf

docker run -t -v "${PWD}/osrm:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-partition /data/china-latest.osrm

docker run -t -v "${PWD}/osrm:/data" ghcr.io/project-osrm/osrm-backend \
  osrm-customize /data/china-latest.osrm

# Start server
docker run -d --name osrm -p 5000:5000 -v "${PWD}/osrm:/data" \
  ghcr.io/project-osrm/osrm-backend \
  osrm-routed --algorithm mld /data/china-latest.osrm

# Route request
curl "http://localhost:5000/route/v1/driving/116.3,39.9;116.5,40.0?overview=full&geometries=geojson"

# Table (distance matrix)
curl "http://localhost:5000/table/v1/driving/116.3,39.9;116.5,40.0;116.4,39.8"

# Nearest road
curl "http://localhost:5000/nearest/v1/driving/116.3,39.9?number=3"
```

### Valhalla — Full-Featured Routing

```bash
# Start Valhalla
docker run -d --name valhalla -p 8002:8002 \
  -v "${PWD}/valhalla:/custom_files" \
  ghcr.io/gis-ops/docker-valhalla/valhalla:latest

# Route request
curl -X POST http://localhost:8002/route \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [
      {"lat": 39.9, "lon": 116.3},
      {"lat": 40.0, "lon": 116.5}
    ],
    "costing": "auto",
    "directions_options": {"units": "kilometers"}
  }'

# Isochrone (reachability area)
curl -X POST http://localhost:8002/isochrone \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [{"lat": 39.9, "lon": 116.4}],
    "costing": "pedestrian",
    "contours": [
      {"time": 5, "color": "ff0000"},
      {"time": 10, "color": "ffff00"},
      {"time": 15, "color": "00ff00"}
    ]
  }'

# Map matching (GPS trace to road)
curl -X POST http://localhost:8002/trace_route \
  -H "Content-Type: application/json" \
  -d '{
    "shape": [
      {"lat": 39.9001, "lon": 116.3001},
      {"lat": 39.9010, "lon": 116.3020},
      {"lat": 39.9050, "lon": 116.3100}
    ],
    "costing": "auto",
    "shape_match": "map_snap"
  }'
```

### pgRouting — SQL-Based Routing

```sql
-- Create routing topology
SELECT pgr_createTopology('roads', 0.00001, 'geom', 'id');

-- Shortest path (Dijkstra)
SELECT seq, node, edge, cost, agg_cost, ST_AsGeoJSON(geom) AS geometry
FROM pgr_dijkstra(
    'SELECT id, source, target, ST_Length(ST_Transform(geom, 3857)) AS cost FROM roads',
    (SELECT source FROM roads ORDER BY geom <-> ST_SetSRID(ST_MakePoint(116.3, 39.9), 4326) LIMIT 1),
    (SELECT source FROM roads ORDER BY geom <-> ST_SetSRID(ST_MakePoint(116.5, 40.0), 4326) LIMIT 1),
    directed := false
)
JOIN roads ON edge = roads.id;

-- Driving distance (isochrone via SQL)
SELECT ST_ConcaveHull(ST_Collect(geom), 0.8) AS isochrone
FROM (
    SELECT node, agg_cost, r.geom
    FROM pgr_drivingDistance(
        'SELECT id, source, target, ST_Length(ST_Transform(geom, 3857))/1000 AS cost FROM roads',
        (SELECT source FROM roads ORDER BY geom <-> ST_SetSRID(ST_MakePoint(116.4, 39.9), 4326) LIMIT 1),
        5.0,  -- 5 km
        directed := false
    ) dd
    JOIN roads r ON dd.edge = r.id
) sub;
```

### Routing Engine Comparison

| Engine | Language | Speed | Profiles | Isochrones | Map Matching | Best For |
|--------|----------|-------|----------|------------|-------------|----------|
| OSRM | C++ | Fastest | car/bike/foot | Via plugin | Yes | High-volume routing |
| Valhalla | C++ | Fast | 10+ modes | Native | Native | Multi-modal, features |
| GraphHopper | Java | Fast | Custom | Yes | Yes | Flexible profiles |
| pgRouting | SQL/C | Medium | Custom SQL | Via SQL | No | DB-integrated routing |

---

## Message Queues & Background Processing

### Celery + GDAL — Async Raster Processing

```python
# tasks.py
from celery import Celery
from osgeo import gdal
import subprocess

app = Celery('geo_tasks', broker='redis://localhost:6379/0')

@app.task(bind=True)
def generate_cog(self, input_path: str, output_path: str):
    """Convert raster to Cloud Optimized GeoTIFF"""
    self.update_state(state='PROCESSING', meta={'step': 'converting'})

    gdal.Translate(
        output_path, input_path,
        format='COG',
        creationOptions=['COMPRESS=ZSTD', 'OVERVIEW_RESAMPLING=AVERAGE', 'BLOCKSIZE=512']
    )

    self.update_state(state='PROCESSING', meta={'step': 'uploading'})
    # Upload to S3...
    return {'output': output_path, 'status': 'complete'}

@app.task
def generate_tiles(input_path: str, output_path: str, min_zoom: int = 0, max_zoom: int = 14):
    """Generate PMTiles from GeoJSON"""
    subprocess.run([
        'tippecanoe', '-o', output_path,
        f'--minimum-zoom={min_zoom}', f'--maximum-zoom={max_zoom}',
        '--drop-densest-as-needed',
        input_path
    ], check=True)
    return {'output': output_path}

@app.task
def spatial_analysis(feature_ids: list, buffer_m: float):
    """Run buffer analysis on features"""
    import geopandas as gpd
    from sqlalchemy import create_engine
    engine = create_engine('postgresql://user:pass@localhost/gisdb')
    gdf = gpd.read_postgis(
        f"SELECT * FROM features WHERE id = ANY(%s)",
        engine, params=(feature_ids,), geom_col='geom'
    )
    buffered = gdf.to_crs(3857).buffer(buffer_m).to_crs(4326)
    return buffered.to_json()
```

### Bull/BullMQ — Node.js Job Queue

```typescript
// workers/tileWorker.ts
import { Worker, Queue } from 'bullmq';
import { execSync } from 'child_process';
import IORedis from 'ioredis';

const connection = new IORedis({ host: 'localhost', port: 6379, maxRetriesPerRequest: null });

// Define queue
export const tileQueue = new Queue('tile-generation', { connection });

// Worker
const worker = new Worker('tile-generation', async (job) => {
  const { inputPath, outputPath, minZoom, maxZoom } = job.data;

  await job.updateProgress(10);
  execSync(`tippecanoe -o ${outputPath} --minimum-zoom=${minZoom} --maximum-zoom=${maxZoom} --drop-densest-as-needed ${inputPath}`);

  await job.updateProgress(80);
  // Upload to R2/S3...

  await job.updateProgress(100);
  return { outputPath, status: 'complete' };
}, { connection, concurrency: 2 });

worker.on('completed', (job) => console.log(`Job ${job.id} completed`));
worker.on('failed', (job, err) => console.error(`Job ${job?.id} failed:`, err));

// Add job
await tileQueue.add('generate', {
  inputPath: '/data/buildings.geojson',
  outputPath: '/data/tiles/buildings.pmtiles',
  minZoom: 2, maxZoom: 14,
});
```

---

## Service Comparison Summary

| Service | Type | Language | OGC | Self-Hosted | Min RAM | Best For |
|---------|------|----------|-----|-------------|---------|----------|
| Martin | Tile server | Rust | No | Yes | 128 MB | Fast vector tiles |
| pg_tileserv | Tile server | Go | No | Yes | 64 MB | Quick prototyping |
| TiTiler | Raster tiles | Python | No | Yes | 512 MB | Dynamic COG serving |
| GeoServer | Full server | Java | Full | Yes | 2 GB | Enterprise OGC |
| pygeoapi | OGC API | Python | OGC API | Yes | 256 MB | Modern OGC |
| FastAPI | API framework | Python | Via code | Yes | 128 MB | Custom spatial APIs |
| GeoDjango | API framework | Python | Via code | Yes | 256 MB | Admin-heavy apps |
| Nominatim | Geocoding | C++ | No | Yes | 2 GB+ | OSM geocoding |
| OSRM | Routing | C++ | No | Yes | 4 GB+ | High-perf routing |
| Valhalla | Routing | C++ | No | Yes | 4 GB+ | Multi-modal routing |
| pgRouting | Routing | SQL/C | No | Yes | — | DB-integrated routing |
