# Full-Stack GIS Architecture — 2025 Complete Guide

Designing a geospatial web application requires careful consideration of how data flows from ingestion to rendering. This guide covers architecture patterns, reference stacks, API design, data pipelines, and infrastructure patterns for building scalable GIS applications.

> **Quick Picks**
> - **Modern full-stack:** PostGIS + Martin + MapLibre + Next.js 15
> - **Zero-infrastructure:** S3/R2 COG + PMTiles + static frontend
> - **Enterprise/OGC:** PostGIS + GeoServer + Spring Boot
> - **Rapid prototyping:** Supabase + MapLibre + SvelteKit
> - **Analytics:** DuckDB + GeoParquet + Observable Framework

---

## Architecture Patterns

### Monolith vs Microservices vs Modular Monolith

**Monolithic Architecture** — Single deployment unit. Best for small teams and simple use cases.

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

**Microservices Architecture** — Each concern is a separate service. Best for large teams.

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

**Modular Monolith** — Single deployment, but internally organized as modules. Best balance for medium teams.

```
┌───────────────────────────────────────────────┐
│            Modular Monolith                    │
│  ┌───────────┐ ┌───────────┐ ┌──────────────┐│
│  │  Tile     │ │  Feature  │ │  Processing  ││
│  │  Module   │ │  Module   │ │  Module      ││
│  │  /tiles/* │ │  /api/*   │ │  /jobs/*     ││
│  └─────┬─────┘ └─────┬─────┘ └──────┬───────┘│
│        └──────────────┼──────────────┘        │
│              ┌────────▼────────┐               │
│              │    Shared DB    │               │
│              │    (PostGIS)    │               │
│              └─────────────────┘               │
└───────────────────────────────────────────────┘
```

### Decision Matrix

| Factor | Monolith | Modular Monolith | Microservices |
|--------|----------|-----------------|---------------|
| Team size | 1-5 | 3-15 | 10+ |
| Deployment | Single | Single | Multiple |
| Scaling | Vertical | Vertical + partial horizontal | Per-service |
| Data sources | 1-2 | 2-5 | Many |
| Complexity | Low | Medium | High |
| Time to market | Fastest | Fast | Slow |
| Best for GIS when | Prototype/MVP | Production apps | Enterprise platform |

### Event-Driven GIS Architecture

Spatial events (feature created, boundary changed, sensor reading) flow through an event bus.

```
┌──────────┐    ┌──────────────┐    ┌──────────────┐
│  Sensor  │───▶│              │───▶│  Tile Cache  │
│  Ingest  │    │   Event Bus  │    │  Invalidator │
└──────────┘    │  (Kafka /    │    └──────────────┘
┌──────────┐    │   Redis      │    ┌──────────────┐
│  Feature │───▶│   Streams)   │───▶│  Spatial     │
│  Editor  │    │              │    │  Aggregator  │
└──────────┘    └──────────────┘    └──────────────┘
                       │            ┌──────────────┐
                       └───────────▶│  Notification│
                                    │  Service     │
                                    └──────────────┘
```

```python
# Event-driven spatial processing with Redis Streams
import redis
import json

r = redis.Redis()

# Producer: emit spatial event
def emit_feature_event(event_type: str, feature: dict):
    r.xadd('spatial-events', {
        'type': event_type,
        'feature_id': str(feature['id']),
        'bbox': json.dumps(feature['bbox']),
        'payload': json.dumps(feature),
        'timestamp': str(time.time()),
    })

# Consumer: invalidate tile cache for affected area
def tile_cache_invalidator():
    last_id = '0'
    while True:
        events = r.xread({'spatial-events': last_id}, block=5000, count=100)
        for stream, messages in events:
            for msg_id, data in messages:
                bbox = json.loads(data[b'bbox'])
                invalidate_tiles_in_bbox(bbox)
                last_id = msg_id
```

### Serverless-First Architecture

Zero servers to manage. Everything runs on cloud functions and object storage.

```
┌──────────────────────────────────────────────────┐
│                   CDN (Cloudflare)                │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │
│  │ Frontend │  │ PMTiles  │  │   COG Files    │ │
│  │ (Vercel) │  │ (R2)     │  │   (S3/R2)      │ │
│  └────┬─────┘  └────┬─────┘  └───────┬────────┘ │
│       │              │                │          │
└───────┼──────────────┼────────────────┼──────────┘
        │              │                │
   Range requests  Range requests  Range requests
   (HTML/JS/CSS)   (vector tiles)  (raster tiles)
        │              │                │
┌───────▼──────────────▼────────────────▼──────────┐
│                 Browser (MapLibre)                 │
│   PMTiles protocol  +  geotiff.js  +  DuckDB-WASM│
└───────────────────────────────────────────────────┘
```

### Edge Computing for GIS

```
┌─────────────────────────────────────────────┐
│       Cloudflare Workers (Edge)             │
│  ┌─────────────┐  ┌───────────────────────┐│
│  │ Tile Router │  │ Spatial Query Cache  ││
│  │ (PMTiles    │  │ (Workers KV / D1)    ││
│  │  protocol)  │  │                       ││
│  └──────┬──────┘  └───────────┬───────────┘│
│         │                     │            │
└─────────┼─────────────────────┼────────────┘
          │                     │
   ┌──────▼──────┐    ┌────────▼────────┐
   │  R2 Bucket  │    │  Origin API     │
   │  (PMTiles)  │    │  (PostGIS)      │
   └─────────────┘    └─────────────────┘
```

---

## Reference Architectures

### Stack A: PostGIS + Martin + Next.js 15 (Modern Full-Stack)

The most popular production stack in 2025.

```
┌─────────────────────────────────────────┐
│        Browser                          │
│  Next.js App (MapLibre + react-map-gl)  │
└──────────────┬──────────────────────────┘
               │
        ┌──────▼──────┐
        │   Nginx /   │
        │   Caddy     │
        └──────┬──────┘
               │
  ┌────────────┼────────────┐
  │            │            │
┌─▼──────┐ ┌──▼───────┐ ┌──▼───────┐
│ Martin │ │ Next.js  │ │ Pelias / │
│ :3000  │ │ API      │ │ Nominatim│
│ /tiles │ │ :3001    │ │ :4000    │
└───┬────┘ └────┬─────┘ └──┬───────┘
    │           │           │
    └───────────┼───────────┘
                │
         ┌──────▼──────┐
         │   PostGIS   │
         │   :5432     │
         └─────────────┘
```

```yaml
# docker-compose.yml — Production-ready
version: '3.8'
services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: gisdb
      POSTGRES_USER: gisuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gisuser -d gisdb"]
      interval: 10s
      timeout: 5s
      retries: 5

  martin:
    image: ghcr.io/maplibre/martin
    environment:
      DATABASE_URL: postgresql://gisuser:${DB_PASSWORD}@postgis/gisdb
    ports: ["3000:3000"]
    depends_on:
      postgis: { condition: service_healthy }
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000/health"]
      interval: 10s

  app:
    build:
      context: ./app
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://gisuser:${DB_PASSWORD}@postgis/gisdb
      MARTIN_URL: http://martin:3000
    ports: ["3001:3000"]
    depends_on:
      postgis: { condition: service_healthy }
      martin: { condition: service_healthy }

  caddy:
    image: caddy:2-alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
    depends_on: [app, martin]

volumes:
  pgdata:
```

```
# Caddyfile — Reverse proxy
example.com {
    handle /tiles/* {
        reverse_proxy martin:3000
        header Cache-Control "public, max-age=86400"
    }
    handle /api/* {
        reverse_proxy app:3001
    }
    handle {
        reverse_proxy app:3001
    }
}
```

**Monthly cost:** ~$20-30 on a single VPS (Hetzner/DigitalOcean)

### Stack B: S3 COG + PMTiles + Vite (Serverless)

Zero infrastructure. Everything served from object storage + CDN.

```
┌──────────────────────────────────────────────┐
│           Cloudflare CDN                      │
│  ┌────────────┐  ┌────────────┐              │
│  │ Frontend   │  │ PMTiles    │              │
│  │ (Vercel)   │  │ (R2)       │              │
│  └─────┬──────┘  └──────┬─────┘              │
│        │                │                    │
└────────┼────────────────┼────────────────────┘
         │                │
    ┌────▼────┐     ┌─────▼─────┐
    │ Vite    │     │ MapLibre  │
    │ App     │     │ + pmtiles │
    │         │     │ protocol  │
    └─────────┘     └───────────┘
```

```bash
# Build pipeline
# 1. Generate PMTiles
tippecanoe -o tiles/buildings.pmtiles --minimum-zoom=2 --maximum-zoom=14 \
  --drop-densest-as-needed buildings.geojson

# 2. Upload to R2
rclone copy tiles/ r2:my-bucket/tiles/

# 3. Deploy frontend
cd frontend && npm run build && npx vercel --prod
```

```javascript
// Frontend: MapLibre + PMTiles from R2
import { Protocol } from 'pmtiles';
import maplibregl from 'maplibre-gl';

const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      buildings: {
        type: 'vector',
        url: 'pmtiles://https://tiles.example.com/buildings.pmtiles',
      },
    },
    layers: [{
      id: 'buildings-fill', type: 'fill',
      source: 'buildings', 'source-layer': 'buildings',
      paint: { 'fill-color': '#088', 'fill-opacity': 0.6 },
    }],
  },
});
```

**Monthly cost:** ~$0-5 (R2 free tier + Vercel free tier)

### Stack C: PostGIS + GeoServer + Spring Boot (Enterprise)

OGC-compliant enterprise stack with full WMS/WFS/WMTS support.

```
┌────────────────────────────────────────────────┐
│        Enterprise GIS Platform                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
│  │GeoServer │  │ Spring   │  │  React +     │ │
│  │ WMS/WFS  │  │ Boot API │  │  OpenLayers  │ │
│  │ WMTS     │  │          │  │              │ │
│  └────┬─────┘  └────┬─────┘  └──────────────┘ │
│       │              │                         │
│  ┌────▼──────────────▼─────┐                   │
│  │       PostGIS           │                   │
│  │  + GeoWebCache          │                   │
│  └─────────────────────────┘                   │
└────────────────────────────────────────────────┘
```

### Stack D: Supabase + MapLibre + SvelteKit (Rapid Prototyping)

From idea to deployed app in hours. Supabase provides PostGIS + Auth + Realtime + Storage.

```
┌────────────────────────────────────────┐
│        SvelteKit App                   │
│  ┌──────────┐  ┌───────────────────┐  │
│  │ MapLibre │  │  Supabase Client  │  │
│  │ GL JS    │  │  (Auth + DB + RT) │  │
│  └────┬─────┘  └────────┬──────────┘  │
│       │                 │             │
└───────┼─────────────────┼─────────────┘
        │                 │
┌───────▼─────────────────▼─────────────┐
│           Supabase                     │
│  ┌─────────┐ ┌──────┐ ┌────────────┐ │
│  │PostGIS  │ │ Auth │ │  Realtime  │ │
│  │(spatial │ │      │ │  (WebSocket│ │
│  │ queries)│ │      │ │   changes) │ │
│  └─────────┘ └──────┘ └────────────┘ │
└───────────────────────────────────────┘
```

```typescript
// Supabase spatial queries in SvelteKit
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_ANON_KEY!
);

// Fetch features within bbox
async function getFeaturesInBBox(bbox: [number, number, number, number]) {
  const { data, error } = await supabase.rpc('features_in_bbox', {
    min_lng: bbox[0], min_lat: bbox[1],
    max_lng: bbox[2], max_lat: bbox[3],
  });
  return data;
}

// Supabase PostGIS function
// CREATE FUNCTION features_in_bbox(min_lng float, min_lat float, max_lng float, max_lat float)
// RETURNS SETOF json AS $$
//   SELECT jsonb_build_object(
//     'type', 'Feature',
//     'properties', jsonb_build_object('id', id, 'name', name),
//     'geometry', ST_AsGeoJSON(geom)::jsonb
//   ) FROM features
//   WHERE geom && ST_MakeEnvelope(min_lng, min_lat, max_lng, max_lat, 4326)
//   LIMIT 5000;
// $$ LANGUAGE sql STABLE;

// Real-time spatial updates
supabase
  .channel('spatial-changes')
  .on('postgres_changes', {
    event: '*',
    schema: 'public',
    table: 'features',
  }, (payload) => {
    console.log('Feature changed:', payload);
    // Update map layer
  })
  .subscribe();
```

### Stack E: DuckDB + GeoParquet + Observable (Analytics)

For spatial data analysis dashboards without a server.

```javascript
// Observable Framework notebook
import * as duckdb from '@duckdb/duckdb-wasm';

const db = await duckdb.AsyncDuckDB.create();
await db.open({ query: { castBigIntToDouble: true } });
const conn = await db.connect();

// Load GeoParquet directly
await conn.query(`
  INSTALL spatial; LOAD spatial;
  CREATE TABLE buildings AS
  SELECT * FROM read_parquet('https://data.example.com/buildings.parquet');
`);

// Spatial analysis in SQL
const result = await conn.query(`
  SELECT
    neighborhood,
    COUNT(*) as building_count,
    AVG(height) as avg_height,
    SUM(ST_Area(ST_Transform(geom, 3857))) / 1e6 as total_area_km2
  FROM buildings
  GROUP BY neighborhood
  ORDER BY building_count DESC
  LIMIT 20
`);
```

### Stack F: Real-Time IoT — TimescaleDB + MQTT + deck.gl

```
┌──────────┐     ┌──────────┐     ┌──────────────────┐
│  IoT     │────▶│  MQTT    │────▶│  Ingest Service  │
│  Sensors │     │  Broker  │     │  (Node.js)       │
└──────────┘     │(Mosquitto│     └────────┬─────────┘
                 └──────────┘              │
                                    ┌──────▼──────┐
                                    │TimescaleDB  │
                                    │(+ PostGIS)  │
                                    └──────┬──────┘
                                           │
                                    ┌──────▼──────┐
                                    │  Dashboard  │
                                    │ (deck.gl +  │
                                    │  WebSocket) │
                                    └─────────────┘
```

---

## Stack Comparison

| Stack | Complexity | Scalability | Cost/mo | Best For |
|-------|-----------|-------------|---------|----------|
| PostGIS + Martin + Next.js | Medium | High | $20-30 | Modern production apps |
| S3 COG + PMTiles + Vite | Very Low | Very High | $0-5 | Static/serverless |
| PostGIS + GeoServer + Spring | High | Medium | $40-80 | Enterprise OGC |
| Supabase + MapLibre + SvelteKit | Low | Medium | $0-25 | Rapid prototyping |
| DuckDB + GeoParquet + Observable | Low | Low-Medium | $0-5 | Analytics dashboards |
| TimescaleDB + MQTT + deck.gl | High | High | $50-100 | Real-time IoT |

---

## API Design Patterns

### REST for Spatial Data

```
# Standard spatial REST endpoints
GET  /api/v1/collections                           # List collections
GET  /api/v1/collections/{id}/items                # List features
GET  /api/v1/collections/{id}/items?bbox=a,b,c,d   # Spatial filter
GET  /api/v1/collections/{id}/items?filter=height>50  # CQL filter
GET  /api/v1/collections/{id}/items/{featureId}    # Single feature
POST /api/v1/collections/{id}/items                # Create feature
PUT  /api/v1/collections/{id}/items/{featureId}    # Update feature
DELETE /api/v1/collections/{id}/items/{featureId}  # Delete feature

# Tile endpoints
GET  /api/v1/tiles/{source}/{z}/{x}/{y}.mvt        # Vector tile
GET  /api/v1/tiles/{source}/{z}/{x}/{y}.png        # Raster tile

# Spatial operations
POST /api/v1/spatial/buffer                         # Buffer geometry
POST /api/v1/spatial/intersect                      # Intersection
POST /api/v1/spatial/union                          # Union
GET  /api/v1/spatial/within?point=lng,lat&radius=1000  # Point query
```

### GraphQL + PostGIS

```graphql
# schema.graphql
type Feature {
  id: ID!
  name: String!
  category: String
  value: Float
  geometry: GeoJSON!
  area_m2: Float
  centroid: GeoJSON
}

type Query {
  features(bbox: BBoxInput, category: String, limit: Int): FeatureCollection!
  feature(id: ID!): Feature
  nearestFeatures(point: PointInput!, radius: Float!, limit: Int): [Feature!]!
  spatialStatistics(bbox: BBoxInput!): SpatialStats!
}

type Mutation {
  createFeature(input: FeatureInput!): Feature!
  updateFeature(id: ID!, input: FeatureInput!): Feature!
  deleteFeature(id: ID!): Boolean!
}

type Subscription {
  featureChanged(bbox: BBoxInput): Feature!
}

input BBoxInput { minLng: Float! minLat: Float! maxLng: Float! maxLat: Float! }
input PointInput { lng: Float! lat: Float! }
```

```typescript
// GraphQL resolver with PostGIS
const resolvers = {
  Query: {
    features: async (_, { bbox, category, limit = 1000 }, { db }) => {
      let where = 'WHERE 1=1';
      const params: any[] = [];

      if (bbox) {
        params.push(bbox.minLng, bbox.minLat, bbox.maxLng, bbox.maxLat);
        where += ` AND geom && ST_MakeEnvelope($${params.length-3}, $${params.length-2}, $${params.length-1}, $${params.length}, 4326)`;
      }
      if (category) {
        params.push(category);
        where += ` AND category = $${params.length}`;
      }
      params.push(limit);

      const result = await db.query(`
        SELECT id, name, category, value,
               ST_AsGeoJSON(geom)::json AS geometry,
               ST_Area(ST_Transform(geom, 3857)) AS area_m2
        FROM features ${where} LIMIT $${params.length}
      `, params);

      return {
        type: 'FeatureCollection',
        features: result.rows.map(r => ({
          ...r, geometry: r.geometry,
        })),
      };
    },
  },
};
```

### gRPC for Service-to-Service

```protobuf
// spatial.proto
syntax = "proto3";
package spatial;

service SpatialService {
  rpc GetFeatures (FeaturesRequest) returns (stream Feature);
  rpc GetTile (TileRequest) returns (TileResponse);
  rpc SpatialQuery (SpatialQueryRequest) returns (FeatureCollection);
}

message FeaturesRequest {
  BBox bbox = 1;
  string category = 2;
  int32 limit = 3;
}

message BBox {
  double min_lng = 1;
  double min_lat = 2;
  double max_lng = 3;
  double max_lat = 4;
}

message Feature {
  int64 id = 1;
  string name = 2;
  string category = 3;
  double value = 4;
  bytes geometry_wkb = 5;  // WKB-encoded geometry
}

message TileRequest {
  int32 z = 1;
  int32 x = 2;
  int32 y = 3;
  string source = 4;
}

message TileResponse {
  bytes data = 1;  // MVT or PNG
  string content_type = 2;
}
```

---

## Data Pipeline Architecture

### ETL Pipeline — ogr2ogr + PostGIS

```bash
#!/bin/bash
# etl-pipeline.sh — Complete data ingestion pipeline

# 1. Download data
wget -O buildings.geojson "https://data.example.com/buildings.geojson"

# 2. Validate and fix geometry
ogr2ogr -f GeoJSON validated.geojson buildings.geojson \
  -makevalid \
  -t_srs EPSG:4326

# 3. Import to PostGIS
ogr2ogr -f PostgreSQL \
  PG:"host=localhost dbname=gisdb user=gisuser" \
  validated.geojson \
  -nln buildings \
  -overwrite \
  -lco GEOMETRY_NAME=geom \
  -lco FID=id \
  -lco PRECISION=NO

# 4. Create indexes
psql -d gisdb -c "
  CREATE INDEX IF NOT EXISTS idx_buildings_geom ON buildings USING GIST (geom);
  CREATE INDEX IF NOT EXISTS idx_buildings_type ON buildings (type);
  ANALYZE buildings;
  CLUSTER buildings USING idx_buildings_geom;
"

# 5. Generate tiles
ogr2ogr -f FlatGeobuf buildings.fgb PG:"dbname=gisdb" buildings
tippecanoe -o buildings.pmtiles \
  --minimum-zoom=10 --maximum-zoom=15 \
  --drop-densest-as-needed \
  buildings.fgb

# 6. Upload tiles to R2
rclone copy buildings.pmtiles r2:tiles/
```

### ELT with dbt + PostGIS

```sql
-- models/staging/stg_buildings.sql
{{ config(materialized='view') }}

SELECT
    id,
    name,
    height,
    type,
    ST_SetSRID(geom, 4326) AS geom,
    ST_Area(ST_Transform(geom, 3857)) AS area_m2
FROM {{ source('raw', 'buildings_import') }}
WHERE ST_IsValid(geom)
  AND ST_GeometryType(geom) = 'ST_MultiPolygon'

-- models/marts/mart_building_stats.sql
{{ config(materialized='table', post_hook="CREATE INDEX IF NOT EXISTS idx_bstats_geom ON {{ this }} USING GIST (geom);") }}

SELECT
    neighborhood,
    COUNT(*) AS building_count,
    AVG(height) AS avg_height,
    SUM(area_m2) AS total_area_m2,
    ST_ConvexHull(ST_Collect(geom)) AS geom
FROM {{ ref('stg_buildings') }}
GROUP BY neighborhood
```

### Streaming Pipeline — Kafka + PostGIS

```python
# Kafka consumer for real-time spatial data
from kafka import KafkaConsumer
import json
import psycopg2

consumer = KafkaConsumer(
    'sensor-readings',
    bootstrap_servers='localhost:9092',
    value_deserializer=lambda m: json.loads(m.decode('utf-8')),
    group_id='spatial-ingest',
)

conn = psycopg2.connect("dbname=gisdb user=gisuser")
cur = conn.cursor()

for message in consumer:
    data = message.value
    cur.execute(
        """INSERT INTO sensor_readings (sensor_id, value, timestamp, geom)
           VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326))""",
        (data['sensor_id'], data['value'], data['timestamp'],
         data['longitude'], data['latitude'])
    )
    if message.offset % 100 == 0:
        conn.commit()
```

---

## Infrastructure Patterns

### Database: Read Replicas + Partitioning

```sql
-- Partition by time (for time-series spatial data)
CREATE TABLE sensor_readings (
    id BIGSERIAL,
    sensor_id INTEGER,
    value DOUBLE PRECISION,
    geom GEOMETRY(Point, 4326),
    created_at TIMESTAMPTZ NOT NULL
) PARTITION BY RANGE (created_at);

CREATE TABLE sensor_readings_2024_q1 PARTITION OF sensor_readings
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');
CREATE TABLE sensor_readings_2024_q2 PARTITION OF sensor_readings
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');

-- Spatial index on each partition
CREATE INDEX ON sensor_readings_2024_q1 USING GIST (geom);
CREATE INDEX ON sensor_readings_2024_q2 USING GIST (geom);

-- Partition by geometry (grid)
CREATE TABLE features (
    id SERIAL,
    name TEXT,
    geom GEOMETRY(Geometry, 4326),
    grid_cell INTEGER GENERATED ALWAYS AS (
        floor(ST_X(ST_Centroid(geom)) / 10)::int * 1000 +
        floor(ST_Y(ST_Centroid(geom)) / 10)::int
    ) STORED
) PARTITION BY LIST (grid_cell);
```

### Caching Layers

```
Request flow:
Browser → Service Worker Cache → CDN Cache → Nginx Cache → Tile Server → PostGIS
  (1ms)        (5ms)              (20ms)       (50ms)        (200ms)      (500ms)

Cache strategy by data type:
┌──────────────┬─────────────────┬──────────────┬──────────────┐
│   Data Type  │   TTL           │   Strategy   │   Layer      │
├──────────────┼─────────────────┼──────────────┼──────────────┤
│ Base tiles   │ 30 days         │ Cache-first  │ CDN + SW     │
│ Data tiles   │ 1-24 hours      │ Stale-while  │ CDN          │
│ Feature API  │ 5-30 min        │ Network-first│ Redis        │
│ Search       │ 1 hour          │ Cache-first  │ Redis        │
│ Static assets│ 1 year          │ Cache-first  │ CDN + SW     │
└──────────────┴─────────────────┴──────────────┴──────────────┘
```

### Observability

```yaml
# OpenTelemetry for spatial API
# docker-compose additions
services:
  otel-collector:
    image: otel/opentelemetry-collector-contrib:latest
    ports: ["4317:4317", "4318:4318"]

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    ports: ["3001:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

```python
# Custom metrics for spatial API
from prometheus_client import Histogram, Counter, Gauge

tile_request_duration = Histogram(
    'tile_request_duration_seconds',
    'Time to serve a tile',
    ['source', 'zoom_level']
)

feature_query_duration = Histogram(
    'feature_query_duration_seconds',
    'Time to query features',
    ['collection', 'has_bbox']
)

cache_hit_total = Counter(
    'cache_hit_total',
    'Cache hits',
    ['cache_layer']  # cdn, nginx, redis, miss
)

active_connections = Gauge(
    'postgis_active_connections',
    'Active PostGIS connections'
)
```

### Cost Modeling

| Scale | Stack | Hosting | DB | CDN | Total/mo |
|-------|-------|---------|-----|-----|----------|
| 1K users/day | Martin + Next.js | $10 VPS | Included | $0 | ~$10 |
| 10K users/day | Martin + Next.js | $20 VPS | Included | $10 | ~$30 |
| 100K users/day | K8s | $100 | $50 RDS | $30 | ~$180 |
| 1M users/day | K8s + CDN | $500 | $200 RDS | $100 | ~$800 |
| Static tiles | PMTiles + Vercel | $0 | N/A | $0-5 R2 | ~$0-5 |
