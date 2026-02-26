# Deployment — 2025 Complete Guide

Taking a GIS application from development to production involves containerization, infrastructure choices, CI/CD pipelines, monitoring, and disaster recovery. This guide covers deployment strategies from simple Docker setups to scalable cloud architectures.

> **Quick Picks**
> - **Cheapest:** PMTiles on Cloudflare R2 + frontend on Vercel ($0-5/mo)
> - **Full-stack dev:** Docker Compose (PostGIS + Martin + Frontend)
> - **Small team production:** Docker Compose on $20/mo VPS
> - **Scale:** Kubernetes with managed PostGIS (RDS/Cloud SQL)
> - **Serverless raster:** Lambda + TiTiler + S3 COGs

---

## Docker for GIS

### Production Dockerfiles

```dockerfile
# PostGIS with extensions — Dockerfile.postgis
FROM postgis/postgis:16-3.4
RUN apt-get update && apt-get install -y \
    postgresql-16-pgrouting \
    postgresql-16-h3 \
    postgresql-16-ogr-fdw \
  && rm -rf /var/lib/apt/lists/*

COPY init.sql /docker-entrypoint-initdb.d/
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD pg_isready -U $POSTGRES_USER -d $POSTGRES_DB
```

```dockerfile
# Next.js frontend — Dockerfile.frontend (multi-stage)
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:20-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/public ./public

USER node
EXPOSE 3000
CMD ["node", "server.js"]
```

```dockerfile
# FastAPI spatial API — Dockerfile.api
FROM python:3.12-slim
RUN apt-get update && apt-get install -y libgdal-dev && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
USER nobody
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose — Minimal Stack

```yaml
# docker-compose.yml — PostGIS + Martin + Frontend
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

  frontend:
    build: ./frontend
    ports: ["8080:3000"]
    environment:
      NEXT_PUBLIC_TILE_URL: http://localhost:3000
    depends_on:
      martin: { condition: service_healthy }

volumes:
  pgdata:
```

### Docker Compose — Full Stack with Monitoring

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: gisdb
      POSTGRES_USER: gisuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_INITDB_ARGS: "--data-checksums"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
      - ./backups:/backups
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    deploy:
      resources:
        limits: { memory: 4G, cpus: '2' }
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gisuser -d gisdb"]
      interval: 10s

  martin:
    image: ghcr.io/maplibre/martin
    volumes:
      - ./martin-config.yaml:/config.yaml
      - ./data/tiles:/data/tiles
    command: ["--config", "/config.yaml"]
    deploy:
      resources:
        limits: { memory: 1G, cpus: '2' }
    depends_on:
      postgis: { condition: service_healthy }

  caddy:
    image: caddy:2-alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./Caddyfile:/etc/caddy/Caddyfile
      - caddy_data:/data
      - caddy_config:/config
    depends_on: [frontend, martin]

  frontend:
    build: ./frontend
    environment:
      DATABASE_URL: postgresql://gisuser:${DB_PASSWORD}@postgis/gisdb
      MARTIN_URL: http://martin:3000

  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports: ["9090:9090"]

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports: ["3001:3000"]

  postgres-exporter:
    image: prometheuscommunity/postgres-exporter:latest
    environment:
      DATA_SOURCE_NAME: postgresql://gisuser:${DB_PASSWORD}@postgis/gisdb?sslmode=disable
    depends_on:
      postgis: { condition: service_healthy }

  # Backup
  backup:
    image: prodrigestivill/postgres-backup-local:latest
    environment:
      POSTGRES_HOST: postgis
      POSTGRES_DB: gisdb
      POSTGRES_USER: gisuser
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      SCHEDULE: "@daily"
      BACKUP_KEEP_DAYS: 7
      BACKUP_KEEP_WEEKS: 4
    volumes:
      - ./backups:/backups
    depends_on:
      postgis: { condition: service_healthy }

volumes:
  pgdata:
  caddy_data:
  caddy_config:
  prometheus_data:
  grafana_data:
```

```
# Caddyfile — Auto HTTPS reverse proxy
example.com {
    # Tiles with aggressive caching
    handle /tiles/* {
        reverse_proxy martin:3000
        header Cache-Control "public, max-age=86400, stale-while-revalidate=604800"
        header Access-Control-Allow-Origin "*"
    }

    # API
    handle /api/* {
        reverse_proxy frontend:3000
    }

    # Frontend
    handle {
        reverse_proxy frontend:3000
    }

    # Security headers
    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "DENY"
        Referrer-Policy "strict-origin-when-cross-origin"
    }

    # Compression
    encode zstd gzip
}
```

### Docker Security

```yaml
# Security best practices
services:
  martin:
    image: ghcr.io/maplibre/martin
    read_only: true                    # Read-only filesystem
    security_opt:
      - no-new-privileges:true         # No privilege escalation
    cap_drop:
      - ALL                            # Drop all Linux capabilities
    tmpfs:
      - /tmp                           # Writable tmp in memory

  postgis:
    image: postgis/postgis:16-3.4
    security_opt:
      - no-new-privileges:true
    # Never expose DB port in production
    # ports: ["5432:5432"]  # REMOVE THIS
```

---

## Kubernetes

### K8s Manifests — PostGIS StatefulSet

```yaml
# postgis-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgis
spec:
  serviceName: postgis
  replicas: 1
  selector:
    matchLabels: { app: postgis }
  template:
    metadata:
      labels: { app: postgis }
    spec:
      containers:
        - name: postgis
          image: postgis/postgis:16-3.4
          ports:
            - containerPort: 5432
          env:
            - name: POSTGRES_DB
              value: gisdb
            - name: POSTGRES_USER
              valueFrom:
                secretKeyRef: { name: postgis-secret, key: username }
            - name: POSTGRES_PASSWORD
              valueFrom:
                secretKeyRef: { name: postgis-secret, key: password }
          resources:
            requests: { memory: "2Gi", cpu: "1" }
            limits: { memory: "4Gi", cpu: "2" }
          volumeMounts:
            - name: pgdata
              mountPath: /var/lib/postgresql/data
          livenessProbe:
            exec:
              command: ["pg_isready", "-U", "gisuser", "-d", "gisdb"]
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            exec:
              command: ["pg_isready", "-U", "gisuser", "-d", "gisdb"]
            initialDelaySeconds: 5
            periodSeconds: 5
  volumeClaimTemplates:
    - metadata: { name: pgdata }
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests: { storage: 100Gi }
        storageClassName: gp3
```

### Martin Deployment with HPA

```yaml
# martin-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: martin
spec:
  replicas: 2
  selector:
    matchLabels: { app: martin }
  template:
    metadata:
      labels: { app: martin }
    spec:
      containers:
        - name: martin
          image: ghcr.io/maplibre/martin
          ports: [{ containerPort: 3000 }]
          env:
            - name: DATABASE_URL
              valueFrom:
                secretKeyRef: { name: postgis-secret, key: url }
          resources:
            requests: { memory: "256Mi", cpu: "250m" }
            limits: { memory: "512Mi", cpu: "1" }
          livenessProbe:
            httpGet: { path: /health, port: 3000 }
            initialDelaySeconds: 10
          readinessProbe:
            httpGet: { path: /health, port: 3000 }
            initialDelaySeconds: 5
---
# martin-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: martin-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: martin
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target: { type: Utilization, averageUtilization: 70 }
    - type: Resource
      resource:
        name: memory
        target: { type: Utilization, averageUtilization: 80 }
```

### Ingress — Path-Based Routing

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: gis-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-buffering: "on"
    nginx.ingress.kubernetes.io/configuration-snippet: |
      add_header Cache-Control "public, max-age=86400" always;
spec:
  tls:
    - hosts: [gis.example.com]
      secretName: gis-tls
  rules:
    - host: gis.example.com
      http:
        paths:
          - path: /tiles
            pathType: Prefix
            backend:
              service: { name: martin, port: { number: 3000 } }
          - path: /api
            pathType: Prefix
            backend:
              service: { name: api, port: { number: 8000 } }
          - path: /
            pathType: Prefix
            backend:
              service: { name: frontend, port: { number: 3000 } }
```

### Backup CronJob

```yaml
# backup-cronjob.yaml
apiVersion: batch/v1
kind: CronJob
metadata:
  name: postgis-backup
spec:
  schedule: "0 2 * * *"  # 2 AM daily
  jobTemplate:
    spec:
      template:
        spec:
          containers:
            - name: backup
              image: postgres:16-alpine
              command:
                - /bin/sh
                - -c
                - |
                  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
                  pg_dump -Fc $DATABASE_URL > /backups/gisdb_${TIMESTAMP}.dump
                  # Keep last 7 days
                  find /backups -name "*.dump" -mtime +7 -delete
                  # Upload to S3
                  aws s3 cp /backups/gisdb_${TIMESTAMP}.dump s3://backups/postgis/
              env:
                - name: DATABASE_URL
                  valueFrom:
                    secretKeyRef: { name: postgis-secret, key: url }
              volumeMounts:
                - name: backups
                  mountPath: /backups
          volumes:
            - name: backups
              persistentVolumeClaim: { claimName: backup-pvc }
          restartPolicy: OnFailure
```

---

## Serverless & Edge

### AWS Lambda + TiTiler — SAM Template

```yaml
# template.yaml (AWS SAM)
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Globals:
  Function:
    Timeout: 30
    MemorySize: 1536
    Runtime: python3.12
    Environment:
      Variables:
        GDAL_CACHEMAX: 512
        GDAL_DISABLE_READDIR_ON_OPEN: EMPTY_DIR
        VSI_CACHE: "TRUE"
        VSI_CACHE_SIZE: 536870912

Resources:
  TiTilerFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: handler.handler
      CodeUri: ./src
      Events:
        Api:
          Type: HttpApi
          Properties:
            Path: /{proxy+}
            Method: ANY
      Policies:
        - S3ReadPolicy:
            BucketName: !Ref COGBucket

  COGBucket:
    Type: AWS::S3::Bucket
    Properties:
      BucketName: gis-cog-data
      CorsConfiguration:
        CorsRules:
          - AllowedOrigins: ["*"]
            AllowedMethods: [GET, HEAD]
            AllowedHeaders: ["*"]

Outputs:
  ApiUrl:
    Value: !Sub "https://${ServerlessHttpApi}.execute-api.${AWS::Region}.amazonaws.com"
```

### Cloudflare Workers + R2 — PMTiles at the Edge

```javascript
// worker.js — Serve PMTiles from R2 at the edge
import { PMTiles } from 'pmtiles';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const path = url.pathname;

    // CORS preflight
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': '*',
          'Access-Control-Allow-Methods': 'GET, HEAD',
          'Access-Control-Allow-Headers': 'Range',
        },
      });
    }

    // Parse tile request: /tiles/{name}/{z}/{x}/{y}.mvt
    const match = path.match(/^\/tiles\/(\w+)\/(\d+)\/(\d+)\/(\d+)\.(mvt|pbf)$/);
    if (!match) {
      return new Response('Not found', { status: 404 });
    }

    const [, name, z, x, y] = match;

    // Open PMTiles from R2
    const pmtiles = new PMTiles(
      new R2Source(env.TILES_BUCKET, `${name}.pmtiles`)
    );

    const tile = await pmtiles.getZxy(Number(z), Number(x), Number(y));
    if (!tile?.data) {
      return new Response('', { status: 204 });
    }

    return new Response(tile.data, {
      headers: {
        'Content-Type': 'application/x-protobuf',
        'Cache-Control': 'public, max-age=86400',
        'Access-Control-Allow-Origin': '*',
      },
    });
  },
};

class R2Source {
  constructor(bucket, key) {
    this.bucket = bucket;
    this.key = key;
  }

  async getBytes(offset, length) {
    const obj = await this.bucket.get(this.key, {
      range: { offset, length },
    });
    const data = await obj.arrayBuffer();
    return { data: new Uint8Array(data) };
  }
}
```

### Supabase — Full Platform

```sql
-- Supabase: Enable PostGIS
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create spatial table
CREATE TABLE features (
    id BIGSERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    geom GEOMETRY(Geometry, 4326),
    created_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Spatial index
CREATE INDEX idx_features_geom ON features USING GIST (geom);

-- Row Level Security (spatial)
ALTER TABLE features ENABLE ROW LEVEL SECURITY;

-- Users can only see features in their assigned region
CREATE POLICY "Users see features in their region"
ON features FOR SELECT
USING (
    ST_Within(
        geom,
        (SELECT region_geom FROM user_regions WHERE user_id = auth.uid())
    )
);

-- RPC function for spatial queries
CREATE FUNCTION features_in_bbox(min_lng float, min_lat float, max_lng float, max_lat float)
RETURNS json AS $$
    SELECT json_build_object(
        'type', 'FeatureCollection',
        'features', COALESCE(json_agg(json_build_object(
            'type', 'Feature',
            'properties', json_build_object('id', id, 'name', name, 'category', category),
            'geometry', ST_AsGeoJSON(geom)::json
        )), '[]'::json)
    )
    FROM features
    WHERE geom && ST_MakeEnvelope(min_lng, min_lat, max_lng, max_lat, 4326)
    LIMIT 5000;
$$ LANGUAGE sql STABLE;
```

### Cost Comparison

| Stack | 1K/day | 10K/day | 100K/day |
|-------|--------|---------|----------|
| Docker Compose (VPS) | $10-20 | $40-80 | N/A |
| Kubernetes (AWS/GCP) | $100-200 | $200-500 | $500-2000 |
| Serverless (Lambda) | $5-15 | $30-100 | $200-800 |
| Static (PMTiles+CDN) | $0-5 | $5-15 | $15-50 |
| Supabase | $0-25 | $25-100 | $100-400 |
| Managed (Mapbox/CARTO) | $0-100 | $200-500 | $1000-5000 |

---

## CI/CD for GIS

### GitHub Actions — Complete Workflow

```yaml
# .github/workflows/gis-ci.yml
name: GIS CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgis:
        image: postgis/postgis:16-3.4
        env:
          POSTGRES_DB: testdb
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
        ports: ["5432:5432"]
        options: >-
          --health-cmd "pg_isready -U test -d testdb"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with: { node-version: '20' }

      - name: Install dependencies
        run: npm ci

      - name: Lint
        run: npm run lint

      - name: Type check
        run: npm run typecheck

      - name: Run spatial tests
        run: npm test
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/testdb

      - name: Validate map styles
        run: |
          npx @maplibre/maplibre-gl-style-spec validate src/styles/*.json

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          push: ${{ github.ref == 'refs/heads/main' }}
          tags: ghcr.io/${{ github.repository }}/app:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  generate-tiles:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Install tippecanoe
        run: |
          sudo apt-get update && sudo apt-get install -y git build-essential libsqlite3-dev zlib1g-dev
          git clone https://github.com/felt/tippecanoe.git
          cd tippecanoe && make -j && sudo make install

      - name: Generate tiles
        run: |
          tippecanoe -o tiles/buildings.pmtiles \
            --minimum-zoom=2 --maximum-zoom=14 \
            --drop-densest-as-needed \
            data/buildings.geojson

      - name: Upload to R2
        run: |
          rclone copy tiles/ r2:${{ secrets.R2_BUCKET }}/tiles/
        env:
          RCLONE_CONFIG_R2_TYPE: s3
          RCLONE_CONFIG_R2_PROVIDER: Cloudflare
          RCLONE_CONFIG_R2_ACCESS_KEY_ID: ${{ secrets.R2_ACCESS_KEY }}
          RCLONE_CONFIG_R2_SECRET_ACCESS_KEY: ${{ secrets.R2_SECRET_KEY }}
          RCLONE_CONFIG_R2_ENDPOINT: ${{ secrets.R2_ENDPOINT }}

  deploy:
    needs: [build, generate-tiles]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4

      - name: Deploy to VPS
        run: |
          ssh ${{ secrets.DEPLOY_HOST }} "
            cd /opt/gis-app &&
            docker compose pull &&
            docker compose up -d --remove-orphans &&
            docker system prune -f
          "

      - name: Smoke test
        run: |
          sleep 10
          curl -sf https://gis.example.com/health || exit 1
          curl -sf https://gis.example.com/tiles/buildings/12/3421/1564 -o /dev/null || exit 1
```

### Database Migrations

```bash
# Using golang-migrate
# Create migration
migrate create -ext sql -dir migrations add_buildings_table

# migrations/000001_add_buildings_table.up.sql
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE buildings (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    height FLOAT,
    type TEXT,
    geom GEOMETRY(Polygon, 4326)
);

CREATE INDEX idx_buildings_geom ON buildings USING GIST (geom);

# Run migrations
migrate -path migrations -database "${DATABASE_URL}" up
```

---

## Monitoring & Observability

### Prometheus + Grafana Dashboards

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'martin'
    static_configs:
      - targets: ['martin:3000']
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'caddy'
    static_configs:
      - targets: ['caddy:2019']
```

### Key Metrics & Alerts

```yaml
# alerting rules
groups:
  - name: gis-alerts
    rules:
      - alert: HighTileLatency
        expr: histogram_quantile(0.95, rate(tile_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels: { severity: warning }
        annotations: { summary: "Tile latency p95 > 1s" }

      - alert: PostGISConnectionsHigh
        expr: pg_stat_activity_count > 80
        for: 5m
        labels: { severity: critical }
        annotations: { summary: "PostGIS connections > 80%" }

      - alert: DiskSpaceLow
        expr: node_filesystem_avail_bytes{mountpoint="/var/lib/postgresql"} / node_filesystem_size_bytes < 0.1
        for: 10m
        labels: { severity: critical }
        annotations: { summary: "PostGIS disk space < 10%" }

      - alert: TileErrorRate
        expr: rate(tile_request_errors_total[5m]) / rate(tile_request_total[5m]) > 0.05
        for: 5m
        labels: { severity: warning }
        annotations: { summary: "Tile error rate > 5%" }
```

### OpenTelemetry Tracing

```python
# Trace a spatial request end-to-end
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor

tracer = trace.get_tracer(__name__)

FastAPIInstrumentor.instrument_app(app)
AsyncPGInstrumentor().instrument()

@app.get("/features")
async def get_features(bbox: str):
    with tracer.start_as_current_span("get_features") as span:
        span.set_attribute("bbox", bbox)
        span.set_attribute("service", "feature-api")

        with tracer.start_as_current_span("parse_bbox"):
            coords = [float(c) for c in bbox.split(",")]

        with tracer.start_as_current_span("query_postgis"):
            result = await db.fetch(...)
            span.set_attribute("feature_count", len(result))

        with tracer.start_as_current_span("build_geojson"):
            geojson = build_feature_collection(result)

        return geojson
```

---

## Disaster Recovery & Backup

### PostGIS Backup Strategy

```bash
#!/bin/bash
# backup.sh — Comprehensive PostGIS backup

DB_URL="postgresql://gisuser:pass@localhost/gisdb"
BACKUP_DIR="/backups"
S3_BUCKET="s3://gis-backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 1. Custom format dump (most flexible)
pg_dump -Fc -f "${BACKUP_DIR}/gisdb_${TIMESTAMP}.dump" "${DB_URL}"

# 2. Verify backup
pg_restore --list "${BACKUP_DIR}/gisdb_${TIMESTAMP}.dump" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERROR: Backup verification failed"
    exit 1
fi

# 3. Upload to S3
aws s3 cp "${BACKUP_DIR}/gisdb_${TIMESTAMP}.dump" "${S3_BUCKET}/daily/"

# 4. Cleanup old local backups (keep 7 days)
find "${BACKUP_DIR}" -name "*.dump" -mtime +7 -delete

# 5. WAL archiving for point-in-time recovery
# postgresql.conf:
#   archive_mode = on
#   archive_command = 'aws s3 cp %p s3://gis-backups/wal/%f'
```

### Multi-Region Deployment

```
                    ┌──────────────┐
                    │  DNS (Route53 │
                    │  / Cloudflare)│
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──┐  ┌─────▼─────┐  ┌──▼────────┐
       │ US-East │  │ EU-West   │  │ AP-East   │
       │ Region  │  │ Region    │  │ Region    │
       │         │  │           │  │           │
       │ Martin  │  │ Martin    │  │ Martin    │
       │ App     │  │ App       │  │ App       │
       │ PostGIS │  │ PostGIS   │  │ PostGIS   │
       │ (primary│  │ (replica) │  │ (replica) │
       └─────────┘  └───────────┘  └───────────┘
              │                         ▲
              └─────── WAL replication ──┘
```

---

## Deployment Comparison

| Approach | Complexity | Cost | Scale | Recovery | Best For |
|----------|-----------|------|-------|----------|----------|
| Docker Compose (VPS) | Low | $10-40/mo | Limited | Manual | Small teams |
| Kubernetes | High | $100-2000/mo | High | Automated | Enterprise |
| Serverless | Medium | $0-800/mo | Auto | Built-in | Bursty workloads |
| Static (PMTiles+CDN) | Very Low | $0-50/mo | Global | N/A | Static data |
| Supabase | Low | $0-400/mo | Medium | Built-in | Rapid prototyping |
