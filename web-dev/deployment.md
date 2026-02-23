# Deployment

Taking a GIS application from development to production involves containerization, infrastructure choices, CI/CD pipelines, and monitoring. This guide covers deployment strategies from simple Docker setups to scalable cloud architectures.

> **Quick Picks**
> - **Cheapest deployment:** PMTiles on Cloudflare R2 + frontend on Vercel/Netlify
> - **Full-stack local dev:** Docker Compose (PostGIS + Martin + Frontend)
> - **Production small team:** Docker Compose on a $20/mo VPS
> - **Production at scale:** Kubernetes with managed PostGIS (RDS/Cloud SQL)
> - **Serverless raster:** Lambda + TiTiler + S3 COGs

---

## Docker for GIS Stack

Docker simplifies deploying the GIS stack by packaging each service with its dependencies.

### Common GIS Docker Images

| Image | Purpose | Base |
|-------|---------|------|
| `postgis/postgis` | PostgreSQL + PostGIS | Debian |
| `kartoza/geoserver` | GeoServer with plugins | Ubuntu |
| `ghcr.io/maplibre/martin` | Martin tile server | Alpine |
| `pramsey/pg_tileserv` | pg_tileserv | Alpine |
| `pelias/api` | Pelias geocoding API | Node |
| `osrm/osrm-backend` | OSRM routing engine | Debian |

### Example Docker Compose

```yaml
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
    ports:
      - "5432:5432"

  martin:
    image: ghcr.io/maplibre/martin
    environment:
      DATABASE_URL: postgresql://gisuser:${DB_PASSWORD}@postgis/gisdb
    ports:
      - "3000:3000"
    depends_on:
      - postgis

  frontend:
    build: ./frontend
    ports:
      - "8080:80"

volumes:
  pgdata:
```

### Full Docker Compose: PostGIS + Martin + Frontend + GeoServer

```yaml
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
    ports:
      - "5432:5432"

  martin:
    image: ghcr.io/maplibre/martin
    environment:
      DATABASE_URL: postgresql://gisuser:${DB_PASSWORD}@postgis/gisdb
    ports:
      - "3000:3000"
    depends_on:
      - postgis

  geoserver:
    image: kartoza/geoserver:2.24.1
    environment:
      GEOSERVER_ADMIN_PASSWORD: ${GS_PASSWORD}
    ports:
      - "8080:8080"
    volumes:
      - gs_data:/opt/geoserver/data_dir

  osrm:
    image: osrm/osrm-backend
    command: osrm-routed --algorithm mld /data/map.osrm
    volumes:
      - ./osrm-data:/data
    ports:
      - "5001:5000"

  frontend:
    build: ./frontend
    ports:
      - "80:80"
    depends_on:
      - martin

volumes:
  pgdata:
  gs_data:
```

---

## Kubernetes / Cloud Deployment

For production workloads requiring high availability and auto-scaling.

### Architecture

```
┌─────────────────────────────────────────────┐
│  Kubernetes Cluster                         │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ Frontend  │  │  Tile    │  │  API     │ │
│  │ Deployment│  │  Server  │  │  Server  │ │
│  │ (3 pods)  │  │ (3 pods) │  │ (3 pods) │ │
│  └─────┬─────┘  └─────┬────┘  └─────┬────┘ │
│        │              │             │       │
│  ┌─────▼──────────────▼─────────────▼────┐  │
│  │            Ingress / Load Balancer     │  │
│  └───────────────────────────────────────┘  │
│                                             │
│  ┌──────────────────┐  ┌────────────────┐  │
│  │  PostGIS StatefulSet │  │  Redis Cache  │  │
│  │  (primary + replica) │  │  (3 pods)     │  │
│  └──────────────────┘  └────────────────┘  │
└─────────────────────────────────────────────┘
```

### Cloud Provider Options

| Provider | Managed Database | Object Storage | CDN | Compute |
|----------|-----------------|----------------|-----|---------|
| AWS | RDS for PostgreSQL | S3 | CloudFront | ECS / EKS |
| GCP | Cloud SQL | GCS | Cloud CDN | GKE / Cloud Run |
| Azure | Azure Database | Blob Storage | Azure CDN | AKS / Container Apps |

### Terraform Example (AWS RDS + ECS)

```hcl
resource "aws_db_instance" "postgis" {
  engine               = "postgres"
  engine_version       = "16"
  instance_class       = "db.t3.medium"
  allocated_storage    = 100
  db_name              = "gisdb"
  username             = var.db_user
  password             = var.db_password
  publicly_accessible  = false
}

resource "aws_ecs_service" "martin" {
  name            = "martin-tiles"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.martin.arn
  desired_count   = 2
  launch_type     = "FARGATE"
}
```

---

## Serverless GIS

Serverless architectures minimize infrastructure management and scale to zero when idle.

### Lambda + S3 COG

Serve raster data from Cloud Optimized GeoTIFFs on S3 using serverless compute.

```
Client  ──▶  API Gateway  ──▶  Lambda (TiTiler)  ──▶  S3 (COG files)
```

- **Pros**: Scale to zero, pay-per-request, no server management
- **Cons**: Cold starts, limited execution time, complex debugging

### Cloud Functions + GeoParquet

Process vector data from GeoParquet files on cloud storage.

### Serverless Use Cases

| Use Case | Architecture | Cost Profile |
|----------|-------------|-------------|
| Raster tile serving | Lambda + S3 COG + TiTiler | Low (pay per request) |
| Vector tile serving | CDN + PMTiles | Very Low (storage + CDN) |
| Geocoding API | Cloud Function + Pelias | Low-Medium |
| Geoprocessing jobs | Step Functions + Lambda | Variable |

### PMTiles on Cloudflare R2 (Cheapest Deployment)

Cloudflare R2 offers free egress, making it ideal for serving PMTiles globally.

**Setup:**
1. Create an R2 bucket in Cloudflare dashboard
2. Upload your `.pmtiles` file
3. Set CORS headers: `Access-Control-Allow-Origin: *`
4. Enable public access or use a Worker for custom routing
5. Use the PMTiles JS library in MapLibre to read tiles via range requests

**Cost:** Free for most use cases (10 GB storage free, no egress charges).

### Vercel/Netlify for Frontend + CDN Tiles

Deploy your frontend (Next.js, Vite, SvelteKit) to Vercel or Netlify. Serve tiles from R2/S3.

```
Frontend (Vercel)  -->  Tile Source (R2/S3 PMTiles)
     |                       |
     v                       v
  Next.js app          Range requests
  SSR/SSG              No server needed
  API routes           Global CDN
```

---

## Static Map Hosting

The simplest and cheapest deployment option: pre-generate tiles and host them on static infrastructure.

### PMTiles + CDN

PMTiles is a single-file tile archive format that supports HTTP range requests, making it ideal for static hosting.

```
┌──────────┐     Range Requests     ┌──────────┐
│  Browser  │  ───────────────────▶  │  CDN /   │
│ (MapLibre │                        │  S3      │
│  + pmtiles│                        │ (.pmtiles│
│  protocol)│  ◀───────────────────  │  file)   │
└──────────┘     Tile Bytes          └──────────┘
```

**Steps:**
1. Generate PMTiles with tippecanoe: `tippecanoe -o tiles.pmtiles input.geojson`
2. Upload to S3 / GCS / R2 with CORS headers
3. Point MapLibre to the PMTiles URL using the pmtiles protocol

- **Pros**: No server, no database, extremely cheap, globally fast via CDN
- **Cons**: Static data only, requires regeneration on updates

### Static Hosting Providers

| Provider | Free Tier | Best For |
|----------|-----------|----------|
| Cloudflare R2 | 10 GB storage, free egress | PMTiles, static assets |
| AWS S3 + CloudFront | 5 GB storage | Full-featured CDN |
| GitHub Pages | 1 GB storage | Small demos, prototypes |
| Netlify / Vercel | 100 GB bandwidth | Frontend + PMTiles |


---

## CI/CD for GIS Projects

### Pipeline Stages

```
Code Push  ──▶  Lint/Test  ──▶  Build  ──▶  Deploy  ──▶  Validate
                  │                │           │            │
              Unit tests       Docker      Staging      Smoke tests
              Linting          images      then Prod    Tile checks
              Type check       Tile gen                 API health
```

### GIS-Specific CI Checks

- Validate GeoJSON schema and geometry validity
- Run spatial query tests against a PostGIS test database
- Check tile generation output (size, zoom levels, coverage)
- Verify CRS consistency across data sources
- Lint map styles (MapLibre style spec validation)

### GitHub Actions Workflow Example

```yaml
name: GIS CI/CD
on: [push]
jobs:
  test-and-deploy:
    runs-on: ubuntu-latest
    services:
      postgis:
        image: postgis/postgis:16-3.4
        env:
          POSTGRES_DB: testdb
          POSTGRES_PASSWORD: testpass
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - name: Run spatial tests
        run: |
          pip install geopandas pytest sqlalchemy geoalchemy2
          pytest tests/
      - name: Build frontend
        run: cd frontend && npm ci && npm run build
      - name: Deploy to Vercel
        if: github.ref == 'refs/heads/main'
        run: npx vercel --prod --token=${{ secrets.VERCEL_TOKEN }}
```

---

## Monitoring and Observability

### Key Metrics

| Metric | Target | Tool |
|--------|--------|------|
| Tile response time (p95) | < 200ms cached, < 1s uncached | Prometheus + Grafana |
| API response time (p95) | < 500ms | Prometheus + Grafana |
| Error rate | < 1% | Sentry, Datadog |
| Database connection pool usage | < 80% | pg_stat_activity |
| Tile cache hit ratio | > 90% | CDN analytics |
| Frontend FPS | > 30fps during interaction | Real User Monitoring |

### Logging

- Structured JSON logs with request metadata (bbox, zoom, layer)
- Slow query logging in PostGIS (`log_min_duration_statement`)
- Tile server access logs with response sizes


---

## Cost Optimization Tips

- **Use PMTiles + CDN** for static data instead of running tile servers
- **Right-size containers**: PostGIS needs memory; tile servers need CPU
- **Cache aggressively**: tiles rarely change, set long Cache-Control headers
- **Use spot/preemptible instances** for batch geoprocessing
- **Compress everything**: ZSTD for COGs, gzip/brotli for vector tiles
- **Scale to zero**: use serverless for infrequent workloads
- **Monitor egress**: CDN egress costs can surprise; use Cloudflare R2 for free egress

---

## Deployment Comparison

| Approach | Complexity | Cost | Scale | Best For |
|----------|-----------|------|-------|----------|
| Docker Compose (single server) | Low | Low | Limited | Small teams, dev/staging |
| Kubernetes | High | Medium-High | High | Enterprise, multi-service |
| Serverless (Lambda + S3) | Medium | Low-Variable | Auto | Bursty workloads |
| Static (PMTiles + CDN) | Very Low | Very Low | Global | Static tile datasets |
| Managed services (Mapbox, CARTO) | Low | High | High | Fast time-to-market |
### Cost Estimates at Scale

| Approach | 1K users/day | 10K users/day | 100K users/day |
|---|---|---|---|
| Docker Compose (VPS) | $10-20/mo | $40-80/mo | Not recommended |
| Kubernetes (cloud) | $100-200/mo | $200-500/mo | $500-2000/mo |
| Serverless (Lambda) | $5-15/mo | $30-100/mo | $200-800/mo |
| Static (PMTiles + CDN) | $0-5/mo | $5-15/mo | $15-50/mo |
| Managed (Mapbox/CARTO) | $0-100/mo | $200-500/mo | $1000-5000/mo |
