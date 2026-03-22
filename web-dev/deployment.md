# Deployment -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**Development/staging:** Docker Compose. **Small team production (< 100 concurrent):** Docker Compose on $20-40/mo VPS. **Enterprise scale:** Kubernetes with managed PostGIS. **Static data at global scale:** PMTiles on Cloudflare R2 ($0-5/mo). **Monitoring:** Prometheus + Grafana + OpenTelemetry (non-negotiable at scale).

---

## Tier 1 -- Production First Choices

---

### Kubernetes -- Enterprise Orchestration

The standard for large-team, multi-service GIS deployments. Auto-scaling, rolling updates, health-based routing, secrets management. **Use K8s when your team has 5+ engineers and production demands justify the operational cost.**

**Why Tier 1:** Native Prometheus scraping, liveness/readiness probes, and kubectl/Helm/ArgoCD pipelines make operations manageable at scale. Requires at least 1 dedicated DevOps engineer.

```yaml
# martin-deployment.yaml -- with HPA
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
          readinessProbe:
            httpGet: { path: /health, port: 3000 }
---
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
```

**Caveats:**
- **PostGIS on K8s is risky.** StatefulSet for databases is operationally challenging. PV resizing, backup CronJobs, failover -- all manual. **Use managed databases (RDS, Cloud SQL) instead.**
- **Tile cache coherence.** Multiple Martin pods with in-memory caches will have different cache states.
- **HPA flapping.** PostGIS query latency spikes during vacuum/analyze can cause HPA to scale up Martin pods unnecessarily. Custom metrics (tile request rate) are better than CPU for HPA triggers.
- **Secret management.** DATABASE_URL in Kubernetes Secrets is base64-encoded, not encrypted. Use Sealed Secrets or External Secrets Operator.
- **Cost.** Managed K8s (EKS, GKE) has a control plane cost ($75/mo for EKS) plus node costs.
- **Anti-pattern: Using Kubernetes for a small team with < 100 concurrent users.** Docker Compose on a $20/month VPS is 10x simpler and sufficient.

---

### Docker / Docker Compose -- The Foundation

Every GIS stack starts with Docker. Compose is the standard for development, staging, and small-scale production.

**Why Tier 1:** Essential for development and CI/CD. Production Compose deployments work for single-server scenarios up to ~100 concurrent users.

```yaml
# docker-compose.yml -- PostGIS + Martin + Frontend
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
    # NEVER expose in production:
    # ports: ["5432:5432"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gisuser -d gisdb"]
      interval: 10s
      timeout: 5s
      retries: 5
    deploy:
      resources:
        limits: { memory: 4G, cpus: '2' }

  martin:
    image: ghcr.io/maplibre/martin
    environment:
      DATABASE_URL: postgresql://gisuser:${DB_PASSWORD}@postgis/gisdb
    ports: ["3000:3000"]
    depends_on:
      postgis: { condition: service_healthy }
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL

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

**Caveats:**
- **Compose is not production orchestration.** No auto-restart across hosts, no rolling updates, no health-based traffic routing. Single-VPS deployments only.
- **Volume data loss.** `docker compose down -v` destroys your PostGIS data. One junior dev mistake = data loss. Named volumes + backup strategy is mandatory.
- **Network isolation gaps.** Default bridge network means all containers can talk to each other. PostGIS should not be accessible from the frontend container.
- **Resource limits are off by default.** Without `deploy.resources.limits`, a runaway Martin or PostGIS process can OOM the host.
- **Log management.** Default JSON log driver fills disk. Must configure log rotation or ship to a log aggregator.
- **Anti-pattern: Exposing PostGIS port 5432 in production.** Only expose through internal Docker network. Use Nginx/Caddy as the public-facing proxy.

---

### Prometheus + Grafana + OpenTelemetry -- Monitoring Stack

**Non-negotiable for any production GIS deployment.** Tile latency, PostGIS connections, disk space, error rates -- you must be able to observe these.

**Why Tier 1:** Industry standard. Martin exposes Prometheus metrics natively. postgres_exporter covers database health. OpenTelemetry traces spatial request flows end-to-end.

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
```

```yaml
# Key alert rules
groups:
  - name: gis-alerts
    rules:
      - alert: HighTileLatency
        expr: histogram_quantile(0.95, rate(tile_request_duration_seconds_bucket[5m])) > 1
        for: 5m
      - alert: PostGISConnectionsHigh
        expr: pg_stat_activity_count > 80
        for: 5m
      - alert: DiskSpaceLow
        expr: node_filesystem_avail_bytes{mountpoint="/var/lib/postgresql"} / node_filesystem_size_bytes < 0.1
        for: 10m
```

**Caveats:**
- **Prometheus storage.** Default local storage is not durable. For production, need Thanos or Mimir for long-term retention and HA.
- **Cardinality explosion.** Tile server metrics with per-tile-coordinate labels create millions of time series and crash Prometheus. Use bucket labels (zoom level only), never full z/x/y.
- **Grafana dashboard sprawl.** Easy to create dashboards, hard to maintain them. Dashboard-as-code (Terraform/Jsonnet) is recommended.
- **Alert fatigue.** Without careful tuning, alerts for tile latency fire during normal traffic spikes. Start with lenient thresholds.
- **OpenTelemetry SDK overhead.** Tracing every tile request adds 1-5ms latency. Sample at 1-10% in production.

### Observability Checklist

| Component | Metrics Endpoint | Key Indicators | Alert Threshold |
|-----------|-----------------|----------------|-----------------|
| Martin | `/metrics` (Prometheus) | tile_request_duration, cache_hit_ratio | p95 latency > 1s |
| PostGIS | postgres_exporter:9187 | pg_stat_activity, dead tuples, replication lag | connections > 80% max |
| Nginx tile cache | access log / stub_status | cache hit rate, 5xx rate | hit rate < 80% |
| Application API | OpenTelemetry SDK | request duration, error rate | error rate > 1% |
| Disk | node_exporter | filesystem available | < 10% free |

---

### Serverless -- Lambda / Cloudflare Workers / Supabase

Auto-scaling, pay-per-use infrastructure for specific GIS workloads.

| Platform | Best For | Key Limitation |
|----------|----------|----------------|
| **Lambda** | TiTiler raster serving | 3-8s cold starts with GDAL |
| **Cloudflare Workers** | PMTiles protocol handler | 1MB bundle limit |
| **Supabase** | Rapid prototyping with PostGIS | Connection limits, vendor lock-in |

```javascript
// Cloudflare Worker -- Serve PMTiles from R2
import { PMTiles } from 'pmtiles';

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const match = url.pathname.match(/^\/tiles\/(\w+)\/(\d+)\/(\d+)\/(\d+)\.(mvt|pbf)$/);
    if (!match) return new Response('Not found', { status: 404 });

    const [, name, z, x, y] = match;
    const pmtiles = new PMTiles(new R2Source(env.TILES_BUCKET, `${name}.pmtiles`));
    const tile = await pmtiles.getZxy(Number(z), Number(x), Number(y));
    if (!tile?.data) return new Response('', { status: 204 });

    return new Response(tile.data, {
      headers: {
        'Content-Type': 'application/x-protobuf',
        'Cache-Control': 'public, max-age=86400',
        'Access-Control-Allow-Origin': '*',
      },
    });
  },
};
```

**Caveats:**
- **Lambda cold starts with GDAL.** TiTiler on Lambda has 3-8 second cold starts. Provisioned concurrency costs $15-50/mo per concurrent execution.
- **Cloudflare Workers size limits.** 1MB compressed worker size. Cannot bundle large WASM modules (DuckDB-WASM, GDAL).
- **Supabase connection pooling.** Under heavy PostGIS query load, connection limits can be hit. Pro plan ($25/mo) caps at 200 connections.
- **Vendor lock-in.** Each platform's API and deployment model is different. Migrating from Lambda to Workers is a rewrite.
- **Cost unpredictability.** Per-request pricing means a bot scanning your tile endpoint can generate a surprise bill. Rate limiting is essential.
- **Anti-pattern: Serverless for high-throughput tile serving.** Per-request pricing makes serverless expensive at scale. PMTiles on CDN is cheaper.

---

## CI/CD for GIS

### GitHub Actions -- Complete Workflow

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
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: npm ci
      - run: npm run lint && npm run typecheck
      - run: npm test
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/testdb
      - run: npx @maplibre/maplibre-gl-style-spec validate src/styles/*.json

  generate-tiles:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Generate tiles
        run: |
          # WARNING: tippecanoe can OOM on runners with 7GB RAM for large datasets
          tippecanoe -o tiles/buildings.pmtiles \
            --minimum-zoom=2 --maximum-zoom=14 \
            --drop-densest-as-needed \
            data/buildings.geojson
      - name: Upload to R2
        run: rclone copy tiles/ r2:${{ secrets.R2_BUCKET }}/tiles/
```

**Caveats:**
- **Runner memory limits.** GitHub Actions runners have ~7GB RAM. Large tippecanoe jobs will OOM. Use self-hosted runners for planet-scale tile generation.
- **PostGIS container startup.** The PostGIS service container takes 5-15 seconds to start. Tests must wait for the health check.
- **Style validation.** `maplibre-gl-style-spec validate` catches style JSON errors before deployment but cannot verify visual rendering.

---

## Disaster Recovery & Backup

### PostGIS Backup Strategy

```bash
#!/bin/bash
# backup.sh -- production PostGIS backup
DB_URL="postgresql://gisuser:pass@localhost/gisdb"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Custom format dump (most flexible for restore)
pg_dump -Fc -f "/backups/gisdb_${TIMESTAMP}.dump" "${DB_URL}"

# Verify backup
pg_restore --list "/backups/gisdb_${TIMESTAMP}.dump" > /dev/null 2>&1 || exit 1

# Upload to S3
aws s3 cp "/backups/gisdb_${TIMESTAMP}.dump" "s3://gis-backups/daily/"

# Cleanup old local backups (keep 7 days)
find "/backups" -name "*.dump" -mtime +7 -delete
```

**Caveats:**
- **pg_dump locks tables briefly.** On large spatial tables (100M+ rows), this can cause query timeouts. Use `--no-synchronized-snapshots` for parallel dumps on replicas.
- **Backup size for spatial data.** PostGIS databases with large geometries produce large dumps. A 50GB database can produce a 30GB dump.
- **Point-in-time recovery requires WAL archiving.** `pg_dump` gives you snapshots, not continuous recovery. Enable `archive_mode = on` for RPO near zero.
- **Never rely on a single backup location.** S3 replication or cross-region copies are essential.

---

## Cost Comparison

| Stack | 1K users/day | 10K users/day | 100K users/day | Min Team |
|-------|-------------|--------------|----------------|----------|
| Docker Compose (VPS) | $10-20 | $40-80 | N/A (scale limit) | 1 |
| Kubernetes (AWS/GCP) | $100-200 | $200-500 | $500-2000 | 3+ (1 DevOps) |
| Serverless (Lambda) | $5-15 | $30-100 | $200-800 | 1 |
| Static (PMTiles+CDN) | $0-5 | $5-15 | $15-50 | 1 |
| Supabase | $0-25 | $25-100 | $100-400 | 1 |

---

## Tier 2 -- Specialized / Complementary

---

### Caddy / Nginx -- Reverse Proxy & Tile Cache

**Essential for any production deployment.** Put a reverse proxy in front of Martin and PostGIS.

**Why Tier 2 (complementary):** Caddy for simplicity (auto HTTPS, zero config). Nginx for maximum control and tile caching at scale. Not standalone deployment solutions -- they complement Tier 1 approaches.

```
# Caddyfile -- Auto HTTPS reverse proxy
example.com {
    handle /tiles/* {
        reverse_proxy martin:3000
        header Cache-Control "public, max-age=86400, stale-while-revalidate=604800"
        header Access-Control-Allow-Origin "*"
    }
    handle /api/* {
        reverse_proxy frontend:3000
    }
    handle {
        reverse_proxy frontend:3000
    }
    encode zstd gzip
}
```

```nginx
# Nginx tile cache -- critical for production Martin deployments
proxy_cache_path /var/cache/tiles levels=1:2 keys_zone=tiles:100m max_size=50g inactive=30d;

location /tiles/ {
    proxy_pass http://martin:3000/;
    proxy_cache tiles;
    proxy_cache_valid 200 24h;
    proxy_cache_valid 404 1m;
    proxy_cache_use_stale error timeout updating;
    proxy_cache_lock on;
    add_header X-Cache-Status $upstream_cache_status;
    add_header Cache-Control "public, max-age=86400";
}
```

**Caveats:**
- **Nginx tile caching requires careful configuration.** `proxy_cache_path`, `proxy_cache_key`, and `proxy_cache_valid` must be set correctly or you'll cache wrong tiles.
- **Caddy is newer with smaller enterprise footprint.** Less battle-tested than Nginx for massive scale.
- **Anti-pattern: Not enabling compression for tiles.** Brotli/gzip on MVT tiles reduces bandwidth 60-70%. Always enable.

---

### Vercel / Netlify -- Frontend Hosting

CDN-backed hosting for Next.js, SvelteKit, and static frontends. **Not for backend GIS services.**

**Caveats:**
- **Cannot run PostGIS, Martin, or long-running processes.** Frontend hosting only.
- **Edge function limits.** Execution time, memory, and bundle size limits constrain spatial processing.
- **Cost at scale.** Enterprise features and high bandwidth can become expensive.

Combined with PMTiles on CDN, Vercel/Netlify gives you a complete static GIS stack for $0-5/mo.
