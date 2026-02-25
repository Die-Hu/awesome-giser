# Automation Workflows

> Expert prompts for batch processing, ETL pipelines, report generation, quality assurance, CI/CD, Claude Code agent workflows, project bootstrap, and infrastructure-as-code -- all tuned for geospatial workloads.

**Quick Picks**
- **Most Useful**: [P21 -- Full GIS Project Bootstrap](#prompt-21--full-gis-project-bootstrap) -- set up an entire GIS project from scratch in one prompt
- **Time Saver**: [P17-P20 -- Claude Code Agent Workflows](#6-claude-code-agent-workflows) -- leverage agentic AI for multi-file GIS development
- **Cutting Edge**: [P15 -- Spatial Data Validation in CI](#prompt-15--spatial-data-validation-in-ci) -- automated data quality gates in your CI pipeline

---

## Table of Contents

1. [Batch Processing](#1-batch-processing) (P1--P3)
2. [ETL Pipelines](#2-etl-pipelines) (P4--P7)
3. [Report Generation](#3-report-generation) (P8--P10)
4. [Quality Assurance](#4-quality-assurance) (P11--P13)
5. [CI/CD for GIS](#5-cicd-for-gis) (P14--P16)
6. [Claude Code Agent Workflows](#6-claude-code-agent-workflows) (P17--P20)
7. [Project Bootstrap](#7-project-bootstrap) (P21--P23)
8. [Infrastructure as Code](#8-infrastructure-as-code) (P24--P25)

---

## 1. Batch Processing

```
                          Batch Processing Pipeline
  +-----------+      +-----------+      +-----------+      +----------+
  | P1: Batch |----->| P2: Watch |----->| P3: GDAL  |----->| Combined |
  | Geoprocess|      | Folder    |      | Pipeline  |      | Daemon   |
  | (parallel)|      | (daemon)  |      | (Makefile)|      | System   |
  +-----------+      +-----------+      +-----------+      +----------+
       |                   |                  |
       v                   v                  v
  concurrent.futures  watchdog+systemd   make -j parallel
```

---

### Prompt 1 -- Parallel Batch Geoprocessing

#### Scenario (实际场景)

You have hundreds or thousands of spatial files that need identical processing steps. Manual execution is infeasible; you need parallel, resumable batch processing with full audit trails.

#### Roles

Act as a Senior GIS Automation Engineer building a production-grade batch processing CLI.

#### Prompt Template

```text
Write a Python CLI script for parallel batch geoprocessing.

INPUT:
- Directory of [FILE_TYPE, e.g., GeoTIFF rasters / GeoPackage vectors]
- Boundary file for clipping: [BOUNDARY_PATH]
- Target CRS: [EPSG_CODE]

PROCESSING per file:
1. [STEP_1, e.g., "Clip to study area boundary"]
2. [STEP_2, e.g., "Reproject to TARGET_CRS"]
3. [STEP_3, e.g., "Calculate NDVI as new band"]
4. [STEP_4, e.g., "Resample to 10m bilinear"]

REQUIREMENTS:
- concurrent.futures.ProcessPoolExecutor with [N] workers
- Idempotent: skip files whose output already exists and is newer
- --dry-run flag: list work without executing
- Structured JSON logging: file, status, duration_s, output_path, error
- On failure: log error, continue batch, copy source to errors/
- Summary CSV at end: filename, status, duration, output_bytes
- CLI via Typer: input_dir, output_dir, --boundary, --crs, --workers,
  --dry-run, --verbose
- Use [LIBRARY: rasterio / geopandas+pyogrio] with context managers
- pathlib.Path throughout; type hints on all functions

Generate a single self-contained script with if __name__ == "__main__".
```

#### Variables to Customize

- `FILE_TYPE` -- GeoTIFF, GeoPackage, GeoJSON, Shapefile
- `STEP_1..N` -- processing operations per file
- `LIBRARY` -- rasterio for rasters, geopandas for vectors
- `N` -- worker count (default: cpu_count - 1)
- `EPSG_CODE` -- target coordinate reference system

#### Expected Output

A single Python file (~200 lines) with Typer CLI, parallel execution, idempotent skip logic, JSON logging, summary CSV, and --dry-run support.

#### Validation Checklist

- [ ] Running twice produces identical output (idempotent)
- [ ] --dry-run lists files without side effects
- [ ] Failed files do not halt the batch
- [ ] Summary CSV row count matches input file count
- [ ] JSON log entries contain all required fields

#### Cost Optimization

Use --dry-run first to verify file counts. Set --workers to match vCPU count on cloud VMs. For cloud batch, use spot/preemptible instances for 60-70% cost savings.

**Dark Arts Tip:** Wrap the whole thing in GNU `parallel` if your files are on a shared NFS mount and you want to distribute across multiple machines without a scheduler.

#### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- rasterio, geopandas, shapely
- [CLI Tools](../tools/cli-tools.md) -- GDAL command-line utilities

#### Extensibility

Chain with P2 (Watch Folder) to auto-trigger batches on new data. Add Prometheus metrics endpoint for long-running production batches.

---

### Prompt 2 -- Watch Folder Auto-Processor

#### Scenario (实际场景)

Your team receives data drops from field crews or external partners into a shared folder. You need an always-on daemon that detects new files, validates them, processes them, and archives the originals -- all without human intervention.

#### Roles

Act as a DevOps Engineer building a file-watching daemon for automated spatial data processing.

#### Prompt Template

```text
Write a Python daemon that watches a directory for new spatial files
and auto-processes them.

WATCH CONFIG:
- Directory: [WATCH_DIR]
- File pattern: [GLOB, e.g., "*.gpkg", "*.tif"]
- Polling interval: [SECONDS, e.g., 5]

ON NEW FILE:
1. Wait for file stability (size unchanged for 10s)
2. Validate: readable, expected CRS [EPSG], expected schema/bands
3. Process: [PROCESSING_STEPS]
4. On success: move output to [OUTPUT_DIR], archive original to
   [ARCHIVE_DIR]/YYYY-MM-DD/
5. On failure: move to [ERROR_DIR], log full traceback
6. Notify: [METHOD: Slack webhook / email / log only]

REQUIREMENTS:
- Use watchdog library with FileSystemEventHandler
- Graceful shutdown on SIGTERM/SIGINT
- PID file at /var/run/geo-watcher.pid
- Rotating file log (10 MB, 5 backups)
- Health check endpoint on port [PORT] (optional, via threading)

DEPLOYMENT:
- Generate systemd unit file (Linux) AND launchd plist (macOS)
- Include install.sh that copies files and enables the service
- Support .env file for configuration overrides

Generate: watcher.py, systemd/geo-watcher.service, launchd/
com.geo.watcher.plist, install.sh
```

#### Variables to Customize

- `WATCH_DIR`, `OUTPUT_DIR`, `ARCHIVE_DIR`, `ERROR_DIR` -- directory paths
- `GLOB` -- file pattern to watch
- `PROCESSING_STEPS` -- what to do with each file
- `METHOD` -- notification channel (Slack, email, log)
- `PORT` -- health check port

#### Expected Output

Four files: the Python watcher daemon, a systemd unit, a launchd plist, and an install script. Total ~250 lines.

#### Validation Checklist

- [ ] Daemon starts and creates PID file
- [ ] Dropping a file triggers processing within polling interval
- [ ] Corrupt files route to error directory
- [ ] Duplicate file drops do not reprocess (archive check)
- [ ] SIGTERM triggers graceful shutdown

#### Cost Optimization

Run on the cheapest always-on instance (t3.micro / e2-micro). For bursty workloads, use a serverless trigger (S3 event + Lambda / GCS trigger + Cloud Function) instead.

**Dark Arts Tip:** Use `inotifywait` in a bash loop if you need a 10-line solution and Python is overkill. Not maintainable, but sometimes that is the point.

#### Related Pages

- [CLI Tools](../tools/cli-tools.md) -- inotifywait, entr
- [Python Libraries](../tools/python-libraries.md) -- watchdog

#### Extensibility

Add a Redis queue between detection and processing for multi-worker scaling. Chain with P1 for parallel processing of large batches that arrive together.

---

### Prompt 3 -- GDAL Batch Pipeline

#### Scenario (实际场景)

You prefer the speed and reliability of native GDAL CLI tools over Python wrappers for heavy raster/vector operations. You need a reproducible pipeline expressed as a Makefile with parallel execution support.

#### Roles

Act as a GIS Systems Engineer building a Makefile-driven GDAL processing pipeline.

#### Prompt Template

```text
Write a Makefile-based GDAL batch processing pipeline.

INPUT: directory of [FILE_TYPE] files in [INPUT_DIR]
OUTPUT: processed files in [OUTPUT_DIR]

PROCESSING CHAIN (each step feeds the next):
1. [STEP_1, e.g., "gdalwarp -t_srs EPSG:3857 -r bilinear"]
2. [STEP_2, e.g., "gdal_calc.py -A input --calc='(A>0)*A'"]
3. [STEP_3, e.g., "gdal_translate -of COG -co COMPRESS=ZSTD"]
4. [STEP_4, e.g., "gdaladdo -r average 2 4 8 16"]

MAKEFILE REQUIREMENTS:
- Automatic dependency tracking: only reprocess changed inputs
- `make -j$(nproc)` for parallel execution
- Intermediate files in tmp/ (cleaned by `make clean`)
- Targets: all, clean, validate, report
- `make validate`: run ogrinfo/gdalinfo checks on outputs
- `make report`: print summary (file count, total size, CRS)
- Variables at top: INPUT_DIR, OUTPUT_DIR, CRS, COMPRESS
- Include a shell wrapper: run.sh that calls make with logging

Also generate:
- A companion script validate_outputs.sh using gdalinfo/ogrinfo
- A README section explaining how to run with GNU parallel as
  alternative: find input/ -name "*.tif" | parallel -j8 process.sh {}
```

#### Variables to Customize

- `FILE_TYPE` -- GeoTIFF, VRT, GeoPackage
- `STEP_1..N` -- GDAL command chain
- `CRS` -- target spatial reference
- `COMPRESS` -- COG compression (ZSTD, LZ4, DEFLATE)

#### Expected Output

Makefile (~80 lines), run.sh wrapper, validate_outputs.sh, and usage instructions. All GDAL operations use proper exit code checking.

#### Validation Checklist

- [ ] `make -j4` processes files in parallel without conflicts
- [ ] `make` with no changes is a no-op (up-to-date)
- [ ] `make clean` removes only intermediate files
- [ ] `make validate` exits non-zero on corrupt outputs

#### Cost Optimization

COG with ZSTD compression reduces storage costs by 40-60% over uncompressed GeoTIFF. Use `gdal_translate -co NUM_THREADS=ALL_CPUS` for compression parallelism within single files.

**Dark Arts Tip:** Chain `gdalbuildvrt` to create a virtual mosaic of your inputs first, then process the VRT as a single file. Avoids per-file overhead entirely for operations that support it.

#### Related Pages

- [CLI Tools](../tools/cli-tools.md) -- gdal_translate, gdalwarp, ogr2ogr, gdal_calc.py

#### Extensibility

Wrap the Makefile in a Snakemake or Nextflow workflow for cluster execution. Add a `make docker` target that runs everything inside a GDAL container for reproducibility.

---

## 2. ETL Pipelines

```
                          ETL Pipeline Architecture
  +--------+    +--------+    +--------+    +--------+    +--------+
  |  P4:   |    |  P5:   |    |  P6:   |    |  P7:   |    | Target |
  | Multi- |--->| Incre- |--->| Real-  |--->| Lake   |--->|PostGIS |
  | Source  |    | mental |    | Time   |    | to     |    |Feature |
  | ETL    |    | Sync   |    | Stream |    | Store  |    | Store  |
  +--------+    +--------+    +--------+    +--------+    +--------+
   Dagster/      cursor/       WebSocket/   GeoParquet    spatial
   Prefect       upsert        MQTT+buffer  S3->DuckDB    indexes
```

---

### Prompt 4 -- Spatial ETL Pipeline

#### Scenario (实际场景)

You must unify data from multiple sources -- shapefiles from an FTP, GeoJSON from a REST API, CSVs with coordinates from a partner -- into a single clean PostGIS table with consistent schema and CRS. This runs on a schedule via an orchestrator.

#### Roles

Act as a Data Engineer designing a production spatial ETL pipeline with full lineage tracking.

#### Prompt Template

```text
Design and implement a spatial ETL pipeline in Python.

EXTRACT:
- Source 1: [FORMAT] from [LOCATION], updated [FREQUENCY]
  e.g., "Shapefiles from FTP ftp://data.gov/parcels/, weekly"
- Source 2: [FORMAT] from [LOCATION], updated [FREQUENCY]
  e.g., "GeoJSON from REST API https://api.city.gov/permits, daily"
- Source 3: [FORMAT] from [LOCATION], updated [FREQUENCY]
  e.g., "CSV with lat/lon from S3 bucket, monthly"

TRANSFORM:
- Unify CRS to [TARGET_EPSG]
- Standardize columns: [MAPPING, e.g., OBJECTID->id, Shape_Area->area_m2]
- Clean geometries: fix invalid (make_valid), drop empty/null
- Deduplicate on [KEY_COLUMNS]
- Spatial join with [REFERENCE_DATASET] to add [ENRICHMENT_COLUMN]
- Validate: [RULES, e.g., "area_m2 > 0", "name IS NOT NULL"]
- Reject invalid rows to quarantine table with rejection reason

LOAD:
- Target: PostGIS [SCHEMA.TABLE]
- Mode: [upsert on KEY / truncate-reload / append]
- Create spatial index (GIST) and attribute indexes on [COLUMNS]
- Update materialized views after load

ORCHESTRATION:
- Framework: [Dagster / Prefect / plain Python]
- Schedule: [CRON_EXPRESSION]
- Retry policy: 3 attempts with exponential backoff
- Alerting: [Slack / email / PagerDuty] on failure
- Log row counts at every stage: extracted, valid, rejected, loaded

Generate: pipeline.py, models.py (Pydantic schemas), config.yaml,
and orchestrator DAG/flow definition.
```

#### Variables to Customize

- Source formats, locations, and update frequencies
- `TARGET_EPSG` -- unified coordinate reference system
- `MAPPING` -- column name standardization map
- `KEY_COLUMNS` -- deduplication and upsert keys
- Orchestration framework and schedule

#### Expected Output

Four files: pipeline module, Pydantic data models, config YAML, and orchestrator definition. Each extract/transform/load step is a separate function with logging.

#### Validation Checklist

- [ ] Row counts logged at each stage sum correctly (extracted = loaded + rejected)
- [ ] Re-running the pipeline is idempotent (upsert mode)
- [ ] Invalid geometries route to quarantine, not the main table
- [ ] Orchestrator retries on transient failures (network, DB lock)
- [ ] Spatial index exists on the target table after load

#### Cost Optimization

Use `pyogrio` engine for 3-5x faster reads than Fiona. Load via `COPY` (psycopg `copy_expert`) instead of row-by-row INSERT for 10x throughput. Schedule during off-peak hours for shared DB clusters.

**Dark Arts Tip:** For one-off loads where you do not need orchestration overhead, `ogr2ogr -f PostgreSQL` with `-lco SPATIAL_INDEX=YES` is a single command that handles CRS reprojection, schema creation, and spatial indexing.

#### Related Pages

- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- Dagster, Prefect, dbt
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS, DuckDB Spatial

#### Extensibility

Add a data lineage layer (OpenLineage) to track field-level provenance. Add Great Expectations for declarative data quality contracts between pipeline stages.

---

### Prompt 5 -- Incremental API Sync

#### Scenario (实际场景)

You maintain a PostGIS mirror of an external geospatial API. Full re-syncs are too slow and expensive; you need cursor-based incremental sync with proper upsert logic and rate limit awareness.

#### Roles

Act as a Backend Engineer implementing an incremental API-to-database sync with exactly-once semantics.

#### Prompt Template

```text
Write a Python script for incremental API-to-PostGIS sync.

API:
- Base URL: [API_URL]
- Auth: [METHOD: API key header / OAuth2 client_credentials / none]
- Pagination: [TYPE: cursor / offset-limit / link-header]
- Updated-since filter: [PARAM, e.g., "?modified_after=ISO_DATE"]
- Response format: [GeoJSON FeatureCollection / JSON array with lat/lon]
- Rate limit: [N] requests per [WINDOW]

TARGET TABLE:
- PostGIS: [SCHEMA.TABLE]
- Primary key: [PK_COLUMN]
- Geometry: geom (SRID [EPSG])
- Columns: [COLUMN_LIST with types]

SYNC LOGIC:
1. Read last_sync_cursor from metadata table (sync_state)
2. Fetch pages of records modified since cursor
3. For each record:
   - Validate geometry (is_valid, within expected bounds)
   - Upsert: INSERT ON CONFLICT (PK) DO UPDATE
4. Update sync_state with new cursor on success
5. On partial failure: commit what succeeded, record failures
6. Respect rate limits: adaptive backoff with token bucket

REQUIREMENTS:
- httpx for async HTTP with retry (tenacity)
- psycopg 3 with connection pooling
- Structured logging: records_fetched, inserted, updated, failed
- --full-sync flag to ignore cursor and reload everything
- Cron-compatible: exit 0 on success, 1 on partial, 2 on total fail
```

#### Variables to Customize

- `API_URL` -- base endpoint URL
- `METHOD` -- authentication approach
- `TYPE` -- pagination strategy
- `PK_COLUMN` -- primary key for upsert conflict resolution
- `N`, `WINDOW` -- rate limit parameters

#### Expected Output

Single Python script (~180 lines) with async HTTP client, connection pool, upsert logic, cursor management, and rate limiting.

#### Validation Checklist

- [ ] Incremental sync fetches only new/modified records
- [ ] --full-sync re-downloads everything
- [ ] Rate limits are never exceeded
- [ ] Partial failures do not corrupt sync_state cursor
- [ ] Script is cron-safe (no zombie processes, proper exit codes)

#### Cost Optimization

Cursor-based pagination avoids re-fetching unchanged records. Connection pooling reduces PostGIS connection overhead. Run during API off-peak hours if rate limits are generous at night.

**Dark Arts Tip:** If the API lacks an `updated_after` filter, hash each record and compare against stored hashes to detect changes client-side. Ugly but effective.

#### Related Pages

- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- API integration patterns
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS upsert patterns

#### Extensibility

Wrap in a Prefect flow for built-in retries, scheduling, and observability. Add CDC (Change Data Capture) on the PostGIS side to propagate changes downstream.

---

### Prompt 6 -- Real-Time Streaming ETL

#### Scenario (实际场景)

You receive high-frequency spatial data from IoT sensors, vehicle trackers, or WebSocket feeds. You need to buffer incoming events, validate and enrich them, and batch-insert into PostGIS at a sustainable rate without overwhelming the database.

#### Roles

Act as a Streaming Data Engineer building a real-time spatial data ingestion pipeline.

#### Prompt Template

```text
Write a Python real-time streaming ETL for spatial data.

INGESTION SOURCE:
- Protocol: [WebSocket / MQTT / Kafka / SSE]
- Endpoint: [URL_OR_BROKER]
- Message format: [JSON with lat/lon / GeoJSON Feature / Protobuf]
- Expected rate: [N] messages per second

PROCESSING:
1. Parse and validate each message (schema + coordinate bounds)
2. Convert to geometry (Point SRID [EPSG])
3. Enrich: reverse geocode to [ADMIN_LEVEL] using local lookup
4. Buffer in memory (max [BUFFER_SIZE] records or [FLUSH_SECONDS]s)
5. Batch INSERT into PostGIS [SCHEMA.TABLE]
6. On buffer flush: log batch size, insert duration, errors

RESILIENCE:
- Reconnect on WebSocket/MQTT disconnect (exponential backoff)
- Dead letter queue for unparseable messages (write to DLQ table)
- Backpressure: if DB insert lags, slow consumption rate
- Graceful shutdown: flush buffer on SIGTERM before exit

REQUIREMENTS:
- asyncio event loop with websockets / aiomqtt / confluent-kafka
- psycopg 3 async with COPY for batch inserts
- Prometheus metrics: messages_received, messages_processed,
  buffer_size, insert_latency_seconds, errors_total
- Health endpoint on /healthz (aiohttp or similar)

Generate: streamer.py, config.yaml, docker-compose.yml (app + PostGIS
+ optional Mosquitto/Redpanda for testing)
```

#### Variables to Customize

- Protocol and endpoint for data source
- `BUFFER_SIZE` and `FLUSH_SECONDS` -- batching parameters
- `ADMIN_LEVEL` -- enrichment granularity (city, district, country)
- `EPSG` -- coordinate reference system for geometries

#### Expected Output

Three files: the async streaming processor, configuration YAML, and a docker-compose for local testing. Total ~300 lines.

#### Validation Checklist

- [ ] Messages arrive and batch-insert at expected throughput
- [ ] Dead letter queue captures malformed messages
- [ ] Reconnection works after broker/WebSocket restart
- [ ] Graceful shutdown flushes remaining buffer
- [ ] Prometheus metrics are exposed and accurate

#### Cost Optimization

Batch INSERTs via COPY are 50-100x cheaper in DB resources than row-by-row. Tune buffer size to balance latency vs. throughput. Use TimescaleDB hypertables if time-series queries dominate.

**Dark Arts Tip:** For extreme throughput (>10k msg/s), skip PostGIS entirely and write to GeoParquet files on object storage. Query later with DuckDB Spatial. The database is often the bottleneck.

#### Related Pages

- [Realtime & Offline Advanced](../js-bindbox/realtime-offline-advanced.md) -- WebSocket patterns
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS, TimescaleDB

#### Extensibility

Add Apache Kafka as a durable buffer between ingestion and processing for true exactly-once semantics. Add a materialized view in PostGIS for real-time aggregations.

---

### Prompt 7 -- Data Lake to Feature Store

#### Scenario (实际场景)

Your organization stores raw spatial data as GeoParquet files in S3/GCS. You need a pipeline that reads from the data lake, transforms with DuckDB Spatial (for speed), and loads curated features into PostGIS for application consumption.

#### Roles

Act as a Data Platform Engineer building a lakehouse-to-feature-store pipeline for geospatial data.

#### Prompt Template

```text
Write a Python pipeline: GeoParquet data lake -> DuckDB -> PostGIS.

DATA LAKE:
- Location: [S3_BUCKET / GCS_BUCKET / local path]
- Format: GeoParquet, partitioned by [PARTITION_KEY, e.g., year/month]
- Schema: [COLUMNS with types, including geometry WKB column]
- Size: ~[SIZE, e.g., "50 GB across 500 files"]

TRANSFORM (DuckDB Spatial):
- Install and load spatial extension
- Read GeoParquet with partition pruning: WHERE [FILTER]
- Transform: [OPERATIONS, e.g., ST_Transform, ST_Buffer, spatial join]
- Aggregate: [GROUP_BY columns, aggregation functions]
- Output: cleaned feature table with [OUTPUT_COLUMNS]

LOAD (PostGIS):
- Attach PostgreSQL via DuckDB postgres extension
  OR export to temp GeoParquet then load via ogr2ogr
- Target: [SCHEMA.TABLE]
- Mode: [truncate-reload / upsert on KEY]
- Post-load: VACUUM ANALYZE, refresh materialized views
- Verify: row count, spatial extent, null checks

REQUIREMENTS:
- DuckDB Python API (import duckdb)
- S3 credentials via environment variables or IAM role
- Log: read time, transform time, load time, row counts
- CLI: --partition-filter, --target-table, --dry-run
```

#### Variables to Customize

- `S3_BUCKET` -- data lake location and access credentials
- `PARTITION_KEY` -- how data is organized (date, region, etc.)
- `OPERATIONS` -- DuckDB Spatial SQL transformations
- `OUTPUT_COLUMNS` -- final feature schema
- Load mode -- truncate-reload vs. upsert

#### Expected Output

Single Python script with DuckDB SQL templates, S3 integration, PostGIS loading, and CLI interface. ~150 lines.

#### Validation Checklist

- [ ] Partition pruning reduces data scanned (check DuckDB EXPLAIN)
- [ ] Output row count matches expected after filters
- [ ] PostGIS spatial extent matches source data extent
- [ ] VACUUM ANALYZE runs after load

#### Cost Optimization

DuckDB processes GeoParquet locally without a cluster -- zero compute cost beyond the machine running it. Partition pruning avoids reading unnecessary files from S3 (saves egress). Use `s3_region` setting to co-locate compute.

**Dark Arts Tip:** DuckDB can query GeoParquet files directly on S3 via HTTP range requests without downloading them. For exploratory queries, skip the download entirely: `SELECT * FROM read_parquet('s3://bucket/data.parquet') LIMIT 100`.

#### Related Pages

- [Spatial Databases](../tools/spatial-databases.md) -- DuckDB Spatial, PostGIS
- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- GeoParquet, data lake patterns

#### Extensibility

Schedule with Dagster assets for dependency-aware orchestration. Add a reverse path (PostGIS -> GeoParquet) for data lake backfills. Layer in Delta Lake or Iceberg for ACID transactions on the lake.

---

## 3. Report Generation

```
                       Report Generation Pipeline
  +----------+      +----------+      +----------+
  | P8: Map  |      | P9: Dash |      | P10:     |
  | Report   |      | Data     |      | Client   |
  | (PDF/    |      | Prep     |      | Delivery |
  |  HTML)   |      | (cache)  |      | (CN/EN)  |
  +----------+      +----------+      +----------+
       |                 |                  |
       v                 v                  v
  WeasyPrint/       Streamlit/Dash     Jinja2+WeasyPrint
  ReportLab         GeoJSON+cache      bilingual PDF
```

---

### Prompt 8 -- Automated Map Report

#### Scenario (实际场景)

You produce recurring reports (monthly environmental monitoring, quarterly land-use change) that include maps, charts, tables, and narrative text. Manual assembly in Word takes hours; you want full automation from data to PDF.

#### Roles

Act as a GIS Analyst automating report generation with programmatic maps and charts.

#### Prompt Template

```text
Write a Python script that generates an automated [REPORT_TYPE]
report as [FORMAT: PDF / HTML / both].

INPUT DATA:
- Raster: [RASTER_DESC, e.g., "monthly NDVI composite GeoTIFF"]
- Vector: [VECTOR_DESC, e.g., "admin boundaries GeoPackage"]
- Statistics: [STATS_DESC, e.g., "zonal stats CSV per district"]
- Previous period data for comparison: [PREVIOUS_DATA]

REPORT SECTIONS:
1. Title page: title, date range, author, organization logo
2. Executive summary (auto-generated from data thresholds)
3. Study area overview map (contextmaps with basemap)
4. Thematic map: [MAP_DESC, e.g., "NDVI choropleth by district"]
5. Change detection map: current vs. previous period
6. Statistics table: [COLUMNS, e.g., district, mean, delta, trend]
7. Time series chart: [CHART_DESC, e.g., "12-month trend per region"]
8. Auto-narrative using conditional logic:
   "District X showed a [increase/decrease] of [N]% compared to
   [previous period], [exceeding/within] the [threshold] threshold."
9. Methodology and data sources appendix

REQUIREMENTS:
- [LIBRARY: WeasyPrint + Jinja2 / ReportLab / python-docx]
- matplotlib or plotly for charts; contextily for basemaps
- Parameterized: report_date, input_paths, output_path, locale
- Generate both individual district reports and a summary report
- Embed maps as high-DPI PNG (300 dpi) in the document
```

#### Variables to Customize

- `REPORT_TYPE` -- vegetation health, air quality, flood risk, etc.
- `FORMAT` -- PDF (WeasyPrint), HTML, DOCX
- `MAP_DESC` and `CHART_DESC` -- specific visualizations
- `LIBRARY` -- report generation library

#### Expected Output

Python script with Jinja2 HTML template, map/chart generation functions, and auto-narrative logic. Produces a print-ready PDF.

#### Validation Checklist

- [ ] PDF renders correctly with embedded maps at 300 DPI
- [ ] Auto-narrative text matches actual data values
- [ ] Change detection correctly identifies increase/decrease
- [ ] Report runs end-to-end with sample data without errors

#### Cost Optimization

Pre-render basemap tiles and cache them locally. Use matplotlib Agg backend (non-interactive) for headless server generation. Batch-generate district reports in parallel with concurrent.futures.

**Dark Arts Tip:** For quick-and-dirty reports, use `fpdf2` -- it is pure Python, no system dependencies (unlike WeasyPrint which needs Cairo), and generates PDFs in milliseconds. Trade-off: limited CSS support.

#### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- matplotlib, contextily, folium

#### Extensibility

Add email delivery via SMTP or S3 upload for automated distribution. Use LaTeX via Jinja2 templates for publication-quality typesetting.

---

### Prompt 9 -- Dashboard Data Preparation

#### Scenario (实际场景)

Your team maintains a Streamlit or Dash dashboard that visualizes spatial data. The dashboard is slow because it processes raw data on every page load. You need a data preparation layer that pre-computes aggregates, generates optimized GeoJSON, and caches results.

#### Roles

Act as a Full-Stack GIS Developer building a data preparation pipeline for a spatial dashboard.

#### Prompt Template

```text
Write a data preparation pipeline and dashboard layout for
[DASHBOARD_FRAMEWORK: Streamlit / Dash / Panel].

TOPIC: [TOPIC, e.g., "real-time air quality monitoring stations"]

DATA PREP (runs on schedule, not on page load):
1. Fetch latest data from [SOURCE]
2. Clean: remove outliers ([METHOD]), fill gaps ([FILL_METHOD])
3. Derive metrics: [METRICS, e.g., "AQI from raw PM2.5/O3/NO2"]
4. Aggregate to [SPATIAL_LEVEL] and [TEMPORAL_LEVEL]
5. Generate: map_data.geojson (features + display attributes)
6. Generate: timeseries.json (per-station time arrays)
7. Generate: summary.json (current stats, alerts)
8. Cache to [CACHE: Redis / filesystem / SQLite] with TTL [SECONDS]

DASHBOARD LAYOUT:
- Interactive map (folium / pydeck / Mapbox) with color-coded markers
- Sidebar: date range picker, metric selector, region filter
- Time series chart below map (plotly)
- Summary cards: current max, min, mean, stations exceeding threshold
- Auto-refresh every [INTERVAL] seconds
- Mobile-responsive layout

REQUIREMENTS:
- Separate data_prep.py (schedulable) from app.py (dashboard)
- data_prep.py: CLI with --force-refresh flag
- app.py: reads only from cache, never from raw source
- GeoJSON simplified to <500KB for fast map rendering
```

#### Variables to Customize

- `DASHBOARD_FRAMEWORK` -- Streamlit, Dash, Panel
- `SOURCE` -- API endpoint, database, or file
- `SPATIAL_LEVEL` and `TEMPORAL_LEVEL` -- aggregation granularity
- `CACHE` -- caching backend and TTL
- `INTERVAL` -- dashboard auto-refresh period

#### Expected Output

Two Python files: data_prep.py (scheduled ETL to cache) and app.py (dashboard reading from cache). Plus a requirements.txt.

#### Validation Checklist

- [ ] Dashboard loads in under 2 seconds from cache
- [ ] data_prep.py runs independently on a schedule
- [ ] GeoJSON file size is under 500KB after simplification
- [ ] Filters update map and chart reactively
- [ ] Auto-refresh works without full page reload

#### Cost Optimization

Pre-compute aggregates to avoid expensive on-the-fly spatial operations. Simplify geometries (Douglas-Peucker) to reduce GeoJSON payload. Use CDN for static assets if hosting publicly.

**Dark Arts Tip:** For Streamlit, use `st.cache_data` with TTL matching your data_prep schedule. But if your data exceeds 100MB, switch to a file-based cache -- Streamlit's in-memory cache will eat your RAM.

#### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- folium, plotly, streamlit

#### Extensibility

Add WebSocket push for true real-time updates (bypass polling). Deploy behind nginx with caching headers for multi-user scalability.

---

### Prompt 10 -- Client Delivery Report (甲方交付报告)

#### Scenario (实际场景)

You are delivering spatial analysis results to a Chinese government client (甲方) who requires formal bilingual reports with executive summaries, methodology sections, quality assessments, and compliance statements. This is standard in Chinese GIS consulting.

#### Roles

Act as a Senior GIS Consultant preparing a formal bilingual (Chinese/English) project delivery report.

#### Prompt Template

```text
Write a Python script that generates a bilingual (Chinese/English)
client delivery report for a GIS project.

PROJECT:
- Name: [PROJECT_NAME] / [PROJECT_NAME_CN]
- Client: [CLIENT_NAME]
- Deliverables: [LIST, e.g., "land use classification map, change
  detection analysis, accuracy assessment"]

REPORT STRUCTURE (双语结构):
1. Cover page / 封面: project name, client, contractor, date, version
2. Table of contents / 目录 (auto-generated)
3. Executive summary / 执行摘要: key findings in 1 page
4. Project overview / 项目概述: objectives, scope, study area map
5. Data sources / 数据来源: table of all input datasets with metadata
6. Methodology / 技术方法: processing workflow diagram, parameters
7. Results / 成果:
   - Thematic maps (embedded at 300 DPI)
   - Statistics tables
   - Accuracy assessment (confusion matrix, kappa, OA, PA, UA)
8. Quality assessment / 质量评估:
   - Positional accuracy (RMSE)
   - Attribute accuracy
   - Completeness
   - Logical consistency
9. Conclusions / 结论
10. Appendices / 附录: raw data catalog, processing logs

REQUIREMENTS:
- WeasyPrint + Jinja2 with bilingual HTML template
- Support both zh-CN and en-US locales
- Page numbers, headers, footers with project name
- Professional formatting: GB/T standards for Chinese documents
- Embed maps generated by matplotlib with Chinese font support
  (SimHei / Noto Sans CJK)
- CLI: --project-config config.yaml --locale [zh/en/both]
  --output report.pdf
```

#### Variables to Customize

- `PROJECT_NAME` / `PROJECT_NAME_CN` -- bilingual project name
- Report sections -- add/remove based on contract requirements
- `GB/T` standard -- specific Chinese national standards to reference
- Font -- SimHei, Noto Sans CJK, or other CJK-compatible font

#### Expected Output

Jinja2 HTML template with bilingual sections, Python generation script, and a sample config.yaml. Produces a professional PDF suitable for government delivery.

#### Validation Checklist

- [ ] Chinese characters render correctly in PDF (font embedding)
- [ ] Both --locale zh and --locale en produce valid reports
- [ ] Accuracy metrics (kappa, OA) are calculated correctly
- [ ] Table of contents page numbers match actual pages
- [ ] Maps are legible at printed A3/A4 size

#### Cost Optimization

Template once, generate many. Store the Jinja2 template as a reusable asset across projects. Parameterize everything via config.yaml to avoid code changes per project.

**Dark Arts Tip:** Most Chinese government clients want the report as a Word document, not PDF. Use `python-docx` with a pre-styled .docx template (from your company's standard template) and inject content programmatically. Looks hand-crafted; generated in seconds.

#### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- WeasyPrint, Jinja2, python-docx

#### Extensibility

Add a review/approval workflow: generate draft, send for review via email, incorporate comments, regenerate final. Version control the config.yaml alongside the project repo.

---

## 4. Quality Assurance

```
                       QA Pipeline Architecture
  +----------+      +----------+      +----------+
  | P11:     |      | P12:     |      | P13:     |
  | Spatial  |----->| Regress. |----->| Data     |
  | Validator|      | Testing  |      | Diff &   |
  | (schema+ |      | (pytest) |      | Change   |
  |  geom)   |      |          |      | Tracking |
  +----------+      +----------+      +----------+
       |                 |                  |
       v                 v                  v
  PASS/FAIL +       JUnit XML for     Change report +
  flagged layer     CI integration     versioned diffs
```

---

### Prompt 11 -- Spatial Data Validator

#### Scenario (实际场景)

You receive data deliveries from external vendors or field teams. Before ingestion, every delivery must pass a comprehensive validation gate covering schema, geometry, attributes, topology, and CRS. Failures must block the pipeline and produce actionable reports.

#### Roles

Act as a QA Engineer building a spatial data validation framework with CI/CD-compatible exit codes.

#### Prompt Template

```text
Write a Python spatial data validation framework.

INPUT: [FORMAT, e.g., GeoPackage] at [INPUT_PATH]

SCHEMA VALIDATION:
- Required layers: [LAYER_LIST]
- Required columns per layer: [COLUMNS with expected dtypes]
- Geometry type per layer: [Point / LineString / Polygon / Multi*]

GEOMETRY VALIDATION:
- No null or empty geometries
- No invalid geometries (is_valid check, report reason)
- No duplicate geometries (WKB hash comparison)
- All features within [BOUNDS: bbox or reference boundary file]
- Minimum feature size: [THRESHOLD, e.g., polygons > 1 m2]

ATTRIBUTE VALIDATION:
- Column [A]: NOT NULL, UNIQUE
- Column [B]: value IN [ALLOWED_VALUES]
- Column [C]: numeric, range [MIN] to [MAX]
- Column [D]: valid ISO 8601 date
- Custom rule: [EXPRESSION, e.g., "area_m2 == geom.area"]

TOPOLOGY VALIDATION:
- No overlapping polygons within same layer
- No gaps between adjacent polygons (optional, tolerance [T])

CRS VALIDATION:
- Must be EPSG:[CODE]. Report actual CRS if different.
- Do NOT auto-reproject (report only).

OUTPUT:
- JSON report: {overall: PASS/FAIL, checks: [{name, status,
  count_issues, sample_feature_ids}]}
- Export flagged features to [OUTPUT_PATH] as separate layer
- Exit code: 0=PASS, 1=FAIL, 2=ERROR (file unreadable)
- Optional: Markdown summary for PR comment integration
```

#### Variables to Customize

- `FORMAT` -- GeoPackage, GeoJSON, GeoParquet, Shapefile
- `LAYER_LIST` -- expected layer names
- Column rules -- NOT NULL, UNIQUE, range, enum, regex
- `BOUNDS` -- bounding box or boundary file for extent check
- Topology rules -- overlaps, gaps, tolerance

#### Expected Output

Python module with a ValidationRunner class, rule definitions in YAML, JSON report output, and CLI entry point. ~250 lines.

#### Validation Checklist

- [ ] Exit code 0 for valid data, 1 for invalid, 2 for unreadable
- [ ] Flagged features layer contains exactly the invalid features
- [ ] JSON report includes sample feature IDs for each failure
- [ ] Markdown output is valid and renders in GitHub PR comments
- [ ] Validator runs in under 30 seconds for 100K-feature datasets

#### Cost Optimization

Run validation before any expensive processing to fail fast. Cache rule definitions in YAML for reuse across projects. Use spatial indexing (STRtree) for topology checks on large datasets.

**Dark Arts Tip:** For quick schema validation without writing code, use `ogrinfo -json` piped into `jq` to check layer names, column types, and geometry types in a single shell command. Wrap it in a GitHub Actions step for instant CI.

#### Related Pages

- [CLI Tools](../tools/cli-tools.md) -- ogrinfo for quick schema inspection
- [Python Libraries](../tools/python-libraries.md) -- shapely validation functions

#### Extensibility

Define rules as a YAML contract shared between data producers and consumers. Add a web UI (FastAPI + htmx) for non-technical users to upload and validate files interactively.

---

### Prompt 12 -- Regression Testing for GIS Pipelines

#### Scenario (实际场景)

You maintain a GIS processing pipeline and need to ensure that code changes do not silently alter outputs. You want a pytest suite that compares pipeline outputs against known-good reference data, with tolerance for acceptable floating-point differences.

#### Roles

Act as a QA Automation Engineer building regression tests for a spatial data pipeline.

#### Prompt Template

```text
Write a pytest test suite for regression testing a GIS pipeline.

PIPELINE: [DESCRIPTION, e.g., "clips parcels to city boundary,
reprojects to EPSG:3857, calculates area, classifies by zone"]

INPUT FIXTURE: [FIXTURE_DESC, e.g., "tests/fixtures/parcels.geojson
(50 features) + tests/fixtures/city_boundary.geojson"]

REFERENCE OUTPUT: [REF_DESC, e.g., "tests/fixtures/expected_output.gpkg
(known-good result from validated pipeline run)"]

TESTS:
1. Output file exists and is readable (pyogrio.read_dataframe)
2. CRS matches EPSG:[CODE] (compare .crs.to_epsg())
3. Feature count within tolerance: [N] +/- [M]
4. Bounding box within tolerance: [T] meters/degrees
5. Schema: exact column names and dtypes match reference
6. Geometry types all [EXPECTED_TYPE] (no mixed types)
7. No null geometries; no nulls in [REQUIRED_COLUMNS]
8. Statistical comparison: mean/std of [NUMERIC_COL] within [P]%
   of reference values
9. Geometry hash: sorted WKB hashes match (detect geometry changes)
10. Performance: pipeline completes in under [TIMEOUT] seconds

FRAMEWORK:
- pytest with conftest.py for shared fixtures and tmp_path
- Parameterize tests across [SCENARIOS, e.g., multiple input files]
- Generate JUnit XML: pytest --junitxml=report.xml
- Coverage: pytest-cov with [THRESHOLD]% minimum

Generate: conftest.py, test_pipeline.py, pytest.ini, and a Makefile
target `make test`.
```

#### Variables to Customize

- `DESCRIPTION` -- what the pipeline does
- `FIXTURE_DESC` -- test input data description
- `N`, `M` -- expected feature count and tolerance
- `P` -- percentage tolerance for statistical comparison
- `TIMEOUT` -- performance threshold in seconds

#### Expected Output

Three files: conftest.py (fixtures), test_pipeline.py (test cases), and pytest.ini (configuration). All tests are parameterizable.

#### Validation Checklist

- [ ] Tests pass against known-good reference data
- [ ] Intentionally broken pipeline causes test failures
- [ ] JUnit XML report generates correctly for CI
- [ ] Performance test enforces timeout
- [ ] Coverage meets threshold

#### Cost Optimization

Use small fixture datasets (<100 features) to keep test execution under 10 seconds. Store fixtures as GeoJSON (human-readable diffs in version control). Run expensive integration tests only on main branch, not on every PR.

**Dark Arts Tip:** Generate your reference data by running the pipeline once, manually verifying the output, then freezing it as a fixture. `pytest --snapshot-update` patterns (via pytest-snapshot) can automate this freeze/compare cycle.

#### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- pytest, geopandas testing utilities
- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- pipeline testing patterns

#### Extensibility

Add visual regression tests: render output as a PNG map, compare against reference image using perceptual hashing (imagehash library). Catches styling and symbology regressions.

---

### Prompt 13 -- Data Diff and Change Tracking

#### Scenario (实际场景)

You need to compare two versions of a spatial dataset -- perhaps before and after an update, or between a vendor delivery and your current database -- and produce a detailed change report showing additions, deletions, modifications, and geometry changes.

#### Roles

Act as a Data Quality Analyst building a spatial data differencing tool.

#### Prompt Template

```text
Write a Python tool that compares two versions of a spatial dataset
and produces a change report.

INPUTS:
- Version A (previous): [PATH_A, format]
- Version B (current): [PATH_B, format]
- Match key: [KEY_COLUMN] (unique identifier for features)

DIFF LOGIC:
1. Load both datasets with pyogrio
2. Identify:
   - ADDED: keys in B not in A
   - DELETED: keys in A not in B
   - MODIFIED: keys in both, but attributes differ
   - GEOMETRY_CHANGED: keys in both, but geometry WKB differs
   - UNCHANGED: keys in both, identical attributes and geometry
3. For MODIFIED features: list which columns changed, old/new values
4. For GEOMETRY_CHANGED: calculate Hausdorff distance between
   old and new geometry
5. Summary statistics:
   - Total features: A count, B count
   - Added/deleted/modified/geometry_changed/unchanged counts
   - Area of added/deleted polygons (if polygon layer)

OUTPUT:
- JSON change report with full details
- GeoPackage with layers: added, deleted, modified, geom_changed
  (each with change metadata columns)
- Markdown summary table for human review
- Optional: HTML map showing changes (added=green, deleted=red,
  modified=yellow) using folium

CLI: diff_spatial.py --previous A.gpkg --current B.gpkg --key id
     --output changes/ --format [json/gpkg/md/html/all]
```

#### Variables to Customize

- `PATH_A`, `PATH_B` -- the two dataset versions
- `KEY_COLUMN` -- unique identifier for matching features
- Output formats -- JSON, GeoPackage, Markdown, HTML map
- Hausdorff distance threshold for "significant" geometry change

#### Expected Output

Single Python script with diffing logic, multiple output format generators, and CLI interface. ~200 lines.

#### Validation Checklist

- [ ] Added + Deleted + Modified + Unchanged = max(count_A, count_B) logic is sound
- [ ] Geometry changes detected even when attributes are identical
- [ ] Hausdorff distance is calculated correctly
- [ ] Output GeoPackage layers contain correct features
- [ ] HTML map renders with correct color coding

#### Cost Optimization

For large datasets (>1M features), hash attributes and geometry (WKB hash) first and only do detailed comparison on hash mismatches. Use DuckDB for the join/diff if datasets exceed memory.

**Dark Arts Tip:** If your data is in PostGIS or GeoPackage, use [Kart](https://kartproject.org/) -- it is Git for geospatial data. It handles versioning, diffing, merging, and conflict resolution natively. Far superior to building your own diff tool.

#### Related Pages

- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- Kart version control
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS table comparison

#### Extensibility

Integrate with Kart for version-controlled spatial datasets. Add a scheduled job that diffs the latest delivery against the previous and emails a change summary to stakeholders.

---

## 5. CI/CD for GIS

```
                        CI/CD Pipeline for GIS
  +----------+      +----------+      +----------+
  | P14: GH  |      | P15: Data|      | P16:     |
  | Actions  |----->| Valid.   |----->| Automated|
  | (lint+   |      | in CI    |      | Deploy   |
  |  test)   |      | (PR gate)|      | (Docker+ |
  +----------+      +----------+      |  cloud)  |
       |                 |            +----------+
       v                 v                  |
  ruff+mypy+        PR comment with         v
  pytest+Docker     PASS/FAIL table    ECS/Cloud Run
```

---

### Prompt 14 -- GitHub Actions for Geospatial

#### Scenario (实际场景)

You have a geospatial Python project on GitHub. You need a CI workflow that handles the notoriously tricky GDAL/GEOS/PROJ system dependencies, runs linting, type checking, tests with coverage, and optionally builds and pushes a Docker image.

#### Roles

Act as a DevOps Engineer setting up CI/CD for a geospatial Python project with system-level spatial dependencies.

#### Prompt Template

```text
Write a GitHub Actions CI workflow for a geospatial Python project.

PROJECT:
- Python version: [PY_VERSION, e.g., 3.12]
- Dependency management: [pyproject.toml / requirements.txt / conda]
- Spatial system deps: GDAL, GEOS, PROJ (need apt-get)
- Test framework: pytest + pytest-cov
- Linter: ruff (lint + format check)
- Type checker: mypy with spatial stubs

WORKFLOW (.github/workflows/ci.yml):
1. Trigger: push to main, pull_request to main
2. Matrix: [OS_LIST, e.g., ubuntu-latest, optionally macos-latest]
3. Steps:
   a. Checkout code
   b. Install system spatial libs:
      apt-get install -y gdal-bin libgdal-dev libgeos-dev libproj-dev
   c. Set GDAL_VERSION env var from gdal-config --version
   d. Setup Python [PY_VERSION]
   e. Cache pip dependencies (hash of pyproject.toml)
   f. Install: pip install -e ".[dev]"
   g. Lint: ruff check src/ tests/
   h. Format check: ruff format --check src/ tests/
   i. Type check: mypy src/ --ignore-missing-imports
   j. Test: pytest --cov=src --cov-report=xml --junitxml=report.xml
   k. Upload coverage to [Codecov / as artifact]
   l. Fail if coverage < [THRESHOLD]%
4. On push to main (additional):
   m. Build Docker image
   n. Push to [REGISTRY, e.g., ghcr.io/ORG/REPO]
   o. Tag with git SHA and "latest"

ALSO GENERATE:
- .pre-commit-config.yaml (ruff, mypy, trailing-whitespace, eof)
- Makefile targets: lint, format, typecheck, test, ci (all of above)
```

#### Variables to Customize

- `PY_VERSION` -- Python version (3.11, 3.12)
- `OS_LIST` -- CI runner operating systems
- `THRESHOLD` -- minimum coverage percentage
- `REGISTRY` -- Docker image registry (ghcr.io, Docker Hub, ECR)

#### Expected Output

Three files: ci.yml workflow, .pre-commit-config.yaml, and Makefile. The workflow handles GDAL installation, caching, full lint/test pipeline, and conditional Docker build.

#### Validation Checklist

- [ ] GDAL installs correctly and gdal-config --version succeeds
- [ ] pip install with GDAL bindings does not fail on version mismatch
- [ ] Coverage report uploads and threshold gate works
- [ ] Docker image builds and pushes only on main branch
- [ ] Pre-commit hooks match CI checks (no local/CI drift)

#### Cost Optimization

Cache pip dependencies aggressively (saves 30-60s per run). Use `ubuntu-latest` only unless macOS is required (macOS runners cost 10x). Pin GDAL version to avoid surprise breakage from apt updates.

**Dark Arts Tip:** Use `ghcr.io/osgeo/gdal:ubuntu-small-latest` as your CI container image instead of installing GDAL from apt. It has everything pre-installed and shaves 2-3 minutes off every CI run. Set `container: ghcr.io/osgeo/gdal:ubuntu-small-3.9.0` in your job.

#### Related Pages

- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- CI/CD patterns
- [CLI Tools](../tools/cli-tools.md) -- GDAL system dependencies

#### Extensibility

Add a `release.yml` workflow that auto-publishes to PyPI on tagged releases. Add a `docs.yml` that builds and deploys MkDocs to GitHub Pages.

---

### Prompt 15 -- Spatial Data Validation in CI

#### Scenario (实际场景)

Your repository contains spatial data files (GeoPackage, GeoJSON, GeoParquet) alongside code. You want every pull request that modifies data files to trigger automated validation, post results as a PR comment, and block merge on failure.

#### Roles

Act as a CI/CD Engineer building automated spatial data quality gates for pull requests.

#### Prompt Template

```text
Write a GitHub Actions workflow that validates spatial data on PRs.

REPOSITORY:
- Data directory: [DATA_DIR, e.g., "data/"]
- Formats: [FORMATS, e.g., GeoPackage, GeoJSON, GeoParquet]
- Validation rules: defined in [RULES_FILE, e.g., validation/rules.yaml]
- Python version: [PY_VERSION]

WORKFLOW (.github/workflows/validate-data.yml):
1. Trigger: pull_request when paths match [DATA_DIR]/** or validation/**
2. Setup: Ubuntu, Python, apt-get install gdal-bin libgdal-dev
3. Detect changed data files:
   git diff --name-only --diff-filter=AM ${{ github.event.pull_request.base.sha }} HEAD
   | grep -E '\.(gpkg|geojson|parquet)$'
4. For each changed file, run validation:
   - Schema: required columns + types
   - CRS: matches expected EPSG
   - Geometry: valid, non-null, non-empty, within bounds
   - Attributes: rules from YAML (not-null, ranges, enums)
   - File size: warn if > [SIZE_LIMIT]
5. Generate Markdown report table:
   | File | Check | Status | Details |
6. Post report as PR comment (peter-evans/create-or-update-comment)
   Update existing comment on re-push (use comment identifier)
7. Set workflow status: fail if any check fails (blocks merge)

ALSO GENERATE:
- validation/rules.yaml with example rules per file
- validation/validate.py reading rules.yaml and checking files
- README badge: ![Data Validation](url/badge.svg)
```

#### Variables to Customize

- `DATA_DIR` -- path to spatial data in the repository
- `FORMATS` -- file extensions to validate
- `RULES_FILE` -- YAML file defining validation rules
- `SIZE_LIMIT` -- file size warning threshold

#### Expected Output

Three files: GitHub Actions workflow YAML, validation Python script, and rules YAML. The workflow detects changed files, validates them, and posts results as a PR comment.

#### Validation Checklist

- [ ] Workflow triggers only when data files change
- [ ] PR comment updates on re-push (no duplicate comments)
- [ ] Failed validation blocks merge via required status check
- [ ] Unchanged data files are not re-validated
- [ ] Badge URL is correct and shows current status

#### Cost Optimization

Only validate changed files (not the entire data directory) to minimize CI runtime. Cache GDAL installation using actions/cache. Keep data files small in the repo; use Git LFS for anything over 50MB.

**Dark Arts Tip:** Use `tj-actions/changed-files` action instead of raw `git diff` -- it handles edge cases with merge commits, force pushes, and first-time PRs that `git diff` gets wrong. Saves hours of debugging.

#### Related Pages

- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- data quality patterns
- [CLI Tools](../tools/cli-tools.md) -- ogrinfo for quick validation

#### Extensibility

Add GeoParquet schema validation using Apache Arrow metadata. Add visual diff: render before/after maps and attach as PR comment images.

---

### Prompt 16 -- Automated Deployment

#### Scenario (实际场景)

You need to containerize a GIS application with its GDAL/GEOS/PROJ dependencies, set up local development with docker-compose (app + PostGIS), and deploy to cloud infrastructure (ECS, Cloud Run, or similar) with monitoring and scheduled execution.

#### Roles

Act as a Cloud Infrastructure Engineer deploying a containerized GIS application to production.

#### Prompt Template

```text
Write deployment configuration for a GIS application.

APPLICATION: [APP_DESC, e.g., "spatial ETL pipeline that processes
satellite imagery and loads results into PostGIS"]
Schedule: [CRON, e.g., "0 3 * * 1" -- every Monday 3am UTC]
Cloud provider: [PROVIDER: AWS / GCP / Azure]

1. DOCKERFILE (multi-stage):
   - Builder: FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.0
     Install Python, pip install with --no-cache-dir
   - Runtime: FROM python:3.12-slim
     COPY --from=builder GDAL libs + Python packages
     Non-root user, HEALTHCHECK, LABEL metadata
   - Target image size: < 500 MB

2. DOCKER-COMPOSE (local dev):
   - app: builds from Dockerfile, mounts src/ as volume
   - postgis: postgis/postgis:16-3.4, persistent volume
   - pgadmin: optional, for database inspection
   - Network: shared bridge network
   - .env file for DB credentials (with .env.example committed)

3. CLOUD DEPLOYMENT ([PROVIDER]):
   AWS:  ECS Fargate task definition + EventBridge Scheduler rule
   GCP:  Cloud Run Job + Cloud Scheduler
   Azure: Container Apps Job + Timer trigger
   - Environment variables from [Secrets Manager / Secret Manager]
   - VPC/VPN access to database
   - Log forwarding to [CloudWatch / Cloud Logging / Monitor]
   - Alert on: task failure, runtime > [TIMEOUT], OOM kill

4. MAKEFILE:
   build, run-local, test, push, deploy-dev, deploy-prod, logs, ssh

5. MONITORING:
   - Structured JSON logs with correlation ID
   - Metrics: runtime_seconds, records_processed, errors_count
   - Dashboard: [Grafana / CloudWatch Dashboard / Cloud Monitoring]
   - PagerDuty / Slack alert on failure
```

#### Variables to Customize

- `APP_DESC` -- what the application does
- `PROVIDER` -- AWS, GCP, or Azure
- `CRON` -- execution schedule
- `TIMEOUT` -- maximum allowed runtime before alert
- Database connection details

#### Expected Output

Five files: Dockerfile, docker-compose.yml, cloud deployment config (task def or job YAML), Makefile, and monitoring/alerting config. Total ~350 lines.

#### Validation Checklist

- [ ] Docker image builds and runs locally
- [ ] docker-compose up starts app + PostGIS successfully
- [ ] Cloud deployment config passes validation (aws ecs validate, etc.)
- [ ] Secrets are not hardcoded (env vars / secrets manager only)
- [ ] Alert fires on simulated failure

#### Cost Optimization

Multi-stage Docker build reduces image size (smaller = faster pull = cheaper). Use Fargate Spot / Cloud Run min-instances=0 for scheduled jobs. Set memory/CPU limits tight to avoid over-provisioning. Estimated monthly cost for weekly job: $2-10.

**Dark Arts Tip:** For scheduled jobs that run less than 15 minutes, AWS Lambda with a container image (up to 10GB) is cheaper than ECS Fargate. Package your GDAL container as a Lambda, trigger from EventBridge. Sounds wrong, works great.

#### Related Pages

- [Server & Publishing](../tools/server-publishing.md) -- tile servers, deployment patterns
- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- pipeline deployment

#### Extensibility

Add blue/green deployment for zero-downtime updates. Add Terraform (see P24) to manage the cloud infrastructure as code. Add a rollback Makefile target.

---

## 6. Claude Code Agent Workflows

```
                    Agent Workflow Architecture
  +----------+      +----------+      +----------+      +----------+
  | P17:     |      | P18:     |      | P19:     |      | P20:     |
  | CLAUDE.md|----->| Multi-   |----->| MCP      |----->| Cursor/  |
  | Config   |      | Agent    |      | Server   |      | Windsurf |
  | (rules)  |      | Pipeline |      | Workflow |      | Multi-   |
  +----------+      +----------+      +----------+      | File     |
       |                 |                  |            +----------+
       v                 v                  v                  |
  System prompt     5-agent chain     PostGIS/STAC/GDAL       v
  for GIS projects  with handoffs     server integration  Composer/Cascade
```

---

### Prompt 17 -- CLAUDE.md GIS Project Config

#### Scenario (实际场景)

You are using Claude Code (Anthropic's CLI agent) on a GIS project and want to set up a comprehensive CLAUDE.md system prompt that encodes your team's spatial coding standards, preferred libraries, and common pitfalls to avoid. This ensures consistent, high-quality code generation across all interactions.

#### Roles

Act as a Tech Lead writing a CLAUDE.md configuration file for a GIS Python project.

#### Prompt Template

```text
Write a CLAUDE.md file for a GIS Python project that will guide
Claude Code (or any AI coding agent) to produce high-quality
spatial code.

PROJECT: [PROJECT_NAME]
TYPE: [TYPE, e.g., "spatial analysis library", "ETL pipeline",
      "web mapping backend", "QGIS plugin"]
PYTHON: [VERSION]
KEY DEPS: [DEPS, e.g., geopandas, rasterio, shapely, pyproj, FastAPI]

CLAUDE.md SHOULD INCLUDE:

## Project Overview
- One paragraph describing what the project does
- Architecture diagram (ASCII)
- Key file paths: src/, tests/, data/, config/

## Code Standards
- CRS-first: every function that accepts geometry MUST also accept
  or document its expected CRS. Never assume EPSG:4326.
- pathlib.Path for ALL file paths. No os.path, no string concat.
- Context managers for ALL file I/O (rasterio.open, fiona.open).
- Geometry validation: check is_valid before operations; use
  shapely.validation.make_valid() as fallback.
- I/O engine: use pyogrio for GeoPandas read/write (not Fiona).
- Type hints on ALL function signatures. Use numpy.typing for arrays.
- No shapefile output. Prefer GeoParquet > GeoPackage > GeoJSON.
- Logging: use structlog, not print(). JSON format for production.
- Testing: pytest + small fixtures in tests/fixtures/ (< 100 features).

## Architecture Decisions
- [ADR_1, e.g., "We use Dagster for orchestration because..."]
- [ADR_2, e.g., "We store config in YAML, not env vars, because..."]
- [ADR_3, e.g., "We target PostGIS 16 + GEOS 3.12 minimum"]

## Common Tasks
- "Add a new processing function" -> create in src/processing/,
  add tests in tests/test_processing.py, update CLI in src/cli.py
- "Fix a failing test" -> read the test, read the function,
  check CRS and geometry validity first (90% of spatial bugs)
- "Add a new data source" -> create extractor in src/extract/,
  add to pipeline DAG, add integration test

## Do NOT
- Do not use shapefile format for any output
- Do not use .to_file() without specifying driver explicitly
- Do not hardcode EPSG codes; use config.TARGET_CRS
- Do not use pandas merge for spatial joins; use geopandas.sjoin
- Do not commit data files (*.gpkg, *.tif, *.parquet) to git
```

#### Variables to Customize

- `PROJECT_NAME` -- name and description
- `TYPE` -- project type (library, pipeline, API, plugin)
- `DEPS` -- key dependencies
- Architecture decisions specific to your project
- Common task workflows specific to your codebase

#### Expected Output

A CLAUDE.md file (~80-120 lines) that can be placed in the project root. It serves as a system prompt for Claude Code and documentation for human developers simultaneously.

#### Validation Checklist

- [ ] Claude Code reads and follows the rules when generating code
- [ ] CRS is explicitly handled in all generated spatial functions
- [ ] pathlib.Path is used throughout (no os.path)
- [ ] Generated tests use fixtures from tests/fixtures/
- [ ] No shapefile outputs in any generated code

#### Cost Optimization

A well-written CLAUDE.md reduces back-and-forth iterations with the agent by 50-70%. Invest 30 minutes writing it once; save hours across the project lifetime. Keep it under 200 lines -- agents perform worse with overly long system prompts.

**Dark Arts Tip:** Add a `## Known Bugs` section listing current issues. Claude Code will avoid those patterns and may even fix them proactively when working nearby. It reads the whole CLAUDE.md on every invocation.

#### Related Pages

- [AI Agent Patterns](ai-agent-patterns.md) -- agent architecture patterns

#### Extensibility

Add a `.claude/` directory with task-specific instruction files. Reference them from CLAUDE.md for deep-dive guidance on complex subsystems.

---

### Prompt 18 -- Multi-Agent GIS Pipeline

#### Scenario (实际场景)

You want to use Claude Code (or a similar agent) in a multi-step pipeline where each step has a specialized "role" -- one agent scouts data, another engineers the pipeline, another analyzes results, another creates maps, and a final one writes the report. Each agent receives the previous agent's output as context.

#### Roles

Act as an AI Architect designing a multi-agent pipeline for end-to-end GIS project execution.

#### Prompt Template

```text
Design a 5-agent pipeline for a GIS project using Claude Code.
Each agent is a separate Claude Code invocation with specialized
context. Output from one agent feeds as input to the next.

PROJECT: [PROJECT_DESC, e.g., "Urban heat island analysis for
[CITY] using Landsat thermal imagery"]

AGENT CHAIN:

Agent 1 -- Data Scout
  Context: "You are a geospatial data researcher."
  Task: Find and catalog all available datasets for [PROJECT].
  Output: data_catalog.yaml with:
    - source URL, format, CRS, temporal range, spatial extent
    - download instructions (API, FTP, direct link)
    - license and attribution requirements
  Hand off: data_catalog.yaml -> Agent 2

Agent 2 -- Data Engineer
  Context: "You are a spatial data engineer. Read data_catalog.yaml."
  Task: Build ETL pipeline to download, clean, and unify all sources.
  Output: src/etl/ module with extract, transform, load functions
  Tests: tests/test_etl.py
  Hand off: processed data in data/processed/ -> Agent 3

Agent 3 -- Spatial Analyst
  Context: "You are a spatial analyst. Data is in data/processed/."
  Task: Perform [ANALYSIS, e.g., "LST calculation, UHI delineation,
  correlation with land cover and population density"]
  Output: src/analysis/ module with analysis functions
  Results: results/ directory with GeoPackage + statistics CSV
  Hand off: results/ -> Agent 4

Agent 4 -- Cartographer
  Context: "You are a cartographer. Results are in results/."
  Task: Create publication-quality maps for [AUDIENCE].
  Output: src/maps/ module generating:
    - Overview map, thematic map, comparison map
    - Consistent style: [COLOR_SCHEME], [PROJECTION], [DPI]
  Hand off: maps/ directory with PNGs -> Agent 5

Agent 5 -- Report Writer
  Context: "You are a technical writer. Maps in maps/, stats in results/"
  Task: Generate final report combining all outputs.
  Output: reports/final_report.pdf with maps, charts, narrative

ORCHESTRATION:
- run_pipeline.sh that invokes each agent sequentially
- Each agent writes to a known output directory
- Checkpoint after each agent (re-run from any stage)
- Total pipeline completes in under [TIME_LIMIT]
```

#### Variables to Customize

- `PROJECT_DESC` -- the end-to-end project description
- `ANALYSIS` -- specific spatial analysis to perform
- `AUDIENCE` -- who the maps and report are for
- `COLOR_SCHEME`, `PROJECTION`, `DPI` -- cartographic standards
- `TIME_LIMIT` -- total pipeline time budget

#### Expected Output

A run_pipeline.sh orchestration script, five agent prompt files (agent_1_scout.md through agent_5_writer.md), and a shared project structure. Each agent prompt is self-contained with context and expected outputs.

#### Validation Checklist

- [ ] Each agent produces its expected output files
- [ ] Agent 2 reads Agent 1's data_catalog.yaml correctly
- [ ] Pipeline can be re-run from any intermediate stage
- [ ] Final report contains maps and statistics from upstream agents
- [ ] Total runtime is within time budget

#### Cost Optimization

Each agent invocation costs API tokens. Keep agent prompts focused and concise (under 500 tokens of instruction). Pass only file paths between agents, not file contents. Use Claude Haiku for Agent 1 (research) and Agent 5 (writing); use Claude Opus for Agent 3 (analysis).

**Dark Arts Tip:** Run Agent 1 and Agent 4 prompts through Claude on the web (free tier) since they produce text/YAML, not code. Reserve Claude Code (paid) for Agents 2, 3, and 5 which need to read/write files and run tests.

#### Related Pages

- [AI Agent Patterns](ai-agent-patterns.md) -- multi-agent orchestration

#### Extensibility

Add a "Reviewer" agent (Agent 6) that reads all outputs, runs validation, and produces a QA report. Add parallel execution for independent agents (e.g., Agent 4 and Agent 5 can start simultaneously if results/ is ready).

---

### Prompt 19 -- Claude Code + MCP Server Workflow

#### Scenario (实际场景)

You want Claude Code to interact directly with spatial infrastructure -- querying PostGIS, searching STAC catalogs, running GDAL operations -- through Model Context Protocol (MCP) servers. This turns Claude Code from a code generator into an interactive spatial data operator.

#### Roles

Act as an AI Platform Engineer configuring MCP servers for Claude Code to interact with spatial infrastructure.

#### Prompt Template

```text
Configure Claude Code with MCP servers for spatial data operations.

MCP SERVERS TO SET UP:

1. PostGIS MCP Server
   - Connection: postgresql://[USER]:[PASS]@[HOST]:5432/[DB]
   - Capabilities: execute SQL, list tables, describe schema
   - Use case: Claude Code queries spatial data directly
   - Example: "Show me all parcels within 500m of the new school site"
   - Config in .claude/mcp.json or claude_desktop_config.json

2. STAC MCP Server (if using satellite imagery)
   - Catalog URL: [STAC_URL, e.g., https://earth-search.aws.element84.com/v1]
   - Capabilities: search items, filter by bbox/datetime/collection
   - Use case: Claude Code finds and selects satellite imagery
   - Example: "Find all Sentinel-2 scenes for [AOI] in June 2025
     with < 10% cloud cover"

3. Filesystem MCP Server (for data access)
   - Allowed paths: [DATA_DIR, OUTPUT_DIR]
   - Capabilities: read/write spatial files, list directories
   - Use case: Claude Code reads GeoPackage metadata, writes results

WORKFLOW EXAMPLE:
User: "Analyze tree canopy coverage per neighborhood"
Agent flow:
  1. [PostGIS MCP] -> query neighborhoods table for geometries
  2. [STAC MCP] -> find latest high-res imagery for the city bbox
  3. [Code] -> write Python script for tree canopy classification
  4. [Filesystem MCP] -> save results as GeoPackage
  5. [PostGIS MCP] -> load results, run zonal stats SQL
  6. [Code] -> generate summary report

GENERATE:
- .claude/mcp.json with server configurations
- README section explaining setup and usage
- Example prompts that leverage MCP servers
- Safety: read-only mode for production PostGIS, read-write for dev
```

#### Variables to Customize

- PostGIS connection string and access level
- STAC catalog URL
- Allowed filesystem paths
- Safety constraints (read-only vs. read-write)

#### Expected Output

MCP configuration file, setup instructions, and 5-10 example prompts demonstrating spatial MCP workflows. Total ~100 lines of config + documentation.

#### Validation Checklist

- [ ] Claude Code connects to PostGIS MCP and lists tables
- [ ] STAC search returns expected results for test query
- [ ] Filesystem MCP respects allowed path restrictions
- [ ] Production database is read-only (no accidental writes)
- [ ] Example prompts produce correct results

#### Cost Optimization

MCP server calls are free (local). The cost is Claude Code API tokens for reasoning. Use specific, well-scoped queries to minimize back-and-forth. Cache STAC search results locally for repeated queries.

**Dark Arts Tip:** Add a DuckDB MCP server for analytics queries on local GeoParquet files. It is faster than PostGIS for ad-hoc analytical queries and does not require a running database server. Claude Code can query terabytes of GeoParquet without loading into memory.

#### Related Pages

- [AI Agent Patterns](ai-agent-patterns.md) -- MCP architecture and server patterns
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS, DuckDB

#### Extensibility

Build a custom MCP server that wraps GDAL/OGR operations (reproject, convert, info) as tools. Claude Code can then run GDAL operations without writing shell commands.

---

### Prompt 20 -- Cursor/Windsurf Multi-File Workflow

#### Scenario (实际场景)

You are using Cursor (with Composer) or Windsurf (with Cascade) to generate or refactor a multi-file GIS project. You need to provide enough context for the AI to generate consistent, interconnected code across multiple files in a single pass.

#### Roles

Act as a GIS Developer using AI-enhanced IDEs for multi-file spatial project generation.

#### Prompt Template

```text
I am working in [IDE: Cursor / Windsurf]. I need to [ACTION: create /
refactor / extend] a multi-file GIS project.

PROJECT CONTEXT (reference these files with @ mentions):
@src/config.py       -- CRS constants, paths, settings
@src/data_loader.py  -- existing data I/O functions
@pyproject.toml      -- current dependencies
@tests/conftest.py   -- shared test fixtures

TASK: [TASK_DESC, e.g., "Add hotspot analysis module"]

FILES TO GENERATE/MODIFY:
1. @src/analysis/[MODULE].py
   - Implement [ALGORITHM, e.g., Getis-Ord Gi* using PySAL/esda]
   - Input: GeoDataFrame with [REQUIRED_COLUMNS]
   - Output: GeoDataFrame with hotspot_z, hotspot_p, hotspot_class
   - Use CRS from config.TARGET_CRS
   - Handle edge cases: too few features, missing values, islands

2. @src/analysis/__init__.py
   - Export new module's public functions

3. @src/cli.py
   - Add [SUBCOMMAND] to existing Typer CLI
   - Arguments: input_path, output_path, --distance-band, --alpha

4. @tests/test_[MODULE].py
   - Test with fixture GeoDataFrame (conftest.py)
   - Test: correct output columns, z-scores are numeric,
     classification matches expected for known input
   - Test: edge case with 3 features (minimum for spatial stats)

5. @pyproject.toml
   - Add [NEW_DEPS] to dependencies

CONSTRAINTS:
- Follow existing code patterns in @src/data_loader.py
- All functions: type hints + Google-style docstrings
- Tests use existing conftest.py fixtures (do not create new ones)
- No shapefile I/O; use GeoPackage or GeoParquet

Generate all files in one Composer/Cascade pass. I will review
the unified diff before accepting.
```

#### Variables to Customize

- `IDE` -- Cursor or Windsurf
- `ACTION` -- create, refactor, or extend
- `MODULE` -- new module name
- `ALGORITHM` -- spatial algorithm to implement
- `NEW_DEPS` -- additional Python packages needed

#### Expected Output

Multi-file generation instructions that Cursor Composer or Windsurf Cascade can execute in a single pass, producing consistent, interconnected files.

#### Validation Checklist

- [ ] All generated files reference each other correctly (imports work)
- [ ] New CLI subcommand integrates with existing CLI
- [ ] Tests pass with existing conftest.py fixtures
- [ ] Code style matches existing codebase patterns
- [ ] pyproject.toml additions are valid

#### Cost Optimization

Reference existing files with @ mentions to give the AI maximum context without re-explaining your codebase. Keep the task description focused -- one feature per Composer pass. Review the diff carefully; multi-file generation occasionally creates circular imports.

**Dark Arts Tip:** In Cursor, use `.cursorrules` (project root) to set persistent rules like "always use pyogrio engine" and "never output shapefiles." It is the Cursor equivalent of CLAUDE.md and saves you from repeating instructions in every Composer prompt.

#### Related Pages

- [AI Agent Patterns](ai-agent-patterns.md) -- IDE agent integration

#### Extensibility

Create a library of Composer prompt templates for common GIS tasks (add data source, add analysis, add visualization). Store them in a `prompts/` directory in your project for team reuse.

---

## 7. Project Bootstrap

```
                     Project Bootstrap Pipeline
  +----------+      +----------+      +----------+
  | P21: Full|      | P22: Web |      | P23:     |
  | GIS Proj |      | Mapping  |      | QGIS     |
  | Bootstrap|      | Project  |      | Plugin   |
  | (Python) |      | (monorepo|      | Scaffold |
  +----------+      +----------+      +----------+
       |                 |                  |
       v                 v                  v
  pyproject.toml+   MapLibre+FastAPI+   Plugin scaffold+
  src+tests+Docker  PostGIS monorepo   Processing+tests+CI
```

---

### Prompt 21 -- Full GIS Project Bootstrap

#### Scenario (实际场景)

You are starting a new GIS Python project and want the entire structure -- package config, source code, tests, Docker, CI, Makefile -- generated in one shot with working (not placeholder) content. After `make install && make test`, everything should pass.

#### Roles

Act as a Senior Python Developer bootstrapping a production-ready GIS project from scratch.

#### Prompt Template

```text
Bootstrap a complete GIS Python project from scratch.

PROJECT:
- Name: [PROJECT_NAME]
- Description: [ONE_SENTENCE]
- Type: [TYPE: "spatial analysis library" / "ETL pipeline" /
  "web mapping backend" / "remote sensing toolkit"]
- Python: [PY_VERSION, e.g., 3.12]
- License: [LICENSE, e.g., MIT]
- Target CRS: EPSG:[CODE]

GENERATE ALL FILES WITH WORKING CONTENT (no TODOs, no placeholders):

[project-name]/
  pyproject.toml          # hatch/setuptools, pinned spatial deps
  README.md               # overview, install, usage, contributing
  .gitignore              # Python + GIS (*.shp,*.tif,*.gpkg,*.parquet)
  .pre-commit-config.yaml # ruff, mypy, trailing-whitespace
  .github/workflows/
    ci.yml                # GDAL apt-get, ruff, mypy, pytest+cov
  src/[package]/
    __init__.py           # version from importlib.metadata
    config.py             # pydantic-settings: CRS, paths, DB URL
    io.py                 # read_vector, write_vector, read_raster
    processing.py         # 2-3 real spatial functions (buffer, clip, stats)
    cli.py                # Typer: process, validate, info commands
    py.typed              # PEP 561 marker
  tests/
    conftest.py           # GeoDataFrame fixtures (points, polygons)
    fixtures/
      sample_points.geojson    # 10 point features
      sample_polygons.geojson  # 5 polygon features
    test_io.py            # test read/write roundtrip
    test_processing.py    # test buffer, clip, stats
  Dockerfile              # multi-stage, GDAL, < 500MB
  docker-compose.yml      # app + PostGIS:16-3.4
  Makefile                # install, test, lint, format, docker-build

KEY DETAILS:
- pyproject.toml: geopandas>=1.0, rasterio>=1.3, shapely>=2.0,
  pyproj>=3.6, pyogrio>=0.9, typer>=0.12
- config.py: DEFAULT_CRS, TARGET_CRS, DATA_DIR, OUTPUT_DIR via
  pydantic-settings (reads from env vars with defaults)
- io.py: pyogrio engine, CRS validation on read, format auto-detect
- processing.py: buffer_features(gdf, distance), clip_to_boundary(gdf,
  boundary), zonal_stats(zones, values_raster) -- real implementations
- cli.py: `process` runs buffer+clip, `validate` checks CRS+schema,
  `info` prints dataset summary (CRS, extent, feature count)
- Dockerfile: FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.0 as builder
- ci.yml: uses the GDAL container as job container for speed

GOAL: `git init && make install && make test` succeeds first try.
```

#### Variables to Customize

- `PROJECT_NAME` -- package name (PEP 508 compatible)
- `TYPE` -- project type determines which processing.py functions to generate
- `PY_VERSION` -- Python version
- `CODE` -- target EPSG code for the project
- `LICENSE` -- open source license

#### Expected Output

A complete project directory with 15+ files, all containing working code. Total ~800 lines across all files.

#### Validation Checklist

- [ ] `make install` succeeds (all deps install)
- [ ] `make test` passes (all tests green)
- [ ] `make lint` passes (ruff + mypy clean)
- [ ] `docker-compose up` starts app + PostGIS
- [ ] `python -m [package].cli --help` shows commands

#### Cost Optimization

This prompt is long but saves 2-4 hours of manual project setup. Use it once per project. Store the output as a cookiecutter/copier template for future reuse.

**Dark Arts Tip:** After generating, immediately run `make test` inside Claude Code and let it fix any failures. The agent is better at debugging its own output than generating perfect code on the first try. Budget 2-3 iteration rounds.

#### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- spatial dependency versions
- [ETL & Data Engineering](../tools/etl-data-engineering.md) -- project structure patterns

#### Extensibility

Convert the generated project into a copier or cookiecutter template with Jinja2 variables. Maintain it as your organization's GIS project standard.

---

### Prompt 22 -- Web Mapping Project Bootstrap

#### Scenario (实际场景)

You are building a web mapping application with a JavaScript frontend (MapLibre GL JS), a Python backend (FastAPI), and a PostGIS database. You need a monorepo structure with both frontend and backend, shared Docker setup, and CI for both.

#### Roles

Act as a Full-Stack GIS Developer bootstrapping a web mapping monorepo.

#### Prompt Template

```text
Bootstrap a web mapping monorepo with frontend + backend + database.

PROJECT: [PROJECT_NAME]
DESCRIPTION: [ONE_SENTENCE]

ARCHITECTURE:
- Frontend: MapLibre GL JS + [FRAMEWORK: Svelte / React / Vue / vanilla]
- Backend: FastAPI + SQLAlchemy + GeoAlchemy2
- Database: PostGIS 16
- Tile serving: [Martin / TiTiler / pg_tileserv]

GENERATE MONOREPO:

[project-name]/
  docker-compose.yml        # frontend-dev, backend, postgis, [tile-server]
  Makefile                  # install, dev, test, build, deploy
  .github/workflows/ci.yml # lint+test both frontend and backend
  frontend/
    package.json            # MapLibre GL JS, [FRAMEWORK] deps
    src/
      App.[ext]             # Map component with layer toggle, popup
      api.js                # Fetch helpers for backend endpoints
      styles/map-style.json # MapLibre style with vector tile source
    vite.config.js          # Dev server with API proxy to backend
  backend/
    pyproject.toml          # FastAPI, GeoAlchemy2, uvicorn
    src/[package]/
      __init__.py
      config.py             # DB URL, CRS, CORS origins
      database.py           # async SQLAlchemy engine + session
      models.py             # GeoAlchemy2 models with geometry columns
      schemas.py            # Pydantic + GeoJSON response models
      routes/
        features.py         # CRUD + spatial query endpoints
        tiles.py            # MVT tile endpoint (if not using Martin)
      main.py               # FastAPI app with CORS, lifespan
    tests/
      conftest.py           # Test DB with PostGIS, fixture data
      test_features.py      # Test CRUD and spatial queries
    alembic/
      env.py                # GeoAlchemy2-aware migrations
      versions/             # Initial migration with spatial table
  scripts/
    seed_data.py            # Load sample data into PostGIS

KEY ENDPOINTS:
- GET /api/features?bbox=x1,y1,x2,y2 -- spatial query within bbox
- GET /api/features/{id} -- single feature as GeoJSON
- POST /api/features -- create feature (GeoJSON body)
- GET /api/tiles/{z}/{x}/{y}.mvt -- vector tiles (optional)

GOAL: `docker-compose up` starts everything; frontend at :5173,
backend at :8000, PostGIS at :5432.
```

#### Variables to Customize

- `FRAMEWORK` -- frontend framework (Svelte, React, Vue)
- Tile server -- Martin, TiTiler, pg_tileserv, or built-in endpoint
- API endpoints -- customize for your data model
- Authentication -- add JWT/OAuth if needed

#### Expected Output

Monorepo with frontend, backend, database, and Docker setup. ~600 lines across all files. Frontend shows an interactive map; backend serves GeoJSON and spatial queries.

#### Validation Checklist

- [ ] `docker-compose up` starts all services
- [ ] Frontend map loads and displays data
- [ ] Backend /api/features?bbox= returns GeoJSON
- [ ] Alembic migration creates spatial table
- [ ] CI tests pass for both frontend and backend

#### Cost Optimization

Use Martin for vector tiles -- it reads directly from PostGIS with zero config, eliminating a custom tile endpoint. Use Vite proxy for development to avoid CORS complexity. Keep PostGIS on the smallest instance that fits your data.

**Dark Arts Tip:** Skip the backend entirely for read-only maps. Use Martin + PostGIS + a static frontend (hosted on S3/Cloudflare Pages). Martin serves MVT tiles directly from PostGIS. Your "backend" is SQL views. Total infrastructure cost: ~$15/month.

#### Related Pages

- [Framework Integration](../js-bindbox/framework-integration.md) -- MapLibre + frameworks
- [Server & Publishing](../tools/server-publishing.md) -- Martin, TiTiler, pg_tileserv

#### Extensibility

Add authentication (FastAPI-Users + OAuth). Add WebSocket endpoint for real-time feature updates. Add a mobile-responsive layout with responsive map controls.

---

### Prompt 23 -- QGIS Plugin Project Bootstrap

#### Scenario (实际场景)

You are developing a QGIS plugin with a Processing provider (for the Processing toolbox) and want a complete project scaffold with UI, tests, CI (including pytest-qgis), and packaging for the QGIS Plugin Repository.

#### Roles

Act as a QGIS Plugin Developer bootstrapping a plugin project with Processing integration and CI.

#### Prompt Template

```text
Bootstrap a complete QGIS plugin project with Processing provider.

PLUGIN:
- Name: [PLUGIN_NAME]
- Display name: [DISPLAY_NAME]
- Description: [ONE_SENTENCE]
- Category: [Analysis / Vector / Raster / Database / Web]
- Min QGIS version: [MIN_VERSION, e.g., 3.34]
- Author: [AUTHOR]

GENERATE PROJECT:

[plugin-name]/
  metadata.txt              # QGIS plugin metadata (name, version, etc.)
  __init__.py               # classFactory entry point
  plugin.py                 # Main plugin class (initGui, unload)
  provider.py               # QgsProcessingProvider subclass
  algorithms/
    __init__.py
    [algorithm_1].py        # QgsProcessingAlgorithm: [ALG_DESC_1]
    [algorithm_2].py        # QgsProcessingAlgorithm: [ALG_DESC_2]
  ui/
    dialog.py               # Optional: custom dialog (QDialog)
    dialog.ui               # Qt Designer UI file
    resources.qrc           # Icons and resources
  tests/
    conftest.py             # pytest-qgis fixtures (qgis_app, etc.)
    test_[algorithm_1].py   # Test algorithm with sample data
    test_[algorithm_2].py
    fixtures/
      sample_layer.geojson  # Test data
  scripts/
    compile_resources.sh    # pyrcc5 resources.qrc -> resources_rc.py
    package.sh              # Zip plugin for QGIS repo upload
    install_dev.sh          # Symlink to QGIS plugin directory
  .github/workflows/
    ci.yml                  # pytest-qgis on Ubuntu + QGIS Docker
    release.yml             # Package and upload on tag
  Makefile                  # test, lint, package, install-dev

ALGORITHM TEMPLATE ([algorithm_1].py):
- Subclass QgsProcessingAlgorithm
- Define: name, displayName, group, groupId, shortHelpString
- Parameters: INPUT (vector layer), [PARAM_2], OUTPUT
- processAlgorithm: implement [ALG_DESC_1]
- Use QgsProcessingFeedback for progress reporting
- Handle CRS: reproject if input CRS != expected

CI (.github/workflows/ci.yml):
- Container: qgis/qgis:[QGIS_VERSION]
- Install pytest, pytest-qgis
- Run: pytest tests/ --qgis-no-gui

PACKAGING (scripts/package.sh):
- Compile .qrc resources
- Copy plugin files to temp dir (exclude tests/, .git/)
- Zip as [plugin-name].zip for QGIS Plugin Repository upload
```

#### Variables to Customize

- `PLUGIN_NAME` -- Python package name
- `DISPLAY_NAME` -- human-readable name in QGIS
- `ALG_DESC_1`, `ALG_DESC_2` -- Processing algorithm descriptions
- `MIN_VERSION` -- minimum QGIS version
- `QGIS_VERSION` -- Docker image tag for CI

#### Expected Output

Complete plugin project with Processing provider, two algorithms, Qt UI, tests, CI, and packaging scripts. ~500 lines across all files.

#### Validation Checklist

- [ ] Plugin loads in QGIS without errors
- [ ] Algorithms appear in Processing toolbox
- [ ] Tests pass with pytest-qgis
- [ ] `make package` creates a valid .zip
- [ ] CI runs tests in QGIS Docker container

#### Cost Optimization

Use the QGIS Docker image for CI to avoid installing QGIS from source (saves 10+ minutes per CI run). Use `install_dev.sh` for symlink-based development (instant reload, no re-packaging).

**Dark Arts Tip:** Skip the custom QDialog entirely and use Processing parameters exclusively. The Processing framework auto-generates a perfectly functional dialog. Custom dialogs are only justified for truly complex interfaces -- and they triple your maintenance burden.

#### Related Pages

- [Desktop GIS](../tools/desktop-gis.md) -- QGIS plugin development

#### Extensibility

Add a `help/` directory with Sphinx docs that embed into QGIS Help. Add i18n support with `.ts` translation files for multilingual plugins.

---

## 8. Infrastructure as Code

```
                   Infrastructure as Code
  +--------------+            +--------------+
  | P24:         |            | P25:         |
  | Terraform    |            | Docker       |
  | (PostGIS RDS |            | Recipes      |
  |  + tile      |            | (GDAL, Martin|
  |  server +CDN)|            |  TiTiler,    |
  +--------------+            |  PostGIS)    |
        |                     +--------------+
        v                           |
  HCL modules for                   v
  spatial infra              Optimized Dockerfiles
```

---

### Prompt 24 -- Terraform for Spatial Infrastructure

#### Scenario (实际场景)

You need to provision cloud infrastructure for a geospatial application: a PostGIS database, a tile server (Martin or TiTiler), a CDN for static tiles (PMTiles), and supporting networking and IAM. You want this defined as Terraform modules for repeatability across environments (dev/staging/prod).

#### Roles

Act as a Cloud Infrastructure Engineer writing Terraform modules for a geospatial application stack.

#### Prompt Template

```text
Write Terraform modules for a spatial application infrastructure.

CLOUD PROVIDER: [PROVIDER: AWS / GCP / Azure]
ENVIRONMENT: [ENV: dev / staging / prod]

MODULES:

1. module "database" -- PostGIS
   - [AWS]: RDS PostgreSQL [VERSION] with PostGIS extension
   - Instance: [INSTANCE_TYPE, e.g., db.t3.medium]
   - Storage: [SIZE]GB gp3, auto-scaling to [MAX]GB
   - Enable PostGIS, postgis_raster, postgis_topology extensions
   - Private subnet, security group (port 5432 from app SG only)
   - Automated backups: 7 days retention
   - Parameter group: shared_buffers, work_mem, effective_cache_size
     tuned for spatial workloads (large random reads)
   - Output: connection string (as SSM parameter / secret)

2. module "tile_server" -- Martin or TiTiler
   - [AWS]: ECS Fargate service / [GCP]: Cloud Run
   - Image: [TILE_SERVER: ghcr.io/maplibre/martin / developmentseed/titiler]
   - Environment: DATABASE_URL from secrets manager
   - Auto-scaling: min [MIN], max [MAX] based on CPU/request count
   - Health check: /healthz
   - ALB/Cloud Load Balancer with HTTPS (ACM/managed cert)

3. module "cdn" -- Static tile hosting
   - [AWS]: S3 bucket + CloudFront / [GCP]: GCS + Cloud CDN
   - Origin: S3 bucket for PMTiles files
   - Cache behavior: cache GETs for 24h, custom cache key by path
   - CORS headers for MapLibre GL JS access
   - Invalidation on new PMTiles upload (via Lambda/Cloud Function)

4. module "networking" -- VPC
   - VPC with public + private subnets (2 AZs minimum)
   - NAT gateway for private subnet internet access
   - VPC endpoints for S3 (gateway) and Secrets Manager (interface)

5. module "monitoring"
   - CloudWatch / Cloud Monitoring dashboards
   - Alarms: DB CPU > 80%, DB storage < 20%, tile server 5xx > 1%
   - SNS topic / notification channel for alerts

STRUCTURE:
  terraform/
    main.tf           # Module composition
    variables.tf      # Environment-specific vars
    outputs.tf        # Connection strings, URLs
    terraform.tfvars  # dev values (gitignored)
    modules/
      database/       # main.tf, variables.tf, outputs.tf
      tile_server/
      cdn/
      networking/
      monitoring/

Include a cost estimate comment for each module (dev environment).
```

#### Variables to Customize

- `PROVIDER` -- AWS, GCP, or Azure
- `TILE_SERVER` -- Martin, TiTiler, pg_tileserv
- Instance sizes -- adjust per environment
- Database version and extensions
- CDN caching rules

#### Expected Output

Terraform module structure with 5 modules, root composition, and variable definitions. ~400 lines of HCL. Includes cost estimate comments.

#### Validation Checklist

- [ ] `terraform validate` passes for all modules
- [ ] `terraform plan` shows expected resources
- [ ] Database is in private subnet (not publicly accessible)
- [ ] Tile server auto-scales based on load
- [ ] CDN serves PMTiles with correct CORS headers

#### Cost Optimization

Dev environment: use db.t3.micro (free tier eligible), single-AZ, no NAT gateway (use VPC endpoints instead). Estimated dev cost: $30-50/month. Prod: use reserved instances for database (30-40% savings). PMTiles on S3 + CloudFront is dramatically cheaper than a tile server for static tilesets.

**Dark Arts Tip:** For dev/staging, skip Terraform entirely and use docker-compose with PostGIS + Martin on a single $5/month VPS (Hetzner, DigitalOcean). Reserve Terraform for production only. Over-engineering dev infrastructure is the most common IaC mistake in small teams.

#### Related Pages

- [Server & Publishing](../tools/server-publishing.md) -- Martin, TiTiler, PMTiles hosting
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS configuration

#### Extensibility

Add a `module "bastion"` for SSH tunneling to the private database. Add Terragrunt for DRY multi-environment management. Add a GitHub Actions workflow for `terraform plan` on PR and `terraform apply` on merge to main.

---

### Prompt 25 -- Docker Recipes for GIS

#### Scenario (实际场景)

You need optimized Dockerfiles for common GIS components: a Python spatial analysis environment with GDAL, a PostGIS database with custom extensions, a Martin tile server, and a TiTiler raster tile server. Each image should be production-ready with minimal size and proper security.

#### Roles

Act as a Container Engineer writing optimized, secure Dockerfiles for geospatial applications.

#### Prompt Template

```text
Write optimized Dockerfiles for common GIS components.

RECIPE 1: Python Spatial Analysis
  Base: ghcr.io/osgeo/gdal:ubuntu-small-[GDAL_VERSION]
  Python: [PY_VERSION]
  Packages: geopandas, rasterio, shapely, pyproj, fiona, pyogrio,
            scipy, scikit-learn, matplotlib
  Multi-stage: builder installs all deps, runtime copies wheels
  Non-root user: appuser (UID 1000)
  Target size: < 400 MB
  HEALTHCHECK: python -c "import geopandas; print('ok')"

RECIPE 2: PostGIS with Extensions
  Base: postgis/postgis:[PG_VERSION]-[POSTGIS_VERSION]
  Extensions: postgis, postgis_raster, postgis_topology,
              pg_trgm, btree_gist, [EXTRA_EXTENSIONS]
  Init scripts: create spatial database, enable extensions,
                set tuning parameters (shared_buffers, work_mem)
  Persistent volume: /var/lib/postgresql/data
  Locale: en_US.UTF-8

RECIPE 3: Martin Vector Tile Server
  Base: ghcr.io/maplibre/martin:[MARTIN_VERSION]
  Config: martin-config.yaml with PostGIS connection,
          auto-discover tables, sprite sources, font sources
  Expose: 3000
  Health: /healthz
  Target size: < 50 MB (Martin is a single binary)

RECIPE 4: TiTiler Raster Tile Server
  Base: python:3.12-slim
  Packages: titiler.core, titiler.extensions, titiler.mosaic
  Optimizations: GDAL_CACHEMAX, CPL_VSIL_CURL_ALLOWED_EXTENSIONS,
                 GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR,
                 VSI_CACHE=TRUE, VSI_CACHE_SIZE=536870912
  Expose: 8000
  Uvicorn workers: [WORKERS, e.g., 4]
  Target size: < 300 MB

FOR EACH RECIPE:
- .dockerignore file
- Build command with cache mount: docker buildx build --cache-from
- docker-compose.yml snippet showing the service
- Environment variables documented as comments
- Security: no root, no unnecessary packages, pinned versions

ALSO GENERATE:
- A combined docker-compose.yml with all four services + networking
- A Makefile with: build-all, up, down, logs, shell-[service]
```

#### Variables to Customize

- `GDAL_VERSION` -- GDAL version (3.8, 3.9)
- `PG_VERSION` -- PostgreSQL version (15, 16, 17)
- `POSTGIS_VERSION` -- PostGIS version (3.4, 3.5)
- `MARTIN_VERSION` -- Martin version
- `WORKERS` -- uvicorn worker count
- `EXTRA_EXTENSIONS` -- additional PostgreSQL extensions

#### Expected Output

Four Dockerfiles, a combined docker-compose.yml, .dockerignore, and Makefile. Each Dockerfile includes build optimization comments and size targets. ~300 lines total.

#### Validation Checklist

- [ ] All images build successfully
- [ ] Image sizes meet targets (check with `docker images`)
- [ ] `docker-compose up` starts all services and they communicate
- [ ] Martin serves vector tiles from PostGIS tables
- [ ] TiTiler serves raster tiles from COG files
- [ ] No containers run as root

#### Cost Optimization

Smaller images = faster pulls = lower registry storage costs. Multi-stage builds reduce Python spatial image from ~1.5GB to ~400MB. Use `--cache-from` for CI builds to reuse layers. Pin base image digests (not just tags) for reproducible builds.

**Dark Arts Tip:** For the Python spatial image, install everything via conda-forge instead of pip+apt. The conda-forge `gdal` package bundles GEOS, PROJ, and all C libraries in a single consistent environment. Trade-off: larger image (~800MB), but zero dependency conflicts. Use `mambaforge` for fast installs: `mamba install -y geopandas rasterio` takes 30 seconds vs. 3 minutes for pip + apt.

#### Related Pages

- [Server & Publishing](../tools/server-publishing.md) -- Martin, TiTiler deployment
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS Docker setup

#### Extensibility

Add a NGINX reverse proxy recipe with tile caching (proxy_cache). Add a pgAdmin recipe for database administration. Add a Jupyter Lab recipe with all spatial packages pre-installed for interactive analysis.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)

