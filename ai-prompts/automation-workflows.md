# Automation Workflows

> Prompt templates for automating repetitive GIS tasks, building ETL pipelines, generating reports, running quality assurance, and setting up CI/CD for geospatial projects. Each section includes prompt chains showing how multiple prompts work together.

> **Quick Picks**
> - 🏆 **Most Useful**: [Full Project Bootstrap](#prompt-3--full-gis-project-bootstrap) — set up an entire GIS project from scratch in one prompt
> - 🚀 **Time Saver**: [Claude Code / Cursor Workflows](#ai-coding-assistant-workflows) — leverage agentic AI to automate multi-file GIS development
> - 🆕 **Cutting Edge**: [GitHub Actions Spatial Data Validation](#prompt-3--github-actions-spatial-data-validation) — automated data quality gates in your CI pipeline

---

## Table of Contents

- [Batch Processing](#batch-processing)
- [ETL Pipelines](#etl-pipelines)
- [Report Generation](#report-generation)
- [Quality Assurance](#quality-assurance)
- [CI/CD for GIS](#cicd-for-gis)
- [AI Coding Assistant Workflows](#ai-coding-assistant-workflows)
- [Project Bootstrap](#project-bootstrap)

---

## Batch Processing

### Prompt 1 — Batch Geoprocessing Script

**Context:** You have hundreds of files that need the same processing steps applied. Manual processing is not feasible.

**Template:**

```
Write a Python script to batch-process all [file type, e.g., GeoTIFF rasters] in [input directory].

For each file, perform these steps:
1. [Step 1, e.g., "Clip to study area boundary from [boundary file]"]
2. [Step 2, e.g., "Reproject from source CRS to [target EPSG]"]
3. [Step 3, e.g., "Calculate NDVI and save as a new band"]
4. [Step 4, e.g., "Resample to [target resolution] using bilinear interpolation"]

Requirements:
- Process files in parallel using [multiprocessing / concurrent.futures] with [N] workers
- Skip files that already have output (idempotent — safe to re-run)
- Log each file: start time, end time, success/failure, output path
- If a file fails, log the error and continue with the next file (do not stop the batch)
- Write a summary CSV at the end: filename, status, processing time, output size
- Support a --dry-run flag that lists what would be processed without doing it
- Accept CLI arguments: input dir, output dir, boundary file, CRS, workers

Use [library: Rasterio / GDAL / GeoPandas] for processing.
```

**Variables to customize:**
- File type and processing steps
- Parallelism settings
- Library preference
- CLI arguments

**Expected output format:** A complete Python CLI script with logging, parallelism, and error handling.

---

### Prompt 2 — Watch Folder and Auto-Process

**Context:** You receive new data files regularly and want them processed automatically when they appear.

**Template:**

```
Write a Python script that watches [directory] for new [file type] files and automatically processes them.

When a new file is detected:
1. Wait for the file to finish writing (check file size stability)
2. Validate the file (readable, expected CRS, expected schema/bands)
3. Run processing: [describe steps]
4. Move the processed file to [output directory]
5. Move the original to [archive directory]
6. If processing fails, move to [error directory]
7. Send a notification [method: email / Slack webhook / log file]

Use the watchdog library for file system monitoring.
Include a systemd service file (Linux) or launchd plist (macOS) for running as a daemon.
```

**Variables to customize:**
- Directory and file type
- Processing steps
- Notification method
- Deployment platform

**Expected output format:** Python script, watchdog configuration, and service file for daemonization.

---

### Workflow Diagram — Batch Processing Chain

```
Prompt Chain: Batch Processing Automation

  [Prompt 1: Scaffold]          [Prompt 2: Error Handling]        [Prompt 3: Monitoring]
  "Write a batch script     --> "Add robust error handling    --> "Add a watch mode that
   that processes all            with retry logic for              auto-processes new
   shapefiles in a dir"          transient failures"               files as they arrive"
         |                             |                                  |
         v                             v                                  v
  +----------------+          +------------------+            +--------------------+
  | Basic Script   |          | Retry + Logging  |            | Daemon Service     |
  | - Loop files   |          | - 3 retries      |            | - watchdog         |
  | - Process each |          | - Exponential    |            | - systemd/launchd  |
  | - Save output  |          |   backoff        |            | - Notification     |
  +----------------+          | - Error dir      |            +--------------------+
                              +------------------+
```

---

## ETL Pipelines

### Prompt 1 — Spatial ETL Pipeline

**Context:** You need to extract data from multiple sources, transform it into a unified schema, and load it into a spatial database.

**Template:**

```
Design a spatial ETL pipeline in Python that:

EXTRACT:
- Source 1: [format, e.g., Shapefiles from an FTP server at [URL], updated weekly]
- Source 2: [format, e.g., GeoJSON from a REST API at [endpoint], real-time]
- Source 3: [format, e.g., CSV with lat/lon from [source], daily dump]

TRANSFORM:
- Unify CRS to [EPSG code]
- Standardize column names: [mapping, e.g., "OBJECTID" -> "id", "Shape_Area" -> "area_m2"]
- Clean geometries (fix invalid, remove empty)
- Deduplicate based on [key columns]
- Enrich: spatial join with [reference dataset] to add [column]
- Validate: [rules, e.g., "area_m2 must be > 0", "name must not be null"]

LOAD:
- Target: [PostGIS / GeoPackage / BigQuery GIS / Snowflake]
- Schema: [table name, columns with types]
- Mode: [upsert on key / truncate and reload / append only]
- Create spatial index on geometry column

Requirements:
- Use [Prefect / Airflow / dagster / plain Python] for orchestration
- Idempotent: safe to re-run without duplicating data
- Log every step with row counts (extracted, transformed, loaded, rejected)
- Handle source downtime gracefully (retry with backoff)
```

**Variables to customize:**
- Data sources with formats and update frequencies
- Transformation rules and schema mapping
- Target database and load mode
- Orchestration framework

**Expected output format:** Python ETL pipeline with extraction, transformation, loading, and orchestration code.

---

### Prompt 2 — Incremental Sync from API to Database

**Context:** You want to keep a PostGIS table in sync with an external API that provides geospatial data.

**Template:**

```
Write a Python script that incrementally syncs data from [API endpoint] to a PostGIS table.

API details:
- Base URL: [url]
- Authentication: [API key / OAuth / none]
- Pagination: [method, e.g., offset/limit, cursor-based, page number]
- Filter for updates: [method, e.g., ?updated_after=YYYY-MM-DD]
- Response format: [GeoJSON / JSON with lat/lon]
- Rate limit: [N requests per minute]

PostGIS table:
- Name: [schema.table]
- Primary key: [column]
- Geometry column: geom (SRID [code])
- Other columns: [list]

Sync logic:
1. Query the last sync timestamp from a metadata table
2. Fetch records updated since that timestamp
3. For each record: INSERT if new, UPDATE if existing (upsert)
4. Update the sync timestamp on success
5. Handle: API errors, network timeouts, malformed records
6. Log: records fetched, inserted, updated, skipped, failed

Schedule: run as a cron job every [interval].
```

**Variables to customize:**
- API details and authentication
- Database schema
- Sync frequency
- Error handling preferences

**Expected output format:** Python script with API client, database operations, and cron configuration.

---

### Workflow Diagram — ETL Pipeline

```
Prompt Chain: Spatial ETL Development

  [Prompt 1: Design]            [Prompt 2: Implement]          [Prompt 3: Schedule]
  "Design the data flow     --> "Implement the extract,    --> "Add scheduling with
   and schema mapping"          transform, load functions"      Prefect/Airflow and
                                                                alerting on failure"
         |                             |                                |
         v                             v                                v
  +------------------+        +-------------------+         +---------------------+
  | Architecture     |        | Python ETL Code   |         | DAG / Flow          |
  | - Source schemas |        | - extract()       |         | - Schedule (cron)   |
  | - Target schema  |        | - transform()     |         | - Retry policy      |
  | - Mapping rules  |        | - load()          |         | - Slack/email alert |
  | - Data flow      |        | - validate()      |         | - Dashboard         |
  +------------------+        +-------------------+         +---------------------+
```

---

## Report Generation

### Prompt 1 — Automated Map Report

**Context:** You regularly produce reports that include maps, statistics, and text for a recurring analysis (e.g., monthly environmental monitoring).

**Template:**

```
Write a Python script that generates an automated report for [report type, e.g., "monthly vegetation
health monitoring"] as a [PDF / HTML / DOCX].

Input data:
- Raster: [e.g., monthly NDVI composite]
- Vector: [e.g., administrative boundaries]
- Statistics: [e.g., CSV with zonal statistics per district]

Report contents:
1. Title page with: title, date, author, organization logo
2. Executive summary (auto-generated text summarizing key findings)
3. Study area map (overview)
4. Thematic map: [description, e.g., NDVI choropleth by district]
5. Change map: [description, e.g., comparison with previous month]
6. Statistics table: [columns, e.g., district, mean NDVI, change, trend]
7. Time series chart: [description, e.g., 12-month NDVI trend line per district]
8. Automated narrative: text describing the results using conditional logic
   (e.g., "District X showed a [increase/decrease] of [value]% compared to last month")
9. Data sources and methodology page

Use [library: ReportLab / WeasyPrint / python-docx / Jinja2 + HTML] for report generation.
Use matplotlib or plotly for charts and maps.
Parameterize: report date, input file paths, output path.
```

**Variables to customize:**
- Report type and frequency
- Input data sources
- Report sections and content
- Output format and library

**Expected output format:** Python script generating a complete report with maps, charts, tables, and narrative text.

---

### Prompt 2 — Dashboard Data Preparation

**Context:** You maintain a web dashboard that displays geospatial data and need to automate the data preparation.

**Template:**

```
Write a Python script that prepares data for a [dashboard framework: Streamlit / Dash / Panel] dashboard
showing [topic, e.g., "real-time air quality across monitoring stations"].

Data preparation steps:
1. Fetch latest data from [source]
2. Clean and validate (remove outliers, fill gaps using [method])
3. Calculate derived metrics: [list, e.g., AQI from raw pollutant values]
4. Aggregate to [spatial level, e.g., district] and [temporal level, e.g., hourly]
5. Generate GeoJSON for the map layer with all display attributes
6. Generate time series JSON for charts
7. Cache results to [Redis / file / database] with [TTL]

Also write the dashboard layout code that:
- Shows an interactive map (folium / pydeck) with color-coded stations
- Sidebar with filters: date range, pollutant type, district
- Time series chart below the map
- Summary statistics cards (current max, min, average)
- Auto-refreshes every [interval]
```

**Variables to customize:**
- Dashboard framework
- Data source and topic
- Aggregation levels
- Refresh interval

**Expected output format:** Data preparation script plus dashboard layout code.

---

## Quality Assurance

### Prompt 1 — Spatial Data Validation

**Context:** You receive data deliveries from external providers and need to validate them against a specification before ingestion.

**Template:**

```
Write a Python validation script that checks a [format, e.g., GeoPackage] against these quality rules:

Schema validation:
- Required layers: [list layer names]
- Required columns per layer: [list columns with expected types]
- Geometry type per layer: [Point / LineString / Polygon / Multi*]

Geometry validation:
- No null geometries
- No invalid geometries (self-intersection, ring self-intersection)
- No duplicate geometries
- All geometries within [bounding box or reference boundary]
- Minimum feature size: [threshold, e.g., polygons > 1 m²]

Attribute validation:
- Column [A]: not null, unique
- Column [B]: value in [allowed values list]
- Column [C]: numeric, range [min] to [max]
- Column [D]: valid date format (ISO 8601)
- Column [E]: valid URL format (if present)

CRS validation:
- Must be [EPSG code]
- If different, report the actual CRS but do NOT auto-reproject

Topology validation:
- No overlapping polygons within the same layer (if applicable)
- No gaps between adjacent polygons (if applicable)

Output:
- PASS / FAIL overall verdict
- Detailed report: one row per check with status, count of issues, sample IDs
- Export failed features as a separate layer for review
- Return exit code 0 for pass, 1 for fail (usable in CI/CD pipelines)
```

**Variables to customize:**
- File format and schema requirements
- Geometry and attribute validation rules
- Topology rules
- Output format

**Expected output format:** Python validation script with detailed reporting and CI/CD-compatible exit codes.

---

### Prompt 2 — Automated Regression Testing for GIS Outputs

**Context:** You have a GIS processing pipeline and want to ensure outputs remain consistent when you change the code.

**Template:**

```
Write a pytest test suite that validates the outputs of a GIS processing pipeline.

Pipeline: [describe what it does, e.g., "clips, reprojects, and classifies land cover"]
Input: [test fixture data — describe or provide path]
Expected output: [reference output to compare against]

Tests to include:
1. Output file exists and is readable
2. Output CRS matches expected [EPSG code]
3. Feature count matches expected [N] (within tolerance of +/- [M])
4. Bounding box matches expected extent (within [tolerance] degrees/meters)
5. Attribute schema matches expected columns and types
6. Geometry types are all [expected type]
7. No null geometries or null values in required columns
8. Statistical comparison: mean/std of column [X] within [tolerance]% of reference
9. Visual regression: hash of rendered map image matches reference (optional)
10. Performance: processing time under [threshold] seconds

Use pytest fixtures for setup/teardown of temporary directories.
Include a conftest.py with shared test data paths.
Generate a JUnit XML report for CI integration.
```

**Variables to customize:**
- Pipeline description
- Test data and reference outputs
- Tolerance values
- Performance thresholds

**Expected output format:** pytest test suite with conftest.py and CI configuration.

---

### Workflow Diagram — QA Pipeline

```
Prompt Chain: Quality Assurance Automation

  [Prompt 1: Spec]              [Prompt 2: Validator]          [Prompt 3: CI Integration]
  "Define the data quality  --> "Write the validation       --> "Integrate the validator
   rules as a JSON schema"      script implementing these       into a GitHub Actions
                                rules"                          workflow"
         |                             |                                |
         v                             v                                v
  +------------------+        +-------------------+         +---------------------+
  | QA Specification |        | Validation Script |         | CI/CD Pipeline      |
  | - Schema rules   |        | - check_schema()  |         | - On PR: validate   |
  | - Geometry rules |        | - check_geom()    |         | - On push: test     |
  | - Attribute rules|        | - check_attr()    |         | - Report as comment |
  | - Topology rules |        | - check_topo()    |         | - Block merge if    |
  +------------------+        | - report()        |         |   validation fails  |
                              +-------------------+         +---------------------+
```

---

## CI/CD for GIS

### Prompt 1 — GitHub Actions for Geospatial Project

**Context:** You have a geospatial Python project on GitHub and want automated testing, linting, and deployment.

**Template:**

```
Write a GitHub Actions workflow (.github/workflows/ci.yml) for a geospatial Python project.

Project details:
- Python version: [version]
- Dependencies: managed via [requirements.txt / pyproject.toml / conda environment.yml]
- Spatial dependencies: GDAL, GEOS, PROJ (need system-level installation)
- Test framework: pytest with [plugins: pytest-cov, pytest-qgis]
- Linter: [ruff / flake8 / black + isort]

Workflow steps:
1. Trigger on: push to main, pull requests to main
2. Matrix: test on [Ubuntu latest, optionally macOS]
3. Set up Python and install system spatial libs (apt-get install gdal-bin libgdal-dev)
4. Install Python dependencies
5. Run linter and type checker (mypy)
6. Run tests with coverage (fail if coverage < [threshold]%)
7. Upload coverage report to [Codecov / as artifact]
8. If on main branch push: build and push Docker image to [registry]
9. Deploy documentation to GitHub Pages (if docs/ exists)

Also write:
- A Dockerfile for the project (with GDAL properly installed)
- A .pre-commit-config.yaml for local development
```

**Variables to customize:**
- Python version and dependency management
- Test framework and coverage threshold
- Docker registry
- OS matrix

**Expected output format:** GitHub Actions YAML, Dockerfile, and pre-commit config.

---

### Prompt 2 — Automated Data Pipeline Deployment

**Context:** You want to deploy a GIS data processing pipeline that runs on a schedule in the cloud.

**Template:**

```
Write the infrastructure and deployment configuration for a scheduled GIS data pipeline.

Pipeline: [describe, e.g., "downloads Sentinel-2 imagery weekly, processes it, and updates a PostGIS database"]
Schedule: [cron expression, e.g., "every Monday at 03:00 UTC"]
Cloud provider: [AWS / GCP / Azure]

Create:
1. Dockerfile with all spatial dependencies (GDAL, Python, etc.)
2. Docker Compose for local development (app + PostGIS)
3. Cloud deployment config:
   - [AWS]: ECS task definition + EventBridge rule
   - [GCP]: Cloud Run job + Cloud Scheduler
   - [Azure]: Container Instance + Logic App timer
4. Environment variable management (secrets for API keys, DB credentials)
5. Monitoring: CloudWatch / Cloud Logging alerts on failure
6. Cost estimate comment (approximate monthly cost for the chosen configuration)

Include a Makefile with targets: build, test, deploy-dev, deploy-prod, logs.
```

**Variables to customize:**
- Pipeline description and schedule
- Cloud provider
- Database configuration
- Monitoring preferences

**Expected output format:** Dockerfile, docker-compose.yml, cloud deployment configs, and Makefile.

---

### Prompt 3 — GitHub Actions Spatial Data Validation

**Context:** You want to add automated spatial data quality checks to your CI pipeline so that every pull request that modifies data files is validated before merge.

**Template:**

```
Write a GitHub Actions workflow (.github/workflows/validate-spatial-data.yml) that automatically
validates geospatial data files on every pull request.

Project context:
- Repository contains spatial data files in: [directory, e.g., "data/"]
- Formats: [GeoPackage / GeoJSON / GeoParquet / Shapefiles]
- Python version: [version, e.g., 3.11]
- Validation rules are defined in: [file, e.g., "validation/rules.yaml"]

Workflow behavior:
1. Trigger: on pull_request when files in [data directory] are modified
   - Use paths filter: paths: ['data/**', 'validation/**']
2. Set up environment:
   - Ubuntu latest
   - Python with spatial dependencies (GDAL, GEOS, PROJ via apt-get)
   - pip install geopandas pyarrow shapely fiona
3. Detect which data files changed:
   - Use git diff to find modified/added files matching spatial formats
   - Store the list as a step output
4. For each changed file, run validation:
   - Schema check: required columns exist with correct types
   - CRS check: matches expected EPSG code
   - Geometry validity: no null, no invalid, no empty geometries
   - Bounding box check: all features within expected study area
   - Attribute checks: defined in rules.yaml (not-null, value ranges, allowed values)
   - File size check: warn if file exceeds [threshold, e.g., 100 MB]
5. Generate a validation report:
   - Format as a Markdown table: file, check, status (PASS/FAIL), details
   - Post the report as a PR comment using actions/github-script or peter-evans/create-or-update-comment
   - If any check fails, set the workflow to failed (block merge)
6. Caching:
   - Cache pip dependencies for faster runs
   - Cache the GDAL installation if possible

Also provide:
- The validation/rules.yaml schema file with example rules
- The Python validation script (validation/validate.py) that reads rules.yaml and checks files
- A badge for the README: ![Data Validation](https://github.com/[owner]/[repo]/actions/workflows/validate-spatial-data.yml/badge.svg)

The script should exit 0 for all-pass, 1 for any-fail, and print a summary table to stdout.
```

**Variables to customize:**
- Data directory and formats
- Validation rules
- PR comment format
- File size thresholds

**Expected output format:** GitHub Actions YAML, validation script, rules YAML, and README badge.

---

### Workflow Diagram — CI/CD Pipeline

```
Prompt Chain: Full CI/CD Setup

  [Prompt 1: Local Dev]         [Prompt 2: CI Pipeline]        [Prompt 3: CD Pipeline]
  "Set up pre-commit hooks, --> "Write GitHub Actions for  --> "Add automated deployment
   Docker Compose, and          linting, testing, and           to cloud with scheduled
   local test runner"           coverage reporting"             execution"
         |                             |                                |
         v                             v                                v
  +------------------+        +-------------------+         +---------------------+
  | Local Setup      |        | CI Workflow        |         | CD Workflow         |
  | - pre-commit     |        | - Lint (ruff)     |         | - Build Docker      |
  | - docker-compose |        | - Test (pytest)   |         | - Push to registry  |
  | - Makefile       |        | - Coverage check  |         | - Deploy to cloud   |
  | - .env.example   |        | - Type check      |         | - Schedule trigger  |
  +------------------+        +-------------------+         | - Alert on failure  |
                                                            +---------------------+
```

---

## AI Coding Assistant Workflows

### Prompt 1 — Claude Code GIS Project Workflow

**Context:** You are using Claude Code (the CLI agent) or a similar agentic coding tool to work on a GIS project. These prompts are designed to be pasted as instructions or system prompts that guide the AI agent through multi-step spatial workflows.

**Template:**

```
You are a GIS developer working on a [project type, e.g., "spatial analysis pipeline" /
"QGIS plugin" / "web mapping application"]. The project is in [language: Python / JavaScript / TypeScript].

Project structure:
[describe or paste `tree` output of your project]

Your working environment:
- Python [version] with GeoPandas [version], Rasterio [version], [other key packages]
- OS: [macOS / Ubuntu / Windows]
- Spatial system packages: GDAL [version], PROJ [version], GEOS [version]

RULES FOR GIS CODE:
1. ALWAYS specify CRS explicitly — never assume EPSG:4326
2. ALWAYS use pathlib.Path for file paths, never string concatenation
3. ALWAYS close raster files after reading (use context managers: `with rasterio.open(...) as src:`)
4. ALWAYS handle geometry errors: check is_valid before operations, use make_valid() as fallback
5. ALWAYS use the pyogrio engine for GeoPandas I/O (faster than Fiona for most operations)
6. ALWAYS add type hints to function signatures
7. NEVER use shapefile as output format — prefer GeoPackage or GeoParquet
8. NEVER use pandas merge for spatial operations — use geopandas sjoin
9. When writing tests, use small fixture datasets (< 100 features) stored as GeoJSON in tests/fixtures/
10. Pin all dependencies in pyproject.toml with minimum version bounds

CURRENT TASK:
[describe what you want the agent to do, e.g.,
"Create a Python module that performs zonal statistics for multiple rasters across a set of polygons.
The module should:
1. Accept a directory of GeoTIFF rasters and a polygon GeoPackage
2. For each raster, calculate mean, median, std, min, max per polygon
3. Output a single GeoPackage with all statistics joined to the polygons
4. Include a CLI interface using Typer
5. Include pytest tests with a small fixture raster and polygon set"]

Start by reading the existing code, then implement the changes incrementally.
Run tests after each significant change.
```

**Variables to customize:**
- Project type and language
- Environment details (Python version, packages)
- GIS coding rules (adjust to your team's conventions)
- Current task description

**Expected output format:** A system prompt / instruction set for use with Claude Code, Cursor, or Windsurf.

---

### Prompt 2 — Cursor / Windsurf Multi-File GIS Workflow

**Context:** You are using Cursor or Windsurf (AI-enhanced IDE) and want to generate or refactor a multi-file GIS project using the Composer / multi-file edit feature.

**Template:**

```
I am working in [Cursor / Windsurf] on a GIS project. I need you to [action: create / refactor / extend]
the following across multiple files:

PROJECT CONTEXT:
@[reference key files the AI should read first, e.g.:]
@src/config.py — project configuration and CRS constants
@src/data_loader.py — existing data loading functions
@pyproject.toml — dependencies

TASK:
[describe the multi-file change, e.g.,
"Add a new spatial analysis module that:
1. @src/analysis/hotspot.py — implements Getis-Ord Gi* hotspot analysis using PySAL
2. @src/analysis/__init__.py — exports the new module
3. @src/cli.py — adds a 'hotspot' subcommand to the existing CLI
4. @tests/test_hotspot.py — pytest tests with a fixture GeoPackage of census tracts
5. @pyproject.toml — add esda and libpysal to dependencies"]

REQUIREMENTS:
- All spatial operations must use CRS from config.TARGET_CRS
- New modules must follow the existing code style (check @src/data_loader.py for patterns)
- Tests must use the existing conftest.py fixtures
- All functions must have Google-style docstrings
- Type hints required on all public functions

Generate all files in one pass. I will review the diff before accepting.
```

**Variables to customize:**
- IDE (Cursor / Windsurf)
- Referenced files
- Multi-file task description
- Coding standards and conventions

**Expected output format:** Multi-file generation instructions for use in Cursor Composer or Windsurf Cascade.

---

## Project Bootstrap

### Prompt 3 — Full GIS Project Bootstrap

**Context:** You are starting a new GIS project from scratch and want to set up the entire project structure, dependencies, configuration, and boilerplate in one shot.

**Template:**

```
Bootstrap a complete [project type] GIS project from scratch.

Project type: [choose one:
  - "Spatial analysis Python package" (library + CLI)
  - "Web mapping application" (MapLibre frontend + Python API backend)
  - "QGIS plugin with Processing provider"
  - "Geospatial ETL pipeline" (scheduled data pipeline)
  - "Remote sensing analysis project" (Jupyter notebooks + scripts)]

Project name: [name]
Description: [one sentence]
Python version: [3.11 / 3.12]
License: [MIT / Apache-2.0 / GPL-3.0]

Generate the COMPLETE project structure with all files:

```
[project-name]/
+-- pyproject.toml              # Dependencies, scripts, metadata (use hatch/setuptools)
+-- README.md                   # Project overview, installation, usage examples
+-- .gitignore                  # Python + GIS-specific ignores (.shp, .tif, .gpkg, __pycache__)
+-- .pre-commit-config.yaml     # ruff, mypy, trailing-whitespace
+-- .github/
|   +-- workflows/
|       +-- ci.yml              # Lint + test + coverage on push/PR
|       +-- validate-data.yml   # Spatial data validation (if data dir exists)
+-- src/[package_name]/
|   +-- __init__.py             # Version from importlib.metadata
|   +-- config.py               # CRS constants, paths, settings (use pydantic-settings)
|   +-- io.py                   # Spatial data I/O helpers (read/write GeoPackage, GeoParquet)
|   +-- processing.py           # Core spatial processing functions (placeholder)
|   +-- cli.py                  # Typer CLI with --verbose, --version flags
|   +-- py.typed                # PEP 561 marker
+-- tests/
|   +-- conftest.py             # Shared fixtures: sample GeoDataFrame, temp directories
|   +-- fixtures/
|   |   +-- sample_points.geojson   # 10 point features for testing
|   |   +-- sample_polygons.geojson # 5 polygon features for testing
|   +-- test_io.py              # Tests for I/O functions
|   +-- test_processing.py      # Tests for processing functions
+-- notebooks/                  # (if applicable)
|   +-- 01_explore.ipynb        # Data exploration template
+-- data/                       # .gitignore'd, with README explaining expected data
|   +-- README.md
+-- Dockerfile                  # Python + GDAL/GEOS/PROJ, multi-stage build
+-- docker-compose.yml          # App + PostGIS for local dev
+-- Makefile                    # install, test, lint, format, docker-build, clean
```

For each file, generate COMPLETE, working content — not placeholders or TODOs.

Key details:
- pyproject.toml: pin minimum versions for all spatial deps (geopandas>=0.14, rasterio>=1.3, shapely>=2.0, pyproj>=3.6, fiona>=1.9)
- .gitignore: include *.shp, *.shx, *.dbf, *.prj, *.cpg, *.tif, *.tiff, *.gpkg, *.parquet (data files should not be committed)
- config.py: define DEFAULT_CRS = "EPSG:4326", TARGET_CRS = "EPSG:[project CRS]", DATA_DIR, OUTPUT_DIR using pathlib
- io.py: implement read_vector(path) and write_vector(gdf, path) with format auto-detection and CRS validation
- cli.py: implement at least a `process` and `validate` command with real argument parsing
- conftest.py: create GeoDataFrame fixtures with 10 points and 5 polygons using Shapely
- Dockerfile: multi-stage build — builder stage installs GDAL from apt, final stage copies wheels
- Makefile: `make install` (pip install -e .[dev]), `make test` (pytest), `make lint` (ruff check + mypy)
- ci.yml: install GDAL via apt-get, run make lint && make test

The goal is: after running `git init && git add . && make install && make test`, everything works.
```

**Variables to customize:**
- Project type, name, description
- Python version and license
- Target CRS for the project
- Additional directories or files needed
- Database requirements (PostGIS, SQLite)

**Expected output format:** Complete directory tree with all files containing working, non-placeholder code.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
