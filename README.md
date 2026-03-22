<div align="center">

# Awesome GISer

**A curated GIS knowledge base — free geodata, tool comparisons, spatial analysis, web mapping & AI prompts**

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

<br>

<picture>
  <img src="assets/demo.gif" alt="Awesome GISer demo — browsing free geospatial data sources, GIS tool comparisons, web mapping libraries, and AI prompts for spatial analysis" width="720">
</picture>

<br>

**8 modules · 80+ deep-dive pages** — not a link list, but working knowledge with code, comparisons & field-tested patterns

</div>

<!-- AI file navigation index:
data-sources/ (17 pages): satellite-imagery.md, vector-data.md, elevation-terrain.md, administrative-boundaries.md, climate-weather.md, china-specific.md, ml-training-data.md, urban-planning-smart-cities.md, transportation-autonomous-driving.md, healthcare-public-health.md, agriculture-food-security.md, natural-disasters-emergency-response.md, ocean-maritime.md, energy-power-infrastructure.md, telecommunications-connectivity.md, logistics-delivery-urban-mobility.md, space-aerospace.md
tools/ (13 pages): desktop-gis.md, mobile-gis.md, cli-tools.md, spatial-databases.md, server-publishing.md, cloud-platforms.md, python-libraries.md, web-mapping.md, remote-sensing.md, ai-ml-geospatial.md, 3d-visualization.md, geocoding-routing.md, etl-data-engineering.md
js-bindbox/ (9 pages): 2d-mapping.md, 3d-mapping.md, spatial-analysis.md, data-formats-loading.md, tile-servers.md, framework-integration.md, charting-integration.md, performance-optimization.md, realtime-offline-advanced.md
ai-prompts/ (8 pages): data-analysis-prompts.md, remote-sensing-prompts.md, map-styling-prompts.md, plugin-dev-prompts.md, academic-writing-prompts.md, automation-workflows.md, client-delivery-prompts.md, ai-agent-patterns.md
visualization/ (10 pages): thematic-maps.md, 3d-visualization.md, temporal-animation.md, cartography-design.md, dashboards.md, storytelling-scrollytelling.md, large-scale-visualization.md, scientific-visualization.md, network-graph-visualization.md, ai-ml-visualization.md
data-analysis/ (8 pages): python-stack.md, r-stack.md, geospatial-sql.md, spatial-statistics.md, ml-gis.md, cloud-native-analytics.md, time-series-analysis.md, workflow-templates.md
web-dev/ (8 pages): fullstack-architecture.md, frontend-integration.md, backend-services.md, performance.md, deployment.md, realtime-and-collaboration.md, testing-and-security.md, pwa-and-offline.md
academic/ (7 pages): journals-list.md, writing-templates.md, cartography-standards.md, research-tools.md, submission-guide.md, academic-platforms.md, reproducibility-open-science.md
-->

---

## What You'll Find

| Your need | Where to go |
|-----------|------------|
| **Free geospatial data** — satellite imagery, elevation, boundaries, climate | [📡 Data Sources](data-sources/) — 17 pages across 10 domains |
| **GIS tool comparisons** — QGIS vs ArcGIS, PostGIS vs DuckDB Spatial | [🛠️ Tools](tools/) — 13 pages with SOTA vs Budget picks |
| **Web mapping libraries** — Leaflet, MapLibre, CesiumJS, deck.gl | [🗺️ JS Bindbox](js-bindbox/) — Quick decision guide with unified tool index |
| **Spatial analysis recipes** — Python, R, SQL with code examples | [📊 Data Analysis](data-analysis/) — 8 workflow-ready pages |
| **Map visualization** — thematic maps, 3D, dashboards, cartography | [🎨 Visualization](visualization/) — 10 pages up to GPU rendering |
| **GIS web app development** — architecture, tiling, deployment | [🌐 Web Dev](web-dev/) — Enterprise reference with tiered recommendations |
| **AI prompts for GIS** — analysis, remote sensing, map styling | [🤖 AI Prompts](ai-prompts/) — 8 pages of ready-to-use prompts |
| **Academic publishing** — journals, templates, submission guides | [📚 Academic](academic/) — 7 pages for researchers |

---

## Quick Start

**Need free data?** Start with [Data Sources](data-sources/) — pick a domain like [Urban Planning](data-sources/urban-planning-smart-cities.md), [Transportation](data-sources/transportation-autonomous-driving.md), or [Agriculture](data-sources/agriculture-food-security.md)

**Building something?** [JS Bindbox](js-bindbox/) for quick tool picks (small projects), then [Web Dev](web-dev/) for [architecture](web-dev/fullstack-architecture.md) and [deployment](web-dev/deployment.md) (large projects)

**Doing analysis?** [Python Stack](data-analysis/python-stack.md) · [R Stack](data-analysis/r-stack.md) · [Geospatial SQL](data-analysis/geospatial-sql.md) · [ML for GIS](data-analysis/ml-gis.md)

**Writing a paper?** [Journals](academic/journals-list.md) · [Templates](academic/writing-templates.md) · [Figure Standards](academic/cartography-standards.md) · [Submission Guide](academic/submission-guide.md)

---

## What's Hot in 2025

| Technology | Why it matters |
|-----------|---------------|
| **[GeoParquet + Overture Maps](data-sources/vector-data.md)** | Query billions of features from S3 with DuckDB — no downloads |
| **[PMTiles](js-bindbox/tile-servers.md)** | Serve vector tiles from static storage — no tile server needed |
| **[DuckDB Spatial](tools/spatial-databases.md)** | Run spatial SQL on your laptop at cluster speeds |
| **[MapLibre GL JS](js-bindbox/2d-mapping.md)** | Open-source WebGL vector maps — no API key required |
| **[deck.gl](js-bindbox/3d-mapping.md)** | GPU-accelerated rendering — millions of points at 60fps |
| **[COG + STAC](data-sources/satellite-imagery.md)** | Stream exactly the raster pixels you need over HTTP |
| **[SAM / TorchGeo](tools/ai-ml-geospatial.md)** | Foundation models for Earth observation — fine-tune, don't train |

---

## Use with AI Assistants

This repo works as a **context library** for AI-assisted GIS work. Every page is self-contained and structured for machine readability.

```bash
# Use with Claude Code
cd awesome-giser
claude "Build a flood risk dashboard using free elevation data and MapLibre"
```

Copy any prompt from [`ai-prompts/`](ai-prompts/) directly into ChatGPT, Claude, or Cursor — each is self-contained with GIS-specific context.

<details>
<summary>How an AI agent navigates this repo</summary>

```text
Your GIS task
     ├── data-sources/    → find the right datasets
     ├── tools/           → pick the best tool
     ├── ai-prompts/      → get prompts for each step
     ├── data-analysis/   → run spatial analysis
     ├── visualization/   → create maps & dashboards
     ├── js-bindbox/      → build web maps
     ├── web-dev/         → deploy to production
     └── academic/        → write up & publish
```

</details>

---

## Start Exploring

Every page is self-contained — pick any module and dive in. Star the repo to track updates, or check [What's Hot in 2025](#whats-hot-in-2025) for the latest tools reshaping GIS work.

---

## Contributing

Corrections, new entries, and translations are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[CC BY 4.0](LICENSE) — share and adapt freely with attribution.
