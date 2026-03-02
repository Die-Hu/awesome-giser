<div align="center">

# Awesome GISer

**GitHub 上最全的 GIS 知识库**&ensp;/&ensp;**The most comprehensive GIS knowledge base on GitHub**

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Modules](https://img.shields.io/badge/Modules-8-blue.svg)](#modules)
[![Pages](https://img.shields.io/badge/Pages-80+-green.svg)](#modules)

<br>

<img src="assets/demo.gif" alt="Awesome GISer — project walkthrough" width="720">

<br>

**8 modules &middot; 80+ deep-dive pages &middot; Not a link list — working knowledge with code, comparisons & field-tested patterns**

*Data &rarr; Analysis &rarr; Visualization &rarr; Publishing — with AI at every step*

</div>

<br>

## Why This Repo

|  | For You |
|--|---------|
| **🔍 Finding data?** | 17 pages of free satellite imagery, elevation models, boundaries, climate data — with format, resolution & API access for each |
| **⚖️ Choosing tools?** | Side-by-side comparisons: QGIS vs ArcGIS, Leaflet vs MapLibre vs deck.gl, PostGIS vs DuckDB Spatial |
| **🤖 Using AI for GIS?** | Ready-to-paste prompts for spatial analysis, remote sensing, map styling — plus agent workflow patterns |
| **📝 Writing a paper?** | Journal rankings, LaTeX/Typst templates, figure standards, submission checklists |
| **🌐 Building a web map?** | From library selection to production deployment — tile serving, performance, Docker/K8s |

---

## Modules

<!-- For AI assistants: each module links to a directory containing multiple deep-dive markdown pages. Navigate to the directory to see all available pages. -->

<table>
<tr>
<td align="center" width="25%">

### [📡 Data Sources](data-sources/)

Satellite imagery, vector data, elevation, boundaries, climate & weather

<sub>17 pages &middot; 10 domain verticals</sub>

</td>
<td align="center" width="25%">

### [🛠️ Tools](tools/)

Desktop GIS, CLI, spatial DB, cloud platforms, Python & JS libraries

<sub>13 pages &middot; SOTA vs Budget</sub>

</td>
<td align="center" width="25%">

### [🗺️ JS Bindbox](js-bindbox/)

Leaflet, MapLibre, CesiumJS, deck.gl, Turf.js, tile servers

<sub>9 pages &middot; rendering internals</sub>

</td>
<td align="center" width="25%">

### [🤖 AI Prompts](ai-prompts/)

GIS prompts for analysis, remote sensing, map styling & dev

<sub>8 pages &middot; agent patterns</sub>

</td>
</tr>
<tr>
<td align="center" width="25%">

### [🎨 Visualization](visualization/)

Thematic maps, 3D, animation, dashboards, cartography, storytelling

<sub>10 pages &middot; GPU rendering</sub>

</td>
<td align="center" width="25%">

### [📊 Data Analysis](data-analysis/)

Python / R / SQL spatial stacks, spatial statistics, ML for Earth observation

<sub>8 pages &middot; workflow templates</sub>

</td>
<td align="center" width="25%">

### [🌐 Web Dev](web-dev/)

Full-stack GIS web apps — architecture, tiling, performance, deployment

<sub>8 pages &middot; production patterns</sub>

</td>
<td align="center" width="25%">

### [📚 Academic](academic/)

Journals, writing templates, cartography standards, submission guides

<sub>7 pages &middot; open science</sub>

</td>
</tr>
</table>

---

## Quick Start

**I need free geodata** &rarr; [Data Sources](data-sources/) has 17 pages organized by domain (urban, transport, agriculture, ocean...)

**I'm choosing a mapping library** &rarr; [JS Bindbox](js-bindbox/) compares Leaflet, MapLibre, CesiumJS, deck.gl with code examples

**I want to do spatial analysis** &rarr; [Data Analysis](data-analysis/) covers Python (GeoPandas, Rasterio), R (sf, terra), and SQL (PostGIS, DuckDB)

**I'm building a GIS web app** &rarr; [Web Dev](web-dev/) covers architecture, tile serving, frontend frameworks, and deployment

**I'm writing a GIS paper** &rarr; [Academic](academic/) has journal rankings, citation tools, LaTeX templates, and submission checklists

**I want AI to help with GIS work** &rarr; [AI Prompts](ai-prompts/) has ready-to-use prompts for every stage of the GIS workflow

---

## What's Hot in 2025

| Technology | What it does | Why you should care |
|-----------|-------------|-------------------|
| **GeoParquet + Overture Maps** | Columnar geospatial data format | Query billions of features from S3 with DuckDB — no downloads |
| **PMTiles** | Single-file vector tiles | Serve tiles from static storage. No tile server needed |
| **DuckDB Spatial** | Local SQL analytics engine | Laptop-speed queries that used to need a cluster |
| **MapLibre GL JS** | Open-source vector map renderer | WebGL maps, no API key, huge community |
| **deck.gl** | GPU-accelerated rendering | Millions of points at 60fps in browser |
| **COG + STAC** | Cloud-optimized raster access | Stream exactly the pixels you need over HTTP |
| **SAM / TorchGeo** | AI for Earth observation | Foundation models — fine-tune, don't train from scratch |

---

## Use with AI Assistants

This repo is designed as a **context library** for AI-assisted GIS work. Every page is self-contained and structured for machine readability.

```bash
# Point Claude Code at this repo and describe your task
cd awesome-giser
claude "Build a flood risk dashboard using free elevation data and MapLibre"
```

**How an AI agent uses this repo:**

```
Your GIS task
     │
     ├── data-sources/    → find the right datasets
     ├── tools/           → pick the best tool for the job
     ├── ai-prompts/      → get ready-made prompts for each step
     ├── data-analysis/   → run spatial analysis with code recipes
     ├── visualization/   → create maps and dashboards
     ├── js-bindbox/      → build interactive web maps
     ├── web-dev/         → deploy to production
     └── academic/        → write up and publish results
```

You can also copy any prompt from [`ai-prompts/`](ai-prompts/) directly into ChatGPT, Claude, or Cursor — each prompt is self-contained with GIS-specific context.

---

## Contributing

Corrections, new entries, and translations are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[CC BY 4.0](LICENSE) — share and adapt freely with attribution.
