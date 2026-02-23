# 🌍 Awesome GISer

**The most comprehensive GIS encyclopedia on GitHub**

> *Data → Analysis → Visualization → Publishing — with AI at every step*

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Last Updated](https://img.shields.io/badge/Last_Updated-2025-blue.svg)](#sota-highlights-2025)

A curated knowledge base covering the **full GISer workflow**: from raw data acquisition, through spatial analysis and visualization, all the way to web publishing and academic output. Whether you are a student, developer, or researcher, this repository is your one-stop reference.

### What makes this different?

- **AI Prompts for GIS** — Ready-to-use prompts for data wrangling, remote sensing interpretation, cartographic styling, and spatial development.
- **SOTA vs Budget comparisons** — Every tool section highlights both the state-of-the-art option and a practical free/low-cost alternative so you can make informed decisions.
- **Full pipeline coverage** — Eight interconnected modules that mirror how real GIS work flows: Data → Analysis → Visualization → Publishing.
- **Agent-friendly design** — Every module can be orchestrated by AI coding assistants like Claude Code or Cursor for automated GIS workflows.

---

## Table of Contents

| # | Module | Directory | Description |
|---|--------|-----------|-------------|
| 1 | 📡 **Data Sources** | [`data-sources/`](data-sources/) | Free satellite imagery, vector datasets, elevation models, administrative boundaries, and climate data |
| 2 | 🛠️ **Tools** | [`tools/`](tools/) | Desktop GIS, CLI utilities, spatial databases, and cloud platforms |
| 3 | 🗺️ **JS Bindbox** | [`js-bindbox/`](js-bindbox/) | 2D/3D mapping libraries, spatial analysis in the browser, and tile servers |
| 4 | 🤖 **AI Prompts** | [`ai-prompts/`](ai-prompts/) | GIS-specific prompts for data analysis, remote sensing, styling, and development |
| 5 | 🎨 **Visualization** | [`visualization/`](visualization/) | Thematic maps, animations, 3D visualization, dashboards, and cartographic design |
| 6 | 📊 **Data Analysis** | [`data-analysis/`](data-analysis/) | Python/R spatial stacks, spatial statistics, and machine learning for GIS |
| 7 | 🌐 **Web Dev** | [`web-dev/`](web-dev/) | Full-stack GIS web architecture, frontend/backend frameworks, and performance tuning |
| 8 | 📚 **Academic** | [`academic/`](academic/) | Journals, writing templates, cartography standards, and submission guides |

---

## Agent-Powered GIS Pipeline

This is what sets Awesome GISer apart: the entire repo is structured as an **agent-orchestratable pipeline**.

```
┌─────────────┐    ┌──────────────┐    ┌────────────────┐    ┌──────────────┐
│  📡 Data     │───▶│  📊 Analysis  │───▶│  🎨 Visualize  │───▶│  📚 Publish   │
│  Acquisition │    │  & Processing │    │  & Web Dev     │    │  & Share     │
└─────────────┘    └──────────────┘    └────────────────┘    └──────────────┘
       │                  │                    │                     │
   🤖 AI Prompts     🤖 AI Prompts       🤖 AI Prompts        🤖 AI Prompts
   assist at          assist at            assist at             assist at
   every step         every step           every step            every step
```

Each module in this repo is designed to be **agent-friendly**. You can use Claude Code, Cursor, or any AI coding assistant to orchestrate multi-step GIS workflows by referencing the prompts and resources in this collection.

### Example: Agent Team Workflow

```
# Use Claude Code to build a land cover change analysis
#
# Agent Team Setup:
# ├── data-agent     → Uses data-sources/ to find Sentinel-2 imagery
# ├── analysis-agent → Uses data-analysis/ + ai-prompts/ to run classification
# ├── viz-agent      → Uses visualization/ + js-bindbox/ to create web map
# └── report-agent   → Uses academic/ + ai-prompts/ to write findings
```

Point your AI assistant at this repo and describe your GIS task. The assistant can navigate the modules, pull relevant prompts, reference tool comparisons, and chain together a complete workflow — from data discovery to published output.

---

## SOTA Highlights 2025

Cutting-edge tools and standards featured across the modules:

| Category | Highlights |
|----------|-----------|
| **Vector Data** | GeoParquet & Overture Maps — columnar geospatial data at planetary scale |
| **Tile Hosting** | PMTiles — serverless, single-file vector tile hosting with no backend |
| **Local Analytics** | DuckDB Spatial — blazing-fast SQL analytics on geospatial data, right on your laptop |
| **AI for Geospatial** | Segment Anything Model (SAM) — zero-shot segmentation for remote sensing imagery |
| **LLM Automation** | Claude / GPT for GIS — natural-language spatial queries, code generation, and analysis |
| **Web Mapping** | MapLibre + deck.gl — open-source, GPU-accelerated 2D/3D map rendering |
| **Cloud-Native** | Cloud-Optimized GeoTIFF (COG) + STAC — streaming raster access without full downloads |

---

## Quick Start

Pick a path that matches your role and jump right in:

### "I'm a **Student**"

Start by exploring where to find free data, learn the academic landscape, then dive into analysis:

1. [Data Sources](data-sources/) — Find free satellite imagery and open datasets for your coursework.
2. [Academic](academic/) — Discover key journals, citation tools, and writing templates.
3. [Data Analysis](data-analysis/) — Learn Python and R spatial stacks for your thesis work.

### "I'm a **Developer**"

Focus on the libraries, web frameworks, and tooling that power modern GIS applications:

1. [JS Bindbox](js-bindbox/) — Compare mapping libraries (Mapbox GL, Leaflet, Cesium, deck.gl, etc.).
2. [Web Dev](web-dev/) — Architect full-stack GIS apps with best practices for tiling, caching, and APIs.
3. [Tools](tools/) — Set up spatial databases, CLI pipelines, and cloud processing.

### "I'm a **Researcher**"

Leverage AI to accelerate your research, run advanced spatial analysis, and publish effectively:

1. [AI Prompts](ai-prompts/) — Use curated prompts to speed up data cleaning, interpretation, and writing.
2. [Data Analysis](data-analysis/) — Apply spatial statistics and ML models to your research questions.
3. [Academic](academic/) — Navigate journal selection, figure standards, and peer review.

---

## How to Use This Repo with AI Assistants

This repository is designed to serve as a **context library** for AI-assisted GIS work. Here is how to get the most out of it:

- **Copy any prompt from `ai-prompts/` into ChatGPT or Claude** — Each prompt is self-contained and ready to use. Paste it in, fill in the bracketed variables, and get results immediately.

- **Reference tool comparisons when asking AI to architect a solution** — When you ask "What is the best way to serve vector tiles?", point the AI at `tools/` and `js-bindbox/` so it can give you an answer grounded in the curated options here.

- **Use workflow templates as starting points for AI-assisted development** — The modules are structured as pipeline stages. Tell your AI assistant which stage you are at, and it can pull the right resources.

- **Point your Claude Code or Cursor at this repo as context for GIS projects** — Add this repository to your project context so your AI assistant understands the full landscape of data sources, tools, and best practices available to you.

```bash
# Example: open this repo as context in Claude Code
cd /path/to/awesome-giser
claude "Help me build a flood risk dashboard using free elevation data and MapLibre"
```

---

## Contributing

We welcome contributions of all kinds — new resource entries, corrections, translations, and entire new sections. Please read our [Contributing Guide](CONTRIBUTING.md) before submitting a pull request.

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](LICENSE).

You are free to share and adapt the material for any purpose, even commercially, as long as you give appropriate credit.
