# Visualization

> Techniques and tools for effective geospatial visualization -- from static thematic maps to interactive 3D dashboards, from scientific publication figures to billion-point GPU-accelerated renderings.

> **Quick Picks**
> - **SOTA Thematic**: [D3-geo](https://d3js.org) + [Observable Plot](https://observablehq.com/plot/) -- publication-quality maps with full control
> - **SOTA 3D**: [CesiumJS](https://cesium.com) + 3D Tiles 1.1 -- full globe with terrain, buildings, point clouds
> - **SOTA Dashboard**: [Streamlit](https://streamlit.io) + pydeck -- geo dashboard in <50 lines of Python
> - **SOTA Large-Scale**: [deck.gl](https://deck.gl) -- GPU-accelerated rendering for millions of features
> - **Fastest Setup**: [Kepler.gl](https://kepler.gl) -- drag-and-drop, zero code, export self-contained HTML

## Table of Contents

### Core Visualization
| Page | Lines | Description |
|------|-------|-------------|
| [Thematic Maps](thematic-maps.md) | 1,472 | Choropleth, heat maps, proportional symbols, flow maps, bivariate maps, dot density, cartograms, classification methods, color science, statistical mapping |
| [3D Visualization](3d-visualization.md) | 1,923 | CesiumJS + 3D Tiles, terrain rendering, urban digital twins, point clouds (Potree), volumetric data, photogrammetry, Gaussian splatting, game engines, WebXR |
| [Temporal Animation](temporal-animation.md) | 2,511 | Time-series animation, deck.gl TripsLayer, MapLibre temporal, D3 transitions, QGIS Temporal Controller, GEE timelapse, real-time streaming, video export |
| [Cartography & Design](cartography-design.md) | 939 | Design principles, Bertin's visual variables, typography, CJK labels, color theory (Oklab), accessibility (WCAG), print cartography, basemap design |

### Specialized Domains
| Page | Lines | Description |
|------|-------|-------------|
| [Scientific Visualization](scientific-visualization.md) | 1,595 | Statistics, ecology, biology, social science, climate, multi-dimensional data, publication-quality output with matplotlib/D3/Observable |
| [AI/ML Visualization](ai-ml-visualization.md) | 2,169 | SHAP/LIME explainability, neural network architecture viz, training dashboards, uncertainty quantification, foundation model interpretation |
| [Network & Graph Visualization](network-graph-visualization.md) | 1,968 | D3 force-directed, Cytoscape.js, Graphistry (GPU), transportation/utility/ecological/social networks, knowledge graphs, edge bundling |

### Applications & Infrastructure
| Page | Lines | Description |
|------|-------|-------------|
| [Dashboards & Interactive Apps](dashboards.md) | 2,086 | Streamlit, Dash, Panel, Observable Framework, Evidence.dev, Kepler.gl, Grafana, real-time ops, ML monitoring, deployment, design patterns |
| [Storytelling & Scrollytelling](storytelling-scrollytelling.md) | 2,572 | Scrollama.js, MapLibre + scroll, deck.gl narratives, StoryMapJS, ArcGIS StoryMaps, data journalism, multimedia, 甲方交付, accessibility |
| [Large-Scale Visualization](large-scale-visualization.md) | 3,280 | GPU rendering (WebGL/WebGPU), deck.gl deep dive, tiling strategies, spatial indexing (H3/S2), streaming, DuckDB WASM, memory management, benchmarking |

---

**Total: 20,500+ lines across 10 expert-level pages.**

## What Makes This Different

- **Cross-Disciplinary**: Not just maps -- covers biology, ecology, social science, statistics, neural networks, climate science
- **SOTA Tools**: Every section covers the current best-in-class tools with runnable code examples
- **Performance at Scale**: From 1K features to 1B+ points -- benchmarks, GPU techniques, memory management
- **Publication Quality**: Matplotlib, D3, Observable Plot output for journal papers and reports
- **Full Stack**: Python (Streamlit/Dash/Panel) + JavaScript (deck.gl/MapLibre/D3) + Desktop (QGIS/Blender)
- **Accessibility**: WCAG compliance, CVD-safe palettes, screen reader support, keyboard navigation
- **Production Ready**: Deployment guides, Docker recipes, CI/CD, authentication patterns
- **Dark Arts**: Every page includes expert tricks and unconventional techniques

## How to Choose

| Goal | Start Here |
|------|-----------|
| Choropleth / thematic map | [Thematic Maps](thematic-maps.md) → D3 or QGIS |
| 3D terrain or buildings | [3D Visualization](3d-visualization.md) → CesiumJS |
| Point cloud (LiDAR) | [3D Visualization](3d-visualization.md) → Potree |
| Animate data over time | [Temporal Animation](temporal-animation.md) → Kepler.gl or deck.gl |
| Color & typography | [Cartography & Design](cartography-design.md) → ColorBrewer + Noto |
| Scientific publication figure | [Scientific Visualization](scientific-visualization.md) → matplotlib or Observable Plot |
| Explain ML model predictions | [AI/ML Visualization](ai-ml-visualization.md) → SHAP + Folium |
| Network / graph analysis | [Network & Graph](network-graph-visualization.md) → D3 force or Graphistry |
| Interactive dashboard | [Dashboards](dashboards.md) → Streamlit (Python) or Observable (JS) |
| Scrollytelling narrative | [Storytelling](storytelling-scrollytelling.md) → Scrollama + MapLibre |
| Millions of points (GPU) | [Large-Scale](large-scale-visualization.md) → deck.gl |
| Real-time streaming | [Temporal Animation](temporal-animation.md) → WebSocket + MapLibre |
| Client deliverable / 甲方报告 | [Storytelling](storytelling-scrollytelling.md) → ArcGIS StoryMaps or custom |
| No-code quick viz | [Dashboards](dashboards.md) → Kepler.gl |

## Tool Quick Reference

| Tool | Type | Best For | Pages |
|------|------|----------|-------|
| **D3.js / D3-geo** | JS Library | Custom thematic maps, charts, networks | Thematic, Scientific, Network |
| **deck.gl** | JS Library | GPU-accelerated large-scale viz | Large-Scale, Temporal, 3D, Dashboards |
| **CesiumJS** | JS Library | 3D globe, terrain, 3D Tiles | 3D Visualization |
| **MapLibre GL JS** | JS Library | Vector tile maps, 3D terrain | Temporal, Storytelling, Large-Scale |
| **Kepler.gl** | No-Code | Quick exploratory geo-viz | Dashboards, Temporal, Thematic |
| **Observable Plot** | JS Library | Statistical + geo charts | Scientific, Thematic |
| **Streamlit** | Python | Rapid dashboard prototyping | Dashboards |
| **Potree** | JS Library | Billion-point LiDAR viewer | 3D Visualization |
| **QGIS** | Desktop | Print cartography, analysis | Cartography, Temporal, Thematic |
| **Graphistry** | Platform | Million-node GPU graph viz | Network & Graph |
| **Scrollama** | JS Library | Scroll-driven narratives | Storytelling |
| **SHAP** | Python | ML model explainability | AI/ML Visualization |

## Cross-References

- **[JS Bindbox](../js-bindbox/)** -- JavaScript mapping libraries (Leaflet, MapLibre, CesiumJS, deck.gl) that power these visualizations
- **[Tools](../tools/)** -- Desktop GIS, Python libraries, CLI tools, spatial databases
- **[Data Sources](../data-sources/)** -- Satellite imagery, vector data, elevation, climate data to visualize
- **[AI Prompts](../ai-prompts/)** -- LLM prompts for generating visualization code and map styling
