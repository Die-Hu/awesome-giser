# 🤖 AI Prompts for GIS

> Curated collection of AI prompts and workflows specifically designed for GIS professionals. This is what makes this repo unique — no other awesome list covers GIS-specific AI skills.

Whether you are cleaning messy spatial data, styling a publication-quality map, writing a QGIS plugin, or drafting a research paper, the prompt templates in this module give you a head start. Each prompt has been tested with mainstream LLMs and refined by practicing GISers.

---

## Why AI Prompts for GIS?

GIS work sits at the intersection of spatial data, domain expertise, and software engineering. Generic AI prompts often produce code that ignores coordinate reference systems, mishandles geometry types, or uses deprecated spatial libraries. **GIS-specific prompts solve this by baking domain knowledge into every request.**

What you get from purpose-built GIS prompts:

- **CRS-aware code** — prompts explicitly require EPSG codes, so the AI never silently drops or guesses your projection.
- **Format fluency** — prompts name the exact geospatial format (GeoPackage, GeoParquet, Cloud-Optimized GeoTIFF) instead of leaving it to chance.
- **Library precision** — prompts target the right tool for the job: GeoPandas for vector, Rasterio for raster, PySAL for spatial statistics, ee for Earth Engine.
- **Reproducibility** — each prompt includes version pins, parameter explanations, and validation steps so outputs can be audited.
- **Time savings** — a well-crafted GIS prompt replaces 30 minutes of StackOverflow / GIS StackExchange searching with a single copy-paste.

If you have ever asked ChatGPT to "write a spatial join" and received code that imports `shapely` but forgets `geopandas`, or asks for `fiona` without setting the driver, these prompts exist for you.

---

## Table of Contents

| Page | What You Will Find |
|---|---|
| [Data Analysis Prompts](data-analysis-prompts.md) | Cleaning, spatial joins, statistics, format conversion, GeoParquet |
| [Remote Sensing Prompts](remote-sensing-prompts.md) | Classification, SAM segmentation, indices, change detection, GEE code |
| [Map Styling Prompts](map-styling-prompts.md) | Color palettes, MapLibre style JSON, dark mode, SLD/QML, SVG icons |
| [Plugin Development Prompts](plugin-dev-prompts.md) | QGIS plugins & Processing algorithms, MapLibre controls, CLI tools |
| [Academic Writing Prompts](academic-writing-prompts.md) | Lit reviews, methodology, study area maps, LaTeX tables, reviewer responses |
| [Automation Workflows](automation-workflows.md) | Batch processing, ETL, Claude Code workflows, project bootstrap, CI/CD |

---

## Prompt Engineering Tips for GIS

Getting great results from AI on spatial tasks requires GIS-specific prompt hygiene. Follow these rules and your outputs will be dramatically better.

### 1. Always Specify the CRS

Bad: *"Reproject my data to a projected CRS"*
Good: *"Reproject from EPSG:4326 (WGS 84) to EPSG:32650 (UTM Zone 50N) using pyproj 3.6+"*

Without an explicit EPSG code, the model will guess — and guess wrong for your hemisphere.

### 2. Name the File Format and Driver

Bad: *"Load my spatial data"*
Good: *"Load a GeoPackage file (roads.gpkg, layer name: major_roads) using GeoPandas with the pyogrio engine"*

GIS has dozens of vector and raster formats. Be explicit about format, layer name, and the I/O driver.

### 3. Specify Desired Output Format

Tell the AI exactly what you want returned:
- "Return a standalone Python script with a `main()` function and CLI arguments"
- "Return a PostGIS SQL query with comments explaining each clause"
- "Return a MapLibre style JSON object, not JavaScript code"
- "Return LaTeX code for a table, not Markdown"

### 4. Include Domain Terminology

Use the precise GIS terms: *spatial join* (not "merge by location"), *zonal statistics* (not "aggregate raster by polygon"), *cloud masking* (not "remove white pixels"), *topology validation* (not "check shapes overlap").

The more domain-specific your language, the more the model activates the right knowledge.

### 5. Pin Library Versions

Bad: *"Use GDAL to convert"*
Good: *"Use GDAL 3.8+ (Python bindings via osgeo) or rasterio 1.3+ backed by GDAL 3.8"*

Spatial libraries change APIs between versions. Pin at least the major version.

### 6. Provide a Data Sample

Paste 3-5 rows of your attribute table or describe the band layout of your raster. This eliminates hallucinated column names and wrong band indices.

### 7. Ask for Validation

End every prompt with: *"Add a validation step that confirms the output CRS, feature count, and bounding box match expectations."* This catches silent failures that plague geospatial pipelines.

---

## Agent Team Patterns

Modern AI tools (Claude Code, Cursor Composer, ChatGPT Canvas) support multi-step agent workflows. You can **chain prompts across modules** to build complete GIS pipelines. Here are proven patterns:

### Land Cover Analysis Pipeline

```
# Agent Team Pattern: Land Cover Analysis
#
# Step 1: remote-sensing-prompts.md → "Sentinel-2 Cloud Masking" to download & preprocess imagery
# Step 2: remote-sensing-prompts.md → "Supervised Classification Workflow" to run classification
# Step 3: data-analysis-prompts.md  → "Validate Data Quality" to QA the classification output
# Step 4: map-styling-prompts.md    → "Accessibility-Compliant Color Scheme" to style the output
# Step 5: academic-writing-prompts.md → "Accuracy Assessment Reporting" to write up results
```

### Urban Planning Report Pipeline

```
# Agent Team Pattern: Urban Planning Report
#
# Step 1: data-analysis-prompts.md   → "GeoParquet with DuckDB" to query large urban datasets
# Step 2: data-analysis-prompts.md   → "Spatial Autocorrelation Report" to find hotspots
# Step 3: map-styling-prompts.md     → "Dark Mode Basemap" to create a presentation-ready base
# Step 4: map-styling-prompts.md     → "Data-Driven Layer Styling" to overlay analysis results
# Step 5: automation-workflows.md    → "Automated Map Report" to generate the PDF
```

### Plugin Development Pipeline

```
# Agent Team Pattern: QGIS Plugin from Scratch
#
# Step 1: plugin-dev-prompts.md      → "Plugin Scaffolding" to generate project structure
# Step 2: plugin-dev-prompts.md      → "QGIS Processing Algorithm" to implement core logic
# Step 3: automation-workflows.md    → "GitHub Actions for Geospatial Project" to set up CI
# Step 4: academic-writing-prompts.md → "Describe a Processing Workflow" to write documentation
```

### Disaster Response Pipeline

```
# Agent Team Pattern: Rapid Flood Mapping
#
# Step 1: remote-sensing-prompts.md  → "Automated Flood Mapping in GEE" to detect flood extent
# Step 2: data-analysis-prompts.md   → "Point-in-Polygon Aggregation" to count affected buildings
# Step 3: map-styling-prompts.md     → "Generate SVG Map Icons" to create damage markers
# Step 4: automation-workflows.md    → "Dashboard Data Preparation" to publish live dashboard
```

---

## AI Tool Compatibility Guide

Not all prompts work equally well across AI tools. Here is a guide based on testing.

| Prompt Category | Claude (Sonnet/Opus) | ChatGPT (GPT-4o) | GitHub Copilot | Cursor / Windsurf | Google Gemini |
|---|---|---|---|---|---|
| **Data Analysis** | Excellent — handles large data samples in context | Excellent — strong pandas/GeoPandas knowledge | Good for inline completion | Excellent with file references | Good |
| **Remote Sensing** | Very good — accurate rasterio/xarray code | Very good — sometimes hallucinates band indices | Moderate — needs more context | Good with project files open | Excellent for GEE (native) |
| **Map Styling** | Excellent — produces valid JSON consistently | Good — occasionally malforms style JSON | Poor — not ideal for JSON generation | Good for style files | Moderate |
| **Plugin Development** | Excellent — strong PyQGIS, understands Qt | Good — may miss QGIS-specific API nuances | Excellent for code completion | Excellent for full-file generation | Moderate |
| **Academic Writing** | Best-in-class — nuanced academic tone | Very good — sometimes over-formal | N/A | N/A | Good |
| **Automation / CI/CD** | Excellent — Docker, GitHub Actions, cloud configs | Very good | Good for YAML snippets | Excellent | Good |

### Tool-Specific Tips

- **Claude**: Attach your actual data files (CSV, GeoJSON, even small GeoTIFFs as base64). Claude's large context window lets you include full datasets. Use Claude Code or the API for multi-file projects.
- **ChatGPT**: Use the Code Interpreter for prompts that involve data analysis — it can run Python and show you the output inline. Attach screenshots of your map for styling prompts.
- **GitHub Copilot**: Convert the prompt templates into code comments at the top of your file. Copilot works best when it can see your existing imports and data structures.
- **Cursor / Windsurf**: Reference files in your project with `@filename` syntax. The full-codebase context makes these ideal for plugin development and multi-file workflows.
- **Google Gemini**: Best for Google Earth Engine prompts — Gemini has native awareness of the GEE API. Use it for GEE JavaScript code generation.

---

## Usage Guide

### Tips for Better Results

1. **Be specific about your data** — mention the CRS (e.g., EPSG:4326), file format, column names, and row count.
2. **State the output format** — "Return a Python script", "Return a PostGIS query", "Return a Mapbox style JSON".
3. **Iterate** — use follow-up prompts to refine. The templates here are starting points.
4. **Provide samples** — paste 5-10 rows of your attribute table so the model understands your schema.
5. **Mention constraints** — memory limits, coordinate system requirements, software version.
6. **Chain prompts** — use the Agent Team Patterns above to build multi-step workflows across modules.
7. **Validate outputs** — always ask for a validation step that checks CRS, feature count, and bounding box.

---

## Prompt Format Convention

Every prompt in this collection follows a consistent structure:

- **Context** — a short paragraph explaining **when** and **why** you would use this prompt.
- **Template** — the actual text you copy-paste into your AI tool. Variables you need to customize are wrapped in `[square brackets]`.
- **Variables to customize** — a checklist of what to fill in before pasting.
- **Expected output format** — what a good AI response looks like, so you can judge quality.

---

## Contributing

Have a GIS prompt that saved you hours of work? Open a PR! Please follow the format above and include at least one tested example output. Bonus points for prompts that work across multiple AI tools.

---

[Back to Main README](../README.md)
