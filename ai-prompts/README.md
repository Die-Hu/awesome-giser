# AI Prompts for GIS -- Strategic Prompt Engineering Guide

> A comprehensive framework for GIS professionals who use AI to deliver professional-grade spatial analysis, mapping, and data products. This guide encodes domain expertise into reusable prompt patterns so that every interaction with an LLM produces CRS-aware, version-pinned, validation-ready output.

---

## Vision

GIS professionals face a unique challenge with AI tools: generic prompts produce code that silently drops coordinate reference systems, uses deprecated spatial libraries, hallucinates band indices, and generates output in the wrong format. The cost of these failures is not just wasted time -- it is wrong analysis delivered to clients.

This section of awesome-giser exists to solve that problem. Every prompt template, pattern, and workflow here is designed with one goal: **help GIS professionals leverage AI to deliver professional results to clients (甲方) while reducing cost and increasing quality.**

The prompts in this collection encode three layers of expertise:

1. **GIS Domain Knowledge** -- correct spatial operations, CRS handling, format fluency, library APIs
2. **LLM Interaction Expertise** -- prompt structure, output control, cost optimization, context management
3. **Client Delivery Standards** -- reproducibility, validation, reporting, and documentation

When these three layers align, a single prompt can replace hours of manual work and produce output that passes professional review.

---

## Quick Navigation Matrix

Find the right page and prompt for your goal:

| Your Goal | Recommended Page | Key Prompt(s) | Estimated Tokens |
|---|---|---|---|
| Clean and join messy spatial data | [data-analysis-prompts.md](data-analysis-prompts.md) | Spatial Join with CRS Handling, GeoParquet Workflow | Low (Haiku) |
| Classify satellite imagery | [remote-sensing-prompts.md](remote-sensing-prompts.md) | Supervised Classification, SAM Segmentation | High (Opus) |
| Style a publication-quality map | [map-styling-prompts.md](map-styling-prompts.md) | Accessible Color Scheme, MapLibre Style JSON | Medium (Sonnet) |
| Build a QGIS plugin | [plugin-dev-prompts.md](plugin-dev-prompts.md) | Plugin Scaffolding, Processing Algorithm | High (Opus) |
| Write a methods section for a paper | [academic-writing-prompts.md](academic-writing-prompts.md) | Methodology Description, Accuracy Assessment | Medium (Sonnet) |
| Automate a multi-step pipeline | [automation-workflows.md](automation-workflows.md) | Claude Code Workflow, Batch Processing | Medium (Sonnet) |
| Prepare a deliverable package for a client | [client-delivery-prompts.md](client-delivery-prompts.md) | Report Generation, Map Atlas, QA Checklist | Medium (Sonnet) |
| Orchestrate multi-agent GIS workflows | [ai-agent-patterns.md](ai-agent-patterns.md) | Agent Pipeline, MCP Integration, Skill Chaining | High (Opus) |

---

## Table of Contents -- Prompt Files

| File | Description | Prompts | Lines |
|---|---|---|---|
| [data-analysis-prompts.md](data-analysis-prompts.md) | Cleaning, spatial joins, statistics, format conversion, GeoParquet, DuckDB spatial queries, topology validation | 10 | 451 |
| [remote-sensing-prompts.md](remote-sensing-prompts.md) | Supervised/unsupervised classification, SAM segmentation, spectral indices (NDVI, NDWI, NDBI), change detection, Google Earth Engine code generation, cloud masking | 12 | 620 |
| [map-styling-prompts.md](map-styling-prompts.md) | Color palettes (accessibility-compliant), MapLibre GL style JSON, dark mode basemaps, SLD/QML generation, SVG icon creation, data-driven styling | 10 | 554 |
| [plugin-dev-prompts.md](plugin-dev-prompts.md) | QGIS plugin scaffolding, Processing algorithms, MapLibre GL JS custom controls, CLI tool generation, PyQGIS scripting, Qt widget design | 10 | 574 |
| [academic-writing-prompts.md](academic-writing-prompts.md) | Literature reviews, methodology sections, study area descriptions, LaTeX table generation, reviewer response drafting, accuracy assessment reporting | 12 | 625 |
| [automation-workflows.md](automation-workflows.md) | Batch raster/vector processing, ETL pipelines, Claude Code project workflows, GitHub Actions CI/CD, Docker containerization, dashboard data preparation | 12 | 778 |
| [client-delivery-prompts.md](client-delivery-prompts.md) | Report generation for clients, map atlas production, quality assurance checklists, metadata packaging, project handoff documentation | NEW | -- |
| [ai-agent-patterns.md](ai-agent-patterns.md) | Multi-agent orchestration, MCP server integration, Skill-based prompt modules, agent pipeline design, cost-optimized agent routing | NEW | -- |

---

## Modern AI Toolchain (2025-2026)

The GIS-AI landscape has matured significantly. Understanding the available tools and their strengths is essential for choosing the right approach for each task.

### Claude Code

Claude Code is the primary recommended tool for complex GIS workflows that involve code generation, file manipulation, and multi-step reasoning.

**Key capabilities for GIS work:**

- **System Prompts and CLAUDE.md**: Define project-level instructions that persist across sessions. A GIS project's `CLAUDE.md` can encode CRS conventions, preferred libraries, output format standards, and validation requirements so that every prompt in the project inherits these constraints automatically.

```markdown
# Example CLAUDE.md for a GIS project
## Project Conventions
- Default CRS: EPSG:4326 for storage, EPSG:3857 for web display
- Vector format: GeoPackage (.gpkg) for interchange, GeoParquet for analysis
- Raster format: Cloud-Optimized GeoTIFF (COG) with LZW compression
- Python environment: Python 3.11+, GeoPandas 0.14+, Rasterio 1.3+, Shapely 2.0+
- Always validate output: check CRS, feature count, bounding box, and geometry validity
- Coordinate order: longitude, latitude (x, y) -- never lat/lon
```

- **Skills System**: Reusable `.md` files stored in `~/.claude/skills/` that encode domain expertise. A GIS professional can create skills for specific domains:

```
~/.claude/skills/
  gis-data-processing/
    SKILL.md              # Overview and activation rules
    rules/
      crs-handling.md     # CRS transformation rules and common pitfalls
      vector-formats.md   # Format-specific guidance (GPKG, GeoParquet, Shapefile)
      raster-processing.md  # COG creation, band math, resampling methods
      validation.md       # Standard validation checks for every output
  remote-sensing/
    SKILL.md
    rules/
      sentinel-2.md       # Band designations, resolution, processing levels
      landsat.md           # Collection 2 specifics, thermal band handling
      sar.md               # Sentinel-1 preprocessing, speckle filtering
```

- **Hooks**: Pre- and post-execution hooks that can enforce GIS-specific rules. For example, a post-commit hook that validates all output GeoPackages have a defined CRS, or a pre-prompt hook that injects the current project's CRS context.

- **Agent Workflows**: Claude Code can operate as an autonomous agent, reading files from your GIS project, running Python scripts, inspecting outputs, and iterating until the result passes validation. This is the most powerful mode for complex multi-step GIS analysis.

### Claude Team / Multi-Agent Orchestration

For complex GIS projects, multiple AI agents can be coordinated to handle different aspects of the workflow:

```
Agent 1 (Data Acquisition)    -- Downloads and preprocesses satellite imagery
    |
Agent 2 (Analysis)           -- Runs classification, computes indices
    |
Agent 3 (Cartography)        -- Styles maps, generates legends, creates layouts
    |
Agent 4 (Reporting)          -- Writes methodology, produces deliverable documents
    |
Agent 5 (QA)                 -- Validates every output against acceptance criteria
```

Each agent can use a different model tier based on task complexity. Data acquisition and simple format conversion use Haiku. Analysis and cartography use Sonnet. Complex reasoning and report generation use Opus.

### MCP Servers (Model Context Protocol)

MCP connects AI models directly to external data sources and tools. For GIS professionals, the relevant MCP servers include:

| MCP Server | Purpose | Example Use |
|---|---|---|
| **PostGIS MCP** | Query spatial databases directly from AI prompts | "Find all parcels within 500m of the river using ST_DWithin" |
| **STAC MCP** | Search and access satellite imagery catalogs | "Find Sentinel-2 scenes for this AOI with less than 10% cloud cover" |
| **Overpass MCP** | Query OpenStreetMap data in real-time | "Extract all building footprints in this bounding box" |
| **Filesystem MCP** | Read and write local GIS files | "Read the attribute table of parcels.gpkg and summarize column statistics" |
| **PostgreSQL MCP** | General database queries including spatial extensions | "Run this zonal statistics query against the land_use table" |
| **GitHub MCP** | Access GIS code repositories and issue trackers | "Review the latest changes to the processing pipeline" |

**MCP integration example:**

```
# With PostGIS MCP connected, the AI can directly query your spatial database:
"Using the PostGIS database, find all road segments within the flood_zone polygon
 where road_class IN ('primary', 'secondary'). Return the total length in kilometers
 grouped by road_class. Use ST_Intersection to clip roads to the flood zone boundary.
 Output CRS: EPSG:32650."
```

The AI executes the query, inspects the results, and can iterate on the SQL without the user needing to copy-paste between tools.

### Skills as Reusable Prompt Modules

Skills are markdown files that encode domain expertise and can be loaded on demand. They function as "prompt libraries" -- instead of writing the same GIS constraints into every prompt, you encode them once and reference them.

**Advantages for GIS work:**

- Encode CRS conventions, library preferences, and validation rules once
- Share standardized prompt modules across a team
- Version control your prompt expertise alongside your code
- Reduce token costs by eliminating repeated context

**Example skill structure for remote sensing:**

```markdown
# Remote Sensing Analysis Skill

## When to activate
This skill applies when the user asks about satellite imagery processing,
spectral analysis, land cover classification, or change detection.

## Rules
1. Always specify the satellite platform and processing level
2. Include cloud masking as a preprocessing step unless explicitly excluded
3. Use Rasterio 1.3+ for file I/O, never raw GDAL Python bindings
4. Validate output rasters: check nodata value, CRS, resolution, and extent
5. For Sentinel-2: use L2A (BOA reflectance), bands are 0-10000 scale
6. For Landsat: use Collection 2 Level-2, apply scale factors
```

### IDE-Specific Prompt Strategies

#### Cursor

Cursor's strength is full-project context. GIS-specific strategies:

- Use `@file` references to point at your data schema, existing processing scripts, or style files
- Place a `.cursorrules` file in your GIS project root with CRS conventions and library preferences
- Use Composer mode for multi-file refactoring (e.g., updating a processing pipeline across multiple modules)
- Reference your GeoPackage schema by opening the metadata table

#### Windsurf

Windsurf's Cascade feature excels at multi-step GIS workflows:

- Describe the full pipeline in natural language and let Cascade decompose it
- Use file references to ground the AI in your actual data schema
- Particularly effective for QGIS plugin development where multiple files must be coordinated

#### GitHub Copilot

Best for inline code completion during GIS development:

- Write detailed docstrings and comments that include CRS, format, and library version info
- Copilot uses your open files as context -- keep your data schema file and imports visible
- Less effective for generating complete workflows from scratch; better as a completion assistant

### AI Tool Comparison for GIS Tasks

| GIS Task | Best Tool | Why | Model Tier |
|---|---|---|---|
| Quick format conversion script | Copilot / Cursor | Inline completion with project context | Haiku |
| Complex spatial analysis pipeline | Claude Code (agent mode) | Multi-step reasoning, file access, iteration | Opus |
| QGIS plugin development | Cursor / Windsurf | Full-project context, multi-file editing | Sonnet |
| Satellite imagery processing | Claude Code + MCP | Direct access to STAC catalogs and databases | Opus |
| Map style JSON generation | Claude (API or Code) | Consistent JSON output, large context for examples | Sonnet |
| PostGIS query optimization | Claude Code + PostGIS MCP | Direct database access for testing queries | Sonnet |
| Academic paper writing | Claude (conversation) | Nuanced tone, long-form coherence | Sonnet/Opus |
| Batch data cleaning | Claude Code (agent) | File system access, iterative validation | Sonnet |
| GEE JavaScript code | Gemini | Native GEE API awareness | -- |
| Quick one-off data questions | ChatGPT + Code Interpreter | Immediate execution and visualization | -- |

---

## Prompt Engineering Principles for GIS

These are advanced, GIS-specific principles. They go beyond generic "be specific" advice and address the failure modes unique to spatial computing.

### Principle 1 -- CRS-First Rule

Every prompt that involves spatial data must specify the coordinate reference system before anything else. This is the single most common source of errors in AI-generated GIS code.

**Pattern:**

```
Working CRS: EPSG:32650 (UTM Zone 50N)
Input data CRS: EPSG:4326 (WGS 84 geographic)
Output CRS: EPSG:32650
Reproject input to working CRS before any distance or area calculations.
```

**Why this matters:** LLMs default to EPSG:4326 for everything. If you compute distances in EPSG:4326, you get degrees, not meters. If you compute areas, you get square degrees. These are not useful values and will produce wrong results that look plausible.

**Common failures without CRS specification:**

- Buffer operations in geographic CRS producing elliptical buffers
- Area calculations returning values 12 orders of magnitude wrong
- Nearest-neighbor searches using Euclidean distance on geographic coordinates
- Spatial joins failing silently because inputs are in different CRS

### Principle 2 -- Library Version Pinning

Spatial libraries have significant API changes between versions. Shapely 2.0 changed the entire geometry creation API. GeoPandas 0.14 deprecated `append()`. GDAL 3.x changed axis order conventions.

**Pattern:**

```
Environment:
- Python 3.11+
- GeoPandas 0.14+ (with pyogrio engine, NOT fiona)
- Shapely 2.0+ (use shapely.geometry, NOT shapely.ops for new API)
- Rasterio 1.3+ (backed by GDAL 3.8+)
- pyproj 3.6+
- DuckDB 0.10+ with spatial extension
```

**Critical version boundaries to know:**

| Library | Version Boundary | What Changed |
|---|---|---|
| Shapely | 1.x to 2.0 | Geometry creation, operations API, performance model |
| GeoPandas | 0.13 to 0.14 | pyogrio default engine, deprecation of append |
| GDAL | 2.x to 3.x | Axis order (lat/lon vs lon/lat), new drivers |
| pyproj | 2.x to 3.x | Transformer API, always_xy parameter |
| Rasterio | 1.2 to 1.3 | COG driver improvements, new resampling methods |
| DuckDB Spatial | 0.9 to 0.10 | ST function naming, GeoParquet native support |

### Principle 3 -- Validation Loop

Every prompt should include an explicit validation step. GIS pipelines fail silently more often than they fail loudly. A spatial join that drops 40% of features due to a CRS mismatch will not raise an error -- it will just return fewer features.

**Pattern:**

```
After processing, validate the output:
1. Confirm output CRS matches EPSG:32650
2. Report feature count (input vs output -- flag if difference > 5%)
3. Print bounding box and confirm it falls within expected extent
4. Check for null geometries and report count
5. Validate geometry types (all Polygon, no unexpected MultiPolygon)
6. Compute basic statistics on key columns and print summary
7. If any validation fails, stop and report the issue before writing output
```

### Principle 4 -- Data Schema Declaration

Before asking the AI to process your data, describe its schema explicitly. This eliminates hallucinated column names, wrong data types, and incorrect geometry assumptions.

**Pattern:**

```
Input dataset: buildings.gpkg (layer: residential)
Schema:
  - id: integer (primary key)
  - address: string
  - floors: integer (1-50)
  - area_sqm: float (building footprint area in square meters)
  - built_year: integer (1900-2025)
  - geometry: MultiPolygon (EPSG:4326)
  - Row count: approximately 45,000
  - Null values: address has ~5% nulls, built_year has ~15% nulls

Sample rows:
| id | address | floors | area_sqm | built_year |
|----|---------|--------|----------|------------|
| 1  | 123 Main St | 3 | 450.2 | 1995 |
| 2  | NULL | 12 | 1200.8 | 2010 |
| 3  | 45 Park Ave | 1 | 85.3 | NULL |
```

### Principle 5 -- Output Format Specification

Explicitly state every aspect of the desired output. GIS has dozens of formats, each with format-specific options that matter.

**Pattern:**

```
Output requirements:
- Format: GeoPackage (.gpkg)
- Layer name: analysis_result
- CRS: EPSG:32650
- Encoding: UTF-8
- Include these columns: [list]
- Geometry type: Polygon (explode MultiPolygon to Polygon)
- Coordinate precision: 6 decimal places
- Nodata value (for raster): -9999
- Compression (for raster): LZW
- File naming: {region}_{date}_{analysis_type}.gpkg
```

### Principle 6 -- Error Budget

Tell the AI what level of accuracy is acceptable. This prevents over-engineering simple tasks and ensures critical tasks receive appropriate rigor.

**Pattern:**

```
Accuracy requirements:
- Positional accuracy: +/- 10 meters acceptable (this is a regional overview, not cadastral)
- Classification accuracy: target 85% overall accuracy, 80% per-class minimum
- Area calculation tolerance: +/- 2% compared to reference
- Temporal resolution: monthly composites are sufficient, daily not needed
- Edge cases: parcels smaller than 100 sqm can be excluded
```

### Principle 7 -- Cost-Aware Prompting

Different GIS tasks require different levels of AI reasoning. Using Opus for a simple format conversion wastes money. Using Haiku for a complex spatial analysis wastes time debugging wrong output.

**Task-to-model mapping:**

| Task Complexity | Model | Approximate Cost per Prompt | Examples |
|---|---|---|---|
| Simple | Haiku | $0.001 - $0.01 | Format conversion, CRS reprojection, simple queries |
| Standard | Sonnet | $0.01 - $0.10 | Spatial joins, styling, standard analysis workflows |
| Complex | Opus | $0.10 - $1.00 | Multi-step analysis, plugin architecture, novel algorithms |
| Specialized | Domain-specific | Varies | GEE code (Gemini), inline completion (Copilot) |

**Cost reduction strategies:**

- Use CLAUDE.md and Skills to avoid repeating context in every prompt
- Use MCP to let the AI read data directly instead of pasting schemas
- Batch simple operations into a single prompt instead of multiple calls
- Cache intermediate results so failed prompts do not require full re-execution

### Principle 8 -- Prompt Chaining

Complex GIS workflows should be decomposed into sequential prompts with explicit handoff points. Each prompt in the chain produces a defined output that becomes the input for the next prompt.

**Pattern:**

```
Chain: Urban Heat Island Analysis

Step 1 (Sonnet): Data Acquisition
  Input: AOI bounding box, date range
  Output: Downloaded Landsat 8/9 scenes, cloud-masked, stacked
  Handoff: file paths and metadata JSON

Step 2 (Haiku): LST Calculation
  Input: Stacked thermal bands from Step 1
  Output: Land Surface Temperature raster (Celsius, COG format)
  Handoff: LST raster path, statistics summary

Step 3 (Sonnet): Spatial Analysis
  Input: LST raster + urban land use polygons
  Output: Zonal statistics (mean LST per land use class), hotspot identification
  Handoff: GeoPackage with analysis results

Step 4 (Opus): Interpretation and Report
  Input: Zonal statistics + hotspot map + city context
  Output: Written analysis with recommendations, styled maps
  Handoff: Final deliverable package
```

### Principle 9 -- Context Window Management

GIS datasets are often too large to paste into a prompt. Use these strategies to feed spatial data into AI context efficiently:

**Strategy 1 -- Schema + Sample:**
Paste the schema definition and 5-10 representative rows instead of the full dataset. This uses minimal tokens while giving the AI enough information to generate correct code.

**Strategy 2 -- Statistics Summary:**
Run `gdf.describe()`, `gdf.dtypes`, and `gdf.total_bounds` locally, then paste the summary. The AI can write processing code from statistical profiles.

**Strategy 3 -- MCP Direct Access:**
Connect the AI to your data via MCP. The AI can run `SELECT * FROM table LIMIT 5` itself, inspect the schema, and iterate on queries without manual copy-paste.

**Strategy 4 -- Progressive Disclosure:**
Start with a high-level description. Let the AI ask clarifying questions. Provide specifics only for the aspects the AI needs. This minimizes token usage while maximizing accuracy.

**Strategy 5 -- File Reference (IDE tools):**
In Cursor or Windsurf, use `@filename` to reference your data files. The tool extracts relevant context automatically.

---

## Multi-Role Prompt Patterns

The most effective GIS prompts incorporate multiple perspectives. Each role catches errors that other roles miss. When constructing a prompt for a complex task, consider which roles are relevant and include their requirements explicitly.

### GIS Expert Role

The GIS Expert ensures technical accuracy of spatial operations.

**What this role checks:**

- Correct spatial operation (intersection vs union vs difference)
- Appropriate algorithm choice (e.g., KD-tree for nearest neighbor, R-tree for spatial index)
- Proper handling of edge cases (antimeridian, poles, self-intersecting geometries)
- Correct CRS usage and transformation
- Appropriate resolution and scale for the analysis

**Prompt injection pattern:**

```
As a GIS expert, ensure that:
- All distance calculations use a projected CRS appropriate for the study area
- Spatial indices are created before join operations on datasets > 10,000 features
- Geometry validity is checked with ST_IsValid / shapely.validation before operations
- The algorithm handles MultiGeometry inputs correctly
- Topology is preserved (no slivers, gaps, or overlaps in output polygons)
```

### LLM Expert Role

The LLM Expert optimizes the prompt structure for better AI output.

**What this role checks:**

- Prompt is structured for clear parsing (constraints before request, not after)
- Output format is unambiguous (JSON schema, code with type hints, structured markdown)
- Few-shot examples are included for complex output formats
- Token budget is appropriate for the task
- Prompt avoids known failure modes (e.g., asking for too many things at once)

**Prompt injection pattern:**

```
Structure your response as follows:
1. First, state your understanding of the task in one sentence
2. Then provide the complete code in a single fenced code block
3. After the code, list all assumptions you made
4. Finally, provide the validation commands to verify the output
Do not include explanatory text between code sections.
```

### Client (甲方) Role

The Client Role ensures the output meets business and deliverable requirements.

**What this role checks:**

- Output is in the format the client expects (PDF report, web map, data package)
- Visualizations use the client's brand colors and style guidelines
- Metadata is complete (data sources, processing date, analyst name, methodology)
- Results are presented in business terms, not technical jargon
- Deliverable includes everything needed for the client to use the results independently

**Prompt injection pattern:**

```
The deliverable must include:
- Executive summary (one page, non-technical language)
- Map outputs at A3 print resolution (300 DPI)
- Data dictionary for all delivered datasets
- Methodology description suitable for a non-GIS audience
- All source data citations with access dates
- Processing log showing software versions and parameters used
```

### QA Role

The QA Role defines validation rules and acceptance criteria.

**What this role checks:**

- Output passes all validation checks (CRS, feature count, extent, geometry validity)
- Edge cases are handled (empty geometries, null values, out-of-range coordinates)
- Results are reproducible (same input produces same output)
- Accuracy meets the specified error budget
- No data loss occurred during processing (input vs output feature counts)

**Prompt injection pattern:**

```
Quality assurance requirements:
- Assert output feature count equals input feature count (or explain any difference)
- Assert all geometries are valid (run ST_MakeValid if needed, report count of repaired)
- Assert output bounding box is within expected extent +/- 1%
- Assert no coordinate values outside valid range for the CRS
- Assert all required columns are present and have correct data types
- Log all warnings and write them to a QA report file
```

### Human Operator Role

The Human Operator Role defines customization points and ergonomics.

**What this role checks:**

- Variables that need human input are clearly marked
- The prompt is parameterizable for different study areas, dates, or datasets
- Progress reporting is included for long-running operations
- Error messages are actionable (tell the operator what to do, not just what went wrong)
- The workflow can be resumed from intermediate steps if interrupted

**Prompt injection pattern:**

```
Mark all user-customizable values with [BRACKETS] and list them at the top:
- [AOI_PATH]: path to the area of interest polygon
- [DATE_START]: start date in YYYY-MM-DD format
- [DATE_END]: end date in YYYY-MM-DD format
- [OUTPUT_DIR]: directory for output files
- [TARGET_CRS]: EPSG code for output coordinate reference system

Print progress messages at each major step.
If any step fails, save intermediate results and print a message explaining
how to resume from that step.
```

### Combining Roles in a Single Prompt

For critical deliverables, combine multiple roles:

```
You are working on a flood risk assessment for a municipal client (甲方).

[GIS Expert] Use Sentinel-1 SAR data (VV polarization, descending orbit) to detect
surface water extent. Apply Lee speckle filter (7x7 window) before thresholding.
Working CRS: EPSG:32650.

[LLM Expert] Return a single Python script with clear section comments. Use type hints.
Include a main() function with argparse for all parameters.

[Client/甲方] The output must include: (1) flood extent polygon (GeoPackage),
(2) affected buildings count by administrative district, (3) one-page summary
with map suitable for a non-technical government audience.

[QA] Validate that the flood extent polygon has no self-intersections, all
building counts are non-negative, and the total affected area is plausible
(flag if > 50% of the study area is classified as flooded).

[Operator] Parameterize for different SAR scenes and study areas. Print estimated
processing time at start. Save intermediate SAR preprocessing results so the
classification threshold can be adjusted without reprocessing.
```

---

## Integration with awesome-giser Library

The prompts in this section are designed to work with the resources cataloged elsewhere in awesome-giser. Cross-referencing these resources in your prompts provides the AI with richer context and produces more accurate, practical output.

### Cross-Reference Table

| Prompt Category | Relevant awesome-giser Pages | How to Reference in Prompts |
|---|---|---|
| Data Acquisition | [satellite-imagery.md](../data-sources/satellite-imagery.md), [vector-data.md](../data-sources/vector-data.md), [elevation-terrain.md](../data-sources/elevation-terrain.md) | "Download Sentinel-2 L2A data from [source] as described in the satellite imagery catalog" |
| Data Processing | [python-libraries.md](../tools/python-libraries.md), [cli-tools.md](../tools/cli-tools.md), [etl-data-engineering.md](../tools/etl-data-engineering.md) | "Process using GDAL CLI tools; refer to the CLI tools reference for flags and options" |
| Spatial Databases | [spatial-databases.md](../tools/spatial-databases.md), [cloud-platforms.md](../tools/cloud-platforms.md) | "Store results in PostGIS; use conventions from the spatial databases guide" |
| Web Visualization | [2d-mapping.md](../js-bindbox/2d-mapping.md), [3d-mapping.md](../js-bindbox/3d-mapping.md) | "Visualize using MapLibre GL JS with the patterns from the 2D mapping reference" |
| Remote Sensing | [remote-sensing.md](../tools/remote-sensing.md), [ai-ml-geospatial.md](../tools/ai-ml-geospatial.md) | "Apply classification using tools from the remote sensing toolkit catalog" |
| Performance | [performance-optimization.md](../js-bindbox/performance-optimization.md), [tile-servers.md](../js-bindbox/tile-servers.md) | "Optimize tile rendering using strategies from the performance guide" |
| Academic | [academic/](../academic/) | "Follow citation conventions and methodology standards from the academic section" |
| China-Specific | [china-specific.md](../data-sources/china-specific.md) | "Use GCJ-02/BD-09 coordinate handling as documented in the China-specific data guide" |
| 3D Content | [3d-visualization.md](../tools/3d-visualization.md), [3d-mapping.md](../js-bindbox/3d-mapping.md) | "Render 3D buildings using CesiumJS patterns from the 3D mapping reference" |

### Example: Cross-Referenced Prompt

```
I need to create a web map showing urban heat islands in Shenzhen.

Data source: Use Landsat 8/9 thermal bands. See the satellite imagery catalog
(../data-sources/satellite-imagery.md) for USGS EarthExplorer access details.

Processing: Calculate Land Surface Temperature using the split-window algorithm.
Use rasterio 1.3+ for raster I/O (see ../tools/python-libraries.md for installation).

Visualization: Display the LST overlay on a MapLibre GL JS basemap.
Use the dark mode basemap pattern from map-styling-prompts.md.
Follow the performance optimization patterns from ../js-bindbox/performance-optimization.md
for large raster tile serving.

China-specific: Apply GCJ-02 offset correction for basemap alignment.
See ../data-sources/china-specific.md for the coordinate transformation approach.

Output CRS: EPSG:4490 (CGCS2000) for the analysis, EPSG:3857 for web display.
```

---

## Prompt Template Anatomy

Every prompt in this collection follows a standardized structure. This consistency makes prompts predictable, maintainable, and easy to customize.

### Standard Structure

```markdown
## Prompt N -- [Descriptive Name]

### Scenario (实际场景)
A 2-3 sentence description of when and why you would use this prompt.
Includes the typical user, the problem they face, and the expected outcome.

### Roles Involved (角色)
Which of the multi-role perspectives are active in this prompt:
- [ ] GIS Expert
- [ ] LLM Expert
- [ ] Client/甲方
- [ ] QA
- [ ] Human Operator

### Prompt Template
The actual text to paste into your AI tool. Variables are marked with
[SQUARE_BRACKETS] and listed in the customization section below.

### Variables to Customize
| Variable | Description | Example Value |
|---|---|---|
| [AOI] | Area of interest | "Shenzhen city boundary" |
| [CRS] | Target CRS | "EPSG:32650" |

### Expected Output
Description of what a correct AI response looks like.
Includes format, structure, and key elements to verify.

### Validation Checklist
- [ ] Output CRS matches specification
- [ ] Feature count is within expected range
- [ ] All required columns are present
- [ ] Geometries are valid
- [ ] No unexpected null values

### Cost Optimization Notes
- Recommended model tier: Sonnet
- Estimated tokens: 2,000 input / 4,000 output
- Can be downgraded to Haiku if: [conditions]
- Consider caching: [what to cache]

### Related awesome-giser Pages
- [page-name.md](../path/to/page.md) -- how it relates

### Extensibility Notes
How to modify this prompt for related use cases.
Common variations and their adjustments.
```

### Why This Structure Matters

**Scenario** prevents misuse. A prompt designed for regional-scale analysis should not be applied to cadastral-precision work without modification.

**Roles** make the quality dimensions explicit. A prompt with only the GIS Expert role active will produce technically correct but potentially undeliverable output. Adding the Client role ensures the output meets business requirements.

**Variables** make prompts reusable. Instead of rewriting a prompt for each project, swap the variables.

**Validation Checklist** catches failures before they reach the client. Every prompt should produce output that can be verified against objective criteria.

**Cost Notes** prevent overspending. A team running 100 prompts per day needs to be intentional about model selection.

**Extensibility Notes** teach the user how to adapt prompts, building their own expertise over time.

---

## Cost Optimization Guide

AI-assisted GIS work can be cost-effective or expensive depending on how prompts are structured and which models are used. This section provides concrete guidance for optimizing cost without sacrificing quality.

### Token Cost Comparison by Model (2025 Pricing)

| Model | Input (per 1M tokens) | Output (per 1M tokens) | Best For |
|---|---|---|---|
| Claude Haiku | ~$0.25 | ~$1.25 | Format conversion, simple queries, data validation |
| Claude Sonnet | ~$3.00 | ~$15.00 | Standard analysis, styling, code generation |
| Claude Opus | ~$15.00 | ~$75.00 | Complex reasoning, multi-step analysis, architecture |
| GPT-4o | ~$2.50 | ~$10.00 | General analysis, Code Interpreter for visualization |
| GPT-4o mini | ~$0.15 | ~$0.60 | Simple tasks, high-volume batch operations |
| Gemini 2.5 Pro | ~$1.25 | ~$10.00 | Long-context GEE code, large dataset analysis |

*Prices are approximate and subject to change. Check provider pricing pages for current rates.*

### Model Selection Decision Tree

```
Is the task a simple transformation or format conversion?
  YES --> Haiku / GPT-4o mini
  NO  --> Does the task require multi-step reasoning or novel algorithm design?
            YES --> Opus (justify the cost with task value)
            NO  --> Is the output a structured format (JSON, SQL, style spec)?
                      YES --> Sonnet (reliable structured output)
                      NO  --> Is it GEE JavaScript code?
                                YES --> Gemini (native GEE awareness)
                                NO  --> Sonnet (general purpose)
```

### Batch vs Interactive Strategies

**Batch mode** (lower cost, higher throughput):

- Collect multiple similar operations and submit as a single prompt
- Use Claude Batch API for processing hundreds of similar spatial queries
- Example: "For each of the following 50 administrative districts, compute the zonal mean NDVI from the attached raster"
- Cost savings: 30-50% compared to individual prompts due to shared context

**Interactive mode** (higher cost, better for exploration):

- Use for iterative analysis where each step depends on the previous result
- Appropriate for complex spatial analysis where the approach may need adjustment
- Use Claude Code agent mode to let the AI iterate autonomously
- Cost management: set a token budget and let the agent optimize within it

### How Skills and MCP Reduce Token Costs

**Without Skills (repeated context):**

Each prompt must include the full project context: CRS conventions, library versions, validation requirements, output format specifications. For a typical GIS project, this is 500-1,000 tokens of repeated context per prompt. Over 100 prompts, that is 50,000-100,000 wasted tokens.

**With Skills (encoded context):**

The Skill file is loaded once per session. All prompts in that session inherit the context automatically. The repeated context cost drops to near zero.

**Without MCP (manual data pasting):**

The user must manually run queries, copy results, and paste them into the prompt. This adds tokens for the pasted data and often requires multiple round-trips as the AI requests additional information.

**With MCP (direct data access):**

The AI queries the database or file system directly, retrieving only the data it needs. No manual copy-paste, no wasted tokens on unnecessary data, and the AI can iterate on queries without user intervention.

**Cost savings estimate for a typical GIS project:**

| Approach | Tokens per Task (avg) | Tasks per Day | Daily Cost (Sonnet) |
|---|---|---|---|
| No Skills, no MCP | 8,000 | 50 | ~$3.60 |
| With Skills | 5,000 | 50 | ~$2.25 |
| With Skills + MCP | 3,500 | 50 | ~$1.58 |

Annual savings for a team of 5: approximately $3,000-$5,000 depending on usage patterns.

---

## Agent Team Patterns

These patterns demonstrate how to chain prompts across modules to build complete GIS pipelines. Each pattern specifies the model tier, input/output handoffs, and validation gates.

### Land Cover Analysis Pipeline

```
Agent Team: Land Cover Analysis

Step 1: Data Acquisition (Sonnet)
  Source: remote-sensing-prompts.md -- "Sentinel-2 Cloud Masking"
  Input: AOI polygon, date range
  Output: Cloud-free Sentinel-2 composite (COG)
  Validate: Band count, spatial extent, cloud percentage < threshold

Step 2: Classification (Opus)
  Source: remote-sensing-prompts.md -- "Supervised Classification Workflow"
  Input: Cloud-free composite + training samples
  Output: Classified raster (COG) + confusion matrix
  Validate: Class count matches training data, no unclassified pixels in AOI

Step 3: Quality Assessment (Sonnet)
  Source: data-analysis-prompts.md -- "Validate Data Quality"
  Input: Classified raster + validation samples
  Output: Accuracy assessment report (overall, per-class, kappa)
  Validate: Overall accuracy >= 85%, per-class >= 80%

Step 4: Cartography (Sonnet)
  Source: map-styling-prompts.md -- "Accessibility-Compliant Color Scheme"
  Input: Classified raster + class names
  Output: Styled map (PNG, 300 DPI) + legend
  Validate: Color contrast meets WCAG AA, legend matches classes

Step 5: Reporting (Opus)
  Source: academic-writing-prompts.md -- "Accuracy Assessment Reporting"
  Input: All outputs from Steps 1-4
  Output: Methods section + results section + figures
  Validate: All figures referenced in text, statistics match computed values
```

### Urban Planning Report Pipeline

```
Agent Team: Urban Planning Report

Step 1: Data Query (Haiku)
  Source: data-analysis-prompts.md -- "GeoParquet with DuckDB"
  Input: Urban datasets (buildings, roads, land use, demographics)
  Output: Filtered and joined dataset for study area
  Validate: Feature counts, join completeness

Step 2: Spatial Analysis (Sonnet)
  Source: data-analysis-prompts.md -- "Spatial Autocorrelation Report"
  Input: Joined dataset from Step 1
  Output: Hotspot analysis results, spatial statistics
  Validate: Statistical significance thresholds, cluster validity

Step 3: Base Map (Sonnet)
  Source: map-styling-prompts.md -- "Dark Mode Basemap"
  Input: Study area extent
  Output: MapLibre style JSON for basemap
  Validate: Style JSON is valid, renders correctly

Step 4: Thematic Overlay (Sonnet)
  Source: map-styling-prompts.md -- "Data-Driven Layer Styling"
  Input: Analysis results + basemap style
  Output: Complete map style with thematic overlay
  Validate: Data-driven expressions produce correct classes

Step 5: Report Generation (Opus)
  Source: automation-workflows.md -- "Automated Map Report"
  Input: All maps + analysis results
  Output: PDF report with maps, charts, and narrative
  Validate: All maps render, page layout is correct, narrative matches data
```

### Plugin Development Pipeline

```
Agent Team: QGIS Plugin from Scratch

Step 1: Scaffolding (Sonnet)
  Source: plugin-dev-prompts.md -- "Plugin Scaffolding"
  Input: Plugin name, description, required functionality
  Output: Complete plugin directory structure
  Validate: metadata.txt is valid, plugin loads in QGIS

Step 2: Core Logic (Opus)
  Source: plugin-dev-prompts.md -- "QGIS Processing Algorithm"
  Input: Algorithm specification, input/output parameter definitions
  Output: Processing algorithm implementation
  Validate: Algorithm runs in QGIS Processing Toolbox, outputs are correct

Step 3: CI/CD (Sonnet)
  Source: automation-workflows.md -- "GitHub Actions for Geospatial Project"
  Input: Plugin repository structure
  Output: GitHub Actions workflow for testing and packaging
  Validate: CI passes, plugin ZIP is generated

Step 4: Documentation (Sonnet)
  Source: academic-writing-prompts.md -- "Describe a Processing Workflow"
  Input: Plugin functionality, algorithm parameters
  Output: User documentation and developer guide
  Validate: Documentation covers all parameters, includes examples
```

### Disaster Response Pipeline

```
Agent Team: Rapid Flood Mapping

Step 1: Flood Detection (Opus)
  Source: remote-sensing-prompts.md -- "Automated Flood Mapping in GEE"
  Input: AOI, event date, pre-event baseline date
  Output: Flood extent polygon (GeoPackage)
  Validate: Extent is plausible, no false positives in known dry areas

Step 2: Impact Assessment (Sonnet)
  Source: data-analysis-prompts.md -- "Point-in-Polygon Aggregation"
  Input: Flood extent + building/population datasets
  Output: Affected feature counts by administrative unit
  Validate: Counts are non-negative, totals match sum of parts

Step 3: Symbology (Haiku)
  Source: map-styling-prompts.md -- "Generate SVG Map Icons"
  Input: Icon requirements (damage levels, facility types)
  Output: SVG icon set
  Validate: Icons render at multiple sizes, accessible colors

Step 4: Dashboard (Sonnet)
  Source: automation-workflows.md -- "Dashboard Data Preparation"
  Input: All analysis results
  Output: Dashboard-ready data (GeoJSON, summary statistics JSON)
  Validate: GeoJSON is valid, statistics are internally consistent
```

---

## AI Tool Compatibility Matrix

Detailed compatibility assessment based on testing across prompt categories and AI tools.

| Prompt Category | Claude Opus | Claude Sonnet | Claude Haiku | GPT-4o | Gemini 2.5 | Copilot | Cursor/Windsurf |
|---|---|---|---|---|---|---|---|
| Data Analysis | Excellent | Excellent | Good | Excellent | Good | Good (inline) | Excellent |
| Remote Sensing | Excellent | Very Good | Poor | Very Good | Excellent (GEE) | Moderate | Good |
| Map Styling | Excellent | Excellent | Good | Good | Moderate | Poor | Good |
| Plugin Development | Excellent | Excellent | Moderate | Good | Moderate | Excellent | Excellent |
| Academic Writing | Excellent | Excellent | Moderate | Very Good | Good | N/A | N/A |
| Automation/CI | Excellent | Excellent | Good | Very Good | Good | Good (YAML) | Excellent |
| Agent Pipelines | Excellent | Good | Poor | Moderate | Moderate | N/A | Moderate |
| PostGIS Queries | Excellent | Excellent | Good | Very Good | Good | Good (inline) | Good |

### Tool-Specific Guidance

**Claude (API, Code, or Conversation)**

- Attach actual data files when using Claude Code -- GeoJSON, CSV, small rasters as base64
- Use the large context window (200K tokens) to include full data schemas, sample data, and reference documentation
- For multi-file projects, use Claude Code's agent mode which can read your project structure, run scripts, and iterate
- Set up CLAUDE.md with your project's GIS conventions to eliminate repeated context
- Use Skills for domain-specific rules that apply across multiple projects

**ChatGPT (GPT-4o, Code Interpreter)**

- Use Code Interpreter for prompts that benefit from immediate execution and visualization
- Attach screenshots of your current map for styling prompts -- GPT-4o's vision helps it understand the context
- Less effective for large-scale code generation; better for focused, single-task prompts
- Canvas mode is useful for iterating on longer scripts with inline editing

**GitHub Copilot**

- Convert prompt templates into detailed code comments at the top of your file
- Copilot performs best when it can see your existing imports, data structures, and function signatures
- Less effective for generating complete workflows from scratch
- Excellent for repetitive GIS code patterns (e.g., writing similar processing functions for different layers)

**Cursor / Windsurf**

- Reference project files with `@filename` to ground the AI in your actual code and data
- Full-codebase context makes these tools ideal for QGIS plugin development and multi-file refactoring
- Use `.cursorrules` (Cursor) or equivalent project rules to encode GIS conventions
- Composer mode (Cursor) / Cascade (Windsurf) handle multi-file changes well

**Google Gemini**

- Best choice for Google Earth Engine JavaScript code generation due to native API awareness
- Large context window (1M+ tokens) can ingest entire GIS project structures
- Good for long-document analysis (e.g., reading spatial standards, processing documentation)
- Grounding with Google Search is useful for finding current API documentation

---

## Contributing

### Adding a New Prompt

1. Choose the appropriate file based on the prompt category
2. Follow the Prompt Template Anatomy structure documented above
3. Include at least one tested example output
4. Specify which AI tools and model tiers the prompt has been tested with
5. Include validation checklist items specific to the prompt
6. Add cost optimization notes
7. Cross-reference relevant awesome-giser pages

### Adding a New Prompt File

1. Create the `.md` file in this directory
2. Follow the naming convention: `{category}-prompts.md`
3. Add the file to the Table of Contents in this README
4. Include at least 5 tested prompts
5. Each prompt must follow the standard template anatomy

### Quality Standards

- Every prompt must produce output that passes its own validation checklist
- Prompts must be tested with at least two different AI tools
- Version pins must reflect currently supported library versions
- CRS must always be explicitly specified, never assumed

---

## Glossary

| Term | Definition |
|---|---|
| AOI | Area of Interest -- the geographic extent for analysis |
| CRS | Coordinate Reference System -- defines how coordinates map to locations on Earth |
| EPSG | European Petroleum Survey Group -- registry of CRS definitions (e.g., EPSG:4326) |
| COG | Cloud-Optimized GeoTIFF -- a raster format optimized for HTTP range requests |
| GeoParquet | A columnar storage format for geospatial vector data, based on Apache Parquet |
| MCP | Model Context Protocol -- Anthropic's protocol for connecting AI to external tools |
| Skill | A reusable prompt module (`.md` file) that encodes domain expertise for Claude |
| Token | The basic unit of text that LLMs process; roughly 4 characters or 3/4 of a word |
| Prompt Chain | A sequence of prompts where each output feeds into the next prompt as input |
| Few-Shot | Including examples of desired input-output pairs in the prompt |
| 甲方 | Client or commissioning party (Chinese term common in project delivery contexts) |

---

[Back to Main README](../README.md)
