# Plugin & Extension Development Prompts

> Expert prompts for QGIS plugins, web map plugins, MCP servers, Claude Skills, CLI tools, and IDE extensions.

**Quick Reference**

| Section | Prompts | Focus |
|---------|---------|-------|
| [QGIS Plugin Development](#1-qgis-plugin-development) | P1--P4 | Desktop GIS plugins, Processing algorithms |
| [Web Map Plugins](#2-web-map-plugins) | P5--P8 | Leaflet, MapLibre, deck.gl |
| [MCP Server Development](#3-mcp-server-development) | P9--P12 | AI-to-GIS tool bridges |
| [Claude Skills for GIS](#4-claude-skills-for-gis) | P13--P15 | Reusable AI skill packages |
| [Python CLI Tools](#5-python-cli-tools) | P16--P18 | Command-line GIS utilities |
| [VS Code Extensions](#6-vs-code-extensions) | P19--P20 | Editor productivity |
| [ArcGIS Python Toolbox](#7-arcgis-python-toolbox) | P21--P22 | ArcPy toolboxes and migration |

---

## 1. QGIS Plugin Development

---

## Prompt 1 -- Plugin Scaffolding

### Scenario (实际场景)

You are starting a new QGIS plugin from scratch and need a complete, production-ready project structure. The scaffold must include metadata, UI dialogs, resource compilation, and a Makefile for deployment -- all targeting QGIS 3.34 LTS or later.

### Roles

**User** = GIS developer specifying plugin requirements | **AI** = Senior PyQGIS plugin architect

### Prompt Template

```text
You are a Senior PyQGIS Plugin Architect.

Create a complete QGIS plugin scaffold for "{plugin_name}" that {brief_description}.

Target: QGIS {qgis_version}+ / PyQGIS / Python 3.10+

Required file structure:
  {plugin_name}/
    __init__.py          # initGui / unload entry
    plugin.py            # Main class: toolbar, menu, iface
    dialog.py            # QDialog subclass
    dialog_base.ui       # Qt Designer .ui
    metadata.txt         # Full QGIS metadata
    resources.qrc        # Icons, assets
    resources_rc.py      # Compiled resources
    Makefile             # compile, deploy, clean, zip
    tests/
      conftest.py        # pytest-qgis fixtures
      test_plugin.py     # Smoke tests

metadata.txt fields:
  name={plugin_name}
  qgisMinimumVersion={qgis_version}
  category={category}
  tags={tags}
  description={one_line_description}

Dialog must include: {ui_elements}.
Separate core logic from UI. Include i18n stub (i18n/ dir).
Add unload() that removes all toolbar actions and cleans connections.
Makefile targets: compile (pyrcc5), deploy (~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/), clean, zip.
```

### Variables to Customize

- `{plugin_name}` -- PascalCase plugin name (e.g., `FloodMapper`)
- `{brief_description}` -- what the plugin does
- `{qgis_version}` -- minimum version (e.g., `3.34`)
- `{category}` -- Raster / Vector / Analysis / Web
- `{tags}` -- comma-separated keywords
- `{ui_elements}` -- e.g., "QgsMapLayerComboBox, QDoubleSpinBox, QPushButton"

### Expected Output

Multiple files forming a complete plugin directory, installable by copying to the QGIS plugin folder.

### Validation Checklist

- [ ] `metadata.txt` passes `qgis-plugin-ci` lint
- [ ] Plugin loads in QGIS without errors in the Python console
- [ ] Makefile `deploy` target copies to the correct profile path
- [ ] `pytest --qgis` passes on the scaffold tests
- [ ] `unload()` removes all menu entries and toolbar icons

### Cost Optimization

Use a focused system prompt. A single-shot generation with the full file tree keeps token count below 3K output tokens. Avoid iterating on UI design in the LLM -- use Qt Designer for visual layout.

### Dark Arts Tip

Ask the AI to also produce a `pb_tool.cfg` for the `pb_tool` deployment utility, then run `pb_tool deploy` instead of a custom Makefile -- it handles profile detection automatically.

### Related Pages

- [Desktop GIS Tools](../tools/desktop-gis.md) -- QGIS ecosystem overview
- [Data Analysis Prompts](data-analysis-prompts.md) -- PyQGIS analysis patterns

### Extensibility

Add a `provider.py` to register a Processing provider alongside the UI plugin, giving users both toolbar access and Processing toolbox integration.

---

## Prompt 2 -- Processing Algorithm Provider

### Scenario (实际场景)

You need a custom geoprocessing tool that integrates with the QGIS Processing framework. Processing algorithms appear in the toolbox, work in graphical models, run from `qgis_process` CLI, and support batch execution natively.

### Roles

**User** = GIS analyst defining algorithm logic | **AI** = QgsProcessingAlgorithm specialist

### Prompt Template

```text
You are a QgsProcessingAlgorithm specialist for QGIS {qgis_version}+.

Create a Processing algorithm provider plugin:

Algorithm: {algorithm_name}
Display name: "{display_name}"
Group: "{group_name}"
Purpose: {purpose_description}

Parameters (initAlgorithm):
{parameter_list}

processAlgorithm logic:
1. Validate inputs
2. {step_1}
3. {step_2}
4. {step_n}
5. Write OUTPUT with new attribute columns: {output_fields}

Requirements:
- feedback.setProgress() per feature iteration
- feedback.isCanceled() check in inner loop
- feedback.pushInfo() for key milestones
- Python 3.10+ type hints throughout
- createInstance() returns new class instance

Provide these files:
  {provider_id}/
    __init__.py           # QgsProcessingProviderPlugin
    provider.py           # QgsProcessingProvider subclass
    alg_{algorithm_name}.py
    metadata.txt
    tests/test_alg.py     # pytest with QgsProcessingFeedback

The algorithm must be callable via:
  qgis_process run "{provider_id}:{algorithm_name}" \
    -- PARAM1=value PARAM2=value OUTPUT=result.gpkg
```

### Variables to Customize

- `{algorithm_name}` -- snake_case identifier
- `{parameter_list}` -- typed parameter definitions (QgsProcessingParameterVectorLayer, etc.)
- `{purpose_description}` -- what the algorithm computes
- `{output_fields}` -- new attribute columns added to the output

### Expected Output

Complete Processing provider plugin with algorithm class, provider registration, metadata, and pytest tests.

### Validation Checklist

- [ ] Algorithm appears in the Processing Toolbox under the correct group
- [ ] `qgis_process run` executes without errors
- [ ] Progress bar advances during execution
- [ ] Cancellation stops processing within one feature cycle
- [ ] Output GeoPackage contains expected attribute columns

### Cost Optimization

Processing algorithms are highly templated. Provide the parameter list precisely to avoid a clarification round-trip. One-shot generation is usually sufficient.

### Dark Arts Tip

Include a `flags()` override returning `QgsProcessingAlgorithm.FlagNoThreading` if your algorithm uses non-thread-safe libraries (matplotlib, some rasterio calls). This prevents crashes in background execution.

### Related Pages

- [Desktop GIS Tools](../tools/desktop-gis.md) -- Processing framework details
- [CLI Tools](../tools/cli-tools.md) -- `qgis_process` command reference

### Extensibility

Chain multiple algorithms in a `QgsProcessingMultiStepAlgorithm` for complex workflows, or expose them as WPS services via QGIS Server.

---

## Prompt 3 -- Plugin Feature Implementation

### Scenario (实际场景)

You have an existing QGIS plugin and need to add a new feature that requires both UI changes (Qt widgets) and backend logic (PyQGIS API). The feature should include tests written with `pytest-qgis`.

### Roles

**User** = Plugin maintainer describing the feature | **AI** = PyQGIS + Qt integration developer

### Prompt Template

```text
You are a PyQGIS + Qt integration developer.

I have an existing QGIS plugin "{plugin_name}" that {current_functionality}.
I want to add: {feature_description}.

Current structure:
  Main class: {main_class} in {main_file}
  Dialog: {dialog_class} in {dialog_file}
  Logic: {logic_module}

Implement:
1. UI additions to {dialog_file}:
   - {widget_list}
   - Connect signals to slots
2. Backend logic in {logic_module}:
   - {backend_steps}
   - Return results as {result_type}
3. Progress reporting via QgsMessageBar
4. Error handling: catch {common_errors}, show user-friendly messages
5. Unit tests in tests/test_{feature_name}.py using pytest-qgis:
   - Test core logic with mock layers
   - Test UI signal/slot connections
   - Test error paths

If this feature is better as a Processing algorithm, say so and
provide both: the algorithm and a thin UI wrapper that calls it.
```

### Variables to Customize

- `{feature_description}` -- the new capability
- `{widget_list}` -- Qt widgets to add (QgsMapLayerComboBox, QProgressBar, etc.)
- `{backend_steps}` -- algorithm steps in the logic module
- `{common_errors}` -- CRS mismatch, empty selection, invalid geometry

### Expected Output

Python code for UI modifications, backend logic, and pytest tests. Optionally a Processing algorithm if the feature is better suited to that pattern.

### Validation Checklist

- [ ] New widgets appear in the dialog and are functional
- [ ] Backend logic handles edge cases (empty layers, null geometries)
- [ ] QgsMessageBar shows progress and errors appropriately
- [ ] All tests pass with `pytest --qgis`

### Cost Optimization

Provide the existing file contents (or key excerpts) in the prompt. Without them, the AI invents a structure that may not match yours, wasting a round-trip.

### Dark Arts Tip

Use `QgsApplication.taskManager().addTask(QgsTask.fromFunction(...))` for long-running operations. This runs the work in a background thread with built-in progress and cancellation, keeping the UI responsive.

### Related Pages

- [Desktop GIS Tools](../tools/desktop-gis.md)
- [Data Analysis Prompts](data-analysis-prompts.md)

### Extensibility

Wrap the feature as a Processing algorithm so it is available in both the plugin dialog and the Processing Toolbox / Sketcher models.

---

## Prompt 4 -- Plugin Debugging & Migration

### Scenario (实际场景)

Your QGIS plugin has broken after upgrading QGIS versions, or it has a runtime bug you cannot diagnose. You need help identifying the root cause, understanding API changes, and producing a fix.

### Roles

**User** = Plugin developer with a traceback | **AI** = QGIS API migration and debugging specialist

### Prompt Template

```text
You are a QGIS API migration and debugging specialist.

Plugin: "{plugin_name}"
QGIS version: {qgis_version_from} -> {qgis_version_to}
Python: {python_version} | OS: {os}

Problem: {symptom_description}

Error output:
```
{traceback}
```

Relevant code:
```python
{code_snippet}
```

Tasks:
1. Diagnose the root cause
2. If API change: identify the specific deprecation / removal
   and cite the QGIS changelog or migration docs
3. Provide a fix that works on {qgis_version_to}
4. If possible, provide a compatibility shim for both versions
5. Suggest a test to prevent regression
6. List any other deprecated calls in the snippet that will
   break in future versions
```

### Variables to Customize

- `{qgis_version_from}` / `{qgis_version_to}` -- version pair
- `{traceback}` -- full error output
- `{code_snippet}` -- the failing code section

### Expected Output

Root cause analysis, corrected code, compatibility notes, and a regression test.

### Validation Checklist

- [ ] Fix resolves the original error
- [ ] Plugin loads cleanly on the target QGIS version
- [ ] No new deprecation warnings in the Python console
- [ ] Regression test covers the failure mode

### Cost Optimization

Always include the traceback and QGIS version numbers. Without them, the AI guesses at the API change, producing generic advice instead of a targeted fix.

### Dark Arts Tip

Run `python3 -c "from qgis.core import Qgis; print(Qgis.QGIS_VERSION)"` inside the QGIS Python console and pipe your plugin through `pylint --load-plugins=qgis_sketcher` to catch deprecated API calls before they break.

### Related Pages

- [Desktop GIS Tools](../tools/desktop-gis.md)

### Extensibility

Maintain a `compat.py` module in your plugin that wraps version-specific API calls behind a stable interface, making future migrations trivial.

---

## 2. Web Map Plugins

---

## Prompt 5 -- Leaflet Control Plugin

### Scenario (实际场景)

You want to create a reusable Leaflet plugin that adds a custom map control, published as an ES module on npm. The control must be accessible, mobile-friendly, and ship with TypeScript type declarations.

### Roles

**User** = Web map developer specifying control behavior | **AI** = Leaflet plugin engineer

### Prompt Template

```text
You are a Leaflet plugin engineer.

Create a Leaflet control plugin:
  Name: L.Control.{ControlName}
  Behavior: {behavior_description}
  Position: {position}

Requirements:
- ES module with named export
- Extend L.Control: initialize(), onAdd(map), onRemove(map)
- Options with defaults: {options_list}
- Custom events via map.fire(): {events_list}
- ARIA labels on all interactive elements
- Keyboard navigable (Tab, Enter, Escape)
- Mobile: touch events, 44px minimum tap targets
- CSS: self-contained, BEM naming, no external deps
- TypeScript .d.ts declarations

Deliverables:
  src/index.js          # Plugin source
  src/style.css         # Scoped CSS
  index.d.ts            # TypeScript types
  package.json          # name, main, module, types, peerDeps: leaflet
  demo/index.html       # Working demo with OSM tiles
  README.md             # Usage + API docs

Leaflet peer dependency: ^1.9.0
No external runtime dependencies.
```

### Variables to Customize

- `{ControlName}` -- e.g., `CoordinateDisplay`, `OpacitySlider`, `MeasureTool`
- `{behavior_description}` -- full UX flow
- `{options_list}` -- option name, type, default value
- `{events_list}` -- event name and payload shape

### Expected Output

Complete npm-publishable plugin with source, types, CSS, demo page, and package.json.

### Validation Checklist

- [ ] `npm pack` produces a valid tarball
- [ ] Demo page renders control in all four positions
- [ ] Screen reader announces control purpose (ARIA)
- [ ] Control works on iOS Safari and Android Chrome

### Cost Optimization

Keep the behavior description precise. Ambiguous UX descriptions cause multiple revision rounds. If possible, sketch the UI flow as numbered steps.

### Dark Arts Tip

Use `L.DomEvent.disableClickPropagation(container)` and `L.DomEvent.disableScrollPropagation(container)` in `onAdd()` to prevent map interactions from bleeding through the control -- a bug that plagues many published plugins.

### Related Pages

- [2D Mapping Libraries](../js-bindbox/2d-mapping.md) -- Leaflet ecosystem
- [Map Styling Prompts](map-styling-prompts.md) -- Cartographic controls

### Extensibility

Add a `L.control.{controlName}` factory function following Leaflet convention, and register it as a Leaflet plugin on the plugins page.

---

## Prompt 6 -- MapLibre Custom Control

### Scenario (实际场景)

You need a custom MapLibre GL JS control written in TypeScript that implements the `IControl` interface, emits custom events, encapsulates its CSS, and works on mobile devices.

### Roles

**User** = Frontend developer defining control behavior | **AI** = MapLibre TypeScript control specialist

### Prompt Template

```text
You are a MapLibre GL JS TypeScript control specialist.

Create a custom control:
  Class: {ClassName}
  Behavior: {behavior_description}

Implement IControl interface:
  onAdd(map: maplibregl.Map): HTMLElement
  onRemove(map: maplibregl.Map): void
  getDefaultPosition(): ControlPosition  // '{position}'

Options interface:
```typescript
interface {ClassName}Options {
  {options_block}
}
```

Custom events (via map.fire()):
  {events_list}

Requirements:
- TypeScript strict mode, no `any`
- CSS encapsulated via Shadow DOM or scoped class prefix
- Mobile: touch events, 44px tap targets
- Cleanup: remove all listeners and DOM in onRemove()
- No runtime dependencies beyond maplibre-gl

Project structure:
  src/index.ts
  src/styles.css
  package.json          # main, module, types
  tsconfig.json         # ES2020, maplibre-gl 4.x types
  demo/index.html       # Live demo

Include usage example:
```typescript
import { {ClassName} } from '{package_name}';
map.addControl(new {ClassName}({ ... }), '{position}');
```
```

### Variables to Customize

- `{ClassName}` -- PascalCase control name
- `{behavior_description}` -- full user interaction flow
- `{options_block}` -- TypeScript interface properties
- `{events_list}` -- event names with payload types

### Expected Output

TypeScript source, compiled JS, CSS, package.json, tsconfig, and a working demo page.

### Validation Checklist

- [ ] `tsc --noEmit` passes with zero errors
- [ ] Control renders correctly on MapLibre GL JS 4.x
- [ ] Events fire with correct payload types
- [ ] `onRemove()` leaves no orphaned DOM nodes or listeners

### Cost Optimization

Provide the `{options_block}` as a typed TypeScript interface directly. This eliminates ambiguity and lets the AI generate matching implementation code in one pass.

### Dark Arts Tip

For controls that need map canvas interaction (drawing, measuring), use `map.getCanvasContainer().style.cursor = 'crosshair'` on activate and restore on deactivate. Many tutorials forget cursor management.

### Related Pages

- [2D Mapping Libraries](../js-bindbox/2d-mapping.md) -- MapLibre ecosystem
- [Map Styling Prompts](map-styling-prompts.md)

### Extensibility

Wrap the control as a React/Vue component with a thin adapter, re-exporting the core class for framework-agnostic use.

---

## Prompt 7 -- MapLibre Custom Layer (WebGL)

### Scenario (实际场景)

You need a WebGL-based custom layer in MapLibre GL JS that renders data using shaders -- for example, animated particle flows, GPU-accelerated heatmaps, or custom raster visualization. This requires implementing the `CustomLayerInterface`.

### Roles

**User** = Developer describing visualization | **AI** = WebGL/MapLibre custom layer engineer

### Prompt Template

```text
You are a WebGL/MapLibre custom layer engineer.

Create a MapLibre GL JS custom layer:
  id: "{layer_id}"
  type: "custom"
  Visualization: {viz_description}

Implement CustomLayerInterface:
  onAdd(map, gl): void    # Create shaders, buffers, textures
  render(gl, args): void  # Draw call per frame
  onRemove(map, gl): void # Cleanup GPU resources

Shader requirements:
  Vertex: {vertex_shader_description}
  Fragment: {fragment_shader_description}

Data source: {data_source_type}
Update method: {update_strategy}

Performance targets:
- {max_features} features at 60fps
- Efficient buffer updates (subData, not full realloc)
- requestAnimationFrame for animations

Provide:
  src/custom-layer.ts     # CustomLayerInterface impl
  src/shaders/vert.glsl   # Vertex shader
  src/shaders/frag.glsl   # Fragment shader
  src/utils.ts            # Projection helpers (lngLat -> mercator)
  demo/index.html         # With sample data

Use maplibregl.MercatorCoordinate for geo-to-screen projection.
Handle map pitch and bearing in the vertex shader.
```

### Variables to Customize

- `{layer_id}` -- unique layer identifier
- `{viz_description}` -- what the layer renders (particles, isolines, etc.)
- `{vertex_shader_description}` / `{fragment_shader_description}` -- shader behavior
- `{data_source_type}` -- GeoJSON, binary buffer, tile URL

### Expected Output

TypeScript implementation, GLSL shaders, projection utilities, and a demo with sample data.

### Validation Checklist

- [ ] Layer renders correctly at multiple zoom levels
- [ ] GPU resources are fully released in `onRemove()`
- [ ] No WebGL errors in the console
- [ ] Frame rate stays above 55fps with target feature count

### Cost Optimization

WebGL code is error-prone in LLM output. Request the shaders as separate `.glsl` files, not inline strings, so they are easier to debug. Provide a small sample GeoJSON to avoid the AI inventing data structures.

### Dark Arts Tip

Use `gl.getExtension('OES_element_index_uint')` to enable 32-bit index buffers -- without it, you are limited to 65K vertices per draw call, which is easy to exceed with dense geospatial data.

### Related Pages

- [3D Mapping Libraries](../js-bindbox/3d-mapping.md) -- WebGL ecosystem
- [Map Styling Prompts](map-styling-prompts.md)

### Extensibility

Abstract the shader and buffer setup into a base class, then create derived layers (particle, heatmap, vector field) that only override the shader pair and data binding.

---

## Prompt 8 -- deck.gl Custom Layer

### Scenario (实际场景)

You need a custom deck.gl layer -- either a `CompositeLayer` that composes existing layers, or a `Layer` subclass with custom GPU rendering -- for geospatial visualization that goes beyond deck.gl's built-in layer catalog.

### Roles

**User** = Visualization developer specifying layer behavior | **AI** = deck.gl layer extension engineer

### Prompt Template

```text
You are a deck.gl layer extension engineer.

Create a custom deck.gl layer:
  Class: {LayerName}Layer extends {base_class}
  Visualization: {viz_description}

If CompositeLayer:
  renderLayers(): return composed sub-layers
  Define props interface with defaultProps

If Layer (GPU):
  Provide vertex + fragment shaders in shaders/ dir
  Define attributes in initializeState()
  Update buffers in updateState()
  Use luma.gl Model for draw calls

Props interface:
```typescript
interface {LayerName}LayerProps extends LayerProps {
  {props_block}
}
```

Requirements:
- TypeScript
- deck.gl 9.x / luma.gl 9.x APIs
- Picking support: getPickingInfo()
- Transitions: support updateTriggers
- Coordinate system: COORDINATE_SYSTEM.LNGLAT
- Handle data updates efficiently (diffProps)

Deliverables:
  src/{layer-name}-layer.ts
  src/shaders/ (if GPU layer)
  demo/app.tsx            # React demo with DeckGL component
  package.json
```

### Variables to Customize

- `{LayerName}` -- PascalCase layer name
- `{base_class}` -- `CompositeLayer` or `Layer`
- `{viz_description}` -- rendering behavior
- `{props_block}` -- typed layer properties

### Expected Output

TypeScript layer class, optional shaders, a React demo application, and package.json.

### Validation Checklist

- [ ] Layer renders in a deck.gl `DeckGL` component
- [ ] Picking returns correct feature info on hover/click
- [ ] `updateTriggers` cause re-render when dependencies change
- [ ] No GPU memory leaks on repeated data updates

### Cost Optimization

Specify whether you need a `CompositeLayer` or a raw `Layer`. CompositeLayer prompts are simpler and produce more reliable output. Only request a raw GPU Layer when you genuinely need custom shaders.

### Dark Arts Tip

When extending `Layer`, override `draw()` instead of `render()` and use `this.state.model.setUniforms(...)` to pass per-frame uniforms like time for animations. This avoids the common mistake of creating new uniform buffers each frame.

### Related Pages

- [3D Mapping Libraries](../js-bindbox/3d-mapping.md) -- deck.gl ecosystem
- [Map Styling Prompts](map-styling-prompts.md)

### Extensibility

Publish the layer as a standalone npm package with a `/shaders` export, allowing consumers to customize the GLSL without forking.

---

## 3. MCP Server Development

> Model Context Protocol (MCP) servers expose geospatial tools to AI assistants. Each server
> provides a set of typed tools that an LLM can invoke to query databases, search catalogs,
> convert formats, or fetch OpenStreetMap data.

---

## Prompt 9 -- PostGIS MCP Server

### Scenario (实际场景)

You want to connect an AI assistant (Claude, etc.) directly to a PostGIS database so it can explore schemas, run spatial queries, and return GeoJSON results. The MCP server acts as a secure bridge with parameterized queries to prevent SQL injection.

### Roles

**User** = GIS engineer defining database access patterns | **AI** = MCP server architect (TypeScript)

### Prompt Template

```text
You are an MCP server architect specializing in geospatial backends.

Create a PostGIS MCP Server using TypeScript and @modelcontextprotocol/sdk.

Server name: "postgis-server"
Transport: stdio

Tools to implement:
  list_tables:
    description: "List all spatial tables with geometry type and SRID"
    params: { schema?: string }
    returns: Array<{ table, geom_column, geom_type, srid, row_count }>

  get_extent:
    description: "Get the bounding box of a spatial table"
    params: { table: string, where?: string }
    returns: { bbox: [minx, miny, maxx, maxy], srid: number }

  spatial_query:
    description: "Run a spatial SQL query, return GeoJSON FeatureCollection"
    params: {
      sql: string,        // SELECT with geometry column
      max_rows?: number   // default 1000
    }
    returns: GeoJSON FeatureCollection

  find_nearest:
    description: "Find N nearest features to a point"
    params: {
      table: string,
      lon: number, lat: number,
      n?: number,         // default 5
      max_distance_m?: number
    }
    returns: GeoJSON FeatureCollection with distance_m property

Security:
- Use parameterized queries (pg library $1, $2 syntax)
- Read-only connection (SET default_transaction_read_only = on)
- SQL allowlist: only SELECT statements (reject INSERT/UPDATE/DELETE)
- Connection string from environment variable POSTGIS_URL

Project structure:
  src/
    index.ts              # Server entry, McpServer setup
    tools/
      list-tables.ts
      get-extent.ts
      spatial-query.ts
      find-nearest.ts
    db.ts                 # Connection pool (pg.Pool)
    validators.ts         # Input validation with zod
  package.json            # bin entry for npx execution
  tsconfig.json
  README.md               # Setup + Claude Desktop config example

Include Claude Desktop MCP config snippet:
{
  "mcpServers": {
    "postgis": {
      "command": "npx",
      "args": ["-y", "postgis-mcp-server"],
      "env": { "POSTGIS_URL": "postgresql://..." }
    }
  }
}
```

### Variables to Customize

- Additional tools (e.g., `get_column_stats`, `spatial_join`)
- Schema restrictions (limit to specific schemas)
- Row limits and timeout settings
- Authentication method (connection string vs. individual params)

### Expected Output

Complete TypeScript MCP server project with all tool handlers, connection management, input validation, and a Claude Desktop configuration example.

### Validation Checklist

- [ ] `npx postgis-mcp-server` starts without errors
- [ ] `list_tables` returns correct geometry metadata
- [ ] `spatial_query` returns valid GeoJSON with correct CRS
- [ ] SQL injection attempts are rejected (DROP, INSERT, etc.)
- [ ] Connection pool handles concurrent tool calls

### Cost Optimization

Define tool schemas precisely using Zod. The AI generates cleaner validation code when schemas are explicit. Avoid asking for "general SQL access" -- scoped tools produce more reliable and secure implementations.

### Dark Arts Tip

Add a `query_explain` tool that runs `EXPLAIN (FORMAT JSON)` on queries before execution. This lets the AI self-optimize: if the plan shows a sequential scan on a large table, it can add a spatial index hint or `ST_DWithin` filter.

### Related Pages

- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS setup and indexing
- [AI Agent Patterns](ai-agent-patterns.md) -- MCP tool design patterns

### Extensibility

Add a `Resources` endpoint that exposes table schemas as MCP resources, allowing the AI to read column definitions before constructing queries.

---

## Prompt 10 -- STAC Catalog MCP Server

### Scenario (实际场景)

You want an AI assistant to search SpatioTemporal Asset Catalogs (STAC) for satellite imagery, inspect metadata, and download assets. The MCP server wraps `pystac-client` and provides structured tools for catalog interaction.

### Roles

**User** = Remote sensing analyst defining catalog access | **AI** = MCP server architect (Python)

### Prompt Template

```text
You are an MCP server architect for remote sensing workflows.

Create a STAC Catalog MCP Server in Python using mcp[cli] (Python SDK).

Server name: "stac-server"
Transport: stdio

Tools:
  search_items:
    description: "Search a STAC catalog for items matching criteria"
    params: {
      catalog_url: string,     # e.g., "https://earth-search.aws.element84.com/v1"
      collection: string,      # e.g., "sentinel-2-l2a"
      bbox?: [w, s, e, n],
      datetime?: string,       # RFC 3339 range "2024-01/2024-06"
      query?: dict,            # e.g., {"eo:cloud_cover": {"lt": 20}}
      max_items?: int          # default 10
    }
    returns: { items: [{ id, datetime, bbox, cloud_cover, thumbnail_url }] }

  get_item_assets:
    description: "List assets of a specific STAC item"
    params: { catalog_url, collection, item_id }
    returns: { assets: [{ key, href, type, title, file_size }] }

  get_item_metadata:
    description: "Get full STAC item metadata including all properties"
    params: { catalog_url, collection, item_id }
    returns: Full STAC item JSON

  download_asset:
    description: "Download a STAC asset to local directory"
    params: { href: string, output_dir?: string }
    returns: { local_path, file_size_mb }

Dependencies: pystac-client, httpx, mcp[cli]

Project structure:
  stac_mcp/
    __init__.py
    server.py               # MCP server setup
    tools/
      search.py
      assets.py
      download.py
    auth.py                  # Optional bearer token for private catalogs
  pyproject.toml
  README.md
```

### Variables to Customize

- Default catalog URL (Earth Search, Planetary Computer, custom)
- Authentication method (none, API key, bearer token)
- Download size limits
- Additional tools (e.g., `generate_mosaic`, `compute_statistics`)

### Expected Output

Python MCP server with tool handlers, pystac-client integration, and setup instructions for Claude Desktop.

### Validation Checklist

- [ ] `search_items` returns results from Earth Search v1
- [ ] `get_item_assets` lists all asset keys with correct HREFs
- [ ] `download_asset` saves file and returns valid local path
- [ ] Cloud cover filter produces correctly filtered results

### Cost Optimization

Use `max_items` aggressively. STAC searches can return thousands of items; returning all of them to an LLM wastes context tokens. Default to 10 and let the AI request more if needed.

### Dark Arts Tip

Add a `preview_item` tool that returns the thumbnail URL as a Markdown image link. Many STAC items include RGB quicklook thumbnails -- surfacing them lets the AI "see" the imagery and advise the user on cloud cover or scene quality without downloading full assets.

### Related Pages

- [Satellite Imagery Sources](../data-sources/satellite-imagery.md) -- STAC catalogs directory
- [Remote Sensing Prompts](remote-sensing-prompts.md) -- Analysis workflows

### Extensibility

Add an MCP `Resource` that exposes collection metadata, so the AI can browse available collections before searching. Chain with a GDAL MCP server for post-download processing.

---

## Prompt 11 -- GDAL/OGR MCP Server

### Scenario (实际场景)

You want an AI assistant to perform format conversion, reprojection, clipping, and inspection of geospatial files using GDAL/OGR -- without the user needing to remember GDAL command syntax.

### Roles

**User** = GIS data engineer defining conversion needs | **AI** = MCP server architect wrapping GDAL

### Prompt Template

```text
You are an MCP server architect wrapping GDAL/OGR for AI assistants.

Create a GDAL/OGR MCP Server in Python using mcp[cli].

Server name: "gdal-server"
Transport: stdio

Tools:
  ogrinfo:
    description: "Inspect vector file: layers, fields, geometry type, CRS, extent, feature count"
    params: { path: string, layer?: string }
    returns: { layers: [{ name, geom_type, srid, feature_count, fields, extent }] }

  gdalinfo:
    description: "Inspect raster file: dimensions, bands, CRS, extent, pixel size, nodata"
    params: { path: string }
    returns: { size, bands, crs, extent, pixel_size, nodata_values }

  ogr2ogr:
    description: "Convert/reproject/clip vector data"
    params: {
      input: string,
      output: string,
      output_format?: string,   # GPKG, GeoJSON, FlatGeobuf, Parquet
      target_crs?: string,      # EPSG:XXXX
      clip_bbox?: [w,s,e,n],
      where?: string,           # OGR SQL WHERE clause
      select_fields?: string[]
    }
    returns: { output_path, feature_count, format }

  gdal_translate:
    description: "Convert/reproject/clip raster data"
    params: {
      input: string,
      output: string,
      output_format?: string,   # GTiff, COG, PNG
      target_crs?: string,
      clip_bbox?: [w,s,e,n],
      resolution?: [xres, yres]
    }
    returns: { output_path, size, bands }

  gdalwarp:
    description: "Reproject and warp raster data with resampling"
    params: {
      input: string,
      output: string,
      target_crs: string,
      resampling?: string,      # near, bilinear, cubic, average
      resolution?: [xres, yres]
    }
    returns: { output_path, size, crs }

Security:
- Validate all paths are within allowed directories (GDAL_ALLOWED_DIRS env)
- Reject path traversal (../)
- Set GDAL_CACHEMAX and timeout per operation

Dependencies: GDAL Python bindings (osgeo), mcp[cli]

Project structure:
  gdal_mcp/
    server.py
    tools/
      vector_info.py
      raster_info.py
      vector_convert.py
      raster_convert.py
    security.py             # Path validation
    gdal_helpers.py         # Common GDAL config
  pyproject.toml
```

### Variables to Customize

- Allowed directory paths
- Maximum file size for operations
- Additional tools (e.g., `build_vrt`, `rasterize`, `polygonize`)
- GDAL configuration options (CACHEMAX, NUM_THREADS)

### Expected Output

Python MCP server with GDAL tool handlers, path security, and configuration management.

### Validation Checklist

- [ ] `ogrinfo` returns correct metadata for GeoPackage and Shapefile
- [ ] `ogr2ogr` converts between formats with correct CRS transformation
- [ ] Path traversal attempts are rejected
- [ ] Operations respect timeout limits

### Cost Optimization

Return metadata summaries, not full file contents. For `ogrinfo`, return field names and types but not all feature values. The AI needs schema info to reason about the data, not the data itself.

### Dark Arts Tip

Set `GDAL_NUM_THREADS=ALL_CPUS` and `GDAL_CACHEMAX=512` as environment defaults in the server startup. Also set `CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES` for COG output -- without it, COG creation fails on some cloud storage backends.

### Related Pages

- [CLI Tools](../tools/cli-tools.md) -- GDAL command reference
- [AI Agent Patterns](ai-agent-patterns.md) -- Tool chaining patterns

### Extensibility

Add a `batch_convert` tool that takes a glob pattern and processes all matching files in parallel using `concurrent.futures.ProcessPoolExecutor`.

---

## Prompt 12 -- Overpass API MCP Server

### Scenario (实际场景)

You want an AI assistant to query OpenStreetMap data via the Overpass API. The server translates high-level requests (bbox, tags, feature types) into Overpass QL queries and returns GeoJSON results.

### Roles

**User** = Mapper or analyst defining OSM query needs | **AI** = MCP server architect wrapping Overpass API

### Prompt Template

```text
You are an MCP server architect for OpenStreetMap data access.

Create an Overpass API MCP Server in TypeScript using @modelcontextprotocol/sdk.

Server name: "overpass-server"
Transport: stdio

Tools:
  query_features:
    description: "Query OSM features by bounding box and tags, return GeoJSON"
    params: {
      bbox: [south, west, north, east],
      tags: Record<string, string | string[]>,
        // e.g., {"amenity": "hospital"} or {"highway": ["primary","secondary"]}
      element_type?: "node" | "way" | "relation" | "nwr",  // default "nwr"
      max_results?: number,
      include_metadata?: boolean   // user, timestamp, version
    }
    returns: GeoJSON FeatureCollection with OSM tags as properties

  query_around:
    description: "Find OSM features within radius of a point"
    params: {
      lat: number, lon: number,
      radius_m: number,           // max 5000
      tags: Record<string, string | string[]>
    }
    returns: GeoJSON FeatureCollection

  get_relation:
    description: "Get a specific OSM relation as GeoJSON (e.g., admin boundary)"
    params: { relation_id: number }
    returns: GeoJSON Feature with relation tags and geometry

  raw_overpass:
    description: "Execute a raw Overpass QL query (advanced)"
    params: { query: string, format?: "json" | "geojson" }
    returns: Overpass API response

Implementation details:
- Use Overpass API endpoint: https://overpass-api.de/api/interpreter
- Build Overpass QL from params (do NOT expose raw QL by default)
- Convert Overpass JSON to standard GeoJSON FeatureCollection
- Handle way geometry reconstruction (resolve node references)
- Rate limiting: max 2 requests per 10 seconds
- Timeout: [timeout:30] in Overpass QL
- Retry on 429 with exponential backoff

Project structure:
  src/
    index.ts
    tools/
      query-features.ts
      query-around.ts
      get-relation.ts
      raw-overpass.ts
    overpass-client.ts     # HTTP client with rate limiting
    overpass-builder.ts    # Build Overpass QL from params
    geojson-converter.ts   # Overpass JSON -> GeoJSON
  package.json
  tsconfig.json
```

### Variables to Customize

- Overpass API endpoint (public or private instance)
- Rate limiting parameters
- Maximum bbox area (prevent overly broad queries)
- Additional tools (e.g., `count_features`, `get_changesets`)

### Expected Output

TypeScript MCP server with Overpass QL builder, GeoJSON converter, rate limiting, and all tool handlers.

### Validation Checklist

- [ ] `query_features` returns hospitals in a given bbox
- [ ] Way geometries are complete (not just node references)
- [ ] Rate limiter prevents 429 errors on rapid calls
- [ ] `get_relation` returns admin boundary as a valid polygon

### Cost Optimization

Limit bbox area (e.g., max 0.25 square degrees) and max_results. A careless query for `building=*` in a city returns millions of features, blowing up context windows and API quotas.

### Dark Arts Tip

Add `out geom;` instead of `out body; >; out skel qt;` in the Overpass QL builder. The `out geom` form returns full geometry inline, avoiding the need for a second pass to resolve node coordinates -- halving response time for way queries.

### Related Pages

- [Vector Data Sources](../data-sources/vector-data.md) -- OSM data access
- [AI Agent Patterns](ai-agent-patterns.md) -- MCP tool design

### Extensibility

Cache frequent queries using a local SQLite store keyed by a hash of the Overpass QL. OSM data changes infrequently enough that a 1-hour TTL dramatically reduces API calls.

---

## 4. Claude Skills for GIS

> Claude Skills are reusable knowledge packages stored in `~/.claude/skills/`. Each skill has
> a `SKILL.md` manifest and a `rules/` directory with focused rule files that Claude reads
> before writing code. Skills encode domain expertise that persists across sessions.

---

## Prompt 13 -- Spatial Analysis Skill

### Scenario (实际场景)

You want to create a Claude Skill that encodes best practices for spatial analysis -- CRS handling, geometry validation, and output format selection. Once installed, Claude automatically applies these rules whenever it writes geospatial code.

### Roles

**User** = GIS team lead defining coding standards | **AI** = Claude Skill author

### Prompt Template

```text
You are a Claude Skill author for geospatial development teams.

Create a Claude Skill for spatial analysis best practices.

Skill directory: ~/.claude/skills/spatial-analysis/

SKILL.md contents:
  - Name: Spatial Analysis Best Practices
  - Description: Rules for CRS handling, geometry validation,
    and output format selection in Python geospatial code
  - When to apply: Any task involving GeoPandas, Shapely,
    PyProj, Fiona, or Rasterio
  - Rule files to read (in order):
    1. rules/crs-handling.md
    2. rules/geometry-validation.md
    3. rules/output-formats.md

rules/crs-handling.md:
  - Always check CRS before spatial operations
  - Reproject to a projected CRS (not EPSG:4326) before
    distance/area calculations
  - Use pyproj.Transformer, not deprecated transform()
  - Prefer UTM zone auto-detection for local analysis
  - When combining datasets, reproject to the CRS of the
    largest dataset to minimize distortion
  - Include CRS in all output files
  - Example code patterns for each rule

rules/geometry-validation.md:
  - Run shapely.validation.make_valid() on input geometries
  - Check for: empty, null, self-intersecting, duplicate vertices
  - Buffer(0) as a quick fix, make_valid() as the proper fix
  - Log invalid geometry counts, do not silently drop features
  - Validate topology after overlay operations
  - Example code patterns

rules/output-formats.md:
  - Default to GeoPackage for vector output
  - Use GeoParquet for analytical workflows (large datasets)
  - Use GeoJSON only for web consumption (< 10MB)
  - Use COG (Cloud Optimized GeoTiff) for raster output
  - Always include metadata: CRS, creation date, source attribution
  - Set appropriate coordinate precision (6 decimal places for
    EPSG:4326, 2 for projected CRS in meters)
  - Example code patterns

Each rule file should be 30-50 lines, with concrete code examples
showing the right and wrong way to do things.
```

### Variables to Customize

- Rule topics (add `rules/indexing.md`, `rules/parallel-processing.md`)
- Target libraries (GeoPandas vs. pure GDAL/OGR)
- Team-specific conventions (naming, logging format)

### Expected Output

Complete skill directory with `SKILL.md` manifest and three rule files, each containing best-practice rules with code examples.

### Validation Checklist

- [ ] `SKILL.md` follows the Claude Skills manifest format
- [ ] Each rule file contains concrete code examples (right vs. wrong)
- [ ] Rules are actionable, not just advisory
- [ ] Claude reads and applies the rules when writing spatial code

### Cost Optimization

Keep each rule file under 50 lines. Claude reads all referenced rules into context -- bloated rule files waste tokens on every invocation. Focus on the highest-impact rules.

### Dark Arts Tip

Add a `rules/common-mistakes.md` that lists the top 10 PyQGIS/GeoPandas mistakes you have seen in your codebase. Claude will actively avoid them. This is more effective than positive rules because it targets specific failure modes.

### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- GeoPandas, Shapely, PyProj
- [Data Analysis Prompts](data-analysis-prompts.md) -- Analysis workflows

### Extensibility

Add a `rules/testing.md` that mandates test patterns: fixture GeoPackages, CRS round-trip tests, and geometry invariant assertions.

---

## Prompt 14 -- Cartographic Style Skill

### Scenario (实际场景)

You want Claude to produce cartographically sound maps every time, following established rules for color schemes, label placement, and scale-dependent styling. This skill encodes cartographic design expertise.

### Roles

**User** = Cartographer defining visual standards | **AI** = Claude Skill author for cartography

### Prompt Template

```text
You are a Claude Skill author specializing in cartographic design.

Create a Claude Skill for cartographic styling best practices.

Skill directory: ~/.claude/skills/cartographic-style/

SKILL.md:
  - Name: Cartographic Style Guide
  - When to apply: Any task producing styled maps (Mapbox/MapLibre
    styles, QGIS QML, SLD, Python matplotlib/folium maps)

Rule files:

rules/color-schemes.md:
  - Use colorbrewer2.org palettes for thematic maps
  - Sequential: for ordered data (population density)
  - Diverging: for data with meaningful midpoint (temp anomaly)
  - Qualitative: for categorical data (land use)
  - Max 7 classes for choropleth (cognitive limit)
  - Ensure WCAG 2.1 AA contrast (4.5:1 minimum)
  - Test with colorblind simulation (deuteranopia, protanopia)
  - Provide hex codes, not color names

rules/label-placement.md:
  - Point labels: prefer upper-right, avoid overlap
  - Line labels: follow line direction, repeat at intervals
  - Polygon labels: centered, scale font with polygon area
  - Priority: larger features labeled first
  - Halo/buffer: 1-2px white halo for readability
  - Never rotate labels more than 30 degrees
  - Min font size: 8pt print, 12px screen

rules/scale-dependent-styling.md:
  - Define zoom/scale ranges for each layer
  - Simplify geometry at small scales (ST_Simplify)
  - Hide detail labels below zoom 12
  - Increase line width at large scales
  - Provide specific minzoom/maxzoom for each layer type:
    buildings: 14+, roads: 8+, admin boundaries: 4+
  - Example style blocks for MapLibre and QGIS
```

### Variables to Customize

- Target styling format (MapLibre JSON, QGIS QML, SLD, CSS)
- Organization brand colors to incorporate
- Map type focus (thematic, topographic, reference)

### Expected Output

Skill directory with manifest and three rule files containing cartographic rules and style code examples.

### Validation Checklist

- [ ] Color palettes pass colorblind simulation tests
- [ ] Label rules include concrete size and placement values
- [ ] Scale-dependent rules specify exact zoom/scale thresholds
- [ ] Examples are provided in at least two styling formats

### Cost Optimization

Reference external resources (colorbrewer2.org, WCAG guidelines) by URL rather than embedding large color tables. Claude can fetch or recall these at inference time.

### Dark Arts Tip

Include a `rules/anti-patterns.md` with screenshots described in text (or URLs) of common cartographic mistakes: rainbow color scales, overlapping labels, invisible features at print scale. Negative examples are highly effective.

### Related Pages

- [Map Styling Prompts](map-styling-prompts.md) -- Styling workflows
- [2D Mapping Libraries](../js-bindbox/2d-mapping.md) -- MapLibre style spec

### Extensibility

Add `rules/print-export.md` for DPI settings, bleed margins, and CMYK color considerations when producing maps for print.

---

## Prompt 15 -- Data Quality Skill

### Scenario (实际场景)

You want Claude to automatically check and enforce data quality standards whenever it processes geospatial data -- verifying geometry validity, CRS consistency, schema completeness, and attribute integrity.

### Roles

**User** = Data quality manager defining standards | **AI** = Claude Skill author for data QA/QC

### Prompt Template

```text
You are a Claude Skill author for geospatial data quality assurance.

Create a Claude Skill for data quality enforcement.

Skill directory: ~/.claude/skills/data-quality/

SKILL.md:
  - Name: Geospatial Data Quality Standards
  - When to apply: Any task reading, transforming, or writing
    geospatial data (vector or raster)

Rule files:

rules/geometry-validity.md:
  - Validate all input geometries before processing
  - Check: is_valid, is_empty, is_simple (for lines)
  - Report counts: valid, invalid, empty, null
  - Repair strategy: make_valid() -> buffer(0) -> log & skip
  - After overlay: re-validate (common source of slivers)
  - No silent dropping: always log skipped features with FID
  - Code pattern: validation wrapper function

rules/crs-verification.md:
  - Every dataset must have a defined CRS (reject undefined)
  - When combining: assert CRS match before join/overlay
  - When reprojecting: verify result CRS matches target
  - Log CRS transformations: source -> target
  - Warn if geographic CRS used for distance/area operations
  - Code pattern: CRS guard decorator

rules/schema-completeness.md:
  - Define expected schema (field names, types, nullable)
  - Validate on load: missing fields, extra fields, type mismatches
  - Null audit: report null percentage per field
  - Referential integrity: foreign keys resolve to valid features
  - Unique constraints: no duplicate IDs
  - Code pattern: schema validator class with YAML config
```

### Variables to Customize

- Quality thresholds (e.g., max 5% null values, max 1% invalid geometries)
- Schema definition format (YAML, JSON Schema, Pandera)
- Target data formats and pipelines

### Expected Output

Skill directory with manifest and three rule files, each containing validation rules, code patterns, and example configurations.

### Validation Checklist

- [ ] Geometry validation catches known invalid WKT examples
- [ ] CRS verification rejects undefined-CRS datasets
- [ ] Schema validator detects missing and mistyped fields
- [ ] All code patterns are copy-paste ready

### Cost Optimization

Keep validation code patterns generic (function-based, not class-heavy). Claude produces more consistent results when the pattern is simple enough to inline.

### Dark Arts Tip

Add a `rules/profiling.md` that generates a data quality report card (HTML or Markdown table) summarizing all checks. When the AI produces this report first, it catches issues before writing downstream analysis code -- preventing wasted iterations.

### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- Validation libraries
- [Data Analysis Prompts](data-analysis-prompts.md) -- QA workflows

### Extensibility

Add `rules/raster-quality.md` for nodata consistency, band count verification, and pixel value range checks.

---

## 5. Python CLI Tools

---

## Prompt 16 -- Typer/Click GIS CLI

### Scenario (实际场景)

You have GIS scripts that you run manually and want to turn them into polished, reusable command-line tools with proper argument parsing, rich terminal output, help text, and error handling.

### Roles

**User** = GIS developer specifying CLI behavior | **AI** = Python CLI tool engineer

### Prompt Template

```text
You are a Python CLI tool engineer for geospatial workflows.

Create a CLI tool using {framework} that performs {operation_description}.

Tool name: {tool_name}

Commands:
  {command_1}: {description_1}
    args: {args_1}
    options: {options_1}
  {command_2}: {description_2}
    args: {args_2}
    options: {options_2}

Example usage:
```bash
{tool_name} {command_1} input.gpkg output.gpkg --crs EPSG:32650
{tool_name} {command_2} ./data/ --pattern "*.shp" --workers 4
```

Requirements:
- {framework} with type hints for auto-generated help
- Rich 13+ for colored output, progress bars, tables
- GeoPandas 0.14+ for spatial I/O
- --verbose flag (debug logging)
- --dry-run flag (show plan without executing)
- --output-format: gpkg, geojson, shp, parquet
- Exit codes: 0=success, 1=error, 2=validation failure
- Structured logging (Python logging module)
- Catch common errors: CRS undefined, file not found, invalid geometry
  -> user-friendly messages with fix suggestions

Project structure:
  src/{package}/
    __init__.py
    cli.py              # Entry point
    core.py             # Business logic (testable)
    validators.py       # Input validation
  tests/
    test_core.py        # pytest unit tests
    test_cli.py         # CliRunner integration tests
  pyproject.toml        # [project.scripts] entry

Installable via: pip install . && {tool_name} --help
```

### Variables to Customize

- `{framework}` -- Typer 0.9+ or Click 8.1+
- `{tool_name}` -- CLI command name
- `{operation_description}` -- what the tool does
- `{command_1}`, `{command_2}` -- subcommand definitions

### Expected Output

Complete Python package with CLI, core logic, validators, tests, and pyproject.toml.

### Validation Checklist

- [ ] `{tool_name} --help` shows all commands and options
- [ ] `--dry-run` prints plan without side effects
- [ ] Exit code is 2 on validation failure (not 1)
- [ ] Rich progress bar displays during long operations

### Cost Optimization

Separate CLI definition (`cli.py`) from business logic (`core.py`). This lets you iterate on the logic with the AI without regenerating the CLI boilerplate each time.

### Dark Arts Tip

Add a `{tool_name} config` subcommand that reads `~/.config/{tool_name}/config.toml` for default CRS, output format, and parallel worker count. Users set preferences once and forget -- and the AI can write config files too.

### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- GeoPandas, Fiona
- [CLI Tools](../tools/cli-tools.md) -- GDAL/OGR CLI patterns

### Extensibility

Add a plugin system using `importlib.metadata.entry_points()` so users can install third-party subcommands as separate packages.

---

## Prompt 17 -- GDAL Wrapper CLI

### Scenario (实际场景)

GDAL/OGR commands are powerful but have complex syntax that is hard to remember. You want a user-friendly Python CLI wrapper that provides intuitive subcommands with sensible defaults, batch processing, and parallel execution.

### Roles

**User** = GIS data engineer tired of GDAL syntax | **AI** = CLI wrapper developer

### Prompt Template

```text
You are a CLI developer wrapping GDAL/OGR for user-friendly access.

Create a Python CLI tool "{tool_name}" that wraps common GDAL/OGR operations.

Subcommands:
  info {file}              # ogrinfo/gdalinfo with pretty-printed Rich output
  convert {in} {out}       # ogr2ogr/gdal_translate with format auto-detection
  reproject {in} {out} --crs EPSG:XXXX
  clip {in} {out} --bbox W,S,E,N | --mask boundary.gpkg
  merge {files...} {out}   # Merge multiple files into one
  validate {file}          # Check geometry validity, CRS, schema

Global options:
  --parallel N             # Process N files concurrently (default: cpu_count)
  --overwrite              # Overwrite existing output
  --quiet / --verbose

Requirements:
- Auto-detect vector vs. raster from file extension / driver
- Batch mode: accept glob patterns ("*.shp")
- Use subprocess for GDAL CLI or GDAL Python bindings (configurable)
- Rich tables for info output, Rich progress for batch operations
- Sensible defaults: GPKG output, preserve CRS, creation options
  (GPKG: fid=fid; GeoTIFF: COMPRESS=DEFLATE, TILED=YES)

Include GDAL creation option presets:
  COG: COMPRESS=DEFLATE, BLOCKSIZE=512, OVERVIEW_RESAMPLING=AVERAGE
  GPKG: fid=fid
  FlatGeobuf: SPATIAL_INDEX=YES
```

### Variables to Customize

- `{tool_name}` -- command name (e.g., `geo`, `gdx`, `gdalx`)
- Additional subcommands (rasterize, polygonize, hillshade)
- Default creation options per format

### Expected Output

Python CLI package with subcommands wrapping GDAL operations, batch support, and format presets.

### Validation Checklist

- [ ] `info` produces readable output for both vector and raster files
- [ ] `convert` auto-detects format and applies correct creation options
- [ ] `--parallel` processes multiple files concurrently
- [ ] `clip` works with both bbox and mask polygon

### Cost Optimization

Focus the prompt on the subcommands you actually need. Each subcommand is independent, so you can generate them incrementally across sessions rather than all at once.

### Dark Arts Tip

Embed a `--explain` flag that prints the equivalent raw GDAL/OGR command instead of executing it. This teaches users the underlying GDAL syntax while still providing the convenience wrapper.

### Related Pages

- [CLI Tools](../tools/cli-tools.md) -- GDAL command reference
- [Python Libraries](../tools/python-libraries.md) -- GDAL Python bindings

### Extensibility

Add a `recipe` subcommand that reads YAML workflow files chaining multiple operations (convert -> reproject -> clip -> validate) in a reproducible pipeline.

---

## Prompt 18 -- Data Pipeline CLI

### Scenario (实际场景)

You need an ETL (Extract, Transform, Load) command-line tool that pulls data from APIs, transforms it spatially, and loads it into PostGIS or GeoPackage. The pipeline must be reproducible and configurable.

### Roles

**User** = Data engineer defining pipeline steps | **AI** = ETL pipeline CLI developer

### Prompt Template

```text
You are an ETL pipeline CLI developer for geospatial data.

Create a CLI tool "{tool_name}" for spatial data pipelines.

Subcommands:
  extract:
    Sources: {source_list}
    Output: raw data to staging directory
    Options: --bbox, --date-range, --api-key (env var fallback)

  transform:
    Operations:
      - Reproject to target CRS
      - Validate and repair geometries
      - Standardize schema (rename fields, cast types)
      - Spatial join / clip to AOI
      - Deduplicate by geometry hash
    Input: staging directory
    Output: processed directory

  load:
    Targets: PostGIS table / GeoPackage / GeoParquet
    Options: --dsn (connection string), --table, --mode (append/replace)
    Features: upsert by primary key, create spatial index

  run:
    Execute full pipeline: extract -> transform -> load
    Config file: pipeline.yaml
    Options: --config, --dry-run, --resume (skip completed steps)

pipeline.yaml example:
```yaml
pipeline:
  name: {pipeline_name}
  extract:
    source: {source_type}
    bbox: [w, s, e, n]
  transform:
    target_crs: EPSG:32650
    validate_geometry: true
    schema:
      rename:
        old_name: new_name
  load:
    target: postgis
    dsn: $POSTGIS_URL
    table: {table_name}
    mode: replace
```

Requirements:
- Idempotent: re-running produces same result
- Checkpoints: save state after each phase
- Logging: structured JSON logs to file
- Error handling: per-feature error isolation, continue on failure
```

### Variables to Customize

- `{source_list}` -- API endpoints, file directories, STAC catalogs
- `{pipeline_name}` -- descriptive name for the pipeline
- Transform operations specific to your data
- Load target and mode

### Expected Output

Python CLI package with extract/transform/load subcommands, YAML config parser, and checkpoint management.

### Validation Checklist

- [ ] `run --dry-run` prints the full plan without executing
- [ ] `run --resume` skips completed phases
- [ ] Per-feature errors are logged but do not stop the pipeline
- [ ] Loaded data has spatial index in PostGIS

### Cost Optimization

Generate the `pipeline.yaml` schema and `run` subcommand first, then iterate on individual phase implementations. This avoids regenerating the entire tool when tweaking one phase.

### Dark Arts Tip

Add a `{tool_name} diff` subcommand that compares the current source data against the loaded target and shows what would change (new features, modified features, deleted features). This is invaluable for debugging data drift.

### Related Pages

- [Python Libraries](../tools/python-libraries.md) -- GeoPandas, SQLAlchemy
- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS loading

### Extensibility

Support pipeline composition: reference other pipeline YAML files as steps, enabling modular, reusable pipeline components.

---

## 6. VS Code Extensions

---

## Prompt 19 -- GeoJSON Preview Extension

### Scenario (实际场景)

You want a VS Code extension that renders GeoJSON files on an interactive Leaflet map in a webview panel, with syntax validation, coordinate display, and feature inspection on hover.

### Roles

**User** = GIS developer wanting in-editor map preview | **AI** = VS Code extension developer

### Prompt Template

```text
You are a VS Code extension developer for geospatial file support.

Create a VS Code extension: "{extension_name}"
Display name: "GeoJSON Preview"

Features:
1. Webview panel with Leaflet map rendering the active GeoJSON file
   - Auto-update on file save
   - Fit map to feature bounds
   - Feature popup on click (show all properties)
   - Coordinate display in status bar (cursor position on map)

2. GeoJSON syntax validation
   - Validate against RFC 7946
   - Inline diagnostics (red squiggles) for:
     - Invalid geometry types
     - Coordinates out of range (lon: -180..180, lat: -90..90)
     - Missing "type" field
     - Antimeridian crossing issues
   - Quick fixes: swap lat/lon, add missing "type"

3. Feature count and bbox in status bar
4. Command: "GeoJSON: Copy as WKT" -- convert selected feature
5. Command: "GeoJSON: Reformat" -- pretty-print with sorted keys

Extension structure:
  src/
    extension.ts          # activate/deactivate
    preview-panel.ts      # Webview provider
    validator.ts          # GeoJSON validation
    commands.ts           # Command handlers
  media/
    preview.html          # Webview template
    preview.js            # Leaflet initialization
    preview.css
  package.json            # contributes: commands, languages, menus
  tsconfig.json
  esbuild.config.js       # Bundler

VS Code engine: ^{vscode_version}
Bundler: esbuild
Language: TypeScript
```

### Variables to Customize

- `{extension_name}` -- npm package name (e.g., `geojson-preview`)
- `{vscode_version}` -- minimum VS Code version (e.g., `1.85.0`)
- Additional validation rules
- Base map tile provider (OSM, Carto, custom)

### Expected Output

Complete VS Code extension project with webview panel, validator, commands, and package.json configuration.

### Validation Checklist

- [ ] Webview renders GeoJSON features on map tiles
- [ ] Validation catches coordinates outside WGS84 range
- [ ] Map auto-updates when the file is saved
- [ ] Extension activates only for `.geojson` and `.json` files with GeoJSON content

### Cost Optimization

Use `esbuild` for bundling (not webpack) -- faster builds and simpler config. The webview HTML should inline Leaflet from a CDN at a pinned version to avoid bundling it.

### Dark Arts Tip

Register a `CustomEditorProvider` instead of a plain webview command. This makes the map appear as a side-by-side editor tab that stays in sync with the text editor -- a much more polished UX than a standalone panel.

### Related Pages

- [2D Mapping Libraries](../js-bindbox/2d-mapping.md) -- Leaflet reference
- [Map Styling Prompts](map-styling-prompts.md)

### Extensibility

Add support for `.fgb` (FlatGeobuf) and `.parquet` (GeoParquet) preview using browser-compatible parsers (flatgeobuf.js, parquet-wasm).

---

## Prompt 20 -- Spatial SQL Extension

### Scenario (实际场景)

You want a VS Code extension that provides PostGIS autocompletion, function documentation on hover, query execution with result preview, and a map panel that renders query results as GeoJSON.

### Roles

**User** = Spatial SQL developer wanting in-editor intelligence | **AI** = VS Code Language Server developer

### Prompt Template

```text
You are a VS Code extension developer with Language Server Protocol expertise.

Create a VS Code extension: "{extension_name}"
Display name: "Spatial SQL"

Features:
1. PostGIS function autocompletion
   - All ST_* functions with signatures
   - Trigger on "ST_" prefix
   - Include function description in completion item detail
   - Parameter snippets with placeholders

2. Hover documentation
   - Hover over any PostGIS function -> show signature, description,
     return type, and a short example
   - Data source: embedded JSON catalog of PostGIS functions

3. Query execution
   - Command: "Spatial SQL: Run Query" (Ctrl+Shift+Enter)
   - Connect to PostGIS (connection settings in VS Code config)
   - Display results in a table (webview)
   - If result contains a geometry column, show "View on Map" button

4. Map result viewer
   - Webview with Leaflet map
   - Render query results as GeoJSON layer
   - Feature popup with attribute table
   - Support for point, line, polygon, multi-* types

Configuration (contributes.configuration):
  spatialSql.connections: array of {name, host, port, database, user}
  spatialSql.defaultConnection: string
  spatialSql.maxRows: number (default 1000)

Extension structure:
  src/
    extension.ts
    language-server/
      server.ts            # LSP server
      completions.ts       # PostGIS function completions
      hover.ts             # Hover provider
    query-runner.ts        # pg client, query execution
    map-panel.ts           # Leaflet webview for results
    postgis-catalog.json   # Function reference data
  media/
    map.html, map.js, map.css
  package.json
  tsconfig.json
```

### Variables to Customize

- `{extension_name}` -- package name
- Database connection method (settings, .env, prompted)
- PostGIS function catalog scope (all functions vs. common subset)
- Additional features (query formatting, EXPLAIN visualization)

### Expected Output

VS Code extension with LSP server for PostGIS completions, hover docs, query execution, and a map result viewer.

### Validation Checklist

- [ ] `ST_` prefix triggers completion with function signatures
- [ ] Hover shows function documentation with examples
- [ ] Query results render in a table webview
- [ ] Geometry results display on a Leaflet map

### Cost Optimization

Embed the PostGIS function catalog as a static JSON file rather than querying the database for function metadata. This avoids a database round-trip on every keystroke and works even when disconnected.

### Dark Arts Tip

Parse the SQL query AST (using `pgsql-ast-parser` or similar) to detect which tables are referenced, then provide column-name autocompletion by querying `information_schema.columns` for those specific tables. This bridges the gap between generic SQL extensions and a true spatial IDE.

### Related Pages

- [Spatial Databases](../tools/spatial-databases.md) -- PostGIS reference
- [Data Analysis Prompts](data-analysis-prompts.md) -- SQL analysis patterns

### Extensibility

Add support for DuckDB Spatial and SpatiaLite by abstracting the database adapter. The PostGIS function catalog can be swapped for a DuckDB spatial function catalog with the same completion interface.

---

## 7. ArcGIS Python Toolbox

---

## Prompt 21 -- Python Toolbox (.pyt)

### Scenario (实际场景)

You need a custom ArcGIS Python Toolbox (.pyt) with proper parameter validation, progress reporting, scratch workspace management, and descriptive messaging. The toolbox must work in both ArcGIS Pro UI and command-line execution.

### Roles

**User** = ArcGIS developer defining tool requirements | **AI** = ArcPy toolbox specialist

### Prompt Template

```text
You are an ArcPy toolbox specialist for ArcGIS Pro {arcgis_version}+.

Create a Python Toolbox (.pyt) called "{toolbox_name}" containing
a tool called "{tool_name}" that {tool_description}.

Parameters:
1. {param_name}: {type} ({direction}, {required})
   - Filter: {filter_description}
   - Default: {default_value}
2. {param_name}: {type} ({direction}, {required})
[... repeat]

Implementation requirements:
- getParameterInfo(): define all parameters with proper datatype,
  direction, parameterType, filter
- updateParameters(): dynamic parameter behavior
  (e.g., populate field list when layer changes)
- updateMessages(): custom validation messages
  (e.g., "CRS mismatch between input layers")
- execute():
  1. arcpy.SetProgressor("step", "Processing...", 0, {total}, 1)
  2. Write intermediate data to arcpy.env.scratchGDB
  3. {processing_steps}
  4. arcpy.AddMessage() for info, arcpy.AddWarning() for issues
  5. Clean up scratch data in a finally block
  6. Set output parameter for model chaining

Additional:
- isLicensed(): check for required license level / extensions
- Support isBackground = True if thread-safe
- Handle SR mismatches: reproject inputs to match before processing
- Include XML item description: summary, usage, param descriptions

Deliver a single .pyt file with Toolbox class and Tool class(es).
```

### Variables to Customize

- `{toolbox_name}` / `{tool_name}` -- names
- `{tool_description}` -- what the tool does
- Parameter definitions (name, type, direction, filters)
- `{processing_steps}` -- arcpy operations in execute()

### Expected Output

A single `.pyt` file with complete Toolbox and Tool classes, including parameter validation, progress reporting, and cleanup.

### Validation Checklist

- [ ] Toolbox loads in ArcGIS Pro Catalog pane without errors
- [ ] Parameter validation catches CRS mismatches before execution
- [ ] Progress bar advances during processing
- [ ] Scratch data is cleaned up even if the tool fails
- [ ] Tool works in ModelBuilder (output parameter is chainable)

### Cost Optimization

ArcPy toolboxes follow a rigid class structure. Provide the parameter definitions precisely (name, arcpy datatype string, direction) to avoid ambiguity. One-shot generation works well for this pattern.

### Dark Arts Tip

Use `arcpy.env.autoCancelling = False` and check `arcpy.env.isCancelled` manually inside your loop. This gives you control over when cancellation happens, letting you clean up properly rather than leaving corrupt partial outputs.

### Related Pages

- [Desktop GIS Tools](../tools/desktop-gis.md) -- ArcGIS ecosystem
- [Data Analysis Prompts](data-analysis-prompts.md)

### Extensibility

Add multiple Tool classes to the same .pyt file, organized into tool families. Use a shared utility module imported alongside the .pyt for common functions.

---

## Prompt 22 -- Model Builder to Python Migration

### Scenario (实际场景)

You have an existing ArcGIS Model Builder workflow that you need to convert to a maintainable Python script or Python Toolbox. The migration should add error handling, logging, parameterization, and reproducibility that Model Builder lacks.

### Roles

**User** = GIS analyst with a Model Builder workflow | **AI** = ArcPy migration specialist

### Prompt Template

```text
You are an ArcPy migration specialist converting Model Builder
workflows to production Python scripts.

My Model Builder workflow performs these steps:
1. {step_1_tool}: {step_1_params}
2. {step_2_tool}: {step_2_params}
3. {step_3_tool}: {step_3_params}
[... list all steps]

Model inputs: {input_list}
Model outputs: {output_list}
Environment settings: {env_settings}

Convert to {output_format}:
  A) Standalone Python script with argparse
  B) Python Toolbox (.pyt)
  C) Both

Add these improvements:
1. Error handling per step:
   - Try/except around each arcpy tool call
   - Log error and either continue or abort (configurable)
   - Capture arcpy.GetMessages() on failure
2. Progress reporting (arcpy.SetProgressor or print)
3. Parameterize all hardcoded values (paths, field names, thresholds)
4. Logging to file (Python logging module, not just AddMessage)
5. Input validation before processing starts:
   - Check files exist
   - Check CRS compatibility
   - Check required fields present
6. Intermediate outputs to scratch workspace
7. Cleanup on success and failure (finally block)
8. Elapsed time reporting per step and total

If Option A: include argparse with --help documentation.
If Option B: include full parameter definitions with filters.
```

### Variables to Customize

- Model steps with arcpy tool names and parameters
- Input/output definitions
- Environment settings (CRS, extent, cell size)
- `{output_format}` -- script, toolbox, or both

### Expected Output

Python script and/or .pyt file with all improvements over the original Model Builder workflow, including error handling, logging, and parameterization.

### Validation Checklist

- [ ] All Model Builder steps are represented in the Python code
- [ ] Error handling wraps each arcpy tool call
- [ ] Script produces identical output to the original model on test data
- [ ] Logging captures step timing and any warnings

### Cost Optimization

Export the model as a Python script first (`Model > Export > To Python Script`) and provide that export to the AI. This gives it exact tool names and parameter values, eliminating guesswork and producing a more accurate migration.

### Dark Arts Tip

Add a `--validate-only` flag that runs all input validation and environment checks without executing any arcpy tools. This lets you verify the script configuration on a remote server (where test runs are slow) before committing to a full execution.

### Related Pages

- [Desktop GIS Tools](../tools/desktop-gis.md) -- ArcGIS Pro scripting
- [Automation Workflows](automation-workflows.md)

### Extensibility

Wrap the migrated script as a step in an `arcpy.da.Walk()`-based batch processor that applies the workflow to every geodatabase in a directory tree.

---

[Back to AI Prompts](README.md) &middot; [Back to Main README](../README.md)
