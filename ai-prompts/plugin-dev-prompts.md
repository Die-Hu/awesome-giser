# Plugin & Extension Development Prompts

> Prompt templates for developing QGIS plugins, ArcGIS Python toolboxes, Leaflet/MapLibre plugins, CLI tools, and GIS-related VS Code extensions.

> **Quick Picks**
> - 🏆 **Most Useful**: [QGIS Processing Algorithm](#prompt-4--qgis-processing-algorithm) — the standard way to add custom tools to QGIS, works in GUI and CLI
> - 🚀 **Time Saver**: [Python CLI Tool for GIS](#prompt-3--python-cli-tool-for-gis-operations) — turn any GIS script into a polished command-line tool in minutes
> - 🆕 **Cutting Edge**: [MapLibre GL JS Custom Control](#prompt-3--maplibre-gl-js-custom-control) — build reusable UI controls for modern web maps

---

## Table of Contents

- [QGIS Plugin Development](#qgis-plugin-development)
- [ArcGIS Python Toolbox](#arcgis-python-toolbox)
- [Leaflet / MapLibre Plugin](#leaflet--maplibre-plugin)
- [Python CLI Tools](#python-cli-tools)
- [VS Code Extension for GIS](#vs-code-extension-for-gis)

---

## QGIS Plugin Development

### Prompt 1 — Plugin Scaffolding

**Context:** You want to create a new QGIS plugin from scratch with proper project structure.

**Template:**

```
Create a complete QGIS plugin scaffold for a plugin called "[plugin name]" that [brief description of functionality].

The plugin should:
- Target QGIS [version, e.g., 3.34+] and PyQGIS API
- Include: __init__.py, main plugin class, dialog class with .ui file, metadata.txt, resources.qrc
- Use the standard QGIS plugin directory structure
- Add a toolbar button with a custom icon placeholder
- Include a dialog (QDialog) with [describe UI elements: dropdown, file picker, checkbox, etc.]
- Implement the core logic in a separate module (not in the dialog class)
- Include proper cleanup in unload()
- Add i18n support (.ts file structure)

For metadata.txt, include:
- name, qgisMinimumVersion, description, version, author, email
- category: [Raster / Vector / Analysis / Web]
- tags: [relevant tags]
- repository and tracker URLs (placeholder)

Also create a Makefile with targets for: compile resources, deploy to QGIS plugin directory, clean, zip for upload.
```

**Variables to customize:**
- Plugin name and description
- QGIS version target
- UI elements needed
- Plugin category and tags

**Expected output format:** Multiple files with the complete plugin directory structure.

---

### Prompt 2 — Plugin Feature Implementation

**Context:** You have an existing QGIS plugin and want to add a specific feature.

**Template:**

```
I have a QGIS plugin that [current functionality]. I want to add the following feature:
[describe feature, e.g., "batch reproject all loaded vector layers to a user-selected CRS"].

Current plugin structure:
- Main class: [class name] in [file]
- Dialog class: [class name] in [file]
- Processing logic in: [file]

Write the implementation including:
1. UI additions to the dialog (describe the widgets needed: CRS selector, layer list, progress bar)
2. Backend logic using PyQGIS API
3. Progress reporting via QgsMessageBar or a progress dialog
4. Error handling with user-friendly error messages
5. Undo support if applicable
6. Unit tests using pytest-qgis

Use QgsProcessingAlgorithm if this would work better as a Processing provider.
```

**Variables to customize:**
- Existing plugin structure
- Feature description
- UI requirements
- Whether to use Processing framework

**Expected output format:** Python code for the new feature, dialog modifications, and tests.

---

### Prompt 3 — Debugging a QGIS Plugin

**Context:** Your plugin has a bug and you need help diagnosing and fixing it.

**Template:**

```
My QGIS plugin "[name]" has the following issue:
[Describe the bug — what happens, what should happen, when it occurs]

Error from QGIS Python console / log:
```
[paste traceback or error message]
```

Relevant code:
```python
[paste the relevant section of your plugin code]
```

QGIS version: [version], OS: [Windows/macOS/Linux], Python: [version]

Help me:
1. Diagnose the root cause
2. Suggest a fix with code
3. Explain why the fix works
4. Suggest a way to prevent this class of bug in the future
5. If it is a common PyQGIS pitfall, explain the underlying API behavior
```

**Variables to customize:**
- Bug description and symptoms
- Error traceback
- Relevant code
- Environment details

**Expected output format:** Diagnosis, fixed code, and preventive guidance.

---

### Prompt 4 — QGIS Processing Algorithm

**Context:** You want to create a reusable geoprocessing tool that integrates with the QGIS Processing framework. Processing algorithms appear in the toolbox, can be used in graphical models, run from the command line via `qgis_process`, and support batch execution out of the box.

**Template:**

```
Create a QGIS Processing algorithm (QgsProcessingAlgorithm subclass) for QGIS 3.34+ that
[description, e.g., "calculates a weighted accessibility score for each polygon based on
proximity to multiple point feature types (schools, hospitals, transit stops)"].

Algorithm metadata:
- Name: [snake_case_name]
- Display name: "[Human Readable Name]"
- Group: "[group name, e.g., Accessibility Analysis]"
- Short help string: "[one-sentence description]"
- Tags: [list tags]

Parameters (define in initAlgorithm):
- INPUT_POLYGONS: QgsProcessingParameterVectorLayer (Polygon)
- INPUT_POINTS: QgsProcessingParameterMultipleLayers (Point) — multiple point layers
- WEIGHTS: QgsProcessingParameterString — comma-separated weights for each point layer
- DISTANCE_DECAY: QgsProcessingParameterEnum — ["Linear", "Inverse Distance", "Gaussian"]
- MAX_DISTANCE: QgsProcessingParameterDistance — search radius in meters
- OUTPUT: QgsProcessingParameterVectorDestination

Processing logic (in processAlgorithm):
1. Validate that WEIGHTS count matches INPUT_POINTS count
2. For each polygon, for each point layer:
   - Find all points within MAX_DISTANCE of the polygon centroid
   - Apply the selected distance decay function
   - Multiply by the corresponding weight
3. Sum weighted scores into a composite accessibility index
4. Normalize to 0-100 range
5. Add the score as a new attribute column
6. Report progress using feedback.setProgress() and feedback.pushInfo()
7. Check for feedback.isCanceled() in the inner loop

Include:
- Proper type hints (Python 3.10+ style)
- A createInstance() method returning a new instance of the class
- A provider plugin that registers this algorithm as a Processing provider
- The provider's __init__.py, provider class, and metadata.txt
- An algorithm help string with parameter descriptions
- A pytest test using QgsProcessingFeedback and a fixture GeoPackage

The algorithm should be installable as a QGIS plugin and also callable via:
  qgis_process run "provider:algorithm_name" -- INPUT_POLYGONS=... INPUT_POINTS=... OUTPUT=...
```

**Variables to customize:**
- Algorithm functionality and parameters
- Distance decay options
- Provider group and metadata
- Test data requirements

**Expected output format:** Complete Processing provider plugin with algorithm, provider class, metadata, and tests.

---

## ArcGIS Python Toolbox

### Prompt 1 — Script Tool Scaffolding

**Context:** You want to create a custom ArcGIS Python Toolbox (.pyt) with one or more script tools.

**Template:**

```
Create an ArcGIS Python Toolbox (.pyt file) called "[toolbox name]" containing a tool called "[tool name]"
that [description of what the tool does, e.g., "calculates viewshed from multiple observer points
and merges the results into a single visibility raster"].

Parameters:
1. [Parameter name]: [type — Feature Layer / Raster / Field / String / Long / Boolean], [direction: Input/Output], [required/optional]
2. [Parameter name]: [type], [direction], [required/optional]
[... repeat]

The tool should:
- Validate parameters in updateParameters() and updateMessages()
- Use arcpy.SetProgressor for progress reporting
- Write intermediate results to arcpy.env.scratchGDB
- Handle spatial reference mismatches (reproject if needed)
- Include descriptive messages via arcpy.AddMessage / arcpy.AddWarning
- Clean up temporary data in a finally block
- Support running in background (isBackground = True if appropriate)

Include the item description XML for the tool (summary, usage, parameter descriptions).
```

**Variables to customize:**
- Toolbox and tool names
- Parameter definitions
- Processing logic description
- Background execution support

**Expected output format:** Complete .pyt file with Toolbox and Tool classes.

---

### Prompt 2 — Migrating a Model Builder Workflow to Python

**Context:** You have an existing Model Builder model and want to convert it to a reusable Python script or toolbox.

**Template:**

```
I have a Model Builder workflow in ArcGIS Pro that does the following steps:
1. [Step 1: tool name and parameters]
2. [Step 2: tool name and parameters]
3. [Step 3: tool name and parameters]
[... repeat all steps]

The model uses these input parameters: [list]
The model produces these outputs: [list]
Environment settings: [CRS, extent, cell size, etc.]

Convert this into:
- Option A: A standalone arcpy Python script with argument parsing (argparse)
- Option B: A Python Toolbox (.pyt) with proper parameter definitions

Add the following improvements over the original model:
- Error handling for each step (log and continue or log and stop)
- Progress reporting
- Input validation
- Parameterize any hardcoded values
- Add logging to a file
```

**Variables to customize:**
- Model steps with tool names and parameters
- Input/output parameters
- Environment settings
- Preferred output format (script vs toolbox)

**Expected output format:** Python script or .pyt file with improvements.

---

## Leaflet / MapLibre Plugin

### Prompt 1 — Custom Leaflet Control Plugin

**Context:** You want to create a reusable Leaflet plugin that adds a custom map control.

**Template:**

```
Write a Leaflet plugin (ES module) that adds a [control type, e.g., "coordinate display",
"layer opacity slider", "measurement tool", "geocoder search bar"] to the map.

Plugin name: L.Control.[Name]
Behavior: [describe in detail what the control does]

Requirements:
- Extend L.Control with proper initialize(), onAdd(), onRemove()
- Position: [topright / topleft / bottomright / bottomleft]
- Responsive: work on both desktop and mobile
- Accessible: keyboard navigable, ARIA labels
- Configurable via options: [list options with defaults]
- Emit custom events: [list events, e.g., "measurement:complete"]
- No external dependencies beyond Leaflet
- Include CSS (embedded or separate file)
- Include TypeScript type declarations (.d.ts)

Also provide:
- A demo HTML page showing the plugin in use
- npm package.json for publishing
- Usage documentation with code examples
```

**Variables to customize:**
- Control type and behavior
- Configuration options
- Events emitted
- Position

**Expected output format:** JavaScript plugin file, CSS, TypeScript types, demo page, and package.json.

---

### Prompt 2 — MapLibre Custom Layer Plugin

**Context:** You want to create a MapLibre GL JS plugin that renders a custom layer type.

**Template:**

```
Write a MapLibre GL JS plugin that adds a [layer type, e.g., "animated flow lines",
"heatmap with time slider", "3D extruded buildings with shadows"] layer.

Plugin API:
```js
const layer = new MaplibreGIS.[PluginName](options);
map.addControl(layer); // or map.addLayer(layer.getLayerSpec());
```

Data source: [GeoJSON / vector tiles / custom API endpoint]
Options:
- [option 1]: [type, default, description]
- [option 2]: [type, default, description]

Features:
- [describe rendering behavior]
- Smooth animations at 60fps using requestAnimationFrame
- Efficient rendering for up to [N] features
- Responds to map zoom and pan events
- Proper cleanup on remove

Include a complete working example with sample data.
```

**Variables to customize:**
- Layer type and visual behavior
- Data source type
- Configuration options
- Performance target (max feature count)

**Expected output format:** JavaScript plugin, example HTML page, and sample GeoJSON data.

---

### Prompt 3 — MapLibre GL JS Custom Control

**Context:** You want to create a reusable MapLibre GL JS control (TypeScript) that adds spatial analysis or UI functionality to a web map.

**Template:**

```
Write a MapLibre GL JS custom control in TypeScript that implements
[control type, e.g., "a measurement tool" / "a spatial bookmarks manager" / "a layer switcher with opacity sliders"
/ "a coordinate format converter (DD, DMS, MGRS, UTM)" / "a draw-and-query tool"].

Control behavior:
[describe the full UX flow, e.g., "User clicks the control button to activate measurement mode.
Click on the map to add vertices. Double-click to finish. Display the total distance (geodesic)
in meters/km and the area (if polygon) in m²/hectares. Show running totals as the user draws.
Press Escape to cancel. Support undo (Ctrl+Z) for the last vertex."]

Requirements:
- Implement IControl interface: onAdd(map), onRemove(map), getDefaultPosition()
- Position: [topright / topleft / bottomright / bottomleft]
- TypeScript with strict mode, full type annotations
- Emit custom events using map.fire():
  - '[control]:start' — when the tool is activated
  - '[control]:update' — on each interaction (with interim results)
  - '[control]:complete' — when a measurement/action is finished (with final results)
- Support configuration via constructor options:
  ```typescript
  interface [ControlName]Options {
    units?: 'metric' | 'imperial';
    lineColor?: string;
    fillColor?: string;
    precision?: number;
    [additional options]
  }
  ```
- CSS: include a self-contained CSS stylesheet (no external dependencies)
- Accessibility: keyboard navigable, ARIA labels on buttons, focus management
- Mobile: touch events, appropriately sized tap targets (44px minimum)
- Cleanup: remove all event listeners and DOM elements in onRemove()

Project structure:
- src/index.ts — control class
- src/styles.css — scoped CSS
- dist/ — compiled JS + CSS
- package.json — with "main", "module", "types" fields
- tsconfig.json — targeting ES2020, MapLibre GL JS 4.x types
- A demo/index.html showing the control in use with a real map

Usage example in the README:
```typescript
import { [ControlName] } from '[package-name]';
const map = new maplibregl.Map({ ... });
map.addControl(new [ControlName]({ units: 'metric' }), 'top-right');
map.on('[control]:complete', (e) => console.log(e.detail));
```
```

**Variables to customize:**
- Control type and behavior
- Events emitted
- Configuration options
- MapLibre GL JS version target

**Expected output format:** TypeScript source, CSS, compiled output, package.json, demo page.

---

## Python CLI Tools

### Prompt 3 — Python CLI Tool for GIS Operations

**Context:** You have GIS scripts that you run manually and want to turn them into polished, reusable command-line tools with proper argument parsing, help text, and error handling. Click or Typer are the modern Python CLI frameworks.

**Template:**

```
Create a Python CLI tool using [Typer (0.9+) / Click (8.1+)] that performs
[operation, e.g., "batch reprojection of vector files" / "raster clipping to boundaries" /
"spatial join between two datasets" / "GeoJSON to GeoPackage conversion with validation"].

Tool name: [name, e.g., "gis-toolkit" or "spatial-convert"]

Commands (if multi-command tool):
1. [command name]: [description]
   - Arguments: [list positional args]
   - Options: [list optional flags with defaults]
2. [command name]: [description]
   - Arguments: ...
   - Options: ...
[... or single command if simple]

Example CLI usage:
```bash
# Single command example
spatial-convert reproject input.gpkg output.gpkg --from-crs EPSG:4326 --to-crs EPSG:32650

# Batch example
spatial-convert batch-reproject ./input_dir/ ./output_dir/ --to-crs EPSG:32650 --format gpkg --workers 4

# Validation example
spatial-convert validate input.gpkg --rules rules.yaml --report report.json
```

Requirements:
- Use Typer with type hints for auto-generated help text (or Click with decorators)
- Rich (13.0+) for colored terminal output, progress bars, and tables
- GeoPandas (0.14+) or Fiona (1.9+) for spatial I/O
- Structured logging using Python logging module (--verbose flag for debug level)
- Error handling: catch common GIS errors (CRS undefined, file not found, invalid geometry)
  and display user-friendly messages with suggestions
- Support both file paths and glob patterns as input (e.g., "*.shp")
- --dry-run flag that shows what would happen without executing
- --output-format option: gpkg, geojson, shp, parquet
- Exit codes: 0 success, 1 error, 2 validation failure

Project structure:
- src/[package_name]/__init__.py
- src/[package_name]/cli.py — CLI entry point
- src/[package_name]/core.py — business logic (testable without CLI)
- src/[package_name]/validators.py — validation functions
- tests/test_core.py — pytest tests for core logic
- tests/test_cli.py — CLI integration tests using CliRunner
- pyproject.toml — with [project.scripts] entry point: [tool-name] = "[package]:cli.app"
- README.md with installation and usage examples

The tool should be installable via: pip install . && [tool-name] --help
```

**Variables to customize:**
- CLI framework (Typer or Click)
- Tool name and commands
- GIS operations performed
- Input/output format options
- Project structure preferences

**Expected output format:** Complete Python package with CLI, core logic, tests, and pyproject.toml.

---

## VS Code Extension for GIS

### Prompt 1 — GIS File Syntax Support

**Context:** You want to create a VS Code extension that provides syntax highlighting, validation, or previewing for a GIS file format.

**Template:**

```
Create a VS Code extension that provides support for [file format, e.g., SLD / QML / GeoJSON / Mapbox Style JSON].

Features:
1. Syntax highlighting using a TextMate grammar (.tmLanguage.json)
2. Schema validation (report errors inline) using [JSON Schema / XML Schema]
3. Autocompletion for [keywords / property names / CRS codes / color values]
4. Hover information for [properties / functions / CRS definitions]
5. A preview panel that renders [the styled map / the geometry on a simple map]
6. Snippets for common patterns (e.g., a basic SLD rule, a Mapbox layer definition)

Extension structure:
- package.json with contributes: languages, grammars, jsonValidation, commands, snippets
- src/extension.ts with activate/deactivate
- syntaxes/ for grammar files
- schemas/ for validation schemas
- snippets/ for snippet definitions

Target VS Code engine: ^[version, e.g., 1.85.0]
Language: TypeScript
Bundler: esbuild
```

**Variables to customize:**
- Target file format
- Feature set (syntax, validation, completion, preview)
- VS Code engine version

**Expected output format:** Complete extension project structure with all source files.

---

### Prompt 2 — GIS Productivity Extension

**Context:** You want to build a VS Code extension that helps GIS developers be more productive.

**Template:**

```
Create a VS Code extension called "[name]" that helps GIS developers by providing:

1. [Feature: e.g., "CRS lookup command — search EPSG codes and insert WKT/PROJ strings"]
2. [Feature: e.g., "GeoJSON preview — render GeoJSON files on an interactive map in a webview panel"]
3. [Feature: e.g., "Coordinate conversion — select coordinates in the editor and convert between formats"]
4. [Feature: e.g., "Quick documentation — hover over PyQGIS/ArcPy functions to see API docs"]

For each feature, implement:
- A command registered in package.json
- Keyboard shortcut (customizable)
- Status bar item if applicable
- Settings in contributes.configuration

Use the VS Code Webview API for any map rendering (embed Leaflet in the webview).
Use the VS Code Language Server Protocol for hover/completion if needed.

Include test files using @vscode/test-electron.
```

**Variables to customize:**
- Extension name and feature list
- Target GIS libraries for documentation
- Keyboard shortcuts

**Expected output format:** TypeScript extension project with all features implemented.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
