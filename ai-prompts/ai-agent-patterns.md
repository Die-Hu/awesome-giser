# AI Agent Patterns for GIS

> Architectural patterns, prompt templates, and orchestration strategies for building AI-powered GIS workflows using Claude Code, MCP servers, Skills, and multi-agent pipelines. Every pattern is designed to be immediately implementable by a GIS professional.

> **Quick Picks**
> - Best Starting Point: [P1: CLAUDE.md Project Configuration](#p1--claudemd-project-configuration-for-gis) -- set up any GIS project for AI-assisted development
> - Most Powerful: [P15: Data-to-Insight Pipeline](#p15--data-to-insight-pipeline) -- five agents from raw data to client deliverable
> - Biggest ROI: [P6: PostGIS MCP Server](#p6--postgis-mcp-server-architecture) -- give your AI direct access to spatial databases

---

## Table of Contents

- [1. Claude Code for GIS](#1-claude-code-for-gis)
  - [P1: CLAUDE.md Project Configuration](#p1--claudemd-project-configuration-for-gis)
  - [P2: Claude Code Hooks for GIS](#p2--claude-code-hooks-for-gis)
  - [P3: Multi-Turn GIS Analysis Session](#p3--multi-turn-gis-analysis-session)
  - [P4: Claude Code + Git Workflow](#p4--claude-code--git-workflow)
  - [P5: Debugging GIS Code with Claude Code](#p5--debugging-gis-code-with-claude-code)
- [2. MCP Server Patterns for GIS](#2-mcp-server-patterns-for-gis)
  - [P6: PostGIS MCP Server Architecture](#p6--postgis-mcp-server-architecture)
  - [P7: STAC Catalog MCP Server](#p7--stac-catalog-mcp-server)
  - [P8: GDAL Processing MCP Server](#p8--gdal-processing-mcp-server)
  - [P9: Geocoding and Routing MCP Server](#p9--geocoding-and-routing-mcp-server)
  - [P10: Tile Server MCP Server](#p10--tile-server-mcp-server)
- [3. Claude Skills for GIS](#3-claude-skills-for-gis)
  - [P11: Spatial Analysis Skill](#p11--spatial-analysis-skill)
  - [P12: Cartographic Design Skill](#p12--cartographic-design-skill)
  - [P13: Data Engineering Skill](#p13--data-engineering-skill)
  - [P14: Remote Sensing Skill](#p14--remote-sensing-skill)
- [4. Multi-Agent Orchestration](#4-multi-agent-orchestration)
  - [P15: Data-to-Insight Pipeline](#p15--data-to-insight-pipeline)
  - [P16: Parallel Analysis Team](#p16--parallel-analysis-team)
  - [P17: Review and QA Agent](#p17--review-and-qa-agent)
  - [P18: Client Delivery Agent Team](#p18--client-delivery-agent-team)
- [5. Cost Optimization Patterns](#5-cost-optimization-patterns)
  - [P19: Model Tier Strategy for GIS](#p19--model-tier-strategy-for-gis)
  - [P20: Context Window Management](#p20--context-window-management)
- [6. Integration Patterns](#6-integration-patterns)
  - [P21: AI + Web Map Application](#p21--ai--web-map-application)
  - [P22: AI-Powered GIS Dashboard](#p22--ai-powered-gis-dashboard)

---

## How This Page Works

Each pattern follows a consistent structure:

| Section | Purpose |
|---------|---------|
| **Scenario (shiji changijng)** | The real-world situation where this pattern applies |
| **Roles** | Who is involved -- human, AI agent, or system |
| **Prompt Template or Pattern** | The actual template, configuration, or architecture |
| **Variables to Customize** | What you change for your specific project |
| **Expected Output** | What a successful application produces |
| **Validation Checklist** | How to verify the result is correct |
| **Cost Optimization** | How to minimize token and compute costs |
| **Related Pages** | Cross-references to other awesome-giser resources |
| **Extensibility** | How to extend the pattern for advanced use cases |

---

## 1. Claude Code for GIS

Claude Code is Anthropic's agentic CLI for Claude. It operates directly in your terminal, reads and writes files, runs commands, and manages git workflows. For GIS professionals, it transforms how you build spatial analysis pipelines, debug GDAL/GeoPandas code, and deliver client projects.

The five patterns in this section configure Claude Code as a GIS-native development environment.

---

### P1 -- CLAUDE.md Project Configuration for GIS

**Scenario (shiji changjing):** You are starting a new GIS project -- flood risk analysis, land cover classification, urban accessibility modeling -- and want every Claude Code interaction to produce CRS-aware, validation-ready, correctly-formatted code from the first prompt.

**Roles:**
- Human: GIS project lead who defines project standards
- Claude Code: development assistant that follows the CLAUDE.md configuration

**Pattern: Complete CLAUDE.md File**

Place this file at the root of your GIS project repository. Claude Code reads it automatically on every session start.

```yaml
# CLAUDE.md

## Project Overview
# [PROJECT_NAME]: [ONE_LINE_DESCRIPTION]
# Example: Shenzhen Flood Risk Assessment: Multi-criteria analysis combining
# DEM, land cover, rainfall, and drainage network data to produce ward-level
# flood risk scores for the Shenzhen Municipal Planning Bureau.

## Directory Structure
# src/           -- Python source modules
# notebooks/     -- Jupyter notebooks for exploration (numbered: 01_, 02_, ...)
# data/
#   raw/         -- Original data files (NEVER modify)
#   interim/     -- Intermediate processing results
#   processed/   -- Final analysis-ready datasets
#   external/    -- Third-party reference data
# outputs/
#   maps/        -- Generated map images and interactive HTML
#   reports/     -- Generated PDF/DOCX reports
#   tables/      -- Summary statistics CSV/XLSX
# tests/         -- pytest tests mirroring src/ structure
# configs/       -- YAML configuration files for analysis parameters
# docs/          -- Project documentation

## Coding Standards

### CRS-First Rule
# Every function that accepts or returns spatial data MUST:
# 1. Document the expected CRS in its docstring
# 2. Validate CRS on input (assert or raise ValueError)
# 3. Never silently reproject -- always log CRS transformations
# 4. Use pyproj.CRS objects for comparison, never string matching

### Python Style
# - Use pathlib.Path for ALL file paths, never os.path
# - Use context managers for file I/O and database connections
# - Type hints on all function signatures (use geopandas.GeoDataFrame, not Any)
# - Docstrings: Google style with Args, Returns, Raises, Example sections
# - Maximum function length: 50 lines. Extract helpers aggressively.
# - No global mutable state. Pass configuration as function arguments.

### Library Versions (pin these)
# geopandas >= 0.14.0
# shapely >= 2.0.0
# rasterio >= 1.3.0
# pyproj >= 3.6.0
# fiona >= 1.9.0
# pandas >= 2.1.0
# numpy >= 1.25.0
# matplotlib >= 3.8.0
# xarray >= 2023.10.0
# duckdb >= 0.9.0

### Testing Conventions
# - Every module in src/ has a corresponding test file in tests/
# - Test spatial operations with known-answer geometries
# - Use pytest fixtures for loading test data
# - Test CRS handling explicitly: wrong CRS input must raise, not silently pass
# - Tolerance-aware assertions for floating-point geometry comparisons
# - Minimum test coverage: 80%

### Output Format Rules
# - Vector output: GeoPackage (.gpkg) or GeoParquet (.parquet). NEVER Shapefile.
# - Raster output: Cloud-Optimized GeoTIFF (COG) with LZW compression
# - Tables: CSV with UTF-8 encoding and ISO 8601 dates
# - Maps: PNG at 300 DPI for print, HTML for interactive
# - Reports: Markdown source, rendered to PDF via pandoc

### GIS-Specific Rules
# - NEVER output Shapefiles. Use GeoPackage or GeoParquet.
# - ALWAYS validate geometry after any spatial operation (gdf.is_valid.all())
# - ALWAYS specify CRS explicitly when creating new GeoDataFrames
# - Prefer projected CRS for distance/area calculations, geographic CRS for display
# - Log all coordinate transformations with source and target EPSG codes
# - Use well-known-text (WKT2) for CRS serialization, not PROJ4 strings
# - Handle antimeridian-crossing geometries explicitly
# - Check for and handle null/empty geometries before any operation
# - When working with Chinese administrative boundaries, use GCJ-02 to WGS-84
#   transformation where required and document the offset correction applied

### Commit Message Convention
# type(scope): description
# Types: feat, fix, data, analysis, viz, docs, test, refactor, ci
# Scope: module name or data layer
# Example: feat(flood-model): add drainage network weighted scoring
# Example: data(dem): reproject SRTM tiles to EPSG:4547

### When Asked to Generate a Map
# 1. Always include: title, north arrow indicator, scale bar, legend, CRS label
# 2. Use colorblind-safe palettes (viridis, cividis, or ColorBrewer qualitative)
# 3. Label CRS in the map margin (e.g., "EPSG:4326 / WGS 84")
# 4. For Chinese text, use SimHei or Noto Sans CJK SC font
# 5. Export at 300 DPI minimum for any deliverable
```

**Variables to Customize:**
- Project name and one-line description
- Directory structure (adapt to your team conventions)
- Library versions (pin to your environment)
- CRS conventions specific to your study area
- Commit message scopes relevant to your project layers
- Font choices for your target language

**Expected Output:** Every Claude Code session in this project will produce code that follows these standards without being reminded. CRS validation, GeoPackage output, type hints, and pathlib usage become automatic.

**Validation Checklist:**
- [ ] CLAUDE.md is at the project root (not in a subdirectory)
- [ ] Library versions match your installed environment
- [ ] CRS rules reference the EPSG codes relevant to your study area
- [ ] Output format rules align with your client delivery requirements
- [ ] Test conventions match your CI/CD pipeline expectations

**Cost Optimization:** CLAUDE.md is loaded once per session. Keep it under 200 lines to minimize context window consumption. Move detailed per-module instructions into `.claude/` subdirectories or Skills.

**See also:** [automation-workflows.md](automation-workflows.md) for project bootstrap prompts, [data-analysis-prompts.md](data-analysis-prompts.md) for coding patterns

**Extensibility:** Add per-directory CLAUDE.md files for module-specific instructions. For example, `src/raster/CLAUDE.md` can contain rasterio-specific patterns while `src/vector/CLAUDE.md` covers GeoPandas conventions.

---

### P2 -- Claude Code Hooks for GIS

**Scenario (shiji changjing):** You want automated quality gates that fire every time Claude Code writes a file or you make a commit. These hooks catch CRS mismatches, invalid geometries, and style violations before they reach your repository.

**Roles:**
- Human: defines quality standards
- Claude Code Hooks: automated pre-commit and post-file-write scripts
- CI Pipeline: final validation gate

**Pattern: Hook Configuration and Scripts**

Claude Code supports hooks defined in `.claude/hooks/`. These run automatically at specific lifecycle points.

**Directory Structure:**

```
.claude/
  hooks/
    pre-commit.sh          # Runs before any git commit via Claude Code
    post-file-write.py     # Runs after Claude Code writes/edits a Python file
    validate-spatial.py    # Shared spatial validation logic
```

**Hook 1: Pre-Commit Spatial Validation (.claude/hooks/pre-commit.sh)**

```bash
#!/usr/bin/env bash
# .claude/hooks/pre-commit.sh
# Validates all staged spatial data files before committing.

set -euo pipefail

STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

# Check for accidentally staged shapefiles
SHAPEFILES=$(echo "$STAGED_FILES" | grep -E '\.(shp|shx|dbf|prj)$' || true)
if [ -n "$SHAPEFILES" ]; then
    echo "ERROR: Shapefiles detected in staging area. Use GeoPackage or GeoParquet instead."
    echo "$SHAPEFILES"
    echo "Convert with: ogr2ogr output.gpkg input.shp"
    exit 1
fi

# Check for large raster files that should use Git LFS
LARGE_RASTERS=$(echo "$STAGED_FILES" | grep -E '\.(tif|tiff|img|nc)$' || true)
for f in $LARGE_RASTERS; do
    SIZE=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 10485760 ]; then  # 10 MB
        echo "ERROR: Raster file $f is $(( SIZE / 1048576 )) MB."
        echo "Files over 10 MB should use Git LFS: git lfs track '*.tif'"
        exit 1
    fi
done

# Validate Python files for CRS handling
PYTHON_FILES=$(echo "$STAGED_FILES" | grep '\.py$' || true)
for f in $PYTHON_FILES; do
    # Check for bare EPSG string comparisons (fragile pattern)
    if grep -n 'crs == "EPSG' "$f" 2>/dev/null; then
        echo "WARNING in $f: String-based CRS comparison detected."
        echo "Use pyproj.CRS objects: pyproj.CRS(gdf.crs) == pyproj.CRS('EPSG:4326')"
    fi
    # Check for os.path usage
    if grep -n 'import os\.path\|from os\.path\|os\.path\.' "$f" 2>/dev/null; then
        echo "WARNING in $f: os.path usage detected. Use pathlib.Path instead."
    fi
done

echo "Pre-commit spatial validation passed."
```

**Hook 2: Post-File-Write Validation (.claude/hooks/post-file-write.py)**

```python
#!/usr/bin/env python3
"""
.claude/hooks/post-file-write.py
Runs after Claude Code creates or modifies a Python file.
Validates GIS coding standards automatically.
"""
import ast
import sys
from pathlib import Path


def check_crs_validation(tree: ast.AST, filepath: str) -> list[str]:
    """Check that functions handling spatial data validate CRS."""
    warnings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Check if function has GeoDataFrame in annotations
            has_geo_param = False
            for arg in node.args.args:
                if arg.annotation and "GeoDataFrame" in ast.dump(arg.annotation):
                    has_geo_param = True
            if has_geo_param:
                # Check function body for CRS validation
                body_source = ast.dump(node)
                if "crs" not in body_source.lower():
                    warnings.append(
                        f"{filepath}:{node.lineno} -- function '{node.name}' "
                        f"accepts GeoDataFrame but does not reference CRS"
                    )
    return warnings


def check_context_managers(tree: ast.AST, filepath: str) -> list[str]:
    """Check that file operations use context managers."""
    warnings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                func_name = node.func.id
            if func_name in ("open", "connect"):
                # Walk up to check if inside a With block
                # Simplified check: warn if open() is a standalone Expr
                pass  # Full implementation would track parent nodes
    return warnings


def check_type_hints(tree: ast.AST, filepath: str) -> list[str]:
    """Check that function signatures have type hints."""
    warnings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_"):
                continue
            if node.returns is None:
                warnings.append(
                    f"{filepath}:{node.lineno} -- function '{node.name}' "
                    f"missing return type hint"
                )
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                if arg.annotation is None:
                    warnings.append(
                        f"{filepath}:{node.lineno} -- function '{node.name}' "
                        f"parameter '{arg.arg}' missing type hint"
                    )
    return warnings


def main() -> None:
    if len(sys.argv) < 2:
        return

    filepath = sys.argv[1]
    if not filepath.endswith(".py"):
        return

    path = Path(filepath)
    if not path.exists():
        return

    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError:
        print(f"SYNTAX ERROR in {filepath} -- file cannot be parsed")
        sys.exit(1)

    all_warnings = []
    all_warnings.extend(check_crs_validation(tree, filepath))
    all_warnings.extend(check_type_hints(tree, filepath))

    for w in all_warnings:
        print(f"  HOOK WARNING: {w}")

    if all_warnings:
        print(f"\n  {len(all_warnings)} warning(s) found. Review before committing.")


if __name__ == "__main__":
    main()
```

**Variables to Customize:**
- File size threshold for Git LFS warning (default: 10 MB)
- Which coding patterns to flag (os.path, string CRS comparison)
- Which functions require CRS validation (based on parameter type hints)
- Additional linting rules for your team conventions

**Expected Output:** Immediate feedback when Claude Code writes code that violates GIS project standards. Shapefiles are blocked from commits. CRS-unaware functions are flagged. os.path usage is warned.

**Validation Checklist:**
- [ ] Hook scripts are executable (`chmod +x`)
- [ ] Python hook uses the correct interpreter path for your environment
- [ ] Pre-commit hook tests pass on a clean repository
- [ ] Post-file-write hook does not block valid code

**Cost Optimization:** Hooks run locally with zero token cost. They prevent bad code from being committed, saving the cost of debugging sessions later.

**See also:** [automation-workflows.md](automation-workflows.md) for CI/CD patterns

**Extensibility:** Add a `validate-spatial.py` shared module that both hooks import for geometry validation. Extend post-file-write to check rasterio code for proper `with` block usage around dataset handles.

---

### P3 -- Multi-Turn GIS Analysis Session

**Scenario (shiji changjing):** You need to conduct a complex spatial analysis -- for example, identifying optimal locations for emergency shelters based on population density, flood risk, road accessibility, and existing facility coverage. This requires multiple turns: data exploration, analysis design, implementation, testing, visualization, and documentation.

**Roles:**
- Human: GIS analyst guiding the analysis
- Claude Code: implements each step, maintaining context across turns

**Prompt Template:**

The following is a six-turn conversation template. Each turn builds on the results of the previous turn. Copy the turns sequentially into your Claude Code session.

```
TURN 1 -- DATA EXPLORATION
===========================
I am starting a [analysis type] analysis for [study area].

Available data files:
- [file1.gpkg]: [description, CRS, feature count if known]
- [file2.tif]: [description, CRS, resolution if known]
- [file3.csv]: [description, columns of interest]

For each file:
1. Load it and report: CRS, extent (bounding box), feature/pixel count, column names or band count
2. Check for null/invalid geometries or NoData values
3. Identify CRS conflicts between datasets
4. Generate a quick summary table comparing all datasets

Do NOT start the analysis yet. Just explore and report.
```

```
TURN 2 -- ANALYSIS DESIGN
===========================
Based on the data exploration, here is my analysis plan:

Objective: [specific research question or deliverable]
Method: [analytical approach, e.g., multi-criteria decision analysis, spatial clustering]
Steps:
1. [Preprocessing step, e.g., reproject all layers to EPSG:XXXX]
2. [Step 2, e.g., calculate distance-to-road raster]
3. [Step 3, e.g., normalize all criteria to 0-1 range]
4. [Step 4, e.g., weighted overlay with weights: criterion_a=0.3, criterion_b=0.25, ...]
5. [Step 5, e.g., extract top-N candidate locations]

Before implementing, review this plan and:
- Flag any methodological issues
- Suggest missing preprocessing steps
- Confirm data compatibility for each step
- Recommend appropriate spatial resolution and CRS for the analysis
```

```
TURN 3 -- IMPLEMENTATION
===========================
The analysis plan looks good with the adjustments you suggested. Now implement it.

Requirements:
- Create a module at src/analysis/[module_name].py
- Each step should be a separate function with clear inputs/outputs
- All intermediate results saved to data/interim/ as GeoPackage or COG
- Use logging (not print) to track progress
- Include a main() function that runs the full pipeline
- Add docstrings explaining the GIS methodology, not just the code
```

```
TURN 4 -- TESTING
===========================
Write tests for the analysis module you just created.

Test strategy:
- Unit tests for each function using small synthetic geometries
- Integration test that runs the full pipeline on a small subset of the real data
- Edge case tests: empty geometries, single-feature inputs, CRS mismatch handling
- Regression test: known-answer test with a precomputed expected result

Place tests in tests/test_[module_name].py. Run them and fix any failures.
```

```
TURN 5 -- VISUALIZATION
===========================
Create visualizations of the analysis results.

Required outputs:
1. Overview map showing [all criteria layers and final result]
   - Use [colorblind-safe palette]
   - Include: title, legend, scale bar, CRS label, north indicator
   - Annotate top-[N] candidate locations
2. Detail map zoomed to the top [N] results
3. Summary statistics chart (bar or box plot of scores by [grouping variable])
4. Interactive HTML map using folium or leafmap for client review

Save to outputs/maps/ with descriptive filenames.
```

```
TURN 6 -- DOCUMENTATION
===========================
Generate project documentation:

1. Analysis report (outputs/reports/analysis_report.md):
   - Executive summary (3-5 sentences)
   - Data sources table (name, source URL, CRS, date, license)
   - Methodology section with the analysis steps and parameters
   - Results with embedded map references
   - Limitations and assumptions
   - Reproducibility instructions

2. Update the project README.md with:
   - How to set up the environment (conda/pip)
   - How to run the analysis (make commands or CLI)
   - Where to find outputs

3. Generate a data lineage diagram (ASCII or Mermaid) showing
   input data -> processing steps -> output products
```

**Variables to Customize:**
- Analysis type and study area
- Data files and their properties
- Analytical method and parameters
- Visualization requirements and palette preferences
- Report format and audience (technical vs executive)

**Expected Output:** A complete analysis project with modular code, tests, visualizations, and documentation -- all built incrementally across six turns with Claude Code maintaining context throughout.

**Validation Checklist:**
- [ ] Turn 1 output identifies all CRS conflicts
- [ ] Turn 2 design addresses all data compatibility issues
- [ ] Turn 3 code follows CLAUDE.md standards (type hints, pathlib, CRS validation)
- [ ] Turn 4 tests achieve 80%+ coverage of the analysis module
- [ ] Turn 5 maps include all required cartographic elements
- [ ] Turn 6 documentation enables another analyst to reproduce the work

**Cost Optimization:** Each turn sends only the new prompt plus Claude Code's accumulated context. Keep Turn 1 data summaries concise -- report statistics, not raw data. If the dataset description is large, use MCP (see P6) instead of pasting data into the prompt.

**See also:** [data-analysis-prompts.md](data-analysis-prompts.md) for individual analysis prompts, [map-styling-prompts.md](map-styling-prompts.md) for visualization patterns

**Extensibility:** Add a Turn 7 for peer review simulation where you ask Claude Code to critique its own analysis and suggest improvements. Add a Turn 8 for client presentation generation.

---

### P4 -- Claude Code + Git Workflow

**Scenario (shiji changjing):** You manage a GIS project repository with multiple analysis branches, spatial data versioning challenges, and a team that needs clear commit histories and PR descriptions. Claude Code can automate the git workflow to maintain clean project history.

**Roles:**
- Human: project lead making analysis decisions
- Claude Code: manages branches, commits, and PRs
- GitHub/GitLab: hosting and review platform

**Prompt Template:**

```
BRANCH MANAGEMENT
==================
I am working on [analysis description] for the [project name] project.

Create a feature branch following our naming convention:
  [type]/[description]
  Types: analysis/, data/, viz/, fix/, docs/
  Example: analysis/flood-risk-scoring

Branch from: [base branch, e.g., main or develop]

After creating the branch, set up the directory structure for this analysis
following the project conventions in CLAUDE.md.
```

```
COMMIT CONVENTIONS FOR SPATIAL DATA
=====================================
I have completed [description of work done]. Create a commit.

Follow these conventions:
- Commit message format: type(scope): description
- For spatial data changes, include in the commit body:
  - CRS of affected datasets
  - Bounding box of study area
  - Number of features/pixels affected
  - Any schema changes (added/removed/renamed columns)

Example commit message:
  data(land-cover): reclassify Sentinel-2 scene to 7 LULC classes

  Source: Sentinel-2 L2A, 2024-06-15, T50RKR
  CRS: EPSG:32650 (WGS 84 / UTM zone 50N)
  Extent: 113.8E-114.5E, 22.4N-22.9N (Shenzhen municipality)
  Output: data/processed/lulc_shenzhen_2024.gpkg (47,832 polygons)
  Classification: Random Forest, OA=87.3%, Kappa=0.84
```

```
PULL REQUEST WITH MAP CONTEXT
===============================
I am ready to merge [branch name] into [base branch].

Create a pull request with:
1. Title: concise description of the analysis/feature
2. Body:
   - Summary: what this branch adds (2-3 sentences)
   - Data changes: list any new/modified datasets with CRS and extent
   - Analysis method: brief methodology description
   - Key results: 1-3 headline findings
   - Map outputs: list generated maps with descriptions
   - Testing: what tests were added/run
   - Checklist:
     - [ ] All outputs use GeoPackage or GeoParquet (no Shapefiles)
     - [ ] CRS is consistent across all outputs
     - [ ] Tests pass locally
     - [ ] Documentation updated
     - [ ] Map outputs include cartographic elements (title, legend, scale)
```

**Variables to Customize:**
- Branch naming convention (adapt to your team)
- Commit message format and required metadata
- PR template sections and checklist items
- Base branch name

**Expected Output:** Clean git history with spatial-data-aware commit messages, feature branches per analysis task, and PR descriptions that give reviewers full context about spatial data changes.

**Validation Checklist:**
- [ ] Branch name follows the convention
- [ ] Commit messages include CRS and extent for spatial data changes
- [ ] PR description includes all required sections
- [ ] No Shapefiles in the commit history

**Cost Optimization:** Git operations are local commands with zero token cost beyond the prompt. Keep commit messages informative but concise -- the metadata (CRS, extent, feature count) costs negligible tokens but saves reviewers significant time.

**See also:** [automation-workflows.md](automation-workflows.md) for CI/CD patterns

**Extensibility:** Add a `post-merge` hook that regenerates a data catalog (`data/README.md`) listing all datasets in the repository with their CRS, extent, and lineage. This keeps spatial metadata in sync with the codebase.

---

### P5 -- Debugging GIS Code with Claude Code

**Scenario (shiji changjing):** Your GDAL/GeoPandas/Rasterio code is failing with cryptic errors -- topology exceptions, CRS transformation failures, memory errors on large rasters, or PyQGIS segfaults. You need systematic debugging that accounts for GIS-specific pitfalls.

**Roles:**
- Human: developer experiencing the bug
- Claude Code: debugger with GIS domain knowledge

**Prompt Template:**

```
GIS CODE DEBUGGING REQUEST
============================

## Error Description
[What you expected to happen vs what actually happened]

## Traceback
```
[Paste the full traceback here]
```

## Relevant Code
```python
[Paste the function or code block that fails]
```

## Data Context
- Input file(s): [format, CRS, approximate size, feature/pixel count]
- Sample of data (first 5 rows or a 100x100 pixel window):
  [Paste a small sample or describe the data structure]

## Environment
- Python: [version]
- OS: [macOS/Linux/Windows]
- Key library versions: [geopandas, shapely, rasterio, gdal, pyproj, fiona]
- Running inside: [script / Jupyter / QGIS Python console / Docker]

## What I Have Already Tried
- [List debugging steps already taken]

## Suspected Cause (if any)
[Your hypothesis about what is going wrong]

---

DEBUG THIS SYSTEMATICALLY:
1. Identify the root cause from the traceback
2. Check for these common GIS pitfalls:
   - CRS mismatch between layers
   - Invalid geometries (self-intersections, null, empty)
   - Mixed geometry types in a single layer
   - Antimeridian or pole-crossing coordinates
   - Integer overflow in large raster calculations (use float64)
   - File lock contention (Shapefile opened in QGIS while script runs)
   - GDAL driver not available or misconfigured
   - Memory mapping failure on large files
   - Encoding issues with Chinese/CJK attribute values
   - GCJ-02 vs WGS-84 coordinate offset in Chinese data
3. Provide the fix with explanation
4. Suggest a defensive coding pattern to prevent this class of bug
```

**Variables to Customize:**
- Error description and traceback
- Code block and data context
- Environment details
- Previous debugging attempts

**Expected Output:** Root cause identification, a working fix, and a defensive coding pattern that prevents the same class of error in future code.

**Validation Checklist:**
- [ ] Root cause matches the traceback evidence
- [ ] Fix addresses the root cause, not just the symptom
- [ ] Defensive pattern is general enough to catch related issues
- [ ] Fix maintains CRS integrity and geometry validity

**Cost Optimization:** Always include the full traceback -- it is more token-efficient than multiple rounds of Claude Code asking for it. Include a small data sample rather than describing the data in words. Trim the code to the relevant function, not the entire module.

**Common GIS Pitfalls Quick Reference:**

| Error Pattern | Likely Cause | Quick Fix |
|---|---|---|
| `TopologicalError` | Self-intersecting geometry | `geometry.make_valid()` or `geometry.buffer(0)` |
| `CRSError: Invalid projection` | PROJ database not found | Set `PROJ_LIB` environment variable |
| `ValueError: CRS mismatch` | Joining layers with different CRS | `gdf.to_crs(target_crs)` before the join |
| `MemoryError` on raster | Loading full raster into RAM | Use `rasterio` windowed reads |
| `CPLE_AppDefined` | GDAL driver issue | Check `gdal.GetDriverByName()` returns non-None |
| `UnicodeDecodeError` on DBF | Chinese characters in Shapefile | `gpd.read_file(path, encoding='gbk')` |
| Segmentation fault in PyQGIS | Accessing deleted C++ object | Hold Python reference to QgsProject.instance() |
| Silent wrong results | GCJ-02 data treated as WGS-84 | Apply coordinate correction before analysis |

**See also:** [data-analysis-prompts.md](data-analysis-prompts.md) for geometry fixing prompts, [plugin-dev-prompts.md](plugin-dev-prompts.md) for PyQGIS patterns

**Extensibility:** Build a project-specific "known bugs" section in CLAUDE.md that lists bugs you have encountered and their fixes. Claude Code will reference this automatically in future sessions, avoiding repeated debugging of the same issue class.

---

## 2. MCP Server Patterns for GIS

The Model Context Protocol (MCP) allows AI assistants to interact with external systems through a standardized interface of **tools** (functions the AI can call), **resources** (data the AI can read), and **prompts** (templates the AI can use). For GIS, MCP servers turn spatial infrastructure -- databases, catalogs, processing engines, tile servers -- into tools that the AI can use conversationally.

```
Architecture Overview:

  ┌─────────────────────────────────────────────────────────────┐
  │                     Claude / AI Assistant                    │
  │                                                             │
  │  "Find all buildings within 500m of the new metro station"  │
  └──────────┬──────────────────┬──────────────────┬────────────┘
             │ MCP              │ MCP              │ MCP
             v                  v                  v
  ┌──────────────────┐ ┌──────────────┐ ┌────────────────────┐
  │  PostGIS MCP     │ │  STAC MCP    │ │  GDAL MCP          │
  │  Server (P6)     │ │  Server (P7) │ │  Server (P8)       │
  │                  │ │              │ │                    │
  │  - spatial_query │ │  - search    │ │  - reproject       │
  │  - find_nearest  │ │  - download  │ │  - clip            │
  │  - buffer_query  │ │  - metadata  │ │  - convert         │
  └────────┬─────────┘ └──────┬───────┘ └─────────┬──────────┘
           │                  │                    │
           v                  v                    v
  ┌──────────────────┐ ┌──────────────┐ ┌────────────────────┐
  │    PostGIS       │ │  STAC API    │ │  Local / Cloud     │
  │    Database      │ │  (e.g.       │ │  Filesystem        │
  │                  │ │  Element84)  │ │                    │
  └──────────────────┘ └──────────────┘ └────────────────────┘
```

Each MCP server below includes: purpose, tool definitions, a TypeScript implementation outline, and configuration for Claude Code.

---

### P6 -- PostGIS MCP Server Architecture

**Scenario (shiji changjing):** Your organization has spatial data in PostGIS -- buildings, roads, parcels, utilities, environmental monitoring stations. You want the AI to query this data conversationally: "Show me all parcels within the flood zone that are zoned residential" or "Find the 5 nearest hospitals to this coordinate."

**Roles:**
- Human: asks spatial questions in natural language
- Claude (via MCP): translates questions to spatial SQL and executes them
- PostGIS MCP Server: provides tools for safe, parameterized spatial queries
- PostGIS Database: stores and indexes spatial data

**Pattern: MCP Server Tool Definitions**

```typescript
// postgis-mcp-server/src/tools.ts
// Tool definitions for the PostGIS MCP server

import { z } from "zod";

export const tools = {
  list_tables: {
    description: "List all spatial tables in the database with their geometry type and SRID",
    parameters: z.object({
      schema: z.string().default("public").describe("Database schema to list tables from"),
    }),
    // Returns: [{table_name, geometry_column, geometry_type, srid, row_count}]
  },

  describe_table: {
    description: "Get column names, types, and spatial metadata for a table",
    parameters: z.object({
      table_name: z.string().describe("Name of the table to describe"),
      schema: z.string().default("public"),
    }),
    // Returns: {columns: [{name, type, nullable}], geometry: {column, type, srid}, indexes: [...]}
  },

  spatial_query: {
    description:
      "Execute a read-only spatial SQL query. Only SELECT statements allowed. " +
      "Results are returned as GeoJSON FeatureCollection. Limit 1000 rows by default.",
    parameters: z.object({
      sql: z.string().describe("SQL SELECT query (read-only, no INSERT/UPDATE/DELETE)"),
      limit: z.number().default(1000).describe("Maximum rows to return"),
    }),
    // Validation: reject any non-SELECT statement
    // Returns: GeoJSON FeatureCollection
  },

  get_extent: {
    description: "Get the bounding box (extent) of a spatial table",
    parameters: z.object({
      table_name: z.string(),
      schema: z.string().default("public"),
    }),
    // Query: SELECT ST_Extent(geom) FROM table
    // Returns: {xmin, ymin, xmax, ymax, srid}
  },

  get_srid: {
    description: "Get the SRID (coordinate reference system ID) of a spatial table",
    parameters: z.object({
      table_name: z.string(),
      schema: z.string().default("public"),
    }),
    // Query: SELECT Find_SRID(schema, table, geom_column)
    // Returns: {srid: number, proj4: string, authority: string}
  },

  find_nearest: {
    description: "Find the K nearest features in a table to a given point",
    parameters: z.object({
      longitude: z.number().describe("Longitude of the query point (WGS84)"),
      latitude: z.number().describe("Latitude of the query point (WGS84)"),
      table_name: z.string().describe("Table to search"),
      k: z.number().default(5).describe("Number of nearest features to return"),
      max_distance_meters: z.number().optional().describe("Maximum search radius in meters"),
      schema: z.string().default("public"),
    }),
    // Uses: ST_DWithin + ORDER BY ST_Distance with KNN index
    // Returns: GeoJSON FeatureCollection with distance_meters property
  },

  buffer_query: {
    description: "Find all features in a table within a buffer distance of a geometry",
    parameters: z.object({
      wkt: z.string().describe("WKT geometry to buffer (in EPSG:4326)"),
      distance_meters: z.number().describe("Buffer distance in meters"),
      table_name: z.string(),
      schema: z.string().default("public"),
      where: z.string().optional().describe("Additional WHERE clause filter"),
    }),
    // Uses: ST_DWithin with geography cast for meter-based distance
    // Returns: GeoJSON FeatureCollection
  },
};
```

**Server Implementation Outline:**

```typescript
// postgis-mcp-server/src/index.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import pg from "pg";

const pool = new pg.Pool({
  connectionString: process.env.DATABASE_URL,
  max: 5,
  // IMPORTANT: use a read-only database user
});

const server = new Server(
  { name: "postgis-mcp", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// Register tool handlers
server.setRequestHandler("tools/list", async () => ({
  tools: Object.entries(tools).map(([name, def]) => ({
    name,
    description: def.description,
    inputSchema: zodToJsonSchema(def.parameters),
  })),
}));

server.setRequestHandler("tools/call", async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "spatial_query": {
      // SECURITY: validate SQL is read-only
      const sql = args.sql.trim();
      if (!/^SELECT/i.test(sql)) {
        throw new Error("Only SELECT queries are allowed");
      }
      if (/\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE)\b/i.test(sql)) {
        throw new Error("Mutation queries are not allowed");
      }
      const result = await pool.query(
        `SELECT json_build_object(
          'type', 'FeatureCollection',
          'features', COALESCE(json_agg(ST_AsGeoJSON(t.*)::json), '[]'::json)
        ) AS geojson
        FROM (${sql} LIMIT $1) AS t`,
        [args.limit || 1000]
      );
      return { content: [{ type: "text", text: JSON.stringify(result.rows[0].geojson) }] };
    }
    // ... other tool handlers
  }
});

const transport = new StdioServerTransport();
await server.connect(transport);
```

**Claude Code Configuration (claude_desktop_config.json or .mcp.json):**

```json
{
  "mcpServers": {
    "postgis": {
      "command": "node",
      "args": ["./postgis-mcp-server/dist/index.js"],
      "env": {
        "DATABASE_URL": "postgresql://readonly_user:password@localhost:5432/gis_db"
      }
    }
  }
}
```

**Variables to Customize:**
- Database connection string and credentials
- Schema name (if not `public`)
- Row limit defaults
- Additional tools (aggregate statistics, geometry simplification, tile generation)
- SQL allowlist/blocklist patterns for security

**Expected Output:** Natural language spatial queries are translated to PostGIS SQL automatically. The AI can explore your database schema, run spatial queries, and return GeoJSON results -- all through conversation.

**Validation Checklist:**
- [ ] Database user is read-only (no INSERT/UPDATE/DELETE/DROP permissions)
- [ ] SQL injection protection: parameterized queries, statement validation
- [ ] Row limits prevent accidentally returning millions of rows
- [ ] SRID handling: tool converts between EPSG:4326 input and table's native SRID
- [ ] Connection pool has reasonable limits (max 5-10 connections)
- [ ] Error messages do not leak database credentials or internal schema details

**Cost Optimization:** MCP tools return data on-demand, so the AI only loads what it needs into context. A `find_nearest(k=5)` call returns 5 features instead of loading an entire table. Always use `limit` parameters. For large result sets, return summary statistics first and let the AI request details.

**Dark Arts Tip:** Create a `semantic_search` tool that uses a table of natural-language descriptions of your layers (maintained by your team) so the AI can find the right table without knowing the exact name. For example: "find building data" matches `osm_buildings_shenzhen` because its description contains "OpenStreetMap building footprints for Shenzhen."

**See also:** [../tools/spatial-databases.md](../tools/spatial-databases.md) for PostGIS setup and optimization

**Extensibility:** Add tools for write operations behind an authentication gate: `insert_feature`, `update_attribute`, `create_view`. Add a `generate_map_tile` tool that renders a specific bbox as a PNG using ST_AsMVT or a rendering engine.

---

### P7 -- STAC Catalog MCP Server

**Scenario (shiji changjing):** You need satellite imagery for a project -- Sentinel-2 scenes with less than 10% cloud cover over your study area from the last month. Instead of writing pystac-client code, you want to ask the AI: "Find cloud-free Sentinel-2 imagery over Shenzhen from January 2025."

**Roles:**
- Human: describes imagery needs in natural language
- Claude (via MCP): searches STAC catalogs and manages downloads
- STAC MCP Server: wraps pystac-client as MCP tools
- STAC API: remote catalog (e.g., Element84 Earth Search, Microsoft Planetary Computer)

**Pattern: MCP Server Tool Definitions**

```typescript
// stac-mcp-server/src/tools.ts

export const tools = {
  get_collections: {
    description:
      "List all available collections in the STAC catalog with their " +
      "descriptions, spatial extent, and temporal range",
    parameters: z.object({}),
    // Returns: [{id, title, description, spatial_extent, temporal_extent}]
  },

  search_items: {
    description:
      "Search for satellite imagery items matching spatial, temporal, and " +
      "property filters. Returns metadata, NOT the actual imagery data.",
    parameters: z.object({
      bbox: z.array(z.number()).length(4)
        .describe("Bounding box [west, south, east, north] in EPSG:4326"),
      datetime: z.string()
        .describe("ISO 8601 datetime range, e.g., '2025-01-01/2025-01-31'"),
      collections: z.array(z.string())
        .describe("Collection IDs to search, e.g., ['sentinel-2-l2a']"),
      query: z.record(z.any()).optional()
        .describe("Property filters, e.g., {'eo:cloud_cover': {'lt': 10}}"),
      limit: z.number().default(10)
        .describe("Maximum items to return"),
      sortby: z.string().optional()
        .describe("Sort field, e.g., '-properties.eo:cloud_cover'"),
    }),
    // Returns: [{id, datetime, cloud_cover, bbox, thumbnail_url, asset_keys}]
  },

  get_item_assets: {
    description:
      "Get detailed asset information for a specific STAC item, " +
      "including download URLs, file sizes, and band information",
    parameters: z.object({
      collection: z.string().describe("Collection ID"),
      item_id: z.string().describe("Item ID from search results"),
    }),
    // Returns: [{key, title, href, type, file_size, eo_bands}]
  },

  download_asset: {
    description:
      "Download a specific asset (e.g., a band or thumbnail) to a local path",
    parameters: z.object({
      url: z.string().describe("Asset URL from get_item_assets"),
      output_path: z.string().describe("Local file path to save the asset"),
      bbox_clip: z.array(z.number()).length(4).optional()
        .describe("Optional: clip to bbox during download (COG only)"),
    }),
    // Uses: HTTP range requests for COG partial download
    // Returns: {path, file_size, crs, resolution}
  },

  preview_item: {
    description:
      "Generate a quick-look preview of a STAC item using its thumbnail or " +
      "visual bands. Returns a file path to the preview image.",
    parameters: z.object({
      collection: z.string(),
      item_id: z.string(),
      bands: z.array(z.string()).optional()
        .describe("Band names for RGB composite, e.g., ['B04', 'B03', 'B02']"),
      bbox: z.array(z.number()).length(4).optional()
        .describe("Clip preview to bbox"),
    }),
    // Returns: {preview_path, crs, extent}
  },
};
```

**Claude Code Configuration:**

```json
{
  "mcpServers": {
    "stac": {
      "command": "python",
      "args": ["-m", "stac_mcp_server"],
      "env": {
        "STAC_API_URL": "https://earth-search.aws.element84.com/v1",
        "DOWNLOAD_DIR": "./data/raw/imagery",
        "PLANETARY_COMPUTER_KEY": "optional-api-key"
      }
    }
  }
}
```

**Example Conversation Flow:**

```
Human: Find cloud-free Sentinel-2 imagery over Shenzhen from January 2025.

AI [calls search_items]:
  bbox: [113.75, 22.39, 114.62, 22.86]
  datetime: "2025-01-01/2025-01-31"
  collections: ["sentinel-2-l2a"]
  query: {"eo:cloud_cover": {"lt": 10}}
  sortby: "-properties.datetime"

AI: Found 4 scenes with <10% cloud cover:
  1. S2B_T50RKR_20250128 -- 3.2% cloud cover
  2. S2A_T50RKR_20250118 -- 7.1% cloud cover
  3. S2A_T50RKR_20250108 -- 5.4% cloud cover
  4. S2B_T50RKR_20250103 -- 8.9% cloud cover
  Which scene would you like to work with?