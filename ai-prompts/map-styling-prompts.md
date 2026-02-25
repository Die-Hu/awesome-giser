# Map Styling Prompts

> Expert prompt templates for cartographic color design, MapLibre GL JS style authoring, desktop GIS symbology, label placement, legend composition, icon systems, and chart-map integration.

> **Quick Picks**
> - **Most Useful**: [P1 -- Scientific Color Palette Generator](#prompt-1--scientific-color-palette-generator) -- accessible, perceptually uniform palettes should be the default
> - **Time Saver**: [P7 -- Dark Mode Map Style](#prompt-7--dark-mode-map-style) -- production-ready dark style JSON in one prompt
> - **Cutting Edge**: [P3 -- Bivariate/Multivariate Color Scheme](#prompt-3--bivariatemultivariate-color-scheme) -- bivariate choropleth matrices with accessibility built in
> - **Print Pro**: [P12 -- Print-Ready Cartographic Style](#prompt-12--print-ready-cartographic-style) -- CMYK-safe, 300 DPI line weights

---

## Table of Contents

- [1. Color and Visual Design](#1-color-and-visual-design) (P1--P4)
- [2. MapLibre GL JS Styling](#2-maplibre-gl-js-styling) (P5--P9)
- [3. Desktop GIS Styling](#3-desktop-gis-styling) (P10--P12)
- [4. Labels and Typography](#4-labels-and-typography) (P13--P15)
- [5. Legend and Map Furniture](#5-legend-and-map-furniture) (P16--P18)
- [6. Icons and Symbols](#6-icons-and-symbols) (P19--P20)
- [7. Chart and Dashboard Integration](#7-chart-and-dashboard-integration) (P21--P22)

---

## 1. Color and Visual Design

---

### Prompt 1 -- Scientific Color Palette Generator

#### Scenario (actual use / 实际场景)

You are producing a choropleth or heatmap for a peer-reviewed journal, a government report, or a public dashboard. The palette must be colorblind-safe, perceptually uniform, print-friendly, and backed by published color science. You want palettes comparable to Okabe-Ito, viridis, or ColorBrewer -- not ad-hoc hex guesses.

#### Roles

| Role | Description |
|------|-------------|
| **User** | GIS analyst or cartographer preparing thematic maps |
| **AI** | Color scientist and cartographic design expert |

#### Prompt Template

```text
You are a color scientist specializing in cartographic visualization. Generate a
scientifically rigorous color palette for the following map:

MAP TYPE: {{map_type}}                     # choropleth | heatmap | dot density
VARIABLE: {{variable}}                     # e.g., "median household income ($22k--$180k)"
DATA DISTRIBUTION: {{distribution}}        # normal | right-skewed | bimodal | uniform
PALETTE TYPE: {{palette_type}}             # sequential | diverging | qualitative
NUMBER OF CLASSES: {{n_classes}}           # 5--9 recommended
MIDPOINT (if diverging): {{midpoint}}      # e.g., 0 for change data

REQUIREMENTS:
1. Perceptually uniform -- monotonically increasing L* in CIELAB for sequential;
   symmetric L* curve for diverging.
2. Colorblind-safe -- distinguishable under deuteranopia, protanopia, and
   tritanopia. Use Brettel/Vienot simulation to verify.
3. Minimum delta-E (CIEDE2000) of {{min_delta_e}} between adjacent classes
   (recommend >= 15 for 7 classes, >= 20 for 5 classes).
4. WCAG 2.1 AA contrast ratio >= 4.5:1 for any text overlaid on each fill.
5. Print-safe -- distinguishable when photocopied in grayscale.

OUTPUT FORMAT for each color:
| Class | Hex     | RGB           | HSL             | CIELAB L* | Deuteranopia Hex | Protanopia Hex |
|-------|---------|---------------|-----------------|-----------|------------------|----------------|

Then provide:
A. Contrast table -- each fill vs white (#FFFFFF) and dark (#1A1A2E) backgrounds,
   reporting the ratio and PASS/FAIL for WCAG AA.
B. Recommended label color (black or white) per class using the higher contrast.
C. Multi-format export:
   - CSS custom properties:    --class-1: #hex; ...
   - d3.scaleThreshold():      d3.scaleThreshold().domain([...]).range([...])
   - MapLibre expression:      ["interpolate", ["linear"], ["get", "value"], ...]
   - Python matplotlib:        ListedColormap([...])
   - ColorBrewer-style array:  ["#hex", "#hex", ...]
D. Name the palette (e.g., "income-viridis-7") and state which published palette
   it most closely resembles (viridis, cividis, Okabe-Ito, etc.).

DARK ARTS TIP: If the user requests exactly 5 sequential classes, start with
ColorBrewer's YlGnBu and shift L* endpoints to 95 and 25 for maximum range.
For diverging palettes, pin the midpoint to L*=85 so the "neutral" class reads
as clearly light against both dark and light basemaps.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{map_type}}` | choropleth | Also: heatmap, graduated symbol |
| `{{variable}}` | PM2.5 concentration (0--300 ug/m3) | Include units and range |
| `{{distribution}}` | right-skewed | Drives classification advice |
| `{{palette_type}}` | sequential | sequential / diverging / qualitative |
| `{{n_classes}}` | 7 | 5--9 for choropleth, 3--5 for bivariate axis |
| `{{midpoint}}` | 0 | Only for diverging; leave blank for sequential |
| `{{min_delta_e}}` | 18 | CIEDE2000 units |

#### Expected Output

A Markdown table of hex/RGB/HSL/L* values with simulated CVD columns, a WCAG contrast matrix, and copy-paste code blocks for CSS, D3, MapLibre, and Python.

#### Validation Checklist

- [ ] L* values are monotonically increasing (sequential) or symmetric (diverging)
- [ ] All adjacent pairs have delta-E >= specified minimum
- [ ] Deuteranopia simulation shows distinct hues or distinct lightness
- [ ] WCAG AA passes for the chosen label color on every fill
- [ ] Grayscale rendering shows distinct steps (no two classes merge)

#### Cost Optimization

- Single-turn prompt; no iteration needed if distribution is stated up front.
- If you already know you want viridis/cividis, state it -- the AI skips the search.
- Batch multiple palettes (sequential + diverging for the same variable) in one call.

#### Related Pages

- [2D Mapping (MapLibre basics)](../js-bindbox/2d-mapping.md)
- [Performance Optimization](../js-bindbox/performance-optimization.md)
- [Desktop GIS (QGIS color ramps)](../tools/desktop-gis.md)

#### Extensibility

- Chain with P4 (Thematic Map Classification) to get class breaks + palette in one pass.
- Feed the output CSS variables into P16 (Responsive Map Legend) for a synchronized legend.
- Export the Python colormap to P10 (QGIS QML Style Generator) for desktop use.

---

### Prompt 2 -- Brand-Aligned Map Theme

#### Scenario (actual use / 实际场景)

Your organization has strict brand guidelines -- primary, secondary, and accent colors with defined hex codes. You need a complete map color system (basemap tints, data fills, labels, UI controls) that feels "on brand" while remaining cartographically sound, plus a dark mode variant.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web map developer or design team lead |
| **AI** | Brand design consultant and cartographic color theorist |

#### Prompt Template

```text
You are a brand-design consultant who specializes in cartographic applications.
Convert the following corporate brand guidelines into a complete map color system.

BRAND COLORS:
  Primary:    {{primary_hex}}     # e.g., #0052CC
  Secondary:  {{secondary_hex}}   # e.g., #FF6B35
  Accent:     {{accent_hex}}      # e.g., #00B8D9
  Neutral:    {{neutral_hex}}     # e.g., #6B778C
  Background: {{bg_hex}}          # e.g., #FFFFFF

DELIVERABLES:

1. BASEMAP TINT SYSTEM -- derive water, land, road, and building fills from the
   brand palette. Each fill must have contrast ratio >= 1.5:1 against its
   neighbor (water vs land). Provide:
   - water_fill, water_outline
   - land_fill, park_fill
   - road_highway, road_major, road_minor, road_path
   - building_fill, building_outline
   - boundary_line

2. DATA VISUALIZATION PALETTE -- 6-class sequential and 7-class diverging
   palettes anchored to the brand primary/secondary. Ensure:
   - L* endpoints span at least 40 units
   - Colorblind-safe
   - delta-E >= 15 between adjacent classes

3. UI COMPONENT COLORS:
   - Popup background, popup text, popup border
   - Tooltip background, tooltip text
   - Control button fill, icon color, hover state
   - Selection highlight, hover highlight

4. DARK MODE VARIANT -- invert the entire system for dark backgrounds:
   - Dark background:   {{dark_bg}}   # e.g., #0D1117
   - Recalculate all fills, ensuring brand recognition is maintained
   - Label colors flip to light; halos flip to dark
   - Data palettes retain the same hue order but shift L* range

5. TOKEN EXPORT:
   - CSS custom properties (light and dark)
   - JSON design token file (compatible with Style Dictionary)
   - MapLibre style JSON snippet (paint properties only)

FORMAT: Return as a structured JSON with keys: basemap, data_palettes, ui, dark,
and tokens.

DARK ARTS TIP: When deriving basemap tints from brand colors, desaturate by 70%
and shift L* toward the background. Brand colors at full saturation overwhelm
geography -- the basemap should whisper the brand, not shout it.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{primary_hex}}` | #0052CC | Main brand color |
| `{{secondary_hex}}` | #FF6B35 | Secondary / CTA color |
| `{{accent_hex}}` | #00B8D9 | Accent for highlights |
| `{{neutral_hex}}` | #6B778C | Text and UI neutral |
| `{{bg_hex}}` | #FFFFFF | Light mode background |
| `{{dark_bg}}` | #0D1117 | Dark mode background |

#### Expected Output

A structured JSON with five sections (basemap, data_palettes, ui, dark, tokens), each containing hex values. CSS custom properties and a MapLibre paint snippet are included in the tokens section.

#### Validation Checklist

- [ ] Basemap fills are desaturated enough that data layers dominate visually
- [ ] Sequential palette L* range >= 40 units
- [ ] Dark mode retains brand hue identity (hue angle within 15 degrees of original)
- [ ] All UI text meets WCAG AA on its respective background
- [ ] Light and dark CSS tokens use the same variable names

#### Cost Optimization

- Provide the full brand guide (PDF link or hex list) in the first message to avoid a clarification round.
- If you only need basemap tints (not data palettes), remove deliverable 2 to cut output tokens by 30%.

#### Related Pages

- [Framework Integration (React/Vue theming)](../js-bindbox/framework-integration.md)
- [2D Mapping (style loading)](../js-bindbox/2d-mapping.md)

#### Extensibility

- Feed the basemap tint system into P5 (Complete Basemap Style) for a full style JSON.
- Use the dark mode variant with P7 (Dark Mode Map Style).
- Apply UI component colors in P22 (Dashboard Layout with Map).

---

### Prompt 3 -- Bivariate/Multivariate Color Scheme

#### Scenario (actual use / 实际场景)

You need to show two variables simultaneously on a single choropleth -- for example, income vs. education, temperature vs. precipitation, or risk vs. vulnerability. A bivariate color matrix is the standard approach, but designing one that is readable and accessible is notoriously difficult.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Data analyst or cartographer creating bivariate maps |
| **AI** | Multivariate visualization specialist and color theorist |

#### Prompt Template

```text
You are a multivariate visualization specialist. Design a bivariate color scheme
for the following map:

VARIABLE X (horizontal axis): {{var_x}}    # e.g., "Median Income ($20k--$120k)"
VARIABLE Y (vertical axis):   {{var_y}}    # e.g., "% College Educated (5%--75%)"
MATRIX SIZE: {{matrix_size}}               # 3x3 | 4x4 | 3x4
GEOMETRY: {{geometry}}                     # polygon (choropleth) | hex grid

DESIGN CONSTRAINTS:
1. Corner anchors:
   - Low X, Low Y (bottom-left):  desaturated / neutral (e.g., #E8E8E8)
   - High X, Low Y (bottom-right): strong hue A (e.g., blue family)
   - Low X, High Y (top-left):     strong hue B (e.g., red/orange family)
   - High X, High Y (top-right):   mixed hue (purple if A=blue, B=red)
2. Each axis must have monotonically increasing L* OR increasing saturation
   so that EITHER variable can be read independently.
3. Colorblind-safe: the two hue axes must be distinguishable under
   deuteranopia. Preferred axis pair: blue vs. orange (safe), or
   purple vs. green (safe). AVOID red vs. green.
4. Minimum delta-E (CIEDE2000) of 12 between any two adjacent cells.
5. The diagonal (low-low to high-high) must read as a clear gradient.

OUTPUT:
A. The matrix as a 2D array of hex codes:
   ```
   [["#E8E8E8", "#...", "#..."],
    ["#...",     "#...", "#..."],
    ["#...",     "#...", "#..."]]
   ```
B. For each cell: hex, RGB, CIELAB L*a*b*, and simulated deuteranopia hex.
C. A value-by-alpha variant: same matrix but using opacity (0.3--1.0) on
   a single hue to encode variable Y, while hue encodes variable X.
D. MapLibre GL expression using nested ["match", ...] or ["interpolate", ...]
   to apply the matrix based on classified property values.
E. SVG legend: a square grid legend with axis labels, tick marks,
   and corner annotations.
F. CSS Grid snippet to render the legend in HTML:
   ```css
   .bivariate-legend {
     display: grid;
     grid-template-columns: repeat({{matrix_size_x}}, 32px);
     grid-template-rows: repeat({{matrix_size_y}}, 32px);
     transform: rotate(-90deg); /* Y axis reads bottom-to-top */
   }
   ```

DARK ARTS TIP: For 4x4 matrices, most readers cannot distinguish 16 colors
reliably. Consider providing a "simplified" 3x3 version alongside the full 4x4
for use at smaller map sizes or lower zoom levels.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{var_x}}` | Median Income ($20k--$120k) | X-axis variable and range |
| `{{var_y}}` | % College Educated (5%--75%) | Y-axis variable and range |
| `{{matrix_size}}` | 3x3 | 3x3 typical; 4x4 for experts |
| `{{geometry}}` | polygon | polygon or hex grid |

#### Expected Output

A 2D hex array, per-cell CIELAB values with CVD simulation, a value-by-alpha variant, a MapLibre expression, an SVG legend, and a CSS Grid legend snippet.

#### Validation Checklist

- [ ] Each row and each column independently shows a clear gradient
- [ ] Deuteranopia simulation still distinguishes the two axes
- [ ] No two adjacent cells have delta-E < 12
- [ ] The diagonal from low-low to high-high is visually smooth
- [ ] Legend is readable at 200px width on mobile

#### Cost Optimization

- A 3x3 matrix (9 colors) is usually sufficient and cheaper to generate than 4x4 (16).
- Skip the value-by-alpha variant if your map library does not support per-feature opacity expressions.

#### Related Pages

- [Charting Integration (synchronized bivariate legend)](../js-bindbox/charting-integration.md)
- [Spatial Analysis (classification for bivariate)](../js-bindbox/spatial-analysis.md)

#### Extensibility

- Combine with P4 (Thematic Map Classification) to determine class breaks for each axis.
- Feed the legend SVG into P16 (Responsive Map Legend) for interactive toggling.
- Use the MapLibre expression in P6 (Data-Driven Styling) for the full layer definition.

---

### Prompt 4 -- Thematic Map Classification

#### Scenario (actual use / 实际场景)

You have a continuous numeric variable (population density, PM2.5, property values) and need to choose the right classification method. The wrong method can radically misrepresent geographic patterns. You want the AI to analyze your data distribution and recommend a method with class breaks.

#### Roles

| Role | Description |
|------|-------------|
| **User** | GIS analyst preparing a thematic map |
| **AI** | Quantitative cartographer and spatial statistician |

#### Prompt Template

```text
You are a quantitative cartographer. Given the following dataset summary,
recommend a classification method and compute class breaks.

VARIABLE: {{variable}}                # e.g., "Population Density (persons/km2)"
RECORD COUNT: {{n}}                   # e.g., 3,142 (US counties)
SUMMARY STATISTICS:
  Min:    {{min}}
  Max:    {{max}}
  Mean:   {{mean}}
  Median: {{median}}
  Std Dev: {{std_dev}}
  Skewness: {{skewness}}             # positive = right-skewed
  Kurtosis: {{kurtosis}}
  Histogram (10 bins): {{histogram}}  # e.g., [1420, 890, 340, 210, 120, 80, 42, 20, 12, 8]
  Notable outliers: {{outliers}}      # e.g., "New York County = 27,751; top 5 are > 10,000"

DESIRED NUMBER OF CLASSES: {{n_classes}}   # or "recommend"

TASKS:
1. RECOMMEND a classification method from:
   - Natural Breaks (Jenks)
   - Quantile (equal count)
   - Equal Interval
   - Standard Deviation
   - Head/Tail Breaks (for heavy-tailed distributions)
   - Manual / Custom
   Justify the choice based on the distribution shape and map purpose.

2. COMPUTE class breaks for the recommended method AND for two alternatives.
   For each method, provide:
   | Class | Range          | Count | % of Total | Suggested Label |
   |-------|----------------|-------|------------|-----------------|

3. GENERATE a text-based histogram (ASCII art, 40 chars wide) with the class
   breaks overlaid as markers, so the user can visually assess fit.

4. WARN about:
   - Empty classes (no features fall in a class)
   - One class containing > 50% of features (quantile artifact)
   - Outlier distortion (equal interval dominated by tail)

5. PROVIDE code for the recommended breaks:
   - MapLibre "step" expression:
     ["step", ["get", "{{property}}"], "#color1", break1, "#color2", ...]
   - D3.js:
     d3.scaleThreshold().domain([break1, break2, ...]).range([...])
   - Python (mapclassify):
     mc.NaturalBreaks(values, k={{n_classes}})
   - QGIS: QgsGraduatedSymbolRenderer settings

DARK ARTS TIP: For right-skewed data (skewness > 1.5), Natural Breaks often
lumps most features into one or two classes. Head/Tail Breaks or a log transform
+ Equal Interval usually produces a more informative map. Always show the
histogram to the client -- it makes the choice self-evident.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{variable}}` | Population Density (persons/km2) | Include units |
| `{{n}}` | 3142 | Feature count |
| `{{min}}, {{max}}, {{mean}}, {{median}}` | 0.1, 27751, 263, 88 | Summary stats |
| `{{std_dev}}, {{skewness}}, {{kurtosis}}` | 1740, 8.2, 95.4 | Distribution shape |
| `{{histogram}}` | [1420, 890, 340, ...] | 10-bin counts |
| `{{n_classes}}` | 5 | Or "recommend" |

#### Expected Output

A recommendation with justification, three classification tables, an ASCII histogram, warnings, and code snippets for MapLibre/D3/Python/QGIS.

#### Validation Checklist

- [ ] No empty classes in the recommended method
- [ ] No single class contains more than 40% of features (unless justified)
- [ ] Breaks are rounded to meaningful values (not 17,342.7 -- use 17,500 or 18,000)
- [ ] The ASCII histogram visually confirms good class separation
- [ ] MapLibre step expression is syntactically valid

#### Cost Optimization

- Include the histogram in the first message; it prevents a "can you provide the distribution?" follow-up.
- If you know you want Jenks, say so -- the AI skips the comparison.

#### Related Pages

- [Spatial Analysis (statistical methods)](../js-bindbox/spatial-analysis.md)
- [Python Libraries (mapclassify, geopandas)](../tools/python-libraries.md)
- [Desktop GIS (QGIS graduated renderer)](../tools/desktop-gis.md)

#### Extensibility

- Feed the breaks directly into P1 (Scientific Color Palette) to get a matched palette.
- Use the MapLibre step expression in P6 (Data-Driven Styling).
- Export the QGIS settings into P10 (QGIS QML Style Generator).

---

## 2. MapLibre GL JS Styling

---

### Prompt 5 -- Complete Basemap Style

#### Scenario (actual use / 实际场景)

You need a complete, production-ready MapLibre GL style JSON -- not just a few layers, but the entire basemap: water, land, parks, roads (full hierarchy), buildings, administrative boundaries, labels, and POIs. This is the foundational style that every data layer sits on top of.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web map developer building a custom basemap |
| **AI** | MapLibre GL style author and vector tile specialist |

#### Prompt Template

```text
You are a senior MapLibre GL style author. Generate a complete, valid MapLibre
GL style JSON (version 8) for a {{theme}} basemap.

TILE SOURCE: {{tile_source}}
SOURCE URL: {{source_url}}
GLYPHS: {{glyphs_url}}
SPRITE: {{sprite_url}}

TARGET USE: {{target_use}}

LAYER SPECIFICATIONS (in rendering order, bottom to top):

1. BACKGROUND: fill {{bg_color}}
2. WATER:
   - Oceans/seas: {{water_color}}
   - Rivers/streams: slightly lighter variant, visible z8+
   - Water labels: italic, {{water_label_color}}, z6+
3. LAND USE:
   - Parks/green: {{park_color}}, z10+
   - Industrial: {{industrial_color}}, z12+
   - Residential: {{residential_color}}, z12+
4. TERRAIN: hillshade layer, opacity {{hillshade_opacity}}, z2+
5. ROADS (hierarchical):
   - Motorway:   casing {{motorway_casing}} + fill {{motorway_fill}},
                  width interpolate z5:1px -> z10:3px -> z16:12px
   - Trunk:      casing + fill, z6+
   - Primary:    casing + fill, z8+
   - Secondary:  fill only, z10+
   - Tertiary:   fill only, z12+
   - Residential: fill only, z14+
   - Path/footway: dashed, z15+
   - Bridges: raise with sort-key, add shadow
   - Tunnels: reduce opacity to 0.5, dashed casing
6. BUILDINGS:
   - Flat fill {{building_color}} at z14+
   - 3D extrusion optional: fill-extrusion-height from "height" property, z15+
7. BOUNDARIES:
   - Country: {{boundary_color}}, solid, width 1.5px, z1+
   - State/province: dashed, width 0.75px, z4+
   - County/district: dotted, width 0.5px, z8+
8. LABELS:
   - Countries: bold, size interpolate z2:10px -> z6:18px
   - States: medium, z4+
   - Cities: size by "rank" property, z5+ for rank<=3, z8+ for rank<=6, z10+ all
   - Road labels: {{road_label_color}}, z14+, symbol-placement: line
   - POI labels: z15+, icon + text, priority by category
   - All labels: text-halo-color {{halo_color}}, text-halo-width 1.5

ZOOM TRANSITIONS:
- All width/size properties must use ["interpolate", ["exponential", 1.5], ["zoom"], ...]
- Visibility toggled via minzoom/maxzoom, NOT filter expressions

Return the complete JSON. It must be loadable as:
  const map = new maplibregl.Map({ style: 'basemap.json' });

DARK ARTS TIP: The most common basemap mistake is making roads too prominent.
At z5--z10, roads should be barely distinguishable from land -- they are context,
not content. Reserve visual weight for z14+ where navigation matters.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{theme}}` | light minimal | light / dark / topographic / satellite-hybrid |
| `{{tile_source}}` | OpenMapTiles | OpenMapTiles / Protomaps / custom |
| `{{source_url}}` | https://tiles.example.com/{z}/{x}/{y}.pbf | Or PMTiles URL |
| `{{bg_color}}` | #F2F0EB | Background fill |
| `{{water_color}}` | #AADAFF | Water body fill |
| `{{park_color}}` | #C8FACC | Park/green area fill |

#### Expected Output

A complete, valid MapLibre GL style JSON (typically 200--500 lines) with sources, glyphs, sprite, and a full layers array. Loadable directly by MapLibre GL JS 4.x.

#### Validation Checklist

- [ ] JSON is valid (no trailing commas, correct bracket nesting)
- [ ] Layers are in correct rendering order (background first, labels last)
- [ ] All `source-layer` values match the tile source schema
- [ ] Road widths use exponential interpolation, not linear
- [ ] Every label layer has `text-halo-width` > 0 for readability

#### Cost Optimization

- Specify the tile source schema explicitly (OpenMapTiles vs Mapbox) to avoid guessing.
- If you only need a minimal basemap (no POIs, no buildings), say so to reduce output.

#### Related Pages

- [2D Mapping (MapLibre setup and layers)](../js-bindbox/2d-mapping.md)
- [Tile Servers (vector tile sources)](../js-bindbox/tile-servers.md)
- [Performance Optimization (style performance)](../js-bindbox/performance-optimization.md)

#### Extensibility

- Use as the base for P7 (Dark Mode) by asking the AI to invert this style.
- Add P8 (3D Extrusion) layers on top of this basemap.
- Feed into P2 (Brand-Aligned Theme) to reskin for corporate use.

---

### Prompt 6 -- Data-Driven Styling

#### Scenario (actual use / 实际场景)

You have a GeoJSON or vector tile layer with rich attribute data and need to encode multiple variables visually -- color by category, size by magnitude, opacity by recency. MapLibre expressions are powerful but have a steep learning curve. This prompt generates production-ready expressions with performance considerations for large datasets.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web map developer adding data layers |
| **AI** | MapLibre GL expression expert and data visualization engineer |

#### Prompt Template

```text
You are a MapLibre GL expression expert. Write data-driven layer definitions
for the following dataset:

GEOMETRY: {{geometry_type}}            # point | line | polygon
SOURCE TYPE: {{source_type}}           # geojson | vector tiles
FEATURE COUNT: {{feature_count}}       # e.g., 50,000 points
PROPERTIES:
{{#each properties}}
  - "{{name}}": {{type}}, range: {{range}}, example: {{example}}
{{/each}}

VISUAL ENCODINGS:
1. COLOR driven by "{{color_property}}":
   {{color_rule}}
   # e.g., categorical: match "type" -> {"earthquake":"#E63946","explosion":"#457B9D"}
   # e.g., continuous:  interpolate "magnitude" 1.0->#FEEDDE, 9.0->#A63603

2. SIZE/WIDTH driven by "{{size_property}}":
   {{size_rule}}
   # e.g., circle-radius: interpolate "magnitude" 1->2px, 5->8px, 9->20px
   # Must use ["interpolate", ["exponential", 1.5], ["get", "magnitude"], ...]

3. OPACITY driven by "{{opacity_property}}":
   {{opacity_rule}}
   # e.g., time-decay: features from today=1.0, 7 days ago=0.5, 30 days=0.2

4. FILTER:
   {{filter_expression}}
   # e.g., ["all", [">=", ["get", "magnitude"], 2.5], ["has", "location"]]

INTERACTION:
- Hover: highlight with {{hover_effect}}
  (e.g., increase radius by 50%, brighten fill by 20%, add stroke)
- Click: show popup with fields: {{popup_fields}}

PERFORMANCE:
- If feature_count > 10,000: suggest clustering or viewport filtering
- Use "feature-state" for hover instead of re-rendering the layer
- Prefer "match" over "case" for categorical data (faster evaluation)

Return:
A. map.addSource() call
B. map.addLayer() call(s) -- separate layers for fill and outline if polygon
C. Hover handler using map.setFeatureState()
D. Click handler with popup
E. Performance notes specific to this dataset size

DARK ARTS TIP: For large point datasets (>50k), use two layers: a "case"
fill layer at low zoom with aggressive clustering, and the full expression
layer at high zoom with a minzoom of 10. MapLibre evaluates expressions per
feature per frame -- fewer visible features = faster rendering.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{geometry_type}}` | point | point / line / polygon |
| `{{feature_count}}` | 50000 | Determines performance advice |
| `{{color_property}}` | type | Property name for color encoding |
| `{{size_property}}` | magnitude | Property name for size encoding |
| `{{popup_fields}}` | ["name","magnitude","time"] | Fields shown on click |

#### Expected Output

Complete JavaScript code with `addSource()`, `addLayer()` (with full MapLibre expressions), hover/click handlers using `featureState`, and performance annotations as comments.

#### Validation Checklist

- [ ] All expression syntax is valid MapLibre 4.x (no deprecated Mapbox-only features)
- [ ] `"match"` expressions include a fallback/default value
- [ ] Hover uses `featureState` (not `queryRenderedFeatures` + `setPaintProperty`)
- [ ] Interpolation type matches data nature (exponential for sizes, linear for colors)
- [ ] Popup HTML escapes user data to prevent XSS

#### Cost Optimization

- Provide the exact property names and types in the first message; guessing costs a follow-up.
- If you do not need interaction (hover/click), omit that section to save 30% of output tokens.

#### Related Pages

- [2D Mapping (layer management)](../js-bindbox/2d-mapping.md)
- [Performance Optimization (feature-state, clustering)](../js-bindbox/performance-optimization.md)
- [Data Formats and Loading (GeoJSON, vector tiles)](../js-bindbox/data-formats-loading.md)

#### Extensibility

- Chain with P1 (Scientific Color Palette) to get the color array for the match expression.
- Add P9 (Animated Styles) for smooth transitions between data states.
- Combine with P20 (Custom Marker and Cluster Styling) for clustered point layers.

---

### Prompt 7 -- Dark Mode Map Style

#### Scenario (actual use / 实际场景)

You have an existing light basemap style (or no style at all) and need a polished dark mode variant. Dark maps are standard for dashboards, night-mode apps, and data visualization where bright overlays need a muted background. Getting dark mode right is harder than it looks -- naive inversion creates unreadable labels and garish water.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web developer or designer implementing dark mode |
| **AI** | Dark UI specialist and MapLibre style engineer |

#### Prompt Template

```text
You are a dark-UI specialist and MapLibre style engineer. Create a production-
quality dark mode basemap style, OR convert the following light style to dark.

INPUT (choose one):
A. START FROM SCRATCH -- tile source: {{tile_source}}, URL: {{source_url}}
B. CONVERT EXISTING -- paste or reference a light style JSON: {{light_style_ref}}

DARK MODE PALETTE:
  Background:     {{dark_bg}}          # e.g., #0D1117 (GitHub dark)
  Surface:        {{dark_surface}}     # e.g., #161B22
  Border:         {{dark_border}}      # e.g., #30363D
  Text primary:   {{dark_text}}        # e.g., #C9D1D9
  Text secondary: {{dark_text_sec}}    # e.g., #8B949E
  Accent:         {{dark_accent}}      # e.g., #58A6FF

CONVERSION RULES:
1. WATER: darken to near-background; avoid pure black. Use {{water_dark}}.
   No wave patterns -- flat fill only.
2. LAND/BACKGROUND: {{dark_bg}}. Parks: barely perceptible tint (L* +3 above bg).
3. ROADS:
   - Reduce contrast from light mode by 70%
   - Motorway: {{dark_border}}, width same as light
   - Minor roads: visible only z13+, one shade above background
   - Road casings: 1px darker than fill (not lighter as in light mode)
4. BUILDINGS: {{dark_surface}}, z14+, flat fill. No extrusion shadows
   (they disappear against dark backgrounds).
5. LABELS:
   - Primary: {{dark_text}}, NOT pure white (#FFFFFF causes halation)
   - Secondary: {{dark_text_sec}}
   - Text halo: {{dark_bg}}, width 1.5px (wider than light mode -- dark halos
     need more spread to separate from dark backgrounds)
   - Road labels: {{dark_text_sec}}, z14+
6. BOUNDARIES: {{dark_border}}, dashed for admin level >= 4
7. HILLSHADE: highlight {{dark_surface}}, shadow {{dark_bg}}, opacity 0.05
   (almost invisible -- just enough to hint at terrain)
8. POIs: hidden by default. If shown, use icon-opacity 0.6.

CRITICAL DETAILS:
- "sky" layer: remove or set to transparent (sky gradients look wrong in dark)
- icon-color: invert any icons that were designed for light backgrounds
- raster layers (satellite, hillshade): multiply blend mode or reduce opacity

Return the complete style JSON. If converting, return ONLY the changed layers
as a diff, plus the full JSON.

DARK ARTS TIP: The number one dark mode mistake is insufficient halo width on
labels. In light mode, a 1px halo suffices because light backgrounds naturally
separate dark text. In dark mode, bump halos to 1.5--2px because dark-on-dark
edges blur together. Also, never use pure white text (#FFFFFF) -- it causes
"halation" (optical glow) on dark backgrounds. Use #C9D1D9 or #E6EDF3 instead.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{dark_bg}}` | #0D1117 | Main background |
| `{{dark_surface}}` | #161B22 | Elevated surfaces |
| `{{dark_border}}` | #30363D | Lines and borders |
| `{{dark_text}}` | #C9D1D9 | Primary label color |
| `{{tile_source}}` | OpenMapTiles | Tile source |

#### Expected Output

A complete MapLibre GL style JSON (version 8) optimized for dark mode, or a layer-by-layer diff from the input light style plus the full dark JSON.

#### Validation Checklist

- [ ] No pure white (#FFFFFF) used for any text
- [ ] Water is darker than land (not the same color)
- [ ] Road casings are darker than road fill (inverted from light mode)
- [ ] Label halos are >= 1.5px
- [ ] Hillshade opacity is <= 0.1 (barely visible)
- [ ] No "sky" layer or it is transparent

#### Cost Optimization

- If converting an existing style, paste only the layers array (not the full JSON) to reduce input tokens.
- Specify the dark palette upfront to avoid a "what shade of dark?" follow-up.

#### Related Pages

- [Performance Optimization (render cost of dark styles)](../js-bindbox/performance-optimization.md)
- [Framework Integration (CSS dark mode toggle)](../js-bindbox/framework-integration.md)

#### Extensibility

- Pair with P2 (Brand-Aligned Theme) for corporate dark mode.
- Use as the base for P21 (Synchronized Map + Chart Theme) so charts match.
- Add P8 (3D Extrusion) with adjusted lighting for dark backgrounds.

---

### Prompt 8 -- 3D Extrusion Styling

#### Scenario (actual use / 实际场景)

You want to add 3D building extrusions, data-driven height columns (e.g., population towers), or terrain-following fill-extrusion layers. This requires fill-extrusion paint properties, lighting configuration, and often a deck.gl overlay for advanced use cases.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web map developer adding 3D visualization |
| **AI** | 3D cartography and WebGL rendering specialist |

#### Prompt Template

```text
You are a 3D cartography specialist. Create fill-extrusion styling for MapLibre
GL JS with the following requirements:

USE CASE: {{use_case}}
# e.g., "3D building heights from OSM data"
# e.g., "Population density columns per census tract"
# e.g., "Revenue by sales region with height = revenue, color = growth"

TILE SOURCE: {{tile_source}}
SOURCE LAYER: {{source_layer}}           # e.g., "building" for OpenMapTiles
HEIGHT PROPERTY: {{height_property}}     # e.g., "height" or "render_height"
COLOR PROPERTY: {{color_property}}       # e.g., "type" (categorical) or "revenue" (numeric)

STYLING SPEC:
1. FILL-EXTRUSION BASE:
   - fill-extrusion-height: data-driven from "{{height_property}}"
     Scale: {{height_scale}}             # e.g., "1:1 meters" or "value * 100"
   - fill-extrusion-base: {{base_property}} or 0
   - fill-extrusion-opacity: {{opacity}}  # 0.7--0.9 typical

2. COLOR:
   {{color_spec}}
   # e.g., interpolate "revenue" $0->#4575B4, $500k->#D73027
   # e.g., match "type": residential->#A8D5BA, commercial->#F4A582

3. LIGHTING:
   - Anchor: "viewport" (light follows camera) or "map" (fixed sun position)
   - light.position: [{{azimuth}}, {{altitude}}, {{intensity}}]
     # Default: [1.15, 210, 30] -- afternoon sun from southwest
   - light.color: {{light_color}}         # #FFFFFF for neutral
   - light.intensity: {{light_intensity}} # 0.3--0.5

4. CAMERA:
   - Initial pitch: {{pitch}}             # 45--60 degrees
   - Initial bearing: {{bearing}}         # slight rotation looks better than 0
   - maxPitch: 70 (prevent horizon artifacts)

5. DECK.GL OVERLAY (if needed):
   - ColumnLayer or HexagonLayer for aggregated data
   - Provide deck.gl layer config that integrates with the MapLibre map instance
   - Include interleaving: map.addLayer(deckOverlay)

Return:
A. MapLibre style JSON layer definition(s) for fill-extrusion
B. Light configuration
C. Map initialization with pitch/bearing
D. Optional deck.gl overlay code
E. Performance notes (extrusion is expensive -- recommend maxzoom, feature limits)

DARK ARTS TIP: fill-extrusion-opacity below 0.6 causes z-fighting artifacts
where adjacent buildings overlap. Keep it at 0.7+. For dark basemaps, increase
light.intensity to 0.5 -- dark extrusions with weak light look like black blobs.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{use_case}}` | 3D building heights | Drives the overall approach |
| `{{height_property}}` | height | GeoJSON/tile property for extrusion height |
| `{{color_property}}` | type | Property for fill color |
| `{{pitch}}` | 50 | Camera pitch in degrees |
| `{{light_intensity}}` | 0.4 | 0.0--1.0; higher for dark basemaps |

#### Expected Output

MapLibre layer definitions with fill-extrusion paint properties, light configuration, camera setup JavaScript, optional deck.gl code, and performance annotations.

#### Validation Checklist

- [ ] fill-extrusion-opacity >= 0.7 to avoid z-fighting
- [ ] Light position creates visible shadow contrast
- [ ] Height expression handles null/missing values with a fallback
- [ ] maxPitch is set to prevent horizon rendering artifacts
- [ ] Performance: minzoom set so extrusions only render when meaningful

#### Cost Optimization

- If you only need buildings (not data columns), skip the deck.gl section.
- Specify the exact source-layer name to avoid trial and error.

#### Related Pages

- [3D Mapping (terrain, extrusions, deck.gl)](../js-bindbox/3d-mapping.md)
- [Performance Optimization (WebGL budget)](../js-bindbox/performance-optimization.md)

#### Extensibility

- Layer on P7 (Dark Mode) basemap for dramatic urban visualization.
- Combine with P9 (Animated Styles) for height transitions on data updates.
- Use P6 (Data-Driven Styling) expressions for the color encoding.

---

### Prompt 9 -- Animated and Interactive Styles

#### Scenario (actual use / 实际场景)

You want smooth visual feedback -- hover highlights, click selections, animated transitions between data states, pulsing markers, or flowing line animations. MapLibre supports feature-state for hover, but anything beyond that requires `requestAnimationFrame` or CSS/WebGL tricks.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Frontend developer adding polish to a web map |
| **AI** | Interactive visualization engineer and animation specialist |

#### Prompt Template

```text
You are an interactive visualization engineer specializing in web maps. Create
animated and interactive styling for the following MapLibre GL map:

INTERACTION TYPE: {{interaction_type}}
# Options:
# A. hover-highlight: brighten/enlarge on mouseover using feature-state
# B. click-select: persistent selection with outline/glow, multi-select support
# C. animated-transition: smooth morph between two data states (e.g., 2020->2025)
# D. pulsing-marker: repeating scale animation on point features
# E. flowing-line: dashed line with animated dash-offset (GPS track, route)
# F. time-series-playback: animate through timesteps with a scrubber

LAYER TYPE: {{layer_type}}               # circle | fill | line | symbol
SOURCE: {{source_description}}

SPECIFICATIONS:
{{#if hover_highlight}}
HOVER:
- Use map.setFeatureState({ id }, { hover: true }) on "mousemove"
- Paint property: ["case", ["boolean", ["feature-state", "hover"], false],
    {{hover_paint}}, {{default_paint}}]
- Transition: paint-property-transition: { duration: 200, delay: 0 }
- Clear on "mouseleave"
- Cursor: change to "pointer" on hover
{{/if}}

{{#if click_select}}
CLICK:
- Toggle "selected" feature-state on click
- Selected style: {{selected_style}}
  # e.g., thick outline #FFD700, fill-opacity 0.9
- Support multi-select with Shift+Click
- Deselect all on Escape key or click on empty area
- Emit custom event with selected feature IDs
{{/if}}

{{#if animated_transition}}
TRANSITION:
- Animate from data state A to state B over {{duration}}ms
- Use requestAnimationFrame to interpolate paint properties
- Easing: {{easing}}  # linear | ease-in-out | cubic-bezier(0.4, 0, 0.2, 1)
- Update the source data, not the style (for smooth GeoJSON morphing)
{{/if}}

{{#if pulsing_marker}}
PULSE:
- Animate circle-radius between {{min_radius}} and {{max_radius}}
- Animation duration: {{pulse_duration}}ms per cycle
- Use requestAnimationFrame + map.setPaintProperty()
- Optional: animate circle-opacity inversely (larger = more transparent)
{{/if}}

{{#if flowing_line}}
FLOW:
- Dashed line: line-dasharray [{{dash}}, {{gap}}]
- Animate line-dashoffset using requestAnimationFrame
- Speed: {{flow_speed}} pixels per second
- Direction: follow line geometry direction
{{/if}}

Return:
A. Complete JavaScript code with map setup, source, layer, and event handlers
B. CSS for cursor changes and any HTML overlays
C. Cleanup function to remove listeners and cancel animation frames
D. Performance notes (animation budget: aim for <4ms per frame)

DARK ARTS TIP: Never animate paint properties directly with setPaintProperty()
in a requestAnimationFrame loop for large datasets -- it triggers a full style
recomputation. Instead, use feature-state for per-feature animations (hover,
select) and reserve setPaintProperty() for global changes (opacity of entire
layer). For line flow animation, manipulate line-dashoffset which is GPU-
accelerated and does not trigger relayout.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{interaction_type}}` | hover-highlight | Choose one or combine |
| `{{layer_type}}` | fill | circle / fill / line / symbol |
| `{{hover_paint}}` | #FFD700, opacity 0.9 | Hover appearance |
| `{{pulse_duration}}` | 1500 | Milliseconds per pulse cycle |
| `{{flow_speed}}` | 20 | Pixels per second for line flow |

#### Expected Output

Complete JavaScript with event handlers, animation loop, cleanup function, CSS, and performance notes. Code should be copy-paste ready.

#### Validation Checklist

- [ ] All event listeners have corresponding removal in cleanup function
- [ ] `requestAnimationFrame` calls are cancelled in cleanup (`cancelAnimationFrame`)
- [ ] Hover uses `featureState`, not `queryRenderedFeatures` + `setPaintProperty`
- [ ] Animation loop measures elapsed time (not frame count) for consistent speed
- [ ] Cursor changes to "pointer" on interactive features

#### Cost Optimization

- Request only the interaction types you need. Each type adds ~30 lines of output.
- If your features have no stable `id`, mention that -- the AI will add `generateId: true` to the source.

#### Related Pages

- [Realtime, Offline, and Advanced (WebSocket updates, live data)](../js-bindbox/realtime-offline-advanced.md)
- [Performance Optimization (animation frame budget)](../js-bindbox/performance-optimization.md)
- [Framework Integration (React state + map state sync)](../js-bindbox/framework-integration.md)

#### Extensibility

- Combine hover-highlight with P6 (Data-Driven Styling) for rich interactive data maps.
- Use time-series-playback with P21 (Synchronized Map + Chart) for temporal dashboards.
- Add pulsing markers to P20 (Custom Marker and Cluster Styling) for alert indicators.

---

## 3. Desktop GIS Styling

---

### Prompt 10 -- QGIS QML Style Generator

#### Scenario (actual use / 实际场景)

You have a vector layer in QGIS and need a reusable QML style file with rule-based, graduated, or categorized symbology. Manually configuring complex rule trees through the GUI is tedious and error-prone. The AI can generate the QML XML directly.

#### Roles

| Role | Description |
|------|-------------|
| **User** | GIS analyst working in QGIS |
| **AI** | QGIS style author and OGC symbology specialist |

#### Prompt Template

```text
You are a QGIS style author. Generate a complete QML style file for the
following layer:

LAYER: {{layer_name}}
GEOMETRY: {{geometry_type}}              # Polygon | LineString | Point
CRS: {{crs}}                            # e.g., EPSG:4326

SYMBOLOGY TYPE: {{symbology_type}}
# Options: graduated | categorized | rule-based | single-symbol

{{#if graduated}}
GRADUATED:
  Attribute: {{attribute}}
  Method: {{method}}                     # jenks | quantile | equal_interval | pretty_breaks
  Classes: {{n_classes}}
  Color ramp: {{color_ramp}}             # e.g., "Viridis" | custom hex list
  Class breaks: {{breaks}}               # optional; if omitted, compute from method
  Symbol:
    Fill opacity: {{fill_opacity}}       # 0--100
    Stroke color: {{stroke_color}}       # e.g., #333333
    Stroke width: {{stroke_width}}mm
{{/if}}

{{#if categorized}}
CATEGORIZED:
  Attribute: {{attribute}}
  Categories:
  {{#each categories}}
    "{{value}}": fill={{fill_hex}}, stroke={{stroke_hex}}, label="{{label}}"
  {{/each}}
  Default symbol: fill={{default_fill}}, label="Other"
{{/if}}

{{#if rule_based}}
RULE-BASED:
  {{#each rules}}
  Rule {{index}}: "{{name}}"
    Filter: {{qgis_expression}}          # e.g., "population" > 100000
    Symbol: fill={{fill}}, stroke={{stroke}}, width={{width}}
    Scale range: {{min_scale}} -- {{max_scale}}
    Else: {{is_else}}
  {{/each}}
{{/if}}

LABELING:
  Enabled: {{label_enabled}}
  Field: {{label_field}}
  Font: {{font_family}}, {{font_size}}pt
  Color: {{label_color}}
  Halo: size={{halo_size}}mm, color={{halo_color}}
  Placement: {{placement}}               # Around Point | Parallel | Curved | Horizontal
  Priority: {{priority_field}}           # Higher value = placed first
  Scale-dependent: show at 1:{{min_denom}} -- 1:{{max_denom}}

OUTPUT:
- Complete QML XML with <qgis> root, <renderer-v2>, <labeling>, <symbol> elements
- Include <edittypes> and <fieldConfiguration> if label field is specified
- Add <layerproperties> for default scale range

The QML must be importable via: Layer > Properties > Style > Load Style.

DARK ARTS TIP: QGIS QML files silently ignore unknown elements, so you can
embed custom metadata in <customproperties> for documentation. Add a
<property key="ai_generated" value="true"/> so future users know the style
was machine-generated and can regenerate it from this prompt.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{layer_name}}` | census_tracts | Must match the layer name in QGIS |
| `{{symbology_type}}` | graduated | graduated / categorized / rule-based |
| `{{attribute}}` | population_density | Classification field |
| `{{method}}` | jenks | Classification method |
| `{{color_ramp}}` | Viridis | Or list of hex codes |

#### Expected Output

A complete QML XML file (typically 80--200 lines) with `<renderer-v2>`, `<symbols>`, `<labeling>`, and `<customproperties>` sections.

#### Validation Checklist

- [ ] XML is well-formed (close all tags, escape special characters)
- [ ] `<renderer-v2 type="...">` matches the symbology type
- [ ] Color values are in hex format without alpha (QGIS uses separate opacity)
- [ ] Label placement type matches the geometry type
- [ ] Scale denominators are in correct order (min_denom > max_denom)

#### Cost Optimization

- If you have known class breaks, provide them to skip the "compute breaks" instruction.
- Omit labeling section if you only need symbology.

#### Related Pages

- [Desktop GIS (QGIS features and plugins)](../tools/desktop-gis.md)
- [Python Libraries (PyQGIS for automation)](../tools/python-libraries.md)

#### Extensibility

- Feed class breaks from P4 (Thematic Map Classification) into the graduated renderer.
- Use colors from P1 (Scientific Color Palette) in the color ramp.
- Chain with P12 (Print-Ready Style) for map composition export.

---

### Prompt 11 -- SLD for GeoServer

#### Scenario (actual use / 实际场景)

You are publishing a WMS layer through GeoServer and need an OGC SLD (Styled Layer Descriptor) document. SLD 1.1 XML is verbose and easy to get wrong. The AI generates valid, complete SLD with rule-based, scale-dependent symbology and label placement.

#### Roles

| Role | Description |
|------|-------------|
| **User** | GIS server administrator or backend developer |
| **AI** | OGC standards expert and GeoServer configuration specialist |

#### Prompt Template

```text
You are an OGC standards expert. Write a valid SLD 1.1.0 document for publishing
a {{geometry_type}} layer through GeoServer.

LAYER: {{layer_name}}
TITLE: {{title}}
ABSTRACT: {{abstract}}

RULES:
{{#each rules}}
Rule {{index}}: "{{name}}"
  Filter: <ogc:PropertyIs{{operator}}>
            <ogc:PropertyName>{{property}}</ogc:PropertyName>
            <ogc:Literal>{{value}}</ogc:Literal>
          </ogc:PropertyIs{{operator}}>
  MinScaleDenominator: {{min_scale}}
  MaxScaleDenominator: {{max_scale}}
  Symbolizer:
    {{#if polygon}}
    <se:PolygonSymbolizer>
      <se:Fill><se:SvgParameter name="fill">{{fill_hex}}</se:SvgParameter>
              <se:SvgParameter name="fill-opacity">{{opacity}}</se:SvgParameter></se:Fill>
      <se:Stroke><se:SvgParameter name="stroke">{{stroke_hex}}</se:SvgParameter>
                 <se:SvgParameter name="stroke-width">{{width}}</se:SvgParameter></se:Stroke>
    </se:PolygonSymbolizer>
    {{/if}}
    {{#if point}}
    <se:PointSymbolizer>
      <se:Graphic>
        <se:Mark><se:WellKnownName>circle</se:WellKnownName>
          <se:Fill>...</se:Fill><se:Stroke>...</se:Stroke>
        </se:Mark>
        <se:Size>{{size}}</se:Size>
      </se:Graphic>
    </se:PointSymbolizer>
    {{/if}}
{{/each}}

LABELS:
  Property: {{label_property}}
  Font: {{font_family}}, size {{font_size}}
  Color: {{label_color}}
  Halo radius: {{halo_radius}}
  Placement:
    {{#if point}} <se:PointPlacement> with anchor and displacement {{/if}}
    {{#if line}} <se:LinePlacement> with perpendicular offset {{/if}}
  Vendor options:
    - conflictResolution: true
    - goodnessOfFit: 0.3
    - maxDisplacement: 50
    - group: {{group_labels}}           # true for road labels

OUTPUT:
- Complete SLD 1.1.0 XML with proper namespaces:
  xmlns:sld, xmlns:se, xmlns:ogc, xmlns:xlink
- Valid against the OGC SLD 1.1 schema
- Include <?xml?> declaration and encoding
- Ready for upload to GeoServer via REST API or admin console

DARK ARTS TIP: GeoServer vendor options are where the real power lives.
<VendorOption name="spaceAround">5</VendorOption> prevents label overlap better
than conflictResolution alone. For road labels, always add
<VendorOption name="followLine">true</VendorOption> and
<VendorOption name="maxAngleDelta">25</VendorOption> to prevent ugly bends.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{layer_name}}` | census_tracts | GeoServer layer name |
| `{{geometry_type}}` | Polygon | Polygon / Point / LineString |
| `{{rules}}` | (see template) | Array of classification rules |
| `{{label_property}}` | name | Attribute for labels |

#### Expected Output

A complete SLD 1.1.0 XML document (typically 100--300 lines) with proper OGC namespaces, rules, symbolizers, label placement, and GeoServer vendor options.

#### Validation Checklist

- [ ] All OGC namespaces are declared in the root element
- [ ] Scale denominators are consistent (min < max within each rule)
- [ ] Filter expressions use correct OGC operators
- [ ] Label halo is defined (Halo element with Radius and Fill)
- [ ] VendorOption elements are inside the TextSymbolizer

#### Cost Optimization

- If you only need symbolizers (no labels), omit the LABELS section.
- For simple graduated symbology, provide the break values to avoid the AI guessing.

#### Related Pages

- [Server Publishing (GeoServer, MapServer)](../tools/server-publishing.md)
- [Web Mapping (WMS/WFS clients)](../tools/web-mapping.md)

#### Extensibility

- Use colors from P1 (Scientific Color Palette) in the SLD fill values.
- Chain with P10 (QGIS QML) for identical styling across desktop and server.
- Reference class breaks from P4 (Thematic Map Classification).

---

### Prompt 12 -- Print-Ready Cartographic Style

#### Scenario (actual use / 实际场景)

You are preparing a map for print: journal publication, government report, poster, or atlas page. Print maps have fundamentally different requirements than screen maps -- CMYK color space, higher DPI line weights, physical font sizes, and no interactivity to compensate for poor design.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Cartographer preparing print-quality maps |
| **AI** | Print production specialist and traditional cartographer |

#### Prompt Template

```text
You are a print production specialist and traditional cartographer. Design
styling rules for a {{map_type}} map intended for print.

OUTPUT MEDIUM:
  Format: {{format}}                    # A4 portrait | A3 landscape | A0 poster
  DPI: {{dpi}}                          # 300 for publication, 150 for draft
  Color mode: {{color_mode}}            # CMYK | RGB (for digital print)
  Bleed: {{bleed}}mm                    # 3mm standard

MAP SPECIFICATIONS:
  Scale: 1:{{scale}}
  Extent: {{extent_description}}
  Projection: {{projection}}

STYLING RULES FOR PRINT:

1. LINE WEIGHTS (at {{dpi}} DPI):
   - Hairline minimum: {{hairline}}mm   # 0.1mm at 300 DPI = 1.18px
   - Country boundary: {{boundary_weight}}mm
   - State boundary: {{state_weight}}mm
   - Coastline: {{coast_weight}}mm
   - Road hierarchy:
     Motorway: {{motorway_weight}}mm casing + {{motorway_fill_weight}}mm fill
     Primary: {{primary_weight}}mm
     Secondary: {{secondary_weight}}mm
     Minor: {{minor_weight}}mm (ensure it survives halftone screening)
   - CRITICAL: lines below 0.08mm will disappear in offset printing

2. COLORS (CMYK-safe):
   - Provide all colors as CMYK percentages AND hex equivalents
   - Avoid CMYK values where any channel is below 5% (dot gain makes them
     unpredictable)
   - Water: C={{water_c}} M={{water_m}} Y={{water_y}} K={{water_k}}
   - Land: ...
   - For sequential palettes: ensure CMYK total ink coverage < 300%
     (ideal < 280% for uncoated paper)

3. FONTS:
   - Body labels: {{body_font}}, minimum size {{min_font_size}}pt
     (6pt absolute minimum for journals, 8pt for general audience)
   - Title: {{title_font}}, {{title_size}}pt
   - Ensure font is embeddable (no licensing restrictions for PDF)
   - Provide point sizes, not pixel sizes

4. MAP FURNITURE:
   - Scale bar: graphical (not text), showing {{scale_bar_units}},
     width = {{scale_bar_width}}mm, subdivisions at half and full
   - North arrow: simple, {{north_arrow_size}}mm, placed {{north_arrow_pos}}
   - Legend: box width {{legend_width}}mm, swatch size {{swatch_size}}mm
   - Title block: include title, subtitle, data source, date, CRS, author
   - Neatline: {{neatline_weight}}mm solid black frame

5. SERIES CONSISTENCY (if multiple maps):
   - All maps share identical: color palette, legend position, scale bar style,
     font hierarchy, neatline weight
   - Map number/title in consistent position

DELIVERABLES:
A. Color specification table (CMYK + hex + Pantone closest match)
B. Line weight specification table (mm + points + pixels at {{dpi}})
C. Font specification table (family, weight, size, tracking)
D. QGIS print layout template (.qpt) OR Sketched layout dimensions
E. Preflight checklist for the print shop

DARK ARTS TIP: The single most common print map failure is insufficient line
weight. What looks crisp at 100% on screen (0.5px) becomes invisible in print.
Rule of thumb: multiply your screen line weight by 2.5 for 300 DPI output.
And always request a proof print at actual size before the full run.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{format}}` | A4 portrait | Page size and orientation |
| `{{dpi}}` | 300 | Print resolution |
| `{{color_mode}}` | CMYK | CMYK for offset, RGB for digital |
| `{{scale}}` | 250000 | Map scale denominator |
| `{{body_font}}` | Noto Sans | Must be embeddable |

#### Expected Output

Specification tables for colors (CMYK/hex), line weights (mm/pt/px), fonts (family/size), a layout template, and a preflight checklist.

#### Validation Checklist

- [ ] No line weight below 0.08mm (will vanish in print)
- [ ] CMYK total ink < 300% for every color
- [ ] Minimum font size >= 6pt for all labels
- [ ] Scale bar is graphical, not just text ("1:250,000")
- [ ] Neatline is present and defined

#### Cost Optimization

- If producing a single map (not a series), omit the series consistency section.
- Provide paper stock (coated/uncoated) to get accurate CMYK advice in one round.

#### Related Pages

- [Desktop GIS (QGIS print composer)](../tools/desktop-gis.md)
- [CLI Tools (GDAL/OGR for data prep)](../tools/cli-tools.md)

#### Extensibility

- Use P1 (Scientific Color Palette) and convert the hex output to CMYK using this prompt.
- Feed into P10 (QGIS QML) for the symbology that matches these print specs.
- Apply P13 (Label Placement) rules with print-specific font sizes.

---

## 4. Labels and Typography

---

### Prompt 13 -- Intelligent Label Placement

#### Scenario (actual use / 实际场景)

Label placement is the hardest unsolved problem in automated cartography. You need collision-free, priority-ranked labels with proper halos, curved placement for rivers, and stacking for dense urban areas. This prompt covers both MapLibre (web) and QGIS (desktop) label engines.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Cartographer or web map developer |
| **AI** | Label placement algorithm specialist and typographic cartographer |

#### Prompt Template

```text
You are a label placement specialist. Configure intelligent label placement for
the following map:

PLATFORM: {{platform}}                   # MapLibre GL JS | QGIS | Both
FEATURE TYPE: {{feature_type}}           # cities | rivers | regions | roads | mixed
GEOMETRY: {{geometry}}                   # point | line | polygon
LABEL FIELD: {{label_field}}
FEATURE COUNT: {{feature_count}}         # affects collision density

PLACEMENT RULES:

1. POSITION PRIORITY (for point features):
   - Preferred: upper-right of anchor
   - Fallback order: upper-left, lower-right, lower-left, right, left
   - Max displacement: {{max_displacement}}px before giving up
   - For MapLibre:
     text-variable-anchor: ["top-right","top-left","bottom-right","bottom-left"]
     text-radial-offset: 1.2

2. LINE FOLLOWING (for rivers, roads):
   - symbol-placement: "line" (MapLibre) or "Curved" (QGIS)
   - Repeat distance: {{repeat_distance}}px (prevent duplicate labels)
   - Max angle between characters: {{max_angle}}deg (prevent illegible bends)
   - For MapLibre: text-max-angle: 30

3. POLYGON LABELING:
   - Placement: horizontal at polygon centroid, OR largest inscribed rectangle
   - Overlap: text-allow-overlap: false
   - For large polygons: repeat label if polygon spans > 3x label width

4. COLLISION AVOIDANCE:
   - Priority field: "{{priority_field}}"  # higher value = placed first
   - For MapLibre: symbol-sort-key: ["get", "{{priority_field}}"]
   - Group labels by category: {{group_categories}}
   - Buffer around labels: {{buffer}}px (prevents crowding)

5. TEXT STYLING:
   - Font: {{font_family}}
   - Size: {{font_size}}px (data-driven if needed)
   - Color: {{text_color}}
   - Halo color: {{halo_color}}, width: {{halo_width}}px
   - Letter spacing: {{letter_spacing}}em (use 0.05--0.1 for uppercase)
   - Text transform: {{text_transform}}  # uppercase | none

6. CURVED / ITALIC CONVENTIONS:
   - Water features: italic
   - Administrative regions: UPPERCASE, letter-spacing 0.1em
   - Cities: title case, bold for capitals
   - Mountains/peaks: italic, smaller size

RETURN:
A. MapLibre addLayer() with complete layout + paint properties
B. QGIS labeling configuration (PAL engine settings) as QML snippet
C. Priority ranking table with recommended sort-key values

DARK ARTS TIP: MapLibre's collision detection is greedy-first-come -- it places
labels in the order they appear in the tile, NOT by your priority field, unless
you set symbol-sort-key. Always set symbol-sort-key to your priority field,
AND set symbol-z-order to "source" (not "auto") to ensure deterministic
placement order.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{platform}}` | MapLibre GL JS | MapLibre / QGIS / Both |
| `{{feature_type}}` | cities | cities / rivers / roads / mixed |
| `{{priority_field}}` | population | Higher = placed first |
| `{{halo_width}}` | 1.5 | px for web, mm for print |

#### Expected Output

MapLibre `addLayer()` code with layout/paint properties, QGIS QML labeling snippet, and a priority ranking table.

#### Validation Checklist

- [ ] `symbol-sort-key` is set when priority matters
- [ ] `text-allow-overlap: false` is explicitly set
- [ ] Halo width > 0 for readability
- [ ] Line labels have `text-max-angle` set
- [ ] Repeat distance prevents duplicate labels on long features

#### Cost Optimization

- Specify the platform (MapLibre or QGIS) to avoid getting code for both.
- If all features are the same type (e.g., only cities), skip multi-geometry sections.

#### Related Pages

- [2D Mapping (symbol layers)](../js-bindbox/2d-mapping.md)
- [Desktop GIS (QGIS labeling engine)](../tools/desktop-gis.md)

#### Extensibility

- Combine with P14 (Bilingual Labels) for dual-language placement.
- Feed priority ranking into P15 (Dynamic Label Density) for zoom-dependent visibility.
- Use typography rules in P12 (Print-Ready Style) for publication maps.

---

### Prompt 14 -- Bilingual Labels (Chinese and English / 中英双语)

#### Scenario (actual use / 实际场景)

You are building a map for a Chinese-speaking audience that also needs English labels, or an international map of China. CJK (Chinese/Japanese/Korean) text has unique rendering challenges: no word boundaries, different baseline alignment, and font fallback chains. This prompt handles dual-label stacking, font configuration, and CJK-specific rendering.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web map developer or cartographer working with CJK content |
| **AI** | CJK typography specialist and multilingual cartographer |

#### Prompt Template

```text
You are a CJK typography specialist and multilingual cartographer. Configure
bilingual (Chinese + English) labels for a MapLibre GL JS map.

PRIMARY LANGUAGE: {{primary_lang}}       # zh-Hans | zh-Hant | ja | ko
SECONDARY LANGUAGE: {{secondary_lang}}   # en | fr | de
LABEL PROPERTIES:
  Chinese name: "{{zh_property}}"        # e.g., "name:zh" or "name_zh"
  English name: "{{en_property}}"        # e.g., "name:en" or "name_en"
  Fallback: "{{fallback_property}}"      # e.g., "name" (local name)

STACKING:
  Order: {{stack_order}}                 # zh-top | en-top
  Separator: {{separator}}              # newline | " / " | " "
  Example output: "北京\nBeijing" or "北京 / Beijing"

FONT CONFIGURATION:
  Chinese fonts: {{zh_fonts}}
  # Recommended chain: ["Noto Sans CJK SC Regular", "Noto Sans Regular"]
  English fonts: {{en_fonts}}
  # Recommended chain: ["Noto Sans Regular", "Open Sans Regular"]
  Font size ratio: {{size_ratio}}        # e.g., 1:0.75 (Chinese:English)

RENDERING DETAILS:
1. text-field expression for stacked bilingual labels:
   ["format",
     ["get", "{{zh_property}}"], { "font-scale": 1.0 },
     "\n", {},
     ["get", "{{en_property}}"], { "font-scale": {{en_scale}} }
   ]

2. text-font: must include CJK font FIRST in the array:
   ["Noto Sans CJK SC Regular", "Noto Sans Regular"]

3. CJK-specific settings:
   - text-writing-mode: ["horizontal", "vertical"]
     (for vertical labels on vertical text features, rare but supported)
   - text-max-width: {{max_width}}
     (CJK characters are wider; reduce max-width by ~30% vs Latin)
   - Character spacing: 0 for Chinese (CJK has built-in spacing)
   - Line height: {{line_height}} (1.2--1.4 for stacked labels)

4. Fallback logic:
   ["case",
     ["all", ["has", "{{zh_property}}"], ["has", "{{en_property}}"]],
     ["format", ["get", "{{zh_property}}"], {}, "\n", {},
      ["get", "{{en_property}}"], { "font-scale": {{en_scale}} }],
     ["has", "{{zh_property}}"],
     ["get", "{{zh_property}}"],
     ["get", "{{fallback_property}}"]
   ]

5. Zoom-dependent simplification:
   - z < 5: Chinese only (save space at country scale)
   - z 5--10: Chinese + English for major features
   - z > 10: Chinese + English for all labeled features

GLYPHS URL:
  Must serve CJK ranges (U+4E00--U+9FFF, U+3400--U+4DBF).
  Recommended: {{glyphs_url}}
  # e.g., "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf"
  # or self-hosted with CJK subsets

Return:
A. MapLibre addLayer() for city labels with bilingual stacking
B. MapLibre addLayer() for road labels (line placement, primary language only)
C. Glyphs configuration notes (which font PBFs to host)
D. CSS @font-face declarations if using custom web fonts for popups

DARK ARTS TIP: MapLibre's glyph PBF files are split by Unicode range (0-255,
256-511, ...). CJK characters live in ranges 19968+ which means the glyph
server must have those range files. If you see blank labels for Chinese text,
the glyph server is missing CJK ranges. Test by loading range "19968-20223.pbf"
directly in the browser. Self-hosting fonts with @AliasCSS/font-glyphs is
often more reliable than public glyph servers for CJK.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{primary_lang}}` | zh-Hans | Simplified Chinese |
| `{{zh_property}}` | name:zh | Tile property for Chinese name |
| `{{en_property}}` | name:en | Tile property for English name |
| `{{size_ratio}}` | 1:0.75 | Chinese to English font size ratio |
| `{{glyphs_url}}` | (see template) | Must include CJK ranges |

#### Expected Output

MapLibre `addLayer()` calls with `format` expressions for stacked bilingual labels, font configuration, glyph server notes, and CSS font-face declarations.

#### Validation Checklist

- [ ] CJK font is listed FIRST in the `text-font` array
- [ ] Fallback handles missing Chinese or English gracefully
- [ ] Glyph URL serves CJK Unicode ranges (test U+4E00 = "一")
- [ ] English sub-label has reduced `font-scale` (0.7--0.85)
- [ ] Zoom-dependent logic reduces to single language at low zoom

#### Cost Optimization

- If you only need Chinese labels (no English), skip the stacking -- just use `["get", "name:zh"]`.
- Specify the exact tile property names to avoid a clarification round.

#### Related Pages

- [2D Mapping (symbol layer configuration)](../js-bindbox/2d-mapping.md)
- [Tile Servers (glyph hosting)](../js-bindbox/tile-servers.md)

#### Extensibility

- Combine with P13 (Intelligent Label Placement) for collision-aware bilingual labels.
- Use with P15 (Dynamic Label Density) for zoom-dependent language switching.
- Apply font configuration to P5 (Complete Basemap Style) glyphs section.

---

### Prompt 15 -- Dynamic Label Density

#### Scenario (actual use / 实际场景)

At zoom 3, you want only country names. At zoom 8, add major cities. At zoom 12, add towns. At zoom 15, add street names. Managing zoom-dependent label visibility with importance ranking is essential for any multi-scale web map. This prompt generates a complete label density strategy.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web map developer working on multi-scale labeling |
| **AI** | Multi-scale cartography specialist |

#### Prompt Template

```text
You are a multi-scale cartography specialist. Design a zoom-dependent label
density strategy for a MapLibre GL JS map.

LABEL CATEGORIES (in priority order):
{{#each categories}}
{{index}}. {{name}}:
   Source layer: "{{source_layer}}"
   Label field: "{{label_field}}"
   Importance field: "{{importance_field}}"  # e.g., "rank", "population", "class"
   Min zoom: {{min_zoom}}
   Max zoom: {{max_zoom}}
   Importance thresholds by zoom:
     z{{low_zoom}}: show only if {{importance_field}} {{operator}} {{low_threshold}}
     z{{mid_zoom}}: show only if {{importance_field}} {{operator}} {{mid_threshold}}
     z{{high_zoom}}: show all
{{/each}}

DENSITY TARGETS:
  Maximum labels visible at any zoom: ~{{max_labels}}
  # Guide: 20--30 at z3, 50--80 at z8, 100--150 at z12, unlimited at z15+

CLUSTERING (for point labels at low zoom):
  At z < {{cluster_threshold}}: cluster nearby labels, show only highest-priority
  Method: {{cluster_method}}             # MapLibre clustering | pre-computed in tiles

FONT SIZE SCALING:
  Base size: {{base_size}}px at z10
  Scale factor: ["interpolate", ["linear"], ["zoom"],
    {{min_zoom}}, {{size_at_min}},
    {{max_zoom}}, {{size_at_max}}
  ]
  Importance modifier: multiply base by (1 + 0.2 * rank_normalized)

COLLISION PRIORITY:
  symbol-sort-key:
    ["*", -1, ["get", "{{importance_field}}"]]  # negate so highest value = first
  symbol-z-order: "source"

Return:
A. One map.addLayer() call per label category with zoom-dependent filters
B. A zoom-level matrix table:
   | Zoom | Countries | States | Cities    | Towns    | Streets |
   |------|-----------|--------|-----------|----------|---------|
   | 3    | all       | --     | capitals  | --       | --      |
   | 5    | all       | major  | rank<=3   | --       | --      |
   | ...  | ...       | ...    | ...       | ...      | ...     |
C. Performance analysis: estimated label count per zoom level

DARK ARTS TIP: Do not rely solely on minzoom/maxzoom for density control --
they are binary (on/off). Instead, use zoom-dependent filter expressions:
[">=", ["get", "population"], ["interpolate", ["linear"], ["zoom"], 5, 500000, 10, 50000, 14, 0]]
This creates a smooth "fade-in" of labels as you zoom in, matching the density
to the available screen space at each level.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{categories}}` | [countries, states, cities, towns, streets] | In priority order |
| `{{max_labels}}` | 50 at z8 | Density target |
| `{{importance_field}}` | population | Ranking attribute |
| `{{cluster_threshold}}` | 8 | Below this zoom, cluster |

#### Expected Output

Multiple `addLayer()` calls with zoom-dependent filter expressions, a zoom-level matrix table, and estimated label counts per zoom.

#### Validation Checklist

- [ ] Higher-priority categories have lower `symbol-sort-key` values
- [ ] Filter expressions use interpolated thresholds, not hard minzoom cutoffs
- [ ] No zoom level exceeds the density target by more than 20%
- [ ] Capitals and country names are always visible at their minimum zoom
- [ ] Street labels appear only at z14+ (they overwhelm at lower zooms)

#### Cost Optimization

- Provide the importance field name and its value range in the first message.
- If your tiles have pre-computed ranks, state that to skip the "how to rank" discussion.

#### Related Pages

- [2D Mapping (layer filtering)](../js-bindbox/2d-mapping.md)
- [Performance Optimization (label rendering cost)](../js-bindbox/performance-optimization.md)

#### Extensibility

- Combine with P13 (Intelligent Label Placement) for collision-aware density control.
- Use with P14 (Bilingual Labels) to control when the secondary language appears.
- Feed the zoom matrix into P5 (Complete Basemap Style) for integrated label layers.

---

## 5. Legend and Map Furniture

---

### Prompt 16 -- Responsive Map Legend

#### Scenario (actual use / 实际场景)

Your web map needs a legend that is not a static image but an interactive HTML/CSS component. It should toggle layers on and off, synchronize with the current map state (which layers are visible, what zoom level), collapse on mobile, and update when the data changes.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Frontend developer building a map UI |
| **AI** | UI/UX engineer specializing in map interfaces |

#### Prompt Template

```text
You are a UI/UX engineer specializing in map interfaces. Build a responsive,
interactive map legend component for a MapLibre GL JS application.

MAP LAYERS:
{{#each layers}}
  "{{id}}": {
    type: "{{type}}",                    # fill | circle | line | symbol
    label: "{{label}}",
    style: {{style_summary}},            # e.g., "graduated blue 5-class"
    toggleable: {{toggleable}},          # true | false
    initially_visible: {{visible}}
  }
{{/each}}

LEGEND TYPES NEEDED:
{{#each legend_items}}
  - {{layer_id}}: {{legend_type}}
    # Options:
    # graduated: color swatches with class ranges
    # categorized: color + label per category
    # continuous: gradient bar with min/max labels
    # proportional: circles of varying size
    # line-style: line samples with dash patterns
{{/each}}

RESPONSIVE BEHAVIOR:
  Desktop (> {{breakpoint}}px):
    Position: {{desktop_position}}       # top-right | bottom-left | sidebar
    Width: {{desktop_width}}px
    Expanded by default

  Mobile (<= {{breakpoint}}px):
    Collapsed to a button "Legend" / icon
    Slides in from {{mobile_direction}}  # bottom | right
    Max height: 60vh

INTERACTIVITY:
  - Click layer name: toggle visibility (map.setLayoutProperty)
  - Opacity slider per layer (optional): {{include_opacity_slider}}
  - Highlight legend item on map hover (bidirectional sync)
  - Update legend when map zoom changes (hide items for invisible layers)

STYLING:
  Theme: {{theme}}                       # light | dark | auto (match map)
  Font: {{font}}
  Border radius: {{border_radius}}px
  Shadow: {{shadow}}
  Background: semi-transparent ({{bg_opacity}})

Return:
A. HTML structure
B. CSS (responsive, with media queries)
C. JavaScript: initialization, event binding, map sync
D. Accessibility: ARIA labels, keyboard navigation, screen reader support

DARK ARTS TIP: Do not build the legend from scratch with innerHTML. Instead,
read the map style programmatically: map.getStyle().layers gives you every
layer's paint properties. Iterate to auto-generate legend items:
  map.getStyle().layers.forEach(layer => {
    const color = map.getPaintProperty(layer.id, 'fill-color');
    // build legend swatch from actual paint values
  });
This way the legend is always in sync with the style, even if someone edits
the style JSON later.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{layers}}` | (see template) | Array of map layers |
| `{{breakpoint}}` | 768 | Mobile breakpoint in px |
| `{{desktop_position}}` | top-right | Legend position |
| `{{theme}}` | dark | Match your map theme |

#### Expected Output

HTML, CSS, and JavaScript for a responsive legend component. Includes ARIA attributes, media queries, and map event listeners for synchronization.

#### Validation Checklist

- [ ] Legend reflects actual map paint properties, not hardcoded values
- [ ] Toggle buttons use `map.setLayoutProperty(id, 'visibility', ...)`
- [ ] Mobile collapsed state is accessible (button has ARIA label)
- [ ] Legend updates on `map.on('zoomend')` for zoom-dependent layers
- [ ] Colors in legend swatches match the map exactly

#### Cost Optimization

- If you use React/Vue, specify the framework to get a component instead of vanilla JS.
- Omit the opacity slider if not needed -- it adds significant complexity.

#### Related Pages

- [Framework Integration (React/Vue components)](../js-bindbox/framework-integration.md)
- [2D Mapping (layer management API)](../js-bindbox/2d-mapping.md)

#### Extensibility

- Pair with P3 (Bivariate Color Scheme) for a matrix legend component.
- Use with P21 (Synchronized Map + Chart Theme) to share color tokens between legend and charts.
- Add P19 (SVG Icon System) icons to legend entries.

---

### Prompt 17 -- Scale Bar and North Arrow

#### Scenario (actual use / 实际场景)

Your map needs a scale bar and north arrow -- either as SVG for a static/print map or as a dynamic HTML/JS component for a web map that updates as the user pans and zooms. Web Mercator distortion at high latitudes makes scale bars non-trivial.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Cartographer or web developer adding map furniture |
| **AI** | Cartographic furniture designer and projection mathematician |

#### Prompt Template

```text
You are a cartographic furniture designer. Create a scale bar and north arrow
for the following map:

MAP TYPE: {{map_type}}                   # web (MapLibre) | print (QGIS/static)
PROJECTION: {{projection}}              # EPSG:3857 (Web Mercator) | EPSG:4326 | ...

{{#if web}}
WEB SCALE BAR:
  - Dynamic: updates on map move/zoom events
  - Units: {{units}}                     # metric | imperial | both (dual bar)
  - Max width: {{max_width}}px
  - Position: {{position}}               # inside map as overlay | outside in UI
  - Style: {{style}}                     # simple line | alternating black/white blocks
  - Latitude correction: account for Web Mercator distortion
    true_scale = nominal_scale * cos(latitude_radians)
  - Implementation:
    map.on('moveend', () => {
      const center = map.getCenter();
      const metersPerPixel = /* calculate from zoom + latitude */;
      updateScaleBar(metersPerPixel);
    });

WEB NORTH ARROW:
  - Rotates with map bearing: map.getBearing()
  - SVG arrow that transforms: transform: rotate(${-bearing}deg)
  - Hidden when bearing === 0 (north is up, arrow is redundant)
  - Smooth CSS transition on rotation
{{/if}}

{{#if print}}
PRINT SCALE BAR:
  - Fixed scale: 1:{{scale}}
  - Physical width: {{bar_width_mm}}mm
  - Units: {{units}}
  - Style: alternating blocks (cartographic standard)
  - Subdivisions: {{subdivisions}} (e.g., 5 primary, 5 subdivision on first block)
  - Font: {{font}}, size {{font_size}}pt
  - Line weight: {{line_weight}}mm

PRINT NORTH ARROW:
  - Style: {{arrow_style}}              # simple | compass rose | minimal
  - Size: {{arrow_size}}mm
  - Include magnetic declination note: {{include_declination}}
{{/if}}

RETURN:
A. SVG code for the scale bar
B. SVG code for the north arrow
C. JavaScript for dynamic updates (web only)
D. CSS for positioning and animation
E. Web Mercator distortion formula and implementation

DARK ARTS TIP: MapLibre's built-in ScaleControl is adequate for most cases:
  map.addControl(new maplibregl.ScaleControl({ maxWidth: 150, unit: 'metric' }));
But it does not support dual-unit display, custom styling, or alternating blocks.
If you need any of those, build a custom control. Also, at latitudes above 60N
(Scandinavia, Alaska), Web Mercator distortion exceeds 50% -- consider adding
a distortion disclaimer text next to the scale bar.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{map_type}}` | web | web or print |
| `{{projection}}` | EPSG:3857 | Determines distortion correction |
| `{{units}}` | metric | metric / imperial / both |
| `{{style}}` | alternating blocks | Visual style of scale bar |

#### Expected Output

SVG code for scale bar and north arrow, JavaScript for dynamic updates (web), CSS for positioning, and the Web Mercator distortion formula.

#### Validation Checklist

- [ ] Scale bar accounts for latitude-dependent Web Mercator distortion
- [ ] North arrow rotates correctly (negative bearing for visual rotation)
- [ ] Print scale bar has subdivisions on the first segment
- [ ] Units are clearly labeled (km, mi, not ambiguous abbreviations)
- [ ] SVG uses `viewBox` for resolution independence

#### Cost Optimization

- For web maps, consider using MapLibre's built-in `ScaleControl` and only request custom north arrow.
- If your map never rotates (bearing always 0), skip the north arrow.

#### Related Pages

- [2D Mapping (map controls)](../js-bindbox/2d-mapping.md)
- [Desktop GIS (QGIS print layout furniture)](../tools/desktop-gis.md)

#### Extensibility

- Include in P18 (Map Layout Composition) for complete map furniture placement.
- Use print version in P12 (Print-Ready Cartographic Style) specifications.
- Style consistently with P16 (Responsive Legend) using shared CSS variables.

---

### Prompt 18 -- Map Layout Composition

#### Scenario (actual use / 实际场景)

You need to compose a complete map page: title, subtitle, map body, legend, scale bar, north arrow, source/attribution, and inset/locator map. Whether for a web page or a print layout, the spatial arrangement of these elements follows established cartographic conventions.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Cartographer or developer composing a map page |
| **AI** | Map composition specialist and layout designer |

#### Prompt Template

```text
You are a map composition specialist. Design a complete map layout for:

MEDIUM: {{medium}}                       # web page | print A4 | print A3 | poster A0
ORIENTATION: {{orientation}}             # portrait | landscape
MAP PURPOSE: {{purpose}}                 # reference | thematic | dashboard | atlas page

ELEMENTS TO PLACE:
1. Title:    "{{title}}"
2. Subtitle: "{{subtitle}}"
3. Map body: {{map_dimensions}}          # e.g., full bleed | 80% of page | specific mm
4. Legend:   {{legend_type}}             # graduated | categorized | bivariate matrix
5. Scale bar: {{scale_bar_type}}         # graphical | text
6. North arrow: {{north_arrow}}          # yes | no (omit if north is always up)
7. Source/attribution: "{{source_text}}"
8. Inset map: {{inset}}                  # locator map | overview | none
9. Graticule: {{graticule}}              # yes | no
10. Map number: {{map_number}}           # for atlas series, e.g., "Sheet 12 of 48"

{{#if web}}
WEB LAYOUT:
  Viewport: {{viewport_width}}px x {{viewport_height}}px
  CSS approach: {{css_approach}}         # CSS Grid | Flexbox | absolute positioning
  Responsive breakpoints: {{breakpoints}} # e.g., 768px, 1024px

  Return:
  A. HTML structure with semantic elements
  B. CSS Grid template with named areas:
     ```css
     .map-page {
       display: grid;
       grid-template-areas:
         "title    title"
         "map      legend"
         "source   scalebar";
       grid-template-columns: 1fr {{legend_width}}px;
       grid-template-rows: auto 1fr auto;
     }
     ```
  C. Responsive CSS for mobile (legend collapses below map)
  D. JavaScript to initialize MapLibre in the map area
{{/if}}

{{#if print}}
PRINT LAYOUT:
  Page: {{page_size}} {{orientation}}
  Margins: top={{margin_top}}mm, right={{margin_right}}mm,
           bottom={{margin_bottom}}mm, left={{margin_left}}mm
  Map frame: neatline {{neatline_weight}}mm

  Return:
  A. Dimensioned layout diagram (ASCII or SVG)
  B. QGIS print layout template (.qpt) XML structure
  C. Element positions in mm from top-left:
     | Element      | X (mm) | Y (mm) | W (mm) | H (mm) |
     |--------------|--------|--------|--------|--------|
     | Title        | ...    | ...    | ...    | ...    |
     | Map frame    | ...    | ...    | ...    | ...    |
     | Legend       | ...    | ...    | ...    | ...    |
     | Scale bar    | ...    | ...    | ...    | ...    |
     | ...          | ...    | ...    | ...    | ...    |
{{/if}}

COMPOSITION RULES:
- Map body gets the largest area (minimum 60% of page)
- Legend placed right of or below the map, never overlapping data-dense areas
- Title at top, source/attribution at bottom
- Scale bar near bottom of map frame
- North arrow top-right of map frame (only if map is rotated)
- Inset map: bottom-left or top-left corner, with its own neatline
- Visual hierarchy: Title > Map > Legend > Scale bar > Source

DARK ARTS TIP: For web layouts, CSS Grid with named template areas is far
superior to absolute positioning -- it reflows naturally at different viewport
sizes. For print, QGIS print composer is the tool of choice, but define all
positions in mm from the start (do not eyeball it). A consistent grid system
(e.g., 5mm grid snap in QGIS) prevents misalignment across a map series.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{medium}}` | web page | web / print A4 / poster |
| `{{orientation}}` | landscape | portrait / landscape |
| `{{purpose}}` | thematic | reference / thematic / dashboard |
| `{{css_approach}}` | CSS Grid | CSS Grid / Flexbox |

#### Expected Output

For web: HTML, CSS Grid layout, and responsive styles. For print: a dimensioned element position table and QGIS layout template XML.

#### Validation Checklist

- [ ] Map body occupies >= 60% of total area
- [ ] Legend does not overlap data-dense map areas
- [ ] Title is the most prominent text element
- [ ] Source/attribution is present (legal requirement for most data)
- [ ] Print layout margins are >= 10mm (trimming safety)

#### Cost Optimization

- Specify web or print (not both) to halve the output.
- For web, state your CSS framework (Tailwind, Bootstrap) to get compatible code.

#### Related Pages

- [Framework Integration (layout in React/Vue)](../js-bindbox/framework-integration.md)
- [Desktop GIS (QGIS print composer)](../tools/desktop-gis.md)

#### Extensibility

- Place P16 (Responsive Legend) in the legend grid area.
- Place P17 (Scale Bar) in the scale bar grid area.
- Use with P12 (Print-Ready Style) for complete print map production.

---

## 6. Icons and Symbols

---

### Prompt 19 -- SVG Map Icon System

#### Scenario (actual use / 实际场景)

You need a consistent set of POI (point of interest) icons for your web map -- hospitals, schools, parks, restaurants, transit stops. The icons must work as a MapLibre sprite sheet, render crisply at small sizes, and maintain visual consistency. You also want QGIS-compatible SVG markers.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Map designer or web developer |
| **AI** | Icon designer and sprite sheet engineer |

#### Prompt Template

```text
You are an icon designer specializing in cartographic symbols. Create a consistent
SVG icon system for a {{map_theme}} map.

ICONS NEEDED:
{{#each icons}}
{{index}}. "{{name}}": {{description}}
   Semantic color: {{color}}             # or "neutral" for monochrome system
{{/each}}

DESIGN SPECIFICATIONS:
  Viewbox: 0 0 {{size}} {{size}}        # 24 for web, 32 for retina, 64 for print
  Style: {{style}}                       # flat | outlined | filled | material
  Stroke width: {{stroke_width}}px (for outlined style)
  Corner radius: {{corner_radius}}px (for background shapes)
  Background shape: {{bg_shape}}         # none | circle | rounded-square | pin
  Pixel alignment: yes (all coordinates snap to integer or half-pixel)
  Color system: {{color_system}}
  # mono: single color + transparent (user can recolor)
  # semantic: each icon uses its category color
  # dual-tone: fill + lighter fill for depth

SVG REQUIREMENTS:
  - No external dependencies (<use xlink:href> to external files)
  - No embedded fonts (convert text to paths)
  - No raster images embedded
  - Optimized (minimal path commands, no unnecessary groups)
  - param(fill) and param(outline) placeholders for QGIS compatibility:
    fill="param(fill) {{default_fill}}" stroke="param(outline) {{default_stroke}}"

SPRITE SHEET FOR MAPLIBRE:
  Generate a combined sprite sheet:
  A. sprite.png (or sprite@2x.png for retina):
     All icons arranged in a grid, {{sprite_padding}}px padding between icons
  B. sprite.json:
     {
       "{{icon_name}}": {
         "x": {{x}}, "y": {{y}},
         "width": {{size}}, "height": {{size}},
         "pixelRatio": {{pixel_ratio}}
       },
       ...
     }

  Usage in MapLibre:
    map.addImage('hospital', img);
    // or via style JSON: "sprite": "https://example.com/sprites/map-icons"

RETURN PER ICON:
1. Standalone SVG code block
2. QGIS-compatible SVG (with param placeholders)
3. Base64 data URI for inline JavaScript use

RETURN FOR SPRITE:
4. sprite.json index
5. Instructions for generating sprite.png using spritezero-cli or @AliasCSS/sprites

DARK ARTS TIP: For MapLibre, the most reliable icon loading method is
map.loadImage() + map.addImage(), not embedding in the sprite URL. Sprite URLs
require a server that serves both sprite.json and sprite.png at the same path.
If you do not have that infrastructure, load icons individually. For retina
displays, always provide @2x versions (double the pixel dimensions, same
viewbox) -- MapLibre checks for sprite@2x.json automatically.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{icons}}` | [{name: "hospital", description: "red cross on white circle"}] | Icon list |
| `{{size}}` | 24 | Viewbox dimensions |
| `{{style}}` | filled | flat / outlined / filled |
| `{{bg_shape}}` | circle | Background behind icon |
| `{{color_system}}` | semantic | mono / semantic / dual-tone |

#### Expected Output

Individual SVG code blocks per icon, QGIS-compatible variants, base64 data URIs, sprite.json index, and sprite generation instructions.

#### Validation Checklist

- [ ] All SVGs have identical viewBox dimensions
- [ ] Icons are visually distinguishable at 16x16px rendered size
- [ ] No embedded fonts or external references
- [ ] QGIS param(fill) placeholders are present
- [ ] sprite.json coordinates are accurate (no overlap)

#### Cost Optimization

- Request 5--10 icons per prompt. More than 15 may hit output limits.
- If you do not need QGIS compatibility, omit the param() placeholders.

#### Related Pages

- [2D Mapping (adding images and icons)](../js-bindbox/2d-mapping.md)
- [Desktop GIS (QGIS SVG markers)](../tools/desktop-gis.md)

#### Extensibility

- Use icons in P16 (Responsive Legend) entries.
- Apply to P20 (Custom Marker and Cluster Styling) for cluster center icons.
- Reference from P5 (Complete Basemap Style) sprite configuration.

---

### Prompt 20 -- Custom Marker and Cluster Styling

#### Scenario (actual use / 实际场景)

You have a large point dataset (thousands to millions of features) and need custom markers that cluster at low zoom. Clusters should show count, optionally a pie chart of categories, and expand smoothly on click. Individual markers need custom styling by category.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Web map developer handling large point datasets |
| **AI** | Cluster visualization specialist and MapLibre performance expert |

#### Prompt Template

```text
You are a cluster visualization specialist. Implement custom marker and cluster
styling for MapLibre GL JS.

DATA:
  Source: {{source_type}}                # GeoJSON | vector tiles
  Feature count: {{feature_count}}
  Category property: "{{category_prop}}" # e.g., "type"
  Categories: {{categories}}             # e.g., ["restaurant","hotel","museum","park"]
  Category colors: {{category_colors}}   # e.g., {"restaurant":"#E63946","hotel":"#457B9D",...}

INDIVIDUAL MARKERS (unclustered):
  Style: {{marker_style}}
  # Options:
  # A. circle-layer: simple colored circles by category
  # B. icon-image: custom SVG icons per category (from sprite)
  # C. html-marker: full HTML/CSS markers via Marker class
  Size: {{marker_size}}px
  Border: {{marker_border}}              # e.g., 2px white

CLUSTERING:
  Enabled: true
  Radius: {{cluster_radius}}px           # typically 50--80
  Max zoom: {{cluster_max_zoom}}         # stop clustering at this zoom

  Cluster appearance: {{cluster_style}}
  # Options:
  # A. sized-circle: radius proportional to point_count
  #    Small (< 10): {{small_color}}, radius 15px
  #    Medium (10-100): {{medium_color}}, radius 25px
  #    Large (> 100): {{large_color}}, radius 35px
  # B. pie-chart: donut/pie showing category distribution
  #    Requires clusterProperties to aggregate per category:
  #    clusterProperties: {
  #      "restaurant_count": ["+", ["case", ["==", ["get","type"],"restaurant"], 1, 0]],
  #      "hotel_count": ["+", ["case", ["==", ["get","type"],"hotel"], 1, 0]],
  #      ...
  #    }
  #    Render pie chart as canvas image or SVG overlay
  # C. text-only: show count number on neutral circle

  Cluster label: point_count (centered, white text, bold)

INTERACTIONS:
  - Click cluster: zoom to expand (flyTo the cluster's expansion zoom)
    map.on('click', 'clusters', (e) => {
      const source = map.getSource('{{source_id}}');
      source.getClusterExpansionZoom(clusterId, (err, zoom) => {
        map.flyTo({ center: coordinates, zoom });
      });
    });
  - Hover cluster: show tooltip "N features"
  - Click individual marker: popup with {{popup_fields}}
  - Hover individual: {{hover_effect}}

ANIMATION:
  - Clusters animate size on hover (scale 1 -> 1.15, 200ms ease-out)
  - Individual markers pulse briefly when first appearing after cluster expansion

Return:
A. map.addSource() with cluster configuration and clusterProperties
B. Three addLayer() calls: clusters, cluster-count, unclustered-point
C. Event handlers: cluster click (expand), marker click (popup), hover effects
D. Pie chart cluster implementation (if selected):
   - Canvas rendering function
   - map.addImage() for each unique pie composition
   - Or: HTML marker overlay approach
E. Performance notes for {{feature_count}} features

DARK ARTS TIP: MapLibre's built-in clustering recalculates on every zoom change
and cannot animate the transition. For smooth cluster animations (splitting,
merging), use supercluster directly (it is the library MapLibre uses internally)
and manage the cluster lifecycle yourself. Also, clusterProperties expressions
are evaluated per feature per zoom -- keep them simple. Complex case/match
expressions in clusterProperties can freeze the map with >100k features.
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{feature_count}}` | 25000 | Drives performance advice |
| `{{category_prop}}` | type | Property for categorization |
| `{{cluster_style}}` | pie-chart | sized-circle / pie-chart / text-only |
| `{{cluster_radius}}` | 60 | Pixel radius for clustering |
| `{{popup_fields}}` | ["name","address","rating"] | Fields in click popup |

#### Expected Output

Complete JavaScript with source (including clusterProperties), three layer definitions, event handlers, optional pie chart rendering code, and performance notes.

#### Validation Checklist

- [ ] Source has `cluster: true` and `clusterRadius` set
- [ ] clusterProperties expressions are syntactically valid
- [ ] Cluster click handler uses `getClusterExpansionZoom` (not arbitrary zoom increment)
- [ ] Unclustered points have a filter: `["!", ["has", "point_count"]]`
- [ ] Performance: tested with stated feature count without frame drops

#### Cost Optimization

- Pie chart clusters are complex (~80 lines). If sized-circle suffices, use that.
- Pre-aggregate categories in the data pipeline instead of using clusterProperties for large datasets.

#### Related Pages

- [Performance Optimization (clustering, WebGL budget)](../js-bindbox/performance-optimization.md)
- [Data Formats and Loading (GeoJSON optimization)](../js-bindbox/data-formats-loading.md)

#### Extensibility

- Use category colors from P1 (Scientific Color Palette) or P2 (Brand Theme).
- Apply icons from P19 (SVG Icon System) to individual markers.
- Combine with P9 (Animated Styles) for pulsing alert markers.

---

## 7. Chart and Dashboard Integration

---

### Prompt 21 -- Synchronized Map + Chart Theme

#### Scenario (actual use / 实际场景)

Your application has both a map (MapLibre GL) and charts (ECharts, D3, Chart.js, or Vega-Lite). The map uses blue-to-red for temperature; the chart uses a different blue-to-red. This visual inconsistency confuses users. You need a unified color and typography system across map and chart components.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Full-stack developer building a geospatial dashboard |
| **AI** | Data visualization design system architect |

#### Prompt Template

```text
You are a data visualization design system architect. Create a unified theme
that synchronizes colors, typography, and interaction states between a MapLibre
GL JS map and {{chart_library}} charts.

CHART LIBRARY: {{chart_library}}         # ECharts | D3.js | Chart.js | Vega-Lite
MAP VARIABLES: {{map_variables}}         # variables shown on the map
CHART VARIABLES: {{chart_variables}}     # variables shown in charts
SHARED VARIABLES: {{shared_variables}}   # variables appearing in BOTH map and charts

THEME REQUIREMENTS:

1. COLOR TOKEN SYSTEM:
   Define a single source of truth:
   ```javascript
   const theme = {
     // Sequential palette (shared between map choropleth and chart bars)
     sequential: ['#F7FBFF','#DEEBF7','#C6DBEF','#9ECAE1','#6BAED6',
                   '#4292C6','#2171B5','#084594'],
     // Diverging palette (shared between map and chart)
     diverging: ['#B2182B','#D6604D','#F4A582','#FDDBC7',
                 '#D1E5F0','#92C5DE','#4393C3','#2166AC'],
     // Categorical palette (for different data series)
     categorical: ['#1F77B4','#FF7F0E','#2CA02C','#D62728',
                    '#9467BD','#8C564B','#E377C2','#7F7F7F'],
     // Backgrounds
     bg: { light: '#FFFFFF', dark: '#0D1117' },
     surface: { light: '#F6F8FA', dark: '#161B22' },
     text: { light: '#24292F', dark: '#C9D1D9' },
     grid: { light: '#E1E4E8', dark: '#30363D' },
     // Interaction
     hover: '#FFD33D',
     selection: '#58A6FF',
   };
   ```

2. MAP APPLICATION:
   - MapLibre paint properties using theme.sequential for choropleth:
     ["interpolate", ["linear"], ["get", "value"],
       min, theme.sequential[0], ..., max, theme.sequential[7]]
   - Highlight on hover: theme.hover
   - Selected feature outline: theme.selection

3. CHART APPLICATION ({{chart_library}}):
   {{#if echarts}}
   - ECharts theme registration:
     echarts.registerTheme('geo-dashboard', {
       color: theme.categorical,
       backgroundColor: theme.bg.light,
       textStyle: { color: theme.text.light },
       // ... complete theme object
     });
   - Bar/line charts use theme.sequential for single-variable
   - Scatter plots use theme.categorical for multi-series
   {{/if}}
   {{#if d3}}
   - D3 scale: d3.scaleOrdinal(theme.categorical)
   - D3 sequential: d3.scaleSequential(d3.interpolateRgbBasis(theme.sequential))
   - Axis styling: .tick line { stroke: theme.grid.light }
   {{/if}}

4. CROSS-HIGHLIGHT:
   - When user hovers a map feature, highlight the corresponding chart element
   - When user hovers a chart bar/point, highlight the corresponding map feature
   - Both use the same theme.hover color and transition duration (200ms)

5. DARK MODE TOGGLE:
   - Single function to switch all components:
     function setDarkMode(isDark) {
       // Update MapLibre style
       // Update chart theme
       // Update CSS variables
     }

Return:
A. Theme token object (JavaScript)
B. CSS custom properties (light + dark)
C. MapLibre style snippet using theme tokens
D. {{chart_library}} theme configuration
E. Cross-highlight event handler code
F. Dark mode toggle function

DARK ARTS TIP: The most maintainable approach is CSS custom properties as the
single source of truth. MapLibre cannot read CSS variables directly, but you can
read them in JavaScript:
  getComputedStyle(document.documentElement).getPropertyValue('--color-primary')
and pass them to map.setPaintProperty(). This way, changing a CSS variable
updates both the chart (which reads CSS natively) and the map (via the JS bridge).
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{chart_library}}` | ECharts | ECharts / D3.js / Chart.js / Vega-Lite |
| `{{shared_variables}}` | ["temperature","population"] | Variables shown in both map and charts |

#### Expected Output

A theme token object, CSS custom properties, MapLibre paint snippet, chart theme configuration, cross-highlight handlers, and dark mode toggle function.

#### Validation Checklist

- [ ] Same hex values used in map and chart for the same variable
- [ ] Hover color is identical in both components
- [ ] Dark mode toggle updates all three systems (CSS, map, chart)
- [ ] Categorical palette has enough colors for the number of data series
- [ ] Grid lines and axis colors use the theme's grid token

#### Cost Optimization

- Specify the chart library upfront to get targeted theme code.
- If you do not need cross-highlighting, omit that section.

#### Related Pages

- [Charting Integration (ECharts, D3, Vega with maps)](../js-bindbox/charting-integration.md)
- [Framework Integration (state management for sync)](../js-bindbox/framework-integration.md)

#### Extensibility

- Use the theme tokens in P16 (Responsive Legend) for consistent legend styling.
- Apply to P22 (Dashboard Layout) for the complete dashboard design system.
- Feed the sequential palette into P1 (Scientific Color Palette) for validation.

---

### Prompt 22 -- Dashboard Layout with Map

#### Scenario (actual use / 实际场景)

You are building a full geospatial dashboard: a large map panel, sidebar with filters, multiple chart panels, a data table, and a header. The layout must be responsive, performant, and follow dashboard UX conventions. This prompt generates the complete layout structure.

#### Roles

| Role | Description |
|------|-------------|
| **User** | Full-stack developer building a geospatial dashboard |
| **AI** | Dashboard UX architect and responsive layout engineer |

#### Prompt Template

```text
You are a dashboard UX architect. Design a responsive geospatial dashboard
layout with the following panels:

FRAMEWORK: {{framework}}                 # Vanilla | React | Vue 3 | Svelte
CSS APPROACH: {{css_approach}}           # CSS Grid | Tailwind CSS | Bootstrap 5

PANELS:
1. HEADER: app title, user menu, dark mode toggle
   Height: {{header_height}}px fixed

2. SIDEBAR (left):
   Width: {{sidebar_width}}px on desktop, collapsible on mobile
   Contents:
   - Layer toggle checkboxes
   - Filter controls: {{filters}}
     # e.g., date range picker, category multi-select, numeric slider
   - "Apply" / "Reset" buttons

3. MAP (main):
   Takes remaining space (flex: 1 or grid fr)
   MapLibre GL JS instance
   Overlay controls: zoom, geolocate, legend (from P16)

4. CHART PANEL (right or bottom):
   {{chart_panel_position}}              # right sidebar | bottom panel | both
   Charts: {{charts}}
   # e.g., ["bar chart of top categories", "time series line chart", "summary KPIs"]
   Resizable: {{resizable}}             # true | false (drag handle)

5. DATA TABLE (bottom):
   Collapsible panel at page bottom
   Shows selected/filtered features
   Columns: {{table_columns}}
   Sortable, paginated

RESPONSIVE BREAKPOINTS:
  Desktop (>= {{desktop_bp}}px):
    Full layout: sidebar | map + charts | optional right panel
  Tablet ({{tablet_bp}}--{{desktop_bp}}px):
    Sidebar collapses to hamburger menu
    Charts move below map (vertical stack)
  Mobile (< {{tablet_bp}}px):
    Map fills viewport
    Sidebar as bottom sheet
    Charts in horizontal scroll carousel
    Table hidden (show "View data" button)

LAYOUT CSS:
```css
.dashboard {
  display: grid;
  grid-template-areas:
    "header  header  header"
    "sidebar map     charts"
    "sidebar table   table";
  grid-template-columns: {{sidebar_width}}px 1fr {{chart_panel_width}}px;
  grid-template-rows: {{header_height}}px 1fr {{table_height}}px;
  height: 100vh;
  overflow: hidden;
}

@media (max-width: {{tablet_bp}}px) {
  .dashboard {
    grid-template-areas:
      "header"
      "map"
      "charts"
      "table";
    grid-template-columns: 1fr;
    grid-template-rows: {{header_height}}px 1fr auto auto;
  }
}
```

INTERACTION PATTERNS:
  - Filter change: update map filter + re-query charts
  - Map click: select feature, scroll table to row, highlight in charts
  - Chart click: zoom map to features in that category
  - Table row click: fly to feature on map
  - All interactions use a shared state store:
    {{#if react}} React Context or Zustand {{/if}}
    {{#if vue}} Pinia store {{/if}}
    {{#if vanilla}} Custom EventEmitter or pub/sub {{/if}}

PERFORMANCE:
  - Map resizes on panel resize: map.resize() on ResizeObserver
  - Charts lazy-load when panel is opened
  - Debounce filter changes (300ms) before querying
  - Virtual scroll for table with > 1000 rows

Return:
A. HTML structure (semantic, accessible)
B. CSS (Grid layout with responsive media queries)
C. JavaScript/Framework component skeleton:
   - App shell with state management
   - MapLibre initialization with resize handling
   - Chart initialization (placeholder for P21 theme)
   - Filter-to-map binding
   - Cross-component event flow diagram (text-based)
D. Accessibility: skip navigation, focus management, ARIA landmarks

DARK ARTS TIP: The number one dashboard performance killer is re-rendering
charts on every map move event. Debounce map 'moveend' with a 300ms delay
before updating charts. Also, initialize MapLibre AFTER the CSS Grid has
rendered (use requestAnimationFrame or a ResizeObserver) -- if the map
container has zero dimensions at init time, MapLibre silently renders nothing
and never recovers until you call map.resize().
```

#### Variables to Customize

| Variable | Example | Notes |
|----------|---------|-------|
| `{{framework}}` | React | Vanilla / React / Vue 3 / Svelte |
| `{{css_approach}}` | CSS Grid | CSS Grid / Tailwind / Bootstrap |
| `{{sidebar_width}}` | 280 | px on desktop |
| `{{chart_panel_position}}` | right sidebar | right / bottom / both |
| `{{desktop_bp}}` | 1200 | Desktop breakpoint in px |
| `{{tablet_bp}}` | 768 | Tablet breakpoint in px |

#### Expected Output

HTML structure, CSS Grid layout with responsive media queries, JavaScript/framework component skeleton with state management, event flow diagram, and accessibility markup.

#### Validation Checklist

- [ ] Map container has `width: 100%; height: 100%` within its grid area
- [ ] `map.resize()` is called on panel resize events
- [ ] Sidebar collapse does not cause map to lose its center position
- [ ] All interactive elements are keyboard accessible
- [ ] Mobile layout is usable (map is not smaller than 300x400px)
- [ ] Filter debounce prevents rapid re-rendering

#### Cost Optimization

- Specify the framework to avoid getting code for all four options.
- If you do not need the data table, omit panel 5 to reduce complexity significantly.
- Provide the chart library name so it matches P21 (Synchronized Theme) output.

#### Related Pages

- [Framework Integration (React/Vue/Svelte with MapLibre)](../js-bindbox/framework-integration.md)
- [Charting Integration (ECharts/D3 in dashboards)](../js-bindbox/charting-integration.md)
- [Performance Optimization (dashboard rendering)](../js-bindbox/performance-optimization.md)

#### Extensibility

- Apply P21 (Synchronized Map + Chart Theme) for consistent colors.
- Place P16 (Responsive Legend) in the map overlay area.
- Use P6 (Data-Driven Styling) for the map layers driven by filters.
- Connect filter controls to P4 (Thematic Map Classification) for dynamic reclassification.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
