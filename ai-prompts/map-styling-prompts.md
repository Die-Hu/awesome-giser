# Map Styling Prompts

> Prompt templates for color scheme generation, Mapbox/MapLibre style JSON, QGIS SLD/QML styles, legend design, and cartographic labeling.

> **Quick Picks**
> - 🏆 **Most Useful**: [Colorblind-Safe Palette](#prompt-1--colorblind-safe-palette) — accessible maps should be the default, not an afterthought
> - 🚀 **Time Saver**: [Dark Mode Basemap](#prompt-3--dark-mode-basemap) — production-ready dark style JSON in one prompt
> - 🆕 **Cutting Edge**: [Accessibility-Compliant Color Scheme](#prompt-4--accessibility-compliant-color-scheme) — WCAG 2.1 AA compliant maps with automated contrast checking

---

## Table of Contents

- [Color Scheme Generation](#color-scheme-generation)
- [MapLibre / Mapbox Style JSON](#maplibre--mapbox-style-json)
- [QGIS SLD / QML Styles](#qgis-sld--qml-styles)
- [Legend Design](#legend-design)
- [Label Placement](#label-placement)
- [SVG Map Icons](#svg-map-icons)

---

## Color Scheme Generation

### Prompt 1 — Colorblind-Safe Palette

**Context:** You need a color palette for a thematic map that is accessible to readers with color vision deficiencies.

**Template:**

```
Generate a [diverging / sequential / qualitative] color palette for a [map type, e.g., choropleth]
showing [variable, e.g., population density change from -20% to +40%].

Requirements:
- [N] classes/stops
- Colorblind-safe (deuteranopia and protanopia friendly)
- Print-friendly (distinguishable in grayscale)
- Perceptually uniform lightness progression
- Return as: hex codes, RGB tuples, and a CSS/JSON-ready array
- Suggest a neutral/midpoint color for [diverging: the zero-change class]

Also suggest a background color and label color that provide sufficient contrast (WCAG AA).
```

**Variables to customize:**
- Palette type (diverging, sequential, qualitative)
- Number of classes
- Variable description and data range
- Accessibility requirements

**Expected output format:** A list of hex codes with RGB values, a JSON array, and contrast recommendations.

---

### Prompt 2 — Thematic Palette from Reference

**Context:** You want a color scheme that matches a design reference (branding, existing publication, or aesthetic theme).

**Template:**

```
I am creating a map series for [context, e.g., a municipal government report / an environmental NGO].
The brand colors are: [list primary and secondary hex codes].

Generate a [sequential / diverging] color palette with [N] classes that:
- Incorporates the brand colors as anchors
- Maintains perceptual uniformity between steps
- Works on both light and dark basemaps
- Is distinguishable in print

Return the palette as:
1. Hex codes
2. Mapbox GL expression: ["interpolate", ["linear"], ["get", "value"], ...]
3. QGIS graduated renderer XML snippet
```

**Variables to customize:**
- Brand colors
- Palette type and number of classes
- Basemap background
- Output formats needed

**Expected output format:** Multi-format color definitions ready for copy-paste.

---

## MapLibre / Mapbox Style JSON

### Prompt 1 — Custom Basemap Style

**Context:** You want to create a custom basemap style for a web map application.

**Template:**

```
Create a Mapbox GL / MapLibre style JSON for a [theme, e.g., dark mode / minimal / topographic] basemap.

Source: [vector tile source, e.g., OpenMapTiles / Mapbox / Protomaps]
Target use: [e.g., data visualization overlay — needs to be subtle and not compete with data layers]

Layer styling requirements:
- Water: [color]
- Land/background: [color]
- Roads: [hierarchical styling — highway, major, minor, path]
- Buildings: [3D extrusion / flat fill / hide]
- Labels: [language preference, font, halo]
- Boundaries: [admin level styling]
- Terrain: [hillshade yes/no]

Return the complete style JSON with:
- "sources" block for the tile source
- "layers" array with all styled layers
- "glyphs" and "sprite" URLs
- Appropriate zoom-level transitions (minzoom/maxzoom)
```

**Variables to customize:**
- Theme and color mood
- Tile source
- Layer visibility and styling preferences
- Label language

**Expected output format:** A complete Mapbox/MapLibre style JSON file.

---

### Prompt 2 — Data-Driven Layer Styling

**Context:** You want to style a GeoJSON data layer with data-driven expressions in Mapbox/MapLibre.

**Template:**

```
Write a MapLibre GL JS layer definition for a [geometry type: point / line / polygon] layer
displaying [data description, e.g., earthquake events].

The GeoJSON properties include:
- [property A]: [type, range, e.g., "magnitude": number, 1.0 - 9.0]
- [property B]: [type, categories, e.g., "type": string, one of "earthquake", "explosion", "quarry blast"]
- [property C]: [type, e.g., "depth_km": number, 0 - 700]

Styling rules:
- [Size / width / radius] driven by [property A]: [describe mapping]
- [Color] driven by [property B]: [categorical mapping] or [property C]: [gradient]
- [Opacity] driven by [property, e.g., age — more recent = more opaque]
- Interactive: highlight on hover, popup on click showing [fields]

Return the `map.addLayer()` call with Mapbox GL expressions and a matching `map.on('click', ...)`
event handler.
```

**Variables to customize:**
- Geometry type
- Data properties and value ranges
- Visual encodings (size, color, opacity)
- Interaction behavior

**Expected output format:** JavaScript code with `addLayer()` and event handlers.

---

### Prompt 3 — Dark Mode Basemap

**Context:** You need a dark basemap style for data visualization overlays, dashboards, or presentations where the data layers must stand out against a muted background.

**Template:**

```
Create a complete MapLibre GL style JSON for a dark mode basemap optimized for data visualization overlays.

Tile source: [OpenMapTiles / Protomaps PMTiles / Mapbox Vector Tiles]
Source URL: [tile URL or PMTiles URL]

Design requirements:
- Background: #0d1117 (GitHub dark) or [custom dark color]
- Water: #0a1628 with subtle wave pattern or flat fill
- Land: #161b22 or [custom]
- Roads hierarchy (all desaturated, low contrast):
  - Motorway: #30363d, width: zoom-interpolated 1px at z5 to 4px at z14
  - Primary: #21262d, width: 0.5px at z8 to 3px at z14
  - Secondary: #21262d, width: 0.5px at z10 to 2px at z16
  - Minor roads: visible only at z13+, #1c2128
- Buildings: #1c2128 at z14+, no 3D extrusion (flat, subtle)
- Boundaries:
  - Country: dashed #30363d, width 1px
  - State/province: dotted #21262d, width 0.5px, visible z4+
- Labels:
  - Primary: #8b949e, font: ["Noto Sans Regular"]
  - City labels: scaled by population rank, z5+ for capitals, z8+ for major cities
  - Road labels: #484f58, z14+
  - Text halo: #0d1117, width 1px
- Terrain hillshade: very subtle, opacity 0.08, highlight #1c2128, shadow #010409
- POIs: hidden by default (this is a data viz basemap, not a reference map)

Return the complete style JSON with:
- "version": 8
- "name": "Dark Mode Data Viz"
- "sources": { ... } block with the tile source
- "glyphs": "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf" (or [custom])
- "layers": [ ... ] complete array with all layers in correct order
- Proper minzoom/maxzoom for each layer
- zoom-interpolate expressions for road widths and label sizes

The style should be a valid, self-contained JSON file that works when loaded directly:
  const map = new maplibregl.Map({ container: 'map', style: 'dark-mode.json' });
```

**Variables to customize:**
- Tile source and URL
- Color palette (GitHub Dark, Dracula, custom)
- Label language
- Feature visibility preferences

**Expected output format:** A complete, valid MapLibre style JSON file (50-200 lines depending on complexity).

---

### Prompt 4 — Accessibility-Compliant Color Scheme

**Context:** You need a map color scheme that meets WCAG 2.1 AA accessibility standards, is usable for people with all types of color vision deficiencies, and works in both screen and print contexts.

**Template:**

```
Design an accessibility-compliant color scheme for a [map type, e.g., choropleth / categorical / bivariate]
map showing [variable(s), e.g., "population density across census tracts"].

Requirements:
- Number of classes: [N]
- Palette type: [sequential / diverging / qualitative]
- Must pass ALL of these checks:
  1. WCAG 2.1 AA contrast ratio >= 4.5:1 between text labels and fill colors
  2. Distinguishable under deuteranopia (red-green, ~8% of males)
  3. Distinguishable under protanopia (red deficiency)
  4. Distinguishable under tritanopia (blue-yellow, rare but should still work)
  5. Distinguishable in grayscale print (monotonically increasing lightness)
  6. Minimum delta-E (CIEDE2000) of 20 between adjacent classes

For each color in the palette, return:
- Hex code
- RGB values
- HSL values
- CIELAB L* value (for verifying lightness progression)
- Simulated appearance under deuteranopia and protanopia (use Brettel/Vienot simulation)

Also provide:
- A recommended background color with its contrast ratios against each class
- A recommended text/label color for each class fill (black or white, whichever has higher contrast)
- The palette formatted as:
  1. CSS custom properties: --class-1: #hex; --class-2: #hex; ...
  2. Mapbox GL expression: ["interpolate", ["linear"], ["get", "value"], stop1, color1, ...]
  3. QGIS graduated renderer color ramp (comma-separated hex list)
  4. Python matplotlib colormap: ListedColormap([...])
  5. D3.js scale: d3.scaleThreshold().domain([...]).range([...])

If the map type is bivariate (two variables), design a 3x3 or 4x4 matrix palette where both
axes maintain accessibility. Provide the matrix as a 2D hex array.
```

**Variables to customize:**
- Map type and variables
- Number of classes
- Palette type
- Specific accessibility standards to meet
- Output format preferences

**Expected output format:** Detailed color specification with hex/RGB/HSL/LAB values, accessibility validation, and multi-format output.

---

## QGIS SLD / QML Styles

### Prompt 1 — SLD for Graduated Symbology

**Context:** You want to style a WMS/WFS layer using SLD (Styled Layer Descriptor) for sharing via GeoServer or QGIS Server.

**Template:**

```
Write an SLD (Styled Layer Descriptor) XML file for a [geometry type] layer with graduated symbology.

Layer name: [name]
Attribute for classification: [column name]
Classification method: [equal interval / quantile / natural breaks]
Number of classes: [N]
Value range: [min] to [max]

Color ramp: [e.g., YlOrRd / Blues / custom hex list]

Requirements:
- Include a human-readable title and abstract
- Each rule should have a descriptive name (e.g., "Low: 0 - 100")
- Add a stroke (outline) of [color] with width [X]
- Set opacity to [value]
- For polygons, include a PolygonSymbolizer; for points, use a PointSymbolizer with circle marks
- Valid against SLD 1.0 schema

The file should be directly importable into GeoServer or QGIS.
```

**Variables to customize:**
- Geometry type and layer name
- Classification attribute, method, and class count
- Color ramp
- Stroke and opacity settings

**Expected output format:** Complete SLD XML file.

---

### Prompt 2 — QML Style for Categorized Symbology

**Context:** You want to save and share a QGIS categorized style as a QML file.

**Template:**

```
Write a QGIS QML style file for a [geometry type] layer with categorized symbology.

Attribute: [column name]
Categories and colors:
- "[value 1]": [hex color], label: "[display label]"
- "[value 2]": [hex color], label: "[display label]"
- "[value 3]": [hex color], label: "[display label]"
[... repeat for all categories]

Additional styling:
- Stroke color: [hex], width: [mm]
- Fill opacity: [0-100]%
- Default symbol for unmatched values: [color]
- Label field: [column] with font [name] at size [pt], with halo

The QML should include the renderer, labeling, and symbol definitions.
```

**Variables to customize:**
- Categories, colors, and labels
- Stroke and opacity
- Labeling configuration

**Expected output format:** Complete QML XML file importable into QGIS.

---

## Legend Design

### Prompt 1 — Legend Layout Recommendations

**Context:** You need guidance on creating a clear and effective map legend for a specific map type.

**Template:**

```
I am designing a legend for a [map type, e.g., bivariate choropleth / proportional symbol / dot density] map
showing [variables] for [audience, e.g., city planners / general public / academic journal].

Map details:
- Number of classes/categories: [N]
- Symbology type: [graduated color / categorized / proportional circles / hatching]
- Map size: [A4 print / web 1200x800px / poster A0]
- Output: [print / screen / both]

Provide recommendations for:
1. Legend title wording (concise, describes the variable and unit)
2. Class label formatting (number format, rounding, ranges vs break points)
3. Symbol ordering (high-to-low vs low-to-high, and why)
4. Legend position on the map
5. Font sizes for title, labels, and source text
6. Whether to use a continuous gradient or discrete swatches
7. How to handle "No Data" / "Not Applicable" categories

If it is a bivariate map, suggest a 3x3 or 4x4 matrix legend layout.
```

**Variables to customize:**
- Map type, variables, audience
- Number of classes
- Output medium and size

**Expected output format:** Textual recommendations with rationale, optionally with a text-based legend mockup.

---

### Prompt 2 — Generate Legend as SVG

**Context:** You want to produce a standalone legend graphic for embedding in a report or web page.

**Template:**

```
Generate an SVG legend graphic for the following map symbology:

Type: [graduated / categorized / proportional symbol]
Title: "[legend title]"
Classes:
- [label]: [hex color] [optional: size/pattern]
- [label]: [hex color]
- [... repeat]

Requirements:
- SVG width: [X]px, scale to fit
- Font: [sans-serif / specific font]
- Include a "No Data" entry in gray
- Horizontal / Vertical orientation
- Add a subtle border and background

Return raw SVG code that I can embed in an HTML page or import into Inkscape.
```

**Variables to customize:**
- Symbology type and class definitions
- Dimensions and orientation
- Font and styling

**Expected output format:** Raw SVG markup.

---

## Label Placement

### Prompt 1 — Cartographic Labeling Rules

**Context:** You are labeling features on a map and want to follow best practices for readability and aesthetics.

**Template:**

```
I am labeling [feature type, e.g., cities / rivers / regions] on a [scale, e.g., 1:250,000] map
in [software: QGIS / ArcGIS Pro / Mapbox GL].

Feature details:
- Geometry type: [point / line / polygon]
- Label field: [column name]
- Number of features to label: approximately [N]
- Language: [e.g., English / Chinese / mixed bilingual]
- Map purpose: [reference / thematic / navigation]

Provide:
1. Recommended font family and size (in mm for print, px for web)
2. Text halo/buffer settings (color, size)
3. Placement rules (e.g., points: prefer upper-right; lines: follow curvature; polygons: horizontal centered)
4. Priority rules for label conflicts (which features get labeled first)
5. Scale-dependent visibility (which labels show at which zoom levels)
6. Abbreviation rules for long names
7. The specific configuration in [software] — either as menu steps or as a code/JSON snippet

For bilingual labels, recommend stacking order (primary language on top or bottom)
and font size ratio between languages.
```

**Variables to customize:**
- Feature type and geometry
- Software
- Language and bilingual requirements
- Map scale and purpose

**Expected output format:** Labeling rules plus software-specific configuration (QGIS labeling properties or Mapbox layout/paint JSON).

---

### Prompt 2 — Mapbox GL Label Expression

**Context:** You need a Mapbox/MapLibre text layer with dynamic label formatting based on zoom level and feature properties.

**Template:**

```
Write a MapLibre GL JS symbol layer definition for labeling [features, e.g., cities].

Data properties:
- "name": city name (string)
- "name_local": local language name (string)
- "population": number
- "capital": boolean

Labeling rules:
- At zoom < 5: show only capitals, label = name
- At zoom 5-8: show cities with population > [threshold], label = name
- At zoom > 8: show all cities, label = "name\nname_local" (stacked)
- Font size: scaled by population ([min]px to [max]px)
- Color: [hex] with white halo
- Collision priority: capitals first, then by population descending
- Anchor: top, offset [0, 0.5em] below the icon

Return the complete `map.addLayer()` call with Mapbox GL expressions.
```

**Variables to customize:**
- Feature type and properties
- Zoom-dependent rules
- Font scaling range
- Priority rules

**Expected output format:** JavaScript `map.addLayer()` call with all layout and paint properties.

---

## SVG Map Icons

### Prompt 1 — Generate SVG Map Icons

**Context:** You need custom map marker icons or symbol icons for a web map, print map, or legend. AI can generate clean SVG code that you can use directly in MapLibre sprite sheets, Leaflet markers, or QGIS SVG markers.

**Template:**

```
Generate SVG map icons for [use case, e.g., "a city services web map" / "a natural hazard warning system" /
"a tourism map of national parks"].

Icons needed (one SVG per icon):
1. [Icon name]: [description, e.g., "Hospital — red cross on white circle with subtle shadow"]
2. [Icon name]: [description, e.g., "Fire station — flame icon in red, minimal style"]
3. [Icon name]: [description, e.g., "School — book/graduation cap, blue"]
4. [Icon name]: [description, e.g., "Park — tree silhouette, green"]
5. [Icon name]: [description, e.g., "Parking — P letter in blue circle"]
[... add more as needed]

Design specifications:
- Viewbox: 0 0 [size] [size] (e.g., 24x24 for web, 32x32 for retina, 64x64 for print)
- Style: [flat / outlined / filled / Material Design-like / hand-drawn]
- Stroke width: [value]px (if outlined style)
- Color palette: [list primary colors, or "monochrome" / "each icon uses its own semantic color"]
- Background: [none (transparent) / circle / rounded square / pin/teardrop shape]
- Must be pixel-aligned for crisp rendering at small sizes
- No external dependencies (no <use> referencing external files)
- Each SVG should be self-contained with no embedded fonts (use paths instead)

For MapLibre/Mapbox sprite sheet compatibility:
- Also provide a sprite sheet JSON index with icon names, positions, width, height, pixelRatio
- Combine all icons into a single SVG sprite sheet as well

For QGIS SVG marker compatibility:
- Use param(fill) and param(outline) placeholders so QGIS can override colors:
  fill="param(fill) [default hex]" stroke="param(outline) [default hex]"

Return each icon as:
1. Standalone SVG code block
2. Base64 data URI (for inline use in JavaScript: `new Image().src = 'data:image/svg+xml;base64,...'`)
3. Optimized SVG (run through SVGO rules: remove metadata, collapse groups, simplify paths)
```

**Variables to customize:**
- Icon list with descriptions
- Size and style
- Color palette
- Target platform (web/MapLibre, QGIS, print)
- Sprite sheet format needed

**Expected output format:** Individual SVG code blocks per icon, optional sprite sheet, QGIS-compatible versions.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
