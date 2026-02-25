# Cartography & Design

> Map design principles, typography, color theory, accessibility, and styling tools for creating clear, accessible, and beautiful maps -- a state-of-the-art reference for GIS professionals and cartographers.

> **Quick Picks**
> - **SOTA**: [Maputnik](https://maputnik.github.io) -- open-source visual style editor for MapLibre/Mapbox GL styles
> - **Free Best**: [QGIS](https://qgis.org) -- full cartographic control with print layout composer
> - **Fastest Setup**: [Mapbox Studio](https://studio.mapbox.com) -- visual style editor with instant preview

---

## 1. Map Design Principles

Cartographic design is the discipline of translating geospatial data into visual representations that communicate clearly, accurately, and efficiently. Every design decision -- from color to typeface to line weight -- must serve the map's communicative purpose.

### 1.1 Visual Hierarchy

Visual hierarchy controls the order in which a reader perceives information. Without it, a map becomes a flat mosaic of undifferentiated noise.

**Establishing hierarchy through visual weight:**

1. **Figure-ground separation.** The primary subject (figure) must stand apart from its context (ground). A thematic polygon layer, for example, should contrast strongly against the basemap. Techniques include increasing saturation of the figure, desaturating the ground, and adding subtle drop shadows or halos.
2. **Contrast and emphasis.** Size, color value, weight, and saturation are the primary levers. Larger labels and bolder strokes push elements forward; smaller, lighter, and desaturated elements recede.
3. **Visual balance.** Distribute visual weight across the composition so no single quadrant dominates unless the data demands it. An asymmetric layout can be balanced by offsetting a heavy legend against a dense data region.
4. **Selective omission.** Show only what serves the map's purpose. Background features that do not contribute to the reader's understanding should be removed or subdued.

### 1.2 Gestalt Principles Applied to Maps

The Gestalt laws of visual perception explain how humans group visual elements. Cartographers should leverage these principles deliberately:

| Gestalt Principle | Map Application | Example |
|-------------------|-----------------|---------|
| **Proximity** | Features near each other are perceived as grouped | Clustered point symbols imply spatial association |
| **Similarity** | Features with similar appearance are perceived as related | Same-colored polygons read as a single class |
| **Continuity** | The eye follows smooth paths | River labels placed along the curve of the watercourse |
| **Closure** | The mind completes incomplete shapes | Dashed administrative boundaries still read as closed regions |
| **Enclosure** | Bounded areas are perceived as groups | Country shading groups all features within the border |
| **Figure-ground** | The brain separates foreground from background | Thematic data stands out from a muted basemap |

### 1.3 Data-Ink Ratio (Tufte)

Edward Tufte's *data-ink ratio* principle states that the largest share of "ink" (or pixels) on a visualization should encode actual data. Non-data elements -- decorative borders, heavy gridlines, redundant legends -- should be minimized or eliminated.

**Applying data-ink ratio to maps:**

- Remove gratuitous neatlines and decorative borders unless they serve a compositional purpose.
- Reduce basemap detail to the minimum needed for spatial context.
- Avoid 3D chrome effects on legend symbols (bevels, drop shadows on swatches).
- Prefer direct labeling over legend lookup when the number of classes is small (fewer than five).
- Use data-driven styling so that the visual encoding itself carries the information, rather than relying on ancillary text.

### 1.4 Bertin's Visual Variables

Jacques Bertin's *Semiology of Graphics* (1967) established a taxonomy of visual variables that remains foundational. Each variable has different suitability for encoding different data types:

| Visual Variable | Quantitative | Ordinal | Nominal | Example on Map |
|-----------------|:---:|:---:|:---:|----------------|
| **Position** | ++ | ++ | ++ | Feature location on the map plane |
| **Size** | ++ | + | -- | Graduated circles for population |
| **Color value** (lightness) | ++ | ++ | -- | Sequential choropleth (light = low, dark = high) |
| **Color hue** | -- | -- | ++ | Qualitative land-use categories |
| **Orientation** | -- | -- | + | Wind direction arrows |
| **Shape** | -- | -- | ++ | Different point symbols per category |
| **Texture/Pattern** | -- | + | ++ | Hatching for disputed territory |

`++` = highly effective, `+` = usable, `--` = poor fit.

**Modern additions to Bertin's framework:**

- **Color saturation** -- useful for emphasizing uncertainty or confidence intervals.
- **Transparency/opacity** -- critical for overlapping layers in web maps.
- **Animation/motion** -- temporal encoding (see [Temporal Animation](temporal-animation.md)).

### 1.5 Cartographic Generalization

Generalization is the controlled reduction of detail to make a map legible at a given scale. It is one of the most difficult problems in automated cartography.

| Generalization Operator | Description | Example |
|------------------------|-------------|---------|
| **Simplification** | Reduce vertices in a line/polygon while preserving shape | Douglas-Peucker on coastlines |
| **Selection** | Choose which features to include based on importance | Show only primary roads at 1:1M |
| **Displacement** | Shift features apart to avoid overlap | Separate parallel roads |
| **Exaggeration** | Enlarge features that would otherwise be invisible | Widen narrow rivers at small scales |
| **Aggregation** | Merge individual features into groups | Combine building footprints into urban blocks |
| **Smoothing** | Reduce angular noise in lines | Bezier smoothing on contour lines |
| **Typification** | Replace a dense pattern with a representative subset | Reduce a forest of trees to a representative few |
| **Collapse** | Reduce dimensionality (area to point) | A city polygon becomes a point at small scale |

**Tools for generalization:**

- **Mapshaper** ([mapshaper.org](https://mapshaper.org)) -- browser-based, supports Visvalingam and Douglas-Peucker with slider control.
- **PostGIS** -- `ST_Simplify`, `ST_SimplifyPreserveTopology`, `ST_SimplifyVW`.
- **Tippecanoe** -- automated generalization during vector tile creation (drop-densest-as-needed, coalesce).
- **QGIS** -- Sketcher, simplify geometries, variable-based rendering per zoom.

### 1.6 Design Resources

| Resource | URL | Description |
|----------|-----|-------------|
| **Cartography Guide** (Axis Maps) | [axismaps.github.io/thematic-cartography](https://axismaps.github.io/thematic-cartography/) | Comprehensive guide to thematic map design -- classification, color, labeling |
| Mapschool | [mapschool.io](https://mapschool.io) | Free introduction to geo concepts and cartography |
| TypeBrewer | [typebrewer.org](http://typebrewer.org) | Typography guidelines for cartographic labeling |
| Geospatial Design Patterns | [mapstyle.withgoogle.com](https://mapstyle.withgoogle.com) | Google's map styling reference |
| Cartographic Perspectives | [cartographicperspectives.org](https://cartographicperspectives.org) | Peer-reviewed journal on cartographic theory and practice |
| *Designing Better Maps* (Brewer) | ISBN 978-1589484405 | Essential textbook on cartographic design principles |
| *Cartography: Visualization of Geospatial Data* (Kraak & Ormeling) | ISBN 978-1138613966 | Academic standard reference |

---

## 2. Typography for Maps

Typography on maps is not decoration -- it is a core information layer. Label placement, font selection, and text styling communicate feature types, hierarchy, and spatial relationships.

### 2.1 Font Selection Conventions

Cartographic tradition assigns font families to feature types, enabling readers to decode category from text style alone:

| Feature Type | Traditional Convention | Rationale |
|-------------|----------------------|-----------|
| **Water features** (rivers, lakes, oceans) | *Italic* (often serif or sans-serif) | Italics suggest flow and fluidity |
| **Cities and settlements** | Upright sans-serif, weight varies by population | Clean, modern, distinguishable from physical features |
| **Physical features** (mountains, deserts) | Upright serif or italic serif | Classic, scholarly tone |
| **Countries and states** | Uppercase, spaced, sans-serif | Large-area features need wide letter-spacing |
| **Coordinates and grid values** | Monospace or tabular figures | Consistent digit width for alignment |
| **Annotations and callouts** | Condensed sans-serif | Space-efficient, secondary importance |

### 2.2 Recommended Fonts for Cartography

| Font | Type | License | Best For | CJK Support |
|------|------|---------|----------|:-----------:|
| **Noto Sans** | Sans-serif | OFL | Multilingual map labels | Via Noto Sans CJK |
| **Noto Serif** | Serif | OFL | Water features, print maps | Via Noto Serif CJK |
| **Source Sans 3** | Sans-serif | OFL | Clean UI and web maps | No |
| **Open Sans** | Sans-serif | OFL | General purpose | No |
| **Inter** | Sans-serif | OFL | Modern UI, dashboards | No |
| **Lora** | Serif | OFL | Natural features, print | No |
| **Roboto** | Sans-serif | Apache 2.0 | Android/Material Design maps | No |
| **Fira Sans** | Sans-serif | OFL | Technical/scientific maps | No |
| **IBM Plex Sans** | Sans-serif | OFL | Data-driven dashboards | No |
| **Playfair Display** | Serif | OFL | Editorial/artistic maps | No |

### 2.3 CJK Typography Considerations

Maps targeting Chinese, Japanese, or Korean audiences require special typographic attention:

- **Font families.** The Noto CJK family (Noto Sans CJK SC/TC/JP/KR and Noto Serif CJK) provides comprehensive coverage. Source Han Sans/Serif (Adobe) are identical glyphs under a different name.
- **Character width.** CJK characters are full-width. Labels require more horizontal or vertical space than Latin equivalents. Budget approximately 1.5-2x the width of a Latin label for the same semantic content.
- **Vertical text.** Traditional CJK layouts support vertical text direction. Some map styles use vertical labels for narrow features (rivers, roads). MapLibre GL JS supports `text-writing-mode: ['vertical']` for this purpose.
- **Mixed scripts.** A single label may contain CJK characters, Latin letters, and Arabic numerals (e.g., "Beijing / 北京 / 39.9N"). Use fonts that harmonize across scripts; the Noto family is designed for this.
- **Glyph rendering.** CJK glyphs at small sizes may become illegible. Set a minimum text size of 12px (or 10pt for print) for CJK labels.
- **Regional variants.** The same Unicode code point may render differently in Simplified Chinese (SC), Traditional Chinese (TC), Japanese (JP), and Korean (KR). Ensure the correct variant is specified in the font stack.

```json
// MapLibre style.json -- CJK font stack with fallbacks
{
  "glyphs": "https://your-glyph-server.com/fonts/{fontstack}/{range}.pbf",
  "layers": [
    {
      "id": "place-labels-cjk",
      "type": "symbol",
      "layout": {
        "text-font": ["Noto Sans CJK SC Regular", "Noto Sans Regular"],
        "text-field": ["coalesce", ["get", "name:zh"], ["get", "name"]],
        "text-size": ["interpolate", ["linear"], ["zoom"], 8, 12, 14, 16],
        "text-max-width": 8,
        "text-writing-mode": ["horizontal", "vertical"]
      }
    }
  ]
}
```

### 2.4 Font Pairing Matrix

Effective font pairing creates contrast between heading/title and body/label text while maintaining visual harmony. The following matrix covers common map types:

| Map Type | Title Font | Label Font | Annotation Font | Notes |
|----------|-----------|------------|-----------------|-------|
| Modern web map | **Inter Bold** | Noto Sans Regular | Inter Light | Clean geometric pairing |
| Classic editorial | **Playfair Display** | Source Sans 3 | Source Sans 3 Light Italic | High contrast serif/sans |
| Technical/scientific | **Source Sans 3 Bold** | Source Sans 3 Regular | Fira Mono | Minimal, data-focused |
| Multilingual | **Noto Sans Bold** | Noto Sans Regular | Noto Sans Light | Universal script coverage |
| Government/official | **Roboto Bold** | Roboto Regular | Roboto Condensed | Neutral, institutional |
| Historical/vintage | **EB Garamond Bold** | EB Garamond Regular | EB Garamond Italic | Period-appropriate serif |
| Outdoor/topographic | **Fira Sans Bold** | Fira Sans Regular | Fira Sans Condensed | Sturdy at small sizes |
| Dashboard/analytics | **IBM Plex Sans Bold** | IBM Plex Sans Regular | IBM Plex Mono | Data-oriented, tabular nums |
| Artistic/exhibition | **Cormorant Garamond Bold** | Open Sans Light | Open Sans Light Italic | Elegant display contrast |
| Minimalist | **Inter Tight Bold** | Inter Regular | Inter Light | Geometric, understated |
| CJK primary | **Noto Sans CJK Bold** | Noto Sans CJK Regular | Noto Sans CJK Light | Harmonized across scripts |

### 2.5 Label Placement Rules

Label placement is one of the hardest problems in automated cartography (NP-hard in the general case). These conventions guide both manual and algorithmic placement:

**Point features (cities, POIs):**
- Preferred positions in priority order: top-right, top-left, bottom-right, bottom-left, right, left.
- Avoid labels that cross other features or overlap other labels.
- Leader lines connect a label to its feature when displacement is necessary.

**Line features (rivers, roads):**
- Place labels along the line, following its curvature.
- Repeat labels at intervals for long features (every 200-400px on screen).
- River labels should be italic and follow the water flow direction.
- Road labels should be upright (auto-rotated to follow road direction).

**Area features (countries, lakes, regions):**
- Use letter-spacing (tracking) proportional to the area size.
- Curve labels to follow the shape of the area.
- Place inside the polygon when possible; use a leader line if the area is too small.
- Large area labels should be set at a lower visual weight to avoid dominating the map.

### 2.6 Halo and Buffer Strategies

Text halos (outlines/buffers) are essential for legibility when labels overlap variable backgrounds:

| Strategy | When to Use | Implementation |
|----------|-------------|----------------|
| **White halo** | Light basemaps | 1-2px white (#FFFFFF) halo at 70-90% opacity |
| **Dark halo** | Dark basemaps | 1-2px dark (#1a1a2e) halo at 70-90% opacity |
| **Blurred halo** | Complex backgrounds | Gaussian blur on the halo for softer blending |
| **Background panel** | Dense label areas | Semi-transparent rectangle behind label text |
| **Knockout/mask** | High-priority labels | Clip underlying features within the label bounds |

```javascript
// MapLibre GL JS -- label with halo
map.addLayer({
  id: 'city-labels',
  type: 'symbol',
  source: 'places',
  layout: {
    'text-field': ['get', 'name'],
    'text-font': ['Noto Sans Regular'],
    'text-size': ['interpolate', ['linear'], ['zoom'], 6, 10, 12, 16],
    'text-anchor': 'top',
    'text-offset': [0, 0.5]
  },
  paint: {
    'text-color': '#333333',
    'text-halo-color': 'rgba(255, 255, 255, 0.85)',
    'text-halo-width': 1.5,
    'text-halo-blur': 0.5
  }
});
```

### 2.7 Web Font Loading for MapLibre

MapLibre GL JS uses protocol buffer (PBF) encoded SDF (Signed Distance Field) glyph ranges for text rendering. This differs fundamentally from standard CSS web fonts.

**Glyph server setup:**

1. Convert TTF/OTF fonts to PBF glyph ranges using [font-maker](https://github.com/maplibre/font-maker) or [OpenMapTiles fonts](https://github.com/openmaptiles/fonts).
2. Host the generated PBF files on a static file server or CDN.
3. Reference the glyph endpoint in the style JSON:

```json
{
  "glyphs": "https://your-cdn.com/fonts/{fontstack}/{range}.pbf"
}
```

**Pre-built glyph servers:**

| Provider | URL Pattern | Coverage |
|----------|-------------|----------|
| MapTiler | `https://api.maptiler.com/fonts/{fontstack}/{range}.pbf?key=KEY` | 30+ fonts including Noto CJK |
| OpenMapTiles | `https://fonts.openmaptiles.org/{fontstack}/{range}.pbf` | Basic Latin fonts |
| Custom (Martin) | Self-hosted via Martin sprite/font endpoints | Any font you provide |

> **Cross-reference:** For MapLibre GL JS setup and configuration, see [2D Mapping Libraries](../js-bindbox/2d-mapping.md).

---

## 3. Color Theory & Perceptual Science

Color is the most powerful and most frequently misused visual variable in cartography. Effective map coloring requires understanding color science, perception, and cultural context.

### 3.1 Color Spaces for Cartography

Not all color spaces are equally suited for map design. The critical distinction is **perceptual uniformity** -- whether equal numeric steps in the color space produce equal perceived differences.

| Color Space | Perceptually Uniform | Use in Cartography | Notes |
|-------------|:---:|-----|-------|
| **sRGB** | No | Default web display | Non-uniform; avoid for interpolation |
| **HSL/HSV** | No | Quick prototyping | Intuitive but perceptually uneven |
| **CIELAB (L\*a\*b\*)** | Approximately | Academic cartography | Better than sRGB, but not perfect |
| **HCL (Hue-Chroma-Luminance)** | Approximately | ColorBrewer schemes | Intuitive axes, good for cartography |
| **Oklab / Oklch** | Yes | Modern web maps (CSS Color Level 4) | Best current choice for web |
| **CAM16-UCS** | Yes | Research, high-fidelity color design | Accounts for viewing conditions |

**Why perceptual uniformity matters:**

When you interpolate between two colors in sRGB to create a sequential color ramp, the middle steps may appear uneven -- some transitions look like large jumps while others are barely noticeable. This creates false patterns in choropleth maps. Interpolating in Oklab or HCL produces ramps where each step appears equally distinct, ensuring that data differences map faithfully to perceived color differences.

```javascript
// d3-interpolate: perceptually uniform interpolation
import { interpolateHcl, interpolateLab } from 'd3-interpolate';

// WRONG: sRGB interpolation (non-uniform perceived steps)
const srgbScale = d3.scaleLinear()
  .domain([0, 100])
  .range(['#ffffcc', '#006837']);

// BETTER: HCL interpolation (perceptually uniform)
const hclScale = d3.scaleLinear()
  .domain([0, 100])
  .range(['#ffffcc', '#006837'])
  .interpolate(interpolateHcl);

// BEST: Oklab interpolation (CSS Color Level 4)
// In modern CSS: color-mix(in oklch, #ffffcc 50%, #006837)
```

### 3.2 Color Scheme Types

| Scheme Type | Data Type | Example | Design Rule |
|-------------|-----------|---------|-------------|
| **Sequential** | Ordered numeric | Population density | Single hue, light-to-dark |
| **Diverging** | Numeric with midpoint | Temperature anomaly | Two hues diverging from neutral center |
| **Qualitative** | Categorical (nominal) | Land use classes | Maximum perceptual distance between hues |
| **Bivariate** | Two variables simultaneously | Income vs. education | 3x3 or 4x4 color grid |
| **Binary** | Presence/absence | Flood zone yes/no | Two distinct colors |

**Qualitative scheme limits:** The human eye can reliably distinguish 8-12 qualitative hues on a map. Beyond this, consider grouping categories or using a combination of hue and pattern.

### 3.3 Color Harmony

Color harmony principles from art and design apply to cartographic palettes:

- **Complementary** (opposite on the color wheel) -- strong contrast, good for diverging schemes. Use for two-class maps or figure-ground separation.
- **Analogous** (adjacent on the color wheel) -- harmonious, low contrast. Suitable for sequential schemes within a narrow hue range.
- **Triadic** (evenly spaced on the color wheel) -- balanced, high diversity. Works for qualitative schemes with three main classes.
- **Split-complementary** -- one base color plus the two colors adjacent to its complement. Provides contrast with more nuance than pure complementary.
- **Tetradic/rectangular** -- four colors forming a rectangle on the wheel. Use cautiously; works for rich qualitative schemes if value and saturation are carefully controlled.

### 3.4 Cultural Color Associations

Color meaning varies across cultures. Maps intended for international audiences must account for these differences:

| Color | Western | East Asian | Middle Eastern | Implication for Maps |
|-------|---------|------------|----------------|---------------------|
| **Red** | Danger, stop, loss | Luck, prosperity, celebration | Danger, evil | Avoid for positive values in global maps |
| **Green** | Nature, go, growth | Nature, fertility | Islam, paradise | Safe for environmental themes universally |
| **White** | Purity, peace | Death, mourning (trad.) | Purity | Be cautious with white = "empty" assumption |
| **Blue** | Trust, water, cold | Healing, relaxation | Protection, heaven | Generally safe for water features |
| **Yellow** | Caution, warmth | Royalty, sacred | Happiness, wealth | Effective for highlighting |
| **Black** | Death, formality | Power, wealth | Death, evil | Dark basemaps may carry unintended weight |
| **Purple** | Royalty, luxury | Nobility | Wealth | Use for special/unique categories |

### 3.5 Dark Mode Design

Dark basemaps have become standard for data visualization dashboards, control rooms, and night-use applications. Designing effective dark maps requires inverting many assumptions from light-mode cartography.

**Base color principles:**

- Use very dark blue-gray (`#1a1a2e` or `#0d1117`) rather than pure black (`#000000`). Pure black creates excessive contrast and visual fatigue.
- Background water should be slightly lighter than land, reversing the light-mode convention.
- Administrative boundaries work well as subtle lighter lines (`rgba(255, 255, 255, 0.08)`).

**Label and text:**

- Use light gray (`#c0c0c0` to `#e0e0e0`) for primary labels, not pure white.
- Increase halo opacity and width slightly compared to light-mode equivalents.
- Use `text-halo-color: rgba(13, 17, 23, 0.9)` for dark halos.

**Data layers:**

- Use saturated, bright colors for thematic data. Desaturated colors that work on white backgrounds disappear on dark backgrounds.
- Increase opacity of data layers to ensure sufficient contrast.
- Avoid pure yellow on dark backgrounds at small sizes (vibration effect).

**Ready-made dark styles:**

| Style | Provider | URL |
|-------|----------|-----|
| Dark Matter | CARTO | [carto.com/basemaps](https://carto.com/basemaps/) |
| Mapbox Dark | Mapbox | Via Mapbox Studio |
| Dark | MapTiler | [maptiler.com/maps](https://cloud.maptiler.com/maps/) |
| Night (custom) | Maputnik | Build from OSM Liberty base |

### 3.6 Color Tools for Cartographers

| Tool | URL | Purpose | Key Feature |
|------|-----|---------|-------------|
| **ColorBrewer 2.0** | [colorbrewer2.org](https://colorbrewer2.org) | Cartographic color schemes | Colorblind-safe, print-friendly, photocopy-safe filters |
| **CARTOColors** | [carto.com/carto-colors](https://carto.com/carto-colors/) | Perceptually uniform palettes | Designed specifically for maps |
| **Viz Palette** | [projects.susielu.com/viz-palette](https://projects.susielu.com/viz-palette) | Palette distinctiveness testing | Simulates CVD, checks name conflicts |
| **Coolors** | [coolors.co](https://coolors.co) | Color palette generator | Quick generation with lock/unlock |
| **Adobe Color** | [color.adobe.com](https://color.adobe.com) | Color harmony explorer | Wheel-based harmony rules |
| **Crameri Scientific** | [fabiocrameri.ch/colourmaps](https://www.fabiocrameri.ch/colourmaps/) | Perceptually uniform scientific palettes | Designed for geoscience; CVD-safe |
| **Chroma.js** | [gka.github.io/chroma.js](https://gka.github.io/chroma.js/) | Programmatic color manipulation | Bezier interpolation in Lab/Lch |
| **d3-color** | [github.com/d3/d3-color](https://github.com/d3/d3-color) | Color space conversion | HCL, Lab, Cubehelix support |

---

## 4. Accessibility & Inclusive Design

Accessible map design is not optional -- it is a professional obligation. Approximately 8% of males and 0.5% of females of Northern European descent have some form of color vision deficiency (CVD). Additionally, maps must be usable by people with low vision, motor impairments, and cognitive differences.

### 4.1 Color Vision Deficiency (CVD)

| Type | Affected Cone | Prevalence (Males) | Colors Confused |
|------|--------------|:---:|-----------------|
| **Deuteranopia** | Green (M-cone absent) | ~1.2% | Red/green, brown/green |
| **Deuteranomaly** | Green (M-cone shifted) | ~5% | Red/green (milder) |
| **Protanopia** | Red (L-cone absent) | ~1.0% | Red/green, red appears dark |
| **Protanomaly** | Red (L-cone shifted) | ~1.3% | Red/green (milder) |
| **Tritanopia** | Blue (S-cone absent) | ~0.003% | Blue/yellow |
| **Achromatopsia** | All cones | ~0.003% | Complete color blindness |

**Design rules for CVD safety:**

- Never use red and green as the only distinguishing variable.
- Pair color with a second channel: pattern, shape, size, or label.
- Use the blue-orange axis as the primary diverging pair (visible to most CVD types).
- Test every map with a CVD simulator before publication.

### 4.2 WCAG 2.1 Compliance for Maps

The Web Content Accessibility Guidelines (WCAG) 2.1 provide measurable standards:

| Criterion | Level AA | Level AAA | Map Application |
|-----------|----------|-----------|-----------------|
| **Text contrast** | 4.5:1 (normal text), 3:1 (large text) | 7:1 (normal), 4.5:1 (large) | Map labels, legend text, UI text |
| **Non-text contrast** | 3:1 against adjacent colors | -- | Icon boundaries, data layer edges |
| **Focus visible** | Focus indicator on interactive elements | Enhanced focus indicator | Zoom controls, layer toggles, search |
| **Target size** | 24x24 CSS px (2.2) | 44x44 CSS px | Buttons, markers, controls |
| **Resize text** | Up to 200% without loss | -- | Map UI text must reflow |

### 4.3 Redundant Encoding

Redundant encoding uses multiple visual channels to convey the same information, ensuring that if one channel is inaccessible, the information is still available:

| Primary Channel | Redundant Channel | Example |
|----------------|-------------------|---------|
| Color hue | Pattern/texture | Hatched vs. solid fill for two land-use types |
| Color hue | Shape | Different point markers per category |
| Color value | Size | Larger circles for higher values (graduated) |
| Color | Label | Direct value labels on choropleth polygons |
| Color | Position | Ordered legend entries match spatial ordering |

### 4.4 Screen Reader Compatibility

Web maps are notoriously inaccessible to screen reader users. Strategies to improve this:

- **ARIA live regions.** Announce map state changes (zoom level, visible features) via `aria-live="polite"`.
- **Structured alt text.** Provide a meaningful `alt` attribute or `aria-label` that summarizes the map's content and key findings, not just "Map of population density."
- **Data table fallback.** Offer an accessible data table as an alternative view of the same information.
- **Landmark roles.** Use `role="application"` for the map container and provide instructions for keyboard interaction.

```html
<!-- Accessible map container -->
<div id="map"
     role="application"
     aria-label="Interactive choropleth map showing population density
                 across European countries in 2024. Use arrow keys to
                 pan and +/- to zoom. Press Tab to cycle through countries."
     tabindex="0">
</div>
<!-- Alternative data table -->
<details>
  <summary>View data as table</summary>
  <table aria-label="Population density by country">
    <thead><tr><th>Country</th><th>Density (per km2)</th></tr></thead>
    <tbody>
      <tr><td>Netherlands</td><td>521</td></tr>
      <tr><td>Belgium</td><td>383</td></tr>
      <!-- ... -->
    </tbody>
  </table>
</details>
```

### 4.5 Keyboard Navigation

Interactive web maps must be operable without a mouse:

| Action | Key Binding | Implementation |
|--------|-------------|----------------|
| Pan | Arrow keys | Built into MapLibre/Mapbox GL |
| Zoom in | `+` or `=` | Built into MapLibre/Mapbox GL |
| Zoom out | `-` | Built into MapLibre/Mapbox GL |
| Rotate | Shift + Arrow | Built into MapLibre/Mapbox GL |
| Select feature | Enter/Space on focused feature | Custom implementation required |
| Cycle features | Tab | Custom implementation required |
| Close popup | Escape | Built into most popup components |
| Full screen | `F` | Custom implementation |

### 4.6 Touch Target Sizes for Mobile

Mobile maps must accommodate finger-based interaction:

- **Minimum touch target:** 44x44 CSS pixels (Apple HIG) or 48x48 dp (Material Design).
- **Spacing between targets:** At least 8px gap to prevent accidental activation.
- **Interactive markers:** If point features serve as touch targets, use a minimum hit area of 44x44px even if the visual marker is smaller.
- **Pinch-to-zoom:** Ensure this does not conflict with page scroll on mobile browsers.

### 4.7 Accessibility Testing Workflow

1. **Design phase:** Select a CVD-safe palette from ColorBrewer (enable "colorblind safe" filter).
2. **Visual simulation:** Run screenshots through [Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/) for protanopia, deuteranopia, and tritanopia.
3. **Desktop simulation:** Use [Sim Daltonism](https://michelf.ca/projects/sim-daltonism/) (macOS) for real-time CVD preview.
4. **Contrast check:** Verify all text and UI elements with [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/).
5. **Automated audit:** Run [axe DevTools](https://www.deque.com/axe/devtools/) browser extension on the deployed map page.
6. **Keyboard test:** Navigate the entire map interface using only keyboard.
7. **Screen reader test:** Test with VoiceOver (macOS/iOS) or NVDA (Windows).
8. **Mobile test:** Verify touch targets and gestures on actual devices.

---

## 5. Map Styles & Basemaps

The basemap is the cartographic foundation upon which thematic data is presented. A well-chosen basemap provides spatial context without competing with the data layer for visual attention.

### 5.1 Open Basemap Styles

| Style | Provider | Aesthetic | Best For | URL |
|-------|----------|-----------|----------|-----|
| **OSM Liberty** | Community | Clean, colorful | General-purpose web maps | [maputnik.github.io/osm-liberty](https://maputnik.github.io/osm-liberty/) |
| **Positron** | CARTO | Light, minimal | Data overlay on light background | [carto.com/basemaps](https://carto.com/basemaps/) |
| **Dark Matter** | CARTO | Dark, minimal | Dashboard/dark-mode data overlay | [carto.com/basemaps](https://carto.com/basemaps/) |
| **Toner** | Stamen (via Stadia) | High-contrast B&W | Print, reference maps | [stadiamaps.com/stamen](https://stadiamaps.com/stamen/) |
| **Watercolor** | Stamen (via Stadia) | Artistic watercolor | Artistic/exhibition contexts | [stadiamaps.com/stamen](https://stadiamaps.com/stamen/) |
| **Terrain** | Stamen (via Stadia) | Terrain shading | Outdoor/topographic | [stadiamaps.com/stamen](https://stadiamaps.com/stamen/) |
| **OpenMapTiles** | OpenMapTiles | Various | Self-hosted vector tiles | [openmaptiles.org/styles](https://openmaptiles.org/styles/) |
| **Protomaps Light/Dark** | Protomaps | Minimal | PMTiles-based workflows | [protomaps.com](https://protomaps.com) |

### 5.2 Style Editors

| Editor | Open Source | Platform | Output | Key Strengths |
|--------|:---:|----------|--------|---------------|
| **Maputnik** | Yes | Browser | GL Style JSON | No account required, works with any tile source |
| **Mapbox Studio** | No | Browser (SaaS) | GL Style JSON | Polished UX, custom fonts, sprite management |
| **MapTiler Cloud** | No | Browser (SaaS) | GL Style JSON | Integrated tile hosting, auto-updates |
| **QGIS Style Manager** | Yes | Desktop | QML, SLD | Full cartographic control, print layout |
| **ArcGIS Pro** | No | Desktop | lyrx, VTPK | Enterprise integration, advanced symbology |

### 5.3 Custom Basemap Creation Workflow

Creating a custom basemap from scratch:

1. **Choose a vector tile source.** OpenMapTiles, Planetiler, or Protomaps provide global coverage.
2. **Select a starting style.** Begin with OSM Liberty or Positron as a foundation rather than starting from zero.
3. **Open in Maputnik.** Load the style JSON and connect to your tile source.
4. **Adjust the visual hierarchy:**
   - Desaturate or lighten the basemap to create a subdued ground.
   - Remove or simplify layers irrelevant to your use case.
   - Adjust label density and font sizes.
5. **Brand the style.** Apply your organization's color palette to water, land, and road fills.
6. **Test across zoom levels.** Verify legibility from z0 (world) to z18+ (building level).
7. **Export and deploy.** Download the style JSON and host it alongside your application.

### 5.4 Style JSON Specification Deep-Dive

The MapLibre/Mapbox GL Style Specification is the lingua franca of modern vector map styling. Understanding its structure is essential:

```json
{
  "version": 8,
  "name": "My Custom Style",
  "sources": {
    "openmaptiles": {
      "type": "vector",
      "url": "https://tiles.example.com/data/v3.json"
    },
    "terrain-rgb": {
      "type": "raster-dem",
      "url": "https://tiles.example.com/terrain/{z}/{x}/{y}.webp",
      "tileSize": 256
    }
  },
  "sprite": "https://tiles.example.com/sprites/v1/sprite",
  "glyphs": "https://tiles.example.com/fonts/{fontstack}/{range}.pbf",
  "layers": [
    {
      "id": "background",
      "type": "background",
      "paint": { "background-color": "#f8f4f0" }
    },
    {
      "id": "water",
      "type": "fill",
      "source": "openmaptiles",
      "source-layer": "water",
      "paint": {
        "fill-color": "#a0c8f0",
        "fill-opacity": 0.8
      }
    },
    {
      "id": "roads-primary",
      "type": "line",
      "source": "openmaptiles",
      "source-layer": "transportation",
      "filter": ["==", "class", "primary"],
      "paint": {
        "line-color": "#e8a840",
        "line-width": ["interpolate", ["exponential", 1.5], ["zoom"],
          5, 0.5,
          12, 3,
          18, 12
        ]
      }
    }
  ]
}
```

**Key concepts:**

- **Sources** define where data comes from (vector tiles, raster tiles, GeoJSON, images).
- **Layers** define how data is rendered. Layer order = z-order (first layer drawn first, at the bottom).
- **Expressions** enable data-driven styling: `["interpolate", ...]`, `["match", ...]`, `["case", ...]`, `["get", "property_name"]`.
- **Sprite sheets** contain rasterized icons referenced by name in symbol layers.
- **Glyphs** are SDF-encoded font ranges for text rendering (see Section 2.7).

### 5.5 Sprite Sheets and Icon Sets

Sprite sheets are atlas images containing all icons used by a map style, accompanied by a JSON index that maps icon names to pixel coordinates.

| Icon Set | Icons | Focus | License | URL |
|----------|:-----:|-------|---------|-----|
| **Maki** | 200+ | Map POIs (transport, amenities, landmarks) | CC0 | [labs.mapbox.com/maki-icons](https://labs.mapbox.com/maki-icons/) |
| **Temaki** | 500+ | Extended POIs (supplements Maki) | CC0 | [ideditor.github.io/temaki](https://ideditor.github.io/temaki/docs/) |
| **Material Symbols** | 3000+ | General purpose | Apache 2.0 | [fonts.google.com/icons](https://fonts.google.com/icons) |
| **Font Awesome** | 2000+ | General purpose | Free (subset) + Pro | [fontawesome.com](https://fontawesome.com) |
| **Mapbox Maki** | 200+ | Mapbox Studio default set | CC0 | Bundled with Mapbox Studio |

**Generating custom sprite sheets:**

```bash
# Using spritezero-cli to generate a sprite sheet from SVG icons
npm install -g @mapbox/spritezero-cli
spritezero my-sprite ./icons/   # generates my-sprite.png + my-sprite.json
spritezero my-sprite@2x ./icons/ --retina  # 2x for high-DPI displays
```

> **Cross-reference:** For MapLibre GL JS integration patterns and library comparisons, see [2D Mapping Libraries](../js-bindbox/2d-mapping.md).

---

## 6. Legend & Layout Composition

The map legend and surrounding layout elements frame the cartographic narrative. Poor layout undermines even excellent data and styling.

### 6.1 Legend Design Patterns

| Legend Type | Data Type | Design Approach |
|-------------|-----------|-----------------|
| **Graduated symbols** | Quantitative (sizes) | Show 3-5 reference circles with values; nest them concentrically |
| **Categorical/qualitative** | Nominal classes | Color swatches with labels; order by importance or spatial prevalence |
| **Sequential choropleth** | Ordered numeric | Continuous color bar or stepped swatches with class breaks |
| **Diverging choropleth** | Numeric with midpoint | Symmetric color bar with labeled center (zero/mean) |
| **Bivariate** | Two variables | 3x3 or 4x4 color grid with axis labels for each variable |
| **Proportional symbol** | Continuous numeric | Nested circles showing min, median, max values |
| **Continuous ramp** | Unclassed numeric | Smooth gradient bar with labeled endpoints and midpoint |
| **Dot density** | Count per area | "1 dot = N units" statement with example dot |

### 6.2 Interactive Legend Patterns

For web maps, legends should be more than static references. Modern interactive legends serve as filter and exploration controls:

- **Toggle visibility.** Clicking a legend entry toggles the corresponding layer on/off.
- **Highlight on hover.** Hovering over a legend entry highlights matching features on the map.
- **Filter by class.** Clicking isolates a single class, dimming all others.
- **Dynamic range.** For continuous data, a range slider in the legend filters the visible value range.
- **Linked brushing.** Legend interactions are coordinated with companion charts (histogram, scatterplot).

```javascript
// Interactive legend: toggle layer visibility on click
document.querySelectorAll('.legend-item').forEach(item => {
  item.addEventListener('click', () => {
    const layerId = item.dataset.layerId;
    const visibility = map.getLayoutProperty(layerId, 'visibility');
    map.setLayoutProperty(
      layerId,
      'visibility',
      visibility === 'visible' ? 'none' : 'visible'
    );
    item.classList.toggle('legend-item--disabled');
  });
});
```

### 6.3 Layout Grid Systems

**For web maps:**

- Use CSS Grid or Flexbox to position map, legend, title, and controls.
- The map should occupy the largest share of viewport area (70-85%).
- Floating panels (legend, layer selector) should be semi-transparent or collapsible to maximize map real estate.
- Responsive design: stack panels below the map on narrow viewports.

**For print maps:**

- Use a modular grid (e.g., 12-column) aligned to the page margins.
- Standard margin proportions: 1 unit top, 1.5 units bottom, 1 unit sides (accounts for title at top, sources at bottom).
- Align legend, scale bar, and north arrow to grid lines for visual order.
- QGIS Print Layout provides snap-to-grid and guide systems for this purpose.

### 6.4 Scale Bars

Scale bars communicate distance on the map. Key design considerations:

- **Style options:** Alternating fill bar (most common), single-line with ticks, stepped bar, verbal scale ("1 cm = 1 km").
- **Units:** Match the audience's convention (metric for international, imperial for US domestic).
- **Web Mercator distortion:** On Web Mercator projected maps, the scale bar is only accurate at the latitude where it is displayed. MapLibre's `ScaleControl` dynamically adjusts the bar as the user pans. For static maps, consider noting "scale accurate at [latitude]."
- **Placement:** Bottom-left is conventional, but any location that does not obscure key data is acceptable.

```javascript
// MapLibre scale control
map.addControl(new maplibregl.ScaleControl({
  maxWidth: 200,
  unit: 'metric'  // or 'imperial', 'nautical'
}), 'bottom-left');
```

### 6.5 North Arrows

- Include a north arrow only when north is not "up" (e.g., rotated or oblique projections, polar views).
- For standard north-up Web Mercator maps, a north arrow is redundant and adds visual clutter.
- If included, use a simple, understated design. Ornate compass roses are appropriate only for decorative or historical-style maps.
- Place in a corner that does not interfere with data or other elements (top-right is conventional).

### 6.6 Inset and Overview Maps

Inset maps provide geographic context -- showing where the main map area falls within a larger region:

- **Location inset:** Small-scale overview with a rectangle indicating the main map extent.
- **Detail inset:** Large-scale zoom into a dense area (e.g., a city center within a national map).
- **Comparison inset:** Same area at a different time or with different data.

Design rules:
- Give the inset a clear border to separate it from the main map.
- Use a simplified style (fewer labels, lighter colors) so it does not compete.
- Place in a corner where it does not obscure primary content.

### 6.7 Map Margin and Bleed for Print

| Term | Definition | Typical Value |
|------|-----------|---------------|
| **Margin** | Space between content and trim edge | 10-15mm for A-series paper |
| **Bleed** | Content extending beyond the trim edge (cut tolerance) | 3-5mm |
| **Safe area** | Region where critical content must stay | Margin + 5mm inward |
| **Gutter** | Inner margin for bound documents | 15-20mm |

For QGIS Print Layout:
- Set page size with bleed included (e.g., A3 + 6mm bleed = 303 x 426mm).
- Extend background fills to the bleed edge.
- Keep all text and legends within the safe area.

---

## 7. Print Cartography

Despite the dominance of web maps, print cartography remains essential for field work, publications, reports, planning documents, and exhibition displays.

### 7.1 CMYK Color Considerations

Screens display color in RGB (additive). Printers use CMYK (subtractive). Colors that look vivid on screen may appear dull or shifted in print.

- **Gamut limitations.** CMYK cannot reproduce highly saturated blues, cyans, and oranges that are possible in sRGB. Expect these colors to shift.
- **Soft proofing.** Use CMYK preview mode in your design tool before sending to print.
- **ICC profiles.** Use the printer's ICC profile for accurate soft proofing. Generic profiles (e.g., FOGRA39 for European coated paper) are a reasonable default.
- **Rich black.** For large black areas, use rich black (C:40 M:30 Y:30 K:100) rather than pure K:100, which prints as a washed-out gray.

### 7.2 DPI Requirements

| Output | Minimum DPI | Recommended DPI | Notes |
|--------|:-----------:|:---------------:|-------|
| Professional print | 300 | 300-600 | Standard for commercial printing |
| Office/inkjet print | 150 | 200-300 | Acceptable for internal reports |
| Large-format poster | 150 | 200 | Viewed at greater distance |
| Draft/proof | 72-150 | 150 | For review only |
| Screen/web | 72-96 | 72-96 (or 2x for Retina) | DPI is irrelevant; pixel dimensions matter |

### 7.3 QGIS Print Layout Composer

QGIS Print Layout is the state-of-the-art open-source tool for print cartography. Key capabilities:

- **Multiple map frames.** Place multiple map views (main map + insets) on a single layout.
- **Locked scales and extents.** Lock each map frame to a specific scale and extent.
- **Atlas generation.** Automatically generate a series of maps (one per feature in a coverage layer) -- essential for map books and field sheets.
- **Data-defined properties.** Drive label text, colors, and visibility from attribute data.
- **Legends.** Auto-generated legends that update when layer styling changes.
- **Scale bars and north arrows.** Built-in widgets with multiple style options.
- **HTML frames.** Embed dynamic HTML/CSS content in the layout for tables or charts.
- **Export options:** PDF (vector), SVG, PNG, TIFF.

**Atlas generation workflow:**

1. Create a polygon coverage layer (e.g., grid cells, administrative boundaries).
2. Open Print Layout and enable "Generate an Atlas" in the Atlas panel.
3. Set the coverage layer and optional filter expression.
4. Configure the map frame to be "Controlled by Atlas."
5. Use atlas expressions in labels: `[% @atlas_featurenumber %]`, `[% "name" %]`.
6. Export as multi-page PDF or individual image files.

### 7.4 Export Formats

| Format | Vector/Raster | Best For | Notes |
|--------|:---:|----------|-------|
| **PDF** | Vector | Print production, sharing | Preserves text, scales without loss |
| **SVG** | Vector | Post-processing in Illustrator/Inkscape | Editable paths and text |
| **TIFF** | Raster | Archival, georeferenced output | Supports CMYK, LZW compression |
| **EPS** | Vector | Legacy print workflows | Being replaced by PDF |
| **PNG** | Raster | Web, presentations | 8-bit or 24-bit, with transparency |
| **GeoTIFF** | Raster | Georeferenced map images | Embeds CRS and extent metadata |

### 7.5 Overprint and Registration

- **Overprint.** When printing dark text over colored backgrounds, set text to overprint (rather than knockout) to avoid white halos from registration errors.
- **Registration marks.** Include crop marks and registration targets for commercial print runs.
- **Spot colors.** For brand-critical colors, specify Pantone spot colors in addition to CMYK process builds.

---

## 8. Infographic Maps

Infographic maps combine cartographic elements with charts, text, illustrations, and annotations to tell a data story in a single composition. They are increasingly common in journalism, reports, and social media.

### 8.1 Combining Map, Chart, and Text

The key challenge is visual hierarchy across heterogeneous elements:

1. **Map as anchor.** The map should be the largest and most visually prominent element, occupying 50-70% of the composition.
2. **Charts as supporting evidence.** Bar charts, line charts, or pie charts placed adjacent to the map provide statistical context. Align chart axes to map regions where possible.
3. **Text as narrative.** Headline, subheadline, and body text frame the story. Keep body text concise (50-100 words for social media, 200-400 for reports).
4. **Flow direction.** Guide the reader's eye in a Z-pattern (left-right, top-bottom) or an F-pattern. Use visual weight (size, color) to direct attention.

### 8.2 Annotation and Callout Design

Annotations transform a map from a reference tool into a narrative:

- **Leader lines.** Connect callout boxes to specific map locations. Use straight lines with a single bend (not curved or multi-bend).
- **Callout boxes.** Semi-transparent background, rounded corners (2-4px radius), subtle border. Keep text to 1-3 sentences.
- **Numbered markers.** Use numbered circles on the map keyed to a numbered list of annotations outside the map frame.
- **Highlight regions.** Use a semi-transparent overlay or desaturate the entire map except the area of interest.

### 8.3 Data Storytelling through Map Composition

Effective infographic maps follow a narrative arc:

- **Setup.** The title and subtitle establish context ("Where are Europe's forests disappearing?").
- **Evidence.** The map displays the spatial pattern. Color, size, or animation encode the variable.
- **Insight.** Annotations call attention to notable patterns ("Forest loss concentrated in southeastern Romania").
- **Context.** Companion charts or small multiples provide temporal or comparative context.
- **Source.** Attribution and methodology notes establish credibility.

### 8.4 Tools for Infographic Maps

| Tool | Type | Strengths | Limitations |
|------|------|-----------|-------------|
| **Figma** | Browser-based design | Collaborative, component-based, vector export | No native GIS capabilities; import map as image/SVG |
| **Adobe Illustrator** | Desktop design | Industry standard, advanced path editing, CMYK support | Expensive, steep learning curve |
| **Inkscape** | Desktop design (open source) | Free, SVG-native, extensible | Less polished UX than Illustrator |
| **QGIS + post-processing** | GIS + design | Generate map in QGIS, export SVG, refine in Illustrator/Inkscape | Two-step workflow |
| **Observable Framework** | Code-based | Reproducible, interactive, embeddable | Requires JavaScript proficiency |
| **Datawrapper** | Browser-based | Quick, accessible, responsive maps + charts | Limited cartographic control |

**Recommended workflow for publication-quality infographic maps:**

1. Prepare and style the map in QGIS or MapLibre.
2. Export as SVG (for vector editing) or high-DPI PNG (for raster composition).
3. Import into Figma or Illustrator.
4. Add annotations, charts, text, and branding.
5. Export final composition as PDF (print) or PNG/SVG (web).

---

## 9. Tool Comparison

A comprehensive comparison of the major cartographic design tools available to GIS professionals.

### 9.1 Comparison Matrix

| Capability | Maputnik | Mapbox Studio | QGIS | ArcGIS Pro | Figma | Illustrator | Inkscape | MapTiler |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Open source** | Yes | No | Yes | No | No | No | Yes | No |
| **Platform** | Browser | Browser | Desktop | Desktop | Browser | Desktop | Desktop | Browser |
| **Vector tile styling** | ++ | ++ | + | ++ | -- | -- | -- | ++ |
| **Print layout** | -- | -- | ++ | ++ | + | ++ | ++ | -- |
| **Atlas/map series** | -- | -- | ++ | ++ | -- | -- | -- | -- |
| **Custom fonts** | + | ++ | ++ | ++ | ++ | ++ | ++ | ++ |
| **Sprite management** | + | ++ | N/A | N/A | N/A | N/A | N/A | ++ |
| **Data-driven styling** | ++ | ++ | ++ | ++ | -- | -- | -- | ++ |
| **CJK text support** | + | ++ | ++ | ++ | ++ | ++ | + | ++ |
| **CMYK export** | -- | -- | + | ++ | -- | ++ | + | -- |
| **Collaboration** | -- | + | -- | + | ++ | + | -- | + |
| **Cost** | Free | Free tier + paid | Free | License | Free tier + paid | Subscription | Free | Free tier + paid |
| **Output format** | GL JSON | GL JSON | QML, SLD, PDF, SVG | lyrx, VTPK, PDF | SVG, PNG, PDF | AI, SVG, PDF, EPS | SVG, PDF, PNG | GL JSON |

`++` = excellent, `+` = good, `--` = not supported or poor, `N/A` = not applicable.

### 9.2 When to Use Which Tool

| Scenario | Recommended Tool | Rationale |
|----------|-----------------|-----------|
| Styling a MapLibre vector tile basemap | **Maputnik** | Free, browser-based, outputs GL Style JSON directly |
| Creating a branded basemap with custom fonts | **Mapbox Studio** | Superior font and sprite management |
| Producing a print map for a report | **QGIS Print Layout** | Full cartographic layout control, free |
| Generating 500 field maps from a grid | **QGIS Atlas** | Automated map series generation |
| Enterprise GIS with advanced symbology | **ArcGIS Pro** | Unmatched symbology engine for complex cartography |
| Designing an infographic map | **Figma** or **Illustrator** | Design-first tools with precise layout control |
| Quick web map style prototyping | **MapTiler Cloud** | Integrated hosting and style editing |
| Post-processing a map export | **Inkscape** | Free vector editor, strong SVG support |

### 9.3 Dark Arts: Advanced Tips

Undocumented or non-obvious techniques that experienced cartographers rely on:

- **Maputnik: JSON editing.** For complex expressions that the visual editor cannot handle, switch to the JSON editor tab. Paste expressions directly (e.g., multi-stop `interpolate` with `["case", ...]` fallbacks).
- **Mapbox Studio: style duplication.** Duplicate a Mapbox style, then open the duplicate JSON in Maputnik to remove Mapbox token dependencies for use with MapLibre.
- **QGIS: geometry generators.** Use geometry generator symbology to create visual elements (halos, arrows, offset labels) that do not exist in the actual data. Expression: `buffer($geometry, 50)` for a visual buffer ring.
- **QGIS: blending modes.** Apply Multiply, Screen, or Overlay blending modes to layers for effects similar to Photoshop compositing -- useful for hillshade over a hypsometric tint.
- **MapLibre: composite expressions.** Chain `["match", ...]` inside `["interpolate", ...]` to create zoom-dependent, data-driven styling in a single expression.
- **Figma: auto-layout for legends.** Build legend entries as Figma auto-layout components. Adding or removing entries automatically reflows the legend without manual adjustment.
- **Inkscape: trace bitmap.** Convert a raster hillshade to vector contours for stylized terrain depictions in SVG maps.
- **Color theft.** Use a browser eyedropper on well-designed published maps (NYT, Washington Post, NatGeo) to extract their exact color values. Study their choices before designing your own palettes.

---

## 10. Multi-Scale Design

Multi-scale design is the practice of tailoring map content and styling to different zoom levels or cartographic scales. A well-designed multi-scale map reveals appropriate detail at every zoom -- neither overwhelming the reader at small scales nor appearing empty at large scales.

> **Cross-reference:** For vector tile generation with built-in multi-scale generalization, see [Tile Servers](../js-bindbox/tile-servers.md).

### 10.1 Zoom Level Reference

Understanding the relationship between web map zoom levels and cartographic scales is fundamental:

| Zoom | Approx. Scale | Ground Resolution (equator) | Typical Content |
|:----:|---------------|:---------------------------:|-----------------|
| 0 | 1:500,000,000 | 156 km/px | Globe view, continent labels |
| 2 | 1:150,000,000 | 39 km/px | Continent outlines, ocean labels |
| 4 | 1:35,000,000 | 10 km/px | Countries, major water bodies |
| 6 | 1:10,000,000 | 2.4 km/px | States/provinces, large cities |
| 8 | 1:2,000,000 | 600 m/px | Major roads, medium cities |
| 10 | 1:500,000 | 150 m/px | All roads, towns, land use |
| 12 | 1:150,000 | 38 m/px | Streets, neighborhoods, POIs |
| 14 | 1:35,000 | 10 m/px | Building footprints, addresses |
| 16 | 1:8,000 | 2.4 m/px | Detailed building outlines, furniture |
| 18 | 1:2,000 | 0.6 m/px | Maximum detail, survey-grade |

### 10.2 Zoom-Dependent Styling Strategies

**Line width progression:**

Roads, rivers, and boundaries should increase in width as the user zooms in. Use exponential or interpolate expressions to create smooth transitions:

```json
{
  "id": "roads-primary",
  "type": "line",
  "source": "openmaptiles",
  "source-layer": "transportation",
  "filter": ["==", "class", "primary"],
  "paint": {
    "line-width": [
      "interpolate", ["exponential", 1.5], ["zoom"],
      6, 0.5,
      10, 2,
      14, 8,
      18, 24
    ],
    "line-color": "#fbb03b"
  }
}
```

**Label density management:**

The most common multi-scale failure is label clutter. Strategies:

1. **Priority ranking.** Assign numeric priority to features (capital cities > major cities > towns > villages). Use `symbol-sort-key` in MapLibre to render higher-priority labels first.
2. **Min/max zoom.** Set `minzoom` and `maxzoom` on symbol layers so labels appear only at appropriate zoom levels.
3. **Text size scaling.** Increase label size at higher zooms, but not linearly -- use a curve that grows slowly to avoid giant text at z18.
4. **Collision detection.** MapLibre automatically hides colliding labels. Tune `text-allow-overlap`, `text-ignore-placement`, and `text-padding` to control density.
5. **Different label sets per zoom range.** At z0-z5, show only country names. At z6-z9, add state/province names. At z10-z13, add city names. At z14+, add street names and POIs.

```json
{
  "id": "place-city",
  "type": "symbol",
  "source": "openmaptiles",
  "source-layer": "place",
  "minzoom": 6,
  "maxzoom": 14,
  "filter": ["==", "class", "city"],
  "layout": {
    "text-field": "{name}",
    "text-size": ["interpolate", ["linear"], ["zoom"], 6, 11, 12, 16],
    "symbol-sort-key": ["get", "rank"],
    "text-padding": 10,
    "text-max-width": 8
  }
}
```

### 10.3 Feature Generalization by Zoom Level

| Zoom Range | Generalization Strategy | Implementation |
|:----------:|------------------------|----------------|
| **0-3** | Show only continent/ocean boundaries; extreme simplification | Tippecanoe `--minimum-zoom=0 --maximum-zoom=3 -aD` |
| **4-7** | Country boundaries, major rivers, largest cities | Filter by `admin_level <= 2`, `rank <= 3` |
| **8-10** | State boundaries, primary roads, medium cities | Simplify geometry tolerance 50-100m |
| **11-13** | All roads, land use, building clusters | Simplify geometry tolerance 10-20m |
| **14-16** | Individual buildings, street names, POIs | Minimal simplification, full detail |
| **17+** | Maximum detail, address points, utility infrastructure | Raw geometry, no simplification |

**Tippecanoe strategies for multi-scale tiles:**

```bash
# Generate tiles with automatic feature dropping and simplification
tippecanoe \
  --output=output.pmtiles \
  --minimum-zoom=0 \
  --maximum-zoom=14 \
  --drop-densest-as-needed \
  --extend-zooms-if-still-dropping \
  --simplification=10 \
  --detect-shared-borders \
  input.geojson
```

### 10.4 Scale-Dependent Symbology Patterns

**Pattern: Road hierarchy across scales.**

| Zoom | Motorway | Primary | Secondary | Tertiary | Residential |
|:----:|:--------:|:-------:|:---------:|:--------:|:-----------:|
| 5 | 1px line | -- | -- | -- | -- |
| 8 | 2px cased | 1px line | -- | -- | -- |
| 10 | 4px cased | 2px cased | 1px line | -- | -- |
| 12 | 6px cased | 4px cased | 2px cased | 1px line | -- |
| 14 | 10px cased | 6px cased | 4px cased | 2px cased | 1px line |
| 16 | 16px cased | 10px cased | 6px cased | 4px cased | 2px line |

"Cased" means a wider background line (casing) with a narrower fill line on top, creating a bordered road appearance.

**Pattern: Building visibility.**

- z0-z12: No buildings visible.
- z13: Aggregate building footprints into urban area polygons (light gray fill).
- z14: Show individual building footprints (darker gray fill, no outline).
- z15+: Show building footprints with outlines and optional 3D extrusion via `fill-extrusion`.

**Pattern: Boundary detail.**

- z0-z4: Country boundaries only (`admin_level = 2`), 1px solid line.
- z5-z8: Add state/province boundaries (`admin_level = 4`), 0.5px dashed line.
- z9-z12: Add county/district boundaries (`admin_level = 6`), 0.3px dotted line.
- z13+: Add municipal boundaries (`admin_level = 8`), 0.2px dotted line.

### 10.5 Overzooming and Underzooming

**Overzooming** occurs when the user zooms beyond the maximum zoom level of the tile source. MapLibre can render overzoom tiles by upscaling the highest available zoom level, but the data will appear blocky (raster) or sparse (vector).

**Underzooming** occurs when tiles for a given zoom level are not yet available. MapLibre displays parent tiles at lower resolution until the correct tiles load.

**Design considerations:**

- Set `maxzoom` on tile sources to the highest zoom level where data is available. MapLibre will overzoom beyond this.
- For vector tiles, overzoom is usually acceptable up to z16-z18 because the geometry remains sharp.
- For raster tiles, overzoom produces visible pixelation. Limit overzoom to 1-2 levels beyond the native resolution.
- Use the `raster-resampling` property (`"linear"` or `"nearest"`) to control how raster tiles are interpolated during overzoom.

---

## 11. Basemap Design

A basemap is the foundational canvas on which thematic data is displayed. Designing a basemap from scratch -- or customizing an existing one -- is one of the most challenging and rewarding cartographic tasks.

> **Cross-reference:** For data sources used in basemap construction, including OSM data, elevation, and satellite imagery, see [Vector Data](../data-sources/vector-data.md), [Elevation & Terrain](../data-sources/elevation-terrain.md), and [Satellite Imagery](../data-sources/satellite-imagery.md).

### 11.1 Basemap Design Philosophy

A basemap must be **subordinate** to the thematic data overlaid on it. If the basemap competes with the data layer for visual attention, the map fails. This constraint drives every design decision:

- **Desaturate.** Basemap colors should be muted. Use low-chroma versions of the natural colors (soft gray-green for land, soft blue-gray for water).
- **Reduce contrast.** Light basemaps should use a narrow value range (e.g., #f2f2f0 for land, #dae6e8 for water). Dark basemaps should use dark values with minimal variation.
- **Simplify.** Show only the context features necessary for spatial orientation. A land-use polygon layer that adds visual noise without aiding orientation should be removed or heavily simplified.
- **Hierarchy.** Even within the basemap, establish a visual hierarchy: major water bodies > administrative boundaries > major roads > minor roads > land-use fills.

### 11.2 Styling OpenStreetMap Data

OSM is the most common data source for custom basemaps. The OpenMapTiles schema provides a standardized way to access OSM data in vector tiles.

**Key OpenMapTiles layers and their styling:**

| Layer | Contains | Typical Style |
|-------|----------|--------------|
| `water` | Ocean, lake, river polygons | Flat fill #aad3df or desaturated variant |
| `waterway` | River/stream/canal lines | Line, width by class, same color as water polygons |
| `landcover` | Forest, grass, farmland, ice | Subtle green/tan fills, 30-60% opacity |
| `landuse` | Residential, commercial, industrial | Very subtle fills, nearly invisible |
| `transportation` | Roads, railways, paths | Line hierarchy by class (see Section 10.4) |
| `boundary` | Administrative borders | Dashed/dotted lines, weight by admin level |
| `building` | Building footprints | Gray fill, visible at z13+ |
| `place` | City/town/village points | Symbol layer for labels |
| `poi` | Points of interest | Icon + label, visible at z14+ |
| `aeroway` | Runways, taxiways | Light gray fills and lines |
| `park` | Parks, nature reserves | Soft green fill, slightly more saturated than generic landcover |

### 11.3 Terrain Visualization

Terrain representation adds depth and spatial understanding to any basemap. Multiple techniques can be combined for rich topographic rendering.

**Hillshade:**

Hillshade simulates light falling on a 3D surface, creating shadows that reveal topographic relief.

- **Light direction.** The conventional illumination azimuth is 315 degrees (northwest). This leverages the human visual system's assumption that light comes from the upper left. Avoid south-facing illumination, which creates a depth inversion illusion.
- **Altitude angle.** 45 degrees is standard. Lower angles (30 degrees) create more dramatic shadows but may obscure flat areas. Higher angles (60 degrees) produce subtle, soft relief.
- **Multidirectional hillshade.** Combines multiple light directions to eliminate flat spots that occur when slopes are parallel to the light source. QGIS and GDAL both support this.

```bash
# GDAL: generate hillshade from a DEM
gdaldem hillshade input_dem.tif hillshade.tif -z 1.5 -az 315 -alt 45

# Multidirectional hillshade
gdaldem hillshade input_dem.tif hillshade_multi.tif -multidirectional

# Color relief (hypsometric tinting)
gdaldem color-relief input_dem.tif color_ramp.txt color_relief.tif
```

**Hypsometric tinting:**

Color the terrain surface by elevation using a color ramp. Traditional hypsometric schemes use green for lowlands, brown/tan for mountains, and white for snow-capped peaks. Modern cartography often uses more subtle, desaturated schemes.

**Color ramp file for GDAL `color-relief`:**

```
0     20  100  20
500   200 170  80
1500  160 120  60
3000  220 220 220
5000  255 255 255
```

**Contour lines:**

- Generate from DEM using GDAL: `gdal_contour -a elevation -i 100 input_dem.tif contours.gpkg`
- Style index contours (every 500m) with heavier weight than intermediate contours.
- Label contours with elevation values, placed along the line, uphill-facing.

### 11.4 Bathymetry

Underwater terrain visualization follows similar principles to land terrain, but with its own conventions:

- **Color convention.** Light blue for shallow water, dark blue for deep water (inverse of land hypsometric tinting). The standard palette runs from `#d4eef7` (near-shore) to `#08306b` (abyssal).
- **Depth contours (isobaths).** Use labeled contour lines at standard oceanographic intervals: 200m (continental shelf edge), 1000m, 2000m, 3000m, 4000m, 5000m.
- **Submarine hillshade.** Apply hillshade to bathymetric DEMs using the same techniques as land, but with inverted color overlay (dark hillshade on blue gradient).
- **Data sources.** GEBCO (General Bathymetric Chart of the Oceans), NOAA ETOPO, EMODnet (European waters). See [Ocean & Maritime Data](../data-sources/ocean-maritime.md).

### 11.5 Satellite and Aerial Imagery as Basemap

When using imagery as a basemap:

- **Reduce saturation.** Apply a desaturation filter (30-50%) to prevent the imagery from overwhelming thematic data.
- **Increase brightness.** Slightly brighten the imagery to improve label contrast.
- **Overlay a semi-transparent white or dark wash.** A 10-20% white overlay on satellite imagery creates a softer base for light-colored labels.
- **Label styling.** Labels over satellite imagery require strong halos (2-3px) or dark panel backgrounds to remain legible.
- **Performance.** Satellite tile layers are large. Use WebP format tiles where available and set appropriate `maxzoom` to avoid unnecessary tile requests.

---

## 12. Icon & Symbol Design

Icons and symbols are the cartographic vocabulary for point features. Effective icon design requires balancing recognizability, aesthetic consistency, and technical constraints of the rendering engine.

### 12.1 Maki Icons

[Maki](https://labs.mapbox.com/maki-icons/) is the most widely used open-source icon set for map cartography, maintained by Mapbox.

**Key characteristics:**

- **Sizes.** Available in 11px and 15px variants, designed to be pixel-perfect at those sizes.
- **Style.** Monochrome, filled silhouettes with simple geometry. No fine detail that would be lost at small sizes.
- **License.** CC0 (public domain). Free for any use.
- **Format.** SVG source files, pre-built sprite sheets for Mapbox/MapLibre GL.

**Common Maki icons for cartography:**

| Category | Icons | Typical Use |
|----------|-------|-------------|
| **Transportation** | airport, bus, rail, ferry, heliport, fuel | Transit maps, navigation |
| **Amenities** | restaurant, cafe, bar, shop, hospital, pharmacy | POI layers, tourism maps |
| **Recreation** | park, campsite, swimming, skiing, golf | Outdoor/recreation maps |
| **Civic** | town-hall, police, fire-station, library, school | Municipal maps |
| **Nature** | mountain, volcano, waterfall, wetland | Topographic and nature maps |

### 12.2 Custom SVG Markers

When Maki does not cover your needs, create custom SVG markers:

**Design guidelines for map SVG icons:**

1. **Canvas size.** Design on a 15x15 or 24x24 pixel grid. Align to pixel boundaries for crisp rendering.
2. **Simplicity.** Use no more than 2-3 distinct shapes. Fine detail is invisible at map zoom levels.
3. **Monochrome base.** Design in a single fill color. The map style will apply the color at render time via `icon-color` (in MapLibre GL).
4. **Anchor point.** For pin-style markers, set the SVG anchor at the bottom center of the pin tip. For centroid markers, center the anchor.
5. **Stroke-free.** Avoid strokes at small sizes. They add visual noise and may not render at sub-pixel widths.

```svg
<!-- Example: custom factory icon, 15x15 grid -->
<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 15 15">
  <path d="M2,14 L2,8 L5,10 L5,6 L8,8 L8,4 L11,6 L11,1 L13,1 L13,14 Z"
        fill="#333"/>
</svg>
```

### 12.3 Sprite Sheets for MapLibre / Mapbox GL

MapLibre GL JS uses **sprite sheets** -- a single image containing all icons arranged in a grid, plus a JSON index mapping icon names to pixel coordinates within the sheet.

**Sprite sheet structure:**

```
sprite.png        -- composite image of all icons at 1x resolution
sprite@2x.png     -- composite image at 2x resolution (for Retina/HiDPI)
sprite.json       -- index: { "airport": { "x": 0, "y": 0, "width": 15, "height": 15, "pixelRatio": 1 } }
sprite@2x.json    -- index for 2x sprites
```

**Generating sprite sheets:**

| Tool | Description | Command |
|------|-------------|---------|
| **spritezero-cli** | Original Mapbox sprite generator | `spritezero sprite ./svg-dir` |
| **spreet** | Rust-based sprite generator (fast, maintained) | `spreet ./svg-dir sprite` |
| **Martin** | Generates sprites as a server endpoint | Configure sprite source in Martin config |

```bash
# Using spreet to generate a sprite sheet from SVG icons
spreet ./icons sprite
# Outputs: sprite.png, sprite@2x.png, sprite.json, sprite@2x.json
```

### 12.4 SDF Icons for Dynamic Styling

Signed Distance Field (SDF) icons are a rendering technique that allows icons to be dynamically colored and sized at render time, without generating multiple pre-colored variants.

**How SDF works:**

- Each pixel in an SDF icon stores the distance to the nearest edge of the icon shape, rather than a color value.
- The renderer uses this distance field to draw the icon at any size and in any color, with smooth anti-aliased edges.
- This enables `icon-color` and `icon-halo-color` properties in MapLibre GL JS to work on the icon.

**When to use SDF icons:**

| Scenario | SDF | Pre-colored |
|----------|:---:|:-----------:|
| Icons need to change color based on data | Yes | No |
| Icons need halos | Yes | No |
| Multi-colored icons (logos, flags) | No | Yes |
| Simple monochrome symbols | Yes | Either |

**Enabling SDF in MapLibre GL:**

```javascript
// Load an SDF sprite
map.addImage('custom-marker', image, { sdf: true });

// Use it in a layer with dynamic color
map.addLayer({
  id: 'poi-layer',
  type: 'symbol',
  source: 'pois',
  layout: {
    'icon-image': 'custom-marker',
    'icon-size': 1.2
  },
  paint: {
    'icon-color': ['match', ['get', 'category'],
      'restaurant', '#e74c3c',
      'park', '#27ae60',
      'school', '#3498db',
      '#7f8c8d'  // default
    ],
    'icon-halo-color': '#ffffff',
    'icon-halo-width': 1
  }
});
```

### 12.5 Symbol Design Resources

| Resource | URL | Description |
|----------|-----|-------------|
| **Maki Icons** | [labs.mapbox.com/maki-icons](https://labs.mapbox.com/maki-icons/) | 200+ map-specific icons, CC0 |
| **Temaki** | [ideditor.github.io/temaki](https://ideditor.github.io/temaki/) | Extended icon set complementing Maki (OSM-focused) |
| **Map Icons Collection** | [map-icons.com](http://map-icons.com) | Google Maps styled marker icons |
| **Lucide** | [lucide.dev](https://lucide.dev) | Clean, consistent icon set, open source |
| **Simple Icons** | [simpleicons.org](https://simpleicons.org) | Brand/logo icons for POI categories |

---

## 13. Projection Selection Guide

Map projections transform the curved surface of the Earth onto a flat plane. Every projection distorts some combination of area, shape, distance, or direction. Choosing the right projection is a fundamental cartographic decision that affects the truthfulness of the map.

> **Cross-reference:** For D3.js projection implementations and interactive projection explorers, see [2D Mapping Libraries](../js-bindbox/2d-mapping.md). For academic standards on projections, see [Cartography Standards](../academic/cartography-standards.md).

### 13.1 Projection Properties

| Property | Definition | Preserved By |
|----------|-----------|--------------|
| **Conformal** | Preserves local angles and shapes | Mercator, Lambert Conformal Conic, Stereographic |
| **Equal-area** | Preserves relative area | Albers, Mollweide, Goode Homolosine, Equal Earth |
| **Equidistant** | Preserves distance from one or two points | Azimuthal Equidistant, Equidistant Conic |
| **Azimuthal** | Preserves direction from center | Azimuthal Equidistant, Gnomonic, Stereographic |
| **Compromise** | Minimizes overall distortion without preserving any property exactly | Robinson, Natural Earth, Winkel Tripel |

**No projection can preserve all properties simultaneously.** This is a mathematical certainty (Gauss's Theorema Egregium). The cartographer must choose which property matters most for the map's purpose.

### 13.2 Projection Decision Matrix

| Map Purpose | Recommended Projection | Why |
|-------------|----------------------|-----|
| **Web slippy map** | Web Mercator (EPSG:3857) | Industry standard for web tile maps; conformal |
| **World thematic (area comparison)** | Equal Earth, Mollweide | Equal-area preserves truthful area comparisons |
| **World general reference** | Robinson, Natural Earth, Winkel Tripel | Compromise minimizes overall distortion |
| **Continent: North America** | Albers Equal Area (EPSG:5070) | Equal-area, standard for USGS/Census |
| **Continent: Europe** | Lambert Azimuthal Equal Area (EPSG:3035) | EU standard (LAEA Europe) |
| **Country: USA (contiguous)** | Albers Equal Area or Lambert Conformal Conic | Albers for thematic, LCC for navigational |
| **Country: China** | Albers Equal Area (custom) or GK zones | Standard for national mapping |
| **Polar regions** | Stereographic Polar (EPSG:3031/3995) | Minimal distortion at poles |
| **Navigation/routing** | Mercator (conformal) | Straight lines are rhumb lines |
| **Air routes / great circles** | Gnomonic | Straight lines are great circles |
| **City/local area** | UTM zone or State Plane | Minimal distortion at local scales |
| **Oceanographic** | Spilhaus projection | Ocean-centered; shows connected world ocean |

### 13.3 The Web Mercator Problem

Web Mercator (EPSG:3857) is the dominant web mapping projection, but it has serious cartographic limitations:

- **Area distortion.** Greenland appears as large as Africa (actual ratio: 1:14). Alaska appears larger than Mexico (actual ratio: roughly equal). This distortion increases toward the poles.
- **Polar truncation.** Web Mercator cannot display latitudes beyond approximately 85.06 degrees N/S.
- **Not equal-area.** Any choropleth or thematic map in Web Mercator misrepresents area comparisons.

**Mitigations:**

- For thematic world maps, reproject to Equal Earth or Mollweide before publishing.
- If you must use Web Mercator (e.g., for a tiled basemap), add a disclaimer about area distortion.
- Use Tissot's indicatrices (distortion ellipses) to visually communicate the projection's distortion pattern.

### 13.4 D3-Geo Projections

D3.js provides the most comprehensive JavaScript projection library, with over 100 projections available through `d3-geo` and `d3-geo-projection`.

```javascript
import {
  geoMercator,
  geoAlbers,
  geoAlbersUsa,       // Composite: contiguous US + Alaska + Hawaii
  geoEqualEarth,
  geoNaturalEarth1,
  geoOrthographic,     // Globe view
  geoConicEqualArea,
  geoConicConformal,
  geoStereographic,
  geoTransverseMercator
} from 'd3-geo';

import {
  geoRobinson,
  geoMollweide,
  geoWinkelTripel,
  geoGoodeHomolosine,
  geoBertin1953,
  geoInterruptedMollweideHemispheres
} from 'd3-geo-projection';

// Example: Equal Earth projection centered on a custom meridian
const projection = geoEqualEarth()
  .center([0, 0])
  .rotate([-10, 0])   // center on 10 degrees East
  .scale(200)
  .translate([width / 2, height / 2]);
```

### 13.5 Distortion Visualization

Tissot's indicatrix is the standard method for visualizing projection distortion. Place uniform circles on the globe; after projection, their deformation reveals the distortion pattern:

- **Circle remains a circle:** No angular distortion (conformal).
- **Circle becomes an ellipse:** Angular distortion present.
- **All ellipses have equal area:** Equal-area projection.
- **Ellipses vary in size:** Area distortion present.

**Tools for distortion visualization:**

| Tool | Description |
|------|-------------|
| **D3 Tissot** | `d3-geo` can render Tissot's indicatrices directly |
| **Projection Wizard** | [projectionwizard.org](http://projectionwizard.org) -- recommends projections for a given extent |
| **Flex Projector** | Desktop app for creating custom projections with visual distortion feedback |
| **QGIS** | Generate and project Tissot circles using geometry generators |

---

## 14. Color Palettes for Specific Domains

While Sections 3.2-3.3 covered general color theory, specific data domains have established color conventions that cartographers should respect. Using domain-conventional colors improves map readability because the audience already associates certain colors with certain meanings.

> **Cross-reference:** For thematic map types and classification methods, see [Thematic Maps](thematic-maps.md). For domain-specific data sources, see the [Data Sources](../data-sources/README.md) index.

### 14.1 Environmental Science

| Variable | Conventional Palette | Notes |
|----------|---------------------|-------|
| **Vegetation / NDVI** | White-to-green sequential | Green = high vegetation; brown/tan = bare soil |
| **Temperature** | Blue-white-red diverging | Blue = cold, red = hot, white = midpoint |
| **Precipitation** | White-to-blue sequential | Darker blue = more rain; sometimes extended with purple for extremes |
| **Elevation** | Green-brown-white hypsometric | Traditional: green lowlands, brown mountains, white peaks |
| **Fire severity** | Yellow-orange-red-black | Mimics fire colors; black = complete burn |
| **Drought index** | Brown-to-blue diverging | Brown = dry, blue = wet |
| **Air quality (AQI)** | Green-yellow-orange-red-purple | WHO/EPA standard color breaks |
| **Land cover** | Anderson Level I qualitative | Standard: blue (water), green (forest), yellow (agriculture), red (urban), brown (barren) |

**Recommended palettes:**

```
Crameri Scientific:
  Temperature:  "vik" (diverging, blue-white-red)
  Elevation:    "batlow" (sequential, perceptually uniform)
  Vegetation:   "bamako" (sequential, brown-to-green)

ColorBrewer:
  Temperature:  "RdBu" (diverging, 9-class)
  Precipitation: "Blues" (sequential, 7-class)
  Drought:      "BrBG" (diverging, brown-to-green)
```

### 14.2 Public Health

| Variable | Conventional Palette | Notes |
|----------|---------------------|-------|
| **Disease incidence rate** | Yellow-orange-red sequential | Red = high incidence (aligns with "danger" association) |
| **Vaccination coverage** | Green sequential (low saturation) | Green = health, protection |
| **Life expectancy** | Diverging from national average | Blue = above avg, red = below avg |
| **Hospital accessibility** | Green-to-red diverging (with CVD caveat) | Green = good access, red = poor; pair with pattern for CVD |
| **COVID-19 maps** | Orange-red sequential | Widely established during the pandemic |
| **Mortality rate** | Purple sequential | Distinct from incidence rate colors when both on same map |

**Important consideration:** Health maps carry ethical weight. Avoid color choices that stigmatize communities. Red-saturated regions can imply blame rather than highlight need. Consider using narrative context alongside the map. See [Storytelling & Scrollytelling](storytelling-scrollytelling.md) for techniques.

### 14.3 Urban Planning

| Variable | Conventional Palette | Notes |
|----------|---------------------|-------|
| **Zoning** | Qualitative, established by local convention | Residential=yellow, commercial=red, industrial=purple, agricultural=green, mixed-use=orange (US convention) |
| **Building age** | Sequential or qualitative by era | Cool tones for old, warm for new; or distinct hues per historical era |
| **Population density** | Yellow-orange-red sequential | Classic density ramp |
| **Transit accessibility** | Green-yellow-red diverging | Green = high access (walk score) |
| **Noise levels** | Green-yellow-red sequential | Aligned with dB thresholds and EU noise mapping standards |
| **Green space** | Green sequential | Saturation indicates density/quality |
| **Property values** | Blue-purple sequential | Avoids red/green (CVD safe), neutral emotional tone |

### 14.4 Election & Political Maps

Election mapping is one of the most scrutinized and debated areas of cartographic design. Color choices carry political implications.

**Standard conventions by country:**

| Country | Left/Liberal | Right/Conservative | Other |
|---------|-------------|-------------------|-------|
| **USA** | Blue (Democrat) | Red (Republican) | Yellow/gold (Libertarian), Green (Green Party) |
| **UK** | Red (Labour) | Blue (Conservative) | Yellow (Lib Dem), Green (Green), Purple (Reform) |
| **France** | Red/pink (Left) | Blue (Right) | Yellow (centrist), Green (Ecologist) |
| **Germany** | Red (SPD), Dark red (Linke) | Black (CDU/CSU) | Green (Gruene), Yellow (FDP), Blue (AfD) |
| **China** | Red (CPC) | N/A | Typically single-party; maps focus on administrative data |
| **Australia** | Red (Labor) | Blue (Liberal/National) | Green (Greens), Orange (One Nation) |

**Advanced election map techniques:**

- **Bivariate maps.** Show vote share AND turnout simultaneously using a 3x3 color grid.
- **Cartograms.** Resize geographic units by population or electoral weight to counter the visual overrepresentation of large, sparsely populated areas.
- **Purple maps.** Blend red and blue proportionally rather than showing winner-take-all, to reveal the margin of victory.
- **Dot density.** Place one dot per N voters, colored by party, to show geographic distribution without areal distortion.

### 14.5 Socioeconomic Data

| Variable | Conventional Palette | Notes |
|----------|---------------------|-------|
| **Income / GDP** | Blue or green sequential | Neutral, non-judgmental |
| **Inequality (Gini)** | Diverging from neutral | Center at national/global average |
| **Unemployment** | Orange-red sequential | Red = high unemployment |
| **Education attainment** | Blue-purple sequential | Progressive, aspirational tone |
| **Poverty rate** | Brown-orange sequential | Earthy tones; avoid red (implies blame) |
| **Migration (net)** | Blue-red diverging | Blue = net in-migration, red = net out-migration |
| **Demographic composition** | Qualitative (race/ethnicity) | Use maximally distinct hues; standard sets vary by country |

---

## 15. Advanced MapLibre Styling

MapLibre GL's expression system provides a powerful declarative language for data-driven styling. Mastering expressions is essential for creating sophisticated, production-quality web maps.

> **Cross-reference:** For MapLibre GL JS core setup and performance optimization, see [2D Mapping Libraries](../js-bindbox/2d-mapping.md) and [Performance Optimization](../js-bindbox/performance-optimization.md).

### 15.1 Expression Fundamentals

MapLibre expressions are JSON arrays in prefix notation. The first element is the operator, and subsequent elements are arguments:

```json
["operator", "argument1", "argument2"]
```

**Expression categories:**

| Category | Examples | Purpose |
|----------|---------|---------|
| **Data** | `["get", "property"]`, `["has", "property"]`, `["feature-state", "key"]` | Access feature properties |
| **Math** | `["+", a, b]`, `["*", a, b]`, `["log2", x]`, `["abs", x]` | Arithmetic operations |
| **String** | `["concat", a, b]`, `["upcase", s]`, `["slice", s, start, end]` | Text manipulation |
| **Logic** | `["==", a, b]`, `["all", cond1, cond2]`, `["case", ...]` | Conditional logic |
| **Interpolation** | `["interpolate", ...]`, `["step", ...]` | Continuous and discrete ramps |
| **Color** | `["rgb", r, g, b]`, `["to-color", val]` | Color construction |
| **Type** | `["to-number", val]`, `["to-string", val]`, `["typeof", val]` | Type conversion |
| **Zoom** | `["zoom"]` | Current map zoom level |
| **Geometry** | `["geometry-type"]`, `["id"]` | Feature metadata |

### 15.2 Data-Driven Styling Patterns

**Pattern 1: Categorical coloring with `match`.**

```json
{
  "fill-color": [
    "match", ["get", "landuse"],
    "residential", "#f0e6c8",
    "commercial", "#f7d1d1",
    "industrial", "#d6c6e1",
    "retail", "#f5c9c9",
    "park", "#c8e6c9",
    "forest", "#a5d6a7",
    "#e0e0e0"
  ]
}
```

**Pattern 2: Continuous interpolation with `interpolate`.**

```json
{
  "circle-radius": [
    "interpolate", ["linear"], ["get", "population"],
    1000, 3,
    10000, 8,
    100000, 15,
    1000000, 25
  ],
  "circle-color": [
    "interpolate", ["linear"], ["get", "population"],
    1000, "#fee0d2",
    10000, "#fc9272",
    100000, "#de2d26",
    1000000, "#67000d"
  ]
}
```

**Pattern 3: Step function (discrete classes).**

```json
{
  "fill-color": [
    "step", ["get", "density"],
    "#ffffcc",
    10, "#c2e699",
    50, "#78c679",
    100, "#31a354",
    500, "#006837"
  ]
}
```

**Pattern 4: Multi-variable expression (bivariate).**

```json
{
  "fill-color": [
    "case",
    ["all", [">=", ["get", "income"], 50000], [">=", ["get", "education"], 0.5]],
      "#3b4cc0",
    ["all", [">=", ["get", "income"], 50000], ["<", ["get", "education"], 0.5]],
      "#7092c0",
    ["all", ["<", ["get", "income"], 50000], [">=", ["get", "education"], 0.5]],
      "#c07070",
    "#c0b03b"
  ]
}
```

### 15.3 Feature State for Interactive Styling

Feature state allows changing a feature's visual properties without re-rendering the tile data. This is critical for hover and selection effects:

```javascript
// Set feature state on hover
map.on('mousemove', 'counties-layer', (e) => {
  if (e.features.length > 0) {
    // Clear previous hover state
    if (hoveredId !== null) {
      map.setFeatureState(
        { source: 'counties', sourceLayer: 'counties', id: hoveredId },
        { hover: false }
      );
    }
    hoveredId = e.features[0].id;
    map.setFeatureState(
      { source: 'counties', sourceLayer: 'counties', id: hoveredId },
      { hover: true }
    );
  }
});

// Use feature-state in paint properties
map.addLayer({
  id: 'counties-layer',
  type: 'fill',
  source: 'counties',
  'source-layer': 'counties',
  paint: {
    'fill-color': '#627BC1',
    'fill-opacity': [
      'case',
      ['boolean', ['feature-state', 'hover'], false],
      0.9,
      0.5
    ]
  }
});
```

### 15.4 Composite Operations and Layer Ordering

**Layer ordering strategies:**

MapLibre renders layers in the order they appear in the style JSON (first = bottom, last = top). Common ordering:

1. `background` -- base fill
2. `landcover` fills -- forests, grass
3. `water` fills -- oceans, lakes
4. `landuse` fills -- residential, commercial
5. `building` fills -- building footprints
6. `transportation` lines (tunnels) -- below-grade roads
7. `transportation` lines (surface) -- at-grade roads
8. `transportation` lines (bridges) -- above-grade roads
9. `boundary` lines -- administrative borders
10. `waterway` labels -- river/lake names
11. `transportation` labels -- road names
12. `place` labels -- city/town names
13. `poi` labels -- points of interest

**Using `before` to insert layers:**

```javascript
// Insert a data layer below labels but above the basemap
map.addLayer({
  id: 'choropleth',
  type: 'fill',
  source: 'my-data',
  paint: { 'fill-color': '#e74c3c', 'fill-opacity': 0.7 }
}, 'waterway-label');  // <-- insert before this layer
```

### 15.5 Custom Layers (Advanced)

MapLibre supports custom WebGL layers for rendering that goes beyond the built-in layer types (particle systems, custom shaders, 3D models):

```javascript
const customLayer = {
  id: 'custom-webgl',
  type: 'custom',
  renderingMode: '2d', // or '3d'
  onAdd(map, gl) {
    // Initialize WebGL resources (shaders, buffers)
    this.program = createShaderProgram(gl, vertexShader, fragmentShader);
  },
  render(gl, matrix) {
    // Draw using WebGL commands
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.matrixLocation, false, matrix);
    gl.drawArrays(gl.TRIANGLES, 0, this.vertexCount);
  },
  onRemove(map, gl) {
    // Clean up WebGL resources
    gl.deleteProgram(this.program);
  }
};

map.addLayer(customLayer);
```

**Use cases for custom layers:**

- Wind/ocean current particle animations (see [Temporal Animation](temporal-animation.md)).
- Custom heatmap kernels with non-standard weighting.
- 3D point cloud rendering.
- Real-time sensor data visualization with custom shaders.

> **Cross-reference:** For 3D rendering with MapLibre and deck.gl, see [3D Visualization](3d-visualization.md) and [3D Mapping Libraries](../js-bindbox/3d-mapping.md).

---

## 16. Design Systems for Maps

As organizations produce multiple map products across teams, a design system ensures visual consistency, reduces redundant work, and accelerates production.

### 16.1 Map Design Tokens

Design tokens are the atomic building blocks of a design system -- named constants for colors, sizes, fonts, and spacing. Translated to cartography:

**Color tokens:**

```json
{
  "color": {
    "basemap": {
      "land": "#f2f0eb",
      "water": "#c6e2f5",
      "park": "#d5e8d4",
      "building": "#dfdbd7",
      "road-fill": "#ffffff",
      "road-casing": "#c8c4c0"
    },
    "thematic": {
      "sequential": ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"],
      "diverging": ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"],
      "qualitative": ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628"]
    },
    "ui": {
      "primary": "#1a73e8",
      "text-primary": "#202124",
      "text-secondary": "#5f6368",
      "background": "#ffffff",
      "border": "#dadce0"
    }
  }
}
```

**Typography tokens:**

```json
{
  "typography": {
    "map": {
      "title": { "font": "Inter Bold", "size": "24px", "color": "{color.ui.text-primary}" },
      "city-large": { "font": "Noto Sans Bold", "size": "14px", "halo": "1.5px" },
      "city-medium": { "font": "Noto Sans Regular", "size": "12px", "halo": "1.2px" },
      "city-small": { "font": "Noto Sans Regular", "size": "10px", "halo": "1px" },
      "water": { "font": "Noto Sans Italic", "size": "11px", "color": "#4a7fa5" },
      "road": { "font": "Noto Sans Regular", "size": "10px", "halo": "2px" }
    }
  }
}
```

### 16.2 Style Libraries and Reusable Components

**Creating a modular style architecture:**

Rather than maintaining a monolithic `style.json`, decompose it into composable modules:

```
styles/
  tokens/
    colors.json
    typography.json
    sizing.json
  layers/
    basemap/
      water.json
      landcover.json
      transportation.json
      buildings.json
      labels.json
    thematic/
      choropleth.json
      graduated-circles.json
      heatmap.json
  themes/
    light.json
    dark.json
    satellite.json
  build.js          -- assembles modules into final style.json
```

**Build script pattern:**

```javascript
// build.js -- assemble a MapLibre style from modules
import { readFileSync, writeFileSync } from 'fs';

const colors = JSON.parse(readFileSync('./tokens/colors.json'));
const basemapWater = JSON.parse(readFileSync('./layers/basemap/water.json'));
const basemapTransport = JSON.parse(readFileSync('./layers/basemap/transportation.json'));

function resolveTokens(layer, tokens) {
  const str = JSON.stringify(layer);
  const resolved = str.replace(/\{color\.([^}]+)\}/g, (_, path) => {
    return path.split('.').reduce((obj, key) => obj[key], tokens.color);
  });
  return JSON.parse(resolved);
}

const style = {
  version: 8,
  name: 'Organization Standard Light',
  sources: { /* ... */ },
  glyphs: '...',
  sprite: '...',
  layers: [
    ...resolveTokens(basemapWater, { color: colors.color }),
    ...resolveTokens(basemapTransport, { color: colors.color }),
    // ... more layers
  ]
};

writeFileSync('./dist/style.json', JSON.stringify(style, null, 2));
```

### 16.3 Component-Based Cartography

Adopt a component mindset from UI development:

| Component | Contains | Variants |
|-----------|----------|----------|
| **Road network** | Casing, fill, bridge markers, tunnel dash, one-way arrows, labels | Light, dark, satellite overlay |
| **Water system** | Polygon fill, line stroke, labels (italic) | Light, dark |
| **Boundary set** | Admin levels 0-8, each with distinct dash patterns and weights | Light, dark, print |
| **Label system** | Place labels (cities through villages), feature labels, road labels | Latin, CJK, Arabic, multilingual |
| **Terrain** | Hillshade, contour lines, hypsometric tint | Subtle, dramatic, monochrome |
| **Point of Interest** | Icon, label, optional circle background | Tourism, navigation, municipal |

### 16.4 Version Control for Map Styles

Map styles should be version-controlled in Git with the same rigor as application code:

- **Diff-friendly format.** Format `style.json` with consistent 2-space indentation. Use a linter to enforce consistent property ordering.
- **Semantic versioning.** Use SemVer for style releases: MAJOR (breaking layer name changes), MINOR (new layers, new features), PATCH (color tweaks, bug fixes).
- **Visual regression testing.** Render map snapshots at key locations and zoom levels, then compare against baseline images using pixel-diff tools (Pixelmatch, BackstopJS).
- **CI/CD pipeline.** Validate style JSON against the MapLibre style spec on every commit. Deploy to a staging tile server for visual review before production.

```bash
# Validate a MapLibre style JSON
npx @maplibre/maplibre-gl-style-spec validate style.json
```

---

## 17. Cartographic Critique Checklist

A systematic evaluation framework ensures that every map meets professional quality standards before publication. Use this checklist during design reviews.

### 17.1 Purpose and Audience

- [ ] **Clear purpose statement.** Can you articulate in one sentence what this map communicates?
- [ ] **Audience identified.** Is the map designed for the intended audience's expertise level, cultural context, and use environment (print, screen, mobile, projection)?
- [ ] **Map type appropriate.** Is the chosen map type (choropleth, dot density, proportional symbol, etc.) the best fit for the data and message?
- [ ] **Title communicates the finding.** Does the title tell the reader what to look for, not just what the map shows? ("Income inequality has increased in southern regions" vs. "Map of Median Income")

### 17.2 Data Integrity

- [ ] **Classification method justified.** If classes are used, is the method (natural breaks, quantile, equal interval, manual) appropriate for the data distribution?
- [ ] **Normalization.** Are raw counts normalized where appropriate (per capita, per km2, percentage)?
- [ ] **Temporal validity.** Is the data date clearly stated? Are multiple years mixed without acknowledgment?
- [ ] **Source attribution.** Are all data sources credited?
- [ ] **Projection appropriate.** Does the projection suit the geographic extent and map purpose (see Section 13)?

### 17.3 Visual Design

- [ ] **Visual hierarchy.** Do the most important elements draw attention first?
- [ ] **Figure-ground.** Does the thematic data stand clearly apart from the basemap?
- [ ] **Color scheme type matches data type.** Sequential for ordered, diverging for midpoint, qualitative for categorical?
- [ ] **Color scheme is CVD-safe.** Tested with a color blindness simulator?
- [ ] **Contrast meets WCAG.** Labels and UI elements meet 4.5:1 contrast ratio?
- [ ] **Redundant encoding.** Is color the only differentiator, or is there a backup channel (pattern, shape, label)?
- [ ] **No rainbow palette.** Unless the data genuinely maps to spectral categories, avoid the perceptually non-uniform rainbow (jet) colormap.

### 17.4 Typography

- [ ] **Font conventions followed.** Water italic, cities sans-serif, physical features serif?
- [ ] **Label hierarchy clear.** Can the reader distinguish major labels from minor labels by size, weight, and color?
- [ ] **Labels legible.** Halos or panels used where backgrounds are complex? Minimum 10px for screen, 8pt for print?
- [ ] **No overlapping labels.** All text is readable without obstruction?
- [ ] **CJK/multilingual handled.** Correct font fallbacks, adequate sizing, appropriate script variants?

### 17.5 Map Furniture

- [ ] **Legend complete.** Every symbol, color, and line type on the map is explained in the legend?
- [ ] **Legend ordered.** Legend entries follow the same order as the data (high-to-low, or matching the spatial pattern)?
- [ ] **Scale bar present.** Appropriate for the projection; uses round numbers?
- [ ] **North arrow present (if needed).** Only when map is rotated or audience expects it?
- [ ] **Source and date.** Data provenance and temporal reference are stated?
- [ ] **Projection noted.** For print maps and academic publications, the CRS/projection is identified?

### 17.6 Technical Quality

- [ ] **Resolution appropriate.** 300 DPI for print, 72-96 for screen?
- [ ] **File size manageable.** Large vector PDFs simplified? Raster exports optimized?
- [ ] **Interactivity functional (web maps).** Popups, tooltips, zoom, pan all work correctly?
- [ ] **Mobile responsive.** Touch targets adequate? Controls usable on small screens?
- [ ] **Performance acceptable.** Map loads and renders within acceptable time budgets?

### 17.7 Ethical Considerations

- [ ] **No misleading distortion.** Projection does not exaggerate or diminish specific regions in a way that alters the message?
- [ ] **Boundaries handled sensitively.** Disputed territories acknowledged? Administrative boundaries current and defensible?
- [ ] **Privacy protected.** No personally identifiable information at small area levels? Geomasking applied where needed?
- [ ] **Inclusive language.** Labels and annotations use respectful, current terminology?
- [ ] **Disability access.** The map is usable by people with visual, motor, and cognitive disabilities?

---

## 18. AI-Assisted Cartography

Artificial intelligence and large language models (LLMs) are increasingly capable of assisting with cartographic design tasks -- from generating style specifications to suggesting color palettes to automating label placement.

> **Cross-reference:** For AI/ML applications in geospatial analysis (classification, object detection, prediction), see [AI & ML for Geospatial](../tools/ai-ml-geospatial.md) and [AI/ML Visualization](ai-ml-visualization.md). For prompt engineering patterns specific to GIS tasks, see [Map Styling Prompts](../ai-prompts/map-styling-prompts.md).

### 18.1 LLM-Assisted Style Generation

Modern LLMs (GPT-4, Claude, Gemini) can generate valid MapLibre GL style JSON, QGIS QML styles, and CartoCSS from natural language descriptions. This dramatically accelerates the prototyping phase.

**Effective prompting patterns:**

```
Prompt: "Generate a MapLibre GL style layer for a choropleth map of population
density. Use a 5-class sequential green palette from ColorBrewer (YlGn).
The data source is 'census' with source-layer 'tracts'. The density field
is 'pop_per_sqkm'. Use quantile breaks at 100, 500, 1000, 5000."

Prompt: "Create a QGIS QML style for a road network layer. Style highways
as 3px red lines, primary roads as 2px orange lines, secondary roads as
1.5px yellow lines, and all other roads as 0.5px gray lines. Use the
'highway' field for classification."
```

**What LLMs do well:**

- Generate syntactically correct style JSON for MapLibre/Mapbox GL.
- Apply color theory principles when asked (CVD-safe palettes, perceptual uniformity).
- Translate between styling formats (CartoCSS to GL JSON, SLD to QML).
- Write complex expressions (data-driven styling, multi-variable conditionals).
- Explain cartographic conventions and suggest appropriate design choices.

**What LLMs do poorly (current limitations):**

- **Visual judgment.** They cannot see the rendered output and may suggest combinations that look poor in practice. Always render and review.
- **Spatial context.** They do not know the geographic distribution of your data values, so class breaks may be inappropriate.
- **Pixel precision.** Fine-tuning icon alignment, label offsets, and kerning requires visual iteration, not language.
- **Hallucinated properties.** LLMs occasionally generate style properties that do not exist in the specification. Always validate against the spec.

### 18.2 Automated Label Placement

Label placement is a well-studied computational problem. AI approaches are now supplementing traditional algorithmic methods:

**Traditional algorithms:**

| Algorithm | Approach | Strength |
|-----------|----------|----------|
| **Greedy** | Place labels in priority order, first-fit | Fast, simple |
| **Simulated annealing** | Iteratively improve placement by random perturbation | Good global solutions, slow |
| **Genetic algorithm** | Evolve placement configurations over generations | Handles complex constraints |
| **Force-directed** | Labels repel each other like charged particles | Intuitive spacing |

**ML-enhanced approaches:**

- **Reinforcement learning for placement.** Train an agent to sequentially place labels, rewarding readability and penalizing overlaps. Research prototypes show improvement over greedy baselines.
- **Neural collision prediction.** Use a neural network to predict whether a candidate label position will collide with other elements, faster than geometric intersection tests at scale.
- **GAN-based generalization.** Generative adversarial networks can learn map generalization from examples, producing label selections and placements that mimic expert cartographers.

**Practical tools:**

| Tool | Type | Notes |
|------|------|-------|
| **MapLibre collision detection** | Rule-based (built-in) | Automatic overlap prevention with priority system |
| **QGIS PAL** | Rule-based (automated) | Automated label engine with numerous constraint parameters |
| **Mapnik text_placements** | Rule-based | Multiple candidate positions per feature |

### 18.3 AI Color Palette Suggestion

LLMs and specialized tools can suggest color palettes tailored to cartographic requirements:

**Workflow: LLM-assisted palette selection.**

1. Describe your data type (sequential, diverging, qualitative), the number of classes, the domain (environmental, health, urban), and any constraints (CVD-safe, print-safe, dark-mode).
2. Ask the LLM for 2-3 palette options with hex codes.
3. Test the suggested palettes in Viz Palette or Coblis for CVD safety.
4. Refine in Chroma.js or Oklch color space for perceptual uniformity.
5. Apply to the map and visually evaluate.

**Specialized AI palette tools:**

| Tool | Description |
|------|-------------|
| **Huemint** | AI-powered color palette generator for design systems |
| **Khroma** | Uses neural networks trained on your color preferences |
| **Palette.fm** | Generates color palettes from images (extract colors from reference maps) |
| **Adobe Firefly** | Generate color themes from text prompts |

### 18.4 AI for Map Content and Data Enhancement

Beyond styling, AI assists with several cartographic content tasks:

**Feature extraction from satellite imagery:**

- Building footprint extraction (Microsoft, Google, OpenStreetMap AI teams).
- Road network extraction from aerial photos.
- Land cover classification from multispectral imagery.
- See [Remote Sensing](../tools/remote-sensing.md) and [ML Training Data](../data-sources/ml-training-data.md).

**Natural language to map:**

- "Show me median household income by county in Texas" can be parsed by an LLM to generate a complete MapLibre GL style with data source, layer definition, and color ramp.
- This workflow powers emerging "conversational cartography" interfaces.

**Style transfer:**

- Apply the visual aesthetic of one map to another dataset. Research prototypes use neural style transfer adapted for cartographic conventions, preserving readability while matching the target aesthetic.
- Practical application: Generate a "vintage" or "blueprint" style variant of a modern basemap.

### 18.5 AI Cartography Ethical Considerations

- **Validation is mandatory.** AI-generated map styles must be reviewed by a human cartographer. Unchecked output may violate cartographic conventions, misrepresent data, or produce inaccessible designs.
- **Bias in training data.** LLMs trained primarily on English-language cartographic resources may suggest design conventions that are inappropriate for other cultural contexts.
- **Deepfake maps.** Generative AI can create convincing but entirely fabricated maps. Establish provenance and attribution practices for AI-assisted map production.
- **Skill preservation.** Use AI as a tool to accelerate expert cartographers, not as a replacement that degrades the profession's knowledge base.

### 18.6 Practical AI-Assisted Workflow

A recommended workflow for integrating AI tools into cartographic production:

```
1. BRIEF          -- Write a plain-language description of the map's
                     purpose, audience, data, and constraints.

2. GENERATE       -- Use an LLM to produce an initial style.json or
                     QML style from the brief.

3. VALIDATE       -- Run the output through a style spec validator.
                     Fix any hallucinated properties.

4. RENDER         -- Load the style in Maputnik, MapLibre, or QGIS
                     and visually inspect.

5. REFINE         -- Manually adjust colors, label sizes, line widths,
                     and zoom transitions. This step requires human
                     visual judgment.

6. TEST           -- Run accessibility tests (CVD simulation, contrast
                     check, keyboard navigation, screen reader).

7. REVIEW         -- Peer review using the Critique Checklist
                     (Section 17).

8. DEPLOY         -- Export for production (web, print, or both).
```

---

## 19. Further Reading & Cross-References

### Internal References (awesome-giser)

| Topic | Page |
|-------|------|
| 2D web mapping libraries (MapLibre, Leaflet, OpenLayers) | [js-bindbox/2d-mapping.md](../js-bindbox/2d-mapping.md) |
| 3D visualization (deck.gl, CesiumJS, Three.js) | [js-bindbox/3d-mapping.md](../js-bindbox/3d-mapping.md) |
| Tile server infrastructure | [js-bindbox/tile-servers.md](../js-bindbox/tile-servers.md) |
| Python geospatial libraries | [tools/python-libraries.md](../tools/python-libraries.md) |
| Desktop GIS (QGIS, ArcGIS) | [tools/desktop-gis.md](../tools/desktop-gis.md) |
| Thematic map types and classification | [visualization/thematic-maps.md](thematic-maps.md) |
| Temporal animation | [visualization/temporal-animation.md](temporal-animation.md) |
| 3D visualization | [visualization/3d-visualization.md](3d-visualization.md) |
| Scientific visualization | [visualization/scientific-visualization.md](scientific-visualization.md) |
| Storytelling and scrollytelling | [visualization/storytelling-scrollytelling.md](storytelling-scrollytelling.md) |
| AI/ML geospatial tools | [tools/ai-ml-geospatial.md](../tools/ai-ml-geospatial.md) |
| AI prompts for map styling | [ai-prompts/map-styling-prompts.md](../ai-prompts/map-styling-prompts.md) |
| Data sources index | [data-sources/README.md](../data-sources/README.md) |
| Cartography standards (academic) | [academic/cartography-standards.md](../academic/cartography-standards.md) |
| Performance optimization | [js-bindbox/performance-optimization.md](../js-bindbox/performance-optimization.md) |
| Ocean and maritime data | [data-sources/ocean-maritime.md](../data-sources/ocean-maritime.md) |
| Remote sensing tools | [tools/remote-sensing.md](../tools/remote-sensing.md) |

### External References

| Resource | Type | URL |
|----------|------|-----|
| *Cartography: Visualization of Geospatial Data* | Book | Kraak & Ormeling, 4th ed. |
| *Designing Better Maps* | Book | Cynthia Brewer, 2nd ed. |
| *Semiology of Graphics* | Book | Jacques Bertin (1967, translated) |
| *The Visual Display of Quantitative Information* | Book | Edward Tufte |
| *How to Lie with Maps* | Book | Mark Monmonier, 3rd ed. |
| *Cartography: A Compendium of Design Thinking* | Book | Swiss Society of Cartography |
| MapLibre Style Spec | Web | [maplibre.org/maplibre-style-spec](https://maplibre.org/maplibre-style-spec/) |
| Mapbox GL Style Spec | Web | [docs.mapbox.com/style-spec](https://docs.mapbox.com/style-spec/) |
| QGIS Documentation | Web | [docs.qgis.org](https://docs.qgis.org) |
| D3-Geo Projections | Web | [github.com/d3/d3-geo](https://github.com/d3/d3-geo) |
| Axis Maps Cartography Guide | Web | [axismaps.github.io/thematic-cartography](https://axismaps.github.io/thematic-cartography/) |
| Mapschool | Web | [mapschool.io](https://mapschool.io) |
| Cartographic Perspectives Journal | Journal | [cartographicperspectives.org](https://cartographicperspectives.org) |
| somethingaboutmaps (Daniel Huffman) | Blog | [somethingaboutmaps.wordpress.com](https://somethingaboutmaps.wordpress.com) |
| Andy Woodruff's blog | Blog | [andywoodruff.com](https://andywoodruff.com) |
| ICA (International Cartographic Association) | Org | [icaci.org](https://icaci.org) |
| NACIS | Org | [nacis.org](https://nacis.org) |
| Projection Wizard | Web | [projectionwizard.org](http://projectionwizard.org) |

---

[Back to Visualization](README.md) | [Back to Main README](../README.md)