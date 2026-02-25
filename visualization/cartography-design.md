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

## Further Reading

| Resource | Type | URL |
|----------|------|-----|
| *Cartography: Visualization of Geospatial Data* | Book | Kraak & Ormeling, 4th ed. |
| *Designing Better Maps* | Book | Cynthia Brewer, 2nd ed. |
| *Semiology of Graphics* | Book | Jacques Bertin (1967, translated) |
| *The Visual Display of Quantitative Information* | Book | Edward Tufte |
| *How to Lie with Maps* | Book | Mark Monmonier, 3rd ed. |
| Axis Maps Cartography Guide | Web | [axismaps.github.io/thematic-cartography](https://axismaps.github.io/thematic-cartography/) |
| Mapschool | Web | [mapschool.io](https://mapschool.io) |
| Cartographic Perspectives Journal | Journal | [cartographicperspectives.org](https://cartographicperspectives.org) |
| somethingaboutmaps (Daniel Huffman) | Blog | [somethingaboutmaps.wordpress.com](https://somethingaboutmaps.wordpress.com) |
| Andy Woodruff's blog | Blog | [andywoodruff.com](https://andywoodruff.com) |

---

[Back to Visualization](README.md) | [Back to Main README](../README.md)