# Cartography & Design

> Map design principles, typography, color theory, and styling tools for creating clear, accessible, and beautiful maps.

> **Quick Picks**
> - **SOTA**: [Maputnik](https://maputnik.github.io) -- open-source visual style editor for MapLibre/Mapbox GL styles
> - **Free Best**: [QGIS](https://qgis.org) -- full cartographic control with print layout composer
> - **Fastest Setup**: [Mapbox Studio](https://studio.mapbox.com) -- visual style editor with instant preview

## Map Design Principles

Fundamental guidelines for effective cartographic communication.

### Visual Hierarchy

Establish clear importance levels so the reader's eye moves from primary to secondary information.

1. **Figure-ground:** Separate the subject from the background
2. **Contrast:** Use size, color, and weight to differentiate importance
3. **Balance:** Distribute visual weight across the map layout
4. **Simplification:** Remove unnecessary detail; show only what serves the purpose

### Key Principles

| Principle | Description |
|-----------|-------------|
| Purpose-driven | Every design decision should serve the map's communication goal |
| Data-ink ratio | Maximize the share of ink used to present data (Tufte) |
| Generalization | Simplify geometry and content appropriate to scale |
| Consistency | Maintain uniform styling for similar features |

### Design Resources

| Resource | URL | Description |
|----------|-----|-------------|
| **Cartography Guide** (Axis Maps) | [axismaps.github.io/thematic-cartography](https://axismaps.github.io/thematic-cartography/) | Comprehensive guide to thematic map design -- classification, color, labeling |
| Mapschool | [mapschool.io](https://mapschool.io) | Free introduction to geo concepts and cartography |
| Geospatial Design Patterns | [mapstyle.withgoogle.com](https://mapstyle.withgoogle.com) | Google's map styling reference |
| TypeBrewer | [typebrewer.org](http://typebrewer.org) | Typography guidelines for cartographic labeling |

---

## Maputnik -- SOTA Open Source Style Editor

[Maputnik](https://maputnik.github.io) is a free, open-source visual editor for MapLibre GL / Mapbox GL style specifications.

- **Website:** [maputnik.github.io](https://maputnik.github.io)
- **Repository:** [github.com/maputnik/editor](https://github.com/maputnik/editor)
- **Key Strengths:** Browser-based, no account needed, exports standard GL JSON styles, works with any vector tile source
- **Best For:** Creating and editing vector tile styles for MapLibre, Mapbox, or any GL-compatible renderer

### Maputnik Workflow

1. **Open** [maputnik.github.io/editor](https://maputnik.github.io/editor) in your browser
2. **Choose a base style** (OSM Liberty, Positron, Dark Matter, etc.) or start blank
3. **Connect your tile source** -- add a vector tile URL (PMTiles, Martin, MapTiler, etc.)
4. **Edit layers visually:**
   - Click any layer to modify paint/layout properties
   - Use the filter editor for data-driven visibility
   - Drag layers to reorder z-index
5. **Export:** Download the style as JSON -- use directly with MapLibre GL JS

```javascript
// Use a Maputnik-exported style in MapLibre
const map = new maplibregl.Map({
  container: 'map',
  style: '/styles/my-custom-style.json',  // exported from Maputnik
  center: [116.4, 39.9],
  zoom: 10
});
```

> **Tip:** Maputnik can load styles from a URL, making it easy to iterate on styles served by Martin or TileServer GL.

---

## Typography for Maps

Selecting and placing text on maps for readability and aesthetics.

### Guidelines

- **Font choice:** Sans-serif for digital maps; serif for physical/natural features (classic convention)
- **Size hierarchy:** Title > subtitle > labels > annotations
- **Halos/buffers:** Add text halos for legibility over complex backgrounds
- **Placement rules:** Water labels in italic, city labels in upright, area labels spaced/curved

### Recommended Fonts

| Font | Type | Best For |
|------|------|----------|
| Noto Sans | Sans-serif | Multilingual map labels |
| Source Sans Pro | Sans-serif | Clean UI and web maps |
| Open Sans | Sans-serif | General purpose |
| Lora | Serif | Natural features, print maps |
| Inter | Sans-serif | Modern UI, dashboards |

### Font Pairing Examples

| Map Type | Title Font | Label Font | Notes |
|----------|-----------|------------|-------|
| Modern web map | Inter Bold | Noto Sans Regular | Clean, works at all sizes |
| Classic/editorial | Playfair Display | Source Sans Pro | Elegant, good for print |
| Technical/scientific | Source Sans Pro Bold | Source Sans Pro Light | Minimal, data-focused |
| Multilingual | Noto Sans Bold | Noto Sans Regular | Covers 1000+ languages |

---

## Color Theory & Accessibility

Using color effectively and ensuring maps are accessible to all users.

### Color Considerations

- **Sequential schemes** for ordered data (light to dark)
- **Diverging schemes** for data with a meaningful midpoint
- **Qualitative schemes** for categorical data (max 8-10 distinct hues)
- **Avoid** red-green only distinctions (8% of males are red-green color blind)

### Accessibility Checklist

- [ ] Test with color blindness simulator (e.g., [Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/), Sim Daltonism)
- [ ] Ensure sufficient contrast ratio (WCAG AA minimum: 4.5:1 for text)
- [ ] Provide redundant encoding (pattern + color, size + color)
- [ ] Include alt text for web maps
- [ ] Support keyboard navigation for interactive maps
- [ ] Test at different screen sizes and zoom levels

### Color Accessibility Testing Workflow

1. **Design** your color scheme using ColorBrewer (enable "colorblind safe" filter)
2. **Preview** the map normally
3. **Simulate** with [Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/) -- upload a screenshot and check protanopia, deuteranopia, tritanopia
4. **Verify** contrast ratios with [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/)
5. **Add redundancy:** If any classes are indistinguishable under simulation, add patterns, labels, or size variation

| Resource | Purpose |
|----------|---------|
| [ColorBrewer](https://colorbrewer2.org) | Cartographic color schemes with colorblind-safe options |
| [Viz Palette](https://projects.susielu.com/viz-palette) | Check palette distinctiveness and colorblind safety |
| [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/) | Verify text/background contrast ratios |
| [Coblis Simulator](https://www.color-blindness.com/coblis-color-blindness-simulator/) | Simulate all types of color vision deficiency |

---

## Legend & Layout Best Practices

Designing effective map legends, titles, scale bars, and overall layout.

### Legend Design

- Place legend where it does not obscure important map content
- Order legend entries logically (highest to lowest, or categorical grouping)
- Use clear, concise labels; avoid jargon
- Match legend symbols exactly to map symbols

### Layout Elements

| Element | Purpose | Placement |
|---------|---------|-----------|
| Title | Communicate the map's subject | Top center or top left |
| Legend | Decode symbols and colors | Bottom right or floating panel |
| Scale bar | Indicate distance | Bottom left |
| North arrow | Indicate orientation | Top right (only if north is not up) |
| Source / attribution | Credit data providers | Bottom edge |
| Inset map | Provide location context | Corner |

### Layout Composition Tips

- **Web maps:** Use floating panels and collapsible legends to maximize map area
- **Print maps:** Use a clear grid-based layout; QGIS Print Layout is excellent for this
- **Dashboards:** Place map as the primary element with charts and filters around it
- **Storytelling:** Use scrollytelling to reveal map layers progressively

---

## Style Resources

### Map Styles

| Resource | Description |
|----------|-------------|
| [OpenMapTiles styles](https://openmaptiles.org/styles/) | Free basemap styles for vector tiles |
| [Mapbox Gallery](https://www.mapbox.com/gallery) | Community-contributed Mapbox styles |
| [Stamen designs](https://maps.stamen.com/) | Artistic map styles (Toner, Watercolor, Terrain) -- now via Stadia Maps |
| [Positron / Dark Matter](https://carto.com/basemaps/) | CARTO basemap styles optimized for data visualization |
| [OSM Liberty](https://maputnik.github.io/osm-liberty/) | Free MapLibre-compatible style with a clean aesthetic |

### Icon Sets

| Resource | Icons | License |
|----------|-------|---------|
| [Maki](https://labs.mapbox.com/maki-icons/) | Map-oriented POI icons | CC0 |
| [Temaki](https://ideditor.github.io/temaki/docs/) | Extended Maki set (500+ icons) | CC0 |
| Font Awesome | General purpose | Free + Pro |
| [Material Symbols](https://fonts.google.com/icons) | Google's icon set | Apache 2.0 |

### SVG Icon Integration in MapLibre

```javascript
// Load custom SVG icon
const img = await map.loadImage('/icons/hospital.png');
map.addImage('hospital-icon', img.data);

map.addLayer({
  id: 'hospitals',
  type: 'symbol',
  source: 'pois',
  filter: ['==', 'type', 'hospital'],
  layout: {
    'icon-image': 'hospital-icon',
    'icon-size': 0.5,
    'text-field': ['get', 'name'],
    'text-offset': [0, 1.2],
    'text-size': 12,
    'text-font': ['Noto Sans Regular']
  }
});
```

---

## Tools

### QGIS Styling

- Symbol layers, rule-based rendering, data-defined properties
- Print layout composer for publication-quality maps
- Style manager for saving and sharing styles (.qml / .sld)
- **Workflow:** Design style in QGIS -> Export as SLD -> Publish via GeoServer

### Mapbox Studio

- Visual style editor for Mapbox GL styles
- Custom fonts, icons, and sprite sheets
- Data-driven styling with expressions
- **Workflow:** Edit in Studio -> Copy style JSON -> Use with MapLibre (remove token requirement)

### Maputnik

- Open-source visual editor for MapLibre/Mapbox GL styles
- Browser-based, no account required
- Export styles as JSON for any GL-based renderer
- **Workflow:** Open Maputnik -> Connect tile source -> Edit layers -> Export JSON -> Deploy

### Tool Comparison

| Tool | Open Source | Web-Based | Output Format | Vector Tile Styling | Print Layout |
|------|-----------|-----------|---------------|--------------------|----|
| Maputnik | Yes | Yes | GL Style JSON | Yes | No |
| Mapbox Studio | No | Yes | GL Style JSON | Yes | No |
| QGIS | Yes | No (desktop) | QML, SLD, GL JSON (plugin) | Limited | Yes |
| ArcGIS Pro | No | No (desktop) | lyrx, VTPK | Yes | Yes |

### Dark Mode Map Styling

Dark basemaps are increasingly popular for data visualization dashboards. Key considerations:

- **Base color:** Use very dark blue-gray (#1a1a2e) rather than pure black
- **Labels:** Light gray (#c0c0c0) on dark background; increase halo opacity
- **Data layers:** Use saturated, bright colors that stand out against dark backgrounds
- **Ready-made dark styles:** CARTO Dark Matter, Mapbox Dark, MapTiler Dark
