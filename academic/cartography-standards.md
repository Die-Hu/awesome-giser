# Cartography Standards for Publication

Publication-quality maps require attention to cartographic conventions, visual hierarchy, and accessibility. This guide covers essential map elements, CRS selection, color standards, thematic map types, accessibility requirements, journal-specific figure requirements, interactive supplements, and standards organizations.

> **Quick Picks**
> - **Color palettes:** ColorBrewer (colorbrewer2.org) — the gold standard
> - **Colorblind-safe:** Viridis (sequential) or Okabe-Ito (qualitative)
> - **CRS for local studies:** UTM zone for your area
> - **CRS for web maps:** EPSG:3857 (display only, never for analysis)
> - **Figure format:** Vector PDF/EPS preferred; raster at 300+ DPI
> - **Quick check:** Coblis colorblind simulator (https://www.color-blindness.com/coblis-color-blindness-simulator/)

---

## Map Elements Checklist

Every publication map should include these elements. Check before submitting figures.

| Element | Required | Notes |
|---------|----------|-------|
| Title / Caption | Yes | Descriptive, referenced from text body |
| Legend | Yes | Explains all symbols, colors, line styles |
| Scale Bar | Yes | Appropriate units (km, m, mi); avoid rep fractions alone |
| North Arrow | Conditional | Required unless orientation is obvious or CRS axes labeled |
| Data Source Attribution | Yes | Credit data providers |
| Coordinate Reference System | Yes | EPSG code or CRS name in caption or margin |
| Inset / Overview Map | Recommended | Shows study area in broader context |
| Neat Line / Border | Optional | Frames the map extent |
| Graticule / Grid | Optional | Helpful for large-extent maps |
| Projection Note | Conditional | Required where projection affects interpretation |
| Date of Data | Recommended | When was the data collected/acquired? |
| Classification Method | Recommended | How were classes/breaks determined? |

### Common Mistakes in Publication Maps

| Mistake | Impact | Fix |
|---------|--------|-----|
| Missing scale bar | Reader cannot assess distances | Always include |
| Scale bar in degrees | Meaningless for distance | Use projected CRS units |
| Missing CRS info | Cannot reproduce or compare | State EPSG code |
| Legend with hex codes only | Unreadable for humans | Use descriptive labels |
| Too many legend classes | Visual overload | 5-7 classes for choropleth |
| North arrow on web Mercator | North varies across map | Omit or use graticule |
| Tiny text in figures | Unreadable at print size | 8 pt minimum |
| Rainbow colormap | Not perceptually uniform, not colorblind-safe | Use Viridis or ColorBrewer |
| Missing basemap attribution | Copyright violation | Credit OpenStreetMap/other providers |
| Inconsistent symbology across figures | Confusing comparison | Use same colors/classes |

---

## Coordinate Reference Systems Guide

### Common CRS Choices

| CRS | EPSG | Type | Use Case |
|-----|------|------|----------|
| WGS 84 | 4326 | Geographic | Global datasets, GPS data, data exchange |
| WGS 84 / Pseudo-Mercator | 3857 | Projected | Web mapping (Google, OSM, Mapbox) |
| UTM Zones | 326XX / 327XX | Projected | Regional studies, accurate distance/area |
| NAD83 | 4269 | Geographic | North American datasets |
| ETRS89 | 4258 | Geographic | European datasets |
| CGCS2000 | 4490 | Geographic | Chinese datasets (国家2000坐标系) |
| Albers Equal Area | Various | Projected | Area-accurate thematic maps |
| Lambert Conformal Conic | Various | Projected | Mid-latitude regional maps |
| Mollweide | 54009 | Projected | Global thematic maps (equal area) |
| Robinson | 54030 | Projected | Global reference maps (compromise) |

### CRS Selection Flowchart

```
What is your analysis scope?
  │
  GLOBAL ──→ Storage: WGS 84 (4326)
  │          Display: Equal-area (Mollweide) or compromise (Robinson)
  │          Analysis: Appropriate equal-area projection
  │
  CONTINENTAL ──→ Continental projection
  │               Europe: ETRS89-LAEA (3035)
  │               N. America: NAD83 Albers (5070)
  │               Asia: Asia Lambert (102025)
  │
  NATIONAL/REGIONAL ──→ National grid or UTM zone
  │                     UK: British National Grid (27700)
  │                     China: CGCS2000 / Gauss-Krüger
  │                     US: State Plane or UTM
  │
  LOCAL (< 100 km) ──→ UTM zone for your area
                       Note: All UTM zones are 6° wide

Are you measuring areas?
  YES ──→ Must use equal-area projection (Albers, Mollweide, LAEA)
  NO  ──→ Conformal (angles) or compromise is fine

Are you measuring distances/directions?
  YES ──→ Use equidistant or conformal projection
  NO  ──→ Any appropriate projection

Is this for web display?
  YES ──→ Store data in 4326, display in 3857 (automatic in most libraries)
  NO  ──→ Use the most appropriate projected CRS for your analysis
```

### CRS Pitfalls

| Pitfall | Consequence | Prevention |
|---------|-------------|------------|
| Area calculations in 4326 (degrees) | Wrong results (up to 40% error) | Always project first |
| Mixing CRS between layers | Spatial misalignment | Reproject all to common CRS |
| Using Web Mercator for analysis | Severe area distortion at high latitudes | Only for display |
| Wrong UTM zone | Distance/area errors | Check zone at https://mangomap.com/utm |
| Assuming planar geometry in geographic CRS | Wrong buffer/distance results | Use geodesic calculations or project |
| Forgetting to state CRS in paper | Not reproducible | Always document in methods + captions |

### UTM Zone Quick Reference

| Region | UTM Zone(s) | EPSG (North) |
|--------|-------------|-------------|
| Western US (California) | 10-11 | 32610-32611 |
| Eastern US (New York) | 17-18 | 32617-32618 |
| UK | 30 | 32630 |
| Central Europe | 32-33 | 32632-32633 |
| Eastern China | 50-51 | 32650-32651 |
| Japan | 53-54 | 32653-32654 |
| Australia (east) | 55-56 | 32755-32756 (South) |
| Brazil (São Paulo) | 23 | 32723 (South) |

---

## Scale and Generalization

### Scale Selection Guide

| Map Type | Scale Range | Resolution | Use Case |
|----------|------------|------------|----------|
| Global overview | 1:50M – 1:10M | Country-level | Climate, global land cover |
| Continental | 1:10M – 1:1M | Region-level | Continental patterns |
| Regional | 1:1M – 1:100K | City/county | Regional analysis, transport |
| Local | 1:100K – 1:10K | Neighborhood | Urban studies, land use |
| Site-level | 1:10K – 1:1K | Building/parcel | Site planning, surveying |

### Generalization Principles

| Principle | Description | When to Apply |
|-----------|-------------|---------------|
| Selection | Remove less important features | Reducing clutter at small scales |
| Simplification | Reduce vertices, preserve shape | Complex coastlines, boundaries |
| Aggregation | Merge small features into groups | Many small polygons at overview scale |
| Displacement | Shift overlapping features | Roads near rivers, overlapping points |
| Typification | Representative symbols for patterns | Dense building footprints |
| Exaggeration | Enlarge small but important features | Narrow roads on regional maps |

### Scale-Dependent Feature Visibility

```
Zoom Level / Scale        Features Visible
─────────────────────────────────────────
1:50M (global)           Countries, oceans, major rivers
1:10M (continental)      States/provinces, large cities, major roads
1:1M (regional)          Counties, all cities, highways, large water bodies
1:100K (local)           Districts, streets, buildings (aggregated), parks
1:10K (site)             Individual buildings, parcels, street details
1:1K (detailed)          Building interiors, utility lines, vegetation
```

---

## Color Standards for Thematic Maps

### Color Scheme Types

| Scheme | Data Type | Example | Palettes |
|--------|-----------|---------|----------|
| Sequential | Low → High | Population density, elevation | YlOrRd, Blues, Viridis |
| Diverging | Midpoint-centered | Temperature anomaly, change | RdBu, BrBG, PiYG |
| Qualitative | Categories | Land use, soil types | Set2, Paired, Tab10 |
| Binary | Two classes | Urban/rural, yes/no | Simple two-color |

### Color Palette Recommendations

| Context | Recommended Palette | Source | Notes |
|---------|-------------------|--------|-------|
| Sequential (general) | Viridis | matplotlib | Perceptually uniform, colorblind-safe |
| Sequential (temperature) | Inferno, Magma | matplotlib | High contrast |
| Sequential (blue) | Blues | ColorBrewer | Water, precipitation |
| Diverging (temp. anomaly) | RdBu | ColorBrewer | Red = warm, blue = cool |
| Diverging (change) | PiYG, BrBG | ColorBrewer | Clear midpoint |
| Qualitative (≤8 classes) | Set2, Okabe-Ito | ColorBrewer / CUD | Colorblind-safe |
| Qualitative (>8 classes) | Tab20 | matplotlib | May need pattern overlay |
| Elevation / terrain | terrain, dem | cpt-city | Conventional (green-brown-white) |
| Vegetation indices | RdYlGn | ColorBrewer | Red=low, green=high NDVI |
| Bathymetry | deep | cmocean | Ocean depth convention |

### Color Resources

| Resource | URL | Purpose |
|----------|-----|---------|
| ColorBrewer | https://colorbrewer2.org | Cartographic color schemes |
| Viridis (matplotlib) | Built-in | Perceptually uniform sequential |
| CARTOColors | https://carto.com/carto-colors | Web-optimized map colors |
| cmocean | https://matplotlib.org/cmocean/ | Oceanographic color maps |
| Scientific Colour Maps | https://www.fabiocrameri.ch/colourmaps/ | Perceptually uniform, published |
| Coolors | https://coolors.co | General palette generator |
| Color Oracle | https://colororacle.org | Desktop colorblind simulator |
| Coblis | https://www.color-blindness.com/coblis/ | Online colorblind simulator |

### Number of Classes

| Classes | Best For | Classification Methods |
|---------|----------|----------------------|
| 3-5 | Simple message, clear patterns | Equal interval, quantile |
| 5-7 | Detailed choropleth, most publications | Natural breaks (Jenks), quantile |
| 7-9 | Maximum for perception | Natural breaks, manual |
| >9 | Avoid for choropleth | Use continuous gradient instead |

### Classification Methods

| Method | When to Use | Implementation |
|--------|-------------|----------------|
| Equal Interval | Evenly distributed data | `(max - min) / n_classes` |
| Quantile | Equal count per class | `numpy.percentile()` or GIS tool |
| Natural Breaks (Jenks) | Clustered data | `mapclassify.NaturalBreaks()` |
| Standard Deviation | Normally distributed | Mean ± 1σ, ± 2σ |
| Manual | Domain knowledge | Expert-defined thresholds |
| Head/Tail Breaks | Heavy-tailed distributions | `mapclassify.HeadTailBreaks()` |

---

## Accessibility Guidelines

### Colorblind-Safe Design

Approximately 8% of men and 0.5% of women have color vision deficiency.

**Rules:**
1. Never use red-green as the sole differentiator
2. Use ColorBrewer palettes marked as "colorblind safe"
3. Combine color with pattern, texture, or labels (redundant encoding)
4. Test with a colorblind simulator before submitting

### Recommended Colorblind-Safe Palettes

| Palette | Type | Colors | Source |
|---------|------|--------|--------|
| Viridis | Sequential | Purple → Yellow | matplotlib |
| Cividis | Sequential | Blue → Yellow | matplotlib |
| Okabe-Ito | Qualitative | 8 distinct | Color Universal Design |
| ColorBrewer (CB-safe) | Various | Marked on site | colorbrewer2.org |
| Batlow | Sequential | Blue → Red | Scientific Colour Maps |
| Crameri | Various | 30+ maps | Scientific Colour Maps |

### WCAG Compliance for Digital Maps

| Level | Contrast Ratio | Apply To |
|-------|---------------|----------|
| AA (minimum) | 4.5:1 (text), 3:1 (graphics) | All publication figures |
| AAA (enhanced) | 7:1 (text), 4.5:1 (graphics) | Web maps, interactive |

### Accessibility Checklist

- [ ] Colorblind-safe palette used
- [ ] Tested with Color Oracle or Coblis
- [ ] Sufficient contrast between adjacent colors (WCAG AA)
- [ ] Labels and annotations supplement color coding
- [ ] Alt-text descriptions for maps in digital publications
- [ ] Grayscale-readable if journal may print in B&W
- [ ] Font size ≥ 8pt in all figure elements
- [ ] Pattern/texture fills available for categorical maps
- [ ] Interactive maps have keyboard navigation (if web)

---

## Thematic Map Types

### When to Use Each Map Type

| Map Type | Data | Best For | Example |
|----------|------|----------|---------|
| Choropleth | Ratio/rate by area | Population density, income per capita | Census tract median income |
| Graduated Symbol | Count/magnitude at points | City population, earthquake magnitude | City GDP circles |
| Dot Density | Count distributed in area | Population distribution | 1 dot = 1,000 people |
| Isoline/Contour | Continuous surface | Temperature, elevation, pressure | DEM contour lines |
| Heat Map | Point density | Crime hotspots, tweet locations | Kernel density estimation |
| Flow Map | Movement between locations | Migration, trade, commuting | OD flow lines |
| Cartogram | Area proportional to data | Election results, GDP | Country area ∝ population |
| Bivariate | Two variables simultaneously | Income × education | 3×3 color grid |
| Small Multiples | Same area, different times/variables | Land use change over decades | 4-panel time series |
| 3D/Perspective | Elevation + attribute | Urban density, terrain + land use | Extruded building heights |

### Choropleth Best Practices

```
✅ DO:
- Normalize data (rates, percentages, per capita)
- Use appropriate class breaks (Jenks for clustered data)
- Include a clear legend with class ranges
- Use sequential colors for ordered data
- Consider area bias (large rural areas dominate)

❌ DON'T:
- Map raw counts (larger areas = larger counts)
- Use too many classes (5-7 is optimal)
- Use rainbow colormap
- Forget to state the classification method
- Use choropleth for point data
```

### Bivariate Map Color Scheme

```
           Variable A (Low → High)
           ┌─────┬─────┬─────┐
    High   │ ■   │ ■   │ ■   │  (dark purple → dark blue)
           ├─────┼─────┼─────┤
    Med    │ ■   │ ■   │ ■   │  (light purple → medium blue)
           ├─────┼─────┼─────┤
    Low    │ ■   │ ■   │ ■   │  (light pink → light blue)
           └─────┴─────┴─────┘
 Variable B

Use Joshua Stevens' bivariate color schemes:
https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/
```

---

## Journal-Specific Figure Requirements

| Journal | Max Width | Resolution | Format | Color |
|---------|-----------|------------|--------|-------|
| Elsevier (RSE, C&G) | 190 mm (2-col) | 300 DPI | PDF, EPS, TIFF | Free online, charge print |
| T&F (IJGIS, CaGIS) | 170 mm (2-col) | 300 DPI | EPS, TIFF | Free online, charge print |
| Springer (Geoinformatica) | 174 mm (2-col) | 300 DPI | PDF, EPS, TIFF | Free |
| IEEE (TGRS) | 184 mm (2-col) | 300 DPI | PDF, EPS, TIFF | 1st page free, then charge |
| MDPI (Remote Sensing) | No strict max | 300 DPI | PDF, PNG, TIFF | Free (OA) |
| Wiley (Trans. in GIS) | 170 mm (2-col) | 300 DPI | EPS, TIFF, PDF | Free online, charge print |
| ISPRS J-PARS | 190 mm (2-col) | 300 DPI | PDF, EPS, TIFF | Free online, charge print |
| Nature Geoscience | 180 mm (2-col) | 300 DPI | PDF, EPS, AI | Free |
| Copernicus (ESSD, GMD) | 180 mm (2-col) | 300 DPI | PDF, PNG, EPS | Free (OA) |

### Column Width Quick Reference

| Width Type | Typical Size | Use For |
|-----------|-------------|---------|
| Single column | 80-90 mm (3.15-3.54 in) | Simple diagrams, small maps |
| 1.5 column | 130-140 mm (5.12-5.51 in) | Medium maps, charts |
| Double column | 170-190 mm (6.69-7.48 in) | Large maps, multi-panel figures |
| Full page | Up to 250 mm height | Very detailed maps |

### Export Settings for Common Tools

```python
# matplotlib — Publication quality
fig.savefig('fig1.pdf', format='pdf', bbox_inches='tight',
            pad_inches=0.02, transparent=False)
fig.savefig('fig1.tiff', format='tiff', dpi=300, bbox_inches='tight',
            pil_kwargs={'compression': 'tiff_lzw'})

# QGIS — Export as PDF
# Layout > Export as PDF > 300 DPI, embed fonts, simplify geometries OFF

# R/ggplot2 — Publication quality
ggsave("fig1.pdf", width = 190, height = 120, units = "mm", dpi = 300)
ggsave("fig1.tiff", width = 190, height = 120, units = "mm", dpi = 300,
       compression = "lzw")
```

---

## Interactive Figure Supplements

Many journals now accept interactive supplements alongside static figures.

### Options for Interactive Supplements

| Tool | Output | Hosting | Best For |
|------|--------|---------|----------|
| Plotly | HTML widget | GitHub Pages, journal supp. | Charts, 3D scatter |
| Folium/Leaflet | HTML map | GitHub Pages | Interactive web maps |
| deck.gl | HTML map | GitHub Pages | Large-scale 3D viz |
| Observable | Notebook | observablehq.com | Explorable analysis |
| Binder | Jupyter | mybinder.org | Reproducible notebooks |
| Streamlit | Web app | Streamlit Cloud | Data exploration apps |

### Example: Folium Interactive Supplement

```python
import folium
import geopandas as gpd

gdf = gpd.read_file("results.gpkg")
m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

folium.Choropleth(
    geo_data=gdf,
    data=gdf,
    columns=['id', 'temperature'],
    key_on='feature.properties.id',
    fill_color='RdYlBu_r',
    legend_name='LST (°C)'
).add_to(m)

# Add popup with details
folium.GeoJson(
    gdf,
    tooltip=folium.GeoJsonTooltip(fields=['name', 'temperature', 'ndvi'])
).add_to(m)

m.save('interactive_map_supplement.html')
# Host on GitHub Pages and link from paper
```

---

## Standards Organizations

| Organization | Full Name | Relevance |
|-------------|-----------|-----------|
| ICA | International Cartographic Association | Cartographic standards, map design |
| OGC | Open Geospatial Consortium | Geospatial data standards |
| ISO TC 211 | ISO Technical Committee 211 | Geographic information standards |
| FGDC | Federal Geographic Data Committee (US) | Metadata standards, SDI |
| INSPIRE | Infrastructure for Spatial Info in EU | European spatial data standards |
| UN-GGIM | UN Global Geospatial Information Mgmt | Global standards coordination |
| W3C | World Wide Web Consortium | Web map accessibility (WCAG) |

### Key Standards Documents

| Standard | Scope | URL |
|----------|-------|-----|
| ISO 19115 | Geographic information — Metadata | https://www.iso.org/standard/53798.html |
| ISO 19157 | Geographic information — Data quality | https://www.iso.org/standard/32575.html |
| ISO 19110 | Feature cataloguing methodology | https://www.iso.org/standard/57303.html |
| ICA Map Design | Best practices for map design | https://mapdesign.icaci.org/ |
| FGDC Metadata | US metadata standard | https://www.fgdc.gov/metadata |
| WCAG 2.2 | Web content accessibility | https://www.w3.org/WAI/WCAG22/quickref/ |
| OGC SLD | Styled Layer Descriptor (map styling) | https://www.ogc.org/standard/sld/ |
| OGC SE | Symbology Encoding | https://www.ogc.org/standard/se/ |

---

## Map Design Software

### Publication Map Tools

| Tool | Type | Cost | Best For |
|------|------|------|----------|
| QGIS | Desktop GIS | Free | Full-featured print layouts |
| ArcGIS Pro | Desktop GIS | Licensed | Professional cartography |
| Adobe Illustrator | Vector editor | $23/mo | Post-processing GIS exports |
| Inkscape | Vector editor | Free | Free alternative to Illustrator |
| Mapbox Studio | Web | Free tier | Basemap styling |
| MapLibre | Web | Free | Custom web map design |
| Felt | Web | Free tier | Quick collaborative maps |
| Datawrapper | Web | Free tier | Locator maps, choropleths |
| kepler.gl | Web | Free | Large-scale exploration |

### QGIS Print Layout Tips

```
1. Use project-level variables for consistency:
   @project_crs → "EPSG:32632"
   @data_source → "OpenStreetMap contributors"

2. Atlas generation for multi-area maps:
   Layout → Atlas → Generate → Coverage layer

3. Export settings:
   - PDF: Embed fonts, simplify OFF, 300 DPI raster layers
   - SVG: For Illustrator/Inkscape post-processing
   - PNG/TIFF: 300 DPI, world file for georeferencing

4. Consistent styling across figures:
   Save styles as .qml files and reuse across layouts
```

### Post-Processing Workflow (QGIS → Illustrator/Inkscape)

```
1. Export from QGIS as SVG
2. Open in Illustrator/Inkscape
3. Adjust text styles (font, size, placement)
4. Add annotations, callouts, inset maps
5. Fine-tune colors and line weights
6. Export as PDF (vector) or TIFF (300 DPI)
7. Verify colorblind safety with Color Oracle
```
