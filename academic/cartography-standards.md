# Cartography Standards

Publication-quality maps require attention to cartographic conventions, visual hierarchy, and accessibility. This guide covers essential map elements, coordinate reference systems, color standards, and journal-specific requirements.

> **Quick Picks**
> - **Color palettes:** ColorBrewer (colorbrewer2.org) -- the gold standard
> - **Colorblind-safe:** Viridis (sequential) or Okabe-Ito (qualitative)
> - **CRS for local studies:** UTM zone for your area
> - **CRS for web maps:** EPSG:3857 (display only, never for analysis)
> - **Figure format:** Vector PDF/EPS preferred; raster at 300+ DPI

---

## Map Elements Checklist

Every publication map should include the following elements. Use this checklist before submitting figures.

| Element | Required | Notes |
|---------|----------|-------|
| Title / Caption | Yes | Descriptive, referenced from text body |
| Legend | Yes | Explains all symbols, colors, line styles |
| Scale Bar | Yes | Use appropriate units (km, m, mi); avoid representative fractions alone |
| North Arrow | Conditional | Required unless orientation is obvious or CRS axes are labeled |
| Data Source Attribution | Yes | Credit data providers (e.g., "Data: OpenStreetMap contributors") |
| Coordinate Reference System | Yes | State EPSG code or CRS name in caption or map margin |
| Inset / Overview Map | Recommended | Shows study area location in broader context |
| Neat Line / Border | Optional | Frames the map extent |
| Graticule / Grid | Optional | Helpful for large-extent maps; label coordinates |
| Projection Note | Conditional | Required for maps where projection affects interpretation |


---

## Coordinate Reference Systems Guide

### Common CRS Choices

| CRS | EPSG | Type | Use Case |
|-----|------|------|----------|
| WGS 84 | 4326 | Geographic | Global datasets, GPS data, web maps |
| WGS 84 / Pseudo-Mercator | 3857 | Projected | Web mapping (Google, OSM, Mapbox) |
| UTM Zones | 326XX / 327XX | Projected | Regional studies, accurate distance/area |
| NAD83 | 4269 | Geographic | North American datasets |
| ETRS89 | 4258 | Geographic | European datasets |
| Equal Area projections | Various | Projected | Thematic maps showing area comparisons |

### CRS Selection Guidelines

- **Use UTM** for local/regional studies requiring accurate measurements
- **Use WGS 84 (4326)** for global datasets or when interoperability is key
- **Use equal-area projections** for choropleth maps or area comparisons
- **Avoid Web Mercator (3857)** for printed maps (extreme area distortion at high latitudes)
- **Always state the CRS** in your methodology section and figure caption

### CRS Pitfalls

- Mixing CRS between layers causes misalignment
- Area calculations in geographic CRS (degrees) are meaningless
- Some projections distort distance, area, or shape -- choose based on your analysis needs

### CRS Decision Flowchart

```
What is your analysis scope?
  GLOBAL --> WGS 84 (4326) for storage; Equal Area for display
  CONTINENTAL --> Continental projection (e.g., LAEA for Europe)
  NATIONAL/REGIONAL --> UTM zone or national grid
  LOCAL (< 100 km) --> UTM zone

Are you measuring areas or distances?
  YES --> Use a projected CRS (UTM, national grid, equal-area)
  NO  --> Geographic CRS (4326) is fine for display

Is this for web display?
  YES --> Data in 4326, display in 3857 (automatic in most libraries)
  NO  --> Use the most appropriate projected CRS
```

---

## Scale and Generalization Standards

### Scale Selection

| Map Type | Typical Scale Range | Resolution |
|----------|-------------------|------------|
| Global overview | 1:50,000,000 - 1:10,000,000 | Country-level |
| Continental | 1:10,000,000 - 1:1,000,000 | Region-level |
| Regional | 1:1,000,000 - 1:100,000 | City/county-level |
| Local | 1:100,000 - 1:10,000 | Neighborhood-level |
| Site-level | 1:10,000 - 1:1,000 | Building/parcel-level |

### Generalization Principles

- **Selection**: Remove less important features at smaller scales
- **Simplification**: Reduce vertex count while preserving shape character
- **Aggregation**: Merge small features into larger groups
- **Displacement**: Shift overlapping features for readability
- **Typification**: Replace detailed patterns with representative symbols


---

## Color Standards for Thematic Maps

### Sequential Color Schemes

For data that goes from low to high (e.g., population density, elevation).

- Use a single hue with varying lightness
- Recommended palettes: ColorBrewer YlOrRd, Blues, Viridis

### Diverging Color Schemes

For data with a meaningful midpoint (e.g., temperature anomaly, change detection).

- Two hues diverging from a neutral midpoint
- Recommended palettes: ColorBrewer RdBu, BrBG, PiYG

### Qualitative Color Schemes

For categorical data (e.g., land use classes, soil types).

- Distinct hues with similar saturation/lightness
- Recommended palettes: ColorBrewer Set2, Paired, Tab10

### Color Resources

| Resource | URL | Purpose |
|----------|-----|---------|
| ColorBrewer | https://colorbrewer2.org | Cartographic color schemes |
| Viridis | Built into matplotlib, R | Perceptually uniform sequential |
| CARTOColors | https://carto.com/carto-colors | Web-optimized map colors |
| Coolors | https://coolors.co | General color palette generator |


---

## Accessibility Guidelines

### Colorblind-Safe Palettes

Approximately 8% of men and 0.5% of women have some form of color vision deficiency. Maps should be readable by colorblind users.

**Rules:**

- Avoid red-green combinations as the sole differentiator
- Use ColorBrewer palettes marked as "colorblind safe"
- Combine color with pattern, texture, or labels for redundant encoding
- Test maps with a colorblind simulator (e.g., Coblis, Color Oracle)

### Recommended Colorblind-Safe Palettes

| Palette | Type | Source |
|---------|------|--------|
| Viridis | Sequential | matplotlib |
| Cividis | Sequential | matplotlib |
| Okabe-Ito | Qualitative | Color Universal Design |
| ColorBrewer (CB-safe) | Various | colorbrewer2.org |

### Additional Accessibility Considerations

- Ensure sufficient contrast between adjacent colors (WCAG AA minimum)
- Use labels and annotations, not just color, to convey information
- Provide alternative text descriptions for maps in digital publications
- Consider grayscale readability for publications that may be printed in B&W


---

## Journal-Specific Figure Requirements

| Journal | Max Width | Resolution | Format | Color Charge |
|---------|-----------|------------|--------|-------------|
| Elsevier (RSE, C&G) | 190 mm (2-col) | 300 DPI min | PDF, EPS, TIFF | Free (online), charge (print) |
| Taylor & Francis (IJGIS) | 170 mm (2-col) | 300 DPI min | EPS, TIFF | Free (online), charge (print) |
| Springer (Geoinformatica) | 174 mm (2-col) | 300 DPI min | PDF, EPS, TIFF | Free |
| IEEE (TGRS) | 184 mm (2-col) | 300 DPI min | PDF, EPS, TIFF | First page free, then charge |
| MDPI (Remote Sensing) | No strict max | 300 DPI min | PDF, PNG, TIFF | Free (open access) |
| Wiley (Transactions in GIS) | 170 mm (2-col) | 300 DPI min | EPS, TIFF, PDF | Free (online), charge (print) |
| ISPRS J-PARS | 190 mm (2-col) | 300 DPI min | PDF, EPS, TIFF | Free (online), charge (print) |
| Nature Geoscience | 180 mm (2-col) | 300 DPI min | PDF, EPS, AI | Free |

### Journal-Specific Figure Notes

**IJGIS (Taylor & Francis):**
- Single column: 84 mm, double column: 170 mm
- Font in figures: 8-10 pt, sans-serif recommended
- Figures submitted as separate files, not embedded in manuscript
- Color figures free online; print color may incur charges

**Computers & Geosciences (Elsevier):**
- Single column: 90 mm, 1.5 column: 140 mm, double column: 190 mm
- Vector formats strongly preferred (PDF, EPS)
- Avoid thin lines (< 0.5 pt) that may not reproduce well
- Provide alt-text descriptions for accessibility

**Remote Sensing of Environment (Elsevier):**
- Same as C&G format requirements
- Maps must include complete cartographic elements
- Supplementary figures can be in lower resolution

**IEEE TGRS:**
- Single column: 88 mm, double column: 184 mm
- Color figures in first page free; additional pages charged per figure
- Recommend using IEEE color palette for consistency
- Figures must be readable when printed in grayscale

---

## Standards Organizations

| Organization | Full Name | Relevance |
|-------------|-----------|-----------|
| ICA | International Cartographic Association | Cartographic standards, map design |
| OGC | Open Geospatial Consortium | Geospatial data standards, interoperability |
| ISO TC 211 | ISO Technical Committee 211 | Geographic information standards (ISO 19100 series) |
| FGDC | Federal Geographic Data Committee (US) | Metadata standards, spatial data infrastructure |
| INSPIRE | Infrastructure for Spatial Information in the EU | European spatial data standards |

### Key Standards Documents

- **ISO 19115**: Geographic information -- Metadata
- **ISO 19157**: Geographic information -- Data quality
- **ICA Commission on Map Design**: Best practices for map design
- **FGDC Content Standard for Digital Geospatial Metadata**: US metadata standard

**Links:**
- [ISO 19115 (Metadata)](https://www.iso.org/standard/53798.html)
- [ISO 19157 (Data Quality)](https://www.iso.org/standard/32575.html)
- [ICA Commission on Map Design](https://mapdesign.icaci.org/)
- [FGDC Metadata Standard](https://www.fgdc.gov/metadata)
