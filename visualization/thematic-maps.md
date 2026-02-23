# Thematic Maps

> Techniques for representing spatial data through visual variables -- color, size, shape, and pattern.

> **Quick Picks**
> - **SOTA**: [Kepler.gl](https://kepler.gl) -- drag-and-drop thematic mapping with zero code
> - **Free Best**: [QGIS](https://qgis.org) -- full classification control, print-quality output
> - **Fastest Setup**: [Kepler.gl](https://kepler.gl) -- drop a CSV/GeoJSON, get an interactive map in seconds

## Choropleth Maps

Maps where areas are shaded in proportion to a statistical variable.

### Classification Methods

| Method | Description | Best For | Watch Out |
|--------|-------------|----------|-----------|
| **Equal Interval** | Divides range into equal-width classes | Evenly distributed data | Skewed data creates empty classes |
| **Quantile** | Each class has equal number of features | Emphasizing rank/distribution | Can split similar values into different classes |
| **Natural Breaks (Jenks)** | Minimizes within-class variance | Revealing natural groupings | Not comparable across datasets |
| **Standard Deviation** | Classes based on distance from mean | Showing deviation from average | Assumes roughly normal distribution |
| **Manual** | User-defined breakpoints | Domain-specific thresholds | Requires domain knowledge |

### Classification Method Comparison

When choosing a classification method, consider your data distribution:

- **Uniform data** -> Equal Interval works well
- **Skewed data** (common in population, income) -> Quantile or Natural Breaks
- **Data with known thresholds** (e.g., air quality index levels) -> Manual breaks
- **Comparing to a baseline** -> Standard Deviation

> **Rule of thumb:** Start with Natural Breaks (Jenks) for exploratory analysis. Switch to Quantile if you need equal visual weight per class. Use Manual when your domain has established thresholds.

### Color Schemes

- **Sequential:** Light-to-dark for ordered data (e.g., population density)
- **Diverging:** Two-hue scheme for data with a meaningful midpoint (e.g., temperature anomaly)
- **Qualitative:** Distinct hues for categorical data (e.g., land use types)

### Color Palette Resources

| Resource | URL | Type | Notes |
|----------|-----|------|-------|
| ColorBrewer 2.0 | [colorbrewer2.org](https://colorbrewer2.org) | Sequential, diverging, qualitative | The classic -- includes colorblind-safe flags |
| CARTOColors | [carto.com/carto-colors](https://carto.com/carto-colors) | Sequential, diverging, qualitative | Modern palettes designed for map visualization |
| Viridis | [bids.github.io/colormap](https://bids.github.io/colormap/) | Sequential (perceptually uniform) | Colorblind-safe, prints well in grayscale |
| d3-scale-chromatic | [d3js.org/d3-scale-chromatic](https://d3js.org/d3-scale-chromatic) | All types | JS-ready, includes all ColorBrewer schemes |
| Chroma.js | [gka.github.io/chroma.js](https://gka.github.io/chroma.js/) | Programmatic | Generate custom scales, interpolate colors |
| Viz Palette | [projects.susielu.com/viz-palette](https://projects.susielu.com/viz-palette) | Testing tool | Check palette distinctiveness and colorblind safety |

---

## Kepler.gl for Quick Thematic Mapping

[Kepler.gl](https://kepler.gl) is an open-source tool by Vis.gl for creating thematic maps without code:

- **Input:** Drop CSV, GeoJSON, or Arrow files directly into the browser
- **Map types:** Choropleth, point, cluster, hex bin, heat map, arc, line, grid, H3
- **Classification:** Built-in quantile, equal interval, and custom break support
- **Export:** Image, HTML (self-contained), JSON config
- **Sharing:** Save and share map configurations as JSON

**When to use Kepler.gl vs code:**
- Use Kepler.gl for exploration, stakeholder demos, and quick analysis
- Graduate to D3/MapLibre/deck.gl when you need custom interactivity, embedded dashboards, or production deployment

---

## Point & Cluster Maps

Represent discrete locations with symbols, optionally clustered at lower zoom levels.

- **Proportional symbols:** Circle size mapped to a variable
- **Categorical markers:** Icon or color by category
- **Clustering:** Aggregate dense points (Supercluster, Leaflet.markercluster)

### Clustering Configuration

```javascript
// MapLibre GL JS built-in clustering
map.addSource('earthquakes', {
  type: 'geojson',
  data: '/data/earthquakes.geojson',
  cluster: true,
  clusterMaxZoom: 14,
  clusterRadius: 50
});

map.addLayer({
  id: 'clusters',
  type: 'circle',
  source: 'earthquakes',
  filter: ['has', 'point_count'],
  paint: {
    'circle-color': [
      'step', ['get', 'point_count'],
      '#51bbd6', 100, '#f1f075', 750, '#f28cb1'
    ],
    'circle-radius': ['step', ['get', 'point_count'], 20, 100, 30, 750, 40]
  }
});
```

---

## Heat Maps

Continuous surface representation of point density or weighted values.

- **Kernel density estimation** for smooth surfaces
- **Configurable:** radius, intensity, color gradient
- **Libraries:** Leaflet.heat, MapLibre heatmap layer, deck.gl HeatmapLayer

### Heat Map Parameter Tuning

| Parameter | Effect | Typical Range |
|-----------|--------|---------------|
| Radius | Size of influence per point | 10-50 pixels |
| Intensity | Multiplier for density values | 0.1-5.0 |
| Weight | Per-point contribution | Property-driven or uniform |
| Max Zoom | Level at which full intensity applies | 15-18 |
| Color Ramp | Maps density to color gradient | Blue-Yellow-Red is common |

> **Tip:** Heat maps are best for showing density patterns. Do not use them for data that should be shown as exact values -- use proportional symbols instead.

---

## Flow Maps (OD Visualization)

Visualize movement, migration, or trade between origin-destination pairs.

- **Arc maps:** Curved lines between OD pairs
- **[Flowmap.blue](https://flowmap.blue):** Dedicated flow visualization tool (see Temporal Animation page)
- **deck.gl ArcLayer / LineLayer:** GPU-accelerated flow rendering

### Flow Map Data Format

```json
[
  { "origin": [116.4, 39.9], "dest": [121.5, 31.2], "count": 15000 },
  { "origin": [116.4, 39.9], "dest": [113.3, 23.1], "count": 8500 }
]
```

Key styling decisions:
- **Line width** proportional to flow volume
- **Color** for direction or category
- **Opacity** to handle overlapping flows
- **Curvature** to distinguish direction (clockwise = outgoing convention)

---

## Bivariate & Multivariate Maps

Show two or more variables simultaneously on a single map.

- **Bivariate choropleth:** 3x3 or 4x4 color grids combining two variables
- **Dot density:** Multiple colored dots for different categories
- **Small multiples:** Side-by-side maps for comparison

### Bivariate Legend Construction

A 3x3 bivariate choropleth uses a grid of 9 colors, blending two sequential schemes:

```
           Variable B (low -> high)
            Low     Med     High
V   High | #574249 #627F8C #64ACBE
a   Med  | #985356 #AD9EA5 #B0D5DF
r   Low  | #C85A5A #E4ACAC #E8E8E8
A
```

Steps:
1. Classify each variable into 3 classes (tertiles/quantiles)
2. Assign each combination a color from the 3x3 grid
3. Include the 2D legend on the map -- readers need it to decode

> **Resource:** Joshua Stevens' guide to bivariate choropleth maps is the definitive reference.

---

## Tool Recommendations

| Map Type | Recommended Tools |
|----------|------------------|
| Choropleth | QGIS, D3.js, Mapbox Studio, Kepler.gl |
| Point & Cluster | Leaflet + Supercluster, MapLibre, deck.gl |
| Heat Map | Leaflet.heat, MapLibre, deck.gl HeatmapLayer |
| Flow Map | Flowmap.blue, deck.gl ArcLayer, D3.js |
| Bivariate | QGIS (manual), D3.js, Observable Plot |

---

## Accessibility Guidelines for Color

- Always test with a colorblind simulator ([Coblis](https://www.color-blindness.com/coblis-color-blindness-simulator/), Sim Daltonism)
- Use ColorBrewer's "colorblind safe" filter when choosing palettes
- Provide redundant encoding: pattern + color, or size + color
- For sequential schemes, Viridis family (viridis, plasma, inferno, magma) is perceptually uniform and colorblind-safe
- Maximum 7 classes for choropleth maps (5 is often better)
- Include numeric values in tooltips and legends, not just color
