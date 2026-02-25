# Thematic Maps

> Techniques for representing spatial data through visual variables -- color, size, shape, texture, orientation, and pattern. This guide covers classification theory, SOTA tooling, color science, and production-ready code for every major thematic map type.

> **Quick Picks**
> - **SOTA Interactive**: [Kepler.gl](https://kepler.gl) -- drag-and-drop thematic mapping with zero code
> - **SOTA Programmatic**: [Observable Plot](https://observablehq.com/plot/) + [D3.js](https://d3js.org) -- maximum control, publication quality
> - **Free Desktop**: [QGIS](https://qgis.org) -- full classification control, print-quality output
> - **GPU Performance**: [deck.gl](https://deck.gl) -- millions of features at 60 fps
> - **Fastest Setup**: [Kepler.gl](https://kepler.gl) -- drop a CSV/GeoJSON, get an interactive map in seconds

**Cross-references:**
[2D Mapping Libraries](../js-bindbox/2d-mapping.md) |
[Charting Integration](../js-bindbox/charting-integration.md) |
[Python Libraries](../tools/python-libraries.md) |
[Desktop GIS](../tools/desktop-gis.md) |
[Data Sources](../data-sources/)

---

## Table of Contents

1. [Choropleth Maps](#choropleth-maps)
2. [Heat Maps and Density Surfaces](#heat-maps-and-density-surfaces)
3. [Flow Maps and OD Visualization](#flow-maps-and-od-visualization)
4. [Bivariate and Multivariate Maps](#bivariate-and-multivariate-maps)
5. [Proportional Symbols and Graduated Maps](#proportional-symbols-and-graduated-maps)
6. [Point and Cluster Maps](#point-and-cluster-maps)
7. [Statistical Maps](#statistical-maps)
8. [Specialized Thematic Maps](#specialized-thematic-maps)
9. [Color Science](#color-science)
10. [Tool Comparison Matrix](#tool-comparison-matrix)

---

## Choropleth Maps

Maps where areas are shaded in proportion to a statistical variable. The single most common thematic map type, and the one most frequently done poorly.

### Classification Methods Deep-Dive

| Method | Algorithm | Best For | Pitfalls | Classes |
|--------|-----------|----------|----------|---------|
| **Equal Interval** | `(max - min) / k` | Evenly distributed data, intuitive legends | Skewed data creates empty classes; most real-world data is skewed | 3-7 |
| **Quantile** | Each class holds `n/k` features | Emphasizing rank; guaranteeing visual variety | Splits identical values across classes; misleading gaps | 3-7 |
| **Natural Breaks (Jenks)** | Minimizes within-class variance (GVF) | Revealing natural groupings in data | Not comparable across datasets; computationally expensive for large n | 3-7 |
| **Fisher-Jenks** | Exact optimization variant of Jenks | Same as Jenks with guaranteed optimality | Even slower than Jenks; use `jenkspy` Python library for speed | 3-7 |
| **Head/Tail Breaks** | Recursive split around the mean | Heavy-tailed distributions (city populations, wealth) | Produces unequal class counts by design; requires right-skewed data | Variable |
| **Standard Deviation** | Classes at intervals of sigma from mean | Showing deviation from average | Assumes approximate normality; misleading otherwise | 4-6 |
| **Pretty Breaks** | Rounds breakpoints to "nice" numbers | Public-facing maps with readable legends | Aesthetic, not analytical; may obscure real patterns | 3-7 |
| **Geometric Interval** | Geometric progression between breaks | Data spanning several orders of magnitude | Difficult to explain to non-technical audiences | 3-7 |
| **Manual / Domain** | User-defined breakpoints | Domain-specific thresholds (AQI, BMI, poverty lines) | Requires domain knowledge; subjective | Any |

#### Choosing a Method: Decision Tree

```
Is your data categorical?
  YES -> Use qualitative colors, not classification
  NO  -> Is there a domain standard for breakpoints?
           YES -> Manual breaks
           NO  -> Is data heavily right-skewed?
                    YES -> Head/Tail Breaks or Geometric Interval
                    NO  -> Is data approximately normal?
                             YES -> Standard Deviation
                             NO  -> Do you need cross-dataset comparison?
                                      YES -> Equal Interval or Manual
                                      NO  -> Start with Jenks, verify with Quantile
```

#### Goodness of Variance Fit (GVF)

The GVF statistic measures how well your classification captures the structure of your data. A GVF of 1.0 means perfect classification; below 0.8 is typically poor.

```python
# Python: Compare classification methods using mapclassify
import mapclassify
import geopandas as gpd

gdf = gpd.read_file("census_tracts.geojson")
values = gdf["median_income"].values

schemes = {
    "Quantile":       mapclassify.Quantiles(values, k=5),
    "Jenks":          mapclassify.NaturalBreaks(values, k=5),
    "Fisher-Jenks":   mapclassify.FisherJenks(values, k=5),
    "Equal Interval": mapclassify.EqualInterval(values, k=5),
    "Head/Tail":      mapclassify.HeadTailBreaks(values),
}

for name, scheme in schemes.items():
    adcm = scheme.adcm  # absolute deviation around class medians
    print(f"{name:20s}  classes={scheme.k}  ADCM={adcm:,.0f}")
```

### Normalization: The Most Common Choropleth Mistake

Raw counts on choropleth maps are almost always misleading because larger areas or more populous areas dominate visually. Always normalize.

| Normalization | Formula | Use Case |
|---------------|---------|----------|
| **Per capita** | `value / population` | Crime, disease, economic indicators |
| **Per area** | `value / area_km2` | Density of anything |
| **Rate** | `(events / at_risk_population) * multiplier` | Epidemiology, demographics |
| **Percentage** | `(part / whole) * 100` | Vote share, land use proportion |
| **Z-score** | `(x - mean) / std` | Comparing across disparate scales |
| **Percentile rank** | `rank(x) / n * 100` | Relative standing |
| **Index** | `x / baseline * 100` | Change over time, cost of living |

> **Dark Arts Tip:** When a stakeholder insists on mapping raw counts, add a second map panel showing the normalized version. Let them see the difference. If they still want counts, use proportional symbols instead of choropleth -- at least the visual variable (size) does not imply area-based density.

### Small Multiples for Temporal Comparison

Small multiples are a powerful alternative to animation for showing change over time or across categories.

```javascript
// Observable Plot: small multiple choropleth
import * as Plot from "@observablehq/plot";

Plot.plot({
  facet: { data: censusData, x: "year" },
  projection: { type: "albers-usa" },
  marks: [
    Plot.geo(censusData, {
      fill: "unemployment_rate",
      stroke: "#fff",
      strokeWidth: 0.5,
    }),
  ],
  color: {
    scheme: "YlOrRd",
    type: "quantile",
    n: 5,
    label: "Unemployment rate (%)",
  },
  width: 900,
  height: 200,
})
```

### Code: D3.js Choropleth with Jenks Breaks

```javascript
import * as d3 from "d3";
import * as topojson from "topojson-client";

const width = 960, height = 600;
const svg = d3.select("#map").append("svg")
  .attr("viewBox", [0, 0, width, height]);

// Load data
const [us, data] = await Promise.all([
  d3.json("counties-albers-10m.json"),
  d3.csv("unemployment.csv", d3.autoType),
]);

const counties = topojson.feature(us, us.objects.counties);
const valueMap = new Map(data.map(d => [d.fips, d.rate]));

// Jenks breaks via simple-statistics
import { ckmeans } from "simple-statistics";
const values = data.map(d => d.rate);
const clusters = ckmeans(values, 5);
const breaks = clusters.map(c => c[0]);

const color = d3.scaleThreshold()
  .domain(breaks.slice(1))
  .range(d3.schemeYlOrRd[5]);

const path = d3.geoPath();

svg.selectAll("path")
  .data(counties.features)
  .join("path")
    .attr("d", path)
    .attr("fill", d => {
      const v = valueMap.get(d.id);
      return v != null ? color(v) : "#ccc";
    })
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.3);

// Legend
const legend = svg.append("g")
  .attr("transform", `translate(600,20)`);

// (add swatches and labels for each break)
```

### Code: Python Choropleth (geopandas + matplotlib)

```python
import geopandas as gpd
import matplotlib.pyplot as plt
import mapclassify

gdf = gpd.read_file("us_counties.geojson")
gdf["rate"] = gdf["unemployed"] / gdf["labor_force"] * 100

scheme = mapclassify.FisherJenks(gdf["rate"], k=5)
gdf["class"] = scheme.yb

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
gdf.plot(
    column="rate",
    scheme="FisherJenks",
    k=5,
    cmap="YlOrRd",
    legend=True,
    legend_kwds={"title": "Unemployment Rate (%)", "fmt": "{:.1f}"},
    edgecolor="#ffffff",
    linewidth=0.3,
    ax=ax,
    missing_kwds={"color": "lightgrey", "label": "No data"},
)
ax.set_axis_off()
ax.set_title("County Unemployment Rate, 2024", fontsize=16, pad=20)
plt.tight_layout()
plt.savefig("choropleth.png", dpi=300, bbox_inches="tight")
```

### Code: MapLibre Data-Driven Styling

```javascript
map.addSource("counties", {
  type: "vector",
  url: "pmtiles://counties.pmtiles",
});

map.addLayer({
  id: "county-fill",
  type: "fill",
  source: "counties",
  "source-layer": "counties",
  paint: {
    "fill-color": [
      "interpolate", ["linear"],
      ["get", "unemployment_rate"],
      2, "#ffffcc",
      4, "#fed976",
      6, "#fd8d3c",
      8, "#e31a1c",
      12, "#800026"
    ],
    "fill-opacity": 0.85,
    "fill-outline-color": "#ffffff"
  }
});
```

### SOTA Tools for Choropleth

| Tool | Strength | Classification Support | Output |
|------|----------|----------------------|--------|
| **D3.js** | Full control, SVG/Canvas | Manual via `d3-scale` + `simple-statistics` | Web |
| **Observable Plot** | Declarative, fast prototyping | Built-in `quantile`, `quantize`, `threshold` | Web |
| **Vega-Lite** | Grammar of graphics, concise spec | `quantile`, `quantize`, threshold | Web |
| **QGIS** | Print layouts, 20+ methods | All methods built-in, histogram preview | Print/Image |
| **Kepler.gl** | Zero code, GPU | Quantile, equal interval, custom | Web/Image |
| **MapLibre** | Vector tiles, fast render | Expressions (`step`, `interpolate`) | Web |
| **Python (geopandas)** | Data science pipeline | Via `mapclassify` (15+ methods) | Image/PDF |
| **R (tmap)** | Publication quality | `style` param: jenks, fisher, headtails, etc. | Image/PDF |

---

## Heat Maps and Density Surfaces

Continuous surface representations of point density or weighted values. Heat maps reveal spatial patterns that are invisible in raw point clouds.

### Kernel Density Estimation (KDE) Parameters

The KDE transforms discrete points into a continuous density surface. Two critical parameters control the result.

#### Bandwidth (h) Selection

The bandwidth is the most important parameter. Too small and the surface is spiky/noisy; too large and real patterns are smoothed away.

| Method | Description | When to Use |
|--------|-------------|-------------|
| **Silverman's rule of thumb** | `h = 0.9 * min(std, IQR/1.34) * n^(-1/5)` | Quick default for unimodal data |
| **Scott's rule** | `h = 1.06 * std * n^(-1/5)` | Gaussian-like data |
| **Cross-validation (LSCV)** | Minimizes integrated squared error | Optimal but computationally expensive |
| **Manual / domain** | Set by analyst | Known spatial process scale (e.g., 1 km for urban crime) |

#### Kernel Types

| Kernel | Shape | Notes |
|--------|-------|-------|
| **Gaussian** | Bell curve, infinite extent | Most common; theoretically sound |
| **Epanechnikov** | Parabolic, finite extent | Statistically optimal (min MISE); slightly faster |
| **Quartic (biweight)** | Smooth, finite extent | Good balance of smoothness and compactness |
| **Triangular** | Linear decay | Fast to compute; less smooth |
| **Uniform** | Flat within radius | Simple counting; not recommended for visualization |

> **Dark Arts Tip:** Most web mapping libraries use a simplified kernel (often triangular or Gaussian approximation) with pixel-based radius rather than geographic distance. For analytically sound KDE, compute the surface server-side (Python `scipy.stats.gaussian_kde`, R `spatstat::density.ppp`) and serve as a raster tile layer.

### Hexbin Aggregation

Hexagonal binning avoids the visual bias of square grids (which create artificial N-S-E-W patterns) and provides consistent adjacency (each hex has exactly 6 neighbors).

#### H3: Uber's Hierarchical Hexagonal Index

```javascript
// deck.gl H3HexagonLayer with h3-js
import { H3HexagonLayer } from "@deck.gl/geo-layers";
import { cellToLatLng, latLngToCell } from "h3-js";

// Aggregate points into H3 cells at resolution 7 (~1.2 km edge)
const resolution = 7;
const hexCounts = new Map();
points.forEach(([lng, lat]) => {
  const cell = latLngToCell(lat, lng, resolution);
  hexCounts.set(cell, (hexCounts.get(cell) || 0) + 1);
});

const hexLayer = new H3HexagonLayer({
  id: "h3-hexagons",
  data: Array.from(hexCounts, ([hex, count]) => ({ hex, count })),
  getHexagon: d => d.hex,
  getFillColor: d => colorScale(d.count),
  getElevation: d => d.count,
  elevationScale: 20,
  extruded: true,
  pickable: true,
});
```

#### deck.gl HexagonLayer (On-the-Fly Binning)

```javascript
import { HexagonLayer } from "@deck.gl/aggregation-layers";

const hexLayer = new HexagonLayer({
  id: "hex-bin",
  data: points,
  getPosition: d => [d.lng, d.lat],
  radius: 1000,           // meters
  elevationRange: [0, 3000],
  elevationScale: 4,
  extruded: true,
  colorRange: [
    [255, 255, 178], [254, 204, 92], [253, 141, 60],
    [240, 59, 32], [189, 0, 38]
  ],
  coverage: 0.9,
  upperPercentile: 95,    // clip outliers
});
```

### Contour Density Maps

Contour (isoline) representations of density provide a cleaner alternative to pixel-based heat maps.

```javascript
// D3.js: contour density from points
import { contourDensity } from "d3-contour";
import { geoPath, geoAlbersUsa } from "d3-geo";

const projection = geoAlbersUsa().fitSize([width, height], statesMesh);

const densityContours = contourDensity()
  .x(d => projection([d.lng, d.lat])[0])
  .y(d => projection([d.lng, d.lat])[1])
  .size([width, height])
  .bandwidth(15)
  .thresholds(12)
  (points);

svg.selectAll("path.contour")
  .data(densityContours)
  .join("path")
    .attr("class", "contour")
    .attr("d", geoPath())
    .attr("fill", d => colorScale(d.value))
    .attr("stroke", "#333")
    .attr("stroke-width", 0.3)
    .attr("fill-opacity", 0.6);
```

### MapLibre GL JS Heatmap Layer

```javascript
map.addLayer({
  id: "crime-heat",
  type: "heatmap",
  source: "crime-points",
  maxzoom: 15,
  paint: {
    // Weight by severity
    "heatmap-weight": [
      "interpolate", ["linear"], ["get", "severity"],
      0, 0,  5, 1
    ],
    // Increase intensity with zoom
    "heatmap-intensity": [
      "interpolate", ["linear"], ["zoom"],
      0, 1,  15, 3
    ],
    // Increase radius with zoom
    "heatmap-radius": [
      "interpolate", ["linear"], ["zoom"],
      0, 2,  15, 30
    ],
    // Color ramp
    "heatmap-color": [
      "interpolate", ["linear"], ["heatmap-density"],
      0,    "rgba(33,102,172,0)",
      0.2,  "rgb(103,169,207)",
      0.4,  "rgb(209,229,240)",
      0.6,  "rgb(253,219,199)",
      0.8,  "rgb(239,138,98)",
      1,    "rgb(178,24,43)"
    ],
    // Fade out as you zoom in (show points instead)
    "heatmap-opacity": [
      "interpolate", ["linear"], ["zoom"],
      12, 1,  15, 0
    ]
  }
});
```

### SOTA Tools for Density

| Tool | Technique | GPU | Notes |
|------|-----------|-----|-------|
| **deck.gl HeatmapLayer** | WebGL density | Yes | Weighted, zoom-adaptive |
| **deck.gl HexagonLayer** | Hex binning | Yes | Aggregation + 3D extrusion |
| **MapLibre heatmap** | Built-in layer type | Yes | Expression-driven, zoom-adaptive |
| **Kepler.gl** | Hex, grid, heatmap, H3 | Yes | Zero-code; multiple aggregation modes |
| **d3-contour** | KDE contours, SVG | No | Publication quality; small datasets |
| **Python scipy** | True KDE, raster output | No | Analytically correct; serve as tiles |
| **R spatstat** | Advanced KDE | No | Edge correction, adaptive bandwidth |

---

## Flow Maps and OD Visualization

Visualize movement, migration, trade, commuting, or any origin-destination relationship.

### Arc Maps and Desire Lines

The simplest flow representation: a line or arc connecting each origin to each destination, with width or color encoding magnitude.

```javascript
// deck.gl ArcLayer for migration flows
import { ArcLayer } from "@deck.gl/layers";

const arcLayer = new ArcLayer({
  id: "migration-arcs",
  data: flows,
  getSourcePosition: d => d.origin,
  getTargetPosition: d => d.destination,
  getSourceColor: [0, 128, 255],
  getTargetColor: [255, 0, 128],
  getWidth: d => Math.sqrt(d.count) * 0.5,
  getHeight: 0.5,            // arc curvature
  greatCircle: true,          // geodesic arcs for long distances
  pickable: true,
  autoHighlight: true,
});
```

### Edge Bundling for Dense Networks

When you have hundreds or thousands of OD pairs, individual arcs create visual clutter ("hairball" problem). Edge bundling groups nearby flows into shared corridors.

**Algorithms:**

| Algorithm | Complexity | Quality | Notes |
|-----------|------------|---------|-------|
| **Force-directed (FDEB)** | O(n^2 * iterations) | High | Holten & van Wijk, 2009; best visual quality |
| **Kernel density (KDEEB)** | O(n * m) | High | Hurter et al., 2012; image-based, fast |
| **Divided edge bundling** | O(n^2) | Medium | Separates bidirectional flows |
| **Hierarchical** | O(n log n) | Medium | Uses spatial hierarchy; fast for large n |

```python
# Python: force-directed edge bundling
# pip install datashader
import datashader as ds
import datashader.transfer_functions as tf
from datashader.bundling import connect_edges, hammer_bundle

# nodes: DataFrame with x, y columns
# edges: DataFrame with source, target columns referencing node indices
bundled = hammer_bundle(nodes, edges)
cvs = ds.Canvas(plot_width=1200, plot_height=800)
agg = cvs.line(bundled, "x", "y", agg=ds.count())
img = tf.shade(agg, cmap=["#333333", "#4488ff", "#ffffff"], how="log")
```

### Sankey Geography (Flow Sankey Maps)

Combine Sankey diagram logic with geographic positioning. Flows are drawn as ribbons whose width represents magnitude, routed along geographic paths.

### Flowmap.blue

[Flowmap.blue](https://flowmap.blue) is a dedicated open-source tool for OD visualization:

- Input: Google Sheets or CSV with location coordinates and flow data
- Automatic clustering of origins/destinations at different zoom levels
- Animated particle trails showing direction
- Built on deck.gl; handles millions of flows

### OD Matrix Visualization

When geographic layout is secondary, an OD matrix (heatmap table) can reveal patterns that flow maps obscure.

```javascript
// Vega-Lite OD matrix
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": { "url": "od_flows.csv" },
  "mark": "rect",
  "encoding": {
    "x": { "field": "origin", "type": "nominal", "sort": "ascending" },
    "y": { "field": "destination", "type": "nominal", "sort": "ascending" },
    "color": {
      "field": "flow_count",
      "type": "quantitative",
      "scale": { "type": "symlog", "scheme": "blues" }
    }
  }
}
```

### D3 Force-Directed Flow Layout

For network-oriented flow visualization where geographic accuracy is less important than relationship clarity:

```javascript
const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(edges).id(d => d.id).distance(d => 100 / Math.log(d.flow + 1)))
  .force("charge", d3.forceManyBody().strength(-200))
  .force("center", d3.forceCenter(width / 2, height / 2))
  .force("x", d3.forceX().x(d => projection([d.lng, d.lat])[0]).strength(0.3))
  .force("y", d3.forceY().y(d => projection([d.lng, d.lat])[1]).strength(0.3));
```

The `strength` parameter on the geographic forces controls the balance between network layout clarity and geographic fidelity.

---

## Bivariate and Multivariate Maps

Show two or more variables simultaneously on a single map. These are advanced techniques that require careful legend design.

### Bivariate Choropleth

A 3x3 or 4x4 color grid combining two sequential color schemes. Each polygon is assigned a color based on its class in both variables.

#### 3x3 Grid (Recommended Starting Point)

```
           Variable B (low -> high)
            Low     Med     High
V   High | #574249 #627F8C #64ACBE
a   Med  | #985356 #AD9EA5 #B0D5DF
r   Low  | #C85A5A #E4ACAC #E8E8E8
A
```

#### 4x4 Grid (More Detail, Harder to Read)

Use only when both variables have sufficient variation and your audience is technically literate.

#### Implementation: D3.js Bivariate Choropleth

```javascript
// Define the 3x3 color scheme
const bivarColors = [
  ["#e8e8e8", "#b0d5df", "#64acbe"],  // Row: low A
  ["#e4acac", "#ad9ea5", "#627f8c"],  // Row: mid A
  ["#c85a5a", "#985356", "#574249"],  // Row: high A
];

// Classify both variables into tertiles
const classA = d3.scaleQuantile().domain(dataA).range([0, 1, 2]);
const classB = d3.scaleQuantile().domain(dataB).range([0, 1, 2]);

// Color function
function bivarColor(d) {
  const a = classA(d.varA);
  const b = classB(d.varB);
  return bivarColors[a][b];
}

svg.selectAll("path")
  .data(features)
  .join("path")
    .attr("d", path)
    .attr("fill", bivarColor);

// CRITICAL: Build the 2D legend
const legendSize = 100;
const legendSvg = d3.select("#legend").append("svg")
  .attr("width", legendSize + 60).attr("height", legendSize + 60);

for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
    legendSvg.append("rect")
      .attr("x", j * (legendSize / 3) + 30)
      .attr("y", (2 - i) * (legendSize / 3) + 10)
      .attr("width", legendSize / 3)
      .attr("height", legendSize / 3)
      .attr("fill", bivarColors[i][j]);
  }
}
```

> **Resource:** Joshua Stevens' guide to bivariate choropleth maps remains the definitive reference. See also [observablehq.com/@d3/bivariate-choropleth](https://observablehq.com/@d3/bivariate-choropleth).

### Value-by-Alpha Maps

Encode one variable as hue and a second as opacity (alpha). Often used to combine a measure with a reliability/confidence indicator -- e.g., unemployment rate as color, margin of error as opacity.

```javascript
// MapLibre: value-by-alpha
map.addLayer({
  id: "value-by-alpha",
  type: "fill",
  source: "tracts",
  paint: {
    "fill-color": [
      "interpolate", ["linear"], ["get", "unemployment"],
      2, "#2166ac", 6, "#f7f7f7", 10, "#b2182b"
    ],
    "fill-opacity": [
      "interpolate", ["linear"], ["get", "reliability_score"],
      0, 0.15,   // low reliability = near-transparent
      1, 0.9     // high reliability = near-opaque
    ]
  }
});
```

### Dot Density Maps

Place individual dots within polygon boundaries, with each dot representing a fixed number of units. Multiple colors represent different categories (e.g., racial/ethnic dot maps).

**Key considerations:**
- Dot placement must be random within polygons (not gridded)
- One dot = N units (state clearly in legend)
- Overlapping dots create emergent density patterns
- Performant for up to ~500,000 dots in SVG; use Canvas or WebGL beyond that

```python
# Python: dot density map
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import numpy as np

gdf = gpd.read_file("tracts.geojson")
dots_per_person = 100  # 1 dot = 100 people

all_dots = []
for _, row in gdf.iterrows():
    n_dots = int(row["population"] / dots_per_person)
    minx, miny, maxx, maxy = row.geometry.bounds
    dots = []
    while len(dots) < n_dots:
        x = np.random.uniform(minx, maxx)
        y = np.random.uniform(miny, maxy)
        p = Point(x, y)
        if row.geometry.contains(p):
            dots.append(p)
    all_dots.extend(dots)

dot_gdf = gpd.GeoDataFrame(geometry=all_dots)
fig, ax = plt.subplots(figsize=(12, 10))
gdf.boundary.plot(ax=ax, linewidth=0.2, color="#999")
dot_gdf.plot(ax=ax, markersize=0.3, color="#1f78b4", alpha=0.6)
ax.set_axis_off()
```

### Cartograms

Distort polygon geometry so that area becomes proportional to a variable. Preserves topology while encoding magnitude.

| Type | Distortion | Topology | SOTA Tool |
|------|-----------|----------|-----------|
| **Contiguous** | Warps boundaries continuously | Preserved | `topogram` (D3), `cartogram` (R) |
| **Non-contiguous** | Scales polygons from centroid | Broken (gaps) | D3, QGIS cartogram plugin |
| **Dorling** | Replaces polygons with circles | Abstract | D3 `d3.forceSimulation` |
| **Demers** | Replaces with squares | Abstract | D3 |
| **Hexagonal tile** | Each unit = one hex | Abstract | Observable, Pitch Interactive |

```javascript
// Dorling cartogram with D3
const simulation = d3.forceSimulation(features)
  .force("x", d3.forceX(d => centroid(d)[0]).strength(0.5))
  .force("y", d3.forceY(d => centroid(d)[1]).strength(0.5))
  .force("collide", d3.forceCollide(d => radiusScale(d.properties.population) + 1))
  .stop();

for (let i = 0; i < 300; i++) simulation.tick();

svg.selectAll("circle")
  .data(features)
  .join("circle")
    .attr("cx", d => d.x)
    .attr("cy", d => d.y)
    .attr("r", d => radiusScale(d.properties.population))
    .attr("fill", d => colorScale(d.properties.variable));
```

---

## Proportional Symbols and Graduated Maps

Encode a quantitative variable as the size of a point symbol placed at each feature's location or centroid.

### Circle Scaling Methods

| Method | Formula | Visual Effect | Use When |
|--------|---------|---------------|----------|
| **Mathematical (area)** | `r = sqrt(value / pi)` | Exact proportional area | Technically correct; readers underestimate |
| **Flannery** | `r = r1 * (value / value1)^0.5716` | Perceptual compensation | General audiences; compensates for underestimation |
| **Linear radius** | `r = value * scale` | Exaggerates large values | Avoid -- misleading |
| **Log scale** | `r = log(value) * scale` | Compresses range | Very wide value ranges |

```javascript
// D3: Flannery-corrected proportional symbols
const flanneryExponent = 0.5716;
const radiusScale = d3.scalePow()
  .exponent(flanneryExponent)
  .domain([0, d3.max(data, d => d.population)])
  .range([2, 40]);

svg.selectAll("circle")
  .data(data.sort((a, b) => b.population - a.population))  // large behind small
  .join("circle")
    .attr("cx", d => projection([d.lng, d.lat])[0])
    .attr("cy", d => projection([d.lng, d.lat])[1])
    .attr("r", d => radiusScale(d.population))
    .attr("fill", "#4292c6")
    .attr("fill-opacity", 0.7)
    .attr("stroke", "#08519c")
    .attr("stroke-width", 0.5);
```

> **Dark Arts Tip:** Always sort proportional symbols largest-to-smallest so small circles render on top. Without this, small values become invisible behind large ones.

### Pie Chart Maps

Place small pie charts at feature centroids to show both total magnitude (size) and categorical breakdown (slices).

```javascript
// D3: pie chart at each city location
const pie = d3.pie().sort(null).value(d => d.value);
const arc = d3.arc().innerRadius(0);

const cities = svg.selectAll("g.city")
  .data(cityData)
  .join("g")
    .attr("class", "city")
    .attr("transform", d => `translate(${projection([d.lng, d.lat])})`);

cities.each(function(city) {
  const radius = radiusScale(city.total);
  const localArc = arc.outerRadius(radius);
  d3.select(this).selectAll("path")
    .data(pie(city.categories))
    .join("path")
      .attr("d", localArc)
      .attr("fill", d => categoryColor(d.data.name))
      .attr("stroke", "#fff")
      .attr("stroke-width", 0.5);
});
```

### Bar Chart Maps and Waffle Maps

**Bar chart maps** place small bar charts at locations -- useful when you want to show temporal trends at each point (e.g., monthly temperature bars at weather stations).

**Waffle maps** use a grid of small squares (waffle chart) at each location, where each square represents a unit. More readable than pie charts for proportions near each other (e.g., 48% vs 52%).

### MapLibre Expressions for Symbol Sizing

```javascript
map.addLayer({
  id: "proportional-circles",
  type: "circle",
  source: "cities",
  paint: {
    "circle-radius": [
      "interpolate", ["linear"], ["zoom"],
      4, ["*", ["^", ["get", "population"], 0.5716], 0.002],
      10, ["*", ["^", ["get", "population"], 0.5716], 0.01]
    ],
    "circle-color": "#4292c6",
    "circle-opacity": 0.75,
    "circle-stroke-color": "#08519c",
    "circle-stroke-width": 0.5
  }
});
```

---

## Point and Cluster Maps

Represent discrete locations with symbols, aggregating dense point clouds into clusters at lower zoom levels.

### Supercluster Algorithm

[Supercluster](https://github.com/mapbox/supercluster) is the industry-standard spatial clustering library for web maps. It uses a hierarchical KD-tree index for sub-millisecond clustering at any zoom level.

**Key parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `radius` | 40 | Cluster radius in pixels |
| `maxZoom` | 16 | Maximum zoom to cluster at |
| `minZoom` | 0 | Minimum zoom to cluster at |
| `minPoints` | 2 | Minimum points to form a cluster |
| `extent` | 512 | Tile extent (radius is calculated relative to this) |
| `nodeSize` | 64 | KD-tree node size; affects performance |
| `map` | `null` | Function to compute cluster properties on initial load |
| `reduce` | `null` | Function to merge cluster properties during aggregation |

```javascript
import Supercluster from "supercluster";

const index = new Supercluster({
  radius: 60,
  maxZoom: 14,
  // Aggregate properties during clustering
  map: (props) => ({
    sum: props.magnitude,
    count: 1,
    maxMag: props.magnitude,
  }),
  reduce: (accumulated, props) => {
    accumulated.sum += props.sum;
    accumulated.count += props.count;
    accumulated.maxMag = Math.max(accumulated.maxMag, props.maxMag);
  },
});

index.load(geojsonFeatures);

// Get clusters for current viewport
const clusters = index.getClusters(
  [west, south, east, north],  // bbox
  Math.floor(map.getZoom())
);
```

### MapLibre GL JS Built-In Clustering

```javascript
map.addSource("earthquakes", {
  type: "geojson",
  data: "/data/earthquakes.geojson",
  cluster: true,
  clusterMaxZoom: 14,
  clusterRadius: 50,
  clusterProperties: {
    // Aggregate custom properties
    "maxMag": ["max", ["get", "magnitude"]],
    "totalCount": ["+", 1],
    "hasLarge": ["any", [">=", ["get", "magnitude"], 6.0]],
  }
});

// Cluster circles with stepped color
map.addLayer({
  id: "clusters",
  type: "circle",
  source: "earthquakes",
  filter: ["has", "point_count"],
  paint: {
    "circle-color": [
      "step", ["get", "point_count"],
      "#51bbd6", 100, "#f1f075", 750, "#f28cb1"
    ],
    "circle-radius": ["step", ["get", "point_count"], 20, 100, 30, 750, 40]
  }
});

// Cluster count labels
map.addLayer({
  id: "cluster-count",
  type: "symbol",
  source: "earthquakes",
  filter: ["has", "point_count"],
  layout: {
    "text-field": ["get", "point_count_abbreviated"],
    "text-font": ["DIN Pro Medium"],
    "text-size": 12,
  }
});

// Unclustered points
map.addLayer({
  id: "unclustered-point",
  type: "circle",
  source: "earthquakes",
  filter: ["!", ["has", "point_count"]],
  paint: {
    "circle-color": [
      "interpolate", ["linear"], ["get", "magnitude"],
      2, "#2dc4b2", 4, "#e3611c", 6, "#c70039"
    ],
    "circle-radius": [
      "interpolate", ["linear"], ["get", "magnitude"],
      2, 4, 6, 16
    ],
    "circle-stroke-width": 1,
    "circle-stroke-color": "#fff"
  }
});

// Click to zoom into cluster
map.on("click", "clusters", async (e) => {
  const features = map.queryRenderedFeatures(e.point, { layers: ["clusters"] });
  const clusterId = features[0].properties.cluster_id;
  const zoom = await map.getSource("earthquakes").getClusterExpansionZoom(clusterId);
  map.easeTo({ center: features[0].geometry.coordinates, zoom });
});
```

### Donut Clusters (Pie Chart Inside Clusters)

A popular technique that shows categorical breakdown within each cluster using a donut/pie chart rendered as an HTML marker or Canvas element.

```javascript
// Create donut chart markers for clusters
function createDonutChart(props) {
  const offsets = [];
  const categories = ["cat_a", "cat_b", "cat_c"];
  const colors = ["#e41a1c", "#377eb8", "#4daf4a"];
  let total = 0;

  categories.forEach(cat => {
    offsets.push(total);
    total += props[cat] || 0;
  });

  const r = Math.min(50, 10 + Math.sqrt(total) * 3);
  const r0 = Math.round(r * 0.6);  // inner radius for donut hole
  const w = r * 2;

  let html = `<div><svg width="${w}" height="${w}" viewBox="0 0 ${w} ${w}" ` +
    `text-anchor="middle" style="display:block">`;

  categories.forEach((cat, i) => {
    const pct = (props[cat] || 0) / total;
    if (pct > 0) {
      html += donutSegment(offsets[i] / total, (offsets[i] + props[cat]) / total, r, r0, colors[i]);
    }
  });

  html += `<circle cx="${r}" cy="${r}" r="${r0}" fill="white"/>`;
  html += `<text dominant-baseline="central" transform="translate(${r},${r})">${total}</text>`;
  html += `</svg></div>`;

  const el = document.createElement("div");
  el.innerHTML = html;
  return el.firstChild;
}
```

### deck.gl ScatterplotLayer and IconLayer

```javascript
import { ScatterplotLayer, IconLayer } from "@deck.gl/layers";

// High-performance scatterplot for millions of points
const scatterLayer = new ScatterplotLayer({
  id: "scatter",
  data: points,
  getPosition: d => [d.lng, d.lat],
  getRadius: d => d.value * 10,
  getFillColor: d => colorScale(d.category),
  radiusMinPixels: 2,
  radiusMaxPixels: 40,
  opacity: 0.8,
  pickable: true,
  antialiasing: true,
});

// Icon layer for categorical point maps
const iconLayer = new IconLayer({
  id: "icons",
  data: pois,
  getIcon: d => ({
    url: `/icons/${d.type}.png`,
    width: 64,
    height: 64,
    anchorY: 64,
  }),
  getPosition: d => [d.lng, d.lat],
  getSize: 24,
  sizeScale: 1,
  pickable: true,
});
```

---

## Statistical Maps

Maps designed for spatial statistical analysis and cross-disciplinary research. These go beyond simple thematic representation to encode analytical results.

### Box Maps (Spatial Box Plots)

A box map classifies features using box plot statistics: values below Q1 - 1.5*IQR are lower outliers, Q1-Q2 is the lower quartile, Q2-Q3 is the upper quartile, and values above Q3 + 1.5*IQR are upper outliers. Six classes total.

```python
# Python: box map using geopandas and mapclassify
import geopandas as gpd
import mapclassify
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

gdf = gpd.read_file("census_tracts.geojson")
values = gdf["median_income"].values

# Box map classification
box = mapclassify.BoxPlot(values)
gdf["box_class"] = box.yb

# Six-class color scheme: lower outlier, lower hinge, Q2, Q3, upper hinge, upper outlier
colors = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"]
cmap = ListedColormap(colors[:box.k])

fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(column="box_class", cmap=cmap, legend=True, ax=ax,
         edgecolor="#999", linewidth=0.2)
ax.set_axis_off()
ax.set_title("Median Income: Box Map Classification", fontsize=14)
```

### LISA Cluster Maps (Local Indicators of Spatial Association)

LISA maps visualize spatial autocorrelation: where similar values cluster (hot spots, cold spots) and where dissimilar values are neighbors (spatial outliers).

**Cluster types:**
- **High-High (HH):** High values surrounded by high values (hot spots)
- **Low-Low (LL):** Low values surrounded by low values (cold spots)
- **High-Low (HL):** High value surrounded by low values (spatial outlier)
- **Low-High (LH):** Low value surrounded by high values (spatial outlier)
- **Not significant:** No statistically significant local autocorrelation

```python
# Python: LISA cluster map using PySAL
import geopandas as gpd
import libpysal
from esda.moran import Moran_Local
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

gdf = gpd.read_file("counties.geojson")
y = gdf["poverty_rate"].values

# Build spatial weights (Queen contiguity)
w = libpysal.weights.Queen.from_dataframe(gdf)
w.transform = "r"  # row-standardize

# Compute Local Moran's I
lisa = Moran_Local(y, w, permutations=999)

# Assign cluster labels (significant at p < 0.05)
sig = lisa.p_sim < 0.05
spots = lisa.q  # 1=HH, 2=LH, 3=LL, 4=HL
labels = np.where(sig, spots, 0)  # 0 = not significant
gdf["lisa_cluster"] = labels

# Plot
colors = ["#d3d3d3", "#e31a1c", "#4575b4", "#2166ac", "#ef8a62"]
# 0=ns(grey), 1=HH(red), 2=LH(blue), 3=LL(darkblue), 4=HL(orange)
cmap = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(12, 10))
gdf.plot(column="lisa_cluster", cmap=cmap, legend=True, ax=ax,
         edgecolor="#999", linewidth=0.2)
ax.set_axis_off()
ax.set_title("Poverty Rate: LISA Cluster Map (p < 0.05)", fontsize=14)

# Print global Moran's I for context
from esda.moran import Moran
mi = Moran(y, w, permutations=999)
print(f"Global Moran's I: {mi.I:.4f}  (p={mi.p_sim:.4f})")
```

### Election and Voting Maps

Voting maps require special attention because winner-take-all choropleth maps systematically mislead by conflating geographic area with population.

**Alternatives to binary red/blue maps:**

| Technique | What it Shows | Tool |
|-----------|---------------|------|
| **Purple maps** | Blend winner color with margin of victory | D3, QGIS |
| **Swing maps** | Change from previous election | D3, Observable |
| **Dot density** | One dot per N votes, colored by party | D3, Python |
| **Cartograms** | Area proportional to votes or population | D3, `topogram` |
| **Graduated symbols** | Circle size = total votes, color = winner | D3, MapLibre |
| **Bivariate** | Turnout + margin simultaneously | D3 |

> **Dark Arts Tip:** The single most impactful improvement to any election map is switching from area choropleth to a Dorling cartogram or dot density. The default county choropleth exaggerates rural land area and is fundamentally misleading about the distribution of voters.

### Demographic Pyramids on Maps

Place small population pyramids at each administrative unit centroid. Best implemented as D3 small multiples with geographic positioning.

```javascript
// D3: positioned population pyramids
const pyramids = svg.selectAll("g.pyramid")
  .data(regions)
  .join("g")
    .attr("class", "pyramid")
    .attr("transform", d => {
      const [x, y] = projection(d3.geoCentroid(d));
      return `translate(${x - 20}, ${y - 30})`;
    });

pyramids.each(function(region) {
  const g = d3.select(this);
  const ageGroups = region.properties.age_data;
  const maxPop = d3.max(ageGroups, d => Math.max(d.male, d.female));
  const barScale = d3.scaleLinear().domain([0, maxPop]).range([0, 18]);

  ageGroups.forEach((ag, i) => {
    // Male bars (left)
    g.append("rect")
      .attr("x", 20 - barScale(ag.male))
      .attr("y", i * 4).attr("width", barScale(ag.male))
      .attr("height", 3).attr("fill", "#4292c6");
    // Female bars (right)
    g.append("rect")
      .attr("x", 22).attr("y", i * 4)
      .attr("width", barScale(ag.female))
      .attr("height", 3).attr("fill", "#e377c2");
  });
});
```

### SOTA Tools for Statistical Maps

| Tool | Capabilities | Language |
|------|-------------|----------|
| **GeoDa** | LISA, box maps, spatial regression, conditional maps | GUI (C++) |
| **PySAL / esda** | Moran's I, LISA, Getis-Ord G*, spatial outliers | Python |
| **R spdep + tmap** | Full spatial statistics + publication maps | R |
| **R ggplot2 + sf** | Grammar of graphics for spatial data | R |
| **D3.js** | Custom statistical maps with full control | JavaScript |
| **Observable Plot** | Fast statistical exploration with facets | JavaScript |

---

## Specialized Thematic Maps

Advanced and less common map types for specific analytical needs.

### Dasymetric Maps

Dasymetric mapping refines choropleth maps by redistributing data from administrative boundaries to areas where the phenomenon actually occurs, using ancillary data (land cover, building footprints, nighttime lights).

**Example:** Instead of showing population density per county (which includes forests, lakes, and empty land), redistribute population only to residential/urban land cover zones.

```python
# Conceptual Python workflow for dasymetric mapping
import geopandas as gpd
import rasterstats

# 1. Load census tracts and land cover
tracts = gpd.read_file("census_tracts.geojson")
# Land cover raster: 1=developed, 2=agricultural, 3=forest, 4=water, etc.

# 2. Calculate developed area per tract
stats = rasterstats.zonal_stats(tracts, "landcover.tif", categorical=True)
tracts["developed_area_km2"] = [
    s.get(1, 0) * pixel_area_km2 for s in stats
]

# 3. Compute dasymetric density
tracts["dasymetric_density"] = (
    tracts["population"] / tracts["developed_area_km2"]
)
# Areas with no developed land get NaN (show as no data)

# 4. Alternatively, use binary dasymetric with building footprints
buildings = gpd.read_file("building_footprints.geojson")
# Allocate population proportionally to total building area per tract
```

> **Dark Arts Tip:** Dasymetric maps are dramatically more honest than standard choropleth for any variable related to human activity. If you have land cover or building footprint data, always offer a dasymetric alternative. The visual difference is striking and the analytical improvement is real.

### Isarithmic (Isopleth / Isoline) Maps

Map continuous phenomena using contour lines or filled contours. Classic examples: temperature, precipitation, elevation, air pressure.

**Two subtypes:**
- **Isometric:** True point measurements (e.g., temperature at weather stations), interpolated
- **Isopleth:** Derived ratios (e.g., population density), interpolated from area centroids

| Interpolation Method | Best For | Notes |
|---------------------|----------|-------|
| **IDW** | General purpose | Simple, fast; bull's-eye artifacts around points |
| **Kriging** | Geostatistical data | Optimal if variogram model is correct |
| **Spline** | Smooth surfaces | Can overshoot; no error estimate |
| **Natural neighbor** | Irregularly spaced data | No parameters to tune; respects data extent |

### Anamorphic Maps

Maps where a geometric property (area, distance) is distorted to represent a variable. Cartograms (covered above) are the most common form. Other types include:

- **Distance cartograms:** Distort space so that distance represents travel time rather than Euclidean distance
- **Central-place anamorphoses:** City at center, surrounding space distorted by accessibility

### Prism Maps (3D Thematic)

Extrude polygon boundaries vertically in proportion to a data value, creating a 3D "cityscape" effect.

```javascript
// deck.gl GeoJsonLayer with extrusion
import { GeoJsonLayer } from "@deck.gl/layers";

const prismLayer = new GeoJsonLayer({
  id: "prism",
  data: "census_tracts.geojson",
  extruded: true,
  wireframe: true,
  getElevation: d => d.properties.population_density * 5,
  getFillColor: d => colorScale(d.properties.population_density),
  getLineColor: [80, 80, 80],
  lineWidthMinPixels: 1,
  pickable: true,
  material: {
    ambient: 0.35,
    diffuse: 0.6,
    shininess: 32,
  },
});
```

### Specialized Map Type Summary

| Map Type | Data Requirement | When to Use | SOTA Tool |
|----------|-----------------|-------------|-----------|
| **Dasymetric** | Polygon data + ancillary raster/vector | Honest density mapping | Python + rasterstats |
| **Isarithmic** | Point samples or raster | Continuous phenomena | QGIS, Python scipy, GMT |
| **Anamorphic** | Network or accessibility data | Travel time, accessibility | D3, Scapetoad |
| **Prism (3D)** | Polygon + quantitative variable | Dramatic presentations | deck.gl, CesiumJS, QGIS 3D |

---

## Color Science

Color choice is the single most impactful design decision on any thematic map. Poor color choices produce maps that are misleading, inaccessible, or both.

### Perceptual Uniformity

A perceptually uniform color space ensures that equal numerical differences in the data produce equal perceived differences in color. This is critical for accurate map reading.

| Color Space | Uniformity | Use Case |
|-------------|-----------|----------|
| **sRGB** | Non-uniform | Screen display (never interpolate in this space) |
| **CIELAB (L\*a\*b\*)** | Approximately uniform | Classic standard; most tools support it |
| **Oklab** | Improved uniformity over CIELAB | Modern CSS (`oklch()`); best for web |
| **CAM16-UCS** | High uniformity | Research-grade; computationally expensive |
| **Jzazbz** | High dynamic range | HDR displays |

> **Dark Arts Tip:** When interpolating colors for a sequential scale, always interpolate in a perceptually uniform space (CIELAB or Oklab), not in RGB. RGB interpolation produces muddy, non-uniform gradients. Chroma.js does CIELAB interpolation by default; d3-interpolate offers `d3.interpolateLab`.

### Color Palette Systems

#### Sequential (Ordered Data)

| Palette Family | Source | Notes |
|---------------|--------|-------|
| **Viridis** | matplotlib | Perceptually uniform, colorblind-safe, prints in grayscale |
| **Plasma, Inferno, Magma, Cividis** | matplotlib | Viridis siblings; Cividis optimized for deuteranopia |
| **YlOrRd, Blues, Greens** | ColorBrewer | Classic cartographic; 3-9 class versions |
| **Mako, Rocket, Flare** | seaborn | Modern, perceptually uniform |
| **Batlow, Bamako, Devon** | Crameri Scientific | Published with perceptual uniformity proofs |
| **CARTOColors Mint, Burg, Teal** | CARTO | Web-optimized, modern aesthetic |

#### Diverging (Data with Meaningful Midpoint)

| Palette | Source | Notes |
|---------|--------|-------|
| **RdBu, RdYlBu, PiYG** | ColorBrewer | Red-Blue is the classic; avoid for election maps (political connotation) |
| **BrBG, PRGn** | ColorBrewer | Alternative hue pairs without political baggage |
| **Vik, Berlin, Broc** | Crameri Scientific | Perceptually uniform diverging |
| **CARTOColors Temps, Tropic** | CARTO | Modern web-optimized |

#### Qualitative (Categorical Data)

| Palette | Max Classes | Source | Notes |
|---------|-------------|--------|-------|
| **Set1, Set2, Paired** | 8-12 | ColorBrewer | Classic; Set1 is high saturation |
| **Tableau 10, 20** | 10, 20 | Tableau | Carefully balanced |
| **Observable 10** | 10 | Observable | D3 default categorical |
| **Bold, Vivid, Safe** | 12 | CARTOColors | Web-optimized |

### Scientific Colour Maps (Crameri)

Fabio Crameri's [Scientific Colour Maps](https://www.fabiocrameri.ch/colourmaps/) are the gold standard for research publication. They are:
- Perceptually uniform (proven, not just claimed)
- Colorblind-safe
- Readable in grayscale
- Available for all major platforms (matplotlib, R, QGIS, GMT, Paraview)

```python
# Python: use Crameri scientific colour maps
# pip install cmcrameri
import cmcrameri.cm as cmc
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
gdf.plot(column="temperature", cmap=cmc.batlow, ax=ax, legend=True)
```

### Colorblind Simulation and WCAG Compliance

Approximately 8% of men and 0.5% of women have some form of color vision deficiency. Always test your maps.

**Testing tools:**

| Tool | Platform | Notes |
|------|----------|-------|
| **Viz Palette** | Web | Real-time distinctiveness + CVD simulation |
| **Coblis** | Web | Upload image, simulate 8 CVD types |
| **Sim Daltonism** | macOS | System-wide live filter |
| **Color Oracle** | Win/Mac/Linux | System-wide simulation |
| **Chromatic Vision Simulator** | iOS/Android | Camera-based real-time |

**WCAG guidelines for maps:**
- 3:1 minimum contrast ratio for graphical elements against background
- Do not rely on color alone to convey meaning (add pattern, labels, or size)
- Provide text alternatives (tooltips, data tables)
- Test with grayscale: if your map is unreadable in grayscale, add redundant encoding

### Dark Mode Palettes

Dark basemaps require adjusted palettes. Colors that work on white backgrounds often fail on dark backgrounds.

**Principles:**
- Increase lightness of fill colors (L* > 40 in CIELAB)
- Use lighter, desaturated outlines instead of dark ones
- Reduce fill opacity slightly to let the basemap show through
- Test sequential palettes: many "light-to-dark" ramps need to be reversed or adjusted
- Viridis and Inferno work well on both light and dark backgrounds

```javascript
// Chroma.js: generate a dark-mode-friendly sequential scale
import chroma from "chroma-js";

const darkModeScale = chroma.scale(["#1a1a2e", "#16213e", "#0f3460", "#e94560"])
  .mode("lab")
  .domain([0, 100])
  .classes(5);

// Or adjust an existing palette for dark backgrounds
const adjusted = chroma.scale("YlOrRd")
  .mode("lab")
  .correctLightness()  // ensure perceptual uniformity
  .domain([0, 100]);
```

### Key Color Libraries

| Library | Language | Strengths |
|---------|----------|-----------|
| **Chroma.js** | JavaScript | Interpolation in CIELAB/Oklab, scale generation, CVD simulation |
| **d3-scale-chromatic** | JavaScript | All ColorBrewer + additional schemes, ready for D3 scales |
| **d3-interpolate** | JavaScript | `interpolateLab`, `interpolateHcl` for perceptually uniform gradients |
| **paletteer** | R | Unified interface to 2,500+ palettes from 65+ packages |
| **cmcrameri** | Python | Crameri scientific colour maps for matplotlib |
| **matplotlib** | Python | Viridis family built-in; register custom colormaps |
| **colorspace** | R | HCL-based palette construction, CVD simulation |

---

## Tool Comparison Matrix

Comprehensive comparison of tools for thematic map creation. Ratings: strong support, partial support, or not supported.

### Map Type Support

| Tool | Choropleth | Heat/Density | Flow/Arc | Bivariate | Proportional | Cluster | Statistical | 3D Thematic |
|------|-----------|-------------|----------|-----------|-------------|---------|------------|------------|
| **D3.js** | Full | Full | Full | Full | Full | Manual | Full | Limited |
| **Observable Plot** | Full | Partial | Partial | Full | Full | No | Partial | No |
| **Vega-Lite** | Full | Partial | Partial | Full | Full | No | Partial | No |
| **QGIS** | Full | Full | Partial | Manual | Full | Partial | Full | Full |
| **Kepler.gl** | Full | Full | Full | No | Partial | Full | No | Full |
| **MapLibre GL JS** | Full | Full | Partial | Partial | Full | Full | No | No |
| **deck.gl** | Full | Full | Full | Partial | Full | Full | No | Full |
| **ECharts** | Full | Full | Full | No | Full | No | No | Full |
| **AntV L7** | Full | Full | Full | No | Full | Full | No | Full |
| **R tmap** | Full | Full | Partial | Full | Full | No | Full | No |
| **R ggplot2+sf** | Full | Partial | Partial | Full | Full | No | Full | No |
| **Python geopandas** | Full | Partial | No | Partial | Full | No | Full | No |
| **Python folium** | Full | Full | Partial | No | Partial | Full | No | No |

### Performance and Workflow

| Tool | Max Features | Learning Curve | Interactivity | Export Formats | Ecosystem |
|------|-------------|----------------|---------------|----------------|-----------|
| **D3.js** | ~50K (SVG), ~500K (Canvas) | Steep | Full custom | SVG, PNG, PDF | npm, Observable |
| **Observable Plot** | ~100K | Moderate | Basic built-in | SVG, PNG | Observable |
| **Vega-Lite** | ~100K | Moderate | Declarative selections | SVG, PNG, Canvas | npm, Python (Altair) |
| **QGIS** | Millions | Moderate | Limited (print-oriented) | PDF, SVG, PNG, GeoTIFF | Plugins |
| **Kepler.gl** | ~1M (GPU) | Low | Full built-in | HTML, JSON, Image | React, Jupyter |
| **MapLibre GL JS** | ~1M (vector tiles) | Moderate | Full custom | Canvas/PNG | npm, plugins |
| **deck.gl** | Millions (GPU) | Moderate-Steep | Full custom | Canvas/PNG | npm, React, Jupyter |
| **ECharts** | ~100K | Low-Moderate | Built-in rich | Canvas, SVG | npm |
| **AntV L7** | ~1M (GPU) | Moderate | Built-in | Canvas | npm |
| **R tmap** | ~500K | Moderate | Leaflet (view mode) | PDF, PNG, HTML | CRAN |
| **R ggplot2+sf** | ~500K | Moderate-Steep | plotly conversion | PDF, PNG, SVG | CRAN |
| **Python geopandas** | ~500K | Low-Moderate | matplotlib interactive | PNG, PDF, SVG | pip |
| **Python folium** | ~100K | Low | Leaflet-based | HTML | pip |

### Decision Guide: Choosing a Tool

```
What is your primary output?
  PRINT / PUBLICATION
    -> R (tmap or ggplot2+sf) for reproducible, publication-quality
    -> QGIS for complex layouts with multiple map frames
    -> Python geopandas+matplotlib for data science pipelines
  INTERACTIVE WEB
    -> Simple dashboard? Kepler.gl or folium
    -> Custom interactions? MapLibre GL JS or deck.gl
    -> Maximum control? D3.js
    -> Declarative / notebook? Vega-Lite (via Altair in Python)
  EXPLORATORY ANALYSIS
    -> Kepler.gl for zero-code exploration
    -> GeoDa for spatial statistics
    -> QGIS for general GIS analysis
  PRESENTATION / STAKEHOLDER DEMO
    -> Kepler.gl (share as HTML)
    -> Observable notebook (live, shareable)
    -> ECharts or AntV L7 for Chinese-market projects
```

---

## Accessibility Checklist

Before publishing any thematic map, verify:

- [ ] **Color**: Tested with at least 2 CVD simulators (deuteranopia + protanopia)
- [ ] **Redundant encoding**: Color is not the only visual variable (add pattern, size, labels, or tooltips)
- [ ] **Legend**: Clear, complete, with units; positioned where it will not be cropped
- [ ] **Classification**: Method stated in legend or caption; breakpoints visible
- [ ] **Normalization**: Rates used instead of raw counts for choropleth; stated explicitly
- [ ] **Contrast**: 3:1 minimum for graphical elements (WCAG 2.1 Level AA)
- [ ] **Text alternatives**: Alt text for static images; data table available for interactive maps
- [ ] **Keyboard navigation**: Interactive maps support Tab, Enter, Escape
- [ ] **Classes**: Maximum 7 for choropleth (5 preferred); justified if more
- [ ] **No data**: Missing values shown distinctly (grey with "No data" label), not as zero

---

## Further Reading

- Brewer, C. A. (2005). *Designing Better Maps: A Guide for GIS Users*. Esri Press.
- Slocum, T. A. et al. (2022). *Thematic Cartography and Geovisualization*. 4th ed. CRC Press.
- Crameri, F., Shephard, G. E., & Heron, P. J. (2020). "The misuse of colour in science communication." *Nature Communications*, 11(1), 5444.
- Stevens, J. (2015). "Bivariate Choropleth Maps: A How-to Guide." [joshuastevens.net](https://www.joshuastevens.net/cartography/make-a-bivariate-choropleth-map/).
- Munzner, T. (2014). *Visualization Analysis and Design*. CRC Press.

---

[Back to Visualization](README.md) | [Back to Main README](../README.md)
