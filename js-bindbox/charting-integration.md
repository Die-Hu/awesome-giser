# Charting & Map Integration

> Libraries and patterns for integrating charts, graphs, and statistical visualizations with geospatial maps.

> **Quick Picks**
> - **SOTA**: [D3.js](https://d3js.org) + MapLibre -- full control over cartographic + statistical rendering
> - **Free Best**: [Apache ECharts 5.x](https://echarts.apache.org) -- all-in-one chart + map with linked views
> - **Fastest Setup**: [Observable Framework](https://observablehq.com/framework) -- markdown-based geo dashboards with live data

## D3.js Geo

D3's geographic projection and path rendering system, the foundation for custom cartographic visualizations.

- **Website:** [d3js.org](https://d3js.org)
- **Key Strengths:** Full projection control, data-driven bindings, choropleth/cartogram support
- **Best For:** Custom choropleth maps, cartograms, publication-quality geo-viz

### Key Modules

- `d3-geo` -- Projections, paths, graticules
- `d3-geo-projection` -- Extended projection library (100+ projections)
- `d3-scale-chromatic` -- Color schemes for thematic maps

### Install

```bash
npm install d3 d3-geo d3-geo-projection d3-scale-chromatic topojson-client
```

### Choropleth Example

```javascript
import * as d3 from 'd3';
import * as topojson from 'topojson-client';

const width = 960, height = 600;
const svg = d3.select('#map').append('svg')
  .attr('viewBox', [0, 0, width, height]);

const projection = d3.geoMercator().fitSize([width, height], geojson);
const path = d3.geoPath(projection);

// Color scale
const color = d3.scaleQuantize()
  .domain([0, d3.max(geojson.features, d => d.properties.density)])
  .range(d3.schemeBlues[7]);

svg.selectAll('path')
  .data(geojson.features)
  .join('path')
  .attr('d', path)
  .attr('fill', d => color(d.properties.density))
  .attr('stroke', '#fff')
  .attr('stroke-width', 0.5)
  .append('title')
  .text(d => `${d.properties.name}: ${d.properties.density}`);
```

### D3 + Mapbox/MapLibre Integration Pattern

Overlay D3 SVG on a MapLibre map so features reproject and zoom with the basemap:

```javascript
import maplibregl from 'maplibre-gl';
import * as d3 from 'd3';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [-96, 38], zoom: 4
});

map.on('load', () => {
  const container = map.getCanvasContainer();
  const svg = d3.select(container).append('svg')
    .style('position', 'absolute')
    .style('width', '100%')
    .style('height', '100%');

  function project(d) {
    const p = map.project(new maplibregl.LngLat(d[0], d[1]));
    return [p.x, p.y];
  }

  const path = d3.geoPath().projection(
    d3.geoTransform({
      point: function(lng, lat) {
        const [x, y] = project([lng, lat]);
        this.stream.point(x, y);
      }
    })
  );

  const features = svg.selectAll('path')
    .data(geojson.features)
    .join('path')
    .attr('d', path)
    .attr('fill', d => colorScale(d.properties.value))
    .attr('stroke', '#333')
    .attr('opacity', 0.7);

  // Redraw on map movement
  map.on('viewreset', () => features.attr('d', path));
  map.on('move', () => features.attr('d', path));
});
```

---

## Apache ECharts 5.x

Apache ECharts includes built-in map and geo components for combining charts with geographic visualization.

- **Website:** [echarts.apache.org](https://echarts.apache.org)
- **Key Strengths:** Rich chart types, map series, tooltip integration, large dataset support
- **Best For:** Dashboard map-chart combos, scatter/bubble maps, linked views

### Install

```bash
npm install echarts
```

### ECharts 5.x Geo Features

- **GeoJSON registration:** Load custom region boundaries
- **Visual map:** Continuous and piecewise data mapping to color
- **Scatter on map:** Plot points with size/color data encoding
- **Lines series:** Flow/OD lines with animation
- **Large dataset support:** `dataset` component with `transform` for filtering/sorting
- **Universal transitions:** Animated transitions between chart types

### Map + Chart Linked Example

```javascript
import * as echarts from 'echarts';

// Register GeoJSON for your region
echarts.registerMap('china', chinaGeoJSON);

const chart = echarts.init(document.getElementById('dashboard'));

chart.setOption({
  // Shared dataset
  dataset: {
    source: [
      ['province', 'gdp', 'population'],
      ['Beijing',   40269, 2189],
      ['Shanghai',  43214, 2487],
      ['Guangdong', 124369, 12684],
      // ...
    ]
  },
  // Grid layout: map on left, bar chart on right
  grid: { left: '55%', right: '5%', top: '15%', bottom: '10%' },
  xAxis: { type: 'value' },
  yAxis: { type: 'category', inverse: true },
  geo: {
    map: 'china',
    left: '5%', right: '50%',
    roam: true,
    emphasis: { itemStyle: { areaColor: '#fdb462' } }
  },
  visualMap: {
    type: 'continuous', min: 0, max: 150000,
    inRange: { color: ['#e0f3f8', '#313695'] },
    seriesIndex: 0
  },
  series: [
    {
      type: 'map', map: 'china', geoIndex: 0,
      encode: { value: 'gdp', itemName: 'province' }
    },
    {
      type: 'bar', encode: { x: 'gdp', y: 'province' },
      label: { show: true, position: 'right' }
    }
  ]
});
```

---

## Observable Framework

Observable Framework is a markdown-based static site generator for data apps with built-in reactive data loaders.

- **Website:** [observablehq.com/framework](https://observablehq.com/framework)
- **Key Strengths:** Markdown + JS code blocks, reactive data flow, built-in Plot/D3/MapLibre, deploys as static site
- **Best For:** Data journalism, geo dashboards, shareable analytical reports

### Quick Geo Dashboard

In an Observable Framework project, write Markdown with embedded JS:

````markdown
# Regional Population Dashboard

```js
const data = await FileAttachment("data/regions.geojson").json();
```

```js
// Interactive MapLibre map
const map = new maplibregl.Map({
  container: display(document.createElement("div")),
  style: "https://demotiles.maplibre.org/style.json",
  center: [-98, 39], zoom: 4
});
```

```js
// Linked histogram using Observable Plot
Plot.plot({
  marks: [
    Plot.rectY(data.features, Plot.binX(
      { y: "count" },
      { x: d => d.properties.population, fill: "steelblue" }
    )),
    Plot.ruleY([0])
  ]
})
```
````

> Observable Framework deploys as a static site -- no server required. Data loaders run at build time, so dashboards are fast even with heavy data processing.

---

## Observable Plot + Geo

Observable Plot provides a concise, high-level API for geo-aware statistical graphics.

- **Website:** [observablehq.com/plot](https://observablehq.com/plot)
- **Key Strengths:** Declarative API, geo mark, faceting, automatic scales
- **Best For:** Exploratory geo-statistical analysis, quick geo plots

### Install

```bash
npm install @observablehq/plot
```

```javascript
import * as Plot from '@observablehq/plot';

// Choropleth in ~5 lines
const chart = Plot.plot({
  projection: 'albers-usa',
  color: { scheme: 'blues', legend: true },
  marks: [
    Plot.geo(statesGeoJSON, {
      fill: d => populationMap.get(d.properties.name),
      stroke: 'white',
      title: d => `${d.properties.name}: ${populationMap.get(d.properties.name)}`
    })
  ]
});
document.getElementById('chart').append(chart);
```

---

## Plotly.js Maps

Plotly.js includes map trace types for combining geographic and statistical visualizations.

- **Website:** [plotly.com/javascript](https://plotly.com/javascript)
- **Key Strengths:** Interactive, scattergeo/choropleth traces, Mapbox integration, Python/R interop
- **Best For:** Interactive dashboards, scientific geo-visualization

### Install

```bash
npm install plotly.js-dist
```

```javascript
import Plotly from 'plotly.js-dist';

// Choropleth map
Plotly.newPlot('map', [{
  type: 'choropleth',
  locations: ['CA', 'TX', 'NY'],
  z: [39.5, 29.0, 19.5],
  locationmode: 'USA-states',
  colorscale: 'Viridis'
}], {
  geo: { scope: 'usa', showlakes: true, lakecolor: '#99c0db' }
});

// Scattergeo for point data
Plotly.newPlot('scatter', [{
  type: 'scattergeo',
  lat: [39.9, 31.2, 23.1],
  lon: [116.4, 121.5, 113.3],
  text: ['Beijing', 'Shanghai', 'Guangzhou'],
  marker: { size: [20, 18, 15], color: ['red', 'blue', 'green'] }
}]);
```

---

## Integration Patterns: Chart + Map Sync

Patterns for synchronizing chart selections with map views for coordinated multi-view analysis.

### Pattern 1: Click-to-Filter

Select a region on the map to filter data in an adjacent chart.

### Pattern 2: Brushing & Linking

Brush a range on a chart (e.g., histogram) and highlight corresponding features on the map.

### Pattern 3: Tooltip Coordination

Hover over a chart element to highlight the geographic feature and vice versa.

### Pattern 4: Shared State Management

Use a shared state store (e.g., Zustand, Redux) to keep map and chart views in sync.

| Pattern | Complexity | Libraries | Use Case |
|---------|-----------|-----------|----------|
| Click-to-Filter | Low | Any map + chart lib | Region-based dashboards |
| Brushing & Linking | Medium | D3 + Leaflet/MapLibre | Exploratory analysis |
| Tooltip Coordination | Low | ECharts or Plotly | Hover-based insight |
| Shared State | Medium-High | React + MapLibre + D3 | Full dashboard applications |

### Synchronized Brushing Example (D3 + MapLibre)

```javascript
import * as d3 from 'd3';
import maplibregl from 'maplibre-gl';

// Shared selection state
let selectedIds = new Set();

// --- Histogram with brush ---
const brush = d3.brushX()
  .extent([[0, 0], [chartWidth, chartHeight]])
  .on('brush end', ({ selection }) => {
    if (!selection) {
      selectedIds.clear();
    } else {
      const [x0, x1] = selection.map(xScale.invert);
      selectedIds = new Set(
        data.filter(d => d.value >= x0 && d.value <= x1).map(d => d.id)
      );
    }
    updateMap();
    updateBars();
  });

svg.append('g').call(brush);

// --- Map responds to brush ---
function updateMap() {
  map.setPaintProperty('points-layer', 'circle-opacity', [
    'case',
    ['in', ['get', 'id'], ['literal', [...selectedIds]]],
    1.0,  // selected
    0.15  // unselected
  ]);
}

// --- Bars respond to brush ---
function updateBars() {
  svg.selectAll('.bar')
    .attr('opacity', d => selectedIds.size === 0 || selectedIds.has(d.id) ? 1 : 0.15);
}
```

## Comparison Table

| Library | Map Types | Chart Types | Bundle Size | Framework | Best For |
|---------|-----------|-------------|-------------|-----------|----------|
| D3.js | Custom (any projection) | Unlimited (custom) | ~30 KB (d3-geo) | Vanilla JS | Full control |
| ECharts 5.x | Registered GeoJSON maps | 20+ chart types | ~300 KB | Vanilla / Vue / React | All-in-one dashboards |
| Observable Plot | Built-in projections | Statistical marks | ~80 KB | Observable / vanilla | Quick geo-stat analysis |
| Plotly.js | Scattergeo, choropleth, Mapbox | 40+ trace types | ~1 MB | Vanilla / React / Python | Interactive dashboards |
| Observable Framework | MapLibre, D3, Plot | Any JS library | Static site | Markdown + JS | Data journalism |
