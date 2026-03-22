# Charting & Geo-Visualization Integration

> Data validated: 2026-03-21

## 30-Second Decision

| I need... | Use this | Startup time |
|-----------|----------|-------------|
| Fastest choropleth / geo chart | Observable Plot | 5 min |
| All-in-one map + charts dashboard | ECharts | 30 min |
| Python/R interop dashboard | Plotly.js | 30 min |
| Grammar-of-graphics, academic | Vega-Lite | 30 min |
| Full custom cartographic control | D3-geo | Hours |
| 1M+ points on a basemap | deck.gl ([see 3d-mapping](3d-mapping.md)) | Hours |
| Chinese market, AMap basemap | AntV L7 | 1 hour |

---

## Detailed Guide (by startup time)

### 1. Observable Plot

The fastest path to a geo chart. Declarative API built on D3, `Plot.geo()` creates choropleths in one function call.

**Quick facts:** ~50KB gzip | Easy | ~4.8K GitHub stars | Production-readiness: 3/5

```javascript
import * as Plot from '@observablehq/plot';
import { feature } from 'topojson-client';

const states = feature(us, us.objects.states);
const chart = Plot.plot({
  projection: 'albers-usa',
  marks: [
    Plot.geo(states, {
      fill: d => populationMap.get(d.id),
      stroke: 'white',
      title: d => d.properties.name
    })
  ]
});
document.getElementById('chart').append(chart);
```

**Small project:** Best choice for quick geo charts -- one function call = choropleth map. Small bundle, no config. Perfect for dashboards and reports.

**Key caveats:**
- Not interactive by default (static SVG)
- SVG performance ceiling at ~10K features
- Smaller community, fewer examples outside Observable ecosystem

---

### 2. ECharts

All-in-one chart + map in a single library. 30+ chart types plus a built-in geo/map component.

**Quick facts:** ~800KB full (tree-shakable, geo ~200KB gzip) | ~65K GitHub stars | ~1.1M npm/week | Apache project

```javascript
import * as echarts from 'echarts';
const chart = echarts.init(document.getElementById('chart'));
chart.setOption({
  geo: { map: 'china', roam: true, label: { show: true } },
  series: [{
    type: 'scatter', coordinateSystem: 'geo',
    data: [[116.4, 39.9, 100], [121.4, 31.2, 200]],
    encode: { value: 2 },
    symbolSize: d => Math.sqrt(d[2]) * 2
  }]
});
```

**Small project:** Great for dashboards that need both charts and maps -- one library handles everything. Rich built-in chart types. Good Chinese documentation.

**Key caveats:**
- Bundle is massive (~800KB full); must tree-shake aggressively
- Map component is simplified -- no vector tiles, no basemaps. Use MapLibre for the map, ECharts for charts.
- Config objects become deeply nested (500+ lines) for complex dashboards
- **Anti-pattern:** Using ECharts map as your primary map component. Use MapLibre for maps, ECharts for charts.

---

### 3. Plotly.js

Python/R interop is the killer feature. Same API in Python (Dash), R, Julia, and JS.

**Quick facts:** ~1MB+ gzip (massive) | ~17.7K GitHub stars | npm: ~235K/week for plotly.js core (plotly.js-dist and other distribution packages account for additional downloads)

```javascript
import Plotly from 'plotly.js-dist';
Plotly.newPlot('map', [{
  type: 'scattergeo',
  lat: [39.9, 31.2, 23.1],
  lon: [116.4, 121.4, 113.3],
  text: ['Beijing', 'Shanghai', 'Guangzhou'],
  mode: 'markers',
  marker: { size: [20, 30, 15], color: ['red', 'blue', 'green'] }
}], {
  geo: { scope: 'asia', projection: { type: 'natural earth' } }
});
```

**Small project:** Good when you're already using Python/Dash or need cross-language consistency. But the 1MB+ bundle is a serious downside for pure JS projects -- consider ECharts or Observable Plot instead.

**Key caveats:**
- Enormous bundle (~1MB+ gzip for full, ~3MB+ complete)
- Map charts depend on Mapbox GL JS v1 (pre-license-change); MapLibre integration requires workarounds
- Performance ceiling at 50K+ points
- Dash (Python) is the primary use case; JS-only gets less attention

---

### 4. Vega-Lite

Grammar-of-graphics for declarative visualization. JSON specification describes the chart.

**Quick facts:** ~300KB gzip (with Vega runtime) | ~5.1K GitHub stars | 4/5 production-readiness

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": { "url": "data/airports.csv" },
  "projection": { "type": "albersUsa" },
  "mark": "circle",
  "encoding": {
    "longitude": { "field": "longitude", "type": "quantitative" },
    "latitude": { "field": "latitude", "type": "quantitative" },
    "size": { "field": "passengers", "type": "quantitative" }
  }
}
```

**Small project:** Good for academic/research contexts. Verbose JSON spec -- for simple geo charts, Observable Plot is faster.

**Key caveats:**
- Limited map projections compared to D3-geo
- No basemaps, no tiles, no standard web map interaction
- Spec-driven means limited escape hatches
- Performance limited at 10K+ marks

---

### 5. D3-geo -- Custom Cartography Engine

Maximum control over every pixel. 150+ projections, publication-quality output. NYT, WaPo, Bloomberg.

**Quick facts:** d3-geo ~30KB gzip | Hard learning curve | d3 main ~110K GitHub stars | Stable (not actively developed since mid-2024)

```javascript
import { geoNaturalEarth1, geoPath } from 'd3-geo';
import { select } from 'd3-selection';

const projection = geoNaturalEarth1().fitSize([800, 500], geojson);
const path = geoPath(projection);

select('#map').selectAll('path')
  .data(geojson.features).join('path')
  .attr('d', path)
  .attr('fill', d => colorScale(d.properties.value));
```

**Small project:** Not recommended unless you specifically need custom projections (Dymaxion, Goode Homolosine), publication-quality SVG, or full pixel-level control. For standard choropleths, use Observable Plot.

**Key caveats:**
- Not a map library -- no zoom/pan, no tiles, no basemaps
- SVG performance ceiling at >10K features
- Steep learning curve (selections, joins, enter/update/exit)
- **Anti-pattern:** Using D3 for interactive slippy maps. Use MapLibre for that.

---

## Low Priority

### AntV L7

Chinese market geo-visualization. WebGL rendering, AMap (Gaode) basemap. Strong in Ant Group / Alibaba ecosystem.

**~4K GitHub stars.** Quick setup for Chinese market with AMap API key. Not worth the effort for international projects -- use deck.gl or ECharts.

### Highcharts Maps

Enterprise charting with commercial license. Basic map charting with professional support. Only justified when your enterprise already has a Highcharts license. For small projects, use Observable Plot or ECharts (free).

---

## Decision Flowchart

```
Do you need an interactive slippy map?
+-- YES -> Use MapLibre (see 2d-mapping.md) + a charting library
|          All-in-one dashboard -> ECharts
|          Python interop -> Plotly.js
+-- NO -> Static cartographic visualization?
    +-- Quick choropleth -> Observable Plot
    +-- Custom projections / publication -> D3-geo
    +-- Grammar-of-graphics / academic -> Vega-Lite
    +-- Dashboard with charts AND a map -> ECharts
```
