# Charting & Geo-Visualization Integration -- Deep Dive

> A definitive expert reference for integrating charting libraries with geospatial maps. Written from the perspective of a developer building production geo dashboards.

---

## Quick Picks

| Goal | Recommendation |
|------|---------------|
| Full cartographic control | **D3.js** (`d3-geo` + Canvas) |
| All-in-one map + charts | **Apache ECharts 5.x** |
| Fastest choropleth to screen | **Observable Plot** `Plot.geo()` |
| Python/R interop dashboard | **Plotly.js** + Dash |
| Grammar-of-graphics declarative | **Vega-Lite** |
| Chinese market / AMap basemap | **AntV L7** |
| Enterprise / commercial support | **Highcharts Maps** |
| Markdown-first geo dashboards | **Observable Framework** |
| 1M+ points on basemap | **deck.gl** (see `webgl-rendering.md`) |

---

## D3-geo -- Custom Cartography Engine

### Why D3 Still Matters

D3 is the substrate of the geo-visualization ecosystem. Most other libraries either wrap it or were inspired by it.

- **150+ projections** via `d3-geo-projection` -- no other JavaScript library comes close. Includes interrupted, composite, and unconventional projections (Dymaxion, Goode Homolosine, etc.)
- **Pixel-perfect control** over every single visual element -- stroke width, fill, clip path, graticule density
- **Publication-quality SVG output** -- export directly to print-ready SVG; used in _New York Times_, _Washington Post_ graphics
- **Foundation of the ecosystem** -- Observable Plot, Vega-Lite, and many dashboard libraries use D3 under the hood
- **Composable modules** -- you pay only for what you use; `d3-geo` alone is ~30 KB gzipped

### Performance Profile

SVG rendering tops out at roughly 10K GeoJSON features before interaction starts lagging. With `topojson-client` simplification at render time you can push to ~50K features before visual degradation matters.

For data-dense scenarios, switch to Canvas. The `d3.geoPath().context(canvasCtx)` pattern draws directly to a 2D canvas context and handles 100K+ features at 60fps because there is no DOM overhead.

```javascript
// Canvas rendering pattern -- handles 100K+ features
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const projection = d3.geoMercator().fitSize([canvas.width, canvas.height], geojson);

// The key: pass the canvas context directly to geoPath
const path = d3.geoPath(projection, ctx);

function draw(features, colorFn) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (const feature of features) {
    ctx.beginPath();
    path(feature);                        // draws to ctx
    ctx.fillStyle = colorFn(feature);
    ctx.fill();
    ctx.strokeStyle = 'rgba(255,255,255,0.4)';
    ctx.lineWidth = 0.5;
    ctx.stroke();
  }
}
```

**Pre-projection trick**: avoid runtime projection cost for static maps by pre-projecting GeoJSON once at build time or load time with `d3.geoProject()` (from `d3-geo-projection`). At render time use `d3.geoIdentity()` which is near zero cost.

```javascript
import { geoProject } from 'd3-geo-projection';

// Run once at data load time
const projected = geoProject(rawGeojson, d3.geoMercator().fitSize([960, 600], rawGeojson));

// At render time -- no projection math, just coordinate passthrough
const path = d3.geoPath(d3.geoIdentity());
svg.selectAll('path').data(projected.features).join('path').attr('d', path);
```

### Key Modules Deep Dive

**`d3-geo`** -- the core module. Projections, path generators, graticule, spherical math, antimeridian clipping. The `d3.geoPath()` function is the workhorse: given a projection and optionally a canvas context, it serializes GeoJSON geometry to SVG path `d` strings or canvas draw calls.

**`d3-geo-projection`** -- extends `d3-geo` with 150+ additional projections. Key ones for thematic maps: `geoNaturalEarth1` (world maps), `geoConicEqualArea` (US/regional), `geoAzimuthalEqualArea` (polar), `geoInterruptedMollweide` (ocean-aware world).

**`d3-scale-chromatic`** -- ColorBrewer color schemes ported to D3 scales. For geo dashboards: `schemeBlues`, `schemeRdYlGn`, `interpolateViridis`, `schemePaired`. Always use perceptually uniform schemes for continuous data.

```javascript
// Sequential choropleth with auto-domain
const color = d3.scaleSequentialQuantile()
  .domain(data.map(d => d.value).sort(d3.ascending))
  .interpolator(d3.interpolateYlOrRd);
```

**`d3-contour`** -- generates contour lines (isolines) from gridded data. Essential for temperature maps, elevation, density surfaces. Works on flat Float32Array data, not GeoJSON directly.

```javascript
import { contours } from 'd3-contour';
import { geoPath, geoIdentity } from 'd3';

// grid: Float32Array of width*height values
const c = contours().size([width, height]).thresholds(d3.range(0, 100, 5));
const paths = c(grid);

const path = geoPath(geoIdentity().scale(1));
svg.selectAll('path').data(paths).join('path')
  .attr('d', path)
  .attr('fill', d => colorScale(d.value))
  .attr('stroke', 'none')
  .attr('opacity', 0.7);
```

**`d3-delaunay`** -- Delaunay triangulation and Voronoi diagrams. The killer use case for geo dashboards: **Voronoi-based hover detection**. Instead of doing expensive point-in-polygon hit testing against thousands of circles, build a Voronoi diagram once and find the nearest site in O(log n) per mouse event.

```javascript
import { Delaunay } from 'd3-delaunay';

const points = features.map(f => projection(f.geometry.coordinates));
const delaunay = Delaunay.from(points);

canvas.addEventListener('mousemove', (e) => {
  const [mx, my] = d3.pointer(e, canvas);
  const i = delaunay.find(mx, my);     // O(log n) -- fast even for 50K points
  highlightFeature(features[i]);
});
```

**`d3-hexbin`** -- hexagonal binning for point aggregation. Groups nearby points into hexagonal cells with a radius you control. The hexbin count or sum drives cell size or color. Standard tool for dense point maps.

```javascript
import { hexbin } from 'd3-hexbin';

const hb = hexbin()
  .x(d => projection([d.lon, d.lat])[0])
  .y(d => projection([d.lon, d.lat])[1])
  .radius(20)
  .extent([[0, 0], [width, height]]);

const bins = hb(points);
svg.selectAll('path')
  .data(bins)
  .join('path')
  .attr('transform', d => `translate(${d.x},${d.y})`)
  .attr('d', hb.hexagon())
  .attr('fill', d => densityColor(d.length));
```

**`topojson-client`** -- TopoJSON encoding stores shared boundaries once, reducing file size 60-80% vs GeoJSON. Use `topojson.feature()` to extract a FeatureCollection and `topojson.mesh()` to extract borders.

```javascript
import * as topojson from 'topojson-client';

const topo = await fetch('us-counties.topojson').then(r => r.json());
const counties = topojson.feature(topo, topo.objects.counties);
// Mesh for inner borders only (not outer coastline)
const borders = topojson.mesh(topo, topo.objects.counties, (a, b) => a !== b);
```

### Advanced Patterns

#### D3 + MapLibre Overlay (Complete Pattern)

Sync a D3 SVG overlay on a MapLibre WebGL map. Features reproject on every pan/zoom.

```javascript
import maplibregl from 'maplibre-gl';
import * as d3 from 'd3';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [-98, 38.5], zoom: 4
});

map.on('load', () => {
  const container = map.getCanvasContainer();
  const svg = d3.select(container)
    .append('svg')
    .style('position', 'absolute')
    .style('top', 0).style('left', 0)
    .style('width', '100%').style('height', '100%')
    .style('pointer-events', 'none');   // let map handle pan/zoom

  // D3 custom projection that defers to MapLibre's pixel transform
  const transform = d3.geoTransform({
    point(lng, lat) {
      const { x, y } = map.project([lng, lat]);
      this.stream.point(x, y);
    }
  });
  const path = d3.geoPath(transform);

  const features = svg.selectAll('path')
    .data(geojson.features)
    .join('path')
    .attr('fill', d => colorScale(d.properties.value))
    .attr('stroke', '#fff')
    .attr('stroke-width', 0.5)
    .attr('opacity', 0.75)
    .attr('pointer-events', 'all')
    .on('click', (event, d) => showTooltip(event, d));

  function render() {
    features.attr('d', path);
  }

  render();
  map.on('move', render);
  map.on('moveend', render);
  map.on('resize', render);
});
```

#### D3 + Canvas Overlay (Faster for Many Features)

For 10K+ features, skip SVG and render to canvas. Same projection pattern, faster draw.

```javascript
map.on('load', () => {
  const canvas = document.createElement('canvas');
  canvas.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none';
  map.getCanvasContainer().appendChild(canvas);

  function resize() {
    canvas.width = map.getCanvas().width;
    canvas.height = map.getCanvas().height;
  }
  resize();

  const ctx = canvas.getContext('2d');

  const transform = d3.geoTransform({
    point(lng, lat) {
      const { x, y } = map.project([lng, lat]);
      this.stream.point(x, y);
    }
  });
  const path = d3.geoPath(transform, ctx);  // note: pass ctx here

  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const feature of geojson.features) {
      ctx.beginPath();
      path(feature);
      ctx.fillStyle = colorScale(feature.properties.value);
      ctx.globalAlpha = 0.75;
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  render();
  map.on('move', render);
  map.on('resize', () => { resize(); render(); });
});
```

#### Small Multiples (Faceted Maps)

One projection, many subsets. Useful for time series or demographic breakdowns.

```javascript
const years = [2000, 2005, 2010, 2015, 2020];
const cellW = 180, cellH = 110, cols = 3;

const svg = d3.select('#small-multiples')
  .append('svg')
  .attr('viewBox', `0 0 ${cols * cellW} ${Math.ceil(years.length / cols) * cellH}`);

years.forEach((year, i) => {
  const col = i % cols, row = Math.floor(i / cols);
  const g = svg.append('g').attr('transform', `translate(${col * cellW}, ${row * cellH})`);

  const projection = d3.geoAlbersUsa().fitSize([cellW - 10, cellH - 20], us);
  const path = d3.geoPath(projection);

  g.selectAll('path')
    .data(us.features)
    .join('path')
    .attr('d', path)
    .attr('fill', d => colorScale(dataByYearAndFips.get(`${year}-${d.id}`) ?? 0));

  g.append('text').attr('x', 5).attr('y', cellH - 5).text(year).attr('font-size', 11);
});
```

#### Animated Projection Transitions

Morph between two D3 projections using `d3.interpolate` on path `d` strings.

```javascript
const projA = d3.geoMercator().fitSize([width, height], world);
const projB = d3.geoNaturalEarth1().fitSize([width, height], world);
const pathA = d3.geoPath(projA);
const pathB = d3.geoPath(projB);

// Pre-compute both path strings
const dA = features.map(f => pathA(f));
const dB = features.map(f => pathB(f));

svg.selectAll('path').data(features)
  .transition().duration(1500).ease(d3.easeCubicInOut)
  .attrTween('d', (d, i) => d3.interpolatePath(dA[i], dB[i]));
// Requires d3-interpolate-path npm package for smooth morph
```

#### `d3.geoIdentity().fitSize()` Trick

When your data is already in screen-space coordinates (pre-projected raster tiles, canvas coordinates, SVG coordinates), use `geoIdentity` to skip all projection math. The `fitSize` / `fitExtent` call handles the scale and translation to fit a bounding box.

```javascript
// Your GeoJSON features are already in pixel coordinates (e.g., from a tile server)
const path = d3.geoPath(
  d3.geoIdentity()
    .reflectY(true)        // if y-axis is flipped (common in tile space)
    .fitSize([width, height], preProjectedGeoJSON)
);
```

---

## Apache ECharts 5.x -- All-in-One

### Architecture

ECharts uses a declarative **option object** pattern: you describe the entire chart state as a JSON-serializable configuration object and call `setOption()`. This makes it easy to serialize, restore, and diff chart configurations.

ECharts supports multiple **coordinate systems**: `cartesian2d`, `polar`, `geo`, `singleAxis`, `parallel`, `calendar`. A single chart can have multiple coordinate systems. The `geo` coordinate system accepts registered GeoJSON maps and supports pan/zoom.

### Performance Profile

ECharts uses Canvas by default, with opt-in SVG renderer.

- **Canvas (default)**: handles ~100K data points for scatter; ~200K with `large: true` PixelBuffer mode
- **SVG renderer**: better for low-point-count complex charts where you need crisp vector output or DOM accessibility; worse for dense data
- **`large: true` on scatter series**: bypasses DOM entirely, uses PixelBuffer -- renders 1M+ points but loses individual interactivity per point
- **`progressive` rendering**: `progressive: 400, progressiveThreshold: 3000` streams data in batches so the page remains responsive during initial draw
- **`sampling`**: `sampling: 'lttb'` for time series downsampling (Largest Triangle Three Buckets algorithm)

```javascript
// High-performance million-point scatter on geo
chart.setOption({
  geo: { map: 'world', roam: true },
  series: [{
    type: 'scatter',
    coordinateSystem: 'geo',
    data: millionPoints,    // [lng, lat, value] tuples
    large: true,            // PixelBuffer mode -- no per-point DOM
    largeThreshold: 2000,
    symbolSize: 3,
    itemStyle: { color: 'rgba(255, 80, 0, 0.3)' }
  }]
});
```

### Geo-Specific Features

**Map series** (choropleth): register a GeoJSON boundary file with `echarts.registerMap()`, then reference it by name. Supports `nameMap` for remapping property names to display labels.

```javascript
echarts.registerMap('us-states', usStatesGeoJSON);

chart.setOption({
  series: [{
    type: 'map',
    map: 'us-states',
    data: [{ name: 'California', value: 39.5 }, { name: 'Texas', value: 29.1 }],
    nameProperty: 'NAME',    // which GeoJSON property to match on
    emphasis: { label: { show: true }, itemStyle: { areaColor: '#fdb462' } },
    select: { itemStyle: { areaColor: '#e05c1a' } }
  }]
});
```

**Lines series with animation** -- origin-destination flow lines with animated trail effect:

```javascript
series: [{
  type: 'lines',
  coordinateSystem: 'geo',
  data: odPairs.map(d => ({
    coords: [[d.fromLng, d.fromLat], [d.toLng, d.toLat]],
    lineStyle: { color: '#ff6600', width: 1, opacity: 0.4 }
  })),
  effect: {
    show: true,
    period: 4,             // animation cycle seconds
    trailLength: 0.3,      // trail as fraction of line
    symbol: 'arrow',
    symbolSize: 6
  },
  large: true             // for 100K+ OD lines
}]
```

**effectScatter** -- pulsing animated markers, great for highlighting key cities or alert points:

```javascript
series: [{
  type: 'effectScatter',
  coordinateSystem: 'geo',
  data: hotspots.map(d => ({ name: d.name, value: [d.lng, d.lat, d.intensity] })),
  symbolSize: d => Math.sqrt(d[2]) * 4,
  rippleEffect: { period: 3, brushType: 'stroke', scale: 4 },
  itemStyle: { color: '#f00' }
}]
```

**Heatmap on geo**:

```javascript
series: [{
  type: 'heatmap',
  coordinateSystem: 'geo',
  data: points.map(d => [d.lng, d.lat, d.value]),
  pointSize: 12,
  blurSize: 25
}]
```

**Custom series** -- draw arbitrary Canvas primitives inside the geo coordinate system. Use for custom glyphs, complex markers, or anything not covered by built-in series.

### Integration Patterns

#### ECharts Geo + Brush (Geographic Selection)

ECharts has a built-in `brush` component that draws a selection rectangle or polygon on the geo. Selected features are automatically highlighted; you receive the selection in a `brushSelected` event.

```javascript
chart.setOption({
  brush: {
    toolbox: ['rect', 'polygon', 'keep', 'clear'],
    geoIndex: 0          // bind brush to the first geo component
  },
  geo: { map: 'world', roam: false },
  series: [{ type: 'scatter', coordinateSystem: 'geo', data: points }]
});

chart.on('brushSelected', (params) => {
  const selected = params.batch[0].selected[0].dataIndex;  // indices of selected points
  updateSidebarChart(selected.map(i => points[i]));
});
```

#### Linked Brushing: Histogram -> Map Highlight (Complete Code)

```javascript
import * as echarts from 'echarts';

const mapChart = echarts.init(document.getElementById('map'));
const histChart = echarts.init(document.getElementById('hist'));
echarts.registerMap('us', usGeoJSON);

const allData = rawData;  // [{ fips, name, value, lat, lng }]

mapChart.setOption({
  geo: { map: 'us', roam: true, label: { show: false } },
  visualMap: {
    type: 'continuous', min: 0, max: 100,
    inRange: { color: ['#e0f3f8', '#08519c'] },
    seriesIndex: 0
  },
  series: [{
    type: 'map', map: 'us',
    data: allData.map(d => ({ name: d.name, value: d.value }))
  }]
});

histChart.setOption({
  brush: { toolbox: ['lineX', 'clear'] },
  xAxis: { type: 'value', name: 'Value' },
  yAxis: { type: 'value', name: 'Count' },
  series: [{
    type: 'bar',
    data: histogramBins(allData.map(d => d.value), 20)
  }]
});

histChart.on('brushSelected', (params) => {
  if (!params.batch[0]) return;
  const selected = params.batch[0].selected[0];
  const selectedBinIndices = new Set(selected.dataIndex);

  // Build name set from selected histogram bins
  const [lo, hi] = getBinRange(selected.dataIndex);
  const matchedNames = new Set(allData.filter(d => d.value >= lo && d.value <= hi).map(d => d.name));

  // Re-set map data with non-selected as near-zero opacity
  mapChart.setOption({
    series: [{
      data: allData.map(d => ({
        name: d.name,
        value: d.value,
        itemStyle: matchedNames.size === 0 ? {} : {
          opacity: matchedNames.has(d.name) ? 1 : 0.1
        }
      }))
    }]
  });
});
```

#### Time Slider Animation

```javascript
chart.setOption({
  timeline: {
    data: years,
    axisType: 'category',
    autoPlay: true,
    playInterval: 1200,
    loop: true
  },
  options: years.map(year => ({
    series: [{
      type: 'map', map: 'world',
      data: dataByYear[year]
    }]
  }))
});
```

#### Drill-Down Maps

```javascript
// Click province -> load city-level data -> re-render
chart.on('click', async (params) => {
  if (params.componentType !== 'series' || params.seriesType !== 'map') return;
  const provinceName = params.name;

  if (!echarts.getMap(provinceName)) {
    const cityGeo = await fetch(`/geo/${provinceName}.json`).then(r => r.json());
    echarts.registerMap(provinceName, cityGeo);
  }

  const cityData = await fetch(`/data/${provinceName}.json`).then(r => r.json());

  chart.setOption({
    geo: { map: provinceName },
    series: [{ type: 'map', map: provinceName, data: cityData }]
  }, true);   // true = merge replace, not merge update
});
```

### Advanced Tricks

- `visualMap.controller` -- expose a color range slider so users interactively adjust thresholds; great for choropleth where the natural breaks are data-dependent
- `series[i].tooltip.formatter` -- accepts HTML string, embed a sparkline SVG or a mini bar chart inside the tooltip for in-place context
- `echarts.registerMap` + TopoJSON converter: convert TopoJSON at load time (`topojson.feature()`) before registering -- cuts GeoJSON transfer size 60-80%
- `dataset.transform` -- client-side aggregation (`filter`, `sort`, `aggregate`) before rendering; keeps raw data on the client and avoids server round trips for interactive filtering

---

## Observable Plot -- Modern Statistical Graphics

### Why Observable Plot

Observable Plot (from the Observable team, built on D3) offers a high-level **marks + scales** API. You describe what to draw, not how to draw it. Scales, axes, and legends are automatic.

For geo dashboards the payoff is enormous: a choropleth that takes 50 lines of D3 takes 5 lines of Plot.

### Performance

Plot renders to SVG. Comfortable up to ~10K marks. For denser data:

- `Plot.raster()` -- rasterizes a continuous function (or gridded data) to pixels, bypassing SVG entirely. Use for smooth density or temperature surfaces.
- `Plot.contour()` -- generates contour bands client-side from a function or gridded array. Combines well with `Plot.geo()` for overlays.
- `Plot.dot()` with `r: 1` -- small circles at 50K start to lag; use `Plot.raster()` or hexbin instead.

### Code Comparison: D3 vs Plot for the Same Choropleth

**D3 (~50 lines)**:

```javascript
const width = 960, height = 600;
const svg = d3.select('#map').append('svg').attr('viewBox', `0 0 ${width} ${height}`);
const projection = d3.geoAlbersUsa().fitSize([width, height], states);
const path = d3.geoPath(projection);
const valueById = new Map(data.map(d => [d.id, d.unemployment]));
const color = d3.scaleQuantize()
  .domain([0, d3.max(data, d => d.unemployment)])
  .range(d3.schemeBlues[7]);
svg.selectAll('path')
  .data(states.features)
  .join('path')
  .attr('d', path)
  .attr('fill', d => color(valueById.get(d.id) ?? 0))
  .attr('stroke', '#fff').attr('stroke-width', 0.5);
// Add legend...
const legend = svg.append('g').attr('transform', `translate(${width - 120}, ${height - 80})`);
// ... 30 more lines for a decent legend
```

**Observable Plot (~5 lines)**:

```javascript
import * as Plot from '@observablehq/plot';

Plot.plot({
  projection: 'albers-usa',
  color: { scheme: 'blues', legend: true, label: 'Unemployment %' },
  marks: [
    Plot.geo(states, Plot.centroid({ fill: d => unemploymentMap.get(d.id) })),
    Plot.geo(nation, { stroke: '#999', strokeWidth: 0.5 })
  ]
})
```

### Faceted Small Multiples

```javascript
Plot.plot({
  projection: { type: 'albers-usa', domain: us },
  facet: { data: electionData, x: 'year' },
  fx: { tickFormat: '' },
  marks: [
    Plot.geo(states, {
      fill: (d) => resultsByYearAndFips.get(`${facetYear}-${d.id}`) ?? 0
    })
  ]
})
// One line of faceting gives you a grid of maps -- one per year
```

### Contour + Geo Overlay

```javascript
Plot.plot({
  projection: 'mercator',
  marks: [
    Plot.geo(countries, { fill: '#e8e8e8', stroke: '#ccc' }),
    Plot.contour(gridData, {
      x: 'longitude', y: 'latitude', value: 'temperature',
      thresholds: d3.range(-20, 40, 5),
      fill: Plot.identity,
      fillOpacity: 0.5,
      stroke: 'white',
      strokeWidth: 0.5
    })
  ]
})
```

---

## Plotly.js -- Interactive Scientific Visualization

### Geo Trace Types

| Trace Type | Basemap | Max Points (comfortable) | Notes |
|------------|---------|--------------------------|-------|
| `scattergeo` | Built-in world map | ~10K | No tile basemap needed |
| `choropleth` | Built-in world/US | ~1K regions | Country/state ISO codes |
| `scattermapbox` | Mapbox/MapLibre tiles | ~100K (WebGL) | Requires token or open tiles |
| `choroplethmapbox` | Mapbox/MapLibre tiles | ~10K regions | GeoJSON boundary required |
| `densitymapbox` | Mapbox/MapLibre tiles | ~1M (GPU aggregated) | Gaussian heatmap |

### Performance Notes

- `scattergeo` uses SVG -- fine for up to 10K points; beyond that use `scattermapbox`
- `scattermapbox` is WebGL-backed via Mapbox GL; handles 100K+ points with `marker.opacity` and `marker.size` encoding
- `densitymapbox` does GPU-side Gaussian blurring -- excellent for millions of events; no per-point interactivity

```javascript
// choroplethmapbox: custom GeoJSON boundaries (not just ISO codes)
Plotly.newPlot('map', [{
  type: 'choroplethmapbox',
  geojson: customGeoJSON,          // your boundary file
  featureidkey: 'properties.fips', // which property is the join key
  locations: data.map(d => d.fips),
  z: data.map(d => d.value),
  colorscale: 'Viridis',
  marker: { opacity: 0.75 }
}], {
  mapbox: { style: 'carto-positron', zoom: 4, center: { lat: 38, lon: -97 } },
  margin: { t: 0, b: 0, l: 0, r: 0 }
});
```

### Python/R Interop

Plotly has identical APIs across JavaScript (`plotly.js`), Python (`plotly.py`), and R (`plotly`). This makes it the go-to for teams that prototype in Jupyter and deploy in web.

```python
# Python: same geo trace structure
import plotly.express as px
fig = px.choropleth_mapbox(
    df, geojson=geojson, locations='fips',
    color='unemployment', featureidkey='properties.fips',
    mapbox_style='carto-positron', zoom=4,
    center={'lat': 38, 'lon': -97}
)
fig.show()
```

**Dash for production geo dashboards**: Plotly's `dash` framework gives you Python server-side callbacks for massive dataset support. The client renders only the aggregated view; all heavy computation stays in Python/Pandas.

```python
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)
app.layout = html.Div([
    dcc.Graph(id='choropleth'),
    dcc.RangeSlider(id='year-slider', min=2000, max=2023, value=[2015, 2020])
])

@app.callback(Output('choropleth', 'figure'), Input('year-slider', 'value'))
def update_map(year_range):
    filtered = df[(df.year >= year_range[0]) & (df.year <= year_range[1])]
    # server-side aggregation -- client only gets final numbers
    return px.choropleth_mapbox(filtered.groupby('fips').mean().reset_index(), ...)
```

---

## Vega-Lite / Vega -- Grammar of Graphics

### Geographic Marks

Vega-Lite's `geoshape` mark renders GeoJSON/TopoJSON features. The `projection` property accepts any D3 projection name.

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "width": 900, "height": 500,
  "projection": { "type": "albersUsa" },
  "data": { "url": "data/us-10m.json", "format": { "type": "topojson", "feature": "counties" } },
  "transform": [{ "lookup": "id", "from": { "data": { "url": "data/unemployment.csv" }, "key": "id", "fields": ["rate"] } }],
  "mark": "geoshape",
  "encoding": { "color": { "field": "rate", "type": "quantitative", "scale": { "scheme": "blues" } } }
}
```

### Linked Multi-View

Vega-Lite's `selection` (v4) / `param` (v5) system wires interactions across views automatically. A brush on a histogram automatically filters the map.

```json
{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "params": [{ "name": "brush", "select": { "type": "interval", "encodings": ["x"] } }],
  "hconcat": [
    {
      "mark": "bar",
      "params": [{ "name": "brush", "select": "interval" }],
      "encoding": {
        "x": { "bin": true, "field": "value", "type": "quantitative" },
        "y": { "aggregate": "count" },
        "color": { "condition": { "param": "brush", "value": "steelblue" }, "value": "lightgray" }
      }
    },
    {
      "mark": "geoshape",
      "transform": [{ "filter": { "param": "brush" } }],
      "projection": { "type": "albersUsa" },
      "encoding": { "color": { "field": "value", "type": "quantitative" } }
    }
  ]
}
```

### When to Choose Vega-Lite

- Exploratory analysis in Jupyter (via `altair` for Python) or Observable
- JSON-serializable specs that can be saved, versioned, reproduced, and shared
- Auto-interaction (brush, zoom, pan, crossfilter) without writing event handlers
- Embedding charts in CMS systems that accept JSON configuration
- Not for: highly customized visual design, animation-heavy production apps

---

## AntV L7 -- Alibaba's High-Performance Geo Visualization

### What L7 Offers

L7 (Layer 7) is Alibaba's WebGL geo-visualization library. It wraps MapLibre (or AMap/Gaode) as the basemap renderer and adds a rich set of GPU-accelerated layer types. Purpose-built for Chinese market mapping contexts.

```bash
npm install @antv/l7 @antv/l7-maps
```

### Layer Types

| Layer | Use Case |
|-------|----------|
| `PointLayer` | Scatter, bubble, icon markers |
| `LineLayer` | Routes, OD lines, trajectories |
| `PolygonLayer` | Choropleth, 3D extruded polygons |
| `HeatmapLayer` | Gaussian heatmap, hexagon heatmap, grid heatmap |
| `RasterLayer` | Image tiles, satellite imagery |
| `WindLayer` | Wind field visualization |

```javascript
import { Scene, PolygonLayer, PointLayer } from '@antv/l7';
import { GaodeMap } from '@antv/l7-maps';   // or MapLibre

const scene = new Scene({
  id: 'map',
  map: new GaodeMap({ pitch: 45, style: 'dark', zoom: 10, center: [116.4, 39.9] })
});

scene.on('loaded', () => {
  // 3D extruded choropleth
  const polygonLayer = new PolygonLayer({ autoFit: true })
    .source(districtGeoJSON)
    .shape('extrude')
    .size('gdp', [0, 80000])
    .color('gdp', ['#e0f3f8', '#0570b0'])
    .style({ opacity: 0.8 });

  scene.addLayer(polygonLayer);

  // Animated trip lines (trajectory visualization)
  const tripLayer = new LineLayer({ blend: 'additive' })
    .source(tripData, { parser: { type: 'json', x: 'startLng', y: 'startLat', x1: 'endLng', y1: 'endLat' } })
    .size(1)
    .shape('arc')
    .color('speed', ['#00f', '#f00'])
    .animate({ duration: 4, interval: 0.2, trailLength: 0.4 });

  scene.addLayer(tripLayer);
});
```

### When to Use L7

- Chinese market apps where Gaode (AMap) basemap is required (GCJ-02 coordinate system)
- Need animated trip lines or trajectories -- L7's `animate()` is more polished than alternatives
- 3D column / extruded polygon choropleth without deck.gl complexity
- Wind field visualization (meteorological dashboards)
- When you want a tighter integration than deck.gl + MapLibre

---

## Highcharts Maps -- Commercial Option

### Overview

Highcharts Maps is the enterprise-grade, commercially licensed geo-visualization library. Widely deployed in financial, healthcare, and government dashboards.

```bash
npm install highcharts
```

### Key Features

- `Highcharts.mapChart()` with TopoJSON or GeoJSON boundaries
- Drill-down maps with animated zoom transition
- Bubble maps, connection maps (OD lines), dot maps
- React (`highcharts-react-official`), Angular, Vue wrappers
- Accessibility (ARIA, screen reader support) built-in
- Offline export (SVG, PNG, PDF) via `highcharts-export-server`

```javascript
import Highcharts from 'highcharts/highmaps';

Highcharts.mapChart('container', {
  chart: { map: worldTopoJSON },
  title: { text: 'World GDP per Capita' },
  mapNavigation: { enabled: true },
  colorAxis: { min: 0, stops: [[0, '#fff'], [0.5, '#7ab8d9'], [1, '#003d73']] },
  series: [{
    data: gdpData,                          // [{ 'iso-a2': 'US', value: 65000 }, ...]
    mapData: worldTopoJSON,
    joinBy: ['iso-a2', 'code'],
    name: 'GDP per Capita',
    dataLabels: { enabled: true, format: '{point.name}' },
    tooltip: { valueSuffix: ' USD' }
  }]
});
```

---

## Synchronized Multi-View Patterns -- Complete Guide

The real power of geo dashboards comes from linking the map and charts so interactions in one propagate to the other. Here are five production-ready patterns.

### Pattern 1: Click-to-Filter (MapLibre -> ECharts)

User clicks a feature on the MapLibre map; adjacent ECharts bar chart filters to that feature.

```javascript
import maplibregl from 'maplibre-gl';
import * as echarts from 'echarts';

const map = new maplibregl.Map({ container: 'map', style: '...', center: [-98, 38], zoom: 4 });
const chart = echarts.init(document.getElementById('chart'));

let selectedRegion = null;

map.on('click', 'regions-fill', (e) => {
  const feature = e.features[0];
  selectedRegion = feature.properties.fips;

  // Highlight on map using filter expression
  map.setFilter('regions-fill-highlight', ['==', ['get', 'fips'], selectedRegion]);

  // Update chart to show only selected region's time series
  const ts = timeSeriesData.filter(d => d.fips === selectedRegion);
  chart.setOption({
    xAxis: { data: ts.map(d => d.year) },
    series: [{ data: ts.map(d => d.value) }]
  });
});
```

### Pattern 2: Brushing & Linking (D3 Brush -> MapLibre Filter)

Draw a selection rectangle on a D3 histogram; MapLibre map fades out non-matching features.

```javascript
import * as d3 from 'd3';
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({ container: 'map', style: '...', center: [-98, 38], zoom: 4 });

// --- D3 Histogram with brush ---
const xScale = d3.scaleLinear().domain([0, 100]).range([0, histWidth]);
const brush = d3.brushX()
  .extent([[0, 0], [histWidth, histHeight]])
  .on('brush end', onBrush);

histSvg.append('g').attr('class', 'brush').call(brush);

function onBrush({ selection }) {
  if (!selection) {
    // Clear: show all features
    map.setFilter('points-layer', null);
    return;
  }
  const [lo, hi] = selection.map(xScale.invert);

  // MapLibre filter expression: only show points in brushed range
  map.setFilter('points-layer', [
    'all',
    ['>=', ['get', 'value'], lo],
    ['<=', ['get', 'value'], hi]
  ]);

  // Fade non-matching features in a second layer
  map.setPaintProperty('points-layer-dim', 'circle-opacity', [
    'case',
    ['all', ['>=', ['get', 'value'], lo], ['<=', ['get', 'value'], hi]],
    0,    // hide from dim layer (shown in main layer)
    0.08  // show dimmed
  ]);
}
```

### Pattern 3: Crossfilter (Multi-Dimensional)

`crossfilter2` lets multiple dimensions filter a shared dataset simultaneously. Each chart applies one dimension; all charts reflect the combined filter.

```bash
npm install crossfilter2
```

```javascript
import crossfilter from 'crossfilter2';

const cf = crossfilter(rawData);
const byValue  = cf.dimension(d => d.value);
const byRegion = cf.dimension(d => d.region);
const byYear   = cf.dimension(d => d.year);

// Each brush/select updates its own dimension filter
histogram.on('brush', (range) => {
  byValue.filterRange(range);
  redrawAll();
});

mapLibreMap.on('click', 'regions', (e) => {
  byRegion.filterExact(e.features[0].properties.name);
  redrawAll();
});

yearSlider.on('change', (year) => {
  byYear.filterExact(year);
  redrawAll();
});

function redrawAll() {
  const filtered = cf.allFiltered();
  updateMap(filtered);
  updateHistogram(filtered);
  updateTimeSeries(filtered);
  updateKPIs(filtered);
}
```

### Pattern 4: State Management (React + Zustand + MapLibre + D3)

For complex production dashboards, centralize interaction state in a Zustand store.

```javascript
// store.js
import { create } from 'zustand';

export const useDashboardStore = create((set) => ({
  selectedFips: null,
  valueRange: [0, 100],
  selectedYear: 2023,

  setSelectedFips: (fips) => set({ selectedFips: fips }),
  setValueRange: (range) => set({ valueRange: range }),
  setSelectedYear: (year) => set({ selectedYear: year }),
}));

// MapView.jsx
import { useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { useDashboardStore } from './store';

export function MapView() {
  const mapRef = useRef(null);
  const mapInstance = useRef(null);
  const { selectedFips, setSelectedFips, valueRange } = useDashboardStore();

  useEffect(() => {
    mapInstance.current = new maplibregl.Map({ container: mapRef.current, style: '...' });
    mapInstance.current.on('click', 'regions', (e) => {
      setSelectedFips(e.features[0].properties.fips);
    });
  }, []);

  // React to external state changes (e.g., histogram brush)
  useEffect(() => {
    if (!mapInstance.current?.isStyleLoaded()) return;
    mapInstance.current.setFilter('regions-fill', [
      'all',
      ['>=', ['get', 'value'], valueRange[0]],
      ['<=', ['get', 'value'], valueRange[1]]
    ]);
  }, [valueRange]);

  useEffect(() => {
    if (!mapInstance.current?.isStyleLoaded() || !selectedFips) return;
    mapInstance.current.setFilter('regions-highlight', ['==', ['get', 'fips'], selectedFips]);
  }, [selectedFips]);

  return <div ref={mapRef} style={{ width: '100%', height: '100%' }} />;
}

// ChartView.jsx
import * as echarts from 'echarts';
import { useDashboardStore } from './store';

export function ChartView() {
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const { selectedFips, setValueRange } = useDashboardStore();

  useEffect(() => {
    chartInstance.current = echarts.init(chartRef.current);
    chartInstance.current.on('brushSelected', (params) => {
      const range = extractRangeFromBrush(params);
      setValueRange(range);   // -> triggers map update via store
    });
  }, []);

  useEffect(() => {
    if (!selectedFips) return;
    // Highlight selected region in chart
    chartInstance.current?.dispatchAction({ type: 'highlight', name: selectedFips });
  }, [selectedFips]);

  return <div ref={chartRef} style={{ width: '100%', height: '100%' }} />;
}
```

### Pattern 5: Observable Reactivity

In Observable Framework, reactive cells automatically re-run when their dependencies change. No event plumbing needed.

```javascript
// dashboard.md

// Reactive input: user selects a year
const year = view(Inputs.range([2000, 2023], { step: 1, value: 2015, label: 'Year' }));

// Reactive data: auto-filters when year changes
const filtered = data.filter(d => d.year === year);

// Reactive map: re-renders when filtered changes
const mapDiv = display(document.createElement('div'));
const map = new maplibregl.Map({ container: mapDiv, style: '...' });
// ... set source data using filtered

// Reactive chart: also auto-updates
display(Plot.plot({
  marks: [
    Plot.barX(filtered, { x: 'value', y: 'region', sort: { y: '-x' } })
  ]
}));
```

---

## Comprehensive Comparison Matrix

| Library | Geo Support | Chart Types | Bundle (gzip) | TypeScript | Framework | Learning Curve | Best For |
|---------|-------------|-------------|---------------|------------|-----------|----------------|----------|
| D3.js (d3-geo) | Excellent (150+ projections) | Unlimited (custom) | ~30 KB (d3-geo only) | Full | Vanilla / any | Steep | Full cartographic control, publication maps |
| Apache ECharts 5.x | Good (registered GeoJSON) | 20+ built-in | ~300 KB | Full | Vanilla / Vue / React | Moderate | All-in-one dashboards, China market |
| Observable Plot | Good (d3-geo projections) | Statistical marks | ~80 KB | Full | Observable / vanilla | Low | Exploratory geo-stats, rapid prototyping |
| Plotly.js | Good (scattergeo, choropleth, mapbox) | 40+ trace types | ~1 MB | Partial | Vanilla / React / Python | Low-Medium | Scientific dashboards, Python/R teams |
| Vega-Lite | Good (geoshape, d3 projections) | Grammar-based | ~250 KB | Via @types | Observable / Jupyter | Medium | JSON-serializable specs, linked views |
| AntV L7 | Excellent (WebGL layers, AMap) | Geo-specific | ~400 KB | Full | Vanilla / React | Medium | China market, animated trips, 3D columns |
| Highcharts Maps | Good (TopoJSON/GeoJSON) | Rich (commercial) | ~200 KB | Full | React / Angular / Vue | Low | Enterprise, commercial support, accessibility |
| deck.gl | Excellent (WebGL, 1M+ points) | Geo layers only | ~500 KB | Full | React / vanilla | Steep | Massive data, 3D, GPU rendering |

---

## Performance Comparison for Geo Visualization

| Task | D3 (SVG) | D3 (Canvas) | ECharts | Plotly | Observable Plot |
|------|----------|-------------|---------|--------|-----------------|
| 10K point scatter on map | 30 fps (laggy) | 60 fps | 60 fps (Canvas) | 60 fps (mapbox) | 20 fps (SVG) |
| Choropleth (500 regions) | 60 fps | 60 fps | 60 fps | 60 fps | 60 fps |
| 100K OD lines | Not feasible | 30 fps | 60 fps (large: true) | Not feasible | Not feasible |
| 1M points heatmap | Not feasible | Not feasible | GPU (ECharts GL) | GPU (densitymapbox) | Not feasible |
| Time animation (100 frames) | Good (transition) | Excellent | Excellent (timeline) | Good | Poor |
| Initial render, 50 regions | Fast (DOM) | Fast | Fast | Moderate (layout) | Very fast |
| SVG export quality | Publication | N/A (Canvas) | SVG renderer option | Download PNG | Publication |

---

## Advanced Dark Arts

### D3 Canvas -> MapLibre Image Source

Render geo data to an offscreen canvas with D3, then inject it as a `ImageSource` into MapLibre for a hybrid WebGL + D3 approach. Useful for complex custom renders (e.g., contour lines, Voronoi cells) that you want to composite with WebGL layers.

```javascript
// Render complex D3 visualization to offscreen canvas
const offscreen = document.createElement('canvas');
offscreen.width = 4096; offscreen.height = 4096;
const ctx = offscreen.getContext('2d');
// ... draw with D3 geoPath to ctx ...

// Inject as MapLibre raster image source
map.addSource('d3-overlay', {
  type: 'image',
  url: offscreen.toDataURL(),
  coordinates: [[-180, 85.051129], [180, 85.051129], [180, -85.051129], [-180, -85.051129]]
});
map.addLayer({
  id: 'd3-raster',
  type: 'raster',
  source: 'd3-overlay',
  paint: { 'raster-opacity': 0.7 }
});
```

### ECharts registerMap + TopoJSON Size Reduction

TopoJSON encodes shared topology once. Converting to GeoJSON for `registerMap` wastes that. Convert at load time to save 60-80% transfer.

```javascript
import * as topojson from 'topojson-client';
import * as echarts from 'echarts';

const topo = await fetch('counties-10m.json').then(r => r.json());
// Convert on the fly -- no need to store a separate GeoJSON file
const geoJSON = topojson.feature(topo, topo.objects.counties);
echarts.registerMap('us-counties', geoJSON);
```

### Plotly + Dash: Server-Side Aggregation Pattern

For truly massive datasets (100M+ rows), the client should never see raw data. Dash callbacks run on the server; the client sees only aggregated results.

```python
@app.callback(
    Output('map', 'figure'),
    [Input('time-slider', 'value'), Input('region-dropdown', 'value')]
)
def update_map(time_range, region):
    # This runs on the server -- full Pandas/PostGIS power available
    df = query_database(time_range, region)            # returns ~500 rows
    aggregated = df.groupby('hex_id').agg({'value': 'mean'}).reset_index()
    return make_choropleth(aggregated)                 # 500 rows to client, not 100M
```

### D3 `d3-delaunay` Voronoi Hover -- Fastest Hit Detection

Standard approach: on `mousemove`, scan all N points, find closest. O(N) per event.

Voronoi approach: build Delaunay once, O(N log N). Each `mousemove` is O(log N). At 50K points the difference is 50000x faster per event.

```javascript
import { Delaunay } from 'd3-delaunay';

// Build once
const projectedPoints = features.map(f => {
  const [x, y] = projection(f.geometry.coordinates);
  return [x, y];
});
const delaunay = Delaunay.from(projectedPoints);

// O(log N) per event -- runs on every mousemove without jank
svg.on('mousemove', (event) => {
  const [mx, my] = d3.pointer(event);
  const i = delaunay.find(mx, my);
  tooltip.show(features[i]);
  highlightFeature(i);
});
```

### Observable Plot Faceted Maps: Instant Small Multiples

```javascript
// One API call -- automatic facet layout, shared scale, shared projection
Plot.plot({
  projection: { type: 'albers-usa', domain: nation },
  fx: { label: null },
  color: { scheme: 'rdylbu', pivot: 0, legend: true },
  facet: { data: statsByYear, x: 'year' },
  marks: [
    Plot.geo(states, {
      fill: (d) => {
        const row = statsByYear.find(r => r.fips === d.id && r.year === currentFacetYear);
        return row?.change ?? 0;
      },
      stroke: 'white',
      strokeWidth: 0.3
    }),
    Plot.geo(nation, { stroke: '#999', strokeWidth: 0.5 })
  ]
})
```

### ECharts `dataset.transform`: Client-Side Aggregation

Avoid server round trips for interactive filtering by using ECharts' built-in transform pipeline. Data flows from `dataset` through transforms before reaching series.

```javascript
chart.setOption({
  dataset: [
    { id: 'raw', source: rawData },
    {
      id: 'filtered',
      fromDatasetId: 'raw',
      transform: [
        { type: 'filter', config: { dimension: 'year', value: selectedYear } },
        { type: 'sort', config: { dimension: 'value', order: 'desc' } }
      ]
    },
    {
      id: 'top10',
      fromDatasetId: 'filtered',
      transform: { type: 'filter', config: { dimension: '__INDEX__', lt: 10 } }
    }
  ],
  series: [
    { type: 'map', map: 'world', datasetId: 'filtered' },
    { type: 'bar', datasetId: 'top10', encode: { x: 'name', y: 'value' } }
  ]
});
```

### Custom D3 Projection for Non-Standard Coordinate Systems

If your data is in a local CRS (e.g., EPSG:2056 Swiss LV95, or a custom game-world coordinate system), write a raw projection function. D3 projections are just `[lng, lat] -> [x, y]` functions wrapped in D3's infrastructure.

```javascript
// Custom projection: transform Swiss LV95 to screen pixels
// (Replace forward/inverse with your CRS transform)
const swissProjection = d3.geoProjection(function forward(lng, lat) {
  // Approximate LV95 forward transform (use proj4js for production)
  const E = 2600000 + (lng - 7.4386) * 76000;
  const N = 1200000 + (lat - 46.9524) * 111000;
  return [E, N];
}).scale(1).translate([0, 0]);

// Use with fitSize like any projection
swissProjection.fitSize([width, height], swissGeoJSON);
const path = d3.geoPath(swissProjection);
```

---

## Install Reference

```bash
# D3 geo stack
npm install d3 d3-geo d3-geo-projection d3-scale-chromatic d3-delaunay d3-contour d3-hexbin topojson-client

# ECharts
npm install echarts

# Observable Plot
npm install @observablehq/plot

# Plotly
npm install plotly.js-dist
# or minimal build:
npm install plotly.js-geo plotly.js-cartesian

# Vega-Lite
npm install vega vega-lite vega-embed

# AntV L7
npm install @antv/l7 @antv/l7-maps

# Highcharts Maps
npm install highcharts

# Crossfilter for multi-dimensional filtering
npm install crossfilter2

# State management
npm install zustand
```
