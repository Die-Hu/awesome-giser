# Dashboards & Interactive Apps

> Frameworks and platforms for building interactive geospatial dashboards and data exploration applications -- from rapid Python prototypes to enterprise-grade operational centers.

> **Quick Picks**
> - **SOTA Python**: [Streamlit](https://streamlit.io) + pydeck -- geo dashboard in <50 lines, free cloud hosting
> - **SOTA JavaScript**: [Observable Framework](https://observablehq.com/framework) -- markdown-based, static site deploy, free and open source
> - **SOTA SQL-First**: [Evidence.dev](https://evidence.dev) -- SQL + Markdown, DuckDB backend, built-in map components
> - **SOTA No-Code**: [Kepler.gl](https://kepler.gl) -- zero code, drag-and-drop, export self-contained HTML
> - **SOTA Real-Time**: [Grafana](https://grafana.com) -- Geomap panel, PostGIS source, alerting, open source

---

## Table of Contents

1. [Python Dashboards](#python-dashboards)
2. [JavaScript / TypeScript Dashboards](#javascript--typescript-dashboards)
3. [No-Code / Low-Code Platforms](#no-code--low-code-platforms)
4. [Real-Time & Operational Dashboards](#real-time--operational-dashboards)
5. [ML/AI Monitoring Dashboards](#mlai-monitoring-dashboards)
6. [Dashboard Design Patterns](#dashboard-design-patterns)
7. [Data Architecture for Dashboards](#data-architecture-for-dashboards)
8. [Deployment & Scaling](#deployment--scaling)
9. [Open Source Dashboard Templates](#open-source-dashboard-templates)
10. [Tool Comparison Matrix](#tool-comparison-matrix)

---

## Python Dashboards

### Streamlit

- **Language:** Python
- **Map Support:** pydeck, folium, streamlit-folium, `st.map` (native)
- **Key Strengths:** Rapid prototyping, minimal boilerplate, built-in widgets, free cloud hosting
- **Best For:** Data science teams, quick internal dashboards, prototyping before production
- **Docs:** [docs.streamlit.io](https://docs.streamlit.io)

#### Core Geo Components

| Component | Library | Use Case |
|-----------|---------|----------|
| `st.pydeck_chart` | pydeck | 3D extrusions, heatmaps, arc layers, deck.gl ecosystem |
| `st_folium` | streamlit-folium | Choropleth, marker clusters, drawing tools, bidirectional events |
| `st.map` | Built-in | Quick scatter plot on a map (latitude/longitude columns) |
| `st_keplergl` | streamlit-keplergl | Embed Kepler.gl explorer inside Streamlit |

#### Streamlit + pydeck Full Dashboard Recipe

```python
import streamlit as st
import pydeck as pdk
import pandas as pd
import numpy as np

st.set_page_config(page_title="Earthquake Monitor", layout="wide")
st.title("Earthquake Dashboard")

# -- Session State for cross-widget coordination --
if "selected_region" not in st.session_state:
    st.session_state.selected_region = "All"

# -- Caching: load data once, rerun on TTL expiry --
@st.cache_data(ttl=3600)
def load_earthquakes():
    url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"
    df = pd.read_csv(url)
    df["time"] = pd.to_datetime(df["time"])
    return df

df = load_earthquakes()

# -- Sidebar filters --
st.sidebar.header("Filters")
min_mag = st.sidebar.slider("Min Magnitude", 0.0, 9.0, 3.0, 0.1)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(df["time"].min().date(), df["time"].max().date()),
)

filtered = df[
    (df["mag"] >= min_mag)
    & (df["time"].dt.date >= date_range[0])
    & (df["time"].dt.date <= date_range[1])
]

# -- Layout: two columns --
col_map, col_stats = st.columns([3, 1])

with col_map:
    layer = pdk.Layer(
        "ColumnLayer",
        data=filtered,
        get_position=["longitude", "latitude"],
        get_elevation="mag * 10000",
        elevation_scale=100,
        radius=20000,
        get_fill_color="[mag * 30, 100, 200, 180]",
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {"html": "<b>Mag:</b> {mag}<br/><b>Place:</b> {place}"}
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=pdk.ViewState(
                latitude=20, longitude=0, zoom=1.5, pitch=45
            ),
            tooltip=tooltip,
        )
    )

with col_stats:
    st.metric("Total Events", len(filtered))
    st.metric("Max Magnitude", f"{filtered['mag'].max():.1f}")
    st.metric("Avg Depth (km)", f"{filtered['depth'].mean():.1f}")

# -- Linked charts below the map --
st.subheader("Magnitude Distribution")
st.bar_chart(filtered["mag"].value_counts().sort_index())

st.subheader("Events Over Time")
daily = filtered.set_index("time").resample("D").size()
st.line_chart(daily)
```

#### Streamlit + Folium with Bidirectional Events

```python
import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd

st.set_page_config(layout="wide")

@st.cache_data
def load_districts():
    return gpd.read_file("districts.geojson")

gdf = load_districts()

# -- Build folium map --
m = folium.Map(location=[39.9, 116.4], zoom_start=10, tiles="CartoDB positron")

choropleth = folium.Choropleth(
    geo_data=gdf.__geo_interface__,
    data=gdf,
    columns=["district", "population"],
    key_on="feature.properties.district",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="Population",
)
choropleth.add_to(m)

# Add GeoJson tooltips
folium.GeoJson(
    gdf,
    tooltip=folium.GeoJsonTooltip(fields=["district", "population"]),
).add_to(m)

# -- Render with bidirectional events --
output = st_folium(m, width=900, height=500, returned_objects=["last_clicked"])

# -- React to click --
if output["last_clicked"]:
    lat, lng = output["last_clicked"]["lat"], output["last_clicked"]["lng"]
    st.info(f"Clicked at: {lat:.4f}, {lng:.4f}")
    # Reverse geocode or spatial query here
```

#### Multi-Page App Structure

```
my_dashboard/
  pages/
    1_Overview.py          # st.map with summary metrics
    2_Regional_Analysis.py # Choropleth + drill-down charts
    3_Time_Series.py       # Temporal analysis with date filters
    4_Data_Explorer.py     # Raw data table with download
  Home.py                  # Landing page (streamlit run Home.py)
  utils/
    data.py                # Shared @st.cache_data loaders
    styles.py              # Custom CSS injection
```

#### Session State and Caching Best Practices

```python
# -- Session state: persist user selections across reruns --
if "zoom_level" not in st.session_state:
    st.session_state.zoom_level = 10

# -- cache_data: for serializable returns (DataFrames, GeoJSON) --
@st.cache_data(ttl=600)  # 10-minute TTL
def load_geojson(path):
    return gpd.read_file(path)

# -- cache_resource: for non-serializable objects (DB connections, models) --
@st.cache_resource
def get_db_connection():
    return sqlalchemy.create_engine("postgresql://user:pass@host/db")
```

> **Dark Arts:** Use `st.query_params` (Streamlit 1.30+) to sync filter state with URL parameters. This enables bookmarkable dashboard views and deep linking into specific filter configurations.

---

### Dash (Plotly)

- **Language:** Python (also R, Julia bindings)
- **Map Support:** Plotly.js maps (Mapbox, scattergeo, choropleth_mapbox), dash-leaflet, dash-deck
- **Key Strengths:** Production-ready, callback-based interactivity, enterprise features, extensive component ecosystem
- **Best For:** Production dashboards, enterprise deployment, complex multi-page applications
- **Docs:** [dash.plotly.com](https://dash.plotly.com)

#### Callback Architecture

Dash uses a declarative callback system where outputs react to input changes:

```
+------------------+       +------------------+       +------------------+
| Input Component  | ----> |    Callback      | ----> | Output Component |
| (dropdown, slider|       |  (Python func)   |       | (graph, map,     |
|  map click)      |       |  @app.callback   |       |  table, text)    |
+------------------+       +------------------+       +------------------+
         ^                         |
         |                         v
         +---- State (read but ----+
               does not trigger)
```

#### Dash + Plotly Mapbox Choropleth

```python
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import geopandas as gpd
import json

app = Dash(__name__)

gdf = gpd.read_file("regions.geojson")

app.layout = html.Div([
    html.H1("Regional Analysis Dashboard"),
    dcc.Dropdown(
        id="metric-dropdown",
        options=[
            {"label": "Population", "value": "population"},
            {"label": "GDP", "value": "gdp"},
            {"label": "Area (km2)", "value": "area_km2"},
        ],
        value="population",
    ),
    dcc.Graph(id="choropleth-map", style={"height": "70vh"}),
    dcc.Graph(id="bar-chart"),
])

@app.callback(
    [Output("choropleth-map", "figure"), Output("bar-chart", "figure")],
    [Input("metric-dropdown", "value")],
)
def update_dashboard(metric):
    fig_map = px.choropleth_mapbox(
        gdf,
        geojson=json.loads(gdf.to_json()),
        locations=gdf.index,
        color=metric,
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        center={"lat": 39.9, "lon": 116.4},
        zoom=5,
        opacity=0.7,
        hover_data=["name", metric],
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    fig_bar = px.bar(
        gdf.sort_values(metric, ascending=False).head(20),
        x="name", y=metric,
        title=f"Top 20 Regions by {metric.title()}",
    )
    return fig_map, fig_bar

if __name__ == "__main__":
    app.run(debug=True)
```

#### dash-leaflet for Advanced Map Interactivity

```python
import dash_leaflet as dl
from dash import Dash, html, Input, Output
import json

app = Dash(__name__)

app.layout = html.Div([
    dl.Map([
        dl.TileLayer(
            url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png"
        ),
        dl.GeoJSON(
            id="geojson-layer",
            url="/assets/regions.geojson",
            zoomToBounds=True,
            hoverStyle={"weight": 5, "color": "#666", "dashArray": ""},
        ),
    ], id="map", style={"height": "80vh"}),
    html.Div(id="feature-info"),
])

@app.callback(
    Output("feature-info", "children"),
    Input("geojson-layer", "click_feature"),
)
def show_feature_info(feature):
    if not feature:
        return "Click a region for details."
    props = feature["properties"]
    return html.Ul([html.Li(f"{k}: {v}") for k, v in props.items()])
```

#### Enterprise Deployment Pattern

```
+-------------------+     +-------------------+     +-------------------+
| Load Balancer     | --> | Gunicorn Workers  | --> | Redis / Memcached |
| (nginx / ALB)     |     | (4-8 per node)    |     | (callback cache)  |
+-------------------+     +-------------------+     +-------------------+
                                   |
                          +-------------------+
                          | PostgreSQL/PostGIS|
                          | (data backend)    |
                          +-------------------+
```

```python
# production launch with gunicorn
# gunicorn app:server --workers 4 --bind 0.0.0.0:8050

# In app.py:
app = Dash(__name__)
server = app.server  # expose Flask server for gunicorn

# Enable background callbacks for long-running queries
from dash.long_callback import DiskcacheLongCallbackManager
import diskcache

cache = diskcache.Cache("./cache")
long_callback_manager = DiskcacheLongCallbackManager(cache)
```

> **Dark Arts:** Use `dash.ctx.triggered_id` to determine which input fired a callback when multiple inputs share the same callback. Combine with `dash.no_update` to selectively skip outputs.

---

### Panel (HoloViz)

- **Language:** Python
- **Map Support:** GeoViews, hvPlot geographic, Folium, deck.gl, ipyleaflet
- **Key Strengths:** Notebook-to-dashboard pipeline, reactive pipelines, deeply integrated with HoloViz ecosystem
- **Best For:** Complex analytical dashboards, Jupyter-native workflows, scientific visualization
- **Docs:** [panel.holoviz.org](https://panel.holoviz.org)

#### GeoViews + hvPlot Geographic Dashboard

```python
import panel as pn
import geoviews as gv
import holoviews as hv
import geopandas as gpd
import hvplot.pandas  # noqa: F401 -- activates .hvplot accessor

pn.extension("tabulator")
gv.extension("bokeh")

gdf = gpd.read_file("world_cities.geojson")

# -- Reactive widgets --
continent_select = pn.widgets.Select(
    name="Continent", options=["All"] + sorted(gdf["continent"].unique().tolist())
)
pop_slider = pn.widgets.RangeSlider(
    name="Population Range", start=0, end=int(gdf["population"].max()),
    value=(100_000, int(gdf["population"].max())), step=10_000,
)

# -- Reactive pipeline --
def filtered_data(continent, pop_range):
    out = gdf.copy()
    if continent != "All":
        out = out[out["continent"] == continent]
    out = out[(out["population"] >= pop_range[0]) & (out["population"] <= pop_range[1])]
    return out

# -- Reactive plots --
@pn.depends(continent_select, pop_slider)
def geo_plot(continent, pop_range):
    subset = filtered_data(continent, pop_range)
    return subset.hvplot.points(
        "longitude", "latitude",
        geo=True, color="population", size=50,
        tiles="CartoLight", cmap="plasma",
        hover_cols=["city", "population", "continent"],
        title="World Cities",
    ).opts(width=800, height=500)

@pn.depends(continent_select, pop_slider)
def bar_plot(continent, pop_range):
    subset = filtered_data(continent, pop_range)
    top = subset.nlargest(15, "population")
    return top.hvplot.barh(
        y="city", x="population", color="continent",
        title="Top 15 Cities by Population",
    ).opts(width=400, height=400)

# -- Layout --
dashboard = pn.template.FastListTemplate(
    title="World Cities Explorer",
    sidebar=[continent_select, pop_slider],
    main=[
        pn.Row(geo_plot, bar_plot),
        pn.widgets.Tabulator(
            pn.bind(filtered_data, continent_select, pop_slider),
            page_size=20, theme="fast",
        ),
    ],
)
dashboard.servable()
```

#### Notebook-to-Dashboard Workflow

```bash
# Develop in Jupyter
jupyter lab my_analysis.ipynb

# Serve the same notebook as a dashboard (Panel reads .servable() cells)
panel serve my_analysis.ipynb --show --autoreload

# Convert to standalone WASM app (runs entirely in browser)
panel convert my_analysis.ipynb --to pyodide-worker --out ./dist
```

---

### Solara

- **Language:** Python
- **Map Support:** ipyleaflet, ipywidgets ecosystem, pydeck
- **Key Strengths:** Vue-style reactive framework, works in Jupyter and as standalone web app, same code both contexts
- **Best For:** Teams bridging Jupyter exploration and production apps
- **Docs:** [solara.dev](https://solara.dev)

#### Solara Geo Dashboard

```python
import solara
import ipyleaflet
import pandas as pd

df = pd.read_csv("stations.csv")
center = solara.reactive((39.9, 116.4))
zoom = solara.reactive(10)

@solara.component
def StationMap():
    m = ipyleaflet.Map.element(
        center=center.value, zoom=zoom.value,
        layout={"height": "500px"},
    )
    for _, row in df.iterrows():
        ipyleaflet.Marker.element(
            location=(row["lat"], row["lon"]),
            title=row["name"],
        )

@solara.component
def StationTable():
    solara.DataFrame(df, items_per_page=10)

@solara.component
def Page():
    solara.Title("Station Monitor")
    with solara.Columns([2, 1]):
        StationMap()
        StationTable()
```

```bash
# Run as standalone web app
solara run app.py

# Or use in Jupyter -- same component renders inline
```

---

### Marimo

- **Language:** Python
- **Map Support:** Any ipywidgets-compatible library (folium, pydeck, ipyleaflet), Plotly
- **Key Strengths:** Reactive notebook-as-dashboard, dependency graph execution (no hidden state), git-friendly `.py` format
- **Best For:** Reproducible analytical dashboards, notebook replacement
- **Docs:** [marimo.io](https://marimo.io)

#### Marimo Reactive Geo Notebook

```python
import marimo as mo

# Cell 1: UI elements (automatically creates reactive dependencies)
slider = mo.ui.slider(start=1, stop=9, value=3, label="Min Magnitude")
slider

# Cell 2: Data loading (reruns when slider changes because it reads slider.value)
import pandas as pd
df = pd.read_csv("earthquakes.csv")
filtered = df[df["magnitude"] >= slider.value]
mo.md(f"**{len(filtered)}** earthquakes with magnitude >= {slider.value}")

# Cell 3: Map (reruns when filtered changes)
import pydeck as pdk
layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered,
    get_position=["longitude", "latitude"],
    get_radius="magnitude * 20000",
    get_fill_color="[255, magnitude * 30, 0, 160]",
)
deck = pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(
    latitude=35, longitude=135, zoom=4,
))
deck

# Cell 4: Chart (reruns when filtered changes)
import altair as alt
chart = alt.Chart(filtered).mark_bar().encode(
    x=alt.X("magnitude:Q", bin=True),
    y="count()",
)
chart
```

```bash
# Run as notebook
marimo edit app.py

# Run as read-only dashboard
marimo run app.py

# Export to HTML
marimo export html app.py -o dashboard.html
```

---

### Python Dashboard Comparison

| Feature | Streamlit | Dash | Panel | Solara | Marimo |
|---------|-----------|------|-------|--------|--------|
| Learning Curve | Low | Medium | Medium-High | Medium | Low |
| Reactivity Model | Top-down rerun | Callbacks | Reactive params | Vue-style reactive | Dependency graph |
| Jupyter Native | No (separate process) | No | Yes | Yes | Yes (is a notebook) |
| Multi-Page | Built-in (pages/) | dash.page_registry | Built-in | Router | Single file |
| Caching | `@st.cache_data` | flask-caching | `pn.cache` | Built-in | Automatic (DAG) |
| Custom Components | Streamlit Components API | React components | Bokeh extensions | ipywidgets | ipywidgets |
| Real-Time | `st.experimental_rerun` | Interval + WebSocket | periodic callback | reactive state | Not built-in |
| WASM/Static Export | No | No | Yes (Pyodide) | No | Yes (HTML export) |
| Deployment | Streamlit Cloud (free) | Dash Enterprise | Panel Server | Solara Server | marimo cloud |
| Best Map Library | pydeck | Plotly mapbox | GeoViews/hvPlot | ipyleaflet | Any (pydeck, folium) |

---

## JavaScript / TypeScript Dashboards

### Observable Framework

- **Language:** JavaScript / Markdown
- **Map Support:** D3, Observable Plot, MapLibre GL, deck.gl
- **Key Strengths:** Reactive dataflow, data loaders (Python/SQL/shell), static site generation, free hosting
- **Best For:** Data journalism, shareable analytical reports, public-facing dashboards
- **Docs:** [observablehq.com/framework](https://observablehq.com/framework)

#### Observable Data Loader + MapLibre Dashboard

```markdown
---
title: Regional Analysis
---

# Regional Analysis

Data is loaded at build time via a Python data loader:

\`\`\`js
const regions = FileAttachment("data/regions.json").json();
\`\`\`

\`\`\`js
// Interactive filter
const minPop = view(Inputs.range([0, 10_000_000], {
  step: 100_000, label: "Min Population", value: 500_000
}));
\`\`\`

\`\`\`js
// MapLibre GL map
const container = display(html\`<div style="height:500px">\`);
const map = new maplibregl.Map({
  container,
  style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
  center: [-98, 39],
  zoom: 4,
});

map.on("load", () => {
  map.addSource("regions", {
    type: "geojson",
    data: {
      type: "FeatureCollection",
      features: regions.features.filter(
        f => f.properties.population >= minPop
      ),
    },
  });
  map.addLayer({
    id: "regions-fill",
    type: "fill",
    source: "regions",
    paint: {
      "fill-color": ["interpolate", ["linear"],
        ["get", "population"],
        0, "#ffffcc", 5000000, "#bd0026"
      ],
      "fill-opacity": 0.7,
    },
  });
});
\`\`\`

\`\`\`js
// Linked bar chart with Observable Plot
Plot.plot({
  marginLeft: 120,
  marks: [
    Plot.barX(
      regions.features
        .map(f => f.properties)
        .filter(d => d.population >= minPop)
        .sort((a, b) => b.population - a.population)
        .slice(0, 15),
      { y: "name", x: "population", fill: "steelblue", sort: { y: "-x" } }
    ),
    Plot.ruleX([0]),
  ]
})
\`\`\`
```

#### Data Loader Pattern (Python backend, JS frontend)

```
src/
  data/
    regions.json.py    # Python script that outputs JSON to stdout
    timeseries.csv.sh  # Shell script that queries DB and outputs CSV
  index.md             # Dashboard markdown with JS visualization
observablehq.config.js
```

```python
# src/data/regions.json.py -- runs at build time
import geopandas as gpd
import sys

gdf = gpd.read_file("data/raw/regions.shp")
gdf = gdf[["name", "population", "geometry"]]
sys.stdout.write(gdf.to_json())
```

> **Dark Arts:** Observable data loaders run at build time, so you can use heavy Python libraries (geopandas, scikit-learn, rasterio) for data prep without shipping them to the browser. The client only receives the processed JSON/CSV.

---

### Evidence.dev

- **Language:** Markdown + SQL
- **Map Support:** Built-in `<PointMap>`, `<AreaMap>`, `<BubbleMap>` components
- **Key Strengths:** SQL-first, DuckDB backend (zero config), version-controllable, static site deployment
- **Best For:** Analytics reports, KPI dashboards with spatial context, data teams using SQL
- **Docs:** [docs.evidence.dev](https://docs.evidence.dev)

#### Evidence Fleet Dashboard

````markdown
---
title: Fleet Operations Report
---

```sql fleet_positions
SELECT
    vehicle_id,
    lat, lng,
    speed_kmh,
    status,
    last_updated
FROM fleet_tracking
WHERE last_updated > now() - interval '2 hours'
```

```sql fleet_summary
SELECT
    status,
    count(*) as vehicle_count,
    avg(speed_kmh) as avg_speed
FROM ${fleet_positions}
GROUP BY status
```

## Live Fleet Map

<PointMap
    data={fleet_positions}
    lat=lat
    long=lng
    color=status
    colorPalette={['#22c55e', '#eab308', '#ef4444']}
    size=speed_kmh
    sizeFmt="0.0"
    tooltipType=hover
    tooltip={[
        {id: 'vehicle_id', showColumnName: true},
        {id: 'speed_kmh', showColumnName: true, fmt: '0.0'},
        {id: 'status', showColumnName: true}
    ]}
    height=450
/>

## Status Summary

<DataTable data={fleet_summary} />

<BarChart
    data={fleet_summary}
    x=status
    y=vehicle_count
    fillColor="#3b82f6"
/>
````

#### DuckDB Integration for Local Analytics

```yaml
# evidence.config.yaml
sources:
  local:
    type: duckdb
    filename: ./data/analytics.duckdb

  parquet_files:
    type: duckdb
    # DuckDB can query Parquet files directly
    filename: ":memory:"
```

```sql
-- Query Parquet files directly in Evidence SQL blocks
SELECT * FROM read_parquet('./data/spatial/*.parquet')
WHERE ST_Within(geometry, ST_GeomFromText('POLYGON((...))'))
```

---

### Retool / Appsmith

- **Type:** Low-code internal tool builders
- **Map Support:** Map widget (Mapbox-based), custom components
- **Key Strengths:** Drag-and-drop UI, direct database connections, RBAC, audit logs
- **Best For:** Internal operations tools, admin panels, CRUD applications with map context

#### Retool Geo Pattern

```
+---------------------+       +---------------------+
| Retool Canvas       |       | Data Sources        |
|                     |       |                     |
| [Map Widget]--------+------>| PostgreSQL/PostGIS  |
| [Table Widget]------+       | REST API            |
| [Form Widget]-------+       | GraphQL             |
| [Chart Widget]------+       | Google Sheets       |
+---------------------+       +---------------------+
         |
         v
  SQL Transformer:
  SELECT id, name, ST_AsGeoJSON(geom) as geojson
  FROM locations
  WHERE status = {{ dropdown1.value }}
```

| Feature | Retool | Appsmith |
|---------|--------|----------|
| Hosting | Cloud + Self-hosted | Cloud + Self-hosted (open source) |
| Map Widget | Built-in (Mapbox) | Built-in (Google Maps) |
| Custom JS | Yes | Yes |
| Git Sync | Yes | Yes |
| Pricing | Free tier + per-user | Free (open source) + Business |
| Best For | Teams with existing DBs | Open-source preference |

---

### Tremor

- **Language:** React / TypeScript
- **Map Support:** No built-in map (combine with react-map-gl or MapLibre React)
- **Key Strengths:** Beautiful dashboard components, Tailwind CSS based, chart primitives (Area, Bar, Donut, etc.)
- **Best For:** Developer-built dashboards with custom styling requirements
- **Docs:** [tremor.so](https://tremor.so)

```tsx
import { Card, Metric, Text, AreaChart, Grid } from "@tremor/react";
import Map from "react-map-gl/maplibre";

// Tremor provides the charts and KPI cards, MapLibre provides the map
export default function Dashboard({ data }) {
  return (
    <main className="p-6">
      <Grid numItems={3} className="gap-4 mb-6">
        <Card>
          <Text>Total Events</Text>
          <Metric>{data.totalEvents.toLocaleString()}</Metric>
        </Card>
        <Card>
          <Text>Active Sensors</Text>
          <Metric>{data.activeSensors}</Metric>
        </Card>
        <Card>
          <Text>Avg Response Time</Text>
          <Metric>{data.avgResponseMs}ms</Metric>
        </Card>
      </Grid>
      <div className="grid grid-cols-2 gap-4">
        <Card>
          <Map
            initialViewState={{ longitude: -122.4, latitude: 37.8, zoom: 11 }}
            style={{ height: 400 }}
            mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
          />
        </Card>
        <Card>
          <AreaChart
            data={data.timeseries}
            index="date"
            categories={["events", "alerts"]}
            colors={["blue", "red"]}
          />
        </Card>
      </div>
    </main>
  );
}
```

---

### Grafana

- **Language:** Configuration (no code for standard use), plugins in Go/TypeScript
- **Map Support:** Geomap panel (markers, heatmap, route), WMS overlay, GeoJSON layer
- **Key Strengths:** Time-series native, 100+ data source plugins, alerting, open source
- **Best For:** Real-time monitoring, DevOps, IoT, operational dashboards
- **Docs:** [grafana.com/docs](https://grafana.com/docs)

#### Geomap Panel Configuration

```json
{
  "type": "geomap",
  "title": "Sensor Locations",
  "options": {
    "view": {
      "id": "coords",
      "lat": 39.9,
      "lon": 116.4,
      "zoom": 10
    },
    "basemap": {
      "type": "carto",
      "config": { "theme": "light" }
    },
    "layers": [
      {
        "type": "markers",
        "config": {
          "style": {
            "size": { "field": "value", "min": 5, "max": 30 },
            "color": {
              "field": "status",
              "fixed": "#73BF69"
            },
            "symbol": "circle"
          }
        }
      }
    ]
  },
  "datasource": {
    "type": "postgres",
    "uid": "postgis-ds"
  },
  "targets": [
    {
      "rawSql": "SELECT name, value, status, latitude, longitude FROM sensors WHERE $__timeFilter(updated_at)",
      "format": "table"
    }
  ]
}
```

#### PostGIS Data Source Query

```sql
-- Grafana PostgreSQL data source: query PostGIS directly
SELECT
    name,
    ST_Y(geom) as latitude,
    ST_X(geom) as longitude,
    value,
    CASE
        WHEN value > 100 THEN 'critical'
        WHEN value > 50  THEN 'warning'
        ELSE 'normal'
    END as status,
    updated_at as time
FROM sensor_readings
WHERE $__timeFilter(updated_at)
ORDER BY updated_at DESC
```

> **Cross-reference:** See [../js-bindbox/framework-integration.md](../js-bindbox/framework-integration.md) and [../js-bindbox/charting-integration.md](../js-bindbox/charting-integration.md) for deeper integration patterns between JS mapping and charting libraries.

---

## No-Code / Low-Code Platforms

### Kepler.gl

- **Type:** Open-source web application and embeddable component
- **Key Features:** Drag-and-drop, 10+ layer types, time playback, filter widgets, GPU-accelerated rendering
- **Best For:** Quick data exploration without coding, sharing visual analyses
- **Export:** Self-contained HTML, JSON config, image
- **Docs:** [kepler.gl](https://kepler.gl)

#### DuckDB + Kepler.gl (Foursquare Spatial Desktop)

The Foursquare Spatial Desktop (formerly Unfolded Studio) integrates DuckDB for querying large local datasets directly in the browser:

```
+---------------------+       +---------------------+       +---------------------+
| Local Files         | ----> | DuckDB WASM         | ----> | Kepler.gl Renderer  |
| (CSV, Parquet,      |       | (in-browser SQL)    |       | (GPU layers, time   |
|  GeoJSON, Shapefile)|       |                     |       |  playback, filters) |
+---------------------+       +---------------------+       +---------------------+
```

#### Jupyter + Kepler.gl

```python
from keplergl import KeplerGl
import geopandas as gpd

gdf = gpd.read_file("taxi_trips.geojson")

map_config = {
    "version": "v1",
    "config": {
        "mapStyle": {"styleType": "dark"},
        "visState": {
            "layers": [{
                "type": "arc",
                "config": {
                    "dataId": "trips",
                    "columns": {
                        "lat0": "pickup_lat", "lng0": "pickup_lng",
                        "lat1": "dropoff_lat", "lng1": "dropoff_lng",
                    },
                    "color": [255, 153, 31],
                },
            }],
        },
    },
}

kmap = KeplerGl(height=600, data={"trips": gdf}, config=map_config)
kmap  # renders in Jupyter

# Export
kmap.save_to_html(file_name="taxi_arcs.html")
```

---

### CARTO Builder

- **Type:** Cloud platform (SaaS)
- **Key Features:** Spatial SQL (BigQuery/Snowflake/Redshift native), widget builder, data enrichment from CARTO Data Observatory, sharing and embedding
- **Best For:** Location intelligence, team collaboration, enterprise spatial analytics
- **Docs:** [docs.carto.com](https://docs.carto.com)

#### CARTO Workflow

```
+---------------------+       +---------------------+       +---------------------+
| Cloud Data          | ----> | CARTO Workspace     | ----> | CARTO Builder       |
| Warehouse           |       | (connections,       |       | (map + widgets +    |
| (BigQuery, Snowflake|       |  spatial SQL,       |       |  interactivity)     |
|  Redshift, Postgres)|       |  data enrichment)   |       |                     |
+---------------------+       +---------------------+       +---------------------+
```

Key capabilities:
- **Spatial indexes:** H3 and Quadbin tiling for massive datasets
- **Data enrichment:** Demographic, POI, and boundary data from Data Observatory
- **Widgets:** Histogram, category, formula, time-series linked to map
- **Sharing:** Public URLs, password-protected, embed iframe

---

### ArcGIS Dashboards

- **Type:** Enterprise platform (part of ArcGIS Online / Portal)
- **Key Features:** Real-time data feeds, ArcGIS ecosystem integration, configurable widgets (gauge, list, serial chart, indicator), operations view
- **Best For:** Government, utilities, emergency management, enterprise GIS operations
- **Docs:** [doc.arcgis.com/en/dashboards](https://doc.arcgis.com/en/dashboards/)

---

### Google Looker Studio

- **Type:** Free cloud platform (Google)
- **Key Features:** Google Sheets / BigQuery / 800+ connectors, map chart (bubble, filled), calculated fields, sharing via Google account
- **Best For:** Quick business dashboards, teams already in Google ecosystem, free reporting
- **Docs:** [lookerstudio.google.com](https://lookerstudio.google.com)

#### Looker Studio Geo Patterns

| Map Type | Data Requirement | Use Case |
|----------|-----------------|----------|
| Bubble Map | Lat/Lng columns | Point locations (stores, sensors) |
| Filled Map | ISO country/region codes or US state | Choropleth by geography |
| Google Maps | Lat/Lng | Street-level context |

> **Dark Arts:** Use BigQuery GIS functions (`ST_GEOGPOINT`, `ST_DISTANCE`, `ST_WITHIN`) in Looker Studio custom SQL to perform spatial queries before visualization. This offloads spatial computation to BigQuery rather than trying to do it in the dashboard.

---

### Power BI

- **Type:** Enterprise BI platform (Microsoft)
- **Map Support:** ArcGIS Maps for Power BI, Shape Map, Azure Maps, custom Mapbox visuals
- **Key Features:** DAX formula language, DirectQuery to PostGIS, row-level security, natural language Q&A
- **Best For:** Enterprise reporting, Microsoft ecosystem integration
- **Docs:** [learn.microsoft.com/power-bi](https://learn.microsoft.com/en-us/power-bi/)

#### Shape Map with TopoJSON

```
1. Import TopoJSON file (custom regions, districts, parcels)
2. Map: Shape Map visual
3. Location field: match property key in TopoJSON
4. Color saturation: numeric measure
5. DAX measure example:
   Avg Price Per SqM = AVERAGE(Properties[price]) / AVERAGE(Properties[area_sqm])
```

---

### Tableau

- **Type:** Enterprise BI platform (Salesforce)
- **Map Support:** Built-in geocoding, spatial joins, dual-axis maps, map layers, WMS connections
- **Key Features:** Spatial file support (Shapefile, GeoJSON, KML), LOD expressions for spatial aggregation, Tableau Public (free sharing)
- **Best For:** Complex visual analytics, spatial joins with business data, storytelling
- **Docs:** [help.tableau.com](https://help.tableau.com)

#### Spatial Join in Tableau

```
1. Connect to spatial file (districts.shp) and CSV (sales.csv)
2. Join: Inner join where [sales.Point] intersects [districts.Geometry]
3. Result: each sales record tagged with its containing district
4. Viz: Filled map colored by SUM(sales) per district
5. LOD expression for density:
   { FIXED [District] : SUM([Sales]) / ATTR([Area_km2]) }
```

---

### No-Code / Low-Code Comparison

| Platform | Cost | Data Limit (Free) | Custom Code | Sharing | Spatial SQL | Self-Host |
|----------|------|--------------------|-------------|---------|-------------|-----------|
| Kepler.gl | Free | Browser memory | JSON config | HTML export | No (DuckDB via Spatial Desktop) | Yes |
| CARTO Builder | Paid (free trial) | 100MB (trial) | SQL + JS SDK | URL / embed | Yes (native) | No |
| ArcGIS Dashboards | ArcGIS license | Per org plan | Arcade expressions | Portal / public | Via ArcGIS | Yes (Portal) |
| Looker Studio | Free | 100K rows/query | Calculated fields | Google account | Via BigQuery | No |
| Power BI | Free tier + Pro | 1GB dataset | DAX + R/Python | Workspace / publish | Via DirectQuery | Yes (Report Server) |
| Tableau | Paid (Public free) | 10M rows (Public) | LOD + Python/R | Tableau Server / Public | Spatial joins | Yes (Server) |

---

## Real-Time & Operational Dashboards

### Architecture Overview

```
+-------------------+     +-------------------+     +-------------------+
| Data Sources      |     | Spatial Middleware |     | Dashboard Frontend|
|                   |     |                   |     |                   |
| IoT Sensors   ----+---->| PostGIS           |---->| Grafana           |
| GPS Trackers  ----+     | (spatial storage) |     | (time-series +   |
| Weather APIs  ----+     |                   |     |  geomap panels)  |
| Social Media  ----+     | Martin / pg_tileserv     |                   |
+-------------------+     | (vector tiles)    |     | Custom Frontend   |
                          |                   |     | (MapLibre + D3)   |
                          | GeoServer         |     |                   |
                          | (WMS/WFS/OGC)     |     | Streamlit / Dash  |
                          +-------------------+     +-------------------+
                                   |
                          +-------------------+
                          | Message Broker    |
                          | (Redis Streams /  |
                          |  Kafka / MQTT)    |
                          +-------------------+
```

### WebSocket / SSE for Live Map Updates

```javascript
// Client-side: WebSocket connection for live vehicle positions
const ws = new WebSocket("wss://api.example.com/fleet/live");
const vehicleSource = map.getSource("vehicles");

ws.onmessage = (event) => {
    const positions = JSON.parse(event.data);
    // Update GeoJSON source in place -- MapLibre re-renders automatically
    vehicleSource.setData({
        type: "FeatureCollection",
        features: positions.map(v => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [v.lng, v.lat] },
            properties: { id: v.id, speed: v.speed, status: v.status },
        })),
    });
};

// Server-Sent Events alternative (simpler, HTTP-based, one-way)
const sse = new EventSource("/api/sensors/stream");
sse.addEventListener("sensor-update", (event) => {
    const reading = JSON.parse(event.data);
    updateSensorMarker(reading);
    updateTimeSeriesChart(reading);
});
```

### IoT Sensor Dashboard Patterns

#### Air Quality Monitoring

```python
# Streamlit real-time air quality dashboard
import streamlit as st
import pydeck as pdk
import psycopg2
import pandas as pd
import time

st.set_page_config(layout="wide")
placeholder = st.empty()

@st.cache_resource
def get_conn():
    return psycopg2.connect("postgresql://user:pass@host/airquality")

def fetch_latest():
    query = """
        SELECT station_id, name, lat, lng, pm25, pm10, aqi,
               updated_at
        FROM sensor_readings_latest
        WHERE updated_at > now() - interval '10 minutes'
    """
    return pd.read_sql(query, get_conn())

# Auto-refresh loop
while True:
    df = fetch_latest()
    with placeholder.container():
        col1, col2, col3 = st.columns(3)
        col1.metric("Active Stations", len(df))
        col2.metric("Avg AQI", f"{df['aqi'].mean():.0f}")
        col3.metric("Max PM2.5", f"{df['pm25'].max():.1f}")

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df,
            get_position=["lng", "lat"],
            get_radius="aqi * 50",
            get_fill_color="""[
                aqi > 150 ? 255 : aqi > 100 ? 255 : aqi > 50 ? 255 : 0,
                aqi > 150 ? 0 : aqi > 100 ? 165 : aqi > 50 ? 255 : 128,
                0, 180
            ]""",
            pickable=True,
        )
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=pdk.ViewState(
            latitude=df["lat"].mean(), longitude=df["lng"].mean(), zoom=10,
        )))
    time.sleep(30)  # refresh every 30 seconds
```

### Alerting and Geofencing Rules

```sql
-- PostGIS geofence check: vehicles outside their assigned zone
CREATE OR REPLACE FUNCTION check_geofence()
RETURNS TABLE(vehicle_id int, vehicle_name text, zone_name text, distance_m float)
AS $$
    SELECT
        v.id, v.name, z.name,
        ST_Distance(v.position::geography, z.boundary::geography)
    FROM vehicles v
    JOIN assigned_zones az ON az.vehicle_id = v.id
    JOIN zones z ON z.id = az.zone_id
    WHERE NOT ST_Within(v.position, z.boundary)
      AND v.last_updated > now() - interval '5 minutes'
$$ LANGUAGE sql;

-- Grafana alert rule: query this function every minute
-- Alert condition: any rows returned = geofence violation
```

> **Cross-reference:** See [../js-bindbox/realtime-offline-advanced.md](../js-bindbox/realtime-offline-advanced.md) for WebSocket patterns, offline sync, and real-time tile serving.

---

## ML/AI Monitoring Dashboards

### Model Performance by Geography

When deploying ML models that operate on geospatial data (land use classification, property valuation, demand forecasting), monitoring performance by region is critical:

```python
# Panel dashboard: model accuracy heatmap by region
import panel as pn
import geoviews as gv
import geopandas as gpd
import holoviews as hv

gv.extension("bokeh")

# Load regions with model metrics
regions = gpd.read_file("regions_with_metrics.geojson")
# Columns: region_name, geometry, accuracy, precision, recall, f1, n_predictions

metric_select = pn.widgets.Select(
    name="Metric", options=["accuracy", "precision", "recall", "f1"]
)

@pn.depends(metric_select)
def performance_map(metric):
    polys = gv.Polygons(regions, vdims=[metric, "region_name", "n_predictions"])
    return polys.opts(
        color=metric, cmap="RdYlGn", colorbar=True,
        tools=["hover"], width=800, height=500,
        title=f"Model {metric.title()} by Region",
    )

@pn.depends(metric_select)
def distribution_chart(metric):
    return regions.hvplot.hist(
        metric, bins=20, title=f"{metric.title()} Distribution",
        width=400, height=300, color="#3b82f6",
    )

dashboard = pn.Column(
    "## ML Model Performance Monitor",
    metric_select,
    pn.Row(performance_map, distribution_chart),
)
dashboard.servable()
```

### Data Drift Detection -- Spatial View

```
+---------------------+       +---------------------+       +---------------------+
| Training Data       |       | Drift Detector      |       | Dashboard           |
| Distribution        | ----> | (per-region stats)  | ----> | Choropleth of drift |
|                     |       |                     |       | magnitude + alerts  |
| Production Data     | ----> | KL divergence,      |       | Time-series of      |
| Distribution        |       | PSI, Wasserstein    |       | drift per region    |
+---------------------+       +---------------------+       +---------------------+
```

```python
# Compute drift per region
from scipy.stats import wasserstein_distance
import numpy as np

def compute_regional_drift(train_df, prod_df, feature, region_col="region"):
    results = []
    for region in train_df[region_col].unique():
        train_vals = train_df[train_df[region_col] == region][feature].dropna()
        prod_vals = prod_df[prod_df[region_col] == region][feature].dropna()
        if len(train_vals) > 10 and len(prod_vals) > 10:
            drift = wasserstein_distance(train_vals, prod_vals)
            results.append({"region": region, "drift": drift})
    return pd.DataFrame(results)
```

### Feature Importance Maps

```python
# Visualize which features matter most in each region
import shap
import geopandas as gpd

# After training regional models or a single model with regional slicing:
def regional_feature_importance(model, X_test, regions):
    explainer = shap.TreeExplainer(model)
    results = []
    for region in regions.unique():
        mask = X_test["region"] == region
        shap_values = explainer.shap_values(X_test[mask].drop(columns=["region"]))
        importance = np.abs(shap_values).mean(axis=0)
        top_feature = X_test.columns[np.argmax(importance)]
        results.append({
            "region": region,
            "top_feature": top_feature,
            "importance_score": importance.max(),
        })
    return pd.DataFrame(results)
```

### SOTA ML Monitoring Platforms

| Platform | Geo Support | Open Source | Key Feature |
|----------|-------------|-------------|-------------|
| [Evidently AI](https://evidentlyai.com) | Custom (via reports) | Yes | Data drift, model performance reports |
| [WhyLabs](https://whylabs.ai) | Segment by region | No (SaaS) | Statistical profiling, drift alerts |
| [Arize](https://arize.com) | Tag by geography | No (SaaS) | Embedding drift, performance tracing |
| [NannyML](https://nannyml.com) | Custom segmentation | Yes | Estimated performance without labels |
| [Prometheus + Grafana](https://prometheus.io) | Grafana Geomap | Yes | Custom metrics, alerting, dashboards |

> **Cross-reference:** See [ai-ml-visualization.md](ai-ml-visualization.md) for ML model output visualization techniques.

---

## Dashboard Design Patterns

### Layout Patterns

#### Map-Centric Layout

The map dominates the viewport. Filters and charts are overlaid or docked to the side. Best for spatial exploration where geography is the primary dimension.

```
+--------------------------------------------------------------+
| Header / Toolbar                                [Filters v]  |
+--------+-----------------------------------------------------+
| Sidebar|                                                     |
| (legend|                    MAP                               |
|  layers|              (full viewport)                        |
|  filter|                                                     |
|  list) |  +------------------+                               |
|        |  | Floating Panel   |                               |
|        |  | (chart / details)|                               |
+--------+--+------------------+-------------------------------+
```

#### Chart-Centric Layout

Charts and KPI cards take priority. A smaller map provides spatial context. Best for analytical dashboards where numeric trends matter more than geography.

```
+--------------------------------------------------------------+
| Header                                                       |
+-------------------+-------------------+----------------------+
| KPI Card 1        | KPI Card 2        | KPI Card 3           |
| Total: 12,345     | Growth: +4.2%     | Alerts: 3            |
+-------------------+-------------------+----------------------+
| Line Chart (time-series)              | Map (context)        |
|                                       | +------------------+ |
|                                       | |                  | |
+---------------------------------------+ |   Small map      | |
| Bar Chart (categories)               | |                  | |
|                                       | +------------------+ |
+---------------------------------------+----------------------+
```

#### Split-View Layout

Map and data table share equal space. Selecting a table row highlights on the map and vice versa. Best for data editing and inspection workflows.

```
+--------------------------------------------------------------+
| Header / Filters                                             |
+------------------------------+-------------------------------+
|                              |                               |
|         MAP                  |        DATA TABLE             |
|    (click to select)  <----> |    (click to pan map)         |
|                              |                               |
|                              |                               |
+------------------------------+-------------------------------+
| Status Bar: 1,234 features | Selected: Feature #42          |
+--------------------------------------------------------------+
```

#### Drill-Down Layout

Overview map leads to region detail, then individual feature detail. Uses breadcrumb navigation.

```
Level 1: Country overview   -->  Level 2: Province detail  -->  Level 3: Site detail
+----------------------+         +---------------------+        +-------------------+
| [Country choropleth] |  click  | [Province map]      | click  | [Site floor plan] |
| Summary stats below  |  --->   | Province KPIs       | --->   | Sensor readings   |
+----------------------+         +---------------------+        +-------------------+
                                  Breadcrumb: Country > Province > Site
```

### Responsive Design for Mobile

```css
/* CSS Grid responsive dashboard layout */
.dashboard-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr auto;
    gap: 16px;
    padding: 16px;
    height: 100vh;
}

.dashboard-grid .map-panel {
    grid-column: 1 / -1;  /* map spans full width on all sizes */
}

/* Mobile: stack everything vertically */
@media (max-width: 768px) {
    .dashboard-grid {
        grid-template-columns: 1fr;
    }
    .dashboard-grid .map-panel {
        height: 50vh;  /* limit map height on mobile */
    }
    .dashboard-grid .sidebar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        height: auto;
        max-height: 40vh;
        overflow-y: auto;
        z-index: 1000;
        transform: translateY(calc(100% - 48px));  /* collapsed by default */
        transition: transform 0.3s;
    }
    .dashboard-grid .sidebar.expanded {
        transform: translateY(0);
    }
}
```

Key mobile considerations:
- Map touch gestures conflict with page scroll -- use `cooperativeGestures` in MapLibre or require two-finger pan
- Collapse sidebars into bottom sheets or slide-over panels
- Reduce chart complexity (fewer data points, simpler tooltips)
- Test on real devices; browser DevTools emulation misses touch event nuances

### Filter Architecture

#### Global Filters with URL State Sync

```javascript
// URL state sync: filters are encoded in URL search params
// Enables bookmarking, sharing, and browser back/forward navigation

function getFiltersFromURL() {
    const params = new URLSearchParams(window.location.search);
    return {
        region: params.get("region") || "all",
        dateFrom: params.get("from") || "2025-01-01",
        dateTo: params.get("to") || "2025-12-31",
        minValue: parseFloat(params.get("min")) || 0,
    };
}

function setFiltersToURL(filters) {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([k, v]) => {
        if (v !== null && v !== undefined) params.set(k, v);
    });
    // Replace state without page reload
    window.history.replaceState({}, "", `?${params.toString()}`);
}

// Streamlit equivalent:
// st.query_params["region"] = selected_region
```

#### Cascading Filters

```python
# Dash cascading filters: country -> province -> city
@app.callback(
    Output("province-dropdown", "options"),
    Input("country-dropdown", "value"),
)
def update_provinces(country):
    provinces = df[df["country"] == country]["province"].unique()
    return [{"label": p, "value": p} for p in sorted(provinces)]

@app.callback(
    Output("city-dropdown", "options"),
    Input("province-dropdown", "value"),
)
def update_cities(province):
    cities = df[df["province"] == province]["city"].unique()
    return [{"label": c, "value": c} for c in sorted(cities)]
```

### Color Consistency Between Map and Charts

```python
# Define a single color scale and reuse across map and charts
import plotly.express as px

COLOR_SCALE = px.colors.sequential.Viridis
COLOR_MAP = {
    "residential": "#1f77b4",
    "commercial": "#ff7f0e",
    "industrial": "#2ca02c",
    "agricultural": "#d62728",
}

# Use same COLOR_MAP for both map fill-color and chart bar colors
# This ensures visual consistency when users cross-reference map and chart
```

> **Dark Arts:** When using diverging color scales (e.g., RdBu for positive/negative), always center the midpoint on zero or the meaningful threshold, not on the data median. A median-centered diverging scale can mislead readers when the data is skewed.

### Loading States and Skeleton Screens

```python
# Streamlit: show skeleton while data loads
import streamlit as st

with st.spinner("Loading spatial data..."):
    gdf = load_large_geodataframe()

# For individual components, use st.empty() as placeholder
map_placeholder = st.empty()
map_placeholder.info("Rendering map...")
# ... build map ...
map_placeholder.pydeck_chart(deck)
```

```javascript
// JavaScript: skeleton screen while map tiles load
function showMapSkeleton(container) {
    container.innerHTML = `
        <div class="skeleton-map" style="
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            height: 100%; border-radius: 8px;
        "></div>
    `;
}

map.on("load", () => {
    document.querySelector(".skeleton-map")?.remove();
});
```

### Accessibility (WCAG for Dashboards)

| Guideline | Implementation |
|-----------|---------------|
| Color contrast | Minimum 4.5:1 ratio for text, 3:1 for UI components. Test with [WebAIM Contrast Checker](https://webaim.org/resources/contrastchecker/) |
| Color blindness | Never use color alone to encode information. Add patterns, labels, or icons. Use colorblind-safe palettes (ColorBrewer) |
| Keyboard navigation | All interactive elements focusable via Tab. Map controls accessible via keyboard |
| Screen readers | Add ARIA labels to charts (`aria-label="Bar chart showing population by region"`). Provide data tables as alternative to visual charts |
| Text alternatives | Alt text for map screenshots. Summary text for complex visualizations |
| Motion | Respect `prefers-reduced-motion` media query. Disable animations for map transitions |
| Focus indicators | Visible focus ring on interactive map features and dashboard widgets |

```css
/* Respect user motion preferences */
@media (prefers-reduced-motion: reduce) {
    .map-container * {
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
    }
}
```

---

## Data Architecture for Dashboards

### Pre-Aggregation Strategies

Raw spatial data is often too large to query in real time. Pre-aggregation materializes common queries:

```sql
-- Materialized view: daily aggregates by district
CREATE MATERIALIZED VIEW mv_daily_district_stats AS
SELECT
    d.id AS district_id,
    d.name AS district_name,
    d.geom AS geometry,
    date_trunc('day', e.timestamp) AS date,
    count(*) AS event_count,
    avg(e.value) AS avg_value,
    max(e.value) AS max_value,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY e.value) AS p95_value
FROM events e
JOIN districts d ON ST_Within(e.geom, d.geom)
GROUP BY d.id, d.name, d.geom, date_trunc('day', e.timestamp);

-- Refresh on schedule (e.g., every hour via pg_cron)
-- SELECT cron.schedule('refresh-stats', '0 * * * *', 'REFRESH MATERIALIZED VIEW CONCURRENTLY mv_daily_district_stats');

CREATE INDEX idx_mv_district_date ON mv_daily_district_stats (district_id, date);
```

### Tile-Based Spatial Aggregation

For massive point datasets (millions of events), aggregate into spatial tiles before visualization:

```
+---------------------+       +---------------------+       +---------------------+
| Raw Point Data      |       | H3 / S2 / Quadkey   |       | Dashboard           |
| (100M+ records)     | ----> | Aggregation         | ----> | Hexbin / Grid Map   |
|                     |       | (pre-computed cells) |       | (fast render)       |
+---------------------+       +---------------------+       +---------------------+
```

```sql
-- H3 hexagonal aggregation (using h3-pg extension)
SELECT
    h3_lat_lng_to_cell(geom, 7) AS h3_index,  -- resolution 7, ~5km
    count(*) AS point_count,
    avg(value) AS avg_value,
    h3_cell_to_boundary_geometry(h3_lat_lng_to_cell(geom, 7)) AS hex_geometry
FROM events
WHERE timestamp > now() - interval '30 days'
GROUP BY h3_lat_lng_to_cell(geom, 7);
```

```python
# Python: H3 aggregation with h3-py
import h3
import pandas as pd

def aggregate_to_h3(df, lat_col, lng_col, resolution=7):
    df["h3_index"] = df.apply(
        lambda row: h3.latlng_to_cell(row[lat_col], row[lng_col], resolution),
        axis=1,
    )
    agg = df.groupby("h3_index").agg(
        count=("h3_index", "size"),
        avg_value=("value", "mean"),
    ).reset_index()
    # Add hex boundary geometry for visualization
    agg["geometry"] = agg["h3_index"].apply(
        lambda h: h3.cell_to_boundary(h)
    )
    return agg
```

### Caching Layers

```
+----------+     +----------+     +----------+     +----------+
| Database | --> | Server   | --> | CDN      | --> | Browser  |
| (PostGIS)|     | Cache    |     | (Cloud-  |     | Cache    |
|          |     | (Redis)  |     |  front)  |     | (SW +    |
|          |     |          |     |          |     |  IndexDB)|
+----------+     +----------+     +----------+     +----------+
  Source of       Seconds TTL      Minutes TTL      Hours TTL
  truth           Hot queries      Tile cache        Offline
```

```python
# Redis caching for Dash / Flask
import redis
import json
import hashlib

r = redis.Redis(host="localhost", port=6379, db=0)

def cached_query(sql, params, ttl=300):
    """Cache SQL query results in Redis for `ttl` seconds."""
    key = hashlib.md5(f"{sql}:{params}".encode()).hexdigest()
    cached = r.get(key)
    if cached:
        return json.loads(cached)
    result = execute_query(sql, params)
    r.setex(key, ttl, json.dumps(result))
    return result
```

```javascript
// Service Worker cache for map tiles and API responses
self.addEventListener("fetch", (event) => {
    const url = new URL(event.request.url);

    // Cache map tiles aggressively (they rarely change)
    if (url.pathname.includes("/tiles/")) {
        event.respondWith(
            caches.open("map-tiles-v1").then(cache =>
                cache.match(event.request).then(cached =>
                    cached || fetch(event.request).then(response => {
                        cache.put(event.request, response.clone());
                        return response;
                    })
                )
            )
        );
    }

    // Cache API responses with network-first strategy
    if (url.pathname.startsWith("/api/")) {
        event.respondWith(
            fetch(event.request)
                .then(response => {
                    const clone = response.clone();
                    caches.open("api-cache-v1").then(c => c.put(event.request, clone));
                    return response;
                })
                .catch(() => caches.match(event.request))
        );
    }
});
```

### DuckDB WASM for Client-Side Analytics

DuckDB compiled to WebAssembly enables SQL analytics directly in the browser, eliminating server round-trips for interactive filtering:

```javascript
import * as duckdb from "@duckdb/duckdb-wasm";

// Initialize DuckDB in the browser
const db = await duckdb.AsyncDuckDB.create();
await db.open();
const conn = await db.connect();

// Load Parquet file directly from URL
await conn.query(`
    CREATE TABLE events AS
    SELECT * FROM read_parquet('https://data.example.com/events.parquet')
`);

// Interactive filtering -- runs entirely client-side, no server needed
async function filterByRegion(regionId) {
    const result = await conn.query(`
        SELECT h3_index, count(*) as cnt, avg(value) as avg_val
        FROM events
        WHERE region_id = ${regionId}
        GROUP BY h3_index
    `);
    updateMapLayer(result.toArray());
}
```

> **Dark Arts:** For datasets under 500MB, DuckDB WASM often outperforms server-side queries because it eliminates network latency entirely. Load a Parquet file at page init, then all subsequent filter/aggregate operations are instant.

> **Cross-reference:** See [../tools/spatial-databases.md](../tools/spatial-databases.md) for PostGIS, DuckDB Spatial, and other database configurations.

---

## Deployment & Scaling

### Platform Deployment Guide

| Framework | Platform | Setup Complexity | Cost (Hobby) | Cost (Production) |
|-----------|----------|------------------|--------------|-------------------|
| Streamlit | Streamlit Cloud | Minimal (git push) | Free (1 app) | $250/mo (team) |
| Streamlit | HuggingFace Spaces | Minimal | Free (2 vCPU) | $9+/mo (GPU) |
| Streamlit | Docker + VM | Medium | $5/mo (DigitalOcean) | $20+/mo |
| Dash | Dash Enterprise | Medium | N/A | Contact sales |
| Dash | Docker + gunicorn | Medium | $5/mo | $20+/mo |
| Dash | Render | Low | Free (750h/mo) | $7+/mo |
| Panel | Panel Server | Medium | Self-host | Self-host |
| Panel | WASM export | Low | Free (static host) | Free (CDN) |
| Observable | Observable Cloud | Minimal | Free | $16/mo (team) |
| Observable | Vercel / Netlify | Low | Free | Free (100GB BW) |
| Evidence | Vercel / Netlify | Low | Free | Free (100GB BW) |
| Evidence | Evidence Cloud | Minimal | Free (beta) | TBD |
| Grafana | Grafana Cloud | Low | Free (3 users) | $29+/mo |
| Grafana | Docker / K8s | Medium | Self-host | Self-host |
| Kepler.gl | Any web server | None (HTML file) | Free | Free |

### Docker + Nginx Reverse Proxy

```yaml
# docker-compose.yml for a Streamlit geo dashboard
version: "3.9"

services:
  dashboard:
    build: .
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_HEADLESS=true
      - STREAMLIT_SERVER_FILE_WATCHER_TYPE=none  # disable in production
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
      - ./certs:/etc/nginx/certs
    depends_on:
      - dashboard
    restart: unless-stopped

  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: geodashboard
      POSTGRES_USER: gis
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped

volumes:
  pgdata:
```

```nginx
# nginx.conf -- reverse proxy with WebSocket support for Streamlit
server {
    listen 80;
    server_name dashboard.example.com;

    location / {
        proxy_pass http://dashboard:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_read_timeout 86400;  # long timeout for WebSocket
    }

    # Cache static assets
    location /static/ {
        proxy_pass http://dashboard:8501/static/;
        expires 7d;
        add_header Cache-Control "public, immutable";
    }
}
```

### Static Site Deployment (Observable / Evidence)

```bash
# Observable Framework: build and deploy
npx observable build          # generates dist/ with static HTML/JS
npx wrangler pages deploy dist  # deploy to Cloudflare Pages
# or: netlify deploy --prod --dir=dist
# or: vercel deploy dist

# Evidence: build and deploy
npm run build                 # generates build/ with static HTML
netlify deploy --prod --dir=build
```

### Authentication and Access Control

| Method | Complexity | Best For | Example |
|--------|-----------|----------|---------|
| HTTP Basic Auth (nginx) | Low | Internal tools | `.htpasswd` file |
| OAuth2 Proxy | Medium | SSO integration | [oauth2-proxy](https://github.com/oauth2-proxy/oauth2-proxy) in front of dashboard |
| Streamlit auth | Low | Streamlit apps | `st.experimental_user` + allow list |
| Dash Flask-Login | Medium | Dash apps | Session-based auth with user DB |
| Cloudflare Access | Low | Any web app | Zero Trust tunnel, no code changes |
| VPN / Tailscale | Low | Private access | Network-level, no app changes |

```python
# Streamlit simple auth gate (Streamlit 1.30+)
import streamlit as st
import hmac

def check_password():
    """Return True if the user has entered correct credentials."""
    def login_form():
        with st.form("credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets["passwords"][st.session_state["username"]],
        ):
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False

    if st.session_state.get("authenticated"):
        return True
    login_form()
    if "authenticated" in st.session_state and not st.session_state["authenticated"]:
        st.error("Invalid credentials")
    return False

if not check_password():
    st.stop()

# -- Dashboard content below (only runs if authenticated) --
st.title("Protected Dashboard")
```

---

## Open Source Dashboard Templates

### Production-Ready Starter Templates

| Template | Stack | Features | Link |
|----------|-------|----------|------|
| Streamlit Geo Starter | Streamlit + pydeck + GeoPandas | Map, charts, filters, multi-page | [streamlit.io/gallery](https://streamlit.io/gallery) |
| Streamlit + DuckDB | Streamlit + DuckDB + st-aggrid | SQL queries, data table, export | [github.com/streamlit/example-app-duckdb](https://github.com/streamlit) |
| Dash Leaflet Demo | Dash + dash-leaflet + PostGIS | Choropleth, click events, detail panel | [dash-leaflet.com](https://dash-leaflet.com) |
| Panel GeoViews Gallery | Panel + GeoViews + hvPlot | Notebook to dashboard, reactive | [panel.holoviz.org/gallery](https://panel.holoviz.org/gallery) |
| Kepler.gl React Embed | React + Kepler.gl + Redux | Embeddable map explorer | [github.com/keplergl/kepler.gl/examples](https://github.com/keplergl/kepler.gl/tree/master/examples) |
| Observable Geo Report | Observable + D3 + Plot | Scrollytelling, spatial analysis | [observablehq.com/collection/@observablehq/geo](https://observablehq.com/collection/@observablehq/geo) |
| Evidence Spatial | Evidence + DuckDB Spatial | SQL-based spatial dashboard | [github.com/evidence-dev/evidence](https://github.com/evidence-dev/evidence) |
| Grafana + PostGIS | Grafana + PostGIS + Martin | Real-time sensor monitoring | [github.com/grafana/grafana](https://github.com/grafana/grafana) |
| Tremor + MapLibre | React + Tremor + MapLibre GL | KPI cards + interactive map | [tremor.so/docs](https://tremor.so/docs) |

### Example Repos Worth Studying

| Repository | Description | Stars |
|------------|-------------|-------|
| [streamlit-folium](https://github.com/randyzwitch/streamlit-folium) | Bidirectional Folium integration for Streamlit | 1K+ |
| [dash-leaflet](https://github.com/thedirtyfew/dash-leaflet) | Leaflet maps for Dash with full event support | 500+ |
| [keplergl/kepler.gl](https://github.com/keplergl/kepler.gl) | GPU-powered map visualization (Uber) | 10K+ |
| [evidence-dev/evidence](https://github.com/evidence-dev/evidence) | SQL-first BI framework | 4K+ |
| [grafana/grafana](https://github.com/grafana/grafana) | Observability platform with Geomap panel | 65K+ |
| [holoviz/panel](https://github.com/holoviz/panel) | High-level app and dashboarding for Python | 4K+ |
| [marimo-team/marimo](https://github.com/marimo-team/marimo) | Reactive notebook and dashboard | 8K+ |
| [widgetti/solara](https://github.com/widgetti/solara) | React-style framework for ipywidgets | 1.5K+ |

---

## Tool Comparison Matrix

### Comprehensive Feature Comparison

| | Streamlit | Dash | Panel | Observable | Evidence | Grafana | Kepler.gl | CARTO | ArcGIS | Tableau | Power BI | Looker Studio | Retool |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Type** | Code | Code | Code | Code | Code (SQL) | Config | No-code | Low-code | No-code | Low-code | Low-code | No-code | Low-code |
| **Language** | Python | Python | Python | JS/MD | SQL/MD | Config | None | SQL | Config | GUI/LOD | DAX | GUI | JS/SQL |
| **Map Quality** | Good | Good | Good | Excellent | Basic | Good | Excellent | Excellent | Excellent | Good | Good | Basic | Basic |
| **Learning Curve** | Low | Medium | High | Medium | Low | Low | None | Low | Medium | Medium | Medium | Low | Low |
| **Real-Time** | Polling | WebSocket | Periodic | Loaders | DB poll | Native | No | Yes | Yes | Extract | Import | Extract | Yes |
| **Offline / WASM** | No | No | Pyodide | Static | Static | No | HTML export | No | No | No | No | No | No |
| **Collaboration** | Git | Git | Git | Git/Cloud | Git | JSON | JSON config | Cloud | Portal | Server | Workspace | Google | Cloud |
| **Custom Code** | Full | Full | Full | Full | SQL only | Plugins | JSON | SDK | Arcade | Calcs | DAX+R | Calcs | JS |
| **Self-Host** | Yes | Yes | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | Yes | No | Yes |
| **Free Tier** | 1 app | OSS | OSS | Free | OSS | OSS | Full | Trial | No | Public | 1GB | Full | 5 users |
| **Enterprise** | No | Yes | No | Teams | Cloud | Cloud | No | Yes | Yes | Yes | Yes | No | Yes |
| **Mobile** | Responsive | Manual | Manual | Responsive | Responsive | App | No | Yes | App | App | App | Responsive | App |

### Decision Matrix: Choosing the Right Tool

```
START
  |
  v
Do you need real-time (<1s latency)?
  YES --> Grafana (time-series native) or Custom (WebSocket + MapLibre)
  NO
  |
  v
Is your team primarily SQL-skilled?
  YES --> Evidence.dev (SQL + Markdown, static deploy)
  NO
  |
  v
Is your team primarily Python-skilled?
  YES --> How complex is the dashboard?
          Simple prototype --> Streamlit (fastest to build)
          Production + callbacks --> Dash (enterprise-ready)
          Jupyter-native + scientific --> Panel (HoloViz ecosystem)
  NO
  |
  v
Is your team primarily JavaScript-skilled?
  YES --> Observable Framework (static, data loaders) or Tremor + MapLibre (React)
  NO
  |
  v
Do you need zero code?
  YES --> Is data exploration the main goal?
          YES --> Kepler.gl (drag-and-drop, GPU rendering)
          NO --> Enterprise context?
                 YES --> ArcGIS Dashboards / CARTO
                 NO --> Looker Studio (free) / Tableau Public (free sharing)
```

### Cost Estimation (Monthly, Production Use)

| Scenario | Low Cost Option | Estimated Cost | High-End Option | Estimated Cost |
|----------|----------------|----------------|-----------------|----------------|
| Internal team dashboard (5 users) | Streamlit Cloud Free | $0 | Dash Enterprise | $500+ |
| Public-facing analytics site | Observable + Netlify | $0 | Tableau Server | $1,000+ |
| Real-time IoT monitoring (50 sensors) | Grafana OSS + Docker | $10 (VM) | Grafana Cloud Pro | $29+ |
| Enterprise BI with maps (100 users) | Power BI Free + ArcGIS | $0-500 | Tableau + ArcGIS | $5,000+ |
| Data exploration (ad-hoc) | Kepler.gl | $0 | CARTO Builder | $300+ |
| SQL analytics report | Evidence + Vercel | $0 | Looker (Google Cloud) | $5,000+ |

---

## Appendix: Quick Start Commands

```bash
# -- Streamlit --
pip install streamlit pydeck streamlit-folium geopandas
streamlit run dashboard.py

# -- Dash --
pip install dash dash-leaflet pandas geopandas
python app.py  # runs on localhost:8050

# -- Panel --
pip install panel geoviews hvplot geopandas
panel serve dashboard.py --show --autoreload

# -- Solara --
pip install solara ipyleaflet
solara run app.py

# -- Marimo --
pip install marimo pydeck altair
marimo edit dashboard.py

# -- Observable Framework --
npm create @observablehq
npm run dev  # localhost:3000

# -- Evidence --
npx degit evidence-dev/template my-report
cd my-report && npm install && npm run dev

# -- Grafana (Docker) --
docker run -d -p 3000:3000 --name grafana grafana/grafana-oss

# -- Kepler.gl (Python) --
pip install keplergl
# Use in Jupyter, no separate server needed
```

---

[Back to Visualization](README.md) | [Back to Main README](../README.md)
