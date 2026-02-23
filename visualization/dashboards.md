# Dashboards & Interactive Apps

> Frameworks and platforms for building interactive geospatial dashboards and data exploration applications.

> **Quick Picks**
> - **SOTA**: [Streamlit](https://streamlit.io) + pydeck -- Python geo dashboard in <50 lines of code
> - **Free Best**: [Observable Framework](https://observablehq.com/framework) -- markdown-based, deploys as static site, free and open source
> - **Fastest Setup**: [Kepler.gl](https://kepler.gl) -- zero code, drag-and-drop, export self-contained HTML

## Code-Based Frameworks

### Streamlit

- **Language:** Python
- **Map Support:** pydeck, folium, Streamlit native map component
- **Key Strengths:** Rapid prototyping, minimal code, built-in widgets
- **Best For:** Data science teams, quick internal dashboards

#### Streamlit + pydeck Recipe

```python
import streamlit as st
import pydeck as pdk
import pandas as pd

st.title("Earthquake Dashboard")

# Load data
df = pd.read_csv("earthquakes.csv")

# Sidebar filters
min_mag = st.sidebar.slider("Min Magnitude", 0.0, 9.0, 3.0)
filtered = df[df["magnitude"] >= min_mag]

# pydeck 3D map
layer = pdk.Layer(
    "ColumnLayer",
    data=filtered,
    get_position=["longitude", "latitude"],
    get_elevation="magnitude * 10000",
    elevation_scale=100,
    radius=20000,
    get_fill_color="[magnitude * 30, 100, 200]",
    pickable=True,
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(latitude=35, longitude=135, zoom=4, pitch=45),
))

# Linked chart
st.bar_chart(filtered["magnitude"].value_counts().sort_index())
```

#### Streamlit + Folium Recipe

```python
import streamlit as st
import folium
from streamlit_folium import st_folium

m = folium.Map(location=[39.9, 116.4], zoom_start=10)
folium.Choropleth(
    geo_data="districts.geojson",
    data=df, columns=["district", "value"],
    key_on="feature.properties.name",
    fill_color="YlOrRd"
).add_to(m)

output = st_folium(m, width=700, height=500)
# output["last_clicked"] gives click coordinates for interactivity
```

### Panel (HoloViz)

- **Language:** Python
- **Map Support:** GeoViews, hvPlot, Folium, deck.gl
- **Key Strengths:** Notebook-to-dashboard, reactive pipelines, multi-page apps
- **Best For:** Complex analytical dashboards, Jupyter-native workflows

### Dash (Plotly)

- **Language:** Python (also R, Julia)
- **Map Support:** Plotly.js maps, Mapbox integration, deck.gl
- **Key Strengths:** Production-ready, callback-based interactivity, enterprise features
- **Best For:** Production dashboards, enterprise deployment

### Observable Framework

- **Language:** JavaScript / Markdown
- **Map Support:** D3, MapLibre, deck.gl, Plot
- **Key Strengths:** Reactive dataflow, live data loaders, static site generation
- **Best For:** Data journalism, shareable analytical reports

#### Observable Geo Dashboard Example

```markdown
# Regional Analysis

\`\`\`js
const data = await FileAttachment("data/regions.geojson").json();
const map = new maplibregl.Map({
  container: display(html`<div style="height:400px">`),
  style: "https://demotiles.maplibre.org/style.json",
  center: [-98, 39], zoom: 4
});
\`\`\`

\`\`\`js
Plot.plot({
  projection: "albers-usa",
  marks: [Plot.geo(data, { fill: d => d.properties.value })]
})
\`\`\`
```

---

## Grafana + GeoServer for Real-Time Dashboards

For operational / real-time geospatial monitoring, combine Grafana with GeoServer:

- **Grafana** provides time-series charts, alerts, and the Geomap panel
- **GeoServer** serves WMS/WFS from PostGIS for live spatial data
- **Architecture:** PostGIS (live data) -> GeoServer (WMS/WFS) -> Grafana (display + alerts)

### Setup Pattern

1. **PostGIS:** Store time-stamped geospatial data (sensor readings, fleet positions)
2. **GeoServer:** Publish PostGIS tables as WMS/WFS layers
3. **Grafana:** Use Geomap panel with WMS overlay or PostgreSQL data source for direct queries
4. **Alerts:** Configure Grafana alerts on spatial conditions (e.g., vehicle outside geofence)

| Component | Role | Open Source |
|-----------|------|-------------|
| PostGIS | Spatial database | Yes |
| GeoServer | OGC tile/feature server | Yes |
| Grafana | Dashboard + alerts | Yes (OSS edition) |

---

## Evidence.dev for Markdown-Based Dashboards

[Evidence](https://evidence.dev) is a framework for building data dashboards with SQL and Markdown.

- **Language:** Markdown + SQL
- **Key Strengths:** SQL-first, version-controllable, deploys as static site, built-in chart components
- **Best For:** Analytics reports, KPI dashboards with spatial context

### Geo Dashboard Example

```markdown
# Fleet Report

\`\`\`sql fleet_positions
SELECT vehicle_id, lat, lng, speed, timestamp
FROM fleet_tracking
WHERE timestamp > now() - interval '1 hour'
\`\`\`

<PointMap
  data={fleet_positions}
  lat=lat
  long=lng
  color=speed
  colorPalette={['green', 'yellow', 'red']}
  tooltipType=hover
  tooltip={[
    {id: 'vehicle_id', showColumnName: true},
    {id: 'speed', showColumnName: true, fmt: '0.0'}
  ]}
/>
```

> Evidence connects directly to databases (PostgreSQL, DuckDB, BigQuery, Snowflake) and renders charts from SQL query results.

---

## No-Code / Low-Code Platforms

### Kepler.gl

- **Type:** Open-source web app
- **Key Features:** Drag-and-drop, multiple layer types, time playback, filter widgets
- **Best For:** Quick data exploration without any coding
- **Export:** Self-contained HTML, JSON config, image

### CARTO Builder

- **Type:** Cloud platform
- **Key Features:** Spatial SQL, widget builder, data enrichment, sharing
- **Best For:** Location intelligence, team collaboration

### ArcGIS Dashboards

- **Type:** Enterprise platform
- **Key Features:** Real-time data, ArcGIS integration, configurable widgets
- **Best For:** Enterprise/government operations dashboards

---

## Open Source Dashboard Templates

| Template | Stack | Features | Source |
|----------|-------|----------|--------|
| Streamlit geo starter | Streamlit + pydeck | Map + charts + filters | [streamlit.io/gallery](https://streamlit.io/gallery) |
| Kepler.gl embed | React + Kepler.gl | Embeddable map explorer | [github.com/keplergl/kepler.gl](https://github.com/keplergl/kepler.gl/tree/master/examples) |
| Observable geo report | Observable + D3 | Scrollytelling + maps | [observablehq.com/collection/@observablehq/geo](https://observablehq.com/collection/@observablehq/geo) |
| Evidence spatial | Evidence + DuckDB | SQL-based spatial dashboard | [github.com/evidence-dev/evidence](https://github.com/evidence-dev/evidence) |

---

## Comparison Table

| Tool | Code Required | Map Support | Real-time | Sharing | Best For |
|------|---------------|-------------|-----------|---------|----------|
| Streamlit | Low (Python) | pydeck, folium | WebSocket | Streamlit Cloud | Rapid prototyping |
| Panel | Medium (Python) | GeoViews, deck.gl | Yes | Panel Server | Complex analytics |
| Dash | Medium (Python) | Plotly maps, Mapbox | WebSocket | Dash Enterprise | Production apps |
| Observable | Medium (JS) | D3, MapLibre | Data loaders | Observable Cloud | Data journalism |
| Evidence | Low (SQL + MD) | Built-in PointMap | DB polling | Static deploy | SQL-native analytics |
| Grafana | Config (no code) | Geomap panel, WMS | Native | Grafana Cloud | Real-time monitoring |
| Kepler.gl | None | Built-in | No | Export HTML | Quick exploration |
| CARTO Builder | None / SQL | Built-in | Yes | URL sharing | Location intelligence |
| ArcGIS Dashboards | None / config | ArcGIS layers | Yes | Portal sharing | Enterprise operations |

## Deployment & Scaling

| Framework | Deploy To | Scaling Model | Cost |
|-----------|-----------|---------------|------|
| Streamlit | Streamlit Cloud, Docker, any VM | Single process (use multiple replicas) | Free tier + usage |
| Dash | Dash Enterprise, Heroku, Docker | Multi-worker (gunicorn) | Free (OSS) or Enterprise |
| Observable | Observable Cloud, any static host | Static site (CDN scales) | Free |
| Evidence | Vercel, Netlify, any static host | Static site (CDN scales) | Free |
| Grafana | Docker, Kubernetes, Grafana Cloud | Horizontal scaling | Free (OSS) or Cloud |
| Kepler.gl | Export HTML, any web server | Static file | Free |
