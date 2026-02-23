# Temporal Animation

> Tools and techniques for animating geospatial data over time to reveal patterns, trends, and movement.

> **Quick Picks**
> - **SOTA**: [deck.gl TripsLayer](https://deck.gl/docs/api-reference/geo-layers/trips-layer) -- GPU-accelerated trajectory animation at scale
> - **Free Best**: [Kepler.gl](https://kepler.gl) -- drag-and-drop time playback, zero code
> - **Fastest Setup**: [Felt](https://felt.com) -- upload data, share an animated temporal map in minutes

## Time-Series Map Animation

Animating spatial data across time steps to show change, growth, or movement.

### Approaches

| Approach | Description | Best For | Example |
|----------|-------------|----------|---------|
| **Frame-based** | Render map at each time step, cycle through frames | Discrete snapshots (annual census, monthly satellite) | QGIS Temporal Controller |
| **Interpolated** | Smoothly transition between time steps | Continuous movement (vehicle tracking) | deck.gl TripsLayer |
| **Streaming** | Continuously update with real-time data | Live dashboards (IoT, traffic) | MapLibre + WebSocket |

### Key Considerations

- **Temporal resolution:** Match animation speed to data frequency (seconds, hours, days, years)
- **Playback controls:** Play, pause, speed, scrub bar, current timestamp display
- **Legend and timestamp:** Always show the current time prominently
- **Performance:** For large time-series, pre-aggregate or use GPU rendering (deck.gl)

---

## Kepler.gl Temporal Viz

Kepler.gl provides built-in time playback for any dataset with a timestamp column.

- **Features:** Time filter widget, playback speed control, time window aggregation
- **Formats:** CSV/GeoJSON with ISO 8601 timestamps or Unix epoch
- **Best For:** Quick exploratory temporal analysis without coding

### Configuration Walkthrough

1. **Upload data** with a timestamp column (e.g., `timestamp`, `date`, `datetime`)
2. **Add a filter** -- Kepler auto-detects time columns and shows a time slider
3. **Configure playback:**
   - Drag slider handles to set time window width
   - Press play to animate through the time range
   - Adjust speed with the speed control
4. **Time window modes:**
   - *Moving window* -- shows a fixed-duration sliding window
   - *Cumulative* -- shows all data up to the current time
   - *Point* -- shows only data at the exact current timestamp
5. **Export:** Save as HTML for self-contained sharing or export frames for video

---

## deck.gl TripsLayer

Animated trail rendering for trajectory and movement data.

- **Features:** Trail length, fade effect, loop animation, timestamp-based playback
- **Data Format:** Array of waypoints with timestamps
- **Best For:** Vehicle tracking, transport flow, migration patterns

### Code Example

```javascript
import { Deck } from '@deck.gl/core';
import { TripsLayer } from '@deck.gl/geo-layers';

let currentTime = 0;

const layer = new TripsLayer({
  id: 'trips',
  data: '/data/trips.json',
  getPath: d => d.waypoints.map(wp => [wp.lng, wp.lat]),
  getTimestamps: d => d.waypoints.map(wp => wp.timestamp),
  getColor: [253, 128, 93],
  widthMinPixels: 2,
  trailLength: 200,       // trail length in timestamp units
  currentTime,
  shadowEnabled: false
});

// Animation loop
function animate() {
  currentTime += 1;
  layer.setProps({ currentTime });
  requestAnimationFrame(animate);
}
animate();
```

### Data Format Spec

```json
[
  {
    "id": "trip_001",
    "waypoints": [
      { "lng": 116.40, "lat": 39.90, "timestamp": 0 },
      { "lng": 116.42, "lat": 39.91, "timestamp": 100 },
      { "lng": 116.45, "lat": 39.93, "timestamp": 250 }
    ]
  }
]
```

---

## Felt.com for Easy Temporal Viz

[Felt](https://felt.com) is a collaborative web mapping platform with built-in temporal visualization.

- **Features:** Upload time-stamped data, automatic time slider, collaborative editing
- **Best For:** Team-based temporal mapping without code
- **Sharing:** Share via URL, embed in pages, comment and annotate

> Felt is ideal for non-technical stakeholders who need to explore temporal data. For custom animation, use deck.gl or D3.

---

## Flowmap.blue for Flow Visualization

[Flowmap.blue](https://flowmap.blue) is a dedicated tool for visualizing origin-destination (flow) data over time.

- **Features:** Animated flow lines, time filtering, clustering, dark/light themes
- **Input:** Google Sheets or CSV with origin, destination, count, and optional timestamp
- **Best For:** Commuting patterns, migration, trade flows, supply chains

### Data Format

Flowmap.blue expects three sheets/tables:

| Table | Columns |
|-------|---------|
| **Locations** | id, name, lat, lon |
| **Flows** | origin, dest, count |
| **Properties** (optional) | title, description, mapbox_token |

---

## Google Earth Engine Time Lapse

[Google Earth Engine Timelapse](https://earthengine.google.com/timelapse/) provides ready-made temporal visualizations of satellite imagery from 1984 to present.

- **Coverage:** Global, annual composites from Landsat and Sentinel-2
- **Features:** Pan, zoom, search locations, embed as iframe
- **Best For:** Demonstrating land use change, urbanization, deforestation, glacier retreat
- **Custom:** Use the Earth Engine JS API to build custom time-lapse animations from any image collection

```javascript
// Earth Engine JS API: Export time-lapse frames
var collection = ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
  .filterBounds(geometry)
  .filterDate('2015-01-01', '2024-12-31')
  .filter(ee.Filter.lt('CLOUD_COVER', 20));

// Create annual median composites
var years = ee.List.sequence(2015, 2024);
var annualComposites = years.map(function(year) {
  return collection
    .filter(ee.Filter.calendarRange(year, year, 'year'))
    .median()
    .set('year', year);
});
```

---

## QGIS Temporal Controller

QGIS includes a built-in temporal controller for animating map layers over time.

- **Features:** Frame-by-frame export, configurable time range, multiple layer sync
- **Data Sources:** Any layer with a date/time field
- **Export:** Image sequence for GIF/video creation
- **Best For:** Desktop-based temporal analysis and animation export

### Setup Guide

1. Open layer properties -> **Temporal** tab
2. Set **Configuration:** "Single Field with Date/Time"
3. Select the timestamp field
4. Open **Temporal Controller** panel (View -> Panels -> Temporal Controller)
5. Set time range, step size, and frame rate
6. Click play to preview; use the export button to save image sequence
7. Convert frames to video: `ffmpeg -framerate 10 -i frame_%04d.png -c:v libx264 output.mp4`

---

## Export Formats

| Format | Type | Interactivity | File Size | Best For |
|--------|------|---------------|-----------|----------|
| GIF | Animated image | None | Medium | Social media, simple loops |
| MP4/WebM | Video | None | Small | Presentations, embedding |
| Web animation | HTML/JS | Full | N/A (hosted) | Interactive exploration |
| Lottie | Vector animation | Limited | Small | UI-embedded animations |

### Export Workflow

| Output | Tool Chain |
|--------|-----------|
| GIF | QGIS Temporal -> PNG sequence -> `ffmpeg` or ImageMagick `convert` |
| MP4 | QGIS Temporal -> PNG sequence -> `ffmpeg -framerate 10 -i frame_%04d.png output.mp4` |
| Web | deck.gl / Kepler.gl -> Export HTML or deploy React app |
| Embed | Kepler.gl export HTML -> iframe embed |

---

## Tools Comparison

| Tool | Type | Interactivity | Export | Coding | Best For |
|------|------|---------------|--------|--------|----------|
| Kepler.gl | Web app | High | Image/Video/HTML | None | Quick exploratory analysis |
| deck.gl TripsLayer | JS library | Full | Web | JS | Custom trip animations |
| QGIS Temporal | Desktop | Playback | Image sequence | None | Desktop analysis, print |
| D3.js + transitions | JS library | Full | Web/SVG | JS | Custom animated charts |
| MapLibre + setData | JS library | Full | Web | JS | Real-time data updates |
| Felt | Web platform | Medium | Sharing link | None | Collaborative temporal maps |
| Flowmap.blue | Web tool | Medium | Image/embed | None | Origin-destination flows |
| GEE Timelapse | Web platform | Pan/zoom | Embed | Optional | Satellite time-lapse |

### Performance Guidelines

| Dataset Size | Recommended Approach |
|-------------|---------------------|
| <10k features | Any tool works; D3 transitions are smooth |
| 10k - 100k features | MapLibre with setData or deck.gl |
| 100k - 1M features | deck.gl with binary data (Arrow/Parquet) |
| >1M features | Pre-aggregate to grid/H3, then animate aggregates |
