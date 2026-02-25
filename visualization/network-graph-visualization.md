# Network & Graph Visualization

> Techniques and tools for visualizing spatial networks, knowledge graphs, social networks, transportation systems, and ecological connectivity -- turning relational data into geographic insight.

> **Quick Picks**
> - **SOTA Interactive**: [Graphistry](https://www.graphistry.com) -- GPU-accelerated graph analytics, million-node visualization in the browser
> - **SOTA Programmatic**: [OSMnx](https://osmnx.readthedocs.io) + [NetworkX](https://networkx.org) -- download, model, analyze, and visualize real-world spatial networks in Python
> - **Free Desktop**: [Gephi](https://gephi.org) -- the "Photoshop of graphs," supports GeoLayout for spatial positioning
> - **GPU Web**: [Sigma.js v3](https://www.sigmajs.org) -- WebGL-powered graph rendering with thousands of nodes at 60 fps
> - **Fastest Setup**: [vis.js](https://visjs.org) -- drop-in network visualization with zero build step
> - **Best for Flows**: [Flowmap.blue](https://www.flowmap.blue) -- instant origin-destination visualization from a Google Sheet

**Cross-references:**
[Thematic Maps](thematic-maps.md) |
[3D Visualization](3d-visualization.md) |
[Dashboards](dashboards.md) |
[Temporal Animation](temporal-animation.md) |
[Cartography & Design](cartography-design.md) |
[2D Mapping Libraries](../js-bindbox/2d-mapping.md) |
[Python Libraries](../tools/python-libraries.md) |
[Data Sources](../data-sources/)

---

## Table of Contents

| # | Section | Description |
|---|---------|-------------|
| 1 | [Spatial Network Fundamentals](#1-spatial-network-fundamentals) | Graph theory basics for GIS, spatial vs. abstract networks |
| 2 | [D3.js Force-Directed Graphs](#2-d3js-force-directed-graphs) | Force simulation, geo-constrained layouts, zoom/pan |
| 3 | [Cytoscape.js](#3-cytoscapejs) | Network analysis + visualization, spatial layout algorithms |
| 4 | [Graphistry & GPU-Accelerated Visualization](#4-graphistry--gpu-accelerated-visualization) | Million-node graphs, RAPIDS integration |
| 5 | [deck.gl Graph Layers](#5-deckgl-graph-layers) | ArcLayer for OD flows, ScatterplotLayer + LineLayer for networks |
| 6 | [Vis.js & Sigma.js](#6-visjs--sigmajs) | Lightweight graph visualization, WebGL rendering |
| 7 | [Kepler.gl Arc/Line Layers](#7-keplergl-arcline-layers) | Origin-destination visualization, flow maps |
| 8 | [Transportation Networks](#8-transportation-networks) | Road networks (OSMnx), transit (GTFS), routing visualization |
| 9 | [Utility Networks](#9-utility-networks) | Power grid, water/sewer, telecom networks |
| 10 | [Social Network Analysis](#10-social-network-analysis) | Community detection, influence mapping, event networks |
| 11 | [Ecological Networks](#11-ecological-networks) | Food webs, species interactions, habitat connectivity |
| 12 | [Supply Chain & Logistics](#12-supply-chain--logistics) | Supply chain graphs, shipping routes, warehouse networks |
| 13 | [Knowledge Graphs](#13-knowledge-graphs) | Geospatial knowledge graphs, linked data, ontology mapping |
| 14 | [Network Analysis Tools](#14-network-analysis-tools) | NetworkX, igraph, graph-tool, Gephi, Neo4j spatial |
| 15 | [Performance & Scaling](#15-performance--scaling) | WebGL rendering, LOD, edge bundling, clustering |
| 16 | [Tool Comparison Matrix](#16-tool-comparison-matrix) | Comprehensive feature comparison |

---

## 1. Spatial Network Fundamentals

Spatial networks are graphs whose nodes and edges are embedded in geographic space. Unlike abstract graphs, every node has coordinates and every edge has a real-world geometry. This constraint fundamentally changes how you visualize and analyze them.

### Graph Theory Essentials for GIS

| Concept | Definition | GIS Example |
|---------|-----------|-------------|
| **Node (vertex)** | An entity in the graph | Road intersection, transit stop, city, sensor |
| **Edge (link)** | A relationship between two nodes | Road segment, flight route, power line |
| **Degree** | Number of edges connected to a node | Intersection valence (3-way, 4-way) |
| **Weight** | Numeric value on an edge | Distance, travel time, bandwidth, flow volume |
| **Directed graph** | Edges have direction | One-way streets, river flow, supply chain |
| **Planar graph** | No edges cross (embeddable in 2D without crossings) | Road networks (mostly planar), parcels |
| **Connected component** | Maximal subgraph where every node is reachable | Isolated road islands, disconnected utility zones |
| **Betweenness centrality** | Fraction of shortest paths passing through a node/edge | Traffic bottlenecks, critical infrastructure |
| **Clustering coefficient** | How connected a node's neighbors are to each other | Grid neighborhoods vs. cul-de-sac enclaves |

### Spatial Networks vs. Abstract Networks

| Property | Spatial Network | Abstract Network |
|----------|----------------|------------------|
| **Node positions** | Fixed by real-world coordinates | Determined by layout algorithm |
| **Edge geometry** | Follows physical paths (roads, pipes, rivers) | Straight lines or curves for aesthetics |
| **Planarity** | Mostly planar (edges rarely cross in 2D) | Often highly non-planar |
| **Degree distribution** | Narrow (most intersections are 3-4 way) | Often heavy-tailed (power law) |
| **Clustering** | Constrained by geometry (nearby nodes connect) | Can connect any nodes regardless of position |
| **Distance metric** | Network distance != Euclidean distance | Graph distance only |
| **Visualization** | Map-based, preserving geography | Force-directed or hierarchical layouts |

### Key Data Structures

```python
# NetworkX: The Python standard for graph operations
import networkx as nx

# Undirected spatial graph
G = nx.Graph()
G.add_node("A", lon=-73.98, lat=40.75, name="Times Square")
G.add_node("B", lon=-73.97, lat=40.76, name="Central Park South")
G.add_edge("A", "B", length=450, travel_time=120, highway="primary")

# GeoDataFrame <-> NetworkX round-trip via OSMnx
import osmnx as ox
G = ox.graph_from_place("Manhattan, New York", network_type="drive")
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)     # graph -> GeoDataFrames
G2 = ox.graph_from_gdfs(nodes_gdf, edges_gdf)   # GeoDataFrames -> graph

# igraph: faster for large-scale graph algorithms
import igraph as ig
G_ig = ig.Graph.from_networkx(G)  # convert from NetworkX

# graph-tool: fastest pure-Python-free option (C++ core)
# import graph_tool.all as gt  # requires separate installation
```

### Primal vs. Dual Graph Representations

Road networks can be modeled two ways, and the choice fundamentally changes what analysis reveals:

| Representation | Nodes | Edges | Reveals |
|---------------|-------|-------|---------|
| **Primal** (standard) | Intersections | Road segments | Shortest paths, accessibility, betweenness |
| **Dual** (line graph) | Road segments (streets) | Shared intersections | Street continuity, named-route structure, navigation complexity |

```python
# Primal: standard OSMnx graph
G_primal = ox.graph_from_place("Barcelona, Spain", network_type="drive")

# Dual: convert to line graph (streets become nodes)
G_dual = nx.line_graph(G_primal)
# In the dual, high-degree nodes = streets that intersect many others
# This reveals "integration" -- a core concept in Space Syntax theory.
```

> **SOTA reference**: For Space Syntax analysis, see [depthmapX](https://github.com/SpaceGroupUCL/depthmapX) (free desktop tool) or [momepy](http://docs.momepy.org) (Python library for urban morphology metrics).

---

## 2. D3.js Force-Directed Graphs

Force-directed layouts simulate physical forces -- nodes repel each other (charge), edges act as springs (link force), and the system settles into an equilibrium that reveals cluster structure. When combined with geographic constraints, these layouts bridge the gap between abstract network topology and spatial reality.

### Force Simulation Architecture

D3's force simulation consists of composable forces:

| Force | Function | Default | Geo Adaptation |
|-------|----------|---------|----------------|
| `forceLink` | Spring-like pull between connected nodes | distance=30 | Set distance from real-world edge length |
| `forceManyBody` | Electrostatic charge (repulsion/attraction) | strength=-30 | Reduce for dense urban networks |
| `forceCenter` | Gravity toward a central point | center of SVG | Center on map projection center |
| `forceCollide` | Prevent node overlap | radius=1 | Scale radius by node importance |
| `forceX` / `forceY` | Pull toward x/y positions | none | **Key for geo anchoring** |
| Custom force | Any function modifying `vx`/`vy` | n/a | Geographic pull, boundary constraint |

### Geographic Anchoring: Nodes Pinned to Coordinates

```javascript
// D3 force-directed graph with geographic anchoring
const width = 960, height = 600;
const svg = d3.select("#graph").append("svg")
  .attr("width", width).attr("height", height);

// Projection: geographic coordinates -> pixel coordinates
const projection = d3.geoMercator()
  .center([116.4, 39.9])  // Beijing
  .scale(50000)
  .translate([width / 2, height / 2]);

d3.json("city_network.json").then(data => {
  // Pin nodes to geographic coordinates
  data.nodes.forEach(d => {
    const [x, y] = projection([d.lon, d.lat]);
    d.fx = x;  // fx/fy = fixed position in D3 force
    d.fy = y;
  });

  // Curved edges to avoid overlap
  const link = svg.selectAll(".link")
    .data(data.edges)
    .join("path")
    .attr("class", "link")
    .attr("fill", "none")
    .attr("stroke", "#4a90d9")
    .attr("stroke-width", d => Math.sqrt(d.weight))
    .attr("stroke-opacity", 0.6);

  const node = svg.selectAll(".node")
    .data(data.nodes)
    .join("circle")
    .attr("class", "node")
    .attr("r", d => Math.sqrt(d.population) * 0.001)
    .attr("fill", d => colorByRegion(d.region))
    .attr("stroke", "#fff")
    .attr("stroke-width", 0.5);

  const simulation = d3.forceSimulation(data.nodes)
    .force("link", d3.forceLink(data.edges).id(d => d.id))
    .on("tick", () => {
      link.attr("d", linkArc);
      node.attr("cx", d => d.x).attr("cy", d => d.y);
    });

  function linkArc(d) {
    const dx = d.target.x - d.source.x;
    const dy = d.target.y - d.source.y;
    const dr = Math.sqrt(dx * dx + dy * dy) * 1.5;
    return `M${d.source.x},${d.source.y}A${dr},${dr} 0 0,1 ${d.target.x},${d.target.y}`;
  }
});
```

### Soft Anchoring: Custom Geographic Force

When you want topology to influence layout while preserving approximate geography:

```javascript
// Custom force that pulls nodes toward their geographic positions
function forceGeographic(projection, strength = 0.1) {
  let nodes;

  function force(alpha) {
    for (const node of nodes) {
      const [tx, ty] = projection([node.lon, node.lat]);
      node.vx += (tx - node.x) * strength * alpha;
      node.vy += (ty - node.y) * strength * alpha;
    }
  }

  force.initialize = function(_nodes) { nodes = _nodes; };
  force.strength = function(s) { strength = s; return force; };
  return force;
}

const simulation = d3.forceSimulation(nodes)
  .force("link", d3.forceLink(edges).distance(50))
  .force("charge", d3.forceManyBody().strength(-100))
  .force("geographic", forceGeographic(projection, 0.3))
  .force("collide", d3.forceCollide().radius(8));
```

### Animated Geographic-to-Topological Transition

```javascript
// Toggle between geographic layout and force layout
let isGeographic = true;

function toggleLayout() {
  isGeographic = !isGeographic;

  if (isGeographic) {
    nodes.forEach(d => {
      const [x, y] = projection([d.lon, d.lat]);
      d.fx = x; d.fy = y;
    });
  } else {
    // Release nodes, let force simulation take over
    nodes.forEach(d => { d.fx = null; d.fy = null; });
    simulation.alpha(0.8).restart();
  }
}
// This transition reveals hidden community structure that
// geography alone may obscure.
```

### Zoom and Pan with Semantic Zoom

```javascript
// Semantic zoom: show more detail at higher zoom levels
const zoom = d3.zoom()
  .scaleExtent([0.5, 20])
  .on("zoom", (event) => {
    const { transform } = event;
    // Geometric zoom for edges
    linkGroup.attr("transform", transform);
    // Semantic zoom for nodes: keep readable size
    nodeGroup.attr("transform", transform);
    nodeGroup.selectAll("circle")
      .attr("r", d => baseRadius(d) / transform.k);
    // Show labels only when zoomed in
    labelGroup.attr("transform", transform)
      .style("display", transform.k > 3 ? "block" : "none");
    labelGroup.selectAll("text")
      .style("font-size", `${12 / transform.k}px`);
  });

svg.call(zoom);
```

### D3 + Leaflet/MapLibre Integration

```javascript
// Overlay D3 force graph on a Leaflet map
const map = L.map("map").setView([39.9, 116.4], 10);
L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png")
  .addTo(map);

// SVG overlay
const svgOverlay = d3.select(map.getPanes().overlayPane).append("svg");
const g = svgOverlay.append("g").attr("class", "leaflet-zoom-hide");

function projectPoint(lon, lat) {
  const point = map.latLngToLayerPoint(L.latLng(lat, lon));
  return [point.x, point.y];
}

// Reproject on map zoom/pan
map.on("zoomend moveend", () => {
  data.nodes.forEach(d => {
    const [x, y] = projectPoint(d.lon, d.lat);
    d.fx = x; d.fy = y;
  });
  simulation.alpha(0.3).restart();
});
```

> **SOTA**: [D3.js v7](https://d3js.org) (2021+) uses ES modules natively. For new projects, import forces individually: `import { forceSimulation, forceLink } from "d3-force"`.

---

## 3. Cytoscape.js

[Cytoscape.js](https://js.cytoscape.org) is a graph theory library for visualization and analysis. Originally built for bioinformatics, it has become a general-purpose network visualization engine with strong layout algorithms and an extension ecosystem.

### Why Cytoscape.js for GIS

| Feature | Benefit for GIS |
|---------|----------------|
| 30+ layout algorithms | Preset (fixed coordinates), concentric, breadthfirst -- all useful for spatial networks |
| Graph analysis built-in | Centrality, shortest path, clustering -- no separate library needed |
| Extensions | `cytoscape-leaflet` for map overlay, `cytoscape-cola` for constraint-based layout |
| SBGN support | Standardized biological notation -- useful for ecological networks |
| Export | PNG, SVG, JSON -- straightforward publication workflows |

### Basic Spatial Network with Preset Layout

```javascript
import cytoscape from "cytoscape";

const cy = cytoscape({
  container: document.getElementById("cy"),
  elements: [
    // Nodes with geographic positions (projected to screen coords)
    { data: { id: "station_a", label: "Dongzhimen", type: "metro" },
      position: { x: 520, y: 180 } },
    { data: { id: "station_b", label: "Guomao", type: "metro" },
      position: { x: 580, y: 320 } },
    { data: { id: "station_c", label: "Xizhimen", type: "metro" },
      position: { x: 320, y: 200 } },
    // Edges
    { data: { source: "station_a", target: "station_b", line: "Line 2" } },
    { data: { source: "station_a", target: "station_c", line: "Line 13" } },
  ],
  layout: { name: "preset" },  // use node.position coordinates
  style: [
    { selector: "node",
      style: {
        "background-color": d => d.data("type") === "metro" ? "#e74c3c" : "#3498db",
        "label": "data(label)",
        "font-size": "10px",
        "text-valign": "bottom",
        "width": 20, "height": 20
      }
    },
    { selector: "edge",
      style: {
        "line-color": "#7f8c8d",
        "width": 2,
        "curve-style": "bezier",
        "label": "data(line)",
        "font-size": "8px"
      }
    }
  ]
});
```

### Cytoscape.js + Leaflet Overlay

```javascript
import cytoscape from "cytoscape";
import leaflet from "cytoscape-leaf";  // cytoscape-leaflet extension
cytoscape.use(leaflet);

const cy = cytoscape({ container: document.getElementById("cy"), /* ... */ });

// Enable Leaflet underlay
const leaf = cy.leaflet({
  container: document.createElement("div"),
  tileLayer: {
    url: "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png"
  }
});
```

### Graph Analysis: Centrality and Shortest Path

```javascript
// Betweenness centrality (identifies critical nodes)
const bc = cy.elements().betweennessCentrality({
  weight: e => e.data("travel_time")
});
cy.nodes().forEach(node => {
  const centrality = bc.betweenness(node);
  node.style("width", 10 + centrality * 100);
  node.style("height", 10 + centrality * 100);
});

// Shortest path between two stations
const dijkstra = cy.elements().dijkstra({
  root: "#station_a",
  weight: e => e.data("travel_time")
});
const pathToC = dijkstra.pathTo(cy.getElementById("station_c"));
pathToC.style({ "line-color": "#f1c40f", "width": 4 });
```

> **SOTA**: [Cytoscape.js 3.x](https://js.cytoscape.org) supports WebGL rendering via the `cytoscape-canvas` extension for graphs exceeding 10K nodes. For desktop analysis, use [Cytoscape Desktop](https://cytoscape.org) (Java-based, supports plugins for GeoSPARQL import).

---

## 4. Graphistry & GPU-Accelerated Visualization

[Graphistry](https://www.graphistry.com) is a GPU-accelerated graph visualization platform that renders million-node graphs interactively in the browser. It combines NVIDIA GPU compute with a purpose-built visual analytics engine.

### When You Need Graphistry

| Scenario | Scale | Why Graphistry |
|----------|-------|---------------|
| Cybersecurity event graphs | 1M+ edges | Real-time filtering of network traffic graphs |
| Social media analysis | 500K+ nodes | Community detection at scale |
| Supply chain mapping | 100K+ edges | End-to-end visibility of global supply chains |
| Telecom network topology | 1M+ nodes | Full infrastructure graph with live metrics |
| Any graph that crashes Gephi | >100K edges | GPU renders what CPU-based tools cannot |

### Graphistry + PyGraphistry Python Client

```python
import graphistry
import pandas as pd

# Authenticate (self-hosted or Graphistry Hub)
graphistry.register(api=3, protocol="https",
                    server="hub.graphistry.com",
                    username="user", password="pass")

# Create edge dataframe
edges = pd.DataFrame({
    "src": ["Beijing", "Beijing", "Shanghai", "Guangzhou", "Shenzhen"],
    "dst": ["Shanghai", "Guangzhou", "Shenzhen", "Shenzhen", "Hong Kong"],
    "passengers": [12000, 8500, 6000, 9500, 7200],
    "route_type": ["HSR", "flight", "HSR", "HSR", "metro"]
})

# Create node dataframe with coordinates
nodes = pd.DataFrame({
    "city": ["Beijing", "Shanghai", "Guangzhou", "Shenzhen", "Hong Kong"],
    "lat": [39.9, 31.2, 23.1, 22.5, 22.3],
    "lon": [116.4, 121.5, 113.3, 114.1, 114.2],
    "population_m": [21.5, 24.9, 15.3, 17.6, 7.5]
})

# Visualize
g = graphistry.edges(edges, "src", "dst") \
    .nodes(nodes, "city") \
    .bind(point_size="population_m", edge_weight="passengers") \
    .encode_edge_color("route_type", categorical_mapping={
        "HSR": "#e74c3c", "flight": "#3498db", "metro": "#2ecc71"
    }) \
    .settings(url_params={"pointSize": 0.5, "edgeCurvature": 0.3})

g.plot()  # returns an iframe URL for browser-based exploration
```

### RAPIDS cuGraph Integration

For graphs that exceed CPU memory or require sub-second analytics on billion-edge datasets:

```python
import cugraph
import cudf

# Load edges into GPU memory
edges_cudf = cudf.DataFrame({
    "src": [0, 0, 1, 2, 3],
    "dst": [1, 2, 3, 3, 4],
    "weight": [1.0, 2.0, 1.5, 1.0, 3.0]
})

# Build graph on GPU
G_gpu = cugraph.Graph()
G_gpu.from_cudf_edgelist(edges_cudf, source="src",
                          destination="dst", edge_attr="weight")

# PageRank on GPU (orders of magnitude faster than NetworkX)
pagerank = cugraph.pagerank(G_gpu)

# Louvain community detection on GPU
parts, modularity = cugraph.louvain(G_gpu)
print(f"Modularity: {modularity:.4f}, Communities: {parts['partition'].nunique()}")

# Pipe results to Graphistry for visualization
g = graphistry.edges(edges_cudf, "src", "dst") \
    .nodes(pagerank.rename(columns={"vertex": "node", "pagerank": "pr"}), "node") \
    .bind(point_size="pr")
g.plot()
```

> **SOTA**: Graphistry 2.40+ (2024) supports [GFQL](https://github.com/graphistry/pygraphistry/blob/master/docs/gfql.md) (Graph Frame Query Language), a dataframe-native graph query language that avoids the need for a separate graph database.

---

## 5. deck.gl Graph Layers

[deck.gl](https://deck.gl) is the premier WebGL-powered visualization framework for large-scale geospatial data. Its layered architecture makes it ideal for combining network visualization with other spatial data.

### ArcLayer for Origin-Destination Flows

```javascript
import { Deck } from "@deck.gl/core";
import { ArcLayer } from "@deck.gl/layers";

const arcLayer = new ArcLayer({
  id: "flight-arcs",
  data: "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/arc/counties.json",
  getSourcePosition: d => d.from.coordinates,
  getTargetPosition: d => d.to.coordinates,
  getSourceColor: [0, 128, 255],
  getTargetColor: [255, 0, 128],
  getWidth: d => Math.sqrt(d.passengers) * 0.5,
  greatCircle: true,       // critical for long-distance routes
  numSegments: 50,          // smoothness of the arc
  widthMinPixels: 1,
  opacity: 0.4
});

new Deck({
  mapStyle: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
  initialViewState: { longitude: 0, latitude: 20, zoom: 2 },
  controller: true,
  layers: [arcLayer]
});
```

### ScatterplotLayer + LineLayer for Networks

```javascript
import { ScatterplotLayer, LineLayer } from "@deck.gl/layers";

// Nodes: cities with population
const nodeLayer = new ScatterplotLayer({
  id: "network-nodes",
  data: cities,
  getPosition: d => [d.lon, d.lat],
  getRadius: d => Math.sqrt(d.population) * 10,
  getFillColor: d => d.isHub ? [255, 107, 107] : [78, 205, 196],
  radiusMinPixels: 3,
  radiusMaxPixels: 30,
  pickable: true,
  onClick: ({ object }) => showCityDetails(object)
});

// Edges: connections with flow volume
const edgeLayer = new LineLayer({
  id: "network-edges",
  data: connections,
  getSourcePosition: d => [d.from_lon, d.from_lat],
  getTargetPosition: d => [d.to_lon, d.to_lat],
  getColor: d => flowColorScale(d.volume),
  getWidth: d => Math.log(d.volume + 1) * 0.5,
  widthMinPixels: 0.5,
  widthMaxPixels: 8,
  opacity: 0.6,
  pickable: true
});
```

### TripsLayer for Animated Network Flows

```javascript
import { TripsLayer } from "@deck.gl/geo-layers";

const tripsLayer = new TripsLayer({
  id: "network-trips",
  data: vehicleTraces,
  getPath: d => d.waypoints.map(w => [w.lon, w.lat]),
  getTimestamps: d => d.waypoints.map(w => w.timestamp),
  getColor: d => d.type === "freight" ? [255, 165, 0] : [0, 150, 255],
  getWidth: 3,
  trailLength: 200,
  currentTime: animationTime,
  opacity: 0.8
});
```

### Combining Network + Heatmap + Polygon Layers

deck.gl's composability is its superpower:

```javascript
import { HeatmapLayer, GeoJsonLayer } from "@deck.gl/layers";

const layers = [
  // Base: admin boundaries colored by metric
  new GeoJsonLayer({
    id: "admin-boundaries",
    data: adminGeoJSON,
    getFillColor: d => choroplethColor(d.properties.gdp_per_capita),
    getLineColor: [80, 80, 80],
    lineWidthMinPixels: 1,
    opacity: 0.3
  }),
  // Middle: connection network
  edgeLayer,
  nodeLayer,
  // Top: activity heatmap
  new HeatmapLayer({
    id: "node-activity-heat",
    data: nodeActivityLog,
    getPosition: d => [d.lon, d.lat],
    getWeight: d => d.event_count,
    radiusPixels: 30,
    intensity: 2,
    opacity: 0.4
  })
];
```

> **SOTA**: deck.gl v9 (2024) supports [GPU data filtering](https://deck.gl/docs/api-reference/extensions/data-filter-extension) -- filter millions of edges in real-time without re-uploading data to the GPU.

**Cross-ref:** [3D Visualization](3d-visualization.md) for deck.gl's 3D layers (HexagonLayer, PointCloudLayer).

---

## 6. Vis.js & Sigma.js

### Vis.js Network

[vis.js](https://visjs.github.io/vis-network/docs/network/) is the fastest path from zero to a working network visualization. No build step, no bundler -- just a script tag and data.

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
  <style>#network { width: 100%; height: 600px; border: 1px solid #ccc; }</style>
</head>
<body>
  <div id="network"></div>
  <script>
    const nodes = new vis.DataSet([
      { id: 1, label: "Beijing", x: 1164, y: -399, fixed: true,
        size: 30, color: "#e74c3c" },
      { id: 2, label: "Shanghai", x: 1215, y: -312, fixed: true,
        size: 28, color: "#e74c3c" },
      { id: 3, label: "Guangzhou", x: 1133, y: -231, fixed: true,
        size: 22, color: "#f39c12" },
      { id: 4, label: "Chengdu", x: 1040, y: -306, fixed: true,
        size: 20, color: "#f39c12" }
    ]);

    const edges = new vis.DataSet([
      { from: 1, to: 2, value: 12000, title: "HSR: 4.5h" },
      { from: 1, to: 3, value: 8500, title: "Flight: 3h" },
      { from: 2, to: 3, value: 6000, title: "HSR: 6.5h" },
      { from: 1, to: 4, value: 5000, title: "Flight: 2.5h" },
      { from: 3, to: 4, value: 3000, title: "HSR: 8h" }
    ]);

    const options = {
      nodes: { shape: "dot", font: { color: "#ffffff" } },
      edges: {
        scaling: { min: 1, max: 8, label: { enabled: false } },
        smooth: { type: "curvedCW", roundness: 0.2 },
        color: { color: "#4a90d9", highlight: "#f1c40f" }
      },
      physics: { enabled: false },
      interaction: { hover: true, tooltipDelay: 100 }
    };

    new vis.Network(document.getElementById("network"),
                    { nodes, edges }, options);
  </script>
</body>
</html>
```

### Sigma.js v3: WebGL Graph Rendering

[Sigma.js v3](https://www.sigmajs.org) is a complete rewrite using WebGL, capable of rendering 50K+ nodes at 60 fps.

```javascript
import Graph from "graphology";
import Sigma from "sigma";
import forceAtlas2 from "graphology-layout-forceatlas2";

const graph = new Graph();

graph.addNode("beijing", {
  x: 116.4, y: 39.9,
  size: 15, label: "Beijing",
  color: "#e74c3c"
});
graph.addNode("shanghai", {
  x: 121.5, y: 31.2,
  size: 13, label: "Shanghai",
  color: "#e74c3c"
});
graph.addEdge("beijing", "shanghai", {
  weight: 12000, color: "#4a90d9", type: "line"
});

const renderer = new Sigma(graph, document.getElementById("sigma-container"), {
  renderEdgeLabels: true,
  defaultEdgeType: "line",
  labelRenderedSizeThreshold: 8,
  zIndex: true
});

// ForceAtlas2 layout refinement
const settings = forceAtlas2.inferSettings(graph);
forceAtlas2.assign(graph, { settings, iterations: 100 });
```

### Vis.js vs. Sigma.js Decision Matrix

| Criterion | vis.js | Sigma.js v3 |
|-----------|--------|-------------|
| **Max nodes (interactive)** | ~5,000 | ~50,000 |
| **Renderer** | Canvas 2D | WebGL |
| **Setup complexity** | Single script tag | npm + bundler |
| **Layout algorithms** | Built-in (Barnes-Hut, etc.) | Via graphology plugins |
| **Graph analysis** | None built-in | Full graphology ecosystem |
| **Customization** | Options object | Shader-level control |
| **Best for** | Quick prototypes, embedded widgets | Production graph explorers |

---

## 7. Kepler.gl Arc/Line Layers

[Kepler.gl](https://kepler.gl) is Uber's open-source geospatial analysis tool. For network visualization, its Arc and Line layers turn origin-destination data into immediate visual insights -- no code required.

### Origin-Destination Flow Maps

Kepler.gl expects data with source/target coordinates:

```csv
origin_lat,origin_lon,dest_lat,dest_lon,passengers,route_name
39.9042,116.4074,31.2304,121.4737,12000,Beijing-Shanghai
39.9042,116.4074,23.1291,113.2644,8500,Beijing-Guangzhou
31.2304,121.4737,22.5431,114.0579,6000,Shanghai-Shenzhen
```

### Programmatic Kepler.gl Configuration

```python
from keplergl import KeplerGl
import pandas as pd

flows = pd.DataFrame({
    "origin_lat": [39.90, 39.90, 31.23, 23.13],
    "origin_lon": [116.41, 116.41, 121.47, 113.26],
    "dest_lat": [31.23, 23.13, 22.54, 22.54],
    "dest_lon": [121.47, 113.26, 114.06, 114.06],
    "volume": [12000, 8500, 6000, 9500],
    "mode": ["HSR", "flight", "HSR", "HSR"]
})

config = {
    "version": "v1",
    "config": {
        "visState": {
            "layers": [{
                "type": "arc",
                "config": {
                    "dataId": "flows",
                    "columns": {
                        "lat0": "origin_lat", "lng0": "origin_lon",
                        "lat1": "dest_lat", "lng1": "dest_lon"
                    },
                    "color": [255, 107, 107],
                    "sizeField": { "name": "volume", "type": "integer" },
                    "colorField": { "name": "mode", "type": "string" },
                    "visConfig": {
                        "opacity": 0.6,
                        "thickness": 3,
                        "targetColor": [78, 205, 196]
                    }
                }
            }]
        },
        "mapStyle": { "styleType": "dark" }
    }
}

m = KeplerGl(height=600, data={"flows": flows}, config=config)
m  # renders in Jupyter
```

### Flowmap.blue: Instant OD Visualization

[Flowmap.blue](https://www.flowmap.blue) is the fastest way to create a publication-quality flow map:

1. Create a Google Sheet with three tabs: `locations` (id, name, lat, lon), `flows` (origin, dest, count), `properties` (optional config)
2. Publish the sheet to the web
3. Paste the sheet URL into flowmap.blue

> **SOTA flow mapping tools:**
> - **Flowmap.blue** -- Zero-code, Google Sheets input, animated arcs with automatic clustering
> - **Kepler.gl** -- Drag-and-drop, supports arc/line/hexbin layers, exports as HTML
> - **deck.gl ArcLayer** -- Programmatic, maximum control, WebGL performance
> - **Movingpandas** -- Python library for movement data analysis and trajectory visualization

**Cross-ref:** [Thematic Maps](thematic-maps.md) for choropleth-based flow representation.

---

## 8. Transportation Networks

Transportation networks are among the most data-rich spatial graphs, with standardized formats (GTFS, OSM), well-established analysis methods, and direct policy relevance.

### Road Networks: OSMnx + NetworkX

```python
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt

G = ox.graph_from_place("Shenzhen, China", network_type="drive")

stats = ox.basic_stats(G)
print(f"Nodes: {stats['n']:,}  Edges: {stats['m']:,}")
print(f"Average street length: {stats['street_length_avg']:.0f} m")

# Edge coloring by speed
ec = ox.plot.get_edge_colors_by_attr(G, attr="speed_kph", cmap="RdYlGn")
fig, ax = ox.plot_graph(
    G, edge_color=ec, edge_linewidth=0.5,
    node_size=0, bgcolor="#1a1a2e",
    figsize=(16, 16), save=True,
    filepath="shenzhen_speed.png", dpi=300
)
```

**Edge Coloring Strategies**

| Attribute | Color Map | Interpretation |
|-----------|-----------|----------------|
| `speed_kph` | `RdYlGn` (diverging) | Green = fast arterials, red = slow local streets |
| `length` | `viridis` (sequential) | Longer edges glow brighter -- reveals grid vs. organic layout |
| `highway` (categorical) | Custom palette | Motorway = red, primary = orange, residential = gray |
| Betweenness centrality | `plasma` | Hot edges carry more shortest paths -- identifies bottlenecks |
| Travel time | `coolwarm` | Blue = quick, red = congested |

**Betweenness Centrality Reveals Urban Spine**

```python
import networkx as nx

# For cities > 50K edges, sample: nx.edge_betweenness_centrality(G, k=500)
ebc = nx.edge_betweenness_centrality(G, weight="length", k=1000)
nx.set_edge_attributes(G, ebc, "betweenness")

ec = ox.plot.get_edge_colors_by_attr(G, attr="betweenness", cmap="inferno")
fig, ax = ox.plot_graph(
    G, edge_color=ec, edge_linewidth=0.8,
    node_size=0, bgcolor="#0d0d0d"
)
# The bright edges form the city's "structural spine."
```

### River & Stream Networks

Hydrological networks are naturally directed acyclic graphs (DAGs). Strahler stream order encodes hierarchy.

| Strahler Order | Description | Visual Treatment |
|----------------|-------------|------------------|
| 1 | Headwater streams, no tributaries | Thin, light blue, 0.5px |
| 2 | Two order-1 streams merge | Medium, 1px |
| 3 | Two order-2 streams merge | Thicker, 1.5px |
| 4+ | Major rivers | Bold, 2-4px, darker blue |

```python
import geopandas as gpd
import matplotlib.pyplot as plt

streams = gpd.read_file("nhdplus_streams.gpkg")

fig, ax = plt.subplots(figsize=(12, 16), facecolor="#0a0a1a")
ax.set_facecolor("#0a0a1a")

for order in sorted(streams["strahler"].unique()):
    subset = streams[streams["strahler"] == order]
    lw = 0.3 + order * 0.5
    alpha = 0.4 + min(order * 0.15, 0.6)
    subset.plot(ax=ax, color="#4fc3f7", linewidth=lw, alpha=alpha)

ax.set_axis_off()
```

**Key datasets:**
- **NHDPlus** (US) -- National Hydrography Dataset with flow direction, Strahler order, and catchments
- **HydroSHEDS** (global) -- Derived from SRTM, includes river networks at multiple resolutions
- **EU-Hydro** (Europe) -- Pan-European river network from Copernicus

### Transit Networks: GTFS Visualization

```python
import partridge as ptg
import networkx as nx
import folium

feed = ptg.load_feed("city_transit.zip")

G = nx.DiGraph()
for trip_id, group in feed.stop_times.groupby("trip_id"):
    stops = group.sort_values("stop_sequence")
    route_id = feed.trips[feed.trips.trip_id == trip_id].route_id.iloc[0]
    for i in range(len(stops) - 1):
        s1 = stops.iloc[i]
        s2 = stops.iloc[i + 1]
        G.add_edge(
            s1.stop_id, s2.stop_id,
            route=route_id,
            travel_time=s2.arrival_time - s1.departure_time
        )

for _, stop in feed.stops.iterrows():
    if stop.stop_id in G:
        G.nodes[stop.stop_id]["lat"] = stop.stop_lat
        G.nodes[stop.stop_id]["lon"] = stop.stop_lon
        G.nodes[stop.stop_id]["name"] = stop.stop_name

hubs = sorted(G.nodes(), key=lambda n: G.degree(n), reverse=True)[:20]

m = folium.Map(location=[feed.stops.stop_lat.mean(),
                          feed.stops.stop_lon.mean()], zoom_start=12)
for u, v, data in G.edges(data=True):
    if "lat" in G.nodes[u] and "lat" in G.nodes[v]:
        folium.PolyLine(
            [[G.nodes[u]["lat"], G.nodes[u]["lon"]],
             [G.nodes[v]["lat"], G.nodes[v]["lon"]]],
            weight=1, color="#4a90d9", opacity=0.4
        ).add_to(m)

for hub in hubs:
    if "lat" in G.nodes[hub]:
        folium.CircleMarker(
            [G.nodes[hub]["lat"], G.nodes[hub]["lon"]],
            radius=G.degree(hub) * 0.5,
            color="#ff6b6b", fill=True
        ).add_to(m)
```

### Isochrone Networks

```python
import osmnx as ox
import networkx as nx

G = ox.graph_from_point((39.9042, 116.4074), dist=3000, network_type="walk")

for u, v, data in G.edges(data=True):
    data["travel_time"] = data["length"] / (4.5 * 1000 / 3600)

origin = ox.nearest_nodes(G, 116.4074, 39.9042)
subgraph = nx.ego_graph(G, origin, radius=900, distance="travel_time")

ec = ["#2ecc71" if e in subgraph.edges() else "#333333" for e in G.edges()]
ew = [1.5 if e in subgraph.edges() else 0.3 for e in G.edges()]
fig, ax = ox.plot_graph(G, edge_color=ec, edge_linewidth=ew,
                        node_size=0, bgcolor="#0d0d0d")
```

### Routing Engines

| Engine | Language | Strengths | URL |
|--------|----------|-----------|-----|
| **OSRM** | C++ | Fastest for car routing, contraction hierarchies | [project-osrm.org](http://project-osrm.org) |
| **Valhalla** | C++ | Multi-modal, isochrones, map matching | [github.com/valhalla](https://github.com/valhalla/valhalla) |
| **OpenRouteService** | Java | Wheelchair routing, avoid areas, elevation | [openrouteservice.org](https://openrouteservice.org) |
| **GraphHopper** | Java | Customizable profiles, commercial API | [graphhopper.com](https://www.graphhopper.com) |
| **r5py** | Python/Java | Multi-modal accessibility, GTFS+OSM | [r5py.readthedocs.io](https://r5py.readthedocs.io) |

**Cross-ref:** [Geocoding & Routing](../tools/geocoding-routing.md) for routing engine setup guides.

---

## 9. Utility Networks

Infrastructure networks share a common pattern: hierarchical topology with critical nodes and redundancy constraints.

### Power Grid Visualization

| Data Source | Coverage | Access |
|-------------|----------|--------|
| [OpenInfraMap](https://openinframap.org) | Global (OSM-based) | Free |
| [GridKit](https://github.com/bdw/GridKit) | Global extracted from OSM | Free |
| EIA Form 860/861 (US) | US power plants + utilities | Free |
| ENTSO-E (Europe) | European transmission grid | Restricted |

```python
import requests
import networkx as nx

overpass_query = """
[out:json][timeout:120];
area["name"="Guangdong"]->.searchArea;
(
  way["power"="line"](area.searchArea);
  node["power"="substation"](area.searchArea);
);
out body; >; out skel qt;
"""

response = requests.get(
    "http://overpass-api.de/api/interpreter",
    params={"data": overpass_query}
)
data = response.json()

G = nx.Graph()
for element in data["elements"]:
    if element["type"] == "node" and "tags" in element:
        if element["tags"].get("power") == "substation":
            G.add_node(element["id"],
                       lat=element["lat"], lon=element["lon"],
                       voltage=element["tags"].get("voltage", "unknown"))
```

### Water/Sewer Network Analysis

[WNTR](https://wntr.readthedocs.io) (Water Network Tool for Resilience) provides graph-based analysis of water distribution networks modeled in [EPANET](https://www.epa.gov/water-research/epanet).

```python
import wntr
import networkx as nx

wn = wntr.network.WaterNetworkModel("city_water.inp")
G = wn.to_graph()
print(f"Junctions: {wn.num_junctions}, Pipes: {wn.num_pipes}")

# Vulnerability: find single points of failure
cut_nodes = list(nx.articulation_points(G.to_undirected()))
print(f"Single points of failure: {len(cut_nodes)}")

bridges = list(nx.bridges(G.to_undirected()))
print(f"Critical pipes (bridges): {len(bridges)}")

# Simulate pipe break and measure impact
wn_damaged = wntr.morph.split_pipe(wn, "pipe_42", "pipe_42_break", "new_junction")
sim = wntr.sim.WNTRSimulator(wn_damaged)
results = sim.run_sim()
pressure = results.node["pressure"]
low_pressure = pressure.min(axis=0) < 20  # 20 psi minimum
print(f"Affected junctions: {low_pressure.sum()}")
```

### Telecom Network Topology

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

backbone = ["DC_North", "DC_South", "DC_East", "CO_1", "CO_2", "CO_3"]
for node in backbone:
    G.add_node(node, tier="backbone", size=30)

distribution = [f"dist_{i}" for i in range(20)]
for node in distribution:
    G.add_node(node, tier="distribution", size=15)
    G.add_edge(node, backbone[hash(node) % len(backbone)],
               bandwidth_gbps=10, link_type="fiber")

access = [f"access_{i}" for i in range(100)]
for node in access:
    G.add_node(node, tier="access", size=5)
    G.add_edge(node, distribution[hash(node) % len(distribution)],
               bandwidth_gbps=1, link_type="fiber_or_copper")

color_map = {"backbone": "#e74c3c", "distribution": "#f39c12", "access": "#3498db"}
nc = [color_map[G.nodes[n]["tier"]] for n in G.nodes()]
ns = [G.nodes[n]["size"] for n in G.nodes()]

pos = nx.kamada_kawai_layout(G)
nx.draw(G, pos, node_color=nc, node_size=ns, edge_color="#555555",
        width=0.5, alpha=0.7, with_labels=False)
```

---

## 10. Social Network Analysis

Social networks gain a spatial dimension when users are geo-tagged or communication patterns are tied to locations.

### Geo-Tagged Social Media Networks

| Platform | Geo Data Available | Network Type |
|----------|--------------------|-------------|
| **Twitter/X** | Tweet coordinates, user location text | Follow/retweet/mention |
| **Weibo** | Check-in coordinates, POI tags | Follow/repost/comment |
| **Instagram** | Photo geotags, location stickers | Follow/like/comment |
| **Flickr** | Photo EXIF coordinates | Fave/group membership |
| **Facebook** | City-level (Social Connectedness Index) | Friendship |

### Building a Spatial Retweet Network

```python
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

retweets = pd.read_csv("geo_retweets.csv")

G = nx.DiGraph()
for _, row in retweets.iterrows():
    G.add_node(row["source_user"],
               pos=(row["source_lon"], row["source_lat"]))
    G.add_node(row["target_user"],
               pos=(row["target_lon"], row["target_lat"]))
    G.add_edge(row["source_user"], row["target_user"])

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
fig, ax = plt.subplots(figsize=(18, 10))
world.plot(ax=ax, color="#1a1a2e", edgecolor="#2a2a4e")

pos = nx.get_node_attributes(G, "pos")
nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#ff6b6b",
                       alpha=0.15, width=0.3, arrows=False)
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=5,
                       node_color="#4ecdc4", alpha=0.6)
ax.set_xlim([-180, 180]); ax.set_ylim([-60, 85])
ax.set_axis_off()
```

### Community Detection

| Algorithm | Complexity | Best For |
|-----------|-----------|----------|
| **Louvain** | O(n log n) | Large networks, fast modularity optimization |
| **Leiden** | O(n log n) | Improved Louvain, guarantees connected communities |
| **Label Propagation** | O(m) | Very large networks, near-linear time |
| **Girvan-Newman** | O(m^2 n) | Small networks, edge betweenness based |
| **Spectral** | O(n^3) | Well-separated clusters |

```python
from networkx.community import louvain_communities
import matplotlib.cm as cm

G = ox.graph_from_place("Barcelona, Spain", network_type="drive")
G_undirected = G.to_undirected()

communities = louvain_communities(G_undirected, weight="length", seed=42)
print(f"Detected {len(communities)} communities")

node_colors = {}
cmap = cm.get_cmap("tab20", len(communities))
for i, community in enumerate(communities):
    for node in community:
        node_colors[node] = cmap(i)

nc = [node_colors.get(n, (0.5, 0.5, 0.5, 1.0)) for n in G.nodes()]
fig, ax = ox.plot_graph(
    G, node_color=nc, node_size=3,
    edge_color="#333333", edge_linewidth=0.3,
    bgcolor="#0d0d0d", figsize=(16, 16)
)
```

### Influence Propagation Models

| Model | Mechanism | Spatial Variant |
|-------|-----------|----------------|
| **SIR/SIS** | Susceptible-Infected-Recovered | Distance-weighted transmission |
| **Independent Cascade** | One chance to infect neighbors | Edge weights decay with distance |
| **Linear Threshold** | Fraction threshold activation | Threshold varies by local density |
| **Hawkes Process** | Self-exciting point process | Spatiotemporal triggering kernel |

### Geographic Community Constraints

```python
import leidenalg
import igraph as ig

G_ig = ig.Graph.from_networkx(G_nx)

partition_topo = leidenalg.find_partition(G_ig, leidenalg.ModularityVertexPartition)

# Add distance-based edge weights for geographic coherence
for e in G_ig.es:
    src, tgt = e.source, e.target
    d = haversine(G_ig.vs[src]["lat"], G_ig.vs[src]["lon"],
                  G_ig.vs[tgt]["lat"], G_ig.vs[tgt]["lon"])
    e["geo_weight"] = 1.0 / (1.0 + d)

partition_geo = leidenalg.find_partition(
    G_ig, leidenalg.ModularityVertexPartition, weights="geo_weight"
)
```

---

## 11. Ecological Networks

Ecological networks model relationships between species, habitats, and landscape elements.

### Food Webs

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

species = {
    "Phytoplankton": 1, "Seagrass": 1, "Detritus": 1,
    "Zooplankton": 2, "Amphipods": 2, "Snails": 2,
    "Small fish": 3, "Crabs": 3, "Shrimp": 3,
    "Large fish": 4, "Wading birds": 4,
    "Dolphins": 5, "Eagles": 5
}

for sp, level in species.items():
    G.add_node(sp, trophic_level=level)

edges = [
    ("Zooplankton", "Phytoplankton"), ("Amphipods", "Detritus"),
    ("Snails", "Seagrass"), ("Small fish", "Zooplankton"),
    ("Crabs", "Amphipods"), ("Shrimp", "Detritus"),
    ("Large fish", "Small fish"), ("Large fish", "Shrimp"),
    ("Wading birds", "Crabs"), ("Wading birds", "Small fish"),
    ("Dolphins", "Large fish"), ("Eagles", "Large fish"),
    ("Eagles", "Wading birds")
]
G.add_edges_from(edges)

# Layout: x = spread within trophic level, y = trophic level
pos = {}
for level in range(1, 6):
    nodes_at_level = [n for n, d in G.nodes(data=True)
                      if d["trophic_level"] == level]
    for i, node in enumerate(nodes_at_level):
        pos[node] = (i - len(nodes_at_level)/2, level)

fig, ax = plt.subplots(figsize=(14, 10), facecolor="#0a0a1a")
ax.set_facecolor("#0a0a1a")
cmap = plt.cm.YlOrRd
nc = [cmap(species[n] / 5) for n in G.nodes()]

nx.draw(G, pos, ax=ax, with_labels=True, node_color=nc,
        node_size=800, font_size=8, font_color="white",
        edge_color="#555555", width=1.5, arrows=True,
        arrowsize=12, connectionstyle="arc3,rad=0.1")
```

### Species Interaction Networks

```python
import networkx as nx
from networkx.algorithms import bipartite

B = nx.Graph()
plants = ["Rosa", "Lavandula", "Salvia", "Helianthus", "Trifolium"]
pollinators = ["Apis_mellifera", "Bombus_terrestris", "Osmia_bicornis",
               "Syrphus_ribesii", "Vanessa_cardui"]

B.add_nodes_from(plants, bipartite=0)
B.add_nodes_from(pollinators, bipartite=1)

interactions = [
    ("Apis_mellifera", "Rosa", 45), ("Apis_mellifera", "Lavandula", 120),
    ("Apis_mellifera", "Trifolium", 80), ("Bombus_terrestris", "Salvia", 60),
    ("Bombus_terrestris", "Trifolium", 90), ("Osmia_bicornis", "Rosa", 30),
    ("Syrphus_ribesii", "Helianthus", 25), ("Vanessa_cardui", "Lavandula", 15),
    ("Vanessa_cardui", "Helianthus", 20)
]
for poll, plant, visits in interactions:
    B.add_edge(poll, plant, weight=visits)
```

### Habitat Connectivity: Circuitscape

[Circuitscape](https://circuitscape.org) models landscape connectivity using circuit theory.

| Tool | Method | Scale | Language |
|------|--------|-------|----------|
| [Circuitscape](https://circuitscape.org) | Circuit theory (resistance distance) | Landscape | Julia |
| [Linkage Mapper](https://linkagemapper.org) | Least-cost corridors | Landscape | ArcGIS toolbox |
| [Conefor](http://www.conefor.org) | Graph-theoretic indices (PC, IIC) | Patch network | Windows |
| [grainscape](https://github.com/achubaty/grainscape) | Minimum planar graph | Landscape | R |
| [graphab](https://sourcesup.renater.fr/graphab/) | Graph-based connectivity | Patch network | Java GUI |

```python
import rasterio
import numpy as np

# Create resistance surface from land cover
resistance_map = {
    1: 1,     # Forest -> low resistance
    2: 5,     # Grassland -> moderate
    3: 50,    # Agriculture -> high
    4: 1000,  # Urban -> near-barrier
    5: -9999  # Water -> absolute barrier
}

with rasterio.open("landcover.tif") as src:
    lc = src.read(1)
    resistance = np.vectorize(resistance_map.get)(lc)
    profile = src.profile.copy()

with rasterio.open("resistance.tif", "w", **profile) as dst:
    dst.write(resistance, 1)
```

### Landscape Graph Metrics

| Metric | Intuition | Use |
|--------|----------|-----|
| **Probability of Connectivity (PC)** | Probability two random points are connected | Overall landscape connectivity |
| **Integral Index of Connectivity (IIC)** | Binary connected/not | Quick connectivity assessment |
| **Betweenness Centrality (BC)** | How often a patch is on shortest paths | Identify stepping stone habitats |
| **dPC** | Change in PC if patch is removed | Rank patches by conservation priority |

---

## 12. Supply Chain & Logistics

Supply chain networks are weighted directed graphs where edges carry flow volumes.

### Supply Chain Graph Visualization

```python
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()

raw_materials = ["Iron Mine (Brazil)", "Lithium (Chile)", "Copper (Congo)"]
processors = ["Smelter (China)", "Refinery (Korea)", "Battery Plant (Japan)"]
manufacturers = ["EV Factory (Germany)", "EV Factory (USA)"]
distributors = ["Port Rotterdam", "Port LA", "Port Shanghai"]
retailers = ["EU Dealers", "US Dealers", "Asia Dealers"]

for node in raw_materials: G.add_node(node, tier=0)
for node in processors: G.add_node(node, tier=1)
for node in manufacturers: G.add_node(node, tier=2)
for node in distributors: G.add_node(node, tier=3)
for node in retailers: G.add_node(node, tier=4)

G.add_edge("Iron Mine (Brazil)", "Smelter (China)", tons=50000, lead_days=45)
G.add_edge("Lithium (Chile)", "Battery Plant (Japan)", tons=5000, lead_days=30)
G.add_edge("Copper (Congo)", "Refinery (Korea)", tons=20000, lead_days=25)
G.add_edge("Smelter (China)", "EV Factory (Germany)", tons=30000, lead_days=35)
G.add_edge("Battery Plant (Japan)", "EV Factory (USA)", tons=3000, lead_days=20)
G.add_edge("Refinery (Korea)", "EV Factory (Germany)", tons=15000, lead_days=15)
G.add_edge("EV Factory (Germany)", "Port Rotterdam", units=10000, lead_days=5)
G.add_edge("EV Factory (USA)", "Port LA", units=8000, lead_days=3)
G.add_edge("Port Rotterdam", "EU Dealers", units=10000, lead_days=7)
G.add_edge("Port LA", "US Dealers", units=8000, lead_days=5)

pos = nx.multipartite_layout(G, subset_key="tier", align="horizontal")

tier_colors = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71",
               3: "#3498db", 4: "#9b59b6"}
nc = [tier_colors[G.nodes[n]["tier"]] for n in G.nodes()]

fig, ax = plt.subplots(figsize=(18, 8), facecolor="#0d0d0d")
ax.set_facecolor("#0d0d0d")
nx.draw(G, pos, ax=ax, with_labels=True, node_color=nc,
        node_size=1200, font_size=7, font_color="white",
        edge_color="#666666", width=1.5, arrows=True, arrowsize=15)
```

### Shipping Route Visualization

```python
import geopandas as gpd
from shapely.geometry import LineString

routes = gpd.GeoDataFrame({
    "route": ["Asia-Europe", "Transpacific", "Asia-Africa"],
    "teu_annual_m": [24.0, 28.0, 8.0],
    "geometry": [
        LineString([(121.5, 31.2), (104.0, 1.3), (43.1, 12.8),
                    (32.3, 31.2), (3.0, 51.9)]),
        LineString([(121.5, 31.2), (140.0, 35.0), (-122.4, 37.8)]),
        LineString([(121.5, 31.2), (104.0, 1.3), (39.7, -4.0),
                    (18.4, -33.9)])
    ]
}, crs="EPSG:4326")

world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
fig, ax = plt.subplots(figsize=(18, 10), facecolor="#0a0a1a")
world.plot(ax=ax, color="#1a1a2e", edgecolor="#2a2a4e")
routes.plot(ax=ax, column="teu_annual_m", cmap="YlOrRd",
            linewidth=routes["teu_annual_m"] / 5, legend=True, alpha=0.8)
ax.set_axis_off()
```

### OD Matrix and Desire Lines

```python
import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString

od_matrix = np.array([
    [0, 1200, 800, 300],
    [1100, 0, 600, 200],
    [750, 550, 0, 400],
    [280, 180, 350, 0]
])
zones = ["CBD", "North", "East", "South"]
centroids = {
    "CBD": (116.40, 39.90), "North": (116.40, 40.05),
    "East": (116.55, 39.90), "South": (116.40, 39.75)
}

median_flow = np.median(od_matrix[od_matrix > 0])

lines = []
for i, origin in enumerate(zones):
    for j, dest in enumerate(zones):
        if i != j and od_matrix[i, j] > median_flow:
            lines.append({
                "geometry": LineString([centroids[origin], centroids[dest]]),
                "flow": od_matrix[i, j],
                "origin": origin, "destination": dest
            })

desire_lines = gpd.GeoDataFrame(lines, crs="EPSG:4326")
desire_lines.plot(
    column="flow", cmap="YlOrRd",
    linewidth=desire_lines["flow"] / desire_lines["flow"].max() * 5,
    legend=True, figsize=(10, 10)
)
```

**Cross-ref:** [Logistics, Delivery & Urban Mobility](../data-sources/logistics-delivery-urban-mobility.md) for OD datasets and GTFS feeds.

---

## 13. Knowledge Graphs

Knowledge graphs encode entities and their relationships as typed nodes and edges. In the GIS domain, spatial knowledge graphs connect places, features, administrative hierarchies, and thematic concepts into queryable, visual structures.

### GIS Ontologies

| Ontology | Domain | Key Concepts | Use Case |
|----------|--------|-------------|----------|
| **GeoSPARQL** | OGC standard | `geo:Feature`, `geo:Geometry`, topological relations | Querying spatial relationships in RDF triplestores |
| **schema.org Place** | Web-wide | `Place`, `GeoCoordinates`, `containedInPlace` | SEO for locations, structured web data |
| **DBpedia/Wikidata** | Encyclopedic | `dbo:Place`, `wdt:P625` (coordinates) | Linked Open Data enrichment |
| **INSPIRE** | EU SDI | Feature types for 34 spatial data themes | European NSDI interoperability |
| **CityGML** | Urban models | `Building`, `Transportation`, `LandUse`, LOD 0-4 | 3D city model semantics |

### GeoSPARQL Query Example

```sparql
# Find all rivers that flow through protected areas
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX ex: <http://example.org/features/>

SELECT ?river ?protectedArea
WHERE {
  ?river a ex:River ;
         geo:hasGeometry ?riverGeom .
  ?protectedArea a ex:ProtectedArea ;
                 geo:hasGeometry ?paGeom .
  ?riverGeom geo:asWKT ?riverWKT .
  ?paGeom geo:asWKT ?paWKT .
  FILTER(geof:sfIntersects(?riverWKT, ?paWKT))
}
```

### Linked Open Data Spatial Graphs

The Linked Open Data cloud contains billions of spatial triples. Visualizing connections between geographic entities reveals patterns invisible in tabular data.

```python
from SPARQLWrapper import SPARQLWrapper, JSON
import networkx as nx

sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
sparql.setQuery("""
SELECT ?city ?cityLabel ?country ?countryLabel ?pop ?lat ?lon
WHERE {
  ?city wdt:P31 wd:Q515 ;
        wdt:P17 ?country ;
        wdt:P1082 ?pop ;
        wdt:P625 ?coord .
  ?country wdt:P30 wd:Q48 .
  FILTER(?pop > 1000000)
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
  BIND(geof:latitude(?coord) AS ?lat)
  BIND(geof:longitude(?coord) AS ?lon)
}
LIMIT 200
""")
sparql.setReturnFormat(JSON)
results = sparql.query().convert()

G = nx.Graph()
for r in results["results"]["bindings"]:
    city = r["cityLabel"]["value"]
    country = r["countryLabel"]["value"]
    G.add_node(city, type="city",
               lat=float(r["lat"]["value"]),
               lon=float(r["lon"]["value"]),
               pop=int(r["pop"]["value"]))
    G.add_node(country, type="country")
    G.add_edge(city, country)
```

### SKOS Concept Hierarchies for Land Use

SKOS (Simple Knowledge Organization System) is ideal for visualizing classification schemes like land use taxonomies.

```
SKOS Hierarchy: CORINE Land Cover
================================================
Artificial Surfaces (Level 1)
  |-- Urban Fabric (Level 2)
  |     |-- Continuous urban fabric (Level 3)
  |     |-- Discontinuous urban fabric
  |-- Industrial/Commercial (Level 2)
  |     |-- Industrial/commercial units
  |     |-- Road/rail networks
  |     |-- Port areas
  |     |-- Airports
  |-- Mine/Dump/Construction (Level 2)
  |-- Artificial non-agricultural vegetation (Level 2)
        |-- Green urban areas
        |-- Sport and leisure facilities
```

### Knowledge Graph Visualization Tools

| Tool | Type | Max Nodes | Spatial Support | Best For |
|------|------|-----------|----------------|----------|
| **Neo4j Bloom** | Commercial | 100K+ | `point()` type with lat/lon | Enterprise knowledge graphs |
| **Cytoscape** | Desktop (free) | ~50K | Via plugins | Biological networks, ontology exploration |
| **Gephi** | Desktop (free) | ~100K | GeoLayout plugin | Publication-quality static renders |
| **vis.js** | JavaScript | ~5K | Manual positioning | Quick prototypes, embedded in web apps |
| **Sigma.js** | JavaScript (WebGL) | ~50K | Manual positioning | Large web-based graph exploration |
| **Graphistry** | Cloud (GPU) | 1M+ | Lat/lon column mapping | Massive graphs, security analytics |

### Neo4j for Spatial Knowledge Graphs

Neo4j natively supports `point` types with CRS:

```cypher
// Create spatial nodes
CREATE (bj:City {name: "Beijing",
  location: point({latitude: 39.9, longitude: 116.4})})
CREATE (sh:City {name: "Shanghai",
  location: point({latitude: 31.2, longitude: 121.5})})
CREATE (bj)-[:CONNECTED_BY {type: "HSR", travel_hours: 4.5}]->(sh)

// Find all cities within 500km of Beijing
MATCH (c:City)
WHERE point.distance(c.location,
  point({latitude: 39.9, longitude: 116.4})) < 500000
RETURN c.name,
  point.distance(c.location,
    point({latitude: 39.9, longitude: 116.4})) / 1000 AS dist_km
ORDER BY dist_km
```

### Ontology Mapping Visualization

Ontology alignment between different spatial data standards can be visualized as a bipartite graph connecting equivalent concepts:

```python
import networkx as nx
import matplotlib.pyplot as plt

# Mapping between INSPIRE and CityGML concepts
B = nx.Graph()

inspire_concepts = [
    "INSPIRE:Building", "INSPIRE:Road", "INSPIRE:WaterBody",
    "INSPIRE:LandUse", "INSPIRE:ProtectedSite"
]
citygml_concepts = [
    "CityGML:Building", "CityGML:Road", "CityGML:WaterBody",
    "CityGML:LandUse", "CityGML:CityFurniture"
]

B.add_nodes_from(inspire_concepts, ontology="INSPIRE")
B.add_nodes_from(citygml_concepts, ontology="CityGML")

# Alignment edges (exact match, broad match, etc.)
alignments = [
    ("INSPIRE:Building", "CityGML:Building", "exactMatch"),
    ("INSPIRE:Road", "CityGML:Road", "exactMatch"),
    ("INSPIRE:WaterBody", "CityGML:WaterBody", "exactMatch"),
    ("INSPIRE:LandUse", "CityGML:LandUse", "broadMatch"),
]
for src, tgt, match_type in alignments:
    B.add_edge(src, tgt, match_type=match_type)

# Bipartite layout
pos = nx.bipartite_layout(B, inspire_concepts, align="vertical")
edge_colors = {"exactMatch": "#2ecc71", "broadMatch": "#f39c12"}
ec = [edge_colors.get(B.edges[e]["match_type"], "#999") for e in B.edges()]

nx.draw(B, pos, with_labels=True, node_color=["#3498db"]*5 + ["#e74c3c"]*5,
        edge_color=ec, width=2, font_size=8, node_size=800)
```

> **SOTA**: [Apache Jena](https://jena.apache.org) with GeoSPARQL extension for RDF-based spatial knowledge graphs. For property graphs, [Neo4j](https://neo4j.com) with the [neo4j-spatial](https://github.com/neo4j-contrib/spatial) plugin provides R-tree indexing and spatial queries within Cypher.

---

## 14. Network Analysis Tools

A comprehensive reference for graph analysis software and libraries used in geospatial workflows.

### Python Libraries

| Library | Strengths | Max Scale | Language Core | URL |
|---------|----------|-----------|---------------|-----|
| **NetworkX** | Universal, well-documented, huge algorithm library | ~100K nodes | Pure Python | [networkx.org](https://networkx.org) |
| **igraph** | Fast C core, R and Python bindings | ~1M nodes | C | [igraph.org](https://igraph.org) |
| **graph-tool** | Fastest Python graph library (C++/Boost), statistical inference | ~10M nodes | C++ | [graph-tool.skewed.de](https://graph-tool.skewed.de) |
| **OSMnx** | Download + analyze OpenStreetMap networks | City-scale | Python (NetworkX) | [osmnx.readthedocs.io](https://osmnx.readthedocs.io) |
| **cuGraph (RAPIDS)** | GPU-accelerated graph algorithms | ~1B edges | CUDA/C++ | [rapids.ai/cugraph](https://rapids.ai/cugraph) |
| **Networkit** | Parallel algorithms for large networks | ~1B edges | C++/Python | [networkit.github.io](https://networkit.github.io) |
| **PyVis** | Quick interactive HTML network visualizations | ~5K nodes | Python | [pyvis.readthedocs.io](https://pyvis.readthedocs.io) |
| **graphology** | Robust graph data model (used by Sigma.js) | ~100K nodes | JavaScript | [graphology.github.io](https://graphology.github.io) |

### Performance Comparison: Betweenness Centrality

Benchmarks for computing betweenness centrality on a 50K-node road network:

| Library | Time | Memory | Notes |
|---------|------|--------|-------|
| NetworkX | ~180s | ~2 GB | Pure Python, single-threaded |
| igraph | ~12s | ~800 MB | C core, exact algorithm |
| graph-tool | ~5s | ~600 MB | C++ core with OpenMP parallelism |
| Networkit | ~3s | ~500 MB | Parallel C++ with approximation |
| cuGraph | ~0.5s | ~4 GB GPU | Requires NVIDIA GPU |

### Desktop Tools

| Tool | Type | Strengths | Max Scale |
|------|------|-----------|-----------|
| **Gephi** | Free desktop | Beautiful exports, GeoLayout plugin, streaming | ~100K nodes |
| **Cytoscape Desktop** | Free desktop | Biological networks, extensive plugin ecosystem | ~50K nodes |
| **yEd** | Free desktop | Automatic layout algorithms, BPMN/UML support | ~10K nodes |
| **Pajek** | Free desktop | Large network analysis, longitudinal networks | ~1M nodes |
| **NodeXL** | Excel add-in | Social network analysis, Twitter integration | ~10K nodes |

### Gephi GeoLayout Workflow

Gephi's GeoLayout plugin positions nodes by their lat/lon attributes:

1. Import network as GEXF/GraphML with `latitude` and `longitude` node attributes
2. Open **Layout** panel, select **GeoLayout**
3. Map `latitude` to Y, `longitude` to X
4. Choose projection (Mercator or Equirectangular)
5. Run layout -- nodes snap to geographic positions
6. Apply **ForceAtlas2** with low gravity for local refinement
7. Export as PDF/SVG for publication

### Neo4j Spatial

[Neo4j](https://neo4j.com) with the [neo4j-spatial](https://github.com/neo4j-contrib/spatial) extension adds:

- R-tree spatial indexing for nodes with point geometry
- Spatial search queries (within distance, bounding box, polygon containment)
- Integration with graph algorithms (PageRank, community detection, shortest path)

```cypher
// Create spatial index
CALL spatial.addPointLayer('locations')

// Add nodes to spatial index
MATCH (c:City)
CALL spatial.addNode('locations', c) YIELD node
RETURN count(node)

// Find cities within 100km of a point
CALL spatial.withinDistance('locations',
  {latitude: 39.9, longitude: 116.4}, 100.0) YIELD node
RETURN node.name, node.population
ORDER BY node.population DESC
```

### Graph Database Comparison for GIS

| Database | Graph Model | Spatial Support | Query Language | Scale |
|----------|------------|----------------|----------------|-------|
| **Neo4j** | Property graph | Native `point()` type, R-tree | Cypher | Billions of edges |
| **Amazon Neptune** | Property graph + RDF | GeoSPARQL, WKT | Gremlin, SPARQL | Managed, auto-scaling |
| **ArangoDB** | Multi-model | GeoJSON indexing | AQL | Billions of docs |
| **TigerGraph** | Property graph | Geospatial UDFs | GSQL | Trillions of edges |
| **Apache AGE** | Property graph (Postgres extension) | PostGIS integration | Cypher + SQL | Postgres scale |
| **JanusGraph** | Property graph | Geo predicates via Elasticsearch | Gremlin | Distributed |

---

## 15. Performance & Scaling

Rendering graphs with tens of thousands of nodes and millions of edges in a browser requires deliberate performance engineering. This section covers the techniques that make large graph visualization possible.

### WebGL Rendering Fundamentals

| Technique | Description | Impact |
|-----------|------------|--------|
| **Instanced rendering** | Draw all nodes in one draw call using instanced arrays | 100x speedup vs. individual draw calls |
| **Index buffer** | Share vertices between edges | 50% memory reduction for edge geometry |
| **Float packing** | Encode multiple attributes (color, size) into fewer floats | Reduced GPU memory bandwidth |
| **Texture atlas** | Store node icons in a single texture | One texture bind vs. per-node bind |
| **Stencil buffer** | Efficient hit testing without ray casting | Sub-ms hover/click detection |

### Level-of-Detail (LOD) for Large Graphs

When a graph has 100K+ nodes, you cannot render every element at every zoom level. LOD strategies:

| Zoom Level | Strategy | What to Show |
|------------|----------|-------------|
| **Overview** (zoom 0-4) | Node clustering + edge bundling | Cluster centroids, bundled meta-edges |
| **Region** (zoom 5-8) | Top-K nodes by centrality | Hub nodes, major corridors |
| **Detail** (zoom 9-12) | All nodes in viewport | Full local structure, labels |
| **Street** (zoom 13+) | Node + edge attributes | All details, tooltips, editing |

```javascript
// LOD implementation with deck.gl
function getGraphLayers(zoom, visibleBounds) {
  if (zoom < 5) {
    // Cluster nodes, show only cluster representatives
    return [
      new ScatterplotLayer({
        data: clusterCentroids,
        getPosition: d => [d.lon, d.lat],
        getRadius: d => Math.sqrt(d.clusterSize) * 500,
        getFillColor: [255, 107, 107, 180]
      }),
      new LineLayer({
        data: bundledEdges,
        getSourcePosition: d => d.sourceCluster,
        getTargetPosition: d => d.targetCluster,
        getWidth: d => Math.log(d.aggregateWeight),
        getColor: [74, 144, 217, 100]
      })
    ];
  } else {
    // Show individual nodes within viewport
    const visibleNodes = nodes.filter(n =>
      n.lon >= visibleBounds.west && n.lon <= visibleBounds.east &&
      n.lat >= visibleBounds.south && n.lat <= visibleBounds.north
    );
    return [
      new ScatterplotLayer({ data: visibleNodes, /* ... */ }),
      new LineLayer({ data: getEdgesForNodes(visibleNodes), /* ... */ })
    ];
  }
}
```

### Edge Bundling

Edge bundling reduces visual clutter by routing edges through shared pathways, similar to how roads converge into highways.

| Algorithm | Complexity | Quality | Implementation |
|-----------|-----------|---------|----------------|
| **Force-Directed Edge Bundling (FDEB)** | O(E * iterations) | High | [d3-force-bundle](https://github.com/nickcollins/d3-force-bundle) |
| **Kernel Density Edge Bundling (KDEEB)** | O(E * K) | High | Research implementations |
| **Divided Edge Bundling** | O(E * K) | Directional | [d3-edge-bundling](https://github.com) |
| **Hierarchical Edge Bundling** | O(E log N) | Good for trees | Built into D3 (`d3.lineRadial`) |
| **GPU-based bundling** | O(E) amortized | Real-time | Graphistry (built-in) |

```javascript
// Force-directed edge bundling with D3
import { ForceEdgeBundling } from "d3-force-bundle";

const bundling = ForceEdgeBundling()
  .nodes(nodes)
  .edges(edges)
  .bundling_stiffness(0.1)  // lower = more bundling
  .step_size(0.1)
  .iterations(60);

const bundledPaths = bundling();
// Each edge becomes a polyline with intermediate control points
// Render with d3.line() using curveBundle interpolation
```

### Graph Clustering for Scalability

For graphs exceeding render capacity, cluster nodes into super-nodes:

```python
import networkx as nx
from networkx.community import louvain_communities

# Detect communities for clustering
G = nx.read_graphml("large_network.graphml")
communities = louvain_communities(G, resolution=1.5)

# Build clustered graph: one super-node per community
G_clustered = nx.Graph()
for i, community in enumerate(communities):
    subgraph = G.subgraph(community)
    centroid_lon = sum(G.nodes[n]["lon"] for n in community) / len(community)
    centroid_lat = sum(G.nodes[n]["lat"] for n in community) / len(community)
    G_clustered.add_node(f"cluster_{i}",
                         size=len(community),
                         lon=centroid_lon, lat=centroid_lat)

# Add edges between clusters (aggregate edge weights)
community_map = {}
for i, community in enumerate(communities):
    for node in community:
        community_map[node] = i

for u, v, data in G.edges(data=True):
    cu, cv = community_map[u], community_map[v]
    if cu != cv:
        key = (f"cluster_{cu}", f"cluster_{cv}")
        if G_clustered.has_edge(*key):
            G_clustered.edges[key]["weight"] += data.get("weight", 1)
        else:
            G_clustered.add_edge(*key, weight=data.get("weight", 1))
```

### Viewport Culling and Spatial Indexing

Only render nodes and edges visible in the current viewport:

```javascript
// R-tree spatial index for fast viewport queries
import RBush from "rbush";

const tree = new RBush();
const items = nodes.map(node => ({
  minX: node.lon, minY: node.lat,
  maxX: node.lon, maxY: node.lat,
  node: node
}));
tree.load(items);

// Query visible nodes on viewport change
function getVisibleNodes(bounds) {
  return tree.search({
    minX: bounds.west, minY: bounds.south,
    maxX: bounds.east, maxY: bounds.north
  }).map(item => item.node);
}

// For edges: index by bounding box of source + target
const edgeTree = new RBush();
const edgeItems = edges.map(edge => ({
  minX: Math.min(edge.srcLon, edge.dstLon),
  minY: Math.min(edge.srcLat, edge.dstLat),
  maxX: Math.max(edge.srcLon, edge.dstLon),
  maxY: Math.max(edge.srcLat, edge.dstLat),
  edge: edge
}));
edgeTree.load(edgeItems);
```

### Performance Budget Guidelines

| Metric | Target | Consequence if Exceeded |
|--------|--------|------------------------|
| **Frame time** | <16ms (60 fps) | Laggy interaction, user frustration |
| **Initial load** | <3s for graph render | User abandonment |
| **Memory (JS heap)** | <500 MB | Tab crash on mobile, swap on desktop |
| **GPU memory** | <1 GB | Texture eviction, rendering artifacts |
| **Draw calls** | <100 per frame | CPU bottleneck, stuttering |

### Scaling Thresholds by Tool

| Tool | Smooth (<16ms) | Usable (<50ms) | Maximum |
|------|---------------|----------------|---------|
| **vis.js** | 2K nodes / 5K edges | 5K / 15K | ~10K / 30K |
| **Sigma.js v3** | 20K / 50K | 50K / 200K | ~100K / 500K |
| **D3 force (SVG)** | 500 / 1K | 2K / 5K | ~5K / 10K |
| **D3 force (Canvas)** | 5K / 10K | 10K / 30K | ~20K / 50K |
| **deck.gl LineLayer** | 100K / 500K | 500K / 2M | ~1M / 5M |
| **Graphistry (GPU)** | 200K / 1M | 1M / 5M | ~5M / 20M |
| **Gephi** | 10K / 50K | 50K / 200K | ~500K / 2M |

---

## 16. Tool Comparison Matrix

### Visualization Libraries

| Tool | Type | Renderer | Max Nodes | Spatial | Layout | Analysis | License | URL |
|------|------|----------|-----------|---------|--------|----------|---------|-----|
| **D3.js** | JS library | SVG/Canvas | ~5K (SVG) / ~20K (Canvas) | Map projection integration | Force, custom | None built-in | BSD-3 | [d3js.org](https://d3js.org) |
| **Sigma.js v3** | JS library | WebGL | ~100K | Manual positioning | ForceAtlas2, custom | Via graphology | MIT | [sigmajs.org](https://www.sigmajs.org) |
| **vis.js** | JS library | Canvas | ~10K | Manual positioning | Barnes-Hut, hierarchical | None | Apache-2.0 | [visjs.org](https://visjs.org) |
| **Cytoscape.js** | JS library | Canvas/WebGL | ~50K | cytoscape-leaflet | 30+ algorithms | Centrality, paths | MIT | [js.cytoscape.org](https://js.cytoscape.org) |
| **deck.gl** | JS framework | WebGL | ~1M edges | Native geo layers | N/A (data-driven) | None | MIT | [deck.gl](https://deck.gl) |
| **Kepler.gl** | Web app | WebGL (deck.gl) | ~500K points | Native | N/A | Filtering only | MIT | [kepler.gl](https://kepler.gl) |
| **Graphistry** | Cloud/self-hosted | GPU (WebGL) | ~5M | Lat/lon mapping | GPU force layout | GFQL, cuGraph | Commercial | [graphistry.com](https://www.graphistry.com) |
| **Gephi** | Desktop | OpenGL | ~500K | GeoLayout plugin | ForceAtlas2, Fruchterman-Reingold | Full suite | GPL | [gephi.org](https://gephi.org) |
| **Cytoscape Desktop** | Desktop | Java2D | ~50K | Plugins | 30+ algorithms | Full suite | LGPL | [cytoscape.org](https://cytoscape.org) |
| **Flowmap.blue** | Web app | WebGL (deck.gl) | ~50K flows | Native | N/A | Clustering | MIT | [flowmap.blue](https://www.flowmap.blue) |

### Analysis Libraries

| Library | Language | Speed | Max Scale | Spatial Integration | Best For |
|---------|----------|-------|-----------|-------------------|----------|
| **NetworkX** | Python | Slow | ~100K nodes | OSMnx bridge | Prototyping, teaching |
| **igraph** | C (Python/R) | Fast | ~1M nodes | Manual | Production analysis |
| **graph-tool** | C++ (Python) | Fastest | ~10M nodes | Manual | Research, statistical inference |
| **cuGraph** | CUDA (Python) | GPU | ~1B edges | Via cuSpatial | Big data, enterprise |
| **Networkit** | C++ (Python) | Fast (parallel) | ~1B edges | Manual | Large-scale social networks |
| **Neo4j** | Java | Fast (indexed) | Billions | Native `point()` | Knowledge graphs, enterprise |
| **OSMnx** | Python | Moderate | City-scale | Native | Urban network analysis |
| **WNTR** | Python | Moderate | Utility-scale | EPANET models | Water network resilience |
| **Circuitscape** | Julia | Fast | Landscape-scale | Raster-based | Habitat connectivity |

### Decision Flowchart

```
START: What is your graph visualization need?
  |
  |-- How many nodes/edges?
  |     |-- < 5K nodes     --> vis.js (quickest setup)
  |     |-- 5K - 50K nodes --> Sigma.js v3 or Cytoscape.js
  |     |-- 50K - 500K     --> Gephi (desktop) or deck.gl (web)
  |     |-- > 500K         --> Graphistry (GPU) or deck.gl
  |
  |-- Need geographic positioning?
  |     |-- Yes, on a map    --> deck.gl, Kepler.gl, D3+Leaflet
  |     |-- Yes, rough layout --> Gephi GeoLayout, Sigma.js
  |     |-- No, topology only --> D3 force, Gephi ForceAtlas2
  |
  |-- Need built-in analysis?
  |     |-- Centrality, paths, communities --> Cytoscape.js, Gephi
  |     |-- Python analysis pipeline        --> NetworkX, igraph
  |     |-- GPU-scale analysis              --> cuGraph + Graphistry
  |
  |-- Origin-destination flows?
  |     |-- No code         --> Flowmap.blue, Kepler.gl
  |     |-- Programmatic    --> deck.gl ArcLayer
  |     |-- Static print    --> matplotlib + geopandas desire lines
  |
  |-- Knowledge graph / RDF?
  |     |-- Yes --> Neo4j Bloom, Apache Jena + D3
  |     |-- No  --> See above by scale
```

### When to Use What: Quick Reference

| Scenario | Primary Tool | Alternative |
|----------|-------------|-------------|
| City road network analysis | OSMnx + NetworkX | igraph for speed |
| Transit system mapping | Partridge + Folium | Kepler.gl for no-code |
| OD flow visualization | deck.gl ArcLayer | Flowmap.blue for no-code |
| Social network on map | Sigma.js + Leaflet | Graphistry for scale |
| Infrastructure vulnerability | WNTR / NetworkX | Neo4j for persistent storage |
| Ecological connectivity | Circuitscape | Conefor for patch metrics |
| Supply chain mapping | NetworkX + matplotlib | Graphistry for live dashboards |
| Knowledge graph exploration | Neo4j Bloom | Cytoscape Desktop |
| Publication-quality render | Gephi | D3.js for web-native |
| Real-time million-edge graph | Graphistry | deck.gl LineLayer |

---

## Further Reading

- Barthelemy, M. (2011). *Spatial Networks*. Physics Reports, 499(1-3), 1-101. -- The foundational review of spatial network science.
- Boeing, G. (2017). *OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks*. Computers, Environment and Urban Systems, 65, 126-139.
- Saura, S. & Torne, J. (2009). *Conefor Sensinode 2.2: A Software Package for Quantifying the Importance of Habitat Patches for Landscape Connectivity*. Environmental Modelling & Software.
- McRae, B.H. et al. (2008). *Using Circuit Theory to Model Connectivity in Ecology, Evolution, and Conservation*. Ecology, 89(10), 2712-2724.
- Jacomy, M. et al. (2014). *ForceAtlas2, a Continuous Graph Layout Algorithm for Handy Network Visualization*. PLOS ONE.

**Cross-ref:** [Scientific Visualization](scientific-visualization.md) for advanced network rendering techniques | [AI/ML Visualization](ai-ml-visualization.md) for graph neural network visualization | [Storytelling & Scrollytelling](storytelling-scrollytelling.md) for narrative network presentations.