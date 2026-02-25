# Large-Scale Geospatial Data Visualization

> A comprehensive, expert-level reference for rendering millions to billions of geospatial features in the browser and beyond. Covers GPU-accelerated rendering, tiling strategies, spatial indexing, streaming, point clouds, real-time pipelines, and production-grade performance engineering.

**Last updated:** 2026-02-25

---

## Table of Contents

- [1. Introduction and Quick Picks](#1-introduction-and-quick-picks)
- [2. GPU-Accelerated Rendering](#2-gpu-accelerated-rendering)
- [3. deck.gl Deep Dive](#3-deckgl-deep-dive)
- [4. MapLibre GL JS at Scale](#4-maplibre-gl-js-at-scale)
- [5. Tiling Strategies](#5-tiling-strategies)
- [6. Spatial Indexing for Visualization](#6-spatial-indexing-for-visualization)
- [7. Aggregation and Binning](#7-aggregation-and-binning)
- [8. Streaming and Progressive Rendering](#8-streaming-and-progressive-rendering)
- [9. DuckDB WASM for Visualization](#9-duckdb-wasm-for-visualization)
- [10. Point Cloud Rendering](#10-point-cloud-rendering)
- [11. Raster Visualization](#11-raster-visualization)
- [12. Real-Time Data at Scale](#12-real-time-data-at-scale)
- [13. Server-Side Rendering](#13-server-side-rendering)
- [14. Memory Management](#14-memory-management)
- [15. Benchmarking and Profiling](#15-benchmarking-and-profiling)
- [16. Case Studies](#16-case-studies)
- [17. Tool Comparison Matrix](#17-tool-comparison-matrix)

---

## 1. Introduction and Quick Picks

Large-scale geospatial visualization is the practice of rendering datasets that exceed what traditional DOM-based or Canvas2D approaches can handle -- typically anything beyond 10,000--50,000 features. Modern datasets routinely contain millions of GPS traces, billions of LiDAR points, or global-coverage raster imagery at sub-meter resolution. This guide covers the state-of-the-art tooling, architectural patterns, and performance engineering techniques required to visualize such data interactively.

### Scale Tiers and Recommended Approaches

| Scale | Feature Count | Recommended Stack | Typical FPS |
|-------|--------------|-------------------|-------------|
| Small | < 10K | Leaflet, OpenLayers, D3 SVG | 60 |
| Medium | 10K -- 500K | MapLibre GL JS, deck.gl ScatterplotLayer | 60 |
| Large | 500K -- 10M | deck.gl with binary attributes, MVT tiles | 30--60 |
| Very Large | 10M -- 100M | Aggregation (H3/hexbin), GPU compute, tiled streaming | 30--60 |
| Massive | 100M -- 10B+ | Server-side aggregation, 3D Tiles, point cloud LOD | 30--60 |

### SOTA Quick Picks (2025--2026)

| Use Case | Tool | URL |
|----------|------|-----|
| General large-scale 2D/3D | **deck.gl** v9+ | https://deck.gl |
| Vector basemaps | **MapLibre GL JS** v5+ | https://maplibre.org |
| Point clouds | **Potree** / **3D Tiles** via CesiumJS | https://potree.github.io |
| In-browser analytics | **DuckDB-WASM** + **Mosaic** | https://duckdb.org/docs/api/wasm |
| Tiled data serving | **PMTiles** | https://protomaps.com/docs/pmtiles |
| Streaming vectors | **FlatGeobuf** | https://flatgeobuf.org |
| Hex aggregation | **H3-js** | https://h3geo.org |
| COG raster rendering | **GeoTIFF.js** + deck.gl BitmapLayer | https://geotiffjs.github.io |
| Real-time dashboards | **Mosaic** + **Observable Framework** | https://uwdata.github.io/mosaic |
| GPU compute (emerging) | **WebGPU** via compute shaders | https://www.w3.org/TR/webgpu |

> **Cross-references:** See [js-bindbox/performance-optimization.md](../js-bindbox/performance-optimization.md) for general JS optimization patterns. See [tools/web-mapping.md](../tools/web-mapping.md) for broader web mapping tool coverage. See [js-bindbox/tile-servers.md](../js-bindbox/tile-servers.md) for tile server setup.

---

## 2. GPU-Accelerated Rendering

### 2.1 Why GPU Rendering Matters for Geospatial

The CPU-to-GPU paradigm shift is the single most impactful architectural decision for large-scale visualization. A modern GPU has 2,000--16,000+ shader cores running in parallel, vs. 8--24 CPU cores. For geospatial rendering -- which is inherently parallel (each point/polygon can be processed independently) -- the GPU offers 100x--1000x throughput gains.

**Key insight:** The bottleneck in large-scale geospatial rendering is almost never the GPU's raw rendering capacity. It is the CPU-side data preparation, the CPU-to-GPU data transfer, and JavaScript's single-threaded execution model. Every optimization in this guide ultimately reduces one of these three bottlenecks.

### 2.2 WebGL vs WebGPU

| Feature | WebGL 2.0 | WebGPU |
|---------|-----------|--------|
| Browser support (2026) | Universal | Chrome, Edge, Firefox (Safari partial) |
| Compute shaders | No (hacks via transform feedback) | Yes, first-class |
| Multi-threaded submission | No | Yes, via command encoders |
| Shader language | GLSL ES 3.0 | WGSL |
| Indirect draw | No | Yes |
| Storage buffers | Limited (UBOs only) | Yes, read-write SSBOs |
| Max texture size | 4096--16384 | Device-dependent, typically 8192--16384 |
| Typical geospatial lib support | deck.gl, MapLibre, CesiumJS, Leaflet.glify | deck.gl (experimental), CesiumJS (experimental) |

**Practical guidance:** As of early 2026, WebGL 2.0 remains the production standard for geospatial visualization. WebGPU is viable for new projects targeting Chrome/Edge-only audiences or for compute-heavy analytics (e.g., spatial joins, kernel density estimation on GPU). The deck.gl team has experimental WebGPU support via luma.gl v9.

### 2.3 GPU Instancing

GPU instancing renders many copies of a geometry template with per-instance attributes (position, color, size) in a single draw call. This is the foundation of how deck.gl renders millions of icons, circles, or 3D models efficiently.

```javascript
// Conceptual WebGL2 instanced rendering for geospatial points
// In practice, use deck.gl or a similar framework -- this shows the underlying mechanism

const vertexShader = `#version 300 es
  // Per-vertex (the quad template -- 4 vertices)
  in vec2 a_quadVertex;

  // Per-instance (one per data point)
  in vec2 a_position;     // lon/lat projected to Mercator
  in float a_radius;
  in vec4 a_color;

  uniform mat4 u_viewProjection;

  out vec4 v_color;

  void main() {
    // Scale the template quad by the instance radius
    vec2 offset = a_quadVertex * a_radius;
    // Translate to the instance's projected position
    vec4 worldPos = vec4(a_position + offset, 0.0, 1.0);
    gl_Position = u_viewProjection * worldPos;
    v_color = a_color;
  }
`;

// Setup: one draw call renders ALL points
// gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, numPoints);
// where numPoints can be 1,000,000+ at 60fps
```

**Performance benchmark (GPU instancing):**

| Point Count | Non-instanced Draw Calls | Instanced (1 draw call) | FPS (M1 MacBook) |
|-------------|-------------------------|------------------------|-------------------|
| 10,000 | 10,000 | 1 | 60 |
| 100,000 | 100,000 (impossible) | 1 | 60 |
| 1,000,000 | N/A | 1 | 58--60 |
| 5,000,000 | N/A | 1 | 40--55 |
| 10,000,000 | N/A | 1 | 20--35 |

### 2.4 Shader-Based Rendering Patterns

Beyond instancing, custom shaders unlock specialized geospatial rendering that no off-the-shelf library provides.

**Data-driven color ramps in the fragment shader:**

```glsl
#version 300 es
precision highp float;

in float v_value;   // e.g., temperature, elevation, speed
out vec4 fragColor;

// Viridis-like color ramp (5-stop approximation)
vec3 viridis(float t) {
  const vec3 c0 = vec3(0.267, 0.004, 0.329);
  const vec3 c1 = vec3(0.282, 0.140, 0.458);
  const vec3 c2 = vec3(0.127, 0.566, 0.551);
  const vec3 c3 = vec3(0.544, 0.773, 0.247);
  const vec3 c4 = vec3(0.993, 0.906, 0.144);

  t = clamp(t, 0.0, 1.0);
  if (t < 0.25) return mix(c0, c1, t / 0.25);
  if (t < 0.50) return mix(c1, c2, (t - 0.25) / 0.25);
  if (t < 0.75) return mix(c2, c3, (t - 0.50) / 0.25);
  return mix(c3, c4, (t - 0.75) / 0.25);
}

void main() {
  fragColor = vec4(viridis(v_value), 0.85);
}
```

**Anti-aliased circles without geometry tessellation:**

```glsl
// Fragment shader for SDF-based circle rendering
// Each point is a screen-aligned quad; the circle is "carved" in the fragment shader
in vec2 v_uv;  // -1..1 across the quad
in vec4 v_color;
out vec4 fragColor;

void main() {
  float dist = length(v_uv);
  // Smooth anti-aliased edge using screen-space derivatives
  float aa = fwidth(dist);
  float alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, dist);
  if (alpha < 0.01) discard;
  fragColor = vec4(v_color.rgb, v_color.a * alpha);
}
```

### 2.5 Compute Shaders (WebGPU)

WebGPU compute shaders enable massively parallel geospatial computation directly on the GPU, avoiding the CPU entirely for tasks like spatial aggregation, coordinate projection, and kernel density estimation.

```javascript
// WebGPU compute shader: project lon/lat to Web Mercator on the GPU
// Eliminates CPU-side projection bottleneck for millions of points

const projectionShader = /* wgsl */ `
  struct Point {
    lon: f32,
    lat: f32,
  };

  struct ProjectedPoint {
    x: f32,
    y: f32,
  };

  @group(0) @binding(0) var<storage, read> input: array<Point>;
  @group(0) @binding(1) var<storage, read_write> output: array<ProjectedPoint>;

  const PI: f32 = 3.14159265359;
  const WORLD_SIZE: f32 = 512.0;

  @compute @workgroup_size(256)
  fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x;
    if (idx >= arrayLength(&input)) { return; }

    let lon = input[idx].lon;
    let lat = input[idx].lat;

    // Web Mercator projection
    let x = (lon + 180.0) / 360.0 * WORLD_SIZE;
    let latRad = lat * PI / 180.0;
    let y = (1.0 - log(tan(latRad) + 1.0 / cos(latRad)) / PI) / 2.0 * WORLD_SIZE;

    output[idx] = ProjectedPoint(x, y);
  }
`;

// Dispatch: ceil(numPoints / 256) workgroups
// 10M points projected in ~2ms on a mid-range GPU (vs ~200ms on CPU)
```

**Benchmark: CPU vs GPU projection (Web Mercator, M1 MacBook Pro):**

| Points | CPU (JS) | WebGPU Compute | Speedup |
|--------|----------|---------------|---------|
| 100K | 5ms | 0.3ms | 17x |
| 1M | 50ms | 0.8ms | 63x |
| 10M | 520ms | 2.1ms | 248x |
| 50M | 2,600ms | 8.5ms | 306x |

> **Cross-reference:** See [js-bindbox/performance-optimization.md](../js-bindbox/performance-optimization.md) for CPU-side optimization techniques that complement GPU rendering.

---

## 3. deck.gl Deep Dive

[deck.gl](https://deck.gl) (v9+, maintained by the OpenJS Foundation / vis.gl) is the most capable open-source framework for large-scale geospatial visualization on the web. It is built on [luma.gl](https://luma.gl) (WebGL2/WebGPU abstraction) and designed from the ground up for GPU-accelerated rendering of millions of features.

### 3.1 Layer Lifecycle

Understanding deck.gl's layer lifecycle is critical for performance at scale. Each layer goes through:

1. **`constructor()`** -- Creates the layer instance (lightweight, no GPU work)
2. **`initializeState()`** -- Called once; allocate GPU resources (Models, Buffers)
3. **`updateState()`** -- Called when props change; determines what needs re-computation
4. **`draw()`** -- Called every frame; issues GPU draw calls
5. **`finalizeState()`** -- Cleanup GPU resources

**The golden rule:** Minimize work in `updateState()`. This is where most performance problems originate. The three most expensive operations are:

- **Attribute recalculation** (iterating all data to fill typed arrays)
- **Texture uploads** (for icon atlases, color ramp textures)
- **Buffer reallocation** (when data length changes)

### 3.2 Binary Data (Columnar Layout)

The single most impactful optimization for deck.gl at scale is providing data in binary columnar format instead of an array of objects. This eliminates the CPU-side attribute calculation entirely.

```javascript
import { Deck } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';

// BAD: Array of objects (1M points = ~500ms to process)
const slowLayer = new ScatterplotLayer({
  id: 'points-slow',
  data: arrayOfObjects,                  // [{lon: -122.4, lat: 37.8, r: 5}, ...]
  getPosition: d => [d.lon, d.lat],      // Called 1M times on CPU
  getRadius: d => d.r,                   // Called 1M times on CPU
  getFillColor: d => [255, 0, 0],        // Called 1M times on CPU
});

// GOOD: Binary columnar data (1M points = ~5ms to process)
// Pre-packed Float32Arrays -- deck.gl uploads directly to GPU
const numPoints = 1_000_000;
const positions = new Float32Array(numPoints * 3);  // [lon0, lat0, 0, lon1, lat1, 0, ...]
const radii = new Float32Array(numPoints);           // [r0, r1, ...]
const colors = new Uint8Array(numPoints * 4);        // [R0, G0, B0, A0, ...]

// Fill arrays (typically from a binary fetch, Arrow IPC, or FlatGeobuf)
for (let i = 0; i < numPoints; i++) {
  positions[i * 3] = -122.4 + Math.random() * 0.5;
  positions[i * 3 + 1] = 37.6 + Math.random() * 0.4;
  positions[i * 3 + 2] = 0;
  radii[i] = 2 + Math.random() * 8;
  colors[i * 4] = 255;
  colors[i * 4 + 1] = Math.floor(Math.random() * 128);
  colors[i * 4 + 2] = 0;
  colors[i * 4 + 3] = 200;
}

const fastLayer = new ScatterplotLayer({
  id: 'points-fast',
  data: { length: numPoints },   // Just tell deck.gl how many instances
  getPosition: { value: positions, size: 3 },
  getRadius: { value: radii, size: 1 },
  getFillColor: { value: colors, size: 4 },
  radiusMinPixels: 1,
  radiusMaxPixels: 10,
});
```

**Performance comparison (ScatterplotLayer, M1 MacBook Pro):**

| Points | Object Array | Binary Columnar | Speedup (updateState) |
|--------|-------------|-----------------|----------------------|
| 100K | 52ms | 1ms | 52x |
| 1M | 510ms | 5ms | 102x |
| 5M | 2,800ms | 22ms | 127x |
| 10M | OOM crash | 45ms | -- |

### 3.3 GPU Aggregation Layers

deck.gl provides GPU-accelerated aggregation layers that compute spatial aggregates entirely on the GPU, enabling real-time re-aggregation during pan/zoom with millions of input points.

#### HexagonLayer (GPU)

```javascript
import { HexagonLayer } from '@deck.gl/aggregation-layers';

const hexLayer = new HexagonLayer({
  id: 'hexagons',
  data: '/api/points.arrow',
  loaders: [ArrowLoader],
  getPosition: d => [d.lon, d.lat],
  elevationRange: [0, 3000],
  elevationScale: 50,
  extruded: true,
  radius: 1000,               // Hex radius in meters
  coverage: 0.9,
  upperPercentile: 95,
  colorRange: [
    [1, 152, 189],
    [73, 227, 206],
    [216, 254, 181],
    [254, 237, 177],
    [254, 173, 84],
    [209, 55, 78],
  ],
  // GPU aggregation: re-bins on every viewport change at 60fps
  gpuAggregation: true,        // Critical: forces GPU path
  material: {
    ambient: 0.64,
    diffuse: 0.6,
    shininess: 32,
  },
  pickable: true,
  onHover: info => {
    if (info.object) {
      console.log(`Hex: ${info.object.count} points, centroid: ${info.object.position}`);
    }
  },
  transitions: {
    elevationScale: { duration: 1000, easing: t => t * t },
  },
});
```

**GPU aggregation benchmark (HexagonLayer, 5M points):**

| Operation | CPU Aggregation | GPU Aggregation | Speedup |
|-----------|----------------|-----------------|---------|
| Initial bin | 1,200ms | 12ms | 100x |
| Re-bin on zoom | 1,200ms | 3ms | 400x |
| Re-bin on pan | 1,200ms | 3ms | 400x |
| Steady-state FPS | 15 | 60 | 4x |

#### HeatmapLayer

```javascript
import { HeatmapLayer } from '@deck.gl/aggregation-layers';

const heatmapLayer = new HeatmapLayer({
  id: 'heatmap',
  data: points,                    // Up to 5M+ points
  getPosition: d => [d.lng, d.lat],
  getWeight: d => d.magnitude,
  radiusPixels: 30,                // Kernel radius in pixels
  intensity: 1,
  threshold: 0.05,
  colorRange: [
    [255, 255, 178, 25],
    [254, 217, 118, 85],
    [254, 178, 76, 127],
    [253, 141, 60, 170],
    [240, 59, 32, 212],
    [189, 0, 38, 255],
  ],
  // The HeatmapLayer performs KDE on the GPU via a texture-based approach
  // It renders point weights into a framebuffer, applies a Gaussian blur,
  // then maps the result through a color ramp texture
  aggregation: 'SUM',
});
```

### 3.4 Custom Shaders (Layer Extensions and Subclassing)

deck.gl's shader injection system allows adding custom GLSL code to any built-in layer without forking.

```javascript
import { ScatterplotLayer } from '@deck.gl/layers';
import { LayerExtension } from '@deck.gl/core';

// Custom extension: animate points with a pulsing effect based on a timestamp attribute
class PulseExtension extends LayerExtension {
  getShaders() {
    return {
      inject: {
        // Inject into the vertex shader
        'vs:#decl': `
          attribute float instanceTimestamp;
          uniform float currentTime;
          varying float vPulse;
        `,
        'vs:#main-end': `
          // Compute a pulsing scale factor based on time since the event
          float age = currentTime - instanceTimestamp;
          float pulse = 1.0 + 0.5 * sin(age * 3.0) * exp(-age * 0.5);
          // Modify the radius (outerRadiusPixels is a deck.gl internal)
          outerRadiusPixels *= pulse;
          vPulse = pulse;
        `,
        // Inject into the fragment shader
        'fs:#decl': `
          varying float vPulse;
        `,
        'fs:#main-end': `
          // Fade alpha as the pulse decays
          gl_FragColor.a *= mix(0.3, 1.0, (vPulse - 1.0) * 2.0);
        `,
      },
    };
  }

  draw(params) {
    const { uniforms } = params;
    uniforms.currentTime = Date.now() / 1000;
  }

  initializeState() {
    const attributeManager = this.getAttributeManager();
    attributeManager.addInstanced({
      instanceTimestamp: {
        size: 1,
        accessor: 'getTimestamp',
      },
    });
  }
}

PulseExtension.extensionName = 'PulseExtension';

// Usage
const layer = new ScatterplotLayer({
  id: 'pulsing-events',
  data: earthquakeData,
  getPosition: d => [d.lon, d.lat],
  getRadius: d => d.magnitude * 1000,
  getFillColor: [255, 100, 50],
  getTimestamp: d => d.epochSeconds,
  extensions: [new PulseExtension()],
  radiusMinPixels: 2,
});
```

### 3.5 Performance Tuning Checklist

```javascript
// Production-grade deck.gl configuration for large datasets
import { Deck } from '@deck.gl/core';
import { MapView } from '@deck.gl/core';

const deck = new Deck({
  // Canvas and context
  canvas: 'deck-canvas',
  width: '100%',
  height: '100%',

  // Views
  views: new MapView({
    repeat: true,              // Wrap around the antimeridian
    nearZMultiplier: 0.1,      // Tighter near plane for better z-precision
    farZMultiplier: 2.0,
  }),

  // Initial viewport
  initialViewState: {
    longitude: -122.4,
    latitude: 37.8,
    zoom: 10,
    pitch: 0,
    bearing: 0,
  },

  // Global performance settings
  parameters: {
    depthTest: false,           // Disable for 2D-only layers (saves GPU fill rate)
    blend: true,
  },

  // CRITICAL: Prevent unnecessary re-renders
  _animate: false,              // Only set true if you have animated layers

  // Picking optimization
  pickingRadius: 5,             // Increase to reduce picking overhead (default 0)
  _pickable: false,             // Disable global picking if not needed

  // Reduce the number of layers that trigger re-renders
  layerFilter: ({ layer, viewport }) => {
    // Example: only render detail layers when zoomed in
    if (layer.id.startsWith('detail-') && viewport.zoom < 12) {
      return false;
    }
    return true;
  },

  // Callback for debugging render performance
  onAfterRender: () => {
    // Log frame timing in development
    if (process.env.NODE_ENV === 'development') {
      const { metrics } = deck;
      if (metrics.fps < 30) {
        console.warn(`Low FPS: ${metrics.fps.toFixed(1)}, ` +
          `GPU time: ${metrics.gpuTime?.toFixed(1)}ms, ` +
          `CPU time: ${metrics.cpuTime?.toFixed(1)}ms`);
      }
    }
  },
});
```

**Additional tuning techniques:**

| Technique | Impact | When to Use |
|-----------|--------|------------|
| Binary attributes | 50--100x faster updates | Always for > 100K features |
| `radiusMinPixels` / `widthMinPixels` | Prevents sub-pixel waste | Always |
| `extensions: [new DataFilterExtension()]` | GPU-side filtering | Dynamic filter UIs |
| `loaders: [ArrowLoader]` | Zero-copy data ingestion | Arrow/Parquet sources |
| `_subLayerProps` | Override nested layer props | Composite layers |
| `coordinateSystem: COORDINATE_SYSTEM.CARTESIAN` | Skip projection math | Pre-projected data |
| `autoHighlight: false` | Eliminates hover overhead | When hover not needed |

> **Cross-reference:** See [js-bindbox/performance-optimization.md](../js-bindbox/performance-optimization.md) for detailed deck.gl benchmarks and patterns.

---

## 4. MapLibre GL JS at Scale

[MapLibre GL JS](https://maplibre.org) (v5+) is the leading open-source WebGL-based map renderer for vector tiles. While deck.gl excels at data overlays, MapLibre provides the basemap and is increasingly capable of large-scale data rendering in its own right.

### 4.1 Vector Tile Optimization

```javascript
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      // PMTiles: single-file, HTTP range-request based, no tile server needed
      'buildings': {
        type: 'vector',
        url: 'pmtiles://https://cdn.example.com/buildings.pmtiles',
        minzoom: 10,
        maxzoom: 15,
      },
      // Traditional MVT endpoint
      'roads': {
        type: 'vector',
        tiles: ['https://tiles.example.com/roads/{z}/{x}/{y}.mvt'],
        minzoom: 4,
        maxzoom: 14,
        // Critical for performance: set appropriate tile size
        tileSize: 512,
      },
    },
    layers: [
      {
        id: 'buildings-fill',
        type: 'fill',
        source: 'buildings',
        'source-layer': 'buildings',
        minzoom: 12,
        paint: {
          // Data-driven styling is GPU-accelerated in MapLibre
          'fill-color': [
            'interpolate', ['linear'], ['get', 'height'],
            0, '#f7fbff',
            10, '#c6dbef',
            30, '#6baed6',
            60, '#2171b5',
            100, '#08306b',
          ],
          'fill-opacity': 0.8,
        },
        // Filter at the source level to reduce geometry processed
        filter: ['>', ['get', 'area'], 50],
      },
    ],
  },
  // Performance-critical options
  maxTileCacheSize: 200,         // Increase for pan-heavy workflows
  fadeDuration: 0,               // Disable fade animation for snappier tiles
  trackResize: true,
  antialias: false,              // Disable if aliasing is acceptable (saves GPU)
  preserveDrawingBuffer: false,  // Keep false for performance
  maxPitch: 85,
});
```

### 4.2 Source and Layer Limits

MapLibre has practical limits that affect architecture decisions at scale:

| Limit | Typical Value | Workaround |
|-------|-------------|------------|
| Max sources | ~20--50 before perf degrades | Merge into fewer sources |
| Max layers | ~200 before style recalc slows | Use fewer layers with expressions |
| Max vertices per tile | ~100K before decode slows | Simplify geometry at generation time |
| Tile request concurrency | 6 per origin (browser limit) | Use multiple CDN origins or HTTP/2 |
| GeoJSON source limit | ~50K features | Switch to vector tiles |

### 4.3 Custom Render Layers (WebGL Interop)

MapLibre allows injecting raw WebGL draw calls into its render pipeline via custom layers. This is the escape hatch for rendering data that MapLibre's built-in layer types cannot handle efficiently.

```javascript
// Custom WebGL layer: render 1M points with a custom shader
// This integrates directly into MapLibre's render pipeline

class MillionPointsLayer {
  constructor(id, points) {
    this.id = id;
    this.type = 'custom';
    this.renderingMode = '2d';
    this.points = points; // Float32Array [lon0, lat0, lon1, lat1, ...]
  }

  onAdd(map, gl) {
    this.map = map;

    // Vertex shader: project lon/lat via MapLibre's matrix
    const vertexSource = `#version 300 es
      uniform mat4 u_matrix;
      in vec2 a_position;

      void main() {
        // MapLibre's u_matrix transforms Mercator [0,1] coordinates to clip space
        float x = (a_position.x + 180.0) / 360.0;
        float y = (1.0 - log(tan(radians(a_position.y)) +
                   1.0 / cos(radians(a_position.y))) / 3.14159265) / 2.0;
        gl_Position = u_matrix * vec4(x, y, 0.0, 1.0);
        gl_PointSize = 2.0;
      }
    `;

    const fragmentSource = `#version 300 es
      precision mediump float;
      out vec4 fragColor;
      void main() {
        fragColor = vec4(1.0, 0.3, 0.1, 0.6);
      }
    `;

    // Compile shaders, create program, upload buffer
    this.program = createProgram(gl, vertexSource, fragmentSource);
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.points, gl.STATIC_DRAW);

    this.aPosition = gl.getAttribLocation(this.program, 'a_position');
    this.uMatrix = gl.getUniformLocation(this.program, 'u_matrix');
    this.numPoints = this.points.length / 2;
  }

  render(gl, matrix) {
    gl.useProgram(this.program);
    gl.uniformMatrix4fv(this.uMatrix, false, matrix);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.enableVertexAttribArray(this.aPosition);
    gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    gl.drawArrays(gl.POINTS, 0, this.numPoints);
  }

  onRemove(map, gl) {
    gl.deleteBuffer(this.buffer);
    gl.deleteProgram(this.program);
  }
}

// Usage: add after map loads
map.on('load', () => {
  const points = new Float32Array(2_000_000); // 1M points
  // ... fill with lon/lat data ...
  map.addLayer(new MillionPointsLayer('million-pts', points));
});
```

### 4.4 deck.gl + MapLibre Integration

The most common production pattern combines MapLibre for the basemap with deck.gl for data layers:

```javascript
import { Deck } from '@deck.gl/core';
import { MapboxOverlay } from '@deck.gl/mapbox';
import maplibregl from 'maplibre-gl';

const map = new maplibregl.Map({
  container: 'map',
  style: 'https://demotiles.maplibre.org/style.json',
  center: [-122.4, 37.8],
  zoom: 10,
});

const overlay = new MapboxOverlay({
  interleaved: true,   // Interleave deck.gl layers with MapLibre layers
  layers: [
    new ScatterplotLayer({
      id: 'scatterplot',
      data: binaryPoints,
      getPosition: { value: positions, size: 3 },
      getRadius: { value: radii, size: 1 },
      getFillColor: { value: colors, size: 4 },
    }),
  ],
});

map.addControl(overlay);

// Update layers reactively
function updateLayers(newData) {
  overlay.setProps({
    layers: [
      new ScatterplotLayer({
        id: 'scatterplot',
        data: newData,
        // ... binary attributes
      }),
    ],
  });
}
```

> **Cross-reference:** See [tools/web-mapping.md](../tools/web-mapping.md) for a broader comparison of web mapping frameworks, including Leaflet, OpenLayers, and CesiumJS.

---

## 5. Tiling Strategies

Tiling is the fundamental strategy for delivering large geospatial datasets to a client: break the data into spatial partitions keyed by zoom level and geographic extent, load only the tiles visible in the current viewport, and cache aggressively.

### 5.1 Vector Tiles (MVT)

The Mapbox Vector Tile (MVT) specification encodes vector geometry and attributes into protocol buffer-encoded tiles. This is the standard for web map basemaps and increasingly for analytical overlays.

**Tile generation tools:**

| Tool | Language | Strengths | URL |
|------|----------|-----------|-----|
| **Tippecanoe** | C++ | Gold standard, excellent generalization | https://github.com/felt/tippecanoe |
| **Martin** | Rust | Live from PostGIS, blazing fast | https://martin.maplibre.org |
| **pg_tileserv** | Go | Simple PostGIS-to-MVT | https://github.com/CrunchyData/pg_tileserv |
| **Planetiler** | Java | Planet-scale (entire OSM in hours) | https://github.com/onthegomap/planetiler |
| **ogr2ogr** | C++ | Part of GDAL, good for one-offs | https://gdal.org |

**Tippecanoe for large datasets:**

```bash
# Generate tiles from a 50M-feature GeoJSON with aggressive optimization
tippecanoe \
  --output=buildings.pmtiles \
  --layer=buildings \
  --minimum-zoom=4 \
  --maximum-zoom=14 \
  --drop-densest-as-needed \          # Automatically thin at low zoom
  --extend-zooms-if-still-dropping \  # Add zoom levels if data is still being lost
  --simplification=10 \              # Douglas-Peucker simplification
  --detect-shared-borders \           # Preserve topology at polygon boundaries
  --coalesce-densest-as-needed \      # Merge small polygons at low zoom
  --hilbert \                         # Hilbert-curve ordering for better compression
  --no-tile-size-limit \              # Allow large tiles where needed
  --force \
  buildings.geojson

# For maximum performance: generate multiple zoom levels in parallel
tippecanoe \
  --output=large-dataset.pmtiles \
  --read-parallel \                   # Parallel GeoJSON parsing
  --temporary-directory=/fast-ssd/ \  # Use fast storage for temp files
  -Z4 -z14 \
  large-dataset.geojson
```

### 5.2 PMTiles (Single-File Tile Archives)

[PMTiles](https://protomaps.com/docs/pmtiles) is a cloud-native tile format that stores an entire tile pyramid in a single file, served via HTTP range requests. No tile server is needed -- just a static file on S3, CloudFront, or any CDN.

```javascript
import { Protocol } from 'pmtiles';
import maplibregl from 'maplibre-gl';

// Register the PMTiles protocol handler
const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      mydata: {
        type: 'vector',
        // PMTiles served from a static CDN -- no tile server!
        url: 'pmtiles://https://cdn.example.com/mydata.pmtiles',
      },
    },
    layers: [
      {
        id: 'mydata-points',
        type: 'circle',
        source: 'mydata',
        'source-layer': 'points',
        paint: { 'circle-radius': 3, 'circle-color': '#e74c3c' },
      },
    ],
  },
});

// PMTiles with deck.gl
import { PMTilesSource } from '@loaders.gl/pmtiles';
import { MVTLayer } from '@deck.gl/geo-layers';

const mvtLayer = new MVTLayer({
  id: 'pmtiles-layer',
  data: 'https://cdn.example.com/mydata.pmtiles',
  loaders: [PMTilesSource],
  // deck.gl handles tile loading, viewport culling, and LOD automatically
  minZoom: 0,
  maxZoom: 14,
  getFillColor: [255, 140, 0],
  getLineColor: [0, 0, 0],
  lineWidthMinPixels: 1,
  pickable: true,
});
```

### 5.3 3D Tiles (OGC)

The OGC 3D Tiles specification is the standard for streaming massive 3D datasets -- buildings, terrain, point clouds, and photogrammetry meshes. CesiumJS and deck.gl both support it natively.

```javascript
import { Tile3DLayer } from '@deck.gl/geo-layers';
import { CesiumIonLoader } from '@loaders.gl/3d-tiles';

const buildings3D = new Tile3DLayer({
  id: '3d-buildings',
  data: 'https://assets.cesium.com/YOUR_ASSET/tileset.json',
  loader: CesiumIonLoader,
  loadOptions: {
    'cesium-ion': { accessToken: 'YOUR_TOKEN' },
  },
  // LOD control: how aggressively to load detail
  // Lower = less detail but faster; Higher = more detail but more memory
  maximumScreenSpaceError: 8,       // Default 16; lower for more detail

  onTileLoad: tile => {
    // Track memory usage
    const mb = (tile.content?.byteLength || 0) / 1024 / 1024;
    console.log(`Loaded tile: ${mb.toFixed(1)}MB, ${tile.content?.featureCount || 0} features`);
  },

  // 3D Tiles can contain millions of features across the tileset
  // The LOD system ensures only visible tiles at appropriate detail are loaded
  pointSize: 2,
});
```

### 5.4 Raster Tiles

For satellite imagery, elevation data, or any raster source, tile pyramids (XYZ/TMS) remain essential.

| Format | Description | Best For |
|--------|-------------|----------|
| XYZ PNG/JPEG | Traditional raster tiles | Basemap imagery |
| WebP tiles | 30% smaller than JPEG at similar quality | Modern basemaps |
| Cloud-Optimized GeoTIFF (COG) | Single file, range-request access | Analytical rasters |
| MBTiles | SQLite container for tiles | Offline / mobile |
| PMTiles (raster) | Single file, CDN-friendly | Serverless raster |

### 5.5 Dynamic vs Static Tiling

| Approach | Latency | Cost | Freshness | Best For |
|----------|---------|------|-----------|----------|
| Static (pre-generated) | ~10ms (CDN) | Low (storage only) | Stale until re-generated | Basemaps, archival data |
| Dynamic (on-demand) | ~50--200ms | Higher (compute) | Real-time | PostGIS data, live queries |
| Hybrid (static + dynamic overlay) | Mixed | Medium | Basemap stale, overlay fresh | Most production systems |

**Martin (Rust) for dynamic PostGIS tiles:**

```yaml
# martin-config.yaml
listen_addresses: '0.0.0.0:3000'
postgres:
  connection_string: 'postgresql://user:pass@localhost/geodb'
  auto_publish:
    tables:
      from: public
      # Martin auto-discovers PostGIS tables and serves them as MVT
    functions: true
      # Also serves PostGIS functions as tile endpoints

# Custom function-based tile source with filtering
# CREATE FUNCTION tiles.filtered_points(z integer, x integer, y integer, query_params json)
# RETURNS bytea AS $$
#   SELECT ST_AsMVT(mvtgeom, 'points', 4096, 'geom') FROM (
#     SELECT ST_AsMVTGeom(geom, ST_TileEnvelope(z, x, y), 4096, 256, true) AS geom,
#            category, value
#     FROM points
#     WHERE geom && ST_TileEnvelope(z, x, y)
#       AND category = (query_params->>'category')::text
#   ) mvtgeom;
# $$ LANGUAGE sql STABLE;
```

> **Cross-reference:** See [js-bindbox/tile-servers.md](../js-bindbox/tile-servers.md) for comprehensive tile server setup, including Martin, TiTiler, and pg_tileserv. See [data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) for satellite raster tile sources.

---

## 6. Spatial Indexing for Visualization

Spatial indexing structures are the invisible backbone of large-scale geospatial visualization. They determine how quickly you can answer "which features are visible in this viewport?" -- a query executed on every pan/zoom interaction.

### 6.1 H3 Hexagonal Grid

[H3](https://h3geo.org) (by Uber) is a hierarchical hexagonal grid system that partitions the globe into hexagons at 16 resolution levels. It is the de facto standard for hexagonal binning in geospatial analytics and visualization.

```javascript
import { latLngToCell, cellToBoundary, getResolution, gridDisk } from 'h3-js';

// Index points into H3 cells for aggregation
function aggregateToH3(points, resolution) {
  const cellCounts = new Map();

  for (const { lat, lng, weight } of points) {
    const cell = latLngToCell(lat, lng, resolution);
    const existing = cellCounts.get(cell) || { count: 0, totalWeight: 0 };
    existing.count++;
    existing.totalWeight += weight || 1;
    cellCounts.set(cell, existing);
  }

  // Convert to deck.gl-compatible polygon data
  const polygons = [];
  for (const [cell, { count, totalWeight }] of cellCounts) {
    const boundary = cellToBoundary(cell, true); // [lng, lat] format
    polygons.push({
      polygon: boundary,
      count,
      avgWeight: totalWeight / count,
      cell,
    });
  }
  return polygons;
}

// Adaptive resolution based on zoom level
function h3ResolutionForZoom(zoom) {
  // Empirically tuned: each H3 resolution level ~= 1.5 map zoom levels
  const resolutions = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11];
  return resolutions[Math.min(zoom, resolutions.length - 1)];
}

// Render with deck.gl H3HexagonLayer (optimized for H3 cells)
import { H3HexagonLayer } from '@deck.gl/geo-layers';

const h3Layer = new H3HexagonLayer({
  id: 'h3-hexagons',
  data: aggregateToH3(points, 7),
  getHexagon: d => d.cell,
  getFillColor: d => colorScale(d.count),
  getElevation: d => d.count,
  elevationScale: 20,
  extruded: true,
  pickable: true,
  filled: true,
  wireframe: false,
  // H3HexagonLayer is highly optimized: it generates hex geometry on the GPU
  // from just the cell index, avoiding polygon boundary computation on CPU
  highPrecision: false, // Set true only near poles where hex distortion is large
});
```

**H3 resolution reference:**

| Resolution | Avg Hex Area | Avg Edge Length | Typical Use |
|-----------|-------------|----------------|-------------|
| 0 | 4,357,449 km2 | 1,108 km | Continental |
| 3 | 12,393 km2 | 59 km | Country |
| 5 | 253 km2 | 8.5 km | Metro area |
| 7 | 5.16 km2 | 1.2 km | Neighborhood |
| 9 | 0.105 km2 | 174 m | Block |
| 11 | 0.002 km2 | 24 m | Building |
| 13 | 0.00004 km2 | 3.3 m | Sub-building |
| 15 | 0.0000009 km2 | 0.5 m | Precise |

### 6.2 S2 Cells

[S2 Geometry](https://s2geometry.io) (by Google) uses a Hilbert curve to map the sphere to a quad-tree of cells. S2 cells are squares (on the sphere) and have excellent containment and covering properties, making them ideal for spatial queries.

```javascript
// S2 cell-based spatial filtering with the s2-geometry library
import { S2RegionCoverer, S2LatLngRect, S2LatLng, S2CellId } from 's2-geometry';

// Compute S2 cells that cover the current viewport
function getViewportCovering(bounds, maxCells = 20) {
  const rect = S2LatLngRect.fromPointPair(
    S2LatLng.fromDegrees(bounds.south, bounds.west),
    S2LatLng.fromDegrees(bounds.north, bounds.east)
  );

  const coverer = new S2RegionCoverer();
  coverer.setMaxCells(maxCells);
  coverer.setMinLevel(1);
  coverer.setMaxLevel(20);

  const covering = coverer.getCovering(rect);
  return covering.map(cellId => cellId.toToken());
}

// Use S2 tokens to query pre-indexed data
// SQL: SELECT * FROM points WHERE s2_token BETWEEN :cell_range_min AND :cell_range_max
```

### 6.3 Client-Side Spatial Indices (Flatbush / kdbush)

For client-side spatial queries on tens of thousands to millions of features, [Flatbush](https://github.com/mourner/flatbush) (static R-tree) and [kdbush](https://github.com/mourner/kdbush) (static KD-tree for points) by Vladimir Agafonkin are the fastest options.

```javascript
import Flatbush from 'flatbush';

// Build a static spatial index for 2M bounding boxes
const numFeatures = 2_000_000;
const index = new Flatbush(numFeatures);

// Add all features (this is the build step, ~300ms for 2M items)
for (let i = 0; i < numFeatures; i++) {
  index.add(minXs[i], minYs[i], maxXs[i], maxYs[i]);
}
index.finish(); // Builds the packed R-tree

// Query: find all features in the current viewport
// This returns indices into the original array, ~0.1ms for a typical viewport
const visibleIndices = index.search(viewMinX, viewMinY, viewMaxX, viewMaxY);
console.log(`${visibleIndices.length} features in viewport (of ${numFeatures} total)`);

// Use visibleIndices to render only the features in view
// This is how FlatGeobuf, deck.gl's TileLayer, and many libraries cull data

// For point data, kdbush is even faster:
import KDBush from 'kdbush';

const pointIndex = new KDBush(numPoints);
for (let i = 0; i < numPoints; i++) {
  pointIndex.add(lons[i], lats[i]);
}
pointIndex.finish();

// Range query
const inRange = pointIndex.range(minLon, minLat, maxLon, maxLat);

// Radius query (within N units of a point)
const nearby = pointIndex.within(centerLon, centerLat, radius);
```

**Benchmark (spatial index build + query, M1 MacBook):**

| Library | Features | Build Time | Query Time (viewport) | Memory |
|---------|----------|-----------|----------------------|--------|
| Flatbush | 100K | 8ms | 0.02ms | 2.4MB |
| Flatbush | 1M | 85ms | 0.05ms | 24MB |
| Flatbush | 10M | 920ms | 0.1ms | 240MB |
| kdbush | 100K | 6ms | 0.01ms | 1.6MB |
| kdbush | 1M | 65ms | 0.03ms | 16MB |
| kdbush | 10M | 710ms | 0.08ms | 160MB |

### 6.4 Quadtree

Quadtrees recursively subdivide 2D space into four quadrants. They are the foundation of tile pyramids (each tile is a quadtree node) and are useful for adaptive level-of-detail.

```javascript
// Simple quadtree for viewport culling and LOD
class QuadTree {
  constructor(bounds, maxDepth = 20, maxPoints = 100) {
    this.bounds = bounds; // { x, y, width, height }
    this.maxDepth = maxDepth;
    this.maxPoints = maxPoints;
    this.points = [];
    this.children = null;
    this.depth = 0;
  }

  insert(point) {
    if (!this._contains(point)) return false;

    if (this.points.length < this.maxPoints || this.depth >= this.maxDepth) {
      this.points.push(point);
      return true;
    }

    if (!this.children) this._subdivide();

    for (const child of this.children) {
      if (child.insert(point)) return true;
    }
    return false;
  }

  // Query: return points visible in viewport, with LOD
  // At low zoom, return aggregate counts; at high zoom, return individual points
  queryViewport(viewport, targetCount = 50000) {
    const results = [];
    this._queryRecursive(viewport, targetCount, results);
    return results;
  }

  _queryRecursive(viewport, budget, results) {
    if (!this._intersects(viewport)) return;

    // LOD: if this node's points fit in the budget, return them directly
    if (this.points.length <= budget || !this.children) {
      for (const p of this.points) {
        if (this._pointInRect(p, viewport)) results.push(p);
      }
      return;
    }

    // Otherwise, recurse into children with divided budget
    const childBudget = Math.floor(budget / 4);
    for (const child of this.children) {
      child._queryRecursive(viewport, childBudget, results);
    }
  }

  _subdivide() {
    const { x, y, width, height } = this.bounds;
    const hw = width / 2, hh = height / 2;
    this.children = [
      new QuadTree({ x, y, width: hw, height: hh }, this.maxDepth, this.maxPoints),
      new QuadTree({ x: x + hw, y, width: hw, height: hh }, this.maxDepth, this.maxPoints),
      new QuadTree({ x, y: y + hh, width: hw, height: hh }, this.maxDepth, this.maxPoints),
      new QuadTree({ x: x + hw, y: y + hh, width: hw, height: hh }, this.maxDepth, this.maxPoints),
    ];
    this.children.forEach(c => { c.depth = this.depth + 1; });
    // Re-insert existing points into children
    for (const p of this.points) {
      for (const child of this.children) {
        if (child.insert(p)) break;
      }
    }
    this.points = [];
  }

  _contains(p) {
    return p.x >= this.bounds.x && p.x < this.bounds.x + this.bounds.width &&
           p.y >= this.bounds.y && p.y < this.bounds.y + this.bounds.height;
  }

  _intersects(rect) {
    return !(rect.x > this.bounds.x + this.bounds.width ||
             rect.x + rect.width < this.bounds.x ||
             rect.y > this.bounds.y + this.bounds.height ||
             rect.y + rect.height < this.bounds.y);
  }

  _pointInRect(p, rect) {
    return p.x >= rect.x && p.x <= rect.x + rect.width &&
           p.y >= rect.y && p.y <= rect.y + rect.height;
  }
}
```

> **Cross-reference:** See [js-bindbox/spatial-analysis.md](../js-bindbox/spatial-analysis.md) for spatial analysis algorithms that leverage these index structures.

---

## 7. Aggregation and Binning

When individual feature rendering is no longer feasible (typically above 1--5M points), aggregation transforms individual features into statistical summaries over spatial bins. This is the primary strategy for scaling to billions of features.

### 7.1 Hexagonal Binning (Hexbin)

Hexagons tile the plane with the best area-to-perimeter ratio (minimizing edge effects) and provide 6-connectivity (each neighbor shares an edge). H3 hexbins are preferred for geospatial data.

```javascript
// Server-side H3 hexbin aggregation with DuckDB
// This is far more efficient than client-side for > 10M points
const query = `
  INSTALL h3 FROM community;
  LOAD h3;

  SELECT
    h3_latlng_to_cell(lat, lng, 7) AS cell,
    COUNT(*) AS count,
    AVG(speed) AS avg_speed,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY speed) AS p95_speed,
    MIN(timestamp) AS earliest,
    MAX(timestamp) AS latest
  FROM trips
  WHERE timestamp BETWEEN '2025-01-01' AND '2025-01-31'
  GROUP BY cell
  HAVING count > 5
  ORDER BY count DESC
`;

// Client-side: d3-hexbin for smaller datasets
import { hexbin as d3Hexbin } from 'd3-hexbin';

const hexbin = d3Hexbin()
  .x(d => projection([d.lng, d.lat])[0])
  .y(d => projection([d.lng, d.lat])[1])
  .radius(10); // Pixel radius

const bins = hexbin(data);
// bins is an array of arrays, each with a .x, .y centroid
```

### 7.2 Grid Binning

Rectangular grid bins are simpler to compute and align with raster/tile grids.

```javascript
// Fast grid binning for 10M+ points using typed arrays
function gridBin(lons, lats, values, resolution = 0.01) {
  // resolution in degrees; 0.01 ~= 1km at equator
  const minLon = -180, maxLon = 180;
  const minLat = -90, maxLat = 90;
  const cols = Math.ceil((maxLon - minLon) / resolution);
  const rows = Math.ceil((maxLat - minLat) / resolution);

  // Use typed arrays for count and sum (no object creation)
  const counts = new Uint32Array(cols * rows);
  const sums = new Float64Array(cols * rows);

  for (let i = 0; i < lons.length; i++) {
    const col = Math.floor((lons[i] - minLon) / resolution);
    const row = Math.floor((lats[i] - minLat) / resolution);
    if (col >= 0 && col < cols && row >= 0 && row < rows) {
      const idx = row * cols + col;
      counts[idx]++;
      sums[idx] += values[i];
    }
  }

  return { counts, sums, cols, rows, resolution };
}

// Benchmarks: 10M points, 0.01deg resolution
// CPU (single-threaded JS): ~800ms
// CPU (Web Worker, 4 threads, merge): ~250ms
// GPU (WebGPU compute): ~15ms
```

### 7.3 Contour and Density Estimation

Kernel Density Estimation (KDE) produces smooth continuous surfaces from point data, rendered as filled contours or heatmaps.

```javascript
// d3-contour for isoband generation from KDE
import { contourDensity } from 'd3-contour';
import { geoPath, geoMercator } from 'd3-geo';

const projection = geoMercator().fitSize([width, height], geojsonBounds);

const contours = contourDensity()
  .x(d => projection([d.lng, d.lat])[0])
  .y(d => projection([d.lng, d.lat])[1])
  .weight(d => d.magnitude)
  .size([width, height])
  .bandwidth(20)                 // Kernel bandwidth in pixels
  .thresholds(15)                // Number of contour levels
  (earthquakeData);

// Render contours as SVG paths or convert to GeoJSON for deck.gl
const pathGenerator = geoPath();
contours.forEach((contour, i) => {
  svg.append('path')
    .attr('d', pathGenerator(contour))
    .attr('fill', colorScale(contour.value))
    .attr('opacity', 0.6);
});

// For larger datasets, use deck.gl's ContourLayer (GPU-accelerated)
import { ContourLayer } from '@deck.gl/aggregation-layers';

const contourLayer = new ContourLayer({
  id: 'contours',
  data: points,
  getPosition: d => [d.lng, d.lat],
  getWeight: d => d.value,
  contours: [
    { threshold: 1, color: [255, 255, 178], strokeWidth: 0 },
    { threshold: 5, color: [254, 204, 92], strokeWidth: 0 },
    { threshold: 10, color: [253, 141, 60], strokeWidth: 0 },
    { threshold: 20, color: [240, 59, 32], strokeWidth: 0 },
    { threshold: 50, color: [189, 0, 38], strokeWidth: 0 },
  ],
  cellSize: 200,                 // Aggregation cell size in meters
  gpuAggregation: true,
});
```

### 7.4 Server-Side vs Client-Side Aggregation

| Criteria | Server-Side | Client-Side |
|----------|------------|-------------|
| Data size | Unlimited | < 5--10M points (memory) |
| Latency | Network + compute | Instant after initial load |
| Interactivity | Re-query on every change | 60fps re-aggregation |
| Freshness | Real-time | Snapshot at load time |
| Complexity | Requires backend infra | Pure frontend |
| Best tools | DuckDB, PostGIS, BigQuery | deck.gl GPU layers, d3, H3-js |

**Recommended pattern for production:**

```
10B raw points (PostGIS/BigQuery)
    |
    v  [Server-side pre-aggregation to H3 res 5-9]
5M hex aggregates
    |
    v  [Stream as Arrow IPC or FlatGeobuf]
Client (deck.gl H3HexagonLayer)
    |
    v  [Client-side re-aggregation for zoom refinement]
Display
```

> **Cross-reference:** See [data-analysis/](../data-analysis/) for broader geospatial data analysis techniques.

---

## 8. Streaming and Progressive Rendering

For datasets too large to load in a single request, streaming enables progressive rendering: features appear incrementally as they arrive, starting with those nearest the viewport.

### 8.1 FlatGeobuf Spatial Streaming

[FlatGeobuf](https://flatgeobuf.org) is a binary geospatial format with a built-in spatial index (packed Hilbert R-tree) that enables HTTP range-request-based spatial queries. You can fetch only the features in a bounding box without downloading the entire file.

```javascript
import { deserialize } from 'flatgeobuf/lib/mjs/geojson.js';

// Stream features within a bounding box from a remote FlatGeobuf file
// Only the relevant portions of the file are downloaded (via HTTP range requests)
async function streamFeaturesInViewport(url, bounds) {
  const rect = {
    minX: bounds.west,
    minY: bounds.south,
    maxX: bounds.east,
    maxY: bounds.north,
  };

  const features = [];
  const iter = deserialize(url, rect);

  for await (const feature of iter) {
    features.push(feature);

    // Progressive rendering: update the map every 10,000 features
    if (features.length % 10000 === 0) {
      updateMapLayer(features);
    }
  }

  // Final render with all features
  updateMapLayer(features);
  console.log(`Loaded ${features.length} features in viewport`);
}

// Example: 100M-feature FlatGeobuf on S3 (2.5GB file)
// Viewport query: ~50K features returned
// Total bytes downloaded: ~5MB (of 2.5GB)
// Time: ~800ms (including network)

function updateMapLayer(features) {
  const geojson = { type: 'FeatureCollection', features };
  map.getSource('streaming-data')?.setData(geojson);
}

// Usage with map viewport events
map.on('moveend', () => {
  const bounds = map.getBounds();
  streamFeaturesInViewport(
    'https://cdn.example.com/data.fgb',
    {
      west: bounds.getWest(),
      south: bounds.getSouth(),
      east: bounds.getEast(),
      north: bounds.getNorth(),
    }
  );
});
```

### 8.2 Apache Arrow IPC Streaming

Apache Arrow's IPC streaming format enables columnar data streaming with zero-copy memory mapping. Combined with [arrow-js](https://github.com/apache/arrow/tree/main/js), this is the fastest way to stream tabular geospatial data to deck.gl.

```javascript
import { tableFromIPC } from 'apache-arrow';

// Stream Arrow IPC data and feed directly to deck.gl binary attributes
async function loadArrowData(url) {
  const response = await fetch(url);
  const buffer = await response.arrayBuffer();
  const table = tableFromIPC(buffer);

  // Extract columns as typed arrays (zero-copy if the buffer alignment matches)
  const lons = table.getChild('longitude').toArray();   // Float64Array
  const lats = table.getChild('latitude').toArray();     // Float64Array
  const values = table.getChild('value').toArray();      // Float32Array

  // Convert to interleaved positions for deck.gl
  const positions = new Float32Array(table.numRows * 3);
  for (let i = 0; i < table.numRows; i++) {
    positions[i * 3] = lons[i];
    positions[i * 3 + 1] = lats[i];
    positions[i * 3 + 2] = 0;
  }

  return {
    length: table.numRows,
    positions,
    values,
  };
}

// Arrow IPC streaming (record batch by record batch)
async function streamArrowBatches(url, onBatch) {
  const response = await fetch(url);
  const reader = response.body.getReader();

  // Use Arrow's streaming reader
  const { RecordBatchReader } = await import('apache-arrow');

  const arrowReader = await RecordBatchReader.from(
    readableStreamToAsyncIterator(reader)
  );

  let totalRows = 0;
  for await (const batch of arrowReader) {
    totalRows += batch.numRows;
    onBatch(batch, totalRows);
  }
}

// Helper: convert ReadableStream to async iterator
async function* readableStreamToAsyncIterator(reader) {
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    yield value;
  }
}
```

### 8.3 Progressive Loading Patterns

```javascript
// Pattern: Level-of-Detail progressive loading
// 1. Load a low-resolution overview immediately
// 2. Progressively refine as the user zooms in

class ProgressiveLoader {
  constructor(deck, baseUrl) {
    this.deck = deck;
    this.baseUrl = baseUrl;
    this.lodLevels = [
      { zoom: [0, 6], resolution: 'low', file: 'data-100k.arrow' },
      { zoom: [6, 10], resolution: 'medium', file: 'data-1m.arrow' },
      { zoom: [10, 14], resolution: 'high', file: 'data-10m.fgb' },  // spatial query
      { zoom: [14, 22], resolution: 'full', file: 'data-full.fgb' },  // spatial query
    ];
    this.cache = new Map();
  }

  async onViewStateChange(viewState) {
    const zoom = viewState.zoom;
    const lod = this.lodLevels.find(l => zoom >= l.zoom[0] && zoom < l.zoom[1]);
    if (!lod) return;

    if (lod.resolution === 'high' || lod.resolution === 'full') {
      // Use spatial streaming for detail levels
      await this.loadSpatial(lod.file, viewState);
    } else {
      // Use pre-aggregated overview
      await this.loadOverview(lod.file);
    }
  }

  async loadOverview(file) {
    if (this.cache.has(file)) return;
    const data = await loadArrowData(`${this.baseUrl}/${file}`);
    this.cache.set(file, data);
    this.updateLayers(data);
  }

  async loadSpatial(file, viewState) {
    // Only load features in the current viewport
    const bounds = this.viewStateToBounds(viewState);
    // Debounce to avoid excessive requests during pan
    clearTimeout(this._debounce);
    this._debounce = setTimeout(async () => {
      await streamFeaturesInViewport(`${this.baseUrl}/${file}`, bounds);
    }, 200);
  }

  viewStateToBounds(vs) {
    // Approximate bounds from center + zoom
    const span = 360 / Math.pow(2, vs.zoom);
    return {
      west: vs.longitude - span / 2,
      east: vs.longitude + span / 2,
      south: vs.latitude - span / 3,
      north: vs.latitude + span / 3,
    };
  }
}
```

> **Cross-reference:** See [js-bindbox/data-formats-loading.md](../js-bindbox/data-formats-loading.md) for comprehensive coverage of geospatial data formats and loading strategies.

---

## 9. DuckDB WASM for Visualization

[DuckDB-WASM](https://duckdb.org/docs/api/wasm) brings a full analytical SQL database to the browser, enabling SQL-powered geospatial analytics directly on the client. Combined with the [spatial extension](https://duckdb.org/docs/extensions/spatial.html) and visualization frameworks like [Mosaic](https://uwdata.github.io/mosaic) or [Observable Framework](https://observablehq.com/framework), it creates a powerful pattern for interactive geospatial dashboards.

### 9.1 Setup and Spatial Extension

```javascript
import * as duckdb from '@duckdb/duckdb-wasm';

// Initialize DuckDB-WASM with spatial extension
async function initDuckDB() {
  const JSDELIVR_BUNDLES = duckdb.getJsDelivrBundles();
  const bundle = await duckdb.selectBundle(JSDELIVR_BUNDLES);

  const worker = new Worker(bundle.mainWorker);
  const logger = new duckdb.ConsoleLogger();
  const db = new duckdb.AsyncDuckDB(logger, worker);
  await db.instantiate(bundle.mainModule, bundle.pthreadWorker);

  const conn = await db.connect();

  // Install and load spatial extension
  await conn.query(`INSTALL spatial; LOAD spatial;`);
  await conn.query(`INSTALL h3 FROM community; LOAD h3;`);

  return { db, conn };
}
```

### 9.2 In-Browser SQL Analytics for Visualization

```javascript
// Load a Parquet file directly from a URL (DuckDB streams it via HTTP range requests)
await conn.query(`
  CREATE TABLE trips AS
  SELECT * FROM read_parquet('https://data.example.com/nyc-taxi-2024.parquet');
`);

// Spatial aggregation: H3 hexbin with DuckDB
const hexbinResult = await conn.query(`
  SELECT
    h3_latlng_to_cell(pickup_latitude, pickup_longitude, 8) AS h3_cell,
    COUNT(*) AS trip_count,
    AVG(fare_amount) AS avg_fare,
    AVG(trip_distance) AS avg_distance,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY tip_amount) AS median_tip
  FROM trips
  WHERE pickup_latitude BETWEEN 40.5 AND 41.0
    AND pickup_longitude BETWEEN -74.3 AND -73.7
  GROUP BY h3_cell
  HAVING trip_count > 10
  ORDER BY trip_count DESC
`);

// Convert DuckDB result to deck.gl-compatible data
const rows = hexbinResult.toArray();
const h3Data = rows.map(row => ({
  cell: row.h3_cell,
  count: Number(row.trip_count),
  avgFare: Number(row.avg_fare),
}));

// Render with deck.gl H3HexagonLayer
const layer = new H3HexagonLayer({
  id: 'duckdb-hexbins',
  data: h3Data,
  getHexagon: d => d.cell,
  getFillColor: d => colorScale(d.count),
  getElevation: d => d.count,
  extruded: true,
});
```

### 9.3 DuckDB + Mosaic for Interactive Dashboards

[Mosaic](https://uwdata.github.io/mosaic) is a framework that connects DuckDB-WASM to interactive visualizations with cross-filtering.

```javascript
// Mosaic + vgplot: interactive geospatial dashboard
// This creates linked views where filtering one chart updates all others

import { coordinator, wasmConnector } from '@uwdata/mosaic-core';
import { plot, from, hexbin, colorScale } from '@uwdata/vgplot';

// Connect Mosaic to DuckDB-WASM
const wasm = await wasmConnector({ log: false });
await coordinator().databaseConnector(wasm);

// Load data
await wasm.query(`
  CREATE TABLE earthquakes AS
  SELECT * FROM read_csv_auto('https://earthquake.usgs.gov/data/comcat/latest.csv');
`);

// Create a linked dashboard with cross-filtering
const mapView = plot(
  hexbin(
    from('earthquakes'),
    {
      x: 'longitude',
      y: 'latitude',
      fill: 'count',
      binWidth: 1,
    }
  ),
  {
    projection: 'mercator',
    colorScale: 'log',
    width: 800,
    height: 500,
  }
);

const magnitudeHist = plot(
  from('earthquakes', { filterBy: mapView }),
  {
    mark: 'rectY',
    x: { field: 'mag', bin: true },
    y: 'count',
    fill: 'steelblue',
    width: 400,
    height: 200,
  }
);

// Brushing on the map filters the histogram, and vice versa
document.getElementById('dashboard').append(mapView, magnitudeHist);
```

### 9.4 Performance Characteristics

**DuckDB-WASM benchmarks (M1 MacBook, Chrome):**

| Operation | 1M rows | 10M rows | 100M rows |
|-----------|---------|----------|-----------|
| Load Parquet (HTTP) | 0.8s | 4.2s | 38s |
| COUNT(*) | 2ms | 15ms | 120ms |
| GROUP BY (H3 hex) | 180ms | 1.2s | 11s |
| Spatial join (ST_Contains) | 250ms | 2.5s | 22s |
| Window function | 90ms | 800ms | 7.5s |

> **Cross-reference:** See [data-sources/](../data-sources/) for geospatial data sources that work well with DuckDB Parquet ingestion.

---

## 10. Point Cloud Rendering

Point cloud data (LiDAR, photogrammetry, SfM) routinely contains billions of points. Specialized rendering techniques are required: octree-based level-of-detail, frustum culling, and GPU-optimized point splatting.

### 10.1 Potree

[Potree](https://potree.github.io) is the gold standard open-source web viewer for massive point clouds. It uses an octree-based LOD hierarchy and progressive loading to render billions of points interactively.

```javascript
// Potree viewer setup for a billion-point LiDAR scan
import * as Potree from 'potree';

const viewer = new Potree.Viewer(document.getElementById('potree-container'));
viewer.setEDLEnabled(true);          // Eye-Dome Lighting for depth perception
viewer.setFOV(60);
viewer.setPointBudget(5_000_000);    // Max points rendered per frame
viewer.setMinNodeSize(30);           // Min pixel size before loading children

// Load the octree
Potree.loadPointCloud(
  'https://cdn.example.com/pointcloud/metadata.json',
  'my-cloud',
  (e) => {
    const cloud = e.pointcloud;
    viewer.scene.addPointCloud(cloud);

    // Configure material
    cloud.material.size = 1;
    cloud.material.pointSizeType = Potree.PointSizeType.ADAPTIVE;
    cloud.material.activeAttributeName = 'classification';

    // Color by classification (LAS standard)
    cloud.material.intensityRange = [0, 65535];

    // Fit view to point cloud
    viewer.fitToScreen();
  }
);

// Potree handles:
// - Octree traversal (loads ~5M points from 1B+ total)
// - Frustum culling (only visible octants)
// - Screen-space error metric (more detail where you look)
// - Progressive loading (coarse first, then refine)
// - GPU point splatting (anti-aliased circles)
```

**Point budget guidelines:**

| GPU Tier | Max Point Budget | Typical FPS |
|----------|-----------------|-------------|
| Integrated (Intel/AMD APU) | 1--3M | 30--60 |
| Mid-range (RTX 3060, M1) | 3--8M | 60 |
| High-end (RTX 4090, M3 Max) | 8--20M | 60 |

### 10.2 CesiumJS with 3D Tiles Point Clouds

```javascript
import { Viewer, Cesium3DTileset, Cesium3DTileStyle } from 'cesium';

const viewer = new Viewer('cesiumContainer', {
  terrain: false,
  baseLayerPicker: false,
});

const tileset = await Cesium3DTileset.fromUrl(
  'https://assets.cesium.com/YOUR_ASSET/tileset.json',
  {
    maximumScreenSpaceError: 8,       // Lower = more detail
    maximumMemoryUsage: 2048,         // MB, increase for high-end GPUs
    preloadWhenHidden: true,
    preferLeaves: true,               // Prefer leaf tiles for detail
    dynamicScreenSpaceError: true,    // Adjust SSE based on camera velocity
    dynamicScreenSpaceErrorDensity: 0.00278,
    dynamicScreenSpaceErrorFactor: 4.0,
  }
);

viewer.scene.primitives.add(tileset);

// Style: color by elevation, size by intensity
tileset.style = new Cesium3DTileStyle({
  color: {
    conditions: [
      ['${elevation} > 500', 'color("red")'],
      ['${elevation} > 200', 'color("orange")'],
      ['${elevation} > 50', 'color("yellow")'],
      ['true', 'color("green")'],
    ],
  },
  pointSize: '${intensity} / 100.0 + 2.0',
});
```

### 10.3 deck.gl PointCloudLayer

```javascript
import { PointCloudLayer } from '@deck.gl/layers';

// For moderate point clouds (up to ~20M points in browser memory)
const pointCloudLayer = new PointCloudLayer({
  id: 'point-cloud',
  data: '/data/lidar-scan.arrow',
  loaders: [ArrowLoader],
  // Binary attributes for maximum performance
  getPosition: { value: positionBuffer, size: 3 },   // Float32Array
  getColor: { value: colorBuffer, size: 3 },          // Uint8Array
  getNormal: { value: normalBuffer, size: 3 },        // Float32Array
  pointSize: 2,
  coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
  coordinateOrigin: [-122.4, 37.8, 0],  // Local origin for float precision
  material: {
    ambient: 0.2,
    diffuse: 0.8,
    shininess: 32,
  },
  // sizeUnits: 'meters' for real-world sizing
  sizeUnits: 'pixels',
  opacity: 1.0,
});
```

### 10.4 Octree LOD Architecture

```
Level 0 (root):     1 node, ~10K points (entire cloud subsampled)
Level 1:            8 nodes, ~80K points total
Level 2:            64 nodes, ~640K points total
Level 3:            512 nodes, ~5M points total
...
Level 10:           ~1B nodes, ~10B points total (leaf level)

At any time, only ~5M points are loaded (the "point budget").
As the camera moves, nodes are loaded/unloaded to maintain the budget.

Screen-Space Error (SSE) determines which nodes to load:
  SSE = (node bounding box size in pixels) / (node point spacing in pixels)
  If SSE > threshold (e.g., 16), subdivide and load children.
  If SSE < threshold, this node is sufficient detail.
```

> **Cross-reference:** See [visualization/3d-visualization.md](./3d-visualization.md) for broader 3D visualization techniques including terrain, buildings, and photogrammetry.

---

## 11. Raster Visualization

Raster data -- satellite imagery, elevation models, climate grids, land use classifications -- often exceeds gigabytes or terabytes. Cloud-Optimized GeoTIFF (COG) and dynamic rendering pipelines enable interactive visualization without downloading entire files.

### 11.1 Cloud-Optimized GeoTIFF (COG) Rendering

A COG is a regular GeoTIFF file organized with internal tiling and overviews so that HTTP range requests can fetch only the portion needed for the current viewport and zoom level.

```javascript
import GeoTIFF from 'geotiff';
import { BitmapLayer } from '@deck.gl/layers';

// Read a COG and render a viewport-specific tile
async function renderCOGViewport(url, viewport) {
  const tiff = await GeoTIFF.fromUrl(url, {
    allowFullFile: false,   // Force range requests
    cacheSize: 100,         // Cache up to 100 tiles internally
  });

  // Get the image (overview) appropriate for the current zoom
  const image = await tiff.getImage(0); // Full resolution
  const overviewLevel = selectOverview(image, viewport.zoom);
  const overviewImage = await tiff.getImage(overviewLevel);

  // Read only the pixels covering the viewport
  const [west, south, east, north] = viewport.bounds;
  const data = await overviewImage.readRasters({
    bbox: [west, south, east, north],
    width: 512,   // Output resolution (pixels)
    height: 512,
    interleave: true,
    pool: workerPool,  // Use Web Workers for decoding
  });

  // Apply color mapping (e.g., NDVI: red-yellow-green)
  const rgba = applyColorMap(data, {
    min: -0.2,
    max: 0.9,
    colorMap: 'RdYlGn',
    noData: -9999,
  });

  // Render as a deck.gl BitmapLayer
  return new BitmapLayer({
    id: `cog-${west}-${south}`,
    bounds: [west, south, east, north],
    image: {
      data: rgba,
      width: 512,
      height: 512,
    },
    opacity: 0.8,
    textureParameters: {
      minFilter: 'linear',
      magFilter: 'linear',
    },
  });
}

// Select the best overview level for the current zoom
function selectOverview(image, zoom) {
  const fullRes = image.getResolution();
  const targetRes = 360 / (256 * Math.pow(2, zoom)); // degrees per pixel
  const overviews = image.fileDirectory.SubIFDs || [];

  for (let i = 0; i < overviews.length; i++) {
    const ovRes = fullRes[0] * Math.pow(2, i + 1);
    if (ovRes <= targetRes) return i + 1;
  }
  return 0; // Full resolution
}

// Color mapping function
function applyColorMap(rasterData, options) {
  const { min, max, noData } = options;
  const range = max - min;
  const rgba = new Uint8Array(rasterData.length * 4 / rasterData.length * 4);
  const numPixels = rasterData.width * rasterData.height;

  const output = new Uint8ClampedArray(numPixels * 4);
  for (let i = 0; i < numPixels; i++) {
    const value = rasterData[i];
    if (value === noData || isNaN(value)) {
      output[i * 4 + 3] = 0; // Transparent
      continue;
    }
    const normalized = Math.max(0, Math.min(1, (value - min) / range));
    const [r, g, b] = interpolateRdYlGn(normalized);
    output[i * 4] = r;
    output[i * 4 + 1] = g;
    output[i * 4 + 2] = b;
    output[i * 4 + 3] = 255;
  }
  return { data: output, width: rasterData.width, height: rasterData.height };
}
```

### 11.2 TiTiler for Dynamic Raster Tiles

[TiTiler](https://developmentseed.org/titiler/) is a FastAPI-based dynamic tile server for COGs.

```javascript
// TiTiler serves COGs as raster tiles dynamically
// No pre-generation needed -- tiles are computed on-the-fly from the COG

const tiTilerBase = 'https://titiler.example.com';
const cogUrl = 'https://s3.amazonaws.com/bucket/elevation.tif';

// Use as a raster tile source in MapLibre
map.addSource('dem', {
  type: 'raster',
  tiles: [
    `${tiTilerBase}/cog/tiles/{z}/{x}/{y}.png?` +
    `url=${encodeURIComponent(cogUrl)}` +
    `&rescale=0,4000` +
    `&colormap_name=terrain` +
    `&return_mask=true`
  ],
  tileSize: 256,
  attribution: 'Copernicus DEM',
});

map.addLayer({
  id: 'dem-layer',
  type: 'raster',
  source: 'dem',
  paint: { 'raster-opacity': 0.7 },
});

// Multi-band composite (e.g., Sentinel-2 true color)
const sentinel2Url = 'https://s3.amazonaws.com/bucket/sentinel2.tif';
const trueColorTiles = `${tiTilerBase}/cog/tiles/{z}/{x}/{y}.png?` +
  `url=${encodeURIComponent(sentinel2Url)}` +
  `&bidx=4&bidx=3&bidx=2` +  // Bands: Red, Green, Blue
  `&rescale=0,3000&rescale=0,3000&rescale=0,3000`;
```

### 11.3 Multi-Band Composites and Band Math

```javascript
// Client-side band math for NDVI from a multi-band COG
async function computeNDVI(cogUrl, viewport) {
  const tiff = await GeoTIFF.fromUrl(cogUrl);
  const image = await tiff.getImage(0);

  const [west, south, east, north] = viewport.bounds;

  // Read NIR (band 5) and Red (band 4) simultaneously
  const [nir, red] = await Promise.all([
    image.readRasters({
      bbox: [west, south, east, north],
      width: 512,
      height: 512,
      samples: [4],  // NIR band (0-indexed)
    }),
    image.readRasters({
      bbox: [west, south, east, north],
      width: 512,
      height: 512,
      samples: [3],  // Red band (0-indexed)
    }),
  ]);

  // Compute NDVI = (NIR - Red) / (NIR + Red)
  const nirData = nir[0];
  const redData = red[0];
  const ndvi = new Float32Array(nirData.length);

  for (let i = 0; i < nirData.length; i++) {
    const sum = nirData[i] + redData[i];
    ndvi[i] = sum === 0 ? 0 : (nirData[i] - redData[i]) / sum;
  }

  return ndvi;
}

// For GPU-accelerated band math, use a WebGL shader:
const bandMathShader = `
  precision highp float;
  uniform sampler2D u_nirBand;
  uniform sampler2D u_redBand;
  varying vec2 v_texCoord;

  vec3 ndviColorRamp(float ndvi) {
    // Brown -> Yellow -> Green color ramp
    if (ndvi < 0.0) return vec3(0.5, 0.3, 0.1);
    if (ndvi < 0.2) return mix(vec3(0.8, 0.7, 0.4), vec3(0.9, 0.9, 0.2), ndvi / 0.2);
    if (ndvi < 0.5) return mix(vec3(0.9, 0.9, 0.2), vec3(0.2, 0.8, 0.2), (ndvi - 0.2) / 0.3);
    return mix(vec3(0.2, 0.8, 0.2), vec3(0.0, 0.4, 0.0), (ndvi - 0.5) / 0.5);
  }

  void main() {
    float nir = texture2D(u_nirBand, v_texCoord).r;
    float red = texture2D(u_redBand, v_texCoord).r;
    float ndvi = (nir + red) == 0.0 ? 0.0 : (nir - red) / (nir + red);
    gl_FragColor = vec4(ndviColorRamp(ndvi), 1.0);
  }
`;
```

> **Cross-reference:** See [data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) for satellite imagery sources and access patterns. See [data-sources/elevation-terrain.md](../data-sources/elevation-terrain.md) for DEM data.

---

## 12. Real-Time Data at Scale

Real-time geospatial visualization connects live data streams -- vehicle positions, sensor readings, AIS ship tracks, weather updates -- to interactive maps with sub-second latency.

### 12.1 WebSocket Streaming

```javascript
// WebSocket client for real-time vehicle positions
class RealtimeVehicleLayer {
  constructor(deckInstance) {
    this.deck = deckInstance;
    this.vehicles = new Map(); // vehicleId -> {lon, lat, heading, speed, timestamp}
    this.positions = null;     // Float32Array for deck.gl binary attributes
    this.colors = null;        // Uint8Array
    this.dirty = false;
  }

  connect(wsUrl) {
    this.ws = new WebSocket(wsUrl);
    this.ws.binaryType = 'arraybuffer'; // Receive binary for efficiency

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        this.handleBinaryUpdate(event.data);
      } else {
        this.handleJsonUpdate(JSON.parse(event.data));
      }
    };

    // Render loop: update deck.gl at most 30fps regardless of message rate
    this.renderInterval = setInterval(() => this.render(), 33);
  }

  handleBinaryUpdate(buffer) {
    // Binary protocol: [vehicleId(u32), lon(f32), lat(f32), heading(f16), speed(f16)]
    // 14 bytes per vehicle update
    const view = new DataView(buffer);
    const numUpdates = buffer.byteLength / 14;

    for (let i = 0; i < numUpdates; i++) {
      const offset = i * 14;
      const id = view.getUint32(offset, true);
      const lon = view.getFloat32(offset + 4, true);
      const lat = view.getFloat32(offset + 8, true);
      const heading = view.getUint16(offset + 12, true) / 100; // 0.01 degree precision

      this.vehicles.set(id, { lon, lat, heading, timestamp: Date.now() });
    }
    this.dirty = true;
  }

  handleJsonUpdate(updates) {
    for (const u of updates) {
      this.vehicles.set(u.id, {
        lon: u.lon,
        lat: u.lat,
        heading: u.heading,
        speed: u.speed,
        timestamp: Date.now(),
      });
    }
    this.dirty = true;
  }

  render() {
    if (!this.dirty) return;
    this.dirty = false;

    const count = this.vehicles.size;
    // Reuse buffers if possible (avoid GC)
    if (!this.positions || this.positions.length < count * 3) {
      this.positions = new Float32Array(count * 3);
      this.colors = new Uint8Array(count * 4);
      this.angles = new Float32Array(count);
    }

    let i = 0;
    const now = Date.now();
    for (const [, v] of this.vehicles) {
      this.positions[i * 3] = v.lon;
      this.positions[i * 3 + 1] = v.lat;
      this.positions[i * 3 + 2] = 0;

      // Color by staleness: green (fresh) -> red (stale)
      const age = (now - v.timestamp) / 60000; // minutes
      const freshness = Math.max(0, 1 - age / 5);
      this.colors[i * 4] = Math.floor(255 * (1 - freshness));
      this.colors[i * 4 + 1] = Math.floor(255 * freshness);
      this.colors[i * 4 + 2] = 0;
      this.colors[i * 4 + 3] = 200;

      this.angles[i] = v.heading;
      i++;
    }

    this.deck.setProps({
      layers: [
        new ScatterplotLayer({
          id: 'vehicles',
          data: { length: count },
          getPosition: { value: this.positions, size: 3 },
          getFillColor: { value: this.colors, size: 4 },
          getRadius: 50,
          radiusMinPixels: 3,
          radiusMaxPixels: 15,
          // Important: use updateTriggers to force re-render when data changes
          updateTriggers: {
            getPosition: now,
            getFillColor: now,
          },
        }),
      ],
    });
  }

  disconnect() {
    clearInterval(this.renderInterval);
    this.ws?.close();
  }
}

// Usage
const rtLayer = new RealtimeVehicleLayer(deck);
rtLayer.connect('wss://stream.example.com/vehicles');
```

### 12.2 MQTT for IoT Sensor Networks

```javascript
import mqtt from 'mqtt';

// MQTT client for IoT sensor data (e.g., air quality monitors)
const client = mqtt.connect('wss://broker.example.com:8884/mqtt', {
  clientId: `viz-${Date.now()}`,
  clean: true,
});

const sensors = new Map();

client.on('connect', () => {
  // Subscribe to all sensor topics with a wildcard
  client.subscribe('sensors/+/telemetry', { qos: 0 });
  // QoS 0 = at most once (fastest, acceptable for viz)
});

client.on('message', (topic, message) => {
  const sensorId = topic.split('/')[1];
  const data = JSON.parse(message.toString());

  sensors.set(sensorId, {
    lon: data.lon,
    lat: data.lat,
    pm25: data.pm25,
    temperature: data.temp,
    humidity: data.humidity,
    timestamp: data.ts,
  });

  // Throttled rendering (see WebSocket example above)
});
```

### 12.3 Apache Kafka to Browser Pipeline

For enterprise-scale real-time pipelines, Apache Kafka is the standard message broker. The pattern for connecting Kafka to browser visualization:

```
Kafka Topic (e.g., vehicle-positions)
    |
    v
Kafka Consumer (Node.js / Python)
    |
    v  [Filter, aggregate, downsample]
WebSocket Server (e.g., Socket.IO, ws)
    |
    v  [Browser connects via WSS]
Browser (deck.gl real-time layer)
```

```javascript
// Server-side: Kafka consumer -> WebSocket bridge (Node.js)
import { Kafka } from 'kafkajs';
import { WebSocketServer } from 'ws';

const kafka = new Kafka({ brokers: ['kafka:9092'] });
const consumer = kafka.consumer({ groupId: 'viz-bridge' });

const wss = new WebSocketServer({ port: 8080 });
const clients = new Set();

wss.on('connection', ws => {
  clients.add(ws);
  ws.on('close', () => clients.delete(ws));
});

await consumer.connect();
await consumer.subscribe({ topic: 'vehicle-positions', fromBeginning: false });

// Batch and throttle messages to avoid overwhelming browser clients
let buffer = [];
const FLUSH_INTERVAL = 100; // ms
const MAX_BATCH = 1000;

setInterval(() => {
  if (buffer.length === 0) return;
  const batch = buffer.splice(0, MAX_BATCH);
  const payload = JSON.stringify(batch);
  for (const ws of clients) {
    if (ws.readyState === 1) ws.send(payload);
  }
}, FLUSH_INTERVAL);

await consumer.run({
  eachMessage: async ({ message }) => {
    const value = JSON.parse(message.value.toString());
    buffer.push({
      id: value.vehicle_id,
      lon: value.longitude,
      lat: value.latitude,
      heading: value.heading,
      speed: value.speed,
    });
  },
});
```

### 12.4 Live Dashboard Patterns

```javascript
// Pattern: Time-windowed live dashboard
// Shows last N minutes of data with automatic expiry

class TimeWindowedLayer {
  constructor(windowMinutes = 30) {
    this.windowMs = windowMinutes * 60 * 1000;
    this.events = [];         // Sorted by timestamp
    this.cleanupInterval = setInterval(() => this.cleanup(), 10000);
  }

  addEvent(event) {
    this.events.push({ ...event, _ts: Date.now() });
    // Binary insert to keep sorted (for efficient cleanup)
  }

  cleanup() {
    const cutoff = Date.now() - this.windowMs;
    // Remove old events from the front (they are sorted by time)
    let removeCount = 0;
    while (removeCount < this.events.length && this.events[removeCount]._ts < cutoff) {
      removeCount++;
    }
    if (removeCount > 0) {
      this.events.splice(0, removeCount);
    }
  }

  getLayer() {
    const now = Date.now();
    return new ScatterplotLayer({
      id: 'time-windowed',
      data: this.events,
      getPosition: d => [d.lon, d.lat],
      getFillColor: d => {
        // Fade from bright to dim based on age
        const age = (now - d._ts) / this.windowMs;
        const alpha = Math.floor(255 * (1 - age));
        return [255, 100, 50, alpha];
      },
      getRadius: 100,
      radiusMinPixels: 2,
      updateTriggers: {
        getFillColor: Math.floor(now / 1000), // Update every second
      },
    });
  }
}
```

> **Cross-reference:** See [js-bindbox/realtime-offline-advanced.md](../js-bindbox/realtime-offline-advanced.md) for real-time and offline patterns in web GIS applications.

---

## 13. Server-Side Rendering

Not all visualization needs to be interactive. Server-side rendering (SSR) generates static map images for reports, thumbnails, email, social media previews, and PDF exports.

### 13.1 MapLibre Native (maplibre-rs)

[maplibre-rs](https://github.com/maplibre/maplibre-rs) is a Rust implementation of the MapLibre renderer, suitable for headless rendering on servers.

```rust
// Rust: headless map rendering with maplibre-rs
// Generates a PNG image of a styled vector tile map

use maplibre::headless::HeadlessMap;
use maplibre::style::Style;

async fn render_map_image(
    center: (f64, f64),
    zoom: f32,
    width: u32,
    height: u32,
    style_url: &str,
) -> Vec<u8> {
    let style = Style::from_url(style_url).await.unwrap();

    let map = HeadlessMap::new(width, height, style);
    map.set_center(center.0, center.1);
    map.set_zoom(zoom);

    // Render and export as PNG
    let png_bytes = map.render_to_png().await.unwrap();
    png_bytes
}
```

### 13.2 Node.js Headless Rendering with sharp + canvas

```javascript
// Generate a map thumbnail with deck.gl's headless rendering
import { Deck } from '@deck.gl/core';
import { ScatterplotLayer } from '@deck.gl/layers';
import * as gl from 'gl';  // headless-gl
import sharp from 'sharp';

async function renderMapThumbnail(data, viewState, width = 800, height = 600) {
  // Create a headless WebGL context
  const glContext = gl(width, height, { preserveDrawingBuffer: true });

  const deck = new Deck({
    gl: glContext,
    width,
    height,
    initialViewState: viewState,
    layers: [
      new ScatterplotLayer({
        id: 'points',
        data,
        getPosition: d => [d.lon, d.lat],
        getFillColor: [255, 100, 50],
        getRadius: 100,
      }),
    ],
  });

  // Wait for rendering to complete
  await new Promise(resolve => {
    deck.setProps({
      onAfterRender: () => resolve(),
    });
  });

  // Read pixels from the framebuffer
  const pixels = new Uint8Array(width * height * 4);
  glContext.readPixels(0, 0, width, height, glContext.RGBA, glContext.UNSIGNED_BYTE, pixels);

  // Flip vertically (WebGL origin is bottom-left)
  const flipped = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const srcRow = (height - y - 1) * width * 4;
    const dstRow = y * width * 4;
    flipped.set(pixels.subarray(srcRow, srcRow + width * 4), dstRow);
  }

  // Convert to PNG with sharp
  const png = await sharp(Buffer.from(flipped), {
    raw: { width, height, channels: 4 },
  })
    .png({ quality: 90 })
    .toBuffer();

  deck.finalize();
  return png;
}
```

### 13.3 PDF Map Export

```javascript
// Client-side PDF export with jsPDF + map canvas
import { jsPDF } from 'jspdf';

function exportMapToPDF(map, title, metadata) {
  return new Promise((resolve) => {
    // Force a high-DPI render
    const dpi = 300;
    const canvas = map.getCanvas();
    const originalPixelRatio = window.devicePixelRatio;

    // Temporarily set high DPI for export
    Object.defineProperty(window, 'devicePixelRatio', { value: dpi / 96 });
    map.resize();

    map.once('render', () => {
      const imgData = canvas.toDataURL('image/png');

      const pdf = new jsPDF({
        orientation: 'landscape',
        unit: 'mm',
        format: 'a4',
      });

      // A4 landscape: 297mm x 210mm
      const mapWidth = 277;   // 10mm margins
      const mapHeight = 170;
      const mapTop = 25;

      // Title
      pdf.setFontSize(16);
      pdf.text(title, 148.5, 15, { align: 'center' });

      // Map image
      pdf.addImage(imgData, 'PNG', 10, mapTop, mapWidth, mapHeight);

      // Border around map
      pdf.setDrawColor(100);
      pdf.rect(10, mapTop, mapWidth, mapHeight);

      // Legend / metadata
      pdf.setFontSize(8);
      pdf.text(`Generated: ${new Date().toISOString()}`, 10, 200);
      pdf.text(`Center: ${map.getCenter().toArray().map(c => c.toFixed(4)).join(', ')}`, 10, 204);
      pdf.text(`Zoom: ${map.getZoom().toFixed(1)}`, 100, 204);

      // Scale bar
      if (metadata?.scaleBar) {
        // ... draw scale bar
      }

      // Restore DPI
      Object.defineProperty(window, 'devicePixelRatio', { value: originalPixelRatio });
      map.resize();

      resolve(pdf.output('blob'));
    });

    map.triggerRepaint();
  });
}
```

> **Cross-reference:** See [tools/server-publishing.md](../tools/server-publishing.md) for map server and publishing infrastructure.

---

## 14. Memory Management

Memory is the limiting factor for client-side large-scale visualization. A billion Float32 values consumes 4GB of RAM. Careful memory management is essential.

### 14.1 ArrayBuffer Pooling

```javascript
// ArrayBuffer pool to avoid GC pressure from frequent allocations
class BufferPool {
  constructor() {
    this.pools = new Map(); // size -> [ArrayBuffer, ...]
    this.maxPoolSize = 20;  // Max buffers to keep per size class
    this.totalAllocated = 0;
    this.totalReused = 0;
  }

  acquire(byteLength) {
    // Round up to nearest power of 2 to improve reuse
    const size = this._nextPowerOf2(byteLength);
    const pool = this.pools.get(size);

    if (pool && pool.length > 0) {
      this.totalReused++;
      return pool.pop();
    }

    this.totalAllocated++;
    return new ArrayBuffer(size);
  }

  release(buffer) {
    const size = buffer.byteLength;
    if (!this.pools.has(size)) {
      this.pools.set(size, []);
    }
    const pool = this.pools.get(size);
    if (pool.length < this.maxPoolSize) {
      pool.push(buffer);
    }
    // If pool is full, let the buffer be GC'd
  }

  // Typed array wrappers
  acquireFloat32Array(length) {
    const buffer = this.acquire(length * 4);
    return new Float32Array(buffer, 0, length);
  }

  acquireUint8Array(length) {
    const buffer = this.acquire(length);
    return new Uint8Array(buffer, 0, length);
  }

  releaseTypedArray(typedArray) {
    this.release(typedArray.buffer);
  }

  _nextPowerOf2(n) {
    n--;
    n |= n >> 1; n |= n >> 2; n |= n >> 4;
    n |= n >> 8; n |= n >> 16;
    return n + 1;
  }

  stats() {
    let pooledBytes = 0;
    for (const [size, pool] of this.pools) {
      pooledBytes += size * pool.length;
    }
    return {
      totalAllocated: this.totalAllocated,
      totalReused: this.totalReused,
      reuseRate: this.totalReused / (this.totalAllocated + this.totalReused),
      pooledBytes,
      pooledMB: (pooledBytes / 1048576).toFixed(1),
    };
  }
}

// Global pool instance
const bufferPool = new BufferPool();

// Usage in a visualization layer
function updateLayerData(newData) {
  // Release old buffers
  if (currentPositions) bufferPool.releaseTypedArray(currentPositions);
  if (currentColors) bufferPool.releaseTypedArray(currentColors);

  // Acquire new buffers from pool (likely reused, avoiding GC)
  currentPositions = bufferPool.acquireFloat32Array(newData.length * 3);
  currentColors = bufferPool.acquireUint8Array(newData.length * 4);

  // Fill buffers...
}
```

### 14.2 Web Worker Offloading

```javascript
// worker.js -- Offload data processing to a Web Worker
// This prevents main-thread jank during large data operations

self.onmessage = async function (e) {
  const { type, payload } = e.data;

  switch (type) {
    case 'PROCESS_GEOJSON': {
      const { features } = payload;
      const count = features.length;

      // Allocate typed arrays in the worker
      const positions = new Float32Array(count * 3);
      const colors = new Uint8Array(count * 4);

      for (let i = 0; i < count; i++) {
        const coords = features[i].geometry.coordinates;
        positions[i * 3] = coords[0];
        positions[i * 3 + 1] = coords[1];
        positions[i * 3 + 2] = 0;

        const value = features[i].properties.value;
        const [r, g, b] = valueToColor(value);
        colors[i * 4] = r;
        colors[i * 4 + 1] = g;
        colors[i * 4 + 2] = b;
        colors[i * 4 + 3] = 200;
      }

      // Transfer ownership to main thread (zero-copy!)
      self.postMessage(
        { type: 'PROCESSED', positions, colors, count },
        [positions.buffer, colors.buffer]  // Transferable objects
      );
      break;
    }

    case 'BUILD_SPATIAL_INDEX': {
      const { lons, lats } = payload;
      const Flatbush = (await import('flatbush')).default;
      const index = new Flatbush(lons.length);
      for (let i = 0; i < lons.length; i++) {
        index.add(lons[i], lats[i], lons[i], lats[i]);
      }
      index.finish();

      // Transfer the index's internal ArrayBuffer
      self.postMessage(
        { type: 'INDEX_BUILT', data: index.data },
        [index.data]
      );
      break;
    }
  }
};

function valueToColor(v) {
  // Simple red-blue diverging scale
  const t = Math.max(0, Math.min(1, v));
  return [Math.floor(255 * t), 50, Math.floor(255 * (1 - t))];
}
```

```javascript
// Main thread: use the worker
const worker = new Worker('/worker.js', { type: 'module' });

worker.onmessage = (e) => {
  const { type, positions, colors, count } = e.data;
  if (type === 'PROCESSED') {
    // Data arrived via zero-copy transfer -- update the deck.gl layer
    deck.setProps({
      layers: [
        new ScatterplotLayer({
          id: 'worker-processed',
          data: { length: count },
          getPosition: { value: positions, size: 3 },
          getFillColor: { value: colors, size: 4 },
          radiusMinPixels: 2,
        }),
      ],
    });
  }
};

// Send data to worker for processing
async function loadAndProcess(url) {
  const response = await fetch(url);
  const geojson = await response.json();
  worker.postMessage({
    type: 'PROCESS_GEOJSON',
    payload: geojson,
  });
}
```

### 14.3 Memory Budget Guidelines

| Component | Typical Budget | Notes |
|-----------|---------------|-------|
| Total JS heap | 2--4 GB | Chrome limits vary by device |
| GPU memory (textures) | 256MB -- 2GB | Shared with OS compositor |
| Single typed array | < 500MB | Avoid single > 1GB allocations |
| Tile cache (MVT) | 50--200MB | MapLibre `maxTileCacheSize` |
| Point cloud budget | 5--15M points | ~200--600MB with XYZRGB |
| Worker memory | 512MB each | Each worker has its own heap |

**Memory estimation formulas:**

```
Points (XY + color):      numPoints * (4*2 + 4) = 12 bytes/point
Points (XYZ + RGBA):      numPoints * (4*3 + 4) = 16 bytes/point
Polygons (avg 20 verts):  numPolygons * 20 * (4*2) = 160 bytes/polygon
Raster tile (256x256 RGBA): 256KB per tile
Vector tile (typical):     20--100KB per tile (compressed)
```

> **Cross-reference:** See [js-bindbox/performance-optimization.md](../js-bindbox/performance-optimization.md) for JavaScript memory management best practices.

---

## 15. Benchmarking and Profiling

Rigorous performance measurement separates amateur from production-grade visualization. This section covers the tools and techniques for measuring and improving rendering performance.

### 15.1 FPS Measurement

```javascript
// Production-grade FPS monitor
class FPSMonitor {
  constructor(options = {}) {
    this.sampleWindow = options.sampleWindow || 60; // frames
    this.frameTimes = new Float64Array(this.sampleWindow);
    this.frameIndex = 0;
    this.lastTimestamp = 0;
    this.frameCount = 0;
    this.warningThreshold = options.warningThreshold || 24;
    this.onWarning = options.onWarning || (() => {});
  }

  tick(timestamp = performance.now()) {
    if (this.lastTimestamp > 0) {
      const dt = timestamp - this.lastTimestamp;
      this.frameTimes[this.frameIndex % this.sampleWindow] = dt;
      this.frameIndex++;
      this.frameCount++;

      // Check for sustained low FPS
      if (this.frameIndex >= this.sampleWindow) {
        const stats = this.getStats();
        if (stats.avgFPS < this.warningThreshold) {
          this.onWarning(stats);
        }
      }
    }
    this.lastTimestamp = timestamp;
  }

  getStats() {
    const count = Math.min(this.frameIndex, this.sampleWindow);
    if (count === 0) return { avgFPS: 0, p95FrameTime: 0, p99FrameTime: 0 };

    let sum = 0;
    const sorted = [];
    for (let i = 0; i < count; i++) {
      sum += this.frameTimes[i];
      sorted.push(this.frameTimes[i]);
    }
    sorted.sort((a, b) => a - b);

    return {
      avgFPS: 1000 / (sum / count),
      avgFrameTime: sum / count,
      p50FrameTime: sorted[Math.floor(count * 0.50)],
      p95FrameTime: sorted[Math.floor(count * 0.95)],
      p99FrameTime: sorted[Math.floor(count * 0.99)],
      minFPS: 1000 / sorted[count - 1],
      maxFPS: 1000 / sorted[0],
      totalFrames: this.frameCount,
    };
  }

  // Integration with requestAnimationFrame
  start() {
    const loop = (timestamp) => {
      this.tick(timestamp);
      this._raf = requestAnimationFrame(loop);
    };
    this._raf = requestAnimationFrame(loop);
  }

  stop() {
    cancelAnimationFrame(this._raf);
  }
}

// Usage
const fps = new FPSMonitor({
  warningThreshold: 24,
  onWarning: (stats) => {
    console.warn(`Low FPS detected: ${stats.avgFPS.toFixed(1)} FPS ` +
      `(p95 frame time: ${stats.p95FrameTime.toFixed(1)}ms)`);
    // Automatically reduce quality
    reduceLOD();
  },
});
fps.start();
```

### 15.2 Chrome DevTools GPU Profiling

**Step-by-step profiling workflow:**

1. **Performance tab recording:**
   - Open DevTools -> Performance tab
   - Enable "Screenshots" and "Web Vitals"
   - Click Record, interact with the map for 5-10 seconds, stop
   - Look for: long frames (> 16.67ms), long tasks, excessive GC

2. **GPU profiling (chrome://gpu):**
   - Navigate to `chrome://gpu` to check WebGL/WebGPU capabilities
   - Verify hardware acceleration is enabled
   - Check for WebGL context loss warnings

3. **Memory tab:**
   - Take heap snapshots before and after loading data
   - Compare to find leaked objects
   - Check for detached DOM trees (common with map popups)

4. **WebGL-specific:**
   ```javascript
   // Get WebGL debug info
   const gl = canvas.getContext('webgl2');
   const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
   if (debugInfo) {
     console.log('GPU Vendor:', gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL));
     console.log('GPU Renderer:', gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL));
   }

   // Monitor WebGL resource usage
   console.log('Max texture size:', gl.getParameter(gl.MAX_TEXTURE_SIZE));
   console.log('Max vertex attribs:', gl.getParameter(gl.MAX_VERTEX_ATTRIBS));
   console.log('Max uniform vectors:', gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS));
   ```

### 15.3 WebGL Inspector Tools

| Tool | Purpose | URL |
|------|---------|-----|
| **Spector.js** | Capture/inspect WebGL frames | https://spector.babylonjs.com |
| **WebGL Inspector** | Capture draw calls, state | https://benvanik.github.io/WebGL-Inspector |
| **RenderDoc** | GPU frame capture (native) | https://renderdoc.org |
| **Chrome Tracing** | Low-level GPU timing | `chrome://tracing` |
| **GPU.js** | Compute benchmarks | https://gpu.rocks |

```javascript
// Spector.js integration for debugging WebGL calls
import { Spector } from 'spectorjs';

const spector = new Spector();

// Capture a single frame
document.getElementById('capture-btn').onclick = () => {
  spector.captureNextFrame(canvas);
};

spector.onCapture.add((capture) => {
  console.log('Draw calls:', capture.commands.length);
  console.log('State changes:', capture.commands.filter(c => c.name.startsWith('bindBuffer')).length);

  // Analyze for performance issues
  const drawCalls = capture.commands.filter(c =>
    c.name === 'drawArrays' || c.name === 'drawElements' || c.name === 'drawArraysInstanced'
  );
  console.log(`Frame: ${drawCalls.length} draw calls`);

  // Look for excessive state changes (common perf issue)
  let texBinds = 0;
  let progSwitches = 0;
  for (const cmd of capture.commands) {
    if (cmd.name === 'bindTexture') texBinds++;
    if (cmd.name === 'useProgram') progSwitches++;
  }
  console.log(`Texture binds: ${texBinds}, Program switches: ${progSwitches}`);
});
```

### 15.4 Performance Budgets

| Metric | Target (Desktop) | Target (Mobile) | Measurement |
|--------|-----------------|-----------------|-------------|
| Frame rate | >= 30 FPS sustained | >= 24 FPS sustained | FPSMonitor |
| Frame time (p95) | < 33ms | < 42ms | Performance API |
| Time to first tile | < 1s | < 2s | Performance API |
| Time to interactive | < 3s | < 5s | Lighthouse |
| Memory (JS heap) | < 1GB | < 512MB | `performance.memory` |
| GPU memory | < 1GB | < 256MB | Estimated from resources |
| Initial bundle size | < 500KB (gzipped) | < 300KB (gzipped) | Webpack analyzer |
| Tile request latency (p50) | < 100ms | < 200ms | Resource timing |

```javascript
// Automated performance budget enforcement
class PerformanceBudget {
  constructor(budgets) {
    this.budgets = budgets;
    this.violations = [];
  }

  check() {
    this.violations = [];

    // Memory check
    if (performance.memory) {
      const heapMB = performance.memory.usedJSHeapSize / 1048576;
      if (heapMB > this.budgets.maxHeapMB) {
        this.violations.push({
          metric: 'JS Heap',
          value: `${heapMB.toFixed(0)}MB`,
          budget: `${this.budgets.maxHeapMB}MB`,
        });
      }
    }

    // FPS check (requires FPSMonitor)
    if (this.fpsMonitor) {
      const stats = this.fpsMonitor.getStats();
      if (stats.avgFPS < this.budgets.minFPS) {
        this.violations.push({
          metric: 'FPS',
          value: stats.avgFPS.toFixed(1),
          budget: this.budgets.minFPS,
        });
      }
    }

    return this.violations;
  }

  enforce(onViolation) {
    setInterval(() => {
      const violations = this.check();
      if (violations.length > 0) {
        onViolation(violations);
      }
    }, 5000);
  }
}

// Usage
const budget = new PerformanceBudget({
  maxHeapMB: 1024,
  minFPS: 30,
  maxFrameTimeP95: 33,
});

budget.enforce((violations) => {
  console.warn('Performance budget violations:', violations);
  // Automatically degrade: reduce point count, disable effects, etc.
});
```

> **Cross-reference:** See [js-bindbox/performance-optimization.md](../js-bindbox/performance-optimization.md) for a comprehensive performance optimization guide.

---

## 16. Case Studies

### 16.1 Billion-Point NYC Taxi Trip Visualization

**Dataset:** 1.1 billion NYC taxi trips (2009--2024), ~150GB raw CSV, ~40GB Parquet.

**Architecture:**

```
Data Pipeline:
  Raw CSV -> DuckDB (local) -> Parquet (partitioned by month)
                             -> H3 pre-aggregation (res 7-10) -> Arrow IPC files

Serving:
  Pre-aggregated Arrow IPC files on CloudFront CDN
  Full-resolution data available via DuckDB-WASM (Parquet on S3)

Client:
  MapLibre GL JS (basemap)
  + deck.gl H3HexagonLayer (hex aggregation)
  + deck.gl ArcLayer (trip trajectories, sampled)
  + DuckDB-WASM for on-demand analytical queries
```

**Key metrics:**

| Metric | Value |
|--------|-------|
| Total data points | 1.1 billion |
| Initial load time | 1.8s (H3 res-7 aggregates: 2MB Arrow) |
| Zoom 0--8 | Pre-aggregated H3 (800K hexagons) |
| Zoom 8--12 | DuckDB-WASM query (H3 res-9, ~5M hexagons) |
| Zoom 12+ | FlatGeobuf spatial streaming (individual trips) |
| FPS (pan/zoom) | 55--60 |
| Memory (peak) | 850MB |
| Bundle size | 380KB gzipped |

**Pre-aggregation SQL (DuckDB):**

```sql
-- Pre-compute H3 aggregates at multiple resolutions
-- Run once; serve the result as static Arrow IPC files

COPY (
  SELECT
    h3_latlng_to_cell(pickup_latitude, pickup_longitude, 7) AS h3_cell,
    7 AS h3_resolution,
    COUNT(*) AS trip_count,
    AVG(fare_amount) AS avg_fare,
    AVG(tip_amount) AS avg_tip,
    AVG(trip_distance) AS avg_distance,
    date_trunc('month', pickup_datetime) AS month
  FROM read_parquet('s3://bucket/nyc-taxi/yellow_tripdata_*.parquet')
  WHERE pickup_latitude BETWEEN 40.4 AND 41.0
    AND pickup_longitude BETWEEN -74.3 AND -73.5
  GROUP BY h3_cell, month
) TO 'h3_agg_res7.arrow' (FORMAT 'arrow');
```

### 16.2 Global AIS Ship Tracking

**Dataset:** 300,000+ active vessel positions updated every 2--30 seconds via AIS (Automatic Identification System), with 6-month historical tracks.

**Architecture:**

```
AIS Data Feed (terrestrial + satellite)
    |
    v
Apache Kafka (partitioned by MMSI hash)
    |
    +-> Real-time path: Kafka -> Node.js consumer -> WebSocket -> Browser
    |                           (filter by viewport, downsample to 30s intervals)
    |
    +-> Historical path: Kafka -> ClickHouse (time-series DB)
                                    |
                                    v
                         PostGIS (spatial queries) -> Martin (MVT) -> MapLibre
```

**Key metrics:**

| Metric | Value |
|--------|-------|
| Active vessels displayed | 300,000+ |
| Position update rate | ~50,000 messages/second (total) |
| Browser update rate | 10 fps (100ms batches) |
| Historical track length | 6 months |
| Zoom 0--6 | HexagonLayer GPU aggregation |
| Zoom 6--10 | ScatterplotLayer (binary, all 300K vessels) |
| Zoom 10+ | Individual vessel icons + trail polylines |
| FPS | 45--60 |
| WebSocket bandwidth | ~200KB/s (compressed binary) |

```javascript
// AIS-specific optimizations
const aisLayer = new ScatterplotLayer({
  id: 'vessels',
  data: { length: vesselCount },
  getPosition: { value: vesselPositions, size: 3 },
  getFillColor: { value: vesselColors, size: 4 },
  getRadius: 500,
  radiusMinPixels: 2,
  radiusMaxPixels: 12,
  // Key optimization: don't recalculate attributes, just upload new buffer data
  dataComparator: () => false,  // Always treat data as changed
  updateTriggers: {
    getPosition: frameCounter,  // Trigger update each frame
    getFillColor: frameCounter,
  },
});

// Vessel trail rendering with TripsLayer (animated temporal paths)
import { TripsLayer } from '@deck.gl/geo-layers';

const trailLayer = new TripsLayer({
  id: 'vessel-trails',
  data: vesselTracks,
  getPath: d => d.waypoints.map(w => [w.lon, w.lat]),
  getTimestamps: d => d.waypoints.map(w => w.timestamp),
  getColor: d => d.vesselType === 'cargo' ? [0, 128, 255] : [255, 64, 0],
  widthMinPixels: 2,
  trailLength: 3600,  // Show last hour of trail
  currentTime: Date.now() / 1000,
  shadowEnabled: false,
});
```

### 16.3 Satellite Constellation Monitoring

**Dataset:** 8,000+ active satellites (LEO, MEO, GEO) with TLE (Two-Line Element) orbital parameters, updated every 6 hours, propagated to real-time positions client-side.

**Architecture:**

```
Space-Track.org / CelesTrak (TLE data, updated every 6h)
    |
    v
Backend (Python): Download TLEs -> Store in PostgreSQL
    |
    v  [API: GET /api/tles?epoch=latest]
Browser:
    |
    +-> satellite.js (SGP4 propagation in Web Worker)
    |     Propagates all 8,000 TLEs to current epoch every second
    |
    +-> deck.gl ScatterplotLayer (satellite positions)
    +-> deck.gl LineLayer (orbital tracks, next 90 minutes)
    +-> CesiumJS (optional: 3D globe with atmospheric effects)
```

```javascript
// SGP4 orbital propagation in a Web Worker
// worker-sgp4.js
import { twoline2satrec, propagate, gstime, eciToGeodetic } from 'satellite.js';

let satellites = [];

self.onmessage = (e) => {
  if (e.data.type === 'LOAD_TLES') {
    satellites = e.data.tles.map(tle => ({
      satrec: twoline2satrec(tle.line1, tle.line2),
      name: tle.name,
      noradId: tle.noradId,
    }));
  }

  if (e.data.type === 'PROPAGATE') {
    const date = new Date(e.data.timestamp);
    const gmst = gstime(date);
    const positions = new Float32Array(satellites.length * 3);
    const velocities = new Float32Array(satellites.length);

    for (let i = 0; i < satellites.length; i++) {
      const { satrec } = satellites[i];
      const posVel = propagate(satrec, date);

      if (posVel.position) {
        const geo = eciToGeodetic(posVel.position, gmst);
        positions[i * 3] = (geo.longitude * 180) / Math.PI;
        positions[i * 3 + 1] = (geo.latitude * 180) / Math.PI;
        positions[i * 3 + 2] = geo.height; // km above sea level

        const vel = posVel.velocity;
        velocities[i] = Math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2);
      }
    }

    self.postMessage(
      { type: 'POSITIONS', positions, velocities, timestamp: e.data.timestamp },
      [positions.buffer, velocities.buffer]
    );
  }
};
```

**Key metrics:**

| Metric | Value |
|--------|-------|
| Satellites tracked | 8,000+ |
| SGP4 propagation (8K sats) | ~4ms (Web Worker) |
| Orbital track computation (90min) | ~35ms (90 propagation steps per sat) |
| Update rate | 1 Hz (position), 0.1 Hz (tracks) |
| FPS | 60 |
| Memory | 120MB |

> **Cross-reference:** See [data-sources/space-aerospace.md](../data-sources/space-aerospace.md) for satellite and aerospace data sources.

---

## 17. Tool Comparison Matrix

### 17.1 Rendering Frameworks

| Feature | deck.gl v9 | MapLibre GL JS v5 | CesiumJS | Leaflet.glify | OpenLayers |
|---------|-----------|-------------------|----------|---------------|------------|
| **Max points (interactive)** | 10M+ | 1M (MVT) | 500K (direct), 1B+ (3D Tiles) | 1M | 500K |
| **GPU instancing** | Yes | Internal | Yes | Yes | No |
| **WebGPU support** | Experimental | Planned | Experimental | No | No |
| **Custom shaders** | Yes (injection) | Yes (custom layers) | Yes | Limited | No |
| **3D support** | Yes (globe + flat) | Pitch/bearing only | Full 3D globe | No | No |
| **Binary data input** | Native | Via MVT | Via 3D Tiles | Partial | No |
| **Picking (hover/click)** | GPU-based (fast) | Feature state | Yes | Limited | Feature-based |
| **Bundle size** | ~200KB gz | ~180KB gz | ~500KB gz | ~10KB gz | ~150KB gz |
| **License** | MIT | BSD-3 | Apache 2.0 | MIT | BSD-2 |
| **Best for** | Data overlays | Basemap + light data | 3D, terrain, point clouds | Simple large points | Enterprise GIS |

### 17.2 Tiling and Data Formats

| Format | Type | Spatial Query | Streaming | Browser Decode | Best For |
|--------|------|--------------|-----------|---------------|----------|
| **MVT** | Vector tiles | By tile grid | Tile-by-tile | Fast (protobuf) | Basemaps, features |
| **PMTiles** | Tile archive | By tile grid | Range requests | Fast | Serverless hosting |
| **FlatGeobuf** | Vector features | Bbox query | Spatial streaming | Fast (flatbuffers) | Large feature collections |
| **Arrow IPC** | Columnar tabular | No (use DuckDB) | Record batch | Zero-copy | Analytics, deck.gl binary |
| **GeoParquet** | Columnar tabular | Via spatial metadata | Partial (row groups) | Via DuckDB-WASM | Data lakes, archives |
| **COG** | Raster | Bbox + overview | Range requests | Via GeoTIFF.js | Satellite imagery, DEMs |
| **3D Tiles** | 3D meshes/points | By tile hierarchy | LOD streaming | glTF decoder | Buildings, point clouds |
| **MBTiles** | Tile archive | By tile grid | No (requires server) | Fast | Offline / mobile |

### 17.3 Spatial Indexing

| Index | Type | Build Time (1M pts) | Query Time | Memory | Best For |
|-------|------|---------------------|-----------|--------|----------|
| **Flatbush** | Static R-tree | 85ms | 0.05ms | 24MB | Client bbox query |
| **kdbush** | Static KD-tree | 65ms | 0.03ms | 16MB | Client point query |
| **H3** | Hexagonal grid | N/A (hash) | O(1) per point | Minimal | Aggregation, analytics |
| **S2** | Spherical cells | N/A (hash) | O(1) per point | Minimal | Covering, containment |
| **R-tree (rbush)** | Dynamic R-tree | 200ms | 0.1ms | 48MB | Mutable feature sets |
| **PostGIS GiST** | Server R-tree | ~30s | ~1ms (w/ disk) | On disk | Server-side SQL |

### 17.4 Scale Capability Summary

| Approach | Point Limit | Polygon Limit | Raster Limit | Interaction Level |
|----------|-------------|--------------|-------------|-------------------|
| Raw GeoJSON + Leaflet | 10K | 5K | N/A | Full |
| MapLibre + MVT | 1M display / unlimited source | 500K display | 10K tiles | Full |
| deck.gl binary | 10M | 1M | Via BitmapLayer | Full |
| deck.gl + aggregation | 100M (pre-aggregated) | N/A | N/A | Aggregate hover |
| deck.gl + tiling | Unlimited (LOD) | Unlimited (LOD) | Unlimited (COG) | Tile-level |
| DuckDB-WASM analytics | 100M rows | N/A | N/A | SQL-level |
| Potree / 3D Tiles | 10B+ (octree LOD) | N/A | N/A | Navigate + point pick |
| Server aggregation + client | Unlimited | Unlimited | Unlimited | Re-query |

---

## Further Reading and Resources

### Essential Libraries

| Library | Purpose | URL |
|---------|---------|-----|
| deck.gl | GPU data visualization | https://deck.gl |
| MapLibre GL JS | Vector map rendering | https://maplibre.org |
| CesiumJS | 3D globe and terrain | https://cesium.com/cesiumjs |
| loaders.gl | Data format loading | https://loaders.gl |
| luma.gl | WebGL/WebGPU abstraction | https://luma.gl |
| Potree | Point cloud rendering | https://potree.github.io |
| Flatbush | Spatial indexing | https://github.com/mourner/flatbush |
| H3-js | Hexagonal grid | https://github.com/uber/h3-js |
| GeoTIFF.js | COG reading | https://geotiffjs.github.io |
| FlatGeobuf | Spatial streaming | https://flatgeobuf.org |
| DuckDB-WASM | In-browser SQL | https://duckdb.org/docs/api/wasm |
| Mosaic | DuckDB visualization | https://uwdata.github.io/mosaic |
| PMTiles | Serverless tiles | https://protomaps.com/docs/pmtiles |
| Tippecanoe | Tile generation | https://github.com/felt/tippecanoe |
| Martin | Dynamic tile server | https://martin.maplibre.org |
| TiTiler | Dynamic raster tiles | https://developmentseed.org/titiler |
| satellite.js | Orbital propagation | https://github.com/shashwatak/satellite-js |
| Apache Arrow JS | Columnar data | https://arrow.apache.org/docs/js |

### Key Specifications

| Specification | URL |
|--------------|-----|
| Mapbox Vector Tile (MVT) | https://github.com/mapbox/vector-tile-spec |
| OGC 3D Tiles | https://www.ogc.org/standard/3dtiles |
| Cloud-Optimized GeoTIFF | https://www.cogeo.org |
| PMTiles | https://github.com/protomaps/PMTiles/blob/main/spec/v3/spec.md |
| GeoParquet | https://geoparquet.org |
| FlatGeobuf | https://github.com/flatgeobuf/flatgeobuf/blob/master/SPEC.md |
| WebGPU | https://www.w3.org/TR/webgpu |
| WebGL 2.0 | https://registry.khronos.org/webgl/specs/latest/2.0 |
| H3 | https://h3geo.org/docs |
| S2 Geometry | https://s2geometry.io |

### Cross-References in This Repository

| Topic | Path |
|-------|------|
| Performance optimization (JS) | [js-bindbox/performance-optimization.md](../js-bindbox/performance-optimization.md) |
| Tile servers | [js-bindbox/tile-servers.md](../js-bindbox/tile-servers.md) |
| Data formats and loading | [js-bindbox/data-formats-loading.md](../js-bindbox/data-formats-loading.md) |
| Spatial analysis | [js-bindbox/spatial-analysis.md](../js-bindbox/spatial-analysis.md) |
| Real-time and offline | [js-bindbox/realtime-offline-advanced.md](../js-bindbox/realtime-offline-advanced.md) |
| Web mapping tools | [tools/web-mapping.md](../tools/web-mapping.md) |
| Server publishing | [tools/server-publishing.md](../tools/server-publishing.md) |
| 3D visualization | [visualization/3d-visualization.md](./3d-visualization.md) |
| Scientific visualization | [visualization/scientific-visualization.md](./scientific-visualization.md) |
| Dashboards | [visualization/dashboards.md](./dashboards.md) |
| Satellite imagery | [data-sources/satellite-imagery.md](../data-sources/satellite-imagery.md) |
| Elevation and terrain | [data-sources/elevation-terrain.md](../data-sources/elevation-terrain.md) |
| Space and aerospace | [data-sources/space-aerospace.md](../data-sources/space-aerospace.md) |

---

*This guide is part of the [awesome-giser](../README.md) repository. Contributions welcome -- see [CONTRIBUTING.md](../CONTRIBUTING.md).*
