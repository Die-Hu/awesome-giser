# Data Storytelling, Scrollytelling & Narrative Maps

> Techniques and tools for transforming geospatial data into compelling narratives -- from scroll-driven map experiences to cinematic flyovers, multimedia integration, scientific communication, client deliverables, and accessible interactive stories.

**Why it matters:** Raw maps inform. Story maps persuade. The difference between a dashboard and a Pulitzer-winning interactive is narrative structure, pacing, and intentional visual design. This guide covers the full spectrum from code-level scroll triggers to platform-based story builders, with runnable code examples throughout.

---

## Quick Picks -- What To Use Right Now

| Need | Tool | Link | Notes |
|------|------|------|-------|
| Open-source scrollytelling | **Scrollama + MapLibre** | [scrollama](https://github.com/russellsamora/scrollama) / [MapLibre](https://maplibre.org/) | Best flexibility, zero vendor lock-in |
| Zero-code story map | **Knight Lab StoryMapJS** | [storymap.knightlab.com](https://storymap.knightlab.com/) | Free, academic-friendly, embeddable |
| Enterprise narrative | **ArcGIS StoryMaps** | [storymaps.arcgis.com](https://storymaps.arcgis.com/) | Polished UI, org-level sharing |
| Developer-first template | **Mapbox Storytelling** | [github.com/mapbox/storytelling](https://github.com/mapbox/storytelling) | Config-driven, quick prototype |
| React + 3D narratives | **deck.gl + Scrollama** | [deck.gl](https://deck.gl/) | GPU-accelerated layers, view state transitions |
| Data-reactive documents | **Observable Framework** | [observablehq.com/framework](https://observablehq.com/framework) | Notebook-to-site pipeline |
| Before/after comparison | **Juxtapose.js** | [juxtapose.knightlab.com](https://juxtapose.knightlab.com/) | Image slider, widely adopted |
| Cinematic flyovers | **Google Earth Studio** | [earth.google.com/studio](https://earth.google.com/studio) | Keyframe animation, 4K export |
| Collaborative mapping | **Felt** | [felt.com](https://felt.com/) | Real-time multiplayer, embed/share |
| Timeline narratives | **TimelineJS** | [timeline.knightlab.com](https://timeline.knightlab.com/) | Pairs with StoryMapJS |

---

## Table of Contents

1. [Introduction & Quick Picks](#quick-picks----what-to-use-right-now)
2. [Scrollytelling Fundamentals](#1-scrollytelling-fundamentals)
3. [Scrollama.js](#2-scrollamajs)
4. [MapLibre + Scrollytelling](#3-maplibre--scrollytelling)
5. [deck.gl + Scrollytelling](#4-deckgl--scrollytelling)
6. [Observable Framework](#5-observable-framework)
7. [StoryMapJS (Knight Lab)](#6-storymapjs-knight-lab)
8. [ArcGIS StoryMaps](#7-arcgis-storymaps)
9. [Mapbox Storytelling Template](#8-mapbox-storytelling-template)
10. [Journalism & Data Storytelling](#9-journalism--data-storytelling)
11. [Interactive Narratives](#10-interactive-narratives)
12. [Multimedia Integration](#11-multimedia-integration)
13. [Scientific Communication](#12-scientific-communication)
14. [Client Deliverables](#13-client-deliverables--甲方交付)
15. [Accessibility in Storytelling](#14-accessibility-in-storytelling)
16. [Performance Optimization](#15-performance-optimization)
17. [Publishing & Hosting](#16-publishing--hosting)
18. [Tool Comparison Matrix](#17-tool-comparison-matrix)

---

## 1. Scrollytelling Fundamentals

Scrollytelling couples the user's scroll position with map state changes -- camera moves, layer toggles, annotation reveals, and data transitions. It is the dominant pattern in modern data journalism and increasingly in scientific communication and client reporting.

### Core Vocabulary

| Concept | Description |
|---------|-------------|
| **Scroll trigger** | A DOM element whose intersection with the viewport fires a callback |
| **Step** | A discrete narrative unit (text block + map state) |
| **Fly-to** | Animated camera transition to a new center/zoom/bearing/pitch |
| **Layer toggle** | Showing or hiding map layers in sync with scroll position |
| **Sticky map** | The map remains fixed (`position: sticky`) while text scrolls over it |
| **Progress** | A 0-1 value indicating how far through a step the user has scrolled |
| **Waypoint** | A named scroll position that triggers a discrete event |
| **Continuous scroll** | Scroll position directly drives a continuous variable (opacity, extrusion height) |

### Waypoint-Based vs. Continuous Scroll

The two fundamental scrollytelling paradigms:

```
WAYPOINT-BASED (discrete)
─────────────────────────
Step 1  ████████░░░░░░░░░░░░░░░  --> flyTo(Beijing, zoom=10)
Step 2  ░░░░░░░░████████░░░░░░░  --> flyTo(Shanghai, zoom=12)
Step 3  ░░░░░░░░░░░░░░░░████████  --> flyTo(Guangzhou, zoom=11)

CONTINUOUS (interpolated)
─────────────────────────
Scroll  ▓▓▓▓▒▒▒▒░░░░░░░░░░░░░░  --> opacity = scrollProgress * 1.0
        0.0 ──────────────── 1.0     zoom = lerp(3, 16, scrollProgress)
```

**When to use which:**
- **Waypoint:** Distinct chapters, location jumps, layer swaps. Simpler to build and debug.
- **Continuous:** Smooth data transitions, camera orbits, time-scrubbing. Requires careful easing.

### Scroll-Driven Map Transition Types

```
fly-to        --> Change center, zoom, bearing, pitch simultaneously
ease-to       --> Smooth pan without zoom change
zoom-to       --> Isolate zoom level change (useful for "drilling down")
layer-swap    --> Cross-fade between layers at a scroll threshold
orbit         --> Rotate bearing continuously while center stays fixed
tilt-reveal   --> Increase pitch to reveal 3D extrusions as user scrolls
```

### The Sticky Map Pattern (CSS)

The foundational layout for all scrollytelling: a sticky map container with scrolling text overlay.

```css
/* The scrolly wrapper */
.scrolly {
  position: relative;
}

/* Sticky map -- stays fixed while text scrolls */
.scrolly__map {
  position: sticky;
  top: 0;
  width: 100%;
  height: 100vh;
  z-index: 0;
}

/* Text column -- overlays the map */
.scrolly__text {
  position: relative;
  z-index: 1;
  pointer-events: none;      /* allow map interaction through gaps */
}

/* Individual step cards */
.scrolly__step {
  min-height: 80vh;           /* enough scroll distance between triggers */
  display: flex;
  align-items: center;
  padding: 2rem;
  pointer-events: auto;       /* re-enable interaction on cards */
}

.scrolly__step-content {
  background: rgba(255, 255, 255, 0.92);
  padding: 1.5rem 2rem;
  border-radius: 4px;
  max-width: 420px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);
}

/* Active step highlight */
.scrolly__step.is-active .scrolly__step-content {
  border-left: 4px solid #2563eb;
}
```

### Intersection Observer API (Zero-Dependency Scroll Triggers)

For projects that cannot include Scrollama, the native Intersection Observer API provides the same capability with zero dependencies.

```javascript
class ScrollyMap {
  constructor(mapInstance, steps) {
    this.map = mapInstance;
    this.steps = steps;
    this.observer = new IntersectionObserver(
      this.handleIntersection.bind(this),
      {
        root: null,                          // viewport
        rootMargin: "0px 0px -40% 0px",      // trigger when step top hits 60% mark
        threshold: [0, 0.25, 0.5, 0.75, 1]
      }
    );

    document.querySelectorAll("[data-scroll-step]").forEach(el => {
      this.observer.observe(el);
    });
  }

  handleIntersection(entries) {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;

      const stepIndex = parseInt(entry.target.dataset.scrollStep, 10);
      const config = this.steps[stepIndex];

      if (entry.intersectionRatio > 0.5) {
        this.map.flyTo({
          center: config.center,
          zoom: config.zoom,
          bearing: config.bearing || 0,
          pitch: config.pitch || 0,
          duration: config.duration || 1500
        });

        // Toggle layers
        config.showLayers?.forEach(id =>
          this.map.setLayoutProperty(id, "visibility", "visible")
        );
        config.hideLayers?.forEach(id =>
          this.map.setLayoutProperty(id, "visibility", "none")
        );
      }
    });
  }

  destroy() {
    this.observer.disconnect();
  }
}
```

### Dark Arts: Scroll UX Tricks

- **Parallax depth:** Assign different scroll speeds to map overlays using `transform: translateY(calc(var(--scroll) * 0.3))`.
- **Snap points:** Use `scroll-snap-type: y mandatory` on the scroll container for chapter-to-chapter snapping.
- **Preloading:** Trigger `map.flyTo` one step early with `duration: 0` and then animate opacity, so tiles are already loaded when the user arrives.
- **Mobile override:** On viewports below 768px, switch from sticky-side layout to stacked (text above, map below) to avoid cramped panels.
- **Scroll hijack prevention:** Never override the user's scroll velocity or direction. Use `passive: true` on scroll listeners.

> **Cross-reference:** For responsive CSS patterns, see [../web-dev/frontend-integration.md](../web-dev/frontend-integration.md). For MapLibre fundamentals, see [../js-bindbox/](../js-bindbox/).

---

## 2. Scrollama.js

[Scrollama](https://github.com/russellsamora/scrollama) (MIT license) is the de facto standard for scroll-driven storytelling. It wraps the Intersection Observer API with a clean, ergonomic interface.

**Install:**

```bash
npm install scrollama
# or CDN:
```

```html
<script src="https://unpkg.com/scrollama@3.x/scrollama.min.js"></script>
```

### Basic Setup

```javascript
import scrollama from "scrollama";

const scroller = scrollama();

scroller
  .setup({
    step: ".step",           // CSS selector for step elements
    offset: 0.5,             // trigger at vertical center of viewport (0-1)
    progress: true,          // enable onStepProgress with 0-1 value
    threshold: 4,            // number of trigger pixels (higher = smoother progress)
    debug: false             // set true to show trigger line
  })
  .onStepEnter(({ element, index, direction }) => {
    // Fires when a step enters the trigger offset
    console.log(`Entered step ${index} scrolling ${direction}`);
    element.classList.add("is-active");
  })
  .onStepExit(({ element, index, direction }) => {
    // Fires when a step exits the trigger offset
    element.classList.remove("is-active");
  })
  .onStepProgress(({ element, index, progress }) => {
    // Fires continuously as user scrolls through a step
    // progress: 0.0 (just entered) to 1.0 (about to exit)
    console.log(`Step ${index}: ${(progress * 100).toFixed(0)}%`);
  });

// IMPORTANT: handle resize events
window.addEventListener("resize", scroller.resize);
```

### Debug Mode

Enable debug to see where your trigger line sits:

```javascript
scroller.setup({
  step: ".step",
  offset: 0.5,
  debug: true       // renders a red line at the offset position
});
```

### Responsive Offset Strategy

```javascript
// Different trigger points for mobile vs desktop
function getOffset() {
  return window.innerWidth < 768 ? 0.7 : 0.5;
}

scroller.setup({
  step: ".step",
  offset: getOffset()
});

window.addEventListener("resize", () => {
  scroller.setup({ step: ".step", offset: getOffset() });
  scroller.resize();
});
```

### Multiple Scrollers on One Page

```javascript
// Separate scrollers for different story sections
const introScroller = scrollama();
const analysisScroller = scrollama();

introScroller.setup({ step: "#intro .step", offset: 0.5 })
  .onStepEnter(handleIntroStep);

analysisScroller.setup({ step: "#analysis .step", offset: 0.4 })
  .onStepEnter(handleAnalysisStep);
```

### Scrollama with GSAP (Advanced Animation)

```javascript
import { gsap } from "gsap";

scroller
  .setup({ step: ".step", offset: 0.5, progress: true })
  .onStepEnter(({ index }) => {
    // Trigger GSAP timeline for complex animations
    const tl = gsap.timeline();
    tl.to("#map-overlay", { opacity: 1, duration: 0.5 })
      .to("#chart-container", { y: 0, opacity: 1, duration: 0.8 }, "-=0.3")
      .to(".annotation", { scale: 1, opacity: 1, stagger: 0.1 }, "-=0.4");
  })
  .onStepProgress(({ index, progress }) => {
    // Drive continuous GSAP animation with scroll progress
    gsap.to("#data-ring", { rotation: progress * 360, duration: 0 });
  });
```

---

## 3. MapLibre + Scrollytelling

MapLibre GL JS is the premier open-source WebGL map renderer. Combined with Scrollama, it provides the most flexible scrollytelling stack available.

### Dependencies

```html
<link rel="stylesheet" href="https://unpkg.com/maplibre-gl@4.x/dist/maplibre-gl.css" />
<script src="https://unpkg.com/maplibre-gl@4.x/dist/maplibre-gl.js"></script>
<script src="https://unpkg.com/scrollama@3.x/scrollama.min.js"></script>
```

### Complete Template

**HTML:**

```html
<div id="scrolly">
  <div id="map-container">
    <div id="map"></div>
  </div>
  <div id="story">
    <div class="step" data-step="0">
      <h2>Chapter 1: The Overview</h2>
      <p>We begin with a continental view of population density across China.</p>
    </div>
    <div class="step" data-step="1">
      <h2>Chapter 2: Urban Cores</h2>
      <p>Zooming into the Beijing-Tianjin-Hebei metropolitan cluster...</p>
    </div>
    <div class="step" data-step="2">
      <h2>Chapter 3: Street-Level Detail</h2>
      <p>At street level, individual building footprints reveal urban density.</p>
    </div>
  </div>
</div>
```

**CSS:**

```css
#scrolly {
  position: relative;
}

#map-container {
  position: sticky;
  top: 0;
  width: 100%;
  height: 100vh;
  z-index: 0;
}

#map {
  width: 100%;
  height: 100%;
}

#story {
  position: relative;
  z-index: 1;
  pointer-events: none;
}

.step {
  min-height: 80vh;
  display: flex;
  align-items: center;
  justify-content: flex-start;
  padding: 2rem;
  pointer-events: auto;
}

.step > * {
  background: rgba(255, 255, 255, 0.92);
  padding: 1.5rem 2rem;
  border-radius: 4px;
  max-width: 420px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.12);
}

.step.is-active > * {
  border-left: 4px solid #2563eb;
}
```

**JavaScript:**

```javascript
// Chapter configuration array
const chapters = [
  {
    id: "overview",
    center: [104.0, 35.0],
    zoom: 3.5,
    bearing: 0,
    pitch: 0,
    layers: ["population-density"],
    onEnter: (map) => {
      map.setLayoutProperty("population-density", "visibility", "visible");
      map.setLayoutProperty("building-footprints", "visibility", "none");
    }
  },
  {
    id: "urban-cores",
    center: [116.4, 39.9],
    zoom: 9,
    bearing: 20,
    pitch: 45,
    layers: ["population-density", "urban-boundaries"],
    onEnter: (map) => {
      map.setLayoutProperty("urban-boundaries", "visibility", "visible");
    }
  },
  {
    id: "detail",
    center: [116.39, 39.91],
    zoom: 16,
    bearing: -15,
    pitch: 60,
    layers: ["building-footprints"],
    onEnter: (map) => {
      map.setLayoutProperty("population-density", "visibility", "none");
      map.setLayoutProperty("building-footprints", "visibility", "visible");
    }
  }
];

// Initialize map
const map = new maplibregl.Map({
  container: "map",
  style: "https://demotiles.maplibre.org/style.json",
  center: chapters[0].center,
  zoom: chapters[0].zoom,
  bearing: chapters[0].bearing,
  pitch: chapters[0].pitch,
  interactive: false       // disable user interaction for pure narrative control
});

// Initialize Scrollama
const scroller = scrollama();

scroller
  .setup({
    step: ".step",
    offset: 0.5,
    progress: true
  })
  .onStepEnter(({ index, direction }) => {
    // Update active step styling
    document.querySelectorAll(".step").forEach((el, i) => {
      el.classList.toggle("is-active", i === index);
    });

    // Fly to chapter location
    const ch = chapters[index];
    map.flyTo({
      center: ch.center,
      zoom: ch.zoom,
      bearing: ch.bearing,
      pitch: ch.pitch,
      duration: 2000,
      essential: true
    });

    // Execute chapter-specific layer logic
    if (ch.onEnter) ch.onEnter(map);
  })
  .onStepProgress(({ index, progress }) => {
    // Use progress (0-1) for continuous animations
    if (index === 1) {
      map.setPaintProperty("urban-boundaries", "line-opacity", progress);
    }
  });

// Handle resize
window.addEventListener("resize", scroller.resize);
```

### Camera Orbit Animation

Useful as an idle animation or for dramatic reveals:

```javascript
function startOrbit(map, center, { rpm = 1, zoom = 14, pitch = 60 } = {}) {
  map.jumpTo({ center, zoom, pitch });
  let bearing = 0;
  const degreesPerFrame = (rpm * 360) / 60 / 60; // 60fps assumed

  function rotate() {
    bearing = (bearing + degreesPerFrame) % 360;
    map.rotateTo(bearing, { duration: 0 });
    map._orbitAnimationId = requestAnimationFrame(rotate);
  }

  rotate();
}

function stopOrbit(map) {
  if (map._orbitAnimationId) {
    cancelAnimationFrame(map._orbitAnimationId);
    map._orbitAnimationId = null;
  }
}

// Usage: orbit during a chapter, stop when user scrolls away
scroller.onStepEnter(({ index }) => {
  if (index === 2) startOrbit(map, [116.39, 39.91]);
  else stopOrbit(map);
});
```

### Layer Cross-Fade

```javascript
function crossFadeLayers(map, outLayerId, inLayerId, duration = 1000) {
  const start = performance.now();
  const outProp = getOpacityProp(map.getLayer(outLayerId).type);
  const inProp = getOpacityProp(map.getLayer(inLayerId).type);

  map.setLayoutProperty(inLayerId, "visibility", "visible");
  map.setPaintProperty(inLayerId, inProp, 0);

  function tick(now) {
    const t = Math.min((now - start) / duration, 1);
    const eased = t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; // ease-in-out quad

    map.setPaintProperty(outLayerId, outProp, 1 - eased);
    map.setPaintProperty(inLayerId, inProp, eased);

    if (t < 1) {
      requestAnimationFrame(tick);
    } else {
      map.setLayoutProperty(outLayerId, "visibility", "none");
    }
  }

  requestAnimationFrame(tick);
}

function getOpacityProp(layerType) {
  const mapping = {
    fill: "fill-opacity",
    line: "line-opacity",
    circle: "circle-opacity",
    symbol: "icon-opacity",
    raster: "raster-opacity",
    heatmap: "heatmap-opacity",
    "fill-extrusion": "fill-extrusion-opacity"
  };
  return mapping[layerType] || "fill-opacity";
}
```

> **Cross-reference:** For MapLibre style specification, layer types, and source configuration, see [../js-bindbox/](../js-bindbox/). For 3D terrain and fill-extrusion, see [3d-visualization.md](3d-visualization.md).

---

## 4. deck.gl + Scrollytelling

[deck.gl](https://deck.gl/) provides GPU-accelerated layers for large-scale data visualization. Combined with scroll triggers, it enables narrative-driven exploration of millions of data points.

### React + deck.gl + Scrollama

```bash
npm install @deck.gl/react @deck.gl/layers @deck.gl/core react-scrollama
```

```jsx
import React, { useState, useCallback } from "react";
import { Scrollama, Step } from "react-scrollama";
import DeckGL from "@deck.gl/react";
import { Map } from "react-map-gl/maplibre";
import { ScatterplotLayer, ArcLayer, HexagonLayer } from "@deck.gl/layers";

const CHAPTERS = [
  {
    id: "origins",
    viewState: { longitude: 116.4, latitude: 39.9, zoom: 5, pitch: 0, bearing: 0 },
    title: "Migration Origins",
    text: "280 million internal migrants move across China each year.",
    getLayer: (data) =>
      new ScatterplotLayer({
        id: "origins",
        data: data.origins,
        getPosition: (d) => d.coordinates,
        getRadius: (d) => Math.sqrt(d.population) * 50,
        getFillColor: [65, 182, 196, 180],
        pickable: true
      })
  },
  {
    id: "flows",
    viewState: { longitude: 110, latitude: 33, zoom: 4.5, pitch: 45, bearing: 20 },
    title: "Migration Flows",
    text: "Arcs show the top 500 migration corridors by volume.",
    getLayer: (data) =>
      new ArcLayer({
        id: "arcs",
        data: data.flows,
        getSourcePosition: (d) => d.from,
        getTargetPosition: (d) => d.to,
        getSourceColor: [65, 182, 196],
        getTargetColor: [255, 127, 14],
        getWidth: (d) => Math.log(d.count),
        pickable: true
      })
  },
  {
    id: "density",
    viewState: { longitude: 121.47, latitude: 31.23, zoom: 10, pitch: 55, bearing: -20 },
    title: "Destination Density",
    text: "Shanghai absorbs the largest share, concentrated in Pudong.",
    getLayer: (data) =>
      new HexagonLayer({
        id: "hexagons",
        data: data.destinations,
        getPosition: (d) => d.coordinates,
        radius: 500,
        elevationScale: 50,
        extruded: true,
        pickable: true,
        colorRange: [
          [1, 152, 189], [73, 227, 206], [216, 254, 181],
          [254, 237, 177], [254, 173, 84], [209, 55, 78]
        ]
      })
  }
];

function StoryApp({ data }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [viewState, setViewState] = useState(CHAPTERS[0].viewState);

  const onStepEnter = useCallback(({ data: stepIndex }) => {
    setCurrentStep(stepIndex);
    setViewState(CHAPTERS[stepIndex].viewState);
  }, []);

  const chapter = CHAPTERS[currentStep];
  const layers = chapter.getLayer(data) ? [chapter.getLayer(data)] : [];

  return (
    <div style={{ position: "relative" }}>
      {/* Sticky map */}
      <div style={{ position: "sticky", top: 0, height: "100vh", zIndex: 0 }}>
        <DeckGL
          viewState={viewState}
          onViewStateChange={({ viewState: vs }) => setViewState(vs)}
          layers={layers}
          controller={false}
        >
          <Map mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json" />
        </DeckGL>
      </div>

      {/* Scrolling text */}
      <div style={{ position: "relative", zIndex: 1, pointerEvents: "none" }}>
        <Scrollama offset={0.5} onStepEnter={onStepEnter}>
          {CHAPTERS.map((ch, i) => (
            <Step key={ch.id} data={i}>
              <div style={{
                minHeight: "80vh",
                display: "flex",
                alignItems: "center",
                padding: "2rem",
                pointerEvents: "auto"
              }}>
                <div style={{
                  background: "rgba(0,0,0,0.85)",
                  color: "#fff",
                  padding: "1.5rem 2rem",
                  borderRadius: 8,
                  maxWidth: 400,
                  borderLeft: i === currentStep ? "4px solid #41b6c4" : "none"
                }}>
                  <h2>{ch.title}</h2>
                  <p>{ch.text}</p>
                </div>
              </div>
            </Step>
          ))}
        </Scrollama>
      </div>
    </div>
  );
}
```

### View State Transition Animation

```javascript
import { FlyToInterpolator } from "@deck.gl/core";

// Smooth animated transitions between chapters
function getTransitionViewState(targetViewState) {
  return {
    ...targetViewState,
    transitionDuration: 2000,
    transitionInterpolator: new FlyToInterpolator(),
    transitionEasing: (t) => t * (2 - t)  // ease-out quadratic
  };
}

// In onStepEnter callback:
setViewState(getTransitionViewState(CHAPTERS[stepIndex].viewState));
```

### Layer Fade Transitions

```javascript
// Animate layer opacity based on scroll progress
function getLayerWithProgress(chapter, data, progress) {
  const layer = chapter.getLayer(data);
  return layer.clone({
    opacity: progress,                         // 0 to 1 as user scrolls
    elevationScale: progress * 50,             // extrusions grow with scroll
    radiusScale: 0.5 + progress * 0.5         // points grow from 50% to 100%
  });
}
```

> **Cross-reference:** For deck.gl layer catalog and performance tuning, see [../js-bindbox/](../js-bindbox/). For 3D hexagon and terrain layers, see [3d-visualization.md](3d-visualization.md).

---

## 5. Observable Framework

[Observable Framework](https://observablehq.com/framework) is the successor to Observable notebooks for publishing data-driven websites. It supports scroll-driven reactivity natively and is ideal for data storytelling.

### Scroll-Reactive Map in Observable

```javascript
// In an Observable Framework .md file with embedded JS blocks

// Declare a scroll-reactive variable using Inputs
const step = view(Inputs.range([0, 4], { step: 1, label: "Chapter" }));

// Define location data
const locations = [
  { center: [104.0, 35.0], zoom: 3.5, label: "China Overview" },
  { center: [116.4, 39.9], zoom: 10, label: "Beijing" },
  { center: [121.5, 31.2], zoom: 11, label: "Shanghai" },
  { center: [113.3, 23.1], zoom: 10, label: "Guangzhou" },
  { center: [104.1, 30.6], zoom: 10, label: "Chengdu" }
];

// Reactive map that updates whenever `step` changes
const mapContainer = (() => {
  const container = document.createElement("div");
  container.style.height = "500px";

  const m = new maplibregl.Map({
    container,
    style: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    center: locations[step].center,
    zoom: locations[step].zoom,
    pitch: 40
  });

  return container;
})();
```

### Data-Driven Narrative with Observable Plot

```javascript
// Combine maps with inline charts for rich narratives
import * as Plot from "@observablehq/plot";

// Reactive chart that responds to the same step variable
const chart = Plot.plot({
  marks: [
    Plot.barY(populationData.filter(d => d.region === locations[step].label), {
      x: "year",
      y: "population",
      fill: "#41b6c4"
    }),
    Plot.ruleY([0])
  ],
  width: 600,
  height: 300,
  marginBottom: 40,
  x: { label: "Year" },
  y: { label: "Population (millions)" }
});
```

### Publishing Observable Framework Sites

```bash
# Create project
npm create @observablehq/framework my-story

# Develop locally
cd my-story
npm run dev

# Build static site
npm run build

# Deploy to Observable Cloud
npx observable deploy
```

> **Cross-reference:** For dashboard integration with Observable, see [dashboards.md](dashboards.md). For data loading pipelines, see [../data-analysis/](../data-analysis/).

---

## 6. StoryMapJS (Knight Lab)

[StoryMapJS](https://storymap.knightlab.com/) is a free, open-source tool from Northwestern University's Knight Lab. It requires zero code and is widely used in journalism and education.

### When to Use StoryMapJS

- Quick academic presentations with location-based narratives
- Journalism stories with fewer than 20 slides
- Projects where the team has no front-end developers
- Embedding in CMS platforms (WordPress, Squarespace)

### Embedding

```html
<iframe
  src="https://storymap.knightlab.com/edit/?id=YOUR_ID"
  width="100%"
  height="650"
  frameborder="0"
  title="Migration Story Map"
></iframe>
```

### Programmatic Creation via JSON

For automated or data-driven story maps, construct the JSON directly:

```json
{
  "storymap": {
    "slides": [
      {
        "type": "overview",
        "text": {
          "headline": "Silk Road Trade Routes",
          "text": "Following the ancient paths of commerce across Central Asia."
        },
        "media": {
          "url": "https://example.com/silk-road-overview.jpg",
          "caption": "Overview of major Silk Road routes",
          "credit": "Historical Atlas Project"
        }
      },
      {
        "location": {
          "lat": 39.9042,
          "lon": 116.4074,
          "zoom": 8
        },
        "text": {
          "headline": "Beijing: Eastern Terminus",
          "text": "The capital served as the starting point for diplomatic and trade missions heading west."
        },
        "media": {
          "url": "https://example.com/beijing-gate.jpg",
          "caption": "The Forbidden City, hub of imperial commerce"
        }
      },
      {
        "location": {
          "lat": 36.7538,
          "lon": 66.8970,
          "zoom": 7
        },
        "text": {
          "headline": "Balkh: Crossroads of Civilizations",
          "text": "One of the oldest cities in the world, where Greek, Persian, and Indian cultures converged."
        },
        "media": {
          "url": "https://example.com/balkh-ruins.jpg"
        }
      }
    ]
  }
}
```

### Self-Hosting StoryMapJS

```bash
git clone https://github.com/NUKnightLab/StoryMapJS.git
cd StoryMapJS
npm install
npm run build
# Deploy the /dist folder to your server
```

### Limitations

- No custom basemaps (limited to OpenStreetMap, Stamen, and a few others)
- No programmatic layer toggling
- Limited responsive design options
- No offline mode without self-hosting
- Media must be hosted externally (URLs only)

---

## 7. ArcGIS StoryMaps

The enterprise standard for narrative mapping. Part of the ArcGIS ecosystem with deep integration into ArcGIS Online and Enterprise.

### Layout Patterns

| Layout | Description | Best For |
|--------|-------------|----------|
| **Sidecar** | Scrolling text panel beside a sticky media panel | Deep dives, analysis narratives |
| **Guided Map Tour** | Numbered points on a map, each with a popup narrative | Field reports, site surveys |
| **Slideshow** | Full-screen media slides with text overlays | Presentations, executive summaries |
| **Swipe** | Two layers with a draggable divider | Before/after, temporal comparison |
| **Explorer** | Tabbed interface for self-directed exploration | Dashboards, multi-theme stories |

### Embedding

```html
<iframe
  src="https://storymaps.arcgis.com/stories/YOUR_STORY_ID"
  width="100%"
  height="600"
  frameborder="0"
  allowfullscreen
  allow="geolocation"
  title="Sea Level Rise Story Map"
></iframe>
```

### Data Integration Pattern

```
ArcGIS Online Feature Layer
        |
        v
StoryMap Sidecar Block
        |
        +--> Express Map (quick point/line/polygon)
        +--> Web Map (full symbology, popups, filters)
        +--> Web App (Instant App or Experience Builder)
        +--> Embedded Dashboard
```

### When to Choose ArcGIS StoryMaps

- Organization already has ArcGIS licenses
- Non-technical authors need WYSIWYG editing
- Compliance/governance requires Esri infrastructure
- Stories need to reference live, updating feature services

### Limitations

- Requires ArcGIS Online or Enterprise subscription (Creator license minimum)
- Custom CSS/JS is not supported inside the platform
- Export to standalone HTML is not available -- stories live on Esri's servers
- Performance depends on Esri's CDN and hosting infrastructure

> **Cross-reference:** For ArcGIS web map and feature layer configuration, see [../js-bindbox/](../js-bindbox/). For ArcGIS deployment patterns, see [../web-dev/deployment.md](../web-dev/deployment.md).

---

## 8. Mapbox Storytelling Template

[Mapbox Storytelling](https://github.com/mapbox/storytelling) is a config-driven template that produces polished scroll-driven narratives with minimal code.

### Setup

```bash
git clone https://github.com/mapbox/storytelling.git
cd storytelling
# Edit config.js, then open index.html
```

### Configuration

```javascript
// config.js
const config = {
  style: "mapbox://styles/mapbox/dark-v11",
  accessToken: "pk.YOUR_TOKEN",
  showMarkers: false,
  markerColor: "#3FB1CE",
  theme: "dark",                    // light | dark
  alignment: "left",                // left | right | center | full
  title: "Rising Sea Levels",
  subtitle: "How coastal cities are adapting to a changing climate",
  byline: "By Jane Doe, Environmental Desk",
  footer: "Source: NOAA, NASA. Published February 2026.",
  chapters: [
    {
      id: "intro",
      title: "The Global Picture",
      image: "images/sea-level-chart.png",
      description: "Sea levels have risen 21-24 cm since 1880. The rate is accelerating.",
      location: {
        center: [-40, 20],
        zoom: 2.5,
        pitch: 0,
        bearing: 0
      },
      mapAnimation: "flyTo",
      rotateAnimation: false,
      onChapterEnter: [
        { layer: "sea-level-global", opacity: 0.6 }
      ],
      onChapterExit: [
        { layer: "sea-level-global", opacity: 0 }
      ]
    },
    {
      id: "miami",
      title: "Miami: Ground Zero",
      description: "Built on porous limestone, Miami cannot simply build higher seawalls...",
      location: {
        center: [-80.19, 25.76],
        zoom: 11,
        pitch: 45,
        bearing: -10
      },
      mapAnimation: "flyTo",
      onChapterEnter: [
        { layer: "flood-zones-miami", opacity: 0.7 },
        { layer: "buildings-3d", opacity: 0.9 }
      ],
      onChapterExit: [
        { layer: "flood-zones-miami", opacity: 0 }
      ]
    },
    {
      id: "shanghai",
      title: "Shanghai: Engineering Response",
      description: "The Huangpu River flood barrier protects 24 million residents...",
      location: {
        center: [121.49, 31.23],
        zoom: 12,
        pitch: 50,
        bearing: 30
      },
      mapAnimation: "flyTo",
      onChapterEnter: [
        { layer: "flood-infrastructure", opacity: 1 }
      ],
      onChapterExit: [
        { layer: "flood-infrastructure", opacity: 0 }
      ]
    }
  ]
};
```

### Custom CSS Overrides

```css
/* Override default Mapbox storytelling styles */
#header {
  background: linear-gradient(to bottom, rgba(0,0,0,0.9), rgba(0,0,0,0.6));
  padding: 4rem 2rem;
}

#header h1 {
  font-family: "Playfair Display", serif;
  font-size: 3rem;
  letter-spacing: -0.02em;
}

.step {
  opacity: 0.3;
  transition: opacity 0.5s ease;
}

.step.active {
  opacity: 1;
}

/* Responsive: switch to center alignment on mobile */
@media (max-width: 768px) {
  #features {
    width: 90%;
    margin: 0 auto;
  }
}
```

---

## 9. Journalism & Data Storytelling

The visual language of data journalism has been refined by newsrooms including the New York Times, Washington Post, Reuters Graphics, The Guardian, and Bloomberg CityLab.

### Design Principles (Observed Across Top Newsrooms)

1. **Lead with geography, annotate with data.** The map is the canvas; annotations carry the argument.
2. **Minimal chrome.** Remove default map controls, legends, and UI that distract from the story.
3. **Deliberate color.** A single accent hue against a neutral basemap. Never a rainbow palette.
4. **Contextual labels.** Place names appear only where the narrative references them.
5. **Source attribution.** Always visible, always specific (dataset name + date, not just "various sources").
6. **Mobile-first.** Design for phone screens; enhance for desktop.

### NYT-Style Annotation Layer

```javascript
// Custom annotation markers for editorial callouts
function addAnnotation(map, { lngLat, text, anchor = "top", offset = [0, -10] }) {
  const el = document.createElement("div");
  el.className = "map-annotation";
  el.innerHTML = `
    <div class="annotation-line"></div>
    <div class="annotation-text">${text}</div>
  `;

  new maplibregl.Marker({ element: el, anchor })
    .setLngLat(lngLat)
    .setOffset(offset)
    .addTo(map);
}

// Usage
addAnnotation(map, {
  lngLat: [116.39, 39.91],
  text: "Tiananmen Square: 44 hectares,<br>capacity 500,000 people"
});
```

```css
.map-annotation {
  pointer-events: none;
  text-align: center;
}
.annotation-line {
  width: 1px;
  height: 30px;
  background: #333;
  margin: 0 auto;
}
.annotation-text {
  font: 13px/1.4 "Georgia", serif;
  color: #333;
  max-width: 160px;
  padding: 4px 0;
}
```

### Reuters Graphics Animated Frontline Pattern

```javascript
// Animated polygon boundary that updates by date
async function animateFrontline(map, geojsonUrl, dates) {
  const response = await fetch(geojsonUrl);
  const allData = await response.json();

  let currentIndex = 0;

  function updateFrame() {
    const date = dates[currentIndex];
    const filtered = {
      type: "FeatureCollection",
      features: allData.features.filter(f => f.properties.date === date)
    };

    map.getSource("frontline").setData(filtered);
    document.getElementById("date-label").textContent = date;

    currentIndex = (currentIndex + 1) % dates.length;
    setTimeout(updateFrame, 500);   // 500ms per frame
  }

  map.addSource("frontline", {
    type: "geojson",
    data: { type: "FeatureCollection", features: [] }
  });
  map.addLayer({
    id: "frontline-fill",
    type: "fill",
    source: "frontline",
    paint: { "fill-color": "#d32f2f", "fill-opacity": 0.4 }
  });
  map.addLayer({
    id: "frontline-line",
    type: "line",
    source: "frontline",
    paint: { "line-color": "#d32f2f", "line-width": 2 }
  });

  updateFrame();
}
```

### Small Multiples for Comparison

```javascript
function createSmallMultiples(data, months, containerId) {
  const container = document.getElementById(containerId);
  container.style.display = "grid";
  container.style.gridTemplateColumns = "repeat(4, 1fr)";
  container.style.gap = "8px";

  months.forEach((month) => {
    const mapDiv = document.createElement("div");
    mapDiv.id = `map-${month}`;
    mapDiv.style.height = "200px";
    mapDiv.style.position = "relative";
    container.appendChild(mapDiv);

    const label = document.createElement("div");
    label.textContent = month;
    label.style.cssText =
      "position:absolute;top:4px;left:4px;font:bold 12px sans-serif;z-index:1;color:#333;";
    mapDiv.appendChild(label);

    const map = new maplibregl.Map({
      container: mapDiv,
      style: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
      center: [116.4, 39.9],
      zoom: 9,
      interactive: false,
      attributionControl: false
    });

    map.on("load", () => {
      map.addSource("aqi", { type: "geojson", data: filterByMonth(data, month) });
      map.addLayer({
        id: "aqi-heat",
        type: "heatmap",
        source: "aqi",
        paint: { "heatmap-intensity": 1, "heatmap-radius": 20, "heatmap-opacity": 0.7 }
      });
    });
  });
}
```

### SOTA Newsroom Examples

| Publication | Project | Technique |
|-------------|---------|-----------|
| **NYT** | "How the Virus Got Out" | Animated flow maps + scroll triggers |
| **Washington Post** | Climate change series | Scrolly + 3D terrain + data overlays |
| **Reuters Graphics** | Ukraine conflict tracker | Daily-updated animated frontline maps |
| **The Guardian** | Australian bushfires | Satellite overlay + timeline scrubber |
| **Bloomberg CityLab** | Global housing crisis | Small multiples + annotation layers |
| **The Pudding** | Film dialogue mapped | Scrollytelling + D3 + custom maps |
| **ProPublica** | Sacrifice Zones | Environmental justice mapping + interviews |
| **The Economist** | Election cartograms | Hexbin maps + small multiples |
| **South China Morning Post** | Belt and Road | 3D globe + animated trade routes |
| **Financial Times** | Brexit impact maps | Choropleth + searchable constituencies |

> **Cross-reference:** For thematic map techniques referenced by newsrooms, see [thematic-maps.md](thematic-maps.md). For temporal animation, see [temporal-animation.md](temporal-animation.md).

---

## 10. Interactive Narratives

Beyond linear scrollytelling, interactive narratives give users agency in how they explore a story.

### Guided Tour with Forward/Back Controls

```javascript
class GuidedTour {
  constructor(map, steps) {
    this.map = map;
    this.steps = steps;
    this.current = 0;
    this.popup = new maplibregl.Popup({
      closeButton: false,
      closeOnClick: false,
      maxWidth: "320px"
    });
    this.renderControls();
    this.go(0);
  }

  renderControls() {
    const panel = document.createElement("div");
    panel.id = "tour-panel";
    panel.innerHTML = `
      <div id="tour-content"></div>
      <div id="tour-nav">
        <button id="tour-prev" disabled>Previous</button>
        <span id="tour-progress">1 / ${this.steps.length}</span>
        <button id="tour-next">Next</button>
      </div>
    `;
    document.body.appendChild(panel);

    document.getElementById("tour-prev").onclick = () => this.go(this.current - 1);
    document.getElementById("tour-next").onclick = () => this.go(this.current + 1);

    // Keyboard navigation
    document.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") this.go(this.current + 1);
      if (e.key === "ArrowLeft") this.go(this.current - 1);
    });
  }

  go(index) {
    if (index < 0 || index >= this.steps.length) return;
    this.current = index;
    const step = this.steps[index];

    this.map.flyTo({
      center: step.center,
      zoom: step.zoom,
      bearing: step.bearing || 0,
      pitch: step.pitch || 0,
      duration: 1800
    });

    this.popup
      .setLngLat(step.center)
      .setHTML(`<h3>${step.title}</h3><p>${step.description}</p>`)
      .addTo(this.map);

    document.getElementById("tour-content").innerHTML =
      `<h3>${step.title}</h3><p>${step.description}</p>`;
    document.getElementById("tour-progress").textContent =
      `${index + 1} / ${this.steps.length}`;
    document.getElementById("tour-prev").disabled = index === 0;
    document.getElementById("tour-next").disabled = index === this.steps.length - 1;

    if (step.onActivate) step.onActivate(this.map);
  }
}
```

### Branching Narratives (Choose Your Path)

```javascript
const storyGraph = {
  start: {
    title: "The Watershed",
    text: "This river basin supplies water to 3 million people. Where should we investigate?",
    center: [103.8, 30.6],
    zoom: 8,
    choices: [
      { label: "Upstream: The Dam", next: "dam" },
      { label: "Midstream: The City", next: "city" },
      { label: "Downstream: The Delta", next: "delta" }
    ]
  },
  dam: {
    title: "The Dam",
    text: "Built in 2008, this dam altered the entire flow regime. Sediment transport dropped 78%.",
    center: [103.2, 31.1],
    zoom: 13,
    layers: ["dam-infrastructure", "reservoir-extent"],
    choices: [
      { label: "Examine sediment impact", next: "sediment" },
      { label: "Return to overview", next: "start" }
    ]
  },
  city: {
    title: "The City",
    text: "Urban runoff introduces pollutants at 47 discharge points along a 12km stretch.",
    center: [104.1, 30.6],
    zoom: 12,
    layers: ["discharge-points", "water-quality"],
    choices: [
      { label: "View pollution hotspots", next: "pollution" },
      { label: "Return to overview", next: "start" }
    ]
  },
  delta: {
    title: "The Delta",
    text: "Subsidence and reduced sediment supply are causing the delta to shrink by 2.4 km/year.",
    center: [104.8, 29.5],
    zoom: 11,
    layers: ["delta-erosion", "land-use-change"],
    choices: [
      { label: "View historical shoreline", next: "shoreline" },
      { label: "Return to overview", next: "start" }
    ]
  }
};

function renderStoryNode(nodeId) {
  const node = storyGraph[nodeId];

  map.flyTo({ center: node.center, zoom: node.zoom, duration: 2000 });

  // Toggle layer visibility
  Object.values(storyGraph).forEach(n => {
    n.layers?.forEach(layerId => {
      const visible = node.layers?.includes(layerId) ? "visible" : "none";
      if (map.getLayer(layerId)) {
        map.setLayoutProperty(layerId, "visibility", visible);
      }
    });
  });

  const panel = document.getElementById("narrative-panel");
  panel.innerHTML = `
    <h2>${node.title}</h2>
    <p>${node.text}</p>
    <div class="choices">
      ${node.choices.map(c =>
        `<button onclick="renderStoryNode('${c.next}')">${c.label}</button>`
      ).join("")}
    </div>
  `;
}
```

### Progressive Disclosure (Layer Reveal System)

```javascript
class ProgressiveMap {
  constructor(map, layerGroups) {
    this.map = map;
    this.layerGroups = layerGroups;
    this.revealedCount = 0;
  }

  revealNext() {
    if (this.revealedCount >= this.layerGroups.length) return false;

    const group = this.layerGroups[this.revealedCount];
    group.layers.forEach(layerId => {
      this.map.setLayoutProperty(layerId, "visibility", "visible");
      this.animateOpacity(layerId, 0, group.opacity || 1, 800);
    });

    this.revealedCount++;
    return true;
  }

  animateOpacity(layerId, from, to, duration) {
    const start = performance.now();
    const prop = getOpacityProp(this.map.getLayer(layerId).type);

    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = t * (2 - t);
      this.map.setPaintProperty(layerId, prop, from + (to - from) * eased);
      if (t < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }
}

// Usage
const progressive = new ProgressiveMap(map, [
  { layers: ["basemap-labels"], opacity: 1 },
  { layers: ["terrain-contours"], opacity: 0.4 },
  { layers: ["rivers", "lakes"], opacity: 0.8 },
  { layers: ["population-heatmap"], opacity: 0.6 },
  { layers: ["infrastructure-points"], opacity: 1 }
]);
```

---

## 11. Multimedia Integration

Rich storytelling combines maps with video, audio, image sliders, and other media to create immersive experiences.

### Video Background Maps

Use HTML5 video as a background behind a map overlay, or embed video within scroll steps:

```html
<div class="video-map-container">
  <!-- Full-screen background video -->
  <video id="bg-video" autoplay muted loop playsinline
         style="position:fixed; top:0; left:0; width:100%; height:100%; object-fit:cover; z-index:-1;">
    <source src="drone-flyover.mp4" type="video/mp4" />
  </video>

  <!-- Semi-transparent map overlay -->
  <div id="map" style="position:fixed; top:0; left:0; width:100%; height:100%; opacity:0.7;"></div>

  <!-- Scrolling narrative -->
  <div id="story" style="position:relative; z-index:2;">
    <div class="step" data-step="0">
      <div class="step-content">
        <h2>The Landscape Before Development</h2>
        <p>Drone footage from 2019 shows unbroken forest canopy across the watershed.</p>
      </div>
    </div>
  </div>
</div>
```

### Scroll-Synced Video Playback

```javascript
// Sync video playback position to scroll progress
function syncVideoToScroll(videoElement, scrollContainer) {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const rect = scrollContainer.getBoundingClientRect();
          const scrollFraction = Math.max(0, Math.min(1,
            -rect.top / (rect.height - window.innerHeight)
          ));
          videoElement.currentTime = scrollFraction * videoElement.duration;
        }
      });
    },
    { threshold: Array.from({ length: 100 }, (_, i) => i / 100) }
  );

  observer.observe(scrollContainer);
}

// Usage
const video = document.getElementById("bg-video");
video.pause(); // disable autoplay for scroll-driven control
const container = document.getElementById("scrolly");
syncVideoToScroll(video, container);
```

### Before/After Image Comparison with Juxtapose.js

```html
<link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">

<div id="satellite-comparison" style="width:100%; max-width:800px;">
  <div class="juxtapose"
       data-startingposition="50"
       data-showlabels="true"
       data-showcredits="true"
       data-animate="true"
       data-makeresponsive="true">
    <img src="before-2015.jpg"
         data-label="2015"
         data-credit="Sentinel-2, ESA" />
    <img src="after-2024.jpg"
         data-label="2024"
         data-credit="Sentinel-2, ESA" />
  </div>
</div>

<script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
```

### Map-Based Swipe Comparison (Dual MapLibre Instances)

```javascript
function createSwipeComparison(containerId, beforeStyle, afterStyle, center, zoom) {
  const container = document.getElementById(containerId);
  container.style.position = "relative";
  container.style.overflow = "hidden";
  container.style.height = "500px";

  // Create two map containers
  ["before", "after"].forEach(id => {
    const div = document.createElement("div");
    div.id = `${id}-map`;
    div.style.cssText = "position:absolute;top:0;left:0;width:100%;height:100%;";
    container.appendChild(div);
  });

  const beforeMap = new maplibregl.Map({
    container: "before-map",
    style: beforeStyle,
    center, zoom
  });

  const afterMap = new maplibregl.Map({
    container: "after-map",
    style: afterStyle,
    center, zoom
  });

  // Sync camera
  beforeMap.on("move", () => {
    afterMap.jumpTo({
      center: beforeMap.getCenter(),
      zoom: beforeMap.getZoom(),
      bearing: beforeMap.getBearing(),
      pitch: beforeMap.getPitch()
    });
  });

  // Swipe slider
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = 0;
  slider.max = 100;
  slider.value = 50;
  slider.style.cssText = `
    position:absolute; bottom:20px; left:10%; width:80%; z-index:10;
    appearance:none; height:4px; background:#fff; border-radius:2px; cursor:pointer;
  `;
  container.appendChild(slider);

  // Labels
  ["before", "after"].forEach((label, i) => {
    const el = document.createElement("div");
    el.textContent = label.toUpperCase();
    el.style.cssText = `
      position:absolute; top:10px; ${i === 0 ? "left" : "right"}:10px;
      background:rgba(0,0,0,0.7); color:#fff; padding:4px 8px;
      font:bold 11px sans-serif; border-radius:3px; z-index:10;
    `;
    container.appendChild(el);
  });

  slider.addEventListener("input", (e) => {
    document.getElementById("after-map").style.clipPath =
      `inset(0 0 0 ${e.target.value}%)`;
  });

  // Initial clip
  document.getElementById("after-map").style.clipPath = "inset(0 0 0 50%)";
}
```

### Audio Narration Synchronized to Scroll

```javascript
class AudioNarrator {
  constructor(audioSrc, chapters) {
    this.audio = new Audio(audioSrc);
    this.chapters = chapters; // [{ startTime: 0, stepIndex: 0 }, ...]
    this.audio.preload = "auto";
  }

  syncToStep(stepIndex) {
    const chapter = this.chapters.find(c => c.stepIndex === stepIndex);
    if (!chapter) return;

    this.audio.currentTime = chapter.startTime;
    this.audio.play().catch(() => {
      // Autoplay blocked -- show play button
      document.getElementById("audio-play-btn").style.display = "block";
    });
  }

  pause() {
    this.audio.pause();
  }
}

// Usage with Scrollama
const narrator = new AudioNarrator("narration.mp3", [
  { stepIndex: 0, startTime: 0 },
  { stepIndex: 1, startTime: 45 },
  { stepIndex: 2, startTime: 92 },
  { stepIndex: 3, startTime: 148 }
]);

scroller.onStepEnter(({ index }) => {
  narrator.syncToStep(index);
});

scroller.onStepExit(() => {
  narrator.pause();
});
```

### Multimedia Tools Summary

| Tool | Media Type | Integration |
|------|-----------|-------------|
| **Juxtapose.js** | Before/after images | Embed via HTML/JS |
| **Howler.js** | Audio playback | npm package, robust mobile support |
| **Plyr** | Video player | Customizable HTML5 video UI |
| **Lottie** | Vector animations | After Effects -> JSON -> web |
| **Theatre.js** | Keyframe animation | React/vanilla JS, timeline editor |
| **GSAP ScrollTrigger** | Scroll-linked animation | Plugin for GSAP animation library |

> **Cross-reference:** For video map export and cinematic rendering, see [temporal-animation.md](temporal-animation.md). For 3D scene animation, see [3d-visualization.md](3d-visualization.md).

---

## 12. Scientific Communication

Maps in research papers, conference presentations, and grant proposals require different treatment than journalistic pieces: emphasis on reproducibility, precise legends, projection metadata, and figure-quality exports.

### Research Paper Figure Animation

Turn a static figure into an animated supplement that reviewers and readers can explore:

```javascript
// Exportable figure with exact dimensions for journal submission
function createJournalFigure(map, {
  width = 180,          // mm (single column for Nature/Science)
  height = 120,         // mm
  dpi = 300,
  title = "",
  legend = null,
  scalebar = true
} = {}) {
  const pxWidth = Math.round(width * dpi / 25.4);
  const pxHeight = Math.round(height * dpi / 25.4);

  // Create off-screen canvas at target resolution
  const canvas = document.createElement("canvas");
  canvas.width = pxWidth;
  canvas.height = pxHeight;
  const ctx = canvas.getContext("2d");

  // Draw map canvas
  const mapCanvas = map.getCanvas();
  ctx.drawImage(mapCanvas, 0, 0, pxWidth, pxHeight);

  // Draw title
  if (title) {
    ctx.font = `bold ${Math.round(12 * dpi / 72)}px Arial`;
    ctx.fillStyle = "#000";
    ctx.fillText(title, 20, Math.round(20 * dpi / 72));
  }

  // Draw scale bar
  if (scalebar) {
    drawScaleBar(ctx, map, pxWidth, pxHeight, dpi);
  }

  // Export
  return canvas.toDataURL("image/png");
}

function drawScaleBar(ctx, map, canvasWidth, canvasHeight, dpi) {
  const bounds = map.getBounds();
  const metersPerPixel = bounds.getNorthEast().distanceTo(bounds.getSouthWest()) /
    Math.sqrt(canvasWidth ** 2 + canvasHeight ** 2);

  // Pick nice round distance
  const targetWidthPx = canvasWidth * 0.2;
  const targetMeters = metersPerPixel * targetWidthPx;
  const niceDistance = getNiceRoundNumber(targetMeters);
  const barWidth = niceDistance / metersPerPixel;

  const x = canvasWidth - barWidth - 40;
  const y = canvasHeight - 40;

  ctx.fillStyle = "#000";
  ctx.fillRect(x, y, barWidth, 4);
  ctx.font = `${Math.round(9 * dpi / 72)}px Arial`;
  ctx.fillText(`${niceDistance >= 1000 ? niceDistance / 1000 + " km" : niceDistance + " m"}`,
    x + barWidth / 2 - 20, y - 6);
}
```

### Conference Presentation with reveal.js + Live Maps

```html
<!-- Slide with embedded interactive map -->
<section data-background-iframe="research-map.html"
         data-background-interactive>
  <div style="position:absolute; bottom:20px; left:20px;
              background:rgba(255,255,255,0.9); padding:12px 16px;
              border-radius:4px; font-size:0.8em;">
    <strong>Fig. 3:</strong> Spatial distribution of PM2.5 concentrations.
    Pan and zoom to explore.
  </div>
</section>
```

### Animated Figure Supplement (GIF/MP4 Export)

```javascript
// Record map animation as frame sequence for GIF/video
async function recordMapAnimation(map, frames, fps = 10) {
  const canvas = map.getCanvas();
  const images = [];

  for (let i = 0; i < frames.length; i++) {
    const frame = frames[i];

    // Apply frame state
    map.jumpTo({
      center: frame.center,
      zoom: frame.zoom,
      bearing: frame.bearing || 0,
      pitch: frame.pitch || 0
    });

    // Wait for tiles to load
    await new Promise(resolve => {
      if (map.loaded()) resolve();
      else map.once("idle", resolve);
    });

    // Capture frame
    images.push(canvas.toDataURL("image/png"));

    // Update progress
    console.log(`Frame ${i + 1}/${frames.length}`);
  }

  return images; // Array of data URLs -- feed to gif.js or ffmpeg.wasm
}
```

### Tools for Scientific Map Communication

| Tool | Use Case | Output |
|------|----------|--------|
| **QGIS Print Layout** | Publication-quality static figures | PNG, SVG, PDF |
| **matplotlib + cartopy** | Python-based figure generation | PNG, SVG, PDF |
| **R ggplot2 + sf** | Statistical maps with ggplot grammar | PNG, SVG, PDF |
| **MapLibre + html2canvas** | Web-based figure export | PNG |
| **Inkscape / Illustrator** | Post-processing vector map figures | SVG, AI, EPS |
| **reveal.js** | Live map presentations | HTML (self-contained) |
| **Marp** | Markdown-based slides with map images | HTML, PDF, PPTX |

> **Cross-reference:** For scientific visualization techniques (color scales, projections, legends), see [scientific-visualization.md](scientific-visualization.md). For cartographic design, see [cartography-design.md](cartography-design.md).

---

## 13. Client Deliverables -- 甲方交付

For GIS professionals in consulting, government, and engineering, story maps serve as deliverables that communicate project findings to stakeholders (甲方/clients) more effectively than static PDFs.

### Project Report as Interactive Story Map

**Structure template for a typical land-use planning deliverable:**

```
Chapter 1: Project Overview          --> National/regional context, project boundary
Chapter 2: Current Conditions        --> Existing land use, zoning, infrastructure
Chapter 3: Analysis Results          --> Suitability analysis, constraint maps
Chapter 4: Scenario Comparison       --> Side-by-side scenarios with swipe tool
Chapter 5: Recommended Plan          --> Proposed layout with 3D visualization
Chapter 6: Implementation Timeline   --> Phased development, cost estimates
Appendix:  Data Sources & Methods    --> Metadata, methodology notes
```

### Bilingual Story Map (Chinese/English)

```javascript
// Language switcher for bilingual deliverables
const i18n = {
  zh: {
    title: "城市更新项目分析报告",
    chapter1: "项目概况",
    chapter2: "现状分析",
    chapter3: "规划方案",
    overview: "本项目位于市中心核心区，总面积约 2.3 平方公里。",
    dataSource: "数据来源：自然资源局，2025年"
  },
  en: {
    title: "Urban Renewal Project Analysis Report",
    chapter1: "Project Overview",
    chapter2: "Current Conditions",
    chapter3: "Proposed Plan",
    overview: "The project site covers approximately 2.3 km2 in the central business district.",
    dataSource: "Data source: Bureau of Natural Resources, 2025"
  }
};

let currentLang = "zh";

function setLanguage(lang) {
  currentLang = lang;
  const t = i18n[lang];

  document.getElementById("story-title").textContent = t.title;
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.dataset.i18n;
    if (t[key]) el.textContent = t[key];
  });

  // Update active language button
  document.querySelectorAll(".lang-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });
}
```

### Stakeholder Engagement Patterns

| Audience | Story Pattern | Key Features |
|----------|--------------|--------------|
| **Government officials (领导)** | Executive summary, 3-5 slides max | Large text, clear conclusions, 3D renders |
| **Technical reviewers (专家评审)** | Full analysis with methodology | Data sources, scale bars, coordinate refs |
| **Public consultation (公众参与)** | Simple guided tour | Plain language, before/after, feedback form |
| **Investment committee (投资方)** | ROI-focused scenarios | Cost overlays, phased development, area stats |

### Print-Friendly Export

```css
/* Print stylesheet for story map deliverables */
@media print {
  .scrolly__map {
    position: relative !important;   /* unstick the map */
    height: 400px !important;
    page-break-after: always;
  }

  .scrolly__step {
    min-height: auto !important;
    page-break-inside: avoid;
  }

  .step-content {
    background: white !important;
    box-shadow: none !important;
    border: 1px solid #ccc !important;
  }

  /* Hide interactive controls */
  .maplibregl-ctrl-group,
  .tour-nav,
  .lang-switcher {
    display: none !important;
  }

  /* Force map image capture */
  .map-print-placeholder {
    display: block !important;
  }
}
```

### Offline Deliverable Package

```bash
# Package a self-contained story map for offline delivery
# Useful when clients have restricted internet access

# 1. Build the story site
npm run build

# 2. Download tile cache for the project area
# Using mbutil or tippecanoe-generated mbtiles
node scripts/cache-tiles.js \
  --bounds "120.1,30.2,121.5,31.5" \
  --zoom "1-16" \
  --style "style.json" \
  --output "tiles/"

# 3. Bundle into a single directory
cp -r dist/ deliverable/
cp -r tiles/ deliverable/tiles/
# Update style.json to point to local tile paths

# 4. Package
zip -r "项目交付物_$(date +%Y%m%d).zip" deliverable/

# Client opens deliverable/index.html in any modern browser
```

> **Cross-reference:** For deployment strategies, see [../web-dev/deployment.md](../web-dev/deployment.md). For backend tile serving, see [../web-dev/backend-services.md](../web-dev/backend-services.md).

---

## 14. Accessibility in Storytelling

Accessible story maps ensure that all users -- including those with motor, visual, or cognitive disabilities -- can consume the narrative.

### Reduced Motion

```css
/* Respect user preference for reduced motion */
@media (prefers-reduced-motion: reduce) {
  .scrolly__map {
    /* Disable fly-to animations */
  }

  .step-content {
    transition: none !important;
  }

  .map-annotation {
    animation: none !important;
  }
}
```

```javascript
// Check preference in JavaScript
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function navigateToChapter(chapter) {
  if (prefersReducedMotion) {
    // Instant jump instead of animated fly-to
    map.jumpTo({
      center: chapter.center,
      zoom: chapter.zoom,
      bearing: chapter.bearing,
      pitch: chapter.pitch
    });
  } else {
    map.flyTo({
      center: chapter.center,
      zoom: chapter.zoom,
      bearing: chapter.bearing,
      pitch: chapter.pitch,
      duration: 2000,
      essential: true
    });
  }
}
```

### Keyboard Navigation

```javascript
// Full keyboard navigation for story maps
class AccessibleStoryMap {
  constructor(map, chapters) {
    this.map = map;
    this.chapters = chapters;
    this.currentIndex = 0;

    // Keyboard bindings
    document.addEventListener("keydown", (e) => {
      switch (e.key) {
        case "ArrowDown":
        case "ArrowRight":
        case " ":             // spacebar
          e.preventDefault();
          this.next();
          break;
        case "ArrowUp":
        case "ArrowLeft":
          e.preventDefault();
          this.previous();
          break;
        case "Home":
          e.preventDefault();
          this.goTo(0);
          break;
        case "End":
          e.preventDefault();
          this.goTo(this.chapters.length - 1);
          break;
        case "Escape":
          this.announceCurrentStep();
          break;
      }
    });
  }

  next() {
    if (this.currentIndex < this.chapters.length - 1) {
      this.goTo(this.currentIndex + 1);
    }
  }

  previous() {
    if (this.currentIndex > 0) {
      this.goTo(this.currentIndex - 1);
    }
  }

  goTo(index) {
    this.currentIndex = index;
    const ch = this.chapters[index];

    navigateToChapter(ch);

    // Update ARIA attributes
    document.querySelectorAll(".step").forEach((el, i) => {
      el.setAttribute("aria-current", i === index ? "step" : "false");
      el.setAttribute("tabindex", i === index ? "0" : "-1");
    });

    // Focus the active step
    document.querySelectorAll(".step")[index]?.focus();

    this.announceCurrentStep();
  }

  announceCurrentStep() {
    const ch = this.chapters[this.currentIndex];
    const announcement = `Step ${this.currentIndex + 1} of ${this.chapters.length}: ${ch.title}. ${ch.description}`;

    // Use ARIA live region for screen reader announcement
    let liveRegion = document.getElementById("story-announcer");
    if (!liveRegion) {
      liveRegion = document.createElement("div");
      liveRegion.id = "story-announcer";
      liveRegion.setAttribute("role", "status");
      liveRegion.setAttribute("aria-live", "polite");
      liveRegion.setAttribute("aria-atomic", "true");
      liveRegion.className = "sr-only";    // visually hidden
      document.body.appendChild(liveRegion);
    }
    liveRegion.textContent = announcement;
  }
}
```

### ARIA Markup for Story Steps

```html
<div id="story" role="region" aria-label="Story narrative">
  <div class="step"
       role="article"
       aria-label="Chapter 1: The Overview"
       aria-current="step"
       tabindex="0"
       data-step="0">
    <div class="step-content">
      <h2 id="step-0-title">Chapter 1: The Overview</h2>
      <p>We begin with a continental view of population density.</p>
      <p class="sr-only">
        The map shows East Asia centered on China at zoom level 3.5.
        A population density heatmap is visible, with darker shading
        indicating higher population concentration in the east.
      </p>
    </div>
  </div>
</div>

<!-- Screen reader only class -->
<style>
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}
</style>
```

### Accessibility Checklist for Story Maps

| Requirement | Implementation | WCAG Level |
|-------------|---------------|------------|
| Keyboard-navigable steps | Tab/Arrow key support | A |
| Screen reader narration | ARIA live regions + alt text | A |
| Reduced motion support | `prefers-reduced-motion` media query | AA |
| Color contrast | 4.5:1 ratio for text, 3:1 for UI | AA |
| Focus indicators | Visible focus ring on active step | AA |
| Skip navigation | "Skip to main content" link | A |
| Text alternatives | Alt text for all map states | A |
| Resizable text | Supports up to 200% zoom | AA |
| No seizure triggers | Avoid flashing animations > 3/sec | A |

> **Cross-reference:** For accessible color palettes, see [cartography-design.md](cartography-design.md). For accessible dashboard patterns, see [dashboards.md](dashboards.md).

---

## 15. Performance Optimization

Story maps load many assets -- map tiles, images, video, fonts, data. Optimization is critical for mobile and low-bandwidth users.

### Lazy Loading Map Instances

```javascript
// Only initialize maps when they scroll into view
class LazyMapLoader {
  constructor() {
    this.observer = new IntersectionObserver(
      (entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            this.initMap(entry.target);
            this.observer.unobserve(entry.target);
          }
        });
      },
      { rootMargin: "200px 0px" }  // start loading 200px before visible
    );

    document.querySelectorAll("[data-lazy-map]").forEach(el => {
      this.observer.observe(el);
    });
  }

  initMap(container) {
    const config = JSON.parse(container.dataset.lazyMap);
    const map = new maplibregl.Map({
      container,
      style: config.style,
      center: config.center,
      zoom: config.zoom,
      interactive: config.interactive ?? true
    });

    container.mapInstance = map;
  }
}

// Usage
// <div data-lazy-map='{"style":"...","center":[116.4,39.9],"zoom":10}'
//      style="height:400px;"></div>
new LazyMapLoader();
```

### Asset Preloading Strategy

```javascript
// Preload assets for upcoming chapters
function preloadChapterAssets(chapters, currentIndex, lookahead = 2) {
  for (let i = currentIndex + 1; i <= Math.min(currentIndex + lookahead, chapters.length - 1); i++) {
    const ch = chapters[i];

    // Preload images
    if (ch.image) {
      const img = new Image();
      img.src = ch.image;
    }

    // Preload GeoJSON data
    if (ch.dataUrl) {
      fetch(ch.dataUrl).then(r => r.json()).then(data => {
        ch._cachedData = data;
      });
    }

    // Warm tile cache by briefly flying to target location
    // (tiles load but map state reverts immediately)
    if (ch.center && ch.zoom) {
      const tempMap = document.createElement("div");
      tempMap.style.cssText = "position:absolute;width:1px;height:1px;opacity:0;pointer-events:none;";
      document.body.appendChild(tempMap);

      const preloadMap = new maplibregl.Map({
        container: tempMap,
        style: map.getStyle(),
        center: ch.center,
        zoom: ch.zoom
      });

      preloadMap.once("idle", () => {
        preloadMap.remove();
        tempMap.remove();
      });
    }
  }
}
```

### Image Optimization

```html
<!-- Use responsive images with srcset -->
<img srcset="figure-400w.webp 400w,
             figure-800w.webp 800w,
             figure-1200w.webp 1200w"
     sizes="(max-width: 600px) 400px,
            (max-width: 1000px) 800px,
            1200px"
     src="figure-800w.webp"
     alt="Land use change 2015-2024"
     loading="lazy"
     decoding="async" />
```

### Mobile Optimization Checklist

| Optimization | Technique | Impact |
|-------------|-----------|--------|
| **Reduce tile quality** | Lower DPI tiles on mobile (256px instead of 512px) | 60-75% bandwidth reduction |
| **Disable 3D** | Set `pitch: 0` on mobile, remove fill-extrusions | Major GPU savings |
| **Simplify geometry** | Use tippecanoe `--simplification` for vector tiles | Smaller tile sizes |
| **Defer non-critical layers** | Load annotation/label layers after initial paint | Faster first paint |
| **Compress GeoJSON** | Use TopoJSON or FlatGeobuf instead of GeoJSON | 50-90% size reduction |
| **Service Worker caching** | Cache tiles and assets for repeat visits | Instant reload |

### Performance Budget

```
Target: First Contentful Paint < 2s on 4G mobile

Budget allocation:
  HTML + CSS + JS bundle:    < 200 KB (gzipped)
  Map tiles (initial view):  < 500 KB
  Images (above the fold):   < 300 KB
  Fonts:                     < 100 KB
  Total initial load:        < 1.1 MB
```

> **Cross-reference:** For web performance optimization patterns, see [../web-dev/performance.md](../web-dev/performance.md). For tile optimization, see [../web-dev/backend-services.md](../web-dev/backend-services.md).

---

## 16. Publishing & Hosting

Getting story maps from development to production.

### Static Site Generators

Story maps are inherently static content (HTML + JS + CSS + assets). Static site generators provide build tooling, templating, and optimized output.

| Generator | Language | Best For |
|-----------|----------|----------|
| **Astro** | JS/TS | Island architecture, partial hydration, great for map-heavy pages |
| **Next.js (static export)** | React | Teams already using React + deck.gl |
| **Eleventy (11ty)** | JS | Minimal, fast builds, Markdown-first |
| **Hugo** | Go | Fastest builds, good for multi-page story collections |
| **Quarto** | R/Python | Academic publishing, Jupyter/R Markdown integration |
| **Observable Framework** | JS | Data-driven stories with reactive charts |
| **VitePress** | Vue | Documentation-style story archives |

### GitHub Pages Deployment

```yaml
# .github/workflows/deploy.yml
name: Deploy Story Map

on:
  push:
    branches: [main]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install and build
        run: |
          npm ci
          npm run build

      - name: Upload artifact
        uses: actions/upload-pages-artifact@v3
        with:
          path: dist/

      - name: Deploy to GitHub Pages
        uses: actions/deploy-pages@v4
```

### Vercel Deployment

```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "framework": null,
  "rewrites": [
    { "source": "/(.*)", "destination": "/index.html" }
  ],
  "headers": [
    {
      "source": "/tiles/(.*)",
      "headers": [
        { "key": "Cache-Control", "value": "public, max-age=31536000, immutable" }
      ]
    }
  ]
}
```

### Netlify Deployment

```toml
# netlify.toml
[build]
  command = "npm run build"
  publish = "dist"

[[headers]]
  for = "/tiles/*"
  [headers.values]
    Cache-Control = "public, max-age=31536000, immutable"

[[headers]]
  for = "/*.js"
  [headers.values]
    Cache-Control = "public, max-age=86400"
```

### CMS Integration

For organizations that need non-developers to update story content:

```javascript
// Headless CMS integration pattern
// Story content from Strapi, Contentful, Sanity, or similar

async function loadStoryFromCMS(storyId) {
  // Example: Strapi API
  const response = await fetch(`https://cms.example.com/api/stories/${storyId}?populate=chapters`);
  const { data } = await response.json();

  return {
    title: data.attributes.title,
    subtitle: data.attributes.subtitle,
    chapters: data.attributes.chapters.map(ch => ({
      title: ch.title,
      description: ch.description,
      center: [ch.longitude, ch.latitude],
      zoom: ch.zoom,
      bearing: ch.bearing || 0,
      pitch: ch.pitch || 0,
      image: ch.image?.data?.attributes?.url,
      layers: ch.layers || []
    }))
  };
}

// Initialize story from CMS data
loadStoryFromCMS("sea-level-rise").then(story => {
  document.getElementById("story-title").textContent = story.title;
  initScrollyMap(map, story.chapters);
});
```

### Embedding in External Sites

```html
<!-- WordPress shortcode (via custom plugin or raw HTML block) -->
<iframe
  src="https://yourdomain.com/story-map/"
  width="100%"
  height="700"
  frameborder="0"
  allow="fullscreen; geolocation"
  loading="lazy"
  title="Interactive Story Map: Sea Level Rise"
></iframe>

<!-- Responsive embed wrapper -->
<style>
.story-embed {
  position: relative;
  padding-bottom: 56.25%;  /* 16:9 aspect ratio */
  height: 0;
  overflow: hidden;
}
.story-embed iframe {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: 0;
}
</style>
<div class="story-embed">
  <iframe src="https://yourdomain.com/story-map/" title="Story Map"></iframe>
</div>
```

> **Cross-reference:** For full deployment architecture, see [../web-dev/deployment.md](../web-dev/deployment.md). For fullstack patterns, see [../web-dev/fullstack-architecture.md](../web-dev/fullstack-architecture.md).

---

## 17. Tool Comparison Matrix

### Comprehensive Feature Comparison

| Feature | Scrollama + MapLibre | deck.gl + React | Mapbox Storytelling | ArcGIS StoryMaps | StoryMapJS | Observable Framework | Felt | Google Earth Studio |
|---------|---------------------|----------------|-------------------|-----------------|------------|---------------------|------|-------------------|
| **License** | MIT / BSD | MIT | BSD | Proprietary | MPL 2.0 | ISC | Proprietary | Free (Google) |
| **Cost** | Free | Free | Free + token | $$ (ArcGIS) | Free | Free / $$ | Freemium | Free |
| **Code Required** | Yes (JS) | Yes (React) | Minimal (config) | No | No | Some (JS) | No | No |
| **Custom Basemaps** | Any | Any | Mapbox styles | ArcGIS styles | Limited | Any | Yes | Google only |
| **3D Support** | Terrain + extrusions | Full 3D layers | Terrain + extrusions | Basic 3D | No | Via libraries | No | Full 3D globe |
| **Large Datasets** | Medium | Excellent (GPU) | Medium | Medium | Small | Medium | Small | N/A |
| **Scroll-Driven** | Native | Via react-scrollama | Built-in | Sidecar layout | No | Custom | No | N/A |
| **Collaboration** | Via Git | Via Git | Via Git | Org-level | No | Multiplayer | Real-time | No |
| **Mobile Support** | Manual responsive | Manual responsive | Good | Excellent | Good | Good | Good | Output video |
| **Offline Capable** | Yes (self-host) | Yes (self-host) | Yes (self-host) | No | Self-host | Yes (build) | No | Export frames |
| **Accessibility** | Manual (full control) | Manual (full control) | Manual | Good (WCAG) | Basic | Manual | Basic | N/A |
| **Output Format** | HTML/JS | HTML/JS | HTML/JS | Hosted URL | Hosted/JSON | HTML | URL/PNG | Image sequence |
| **Data Integration** | Any API/format | Any API/format | Mapbox sources | ArcGIS services | Images/media | Any API | CSV/GeoJSON/SHP | KML |
| **Learning Curve** | Medium | High | Low | Low | Very Low | Medium | Very Low | Medium |
| **Best For** | Custom journalism | Data-intensive 3D | Quick prototypes | Enterprise orgs | Academic/quick | Data-reactive docs | Team collaboration | Cinematic video |

### Decision Flowchart

```
START: What are you building?
  |
  +--> "A quick story for a class or blog post"
  |      --> StoryMapJS or Felt
  |
  +--> "An enterprise deliverable for a client with ArcGIS"
  |      --> ArcGIS StoryMaps
  |
  +--> "A data journalism piece with custom animations"
  |      |
  |      +--> "Need to render millions of points?"
  |      |      --> deck.gl + React + Scrollama
  |      |
  |      +--> "Standard map layers, editorial control?"
  |             --> Scrollama + MapLibre (custom build)
  |             --> or Mapbox Storytelling (quick start)
  |
  +--> "A data-driven reactive document with charts"
  |      --> Observable Framework
  |
  +--> "A cinematic flyover video for social media / TV"
  |      --> Google Earth Studio
  |
  +--> "A collaborative team map for stakeholder review"
         --> Felt
```

### Ecosystem Links

| Resource | URL |
|----------|-----|
| Scrollama | [github.com/russellsamora/scrollama](https://github.com/russellsamora/scrollama) |
| react-scrollama | [github.com/jsonkao/react-scrollama](https://github.com/jsonkao/react-scrollama) |
| MapLibre GL JS | [maplibre.org](https://maplibre.org/) |
| deck.gl | [deck.gl](https://deck.gl/) |
| Mapbox Storytelling | [github.com/mapbox/storytelling](https://github.com/mapbox/storytelling) |
| ArcGIS StoryMaps | [storymaps.arcgis.com](https://storymaps.arcgis.com/) |
| StoryMapJS | [storymap.knightlab.com](https://storymap.knightlab.com/) |
| TimelineJS | [timeline.knightlab.com](https://timeline.knightlab.com/) |
| Juxtapose.js | [juxtapose.knightlab.com](https://juxtapose.knightlab.com/) |
| Observable Framework | [observablehq.com/framework](https://observablehq.com/framework) |
| Felt | [felt.com](https://felt.com/) |
| Google Earth Studio | [earth.google.com/studio](https://earth.google.com/studio) |
| reveal.js | [revealjs.com](https://revealjs.com/) |
| GSAP ScrollTrigger | [gsap.com/docs/v3/Plugins/ScrollTrigger](https://gsap.com/docs/v3/Plugins/ScrollTrigger/) |
| Lottie | [airbnb.io/lottie](https://airbnb.io/lottie/) |
| Theatre.js | [theatrejs.com](https://www.theatrejs.com/) |

---

## Cross-Reference Index

| Topic | Page |
|-------|------|
| MapLibre GL JS fundamentals | [../js-bindbox/](../js-bindbox/) |
| 3D visualization (terrain, extrusions, point clouds) | [3d-visualization.md](3d-visualization.md) |
| Temporal animation and time-series | [temporal-animation.md](temporal-animation.md) |
| Thematic map design (choropleth, heatmap, flow) | [thematic-maps.md](thematic-maps.md) |
| Dashboard and interactive app frameworks | [dashboards.md](dashboards.md) |
| Cartographic design and color theory | [cartography-design.md](cartography-design.md) |
| Scientific visualization techniques | [scientific-visualization.md](scientific-visualization.md) |
| Network and graph visualization | [network-graph-visualization.md](network-graph-visualization.md) |
| AI/ML visualization | [ai-ml-visualization.md](ai-ml-visualization.md) |
| Frontend framework integration | [../web-dev/frontend-integration.md](../web-dev/frontend-integration.md) |
| Deployment and hosting | [../web-dev/deployment.md](../web-dev/deployment.md) |
| Backend services and tile serving | [../web-dev/backend-services.md](../web-dev/backend-services.md) |
| Performance optimization | [../web-dev/performance.md](../web-dev/performance.md) |
| Data sources for storytelling | [../data-sources/](../data-sources/) |
| Data analysis pipelines | [../data-analysis/](../data-analysis/) |
