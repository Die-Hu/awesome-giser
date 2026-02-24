# Real-Time, Offline & Advanced Patterns

> **Quick Picks**
> - **SOTA**: WebSocket + MapLibre for live vehicle/fleet tracking
> - **Free Best**: Service Worker + PMTiles for fully offline maps
> - **Fastest Setup**: MapLibre + SSE for simple one-way live data streams

---

## Table of Contents

1. [Real-Time Data Visualization](#real-time-data-visualization)
2. [Offline Maps & PWA](#offline-maps--pwa)
3. [Collaborative & Multi-User Maps](#collaborative--multi-user-maps)
4. [Testing Geo Web Apps](#testing-geo-web-apps)
5. [Accessibility (a11y) for Maps](#accessibility-a11y-for-maps)
6. [Internationalization (i18n)](#internationalization-i18n)
7. [Security](#security)
8. [Deployment Patterns](#deployment-patterns)
9. [Monitoring in Production](#monitoring-in-production)
10. [Advanced Dark Arts](#advanced-dark-arts)

---

## Real-Time Data Visualization

### WebSocket Patterns

WebSockets give you a persistent, full-duplex channel. The three dominant geo use cases:

| Use Case | Pattern | Notes |
|---|---|---|
| Vehicle/fleet tracking | WebSocket → update GeoJSON source → smooth animation | Most common real-time geo pattern |
| IoT sensor dashboards | WebSocket → update feature properties → data-driven styling | Use `setFeatureState()` for perf |
| Collaborative mapping | WebSocket → sync edits between users | Layer CRDT (Yjs) on top for conflicts |

### Complete Vehicle Tracking Example

```javascript
// WebSocket → MapLibre real-time vehicle tracking with smooth interpolation
// Handles: connection, message parsing, smooth animation, trail rendering, direction indicator

class VehicleTracker {
  constructor(map) {
    this.map = map;
    this.vehicles = new Map();   // vehicleId → { lng, lat, bearing, trail: [] }
    this.ws = null;
    this.animFrameId = null;
    this._initSources();
    this._connectWS();
    this._startRenderLoop();
  }

  _initSources() {
    // Vehicle point source
    this.map.addSource('vehicles', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    });

    // Trail line source
    this.map.addSource('vehicle-trails', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    });

    // Render layers
    this.map.addLayer({
      id: 'vehicle-trails-layer',
      type: 'line',
      source: 'vehicle-trails',
      paint: {
        'line-color': ['get', 'color'],
        'line-width': 2,
        'line-opacity': 0.6,
      },
    });

    this.map.addLayer({
      id: 'vehicles-layer',
      type: 'symbol',
      source: 'vehicles',
      layout: {
        'icon-image': 'vehicle-arrow',   // pre-loaded sprite
        'icon-size': 0.8,
        'icon-rotate': ['get', 'bearing'],
        'icon-rotation-alignment': 'map',
        'icon-allow-overlap': true,
      },
    });
  }

  _connectWS() {
    this.ws = new WebSocket('wss://fleet.example.com/live');

    this.ws.onmessage = (event) => {
      const updates = JSON.parse(event.data); // [{ id, lng, lat, speed, heading }]
      updates.forEach(u => this._receiveUpdate(u));
    };

    this.ws.onclose = () => {
      // Exponential backoff reconnect
      setTimeout(() => this._connectWS(), 3000);
    };
  }

  _receiveUpdate({ id, lng, lat, speed, heading }) {
    const existing = this.vehicles.get(id);
    const now = performance.now();

    if (existing) {
      // Store previous position as animation start; new position as target
      existing.prevLng   = existing.lng;
      existing.prevLat   = existing.lat;
      existing.targetLng = lng;
      existing.targetLat = lat;
      existing.bearing   = heading;
      existing.animStart = now;
      existing.animDuration = Math.max(100, Math.min(1000, 1000 / Math.max(speed, 1)));

      // Append to trail (keep last 50 points)
      existing.trail.push([lng, lat]);
      if (existing.trail.length > 50) existing.trail.shift();
    } else {
      this.vehicles.set(id, {
        lng, lat,
        prevLng: lng, prevLat: lat,
        targetLng: lng, targetLat: lat,
        bearing: heading,
        animStart: now,
        animDuration: 500,
        trail: [[lng, lat]],
        color: randomColor(),
      });
    }
  }

  _startRenderLoop() {
    const tick = () => {
      this._flushToMap();
      this.animFrameId = requestAnimationFrame(tick);
    };
    this.animFrameId = requestAnimationFrame(tick);
  }

  _flushToMap() {
    const now = performance.now();
    const vehicleFeatures = [];
    const trailFeatures   = [];

    this.vehicles.forEach((v, id) => {
      // Interpolate position
      const t = Math.min(1, (now - v.animStart) / v.animDuration);
      const easedT = easeInOut(t);
      const lng = lerp(v.prevLng, v.targetLng, easedT);
      const lat = lerp(v.prevLat, v.targetLat, easedT);

      vehicleFeatures.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [lng, lat] },
        properties: { id, bearing: v.bearing },
      });

      if (v.trail.length > 1) {
        trailFeatures.push({
          type: 'Feature',
          geometry: { type: 'LineString', coordinates: v.trail },
          properties: { id, color: v.color },
        });
      }
    });

    this.map.getSource('vehicles').setData({
      type: 'FeatureCollection', features: vehicleFeatures,
    });
    this.map.getSource('vehicle-trails').setData({
      type: 'FeatureCollection', features: trailFeatures,
    });
  }

  destroy() {
    cancelAnimationFrame(this.animFrameId);
    this.ws?.close();
  }
}

// Helpers
function lerp(a, b, t) { return a + (b - a) * t; }
function easeInOut(t) { return t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t; }
function randomColor() {
  return '#' + Math.floor(Math.random() * 0xffffff).toString(16).padStart(6, '0');
}
```

Key engineering decisions in the above:
- **Interpolation over time**: each server update sets `prevPos` and `targetPos`; the render loop smoothly lerps between them using the reported speed as the animation duration hint.
- **Trails as LineString**: accumulating positions into a capped array gives a breadcrumb trail without rebuilding the entire dataset each frame.
- **Direction indicator**: `icon-rotate: ['get', 'bearing']` in the layer layout rotates a sprite arrow per feature — no custom Canvas needed.
- **Exponential backoff**: the `onclose` reconnect prevents hammering the server after a drop.

---

### Server-Sent Events (SSE)

SSE is the right tool when you only need server-to-client push. It is simpler than WebSocket, auto-reconnects natively, and works over plain HTTP/2.

```javascript
// Earthquake alert stream via SSE
class EarthquakeAlertLayer {
  constructor(map) {
    this.map = map;
    this.features = [];
    this._initLayer();
    this._listenSSE();
  }

  _initLayer() {
    this.map.addSource('quakes', {
      type: 'geojson',
      data: { type: 'FeatureCollection', features: [] },
    });
    this.map.addLayer({
      id: 'quakes-circles',
      type: 'circle',
      source: 'quakes',
      paint: {
        'circle-radius': ['interpolate', ['linear'], ['get', 'magnitude'], 3, 5, 7, 30],
        'circle-color': ['interpolate', ['linear'], ['get', 'magnitude'],
          3, '#ffffb2', 5, '#fd8d3c', 7, '#d7191c'],
        'circle-opacity': 0.75,
      },
    });
  }

  _listenSSE() {
    const source = new EventSource('https://alerts.example.com/earthquakes/stream');

    source.addEventListener('quake', (event) => {
      const quake = JSON.parse(event.data);
      this.features.push({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [quake.lon, quake.lat] },
        properties: { magnitude: quake.magnitude, place: quake.place },
      });
      // Keep last 200 quakes for performance
      if (this.features.length > 200) this.features.shift();
      this.map.getSource('quakes').setData({
        type: 'FeatureCollection', features: this.features,
      });
    });

    source.onerror = (err) => {
      // EventSource auto-reconnects; no manual backoff needed
      console.warn('SSE connection error, auto-reconnecting…', err);
    };
  }
}
```

When to choose SSE over WebSocket:
- Server sends data, client only reads (weather, alerts, sensor streams)
- You want free auto-reconnect with `Last-Event-ID` resume
- You are behind HTTP/2 multiplexing (many SSE connections are cheap)
- You want to avoid WebSocket upgrade negotiation overhead

---

### MQTT (IoT)

For smart-city and environmental sensor networks, MQTT is the standard protocol. `mqtt.js` runs in the browser via WebSocket transport.

```javascript
import mqtt from 'mqtt';

class IoTSensorLayer {
  constructor(map) {
    this.map = map;
    this.sensors = new Map();
    this._pendingUpdates = [];
    this._initLayer();
    this._connectMQTT();
    this._startBatchFlush();
  }

  _connectMQTT() {
    this.client = mqtt.connect('wss://mqtt.example.com:8084/mqtt', {
      username: 'readonly',
      password: import.meta.env.VITE_MQTT_TOKEN,
    });

    this.client.on('connect', () => {
      this.client.subscribe('city/sensors/+/geo');
      this.client.subscribe('city/sensors/+/air-quality');
    });

    this.client.on('message', (topic, payload) => {
      const sensorId = topic.split('/')[2];
      const data = JSON.parse(payload.toString());
      this._pendingUpdates.push({ sensorId, ...data });
    });
  }

  _startBatchFlush() {
    // Batch all MQTT messages received within a 100ms window
    setInterval(() => {
      if (this._pendingUpdates.length === 0) return;
      this._pendingUpdates.forEach(u => {
        this.sensors.set(u.sensorId, u);
      });
      this._pendingUpdates = [];
      this._flushToMap();
    }, 100);
  }

  _flushToMap() {
    const features = Array.from(this.sensors.values()).map(s => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [s.lng, s.lat] },
      properties: { aqi: s.aqi, pm25: s.pm25, temp: s.temp },
    }));
    this.map.getSource('sensors').setData({
      type: 'FeatureCollection', features,
    });
  }
}
```

MQTT best practices for geo apps:
- **Throttle at the batch layer**: MQTT can deliver thousands of messages/sec; batch into a 100ms window before calling `setData()`.
- **Topic hierarchy**: `city/sensors/{id}/air-quality` lets you subscribe to a sensor type across all IDs with wildcard `+`.
- **QoS 0** for position updates (best-effort, lowest overhead). **QoS 1** for edit confirmations.

---

### Animation Patterns Summary

| Pattern | Use Case | Library/API |
|---|---|---|
| `requestAnimationFrame` loop | Smooth 60fps position updates | Browser native |
| Lerp + easing | Smooth movement between position reports | Utility functions |
| `deck.gl` TripsLayer | Animated trajectory replay with time scrubber | deck.gl |
| `MapLibre setData()` throttle | Batch source updates, max ~30/sec | MapLibre GL JS |
| `setFeatureState()` | Update visual properties without full re-render | MapLibre GL JS |
| `flyTo` with easing | Camera transitions to new areas | MapLibre GL JS |

### Real-Time Performance Guidelines

- **Max update rate**: 30-60 `setData()` calls/sec before you hit the render bottleneck. Beyond this, batch.
- **Delta updates**: send only changed features from the server, not the full collection. Client merges.
- **`setFeatureState()`**: for property-only changes (color, opacity, count) this avoids rebuilding the GeoJSON buffer entirely — typically 5-10x faster than `setData()`.
- **Web Workers**: parse large JSON payloads off the main thread, post the processed result back to avoid jank.
- **Typed arrays**: for very high-frequency point data (>10k points at 30fps) consider a custom WebGL layer using `Float32Array` positions instead of GeoJSON.

---

## Offline Maps & PWA

### Service Worker Tile Caching

The fundamental offline map pattern: intercept tile requests in a Service Worker and serve from the Cache API.

```javascript
// service-worker.js  (register with: navigator.serviceWorker.register('/sw.js'))

const TILE_CACHE   = 'map-tiles-v1';
const APP_CACHE    = 'app-shell-v1';

const APP_SHELL_ASSETS = [
  '/',
  '/index.html',
  '/assets/main.js',
  '/assets/main.css',
  'https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.js',
  'https://unpkg.com/maplibre-gl@latest/dist/maplibre-gl.css',
];

// Pre-cache app shell on install
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(APP_CACHE).then(cache => cache.addAll(APP_SHELL_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    // Clean up old caches
    caches.keys().then(keys =>
      Promise.all(keys
        .filter(k => k !== TILE_CACHE && k !== APP_CACHE)
        .map(k => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Tile request detection
function isTileRequest(request) {
  const url = new URL(request.url);
  return (
    /\/\d+\/\d+\/\d+\.(png|pbf|mvt)/.test(url.pathname) ||
    url.pathname.endsWith('.pmtiles')
  );
}

// Network-first for API calls, cache-first for tiles, stale-while-revalidate for styles
self.addEventListener('fetch', (event) => {
  const { request } = event;

  if (isTileRequest(request)) {
    // Cache-first: serve from cache, fall back to network and update cache
    event.respondWith(
      caches.open(TILE_CACHE).then(async (cache) => {
        const cached = await cache.match(request);
        if (cached) return cached;

        try {
          const response = await fetch(request);
          if (response.ok) {
            cache.put(request, response.clone());
          }
          return response;
        } catch (err) {
          // Offline and not cached — return a transparent 1x1 PNG to avoid map errors
          return new Response(
            atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=='),
            { headers: { 'Content-Type': 'image/png' } }
          );
        }
      })
    );
    return;
  }

  // App shell: cache-first
  event.respondWith(
    caches.match(request).then(cached => cached || fetch(request))
  );
});
```

### Tile Pre-Seeding for Known Areas

When the user nominates an offline area, seed the cache proactively:

```javascript
// Pre-cache all tiles in a bounding box for zoom levels 10-15
async function seedTileCache(bbox, minZoom, maxZoom, tileUrlTemplate) {
  const cache = await caches.open('map-tiles-v1');
  const tiles = tilesInBbox(bbox, minZoom, maxZoom);  // your tile enumeration utility

  const BATCH = 10;  // concurrent requests
  for (let i = 0; i < tiles.length; i += BATCH) {
    const batch = tiles.slice(i, i + BATCH);
    await Promise.all(
      batch.map(async ({ z, x, y }) => {
        const url = tileUrlTemplate
          .replace('{z}', z).replace('{x}', x).replace('{y}', y);
        if (await cache.match(url)) return;  // already cached
        try {
          const res = await fetch(url);
          if (res.ok) await cache.put(url, res);
        } catch (_) { /* skip on error */ }
      })
    );
    // Report progress
    self.clients.matchAll().then(clients =>
      clients.forEach(c => c.postMessage({
        type: 'SEED_PROGRESS',
        done: Math.min(i + BATCH, tiles.length),
        total: tiles.length,
      }))
    );
  }
}

// Tile count estimator (warn user before seeding large areas)
function estimateTileCount(bbox, minZoom, maxZoom) {
  let count = 0;
  for (let z = minZoom; z <= maxZoom; z++) {
    const scale = Math.pow(2, z);
    const [minTileX, maxTileY] = lngLatToTile(bbox[0], bbox[1], z);
    const [maxTileX, minTileY] = lngLatToTile(bbox[2], bbox[3], z);
    count += (maxTileX - minTileX + 1) * (maxTileY - minTileY + 1);
  }
  return count;
}
```

---

### PMTiles for Offline

PMTiles is the cleanest offline vector tile solution: one static file, no tile server needed.

```javascript
// Offline PMTiles from IndexedDB (for truly offline PWA)
import { PMTiles, Protocol } from 'pmtiles';

async function loadOfflinePMTiles(maplibreMap, storedBlob) {
  // storedBlob came from IndexedDB (saved during online session)
  const objectURL = URL.createObjectURL(storedBlob);
  const pmtiles = new PMTiles(objectURL);

  // Register custom protocol handler
  const protocol = new Protocol();
  maplibregl.addProtocol('pmtiles', protocol.tile);

  // Read metadata to build style source
  const header = await pmtiles.getHeader();
  const metadata = await pmtiles.getMetadata();

  maplibreMap.addSource('offline-basemap', {
    type: 'vector',
    url: `pmtiles://${objectURL}`,
    minzoom: header.minZoom,
    maxzoom: header.maxZoom,
  });
}

// Download and store PMTiles file to IndexedDB
async function downloadAndStorePMTiles(url, key, onProgress) {
  const response = await fetch(url);
  const reader = response.body.getReader();
  const contentLength = +response.headers.get('Content-Length');
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received / contentLength);
  }

  const blob = new Blob(chunks, { type: 'application/octet-stream' });

  const db = await openIDB('offline-maps', 1, db => {
    db.createObjectStore('pmtiles-files');
  });
  const tx = db.transaction('pmtiles-files', 'readwrite');
  await tx.objectStore('pmtiles-files').put(blob, key);
  return blob;
}
```

PMTiles CLI region extraction:

```bash
# Extract a bounding box region from a global PMTiles file
pmtiles extract world.pmtiles my-region.pmtiles \
  --bbox="-74.1,40.6,-73.7,40.9" \
  --minzoom=0 --maxzoom=14

# Inspect a PMTiles file
pmtiles show my-region.pmtiles

# Serve locally during development
pmtiles serve my-region.pmtiles --port 8080
```

---

### Offline Vector Data (Edits & Sync)

```javascript
// IndexedDB-backed offline feature store with sync queue
class OfflineFeatureStore {
  constructor(dbName = 'geo-app') {
    this.dbName = dbName;
    this._db = null;
  }

  async open() {
    this._db = await openIDB(this.dbName, 1, (db) => {
      db.createObjectStore('features', { keyPath: 'id' });
      db.createObjectStore('sync-queue', { autoIncrement: true });
    });
  }

  async saveFeature(feature) {
    const tx = this._db.transaction(['features', 'sync-queue'], 'readwrite');
    tx.objectStore('features').put(feature);
    tx.objectStore('sync-queue').add({
      op: 'upsert',
      feature,
      timestamp: Date.now(),
    });
    await tx.done;
  }

  async deleteFeature(id) {
    const tx = this._db.transaction(['features', 'sync-queue'], 'readwrite');
    tx.objectStore('features').delete(id);
    tx.objectStore('sync-queue').add({ op: 'delete', id, timestamp: Date.now() });
    await tx.done;
  }

  async syncToServer(apiUrl) {
    if (!navigator.onLine) return;

    const tx = this._db.transaction('sync-queue', 'readonly');
    const queue = await tx.objectStore('sync-queue').getAll();
    if (queue.length === 0) return;

    const response = await fetch(`${apiUrl}/sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ operations: queue }),
    });

    if (response.ok) {
      // Clear synced items
      const clearTx = this._db.transaction('sync-queue', 'readwrite');
      await clearTx.objectStore('sync-queue').clear();
    }
  }
}

// Register for Background Sync (fires when connectivity returns)
async function registerBackgroundSync() {
  const reg = await navigator.serviceWorker.ready;
  if ('sync' in reg) {
    await reg.sync.register('geo-feature-sync');
  }
}

// In service-worker.js:
self.addEventListener('sync', (event) => {
  if (event.tag === 'geo-feature-sync') {
    event.waitUntil(syncFeaturesToServer());
  }
});
```

---

### Offline-First Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ONLINE STATE                                           │
│  fetch data from API → render on map → cache in IDB    │
│  fetch tiles from CDN → cache in Cache API             │
└──────────────────────────┬──────────────────────────────┘
                           │ connection lost
┌──────────────────────────▼──────────────────────────────┐
│  OFFLINE STATE                                          │
│  read features from IDB → render on map                │
│  serve tiles from Cache API (SW intercepts)            │
│  user edits → write to IDB → add to sync-queue        │
└──────────────────────────┬──────────────────────────────┘
                           │ connection restored
┌──────────────────────────▼──────────────────────────────┐
│  SYNC STATE                                             │
│  Background Sync fires → POST sync-queue to server     │
│  server merges edits → returns authoritative state     │
│  refresh IDB cache → re-render map                    │
└─────────────────────────────────────────────────────────┘
```

---

### Progressive Web App (PWA) Setup

```json
// manifest.json
{
  "name": "GeoField App",
  "short_name": "GeoField",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#1a1a2e",
  "theme_color": "#0f3460",
  "orientation": "any",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ]
}
```

```javascript
// main.js — register service worker
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    try {
      const reg = await navigator.serviceWorker.register('/sw.js');
      console.log('SW registered:', reg.scope);

      // Listen for SW updates
      reg.addEventListener('updatefound', () => {
        const newWorker = reg.installing;
        newWorker.addEventListener('statechange', () => {
          if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
            showUpdateBanner();  // "New version available — refresh"
          }
        });
      });
    } catch (err) {
      console.error('SW registration failed:', err);
    }
  });
}
```

---

## Collaborative & Multi-User Maps

### Real-Time Collaboration with Yjs + MapLibre

Yjs provides conflict-free replicated data types (CRDTs) that auto-merge edits from multiple users without a central coordinator.

```javascript
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { awarenessStatesToArray } from 'y-protocols/awareness';

class CollaborativeMapEditor {
  constructor(maplibreMap, roomId) {
    this.map = maplibreMap;

    // Shared document
    this.ydoc = new Y.Doc();

    // Shared types — features keyed by feature ID
    this.yfeatures = this.ydoc.getMap('features');

    // WebSocket sync provider
    this.provider = new WebsocketProvider(
      'wss://collab.example.com',
      roomId,
      this.ydoc
    );

    // Awareness: broadcasts user cursor/selection state
    this.awareness = this.provider.awareness;
    this.awareness.setLocalState({
      user: { name: 'User', color: randomColor() },
      cursor: null,   // [lng, lat] or null
    });

    this._bindYjsToMap();
    this._bindMapToYjs();
    this._renderOtherCursors();
  }

  _bindYjsToMap() {
    // When any remote edit comes in, re-render the shared features
    this.yfeatures.observeDeep(() => {
      const features = Array.from(this.yfeatures.values()).map(ymap => ({
        type: 'Feature',
        id: ymap.get('id'),
        geometry: JSON.parse(ymap.get('geometry')),
        properties: ymap.get('properties').toJSON(),
      }));
      this.map.getSource('shared-features').setData({
        type: 'FeatureCollection', features,
      });
    });
  }

  addFeature(feature) {
    const ymap = new Y.Map();
    ymap.set('id', feature.id);
    ymap.set('geometry', JSON.stringify(feature.geometry));
    const props = new Y.Map();
    Object.entries(feature.properties).forEach(([k, v]) => props.set(k, v));
    ymap.set('properties', props);
    this.yfeatures.set(feature.id, ymap);
  }

  updateFeatureProperty(featureId, key, value) {
    const ymap = this.yfeatures.get(featureId);
    if (ymap) {
      this.ydoc.transact(() => {
        ymap.get('properties').set(key, value);
      });
    }
  }

  _bindMapToYjs() {
    // Broadcast cursor position to other users
    this.map.on('mousemove', (e) => {
      this.awareness.setLocalStateField('cursor', [e.lngLat.lng, e.lngLat.lat]);
    });
  }

  _renderOtherCursors() {
    // Re-render all other users' cursors on any awareness change
    this.awareness.on('change', () => {
      const states = awarenessStatesToArray(this.awareness.states);
      const cursors = states
        .filter(s => s.clientId !== this.ydoc.clientID && s.cursor)
        .map(s => ({
          type: 'Feature',
          geometry: { type: 'Point', coordinates: s.cursor },
          properties: { color: s.user?.color, name: s.user?.name },
        }));
      this.map.getSource('user-cursors').setData({
        type: 'FeatureCollection', features: cursors,
      });
    });
  }
}
```

---

## Testing Geo Web Apps

### Unit Testing Spatial Logic

Turf.js operations are pure functions — easy to unit test. Use `vitest` (fast, ESM-native) or `jest`.

```javascript
// spatial.test.js — vitest
import { describe, test, expect } from 'vitest';
import * as turf from '@turf/turf';

describe('Spatial Analysis', () => {
  test('buffer creates correct polygon', () => {
    const point = turf.point([0, 0]);
    const buffered = turf.buffer(point, 1, { units: 'kilometers' });
    expect(buffered.geometry.type).toBe('Polygon');
    // π * (1000m)² ≈ 3,141,592 m²
    expect(turf.area(buffered)).toBeCloseTo(3_141_592, -2);
  });

  test('point in polygon returns true for contained point', () => {
    const polygon = turf.polygon([[
      [0, 0], [10, 0], [10, 10], [0, 10], [0, 0],
    ]]);
    expect(turf.booleanPointInPolygon(turf.point([5, 5]), polygon)).toBe(true);
    expect(turf.booleanPointInPolygon(turf.point([15, 5]), polygon)).toBe(false);
  });

  test('coordinate round-trip from WGS84 to Web Mercator', () => {
    const original = [13.4050, 52.5200];  // Berlin
    const mercator = toWebMercator(original);
    const backToWgs84 = fromWebMercator(mercator);
    expect(backToWgs84[0]).toBeCloseTo(original[0], 5);
    expect(backToWgs84[1]).toBeCloseTo(original[1], 5);
  });

  test('spatial index returns features in bbox', () => {
    const index = buildRTree(testFeatures);
    const results = index.search({ minX: -1, minY: -1, maxX: 1, maxY: 1 });
    expect(results.length).toBeGreaterThan(0);
    results.forEach(f => {
      expect(f.geometry.coordinates[0]).toBeGreaterThanOrEqual(-1);
    });
  });
});
```

### Component Testing with MapLibre Mock

MapLibre requires WebGL, which is unavailable in most CI environments. Mock the map for component tests.

```javascript
// __mocks__/maplibre-gl.js
const mockMap = {
  addSource: vi.fn(),
  addLayer: vi.fn(),
  removeSource: vi.fn(),
  removeLayer: vi.fn(),
  getSource: vi.fn(() => ({ setData: vi.fn() })),
  on: vi.fn((event, handler) => { mockMap._handlers[event] = handler; }),
  fire: vi.fn((event, data) => { mockMap._handlers[event]?.(data); }),
  _handlers: {},
};

export default {
  Map: vi.fn(() => mockMap),
  Popup: vi.fn(() => ({
    setLngLat: vi.fn().mockReturnThis(),
    setHTML: vi.fn().mockReturnThis(),
    addTo: vi.fn().mockReturnThis(),
    remove: vi.fn(),
  })),
  LngLat: vi.fn((lng, lat) => ({ lng, lat })),
};

// MapLayer.test.jsx
import { render, screen, fireEvent } from '@testing-library/react';
import { MapLayer } from './MapLayer';

test('renders layer and handles feature click', async () => {
  const onFeatureClick = vi.fn();
  render(<MapLayer onFeatureClick={onFeatureClick} />);

  // Simulate a map click event via mock
  const { Map } = await import('maplibre-gl');
  const mapInstance = Map.mock.results[0].value;
  mapInstance.fire('click', {
    features: [{ properties: { id: '123', name: 'Test Feature' } }],
  });

  expect(onFeatureClick).toHaveBeenCalledWith(
    expect.objectContaining({ properties: { id: '123', name: 'Test Feature' } })
  );
});

test('popup shows feature name', async () => {
  render(<MapLayer showPopups />);
  // ...trigger click, assert popup HTML
});
```

### Visual Regression Testing with Playwright

```javascript
// tests/map-visual.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Map visual regression', () => {
  test('basemap renders at default viewport', async ({ page }) => {
    await page.goto('/map');
    // Wait for MapLibre idle event (tiles loaded)
    await page.waitForFunction(() => {
      return new Promise(resolve => {
        const map = window.__mapInstance__;
        if (!map) return resolve(false);
        map.once('idle', () => resolve(true));
      });
    }, { timeout: 15000 });

    await expect(page).toHaveScreenshot('map-default.png', {
      maxDiffPixelRatio: 0.02,  // allow 2% pixel difference
    });
  });

  test('choropleth renders at zoom 10', async ({ page }) => {
    await page.goto('/map?z=10&lng=13.40&lat=52.52');
    await page.waitForFunction(() => window.__mapIdle__);
    await expect(page.locator('#map-container'))
      .toHaveScreenshot('choropleth-z10.png', { maxDiffPixelRatio: 0.03 });
  });

  test('dark mode map renders correctly', async ({ page }) => {
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.goto('/map');
    await page.waitForFunction(() => window.__mapIdle__);
    await expect(page).toHaveScreenshot('map-dark-mode.png');
  });
});
```

```javascript
// playwright.config.ts
import { defineConfig } from '@playwright/test';
export default defineConfig({
  snapshotPathTemplate: '{testDir}/__snapshots__/{testFilePath}/{arg}{ext}',
  use: { viewport: { width: 1280, height: 720 } },
  projects: [
    { name: 'chromium', use: { browserName: 'chromium' } },
    { name: 'mobile', use: { viewport: { width: 390, height: 844 } } },
  ],
});
```

### End-to-End Testing

```javascript
// tests/map-interactions.spec.ts
import { test, expect } from '@playwright/test';

test('user can draw a polygon and see area calculation', async ({ page }) => {
  await page.goto('/map');
  await page.waitForFunction(() => window.__mapIdle__);

  // Click draw polygon button
  await page.click('[data-testid="draw-polygon"]');

  // Click 4 points on the map canvas to form a polygon
  const mapCanvas = page.locator('canvas.maplibregl-canvas');
  await mapCanvas.click({ position: { x: 400, y: 300 } });
  await mapCanvas.click({ position: { x: 500, y: 300 } });
  await mapCanvas.click({ position: { x: 500, y: 400 } });
  await mapCanvas.click({ position: { x: 400, y: 400 } });
  await mapCanvas.dblclick({ position: { x: 400, y: 300 } });  // close polygon

  // Assert area is displayed
  const areaDisplay = page.locator('[data-testid="polygon-area"]');
  await expect(areaDisplay).toBeVisible();
  const areaText = await areaDisplay.textContent();
  expect(parseFloat(areaText)).toBeGreaterThan(0);
});

test('search returns result and flies to it', async ({ page }) => {
  await page.goto('/map');
  await page.fill('[data-testid="search-input"]', 'Berlin');
  await page.click('[data-testid="search-result-0"]');

  // Map should have flown to Berlin area
  await page.waitForFunction(() => window.__mapIdle__);
  const center = await page.evaluate(() => {
    const map = window.__mapInstance__;
    return map.getCenter();
  });
  expect(center.lng).toBeCloseTo(13.4, 0);
  expect(center.lat).toBeCloseTo(52.5, 0);
});
```

### Performance Testing

```javascript
// Lighthouse CI config
// lighthouserc.js
module.exports = {
  ci: {
    collect: { url: ['http://localhost:3000/map'], numberOfRuns: 3 },
    assert: {
      assertions: {
        'categories:performance': ['error', { minScore: 0.7 }],
        'first-contentful-paint': ['warn', { maxNumericValue: 2000 }],
        'interactive': ['error', { maxNumericValue: 5000 }],
      },
    },
    upload: { target: 'temporary-public-storage' },
  },
};

// Custom map performance metrics
function measureMapPerformance(map) {
  const metrics = {};

  const mapStartTime = performance.now();

  new PerformanceObserver((list) => {
    list.getEntries().forEach(entry => {
      if (entry.name.includes('maplibre')) {
        metrics.firstPaint = entry.startTime;
      }
    });
  }).observe({ type: 'paint', buffered: true });

  map.once('load', () => {
    metrics.mapLoad = performance.now() - mapStartTime;
  });

  map.once('idle', () => {
    metrics.mapIdle = performance.now() - mapStartTime;
    console.table(metrics);

    // Report to your analytics backend
    fetch('/api/metrics', {
      method: 'POST',
      body: JSON.stringify({ ...metrics, url: location.pathname }),
    });
  });
}
```

---

## Accessibility (a11y) for Maps

Maps are inherently visual and present real accessibility challenges. Here are the practical solutions.

### Keyboard Navigation

MapLibre supports keyboard controls natively, but interactive features need additional work:

```javascript
// Make feature popups keyboard accessible
map.on('click', 'poi-layer', (e) => {
  const feature = e.features[0];
  const popup = new maplibregl.Popup()
    .setLngLat(feature.geometry.coordinates)
    .setHTML(`
      <div role="dialog" aria-label="${feature.properties.name}" tabindex="-1">
        <h3>${feature.properties.name}</h3>
        <p>${feature.properties.description}</p>
        <button onclick="this.closest('.maplibregl-popup').querySelector('[aria-label]').remove()">
          Close
        </button>
      </div>
    `)
    .addTo(map);

  // Move focus into the popup
  setTimeout(() => {
    popup.getElement().querySelector('[tabindex="-1"]')?.focus();
  }, 50);
});

// Keyboard shortcut to cycle through features
let featureFocusIndex = 0;
document.addEventListener('keydown', (e) => {
  if (e.key === 'F6') {  // "focus map features"
    const features = map.queryRenderedFeatures({ layers: ['poi-layer'] });
    if (!features.length) return;
    featureFocusIndex = (featureFocusIndex + 1) % features.length;
    const f = features[featureFocusIndex];
    map.flyTo({ center: f.geometry.coordinates, zoom: Math.max(map.getZoom(), 14) });
    announceToScreenReader(`Feature: ${f.properties.name}`);
  }
});
```

### Screen Reader Support

```html
<!-- Map container with ARIA -->
<div
  id="map"
  role="application"
  aria-label="Interactive map"
  aria-describedby="map-description"
  tabindex="0"
>
  <!-- Screen reader description, hidden visually -->
  <div id="map-description" class="sr-only">
    Interactive map showing points of interest. Use arrow keys to pan,
    +/- to zoom, Tab to navigate controls.
  </div>

  <!-- Live region for dynamic updates (new alerts, search results) -->
  <div id="map-live-region" aria-live="polite" aria-atomic="false" class="sr-only"></div>

  <!-- Skip link -->
  <a href="#map-data-table" class="skip-link">Skip to data table</a>
</div>

<!-- Alternative data table (same data as map, accessible) -->
<section id="map-data-table" aria-label="Map data table">
  <h2>Locations</h2>
  <table>
    <thead><tr><th>Name</th><th>Latitude</th><th>Longitude</th><th>Category</th></tr></thead>
    <tbody id="feature-table-body"><!-- populated by JS --></tbody>
  </table>
</section>
```

```javascript
// Announce map events to screen readers
function announceToScreenReader(message) {
  const region = document.getElementById('map-live-region');
  region.textContent = '';  // clear first to re-trigger for same message
  requestAnimationFrame(() => { region.textContent = message; });
}

// Update table alongside map
function syncTableWithMap(features) {
  const tbody = document.getElementById('feature-table-body');
  tbody.innerHTML = features.map(f => `
    <tr>
      <td><button onclick="flyToFeature('${f.id}')">${f.properties.name}</button></td>
      <td>${f.geometry.coordinates[1].toFixed(5)}</td>
      <td>${f.geometry.coordinates[0].toFixed(5)}</td>
      <td>${f.properties.category}</td>
    </tr>
  `).join('');
}
```

### Color Accessibility

```javascript
// ColorBrewer colorblind-safe sequential palette
const COLORBLIND_SAFE_SEQUENTIAL = [
  '#ffffd9', '#edf8b1', '#c7e9b4', '#7fcdbb',
  '#41b6c4', '#1d91c0', '#225ea8', '#253494', '#081d58',
];

// Pattern fills alongside color coding (for print/colorblind users)
// Add hatching pattern to fill layers using MapLibre image API
async function addHatchPattern(map, id, color, angle = 45) {
  const size = 16;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = size;
  const ctx = canvas.getContext('2d');
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.translate(size / 2, size / 2);
  ctx.rotate((angle * Math.PI) / 180);
  ctx.beginPath();
  ctx.moveTo(-size, 0); ctx.lineTo(size, 0);
  ctx.stroke();
  return new Promise(resolve =>
    canvas.toBlob(blob => {
      createImageBitmap(blob).then(img => {
        map.addImage(id, img, { pixelRatio: 2 });
        resolve();
      });
    })
  );
}
```

---

## Internationalization (i18n)

### Multi-Language Map Labels

```javascript
// MapLibre style expression for multi-language labels
// Falls back: preferred language → English → any name available
function getNameExpression(languageCode) {
  return [
    'coalesce',
    ['get', `name:${languageCode}`],
    ['get', 'name:en'],
    ['get', 'name'],
  ];
}

// Apply to all label layers in style
function setMapLanguage(map, languageCode) {
  const style = map.getStyle();
  style.layers
    .filter(l => l.layout?.['text-field'])
    .forEach(layer => {
      map.setLayoutProperty(layer.id, 'text-field', getNameExpression(languageCode));
    });
}

// RTL text support (Arabic, Hebrew, etc.)
import { useRTLTextPlugin } from 'maplibre-gl';
maplibregl.setRTLTextPlugin(
  'https://unpkg.com/@mapbox/mapbox-gl-rtl-text@0.2.3/mapbox-gl-rtl-text.min.js',
  null,
  true  // lazy load
);
```

### Coordinate Display Localization

```javascript
// Format coordinates based on user locale and preference
function formatCoordinate(value, type, format = 'decimal', locale = navigator.language) {
  if (format === 'decimal') {
    return new Intl.NumberFormat(locale, {
      minimumFractionDigits: 5,
      maximumFractionDigits: 5,
    }).format(value) + (type === 'lat' ? '°N' : '°E');
  }

  if (format === 'dms') {
    const abs = Math.abs(value);
    const deg = Math.floor(abs);
    const min = Math.floor((abs - deg) * 60);
    const sec = ((abs - deg - min / 60) * 3600).toFixed(1);
    const dir = type === 'lat' ? (value >= 0 ? 'N' : 'S') : (value >= 0 ? 'E' : 'W');
    return `${deg}°${min}′${sec}″${dir}`;
  }
}
```

---

## Security

### API Key Protection

```javascript
// NEVER do this — exposes key in client bundle
const map = new maplibregl.Map({
  style: `https://api.maptiler.com/maps/streets/style.json?key=YOUR_SECRET_KEY`,
});

// DO this — proxy through your backend
// backend/tile-proxy.ts (Express)
import express from 'express';
import { createProxyMiddleware } from 'http-proxy-middleware';
import rateLimit from 'express-rate-limit';

const tileProxy = express.Router();

tileProxy.use(rateLimit({ windowMs: 60_000, max: 500 }));

tileProxy.use('/tiles', createProxyMiddleware({
  target: 'https://api.maptiler.com',
  changeOrigin: true,
  pathRewrite: (path) => `${path}?key=${process.env.MAPTILER_API_KEY}`,
  onProxyRes: (proxyRes) => {
    // Strip upstream CORS headers, add your own
    proxyRes.headers['access-control-allow-origin'] = 'https://yourapp.com';
  },
}));

// In MapLibre style JSON — point to your proxy
// "tiles": ["https://yourapp.com/tiles/maps/streets/{z}/{x}/{y}.png"]
```

### User-Uploaded GeoJSON Sanitization

```javascript
function sanitizeGeoJSON(input, options = {}) {
  const {
    maxFeatures = 10_000,
    maxFileSizeBytes = 50 * 1024 * 1024,  // 50 MB
    allowedGeometryTypes = ['Point', 'LineString', 'Polygon', 'MultiPoint',
                            'MultiLineString', 'MultiPolygon'],
  } = options;

  if (JSON.stringify(input).length > maxFileSizeBytes) {
    throw new Error('GeoJSON file too large');
  }

  if (!['FeatureCollection', 'Feature'].includes(input.type)) {
    throw new Error('Invalid GeoJSON type');
  }

  const features = input.type === 'FeatureCollection' ? input.features : [input];

  if (features.length > maxFeatures) {
    throw new Error(`Too many features: ${features.length} > ${maxFeatures}`);
  }

  return {
    type: 'FeatureCollection',
    features: features.map(f => {
      if (!allowedGeometryTypes.includes(f.geometry?.type)) {
        throw new Error(`Disallowed geometry type: ${f.geometry?.type}`);
      }

      validateCoordinates(f.geometry);

      return {
        type: 'Feature',
        geometry: f.geometry,
        // Strip and re-serialize properties to eliminate prototype pollution / injection
        properties: JSON.parse(
          JSON.stringify(f.properties ?? {})
            .replace(/<script[^>]*>.*?<\/script>/gi, '')  // strip script tags
        ),
      };
    }),
  };
}

function validateCoordinates(geometry) {
  const validate = ([lng, lat]) => {
    if (lng < -180 || lng > 180) throw new Error(`Invalid longitude: ${lng}`);
    if (lat < -90 || lat > 90) throw new Error(`Invalid latitude: ${lat}`);
  };
  // Recurse through all coordinate arrays based on geometry type
  // ...
}
```

---

## Deployment Patterns

### Static JAMstack (PMTiles + MapLibre)

```
Cost: ~$0 for most apps
Scale: effectively infinite (CDN)
Architecture: No backend required

dist/
├── index.html
├── assets/main.js
├── assets/main.css
├── tiles/region.pmtiles    (or hosted on R2/S3)
└── data/features.geojson
```

```bash
# Deploy to Cloudflare Pages with R2 for PMTiles
wrangler pages deploy dist --project-name geo-app
wrangler r2 object put geo-tiles/region.pmtiles --file ./region.pmtiles
```

### Docker Compose (Full Stack)

```yaml
# docker-compose.yml — PostGIS + Martin tile server + App
version: '3.9'
services:
  db:
    image: postgis/postgis:15-3.4
    environment:
      POSTGRES_DB: geodata
      POSTGRES_USER: geo
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/init.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U geo -d geodata"]
      interval: 5s
      timeout: 5s
      retries: 10

  martin:
    image: maplibre/martin:latest
    depends_on:
      db: { condition: service_healthy }
    environment:
      DATABASE_URL: postgresql://geo:${DB_PASSWORD}@db/geodata
    ports: ["3000:3000"]
    command: ["--listen-addresses", "0.0.0.0:3000", "--auto-publish"]

  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports: ["80:80"]
    depends_on: [martin]
    environment:
      VITE_TILE_SERVER_URL: http://martin:3000
```

```dockerfile
# Dockerfile — Multi-stage build
FROM node:20-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM nginx:1.25-alpine AS production
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### Cloudflare Workers for Tile Transformation

```javascript
// workers/tile-transform.js
// On-the-fly tile transformation at the edge (zero cold start)
export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const [, z, x, y] = url.pathname.match(/\/(\d+)\/(\d+)\/(\d+)\.pbf$/) ?? [];

    if (!z) return new Response('Not found', { status: 404 });

    // Fetch from R2 bucket
    const object = await env.TILES_BUCKET.get(`${z}/${x}/${y}.pbf`);
    if (!object) return new Response('Tile not found', { status: 404 });

    const buffer = await object.arrayBuffer();

    return new Response(buffer, {
      headers: {
        'Content-Type': 'application/x-protobuf',
        'Content-Encoding': 'gzip',
        'Cache-Control': 'public, max-age=86400, s-maxage=604800',
        'Access-Control-Allow-Origin': '*',
        'Vary': 'Accept-Encoding',
      },
    });
  },
};
```

### CDN Strategy

```
Tile CDN (immutable assets):
  Cache-Control: public, max-age=31536000, immutable
  → Tiles are immutable at {z}/{x}/{y} — cache forever

App assets (hashed filenames from build tool):
  main.abc123.js → Cache-Control: max-age=31536000, immutable
  index.html     → Cache-Control: no-cache (always re-validate)

Dynamic API responses:
  Cache-Control: public, max-age=60, s-maxage=300
  Surrogate-Key: region-{bbox} (for tag-based purging)
```

---

## Monitoring in Production

```javascript
// map-monitoring.js — production error tracking and performance
export function attachMapMonitoring(map, analytics) {
  // Tile load errors
  map.on('error', (e) => {
    analytics.captureError('map_error', {
      message: e.error?.message,
      sourceLayer: e.sourceLayer,
      tile: e.tile,
      url: window.location.pathname,
    });
  });

  // WebGL context loss (typically GPU memory pressure)
  map.getCanvas().addEventListener('webglcontextlost', () => {
    analytics.captureEvent('webgl_context_lost');
    // Show fallback static map image
    document.getElementById('map-fallback').style.display = 'block';
  });

  map.getCanvas().addEventListener('webglcontextrestored', () => {
    analytics.captureEvent('webgl_context_restored');
    document.getElementById('map-fallback').style.display = 'none';
  });

  // Performance: time to first idle (tiles fully loaded)
  const loadStart = performance.now();
  map.once('idle', () => {
    const ttfi = performance.now() - loadStart;
    analytics.captureMetric('map_time_to_first_idle_ms', ttfi);
  });

  // Track viewport for tile cache warming (anonymized — just bbox)
  let viewportDebounce;
  map.on('moveend', () => {
    clearTimeout(viewportDebounce);
    viewportDebounce = setTimeout(() => {
      const bounds = map.getBounds();
      analytics.captureEvent('viewport_change', {
        zoom: Math.round(map.getZoom()),
        // Rough bbox only — no precise center to avoid tracking
        bbox: [
          Math.round(bounds.getWest()), Math.round(bounds.getSouth()),
          Math.round(bounds.getEast()), Math.round(bounds.getNorth()),
        ],
      });
    }, 2000);
  });
}

// React error boundary for map component
import { Component } from 'react';

export class MapErrorBoundary extends Component {
  state = { hasError: false, error: null };

  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }

  componentDidCatch(error, info) {
    console.error('Map component crashed:', error, info);
    // Report to error tracking service
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="map-error-fallback">
          <img src="/static/fallback-map.jpg" alt="Map unavailable" />
          <p>Map failed to load. <button onClick={() => this.setState({ hasError: false })}>
            Retry
          </button></p>
        </div>
      );
    }
    return this.props.children;
  }
}
```

---

## Advanced Dark Arts

These are the cutting-edge browser APIs that most geo developers have not yet reached for. Each one solves a real production problem.

### Predictive Tile Loading

```javascript
// Analyze scroll velocity and direction to preload tiles the user is heading toward
class PredictiveTileLoader {
  constructor(map) {
    this.map = map;
    this._velocityHistory = [];
    this._lastBounds = null;
    map.on('move', () => this._recordMovement());
    setInterval(() => this._preload(), 500);
  }

  _recordMovement() {
    this._velocityHistory.push({
      time: Date.now(),
      center: this.map.getCenter(),
    });
    if (this._velocityHistory.length > 10) this._velocityHistory.shift();
  }

  _preload() {
    if (this._velocityHistory.length < 3) return;
    const recent = this._velocityHistory.slice(-3);
    const dt = recent[2].time - recent[0].time;
    if (dt === 0) return;

    const dlng = (recent[2].center.lng - recent[0].center.lng) / dt * 500;
    const dlat = (recent[2].center.lat - recent[0].center.lat) / dt * 500;

    // Trigger map to load tiles in predicted direction by peeking at bounds
    const predictedCenter = {
      lng: recent[2].center.lng + dlng,
      lat: recent[2].center.lat + dlat,
    };
    this.map.setCenter(predictedCenter);  // triggers tile load, then snap back
    requestAnimationFrame(() =>
      this.map.setCenter(recent[2].center)  // immediately return — tiles are loading in bg
    );
  }
}
```

### Web Locks API — Prevent Concurrent Tab Sync

```javascript
// Only one browser tab syncs offline edits at a time
async function syncWithLock() {
  try {
    await navigator.locks.request('geo-offline-sync', { mode: 'exclusive' }, async () => {
      console.log('Lock acquired — syncing...');
      await syncFeaturesToServer();
      // Lock auto-releases when this async function returns
    });
  } catch (err) {
    if (err.name === 'AbortError') {
      console.log('Another tab is syncing, skipping');
    }
  }
}
```

### Broadcast Channel — Cross-Tab Map State Sync

```javascript
// Sync map view and layer visibility across browser tabs
const mapChannel = new BroadcastChannel('geo-app-map-state');

// Sender tab
map.on('moveend', () => {
  mapChannel.postMessage({
    type: 'VIEW_CHANGE',
    center: map.getCenter(),
    zoom: map.getZoom(),
    bearing: map.getBearing(),
    pitch: map.getPitch(),
  });
});

// Receiver tab
mapChannel.onmessage = (event) => {
  if (event.data.type === 'VIEW_CHANGE') {
    map.jumpTo({
      center: event.data.center,
      zoom: event.data.zoom,
      bearing: event.data.bearing,
      pitch: event.data.pitch,
    });
  }
};
```

### Compression Streams API — Decompress Tile Responses

```javascript
// Decompress gzipped tile data without a library
async function decompressGzipTile(response) {
  if (!response.headers.get('content-encoding')?.includes('gzip')) {
    return response.arrayBuffer();
  }

  const ds = new DecompressionStream('gzip');
  const stream = response.body.pipeThrough(ds);
  const reader = stream.getReader();
  const chunks = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }

  const totalLength = chunks.reduce((acc, c) => acc + c.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;
  chunks.forEach(chunk => { result.set(chunk, offset); offset += chunk.length; });
  return result.buffer;
}
```

### High-Accuracy Location Tracking for Field Apps

```javascript
class FieldLocationTracker {
  constructor(map) {
    this.map = map;
    this.watchId = null;
    this.positions = [];
  }

  start() {
    if (!('geolocation' in navigator)) {
      throw new Error('Geolocation not available');
    }

    this.watchId = navigator.geolocation.watchPosition(
      (position) => this._onPosition(position),
      (error) => this._onError(error),
      {
        enableHighAccuracy: true,   // GPS vs network location
        maximumAge: 0,              // no cached position
        timeout: 5000,
      }
    );
  }

  _onPosition(position) {
    const { longitude, latitude, accuracy, heading, speed } = position.coords;

    // Apply Kalman filter for noise reduction (simplified)
    const filtered = this._kalmanFilter(longitude, latitude, accuracy);

    this.positions.push(filtered);

    this.map.getSource('my-location').setData({
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          geometry: { type: 'Point', coordinates: [filtered.lng, filtered.lat] },
          properties: { accuracy, heading: heading ?? 0, speed: speed ?? 0 },
        },
        {
          // Accuracy circle
          type: 'Feature',
          geometry: turf.circle([filtered.lng, filtered.lat], accuracy / 1000).geometry,
          properties: {},
        },
      ],
    });
  }

  _onError(error) {
    const messages = {
      1: 'Location permission denied',
      2: 'Location unavailable',
      3: 'Location request timed out',
    };
    console.error(messages[error.code] ?? 'Unknown location error');
  }

  stop() {
    if (this.watchId !== null) {
      navigator.geolocation.clearWatch(this.watchId);
      this.watchId = null;
    }
  }
}
```

### DeviceOrientation API — Compass for AR Map Overlays

```javascript
// Show map north indicator and rotate basemap to device compass heading
async function enableCompassMode(map) {
  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    // iOS 13+ requires explicit permission
    const permission = await DeviceOrientationEvent.requestPermission();
    if (permission !== 'granted') throw new Error('Compass permission denied');
  }

  let smoothedAlpha = null;

  window.addEventListener('deviceorientationabsolute', (event) => {
    if (event.alpha === null) return;

    // Smooth compass jitter with exponential moving average
    if (smoothedAlpha === null) {
      smoothedAlpha = event.alpha;
    } else {
      smoothedAlpha = 0.85 * smoothedAlpha + 0.15 * event.alpha;
    }

    // Rotate map bearing to match device compass
    map.setBearing(smoothedAlpha);
  }, true);
}
```

### WebTransport — Next-Gen Real-Time (Experimental)

```javascript
// WebTransport: lower latency than WebSocket, unreliable datagrams for position updates
async function connectWebTransport(url) {
  const transport = new WebTransport(url);
  await transport.ready;

  // Use unreliable datagrams for position updates (UDP-like, no retry overhead)
  const writer = transport.datagrams.writable.getWriter();
  const reader = transport.datagrams.readable.getReader();

  // Send position updates as compact binary (not JSON)
  function sendPosition(lng, lat, heading) {
    const buf = new Float32Array([lng, lat, heading]);
    writer.write(buf.buffer);
  }

  // Receive position updates
  async function receivePositions() {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const view = new Float32Array(value);
      updateVehiclePosition(view[0], view[1], view[2]);
    }
  }

  receivePositions();
  return { sendPosition, transport };
}
```

### WebGPU Compute Shaders for Raster Analysis (Experimental)

```javascript
// GPU-accelerated NDVI calculation on satellite imagery tiles
// (requires WebGPU support: Chrome 113+, requires HTTPS)
async function computeNDVI(redBand, nirBand) {
  const adapter = await navigator.gpu.requestAdapter();
  const device = await adapter.requestDevice();

  const shaderCode = `
    @group(0) @binding(0) var<storage, read> red: array<f32>;
    @group(0) @binding(1) var<storage, read> nir: array<f32>;
    @group(0) @binding(2) var<storage, read_write> ndvi: array<f32>;

    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) id: vec3<u32>) {
      let i = id.x;
      let r = red[i];
      let n = nir[i];
      ndvi[i] = (n - r) / (n + r + 0.0001);  // avoid division by zero
    }
  `;

  const shaderModule = device.createShaderModule({ code: shaderCode });
  // ... create buffers, bind groups, dispatch compute pass
  // Result: ndvi Float32Array at GPU speed, in the browser
}
```

---

## Quick Reference: Choosing the Right Pattern

| Requirement | Solution |
|---|---|
| Live vehicle tracking | WebSocket + MapLibre + requestAnimationFrame interpolation |
| Simple server push (alerts) | SSE + EventSource |
| IoT sensor network | MQTT.js + 100ms batch flush |
| Offline maps, no backend | PMTiles + Service Worker (cache-first tiles) |
| Offline with user edits | IndexedDB + sync queue + Background Sync API |
| Multi-user collaborative editing | Yjs CRDT + WebSocket provider |
| Animated historical trajectories | deck.gl TripsLayer |
| Unit testing spatial logic | vitest + turf.js mocked geometries |
| Visual regression tests | Playwright `toHaveScreenshot()` |
| API key protection | Backend proxy + domain restriction |
| Field GPS tracking | `watchPosition()` + Kalman filter |
| Cross-tab state sync | BroadcastChannel API |
| Prevent concurrent sync | Web Locks API |
| Deploy with zero backend | Vercel/Netlify/CF Pages + PMTiles on R2 |
| Production tile CDN | `Cache-Control: immutable` + Cloudflare CDN |
