# Real-Time, Offline & Advanced Patterns

> Data validated: 2026-03-21
>
> For enterprise real-time architecture (Debezium CDC, LISTEN/NOTIFY, scaling strategies) -> [web-dev/realtime-and-collaboration.md](../web-dev/realtime-and-collaboration.md)
> For testing and security deep dives -> [web-dev/testing-and-security.md](../web-dev/testing-and-security.md)

## 30-Second Decision

| I need... | Use this | Startup time |
|-----------|----------|-------------|
| Simple live data stream (one-way) | SSE (Server-Sent Events) | 15 min |
| Bi-directional real-time | Socket.io | 30 min |
| Quick real-time with auth | Supabase Realtime | 30 min |
| IoT sensor streams | MQTT (mqtt.js) | 1 hour |
| Offline maps (PWA) | Service Worker + PMTiles | 2 hours |
| Collaborative map editing | Yjs CRDT | Half a day |
| Enterprise CDC pipeline | -> [web-dev/realtime-and-collaboration.md](../web-dev/realtime-and-collaboration.md) | Days |

---

## Real-Time Data

### 1. Server-Sent Events (SSE)

Simplest real-time pattern. One-way server-to-client over HTTP. Zero client-side library.

```javascript
const source = new EventSource('/api/locations/stream');
source.onmessage = (event) => {
  const { vehicleId, lng, lat } = JSON.parse(event.data);
  updateVehiclePosition(vehicleId, [lng, lat]);
};
// Auto-reconnects on connection loss
```

**Small project:** Best starting point -- zero library, auto-reconnection, works through corporate proxies.

**Key caveats:**
- One-way only (server -> client)
- Browser limit: ~6 SSE connections per domain; use a single multiplexed connection
- Text-only (UTF-8); no binary data

---

### 2. Supabase Realtime

Managed real-time with zero infrastructure. PostgreSQL change detection + PostGIS = spatial real-time out of the box.

```javascript
import { createClient } from '@supabase/supabase-js';
const supabase = createClient('https://xxx.supabase.co', 'public-anon-key');

const channel = supabase
  .channel('map-updates')
  .on('postgres_changes', {
    event: 'INSERT', schema: 'public', table: 'features'
  }, (payload) => addFeatureToMap(payload.new))
  .subscribe();
```

**Small project:** Excellent -- zero infra, auth + DB + real-time in one, PostGIS native, free tier (200 concurrent connections).

**Key caveats:**
- Latency 100-500ms (not instant)
- Connection limits (free: 200, pro: 500)
- RLS + spatial conditions add overhead to broadcasts
- **Anti-pattern:** Using for high-frequency IoT (>10 events/sec). Use MQTT instead.

---

### 3. Socket.io

Standard for bi-directional real-time. WebSocket + fallbacks, rooms, reconnection.

**Quick facts:** ~50KB gzip | ~63K GitHub stars | ~8M npm/week | 5/5 production-readiness

```javascript
import { io } from 'socket.io-client';
const socket = io('https://api.example.com');

socket.emit('join-region', { bbox: [116.0, 39.5, 117.0, 40.5] });
socket.on('vehicle-update', ({ id, lng, lat, bearing }) => {
  updateVehicle(id, [lng, lat], bearing);
});

map.on('moveend', () => {
  const b = map.getBounds();
  socket.emit('join-region', {
    bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()]
  });
});
```

**Small project:** Safe choice for real-time -- largest ecosystem, auto-reconnection, room-based filtering.

**Key caveats:**
- Scaling requires Redis adapter for multi-server
- Memory: ~10-50KB per connection; 10K connections = 100-500MB
- Reconnection storms after restart (use exponential backoff + jitter)
- **Anti-pattern:** Broadcasting all GPS updates to all clients. Use room/bbox filtering.

For Socket.io scaling architecture -> [web-dev/realtime-and-collaboration.md](../web-dev/realtime-and-collaboration.md)

---

### 4. MQTT (mqtt.js)

IoT and sensor network standard. Lightweight pub-sub for millions of connections.

**Quick facts:** mqtt.js ~30KB gzip | 5/5 production-readiness (IoT standard)

```javascript
import mqtt from 'mqtt';

// Production: MUST use wss:// (TLS). ws:// is for development only.
const client = mqtt.connect('wss://broker.example.com:8884');

client.subscribe('sensors/+/temperature');
client.on('message', (topic, message) => {
  const sensorId = topic.split('/')[1];
  const { value, lat, lng } = JSON.parse(message);
  updateSensorOnMap(sensorId, [lng, lat], value);
});
```

**Small project:** Overkill unless you specifically need IoT sensors, QoS guarantees, or thousands of device connections.

**Key caveats:**
- Browser connects via WebSocket transport
- Mosquitto is single-node; for HA, use EMQX (open-source) -- not Mosquitto bridging
- Topic design is critical -- bad hierarchies cause subscription explosions
- **Production: always use `wss://` (TLS), never `ws://`**

---

## Offline Maps & PWA

### 5. Service Worker + PMTiles

Fully offline maps with zero server. Cache PMTiles in a service worker.

```javascript
// service-worker.js (Workbox)
import { registerRoute } from 'workbox-routing';
import { CacheFirst, StaleWhileRevalidate } from 'workbox-strategies';

registerRoute(
  ({ url }) => url.pathname.endsWith('.pmtiles'),
  new CacheFirst({ cacheName: 'tile-cache' })
);
registerRoute(
  ({ url }) => url.pathname.match(/\.(js|css|png|json)$/),
  new StaleWhileRevalidate({ cacheName: 'map-assets' })
);
```

**Key caveats:**
- PMTiles use Range requests -- default Workbox strategies don't cache 206 responses correctly; need custom handler
- Storage quotas: Chrome ~60% of disk, Safari ~1GB
- iOS Safari IndexedDB corruption bugs -- always have server-side backup

### 6. IndexedDB + Dexie.js

Offline vector data storage. Store and query GeoJSON features locally.

```javascript
import Dexie from 'dexie';
const db = new Dexie('geo-app');
db.version(1).stores({
  features: '++id, type, [minLng+minLat+maxLng+maxLat]',
  edits: '++id, featureId, timestamp, synced'
});

await db.features.bulkPut(geojsonFeatures.map(f => ({
  ...f, ...turf.bbox(f)
})));
```

**Key caveats:** No spatial indexing in IndexedDB. Must implement sync logic manually. Transaction deadlocks possible.

For detailed PWA/offline architecture -> [web-dev/pwa-and-offline.md](../web-dev/pwa-and-offline.md)

---

## Collaborative Editing

### 7. Yjs (CRDT)

Real-time collaborative map editing with conflict-free merging. Used by Notion, Jupyter.

**Quick facts:** ~15KB gzip | ~21K GitHub stars

**Production-readiness: 4/5 -- with prerequisites.** This rating assumes you use y-redis + periodic document compression. With default y-websocket server only (all docs in memory), production-readiness drops to 2/5.

```javascript
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';

const doc = new Y.Doc();
const provider = new WebsocketProvider('wss://sync.example.com', 'map-room', doc);
const features = doc.getMap('features');

features.set('building-1', {
  type: 'Feature',
  geometry: { type: 'Polygon', coordinates: [...] },
  properties: { name: 'HQ', editor: 'user-123' }
});

features.observe((event) => {
  event.changes.keys.forEach((change, key) => {
    if (change.action === 'add' || change.action === 'update')
      updateFeatureOnMap(key, features.get(key));
    else if (change.action === 'delete')
      removeFeatureFromMap(key);
  });
});
```

**Small project:** Overkill unless you genuinely need multiple users editing simultaneously. For single-user editing, a REST API is 10x simpler.

**Key caveats:**
- Concurrent edits to same polygon vertex can produce geometrically invalid results
- Document size grows monotonically (CRDT tombstones) -- need periodic garbage collection
- y-websocket default server keeps all docs in memory -- not production-safe
- **Anti-pattern:** Using Yjs for non-collaborative editing

---

## Quick Testing Reference

| Test Type | Tool | What to Test |
|-----------|------|--------------|
| Unit | Vitest | Spatial calculations, data transforms |
| Integration | Testcontainers + PostGIS | DB queries, API endpoints |
| E2E | Playwright | Map rendering, user interactions |

For detailed testing patterns and code examples -> [web-dev/testing-and-security.md](../web-dev/testing-and-security.md)

---

## Security Checklist

| Risk | Mitigation |
|------|------------|
| API key exposure | Server-side proxy, never embed in client JS |
| SQL injection in spatial queries | Parameterized queries, never string-concat ST_ functions |
| GeoJSON injection | Validate all user-submitted GeoJSON server-side |
| Tile endpoint abuse | Rate-limit; bots generate surprise CDN bills |
| Location privacy | Hash/quantize coords before storing |

For detailed security patterns -> [web-dev/testing-and-security.md](../web-dev/testing-and-security.md)

---

## Deployment Quick Reference

```
Small project:
  Static Host (Vercel/CF) --> PMTiles on CDN (S3/R2)      $0-5/mo

Medium:
  Frontend (Vercel) --> Martin (Docker) --> PostGIS (Docker)  $20-40/mo

Production:
  Frontend --> CDN Cache --> Martin (k8s) --> PostGIS (Managed)
```

For full deployment architecture -> [web-dev/deployment.md](../web-dev/deployment.md)
