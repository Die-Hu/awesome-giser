# Real-Time & Collaboration -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**Fleet tracking:** Socket.io + MapLibre real-time layer updates. **IoT sensors:** MQTT + TimescaleDB + deck.gl. **Collaborative editing:** Yjs CRDT (with y-redis + periodic compaction). **Database change notifications:** PostGIS LISTEN/NOTIFY (90% of use cases). **Guaranteed delivery at scale:** Debezium CDC + Kafka. **Rapid prototyping:** Supabase Realtime. **One-way server push:** SSE (simpler than WebSocket when bidirectional is not needed).

---

## Tier 1 -- Production First Choices

---

### Socket.io -- Real-Time Spatial Events

The most popular WebSocket library. Built-in reconnection, room-based broadcasting, Redis adapter for horizontal scaling.

**Why Tier 1:** Millions of production deployments. The standard for web-based real-time spatial applications (fleet tracking, dashboards). For enterprise scale (>10K concurrent connections), use Redis adapter + multiple Socket.io pods behind a sticky-session load balancer.

#### Server (Node.js)

```typescript
// server.ts -- Fleet tracking WebSocket server
import { Server } from 'socket.io';
import { createServer } from 'http';
import { Pool } from 'pg';

const httpServer = createServer();
const io = new Server(httpServer, {
  cors: { origin: '*', methods: ['GET', 'POST'] },
  transports: ['websocket', 'polling'],
});

const pool = new Pool({ connectionString: process.env.DATABASE_URL });
const vehiclePositions = new Map<string, {
  id: string; lng: number; lat: number; speed: number; heading: number; timestamp: number;
}>();

// Handle GPS device connections
io.of('/vehicles').on('connection', (socket) => {
  const vehicleId = socket.handshake.auth.vehicleId;

  socket.on('position', async (data) => {
    const position = {
      id: vehicleId, lng: data.lng, lat: data.lat,
      speed: data.speed, heading: data.heading, timestamp: Date.now(),
    };
    vehiclePositions.set(vehicleId, position);

    // Broadcast to all dashboard clients
    io.of('/dashboard').emit('vehicle:update', position);

    // Store in database
    await pool.query(
      `INSERT INTO vehicle_tracks (vehicle_id, geom, speed, heading, timestamp)
       VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3), 4326), $4, $5, NOW())`,
      [vehicleId, data.lng, data.lat, data.speed, data.heading]
    );
  });

  socket.on('disconnect', () => {
    vehiclePositions.delete(vehicleId);
    io.of('/dashboard').emit('vehicle:offline', { id: vehicleId });
  });
});

// Handle dashboard client connections
io.of('/dashboard').on('connection', (socket) => {
  const allPositions = Array.from(vehiclePositions.values());
  socket.emit('vehicles:init', {
    type: 'FeatureCollection',
    features: allPositions.map((v) => ({
      type: 'Feature',
      properties: { id: v.id, speed: v.speed, heading: v.heading },
      geometry: { type: 'Point', coordinates: [v.lng, v.lat] },
    })),
  });

  socket.on('subscribe:bbox', (bbox) => {
    socket.join(`bbox:${bbox.join(',')}`);
  });
});

// Periodic geofence check
setInterval(async () => {
  for (const [id, pos] of vehiclePositions) {
    const result = await pool.query(
      `SELECT zone_name FROM geofences
       WHERE ST_Contains(geom, ST_SetSRID(ST_MakePoint($1, $2), 4326))`,
      [pos.lng, pos.lat]
    );
    if (result.rows.length > 0) {
      io.of('/dashboard').emit('geofence:enter', {
        vehicleId: id, zone: result.rows[0].zone_name, position: [pos.lng, pos.lat],
      });
    }
  }
}, 5000);

httpServer.listen(3001);
```

#### Client (MapLibre + React)

```jsx
// components/FleetMap.jsx
import { useEffect, useRef, useState } from 'react';
import Map, { Source, Layer } from 'react-map-gl/maplibre';
import { io } from 'socket.io-client';

export default function FleetMap() {
  const [vehicles, setVehicles] = useState(null);

  useEffect(() => {
    const socket = io('http://localhost:3001/dashboard', { transports: ['websocket'] });

    socket.on('vehicles:init', (geojson) => setVehicles(geojson));

    socket.on('vehicle:update', (position) => {
      setVehicles((prev) => {
        if (!prev) return prev;
        const features = prev.features.map((f) =>
          f.properties.id === position.id
            ? {
                ...f,
                properties: { ...f.properties, speed: position.speed, heading: position.heading },
                geometry: { type: 'Point', coordinates: [position.lng, position.lat] },
              }
            : f
        );
        if (!features.some((f) => f.properties.id === position.id)) {
          features.push({
            type: 'Feature',
            properties: { id: position.id, speed: position.speed, heading: position.heading },
            geometry: { type: 'Point', coordinates: [position.lng, position.lat] },
          });
        }
        return { ...prev, features };
      });
    });

    socket.on('vehicle:offline', ({ id }) => {
      setVehicles((prev) => {
        if (!prev) return prev;
        return { ...prev, features: prev.features.filter((f) => f.properties.id !== id) };
      });
    });

    return () => { socket.disconnect(); };
  }, []);

  return (
    <Map
      initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 11 }}
      style={{ width: '100%', height: '100vh' }}
      mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
    >
      {vehicles && (
        <Source id="vehicles" type="geojson" data={vehicles}>
          <Layer
            id="vehicle-circles"
            type="circle"
            paint={{
              'circle-radius': 8,
              'circle-color': [
                'interpolate', ['linear'], ['get', 'speed'],
                0, '#ff0000', 30, '#ffff00', 60, '#00ff00',
              ],
              'circle-stroke-width': 2,
              'circle-stroke-color': '#fff',
            }}
          />
          <Layer
            id="vehicle-labels"
            type="symbol"
            layout={{ 'text-field': ['get', 'id'], 'text-size': 10, 'text-offset': [0, 1.5] }}
            paint={{ 'text-color': '#fff' }}
          />
        </Source>
      )}
    </Map>
  );
}
```

#### Scaling with Redis

```typescript
// Horizontal scaling: multiple WebSocket servers behind load balancer
import { createAdapter } from '@socket.io/redis-adapter';
import { createClient } from 'redis';

const pubClient = createClient({ url: 'redis://localhost:6379' });
const subClient = pubClient.duplicate();
await pubClient.connect();
await subClient.connect();

const io = new Server(httpServer);
io.adapter(createAdapter(pubClient, subClient));
// Now multiple Socket.io servers share state via Redis
```

**Caveats:**
- **Scaling requires Redis.** Single-server Socket.io is easy. Multi-server requires `@socket.io/redis-adapter`. Redis becomes a SPOF for real-time features.
- **Memory per connection.** Each WebSocket connection consumes ~10-50KB. 10K concurrent connections = 100-500MB. Must monitor server memory.
- **Reconnection storms.** After a server restart, all clients reconnect simultaneously. This can overwhelm the server. Use exponential backoff and jitter (built-in but must be configured).
- **HTTP polling fallback.** Socket.io falls back to HTTP long-polling if WebSocket upgrade fails. This is 10x more resource-intensive per connection. Corporate proxies often block WebSocket upgrades.
- **Binary protocol overhead.** Socket.io adds its own framing protocol on top of WebSocket. For high-frequency GPS updates (10Hz), raw WebSocket or MQTT is more efficient.

---

### MQTT -- IoT Sensor Networks

The IoT industry standard for lightweight messaging. Ideal for high-frequency sensor data (GPS, air quality, temperature) where thousands of devices publish data. Mosquitto is the most popular open-source MQTT broker for development and single-node deployments.

**Why Tier 1:** Industry standard for IoT. Handles 100K+ connections on a single Mosquitto node. For the full sensor-to-map pipeline: sensors -> MQTT -> ingest service -> TimescaleDB (time-series) + PostGIS (spatial) -> Martin (tiles) + Socket.io (dashboard).

**WARNING: For production HA deployments, use EMQX (open-source edition) instead of attempting Mosquitto bridging.** Mosquitto is single-node by design. Multi-node Mosquitto bridge configurations are fragile and not suitable for production HA. EMQX provides native clustering, built-in dashboard, dynamic ACL via HTTP/database backends, and horizontal scaling out of the box.

```yaml
# docker-compose.yml -- MQTT sensor infrastructure
version: '3.8'
services:
  mosquitto:
    image: eclipse-mosquitto:2
    ports: ["1883:1883", "9001:9001"]  # MQTT + WebSocket
    volumes:
      - ./mosquitto.conf:/mosquitto/config/mosquitto.conf

  timescaledb:
    image: timescale/timescaledb-ha:pg16
    environment:
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports: ["5432:5432"]

  ingest:
    build: ./ingest
    environment:
      MQTT_BROKER: mqtt://mosquitto:1883
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@timescaledb/postgres
    depends_on: [mosquitto, timescaledb]
```

```typescript
// ingest/index.ts -- MQTT to TimescaleDB + PostGIS
import mqtt from 'mqtt';
import { Pool } from 'pg';

const client = mqtt.connect(process.env.MQTT_BROKER!);
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

await pool.query(`
  CREATE EXTENSION IF NOT EXISTS postgis;
  CREATE EXTENSION IF NOT EXISTS timescaledb;

  CREATE TABLE IF NOT EXISTS sensor_readings (
    time TIMESTAMPTZ NOT NULL,
    sensor_id TEXT NOT NULL,
    sensor_type TEXT NOT NULL,
    value DOUBLE PRECISION,
    geom GEOMETRY(Point, 4326)
  );
  SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);
  CREATE INDEX ON sensor_readings USING GIST (geom, time);
`);

client.subscribe('sensors/#');

let batch: any[] = [];
const BATCH_SIZE = 100;

client.on('message', async (topic, message) => {
  const parts = topic.split('/'); // sensors/{region}/{type}/{id}
  const data = JSON.parse(message.toString());

  batch.push({
    time: new Date(data.timestamp || Date.now()),
    sensor_id: parts[3], sensor_type: parts[2],
    value: data.value, lng: data.longitude, lat: data.latitude,
  });

  if (batch.length >= BATCH_SIZE) {
    const values = batch.map((b, i) => {
      const o = i * 6;
      return `($${o+1}, $${o+2}, $${o+3}, $${o+4}, ST_SetSRID(ST_MakePoint($${o+5}, $${o+6}), 4326))`;
    }).join(',');
    const params = batch.flatMap((b) => [b.time, b.sensor_id, b.sensor_type, b.value, b.lng, b.lat]);
    await pool.query(
      `INSERT INTO sensor_readings (time, sensor_id, sensor_type, value, geom) VALUES ${values}`,
      params
    );
    batch = [];
  }
});
```

```javascript
// Dashboard: MQTT over WebSocket + MapLibre
import mqtt from 'mqtt';

const client = mqtt.connect('ws://localhost:9001');
client.subscribe('sensors/+/air_quality/+');

const sensorData = new Map();

client.on('message', (topic, message) => {
  const data = JSON.parse(message.toString());
  const sensorId = topic.split('/')[3];
  sensorData.set(sensorId, { ...data, id: sensorId, timestamp: Date.now() });

  map.getSource('sensors')?.setData({
    type: 'FeatureCollection',
    features: Array.from(sensorData.values()).map((s) => ({
      type: 'Feature',
      properties: { id: s.id, value: s.value },
      geometry: { type: 'Point', coordinates: [s.longitude, s.latitude] },
    })),
  });
});
```

**Caveats:**
- **No built-in persistence for messages.** QoS 2 retained messages are stored but history is not. Need TimescaleDB/InfluxDB for time-series storage.
- **Topic ACL management.** ACL files must be manually managed. No dynamic ACL API without plugins. Large deployments need a custom auth plugin (or use EMQX which has built-in HTTP/database ACL backends).
- **WebSocket bridge overhead.** Browser clients connect via WebSocket transport. The ws:// to MQTT translation adds latency and memory overhead.
- **Mosquitto clustering is limited.** Mosquitto is single-node. For HA, use EMQX (open-source, native clustering) or HiveMQ (commercial). Do not attempt multi-Mosquitto bridge configurations in production -- they are fragile and poorly documented.
- **Retained messages can accumulate.** If thousands of sensors publish retained messages, new subscribers receive thousands of messages on connect, causing browser freezes.

---

### PostGIS LISTEN/NOTIFY -- Lightweight Change Notifications

Built into PostgreSQL. Zero additional infrastructure. Sufficient for 90% of spatial change notification use cases.

**Why Tier 1:** The pragmatic choice for most GIS applications. Handles tile cache invalidation, dashboard refresh, and simple real-time updates without any additional infrastructure. Use Debezium only when you need guaranteed delivery, multi-consumer, or at-scale streaming.

```sql
-- Trigger for real-time spatial notifications
CREATE OR REPLACE FUNCTION notify_spatial_change()
RETURNS trigger AS $$
DECLARE
  payload json;
BEGIN
  payload = json_build_object(
    'operation', TG_OP,
    'table', TG_TABLE_NAME,
    'id', COALESCE(NEW.id, OLD.id),
    'bbox', CASE
      WHEN TG_OP = 'DELETE' THEN ST_AsGeoJSON(ST_Envelope(OLD.geom))::json
      ELSE ST_AsGeoJSON(ST_Envelope(NEW.geom))::json
    END
  );
  PERFORM pg_notify('spatial_changes', payload::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER features_notify
AFTER INSERT OR UPDATE OR DELETE ON features
FOR EACH ROW EXECUTE FUNCTION notify_spatial_change();
```

```python
# Python listener for PostGIS notifications
import asyncio
import asyncpg
import json

async def listen_spatial_changes():
    conn = await asyncpg.connect(dsn='postgresql://user:pass@localhost/gisdb')

    async def handler(conn, pid, channel, payload):
        data = json.loads(payload)
        print(f"Spatial change: {data['operation']} on {data['table']} id={data['id']}")
        await invalidate_tiles_for_bbox(data['bbox'])

    await conn.add_listener('spatial_changes', handler)
    while True:
        await asyncio.sleep(1)

asyncio.run(listen_spatial_changes())
```

**Caveats:**
- **Payload size limit.** NOTIFY payload is limited to 8000 bytes. Large geometries must be referenced by ID, not included inline.
- **No guaranteed delivery.** If the listener is disconnected when a notification fires, it's lost. Not suitable for mission-critical event delivery.
- **Single connection.** Each LISTEN consumes a PostgreSQL connection. For multiple channels, use a single connection with multiple LISTEN commands.
- **No persistence.** Notifications are fire-and-forget. No replay capability. For guaranteed delivery, use Debezium CDC.

---

## Tier 2 -- Specialized Use Cases

---

### Yjs CRDT -- Collaborative Map Editing

Conflict-free replicated data types for real-time collaborative editing. Used by Notion, Jupyter, many collaborative editors. Works well for multi-user map drawing and feature editing.

**Why Tier 2:** Production-readiness 4/5 **when using y-redis for persistence + periodic document compaction**. Drops to 2/5 with the default y-websocket server, which keeps all documents in memory with no compaction -- unsuitable for production collaborative GIS where documents grow unboundedly as features are edited.

```typescript
// Collaborative drawing with conflict-free replicated data
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import MapboxDraw from '@mapbox/mapbox-gl-draw';

const ydoc = new Y.Doc();
const provider = new WebsocketProvider('wss://sync.example.com', 'map-room-123', ydoc);

const yFeatures = ydoc.getMap('features');

// Awareness (cursor positions, user info)
const awareness = provider.awareness;
awareness.setLocalState({
  user: { name: 'Alice', color: '#ff0000' },
  cursor: null,
});

const draw = new MapboxDraw({
  displayControlsDefault: false,
  controls: { polygon: true, line_string: true, point: true, trash: true },
});
map.addControl(draw);

// Sync local draws to Yjs
map.on('draw.create', (e) => {
  for (const feature of e.features) {
    yFeatures.set(feature.id, feature);
  }
});

map.on('draw.update', (e) => {
  for (const feature of e.features) {
    yFeatures.set(feature.id, feature);
  }
});

map.on('draw.delete', (e) => {
  for (const feature of e.features) {
    yFeatures.delete(feature.id);
  }
});

// Sync remote changes to MapboxDraw
yFeatures.observe((event) => {
  event.changes.keys.forEach((change, key) => {
    if (change.action === 'add' || change.action === 'update') {
      const feature = yFeatures.get(key);
      draw.add(feature);
    } else if (change.action === 'delete') {
      draw.delete(key);
    }
  });
});

// Share cursor position on map
map.on('mousemove', (e) => {
  awareness.setLocalStateField('cursor', { lng: e.lngLat.lng, lat: e.lngLat.lat });
});

// Show other users' cursors
awareness.on('change', () => {
  const states = Array.from(awareness.getStates().entries());
  const cursors = states
    .filter(([id]) => id !== ydoc.clientID)
    .map(([, state]) => state)
    .filter((s) => s.cursor);

  map.getSource('cursors')?.setData({
    type: 'FeatureCollection',
    features: cursors.map((s) => ({
      type: 'Feature',
      properties: { name: s.user.name, color: s.user.color },
      geometry: { type: 'Point', coordinates: [s.cursor.lng, s.cursor.lat] },
    })),
  });
});
```

**Caveats:**
- **Document size growth.** CRDT documents grow monotonically. Undo history is retained forever. For spatial features that are frequently edited, the Yjs document can grow to 10-100x the actual data size. **Periodic compaction is essential for production use.**
- **Conflict resolution is automatic but opaque.** When two users edit the same feature simultaneously, Yjs resolves the conflict deterministically but may produce unexpected geometry. No "resolve conflict" UI by default.
- **y-websocket server scaling.** The default y-websocket server keeps all documents in memory. Large maps with thousands of features can consume gigabytes. **Must use y-redis for multi-server deployments and persistence.**
- **Awareness protocol overhead.** Broadcasting cursor positions to all participants at 60fps generates significant WebSocket traffic. Must throttle to 5-10fps for maps.
- **GeoJSON-specific challenges.** Yjs works on JSON structures. GeoJSON coordinate arrays (deeply nested) are expensive to diff and sync. Editing a polygon vertex generates larger deltas than editing a text character.

---

### Supabase Realtime -- Managed Database Subscriptions

Listen to PostGIS table changes in real-time via managed WebSocket connections. Built on PostgreSQL's replication, with presence support for collaborative features.

**Why Tier 2:** The fastest way to add real-time spatial features. Excellent for rapid development. For enterprise, evaluate connection limits carefully and plan for self-hosting if vendor lock-in is a concern.

```typescript
import { createClient } from '@supabase/supabase-js';

const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

// Listen to spatial table changes
const channel = supabase
  .channel('spatial-changes')
  .on('postgres_changes', {
    event: '*',
    schema: 'public',
    table: 'features',
  }, (payload) => {
    switch (payload.eventType) {
      case 'INSERT': addFeatureToMap(payload.new); break;
      case 'UPDATE': updateFeatureOnMap(payload.new); break;
      case 'DELETE': removeFeatureFromMap(payload.old.id); break;
    }
  })
  .subscribe();

// Presence: who is viewing the map
const presenceChannel = supabase.channel('map-presence');

presenceChannel
  .on('presence', { event: 'sync' }, () => {
    const state = presenceChannel.presenceState();
    updateUserCursors(state);
  })
  .subscribe(async (status) => {
    if (status === 'SUBSCRIBED') {
      await presenceChannel.track({
        user_id: currentUser.id,
        name: currentUser.name,
        viewport: map.getBounds().toArray(),
      });
    }
  });

map.on('moveend', () => {
  presenceChannel.track({
    user_id: currentUser.id,
    name: currentUser.name,
    viewport: map.getBounds().toArray(),
  });
});
```

**Caveats:**
- **Realtime channel limits.** Free tier: 200 concurrent connections, 2 channels. Pro tier: 500 connections. For large collaborative maps, this can be limiting.
- **Postgres Changes listener delay.** `postgres_changes` events are not instant -- there's a 100-500ms delay. For real-time vehicle tracking, this is too slow. Use direct WebSocket for sub-100ms requirements.
- **RLS + Realtime performance.** Row-Level Security policies are evaluated for every broadcast message. Complex spatial RLS policies (ST_Intersects checks) can slow down broadcast significantly.
- **Vendor lock-in.** Supabase-specific APIs don't have portable equivalents.
- **Self-hosting complexity.** Self-hosted Supabase requires 10+ services. The Docker Compose file is complex.

---

### Debezium CDC -- Enterprise Change Data Capture

Change Data Capture via PostgreSQL logical replication. Streams every spatial table change to Kafka for multi-consumer processing. Enterprise-grade guaranteed delivery.

**Why Tier 2:** The right tool when you need guaranteed delivery of every spatial change, multiple consumers (tile cache invalidation + analytics + alerting), and replay capability. Overkill for most GIS apps -- use PostGIS LISTEN/NOTIFY first.

```json
{
  "name": "postgis-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgis",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "password",
    "database.dbname": "gisdb",
    "topic.prefix": "gis",
    "table.include.list": "public.features,public.buildings",
    "column.include.list": "public.features.id,public.features.name,public.features.geom",
    "plugin.name": "pgoutput",
    "transforms": "unwrap",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter"
  }
}
```

**Caveats:**
- **Infrastructure weight.** Kafka + Zookeeper (or KRaft) + Kafka Connect is 3-6 services before you write a single line of application code. Each needs monitoring.
- **Geometric data serialization.** PostGIS geometry columns are serialized as WKB hex strings in CDC events. Consumer code must parse these with a spatial library.
- **Slot management.** PostgreSQL replication slots used by Debezium prevent WAL cleanup. If the connector goes down for hours, WAL files accumulate and can fill the disk, crashing PostgreSQL.
- **Schema evolution.** Adding a column to a spatial table requires connector restart. Schema registry adds another service to manage.
- **Anti-pattern: Using Debezium when PostGIS LISTEN/NOTIFY would suffice.** LISTEN/NOTIFY handles 90% of change notification use cases without Kafka.

---

## Server-Sent Events (SSE)

### SSE for Spatial Notifications

One-way server-to-client streaming. Simpler than WebSocket, with built-in reconnection.

```python
# FastAPI SSE endpoint for spatial updates
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio, json

app = FastAPI()
spatial_events = asyncio.Queue()

async def event_generator(request: Request, bbox=None):
    while True:
        if await request.is_disconnected():
            break
        try:
            event = await asyncio.wait_for(spatial_events.get(), timeout=30)
            if bbox:
                coords = event.get('coordinates', [0, 0])
                if not (bbox[0] <= coords[0] <= bbox[2] and bbox[1] <= coords[1] <= bbox[3]):
                    continue
            yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        except asyncio.TimeoutError:
            yield f": keepalive\n\n"

@app.get("/events/spatial")
async def spatial_sse(request: Request, bbox: str = None):
    parsed_bbox = [float(c) for c in bbox.split(',')] if bbox else None
    return StreamingResponse(
        event_generator(request, parsed_bbox),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
```

### SSE vs WebSocket Decision Matrix

| Factor | SSE | WebSocket |
|--------|-----|-----------|
| Direction | Server -> Client only | Bidirectional |
| Protocol | HTTP/1.1+ | Upgrade to ws:// |
| Auto-reconnect | Built-in | Manual |
| Binary data | No (text only) | Yes |
| Max connections | ~6 per domain (HTTP/1.1) | Unlimited |
| Best for GIS | Notifications, status | Real-time tracking, collaboration |

---

## Architecture Patterns

### Event-Driven Spatial Architecture

```
┌──────────┐  ┌──────────┐  ┌──────────┐
│  GPS     │  │  Sensors │  │  Editor  │
│  Devices │  │  (IoT)   │  │  (Web)   │
└────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │              │
     └─────────────┼──────────────┘
                   │
            ┌──────▼──────┐
            │  Event Bus  │  (Kafka / Redis Streams)
            └──────┬──────┘
                   │
     ┌─────────────┼─────────────┐
     │             │             │
┌────▼────┐  ┌────▼────┐  ┌────▼────┐
│  Store  │  │  Cache  │  │  Alert  │
│ (PostGIS│  │  Inval  │  │ Service │
│  + TSDB)│  │  (tiles)│  │(geofence│
└─────────┘  └─────────┘  └─────────┘
```

### Geofencing -- Real-Time Point-in-Polygon

```python
from rtree import index
from shapely.geometry import shape, Point

class GeofenceManager:
    def __init__(self):
        self.idx = index.Index()
        self.zones = {}

    def add_zone(self, zone_id: str, geojson: dict):
        geom = shape(geojson['geometry'])
        self.zones[zone_id] = {'geom': geom, 'properties': geojson['properties']}
        self.idx.insert(hash(zone_id), geom.bounds, obj=zone_id)

    def check_point(self, lng: float, lat: float) -> list:
        point = Point(lng, lat)
        candidates = list(self.idx.intersection((lng, lat, lng, lat), objects=True))
        results = []
        for item in candidates:
            zone_id = item.object
            zone = self.zones[zone_id]
            if zone['geom'].contains(point):
                results.append({'zone_id': zone_id, 'zone_name': zone['properties'].get('name')})
        return results

gf = GeofenceManager()
gf.add_zone('zone1', zone1_geojson)
zones = gf.check_point(116.4, 39.9)
if zones:
    emit_geofence_alert(vehicle_id, zones)
```

### Rate Limiting Spatial Streams

```typescript
class SpatialThrottler {
  private lastEmit = new Map<string, number>();
  private minInterval: number;

  constructor(minIntervalMs: number = 1000) {
    this.minInterval = minIntervalMs;
  }

  shouldEmit(entityId: string): boolean {
    const now = Date.now();
    const last = this.lastEmit.get(entityId) || 0;
    if (now - last >= this.minInterval) {
      this.lastEmit.set(entityId, now);
      return true;
    }
    return false;
  }
}

const throttler = new SpatialThrottler(1000); // Max 1 update/second per vehicle

socket.on('position', (data) => {
  if (throttler.shouldEmit(data.vehicleId)) {
    io.of('/dashboard').emit('vehicle:update', data);
  }
});
```

### Robust WebSocket Reconnection

```typescript
class SpatialWebSocket {
  private ws: WebSocket | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(private url: string) {
    this.connect();
  }

  private connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      this.reconnectDelay = 1000;
      this.startHeartbeat();
    };

    this.ws.onclose = () => {
      this.stopHeartbeat();
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'pong') return;
      this.handleMessage(data);
    };
  }

  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }));
      }
    }, 30000);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) clearInterval(this.heartbeatInterval);
  }

  private handleMessage(data: any) {
    // Override in subclass or use event emitter
  }

  send(data: any) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  close() {
    this.stopHeartbeat();
    this.ws?.close();
  }
}
```
