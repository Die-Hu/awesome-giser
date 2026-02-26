# Real-Time & Collaborative GIS — 2025 Complete Guide

Real-time geospatial applications power fleet tracking, IoT sensor dashboards, collaborative map editing, and live event monitoring. This guide covers WebSocket, SSE, MQTT, CRDT-based collaboration, and streaming architectures for spatial data.

> **Quick Picks**
> - **Fleet tracking:** Socket.io + MapLibre real-time layer updates
> - **IoT sensors:** MQTT (Mosquitto) + time-series DB + deck.gl
> - **Collaborative editing:** Yjs CRDT + MapLibre Draw
> - **Database changes:** Supabase Realtime or PostGIS LISTEN/NOTIFY
> - **Streaming pipeline:** Kafka + Debezium CDC for spatial tables

---

## WebSocket for Spatial Data

### Socket.io + GeoJSON — Real-Time Fleet Tracking

#### Server (Node.js)

```typescript
// server.ts — Fleet tracking WebSocket server
import { Server } from 'socket.io';
import { createServer } from 'http';
import { Pool } from 'pg';

const httpServer = createServer();
const io = new Server(httpServer, {
  cors: { origin: '*', methods: ['GET', 'POST'] },
  transports: ['websocket', 'polling'],
});

const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// Store latest vehicle positions
const vehiclePositions = new Map<string, {
  id: string;
  lng: number;
  lat: number;
  speed: number;
  heading: number;
  timestamp: number;
}>();

// Handle GPS device connections
io.of('/vehicles').on('connection', (socket) => {
  const vehicleId = socket.handshake.auth.vehicleId;
  console.log(`Vehicle ${vehicleId} connected`);

  // Receive GPS updates from vehicles
  socket.on('position', async (data) => {
    const position = {
      id: vehicleId,
      lng: data.lng,
      lat: data.lat,
      speed: data.speed,
      heading: data.heading,
      timestamp: Date.now(),
    };

    vehiclePositions.set(vehicleId, position);

    // Broadcast to all dashboard clients
    io.of('/dashboard').emit('vehicle:update', position);

    // Store in database (batched)
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
  // Send current state of all vehicles
  const allPositions = Array.from(vehiclePositions.values());
  socket.emit('vehicles:init', {
    type: 'FeatureCollection',
    features: allPositions.map((v) => ({
      type: 'Feature',
      properties: { id: v.id, speed: v.speed, heading: v.heading },
      geometry: { type: 'Point', coordinates: [v.lng, v.lat] },
    })),
  });

  // Subscribe to a geographic area
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
        vehicleId: id,
        zone: result.rows[0].zone_name,
        position: [pos.lng, pos.lat],
      });
    }
  }
}, 5000);

httpServer.listen(3001, () => console.log('WebSocket server on :3001'));
```

#### Client (MapLibre + React)

```jsx
// components/FleetMap.jsx
import { useEffect, useRef, useState, useCallback } from 'react';
import Map, { Source, Layer } from 'react-map-gl/maplibre';
import { io } from 'socket.io-client';

export default function FleetMap() {
  const [vehicles, setVehicles] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    const socket = io('http://localhost:3001/dashboard', {
      transports: ['websocket'],
    });
    socketRef.current = socket;

    // Initial state
    socket.on('vehicles:init', (geojson) => {
      setVehicles(geojson);
    });

    // Real-time updates
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
        // Add if new
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

    // Vehicle went offline
    socket.on('vehicle:offline', ({ id }) => {
      setVehicles((prev) => {
        if (!prev) return prev;
        return { ...prev, features: prev.features.filter((f) => f.properties.id !== id) };
      });
    });

    // Geofence alerts
    socket.on('geofence:enter', (alert) => {
      console.log(`Vehicle ${alert.vehicleId} entered ${alert.zone}`);
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
                0, '#ff0000',
                30, '#ffff00',
                60, '#00ff00',
              ],
              'circle-stroke-width': 2,
              'circle-stroke-color': '#fff',
            }}
          />
          <Layer
            id="vehicle-labels"
            type="symbol"
            layout={{
              'text-field': ['get', 'id'],
              'text-size': 10,
              'text-offset': [0, 1.5],
            }}
            paint={{ 'text-color': '#fff' }}
          />
        </Source>
      )}
    </Map>
  );
}
```

### Scaling WebSocket with Redis Pub/Sub

```typescript
// Horizontal scaling: multiple WebSocket servers behind load balancer
import { Server } from 'socket.io';
import { createAdapter } from '@socket.io/redis-adapter';
import { createClient } from 'redis';

const pubClient = createClient({ url: 'redis://localhost:6379' });
const subClient = pubClient.duplicate();

await pubClient.connect();
await subClient.connect();

const io = new Server(httpServer);
io.adapter(createAdapter(pubClient, subClient));

// Now multiple Socket.io servers share state via Redis
// Messages published on one server are received by clients on all servers
```

---

## Server-Sent Events (SSE)

### SSE for Spatial Notifications

```python
# FastAPI SSE endpoint for spatial updates
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json

app = FastAPI()

# Shared event queue
spatial_events = asyncio.Queue()

async def event_generator(request: Request, bbox=None):
    while True:
        if await request.is_disconnected():
            break

        try:
            event = await asyncio.wait_for(spatial_events.get(), timeout=30)

            # Filter by bbox if provided
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
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

# Trigger events (from data changes, sensors, etc.)
@app.post("/events/emit")
async def emit_event(event: dict):
    await spatial_events.put(event)
    return {"status": "emitted"}
```

```javascript
// Client: EventSource + MapLibre
const bbox = map.getBounds().toArray().flat().join(',');
const eventSource = new EventSource(`/events/spatial?bbox=${bbox}`);

eventSource.addEventListener('feature:created', (e) => {
  const feature = JSON.parse(e.data);
  const source = map.getSource('realtime');
  const data = source._data;
  data.features.push(feature);
  source.setData(data);
});

eventSource.addEventListener('feature:updated', (e) => {
  const updated = JSON.parse(e.data);
  // Update specific feature in source
});

// Reconnect on map move with new bbox
map.on('moveend', () => {
  eventSource.close();
  const newBbox = map.getBounds().toArray().flat().join(',');
  // Reconnect with new bbox...
});
```

### SSE vs WebSocket Decision Matrix

| Factor | SSE | WebSocket |
|--------|-----|-----------|
| Direction | Server → Client only | Bidirectional |
| Protocol | HTTP/1.1+ | Upgrade to ws:// |
| Auto-reconnect | Built-in | Manual |
| Binary data | No (text only) | Yes |
| Max connections | ~6 per domain (HTTP/1.1) | Unlimited |
| Best for GIS | Notifications, status | Real-time tracking, collaboration |

---

## MQTT for IoT GIS

### Mosquitto + Sensor Network

```yaml
# docker-compose.yml — MQTT sensor infrastructure
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
    volumes:
      - tsdata:/home/postgres/pgdata

  ingest:
    build: ./ingest
    environment:
      MQTT_BROKER: mqtt://mosquitto:1883
      DATABASE_URL: postgresql://postgres:${DB_PASSWORD}@timescaledb/postgres
    depends_on: [mosquitto, timescaledb]

volumes:
  tsdata:
```

```ini
# mosquitto.conf
listener 1883
listener 9001
protocol websockets

allow_anonymous false
password_file /mosquitto/config/passwords

# Topic ACL
acl_file /mosquitto/config/acl
```

```
# mosquitto ACL
# Sensors can publish to their own topic
pattern readwrite sensors/%u/#

# Dashboard can subscribe to all sensors
user dashboard
topic read sensors/#

# Topic structure:
# sensors/{region}/{sensor_type}/{sensor_id}
# sensors/beijing/air_quality/aq001
# sensors/shanghai/temperature/temp042
```

### Ingest Service — MQTT to TimescaleDB + PostGIS

```typescript
// ingest/index.ts
import mqtt from 'mqtt';
import { Pool } from 'pg';

const client = mqtt.connect(process.env.MQTT_BROKER!);
const pool = new Pool({ connectionString: process.env.DATABASE_URL });

// Create hypertable for time-series spatial data
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

// Subscribe to all sensor topics
client.subscribe('sensors/#');

// Batch insert buffer
let batch: any[] = [];
const BATCH_SIZE = 100;

client.on('message', async (topic, message) => {
  const parts = topic.split('/'); // sensors/{region}/{type}/{id}
  const data = JSON.parse(message.toString());

  batch.push({
    time: new Date(data.timestamp || Date.now()),
    sensor_id: parts[3],
    sensor_type: parts[2],
    value: data.value,
    lng: data.longitude,
    lat: data.latitude,
  });

  if (batch.length >= BATCH_SIZE) {
    const values = batch.map((b, i) => {
      const offset = i * 5;
      return `($${offset+1}, $${offset+2}, $${offset+3}, $${offset+4}, ST_SetSRID(ST_MakePoint($${offset+5}, $${offset+6}), 4326))`;
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

### Browser MQTT Client + MapLibre

```javascript
// Dashboard: MQTT over WebSocket + MapLibre
import mqtt from 'mqtt';

const client = mqtt.connect('ws://localhost:9001');

client.subscribe('sensors/+/air_quality/+');

const sensorData = new Map();

client.on('message', (topic, message) => {
  const data = JSON.parse(message.toString());
  const sensorId = topic.split('/')[3];

  sensorData.set(sensorId, {
    ...data,
    id: sensorId,
    timestamp: Date.now(),
  });

  // Update MapLibre source
  const geojson = {
    type: 'FeatureCollection',
    features: Array.from(sensorData.values()).map((s) => ({
      type: 'Feature',
      properties: {
        id: s.id,
        value: s.value,
        aqi: getAQILevel(s.value),
      },
      geometry: {
        type: 'Point',
        coordinates: [s.longitude, s.latitude],
      },
    })),
  };

  map.getSource('sensors')?.setData(geojson);
});

function getAQILevel(pm25) {
  if (pm25 <= 50) return 'good';
  if (pm25 <= 100) return 'moderate';
  if (pm25 <= 150) return 'unhealthy-sensitive';
  return 'unhealthy';
}
```

---

## Collaborative Map Editing

### Yjs CRDT + MapLibre Draw

```typescript
// Collaborative drawing with conflict-free replicated data
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import MapboxDraw from '@mapbox/mapbox-gl-draw';

// Shared document
const ydoc = new Y.Doc();
const provider = new WebsocketProvider('wss://sync.example.com', 'map-room-123', ydoc);

// Shared GeoJSON features
const yFeatures = ydoc.getMap('features');

// Awareness (cursor positions, user info)
const awareness = provider.awareness;
awareness.setLocalState({
  user: { name: 'Alice', color: '#ff0000' },
  cursor: null,
  viewport: null,
});

// Initialize MapboxDraw
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
      const existing = draw.get(key);
      if (!existing) {
        draw.add(feature);
      } else {
        draw.add(feature); // Replace
      }
    } else if (change.action === 'delete') {
      draw.delete(key);
    }
  });
});

// Share cursor position on map
map.on('mousemove', (e) => {
  awareness.setLocalStateField('cursor', {
    lng: e.lngLat.lng,
    lat: e.lngLat.lat,
  });
});

// Show other users' cursors
awareness.on('change', () => {
  const states = Array.from(awareness.getStates().entries());
  const cursors = states
    .filter(([id]) => id !== ydoc.clientID)
    .map(([, state]) => state)
    .filter((s) => s.cursor);

  // Update cursor layer
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

### Supabase Realtime for Spatial Tables

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
      case 'INSERT': {
        // Add new feature to map
        const feature = payload.new;
        addFeatureToMap(feature);
        break;
      }
      case 'UPDATE': {
        updateFeatureOnMap(payload.new);
        break;
      }
      case 'DELETE': {
        removeFeatureFromMap(payload.old.id);
        break;
      }
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

// Update presence on map move
map.on('moveend', () => {
  presenceChannel.track({
    user_id: currentUser.id,
    name: currentUser.name,
    viewport: map.getBounds().toArray(),
  });
});
```

---

## Streaming & Change Data Capture

### PostGIS LISTEN/NOTIFY

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
        # Forward to WebSocket clients, invalidate tile cache, etc.
        await invalidate_tiles_for_bbox(data['bbox'])

    await conn.add_listener('spatial_changes', handler)

    # Keep alive
    while True:
        await asyncio.sleep(1)

asyncio.run(listen_spatial_changes())
```

### Debezium CDC for Spatial Tables

```json
// Debezium connector config for PostGIS
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

### GraphQL Subscriptions

```typescript
// GraphQL subscription for spatial data changes
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { useServer } from 'graphql-ws/lib/use/ws';
import { makeExecutableSchema } from '@graphql-tools/schema';

const typeDefs = `
  type Feature {
    id: ID!
    name: String!
    geometry: JSON!
  }

  type Subscription {
    featureChanged(bbox: [Float!]): Feature!
  }
`;

const resolvers = {
  Subscription: {
    featureChanged: {
      subscribe: async function* (_, { bbox }) {
        // Listen to PostGIS NOTIFY
        const conn = await pool.connect();
        await conn.query("LISTEN spatial_changes");

        try {
          while (true) {
            const notification = await waitForNotification(conn);
            const data = JSON.parse(notification.payload);

            // Filter by bbox if provided
            if (bbox && !bboxIntersects(data.bbox, bbox)) continue;

            const feature = await getFeature(data.id);
            yield { featureChanged: feature };
          }
        } finally {
          conn.release();
        }
      },
    },
  },
};
```

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

### Geofencing — Real-Time Point-in-Polygon

```python
# In-memory geofence checker with R-tree
from rtree import index
from shapely.geometry import shape, Point
import json

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
                results.append({
                    'zone_id': zone_id,
                    'zone_name': zone['properties'].get('name'),
                })
        return results

# Usage
gf = GeofenceManager()
gf.add_zone('zone1', zone1_geojson)

# On each GPS update:
zones = gf.check_point(116.4, 39.9)
if zones:
    emit_geofence_alert(vehicle_id, zones)
```

### Rate Limiting Spatial Streams

```typescript
// Throttle GPS updates to prevent overwhelming clients
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

### Connection Management

```typescript
// Robust WebSocket client with reconnection
class SpatialWebSocket {
  private ws: WebSocket | null = null;
  private url: string;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private heartbeatInterval: NodeJS.Timeout | null = null;

  constructor(url: string) {
    this.url = url;
    this.connect();
  }

  private connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onopen = () => {
      console.log('Connected');
      this.reconnectDelay = 1000; // Reset delay
      this.startHeartbeat();
    };

    this.ws.onclose = () => {
      this.stopHeartbeat();
      console.log(`Reconnecting in ${this.reconnectDelay}ms...`);
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'pong') return; // Heartbeat response
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
