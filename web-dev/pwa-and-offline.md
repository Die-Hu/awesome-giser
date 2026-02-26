# PWA & Offline Maps — 2025 Complete Guide

Progressive Web Apps enable map applications to work offline, load instantly, and feel native. This guide covers Service Worker strategies for tile caching, offline vector/raster data storage, data synchronization, and native-like features for field survey and mobile GIS applications.

> **Quick Picks**
> - **Tile caching:** Workbox 7 with cache-first strategy
> - **Offline tiles:** PMTiles in IndexedDB or Cache API
> - **Offline vectors:** IndexedDB (Dexie.js) + client-side R-tree (rbush)
> - **Data sync:** Background Sync API + conflict resolution
> - **SQL in browser:** DuckDB-WASM or sql.js for offline spatial queries
> - **Field survey:** Geolocation API + offline storage + sync on reconnect

---

## PWA Fundamentals for Map Apps

### Web App Manifest

```json
{
  "name": "GIS Field Survey",
  "short_name": "FieldGIS",
  "description": "Offline-capable map application for field surveys",
  "start_url": "/map",
  "display": "standalone",
  "orientation": "any",
  "theme_color": "#1a73e8",
  "background_color": "#ffffff",
  "icons": [
    { "src": "/icons/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icons/icon-512.png", "sizes": "512x512", "type": "image/png" },
    { "src": "/icons/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable" }
  ],
  "categories": ["utilities", "productivity"],
  "screenshots": [
    { "src": "/screenshots/map-wide.png", "sizes": "1280x720", "type": "image/png", "form_factor": "wide" },
    { "src": "/screenshots/map-narrow.png", "sizes": "750x1334", "type": "image/png", "form_factor": "narrow" }
  ]
}
```

### Service Worker — Tile Caching with Workbox

```javascript
// sw.js — Complete Service Worker for map app
import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { CacheFirst, StaleWhileRevalidate, NetworkFirst } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';
import { BackgroundSyncPlugin } from 'workbox-background-sync';

// Precache app shell
precacheAndRoute(self.__WB_MANIFEST);

// 1. Basemap tiles — Cache first (rarely change)
registerRoute(
  ({ url }) => url.pathname.match(/\/tiles\/basemap\/\d+\/\d+\/\d+/),
  new CacheFirst({
    cacheName: 'basemap-tiles',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] }),
      new ExpirationPlugin({
        maxEntries: 20000,    // ~200MB at 10KB/tile
        maxAgeSeconds: 90 * 24 * 60 * 60, // 90 days
        purgeOnQuotaError: true,
      }),
    ],
  })
);

// 2. Data tiles — Stale-while-revalidate (may change)
registerRoute(
  ({ url }) => url.pathname.match(/\/tiles\/data\/\d+\/\d+\/\d+/),
  new StaleWhileRevalidate({
    cacheName: 'data-tiles',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] }),
      new ExpirationPlugin({
        maxEntries: 10000,
        maxAgeSeconds: 24 * 60 * 60, // 1 day
      }),
    ],
  })
);

// 3. PMTiles files — Cache first (range requests)
registerRoute(
  ({ url }) => url.pathname.endsWith('.pmtiles'),
  new CacheFirst({
    cacheName: 'pmtiles-files',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200, 206] }),
      new ExpirationPlugin({ maxEntries: 10 }),
    ],
  })
);

// 4. Feature API — Network first with offline fallback
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/features'),
  new NetworkFirst({
    cacheName: 'feature-api',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 500,
        maxAgeSeconds: 7 * 24 * 60 * 60, // 1 week
      }),
    ],
    networkTimeoutSeconds: 5,
  })
);

// 5. Map fonts and sprites — Cache first
registerRoute(
  ({ url }) => url.pathname.match(/\/(font|sprite)\//),
  new CacheFirst({
    cacheName: 'map-assets',
    plugins: [
      new ExpirationPlugin({ maxAgeSeconds: 365 * 24 * 60 * 60 }),
    ],
  })
);

// 6. Background sync for offline edits
const bgSyncPlugin = new BackgroundSyncPlugin('feature-edits-queue', {
  maxRetentionTime: 7 * 24 * 60, // 7 days in minutes
});

registerRoute(
  ({ url, request }) =>
    url.pathname.startsWith('/api/features') && ['POST', 'PUT', 'DELETE'].includes(request.method),
  new NetworkFirst({
    plugins: [bgSyncPlugin],
  }),
  'POST'
);

// Offline notification
self.addEventListener('fetch', (event) => {
  if (!navigator.onLine) {
    // Notify app we're offline
    self.clients.matchAll().then((clients) => {
      clients.forEach((client) => client.postMessage({ type: 'OFFLINE' }));
    });
  }
});
```

### Register Service Worker

```javascript
// main.js
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    const registration = await navigator.serviceWorker.register('/sw.js');

    // Listen for updates
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      newWorker?.addEventListener('statechange', () => {
        if (newWorker.state === 'activated') {
          // Show "update available" toast
          showUpdateNotification();
        }
      });
    });

    // Listen for offline/online messages from SW
    navigator.serviceWorker.addEventListener('message', (event) => {
      if (event.data.type === 'OFFLINE') {
        showOfflineBanner();
      }
    });
  });
}
```

---

## Offline Tile Storage

### Tile Download Manager

```typescript
// lib/tileDownloader.ts — Download tiles for offline use
import { openDB, DBSchema } from 'idb';

interface TileDB extends DBSchema {
  tiles: {
    key: string; // "{source}/{z}/{x}/{y}"
    value: {
      key: string;
      data: ArrayBuffer;
      contentType: string;
      downloadedAt: number;
      size: number;
    };
  };
  regions: {
    key: string;
    value: {
      name: string;
      bbox: [number, number, number, number];
      minZoom: number;
      maxZoom: number;
      tileCount: number;
      totalSize: number;
      downloadedAt: number;
      status: 'pending' | 'downloading' | 'complete' | 'error';
    };
  };
}

const dbPromise = openDB<TileDB>('offline-tiles', 1, {
  upgrade(db) {
    db.createObjectStore('tiles', { keyPath: 'key' });
    db.createObjectStore('regions', { keyPath: 'name' });
  },
});

export class TileDownloader {
  private abortController: AbortController | null = null;

  // Calculate tiles in a bbox at zoom levels
  static tilesInBBox(
    bbox: [number, number, number, number],
    minZoom: number,
    maxZoom: number
  ): { z: number; x: number; y: number }[] {
    const tiles: { z: number; x: number; y: number }[] = [];

    for (let z = minZoom; z <= maxZoom; z++) {
      const n = Math.pow(2, z);
      const xMin = Math.floor(((bbox[0] + 180) / 360) * n);
      const xMax = Math.floor(((bbox[2] + 180) / 360) * n);
      const yMin = Math.floor((1 - Math.log(Math.tan((bbox[3] * Math.PI) / 180) +
        1 / Math.cos((bbox[3] * Math.PI) / 180)) / Math.PI) / 2 * n);
      const yMax = Math.floor((1 - Math.log(Math.tan((bbox[1] * Math.PI) / 180) +
        1 / Math.cos((bbox[1] * Math.PI) / 180)) / Math.PI) / 2 * n);

      for (let x = xMin; x <= xMax; x++) {
        for (let y = yMin; y <= yMax; y++) {
          tiles.push({ z, x, y });
        }
      }
    }

    return tiles;
  }

  // Estimate download size
  static estimateSize(tileCount: number, avgTileSizeKB: number = 15): string {
    const totalMB = (tileCount * avgTileSizeKB) / 1024;
    if (totalMB < 1) return `${Math.round(tileCount * avgTileSizeKB)} KB`;
    if (totalMB < 1024) return `${totalMB.toFixed(1)} MB`;
    return `${(totalMB / 1024).toFixed(1)} GB`;
  }

  async downloadRegion(
    name: string,
    tileUrl: string,  // e.g. "https://tiles.example.com/{z}/{x}/{y}.mvt"
    bbox: [number, number, number, number],
    minZoom: number,
    maxZoom: number,
    onProgress?: (downloaded: number, total: number) => void
  ): Promise<void> {
    this.abortController = new AbortController();
    const db = await dbPromise;
    const tiles = TileDownloader.tilesInBBox(bbox, minZoom, maxZoom);

    // Save region metadata
    await db.put('regions', {
      name, bbox, minZoom, maxZoom,
      tileCount: tiles.length,
      totalSize: 0,
      downloadedAt: Date.now(),
      status: 'downloading',
    });

    let downloaded = 0;
    let totalSize = 0;
    const concurrency = 6;

    // Download with concurrency limit
    const queue = [...tiles];
    const workers = Array.from({ length: concurrency }, async () => {
      while (queue.length > 0) {
        if (this.abortController?.signal.aborted) return;

        const tile = queue.shift()!;
        const url = tileUrl
          .replace('{z}', String(tile.z))
          .replace('{x}', String(tile.x))
          .replace('{y}', String(tile.y));
        const key = `${name}/${tile.z}/${tile.x}/${tile.y}`;

        try {
          const response = await fetch(url, { signal: this.abortController?.signal });
          if (!response.ok) continue;

          const data = await response.arrayBuffer();
          const contentType = response.headers.get('content-type') || 'application/x-protobuf';

          await db.put('tiles', {
            key, data, contentType,
            downloadedAt: Date.now(),
            size: data.byteLength,
          });

          downloaded++;
          totalSize += data.byteLength;
          onProgress?.(downloaded, tiles.length);
        } catch (err) {
          if ((err as Error).name === 'AbortError') return;
          console.warn(`Failed to download tile ${key}:`, err);
        }
      }
    });

    await Promise.all(workers);

    // Update region status
    await db.put('regions', {
      name, bbox, minZoom, maxZoom,
      tileCount: tiles.length,
      totalSize,
      downloadedAt: Date.now(),
      status: this.abortController?.signal.aborted ? 'error' : 'complete',
    });
  }

  cancel() {
    this.abortController?.abort();
  }

  async getRegions() {
    const db = await dbPromise;
    return db.getAll('regions');
  }

  async deleteRegion(name: string) {
    const db = await dbPromise;
    const tx = db.transaction(['tiles', 'regions'], 'readwrite');
    // Delete all tiles for this region
    const allKeys = await tx.objectStore('tiles').getAllKeys();
    for (const key of allKeys) {
      if (key.toString().startsWith(`${name}/`)) {
        await tx.objectStore('tiles').delete(key);
      }
    }
    await tx.objectStore('regions').delete(name);
    await tx.done;
  }
}
```

### Service Worker Tile Interceptor

```javascript
// In sw.js — Serve tiles from IndexedDB when offline
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Intercept tile requests
  if (url.pathname.match(/\/\d+\/\d+\/\d+\.(mvt|pbf|png)/)) {
    event.respondWith(
      (async () => {
        // Try network first
        try {
          const response = await fetch(event.request);
          return response;
        } catch {
          // Offline: check IndexedDB
          const db = await openDB('offline-tiles', 1);
          const key = url.pathname; // or build key from z/x/y

          const cached = await db.get('tiles', key);
          if (cached) {
            return new Response(cached.data, {
              headers: { 'Content-Type': cached.contentType },
            });
          }

          return new Response('', { status: 204 }); // No tile available
        }
      })()
    );
  }
});
```

### PMTiles Offline — Full File Download

```typescript
// Download entire PMTiles file for offline use
async function downloadPMTilesForOffline(url: string, name: string, onProgress?: (pct: number) => void) {
  const response = await fetch(url);
  const contentLength = Number(response.headers.get('content-length') || 0);
  const reader = response.body!.getReader();

  const chunks: Uint8Array[] = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.((received / contentLength) * 100);
  }

  // Combine chunks
  const blob = new Blob(chunks);

  // Store in Cache API (supports large files better than IndexedDB)
  const cache = await caches.open('pmtiles-offline');
  await cache.put(
    new Request(`offline-pmtiles://${name}`),
    new Response(blob, { headers: { 'Content-Type': 'application/octet-stream' } })
  );
}

// Load offline PMTiles
import { PMTiles } from 'pmtiles';

class OfflinePMTilesSource {
  constructor(private name: string) {}

  async getBytes(offset: number, length: number) {
    const cache = await caches.open('pmtiles-offline');
    const response = await cache.match(`offline-pmtiles://${this.name}`);
    if (!response) throw new Error('PMTiles not found in cache');

    const blob = await response.blob();
    const slice = blob.slice(offset, offset + length);
    const buffer = await slice.arrayBuffer();
    return { data: new Uint8Array(buffer) };
  }
}

// Use with MapLibre
const offlineSource = new OfflinePMTilesSource('basemap');
const pmtiles = new PMTiles(offlineSource);
// Register as pmtiles protocol source...
```

---

## Offline Vector Data

### IndexedDB for GeoJSON with Dexie.js

```typescript
// lib/offlineFeatures.ts
import Dexie, { Table } from 'dexie';

interface OfflineFeature {
  id?: number;
  serverId?: string;        // Server-side ID (null for local-only)
  name: string;
  category: string;
  geometry: any;             // GeoJSON geometry
  bbox: [number, number, number, number]; // For spatial queries
  syncStatus: 'synced' | 'created' | 'modified' | 'deleted';
  lastModified: number;
}

class GeoDatabase extends Dexie {
  features!: Table<OfflineFeature>;

  constructor() {
    super('geo-database');
    this.version(1).stores({
      features: '++id, serverId, category, syncStatus, [bbox.0+bbox.1+bbox.2+bbox.3]',
    });
  }
}

const db = new GeoDatabase();

// Spatial query using bbox filtering
async function getFeaturesInBBox(bbox: [number, number, number, number]) {
  const allFeatures = await db.features
    .where('syncStatus').notEqual('deleted')
    .toArray();

  // Client-side bbox filter
  return allFeatures.filter((f) =>
    f.bbox[0] <= bbox[2] && f.bbox[2] >= bbox[0] &&
    f.bbox[1] <= bbox[3] && f.bbox[3] >= bbox[1]
  );
}

// Add feature (offline-first)
async function addFeature(geojson: any) {
  const bbox = turf.bbox(geojson);
  await db.features.add({
    name: geojson.properties.name,
    category: geojson.properties.category,
    geometry: geojson.geometry,
    bbox: bbox as [number, number, number, number],
    syncStatus: 'created',
    lastModified: Date.now(),
  });
}

// Get pending changes for sync
async function getPendingChanges() {
  return db.features
    .where('syncStatus')
    .anyOf(['created', 'modified', 'deleted'])
    .toArray();
}
```

### Client-Side R-tree with rbush

```typescript
// Spatial index for fast client-side queries
import RBush from 'rbush';

interface SpatialItem {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  id: number;
  feature: any;
}

class SpatialIndex {
  private tree = new RBush<SpatialItem>();

  loadFeatures(geojson: any) {
    const items: SpatialItem[] = geojson.features.map((f: any, i: number) => {
      const bbox = turf.bbox(f);
      return {
        minX: bbox[0], minY: bbox[1],
        maxX: bbox[2], maxY: bbox[3],
        id: i, feature: f,
      };
    });
    this.tree.load(items);
  }

  queryBBox(bbox: [number, number, number, number]): any[] {
    const results = this.tree.search({
      minX: bbox[0], minY: bbox[1],
      maxX: bbox[2], maxY: bbox[3],
    });
    return results.map((r) => r.feature);
  }

  queryNearby(lng: number, lat: number, radiusDeg: number): any[] {
    return this.queryBBox([
      lng - radiusDeg, lat - radiusDeg,
      lng + radiusDeg, lat + radiusDeg,
    ]);
  }
}

const index = new SpatialIndex();
index.loadFeatures(geojson);
const nearby = index.queryBBox([116.3, 39.8, 116.5, 40.0]);
```

---

## Data Synchronization

### Offline-First Sync Strategy

```typescript
// lib/syncManager.ts
class SyncManager {
  private syncInProgress = false;

  constructor(
    private apiUrl: string,
    private db: GeoDatabase
  ) {
    // Listen for online events
    window.addEventListener('online', () => this.sync());

    // Register Background Sync
    if ('serviceWorker' in navigator && 'SyncManager' in window) {
      navigator.serviceWorker.ready.then((reg) => {
        return (reg as any).sync.register('sync-features');
      });
    }
  }

  async sync(): Promise<{ uploaded: number; downloaded: number; conflicts: number }> {
    if (this.syncInProgress || !navigator.onLine) {
      return { uploaded: 0, downloaded: 0, conflicts: 0 };
    }

    this.syncInProgress = true;
    let uploaded = 0, downloaded = 0, conflicts = 0;

    try {
      // 1. Upload local changes
      const pendingChanges = await this.db.features
        .where('syncStatus').anyOf(['created', 'modified', 'deleted'])
        .toArray();

      for (const feature of pendingChanges) {
        try {
          switch (feature.syncStatus) {
            case 'created': {
              const res = await fetch(`${this.apiUrl}/features`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  type: 'Feature',
                  geometry: feature.geometry,
                  properties: { name: feature.name, category: feature.category },
                }),
              });
              const data = await res.json();
              await this.db.features.update(feature.id!, {
                serverId: data.id,
                syncStatus: 'synced',
              });
              uploaded++;
              break;
            }
            case 'modified': {
              const res = await fetch(`${this.apiUrl}/features/${feature.serverId}`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                  type: 'Feature',
                  geometry: feature.geometry,
                  properties: { name: feature.name, category: feature.category },
                }),
              });

              if (res.status === 409) {
                // Conflict! Server version changed
                conflicts++;
                await this.resolveConflict(feature, await res.json());
              } else {
                await this.db.features.update(feature.id!, { syncStatus: 'synced' });
                uploaded++;
              }
              break;
            }
            case 'deleted': {
              await fetch(`${this.apiUrl}/features/${feature.serverId}`, {
                method: 'DELETE',
              });
              await this.db.features.delete(feature.id!);
              uploaded++;
              break;
            }
          }
        } catch (err) {
          console.warn(`Failed to sync feature ${feature.id}:`, err);
        }
      }

      // 2. Download server changes
      const lastSync = localStorage.getItem('lastSyncTimestamp') || '0';
      const res = await fetch(`${this.apiUrl}/features?modified_since=${lastSync}`);
      const serverFeatures = await res.json();

      for (const sf of serverFeatures.features) {
        const existing = await this.db.features
          .where('serverId').equals(sf.properties.id)
          .first();

        if (!existing) {
          // New feature from server
          const bbox = turf.bbox(sf);
          await this.db.features.add({
            serverId: sf.properties.id,
            name: sf.properties.name,
            category: sf.properties.category,
            geometry: sf.geometry,
            bbox: bbox as [number, number, number, number],
            syncStatus: 'synced',
            lastModified: Date.now(),
          });
          downloaded++;
        } else if (existing.syncStatus === 'synced') {
          // Update from server (no local changes)
          const bbox = turf.bbox(sf);
          await this.db.features.update(existing.id!, {
            name: sf.properties.name,
            geometry: sf.geometry,
            bbox: bbox as [number, number, number, number],
            lastModified: Date.now(),
          });
          downloaded++;
        }
        // If local has changes, keep local version (conflict will resolve on upload)
      }

      localStorage.setItem('lastSyncTimestamp', String(Date.now()));
    } finally {
      this.syncInProgress = false;
    }

    return { uploaded, downloaded, conflicts };
  }

  private async resolveConflict(local: OfflineFeature, server: any) {
    // Strategy: last-write-wins based on timestamp
    // Alternative strategies: manual merge UI, server-wins, etc.
    const serverTime = new Date(server.properties.modified_at).getTime();
    if (local.lastModified > serverTime) {
      // Local wins — retry upload
      await this.db.features.update(local.id!, { syncStatus: 'modified' });
    } else {
      // Server wins — accept server version
      const bbox = turf.bbox(server);
      await this.db.features.update(local.id!, {
        name: server.properties.name,
        geometry: server.geometry,
        bbox: bbox as [number, number, number, number],
        syncStatus: 'synced',
        lastModified: serverTime,
      });
    }
  }
}
```

### Field Survey App — Complete Example

```jsx
// components/FieldSurvey.jsx — Offline-first field data collection
import { useState, useEffect, useCallback } from 'react';
import Map, { Marker, GeolocateControl } from 'react-map-gl/maplibre';

export default function FieldSurvey() {
  const [position, setPosition] = useState(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [pendingCount, setPendingCount] = useState(0);
  const [surveyMode, setSurveyMode] = useState(false);

  useEffect(() => {
    const online = () => setIsOnline(true);
    const offline = () => setIsOnline(false);
    window.addEventListener('online', online);
    window.addEventListener('offline', offline);
    return () => {
      window.removeEventListener('online', online);
      window.removeEventListener('offline', offline);
    };
  }, []);

  // Track GPS position
  useEffect(() => {
    const watchId = navigator.geolocation.watchPosition(
      (pos) => setPosition({
        lng: pos.coords.longitude,
        lat: pos.coords.latitude,
        accuracy: pos.coords.accuracy,
      }),
      (err) => console.error('GPS error:', err),
      { enableHighAccuracy: true, maximumAge: 5000, timeout: 10000 }
    );
    return () => navigator.geolocation.clearWatch(watchId);
  }, []);

  const collectPoint = useCallback(async () => {
    if (!position) return;

    const feature = {
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [position.lng, position.lat] },
      properties: {
        name: `Survey Point ${Date.now()}`,
        category: 'survey',
        accuracy: position.accuracy,
        timestamp: new Date().toISOString(),
      },
    };

    // Save locally (works offline)
    await addFeature(feature);
    setPendingCount((c) => c + 1);

    // Try to sync if online
    if (navigator.onLine) {
      const result = await syncManager.sync();
      setPendingCount((c) => c - result.uploaded);
    }
  }, [position]);

  return (
    <div className="relative w-full h-screen">
      {/* Status bar */}
      <div className={`absolute top-0 left-0 right-0 z-10 p-2 text-center text-sm
        ${isOnline ? 'bg-green-500' : 'bg-orange-500'} text-white`}>
        {isOnline ? 'Online' : 'Offline'} |
        GPS: {position ? `${position.accuracy.toFixed(0)}m` : 'Acquiring...'} |
        Pending: {pendingCount}
      </div>

      <Map
        initialViewState={{ longitude: 116.4, latitude: 39.9, zoom: 14 }}
        style={{ width: '100%', height: '100%' }}
        mapStyle="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"
      >
        <GeolocateControl position="top-right" trackUserLocation />
        {position && <Marker longitude={position.lng} latitude={position.lat} color="blue" />}
      </Map>

      {/* Collection button */}
      <button
        onClick={collectPoint}
        disabled={!position}
        className="absolute bottom-8 left-1/2 -translate-x-1/2 z-10
          bg-blue-600 text-white rounded-full w-16 h-16 text-2xl
          disabled:bg-gray-400 shadow-lg"
      >
        +
      </button>
    </div>
  );
}
```

---

## Native-Like Features

### Geolocation API — High Accuracy Tracking

```typescript
class LocationTracker {
  private watchId: number | null = null;
  private positions: GeolocationPosition[] = [];

  start(onPosition: (pos: GeolocationPosition) => void) {
    if (!('geolocation' in navigator)) {
      throw new Error('Geolocation not supported');
    }

    this.watchId = navigator.geolocation.watchPosition(
      (position) => {
        this.positions.push(position);
        onPosition(position);
      },
      (error) => {
        switch (error.code) {
          case error.PERMISSION_DENIED:
            console.error('Location permission denied');
            break;
          case error.POSITION_UNAVAILABLE:
            console.error('Location unavailable');
            break;
          case error.TIMEOUT:
            console.error('Location timeout');
            break;
        }
      },
      {
        enableHighAccuracy: true,
        maximumAge: 5000,
        timeout: 15000,
      }
    );
  }

  stop() {
    if (this.watchId !== null) {
      navigator.geolocation.clearWatch(this.watchId);
      this.watchId = null;
    }
  }

  getTrack(): any {
    return {
      type: 'Feature',
      geometry: {
        type: 'LineString',
        coordinates: this.positions.map((p) => [
          p.coords.longitude,
          p.coords.latitude,
          p.coords.altitude || 0,
        ]),
      },
      properties: {
        timestamps: this.positions.map((p) => p.timestamp),
        accuracies: this.positions.map((p) => p.coords.accuracy),
      },
    };
  }
}
```

### Share Map View

```typescript
// Share current map view using Web Share API
async function shareMapView(map: maplibregl.Map) {
  const center = map.getCenter();
  const zoom = map.getZoom();
  const url = new URL(window.location.href);
  url.searchParams.set('lat', center.lat.toFixed(5));
  url.searchParams.set('lng', center.lng.toFixed(5));
  url.searchParams.set('z', zoom.toFixed(1));

  if (navigator.share) {
    await navigator.share({
      title: 'Map View',
      text: `Location: ${center.lat.toFixed(4)}, ${center.lng.toFixed(4)}`,
      url: url.toString(),
    });
  } else {
    await navigator.clipboard.writeText(url.toString());
    // Show "Link copied" toast
  }
}
```

---

## Performance & Storage

### Storage Quota Management

```typescript
async function checkStorageQuota(): Promise<{
  usage: number;
  quota: number;
  percent: number;
  available: number;
}> {
  if ('storage' in navigator && 'estimate' in navigator.storage) {
    const estimate = await navigator.storage.estimate();
    const usage = estimate.usage || 0;
    const quota = estimate.quota || 0;
    return {
      usage,
      quota,
      percent: quota > 0 ? (usage / quota) * 100 : 0,
      available: quota - usage,
    };
  }
  return { usage: 0, quota: 0, percent: 0, available: 0 };
}

// Request persistent storage (prevents browser from evicting data)
async function requestPersistentStorage(): Promise<boolean> {
  if ('storage' in navigator && 'persist' in navigator.storage) {
    return navigator.storage.persist();
  }
  return false;
}

// Format bytes for display
function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`;
}
```

### Network-Aware Tile Loading

```typescript
// Adapt tile quality based on network conditions
function getOptimalTileStrategy(): 'full' | 'reduced' | 'offline-only' {
  const connection = (navigator as any).connection;

  if (!navigator.onLine) return 'offline-only';

  if (connection) {
    // Slow 2G or save-data mode
    if (connection.saveData || connection.effectiveType === 'slow-2g') {
      return 'offline-only';
    }
    // 2G or 3G
    if (['2g', '3g'].includes(connection.effectiveType)) {
      return 'reduced'; // Lower zoom, fewer layers
    }
  }

  return 'full';
}

// Apply strategy to map
function applyTileStrategy(map: maplibregl.Map) {
  const strategy = getOptimalTileStrategy();

  switch (strategy) {
    case 'offline-only':
      // Only use cached tiles
      map.setMaxZoom(14);
      break;
    case 'reduced':
      // Skip high-detail layers
      map.setLayoutProperty('buildings-3d', 'visibility', 'none');
      map.setLayoutProperty('poi-labels', 'visibility', 'none');
      break;
    case 'full':
      // Full experience
      break;
  }
}
```

### Storage Budget Guidelines

| Data Type | Typical Size | Recommended Limit | Notes |
|-----------|-------------|-------------------|-------|
| Basemap tiles (z0-14) | 50-200 MB per region | 500 MB | Cache most-used zoom levels |
| Data tiles | 10-50 MB per dataset | 200 MB | Evict oldest first |
| PMTiles file | 5-500 MB per dataset | 1 GB | Use persistent storage |
| Offline features | 1-50 MB per collection | 100 MB | IndexedDB with Dexie |
| Fonts + sprites | 5-20 MB | 50 MB | Cache indefinitely |
| **Total budget** | — | **~2 GB** | Request persistent storage |
