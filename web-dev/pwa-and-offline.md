# PWA & Offline Maps -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**Tile caching:** Workbox with cache-first for basemaps, stale-while-revalidate for data tiles. **Offline vector storage:** IndexedDB via Dexie.js + client-side R-tree (rbush) for spatial queries. **Data sync:** Background Sync API + last-write-wins conflict resolution (or manual merge for field surveys). **Field survey:** Geolocation API + offline storage + sync on reconnect. **Storage budget:** ~2GB total, request persistent storage via `navigator.storage.persist()`.

---

## Tier 1 -- Production First Choices

---

### Workbox -- Service Worker Toolkit

Google-maintained service worker toolkit. The industry standard for PWA caching strategies. Handles tile caching, API response caching, background sync for offline edits, and precaching of app shell assets.

**Why Tier 1:** The only mature option for offline tile caching in PWAs. Plugin architecture handles tile caching, API caching, and asset precaching in a single consistent framework.

```javascript
// sw.js -- Complete Service Worker for map app
import { precacheAndRoute } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { CacheFirst, StaleWhileRevalidate, NetworkFirst } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';
import { CacheableResponsePlugin } from 'workbox-cacheable-response';
import { BackgroundSyncPlugin } from 'workbox-background-sync';

// Precache app shell
precacheAndRoute(self.__WB_MANIFEST);

// 1. Basemap tiles -- Cache first (rarely change)
registerRoute(
  ({ url }) => url.pathname.match(/\/tiles\/basemap\/\d+\/\d+\/\d+/),
  new CacheFirst({
    cacheName: 'basemap-tiles',
    plugins: [
      new CacheableResponsePlugin({ statuses: [0, 200] }),
      new ExpirationPlugin({
        maxEntries: 20000,    // ~200MB at 10KB/tile
        maxAgeSeconds: 90 * 24 * 60 * 60, // 90 days
        purgeOnQuotaError: true, // CRITICAL: prevents storage quota crashes
      }),
    ],
  })
);

// 2. Data tiles -- Stale-while-revalidate (may change)
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

// 3. PMTiles files -- Cache first (range requests)
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

// 4. Feature API -- Network first with offline fallback
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

// 5. Map fonts and sprites -- Cache first (immutable)
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
```

```javascript
// main.js -- Service Worker registration
if ('serviceWorker' in navigator) {
  window.addEventListener('load', async () => {
    const registration = await navigator.serviceWorker.register('/sw.js');

    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      newWorker?.addEventListener('statechange', () => {
        if (newWorker.state === 'activated') {
          showUpdateNotification();
        }
      });
    });

    navigator.serviceWorker.addEventListener('message', (event) => {
      if (event.data.type === 'OFFLINE') {
        showOfflineBanner();
      }
    });
  });
}
```

**Caveats:**
- **Cache invalidation is hard.** Stale tiles can persist indefinitely with CacheFirst strategy. Users see old data. Must implement cache versioning or max-age limits.
- **Storage quotas vary wildly.** Chrome ~60% of disk, Firefox ~50%, Safari ~1GB. Exceeding quota causes silent failures. Must use `purgeOnQuotaError: true`.
- **SW update race conditions.** When deploying a new service worker version, the old SW serves stale content until all tabs are closed. `skipWaiting()` + `clientsClaim()` can cause mid-session inconsistencies.
- **Range request caching.** PMTiles files use HTTP Range requests. Default Workbox strategies don't cache partial responses correctly. Need custom handler for `206 Partial Content`.
- **Debugging difficulty.** Service worker issues are hard to reproduce. The SW runs in a separate context from the page. Chrome DevTools "Application" tab is essential but still awkward.
- **Anti-pattern: Caching everything.** Caching all tiles at all zoom levels fills storage quickly. Use zoom-level-aware caching (cache z0-14 for basemaps, only visible viewport tiles for data layers).

---

### IndexedDB / Dexie.js -- Offline Vector Storage

IndexedDB is the browser's primary storage for structured offline data. Dexie.js provides a clean Promise-based API on top of it. Together they store GeoJSON features, survey data, and sync metadata for offline-first applications.

**Why Tier 1:** The best IndexedDB wrapper available. For enterprise field survey applications, combine with rbush for spatial queries and a proper sync manager for conflict resolution. Request persistent storage to prevent browser from evicting offline data.

```typescript
// lib/offlineFeatures.ts
import Dexie, { Table } from 'dexie';

interface OfflineFeature {
  id?: number;
  serverId?: string;
  name: string;
  category: string;
  geometry: any;             // GeoJSON geometry
  bbox: [number, number, number, number];
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

  // Client-side bbox filter (IndexedDB has no spatial indexing)
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

**Caveats:**
- **No spatial indexing.** IndexedDB has no concept of spatial queries. All spatial filtering must be done in JavaScript after retrieving records. For 100K+ features, this is slow. Pair with rbush for spatial queries.
- **Storage limits vary by browser.** Safari on iOS limits to ~1GB. Android WebView limits vary by device manufacturer.
- **Transaction deadlocks.** Complex read-write transactions across multiple stores can deadlock. Dexie mitigates this but doesn't eliminate it.
- **Data corruption on mobile.** iOS Safari has known IndexedDB corruption bugs, especially on low storage devices. Always have a server-side backup strategy.
- **No built-in sync.** Must implement sync logic manually. Conflict resolution is domain-specific and error-prone.

---

### rbush -- Client-Side R-tree Spatial Index

In-memory R-tree spatial index for fast client-side bbox and proximity queries. Created by Mapbox, ~6KB gzipped, zero dependencies.

**Why Tier 1:** Essential companion to IndexedDB for offline spatial applications. Use with `flatbush` (static, immutable variant) for read-only datasets -- it's 2x faster and uses less memory.

```typescript
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
    this.tree.load(items); // Bulk load: O(n log n), much faster than individual inserts
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

**Caveats:**
- **Memory overhead.** ~80 bytes per item. 1M features = 80MB just for the index. On mobile devices, this can be prohibitive.
- **Insertion performance.** Bulk loading (`load()`) is O(n log n) and fast. Individual `insert()` calls are O(log n) each but cause tree rebalancing. For streaming data, batch inserts periodically.
- **Bbox-only queries.** rbush only queries by bounding box. Precise point-in-polygon or distance queries require a post-filter step with Turf.js.
- **No persistence.** In-memory only. Must rebuild from IndexedDB data on page reload.

---

## Offline Tile Storage

### Tile Download Manager

```typescript
// lib/tileDownloader.ts -- Download tiles for offline use
import { openDB, DBSchema } from 'idb';

interface TileDB extends DBSchema {
  tiles: {
    key: string;
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

  static estimateSize(tileCount: number, avgTileSizeKB: number = 15): string {
    const totalMB = (tileCount * avgTileSizeKB) / 1024;
    if (totalMB < 1) return `${Math.round(tileCount * avgTileSizeKB)} KB`;
    if (totalMB < 1024) return `${totalMB.toFixed(1)} MB`;
    return `${(totalMB / 1024).toFixed(1)} GB`;
  }

  async downloadRegion(
    name: string,
    tileUrl: string,
    bbox: [number, number, number, number],
    minZoom: number,
    maxZoom: number,
    onProgress?: (downloaded: number, total: number) => void
  ): Promise<void> {
    this.abortController = new AbortController();
    const db = await dbPromise;
    const tiles = TileDownloader.tilesInBBox(bbox, minZoom, maxZoom);

    await db.put('regions', {
      name, bbox, minZoom, maxZoom,
      tileCount: tiles.length,
      totalSize: 0,
      downloadedAt: Date.now(),
      status: 'downloading',
    });

    let downloaded = 0;
    let totalSize = 0;
    const concurrency = 6; // Browser limits ~6 connections per domain

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

          await db.put('tiles', { key, data, contentType, downloadedAt: Date.now(), size: data.byteLength });
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

    await db.put('regions', {
      name, bbox, minZoom, maxZoom,
      tileCount: tiles.length, totalSize, downloadedAt: Date.now(),
      status: this.abortController?.signal.aborted ? 'error' : 'complete',
    });
  }

  cancel() { this.abortController?.abort(); }
}
```

**Caveats:**
- **Tile count grows exponentially with zoom.** A small city at z0-14 might be 50K tiles (500MB). A large region at z0-16 could be millions of tiles (tens of GB). Always show the user estimated download size before starting.
- **Browser connection limits.** ~6 concurrent HTTP connections per domain. Higher concurrency than 6 won't help and may cause timeouts.
- **iOS Safari limitations.** Smaller storage quota (~1GB), stricter cache eviction, no Background Sync API. The worst platform for offline maps.

---

## Data Synchronization

### Offline-First Sync Strategy

```typescript
// lib/syncManager.ts
class SyncManager {
  private syncInProgress = false;

  constructor(private apiUrl: string, private db: GeoDatabase) {
    window.addEventListener('online', () => this.sync());

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
                serverId: data.id, syncStatus: 'synced',
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
                conflicts++;
                await this.resolveConflict(feature, await res.json());
              } else {
                await this.db.features.update(feature.id!, { syncStatus: 'synced' });
                uploaded++;
              }
              break;
            }
            case 'deleted': {
              await fetch(`${this.apiUrl}/features/${feature.serverId}`, { method: 'DELETE' });
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
          .where('serverId').equals(sf.properties.id).first();

        if (!existing) {
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
          const bbox = turf.bbox(sf);
          await this.db.features.update(existing.id!, {
            name: sf.properties.name,
            geometry: sf.geometry,
            bbox: bbox as [number, number, number, number],
            lastModified: Date.now(),
          });
          downloaded++;
        }
      }

      localStorage.setItem('lastSyncTimestamp', String(Date.now()));
    } finally {
      this.syncInProgress = false;
    }

    return { uploaded, downloaded, conflicts };
  }

  private async resolveConflict(local: OfflineFeature, server: any) {
    // Strategy: last-write-wins based on timestamp
    const serverTime = new Date(server.properties.modified_at).getTime();
    if (local.lastModified > serverTime) {
      await this.db.features.update(local.id!, { syncStatus: 'modified' });
    } else {
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

**Caveats:**
- **Conflict resolution is domain-specific.** Last-write-wins is simple but can lose data. For field surveys where multiple users edit the same feature, consider server-wins or manual merge UI.
- **Background Sync API is not universal.** Not available on iOS Safari. Must implement manual sync-on-reconnect as fallback.
- **Sync ordering matters.** Upload local changes before downloading server changes to minimize conflicts.

---

## Field Survey Pattern

### Offline-First Data Collection

```jsx
// components/FieldSurvey.jsx
import { useState, useEffect, useCallback } from 'react';
import Map, { Marker, GeolocateControl } from 'react-map-gl/maplibre';

export default function FieldSurvey() {
  const [position, setPosition] = useState(null);
  const [isOnline, setIsOnline] = useState(navigator.onLine);
  const [pendingCount, setPendingCount] = useState(0);

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

    await addFeature({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [position.lng, position.lat] },
      properties: {
        name: `Survey Point ${Date.now()}`,
        category: 'survey',
        accuracy: position.accuracy,
        timestamp: new Date().toISOString(),
      },
    });
    setPendingCount((c) => c + 1);

    if (navigator.onLine) {
      const result = await syncManager.sync();
      setPendingCount((c) => c - result.uploaded);
    }
  }, [position]);

  return (
    <div className="relative w-full h-screen">
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

## Performance & Storage

### Storage Quota Management

```typescript
async function checkStorageQuota(): Promise<{
  usage: number; quota: number; percent: number; available: number;
}> {
  if ('storage' in navigator && 'estimate' in navigator.storage) {
    const estimate = await navigator.storage.estimate();
    const usage = estimate.usage || 0;
    const quota = estimate.quota || 0;
    return {
      usage, quota,
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
```

### Network-Aware Tile Loading

```typescript
function getOptimalTileStrategy(): 'full' | 'reduced' | 'offline-only' {
  const connection = (navigator as any).connection;

  if (!navigator.onLine) return 'offline-only';

  if (connection) {
    if (connection.saveData || connection.effectiveType === 'slow-2g') return 'offline-only';
    if (['2g', '3g'].includes(connection.effectiveType)) return 'reduced';
  }

  return 'full';
}

function applyTileStrategy(map: maplibregl.Map) {
  const strategy = getOptimalTileStrategy();

  switch (strategy) {
    case 'offline-only':
      map.setMaxZoom(14);
      break;
    case 'reduced':
      map.setLayoutProperty('buildings-3d', 'visibility', 'none');
      map.setLayoutProperty('poi-labels', 'visibility', 'none');
      break;
    case 'full':
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
| **Total budget** | -- | **~2 GB** | Request persistent storage |
