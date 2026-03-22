# Testing & Security -- Enterprise Reference

> Data validated: 2026-03-21

## 30-Second Decision

**Unit testing:** Vitest + Turf.js for spatial functions. **Integration testing:** Testcontainers + PostGIS (never mock spatial SQL). **E2E map testing:** Playwright with MapLibre page objects. **Input validation:** Zod for GeoJSON schema validation at API boundaries. **Auth:** JWT + spatial claims (bbox/region) + PostGIS RLS (be aware of performance cost). **Security essentials:** Parameterized ST_ functions, GeoJSON size/coordinate validation, CORS for tile endpoints, bbox area rate limiting.

---

## Tier 1 -- Production First Choices

---

### Vitest -- Unit Testing Spatial Code

The standard unit testing framework for modern JavaScript/TypeScript projects. Fast, Vite-native, ESM-first. For spatial testing, combine with Turf.js for geometry assertions.

**Why Tier 1:** The standard choice for spatial unit testing. Fast execution, excellent TypeScript support, and Turf.js provides a comprehensive spatial assertion toolkit.

```typescript
// __tests__/spatial.test.ts
import { describe, it, expect } from 'vitest';
import * as turf from '@turf/turf';

describe('Spatial utilities', () => {
  const polygon = turf.polygon([[
    [116.3, 39.8], [116.5, 39.8], [116.5, 40.0], [116.3, 40.0], [116.3, 39.8]
  ]]);

  const pointInside = turf.point([116.4, 39.9]);
  const pointOutside = turf.point([117.0, 40.5]);

  it('point-in-polygon returns true for interior point', () => {
    expect(turf.booleanPointInPolygon(pointInside, polygon)).toBe(true);
  });

  it('point-in-polygon returns false for exterior point', () => {
    expect(turf.booleanPointInPolygon(pointOutside, polygon)).toBe(false);
  });

  it('buffer creates valid geometry', () => {
    const buffered = turf.buffer(pointInside, 5, { units: 'kilometers' });
    expect(buffered.geometry.type).toBe('Polygon');
    expect(buffered.geometry.coordinates[0].length).toBeGreaterThan(10);
  });

  it('area calculation is reasonable', () => {
    const areaKm2 = turf.area(polygon) / 1e6;
    expect(areaKm2).toBeGreaterThan(400);  // ~440 km2
    expect(areaKm2).toBeLessThan(500);
  });

  it('centroid is inside polygon', () => {
    const centroid = turf.centroid(polygon);
    expect(turf.booleanPointInPolygon(centroid, polygon)).toBe(true);
  });

  it('simplify reduces vertex count', () => {
    const complex = turf.randomPolygon(1, { max_radial_length: 0.1, num_vertices: 100 });
    const simplified = turf.simplify(complex, { tolerance: 0.01, highQuality: true });
    const originalVertices = complex.features[0].geometry.coordinates[0].length;
    const simplifiedVertices = simplified.features[0].geometry.coordinates[0].length;
    expect(simplifiedVertices).toBeLessThan(originalVertices);
  });

  it('intersection of overlapping polygons returns geometry', () => {
    const poly2 = turf.polygon([[
      [116.4, 39.9], [116.6, 39.9], [116.6, 40.1], [116.4, 40.1], [116.4, 39.9]
    ]]);
    const intersection = turf.intersect(turf.featureCollection([polygon, poly2]));
    expect(intersection).not.toBeNull();
    expect(intersection!.geometry.type).toBe('Polygon');
  });
});
```

**Caveats:**
- **No WebGL in Node.js.** Vitest runs in Node.js (or jsdom). No WebGL context. Cannot test actual map rendering. Must mock the map instance or use E2E tests for visual verification.
- **Spatial test fixtures are tedious.** Creating GeoJSON fixtures for edge cases (antimeridian crossing, polar regions, self-intersecting polygons) requires spatial domain knowledge. No standard fixture library exists for spatial edge cases.
- **Floating-point precision.** Spatial calculations produce floating-point results. Use `toBeCloseTo()` instead of `toBe()` for coordinate and area assertions.

---

### Testcontainers + PostGIS -- Integration Testing

Spin up a real PostGIS database in Docker for integration tests. Tests run against the actual spatial engine -- no mocking spatial SQL.

**Why Tier 1:** The only way to reliably test spatial SQL queries without mocking. Catches wrong `ST_DWithin` vs `ST_Distance` queries, index misuse, and projection bugs that mocks would miss.

```typescript
// __tests__/integration/spatial-queries.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { GenericContainer, StartedTestContainer } from 'testcontainers';
import { Pool } from 'pg';

let container: StartedTestContainer;
let pool: Pool;

beforeAll(async () => {
  container = await new GenericContainer('postgis/postgis:16-3.4')
    .withEnvironment({
      POSTGRES_DB: 'testdb',
      POSTGRES_USER: 'test',
      POSTGRES_PASSWORD: 'test',
    })
    .withExposedPorts(5432)
    .withStartupTimeout(60000)
    .start();

  pool = new Pool({
    host: container.getHost(),
    port: container.getMappedPort(5432),
    database: 'testdb',
    user: 'test',
    password: 'test',
  });

  await pool.query(`
    CREATE TABLE features (
      id SERIAL PRIMARY KEY,
      name TEXT,
      category TEXT,
      geom GEOMETRY(Geometry, 4326)
    );
    CREATE INDEX ON features USING GIST (geom);

    INSERT INTO features (name, category, geom) VALUES
      ('Park A', 'park', ST_GeomFromText('POLYGON((116.3 39.9, 116.35 39.9, 116.35 39.95, 116.3 39.95, 116.3 39.9))', 4326)),
      ('School B', 'school', ST_SetSRID(ST_MakePoint(116.4, 39.92), 4326)),
      ('Hospital C', 'hospital', ST_SetSRID(ST_MakePoint(116.38, 39.88), 4326));
  `);
}, 120000);

afterAll(async () => {
  await pool.end();
  await container.stop();
});

describe('Spatial queries', () => {
  it('bbox filter returns features within bounds', async () => {
    const result = await pool.query(`
      SELECT id, name FROM features
      WHERE geom && ST_MakeEnvelope(116.25, 39.85, 116.45, 39.95, 4326)
    `);
    expect(result.rows.length).toBeGreaterThanOrEqual(2);
    expect(result.rows.map((r: any) => r.name)).toContain('Park A');
  });

  it('ST_DWithin finds nearby features', async () => {
    const result = await pool.query(`
      SELECT name, ST_Distance(
        ST_Transform(geom, 3857),
        ST_Transform(ST_SetSRID(ST_MakePoint(116.39, 39.91), 4326), 3857)
      ) AS distance_m
      FROM features
      WHERE ST_DWithin(
        ST_Transform(geom, 3857),
        ST_Transform(ST_SetSRID(ST_MakePoint(116.39, 39.91), 4326), 3857),
        5000
      )
      ORDER BY distance_m
    `);
    expect(result.rows.length).toBeGreaterThan(0);
    expect(result.rows[0].distance_m).toBeLessThan(5000);
  });

  it('GeoJSON output is valid', async () => {
    const result = await pool.query(`
      SELECT jsonb_build_object(
        'type', 'FeatureCollection',
        'features', jsonb_agg(jsonb_build_object(
          'type', 'Feature',
          'properties', jsonb_build_object('id', id, 'name', name),
          'geometry', ST_AsGeoJSON(geom)::jsonb
        ))
      ) AS geojson
      FROM features
    `);
    const geojson = result.rows[0].geojson;
    expect(geojson.type).toBe('FeatureCollection');
    expect(geojson.features.length).toBe(3);
    expect(geojson.features[0].geometry.type).toBeDefined();
  });
});
```

**Caveats:**
- **CI Docker requirement.** GitHub Actions supports Docker natively. GitLab CI requires `services: docker:dind`. Some CI environments don't support Docker at all.
- **Container startup time.** PostGIS container takes 5-15 seconds to start. For fast test suites, this is significant. Use `reuse: true` for local development.
- **Test isolation.** Each test suite should get a fresh database or use transactions. Without this, spatial tests interfere with each other (residual features from previous tests).

---

### Playwright -- E2E Map Testing

Microsoft's E2E testing framework. The best option for testing map rendering, interaction, and visual regression in real browsers.

**Why Tier 1:** The only reliable way to test WebGL-rendered maps in real browsers. The MapPage pattern below provides reusable map interaction primitives.

```typescript
// e2e/map.spec.ts
import { test, expect, Page } from '@playwright/test';

class MapPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/map');
    await this.waitForMapLoad();
  }

  async waitForMapLoad() {
    await this.page.waitForFunction(() => {
      const map = (window as any).__map;
      return map && map.loaded() && !map.isMoving();
    }, { timeout: 30000 });
  }

  async getCenter(): Promise<[number, number]> {
    return this.page.evaluate(() => {
      const map = (window as any).__map;
      const center = map.getCenter();
      return [center.lng, center.lat];
    });
  }

  async getZoom(): Promise<number> {
    return this.page.evaluate(() => (window as any).__map.getZoom());
  }

  async clickOnMap(lng: number, lat: number) {
    const point = await this.page.evaluate(([lng, lat]) => {
      const map = (window as any).__map;
      return map.project([lng, lat]);
    }, [lng, lat]);
    await this.page.click('.maplibregl-canvas', { position: { x: point.x, y: point.y } });
  }

  async zoomTo(level: number) {
    await this.page.evaluate((z) => {
      const map = (window as any).__map;
      map.zoomTo(z, { duration: 0 });
    }, level);
    await this.waitForMapLoad();
  }

  async getRenderedFeatures(layerId: string): Promise<any[]> {
    return this.page.evaluate((id) => {
      const map = (window as any).__map;
      return map.queryRenderedFeatures(undefined, { layers: [id] });
    }, layerId);
  }

  async screenshotMap(name: string) {
    const canvas = this.page.locator('.maplibregl-canvas');
    await canvas.screenshot({ path: `screenshots/${name}.png` });
  }
}

test.describe('Map Application', () => {
  let mapPage: MapPage;

  test.beforeEach(async ({ page }) => {
    mapPage = new MapPage(page);
    await mapPage.goto();
  });

  test('map loads at correct initial position', async () => {
    const center = await mapPage.getCenter();
    expect(center[0]).toBeCloseTo(116.4, 0);
    expect(center[1]).toBeCloseTo(39.9, 0);
  });

  test('zoom controls work', async ({ page }) => {
    const initialZoom = await mapPage.getZoom();
    await page.click('.maplibregl-ctrl-zoom-in');
    await mapPage.waitForMapLoad();
    const newZoom = await mapPage.getZoom();
    expect(newZoom).toBeGreaterThan(initialZoom);
  });

  test('clicking feature shows popup', async ({ page }) => {
    await mapPage.clickOnMap(116.4, 39.9);
    await page.waitForSelector('.maplibregl-popup', { timeout: 5000 });
    const popupText = await page.textContent('.maplibregl-popup');
    expect(popupText).toBeTruthy();
  });

  test('layer toggle hides/shows features', async ({ page }) => {
    await page.click('[data-testid="layer-toggle-buildings"]');
    await mapPage.waitForMapLoad();
    const features = await mapPage.getRenderedFeatures('buildings-fill');
    expect(features.length).toBe(0);

    await page.click('[data-testid="layer-toggle-buildings"]');
    await mapPage.waitForMapLoad();
    const featuresAfter = await mapPage.getRenderedFeatures('buildings-fill');
    expect(featuresAfter.length).toBeGreaterThan(0);
  });

  test('visual regression -- map screenshot', async () => {
    await mapPage.zoomTo(12);
    await mapPage.screenshotMap('map-z12');
  });
});
```

**Caveats:**
- **Map loading flakiness.** WebGL-rendered maps have non-deterministic rendering timing. `waitForFunction(() => map.loaded())` can still miss font/sprite loading. Add extra stability waits.
- **Visual regression instability.** Screenshot comparisons of maps are inherently flaky. Label placement, anti-aliasing, and tile rendering order can differ between runs. Increase pixel threshold or mask dynamic areas.
- **Browser memory.** Running 10+ map tests in parallel can consume 4-8GB RAM. Limit Playwright workers on CI.
- **No map-aware selectors.** Must use `page.evaluate()` to interact with map internals. The MapPage pattern above helps but is custom code to maintain.

---

### Zod -- GeoJSON Validation

Runtime schema validation for GeoJSON input. Prevents malformed geometry from reaching the database and provides clear error messages for API consumers.

**Why Tier 1:** Essential at the API boundary. Every spatial endpoint that accepts GeoJSON from clients must validate before passing to PostGIS. Prevents both injection attacks and data corruption.

```typescript
// schemas/geojson.ts
import { z } from 'zod';

const PositionSchema = z.tuple([z.number(), z.number()])
  .or(z.tuple([z.number(), z.number(), z.number()]));

const PointSchema = z.object({
  type: z.literal('Point'),
  coordinates: PositionSchema,
});

const PolygonSchema = z.object({
  type: z.literal('Polygon'),
  coordinates: z.array(z.array(PositionSchema)).refine(
    (rings) => rings.every((ring) => {
      if (ring.length < 4) return false;
      const first = ring[0];
      const last = ring[ring.length - 1];
      return first[0] === last[0] && first[1] === last[1]; // Ring must be closed
    }),
    { message: 'Polygon rings must have >= 4 positions and be closed' }
  ),
});

const GeometrySchema = z.discriminatedUnion('type', [
  PointSchema,
  z.object({ type: z.literal('LineString'), coordinates: z.array(PositionSchema).min(2) }),
  PolygonSchema,
  z.object({ type: z.literal('MultiPoint'), coordinates: z.array(PositionSchema) }),
  z.object({ type: z.literal('MultiLineString'), coordinates: z.array(z.array(PositionSchema)) }),
  z.object({ type: z.literal('MultiPolygon'), coordinates: z.array(z.array(z.array(PositionSchema))) }),
]);

export const FeatureSchema = z.object({
  type: z.literal('Feature'),
  geometry: GeometrySchema,
  properties: z.record(z.unknown()).nullable(),
});

export const FeatureCollectionSchema = z.object({
  type: z.literal('FeatureCollection'),
  features: z.array(FeatureSchema),
});

// Tests
describe('GeoJSON validation', () => {
  it('accepts valid Point feature', () => {
    const result = FeatureSchema.safeParse({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [116.4, 39.9] },
      properties: { name: 'Beijing' },
    });
    expect(result.success).toBe(true);
  });

  it('rejects unclosed polygon ring', () => {
    const result = FeatureSchema.safeParse({
      type: 'Feature',
      geometry: {
        type: 'Polygon',
        coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1]]], // Not closed!
      },
      properties: null,
    });
    expect(result.success).toBe(false);
  });
});
```

**Caveats:**
- **GeoJSON spec allows any coordinate values.** The spec doesn't restrict coordinate ranges. You need additional business logic to validate that coordinates are on Earth (-180 to 180, -90 to 90).
- **Performance with large geometries.** Validating a 100K-vertex polygon with Zod has measurable overhead. For bulk imports, validate a sample or validate server-side only.
- **Missing topology validation.** Zod validates JSON structure, not geometric validity. Self-intersecting polygons and ring orientation errors pass Zod but fail PostGIS.

---

## Authentication & Authorization

### Spatial RBAC -- Row-Level Security

```sql
-- Users have access to specific geographic regions
CREATE TABLE user_regions (
    user_id UUID REFERENCES auth.users(id),
    region_name TEXT,
    geom GEOMETRY(Polygon, 4326),
    permissions TEXT[] DEFAULT '{read}', -- read, write, admin
    PRIMARY KEY (user_id, region_name)
);

ALTER TABLE features ENABLE ROW LEVEL SECURITY;

-- Read policy: users see features within their assigned regions
CREATE POLICY "Users read features in their region"
ON features FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM user_regions ur
        WHERE ur.user_id = auth.uid()
          AND 'read' = ANY(ur.permissions)
          AND ST_Intersects(features.geom, ur.geom)
    )
);

-- Write policy: users can modify features in writable regions
CREATE POLICY "Users write features in their region"
ON features FOR INSERT
WITH CHECK (
    EXISTS (
        SELECT 1 FROM user_regions ur
        WHERE ur.user_id = auth.uid()
          AND 'write' = ANY(ur.permissions)
          AND ST_Within(features.geom, ur.geom)
    )
);

-- Admin can access everything
CREATE POLICY "Admins full access"
ON features FOR ALL
USING (
    EXISTS (
        SELECT 1 FROM user_regions ur
        WHERE ur.user_id = auth.uid()
          AND 'admin' = ANY(ur.permissions)
    )
);
```

**Caveats:**
- **Performance impact is significant.** Spatial RLS policies with `ST_Intersects` are evaluated for every row returned. On tables with 1M+ rows, spatial RLS can cause 10-100x query time increase compared to queries without RLS. Mitigations: use partial GiST indexes on `user_regions.geom`, cache region geometries in application layer, and prefer simpler attribute-based RLS (e.g., `region_id = ANY(user_regions)`) where possible. Benchmark your specific table size before committing to spatial RLS.
- **Tile server bypass.** RLS protects the database, not the tile server. Martin tiles bypass RLS entirely. Tile access control must be at the Nginx/CDN layer.
- **Anti-pattern: Complex spatial RLS on high-throughput tables.** Spatial RLS policies on tables that Martin or Debezium reads cause significant overhead. Use simpler attribute-based RLS where possible.

### JWT with Spatial Claims

```typescript
import jwt from 'jsonwebtoken';

function generateSpatialToken(user: {
  id: string;
  roles: string[];
  regions: { name: string; bbox: number[] }[];
}) {
  return jwt.sign({
    sub: user.id,
    roles: user.roles,
    spatial: {
      regions: user.regions.map((r) => ({ name: r.name, bbox: r.bbox })),
      maxZoom: user.roles.includes('premium') ? 20 : 14,
      maxFeatures: user.roles.includes('premium') ? 10000 : 1000,
    },
  }, process.env.JWT_SECRET!, { expiresIn: '8h' });
}

// Middleware: validate spatial constraints
function spatialAuth(req: Request, res: Response, next: NextFunction) {
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) return res.status(401).json({ error: 'Unauthorized' });

  try {
    const payload = jwt.verify(token, process.env.JWT_SECRET!) as any;
    req.user = payload;

    const bbox = req.query.bbox?.toString().split(',').map(Number);
    if (bbox) {
      const allowed = payload.spatial.regions.some((r: any) =>
        bbox[0] >= r.bbox[0] && bbox[1] >= r.bbox[1] &&
        bbox[2] <= r.bbox[2] && bbox[3] <= r.bbox[3]
      );
      if (!allowed) {
        return res.status(403).json({ error: 'Bbox outside authorized region' });
      }
    }

    next();
  } catch {
    return res.status(401).json({ error: 'Invalid token' });
  }
}
```

### Protecting Tile Endpoints (Nginx + JWT)

```nginx
# Nginx: JWT validation for tile access
location /tiles/ {
    auth_jwt "Tile Access";
    auth_jwt_key_file /etc/nginx/jwt-key.pub;

    set $user_region $jwt_claim_spatial_regions;

    proxy_pass http://martin:3000/;
    proxy_cache tiles;
    proxy_cache_valid 200 24h;
    proxy_cache_key "$uri$user_region"; # Vary cache by user region
}

# Public tiles (basemap, no auth required)
location /tiles/basemap/ {
    proxy_pass http://martin:3000/basemap/;
    proxy_cache tiles;
    proxy_cache_valid 200 30d;
}
```

---

## Security Best Practices

### SQL Injection Prevention in Spatial Queries

```python
# BAD: String concatenation (SQL injection vulnerability!)
@app.get("/features/bad")
async def bad_query(bbox: str):
    result = await db.fetch(
        f"SELECT * FROM features WHERE geom && ST_MakeEnvelope({bbox}, 4326)"  # VULNERABLE!
    )

# GOOD: Parameterized queries
@app.get("/features/good")
async def good_query(bbox: str):
    coords = [float(c) for c in bbox.split(",")]
    if len(coords) != 4:
        raise HTTPException(400, "Invalid bbox")
    if not all(-180 <= coords[i] <= 180 for i in [0, 2]):
        raise HTTPException(400, "Invalid longitude")
    if not all(-90 <= coords[i] <= 90 for i in [1, 3]):
        raise HTTPException(400, "Invalid latitude")

    result = await db.fetch(
        "SELECT * FROM features WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326)",
        *coords  # Parameterized, safe from injection
    )
```

**Caveats:**
- **Spatial functions are SQL injection vectors.** Any ST_ function accepting user input (bbox, WKT, coordinates) must use parameterized queries. String interpolation of bbox strings is the #1 spatial SQL injection vulnerability.
- **WKT/GeoJSON parsing.** Never pass user-supplied WKT directly to `ST_GeomFromText()` without validation. Malformed WKT can cause PostgreSQL errors or unexpected behavior.

### GeoJSON Input Validation

```typescript
const MAX_COORDINATES = 10000;
const MAX_FEATURES = 100;
const MAX_GEOMETRY_SIZE = 1024 * 1024; // 1MB

function validateGeoJSON(input: unknown): { valid: boolean; error?: string } {
  const size = JSON.stringify(input).length;
  if (size > MAX_GEOMETRY_SIZE) {
    return { valid: false, error: `GeoJSON exceeds ${MAX_GEOMETRY_SIZE} bytes` };
  }

  const parsed = FeatureCollectionSchema.safeParse(input);
  if (!parsed.success) {
    return { valid: false, error: parsed.error.message };
  }

  if (parsed.data.features.length > MAX_FEATURES) {
    return { valid: false, error: `Max ${MAX_FEATURES} features allowed` };
  }

  // Coordinate count check (prevent DoS via huge geometries)
  let coordCount = 0;
  for (const feature of parsed.data.features) {
    coordCount += countCoordinates(feature.geometry);
    if (coordCount > MAX_COORDINATES) {
      return { valid: false, error: `Max ${MAX_COORDINATES} coordinates allowed` };
    }
  }

  return { valid: true };
}

function countCoordinates(geometry: any): number {
  const coords = JSON.stringify(geometry.coordinates);
  return (coords.match(/\[[\d.-]+,[\d.-]+/g) || []).length;
}
```

### CORS for Tile Servers

```nginx
location /tiles/ {
    set $cors_origin "";
    if ($http_origin ~* "^https://(app\.example\.com|dashboard\.example\.com)$") {
        set $cors_origin $http_origin;
    }

    add_header Access-Control-Allow-Origin $cors_origin always;
    add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Range, Accept-Encoding" always;
    add_header Access-Control-Expose-Headers "Content-Length, Content-Range" always;
    add_header Access-Control-Max-Age 86400 always;

    if ($request_method = OPTIONS) {
        return 204;
    }

    proxy_pass http://martin:3000/;
}
```

### Rate Limiting Spatial APIs

```python
MAX_BBOX_AREA_DEG2 = 100  # ~100 square degrees

def spatial_rate_limit(func):
    @wraps(func)
    async def wrapper(*args, bbox: str = None, **kwargs):
        if bbox:
            coords = [float(c) for c in bbox.split(",")]
            area = abs(coords[2] - coords[0]) * abs(coords[3] - coords[1])
            if area > MAX_BBOX_AREA_DEG2:
                raise HTTPException(
                    429, f"Bbox area too large. Max {MAX_BBOX_AREA_DEG2} square degrees"
                )
        return await func(*args, bbox=bbox, **kwargs)
    return wrapper
```

**Caveats:**
- **Bbox area-based rate limiting.** Without this, a single request with a world-spanning bbox can return millions of features, crashing the server. Always limit bbox area and feature count.
- **Tile endpoint abuse.** Bots scanning tile endpoints can generate massive CDN bills. Rate limit by IP and require API keys for non-public tiles.

---

## Data Privacy & Compliance

### Spatial Anonymization

```python
import numpy as np

def jitter_coordinates(gdf, max_distance_m=100):
    """Add random displacement to point coordinates for privacy"""
    gdf_3857 = gdf.to_crs(3857)
    angles = np.random.uniform(0, 2 * np.pi, len(gdf_3857))
    distances = np.random.uniform(0, max_distance_m, len(gdf_3857))
    gdf_3857.geometry = gdf_3857.geometry.translate(
        xoff=distances * np.cos(angles),
        yoff=distances * np.sin(angles),
    )
    return gdf_3857.to_crs(4326)

# Grid aggregation: convert points to grid cells
def aggregate_to_grid(gdf, resolution=9):
    """Aggregate points to H3 hexagonal grid"""
    import h3
    gdf['h3_index'] = gdf.geometry.apply(lambda p: h3.latlng_to_cell(p.y, p.x, resolution))
    aggregated = gdf.groupby('h3_index').agg(count=('id', 'count'), avg_value=('value', 'mean')).reset_index()
    aggregated['geometry'] = aggregated['h3_index'].apply(lambda h: Polygon(h3.cell_to_boundary(h)))
    return gpd.GeoDataFrame(aggregated, geometry='geometry', crs=4326)
```

### GDPR for Location Data

```typescript
class LocationDataManager {
  // Right to erasure: delete all location data for a user
  async eraseUserData(userId: string) {
    await db.query('DELETE FROM user_tracks WHERE user_id = $1', [userId]);
    await db.query('DELETE FROM user_locations WHERE user_id = $1', [userId]);
    await db.query('DELETE FROM geofence_events WHERE user_id = $1', [userId]);
    await this.invalidateUserTileCache(userId);
  }

  // Data minimization: only store necessary precision
  async storeLocation(userId: string, lng: number, lat: number) {
    // Round to ~11m precision (4 decimal places)
    const roundedLng = Math.round(lng * 10000) / 10000;
    const roundedLat = Math.round(lat * 10000) / 10000;

    await db.query(
      `INSERT INTO user_locations (user_id, geom, created_at)
       VALUES ($1, ST_SetSRID(ST_MakePoint($2, $3), 4326), NOW())`,
      [userId, roundedLng, roundedLat]
    );
  }

  // Auto-delete old location data
  async cleanupOldData(retentionDays: number = 90) {
    await db.query(
      `DELETE FROM user_locations WHERE created_at < NOW() - make_interval(days => $1)`,
      [retentionDays]
    );
  }
}
```

### Audit Logging

```sql
CREATE TABLE spatial_audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID,
    action TEXT, -- 'read', 'create', 'update', 'delete', 'export'
    table_name TEXT,
    feature_id INTEGER,
    bbox TEXT,
    feature_count INTEGER,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX ON spatial_audit_log (user_id, created_at);
CREATE INDEX ON spatial_audit_log (table_name, action, created_at);
```

**Caveats:**
- **Location data is PII.** Under GDPR, GPS coordinates that can identify an individual's home or workplace are personal data. Must implement right to erasure, data minimization, and retention policies.
- **Coordinate precision leaks identity.** 6 decimal places = 0.11m precision. For anonymized data, round to 3-4 decimal places (11-110m).
- **Audit log storage.** High-traffic spatial APIs can generate millions of audit log entries. Use TimescaleDB hypertables or partition by month.
