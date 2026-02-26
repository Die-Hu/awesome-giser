# Testing & Security for GIS Applications — 2025 Complete Guide

GIS applications present unique testing challenges (map rendering, spatial queries, geometry validation) and security concerns (SQL injection in spatial queries, GeoJSON injection, location privacy). This guide covers comprehensive testing strategies and security best practices.

> **Quick Picks**
> - **Unit testing:** Vitest + Turf.js for spatial functions
> - **Integration testing:** Testcontainers + PostGIS
> - **E2E map testing:** Playwright with MapLibre page objects
> - **Auth:** JWT + spatial claims (bbox/region) + PostGIS RLS
> - **Security:** Parameterized ST_ functions, GeoJSON validation, CORS for tiles

---

## Unit Testing Spatial Code

### Vitest for Spatial Functions

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

### Testing GeoJSON Validation with Zod

```typescript
// schemas/geojson.ts
import { z } from 'zod';

const PositionSchema = z.tuple([z.number(), z.number()]).or(z.tuple([z.number(), z.number(), z.number()]));

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

// __tests__/validation.test.ts
import { describe, it, expect } from 'vitest';
import { FeatureSchema, FeatureCollectionSchema } from '../schemas/geojson';

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
        coordinates: [[[0, 0], [1, 0], [1, 1], [0, 1]]], // Not closed
      },
      properties: null,
    });
    expect(result.success).toBe(false);
  });

  it('rejects coordinates outside valid range', () => {
    const validated = FeatureSchema.safeParse({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [999, 999] },
      properties: null,
    });
    // Note: GeoJSON spec allows any coordinates, but you may want custom validation
    expect(validated.success).toBe(true); // Valid GeoJSON, may want business rule check
  });
});
```

### Snapshot Testing for Map Styles

```typescript
// __tests__/mapStyle.test.ts
import { describe, it, expect } from 'vitest';
import baseStyle from '../styles/base.json';

describe('Map style', () => {
  it('has required sources', () => {
    expect(Object.keys(baseStyle.sources)).toContain('basemap');
    expect(Object.keys(baseStyle.sources)).toContain('buildings');
  });

  it('has layers in correct order', () => {
    const layerIds = baseStyle.layers.map((l: any) => l.id);
    const waterIdx = layerIds.indexOf('water-fill');
    const buildingsIdx = layerIds.indexOf('buildings-fill');
    const labelsIdx = layerIds.indexOf('labels');
    expect(waterIdx).toBeLessThan(buildingsIdx);
    expect(buildingsIdx).toBeLessThan(labelsIdx);
  });

  it('matches snapshot', () => {
    expect(baseStyle).toMatchSnapshot();
  });

  it('all paint properties are valid', () => {
    for (const layer of baseStyle.layers) {
      if (layer.paint) {
        // Check no undefined values
        for (const [key, value] of Object.entries(layer.paint)) {
          expect(value).toBeDefined();
          expect(key).toMatch(/^(fill|line|circle|text|icon|raster|heatmap|hillshade|background)-/);
        }
      }
    }
  });
});
```

---

## Integration Testing

### Testcontainers + PostGIS

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

  // Setup schema
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

  it('ST_Intersects with polygon', async () => {
    const result = await pool.query(`
      SELECT name FROM features
      WHERE ST_Intersects(geom,
        ST_GeomFromText('POLYGON((116.35 39.87, 116.42 39.87, 116.42 39.93, 116.35 39.93, 116.35 39.87))', 4326)
      )
    `);
    expect(result.rows.some((r: any) => r.name === 'School B')).toBe(true);
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

### API Integration Testing

```typescript
// __tests__/integration/api.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import request from 'supertest';
import { app } from '../../src/app';

describe('Feature API', () => {
  it('GET /features returns FeatureCollection', async () => {
    const res = await request(app)
      .get('/features')
      .query({ bbox: '116.3,39.8,116.5,40.0' })
      .expect(200);

    expect(res.body.type).toBe('FeatureCollection');
    expect(Array.isArray(res.body.features)).toBe(true);
  });

  it('GET /features rejects invalid bbox', async () => {
    await request(app)
      .get('/features')
      .query({ bbox: 'invalid' })
      .expect(400);
  });

  it('GET /features respects limit', async () => {
    const res = await request(app)
      .get('/features')
      .query({ bbox: '0,0,180,90', limit: '5' })
      .expect(200);

    expect(res.body.features.length).toBeLessThanOrEqual(5);
  });

  it('POST /features creates valid spatial feature', async () => {
    const res = await request(app)
      .post('/features')
      .send({
        type: 'Feature',
        properties: { name: 'Test Point' },
        geometry: { type: 'Point', coordinates: [116.4, 39.9] },
      })
      .expect(201);

    expect(res.body.id).toBeDefined();
  });

  it('GET /features/nearest returns ordered by distance', async () => {
    const res = await request(app)
      .get('/features/nearest')
      .query({ lng: '116.4', lat: '39.9', radius_m: '5000' })
      .expect(200);

    const distances = res.body.features.map((f: any) => f.properties.distance_m);
    for (let i = 1; i < distances.length; i++) {
      expect(distances[i]).toBeGreaterThanOrEqual(distances[i - 1]);
    }
  });
});
```

---

## E2E Testing Map UIs

### Playwright for Map Applications

```typescript
// e2e/map.spec.ts
import { test, expect, Page } from '@playwright/test';

// Page object for map interactions
class MapPage {
  constructor(private page: Page) {}

  async goto() {
    await this.page.goto('/map');
    await this.waitForMapLoad();
  }

  async waitForMapLoad() {
    // Wait for MapLibre to finish loading tiles
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

  async panTo(lng: number, lat: number) {
    await this.page.evaluate(([lng, lat]) => {
      const map = (window as any).__map;
      map.flyTo({ center: [lng, lat], duration: 0 });
    }, [lng, lat]);
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
    // Toggle buildings layer off
    await page.click('[data-testid="layer-toggle-buildings"]');
    await mapPage.waitForMapLoad();
    const features = await mapPage.getRenderedFeatures('buildings-fill');
    expect(features.length).toBe(0);

    // Toggle back on
    await page.click('[data-testid="layer-toggle-buildings"]');
    await mapPage.waitForMapLoad();
    const featuresAfter = await mapPage.getRenderedFeatures('buildings-fill');
    expect(featuresAfter.length).toBeGreaterThan(0);
  });

  test('visual regression — map screenshot', async () => {
    await mapPage.zoomTo(12);
    await mapPage.screenshotMap('map-z12');
    // Compare with baseline using pixelmatch or Percy
  });
});
```

---

## Authentication & Authorization

### Spatial RBAC — Row-Level Security

```sql
-- Users have access to specific geographic regions
CREATE TABLE user_regions (
    user_id UUID REFERENCES auth.users(id),
    region_name TEXT,
    geom GEOMETRY(Polygon, 4326),
    permissions TEXT[] DEFAULT '{read}', -- read, write, admin
    PRIMARY KEY (user_id, region_name)
);

-- Enable RLS on features table
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

### JWT with Spatial Claims

```typescript
// Generate JWT with spatial constraints
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
      regions: user.regions.map((r) => ({
        name: r.name,
        bbox: r.bbox,
      })),
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

    // Validate bbox request is within user's regions
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

### Martin + Nginx Auth — Protecting Tile Endpoints

```nginx
# Nginx: JWT validation for tile access
location /tiles/ {
    # Validate JWT
    auth_jwt "Tile Access";
    auth_jwt_key_file /etc/nginx/jwt-key.pub;

    # Extract spatial claims
    set $user_region $jwt_claim_spatial_regions;

    # Proxy to Martin
    proxy_pass http://martin:3000/;
    proxy_cache tiles;
    proxy_cache_valid 200 24h;

    # Vary cache by user region
    proxy_cache_key "$uri$user_region";
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
# BAD: String concatenation (SQL injection vulnerability)
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
    # Validate coordinate ranges
    if not all(-180 <= coords[i] <= 180 for i in [0, 2]):
        raise HTTPException(400, "Invalid longitude")
    if not all(-90 <= coords[i] <= 90 for i in [1, 3]):
        raise HTTPException(400, "Invalid latitude")

    result = await db.fetch(
        "SELECT * FROM features WHERE geom && ST_MakeEnvelope($1, $2, $3, $4, 4326)",
        *coords  # Parameterized, safe from injection
    )
```

### GeoJSON Input Validation

```typescript
// Validate user-submitted GeoJSON
import { z } from 'zod';

const MAX_COORDINATES = 10000;
const MAX_FEATURES = 100;
const MAX_GEOMETRY_SIZE = 1024 * 1024; // 1MB

function validateGeoJSON(input: unknown): { valid: boolean; error?: string } {
  // Size check
  const size = JSON.stringify(input).length;
  if (size > MAX_GEOMETRY_SIZE) {
    return { valid: false, error: `GeoJSON exceeds ${MAX_GEOMETRY_SIZE} bytes` };
  }

  // Schema check
  const parsed = FeatureCollectionSchema.safeParse(input);
  if (!parsed.success) {
    return { valid: false, error: parsed.error.message };
  }

  // Feature count check
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

  // Coordinate range check
  for (const feature of parsed.data.features) {
    if (!validateCoordinateRanges(feature.geometry)) {
      return { valid: false, error: 'Coordinates out of valid range' };
    }
  }

  return { valid: true };
}

function countCoordinates(geometry: any): number {
  const coords = JSON.stringify(geometry.coordinates);
  return (coords.match(/\[[\d.-]+,[\d.-]+/g) || []).length;
}

function validateCoordinateRanges(geometry: any): boolean {
  const flat = JSON.stringify(geometry.coordinates);
  const numbers = flat.match(/-?\d+\.?\d*/g)?.map(Number) || [];
  // Simple check: no coordinate > 180 or < -180
  return numbers.every(n => Math.abs(n) <= 180);
}
```

### CORS for Tile Servers

```nginx
# Nginx CORS for tile endpoints
location /tiles/ {
    # Specific origins (preferred)
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
# Rate limit by bbox area (prevent requesting huge areas)
from fastapi import Request, HTTPException
from functools import wraps

MAX_BBOX_AREA_DEG2 = 100  # ~100 square degrees
MAX_REQUESTS_PER_MINUTE = 60

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

@app.get("/features")
@spatial_rate_limit
async def get_features(bbox: str = None):
    # ... query
    pass
```

### Secure File Upload Validation

```python
# Validate uploaded spatial files
import fiona
import geopandas as gpd
from pathlib import Path

ALLOWED_EXTENSIONS = {'.geojson', '.json', '.gpkg', '.shp', '.zip', '.fgb'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_FEATURES = 100000

async def validate_spatial_upload(file_path: Path) -> dict:
    errors = []

    # Extension check
    if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
        errors.append(f"Invalid file type: {file_path.suffix}")
        return {"valid": False, "errors": errors}

    # Size check
    if file_path.stat().st_size > MAX_FILE_SIZE:
        errors.append(f"File too large (max {MAX_FILE_SIZE // 1024 // 1024}MB)")
        return {"valid": False, "errors": errors}

    try:
        gdf = gpd.read_file(file_path)

        # Feature count
        if len(gdf) > MAX_FEATURES:
            errors.append(f"Too many features: {len(gdf)} (max {MAX_FEATURES})")

        # Geometry validity
        invalid = gdf[~gdf.geometry.is_valid]
        if len(invalid) > 0:
            errors.append(f"{len(invalid)} invalid geometries found")

        # CRS check
        if gdf.crs is None:
            errors.append("No CRS defined")
        elif gdf.crs.to_epsg() not in [4326, 3857, None]:
            # Auto-reproject to 4326
            gdf = gdf.to_crs(epsg=4326)

        # Bounds check (must be on Earth)
        bounds = gdf.total_bounds
        if bounds[0] < -180 or bounds[2] > 180 or bounds[1] < -90 or bounds[3] > 90:
            errors.append("Coordinates outside valid range")

    except Exception as e:
        errors.append(f"Failed to read file: {str(e)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "feature_count": len(gdf) if 'gdf' in dir() else 0,
    }
```

---

## Data Privacy & Compliance

### Spatial Anonymization

```python
# Jittering: add random noise to coordinates
import numpy as np

def jitter_coordinates(gdf, max_distance_m=100):
    """Add random displacement to point coordinates"""
    gdf_3857 = gdf.to_crs(3857)
    angles = np.random.uniform(0, 2 * np.pi, len(gdf_3857))
    distances = np.random.uniform(0, max_distance_m, len(gdf_3857))

    gdf_3857.geometry = gdf_3857.geometry.translate(
        xoff=distances * np.cos(angles),
        yoff=distances * np.sin(angles),
    )
    return gdf_3857.to_crs(4326)

# Grid aggregation: convert points to grid cells
def aggregate_to_grid(gdf, cell_size_m=500):
    """Aggregate points to hexagonal grid"""
    import h3
    gdf['h3_index'] = gdf.geometry.apply(
        lambda p: h3.latlng_to_cell(p.y, p.x, 9)  # ~150m resolution
    )
    aggregated = gdf.groupby('h3_index').agg(
        count=('id', 'count'),
        avg_value=('value', 'mean'),
    ).reset_index()
    aggregated['geometry'] = aggregated['h3_index'].apply(
        lambda h: Polygon(h3.cell_to_boundary(h))
    )
    return gpd.GeoDataFrame(aggregated, geometry='geometry', crs=4326)
```

### GDPR for Location Data

```typescript
// Data retention and erasure for location data
class LocationDataManager {
  // Right to erasure: delete all location data for a user
  async eraseUserData(userId: string) {
    await db.query('DELETE FROM user_tracks WHERE user_id = $1', [userId]);
    await db.query('DELETE FROM user_locations WHERE user_id = $1', [userId]);
    await db.query('DELETE FROM geofence_events WHERE user_id = $1', [userId]);
    // Invalidate cached tiles that may contain user data
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
      `DELETE FROM user_locations
       WHERE created_at < NOW() - INTERVAL '${retentionDays} days'`
    );
  }
}
```

### Audit Logging

```sql
-- Audit log for spatial data access
CREATE TABLE spatial_audit_log (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID,
    action TEXT, -- 'read', 'create', 'update', 'delete', 'export'
    table_name TEXT,
    feature_id INTEGER,
    bbox TEXT, -- requested bbox
    feature_count INTEGER, -- number of features returned
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for compliance queries
CREATE INDEX ON spatial_audit_log (user_id, created_at);
CREATE INDEX ON spatial_audit_log (table_name, action, created_at);
```
