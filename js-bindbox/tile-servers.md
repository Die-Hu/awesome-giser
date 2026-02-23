# Tile Servers & Services

> Self-hosted and cloud-based solutions for serving map tiles (vector and raster) to web mapping clients.

> **Quick Picks**
> - **SOTA**: [PMTiles](https://github.com/protomaps/PMTiles) -- serverless tile serving from a single file on any CDN/S3
> - **Free Best**: [Martin](https://github.com/maplibre/martin) -- blazing-fast Rust tile server, directly from PostGIS
> - **Fastest Setup**: PMTiles -- upload one file to S3/CloudFlare R2, done

---

## PMTiles (Protomaps) -- The Serverless Revolution

**PMTiles** is a single-file archive format for tiled data that eliminates the need for a tile server entirely. Tiles are served via HTTP range requests from any static hosting (S3, CloudFlare R2, GitHub Pages, any CDN).

- **Spec:** [github.com/protomaps/PMTiles](https://github.com/protomaps/PMTiles)
- **Key Advantage:** Zero infrastructure -- no tile server, no database, no Docker containers
- **Supports:** Vector tiles (MVT) and raster tiles (PNG/JPEG/WebP)
- **Ecosystem:** CLI tools, MapLibre plugin, Leaflet plugin, QGIS support

### Why PMTiles Matters

| Traditional Tiles | PMTiles |
|-------------------|---------|
| Tile server (Martin, TileServer GL, etc.) | No server needed |
| Database (PostGIS) or directory of files | Single `.pmtiles` file |
| Scaling = more servers | Scaling = CDN handles it |
| Ops burden: Docker, monitoring, patching | Upload file, done |
| Cost: server + bandwidth | Cost: storage + bandwidth only |

### Generate PMTiles

```bash
# Install tippecanoe (vector tile builder)
brew install tippecanoe  # macOS
# or build from source: github.com/felt/tippecanoe

# Convert GeoJSON to PMTiles
tippecanoe -o output.pmtiles -z14 -Z0 --drop-densest-as-needed input.geojson

# Convert MBTiles to PMTiles
pmtiles convert input.mbtiles output.pmtiles
```

### Use PMTiles with MapLibre

```bash
npm install pmtiles maplibre-gl
```

```javascript
import maplibregl from 'maplibre-gl';
import { Protocol } from 'pmtiles';

// Register the PMTiles protocol
const protocol = new Protocol();
maplibregl.addProtocol('pmtiles', protocol.tile);

const map = new maplibregl.Map({
  container: 'map',
  style: {
    version: 8,
    sources: {
      mydata: {
        type: 'vector',
        url: 'pmtiles://https://your-cdn.com/data.pmtiles'
      }
    },
    layers: [{
      id: 'my-layer',
      type: 'fill',
      source: 'mydata',
      'source-layer': 'your_layer_name',
      paint: { 'fill-color': '#088', 'fill-opacity': 0.6 }
    }]
  }
});
```

---

## Self-Hosted Tile Servers

### Martin (MapLibre)

- **Source:** [github.com/maplibre/martin](https://github.com/maplibre/martin)
- **Language:** Rust
- **Key Features:** PostGIS sources, MBTiles, PMTiles, sprites, fonts, health checks
- **Best For:** High-performance PostGIS vector tile serving

```bash
# Docker quickstart
docker run -p 3000:3000 \
  -e DATABASE_URL=postgresql://user:pass@host/db \
  ghcr.io/maplibre/martin

# Or with a config file
martin --config config.yaml
```

```yaml
# martin config.yaml
postgres:
  connection_string: postgresql://user:pass@localhost/geodata
  auto_publish:
    tables: true
    functions: true

pmtiles:
  sources:
    basemap: /data/basemap.pmtiles

sprites:
  paths: /data/sprites

fonts:
  paths: /data/fonts
```

### TileServer GL

- **Source:** [github.com/maptiler/tileserver-gl](https://github.com/maptiler/tileserver-gl)
- **Language:** Node.js
- **Key Features:** MBTiles serving, style rendering, raster from vector, WMTS
- **Best For:** Serving pre-generated MBTiles with raster fallback

```bash
# Docker quickstart
docker run -it -v $(pwd)/data:/data -p 8080:8080 \
  maptiler/tileserver-gl --mbtiles /data/tiles.mbtiles
```

### pg_tileserv

- **Source:** [github.com/CrunchyData/pg_tileserv](https://github.com/CrunchyData/pg_tileserv)
- **Language:** Go
- **Key Features:** Direct PostGIS to MVT, function-based tiles, auto-discovery
- **Best For:** Quick PostGIS vector tile serving without preprocessing

```bash
export DATABASE_URL=postgresql://user:pass@localhost/geodata
pg_tileserv
# Auto-discovers all PostGIS tables and serves as vector tiles
# Browse: http://localhost:7800
```

### t-rex

- **Source:** [github.com/t-rex-tileserver/t-rex](https://github.com/t-rex-tileserver/t-rex)
- **Language:** Rust
- **Key Features:** PostGIS/GDAL sources, built-in cache, TOML config
- **Best For:** Tile generation and caching from PostGIS/GDAL sources

```toml
# t-rex config.toml
[service.mvt]
viewer = true

[datasource]
type = "postgis"
url = "postgresql://user:pass@localhost/geodata"

[[tileset]]
name = "buildings"

[[tileset.layer]]
name = "buildings"
table_name = "buildings"
geometry_field = "geom"
geometry_type = "POLYGON"
```

---

## Tile Format Reference

### MVT (Mapbox Vector Tiles)

The de facto standard for encoding vector tile data in Protocol Buffer format.

- **Encoding:** Protocol Buffers (.pbf)
- **Usage:** Served per tile via `/{z}/{x}/{y}.pbf` endpoints
- **Spec:** [github.com/mapbox/vector-tile-spec](https://github.com/mapbox/vector-tile-spec)

### MBTiles

SQLite-based container format for storing tilesets as a single file.

- **Format:** SQLite database with tiles stored as blobs
- **Metadata:** Zoom range, bounds, format (pbf/png/jpg), attribution
- **Usage:** Used by TileServer GL, Martin, QGIS, Mapbox, tippecanoe output

```bash
# Inspect an MBTiles file
sqlite3 tiles.mbtiles "SELECT name, value FROM metadata;"
# Extract a single tile
sqlite3 tiles.mbtiles "SELECT tile_data FROM tiles WHERE zoom_level=10 AND tile_column=512 AND tile_row=512;"
```

### PMTiles

Single-file archive with internal directory for serverless HTTP range request access.

- **Format:** Custom binary format with clustered tile directory
- **Advantage:** No server needed -- works from any HTTP host supporting range requests
- **Versions:** v3 (current) with optional compression (gzip, brotli, zstd)

### Format Comparison

| Format | Server Required | Single File | Dynamic Sources | Compression | Best For |
|--------|----------------|-------------|-----------------|-------------|----------|
| `/{z}/{x}/{y}.pbf` | Yes (any HTTP) | No (directory) | No | Per-tile gzip | Traditional hosting |
| MBTiles | Yes (tile server) | Yes | No | Per-tile (in blob) | Offline, sharing, archiving |
| PMTiles | No (CDN/S3) | Yes | No | gzip/brotli/zstd | Serverless, static hosting |
| PostGIS -> MVT | Yes (Martin, pg_tileserv) | No | Yes (live query) | Per-tile gzip | Dynamic/live data |

---

## Cloud Tile Services

### Mapbox

- **Website:** [mapbox.com](https://www.mapbox.com)
- **Tiles:** Vector and raster, satellite, terrain, traffic
- **Free Tier:** 200,000 tile requests/month for web
- **Pricing:** Usage-based beyond free tier

### MapTiler

- **Website:** [maptiler.com](https://www.maptiler.com)
- **Tiles:** Vector, raster, hillshade, satellite, OpenStreetMap-based
- **Free Tier:** 100,000 tile requests/month
- **Pricing:** Plans from free to enterprise; also offers self-hosted MapTiler Server

### Stadia Maps

- **Website:** [stadiamaps.com](https://stadiamaps.com)
- **Tiles:** Vector and raster, Stamen styles (Toner, Watercolor, Terrain)
- **Free Tier:** 200,000 credits/month for non-commercial
- **Pricing:** Usage-based for commercial

### Cloud Pricing Quick Comparison

| Provider | Free Tier | Vector Tiles | Satellite | Geocoding | Routing |
|----------|-----------|-------------|-----------|-----------|---------|
| Mapbox | 200k req/mo | Yes | Yes | Yes | Yes |
| MapTiler | 100k req/mo | Yes | Yes | Yes | No |
| Stadia Maps | 200k credits/mo | Yes | No | Yes (Pelias) | No |
| Protomaps (DIY) | Unlimited (self-host) | Yes | No | No | No |

---

## PMTiles vs Traditional Tile Servers

| Aspect | PMTiles on CDN | Martin/pg_tileserv | TileServer GL |
|--------|---------------|-------------------|---------------|
| **Setup** | Upload file to S3/R2 | Docker + PostGIS | Docker + MBTiles |
| **Dynamic data** | No (rebuild file) | Yes (live queries) | No (static MBTiles) |
| **Scalability** | CDN-level (infinite) | Horizontal scaling needed | Horizontal scaling needed |
| **Cost at scale** | Storage + CDN bandwidth | Compute + DB + bandwidth | Compute + bandwidth |
| **Latency** | CDN edge, ~10-50ms | Server location dependent | Server location dependent |
| **Best when** | Data changes rarely | Data changes often | Need raster fallback |

> **Bottom line:** If your data does not change frequently, PMTiles on a CDN is the simplest, cheapest, and most scalable approach. If you need live queries from PostGIS, Martin is the SOTA self-hosted server.

---

## Comparison Table

| Server | Vector | Raster | Source | Protocol | Best For |
|--------|--------|--------|--------|----------|----------|
| PMTiles | Yes | Yes | Static file | HTTP Range | Serverless / static hosting |
| Martin | Yes | No | PostGIS, MBTiles, PMTiles | HTTP | High-perf PostGIS serving |
| TileServer GL | Yes | Yes (rendered) | MBTiles | HTTP, WMTS | MBTiles with raster fallback |
| pg_tileserv | Yes | No | PostGIS | HTTP | Zero-config PostGIS tiles |
| t-rex | Yes | No | PostGIS, GDAL | HTTP | Tile generation + caching |
| Mapbox | Yes | Yes | Cloud | HTTP | Full-stack cloud mapping |
| MapTiler | Yes | Yes | Cloud | HTTP | Cloud tiles, self-host option |
| Stadia Maps | Yes | Yes | Cloud | HTTP | Free non-commercial tiles |

## Deployment Quick Reference

```bash
# PMTiles: Upload and serve (zero ops)
aws s3 cp output.pmtiles s3://my-bucket/tiles/
# Use directly: pmtiles://https://my-bucket.s3.amazonaws.com/tiles/output.pmtiles

# Martin: Docker Compose
docker compose up -d  # with PostGIS + Martin in docker-compose.yml

# TileServer GL: One-liner
docker run -v ./tiles:/data -p 8080:8080 maptiler/tileserver-gl

# pg_tileserv: Binary
DATABASE_URL=postgres://... pg_tileserv --port 7800
```
