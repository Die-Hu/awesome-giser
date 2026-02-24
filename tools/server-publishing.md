# Geospatial Servers & Publishing

**Quick Picks**
- **SOTA**: Martin + PMTiles — modern, fast, zero-ops friendly
- **Free Best**: GeoServer — full OGC stack, enterprise features, massive ecosystem
- **Fastest Setup**: TiTiler (COG/raster) or pg_tileserv (PostGIS vector) — running in under 5 minutes

---

## OGC-Compliant Map Servers

### GeoServer

The Swiss Army knife of geospatial servers. If you need to publish data and you are not sure what protocol the client will ask for, start here. It handles WMS, WFS, WCS, WMTS, WPS, and the modern OGC API Features/Tiles/Maps stack all from one deployment.

**Supported data sources:**
- PostGIS, Shapefile, GeoPackage, GeoTIFF
- NetCDF, S3 COG, MongoDB, Elasticsearch
- WMS/WFS cascading (proxy upstream services)
- ImageMosaic store for tiled raster collections

**Styling: SLD vs CSS Extension**

SLD (Styled Layer Descriptor) is the OGC standard but it is verbose and painful to write by hand. Use the CSS extension instead — same power, human-readable syntax:

```css
/* CSS Extension example — polygon with conditional symbolization */
* {
  fill: [CASE WHEN population > 1000000 THEN '#d73027' WHEN population > 100000 THEN '#fc8d59' ELSE '#fee090' END];
  fill-opacity: 0.7;
  stroke: #333333;
  stroke-width: 0.5;
  label: [name];
  font-size: 11;
  label-anchor: 0.5 0.5;
  [-gt-label-max-displacement: 40];
  [-gt-label-auto-wrap: 80];
}
```

Convert CSS to SLD at any time via the REST API or web UI — useful when you need to export to other servers.

**GeoWebCache (built-in tile cache)**

GeoWebCache ships inside GeoServer and intercepts WMTS and WMS-C requests. Configure backends in `geowebcache.xml`:

```xml
<gwcConfiguration>
  <serviceInformation>
    <title>GeoWebCache</title>
  </serviceInformation>
  <gridSets>
    <!-- WebMercator already built-in -->
  </gridSets>
  <layers>
    <wmsLayer>
      <name>myworkspace:mylayer</name>
      <mimeFormats>
        <string>image/png</string>
        <string>image/webp</string>
        <string>application/vnd.mapbox-vector-tile</string>
      </mimeFormats>
      <gridSubsets>
        <gridSubset><gridSetName>EPSG:900913</gridSetName></gridSubset>
        <gridSubset><gridSetName>EPSG:4326</gridSetName></gridSubset>
      </gridSubsets>
      <blobStoreId>s3BlobStore</blobStoreId>
    </wmsLayer>
  </layers>
  <blobStores>
    <S3BlobStore>
      <id>s3BlobStore</id>
      <enabled>true</enabled>
      <bucket>my-tile-cache</bucket>
      <prefix>geoserver/tiles</prefix>
      <awsAccessKey>AKIAIOSFODNN7EXAMPLE</awsAccessKey>
      <awsSecretKey>wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY</awsSecretKey>
      <maxConnections>50</maxConnections>
    </S3BlobStore>
  </blobStores>
</gwcConfiguration>
```

Seed tiles from the command line (useful for pre-warming a cache after deployment):

```bash
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/gwc/rest/seed/myworkspace:mylayer.json" \
  -H "Content-type: application/json" \
  -d '{
    "seedRequest": {
      "name": "myworkspace:mylayer",
      "srs": {"number": 900913},
      "zoomStart": 0,
      "zoomStop": 14,
      "format": "image/png",
      "type": "seed",
      "threadCount": 4
    }
  }'
```

Check seeding progress:

```bash
curl -u admin:geoserver \
  "http://localhost:8080/geoserver/gwc/rest/seed/myworkspace:mylayer.json"
```

Truncate a layer's cache after data update:

```bash
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/gwc/rest/seed/myworkspace:mylayer.json" \
  -H "Content-type: application/json" \
  -d '{"seedRequest":{"name":"myworkspace:mylayer","type":"truncate"}}'
```

**Performance Tuning**

JVM settings in `setenv.sh` (Tomcat) or `JAVA_OPTS` (Docker env):

```bash
export JAVA_OPTS="\
  -Xms2g -Xmx8g \
  -XX:+UseG1GC \
  -XX:MaxGCPauseMillis=200 \
  -XX:ParallelGCThreads=4 \
  -XX:+HeapDumpOnOutOfMemoryError \
  -Dfile.encoding=UTF-8 \
  -Djavax.servlet.request.encoding=UTF-8 \
  -Dorg.geotools.coverage.jaiext.enabled=true \
  -Djava.awt.headless=true \
  -DGEOSERVER_DATA_DIR=/opt/geoserver/data_dir"
```

Enable native JAI and ImageIO for faster raster rendering (install native libs in the Docker image):

```dockerfile
FROM kartoza/geoserver:2.24.0
RUN apt-get install -y libjai-imageio-core-java && \
    cp /usr/share/java/jai_core.jar /opt/jdk/jre/lib/ext/ && \
    cp /usr/share/java/jai_codec.jar /opt/jdk/jre/lib/ext/ && \
    cp /usr/share/java/mlibwrapper_jai.jar /opt/jdk/jre/lib/ext/
```

PostGIS connection pooling — set in the datastore configuration or via JNDI:

```xml
<!-- web.xml JNDI datasource for high-traffic deployments -->
<Resource name="jdbc/postgis"
  auth="Container"
  type="javax.sql.DataSource"
  driverClassName="org.postgresql.Driver"
  url="jdbc:postgresql://pghost:5432/geodata"
  username="geoserver"
  password="secret"
  maxTotal="50"
  maxIdle="10"
  minIdle="5"
  maxWaitMillis="10000"
  testOnBorrow="true"
  validationQuery="SELECT 1"
  removeAbandonedOnBorrow="true"
  removeAbandonedTimeout="60"/>
```

Enable response compression in GeoServer's `web.xml`:

```xml
<filter>
  <filter-name>GzipFilter</filter-name>
  <filter-class>org.apache.catalina.filters.ExpiresFilter</filter-class>
</filter>
```

Or (better) handle it at Nginx:

```nginx
gzip on;
gzip_types text/plain application/json application/vnd.mapbox-vector-tile image/svg+xml;
gzip_min_length 1024;
gzip_vary on;
```

**Docker deployment**

```yaml
# docker-compose.yml
version: "3.8"
services:
  geoserver:
    image: kartoza/geoserver:2.24.0
    environment:
      GEOSERVER_DATA_DIR: /opt/geoserver/data_dir
      GEOWEBCACHE_CACHE_DIR: /opt/geoserver/gwc_cache
      JAVA_OPTS: "-Xms2g -Xmx6g -XX:+UseG1GC"
      GEOSERVER_ADMIN_USER: admin
      GEOSERVER_ADMIN_PASSWORD: changeme
    volumes:
      - geoserver_data:/opt/geoserver/data_dir
      - gwc_cache:/opt/geoserver/gwc_cache
    ports:
      - "8080:8080"
    depends_on:
      - postgis

  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: geodata
      POSTGRES_USER: geoserver
      POSTGRES_PASSWORD: secret
    volumes:
      - pg_data:/var/lib/postgresql/data
    command: >
      postgres
        -c max_connections=200
        -c shared_buffers=2GB
        -c work_mem=64MB
        -c maintenance_work_mem=512MB
        -c effective_cache_size=6GB
        -c random_page_cost=1.1

volumes:
  geoserver_data:
  gwc_cache:
  pg_data:
```

**REST API for automated layer publishing**

Automate everything you do in the web UI via REST. This is the path to reproducible infrastructure:

```bash
# Create workspace
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/workspaces" \
  -H "Content-Type: application/json" \
  -d '{"workspace":{"name":"production"}}'

# Create PostGIS datastore
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/workspaces/production/datastores" \
  -H "Content-Type: application/json" \
  -d '{
    "dataStore": {
      "name": "postgis_main",
      "connectionParameters": {
        "entry": [
          {"@key":"host","$":"pghost"},
          {"@key":"port","$":"5432"},
          {"@key":"database","$":"geodata"},
          {"@key":"user","$":"geoserver"},
          {"@key":"passwd","$":"secret"},
          {"@key":"dbtype","$":"postgis"},
          {"@key":"max connections","$":"10"},
          {"@key":"min connections","$":"2"},
          {"@key":"Expose primary keys","$":"true"}
        ]
      }
    }
  }'

# Publish a PostGIS table as a layer
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/workspaces/production/datastores/postgis_main/featuretypes" \
  -H "Content-Type: application/json" \
  -d '{
    "featureType": {
      "name": "municipalities",
      "nativeName": "municipalities",
      "title": "Municipalities",
      "srs": "EPSG:4326",
      "projectionPolicy": "REPROJECT_TO_DECLARED"
    }
  }'

# Upload and assign an SLD style
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/styles" \
  -H "Content-Type: application/vnd.ogc.sld+xml" \
  --data-binary @municipalities.sld

# Set default style on layer
curl -u admin:geoserver -XPUT \
  "http://localhost:8080/geoserver/rest/layers/production:municipalities" \
  -H "Content-Type: application/json" \
  -d '{"layer":{"defaultStyle":{"name":"municipalities"}}}'
```

**SQL Views as Virtual Layers**

Create parameterized virtual layers without creating database views. Perfect for user-driven filtering:

```bash
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/workspaces/production/datastores/postgis_main/featuretypes" \
  -H "Content-Type: application/json" \
  -d '{
    "featureType": {
      "name": "filtered_parcels",
      "title": "Filtered Parcels",
      "metadata": {
        "entry": {
          "@key": "JDBC_VIRTUAL_TABLE",
          "virtualTable": {
            "name": "filtered_parcels",
            "sql": "SELECT * FROM parcels WHERE land_use = %land_use% AND area_m2 > %min_area%",
            "escapeSql": false,
            "geometry": {
              "name": "geom",
              "type": "Polygon",
              "srid": 4326
            },
            "parameter": [
              {
                "name": "land_use",
                "defaultValue": "residential",
                "regexpValidator": "^[\\w]+$"
              },
              {
                "name": "min_area",
                "defaultValue": "0",
                "regexpValidator": "^[\\d\\.]+$"
              }
            ]
          }
        }
      }
    }
  }'
```

Query the virtual layer from a WMS client:

```
http://localhost:8080/geoserver/production/wms?
  SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap
  &LAYERS=production:filtered_parcels
  &viewparams=land_use:commercial;min_area:5000
  &BBOX=...&WIDTH=800&HEIGHT=600&SRS=EPSG:4326&FORMAT=image/png
```

**GeoFence for row-level and attribute-level security**

GeoFence is a GeoServer plugin that intercepts requests and applies fine-grained rules based on user identity, IP, time, and spatial extent. Far more powerful than built-in layer security:

- Deny access to features outside a user's assigned territory (spatial filter per user)
- Hide sensitive attribute columns for specific user groups (attribute filtering)
- Limit WMS GetMap to a specific bounding box per user role
- Integrate with LDAP/Active Directory groups

Install via GeoServer plugin manager, then configure rules via the GeoFence REST API or web UI.

---

### MapServer

C-based, compiled binary — no JVM overhead. For simple WMS rendering of large raster datasets or high-throughput deployments, MapServer can outperform GeoServer. Configuration is via `.map` files which are version-control friendly and easy to diff.

**Sample Mapfile:**

```mapfile
MAP
  NAME "production"
  STATUS ON
  SIZE 800 600
  EXTENT -180 -90 180 90
  PROJECTION
    "init=epsg:4326"
  END

  WEB
    METADATA
      "ows_title" "My Map Service"
      "ows_onlineresource" "https://maps.example.com/cgi-bin/mapserv?map=production.map"
      "ows_srs" "EPSG:4326 EPSG:3857"
      "wms_enable_request" "*"
      "wfs_enable_request" "*"
    END
  END

  LAYER
    NAME "municipalities"
    TYPE POLYGON
    STATUS ON
    DATA "/data/municipalities"
    PROJECTION
      "init=epsg:4326"
    END
    CLASS
      EXPRESSION ([population] > 1000000)
      STYLE
        COLOR 215 48 39
        OUTLINECOLOR 50 50 50
        WIDTH 0.5
      END
    END
    CLASS
      STYLE
        COLOR 254 224 144
        OUTLINECOLOR 50 50 50
        WIDTH 0.5
      END
    END
  END
END
```

**MapCache** for tile caching alongside MapServer:

```xml
<!-- mapcache.xml -->
<mapcache>
  <source name="mywms" type="wms">
    <http><url>http://localhost/cgi-bin/mapserv?map=production.map</url></http>
    <getmap><params>
      <LAYERS>municipalities</LAYERS>
      <FORMAT>image/png</FORMAT>
    </params></getmap>
  </source>
  <cache name="disk" type="disk">
    <base>/var/cache/mapcache</base>
    <symlink_blank/>
  </cache>
  <tileset name="municipalities">
    <source>mywms</source>
    <cache>disk</cache>
    <grid>GoogleMapsCompatible</grid>
    <format>PNG</format>
    <metatile>5 5</metatile>
    <expires>3600</expires>
  </tileset>
  <service type="wmts" enabled="true"/>
  <service type="tms" enabled="true"/>
  <service type="xyz" enabled="true"/>
</mapcache>
```

**When to choose MapServer over GeoServer:**
- High-concurrency WMS with simple cartographic rules
- Minimal RAM environment (MapServer uses ~50MB idle vs GeoServer's ~1GB)
- Need a stateless binary deployable via CGI/FastCGI
- Configuration must be in version-controlled text files (no XML data_dir)

**When to stick with GeoServer:**
- You need WPS (geoprocessing over HTTP)
- Complex SLD styling or CSS extension styling
- Multiple simultaneous users editing styles/layers via web UI
- Need the GeoWebCache integration for tile serving
- OGC API support is required

---

### QGIS Server

Expose your QGIS `.qgs`/`.qgz` projects directly as WMS, WFS, WMTS, and OGC API endpoints. The killer feature: your QGIS desktop styling is exactly what you get on the server — no translation layer, no re-learning a styling language.

**Deployment with FastCGI (production-grade):**

```nginx
# /etc/nginx/sites-available/qgis-server
server {
    listen 80;
    server_name maps.example.com;

    location /qgis-server {
        root /var/www/qgis-server;
        gzip off;
        include fastcgi_params;
        fastcgi_param QGIS_SERVER_LOG_STDERR 1;
        fastcgi_param QGIS_SERVER_LOG_LEVEL 0;
        fastcgi_param QGIS_PROJECT_FILE /data/projects/myproject.qgz;
        fastcgi_param MAX_CACHE_LAYERS 100;
        fastcgi_param QGIS_SERVER_PARALLEL_RENDERING 1;
        fastcgi_pass unix:/run/qgis-server/qgis-server.socket;
    }
}
```

**Docker deployment:**

```yaml
version: "3.8"
services:
  qgis-server:
    image: qgis/qgis-server:ltr
    environment:
      QGIS_SERVER_LOG_STDERR: 1
      QGIS_SERVER_LOG_LEVEL: 0
      QGIS_SERVER_PARALLEL_RENDERING: "true"
      QGIS_SERVER_MAX_THREADS: 4
      QGIS_PROJECT_FILE: /data/projects/myproject.qgz
    volumes:
      - /data/qgis-projects:/data/projects:ro
      - /data/geodata:/data/geodata:ro
    ports:
      - "8081:80"
```

**Lizmap** is a web client framework built specifically for QGIS Server. Configure layers, print layouts, attribute tables, editing forms, and popup templates entirely from the QGIS desktop plugin, then deploy. End users get a polished web GIS without writing a single line of frontend code.

---

## Modern Tile Servers

### Martin

Rust-based PostGIS tile server from the Mapbox team. Zero-config startup, sub-millisecond tile generation, and serves PMTiles/MBTiles as well. The best choice for production vector tile serving in 2025+.

**Zero-config startup:**

```bash
# Serve all PostGIS tables as vector tiles instantly
docker run -p 3000:3000 \
  -e DATABASE_URL=postgresql://user:pass@pghost/geodata \
  ghcr.io/maplibre/martin

# Tiles available at: http://localhost:3000/{table_name}/{z}/{x}/{y}
# TileJSON: http://localhost:3000/{table_name}
```

**YAML config for production:**

```yaml
# martin.yaml
listen_addresses: "0.0.0.0:3000"
keep_alive: 75
worker_processes: 8
web_ui: true

postgres:
  connection_string: "postgresql://martin:secret@pghost:5432/geodata"
  pool_size: 20
  auto_publish:
    tables:
      from_schemas:
        - public
        - analysis
      id_columns:
        - id
        - gid
        - fid
    functions: true

pmtiles:
  sources:
    world_basemap:
      path: /data/tiles/world.pmtiles

mbtiles:
  sources:
    satellite:
      path: /data/tiles/satellite.mbtiles

sprites:
  sources:
    default:
      path: /data/sprites/default

fonts:
  - /data/fonts
```

**Function sources — the real power of Martin:**

Write SQL functions that generate tiles dynamically with parameters. This is how you build things like heatmaps, choropleth aggregations, or user-filtered views that would be impossible with a simple table source:

```sql
-- Dynamic heatmap tile function
CREATE OR REPLACE FUNCTION public.incident_heatmap(
  z integer,
  x integer,
  y integer,
  query_params json DEFAULT '{}'
)
RETURNS bytea AS $$
DECLARE
  mvt bytea;
  start_date date := COALESCE((query_params->>'start_date')::date, NOW() - INTERVAL '30 days');
  end_date date := COALESCE((query_params->>'end_date')::date, NOW());
  incident_type text := COALESCE(query_params->>'type', '%');
BEGIN
  SELECT INTO mvt ST_AsMVT(tile, 'incidents', 4096, 'geom')
  FROM (
    SELECT
      ST_AsMVTGeom(
        ST_Transform(location, 3857),
        ST_TileEnvelope(z, x, y),
        4096, 64, true
      ) AS geom,
      severity,
      incident_type,
      occurred_at
    FROM incidents
    WHERE
      location && ST_Transform(ST_TileEnvelope(z, x, y), 4326)
      AND occurred_at BETWEEN start_date AND end_date
      AND incident_type LIKE incident_type
      AND ST_AsMVTGeom(
        ST_Transform(location, 3857),
        ST_TileEnvelope(z, x, y),
        4096, 64, true
      ) IS NOT NULL
  ) tile;
  RETURN mvt;
END;
$$ LANGUAGE plpgsql STABLE PARALLEL SAFE;
```

Query the function source with parameters:

```
http://localhost:3000/rpc/incident_heatmap/{z}/{x}/{y}?start_date=2024-01-01&type=fire
```

**Docker Compose with Martin + PostGIS:**

```yaml
version: "3.8"
services:
  martin:
    image: ghcr.io/maplibre/martin:latest
    command: --config /config/martin.yaml
    volumes:
      - ./martin.yaml:/config/martin.yaml:ro
      - /data/tiles:/data/tiles:ro
      - /data/fonts:/data/fonts:ro
      - /data/sprites:/data/sprites:ro
    ports:
      - "3000:3000"
    depends_on:
      - postgis
    environment:
      DATABASE_URL: postgresql://martin:secret@postgis:5432/geodata

  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: geodata
      POSTGRES_USER: martin
      POSTGRES_PASSWORD: secret
    volumes:
      - pg_data:/var/lib/postgresql/data

volumes:
  pg_data:
```

---

### pg_tileserv

CrunchyData's simpler PostGIS tile server. If you want to get something running in 90 seconds and do not need Martin's advanced features, start here.

```bash
# Run against existing PostGIS
docker run --rm -it -p 7800:7800 \
  -e DATABASE_URL="postgresql://user:pass@pghost/geodata" \
  pramsey/pg_tileserv

# Every table/view/function is now a tile source
# Table tiles: http://localhost:7800/public.municipalities/{z}/{x}/{y}.pbf
# Function tiles: http://localhost:7800/public.my_tile_fn/{z}/{x}/{y}.pbf
# Built-in viewer: http://localhost:7800/
```

The built-in web UI renders each layer with MapLibre GL JS — useful for quick visual QA of your data without setting up a separate map viewer.

**pg_tileserv function example** (simpler syntax than Martin):

```sql
CREATE OR REPLACE FUNCTION public.roads_by_class(
  z integer, x integer, y integer,
  road_class text DEFAULT 'primary'
)
RETURNS bytea AS $$
  SELECT ST_AsMVT(t, 'roads', 4096, 'geom')
  FROM (
    SELECT
      ST_AsMVTGeom(ST_Transform(geom, 3857), ST_TileEnvelope(z,x,y)) AS geom,
      name, ref, maxspeed
    FROM roads
    WHERE class = road_class
      AND geom && ST_Transform(ST_TileEnvelope(z,x,y), 4326)
  ) t;
$$ LANGUAGE sql STABLE PARALLEL SAFE;
```

---

### TiTiler

Python (FastAPI) based COG and STAC raster tile server. The trick here is dynamic tiling — no tile pre-generation, no tile database. Point it at a COG on S3 and it serves tiles on demand.

**Run locally or in Docker:**

```bash
pip install titiler[full]
uvicorn titiler.application.main:app --host 0.0.0.0 --port 8000

# Or Docker
docker run -p 8000:8000 ghcr.io/developmentseed/titiler:latest
```

**Core tile endpoint:**

```
# Tile from a COG on S3
GET /cog/tiles/WebMercatorQuad/{z}/{x}/{y}
  ?url=s3://my-bucket/data/elevation.tif
  &bidx=1
  &colormap_name=terrain
  &rescale=0,3000
  &format=webp

# Preview image
GET /cog/preview
  ?url=s3://my-bucket/data/ndvi_2024.tif
  &expression=b1
  &colormap_name=rdylgn
  &rescale=-1,1

# Statistics (histogram, min, max, mean)
GET /cog/statistics
  ?url=s3://my-bucket/data/dem.tif

# Crop/extract GeoTIFF subset
GET /cog/crop/{minx},{miny},{maxx},{maxy}.tif
  ?url=s3://my-bucket/data/landsat.tif
  &width=512&height=512
```

**Dynamic band math and algorithms via URL params:**

```
# NDVI from Landsat bands
GET /cog/tiles/{z}/{x}/{y}
  ?url=s3://bucket/landsat.tif
  &expression=(b5-b4)/(b5+b4)
  &colormap_name=rdylgn
  &rescale=-1,1

# Hillshade algorithm
GET /cog/tiles/{z}/{x}/{y}
  ?url=s3://bucket/dem.tif
  &algorithm=hillshade
  &algorithm_params={"azimuth":315,"altitude":45,"buffer":3}
```

**titiler-pgstac for STAC mosaics:**

```bash
pip install titiler-pgstac
```

Query a STAC search result as a seamless mosaic:

```
POST /mosaic/register
{
  "collections": ["sentinel-2-l2a"],
  "bbox": [100.0, 1.0, 105.0, 5.0],
  "datetime": "2024-01-01/2024-12-31",
  "filter": {"op":"<=","args":[{"property":"eo:cloud_cover"},10]}
}
# Returns a searchid, then:
GET /mosaic/{searchid}/tiles/{z}/{x}/{y}
  ?assets=B04,B03,B02
  &rescale=0,3000
```

**Deploy on AWS Lambda for serverless raster tiles:**

```bash
pip install mangum titiler[full]
```

```python
# lambda_handler.py
from mangum import Mangum
from titiler.application import app

# Configure for S3 access
import os
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.tiff"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["AWS_REQUEST_PAYER"] = "requester"  # if needed

handler = Mangum(app, lifespan="off")
```

```yaml
# serverless.yml / SAM template excerpt
Resources:
  TiTilerFunction:
    Type: AWS::Serverless::Function
    Properties:
      Handler: lambda_handler.handler
      Runtime: python3.11
      MemorySize: 3008  # max memory = fastest GDAL
      Timeout: 30
      Environment:
        Variables:
          GDAL_CACHEMAX: "512"
          GDAL_DISABLE_READDIR_ON_OPEN: "EMPTY_DIR"
          CPL_VSIL_CURL_CACHE_SIZE: "200000000"
      Events:
        TileApi:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: any
```

Lambda cold starts are 1-3 seconds. Use Provisioned Concurrency for latency-sensitive production deployments.

---

### t-rex and Tegola

**t-rex** (Rust): Fast PostGIS + GDAL vector tile server. Simpler config than Martin:

```toml
# t-rex config.toml
[webserver]
bind = "0.0.0.0"
port = 6767

[[datasource]]
name = "postgis"
dbconn = "postgresql://user:pass@localhost/geodata"

[grid]
predefined = "web_mercator"

[[tileset]]
name = "municipalities"
[[tileset.layer]]
name = "municipalities"
datasource = "postgis"
geometry_type = "POLYGON"
sql = "SELECT gid, name, population, ST_Transform(geom, 3857) AS geom FROM municipalities WHERE geom && !bbox!"
```

**Tegola** (Go): PostGIS + GeoPackage, clean config, built-in cache layer:

```toml
# tegola config.toml
[webserver]
port = ":8080"

[[providers]]
name = "postgis"
type = "postgis"
host = "pghost"
port = 5432
database = "geodata"
user = "tegola"
password = "secret"
  [[providers.layers]]
  name = "land_use"
  sql = """
    SELECT
      ST_AsBinary(ST_Simplify(ST_Transform(geom,3857), !PIXEL_WIDTH! * 2)) AS geom,
      gid, class, area_ha
    FROM land_use
    WHERE geom && ST_Transform(!BBOX!, 4326)
  """

[[maps]]
name = "basemap"
  [[maps.layers]]
  provider_layer = "postgis.land_use"
  min_zoom = 8
  max_zoom = 18

[cache]
type = "s3"
bucket = "tegola-tile-cache"
region = "us-east-1"
```

---

## Serverless & Static Tile Hosting

### PMTiles

A single `.pmtiles` file contains your entire tile archive, indexed for HTTP range request access. Upload to any static host and serve tiles with zero infrastructure — no tile server process, no database, no API gateway.

**Create PMTiles from GeoJSON with tippecanoe:**

```bash
# Install tippecanoe
brew install tippecanoe  # macOS
# or build from source on Linux

# Basic conversion — let tippecanoe choose zoom levels
tippecanoe \
  -o municipalities.pmtiles \
  -zg \
  --drop-densest-as-needed \
  --name "Municipalities" \
  --description "Municipal boundaries 2024" \
  input.geojson

# Multiple layers in one file
tippecanoe \
  -o basemap.pmtiles \
  -zg \
  --drop-densest-as-needed \
  -L roads:roads.geojson \
  -L buildings:buildings.geojson \
  -L landuse:landuse.geojson \
  -L water:water.geojson

# Large dataset: preserve all features, use attribute filtering
tippecanoe \
  -o parcels.pmtiles \
  -z14 -Z10 \
  --no-feature-limit \
  --no-tile-size-limit \
  -y parcel_id -y land_use -y area_m2 -y assessed_value \
  parcels.geojson
```

**pmtiles CLI for inspecting and serving:**

```bash
# Install
pip install pmtiles
# or
go install github.com/protomaps/go-pmtiles@latest

# Inspect metadata
pmtiles show municipalities.pmtiles

# Serve locally for development
pmtiles serve /data/tiles --port 8080
# Tiles at: http://localhost:8080/municipalities/{z}/{x}/{y}.mvt

# Extract a region (country extract from planet)
pmtiles extract \
  https://r2-public.protomaps.com/protomaps-basemap-v4-latest.pmtiles \
  japan.pmtiles \
  --bbox="122.9,24.0,153.9,45.5"

# Convert MBTiles to PMTiles
pmtiles convert input.mbtiles output.pmtiles
```

**Upload to CloudFlare R2 (zero egress cost):**

```bash
# Using rclone
rclone copy municipalities.pmtiles r2:my-tile-bucket/tiles/

# Using AWS CLI (R2 is S3-compatible)
aws s3 cp municipalities.pmtiles s3://my-tile-bucket/tiles/ \
  --endpoint-url https://ACCOUNT_ID.r2.cloudflarestorage.com \
  --content-type application/octet-stream
```

Configure R2 bucket CORS for browser access:

```json
[
  {
    "AllowedOrigins": ["https://myapp.example.com"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Range", "If-Match"],
    "ExposeHeaders": ["Content-Range", "Content-Length", "ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

**MapLibre GL JS client integration:**

```javascript
import { Protocol } from "pmtiles";
import maplibregl from "maplibre-gl";

// Register the pmtiles protocol
const protocol = new Protocol();
maplibregl.addProtocol("pmtiles", protocol.tile.bind(protocol));

const map = new maplibregl.Map({
  container: "map",
  style: {
    version: 8,
    sources: {
      municipalities: {
        type: "vector",
        url: "pmtiles://https://ACCOUNT_ID.r2.cloudflarestorage.com/tiles/municipalities.pmtiles",
      },
    },
    layers: [
      {
        id: "municipalities-fill",
        type: "fill",
        source: "municipalities",
        "source-layer": "municipalities",
        paint: {
          "fill-color": [
            "interpolate", ["linear"],
            ["get", "population"],
            0, "#fee090",
            100000, "#fc8d59",
            1000000, "#d73027"
          ],
          "fill-opacity": 0.7,
        },
      },
    ],
  },
});
```

**Protomaps basemap — the full world in one PMTiles file:**

```javascript
// ~100GB PMTiles file with global OSM basemap
// Served from Cloudflare, or self-host the extract
const BASEMAP_URL = "https://r2-public.protomaps.com/protomaps-basemap-v4-latest.pmtiles";
```

---

### COG (Cloud-Optimized GeoTIFF)

COG is a GeoTIFF with internal tiling and overviews arranged so any HTTP client can request only the bytes needed for a specific extent/zoom — no tile server required.

**Create a COG from any raster:**

```bash
# Using GDAL
gdal_translate \
  -of COG \
  -co COMPRESS=WEBP \
  -co WEBP_LOSSLESS=YES \
  -co OVERVIEW_RESAMPLING=AVERAGE \
  -co BLOCKSIZE=512 \
  input.tif output_cog.tif

# Using rio-cogeo (recommended — validates output)
pip install rio-cogeo
rio cogeo create \
  --cog-profile webp \
  --overview-resampling average \
  --blocksize 512 \
  input.tif output_cog.tif

# Validate the COG
rio cogeo validate output_cog.tif
```

**WEBP compression trick — 40-60% smaller than deflate/LZW:**

```bash
# For elevation data (lossless required)
gdal_translate -of COG \
  -co COMPRESS=WEBP \
  -co WEBP_LOSSLESS=YES \
  dem.tif dem_cog.tif

# For imagery (lossy acceptable)
gdal_translate -of COG \
  -co COMPRESS=WEBP \
  -co WEBP_LEVEL=85 \
  -co ADD_ALPHA=NO \
  imagery.tif imagery_cog.tif
```

**Upload to S3 and access directly from QGIS/GDAL:**

```bash
aws s3 cp dem_cog.tif s3://my-geodata/rasters/

# QGIS: Add Layer > Raster Layer > Source: /vsicurl/https://my-geodata.s3.amazonaws.com/rasters/dem_cog.tif
# Or with credentials: /vsis3/my-geodata/rasters/dem_cog.tif
```

**GDAL environment for fast COG access (set these always):**

```bash
export GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES=YES
export GDAL_HTTP_MULTIPLEX=YES
export GDAL_HTTP_VERSION=2
export CPL_VSIL_CURL_CACHE_SIZE=200000000
export GDAL_CACHEMAX=512
```

**STAC catalogs for organizing COG collections:**

```json
{
  "type": "Catalog",
  "id": "my-imagery",
  "description": "Orthoimagery collection 2024",
  "links": [
    {
      "rel": "item",
      "href": "./tiles/2024-01-15/index.json",
      "type": "application/geo+json"
    }
  ]
}
```

Tools for creating STAC catalogs from COG collections:

```bash
# stactools-core
pip install stactools
stac create-item dem_cog.tif dem_item.json

# Or use pystac for programmatic catalog creation
pip install pystac
```

---

## OGC API Standards (Next-Generation)

The OGC API family replaces the legacy WMS/WFS/WCS/WPS/CSW services with REST+JSON APIs that are easier to consume from modern clients. All use OpenAPI 3.0 specs, return GeoJSON, and follow standard HTTP conventions.

| Legacy Standard | OGC API Replacement |
|---|---|
| WFS | OGC API - Features |
| WMTS | OGC API - Tiles |
| WMS | OGC API - Maps |
| WPS | OGC API - Processes |
| CSW | OGC API - Records |

### pygeoapi

The easiest OGC API server to deploy. Pure Python, YAML config, pluggable backends.

```bash
pip install pygeoapi
```

```yaml
# pygeoapi-config.yml
server:
  bind:
    host: 0.0.0.0
    port: 5000
  url: https://api.maps.example.com
  mimetype: application/json
  encoding: utf-8
  language: en-US
  cors: true
  pretty_print: true
  limit: 10
  map:
    url: https://tile.openstreetmap.org/{z}/{x}/{y}.png
    attribution: "© OpenStreetMap contributors"

logging:
  level: ERROR

metadata:
  identification:
    title: My Geospatial API
    description: OGC API Features and Tiles
    keywords: [geospatial, OGC, API]
    license:
      name: CC-BY 4.0

resources:
  municipalities:
    type: collection
    title: Municipalities
    description: Municipal boundaries with population data
    keywords: [municipalities, boundaries]
    extents:
      spatial:
        bbox: [-180, -90, 180, 90]
        crs: http://www.opengis.net/def/crs/OGC/1.3/CRS84
    links:
      - type: text/html
        rel: canonical
        title: Data source
        href: https://data.example.com/municipalities
    providers:
      - type: feature
        name: PostgreSQL
        data:
          host: pghost
          port: 5432
          dbname: geodata
          user: pygeoapi
          password: secret
          search_path: [public]
        id_field: gid
        table: municipalities
        geom_field: geom

  dem_tiles:
    type: collection
    title: Digital Elevation Model Tiles
    providers:
      - type: tile
        name: MVT-tippecanoe
        data: /data/tiles/contours.pmtiles
        options:
          zoom:
            min: 8
            max: 16
          schemes: [GoogleMapsCompatible]
        format:
          name: pbf
          mimetype: application/vnd.mapbox-vector-tile
```

```bash
# Run with gunicorn for production
gunicorn pygeoapi.flask_app:APP \
  --workers 4 \
  --bind 0.0.0.0:5000 \
  --timeout 90

# Docker
docker run -p 5000:80 \
  -v $(pwd)/pygeoapi-config.yml:/pygeoapi/local.config.yml \
  geopython/pygeoapi
```

**OGC API Features endpoints:**

```
GET /collections                          # list all collections
GET /collections/municipalities           # collection metadata
GET /collections/municipalities/items     # GeoJSON FeatureCollection
GET /collections/municipalities/items?bbox=100,1,105,5&limit=100
GET /collections/municipalities/items?population_gt=1000000
GET /collections/municipalities/items/{id}  # single feature
```

---

## Geocoding & Search Services

### Nominatim

Self-hosted OpenStreetMap geocoder. Free, no API key, no rate limits on your own instance.

**Reality check on hardware requirements:**
- Planet import: 64GB+ RAM during import, 32GB sufficient for serving, 1TB+ NVMe SSD, 2-3 days import time
- Country extract: 8GB RAM, 50-200GB disk, 1-4 hours

```bash
# Docker with country extract (recommended approach for most use cases)
docker run -it \
  -e PBF_URL=https://download.geofabrik.de/europe/germany-latest.osm.pbf \
  -e REPLICATION_URL=https://download.geofabrik.de/europe/germany-updates/ \
  -e REPLICATION_UPDATE_INTERVAL=86400 \
  -e NOMINATIM_PASSWORD=secret \
  -v nominatim-data:/var/lib/postgresql/14/main \
  -p 8088:8080 \
  --shm-size=1g \
  mediagis/nominatim:4.4
```

**API usage:**

```bash
# Forward geocoding
curl "http://localhost:8088/search?q=Berlin+Hauptbahnhof&format=jsonv2&limit=5"

# Structured search (more reliable than free text)
curl "http://localhost:8088/search?street=Unter+den+Linden&city=Berlin&country=Germany&format=jsonv2"

# Reverse geocoding
curl "http://localhost:8088/reverse?lat=52.5200&lon=13.4050&format=jsonv2&zoom=18"

# Autocomplete (needs special configuration)
curl "http://localhost:8088/search?q=Berl&format=jsonv2&addressdetails=1&limit=10"
```

---

### Pelias

Modular geocoder combining Who's On First, OpenStreetMap, OpenAddresses, and Geonames data. Better multilingual support and structured results than Nominatim for many use cases.

```bash
# Clone the docker-compose setup
git clone https://github.com/pelias/docker
cd docker
# Edit .env to select country/city imports
cat > .env << EOF
DATA_DIR=/data/pelias
DOCKER_USER=1000:1000
EOF

# Download data and import (country example: Germany)
./pelias compose pull
./pelias elastic start
./pelias elastic wait
./pelias elastic create
./pelias download all
./pelias prepare all
./pelias import all
./pelias compose up
```

```bash
# Search API
curl "http://localhost:4000/v1/search?text=Brandenburg+Gate&focus.point.lat=52.5200&focus.point.lon=13.4050"

# Autocomplete
curl "http://localhost:4000/v1/autocomplete?text=Brandenb"

# Reverse
curl "http://localhost:4000/v1/reverse?point.lat=52.5163&point.lon=13.3777"

# Structured search
curl "http://localhost:4000/v1/search/structured?address=Pariser+Platz+1&locality=Berlin&country=DEU"
```

---

### Photon

ElasticSearch-backed geocoder from Komoot. Excellent autocomplete performance and multilingual support. Simpler to operate than Pelias.

```bash
# Download pre-built index (easier than building from scratch)
wget https://download1.graphhopper.com/public/photon-db-latest.tar.bz2
tar -xjf photon-db-latest.tar.bz2

# Run
java -jar photon-*.jar -listen-ip 0.0.0.0

# Or Docker
docker run -p 2322:2322 \
  -v /data/photon:/photon/photon_data \
  rtuszik/photon-docker:latest
```

```bash
# Geocode with language preference
curl "http://localhost:2322/api?q=Tokyo+Station&lang=en&limit=5"

# Near a location (bias results)
curl "http://localhost:2322/api?q=cafe&lat=35.6762&lon=139.6503&limit=10"

# Reverse
curl "http://localhost:2322/reverse?lat=35.6762&lon=139.6503&lang=ja"
```

---

## Reverse Proxy & Caching Patterns

### Nginx configuration for GeoServer/tile servers

```nginx
# /etc/nginx/sites-available/geoserver
proxy_cache_path /var/cache/nginx/tiles
  levels=1:2
  keys_zone=tile_cache:100m
  max_size=50g
  inactive=30d
  use_temp_path=off;

upstream geoserver {
    server localhost:8080;
    keepalive 32;
}

server {
    listen 443 ssl http2;
    server_name maps.example.com;

    ssl_certificate /etc/letsencrypt/live/maps.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/maps.example.com/privkey.pem;

    # CORS headers for tile access from any origin
    add_header Access-Control-Allow-Origin "*" always;
    add_header Access-Control-Allow-Methods "GET, HEAD, OPTIONS" always;
    add_header Access-Control-Allow-Headers "Range, If-Modified-Since" always;
    add_header Access-Control-Expose-Headers "Content-Range, Content-Length" always;

    # Handle CORS preflight
    if ($request_method = OPTIONS) {
        return 204;
    }

    # Cache WMTS and WMS GetMap requests
    location ~* /geoserver/(.*/gwc/service/|.*\?.*SERVICE=WMS.*REQUEST=GetMap) {
        proxy_pass http://geoserver;
        proxy_cache tile_cache;
        proxy_cache_valid 200 7d;
        proxy_cache_valid 404 1m;
        proxy_cache_use_stale error timeout updating;
        proxy_cache_background_update on;
        proxy_cache_lock on;
        proxy_cache_key "$scheme$request_method$host$request_uri";

        add_header X-Cache-Status $upstream_cache_status;
        add_header Cache-Control "public, max-age=604800";

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }

    # GeoServer admin — do not cache
    location /geoserver/web {
        proxy_pass http://geoserver;
        proxy_set_header Host $host;
    }

    # Martin tile server
    location /tiles/ {
        proxy_pass http://localhost:3000/;
        proxy_cache tile_cache;
        proxy_cache_valid 200 1d;
        proxy_cache_key "$scheme$request_method$host$request_uri";
        add_header X-Cache-Status $upstream_cache_status;
        gzip on;
        gzip_types application/vnd.mapbox-vector-tile application/json;
    }

    # Static PMTiles served from disk
    location /static/tiles/ {
        alias /data/pmtiles/;
        add_header Content-Type application/octet-stream;
        add_header Accept-Ranges bytes;
        expires 30d;
    }
}
```

### MapProxy — the universal tile proxy

MapProxy sits in front of any WMS/WMTS/TMS/ArcGIS server and caches, transforms, merges, and re-serves tiles. It is the right tool when you need to:
- Cache a slow upstream WMS as fast WMTS tiles
- Merge multiple map services into a single layer
- Reproject tiles from one SRS to another
- Provide a unified API in front of heterogeneous backends

```yaml
# mapproxy.yaml
services:
  wmts:
    restful: true
    kvp: true
  tms:
    use_grid_names: true
  wms:
    srs: ["EPSG:4326", "EPSG:3857"]
    image_formats: ["image/png", "image/webp"]

layers:
  - name: merged_basemap
    title: Merged Basemap
    sources: [merged_cache]

caches:
  merged_cache:
    grids: [GLOBAL_WEBMERCATOR]
    sources: [slow_wms, satellite_wms]
    cache:
      type: file
      directory: /var/cache/mapproxy/merged
    format: image/webp
    request_format: image/png
    minimize_meta_requests: true

sources:
  slow_wms:
    type: wms
    req:
      url: http://internal-server:8080/geoserver/wms
      layers: workspace:baselayer
    coverage:
      bbox: [100.0, 1.0, 120.0, 25.0]
      srs: "EPSG:4326"

  satellite_wms:
    type: wms
    req:
      url: https://external-satellite-service.com/wms
      layers: satellite_2024
    http:
      headers:
        Authorization: "Bearer MYTOKEN"

grids:
  GLOBAL_WEBMERCATOR:
    base: GLOBAL_WEBMERCATOR
    num_levels: 19
```

```bash
# Run MapProxy
pip install mapproxy
mapproxy-util serve-develop mapproxy.yaml

# Or production with gunicorn
gunicorn -k gevent -w 8 \
  --bind 0.0.0.0:8080 \
  "mapproxy.wsgiapp:make_wsgi_app('mapproxy.yaml')"
```

---

## Infrastructure as Code

### Docker Compose: Full Production Stack

```yaml
# docker-compose.production.yml
version: "3.8"

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - nginx_cache:/var/cache/nginx/tiles
      - letsencrypt:/etc/letsencrypt:ro
      - /data/pmtiles:/data/pmtiles:ro
    depends_on:
      - geoserver
      - martin
      - pygeoapi

  geoserver:
    image: kartoza/geoserver:2.24.0
    environment:
      JAVA_OPTS: "-Xms2g -Xmx8g -XX:+UseG1GC"
      GEOSERVER_ADMIN_PASSWORD: "${GEOSERVER_PASSWORD}"
    volumes:
      - geoserver_data:/opt/geoserver/data_dir
    depends_on:
      postgis:
        condition: service_healthy

  martin:
    image: ghcr.io/maplibre/martin:latest
    command: --config /config/martin.yaml
    volumes:
      - ./martin.yaml:/config/martin.yaml:ro
      - /data/tiles:/data/tiles:ro
    environment:
      DATABASE_URL: "postgresql://martin:${PG_PASSWORD}@postgis:5432/geodata"
    depends_on:
      postgis:
        condition: service_healthy

  pygeoapi:
    image: geopython/pygeoapi:latest
    volumes:
      - ./pygeoapi.yaml:/pygeoapi/local.config.yml:ro
    environment:
      PYGEOAPI_CONFIG: /pygeoapi/local.config.yml
      PYGEOAPI_OPENAPI: /pygeoapi/local.openapi.yml

  postgis:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_DB: geodata
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: "${PG_PASSWORD}"
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./pg/postgresql.conf:/etc/postgresql/postgresql.conf:ro
      - ./pg/init:/docker-entrypoint-initdb.d:ro
    command: postgres -c config_file=/etc/postgresql/postgresql.conf
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U admin -d geodata"]
      interval: 10s
      timeout: 5s
      retries: 5

  nominatim:
    image: mediagis/nominatim:4.4
    environment:
      PBF_URL: "https://download.geofabrik.de/asia/japan-latest.osm.pbf"
      REPLICATION_URL: "https://download.geofabrik.de/asia/japan-updates/"
      NOMINATIM_PASSWORD: "${NOMINATIM_PASSWORD}"
    volumes:
      - nominatim_data:/var/lib/postgresql/14/main
    ports:
      - "8088:8080"
    shm_size: "1g"

volumes:
  pg_data:
  geoserver_data:
  nginx_cache:
  nominatim_data:
  letsencrypt:
```

### PostgreSQL tuning config for spatial workloads:

```ini
# postgresql.conf — for a 32GB RAM server with SSD
max_connections = 200
shared_buffers = 8GB
effective_cache_size = 24GB
maintenance_work_mem = 2GB
checkpoint_completion_target = 0.9
wal_buffers = 64MB
default_statistics_target = 500
random_page_cost = 1.1          # SSD: 1.1, HDD: 4.0
effective_io_concurrency = 200  # SSD: 200, HDD: 2
work_mem = 64MB
min_wal_size = 2GB
max_wal_size = 8GB
max_worker_processes = 8
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_parallel_maintenance_workers = 4

# PostGIS-specific
enable_partitionwise_aggregate = on
jit = off  # Disable JIT for spatial queries — often slower with complex geometries
```

---

## Advanced Dark Arts

### Hand-crafted MVT tiles bypassing the tile server

For maximum control and performance, generate tiles directly in PostGIS and serve them from Nginx with a `try_files` cache pattern:

```sql
-- Stored function that generates a complete MVT tile
CREATE OR REPLACE FUNCTION public.get_tile(
  z integer, x integer, y integer
)
RETURNS bytea
LANGUAGE plpgsql STABLE PARALLEL SAFE
AS $$
DECLARE
  bounds geometry := ST_TileEnvelope(z, x, y);
  result bytea;
BEGIN
  WITH
  tile_bounds AS (
    SELECT
      bounds AS geom,
      ST_Transform(bounds, 4326) AS geom_4326
  ),
  roads_layer AS (
    SELECT
      ST_AsMVTGeom(
        ST_Transform(r.geom, 3857),
        bounds,
        4096, 256, true
      ) AS geom,
      r.name, r.class, r.ref,
      CASE WHEN z >= 14 THEN r.maxspeed ELSE NULL END AS maxspeed
    FROM roads r, tile_bounds tb
    WHERE r.geom && tb.geom_4326
      AND ST_AsMVTGeom(ST_Transform(r.geom, 3857), bounds, 4096, 256, true) IS NOT NULL
      AND (
        (z >= 14 AND r.class IN ('residential','tertiary','secondary','primary','trunk','motorway'))
        OR (z >= 10 AND r.class IN ('secondary','primary','trunk','motorway'))
        OR (z >= 7 AND r.class IN ('trunk','motorway'))
      )
  ),
  buildings_layer AS (
    SELECT
      ST_AsMVTGeom(
        ST_Transform(b.geom, 3857),
        bounds,
        4096, 4, true
      ) AS geom,
      b.height, b.type
    FROM buildings b, tile_bounds tb
    WHERE z >= 14
      AND b.geom && tb.geom_4326
  )
  SELECT INTO result
    (
      (SELECT ST_AsMVT(roads_layer, 'roads', 4096, 'geom') FROM roads_layer)
      ||
      (SELECT ST_AsMVT(buildings_layer, 'buildings', 4096, 'geom') FROM buildings_layer)
    );

  RETURN result;
END;
$$;
```

**Nginx tile cache with PostgreSQL fallback** (the `try_files` trick):

```nginx
# Nginx config: serve cached tiles, fall through to generator on miss
location ~ ^/tiles/(\d+)/(\d+)/(\d+)\.pbf$ {
    set $z $1;
    set $x $2;
    set $y $3;

    # Try disk cache first
    try_files /var/cache/tiles/$z/$x/$y.pbf @generate_tile;

    add_header Content-Type "application/vnd.mapbox-vector-tile";
    add_header Content-Encoding gzip;
    add_header Cache-Control "public, max-age=86400";
}

location @generate_tile {
    # Pass to a small FastAPI/Flask app that queries PostGIS
    proxy_pass http://localhost:8001/generate/$z/$x/$y;
    proxy_cache tile_cache;
    proxy_cache_valid 200 24h;
    proxy_cache_key "$z/$x/$y";
    # Store to disk for future try_files hits
    proxy_store /var/cache/tiles/$z/$x/$y.pbf;
    proxy_store_access user:rw group:r all:r;
}
```

### GeoServer SQL Views with user-driven dynamic maps

Combine SQL views with `viewparams` to let client applications drive dynamic spatial queries. Add `%parameter%` placeholders with regex validators:

```
# WMS GetMap with dynamic SQL view parameters
http://geoserver/wms?
  SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1
  &LAYERS=workspace:dynamic_analysis
  &viewparams=start_year:2020;end_year:2024;min_value:1000;region_code:JP
  &BBOX=129.5,30.9,145.9,45.5
  &WIDTH=1024&HEIGHT=768
  &SRS=EPSG:4326
  &FORMAT=image/png
```

The SQL view definition uses `%start_year%`, `%end_year%` etc. with validator regex `^[0-9]{4}$` — GeoServer blocks SQL injection via the validator before passing to the database.

### pg_cron for scheduled materialized spatial views

Heavy spatial aggregations (administrative statistics, density calculations) can be pre-computed on a schedule and served as fast table reads:

```sql
-- Install pg_cron
CREATE EXTENSION pg_cron;

-- Create a materialized spatial analysis table
CREATE MATERIALIZED VIEW public.grid_density_100m AS
SELECT
  g.cell_id,
  g.geom,
  COUNT(p.gid) AS building_count,
  SUM(p.floor_area) AS total_floor_area,
  AVG(p.height) AS avg_height
FROM grid_100m g
LEFT JOIN buildings p ON ST_Within(p.geom, g.geom)
GROUP BY g.cell_id, g.geom
WITH DATA;

CREATE INDEX ON public.grid_density_100m USING GIST (geom);
CREATE INDEX ON public.grid_density_100m (building_count);

-- Refresh every night at 2 AM
SELECT cron.schedule(
  'refresh-density-grid',
  '0 2 * * *',
  'REFRESH MATERIALIZED VIEW CONCURRENTLY public.grid_density_100m'
);

-- More complex: incremental update function
CREATE OR REPLACE FUNCTION public.update_changed_cells()
RETURNS void LANGUAGE plpgsql AS $$
BEGIN
  -- Update only cells containing recently modified buildings
  UPDATE grid_density_100m d
  SET
    building_count = sub.count,
    total_floor_area = sub.area,
    avg_height = sub.avg_h
  FROM (
    SELECT
      g.cell_id,
      COUNT(p.gid) AS count,
      SUM(p.floor_area) AS area,
      AVG(p.height) AS avg_h
    FROM grid_100m g
    JOIN buildings p ON ST_Within(p.geom, g.geom)
    WHERE p.updated_at > NOW() - INTERVAL '1 day'
    GROUP BY g.cell_id
  ) sub
  WHERE d.cell_id = sub.cell_id;
END;
$$;

SELECT cron.schedule('incremental-update', '*/30 * * * *', 'SELECT public.update_changed_cells()');
```

### GeoServer Importer Extension for bulk REST loading

The Importer extension exposes a REST API for uploading and publishing data files (Shapefiles, GeoPackages, GeoTIFFs, CSVs) in batch:

```bash
# Upload and import a Shapefile
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/imports" \
  -H "Content-Type: application/json" \
  -d '{"import":{"targetWorkspace":{"workspace":{"name":"production"}}}}'

# Get the import ID from response, then upload the file
IMPORT_ID=1
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/imports/${IMPORT_ID}/tasks" \
  -F "filedata=@municipalities.zip"

# Run the import
curl -u admin:geoserver -XPOST \
  "http://localhost:8080/geoserver/rest/imports/${IMPORT_ID}"

# Check status
curl -u admin:geoserver \
  "http://localhost:8080/geoserver/rest/imports/${IMPORT_ID}"
```

Automate bulk publishing with a shell script:

```bash
#!/bin/bash
# bulk-publish.sh: Publish all shapefiles in a directory to GeoServer
GS_URL="http://localhost:8080/geoserver"
GS_AUTH="admin:geoserver"
WORKSPACE="production"
DATA_DIR="/data/shapefiles"

for shp in "$DATA_DIR"/*.shp; do
  layer=$(basename "$shp" .shp)
  zipfile="/tmp/${layer}.zip"

  # Zip the shapefile components
  zip -j "$zipfile" "$DATA_DIR/${layer}".{shp,shx,dbf,prj}

  # Create import
  import_id=$(curl -s -u "$GS_AUTH" -XPOST \
    "$GS_URL/rest/imports" \
    -H "Content-Type: application/json" \
    -d "{\"import\":{\"targetWorkspace\":{\"workspace\":{\"name\":\"$WORKSPACE\"}}}}" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['import']['id'])")

  # Upload
  curl -s -u "$GS_AUTH" -XPOST \
    "$GS_URL/rest/imports/${import_id}/tasks" \
    -F "filedata=@${zipfile}"

  # Execute
  curl -s -u "$GS_AUTH" -XPOST \
    "$GS_URL/rest/imports/${import_id}"

  echo "Published: $layer (import $import_id)"
  rm "$zipfile"
done
```

### Headless QGIS for server-side print/export

Run QGIS Server as a print engine — hand it a layout template and data params, get back a high-quality PDF or PNG:

```python
# print_layout.py — invoke QGIS Server's GetPrint request
import requests

params = {
    "SERVICE": "WMS",
    "VERSION": "1.3.0",
    "REQUEST": "GetPrint",
    "FORMAT": "pdf",
    "TRANSPARENT": "true",
    "DPI": "300",
    "CRS": "EPSG:4326",
    "template": "A3 Portrait",  # layout name in the QGIS project
    "map0:EXTENT": "139.5,35.5,140.5,36.0",
    "map0:SCALE": "100000",
    "map0:LAYERS": "basemap,municipalities,roads",
    "ATLAS_PK": "13",  # trigger atlas feature 13
}

response = requests.get(
    "http://localhost:8081/qgis-server",
    params=params,
    timeout=120
)

with open("output.pdf", "wb") as f:
    f.write(response.content)
```

Scale-driven rendering, atlas-driven multi-page exports, and full QGIS symbology all work through this endpoint. This is far simpler than trying to replicate QGIS cartography in a separate rendering engine.

---

## Decision Framework

| Need | Recommended Choice |
|---|---|
| Full OGC WMS/WFS stack, enterprise users | GeoServer |
| PostGIS → vector tiles, production traffic | Martin |
| Quick PostGIS vector tile preview/dev | pg_tileserv |
| COG/raster dynamic tiling, serverless | TiTiler on Lambda |
| Static tile hosting, zero infrastructure | PMTiles on Cloudflare R2 |
| OSM geocoding, self-hosted | Nominatim (country extract) |
| Multilingual geocoding with autocomplete | Photon or Pelias |
| OGC API Features, modern REST | pygeoapi |
| Proxy and cache multiple upstream services | MapProxy |
| QGIS project as a web service | QGIS Server + Lizmap |
| High-concurrency simple WMS, minimal RAM | MapServer |
