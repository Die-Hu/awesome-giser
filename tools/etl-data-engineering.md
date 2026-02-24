# Geospatial ETL, Data Engineering & DevOps

Production geospatial pipelines demand more than a one-off script. This guide covers the full stack: ETL platforms, format conversion, data quality, versioning, containerization, CI/CD, and cloud infrastructure — all from the perspective of someone who has broken production systems and learned from it.

**Quick Picks**
- **SOTA (State of the Art)**: FME + DuckDB Spatial — best combination of GUI power and zero-dependency analytical speed
- **Free Best**: GDAL + Apache Sedona — unbeatable open-source combo for any scale
- **Fastest Setup**: DuckDB Spatial — one binary, no server, runs spatial SQL on files in seconds

---

## Table of Contents

1. [ETL Platforms](#etl-platforms)
2. [Format Conversion Workflows](#format-conversion-workflows)
3. [Data Quality & Validation](#data-quality--validation)
4. [Geospatial Version Control](#geospatial-version-control)
5. [Data Catalogs & Metadata](#data-catalogs--metadata)
6. [Docker & Containerization](#docker--containerization)
7. [CI/CD for Geospatial](#cicd-for-geospatial)
8. [Cloud Infrastructure Patterns](#cloud-infrastructure-patterns)
9. [Performance & Optimization](#performance--optimization)
10. [Advanced Dark Arts](#advanced-dark-arts)

---

## ETL Platforms

### FME (Feature Manipulation Engine)

THE enterprise standard for spatial ETL. 500+ format readers/writers, a visual workflow builder, and a connector ecosystem that covers practically every data source on earth.

**Product Line**

| Product | Description | Deployment |
|---|---|---|
| FME Form | Visual drag-and-drop workspace canvas | Desktop |
| FME Flow | Automated scheduled workflows, web services, notification triggers | Self-hosted server |
| FME Flow Hosted | Fully managed cloud SaaS version | Cloud |

**Core Transformer Categories**

- **Attribute manipulation**: `AttributeManager`, `AttributeCreator`, `AttributeRenamer`, `AttributeFilter`
- **Spatial operations**: `Clipper`, `SpatialFilter`, `Reprojector`, `SpatialRelator`, `AreaCalculator`
- **Data integration**: `FeatureMerger`, `FeatureJoiner`, `DatabaseJoiner`
- **I/O**: `DatabaseInserter`, `FeatureReader`, `FeatureWriter`, `HTTPCaller`
- **Geometry**: `GeometryCoercer`, `Aggregator`, `Dissolver`, `Snapper`

**Format Coverage**

Every database (PostGIS, Oracle Spatial, SQL Server, MySQL, SQLite/SpatiaLite), every vector format (DWG, DGN, Revit, IFC, Shapefile, GeoJSON, GeoPackage, GML, KML, PDF), every raster format (GeoTIFF, ECW, MrSID, NetCDF, HDF5), web services (REST, WFS, WMS, WMTS, STAC), and cloud storage (S3, Azure Blob, GCS).

**Power Tricks**

- **FME workspace as a REST API**: Deploy a workspace on FME Flow, expose it as a Data Download or Data Streaming service. Clients POST parameters, get transformed data back. This is a surprisingly fast way to build a spatial ETL microservice without writing a single line of server code.

- **PythonCaller transformer**: Drop in arbitrary Python when built-in transformers fall short. Full access to the FME Python API, plus any installed Python packages (geopandas, shapely, rasterio, etc.).

```python
# PythonCaller example: buffer features by an attribute value
import fme
import fmeobjects

class FeatureProcessor(object):
    def input(self, feature):
        buffer_dist = feature.getAttribute('buffer_distance_m')
        geom = feature.getGeometry()
        # Use FME geometry API to buffer
        buffered = fmeobjects.FMEGeometryCoercer().coerce(geom)
        feature.setGeometry(buffered)
        self.pyoutput(feature)

    def close(self):
        pass
```

- **FeatureReader/FeatureWriter for dynamic I/O**: Use attribute values to construct dynamic dataset paths or table names at runtime. Combine with `AttributeCreator` to build paths like `s3://bucket/%(region)s/%(year)s/parcels.gpkg` — each feature can write to a different destination.

- **Custom transformers for CI/CD**: Package your reusable logic as a custom transformer (`.fmx`), check it into git, reference it from multiple workspaces. When you update the transformer, all workspaces that reference it pick up the change automatically.

**Cost**: ~$3,000+/yr per license. Free for students via the Safe Software Academic program. FME Community edition is free for non-commercial use with some limitations.

---

### Apache Sedona (formerly GeoSpark)

Distributed spatial processing built on Apache Spark. When your spatial join involves 5 billion GPS points and 50 million parcels, this is where you go.

**Capabilities**

- Spatial SQL on datasets that don't fit on one machine
- Reads: GeoParquet, Shapefile, GeoJSON, WKT/WKB columns in CSV, NetCDF, GeoTIFF
- Spatial joins (point-in-polygon, within, intersects), KNN queries, range queries, raster algebra — all distributed across a Spark cluster
- Integrates with Delta Lake for ACID spatial data lakes

**Wherobots**: Managed Sedona cloud platform. Removes the pain of cluster management and adds a spatial notebook environment with optimized hardware profiles for raster vs. vector workloads.

**PySpark + Sedona: Spatial Join Example**

```python
from pyspark.sql import SparkSession
from sedona.register import SedonaRegistrator
from sedona.utils import SedonaKryoRegistrator, KryoSerializer

spark = (SparkSession.builder
    .appName("SpatialJoin")
    .config("spark.serializer", KryoSerializer.getName)
    .config("spark.kryo.registrator", SedonaKryoRegistrator.getName)
    .config("spark.jars.packages",
            "org.apache.sedona:sedona-spark-3.4_2.12:1.5.1,"
            "org.datasyslab:geotools-wrapper:1.5.1-28.2")
    .getOrCreate())

SedonaRegistrator.registerAll(spark)

# Load a huge point dataset (GPS pings) and polygon dataset (parcels)
points_df = spark.read.format("parquet").load("s3://data/gps_pings/")
parcels_df = spark.read.format("geoparquet").load("s3://data/parcels/")

points_df.createOrReplaceTempView("points")
parcels_df.createOrReplaceTempView("parcels")

# Distributed point-in-polygon join using spatial SQL
result = spark.sql("""
    SELECT
        p.ping_id,
        p.user_id,
        p.timestamp,
        c.parcel_id,
        c.land_use_code
    FROM points p, parcels c
    WHERE ST_Contains(c.geometry, p.geometry)
""")

# Write results as GeoParquet partitioned by date
result.write \
    .partitionBy("date") \
    .format("geoparquet") \
    .mode("overwrite") \
    .save("s3://output/point_in_parcel/")
```

**Raster processing with Sedona:**

```python
# Load and mosaic Sentinel-2 tiles
raster_df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.tif") \
    .load("s3://sentinel2/tiles/")

raster_df.createOrReplaceTempView("rasters")

result = spark.sql("""
    SELECT
        RS_BandAsArray(RS_FromGeoTiff(content), 1) as band1,
        RS_BandAsArray(RS_FromGeoTiff(content), 4) as band4,
        -- NDVI = (NIR - Red) / (NIR + Red)
        RS_NormalizedDifference(
            RS_FromGeoTiff(content), 4, 1
        ) as ndvi
    FROM rasters
""")
```

---

### GeoMesa

Distributed spatio-temporal database sitting on top of HBase, Cassandra, Accumulo, Kafka, Redis, or FileSystem backends. Built for the class of problem where you have billions of timestamped point observations (AIS ship tracks, aircraft positions, cellphone pings, IoT sensor events) and need millisecond-range queries.

**Why not just use PostGIS?** PostGIS tops out around a few hundred million rows before maintenance becomes painful. GeoMesa is designed to scale horizontally across a cluster with no upper bound on data volume.

**Key features**

- Z-order (Z2/Z3) and Hilbert curve spatial indexing for efficient range scans without reading the full dataset
- Time-bounded spatial queries: "all ships in this bounding box in the last 6 hours"
- GeoServer integration via `gt-geomesa` plugin for serving
- GDAL/OGR access via the GeoMesa OGR plugin

**Sample GeoMesa ingest config (HBase backend):**

```yaml
# geomesa-hbase-ds.yml
instanceId: prod-cluster
zookeepers: zk1:2181,zk2:2181,zk3:2181
catalog: geomesa_ais

feature-type:
  name: ais_position
  attributes:
    - name: mmsi
      type: String
      index: true
    - name: timestamp
      type: Date
    - name: speed
      type: Float
    - name: course
      type: Float
    - name: geom
      type: Point
      srid: 4326
      default: true

index:
  type: Z3         # 3D spatiotemporal index (x, y, time)
  resolution: 12   # Index precision
```

---

### Apache Airflow + Geo

Workflow orchestration for multi-step geospatial pipelines. Airflow's DAG model is a natural fit for "download satellite imagery → run atmospheric correction → generate NDVI → load to PostGIS → refresh materialized views → generate tiles" sequences.

**Custom Operators**

```python
# operators/gdal_operator.py
from airflow.models import BaseOperator
import subprocess

class GDALTranslateOperator(BaseOperator):
    """
    Runs gdal_translate with arbitrary options.
    Supports GDAL virtual filesystem paths (/vsicurl/, /vsis3/, /vsigz/).
    """
    def __init__(self, src, dst, options=None, **kwargs):
        super().__init__(**kwargs)
        self.src = src
        self.dst = dst
        self.options = options or []

    def execute(self, context):
        cmd = ["gdal_translate"] + self.options + [self.src, self.dst]
        self.log.info("Running: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        self.log.info(result.stdout)
        return self.dst
```

**DAG: Satellite Imagery Processing Pipeline**

```python
# dags/sentinel2_pipeline.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from operators.gdal_operator import GDALTranslateOperator

default_args = {
    "owner": "geo-team",
    "depends_on_past": False,
    "start_date": datetime(2024, 1, 1),
    "email_on_failure": True,
    "email": ["geo-alerts@company.com"],
    "retries": 2,
    "retry_delay": timedelta(minutes=10),
}

with DAG(
    dag_id="sentinel2_ndvi_pipeline",
    default_args=default_args,
    schedule_interval="0 6 * * *",  # 6am UTC daily
    catchup=False,
    tags=["satellite", "ndvi"],
) as dag:

    def download_scene(**context):
        """Query STAC API for latest Sentinel-2 scene and download."""
        from pystac_client import Client
        import requests, os

        client = Client.open("https://earth-search.aws.element84.com/v1")
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=[110.0, -45.0, 155.0, -10.0],  # Australia
            datetime="2024-01-01/2024-12-31",
            query={"eo:cloud_cover": {"lt": 20}},
            sortby="-datetime",
            max_items=1,
        )
        item = next(search.items())
        nir_url = item.assets["B08"].href   # NIR band
        red_url = item.assets["B04"].href   # Red band
        context["ti"].xcom_push(key="nir_url", value=nir_url)
        context["ti"].xcom_push(key="red_url", value=red_url)
        return item.id

    download = PythonOperator(
        task_id="download_scene",
        python_callable=download_scene,
    )

    def compute_ndvi(**context):
        import rasterio
        import numpy as np
        ti = context["ti"]
        nir_url = ti.xcom_pull(task_ids="download_scene", key="nir_url")
        red_url = ti.xcom_pull(task_ids="download_scene", key="red_url")

        with rasterio.open(f"/vsicurl/{nir_url}") as nir_src, \
             rasterio.open(f"/vsicurl/{red_url}") as red_src:
            nir = nir_src.read(1).astype(float)
            red = red_src.read(1).astype(float)
            ndvi = np.where((nir + red) == 0, -9999, (nir - red) / (nir + red))
            profile = nir_src.profile.copy()
            profile.update(dtype=rasterio.float32, nodata=-9999, compress="deflate")

        out_path = "/tmp/ndvi_today.tif"
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(ndvi.astype(np.float32), 1)
        return out_path

    ndvi = PythonOperator(
        task_id="compute_ndvi",
        python_callable=compute_ndvi,
    )

    to_cog = GDALTranslateOperator(
        task_id="convert_to_cog",
        src="/tmp/ndvi_today.tif",
        dst="/tmp/ndvi_today_cog.tif",
        options=["-of", "COG", "-co", "COMPRESS=DEFLATE",
                 "-co", "OVERVIEW_RESAMPLING=AVERAGE"],
    )

    load_postgis = PostgresOperator(
        task_id="load_to_postgis",
        postgres_conn_id="postgis_prod",
        sql="""
            DELETE FROM raster.ndvi_daily
            WHERE capture_date = CURRENT_DATE;

            INSERT INTO raster.ndvi_daily (capture_date, rast)
            SELECT CURRENT_DATE, rast
            FROM (
                SELECT ST_Tile(
                    ST_AddBand(
                        ST_MakeEmptyRaster(1, 1, 0, 0, 1),
                        ARRAY[ROW(1, '32BF', 0, -9999)]::addbandarg[]
                    ), 256, 256
                ) AS rast
            ) tiles;
        """,
    )

    refresh_tiles = PostgresOperator(
        task_id="refresh_materialized_view",
        postgres_conn_id="postgis_prod",
        sql="REFRESH MATERIALIZED VIEW CONCURRENTLY geo.ndvi_summary_tiles;",
    )

    download >> ndvi >> to_cog >> load_postgis >> refresh_tiles
```

---

### Dagster

Modern data orchestration that treats data as assets rather than tasks. Better developer experience than Airflow: type checking, lineage graphs, per-asset freshness policies, built-in data quality checks.

**Geo pipeline: Daily satellite download → process → PostGIS → tiles**

```python
# geo_pipeline/assets.py
from dagster import asset, AssetIn, Output, MetadataValue
import geopandas as gpd
import pandas as pd
from datetime import date

@asset(
    group_name="satellite",
    description="Latest Sentinel-2 NDVI scene metadata from STAC",
)
def stac_scene_metadata(context) -> dict:
    from pystac_client import Client
    client = Client.open("https://earth-search.aws.element84.com/v1")
    search = client.search(
        collections=["sentinel-2-l2a"],
        bbox=[110.0, -45.0, 155.0, -10.0],
        query={"eo:cloud_cover": {"lt": 15}},
        sortby="-datetime",
        max_items=1,
    )
    item = next(search.items())
    context.add_output_metadata({"scene_id": item.id, "cloud_cover": item.properties["eo:cloud_cover"]})
    return {"id": item.id, "nir": item.assets["B08"].href, "red": item.assets["B04"].href}


@asset(
    group_name="satellite",
    ins={"scene": AssetIn("stac_scene_metadata")},
    description="NDVI raster computed from scene bands",
)
def ndvi_raster(context, scene: dict) -> str:
    import rasterio, numpy as np
    with rasterio.open(f"/vsicurl/{scene['nir']}") as n, \
         rasterio.open(f"/vsicurl/{scene['red']}") as r:
        nir = n.read(1).astype(float)
        red = r.read(1).astype(float)
        ndvi = (nir - red) / (nir + red + 1e-10)
        profile = n.profile
        profile.update(dtype="float32", compress="deflate")
    path = f"/data/ndvi/{date.today()}.tif"
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(ndvi.astype("float32"), 1)
    context.add_output_metadata({"path": path, "shape": str(ndvi.shape)})
    return path


@asset(
    group_name="database",
    ins={"raster_path": AssetIn("ndvi_raster")},
    description="NDVI statistics loaded to PostGIS",
    required_resource_keys={"postgis"},
)
def ndvi_postgis(context, raster_path: str) -> None:
    import subprocess
    conn = context.resources.postgis.connection_string
    subprocess.run([
        "raster2pgsql", "-a", "-C", "-F", "-t", "256x256",
        raster_path, "raster.ndvi_daily"
    ], check=True, capture_output=True)
    context.add_output_metadata({"loaded_to": "raster.ndvi_daily"})
```

---

### Prefect

Python-native workflow orchestration. Friendlier API than Airflow for teams without a dedicated platform engineer. No XML, no complex DAG setup — just Python decorators.

```python
# flows/process_boundaries.py
from prefect import flow, task, get_run_logger
import geopandas as gpd
from sqlalchemy import create_engine
import subprocess

@task(retries=3, retry_delay_seconds=60)
def fetch_boundaries(url: str) -> gpd.GeoDataFrame:
    logger = get_run_logger()
    logger.info(f"Fetching boundaries from {url}")
    gdf = gpd.read_file(url)
    return gdf

@task
def validate_and_fix(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    logger = get_run_logger()
    invalid = gdf[~gdf.geometry.is_valid]
    if len(invalid) > 0:
        logger.warning(f"Fixing {len(invalid)} invalid geometries")
        gdf.loc[~gdf.geometry.is_valid, "geometry"] = (
            gdf[~gdf.geometry.is_valid].geometry.buffer(0)
        )
    return gdf

@task
def reproject(gdf: gpd.GeoDataFrame, target_crs: int) -> gpd.GeoDataFrame:
    return gdf.to_crs(epsg=target_crs)

@task
def load_postgis(gdf: gpd.GeoDataFrame, table: str, conn_str: str) -> int:
    engine = create_engine(conn_str)
    gdf.to_postgis(table, engine, if_exists="replace", index=False)
    return len(gdf)

@task
def generate_tiles(table: str) -> None:
    subprocess.run([
        "tippecanoe",
        "--output=/var/www/tiles/boundaries.pmtiles",
        "--maximum-zoom=12",
        "--minimum-zoom=0",
        "--drop-densest-as-needed",
        f"--read-parallel",
        f"pg:table={table}"
    ], check=True)

@flow(name="Process Administrative Boundaries", log_prints=True)
def process_boundaries_flow(
    source_url: str = "https://example.com/admin_boundaries.gpkg",
    target_crs: int = 4326,
    pg_conn: str = "postgresql://user:pass@localhost:5432/geoapp",
    table_name: str = "admin.boundaries",
):
    raw = fetch_boundaries(source_url)
    valid = validate_and_fix(raw)
    projected = reproject(valid, target_crs)
    count = load_postgis(projected, table_name, pg_conn)
    generate_tiles(table_name)
    print(f"Pipeline complete: {count} features loaded and tiled")

if __name__ == "__main__":
    process_boundaries_flow()
```

---

## Format Conversion Workflows

### The GDAL/OGR Conversion Matrix

Preferred conversion paths between common geospatial formats:

| Source → Target | Command |
|---|---|
| Shapefile → GeoPackage | `ogr2ogr -f GPKG output.gpkg input.shp` |
| GeoPackage → GeoJSON | `ogr2ogr -f GeoJSON -t_srs EPSG:4326 output.geojson input.gpkg` |
| GeoJSON → FlatGeobuf | `ogr2ogr -f FlatGeobuf output.fgb input.geojson` |
| FlatGeobuf → GeoParquet | `ogr2ogr -f Parquet output.parquet input.fgb` |
| GeoParquet → PostGIS | `ogr2ogr -f PostgreSQL PG:"host=localhost dbname=gis" input.parquet -nln schema.table` |
| PostGIS → PMTiles | `ogr2ogr -f GeoJSON /vsistdout/ PG:"..." -sql "SELECT * FROM table" \| tippecanoe -o output.pmtiles --force -zg` |
| GeoTIFF → COG | `gdal_translate -of COG -co COMPRESS=DEFLATE -co OVERVIEW_RESAMPLING=AVERAGE input.tif output_cog.tif` |
| MrSID → GeoTIFF | `gdal_translate -of GTiff -co COMPRESS=DEFLATE input.sid output.tif` |
| NetCDF → GeoTIFF | `gdal_translate NETCDF:"input.nc":variable output.tif -a_srs EPSG:4326` |
| ECW → COG | `gdal_translate -of COG -co COMPRESS=DEFLATE -co BIGTIFF=IF_SAFER input.ecw output_cog.tif` |
| DWG → GeoPackage | `ogr2ogr -f GPKG output.gpkg input.dwg` |

**CRS reprojection during conversion:**

```bash
# Reproject and convert in a single pass
ogr2ogr \
  -f GeoPackage output_wgs84.gpkg input_gda94.shp \
  -s_srs EPSG:4283 \
  -t_srs EPSG:4326 \
  -nlt PROMOTE_TO_MULTI
```

### Bulk Conversion Scripts

**Python batch vector conversion with progress bar:**

```python
#!/usr/bin/env python3
"""Batch convert vector files between formats with parallel processing."""
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse

FORMAT_EXTENSIONS = {
    "GPKG": ".gpkg",
    "GeoJSON": ".geojson",
    "FlatGeobuf": ".fgb",
    "Parquet": ".parquet",
    "ESRI Shapefile": ".shp",
}

def convert_file(src: Path, dst_dir: Path, out_format: str, extra_opts: list = None) -> tuple:
    """Convert a single file, return (src, success, error_msg)."""
    ext = FORMAT_EXTENSIONS.get(out_format, ".out")
    dst = dst_dir / (src.stem + ext)
    cmd = [
        "ogr2ogr",
        "-f", out_format,
        "-nlt", "PROMOTE_TO_MULTI",
        "-makevalid",
        str(dst),
        str(src),
    ] + (extra_opts or [])
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=300)
        return src, True, None
    except subprocess.CalledProcessError as e:
        return src, False, e.stderr.decode()
    except subprocess.TimeoutExpired:
        return src, False, "Timeout"

def batch_convert(
    src_dir: str,
    dst_dir: str,
    src_glob: str = "*.shp",
    out_format: str = "GPKG",
    workers: int = 4,
):
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    files = list(src_path.glob(src_glob))
    if not files:
        print(f"No files matched {src_glob} in {src_dir}")
        return

    print(f"Converting {len(files)} files to {out_format} using {workers} workers")
    errors = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(convert_file, f, dst_path, out_format): f
            for f in files
        }
        with tqdm(total=len(futures), unit="file") as pbar:
            for future in as_completed(futures):
                src, success, err = future.result()
                pbar.update(1)
                pbar.set_postfix_str(src.name[:30])
                if not success:
                    errors.append((src, err))

    print(f"\nDone. {len(files) - len(errors)}/{len(files)} succeeded.")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for src, err in errors:
            print(f"  {src.name}: {err}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir")
    parser.add_argument("dst_dir")
    parser.add_argument("--glob", default="*.shp")
    parser.add_argument("--format", default="GPKG")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    batch_convert(args.src_dir, args.dst_dir, args.glob, args.format, args.workers)
```

**GNU parallel + GDAL for multi-core batch raster conversion to COG:**

```bash
#!/bin/bash
# Convert all GeoTIFFs in ./input to COGs in ./output using all CPU cores
# Requires: gdal, parallel, find

SRC_DIR="./input"
DST_DIR="./output"
mkdir -p "$DST_DIR"

# Build conversion function
convert_to_cog() {
    src="$1"
    dst_dir="$2"
    filename=$(basename "$src" .tif)
    dst="${dst_dir}/${filename}_cog.tif"
    gdal_translate \
        -of COG \
        -co COMPRESS=DEFLATE \
        -co PREDICTOR=2 \
        -co OVERVIEW_RESAMPLING=AVERAGE \
        -co BIGTIFF=IF_SAFER \
        --config GDAL_NUM_THREADS ALL_CPUS \
        --config GDAL_CACHEMAX 2048 \
        "$src" "$dst"
    echo "Done: $filename"
}

export -f convert_to_cog

find "$SRC_DIR" -name "*.tif" -not -name "*_cog.tif" | \
  parallel --jobs $(nproc) --bar \
  convert_to_cog {} "$DST_DIR"

echo "All conversions complete."
```

**Incremental PostGIS loading with ogr2ogr -append:**

```bash
# Initial load (creates the table)
ogr2ogr \
  -f PostgreSQL \
  "PG:host=localhost dbname=gis user=postgres" \
  /data/parcels_2024_01.gpkg \
  -nln cadastre.parcels \
  -lco GEOMETRY_NAME=geom \
  -lco FID=gid \
  -lco SPATIAL_INDEX=GIST \
  -t_srs EPSG:4326

# Incremental append (add new data without recreating the table)
for file in /data/parcels_2024_*.gpkg; do
    echo "Appending $file..."
    ogr2ogr \
      -f PostgreSQL \
      "PG:host=localhost dbname=gis user=postgres" \
      "$file" \
      -nln cadastre.parcels \
      -append \
      -t_srs EPSG:4326
done

# After loading, VACUUM ANALYZE for query planner accuracy
psql -d gis -c "VACUUM ANALYZE cadastre.parcels;"
```

---

## Data Quality & Validation

| Tool | Focus | When to Use |
|---|---|---|
| `ST_IsValid` + `ST_MakeValid` | Invalid geometry repair | PostGIS in-database cleanup |
| `ogr2ogr -makevalid` | Invalid geometry repair | During any ogr2ogr conversion |
| mapshaper | Topology-aware simplification | When simplifying shared boundaries |
| GRASS v.clean | Full topological cleaning | Building clean topology from messy data |
| val3dity | 3D geometry validation | CityGML, 3D GeoJSON, IFC models |
| Great Expectations | Schema + statistical validation | Production pipeline data quality gates |

**PostGIS geometry validation workflow:**

```sql
-- 1. Identify invalid geometries and WHY they are invalid
SELECT
    gid,
    ST_IsValidReason(geom) as reason,
    GeometryType(geom) as geom_type,
    ST_NPoints(geom) as npoints
FROM cadastre.parcels
WHERE NOT ST_IsValid(geom)
ORDER BY reason;

-- Common reasons and fixes:
-- "Ring Self-intersection" → ST_Buffer(geom, 0) or ST_MakeValid(geom)
-- "Duplicate Rings" → ST_MakeValid(geom)
-- "Too few points in geometry" → DELETE these rows

-- 2. Fix in-place (most cases)
UPDATE cadastre.parcels
SET geom = ST_MakeValid(geom)
WHERE NOT ST_IsValid(geom);

-- 3. After fixing, verify no invalids remain
SELECT COUNT(*) as invalid_count
FROM cadastre.parcels
WHERE NOT ST_IsValid(geom);

-- 4. Fix mixed geometry types (sometimes needed after ST_MakeValid)
-- ST_MakeValid on a Polygon can return a GeometryCollection.
-- Force back to single type:
UPDATE cadastre.parcels
SET geom = ST_CollectionExtract(geom, 3)  -- 3 = POLYGON
WHERE GeometryType(geom) NOT IN ('POLYGON', 'MULTIPOLYGON');
```

**Fix mixed geometry types during ogr2ogr import:**

```bash
ogr2ogr \
  -f GPKG output_clean.gpkg \
  input_messy.shp \
  -nlt PROMOTE_TO_MULTI \   # Promotes Polygon to MultiPolygon, etc.
  -makevalid \               # Fixes invalid geometries on the fly
  -skipfailures              # Log errors and continue instead of stopping
```

**Great Expectations spatial pipeline check:**

```python
# ge_spatial_check.py
import great_expectations as gx
import geopandas as gpd
from shapely import is_valid

# Load data to validate
gdf = gpd.read_file("parcels.gpkg")
df = gdf.drop(columns="geometry")  # GE works on pandas DataFrames
df["geom_valid"] = is_valid(gdf.geometry)
df["geom_area_m2"] = gdf.geometry.to_crs(epsg=3857).area

context = gx.get_context()
data_source = context.sources.add_pandas("parcel_check")
data_asset = data_source.add_dataframe_asset("parcels")

batch_request = data_asset.build_batch_request(dataframe=df)
validator = context.get_validator(batch_request=batch_request)

# All geometries must be valid
validator.expect_column_values_to_be_in_set("geom_valid", value_set=[True])

# parcel_id must be unique and not null
validator.expect_column_values_to_not_be_null("parcel_id")
validator.expect_column_values_to_be_unique("parcel_id")

# Area must be within sane bounds (1m² to 100km²)
validator.expect_column_values_to_be_between(
    "geom_area_m2", min_value=1, max_value=100_000_000
)

# land_use_code must match known values
validator.expect_column_values_to_be_in_set(
    "land_use_code",
    value_set=["RESIDENTIAL", "COMMERCIAL", "INDUSTRIAL", "RURAL", "PUBLIC"]
)

results = validator.validate()
if not results.success:
    raise ValueError(f"Data quality check failed:\n{results}")

print("All data quality checks passed.")
```

**mapshaper topology-aware simplification:**

```bash
# Simplify admin boundaries while preserving shared edges between polygons
# Without topology-aware simplification, shared boundaries get gaps/overlaps

mapshaper \
  input_boundaries.gpkg \
  -simplify 10% keep-shapes \
  -o output_simplified.gpkg format=gpkg

# With Douglas-Peucker and explicit tolerance
mapshaper \
  input.geojson \
  -simplify dp 500 \    # 500m tolerance
  -clean \              # Remove slivers and overlaps after simplification
  -o output.geojson

# Batch simplify at multiple zoom levels for tile generation
for pct in 100 50 20 5; do
    mapshaper input.gpkg \
      -simplify "${pct}%" keep-shapes \
      -o "output_${pct}pct.gpkg" format=gpkg
done
```

---

## Geospatial Version Control

### Kart (formerly Sno)

Git-like version control for geospatial data. The key difference from vanilla git LFS or DVC: Kart tracks changes at the **feature level** (individual rows), not at the file level. You can see exactly which polygon changed, what its geometry was before, and what it is now.

**Basic workflow:**

```bash
# Initialize a new Kart repository
kart init my-geo-project
cd my-geo-project

# Import a GeoPackage dataset
kart import --all-tables /data/cadastre.gpkg

# Check what's in the repository
kart status
kart log

# Make some changes to the GeoPackage
# (edit features in QGIS, scripts, etc.)

# See what changed at the feature level
kart diff
# Output shows individual features added/modified/deleted

# Stage and commit
kart commit -m "Update parcel boundaries in LGA 12345 - new survey data"

# See the diff between two commits
kart diff HEAD~1..HEAD

# Branch for experimental data updates
kart checkout -b feature/boundary-update-2024
# ... make changes ...
kart commit -m "Proposed boundary adjustment for review"

# Merge after review
kart checkout main
kart merge feature/boundary-update-2024

# Tag a release
kart tag v2024.01 -m "Cadastre release January 2024"
```

**Kart with PostGIS backend:**

```bash
# Connect a Kart working copy to a PostGIS database
kart create-workingcopy "postgresql://user:pass@localhost:5432/cadastre?schema=public"

# Now edits in PostGIS are tracked by Kart
kart status    # Shows which features have been modified in PostGIS
kart diff      # Shows feature-level differences
kart commit -m "Updated road alignments from aerial survey"
```

**CI/CD integration: validate spatial data on every commit:**

```yaml
# .kart/hooks/pre-commit  (make executable: chmod +x)
#!/bin/bash
set -e

echo "Running pre-commit spatial validation..."

# Export changed features to temp GeoPackage
kart export --output /tmp/changed.gpkg --only-changed

if [ -f /tmp/changed.gpkg ]; then
    # Check for invalid geometries
    python3 - <<'PYEOF'
import subprocess, sys
result = subprocess.run([
    "ogrinfo", "-al", "-so", "/tmp/changed.gpkg"
], capture_output=True, text=True)

import sqlite3
conn = sqlite3.connect("/tmp/changed.gpkg")
# Basic validity check via GPKG
cur = conn.execute("SELECT COUNT(*) FROM gpkg_contents")
print(f"Layers to check: {cur.fetchone()[0]}")
# Add your custom validation logic here
conn.close()
print("Validation passed.")
PYEOF
fi

rm -f /tmp/changed.gpkg
echo "Pre-commit validation complete."
```

### GeoGig

Distributed version control for spatial data. Java-based, branching and merging at the feature level, with a REST API for programmatic access. More mature than Kart for complex merge scenarios (conflict resolution tooling is stronger).

```bash
# Initialize and configure
geogig init
geogig config user.name "Data Engineer"
geogig config user.email "geo@company.com"

# Import from PostGIS
geogig pg import \
  --host localhost --database gis \
  --user postgres --password secret \
  --table cadastre.parcels

# Commit
geogig add
geogig commit -m "Initial parcel import"

# Branch and merge
geogig branch -c feature/update-suburbs
# ... make changes ...
geogig add && geogig commit -m "Updated suburb boundaries"
geogig checkout master
geogig merge feature/update-suburbs

# Export back to PostGIS
geogig pg export \
  --host localhost --database gis \
  --user postgres --password secret \
  HEAD:cadastre/parcels cadastre.parcels_v2

# Show feature-level diff
geogig diff HEAD~1 HEAD -- cadastre/parcels
```

### DVC (Data Version Control)

Git for large datasets. Not geo-specific, but widely used for tracking GeoTIFF rasters, GeoParquet files, and ML model weights that are too large for git proper.

```bash
# Initialize in a git repo
git init
dvc init

# Configure remote storage (S3)
dvc remote add -d s3remote s3://my-geo-data-bucket/dvc-store
dvc remote modify s3remote region ap-southeast-2

# Track large files
dvc add data/landcover_2024.tif
dvc add data/parcels_national.parquet
git add data/.gitignore data/landcover_2024.tif.dvc data/parcels_national.parquet.dvc

# Push data to S3
dvc push

# Team member pulls
git pull
dvc pull   # Downloads data from S3 matching current git commit

# Create a DVC pipeline
cat > dvc.yaml << 'EOF'
stages:
  download_imagery:
    cmd: python scripts/download_imagery.py
    deps:
      - scripts/download_imagery.py
    outs:
      - data/raw/imagery/

  compute_ndvi:
    cmd: python scripts/compute_ndvi.py
    deps:
      - scripts/compute_ndvi.py
      - data/raw/imagery/
    outs:
      - data/processed/ndvi/
    metrics:
      - reports/ndvi_stats.json

  generate_tiles:
    cmd: bash scripts/generate_tiles.sh
    deps:
      - data/processed/ndvi/
    outs:
      - data/tiles/ndvi.pmtiles
EOF

dvc repro    # Runs only stages with changed deps
dvc metrics show   # Compare metrics across git commits
```

---

## Data Catalogs & Metadata

### STAC (SpatioTemporal Asset Catalog)

The standard for cataloging raster data assets. A STAC item is a JSON document describing a geospatial asset: its bounding box, acquisition time, cloud cover, available bands, and links to the actual data files (typically COGs on object storage).

**Creating a STAC item from a COG using pystac:**

```python
import pystac
from pystac.extensions.eo import EOExtension, Band
from pystac.extensions.projection import ProjectionExtension
import rasterio
from rasterio.crs import CRS
from datetime import datetime
from shapely.geometry import mapping, box
import pyproj

def create_stac_item_from_cog(
    cog_path: str,
    s3_uri: str,
    collection_id: str = "my-collection",
    cloud_cover: float = 0.0,
) -> pystac.Item:
    """Create a STAC item for a Cloud Optimized GeoTIFF."""

    with rasterio.open(cog_path) as src:
        bounds = src.bounds
        crs = src.crs
        width = src.width
        height = src.height
        transform = src.transform
        band_count = src.count
        nodata = src.nodata
        capture_date = src.tags().get("ACQUISITION_DATE", datetime.utcnow().isoformat())

    # Reproject bounds to WGS84 for the STAC bbox
    if crs != CRS.from_epsg(4326):
        transformer = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        minx, miny = transformer.transform(bounds.left, bounds.bottom)
        maxx, maxy = transformer.transform(bounds.right, bounds.top)
    else:
        minx, miny, maxx, maxy = bounds.left, bounds.bottom, bounds.right, bounds.top

    bbox = [minx, miny, maxx, maxy]
    footprint = mapping(box(*bbox))

    item = pystac.Item(
        id=s3_uri.split("/")[-1].replace(".tif", ""),
        geometry=footprint,
        bbox=bbox,
        datetime=datetime.fromisoformat(capture_date),
        properties={
            "platform": "sentinel-2",
            "eo:cloud_cover": cloud_cover,
            "processing:level": "L2A",
        },
    )

    # Add EO extension
    eo_ext = EOExtension.ext(item, add_if_missing=True)
    eo_ext.cloud_cover = cloud_cover

    # Add projection extension
    proj_ext = ProjectionExtension.ext(item, add_if_missing=True)
    proj_ext.epsg = int(crs.to_epsg()) if crs.to_epsg() else None
    proj_ext.shape = [height, width]
    proj_ext.transform = list(transform)[:6]

    # Add the COG as an asset
    item.add_asset(
        "data",
        pystac.Asset(
            href=s3_uri,
            media_type=pystac.MediaType.COG,
            roles=["data"],
            extra_fields={"eo:bands": [{"name": f"B{i+1}"} for i in range(band_count)]},
        ),
    )

    # Add overview asset for quick previews
    item.add_asset(
        "thumbnail",
        pystac.Asset(
            href=s3_uri.replace(".tif", "_thumbnail.jpg"),
            media_type=pystac.MediaType.JPEG,
            roles=["thumbnail"],
        ),
    )

    return item


# Build a static STAC catalog from multiple COGs
catalog = pystac.Catalog(id="my-catalog", description="My imagery catalog")
collection = pystac.Collection(
    id="sentinel-2-ndvi",
    description="Sentinel-2 NDVI composites",
    extent=pystac.Extent(
        spatial=pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]]),
        temporal=pystac.TemporalExtent(intervals=[[datetime(2024, 1, 1), None]]),
    ),
)
catalog.add_child(collection)

# Add items
for s3_path in [
    "s3://my-bucket/ndvi/2024_01.tif",
    "s3://my-bucket/ndvi/2024_02.tif",
]:
    item = create_stac_item_from_cog("/tmp/local_copy.tif", s3_path)
    collection.add_item(item)

catalog.normalize_hrefs("./stac-output")
catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
```

**stac-fastapi + pgstac: deploy your own STAC API**

```yaml
# docker-compose.yml for stac-fastapi + pgstac
version: "3.9"
services:
  pgstac:
    image: ghcr.io/stac-utils/pgstac:v0.8.6
    environment:
      POSTGRES_USER: pgstac
      POSTGRES_PASSWORD: pgstacpass
      POSTGRES_DB: pgstac
    volumes:
      - pgstac_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U pgstac"]
      interval: 5s
      timeout: 5s
      retries: 10

  stac-api:
    image: ghcr.io/stac-utils/stac-fastapi-pgstac:latest
    environment:
      APP_HOST: "0.0.0.0"
      APP_PORT: "8080"
      POSTGRES_USER: pgstac
      POSTGRES_PASS: pgstacpass
      POSTGRES_DBNAME: pgstac
      POSTGRES_HOST: pgstac
      POSTGRES_PORT: "5432"
    ports:
      - "8080:8080"
    depends_on:
      pgstac:
        condition: service_healthy

  titiler:
    image: ghcr.io/developmentseed/titiler:latest
    environment:
      PORT: "8000"
      WORKERS_PER_CORE: "1"
    ports:
      - "8000:8000"

volumes:
  pgstac_data:
```

### GeoNetwork

Traditional OGC CSW metadata catalog. ISO 19115/19139 compliant. Required for INSPIRE compliance in the EU. Usually the right tool when your organization has a legal requirement for metadata publication rather than a performance requirement.

### CKAN + ckanext-spatial

Open data portal framework powering data.gov and hundreds of government portals. The `ckanext-spatial` plugin adds map-based search and format-specific preview for geospatial datasets.

```bash
# Deploy CKAN with spatial extension via Docker
docker run -d --name ckan \
  -e CKAN_SQLALCHEMY_URL="postgresql://ckan:ckan@db/ckan" \
  -e CKAN_SOLR_URL="http://solr:8983/solr/ckan" \
  -e CKAN__PLUGINS="stats text_view recline_view spatial_metadata spatial_query" \
  -p 5000:5000 \
  ckan/ckan-base:2.10
```

---

## Docker & Containerization

### Essential Docker Images

| Image | Purpose | Notes |
|---|---|---|
| `postgis/postgis:16-3.4` | PostGIS database | Official, well-maintained |
| `kartoza/geoserver:2.25.0` | GeoServer with plugins | Includes GDAL, ImageMosaic, CSS styles |
| `ghcr.io/maplibre/martin` | Martin vector tile server | Blazing fast Rust-based |
| `ghcr.io/developmentseed/titiler` | TiTiler raster tiles | COG/STAC/MosaicJSON serving |
| `mundialis/esa-snap:latest` | ESA SNAP SAR/optical processing | Large image (~8GB) |
| `osrm/osrm-backend` | OSRM routing engine | Requires pre-processed OSM data |
| `mediagis/nominatim:4.4` | Nominatim geocoder | Requires OSM planet import |
| `ghcr.io/osgeo/gdal:ubuntu-full-3.9.0` | GDAL with all drivers | Use for batch processing jobs |
| `pelias/docker` | Pelias geocoder stack | Multi-container, needs separate setup |
| `valhalla/valhalla` | Valhalla routing + isochrones | Supports multiple travel modes |
| `ghcr.io/stac-utils/stac-fastapi-pgstac` | STAC API server | Needs pgstac DB |
| `pgrouting/pgrouting:15-3.6-3.5` | pgRouting network analysis | PostGIS + pgRouting |

### Docker Compose Recipes

**Recipe 1: Spatial Data Stack (PostGIS + GeoServer + Martin + Nginx)**

```yaml
# docker-compose.spatial-stack.yml
version: "3.9"

services:
  postgis:
    image: postgis/postgis:16-3.4
    container_name: postgis
    restart: unless-stopped
    environment:
      POSTGRES_DB: geoapp
      POSTGRES_USER: geouser
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      PGDATA: /var/lib/postgresql/data/pgdata
    volumes:
      - postgis_data:/var/lib/postgresql/data
      - ./init-scripts:/docker-entrypoint-initdb.d:ro
    ports:
      - "127.0.0.1:5432:5432"
    shm_size: 256mb
    command: >
      postgres
        -c max_connections=200
        -c shared_buffers=512MB
        -c effective_cache_size=1536MB
        -c maintenance_work_mem=128MB
        -c checkpoint_completion_target=0.9
        -c wal_buffers=16MB
        -c default_statistics_target=100
        -c random_page_cost=1.1
        -c effective_io_concurrency=200
        -c work_mem=2621kB
        -c min_wal_size=1GB
        -c max_wal_size=4GB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U geouser -d geoapp"]
      interval: 10s
      timeout: 5s
      retries: 5

  geoserver:
    image: kartoza/geoserver:2.25.0
    container_name: geoserver
    restart: unless-stopped
    environment:
      GEOSERVER_DATA_DIR: /opt/geoserver/data_dir
      GEOWEBCACHE_CACHE_DIR: /opt/geoserver/data_dir/gwc
      GEOSERVER_ADMIN_PASSWORD: ${GEOSERVER_PASSWORD}
      GEOSERVER_ADMIN_USER: admin
      INITIAL_MEMORY: 1G
      MAXIMUM_MEMORY: 4G
      STABLE_EXTENSIONS: "importer,css,inspire,wps,netcdf"
      COMMUNITY_EXTENSIONS: "cog,ogcapi-features"
      POSTGRES_JNDI_ENABLED: "TRUE"
      HOST: postgis
      POSTGRES_PORT: "5432"
      POSTGRES_DB: geoapp
      POSTGRES_USER: geouser
      POSTGRES_PASS: ${POSTGRES_PASSWORD}
    volumes:
      - geoserver_data:/opt/geoserver/data_dir
      - ./data:/data:ro
    depends_on:
      postgis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080/geoserver/web/ || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 10

  martin:
    image: ghcr.io/maplibre/martin:latest
    container_name: martin
    restart: unless-stopped
    environment:
      DATABASE_URL: "postgresql://geouser:${POSTGRES_PASSWORD}@postgis/geoapp"
    command: >
      --listen-addresses 0.0.0.0:3000
      --pool-size 20
    depends_on:
      postgis:
        condition: service_healthy

  nginx:
    image: nginx:alpine
    container_name: nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - ./certbot/www:/var/www/certbot:ro
      - ./certbot/conf:/etc/letsencrypt:ro
      - ./static:/var/www/html:ro
    depends_on:
      - geoserver
      - martin

volumes:
  postgis_data:
  geoserver_data:
```

**Nginx config for the above stack:**

```nginx
# nginx/conf.d/geoapp.conf
upstream geoserver {
    server geoserver:8080;
    keepalive 32;
}

upstream martin {
    server martin:3000;
    keepalive 32;
}

server {
    listen 80;
    server_name geo.example.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name geo.example.com;

    ssl_certificate /etc/letsencrypt/live/geo.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/geo.example.com/privkey.pem;

    # GeoServer OGC services
    location /geoserver/ {
        proxy_pass http://geoserver/geoserver/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
        client_max_body_size 100M;

        # Cache WMS GetMap responses
        location ~ /geoserver/.*/(wms|ows)\?.*SERVICE=WMS.*REQUEST=GetMap {
            proxy_pass http://geoserver;
            proxy_cache geo_cache;
            proxy_cache_valid 200 1h;
            proxy_cache_key "$scheme$request_method$host$request_uri";
            add_header X-Cache-Status $upstream_cache_status;
        }
    }

    # Martin vector tiles
    location /tiles/ {
        proxy_pass http://martin/;
        proxy_set_header Host $host;
        add_header Cache-Control "public, max-age=3600";
        add_header Access-Control-Allow-Origin "*";
        gzip on;
        gzip_types application/x-protobuf;
    }
}
```

**Recipe 2: Processing Stack (GDAL + Python + JupyterLab)**

```yaml
# docker-compose.processing.yml
version: "3.9"

services:
  jupyter:
    build:
      context: .
      dockerfile: Dockerfile.geo
    container_name: geo-jupyter
    ports:
      - "8888:8888"
    environment:
      JUPYTER_ENABLE_LAB: "yes"
      JUPYTER_TOKEN: ${JUPYTER_TOKEN}
    volumes:
      - ./notebooks:/home/jovyan/work
      - ./data:/data
      - ./scripts:/scripts:ro
    command: >
      start-notebook.sh
        --NotebookApp.token=${JUPYTER_TOKEN}
        --NotebookApp.allow_origin='*'
```

```dockerfile
# Dockerfile.geo
FROM ghcr.io/osgeo/gdal:ubuntu-full-3.9.0

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-gdal \
    jupyter \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    jupyterlab \
    geopandas \
    rasterio \
    xarray \
    rioxarray \
    dask \
    dask-geopandas \
    pyarrow \
    duckdb \
    pystac-client \
    stackstac \
    leafmap \
    lonboard \
    shapely \
    pyproj \
    fiona \
    h3 \
    rio-cogeo \
    gdal2tiles \
    matplotlib \
    tqdm \
    sqlalchemy \
    psycopg2-binary

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

**Recipe 3: Routing Stack (OSRM + Valhalla + Nominatim)**

```yaml
# docker-compose.routing.yml
version: "3.9"

services:
  osrm:
    image: osrm/osrm-backend:latest
    container_name: osrm
    restart: unless-stopped
    volumes:
      - ./osrm-data:/data
    # After preprocessing OSM data:
    command: osrm-routed --algorithm mld /data/australia-latest.osrm
    ports:
      - "5000:5000"

  valhalla:
    image: ghcr.io/gis-ops/docker-valhalla/valhalla:latest
    container_name: valhalla
    restart: unless-stopped
    environment:
      tile_urls: "https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf"
      serve_tiles: "True"
      build_time_zones: "True"
      build_elevation: "True"
    volumes:
      - ./valhalla-data:/custom_files
    ports:
      - "8002:8002"

  nominatim:
    image: mediagis/nominatim:4.4
    container_name: nominatim
    restart: unless-stopped
    environment:
      PBF_URL: "https://download.geofabrik.de/australia-oceania/australia-latest.osm.pbf"
      REPLICATION_URL: "https://download.geofabrik.de/australia-oceania-updates/"
      IMPORT_WIKIPEDIA: "false"
      IMPORT_TIGER_ADDRESSES: "false"
      NOMINATIM_PASSWORD: ${NOMINATIM_PASSWORD}
      THREADS: "8"
    volumes:
      - nominatim_data:/var/lib/postgresql/14/main
    ports:
      - "8080:8080"
    shm_size: 1gb

volumes:
  nominatim_data:
```

**OSRM preprocessing script:**

```bash
#!/bin/bash
# preprocess_osrm.sh - Preprocess OSM data for OSRM routing
OSM_FILE="australia-latest.osm.pbf"
PROFILE="car"  # or "foot", "bicycle"

# Download OSM data
wget -P ./osrm-data \
  "https://download.geofabrik.de/australia-oceania/${OSM_FILE}"

# Extract, partition, and customize
docker run --rm -v "$(pwd)/osrm-data:/data" \
  osrm/osrm-backend osrm-extract \
  -p "/opt/car.lua" "/data/${OSM_FILE}"

docker run --rm -v "$(pwd)/osrm-data:/data" \
  osrm/osrm-backend osrm-partition "/data/${OSM_FILE%.osm.pbf}.osrm"

docker run --rm -v "$(pwd)/osrm-data:/data" \
  osrm/osrm-backend osrm-customize "/data/${OSM_FILE%.osm.pbf}.osrm"

echo "OSRM preprocessing complete. Start with docker compose up osrm"
```

---

## CI/CD for Geospatial

### GitHub Actions

**Workflow 1: Spatial data validation on PR**

```yaml
# .github/workflows/validate-geodata.yml
name: Validate Geospatial Data

on:
  pull_request:
    paths:
      - 'data/**/*.gpkg'
      - 'data/**/*.geojson'
      - 'data/**/*.parquet'

jobs:
  validate-geometry:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/osgeo/gdal:ubuntu-full-3.9.0

    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Install Python dependencies
        run: |
          pip install geopandas shapely pyarrow great-expectations

      - name: Detect changed geo files
        id: changed_files
        uses: tj-actions/changed-files@v42
        with:
          files: |
            data/**/*.gpkg
            data/**/*.geojson
            data/**/*.parquet

      - name: Validate geometries
        if: steps.changed_files.outputs.any_changed == 'true'
        run: |
          python3 - << 'PYEOF'
          import geopandas as gpd
          import sys
          import os

          files = """${{ steps.changed_files.outputs.all_changed_files }}""".split()
          errors = []

          for f in files:
              if not os.path.exists(f):
                  continue
              print(f"Validating: {f}")
              try:
                  gdf = gpd.read_file(f)
              except Exception as e:
                  errors.append(f"{f}: Failed to read - {e}")
                  continue

              # Check geometry validity
              invalid = gdf[~gdf.geometry.is_valid]
              if len(invalid) > 0:
                  errors.append(f"{f}: {len(invalid)} invalid geometries found")
                  for _, row in invalid.head(5).iterrows():
                      from shapely.validation import explain_validity
                      reason = explain_validity(row.geometry)
                      errors.append(f"  Row {row.name}: {reason}")

              # Check CRS is defined
              if gdf.crs is None:
                  errors.append(f"{f}: No CRS defined")

              # Check for null geometries
              null_geoms = gdf[gdf.geometry.isna()].shape[0]
              if null_geoms > 0:
                  errors.append(f"{f}: {null_geoms} null geometries found")

              print(f"  OK: {len(gdf)} features, CRS: {gdf.crs}")

          if errors:
              print("\n=== VALIDATION ERRORS ===")
              for e in errors:
                  print(f"ERROR: {e}")
              sys.exit(1)
          else:
              print("\n=== All validations passed ===")
          PYEOF

      - name: Check schema consistency
        if: steps.changed_files.outputs.any_changed == 'true'
        run: |
          python3 scripts/check_schema.py \
            --files "${{ steps.changed_files.outputs.all_changed_files }}" \
            --schema-dir schemas/
```

**Workflow 2: Automated tile generation on data push**

```yaml
# .github/workflows/generate-tiles.yml
name: Generate Vector Tiles

on:
  push:
    branches: [main]
    paths:
      - 'data/boundaries/**'

env:
  S3_BUCKET: my-geo-tiles-bucket
  AWS_REGION: ap-southeast-2

jobs:
  generate-tiles:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4
        with:
          lfs: true

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Install tippecanoe
        run: |
          sudo apt-get install -y build-essential libsqlite3-dev zlib1g-dev
          git clone https://github.com/felt/tippecanoe.git
          cd tippecanoe && make -j$(nproc) && sudo make install

      - name: Install GDAL
        run: |
          sudo add-apt-repository ppa:ubuntugis/ppa
          sudo apt-get update && sudo apt-get install -y gdal-bin python3-gdal

      - name: Convert to GeoJSON for tippecanoe
        run: |
          mkdir -p /tmp/geojson
          for f in data/boundaries/*.gpkg; do
            name=$(basename "$f" .gpkg)
            ogr2ogr -f GeoJSON -t_srs EPSG:4326 \
              /tmp/geojson/${name}.geojson "$f"
          done

      - name: Generate PMTiles
        run: |
          mkdir -p /tmp/tiles
          tippecanoe \
            --output=/tmp/tiles/boundaries.pmtiles \
            --force \
            --maximum-zoom=14 \
            --minimum-zoom=0 \
            --drop-densest-as-needed \
            --extend-zooms-if-still-dropping \
            --read-parallel \
            --no-tile-size-limit \
            /tmp/geojson/*.geojson

      - name: Upload to S3
        run: |
          aws s3 cp /tmp/tiles/boundaries.pmtiles \
            s3://${{ env.S3_BUCKET }}/tiles/boundaries.pmtiles \
            --content-type application/octet-stream \
            --cache-control "public, max-age=3600"

      - name: Invalidate CloudFront cache
        run: |
          aws cloudfront create-invalidation \
            --distribution-id ${{ secrets.CLOUDFRONT_DISTRIBUTION_ID }} \
            --paths "/tiles/*"
```

**Workflow 3: COG creation and STAC catalog update**

```yaml
# .github/workflows/process-rasters.yml
name: Process Rasters to COG and Update STAC

on:
  workflow_dispatch:
    inputs:
      source_path:
        description: S3 path to source rasters
        required: true

jobs:
  process-rasters:
    runs-on: ubuntu-latest
    container:
      image: ghcr.io/osgeo/gdal:ubuntu-full-3.9.0

    steps:
      - uses: actions/checkout@v4

      - name: Install Python deps
        run: pip install rio-cogeo pystac boto3 rasterio tqdm

      - name: Configure AWS
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: ap-southeast-2
        run: |
          python3 scripts/process_to_cog_and_stac.py \
            --source "${{ github.event.inputs.source_path }}" \
            --output "s3://my-cog-bucket/processed/" \
            --stac-catalog "s3://my-cog-bucket/stac/"
```

### Testing Spatial Code

```python
# tests/test_spatial_operations.py
import pytest
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.validation import make_valid
import numpy as np

# ---- Fixtures ----

@pytest.fixture
def sample_points():
    """A small GeoDataFrame of test points in WGS84."""
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3], "value": [10.0, 20.0, 30.0]},
        geometry=[Point(151.2, -33.8), Point(144.9, -37.8), Point(153.0, -27.5)],
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_polygons():
    """Simple test polygons."""
    return gpd.GeoDataFrame(
        {"zone": ["A", "B"]},
        geometry=[
            Polygon([(150, -34), (152, -34), (152, -33), (150, -33), (150, -34)]),
            Polygon([(144, -38), (146, -38), (146, -37), (144, -37), (144, -38)]),
        ],
        crs="EPSG:4326",
    )


# ---- Unit tests ----

def test_point_in_polygon(sample_points, sample_polygons):
    """Points should correctly join to containing polygons."""
    joined = gpd.sjoin(sample_points, sample_polygons, how="left", predicate="within")
    # Point at Sydney (151.2, -33.8) should be in zone A
    sydney = joined[joined["id"] == 1]
    assert sydney["zone"].values[0] == "A"
    # Point at Adelaide (138.6, -34.9) is outside both polygons
    adelaide = joined[joined["id"] == 3]
    assert pd.isna(adelaide["zone"].values[0])


def test_reprojection_preserves_topology(sample_polygons):
    """Reprojecting should not create invalid geometries."""
    reprojected = sample_polygons.to_crs(epsg=3857)
    assert reprojected.geometry.is_valid.all(), "Reprojected geometries should be valid"
    assert reprojected.crs.to_epsg() == 3857


def test_buffer_area_is_positive(sample_points):
    """Buffered points should have positive area."""
    buffered = sample_points.to_crs(epsg=3857).buffer(1000)  # 1km buffer
    assert (buffered.area > 0).all()
    assert (buffered.area > 3_000_000).all()  # Circle of radius 1000m ≈ 3.14M m²


def test_invalid_geometry_is_fixable():
    """ST_MakeValid equivalent: shapely.make_valid should fix self-intersecting ring."""
    # Bowtie polygon (self-intersecting)
    bowtie = Polygon([(0, 0), (2, 2), (2, 0), (0, 2), (0, 0)])
    assert not bowtie.is_valid
    fixed = make_valid(bowtie)
    assert fixed.is_valid
    assert fixed.area > 0


def test_no_null_geometries(sample_points):
    """Dataset should have no null geometries."""
    assert not sample_points.geometry.isna().any()


def test_crs_is_defined(sample_points):
    """All GeoDataFrames should have a defined CRS."""
    assert sample_points.crs is not None
    assert sample_points.crs.to_epsg() == 4326


# ---- Property-based tests with hypothesis ----

# pip install hypothesis hypothesis-geopandas
from hypothesis import given, settings
import hypothesis.strategies as st

@given(
    lon=st.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False),
    lat=st.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False),
    buffer_m=st.floats(min_value=1, max_value=100_000, allow_nan=False),
)
@settings(max_examples=100)
def test_buffer_always_valid(lon, lat, buffer_m):
    """Buffering any valid WGS84 point in Mercator should produce a valid polygon."""
    point = gpd.GeoDataFrame(
        {"x": [1]}, geometry=[Point(lon, lat)], crs="EPSG:4326"
    )
    buffered = point.to_crs(epsg=3857).buffer(buffer_m)
    assert buffered.is_valid.all(), f"Buffer of ({lon}, {lat}) by {buffer_m}m is invalid"
```

---

## Cloud Infrastructure Patterns

### Architecture Comparison

**AWS: S3 + CloudFront + Lambda + RDS PostGIS**

```
COG/PMTiles on S3
      |
CloudFront CDN (global edge caching)
      |
Lambda@Edge (TiTiler for dynamic raster tiling, or direct S3 access)
      |
RDS PostGIS (vector data, spatial queries)
      |
EC2 or ECS (GeoServer, processing workers)
```

```terraform
# terraform/aws-geo-stack/main.tf (abbreviated)
resource "aws_s3_bucket" "tiles" {
  bucket = "my-geo-tiles-${var.environment}"
}

resource "aws_s3_bucket_policy" "tiles_public" {
  bucket = aws_s3_bucket.tiles.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = "*"
      Action    = "s3:GetObject"
      Resource  = "${aws_s3_bucket.tiles.arn}/*"
    }]
  })
}

resource "aws_cloudfront_distribution" "tiles_cdn" {
  origin {
    domain_name = aws_s3_bucket.tiles.bucket_regional_domain_name
    origin_id   = "tiles-s3"
    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.tiles.cloudfront_access_identity_path
    }
  }

  default_cache_behavior {
    target_origin_id       = "tiles-s3"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS"]
    cached_methods         = ["GET", "HEAD"]
    compress               = true

    forwarded_values {
      query_string = false
      cookies { forward = "none" }
      headers = ["Origin", "Access-Control-Request-Headers", "Access-Control-Request-Method"]
    }

    min_ttl     = 0
    default_ttl = 3600
    max_ttl     = 86400
  }

  enabled = true
  comment = "Geospatial tiles CDN"
  price_class = "PriceClass_All"

  restrictions {
    geo_restriction { restriction_type = "none" }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_db_instance" "postgis" {
  identifier              = "postgis-${var.environment}"
  engine                  = "postgres"
  engine_version          = "16.1"
  instance_class          = "db.r6g.xlarge"
  allocated_storage       = 500
  max_allocated_storage   = 2000
  storage_type            = "gp3"
  storage_encrypted       = true
  db_name                 = "geoapp"
  username                = var.db_username
  password                = var.db_password
  parameter_group_name    = aws_db_parameter_group.postgis.name
  vpc_security_group_ids  = [aws_security_group.postgis.id]
  db_subnet_group_name    = aws_db_subnet_group.postgis.name
  backup_retention_period = 30
  deletion_protection     = true
  skip_final_snapshot     = false
  final_snapshot_identifier = "postgis-final-${var.environment}"

  tags = { Name = "PostGIS ${var.environment}" }
}
```

**GCP: Cloud Storage + Cloud Run + Cloud SQL + BigQuery GIS**

```yaml
# cloud-run TiTiler service
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: titiler
  namespace: geo-platform
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/minScale: "1"
        autoscaling.knative.dev/maxScale: "20"
        run.googleapis.com/memory: 2Gi
        run.googleapis.com/cpu: 2
    spec:
      containers:
        - image: ghcr.io/developmentseed/titiler:latest
          env:
            - name: GDAL_CACHEMAX
              value: "512"
            - name: GDAL_NUM_THREADS
              value: "ALL_CPUS"
            - name: VSI_CACHE_SIZE
              value: "536870912"
            - name: CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE
              value: "YES"
```

**BigQuery GIS for analytical queries over massive datasets:**

```sql
-- BigQuery GIS: Find all addresses within 500m of a school
-- (running over 10M+ addresses without provisioning anything)
SELECT
  a.address_id,
  a.street_address,
  a.latitude,
  a.longitude,
  s.school_name,
  ST_DISTANCE(
    ST_GEOGPOINT(a.longitude, a.latitude),
    ST_GEOGPOINT(s.longitude, s.latitude)
  ) as distance_m
FROM
  `project.dataset.addresses` a
CROSS JOIN
  `project.dataset.schools` s
WHERE
  ST_DWITHIN(
    ST_GEOGPOINT(a.longitude, a.latitude),
    ST_GEOGPOINT(s.longitude, s.latitude),
    500  -- 500 meters
  )
ORDER BY
  a.address_id, distance_m
```

### Cost Comparison Table

| Pattern | Monthly Cost (est. ~100GB tiles, 1M req/day) |
|---|---|
| AWS S3 + CloudFront | ~$50-100/mo |
| GCP Cloud Storage + CDN | ~$40-90/mo |
| Azure Blob + CDN | ~$45-95/mo |
| Self-hosted VPS (2 vCPU, 4GB) | ~$20/mo + bandwidth |
| Cloudflare R2 + Pages | ~$15/mo (no egress fees) |

Cloudflare R2 is worth serious consideration for high-traffic tile serving: zero egress fees, global CDN included, S3-compatible API.

---

## Performance & Optimization

### Spatial Indexing in PostGIS

| Index Type | Use Case | When to Choose |
|---|---|---|
| GiST | General spatial queries, all geometry types | Default choice for 99% of cases |
| SP-GiST | Point-heavy datasets, quadtree structure | Pure point data with uniform distribution |
| BRIN | Geographically sorted data (e.g., scans of a study area) | Massive append-only tables where data is inserted in spatial order |
| GIST (3D) | ST_3DIntersects, 3D geometry queries | When you actually need 3D spatial operations |

```sql
-- Create a standard GiST index
CREATE INDEX idx_parcels_geom ON cadastre.parcels USING GIST (geom);

-- Create an SP-GiST index for a pure points table
CREATE INDEX idx_observations_geom ON monitoring.observations USING SPGIST (geom);

-- BRIN for a massive time-ordered spatial table (much smaller index)
CREATE INDEX idx_vessel_track_geom
    ON ais.vessel_tracks USING BRIN (geom)
    WITH (pages_per_range = 128);

-- After creating indexes, update statistics
ANALYZE cadastre.parcels;
```

### Table Partitioning

```sql
-- Partition a massive observation table by year + spatial region
CREATE TABLE sensor.observations (
    observation_id  BIGSERIAL,
    sensor_id       INTEGER NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL,
    value           DOUBLE PRECISION,
    geom            GEOMETRY(Point, 4326) NOT NULL
) PARTITION BY RANGE (observed_at);

-- Create yearly partitions
CREATE TABLE sensor.observations_2023
    PARTITION OF sensor.observations
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');

CREATE TABLE sensor.observations_2024
    PARTITION OF sensor.observations
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');

-- Create spatial indexes on each partition
CREATE INDEX ON sensor.observations_2023 USING GIST (geom);
CREATE INDEX ON sensor.observations_2024 USING GIST (geom);

-- Enable partition pruning (should be on by default in PG 12+)
SET enable_partition_pruning = on;

-- Query automatically scans only the relevant partition
SELECT * FROM sensor.observations
WHERE observed_at >= '2024-06-01'
AND ST_Within(geom, ST_MakeEnvelope(144, -38, 146, -37, 4326));
```

### Materialized Views + pg_cron

```sql
-- Install pg_cron extension
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- Create a materialized view for spatial aggregation
CREATE MATERIALIZED VIEW analytics.parcel_lga_summary AS
SELECT
    l.lga_code,
    l.lga_name,
    COUNT(p.gid) AS parcel_count,
    SUM(ST_Area(p.geom::geography)) / 1e6 AS total_area_km2,
    AVG(p.land_value) AS avg_land_value,
    l.geom AS lga_geom
FROM admin.lgas l
LEFT JOIN cadastre.parcels p ON ST_Within(p.geom, l.geom)
GROUP BY l.lga_code, l.lga_name, l.geom
WITH DATA;

-- Create indexes on the materialized view
CREATE INDEX idx_parcel_lga_summary_geom
    ON analytics.parcel_lga_summary USING GIST (lga_geom);
CREATE UNIQUE INDEX idx_parcel_lga_summary_code
    ON analytics.parcel_lga_summary (lga_code);

-- Schedule automated refresh every night at 2am
SELECT cron.schedule(
    'refresh-parcel-lga-summary',
    '0 2 * * *',
    'REFRESH MATERIALIZED VIEW CONCURRENTLY analytics.parcel_lga_summary'
);

-- Check scheduled jobs
SELECT * FROM cron.job;
```

### PgBouncer Connection Pooling

```ini
# pgbouncer.ini
[databases]
geoapp = host=postgis port=5432 dbname=geoapp

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt

# Transaction pooling is usually right for web/tile services
pool_mode = transaction

# GeoServer might need session pooling (it uses set_config, etc.)
; pool_mode = session

max_client_conn = 500
default_pool_size = 25
min_pool_size = 5
reserve_pool_size = 5
reserve_pool_timeout = 3
max_db_connections = 50

# PostGIS with long-running spatial queries
query_timeout = 300
client_idle_timeout = 60
server_lifetime = 3600
server_idle_timeout = 600

log_connections = 0
log_disconnections = 0
log_pooler_errors = 1
stats_period = 60
```

### Physical Clustering for Spatial Scans

```sql
-- CLUSTER rewrites the table in spatial index order.
-- Subsequent spatial range scans read sequential blocks instead of jumping around the heap.
-- On a table with millions of rows, this can be a 5-10x speedup on spatial queries.

-- First, ensure the GiST index exists
CREATE INDEX IF NOT EXISTS idx_parcels_geom ON cadastre.parcels USING GIST (geom);

-- Cluster (requires AccessExclusive lock — run during maintenance window)
CLUSTER cadastre.parcels USING idx_parcels_geom;

-- Set up auto-cluster with pg_cron (because CLUSTER is not maintained on INSERT/UPDATE)
-- Alternatively, run CLUSTER during nightly maintenance
SELECT cron.schedule(
    'cluster-parcels-weekly',
    '0 3 * * 0',  -- Every Sunday at 3am
    'CLUSTER cadastre.parcels USING idx_parcels_geom; ANALYZE cadastre.parcels;'
);
```

### ST_Subdivide for Complex Polygons

```sql
-- Complex polygons with many vertices are slow in spatial joins.
-- ST_Subdivide breaks them into simpler pieces while preserving spatial coverage.

-- Check complexity of your polygons
SELECT
    gid,
    ST_NPoints(geom) as vertex_count,
    GeometryType(geom) as geom_type
FROM cadastre.complex_zones
ORDER BY vertex_count DESC
LIMIT 10;

-- Create a subdivided copy for use in spatial joins
CREATE TABLE cadastre.complex_zones_subdivided AS
SELECT
    gid,
    zone_code,
    zone_name,
    (ST_Dump(ST_Subdivide(geom, 256))).geom AS geom
    -- 256 = max vertices per sub-polygon (tuning parameter)
FROM cadastre.complex_zones;

CREATE INDEX ON cadastre.complex_zones_subdivided USING GIST (geom);

-- Now use the subdivided table in spatial joins
-- The query planner can prune far more candidates with simpler polygons
EXPLAIN ANALYZE
SELECT DISTINCT ON (p.gid)
    p.gid,
    p.parcel_id,
    z.zone_code
FROM cadastre.parcels p
JOIN cadastre.complex_zones_subdivided z
    ON ST_Intersects(p.geom, z.geom);
```

### GDAL Parallelism

```bash
# GDAL_NUM_THREADS: controls threading in GDAL operations
export GDAL_NUM_THREADS=ALL_CPUS

# GDAL_CACHEMAX: increase block cache (default is only 5% of RAM!)
export GDAL_CACHEMAX=2048   # 2GB cache in MB

# For warping/resampling (gdal_warp)
gdalwarp \
  -wo NUM_THREADS=ALL_CPUS \
  -co COMPRESS=DEFLATE \
  -co TILED=YES \
  -co BIGTIFF=IF_SAFER \
  --config GDAL_NUM_THREADS ALL_CPUS \
  --config GDAL_CACHEMAX 2048 \
  input.tif output_warped.tif

# For COG creation with multithreading
gdal_translate \
  -of COG \
  -co COMPRESS=DEFLATE \
  -co NUM_THREADS=ALL_CPUS \
  --config GDAL_NUM_THREADS ALL_CPUS \
  --config GDAL_CACHEMAX 2048 \
  large_mosaic.tif output_cog.tif
```

---

## Advanced Dark Arts

### ogr2ogr as a Lightweight Spatial ETL Engine

`ogr2ogr -sql` runs SQL (via SQLite dialect) directly on source files. No database needed. This is one of the most underused tricks in the geospatial toolbox.

```bash
# Spatial query on a file — no database, no Python, just ogr2ogr
ogr2ogr \
  -f GeoJSON \
  output_filtered.geojson \
  input.gpkg \
  -sql "SELECT parcel_id, land_use, area_ha, geometry
        FROM parcels
        WHERE area_ha > 100
        AND land_use = 'RURAL'
        ORDER BY area_ha DESC
        LIMIT 1000"

# Spatial clip using -clipsrc (no database needed)
ogr2ogr \
  -f GPKG \
  output_clipped.gpkg \
  input_national.gpkg \
  -clipsrc 144.0 -38.5 146.0 -37.0  # Clip to bounding box

# Clip to polygon from another file
ogr2ogr \
  -f GPKG \
  output_clipped.gpkg \
  input_national.gpkg \
  -clipsrc study_area.gpkg

# Column remapping during import
ogr2ogr \
  -f PostgreSQL \
  "PG:host=localhost dbname=gis" \
  input.gpkg \
  -nln target_table \
  -sql "SELECT old_parcel_id AS parcel_id,
               old_land_use  AS land_use_code,
               old_area      AS area_hectares,
               geometry
        FROM input_layer"
```

### GDAL VRT Files: Virtual Views Over Remote Data

VRT (Virtual Dataset) files let you define views over remote data that don't require downloading. You can composite bands from separate S3 files, apply no-data masks, define virtual raster mosaics — all without a single byte hitting your disk.

```xml
<!-- mosaic.vrt: Mosaic multiple COGs from S3 into a single virtual raster -->
<VRTDataset rasterXSize="20000" rasterYSize="20000">
  <SRS dataAxisToSRSAxisMapping="1,2">EPSG:32754</SRS>
  <GeoTransform>400000, 1.0, 0, 6600000, 0, -1.0</GeoTransform>

  <VRTRasterBand dataType="Float32" band="1">
    <Description>NDVI</Description>
    <NoDataValue>-9999</NoDataValue>

    <SimpleSource>
      <SourceFilename>/vsicurl/https://my-bucket.s3.amazonaws.com/ndvi/tile_001.tif</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>

    <SimpleSource>
      <SourceFilename>/vsicurl/https://my-bucket.s3.amazonaws.com/ndvi/tile_002.tif</SourceFilename>
      <SourceBand>1</SourceBand>
    </SimpleSource>
  </VRTRasterBand>
</VRTDataset>
```

```bash
# Build a VRT mosaic from local files
gdalbuildvrt mosaic.vrt /data/tiles/*.tif

# Build from S3 (streaming, no download)
gdalbuildvrt \
  -input_file_list s3_tile_list.txt \  # Text file with /vsicurl/ paths
  s3_mosaic.vrt

# Multi-band VRT: Combine separate band files into one virtual raster
gdalbuildvrt \
  -separate \
  rgb_composite.vrt \
  red_band.tif green_band.tif blue_band.tif

# Process the VRT as if it were a real raster
gdalinfo s3_mosaic.vrt   # No download — reads metadata only
gdal_translate -of COG s3_mosaic.vrt local_mosaic_cog.tif  # Downloads and processes
```

### Streaming Processing: /vsicurl/ → /vsimem/

Process remote files without ever writing to disk:

```python
import rasterio
from rasterio.io import MemoryFile
import requests

# Stream a COG from S3, compute statistics, never touch disk
with rasterio.open("/vsicurl/https://my-bucket.s3.amazonaws.com/dem.tif") as src:
    # This reads only the metadata and overview tiles (COG structure)
    print(f"CRS: {src.crs}, Size: {src.width}x{src.height}")
    print(f"Bounds: {src.bounds}")

    # Read just a spatial window (COG only fetches needed HTTP ranges)
    window = rasterio.windows.from_bounds(
        left=144.0, bottom=-38.0, right=145.0, top=-37.0,
        transform=src.transform
    )
    data = src.read(1, window=window)

# Using /vsimem/ for in-memory processing in gdal commands
import subprocess

# Download to memory, process, save result — no temp files
proc = subprocess.run([
    "gdal_translate",
    "-of", "MEM",
    "/vsicurl/https://example.com/large_dem.tif",
    "/vsimem/temp.tif"  # Virtual memory filesystem
], capture_output=True)

# Chain /vsimem/ operations for pipeline processing
subprocess.run([
    "gdalwarp",
    "-t_srs", "EPSG:3857",
    "/vsimem/temp.tif",
    "/vsimem/reprojected.tif"
])
```

### PostGIS LISTEN/NOTIFY for Real-time Spatial Event Streaming

```python
# realtime_spatial_events.py
# Pattern: PostgreSQL LISTEN/NOTIFY → Redis → WebSocket clients
# Zero external message brokers needed for moderate throughput

import psycopg2
import select
import json
import redis
from datetime import datetime

def monitor_spatial_events():
    """
    Listen for spatial INSERT events in PostGIS.
    Fires when a new GPS observation is inserted.
    """
    conn = psycopg2.connect("postgresql://user:pass@localhost:5432/geoapp")
    conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute("LISTEN spatial_events;")

    redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

    print("Listening for spatial events...")
    while True:
        # Wait up to 5 seconds for a notification
        if select.select([conn], [], [], 5) == ([], [], []):
            continue  # Timeout, loop again

        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            payload = json.loads(notify.payload)

            # Enrich with reverse geocode if needed
            event = {
                "timestamp": datetime.utcnow().isoformat(),
                "channel": notify.channel,
                "feature_id": payload.get("id"),
                "lat": payload.get("lat"),
                "lon": payload.get("lon"),
                "event_type": payload.get("type"),
            }

            # Publish to Redis for WebSocket broadcasting
            redis_client.publish("geo:events", json.dumps(event))
            redis_client.xadd(  # Also write to Redis Stream for persistence
                "geo:event-stream",
                event,
                maxlen=100_000,
            )
            print(f"Event: {event['feature_id']} at ({event['lat']}, {event['lon']})")

# SQL trigger to fire the notifications
TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION notify_spatial_event()
RETURNS TRIGGER AS $$
BEGIN
    PERFORM pg_notify(
        'spatial_events',
        json_build_object(
            'id',   NEW.observation_id,
            'lat',  ST_Y(NEW.geom),
            'lon',  ST_X(NEW.geom),
            'type', NEW.event_type
        )::text
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER spatial_event_notify
    AFTER INSERT ON monitoring.observations
    FOR EACH ROW
    EXECUTE FUNCTION notify_spatial_event();
"""
```

### Custom GDAL VRT for Band Composition

```python
# build_band_composite_vrt.py
"""
Build a multi-band VRT from separate single-band files.
Useful for Sentinel-2 where each band is a separate file.
"""
from osgeo import gdal
import os

def build_multiband_vrt(
    band_files: dict,  # {"R": "B04.tif", "G": "B03.tif", "B": "B02.tif", "NIR": "B08.tif"}
    output_vrt: str,
):
    """Combine separate band files into a single multi-band VRT."""
    vrt_options = gdal.BuildVRTOptions(separate=True)
    vrt = gdal.BuildVRT(
        output_vrt,
        list(band_files.values()),
        options=vrt_options
    )

    # Set band descriptions
    for i, (band_name, _) in enumerate(band_files.items(), 1):
        band = vrt.GetRasterBand(i)
        band.SetDescription(band_name)

    vrt.FlushCache()
    vrt = None  # Close

    print(f"Created multi-band VRT: {output_vrt}")
    print(f"Bands: {list(band_files.keys())}")

# Example: Sentinel-2 L2A composite
band_files = {
    "Blue":  "/vsicurl/https://s3/sentinel2/B02_10m.tif",
    "Green": "/vsicurl/https://s3/sentinel2/B03_10m.tif",
    "Red":   "/vsicurl/https://s3/sentinel2/B04_10m.tif",
    "NIR":   "/vsicurl/https://s3/sentinel2/B08_10m.tif",
    "SWIR1": "/vsicurl/https://s3/sentinel2/B11_20m.tif",
    "SWIR2": "/vsicurl/https://s3/sentinel2/B12_20m.tif",
}

build_multiband_vrt(band_files, "sentinel2_composite.vrt")

# Then compute NDVI from the VRT without downloading anything
with rasterio.open("sentinel2_composite.vrt") as src:
    # Band 4 = NIR, Band 3 = Red (0-indexed: 3, 2)
    nir = src.read(4).astype(float)
    red = src.read(3).astype(float)
    ndvi = (nir - red) / (nir + red + 1e-10)
```

### FME Automation Triggered by S3 Events

Architecture: S3 Object Created event → SNS → SQS → FME Flow Automation → ETL workspace

```json
// fme-flow-automation-trigger.json
// FME Flow Automation definition (imported via REST API)
{
  "name": "S3_New_File_ETL",
  "description": "Triggered when new file arrives in S3 watched prefix",
  "enabled": true,
  "trigger": {
    "type": "AMAZON_S3",
    "config": {
      "bucket": "incoming-geodata",
      "prefix": "parcels/",
      "events": ["s3:ObjectCreated:*"],
      "sqsQueueUrl": "https://sqs.ap-southeast-2.amazonaws.com/123456/fme-trigger"
    }
  },
  "action": {
    "type": "RUN_WORKSPACE",
    "config": {
      "repository": "ETL_Workspaces",
      "workspace": "parcels_ingest.fmw",
      "parameters": {
        "SOURCE_DATASET": "$(trigger.s3.object.key)",
        "S3_BUCKET": "$(trigger.s3.bucket.name)",
        "TARGET_DB": "postgresql://postgis/geoapp",
        "TARGET_TABLE": "cadastre.parcels_staging",
        "LOG_LEVEL": "MEDIUM"
      }
    }
  }
}
```

```python
# Register the automation via FME Flow REST API
import requests

FME_SERVER = "https://fme.company.com"
FME_TOKEN = "your-fme-flow-token"

with open("fme-flow-automation-trigger.json") as f:
    automation_def = f.read()

response = requests.post(
    f"{FME_SERVER}/fmerest/v3/automations",
    headers={
        "Authorization": f"fmetoken token={FME_TOKEN}",
        "Content-Type": "application/json",
    },
    data=automation_def,
)
response.raise_for_status()
print(f"Automation created: {response.json()['id']}")
```

### DuckDB Spatial: Zero-Setup Spatial SQL

The fastest way to get started with spatial analytics. One binary, no server, native GeoParquet support, runs on files directly.

```python
import duckdb

conn = duckdb.connect()
conn.execute("INSTALL spatial; LOAD spatial;")

# Run spatial SQL directly on files (GeoParquet, GeoJSON, FlatGeobuf, Shapefile)
result = conn.execute("""
    -- Point-in-polygon join between two files, no database, no server
    SELECT
        p.parcel_id,
        p.land_use,
        z.zone_code,
        ST_Area(p.geometry) / 10000 AS area_ha
    FROM read_parquet('parcels.parquet') AS p
    JOIN st_read('zones.gpkg') AS z
        ON ST_Within(p.geometry, z.geometry)
    WHERE p.land_use = 'RURAL'
    ORDER BY area_ha DESC
    LIMIT 100
""").df()

# Export query results to GeoParquet
conn.execute("""
    COPY (
        SELECT * FROM st_read('input.gpkg')
        WHERE ST_Area(geometry) > 1000
    ) TO 'output_filtered.parquet'
    (FORMAT PARQUET, COMPRESSION ZSTD)
""")

# Read from S3 with httpfs extension
conn.execute("INSTALL httpfs; LOAD httpfs;")
conn.execute("SET s3_region='ap-southeast-2';")

result = conn.execute("""
    SELECT COUNT(*) as total_features,
           SUM(ST_Area(geometry)) / 1e6 as total_area_km2
    FROM read_parquet('s3://my-bucket/parcels/australia/*.parquet')
    WHERE state_code = 'VIC'
""").fetchone()

print(f"Victoria: {result[0]:,} parcels, {result[1]:.1f} km²")
```

---

## Quick Reference

### Essential ogr2ogr Flags

```bash
# The most useful flags for production ingestion
ogr2ogr \
  -f "PostgreSQL" \                    # Output format
  "PG:host=... dbname=..." \           # Destination
  input.gpkg \                         # Source
  -nln schema.table_name \             # Target layer name
  -lco GEOMETRY_NAME=geom \            # Geometry column name
  -lco FID=gid \                       # Primary key column name
  -lco SPATIAL_INDEX=GIST \            # Index type
  -nlt PROMOTE_TO_MULTI \              # Normalize mixed geometry types
  -makevalid \                         # Fix invalid geometries
  -t_srs EPSG:4326 \                   # Reproject on the fly
  -clipsrc 144 -38 146 -37 \           # Clip to bounding box
  -where "area_ha > 100" \             # Attribute filter
  -sql "SELECT ... FROM layer" \       # SQL filter (overrides -where)
  -fieldmap "oldcol=0,newcol=1" \      # Field mapping
  -overwrite \                         # Overwrite existing layer
  --config GDAL_NUM_THREADS ALL_CPUS \ # Parallel processing
  --config GDAL_CACHEMAX 1024 \        # 1GB block cache
  -progress                            # Show progress
```

### GDAL Environment Variables Cheat Sheet

```bash
# Performance
GDAL_CACHEMAX=2048              # Block cache in MB (default: ~200MB)
GDAL_NUM_THREADS=ALL_CPUS       # Enable threading
CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES  # Required for some S3 writes

# S3/Cloud access
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_DEFAULT_REGION=ap-southeast-2
GDAL_HTTP_TIMEOUT=60            # HTTP timeout for /vsicurl/
GDAL_HTTP_MAX_RETRY=5           # Retry failed HTTP requests
CPL_CURL_VERBOSE=YES            # Debug HTTP requests (verbose)

# Network virtual filesystems
GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR  # Skip slow directory listings on open
VSI_CACHE=TRUE                          # Enable per-file caching
VSI_CACHE_SIZE=536870912                # 512MB VSI cache
```

---

*Tools are current as of early 2026. Version numbers move fast in this space — always check the official docs for the latest release.*
