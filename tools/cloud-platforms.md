# Cloud Platforms

Cloud-based platforms for large-scale geospatial data analysis, visualization, and application development.

> **Quick Picks**
> - :trophy: **SOTA**: [Google Earth Engine](https://earthengine.google.com) --- petabyte-scale satellite imagery analysis with no infrastructure management
> - :moneybag: **Free Best**: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) --- free STAC catalog, free JupyterHub, excellent for environmental science
> - :zap: **Fastest Setup**: [Fused.io](https://fused.io) --- serverless geo UDFs; write Python, get a tile endpoint in minutes

## Comparison Table

| Platform | Free Tier | Latest Updates | API | Pricing Model |
|----------|-----------|---------------|-----|---------------|
| Google Earth Engine | Free for research/education/nonprofits | Dynamic World land cover, Sentinel-1 ARD, expanded commercial access | Python, JavaScript | Free (non-commercial), paid (commercial) |
| Microsoft Planetary Computer | Free (public datasets + compute hub) | Expanded STAC catalog, Sentinel-2 L2A global, ALOS World 3D | Python, STAC API | Free (compute hub), Azure pricing |
| AWS Earth Search v1 | Free public datasets; AWS free tier | Earth Search v1 STAC API, Overture Maps, Sentinel-2 COGs | REST, Boto3 | Pay-as-you-go |
| Wherobots | Free community tier | Managed Apache Sedona, Spatial SQL, notebook environment | Python, SQL | Freemium, usage-based |
| Fused.io | Free tier available | Serverless UDFs, tile server, integrations with DuckDB and QGIS | Python, HTTP | Freemium, usage-based |
| Mapbox | Free tier (50k map loads/mo) | MapLibre fork ecosystem, Mapbox Standard style, 3D terrain | REST, JS SDK | Freemium, usage-based |
| CARTO | Free tier (limited) | CARTO + BigQuery/Snowflake/Databricks integration, H3 analytics | REST, SQL API | Freemium, enterprise |

## Google Earth Engine

Google's cloud computing platform for geospatial analysis with a multi-petabyte data catalog. GEE continues to expand its data catalog and has opened commercial access for enterprise users.

- **Links**: [earthengine.google.com](https://earthengine.google.com) | [GEE Data Catalog](https://developers.google.com/earth-engine/datasets)

### Key Features

- Petabyte-scale satellite imagery archive (Landsat, Sentinel, MODIS, VIIRS, and 900+ datasets)
- Server-side computation -- no need to download data
- JavaScript (Code Editor) and Python API (`earthengine-api`, `geemap`)
- Community datasets and user asset uploads
- Export to Google Drive, Cloud Storage, or Earth Engine Assets
- Dynamic World: near-real-time global land cover classification
- Sentinel-1 SAR Analysis-Ready Data

### Access

- Free for research, education, and nonprofit use
- Commercial use requires Google Earth Engine for Commercial (contact Google)
- Sign up at [earthengine.google.com](https://earthengine.google.com)

### Quick Start (JavaScript Code Editor)

```javascript
// Load Sentinel-2 surface reflectance, filter and composite
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2024-01-01', '2024-12-31')
  .filterBounds(roi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .median();

Map.addLayer(s2, {bands: ['B4', 'B3', 'B2'], min: 0, max: 3000}, 'Sentinel-2 RGB');

// Compute NDVI
var ndvi = s2.normalizedDifference(['B8', 'B4']).rename('NDVI');
Map.addLayer(ndvi, {min: -0.2, max: 0.8, palette: ['red', 'yellow', 'green']}, 'NDVI');
```

### Quick Start (Python with geemap)

```python
import ee
import geemap

ee.Authenticate()
ee.Initialize(project='my-project')

Map = geemap.Map(center=[39.9, 116.4], zoom=10)

s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate('2024-01-01', '2024-12-31')
      .filterBounds(ee.Geometry.Point(116.4, 39.9))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .median())

vis = {'bands': ['B4', 'B3', 'B2'], 'min': 0, 'max': 3000}
Map.addLayer(s2, vis, 'Sentinel-2')
Map
```

## Microsoft Planetary Computer

Microsoft's platform providing access to environmental datasets with integrated compute through a JupyterHub environment. One of the best free resources for environmental data science.

- **Links**: [planetarycomputer.microsoft.com](https://planetarycomputer.microsoft.com) | [STAC API](https://planetarycomputer.microsoft.com/api/stac/v1)

### Key Features

- STAC-compliant data catalog with Landsat, Sentinel, NAIP, Copernicus DEM, ALOS, and environmental datasets
- Free JupyterHub environment (Planetary Computer Hub) with pre-installed geospatial stack
- Integrates with the PySTAC, xarray, Dask, and stackstac ecosystem
- Hosted on Azure with optional scaling to Azure compute
- Datasets include: Sentinel-2 L2A, Landsat Collection 2, NAIP, Copernicus DEM, Daymet, TerraClimate, ALOS World 3D

### Access

- Public STAC API is freely accessible (no account needed for metadata)
- Compute Hub requires free Microsoft account and approval
- [planetarycomputer.microsoft.com](https://planetarycomputer.microsoft.com)

### Quick Start

```python
import pystac_client
import planetary_computer
import stackstac

# Connect to the STAC API
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace,
)

# Search for Sentinel-2 imagery
search = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[116.0, 39.5, 117.0, 40.5],
    datetime="2024-06-01/2024-08-31",
    query={"eo:cloud_cover": {"lt": 20}},
)

items = search.item_collection()
print(f"Found {len(items)} scenes")

# Load into xarray via stackstac (lazy -- no data downloaded until compute)
stack = stackstac.stack(items, assets=["B04", "B03", "B02", "B08"])
median = stack.median(dim="time")
```

## AWS (Earth Search, S3 COG, Open Data)

Amazon Web Services infrastructure for hosting and processing geospatial data, including the AWS Open Data Program and the Element84-operated Earth Search STAC API.

- **Links**: [Earth Search v1](https://earth-search.aws.element84.com/v1) | [AWS Open Data](https://registry.opendata.aws/)

### Key Features

- AWS Open Data Program hosts free public datasets (Landsat, Sentinel, NAIP, Copernicus DEM, and more)
- Cloud-Optimized GeoTIFF (COG) support for efficient range-request access
- Earth Search v1: STAC-compliant search across major satellite collections
- Scalable compute via Lambda, EC2, SageMaker, and AWS Batch
- Overture Maps Foundation data published as GeoParquet on S3

### Key Datasets on AWS

| Dataset | S3 Bucket | Format | STAC Catalog | Update Cadence |
|---------|-----------|--------|-------------|----------------|
| Sentinel-2 L2A (COG) | sentinel-cogs | COG | Earth Search v1 | ~5 days |
| Landsat Collection 2 | usgs-landsat | COG | Earth Search v1 | ~16 days |
| NAIP Imagery | naip-analytic | COG | Earth Search v1 | ~2-3 year cycle |
| Copernicus DEM 30m | copernicus-dem-30m | COG | Earth Search v1 | Static |
| Overture Maps | overturemaps-us-west-2 | GeoParquet | N/A | Quarterly |
| ERA5 (Climate) | era5-pds | NetCDF | N/A | Monthly |
| NOAA GOES-16/17 | noaa-goes16 | NetCDF | N/A | Real-time |

### Quick Start

```python
from pystac_client import Client

# Connect to Earth Search v1
catalog = Client.open("https://earth-search.aws.element84.com/v1")

# Search for Sentinel-2 data
results = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[116.0, 39.5, 117.0, 40.5],
    datetime="2024-06-01/2024-08-31",
    max_items=10,
)

for item in results.items():
    print(f"{item.id}: cloud_cover={item.properties['eo:cloud_cover']:.1f}%")
```

## Wherobots

Managed cloud platform for Apache Sedona, providing scalable spatial SQL analytics with a notebook-based interface. Think "Databricks for geospatial."

- **Links**: [wherobots.com](https://www.wherobots.com) | [Apache Sedona](https://sedona.apache.org)

### Key Features

- Managed Apache Sedona (distributed spatial SQL engine)
- SQL, Python, and Scala interfaces
- Reads GeoParquet, Shapefile, GeoJSON, COG, and more
- Distributed spatial joins, KNN, and raster processing
- Built-in notebook environment with visualization
- Integrates with Spark ecosystem (Delta Lake, Iceberg)
- WherobotsDB: fully managed spatial data warehouse

### Use Cases

- Large-scale spatial joins (billions of records)
- Building footprint analysis across countries
- Mobility data processing (GPS traces, trajectories)
- Raster analytics at scale

### Quick Start

```python
# Wherobots / Apache Sedona Python
from sedona.spark import SedonaContext

sedona = SedonaContext.builder().getOrCreate()

# Read GeoParquet
df = sedona.read.format("geoparquet").load("s3://my-bucket/buildings.parquet")

# Spatial SQL
df.createOrReplaceTempView("buildings")
result = sedona.sql("""
    SELECT district_name, COUNT(*) as building_count, SUM(ST_Area(geometry)) as total_area
    FROM buildings
    GROUP BY district_name
    ORDER BY building_count DESC
""")
result.show()
```

## Fused.io

Serverless geospatial platform that lets you write Python UDFs (User Defined Functions) and instantly get tile endpoints, HTTP APIs, or batch outputs. Bridges the gap between data science notebooks and production tile servers.

- **Links**: [fused.io](https://www.fused.io) | [Fused Docs](https://docs.fused.io)

### Key Features

- Write Python UDFs; get a live tile server URL instantly
- Serverless execution -- no infrastructure to manage
- Reads from S3, GCS, HTTP (GeoParquet, COG, FlatGeobuf, etc.)
- Integrates with DuckDB, QGIS, Leaflet, MapLibre, Deck.gl
- Community UDF catalog with pre-built geospatial functions
- Low-latency tile serving for web applications

### Use Cases

- Rapid prototyping of spatial analytics as live map tiles
- Serving derived data products (NDVI, building counts, H3 aggregations)
- Connecting Jupyter notebooks to live map interfaces
- Building spatial APIs without backend infrastructure

### Quick Start

```python
# Fused UDF example
import fused

@fused.udf
def udf(bbox: fused.types.Bbox):
    import geopandas as gpd
    import duckdb

    # Query Overture Maps buildings within the current viewport
    con = duckdb.connect()
    con.sql("INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;")

    df = con.sql(f"""
        SELECT geometry, names.primary as name
        FROM read_parquet('s3://overturemaps-us-west-2/release/2024-09-18.0/theme=buildings/type=building/*')
        WHERE bbox.xmin BETWEEN {bbox.xmin} AND {bbox.xmax}
          AND bbox.ymin BETWEEN {bbox.ymin} AND {bbox.ymax}
        LIMIT 5000
    """).df()

    return gpd.GeoDataFrame(df, geometry="geometry")
```

## Mapbox

Platform for custom map design, geocoding, navigation, and spatial data visualization. The Mapbox GL ecosystem (and its open-source fork MapLibre GL) powers a large share of web mapping.

- **Links**: [mapbox.com](https://www.mapbox.com) | [Mapbox GL JS](https://docs.mapbox.com/mapbox-gl-js/)

### Key Features

- Custom map styling with Mapbox Studio
- Vector tile hosting and serving (tileset API)
- Geocoding, routing, isochrone, and matrix APIs
- Mapbox GL JS for high-performance web maps (WebGL-based)
- Mapbox Standard: 3D terrain, atmosphere, and building extrusions
- Satellite imagery basemap

### Free Tier

- 50,000 map loads per month
- 100,000 geocoding requests per month
- 100,000 routing requests per month
- [mapbox.com/pricing](https://www.mapbox.com/pricing)

## CARTO

Cloud-native spatial analytics platform for business intelligence and data science. CARTO operates as a layer on top of your existing cloud data warehouse (BigQuery, Snowflake, Redshift, Databricks).

- **Links**: [carto.com](https://carto.com) | [CARTO Analytics Toolbox](https://docs.carto.com/analytics-toolbox-bigquery/overview/)

### Key Features

- SQL-based spatial analytics (CARTO Analytics Toolbox) deployed inside your data warehouse
- Native integration with BigQuery, Snowflake, Redshift, Databricks, and PostgreSQL
- Builder for no-code spatial visualization and dashboards
- Spatial data enrichment from Data Observatory (demographics, POIs, boundaries)
- H3 spatial indexing and aggregation built into the Analytics Toolbox
- Geocoding, routing, and isoline analysis

### Free Tier

- CARTO Free plan with limited data and map views
- Student and research programs available
- [carto.com/pricing](https://carto.com/pricing)

## Platform Selection Guide

| Use Case | Recommended Platform |
|----------|---------------------|
| Satellite imagery time-series analysis | Google Earth Engine |
| Environmental / climate data science | Microsoft Planetary Computer |
| Custom web maps with design control | Mapbox |
| Spatial analytics on data warehouse | CARTO |
| Distributed spatial SQL at scale | Wherobots (Apache Sedona) |
| Serverless spatial APIs / tile serving | Fused.io |
| General cloud infrastructure + COG hosting | AWS (S3 + Earth Search) |
