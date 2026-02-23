# CLI Tools

Command-line utilities for batch processing, format conversion, data transformation, and automation in geospatial workflows.

> **Quick Picks**
> - :trophy: **SOTA**: [GDAL 3.9+](https://gdal.org) --- the universal Swiss Army knife; every other tool builds on it
> - :moneybag: **Free Best**: [tippecanoe](https://github.com/felt/tippecanoe) --- best-in-class vector tile generation, now maintained by Felt with PMTiles output
> - :zap: **Fastest Setup**: [Rasterio CLI](https://rasterio.readthedocs.io/) --- `pip install rasterio` and you are ready to go

## GDAL/OGR

The foundational geospatial library. GDAL handles raster data, OGR handles vector data. Nearly every GIS tool builds on top of GDAL. Version **3.9+** adds native GeoParquet read/write, FlatGeobuf streaming, improved COG creation, and Arrow-based batch processing for vector data.

- **Current Version**: 3.9 (2024) / 3.10 (2025)
- **Install**: `conda install -c conda-forge gdal` or `brew install gdal` (macOS) or `apt install gdal-bin` (Debian/Ubuntu)
- **Links**: [gdal.org](https://gdal.org) | [GDAL Cheat Sheet](https://github.com/dwtkns/gdal-cheat-sheet)

### Key GDAL Commands

```bash
# Get raster info
gdalinfo input.tif

# Reproject a raster to EPSG:4326 (WGS 84)
gdalwarp -t_srs EPSG:4326 input.tif output.tif

# Convert format (e.g., ERDAS Imagine to GeoTIFF)
gdal_translate -of GTiff input.img output.tif

# Build virtual raster mosaic from many tiles
gdalbuildvrt mosaic.vrt *.tif

# --- Cloud-Optimized GeoTIFF (COG) creation ---
# Basic COG creation
gdal_translate -of COG input.tif output_cog.tif

# COG with LZW compression and overview levels (recommended for web serving)
gdal_translate -of COG \
  -co COMPRESS=LZW \
  -co OVERVIEW_RESAMPLING=LANCZOS \
  -co BLOCKSIZE=512 \
  input.tif output_cog.tif

# Validate a COG
python -m cogeo_mosaic.utils validate output_cog.tif

# Clip raster to bounding box
gdalwarp -te 116.0 39.5 117.0 40.5 input.tif clipped.tif

# Clip raster using a vector mask (shapefile)
gdalwarp -cutline boundary.shp -crop_to_cutline input.tif clipped.tif

# Merge multiple rasters into one
gdal_merge.py -o merged.tif input1.tif input2.tif

# Compute hillshade from a DEM
gdaldem hillshade dem.tif hillshade.tif -z 2 -az 315 -alt 45

# Compute slope from a DEM (in degrees)
gdaldem slope dem.tif slope.tif

# Raster calculator: compute NDVI from bands
gdal_calc.py -A nir.tif -B red.tif --outfile ndvi.tif \
  --calc="(A.astype(float)-B)/(A.astype(float)+B)" --type=Float32

# Polygonize raster to vector
gdal_polygonize.py classified.tif -f "GPKG" output.gpkg

# Convert NetCDF to GeoTIFF (specific subdataset)
gdal_translate NETCDF:"climate.nc":temperature output.tif

# Read raster directly from S3 (COG via /vsicurl/)
gdalinfo /vsicurl/https://example.com/data/imagery.tif
```

### Key OGR Commands (ogr2ogr)

```bash
# Get vector file info (summary)
ogrinfo -al -so input.shp

# --- Format Conversions ---

# Shapefile to GeoJSON
ogr2ogr -f GeoJSON output.geojson input.shp

# GeoJSON to GeoPackage
ogr2ogr -f GPKG output.gpkg input.geojson

# Shapefile to GeoPackage (with reprojection)
ogr2ogr -f GPKG -t_srs EPSG:4326 output.gpkg input.shp

# GeoJSON to FlatGeobuf (great for web streaming)
ogr2ogr -f FlatGeobuf output.fgb input.geojson

# CSV with lat/lon columns to GeoPackage
ogr2ogr -f GPKG output.gpkg input.csv \
  -oo X_POSSIBLE_NAMES=longitude -oo Y_POSSIBLE_NAMES=latitude \
  -a_srs EPSG:4326

# GeoPackage to GeoParquet (GDAL 3.9+)
ogr2ogr -f Parquet output.parquet input.gpkg

# --- SQL Queries on Files ---

# Filter features with SQL
ogr2ogr -sql "SELECT * FROM input WHERE pop > 100000" output.gpkg input.gpkg

# Spatial query: select features within a bounding box
ogr2ogr -spat 116.0 39.5 117.0 40.5 output.gpkg input.gpkg

# Join two layers with SQL
ogr2ogr -sql "SELECT a.*, b.pop FROM layer1 a LEFT JOIN layer2 b ON a.id = b.id" \
  output.gpkg input.gpkg

# --- Clip & Transform ---

# Clip vector to boundary polygon
ogr2ogr -clipsrc boundary.shp output.shp input.shp

# Reproject vector data
ogr2ogr -t_srs EPSG:3857 output.gpkg input.gpkg

# --- PostGIS Import/Export ---

# Import Shapefile into PostGIS
ogr2ogr -f PostgreSQL PG:"host=localhost dbname=mydb user=postgres" input.shp

# Export PostGIS table to GeoJSON
ogr2ogr -f GeoJSON output.geojson \
  PG:"host=localhost dbname=mydb" -sql "SELECT * FROM cities"

# --- Performance Tips ---

# Speed up large file operations with config options
ogr2ogr -f GPKG output.gpkg input.geojson \
  --config OGR_SQLITE_SYNCHRONOUS OFF \
  --config OGR_SQLITE_CACHE 1024

# Read directly from a remote URL (HTTP/S3)
ogr2ogr -f GPKG local.gpkg /vsicurl/https://example.com/data.geojson
```

## Rasterio CLI

Python-based command-line tool for raster data operations, built on GDAL but with a more Pythonic interface. Part of the Mapbox/Rasterio ecosystem.

- **Install**: `pip install rasterio` (includes the `rio` CLI)
- **Links**: [rasterio.readthedocs.io](https://rasterio.readthedocs.io/)

```bash
# Get raster info (JSON output)
rio info input.tif --indent 2

# Reproject to EPSG:4326
rio warp input.tif output.tif --dst-crs EPSG:4326

# Clip to bounding box
rio clip input.tif output.tif --bounds 116.0 39.5 117.0 40.5

# Calculate NDVI (band 4 = NIR, band 3 = Red)
rio calc "(- (read 1 4) (read 1 3)) / (+ (read 1 4) (read 1 3))" input.tif -o ndvi.tif

# Merge multiple rasters
rio merge tile1.tif tile2.tif tile3.tif -o merged.tif

# Create raster overview (pyramids)
rio overview input.tif --resampling lanczos 2 4 8 16

# --- rio-cogeo: Cloud-Optimized GeoTIFF tools ---
# Install: pip install rio-cogeo

# Validate if a file is a valid COG
rio cogeo validate input.tif

# Create a COG with web-optimized profile
rio cogeo create input.tif output_cog.tif --web-optimized --overview-level 5

# Create a COG with specific compression
rio cogeo create input.tif output_cog.tif -p lzw --overview-resampling lanczos
```

## Fiona

Python command-line tool for vector data, the vector counterpart to Rasterio. Provides `fio` commands for reading, converting, and filtering vector features.

- **Install**: `pip install fiona`
- **Links**: [fiona.readthedocs.io](https://fiona.readthedocs.io/)

```bash
# Inspect vector file metadata
fio info input.shp --indent 2

# Convert format (pipe-based approach)
fio cat input.shp | fio load -f GeoJSON output.geojson

# Filter features with an expression
fio cat input.shp --where "population > 50000" | fio load output.geojson

# Get feature count
fio cat input.shp | wc -l

# Reproject and convert in one pipeline
fio cat input.shp --dst-crs EPSG:4326 | fio load -f GPKG output.gpkg

# Dump schema information
fio info input.shp --layer layername
```

## tippecanoe

Tool for creating vector tilesets from GeoJSON data. Originally by Mapbox, now maintained by **Felt** with added support for **PMTiles** output -- a single-file, cloud-optimized vector tile format.

- **Current Version**: 2.x (Felt fork)
- **Install**: `brew install tippecanoe` (macOS) or build from source
- **Links**: [github.com/felt/tippecanoe](https://github.com/felt/tippecanoe)

```bash
# Create MBTiles from GeoJSON (automatic zoom range)
tippecanoe -o output.mbtiles -zg --drop-densest-as-needed input.geojson

# Create PMTiles (single-file cloud-native vector tiles)
tippecanoe -o output.pmtiles -zg --drop-densest-as-needed input.geojson

# Specific zoom range (min 2, max 14)
tippecanoe -o output.pmtiles -Z2 -z14 input.geojson

# Preserve all features at all zoom levels (large output)
tippecanoe -o output.pmtiles -zg --no-feature-limit --no-tile-size-limit input.geojson

# Multiple layers in one tileset
tippecanoe -o output.pmtiles \
  -L buildings:buildings.geojson \
  -L roads:roads.geojson \
  -L pois:pois.geojson

# Generate tiles with specific layer name
tippecanoe -o output.pmtiles -l my_layer -zg input.geojson

# Coalesce (simplify) features for lower zooms and keep detail at higher zooms
tippecanoe -o output.pmtiles -zg \
  --coalesce-densest-as-needed \
  --extend-zooms-if-still-dropping \
  input.geojson
```

### PMTiles Workflow

PMTiles is a single-file archive format for tiled data. Serve it from any static host (S3, CloudFlare R2, GitHub Pages) without a tile server.

```bash
# 1. Create PMTiles from GeoJSON
tippecanoe -o buildings.pmtiles -zg --drop-densest-as-needed buildings.geojson

# 2. Upload to S3 (or any static host)
aws s3 cp buildings.pmtiles s3://my-bucket/tiles/buildings.pmtiles \
  --content-type application/vnd.pmtiles

# 3. Serve with MapLibre GL JS (client-side range requests)
# No tile server needed -- the browser reads byte ranges directly from the file.
```

## PDAL (Point Data Abstraction Library)

The GDAL equivalent for point cloud (LiDAR) data. PDAL reads/writes LAS, LAZ, EPT, COPC, and many other formats with a pipeline-based architecture.

- **Install**: `conda install -c conda-forge pdal` or `brew install pdal`
- **Links**: [pdal.io](https://pdal.io)

```bash
# Get LAS file info
pdal info input.las

# Convert LAS to LAZ (compressed)
pdal translate input.las output.laz

# Create a COPC (Cloud-Optimized Point Cloud) file
pdal translate input.las output.copc.laz --writers.copc.forward=all

# Pipeline: read, filter ground points, write
cat <<'EOF' > ground_pipeline.json
{
  "pipeline": [
    "input.las",
    {
      "type": "filters.smrf",
      "comment": "Simple Morphological Filter for ground classification"
    },
    {
      "type": "filters.range",
      "limits": "Classification[2:2]",
      "comment": "Keep only ground points (class 2)"
    },
    "ground_only.las"
  ]
}
EOF
pdal pipeline ground_pipeline.json

# Pipeline: read, reproject, thin, write
cat <<'EOF' > reproject_pipeline.json
{
  "pipeline": [
    "input.las",
    {
      "type": "filters.reprojection",
      "in_srs": "EPSG:32633",
      "out_srs": "EPSG:4326"
    },
    {
      "type": "filters.sample",
      "radius": 1.0,
      "comment": "Thin to ~1 point per 1m radius"
    },
    "output_thinned.laz"
  ]
}
EOF
pdal pipeline reproject_pipeline.json

# Generate a DEM (raster) from point cloud
pdal translate input.las dem.tif \
  --writers.gdal.resolution=1.0 \
  --writers.gdal.output_type=mean

# Merge multiple LAS files
pdal merge file1.las file2.las file3.las merged.las

# Compute height above ground using HAG filter
cat <<'EOF' > hag_pipeline.json
{
  "pipeline": [
    "classified.las",
    { "type": "filters.hag_nn" },
    {
      "type": "filters.range",
      "limits": "HeightAboveGround[2:]",
      "comment": "Keep only points >= 2m above ground (vegetation/buildings)"
    },
    "above_ground.las"
  ]
}
EOF
pdal pipeline hag_pipeline.json
```

## Additional CLI Tools

| Tool | Purpose | Install | Language | Notes |
|------|---------|---------|----------|-------|
| `mapshaper` | Vector simplification, format conversion, dissolve | `npm install -g mapshaper` | JavaScript | Excellent for topology-aware simplification |
| `geojson-merge` | Merge GeoJSON files | `npm install -g geojson-merge` | JavaScript | Simple merge utility |
| `togeojson` | Convert KML/GPX to GeoJSON | `npm install -g @mapbox/togeojson` | JavaScript | Part of the Mapbox toolchain |
| `csvkit` | CSV processing, SQL on CSV | `pip install csvkit` | Python | Great for geocoding prep and data cleaning |
| `sqlite-utils` | SQLite/SpatiaLite manipulation | `pip install sqlite-utils` | Python | Create, query, and manage spatial SQLite dbs |
| `pdal` | Point cloud (LiDAR) processing | `conda install -c conda-forge pdal` | C++ | Pipeline-based; LAS, LAZ, COPC, EPT formats |
| `las2las` | LAS/LAZ file conversion and filtering | Part of LAStools | C++ | Free for non-commercial use |
| `proj` | Coordinate transformation | `apt install proj-bin` | C | `cs2cs` for batch coord transforms |
| `osmium` | OpenStreetMap data processing | `apt install osmium-tool` | C++ | Filter, merge, extract OSM PBF/XML files |
| `rio-cogeo` | COG creation and validation | `pip install rio-cogeo` | Python | Dedicated COG tooling built on Rasterio |
| `stac-cli` | Browse and search STAC catalogs | `pip install pystac-client[cli]` | Python | Query Earth Search, Planetary Computer, etc. |
| `duckdb` | SQL analytics on spatial files | `brew install duckdb` or `pip install duckdb` | C++ | Reads GeoParquet, GeoJSON, Shapefile natively |
| `pmtiles` | Inspect and serve PMTiles archives | `go install github.com/protomaps/go-pmtiles` | Go | CLI for the PMTiles ecosystem |

## Recommended Workflows

### Shapefile to Web-Ready Vector Tiles

```bash
# 1. Convert Shapefile to GeoJSON (with reprojection to WGS 84)
ogr2ogr -f GeoJSON -t_srs EPSG:4326 buildings.geojson buildings.shp

# 2. Simplify if needed (mapshaper)
mapshaper buildings.geojson -simplify 50% -o simplified.geojson

# 3. Generate PMTiles
tippecanoe -o buildings.pmtiles -zg --drop-densest-as-needed simplified.geojson

# 4. Upload to static hosting
aws s3 cp buildings.pmtiles s3://my-bucket/tiles/
```

### Satellite Imagery to Cloud-Optimized GeoTIFF

```bash
# 1. Mosaic individual scenes
gdalbuildvrt mosaic.vrt scene_*.tif

# 2. Reproject to Web Mercator
gdalwarp -t_srs EPSG:3857 -r bilinear mosaic.vrt reprojected.tif

# 3. Create COG with compression
gdal_translate -of COG -co COMPRESS=DEFLATE -co BLOCKSIZE=512 reprojected.tif final_cog.tif

# 4. Validate
rio cogeo validate final_cog.tif
```

### LiDAR to DEM

```bash
# 1. Classify ground points
pdal translate input.laz classified.laz --filter smrf

# 2. Extract ground-only points and generate DEM
pdal translate classified.laz dem.tif \
  --readers.las.filename=classified.laz \
  --writers.gdal.resolution=0.5 \
  --writers.gdal.output_type=idw

# 3. Generate hillshade for visualization
gdaldem hillshade dem.tif hillshade.tif -z 2 -az 315 -alt 45

# 4. Package as COG
gdal_translate -of COG -co COMPRESS=LZW dem.tif dem_cog.tif
```
