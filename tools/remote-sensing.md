# Remote Sensing & Earth Observation Tools

Tools for processing satellite imagery, aerial photography, LiDAR point clouds, SAR data, and hyperspectral imagery — from raw sensor data to analysis-ready products.

> **Quick Picks**
> - :trophy: **SOTA**: [Google Earth Engine](https://earthengine.google.com) + [geemap](https://geemap.org) --- planetary-scale server-side processing with a Pythonic interface; nothing else comes close for multi-decade time series at global scale
> - :moneybag: **Free Best**: [SNAP](https://step.esa.int/main/toolboxes/snap/) + [Sen2Cor](https://step.esa.int/main/snap-supported-plugins/sen2cor/) --- the official ESA Sentinel processing suite; handles everything from SAR to ocean color out of the box
> - :zap: **Fastest Setup**: [leafmap](https://leafmap.org) + STAC --- `pip install leafmap` and you are querying and visualizing live satellite data from Planetary Computer within 5 minutes

---

## Satellite Data Processing Suites

### SNAP (ESA Sentinel Application Platform)

The official ESA toolbox for Sentinel-1, -2, -3, and -6 data. Java-based desktop GUI plus a powerful `gpt` (Graph Processing Tool) CLI for headless batch processing. Nearly every ESA-affiliated workflow starts here.

- **Version**: 10.x (2024+)
- **Install**: [step.esa.int](https://step.esa.int/main/toolboxes/snap/) — download installer or use Docker
- **Links**: [SNAP docs](https://senbox.atlassian.net/wiki/spaces/SNAP/overview) | [STEP Forum](https://forum.step.esa.int)

#### Sentinel-1 SAR Preprocessing Chain

The canonical preprocessing pipeline for Sentinel-1 GRD data before any backscatter analysis:

```bash
# Full S1 GRD preprocessing via gpt (Graph Processing Tool)
# Step 1: Apply orbit file (precise orbits for accurate geolocation)
gpt Apply-Orbit-File \
  -Ssource=S1A_IW_GRDH_1SDV_20240101.zip \
  -t S1_orb.dim \
  -PorbitType="Sentinel Precise (Auto Download)"

# Step 2: Thermal noise removal (for GRD products)
gpt ThermalNoiseRemoval -Ssource=S1_orb.dim -t S1_tnr.dim

# Step 3: Radiometric calibration to sigma0
gpt Calibration \
  -Ssource=S1_tnr.dim \
  -t S1_cal.dim \
  -PoutputSigmaBand=true \
  -PoutputBetaBand=false \
  -PoutputGammaBand=false

# Step 4: Speckle filtering (Lee Sigma is a good default)
gpt Speckle-Filter \
  -Ssource=S1_cal.dim \
  -t S1_spk.dim \
  -Pfilter="Lee Sigma" \
  -PfilterSizeX=5 \
  -PfilterSizeY=5

# Step 5: Range-Doppler Terrain Correction (geocoding)
gpt Terrain-Correction \
  -Ssource=S1_spk.dim \
  -t S1_TC.dim \
  -PdemName="SRTM 1Sec HGT" \
  -PpixelSpacingInMeter=10.0 \
  -PmapProjection="EPSG:32633"

# Step 6: Linear to dB conversion
gpt LinearToFromdB -Ssource=S1_TC.dim -t S1_TC_dB.dim
```

Chain all steps in a single SNAP Graph XML for reproducibility:

```xml
<!-- snap_s1_grd_preprocessing.xml — run with: gpt snap_s1_grd_preprocessing.xml -->
<graph id="S1_GRD_Preprocessing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>${inputFile}</file>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>false</continueOnFail>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources>
      <sourceProduct refid="Apply-Orbit-File"/>
    </sources>
    <parameters>
      <removeThermalNoise>true</removeThermalNoise>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources>
      <sourceProduct refid="ThermalNoiseRemoval"/>
    </sources>
    <parameters>
      <outputSigmaBand>true</outputSigmaBand>
      <outputBetaBand>false</outputBetaBand>
      <outputGammaBand>false</outputGammaBand>
      <outputImageScaleInDb>false</outputImageScaleInDb>
    </parameters>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources>
      <sourceProduct refid="Calibration"/>
    </sources>
    <parameters>
      <filter>Lee Sigma</filter>
      <filterSizeX>5</filterSizeX>
      <filterSizeY>5</filterSizeY>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources>
      <sourceProduct refid="Speckle-Filter"/>
    </sources>
    <parameters>
      <demName>SRTM 1Sec HGT</demName>
      <pixelSpacingInMeter>10.0</pixelSpacingInMeter>
      <mapProjection>EPSG:32633</mapProjection>
      <nodataValueAtSea>false</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
    </parameters>
  </node>
  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources>
      <sourceProduct refid="Terrain-Correction"/>
    </sources>
    <parameters/>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="LinearToFromdB"/>
    </sources>
    <parameters>
      <file>${outputFile}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
    </parameters>
  </node>
</graph>
```

```bash
# Run the graph with variable substitution
gpt snap_s1_grd_preprocessing.xml \
  -PinputFile=/data/S1A_IW_GRDH_1SDV_20240101.zip \
  -PoutputFile=/output/S1_processed.tif
```

#### Sentinel-2 Atmospheric Correction with Sen2Cor

```bash
# Install Sen2Cor plugin via SNAP plugin manager, then:
# Run L2A processing (L1C → L2A atmospheric correction)
L2A_Process.sh \
  --resolution=10 \
  /data/S2A_MSIL1C_20240101T100031_N0510_R122_T33UVP_20240101T120000.SAFE

# For 20m and 60m resolutions too:
L2A_Process.sh --resolution=20 /data/S2A_MSIL1C_*.SAFE
L2A_Process.sh --resolution=60 /data/S2A_MSIL1C_*.SAFE
```

#### Sentinel-2 Band Math and Indices in SNAP

```bash
# NDVI computation via BandMaths operator
gpt BandMaths \
  -Ssource=/data/S2_L2A.dim \
  -t /output/S2_NDVI.dim \
  -PtargetBandDescriptors='<targetBands>
    <targetBand>
      <name>NDVI</name>
      <type>float32</type>
      <expression>(B8 - B4) / (B8 + B4)</expression>
      <noDataValue>-999</noDataValue>
    </targetBand>
  </targetBands>'
```

#### snappy: Python API for SNAP

```python
import snappy
from snappy import ProductIO, GPF, HashMap

# Read a Sentinel-2 product
product = ProductIO.readProduct('/data/S2A_MSIL1C_20240101.SAFE/MTD_MSIL1C.xml')
print(f"Bands: {list(product.getBandNames())}")
print(f"Scene size: {product.getSceneRasterWidth()} x {product.getSceneRasterHeight()}")

# Apply Subset to AOI before heavy processing (critical for performance)
params = HashMap()
params.put('geoRegion', 'POLYGON((12.0 41.5, 12.5 41.5, 12.5 42.0, 12.0 42.0, 12.0 41.5))')
params.put('copyMetadata', True)
subset = GPF.createProduct('Subset', params, product)

# Resample all bands to 10m
resample_params = HashMap()
resample_params.put('targetResolution', 10)
resampled = GPF.createProduct('Resample', resample_params, subset)

# Write output
ProductIO.writeProduct(resampled, '/output/S2_subset_10m', 'GeoTIFF-BigTIFF')
```

> **TRICK**: Always run `SubsetOp` as your first operator in any SNAP chain. Processing a 100km x 100km AOI subset instead of a full Sentinel-2 granule (110km x 110km) is barely faster, but if your AOI is a city (10km x 10km) it is 100x faster and saves gigabytes of intermediate disk I/O.

> **TRICK**: Use `gpt` inside Docker for reproducible cloud processing:
> ```bash
> docker run --rm -v /data:/data mundialis/esa-snap:ubuntu \
>   gpt /data/graphs/s1_preprocessing.xml \
>   -PinputFile=/data/S1A_scene.zip \
>   -PoutputFile=/data/output/S1_processed.tif
> ```

---

### Orfeo ToolBox (OTB)

CNES (French Space Agency) open-source image processing library. 100+ algorithms, handles massive images without loading them fully into RAM, integrates into QGIS Processing, and is the go-to tool for object-based image analysis (OBIA) at scale.

- **Version**: 9.x
- **Install**: `conda install -c conda-forge otb` or download binaries from [orfeo-toolbox.org](https://www.orfeo-toolbox.org)
- **Links**: [OTB docs](https://www.orfeo-toolbox.org/CookBook/) | [Algorithm list](https://www.orfeo-toolbox.org/CookBook/Applications.html)

```bash
# Pansharpening: merge 10m panchromatic + 30m multispectral to 10m colour
otbcli_Pansharpening \
  -inp /data/Pleiades_PAN.tif \
  -inxs /data/Pleiades_MS.tif \
  -out /output/Pleiades_pansharpen.tif \
  -method bayes

# Large-Scale Mean-Shift (LSMS) segmentation — the gold standard for OBIA
# Step 1: Smoothing
otbcli_MeanShiftSmoothing \
  -in /data/S2_composite.tif \
  -fout /tmp/smooth_spatial.tif \
  -foutpos /tmp/smooth_range.tif \
  -spatialr 16 \
  -ranger 15 \
  -maxiter 100

# Step 2: Segmentation
otbcli_LSMSSegmentation \
  -in /tmp/smooth_spatial.tif \
  -inpos /tmp/smooth_range.tif \
  -out /tmp/segments.tif \
  -spatialr 16 \
  -ranger 15

# Step 3: Small region merging
otbcli_LSMSSmallRegionsMerging \
  -in /data/S2_composite.tif \
  -inseg /tmp/segments.tif \
  -out /tmp/segments_merged.tif \
  -minsize 50

# Step 4: Vectorization
otbcli_LSMSVectorization \
  -in /data/S2_composite.tif \
  -inseg /tmp/segments_merged.tif \
  -out /output/segments.gpkg

# Supervised classification with Random Forest
otbcli_TrainImagesClassifier \
  -io.il /data/S2_composite.tif \
  -io.vd /data/training_samples.gpkg \
  -io.out /models/rf_model.yaml \
  -classifier rf \
  -classifier.rf.max 200 \
  -classifier.rf.nbtrees 100

otbcli_ImageClassifier \
  -in /data/S2_composite.tif \
  -model /models/rf_model.yaml \
  -out /output/classification.tif
```

```python
# OTB Python API
import otbApplication as otb

# Compute NDVI via BandMath
app = otb.Registry.CreateApplication("BandMath")
app.SetParameterStringList("il", ["/data/S2_B08_B04.tif"])
app.SetParameterString("out", "/output/NDVI.tif")
app.SetParameterString("exp", "(im1b1 - im1b2) / (im1b1 + im1b2)")  # B8 = band1, B4 = band2
app.ExecuteAndWriteOutput()
```

> **TRICK**: OTB's LSMS segmentation is the only open-source tool that handles arbitrarily large images through streaming. You can segment a 50,000 x 50,000 pixel mosaic on a laptop with 8 GB RAM — SNAP and Python-based tools will run out of memory long before that.

---

### ERDAS IMAGINE / ENVI (Commercial)

Industry standards for advanced spectral analysis and enterprise workflows.

- **ENVI**: best-in-class hyperspectral analysis, spectral angle mapper (SAM), spectral unmixing, subpixel classification, ENVI SARscape for SAR. IDL scripting for automation.
- **ERDAS IMAGINE**: spatial modeler for complex multi-step workflows, photogrammetric tools (LPS), advanced change detection.
- **When to choose over free tools**: hyperspectral unmixing with >200 bands, enterprise audit trails, support contracts, tight ArcGIS integration.

---

## Cloud-Native EO Processing

### Google Earth Engine (geemap + earthengine-api)

The most powerful planetary-scale EO analysis platform available. Server-side processing means you never download raw data — you send code to the data. Access to the full Landsat archive (1972–present), all Sentinel missions, MODIS, and hundreds of curated datasets.

- **Install**: `pip install earthengine-api geemap`
- **Authentication**: `earthengine authenticate`
- **Links**: [earthengine.google.com](https://earthengine.google.com) | [geemap.org](https://geemap.org) | [GEE Community datasets](https://github.com/samapriya/awesome-gee-community-datasets)

```python
import ee
import geemap

ee.Initialize(project='your-gee-project')

# --- Cloud-free Sentinel-2 composite (percentile trick is better than median) ---
def mask_s2_clouds(image):
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
           qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start'])

# Using percentile composite — captures phenological peak NDVI better than median
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate('2023-06-01', '2023-09-30')
      .filterBounds(ee.Geometry.Rectangle([12.0, 41.5, 13.5, 42.5]))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
      .map(mask_s2_clouds)
      .select(['B2','B3','B4','B8','B11','B12']))

# 75th percentile composite: emphasizes vegetated, cloud-free pixels
composite = s2.reduce(ee.Reducer.percentile([75]))

# --- NDVI time series ---
def add_ndvi(image):
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return image.addBands(ndvi)

ts = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterDate('2020-01-01', '2024-01-01')
      .filterBounds(ee.Geometry.Point([12.5, 41.9]))
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
      .map(mask_s2_clouds)
      .map(add_ndvi))

# Extract time series at a point as a DataFrame
point = ee.Geometry.Point([12.5, 41.9])
chart_data = ts.select('NDVI').getRegion(point, 10).getInfo()

# --- Land cover classification ---
# Load training samples (FeatureCollection with 'landcover' property)
training_fc = ee.FeatureCollection('users/yourname/training_samples_rome')

# Sample composite at training points
training = composite.sampleRegions(
    collection=training_fc,
    properties=['landcover'],
    scale=10
)

# Train Random Forest classifier
classifier = ee.Classifier.smileRandomForest(numberOfTrees=100).train(
    features=training,
    classProperty='landcover',
    inputProperties=composite.bandNames()
)

# Classify the composite
classified = composite.classify(classifier)

# --- Batch zonal statistics export (the right way) ---
aoi_fc = ee.FeatureCollection('users/yourname/municipalities')

stats = composite.select('B8_p75').reduceRegions(
    collection=aoi_fc,
    reducer=ee.Reducer.mean().combine(
        reducer2=ee.Reducer.stdDev(), sharedInputs=True
    ),
    scale=10
)

task = ee.batch.Export.table.toDrive(
    collection=stats,
    description='zonal_stats_NIR',
    fileFormat='CSV',
    folder='GEE_exports'
)
task.start()
```

```python
# geemap interactive visualization
Map = geemap.Map(center=[41.9, 12.5], zoom=9)
Map.addLayer(composite, {'bands': ['B4_p75','B3_p75','B2_p75'], 'min': 0, 'max': 0.3}, 'RGB')
Map.addLayer(classified, {'min': 0, 'max': 5, 'palette': ['green','yellow','red','blue','gray','brown']}, 'Classification')
Map.split_map(
    left_layer='TERRAIN',
    right_layer='HYBRID'
)
# Time-lapse GIF of NDVI change
geemap.create_timelapse(
    collection=ts.select('NDVI'),
    out_gif='/output/ndvi_timelapse.gif',
    start_year=2020,
    end_year=2023,
    bands=['NDVI'],
    vis_params={'min': 0, 'max': 1, 'palette': ['red','yellow','green']},
    dimensions=600
)
Map
```

> **TRICK**: `ee.Reducer.percentile([25,50,75])` for cloud-free composites produces significantly better results than `.median()` in tropical regions where cloud contamination is systematic. The 75th percentile tends to pick up peak-green phenological states in vegetation pixels while rejecting cloud edges.

> **TRICK**: Always use `ee.batch.Export.table.toDrive()` with `ee.FeatureCollection` for zonal statistics instead of `.getInfo()`. The latter has a 5000-feature hard limit and will silently truncate results. The export task handles millions of features.

> **TRICK**: For server-side conditional logic, use `ee.Algorithms.If()` sparingly — it forces materialization of both branches. Instead use `.where()` and `.updateMask()` for raster conditionals, which stay lazy.

---

### openEO

Open standard API for cloud EO processing across multiple backends. One Python client, many backends. Free processing on Copernicus Data Space Ecosystem (30 EUR credits/month by default).

- **Install**: `pip install openeo`
- **Links**: [openeo.org](https://openeo.org) | [Copernicus backend](https://openeo.dataspace.copernicus.eu)

```python
import openeo

# Connect to Copernicus Data Space (free tier)
conn = openeo.connect("https://openeo.dataspace.copernicus.eu").authenticate_oidc()

# Load Sentinel-2 L2A
s2 = conn.load_collection(
    "SENTINEL2_L2A",
    spatial_extent={"west": 12.0, "south": 41.5, "east": 13.0, "north": 42.0},
    temporal_extent=["2023-06-01", "2023-09-01"],
    bands=["B04", "B08", "SCL"]
)

# Cloud masking using SCL (Scene Classification Layer — values 4=vegetation, 5=not-vegetated, 6=water are clean)
def mask_clouds(data):
    scl = data.band("SCL")
    valid_mask = (scl == 4) | (scl == 5) | (scl == 6)
    return data.filter_bbox(valid_mask)

# Compute NDVI and temporal max composite
ndvi = s2.ndvi(nir="B08", red="B04")
ndvi_max = ndvi.reduce_dimension(reducer="max", dimension="t")

# Download result
job = ndvi_max.save_result("GTiff").create_job(title="NDVI max composite")
job.start_and_wait()
results = job.get_results()
results.download_files("/output/ndvi_max/")
```

---

### Sentinel Hub

Commercial cloud EO API with per-km² pricing and a generous free trial. Best for operational monitoring workflows requiring fresh data and OGC service endpoints.

- **Install**: `pip install sentinelhub eo-learn`
- **Links**: [sentinel-hub.com](https://www.sentinel-hub.com) | [eo-learn docs](https://eo-learn.readthedocs.io)

```python
from sentinelhub import (SHConfig, SentinelHubRequest, DataCollection,
                          MimeType, CRS, BBox, bbox_to_dimensions)

config = SHConfig()
config.sh_client_id = 'your-client-id'
config.sh_client_secret = 'your-client-secret'

# Custom evalscript for True Color + NDVI composite output
evalscript = """
//VERSION=3
function setup() {
  return {
    input: ["B02", "B03", "B04", "B08"],
    output: [
      { id: "default", bands: 3, sampleType: "FLOAT32" },
      { id: "ndvi",    bands: 1, sampleType: "FLOAT32" }
    ]
  };
}
function evaluatePixel(sample) {
  let ndvi = (sample.B08 - sample.B04) / (sample.B08 + sample.B04);
  return {
    default: [sample.B04/3000, sample.B03/3000, sample.B02/3000],
    ndvi: [ndvi]
  };
}
"""

bbox = BBox([12.0, 41.5, 13.0, 42.0], crs=CRS.WGS84)
size = bbox_to_dimensions(bbox, resolution=10)

request = SentinelHubRequest(
    evalscript=evalscript,
    input_data=[SentinelHubRequest.input_data(
        data_collection=DataCollection.SENTINEL2_L2A,
        time_interval=('2023-06-01', '2023-09-01'),
        mosaicking_order='leastCC'  # least cloud cover first
    )],
    responses=[
        SentinelHubRequest.output_response('default', MimeType.TIFF),
        SentinelHubRequest.output_response('ndvi', MimeType.TIFF),
    ],
    bbox=bbox,
    size=size,
    config=config
)

data = request.get_data()
rgb_array, ndvi_array = data[0]['default.tif'], data[0]['ndvi.tif']
```

---

## STAC Ecosystem

SpatioTemporal Asset Catalog (STAC) is the standard metadata format for satellite imagery. The ecosystem below lets you discover, stream, and process petabytes of satellite data without managing downloads.

### Key STAC Catalogs

| Catalog | Provider | Key Datasets | URL |
|---|---|---|---|
| Earth Search | AWS / Element84 | Sentinel-2, Landsat C2, NAIP, Copernicus DEM | `https://earth-search.aws.element84.com/v1` |
| Planetary Computer | Microsoft | Sentinel-1/2, Landsat, MODIS, ESA WorldCover | `https://planetarycomputer.microsoft.com/api/stac/v1` |
| Copernicus Data Space | ESA | Sentinel-1/2/3/5P, full archive | `https://catalogue.dataspace.copernicus.eu/stac` |
| USGS STAC | USGS | Landsat Collection 2, 3DEP LiDAR | `https://landsatlook.usgs.gov/stac-server` |

### pystac-client

```python
from pystac_client import Client
import geopandas as gpd
from shapely.geometry import box

# Search Sentinel-2 on AWS Earth Search
catalog = Client.open("https://earth-search.aws.element84.com/v1")

aoi = box(12.0, 41.5, 13.0, 42.0)

items = catalog.search(
    collections=["sentinel-2-l2a"],
    intersects=aoi.__geo_interface__,
    datetime="2023-06-01/2023-09-30",
    query={"eo:cloud_cover": {"lt": 10}}
).item_collection()

print(f"Found {len(items)} scenes")

# Show available assets for first item
for key, asset in items[0].assets.items():
    print(f"  {key}: {asset.href}")
```

### stackstac — STAC to xarray in one line

```python
import stackstac
import pystac_client
import numpy as np

catalog = pystac_client.Client.open("https://earth-search.aws.element84.com/v1")
items = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[12.0, 41.5, 13.0, 42.0],
    datetime="2023-06-01/2023-09-30",
    query={"eo:cloud_cover": {"lt": 15}}
).item_collection()

# STAC items → lazy Dask-backed xarray.DataArray (no download yet!)
stack = stackstac.stack(
    items,
    assets=["red", "nir", "swir16"],
    resolution=10,
    epsg=32633,
    bounds_latlon=[12.0, 41.5, 13.0, 42.0]
)
# stack is shape (time, band, y, x) — fully lazy

# Cloud masking using SCL
scl = stackstac.stack(items, assets=["scl"], resolution=20, epsg=32633,
                       bounds_latlon=[12.0, 41.5, 13.0, 42.0])
valid = scl.isel(band=0).isin([4, 5, 6])  # vegetation, non-veg, water
valid_10m = valid.interp_like(stack, method='nearest')

stack_masked = stack.where(valid_10m)

# Compute NDVI median composite — triggers actual computation
nir = stack_masked.sel(band="nir")
red = stack_masked.sel(band="red")
ndvi = (nir - red) / (nir + red)
ndvi_median = ndvi.median("time").compute()  # now data is downloaded and computed
```

### odc-stac — Open Data Cube STAC loader

```python
import odc.stac
import pystac_client

catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace  # PC requires signed URLs
)

items = catalog.search(
    collections=["sentinel-2-l2a"],
    bbox=[12.0, 41.5, 13.0, 42.0],
    datetime="2023-06-01/2023-09-01"
).item_collection()

# Load + reproject + mosaic in one call — output is xarray.Dataset
ds = odc.stac.load(
    items,
    bands=["B04", "B08", "B11", "SCL"],
    crs="EPSG:32633",
    resolution=10,
    bbox=[12.0, 41.5, 13.0, 42.0],
    groupby="solar_day",     # mosaic tiles from same day
    chunks={"x": 2048, "y": 2048}  # Dask chunks
)
# ds is xr.Dataset with variables B04, B08, B11, SCL; dims (time, y, x)

ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04)
```

---

## Spectral Index & Band Math Reference

Common indices for Sentinel-2 (band numbers) and Landsat 8/9 (band numbers). All formulas follow the standard convention; values in parentheses are typical threshold values for the feature of interest.

### Sentinel-2 Band Reference

| Band | Name | Central wavelength | Resolution |
|---|---|---|---|
| B02 | Blue | 490 nm | 10 m |
| B03 | Green | 560 nm | 10 m |
| B04 | Red | 665 nm | 10 m |
| B08 | NIR | 842 nm | 10 m |
| B05 | Red Edge 1 | 705 nm | 20 m |
| B06 | Red Edge 2 | 740 nm | 20 m |
| B07 | Red Edge 3 | 783 nm | 20 m |
| B8A | NIR narrow | 865 nm | 20 m |
| B11 | SWIR 1 | 1610 nm | 20 m |
| B12 | SWIR 2 | 2190 nm | 20 m |

### Index Formulas

| Index | Formula (S2 bands) | Landsat 8/9 equiv. | Typical range / threshold |
|---|---|---|---|
| **NDVI** | `(B08 - B04) / (B08 + B04)` | `(B5 - B4) / (B5 + B4)` | -1 to 1; vegetation > 0.3 |
| **EVI** | `2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)` | same formula | Reduces soil/atmosphere bias |
| **SAVI** | `1.5 * (B08 - B04) / (B08 + B04 + 0.5)` | same | Sparse vegetation / arid |
| **NDWI** (Gao) | `(B08 - B11) / (B08 + B11)` | `(B5 - B6) / (B5 + B6)` | Vegetation water content |
| **MNDWI** (Xu) | `(B03 - B11) / (B03 + B11)` | `(B3 - B6) / (B3 + B6)` | Open water > 0 |
| **NDBI** | `(B11 - B08) / (B11 + B08)` | `(B6 - B5) / (B6 + B5)` | Built-up > 0 |
| **NDMI** | `(B08 - B11) / (B08 + B11)` | `(B5 - B6) / (B5 + B6)` | Moisture; same as NDWI Gao |
| **NBR** | `(B08 - B12) / (B08 + B12)` | `(B5 - B7) / (B5 + B7)` | Burn severity; dNBR for change |
| **BSI** | `((B11 + B04) - (B08 + B02)) / ((B11 + B04) + (B08 + B02))` | similar | Bare soil > 0 |
| **NDSI** | `(B03 - B11) / (B03 + B11)` | `(B3 - B6) / (B3 + B6)` | Snow > 0.4 |
| **NDRE** | `(B07 - B05) / (B07 + B05)` | N/A (no Red Edge on L8) | Crop chlorophyll content |
| **CHL-RE** | `(B07 / B05) - 1` | N/A | Canopy chlorophyll (mg/m²) |

```python
import rioxarray as rxr
import numpy as np

# Load Sentinel-2 bands (after resampling to 10m)
ds = rxr.open_rasterio('/data/S2_L2A_10m.tif', chunks={'x': 2048, 'y': 2048})
# Assume band order: B02, B03, B04, B08, B11, B12 (0-indexed)
B02 = ds.sel(band=1).astype('float32') / 10000
B03 = ds.sel(band=2).astype('float32') / 10000
B04 = ds.sel(band=3).astype('float32') / 10000
B08 = ds.sel(band=4).astype('float32') / 10000
B11 = ds.sel(band=5).astype('float32') / 10000
B12 = ds.sel(band=6).astype('float32') / 10000

ndvi  = (B08 - B04) / (B08 + B04)
mndwi = (B03 - B11) / (B03 + B11)
ndbi  = (B11 - B08) / (B11 + B08)
nbr   = (B08 - B12) / (B08 + B12)
evi   = 2.5 * (B08 - B04) / (B08 + 6*B04 - 7.5*B02 + 1)

# Stack all indices into a multi-band output
import xarray as xr
indices = xr.concat([ndvi, mndwi, ndbi, nbr, evi], dim='band')
indices['band'] = ['NDVI', 'MNDWI', 'NDBI', 'NBR', 'EVI']
indices.rio.to_raster('/output/spectral_indices.tif', driver='GTiff',
                       compress='LZW', dtype='float32')
```

---

## Classification & Change Detection

### Supervised Classification Workflow

```python
import numpy as np
import rasterio
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import joblib

# --- Step 1: Extract training samples ---
with rasterio.open('/data/spectral_indices.tif') as src:
    img_array = src.read()         # (bands, rows, cols)
    transform = src.transform
    crs = src.crs

training = gpd.read_file('/data/training_samples.gpkg').to_crs(crs)

# Sample raster values at training polygon centroids
from rasterio.sample import sample_gen
coords = [(geom.centroid.x, geom.centroid.y) for geom in training.geometry]
sampled = list(sample_gen(src, coords))  # list of arrays, one per point

X = np.array(sampled)  # (n_samples, n_bands)
y = training['class_id'].values

# --- Step 2: Train & validate ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', oob_score=True, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

print(f"OOB accuracy: {rf.oob_score_:.3f}")
print(classification_report(y_test, rf.predict(X_test)))

# Feature importance
for name, imp in zip(['NDVI','MNDWI','NDBI','NBR','EVI'], rf.feature_importances_):
    print(f"  {name}: {imp:.3f}")

joblib.dump(rf, '/models/rf_landcover_v1.pkl')

# --- Step 3: Apply classifier to full image ---
n_bands, rows, cols = img_array.shape
X_full = img_array.reshape(n_bands, -1).T  # (pixels, bands)
X_full = np.nan_to_num(X_full, nan=-9999)

classified = rf.predict(X_full).reshape(rows, cols).astype('uint8')

with rasterio.open('/output/classified.tif', 'w',
                   driver='GTiff', height=rows, width=cols,
                   count=1, dtype='uint8', crs=crs, transform=transform,
                   compress='LZW') as dst:
    dst.write(classified, 1)
```

### Change Detection: dNBR for Burn Severity

```python
import rasterio
import numpy as np

# Pre-fire and post-fire NBR
with rasterio.open('/data/S2_prefire_NBR.tif') as src:
    nbr_pre = src.read(1).astype('float32')
    profile = src.profile

with rasterio.open('/data/S2_postfire_NBR.tif') as src:
    nbr_post = src.read(1).astype('float32')

# dNBR: positive values = burned, negative = regrowth
dnbr = nbr_pre - nbr_post

# USGS burn severity classes
severity = np.zeros_like(dnbr, dtype='uint8')
severity[dnbr < -0.100] = 1                          # Enhanced regrowth (high)
severity[(dnbr >= -0.100) & (dnbr < 0.100)] = 2      # Unburned
severity[(dnbr >= 0.100) & (dnbr < 0.269)] = 3       # Low severity
severity[(dnbr >= 0.269) & (dnbr < 0.439)] = 4       # Moderate-low
severity[(dnbr >= 0.439) & (dnbr < 0.659)] = 5       # Moderate-high
severity[dnbr >= 0.659] = 6                           # High severity

profile.update(dtype='uint8', count=1, compress='LZW')
with rasterio.open('/output/burn_severity.tif', 'w', **profile) as dst:
    dst.write(severity, 1)
```

### CCDC (Continuous Change Detection and Classification)

CCDC detects change in any pixel over a dense time series by fitting harmonic models and detecting anomalies:

```python
# pyccd library — Python implementation of CCDC algorithm
# pip install pyccd
import ccd
import numpy as np

# Time series for a single pixel: (date_ordinal, [blue, green, red, nir, swir1, swir2, thermal], qa)
result = ccd.detect(dates, blues, greens, reds, nirs, swirs1, swirs2, thermals, qas)

for segment in result['change_models']:
    print(f"Stable period: {segment['start_day']} → {segment['end_day']}")
    print(f"Change detected: {segment.get('break_day', 'no break')}")
```

---

## SAR Processing

### InSAR Pipeline: Surface Deformation Monitoring

InSAR (Interferometric SAR) measures ground surface displacement at millimeter scale. The full chain below produces a deformation map from two Sentinel-1 SLC acquisitions:

```bash
# 1. Download a Sentinel-1 SLC pair (same relative orbit, ~12 days apart)
# Use ASF Data Search: https://search.asf.alaska.edu

# 2. Coregistration + interferogram formation in SNAP
gpt InSAR-Coregistration-Interferogram \
  -Sslave=S1A_20240101_SLC.zip \
  -Smaster=S1A_20231220_SLC.zip \
  -t /tmp/ifg.dim \
  -PcohWinAz=10 \
  -PcohWinRg=3

# 3. Goldstein phase filtering
gpt GoldsteinPhaseFiltering -Ssource=/tmp/ifg.dim -t /tmp/ifg_filt.dim

# 4. Coherence-masked unwrapping via SNAPHU (external tool)
# Export to SNAPHU format first
gpt SnaphuExport -Ssource=/tmp/ifg_filt.dim -PtargetFolder=/tmp/snaphu_export/

# Run SNAPHU (install: sudo apt install snaphu)
snaphu -s /tmp/snaphu_export/Phase_ifg_VV.snaphu.img 20000 \
  -f /tmp/snaphu_export/snaphu.conf \
  -o /tmp/snaphu_export/Unw_Phase.img \
  -c /tmp/snaphu_export/Coherence_VV.snaphu.img

# 5. Import unwrapped phase back to SNAP
gpt SnaphuImport \
  -Ssource=/tmp/ifg_filt.dim \
  -Ssnaphu=/tmp/snaphu_export/Unw_Phase.img \
  -t /tmp/ifg_unw.dim

# 6. Phase to displacement conversion
gpt PhaseToDisplacement -Ssource=/tmp/ifg_unw.dim -t /tmp/displacement.dim

# 7. Terrain correction to geographic coordinates
gpt Terrain-Correction \
  -Ssource=/tmp/displacement.dim \
  -t /output/deformation_map.tif \
  -PdemName="SRTM 1Sec HGT" \
  -PpixelSpacingInMeter=20.0
```

### MintPy: Time-Series InSAR (PSI/SBAS)

MintPy processes a stack of InSAR pairs to produce millimeter-precision deformation time series. Used for subsidence monitoring, volcano inflation/deflation, landslide creep.

- **Install**: `conda install -c conda-forge mintpy`
- **Links**: [MintPy docs](https://mintpy.readthedocs.io)

```bash
# After ISCE2 or HyP3 preprocessing, run MintPy:
smallbaselineApp.py /path/to/smallbaselineApp.cfg

# Config highlights:
# mintpy.load.processor = isce
# mintpy.load.unwFile = ../merged/interferograms/*/filt_*.unw
# mintpy.load.corFile = ../merged/interferograms/*/filt_*.cor
# mintpy.reference.lalo = 41.9, 12.5   (reference point: assumed stable)
# mintpy.troposphericDelay.method = pyaps  (ERA5 atmospheric correction)
# mintpy.deramp = linear                (remove orbital ramp)

# Output: velocity map (mm/yr) + displacement time series in HDF5
```

### ASF HyP3: On-Demand Cloud SAR Processing

Pre-processed, analysis-ready Sentinel-1 data with zero local computation. Free for NASA-funded researchers, very affordable for others.

```python
import hyp3_sdk as sdk

hyp3 = sdk.HyP3(username='your-earthdata-username', password='your-password')

# Request RTC (Radiometric Terrain Correction) products
jobs = hyp3.submit_rtc_job(
    granule='S1A_IW_GRDH_1SDV_20240101T094000_20240101T094025_052028_064A4D',
    name='rome_s1_rtc',
    resolution=10,         # 10m pixels
    dem_matching=True,
    include_dem=False,
    include_inc_map=True,
    speckle_filter=True
)

# Wait for completion and download
jobs = hyp3.watch(jobs)
for job in jobs:
    job.download_files('/output/s1_rtc/')
```

> **TRICK**: Sentinel-1 RTC products from ASF HyP3 are the single biggest time-saver in SAR analysis. Instead of running the 7-step SNAP chain above, you get a georeferenced, terrain-corrected, speckle-filtered GeoTIFF delivered to your S3 bucket. For continental-scale flood mapping, this is the only scalable approach.

---

## LiDAR & Point Cloud Processing

### PDAL (Point Data Abstraction Library)

The GDAL of point clouds. Pipeline-based processing: read → filter → write, with 60+ stages.

- **Install**: `conda install -c conda-forge pdal python-pdal`
- **Links**: [pdal.io](https://pdal.io)

```json
{
  "pipeline": [
    {
      "type": "readers.copc",
      "filename": "https://s3.amazonaws.com/usgs-lidar-public/IA_FullState/ept.json"
    },
    {
      "type": "filters.crop",
      "bounds": "([477000, 479000], [4618000, 4620000])"
    },
    {
      "type": "filters.elm",
      "comment": "Extended Local Minimum — removes low noise (underground points)"
    },
    {
      "type": "filters.outlier",
      "method": "statistical",
      "mean_k": 12,
      "multiplier": 2.2
    },
    {
      "type": "filters.smrf",
      "comment": "Simple Morphological Filter for ground classification"
    },
    {
      "type": "filters.hag_nn",
      "comment": "Height Above Ground computation"
    },
    {
      "type": "writers.gdal",
      "filename": "/output/canopy_height.tif",
      "resolution": 1.0,
      "output_type": "max",
      "dimension": "HeightAboveGround"
    }
  ]
}
```

```bash
# Run PDAL pipeline
pdal pipeline canopy_height_pipeline.json

# Quick info on a point cloud file
pdal info --summary /data/pointcloud.laz

# Tile a massive LAZ file into COPC format for streaming access
pdal translate /data/massive.laz /output/massive.copc.laz \
  --writer copc

# Count points in a spatial query against remote COPC
pdal info \
  --readers.copc.filename="https://s3.amazonaws.com/usgs-lidar-public/IA_FullState/ept.json" \
  --filters.crop.bounds="([477000, 479000], [4618000, 4620000])" \
  --filters.crop.a_srs="EPSG:26915"
```

### lidR (R package): Forestry LiDAR Analysis

```r
library(lidR)

# Read and visualize
las <- readLAS("/data/forest_plot.laz")
plot(las, color = "Z")

# Ground classification (Cloth Simulation Filter)
las <- classify_ground(las, algorithm = csf())

# Normalize heights (subtract DTM)
las <- normalize_height(las, algorithm = tin())

# Digital Terrain Model at 1m resolution
dtm <- rasterize_terrain(las, res = 1, algorithm = tin())

# Canopy Height Model
chm <- rasterize_canopy(las, res = 0.5, algorithm = p2r(0.2))

# Individual Tree Detection
ttops <- locate_trees(chm, algorithm = lmf(ws = 5, hmin = 5))

# Tree segmentation
las_seg <- segment_trees(las, algorithm = dalponte2016(chm, ttops))

# Extract tree metrics
metrics <- crown_metrics(las_seg, func = .stdtreemetrics)
# metrics: tree_id, Z (height), convhull_area (crown area), ...

writeRaster(chm, "/output/canopy_height_model.tif")
st_write(metrics, "/output/tree_metrics.gpkg")
```

### CloudCompare

Open-source 3D point cloud viewer and processor. Essential for visual inspection, ICP registration, and change detection between two point clouds (e.g., repeat surveys).

```bash
# CloudCompare CLI: compute C2C (cloud-to-cloud) distance between two surveys
CloudCompare \
  -SILENT \
  -O /data/survey_2023.laz \
  -O /data/survey_2024.laz \
  -C2C_DIST \
  -SAVE_CLOUDS \
  -C2C_SPLIT_XY_Z \
  -NO_TIMESTAMP
```

---

## Photogrammetry & Structure from Motion

### OpenDroneMap (ODM)

Fully open-source drone image processing: orthomosaic, DSM, DTM, 3D point cloud, 3D textured mesh.

- **Install**: `pip install nodeodm` or use Docker
- **Links**: [opendronemap.org](https://opendronemap.org) | [WebODM](https://webodm.net)

```bash
# Docker: process a folder of drone images
docker run -ti --rm \
  -v /data/drone_images:/datasets/project \
  opendronemap/odm \
  --project-path /datasets \
  project \
  --dsm \
  --dtm \
  --orthophoto-resolution 2 \
  --dem-resolution 5 \
  --mesh-size 300000 \
  --feature-quality high \
  --pc-quality ultra

# Output in /data/drone_images/odm_orthophoto/odm_orthophoto.tif
#            /data/drone_images/odm_dem/dsm.tif
#            /data/drone_images/odm_dem/dtm.tif
#            /data/drone_images/odm_georeferencing/odm_georeferenced_model.laz
```

```python
# PyODM: programmatic ODM via NodeODM REST API
from pyodm import Node

node = Node("localhost", 3000)

task = node.create_task(
    files=['/data/drone_images/' + f for f in os.listdir('/data/drone_images') if f.endswith('.JPG')],
    options={
        "dsm": True,
        "dtm": True,
        "orthophoto-resolution": 2,
        "feature-quality": "high"
    }
)

task.wait_for_completion()
task.download_assets('/output/odm_results/')
```

---

## Hyperspectral Processing

### Spectral Python (SPy)

```python
import spectral as spy
import numpy as np

# Read ENVI format hyperspectral image (common for AVIRIS, PRISMA)
img = spy.open_image('/data/AVIRIS_f180803t01_corr.hdr')
data = img.load()  # (rows, cols, bands)
print(f"Image: {data.shape}, wavelengths: {img.bands.centers[:5]} nm")

# Spectral angle mapper (SAM) classification against a reference library
from spectral import spectral_angles
library = spy.open_image('/data/spectral_library.hdr').load()

# Compute SAM scores between each pixel and all library spectra
angles = spectral_angles(data, library)
classified = np.argmin(angles, axis=2)  # class with minimum angle

# PCA for dimensionality reduction
pc = spy.principal_components(img)
pc_data = pc.transform(data, num_components=10)

# Minimum Noise Fraction (MNF) transform — better than PCA for hyperspectral
mnf = spy.mnf(img)
mnf_data = mnf.transform(data, num_components=10)

# Pixel Purity Index for endmember extraction
ppi_scores = spy.ppi(mnf_data, niters=1000)
# High PPI scores = spectrally pure pixels = candidate endmembers

spy.save_image('/output/classified.hdr', classified.astype('uint16'), dtype=np.uint16)
```

### EMIT (Earth Surface Mineral Dust Source Investigation)

NASA EMIT hyperspectral data (2022+, ISS-mounted) covers 380–2500 nm at 60m resolution globally.

```python
import earthaccess
import xarray as xr

# Authenticate with NASA Earthdata
earthaccess.login()

# Search EMIT L2A reflectance
results = earthaccess.search_data(
    short_name="EMITL2ARFL",
    temporal=("2023-06-01", "2023-08-31"),
    bounding_box=(-120, 35, -115, 40)
)

earthaccess.download(results[:3], local_path='/data/emit/')

# Open EMIT netCDF
ds = xr.open_dataset('/data/emit/EMIT_L2A_RFL_001_20230601T180000.nc',
                     group='reflectance')
# (downtrack, crosstrack, bands)
reflectance = ds['reflectance']
wavelengths = ds['wavelengths']

# Extract SWIR spectra for mineral mapping
swir_mask = (wavelengths > 2000) & (wavelengths < 2500)
swir_cube = reflectance.sel(bands=swir_mask)
```

---

## Advanced Dark Arts

These are non-obvious techniques that separate efficient pipelines from slow ones.

### Lazy Dask-Backed Raster Processing

```python
import rioxarray as rxr
import dask
from dask.distributed import Client, LocalCluster

# Start a local Dask cluster (tune workers/threads to your CPU/RAM)
cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='4GB')
client = Client(cluster)
print(client.dashboard_link)  # Dask dashboard for monitoring

# Open a 50 GB raster lazily — no data loaded yet
big_raster = rxr.open_rasterio(
    '/data/national_S2_mosaic.tif',
    chunks={'band': 1, 'x': 4096, 'y': 4096},  # 4096x4096 Dask chunks
    lock=False  # allow parallel reads
)

# All operations below are lazy (no computation yet)
ndvi = (big_raster.sel(band=4) - big_raster.sel(band=3)) / \
       (big_raster.sel(band=4) + big_raster.sel(band=3))

ndvi_mean = ndvi.mean()  # still lazy!

# Trigger computation — distributed across Dask workers
result = ndvi_mean.compute()
print(f"National mean NDVI: {result.item():.3f}")

# Write output (Dask writes chunk-by-chunk, memory-efficient)
ndvi.rio.to_raster(
    '/output/national_ndvi.tif',
    driver='GTiff',
    compress='LZW',
    tiled=True,
    blockxsize=512,
    blockysize=512
)
```

### Pre-Computing Cloud Masks with s2cloudless

```python
from s2cloudless import S2PixelCloudDetector
import numpy as np
import rasterio

# s2cloudless uses 10 specific bands in this exact order:
# B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12
band_paths = {
    'B01': '/data/S2/B01.tif', 'B02': '/data/S2/B02.tif',
    'B04': '/data/S2/B04.tif', 'B05': '/data/S2/B05.tif',
    'B08': '/data/S2/B08.tif', 'B8A': '/data/S2/B8A.tif',
    'B09': '/data/S2/B09.tif', 'B10': '/data/S2/B10.tif',
    'B11': '/data/S2/B11.tif', 'B12': '/data/S2/B12.tif',
}

bands = []
for band in ['B01','B02','B04','B05','B08','B8A','B09','B10','B11','B12']:
    with rasterio.open(band_paths[band]) as src:
        bands.append(src.read(1).astype('float32') / 10000.0)

# Stack to (1, rows, cols, 10) — batch dimension first
img_stack = np.stack(bands, axis=-1)[np.newaxis, ...]

cloud_detector = S2PixelCloudDetector(threshold=0.4, average_over=4, dilation_size=2)
cloud_proba = cloud_detector.get_cloud_probability_maps(img_stack)   # (1, rows, cols)
cloud_mask  = cloud_detector.get_cloud_masks(img_stack)              # (1, rows, cols) bool

# Write cloud probability as a companion GeoTIFF
with rasterio.open(band_paths['B02']) as src:
    profile = src.profile
    profile.update(count=1, dtype='float32', compress='LZW')

with rasterio.open('/output/cloud_probability.tif', 'w', **profile) as dst:
    dst.write(cloud_proba[0], 1)
```

### Landsat Collection 2 QA_PIXEL Masking

```python
import numpy as np
import rasterio

def decode_landsat_qa(qa_band):
    """
    Landsat Collection 2 QA_PIXEL bit flags:
    Bit 0: Fill
    Bit 1: Dilated Cloud
    Bit 3: Cloud
    Bit 4: Cloud Shadow
    Bit 5: Snow
    Bit 6: Clear
    Bit 7: Water
    """
    fill          = (qa_band & (1 << 0)) > 0
    dilated_cloud = (qa_band & (1 << 1)) > 0
    cloud         = (qa_band & (1 << 3)) > 0
    cloud_shadow  = (qa_band & (1 << 4)) > 0
    snow          = (qa_band & (1 << 5)) > 0
    clear         = (qa_band & (1 << 6)) > 0
    water         = (qa_band & (1 << 7)) > 0

    # Valid = clear AND not fill AND not cloud AND not shadow AND not dilated
    valid_mask = clear & ~fill & ~cloud & ~cloud_shadow & ~dilated_cloud
    return valid_mask, water, snow

with rasterio.open('/data/LC09_L2SP_198029_20240101_QA_PIXEL.TIF') as src:
    qa = src.read(1)

valid, water, snow = decode_landsat_qa(qa)
print(f"Valid pixels: {valid.sum() / valid.size * 100:.1f}%")
print(f"Water pixels: {water.sum() / valid.size * 100:.1f}%")
print(f"Snow pixels:  {snow.sum() / valid.size * 100:.1f}%")
```

### Sentinel-2 SCL as Instant Land Cover Proxy

The Scene Classification Layer (SCL) in Sentinel-2 L2A is a single-band free land cover map delivered with every scene. Use it before training any classifier:

```python
# SCL class values:
SCL_CLASSES = {
    0:  'No Data',
    1:  'Saturated / Defective',
    2:  'Dark Area Pixels',
    3:  'Cloud Shadows',
    4:  'Vegetation',
    5:  'Not Vegetated',
    6:  'Water',
    7:  'Unclassified',
    8:  'Cloud (Medium Prob)',
    9:  'Cloud (High Prob)',
    10: 'Thin Cirrus',
    11: 'Snow or Ice'
}

import rasterio
import numpy as np
from collections import Counter

with rasterio.open('/data/S2_L2A_SCL_20m.tif') as src:
    scl = src.read(1)

# Quick land cover summary
pixel_counts = Counter(scl.ravel())
total = scl.size
for cls_id, count in sorted(pixel_counts.items()):
    pct = count / total * 100
    name = SCL_CLASSES.get(cls_id, 'Unknown')
    print(f"  {cls_id:2d} {name:<25}: {pct:5.1f}%")

# Use SCL as cloud mask (valid = SCL in 4,5,6,7,11)
cloud_free_mask = np.isin(scl, [4, 5, 6, 7, 11])
```

### NDVI Composite with rasterio.merge

```python
import rasterio
from rasterio.merge import merge
from pathlib import Path

# Combine multiple scene NDVIs into one max-value composite
ndvi_files = sorted(Path('/data/ndvi_scenes/').glob('*.tif'))
datasets = [rasterio.open(f) for f in ndvi_files]

# method='max' keeps the highest NDVI value at each pixel
# — equivalent to "greenest pixel" composite, robust against cloud edges
composite, transform = merge(datasets, method='max', nodata=-9999)

profile = datasets[0].profile
profile.update(
    driver='GTiff',
    height=composite.shape[1],
    width=composite.shape[2],
    transform=transform,
    compress='LZW',
    dtype='float32'
)

with rasterio.open('/output/ndvi_max_composite.tif', 'w', **profile) as dst:
    dst.write(composite)

for ds in datasets:
    ds.close()
```

### SNAP Graph Builder for Reproducible Processing

Export your SNAP GUI workflow as an XML graph, commit it to version control, and replay it identically on any machine:

```bash
# From SNAP GUI: Tools → Graph Builder → build your chain visually → Save graph as XML

# Version control the graphs
git add snap_graphs/
git commit -m "Add S1 GRD preprocessing graph with Lee Sigma filter 5x5"

# Run any graph in CI/CD pipeline
gpt /repo/snap_graphs/s1_grd_preprocessing.xml \
  -PinputFile=${INPUT_FILE} \
  -PoutputFile=${OUTPUT_FILE} \
  -q 8  # 8 parallel execution threads
```

### Streaming Remote COPC Point Clouds with PDAL

```python
import pdal
import json

# Query a remote COPC file on AWS S3 without downloading it
pipeline_json = {
    "pipeline": [
        {
            "type": "readers.copc",
            "filename": "https://s3.amazonaws.com/usgs-lidar-public/NY_NewYorkCity_2021/ept.json",
            "bounds": "([978000, 979000], [198000, 199000])"  # spatial crop
        },
        {
            "type": "filters.returns",
            "returns": "first"  # first returns only for surface model
        },
        {
            "type": "writers.gdal",
            "filename": "/output/nyc_dsm_1m.tif",
            "resolution": 1.0,
            "output_type": "max",
            "window_size": 3
        }
    ]
}

pipeline = pdal.Pipeline(json.dumps(pipeline_json))
pipeline.execute()
print(f"Points processed: {pipeline.arrays[0].shape[0]}")
```

### Writing Custom STAC Items for Internal Catalogs

```python
import pystac
from datetime import datetime
from pathlib import Path
import rasterio

def create_stac_item_from_geotiff(tif_path: str, collection_id: str) -> pystac.Item:
    """Generate a STAC Item from a local GeoTIFF."""
    tif_path = Path(tif_path)

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        crs = src.crs
        bbox = [bounds.left, bounds.bottom, bounds.right, bounds.top]

    from shapely.geometry import box, mapping
    from pyproj import Transformer
    transformer = Transformer.from_crs(crs.to_epsg(), 4326, always_xy=True)
    west, south = transformer.transform(bbox[0], bbox[1])
    east, north = transformer.transform(bbox[2], bbox[3])
    bbox_wgs84 = [west, south, east, north]
    geometry = mapping(box(*bbox_wgs84))

    item = pystac.Item(
        id=tif_path.stem,
        geometry=geometry,
        bbox=bbox_wgs84,
        datetime=datetime.utcnow(),
        properties={
            "collection": collection_id,
            "processing:level": "L2A",
            "platform": "sentinel-2a",
        }
    )

    item.add_asset(
        "data",
        pystac.Asset(
            href=str(tif_path.resolve()),
            media_type=pystac.MediaType.GEOTIFF,
            roles=["data"],
            title="Analysis-ready GeoTIFF"
        )
    )

    return item

# Build a local STAC catalog
catalog = pystac.Catalog(id='internal-eo', description='Internal EO data catalog')
collection = pystac.Collection(
    id='s2-ndvi-composites',
    description='Sentinel-2 NDVI max composites',
    extent=pystac.Extent(
        spatial=pystac.SpatialExtent([[-180, -90, 180, 90]]),
        temporal=pystac.TemporalExtent([[datetime(2020, 1, 1), None]])
    )
)

for tif in Path('/data/composites/').glob('*.tif'):
    item = create_stac_item_from_geotiff(str(tif), 's2-ndvi-composites')
    collection.add_item(item)

catalog.add_child(collection)
catalog.normalize_hrefs('/output/stac_catalog/')
catalog.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
```

---

## Quick Reference: Tool Selection Matrix

| Task | Best Free Tool | Best Cloud Tool | Fastest |
|---|---|---|---|
| S1 SAR preprocessing | SNAP gpt | ASF HyP3 | HyP3 (zero local compute) |
| S2 atmospheric correction | Sen2Cor (SNAP) | openEO Copernicus | openEO |
| Global time-series analysis | GEE (geemap) | GEE | GEE |
| Object-based segmentation | OTB LSMS | — | OTB (streaming) |
| STAC data discovery | pystac-client | Planetary Computer | pystac-client |
| Lazy large-raster analysis | stackstac + Dask | — | stackstac |
| Drone photogrammetry | OpenDroneMap | WebODM cloud | ODM Docker |
| LiDAR ground classification | PDAL (csf filter) | — | PDAL pipeline |
| Forestry canopy analysis | lidR (R) | — | lidR |
| Hyperspectral unmixing | Spectral Python | — | ENVI (commercial) |
| InSAR time series | MintPy | — | MintPy + HyP3 |
| Point cloud streaming | PDAL + COPC | — | PDAL COPC reader |

---

## Related Pages

- [CLI Tools](cli-tools.md) — GDAL/OGR, rio-cogeo, geospatial format conversion
- [Cloud Platforms](cloud-platforms.md) — AWS, Azure, GCP geospatial services
- [Spatial Databases](spatial-databases.md) — PostGIS for raster and vector storage
- [Desktop GIS](desktop-gis.md) — QGIS with OTB and GRASS plugins for interactive EO analysis
