# Elevation & Terrain Data

Global and regional elevation datasets including digital elevation models (DEMs), bathymetric data, and LiDAR point clouds.

> **Quick Picks**
> - **SOTA**: [Copernicus GLO-30 DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) -- global 30m DEM from TanDEM-X, free, COG format on AWS
> - **Free Best**: [FABDEM](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn) -- forests and buildings removed from Copernicus DEM, ideal for hydrology
> - **Fastest Setup**: [Mapzen Terrain Tiles on AWS](https://registry.opendata.aws/terrain-tiles/) -- pre-tiled global elevation, ready for web visualization

## Global DEMs

Freely available global digital elevation models for terrain analysis, hydrological modeling, and visualization.

### SRTM

The Shuttle Radar Topography Mission provides near-global elevation coverage from 2000.

| Dataset | Resolution | Coverage | Format | Access | Label |
|---------|-----------|----------|--------|--------|-------|
| SRTM 1 Arc-Second (30m) | ~30m | 60N - 56S | GeoTIFF, HGT | [USGS EarthExplorer](https://earthexplorer.usgs.gov), [GEE](https://earthengine.google.com) | Free |
| SRTM 3 Arc-Second (90m) | ~90m | 60N - 56S | GeoTIFF, HGT | [USGS EarthExplorer](https://earthexplorer.usgs.gov) | Free |
| SRTM Void-Filled (v3) | ~30m | 60N - 56S | GeoTIFF | [USGS EarthExplorer](https://earthexplorer.usgs.gov) | Free |
| SRTM via AWS | ~30m | 60N - 56S | GeoTIFF (COG) | [AWS Open Data](https://registry.opendata.aws/terrain-tiles/) | Free |

### ASTER GDEM

Advanced Spaceborne Thermal Emission and Reflection Radiometer Global DEM.

| Dataset | Resolution | Coverage | Format | Access | Label |
|---------|-----------|----------|--------|--------|-------|
| ASTER GDEM v3 | ~30m | 83N - 83S | GeoTIFF | [NASA Earthdata](https://search.earthdata.nasa.gov), [GEE](https://earthengine.google.com) | Free |
| ASTER GDEM via LP DAAC | ~30m | 83N - 83S | GeoTIFF, COG | [LP DAAC](https://lpdaac.usgs.gov) | Free |

### Copernicus DEM

The Copernicus Digital Elevation Model derived from WorldDEM/TanDEM-X data. Currently the best free global DEM.

| Dataset | Resolution | Coverage | Format | Access | Label |
|---------|-----------|----------|--------|--------|-------|
| GLO-30 | ~30m | Global | COG | [AWS Open Data](https://registry.opendata.aws/copernicus-dem/), [Copernicus PANDA](https://panda.copernicus.eu), [Planetary Computer](https://planetarycomputer.microsoft.com) | Free / SOTA |
| GLO-90 | ~90m | Global | COG | [AWS Open Data](https://registry.opendata.aws/copernicus-dem/), [Copernicus PANDA](https://panda.copernicus.eu) | Free |
| EEA-10 | ~10m | Europe | COG | [Copernicus Data Space](https://dataspace.copernicus.eu) | Free (Europe only) |

### ALOS World 3D

Japan Aerospace Exploration Agency's global elevation dataset.

| Dataset | Resolution | Coverage | Format | Access | Label |
|---------|-----------|----------|--------|--------|-------|
| AW3D30 v3.2 | ~30m | Global | GeoTIFF | [JAXA ALOS Research](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm), [GEE](https://earthengine.google.com) | Free |
| AW3D5 (commercial) | ~5m | Global | GeoTIFF | [JAXA](https://www.aw3d.jp/en/) | Commercial |

### Other Global DEMs

| Dataset | Resolution | Coverage | Format | Access | Label |
|---------|-----------|----------|--------|--------|-------|
| NASADEM | ~30m | 60N - 56S | GeoTIFF | [NASA Earthdata](https://search.earthdata.nasa.gov), [GEE](https://earthengine.google.com) | Free -- reprocessed SRTM |
| FABDEM | ~30m | Global | GeoTIFF | [University of Bristol](https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn) | Free / SOTA -- buildings and forests removed |
| MERIT DEM | ~90m | 60N - 60S | GeoTIFF | [U-Tokyo Hydro Lab](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_DEM/) | Free -- error-reduced |
| MERIT Hydro | ~90m | 60N - 60S | GeoTIFF | [U-Tokyo Hydro Lab](http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/) | Free -- hydrologically conditioned |
| Mapzen Terrain Tiles | ~30m (varies) | Global | Terrarium/Normal PNG, Quantized Mesh | [AWS Open Data](https://registry.opendata.aws/terrain-tiles/) | Free -- web-optimized |
| TanDEM-X 90m | ~90m | Global | GeoTIFF | [DLR](https://tandemx-90m.dlr.de/) | Free (registration) |

## Bathymetry

Seafloor and underwater terrain elevation datasets.

| Dataset | Resolution | Coverage | Format | Access | Label |
|---------|-----------|----------|--------|--------|-------|
| GEBCO 2023 | 15 arc-second (~450m) | Global ocean | NetCDF, GeoTIFF, COG | [gebco.net](https://www.gebco.net) | Free / SOTA |
| ETOPO 2022 (Global Relief Model) | 15/60 arc-second | Global (land + ocean merged) | NetCDF, GeoTIFF, COG | [NOAA NCEI](https://www.ncei.noaa.gov/products/etopo-global-relief-model) | Free / SOTA |
| EMODnet Bathymetry | ~115m | European seas | GeoTIFF, ASCII, WMS | [emodnet.ec.europa.eu](https://emodnet.ec.europa.eu/en/bathymetry) | Free |
| SRTM15+ v2.5 | 15 arc-second | Global (land + ocean) | NetCDF | [topex.ucsd.edu](https://topex.ucsd.edu/marine_grav/srtm15.html) | Free |
| NOAA Coastal Relief Model | 3 arc-second (~90m) | US coastal waters | NetCDF, ASCII | [NOAA NCEI](https://www.ncei.noaa.gov/products/coastal-relief-model) | Free |

## LiDAR Sources

Airborne and terrestrial LiDAR point cloud datasets for high-resolution terrain mapping.

| Dataset | Resolution | Coverage | Format | Access | Label |
|---------|-----------|----------|--------|--------|-------|
| USGS 3DEP | 1m / 0.5m (QL1/QL2) | USA (~85% complete) | LAZ, LAS, COG (derived) | [USGS National Map](https://apps.nationalmap.gov/lidar-explorer/), [AWS](https://registry.opendata.aws/usgs-lidar/) | Free / SOTA |
| AHN (Actueel Hoogtebestand NL) v4 | 0.5m | Netherlands | LAZ, GeoTIFF | [ahn.nl](https://www.ahn.nl) | Free |
| Environment Agency LiDAR | 0.25m-2m | England | ASC, LAZ, GeoTIFF | [data.gov.uk](https://environment.data.gov.uk/DefraDataDownload/?Mode=survey) | Free |
| OpenTopography | Varies (0.25m-5m) | Global (select areas, 300+ datasets) | LAZ, LAS | [opentopography.org](https://opentopography.org) | Free (account required) |
| ICESat-2 ATL08 (Land and Vegetation) | Footprint-level | Global | HDF5 | [NASA Earthdata](https://search.earthdata.nasa.gov), [NSIDC](https://nsidc.org/data/atl08) | Free |
| GEDI L2A/L2B (Forest Structure) | 25m footprint | 51.5N - 51.5S | HDF5 | [NASA Earthdata](https://search.earthdata.nasa.gov), [GEE](https://earthengine.google.com) | Free / SOTA (forest canopy height) |
| EU-DEM v1.1 | 25m | Europe (EEA39) | GeoTIFF | [Copernicus Land Monitoring](https://land.copernicus.eu/imagery-in-situ/eu-dem) | Free |
| Geoscience Australia LiDAR | 1-5m | Australia (select areas) | LAZ | [elvis.ga.gov.au](https://elevation.fsdf.org.au/) | Free |
