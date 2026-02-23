# Satellite Imagery

A curated collection of satellite imagery sources ranging from free government-funded missions to commercial high-resolution providers.

> **Quick Picks**
> - **SOTA**: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) -- free Sentinel data with openEO processing and STAC API
> - **Free Best**: [Google Earth Engine](https://earthengine.google.com) -- petabytes of analysis-ready Landsat/Sentinel/MODIS in one platform
> - **Fastest Setup**: [Sentinel-2 L2A COG on AWS](https://registry.opendata.aws/sentinel-2-l2a-cogs/) -- open S3 bucket, no account needed, COG format

## Free Sources

Freely available satellite imagery products suitable for research, monitoring, and large-scale analysis.

| Source | Provider | Resolution | Revisit | Bands | Format | Access | Label |
|--------|----------|-----------|---------|-------|--------|--------|-------|
| Landsat 8/9 | USGS/NASA | 30m (MS), 15m (Pan) | 8 days (combined) | 11 | COG | [USGS EarthExplorer](https://earthexplorer.usgs.gov), [GEE](https://earthengine.google.com), [STAC on AWS](https://landsatlook.usgs.gov/stac-server) | Free / SOTA |
| Sentinel-2 L2A | ESA/Copernicus | 10m / 20m / 60m | 5 days | 13 | COG (on AWS/GCS), SAFE | [Copernicus Data Space](https://dataspace.copernicus.eu), [AWS S3](https://registry.opendata.aws/sentinel-2-l2a-cogs/), [GEE](https://earthengine.google.com) | Free / SOTA |
| Sentinel-1 GRD & SLC | ESA/Copernicus | 5x20m (IW mode) | 6 days | C-band SAR | SAFE, COG (via ASF) | [Copernicus Data Space](https://dataspace.copernicus.eu), [ASF DAAC](https://search.asf.alaska.edu) | Free / Practical |
| MODIS (Terra/Aqua) | NASA | 250m-1km | 1-2 days | 36 | HDF4, GeoTIFF | [NASA Earthdata](https://search.earthdata.nasa.gov), [GEE](https://earthengine.google.com), [LP DAAC](https://lpdaac.usgs.gov) | Free / Practical |
| VIIRS (Suomi-NPP/NOAA-20) | NASA/NOAA | 375m-750m | Daily | 22 | HDF5, NetCDF | [NASA Earthdata](https://search.earthdata.nasa.gov), [LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov) | Free / Practical |
| Landsat 1-7 Archive | USGS | 30-80m | Historical | Varies | GeoTIFF | [USGS EarthExplorer](https://earthexplorer.usgs.gov), [GEE](https://earthengine.google.com) | Free |
| Harmonized Landsat Sentinel-2 (HLS) | NASA | 30m | 2-3 days | 12+ | COG | [NASA Earthdata](https://search.earthdata.nasa.gov), [LP DAAC](https://lpdaac.usgs.gov) | Free / SOTA |
| Sentinel-3 OLCI/SLSTR | ESA/Copernicus | 300m (OLCI), 500m-1km (SLSTR) | Daily | 21/11 | NetCDF | [Copernicus Data Space](https://dataspace.copernicus.eu) | Free |
| Sentinel-5P TROPOMI | ESA/Copernicus | 3.5x7km | Daily | Atmospheric gases | NetCDF | [Copernicus Data Space](https://dataspace.copernicus.eu), [GEE](https://earthengine.google.com) | Free |
| GOES-16/17/18 | NOAA | 0.5-2km | 5-15 min | 16 | NetCDF | [NOAA CLASS](https://www.avl.class.noaa.gov), [AWS](https://registry.opendata.aws/noaa-goes/) | Free |
| ESA WorldCover 2021 | ESA | 10m | Annual (2020, 2021) | Land cover classes | COG | [ESA WorldCover](https://worldcover2021.esa.int), [GEE](https://earthengine.google.com) | Free / SOTA |
| Dynamic World | Google/WRI | 10m | Near-real-time (Sentinel-2 cadence) | 9 land cover classes | GeoTIFF | [GEE](https://earthengine.google.com), [Dynamic World App](https://dynamicworld.app) | Free / SOTA |
| Google/Microsoft Open Buildings | Google, Microsoft | Building footprints from imagery | One-time + updates | Vector (polygons) | CSV, GeoJSON, GeoParquet | [Google Open Buildings](https://sites.research.google/open-buildings/), [MS Buildings](https://github.com/microsoft/GlobalMLBuildingFootprints) | Free / SOTA |
| NASA Black Marble (VNP46A) | NASA | 500m | Daily nighttime | Night lights | HDF5 | [NASA Earthdata](https://search.earthdata.nasa.gov) | Free |

## Commercial Sources

High-resolution commercial imagery for detailed analysis. Listed for comparison purposes.

| Source | Provider | Resolution | Revisit | Bands | Access | Label |
|--------|----------|-----------|---------|-------|--------|-------|
| PlanetScope | Planet | 3m | Daily | 8 | [Planet Explorer](https://www.planet.com/explorer/) | Commercial / SOTA daily |
| SkySat | Planet | 0.5m | Daily (tasking) | 4 | [Planet Explorer](https://www.planet.com/explorer/) | Commercial |
| WorldView Legion | Maxar | 0.3m | 15x/day revisit | 8+ | [Maxar SecureWatch](https://www.maxar.com) | Commercial / SOTA resolution |
| Pleiades Neo | Airbus | 0.3m | Daily | 6 | [Airbus OneAtlas](https://oneatlas.airbus.com) | Commercial |
| SuperView NEO (SV-1/2) | SIWEI/CGSTL | 0.5m | Variable | 4 | [China Centre for Resources Satellite Data](http://www.cresda.com) | Commercial (China) |
| ICEYE | ICEYE | 0.25m (SAR spotlight) | Daily | X-band SAR | [ICEYE](https://www.iceye.com) | Commercial / SOTA SAR |
| Capella Space | Capella | 0.3m (SAR spotlight) | Sub-hourly | X-band SAR | [Capella Console](https://www.capellaspace.com) | Commercial SAR |
| Umbra | Umbra | 0.16m (SAR) | Tasking | X-band SAR | [Umbra](https://umbra.space) | Commercial / Highest-res SAR |

## Download Tools & Platforms

Tools and platforms for discovering, downloading, and accessing satellite imagery.

| Tool / Platform | Type | Supported Sources | License/Cost | Notes |
|----------------|------|-------------------|--------------|-------|
| [USGS EarthExplorer](https://earthexplorer.usgs.gov) | Web portal | Landsat, MODIS, aerial imagery | Free (account required) | Bulk download available |
| [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) | Web portal + APIs | Sentinel-1/2/3/5P, Copernicus Services | Free (account required) | Replaced old Copernicus Open Access Hub; includes openEO, STAC API | SOTA |
| [Google Earth Engine](https://earthengine.google.com) | Cloud platform | Landsat, Sentinel, MODIS, 900+ datasets | Free for research/education | Python and JS APIs, massive catalog | SOTA |
| [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com) | Cloud platform | Landsat, Sentinel, MODIS, NAIP, Copernicus DEM | Free | STAC API, Dask/xarray integration | SOTA |
| [NASA Earthdata Search](https://search.earthdata.nasa.gov) | Web portal | All NASA missions | Free (account required) | Supports CMR STAC and OPeNDAP |
| [ASF DAAC (Alaska Satellite Facility)](https://search.asf.alaska.edu) | Web portal | Sentinel-1, ALOS PALSAR, UAVSAR | Free (account required) | Best SAR data portal |
| [sentinelsat](https://github.com/sentinelsat/sentinelsat) | Python library | Sentinel missions (via Data Space) | Free / Open-source | CLI and Python API for Copernicus Data Space |
| [landsatxplore](https://github.com/yannforget/landsatxplore) | Python library | Landsat, Sentinel-2 | Free / Open-source | CLI and Python API for USGS EarthExplorer |
| [eodag](https://eodag.readthedocs.io) | Python library | 50+ providers (unified API) | Free / Apache-2.0 | Supports STAC, Copernicus, USGS, Planet, etc. | SOTA |
| [pystac-client](https://github.com/stac-utils/pystac-client) | Python library | Any STAC-compliant catalog | Free / Apache-2.0 | Standard STAC API client |
| [odc-stac](https://github.com/opendatacube/odc-stac) | Python library | Any STAC catalog | Free / Apache-2.0 | Load STAC items directly into xarray |
| [STAC Browser](https://radiantearth.github.io/stac-browser/) | Web app | Any STAC catalog | Free / Open-source | Visual browser for any STAC endpoint |
| [Radiant MLHub](https://mlhub.earth) | Cloud platform | Training data for ML/EO | Free (account required) | STAC-native ML training datasets |

## STAC Catalogs

Key STAC (SpatioTemporal Asset Catalog) endpoints for discovering satellite imagery.

| Catalog | STAC Endpoint | Collections | Notes |
|---------|--------------|-------------|-------|
| Microsoft Planetary Computer | `https://planetarycomputer.microsoft.com/api/stac/v1` | Sentinel-2, Landsat, NAIP, MODIS, Copernicus DEM | Signed URLs via `planetary-computer` Python package |
| Element 84 Earth Search | `https://earth-search.aws.element84.com/v1` | Sentinel-2 L2A COG, Landsat C2 L2, Sentinel-1 GRD | Direct S3 access on AWS, no auth needed |
| USGS LandsatLook | `https://landsatlook.usgs.gov/stac-server` | Landsat Collection 2 | Official USGS STAC endpoint |
| Copernicus Data Space | `https://catalogue.dataspace.copernicus.eu/stac` | All Sentinel missions | ESA official STAC catalog |
| NASA CMR STAC | `https://cmr.earthdata.nasa.gov/stac` | All NASA EOS missions | Federated STAC across DAACs |

## Processing Tips

Guidance on common preprocessing and analysis workflows for satellite imagery.

- **Atmospheric Correction**: Use Sen2Cor for Sentinel-2 or LaSRC for Landsat to convert to surface reflectance. Prefer L2A (already corrected) products when available.
- **Cloud Masking**: Apply QA bands or use [s2cloudless](https://github.com/sentinel-hub/sentinel2-cloud-detector) for Sentinel-2 cloud detection. For Landsat, use CFMask via the QA_PIXEL band.
- **Pansharpening**: Combine multispectral and panchromatic bands for higher effective resolution (e.g., Landsat 8/9 Band 8 at 15m).
- **Index Calculation**: NDVI, NDWI, NDBI, EVI, and other spectral indices for thematic mapping. Use GEE or rasterio/xarray for batch computation.
- **Mosaicking**: Combine multiple scenes for large-area coverage using median composites or best-pixel selection. GEE `.median()` composites are a common approach.
- **Format Considerations**: Prefer Cloud-Optimized GeoTIFF (COG) for efficient cloud-based access. Use [rio-cogeo](https://github.com/cogeotiff/rio-cogeo) to validate or convert.
- **Analysis Ready Data (ARD)**: Landsat Collection 2 and Sentinel-2 L2A are ARD. Use HLS for harmonized Landsat+Sentinel-2 time series.
- **STAC Workflow**: Use `pystac-client` to search catalogs, `odc-stac` or `stackstac` to load into xarray, process with Dask for scalability.
