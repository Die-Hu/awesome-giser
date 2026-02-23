# Climate & Weather Data

Atmospheric reanalysis products, station observations, weather forecasts, and long-term climate projections for geospatial analysis.

> **Quick Picks**
> - **SOTA**: [ERA5 via Copernicus CDS](https://cds.climate.copernicus.eu) -- the gold standard reanalysis, hourly global data from 1940
> - **Free Best**: [CHIRPS](https://www.chc.ucsb.edu/data/chirps) -- high-resolution precipitation data from 1981, ideal for drought monitoring
> - **Fastest Setup**: [Open-Meteo API](https://open-meteo.com) -- free weather API, no API key needed, JSON responses

## Reanalysis Data

Gridded atmospheric datasets combining model output with observations to produce spatially and temporally consistent records.

### ERA5

ECMWF's fifth-generation global reanalysis covering 1940 to present. The most widely used reanalysis product globally.

| Dataset | Resolution | Coverage | Temporal | Format | Access | Label |
|---------|-----------|----------|----------|--------|--------|-------|
| ERA5 Hourly (Pressure Levels) | 0.25 deg (~31km), 37 levels | Global | 1940-present, hourly | NetCDF, GRIB | [Copernicus CDS](https://cds.climate.copernicus.eu) | Free / SOTA |
| ERA5 Hourly (Single Levels) | 0.25 deg (~31km) | Global | 1940-present, hourly | NetCDF, GRIB | [Copernicus CDS](https://cds.climate.copernicus.eu) | Free / SOTA |
| ERA5-Land | 0.1 deg (~9km) | Global land | 1950-present, hourly | NetCDF, GRIB | [Copernicus CDS](https://cds.climate.copernicus.eu) | Free / SOTA |
| ERA5 Monthly Averages | 0.25 deg (~31km) | Global | 1940-present, monthly | NetCDF, GRIB | [Copernicus CDS](https://cds.climate.copernicus.eu) | Free |
| ERA5 on Google Cloud | 0.25 deg | Global | 1940-present | Zarr (via BigQuery) | [Google Cloud Public Datasets](https://cloud.google.com/storage/docs/public-datasets/era5) | Free |
| ERA5 on AWS | 0.25 deg | Global | 1979-present | NetCDF | [AWS Open Data](https://registry.opendata.aws/ecmwf-era5/) | Free |

### MERRA-2

NASA's Modern-Era Retrospective Analysis for Research and Applications, Version 2.

| Dataset | Resolution | Coverage | Temporal | Format | Access | Label |
|---------|-----------|----------|----------|--------|--------|-------|
| MERRA-2 (Full Collection) | 0.5 x 0.625 deg (~50km) | Global | 1980-present, hourly/monthly | NetCDF4 | [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets?project=MERRA-2) | Free |
| MERRA-2 via GEE | 0.5 x 0.625 deg | Global | 1980-present | ImageCollection | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/NASA_GSFC_MERRA) | Free |

### Other Reanalysis Products

| Dataset | Resolution | Coverage | Temporal | Format | Access | Label |
|---------|-----------|----------|----------|--------|--------|-------|
| JRA-55 | ~55km (TL319) | Global | 1958-present | GRIB, NetCDF | [JMA](https://jra.kishou.go.jp/JRA-55/index_en.html) | Free |
| NCEP/NCAR Reanalysis 1 | 2.5 deg (~278km) | Global | 1948-present | NetCDF | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html) | Free |
| NCEP-DOE Reanalysis 2 | 2.5 deg (~278km) | Global | 1979-present | NetCDF | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html) | Free |
| CFS Reanalysis (CFSR) | ~38km (T382) | Global | 1979-2011 | GRIB2 | [NOAA NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/climate-forecast-system) | Free |
| 20th Century Reanalysis v3 | ~75km | Global | 1836-2015 | NetCDF | [NOAA PSL](https://psl.noaa.gov/data/gridded/data.20thC_ReanV3.html) | Free -- longest record |

## Station Data

Point-based observations from weather stations and monitoring networks.

| Dataset | Coverage | Variables | Temporal | Format | Access | Label |
|---------|----------|-----------|----------|--------|--------|-------|
| GHCN-Daily | Global (~100k stations) | Precip, Temp (max/min), Snow | 1763-present, daily | CSV | [NOAA NCEI](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) | Free / Practical |
| GSOD (Global Summary of the Day) | Global (~9000 stations) | Temp, Precip, Wind, Pressure, Visibility | 1929-present, daily | CSV | [NOAA NCEI](https://www.ncei.noaa.gov/access/search/data-search/global-summary-of-the-day) | Free |
| ISD (Integrated Surface Database) | Global (~35k stations) | 100+ variables | 1901-present, hourly/sub-hourly | Fixed-width, CSV | [NOAA NCEI](https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database) | Free |
| CRU TS v4.07 | Global (gridded from stations) | Temp, Precip, Cloud cover, DTR, Vapor pressure | 1901-present, monthly | NetCDF | [UEA CRU](https://crudata.uea.ac.uk/cru/data/hrg/) | Free |
| WorldClim v2.1 | Global (interpolated) | Temp (min/max/mean), Precip | Climatologies (1970-2000) | GeoTIFF | [worldclim.org](https://www.worldclim.org) | Free / Practical |
| CHIRPS v2.0 | 50N-50S | Precipitation | 1981-present, daily/pentad/monthly | GeoTIFF, NetCDF, BIL | [CHG UCSB](https://www.chc.ucsb.edu/data/chirps) | Free / SOTA |
| GPM IMERG | 60N-60S | Precipitation | 2000-present, 30min/daily/monthly | HDF5, NetCDF | [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets?keywords=IMERG) | Free / SOTA |
| TerraClimate | Global (~4km) | Temp, Precip, ET, Soil moisture, 14 variables | 1958-present, monthly | NetCDF | [climatologylab.org](https://www.climatologylab.org/terraclimate.html), [GEE](https://earthengine.google.com) | Free |

## Forecasts

Numerical weather prediction and short-term forecast datasets.

| Dataset | Resolution | Coverage | Lead Time | Format | Access | Label |
|---------|-----------|----------|-----------|--------|--------|-------|
| GFS (Global Forecast System) | 0.25 deg (~28km) | Global | 16 days | GRIB2 | [NOAA NOMADS](https://nomads.ncep.noaa.gov), [AWS](https://registry.opendata.aws/noaa-gfs-bdp-pds/) | Free / Practical |
| ECMWF IFS (HRES) | ~9km (Tco1279) | Global | 10 days (15 days ensemble) | GRIB | [ECMWF](https://www.ecmwf.int) (registered users) | Free (basic) / SOTA |
| ECMWF Open Data | 0.25 deg | Global | 10 days | GRIB2 | [ECMWF Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data) | Free |
| ICON (Icosahedral Nonhydrostatic) | ~13km (global), ~6.5km (EU) | Global / Europe | 7.5 days | GRIB2 | [DWD Open Data](https://opendata.dwd.de/weather/nwp/) | Free |
| NAM (North American Mesoscale) | 12km / 3km (CONUS nest) | North America | 84 hours | GRIB2 | [NOAA NOMADS](https://nomads.ncep.noaa.gov) | Free |
| HRRR (High-Resolution Rapid Refresh) | 3km | CONUS | 48 hours (updated hourly) | GRIB2 | [AWS](https://registry.opendata.aws/noaa-hrrr-pds/), [Google Cloud](https://console.cloud.google.com/marketplace/details/noaa-public/hrrr) | Free / SOTA (CONUS) |
| Open-Meteo API | Varies (best available NWP) | Global | 7-16 days | JSON | [open-meteo.com](https://open-meteo.com) | Free (no API key) |

## Climate Projections

Long-term climate model outputs for future scenario analysis.

| Dataset | Resolution | Scenarios | Temporal | Format | Access | Label |
|---------|-----------|-----------|----------|--------|--------|-------|
| CMIP6 | Varies (50-250km, 30+ models) | SSP1-2.6 to SSP5-8.5 | 2015-2100+ | NetCDF | [ESGF](https://esgf-node.llnl.gov/projects/cmip6/), [GEE](https://earthengine.google.com) | Free / SOTA |
| CORDEX (Regional Downscaling) | ~12-50km (downscaled) | RCP/SSP | 2006-2100 | NetCDF | [ESGF](https://esgf-node.llnl.gov/projects/cordex/), [Copernicus CDS](https://cds.climate.copernicus.eu) | Free |
| NEX-GDDP-CMIP6 | 0.25 deg (~28km, bias-corrected) | SSP2-4.5, SSP5-8.5 | 1950-2100, daily | NetCDF | [NASA NEX](https://www.nccs.nasa.gov/services/data-collections/land-based-products/nex-gddp-cmip6), [GEE](https://earthengine.google.com) | Free / SOTA (bias-corrected) |
| WorldClim Future Bioclim | ~1km (downscaled) | SSP1-2.6 to SSP5-8.5 | 2021-2100 (4 periods) | GeoTIFF | [worldclim.org](https://www.worldclim.org) | Free / Practical |
| CHELSA v2.1 | ~1km (downscaled) | SSP1-2.6 to SSP5-8.5 | 2011-2100 | GeoTIFF | [chelsa-climate.org](https://chelsa-climate.org) | Free / SOTA (high-res) |
| ISIMIP | Varies (0.5 deg common) | SSP1-2.6 to SSP5-8.5 | 1850-2100 | NetCDF | [isimip.org](https://www.isimip.org/outputdata/) | Free -- multi-sector impacts |

## Processing Tools

| Tool | Type | Purpose | License | URL |
|------|------|---------|---------|-----|
| [xarray](https://xarray.dev) | Python library | Multi-dimensional array processing for NetCDF/GRIB/Zarr | Apache-2.0 | [docs.xarray.dev](https://docs.xarray.dev) |
| [cfgrib](https://github.com/ecmwf/cfgrib) | Python library | Read GRIB files into xarray | Apache-2.0 | [PyPI](https://pypi.org/project/cfgrib/) |
| [CDO (Climate Data Operators)](https://code.mpimet.mpg.de/projects/cdo) | CLI tool | Process climate/NWP model output | GPL-2.0 | [MPI-M](https://code.mpimet.mpg.de/projects/cdo) |
| [NCO (NetCDF Operators)](https://nco.sourceforge.net) | CLI tool | Manipulate NetCDF files (subset, merge, compute) | GPL-3.0 | [nco.sourceforge.net](https://nco.sourceforge.net) |
| [Copernicus CDS API](https://cds.climate.copernicus.eu/api-how-to) | Python API | Download ERA5 and other CDS datasets | Free | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu) |
