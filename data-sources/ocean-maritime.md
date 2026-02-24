# Ocean & Maritime Data

Comprehensive ocean, maritime, and coastal geospatial data sources for marine research, shipping, fisheries, and coastal management.

> **Quick Picks**
> - **SOTA**: [Copernicus Marine Service (CMEMS)](https://marine.copernicus.eu) -- free global ocean physics, biogeochemistry, waves, and wind data with Python toolbox access
> - **Free Best**: [Global Fishing Watch](https://globalfishingwatch.org) -- open AIS-based fishing effort data for 190,000+ vessels since 2012
> - **Fastest Setup**: [NOAA Tides & Currents API](https://tidesandcurrents.noaa.gov) -- free REST API, no key needed, JSON/CSV output

## Oceanographic Data

Global ocean temperature, salinity, currents, and related physical/biogeochemical parameters.

| Source | Data Contents | Resolution | Format | Access | Cost | URL |
|--------|--------------|-----------|--------|--------|------|-----|
| Copernicus Marine Service (CMEMS) | Global ocean physics (temperature, salinity, currents, sea level), biogeochemistry, waves, wind from satellite + in-situ + model reanalysis. NRT and multi-year products. | 1/12 degree (~8km) | NetCDF (CF conventions) | Marine Data Store portal, `copernicusmarine` Python toolbox, FTP, OPeNDAP, WMS | Free (registration required) | [marine.copernicus.eu](https://marine.copernicus.eu) |
| World Ocean Database (WOD) | Largest public collection of uniformly formatted ocean profile data: 17M+ casts with temperature, salinity, oxygen, nutrients, chlorophyll across all ocean basins. | Point profiles, irregular | NetCDF, CSV | Web download, WODselect tool, FTP | Free | [ncei.noaa.gov](https://www.ncei.noaa.gov/products/world-ocean-database) |
| Argo Float Profiles | Temperature & salinity profiles of the upper 2,000m ocean from 3,000+ free-drifting floats worldwide. Real-time data within 12 hours; delayed-mode quality-controlled within 1 year. | Point profiles, ~3x3 degree spacing | NetCDF, CSV index | GDAC FTP (Brest & Monterey), `argopy` Python library, ERDDAP | Free | [argo.ucsd.edu](https://argo.ucsd.edu/data/) |
| HYCOM (Hybrid Coordinate Ocean Model) | Global ocean model output: temperature, salinity, currents, SSH at 40 vertical levels. Hourly to daily temporal resolution. Assimilates satellite altimetry and SST. | 1/12 degree (~8km) | NetCDF | OPeNDAP, THREDDS, FTP | Free | [hycom.org](https://www.hycom.org) |

## Sea Surface Temperature (SST)

| Source | Data Contents | Resolution | Format | Access | Cost | URL |
|--------|--------------|-----------|--------|--------|------|-----|
| GHRSST MUR | Multi-scale Ultra-high Resolution SST: daily global foundation SST from multi-sensor fusion (MODIS, AMSR2, VIIRS, in-situ). Available since 2002. Gap-free. | 1km (0.01 degree), daily | NetCDF (GHRSST GDS2) | PO.DAAC (NASA), THREDDS, OPeNDAP, `podaac-data-subscriber`, AWS | Free | [podaac.jpl.nasa.gov](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1) |
| OSTIA (UK Met Office) | Operational SST and Sea Ice Analysis: daily global gap-free SST fusing GHRSST satellite + in-situ data. Used by ECMWF as boundary condition. | 1/20 degree (~6km), daily | NetCDF (GHRSST L4) | Copernicus Marine Data Store, PO.DAAC | Free | [metoffice.gov.uk](https://ghrsst-pp.metoffice.gov.uk/ostia-website/) |
| Reynolds OISST v2.1 | NOAA Optimum Interpolation SST: blends AVHRR satellite and in-situ data. Widely used for climate monitoring and anomaly detection. | 1/4 degree (~25km), daily & monthly | NetCDF | NCEI download, THREDDS, OPeNDAP, ERDDAP | Free | [ncei.noaa.gov](https://www.ncei.noaa.gov/products/optimum-interpolation-sst) |
| MODIS SST | Sea surface temperature from MODIS sensors on Aqua & Terra satellites. Level 2 (swath), Level 3 (mapped) products. Separate day/night retrievals. | 4km and 9km | NetCDF, HDF | NASA OB.DAAC, Earthdata Search, OPeNDAP | Free | [oceancolor.gsfc.nasa.gov](https://oceancolor.gsfc.nasa.gov) |

## Ocean Color & Chlorophyll

| Source | Data Contents | Resolution | Format | Access | Cost | URL |
|--------|--------------|-----------|--------|--------|------|-----|
| NASA Ocean Color Web (OB.DAAC) | Multi-mission archive: chlorophyll-a, water-leaving radiance, PAR, diffuse attenuation from SeaWiFS, MODIS, VIIRS, PACE-OCI. Level 1-3 products since 1996. | 4km (L3), varies (L2) | NetCDF, HDF5 | Earthdata Search, OB.DAAC, OPeNDAP, CMR API | Free (Earthdata login) | [oceancolor.gsfc.nasa.gov](https://oceancolor.gsfc.nasa.gov) |
| Copernicus GlobColour | Merged multi-sensor bio-geo-chemical products: chlorophyll-a, primary production, suspended matter, turbidity. NRT + multi-year (1997-present). L3 daily & L4 monthly. | 4km | NetCDF | Copernicus Marine Data Store, WMS | Free (registration) | [marine.copernicus.eu](https://marine.copernicus.eu) |
| NASA PACE OCI | Latest-generation hyperspectral ocean color from PACE mission (launched Feb 2024). 5nm spectral resolution enabling phytoplankton community composition, aerosol characterization. | 1km | NetCDF | Earthdata Search, OB.DAAC | Free (Earthdata login) | [pace.oceansciences.org](https://pace.oceansciences.org/access_pace_data.htm) |

## Ship Tracking & AIS

| Source | Data Contents | Resolution | Format | Access | Cost | URL |
|--------|--------------|-----------|--------|--------|------|-----|
| Global Fishing Watch | AIS-based apparent fishing effort (2012-present), vessel identity for 190,000+ fishing vessels, transshipment encounters, loitering events, port visits, fishing authorization status. Gridded and vessel-level data. | 0.01 degree grid, vessel tracks | CSV, GeoTIFF, JSON (API) | Data download portal, REST API, `gfwr` R package, Python package | Free (registration) | [globalfishingwatch.org](https://globalfishingwatch.org/datasets-and-code/) |
| MarineTraffic | Real-time vessel positions, vessel details (IMO, MMSI, type, flag, dimensions), port calls, voyage data, historical tracks. Largest global AIS network with 100,000+ terrestrial + satellite AIS stations. | Real-time | JSON (API), CSV export | REST API, web platform | Paid (plans from ~$100/month; enterprise pricing varies) | [marinetraffic.com](https://www.marinetraffic.com) |
| NOAA AIS Data | Historical AIS broadcast points for U.S. EEZ and inland waterways at 1-minute sampling interval. Annual bulk files (100+ GB/year). Contains MMSI, position, speed, heading, ship type. | 1-minute intervals | CSV | AccessAIS web tool, bulk FTP download | Free | [coast.noaa.gov](https://coast.noaa.gov/digitalcoast/tools/ais.html) |
| Spire Maritime | Global satellite AIS from 100+ nanosatellites providing coverage in remote ocean areas where terrestrial AIS cannot reach. Historical data back to 2010. Dark vessel detection. | Near real-time satellite passes | JSON (API), CSV | REST API, UP42 marketplace | Paid (enterprise pricing) | [spire.com](https://spire.com/maritime/) |

## Port & Harbor Data

| Source | Data Contents | Format | Access | Cost | URL |
|--------|--------------|--------|--------|------|-----|
| World Port Index (WPI) | 3,818 world ports with 106 data fields each: coordinates, entrance conditions, channel depths, pilotage requirements, available services, shelter quality, infrastructure details. Updated monthly by NGA. | CSV, Shapefile | NGA MSI direct download, ArcGIS Online | Free | [msi.nga.mil](https://msi.nga.mil/Publications/WPI) |
| UN/LOCODE | UN Code for Trade and Transport Locations: 100,000+ locations in 249 countries including ports, airports, inland terminals. Each location gets a 5-character alphanumeric code used in global trade documentation. | CSV, XML | UNECE download page | Free | [unece.org](https://unece.org/trade/uncefact/unlocode) |
| OpenStreetMap Port Data | Community-mapped port facilities: quays, docks, berths, terminals, cranes, storage areas. Detail varies by region -- excellent in Europe, variable elsewhere. | OSM PBF, SHP, GeoJSON | Overpass API, Geofabrik | Free (ODbL) | [openstreetmap.org](https://www.openstreetmap.org) |

## Coastal & Shoreline Data

| Source | Data Contents | Format | Access | Cost | URL |
|--------|--------------|--------|--------|------|-----|
| NOAA National Shoreline | Authoritative U.S. national shoreline vectors for nautical charts, territorial limits, and EEZ definition. Multiple survey vintages available as polylines organized by region. | Shapefile | NOAA Shoreline Data Explorer | Free | [shoreline.noaa.gov](https://shoreline.noaa.gov) |
| GSHHG | Global Self-consistent, Hierarchical, High-resolution Geography Database: coastline, river, and political boundary vectors in 5 resolution levels (full to crude). Used by GMT and many mapping applications. | Shapefile, NetCDF, binary | NCEI download, GMT library | Free | [ngdc.noaa.gov](https://www.ngdc.noaa.gov/mgg/shorelines/) |
| EU Copernicus Coastal | European coastal zone monitoring: coastal erosion rates, land cover change in coastal strips, storm surge exposure, coastal flood risk assessments. Part of Copernicus Land Monitoring Service. | 10-100m | GeoTIFF, SHP, NetCDF | Copernicus Land portal, WMS/WCS | Free | [land.copernicus.eu](https://land.copernicus.eu/en/products/coastal-zones) |
| OSM Coastline | Community-mapped global coastline, frequently updated. Pre-processed derived products available as water/land polygons and coastline linestrings. | SHP | osmdata.openstreetmap.de | Free (ODbL) | [osmdata.openstreetmap.de](https://osmdata.openstreetmap.de/data/coastlines.html) |

## Tide & Current Data

| Source | Data Contents | Format | Access | Cost | URL |
|--------|--------------|--------|--------|------|-----|
| NOAA Tides & Currents | Real-time and predicted water levels, tidal currents, meteorological data, and datums for 3,000+ U.S. stations. 6-minute interval data with harmonic constituents available. REST API with no key needed. | JSON, XML, CSV, TXT | CO-OPS REST API, web download | Free | [tidesandcurrents.noaa.gov](https://tidesandcurrents.noaa.gov) |
| UHSLC | Research-quality hourly and daily tide gauge data from 500+ globally distributed stations maintained by University of Hawaii Sea Level Center. Fast-delivery (NRT) and delayed (quality-controlled) datasets. | NetCDF, CSV | FTP, ERDDAP, THREDDS | Free | [uhslc.soest.hawaii.edu](https://uhslc.soest.hawaii.edu) |
| FES2014 Global Tide Model | Finite Element Solution: global tide elevations, currents, and loading on 1/16 degree grid with 34 tidal constituents. Assimilates satellite altimetry. ~7 GB total dataset. | NetCDF | AVISO FTP (registration), `aviso-fes` Python package | Free (academic; registration required) | [aviso.altimetry.fr](https://www.aviso.altimetry.fr/en/data/products/auxiliary-products/global-tide-fes.html) |
| TPXO Global Tidal Model | Oregon State University inverse tidal model assimilating satellite altimetry. Multiple resolutions from 1/6 degree global to high-resolution local grids. 15 tidal constituents. | NetCDF, binary | OSU download (registration) | Free (academic); commercial license required for industry | [tpxo.net](https://www.tpxo.net) |

## Marine Protected Areas

| Source | Data Contents | Format | Access | Cost | URL |
|--------|--------------|--------|--------|------|-----|
| WDPA (ProtectedPlanet) | World Database on Protected Areas: the most comprehensive global database of marine and terrestrial protected areas with boundaries, designation type, management categories, governance type. Updated monthly by UNEP-WCMC. | SHP, GeoPackage, FGDB, CSV | ProtectedPlanet.net (registration) | Free | [protectedplanet.net](https://www.protectedplanet.net/en/thematic-areas/wdpa) |
| Marine Protection Atlas | Science-based assessments of global MPAs building on WDPA with standardized conservation impact categorization and zone-level protection assessments. | GDB, CSV | Data request form | Free | [mpatlas.org](https://mpatlas.org) |
| NOAA MPA Inventory | All U.S. Marine Protected Areas with boundaries, management plans, regulations, conservation objectives, and establishment authority. | SHP, GDB, KML | NOAA MPA Center download | Free | [marineprotectedareas.noaa.gov](https://marineprotectedareas.noaa.gov) |

## Sea Level Rise & Projections

| Source | Data Contents | Format | Access | Cost | URL |
|--------|--------------|--------|--------|------|-----|
| NASA Sea Level Change Portal | Satellite altimetry-derived sea level records (1992-present): global mean sea level trend (~3.4mm/yr), regional sea level change maps, ice sheet mass balance, ocean heat content. | NetCDF, CSV | Web portal, PO.DAAC download | Free | [sealevel.nasa.gov](https://sealevel.nasa.gov) |
| IPCC AR6 Sea Level Projections | Global and regional projections 2020-2150 under SSP1-2.6 through SSP5-8.5. Process-level contributions (ice sheets, thermal expansion, glaciers, land water storage). Point-level projections for any coastal location. | NetCDF, CSV | NASA Projection Tool, Zenodo | Free | [sealevel.nasa.gov](https://sealevel.nasa.gov/ipcc-ar6-sea-level-projection-tool) |
| NOAA Sea Level Rise Viewer | U.S. coastal inundation maps showing flooding scenarios from 1-10 ft above mean higher high water. Includes depth rasters, extent polygons, confidence maps, flood frequency, and socioeconomic vulnerability layers. | GeoTIFF, SHP, KMZ | Web viewer, bulk download by state | Free | [coast.noaa.gov/slr](https://coast.noaa.gov/slr/) |
| PSMSL | Permanent Service for Mean Sea Level: monthly and annual mean sea level records from 2,000+ tide gauge stations worldwide. Some records date back to 1807. The primary global sea level reference dataset. | ASCII, CSV | Web download | Free | [psmsl.org](https://www.psmsl.org) |

## Fisheries & Marine Ecology

| Source | Data Contents | Format | Access | Cost | URL |
|--------|--------------|--------|--------|------|-----|
| OBIS | Ocean Biodiversity Information System: 100M+ occurrence records of 160,000+ marine species from thousands of contributing datasets worldwide. Quality-controlled with standardized taxonomy. | CSV, DwC-A, GeoParquet (AWS) | `robis` R package, API, Mapper tool, AWS S3 | Free | [obis.org](https://obis.org) |
| RAM Legacy Stock Assessment DB | Compilation of stock assessment results for commercially exploited marine populations: time series of biomass, exploitation rate, catch for 1,000+ stocks. Essential for fisheries management research. | CSV, RData | Zenodo download, `ramlegacy` R package | Free | [ramlegacy.org](https://www.ramlegacy.org) |
| FAO FishStatJ | Global fisheries and aquaculture production statistics: capture production, aquaculture output, trade flows, and commodities by country, species, and FAO statistical area. 1950-present. | CSV, FishStatJ format | FishStatJ desktop app, web query | Free | [fao.org](https://www.fao.org/fishery/en/statistics/software/fishstatj) |
| GBIF Marine Records | Marine species occurrences within the 2.4B+ total GBIF records. Includes taxonomic backbone, georeferenced specimens, literature links, multimedia. | DwC-A, CSV | Web search, REST API, `rgbif`/`pygbif` | Free | [gbif.org](https://www.gbif.org) |

## Processing Notes

- **Copernicus Marine Toolbox**: Install `copernicusmarine` via pip for programmatic access to all CMEMS products including subsetting and format conversion.
- **ERDDAP**: Many oceanographic datasets are served through ERDDAP servers providing RESTful access in CSV, JSON, NetCDF, and PNG formats.
- **NASA Earthdata**: A single account at [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov) provides access to GHRSST, MODIS SST, Ocean Color, VIIRS, PACE, and hundreds more NASA ocean products.
- **Data Formats**: NetCDF with CF conventions is the dominant format in oceanography. Use `xarray` (Python), `ncdf4` (R), or QGIS for reading.
- **Licensing**: Most government and scientific ocean datasets are free and open. Commercial AIS/vessel tracking services require paid subscriptions for API access and historical data.
