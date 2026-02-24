# Agriculture & Food Security - Geospatial Data Sources

A comprehensive guide to geospatial datasets for crop monitoring, soil analysis, food supply chains, livestock, fisheries, and climate adaptation in agriculture.

---

## Quick Picks

| Use Case | Recommended Source | Why |
|---|---|---|
| Crop type mapping (US) | **USDA CropScape / CDL** | Annual crop-specific raster at 10m resolution (new in 2024), free |
| Crop type mapping (global) | **GEOGLAM-BACS + WorldCereal** | Best available global crop masks, updated annually |
| Soil properties (global) | **ISRIC SoilGrids 250m** | 250m resolution, 6 depths, free WCS/WebDAV access |
| Soil properties (US) | **USDA SSURGO / gSSURGO** | Most detailed US soil survey, free |
| Vegetation health | **MODIS NDVI/EVI** + **HLS-VI (Sentinel-2/Landsat)** | 16-day composites at 250m (MODIS) or 2-3 day at 30m (HLS) |
| Irrigation / water | **FAO AQUASTAT** + **NASA GRACE** | Global irrigated area maps + satellite groundwater monitoring |
| Agricultural statistics | **FAOSTAT** | 245+ countries, production/trade/prices from 1961 to present |
| Food prices & security | **WFP VAM Food Prices** | 1,500+ markets, monthly updates, free CSV download |
| Fishing activity | **Global Fishing Watch** | AIS-based apparent fishing effort, open data, global coverage |
| Drought monitoring | **SPEI Global Drought Monitor** + **GAEZ v4** | Drought indices + agro-ecological zone suitability |

---

## Table of Contents

1. [Crop & Agricultural Land](#1-crop--agricultural-land)
2. [Soil Data](#2-soil-data)
3. [Vegetation Indices & Productivity](#3-vegetation-indices--productivity)
4. [Irrigation & Water for Agriculture](#4-irrigation--water-for-agriculture)
5. [Pest & Disease Monitoring](#5-pest--disease-monitoring)
6. [Agricultural Statistics](#6-agricultural-statistics)
7. [Precision Agriculture Data Sources](#7-precision-agriculture-data-sources)
8. [Food Supply Chain & Markets](#8-food-supply-chain--markets)
9. [Livestock & Fisheries](#9-livestock--fisheries)
10. [Climate Adaptation for Agriculture](#10-climate-adaptation-for-agriculture)

---

## 1. Crop & Agricultural Land

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **USDA CropScape / CDL** | Annual crop-specific land cover for the contiguous US. 100+ crop types classified from satellite imagery with ground truth. New in 2024: 10m resolution (up from 30m). Includes crop mask and planting frequency layers. | GeoTIFF (raster). National 10m CDL ~9.0 GB, 30m resampled ~1.6 GB. | Free. Download via CropScape web app or direct FTP. API access via CropScapeR (R package). | [nassgeodata.gmu.edu/CropScape](https://nassgeodata.gmu.edu/CropScape/) / [nass.usda.gov/Research_and_Science/Cropland](https://www.nass.usda.gov/Research_and_Science/Cropland/) |
| **EU LPIS (Land Parcel Identification System)** | Field-level parcel boundaries for EU member states. Used for CAP (Common Agricultural Policy) subsidy management. Coverage and attributes vary by country. | Vector (Shapefile, GML). Varies by member state. | Access varies by country. Some states publish via national geo-portals (e.g., France RPG, Netherlands BRP). Not all are openly available. | Country-specific portals (e.g., [rpg.ign.fr](https://rpg.ign.fr/) for France) |
| **GEOGLAM Crop Monitor** | Monthly global crop condition assessments for major grains (wheat, maize, rice, soybean). Expert-driven consensus maps using satellite and ground data. Separate monitors for AMIS (G20) and Early Warning (food-insecure countries). | PDF reports, web maps, GeoJSON/Shapefile for crop masks | Free. Reports published monthly. Baseline crop type maps (GEOGLAM-BACS) downloadable at 0.05 degree resolution. | [cropmonitor.org](https://www.cropmonitor.org/) |
| **Global Cropland Extent (GLAD)** | Global maps of cropland extent and change (2000-2019) at 30m resolution from Landsat imagery. Shows cropland expansion patterns over two decades. | GeoTIFF (raster), 30m resolution | Free. Download from University of Maryland GLAD lab. | [glad.umd.edu/dataset/croplands](https://glad.umd.edu/dataset/croplands) |
| **ESA WorldCereal** | Global crop type maps from Sentinel-1/2 at 10m resolution. Covers temporary crops, active cropland, and specific crop types (maize, wheat, etc.). Next major release expected end of 2026. | GeoTIFF (raster), 10m resolution | Free, open. Produced by ESA. | [esa-worldcereal.org](https://esa-worldcereal.org/) |
| **Copernicus Global Land Service** | Global land cover maps at 100m resolution (annually since 2015). Includes cropland classes, fractional cover, and change layers. | GeoTIFF (raster) | Free. Registration required at Copernicus Land portal. | [land.copernicus.eu/global](https://land.copernicus.eu/global/) |

---

## 2. Soil Data

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **ISRIC SoilGrids 250m** | Global gridded soil property predictions at 250m resolution and 6 standard depths (0-5, 5-15, 15-30, 30-60, 60-100, 100-200 cm). Properties: pH, organic carbon, sand/silt/clay, bulk density, CEC, nitrogen, SOC stock. Version 2.0. | Cloud Optimized GeoTIFF, VRT (global). Access via WCS API, WebDAV, or REST API (currently paused). | Free. WCS is the recommended access method for spatial subsets. Full global maps via WebDAV. | [soilgrids.org](https://soilgrids.org/) / [isric.org/explore/soilgrids](https://isric.org/explore/soilgrids) |
| **USDA SSURGO** | Most detailed US soil survey. Soil map units with 50+ properties (texture, drainage, erosion, engineering, crop yields). Vector polygons + tabular data. Updated continuously. | Shapefile + Access DB, or gSSURGO (ESRI File Geodatabase + 10m raster). Also via Soil Data Access (SDA) API in T-SQL. | Free. Download via Web Soil Survey (by AOI), Geospatial Data Gateway (by state), or SDA API. | [nrcs.usda.gov/resources/data-and-reports/ssurgo](https://www.nrcs.usda.gov/resources/data-and-reports/soil-survey-geographic-database-ssurgo) |
| **USDA STATSGO2** | US general soil map. Coarser than SSURGO, designed for regional/national analysis. 1:250,000 scale. Same property set as SSURGO but generalized. | Shapefile + Access DB, ESRI File Geodatabase | Free. Download via Geospatial Data Gateway. | [nrcs.usda.gov/resources/data-and-reports/statsgo](https://www.nrcs.usda.gov/resources/data-and-reports/description-of-statsgo2) |
| **EU LUCAS Topsoil** | Soil samples from ~41,000 field points across the EU (2022 survey). Properties: pH, organic carbon, nitrogen, phosphorus, potassium, CaCO3, texture. Linked to land use observations at 250,000+ points. | CSV (tabular), Shapefile (point locations) | Free after registration at ESDAC (European Soil Data Centre). Topsoil data from JRC. | [esdac.jrc.ec.europa.eu/projects/lucas](https://esdac.jrc.ec.europa.eu/projects/lucas) |
| **ISRIC World Soil Information** | Curated global soil profile database (WoSIS). 200,000+ soil profiles with standardized properties. Also hosts regional soil maps and the World Reference Base for Soil Resources. | CSV, WFS, REST API | Free. Download from ISRIC data hub. | [isric.org](https://www.isric.org/) / [data.isric.org](https://data.isric.org/) |
| **OpenLandMap** | Global open soil, land, and climate layers at 250m-1km resolution. Soil organic carbon, pH, texture, depth to bedrock, land cover, climate variables. | Cloud Optimized GeoTIFF | Free. Access via Zenodo or OpenLandMap viewer. | [openlandmap.org](https://openlandmap.org/) |

---

## 3. Vegetation Indices & Productivity

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **MODIS NDVI/EVI (MOD13)** | 16-day composite vegetation indices from Terra/Aqua satellites. NDVI (continuity with AVHRR) and EVI (improved sensitivity in dense vegetation). Available at 250m, 500m, 1km, and 0.05 degree resolutions. Monthly composites also available. Record since 2000. | HDF4/HDF5, GeoTIFF. Access via NASA Earthdata, LAADS DAAC, or AppEEARS. | Free. Requires NASA Earthdata login (free registration). | [modis.gsfc.nasa.gov/data/dataprod/mod13.php](https://modis.gsfc.nasa.gov/data/dataprod/mod13.php) / [earthdata.nasa.gov](https://www.earthdata.nasa.gov/) |
| **HLS Vegetation Indices (HLS-VI)** | Harmonized Landsat 8/9 + Sentinel-2A/2B/2C vegetation indices at 30m resolution, 2-3 day revisit globally. Nine indices including NDVI, EVI, SAVI, MSAVI, NBR, NBR2, NDMI, TVI, EVI2. Released 2025. | Cloud Optimized GeoTIFF (COG) | Free. Access via NASA CMR, Earthdata Search, LP DAAC. | [earthdata.nasa.gov (HLS-VI announcement)](https://www.earthdata.nasa.gov/data/alerts-outages/harmonized-landsat-sentinel-2-vegetation-indices-data-products-released) |
| **Sentinel-2 L2A** | Multispectral imagery at 10-20m resolution, 5-day revisit. 13 bands. Users can compute custom vegetation indices (NDVI, EVI, NDRE, etc.) from surface reflectance. | JPEG2000 (original), COG (via Copernicus Data Space) | Free. Access via Copernicus Data Space Ecosystem or cloud platforms (AWS, Google Earth Engine). | [dataspace.copernicus.eu](https://dataspace.copernicus.eu/) |
| **MODIS GPP/NPP (MOD17)** | Gross Primary Productivity (GPP) and Net Primary Productivity (NPP) at 500m resolution, 8-day (GPP) and annual (NPP). Global terrestrial carbon flux estimates. Record since 2000. | HDF4, GeoTIFF | Free. NASA Earthdata / LP DAAC. | [earthdata.nasa.gov](https://www.earthdata.nasa.gov/) |
| **GIMMS NDVI 3g** | Long-term NDVI record from AVHRR sensors (1981-2015). 8km resolution, bi-monthly composites. Essential for long-term vegetation trend analysis. | NetCDF | Free. Available from NASA/GSFC. | [nex.nasa.gov](https://www.ncei.noaa.gov/) |

---

## 4. Irrigation & Water for Agriculture

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **FAO AQUASTAT** | Global water and agriculture database. Water withdrawal by sector, irrigated area statistics, dams, wastewater. Country profiles since 1960s. Global Map of Irrigation Areas (GMIA) v5: 5 arc-min raster showing % area equipped for irrigation, with groundwater/surface water breakdown. | CSV (statistics), GeoTIFF/ASCII (GMIA raster) | Free. Statistics via AQUASTAT portal. GMIA download from FAO. | [fao.org/aquastat](https://www.fao.org/aquastat/) |
| **Global Map of Irrigation Areas (GMIA v5)** | Grid showing percentage of area equipped for irrigation at 5 arc-min (~10km) resolution. Separate layers for groundwater vs. surface water irrigation. Reference year ~2005. Covers 301M ha globally. | ASCII grid, GeoTIFF | Free. Download from FAO AQUASTAT geospatial page. | [fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas](https://www.fao.org/aquastat/en/geospatial-information/global-maps-irrigated-areas/) |
| **NASA GRACE / GRACE-FO** | Satellite-based terrestrial water storage anomalies. Monthly gravity field measurements detect groundwater depletion, drought, and flooding. CONUS at 0.125 degree, global at 0.25 degree resolution. Weekly drought indicators derived from GRACE data assimilation. | NetCDF (gridded), PDF/JPG (maps), binary (raw). | Free. NASA GRACE Tellus Data Portal, PO.DAAC, and nasagrace.unl.edu for drought indicators. | [grace.jpl.nasa.gov/data/get-data](https://grace.jpl.nasa.gov/data/get-data/) / [nasagrace.unl.edu](https://nasagrace.unl.edu/) |
| **Global Surface Water Explorer (JRC)** | 38 years (1984-2021) of global surface water occurrence, change, seasonality, and transitions at 30m resolution from Landsat. Maps where water was, is, and trends. | GeoTIFF (raster), 30m | Free. Download from JRC or access via Google Earth Engine. | [global-surface-water.appspot.com](https://global-surface-water.appspot.com/) |
| **OpenET** | US-focused evapotranspiration (ET) estimates at field scale (30m) using 6 satellite-based models. Monthly and seasonal ET for water budgeting and irrigation management. | GeoTIFF, API, web platform | Free for individual field queries. Bulk/API access via paid plans. | [openetdata.org](https://openetdata.org/) |

---

## 5. Pest & Disease Monitoring

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **FAO Desert Locust Watch (DLIS)** | Desert Locust surveillance data: swarm locations, breeding areas, hopper bands, adult groups. Monthly situation bulletins and 6-week forecasts. GIS-based platform (RAMSES) used by affected countries. Locust Hub provides historical and current data on ArcGIS Hub. | PDF (bulletins), GeoJSON/Shapefile (via Locust Hub on ArcGIS), KML | Free. Monthly reports and forecasts. Locust Hub provides downloadable geospatial layers. | [fao.org/locust-watch](https://www.fao.org/locusts/en/) / [locust-hub-hqfao.hub.arcgis.com](https://locust-hub-hqfao.hub.arcgis.com/) |
| **EPPO Global Database** | 98,500+ species of agricultural/forestry interest. For each pest: geographic distribution (world map), host plants, quarantine status, trade pathways. 1,900+ regulated pest species with detailed datasheets. | Web database (searchable), maps, webservices for batch queries (registered users). DarwinCore Archive (DwC-A) for bulk. | Free. Registration encouraged for full access to webservices and contribution features. | [gd.eppo.int](https://gd.eppo.int/) |
| **CABI Invasive Species Compendium** | Encyclopedic datasheets on invasive species worldwide. Distribution maps, host range, biology, impact, management. Expert-written, peer-reviewed. Includes the Horizon Scanning Tool and Pest Risk Analysis Tool. | Web database, DarwinCore Archive (DwC-A). Maps embedded in datasheets. | Free, open access. Updated continuously with new datasheets and articles. | [cabidigitallibrary.org/journal/cabicompendium](https://www.cabidigitallibrary.org/journal/cabicompendium) / [cabi.org/isc](https://www.cabi.org/isc) |
| **USDA APHIS PPQ Pest Tracker** | US pest survey and detection data. Cooperative Agricultural Pest Survey (CAPS) results. Invasive pest distribution by state/county. | Web maps, CSV downloads | Free. Searchable by pest or state. | [pesttracker.com](https://pesttracker.com/) |
| **Plantwise Knowledge Bank** | CABI's plant health advisory platform. Pest distribution maps, diagnostic tools, pest management decision guides for developing countries. | Web platform, PDF factsheets | Free for end users. Some premium features require institutional access. | [plantwise.org](https://www.plantwise.org/) |

---

## 6. Agricultural Statistics

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **FAOSTAT** | Comprehensive food and agriculture statistics for 245+ countries/territories. Domains: production (crops, livestock), trade, food balances, prices, land use, emissions, fertilizers, pesticides, SDG indicators. Time series from 1961 to latest available year. | CSV, Excel (via bulk download or custom query). API available. | Free. Browse and download at fao.org/faostat. Bulk downloads organized by domain. | [fao.org/faostat](https://www.fao.org/faostat/) |
| **USDA NASS Quick Stats** | US agricultural statistics: crop acreage, yields, production, prices, livestock inventory, farm economics. County, state, and national level. Weekly to annual frequency. | CSV, JSON (via API). Compressed .gz files for bulk. | Free. Query via Quick Stats tool, API, or bulk download. | [nass.usda.gov/Quick_Stats](https://www.nass.usda.gov/Quick_Stats/) |
| **Eurostat Agriculture** | EU agricultural statistics: crop production, livestock, farm structure, prices, trade, organic farming. National and regional (NUTS2) level. | CSV, TSV, JSON, SDMX via Eurostat database. Excel for pre-built tables. | Free. Bulk download or query builder at ec.europa.eu/eurostat. | [ec.europa.eu/eurostat/web/agriculture](https://ec.europa.eu/eurostat/web/agriculture) |
| **World Bank Agricultural Data** | Cross-country agricultural indicators: cereal yield, arable land, fertilizer use, agricultural value added, rural population. Part of World Development Indicators. | CSV, Excel, API (World Bank Data API) | Free. | [data.worldbank.org/topic/agriculture-and-rural-development](https://data.worldbank.org/topic/agriculture-and-rural-development) |
| **USDA FAS PSD Online** | USDA Foreign Agricultural Service Production, Supply, and Distribution data. Crop-specific supply and demand estimates for all major producing/consuming countries. | CSV, Excel | Free. Monthly updates with WASDE reports. | [apps.fas.usda.gov/psdonline](https://apps.fas.usda.gov/psdonline/) |

---

## 7. Precision Agriculture Data Sources

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **Fiboa (Field Boundaries for Agriculture)** | Open specification and data collection for agricultural field boundaries. Community-driven, standardized format. Growing number of contributed datasets globally. | GeoParquet, GeoJSON (fiboa spec) | Free, open. Datasets contributed by various organizations. | [fiboa.org](https://fiboa.org/) / [github.com/fiboa](https://github.com/fiboa) |
| **France RPG (Registre Parcellaire Graphique)** | French agricultural parcel boundaries with declared crop types. 6M+ parcels annually. One of the best open LPIS datasets. | Shapefile, GeoJSON | Free. Annual releases from IGN/ASP. | [rpg.ign.fr](https://rpg.ign.fr/) |
| **Netherlands BRP (Basisregistratie Percelen)** | Dutch agricultural parcel boundaries with crop declarations. ~700K parcels. High quality, annual updates. | GeoPackage, Shapefile | Free. Download from PDOK. | [pdok.nl](https://www.pdok.nl/) |
| **FBIS-22M (Field Boundary Instance Segmentation)** | 22 million field boundaries derived from multi-source satellite imagery (Sentinel-2, Planet, Maxar). Training dataset for ML models. | GeoJSON, raster labels | Research dataset. Check publication for access terms. | [arxiv.org/html/2504.02534](https://arxiv.org/html/2504.02534v1) |
| **EOS Crop Monitoring Platform** | Satellite-based crop monitoring with NDVI, field boundary detection, weather data, scouting tools. Covers 60M+ fields. | Web platform, API, GeoTIFF exports | Freemium. Free tier for limited fields. Paid from ~$1/field/season for commercial use. | [eos.com/products/crop-monitoring](https://eos.com/products/crop-monitoring/) |
| **Google Earth Engine (for ag analysis)** | Cloud platform with petabytes of satellite imagery (Sentinel, Landsat, MODIS) and geospatial datasets. Python/JavaScript API for custom crop analysis, yield estimation, change detection. | Analysis platform (raster/vector), export to GeoTIFF/CSV | Free for research and education. Commercial via Google Cloud. | [earthengine.google.com](https://earthengine.google.com/) |
| **Farmonaut / OneSoil** | Commercial precision ag platforms offering crop monitoring, field boundaries, crop type detection from Sentinel-2. OneSoil provides free field boundary maps for 60+ countries. | Web platform, API | OneSoil: free field viewer. Farmonaut: freemium with paid tiers. | [onesoil.ai](https://onesoil.ai/) / [farmonaut.com](https://farmonaut.com/) |

---

## 8. Food Supply Chain & Markets

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **WFP Food Price Data (VAM)** | Monthly food prices from 1,500+ markets in 90+ countries. Staple commodities (cereals, pulses, oil, sugar). Price alerts and forecasts. The longest-running global food price monitoring system. | CSV (via HDX or DataViz export). JSON via API. Interactive dashboards. | Free. Bulk download from Humanitarian Data Exchange (HDX). Visualization at dataviz.vam.wfp.org. | [dataviz.vam.wfp.org/economic/prices](https://dataviz.vam.wfp.org/economic/prices) / [data.humdata.org/dataset/wfp-food-prices](https://data.humdata.org/dataset/wfp-food-prices) |
| **USDA Market News** | Daily/weekly US agricultural market prices: livestock, grain, dairy, fruit, vegetables, cotton. Terminal market reports, national summaries. | Text reports, CSV, API (MARS) | Free. Updated daily. | [marketnews.usda.gov](https://marketnews.usda.gov/) |
| **FAO GIEWS (Food Price Monitoring)** | Global Information and Early Warning System. Food price data, supply/demand briefs, crop prospects, country food situation assessments. | PDF reports, CSV (food prices via FPMA tool) | Free. Monthly FPMA bulletins and country briefs. | [fao.org/giews](https://www.fao.org/giews/) |
| **USDA GATS (Global Agricultural Trade System)** | US agricultural import/export data by commodity, country, and time period. Harmonized System (HS) codes. | CSV, Excel via query builder | Free. | [apps.fas.usda.gov/gats](https://apps.fas.usda.gov/gats/) |
| **CFS Commodity Flow Survey (US)** | US domestic freight shipments by commodity type, mode, origin/destination. Includes agricultural commodities. Conducted every 5 years by Census Bureau + BTS. | CSV, Shapefile (flow maps) | Free. Most recent: 2022 CFS. | [census.gov/programs-surveys/cfs.html](https://www.census.gov/programs-surveys/cfs.html) |
| **FEWS NET (Famine Early Warning)** | USAID-funded food security monitoring for 30+ vulnerable countries. Integrated food security phase classification (IPC), price bulletins, seasonal monitors, livelihood zone maps. | PDF reports, Shapefile (livelihood zones, IPC maps), CSV (prices) | Free. | [fews.net](https://fews.net/) |

---

## 9. Livestock & Fisheries

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **FAO Gridded Livestock of the World (GLW)** | Global distribution and density of livestock species at ~10km (0.083 degree) resolution. Version 4 (reference year 2015): cattle, sheep, goats, buffaloes, horses, pigs, chickens, ducks. Peer-reviewed, modeled from census data and environmental predictors. | GeoTIFF (raster), ~10km resolution | Free. Download from FAO catalog or Harvard Dataverse. | [fao.org/livestock-systems/global-distributions](https://www.fao.org/livestock-systems/global-distributions/en/) / [data.apps.fao.org/catalog/dataset/glw](https://data.apps.fao.org/catalog/dataset/glw) |
| **Global Fishing Watch** | AIS-based apparent fishing effort globally. Vessel tracking, fishing activity hotspots, marine infrastructure detection. Version 3 of downloadable dataset (March 2025). API, R package (gfwr), and Python support. | CSV, GeoJSON, MVT (tiles), GeoParquet. API returns JSON. Data Download Portal for bulk. | Free (open data, CC-BY-4.0). API key required. R package `gfwr` for programmatic access. | [globalfishingwatch.org/datasets-and-code](https://globalfishingwatch.org/datasets-and-code/) |
| **FAO Fisheries & Aquaculture Statistics** | Global capture production and aquaculture production by species, country, and area. Trade in fish products. Fleet statistics. Time series from 1950. | CSV, Excel, FishStatJ software | Free. Part of FAO statistics suite. | [fao.org/fishery/en/statistics](https://www.fao.org/fishery/en/statistics) |
| **GBIF (Global Biodiversity Information Facility)** | Species occurrence records including livestock-related wildlife, disease vectors, and aquatic species. 2.5B+ occurrence records. | DarwinCore Archive (DwC-A), CSV | Free (CC0 to CC-BY-NC). | [gbif.org](https://www.gbif.org/) |
| **RAM Legacy Stock Assessment Database** | Fish stock assessment results for 1,000+ stocks globally. Biomass, fishing mortality, reference points. Key for fisheries sustainability analysis. | CSV, R package (ramlegacy) | Free, open. | [ramlegacy.org](https://www.ramlegacy.org/) |
| **Copernicus Marine Service (CMEMS)** | Ocean variables relevant to fisheries: sea surface temperature, chlorophyll, ocean currents, salinity, primary productivity. Global and regional products. | NetCDF, via API (Copernicus Marine Toolbox) | Free (registration required). | [marine.copernicus.eu](https://marine.copernicus.eu/) |

---

## 10. Climate Adaptation for Agriculture

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **Global SPEI Database** | Standardised Precipitation-Evapotranspiration Index at global scale. Multi-scalar drought index (1-48 month timescales). 0.5 degree resolution, monthly, from 1955 to near-present. | NetCDF | Free. Download from CSIC (Spain). | [spei.csic.es/database.html](https://spei.csic.es/database.html) |
| **US Drought Monitor** | Weekly US drought classification map (D0-D4 intensity). Combines multiple indices and expert input. Vector polygons and tabular data. | Shapefile (weekly polygons), CSV, KML, GeoJSON | Free. Updated every Thursday. | [droughtmonitor.unl.edu](https://droughtmonitor.unl.edu/) |
| **FAO GAEZ v4** | Global Agro-Ecological Zones: land suitability and attainable yields for 49+ crops under current and future climates. Six themes: land/water resources, agro-climatic resources, potential yield, suitability, actual yields, yield gaps. ~10km resolution. | GeoTIFF (raster), CSV (tabular) | Free. Interactive data portal with thousands of downloadable layers. | [gaez.fao.org](https://gaez.fao.org/) |
| **NASA GRACE Drought Indicators** | Weekly groundwater and soil moisture drought indicators derived from GRACE satellite gravity data assimilated into land surface models. CONUS (0.125 deg) and global (0.25 deg). | GeoTIFF, binary, PDF/JPG maps | Free. | [nasagrace.unl.edu](https://nasagrace.unl.edu/) |
| **CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)** | Quasi-global rainfall estimates at 0.05 degree (~5km) resolution, daily and pentadal, from 1981 to near real-time. Blends satellite and station data. Essential for agricultural drought monitoring in data-sparse regions. | GeoTIFF, NetCDF, BIL | Free. Download from UCSB Climate Hazards Center or Google Earth Engine. | [chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps) |
| **CRU TS (Climatic Research Unit Time-Series)** | Monthly gridded climate data (temperature, precipitation, vapor pressure, cloud cover, etc.) at 0.5 degree resolution. 1901 to present. Station-based interpolation. | NetCDF | Free for academic/research use. Registration required. | [crudata.uea.ac.uk/cru/data/hrg](https://crudata.uea.ac.uk/cru/data/hrg/) |
| **AgERA5** | Agro-meteorological indicators derived from ERA5 reanalysis. Daily data at 0.1 degree (~10km) resolution from 1979. Variables: temperature (min/max/mean), precipitation, solar radiation, wind, humidity, reference ET. Specifically designed for agricultural applications. | NetCDF via Copernicus Climate Data Store (CDS) | Free. Download via CDS API or web interface. | [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/) |
| **Growing Degree Days / Growing Season Data** | Various sources provide GDD accumulations and growing season length. USDA provides GDD via PRISM; NASA provides phenology data via MODIS Land Cover Dynamics (MCD12Q2). | GeoTIFF, NetCDF, CSV | Free (PRISM, MODIS). | [prism.oregonstate.edu](https://prism.oregonstate.edu/) / [earthdata.nasa.gov](https://www.earthdata.nasa.gov/) |

---

## Notes & Tips

- **Google Earth Engine (GEE)** is the single most powerful tool for agricultural remote sensing analysis. It provides direct access to MODIS, Sentinel-2, Landsat, CHIRPS, SoilGrids, and hundreds of other datasets in a cloud computing environment. Free for research.
- **USDA CropScape** upgraded to 10m resolution starting with the 2024 CDL -- a major improvement over the previous 30m. Historical data (2008-2023) remains at 30m.
- **SoilGrids REST API** is currently paused (as of early 2026). Use the WCS endpoint or WebDAV for reliable access to SoilGrids data.
- **FAOSTAT** is the gold standard for cross-country agricultural statistics but typically has a 1-2 year lag for the most recent data.
- **WFP food price data** is invaluable for food security analysis -- the Humanitarian Data Exchange (HDX) provides the easiest bulk download option.
- **HIFLD data archive note**: For US agricultural infrastructure (storage facilities, processing plants), check the DataLumos archive since the HIFLD portal went offline in August 2025.
- **Open field boundaries** are becoming increasingly available. The fiboa initiative is standardizing formats, and countries like France (RPG) and the Netherlands (BRP) lead in publishing open parcel data.

---

*Last updated: February 2026*
