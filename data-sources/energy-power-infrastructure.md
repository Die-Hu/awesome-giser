# Energy & Power Infrastructure Data

Geospatial datasets for power plants, transmission networks, solar and wind resources, renewable energy installations, fossil fuel infrastructure, EV charging, and energy storage.

> **Quick Picks**
> - **SOTA**: [Global Energy Monitor Trackers](https://globalenergymonitor.org/) -- actively maintained trackers for coal, gas, solar, wind, oil, nuclear, battery storage with coordinates, updated biannually
> - **Free Best**: [Global Solar Atlas](https://globalsolaratlas.info/) -- World Bank-funded, 250m resolution solar irradiation data for any location on Earth
> - **Fastest Setup**: [OpenChargeMap API](https://openchargemap.org/) -- 300,000+ EV charging locations worldwide, no API key required for basic use

## Power Plant Databases

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [Global Energy Monitor Trackers](https://globalenergymonitor.org/) | Global | Suite of trackers: coal (15K+ units), gas/oil, solar (>=1MW), wind (>=10MW phases), nuclear, battery storage. Each entry has coordinates, capacity (MW), status (operating/construction/announced/retired), ownership, commissioning/retirement dates. | XLSX, CSV | Free download (CC BY 4.0 non-commercial) | Free / SOTA |
| [WRI Global Power Plant DB](https://datasets.wri.org/dataset/globalpowerplantdatabase) | Global (167 countries) | 35,000+ power plants: name, coordinates, capacity (MW), primary/secondary fuel, commissioning year, ownership, estimated generation (GWh). Also on [GEE](https://developers.google.com/earth-engine/datasets/catalog/WRI_GPPD_power_plants). | CSV (~15 MB) | Free (CC BY 4.0) | **Archived** (v1.3.0, June 2021, no updates) |
| [ENTSO-E Transparency](https://transparency.entsoe.eu/) | EU/EEA + UK, Switzerland, W. Balkans (35 countries) | Installed capacity per unit (fuel/capacity MW), actual generation output, cross-border flows, day-ahead prices, outages, load. Real-time to daily. Python: [`entsoe-py`](https://github.com/EnergieID/entsoe-py). | CSV, XML/JSON (REST API) | Free (registration; API token via email) | Free |
| [EIA Electricity Data](https://www.eia.gov/electricity/data.php) | USA (all utility-scale, 1MW+) | Form EIA-860: 10,000+ plants with capacity, fuel, location, ownership, cooling, environmental equipment. Form EIA-923: monthly generation/fuel consumption. API v2 available. | XLSX, CSV, JSON (API) | Free (API key from [eia.gov/opendata](https://www.eia.gov/opendata/)) | Free |
| [IRENA Data](https://www.irena.org/Data) | Global (150+ countries) | Country-level renewable capacity (MW) and generation (GWh) by technology (solar PV, CSP, onshore/offshore wind, hydro, bioenergy, geothermal, marine). 2000-present. | XLSX, PDF | Free (no registration) | Free |

## Transmission & Distribution Lines

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [OSM Power Infrastructure](https://wiki.openstreetmap.org/wiki/Power) | Global (1M+ km mapped) | `power=line` with voltage, `power=substation`, `power=plant`, `power=generator`. Excellent in Europe/NA; sparse in parts of Africa/Central Asia. View on [OpenInfraMap](https://openinframap.org/). | PBF, GeoJSON (Overpass Turbo) | Free (ODbL) | Free |
| [Gridfinder](https://gridfinder.rdrn.me/) | Global | Model-predicted transmission/distribution lines using nighttime lights + OSM + roads. Outputs: vectorized line network (grid.gpkg), connected-area raster (targets.tif), LV density raster (lv.tif). | GeoPackage, GeoTIFF | Free (CC BY 4.0) via [World Bank](https://datacatalog.worldbank.org/search/dataset/0038055) | Free (static, 2019) |
| [HIFLD Transmission Lines](https://catalog.data.gov/dataset/electric-power-transmission-lines) | USA | Transmission line geometries with voltage (kV), owner/operator, status, circuits. **Note**: HIFLD Open portal discontinued Sept 2025; some layers remain on data.gov. | SHP, FGDB, CSV | Free (authorized users) | Free / Restricted |

## Solar Energy Resources

| Source | Coverage | Resolution | Contents | Format | Access | Label |
|--------|----------|-----------|----------|--------|--------|-------|
| [Global Solar Atlas](https://globalsolaratlas.info/) | Global | 250m (irradiance), 1km (PVOUT) | GHI, DNI, DHI, GTI, air temp, PVOUT (kWh/kWp). Long-term averages up to 2023. Interactive PV system simulator. Also on [GEE](https://gee-community-catalog.org/projects/gsa/) and [ENERGYDATA.INFO](https://energydata.info/). | GeoTIFF, CSV (point query) | Free (CC BY 4.0) | Free / SOTA |
| [NSRDB](https://nsrdb.nrel.gov/) | Americas | 4km/30min (standard); 2km/5min (US high-res) | Half-hourly GHI, DNI, DHI + 20 met variables (temp, wind, humidity, clouds, aerosols). Physical Solar Model from satellite. 1998-present. Primary US solar development dataset. | HDF5, CSV (API) | Free (API key from [NREL](https://developer.nrel.gov/)) | Free / SOTA |
| [PVGIS](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en) | Europe, Africa, most of Asia | Point-level | PV energy output for user-defined systems (tilt, azimuth, module type, losses). TMY data for building energy sim. API: `re.jrc.ec.europa.eu/api/v5_3/`. | CSV, JSON (API), EPW | Free | Free |
| [NASA POWER](https://power.larc.nasa.gov/) | Global | 0.5x0.625 deg (~50km) | Solar irradiance, temperature, humidity, wind, precipitation from MERRA-2/CERES. 1981-present. REST API: hourly, daily, monthly, climatology endpoints. | JSON, CSV | Free (no registration) | Free |
| [CAMS Radiation](https://atmosphere.copernicus.eu/solar-radiation) | Europe, Africa, ME, SE Asia, Oceania | ~5km (gridded), 1-min (time series) | Satellite-derived GHI, BHI, DHI, BNI. Point time series 2004 to 2 days ago. | NetCDF, CSV | Free (registration at [ADS](https://ads.atmosphere.copernicus.eu/)) | Free |
| [Solargis](https://solargis.com/) | Global (60N-45S) | ~250m, 10-30min | Commercial-grade: GHI, DNI, DHI, GTI, P50/P90 yield, long-term variability. Industry standard for solar bankability. | CSV, XLSX, API | Paid (~EUR 1,800/yr for Prospect) | Commercial |

## Wind Energy Resources

| Source | Coverage | Resolution | Contents | Format | Access | Label |
|--------|----------|-----------|----------|--------|--------|-------|
| [Global Wind Atlas](https://globalwindatlas.info/) | Global | 250m | Mean wind speed, power density, capacity factors, Weibull parameters at 10/50/100/150/200m heights. ERA5 downscaled via WAsP. GWA 4 uses Copernicus DEM 30m + ESA WorldCover. | GeoTIFF, GWC (WAsP) | Free (CC BY 4.0) | Free / SOTA |
| [NREL WIND Toolkit](https://www.nrel.gov/grid/wind-toolkit.html) | CONUS (+ HI, AK, India) | 2km/hourly (full grid), 5min (120K points) | Wind speed/direction at 10-200m hub heights, temperature, pressure, humidity. WRF model. 2007-2013 (original), 2018-2020 (high-res). | HDF5, CSV (API), SAM format | Free ([API](https://developer.nrel.gov/), [AWS](https://registry.opendata.aws/nrel-pds-wtk/)) | Free |
| [ERA5 Wind Data](https://cds.climate.copernicus.eu/) | Global | 0.25 deg (~31km), hourly | 10m and 100m u/v wind components, plus 37 pressure levels. 1940-present. Input for Global Wind Atlas. On [GCS](https://cloud.google.com/storage/docs/public-datasets/era5) and [AWS](https://registry.opendata.aws/ecmwf-era5/). | NetCDF, GRIB | Free (CDS registration) | Free |

## Renewable Energy Installations

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [USWTDB](https://energy.usgs.gov/uswtdb/) | USA (75,417 turbines, v8.2) | Individual turbine: verified coordinates (10m accuracy), hub height, rotor diameter, capacity (kW), manufacturer, model, project name, year, FAA code, FIPS. Most detailed public turbine database. | SHP, GeoJSON, CSV | Free (public domain), quarterly updates | Free / SOTA |
| [OSM Renewable Energy](https://wiki.openstreetmap.org/wiki/Tag:generator:source%3Dsolar) | Global (variable) | `generator:source=solar`, `wind`, `power=plant/generator`. Capacity, operator, source type attributes. Growing coverage, strong in Europe. | PBF, GeoJSON (Overpass) | Free (ODbL) | Free |
| GEM Solar/Wind Trackers | Global | Utility-scale solar farms (>=1MW) and wind farm phases (>=10MW) with coordinates, capacity, technology, developer, status. | XLSX, CSV | Free (CC BY 4.0 non-commercial) | Free |

## Energy Consumption & Grid Data

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [EIA Electricity](https://www.eia.gov/electricity/) | USA | Monthly generation by fuel/state, retail sales/prices, plant emissions, utility service territories, 5-min grid demand/generation (2019+). API bulk updates twice daily. | XLSX, CSV, JSON (API) | Free | Free |
| [World Bank Energy](https://data.worldbank.org/topic/energy-and-mining) | Global (217 countries) | Electricity access (% pop), consumption (kWh/capita), renewable share, CO2/capita, T&D losses. 1960-present. | CSV, XLSX, JSON (API) | Free (CC BY 4.0) | Free |
| [IEA Data](https://www.iea.org/data-and-statistics) | Global (170+ countries) | Total energy supply by source, generation by fuel, CO2 emissions, energy balances. Free Energy Statistics Data Browser covers 16 topics. Full datasets and WEO projections are paid. | CSV, XLSX | Freemium (basic free; full: paid license) | Freemium |
| [PUDL](https://catalystcoop-pudl.readthedocs.io/) | USA | Catalyst Cooperative project unifying EIA + FERC + EPA data into analysis-ready database: plant generation, fuel, ownership, emissions, utility financials, service territories. | SQLite, Parquet, Python | Free (open source) | Free |

## Nuclear Power

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [IAEA PRIS](https://pris.iaea.org/) | Global (440+ operational reactors, 60+ under construction) | Per-reactor: name, coordinates, country, type (PWR/BWR/PHWR/etc.), thermal/electrical capacity, operator, construction/connection/commercial dates, monthly generation, availability, outages since 1970. | Web queries, PDF reports | Free (no bulk download API) | Free |
| [NRC Reactor Data](https://www.nrc.gov/reactors/operating) | USA (94 reactors at 54 sites) | Name, location, type, capacity, license dates, operator, containment type. [Interactive map](https://www.nrc.gov/reactors/operating/map-power-reactors), [datasets](https://www.nrc.gov/reading-rm/doc-collections/datasets/index). | Web, PDF, CSV | Free | Free |
| [World Nuclear Association](https://world-nuclear.org/) | Global | Detailed reactor specs: vendor, turbine supplier, cooling system, generation, planned builds. | Web, PDF | Free (basic); premium for full data | Freemium |

## Oil & Gas Infrastructure

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [GEM Gas/Oil Trackers](https://globalenergymonitor.org/) | Global | GGIT: gas pipelines (GIS routes), LNG terminals, carriers. GOIT: oil pipelines, refineries, storage/export terminals. Capacity, ownership, status, coordinates. | XLSX, CSV, SHP | Free (CC BY 4.0 non-commercial) | Free |
| [OGIM](https://gee-community-catalog.org/projects/ogim/) | Global | Oil & Gas Infrastructure Mapping by EDF/MethaneSAT: 6.7M features including 4.5M wells, 1.2M km pipelines, processing facilities, compressor stations, storage. | GEE Asset | Free | Free |
| [BOEM Offshore Data](https://www.data.boem.gov/) | US OCS (Gulf, Pacific, Alaska) | Platform structures (complex ID, type, water depth, coordinates, operator, status), pipeline routes (diameter, product, status). GIS at [boem.gov/maps](https://www.boem.gov/oil-gas-energy/mapping-and-data). | SHP, FGDB, CSV | Free (quarterly updates) | Free |
| [EIA Petroleum/Gas](https://www.eia.gov/petroleum/) | USA | Refinery locations/capacity, crude/product pipelines, gas pipelines/storage, processing plants. | XLSX, CSV, JSON (API) | Free | Free |

## Electric Vehicle Charging

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [OpenChargeMap](https://openchargemap.org/) | Global (100+ countries, 300K+ locations) | GPS coordinates, address, charging points, connector types (Type 1/2, CCS, CHAdeMO, Tesla), power (kW), network operator, cost, usage type (public/private), user status/photos. Est. 2011. | JSON, XML (REST API) | Free (Creative Commons), no key for basic use | Free |
| [AFDC Station Locator](https://afdc.energy.gov/stations) | USA + Canada | All alternative fuel stations: EV (L1/L2/DCFC), biodiesel, CNG, E85, H2. Address, GPS, access type, network, connectors, ports, hours, pricing. | JSON, XML, CSV (API); SHP via [data.gov](https://catalog.data.gov/dataset/alternative-fueling-station-locations-422f2) | Free (NREL API key) | Free |
| [OCPI Standard](https://evroaming.org/) | Protocol (no data) | Open Charge Point Interface v2.3.0: standardized data exchange between CPOs and eMSPs. Locations module defines station format. **EU AFIR mandates compliance since April 2025.** | JSON over REST | Free (open protocol) | Standard |

## Battery Storage & Microgrids

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [DOE GESDB](https://gesdb.sandia.gov/) | Global (2,000+ projects) | Grid-scale batteries (Li-ion, flow, lead-acid), pumped hydro, CAES, flywheels, thermal. Technology, power (MW), energy (MWh), duration, status, commissioning, developer, application type. | XLSX, JSON (web export) | Free | Free |
| [EIA Form 860M](https://www.eia.gov/electricity/data/eia860m/) | USA (utility-scale, 1MW+) | All battery storage: capacity (MW/MWh), technology, status, installation date, co-located generation, utility, state/county. 7+ GW installed by 2023. | XLSX, CSV | Free (monthly) | Free |
| GEM Battery Tracker | Global (10MW+) | Grid-scale battery projects with coordinates, capacity, technology, developer, status. | XLSX, CSV | Free (CC BY 4.0 non-commercial) | Free |
| [BloombergNEF Storage DB](https://about.bnef.com/) | Global (18,000+ projects) | Industry-leading commercial database with project specs, developer, revenue model, vendor tier. | XLSX (BNEF Terminal) | Paid (~$20K+/yr) | Commercial |

## Recommended Tools

| Tool | Purpose | URL |
|------|---------|-----|
| [pvlib](https://pvlib-python.readthedocs.io) | Solar energy modeling with NSRDB, PVGIS, TMY data | Python |
| [windpowerlib](https://windpowerlib.readthedocs.io) | Wind power modeling with ERA5, WIND Toolkit | Python |
| [entsoe-py](https://github.com/EnergieID/entsoe-py) | Python client for ENTSO-E Transparency API | Python |
| [SAM](https://sam.nrel.gov) | NREL techno-economic modeling for solar, wind, storage | Desktop |
| [PUDL](https://catalystcoop-pudl.readthedocs.io) | Unified US energy data (EIA + FERC + EPA) | Python |
| [Overpass Turbo](https://overpass-turbo.eu) | Extract OSM power infrastructure data | Web |
