# Healthcare & Public Health Data

Geospatial datasets for healthcare facility mapping, disease surveillance, environmental health, health access modeling, and public health research.

> **Quick Picks**
> - **SOTA**: [IHME GBD Results](https://vizhub.healthdata.org/gbd-results/) -- mortality and morbidity estimates for 292 causes across 204 countries
> - **Free Best**: [Healthsites.io](https://healthsites.io) -- 175K+ geolocated health facilities worldwide, open data, API access
> - **Fastest Setup**: [OpenAQ API](https://openaq.org) -- real-time air quality from 30K+ stations, no API key needed for basic use

## Healthcare Facility Locations

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [Healthsites.io](https://healthsites.io) | Global (175K+ facilities, ~200 countries) | Facility name, type (hospital, clinic, pharmacy), operating hours, emergency capability, wheelchair access, beds, operator. Built on OSM. Strongest in sub-Saharan Africa and South/SE Asia. | GeoJSON, SHP, KML, CSV | Direct download or [REST API v3](https://healthsites.io/api/docs/) | Free (ODbL) |
| [WHO GHFD](https://www.who.int/data/GIS/GHFD) | 29 countries (growing, primarily Africa & SE Asia) | Country-level georeferenced Health Facility Master Lists: facility name, type, ownership (public/private/NGO), admin level, GPS coordinates. Attribute completeness varies by country. | CSV, GeoJSON (varies) | WHO GHFD portal, also via [GRID3](https://grid3.org/) | Free |
| [HIFLD Hospitals](https://hifld-geoplatform.opendata.arcgis.com/datasets/geoplatform::hospitals-1/about) | USA (~7,600 hospitals) | 34 attribute fields: name, address, coordinates, hospital type (general acute, critical access, children, psychiatric, military, VA, etc.), ownership, bed count, staff count, trauma level (I-V), helipad (Y/N), NAICS code. | SHP, FGDB, CSV, KML | HIFLD Open portal or [Data.gov](https://catalog.data.gov/dataset/hospitals-e60a8) | Free |
| [NHS ODS](https://digital.nhs.uk/services/organisation-data-service) | England (all NHS organizations) | Every NHS org: hospitals, GP practices, pharmacies, dental practices, trusts. ODS code, name, address, postcode, type, status, parent org. Updated nightly. | CSV, Excel, FHIR R4 JSON API | [ODS Export](https://www.odsdatasearchandexport.nhs.uk/) | Free |
| OpenStreetMap Healthcare | Global (variable completeness) | Amenities tagged `amenity=hospital`, `clinic`, `pharmacy`, `dentist`, `doctors`, `healthcare=*`. Attributes include name, hours, phone, emergency, beds, operator. Quality varies by region. | PBF, GeoJSON, SHP | [Overpass Turbo](https://overpass-turbo.eu), [Geofabrik](https://download.geofabrik.de), [HOT Export](https://export.hotosm.org) | Free (ODbL) |

## Disease Surveillance & Epidemiology

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [WHO GHO](https://www.who.int/data/gho) | 194 member states | 2,000+ health indicators: disease incidence/prevalence (TB, HIV, malaria, NTDs), immunization coverage, maternal/neonatal mortality, NCD risk factors, health workforce density, expenditure. | CSV, JSON, XML via [OData API](https://www.who.int/data/gho/info/gho-odata-api) | Free | Free |
| [WHO Disease Outbreak News](https://www.who.int/emergencies/disease-outbreak-news) | Global (1996-present) | Official outbreak reports: disease, affected country, date, case/death counts, narrative. [REST API](https://www.who.int/api/news/diseaseoutbreaknews/sfhelp) provides metadata. Structured [Knowledge Graph](https://www.nature.com/articles/s41597-025-05276-2) (RDF/SPARQL) available. | HTML, JSON API, RDF | Free | Free |
| [CDC WONDER](https://wonder.cdc.gov/) | USA (states, counties) | NNDSS: weekly/annual case counts for ~120 notifiable diseases by state, demographics. Plus mortality (ICD-coded), natality, cancer, environmental exposures. | Query interface, text export | Free | Free |
| [ECDC Surveillance Atlas](https://atlas.ecdc.europa.eu/public/) | 30 EU/EEA countries | Case counts for 52 communicable diseases by country, year, age, sex, classification (confirmed/probable). Influenza, TB, Legionnaires', food-borne, STIs, vaccine-preventable. | Interactive atlas, CSV | Free | Free |
| [JHU COVID-19 Archive](https://github.com/CSSEGISandData/COVID-19) | Global (188 countries), US counties | Daily cumulative cases, deaths, recoveries with lat/lon. US county-level with FIPS codes. Jan 2020 - Mar 2023 (archived). | CSV (time series) | Free (CC BY 4.0) | Archived |
| [GIDEON](https://www.gideononline.com/) | Global (230+ countries, 350+ diseases) | Incidence/prevalence by country/year, outbreak timelines, drug resistance, vaccination coverage, endemic maps. Updated multiple times daily. | Web interface | Subscription (institutional) | Commercial |
| [ProMED-mail](https://www.promedmail.org/) | Global (185+ countries, 1994-present) | Expert-curated emerging infectious disease outbreak reports with disease, location, case/death counts, and commentary. | Text (email posts) | Free | Free |

## Environmental Health

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [OpenAQ](https://openaq.org) | Global (130+ countries, 30K+ stations) | Real-time + historical: PM2.5, PM10, O3, NO2, SO2, CO, BC per station with coordinates. API v3 supports bounding box and radius queries. | JSON REST API; CSV/Parquet on [AWS](https://registry.opendata.aws/openaq/) | Free | Free |
| [EPA AQS](https://aqs.epa.gov/aqsweb/airdata/download_files.html) | USA (~5,000 stations) | Hourly/daily/annual summaries for criteria pollutants (PM2.5, PM10, O3, NO2, SO2, CO, Pb) + 190 air toxics. 1957-present. Each record has site ID, coordinates, county FIPS, QA flags. | CSV (zipped), JSON REST API | Free (API key) | Free |
| [AirNow](https://www.airnow.gov) | USA, Canada, Mexico | Current and forecast AQI by reporting area. Real-time conditions map. | JSON, KML, SHP | Free | Free |
| [GEMStat](https://gemstat.org/) | Global (91 countries, 30K stations) | 26M+ freshwater quality measurements across 659 parameters: dissolved oxygen, pH, nutrients, heavy metals, pesticides, fecal coliforms. 1906-2025. | Interactive maps, CSV | Free (registration) | Free |
| [EEA Noise Mapping](https://noise.eea.europa.eu/) | EU/EEA member states | Population-weighted noise exposure from road, rail, air, industry. Lden/Lnight contour maps at 5 dB intervals. Major agglomerations, roads, railways, airports. Updated every 5 years. | Interactive map, downloadable | Free | Free |
| [CDC/ATSDR EJI](https://www.atsdr.cdc.gov/place-health/php/eji/eji-data-download.html) | USA (census tract) | Environmental Justice Index: 36 indicators across environmental burden, social vulnerability, health vulnerability. Also [SVI](https://atsdr.cdc.gov/place-health/php/svi/) and [Heat & Health Index](https://www.atsdr.cdc.gov/place-health/php/hhi/). | CSV, SHP, Geodatabase | Free | Free |

## Health Infrastructure Access

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [MAP Accessibility](https://malariaatlas.org/project-resources/accessibility-to-healthcare/) | Global (85N-60S) | Travel time to nearest hospital/clinic (minutes) via least-cost-path on friction surface (roads, land cover, topography, water). Motorized + walking-only variants. Also on [GEE](https://developers.google.com/earth-engine/datasets/catalog/projects_malariaatlasproject_assets_accessibility_accessibility_to_healthcare_2019). | GeoTIFF (~1km) | Free | Free / SOTA |
| [AccessMod 5](https://www.accessmod.org/) | User-supplied (global capable) | WHO-supported desktop tool for modeling geographic accessibility. Auto-fetches SRTM, WorldPop, Copernicus land cover, OSM roads. Outputs travel-time isochrones, coverage zones, zonal population stats. | Desktop (R/Shiny) | Free / Open Source | Tool |
| [WHO UHC Index](https://data.who.int/indicators/i/3805B1E/9A706FD) | 194 member states | Universal Health Coverage composite score (0-100) from 14 tracer indicators across 4 domains: RMCH, infectious disease, NCDs, service capacity. SDG 3.8.1. | CSV, interactive | Free | Free |

## Demographics for Health

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [DHS Program](https://dhsprogram.com/Methodology/GPS-Data.cfm) | 90+ countries (200+ surveys) | GPS datasets with displaced cluster coordinates (urban: 0-2km; rural: 0-5km, 1% up to 10km) linked to survey responses on fertility, family planning, maternal/child health, nutrition, HIV, malaria, mortality. [Modeled surfaces](https://spatialdata.dhsprogram.com) provide continuous rasters. | SHP (GPS), CSV (covariates), GeoTIFF (modeled) | Free (registration) | Free |
| [UNICEF MICS](https://mics.unicef.org/surveys) | 100+ countries (6 rounds) | Household surveys: child immunization, nutrition, WASH, birth registration, education. GPS clusters since round 3 (2005). Displacement follows DHS convention. | SPSS, CSV (surveys), SHP (GIS) | Free (registration) | Free |
| [IHME GBD](https://vizhub.healthdata.org/gbd-results/) | 204 countries, 660 subnational | GBD 2023: estimates for 292 causes of death, 375 diseases/injuries, 88 risk factors. Incidence, prevalence, mortality, YLLs, YLDs, DALYs by age, sex, year, location. 1990-2023. | CSV (query tool) | Free (registration) | Free / SOTA |
| [GHDx](https://ghdx.healthdata.org/) | Global | Catalog of 100,000+ health-related data sources used as GBD inputs. Searchable metadata with links to original datasets. | Catalog | Free | Free |

## Emergency Medical Services

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [HIFLD EMS Stations](https://hifld-geoplatform.opendata.arcgis.com/datasets/emergency-medical-service-ems-stations) | USA | All EMS stations: government and private, ambulance services, fire-based EMS. Name, address, coordinates, type, owner. | SHP, FGDB, CSV | Free | Free |
| [HIFLD Fire Stations](https://hifld-geoplatform.opendata.arcgis.com) | USA | Nationwide fire stations (often co-located EMS). Name, address, coordinates, department. | SHP, FGDB | Free | Free |
| HIFLD Hospitals (Trauma filter) | USA | Filter trauma level (I-V) and helipad (Y/N) from hospitals dataset for trauma center mapping. | SHP, CSV | Free | Free |
| [FAA Heliport Data](https://www.faa.gov/airports/airport_safety/airportdata_5010) | USA | Registered heliports including hospital helipads. | CSV | Free | Free |

## Pharmaceutical & Supply Chain

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| OSM Pharmacies | Global (varying) | `amenity=pharmacy` with name, operator, hours, dispensing status, contact. Dense in Europe/NA. | JSON (Overpass) | Free (ODbL) | Free |
| [SAMHSA Treatment Locator](https://findtreatment.gov/) | USA | Mental health + substance use facilities: name, address, services, payment types, populations served. Most comprehensive US behavioral health directory. | Web locator, API, CSV ([Data.gov](https://catalog.data.gov/dataset/mental-health-treatement-facilities-locator)) | Free | Free |

## Mortality & Life Expectancy

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [WHO Mortality Database](https://platform.who.int/mortality) | 120+ countries (1950-present) | Deaths by ICD-coded cause, sex, age group, year. Population denominators. 2M+ records per file. | CSV | Free | Free |
| [Human Mortality Database](https://www.mortality.org/) | 41 countries/areas | High-quality harmonized data: life tables (period/cohort), death rates, population by age/sex/year. Near-complete vital registration only. [STMF](https://www.mortality.org/Data/STMF): weekly deaths for COVID-era analysis. | CSV, XLSX | Free (registration) | Free |
| [IHME GBD Mortality](https://vizhub.healthdata.org/gbd-results/) | 204 countries | Modeled mortality filling vital registration gaps: cause-specific rates, life expectancy, YLLs for 292 causes, 1990-2023. | CSV | Free (registration) | Free / SOTA |

## Pandemic Preparedness

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| [WHO Immunization Data](https://immunizationdata.who.int) | 194 member states | Vaccine coverage by antigen (DTP3, MCV1/2, PCV3, HPV, etc.), year, country. District-level for some countries. [WUENIC estimates](https://www.who.int/data/gho/data/themes/immunization) from 1980. | CSV, interactive | Free | Free |
| [OWID COVID Vaccination](https://github.com/owid/covid-19-data) | 169+ countries | Daily: total doses, first/second/booster, per-capita rates. Archived and maintained. | CSV | Free (CC BY 4.0) | Free |
| [Vaccines.gov Locator](https://www.vaccines.gov) | USA | Vaccination site locations (pharmacies, clinics, FQHCs) with hours and availability. | JSON API | Free | Free |

## Cross-Cutting Resources

| Resource | Description | URL |
|----------|-------------|-----|
| [GHDx](https://ghdx.healthdata.org/) | IHME catalog of 100K+ health data sources worldwide | [ghdx.healthdata.org](https://ghdx.healthdata.org/) |
| [HDX](https://data.humdata.org/) | UN OCHA platform with health facility, disease, population data for crisis countries | [data.humdata.org](https://data.humdata.org/) |
| [WHO Data Portal](https://data.who.int/) | Unified WHO health statistics with API | [data.who.int](https://data.who.int/) |
| [Our World in Data](https://ourworldindata.org/health-meta) | Curated global health visualizations, all downloadable | [ourworldindata.org](https://ourworldindata.org/) |
| [WorldPop](https://www.worldpop.org/) | High-resolution population rasters (critical health access denominators) | [worldpop.org](https://www.worldpop.org/) |
| [GRID3](https://grid3.org/) | Georeferenced population and infrastructure for developing countries | [grid3.org](https://grid3.org/) |
