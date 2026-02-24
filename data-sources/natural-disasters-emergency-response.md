# Natural Disasters & Emergency Response Data

Geospatial data sources for earthquake monitoring, volcanic hazards, flood mapping, wildfire tracking, hurricane forecasting, tsunami warnings, drought assessment, landslide inventories, multi-hazard platforms, and emergency response operations. Covers real-time feeds, historical catalogs, risk indices, and crowd-sourced damage assessment.

> **Quick Picks**
> - **SOTA**: [Copernicus Emergency Management Service (EMS)](https://emergency.copernicus.eu) -- on-demand satellite rapid mapping activated within hours for any major disaster worldwide, free GIS-ready products
> - **Free Best**: [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov) -- near-real-time global fire detection from MODIS and VIIRS within 3 hours, free REST API, no registration required for basic access
> - **Fastest Setup**: [USGS Earthquake Hazards GeoJSON Feeds](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php) -- live earthquake data as GeoJSON, no API key, loads directly into any GIS tool

## Earthquake Data

Real-time seismic event feeds, historical catalogs, ground shaking models, and impact assessment products.

### USGS Earthquake Hazards Program

The primary global earthquake data source for most GIS workflows. Provides real-time feeds, the ComCat catalog, and derived hazard products.

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| USGS Real-Time GeoJSON Feeds | Global | Earthquakes updated every minute; feeds available for past hour, day, 7 days, 30 days; filtered by magnitude (M1+, M2.5+, M4.5+, all). Each feature includes magnitude, depth, location, origin time, felt reports, alert level, review status. | GeoJSON | [earthquake.usgs.gov/earthquakes/feed](https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php) -- no API key required | Free / Fastest Setup |
| ComCat (Comprehensive Earthquake Catalog) | Global, 1900-present | Authoritative catalog: >3M events since 1900, full parameter set per event (magnitude, type, depth, focal mechanism, moment tensor, products). Powers the [earthquake.usgs.gov](https://earthquake.usgs.gov) web UI. | GeoJSON, CSV, QuakeML, KML | [Earthquake Catalog API](https://earthquake.usgs.gov/fdsnws/event/1/) -- RESTful FDSN-compliant endpoint; query by bbox, time, depth, magnitude | Free |
| ShakeMap | Global (M4.5+ events) | Ground shaking intensity grids (PGA, PGV, MMI), uncertainty maps, station data, fault rupture model. Generated automatically within minutes; updated as more data arrives. Includes contours and GeoTIFF grids. | GeoJSON, GeoTIFF, ShapeFile, XML | [USGS ShakeMap](https://earthquake.usgs.gov/data/shakemap/) via [ComCat Product API](https://earthquake.usgs.gov/earthquakes/eventpage/) | Free |
| PAGER (Prompt Assessment of Global Earthquakes for Response) | Global (significant events) | Rapid fatality and economic loss estimates using ShakeMap + exposure models. Alert levels: Green / Yellow / Orange / Red. Fatality and loss probability histograms. Issued within 30 min for significant events. | JSON, PDF, GeoJSON | [USGS PAGER](https://earthquake.usgs.gov/data/pager/) -- JSON summary per event via ComCat API | Free |
| Did You Feel It? (DYFI) | Global (public reports) | Geocoded felt reports from the public: intensity, ZIP-code-level and grid aggregation (1km), shaking descriptions. Millions of responses since 1999. Useful for calibrating ShakeMap in data-sparse regions. | GeoJSON, CSV | [DYFI CDI Maps API](https://earthquake.usgs.gov/data/dyfi/) -- per-event endpoint returns aggregated intensity by ZIP and geo grid | Free |
| USGS Hazard Curves & Maps | CONUS, Alaska, Hawaii, territories | Probabilistic Seismic Hazard Analysis (PSHA) grids: PGA, SA at multiple return periods (475yr, 2475yr). Underpins the US National Building Code. Released with each NSHM update. | GeoTIFF, CSV, ESRI Grid | [USGS Geologic Hazards Science Center](https://www.usgs.gov/programs/earthquake-hazards/hazard-maps) -- direct download | Free |

### European Mediterranean Seismological Centre (EMSC)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| EMSC Real-Time Catalog | Euro-Med region (global M4+) | Near-real-time earthquakes from European seismic networks; typically faster reporting than USGS for European events. Felt reports via Lastquake app. | GeoJSON, CSV, QuakeML, JSON | [EMSC API](https://www.seismicportal.eu/fdsnws/event/1/) -- FDSN-compatible; also [WebSocket feed](https://www.seismicportal.eu/realtime.html) for streaming | Free |
| EMSC Felt Reports | Europe (crowd-sourced global) | Eyewitness testimonies with location, intensity, damage descriptions. Published within minutes of felt events. | JSON | [EMSC Testimonies API](https://www.seismicportal.eu/testimonies-ws/) | Free |

### International Seismological Centre (ISC)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| ISC Bulletin | Global, 1904-present | Reviewed and revised earthquake catalog with relocated hypocenters; the authoritative historical archive. Collects data from 3,000+ seismic stations. ISC-GEM catalog: 1900-2014, reviewed M5.5+. | CSV, QuakeML, ISF | [ISC Event Catalog](https://www.isc.ac.uk/iscbulletin/search/bulletin/) -- web search + FTP bulk download | Free |
| ISC-GEM Global Instrumental Earthquake Catalogue | Global, 1900-2014 | Homogeneous Mw catalog for hazard analysis: ~32,000 events M5.5+, uniform moment magnitudes, depth estimates, focal mechanisms. | CSV, Excel | [ISC-GEM Catalog](https://www.isc.ac.uk/iscgem/) -- direct download | Free |

### Global CMT Project

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| Global CMT Catalog | Global, 1976-present | Centroid Moment Tensor solutions for M4.5+ earthquakes: scalar moment, moment magnitude, fault plane solutions, centroid location and depth. >60,000 solutions. Updated monthly with ~2-month lag. | ASCII text, miniseed | [globalcmt.org](https://www.globalcmt.org/CMTsearch.html) -- web search, FTP bulk download | Free |

## Volcanic Activity

Eruption alerts, ash cloud advisories, lava flow mapping, thermal anomaly detection, and deformation monitoring.

### Smithsonian Global Volcanism Program (GVP)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| GVP Volcanoes of the World Database | Global (~1,400 Holocene volcanoes) | Catalog of all known Holocene volcanoes with coordinates, type, elevation, rock type, tectonic setting, eruption history. Eruption database with VEI (Volcanic Explosivity Index) for 10,000+ eruptions. | CSV, KML, API (JSON) | [volcano.si.edu](https://volcano.si.edu) -- bulk download; [GVP API](https://volcano.si.edu/api/) returns JSON | Free |
| Weekly Volcanic Activity Reports | Global | Weekly summary of activity at ~50-100 volcanoes; authored by GVP in partnership with USGS/VAAC. Includes description, imagery, and links to monitoring data. Updated every Wednesday. | HTML, GeoRSS | [Weekly Activity Report](https://volcano.si.edu/reports_weekly.cfm) -- GeoRSS feed available | Free |

### USGS Volcano Hazards Program

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| USGS Volcano Notifications | US volcanoes (169 monitored) | Real-time alert-level changes (Normal / Advisory / Watch / Warning), aviation color codes (Green / Yellow / Orange / Red), eruption notifications. Delivered via email or API. | JSON (ATOM/RSS), email | [Volcano Notifications Service](https://volcanoes.usgs.gov/vns2/) -- subscribe by volcano or all-US feed | Free |
| USGS Volcano Observatory Data | Alaska, Hawaii, Cascades, Yellowstone | Seismicity, ground deformation (GPS/InSAR), gas emissions (SO2, CO2), webcam images, lava flow maps. Observatory-specific dashboards with data downloads. | CSV, KML, GeoTIFF, JSON | [USGS Volcanoes](https://www.usgs.gov/programs/VHP) -- per-observatory portals | Free |

### Volcanic Ash Advisory Centers (VAACs)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| VAAC Advisories (9 Centers) | Global (zonal responsibilities) | Volcanic Ash Advisory (VAA) messages and Volcanic Ash Graphic (VAG): ash cloud extent, altitude, movement forecast 6/12/18 hours. Issued as ICAO-format SIGMET text. Centers: London, Toulouse, Anchorage, Washington DC, Montreal, Darwin, Tokyo, Buenos Aires, Pretoria. | ICAO SIGMET text, KML, GeoJSON | [ICAO VAAC Portal](https://www.ssd.noaa.gov/VAAC/) (NOAA WC-VAAC); London VAAC: [metoffice.gov.uk/aviation/vaac](https://www.metoffice.gov.uk/aviation/vaac) | Free |

### Thermal Anomaly / Hotspot Detection

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| MODVOLC (Hawaii Inst. of Geophysics and Planetology) | Global | Volcanic thermal anomalies from MODIS Terra/Aqua at 1km; daily detection; 2000-present. Pixel-level alert lists with coordinates, spectral radiance, dates. Designed specifically for volcanic hotspot detection. | CSV, KML | [HIGP MODVOLC](https://modis.higp.hawaii.edu) -- map interface + downloadable alert tables | Free |
| NASA FIRMS (volcanoes) | Global | MODIS (1km) and VIIRS (375m) fire/thermal anomaly detections include volcanic hotspots. NRT data within 3 hours. Longer archive than MODVOLC for some regions. Filter by satellite, confidence level. | CSV, SHP, KML, JSON (API) | [FIRMS Fire Map](https://firms.modaps.eosdis.nasa.gov) -- REST API: `firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{source}/{area}/{days}` | Free (API key registration) |
| Copernicus Sentinel-5P SO2 | Global | Daily sulfur dioxide columns from TROPOMI at 3.5x7km; SO2 is the primary volcanic gas tracer; can detect eruption plumes and track them globally. | NetCDF | [Copernicus Data Space](https://dataspace.copernicus.eu), [GEE](https://earthengine.google.com) | Free |

### Global Volcano Model (GVM) / VHUB

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| GVM Global Volcanic Hazard and Risk Study | Global | Volcanic hazard exposure indices, population in volcanic threat zones, Volcanic Threat level rankings for 1,500+ volcanoes. Data from 2015 UN assessment. | CSV, Shapefile | [GVM Portal](https://globalvolcanomodel.org) | Free |

## Flood & Hydrology

Flood extent mapping, streamflow monitoring, probabilistic flood hazard layers, and real-time inundation models.

### Dartmouth Flood Observatory (DFO)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| DFO Global Active Archive of Large Flood Events | Global, 1985-present | ~5,000 major flood events: coordinates, start/end dates, country, displaced persons, deaths, main cause, river, severity index. Event boundaries mapped from Landsat, MODIS, and Sentinel. | CSV, Shapefile, KML, GeoTIFF (flood extent) | [DFO Flood Observatory](https://floodobservatory.colorado.edu/Archives.html) -- direct download | Free |
| DFO Flood Inundation Maps | Global (per major event) | Satellite-derived flood extents mapped from MODIS and Landsat imagery for each major event. Available as GeoTIFF and KML. | GeoTIFF, KML | [DFO Inundation Maps](https://floodobservatory.colorado.edu/FloodMapData.html) | Free |

### GloFAS (Global Flood Awareness System / Copernicus)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| GloFAS River Discharge Reanalysis | Global river network | Daily river discharge 1979-present at ~5km river network. GloFAS-ERA5 reanalysis: 40+ years hydrograph for any river reach globally. | NetCDF | [Copernicus CDS](https://cds.climate.copernicus.eu) -- `cdsapi` Python download | Free |
| GloFAS Real-Time Flood Forecast | Global, 5-30 day lead time | Probabilistic river discharge forecasts at 3-hourly to daily timesteps for 30-day horizon; ensemble flood alert map at 0.1 degree. Updated daily. | NetCDF, WMS | [GloFAS Platform](https://www.globalfloods.eu) -- web map; data via CDS | Free |
| GloFAS Seasonal Outlook | Global | Monthly probability of above-normal/normal/below-normal river discharge for 4-month seasons. | NetCDF | [GloFAS](https://www.globalfloods.eu/seasonal-forecast/) | Free |

### USGS National Water Information System (NWIS)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| USGS Streamflow (NWIS) | USA (~10,000 gages) | Real-time streamflow (instantaneous, 15-min intervals), peak flow, gage height, water quality. Historical records back to 1880s. Gage locations with HUC watershed codes. | JSON, RDB (tab-delimited), WaterML | [waterdata.usgs.gov REST API](https://waterdata.usgs.gov/nwis) -- `dataRetrieval` R package and `dataretrieval` Python package | Free |
| USGS WaterWatch | USA | Real-time streamflow conditions: current flow vs. historical percentiles; flood stage and drought conditions nationwide; daily streamflow maps. | JSON (API), PDF maps | [WaterWatch](https://waterwatch.usgs.gov) | Free |

### FEMA National Flood Hazard Layer (NFHL)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| NFHL (National Flood Hazard Layer) | USA (mapped areas) | Official FEMA Flood Insurance Rate Map (FIRM) data: 100-year and 500-year floodplain zones (A, AE, V, X zones), base flood elevations, floodways, cross-sections. Used for insurance and zoning. | Shapefile, GDB, WMS, REST API | [FEMA MSC](https://msc.fema.gov/portal/home), [NFHL WMS](https://hazards.fema.gov/gis/nfhl/rest/services/public/NFHL/MapServer), ArcGIS REST API | Free |

### Global Flood Models

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| AQUEDUCT Floods (WRI) | Global | Riverine and coastal flood hazard layers at multiple return periods (2 to 1000 year). Baseline and future climate projections (RCP 4.5, 8.5) to 2080. Inundation depth rasters at ~1km. | GeoTIFF, CSV (risk scores) | [WRI Aqueduct](https://www.wri.org/aqueduct), [GEE](https://earthengine.google.com) | Free |
| Global Flood Database (GFD) | Global, 2000-2018 | 913 individual flood events mapped from MODIS by Tellman et al.; inundation extents with duration. Used for flood model validation and exposure estimation. | GeoTIFF | [GFD via GEE](https://global-flood-database.cloudtostreet.ai) | Free |
| JBA Global Flood Maps | Global | Probabilistic flood depth grids (riverine, surface water, coastal) at 30m resolution across 5 return periods. Industry benchmark for insurance pricing. | GeoTIFF | [JBA Risk](https://www.jbarisk.com/flood-services/global-flood-maps/) | Commercial (free sample tiles) |
| GLOFRIS / PCR-GLOBWB | Global | Open-source global flood inundation model results (1km); return periods 10-1000 years; riverine only. Used in WRI Aqueduct v2. | NetCDF | [PBL Netherlands Environmental Assessment Agency](https://www.pbl.nl) | Free |

## Wildfire

Active fire detection, burn severity mapping, fire perimeters, and cumulative burned area products.

### NASA FIRMS (Fire Information for Resource Management System)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| FIRMS MODIS NRT (MOD14/MYD14) | Global | MODIS active fire detections at 1km pixel resolution; Terra and Aqua combined ~2 detections/day at equator. Confidence level (low/nominal/high), FRP (fire radiative power), brightness temperature. NRT within 3 hours; archive to 2000. | CSV, SHP, KML | [FIRMS Download](https://firms.modaps.eosdis.nasa.gov/download/) or REST API: `firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/MODIS_NRT/{bbox}/{days}` | Free (key required) |
| FIRMS VIIRS 375m NRT (VNP14IMGT / VJ114IMGT) | Global | VIIRS active fire detections from S-NPP and NOAA-20 at 375m; ~1 detection/day per satellite. Much higher spatial resolution than MODIS; better for small fires. NRT within 3 hours. | CSV, SHP, KML | [FIRMS Download](https://firms.modaps.eosdis.nasa.gov/download/) -- same REST API with source `VIIRS_SNPP_NRT` or `VIIRS_NOAA20_NRT` | Free (key required) / SOTA |
| FIRMS VIIRS 375m Standard (archive) | Global, 2012-present | Quality-controlled standard science product; same detections as NRT but with enhanced QA. Better for multi-year analysis. | CSV, SHP | [FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/) | Free |
| NASA FIRMS WorldView | Global | Web-based map viewer overlaying MODIS and VIIRS NRT fire detections on satellite imagery. No download required. | Web map | [firms.modaps.eosdis.nasa.gov/map](https://firms.modaps.eosdis.nasa.gov/map/) | Free |

### NIFC / InciWeb (US Wildfire Incidents)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| NIFC Current Wildfires | USA | National Interagency Fire Center real-time fire perimeters (GeoMAC / NIFC Geospatial), fire locations, containment percentage. Updated multiple times daily for active fires. | KML, GeoJSON, REST | [NIFC GeoMAC](https://www.nifc.gov/fire-information/active-fire-mapping) -- [ArcGIS REST service](https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/Current_WildlandFire_Perimeters/FeatureServer) | Free |
| InciWeb | USA | Incident information portal for wildfires, floods, hurricanes: descriptions, maps, news, photos. Not machine-readable API but useful for context. | HTML | [InciWeb](https://inciweb.wildfire.gov) | Free |
| NFPORS (National Fire Plan Operations and Reporting System) | USA | Fuels treatment history: prescribed burns, mechanical treatments. Unit-level polygons with acres, species, dates. | Shapefile, CSV | [NFPORS](https://www.nfpors.gov) | Free |

### Burned Area Products

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| MTBS (Monitoring Trends in Burn Severity) | USA, 1984-present | Landsat-derived burn severity maps for all US fires >1,000 acres (West) / >500 acres (East): burn severity classes, differenced Normalized Burn Ratio (dNBR), fire perimeter polygons. Annual release. | GeoTIFF, SHP | [MTBS](https://www.mtbs.gov/direct-download), [GEE](https://earthengine.google.com/dataset/USFS/GTAC/MTBS/burned_area_boundaries/v1) | Free |
| MCD64A1 MODIS Burned Area | Global, 2001-present | Monthly 500m global burned area from MODIS: burn date, uncertainty, QA. Detects fires >100-500 ha. Monthly composites. | HDF4, GeoTIFF | [LP DAAC](https://lpdaac.usgs.gov/products/mcd64a1v061/), [GEE](https://earthengine.google.com) | Free |
| FireCCI51 / C3S Fire BA | Global, 1982-2020 | ESA CCI 250m burned area from MODIS; considered more accurate than MCD64A1 for some biomes. Long time series. | NetCDF | [ESA CCI](https://www.esa-cci.org/?page_id=78) | Free |
| GOES-16/17 Active Fire (GOES Wildfire ABBA) | Americas | Near-real-time fire detection from GOES-16 (East) and GOES-17/18 (West) at 2km every 5-15 minutes. Excellent for tracking fire spread in real time. | CSV, KML | [GOES Wildfire](https://www.goes.noaa.gov/products/enterprise/ABBAfire/README.txt), NOAA CLASS | Free |

### Canadian and International

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| Canadian Wildland Fire Information System (CWFIS) | Canada | Daily fire hotspots, fire perimeters, fire weather index (FWI) grids for Canada. Updated daily during fire season. | WMS, KML, SHP, CSV | [CWFIS](https://cwfis.cfs.nrcan.gc.ca/home) -- WMS + data download | Free |
| Copernicus EFFIS (European Forest Fire Information System) | Europe, Mediterranean, Middle East | Fire danger forecast (FWI daily), active fire perimeters, burned area statistics, post-fire damage assessment. Near-real-time. | WMS, REST API, SHP | [EFFIS](https://effis.jrc.ec.europa.eu) -- ArcGIS REST API endpoints | Free |
| Global Fire Emissions Database (GFED4) | Global, 1997-present | Monthly gridded fire emissions: CO2, CO, CH4, BC, PM2.5 from fires at 0.25 degree. Derived from MODIS burned area + Carnegie-Ames-Stanford biogeochemical model. | HDF5, NetCDF | [globalfiredata.org](http://www.globalfiredata.org) | Free |

## Hurricane & Tropical Cyclone

Best track databases, forecast advisories, wind radii, and storm surge model outputs.

### IBTrACS (International Best Track Archive for Climate Stewardship)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| IBTrACS v04 | Global, 1842-present | NOAA's definitive global tropical cyclone track database: ~13,600 storms from all basins (NA, EP, WP, NI, SI, SP). Variables: position, wind speed (10-min / 1-min sustained), central pressure, storm size (wind radii at 34/50/64 kt), landfall. Aggregates data from 12+ agencies. Updated weekly. | CSV, NetCDF, SHP | [NOAA NCEI IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) -- `IBTrACS.ALL.v04r00.nc` (full dataset ~200MB) | Free / SOTA |
| IBTrACS on GEE | Global | IBTrACS tracks as Earth Engine FeatureCollections for large-scale analysis without download. | GEE FeatureCollection | [GEE Community Catalog](https://gee-community-catalog.org) | Free |

### National Hurricane Center (NHC) & CPHC

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| NHC Forecast Advisories | Atlantic & Eastern Pacific | Active storm best track, intensity forecasts (5-day track, wind radii), probability cones, watches/warnings. Public advisories every 6 hours; special advisories more frequently. | GeoJSON, KML, SHP, RSS, XML | [NHC GIS Data](https://www.nhc.noaa.gov/gis/) -- advisory shapefiles + GeoJSON in real time; [RSS feeds](https://www.nhc.noaa.gov/rss/) | Free |
| NHC Wind Probability Products | Atlantic & Eastern Pacific | Probability of 34/50/64 kt winds within 5 days; cumulative wind probability grids (0.1 degree). | SHP, KML | [NHC GIS](https://www.nhc.noaa.gov/gis/) -- `{storm_id}_best_track.shp`, `{storm_id}_wsp{kt}knt120hr.zip` | Free |
| NHC Historical Best Track (HURDAT2) | Atlantic (1851-present), E. Pacific (1949-present) | Position, max sustained wind, minimum pressure every 6 hours; landfall flags; ~1,900 Atlantic storms total. Text format with header rows and data rows. | CSV (fixed-width) | [NHC HURDAT2](https://www.nhc.noaa.gov/data/hurdat/) -- direct text file download | Free |

### Joint Typhoon Warning Center (JTWC)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| JTWC Best Track Data | Western Pacific, Indian Ocean, Southern Hemisphere | Historical best track database (ATCF format): position, intensity, wind radii. Real-time advisories for active storms in non-RSMC basins. | ATCF text format, CSV | [JTWC](https://www.metoc.navy.mil/jtwc/jtwc.html), [UCAR RDA JTWC dataset](https://rda.ucar.edu/datasets/ds824.1/) | Free |

### Storm Surge Models

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| SLOSH (Sea Lake and Overland Surges from Hurricanes) | US coastline (64 basins) | NHC-maintained storm surge model for US landfalling tropical cyclones; Maximum Envelope of Water (MEOW) and Maximum of MEOWs (MOM) grids; used for official NHC surge warnings. | Raster grids (ArcGIS), KML | [NHC SLOSH](https://www.nhc.noaa.gov/surge/slosh.php) -- MEOMs available for download per basin | Free |
| ADCIRC Storm Surge (NOAA/FEMA) | US coastlines | High-resolution unstructured-grid surge model used in FEMA Flood Insurance Studies and NOAA operational products. Real-time ADCIRC runs for landfalling US storms. | NetCDF | [ADCIRC](https://adcirc.org), [NOAA Tides & Currents](https://tidesandcurrents.noaa.gov) operational products | Free (model) / Results per event |
| Global Tropical Cyclone Storm Surge Hazard Maps | Global tropical coasts | Return-period storm surge inundation maps (10/100/1000-year) for the global tropics; from JRC/Deltares. | GeoTIFF | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) | Free |

### Regional Meteorological Organizations

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| RSMC Tokyo (JMA) | Western North Pacific | Official best track, intensity, warnings for typhoons. Final best track released post-season. | Text, KML | [JMA Typhoon Information](https://www.jma.go.jp/bosai/map.html#contents=typhoon) | Free |
| RSMC La Réunion (Météo-France) | South Indian Ocean | Best track and advisories for tropical cyclones in South Indian Ocean. | Text, KMZ | [Météo-France](https://www.meteo.fr/temps/domtom/La_Reunion/webcmrs9.0/anglais/index.html) | Free |
| BoM (Australian Bureau of Meteorology) | Australian region / Southern Pacific | Best tracks, tropical cyclone position bulletins, intensity forecasts. | KML, text | [BoM Tropical Cyclones](http://www.bom.gov.au/cyclone/) | Free |

## Tsunami

Warning systems, historical databases, real-time buoy networks, and inundation models.

### NOAA NCEI Historical Tsunami Database

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| NOAA/NGDC Historical Tsunami Events | Global, 2000 BC-present | ~2,600 tsunami events: source location, origin (earthquake, volcano, landslide), magnitude, wave heights at tide gauges, runup measurements, fatalities, damage. Searchable spatial database. | CSV, SHP, JSON (API) | [NCEI Tsunami Events](https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/tsunamis) -- REST API; [NCEI Natural Hazards Viewer](https://www.ngdc.noaa.gov/hazel/hazard-service/) | Free |
| NOAA NCEI Tsunami Runup Database | Global | Point dataset of measured wave heights at specific coastal locations; >50,000 runup measurements linked to parent tsunami events. | CSV, JSON | [NCEI Tsunami Runup API](https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/hazards/events/tsunamis/runups) | Free |

### Pacific Tsunami Warning Center (PTWC) & Tsunami Warning Systems

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| PTWC Real-Time Warnings | Pacific, Caribbean (global threat M7.5+) | Tsunami threat messages (Information / Watch / Warning), estimated arrival times, projected wave heights. Issued via GTS (WMO), email, website. | Text (PTWC product), XML | [PTWC](https://ptwc.weather.gov) -- product feeds; [USGS PTWC partner feeds](https://earthquake.usgs.gov) | Free |
| NTWC (National Tsunami Warning Center) | Alaska, US West Coast, BC Canada | Warnings, watches, advisories, information statements for non-Pacific threats. | Text products | [NTWC](https://www.tsunami.gov) | Free |
| IOC Tsunami Programme | Global | UNESCO IOC coordinates the global Tsunami Warning and Mitigation System (GTWMS). Sea level data from 1,000+ tide gauges via GLOSS network. | Varies | [IOC Sea Level Station Monitoring Facility](http://www.ioc-sealevelmonitoring.org) | Free |

### DART Buoy Network

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| DART (Deep-ocean Assessment and Reporting of Tsunamis) | Global ocean basins (~60 buoys) | Real-time bottom pressure recorder data at 15-second to 1-hour intervals; detects tsunami wave trains in deep ocean. Data used by PTWC and NTWC to confirm tsunami generation and constrain models. | CSV, JSON | [NOAA DART](https://www.ndbc.noaa.gov/dart.shtml) -- [NDBC real-time data](https://www.ndbc.noaa.gov); per-buoy data streams | Free |

### Tsunami Inundation Models & Hazard Maps

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| NOAA NCEI Inundation Maps | US States (Pacific Coast) | Probabilistic and deterministic tsunami inundation hazard maps for Oregon, Washington, California, Hawaii, Alaska. Evacuation zone boundaries. | PDF, SHP | [NCEI Tsunami Inundation](https://www.ncei.noaa.gov/products/natural-hazards/tsunamis/inundation-maps) | Free |
| GEBCO / SRTM30_PLUS (Bathymetry for modeling) | Global oceans | High-resolution bathymetry underlying all tsunami propagation models; 15 arc-second global grids. | NetCDF | [GEBCO](https://www.gebco.net) | Free |

## Drought

Drought severity indices, groundwater anomalies, crop stress indicators, and early warning systems.

### US Drought Monitor (USDM)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| US Drought Monitor Weekly Maps | USA (Contiguous, Alaska, Hawaii, Puerto Rico) | Weekly drought severity classification: D0 (Abnormally Dry) through D4 (Exceptional Drought). Combines SPI, PDSI, soil moisture, streamflow, expert assessment. Published every Thursday. Archive to 2000. | SHP, KML, GeoTIFF | [USDM GIS Data](https://droughtmonitor.unl.edu/DmData/GISData.aspx) -- weekly shapefile download, [USDM REST API](https://usdmdataservices.unl.edu/api/) returns JSON by county/state | Free / Practical |
| USDM Statistics | USA (state, county level) | Weekly drought statistics: percent of area in each D0-D4 category by state, county, HUC. CSV download since 2000. Useful for time-series analysis. | CSV | [USDM Statistics API](https://usdmdataservices.unl.edu/api/) | Free |

### NASA GRACE / GRACE-FO Groundwater

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| GRACE/GRACE-FO Groundwater Anomalies | Global, 2002-present (monthly) | Terrestrial water storage (TWS) anomalies from gravity satellite data at ~300km resolution. GRACE (2002-2017), GRACE-FO (2018-present). Monthly composites. Processed by CSR, MASCON, and JPL solutions. | NetCDF | [NASA GRACE Tellus](https://grace.jpl.nasa.gov/data/get-data/) -- JPL MASCON download; [GEE](https://earthengine.google.com) | Free |
| GLDAS (Global Land Data Assimilation System) | Global | Modeled soil moisture, runoff, evapotranspiration, groundwater; forced by satellite and surface data. 3-hour, monthly; 0.25 and 1.0 degree. Used with GRACE for groundwater isolation. | NetCDF | [NASA GES DISC](https://disc.gsfc.nasa.gov/datasets?keywords=GLDAS) | Free |

### Global Drought Monitoring

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| SPEI Global Drought Monitor | Global, 1950-present | Standardized Precipitation-Evapotranspiration Index at multiple timescales (1-48 months), updated monthly. Based on CRU TS data. 0.5 degree grid. | NetCDF, GeoTIFF | [SPEI Global Drought Monitor](https://spei.csic.es/map/maps.html) -- [bulk download](https://digital.csic.es/handle/10261/202305) | Free |
| Copernicus Global Drought Observatory (GDO) | Global (emphasis Europe) | Composite drought indicator (CDI) combining SPI, soil moisture, FAPAR anomaly. Monthly; 5km over Europe, 25km globally. | NetCDF, WMS | [Copernicus GDO](https://edo.jrc.ec.europa.eu/gdo/php/index.php?id=2001) | Free |
| SPI / PDSI via NOAA CPC | Global | Monthly SPI at 1, 3, 6, 12, 24 month timescales; PDSI monthly from CPC; NOAA Climate Prediction Center products. | NetCDF, ASCII grids | [NOAA CPC](https://www.cpc.ncep.noaa.gov/products/monitoring_and_data/) | Free |
| MODIS Vegetation Stress (NDVI / EVI Anomalies) | Global, 2000-present | MODIS Terra/Aqua 16-day NDVI and EVI at 250m and 500m; anomaly products derived by agencies for drought monitoring. | HDF4, GeoTIFF | [LP DAAC](https://lpdaac.usgs.gov), [GEE](https://earthengine.google.com) | Free |

### Famine Early Warning Systems Network (FEWS NET)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| FEWS NET Food Security & Drought Data | Africa, Central Asia, Latin America, Caribbean | Integrated Phase Classification (IPC) food security maps, livelihood zone maps, rainfall anomaly, NDVI anomaly, price data. Updated monthly. Targets acute food insecurity from drought and conflict. | SHP, PDF, WMS | [FEWS NET Data Center](https://fews.net/data) -- Shapefile downloads + WMS endpoints | Free |
| FEWS NET RFE (Rainfall Estimates) | Africa, 1995-present | Dekadal (10-day) 10km rainfall estimates for Africa from CPC/NOAA; combined microwave-IR algorithm. Used for drought and agricultural monitoring. | GeoTIFF | [FEWS NET](https://earlywarning.usgs.gov/fews/datadownloads) | Free |

## Landslide

Global and regional landslide inventories, hazard models, real-time susceptibility, and satellite-detected mass movements.

### NASA Global Landslide Catalog (GLC)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| NASA GLC | Global, 2007-present | Point database of ~11,000 rainfall-triggered mass movements (landslides, debris flows, mudflows): date, location, fatalities, source URL, trigger, size. Compiled from news and reports. | CSV, KML, SHP | [NASA GLC](https://gpm.nasa.gov/landslides/index.html), [Zenodo](https://zenodo.org/record/159010) | Free |
| LHASA (Landslide Hazard Assessment for Situational Awareness) | Global | NASA near-real-time model combining GPM precipitation with a susceptibility map; produces daily/nowcast landslide hazard assessment at ~1km during rainfall events. Alert levels: none/low/moderate/high. | GeoTIFF, JSON | [LHASA v2](https://gpm.nasa.gov/landslides/tools.html), [GEE App](https://gpm.nasa.gov/landslides/lhasa.html) | Free |

### USGS Landslide Hazards Program

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| USGS Landslide Inventories | USA + global partner inventories | Historical landslide inventories at state and regional scales; point and polygon features with type, date, area. Coverage varies by state -- best for Colorado, Pacific Northwest. | Shapefile, GDB | [USGS Landslides](https://www.usgs.gov/programs/landslide-hazards/data) -- per-project download | Free |
| US National Landslide Hazard Assessment (2017) | CONUS | Relative landslide susceptibility model at ~30m based on slope, lithology, precipitation, land cover. | GeoTIFF | [USGS NHAP](https://www.usgs.gov/programs/landslide-hazards/national-landslide-hazards-assessment) | Free |

### ESA Geohazards Exploitation Platform (GEP)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| ESA GEP InSAR / SBAS Products | Global (Sentinel-1 coverage) | Cloud-based processing of Sentinel-1 SAR time series for ground deformation: displacement maps and time series via SBAS-InSAR and PSI. Identifies slow-moving landslides, subsidence, uplift. | GeoTIFF, CSV | [ESA GEP](https://geohazards-tep.eu) -- cloud processing platform; requires registration | Free (research) |
| Copernicus Ground Motion Service (EGMS) | Europe | Pan-European ground motion measurements from Sentinel-1: mean velocity maps + time series for 5+ million measurement points. Updated every 3 years. | SHP, CSV, NetCDF | [EGMS](https://egms.land.copernicus.eu) | Free |

### Global Landslide Susceptibility / Risk

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| ARUP/Verisk Global Landslide Hazard Map | Global | 1km global landslide susceptibility combining slope, lithology, distance to fault, land cover, seismicity, precipitation. Five susceptibility classes. | GeoTIFF | [UNDRR GAR Atlas](https://www.preventionweb.net/understanding-disaster-risk/hazard-information-profiles) | Free |
| Global Friction Surface (Oxford / Malaria Atlas) | Global | Motorized and walking travel time -- useful as proxy for landslide isolation and evacuation time estimation. | GeoTIFF | [Malaria Atlas Project](https://malariaatlas.org/research-project/accessibility_to_cities/) | Free |

## Multi-Hazard Platforms

Integrated global disaster monitoring, alert systems, and risk indices aggregating multiple hazard types.

### GDACS (Global Disaster Alert and Coordination System)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| GDACS Event Feed | Global (automatic + manual alerts) | Real-time alerts for earthquakes (M5.5+), tropical cyclones, floods, volcanoes, tsunamis, droughts. Alert score (Green/Orange/Red) based on exposure. Published within minutes of event. | RSS, GeoRSS, GeoJSON, KML, REST API | [GDACS API](https://www.gdacs.org/xml/rss.xml) -- GeoRSS feed; [REST endpoint](https://gdacs.org/gdacsapi/api/) | Free |
| GDACS Impact Estimates | Global | Population exposed per alert, country-level exposure breakdown, total affected estimates. Available per event via API. | JSON | [GDACS API](https://www.gdacs.org/gdacsapi/api/) -- `/events/{eventid}/impact` | Free |
| Virtual OSOCC (On-Site Operations Coordination Centre) | Global | Coordination platform for international search-and-rescue: activation reports, team deployment, situation reports. Requires UN account. | Web platform | [VOSOCC](https://vosocc.unocha.org) | Free (registration) |

### ReliefWeb

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| ReliefWeb API | Global | OCHA's humanitarian information portal: 1M+ situation reports, maps, assessments, appeals. Filterable by country, disaster type, organization, date. | JSON (REST API) | [ReliefWeb API](https://apidoc.reliefweb.int) -- `https://api.reliefweb.int/v1/` -- no key needed for basic queries | Free |

### PDC DisasterAWARE

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| PDC DisasterAWARE | Global | Multi-hazard situation awareness platform integrating USGS, FIRMS, NHC, GDACS, and 100+ feeds. Population exposure overlays, impact projections. Used by USAID/OFDA and national civil defense agencies. | Web app, REST API (licensed) | [DisasterAWARE](https://www.pdc.org/tools/disasteraware/) | Free (basic public access); API: Commercial |
| PDC Hazard Exposure Data | Global | Population and infrastructure exposed to multi-hazard footprints; HDX-published summary datasets. | CSV, SHP | [PDC on HDX](https://data.humdata.org/organization/pdc) | Free (HDX) |

### EM-DAT & INFORM Risk Index

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| INFORM Risk Index | Global (191 countries) | Multi-dimensional risk index: Hazard & Exposure, Vulnerability, Lack of Coping Capacity. Composite and 50+ sub-indicators per country. Updated annually; country-level scores and rankings. | Excel, CSV | [INFORM](https://drmkc.jrc.ec.europa.eu/inform-index) -- direct download | Free |
| UN Sendai Monitor | Global | Tracks progress on Sendai Framework targets (A-G): disaster deaths, affected people, economic losses, critical infrastructure damaged, DRR strategies, international cooperation, DRR data. | CSV | [UN Sendai Monitor](https://sendaimonitor.undrr.org) | Free |
| Global Risk Data Platform (UNDRR / UNDP) | Global | Standardized multi-hazard risk data for 200+ countries: exposure, vulnerability, multi-hazard composites. Basis for Global Assessment Report (GAR). | WMS, SHP, GeoTIFF | [UNDRR GRDP](https://www.preventionweb.net/understanding-disaster-risk/risk-data) | Free |

## Emergency Response

Satellite activation, rapid mapping products, and coordination data for active disasters.

### Copernicus Emergency Management Service (Copernicus EMS)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| Copernicus EMS Rapid Mapping | Global (activated on request by authorized body) | Post-disaster maps produced within 24-48 hours from Sentinel, SPOT, Pleiades, and commercial SAR: delineation maps (flood extent, fire perimeter, landslide area), grading maps (building damage), situation maps. Every product is GIS-ready. | GeoPackage, Shapefile, GeoTIFF, PDF | [Copernicus EMS Mapping Portal](https://mapping.emergency.copernicus.eu) -- all products free download | Free / SOTA |
| Copernicus EMS Risk & Recovery | Europe | Pre-disaster flood, wildfire, storm mapping products; post-event damage validation. Longer lead times than Rapid Mapping. | Shapefile, GeoTIFF | [Copernicus EMS](https://emergency.copernicus.eu) | Free |
| Copernicus EMS Validation Products | Europe | Ground-truth validation datasets for testing automated damage detection algorithms. | CSV, Shapefile | [Copernicus EMS](https://emergency.copernicus.eu) | Free |

### UNOSAT (UNITAR Operational Satellite Applications Programme)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| UNOSAT Rapid Mapping Products | Global | Satellite-derived damage assessments and flood maps for humanitarian crises (conflict + natural disasters): building damage by class, displaced persons estimation, infrastructure status. Archive of 1,900+ maps since 2003. | Shapefile, PDF, KML | [UNOSAT Portal](https://unosat.org/products/) -- free download of all GIS products | Free |
| UNOSAT Global Exposure Model | Global | Building footprint exposure at country level for disaster loss modeling; links to HAZUS-style risk frameworks. | SHP, CSV | [UNOSAT](https://unosat.org) | Free (registration) |

### International Charter Space and Major Disasters

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| International Charter | Global (authorized users: civil protection, UN agencies) | Coordinates satellite data acquisition from 17 member space agencies (ESA, NASA, JAXA, ISRO, etc.) for activated disasters; provides raw and processed imagery to crisis managers within hours. Public archive of activated events visible on portal. | Various (agency-dependent) | [disasterscharter.org](https://disasterscharter.org) -- activation archive publicly browsable; data only for authorized entities | Free (for authorized users) |

### Sentinel Asia

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| Sentinel Asia | Asia-Pacific | JAXA-led space-based disaster management support for Asia-Pacific: satellite imagery requests, value-added products (flood maps, damage assessment) for regional member organizations. | GeoTIFF, PDF, KML | [Sentinel Asia Portal](https://sentinel-asia.org) | Free (member countries) |

### Crowd-Sourced / Participatory Response

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| HOT Tasking Manager | Global (disaster activations) | Humanitarian OpenStreetMap Team coordinates volunteer mapping of disaster-affected areas: building footprints, roads, camps. Real-time task progress. Results enter OSM. | OSM PBF, Shapefile, GeoJSON (via Overpass / HOT Export Tool) | [tasks.hotosm.org](https://tasks.hotosm.org) -- tasks + data via [export.hotosm.org](https://export.hotosm.org) | Free / ODbL |
| Ushahidi Platforms | Global (instance-specific) | Open-source crowd-sourcing platform used by responders to collect geocoded incident reports (damage, needs, road closures). Multiple independent deployments per disaster. API available. | GeoJSON, CSV, KML | [Ushahidi](https://www.ushahidi.com) -- platform is open source; individual deployment APIs vary | Free (open source) |
| Crisis Mappers / MapAction | Global | MapAction provides professional crisis mapping and GIS support to disaster response: standardized map products, data management, coordination. | PDF maps, SHP | [MapAction](https://mapaction.org) | Free (products publicly released) |
| iHEAT (iMMAP Humanitarian Emergency Affairs Tool) | Global | Multi-hazard situation awareness and needs analysis for humanitarian coordination. | Web platform, limited API | [iMMAP](https://immap.org) | Restricted (humanitarian organizations) |

### Open Humanitarian Data

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| Humanitarian Data Exchange (HDX) | Global | OCHA's open data platform for humanitarian response: 19,000+ datasets covering disasters, population, infrastructure, health. CKAN-based. | CSV, SHP, GeoTIFF, JSON | [data.humdata.org](https://data.humdata.org) -- [HDX API](https://data.humdata.org/api) CKAN-compatible | Free |
| Fieldmaps | Global | Mobile-first offline mapping platform for humanitarian field data collection. GeoJSON output. | GeoJSON | [fieldmaps.io](https://fieldmaps.io) | Free |

## Historical Disaster Records

Long-term event databases used for loss modeling, risk assessment, and trend analysis.

### EM-DAT (Emergency Events Database / CRED)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| EM-DAT | Global, 1900-present | CRED's International Disaster Database: ~26,000 disasters with onset date, country, event type (natural + technological), deaths, affected, injured, homeless, economic damage (2021 USD). Methodologically consistent since 1988. Updated continuously. | Excel, CSV | [EM-DAT](https://www.emdat.be) -- requires free registration; full dataset download | Free (registration) / Practical |
| EM-DAT API | Global | Programmatic access to EM-DAT records by country, disaster type, date range. | JSON, CSV | [EM-DAT API](https://api.emdat.be/docs) -- free key after registration | Free |

### NatCatSERVICE (Munich Re)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| NatCatSERVICE | Global, 1980-present | Munich Re's proprietary natural catastrophe database: 45,000+ events with insured and total economic losses. Annual loss statistics and trends freely available; full database by subscription. Country and peril breakdowns. | PDF reports, Excel (selected data) | [Munich Re NatCatSERVICE](https://www.munichre.com/en/solutions/for-industry-clients/natcatservice.html) -- annual statistics free; full DB commercial | Free (summaries) / Commercial (full DB) |

### Sigma Explorer (Swiss Re Institute)

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| Sigma Explorer | Global, 1970-present | Swiss Re Institute's sigma catastrophe database: natural and man-made events with insured and economic losses (inflation-adjusted), fatalities. Interactive map and annual sigma reports. Similar scope to NatCatSERVICE. | Web explorer, Excel export | [Sigma Explorer](https://www.sigma-explorer.com) -- interactive tool free; raw data commercial | Free (explorer) / Commercial (raw data) |

### DesInventar

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| DesInventar Sendai | Global (70+ countries, emphasis Latin America, Asia) | Subnational disaster loss database: local-level event records with deaths, affected, housing damage, crops, infrastructure. Time series to 1970s for some countries. Used to implement Sendai Framework local monitoring. Integrates into UNDRR's Sendai Monitor. | CSV, Shapefile, API | [desinventar.net](https://www.desinventar.net) -- free download per country; [Sendai platform](https://www.desinventar.org) | Free |

### Geocoded Datasets

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| GDIS (Global Disaster Impact and Susceptibility) | Global, 1960-2018 | Spatially explicit (GADM-matched) version of EM-DAT records: subnational administrative unit affected, event type, impact. Published by Columbia University / Palisade-IRI. | CSV + Shapefile (GADM) | [GDIS on Zenodo](https://zenodo.org/record/4765737) | Free |
| SHELDUS (Spatial Hazard Events and Losses Database for the United States) | USA (county level) | US county-level losses from 18 natural hazard types since 1960: property damage, crop damage, injuries, fatalities. Monthly temporal resolution. | CSV | [ASU SHELDUS](https://cemhs.asu.edu/sheldus) -- download requires free registration | Free (registration) |
| Geocoded Disasters (GDIS) on HDX | Global | The GDIS dataset published on OCHA HDX for easy discovery alongside other humanitarian data. | CSV | [GDIS on HDX](https://data.humdata.org/dataset/geocoded-disasters) | Free |

## Recommended Tools

Tools commonly used when working with natural disaster and emergency response geospatial data.

| Tool | Type | Purpose | License | URL |
|------|------|---------|---------|-----|
| [QGIS](https://www.qgis.org) | Desktop GIS | Visualize and process all disaster data formats (SHP, GeoTIFF, GeoJSON, GDB); plugins for QuakeML, GeoRSS ingestion | GPL-2.0 | [qgis.org](https://www.qgis.org) |
| [GeoPandas](https://geopandas.org) | Python library | Load, query, and join disaster vector data; dissolve, overlay, and reproject | BSD-3 | [geopandas.org](https://geopandas.org) |
| [xarray](https://xarray.dev) | Python library | Work with NetCDF flood, drought, and surge model outputs in labeled multi-dimensional arrays | Apache-2.0 | [xarray.dev](https://xarray.dev) |
| [GDAL / OGR](https://gdal.org) | CLI + library | Convert between all raster and vector formats; reproject, clip, mosaic | MIT | [gdal.org](https://gdal.org) |
| [obspy](https://www.obspy.org) | Python library | Parse QuakeML, StationXML, and waveform data from seismic networks; query FDSN web services | LGPL-3.0 | [obspy.org](https://www.obspy.org) |
| [Fiona](https://github.com/Toblerity/Fiona) | Python library | Read/write OGC vector formats (Shapefile, GeoPackage, GeoJSON); underpins GeoPandas | BSD-3 | [Fiona on GitHub](https://github.com/Toblerity/Fiona) |
| [rasterio](https://rasterio.readthedocs.io) | Python library | Read/write GeoTIFF and other rasters; windowed reads for large flood/fire grids | BSD-3 | [rasterio.readthedocs.io](https://rasterio.readthedocs.io) |
| [Google Earth Engine](https://earthengine.google.com) | Cloud platform | Access FIRMS fire data, MODIS flood products, GRACE groundwater, and 900+ disaster-relevant datasets at scale | Free (research) | [earthengine.google.com](https://earthengine.google.com) |
| [ArcGIS Living Atlas](https://livingatlas.arcgis.com) | Cloud platform | Curated disaster layers including real-time earthquakes, NHC forecast tracks, FEMA NFHL, and Copernicus EMS products as hosted feature layers | Commercial (ESRI license) | [livingatlas.arcgis.com](https://livingatlas.arcgis.com) |
| [Kepler.gl](https://kepler.gl) | Web app / Python | Visualize large point datasets (FIRMS fires, earthquake catalogs, DART buoys) directly in browser | MIT | [kepler.gl](https://kepler.gl) |
| [PyGMT](https://www.pygmt.org) | Python library | Make publication-quality maps of seismic focal mechanisms, tsunami propagation, hurricane tracks | BSD-3 | [pygmt.org](https://www.pygmt.org) |
| [Hazpy](https://github.com/nhrap-hazus/hazpy) | Python library | FEMA HAZUS interface for earthquake, flood, and hurricane loss estimation workflows | BSD-3 | [nhrap-hazus/hazpy](https://github.com/nhrap-hazus/hazpy) |
| [CliMetLab](https://climetlab.readthedocs.io) | Python library | Unified access to meteorological and climate datasets including ECMWF forecasts used for tropical cyclone guidance | Apache-2.0 | [climetlab.readthedocs.io](https://climetlab.readthedocs.io) |
| [tcrm](https://github.com/GeoscienceAustralia/tcrm) | Python tool | Tropical Cyclone Risk Model from Geoscience Australia: statistical-parametric TC track and wind field simulation | Apache-2.0 | [GeoscienceAustralia/tcrm](https://github.com/GeoscienceAustralia/tcrm) |
