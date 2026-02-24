# Logistics, Delivery & Urban Mobility - Geospatial Data Sources

A comprehensive guide to geospatial datasets powering logistics, delivery networks, ride-hailing, freight, and urban mobility applications.

---

## Quick Picks

| Use Case | Recommended Source | Why |
|---|---|---|
| Free global geocoding | **OpenAddresses** + **Nominatim** | 500M+ addresses; OSM-quality reverse geocoding |
| Self-hosted routing | **OSRM** or **Valhalla** | Blazing fast, fully open-source, OSM-based |
| POI / Places (open) | **Overture Maps Places** | 72M+ POIs, monthly updates, open license |
| POI / Places (commercial) | **Foursquare Places** | 120M+ POIs, rich attributes (hours, popularity) |
| Ride-hailing trip records | **NYC TLC Trip Data** | Billions of rows, monthly updates, truly open |
| Transit feeds | **Mobility Database** (successor to OpenMobilityData) | 6,000+ GTFS/GBFS feeds worldwide |
| Ship / freight tracking | **Global AIS (MarineTraffic / AISHub)** | Real-time vessel positions globally |
| Bike / scooter share | **GBFS feeds via MobilityData** | 1,250+ systems, real-time availability |
| Routing API (managed) | **Mapbox Directions** or **HERE Routing** | Generous free tiers, truck/bike profiles |

---

## Table of Contents

1. [Address & Geocoding](#1-address--geocoding)
2. [Routing & Navigation Engines](#2-routing--navigation-engines)
3. [Points of Interest for Delivery](#3-points-of-interest-for-delivery)
4. [Last-Mile Delivery Optimization](#4-last-mile-delivery-optimization)
5. [Ride-Hailing & Taxi Data](#5-ride-hailing--taxi-data)
6. [Public Transit](#6-public-transit)
7. [Freight & Shipping](#7-freight--shipping)
8. [Warehousing & Distribution Centers](#8-warehousing--distribution-centers)
9. [Food Delivery Specific](#9-food-delivery-specific)
10. [Bike & Micromobility](#10-bike--micromobility)

---

## 1. Address & Geocoding

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **OpenAddresses** | 500M+ global addresses with street number, street name, city, postcode, lon/lat. Sourced from national/state/local governments. | CSV (per country/region), bulk download | Free, open data (most sources require attribution only) | [openaddresses.io](https://openaddresses.io/) / [results.openaddresses.io](https://results.openaddresses.io/) |
| **Google Geocoding API** | Forward & reverse geocoding, address validation, place IDs. Global coverage. | JSON / XML via REST API | Pay-as-you-go. Essentials tier: 10,000 free events/month. ~$5 per 1,000 requests after that. | [developers.google.com/maps/documentation/geocoding](https://developers.google.com/maps/documentation/geocoding) |
| **Nominatim (OSM)** | Forward & reverse geocoding using OpenStreetMap data. Global coverage, multi-language. | JSON / XML via REST API | Free (self-hosted or public instance at nominatim.openstreetmap.org). Public instance: max 1 req/sec, no bulk use. | [nominatim.org](https://nominatim.org/) |
| **Pelias** | Modular open-source geocoder. Imports from OSM, OpenAddresses, Who's on First, Geonames. Fast autocomplete, multi-language. | JSON (GeoJSON) via REST API | Free, self-hosted. Commercial hosting via Geocode Earth (~$50+/mo). Part of Linux Foundation. | [pelias.io](https://pelias.io/) / [github.com/pelias](https://github.com/pelias/pelias) |
| **Photon** | Open-source geocoder for OSM data. Search-as-you-type, multi-language. Uses Elasticsearch/OpenSearch backend. | GeoJSON via REST API | Free, self-hosted. Planet-wide DB ~95 GB disk, recommended 64 GB RAM. Public demo at photon.komoot.io. | [github.com/komoot/photon](https://github.com/komoot/photon) |
| **What3Words** | Divides globe into 3m x 3m squares, each assigned a unique 3-word address. Forward/reverse lookup. | JSON via REST API | Free plan available (low volume). Paid plans for commercial use. Free for emergency services and charities. | [what3words.com](https://accounts.what3words.com/select-plan) |
| **USPS Address Information System** | US postal addresses, ZIP+4 codes, delivery point data. Authoritative USPS source. | Flat files, API (Web Tools) | Free API registration. Bulk AIS product subscription ~$3,690/year. | [usps.com/business/web-tools-apis](https://www.usps.com/business/web-tools-apis/) |
| **Royal Mail PAF (UK)** | UK postal addresses, 30M+ delivery points, postcodes. The definitive UK address database. | CSV, fixed-width, various licensed formats | Commercial license required. Pricing varies by use case (~GBP 1,000+/year for developers). | [royalmail.com/business/services/marketing/data-optimisation/paf](https://www.royalmail.com/business/services/marketing/data-optimisation/paf) |
| **Geocode Earth** (hosted Pelias) | Managed Pelias geocoding service. All Pelias data sources combined. SLA-backed. | JSON (GeoJSON) via REST API | Starts at ~$50/month for 100K requests. Free trial available. | [geocode.earth](https://geocode.earth/) |

---

## 2. Routing & Navigation Engines

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **OSRM (Open Source Routing Machine)** | Fastest open-source routing engine. Car, bike, foot profiles. Uses Contraction Hierarchies (CH). Supports route, table (distance matrix), match, trip APIs. | JSON via REST API | Free, self-hosted (BSD-2 license). Demo server available. Server costs are the only expense. Best for continental-scale matrix queries. | [project-osrm.org](http://project-osrm.org/) / [github.com/Project-OSRM](https://github.com/Project-OSRM/osrm-backend) |
| **Valhalla** | Open-source multi-modal routing engine. Car, bike, pedestrian, transit. Dynamic costing, isochrones, map-matching, elevation-aware. Time-aware routing. | JSON via REST API (Protobuf optional) | Free, self-hosted (MIT license). Low memory footprint, works on lower-spec devices. | [github.com/valhalla/valhalla](https://github.com/valhalla/valhalla) |
| **GraphHopper** | Fast, memory-efficient routing. Car, bike, foot, custom profiles. CH and Landmark algorithms. Turn costs, alternative routes. | JSON via REST API | Open-source (Apache 2.0) for self-hosting. Hosted API: free tier 500 req/day; paid from EUR 89/mo. | [graphhopper.com](https://www.graphhopper.com/) / [github.com/graphhopper](https://github.com/graphhopper/graphhopper) |
| **Google Directions API** | Turn-by-turn directions, traffic-aware routing, multi-modal (driving, walking, cycling, transit). Global coverage with live traffic. | JSON / XML via REST API | Essentials tier: free threshold per SKU. ~$5 per 1,000 route requests (Directions), ~$5 per 1,000 elements (Distance Matrix). | [developers.google.com/maps/documentation/directions](https://developers.google.com/maps/documentation/directions) |
| **HERE Routing API** | Car, truck, bicycle, pedestrian routing. Truck-specific attributes (weight, height, hazmat). Isoline routing, EV range. | JSON via REST API | Freemium: 30,000 routing transactions/month free (no credit card required). Then $0.75 per 1,000. Pro plan $449/mo for 1M transactions. | [developer.here.com/products/routing](https://developer.here.com/products/routing) |
| **Mapbox Directions API** | Driving, walking, cycling directions. Traffic-aware, turn-by-turn, voice guidance. Navigation SDK. | JSON (GeoJSON) via REST API | Free: up to 100,000 directions requests/month (web). Navigation SDK: 100 MAU + 1,000 trips free. Pay-as-you-go after. | [docs.mapbox.com/api/navigation/directions](https://docs.mapbox.com/api/navigation/directions/) |
| **openrouteservice** | Routing, isochrones, matrix, geocoding built on OSM data. Car, HGV, bike, wheelchair profiles. | JSON via REST API | Free: 2,000 req/day (API key). Self-hosted also available (LGPL). | [openrouteservice.org](https://openrouteservice.org/) |

### Routing Engine Selection Guide

| Criterion | OSRM | Valhalla | GraphHopper |
|---|---|---|---|
| Speed (single route) | Fastest | Fast | Very fast |
| Distance matrix | Best at continental scale | Good | Good at national scale |
| Memory usage | Higher (pre-processing) | Lower (dynamic) | Moderate |
| Traffic / time-aware | Limited | Yes (dynamic costing) | Via GraphHopper Maps |
| Isochrones | Via plugin | Built-in | Built-in |
| Truck routing | Limited | Yes | Yes (commercial) |
| License | BSD-2 | MIT | Apache 2.0 |

---

## 3. Points of Interest for Delivery

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **Overture Maps Places** | 72M+ global POIs from Meta, Microsoft, and community sources. Name, category, address, coordinates, websites, phone. Monthly releases. | GeoParquet (primary), GeoJSON | Free, open (CDLA Permissive 2.0). Download via S3, Azure, Python CLI, DuckDB. Latest release: 2026-01-21.0 (~7.2 GB Parquet). | [docs.overturemaps.org/guides/places](https://docs.overturemaps.org/guides/places/) |
| **Foursquare Places** | 120M+ POIs across 200+ countries. 50+ attributes: name, category, hours, chain info, popularity, photos, tips, social media. | JSON via API, bulk CSV/Parquet | Commercial license. API: $18.75 CPM for V3 endpoints. Bulk data licensing negotiated. Open-source tier for basic POI attributes. | [foursquare.com/products/places](https://foursquare.com/products/places/) |
| **Google Places API** | Comprehensive global POIs. Ratings, reviews, photos, hours, contact info, price level, accessibility. | JSON via REST API | Free threshold ~28,000 requests/month. Tiered by fields requested (IDs Only / Basic / Advanced / Preferred). Basic ~$5/1K, Advanced ~$7/1K. | [developers.google.com/maps/documentation/places](https://developers.google.com/maps/documentation/places) |
| **HERE Places** | Global POI database. Categories, hours, ratings, chain info. Integrated with HERE routing for logistics use cases. | JSON via REST API | Included in HERE Freemium (250K transactions/month across all APIs). Dedicated Places: 30K free/month. | [developer.here.com/products/geocoding-and-search](https://developer.here.com/products/geocoding-and-search) |
| **SafeGraph Places** | 49M+ global POIs with 24 attributes, 400+ categories. Foot traffic patterns, spend data, brand info. Now part of Dewey (data marketplace). | CSV, Parquet via data marketplace | Commercial license. Pricing varies by dataset and volume. Academic access programs available. | [safegraph.com/products/places](https://www.safegraph.com/products/places) |
| **OpenStreetMap POIs** | Community-mapped global POIs. Shops, restaurants, ATMs, fuel stations, public facilities. Variable completeness by region. | OSM PBF, GeoJSON (via Overpass API) | Free, open (ODbL). Extract via Overpass API, Geofabrik downloads, or OSM-based tools. | [overpass-turbo.eu](https://overpass-turbo.eu/) |

---

## 4. Last-Mile Delivery Optimization

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **OpenStreetMap Building Entrances** | Building entrance nodes tagged `entrance=*`. Main, service, emergency entrances with coordinates. Variable global coverage. | OSM PBF / GeoJSON via Overpass API | Free, open (ODbL). Query via Overpass: `node["entrance"](bbox)`. | [wiki.openstreetmap.org/wiki/Key:entrance](https://wiki.openstreetmap.org/wiki/Key:entrance) |
| **OpenStreetMap Indoor Mapping** | Indoor floor plans using Simple Indoor Tagging. Rooms, corridors, elevators, stairs. Growing coverage in malls, airports, hospitals. | OSM PBF / GeoJSON | Free, open (ODbL). Tags: `indoor=*`, `level=*`. | [wiki.openstreetmap.org/wiki/Indoor_Mapping](https://wiki.openstreetmap.org/wiki/Indoor_Mapping) |
| **US ZIP Code Boundaries (ZCTA)** | US Census Bureau ZIP Code Tabulation Areas. Polygon boundaries approximating USPS ZIP codes. | Shapefile, GeoJSON, KML | Free from US Census TIGER/Line. Updated with each decennial census. | [census.gov/geographies/mapping-files](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) |
| **UK Postcode Boundaries** | Postcode district, sector, and unit polygons for the UK. Derived from OS Open Data. | GeoJSON, Shapefile, KML | Free (OS OpenData). Detailed unit-level from Ordnance Survey (commercial). Community datasets on GitHub. | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk/) |
| **GeoPostcodes** | Global postal code boundaries for 250+ countries. Polygons, centroids, admin hierarchy. | Shapefile, CSV, GeoJSON, SQL | Commercial. Pricing varies by country count (single country ~$500+, global ~$10,000+). | [geopostcodes.com](https://www.geopostcodes.com/) |
| **Overture Maps Buildings** | 2.3B+ building footprints globally (from Microsoft, OSM, Google). Height, class, name attributes where available. | GeoParquet, GeoJSON | Free, open (CDLA Permissive 2.0). Same access methods as Overture Places. | [docs.overturemaps.org/guides/buildings](https://docs.overturemaps.org/guides/buildings/) |
| **Google Address Validation API** | Sub-locality, landmarks, access points near delivery addresses. Enhances last-mile navigation context. | JSON via REST API | Part of Google Maps Platform pricing. Address Validation ~$5 per 1,000 requests. | [developers.google.com/maps/documentation/address-validation](https://developers.google.com/maps/documentation/address-validation) |

---

## 5. Ride-Hailing & Taxi Data

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **NYC TLC Trip Records** | All taxi (yellow, green) and FHV (Uber, Lyft, etc.) trips in NYC. Pickup/dropoff times, locations (taxi zones), fares, tips, distance, passenger count. Billions of records since 2009. Updated monthly (~2-month lag). | Parquet (current), CSV (legacy) | Free, open data. Direct download from NYC TLC website. ~2-3 GB per month of data. | [nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Chicago TNP Trips (2025+)** | All Transportation Network Provider (Uber/Lyft) trips in Chicago. Pickup/dropoff census tracts, times (rounded to 15 min), fares (rounded to $2.50), tips (rounded to $1). Privacy-protected. | CSV via Chicago Data Portal (Socrata API) | Free, open data. API access available. Historical data also available (2018-2022, 2023-2024 in separate datasets). | [data.cityofchicago.org/d/6dvr-xwnh](https://data.cityofchicago.org/d/6dvr-xwnh) |
| **Chicago Taxi Trips** | Individual taxi trips since 2013. Pickup/dropoff community areas, fare, tip, company, duration. 200M+ records. | CSV via Chicago Data Portal | Free, open data. | [data.cityofchicago.org (Taxi Trips)](https://data.cityofchicago.org/Transportation/Taxi-Trips/wrvz-psew) |
| **Uber Movement** | Historical travel time data between zones in select cities worldwide. | CSV (archived) | **Discontinued.** The platform has been shut down. Some archived datasets remain on Kaggle and academic repositories. Original data no longer available from Uber. | N/A (offline) |
| **NYC TLC Factbook Dashboard** | Interactive dashboard visualizing aggregated TLC trip data: trends, market share, EV adoption, daily averages across all FHV sectors. | Web dashboard (no bulk download) | Free, open. | [TLC Factbook (Medium)](https://medium.com/@NYCTLC) |

---

## 6. Public Transit

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **Mobility Database** (successor to OpenMobilityData) | Catalog of 6,000+ GTFS, GTFS-RT, and GBFS feeds across 75+ countries. Searchable, with feed URLs and metadata. Replaced TransitFeeds/OpenMobilityData (deprecated Dec 2025). | Web catalog linking to GTFS ZIP feeds | Free. Maintained by MobilityData (nonprofit). | [mobilitydatabase.org](https://mobilitydatabase.org/) |
| **Transitland** | Community-curated catalog of transit feeds. API for querying stops, routes, operators, schedules. Links to GTFS source feeds. | GTFS ZIP files; JSON/GeoJSON API | Free, open-source. API available at transit.land. | [transit.land](https://www.transit.land/) |
| **GTFS.org** | The official GTFS specification and resources hub. Links to tools, validators, and data sources for static and real-time transit feeds. | GTFS (ZIP of CSV files), GTFS-RT (Protocol Buffers) | Free. Specification maintained by MobilityData. | [gtfs.org](https://gtfs.org/) |
| **US National Transit Database (NTD)** | FTA-reported data from 900+ US transit agencies. Ridership, finances, service, fleet, safety. Not spatial per se but joinable to GTFS. | Excel, CSV via NTD Data Portal | Free. Updated annually. | [transit.dot.gov/ntd](https://www.transit.dot.gov/ntd) |
| **European NAPs (National Access Points)** | EU member states publish transit data via National Access Points per EU Delegated Regulation 2017/1926. Links to national GTFS/NeTEx feeds. | GTFS, NeTEx (XML), SIRI (real-time) | Free. Varies by country. | [transport.ec.europa.eu](https://transport.ec.europa.eu/) |
| **GTFS-RT (Real-Time)** | Real-time updates to GTFS: vehicle positions, trip updates (delays), service alerts. Protocol Buffers format. | Protocol Buffers (protobuf) | Free where published by agencies. Availability varies by agency. | [gtfs.org/realtime](https://gtfs.org/documentation/realtime/reference/) |

### GTFS Data Structure (Key Files)

| File | Contents |
|---|---|
| `agency.txt` | Transit agency info |
| `routes.txt` | Route definitions (name, type, color) |
| `trips.txt` | Individual trips on routes |
| `stop_times.txt` | Arrival/departure at each stop |
| `stops.txt` | Stop locations (lat/lon, name) |
| `shapes.txt` | Route geometry (polylines) |
| `calendar.txt` | Service days |
| `frequencies.txt` | Headway-based service (expanding support from 2026) |

---

## 7. Freight & Shipping

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **MarineTraffic** | Real-time and historical AIS vessel tracking. Ship positions, routes, port calls, vessel details. 200K+ vessels tracked. | Web platform, API (JSON/XML), CSV exports | Free tier (limited live map). API plans from ~$100/mo. Historical data and analytics: commercial. | [marinetraffic.com](https://www.marinetraffic.com/) |
| **AISHub** | Community AIS data exchange network. Free real-time vessel positions via data sharing. | JSON, XML, KML | Free if you contribute AIS data from your receiver. Otherwise read-only access is limited. | [aishub.net](https://www.aishub.net/) |
| **World Port Index (NGA)** | 3,700+ ports worldwide. Location, facilities, harbor characteristics, services. Published by US National Geospatial-Intelligence Agency. | Shapefile, CSV, Access DB | Free download from NGA Maritime Safety portal. | [msi.nga.mil/Publications/WPI](https://msi.nga.mil/Publications/WPI) |
| **UN/LOCODE** | UN code list for trade and transport locations. 100,000+ locations in 250 countries. Port, airport, road terminal codes with coordinates. | CSV, text files | Free download from UNECE. Updated biannually. | [unece.org/trade/cefact/unlocode](https://unece.org/trade/cefact/unlocode-code-list-country-and-territory) |
| **FreightWaves SONAR** | Proprietary freight market intelligence. Truckload/LTL rates, tender volumes, capacity, fuel prices. 250+ indicators. | Web dashboard, API, CSV export | Commercial subscription. Pricing starts ~$500+/month. Enterprise licensing. | [freightwaves.com/sonar](https://www.freightwaves.com/sonar) |
| **BTS Freight Analysis Framework (FAF)** | US domestic freight flows by mode (truck, rail, water, air, pipeline), commodity, origin/destination. Tonnage and value. | CSV, Shapefile (FAF regions) | Free from Bureau of Transportation Statistics. Updated ~every 5 years with annual provisional estimates. | [bts.gov/faf](https://www.bts.gov/faf) |
| **Kpler** | Commercial maritime intelligence. AIS-based cargo tracking, commodity flows, fleet analytics. | API, web platform | Commercial. Enterprise pricing. | [kpler.com](https://www.kpler.com/) |

---

## 8. Warehousing & Distribution Centers

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **HIFLD (Homeland Infrastructure Foundation-Level Data)** | US critical infrastructure including warehouses, distribution centers, and logistics facilities. Point locations with attributes. 500+ dataset layers total. | Shapefile, CSV, KML, File Geodatabase | Free (was open via ArcGIS Hub). **Note:** HIFLD Open portal went offline Aug 2025. Archived snapshot at DataLumos (415 layers preserved). Access now via DHS GII for authorized users. | [datarescueproject.org (archived)](https://www.datarescueproject.org/hifld-data-saved/) |
| **OpenStreetMap Warehouses** | OSM-tagged warehouses and industrial buildings. Tags: `building=warehouse`, `landuse=industrial`. Variable completeness. | OSM PBF / GeoJSON via Overpass API | Free, open (ODbL). Query: `way["building"="warehouse"](bbox)`. | [overpass-turbo.eu](https://overpass-turbo.eu/) |
| **CoStar / CBRE / Prologis** | Commercial real estate databases with warehouse and distribution center listings. Square footage, vacancy, lease rates, locations. | Web portals, API (commercial clients) | Commercial subscription. CoStar from ~$500/mo. CBRE/Prologis reports often free but data access is commercial. | [costar.com](https://www.costar.com/) |
| **US Census County Business Patterns** | Establishment counts by NAICS code (493 = Warehousing & Storage) per county/ZIP. Employment size classes. | CSV, API | Free from Census Bureau. Annual updates. | [census.gov/programs-surveys/cbp.html](https://www.census.gov/programs-surveys/cbp.html) |

---

## 9. Food Delivery Specific

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **Overture Maps Places (restaurant filter)** | Filter Overture Places by food/restaurant categories. Name, address, coordinates, category. | GeoParquet, GeoJSON | Free, open. Filter by category during download. | [docs.overturemaps.org/guides/places](https://docs.overturemaps.org/guides/places/) |
| **Foursquare Places (restaurant data)** | Detailed restaurant POIs with hours, popularity scores, tips, photos, cuisine type, price level. | JSON API, bulk licensing | Commercial (see Section 3 pricing). Rich attributes for delivery platform analytics. | [foursquare.com/products/places](https://foursquare.com/products/places/) |
| **OSM Restaurants & Food** | Community-mapped restaurants, cafes, fast food, food courts. Tags: `amenity=restaurant`, `cuisine=*`. | OSM PBF / GeoJSON | Free, open (ODbL). | [wiki.openstreetmap.org](https://wiki.openstreetmap.org/) |
| **Yelp Fusion API** | Business listings with reviews, ratings, hours, photos. Strong in US, Canada, UK. Restaurant-heavy dataset. | JSON via REST API | Free: 5,000 API calls/day. No bulk download. Display requirements apply. | [yelp.com/developers](https://www.yelp.com/developers) |
| **USDA Food Environment Atlas** | US county-level food access indicators: restaurant density, grocery store proximity, food deserts, SNAP retailers. | CSV, Excel, interactive map | Free from USDA Economic Research Service. | [ers.usda.gov/data-products/food-environment-atlas](https://www.ers.usda.gov/data-products/food-environment-atlas/) |
| **Ghost / Cloud Kitchen Data** | No single authoritative open dataset exists for cloud kitchens. Best approach: combine Foursquare, Google Places, and local business registries. | Varies | Varies. Delivery platforms do not publish kitchen location data publicly. | N/A |

---

## 10. Bike & Micromobility

| Source | Data Contents | Format | Access / Cost | URL |
|---|---|---|---|---|
| **GBFS Feeds (General Bikeshare Feed Specification)** | Real-time data from 1,250+ shared bike/scooter/moped systems worldwide. Vehicle locations, station status, availability, pricing, geofencing zones. | JSON (GBFS standard) | Free, real-time, read-only. Feed catalog maintained by MobilityData. | [gbfs.org](https://gbfs.org/) / [gbfs.mobilitydata.org](https://gbfs.mobilitydata.org/) |
| **MDS (Mobility Data Specification)** | City-focused standard for regulating shared mobility. Trip data, vehicle status events, geographic policies. More granular than GBFS. | JSON via API | Access typically restricted to city regulators. Operators share data with municipal agencies per local ordinance. | [github.com/openmobilityfoundation](https://github.com/openmobilityfoundation/mobility-data-specification) |
| **NABSA Shared Mobility Data** | North American Bikeshare & Scootershare Association data hub. Ridership statistics, system profiles. | Reports (PDF), some CSV | Free reports. Detailed data via member agencies. | [nabsa.net/resources/data](https://nabsa.net/resources/data/) |
| **OSM Cycling Infrastructure** | Bike lanes, cycle tracks, bicycle parking, repair stations tagged in OSM. Tags: `cycleway=*`, `bicycle=*`, `amenity=bicycle_parking`. | OSM PBF / GeoJSON via Overpass API | Free, open (ODbL). Best extracted via Overpass or specialized tools like CyclOSM. | [wiki.openstreetmap.org/wiki/Bicycle](https://wiki.openstreetmap.org/wiki/Bicycle) |
| **CyclOSM** | Cycling-focused rendering of OSM data. Visualizes bike infrastructure types, surfaces, elevation. | Tile service (raster), data from OSM | Free, open-source. Tiles at cyclosm.org. Data via OSM download. | [cyclosm.org](https://www.cyclosm.org/) |
| **Strava Metro** | Aggregated, anonymized movement data from Strava users. Cycling and pedestrian activity counts on road segments. | CSV, Shapefile via partnership | Commercial. Free for city transportation agencies (apply to Strava Metro program). | [metro.strava.com](https://metro.strava.com/) |
| **Citi Bike / Divvy / Bay Wheels Trip Data** | Individual trip records from major US bikeshare systems. Start/end station, time, duration, user type. | CSV (monthly/quarterly downloads) | Free, open data. Published by Lyft (system operator). | [citibikenyc.com/system-data](https://citibikenyc.com/system-data) |

---

## Notes & Tips

- **Self-hosting vs. managed APIs**: For startups and prototyping, managed APIs (Google, HERE, Mapbox) offer speed. For production at scale, self-hosted engines (OSRM, Valhalla, Pelias) eliminate per-request costs.
- **Overture Maps** is rapidly becoming the go-to open alternative to Google/Foursquare for POI data. Monthly releases are improving quality significantly.
- **GTFS is the universal language of transit data.** Nearly every public transit agency worldwide publishes GTFS. The Mobility Database is now the canonical catalog.
- **Privacy matters**: Ride-hailing datasets (NYC TLC, Chicago) apply deliberate obfuscation (rounding times, aggregating zones) to protect rider privacy.
- **HIFLD data rescue**: Since the HIFLD open portal went offline in August 2025, check the DataLumos archive for the 415 preserved layers if you need US infrastructure facility locations.

---

*Last updated: February 2026*
