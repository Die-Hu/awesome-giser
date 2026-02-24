# Data Sources

Comprehensive catalog of free and open geospatial data sources for GIS professionals, researchers, and developers.

> **Quick Picks**
> - **SOTA**: [Copernicus Data Space Ecosystem](https://dataspace.copernicus.eu) -- free Sentinel data with on-demand processing via openEO
> - **Free Best**: [Google Earth Engine](https://earthengine.google.com) -- petabytes of free analysis-ready imagery for research
> - **Fastest Setup**: [Overture Maps](https://overturemaps.org) -- download GeoParquet vector data in minutes via DuckDB or Amazon S3

> **All sources listed are free or have free tiers.** Where relevant, we include SOTA commercial alternatives for comparison so you can make informed decisions.

### 2025-Era Formats and Standards

- **GeoParquet**: Column-oriented geospatial format for fast analytical queries (used by Overture Maps, Microsoft, etc.)
- **Cloud-Optimized GeoTIFF (COG)**: HTTP range-request friendly raster format -- the standard for cloud-native imagery
- **STAC (SpatioTemporal Asset Catalog)**: JSON-based specification for searching and discovering geospatial assets across providers
- **FlatGeobuf**: Streaming-optimized binary vector format for web applications
- **PMTiles**: Single-file tile archive for serverless vector/raster tile hosting

## Table of Contents

### Foundation Geospatial Data
- [Satellite Imagery](satellite-imagery.md) - Optical and radar imagery from Landsat, Sentinel, MODIS, and commercial providers
- [Vector Data](vector-data.md) - OpenStreetMap, Overture Maps, Natural Earth, government open data portals, and thematic datasets
- [Elevation & Terrain](elevation-terrain.md) - Global DEMs (Copernicus 30m, SRTM, ALOS), bathymetry, and LiDAR point clouds
- [Administrative Boundaries](administrative-boundaries.md) - Country borders, subdivisions, GADM, Natural Earth, and standardized boundary datasets
- [Climate & Weather](climate-weather.md) - ERA5 reanalysis, station observations, forecasts, and CMIP6 climate projections

### Industry & Infrastructure
- [Energy & Power Infrastructure](energy-power-infrastructure.md) - Power plants, transmission networks, solar/wind resources, nuclear, oil & gas, EV charging, battery storage
- [Transportation & Autonomous Driving](transportation-autonomous-driving.md) - HD maps, traffic data, LiDAR datasets (Waymo, nuScenes, KITTI), road networks, AV simulation
- [Telecommunications & Connectivity](telecommunications-connectivity.md) - Cell towers, broadband mapping, submarine cables, spectrum allocation, 5G deployment, satellite comms
- [Logistics, Delivery & Urban Mobility](logistics-delivery-urban-mobility.md) - Address/geocoding, routing engines, POI, last-mile delivery, transit/GTFS, freight, food delivery, micromobility
- [Healthcare & Public Health](healthcare-public-health.md) - Facility locations, disease surveillance, environmental health, health access modeling, demographics, EMS

### Environment & Hazards
- [Ocean & Maritime](ocean-maritime.md) - Sea surface temperature, ocean color, AIS ship tracking, ports, coastal data, tides, marine protected areas
- [Natural Disasters & Emergency Response](natural-disasters-emergency-response.md) - Earthquakes, volcanoes, floods, wildfires, hurricanes, tsunamis, drought, multi-hazard platforms, emergency mapping
- [Agriculture & Food Security](agriculture-food-security.md) - Crop mapping, soil data, vegetation indices, irrigation, pest monitoring, precision agriculture, food supply chains

### Emerging & Specialized
- [Space & Aerospace](space-aerospace.md) - Satellite tracking, space weather, airspace/aviation, flight tracking, drone airspace, planetary GIS, NEOs
- [Urban Planning & Smart Cities](urban-planning-smart-cities.md) - Land use/zoning, building footprints, population, night lights, urban heat, real estate, smart city platforms
- [ML Training Data for Geospatial](ml-training-data.md) - SpaceNet, scene classification, object detection, segmentation, foundation models, benchmarks, synthetic data

### Regional
- [China-Specific Data](china-specific.md) - Government portals, coordinate systems (GCJ-02/BD-09), POI data, and academic sources

---

## Key Highlights

- **18 Topic Areas**: From foundational satellite imagery to specialized autonomous driving and space data
- **Free First**: Every section leads with free and open data sources before listing commercial options
- **Detailed Entries**: Each source includes exact data contents, formats, resolution, update frequency, API details, and access method
- **SOTA Comparison**: Includes state-of-the-art commercial alternatives so you understand the full landscape
- **Quick Picks**: Every page has SOTA, Free Best, and Fastest Setup recommendations
- **Recommended Tools**: Each section lists relevant Python libraries, desktop tools, and web platforms
- **Cloud-Native**: Emphasis on STAC catalogs, COG, GeoParquet, and other cloud-optimized formats for modern workflows
