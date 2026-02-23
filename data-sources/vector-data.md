# Vector Data

Curated vector datasets for GIS analysis, mapping, and application development. Covers global community-driven projects, authoritative government sources, and thematic datasets.

> **Quick Picks**
> - **SOTA**: [Overture Maps Foundation](https://overturemaps.org) -- merged OSM+Microsoft+Meta+Google data in GeoParquet, updated monthly
> - **Free Best**: [OpenStreetMap via Geofabrik](https://download.geofabrik.de) -- the most complete free global vector dataset
> - **Fastest Setup**: [Natural Earth](https://naturalearthdata.com) -- instant download, public domain, perfect for basemaps

## Overture Maps Foundation

The Overture Maps Foundation (Linux Foundation) provides open, interoperable, and quality-assured map data by combining OSM, Microsoft, Meta, and other sources. Data is released monthly in GeoParquet format.

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| Overture Places | Global (53M+ POIs) | GeoParquet | CDLA Permissive 2.0 | [overturemaps.org](https://overturemaps.org) | SOTA |
| Overture Buildings | Global (2.3B+ footprints) | GeoParquet | ODbL + CDLA | [overturemaps.org](https://overturemaps.org) | SOTA |
| Overture Transportation | Global (road network) | GeoParquet | ODbL | [overturemaps.org](https://overturemaps.org) | SOTA |
| Overture Addresses | Global (200M+) | GeoParquet | CDLA Permissive 2.0 | [overturemaps.org](https://overturemaps.org) | SOTA |
| Overture Base (land, water, land use) | Global | GeoParquet | ODbL | [overturemaps.org](https://overturemaps.org) | SOTA |
| Overture Divisions (admin boundaries) | Global | GeoParquet | Various | [overturemaps.org](https://overturemaps.org) | SOTA |

> **Access tip**: Query Overture data directly with DuckDB: `SELECT * FROM read_parquet('s3://overturemaps-us-west-2/release/...')`. See the [Overture docs](https://docs.overturemaps.org) for schema details.

## OpenStreetMap

The world's largest collaborative mapping project, providing freely available vector data for the entire planet.

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| OSM Planet File | Global | PBF, XML | ODbL | [planet.openstreetmap.org](https://planet.openstreetmap.org) | Free |
| Geofabrik Extracts | Regional/Country | SHP, PBF, GeoParquet | ODbL | [download.geofabrik.de](https://download.geofabrik.de) | Free / Practical |
| BBBike Extracts | City-level custom areas | SHP, GeoJSON, PBF, Garmin | ODbL | [extract.bbbike.org](https://extract.bbbike.org) | Free |
| Overpass API | Custom queries (real-time) | JSON, XML, CSV | ODbL | [overpass-turbo.eu](https://overpass-turbo.eu) | Free |
| ohsome API | Historical OSM analytics | JSON, CSV | ODbL | [ohsome.org](https://ohsome.org) | Free / Practical |
| Protomaps OSM Basemap | Global | PMTiles | ODbL | [protomaps.com](https://protomaps.com) | Free / SOTA (tiles) |
| osmium Tool | Processing library | PBF, XML | Open-source | [osmcode.org/osmium-tool](https://osmcode.org/osmium-tool/) | Free |

## Natural Earth

Public domain map dataset available at 1:10m, 1:50m, and 1:110m scales for cartography and GIS.

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| Cultural Vectors (countries, cities, roads) | Global | SHP, GeoJSON, GeoPackage, SQLite | Public Domain | [naturalearthdata.com](https://www.naturalearthdata.com) | Free |
| Physical Vectors (land, ocean, rivers, lakes) | Global | SHP, GeoJSON, GeoPackage, SQLite | Public Domain | [naturalearthdata.com](https://www.naturalearthdata.com) | Free |
| Raster (shaded relief, bathymetry, land cover) | Global | TIFF | Public Domain | [naturalearthdata.com](https://www.naturalearthdata.com) | Free |
| Natural Earth via npm (vector tiles) | Global | TopoJSON, GeoJSON | Public Domain | [github.com/nvkelso/natural-earth-vector](https://github.com/nvkelso/natural-earth-vector) | Free |

## GADM

Database of Global Administrative Areas providing boundaries for all countries and subdivisions.

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| GADM v4.1 | Global (400k+ areas) | SHP, GeoJSON, GPKG, KMZ, R (sp/sf) | Academic/Non-commercial only | [gadm.org](https://gadm.org) | Free (non-commercial) |
| GADM via GEE | Global | FeatureCollection | Academic/Non-commercial | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level0) | Free |

## Google & Microsoft Open Building Footprints

Global-scale building footprint datasets derived from satellite imagery using deep learning.

| Dataset | Coverage | Count | Format | License | URL | Label |
|---------|----------|-------|--------|---------|-----|-------|
| Google Open Buildings v3 | Africa, South/SE Asia, Latin America | 1.8B+ buildings | CSV, GeoJSON | CC BY 4.0 | [sites.research.google/open-buildings](https://sites.research.google/open-buildings/) | Free / SOTA |
| Microsoft Global ML Building Footprints | Global (190+ countries) | 1.2B+ buildings | ODbL | [github.com/microsoft/GlobalMLBuildingFootprints](https://github.com/microsoft/GlobalMLBuildingFootprints) | Free / SOTA |
| Overture Buildings (merged) | Global | 2.3B+ buildings | ODbL + CDLA | [overturemaps.org](https://overturemaps.org) | Free / SOTA |
| Google-Microsoft-OSM Combined (via Vida) | Global | 2.5B+ | GeoParquet, PMTiles | Mixed | [source.coop](https://source.coop) | Free / SOTA |

## Government Open Data Portals

Official government sources for authoritative, high-quality vector datasets.

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| US Census TIGER/Line | USA (roads, rails, water, boundaries) | SHP, GeoJSON | Public Domain | [census.gov/geographies](https://www.census.gov/geographies) | Free |
| Ordnance Survey Open Data | UK (roads, buildings, terrain) | GeoPackage, SHP, GML | OGL v3 | [osdatahub.os.uk](https://osdatahub.os.uk) | Free |
| data.gouv.fr (IGN) | France (admin, cadastre, addresses) | SHP, GeoJSON, GeoPackage | Open License 2.0 | [data.gouv.fr](https://www.data.gouv.fr) | Free |
| GeoBasis-DE (BKG) | Germany (admin, roads, buildings) | SHP, GML, GeoPackage | dl-de/by-2-0 | [gdz.bkg.bund.de](https://gdz.bkg.bund.de) | Free |
| Statistics Canada Boundary Files | Canada (provinces, census tracts) | SHP | Open Government License | [statcan.gc.ca](https://www12.statcan.gc.ca) | Free |
| ABS ASGS | Australia (statistical areas, LGAs) | SHP, GeoJSON, GeoPackage | CC BY 4.0 | [abs.gov.au](https://www.abs.gov.au) | Free |
| Japan GSI Vector Tiles | Japan (buildings, roads, terrain) | GeoJSON, Vector Tiles | CC BY 4.0 | [gsi.go.jp](https://www.gsi.go.jp) | Free |
| Kadaster PDOK | Netherlands (BAG buildings, addresses) | GeoJSON, GeoPackage, WFS | Public Domain | [pdok.nl](https://www.pdok.nl) | Free |
| EU INSPIRE Geoportal | EU member states | GML, GeoJSON (WFS) | Varies | [inspire-geoportal.ec.europa.eu](https://inspire-geoportal.ec.europa.eu) | Free |

## Thematic Datasets

Specialized vector datasets organized by theme.

### Transport

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| Global Roads (gROADS) | Global | SHP | Open | [SEDAC](https://sedac.ciesin.columbia.edu/data/set/groads-global-roads-open-access-v1) | Free |
| OpenFlights (airports, airlines, routes) | Global (10k+ airports) | CSV | ODbL | [openflights.org](https://openflights.org) | Free |
| OpenRailwayMap | Global (from OSM) | GeoJSON, via Overpass | ODbL | [openrailwaymap.org](https://www.openrailwaymap.org) | Free |
| OurAirports | Global (70k+ airports) | CSV | Public Domain | [ourairports.com](https://ourairports.com/data/) | Free |
| Global Shipping Routes (MarineTraffic) | Global | AIS data | Commercial | [marinetraffic.com](https://www.marinetraffic.com) | Commercial |

### Hydrology

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| HydroSHEDS v2 | Global | SHP, GeoTIFF, GeoPackage | Free for non-commercial | [hydrosheds.org](https://www.hydrosheds.org) | Free / SOTA |
| JRC Global Surface Water | Global (1984-2021) | Raster + Vector | Free (Copernicus) | [global-surface-water.appspot.com](https://global-surface-water.appspot.com) | Free |
| NHD (National Hydrography Dataset) | USA | GDB, SHP | Public Domain | [usgs.gov/nhd](https://www.usgs.gov/national-hydrography/national-hydrography-dataset) | Free |
| GloFAS (Global Flood Awareness System) | Global | NetCDF, GeoTIFF | Copernicus License | [globalfloods.eu](https://www.globalfloods.eu) | Free |
| OpenStreetMap Water Features | Global | via Overpass/Geofabrik | ODbL | [overpass-turbo.eu](https://overpass-turbo.eu) | Free |

### Land Use / Land Cover

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| ESA WorldCover 2021 | Global (10m) | COG | CC BY 4.0 | [worldcover2021.esa.int](https://worldcover2021.esa.int) | Free / SOTA |
| Dynamic World (Google) | Global (10m, near-real-time) | GeoTIFF (via GEE) | CC BY 4.0 | [dynamicworld.app](https://dynamicworld.app) | Free / SOTA |
| CORINE Land Cover 2018 | Europe (100m) | SHP, GeoTIFF | Free (Copernicus) | [Copernicus Land Monitoring](https://land.copernicus.eu/pan-european/corine-land-cover) | Free |
| NLCD 2021 | USA (30m) | GeoTIFF | Public Domain | [mrlc.gov](https://www.mrlc.gov) | Free |
| GlobeLand30 | Global (30m, 2000/2010/2020) | GeoTIFF | Free (registration) | [globallandcover.com](http://www.globallandcover.com) | Free |
| ESRI 10m Land Cover | Global (10m, annual) | GeoTIFF | CC BY 4.0 | [livingatlas.arcgis.com](https://livingatlas.arcgis.com/landcover/) | Free |
| Copernicus Global Land Cover | Global (100m) | GeoTIFF | Copernicus License | [Copernicus Land Monitoring](https://land.copernicus.eu/global/) | Free |

### Population & Socioeconomic

| Dataset | Coverage | Format | License | URL | Label |
|---------|----------|--------|---------|-----|-------|
| WorldPop | Global (100m-1km) | GeoTIFF | CC BY 4.0 | [worldpop.org](https://www.worldpop.org) | Free |
| Meta High Resolution Settlement Layer | Global (30m) | GeoTIFF, CSV | CC BY 4.0 | [data.humdata.org](https://data.humdata.org/organization/facebook) | Free / SOTA |
| GPW v4 (Gridded Population of the World) | Global (~1km) | GeoTIFF, NetCDF | CC BY 4.0 | [SEDAC](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4) | Free |
| GHS-POP (Global Human Settlement) | Global (100m-1km) | GeoTIFF | CC BY 4.0 | [ghsl.jrc.ec.europa.eu](https://ghsl.jrc.ec.europa.eu) | Free |
