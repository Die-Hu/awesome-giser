# Tools

Essential GIS tools from desktop applications to cloud platforms, covering the full spectrum of geospatial workflows -- from data acquisition to production deployment.

> **Comparison Approach**: Each section highlights SOTA (state-of-the-art) picks alongside free and open-source alternatives, so you can choose the right tool for your budget and use case. Every page includes **Advanced Dark Arts** -- unconventional tricks and expert techniques that save hours.

## Table of Contents

### Desktop & Mobile
- [Desktop GIS](desktop-gis.md) - QGIS, GRASS GIS, ArcGIS Pro, Global Mapper, SAGA + QGIS plugin ecosystem
- [Mobile GIS](mobile-gis.md) - QField, Mergin Maps, ArcGIS Field Maps, KoBoToolbox, ODK, GNSS survey tools

### Programming & Libraries
- [Python Geospatial Libraries](python-libraries.md) - GeoPandas, Shapely 2.0, Rasterio, xarray, folium, leafmap, pydeck, PySAL, MovingPandas, lonboard + advanced tricks
- [Web Mapping & Visualization](web-mapping.md) - MapLibre GL JS, Leaflet, OpenLayers, Deck.gl, CesiumJS, Turf.js, Kepler.gl, D3-geo, React/Vue/Svelte components

### Data Infrastructure
- [CLI Tools](cli-tools.md) - GDAL/OGR, Rasterio CLI, Fiona, tippecanoe, PDAL, mapshaper, osmium + workflow recipes
- [Spatial Databases](spatial-databases.md) - PostGIS, DuckDB Spatial, SpatiaLite, GeoPackage, H3, GeoParquet, FlatGeobuf
- [Geospatial Servers & Publishing](server-publishing.md) - GeoServer, Martin, TiTiler, pg_tileserv, PMTiles, COG serving, OGC API, pygeoapi, Nominatim
- [ETL, Data Engineering & DevOps](etl-data-engineering.md) - FME, Apache Sedona, Airflow/Dagster, Docker recipes, CI/CD, Kart version control, STAC catalogs

### Analysis & Processing
- [Remote Sensing & EO Tools](remote-sensing.md) - SNAP, Orfeo ToolBox, STAC ecosystem, GEE/geemap, openEO, SAR/InSAR, LiDAR, photogrammetry, hyperspectral
- [Geocoding, Routing & Network Analysis](geocoding-routing.md) - Nominatim, Pelias, OSRM, Valhalla, GraphHopper, pgRouting, OSMnx, libpostal
- [AI & Machine Learning for Geospatial](ai-ml-geospatial.md) - TorchGeo, segment-geospatial (SAM), Prithvi-EO, Raster Vision, PySAL, scikit-learn, YOLO, point cloud ML

### Visualization & Cartography
- [3D GIS, Visualization & Cartography](3d-visualization.md) - CesiumJS, Blender GIS, QGIS 3D, 3D Tiles, Kepler.gl, pydeck, rayshader, scrollytelling, print cartography
- [Cloud Platforms](cloud-platforms.md) - Google Earth Engine, Planetary Computer, AWS Earth Search, Wherobots, Fused.io, Mapbox, CARTO

---

## How to Choose

| Need | Recommended Starting Point |
|------|---------------------------|
| General-purpose GIS analysis | Desktop GIS → [QGIS 3.36+](desktop-gis.md) |
| Python spatial analysis | [GeoPandas + Shapely 2.0](python-libraries.md) |
| Batch processing & automation | [GDAL 3.9+ / tippecanoe / PDAL](cli-tools.md) |
| Large dataset storage & queries | [PostGIS](spatial-databases.md) (production) or [DuckDB Spatial](spatial-databases.md) (analytics) |
| Interactive web maps | [MapLibre GL JS](web-mapping.md) (modern) or [Leaflet](web-mapping.md) (simple) |
| Planetary-scale remote sensing | [Google Earth Engine + geemap](cloud-platforms.md) |
| Satellite image processing | [SNAP + Sen2Cor](remote-sensing.md) (desktop) or [openEO](remote-sensing.md) (cloud) |
| AI/ML on satellite imagery | [TorchGeo](ai-ml-geospatial.md) (training) or [segment-geospatial](ai-ml-geospatial.md) (zero-shot) |
| Tile serving (vector) | [Martin](server-publishing.md) (fast) or [PMTiles](server-publishing.md) (serverless) |
| Tile serving (raster) | [TiTiler](server-publishing.md) (dynamic COG) |
| Geocoding | [Nominatim](geocoding-routing.md) (self-hosted) or [Pelias](geocoding-routing.md) (address-level) |
| Routing & isochrones | [Valhalla](geocoding-routing.md) (full-featured) or [OSRM](geocoding-routing.md) (fastest) |
| 3D visualization | [CesiumJS](3d-visualization.md) (web) or [Blender GIS](3d-visualization.md) (cinematic) |
| Field data collection | [QField 3.x](mobile-gis.md) or [Mergin Maps](mobile-gis.md) |
| Data pipeline orchestration | [Dagster](etl-data-engineering.md) or [Airflow](etl-data-engineering.md) + GDAL |
| Serverless spatial APIs | [Fused.io](cloud-platforms.md) |
| Spatial ETL (enterprise) | [FME](etl-data-engineering.md) |

## Key Highlights

- **13 Topic Areas**: From desktop GIS to AI/ML, from mobile field collection to production tile serving
- **Expert-Level Content**: Each page written by domain specialists with real code examples, not just descriptions
- **Advanced Dark Arts**: Every page includes unconventional tricks, performance hacks, and expert techniques
- **Production-Ready**: Docker recipes, CI/CD workflows, Terraform configs, deployment guides
- **Plugin Ecosystems**: QGIS plugins, Leaflet plugins, GeoServer extensions, MapLibre protocols
- **Complete Code Examples**: Runnable snippets for every tool, not pseudocode
