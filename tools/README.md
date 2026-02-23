# 🔧 Tools

Essential GIS tools from desktop applications to cloud platforms, covering the full spectrum of geospatial workflows.

> **Comparison Approach**: Each section highlights SOTA (state-of-the-art) picks alongside free and open-source alternatives, so you can choose the right tool for your budget and use case.

## Table of Contents

- [Desktop GIS](desktop-gis.md) - Full-featured desktop applications for spatial analysis and cartography
- [CLI Tools](cli-tools.md) - Command-line utilities for batch processing, format conversion, and automation
- [Spatial Databases](spatial-databases.md) - Database engines with spatial extensions for storing and querying geodata
- [Cloud Platforms](cloud-platforms.md) - Cloud-based platforms for large-scale geospatial analysis and visualization
- [Mobile GIS](mobile-gis.md) - Field data collection, navigation, and survey tools for mobile devices

## How to Choose

| Need | Recommended Starting Point |
|------|---------------------------|
| General-purpose GIS analysis | Desktop GIS (QGIS 3.36+) |
| Batch processing & automation | CLI Tools (GDAL 3.9+, tippecanoe, PDAL) |
| Large dataset storage & queries | Spatial Databases (PostGIS or DuckDB Spatial) |
| Fast ad-hoc analytics on files | DuckDB Spatial (zero-config, reads Parquet/GeoJSON/SHP) |
| Planetary-scale remote sensing | Cloud Platforms (Google Earth Engine) |
| Serverless spatial APIs | Cloud Platforms (Fused.io) |
| Field data collection | Mobile GIS (QField 3.x / Mergin Maps) |
