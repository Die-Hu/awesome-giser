# Desktop GIS

Full-featured desktop applications for spatial data visualization, analysis, and cartography.

> **Quick Picks**
> - :trophy: **SOTA**: [QGIS 3.36+](https://qgis.org) --- free, cross-platform, unmatched plugin ecosystem and community momentum
> - :moneybag: **Free Best**: [GRASS GIS 8.4](https://grass.osgeo.org) --- deepest analytical toolbox for raster/vector modeling
> - :zap: **Fastest Setup**: [QGIS](https://qgis.org/download/) --- single installer, runs on Windows/macOS/Linux out of the box

## Comparison Table

| Tool | Type | Latest Version | Cost | Platform | Best For |
|------|------|---------------|------|----------|----------|
| QGIS | Open Source | 3.36 LTR / 3.40 | Free | Win/Mac/Linux | General GIS, cartography, plugin ecosystem |
| GRASS GIS | Open Source | 8.4 | Free | Win/Mac/Linux | Advanced raster/vector analysis, modeling |
| gvSIG | Open Source | 2.6 | Free | Win/Mac/Linux | Lightweight GIS with remote sensing tools |
| ArcGIS Pro | Commercial | 3.4 | ~$100+/yr (personal use) | Windows | Enterprise GIS, 3D, deep learning integration |
| MapInfo Pro | Commercial | 2024 | Subscription | Windows | Business mapping, telecom, utilities |
| Global Mapper | Commercial | 25.1 | ~$550+/yr | Windows | Format support, LiDAR, terrain analysis |
| SAGA GIS | Open Source | 9.4 | Free | Win/Mac/Linux | Terrain analysis, geomorphometry, grid processing |
| Whitebox GIS | Open Source | 2.3 / WhiteboxTools | Free | Win/Mac/Linux | Hydrological modeling, geomorphometric analysis |

## Open Source

### QGIS

The leading free and open-source GIS with a rich plugin ecosystem and strong community. QGIS 3.36+ (LTR) introduced native support for point cloud rendering, improved mesh layer handling, a revamped sketching/annotation system, and performance gains for large vector layers.

- **Current Version**: 3.36 LTR (Sketsketcher) / 3.40 latest
- **Strengths**: Extensive plugin library (1,800+ plugins), strong cartography with print layout manager, Python scripting (PyQGIS), database integration, native point cloud and mesh support
- **Data Formats**: Supports 100+ formats via GDAL/OGR, including GeoPackage, GeoParquet, FlatGeobuf, PMTiles, COG
- **Community**: Active development with 6-month release cadence, large user community, annual QGIS User Conferences
- **What's New in 3.36+**: Native GeoParquet read, improved 3D map views, GPS sketching tools, annotation layer enhancements, temporal controller improvements for animated maps
- **Links**: [qgis.org](https://qgis.org) | [Documentation](https://docs.qgis.org/) | [Plugin Repository](https://plugins.qgis.org/)

### GRASS GIS

Powerful analytical engine for raster, vector, and geospatial modeling. One of the oldest open-source GIS projects (since 1982), GRASS 8.x brings a modernized Python API and improved integration with Jupyter notebooks.

- **Current Version**: 8.4
- **Strengths**: 500+ modules for spatial analysis, hydrological modeling, image processing, and terrain analysis
- **Integration**: Can be used as a backend for QGIS via the Processing framework; also callable from Python and Jupyter
- **Scripting**: Python and shell scripting support; `grass.script` and `grass.jupyter` modules for reproducible workflows
- **Use Cases**: Watershed delineation, viewshed analysis, solar irradiance modeling, landscape ecology metrics
- **Links**: [grass.osgeo.org](https://grass.osgeo.org) | [Manual](https://grass.osgeo.org/grass84/manuals/)

### gvSIG

Lightweight desktop GIS with built-in remote sensing and 3D visualization.

- **Current Version**: 2.6
- **Strengths**: Remote sensing tools, built-in raster calculator, scripting via Jython, 3D view
- **Notable**: Includes gvSIG Mobile for field work; strong adoption in Latin America and Spanish-speaking communities
- **Links**: [gvsig.com](http://www.gvsig.com/en)

## Commercial

### ArcGIS Pro

Esri's flagship desktop GIS application, industry standard in many organizations. ArcGIS Pro 3.x introduced ArcPy 3 (Python 3.11), deeper integration with ArcGIS Online, and an AI-powered geoprocessing assistant.

- **Current Version**: 3.4
- **Strengths**: Enterprise integration, 3D analysis, deep learning tools (Esri Deep Learning frameworks), ArcGIS Online connectivity, ModelBuilder and Notebooks, reality mapping
- **What's New in 3.x**: Python 3.11 runtime, AI/ML object detection, raster analytics in notebooks, improved Voxel layer support, link analysis, knowledge graphs
- **Licensing**: Named user or concurrent use; included with ArcGIS Online subscription
- **Note**: Free for students and educators via Esri education programs; ArcGIS Pro is Windows-only (no native macOS/Linux)
- **Links**: [esri.com/en-us/arcgis/products/arcgis-pro](https://www.esri.com/en-us/arcgis/products/arcgis-pro/overview)

### MapInfo Pro

Established desktop GIS focused on business intelligence and spatial analysis.

- **Current Version**: MapInfo Pro 2024
- **Strengths**: Business mapping, territory management, data visualization, geocoding and routing
- **Integration**: Works with Pitney Bowes data products; integrates with Spectrum Spatial for enterprise deployments
- **Links**: [precisely.com/product/mapinfo-pro](https://www.precisely.com/product/mapinfo-pro)

### Global Mapper

Versatile desktop GIS known for broad format support and terrain tools.

- **Current Version**: 25.1
- **Strengths**: Supports 300+ formats, LiDAR processing (point cloud classification, surface generation), 3D visualization, built-in scripting API
- **Notable**: Often used as a format conversion workhorse; the LiDAR Module is a cost-effective alternative to dedicated point cloud software
- **Links**: [bluemarblegeo.com/global-mapper](https://www.bluemarblegeo.com/global-mapper/)

## QGIS Plugin Highlights

Essential QGIS plugins that extend its capabilities significantly. Install via **Plugins > Manage and Install Plugins** in QGIS.

| Plugin | Purpose | Category | Notes |
|--------|---------|----------|-------|
| QuickOSM | Download OSM data with Overpass queries | Data Acquisition | Replaces manual OSM downloads; runs Overpass API queries directly |
| QuickMapServices | Add basemaps (OSM, Google, Bing, etc.) | Basemaps | One-click access to 30+ basemap providers |
| Semi-Automatic Classification Plugin (SCP) | Remote sensing classification and download | Remote Sensing | Download Sentinel-2/Landsat, perform supervised classification, band math |
| qgis2web | Export maps to Leaflet/OpenLayers/Mapbox GL JS | Web Mapping | Generates self-contained HTML maps from QGIS projects |
| Processing R Provider | Run R scripts in QGIS | Analysis | Bridge between QGIS and R statistical computing |
| TimeManager | Animate temporal data | Visualization | Create time-series animations from attribute-based timestamps |
| Profile Tool | Elevation profile extraction from DEMs | Terrain | Interactive cross-section profiles along drawn lines |
| dzetsaka | Machine learning classification | Remote Sensing | Random Forest, SVM, GMM classifiers for raster data |
| Sketcher | Free sketching and annotation on map canvas | Sketching | Annotate and mark up maps during field review |
| QGIS Hub Plugin Manager | Discover community styles, models, scripts | Community | Centralised access to shared resources |
| MapSwipe Tool | Side-by-side / swipe comparison of layers | Visualization | Compare before/after imagery or classification results |
| Point Sampling Tool | Extract raster values at point locations | Analysis | Sample multi-band rasters to point attribute tables |

## Getting Started Recommendations

- **Beginners**: Start with QGIS -- it is free, cross-platform, and has excellent documentation and tutorials. The built-in training manual walks through core concepts step by step.
- **Enterprise Users**: ArcGIS Pro if your organization already uses Esri products; it offers the smoothest path to ArcGIS Online, Enterprise, and Esri's cloud analytics.
- **Heavy Analysis**: Consider GRASS GIS for complex raster modeling, hydrological analysis, and landscape ecology metrics. Use it standalone or as a QGIS Processing backend.
- **Format Conversion**: Global Mapper excels at handling obscure formats and LiDAR data; GDAL/OGR CLI is the free alternative.
- **Learning Path**: QGIS tutorials at [qgis.org/resources/hub/](https://qgis.org/resources/hub/) and [docs.qgis.org](https://docs.qgis.org/). For ArcGIS Pro, see [learn.arcgis.com](https://learn.arcgis.com/).
- **YouTube Channels**: Klas Karlsson (QGIS), Hatari Labs, Open Source Options, GeoDelta Labs
