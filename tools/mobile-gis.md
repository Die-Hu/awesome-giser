# Mobile GIS

Mobile applications for field data collection, GPS navigation, and surveying on smartphones and tablets.

> **Quick Picks**
> - :trophy: **SOTA**: [QField 3.x](https://qfield.org) --- the best mobile GIS for QGIS users; full offline support, iOS added in 3.0, cloud sync
> - :moneybag: **Free Best**: [KoBoToolbox](https://www.kobotoolbox.org) --- completely free for humanitarian and research use; powerful form builder with geo support
> - :zap: **Fastest Setup**: [Mergin Maps](https://merginmaps.com) --- install app, open shared project, start collecting data in under 5 minutes

## Comparison Table

| Tool | Type | Cost | Platform | Offline | Backend | Latest Version | Best For |
|------|------|------|----------|---------|---------|---------------|----------|
| QField | Field collection | Free | Android, iOS | Yes | QFieldCloud / QGIS | 3.4 | QGIS users, structured field surveys |
| Mergin Maps | Field collection | Freemium | Android, iOS | Yes | Mergin Cloud / QGIS | 2024.x | Team collaboration, differential sync |
| SW Maps | Field collection | Free | Android | Yes | Standalone | 4.x | Lightweight GIS field mapping |
| ArcGIS Field Maps | Field collection | Subscription | Android, iOS | Yes | ArcGIS Online/Enterprise | 2024.x | Esri ecosystem, enterprise |
| KoBoToolbox | Survey/forms | Free | Android, iOS (web) | Yes | KoBoToolbox server | 2024 | Humanitarian, research surveys |
| ODK Collect | Survey/forms | Free | Android | Yes | ODK Central | 2024.x | Humanitarian, custom form logic |
| Epicollect5 | Survey/forms | Free | Android, iOS | Yes | Epicollect5 server | 5.x | Small research projects, citizen science |

## Field Collection

Apps designed for collecting and editing spatial data in the field with map-based interfaces.

### QField

The mobile companion for QGIS, allowing direct use of QGIS projects on mobile devices. QField **3.x** brought iOS support, redesigned sketching tools, improved camera and sensor integration, and cloud-based project management via QFieldCloud.

- **Current Version**: 3.4
- **Platform**: Android, iOS (since QField 3.0)
- **Offline**: Full offline capability with local basemaps, vector layers, and raster tiles
- **Sync**: QFieldCloud for project synchronization, conflict resolution, and team management
- **Features**: GPS tracking with accuracy indicators, camera integration with geotagging, attribute forms with constraints and defaults, sketching on maps, barcode/QR scanning, Sketcher tools, external GNSS receiver support via Bluetooth
- **Workflow**: Design project in QGIS Desktop (styling, forms, relations), package with QFieldSync plugin, deploy to QField for field work, sync back changes via QFieldCloud
- **What's New in 3.x**: iOS native support, cloud-based project management, updated sketching tools, improved navigation and location tracking, NFC tag reading, better handling of relations and value maps
- **Links**: [qfield.org](https://qfield.org) | [QFieldCloud](https://qfield.cloud)

### Mergin Maps

Collaborative field data collection platform with seamless QGIS integration. Mergin Maps (formerly Input app by Lutra Consulting) offers the best differential sync system for team-based field work.

- **Current Version**: 2024.x
- **Platform**: Android, iOS
- **Offline**: Full offline support with differential sync (only changed features are synced, not the whole dataset)
- **Sync**: Mergin Maps Cloud with automatic conflict resolution; self-hosted Community Edition available
- **Features**: Photo capture with geotagging, sketching, attribute forms, GPS tracking, streaming digitizing for line/polygon features, external GNSS support
- **Free Tier**: 100 MB storage, 1 project for individuals
- **Paid Plans**: From EUR 29/month for teams (5 GB storage, unlimited projects)
- **Links**: [merginmaps.com](https://merginmaps.com) | [GitHub](https://github.com/MerginMaps)

### SW Maps

Lightweight GIS field mapping app with support for external GPS receivers and common GIS formats.

- **Current Version**: 4.x
- **Platform**: Android
- **Offline**: Offline basemaps (MBTiles, TPK) and vector layers
- **Features**: Shapefiles, KML, GeoJSON, GeoPackage support; external GNSS receiver support (Bluetooth and USB); real-time NTRIP corrections; coordinate display in multiple CRS
- **Cost**: Free with optional pro features
- **Links**: [Play Store](https://play.google.com/store/apps/details?id=np.com.softwel.swmaps)

### ArcGIS Field Maps

Esri's unified field data collection app replacing Collector and Explorer.

- **Current Version**: 2024.x
- **Platform**: Android, iOS
- **Offline**: Offline areas and map packages with automatic sync when reconnected
- **Sync**: Real-time sync with ArcGIS Online or Enterprise
- **Features**: Smart forms with conditional visibility, location tracking for workforce management, indoor mapping with floor-aware maps, augmented reality measurement, high-accuracy GNSS integration
- **Requires**: ArcGIS Online or Enterprise subscription (Creator license or above)
- **Links**: [esri.com/en-us/arcgis/products/arcgis-field-maps](https://www.esri.com/en-us/arcgis/products/arcgis-field-maps/overview)

## Survey & Form-Based Collection

### KoBoToolbox

Free and open-source tool for field data collection used extensively in humanitarian contexts, development, and academic research. Supports geo questions (GPS point, line, area) natively.

- **Current Version**: 2024
- **Platform**: Android (KoboCollect), iOS/web (Enketo web forms)
- **Offline**: Full offline support in KoboCollect; Enketo web forms can work offline via service worker
- **Features**: Drag-and-drop form builder, GPS point/trace/polygon questions, photo/audio/video capture, skip logic and validation, cascading select lists, repeat groups, multi-language forms
- **Backend**: Free hosted server at kobo.humanitarianresponse.info (UN OCHA) or kf.kobotoolbox.org; self-hosted option available
- **Data Export**: CSV, XLS, KML, GeoJSON, SPSS; REST API for programmatic access
- **Cost**: Completely free for humanitarian and research use; enterprise plans for large-scale commercial use
- **Links**: [kobotoolbox.org](https://www.kobotoolbox.org) | [GitHub](https://github.com/kobotoolbox)

### ODK Collect

Open Data Kit -- the foundational open-source mobile data collection platform. KoBoToolbox is built on top of ODK standards (XLSForm/XForm).

- **Current Version**: 2024.x
- **Platform**: Android
- **Offline**: Full offline support
- **Backend**: ODK Central (self-hosted or cloud); compatible with ODK-compliant servers
- **Features**: XLSForm standard for form design, GPS, photo, audio, barcode, and more. Supports complex form logic, multi-language, and encrypted submissions
- **Links**: [getodk.org](https://getodk.org) | [ODK Central](https://docs.getodk.org/central-intro/)

## Navigation

GPS navigation and outdoor mapping applications useful for GIS fieldwork.

| App | Platform | Offline Maps | Custom Layers | Cost | Notes |
|-----|----------|-------------|---------------|------|-------|
| OsmAnd | Android, iOS | Yes (OSM) | GPX, KML, GeoJSON | Free / Paid | Best OSM-based navigation |
| Locus Map | Android | Yes (OSM, custom) | SHP, GPX, KML, MBTiles | Free / Paid | Feature-rich outdoor navigation |
| Avenza Maps | Android, iOS | Yes (GeoPDF, GeoTIFF) | Custom georeferenced maps | Free / Paid | Best for custom paper map digitization |
| Gaia GPS | Android, iOS | Yes (multiple providers) | GPX, KML | Subscription ($40/yr) | Hiking, backcountry navigation |
| OruxMaps | Android | Yes (multiple providers) | GPX, KML, GeoTIFF | Free | Highly customizable, offline routing |

## Survey Tools

Specialized tools for land surveying, GNSS data collection, and precision measurement.

| Tool | Platform | GNSS Support | RTK | Export Formats | Cost |
|------|----------|-------------|-----|----------------|------|
| Emlid Flow | Android, iOS | Emlid Reach RS3/RX | Yes (NTRIP) | CSV, GeoJSON, DXF, SHP, RINEX | Free (with Emlid hardware) |
| Lefebure NTRIP Client | Android | External receivers | Yes (NTRIP relay) | NMEA | Free |
| GPS Test | Android | Built-in GPS/GLONASS/Galileo | No | N/A | Free |
| MapIt GIS | Android | Built-in + external | Via NTRIP | SHP, KML, DXF, CSV | Free / Paid |
| Field Genius | Android | Multiple receivers (Topcon, Leica, etc.) | Yes | DXF, CSV, survey formats | Paid |
| Tersus Survey | Android, iOS | Tersus GNSS receivers | Yes | CSV, DXF, SHP | Free (with Tersus hardware) |

## Tips for Mobile GIS Fieldwork

- **Offline Preparation**: Always download basemaps and project data before going to the field. Use MBTiles or GeoPackage for offline basemaps. Test your project offline before deploying.
- **Battery Management**: GPS usage drains batteries quickly. Carry external power banks (10,000+ mAh recommended). Reduce screen brightness and close unused apps.
- **External GNSS**: For sub-meter accuracy, use external Bluetooth GNSS receivers (e.g., Emlid Reach RS3, Bad Elf Flex, Trimble R2). QField, Mergin Maps, and SW Maps all support external receivers.
- **Data Backup**: Sync data to cloud regularly; do not rely solely on device storage. QFieldCloud and Mergin Cloud provide automatic sync with conflict resolution.
- **Form Design**: Pre-configure attribute forms with dropdowns, default values, and constraints to minimize field entry errors. Both QField and Mergin Maps respect QGIS form configuration.
- **Coordinate Systems**: Verify your project CRS matches your GPS receiver output. Most mobile apps work in WGS 84 (EPSG:4326) by default.
- **Weather Protection**: Use waterproof phone cases or rugged tablets for fieldwork in wet conditions. Capacitive styluses work with most touchscreens even with gloves.
- **Photo Documentation**: Enable geotagging on camera captures. Take photos in cardinal directions at each survey point for context.
