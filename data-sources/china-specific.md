# China-Specific Data

Geospatial data sources specific to China, including government portals, coordinate system guidance, POI data, map APIs, remote sensing platforms, and academic data repositories.

> **Quick Picks**
> - **SOTA**: [PIE-Engine](https://engine.piesat.cn) -- China's answer to Google Earth Engine, with domestic satellite data and Python/JS APIs
> - **Free Best**: [Tianditu (天地图)](https://www.tianditu.gov.cn) -- free official basemap and geocoding API, legal for use in China
> - **Fastest Setup**: [Amap (高德) JS API](https://lbs.amap.com) -- generous free tier, rich POI data, easy web map integration

> **Important**: China uses non-standard coordinate systems (GCJ-02, BD-09) that apply offsets to WGS-84 coordinates. Always verify and convert coordinates when working with Chinese geospatial data. See the [Coordinate Systems](#coordinate-systems-gcj-02-bd-09) section below.

## Government Portals

Official Chinese government data sources for geospatial information.

| Dataset | Provider | Content | Format | URL | Label |
|---------|----------|---------|--------|-----|-------|
| National Geomatics Center of China (NGCC / 国家基础地理信息中心) | MNR | Basemaps, boundaries, place names, 1:1M vector data | SHP, DWG, various | [ngcc.cn](https://www.ngcc.cn), [webmap.cn](https://www.webmap.cn) | Free (registration) |
| Tianditu (天地图) | NGCC/MNR | Official basemap tiles, geocoding, routing, POI search | Web API (REST/WMTS) | [tianditu.gov.cn](https://www.tianditu.gov.cn) | Free (API key required) / SOTA (China legal) |
| National Earth System Science Data Center (国家地球系统科学数据中心) | CAS | Earth science datasets, climate, ecology, soil | NetCDF, GeoTIFF, SHP, CSV | [geodata.cn](http://www.geodata.cn) | Free (registration) |
| China Meteorological Data Service (中国气象数据网) | CMA | Weather stations, climate records, satellite products | CSV, GRIB, HDF | [data.cma.cn](http://data.cma.cn) | Free (registration, quotas) |
| National Catalogue Service for Geographic Information | NGCC | 1:250K-1:1M topographic maps, imagery | SHP, TIFF, various | [webmap.cn](https://www.webmap.cn) | Free (registration) |
| Ministry of Natural Resources Open Data (自然资源部) | MNR | Land use surveys, ocean data, geological surveys | SHP, PDF, various | [mnr.gov.cn](https://www.mnr.gov.cn) | Free / Restricted |
| National Bureau of Statistics (国家统计局) | NBS | Census, economic statistics, geocodable admin units | CSV, XLS, API | [stats.gov.cn](https://www.stats.gov.cn), [data.stats.gov.cn](https://data.stats.gov.cn) | Free |
| National Platform for Common Geospatial Information Services | MNR | Integrated geospatial services platform | Web Services | [tianditu.gov.cn](https://www.tianditu.gov.cn) | Free |
| China Geological Survey Data (中国地质调查局) | CGS | Geological maps, mineral resources, hydrogeology | SHP, GeoTIFF | [geocloud.cgs.gov.cn](https://geocloud.cgs.gov.cn) | Free (registration) |

## Tianditu (天地图) API

Tianditu is China's official web mapping service operated by NGCC. It provides legally compliant basemaps and GIS services for applications within China.

| Service | Endpoint | Description | Rate Limit | Label |
|---------|----------|-------------|------------|-------|
| Map Tiles (WMTS) | `https://t{0-7}.tianditu.gov.cn/{layer}/wmts` | Basemap tiles (vector, imagery, terrain, annotation) | 10,000 req/day (free) | Free / SOTA |
| Geocoding API | `https://api.tianditu.gov.cn/geocoder` | Address to coordinates (China only) | 10,000 req/day | Free |
| Reverse Geocoding | `https://api.tianditu.gov.cn/geocoder` | Coordinates to address | 10,000 req/day | Free |
| POI Search | `https://api.tianditu.gov.cn/search` | POI keyword and radius search | 10,000 req/day | Free |
| Routing | `https://api.tianditu.gov.cn/drive` | Driving, walking, transit directions | 10,000 req/day | Free |
| Administrative Division Query | `https://api.tianditu.gov.cn/administrative` | Query admin boundaries by code/name | 10,000 req/day | Free |

> **Usage tip**: Tianditu uses CGCS2000 (practically identical to WGS-84). It is the only major Chinese map service that does NOT use GCJ-02 encryption, making it easier to integrate with international data.

## Coordinate Systems (GCJ-02, BD-09)

China mandates encrypted coordinate systems for domestic mapping services. Understanding these is critical for accurate geospatial work.

| System | Used By | Offset from WGS-84 | Notes |
|--------|---------|-------------------|-------|
| WGS-84 | GPS raw, international data, Tianditu | None (reference) | Standard global CRS |
| GCJ-02 (火星坐标/Mars Coordinates) | Amap (Gaode), Tencent Maps, Google Maps CN, Apple Maps CN | ~100-700m | Government-mandated encryption for all domestic maps |
| BD-09 | Baidu Maps only | Additional offset on top of GCJ-02 | Baidu proprietary double encryption |
| CGCS2000 (中国大地坐标系) | Official surveying, Tianditu | ~0m (practically WGS-84) | China Geodetic Coordinate System 2000, national standard |

### Conversion Libraries

Tools for converting between WGS-84, GCJ-02, and BD-09 coordinate systems.

| Library | Language | Conversions | Stars | URL | Label |
|---------|----------|-------------|-------|-----|-------|
| eviltransform | Python, Go, JS, C, Java, Rust, Dart, PHP | WGS-84 <-> GCJ-02 <-> BD-09 | 5k+ | [github.com/googollee/eviltransform](https://github.com/googollee/eviltransform) | SOTA / Multi-language |
| coordtransform | JavaScript, Python | WGS-84 <-> GCJ-02 <-> BD-09 | 4k+ | [github.com/wandergis/coordtransform](https://github.com/wandergis/coordtransform) | Practical / Popular |
| gcoord | JavaScript/TypeScript | WGS-84, GCJ-02, BD-09, EPSG:3857, Mapbar | 1k+ | [github.com/hujiulong/gcoord](https://github.com/hujiulong/gcoord) | SOTA (JS) -- integrates with Leaflet/MapLibre |
| coord-convert (Python) | Python | WGS-84 <-> GCJ-02 <-> BD-09 (batch) | -- | [pypi.org/project/coord-convert](https://pypi.org/project/coord-convert/) | Practical / pip install |
| prcoords | Multiple (JS, Python, C#, Rust) | WGS-84 <-> GCJ-02 <-> BD-09 (precise) | 500+ | [github.com/caoccao/prcoords](https://github.com/Artoria2e5/PRCoords) | Practical / High precision |
| QGIS ChinaGeocodingTools Plugin | QGIS Python Plugin | Batch geocoding + coord conversion | -- | QGIS Plugin Repository | Practical |

> **Legal Note**: Converting from GCJ-02 back to WGS-84 operates in a legal gray area in China. The encryption algorithm is not officially published, and available conversions are reverse-engineered approximations with ~1m accuracy.

## Map APIs (POI Data & Services)

Commercial map APIs with generous free tiers, providing POI data, geocoding, routing, and basemaps for China.

| API | Provider | Free Tier | Coord System | Key Features | URL | Label |
|-----|----------|-----------|--------------|-------------|-----|-------|
| Amap (高德地图) JS/REST API | Alibaba | 5,000 req/day (basic), 300K/day (enterprise) | GCJ-02 | POI search, geocoding, routing, real-time traffic, web map SDK | [lbs.amap.com](https://lbs.amap.com) | SOTA / Practical |
| Baidu Map JS/REST API | Baidu | 5,000-30,000 req/day | BD-09 | POI, geocoding, routing, heatmap, indoor maps | [lbsyun.baidu.com](https://lbsyun.baidu.com) | Practical |
| Tencent Map (腾讯地图) API | Tencent | 10,000 req/day | GCJ-02 | POI, geocoding, routing, WeChat Mini Program integration | [lbs.qq.com](https://lbs.qq.com) | Practical |
| Tianditu API | NGCC/MNR | 10,000 req/day | CGCS2000 (WGS-84 compatible) | Official basemap, geocoding, POI -- legal for government projects | [lbs.tianditu.gov.cn](https://lbs.tianditu.gov.cn) | Free / Official |
| SuperMap iServer REST API | SuperMap | Developer license (free trial) | Multiple CRS supported | Enterprise GIS server, spatial analysis, 3D, tile services | [supermap.com](https://www.supermap.com/en-us/a/product/iServer.html) | Commercial / SOTA (enterprise) |

> **POI Data tip**: Amap and Baidu provide the most comprehensive POI data for mainland China. Use the batch POI scraping tools with caution and always check their Terms of Service. For official/government projects, use Tianditu.

## Remote Sensing Platforms

Chinese satellite programs and remote sensing data access platforms.

| Platform / Satellite | Provider | Resolution | Data Type | Access | Label |
|---------------------|----------|-----------|-----------|--------|-------|
| GaoFen-1 (GF-1) | CNSA/CRESDA | 2m (Pan), 8m (MS), 16m (WFV) | Optical | [cresda.com](http://www.cresda.com), [CNSA Data Center](http://www.cnsa.gov.cn) | Free (registration) |
| GaoFen-2 (GF-2) | CNSA/CRESDA | 0.8m (Pan), 3.2m (MS) | Optical | [cresda.com](http://www.cresda.com) | Free (registration) |
| GaoFen-3 (GF-3) | CNSA/CRESDA | 1-500m (SAR, multi-mode) | C-band SAR | [cresda.com](http://www.cresda.com) | Free (registration) |
| GaoFen-7 (GF-7) | CNSA/CRESDA | 0.65m (Pan), 2.6m (MS), stereo | Optical stereo + laser altimeter | [cresda.com](http://www.cresda.com) | Free (registration) |
| ZiYuan-3 (ZY-3) | CNSA/CRESDA | 2.1m (Pan), 5.8m (MS), stereo | Optical stereo mapping | [cresda.com](http://www.cresda.com) | Free (registration) |
| HuanJing-2 (HJ-2A/2B) | CNSA/CRESDA | 16m (MS), 48m (hyperspectral) | Optical, Hyperspectral, Infrared | [cresda.com](http://www.cresda.com) | Free (registration) |
| Luojia-1 (珞珈一号) | Wuhan University | 130m | Nighttime light (high sensitivity) | [59.175.109.173:8888](http://59.175.109.173:8888) | Free / Academic |
| PIE-Engine (航天宏图) | PIESAT (航天宏图) | N/A (platform) | Multi-source (domestic + international) | [engine.piesat.cn](https://engine.piesat.cn) | Free tier / SOTA |
| AIEarth (阿里云 AI Earth) | Alibaba Cloud | N/A (platform) | Multi-source analysis + AI models | [engine-aiearth.aliyun.com](https://engine-aiearth.aliyun.com) | Free tier |
| RSCloudS (遥感云) | Various | N/A (platform) | Cloud-based RS processing | Multiple providers | Commercial |

### PIE-Engine Details

PIE-Engine is China's leading cloud-based remote sensing platform, offering capabilities similar to Google Earth Engine with domestic satellite data integration.

| Feature | Description |
|---------|-------------|
| Data Catalog | 100+ datasets including GaoFen, Sentinel, Landsat, MODIS, ERA5 |
| Languages | Python SDK (`piesat`), JavaScript API |
| Processing | Server-side parallel computation, GPU acceleration |
| AI Models | Built-in deep learning models for land cover, change detection, object extraction |
| Free Tier | 100 GB storage, moderate compute quota |
| URL | [engine.piesat.cn](https://engine.piesat.cn) |

### SuperMap iServer

SuperMap is China's leading enterprise GIS platform, widely used in government and infrastructure projects.

| Feature | Description |
|---------|-------------|
| Core | Full GIS server: map services, spatial analysis, 3D GIS, big data GIS |
| APIs | REST, OGC (WMS/WFS/WMTS), JavaScript, Python |
| 3D | SuperMap iClient3D for Cesium, WebGL-based 3D visualization |
| Formats | SHP, GeoJSON, UDBX (native), PostGIS, Oracle Spatial, HDFS |
| License | Commercial (developer license available for testing) |
| URL | [supermap.com](https://www.supermap.com) |

## Academic Data Sources

Chinese academic institutions providing geospatial research data.

| Dataset | Institution | Content | Format | Access | Label |
|---------|------------|---------|--------|--------|-------|
| Resource & Environment Science Data Center (资源环境科学与数据中心) | CAS IGSNRR | Land use (1980-2020), population grids, GDP grids, soil types | GeoTIFF, SHP | [resdc.cn](https://www.resdc.cn) | Free (registration) / SOTA (China LULC) |
| National Tibetan Plateau Data Center (TPDC) | CAS | Tibetan Plateau climate, cryosphere, hydrology, ecology | NetCDF, GeoTIFF, CSV | [data.tpdc.ac.cn](https://data.tpdc.ac.cn) | Free (registration) |
| OpenStreetMap China Extract | Geofabrik | OSM vector data for China | PBF, SHP | [download.geofabrik.de](https://download.geofabrik.de/asia/china.html) | Free |
| WorldPop China | WorldPop / University of Southampton | Population grids at 100m/1km | GeoTIFF | [worldpop.org](https://www.worldpop.org) | Free |
| CLCD (China Land Cover Dataset) | Wuhan University (Prof. Yang/Huang) | 30m annual land cover for China, 1985-2023 | GeoTIFF | [GEE](https://developers.google.com/earth-engine/datasets), [Zenodo](https://zenodo.org) | Free / SOTA (China land cover) |
| China City Statistical Yearbooks (中国城市统计年鉴) | NBS / Municipal govts | Urban demographics, economy, infrastructure (geocodable) | PDF, XLS | University libraries, [stats.gov.cn](https://www.stats.gov.cn) | Free / Restricted |
| CHAP (China High-resolution Air Pollutants) | Tsinghua/PKU | PM2.5, NO2, O3, SO2 at 1-10km, daily | NetCDF, GeoTIFF | [weijing-rs.github.io/product](https://weijing-rs.github.io/product.html) | Free / Academic |
| ChinaGEOSS Data Sharing Network | NASG/CAS | Multi-domain Earth observation data | Various | [chinageoss.cn](http://www.chinageoss.cn) | Free (registration) |
| 1km Monthly Climate Data for China (1901-present) | CAS/National Meteorological Info Center | Temperature, precipitation, solar radiation | NetCDF | [data.tpdc.ac.cn](https://data.tpdc.ac.cn) | Free |
| Global Urban Boundaries (GUB) | Tsinghua University | Urban extent delineation for Chinese cities | SHP, GeoTIFF | [data.ess.tsinghua.edu.cn](http://data.ess.tsinghua.edu.cn) | Free / Academic |

## Data Access Tips for China

- **VPN/Network**: Some international data sources (GEE, AWS S3, Google Cloud) may be slow or inaccessible from within mainland China. PIE-Engine and AIEarth provide domestic alternatives.
- **Registration**: Most Chinese government data portals require real-name registration with a Chinese mobile phone number.
- **Language**: Government portals are primarily in Chinese. Use browser translation or look for English interface toggles.
- **Tianditu for compliance**: If building applications for use in China, Tianditu basemaps ensure legal compliance with China's mapping regulations.
- **Coordinate pipeline**: Always establish a clear coordinate conversion pipeline early in your project. Test conversions on known points before batch processing.
- **Academic access**: Many Chinese academic datasets require institutional affiliation. International researchers can often gain access through collaboration agreements.
