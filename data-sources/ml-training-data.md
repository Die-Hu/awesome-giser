# Geospatial AI & Machine Learning Training Data

Comprehensive catalog of labeled datasets, foundation models, and benchmark platforms for training geospatial machine learning models.

> **Quick Picks**
> - **SOTA**: [Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial) -- 600M-parameter NASA/IBM foundation model pre-trained on 4.2M HLS tiles
> - **Free Best**: [TorchGeo](https://github.com/microsoft/torchgeo) -- 125+ geospatial datasets with PyTorch dataloaders, transforms, and pre-trained models
> - **Fastest Setup**: [EuroSAT](https://github.com/phelber/EuroSAT) -- 27,000 Sentinel-2 patches, 10 classes, trainable on a laptop GPU
> - **Dataset Hub**: [Radiant MLHub](https://mlhub.earth/) -- open STAC-compliant catalog of ML-ready geospatial training data

## Labeled Satellite Image Datasets

### SpaceNet Challenges (1-8)

SpaceNet is the premier open satellite imagery challenge series, hosted on AWS Open Data. All datasets use very-high-resolution commercial satellite imagery with expert annotations.

| Challenge | Task | Location(s) | Images / Area | Resolution | Annotations | Format | Year |
|-----------|------|------------|---------------|-----------|-------------|--------|------|
| SpaceNet 1 | Building Detection | Rio de Janeiro | ~6,940 chips (200 km²) | 0.5m (WV-3) | ~383K building polygons | GeoTIFF + GeoJSON | 2017 |
| SpaceNet 2 | Building Detection | Las Vegas, Paris, Shanghai, Khartoum, Rio | ~24,586 chips | 0.3m (WV-3, 3/8-band) | ~685K building polygons | GeoTIFF + GeoJSON | 2017 |
| SpaceNet 3 | Road Network Extraction | Las Vegas, Paris, Shanghai, Khartoum | ~8,000 km of roads | 0.3m (WV-3) | Road centerlines + graph | GeoTIFF + GeoJSON | 2018 |
| SpaceNet 4 | Off-Nadir Building Detection | Atlanta | 27 images (7-54 deg off-nadir) | 0.3m (WV-2) | Building polygons | GeoTIFF + GeoJSON | 2018 |
| SpaceNet 5 | Road Network + Routing | 4 cities | ~2,300 chips, 8,000 km roads | 0.3m (WV-3) | Roads + estimated travel time | GeoTIFF + GeoJSON | 2019 |
| SpaceNet 6 | Multi-Sensor All-Weather | Rotterdam | 120 km², 202 SAR strips | 0.5m (WV-2 + Capella SAR) | ~48K building polygons | GeoTIFF (SAR+EO) + GeoJSON | 2020 |
| SpaceNet 7 | Multi-Temporal Urban Dev. | 100+ global sites | ~40,000 km², 24 monthly mosaics | 4.0m (Planet) | ~11M building polygon labels | GeoTIFF + GeoJSON | 2021 |
| SpaceNet 8 | Flood Detection | Multiple regions | Pre/post-flood image pairs | 0.3-0.5m | Flooded road/building labels | GeoTIFF + GeoJSON | 2022 |

- **Access**: Free on AWS S3 -- `s3://spacenet-dataset/`
- **URL**: [spacenet.ai/datasets](https://spacenet.ai/datasets/)
- **Tools**: `solaris` Python library for SpaceNet utilities

### Scene Classification & Land Use

| Dataset | Samples | Classes | Resolution | Sensor | Size | Format | Access |
|---------|---------|---------|-----------|--------|------|--------|--------|
| [BigEarthNet](https://bigearth.net/) | 590,326 patches (v1); 549K S2 + 321K S1 (v2) | 43 labels (multi-label) | 10/20/60m | Sentinel-2 (+ S1 in v2) | ~66 GB (v1) | GeoTIFF + JSON | Free download |
| [EuroSAT](https://github.com/phelber/EuroSAT) | 27,000 patches | 10 classes | 10m | Sentinel-2 (13 bands) | ~2.8 GB | GeoTIFF / JPEG | Free download |
| [UC Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html) | 2,100 images | 21 classes | 0.3m | USGS aerial | ~317 MB | TIFF (256x256) | Free download |
| [fMoW](https://github.com/fMoW/dataset) | 1,047,691 images | 62 categories | 0.3-0.8m | DigitalGlobe multi-temporal | ~3.5 TB (full) / ~200 GB (RGB) | TIFF / JPEG | Free on AWS S3 |
| [SEN12MS](https://mediatum.ub.tum.de/1474000) | 180,662 triplets (SAR+optical+labels) | IGBP land cover (17 classes) | 10m | Sentinel-1 + Sentinel-2 | ~500 GB | GeoTIFF | Free download |
| [So2Sat LCZ42](https://mediatum.ub.tum.de/1454690) | 400,673 paired patches | 17 Local Climate Zones | 10m | Sentinel-1 + Sentinel-2 | ~56 GB | HDF5 | Free download |
| [RESISC-45](https://www.tensorflow.org/datasets/catalog/resisc45) | 31,500 images | 45 classes | 0.2-30m | Google Earth | ~6 GB | JPEG (256x256) | Free download |
| [AID](https://captain-whu.github.io/AID/) | 10,000 images | 30 classes | 0.5-8m | Google Earth | ~2 GB | JPEG (600x600) | Free download |

> **Tip**: BigEarthNet is the go-to large-scale pre-training dataset. EuroSAT and UC Merced are ideal for quick prototyping.

## Object Detection in Aerial/Satellite Imagery

### Multi-Class Object Detection

| Dataset | Images | Instances | Classes | Annotation Type | Resolution | Format | Access |
|---------|--------|-----------|---------|----------------|-----------|--------|--------|
| [xView](http://xviewdataset.org/) | 1,127 | >1,000,000 | 60 categories | Horizontal BBox | 0.3m (WV-3) | GeoTIFF + TXT | Free (registration) |
| [DOTA v1.0](https://captain-whu.github.io/DOTA/) | 2,806 | 188,282 | 15 categories | Oriented BBox (OBB) | Variable | PNG + TXT | Free download |
| [DOTA v2.0](https://captain-whu.github.io/DOTA/) | 11,268 | 1,793,658 | 18 categories | OBB | Variable | PNG + TXT | Free download |
| [FAIR1M](https://www.researchgate.net/publication/357599892) | >40,000 | >1,000,000 | 37 sub-categories | OBB | 0.3-0.8m | TIFF + XML | Application required |
| [DIOR](https://gcheng-nwpu.github.io/#Datasets) | 23,463 | 192,472 | 20 categories | HBB | 0.5-30m | JPEG + XML (VOC) | Free download |

### Domain-Specific Detection

#### Ship Detection

| Dataset | Images | Instances | Annotation | Sensor | Access |
|---------|--------|-----------|-----------|--------|--------|
| [HRSC2016](https://ieee-dataport.org/documents/hrsc2016-0) | 1,057 | ~2,976 ships | HBB + OBB + segmentation | Optical (Google Earth) | Free download |
| [SSDD](https://github.com/TianwenZhang0825/Official-SSDD) | 1,160 | 2,456 ships | HBB + OBB | SAR (Sentinel-1, RadarSat, TerraSAR-X) | Free download |
| [Airbus Ship Detection](https://www.kaggle.com/c/airbus-ship-detection) | 192,556 | ~81K ships | Instance segmentation (RLE) | Optical (1.5m) | Free (Kaggle) |

#### Vehicle & Aircraft Detection

| Dataset | Images | Instances | Annotation | Access |
|---------|--------|-----------|-----------|--------|
| [COWC](https://gdo152.llnl.gov/cowc/) | 53 large scenes | ~32,716 cars | Point annotations | Free download |
| [RarePlanes](https://www.cosmiqworks.org/RarePlanes/) | 50,253 chips | ~600K aircraft | Instance segmentation + fine-grained attributes | Free (AWS) |

## Semantic Segmentation Datasets

| Dataset | Images | Classes | Resolution | Coverage | Image Size | Access |
|---------|--------|---------|-----------|----------|-----------|--------|
| [LoveDA](https://github.com/Junjue-Wang/LoveDA) | 5,987 | 7 (urban + rural) | 0.3m | 3 Chinese cities | 1024x1024 | Free download |
| [DeepGlobe Land Cover](http://deepglobe.org/) | 1,146 | 7 land-cover classes | 0.5m | DigitalGlobe rural areas | 2448x2448 | Free (registration) |
| [Inria Aerial Image](https://project.inria.fr/aerialimagelabeling/) | 360 tiles | 2 (building / not) | 0.3m | 10 cities (US + Europe), 810 km² | 5000x5000 | Free download |
| [ISPRS Vaihingen](https://www.isprs.org/education/benchmarks.aspx) | 33 patches | 6 classes | 0.09m (9cm) | Vaihingen, Germany | ~2494x2064 | Free (registration) |
| [ISPRS Potsdam](https://www.isprs.org/education/benchmarks.aspx) | 38 patches | 6 classes | 0.05m (5cm) | Potsdam, Germany | 6000x6000 | Free (registration) |
| [OpenEarthMap](https://open-earth-map.org/) | 5,000 | 8 classes | 0.25-0.5m | 97 regions, 44 countries | 1024x1024 | Free download |
| [DynamicEarthNet](https://mediatum.ub.tum.de/1650201) | 75 areas x 24 months | 7 classes | 3m | Planet Fusion, global | 1024x1024 | Free download |

> **Benchmark note**: ISPRS Vaihingen/Potsdam are the standard for VHR urban segmentation. LoveDA is widely used for domain adaptation (urban vs. rural).

## Change Detection Datasets

| Dataset | Image Pairs | Resolution | Area / Scope | Change Type | Format | Access |
|---------|------------|-----------|-------------|------------|--------|--------|
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | 637 pairs | 0.5m | 20 regions in Texas, US | Building construction (5-14 yr span) | PNG (1024x1024) | Free download |
| [WHU Building CD](http://gpcv.whu.edu.cn/data/) | 1 pair (large area) | 0.2m | Christchurch, NZ | Building changes (2012-2016) | TIFF (32507x15354) | Free download |
| [OSCD](https://rcdaudt.github.io/oscd/) | 24 pairs | 10/20/60m | 24 global cities | Urban changes (2015-2018) | GeoTIFF (Sentinel-2, 13 bands) | Free download |
| [S2Looking](https://github.com/S2Looking/Dataset) | 5,000 pairs | 0.5-0.8m | Global rural areas | Building construction + demolition | PNG (1024x1024) | Free download |
| [xBD](https://xview2.org/) | 22,068 pairs | 0.8m | 19 disaster events | Building damage (4-level scale), 850,736 buildings across 45,362 km² | GeoTIFF + GeoJSON | Free (registration) |
| [SYSU-CD](https://github.com/liumency/SYSU-CD) | 20,000 pairs | 0.5m | Hong Kong | Multi-class changes | PNG (256x256) | Free download |

> **Tip**: LEVIR-CD is the most widely benchmarked building change detection dataset. xBD is the standard for post-disaster damage assessment.

## Geospatial Foundation Models

Foundation models pre-trained on large-scale Earth observation data for fine-tuning on downstream tasks.

| Model | Organization | Parameters | Pre-training Data | Architecture | License | URL |
|-------|-------------|-----------|-------------------|-------------|---------|-----|
| [Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) | NASA / IBM | 600M | 4.2M global tiles, HLS (Landsat+Sentinel-2) | ViT + temporal/location embeddings | Apache 2.0 | [HuggingFace](https://huggingface.co/ibm-nasa-geospatial) |
| [Clay v1.5](https://clay-foundation.github.io/model/) | Clay (Development Seed) | ~300M | 70M chips (S1, S2, Landsat, NAIP, MODIS) -- 33.8 TB | ViT MAE | Apache 2.0 | [GitHub](https://github.com/Clay-foundation/model) |
| [SatMAE](https://sustainlab-group.github.io/SatMAE/) | Stanford | ViT-L | fMoW temporal + multi-spectral imagery | ViT MAE (temporal + spectral) | MIT | [GitHub](https://github.com/sustainlab-group/SatMAE) |
| [AlphaEarth Foundations](https://deepmind.google/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/) | Google DeepMind | Undisclosed | Multi-source (optical, radar, LiDAR) | Proprietary | Embeddings free in GEE | [Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) |
| [TorchGeo](https://github.com/microsoft/torchgeo) | Microsoft | Various backbones | Multi-spectral Sentinel-2, Landsat | ResNet, ViT, Swin | MIT | [PyPI](https://pypi.org/project/torchgeo/) |
| [SatCLIP](https://github.com/microsoft/satclip) | Microsoft | -- | Sentinel-2 + location | Contrastive (CLIP-style) | MIT | [GitHub](https://github.com/microsoft/satclip) |

**GEO-Bench Scores**: Prithvi-EO-2.0 achieves **75.6%** average (8% improvement over v1.0). SatMAE achieves up to **14%** improvement in transfer learning over prior methods.

> **Recommendation**: Start with Prithvi-EO-2.0 for best overall performance. Use TorchGeo for the broadest ecosystem of datasets and training pipelines.

## Point Cloud / LiDAR ML Datasets

| Dataset | Points | Classes | Area | Acquisition | Format | Access |
|---------|--------|---------|------|------------|--------|--------|
| [DALES](https://udayton.edu/engineering/research/centers/vision_lab/research/was_702702.php) | ~505M | 8 | 10 km² (Dayton, US) | Aerial LiDAR (~50 pts/m²) | LAS | Free download |
| [Toronto-3D](https://github.com/WeikaiTan/Toronto-3D) | ~78.3M | 8 | 1 km roadway (Toronto) | Mobile Laser Scanning | PLY | Free download |
| [SensatUrban](https://github.com/QingyongHu/SensatUrban) | ~2.85 billion | 13 | 7.6 km² (3 UK cities) | UAV photogrammetry | PLY | Free download |
| [Paris-Lille-3D](https://npm3d.fr/paris-lille-3d) | ~143M | 9 | 1.94 km roadway | Mobile Laser Scanning | PLY | Free download |
| [SemanticKITTI](http://www.semantic-kitti.org/) | ~4.5 billion | 25 (+6 moving) | 39.2 km driving sequences | Velodyne HDL-64E (~120K pts/scan) | BIN + label | Free download |
| [Semantic3D](http://www.semantic3d.net/) | ~4 billion | 8 | Multiple outdoor scenes | Terrestrial Laser Scanning | TXT/LAS | Free download |

## Street-Level Image Datasets

| Dataset | Images | Classes | Coverage | Tasks | Annotation | Access |
|---------|--------|---------|----------|-------|-----------|--------|
| [Mapillary Vistas](https://www.mapillary.com/dataset/vistas) | 25,000 (v2.0) | 124 categories (70 instance) | Global (6 continents) | Semantic + instance + panoptic segmentation | Dense pixel-level | Free (research) |
| [Cityscapes](https://www.cityscapes-dataset.com/) | 5,000 fine + 20,000 coarse | 30 classes | 50 European cities | Semantic/instance segmentation | Pixel-level | Free (registration) |
| [BDD100K](https://bdd-data.berkeley.edu/) | 100,000 frames | 40 classes | US (NY, SF Bay, etc.) | 10 tasks: detection, segmentation, tracking, lane marking | BBox, pixel, lane, drivable | Free (registration) |
| [nuImages](https://www.nuscenes.org/nuimages) | 93,476 | 25 classes | Singapore + Boston | Detection, instance/semantic segmentation | BBox + pixel + attributes | Free (registration) |
| [Global Streetscapes](https://github.com/ualsg/global-streetscapes) | 10 million | Scene-level tags | 688 cities worldwide | Urban analytics, scene understanding | Scene attributes | Free download |

## Geospatial NLP & Geocoding

| Resource | Records | Coverage | Type | Format | Access |
|----------|---------|----------|------|--------|--------|
| [GeoNames](https://www.geonames.org/) | >12 million place names | Global | Hierarchical gazetteer (names, coordinates, admin levels) | TSV / API | Free (CC-BY) |
| [OpenAddresses](https://openaddresses.io/) | >600 million addresses | Global (strongest in EU, US, AU) | Structured addresses with coordinates | CSV / GeoJSON | Free (varies by source) |
| [Who's on First](https://whosonfirst.org/) | ~490 million features | Global | Administrative + venue gazetteer | GeoJSON | Free (CC-BY) |
| [Nominatim](https://nominatim.openstreetmap.org/) | Derived from OSM | Global | Forward/reverse geocoding | API (JSON/XML) | Free (ODbL) |
| [libpostal](https://github.com/openvenues/libpostal) | -- | 60+ languages | Address parsing & normalization (CRF-based) | Library | Free (MIT) |
| [Pelias](https://pelias.io/) | -- | Global | Full-text geocoding engine (OSM + OpenAddresses + GeoNames + WoF) | Self-hosted API | Free (MIT) |

## Climate & Weather ML Datasets

| Dataset | Resolution | Temporal | Size | Task | Access |
|---------|-----------|----------|------|------|--------|
| [WeatherBench 2](https://weatherbench2.readthedocs.io/) | 0.25-5.625 deg | 1959-2023, 6-hourly | ~2 TB+ | Comprehensive weather forecasting benchmark | Free (GCS) |
| [ClimateBench](https://github.com/duncanwp/ClimateBench) | 2.5 x 1.875 deg | Historical + SSP scenarios, annual | ~2 GB | Climate model emulation (NorESM2 / CMIP6) | Free download |
| [ClimateNet](https://gmd.copernicus.org/articles/14/107/2021/) | 25 km | Historical simulation, 3-hourly | ~50 GB | Extreme weather segmentation (tropical cyclone + atmospheric river) | Free download |
| [GraphCast Training Data](https://github.com/google-deepmind/graphcast) | 0.25 deg | 1979-2017, 6-hourly | ~1 TB | Global weather forecasting | Free (GCS) |

> **Recommendation**: Start with WeatherBench 2 for weather prediction research. ClimateBench for climate projection emulation.

## Synthetic Geospatial Data

| Source | Type | Content | Key Details | Access |
|--------|------|---------|------------|--------|
| [CARLA Simulator](https://carla.org/) | Driving simulator | Procedural urban environments (UE4/UE5), 8+ maps | Camera, LiDAR, GNSS, IMU, radar output. Unlimited frames. | Free (MIT) |
| [KITTI-CARLA](https://npm3d.fr/kitti-carla) | Synthetic driving dataset | 7 sequences x 5,000 frames = 35,000 frames | Velodyne HDL-64 + stereo cameras in KITTI format | Free download |
| [RarePlanes Synthetic](https://www.cosmiqworks.org/RarePlanes/) | Synthetic satellite imagery | 50,000+ CG-rendered aircraft on satellite backgrounds | Fine-grained aircraft attributes | Free (AWS) |
| Diffusion-based SAR synthesis | Generative AI | DDPM/diffusion models for SAR image generation | Outperforms GANs for SAR training data augmentation | Research code |

## Benchmark Platforms & Competitions

### Platforms

| Platform | Datasets | Key Features | URL |
|----------|---------|-------------|-----|
| [Radiant MLHub](https://mlhub.earth/) | 50+ curated geospatial ML datasets | STAC-compliant API, Python client, model catalog | [mlhub.earth](https://mlhub.earth/) |
| [TorchGeo](https://github.com/microsoft/torchgeo) | 125+ datasets with PyTorch loaders | Pre-trained models, samplers, transforms, reproducible benchmarks | [torchgeo.readthedocs.io](https://torchgeo.readthedocs.io/) |
| [GEO-Bench](https://github.com/ServiceNow/geo-bench) | 6 classification + 6 segmentation tasks | Standardized evaluation for geospatial foundation models | [GitHub](https://github.com/ServiceNow/geo-bench) |

### Active Competition Series

| Competition | Organizer | Recent Topics (2024-2025) | URL |
|-------------|-----------|--------------------------|-----|
| IEEE GRSS Data Fusion Contest | IEEE GRSS | 2024: Rapid Flood Mapping; 2025: All-Weather Land Cover | [grss-ieee.org](https://www.grss-ieee.org/technical-committees/image-analysis-and-data-fusion/) |
| SpaceNet Challenges | SpaceNet LLC / AWS | SpaceNet 8: Flooded infrastructure detection | [spacenet.ai](https://spacenet.ai/challenges/) |
| Kaggle Remote Sensing | Various | Ship detection, building segmentation, crop classification | [kaggle.com](https://www.kaggle.com/) |
| DrivenData | DrivenData | Disaster response, land cover, agricultural monitoring | [drivendata.org](https://www.drivendata.org/) |
| xView Challenges | DIU / DoD | xView2: Building damage; xView3: Dark vessel detection | [xview2.org](https://xview2.org/) |

## Getting Started Recommendations

### By Skill Level

| Level | Datasets | Why |
|-------|---------|-----|
| Beginner | EuroSAT, UC Merced, Cityscapes | Small, well-documented, trainable on consumer GPU |
| Intermediate | BigEarthNet, DOTA v1.0, LEVIR-CD, LoveDA | Moderate scale, established benchmarks, active leaderboards |
| Advanced | fMoW, SpaceNet, xBD, SEN12MS | Multi-modal, multi-temporal, real-world complexity |
| Research | Prithvi-EO-2.0 fine-tuning, Clay embeddings, WeatherBench 2 | Foundation model workflows, large-scale experiments |

### By Task

| Task | Top Dataset | Runner-Up |
|------|-----------|-----------|
| Scene Classification | BigEarthNet (large) / EuroSAT (small) | fMoW, RESISC-45 |
| Building Extraction | SpaceNet 2 | Inria, Microsoft Building Footprints |
| Object Detection | DOTA v2.0 | FAIR1M, xView |
| Semantic Segmentation | LoveDA | ISPRS Potsdam, DeepGlobe |
| Change Detection | LEVIR-CD | xBD (damage), OSCD (Sentinel-2) |
| Ship Detection | HRSC2016 (optical) / SSDD (SAR) | Airbus Ship Detection |
| Point Cloud Segmentation | SemanticKITTI (driving) / DALES (aerial) | SensatUrban |
| Weather Forecasting | WeatherBench 2 | ERA5 subsets |
| Street Scene | BDD100K | Cityscapes, Mapillary Vistas |

## Further Reading

- [Awesome Satellite Imagery Datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets) -- curated dataset list
- [Satellite Image Deep Learning](https://github.com/satellite-image-deep-learning/datasets) -- comprehensive catalog
- [Awesome Remote Sensing Change Detection](https://github.com/wenhwu/awesome-remote-sensing-change-detection) -- datasets, methods, tools
- [TorchGeo Documentation](https://torchgeo.readthedocs.io/) -- Microsoft's geospatial deep learning library
