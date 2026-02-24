# Transportation & Autonomous Driving

Geospatial data sources for transportation networks, autonomous driving research, traffic analysis, and mobility planning. Covers HD maps, LiDAR point clouds, vehicle trajectories, traffic signals, and simulation environments.

> **Quick Picks**
> - **SOTA**: [Waymo Open Dataset](https://waymo.com/open/) -- largest multi-modal AV dataset with 570+ hours of LiDAR, camera, and motion data
> - **Free Best**: [Overture Transportation](https://overturemaps.org) -- global road network in GeoParquet with monthly updates, queryable via DuckDB
> - **Fastest Setup**: [SUMO Simulator](https://eclipse.dev/sumo/) -- open-source traffic simulation, imports OSM road networks directly

> **Note on commercial HD maps**: HERE HD Live Map, TomTom Orbis Maps, and Mobileye REM are enterprise-only products requiring OEM contracts. Researchers should focus on open datasets and standards like Lanelet2 and OpenDRIVE.

## HD Maps & Lane-Level Data

### Commercial HD Map Products

These are enterprise products requiring OEM/partner contracts. Listed for reference only.

| Product | Provider | Content | Coverage | Format | Cost |
|---------|----------|---------|----------|--------|------|
| HD Live Map | HERE Technologies | Lane-level geometry (borders, center lines), road signs, signals, speed limits, lane markings, surface type, curvature, slope, banking. Near-real-time fleet sensor updates. | 50+ countries (major highways NA/EU/Asia) | NDS.Live, ADASIS v3 | Enterprise only (basic APIs: 5K txn/mo free) |
| Orbis Maps for AD | TomTom | Lane-level topology, boundaries with sub-meter accuracy, 3D geometry, speed restrictions, signs, road furniture. AutoStream delta delivery. | 45+ countries (major roads) | NDS.Live, AutoStream | Enterprise only (dev tier: 50K tiles/day free) |
| Mobileye REM / Roadbook | Mobileye (Intel) | Crowdsourced road model from camera-equipped fleet vehicles (~10 KB/km): lane geometry, road edges, landmarks, barriers. | Global (where Mobileye cars drive) | Proprietary Roadbook format | Enterprise only, bundled with hardware |

### Open Standards & Frameworks

| Standard | Provider | Description | Format | Cost | URL |
|----------|----------|-------------|--------|------|-----|
| OpenDRIVE v1.8.1 | ASAM e.V. | XML road network description for simulation: road geometry, lanes, elevation, markings, signals, junctions. The de facto standard for AV simulation. | `.xodr` (XML) | Free (registration) | [asam.net](https://www.asam.net/standards/detail/opendrive/) |
| Lanelet2 | FZI / KIT | C++/Python HD map framework for automated driving. Models lanes as "lanelets" bounded by linestrings with regulatory elements. Used by Autoware. | `.osm` (extended OSM XML) | Free (BSD) | [GitHub](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) |
| OpenSCENARIO | ASAM e.V. | Dynamic driving scenario descriptions (vehicle behaviors, maneuvers, triggers) for ADAS/AD testing. Companion to OpenDRIVE. | `.xosc` (XML) | Free (registration) | [asam.net](https://www.asam.net/standards/detail/openscenario/) |

> **Tip**: For lane-level data without enterprise contracts, use Lanelet2 maps from OSM or HD maps bundled with open AV datasets (Argoverse 2, nuScenes, Waymo). CARLA generates OpenDRIVE maps procedurally.

## Traffic Flow & Real-Time Data

### Commercial Traffic APIs

| API | Provider | Content | Free Tier | Paid Rate | URL |
|-----|----------|---------|-----------|-----------|-----|
| Traffic API v7 | HERE | Real-time flow (speed, jam factor per segment), incidents, predictive speeds, density tiles. 190+ countries, ~1 min updates. | 5,000 txn/mo | $2.50/1K txn (flow); $5/1K (advanced) | [developer.here.com](https://developer.here.com) |
| Traffic API | TomTom | Real-time flow (current/free-flow speed), incidents, flow tiles, historical stats. 200+ countries, ~1 min updates. | 50K tile + 2.5K non-tile/day | $0.08/1K tile overage | [developer.tomtom.com](https://developer.tomtom.com) |
| Routes API | Google Maps | Traffic-aware routing with real-time and predictive travel times. Three tiers: Basic, Advanced (traffic-aware), Preferred (traffic polylines). | 10K events/mo (Essentials) | Basic $5/1K, Advanced $10/1K, Preferred $15/1K | [developers.google.com](https://developers.google.com/maps/documentation/routes) |

### Open Traffic Sensor Networks

| Dataset | Provider | Content | Coverage | Format | Access | URL |
|---------|----------|---------|----------|--------|--------|-----|
| Caltrans PeMS | California DOT | Real-time and historical flow (volume, occupancy, speed) from 40,000+ detector stations. 30-second granularity, archives back to 1999. | California, USA | CSV, XML | Free (registration) | [pems.dot.ca.gov](https://pems.dot.ca.gov) |
| UK National Highways WebTRIS | National Highways | Real-time motorway/trunk road traffic data (speed, flow, journey times) via DATEX II feed. | England, UK | DATEX II (XML), JSON | Free (API key) | [webtris.nationalhighways.co.uk](https://webtris.nationalhighways.co.uk) |
| NYC TLC Trip Records | NYC TLC | Yellow/green taxi and FHV trip records: pickup/dropoff times, locations, distances, fares, passenger counts. Monthly releases. | New York City | CSV, Parquet | Free | [nyc.gov/tlc](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |

### Archived Datasets

| Dataset | Status | Notes |
|---------|--------|-------|
| Uber Movement | **Discontinued (~2022)** | Archived datasets still accessible on Kaggle and data.world. Originally covered ~50 major cities with aggregated travel times. |

## LiDAR Point Cloud Datasets for AV

Labeled 3D point clouds for training perception algorithms (object detection, segmentation, motion prediction).

| Dataset | Provider | Sensors | Size | Key Content | Format | License | Status | URL |
|---------|----------|---------|------|------------|--------|---------|--------|-----|
| Waymo Open Dataset | Waymo | 5 LiDAR, 5 cameras | ~2.5 TB | 3D/2D bboxes (vehicles, pedestrians, cyclists), semantic/panoptic segmentation. Motion: 104K+ segments, 570+ hrs at 10Hz. WOD-E2E: 4,021 segments. | TFRecord, Parquet | Non-commercial research | **Active** (v1.3.1, Oct 2025) | [waymo.com/open](https://waymo.com/open/) |
| nuScenes | Motional | 1 LiDAR, 6 cameras, 5 radar | ~350 GB | 1.4M camera images, 390K LiDAR sweeps, 1.4M radar sweeps. 23 object classes, 3D bboxes at 2Hz. Includes HD map. | `.bin`, JSON, PNG | CC BY-NC-SA 4.0 | **Active** (devkit v1.2.0) | [nuscenes.org](https://www.nuscenes.org) |
| Argoverse 2 | Argo AI / CMU | 2 LiDARs, 7 ring + 2 stereo cameras | ~1 TB | Sensor (1K scenarios), Motion Forecasting (250K), LiDAR (20K unannotated), Map Change (1K). Rich lane-level HD map. | Feather (Arrow), JSON, PNG | CC BY-NC-SA 4.0 | **Active** | [argoverse.org](https://www.argoverse.org/av2.html) |
| KITTI | KIT / Toyota | 1 Velodyne HDL-64E, 4 cameras | ~180 GB | 7,481 annotated frames for 2D/3D detection, stereo, optical flow, visual odometry. Karlsruhe, Germany. | `.bin`, PNG, TXT | CC BY-NC-SA 3.0 | Legacy (no new data) | [cvlibs.net](https://www.cvlibs.net/datasets/kitti/) |
| A2D2 | Audi | 5 LiDAR, 6 cameras (360 deg) | ~2.3 TB | 41,277 frames with semantic segmentation (38 classes), 12,497 with 3D bboxes, 392,556 sequential unannotated. Three German cities. | HDF5, PNG, JSON | CC BY-ND 4.0 | Available (static) | [a2d2.audi](https://www.a2d2.audi) |
| Lyft Level 5 | Woven Planet (Toyota) | 3 LiDAR, 7 cameras | ~100 GB | 1,000+ hrs driving, 16K miles in Palo Alto. Prediction: 1,000 scenes with trajectories. HD semantic map. | Zarr, Protobuf | CC BY-NC-SA 4.0 | Available (limited maintenance) | [woven-planet.github.io/l5kit](https://woven-planet.github.io/l5kit/) |

> **Choosing a dataset**: Waymo for motion prediction/planning (largest, most active). nuScenes for 3D detection benchmarks (community standard). Argoverse 2 for map-aware perception (richest HD maps). KITTI as a baseline (shows age -- 2012 sensors).

## Road Network Datasets

| Dataset | Provider | Content | Coverage | Format | Update | Cost | URL |
|---------|----------|---------|----------|--------|--------|------|-----|
| OpenStreetMap Roads | OSM Community | Road type, name, speed limit, lanes, surface, one-way, turn restrictions, bridge/tunnel. Richest open road dataset. | Global | PBF, SHP (Geofabrik), GeoParquet (Overture) | Continuous | Free (ODbL) | [download.geofabrik.de](https://download.geofabrik.de) |
| Overture Transportation | Overture Maps Foundation | Road, rail, water segments with connectors. OSM + TomTom merged. Road class, speed limits, access, surface, lanes. | Global | GeoParquet | Monthly | Free (ODbL) | [docs.overturemaps.org](https://docs.overturemaps.org/guides/transportation/) |
| TIGER/Line Roads | US Census Bureau | All US roads by MAF/TIGER class. Road name, type, address ranges, county FIPS. | USA (complete) | SHP, Geodatabase | Annual | Free (public domain) | [census.gov](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) |
| GRIP Global Roads | GLOBIO / PBL Netherlands | 60+ public sources combined. Road type (5 classes), surface, estimated speed. For ecological/environmental modeling. | Global | SHP, GeoJSON | Static (v4, 2018) | Free (CC BY 4.0) | [globio.info](https://www.globio.info/download-grip-dataset) |

> **Overture tip**: Query directly with DuckDB: `SELECT * FROM read_parquet('s3://overturemaps-us-west-2/release/.../theme=transportation/type=segment/*') WHERE bbox.xmin > -122.5 AND bbox.xmax < -122.3;`

## Vehicle Trajectory & GPS Trace Data

### Taxi & Ride-Hailing Trajectories

| Dataset | Content | Coverage | Size | Format | Access | Status |
|---------|---------|----------|------|--------|--------|--------|
| T-Drive | GPS trajectories from 10,357 taxis over 1 week (Feb 2008). ~15M points: taxi ID, timestamp, lon, lat. | Beijing, China | ~800 MB | CSV | Free download (MS Research / Kaggle) | Static (2008) |
| Porto Taxi Trajectories | 1.7M complete taxi trips over 1 year: trip ID, taxi ID, timestamp, call type, GPS polyline. | Porto, Portugal | ~500 MB | CSV | Free (Kaggle) | Static (2013-2014) |
| NYC TLC Trip Records | Yellow/green taxi and FHV records: pickup/dropoff times, locations, distances, fares. Monthly. | New York City | ~200 GB (multi-year) | CSV, Parquet | Free | **Active** (monthly) |

### Research Trajectory Datasets

| Dataset | Content | Coverage | Format | Access | URL |
|---------|---------|----------|--------|--------|-----|
| NGSIM | Vehicle trajectories at 10Hz from overhead cameras: vehicle ID, X/Y coords, velocity, acceleration, lane, headway. US 101, I-80, Lankershim, Peachtree. | 4 US sites | CSV | Free | [data.transportation.gov](https://data.transportation.gov/stories/s/Next-Generation-Simulation-NGSIM-Open-Data/i5zb-xe34/) |
| highD | Drone-recorded highway trajectories. 110,500+ vehicles, 6 locations, 25 fps. Lane-accurate position, speed, acceleration, dimensions, TTC. | 6 German highway segments | CSV | Free (academic, registration) | [levelxdata.com](https://levelxdata.com/highd-dataset/) |
| inD | Drone-recorded intersection trajectories. 11,500+ road users (vehicles, cyclists, pedestrians), 25 fps, lane-level. | 4 German intersections | CSV | Free (academic, registration) | [levelxdata.com](https://levelxdata.com/ind-dataset/) |
| rounD | Drone-recorded roundabout trajectories. Vehicles, cyclists, pedestrians, 25 fps. | 3 German roundabouts | CSV | Free (academic, registration) | [levelxdata.com](https://levelxdata.com/round-dataset/) |

> **highD/inD/rounD** share the same format and tooling. Together they cover highway merging, intersection negotiation, and roundabout behavior -- the three key AV planning scenarios.

## Traffic Signal & Sign Data

### Traffic Sign Datasets

| Dataset | Content | Coverage | Format | Access | URL |
|---------|---------|----------|--------|--------|-----|
| Mapillary Traffic Sign Dataset | 100,000 images with 320,000+ annotated traffic signs across 300+ classes. Global sign taxonomy. | Global (crowdsourced) | JPEG + JSON | Free (CC BY-SA, registration) | [mapillary.com](https://www.mapillary.com/dataset/trafficsign) |
| Mapillary Map Features API | Point-level detections of signs, signals, poles from street-level imagery with class label, position, confidence. | Global (where images exist) | GeoJSON (API) | Free (Meta developer account) | [mapillary.com/developer](https://www.mapillary.com/developer/api-documentation) |
| GTSRB / GTSDB | German Traffic Sign Recognition/Detection Benchmark: 50,000+ images of 43 sign classes. Standard benchmark. | Germany | PPM + CSV | Free | [benchmark.ini.rub.de](https://benchmark.ini.rub.de/gtsrb_news.html) |
| MUTCD 11th Edition | Official US traffic sign specifications: dimensions, colors, symbols, placement rules. Effective Jan 2024, mandatory Jan 2026. | USA | PDF, SVG/DXF | Free | [mutcd.fhwa.dot.gov](https://mutcd.fhwa.dot.gov/) |

### Signal Phase & Timing (SPaT) Data

| Dataset | Content | Coverage | Format | Access |
|---------|---------|----------|--------|--------|
| Tampa CV Pilot SPaT | Signal Phase and Timing messages from connected vehicle pilot. Signal state, timing, intersection geometry (MAP messages). | Tampa, FL | JSON, PCAP | Free ([catalog.data.gov](https://catalog.data.gov/dataset/tampa-cv-pilot-signal-phasing-and-timing-spat-sample)) |
| Utah SPaT Data | Timestamped signal states for up to 8 signal groups per intersection. | Utah (select intersections) | CSV | Free ([catalog.data.gov](https://catalog.data.gov/dataset/traffic-signal-change-and-clearance-interval-pooled-fund-study-utah-spat-data)) |
| Open Traffic Lights | Real-time traffic light state as Linked Open Data. | Belgium (pilot) | JSON-LD | Free ([opentrafficlights.org](https://opentrafficlights.org)) |

> **Reality check**: Comprehensive, real-time signal timing data remains extremely hard to obtain. Most signal timing plans are held by municipal agencies and not published as open data.

## Parking Data

| Dataset | Content | Coverage | Format | Access | URL |
|---------|---------|----------|--------|--------|-----|
| OpenStreetMap Parking | Parking lots (`amenity=parking`), street parking (`parking:lane=*`), garages, capacity, fee, access, surface. Quality varies by region. | Global (best in Europe) | PBF (Overpass API) | Free (ODbL) | [overpass-turbo.eu](https://overpass-turbo.eu) |
| SharedStreets Parking | Curb regulation data standard linking parking rules to street segments. Machine-readable curb data. | USA (pilot cities) | GeoJSON | Free | [sharedstreets.io](https://sharedstreets.io) |

**Notable municipal parking portals**: San Francisco ([data.sfgov.org](https://data.sfgov.org) -- SFpark real-time occupancy), NYC ([data.cityofnewyork.us](https://data.cityofnewyork.us)), LA ([data.lacity.org](https://data.lacity.org)), London ([tfl.gov.uk](https://tfl.gov.uk/info-for/open-data-users/)), Berlin ([daten.berlin.de](https://daten.berlin.de)).

## Autonomous Driving Simulation Environments

| Simulator | Provider | Engine | Key Features | License | Status | URL |
|-----------|----------|--------|-------------|---------|--------|-----|
| CARLA | CVC / Intel / Toyota | Unreal Engine 5 | High-fidelity rendering, multi-sensor (LiDAR, camera, radar, IMU, GNSS), ROS1/2 bridge, Python/C++ API, traffic manager, weather control, procedural maps. Largest AV research community. | MIT | **Active** (v0.10.0 UE5) | [carla.org](https://carla.org) |
| SUMO | DLR (German Aerospace) | Custom (C++) | Microscopic multi-modal traffic sim. TraCI real-time API, OSM import, signal control, emission modeling, demand generation. No 3D rendering. | EPL-2.0 | **Active** (v1.26.0) | [eclipse.dev/sumo](https://eclipse.dev/sumo/) |
| AWSIM | TIER IV | Unity | Designed for Autoware. Native ROS2, Lanelet2 maps, LiDAR/camera/GNSS sensors, digital twin approach. | Apache 2.0 | **Active** | [GitHub](https://github.com/tier4/AWSIM) |
| LGSVL / SVL | LG Electronics | Unity | ROS1/2 native, Autoware/Apollo integration, cloud scenario testing. | Open source | **Discontinued** (Jan 2022, code on GitHub) | [GitHub](https://github.com/lgsvl/simulator) |
| AirSim | Microsoft Research | UE4 | Drones + ground vehicles, physics-based, photorealistic. | MIT | **Archived** (2022). Community fork: [IAMAI](https://github.com/iamaisim/ProjectAirSim) | [GitHub](https://microsoft.github.io/AirSim/) |

### Simulator Comparison

| Feature | CARLA | SUMO | AWSIM |
|---------|-------|------|-------|
| 3D Rendering | High fidelity (UE5) | None (2D) | Medium (Unity) |
| Traffic Sim | Built-in traffic manager | Full microscopic sim | Via Autoware |
| Sensors | LiDAR, camera, radar, GNSS, IMU | N/A | LiDAR, camera, GNSS |
| ROS | ROS1/ROS2 bridge | Via TraCI | ROS2 native |
| Best For | Perception + planning research | Traffic engineering + signals | Autoware development |
| Map Format | OpenDRIVE | SUMO XML / OSM | Lanelet2 / PCD |

## Download Tools

| Tool | Purpose | Install |
|------|---------|---------|
| DuckDB | Query Overture GeoParquet from S3 | `pip install duckdb` |
| Waymo Open Dataset SDK | Parse Waymo TFRecords | `pip install waymo-open-dataset-tf-2-12-0` |
| nuScenes devkit | Load/visualize nuScenes data | `pip install nuscenes-devkit` |
| av2 API | Argoverse 2 data and HD maps | `pip install av2` |
| osmium | Fast OSM PBF processing | `pip install osmium` |
| Open3D | 3D point cloud visualization | `pip install open3d` |

## Data Size Planning

| Dataset | Download Size | Recommended Min RAM |
|---------|-------------|-------------------|
| Waymo Open (full) | ~2.5 TB | 32 GB |
| nuScenes (full) | ~350 GB | 16 GB |
| Argoverse 2 (sensor) | ~1 TB | 32 GB |
| KITTI (all) | ~180 GB | 8 GB |
| A2D2 (full) | ~2.3 TB | 32 GB |
| Overture Transportation (global) | ~20 GB | 8 GB |
