# Geocoding, Routing & Network Analysis

**Quick Picks**

| Need | Tool |
|---|---|
| Best open-source routing | Valhalla (SOTA) |
| Best open-source geocoding | Pelias (SOTA) |
| Fastest free setup (routing + geocoding) | OSRM + Nominatim |
| Quickest Python prototype | `geopy` + OSRM demo server |

---

## Geocoding Engines

Geocoding is deceptively hard. Address data is messy, inconsistent across countries, and geocoder accuracy varies wildly by region. This section covers what actually works in production.

### Nominatim (OSM Geocoder)

Nominatim is the reference geocoder for OpenStreetMap data. It handles forward geocoding (address → coordinates), reverse geocoding (coordinates → address), and structured queries.

**Public API**

```bash
# Forward geocoding
curl "https://nominatim.openstreetmap.org/search?q=1600+Pennsylvania+Ave+NW,+Washington+DC&format=jsonv2&addressdetails=1" \
  -H "User-Agent: MyGISApp/1.0 (contact@example.com)"

# Reverse geocoding
curl "https://nominatim.openstreetmap.org/reverse?lat=38.8977&lon=-77.0366&format=jsonv2" \
  -H "User-Agent: MyGISApp/1.0"

# Structured search with regional bias
curl "https://nominatim.openstreetmap.org/search?street=Pennsylvania+Ave&city=Washington&country=US&format=jsonv2&limit=5&viewbox=-77.12,38.79,-76.91,38.99&bounded=1"

# Country filter
curl "https://nominatim.openstreetmap.org/search?q=Berlin&format=jsonv2&countrycodes=de"
```

**Rate limits**: 1 request/second on the public API. No bulk geocoding. For anything serious, self-host.

**Self-hosted deployment**

Minimum hardware for full planet: 64GB RAM, 1TB NVMe SSD (planet import ~700GB on disk), 48h import time. For a country extract, you need a fraction of this.

```bash
# Download country extract
wget https://download.geofabrik.de/europe/germany-latest.osm.pbf

# Docker setup (mediagis/nominatim image)
docker run -it \
  -e PBF_URL=https://download.geofabrik.de/europe/germany-latest.osm.pbf \
  -e REPLICATION_URL=https://download.geofabrik.de/europe/germany-updates/ \
  -e IMPORT_WIKIPEDIA=true \
  -p 8080:8080 \
  -v nominatim-data:/var/lib/postgresql/14/main \
  --name nominatim \
  mediagis/nominatim:4.4
```

**Full Docker Compose for production**

```yaml
version: '3.8'
services:
  nominatim:
    image: mediagis/nominatim:4.4
    restart: always
    ports:
      - "8080:8080"
    environment:
      PBF_URL: https://download.geofabrik.de/north-america/us/california-latest.osm.pbf
      REPLICATION_URL: https://download.geofabrik.de/north-america/us/california-updates/
      NOMINATIM_PASSWORD: very_secure_password
    volumes:
      - nominatim-data:/var/lib/postgresql/14/main
    shm_size: 1gb

volumes:
  nominatim-data:
```

**Useful output fields**

```json
{
  "place_id": 297985438,
  "osm_type": "way",
  "osm_id": 238241022,
  "place_rank": 30,
  "importance": 0.7,
  "boundingbox": ["38.897", "38.898", "-77.038", "-77.035"],
  "display_name": "White House, 1600, Pennsylvania Avenue Northwest...",
  "address": {
    "house_number": "1600",
    "road": "Pennsylvania Avenue Northwest",
    "city": "Washington",
    "state": "District of Columbia",
    "postcode": "20500",
    "country_code": "us"
  }
}
```

- `place_rank`: lower = more general (4 = country, 30 = house number)
- `importance`: 0–1, derived from Wikipedia links to the place
- `osm_id` + `osm_type`: lets you link back to raw OSM data

**Trick: Photon for autocomplete, Nominatim for final resolution**

Photon (by Komoot) is an Elasticsearch-based geocoder that uses Nominatim import data but returns results much faster and supports fuzzy matching — ideal for live search-as-you-type. Use Photon for the autocomplete dropdown, then resolve the selected result with Nominatim for full structured data.

```bash
# Photon autocomplete
curl "https://photon.komoot.io/api/?q=Berlin+Haupt&limit=5&lang=en"

# Photon self-hosted with Docker
docker run -p 2322:2322 \
  -v /path/to/photon_data:/photon/photon_data \
  komoot/photon:latest

# API
curl "http://localhost:2322/api/?q=Hauptbahnhof&limit=10&bbox=13.088,52.338,13.761,52.675"
```

---

### Pelias (Modular Geocoder)

Pelias is the most production-ready open-source geocoder. It ingests multiple data sources simultaneously and provides better address-level accuracy than Nominatim alone.

**Data sources**
- OSM (points of interest, boundaries)
- Who's On First (administrative hierarchy)
- OpenAddresses (address-level coverage, 600M+ addresses)
- Geonames (place names, populated places)

**Docker Compose deployment**

```bash
git clone https://github.com/pelias/docker.git
cd docker
cp -r projects/portland .  # or any region project
cd portland
# Edit pelias.json to set your data directory
docker-compose pull
docker-compose run --rm prepare                    # download data
docker-compose run --rm schema                     # create Elasticsearch index
docker-compose run --rm whosonfirst               # import admin data
docker-compose run --rm openaddresses             # import addresses
docker-compose run --rm openstreetmap              # import OSM
docker-compose run --rm transit                    # import transit
docker-compose run --rm placeholder               # build placeholder service
docker-compose up -d                               # start all services
```

**API calls**

```bash
# Forward geocoding
curl "http://localhost:4000/v1/search?text=500+Sansome+St+San+Francisco"

# Autocomplete
curl "http://localhost:4000/v1/autocomplete?text=golden+ga"

# Reverse
curl "http://localhost:4000/v1/reverse?point.lat=37.7749&point.lon=-122.4194"

# Structured search
curl "http://localhost:4000/v1/search/structured?address=500+Sansome+St&locality=San+Francisco&region=CA"

# Coarse geocoding (no street-level)
curl "http://localhost:4000/v1/search?text=California&layers=region,country"

# Filter to specific country + boundary
curl "http://localhost:4000/v1/search?text=Museum&boundary.country=GBR&focus.point.lat=51.5&focus.point.lon=-0.12"
```

**Response confidence fields**

```json
{
  "features": [{
    "properties": {
      "confidence": 0.95,
      "match_type": "exact",
      "accuracy": "point",
      "source": "openaddresses",
      "layer": "address"
    }
  }]
}
```

Use `confidence` as a threshold for automatic acceptance vs. manual review in batch geocoding pipelines.

---

### libpostal

libpostal is a C library (with Python bindings via `postal`) trained on tens of millions of OSM addresses worldwide. It does two things: normalize messy address strings into canonical forms, and parse addresses into structured components.

**Installation**

```bash
# Install libpostal C library first
git clone https://github.com/openvenues/libpostal
cd libpostal
./bootstrap.sh
./configure --datadir=/usr/local/share/libpostal
make -j4
sudo make install

# Python bindings
pip install postal
```

**Address normalization (expand_address)**

```python
from postal.expand import expand_address

# Produces all canonical variations
variations = expand_address("30 W 26th St Fl 7")
# ['30 west 26th street floor 7', '30 west 26th street florida 7', ...]

# Use the first result as canonical form before geocoding
canonical = expand_address("Hauptstr. 15, München")[0]
# 'hauptstrasse 15 munchen'
```

**Address parsing (parse_address)**

```python
from postal.parser import parse_address

result = parse_address("The Regency Hotel, 540 Park Avenue, New York, NY 10065")
# [('the regency hotel', 'house'), ('540', 'house_number'),
#  ('park avenue', 'road'), ('new york', 'city'),
#  ('ny', 'state'), ('10065', 'postcode')]

# Build geocoder input from components
components = dict(result)
geocoder_query = f"{components.get('house_number','')} {components.get('road','')} {components.get('city','')}"
```

**Trick: Pre-process with libpostal before any geocoding API call**

Running every address through `expand_address()` before geocoding routinely improves match rates by 15–30% in production pipelines, especially for non-English addresses, abbreviated street types, and non-ASCII characters.

```python
from postal.expand import expand_address
from postal.parser import parse_address
import geopy
from geopy.geocoders import Nominatim

def geocode_robust(raw_address: str) -> dict:
    # Step 1: normalize
    expanded = expand_address(raw_address)
    best = expanded[0] if expanded else raw_address

    # Step 2: parse into components for structured query
    components = dict(parse_address(best))

    geolocator = Nominatim(user_agent="my-app")

    # Step 3: try structured first, fall back to free-form
    result = geolocator.geocode(
        query={
            "street": f"{components.get('house_number','')} {components.get('road','')}".strip(),
            "city": components.get("city", ""),
            "state": components.get("state", ""),
            "postalcode": components.get("postcode", ""),
            "country": components.get("country", ""),
        }
    )
    if result is None:
        result = geolocator.geocode(best)

    return result
```

---

### Commercial Geocoder Comparison

| Service | Accuracy | Global Coverage | Free Tier | Batch API | Reverse | Autocomplete | Approx. Pricing |
|---|---|---|---|---|---|---|---|
| Google Geocoding API | Excellent | Best | $200 credit/mo | No (must loop) | Yes | Places API | $5/1000 |
| HERE Geocoding | Excellent | Excellent | 250K req/mo | Yes | Yes | Yes | $1/1000 after |
| Mapbox Geocoding | Very Good | Very Good | 100K req/mo | No | Yes | Yes | $0.75/1000 |
| ArcGIS World Geocoder | Very Good | Good | 20K credits/mo | Yes | Yes | Yes | Credits-based |
| TomTom Geocoding | Good | Very Good | 2500 req/day | Yes | Yes | Yes | $0.42/1000 |
| Pelias (self-hosted) | Depends on data | OSM-limited | Unlimited | Yes | Yes | Yes | Infra cost only |
| Nominatim (self-hosted) | Good | OSM-limited | Unlimited | Yes | Yes | Via Photon | Infra cost only |

**When to use commercial**: Addresses in countries with poor OSM coverage (much of Africa, parts of Asia/LatAm). For US, Europe, and urban areas globally, Pelias + OpenAddresses rivals Google.

---

### geopy (Python Unified Geocoding Client)

geopy wraps 20+ geocoding services behind a single consistent API.

```python
pip install geopy
```

**Basic usage**

```python
from geopy.geocoders import Nominatim, GoogleV3, Pelias, Photon, ArcGIS, HERE

# Swap services without changing downstream code
geolocator = Nominatim(user_agent="my-app/1.0")
# geolocator = GoogleV3(api_key="YOUR_KEY")
# geolocator = Pelias(domain="localhost:4000")
# geolocator = Photon(user_agent="my-app/1.0")

location = geolocator.geocode("Eiffel Tower, Paris")
print(location.address, location.latitude, location.longitude)

# Reverse
location = geolocator.reverse("48.8584, 2.2945")
print(location.address)
```

**Batch geocoding with RateLimiter**

```python
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

df = pd.read_csv("addresses.csv")  # column 'address'

geolocator = Nominatim(user_agent="batch-geocoder/1.0")

# RateLimiter wraps geocode() to respect 1 req/sec
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.1,    # slightly over 1s for safety
    max_retries=3,
    error_wait_seconds=5.0,
    return_value_on_exception=None
)

tqdm.pandas(desc="Geocoding")
df["location"] = df["address"].progress_apply(geocode)
df["latitude"] = df["location"].apply(lambda x: x.latitude if x else None)
df["longitude"] = df["location"].apply(lambda x: x.longitude if x else None)
df["confidence"] = df["location"].apply(lambda x: x.raw.get("importance") if x else None)

# Flag low-confidence results for manual review
df["needs_review"] = df["confidence"] < 0.5
df.to_csv("geocoded.csv", index=False)
```

**Trick: Parallel geocoding against self-hosted Nominatim**

On a self-hosted instance you're not rate-limited, so use async or threadpool:

```python
import asyncio
import aiohttp
import pandas as pd

async def geocode_one(session, address, semaphore):
    async with semaphore:
        url = "http://localhost:8080/search"
        params = {"q": address, "format": "jsonv2", "limit": 1}
        async with session.get(url, params=params) as resp:
            results = await resp.json()
            return results[0] if results else None

async def batch_geocode(addresses, concurrency=10):
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [geocode_one(session, addr, semaphore) for addr in addresses]
        return await asyncio.gather(*tasks)

df = pd.read_csv("addresses.csv")
results = asyncio.run(batch_geocode(df["address"].tolist(), concurrency=20))
```

---

## Routing Engines

### OSRM (Open Source Routing Machine)

OSRM is the fastest open-source routing engine. Written in C++, it uses Contraction Hierarchies (CH) for query times under 1ms. Trade-off: flexibility. Routing profiles are compiled in, and time-dependent routing requires the MLD pipeline.

**Self-hosted setup (full pipeline)**

```bash
# Download region
wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf

# Using Docker (recommended)
# Step 1: Extract
docker run -t -v "${PWD}:/data" osrm/osrm-backend \
  osrm-extract -p /opt/osrm/profiles/car.lua /data/california-latest.osm.pbf

# Step 2: Partition (for MLD pipeline - supports turn restrictions + time-dependent)
docker run -t -v "${PWD}:/data" osrm/osrm-backend \
  osrm-partition /data/california-latest.osrm

# Step 3: Customize
docker run -t -v "${PWD}:/data" osrm/osrm-backend \
  osrm-customize /data/california-latest.osrm

# Step 4: Run server
docker run -t -i -p 5000:5000 -v "${PWD}:/data" osrm/osrm-backend \
  osrm-routed --algorithm mld /data/california-latest.osrm
```

**Docker Compose for production**

```yaml
version: '3.8'
services:
  osrm-extract:
    image: osrm/osrm-backend
    volumes:
      - ./osrm-data:/data
    command: osrm-extract -p /opt/osrm/profiles/car.lua /data/region.osm.pbf
    profiles: [init]

  osrm-partition:
    image: osrm/osrm-backend
    volumes:
      - ./osrm-data:/data
    command: osrm-partition /data/region.osrm
    profiles: [init]

  osrm-customize:
    image: osrm/osrm-backend
    volumes:
      - ./osrm-data:/data
    command: osrm-customize /data/region.osrm
    profiles: [init]

  osrm-router:
    image: osrm/osrm-backend
    volumes:
      - ./osrm-data:/data
    command: osrm-routed --algorithm mld --max-table-size 10000 /data/region.osrm
    ports:
      - "5000:5000"
    restart: always
```

**API calls**

```bash
# Basic route
curl "http://localhost:5000/route/v1/driving/-122.4194,37.7749;-118.2437,34.0522?overview=full&geometries=geojson&steps=true"

# Multiple waypoints
curl "http://localhost:5000/route/v1/driving/-122.4,37.77;-121.9,37.34;-118.24,34.05?overview=full"

# Multiple route alternatives
curl "http://localhost:5000/route/v1/driving/-122.4194,37.7749;-118.2437,34.0522?alternatives=3&overview=full&geometries=geojson"

# Nearest road snap
curl "http://localhost:5000/nearest/v1/driving/-122.4194,37.7749?number=3"

# Distance/time matrix (table service)
curl "http://localhost:5000/table/v1/driving/-122.4,37.77;-121.9,37.34;-118.24,34.05;-117.16,32.72"

# Map matching (snap GPS trace to road)
curl "http://localhost:5000/match/v1/driving/-122.41,37.775;-122.408,37.773;-122.405,37.771?geometries=geojson&overview=full"

# TSP / trip optimization
curl "http://localhost:5000/trip/v1/driving/-122.4,37.77;-121.9,37.34;-118.24,34.05?roundtrip=true&source=first&destination=last"
```

**Python client**

```python
import requests
import json

BASE = "http://localhost:5000"

def get_route(origin, destination, profile="driving"):
    """origin/destination: (lon, lat) tuples"""
    coords = f"{origin[0]},{origin[1]};{destination[0]},{destination[1]}"
    url = f"{BASE}/route/v1/{profile}/{coords}"
    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "true",
        "annotations": "true"
    }
    r = requests.get(url, params=params)
    data = r.json()
    route = data["routes"][0]
    return {
        "distance_km": route["distance"] / 1000,
        "duration_min": route["duration"] / 60,
        "geometry": route["geometry"]
    }

def get_matrix(locations, profile="driving"):
    """locations: list of (lon, lat) tuples"""
    coords = ";".join(f"{lon},{lat}" for lon, lat in locations)
    url = f"{BASE}/table/v1/{profile}/{coords}"
    r = requests.get(url)
    data = r.json()
    # durations[i][j] = seconds from location i to location j
    return data["durations"]
```

**Trick: Map matching for GPS trace cleaning**

```bash
# GPS trace with timestamps
curl "http://localhost:5000/match/v1/driving/-122.41,37.775;-122.408,37.773;-122.405,37.771?timestamps=1609459200;1609459210;1609459225&geometries=geojson&overview=full&annotations=true"
```

```python
def match_gps_trace(points_with_timestamps):
    """
    points_with_timestamps: list of (lon, lat, unix_timestamp)
    Returns: matched geometry snapped to road network + actual route distance
    """
    coords = ";".join(f"{lon},{lat}" for lon, lat, _ in points_with_timestamps)
    timestamps = ";".join(str(int(ts)) for _, _, ts in points_with_timestamps)

    url = f"{BASE}/match/v1/driving/{coords}"
    params = {
        "timestamps": timestamps,
        "geometries": "geojson",
        "overview": "full",
        "annotations": "true",
        "radiuses": ";".join(["25"] * len(points_with_timestamps))  # 25m snap radius
    }
    r = requests.get(url, params=params)
    data = r.json()

    matchings = data.get("matchings", [])
    total_distance = sum(m["distance"] for m in matchings)
    return {
        "matched_geometry": matchings[0]["geometry"] if matchings else None,
        "actual_distance_m": total_distance,
        "confidence": matchings[0]["confidence"] if matchings else 0
    }
```

**Trick: Pre-compute city-wide travel time matrices**

```python
import numpy as np
import pandas as pd
import h3

def compute_city_matrix(city_center_lat, city_center_lon, resolution=8):
    """Pre-compute travel times from all H3 hex centroids in a city."""
    # Get all H3 cells within ~5km of city center
    center_cell = h3.geo_to_h3(city_center_lat, city_center_lon, resolution)
    cells = list(h3.k_ring(center_cell, 20))  # 20 rings out

    # Get centroid of each cell
    centroids = [h3.h3_to_geo(c) for c in cells]  # (lat, lon)
    locations = [(lon, lat) for lat, lon in centroids]

    # OSRM table can handle 10K x 10K on a self-hosted instance
    # Batch into chunks of 1000 if needed
    matrix = get_matrix(locations[:500])  # subset for demo

    df = pd.DataFrame(matrix, index=cells[:500], columns=cells[:500])
    df.to_parquet("travel_time_matrix.parquet")
    return df
```

---

### Valhalla (Mapbox/Interline)

Valhalla is the most feature-rich open-source routing engine. Beyond basic routing it handles isochrones, elevation-aware routing, time-dependent routing, turn-by-turn navigation, and multimodal trips (walk + transit).

**Docker setup**

```bash
# Download PBF
wget https://download.geofabrik.de/north-america/us/california-latest.osm.pbf -P valhalla_data/

# Build tiles
docker run -dt --name valhalla \
  -p 8002:8002 \
  -v $PWD/valhalla_data:/custom_files \
  -e tile_urls=https://download.geofabrik.de/north-america/us/california-latest.osm.pbf \
  ghcr.io/valhalla/valhalla:latest

# Watch logs
docker logs -f valhalla
```

**Docker Compose**

```yaml
version: '3.8'
services:
  valhalla:
    image: ghcr.io/valhalla/valhalla:latest
    restart: always
    ports:
      - "8002:8002"
    volumes:
      - ./valhalla_tiles:/custom_files
    environment:
      - tile_urls=https://download.geofabrik.de/europe/germany-latest.osm.pbf
      - use_tiles_ignore_pbf=True  # use pre-built tiles on restart
      - serve_tiles=True
      - build_elevation=True       # download and incorporate elevation data
      - build_admins=True
      - build_time_zones=True
    mem_limit: 8g
```

**Routing API**

```bash
# Basic route
curl -X POST http://localhost:8002/route \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [
      {"lon": -122.4194, "lat": 37.7749},
      {"lon": -118.2437, "lat": 34.0522}
    ],
    "costing": "auto",
    "directions_options": {"language": "en-US"}
  }'

# Pedestrian with elevation preference
curl -X POST http://localhost:8002/route \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [
      {"lon": -122.4194, "lat": 37.7749},
      {"lon": -122.4094, "lat": 37.7849}
    ],
    "costing": "pedestrian",
    "costing_options": {
      "pedestrian": {
        "use_hills": 0.1,
        "max_grade": 15
      }
    }
  }'

# Truck routing with vehicle constraints
curl -X POST http://localhost:8002/route \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [
      {"lon": -74.006, "lat": 40.7128},
      {"lon": -73.935, "lat": 40.730}
    ],
    "costing": "truck",
    "costing_options": {
      "truck": {
        "height": 4.11,
        "width": 2.6,
        "length": 21.64,
        "weight": 21.77,
        "axle_load": 9.07,
        "hazmat": false
      }
    }
  }'
```

**Isochrone API**

```bash
# 15-minute and 30-minute drive-time isochrones
curl -X POST http://localhost:8002/isochrone \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [{"lon": -122.4194, "lat": 37.7749}],
    "costing": "auto",
    "contours": [
      {"time": 15, "color": "ff0000"},
      {"time": 30, "color": "00ff00"}
    ],
    "polygons": true,
    "show_locations": true
  }'

# Walk-time isochrone
curl -X POST http://localhost:8002/isochrone \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [{"lon": -122.4194, "lat": 37.7749}],
    "costing": "pedestrian",
    "contours": [
      {"time": 5},
      {"time": 10},
      {"time": 15}
    ],
    "polygons": true
  }'

# Distance-based isochrone (km instead of time)
curl -X POST http://localhost:8002/isochrone \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [{"lon": 2.3522, "lat": 48.8566}],
    "costing": "auto",
    "contours": [{"distance": 10}, {"distance": 25}],
    "polygons": true
  }'
```

**Matrix/sources_to_targets API**

```bash
curl -X POST http://localhost:8002/sources_to_targets \
  -H "Content-Type: application/json" \
  -d '{
    "sources": [
      {"lon": -122.4194, "lat": 37.7749},
      {"lon": -121.886, "lat": 37.338}
    ],
    "targets": [
      {"lon": -118.2437, "lat": 34.0522},
      {"lon": -117.161, "lat": 32.716},
      {"lon": -115.172, "lat": 36.114}
    ],
    "costing": "auto"
  }'
```

**Map matching (Meili engine)**

```bash
curl -X POST http://localhost:8002/trace_route \
  -H "Content-Type: application/json" \
  -d '{
    "shape": [
      {"lat": 37.7749, "lon": -122.4194, "time": 0},
      {"lat": 37.773, "lon": -122.416, "time": 12},
      {"lat": 37.770, "lon": -122.412, "time": 26}
    ],
    "costing": "auto",
    "shape_match": "map_snap",
    "trace_options": {
      "search_radius": 50,
      "gps_accuracy": 10
    }
  }'
```

**Python client**

```python
import requests
import json
from shapely.geometry import shape

VALHALLA = "http://localhost:8002"

def get_isochrone(lon, lat, times_minutes, costing="auto"):
    payload = {
        "locations": [{"lon": lon, "lat": lat}],
        "costing": costing,
        "contours": [{"time": t} for t in times_minutes],
        "polygons": True
    }
    r = requests.post(f"{VALHALLA}/isochrone", json=payload)
    r.raise_for_status()
    geojson = r.json()

    # Return as shapely geometries keyed by time
    result = {}
    for feature in geojson["features"]:
        t = feature["properties"]["contour"]
        result[t] = shape(feature["geometry"])
    return result

# Site selection: which of my stores has worst 15-min coverage?
stores = [
    {"name": "Store A", "lon": -122.4194, "lat": 37.7749},
    {"name": "Store B", "lon": -122.2712, "lat": 37.8044},
]

for store in stores:
    iso = get_isochrone(store["lon"], store["lat"], [15])
    catchment_area_km2 = iso[15].area * (111**2)  # rough conversion
    print(f"{store['name']}: {catchment_area_km2:.1f} km² 15-min catchment")
```

**Trick: Logistics routing with actual truck dimensions**

Most logistics companies use generic "truck" routing that ignores actual vehicle specs. Valhalla's truck costing model accepts real dimensions and will avoid low bridges, weight-restricted roads, and hazmat-banned tunnels.

```python
def route_truck(origin, destination, truck_spec):
    payload = {
        "locations": [
            {"lon": origin[0], "lat": origin[1]},
            {"lon": destination[0], "lat": destination[1]}
        ],
        "costing": "truck",
        "costing_options": {
            "truck": truck_spec  # height, width, length, weight, axle_load, hazmat
        },
        "directions_options": {"units": "kilometers"}
    }
    r = requests.post(f"{VALHALLA}/route", json=payload)
    return r.json()

# 18-wheeler specs
semi_truck = {
    "height": 4.1,      # meters
    "width": 2.59,
    "length": 22.0,
    "weight": 36.0,     # tonnes
    "axle_load": 9.07,
    "hazmat": False
}
```

---

### GraphHopper

Java-based routing engine with flexible speed-up techniques (Contraction Hierarchies, Landmark-based routing). The open-source version provides routing, the commercial version adds VRP optimization.

**Docker**

```bash
docker run -d -p 8989:8989 \
  -v ${PWD}/data:/data \
  israelhikingmap/graphhopper:latest \
  --url https://download.geofabrik.de/europe/switzerland-latest.osm.pbf \
  --host 0.0.0.0
```

**API**

```bash
# Route
curl "http://localhost:8989/route?point=47.3769,8.5417&point=46.9481,7.4474&profile=car&locale=en&calc_points=true"

# Isochrone
curl "http://localhost:8989/isochrone?point=47.3769,8.5417&profile=car&time_limit=900&buckets=3"

# Matrix
curl -X POST "http://localhost:8989/matrix" \
  -H "Content-Type: application/json" \
  -d '{
    "from_points": [[8.5417, 47.3769], [8.55, 47.38]],
    "to_points": [[7.4474, 46.9481]],
    "profile": "car",
    "out_arrays": ["times", "distances"]
  }'
```

**Trick: VRP for delivery route optimization**

GraphHopper's commercial Directions API includes a VRP solver. For open-source alternatives, combine GraphHopper routing with OR-Tools:

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

def solve_vrp(distance_matrix, num_vehicles, depot_index=0):
    """
    distance_matrix: NxN array of travel times (seconds)
    Returns optimized routes for each vehicle.
    """
    manager = pywrapcp.RoutingIndexManager(
        len(distance_matrix), num_vehicles, depot_index
    )
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distance_matrix[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add distance dimension
    routing.AddDimension(transit_callback_index, 0, 3600 * 8, True, "Time")

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        return None

    routes = []
    for vehicle_id in range(num_vehicles):
        index = routing.Start(vehicle_id)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        routes.append(route)

    return routes
```

---

### pgRouting

pgRouting extends PostGIS to add routing capabilities directly inside PostgreSQL. The killer advantage: you can join your routing results directly with spatial tables, filter by attributes, and update costs in real time.

**Setup**

```sql
-- Install extension
CREATE EXTENSION pgrouting;

-- Import OSM data with osm2pgsql or osm2po
-- osm2po generates a routable edge table directly

-- Using osm2po
-- java -jar osm2po-5.x.x.jar tileSize=x prefix=sf \
--   http://download.geofabrik.de/north-america/us/california-latest.osm.pbf

-- Load into PostGIS
psql -d routing_db -f sf_2po_4pgr.sql
```

**Basic routing queries**

```sql
-- Shortest path (Dijkstra)
SELECT seq, node, edge, cost, geom
FROM pgr_dijkstra(
    'SELECT id, source, target, cost, reverse_cost FROM ways',
    source_node_id,
    target_node_id,
    directed := true
) AS route
JOIN ways ON ways.id = route.edge;

-- Find nearest node to a coordinate
SELECT id FROM ways_vertices_pgr
ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(-122.4194, 37.7749), 4326)
LIMIT 1;

-- Full route with geometry
WITH route AS (
    SELECT edge
    FROM pgr_dijkstra(
        'SELECT id, source, target, cost FROM ways',
        (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(-122.4194, 37.7749), 4326) LIMIT 1),
        (SELECT id FROM ways_vertices_pgr ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(-122.2712, 37.8044), 4326) LIMIT 1),
        false
    )
    WHERE edge > 0
)
SELECT ST_Union(ways.the_geom) as route_geometry
FROM route
JOIN ways ON ways.id = route.edge;
```

**Trick: Drive-time isochrones in SQL**

```sql
-- Driving distance isochrone: all nodes reachable within 15 minutes from a point
WITH source AS (
    SELECT id FROM ways_vertices_pgr
    ORDER BY the_geom <-> ST_SetSRID(ST_MakePoint(-122.4194, 37.7749), 4326)
    LIMIT 1
),
reachable AS (
    SELECT *
    FROM pgr_drivingDistance(
        'SELECT id, source, target, cost FROM ways',
        (SELECT id FROM source),
        900,   -- 900 seconds = 15 minutes
        directed := false
    )
)
SELECT
    ST_ConcaveHull(ST_Collect(v.the_geom), 0.95) AS isochrone_geom,
    MAX(agg_cost) / 60 AS max_minutes
FROM reachable r
JOIN ways_vertices_pgr v ON v.id = r.node;
```

**Trick: Dynamic routing with real-time edge cost updates**

```sql
-- Update costs for road closures (e.g., from live traffic feeds)
UPDATE ways SET cost = cost * 9999 WHERE osm_id IN (
    SELECT osm_id FROM road_closures WHERE active = true
);

-- Or use a view with live costs so routing always uses current conditions
CREATE OR REPLACE VIEW ways_live AS
SELECT
    w.id, w.source, w.target,
    CASE
        WHEN c.id IS NOT NULL THEN w.cost * 9999  -- closed road
        WHEN t.congestion > 0.7 THEN w.cost * 2.5  -- heavy traffic
        ELSE w.cost
    END AS cost,
    w.reverse_cost
FROM ways w
LEFT JOIN road_closures c ON c.osm_id = w.osm_id AND c.active = true
LEFT JOIN traffic_speeds t ON t.way_id = w.id;

-- Now route on the live view
SELECT * FROM pgr_dijkstra(
    'SELECT id, source, target, cost FROM ways_live',
    source_id, target_id, directed := true
);
```

---

### openrouteservice (ORS)

HeiGIT's routing service built on GraphHopper, offering one of the best free public APIs for routing, isochrones, and matrices.

**Free API** (public): `https://api.openrouteservice.org/`
Register at `openrouteservice.org` for a free API key (2000 req/day, 500/min).

```bash
# Route
curl -X POST "https://api.openrouteservice.org/v2/directions/driving-car/json" \
  -H "Authorization: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "coordinates": [[-122.4194, 37.7749], [-118.2437, 34.0522]],
    "instructions": true,
    "elevation": true
  }'

# Isochrone
curl -X POST "https://api.openrouteservice.org/v2/isochrones/driving-car" \
  -H "Authorization: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [[-122.4194, 37.7749]],
    "range": [900, 1800],
    "range_type": "time"
  }'

# Wheelchair routing (unique profile)
curl -X POST "https://api.openrouteservice.org/v2/directions/wheelchair/json" \
  -H "Authorization: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "coordinates": [[-0.1276, 51.5074], [-0.1180, 51.5034]],
    "options": {
      "surface_type": "sett",
      "track_type": "grade1",
      "smoothness_type": "good",
      "maximum_sloped_kerb": 0.06,
      "maximum_incline": 6
    }
  }'
```

**QGIS Plugin (ORS Tools)**

Install via QGIS Plugin Manager. Provides GUI for routing, isochrones, and matrix — directly in the QGIS canvas. Ideal for analysts who don't code.

**Self-hosted Docker**

```bash
git clone https://github.com/GIScience/openrouteservice
cd openrouteservice
# Copy PBF to docker/data/
cp your-region.osm.pbf docker/data/

docker-compose up -d
# Wait for graph build (~5-15min for a country)

# Test
curl http://localhost:8080/ors/v2/health
```

---

### routingpy (Python Unified Routing Client)

Like geopy for geocoding, routingpy provides a single Python API across all major routing engines.

```bash
pip install routingpy
```

```python
import routingpy as rp

# OSRM
client = rp.OSRM(base_url="http://localhost:5000")

# Valhalla
client = rp.Valhalla(base_url="http://localhost:8002")

# GraphHopper
client = rp.Graphhopper(api_key="YOUR_KEY")

# OpenRouteService
client = rp.ORS(api_key="YOUR_KEY")

# Google
client = rp.Google(api_key="YOUR_KEY")

# HERE
client = rp.HereMaps(api_key="YOUR_KEY")

# Same API regardless of backend
coords = [(-122.4194, 37.7749), (-118.2437, 34.0522)]

route = client.directions(locations=coords, profile="car")
print(f"Distance: {route.distance/1000:.1f} km")
print(f"Duration: {route.duration/60:.1f} min")
print(f"Geometry points: {len(route.geometry)}")

# Matrix
matrix = client.matrix(locations=coords, profile="car")
print(matrix.durations)  # NxN array of seconds

# Isochrone (where supported)
iso = client.isochrones(
    locations=[(-122.4194, 37.7749)],
    profile="car",
    intervals=[900, 1800]
)
```

---

## Network Analysis

### OSMnx (Python)

OSMnx downloads OpenStreetMap street networks directly into Python as NetworkX graph objects, enabling rich network analysis without any database setup.

```bash
pip install osmnx
```

**Download and inspect networks**

```python
import osmnx as ox
import matplotlib.pyplot as plt

# Download by place name
G = ox.graph_from_place("Manhattan, New York, USA", network_type="drive")
print(ox.basic_stats(G))
# {'n': 4545, 'e': 12078, 'k_avg': 5.31, ...}

# Download by bounding box
G = ox.graph_from_bbox(37.79, 37.77, -122.41, -122.44, network_type="walk")

# Download by point + distance
G = ox.graph_from_point((48.8566, 2.3522), dist=2000, network_type="bike")

# Download any OSM features
cafes = ox.features_from_place("Vienna, Austria", tags={"amenity": "cafe"})
buildings = ox.features_from_place("Berlin Mitte", tags={"building": True})
parks = ox.features_from_place("London", tags={"leisure": "park"})
```

**Shortest path analysis**

```python
import osmnx as ox
import networkx as nx

G = ox.graph_from_place("Berkeley, California, USA", network_type="drive")

# Add travel time to edges (requires speed data)
G = ox.add_edge_speeds(G)   # imputes speed from OSM maxspeed or road type
G = ox.add_edge_travel_times(G)

# Find nearest nodes to coordinates
orig = ox.nearest_node(G, 37.8716, -122.2727)
dest = ox.nearest_node(G, 37.8549, -122.2596)

# Shortest path by travel time
route = nx.shortest_path(G, orig, dest, weight="travel_time")

# Plot
fig, ax = ox.plot_route_folium(G, route, route_color="#cc0000")
fig.save("route.html")

# Distance and time
route_length = sum(ox.utils_graph.get_route_edge_attributes(G, route, "length"))
route_time = sum(ox.utils_graph.get_route_edge_attributes(G, route, "travel_time"))
print(f"Distance: {route_length/1000:.2f} km, Time: {route_time/60:.1f} min")
```

**Centrality and network metrics**

```python
import osmnx as ox
import networkx as nx
import geopandas as gpd

G = ox.graph_from_place("Bologna, Italy", network_type="drive")
G_undirected = ox.get_undirected(G)

# Betweenness centrality (slow for large networks)
bc = nx.betweenness_centrality(G_undirected, weight="length", normalized=True)
nx.set_node_attributes(G, bc, "betweenness")

# Closeness centrality
cc = nx.closeness_centrality(G_undirected)
nx.set_node_attributes(G, cc, "closeness")

# Convert to GeoDataFrame and visualize
nodes, edges = ox.graph_to_gdfs(G)
nodes["betweenness"] = [G.nodes[n].get("betweenness", 0) for n in nodes.index]

ax = nodes.plot(
    column="betweenness",
    cmap="plasma",
    markersize=3,
    legend=True,
    figsize=(12, 12)
)
```

**Isochrone from OSMnx**

```python
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import MultiPoint
from shapely.ops import unary_union

def get_osmnx_isochrone(lat, lon, trip_times_min, speed_kmh=50):
    G = ox.graph_from_point((lat, lon), dist=5000, network_type="drive")
    G = ox.project_graph(G)

    center_node = ox.nearest_node(G, lat, lon)

    # Add travel time weight
    meters_per_minute = speed_kmh * 1000 / 60
    for u, v, k, data in G.edges(data=True, keys=True):
        data["time"] = data["length"] / meters_per_minute

    isochrones = []
    for trip_time in sorted(trip_times_min, reverse=True):
        subgraph = nx.ego_graph(G, center_node, radius=trip_time, distance="time")
        node_points = [
            (data["x"], data["y"])
            for node, data in subgraph.nodes(data=True)
        ]
        bounding_poly = MultiPoint(node_points).convex_hull.buffer(100)
        isochrones.append({"time_min": trip_time, "geometry": bounding_poly})

    return gpd.GeoDataFrame(isochrones, crs=G.graph["crs"])
```

**Trick: Fix OSM intersection topology**

OSM often represents single intersections as clusters of nearby nodes (due to turn lanes, complex junctions). Consolidate them before analysis:

```python
G = ox.graph_from_place("Amsterdam, Netherlands", network_type="drive")

# Before: many micro-nodes at complex intersections
print(f"Nodes before: {len(G.nodes)}")

G_consolidated = ox.consolidate_intersections(
    G,
    tolerance=15,       # meters - merge nodes within this distance
    rebuild_graph=True,
    dead_ends=False
)
print(f"Nodes after: {len(G_consolidated.nodes)}")
# Reduces node count by 20-40% in dense urban networks
```

**Trick: Custom travel speed by road type**

```python
G = ox.graph_from_place("Lyon, France", network_type="drive")

# Override default speed assumptions
hwy_speeds = {
    "motorway": 110,
    "trunk": 90,
    "primary": 60,
    "secondary": 50,
    "tertiary": 40,
    "residential": 30,
    "living_street": 10,
    "unclassified": 40
}
G = ox.add_edge_speeds(G, hwy_speeds=hwy_speeds)
G = ox.add_edge_travel_times(G)
```

---

### pandana (Fast Network Accessibility)

pandana is purpose-built for one thing: "how many amenities of type X are accessible from every node in a network within Y minutes?" It pre-processes the network into a C++ data structure and answers millions of such queries in seconds.

```bash
pip install pandana
```

```python
import pandana as pdna
import osmnx as ox
import geopandas as gpd
import pandas as pd

# Download street network as OSMnx graph, convert to pandana
G = ox.graph_from_place("Portland, Oregon, USA", network_type="walk")
nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=False)

# Build pandana network
network = pdna.Network(
    node_x=nodes["x"],
    node_y=nodes["y"],
    edge_from=edges.reset_index()["u"],
    edge_to=edges.reset_index()["v"],
    edge_weights=edges[["length"]],
    twoway=True
)

network.precompute(3000)  # precompute for 3000m max distance

# Download coffee shops from OSM
coffee = ox.features_from_place("Portland, Oregon, USA", tags={"amenity": "cafe"})
coffee = coffee[coffee.geometry.geom_type == "Point"]

# Set POIs on network
network.set_pois(
    category="coffee",
    maxdist=1500,       # meters
    maxitems=5,         # find up to 5 nearest
    x_col=coffee.geometry.x,
    y_col=coffee.geometry.y
)

# Query: distance to nearest coffee shop for every node
nearest = network.nearest_pois(distance=1500, category="coffee", num_pois=1)
# Returns DataFrame indexed by node_id with distances

# Accessibility score: count of cafes within 800m walk
network.set_pois("coffee", 800, 10, coffee.geometry.x, coffee.geometry.y)
accessibility = network.nearest_pois(800, "coffee", 10, include_poi_ids=False)
accessibility["within_800m_count"] = (accessibility < 800).sum(axis=1)

# Join back to nodes for heatmap
nodes["coffee_access"] = accessibility["within_800m_count"]
nodes_gdf = gpd.GeoDataFrame(nodes, geometry=gpd.points_from_xy(nodes.x, nodes.y), crs=4326)

ax = nodes_gdf.plot(
    column="coffee_access",
    cmap="YlOrRd",
    markersize=1,
    legend=True,
    figsize=(14, 14)
)
```

---

### momepy (Urban Morphology)

momepy provides street network and building pattern analysis built on GeoPandas and NetworkX.

```bash
pip install momepy
```

```python
import momepy
import osmnx as ox
import geopandas as gpd

# Street network metrics
G = ox.graph_from_place("Prague, Czech Republic", network_type="drive")
nodes, edges = ox.graph_to_gdfs(G)

# Node connectivity (degree)
G = momepy.node_degree(G)

# Meshedness (ratio of edges to maximum possible edges)
meshedness = momepy.meshedness(G)

# Mean node distance
mean_dist = momepy.mean_node_dist(G)

# Street alignment (orientation entropy)
alignment = momepy.street_alignment(edges, "bearing")

# Building analysis
buildings = ox.features_from_place("Prague 1, Czech Republic", tags={"building": True})
buildings = buildings[buildings.geometry.geom_type == "Polygon"].to_crs(5514)  # Czech national CRS

# Building orientation
buildings["orientation"] = momepy.orientation(buildings)

# Form factor (compactness)
buildings["form_factor"] = momepy.form_factor(buildings)

# Longest axis length
buildings["lal"] = momepy.longest_axis_length(buildings)
```

---

## Address Parsing & Standardization

```python
# libpostal - best for international data
from postal.parser import parse_address
from postal.expand import expand_address

parse_address("Rue de la Paix 15, 75002 Paris")
# [('rue de la paix', 'road'), ('15', 'house_number'), ('75002', 'postcode'), ('paris', 'city')]

# usaddress - US-specific, handles USPS formatting
pip install usaddress
import usaddress
usaddress.parse("123 Main St NE Suite 200, Springfield, IL 62701")
# [('123', 'AddressNumber'), ('Main', 'StreetName'), ('St', 'StreetNamePostType'),
#  ('NE', 'StreetNamePostDirectional'), ('Suite', 'OccupancyType'), ...]

# Tag the address type
tagged, address_type = usaddress.tag("123 Main St, Springfield IL")
# address_type: 'Street Address', 'PO Box', 'Intersection', 'Ambiguous'

# For dirty data: use usaddress to identify type, libpostal to parse
def parse_smart(address_string):
    try:
        _, addr_type = usaddress.tag(address_string)
    except usaddress.RepeatedLabelError:
        addr_type = "Ambiguous"

    if addr_type == "Ambiguous":
        return dict(parse_address(expand_address(address_string)[0]))
    else:
        tagged_us, _ = usaddress.tag(address_string)
        return dict(tagged_us)
```

---

## Distance & Travel Time Matrices

```python
import numpy as np
import h3
import requests

def osrm_matrix(locations, profile="driving", base_url="http://localhost:5000"):
    """
    locations: list of (lon, lat)
    Returns: dict with 'durations' (seconds) and 'distances' (meters)
    """
    coords = ";".join(f"{lon},{lat}" for lon, lat in locations)
    url = f"{base_url}/table/v1/{profile}/{coords}"
    params = {"annotations": "duration,distance"}
    r = requests.get(url, params=params)
    data = r.json()
    return {
        "durations": np.array(data["durations"]),
        "distances": np.array(data["distances"])
    }

def valhalla_matrix(sources, targets, costing="auto", base_url="http://localhost:8002"):
    """Asymmetric matrix: different sources and targets."""
    payload = {
        "sources": [{"lon": lon, "lat": lat} for lon, lat in sources],
        "targets": [{"lon": lon, "lat": lat} for lon, lat in targets],
        "costing": costing
    }
    r = requests.post(f"{base_url}/sources_to_targets", json=payload)
    data = r.json()
    # Shape: (len(sources), len(targets))
    durations = [[cell["time"] for cell in row] for row in data["sources_to_targets"]]
    distances = [[cell["distance"] * 1000 for cell in row] for row in data["sources_to_targets"]]  # km → m
    return {
        "durations": np.array(durations),
        "distances": np.array(distances)
    }
```

**Trick: Use H3 hex centroids as matrix origins for city-wide accessibility heatmaps**

```python
import h3
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon

def city_accessibility_heatmap(city_bbox, destinations, resolution=9):
    """
    city_bbox: (min_lon, min_lat, max_lon, max_lat)
    destinations: list of (lon, lat) - e.g., hospital locations
    resolution: H3 resolution (9 ≈ 0.1 km² cells)

    Returns GeoDataFrame of H3 cells colored by min travel time to any destination.
    """
    min_lon, min_lat, max_lon, max_lat = city_bbox

    # Fill bbox with H3 cells
    bbox_polygon = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat], [max_lon, min_lat],
            [max_lon, max_lat], [min_lon, max_lat],
            [min_lon, min_lat]
        ]]
    }
    cells = list(h3.polyfill(bbox_polygon, resolution))
    centroids = [(h3.h3_to_geo(c)[1], h3.h3_to_geo(c)[0]) for c in cells]  # (lon, lat)

    print(f"Computing matrix: {len(centroids)} origins × {len(destinations)} destinations")

    # Batch into chunks (OSRM /table can handle up to --max-table-size)
    chunk_size = 500
    min_times = []
    for i in range(0, len(centroids), chunk_size):
        chunk = centroids[i:i+chunk_size]
        all_locs = chunk + destinations
        n_orig = len(chunk)
        n_dest = len(destinations)

        coords = ";".join(f"{lon},{lat}" for lon, lat in all_locs)
        sources = ";".join(str(i) for i in range(n_orig))
        dests = ";".join(str(n_orig + i) for i in range(n_dest))
        url = f"http://localhost:5000/table/v1/driving/{coords}?sources={sources}&destinations={dests}"

        r = requests.get(url)
        matrix = np.array(r.json()["durations"])
        # Min time to any destination for each origin
        min_times.extend(np.min(matrix, axis=1).tolist())

    # Build GeoDataFrame
    def cell_to_polygon(cell):
        boundary = h3.h3_to_geo_boundary(cell, geo_json=True)
        return Polygon(boundary)

    gdf = gpd.GeoDataFrame({
        "h3_index": cells,
        "min_travel_time_min": [t/60 for t in min_times],
        "geometry": [cell_to_polygon(c) for c in cells]
    }, crs=4326)

    return gdf

# Usage
hospitals = [(-122.4194, 37.7749), (-122.2712, 37.8044)]
heatmap = city_accessibility_heatmap(
    (-122.52, 37.70, -122.35, 37.85),
    hospitals,
    resolution=9
)
heatmap.to_file("hospital_accessibility.geojson", driver="GeoJSON")
```

---

## Advanced Dark Arts

### Multi-modal Routing (Car + Transit + Walk)

Valhalla supports multi-modal routing when GTFS transit data is loaded alongside OSM.

```bash
# Add GTFS to Valhalla
docker run -it -v $PWD/valhalla_data:/custom_files \
  ghcr.io/valhalla/valhalla:latest \
  valhalla_convert_transit /custom_files/valhalla.json

curl -X POST http://localhost:8002/route \
  -H "Content-Type: application/json" \
  -d '{
    "locations": [
      {"lon": -122.4194, "lat": 37.7749, "date_time": {"type": 1, "value": "2025-03-15T08:00"}},
      {"lon": -122.2712, "lat": 37.8044}
    ],
    "costing": "multimodal",
    "costing_options": {
      "transit": {
        "use_bus": 1.0,
        "use_rail": 1.0,
        "use_transfers": 0.3
      }
    },
    "date_time": {"type": 1, "value": "2025-03-15T08:00"}
  }'
```

### Custom OSRM Lua Profiles

OSRM routing behavior is fully customizable via Lua scripts. The profile controls which roads are accessible, at what speed, and with what penalties.

```lua
-- emergency_vehicle.lua
-- Emergency vehicle profile: can go wrong way on one-ways, ignores access restrictions

api_version = 4

function setup()
  return {
    properties = {
      max_speed_for_map_matching = 180/3.6,
      weight_name = 'duration',
    }
  }
end

function process_way(profile, way, result, relations)
  local highway = way:get_value_by_key("highway")
  if highway then
    result.forward_mode = mode.driving
    result.backward_mode = mode.driving    -- allow both directions (ignore one-way)
    result.forward_speed = 80             -- assume 80 km/h regardless of limit
    result.backward_speed = 80
  end
end

function process_turn(profile, turn)
  turn.duration = 0  -- no turn penalties for emergency vehicles
end
```

```bash
# Compile with custom profile
docker run -t -v "${PWD}:/data" osrm/osrm-backend \
  osrm-extract -p /data/emergency_vehicle.lua /data/region.osm.pbf
```

### Batch Geocoding Pipeline

Production pipeline for geocoding a CSV with dirty addresses:

```python
import pandas as pd
import logging
from postal.expand import expand_address
from postal.parser import parse_address
import requests
import time
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GeocodingResult:
    input_address: str
    normalized_address: str
    latitude: Optional[float]
    longitude: Optional[float]
    confidence: float
    source: str
    needs_review: bool
    raw: dict

CONFIDENCE_THRESHOLD = 0.6
PELIAS_URL = "http://localhost:4000/v1"

def normalize_address(raw: str) -> str:
    expanded = expand_address(raw)
    return expanded[0] if expanded else raw.lower()

def pelias_geocode(address: str) -> Optional[dict]:
    try:
        r = requests.get(
            f"{PELIAS_URL}/search",
            params={"text": address, "size": 1},
            timeout=5
        )
        data = r.json()
        if data["features"]:
            return data["features"][0]
    except Exception as e:
        logger.warning(f"Pelias error for '{address}': {e}")
    return None

def geocode_row(raw_address: str) -> GeocodingResult:
    normalized = normalize_address(raw_address)
    feature = pelias_geocode(normalized)

    if not feature:
        # Try structured parse
        components = dict(parse_address(normalized))
        street = f"{components.get('house_number','')} {components.get('road','')}".strip()
        city = components.get("city", "")
        structured_query = f"{street}, {city}".strip(", ")
        feature = pelias_geocode(structured_query)

    if feature:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        confidence = props.get("confidence", 0)
        return GeocodingResult(
            input_address=raw_address,
            normalized_address=normalized,
            latitude=coords[1],
            longitude=coords[0],
            confidence=confidence,
            source=props.get("source", "unknown"),
            needs_review=confidence < CONFIDENCE_THRESHOLD,
            raw=props
        )
    else:
        return GeocodingResult(
            input_address=raw_address,
            normalized_address=normalized,
            latitude=None, longitude=None,
            confidence=0, source="failed",
            needs_review=True, raw={}
        )

def run_pipeline(input_csv: str, address_col: str, output_csv: str):
    df = pd.read_csv(input_csv)
    total = len(df)
    logger.info(f"Geocoding {total} addresses...")

    results = []
    for i, raw in enumerate(df[address_col]):
        result = geocode_row(str(raw))
        results.append(result)
        if (i+1) % 100 == 0:
            matched = sum(1 for r in results if r.latitude is not None)
            logger.info(f"Progress: {i+1}/{total} | Match rate: {matched/(i+1)*100:.1f}%")

    result_df = pd.DataFrame([vars(r) for r in results])
    result_df = pd.concat([df, result_df.drop(columns=["input_address"])], axis=1)
    result_df.to_csv(output_csv, index=False)

    # Summary
    matched = result_df["latitude"].notna().sum()
    needs_review = result_df["needs_review"].sum()
    logger.info(f"\nComplete: {matched}/{total} geocoded ({matched/total*100:.1f}%)")
    logger.info(f"Needs manual review: {needs_review} ({needs_review/total*100:.1f}%)")

run_pipeline("dirty_addresses.csv", "address", "geocoded_output.csv")
```

### Drive-time Trade Areas

```python
import requests
import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union
import pandas as pd

def compute_trade_areas(stores_gdf, drive_times_min=[10, 20, 30], costing="auto"):
    """
    For each store, compute drive-time catchment areas.
    Returns a GeoDataFrame with dissolved isochrones per time band.
    """
    all_isochrones = []

    for _, store in stores_gdf.iterrows():
        payload = {
            "locations": [{"lon": store.geometry.x, "lat": store.geometry.y}],
            "costing": costing,
            "contours": [{"time": t} for t in drive_times_min],
            "polygons": True
        }
        r = requests.post("http://localhost:8002/isochrone", json=payload)

        for feature in r.json()["features"]:
            all_isochrones.append({
                "store_id": store.get("store_id", "unknown"),
                "drive_time_min": feature["properties"]["contour"],
                "geometry": shape(feature["geometry"])
            })

    iso_gdf = gpd.GeoDataFrame(all_isochrones, crs=4326)

    # Dissolve by time band to get market-wide coverage
    market_coverage = iso_gdf.dissolve(by="drive_time_min", aggfunc="first").reset_index()

    # How much overlap exists between stores?
    for t in drive_times_min:
        band = iso_gdf[iso_gdf["drive_time_min"] == t]
        total_area = unary_union(band.geometry.values).area
        individual_area = band.geometry.area.sum()
        overlap_pct = (individual_area - total_area) / individual_area * 100
        print(f"{t}-min catchment overlap: {overlap_pct:.1f}%")

    return iso_gdf, market_coverage
```

### GPS Trajectory Cleaning with Map Matching

```python
import requests
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString
import pandas as pd

def clean_gps_traces(traces_df, lat_col="lat", lon_col="lon", time_col="timestamp",
                     trip_id_col="trip_id"):
    """
    Input: raw GPS observations (may have noise, gaps, off-road points)
    Output: road-snapped clean traces with actual route distances
    """
    cleaned = []

    for trip_id, group in traces_df.groupby(trip_id_col):
        group = group.sort_values(time_col)
        coords = list(zip(group[lon_col], group[lat_col]))
        timestamps = group[time_col].astype(int).tolist()

        # Valhalla map matching (handles sparse traces better than OSRM)
        payload = {
            "shape": [
                {"lat": lat, "lon": lon, "time": ts}
                for (lon, lat), ts in zip(coords, timestamps)
            ],
            "costing": "auto",
            "shape_match": "map_snap",
            "trace_options": {
                "search_radius": 50,
                "gps_accuracy": 15,
                "breakage_distance": 2000
            },
            "directions_options": {"units": "kilometers"}
        }

        try:
            r = requests.post("http://localhost:8002/trace_route", json=payload, timeout=30)
            data = r.json()

            if "trip" in data:
                trip = data["trip"]
                total_distance_km = trip["summary"]["length"]
                raw_distance_km = LineString(coords).length * 111  # very rough

                cleaned.append({
                    "trip_id": trip_id,
                    "raw_points": len(coords),
                    "snapped_distance_km": total_distance_km,
                    "raw_distance_km": raw_distance_km,
                    "detour_ratio": total_distance_km / raw_distance_km if raw_distance_km > 0 else None,
                    "status": "matched"
                })
        except Exception as e:
            cleaned.append({
                "trip_id": trip_id,
                "status": f"failed: {e}",
                "snapped_distance_km": None
            })

    return pd.DataFrame(cleaned)
```

### Mining Nominatim `extratags` for Hidden OSM Metadata

Nominatim can return extra OSM tags that aren't part of the standard address schema — building levels, opening hours, phone numbers, and more.

```bash
# Request extratags
curl "https://nominatim.openstreetmap.org/search?q=Empire+State+Building&format=jsonv2&extratags=1&addressdetails=1" \
  -H "User-Agent: MyGISApp/1.0"
```

```python
import requests

def enrich_with_osm_metadata(place_name):
    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        params={
            "q": place_name,
            "format": "jsonv2",
            "extratags": 1,
            "addressdetails": 1,
            "limit": 1
        },
        headers={"User-Agent": "MyGISApp/1.0"}
    )
    results = r.json()
    if not results:
        return None

    result = results[0]
    extratags = result.get("extratags", {})

    return {
        "name": result.get("display_name"),
        "lat": float(result["lat"]),
        "lon": float(result["lon"]),
        "osm_type": result["osm_type"],
        "osm_id": result["osm_id"],
        # Extra metadata
        "building_levels": extratags.get("building:levels"),
        "opening_hours": extratags.get("opening_hours"),
        "phone": extratags.get("phone") or extratags.get("contact:phone"),
        "website": extratags.get("website") or extratags.get("contact:website"),
        "wheelchair": extratags.get("wheelchair"),
        "cuisine": extratags.get("cuisine"),
        "brand": extratags.get("brand"),
        "wikidata": extratags.get("wikidata"),
    }

info = enrich_with_osm_metadata("Flatiron Building, New York")
print(info)
# {'building_levels': '22', 'wikidata': 'Q178437', ...}
```

### Nominatim Lookup by OSM ID

If you already have an OSM object ID (from a spatial query), you can resolve metadata without re-geocoding:

```bash
# Look up by OSM ID (W=way, N=node, R=relation)
curl "https://nominatim.openstreetmap.org/lookup?osm_ids=W238241022,R51477&format=jsonv2&extratags=1" \
  -H "User-Agent: MyGISApp/1.0"
```

---

## Quick Reference: API Cheatsheet

```bash
# OSRM
/route/v1/{profile}/{coords}?overview=full&geometries=geojson&steps=true
/table/v1/{profile}/{coords}?annotations=duration,distance
/match/v1/{profile}/{coords}?timestamps={ts}&geometries=geojson
/trip/v1/{profile}/{coords}?roundtrip=true
/nearest/v1/{profile}/{lon},{lat}?number=1

# Valhalla
POST /route           # routing
POST /isochrone       # drive-time areas
POST /sources_to_targets  # matrix
POST /trace_route     # map matching
POST /trace_attributes    # detailed trace attributes
POST /optimized_route     # TSP

# Nominatim
/search?q={query}&format=jsonv2&addressdetails=1&extratags=1
/reverse?lat={lat}&lon={lon}&format=jsonv2
/lookup?osm_ids={W123,N456}&format=jsonv2

# Pelias
/v1/search?text={query}
/v1/autocomplete?text={partial}
/v1/reverse?point.lat={lat}&point.lon={lon}
/v1/search/structured?address=...&locality=...&region=...

# ORS (public API)
POST /v2/directions/{profile}/json
POST /v2/isochrones/{profile}
POST /v2/matrix/{profile}
```

---

## Deployment Size Reference

| Tool | Min RAM | Recommended RAM | Storage | Build Time |
|---|---|---|---|---|
| Nominatim (country) | 8 GB | 16 GB | 50–150 GB | 2–6 hours |
| Nominatim (planet) | 64 GB | 128 GB | 700 GB+ | 48+ hours |
| Pelias (country) | 8 GB | 16 GB | 50 GB | 1–3 hours |
| Pelias (planet) | 32 GB | 64 GB | 500 GB | 12+ hours |
| OSRM (country) | 4 GB | 8 GB | 10–50 GB | 15–60 min |
| OSRM (planet) | 64 GB | 128 GB | 300 GB | 8–24 hours |
| Valhalla (country) | 4 GB | 8 GB | 5–30 GB | 30–90 min |
| Valhalla (planet) | 32 GB | 64 GB | 200 GB | 8–16 hours |
| pgRouting | Depends on PostGIS | — | Depends | Minutes |
