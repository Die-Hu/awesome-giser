# Workflow Templates

> Five end-to-end geospatial analysis workflow templates. Each includes a clear objective, required data, step-by-step process, recommended tools, expected outputs, and a text-based flowchart. Adapt these to your specific study area and data.

> **Quick Picks**
> - **Planning a new facility:** Site Suitability Analysis (AHP + weighted overlay)
> - **Monitoring land change:** Land Use Change Detection (post-classification comparison)
> - **Measuring service coverage:** Accessibility Analysis (isochrones + 2SFCA)
> - **Environmental review:** Environmental Impact Assessment (buffer + overlay)
> - **Urban climate study:** Urban Heat Island Analysis (LST + hot spots)

---

## Table of Contents

- [Site Suitability Analysis](#site-suitability-analysis)
- [Land Use Change Detection](#land-use-change-detection)
- [Accessibility Analysis](#accessibility-analysis)
- [Environmental Impact Assessment](#environmental-impact-assessment)
- [Urban Heat Island Analysis](#urban-heat-island-analysis)

---

## Site Suitability Analysis

### Objective

Identify the most suitable locations for a facility (e.g., solar farm, school, warehouse, wind turbine) based on multiple weighted criteria.

### Data Required

| Data | Purpose | Example Source |
|---|---|---|
| DEM (Digital Elevation Model) | Slope, aspect, elevation constraints | SRTM, ASTER GDEM, national LiDAR |
| Land use / land cover | Exclude unsuitable areas (water, forest, urban) | CORINE, NLCD, local planning data |
| Road network | Proximity to transport | OpenStreetMap, national road database |
| Utility infrastructure | Proximity to power lines, water supply | Local utility provider |
| Protected areas | Exclusion zones | WDPA, national environment agency |
| Soil type / geology | Foundation suitability, permeability | National soil survey |
| Population / demand data | Proximity to demand centers | Census, WorldPop |

**Data Portals:** [SRTM (USGS EarthExplorer)](https://earthexplorer.usgs.gov/), [CORINE (Copernicus)](https://land.copernicus.eu/pan-european/corine-land-cover), [NLCD (MRLC)](https://www.mrlc.gov/), [OpenStreetMap (Geofabrik)](https://download.geofabrik.de/), [WDPA (protectedplanet.net)](https://www.protectedplanet.net/), [WorldPop](https://www.worldpop.org/)

### Steps

1. **Define criteria** — List all factors, classify as constraints (binary: suitable/unsuitable) or factors (continuous: more/less suitable).
2. **Acquire and preprocess data** — Download, clip to study area, unify CRS, resample rasters to common resolution.
3. **Create constraint layers** — Binary masks (1 = suitable, 0 = excluded). Examples: slope < 15 degrees, not in protected area, not on water.
4. **Create factor layers** — Normalize each factor to a 0-1 scale (or 1-10). Examples: distance to road (closer = higher score), solar irradiance (higher = higher score).
5. **Assign weights** — Use AHP (Analytic Hierarchy Process), Delphi method, or expert judgment. Weights must sum to 1.0.
6. **Weighted overlay** — Multiply each factor by its weight, sum all factors, apply constraints as a final mask.
7. **Classify suitability** — Divide the result into classes (e.g., high / moderate / low / unsuitable) using natural breaks or equal intervals.
8. **Sensitivity analysis** — Vary the weights by +/- 10% and check if the top-ranked areas change.
9. **Validate** — Compare results with known suitable sites, field verification, or expert review.
10. **Report** — Map of suitability classes, area statistics, methodology description.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Data preprocessing | GeoPandas, Rasterio | sf, terra | QGIS, ArcGIS |
| Distance calculation | `scipy.ndimage.distance_transform_edt` | `terra::distance()` | Proximity (raster) |
| Normalization | NumPy, Rasterio | terra | Raster Calculator |
| AHP weights | `ahpy` package | `ahpsurvey` package | Manual / Excel |
| Weighted overlay | NumPy array math | terra | Weighted Overlay tool |
| Classification | NumPy `digitize` | `terra::classify()` | Reclassify |

### Expected Output

- Suitability raster (classified GeoTIFF)
- Suitability map (publication-ready)
- Statistics table: area per class, top-N candidate sites
- Sensitivity analysis summary
- Methodology report

### Flowchart

```
[Define Criteria]
       |
       v
[Acquire Data] --> [Clip & Reproject] --> [Unify Resolution]
       |
       +----------+----------+
       |          |          |
       v          v          v
  [Constraint  [Factor    [Factor
   Layers]     Layer 1]   Layer 2]  ...
   (binary)    (0-1)      (0-1)
       |          |          |
       |          +----+-----+
       |               |
       |               v
       |        [Assign Weights (AHP)]
       |               |
       |               v
       |        [Weighted Sum]
       |               |
       +--------->+<---+
                  |
                  v
         [Apply Constraint Mask]
                  |
                  v
         [Classify Suitability]
                  |
                  v
      [Sensitivity Analysis]
                  |
                  v
         [Validate & Report]
```

---

## Land Use Change Detection

### Objective

Detect and quantify changes in land use / land cover between two or more time periods using satellite imagery.

### Data Required

| Data | Purpose | Example Source |
|---|---|---|
| Satellite imagery (T1) | Baseline land cover | Sentinel-2, Landsat, NAIP |
| Satellite imagery (T2) | Current land cover | Same sensor as T1 (for consistency) |
| Training samples (T1 & T2) | Supervised classification | Field data, image interpretation, existing maps |
| Administrative boundaries | Reporting units | National boundary dataset |
| Reference change data | Validation | High-resolution imagery, field surveys |

### Steps

1. **Image selection** — Choose images from the same season (to minimize phenological differences), same sensor, low cloud cover.
2. **Preprocessing** — Atmospheric correction (to surface reflectance), cloud masking, co-registration.
3. **Classification (T1)** — Supervised classification of the baseline image using training samples.
4. **Classification (T2)** — Supervised classification of the current image using training samples.
5. **Accuracy assessment** — Validate each classification independently (confusion matrix, kappa).
6. **Post-classification comparison** — Create a change matrix by cross-tabulating T1 and T2 classifications.
7. **Change map** — Visualize "from-to" transitions (e.g., Forest -> Urban, Cropland -> Forest).
8. **Area calculation** — Compute area per class and per transition, apply bias-adjusted area estimation if needed.
9. **Statistical summary** — Change matrix table, area per transition, annual rate of change.
10. **Report** — Maps (T1 classified, T2 classified, change), tables, methodology.

### Tools

| Step | Python | R | Desktop GIS / Cloud |
|---|---|---|---|
| Image download | `sentinelsat`, `landsatxplore`, `pystac` | `sen2r`, `getSpatialData` | Copernicus Hub, EarthExplorer |
| Preprocessing | Rasterio, `sen2cor` | `terra`, `sen2r` | SNAP, QGIS SCP |
| Classification | scikit-learn, XGBoost | `caret`, `randomForest` | QGIS SCP, ArcGIS |
| Change detection | NumPy, Rasterio | terra | GEE, QGIS |
| Area estimation | `mapbiomas` methods, NumPy | `terra::freq()` | Manual |

### Expected Output

- Classified maps for T1 and T2
- Change detection map (from-to classes)
- Change matrix (cross-tabulation table)
- Area statistics per transition (in hectares/km2)
- Accuracy assessment reports for each date

### Flowchart

```
[Image T1]              [Image T2]
    |                       |
    v                       v
[Preprocess]           [Preprocess]
    |                       |
    v                       v
[Classify]             [Classify]
    |                       |
    v                       v
[Validate]             [Validate]
    |                       |
    +-----------+-----------+
                |
                v
    [Post-Classification Comparison]
                |
                v
         [Change Matrix]
                |
                v
    [Area Estimation & Statistics]
                |
                v
      [Change Map & Report]
```

---

## Accessibility Analysis

### Objective

Measure how accessible a service or facility is to the surrounding population, using travel time, distance, or cost as the metric.

### Data Required

| Data | Purpose | Example Source |
|---|---|---|
| Road network | Travel routes and speeds | OpenStreetMap (via OSMnx), HERE, TomTom |
| Facility locations | Destinations (hospitals, schools, etc.) | Government open data, POI databases |
| Population data | Demand / catchment | WorldPop, census, GHSL |
| Administrative boundaries | Reporting units | National boundary dataset |
| Public transit routes (optional) | Multi-modal analysis | GTFS feeds, local transit authority |
| Elevation data (optional) | Walking/cycling impedance | SRTM, national LiDAR |

### Steps

1. **Define the service and metric** — What facility? What travel mode? What metric (time, distance, cost)?
2. **Build the network** — Download or prepare the road/transit network as a graph.
3. **Assign travel speeds** — Based on road type, surface, speed limits, or time-of-day (for transit).
4. **Calculate isochrones** — From each facility, compute travel-time zones (e.g., 5, 10, 15, 30 minutes).
5. **Calculate OD matrix** — Origin-destination travel times from population points/centroids to nearest facility.
6. **Compute accessibility indicators:**
   - Travel time to nearest facility
   - Number of facilities within a threshold time
   - Gravity-based accessibility (weighted by facility capacity and distance decay)
   - Two-step floating catchment area (2SFCA)
7. **Map results** — Choropleth of accessibility by area, isochrone overlay.
8. **Identify gaps** — Areas with low accessibility (underserved populations).
9. **Scenario analysis** — Test impact of adding a new facility at candidate locations.
10. **Report** — Accessibility maps, statistics per administrative unit, gap analysis.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Network download | `osmnx` | `osmdata` | OSM download plugin |
| Network analysis | `osmnx`, `networkx`, `pandana` | `sfnetworks`, `dodgr` | QGIS Network Analysis, pgRouting |
| Isochrones | `osmnx.isochrone`, Valhalla API | `osrm`, `opentripplanner` | ORS Tools plugin |
| OD matrix | `pandana`, `r5py` | `r5r`, `accessibility` | OD Cost Matrix (ArcGIS) |
| 2SFCA | `access` package, custom | `accessibility` package | Custom script |
| Population overlay | GeoPandas, `rasterstats` | sf, terra | Zonal Statistics |

### Expected Output

- Travel time raster or polygon layer
- Isochrone maps per facility
- Accessibility score per administrative unit
- Underserved area identification
- Scenario comparison (current vs proposed)

### Flowchart

```
[Facility Locations]     [Road Network]     [Population Data]
        |                     |                    |
        v                     v                    |
  [Geocode/Clean]      [Build Graph +              |
                        Assign Speeds]             |
        |                     |                    |
        +----------+----------+                    |
                   |                               |
                   v                               |
          [Isochrones / OD Matrix]                 |
                   |                               |
                   +-------------------+-----------+
                                       |
                                       v
                           [Accessibility Indicators]
                                       |
                                       v
                              [Gap Identification]
                                       |
                                       v
                          [Scenario Analysis (optional)]
                                       |
                                       v
                              [Map & Report]
```

---

## Environmental Impact Assessment

### Objective

Assess the potential environmental impact of a proposed development (e.g., road, dam, mine, wind farm) on the surrounding landscape, habitats, and resources.

### Data Required

| Data | Purpose | Example Source |
|---|---|---|
| Project footprint | Area directly affected | Project design documents (CAD/GIS) |
| Buffer zones | Area indirectly affected | Regulatory guidelines (e.g., 500m, 1km, 5km) |
| Land cover / habitat | Habitat loss assessment | Sentinel-2 classification, local habitat map |
| Protected areas / sensitive sites | Regulatory compliance | WDPA, Ramsar, national parks data |
| Species occurrence data | Biodiversity impact | GBIF, eBird, national biodiversity databases |
| Hydrology (rivers, watersheds) | Water impact | HydroSHEDS, national hydrography |
| Air quality / noise baseline | Pollution impact | Monitoring stations, models |
| Population / settlements | Social impact | Census, OpenStreetMap |
| Soil / geology | Erosion, contamination risk | National soil survey |
| DEM | Visibility, runoff, slope stability | SRTM, LiDAR |

### Steps

1. **Define the project footprint** — Import or digitize the proposed development boundary.
2. **Create buffer zones** — Generate buffers at regulatory distances (e.g., 500m, 1km, 5km).
3. **Baseline mapping** — Map current land cover, habitats, water bodies, settlements within the impact zone.
4. **Overlay analysis** — Intersect the project footprint and buffers with each environmental layer.
5. **Habitat impact** — Calculate area of each habitat type affected (direct loss + indirect disturbance).
6. **Species impact** — Query species occurrence within the impact zone, flag protected/endangered species.
7. **Hydrological impact** — Identify downstream watersheds, water bodies, and wetlands affected.
8. **Visual impact** — Viewshed analysis from the project site using the DEM.
9. **Cumulative impact** — Overlay with existing developments in the area.
10. **Mitigation mapping** — Identify areas for habitat restoration, buffer planting, wildlife corridors.
11. **Report** — Impact matrices, maps, quantitative summaries, mitigation recommendations.

### Tools

| Step | Python | R | Desktop GIS |
|---|---|---|---|
| Buffer / overlay | GeoPandas | sf | QGIS, ArcGIS |
| Habitat mapping | scikit-learn + Rasterio | terra + caret | QGIS SCP |
| Species query | `pygbif`, custom API | `rgbif` | GBIF web interface |
| Viewshed | `WhiteboxTools`, GDAL | `terra::viewshed()` | QGIS Sketcher, ArcGIS |
| Watershed delineation | `pysheds`, WhiteboxTools | `whitebox` (R), `terra` | QGIS, ArcGIS Hydrology |
| Reporting | Jinja2 + matplotlib | R Markdown | Manual / ArcGIS Layouts |

### Expected Output

- Baseline environment maps
- Impact zone maps (footprint + buffers)
- Habitat impact table (area per type affected)
- Species impact list (species within impact zone, conservation status)
- Viewshed map
- Hydrological impact map (affected watersheds)
- Mitigation plan map
- EIA report with maps, tables, and recommendations

### Flowchart

```
[Project Footprint]
        |
        v
[Create Buffer Zones: 500m, 1km, 5km]
        |
        +---------+---------+---------+---------+
        |         |         |         |         |
        v         v         v         v         v
   [Habitat]  [Species] [Hydrology] [Visual]  [Social]
   [Impact]   [Impact]  [Impact]    [Impact]  [Impact]
        |         |         |         |         |
        +---------+---------+---------+---------+
                            |
                            v
                  [Cumulative Assessment]
                            |
                            v
                   [Mitigation Planning]
                            |
                            v
                     [EIA Report]
```

---

## Urban Heat Island Analysis

### Objective

Map and analyze the urban heat island (UHI) effect by comparing surface temperatures across urban and rural areas, and identify the land cover and urban form factors that contribute to elevated temperatures.

### Data Required

| Data | Purpose | Example Source |
|---|---|---|
| Thermal imagery | Land surface temperature (LST) | Landsat 8/9 Band 10, MODIS LST, ECOSTRESS |
| Multispectral imagery | NDVI, NDBI, albedo | Sentinel-2, Landsat |
| Land cover map | Correlate LST with land cover | CORINE, NLCD, local classification |
| Building footprints | Urban density, impervious fraction | OpenStreetMap, Microsoft Building Footprints |
| Street trees / green space | Cooling effect analysis | Local urban forestry data, OpenStreetMap |
| DEM | Elevation correction | SRTM, LiDAR |
| Weather station data | Validation, air temperature reference | NOAA, national meteorological service |
| Census / population | Vulnerability mapping | Census, WorldPop |

### Steps

1. **Acquire thermal imagery** — Select cloud-free summer daytime (and optionally nighttime) scenes.
2. **Calculate LST** — Convert thermal band digital numbers to at-sensor radiance, then to brightness temperature, then to LST using emissivity correction (NDVI-based or land cover-based).
3. **Calculate urban indices** — NDVI, NDBI, impervious surface fraction, albedo.
4. **Define urban-rural zones** — Delineate urban core, suburban, and rural reference areas.
5. **UHI intensity** — Calculate the LST difference between urban and rural zones (by zone, by pixel).
6. **Correlation analysis** — Correlate LST with NDVI, NDBI, impervious fraction, building density.
7. **Hot spot analysis** — Apply Getis-Ord Gi* to identify statistically significant thermal hot spots.
8. **Vulnerability mapping** — Overlay thermal hot spots with vulnerable populations (elderly, low-income).
9. **Mitigation scenarios** — Model the cooling effect of adding green space or increasing albedo.
10. **Report** — LST maps, UHI intensity maps, correlation plots, hot spot maps, mitigation recommendations.

### Tools

| Step | Python | R | Desktop GIS / Cloud |
|---|---|---|---|
| LST calculation | Rasterio, NumPy | terra | QGIS, GEE |
| Index calculation | Rasterio, NumPy | terra | QGIS Raster Calculator |
| Hot spot analysis | PySAL (esda) | spdep | ArcGIS Hot Spot, GeoDa |
| Correlation | scipy, statsmodels | stats, cor.test | Excel, R |
| Visualization | matplotlib, Folium | tmap, ggplot2 | QGIS, ArcGIS |
| Vulnerability overlay | GeoPandas | sf | QGIS |

### Expected Output

- Land surface temperature map
- UHI intensity map (urban-rural LST difference)
- Correlation scatter plots (LST vs NDVI, LST vs NDBI)
- Thermal hot spot map (Gi* results)
- Vulnerability map (hot spots + population)
- Mitigation scenario comparison
- Analysis report with maps, statistics, and recommendations

### Flowchart

```
[Thermal Imagery]        [Multispectral Imagery]      [Land Cover / Buildings]
       |                         |                             |
       v                         v                             |
  [Calculate LST]         [Calculate NDVI,                     |
                           NDBI, Albedo]                       |
       |                         |                             |
       +------------+------------+-----------------------------+
                    |
                    v
          [Urban-Rural Zone Definition]
                    |
                    v
          [UHI Intensity Calculation]
                    |
          +---------+---------+
          |         |         |
          v         v         v
   [Correlation] [Hot Spot] [Vulnerability
    Analysis]    Analysis]   Mapping]
          |         |         |
          +---------+---------+
                    |
                    v
          [Mitigation Scenarios]
                    |
                    v
              [Report]
```

---

[Back to Data Analysis](README.md) · [Back to Main README](../README.md)
