# Spatial Data Analysis Prompts

> Expert-level prompt templates for spatial data cleaning, operations, statistics, transformation, and cloud-native workflows. Copy the template, fill in the `[variables]`, and paste into your AI tool of choice.

> **Quick Picks**
> - **Most Useful**: [Geometry Repair Pipeline](#prompt-1--geometry-repair-pipeline) -- every GIS analyst hits invalid geometries weekly
> - **Time Saver**: [DuckDB Spatial Analytics](#prompt-18--duckdb-spatial-analytics) -- query billions of rows on GeoParquet without loading into memory
> - **Cutting Edge**: [COG Pipeline](#prompt-19--cog-pipeline) -- cloud-optimized raster workflows with STAC integration

---

## Table of Contents

- [Section 1: Data Cleaning & Validation](#section-1-data-cleaning--validation) (P1-P4)
- [Section 2: Spatial Operations](#section-2-spatial-operations) (P5-P9)
- [Section 3: Statistical Analysis](#section-3-statistical-analysis) (P10-P13)
- [Section 4: Data Transformation](#section-4-data-transformation) (P14-P17)
- [Section 5: Cloud-Native Workflows](#section-5-cloud-native-workflows) (P18-P21)
- [Section 6: Client Deliverable Data Prep](#section-6-client-deliverable-data-prep) (P22-P23)

---

## Section 1: Data Cleaning & Validation

---

### Prompt 1 -- Geometry Repair Pipeline

#### Scenario (实际场景)

You received a dataset from a third-party surveying company containing thousands of polygons with self-intersections, ring ordering errors, and null geometries. Every downstream spatial join and overlay fails because of these invalid geometries. You need a systematic repair pipeline that logs every action taken.

#### Roles

User: Spatial Data Engineer | AI: PostGIS & Shapely Geometry Specialist

#### Prompt Template

```
You are a Senior Spatial Data Engineer specializing in geometry validation and repair.

I have a [format: Shapefile / GeoPackage / GeoJSON] with [N] features representing
[feature_type, e.g., parcel boundaries] in CRS [EPSG code].

Known issues:
- [N1] features fail ST_IsValid / shapely.is_valid
- Specific error types from ST_IsValidReason / explain_validity: [list errors]
- [N2] null geometries, [N3] empty geometries

Write a Python script (GeoPandas 0.14+, Shapely 2.0+, optionally PostGIS via SQLAlchemy) that:
1. Loads the dataset and runs explain_validity() on every geometry
2. Categorizes invalidity reasons into classes (self-intersection, ring-order, duplicate-vertex, etc.)
3. Applies a tiered repair strategy:
   a. First pass: shapely.make_valid() -- handles most cases
   b. Second pass: buffer(0) for remaining ring-order and collapse issues
   c. Third pass: ST_SnapToGrid(geom, [tolerance]) for near-duplicate vertex jitter
4. Preserves original geometry type -- do NOT let Polygon become GeometryCollection
5. Logs a per-feature repair journal: feature_id, original_validity, repair_method, final_validity
6. Exports:
   - Cleaned dataset as GeoPackage (layer: "repaired")
   - Unrepairable features as GeoPackage (layer: "failed")
   - Repair journal as CSV
7. Prints summary: total, repaired, failed, by-error-class breakdown

Performance: use vectorized Shapely 2.0 operations (no row-by-row apply when possible).
```

#### Variables to Customize

- `[format]`: Input file format (Shapefile, GeoPackage, GeoJSON, PostGIS table)
- `[N]`: Feature count
- `[feature_type]`: Domain description (parcels, buildings, wetlands, etc.)
- `[EPSG code]`: Coordinate reference system
- `[tolerance]`: Snap-to-grid tolerance in CRS units

#### Expected Output

A standalone Python script with CLI arguments, structured logging, and three output files (repaired GeoPackage, failed GeoPackage, CSV journal). The script should handle datasets up to 1M features without excessive memory usage.

#### Validation Checklist

- [ ] All output geometries pass `shapely.is_valid`
- [ ] Original geometry type is preserved (no GeometryCollection leakage)
- [ ] Feature count: repaired + failed = original
- [ ] Repair journal CSV has one row per feature with repair method documented
- [ ] CRS metadata is preserved in output GeoPackage

#### Cost Optimization

Use Shapely 2.0 vectorized `make_valid()` on the entire GeoSeries first -- this resolves 90%+ of issues in a single call, avoiding expensive row-by-row iteration.

#### Dark Arts Tip

If `make_valid()` produces a GeometryCollection, extract the largest polygon with `max(geom.geoms, key=lambda g: g.area)` -- this silently discards sliver artifacts that would otherwise break downstream dissolves.

#### Related Pages

- [PostGIS Reference](../tools/postgis-recipes.md) -- ST_MakeValid, ST_IsValidReason
- [Python Geo Stack](../tools/python-geo-stack.md) -- Shapely 2.0 vectorized ops
- [Data Sources Catalog](../data-sources/) -- common sources and their known quality issues

#### Extensibility

Add a `--strict` flag that rejects any geometry requiring more than `buffer(0)`, useful for survey-grade datasets where geometry modification must be minimized.

---

### Prompt 2 -- Schema Standardization

#### Scenario (实际场景)

Your team received datasets from five different municipal bureaus, each with different column names (some in Chinese, e.g., "地块编号", "面积_平方米"), inconsistent data types, mixed encodings, and varying CRS. You need to harmonize them into a single unified schema before loading into the project database.

#### Roles

User: GIS Data Manager | AI: ETL & Schema Normalization Specialist

#### Prompt Template

```
You are a Senior Data Engineer specializing in geospatial ETL and schema harmonization.

I have [N] datasets from different sources that must be merged into a unified schema.

Source datasets:
[For each dataset, list: filename, format, CRS, encoding, column names with types]

Target unified schema:
| Column Name | Type     | Constraints        | Source Mapping                    |
|-------------|----------|--------------------|-----------------------------------|
| [col1]      | [type]   | [NOT NULL, UNIQUE] | dataset_A.[原始列名], dataset_B.X |
| ...         | ...      | ...                | ...                               |

Target CRS: [EPSG code]
Target encoding: UTF-8

Write a Python script (GeoPandas 0.14+, pyogrio engine) that:
1. Reads each dataset with correct encoding (handle GBK, Shift_JIS, Latin-1)
2. Renames columns per the mapping table above
3. Casts data types (string dates -> datetime64, string numbers -> float64)
4. Standardizes string values (strip whitespace, normalize case per column rules)
5. Reprojects all datasets to the target CRS
6. Validates each dataset against the target schema (missing columns, type mismatches)
7. Concatenates into a single GeoDataFrame with a "source_dataset" provenance column
8. Exports to GeoPackage with schema enforcement

Handle Chinese field names (e.g., "地块编号" -> "parcel_id") explicitly in the mapping.
```

#### Variables to Customize

- `[N]`: Number of source datasets
- Source dataset details (format, CRS, encoding, columns)
- Target schema mapping table
- `[EPSG code]`: Target CRS
- String normalization rules per column (lowercase, title case, etc.)

#### Expected Output

A Python ETL script that reads N heterogeneous datasets and produces a single schema-compliant GeoPackage with a provenance column. The script should log all mapping decisions and type coercions.

#### Validation Checklist

- [ ] All output columns match the target schema exactly
- [ ] No encoding errors in Chinese/CJK text fields
- [ ] CRS is uniform across all features in the output
- [ ] Row count of output equals sum of input row counts (minus intentionally dropped rows)
- [ ] Provenance column correctly identifies the source dataset for each feature

#### Cost Optimization

Provide the full column mapping table upfront in the prompt -- this eliminates back-and-forth clarification rounds and lets the AI generate the complete ETL script in a single response.

#### Dark Arts Tip

When column names contain non-ASCII characters, use a `dict` mapping rather than regex -- `rename(columns={"地块编号": "parcel_id"})` is safer and more readable than any pattern-matching approach.

#### Related Pages

- [Data Sources Catalog](../data-sources/) -- encoding and schema notes for common Chinese government datasets
- [Python Geo Stack](../tools/python-geo-stack.md) -- pyogrio engine for fast I/O
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- client-side schema validation

#### Extensibility

Add a YAML-based schema definition file so the same script can be reused across projects by swapping only the config, not the code.

---

### Prompt 3 -- Data Quality Audit

#### Scenario (实际场景)

Before loading a new dataset into your enterprise geodatabase, you need a comprehensive quality audit covering null values, duplicates, outliers, topology errors, and CRS consistency. The audit report will be reviewed by the project manager and attached to the data acceptance document (数据验收报告).

#### Roles

User: QA/QC Lead | AI: Geospatial Data Quality Analyst

#### Prompt Template

```
You are a Senior Geospatial QA Engineer building an automated data quality audit pipeline.

Dataset: [format] file at [path], representing [description].
Expected CRS: [EPSG code]. Expected geometry type: [Polygon/Point/Line].
Expected feature count: approximately [N].

Build a Python script (GeoPandas 0.14+, Shapely 2.0+, Pandas, Jinja2) that runs these checks:

COMPLETENESS:
- Null values per column (count + percentage)
- Missing geometry count
- Required columns present: [list]

UNIQUENESS:
- Duplicate geometries (exact match + near-match within [tolerance])
- Duplicate values in key column [key_col]

VALIDITY:
- Geometry validity (is_valid check with reason for each failure)
- Geometry type consistency (flag mixed types)
- CRS matches expected [EPSG code]
- Attribute value ranges: [col_name] in [[min], [max]]
- Categorical values: [col_name] in [[allowed_values]]

TOPOLOGY:
- Self-intersecting polygons
- Overlapping polygons (pairwise, sample first [N] pairs)
- Sliver polygons (area < [threshold] m2)

OUTPUT:
- HTML report (Jinja2 template) with summary table, charts (matplotlib), and map thumbnails
- GeoPackage with a "qa_flags" column (comma-separated list of failed checks per feature)
- JSON summary for CI/CD integration (exit code 0=pass, 1=fail)

Make all thresholds configurable via a YAML file.
```

#### Variables to Customize

- `[format]`, `[path]`: Input file details
- `[description]`: Dataset domain description
- `[EPSG code]`, geometry type, feature count
- `[key_col]`: Primary key column
- `[tolerance]`: Near-duplicate distance threshold
- Column-specific validation rules (ranges, allowed values)
- `[threshold]`: Sliver polygon area threshold

#### Expected Output

Three output files: an HTML report suitable for stakeholder review, a GeoPackage with per-feature QA flags for spatial inspection, and a JSON summary for automated pipeline integration.

#### Validation Checklist

- [ ] HTML report renders correctly in a browser with all charts and maps
- [ ] Every feature in the GeoPackage has a valid "qa_flags" entry (empty string if no issues)
- [ ] JSON summary exit code accurately reflects pass/fail status
- [ ] All threshold values are read from YAML config, not hardcoded

#### Cost Optimization

Run completeness and uniqueness checks first -- if the dataset has fundamental issues (wrong schema, wrong CRS), skip expensive topology checks and fail fast.

#### Dark Arts Tip

For the overlap check on large polygon datasets, build an STRtree spatial index first and only test pairs whose bounding boxes intersect -- this reduces pairwise comparisons from O(n^2) to near-O(n).

#### Related Pages

- [Data Sources Catalog](../data-sources/) -- expected schemas for common datasets
- [PostGIS Reference](../tools/postgis-recipes.md) -- ST_IsValidReason, topology checks
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- client-side preview of QA results

#### Extensibility

Add a `--profile` mode that generates a comprehensive data profile (value distributions, spatial extent heatmap) alongside the QA checks, similar to `ydata-profiling` but spatially aware.

---

### Prompt 4 -- Coordinate System Detective

#### Scenario (实际场景)

You received a dataset with no CRS metadata (.prj file missing, no PROJJSON). The coordinates look like they could be WGS-84, GCJ-02 (China's offset system), or BD-09 (Baidu's further offset). You need to identify the actual CRS and convert to standard WGS-84 if necessary.

#### Roles

User: Spatial Data Engineer | AI: CRS & Geodesy Specialist

#### Prompt Template

```
You are a Senior Geodesist specializing in coordinate reference system identification,
with deep expertise in Chinese coordinate offset systems (GCJ-02, BD-09).

I have a dataset with [N] features. The CRS metadata is missing or unreliable.
Sample coordinates (first 5 features):
[Paste lon/lat or x/y pairs here]

The data is supposed to represent [feature_type] in [geographic_area].

Write a Python script that:
1. Analyzes coordinate ranges to determine likely CRS family:
   - Geographic (decimal degrees) vs. projected (meters/feet)
   - If projected, estimate UTM zone or regional system from coordinate magnitudes
2. For geographic coordinates in China, detect GCJ-02 / BD-09 offset:
   a. Pick [N_sample] well-distributed points
   b. Reverse-geocode against a known reference (OSM Nominatim or local reference)
   c. Compare against GCJ-02->WGS-84 corrected positions (use eviltransform or coordtransform)
   d. If offset ~= 0.003-0.007 degrees, classify as GCJ-02
   e. If offset pattern matches BD-09 double-transform, classify as BD-09
3. Apply the appropriate correction:
   - GCJ-02 -> WGS-84: iterative method (not approximate)
   - BD-09 -> GCJ-02 -> WGS-84: two-step conversion
4. Validate the correction by overlaying on OSM basemap (folium/contextily)
5. Export corrected dataset with proper EPSG:4326 CRS metadata

Report confidence level (HIGH/MEDIUM/LOW) for the CRS identification.
```

#### Variables to Customize

- `[N]`: Feature count
- Sample coordinates (5-10 representative points)
- `[feature_type]`: What the data represents
- `[geographic_area]`: Expected location
- `[N_sample]`: Number of sample points for offset detection

#### Expected Output

A Python script that identifies the CRS, applies corrections if needed, and exports a properly georeferenced dataset. The script should include a visual validation map and a confidence assessment.

#### Validation Checklist

- [ ] Corrected points visually align with reference features on OSM basemap
- [ ] Offset magnitude is consistent across the dataset (not random noise)
- [ ] Output CRS metadata is correctly set to EPSG:4326
- [ ] Confidence level is documented with supporting evidence

#### Cost Optimization

Include 5-10 sample coordinates directly in the prompt -- this lets the AI perform preliminary analysis without needing to run code, often identifying the CRS in a single response.

#### Dark Arts Tip

In China, most web-scraped coordinate data is GCJ-02 (from Amap/Tencent) or BD-09 (from Baidu). If the source is a government dataset labeled "WGS-84" but points consistently land ~500m northwest of their true position, it is almost certainly undeclared GCJ-02.

#### Related Pages

- [Data Sources Catalog](../data-sources/) -- CRS conventions for Chinese government and commercial data
- [Python Geo Stack](../tools/python-geo-stack.md) -- pyproj, eviltransform, coordtransform
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- client-side GCJ-02 conversion for web maps

#### Extensibility

Wrap the CRS detection logic into a reusable `detect_crs(gdf)` function that returns a `CRSDetectionResult` dataclass with `.crs`, `.confidence`, `.evidence`, and `.correction_applied` fields.

---

## Section 2: Spatial Operations

---

### Prompt 5 -- Advanced Spatial Joins

#### Scenario (实际场景)

You need to perform spatial joins across multiple large datasets -- point-in-polygon, nearest neighbor, and distance-band joins -- and must choose the right engine (PostGIS, GeoPandas, or DuckDB Spatial) based on data volume and performance requirements.

#### Roles

User: Spatial Analyst | AI: Spatial Join Performance Specialist

#### Prompt Template

```
You are a Senior Spatial Data Engineer who optimizes spatial join performance.

I need to perform the following spatial join:
- Left dataset: [description], [N_left] features, [geometry_type], CRS [EPSG]
- Right dataset: [description], [N_right] features, [geometry_type], CRS [EPSG]
- Join type: [point-in-polygon | nearest-neighbor | distance-band [D] meters]
- Columns to carry: [list columns from each side]
- Relationship: [one-to-one | one-to-many | many-to-many]

Provide THREE implementations:

1. **GeoPandas** (for < 500K features):
   - Use sjoin / sjoin_nearest with appropriate `how` and `predicate`
   - Include spatial index hints

2. **PostGIS** (for 500K - 50M features):
   - SQL with ST_Contains / ST_DWithin / ST_DistanceSphere as appropriate
   - Include CREATE INDEX for spatial columns
   - Use LATERAL JOIN for nearest-neighbor

3. **DuckDB Spatial** (for 10M+ features, file-based):
   - SQL on GeoParquet files with spatial extension
   - Demonstrate predicate pushdown benefits

For each implementation:
- Show complete runnable code
- Estimate memory usage and runtime for my data size
- Note gotchas (e.g., CRS must be projected for meter-based distance)
```

#### Variables to Customize

- Dataset descriptions, sizes, geometry types, CRS
- Join type and distance parameter
- Columns to preserve from each dataset
- Join cardinality (one-to-one, one-to-many)

#### Expected Output

Three complete, runnable implementations (GeoPandas, PostGIS, DuckDB) with performance notes and a recommendation for which to use given the stated data volumes.

#### Validation Checklist

- [ ] All three implementations produce identical join results on a test subset
- [ ] CRS handling is explicit (reprojection if source CRSs differ)
- [ ] Spatial indexes are created before the join operations
- [ ] NULL handling: features with no match are preserved (left join) or documented (inner join)

#### Cost Optimization

State the data volume upfront -- this lets the AI skip the small-data GeoPandas solution when you clearly need PostGIS or DuckDB, saving a full implementation worth of tokens.

#### Dark Arts Tip

For nearest-neighbor joins in PostGIS, always use `LATERAL JOIN` with `ORDER BY geom <-> ref_geom LIMIT 1` -- this exploits the GiST index for KNN and is orders of magnitude faster than computing all distances.

#### Related Pages

- [PostGIS Reference](../tools/postgis-recipes.md) -- spatial join patterns and index strategies
- [Python Geo Stack](../tools/python-geo-stack.md) -- GeoPandas sjoin internals
- [Data Sources Catalog](../data-sources/) -- typical dataset sizes for planning

#### Extensibility

Add a fourth implementation using Apache Sedona (GeoSpark) for cluster-scale joins exceeding 100M features, with PySpark integration.

---

### Prompt 6 -- Overlay Analysis

#### Scenario (实际场景)

You need to compute the intersection of land use zones with flood risk areas to determine which parcels fall in high-risk zones. The overlay must preserve attributes from both layers, handle slivers, and maintain topological consistency.

#### Roles

User: Environmental Analyst | AI: Spatial Overlay & Topology Specialist

#### Prompt Template

```
You are a Senior GIS Analyst specializing in overlay analysis and topological integrity.

I need to perform a [intersection | union | difference | symmetric_difference] overlay:
- Layer A: [description], [N_a] [geometry_type] features, CRS [EPSG]
- Layer B: [description], [N_b] [geometry_type] features, CRS [EPSG]

Attributes to preserve:
- From Layer A: [list columns]
- From Layer B: [list columns]

Write a Python script (GeoPandas 0.14+, Shapely 2.0+) that:
1. Validates both layers (fix invalid geometries first)
2. Ensures CRS alignment (reproject if needed)
3. Performs the overlay operation using gpd.overlay()
4. Post-processes results:
   a. Remove sliver polygons (area < [threshold] m2)
   b. Remove null/empty geometries
   c. Recalculate area column in [unit: m2 / hectares / km2]
   d. Dissolve fragments that share the same attribute combination (optional)
5. Validates output topology:
   - No overlapping polygons (if input was non-overlapping)
   - No gaps introduced (compare total area before/after)
   - Geometry type consistency
6. Exports to GeoPackage with metadata

Also provide the equivalent PostGIS SQL using ST_Intersection / ST_Union / ST_Difference
for comparison.

Report: feature count before/after, area before/after, sliver count removed.
```

#### Variables to Customize

- Overlay operation type (intersection, union, difference, symmetric_difference)
- Layer descriptions, sizes, geometry types, CRS
- Attribute columns to preserve
- `[threshold]`: Sliver polygon area threshold
- `[unit]`: Area calculation unit
- Whether to dissolve fragments

#### Expected Output

A Python script performing the overlay with sliver cleanup and topology validation, plus equivalent PostGIS SQL. The script should report area accounting (before/after comparison).

#### Validation Checklist

- [ ] Total area change is within acceptable tolerance (< 0.01% for intersection/union)
- [ ] No sliver polygons remain below the specified threshold
- [ ] All attributes from both layers are preserved in the output
- [ ] Output geometry type is consistent (no mixed Polygon/MultiPolygon issues)

#### Cost Optimization

If only a spatial filter is needed (not a true geometric overlay), use `sjoin` instead of `overlay` -- it is 10-100x faster because it skips geometry computation.

#### Dark Arts Tip

After `gpd.overlay()`, always run `gdf.geometry = gdf.geometry.buffer(0)` to clean up any micro-topology artifacts, then filter `gdf = gdf[gdf.area > threshold]` to remove slivers -- this two-step cleanup prevents downstream topology errors.

#### Related Pages

- [PostGIS Reference](../tools/postgis-recipes.md) -- ST_Intersection, ST_Union optimization
- [Python Geo Stack](../tools/python-geo-stack.md) -- GeoPandas overlay internals
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- Turf.js overlay for small datasets

#### Extensibility

Add support for weighted attribute interpolation (e.g., population proportional to area of intersection) using areal-weighted reaggregation from the `tobler` library.

---

### Prompt 7 -- Buffer & Proximity Analysis

#### Scenario (实际场景)

A planning department needs variable-distance buffers around different facility types (schools get 200m, hospitals get 500m, factories get 1000m) and isochrone-based service areas using actual road networks, not simple Euclidean buffers.

#### Roles

User: Urban Planner | AI: Proximity & Network Analysis Specialist

#### Prompt Template

```
You are a Senior Spatial Analyst specializing in proximity analysis and service areas.

I have a point dataset of [facilities_description] with [N] features in CRS [EPSG].
Each feature has a "type" column with values: [list facility types].

Part 1 -- Variable-Distance Buffers:
Write a Python script (GeoPandas 0.14+) that:
1. Assigns buffer distance based on facility type:
   [type_A]: [distance_A] meters
   [type_B]: [distance_B] meters
   [type_C]: [distance_C] meters
2. Creates buffers in a projected CRS (auto-select UTM zone or use [EPSG_proj])
3. Dissolves overlapping buffers of the same type
4. Calculates buffered area per type
5. Identifies areas covered by multiple facility types (overlay intersection)

Part 2 -- Isochrone Service Areas:
Using OSMnx (1.9+) and NetworkX:
1. Download the road network for [study_area] from OSM
2. For each facility point, calculate [5, 10, 15] minute walking isochrones
   (walking speed: [speed] km/h)
3. Convert isochrones to polygons (convex hull or concave hull)
4. Identify population within each isochrone (join with [population_dataset])

Part 3 -- Gap Analysis:
1. Identify areas within [study_area] NOT covered by any facility buffer
2. Calculate gap area and percentage of total study area
3. Export gap polygons for planning review

Export all results to GeoPackage (one layer per analysis).
```

#### Variables to Customize

- `[facilities_description]`: Type of facilities
- `[N]`: Feature count
- Buffer distances per facility type
- `[EPSG]`, `[EPSG_proj]`: Geographic and projected CRS
- `[study_area]`: Area name for OSM download
- `[speed]`: Walking/driving speed for isochrones
- `[population_dataset]`: Reference population data

#### Expected Output

A multi-part Python script producing variable-distance buffers, network-based isochrones, and a gap analysis, all exported as layers in a single GeoPackage.

#### Validation Checklist

- [ ] Buffers are computed in a projected CRS (not geographic degrees)
- [ ] Dissolved buffers do not have interior rings from overlapping features
- [ ] Isochrone polygons do not extend beyond the road network extent
- [ ] Gap area + covered area = total study area (within tolerance)

#### Cost Optimization

For Part 2, download the OSM network once and cache it locally (`ox.save_graphml`) -- repeated downloads for iterative analysis waste both time and API goodwill.

#### Dark Arts Tip

Euclidean buffers in geographic CRS (EPSG:4326) produce elliptical shapes near the poles and are wrong everywhere. Always reproject to a local projected CRS first. If you are lazy, `gdf.to_crs(gdf.estimate_utm_crs())` auto-selects the correct UTM zone.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- OSMnx, NetworkX integration
- [Data Sources Catalog](../data-sources/) -- OSM data access patterns
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- Turf.js buffer for web previews

#### Extensibility

Replace Euclidean buffers with travel-time isochrones from a routing engine (OSRM, Valhalla) for a more realistic service area analysis.

---

### Prompt 8 -- Spatial Clustering

#### Scenario (实际场景)

You have a large dataset of crime incidents, disease cases, or customer locations and need to identify spatial clusters, classify hot/cold spots, and determine whether the clustering is statistically significant or just visual noise.

#### Roles

User: Spatial Analyst / Epidemiologist | AI: Spatial Statistics & Clustering Specialist

#### Prompt Template

```
You are a Senior Spatial Statistician specializing in point pattern analysis and clustering.

I have a point dataset of [event_type] with [N] features in CRS [EPSG].
Attributes include: [list relevant columns, e.g., timestamp, severity, category].
Study area: [description].

Write a Python script (scikit-learn 1.4+, PySAL 4.9+, GeoPandas 0.14+) that performs:

1. **DBSCAN / HDBSCAN Clustering**:
   - Determine optimal eps using k-distance plot (for DBSCAN)
   - Run HDBSCAN with min_cluster_size=[M] as a robust alternative
   - Assign cluster labels to each point
   - Calculate cluster statistics: size, density, centroid, convex hull

2. **Getis-Ord Gi* Hot Spot Analysis**:
   - Aggregate points to a [hex_size] H3 hexagonal grid (or [cell_size] m regular grid)
   - Calculate event count per cell
   - Compute Gi* z-scores and p-values using PySAL esda
   - Classify cells: Hot Spot (99%, 95%, 90%), Cold Spot, Not Significant

3. **Kernel Density Estimation**:
   - Compute KDE surface using [bandwidth] meter bandwidth
   - Export as GeoTIFF raster at [resolution] meter resolution

4. **Visualization**:
   - DBSCAN clusters as colored points with noise in gray
   - Gi* results as a diverging choropleth (red=hot, blue=cold)
   - KDE as a heatmap overlay on contextily basemap

Export: GeoPackage (clusters, hot_spots layers), GeoTIFF (KDE), PNG maps.
```

#### Variables to Customize

- `[event_type]`: Domain (crime, disease, retail, etc.)
- `[N]`: Point count
- `[EPSG]`: CRS (must be projected for distance-based clustering)
- `[M]`: HDBSCAN min_cluster_size
- `[hex_size]` or `[cell_size]`: Aggregation grid resolution
- `[bandwidth]`, `[resolution]`: KDE parameters

#### Expected Output

A comprehensive clustering analysis script producing labeled clusters, hot spot classification, and a KDE surface, with publication-ready visualizations.

#### Validation Checklist

- [ ] DBSCAN/HDBSCAN uses projected coordinates (not lat/lon degrees)
- [ ] Gi* p-values are corrected for multiple testing (FDR or Bonferroni)
- [ ] KDE bandwidth is justified (not arbitrary)
- [ ] Cluster count is stable across reasonable parameter variations

#### Cost Optimization

Ask for HDBSCAN only (skip DBSCAN) -- HDBSCAN automatically handles varying densities and does not require the eps parameter, saving the entire k-distance plot analysis.

#### Dark Arts Tip

When using Gi* on aggregated grid cells, always include neighbors of empty cells as zeros, not NaN -- dropping empty cells biases the analysis toward finding spurious hot spots by shrinking the denominator.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- PySAL, scikit-learn spatial modules
- [Data Sources Catalog](../data-sources/) -- point event datasets
- [Visualization Techniques](../visualization/) -- choropleth and heatmap styling

#### Extensibility

Add a temporal dimension using ST-DBSCAN (space-time DBSCAN) to detect clusters that are significant in both space and time, useful for disease outbreak or crime wave detection.

---

### Prompt 9 -- Network Analysis

#### Scenario (实际场景)

A logistics company needs to compute shortest paths, service areas, and origin-destination cost matrices on a real road network. The analysis must account for one-way streets, speed limits, and turn restrictions.

#### Roles

User: Transportation Analyst | AI: Network Analysis & Routing Specialist

#### Prompt Template

```
You are a Senior Transportation Engineer specializing in network analysis and routing.

Study area: [city/region].
I need to perform network analysis on the road network.

Part 1 -- Network Preparation (OSMnx 1.9+):
1. Download the drive/walk/bike network for [study_area] from OSM
2. Clean the network: remove disconnected components, simplify
3. Add travel time edge weights based on:
   - Speed limits from OSM tags (or default: [speeds] km/h per road type)
   - Edge length
4. Save as GraphML for reuse

Part 2 -- Shortest Path ([N_pairs] OD pairs):
1. Load OD pairs from [format]: columns [origin_lon, origin_lat, dest_lon, dest_lat]
2. Snap origins and destinations to nearest network nodes
3. Compute shortest path (Dijkstra) by [weight: time / distance]
4. Extract: route geometry (LineString), total distance (m), total time (min)
5. Export route geometries as GeoPackage

Part 3 -- Service Area (isochrone polygons):
1. From [N_facilities] facility locations, compute [5, 10, 15, 30] minute drive-time areas
2. Build isochrone polygons from reachable subgraphs
3. Handle overlapping service areas (union / keep separate)

Part 4 -- OD Cost Matrix:
1. Compute [N_origins] x [N_destinations] cost matrix (time + distance)
2. Export as CSV (origin_id, destination_id, time_min, distance_km)
3. Identify unreachable OD pairs (disconnected network components)

Performance: for large networks (> 100K edges), use igraph backend or OSRM.
```

#### Variables to Customize

- `[study_area]`: City or region name
- Network type (drive, walk, bike)
- `[speeds]`: Default speed limits per road class
- `[N_pairs]`: Number of OD pairs
- `[weight]`: Optimization criterion (time or distance)
- `[N_facilities]`: Number of facility locations for service areas
- Isochrone time breaks

#### Expected Output

A multi-part Python script performing network download, shortest path routing, service area generation, and OD cost matrix computation, all using OSMnx and NetworkX.

#### Validation Checklist

- [ ] Network graph is strongly connected (or largest component is used with a warning)
- [ ] Route geometries follow actual road segments (not straight lines)
- [ ] Service area polygons do not cross water bodies or other barriers
- [ ] OD matrix is symmetric for undirected networks, asymmetric for directed

#### Cost Optimization

For OD matrices with more than 1000 origins, switch from OSMnx/NetworkX to OSRM's table API -- it is compiled C++ and handles 10,000x10,000 matrices in seconds rather than hours.

#### Dark Arts Tip

OSMnx `nearest_nodes()` can snap your origin to a highway service road node instead of the actual highway. Use `nearest_edges()` with interpolation to get the true nearest point on the network, then split the edge to insert your origin as a temporary node.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- OSMnx, NetworkX, igraph
- [Data Sources Catalog](../data-sources/) -- OSM network data, road classification
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- Leaflet Routing Machine for web-based routing

#### Extensibility

Integrate real-time traffic data from a traffic API to compute time-dependent shortest paths that vary by departure time.

---

## Section 3: Statistical Analysis

---

### Prompt 10 -- Exploratory Spatial Data Analysis (ESDA)

#### Scenario (实际场景)

A researcher needs to determine whether a socioeconomic variable (e.g., income, unemployment, disease rate) exhibits spatial autocorrelation -- that is, whether nearby areas have similar values (clustering) or dissimilar values (dispersion), and to identify local hot/cold spots.

#### Roles

User: Researcher / Policy Analyst | AI: Spatial Statistics Specialist (PySAL Expert)

#### Prompt Template

```
You are a Senior Spatial Statistician and PySAL expert.

I have a polygon dataset of [geographic_units, e.g., census tracts] with [N] features.
CRS: [EPSG]. Variable of interest: "[variable_name]" representing [description].
Distribution: [approximately normal / right-skewed / count data].

Write a Python script (PySAL libpysal 4.9+, esda 2.5+, splot, GeoPandas 0.14+) that:

1. **Spatial Weights**:
   - Construct [Queen / Rook / KNN(k=N) / distance-band(d=D)] weights
   - Row-standardize the weights matrix
   - Report: min/max/mean neighbors, islands (features with 0 neighbors)
   - Handle islands: connect to nearest neighbor or warn

2. **Global Spatial Autocorrelation**:
   - Moran's I with [999] permutations
   - Report: I statistic, expected I, z-score, p-value (two-tailed)
   - Moran scatterplot (standardized variable vs spatial lag)
   - Interpret: positive (clustering), negative (dispersion), or random

3. **Local Spatial Autocorrelation (LISA)**:
   - Local Moran's I for each feature
   - Classify into quadrants: HH (hot spot), LL (cold spot), HL, LH
   - Apply FDR correction for multiple testing (p < [alpha])
   - LISA cluster map with significance overlay

4. **Getis-Ord Gi*** (complementary):
   - Compute Gi* for each feature
   - Classify: Hot Spot (90%, 95%, 99%), Cold Spot, Not Significant

5. **Report**:
   - Plain-language interpretation for non-technical stakeholders
   - Export: GeoPackage with LISA + Gi* columns, PNG maps, summary stats CSV

Warn if the variable should be rate-adjusted (e.g., raw counts vs rates).
```

#### Variables to Customize

- `[geographic_units]`: Census tracts, districts, grid cells, etc.
- `[N]`: Feature count
- `[EPSG]`: CRS
- `[variable_name]`, `[description]`: Variable of interest
- Weights type and parameters (Queen, Rook, KNN k, distance-band d)
- `[alpha]`: Significance threshold (default 0.05)
- Permutation count (default 999)

#### Expected Output

A complete ESDA pipeline script producing global and local spatial autocorrelation statistics, classified cluster maps, and a plain-language interpretation suitable for a policy report.

#### Validation Checklist

- [ ] Weights matrix has no islands (or islands are handled explicitly)
- [ ] Moran's I p-value is from permutation test, not asymptotic (more robust for small N)
- [ ] LISA classifications use FDR-corrected p-values (not raw p-values)
- [ ] Variable is a rate or ratio, not raw count (if applicable)
- [ ] Maps include legend, north arrow, and scale bar

#### Cost Optimization

Specify the weights type (Queen vs KNN) in the prompt -- if omitted, the AI will generate code for multiple weights types, tripling the output length and token cost.

#### Dark Arts Tip

Never run Moran's I on raw counts (e.g., total crime). Larger areas or more populated areas will always appear as "hot spots." Always use rates (crime per capita) or empirical Bayes smoothed rates (esda.Moran_BV) to avoid this ecological fallacy.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- PySAL, esda, splot
- [Data Sources Catalog](../data-sources/) -- census and administrative boundary data
- [Visualization Techniques](../visualization/) -- LISA and choropleth map styling

#### Extensibility

Add bivariate Moran's I (Moran_BV) to test spatial correlation between two variables (e.g., income vs education), or add a temporal dimension using Differential Moran's I.

---

### Prompt 11 -- Zonal Statistics

#### Scenario (实际场景)

A climate researcher needs to extract mean temperature, total precipitation, and land cover percentages for each administrative district from multiple raster datasets spanning a 20-year time series.

#### Roles

User: Climate / Environmental Researcher | AI: Raster-Vector Analysis Specialist

#### Prompt Template

```
You are a Senior Environmental Data Scientist specializing in raster-vector integration.

Inputs:
- Zones: [format] with [N_zones] [geometry_type] features, CRS [EPSG_vec]
  Key column: [zone_id_col]
- Rasters: [N_rasters] [format, e.g., GeoTIFF] files representing [description]
  CRS: [EPSG_ras], resolution: [res] meters, nodata: [nodata_val]
  Time range: [start_year] to [end_year], [temporal_frequency]

Write a Python script (rasterstats 0.19+ or exactextract 0.2+, rasterio 1.3+,
GeoPandas 0.14+) that:

1. **Single-raster zonal stats**:
   - For each zone, compute: [mean, median, std, min, max, count, nodata_count]
   - Handle CRS mismatch: reproject zones to raster CRS (not vice versa)
   - Use `exactextract` for sub-pixel accuracy (coverage-weighted)
   - Handle multiband rasters: compute stats per band

2. **Multi-raster time series**:
   - Loop over [N_rasters] files (one per [month/year])
   - Extract zonal stats for each timestep
   - Build a panel DataFrame: zone_id x timestep x statistic
   - Calculate per-zone temporal trends (linear regression slope)

3. **Categorical raster (land cover)**:
   - For each zone, compute fractional coverage per land cover class
   - Output: zone_id, class_1_pct, class_2_pct, ..., dominant_class

4. **Output**:
   - CSV: zone_id, [all_stats], [temporal_columns]
   - GeoPackage: zones with stats joined
   - Time series plot (matplotlib) for [N_sample_zones] sample zones

Use windowed reading (rasterio windows) to handle rasters larger than memory.
```

#### Variables to Customize

- Zone dataset details (format, feature count, geometry type, CRS)
- Raster details (format, CRS, resolution, nodata value, time range)
- Statistics to compute
- Temporal frequency (monthly, annual)
- Number of sample zones for time series plots

#### Expected Output

A Python script performing single-raster, multi-raster time series, and categorical zonal statistics, with outputs as CSV, GeoPackage, and time series plots.

#### Validation Checklist

- [ ] CRS alignment: zones are reprojected to raster CRS (not raster to vector CRS)
- [ ] Nodata pixels are excluded from statistics (count_nodata reported separately)
- [ ] Categorical percentages sum to 100% per zone (within floating-point tolerance)
- [ ] Time series has no gaps (all timesteps present for all zones)

#### Cost Optimization

Use `exactextract` instead of `rasterstats` for polygons -- it handles sub-pixel coverage weighting natively and is 5-10x faster on large zones.

#### Dark Arts Tip

When zones are much larger than raster pixels, `rasterstats` with `all_touched=True` includes edge pixels that barely intersect the zone, inflating the pixel count. Use `exactextract` with coverage fractions instead, or stick with the default `all_touched=False` and accept minor boundary pixel exclusion.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- rasterio, rasterstats, exactextract
- [Data Sources Catalog](../data-sources/) -- climate raster data sources (ERA5, CHELSA, WorldClim)
- [Visualization Techniques](../visualization/) -- time series and choropleth visualizations

#### Extensibility

Replace the per-file loop with `xarray` + `rioxarray` for lazy, chunked processing of multi-terabyte raster time series (e.g., daily ERA5 data).

---

### Prompt 12 -- Spatial Regression

#### Scenario (实际场景)

A researcher is modeling housing prices as a function of structural attributes and neighborhood characteristics, but suspects that standard OLS residuals are spatially autocorrelated, violating independence assumptions. They need to select the appropriate spatial regression model.

#### Roles

User: Quantitative Researcher | AI: Spatial Econometrics & Regression Specialist

#### Prompt Template

```
You are a Senior Spatial Econometrician specializing in regression diagnostics and model selection.

Dataset: [N] [geographic_units] in CRS [EPSG].
Dependent variable: [Y_name] -- [description, distribution]
Independent variables:
- [X1_name]: [description, type]
- [X2_name]: [description, type]
- [X3_name]: [description, type]
- ... (up to [K] predictors)

Write a Python script (PySAL spreg 1.5+, libpysal, esda, statsmodels, GeoPandas) that:

1. **OLS Baseline**:
   - Fit OLS regression: Y ~ X1 + X2 + ... + XK
   - Report coefficients, R-squared, AIC, BIC
   - Run spatial diagnostics on residuals:
     a. Moran's I on residuals (with spatial weights)
     b. Lagrange Multiplier tests: LM-lag, LM-error, Robust LM-lag, Robust LM-error

2. **Model Selection (decision tree)**:
   - If LM-lag significant & LM-error not: Spatial Lag model
   - If LM-error significant & LM-lag not: Spatial Error model
   - If both significant: use Robust versions, or fit both and compare AIC
   - If neither significant: OLS is sufficient

3. **Spatial Models**:
   - Spatial Lag (SAR): Y = rho * W * Y + X * beta + epsilon
   - Spatial Error (SEM): Y = X * beta + u, where u = lambda * W * u + epsilon
   - Report: coefficients, pseudo-R2, AIC, log-likelihood, rho/lambda

4. **GWR (Geographically Weighted Regression)** (mgwr 2.2+):
   - Fit GWR with adaptive bandwidth (AICc optimization)
   - Map coefficient surfaces for each variable
   - Identify variables with significant spatial variation

5. **Comparison Table**:
   | Model | R2/pseudo-R2 | AIC | BIC | Moran's I (resid) | Significant vars |
   Report in plain language which model is preferred and why.
```

#### Variables to Customize

- `[N]`, `[geographic_units]`: Sample size and spatial unit
- `[EPSG]`: CRS
- `[Y_name]`: Dependent variable
- `[X1_name]` ... `[XK_name]`: Independent variables
- Weights type (Queen, KNN, distance-band)

#### Expected Output

A complete spatial regression workflow from OLS diagnostics through model selection to GWR, with a comparison table and plain-language interpretation of results.

#### Validation Checklist

- [ ] OLS residuals are tested for spatial autocorrelation before fitting spatial models
- [ ] LM test decision tree is correctly implemented
- [ ] GWR bandwidth is optimized (AICc), not arbitrary
- [ ] Multicollinearity checked (VIF < 10 for all predictors)
- [ ] Results are interpreted in domain context, not just statistical significance

#### Cost Optimization

If you already know the residuals are spatially autocorrelated (from prior ESDA), skip the OLS diagnostics section and jump straight to spatial model fitting.

#### Dark Arts Tip

Spreg's Spatial Lag model estimates rho via Maximum Likelihood, but if rho is close to 1.0, the model is nearly non-identified. Always check that rho is well within (-1, 1) and examine the likelihood surface for flatness. If rho > 0.8, consider whether you actually have a spatial diffusion process or just omitted-variable bias.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- PySAL spreg, mgwr
- [Data Sources Catalog](../data-sources/) -- socioeconomic data for regression models
- [Visualization Techniques](../visualization/) -- GWR coefficient maps

#### Extensibility

Add a Spatial Durbin Model (SDM) that includes both spatially lagged Y and spatially lagged X variables, as a general specification that nests both SAR and SEM as special cases.

---

### Prompt 13 -- Statistical Test Advisor

#### Scenario (实际场景)

A junior analyst has a research question and a dataset but is unsure which spatial statistical method to use. They need an AI advisor that recommends appropriate tests based on the data characteristics, research question, and assumptions.

#### Roles

User: Junior Analyst / Graduate Student | AI: Spatial Statistics Methodologist & Advisor

#### Prompt Template

```
You are a Professor of Spatial Statistics advising a student on methodology.

My research question: "[research_question]"

My data:
- Spatial unit: [point / polygon / raster / trajectory]
- Sample size: [N]
- Dependent variable: [Y_name], type: [continuous / count / binary / categorical]
  Distribution: [normal / skewed / Poisson / binomial / unknown]
- Independent variables:
  [X1]: [type, description]
  [X2]: [type, description]
  ...
- Temporal component: [yes / no]. If yes: [frequency, time span]
- Study area: [description], CRS: [EPSG]

Based on these characteristics, recommend the most appropriate spatial statistical
methods. For EACH recommendation, provide:

1. **Method name** and one-sentence description
2. **Why it fits** my data and research question
3. **Key assumptions** I must verify
4. **Implementation**: Python library + function name (or R package)
5. **Interpretation guide**: what the key output statistics mean
6. **Limitations**: when this method fails or is misleading
7. **Difficulty level**: Beginner / Intermediate / Advanced

Rank recommendations from simplest to most rigorous.
Provide 3-5 methods spanning the complexity spectrum.

Also flag any data preparation steps I should perform first:
- CRS requirements (projected vs geographic)
- Rate adjustment (if count data)
- Normality transformation
- Spatial weights construction
```

#### Variables to Customize

- `[research_question]`: The substantive research question
- Spatial unit type and sample size
- Variable types and distributions
- Whether temporal data is involved
- Study area and CRS

#### Expected Output

A ranked list of 3-5 spatial statistical methods with implementation guidance, assumptions to verify, interpretation notes, and data preparation prerequisites.

#### Validation Checklist

- [ ] Recommendations match the data type (e.g., count data gets Poisson-based methods)
- [ ] Assumptions are listed and testable
- [ ] Python/R library versions are current (not deprecated)
- [ ] Methods span a range of complexity (at least one simple, one advanced)

#### Cost Optimization

Include the variable distribution type in the prompt -- this prevents the AI from recommending methods that assume normality when your data is count-based (Poisson), saving a round of correction.

#### Dark Arts Tip

If you tell the AI "my data is normally distributed" without checking, it will happily recommend OLS and GWR. Always run a Shapiro-Wilk test first. Spatial data is almost never normally distributed -- right-skewed with outliers is the norm. Log-transform or use a GLM-based spatial model instead.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- PySAL, statsmodels, scikit-learn
- [Data Sources Catalog](../data-sources/) -- common research datasets and their characteristics
- [Visualization Techniques](../visualization/) -- statistical result visualization

#### Extensibility

Build a decision-tree flowchart (as a Mermaid diagram) that maps data type + research question type to the recommended method, for quick reference without AI.

---

## Section 4: Data Transformation

---

### Prompt 14 -- Format Conversion

#### Scenario (实际场景)

Your organization is migrating from legacy Shapefiles to modern formats. You need to batch-convert hundreds of Shapefiles to GeoPackage, convert GeoJSON APIs to GeoParquet for analytics, and geocode CSV files into spatial datasets -- all while preserving CRS, attributes, and handling Chinese encoding.

#### Roles

User: GIS Data Manager | AI: Geospatial Format & Encoding Specialist

#### Prompt Template

```
You are a Senior Geospatial Data Engineer specializing in format migration and encoding.

I need to perform the following conversions:

Task A -- Shapefile to GeoPackage (batch):
- Source: directory with [N_shp] Shapefiles, encoding: [GBK / UTF-8 / mixed]
- Write a Python script (GeoPandas 0.14+, pyogrio engine) that:
  1. Scans directory for all .shp files
  2. Detects encoding per file (chardet fallback to [default_encoding])
  3. Converts each to GeoPackage preserving CRS, attributes, and field types
  4. Handles Shapefile limitations: field name truncation (>10 chars), date types
  5. Logs: filename, feature_count, source_encoding, CRS, output_size
  6. Skips already-converted files (idempotent)

Task B -- GeoJSON to GeoParquet:
- Source: [URL or file] returning GeoJSON (potentially large, [N_features] features)
- Convert to GeoParquet 1.1 with:
  - WKB geometry encoding, Snappy compression
  - PROJJSON CRS metadata, bbox column for spatial filtering
  - Row groups of 100K for efficient predicate pushdown

Task C -- CSV to Spatial:
- Source: CSV with columns [lon_col], [lat_col] (or [address_col] for geocoding)
- Create point geometries in EPSG:4326
- If address-based: use geocoding service [Nominatim / Gaode / Baidu]
- Validate: no null coordinates, points within expected bounding box
- Export to [target_format]

Report: total files, success/failure counts, total feature count, total output size.
```

#### Variables to Customize

- `[N_shp]`: Number of Shapefiles to convert
- `[default_encoding]`: Fallback encoding (GBK for Chinese data)
- GeoJSON source (URL or file path) and feature count
- CSV column names for coordinates or addresses
- Geocoding service preference
- `[target_format]`: Output format for CSV conversion

#### Expected Output

A multi-task Python script handling Shapefile-to-GeoPackage batch conversion, GeoJSON-to-GeoParquet conversion, and CSV-to-spatial geocoding, with comprehensive logging.

#### Validation Checklist

- [ ] GeoPackage field names preserve full names (not truncated to 10 chars like Shapefile)
- [ ] GeoParquet file is readable by both GeoPandas and DuckDB Spatial
- [ ] Chinese characters render correctly in all output formats
- [ ] Feature counts match between source and destination

#### Cost Optimization

For Task A, use `pyogrio` engine (not `fiona`) -- it is 3-5x faster for Shapefile reads and handles encoding detection more gracefully.

#### Dark Arts Tip

Shapefiles store dates as `YYYYMMDD` strings, not actual date types. When converting to GeoPackage, explicitly cast date columns with `pd.to_datetime()` before writing, or they will silently become strings in the output.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- pyogrio, pyarrow GeoParquet
- [Data Sources Catalog](../data-sources/) -- encoding conventions by data source
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- GeoJSON handling in the browser

#### Extensibility

Add a `--validate` flag that reads back each converted file and compares feature counts, CRS, and a random sample of attribute values against the source, providing an automated round-trip validation.

---

### Prompt 15 -- Aggregation & Generalization

#### Scenario (实际场景)

You have a detailed parcel-level dataset with 500,000 polygons that needs to be dissolved to neighborhood boundaries, simplified for web display, and generalized to different zoom levels -- all while preserving topology and minimizing visual artifacts.

#### Roles

User: Cartographer / Web GIS Developer | AI: Generalization & Topology Specialist

#### Prompt Template

```
You are a Senior Cartographer specializing in spatial generalization and multi-scale
representation.

Input: [format] with [N] [geometry_type] features in CRS [EPSG].
Dissolve field: [dissolve_col] (produces ~[N_dissolved] groups).

Write a Python script (GeoPandas 0.14+, Shapely 2.0+) and a mapshaper CLI pipeline:

1. **Dissolve**:
   - Dissolve by [dissolve_col] with aggregations:
     [agg_col_1]: [sum / mean / first / count]
     [agg_col_2]: [sum / mean / first / count]
   - Handle multipart: explode to singlepart if requested
   - Validate: no gaps or overlaps in dissolved output

2. **Simplification** (multi-scale):
   - Level 1 (full detail): tolerance [T1] meters (or retain [P1]% of vertices)
   - Level 2 (city-level): tolerance [T2] meters (retain [P2]%)
   - Level 3 (national): tolerance [T3] meters (retain [P3]%)
   - Use topology-preserving simplification (Visvalingam-Whyatt preferred)
   - Validate: no self-intersections, no topology breaks between neighbors

3. **Mapshaper pipeline** (CLI):
   mapshaper input.geojson \
     -simplify [P]% weighted keep-shapes \
     -filter-slivers min-area=[area]m2 \
     -clean \
     -o output.geojson format=geojson precision=0.000001

4. **Output**: GeoPackage with one layer per simplification level,
   plus GeoJSON for web consumption (with coordinate precision control).

Report: vertex count reduction per level, file size reduction, topology errors.
```

#### Variables to Customize

- `[dissolve_col]`: Column to dissolve by
- Aggregation rules per column
- Simplification tolerances or vertex retention percentages per level
- Whether to explode multipart geometries
- Coordinate precision for GeoJSON output

#### Expected Output

A Python script and mapshaper CLI pipeline producing multi-scale simplified outputs with topology validation and size reduction metrics.

#### Validation Checklist

- [ ] Dissolved output has no gaps or overlaps between neighbors
- [ ] Simplified geometries do not self-intersect
- [ ] Aggregated statistics are correct (spot-check sum/mean values)
- [ ] GeoJSON coordinate precision is appropriate for the target zoom level

#### Cost Optimization

For web tile generation, simplify only once at the most detailed level needed, then let the tile server (e.g., tippecanoe) handle further simplification per zoom level.

#### Dark Arts Tip

GeoPandas `simplify()` uses Douglas-Peucker, which does not preserve topology between adjacent polygons (gaps appear). Use `topojson.Topology()` from the `topojson` Python package or mapshaper's `-simplify` instead -- both maintain shared boundaries.

#### Related Pages

- [Tools Catalog](../tools/) -- mapshaper, tippecanoe
- [Python Geo Stack](../tools/python-geo-stack.md) -- GeoPandas simplify, topojson
- [Visualization Techniques](../visualization/) -- multi-scale web map design

#### Extensibility

Integrate with tippecanoe to produce PMTiles or MBTiles vector tile sets directly from the simplified outputs, enabling a complete generalization-to-tile pipeline.

---

### Prompt 16 -- Raster-Vector Conversion

#### Scenario (实际场景)

You need to vectorize a classified land cover raster into clean polygons for GIS analysis, and conversely rasterize a vector road network into a friction surface for cost-distance modeling. Both operations must handle large rasters (10,000 x 10,000+) without running out of memory.

#### Roles

User: Remote Sensing / GIS Analyst | AI: Raster-Vector Interoperability Specialist

#### Prompt Template

```
You are a Senior Geospatial Engineer specializing in raster-vector conversion at scale.

Task A -- Vectorize (raster to polygon):
- Input: [format, e.g., GeoTIFF] classified raster, [width]x[height] pixels,
  [N_classes] classes, CRS [EPSG], nodata=[nodata_val]
- Write a Python script (rasterio 1.3+, shapely 2.0+, GeoPandas 0.14+) that:
  1. Read raster in windows/chunks (handle rasters larger than memory)
  2. Vectorize using rasterio.features.shapes()
  3. Merge adjacent polygons of the same class (dissolve by class value)
  4. Simplify polygons (tolerance=[T] CRS units) to reduce vertex count
  5. Remove small polygons (area < [min_area] m2)
  6. Add class label column from lookup: {1: "[class_1]", 2: "[class_2]", ...}
  7. Export to GeoPackage

Task B -- Rasterize (vector to raster):
- Input: [format] vector with [N] features, attribute column [burn_col]
- Target grid: resolution [res] meters, extent from [reference_raster or bbox]
- Write a Python script that:
  1. Create target raster profile from reference (or from bbox + resolution)
  2. Rasterize features using rasterio.features.rasterize()
  3. Burn attribute values (not just presence/absence)
  4. Handle overlapping features: [max / min / sum / last] priority
  5. Set nodata for non-feature areas
  6. Export as GeoTIFF with proper CRS and nodata metadata

For both tasks: use windowed processing for rasters exceeding [max_memory] GB.
```

#### Variables to Customize

- Raster dimensions, class count, CRS, nodata value
- Simplification tolerance and minimum polygon area
- Class value-to-label lookup dictionary
- Vector dataset and burn column
- Target raster resolution and extent
- Overlap handling strategy
- Maximum memory budget

#### Expected Output

Two Python scripts: one for vectorization with windowed reading and polygon cleanup, one for rasterization with attribute burning and overlap handling.

#### Validation Checklist

- [ ] Vectorized polygons cover the same extent as the input raster (minus nodata areas)
- [ ] No gaps between adjacent vectorized polygons of different classes
- [ ] Rasterized output has correct resolution and extent matching the reference
- [ ] Burn values match source attribute values (spot-check 10 features)

#### Cost Optimization

For vectorization, apply `rasterio.features.shapes()` per window, then merge polygons across window boundaries at the end -- this keeps peak memory constant regardless of raster size.

#### Dark Arts Tip

`rasterio.features.shapes()` produces one polygon per pixel cluster. On a 10K x 10K raster, this can yield millions of tiny polygons. Always dissolve by class value immediately after vectorization, before any other processing, or you will run out of memory building the GeoDataFrame.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- rasterio, rasterio.features
- [Data Sources Catalog](../data-sources/) -- land cover and elevation rasters
- [Remote Sensing Prompts](remote-sensing-prompts.md) -- classification workflows that feed into vectorization

#### Extensibility

Add a `--smooth` flag that applies morphological operations (erosion + dilation) to the raster before vectorization, producing cleaner polygon boundaries without the staircase effect.

---

### Prompt 17 -- Temporal Data Operations

#### Scenario (实际场景)

A transportation researcher has GPS trajectory data with timestamps and needs to perform temporal joins (match GPS points to road segments by time window), compute speed/acceleration, and aggregate into hourly/daily patterns using a combination of xarray for raster time series and GeoPandas for vector trajectories.

#### Roles

User: Transportation / Movement Researcher | AI: Spatiotemporal Data Specialist

#### Prompt Template

```
You are a Senior Spatiotemporal Data Scientist specializing in movement data and
time series.

Dataset: [N] GPS points with columns [id_col, timestamp_col, lon_col, lat_col, ...].
CRS: EPSG:4326. Time range: [start] to [end]. Sampling interval: ~[interval] seconds.
Unique moving objects: [N_objects].

Write a Python script (GeoPandas 0.14+, MovingPandas 0.18+, xarray 2024+) that:

1. **Trajectory Construction** (MovingPandas):
   - Create TrajectoryCollection from point GeoDataFrame
   - Split trajectories at gaps > [gap_threshold] minutes
   - Calculate per-point: speed (km/h), acceleration, heading

2. **Temporal Joins**:
   - Join trajectories with [event_dataset] by time window:
     Match GPS points to events within +/- [time_window] minutes
   - Join trajectories with road segments by nearest spatial + temporal match
   - Handle time zones: convert all timestamps to [target_tz]

3. **Temporal Aggregation**:
   - Aggregate to [hourly / daily / weekly] summaries per [object / road_segment / zone]
   - Compute: trip count, total distance, mean speed, dwell time
   - Detect temporal patterns: peak hours, weekday vs weekend

4. **Raster Time Series** (xarray + rioxarray):
   - Load [N_timesteps] raster files as xarray DataArray with time dimension
   - Compute temporal statistics: mean, trend (per-pixel linear regression), anomaly
   - Extract time series at [N_points] point locations
   - Export temporal stats as GeoTIFF, time series as CSV

5. **Output**:
   - Trajectories as GeoPackage (LineString geometries with temporal attributes)
   - Aggregated stats as CSV and GeoPackage
   - Time series plots (matplotlib) for sample objects/locations
```

#### Variables to Customize

- GPS data schema (column names, sampling interval)
- `[N_objects]`: Number of unique moving objects
- `[gap_threshold]`: Trajectory splitting threshold
- `[event_dataset]`: Temporal join target
- `[time_window]`: Temporal join tolerance
- `[target_tz]`: Target time zone
- Aggregation granularity and grouping

#### Expected Output

A comprehensive spatiotemporal analysis script handling trajectory construction, temporal joins, temporal aggregation, and raster time series, with outputs in GeoPackage, CSV, and visualizations.

#### Validation Checklist

- [ ] Trajectory speeds are physically plausible (no 1000 km/h GPS jumps)
- [ ] Temporal joins respect both spatial and temporal proximity constraints
- [ ] All timestamps are in the same time zone after conversion
- [ ] Raster time series has no duplicate or missing timesteps

#### Cost Optimization

MovingPandas handles trajectory construction and speed calculation in a few lines -- do not ask the AI to implement these from scratch with raw GeoPandas, as the resulting code will be 10x longer and less robust.

#### Dark Arts Tip

GPS points often have timestamp duplicates (same object, same second, different coordinates). Always `drop_duplicates(subset=[id_col, timestamp_col], keep='first')` before building trajectories, or MovingPandas will raise cryptic indexing errors.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- MovingPandas, xarray, rioxarray
- [Data Sources Catalog](../data-sources/) -- GPS trajectory and raster time series data
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- Deck.gl trip layer for trajectory visualization

#### Extensibility

Add stay-point detection (identify locations where objects stop for > T minutes) using MovingPandas `StopDetector` to enrich trajectories with activity information.

---

## Section 5: Cloud-Native Workflows

---

### Prompt 18 -- DuckDB Spatial Analytics

#### Scenario (实际场景)

You have billions of rows of point data (e.g., cell phone pings, taxi trips, IoT sensor readings) stored as GeoParquet files on S3 or local disk. You need to run analytical SQL queries -- spatial joins, aggregations, filtering by bounding box -- without loading everything into memory.

#### Roles

User: Data Engineer / Analyst | AI: DuckDB Spatial & Cloud-Native Analytics Specialist

#### Prompt Template

```
You are a Senior Analytics Engineer specializing in DuckDB Spatial and cloud-native
geospatial.

Data: [N] GeoParquet files totaling [size] GB / [row_count] rows.
Storage: [local disk / S3 bucket: s3://bucket/path/ / HTTP URL]
Geometry: [point / polygon] in CRS [EPSG].
Key columns: [list columns with types].

Write a Python + DuckDB SQL script (DuckDB 1.1+, duckdb-spatial) that:

1. **Setup**:
   INSTALL spatial; LOAD spatial;
   INSTALL httpfs; LOAD httpfs;  -- if S3/HTTP
   SET s3_region='[region]';     -- if S3

2. **Schema exploration**:
   - Read schema from GeoParquet metadata (no full scan)
   - Count rows using Parquet metadata (instant, no I/O)
   - Sample 10 rows to verify geometry parsing

3. **Spatial queries**:
   a. Bounding box filter:
      SELECT * FROM read_parquet('[path]')
      WHERE ST_Within(geom, ST_MakeEnvelope([xmin],[ymin],[xmax],[ymax]))

   b. Spatial join (point-in-polygon):
      Join points with [polygon_file] using ST_Contains
      Aggregate: COUNT(*), SUM([value_col]) per polygon

   c. Distance query:
      Find all points within [D] meters of ([lon],[lat])

   d. Temporal + spatial filter:
      Points within [bbox] AND [timestamp_col] BETWEEN [start] AND [end]

   e. H3 hexagonal aggregation:
      Aggregate points to H3 resolution [res] hexagons

4. **Export results**:
   - To GeoParquet (for further analysis)
   - To GeoJSON (for visualization, limit [max_features])
   - To CSV with WKT geometry column

5. **Performance benchmarks**:
   Compare each query's wall-clock time against equivalent GeoPandas code.
```

#### Variables to Customize

- GeoParquet file paths and storage location (local, S3, HTTP)
- Row count and file size
- Geometry type and CRS
- Bounding box, point, and distance for spatial queries
- Timestamp column and range for temporal filter
- H3 resolution for hexagonal aggregation
- S3 credentials (if applicable)

#### Expected Output

A Python script with DuckDB SQL queries demonstrating spatial filtering, joins, aggregation, and export from GeoParquet, with performance comparisons against GeoPandas.

#### Validation Checklist

- [ ] DuckDB spatial extension is installed and loaded before any spatial query
- [ ] GeoParquet files are read directly (no intermediate conversion)
- [ ] Predicate pushdown is verified (row groups skipped shown in EXPLAIN)
- [ ] Query results match GeoPandas equivalent on a test subset
- [ ] S3 credentials are not hardcoded (use environment variables or IAM role)

#### Cost Optimization

DuckDB reads GeoParquet row-group statistics (including bbox) to skip irrelevant row groups entirely. Structure your GeoParquet files with spatial sorting (Hilbert curve) to maximize predicate pushdown efficiency.

#### Dark Arts Tip

DuckDB's `ST_Contains` on GeoParquet does not automatically use the Parquet bbox metadata for pushdown -- you must add an explicit bounding box `WHERE` clause on the raw coordinate columns (if available) alongside the `ST_Contains` predicate to get true pushdown. Otherwise it reads all row groups and filters in-memory.

#### Related Pages

- [Tools Catalog](../tools/) -- DuckDB, GeoParquet specification
- [Python Geo Stack](../tools/python-geo-stack.md) -- DuckDB Python API
- [Data Sources Catalog](../data-sources/) -- GeoParquet data sources (Overture Maps, Microsoft Buildings)

#### Extensibility

Add a DuckDB-based ETL pipeline that reads raw CSV/JSON, creates point geometries, spatially sorts by Hilbert curve, and writes optimized GeoParquet -- a complete ingestion pipeline.

---

### Prompt 19 -- COG Pipeline

#### Scenario (实际场景)

You have a collection of satellite imagery or analysis rasters that need to be served via HTTP range requests for web map visualization. You need to create Cloud-Optimized GeoTIFFs (COGs), validate their structure, and serve them through a STAC-compatible tile server.

#### Roles

User: Remote Sensing Engineer | AI: Cloud-Optimized Raster & STAC Specialist

#### Prompt Template

```
You are a Senior Cloud-Native Geospatial Engineer specializing in COG and STAC
workflows.

Input: [N_rasters] [format] rasters, [width]x[height] pixels each,
[N_bands] bands, CRS [EPSG], datatype [dtype], nodata [nodata_val].
Total size: ~[total_size] GB.

Write a Python + GDAL pipeline that:

1. **Create COGs** (GDAL 3.8+ or rio-cogeo 5.0+):
   - Convert each raster to COG with:
     - Internal tiling: 512x512
     - Overview levels: [2, 4, 8, 16, 32] (or auto-compute)
     - Compression: [DEFLATE / LZW / ZSTD / JPEG]
     - Predictor: 2 (integer) or 3 (floating point)
     - Resampling for overviews: [nearest / average / bilinear]
   - Preserve CRS, nodata, band descriptions, and statistics
   - Batch process with multiprocessing (N_workers=[W])

2. **Validate COGs**:
   - Use rio cogeo validate to verify:
     a. Internal tiling is present
     b. Overviews exist at all levels
     c. IFD order is correct (main image first)
     d. Ghost headers are optimized for HTTP range requests
   - Report: valid/invalid count, issues found

3. **Generate STAC metadata**:
   - For each COG, create a STAC Item with:
     - Spatial extent (bbox, geometry)
     - Temporal extent (from filename or metadata)
     - eo:bands extension (band names, wavelengths if applicable)
     - proj extension (CRS, shape, transform)
     - Asset link to the COG file
   - Create a STAC Collection wrapping all Items

4. **Serve** (optional):
   - titiler configuration for dynamic tile serving from COGs
   - Example tile URL and preview map (folium/leaflet)

Report: input/output sizes, compression ratio, validation results.
```

#### Variables to Customize

- Raster format, dimensions, band count, data type, nodata value
- Compression algorithm (DEFLATE for lossless, JPEG for RGB imagery)
- Overview resampling method
- Number of parallel workers
- S3 bucket path (if cloud-hosted)
- STAC Collection metadata (title, description, license)

#### Expected Output

A batch processing pipeline that creates, validates, and catalogs COGs with STAC metadata, optionally configured for dynamic tile serving.

#### Validation Checklist

- [ ] All output files pass `rio cogeo validate` with no warnings
- [ ] Overviews exist at all specified levels
- [ ] HTTP range requests return partial content (test with `curl -r 0-1023`)
- [ ] STAC Items have correct bbox and temporal extent
- [ ] Compression ratio is reasonable (not 1:1 indicating compression failure)

#### Cost Optimization

Use ZSTD compression (GDAL 3.4+) instead of DEFLATE -- it achieves similar ratios at 3-5x faster compression speed, significantly reducing batch processing time for large collections.

#### Dark Arts Tip

If your COG has overviews but tile requests are still slow, check the IFD order with `rio cogeo info`. The main image IFD must come first in the file, followed by overview IFDs in decreasing resolution. If reversed (common with naive GDAL translate), a single tile request requires reading the file header from the end, doubling latency.

#### Related Pages

- [Tools Catalog](../tools/) -- GDAL, rio-cogeo, titiler
- [Data Sources Catalog](../data-sources/) -- satellite imagery sources
- [Remote Sensing Prompts](remote-sensing-prompts.md) -- preprocessing before COG creation

#### Extensibility

Add a mosaic pipeline using `cogeo-mosaic` to create a MosaicJSON that combines multiple COGs into a seamless virtual layer for titiler.

---

### Prompt 20 -- STAC Catalog Operations

#### Scenario (实际场景)

A researcher needs to search, filter, and download satellite imagery from public STAC catalogs (e.g., Element84 Earth Search, Microsoft Planetary Computer) based on spatial extent, date range, cloud cover, and specific band requirements.

#### Roles

User: Remote Sensing Researcher | AI: STAC API & Satellite Data Access Specialist

#### Prompt Template

```
You are a Senior Remote Sensing Data Engineer specializing in STAC-based data access.

I need to find and download satellite imagery matching these criteria:
- STAC API endpoint: [URL, e.g., https://earth-search.aws.element84.com/v1]
- Collection: [collection_id, e.g., sentinel-2-l2a]
- Spatial extent: [bbox: xmin,ymin,xmax,ymax] or [GeoJSON geometry]
- Temporal range: [start_date] to [end_date]
- Filters: cloud cover < [max_cloud]%, [additional filters]
- Bands needed: [list bands, e.g., B02, B03, B04, B08, SCL]

Write a Python script (pystac-client 0.8+, rioxarray 0.15+, stackstac 0.5+) that:

1. **Search**:
   - Connect to STAC API and search with spatial/temporal/property filters
   - Report: total items found, date range, cloud cover distribution
   - Sort by cloud cover ascending

2. **Filter & Select**:
   - Filter to items with < [max_cloud]% cloud cover
   - Select best [N_scenes] scenes (least cloud, best coverage)
   - Verify all required bands are available in each item

3. **Download**:
   - Option A: Download individual band assets to local directory
     (with retry logic and progress bar)
   - Option B: Lazy-load as xarray DataArray using stackstac
     (for analysis without full download)

4. **Mosaic / Composite**:
   - Create a median composite from [N_scenes] scenes
   - Clip to study area extent
   - Export as COG (Cloud-Optimized GeoTIFF)

5. **Metadata**:
   - Save search parameters and selected item IDs for reproducibility
   - Export item metadata as GeoJSON (footprints with properties)
```

#### Variables to Customize

- STAC API endpoint URL
- Collection ID
- Spatial extent (bbox or GeoJSON)
- Date range and cloud cover threshold
- Required bands
- Number of scenes to select
- Whether to download or lazy-load

#### Expected Output

A Python script that searches a STAC catalog, filters and selects scenes, downloads or lazy-loads data, creates a cloud-free composite, and exports as COG with full reproducibility metadata.

#### Validation Checklist

- [ ] STAC search returns results (API endpoint and collection ID are correct)
- [ ] Cloud cover filter is applied correctly (verify against item metadata)
- [ ] All required bands are present in downloaded/loaded data
- [ ] Composite has no nodata gaps within the study area
- [ ] Search parameters are saved for reproducibility

#### Cost Optimization

Use `stackstac.stack()` with `chunksize` parameter to lazy-load only the spatial extent you need -- this avoids downloading full scenes when you only need a small area, reducing download time and storage by 90%+.

#### Dark Arts Tip

Public STAC APIs (especially Planetary Computer) have rate limits. Always add `time.sleep(0.5)` between paginated search requests, and use `pystac_client.Client.open(..., modifier=planetary_computer.sign_inplace)` for Planetary Computer to avoid unsigned URL expiration errors mid-download.

#### Related Pages

- [Data Sources Catalog](../data-sources/) -- STAC API endpoints and collection IDs
- [Remote Sensing Prompts](remote-sensing-prompts.md) -- analysis workflows after data download
- [Tools Catalog](../tools/) -- pystac-client, stackstac, rioxarray

#### Extensibility

Add a STAC-based change detection workflow: search two date ranges, download matching scenes, compute NDVI difference, and flag areas with significant vegetation change.

---

### Prompt 21 -- FlatGeobuf Streaming

#### Scenario (实际场景)

You are building a web map that needs to display vector features on demand. Instead of downloading the entire dataset upfront, you want to use FlatGeobuf's HTTP range-request capability to stream only the features within the current map viewport, enabling instant display of datasets with millions of features.

#### Roles

User: Web GIS Developer | AI: FlatGeobuf & Streaming Vector Specialist

#### Prompt Template

```
You are a Senior Web GIS Engineer specializing in cloud-native vector streaming.

I have a dataset with [N] features ([geometry_type]) in CRS EPSG:4326.
Current format: [source_format]. I want to serve it as FlatGeobuf for streaming.

Part 1 -- Create optimized FlatGeobuf (Python):
Write a script (GeoPandas 0.14+, or ogr2ogr) that:
1. Loads the source dataset
2. Builds a spatial index (R-tree packed Hilbert index)
3. Sorts features by spatial index for optimal streaming
4. Writes to FlatGeobuf format
5. Validates: file has spatial index header, features are spatially sorted
6. Report: file size, feature count, index overhead percentage

Part 2 -- Server configuration:
- Nginx/Apache config to serve .fgb files with:
  - CORS headers (Access-Control-Allow-Origin, Expose-Headers)
  - Accept-Ranges: bytes header
  - Content-Type: application/octet-stream
  - Cache-Control headers for CDN

Part 3 -- Client-side streaming (JavaScript):
- Using flatgeobuf npm package with deserialize() for bbox filtering
- Integrate with [Mapbox GL JS / MapLibre / Leaflet / OpenLayers]
- Add viewport-based loading: fetch features on map move/zoom
- Implement debounce and abort controller for rapid panning
- Show feature count and load time in UI

Part 4 -- Performance comparison:
Compare load time for [bbox] viewport:
- Full GeoJSON download vs FlatGeobuf range request
- Measure: bytes transferred, time to first feature, total load time
```

#### Variables to Customize

- `[N]`: Feature count
- `[geometry_type]`: Point, Polygon, LineString
- `[source_format]`: Source data format
- Web map library (Mapbox GL JS, MapLibre, Leaflet, OpenLayers)
- Server software (Nginx, Apache, S3 static hosting)
- Test bbox for performance comparison

#### Expected Output

A complete pipeline from FlatGeobuf creation (Python) through server configuration to client-side streaming code (JavaScript), with performance benchmarks against full GeoJSON download.

#### Validation Checklist

- [ ] FlatGeobuf file has a spatial index (verify with `ogrinfo`)
- [ ] HTTP range requests work (test with `curl -H "Range: bytes=0-1023"`)
- [ ] CORS headers are correctly configured for cross-origin requests
- [ ] Client-side code handles abort on rapid pan (no stale request race conditions)

#### Cost Optimization

Host FlatGeobuf files on S3 or a CDN with proper CORS -- range requests are charged per-request but transfer only kilobytes per viewport load, making it far cheaper than serving full datasets through a tile server.

#### Dark Arts Tip

FlatGeobuf's spatial index only works if features are written in Hilbert-sorted order. If you write features in arbitrary order, the file will have an index but range requests will return the entire file because every row group overlaps every bbox. Always sort spatially before writing.

#### Related Pages

- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- FlatGeobuf client integration
- [Tools Catalog](../tools/) -- FlatGeobuf, ogr2ogr, static hosting
- [Python Geo Stack](../tools/python-geo-stack.md) -- GeoPandas FlatGeobuf I/O

#### Extensibility

Add a PMTiles fallback for clients that do not support HTTP range requests, using tippecanoe to generate vector tiles from the same source data.

---

## Section 6: Client Deliverable Data Prep

---

### Prompt 22 -- Data Package for Client (甲方数据交付包)

#### Scenario (实际场景)

You are delivering a spatial analysis project to a government client (甲方). The deliverable must include the data itself, comprehensive metadata, a validation report, coordinate system documentation, and a data dictionary -- all formatted per the client's acceptance standards (数据验收标准). Chinese documentation is required.

#### Roles

User: Project Manager / Lead Analyst | AI: Data Delivery & Documentation Specialist

#### Prompt Template

```
You are a Senior GIS Project Manager preparing a data delivery package for a Chinese
government client (甲方), following standard acceptance procedures (数据验收规范).

Project: [project_name]
Client: [client_name]
Deliverables:
- [N_datasets] spatial datasets: [list with descriptions]
- CRS: [EPSG code]
- Formats required: [GeoPackage / Shapefile / FileGDB] + [GeoJSON / KML for preview]

Write a Python script that assembles the delivery package:

1. **Data Export**:
   - Export each dataset to required formats
   - Ensure Chinese field names are preserved (encoding: UTF-8 for GPKG, GBK for SHP)
   - Add metadata tables to GeoPackage (gpkg_metadata, gpkg_metadata_reference)

2. **Metadata Document** (元数据文档):
   Generate a Word document (python-docx) or PDF containing:
   - 数据基本信息: name, CRS, extent, feature count, format
   - 属性字段说明 (data dictionary): field name, type, description, sample values
   - 坐标系说明: CRS details, datum, projection parameters
   - 数据质量说明: completeness, accuracy, consistency metrics
   - 数据来源: source datasets, processing steps, software versions

3. **Validation Report** (质量检查报告):
   - Run all checks from Prompt 3 (Data Quality Audit)
   - Format results as a formal report with 合格/不合格 status
   - Include maps showing spatial extent and sample features

4. **Directory Structure**:
   [project_name]_delivery/
   ├── 数据/                    # Data files
   ├── 元数据/                  # Metadata
   ├── 质量报告/                # Quality reports
   ├── 预览/                    # Preview files
   └── README.txt               # Package description

5. **Checksum & Manifest**:
   - Generate MD5 checksums for all files
   - Create manifest.csv: filename, size, checksum, feature_count
```

#### Variables to Customize

- `[project_name]`, `[client_name]`: Project identification
- Dataset list with descriptions
- Required formats (varies by client)
- CRS and encoding requirements
- Quality check thresholds
- Whether to generate Word (.docx) or PDF reports

#### Expected Output

A Python script that assembles a complete client delivery package with data files, Chinese-language metadata documents, quality reports, preview maps, and a checksum manifest.

#### Validation Checklist

- [ ] All files in the manifest exist and checksums match
- [ ] Chinese text renders correctly in all documents (UTF-8/GBK as appropriate)
- [ ] Shapefile .cpg file specifies correct encoding
- [ ] Quality report status (合格/不合格) is accurate
- [ ] Directory structure matches the client's acceptance template

#### Cost Optimization

Template the metadata document structure once and reuse across projects -- only the variable content changes. Store the template as a .docx with placeholder fields that the script fills in.

#### Dark Arts Tip

Government clients in China often require Shapefile delivery despite its limitations. Always include a `.cpg` file set to `GBK` (or `UTF-8` if the client agrees) -- without it, Chinese attribute text will render as garbled characters in ArcGIS. Also set field name aliases in the accompanying data dictionary since Shapefile truncates field names to 10 characters.

#### Related Pages

- [Data Sources Catalog](../data-sources/) -- government data standards and formats
- [Tools Catalog](../tools/) -- python-docx, folium, GDAL encoding options
- [JavaScript Spatial Ops](../js-bindbox/spatial-ops.md) -- preview map generation

#### Extensibility

Add a `--encrypt` option that password-protects the delivery ZIP file and generates a separate key document, for sensitive government data that requires secure transmission.

---

### Prompt 23 -- Automated Data Profiling Report

#### Scenario (实际场景)

Before sharing a dataset with colleagues or clients, you want to generate a comprehensive profile that reveals everything about the data at a glance: schema, value distributions, spatial extent, geometry statistics, CRS details, and potential quality issues -- all as a self-contained HTML or PDF report.

#### Roles

User: Data Analyst / Data Steward | AI: Geospatial Data Profiling Specialist

#### Prompt Template

```
You are a Senior Data Engineer building an automated geospatial data profiling tool.

Input: [format] file at [path], representing [description].
Expected CRS: [EPSG]. Feature count: approximately [N].

Write a Python script (GeoPandas 0.14+, Shapely 2.0+, matplotlib, Jinja2) that
generates a comprehensive data profile report:

1. **Schema Profile**:
   - Column inventory: name, dtype, null count, null %, unique count
   - For numeric columns: min, max, mean, median, std, percentiles
   - For string columns: min/max length, top 10 values with frequencies
   - For datetime columns: min, max, range, gaps
   - Data type recommendations (e.g., "column X is int64 but could be int16")

2. **Geometry Profile**:
   - Geometry type distribution
   - CRS details: authority, code, datum, unit, is_projected
   - Bounding box: xmin, ymin, xmax, ymax
   - Geometry statistics: vertex count, area (polygons), length (lines)
   - Invalid geometry count with top 5 invalidity reasons
   - Empty geometry count

3. **Spatial Distribution**:
   - Feature density heatmap (matplotlib, overlaid on contextily basemap)
   - Spatial extent map with bounding box
   - Cluster indicator: clustered or uniform?

4. **Quality Flags**:
   - Columns with >50% null values
   - Potential duplicate features
   - Outlier coordinates
   - Mixed geometry types
   - CRS mismatch warnings

5. **Output Formats**:
   - Self-contained HTML (Jinja2 template with embedded base64 PNG charts)
   - PDF version (via weasyprint or pdfkit)
   - JSON summary (machine-readable)

Make the report visually clean with a table of contents, color-coded quality flags
(green=good, yellow=warning, red=critical), and collapsible sections.
```

#### Variables to Customize

- `[format]`, `[path]`: Input file details
- `[description]`: Dataset domain description
- `[EPSG]`: Expected CRS
- `[N]`: Approximate feature count
- Output format preference (HTML, PDF, or both)
- Custom quality thresholds (null percentage warning level, outlier distance)

#### Expected Output

A Python script that generates a publication-quality data profile report as self-contained HTML and/or PDF, with schema analysis, geometry statistics, spatial distribution maps, and color-coded quality flags.

#### Validation Checklist

- [ ] HTML report is fully self-contained (no external dependencies, all images are base64)
- [ ] All charts render correctly and are legible
- [ ] Quality flags accurately reflect the data (spot-check 3 flags manually)
- [ ] JSON summary contains all numeric metrics for pipeline integration
- [ ] Report generates successfully for both small (100 features) and large (1M features) datasets

#### Cost Optimization

For datasets with more than 100K features, compute statistics on a random sample (10K features) for the profile and note the sampling in the report -- this keeps generation time under 30 seconds regardless of dataset size.

#### Dark Arts Tip

When profiling a Shapefile, always read with `encoding='utf-8'` first, catch the `UnicodeDecodeError`, then retry with `encoding='gbk'` -- this two-step approach handles 99% of Chinese Shapefiles without requiring the user to know the encoding in advance.

#### Related Pages

- [Python Geo Stack](../tools/python-geo-stack.md) -- GeoPandas, matplotlib, Jinja2
- [Data Sources Catalog](../data-sources/) -- data quality expectations by source
- [Visualization Techniques](../visualization/) -- report styling and chart design

#### Extensibility

Integrate with Great Expectations or Pandera to define expectation suites that run alongside the profile, turning the report from descriptive (what the data looks like) to prescriptive (whether the data meets requirements).

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)

