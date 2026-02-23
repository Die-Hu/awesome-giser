# Data Analysis Prompts

> Prompt templates for geospatial data cleaning, querying, statistical analysis, and format conversion. Copy the template, fill in the `[variables]`, and paste into your AI tool of choice.

> **Quick Picks**
> - 🏆 **Most Useful**: [Fix Geometry Errors](#prompt-1--fix-geometry-errors) — every GIS analyst hits invalid geometries weekly
> - 🚀 **Time Saver**: [GeoParquet with DuckDB](#prompt-3--geoparquet-with-duckdb) — query millions of features in seconds without loading into memory
> - 🆕 **Cutting Edge**: [Validate Data Quality](#prompt-4--validate-data-quality) — automated QA pipeline that catches errors before they propagate

---

## Table of Contents

- [Data Cleaning & Preprocessing](#data-cleaning--preprocessing)
- [Spatial Joins & Queries](#spatial-joins--queries)
- [Statistical Analysis](#statistical-analysis)
- [Data Format Conversion](#data-format-conversion)
- [Data Quality & Validation](#data-quality--validation)

---

## Data Cleaning & Preprocessing

### Prompt 1 — Fix Geometry Errors

**Context:** You have a shapefile or GeoJSON with invalid geometries (self-intersections, null geometries, duplicate vertices) that cause operations to fail.

**Template:**

```
I have a [shapefile/GeoJSON/GeoPackage] containing [number] features representing [feature type, e.g., building footprints].
The file has the following geometry issues: [describe issues — e.g., self-intersections, null geometries, tiny sliver polygons].
The CRS is [EPSG code].

Write a Python script using GeoPandas (0.14+) and Shapely (2.0+) to:
1. Load the file
2. Report how many features have invalid geometries
3. Attempt to fix geometries using buffer(0) and make_valid
4. Remove any features that remain invalid after fixing
5. Remove duplicate geometries
6. Save the cleaned result to [output format]

Include logging so I can see how many features were fixed vs removed.
```

**Variables to customize:**
- File format and path
- Feature type and count
- Specific geometry issues observed
- EPSG code
- Output format

**Expected output format:** A complete Python script with imports, logging, and a `main()` function.

---

### Prompt 2 — Standardize Attribute Table

**Context:** You received data from multiple sources with inconsistent column names, mixed encodings, and messy string values.

**Template:**

```
I have a GeoDataFrame with these columns: [list columns].
The data has these problems:
- Column [A] has mixed case strings (e.g., "Beijing", "BEIJING", "beijing")
- Column [B] has dates in inconsistent formats (e.g., "2024-01-15", "01/15/2024", "Jan 15, 2024")
- Column [C] has numeric values stored as strings with commas (e.g., "1,234.56")
- Some rows have trailing whitespace in text fields
- Encoding issues in column [D] (Chinese/Japanese characters garbled)

Write a GeoPandas (0.14+) cleaning script that standardizes all columns, with a summary report of changes made.
```

**Variables to customize:**
- Column names and types
- Specific data quality issues
- Target standardization rules (e.g., all lowercase, ISO date format)

**Expected output format:** Python script with a `clean_geodataframe()` function returning the cleaned GDF and a dict of change statistics.

---

### Prompt 3 — Detect and Handle Outliers in Spatial Data

**Context:** Your point dataset may contain GPS errors, coordinate typos, or features that fall outside the expected study area.

**Template:**

```
I have a point dataset of [feature type] in [study area] with CRS [EPSG code].
I suspect some points are erroneous because:
- Some coordinates may be outside [country/region] boundaries
- Some points may be GPS errors (coordinates swapped, wrong hemisphere)
- Some attribute values in column [X] are unrealistic (expected range: [min]-[max])

Write a Python script that:
1. Clips points to a bounding box of [study area extent]
2. Flags points with swapped lat/lon
3. Identifies statistical outliers in column [X] using IQR method
4. Produces a summary and exports flagged vs clean datasets
```

**Variables to customize:**
- Feature type, study area, CRS
- Bounding box coordinates or boundary file
- Attribute column and expected range

**Expected output format:** Python script outputting two GeoPackage layers (clean and flagged) plus a text summary.

---

## Spatial Joins & Queries

### Prompt 1 — Proximity-Based Spatial Join (PostGIS)

**Context:** You need to find features from one table that are within a specified distance of features in another table.

**Template:**

```
Write a PostGIS query to find all [target features, e.g., schools] within [distance, e.g., 500 meters]
of [source features, e.g., flood zones].

Table details:
- Table A: [table name], geometry column: [geom column], SRID: [SRID], key columns: [list]
- Table B: [table name], geometry column: [geom column], SRID: [SRID], key columns: [list]

Requirements:
- Use ST_DWithin for performance (not ST_Distance < X)
- Include the actual distance in the result
- Handle CRS transformation if SRIDs differ
- Return: [list desired output columns]
- Order by distance ascending
```

**Variables to customize:**
- Target and source feature types
- Distance and unit
- Table names, geometry columns, SRIDs
- Desired output columns

**Expected output format:** A single optimized PostGIS SQL query with comments explaining each clause.

---

### Prompt 2 — Point-in-Polygon Aggregation

**Context:** You want to count or summarize point features within polygon boundaries (e.g., incidents per district).

**Template:**

```
I have two GeoPandas GeoDataFrames:
- `points_gdf`: [description, e.g., crime incidents] with columns [list columns]
- `polygons_gdf`: [description, e.g., police districts] with columns [list columns]

Both use CRS [EPSG code].

Write a Python script that:
1. Performs a spatial join (points in polygons)
2. Counts points per polygon
3. Calculates the [mean/sum/median] of column [X] per polygon
4. Calculates point density (count / polygon area in km²)
5. Joins results back to the polygon GeoDataFrame
6. Exports to [format] with a choropleth-ready column
```

**Variables to customize:**
- Point and polygon descriptions
- Column names for aggregation
- Aggregation function (count, mean, sum, etc.)
- Output format

**Expected output format:** Python script with functions for join, aggregation, and export.

---

### Prompt 3 — Nearest Neighbor Lookup

**Context:** For each feature in dataset A, find the closest feature in dataset B and report the distance.

**Template:**

```
Write a Python script using GeoPandas (0.14+) and SciPy (1.11+) (or scikit-learn 1.3+ BallTree) to perform
a nearest-neighbor spatial join:

- For each point in `gdf_a` ([description]), find the nearest feature in `gdf_b` ([description])
- Both are in CRS [EPSG code]
- Return: all columns from gdf_a, the matched ID from gdf_b, and distance in [meters/km]
- Handle large datasets efficiently (gdf_a has [N] rows, gdf_b has [M] rows)

If the CRS is geographic (lat/lon), use haversine distance. If projected, use Euclidean.
```

**Variables to customize:**
- Descriptions and sizes of both datasets
- CRS and distance unit
- Columns to carry over

**Expected output format:** Python script using BallTree for efficiency, with progress reporting for large datasets.

---

## Statistical Analysis

### Prompt 1 — Spatial Autocorrelation Report

**Context:** You want to test whether a variable exhibits spatial clustering and identify hot/cold spots.

**Template:**

```
I have a polygon GeoDataFrame of [geographic units, e.g., census tracts] with a numeric column
"[variable name]" representing [description, e.g., median household income].
CRS: [EPSG code]. Number of features: [N].

Write a Python script using PySAL (libpysal 4.9+ and esda 2.5+) to:
1. Construct a spatial weights matrix (Queen contiguity)
2. Calculate Global Moran's I with significance testing (999 permutations)
3. Calculate Local Moran's I (LISA) for each feature
4. Classify features into HH, HL, LH, LL, and Not Significant clusters (p < 0.05)
5. Add the LISA cluster classification as a new column
6. Export the result and print a summary report

Interpret the results in plain language suitable for a non-technical stakeholder.
```

**Variables to customize:**
- Geographic unit and variable
- Weights type (Queen, Rook, KNN, distance-band)
- Significance threshold
- Number of permutations

**Expected output format:** Python script plus a printed interpretation paragraph.

---

### Prompt 2 — Recommend Appropriate Statistical Tests

**Context:** You are unsure which spatial statistical method to apply to your research question.

**Template:**

```
I am studying [research question, e.g., "whether air pollution levels cluster near highways"].
My data:
- Dependent variable: [name, type, distribution]
- Independent variables: [list with types]
- Spatial unit: [points / polygons / raster cells]
- Sample size: [N]
- Study area: [description]
- CRS: [EPSG code]

Suggest the most appropriate spatial statistical tests or models, explaining:
1. Why each is suitable for my data
2. Assumptions I need to check
3. Python or R libraries to implement each
4. How to interpret the results
5. Common pitfalls to avoid

Rank the options from simplest to most rigorous.
```

**Variables to customize:**
- Research question
- Data characteristics
- Available software

**Expected output format:** A ranked list of 3-5 methods with rationale, implementation guidance, and interpretation notes.

---

## Data Format Conversion

### Prompt 1 — Batch Format Conversion

**Context:** You need to convert a collection of files from one geospatial format to another while preserving CRS and attributes.

**Template:**

```
Write a Python script to batch-convert all [source format, e.g., .shp] files in a directory
to [target format, e.g., GeoPackage] with the following requirements:

- Source directory: [path]
- Output directory: [path]
- Preserve the original CRS (do NOT reproject)
- Preserve all attribute columns and data types
- Handle encoding issues (source files may use [encoding, e.g., GBK/Shift_JIS])
- Log each file: success/failure, feature count, file size
- Skip files that already have a converted counterpart (idempotent)

Use [GDAL/OGR 3.8+ | GeoPandas 0.14+ with pyogrio engine | Fiona 1.9+] as the conversion engine.
```

**Variables to customize:**
- Source and target formats
- Directory paths
- Encoding
- Preferred library

**Expected output format:** A standalone Python script with CLI arguments, logging, and error handling.

---

### Prompt 2 — CRS Transformation with Validation

**Context:** You need to reproject data from one CRS to another with confidence that the transformation is correct.

**Template:**

```
I need to reproject a [format] file from [source CRS, e.g., EPSG:4326] to [target CRS, e.g., EPSG:32650].

Write a Python script using pyproj (3.6+) and GeoPandas (0.14+) that:
1. Loads the file and confirms the source CRS
2. Reprojects to the target CRS using the most accurate transformation pipeline available
3. Validates the result by:
   - Comparing bounding boxes (transformed source vs result)
   - Spot-checking 5 random features' coordinates
   - Ensuring no NaN/Inf coordinates after transformation
4. Reports the transformation pipeline used (PROJ string)
5. Saves the reprojected file

If the source CRS is undefined, prompt the user to specify it rather than guessing.
```

**Variables to customize:**
- File format
- Source and target CRS
- Accuracy requirements

**Expected output format:** Python script with validation report printed to console.

---

### Prompt 3 — GeoParquet with DuckDB

**Context:** You have large geospatial datasets (millions of rows) and need fast analytical queries without loading everything into memory. GeoParquet + DuckDB is the modern high-performance stack for this.

**Template:**

```
I have a [describe dataset, e.g., "building footprints dataset with 12 million polygons"] stored as
[current format, e.g., "a set of Shapefiles / a GeoPackage / CSV with WKT geometry"].
I want to convert it to GeoParquet and run analytical queries using DuckDB.

Part 1 — Convert to GeoParquet:
Write a Python script using GeoPandas (0.14+) with the pyarrow (14.0+) engine to:
1. Read the source data in chunks (to handle large files without running out of memory)
2. Write to GeoParquet format with:
   - Geometry column encoded as WKB (GeoParquet 1.0 spec)
   - Proper CRS metadata (PROJJSON) embedded in the file
   - Snappy compression for a balance of speed and size
   - Row group size of 100,000 for efficient predicate pushdown
3. Report: input row count, output file size, compression ratio

Part 2 — Query with DuckDB:
Write a Python script using DuckDB (1.1+) with the spatial extension to:
1. Install and load the spatial extension: INSTALL spatial; LOAD spatial;
2. Read the GeoParquet file directly (no conversion needed)
3. Run these example queries:
   a. Count features per [category column]
   b. Filter features within a bounding box [xmin, ymin, xmax, ymax] using ST_Within
   c. Calculate total area by [group column] using ST_Area
   d. Find the [N] nearest features to a point [lon, lat] using ST_Distance
   e. Spatial join with a second GeoParquet file [describe] using ST_Intersects
4. Export query results to GeoJSON and CSV

Part 3 — GeoPandas comparison:
Show the equivalent GeoPandas code for query (b) and compare execution time
on a dataset of [N] rows (use %%timeit or time.perf_counter).
```

**Variables to customize:**
- Dataset description and current format
- Target queries and spatial operations
- Category and group columns
- Bounding box and point coordinates for spatial queries

**Expected output format:** Two Python scripts (conversion + querying) with DuckDB SQL, performance comparison notes.

---

## Data Quality & Validation

### Prompt 4 — Validate Data Quality

**Context:** You need an automated QA pipeline that checks incoming geospatial data against a specification before it enters your analytical workflow. This prevents garbage-in-garbage-out problems.

**Template:**

```
Write a Python data quality validation script using GeoPandas (0.14+), Shapely (2.0+), and
optionally Great Expectations (0.18+) or Pandera (0.18+) for a [format, e.g., GeoPackage] dataset.

The dataset represents [description, e.g., "parcel boundaries for a city cadastre"].
CRS should be [EPSG code].

Implement these validation checks:

SCHEMA CHECKS:
- Required columns exist: [list columns with expected dtypes]
- No unexpected columns (warn but don't fail)
- Column [A] is unique (primary key)
- Column [B] has no null values
- Column [C] values are in allowed set: [list allowed values]
- Column [D] is numeric and within range [min, max]

GEOMETRY CHECKS:
- No null geometries
- All geometries are valid (use Shapely 2.0 is_valid)
- All geometries are [expected type: Polygon / MultiPolygon / etc.]
- No empty geometries (is_empty)
- All features fall within [study area bounding box or reference boundary file]
- Minimum area threshold: [value] m² (remove dust polygons)
- No self-intersections

TOPOLOGY CHECKS (optional):
- No overlapping polygons within the dataset
- No gaps between adjacent polygons (tolerance: [value] m)
- No duplicate geometries

CROSS-REFERENCE CHECKS (optional):
- All values in column [X] exist in reference table [describe]
- Spatial coverage: dataset covers at least [percentage]% of the reference boundary

Output:
- Print a summary table: check name, status (PASS/FAIL/WARN), count of issues, sample IDs
- Export a GeoPackage with failed features tagged by which check they failed
- Return exit code 0 for all-pass, 1 for any-fail (CI/CD compatible)
- Generate a JSON report for machine-readable consumption

Make the script configurable via a YAML file so validation rules can be changed without editing code.
```

**Variables to customize:**
- Dataset format and description
- Expected CRS
- Column names, types, and constraints
- Geometry type and spatial bounds
- Topology requirements
- Reference datasets for cross-checks

**Expected output format:** A Python CLI script with YAML config, detailed reporting, and CI/CD-compatible exit codes.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
