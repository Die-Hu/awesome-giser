# Reproducibility & Open Science for GIS Research

A comprehensive guide to making GIS research reproducible, transparent, and openly available. Covers FAIR data principles, computational reproducibility (Docker, Binder, conda), workflow managers, data and code repositories, preregistration, registered reports, open notebooks, and licensing.

> **Quick Picks**
> - **Environment management:** conda/mamba + `environment.yml` (simplest)
> - **Full reproducibility:** Docker + `Dockerfile` (most portable)
> - **Workflow automation:** Snakemake or DVC (data science pipelines)
> - **Code archival:** GitHub + Zenodo (auto-archive releases with DOI)
> - **Data archival:** Zenodo (free, 50 GB, DOI)
> - **Preregistration:** OSF (https://osf.io) or AsPredicted

---

## Why Reproducibility Matters

### The Reproducibility Crisis in GIS

```
Problems in GIS research reproducibility:
❌ "Software X was used for analysis" — no version, no parameters
❌ Data from proprietary source with no access instructions
❌ Custom scripts not shared or documented
❌ Results change with different software versions
❌ CRS/projection issues not documented
❌ Random seeds not set for ML experiments
❌ Processing steps described vaguely

What good reproducibility looks like:
✅ Full environment specification (Docker/conda)
✅ All code in version-controlled repository (GitHub)
✅ Data accessible with DOI (Zenodo) or download scripts
✅ Step-by-step workflow (Makefile/Snakemake)
✅ Random seeds documented
✅ CRS, parameters, software versions all specified
✅ README with "run from scratch" instructions
```

### Levels of Reproducibility

| Level | Description | Effort | Tools |
|-------|-------------|--------|-------|
| 0. Not reproducible | No code/data shared | None | N/A |
| 1. Available | Code/data exist somewhere | Low | GitHub |
| 2. Documented | README explains how to run | Low | GitHub + README |
| 3. Portable | Environment specified | Medium | conda + environment.yml |
| 4. Executable | Can run from scratch | Medium-High | Docker + Makefile |
| 5. One-click | Run in browser | High | Binder / CyberGIS |

---

## FAIR Data Principles

### What Is FAIR?

| Principle | Meaning | GIS Implementation |
|-----------|---------|-------------------|
| **F**indable | Data has persistent identifier, rich metadata | DOI via Zenodo; ISO 19115 metadata |
| **A**ccessible | Data retrievable via standard protocol | HTTPS download; OGC API; STAC catalog |
| **I**nteroperable | Data uses standard formats and vocabularies | GeoPackage, GeoTIFF, GeoParquet; EPSG codes |
| **R**eusable | Data has clear license and provenance | CC BY 4.0; detailed lineage documentation |

### FAIR Checklist for GIS Data

```
Findable:
□ Deposited in a repository (Zenodo, Figshare, Pangaea)
□ Has a DOI
□ Rich metadata (title, authors, description, keywords)
□ Spatial extent documented (bounding box, CRS)
□ Temporal extent documented
□ Indexed by data search engines

Accessible:
□ Downloadable via HTTPS
□ No unnecessary access restrictions
□ Long-term availability guaranteed (institutional repo)
□ API access if applicable (OGC, STAC)

Interoperable:
□ Standard open format (GeoPackage, GeoTIFF, GeoParquet, GeoJSON)
□ CRS documented (EPSG code)
□ Standard metadata schema (ISO 19115, Dublin Core)
□ Attribute names are descriptive (not "col1", "col2")
□ Units specified for all measurements
□ Uses controlled vocabularies where applicable

Reusable:
□ License specified (CC BY 4.0 recommended)
□ Provenance documented (how was data created?)
□ Processing steps documented
□ Quality assessment included
□ README explains structure and usage
```

### GIS Data Formats for FAIR Sharing

| Format | Type | Open | Standard | Best For |
|--------|------|------|----------|----------|
| GeoPackage (.gpkg) | Vector/Raster | ✅ | OGC | Default vector format |
| GeoJSON (.geojson) | Vector | ✅ | RFC 7946 | Web, small datasets |
| GeoParquet (.parquet) | Vector | ✅ | Apache | Large vector datasets |
| Cloud-Optimized GeoTIFF (COG) | Raster | ✅ | OGC | Cloud-native raster |
| GeoTIFF (.tif) | Raster | ✅ | OGC | Standard raster |
| NetCDF (.nc) | Raster/Grid | ✅ | OGC/CF | Climate, time series |
| FlatGeobuf (.fgb) | Vector | ✅ | Community | Streaming vector |
| PMTiles (.pmtiles) | Tiles | ✅ | Community | Serverless vector tiles |
| Shapefile (.shp) | Vector | ⚠️ | Esri | Legacy (avoid for new work) |

---

## Computational Environment Management

### conda / mamba — Simplest Approach

```yaml
# environment.yml — GIS research environment
name: gis-research
channels:
  - conda-forge
dependencies:
  - python=3.12
  - geopandas=1.0
  - rasterio=1.4
  - shapely=2.0
  - pyproj=3.7
  - fiona=1.10
  - matplotlib=3.9
  - scikit-learn=1.5
  - xarray=2024.9
  - rioxarray=0.17
  - contextily=1.6
  - folium=0.17
  - jupyter=1.1
  - dask-geopandas=0.4
  - pysal=24.7
  - mapclassify=2.8
  - pip:
    - torch>=2.4
    - torchgeo>=0.6
    - segment-anything
```

```bash
# Create environment
mamba env create -f environment.yml

# Activate
mamba activate gis-research

# Export (for sharing — exact versions)
mamba env export --no-builds > environment-lock.yml

# Recreate on another machine
mamba env create -f environment-lock.yml
```

### Docker — Full Reproducibility

```dockerfile
# Dockerfile for GIS research
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.9.0

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-venv git && \
    rm -rf /var/lib/apt/lists/*

# Python environment
COPY requirements.txt /tmp/
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Project files
WORKDIR /workspace
COPY . /workspace

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
```

```txt
# requirements.txt
geopandas==1.0.1
rasterio==1.4.1
scikit-learn==1.5.2
matplotlib==3.9.2
xarray==2024.9.0
rioxarray==0.17.0
contextily==1.6.2
jupyter==1.1.1
```

```bash
# Build and run
docker build -t gis-research .
docker run -p 8888:8888 -v $(pwd)/data:/workspace/data gis-research

# For GPU support (deep learning)
docker run --gpus all -p 8888:8888 gis-research
```

### Docker Compose — Multi-Service Research Stack

```yaml
# docker-compose.yml — Research environment with PostGIS
services:
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/workspace/notebooks
      - ./data:/workspace/data
    environment:
      - DATABASE_URL=postgresql://researcher:password@db:5432/research
    depends_on:
      - db

  db:
    image: postgis/postgis:16-3.4
    environment:
      POSTGRES_USER: researcher
      POSTGRES_PASSWORD: password
      POSTGRES_DB: research
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  pgdata:
```

### Binder — One-Click Cloud Execution

```
1. Your GitHub repo has:
   - environment.yml (conda) or requirements.txt (pip) or Dockerfile
   - Jupyter notebooks (.ipynb)

2. Create Binder link:
   https://mybinder.org/v2/gh/USERNAME/REPO/main

3. Add badge to README.md:
   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/USERNAME/REPO/main)

4. Anyone can click the badge → launches Jupyter in browser
   - No installation needed
   - Full environment reproduced
   - Perfect for reviewers and readers

Limitations:
- 1-2 GB RAM
- Sessions time out after ~10 minutes of inactivity
- Not for heavy computation (use CyberGIS-Jupyter instead)
```

### CyberGIS-Jupyter — Heavy GIS Computing

```
URL: https://cybergis.illinois.edu

Features:
- JupyterHub with GIS libraries pre-installed
- Access to HPC resources
- Large-scale spatial analysis
- Free for academic use
- Integration with XSEDE/ACCESS allocations

Use when:
- Analysis needs more than 2 GB RAM
- Processing large raster datasets
- Running ML training on GPUs
- Need persistent storage between sessions
```

### Comparison: conda vs. Docker vs. Binder

| Feature | conda | Docker | Binder |
|---------|-------|--------|--------|
| Setup complexity | Low | Medium | None (for users) |
| Portability | OS-dependent | Full | Full (browser) |
| System libraries | Limited | Full control | Full |
| GPU support | Via CUDA toolkit | Via nvidia-docker | Limited |
| Sharing effort | Share YAML file | Share Dockerfile | Share link |
| Reproducibility | Good (with lock file) | Excellent | Excellent |
| Best for | Daily development | Archival + deployment | Demos + review |

---

## Workflow Managers

### Why Use a Workflow Manager?

```
Without workflow manager:
  "Run script1.py, then script2.py, then script3.R, then..."
  → Steps forgotten, order confused, intermediate files lost

With workflow manager:
  `snakemake all` or `make all` or `dvc repro`
  → Entire pipeline runs in correct order
  → Only re-runs steps whose inputs changed
  → Reproducible from scratch
```

### Tool Comparison

| Tool | Language | Complexity | Best For |
|------|----------|-----------|----------|
| Make | Makefile | Low | Simple pipelines, LaTeX |
| Snakemake | Python-based | Medium | Complex data pipelines |
| DVC (Data Version Control) | CLI + Git | Medium | ML experiments + data versioning |
| Nextflow | Groovy-based | Medium-High | HPC/cloud pipelines |
| Prefect | Python | Medium | Orchestration + monitoring |
| Luigi | Python | Medium | Batch data processing |
| Dagster | Python | Medium-High | Data platform |

### Makefile — Simplest Workflow

```makefile
# Makefile for GIS research pipeline
.PHONY: all clean

all: results/accuracy_report.txt figures/fig1_study_area.pdf \
     figures/fig2_results.pdf paper/main.pdf

# Download data
data/raw/sentinel2.tif:
	python code/01_download_data.py

# Preprocessing
data/processed/ndvi.tif: data/raw/sentinel2.tif
	python code/02_preprocess.py

# Analysis
results/classification.gpkg: data/processed/ndvi.tif data/processed/dem.tif
	python code/03_classify.py

# Accuracy assessment
results/accuracy_report.txt: results/classification.gpkg data/validation/ground_truth.gpkg
	python code/04_accuracy.py

# Generate figures
figures/fig1_study_area.pdf: data/processed/study_area.gpkg
	python code/05_plot_study_area.py

figures/fig2_results.pdf: results/classification.gpkg results/accuracy_report.txt
	python code/06_plot_results.py

# Compile paper
paper/main.pdf: paper/main.tex paper/references.bib figures/*.pdf
	cd paper && latexmk -pdf main.tex

clean:
	rm -rf data/processed/* results/* figures/*.pdf paper/main.pdf
```

### Snakemake — Advanced Workflow

```python
# Snakefile for GIS analysis pipeline
configfile: "config.yaml"

rule all:
    input:
        "results/accuracy_report.txt",
        "figures/fig2_results.pdf",
        "paper/main.pdf"

rule download_sentinel:
    output: "data/raw/sentinel2_{tile}.tif"
    params: tile="{tile}"
    conda: "envs/download.yaml"
    script: "code/01_download.py"

rule preprocess:
    input: "data/raw/sentinel2_{tile}.tif"
    output: "data/processed/ndvi_{tile}.tif"
    conda: "envs/analysis.yaml"
    script: "code/02_preprocess.py"

rule classify:
    input:
        ndvi=expand("data/processed/ndvi_{tile}.tif", tile=config["tiles"]),
        dem="data/processed/dem.tif",
        training="data/training/samples.gpkg"
    output: "results/classification.gpkg"
    conda: "envs/analysis.yaml"
    threads: 4
    script: "code/03_classify.py"

rule accuracy:
    input:
        classification="results/classification.gpkg",
        ground_truth="data/validation/ground_truth.gpkg"
    output: "results/accuracy_report.txt"
    conda: "envs/analysis.yaml"
    script: "code/04_accuracy.py"

rule plot_results:
    input:
        classification="results/classification.gpkg",
        accuracy="results/accuracy_report.txt"
    output: "figures/fig2_results.pdf"
    conda: "envs/plotting.yaml"
    script: "code/06_plot_results.py"

rule compile_paper:
    input:
        tex="paper/main.tex",
        bib="paper/references.bib",
        figs=glob_wildcards("figures/{fig}.pdf")
    output: "paper/main.pdf"
    shell: "cd paper && latexmk -pdf main.tex"
```

### DVC — Data Version Control

```bash
# Initialize DVC in your Git repo
dvc init

# Track a large data file (stored in remote, not Git)
dvc add data/raw/sentinel2_mosaic.tif
git add data/raw/sentinel2_mosaic.tif.dvc .gitignore
git commit -m "Track Sentinel-2 mosaic with DVC"

# Set up remote storage (S3, GCS, Azure, SSH, local)
dvc remote add -d myremote s3://my-bucket/dvc-storage
dvc push

# Define a pipeline stage
dvc stage add -n preprocess \
    -d code/02_preprocess.py \
    -d data/raw/sentinel2_mosaic.tif \
    -o data/processed/ndvi.tif \
    python code/02_preprocess.py

# Reproduce the pipeline
dvc repro

# Track experiments (ML)
dvc exp run --set-param model.lr=0.001
dvc exp show  # Compare experiments
```

### DVC vs. Git LFS

| Feature | DVC | Git LFS |
|---------|-----|---------|
| Storage backend | S3, GCS, Azure, SSH, local | Git LFS server |
| Pipeline management | ✅ | ❌ |
| Experiment tracking | ✅ | ❌ |
| Cost | Free (you pay for storage) | Free tier limited |
| Best for | ML + data science | Simple large file tracking |

---

## Code Sharing Best Practices

### Repository Structure for GIS Research

```
my-gis-research/
├── README.md                 ← Project overview, how to reproduce
├── LICENSE                   ← MIT or Apache-2.0 for code
├── CITATION.cff              ← Machine-readable citation info
├── environment.yml           ← Conda environment
├── Dockerfile                ← Full reproducible environment
├── Makefile                  ← Pipeline automation
├── .gitignore                ← Ignore data/, outputs/, etc.
│
├── code/                     ← Analysis scripts
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_analysis.py
│   ├── 04_accuracy.py
│   ├── 05_visualize.py
│   └── utils/
│       ├── spatial_utils.py
│       └── ml_utils.py
│
├── notebooks/                ← Exploratory Jupyter notebooks
│   ├── 01_exploration.ipynb
│   └── 02_supplementary_analysis.ipynb
│
├── data/                     ← Data directory
│   ├── raw/                  ← Original data (gitignored)
│   ├── processed/            ← Processed data (gitignored)
│   └── README.md             ← Data access instructions
│
├── results/                  ← Output files (gitignored)
│   ├── classification.gpkg
│   └── accuracy_report.txt
│
├── figures/                  ← Publication figures
│   ├── fig1_study_area.pdf
│   └── fig2_results.pdf
│
├── paper/                    ← Manuscript
│   ├── main.tex
│   ├── references.bib
│   └── sections/
│
└── tests/                    ← Unit tests
    └── test_spatial_utils.py
```

### CITATION.cff — Make Your Code Citable

```yaml
# CITATION.cff — Citation File Format
cff-version: 1.2.0
message: "If you use this software, please cite it as below."
type: software
title: "Urban Heat Island Analysis Pipeline"
version: 1.0.0
date-released: 2025-03-15
doi: "10.5281/zenodo.XXXXXXX"
url: "https://github.com/username/uhi-analysis"
license: MIT
authors:
  - family-names: "Author"
    given-names: "First"
    orcid: "https://orcid.org/0000-000X-XXXX-XXXX"
    affiliation: "University Name"
  - family-names: "Author"
    given-names: "Second"
    orcid: "https://orcid.org/0000-000X-XXXX-XXXX"
keywords:
  - GIS
  - urban heat island
  - remote sensing
  - machine learning
```

### GitHub → Zenodo Archival

```
1. Go to https://zenodo.org and link your GitHub account
2. Enable the toggle for your repository
3. On GitHub, create a release:
   - Tag: v1.0.0
   - Title: "Code for [Paper Title]"
   - Description: "Code and analysis scripts for the paper
     '[Title]' published in [Journal]"
4. Zenodo automatically archives and assigns a DOI
5. Add the Zenodo DOI badge to your README:
   [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)]
   (https://doi.org/10.5281/zenodo.XXXXXXX)
6. Reference this DOI in your paper's code availability statement
```

### Software Heritage — Permanent Archival

```
URL: https://www.softwareheritage.org
Purpose: Long-term preservation of source code
Used by: HAL (France), several publishers

How to archive:
1. Visit https://archive.softwareheritage.org/save/
2. Enter your repository URL
3. Software Heritage crawls and archives it
4. Receive a persistent identifier (SWHID)
5. Use SWHID in publications for permanent reference

SWHID format: swh:1:dir:HASH
```

---

## Preregistration

### What Is Preregistration?

```
Definition: Publicly documenting your research plan (hypotheses,
methods, analysis) BEFORE collecting or analyzing data.

Why preregister?
- Prevents HARKing (Hypothesizing After Results are Known)
- Reduces publication bias
- Increases transparency
- Distinguishes confirmatory from exploratory analysis
- Strengthens your paper's credibility

For GIS research:
- Preregister your spatial analysis plan
- Document expected accuracy thresholds
- Specify which models you'll compare
- Define your study area and time period
- State your classification scheme before applying it
```

### Preregistration Platforms

| Platform | URL | Cost | Features |
|----------|-----|------|----------|
| OSF Registries | https://osf.io/registries | Free | Templates, DOI, embargo option |
| AsPredicted | https://aspredicted.org | Free | Simple 8-question template |
| ClinicalTrials.gov | https://clinicaltrials.gov | Free | Clinical trials (health GIS) |
| PROSPERO | https://www.crd.york.ac.uk/prospero/ | Free | Systematic review registration |

### OSF Preregistration Template (Adapted for GIS)

```
1. Research Questions / Hypotheses
   "We hypothesize that urban green spaces > 1 ha will show
   a cooling effect of > 1.5°C on surrounding areas as measured
   by Landsat 8 TIRS land surface temperature."

2. Study Area & Time Period
   "[City], [Country]. Analysis period: June-August 2024.
   CRS: UTM Zone XX (EPSG:326XX)."

3. Data Sources
   "Landsat 8 Level-2 (USGS); OpenStreetMap land use (Geofabrik);
   ERA5 climate reanalysis (CDS)."

4. Analysis Plan
   "Buffer analysis at 100m, 200m, 500m from green space boundaries.
   Linear regression with LST as dependent variable, green space
   area and distance as independent variables. Spatial autocorrelation
   assessed with Moran's I. If SAC present, use spatial lag model."

5. Accuracy / Validation
   "Validate LST against [N] weather station records. Target:
   RMSE < 2.0°C. Classification accuracy target: OA > 85%."

6. Software & Versions
   "Python 3.12, geopandas 1.0, rasterio 1.4, scikit-learn 1.5.
   Full environment in environment.yml."

7. Exploratory Analysis (post-hoc)
   "Any analysis not listed above will be clearly labeled as
   exploratory in the paper."
```

---

## Registered Reports

### What Are Registered Reports?

```
Traditional paper: Results → Paper → Review → Publish/Reject
  Problem: Publication bias (positive results published more)

Registered Report:
  Stage 1: Study design reviewed → In-Principle Acceptance (IPA)
  Stage 2: Conduct study → Report results → Published regardless

  Key benefit: Paper accepted BEFORE results are known
  → No publication bias
  → Negative results published
  → Methodology is rigorous (reviewed before data collection)
```

### Registered Report Timeline

```
Stage 1: Proposal
├── Write: Introduction, Methods, Proposed Analysis
├── Submit to journal
├── Peer review of study DESIGN
└── Decision: In-Principle Acceptance (IPA)
    → Paper WILL be published if you follow the plan

Stage 2: Results
├── Collect data following the registered plan
├── Conduct preregistered analyses
├── Write Results and Discussion
├── Submit complete manuscript
├── Review: Did authors follow the plan?
└── Published (almost always, given IPA)
```

### Journals Supporting Registered Reports (GIS-adjacent)

- Environmental Evidence
- Nature Human Behaviour
- PLOS Biology
- Royal Society Open Science
- PeerJ
- Check the full list: https://www.cos.io/initiatives/registered-reports

---

## Open Notebooks

### What Is an Open Notebook?

```
Open notebook = Making your ENTIRE research process transparent
- Lab notebooks
- Data exploration
- Failed experiments
- Meeting notes
- Intermediate results

Platforms:
- Jupyter Notebooks on GitHub (most common in GIS)
- Observable notebooks (interactive, web-based)
- Quarto documents (R + Python, publishable)
- Open Science Framework (OSF) wiki
```

### Jupyter Notebook Best Practices for Reproducibility

```python
# At the top of every notebook:

# 1. Environment info
import sys
print(f"Python: {sys.version}")

import geopandas as gpd
import rasterio
print(f"GeoPandas: {gpd.__version__}")
print(f"Rasterio: {rasterio.__version__}")

# 2. Random seed
import numpy as np
np.random.seed(42)

# 3. File paths as variables (not hardcoded)
from pathlib import Path
DATA_DIR = Path("../data")
RESULTS_DIR = Path("../results")
FIGURES_DIR = Path("../figures")

# 4. CRS as constant
TARGET_CRS = "EPSG:32650"  # UTM Zone 50N

# 5. Parameters in one place
PARAMS = {
    "buffer_distance": 500,  # meters
    "n_classes": 5,
    "test_size": 0.3,
    "random_state": 42,
}
```

### Quarto — Modern Scientific Publishing

```markdown
---
title: "Urban Heat Island Analysis"
author: "First Author"
format:
  html:
    code-fold: true
    toc: true
  pdf:
    documentclass: elsarticle
execute:
  echo: true
  warning: false
bibliography: references.bib
---

## Study Area

```{python}
import geopandas as gpd
import contextily as ctx

study_area = gpd.read_file("data/study_area.gpkg")
ax = study_area.plot(figsize=(8, 6), alpha=0.5, edgecolor='red')
ctx.add_basemap(ax, crs=study_area.crs)
```

Results show that... [@smith2024; @jones2023]
```

---

## Licensing

### Code Licenses

| License | Permissions | Conditions | Use For |
|---------|-------------|-----------|---------|
| MIT | Use, modify, distribute | Keep license notice | Simple, permissive |
| Apache 2.0 | Use, modify, distribute, patent grant | Keep license + state changes | Patent-safe |
| GPL 3.0 | Use, modify, distribute | Derivatives must be GPL | Copyleft (viral) |
| BSD 3-Clause | Use, modify, distribute | Keep license, no endorsement | Permissive |
| LGPL 3.0 | Use, modify, distribute | Library modifications must be LGPL | Libraries |
| AGPL 3.0 | Use, modify, distribute | Network use = distribution | Web services |

### Data Licenses

| License | Permissions | Recommended For |
|---------|-------------|----------------|
| CC BY 4.0 | Use, share, adapt with attribution | **Default for research data** |
| CC BY-SA 4.0 | Same as CC BY + share-alike | Data that should stay open |
| CC BY-NC 4.0 | Use, share, adapt, non-commercial | Restrictive (not recommended) |
| CC0 | Public domain, no restrictions | Maximum reuse |
| ODbL | Use, share, adapt with attribution | Database-specific (OpenStreetMap) |

### License Decision Flowchart

```
FOR CODE:
Want maximum adoption?
  YES → MIT or Apache 2.0
  NO  → Want derivatives to stay open?
          YES → GPL 3.0
          NO  → MIT

FOR DATA:
Want maximum reuse?
  YES → CC0 (public domain)
  NO  → CC BY 4.0 (attribution required)

Avoid: CC BY-NC (limits commercial use, which includes
many legitimate academic activities)
```

### Adding a License to Your Project

```
1. Code: Add LICENSE file to repository root
   - GitHub: Create new file → name it "LICENSE" → template options appear
   - Choose MIT or Apache 2.0 for research code

2. Data: Add LICENSE-DATA file or specify in README
   - "This dataset is licensed under CC BY 4.0:
     https://creativecommons.org/licenses/by/4.0/"
   - Include in Zenodo metadata when uploading

3. Paper: Usually covered by journal's copyright policy
   - Gold OA: Typically CC BY 4.0
   - Subscription: Copyright transferred to publisher
   - Preprint: Choose CC BY 4.0 when posting
```

---

## Reproducibility Checklist

### Before Submission

```
Code:
□ All analysis code in version-controlled repository
□ Repository is public (or will be at publication)
□ README explains how to reproduce results
□ CITATION.cff file present
□ License file present (MIT/Apache 2.0)
□ Code is commented and organized
□ No hardcoded file paths (use relative paths or config)
□ Random seeds set and documented

Environment:
□ environment.yml or requirements.txt or Dockerfile present
□ Software versions pinned (not floating)
□ Tested: fresh install from environment file works
□ Binder badge added to README (if applicable)

Data:
□ Raw data accessible (DOI, download script, or instructions)
□ Processed data documented (or scripts to regenerate)
□ Metadata included (CRS, attributes, units, provenance)
□ License specified (CC BY 4.0)
□ Data availability statement in paper

Workflow:
□ Pipeline documented (Makefile, Snakemake, or numbered scripts)
□ Can reproduce all results from scratch
□ Can reproduce all figures from scratch
□ Intermediate steps documented

Paper:
□ Data availability statement
□ Code availability statement
□ Software versions mentioned in methods
□ CRS documented in methods and figure captions
□ Parameters and thresholds documented
□ DOIs for code and data repositories
```

---

## Tools Summary

| Stage | Tool | Purpose | Cost |
|-------|------|---------|------|
| Environment | conda/mamba | Python/R environment management | Free |
| Environment | Docker | Full containerization | Free |
| Environment | Binder | Cloud execution | Free |
| Workflow | Make | Simple pipelines | Free |
| Workflow | Snakemake | Complex pipelines | Free |
| Workflow | DVC | Data + ML versioning | Free |
| Code hosting | GitHub | Version control, collaboration | Free |
| Code archival | Zenodo | DOI, long-term preservation | Free |
| Code archival | Software Heritage | Permanent archive | Free |
| Data hosting | Zenodo | General data repository | Free |
| Data hosting | Figshare | Figures, datasets | Free |
| Data hosting | Pangaea | Earth science data | Free |
| Preregistration | OSF | Study plan registration | Free |
| Publishing | Quarto | Reproducible documents | Free |
| Publishing | Jupyter Book | Documentation from notebooks | Free |
| Computing | CyberGIS-Jupyter | Cloud GIS computing | Free |
