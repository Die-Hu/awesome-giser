# Writing Templates

Practical templates and guidelines for writing GIS research papers. This guide covers paper structure, methodology descriptions, study area templates, LaTeX resources, and reference management tools to streamline your writing workflow.

> **Quick Picks**
> - **Elsevier journals (RSE, C&G):** elsarticle.cls ([Overleaf template](https://www.overleaf.com/latex/templates/elsevier-article/cdhrznpfvhnm))
> - **Taylor & Francis (IJGIS):** interact.cls
> - **IEEE (TGRS):** IEEEtran.cls ([Overleaf template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhkpxfmc))
> - **Reference manager:** Zotero + Better BibTeX (free, open source)
> - **Collaborative writing:** Overleaf (LaTeX) or Google Docs (early drafts)

---

## GIS Paper Structure

### Empirical / Application Paper

| Section | Content | Typical Length |
|---------|---------|---------------|
| Abstract | Problem, method, key findings, significance | 150-300 words |
| Introduction | Background, research gap, objectives, contributions | 1-2 pages |
| Related Work / Literature Review | Relevant prior studies, positioning your work | 1-2 pages |
| Study Area | Geographic context, why this area was chosen | 0.5-1 page |
| Data | Data sources, acquisition, preprocessing | 0.5-1 page |
| Methodology | Methods, algorithms, workflow | 2-4 pages |
| Results | Findings with figures, tables, maps | 2-3 pages |
| Discussion | Interpretation, comparison, limitations, implications | 1-2 pages |
| Conclusion | Summary, contributions, future work | 0.5-1 page |

### Methods / Algorithm Paper

| Section | Content |
|---------|---------|
| Abstract | Problem, proposed method, evaluation results |
| Introduction | Motivation, limitations of existing methods, contributions |
| Related Work | Existing approaches and their shortcomings |
| Proposed Method | Algorithm description, formal notation, complexity |
| Experimental Setup | Datasets, baselines, evaluation metrics |
| Results & Analysis | Quantitative comparison, ablation studies |
| Discussion | When it works, when it does not, scalability |
| Conclusion | Summary, code availability, future directions |

### Review / Survey Paper

| Section | Content |
|---------|---------|
| Abstract | Scope, number of reviewed papers, key findings |
| Introduction | Why this review, scope and criteria |
| Review Methodology | Search strategy, inclusion/exclusion criteria |
| Taxonomy / Classification | Categorization of reviewed approaches |
| Detailed Review | Analysis by category |
| Discussion & Trends | Open challenges, emerging directions |
| Conclusion | Summary, recommendations |

### Example Abstract Structure (Application Paper)

```
[1 sentence: Context and motivation]
[1 sentence: Research gap or problem]
[1-2 sentences: What this paper does / proposes]
[1-2 sentences: Method summary]
[1-2 sentences: Key results with numbers]
[1 sentence: Significance and implications]
```

---

## Methodology Description Templates

### Spatial Analysis Methodology

```
The spatial analysis was conducted using [software/library] (version X.X).
[Feature type] data were obtained from [source] at [resolution/scale].
All datasets were reprojected to [CRS, e.g., EPSG:4326 / UTM Zone XX]
to ensure spatial consistency.

[Analysis method, e.g., kernel density estimation / spatial autocorrelation /
hot spot analysis] was applied with the following parameters:
- [Parameter 1]: [value and justification]
- [Parameter 2]: [value and justification]

Statistical significance was assessed using [method] at a confidence level of [X%].
```

### Remote Sensing Methodology

```
[Sensor/satellite] imagery was acquired for the period [date range]
from [data source]. Images were selected based on [criteria, e.g.,
cloud cover < X%, specific season].

Preprocessing steps included:
1. [Atmospheric correction method]
2. [Geometric correction / orthorectification]
3. [Cloud masking]

[Classification / analysis method] was performed using [algorithm]
with [training data description]. Accuracy was assessed using
[method, e.g., confusion matrix with X ground truth points].

Overall accuracy: [X%], Kappa coefficient: [X.XX].
```

### Machine Learning Methodology

```
A [model type] was trained to [task description] using [framework] (version X.X).

Input features included:
- [Feature set 1]: [description and source]
- [Feature set 2]: [description and source]

The dataset was split into training ([X%]), validation ([X%]), and
test ([X%]) sets using [splitting strategy, e.g., spatial cross-validation
to avoid spatial autocorrelation].

Hyperparameters were tuned using [method] over the following search space:
- [Param 1]: [range]
- [Param 2]: [range]

Model performance was evaluated using [metrics: RMSE, F1, IoU, etc.].
```

### GIS Web Application Methodology

```
The web mapping application was developed using [framework, e.g.,
MapLibre GL JS / Leaflet] for the frontend and [backend, e.g.,
PostGIS + FastAPI / GeoServer] for spatial data services.

Geospatial data were stored in [database] with spatial indexing
(GiST) for efficient query performance. Vector tiles were
generated using [tool, e.g., Tippecanoe / Martin] at zoom levels
[range].

The application implements [key features, e.g., spatial query,
geoprocessing, real-time updates] and was deployed using
[infrastructure].
```

---

## Study Area Description Template

```
The study area is located in [region/city/country] (approximately
[lat]°N/S, [lon]°E/W), covering an area of approximately [X] km²
(Figure [N]).

[Physical geography description: terrain, elevation range, climate,
land cover types, etc.]

[Human geography description: population, urbanization level,
land use characteristics, etc.]

This area was selected because [justification: representative of
a broader phenomenon, data availability, policy relevance, etc.].
```


---

## Data Description Template

```
| Dataset | Source | Format | Resolution/Scale | Time Period | Access |
|---------|--------|--------|-----------------|-------------|--------|
| [Name] | [Provider] | [GeoTIFF/SHP/etc.] | [X m / 1:X] | [Date range] | [URL/DOI] |
```

### Example Data Table

```
| Dataset | Source | Format | Resolution | Time Period | Access |
|---------|--------|--------|------------|-------------|--------|
| Sentinel-2 L2A | Copernicus Hub | GeoTIFF | 10 m | Jun-Aug 2023 | https://scihub.copernicus.eu |
| SRTM DEM | USGS | GeoTIFF | 30 m | 2000 | https://earthexplorer.usgs.gov |
| OpenStreetMap roads | Geofabrik | Shapefile | N/A | 2024 extract | https://download.geofabrik.de |
| Census tracts | US Census Bureau | Shapefile | N/A | 2020 | https://www.census.gov/geographies |
```

---

## LaTeX Templates for GIS Journals

### Common Journal Templates

| Journal | Template Source | Format |
|---------|---------------|--------|
| Elsevier journals (RSE, C&G, JAG) | https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions | elsarticle.cls |
| Taylor & Francis (IJGIS, CaGIS) | https://www.tandf.co.uk/journals/authors/latex-templates/ | interact.cls |
| Springer (Geoinformatica) | https://www.springer.com/gp/livingreviews/latex-templates | svjour3.cls |
| ISPRS Archives | https://www.isprs.org/documents/orangebook/app5.aspx | isprs.cls |
| IEEE (TGRS, JSTARS) | https://template-selector.ieee.org/ | IEEEtran.cls |

### Overleaf Template Links

- **Elsevier (elsarticle):** [Overleaf template](https://www.overleaf.com/latex/templates/elsevier-article/cdhrznpfvhnm)
- **Springer Nature:** [Overleaf template](https://www.overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq)
- **IEEE:** [Overleaf template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhkpxfmc)
- **ISPRS Archives:** [Overleaf template](https://www.overleaf.com/latex/templates/isprs-annals-and-archives/tdbqjhgdswby)

### Elsevier GIS Paper Skeleton (elsarticle.cls)

```latex
\documentclass[preprint,12pt]{elsarticle}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{booktabs}  % Professional tables
\usepackage{siunitx}   % SI units

\journal{Computers \& Geosciences}

\begin{document}
\begin{frontmatter}
\title{A Spatial Analysis of Urban Heat Islands Using\\
Machine Learning and Sentinel-2 Imagery}

\author[inst1]{First Author\corref{cor1}}
\ead{first@university.edu}
\author[inst1]{Second Author}
\author[inst2]{Third Author}

\cortext[cor1]{Corresponding author}
\affiliation[inst1]{organization={Department of Geography, University Name},
  city={City}, country={Country}}
\affiliation[inst2]{organization={Institute of Remote Sensing},
  city={City}, country={Country}}

\begin{abstract}
% 150-300 words. Structure: context, gap, method, results, significance.
\end{abstract}

\begin{keyword}
GIS \sep remote sensing \sep urban heat island \sep machine learning \sep Sentinel-2
\end{keyword}
\end{frontmatter}

\section{Introduction}
\section{Study Area and Data}
\section{Methodology}
\section{Results}
\section{Discussion}
\section{Conclusions}

\section*{Data Availability}
The data and code used in this study are available at \url{https://doi.org/XX.XXXX/zenodo.XXXXXXX}.

\section*{CRediT Author Statement}
\textbf{First Author:} Conceptualization, Methodology, Software, Writing -- Original Draft.
\textbf{Second Author:} Data Curation, Validation, Writing -- Review \& Editing.

\bibliographystyle{elsarticle-harv}
\bibliography{references}
\end{document}
```

### IEEE GIS Paper Skeleton (IEEEtran.cls)

```latex
\documentclass[journal]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}

\begin{document}
\title{Deep Learning for Building Extraction\\from High-Resolution Remote Sensing Images}
\author{First Author,~\IEEEmembership{Student Member,~IEEE,}
  and Second Author,~\IEEEmembership{Senior Member,~IEEE}
\thanks{Manuscript received Month DD, YYYY.}
\thanks{F. Author is with the Department of Geomatics, University, City (e-mail: first@uni.edu).}}

\maketitle

\begin{abstract}
% 150-250 words for IEEE.
\end{abstract}

\begin{IEEEkeywords}
Building extraction, deep learning, remote sensing, semantic segmentation
\end{IEEEkeywords}

\section{Introduction}
\section{Related Work}
\section{Proposed Method}
\section{Experiments}
\subsection{Datasets}
\subsection{Evaluation Metrics}
\subsection{Results}
\section{Discussion}
\section{Conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

---

## Figure and Table Formatting Guidelines

### General Requirements

- **Resolution**: 300 DPI minimum for raster figures, vector (PDF/SVG) preferred
- **Width**: Single column (~8.5 cm) or double column (~17.5 cm)
- **Font size**: Labels readable at print size (8-10 pt minimum)
- **Color**: Ensure readability in grayscale if journal prints in B&W
- **File formats**: PDF, EPS, or high-resolution PNG/TIFF

### Map Figure Checklist

- [ ] Title or caption clearly describes the map content
- [ ] Legend explains all symbology
- [ ] Scale bar with appropriate units
- [ ] North arrow (if orientation is not obvious)
- [ ] Coordinate reference system noted
- [ ] Data source attribution
- [ ] Inset/overview map showing study area context

### Table Formatting

- Use horizontal rules only (no vertical rules in most journals)
- Align decimal points in numeric columns
- Include units in column headers, not in cells
- Bold or highlight key results
- Reference tables from the text


---

## Reference Management

### Recommended Tools

| Tool | Cost | Best Feature | Integration |
|------|------|-------------|-------------|
| Zotero | Free | Browser connector, group libraries | Word, LibreOffice, Google Docs, LaTeX (Better BibTeX) |
| Mendeley | Free | PDF annotation, social features | Word, LaTeX |
| Paperpile | $2.99/mo (academic) | Google Docs integration | Google Docs, Word, LaTeX |
| EndNote | $249.95 (institutional) | Journal style library | Word |
| JabRef | Free | BibTeX-native, open source | LaTeX |

### Zotero + Better BibTeX Workflow

1. Install Zotero with the Better BibTeX plugin
2. Configure auto-export of `.bib` file to your LaTeX project directory
3. Use Zotero browser connector to save papers directly from journal websites
4. Organize with collections matching your paper sections
5. Cite in LaTeX using `\cite{citekey}` with auto-generated keys

### GIS Reference Collections

- Create a shared Zotero group for your lab/research group
- Tag references by methodology (e.g., "spatial-autocorrelation", "deep-learning-RS")
- Use Zotero's Related tab to link papers that compare methods

### Recommended Zotero Collection Structure

```
My GIS Research/
  +-- Background & Theory/
  +-- Study Area Context/
  +-- Methodology/
  |     +-- Spatial Analysis Methods/
  |     +-- Machine Learning/
  |     +-- Remote Sensing/
  +-- Data Sources/
  +-- Comparison & Baselines/
  +-- Results Discussion/
  +-- Software & Tools/
```
