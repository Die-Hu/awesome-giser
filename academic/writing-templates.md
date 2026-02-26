# Writing Templates for GIS Research

Practical templates and guidelines for writing GIS research papers. Covers paper structures for every type of GIS paper, methodology description templates, LaTeX and Typst skeletons, AI writing assistants, figure/table formatting, reference management, and CRediT authorship.

> **Quick Picks**
> - **Elsevier journals (RSE, C&G):** elsarticle.cls — [Overleaf template](https://www.overleaf.com/latex/templates/elsevier-article/cdhrznpfvhnm)
> - **Modern alternative:** Typst (typst.app) — faster compilation, simpler syntax
> - **Reference manager:** Zotero + Better BibTeX (free, open source)
> - **AI writing assistant:** Claude / GPT for drafts, Grammarly for grammar
> - **Collaborative writing:** Overleaf (LaTeX) or Typst (web app) or Google Docs (early drafts)

---

## Paper Structures by Type

### 1. Empirical / Application Paper

The most common GIS paper type. Reports a study applying GIS/RS methods to a real-world problem.

| Section | Content | Length | Tips |
|---------|---------|--------|------|
| Abstract | Problem, method, key findings, significance | 150-300 words | Write last; include quantitative results |
| Introduction | Background, research gap, objectives, contributions | 1-2 pages | End with clear research questions or hypotheses |
| Related Work | Relevant prior studies, positioning your work | 1-2 pages | Organize by theme, not chronologically |
| Study Area | Geographic context, why this area was chosen | 0.5-1 page | Include a location map |
| Data | Data sources, acquisition, preprocessing | 0.5-1 page | Use a table for datasets |
| Methodology | Methods, algorithms, workflow | 2-4 pages | Include a workflow diagram |
| Results | Findings with figures, tables, maps | 2-3 pages | Present results neutrally, save interpretation for Discussion |
| Discussion | Interpretation, comparison, limitations, implications | 1-2 pages | Compare with prior work; be honest about limitations |
| Conclusion | Summary, contributions, future work | 0.5-1 page | Don't introduce new information |

### 2. Methods / Algorithm Paper

Proposes a new spatial analysis method, algorithm, or computational approach.

| Section | Content | Tips |
|---------|---------|------|
| Abstract | Problem, proposed method, evaluation results | Include specific performance numbers |
| Introduction | Motivation, limitations of existing methods, contributions | Clearly state what's new |
| Related Work | Existing approaches and their shortcomings | Fair comparison, acknowledge strengths of others |
| Proposed Method | Algorithm description, formal notation, complexity | Pseudocode, diagrams, mathematical formulation |
| Experimental Setup | Datasets, baselines, evaluation metrics | Multiple datasets for generalizability |
| Results & Analysis | Quantitative comparison, ablation studies | Tables with bold best results |
| Discussion | When it works, when it doesn't, scalability | Honest failure case analysis |
| Conclusion | Summary, code availability, future directions | Link to open-source implementation |

### 3. Review / Survey Paper

Systematically reviews the state-of-the-art in a GIS research area.

| Section | Content | Tips |
|---------|---------|------|
| Abstract | Scope, number of reviewed papers, key findings | State the review methodology |
| Introduction | Why this review, scope and criteria | Justify the need — what's missing? |
| Review Methodology | Search strategy, inclusion/exclusion criteria | PRISMA flowchart recommended |
| Taxonomy / Classification | Categorization of reviewed approaches | Create a clear, logical taxonomy |
| Detailed Review | Analysis by category | Use consistent structure per category |
| Discussion & Trends | Open challenges, emerging directions | This is the most valuable section |
| Conclusion | Summary, recommendations | Actionable takeaways |

### 4. Data Descriptor Paper

Describes a new dataset for the GIS/RS community.

| Section | Content | Tips |
|---------|---------|------|
| Abstract | What the dataset is, coverage, resolution, access | Include DOI |
| Background & Summary | Context, motivation, gap in existing data | Why does this dataset matter? |
| Methods | How data was collected, processed, quality controlled | Full reproducibility |
| Data Records | Format, structure, file organization | Tables with attributes and descriptions |
| Technical Validation | Quality assessment, accuracy metrics | Comparison with existing data |
| Usage Notes | How to use the data, known limitations | Code examples for loading |
| Code Availability | Processing scripts, tools used | GitHub + Zenodo DOI |

**Best journals:** ESSD, Scientific Data, Data in Brief, Geoscience Data Journal

### 5. Software Paper

Describes a new GIS software tool, library, or package.

| Section | Content | Tips |
|---------|---------|------|
| Summary | What the software does, in one paragraph | Accessible to non-experts |
| Statement of Need | Gap in existing tools, target users | Who benefits and why? |
| Functionality | Key features, API design | Code examples |
| Implementation | Architecture, dependencies, performance | Design decisions |
| Example Usage | Real-world use case with code | Runnable example |
| Availability | License, repo URL, DOI | JOSS requires open source |

**Best journals:** JOSS (free, fast), C&G, JORS, SoftwareX

---

## Abstract Templates

### Application Paper Abstract

```
[Context] Urban heat islands (UHIs) pose significant health risks in rapidly
urbanizing cities, yet fine-scale UHI mapping remains challenging due to
satellite thermal sensor limitations.

[Gap] Existing studies rely on coarse-resolution thermal imagery (100+ m),
missing intra-urban temperature variability critical for targeted interventions.

[Method] This study integrates Sentinel-2 multispectral imagery (10 m) with
Landsat 8 thermal data through a random forest downscaling approach to produce
10 m land surface temperature maps for [City], [Country].

[Results] The downscaled product achieves RMSE of 1.2°C against in-situ
measurements (n=45) and reveals that small urban green spaces (< 1 ha) reduce
surrounding temperatures by 1.5-2.3°C.

[Significance] These fine-scale maps enable neighborhood-level heat mitigation
planning and demonstrate the feasibility of multi-sensor thermal downscaling
for global urban monitoring.
```

### Methods Paper Abstract

```
[Problem] Efficient spatial join operations are essential for large-scale
geospatial analysis, but existing methods scale poorly beyond billions of
geometry pairs.

[Limitation] Current approaches based on R-tree indexing suffer from O(n log n)
construction time and high memory consumption for massive datasets.

[Proposal] We propose SpatialHash, a GPU-accelerated spatial join algorithm
using hierarchical spatial hashing with adaptive grid resolution.

[Results] Experiments on four benchmark datasets (up to 10 billion geometry
pairs) show SpatialHash achieves 12-47x speedup over state-of-the-art CPU
methods and 3-8x over existing GPU approaches, while reducing memory usage by
60%.

[Impact] Our open-source implementation (https://github.com/...) enables
interactive spatial joins on commodity hardware.
```

---

## Methodology Description Templates

### Spatial Analysis

```
The spatial analysis was conducted using [software/library] (version X.X).
[Feature type] data were obtained from [source] at [resolution/scale].
All datasets were reprojected to [CRS, e.g., EPSG:4326 / UTM Zone XX]
to ensure spatial consistency.

[Analysis method, e.g., kernel density estimation / spatial autocorrelation /
hot spot analysis] was applied with the following parameters:
- [Parameter 1]: [value and justification]
- [Parameter 2]: [value and justification]

Statistical significance was assessed using [method] at a confidence level
of [X%]. Sensitivity analysis was performed by varying [parameter] from
[range] to evaluate robustness.
```

### Remote Sensing Classification

```
[Sensor/satellite] imagery was acquired for the period [date range]
from [data source]. Images were selected based on [criteria, e.g.,
cloud cover < X%, specific season].

Preprocessing steps included:
1. Atmospheric correction using [method, e.g., Sen2Cor, FLAASH]
2. Geometric correction / orthorectification to [CRS]
3. Cloud masking using [method, e.g., s2cloudless, Fmask]

[Classification method] was performed using [algorithm, e.g., Random Forest,
U-Net] with [training data description: source, number of samples per class].

Training data (n=[N]) were collected from [source: field survey, visual
interpretation, existing land use maps] and split into training ([X%]) and
validation ([X%]) sets using [strategy: random, spatial block, stratified].

Accuracy assessment was performed using an independent validation set (n=[N]).
Overall accuracy: [X%], Kappa coefficient: [X.XX], per-class F1 scores
are reported in Table [N].
```

### Machine Learning / Deep Learning

```
A [model type, e.g., ResNet-50, U-Net, XGBoost] was trained to [task
description] using [framework, e.g., PyTorch 2.x, scikit-learn 1.x].

Input features included:
- [Feature set 1]: [description, source, dimensions]
- [Feature set 2]: [description, source, dimensions]

The dataset comprised [N] samples, split into training ([X%]), validation
([X%]), and test ([X%]) sets using [splitting strategy, e.g., spatial
cross-validation with [N] folds to avoid spatial autocorrelation].

Model architecture: [brief description or reference to figure].
Loss function: [e.g., cross-entropy, focal loss, Dice loss].
Optimizer: [e.g., AdamW, lr=1e-4, weight_decay=1e-5].
Training: [N] epochs, batch size [N], early stopping patience [N].

Hyperparameters were tuned using [method, e.g., Optuna, grid search] over:
- [Param 1]: [range and selected value]
- [Param 2]: [range and selected value]

Performance was evaluated using [metrics: RMSE, F1, IoU, mAP] on the
held-out test set. All experiments were conducted on [hardware: GPU type,
RAM] with [random seed] for reproducibility.

Code is available at: [GitHub URL with DOI].
```

### GIS Web Application

```
The web mapping application was developed using [framework, e.g., MapLibre
GL JS 4.x / Leaflet 1.9] for the frontend and [backend, e.g., PostGIS 16
+ FastAPI / GeoServer 2.x] for spatial data services.

Geospatial data were stored in PostgreSQL/PostGIS with GiST spatial
indexing for efficient query performance. Vector tiles were generated
using [tool, e.g., Martin / Tippecanoe] at zoom levels [range].

The application implements [key features, e.g., spatial query, geoprocessing,
real-time updates] and was deployed using [infrastructure, e.g., Docker
on AWS / Kubernetes / Cloudflare].

User evaluation: [N] participants completed [tasks] with average task
completion time of [X] seconds and System Usability Scale score of [X/100].
```

### Study Area Description

```
The study area is located in [region/city/country] (approximately [lat]°N/S,
[lon]°E/W), covering an area of approximately [X] km² (Figure [N]).

[Physical geography: terrain type, elevation range (X-Y m a.s.l.), climate
classification (e.g., Cfa — humid subtropical), mean annual temperature (X°C),
mean annual precipitation (X mm), dominant land cover types.]

[Human geography: population (X million, YYYY census), urbanization rate (X%),
key land use characteristics, relevant socioeconomic context.]

This area was selected because [justification: representative of a broader
phenomenon, data availability, policy relevance, contrasting conditions
for comparative study, prior research baseline exists].
```

### Data Description Table

```markdown
| Dataset | Source | Format | Resolution | Period | Access |
|---------|--------|--------|------------|--------|--------|
| Sentinel-2 L2A | Copernicus | GeoTIFF | 10 m | Jun-Aug 2024 | https://dataspace.copernicus.eu |
| SRTM DEM | USGS | GeoTIFF | 30 m | 2000 | https://earthexplorer.usgs.gov |
| OpenStreetMap | Geofabrik | GeoPackage | N/A | 2024 | https://download.geofabrik.de |
| Census tracts | Census Bureau | Shapefile | N/A | 2020 | https://www.census.gov/geographies |
| ERA5 climate | ECMWF/CDS | NetCDF | 0.25° | 2020-2024 | https://cds.climate.copernicus.eu |
```

---

## LaTeX Templates

### Elsevier (RSE, C&G, JAG) — elsarticle.cls

```latex
\documentclass[preprint,12pt]{elsarticle}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{siunitx}
\usepackage{xcolor}
\usepackage{listings}

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
The data and code used in this study are available at
\url{https://doi.org/XX.XXXX/zenodo.XXXXXXX}.

\section*{CRediT Author Statement}
\textbf{First Author:} Conceptualization, Methodology, Software,
Writing -- Original Draft.
\textbf{Second Author:} Data Curation, Validation,
Writing -- Review \& Editing.
\textbf{Third Author:} Supervision, Funding Acquisition.

\section*{Declaration of Competing Interests}
The authors declare that they have no known competing financial
interests or personal relationships that could have influenced
the work reported in this paper.

\section*{Acknowledgments}
This work was supported by [Funder] (Grant No. [XXX]).
We thank [collaborators] for [contribution].

\bibliographystyle{elsarticle-harv}
\bibliography{references}
\end{document}
```

### IEEE (TGRS, JSTARS) — IEEEtran.cls

```latex
\documentclass[journal]{IEEEtran}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{xcolor}

\begin{document}
\title{Deep Learning for Building Extraction\\
from High-Resolution Remote Sensing Images}

\author{First Author,~\IEEEmembership{Student Member,~IEEE,}
  and Second Author,~\IEEEmembership{Senior Member,~IEEE}
\thanks{Manuscript received Month DD, YYYY.}
\thanks{F. Author is with Department of Geomatics,
  University, City (e-mail: first@uni.edu).}
\thanks{S. Author is with Institute of RS,
  University, City (e-mail: second@uni.edu).}}

\maketitle

\begin{abstract}
% 150-250 words for IEEE.
\end{abstract}

\begin{IEEEkeywords}
Building extraction, deep learning, remote sensing, semantic segmentation
\end{IEEEkeywords}

\section{Introduction}
\IEEEPARstart{T}{he} rapid growth of...

\section{Related Work}
\section{Proposed Method}
\subsection{Network Architecture}
\subsection{Loss Function}
\subsection{Training Strategy}
\section{Experiments}
\subsection{Datasets}
\subsection{Implementation Details}
\subsection{Evaluation Metrics}
\subsection{Results}
\subsection{Ablation Study}
\section{Discussion}
\section{Conclusion}

\section*{Data Availability}
Code and pretrained models are available at
\url{https://github.com/...}.

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

### Springer (Geoinformatica) — svjour3.cls

```latex
\documentclass[smallextended]{svjour3}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{hyperref}

\journalname{GeoInformatica}

\begin{document}
\title{Title of the Paper}
\author{First Author \and Second Author \and Third Author}
\institute{F. Author \at Department, University \\
  \email{first@uni.edu}
  \and S. Author \at Institute, University}

\date{Received: / Accepted: }

\maketitle

\begin{abstract}
Abstract text.
\keywords{GIS \and spatial databases \and query processing}
\end{abstract}

\section{Introduction}
\section{Related Work}
\section{Approach}
\section{Experiments}
\section{Discussion}
\section{Conclusion}

\begin{acknowledgements}
This work was supported by...
\end{acknowledgements}

\bibliographystyle{spmpsci}
\bibliography{references}
\end{document}
```

### Copernicus (ISPRS Archives, ESSD, GMD)

```latex
\documentclass[isprs]{copernicus}
\begin{document}
\title{Title of the Paper}

\Author[1]{First}{Author}
\Author[1]{Second}{Author}
\Author[2]{Third}{Author}

\affil[1]{Department of Geography, University, City, Country}
\affil[2]{Institute of Remote Sensing, City, Country}

\correspondence{First Author (first@university.edu)}

\received{}
\pubdiscuss{}
\revised{}
\accepted{}
\published{}

\firstpage{1}

\maketitle

\begin{abstract}
Abstract text here.
\end{abstract}

\introduction
\section{Study Area and Data}
\section{Methods}
\section{Results}
\section{Discussion}
\conclusions

\codedataavailability{
Code and data are available at \url{https://doi.org/...}.
}

\authorcontribution{FA designed the study...}
\competinginterests{The authors declare no competing interests.}

\begin{acknowledgements}
This work was supported by...
\end{acknowledgements}

\bibliographystyle{copernicus}
\bibliography{references}
\end{document}
```

---

## Typst Templates (Modern Alternative to LaTeX)

Typst is a modern typesetting system that compiles much faster than LaTeX, has simpler syntax, and supports collaborative editing at typst.app.

### Basic GIS Paper Template (Typst)

```typst
#set document(
  title: "Spatial Analysis of Urban Heat Islands",
  author: ("First Author", "Second Author"),
)
#set page(paper: "a4", margin: 2.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.1")

// Title block
#align(center)[
  #text(size: 16pt, weight: "bold")[
    Spatial Analysis of Urban Heat Islands Using \
    Machine Learning and Sentinel-2 Imagery
  ]
  #v(0.5em)
  First Author#super[1]#footnote[Corresponding: first\@uni.edu],
  Second Author#super[1],
  Third Author#super[2]
  #v(0.3em)
  #text(size: 9pt)[
    #super[1]Department of Geography, University \
    #super[2]Institute of Remote Sensing, University
  ]
]

// Abstract
#block(inset: (left: 2em, right: 2em))[
  *Abstract.* Urban heat islands pose significant health risks...
  150-300 words.

  *Keywords:* GIS, remote sensing, urban heat island, machine learning
]

= Introduction
= Study Area and Data
= Methodology
= Results
= Discussion
= Conclusions

// References
#bibliography("references.bib", style: "elsevier-harvard")
```

### Typst vs. LaTeX Comparison

| Feature | LaTeX | Typst |
|---------|-------|-------|
| Compilation speed | Seconds to minutes | Milliseconds |
| Syntax complexity | High (many packages) | Low (built-in features) |
| Error messages | Cryptic | Clear and helpful |
| Collaboration | Overleaf (paid for real-time) | typst.app (free real-time) |
| Journal templates | Extensive (all major publishers) | Growing (community) |
| Maturity | 40+ years | ~2 years (2023-) |
| Package ecosystem | CTAN (thousands) | Typst Universe (growing) |
| **Recommendation** | Required by most journals | Great for drafts, some journals |

### When to Use Typst vs. LaTeX

```
Is the target journal LaTeX-only?
  YES → Use LaTeX with journal template
  NO  → Do you need complex features (TikZ, custom packages)?
          YES → Use LaTeX
          NO  → Is this a draft or internal document?
                  YES → Typst (faster iteration)
                  NO  → LaTeX (wider acceptance), export PDF then
```

---

## AI Writing Assistants for Academic Papers

### AI Tool Comparison (2025)

| Tool | Best For | Cost | Key Feature |
|------|----------|------|-------------|
| Claude (Anthropic) | Long-form writing, analysis | Free–$20/mo | Large context window, nuanced reasoning |
| ChatGPT (OpenAI) | General drafting, brainstorming | Free–$20/mo | Widely adopted, plugins |
| Grammarly | Grammar, clarity, style | Free–$30/mo | Real-time editing, tone detection |
| Writefull | Academic-specific language | Free–$16/mo | Trained on academic papers, journal matching |
| Paperpal | Academic writing | Free–$15/mo | From Cactus (editing company) |
| QuillBot | Paraphrasing, summarizing | Free–$20/mo | Sentence-level rewriting |
| Jenni AI | Academic writing assistant | $12/mo | In-text citations, outline generation |
| SciSpace (Typeset) | Paper reading + writing | Free–$10/mo | Read papers + draft papers |

### Ethical Use of AI in Academic Writing

**Generally Accepted Uses:**
- Brainstorming and outlining
- Grammar and language polishing (especially for non-native speakers)
- Paraphrasing your own text for clarity
- Generating code for data analysis
- Literature search and summarization (verify all citations!)
- Formatting and style consistency

**Generally Not Accepted:**
- Generating entire sections without substantial rewriting
- Fabricating data or results
- Using AI-generated citations without verification
- Submitting AI-generated text as original work without disclosure

**Disclosure Requirements (2025):**
- Most journals now require disclosure of AI tool usage
- Elsevier, Springer Nature, IEEE, T&F all have AI use policies
- AI cannot be listed as an author
- Authors retain full responsibility for content
- Common disclosure: "AI tools were used for language editing / code generation"

### Effective Prompts for GIS Paper Writing

```
# Writing a literature review paragraph
"Summarize the key methods for urban heat island detection from
satellite imagery published 2020-2024. Focus on: (1) thermal
downscaling approaches, (2) machine learning methods, (3) multi-sensor
fusion. For each, note strengths and limitations."

# Improving a methodology description
"Rewrite this methodology paragraph to be more precise and reproducible.
Add specific parameter values, software versions, and justifications:
[paste your draft paragraph]"

# Generating a response to reviewer
"A reviewer says: '[paste reviewer comment]'. My actual response is:
[paste what you did]. Draft a polite, detailed response addressing
their concern, referencing specific changes I made."
```

---

## Figure and Table Formatting

### General Requirements

| Element | Requirement |
|---------|------------|
| Resolution | 300 DPI minimum for raster; vector (PDF/SVG/EPS) preferred |
| Width | Single column (~8.5 cm) or double column (~17.5 cm) |
| Font size | Labels readable at print size (8-10 pt minimum) |
| Color | Ensure readability in grayscale if journal prints in B&W |
| File formats | PDF, EPS, or high-resolution PNG/TIFF |
| Captions | Self-contained: understandable without reading the text |

### Map Figure Checklist

- [ ] Title or caption clearly describes content
- [ ] Legend explains all symbology
- [ ] Scale bar with appropriate units
- [ ] North arrow (if orientation is not obvious)
- [ ] CRS noted in caption or margin
- [ ] Data source attribution
- [ ] Inset/overview map showing study area context
- [ ] Colorblind-safe palette
- [ ] Text labels readable at print size
- [ ] No unnecessary white space

### Publication Figure Generation (Python)

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
import contextily as ctx

# Publication-quality settings
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 10,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Single column figure (8.5 cm = 3.35 in)
fig, ax = plt.subplots(figsize=(3.35, 3.0))

# Double column figure (17.5 cm = 6.89 in)
fig, ax = plt.subplots(figsize=(6.89, 4.0))

# Map with all required elements
gdf = gpd.read_file("study_area.gpkg")
gdf.plot(ax=ax, column='temperature', cmap='RdYlBu_r',
         legend=True, legend_kwds={'label': 'LST (°C)', 'shrink': 0.7})
ctx.add_basemap(ax, crs=gdf.crs, source=ctx.providers.CartoDB.Positron)
ax.add_artist(ScaleBar(1000, location='lower left'))
ax.set_title('')  # Use caption instead of title in-figure

fig.savefig('figure_1.pdf', format='pdf')  # Vector format preferred
fig.savefig('figure_1.tiff', format='tiff', dpi=300)  # Raster backup
```

### Table Formatting Rules

- Use horizontal rules only (no vertical rules in most journals)
- Use `\toprule`, `\midrule`, `\bottomrule` from `booktabs` package
- Align decimal points in numeric columns
- Include units in column headers, not in cells
- Bold or highlight key results (best values)
- Reference every table from the text

```latex
\begin{table}[t]
\centering
\caption{Comparison of classification methods on the study area dataset.
Best results in bold.}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
Method & OA (\%) & Kappa & F1 \\
\midrule
Random Forest    & 87.3 & 0.82 & 0.85 \\
SVM (RBF)        & 85.1 & 0.79 & 0.83 \\
U-Net            & \textbf{92.1} & \textbf{0.89} & \textbf{0.91} \\
DeepLab v3+      & 91.4 & 0.88 & 0.90 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## Reference Management

### Tool Comparison

| Tool | Cost | Best Feature | LaTeX Support | Collaboration |
|------|------|-------------|---------------|---------------|
| Zotero | Free | Browser connector, groups, open source | Excellent (Better BibTeX) | Group libraries |
| Mendeley | Free | PDF annotation, social features | Good | Groups |
| Paperpile | $2.99/mo | Google Docs integration | Good | Shared folders |
| EndNote | $249.95 | Largest style library | Good | Via library sharing |
| JabRef | Free | BibTeX-native, open source | Native | Via Git |
| Zotero + Notero | Free | Notion integration | Via export | Notion workspace |

### Zotero + Better BibTeX Setup

1. Install Zotero (https://www.zotero.org)
2. Install Better BibTeX plugin (https://retorque.re/zotero-better-bibtex/)
3. Install Zotero Connector for your browser
4. Configure citation key format: `auth.lower + year` (e.g., `smith2024`)
5. Set up auto-export of `.bib` file to your LaTeX project directory
6. Organize collections by project or paper section

### Essential Zotero Plugins (2025)

| Plugin | Purpose |
|--------|---------|
| Better BibTeX | Auto-generate citation keys, export .bib |
| Zotero PDF Translate | Translate papers in-app |
| Zotero GPT / Aria | AI-powered paper analysis |
| Zotero Connector | Save papers from browser |
| ZotMoov | Auto-move and rename attachments |
| Zotero Citation Counts | Show citation counts in library |
| Better Notes | Enhanced note-taking with templates |
| Zotero Style | Custom column styles and badges |

### Recommended Collection Structure

```
My GIS Research/
  ├── Background & Theory/
  ├── Study Area Context/
  ├── Methodology/
  │   ├── Spatial Analysis Methods/
  │   ├── Machine Learning/
  │   └── Remote Sensing/
  ├── Data Sources/
  ├── Comparison & Baselines/
  ├── Results Discussion/
  ├── Software & Tools/
  └── To Read/
```

---

## CRediT Author Contributions

The **Contributor Roles Taxonomy (CRediT)** is now required or encouraged by most GIS journals.

### CRediT Roles

| Role | Description |
|------|------------|
| Conceptualization | Ideas, formulation of research goals |
| Methodology | Development of methodology, models |
| Software | Programming, software development |
| Validation | Verification, accuracy assessment |
| Formal Analysis | Statistical analysis, data analysis |
| Investigation | Data collection, fieldwork |
| Resources | Computing, equipment, materials |
| Data Curation | Data management, annotation |
| Writing — Original Draft | Initial manuscript writing |
| Writing — Review & Editing | Critical review, revision |
| Visualization | Maps, figures, data presentation |
| Supervision | Oversight, mentoring |
| Project Administration | Management, coordination |
| Funding Acquisition | Securing financial support |

### Example CRediT Statement

```
Author Contributions:
Conceptualization: F.A. and S.A.; Methodology: F.A.; Software: F.A.;
Validation: F.A. and T.A.; Formal Analysis: F.A.; Investigation: F.A.
and T.A.; Data Curation: T.A.; Writing — Original Draft: F.A.;
Writing — Review & Editing: S.A. and T.A.; Visualization: F.A.;
Supervision: S.A.; Project Administration: S.A.;
Funding Acquisition: S.A.
```

---

## Writing Process Workflow

### Recommended Order of Writing

```
1. Figures and Tables (create before writing — they tell the story)
   │
2. Methods (most straightforward to write)
   │
3. Results (describe what figures/tables show)
   │
4. Discussion (interpret results, compare with literature)
   │
5. Introduction (now you know exactly what to introduce)
   │
6. Conclusion (summarize the story)
   │
7. Abstract (distill everything into 200-300 words)
   │
8. Title (make it specific and informative)
```

### Common Writing Mistakes in GIS Papers

| Mistake | Fix |
|---------|-----|
| Vague study area description | Include coordinates, area, CRS, map |
| Missing software versions | Always state exact versions |
| No error assessment | Include accuracy metrics, confidence intervals |
| Overclaiming in conclusions | Match conclusions to evidence |
| Chronological literature review | Organize by theme/method |
| Results + Discussion merged poorly | Keep Results neutral; interpret in Discussion |
| No data availability statement | Always include, even if "data available on request" |
| Inconsistent terminology | Define terms at first use, use consistently |
| Missing CRS information | State EPSG code in methods and figure captions |
| No workflow diagram | Create a methods flowchart |

### Overleaf Templates Quick Links

| Template | URL |
|----------|-----|
| Elsevier (elsarticle) | https://www.overleaf.com/latex/templates/elsevier-article/cdhrznpfvhnm |
| Springer Nature | https://www.overleaf.com/latex/templates/springer-nature-latex-template/gsvvftmrppwq |
| IEEE Conference | https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhkpxfmc |
| IEEE Journal | https://www.overleaf.com/latex/templates/ieee-photonics-journal-paper-template/bsfjjdkshfmb |
| ISPRS Archives | https://www.overleaf.com/latex/templates/isprs-annals-and-archives/tdbqjhgdswby |
| Copernicus (ESSD, GMD) | https://publications.copernicus.org/for_authors/manuscript_preparation.html |
| ACM SIGSPATIAL | https://www.overleaf.com/latex/templates/acm-conference-proceedings-primary-article-template/wbvnghjbzwpc |
