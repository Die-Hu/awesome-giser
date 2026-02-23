# Academic Writing Prompts

> Prompt templates for literature reviews, methodology sections, results descriptions, figure captions, and responding to peer reviewers in GIS-related research.

> **Quick Picks**
> - 🏆 **Most Useful**: [Full Methodology Draft](#prompt-1--full-methodology-draft) — produces a journal-ready methods section in one pass
> - 🚀 **Time Saver**: [Convert Methodology Notes to Formal Section](#prompt-3--convert-methodology-notes-to-formal-section) — turns messy bullet points into polished academic prose
> - 🆕 **Cutting Edge**: [LaTeX Table from Spatial Analysis Results](#prompt-3--latex-table-from-spatial-analysis-results) — generates publication-quality tables directly from stats output

---

## Table of Contents

- [Literature Review](#literature-review)
- [Methodology Section](#methodology-section)
- [Results Description](#results-description)
- [Figure Captions](#figure-captions)
- [Study Area Maps](#study-area-maps)
- [Response to Reviewers](#response-to-reviewers)

---

## Literature Review

### Prompt 1 — Thematic Literature Summary

**Context:** You are writing the background section of a paper and need to summarize the state of the art in a specific GIS methodology or application area.

**Template:**

```
Summarize the key methodologies for [GIS topic, e.g., "urban heat island mapping using remote sensing"]
published between [start year] and [end year].

Structure the summary as:
1. Evolution of approaches (earliest to most recent)
2. Data sources commonly used (satellite sensors, ground stations, etc.)
3. Methods and algorithms (group by category: statistical, ML, physics-based)
4. Spatial and temporal scales studied
5. Key findings and consensus in the field
6. Identified research gaps and open questions
7. How my proposed approach ([brief description]) fits into the landscape

Style: academic, third person, suitable for a journal article in [target journal, e.g., Remote Sensing of Environment].
Length: approximately [N] words.
Do NOT fabricate references — instead, indicate where citations should go with [CITE] placeholders.
```

**Variables to customize:**
- GIS topic
- Date range
- Your proposed approach
- Target journal and word count

**Expected output format:** A structured literature review draft with `[CITE]` placeholders where references should be inserted.

---

### Prompt 2 — Compare Methods in a Table

**Context:** You want a concise comparison table of existing methods for your literature review or methodology justification.

**Template:**

```
Create an academic comparison table of methods for [task, e.g., "land use / land cover classification"].

Columns:
| Method | Data Requirements | Accuracy Range | Computational Cost | Strengths | Limitations | Representative Studies [CITE] |

Include these methods:
1. [Method A, e.g., Maximum Likelihood Classification]
2. [Method B, e.g., Random Forest]
3. [Method C, e.g., Support Vector Machine]
4. [Method D, e.g., U-Net / Deep Learning]
5. [Method E, e.g., Object-Based Image Analysis (OBIA)]

Fill in realistic values based on general knowledge of the field.
Mark any uncertain entries with [VERIFY].
Use academic tone suitable for a peer-reviewed journal.
```

**Variables to customize:**
- Task/application
- Methods to compare
- Table columns

**Expected output format:** A Markdown table ready for inclusion in a manuscript (may need conversion to LaTeX).

---

### Prompt 3 — Identify Research Gaps

**Context:** You have read a body of literature and need help articulating what is missing.

**Template:**

```
Based on the following summary of existing research on [topic]:

[Paste your notes or a brief paragraph summarizing what has been done]

Help me identify and articulate:
1. Methodological gaps (what techniques haven't been tried or combined?)
2. Spatial gaps (what regions or scales are understudied?)
3. Temporal gaps (what time periods or temporal resolutions are missing?)
4. Data gaps (what data sources could add value but haven't been used?)
5. Application gaps (what practical applications remain unexplored?)

For each gap, write 1-2 sentences in academic style that could serve as justification for my study.
Frame them as motivations, not criticisms of prior work.
```

**Variables to customize:**
- Topic
- Summary of existing research
- Your study's focus

**Expected output format:** Numbered list of gaps with academically phrased justification sentences.

---

## Methodology Section

### Prompt 1 — Full Methodology Draft

**Context:** You need to write the methodology section of a GIS research paper.

**Template:**

```
Write a methodology section for a study using [method, e.g., "geographically weighted regression (GWR)"]
to analyze [data, e.g., "the spatial relationship between PM2.5 concentration and land use variables"].

Study details:
- Study area: [name, extent, why chosen]
- Time period: [dates]
- Dependent variable: [name, source, resolution, unit]
- Independent variables: [list with sources and resolutions]
- Sample size / spatial units: [N, type]
- Software: [e.g., R with GWmodel package / Python with mgwr]
- CRS: [EPSG code]

Section structure:
1. Study area description (1 paragraph)
2. Data acquisition and preprocessing (1-2 paragraphs per data source)
3. Analytical method description with mathematical formulation
4. Model specification and variable selection procedure
5. Validation approach (cross-validation, AICc comparison, residual analysis)
6. Software and computational environment

Style: passive voice, past tense, suitable for [journal].
Include the key equation(s) in LaTeX math notation.
Use [CITE] placeholders for methodological references.
```

**Variables to customize:**
- Method and application
- Study details (area, variables, software)
- Target journal
- Level of mathematical detail

**Expected output format:** Methodology section draft (1,500-2,500 words) with equations in LaTeX and `[CITE]` placeholders.

---

### Prompt 2 — Describe a Processing Workflow

**Context:** You need to describe a multi-step data processing pipeline clearly enough for reproducibility.

**Template:**

```
Write a clear, reproducible description of the following GIS processing workflow for the methodology section:

Steps:
1. [Step: e.g., "Download Sentinel-2 L2A imagery from Copernicus Data Space for June-August 2024"]
2. [Step: e.g., "Apply cloud masking using SCL band, discard scenes with >20% cloud cover"]
3. [Step: e.g., "Calculate monthly median composites"]
4. [Step: e.g., "Compute NDVI, NDWI, NDBI indices"]
5. [Step: e.g., "Perform supervised classification using Random Forest (500 trees)"]
6. [Step: e.g., "Validate using stratified random sample of 500 points"]

For each step, include:
- Input and output data
- Tools/functions used (with version numbers where applicable)
- Key parameter values
- Justification for choices (e.g., why median composite instead of mean)

Write in academic past tense. Keep it concise but complete enough for replication.
```

**Variables to customize:**
- Workflow steps with parameters
- Tools and versions
- Justification details

**Expected output format:** A methodical workflow description suitable for a journal paper.

---

### Prompt 3 — Convert Methodology Notes to Formal Section

**Context:** You have rough notes, bullet points, or a stream-of-consciousness description of what you did in your analysis, and you need to convert them into a properly structured, journal-ready methodology section.

**Template:**

```
Convert the following rough methodology notes into a formal methods section suitable for
submission to [journal name, e.g., "International Journal of Geographical Information Science"].

My rough notes:
"""
[Paste your notes here — these can be messy bullet points, lab notebook entries,
code comments, or informal descriptions. Example:
- downloaded sentinel2 images from copernicus for summer 2023 and 2024
- used sen2cor for atmospheric correction but some were already L2A
- made NDVI for each date, used the SCL band to mask clouds
- did RF classification with 6 classes, used 500 trees
- training data from field survey (GPS points, 200 total) plus some from google earth
- accuracy was 87.3% overall, kappa 0.84
- water class was best (98%), urban-rural mix was worst (72%)
- used QGIS and python with sklearn and rasterio]
"""

Transform these notes into formal academic prose following these rules:
1. Structure: Study Area -> Data Acquisition -> Preprocessing -> Analysis Method -> Validation
2. Voice: passive, past tense ("Sentinel-2 Level-2A imagery was acquired...")
3. Be specific: add detail I probably know but didn't write down:
   - Full sensor name and spatial resolution
   - Band designations used for each index
   - Software versions (ask me to verify if uncertain — use [VERSION?] placeholder)
   - Mathematical formulations for any indices or algorithms
4. Add [CITE] placeholders where methodological references are needed
5. Add [VERIFY] flags where you are inferring details I didn't explicitly state
6. Length: [N] words, or scale proportionally to the detail in my notes
7. Include proper technical terminology (do not simplify for a general audience)
8. End with a paragraph on reproducibility: software, hardware, code availability

Do NOT fabricate specific numbers or results. If I didn't mention a value, use [?] placeholder.
```

**Variables to customize:**
- Your rough notes (the main input)
- Target journal
- Desired length
- Level of detail to infer vs flag

**Expected output format:** Formal methodology section with [CITE], [VERIFY], and [?] placeholders.

---

## Results Description

### Prompt 1 — Describe Spatial Patterns

**Context:** You have analysis results (maps, statistics) and need to describe the spatial patterns in academic prose.

**Template:**

```
Describe the spatial patterns shown in [analysis results, e.g., "a LISA cluster map of median income
across 500 census tracts in [city]"].

Results data:
- High-High clusters: [N] tracts, primarily in [locations]
- Low-Low clusters: [N] tracts, primarily in [locations]
- High-Low outliers: [N] tracts
- Low-High outliers: [N] tracts
- Not significant: [N] tracts
- Global Moran's I: [value], p-value: [value]

Write 2-3 paragraphs that:
1. Report the global statistic and its significance
2. Describe the spatial distribution of clusters (use geographic directions: northern, central, etc.)
3. Highlight notable patterns (e.g., clusters aligned with specific features like rivers, highways)
4. Compare to expected patterns based on prior literature [CITE]
5. Note any surprising or anomalous findings

Tone: objective, academic, third person past tense.
Avoid interpreting causation — save that for the Discussion section.
```

**Variables to customize:**
- Analysis type and results data
- Geographic context
- Notable patterns to highlight

**Expected output format:** 2-3 paragraphs of results description with `[CITE]` placeholders.

---

### Prompt 2 — Accuracy Assessment Reporting

**Context:** You ran a classification or model and need to report accuracy metrics properly.

**Template:**

```
Write the accuracy assessment subsection of a Results section based on these metrics:

Classification type: [e.g., land cover classification, 6 classes]
Validation method: [e.g., stratified random sampling, 500 points, independent from training data]

Confusion matrix:
[paste matrix or describe]

Derived metrics:
- Overall accuracy: [value]%
- Kappa coefficient: [value]
- Per-class Producer's accuracy: [list]
- Per-class User's accuracy: [list]
- F1 scores: [list]

Write the subsection to:
1. State the validation approach and sample size
2. Report overall accuracy and kappa with interpretation (Landis & Koch scale)
3. Present the confusion matrix reference (as Table [N])
4. Discuss which classes performed well and which had confusion
5. Identify the most common misclassification pairs and suggest reasons
6. Compare accuracy to similar studies [CITE]

Keep it concise — 1.5 to 2 paragraphs plus a reference to the confusion matrix table.
```

**Variables to customize:**
- Classification type and class count
- Validation method
- Metrics values
- Confusion pairs of interest

**Expected output format:** Accuracy assessment paragraphs plus a suggested table caption.

---

## Figure Captions

### Prompt 1 — Map Figure Caption

**Context:** You need a descriptive caption for a map figure in a journal manuscript.

**Template:**

```
Write a figure caption for a [map type, e.g., "choropleth map"] showing [variable, e.g., "NDVI change
from 2010 to 2023"] in [study area].

Figure details:
- Map extent: [description]
- Classification: [N] classes using [method, e.g., natural breaks]
- Color scheme: [description, e.g., diverging red-white-green]
- Inset/overview map: [yes/no]
- Scale bar and north arrow: [included]
- Projection: [name]
- Data source: [source and date]
- Sub-figures: [a, b, c descriptions if applicable]

Caption requirements:
- Start with "Figure [N]." or "Fig. [N]."
- First sentence: describe what the map shows
- Second sentence: note the classification method and color scheme
- Third sentence: highlight the key pattern visible
- Final sentence: data source and date (if journal requires it in caption)
- Length: 2-4 sentences, under 75 words
```

**Variables to customize:**
- Map type and variable
- Figure details
- Journal caption format requirements

**Expected output format:** A single figure caption paragraph.

---

### Prompt 2 — Multi-Panel Figure Caption

**Context:** You have a composite figure with multiple panels (sub-figures a, b, c, etc.).

**Template:**

```
Write a caption for a multi-panel figure:

- (a): [description, e.g., "True-color Sentinel-2 composite of the study area, acquired 2024-07-15"]
- (b): [description, e.g., "NDVI map derived from the same image, classified into 5 categories"]
- (c): [description, e.g., "Land cover classification result showing 6 classes"]
- (d): [description, e.g., "Accuracy assessment confusion matrix"]

Overall figure purpose: [e.g., "showing the complete workflow from raw imagery to classified output"]

Follow [journal, e.g., ISPRS Journal] caption style:
- Main caption describing the overall figure
- Sub-captions in parentheses for each panel
- Include data source
```

**Variables to customize:**
- Panel descriptions
- Overall purpose
- Journal style

**Expected output format:** A structured figure caption with main text and sub-panel descriptions.

---

### Prompt 3 — LaTeX Table from Spatial Analysis Results

**Context:** You have spatial analysis results (accuracy metrics, regression coefficients, zonal statistics, etc.) and need to format them as a publication-quality LaTeX table for a journal manuscript.

**Template:**

```
Generate a LaTeX table from the following spatial analysis results:

Table type: [accuracy assessment / regression coefficients / zonal statistics / method comparison / descriptive statistics]

Data:
[Paste your results in any format — CSV, plain text, Python dict, JSON, console output. Example:
class,PA,UA,F1
Water,0.98,0.97,0.975
Forest,0.91,0.89,0.900
Cropland,0.85,0.88,0.865
Urban,0.82,0.79,0.805
Bare Soil,0.78,0.81,0.795]

LaTeX requirements:
- Package: booktabs (use \toprule, \midrule, \bottomrule — no vertical lines)
- Table environment: table with [placement, e.g., htbp]
- Caption: "[compose a descriptive caption based on the data]"
- Label: \label{tab:[table_id]}
- Column formatting:
  - Text columns: left-aligned (l)
  - Numeric columns: right-aligned with consistent decimal places (use siunitx S column type if appropriate)
  - Highlight the best value per column in bold
- Add a table note below (using \footnotesize or threeparttable) explaining abbreviations:
  PA = Producer's Accuracy, UA = User's Accuracy, etc.
- If the table is wide, use \resizebox{\textwidth}{!}{...} or rotate to landscape
- If the table has many rows, suggest splitting into sub-tables or using longtable

Also provide:
1. The raw LaTeX code ready for copy-paste into a .tex file
2. A Markdown version of the same table (for the paper draft in Markdown)
3. A Python snippet using pandas DataFrame.to_latex() that generates the same table programmatically,
   so I can regenerate it when results change

For regression results, include significance stars: * p<0.05, ** p<0.01, *** p<0.001
```

**Variables to customize:**
- Table type and data
- LaTeX packages available
- Caption and label
- Column formatting preferences
- Significance notation (if applicable)

**Expected output format:** LaTeX table code, Markdown equivalent, and a pandas-to-LaTeX Python snippet.

---

## Study Area Maps

### Prompt 1 — Study Area Map with Cartographic Elements

**Context:** You need to create a study area map (often Figure 1) for a journal paper with proper cartographic elements: overview inset, scale bar, north arrow, labels, and clean styling.

**Template:**

```
Write a Python script to create a publication-quality study area map using
[matplotlib + cartopy (0.22+) / GeoPandas (0.14+) plot / contextily (1.5+) for basemap tiles].

Study area details:
- Boundary file: [path to shapefile/GeoPackage of study area boundary]
- CRS: [EPSG code]
- Location: [description, e.g., "Chengdu, Sichuan Province, China"]
- Context: [what is studied here, for the caption]

Map elements:
1. Main map panel:
   - Study area boundary: filled semi-transparent ([color, e.g., light blue, alpha=0.3])
     with solid border ([color], linewidth [width])
   - Basemap: [option: contextily OpenStreetMap tiles / natural earth features / none]
   - Key features labeled: [list, e.g., "major rivers", "district names", "city center"]
   - CRS for display: [projected CRS for the main map, e.g., UTM]

2. Overview inset map (top-right or bottom-left corner):
   - Show the country/region with the study area highlighted as a small red rectangle/polygon
   - Use Natural Earth 1:110m country boundaries (via cartopy or a shapefile)
   - Scale: much smaller than main map, just for geographic context
   - Minimal styling: country outlines, study area marker

3. Scale bar:
   - Position: [bottom-left / bottom-right]
   - Length: [auto-calculated appropriate round number, e.g., 5 km / 10 km / 50 km]
   - Style: alternating black-and-white boxes (cartographic standard)

4. North arrow:
   - Position: [top-right / top-left]
   - Style: simple arrow with "N" label

5. Labels:
   - Feature labels from [column name] in [font, e.g., "DejaVu Sans", size 8]
   - Title (optional): "[study area name]"
   - Coordinate grid or graticule with labels on axes

6. Legend (if applicable):
   - Items: [list, e.g., study boundary, sampling points, rivers]

Export settings:
- DPI: [300 for print / 150 for draft]
- Format: [PDF (vector) / PNG / TIFF]
- Size: [width x height inches, e.g., 7 x 5 for single-column, 3.5 x 3 for half-column]
- Background: white

The output should be camera-ready for journal submission with no manual editing needed.
```

**Variables to customize:**
- Boundary file path and CRS
- Location and geographic context
- Basemap source
- Map elements to include
- Export format and DPI

**Expected output format:** Python script producing a publication-ready study area map as PDF/PNG.

---

## Response to Reviewers

### Prompt 1 — Draft Response to a Specific Comment

**Context:** You received a peer review and need to draft a professional response to a reviewer's comment.

**Template:**

```
Draft a response to this reviewer comment from a peer-reviewed GIS journal:

Reviewer comment:
"[Paste the reviewer's comment exactly]"

Context about my paper:
- Topic: [brief topic]
- The reviewer is questioning: [your interpretation of what they are asking]
- I [agree / partially agree / disagree] with this comment

If I agree:
- Changes I plan to make: [describe revisions]
- New analysis or content to add: [if any]

If I partially agree or disagree:
- My reasoning: [explain why]
- Evidence supporting my position: [cite data, literature]
- Compromise I am willing to make: [if any]

Write the response in a professional, respectful, constructive tone.
Structure:
1. Thank the reviewer for the observation
2. Address the specific point (agree, explain, or justify with evidence)
3. State exactly what was changed in the manuscript (with line/section references as "[Section X]")
4. If a new analysis was added, briefly summarize the result

Length: [brief / moderate / detailed] — match the complexity of the comment.
```

**Variables to customize:**
- Reviewer comment text
- Your paper topic and position
- Planned revisions

**Expected output format:** A professional reviewer response paragraph.

---

### Prompt 2 — Systematic Response Letter

**Context:** You have multiple reviewer comments and need to produce a complete response document.

**Template:**

```
I received reviews from [N] reviewers for my paper "[paper title]" submitted to [journal].

Here are all the comments:

Reviewer 1:
1. "[Comment 1]"
2. "[Comment 2]"
[...]

Reviewer 2:
1. "[Comment 1]"
2. "[Comment 2]"
[...]

Editor:
1. "[Comment 1]"
[...]

For each comment, I will provide my action: [AGREE: description of change] or [REBUT: reasoning].

Generate a complete response letter with:
- Opening paragraph thanking the editor and reviewers
- Comments organized by reviewer, each with:
  - The original comment (quoted in italics)
  - Our response (starting with "Response:")
  - Changes made (starting with "Changes:" with manuscript locations)
- Closing paragraph summarizing key improvements
- Professional, grateful tone throughout — even for comments I disagree with

Format: Markdown suitable for conversion to PDF (use headers, bold, italics).
```

**Variables to customize:**
- Reviewer comments
- Your actions (agree/rebut) for each
- Journal and paper title

**Expected output format:** A complete response-to-reviewers letter in Markdown.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
