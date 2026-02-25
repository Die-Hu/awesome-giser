# Academic Writing Prompts

> Expert prompts for literature reviews, methodology sections, results descriptions, figures, peer review responses, and Chinese academic writing.

> **Quick Picks**
> - **Most Useful**: [Full Methodology Draft](#prompt-5--full-methodology-draft) -- produces a journal-ready methods section in one pass
> - **Time Saver**: [Notes-to-Formal Conversion](#prompt-7--notes-to-formal-conversion) -- turns messy bullet points into polished academic prose
> - **New**: [Chinese Academic Writing](#6-chinese-academic-writing-p20-p22) -- bilingual prompts for Chinese journal submissions

---

## Table of Contents

1. [Literature Review (P1--P4)](#1-literature-review-p1-p4)
2. [Methodology Section (P5--P8)](#2-methodology-section-p5-p8)
3. [Results Description (P9--P12)](#3-results-description-p9-p12)
4. [Figures & Tables (P13--P16)](#4-figures--tables-p13-p16)
5. [Peer Review Response (P17--P19)](#5-peer-review-response-p17-p19)
6. [Chinese Academic Writing (P20--P22)](#6-chinese-academic-writing-p20-p22)

---

## 1. Literature Review (P1--P4)

---

### Prompt 1 -- Thematic Literature Summary

#### Scenario (实际场景)

You are drafting the background section of a GIS research paper and need a structured summary of the state of the art. The review must trace the evolution of approaches, compare data sources and methods, and identify gaps -- all with citation placeholders so you can fill in real references later.

#### Roles

You are an academic literature reviewer specializing in geospatial science, writing for a top-tier remote sensing or GIS journal.

#### Prompt Template

```text
You are an expert GIS literature reviewer. Summarize the key
methodologies for [TOPIC, e.g., "urban heat island mapping using
remote sensing"] published between [START_YEAR] and [END_YEAR].

Structure the summary as follows:
1. Evolution of approaches (earliest to most recent paradigms)
2. Data sources commonly used (satellite sensors, ground stations,
   UAV, crowdsourced) with spatial/temporal resolutions
3. Methods and algorithms grouped by category:
   - Statistical / geostatistical
   - Machine learning / deep learning
   - Physics-based / process models
4. Spatial and temporal scales studied
5. Key findings and emerging consensus
6. Research gaps and open questions
7. How my proposed approach ([BRIEF_DESCRIPTION]) fits into
   this landscape and what it adds

Target journal: [JOURNAL, e.g., Remote Sensing of Environment].
Style: academic third person, ~[WORD_COUNT] words.
Do NOT fabricate references. Use [CITE] where a citation is needed.
Use [VERIFY] if a claim needs fact-checking.
```

#### Variables to Customize

- `TOPIC` -- the GIS methodology or application domain
- `START_YEAR` / `END_YEAR` -- temporal scope of the review
- `BRIEF_DESCRIPTION` -- one-sentence summary of your proposed approach
- `JOURNAL` -- target journal (RSE, IJGIS, CAGEO, ISPRS JPRS, etc.)
- `WORD_COUNT` -- desired length (typically 800--1500 words)

#### Expected Output

A structured literature review draft with `[CITE]` placeholders and `[VERIFY]` flags. Ready for you to insert real references from Zotero, Mendeley, or a `.bib` file.

#### Validation Checklist

- [ ] Every factual claim has a `[CITE]` or `[VERIFY]` placeholder
- [ ] Evolution section covers at least three distinct methodological eras
- [ ] Research gaps are framed as motivations, not criticisms
- [ ] Tone matches the target journal
- [ ] No fabricated author names, years, or DOIs

#### Cost Optimization

Use a smaller model (GPT-4o-mini, Claude Haiku) for the initial outline, then switch to a frontier model for polishing prose and checking logical flow. This cuts cost by ~60%.

> **Dark Arts Tip:** Feed the model 5--10 actual abstracts from your Zotero collection as context. It will mirror the terminology and framing of your sub-field far more accurately than working from general knowledge alone.

#### Related Pages

- [Data Sources](../data-sources/) -- satellite and vector data catalogs referenced in reviews
- [Tools: Python Libraries](../tools/python-libraries.md) -- libraries mentioned in methods comparisons

#### Extensibility

- Chain with **P3 (Research Gap Identification)** to deepen the gap analysis.
- Feed the output into **P4 (Theoretical Framework)** to build a conceptual diagram.
- Pair with a citation manager API (Semantic Scholar, OpenAlex) to auto-fill `[CITE]` placeholders.

---

### Prompt 2 -- Methods Comparison Table

#### Scenario (实际场景)

You need a concise comparison table of competing methods for your literature review or to justify why you chose a particular approach. The table must be rigorous enough for peer review, with columns covering data requirements, accuracy, cost, and trade-offs.

#### Roles

You are a GIS methodologist preparing a systematic comparison for a journal manuscript.

#### Prompt Template

```text
Create an academic comparison table of methods for
[TASK, e.g., "land use / land cover classification from
satellite imagery"].

Columns:
| Method | Data Requirements | Typical Accuracy Range |
| Computational Cost | Strengths | Limitations |
| Representative Studies [CITE] |

Include these methods:
1. [METHOD_A, e.g., Maximum Likelihood Classification]
2. [METHOD_B, e.g., Random Forest]
3. [METHOD_C, e.g., Support Vector Machine]
4. [METHOD_D, e.g., U-Net / Deep Learning]
5. [METHOD_E, e.g., Object-Based Image Analysis (OBIA)]

For each cell:
- Use realistic ranges from the literature (not fabricated numbers)
- Mark uncertain values with [VERIFY]
- Representative Studies column: use [CITE] placeholders
- Computational cost: Low / Medium / High / Very High

Also provide:
1. A brief paragraph (3-4 sentences) summarizing which method
   is most suitable for [MY_CONTEXT]
2. The same table in LaTeX (booktabs, no vertical lines)
```

#### Variables to Customize

- `TASK` -- the GIS application or problem
- `METHOD_A` through `METHOD_E` -- methods to compare (add or remove as needed)
- `MY_CONTEXT` -- your specific study constraints (data availability, scale, budget)

#### Expected Output

A Markdown table plus a LaTeX version (`booktabs` style), with a summary paragraph recommending the best fit for your context.

#### Validation Checklist

- [ ] Accuracy ranges are plausible for the domain (not 99% for everything)
- [ ] Every "Representative Studies" cell has a `[CITE]` placeholder
- [ ] Computational cost reflects current hardware norms
- [ ] LaTeX output compiles without errors

#### Cost Optimization

Tables are token-light. A mid-tier model handles this well. Reserve frontier models for the summary paragraph.

> **Dark Arts Tip:** Ask the model to also generate a `pandas` DataFrame initialization snippet so you can programmatically regenerate the table when you add new methods or update accuracy numbers.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- implementations of each method
- [Tools: CLI Tools](../tools/cli-tools.md) -- command-line tools for benchmarking

#### Extensibility

- Expand to a meta-analysis by adding columns for sample size, study area size, and sensor type.
- Feed into **P1 (Thematic Literature Summary)** as structured evidence.

---

### Prompt 3 -- Research Gap Identification

#### Scenario (实际场景)

You have read a body of literature and have rough notes, but you struggle to articulate what is missing in a way that motivates your own study. This prompt helps you systematically identify and frame five types of gaps as positive research opportunities.

#### Roles

You are a research strategist helping a GIS doctoral student frame their contribution.

#### Prompt Template

```text
Based on the following summary of existing research on [TOPIC]:

"""
[PASTE_NOTES -- these can be rough bullet points, a paragraph,
or even a pasted abstract collection. Example:
- Most UHI studies use Landsat or MODIS at city scale
- Few studies combine LST with social vulnerability indices
- Machine learning used mainly for prediction, not explanation
- Limited work in tropical cities of Southeast Asia]
"""

Identify and articulate research gaps in five dimensions:

1. **Methodological gaps**: What techniques have not been tried
   or combined? What analytical frameworks are missing?
2. **Spatial gaps**: What regions, scales, or spatial contexts
   are understudied?
3. **Temporal gaps**: What time periods, frequencies, or
   temporal dynamics are missing?
4. **Data gaps**: What data sources could add value but have
   not been exploited?
5. **Application gaps**: What practical or policy applications
   remain unexplored?

For each gap:
- Write 2-3 sentences in academic style
- Frame as a motivation ("This presents an opportunity to...")
  rather than a criticism ("Previous studies failed to...")
- Include a [CITE] placeholder where supporting evidence exists
- Rate the gap as: Critical / Important / Emerging

End with a synthesis paragraph connecting the most critical gaps
to [MY_STUDY_FOCUS].
```

#### Variables to Customize

- `TOPIC` -- the research domain
- `PASTE_NOTES` -- your literature notes (any format)
- `MY_STUDY_FOCUS` -- one-sentence description of your planned study

#### Expected Output

A structured gap analysis with five categories, severity ratings, and a synthesis paragraph linking gaps to your research contribution.

#### Validation Checklist

- [ ] Each gap category has at least one identified gap
- [ ] Gaps are framed as opportunities, not attacks on prior work
- [ ] Synthesis paragraph clearly connects gaps to the user's study
- [ ] `[CITE]` placeholders appear where claims reference the literature
- [ ] Severity ratings (Critical / Important / Emerging) are assigned

#### Cost Optimization

This prompt benefits from a strong reasoning model. Use Claude Opus / GPT-4o for the gap analysis, then a lighter model to polish the prose.

> **Dark Arts Tip:** Include a "negative constraint" -- tell the model which gaps you do NOT want it to suggest (e.g., "Do not suggest more data is needed if data availability is not an issue for my study"). This forces more creative and relevant gap identification.

#### Related Pages

- [Data Sources](../data-sources/) -- check for data gaps against available catalogs

#### Extensibility

- Chain with **P1 (Thematic Literature Summary)** as input.
- Feed output into **P4 (Theoretical Framework)** to position your study conceptually.
- Use the severity ratings to prioritize which gaps to address in a multi-paper dissertation.

---

### Prompt 4 -- Theoretical Framework Construction

#### Scenario (实际场景)

You need to build a conceptual or theoretical framework that connects your GIS methods to your research questions, showing how data flows through analytical steps to produce answers. Reviewers increasingly expect a visual framework diagram in the Introduction or Methodology section.

#### Roles

You are an academic research designer who specializes in conceptual modeling for geospatial studies.

#### Prompt Template

```text
Construct a theoretical/conceptual framework for a study that
uses [GIS_METHOD] to investigate [RESEARCH_QUESTION].

Framework components:
- Research questions: [LIST_RQS]
- Theoretical basis: [THEORY, e.g., Tobler's First Law,
  spatial heterogeneity, landscape ecology, environmental
  justice framework]
- Key concepts: [CONCEPTS, e.g., spatial autocorrelation,
  scale dependency, land-atmosphere interaction]
- Data inputs: [DATA_LIST]
- Analytical methods: [METHODS_LIST]
- Expected outputs: [OUTPUTS]

Produce:
1. A textual description of the framework (2-3 paragraphs)
   suitable for a journal paper's Introduction section
2. A Mermaid diagram (```mermaid ... ```) showing the
   relationships between components
3. Optionally, a TikZ/LaTeX version of the same diagram

The framework should show:
- How theory motivates the research questions
- How data feeds into methods
- How methods address each research question
- Feedback loops or iterative steps (if any)

Use [CITE] for theoretical references.
```

#### Variables to Customize

- `GIS_METHOD` -- primary analytical approach
- `RESEARCH_QUESTION` -- the overarching question(s)
- `LIST_RQS` -- numbered research sub-questions
- `THEORY` -- underlying theoretical basis
- `DATA_LIST` / `METHODS_LIST` / `OUTPUTS` -- study components

#### Expected Output

A textual framework description, a Mermaid flowchart, and optionally a TikZ diagram. The text is ready for the Introduction; the diagram is ready for Figure 1 or Figure 2.

#### Validation Checklist

- [ ] Every research question maps to at least one method and output
- [ ] Theoretical basis is explicitly connected to the analytical approach
- [ ] Mermaid diagram renders correctly (test at mermaid.live)
- [ ] No orphan nodes -- every component connects to at least one other

#### Cost Optimization

Mermaid syntax is simple; even a small model can generate it. Use a frontier model only for the textual description. Total cost: minimal.

> **Dark Arts Tip:** Ask the model to generate three alternative framework layouts (linear, cyclical, hierarchical) and pick the one that best fits your study design. Reviewers appreciate a well-chosen structure.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- analytical tools referenced in the framework

#### Extensibility

- Export the Mermaid diagram as SVG for journal submission.
- Use the TikZ version in your LaTeX manuscript directly.
- Chain with **P5 (Full Methodology Draft)** -- the framework provides the skeleton.

---

## 2. Methodology Section (P5--P8)

---

### Prompt 5 -- Full Methodology Draft

#### Scenario (实际场景)

You need to write the complete methodology section of a GIS research paper. This prompt produces a structured draft covering study area, data, methods (with LaTeX equations), validation, and software environment -- all in one pass.

#### Roles

You are a senior GIS researcher drafting a methodology section for a Q1 journal submission.

#### Prompt Template

```text
Write a methodology section for a study using [METHOD, e.g.,
"geographically weighted regression (GWR)"] to analyze
[OBJECTIVE, e.g., "the spatial relationship between PM2.5
and land use variables"].

Study details:
- Study area: [NAME, extent, justification for selection]
- Time period: [DATES]
- Dependent variable: [NAME, source, resolution, unit]
- Independent variables:
  [VAR1: source, resolution]
  [VAR2: source, resolution]
  [VAR3: source, resolution]
- Sample size / spatial units: [N, type, e.g., 500 census tracts]
- Software: [e.g., R 4.3 with GWmodel / Python 3.11 with mgwr]
- CRS: [EPSG code, e.g., EPSG:4326 for storage, EPSG:32650
  for analysis]

Section structure:
1. Study area (1 paragraph + reference to Figure 1)
2. Data acquisition and preprocessing (1-2 paragraphs per
   data source, include temporal alignment and resampling)
3. Analytical method with mathematical formulation:
   - Core equation(s) in LaTeX, e.g.:
     $y_i = \beta_0(u_i, v_i) + \sum_k \beta_k(u_i, v_i) x_{ik} + \epsilon_i$
   - Bandwidth selection (e.g., golden section search, AICc)
   - Kernel function (e.g., adaptive bisquare)
4. Variable selection procedure (VIF, stepwise, AICc)
5. Validation approach (cross-validation, residual analysis)
6. Software and computational environment

Style: passive voice, past tense, suitable for [JOURNAL].
Include key equations in LaTeX math notation.
Use [CITE] for methodological references.
Target length: 1500-2500 words.
```

#### Variables to Customize

- `METHOD` -- analytical method (GWR, Random Forest classification, spatial interpolation, etc.)
- `OBJECTIVE` -- what the analysis investigates
- Study details block -- fill in all specifics
- `JOURNAL` -- target journal for style matching

#### Expected Output

A 1500--2500 word methodology draft with LaTeX equations, `[CITE]` placeholders, and a clear structure matching the target journal's conventions.

#### Validation Checklist

- [ ] All variables described with source, resolution, and units
- [ ] At least one LaTeX equation is included and correct
- [ ] CRS is stated for both storage and analysis
- [ ] Software versions are specified (or flagged with `[VERSION?]`)
- [ ] Validation approach is described with specific metrics

#### Cost Optimization

This is a long-output prompt. Use a frontier model with a high output token limit (Claude Opus, GPT-4o). Consider splitting into sub-prompts (study area, data, methods, validation) if the output gets truncated.

> **Dark Arts Tip:** Paste your actual variable names (column headers from your DataFrame) into the prompt. The model will use them consistently, saving you search-and-replace work later.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- mgwr, PySAL, scikit-learn, rasterio
- [Tools: CLI Tools](../tools/cli-tools.md) -- GDAL, OGR, GRASS GIS commands
- [Data Sources](../data-sources/) -- satellite imagery, census data, OpenStreetMap

#### Extensibility

- Chain with **P8 (Reproducibility Statement)** to add the final reproducibility paragraph.
- Feed into **P6 (Processing Workflow)** for more detail on specific pipeline steps.
- Use the LaTeX equations directly in your `.tex` manuscript.

---

### Prompt 6 -- Processing Workflow Description

#### Scenario (实际场景)

Your methodology involves a multi-step data processing pipeline (download, preprocess, compute indices, classify, validate). You need to describe this pipeline clearly enough that another researcher could replicate it. This prompt produces a step-by-step description with tools, parameters, and justifications.

#### Roles

You are a geospatial data engineer writing reproducible workflow documentation for a methods paper.

#### Prompt Template

```text
Write a clear, reproducible description of the following GIS
processing workflow for the methodology section of a journal paper.

Steps (fill in your actual workflow):
1. [STEP_1, e.g., "Download Sentinel-2 L2A imagery from
   Copernicus Data Space for June-August 2024"]
2. [STEP_2, e.g., "Apply cloud masking using SCL band,
   discard scenes with >20% cloud cover"]
3. [STEP_3, e.g., "Calculate monthly median composites"]
4. [STEP_4, e.g., "Compute NDVI, NDWI, NDBI spectral indices"]
5. [STEP_5, e.g., "Perform supervised classification using
   Random Forest (n_estimators=500)"]
6. [STEP_6, e.g., "Validate using stratified random sample
   of 500 points from independent reference data"]

For each step, describe:
- Input data (format, CRS, resolution)
- Output data (format, CRS, resolution)
- Tool or function used (with version, e.g., rasterio 1.3.9)
- Key parameters and their values
- Justification for methodological choices (e.g., "median
  composite was preferred over mean to reduce cloud artifact
  influence [CITE]")

Write in academic past tense. Keep concise but sufficient
for replication. Total length: ~800-1200 words.

End with a summary workflow diagram description (suitable for
a Mermaid flowchart or a numbered list in a figure).
```

#### Variables to Customize

- `STEP_1` through `STEP_N` -- your actual processing steps
- Tool names and version numbers
- Parameter values and their justifications

#### Expected Output

A structured workflow description (800--1200 words) with tool versions, parameters, and justifications, plus a workflow summary suitable for a figure.

#### Validation Checklist

- [ ] Every step specifies input format, output format, and CRS
- [ ] Tool versions are stated or flagged with `[VERSION?]`
- [ ] At least one methodological choice is justified with `[CITE]`
- [ ] Workflow is linear and reproducible (no ambiguous steps)

#### Cost Optimization

A mid-tier model handles this well. The structure is formulaic. Save frontier tokens for creative writing tasks.

> **Dark Arts Tip:** Paste your actual Python/R script as context. The model will extract parameters, function names, and versions directly, producing a far more accurate description than working from memory.

#### Related Pages

- [Tools: CLI Tools](../tools/cli-tools.md) -- GDAL, sen2cor, SNAP GPT
- [Tools: Python Libraries](../tools/python-libraries.md) -- rasterio, xarray, scikit-learn

#### Extensibility

- Generate a Mermaid flowchart from the workflow summary.
- Chain with **P8 (Reproducibility Statement)** for software/data availability.
- Convert into a Snakemake or Makefile for actual pipeline execution.

---

### Prompt 7 -- Notes-to-Formal Conversion

#### Scenario (实际场景)

You have rough notes, messy bullet points, or lab notebook entries describing what you did in your analysis. You need to transform them into a properly structured, journal-ready methodology section without losing any detail. This is the single most time-saving prompt in this collection.

#### Roles

You are an academic ghostwriter specializing in converting raw research notes into polished GIS methodology prose.

#### Prompt Template

```text
Convert the following rough methodology notes into a formal
methods section suitable for submission to [JOURNAL, e.g.,
"International Journal of Geographical Information Science"].

My rough notes:
"""
[PASTE_NOTES -- messy bullets, lab notebook entries, code
comments, Slack messages to your advisor, or informal
descriptions. Example:
- downloaded sentinel2 from copernicus summer 2023+2024
- used sen2cor for atmo correction, some already L2A
- made NDVI each date, SCL band for cloud mask
- RF classification 6 classes 500 trees
- training: 200 GPS points field survey + google earth
- accuracy 87.3% OA, kappa 0.84
- water best 98%, urban-rural worst 72%
- used QGIS 3.34 and python sklearn rasterio]
"""

Transform into formal academic prose:
1. Structure: Study Area -> Data -> Preprocessing -> Method
   -> Validation
2. Voice: passive, past tense ("Imagery was acquired...")
3. Add detail I know but didn't write:
   - Full sensor names and spatial resolutions
   - Band designations for indices (e.g., NDVI = (B8-B4)/(B8+B4))
   - Software versions: use [VERSION?] if uncertain
   - Mathematical formulations in LaTeX
4. Placeholders:
   - [CITE] where references needed
   - [VERIFY] where you inferred something
   - [?] where a specific number is missing
5. Length: ~[WORD_COUNT] words
6. Use proper technical terminology throughout
7. End with a reproducibility paragraph

Do NOT fabricate specific numbers or results.
```

#### Variables to Customize

- `JOURNAL` -- target journal for style matching
- `PASTE_NOTES` -- your raw notes (the main input; any format works)
- `WORD_COUNT` -- desired output length (typically 1000--2000)

#### Expected Output

A formal methodology section with `[CITE]`, `[VERIFY]`, `[VERSION?]`, and `[?]` placeholders. Ready for review and reference insertion.

#### Validation Checklist

- [ ] All raw notes are accounted for (nothing dropped silently)
- [ ] Inferred details are flagged with `[VERIFY]`
- [ ] Missing numbers use `[?]` instead of fabricated values
- [ ] LaTeX equations are syntactically correct
- [ ] Tone matches the target journal

#### Cost Optimization

Use a frontier model here -- the quality difference between models is most visible in prose polishing tasks. Worth the extra cost.

> **Dark Arts Tip:** Include your actual code file (or key snippets) alongside the notes. The model will extract parameter values, function calls, and library versions directly from the code, dramatically reducing `[VERIFY]` and `[VERSION?]` placeholders.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- for verifying library names and versions
- [Tools: CLI Tools](../tools/cli-tools.md) -- for verifying command-line tool references

#### Extensibility

- Chain with **P8 (Reproducibility Statement)** for the closing paragraph.
- Run a second pass asking the model to remove all remaining placeholders by inferring from context (use with caution).
- Pair with Grammarly or LanguageTool for final proofreading.

---

### Prompt 8 -- Reproducibility Statement

#### Scenario (实际场景)

Journals increasingly require (or strongly encourage) a statement about computational reproducibility: software versions, hardware, code availability, and data access. This prompt generates that statement, including references to STAC catalogs for open data and GitHub for code.

#### Roles

You are a research reproducibility advocate drafting a transparency statement for a GIS journal paper.

#### Prompt Template

```text
Write a reproducibility/data availability statement for a
GIS research paper. This will appear at the end of the
Methodology section or as a standalone section before
References.

Software environment:
- Language: [e.g., Python 3.11.7]
- Key packages: [e.g., geopandas 0.14.3, rasterio 1.3.9,
  scikit-learn 1.4.0, mgwr 2.2.1]
- IDE: [e.g., JupyterLab 4.1, VS Code 1.86]
- OS: [e.g., Ubuntu 22.04 LTS]

Hardware:
- [e.g., Intel i9-13900K, 64 GB RAM, NVIDIA RTX 4090 24 GB]
- Processing time: [e.g., ~4 hours for full pipeline]

Code availability:
- Repository: [URL, e.g., https://github.com/username/repo]
- License: [e.g., MIT]
- DOI: [e.g., Zenodo DOI, or "to be assigned upon acceptance"]

Data availability:
- [DATA_1]: [source, access method, license]
  [e.g., "Sentinel-2 L2A: Copernicus Data Space (STAC API),
  free and open"]
- [DATA_2]: [source, access method, license]
- Derived data: [availability, e.g., "available in the
  repository above" or "available upon request"]

Write 1-2 paragraphs covering all of the above in academic
prose. Follow [JOURNAL] guidelines for data statements.
Include URLs and DOIs where available.
```

#### Variables to Customize

- Software, packages, and their exact versions
- Hardware specifications and processing times
- Code repository URL, license, and DOI
- Data sources with access methods and licenses
- `JOURNAL` -- for format compliance

#### Expected Output

A 1--2 paragraph reproducibility statement suitable for the end of the Methodology section or a standalone "Data and Code Availability" section.

#### Validation Checklist

- [ ] All software versions are specified
- [ ] Code repository URL is included (or placeholder)
- [ ] Data sources include access method and license
- [ ] Hardware is described (important for ML/DL studies)
- [ ] Statement follows the target journal's format

#### Cost Optimization

This is a template-heavy task. A small model (Haiku, GPT-4o-mini) handles it perfectly. No need for a frontier model.

> **Dark Arts Tip:** Run `pip freeze` or `conda list --export` and paste the full output. The model will extract only the relevant packages and their versions, saving you from hunting through dependency lists.

#### Related Pages

- [Data Sources](../data-sources/) -- STAC catalogs and open data sources for the data availability section
- [Tools: Python Libraries](../tools/python-libraries.md) -- package names and typical versions
- [Tools: CLI Tools](../tools/cli-tools.md) -- GDAL, GRASS, SAGA version conventions

#### Extensibility

- Generate a `requirements.txt` or `environment.yml` alongside the statement.
- Create a Dockerfile snippet for full computational reproducibility.
- Link to a Binder badge for one-click reproducibility.

---

## 3. Results Description (P9--P12)

---

### Prompt 9 -- Spatial Pattern Description

#### Scenario (实际场景)

You have produced spatial analysis results -- LISA cluster maps, hotspot analysis, interpolation surfaces -- and need to describe the spatial patterns in academic prose. The challenge is translating visual map patterns into precise, objective text without over-interpreting causation (which belongs in the Discussion).

#### Roles

You are a spatial analyst writing the Results section of a paper for a geographic information science journal.

#### Prompt Template

```text
Describe the spatial patterns shown in [ANALYSIS_TYPE, e.g.,
"a LISA cluster map of median household income across 500
census tracts in [CITY]"].

Results data:
- Global Moran's I: [VALUE], z-score: [VALUE], p < [VALUE]
- High-High clusters: [N] tracts, primarily in [LOCATIONS]
- Low-Low clusters: [N] tracts, primarily in [LOCATIONS]
- High-Low outliers: [N] tracts, notable in [LOCATIONS]
- Low-High outliers: [N] tracts, notable in [LOCATIONS]
- Not significant: [N] tracts
- Spatial weights: [TYPE, e.g., queen contiguity, k=4 nearest
  neighbors, distance band = 2 km]
- Significance level: [e.g., p < 0.05, 999 permutations]

Write 2-3 paragraphs:
1. Report the global statistic, its significance, and what
   it means for the overall spatial pattern
2. Describe the cluster distribution using geographic
   directions (northern, central, peripheral) and named
   landmarks or districts
3. Highlight notable patterns (alignment with rivers,
   highways, administrative boundaries)
4. Note outliers and anomalies
5. Briefly compare to expected patterns from prior work [CITE]

Tone: objective, academic, past tense.
Do NOT interpret causation -- save that for Discussion.
Reference figures as "Figure [N]" and tables as "Table [N]".
```

#### Variables to Customize

- `ANALYSIS_TYPE` -- LISA, Getis-Ord Gi*, kernel density, IDW, kriging, etc.
- Results data block -- fill in from your actual output
- `CITY` / study area name
- Geographic features to reference (rivers, highways, districts)

#### Expected Output

Two to three paragraphs of objective spatial pattern description with figure/table cross-references and `[CITE]` placeholders.

#### Validation Checklist

- [ ] Global statistic is reported with significance level
- [ ] Spatial patterns described using geographic references, not just "areas"
- [ ] No causal claims (correlation and pattern only)
- [ ] Figures and tables are cross-referenced
- [ ] Comparison to prior work uses `[CITE]`

#### Cost Optimization

A mid-tier model handles formulaic results description well. Save frontier tokens for the Discussion section where interpretation matters more.

> **Dark Arts Tip:** Paste a screenshot or text description of your actual LISA map. Visual context (even described verbally) helps the model produce descriptions that match what readers will see in the figure.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- PySAL, esda, splot for spatial statistics

#### Extensibility

- Chain with a Discussion prompt to interpret the patterns you described.
- Generate a LaTeX `\begin{figure}` block referencing the described map.
- Pair with **P13 (Map Figure Caption)** for the corresponding figure caption.

---

### Prompt 10 -- Accuracy Assessment Reporting

#### Scenario (实际场景)

You completed a classification (land cover, land use, change detection) and need to report the accuracy assessment results. This prompt produces the accuracy sub-section with confusion matrix interpretation, per-class metrics, and comparison to prior work -- following the Landis & Koch (1977) kappa interpretation scale.

#### Roles

You are a remote sensing scientist reporting classification accuracy for a peer-reviewed journal.

#### Prompt Template

```text
Write the accuracy assessment subsection of a Results section.

Classification details:
- Type: [e.g., land cover, 6 classes: Water, Forest, Cropland,
  Urban, Bare Soil, Grassland]
- Classifier: [e.g., Random Forest, 500 trees]
- Validation method: [e.g., stratified random sampling, 500
  points, independent from training data]

Metrics:
- Overall accuracy (OA): [VALUE]%
- Kappa coefficient: [VALUE]
- Per-class Producer's Accuracy (PA): [CLASS1: X%, CLASS2: Y%, ...]
- Per-class User's Accuracy (UA): [CLASS1: X%, CLASS2: Y%, ...]
- F1 score per class: [CLASS1: X, CLASS2: Y, ...]

Confusion matrix (optional, paste if available):
[MATRIX or "see Table N"]

Write the subsection to:
1. State the validation approach and sample size
2. Report OA and kappa with Landis & Koch interpretation
   ("substantial agreement" for 0.61-0.80, etc.)
3. Reference the confusion matrix as Table [N]
4. Discuss best- and worst-performing classes
5. Identify the most common misclassification pairs and
   suggest spectral/spatial reasons
6. Compare to similar studies in the region or domain [CITE]

Length: 1.5-2 paragraphs. Concise and data-driven.

Also produce a LaTeX table caption for the confusion matrix:
\caption{Confusion matrix for [CLASSIFICATION_TYPE]...}
```

#### Variables to Customize

- Classification type, classes, and classifier
- Validation method and sample size
- All metric values
- Confusion matrix (optional)

#### Expected Output

An accuracy assessment paragraph (1.5--2 paragraphs) plus a LaTeX table caption for the confusion matrix.

#### Validation Checklist

- [ ] Landis & Koch interpretation matches the kappa value
- [ ] Best and worst classes are explicitly named with metrics
- [ ] Misclassification pairs are explained with spectral/spatial reasoning
- [ ] Table cross-reference is included
- [ ] Comparison to prior work uses `[CITE]`

#### Cost Optimization

Accuracy reporting is highly formulaic. A small model (Haiku, GPT-4o-mini) handles it well. Very low cost.

> **Dark Arts Tip:** Paste the raw `sklearn.metrics.classification_report()` output directly. The model will parse it perfectly and save you from transcribing numbers manually.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- scikit-learn metrics, seaborn for confusion matrix visualization

#### Extensibility

- Chain with **P15 (LaTeX Table Generation)** to produce the full confusion matrix table.
- Add McNemar's test for statistical comparison between two classifiers.
- Generate a per-class bar chart description for a supplementary figure.

---

### Prompt 11 -- Regression Results Narrative

#### Scenario (实际场景)

You ran a regression analysis (OLS, GWR, spatial lag/error model) and need to narrate the results: which variables are significant, what the coefficients mean spatially, how model fit compares across specifications, and whether residuals show spatial autocorrelation.

#### Roles

You are a quantitative geographer reporting regression results for a spatial analysis journal.

#### Prompt Template

```text
Write the regression results subsection for a study comparing
[MODEL_A, e.g., OLS] and [MODEL_B, e.g., GWR] models.

Model results:
OLS:
- R-squared: [VALUE]
- Adjusted R-squared: [VALUE]
- AICc: [VALUE]
- Significant variables (p < 0.05):
  [VAR1: beta = X, SE = Y, p = Z]
  [VAR2: beta = X, SE = Y, p = Z]
- Non-significant: [VAR3: beta = X, p = Z]
- Moran's I of residuals: [VALUE], p = [VALUE]

GWR:
- R-squared range: [MIN -- MAX]
- Mean R-squared: [VALUE]
- AICc: [VALUE]
- Bandwidth: [VALUE, type: adaptive/fixed]
- Coefficient ranges for key variables:
  [VAR1: min = A, median = B, max = C]
  [VAR2: min = A, median = B, max = C]
- Moran's I of residuals: [VALUE], p = [VALUE]

Write 2-3 paragraphs:
1. OLS results: significant predictors, direction of effects,
   model fit (R-squared, AICc), residual autocorrelation
2. GWR improvement: compare AICc, R-squared; describe spatial
   non-stationarity of coefficients
3. Key finding: which variable(s) show the most spatial
   variation in their effect?

Include LaTeX notation for the regression equation:
$y_i = \beta_0 + \beta_1 x_{1i} + ... + \epsilon_i$ (OLS)
$y_i = \beta_0(u_i,v_i) + \beta_1(u_i,v_i)x_{1i} + ... + \epsilon_i$ (GWR)

Reference coefficient maps as Figure [N].
Use [CITE] for methodological references.
```

#### Variables to Customize

- `MODEL_A` / `MODEL_B` -- models being compared
- All coefficient values, standard errors, p-values
- R-squared, AICc, bandwidth, residual Moran's I
- Variable names matching your study

#### Expected Output

Two to three paragraphs comparing model results, with LaTeX equations and figure/table cross-references.

#### Validation Checklist

- [ ] AICc comparison correctly identifies the better model (lower = better)
- [ ] Coefficient signs (positive/negative) are interpreted correctly
- [ ] Residual autocorrelation is discussed for both models
- [ ] LaTeX equations are syntactically correct
- [ ] Spatial non-stationarity is described for GWR coefficients

#### Cost Optimization

A mid-tier model works well for numerical reporting. Use a frontier model only if the interpretation requires nuanced reasoning about coefficient variation.

> **Dark Arts Tip:** Paste your `summary()` output from R or Python directly. The model will parse regression tables more accurately than you can describe them in prose.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- mgwr, PySAL spreg, statsmodels

#### Extensibility

- Chain with **P9 (Spatial Pattern Description)** for coefficient map descriptions.
- Add MGWR (multiscale GWR) results as a third model comparison.
- Generate a LaTeX coefficient summary table using **P15**.

---

### Prompt 12 -- Time Series Results

#### Scenario (实际场景)

You have temporal analysis results -- trend decomposition, change detection, phenological analysis, or anomaly detection from satellite time series -- and need to describe temporal patterns, breakpoints, and seasonal dynamics in academic prose.

#### Roles

You are a remote sensing time series analyst writing results for an Earth observation journal.

#### Prompt Template

```text
Describe the temporal analysis results for [VARIABLE, e.g.,
"NDVI time series from 2000 to 2023"] over [STUDY_AREA].

Time series details:
- Temporal resolution: [e.g., 16-day MODIS composites]
- Spatial aggregation: [e.g., mean per land cover class /
  pixel-level / district-level]
- Analysis method: [e.g., BFAST, Mann-Kendall, STL
  decomposition, harmonic regression]

Key results:
- Overall trend: [DESCRIPTION, e.g., "significant greening
  trend of +0.003 NDVI/year, p < 0.01"]
- Seasonal pattern: [DESCRIPTION, e.g., "peak NDVI in July,
  trough in January, amplitude = 0.35"]
- Breakpoints detected: [DATES, magnitudes, locations]
- Anomalies: [YEARS, magnitude, possible attribution]
- Spatial variation in trends: [DESCRIPTION]

Write 2-3 paragraphs:
1. Overall trend: magnitude, significance, and spatial extent
   of increasing/decreasing trends
2. Seasonality: amplitude, timing (SOS, EOS if phenological),
   changes in seasonal pattern over time
3. Breakpoints and anomalies: timing, magnitude, spatial
   distribution, and tentative attribution (drought, policy
   change, urbanization) with [CITE]

Reference time series plots as Figure [N].
Reference trend maps as Figure [N+1].
Use [CITE] for methodological and contextual references.

Tone: objective, past tense. Attribution should be tentative
("may be associated with", "potentially linked to") unless
supported by strong evidence.
```

#### Variables to Customize

- `VARIABLE` -- NDVI, LST, precipitation, built-up area, etc.
- `STUDY_AREA` -- geographic context
- Time series specifications (sensor, resolution, method)
- Key results data

#### Expected Output

Two to three paragraphs describing temporal trends, seasonality, breakpoints, and anomalies, with figure cross-references and `[CITE]` placeholders.

#### Validation Checklist

- [ ] Trend magnitude and significance are reported with units
- [ ] Seasonal timing uses domain-appropriate terms (SOS, EOS, peak, trough)
- [ ] Breakpoints include both timing and magnitude
- [ ] Attribution language is tentative, not causal
- [ ] Figures are cross-referenced

#### Cost Optimization

Time series descriptions follow a predictable structure. A mid-tier model is sufficient. Save frontier tokens for anomaly interpretation.

> **Dark Arts Tip:** Include the actual time series plot (as a text description or even ASCII art of the trend) in the prompt. The model writes much better descriptions when it can "see" the pattern rather than working from summary statistics alone.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- xarray, pandas, scipy for time series
- [Data Sources](../data-sources/) -- MODIS, Landsat, Sentinel data catalogs

#### Extensibility

- Chain with a Discussion prompt to interpret breakpoints in the context of policy or climate events.
- Generate a supplementary table of per-pixel trend statistics using **P15**.
- Pair with **P13 (Map Figure Caption)** for the trend map figure caption.

---

## 4. Figures & Tables (P13--P16)

---

### Prompt 13 -- Map Figure Caption

#### Scenario (实际场景)

You need a descriptive, journal-compliant caption for a choropleth, heat map, or thematic map figure. Different journals (RSE, ISPRS JPRS, CAGEO) have different caption conventions. This prompt handles the variation and produces a caption that passes editorial review on the first try.

#### Roles

You are a cartographer and academic editor drafting figure captions for a GIS journal manuscript.

#### Prompt Template

```text
Write a figure caption for a [MAP_TYPE, e.g., "choropleth map"
/ "kernel density heat map" / "classified land cover map"]
showing [VARIABLE, e.g., "NDVI change from 2010 to 2023"]
in [STUDY_AREA].

Figure details:
- Map extent: [DESCRIPTION, e.g., "Chengdu metropolitan area,
  30.4-31.0N, 103.6-104.5E"]
- Classification: [N] classes using [METHOD, e.g., natural
  breaks / equal interval / quantile / manual]
- Color scheme: [DESCRIPTION, e.g., "diverging RdYlGn from
  ColorBrewer, red = decrease, green = increase"]
- Inset/overview map: [YES/NO, what it shows]
- Scale bar and north arrow: [INCLUDED / NOT INCLUDED]
- Projection: [NAME, e.g., UTM Zone 48N]
- Data source: [SOURCE and DATE]
- Sub-figures: [DESCRIPTIONS if (a)(b)(c) panels exist]

Caption format for [JOURNAL]:
- RSE style: "Fig. N." prefix, concise, data source in caption
- ISPRS style: "Figure N." prefix, detailed, sub-panel labels
- CAGEO style: "Fig. N." prefix, moderate detail

Produce:
1. The caption (2-4 sentences, under 75 words)
2. A LaTeX \caption{} version with \label{fig:ID}
```

#### Variables to Customize

- `MAP_TYPE` -- choropleth, heat map, dot density, isoline, classified, etc.
- `VARIABLE` -- what the map shows
- `STUDY_AREA` -- geographic extent
- Figure details block -- fill in specifics
- `JOURNAL` -- for format matching

#### Expected Output

A figure caption in both plain text and LaTeX format, tailored to the target journal's style.

#### Validation Checklist

- [ ] Caption starts with the correct prefix for the journal (Fig./Figure)
- [ ] Classification method and color scheme are mentioned
- [ ] Data source and date are included (if journal requires it)
- [ ] Under 75 words for single maps

#### Cost Optimization

Captions are short. Use the cheapest model available. Batch multiple captions in one prompt to reduce overhead.

> **Dark Arts Tip:** Provide the model with one example caption from your target journal. It will mimic the exact style, abbreviation conventions, and level of detail far more reliably than following generic rules.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- matplotlib, cartopy for map generation

#### Extensibility

- Batch all figure captions for a paper in a single prompt.
- Chain with **P16 (Study Area Map Script)** to generate figure + caption together.

---

### Prompt 14 -- Multi-Panel Figure Caption

#### Scenario (实际场景)

You have a composite figure with multiple panels -- (a) true-color image, (b) NDVI map, (c) classification result, (d) accuracy chart -- and need a caption that describes both the overall purpose and each sub-panel. Multi-panel captions are among the most commonly rejected elements in peer review for being incomplete or inconsistent.

#### Roles

You are a scientific illustrator drafting a multi-panel figure caption for a remote sensing journal.

#### Prompt Template

```text
Write a caption for a multi-panel figure with the following
sub-panels:

- (a): [DESCRIPTION, e.g., "True-color Sentinel-2 composite
  of the study area, acquired 2024-07-15"]
- (b): [DESCRIPTION, e.g., "NDVI map derived from (a),
  classified into 5 categories using natural breaks"]
- (c): [DESCRIPTION, e.g., "Land cover classification result
  (6 classes) from Random Forest"]
- (d): [DESCRIPTION, e.g., "Spatial distribution of
  classification errors overlaid on (c)"]

Overall figure purpose: [e.g., "illustrating the complete
workflow from raw imagery to classification and validation"]

Caption requirements for [JOURNAL, e.g., ISPRS Journal]:
- Main sentence: overall figure purpose
- Sub-panel descriptions: "(a) description; (b) description;
  (c) description; (d) description."
- Include data source if required by journal
- Total length: 3-6 sentences

Produce:
1. Plain text caption
2. LaTeX version: \caption{Main text. (a) ...; (b) ...; ...}
   \label{fig:ID}
```

#### Variables to Customize

- Panel descriptions `(a)` through `(d)` or more
- Overall purpose of the composite figure
- `JOURNAL` -- for style compliance
- Number of panels (2--8 typical)

#### Expected Output

A structured multi-panel caption in plain text and LaTeX, with consistent formatting across all sub-panels.

#### Validation Checklist

- [ ] Every panel is described in the caption
- [ ] Sub-panel labels match the figure layout exactly
- [ ] Main sentence states the overall purpose before sub-panel details
- [ ] Consistent style across sub-panels (all past tense or all present)

#### Cost Optimization

Very low token cost. Batch with **P13** if you have multiple figures.

> **Dark Arts Tip:** Number your panels in the prompt even before you finalize the figure layout. If panels change order during revision, update the prompt and regenerate -- faster than editing by hand and less error-prone.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- matplotlib subplot layouts

#### Extensibility

- Generate a LaTeX `\begin{figure*}` environment with `\subfloat` or `subcaption` package code.
- Chain with **P16 (Study Area Map Script)** for panel (a) generation.

---

### Prompt 15 -- LaTeX Table Generation

#### Scenario (实际场景)

You have spatial analysis results (accuracy metrics, regression coefficients, zonal statistics) in CSV, pandas output, or raw console text, and you need a publication-quality LaTeX table with `booktabs` rules, `siunitx` number formatting, significance stars, and a pandas-to-LaTeX snippet for reproducibility.

#### Roles

You are a LaTeX typesetting specialist producing tables for geospatial research papers.

#### Prompt Template

```text
Generate a publication-quality LaTeX table from the following
spatial analysis results.

Table type: [accuracy assessment / regression coefficients /
zonal statistics / method comparison / descriptive statistics]

Data (paste in any format -- CSV, JSON, pandas output, or
plain text):
"""
[PASTE_DATA. Example:
class,PA,UA,F1
Water,0.98,0.97,0.975
Forest,0.91,0.89,0.900
Cropland,0.85,0.88,0.865
Urban,0.82,0.79,0.805
Bare Soil,0.78,0.81,0.795]
"""

LaTeX requirements:
- Packages: booktabs, siunitx (S column for numbers)
- Environment: table [htbp], centered
- Caption: [CAPTION or "auto-generate from data"]
- Label: \label{tab:[TABLE_ID]}
- Formatting:
  - Text columns: l (left-aligned)
  - Numeric columns: S[table-format=1.3] or r
  - Bold the best value per column
  - Significance stars for regression: * p<0.05, ** p<0.01,
    *** p<0.001
- Table notes: \footnotesize, explain abbreviations
- Wide tables: use \resizebox{\textwidth}{!}{...}

Also provide:
1. Raw LaTeX code (copy-paste ready)
2. Markdown version of the same table
3. Python snippet using pandas to_latex() that regenerates
   the table programmatically:
   ```python
   df.to_latex(
       buf="table.tex",
       index=False,
       escape=False,
       column_format="lSSS",
       caption="...",
       label="tab:..."
   )
   ```
```

#### Variables to Customize

- `PASTE_DATA` -- your results in any format
- Table type (determines formatting conventions)
- `CAPTION` -- or let the model auto-generate
- `TABLE_ID` -- LaTeX label
- Column format preferences
- Whether significance stars are needed

#### Expected Output

Three outputs: (1) LaTeX table code, (2) Markdown table, (3) Python `pandas.to_latex()` snippet for reproducibility.

#### Validation Checklist

- [ ] LaTeX compiles without errors (booktabs, siunitx loaded)
- [ ] Best values are bolded consistently
- [ ] Significance stars match the stated p-value thresholds
- [ ] Table notes explain all abbreviations
- [ ] pandas snippet produces equivalent output

#### Cost Optimization

Tables are structured and formulaic. A small model handles this perfectly. Batch multiple tables in one prompt.

> **Dark Arts Tip:** Pipe your `DataFrame.to_string()` output directly into the prompt. The model preserves alignment and column names exactly, eliminating transcription errors that plague manual table creation.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- pandas, stargazer, great_tables

#### Extensibility

- Add a `threeparttable` version for journals that require structured table notes.
- Generate `longtable` for tables spanning multiple pages.
- Create a Typst version alongside LaTeX for modern typesetting workflows.

---

### Prompt 16 -- Study Area Map Script

#### Scenario (实际场景)

Every GIS paper needs a study area map (typically Figure 1). This prompt generates a complete, runnable Python script that produces a publication-quality map with an inset locator, scale bar, north arrow, labels, and 300 DPI export -- ready for journal submission without manual editing in a GIS.

#### Roles

You are a Python cartographer generating a camera-ready study area map for a journal paper.

#### Prompt Template

```text
Write a complete Python script to create a publication-quality
study area map using [LIBRARY: matplotlib + cartopy 0.22+ /
geopandas 0.14+ with contextily 1.5+].

Study area:
- Boundary file: [PATH, e.g., "./data/study_area.shp"]
- CRS: [EPSG, e.g., EPSG:4326 for storage, EPSG:32648 for
  display]
- Location: [DESCRIPTION, e.g., "Wuhan, Hubei Province, China"]

Map elements:
1. Main map:
   - Study area polygon: filled [COLOR, alpha=0.3] with
     solid border [COLOR, linewidth=1.5]
   - Basemap: [contextily OSM / cartopy stock_img / none]
   - Labels: [FEATURES to label, e.g., "Yangtze River",
     district names from 'NAME' column]
   - Coordinate grid with labels on axes

2. Inset locator map (corner: [POSITION]):
   - Country outline from Natural Earth 1:110m
   - Study area shown as red rectangle or point
   - Minimal styling, small size (~25% of main map)

3. Scale bar:
   - Position: [CORNER]
   - Length: auto-calculated appropriate round number
   - Style: alternating black-white boxes

4. North arrow:
   - Position: [CORNER]
   - Simple arrow with "N" label

5. Legend (if applicable):
   - [ITEMS, e.g., study boundary, sampling points, rivers]

Export:
- DPI: 300
- Format: [PDF / PNG / TIFF]
- Size: [WIDTH x HEIGHT inches, e.g., 7x5 single column]
- Background: white
- tight_layout with minimal margins

The script should:
- Handle CRS transformation automatically
- Include error handling for missing files
- Be runnable as: python study_area_map.py
- Produce camera-ready output with no manual editing
```

#### Variables to Customize

- `LIBRARY` -- cartopy vs. geopandas+contextily
- Boundary file path and CRS
- Location description and features to label
- Map element positions and styles
- Export format, DPI, and dimensions

#### Expected Output

A complete, runnable Python script (~80--120 lines) that produces a publication-quality study area map.

#### Validation Checklist

- [ ] Script runs without errors (all imports available via pip)
- [ ] CRS transformation is handled correctly
- [ ] Scale bar length is appropriate for the map extent
- [ ] Inset map shows the study area location in national/regional context
- [ ] Output is 300 DPI and publication-sized

#### Cost Optimization

Code generation benefits from a frontier model for correctness. However, study area maps are well-documented patterns -- a mid-tier model often suffices. Test the output before iterating.

> **Dark Arts Tip:** Include your actual `gdf.total_bounds` output in the prompt so the model can calculate appropriate scale bar length, figure size, and label positions without guessing.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- matplotlib, cartopy, geopandas, contextily

#### Extensibility

- Add multiple study areas (e.g., comparison cities) as sub-panels.
- Generate a companion script for a land cover or results map.
- Pair with **P13 (Map Figure Caption)** for the corresponding caption.

---

## 5. Peer Review Response (P17--P19)

---

### Prompt 17 -- Single Comment Response

#### Scenario (实际场景)

You received a reviewer comment and need to draft a professional, point-by-point response. The response must thank the reviewer, directly address the concern, describe what was changed, and maintain a constructive tone -- even when you disagree.

#### Roles

You are a senior researcher coaching a junior colleague on how to respond to peer review with professionalism and precision.

#### Prompt Template

```text
Draft a response to this reviewer comment from a peer-reviewed
GIS journal submission:

Reviewer comment:
"[PASTE_COMMENT -- the exact text from the review]"

Context:
- Paper topic: [BRIEF_TOPIC]
- The reviewer is questioning: [YOUR_INTERPRETATION]
- My position: [AGREE / PARTIALLY_AGREE / DISAGREE]

If AGREE:
- Changes made: [DESCRIBE_REVISIONS]
- New analysis added: [IF_ANY]
- New text added in: [SECTION/LINE_NUMBERS]

If PARTIALLY_AGREE or DISAGREE:
- My reasoning: [EXPLAIN]
- Supporting evidence: [DATA, LITERATURE, ANALYSIS]
- Compromise offered: [IF_ANY]

Response structure:
1. Thank the reviewer for the [specific aspect: insightful
   observation / important suggestion / careful reading]
2. Address the point directly (no evasion)
3. State exactly what changed: "In the revised manuscript,
   Section [X], Lines [Y-Z], we have..."
4. If new analysis was added, summarize the result briefly
5. If disagreeing, present evidence respectfully: "While we
   appreciate this perspective, our analysis suggests..."

Tone: professional, grateful, constructive.
Length: [BRIEF ~100 words / MODERATE ~200 words / DETAILED
~400 words] -- match the complexity of the comment.
```

#### Variables to Customize

- `PASTE_COMMENT` -- the reviewer's exact text
- `BRIEF_TOPIC` -- your paper's subject
- `YOUR_INTERPRETATION` -- what the reviewer is really asking
- Position (agree/partially agree/disagree) and supporting details
- Desired length

#### Expected Output

A single reviewer response paragraph (or multiple paragraphs for complex comments) ready for inclusion in the response letter.

#### Validation Checklist

- [ ] Opens with a thank-you that references the specific point
- [ ] Directly addresses the concern (no deflection)
- [ ] Manuscript changes are cited with section/line references
- [ ] Disagreements are supported with evidence, not opinion

#### Cost Optimization

Reviewer responses benefit from a strong model -- tone and diplomacy matter. Use Claude Opus or GPT-4o for contentious comments; a mid-tier model for straightforward ones.

> **Dark Arts Tip:** For comments you disagree with, ask the model to generate both an "agree" and a "disagree" version. Often the best response is a hybrid: acknowledge the reviewer's valid sub-point while respectfully defending your core approach.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- for referencing specific analyses in your response

#### Extensibility

- Batch into **P18 (Systematic Response Letter)** for the complete document.
- Chain with **P19 (Revision Strategy Planner)** to prioritize which comments to address first.

---

### Prompt 18 -- Systematic Response Letter

#### Scenario (实际场景)

You have received reviews from multiple reviewers and an editor, with dozens of individual comments. You need to produce a complete, formatted response letter that addresses every point systematically. This is the most stressful part of the revision process, and this prompt structures it into a manageable workflow.

#### Roles

You are an experienced journal author who has revised dozens of manuscripts, drafting a comprehensive response letter.

#### Prompt Template

```text
Generate a complete response-to-reviewers letter for my paper
"[PAPER_TITLE]" submitted to [JOURNAL].

Decision: [MAJOR_REVISION / MINOR_REVISION]

Editor comments:
1. "[EDITOR_COMMENT_1]" -> [MY_ACTION: AGREE: description /
   REBUT: reasoning]

Reviewer 1 comments:
1. "[R1_COMMENT_1]" -> [MY_ACTION]
2. "[R1_COMMENT_2]" -> [MY_ACTION]
3. "[R1_COMMENT_3]" -> [MY_ACTION]

Reviewer 2 comments:
1. "[R2_COMMENT_1]" -> [MY_ACTION]
2. "[R2_COMMENT_2]" -> [MY_ACTION]

[Add more reviewers as needed]

Letter structure:
1. **Opening paragraph**: Thank the editor and reviewers.
   Summarize the key improvements made. State that all
   comments have been addressed point-by-point.

2. **Per-reviewer sections**: For each comment:
   - *Original comment* (quoted in italics)
   - **Response:** Our substantive reply
   - **Changes:** Specific manuscript modifications with
     section and line references

3. **Closing paragraph**: Summarize the most significant
   changes. Express hope that the revision is satisfactory.

Format: Markdown with headers, suitable for PDF conversion.
Tone: professional, grateful, thorough.
Mark any responses you are unsure about with [REVIEW_NEEDED].
```

#### Variables to Customize

- `PAPER_TITLE` and `JOURNAL`
- Decision type (major/minor revision)
- All reviewer and editor comments with your planned actions
- Level of detail for each response

#### Expected Output

A complete response letter in Markdown, organized by reviewer, with quoted comments, responses, and change descriptions. Ready for conversion to PDF.

#### Validation Checklist

- [ ] Every single comment is addressed (none skipped)
- [ ] Opening paragraph sets a positive, constructive tone
- [ ] Each response includes specific manuscript locations for changes
- [ ] Disagreements are evidence-based and respectful
- [ ] `[REVIEW_NEEDED]` flags highlight uncertain responses

#### Cost Optimization

This is a high-value, long-output task. Use a frontier model. The cost (a few dollars) is negligible compared to the hours saved and the importance of getting the revision right.

> **Dark Arts Tip:** Process the comments in order of difficulty: easy wins first (typo fixes, clarifications), then moderate changes, then major disagreements. This builds momentum and gives you a nearly-complete letter before tackling the hardest parts.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- for referencing new analyses added during revision

#### Extensibility

- Generate a revision changelog (diff-style) showing all manuscript changes.
- Create a highlighted PDF of the revised manuscript using `latexdiff`.
- Chain with **P19 (Revision Strategy Planner)** before writing responses.

---

### Prompt 19 -- Revision Strategy Planner

#### Scenario (实际场景)

Before diving into revisions, you need to triage reviewer comments: which are quick fixes, which require new analysis, which conflict with each other, and which you plan to push back on. This prompt creates a prioritized revision plan that saves you from the common mistake of starting with the hardest comment and burning out.

#### Roles

You are a research project manager creating a revision action plan for a GIS manuscript.

#### Prompt Template

```text
Analyze the following reviewer comments for my paper
"[PAPER_TITLE]" and create a prioritized revision strategy.

All comments:
[PASTE_ALL_COMMENTS -- from all reviewers and editor,
numbered sequentially, e.g.:
E1: "The methodology section lacks detail on..."
R1.1: "The authors should compare with..."
R1.2: "Figure 3 is hard to read..."
R2.1: "The statistical analysis does not account for..."
R2.2: "Minor: typo on page 5..."]

Categorize each comment into:
1. **Quick fixes** (typos, formatting, clarifications):
   - Estimated time: <15 min each
   - Do these first for momentum

2. **Moderate revisions** (rewriting paragraphs, adding
   citations, improving figures):
   - Estimated time: 30-60 min each
   - Prioritize by impact on reviewer satisfaction

3. **Major revisions** (new analysis, significant rewriting,
   methodological changes):
   - Estimated time: 2-8 hours each
   - Identify dependencies (which must be done before others)

4. **Push-back candidates** (comments you may respectfully
   disagree with):
   - Draft a brief justification for each
   - Assess risk: will pushing back endanger acceptance?

5. **Conflicting comments** (where reviewers disagree with
   each other):
   - Identify the conflict
   - Suggest a resolution strategy

Output:
- A prioritized task list with time estimates
- A suggested revision order (not sequential by reviewer,
  but by efficiency)
- Total estimated revision time
- Risk assessment: which unaddressed comments could lead
  to rejection?
```

#### Variables to Customize

- `PAPER_TITLE` -- your paper
- `PASTE_ALL_COMMENTS` -- all comments from all reviewers, numbered
- Your initial reactions (optional: "I think R1.3 is wrong because...")

#### Expected Output

A structured revision plan with categorized comments, time estimates, prioritized order, and risk assessment.

#### Validation Checklist

- [ ] Every comment is categorized (none omitted)
- [ ] Time estimates are realistic (not all "15 minutes")
- [ ] Dependencies between revisions are identified
- [ ] Conflicting reviewer comments are flagged
- [ ] Risk assessment highlights potential rejection triggers

#### Cost Optimization

Strategic planning benefits from a strong reasoning model. Use Claude Opus or GPT-4o. The output is short and the cost is minimal, but the value is enormous.

> **Dark Arts Tip:** Ask the model: "Which single revision would most improve the paper's chance of acceptance?" This forces a prioritization insight that is often more valuable than the entire categorized list.

#### Related Pages

- [Tools: Python Libraries](../tools/python-libraries.md) -- for estimating effort on new analyses

#### Extensibility

- Feed the plan into **P18 (Systematic Response Letter)** as a skeleton.
- Track revision progress by checking off completed items.
- Generate a Gantt chart or timeline for multi-week revisions.

---

## 6. Chinese Academic Writing (P20--P22)

---

### Prompt 20 -- Chinese Abstract & Keywords (中文摘要)

#### Scenario (实际场景)

Many Chinese universities and some international journals require a Chinese-language abstract alongside the English manuscript. You need to produce academic Chinese that uses correct GIS terminology (地理信息系统, 遥感, 空间分析) and follows the conventions of Chinese geographic journals such as 地球信息科学学报, 遥感学报, or 地理学报.

#### Roles

You are a bilingual GIS academic (English/Chinese) who publishes in both Chinese and international journals.

#### Prompt Template

```text
Translate and adapt the following English abstract into
academic Chinese (学术中文) suitable for [TARGET:
"a Chinese journal submission" / "the Chinese abstract
section of an English-language paper" / "a Chinese
conference proceedings"].

English abstract:
"""
[PASTE_ENGLISH_ABSTRACT]
"""

Requirements:
1. Use standard GIS/RS terminology in Chinese:
   - GIS: 地理信息系统 (GIS)
   - Remote sensing: 遥感
   - Spatial analysis: 空间分析
   - Land use/land cover: 土地利用/土地覆盖 (LULC)
   - Machine learning: 机器学习
   - Deep learning: 深度学习
   - Accuracy assessment: 精度评价
   - Spatial autocorrelation: 空间自相关
   - Geographically weighted regression: 地理加权回归 (GWR)

2. Follow Chinese academic abstract conventions:
   - Structure: 研究背景 -> 方法 -> 结果 -> 结论
   - Length: [N] Chinese characters (typically 300-500)
   - Formal register (书面语, not 口语)
   - Include the English abbreviation in parentheses on
     first use: e.g., 归一化植被指数 (NDVI)

3. Generate 5-8 Chinese keywords (关键词):
   - Include both the Chinese term and English equivalent
   - Order: most specific to most general
   - Example: 城市热岛; 地表温度; 遥感; 空间分析

4. If targeting a specific Chinese journal, match its
   abstract format:
   - 地球信息科学学报: structured, ~400 chars
   - 遥感学报: concise, ~300 chars
   - 地理学报: detailed, ~500 chars
```

#### Variables to Customize

- `PASTE_ENGLISH_ABSTRACT` -- your English abstract
- `TARGET` -- Chinese journal, English paper's Chinese section, or conference
- Character count target
- Target Chinese journal (if applicable)

#### Expected Output

A Chinese abstract (300--500 characters) with properly formatted GIS terminology and 5--8 bilingual keywords.

#### Validation Checklist

- [ ] All GIS terms use standard Chinese translations with English abbreviations
- [ ] Abstract follows 研究背景-方法-结果-结论 structure
- [ ] Character count is within the target range
- [ ] Keywords are ordered from specific to general
- [ ] Register is formal academic Chinese (书面语)

#### Cost Optimization

Use a model with strong Chinese language capability (Claude Opus, GPT-4o). Smaller models may produce colloquial phrasing or incorrect technical terms. Worth the extra cost for accuracy.

> **Dark Arts Tip:** Provide the model with 2-3 example Chinese abstracts from your target journal. Chinese academic conventions vary significantly by journal, and examples are far more effective than rules for matching the expected style.

#### Related Pages

- [Data Sources](../data-sources/) -- Chinese data portals (国家地球系统科学数据中心, 地理空间数据云)

#### Extensibility

- Chain with **P22 (Bilingual Paper Preparation)** for full dual-language submission.
- Generate a Chinese title (中文题目) alongside the abstract.
- Produce a Chinese author contribution statement (作者贡献).

---

### Prompt 21 -- Chinese Methodology Translation

#### Scenario (实际场景)

You have written your methodology in English and need to translate it into Chinese for a Chinese journal submission or for a bilingual dissertation. The challenge is not just translation but using the correct Chinese technical terminology for GIS methods, software, and statistical procedures -- terminology that varies between mainland China, Taiwan, and Hong Kong conventions.

#### Roles

You are a bilingual geospatial scientist translating a methodology section from English to mainland Chinese academic prose.

#### Prompt Template

```text
Translate the following English methodology section into
academic Chinese (学术中文) for submission to [CHINESE_JOURNAL,
e.g., 地球信息科学学报].

English methodology:
"""
[PASTE_ENGLISH_METHODOLOGY]
"""

Translation requirements:
1. Technical terminology (use mainland China conventions):
   - Random Forest: 随机森林
   - Support Vector Machine: 支持向量机
   - Convolutional Neural Network: 卷积神经网络
   - Principal Component Analysis: 主成分分析
   - Kriging interpolation: 克里金插值
   - Moran's I: 莫兰指数 (Moran's I)
   - Confusion matrix: 混淆矩阵
   - Producer's/User's accuracy: 制图精度/用户精度
   - Overall accuracy: 总体精度
   - Kappa coefficient: Kappa 系数

2. Software and data source names:
   - Keep English names for software: Python, R, QGIS, ArcGIS
   - Translate data source descriptions but keep sensor names
     in English: Sentinel-2, Landsat, MODIS
   - Chinese data sources use Chinese names: 高分系列卫星,
     资源系列卫星, 天地图

3. Mathematical notation:
   - Keep LaTeX equations unchanged
   - Translate surrounding text: "where $x_i$ represents..."
     -> "其中 $x_i$ 表示..."

4. Style:
   - Passive constructions: 本研究采用... / ...被用于...
   - Past tense equivalent: 对...进行了分析
   - Formal register throughout

5. Output a terminology mapping table:
   | English Term | Chinese Term | Pinyin |
   For all technical terms used in the translation.

Mark uncertain translations with [术语待确认].
```

#### Variables to Customize

- `PASTE_ENGLISH_METHODOLOGY` -- your English methods section
- `CHINESE_JOURNAL` -- target Chinese journal
- Regional convention preference (mainland/Taiwan/Hong Kong)
- Whether to include pinyin in the terminology table

#### Expected Output

A translated Chinese methodology section with LaTeX equations preserved, plus a terminology mapping table. Uncertain terms are flagged with `[术语待确认]`.

#### Validation Checklist

- [ ] All technical terms use standard mainland Chinese translations
- [ ] Software names remain in English
- [ ] LaTeX equations are preserved unchanged
- [ ] Terminology mapping table is complete
- [ ] Uncertain translations are flagged

#### Cost Optimization

Translation requires a strong bilingual model. Use Claude Opus or GPT-4o. Do not use a small model for technical Chinese translation -- errors in terminology are difficult to catch if you are not fluent.

> **Dark Arts Tip:** Run the Chinese output through a second prompt asking: "Check this Chinese academic text for any colloquial expressions, incorrect technical terms, or awkward phrasing. List all issues." This self-review catch catches errors that a single-pass translation misses.

#### Related Pages

- [Data Sources](../data-sources/) -- Chinese satellite data sources (高分, 资源, 环境)
- [Tools: Python Libraries](../tools/python-libraries.md) -- library names to keep in English

#### Extensibility

- Chain with **P20 (Chinese Abstract)** for the abstract.
- Generate a Chinese-language figure caption set.
- Produce a Chinese literature review translation using the same terminology table.

---

### Prompt 22 -- Bilingual Paper Preparation

#### Scenario (实际场景)

Some journals and all Chinese doctoral dissertations require both English and Chinese versions of the paper (or at minimum, abstract, keywords, and figure captions in both languages). This prompt manages the bilingual preparation process, ensuring consistent terminology across both languages and producing a reusable terminology mapping table.

#### Roles

You are a bilingual academic editor preparing a GIS manuscript for dual-language submission.

#### Prompt Template

```text
Prepare bilingual (English + Chinese) components for my GIS
paper "[PAPER_TITLE]" / "[中文题目]".

Submission target: [TARGET, e.g., "Chinese journal requiring
full bilingual submission" / "English journal with Chinese
abstract requirement" / "Doctoral dissertation with bilingual
abstract"]

Components to produce:
1. **Title**: English and Chinese (中文题目)
2. **Abstract**: English (~250 words) and Chinese (~400 chars)
3. **Keywords**: 5-8 bilingual pairs
4. **Figure captions**: bilingual for all [N] figures
5. **Table captions**: bilingual for all [N] tables

English paper content:
"""
[PASTE_KEY_SECTIONS -- abstract, figure/table captions,
or full paper if translating everything]
"""

Terminology mapping table (reusable across sections):
| # | English | Chinese | Pinyin | Notes |
|---|---------|---------|--------|-------|
| 1 | [TERM]  | [翻译]   | [拼音]  | [备注] |

Requirements:
- Terminology must be consistent across ALL components
  (same Chinese term for the same English term everywhere)
- Follow [CHINESE_JOURNAL] conventions for Chinese text
- Follow [ENGLISH_JOURNAL] conventions for English text
- Flag any terms where multiple Chinese translations exist
  with [多译, 推荐: X]
- Include domain-specific terms:
  GIS 地理信息系统, RS 遥感, DEM 数字高程模型,
  LULC 土地利用/土地覆盖, POI 兴趣点

Output order:
1. Terminology mapping table (use this as the reference)
2. Bilingual title
3. Bilingual abstract
4. Bilingual keywords
5. Bilingual figure captions
6. Bilingual table captions
```

#### Variables to Customize

- `PAPER_TITLE` / `中文题目` -- paper title in both languages
- `TARGET` -- submission type (journal, dissertation, conference)
- Paper content to translate
- `CHINESE_JOURNAL` / `ENGLISH_JOURNAL` -- for style matching
- Number of figures and tables

#### Expected Output

A complete bilingual package: terminology table, title, abstract, keywords, and all figure/table captions in both languages, with consistent terminology throughout.

#### Validation Checklist

- [ ] Terminology is consistent across all components (spot-check 3 terms)
- [ ] Chinese abstract character count is within target range
- [ ] Keywords appear in matching order in both languages
- [ ] Figure/table captions match their English counterparts in content
- [ ] Ambiguous terms are flagged with `[多译, 推荐: X]`

#### Cost Optimization

This is a comprehensive task requiring a frontier model with strong bilingual capability. Run it as a single prompt to ensure cross-component consistency. Cost: moderate, but saves hours of manual coordination.

> **Dark Arts Tip:** Create the terminology mapping table first (as a standalone prompt), review and approve it, then feed it into the bilingual generation prompt as a constraint. This two-step approach ensures terminology accuracy before the expensive generation step.

#### Related Pages

- [Data Sources](../data-sources/) -- Chinese and international data source names
- [Tools: Python Libraries](../tools/python-libraries.md) -- library names (always English)

#### Extensibility

- Expand to full paper translation for Chinese journal submission.
- Generate a glossary appendix for the dissertation.
- Create a bilingual presentation (PPT) outline for the defense.

---

[Back to AI Prompts](README.md) · [Back to Main README](../README.md)
