# Research Tools

A curated collection of tools for every stage of the GIS research workflow -- from discovering relevant papers to sharing reproducible results. This guide covers literature search, reference management, collaboration platforms, data sharing, and reproducibility tools.

> **Quick Picks**
> - **AI-powered literature search:** Elicit or Consensus (answer questions from papers)
> - **Paper discovery graph:** Connected Papers or Research Rabbit (visual exploration)
> - **Citation search + API:** Semantic Scholar (free API, TLDR summaries)
> - **Reference manager:** Zotero + Better BibTeX (free, LaTeX-native)
> - **Data archival with DOI:** Zenodo (free, GitHub integration)

---

## Literature Discovery

Finding relevant papers efficiently is the foundation of good research. These tools help you discover, track, and explore the GIS literature.

| Tool | URL | Purpose | Cost |
|------|-----|---------|------|
| Google Scholar | https://scholar.google.com | Broad academic search, citation tracking | Free |
| Semantic Scholar | https://www.semanticscholar.org | AI-powered search, citation context, TLDR | Free |
| Connected Papers | https://www.connectedpapers.com | Visual graph of related papers | Free (limited) |
| Lens.org | https://www.lens.org | Scholarly search, patent search, open metadata | Free |
| Web of Science | https://www.webofscience.com | Citation indexing, journal impact factors | Institutional |
| Scopus | https://www.scopus.com | Citation database, author profiles | Institutional |
| Dimensions | https://www.dimensions.ai | Grants + papers + patents linked | Free (limited) |
| ResearchRabbit | https://www.researchrabbit.ai | AI-assisted paper discovery, alerts | Free |
| [Elicit](https://elicit.com) | https://elicit.com | AI research assistant, extracts claims from papers | Free (limited) |
| [Consensus](https://consensus.app) | https://consensus.app | AI-powered answers from academic papers | Free (limited) |
| [Semantic Scholar API](https://api.semanticscholar.org/) | https://api.semanticscholar.org | Programmatic access to 200M+ papers | Free |

### AI-Powered Research Tools

- **Elicit**: Ask a research question, get answers synthesized from papers with citations. Useful for literature reviews: "What methods are used for urban heat island detection?"
- **Consensus**: Similar to Elicit but focused on finding scientific consensus. Good for: "Does NDVI correlate with land surface temperature?"
- **Semantic Scholar API**: Build custom literature analysis pipelines. Query by keywords, author, venue, citation count.

```python
# Semantic Scholar API example
import requests
resp = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",
    params={"query": "GIS spatial autocorrelation urban", "limit": 10,
            "fields": "title,year,citationCount,tldr"})
papers = resp.json()["data"]
```

### Zotero Plugins for GIS Researchers

| Plugin | Purpose | Install |
|---|---|---|
| [Better BibTeX](https://retorque.re/zotero-better-bibtex/) | Auto-generate citation keys, export .bib | Zotero plugin |
| [Zotero PDF Translate](https://github.com/windingwind/zotero-pdf-translate) | Translate papers in-app (useful for non-English papers) | Zotero plugin |
| [ZotFile](http://zotfile.com/) | Auto-rename and organize PDF files | Zotero plugin |
| [Zotero Connector](https://www.zotero.org/download/connectors) | Save papers from browser with one click | Browser extension |
| [Zotero GPT](https://github.com/MuiseDestiny/zotero-gpt) | AI-powered paper analysis within Zotero | Zotero plugin |

### Tips for GIS Literature Search

- Combine geographic terms with method terms (e.g., "spatial autocorrelation" + "urban heat island")
- Use Semantic Scholar's "TLDR" feature to quickly scan relevance
- Set up Google Scholar alerts for your key research topics
- Use Connected Papers to find seminal papers and recent developments in a subfield

### Example Search Strategies

| Research Topic | Search Terms | Good Journals |
|---|---|---|
| Urban heat island | "urban heat island" AND ("land surface temperature" OR LST) AND (GIS OR remote sensing) | RSE, Sustainable Cities |
| Land use change | "land use change" AND ("change detection" OR "post-classification") AND Sentinel-2 | ISPRS, RSE, JAG |
| Spatial accessibility | "spatial accessibility" AND ("two-step floating catchment" OR isochrone) AND health | IJGIS, Health & Place |
| Deep learning + RS | "deep learning" AND ("semantic segmentation" OR "object detection") AND "remote sensing" | TGRS, ISPRS, RSE |

---

## Reference Management

Organize, annotate, and cite your references efficiently.

| Tool | Cost | Best Feature | LaTeX Support | Browser Extension |
|------|------|-------------|---------------|-------------------|
| Zotero | Free | Group libraries, Better BibTeX, open source | Excellent (via Better BibTeX) | Yes |
| Mendeley | Free | PDF annotation, social discovery | Good | Yes |
| Paperpile | $2.99/mo (academic) | Google Docs native integration | Good | Yes |
| EndNote | $249.95 (institutional) | Largest journal style library | Good | Yes |
| JabRef | Free | BibTeX-native, open source | Native | No |

### Recommended Setup: Zotero + Better BibTeX

1. Install Zotero (https://www.zotero.org)
2. Install Better BibTeX plugin (https://retorque.re/zotero-better-bibtex/)
3. Install Zotero Connector for your browser
4. Configure citation key format (e.g., `auth.lower + year`)
5. Set up auto-export of `.bib` file to your project directory
6. Organize collections by project or paper section

### Building a GIS Reference Library

- Create collections for: Methods, Study Area Background, Data Sources, Comparison Papers
- Tag papers by methodology: `remote-sensing`, `machine-learning`, `spatial-statistics`, etc.
- Use Zotero's "Related" feature to link papers comparing similar methods
- Share group libraries with your lab or co-authors


---

## Collaboration

Tools for co-authoring papers, managing projects, and sharing work.

| Tool | Purpose | Cost | Best For |
|------|---------|------|----------|
| Overleaf | Collaborative LaTeX editing | Free (limited), $15/mo (standard) | LaTeX papers, real-time collaboration |
| GitHub | Version control, issue tracking | Free | Code + paper (LaTeX), open science |
| OSF (Open Science Framework) | Project management, preregistration | Free | Full project lifecycle, preprints |
| Google Docs | Real-time collaborative writing | Free | Early drafts, non-LaTeX workflows |
| HackMD / CodiMD | Collaborative Markdown editing | Free / self-hosted | Meeting notes, quick drafts |
| Notion | Project wiki, task tracking | Free (personal) | Lab management, project planning |

### GitHub for Academic Papers

Using Git for paper writing provides version control, collaboration tools, and reproducibility:

```
my-paper/
  ├── paper/
  │   ├── main.tex
  │   ├── figures/
  │   └── references.bib
  ├── code/
  │   ├── analysis.py
  │   └── figures.py
  ├── data/
  │   └── README.md  (data access instructions)
  ├── environment.yml
  └── README.md
```

### GitHub Actions for LaTeX Compilation

```yaml
name: Build LaTeX Paper
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Compile LaTeX
        uses: xu-cheng/latex-action@v3
        with:
          root_file: paper/main.tex
      - name: Upload PDF
        uses: actions/upload-artifact@v4
        with:
          name: paper
          path: paper/main.pdf
```

---

## Data Sharing

Making your research data accessible is increasingly required by journals and funders.

| Platform | Purpose | Cost | DOI | Storage Limit |
|----------|---------|------|-----|---------------|
| Zenodo | General research data, linked to GitHub | Free | Yes | 50 GB per dataset |
| Figshare | Figures, datasets, media | Free (5 GB private) | Yes | 20 GB per file (public) |
| Dryad | Biodiversity and ecology data | $150 per submission | Yes | Unlimited (reasonable) |
| Pangaea | Earth and environmental science | Free | Yes | No strict limit |
| HydroShare | Hydrological data and models | Free | Yes | No strict limit |
| OpenTopography | LiDAR point cloud data | Free | Yes | Specialized |

### Data Sharing Best Practices

- Use standard formats: GeoPackage, GeoJSON, GeoTIFF, GeoParquet
- Include metadata (CRS, attribute descriptions, provenance)
- Provide a README explaining data structure and access
- Choose a permissive license (CC BY 4.0 recommended)
- Link your data DOI in your paper's data availability statement

### Data Sharing Checklist

- [ ] Data in standard open format (GeoPackage, GeoJSON, GeoTIFF, GeoParquet)
- [ ] Metadata included (CRS, attribute descriptions, units, provenance)
- [ ] README.md describing structure and how to use
- [ ] License specified (CC BY 4.0 recommended)
- [ ] DOI assigned (Zenodo, Figshare)
- [ ] Linked from paper's Data Availability Statement

---

## Reproducibility

Ensuring others can reproduce your results is a cornerstone of good science. These tools help you create reproducible computational environments.

| Tool | Purpose | Complexity | Best For |
|------|---------|-----------|----------|
| Binder (mybinder.org) | Run Jupyter notebooks in the cloud | Low | Demos, simple analyses |
| Docker | Containerized reproducible environments | Medium | Complex multi-tool workflows |
| conda / mamba | Python/R environment management | Low | Python/R GIS analysis |
| renv (R) | R package environment management | Low | R-based analysis |
| Poetry (Python) | Python dependency management | Low | Python projects |
| CyberGIS-Jupyter | Cloud-based GIS Jupyter environment | Low | Large-scale spatial analysis |
| Whole Tale | Reproducible research packages | Low | Full paper + code + data bundles |

### Reproducible GIS Analysis Workflow

1. **Environment**: Define dependencies in `environment.yml` (conda) or `Dockerfile`
2. **Data**: Host data on Zenodo/Figshare with DOI, or provide download scripts
3. **Code**: Version control with Git, organize scripts logically
4. **Workflow**: Use Makefiles, Snakemake, or Jupyter notebooks to define execution order
5. **Test**: Verify the full pipeline runs from scratch on a clean environment
6. **Share**: Publish code on GitHub/GitLab, link from paper, add Binder badge

### Example `environment.yml` for GIS Research

```yaml
name: gis-research
channels:
  - conda-forge
dependencies:
  - python=3.11
  - geopandas
  - rasterio
  - shapely
  - pyproj
  - matplotlib
  - scikit-learn
  - jupyter
  - contextily
```

### Dockerfile for GIS Research

```dockerfile
FROM quay.io/jupyter/scipy-notebook:latest
RUN pip install geopandas rasterio shapely pyproj \
    folium contextily pysal scikit-learn xarray rioxarray
```

### Binder Configuration

Add a `binder/environment.yml` to your GitHub repo, then visit `https://mybinder.org/v2/gh/username/repo/main` to launch.

---

## Tool Comparison Summary

| Tool | Purpose | Cost | Integration | Best For |
|------|---------|------|-------------|----------|
| Google Scholar | Literature search | Free | Alerts, library | Broad discovery |
| Semantic Scholar | AI-powered search | Free | API, alerts | Focused discovery |
| Connected Papers | Visual paper graph | Free (limited) | Export | Exploring subfields |
| Zotero | Reference management | Free | LaTeX, Word, Docs | Most researchers |
| Overleaf | LaTeX collaboration | Free/Paid | Git, Zotero | Co-authored papers |
| GitHub | Code + paper versioning | Free | CI/CD, Zenodo | Open science |
| Zenodo | Data/code archival | Free | GitHub, DOI | Data sharing |
| Docker | Reproducible environments | Free | CI/CD, Binder | Complex workflows |
| Binder | Cloud notebooks | Free | GitHub, Zenodo | Quick demos |
| CyberGIS | Cloud GIS computing | Free | JupyterHub | Large-scale analysis |
### Integrated Research Workflow

```
Discover papers (Semantic Scholar, Elicit)
    |
    v
Save to Zotero (browser connector)
    |
    v
Read & annotate PDFs (Zotero built-in reader)
    |
    v
Write paper (Overleaf or VS Code + LaTeX)
    |
    v
Cite references (Better BibTeX auto-export .bib)
    |
    v
Share code (GitHub) + Archive data (Zenodo)
    |
    v
Submit to journal + Post preprint (EarthArXiv)
```
