# Research Tools for GIS Academics

A curated collection of tools for every stage of the GIS research workflow — from AI-powered literature discovery and citation graph exploration to reference management, research data management, lab collaboration, academic social networks, and research impact metrics.

> **Quick Picks**
> - **AI literature search:** Elicit (answer research questions from papers)
> - **Citation graph:** Connected Papers or Litmaps (visual exploration)
> - **Paper discovery API:** Semantic Scholar (free, 200M+ papers)
> - **Reference manager:** Zotero + Better BibTeX (free, open source)
> - **Research assistant:** Perplexity Pro or Claude (AI-powered deep research)
> - **Data archival:** Zenodo (free, GitHub integration, DOI)

---

## AI-Powered Literature Discovery (2025)

### AI Research Assistants

| Tool | URL | Purpose | Cost | Key Feature |
|------|-----|---------|------|-------------|
| Elicit | https://elicit.com | Extract claims from papers | Free (limited) / $12/mo | Systematic review automation |
| Consensus | https://consensus.app | Find scientific consensus | Free (limited) / $10/mo | Yes/No answers with evidence |
| Semantic Scholar | https://www.semanticscholar.org | AI-powered search | Free | TLDR summaries, citation context |
| Perplexity | https://www.perplexity.ai | AI search with sources | Free / $20/mo | Real-time web + academic search |
| SciSpace (Typeset) | https://typeset.io | Read + understand papers | Free / $10/mo | Paper explanations, chat with PDF |
| Undermind | https://www.undermind.ai | Deep literature search | Free (limited) | Iterative search refinement |
| Scite.ai | https://scite.ai | Smart citation context | $15/mo | Shows if papers support/contrast claims |
| Iris.ai | https://iris.ai | Research workspace | Contact sales | Patent + paper analysis |

### How to Use AI for Literature Review

```
Step 1: Broad Discovery
  → Use Elicit: "What methods detect urban heat islands from satellite data?"
  → Returns: Key papers with extracted methods, results, limitations

Step 2: Citation Graph Exploration
  → Pick 3-5 seminal papers from Step 1
  → Use Connected Papers to find related clusters
  → Use Litmaps for temporal evolution

Step 3: Deep Reading
  → Use SciSpace to chat with complex papers
  → Use Semantic Scholar TLDR for quick screening

Step 4: Systematic Search
  → Use Elicit's systematic review feature
  → Define inclusion/exclusion criteria
  → Extract structured data from papers automatically

Step 5: Synthesis
  → Use Claude/GPT to help synthesize findings
  → ALWAYS verify every citation manually
```

### Elicit for GIS Research — Practical Examples

```
# Question-based search (best mode)
"What spatial resolution is needed for urban heat island mapping?"
→ Returns papers with specific resolution recommendations

"How does spatial autocorrelation affect machine learning for land use?"
→ Returns papers discussing SAC issues with specific solutions

"What accuracy do deep learning methods achieve for building extraction?"
→ Returns papers with OA%, IoU, F1 scores extracted automatically
```

### Semantic Scholar API (Programmatic Access)

```python
import requests

# Search papers
resp = requests.get(
    "https://api.semanticscholar.org/graph/v1/paper/search",
    params={
        "query": "GIS urban heat island deep learning",
        "limit": 20,
        "fields": "title,year,citationCount,tldr,openAccessPdf,authors"
    }
)
papers = resp.json()["data"]

for p in papers:
    tldr = p.get("tldr", {}).get("text", "No TLDR")
    print(f"[{p['year']}] {p['title']} (cited: {p['citationCount']})")
    print(f"  TLDR: {tldr}\n")

# Get citation context (who cites whom and how)
resp = requests.get(
    f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations",
    params={"fields": "title,citationCount,contexts"}
)
```

### Google Scholar Advanced Search Tips

```
# Exact phrase search
"spatial autocorrelation" "random forest"

# Author search
author:"tobler" spatial

# Date range
"urban heat island" "deep learning" 2022..2025

# Exclude patents and citations
Settings → uncheck "Include patents" and "Include citations"

# Specific journal
source:"Remote Sensing of Environment" "land surface temperature"

# Set up alerts
Scholar → ☆ → Create alert → Email when new papers match
```

---

## Citation Graph & Discovery Tools

| Tool | URL | Purpose | Cost | Method |
|------|-----|---------|------|--------|
| Connected Papers | https://www.connectedpapers.com | Visual graph of related papers | Free (5/mo) / $5/mo | Co-citation similarity |
| Research Rabbit | https://www.researchrabbit.ai | AI paper discovery, alerts | Free | Citation + content analysis |
| Litmaps | https://www.litmaps.com | Interactive citation maps | Free (limited) / $10/mo | Direct citations, temporal |
| Citation Gecko | https://www.citationgecko.com | Find missing citations | Free | Seed papers → recommendations |
| Inciteful | https://inciteful.xyz | Paper network analysis | Free | Citation graph analytics |
| Open Knowledge Maps | https://openknowledgemaps.org | Visual overview of research topics | Free | PubMed + BASE search |

### Connected Papers Workflow

```
1. Paste a key paper's DOI or title
2. Explore the generated graph:
   - Larger nodes = more citations
   - Closer nodes = more similar
   - Color = publication year
3. Identify clusters:
   - Dense cluster = well-established subfield
   - Bridge papers = connecting different approaches
4. Find:
   - "Prior work" tab → foundational papers
   - "Derivative work" tab → latest developments
5. Repeat with 3-5 seed papers to cover the field
```

### Research Rabbit vs. Connected Papers

| Feature | Connected Papers | Research Rabbit |
|---------|-----------------|-----------------|
| Input | Single paper | Collection of papers |
| Output | Similarity graph | Related papers, alerts |
| Free tier | 5 graphs/month | Unlimited |
| Alerts | No | Yes (new paper alerts) |
| Collaboration | No | Shared collections |
| Best for | Exploring one subfield | Building literature library |

---

## Reference Management

### Tool Comparison (2025)

| Tool | Cost | Open Source | LaTeX | Word | Google Docs | Collaboration |
|------|------|------------|-------|------|-------------|---------------|
| Zotero | Free | ✅ | Excellent | ✅ | Via plugin | Group libraries |
| Mendeley | Free | ❌ | Good | ✅ | ❌ | Groups |
| Paperpile | $2.99/mo | ❌ | Good | ✅ | ✅ (native) | Shared folders |
| EndNote | $249.95 | ❌ | Good | ✅ | ❌ | Via library |
| JabRef | Free | ✅ | Native | ❌ | ❌ | Via Git |
| Citavi | Free (Windows) | ❌ | Good | ✅ | ❌ | Teams |

### Zotero Essential Setup

```
1. Install Zotero 7 (https://www.zotero.org)
2. Install Better BibTeX (https://retorque.re/zotero-better-bibtex/)
3. Install Zotero Connector browser extension
4. Configure:
   - Better BibTeX citation key: [auth:lower][year]
   - Auto-export .bib to LaTeX project folder
   - Sync: Free 300 MB or WebDAV for more
   - File syncing: Use ZotMoov for local storage

5. Essential plugins (2025):
   - Better BibTeX → citation keys + .bib export
   - Zotero PDF Translate → translate papers in-app
   - Zotero GPT / Aria → AI paper analysis
   - ZotMoov → auto-rename and organize PDFs
   - Better Notes → enhanced note-taking
   - Zotero Citation Counts → show citation metrics
   - Zotero Style → custom column display
```

### Zotero Collection Organization for GIS

```
📁 My GIS Research/
  ├── 📁 Current Paper — [Title]/
  │   ├── 📁 Introduction/
  │   ├── 📁 Related Work/
  │   │   ├── 📁 Method A papers/
  │   │   └── 📁 Method B papers/
  │   ├── 📁 Study Area Background/
  │   ├── 📁 Methodology References/
  │   ├── 📁 Comparison Baselines/
  │   └── 📁 Discussion/
  ├── 📁 General GIS Theory/
  ├── 📁 Remote Sensing Methods/
  ├── 📁 Spatial Statistics/
  ├── 📁 Deep Learning + GIS/
  ├── 📁 Software & Tools/
  └── 📁 To Read/
```

---

## Traditional Literature Databases

| Database | URL | Scope | Access |
|----------|-----|-------|--------|
| Google Scholar | https://scholar.google.com | Broadest coverage | Free |
| Web of Science | https://www.webofscience.com | Citation indexing, JCR | Institutional |
| Scopus | https://www.scopus.com | Citation database, h-index | Institutional |
| Dimensions | https://www.dimensions.ai | Papers + grants + patents | Free (limited) |
| Lens.org | https://www.lens.org | Scholarly + patent search | Free |
| BASE | https://www.base-search.net | OA aggregator | Free |
| CORE | https://core.ac.uk | Full-text OA papers | Free |
| OpenAlex | https://openalex.org | Open scholarly metadata | Free (API) |
| CrossRef | https://www.crossref.org | DOI metadata | Free (API) |
| CNKI (知网) | https://www.cnki.net | Chinese academic database | Institutional |
| Wanfang (万方) | https://www.wanfangdata.com.cn | Chinese journals + theses | Institutional |

### OpenAlex — The Open Alternative to Scopus/WoS

OpenAlex is a free, open catalog of the global research system (https://openalex.org).

```python
import requests

# Search for GIS works
resp = requests.get("https://api.openalex.org/works", params={
    "search": "urban heat island GIS remote sensing",
    "filter": "from_publication_date:2022-01-01",
    "sort": "cited_by_count:desc",
    "per_page": 10
})
works = resp.json()["results"]

for w in works:
    print(f"[{w['publication_year']}] {w['title']}")
    print(f"  Cited: {w['cited_by_count']}, OA: {w['open_access']['is_oa']}")
    print(f"  DOI: {w['doi']}\n")

# Author profile
resp = requests.get("https://api.openalex.org/authors", params={
    "search": "Michael Goodchild"
})
```

---

## Full-Text Access

### Legal Ways to Access Papers

| Method | How | Cost |
|--------|-----|------|
| Institutional access | Via university library proxy/VPN | Free (with affiliation) |
| Unpaywall | Browser extension, finds OA versions | Free |
| CORE | Aggregates OA full texts | Free |
| PubMed Central | Biomedical/health OA repository | Free |
| Google Scholar "All versions" | Click to find free versions | Free |
| Author personal page | Many authors self-archive | Free |
| Email the author | Polite request for PDF | Free |
| Interlibrary loan (ILL) | Request through your library | Free (usually) |
| Open Access Button | Finds or requests OA versions | Free |
| Sherpa Romeo | Check self-archiving policies | Free (info only) |

### Unpaywall Browser Extension

```
Install: https://unpaywall.org/products/extension
Works on: Chrome, Firefox

When you visit a paper on a publisher's website:
- Green lock icon → Free legal version available
- Click to access the OA version
- Checks: PubMed Central, institutional repos, preprint servers
```

---

## Research Data Management

### Data Management Plan (DMP)

Most funders now require a DMP. Use DMPTool (https://dmptool.org) to create one.

#### DMP Components

| Component | Content |
|-----------|---------|
| Data description | Types, formats, estimated size |
| Data collection | Methods, tools, quality control |
| Data standards | Metadata standards, naming conventions |
| Data sharing | When, where, how, restrictions |
| Data preservation | Long-term storage, archival |
| Responsibilities | Who manages data at each stage |
| Budget | Storage, curation, publishing costs |

### Metadata Standards for GIS Data

| Standard | Scope | Tool |
|----------|-------|------|
| ISO 19115/19139 | Geographic metadata (XML) | GeoNetwork, pycsw |
| FGDC CSDGM | US geospatial metadata | USGS tools |
| Dublin Core | General research metadata | Most repositories |
| DataCite | DOI metadata | Zenodo, Figshare |
| STAC | Spatiotemporal asset metadata | stac-utils |
| CF Conventions | Climate/forecast NetCDF | NCO, CDO |

### GIS Data Naming Convention

```
[project]_[datatype]_[area]_[date]_[version].[ext]

Examples:
uhi_lst_beijing_202407_v1.tif
transit_stops_newyork_2024_v2.gpkg
ndvi_timeseries_amazon_2020-2024_v1.parquet
```

---

## Lab & Team Collaboration

### Collaboration Platform Comparison

| Tool | Purpose | Cost | Best For |
|------|---------|------|----------|
| Overleaf | Collaborative LaTeX | Free / $15/mo | Co-authored papers |
| Typst.app | Collaborative typesetting | Free | Modern alternative to Overleaf |
| GitHub | Version control, issues | Free | Code + paper (LaTeX/Typst), open science |
| GitLab | Same as GitHub, self-hosted option | Free | Self-hosted, CI/CD |
| OSF | Project management, preregistration | Free | Full project lifecycle |
| Google Docs | Real-time writing | Free | Early drafts, comments |
| Notion | Project wiki, tasks | Free (personal) | Lab management |
| Slack | Team communication | Free / $8/mo | Daily communication |
| Mattermost | Self-hosted Slack | Free | Privacy-sensitive labs |
| HackMD | Collaborative Markdown | Free | Meeting notes, quick docs |
| Zotero Groups | Shared references | Free | Lab reference library |

### GitHub for Academic Research

```
my-research-project/
├── paper/
│   ├── main.tex (or main.typ for Typst)
│   ├── sections/
│   │   ├── introduction.tex
│   │   ├── methods.tex
│   │   └── results.tex
│   ├── figures/
│   │   ├── fig1_study_area.pdf
│   │   └── fig2_results.pdf
│   └── references.bib
├── code/
│   ├── 01_preprocessing.py
│   ├── 02_analysis.py
│   ├── 03_visualization.py
│   └── utils/
├── data/
│   ├── raw/          → .gitignore (large files)
│   ├── processed/    → .gitignore
│   └── README.md     → Data access instructions
├── environment.yml   → Conda environment
├── Dockerfile        → Reproducible environment
├── Makefile          → Build automation
├── LICENSE           → MIT or Apache-2.0
└── README.md         → Project overview
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
          name: paper-pdf
          path: paper/main.pdf
      - name: Release PDF on tag
        if: startsWith(github.ref, 'refs/tags/')
        uses: softprops/action-gh-release@v2
        with:
          files: paper/main.pdf
```

---

## Academic Social Networks & Profiles

### Essential Academic Profiles

| Platform | URL | Purpose | Priority |
|----------|-----|---------|----------|
| ORCID | https://orcid.org | Unique author identifier | **Must have** |
| Google Scholar | https://scholar.google.com/citations | Citation tracking, h-index | **Must have** |
| ResearchGate | https://www.researchgate.net | Paper sharing, Q&A, networking | Recommended |
| Web of Science Researcher ID | Via Clarivate | WoS citation linking | If using WoS |
| Scopus Author ID | Via Scopus | Scopus citation linking | Automatic |
| LinkedIn | https://www.linkedin.com | Professional networking | Recommended |
| X/Twitter | https://x.com | Academic communication | Optional (field-dependent) |
| Mastodon | https://mastodon.social | Decentralized academic community | Growing |
| Bluesky | https://bsky.app | Academic Twitter alternative | Growing |

### ORCID — Why It Matters

```
ORCID = Open Researcher and Contributor ID
- Unique persistent identifier (like DOI for researchers)
- Required by most journals during submission
- Links all your publications across name changes, affiliations
- Format: https://orcid.org/0000-0002-XXXX-XXXX
- Free to create and maintain

Setup checklist:
□ Create ORCID account
□ Link to institutional email
□ Import existing publications (from Scopus, CrossRef)
□ Add education and employment
□ Set profile to public
□ Add to all manuscript submissions
□ Link to Google Scholar, ResearchGate
```

### Research Profile Optimization

```
Google Scholar Profile:
1. Create profile at scholar.google.com
2. Add all your papers (auto-detection + manual)
3. Merge duplicate entries
4. Set up email alerts for new citations
5. Monitor h-index and i10-index
6. Set co-author list

ResearchGate:
1. Upload all papers (full text where allowed)
2. Add research interests and skills
3. Follow researchers in your field
4. Answer questions in your expertise area
5. Share datasets and code
```

---

## Research Impact & Metrics

### Author-Level Metrics

| Metric | What It Measures | Where to Find |
|--------|-----------------|---------------|
| h-index | Papers with ≥ h citations | Google Scholar, Scopus, WoS |
| i10-index | Papers with ≥ 10 citations | Google Scholar |
| Total citations | Overall impact | Google Scholar, Scopus |
| Field-weighted citation impact | Citations vs. field average | SciVal (Scopus) |
| RG Score | ResearchGate engagement | ResearchGate |

### Article-Level Metrics

| Metric | What It Measures | Source |
|--------|-----------------|--------|
| Citation count | Academic impact | Google Scholar, Scopus |
| Altmetric score | Online attention | Altmetric.com |
| Downloads | Reader interest | Publisher websites |
| Mendeley readers | Readership in Mendeley | Mendeley |
| PlumX Metrics | Comprehensive engagement | Plum Analytics |

### Increasing Research Visibility

```
Before publication:
□ Post preprint (EarthArXiv, arXiv)
□ Present at conferences (AGILE, SIGSPATIAL, FOSS4G)

At publication:
□ Share on X/Twitter with visual abstract
□ Post on LinkedIn with plain-language summary
□ Upload to ResearchGate
□ Update Google Scholar and ORCID
□ Share in relevant Slack/Discord communities

After publication:
□ Write a blog post explaining findings
□ Create a short video/talk (YouTube, Vimeo)
□ Share code + data with DOI (Zenodo)
□ Present at department seminars
□ Respond to academic community questions

Ongoing:
□ Monitor Google Scholar alerts for citations
□ Engage with citing papers
□ Update profiles regularly
□ Network at conferences
□ Collaborate internationally
```

---

## GIS Research Search Strategies

### Template Search Queries

| Research Topic | Search Terms | Best Journals |
|---------------|--------------|---------------|
| Urban heat island | "urban heat island" AND ("land surface temperature" OR LST) AND (GIS OR "remote sensing") | RSE, Sustainable Cities |
| Land use change | "land use change" AND ("change detection" OR "post-classification") AND Sentinel-2 | ISPRS, RSE, JAG |
| Spatial accessibility | "spatial accessibility" AND ("floating catchment" OR isochrone) AND health | IJGIS, Health & Place |
| Deep learning + RS | "deep learning" AND ("semantic segmentation" OR "object detection") AND "remote sensing" | TGRS, ISPRS, RSE |
| Spatial epidemiology | "spatial analysis" AND (disease OR epidemic) AND (GIS OR "point pattern") | Int J Health Geogr, Spatial Statistics |
| Urban computing | "urban computing" AND ("points of interest" OR "human mobility" OR "taxi trajectory") | CEUS, IJGIS, KDD |
| Precision agriculture | "precision agriculture" AND (UAV OR drone) AND (NDVI OR "vegetation index") | Computers and Electronics in Agriculture |
| Climate adaptation | "climate adaptation" AND (vulnerability OR resilience) AND GIS | Global Environ Change, Climatic Change |

### PRISMA Flow Diagram for Systematic Reviews

```
Identification:
  Records from databases (n = ???)
  Records from other sources (n = ???)
      ↓
  Records after duplicates removed (n = ???)
      ↓
Screening:
  Records screened (n = ???)
  Records excluded (n = ???)
      ↓
  Full-text articles assessed (n = ???)
  Full-text articles excluded with reasons (n = ???)
      ↓
Included:
  Studies included in qualitative synthesis (n = ???)
  Studies included in quantitative synthesis / meta-analysis (n = ???)
```

---

## Integrated Research Workflow (2025)

```
┌─────────────────────────────────────────────────────┐
│  1. DISCOVER                                         │
│  Semantic Scholar + Elicit + Connected Papers         │
│  → Broad search → Citation graph → AI synthesis      │
├─────────────────────────────────────────────────────┤
│  2. MANAGE                                           │
│  Zotero + Better BibTeX                              │
│  → Save papers → Organize → Auto-export .bib         │
├─────────────────────────────────────────────────────┤
│  3. READ & ANNOTATE                                  │
│  Zotero PDF reader + Zotero GPT                      │
│  → Highlight → Notes → AI summaries                  │
├─────────────────────────────────────────────────────┤
│  4. ANALYZE                                          │
│  Python/R + Git + Docker                             │
│  → Reproducible analysis → Version control            │
├─────────────────────────────────────────────────────┤
│  5. WRITE                                            │
│  Overleaf/Typst + Claude/Grammarly                   │
│  → Structure → Draft → Polish → Co-author review     │
├─────────────────────────────────────────────────────┤
│  6. SHARE                                            │
│  GitHub + Zenodo + EarthArXiv                        │
│  → Code (DOI) → Data (DOI) → Preprint                │
├─────────────────────────────────────────────────────┤
│  7. SUBMIT                                           │
│  Journal editorial system                            │
│  → Cover letter → Formatting → Peer review response  │
├─────────────────────────────────────────────────────┤
│  8. PROMOTE                                          │
│  ORCID + Google Scholar + ResearchGate + X            │
│  → Update profiles → Share → Monitor impact           │
└─────────────────────────────────────────────────────┘
```
