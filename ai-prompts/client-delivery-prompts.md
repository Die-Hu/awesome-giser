# Client Delivery Prompts (甲方交付)

> Expert prompts for GIS project proposals, data deliverables, map products, analysis reports, documentation, presentations, and quality acceptance.
> All prompts support Chinese+English bilingual output. Designed for urban planning, environmental monitoring, transportation, and land management scenarios.

---

## Table of Contents

1. [Project Proposals & Scoping](#1-project-proposals--scoping)
2. [Data Deliverables](#2-data-deliverables)
3. [Map & Visualization Deliverables](#3-map--visualization-deliverables)
4. [Analysis Reports](#4-analysis-reports)
5. [Project Documentation](#5-project-documentation)
6. [Presentation & Communication](#6-presentation--communication)
7. [Quality & Acceptance](#7-quality--acceptance)

---

## 1. Project Proposals & Scoping

---

### Prompt 1 -- GIS Project Technical Proposal (技术方案)

#### Scenario (实际场景)

A municipal natural resources bureau has issued an RFP for a land-use change monitoring system covering 12,000 km2. Your team must submit a technical proposal within two weeks, demonstrating domain expertise and a credible implementation plan. The client expects both Chinese and English executive summaries.

#### Roles

**Project Manager** drafts the proposal; **Technical Lead** validates feasibility; **Client (甲方)** evaluates against RFP scoring criteria.

#### Prompt Template

```text
You are a Senior GIS Project Manager with 15+ years delivering spatial
information systems for government clients in China.

Draft a GIS Project Technical Proposal (技术方案) with the following structure:

1. Project Background (项目背景)
   - Summarize the client need: {{client_need}}
   - Study area: {{study_area}} (include area in km2)
   - Policy/regulatory context: {{policy_context}}

2. Project Objectives (项目目标)
   - List 3-5 SMART objectives aligned with {{client_need}}

3. Technical Approach (技术路线)
   - Data acquisition: list sources from {{data_sources}}
   - Processing pipeline: describe ETL, QA/QC, CRS unification
   - Analysis methods: {{analysis_methods}}
   - Visualization & delivery: web maps, reports, data packages

4. Data Requirements (数据需求)
   - Enumerate datasets, format, resolution, update frequency
   - Reference open data catalogs where applicable

5. Technology Stack (技术选型)
   - GIS platform, database, web framework, cloud/on-premise
   - Justify each choice with 1-2 sentences

6. Project Timeline (项目计划)
   - Gantt-style table: phase, tasks, duration, milestones
   - Total duration: {{total_months}} months

7. Deliverables (交付成果)
   - Numbered list with format and acceptance criteria

8. Team Composition (项目团队)
   - Role, responsibility, person-months for each member

9. Risk Management (风险管理)
   - Top 5 risks with probability, impact, mitigation

Output language: {{language: Chinese | English | Bilingual}}
Format: Markdown with tables. Use professional, formal tone.
```

#### Variables to Customize

- `{{client_need}}` -- e.g., "land-use change detection 2015-2025"
- `{{study_area}}` -- e.g., "Kunming metropolitan area, 12,000 km2"
- `{{policy_context}}` -- e.g., "Third National Land Survey follow-up"
- `{{data_sources}}` -- reference [../data-sources/](../data-sources/) catalog
- `{{analysis_methods}}` -- e.g., "change detection, LULC classification"
- `{{total_months}}` -- project duration
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A 3,000-5,000 word structured proposal with Gantt table, team matrix, risk register, and numbered deliverables list -- ready for client review after minor customization.

#### Validation Checklist

- [ ] All 9 sections present with substantive content
- [ ] Timeline is realistic (no phase shorter than 2 weeks)
- [ ] Deliverables have explicit format and acceptance criteria
- [ ] Risk register includes at least 5 items with mitigation
- [ ] Language matches `{{language}}` parameter

#### Cost Optimization

Use this prompt with a mid-tier model (Claude Sonnet) for first drafts. Reserve Opus for final polish and executive summary refinement. Typical token usage: 2,500 input / 6,000 output.

#### Dark Arts Tip

Embed the client's exact RFP scoring criteria headings as your section titles. Evaluators scan for keyword matches before reading content -- aligning structure to their rubric increases scores with zero extra effort.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [GIS Tools Overview](../tools/)

#### Extensibility

Chain with **P3 (Cost Estimation)** to auto-generate the budget appendix. Feed the deliverables list into **P4 (Spatial Data Package)** for data delivery planning.

---

### Prompt 2 -- Technology Selection Report (技术选型报告)

#### Scenario (实际场景)

Your team must justify choosing PostGIS + GeoServer + MapLibre over ArcGIS Enterprise for a provincial-level flood monitoring platform. The client's IT department insists on a formal comparison report before approving the architecture. Budget sensitivity is high.

#### Roles

**Technical Architect** writes the report; **Project Manager** reviews cost alignment; **Client IT Director (甲方信息中心)** approves or rejects.

#### Prompt Template

```text
You are a GIS Solutions Architect preparing a Technology Selection
Report (技术选型报告) for a government GIS project.

Context:
- Project: {{project_name}}
- Core requirements: {{requirements_list}}
- Budget constraint: {{budget_range}}
- Client environment: {{existing_infra}}

Compare the following technology options for each layer:

| Layer              | Option A           | Option B           | Option C       |
|--------------------|--------------------|--------------------|----------------|
| Spatial Database   | {{db_options}}     |                    |                |
| Map Server         | {{server_options}} |                    |                |
| Frontend Framework | {{frontend_options}}|                   |                |
| Analysis Engine    | {{analysis_options}}|                   |                |

For EACH layer, provide:
1. Feature comparison matrix (support for required features)
2. Performance benchmarks or published references
3. Total Cost of Ownership (license, hosting, maintenance, training)
4. Community/vendor support assessment
5. Risk factors (vendor lock-in, EOL, skill availability)
6. Recommendation with justification

Final section: Integrated Architecture Recommendation
- Diagram description (components + data flow)
- Migration path from {{existing_infra}} if applicable
- Estimated implementation effort (person-days)

Output: {{language}} Markdown with comparison tables.
```

#### Variables to Customize

- `{{project_name}}` -- e.g., "Provincial Flood Monitoring Platform"
- `{{requirements_list}}` -- functional and non-functional requirements
- `{{budget_range}}` -- e.g., "500,000 - 800,000 CNY"
- `{{existing_infra}}` -- e.g., "Windows Server 2019, Oracle 12c"
- `{{db_options}}` / `{{server_options}}` / `{{frontend_options}}` / `{{analysis_options}}`
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A structured comparison report (2,000-4,000 words) with feature matrices, cost tables, and a clear final recommendation backed by quantitative and qualitative evidence.

#### Validation Checklist

- [ ] Every required layer has at least 2 options compared
- [ ] Cost figures are present (even as estimates) for each option
- [ ] Risk factors mention vendor lock-in and skill availability
- [ ] Final recommendation is explicitly stated with rationale

#### Cost Optimization

Pre-fill the feature comparison tables with known specs before sending to the LLM. This reduces output tokens by 30% and improves accuracy since the model supplements rather than invents.

#### Dark Arts Tip

Frame your preferred technology option as the "baseline" that other options are compared against. Anchoring bias means evaluators unconsciously measure alternatives by how far they deviate from the first option presented.

#### Related Pages

- [GIS Tools Overview](../tools/)
- [JavaScript Bindbox](../js-bindbox/)
- [Frontend Frameworks](../js-bindbox/framework-integration.md)

#### Extensibility

Feed the selected stack into **P1 (Technical Proposal)** Section 5, and into **P16 (Technical Documentation)** for architecture docs.

---

### Prompt 3 -- Cost & Resource Estimation

#### Scenario (实际场景)

After winning a GIS platform project for a district-level smart city initiative, you need to produce a detailed cost breakdown for contract negotiation. The client demands transparency on person-days, compute costs, and data procurement. Your finance team needs this to set milestone-based payment schedules.

#### Roles

**Project Manager** builds the estimate; **Finance Controller** validates margins; **Client Procurement (甲方采购)** negotiates against the breakdown.

#### Prompt Template

```text
You are a GIS Project Manager creating a detailed Cost & Resource
Estimation for contract negotiation.

Project scope:
- Name: {{project_name}}
- Duration: {{duration_months}} months
- Deliverables: {{deliverables_list}}
- Team size: {{team_size}} (roles: {{roles}})

Generate the following cost breakdown:

1. Human Resources (人力成本)
   | Role | Monthly Rate (CNY) | Person-Months | Subtotal |
   Assume standard rates for: PM, GIS Analyst, Developer, DBA, QA
   Apply regional adjustment: {{region}}

2. Data Procurement (数据采购)
   | Dataset | Source | License Type | Unit Cost | Quantity | Subtotal |
   Reference: {{data_sources_used}}

3. Software & Licenses (软件许可)
   | Software | License Model | Annual Cost | Duration | Subtotal |
   Include both commercial and open-source (support contracts)

4. Infrastructure (基础设施)
   | Resource | Specification | Monthly Cost | Duration | Subtotal |
   Cloud vs on-premise: {{deployment_model}}

5. Travel & Field Work (差旅及外业)
   Estimate based on {{field_trips}} trips, {{trip_duration}} days each

6. Risk Buffer (风险储备)
   Apply {{risk_percentage}}% contingency, justified by risk category

7. Summary Table
   | Category | Amount (CNY) | % of Total |
   Grand total with and without tax ({{tax_rate}}%)

8. Payment Schedule
   Milestone-based: {{milestone_count}} milestones aligned to deliverables

Output: {{language}} with all currency in CNY.
```

#### Variables to Customize

- `{{project_name}}`, `{{duration_months}}`, `{{team_size}}`, `{{roles}}`
- `{{deliverables_list}}` -- from P1 output
- `{{region}}` -- e.g., "Tier 2 city, Yunnan Province"
- `{{data_sources_used}}` -- reference [../data-sources/](../data-sources/)
- `{{deployment_model}}` -- "Alibaba Cloud" / "on-premise" / "hybrid"
- `{{field_trips}}`, `{{trip_duration}}`, `{{risk_percentage}}`, `{{tax_rate}}`
- `{{milestone_count}}`, `{{language}}`

#### Expected Output

Complete cost breakdown with 7 category tables, grand total, and a milestone payment schedule. Figures are realistic estimates based on standard Chinese GIS industry rates.

#### Validation Checklist

- [ ] All 7 cost categories populated with line items
- [ ] Person-month rates are within industry norms for {{region}}
- [ ] Risk buffer is between 8-15% of subtotal
- [ ] Payment schedule sums to grand total
- [ ] Tax calculation is correct

#### Cost Optimization

Use structured JSON output mode to generate the tables, then render to Markdown. This reduces token waste from formatting. Reuse the deliverables list from P1 output to avoid re-generating scope text.

#### Dark Arts Tip

Pad the risk buffer under "data quality remediation" -- clients never question data quality costs but will scrutinize every developer day. This gives you negotiation room without exposing core margins.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [GIS Tools Overview](../tools/)

#### Extensibility

Output feeds directly into contract appendices. Combine with **P1 (Technical Proposal)** for a complete bid package. Link milestones to **P22 (Acceptance Testing)** criteria.

---

## 2. Data Deliverables

---

### Prompt 4 -- Spatial Data Package (空间数据交付包)

#### Scenario (实际场景)

Your team has completed six months of land parcel surveying and digitization for a county-level natural resources bureau. You must package 47 spatial layers with full documentation into a deliverable that passes the client's acceptance review. The package must include both GeoPackage and GeoParquet formats, ISO 19115 metadata, and bilingual data dictionaries.

#### Roles

**GIS Data Engineer** prepares the package; **QA Lead** validates completeness; **Client Data Manager (甲方数据管理员)** receives and verifies.

#### Prompt Template

```text
You are a GIS Data Engineer preparing a formal Spatial Data Delivery
Package (空间数据交付包) for client handover.

Project context:
- Project: {{project_name}}
- Layer count: {{layer_count}} spatial layers
- Study area: {{study_area}}
- CRS: {{target_crs}} (e.g., CGCS2000 / EPSG:4490)
- Formats required: {{output_formats}}

Generate a complete delivery package specification:

1. Package Structure (目录结构)
   ```
   {{project_code}}_delivery/
   ├── data/
   │   ├── geopackage/   -- all layers in .gpkg
   │   ├── geoparquet/   -- all layers in .parquet
   │   └── tabular/      -- non-spatial reference tables (.csv)
   ├── metadata/
   │   ├── iso19115/     -- per-layer XML metadata
   │   └── lineage/      -- processing lineage documents
   ├── docs/
   │   ├── data_dictionary_cn.xlsx
   │   ├── data_dictionary_en.xlsx
   │   ├── data_dictionary.md
   │   ├── crs_documentation.md
   │   └── known_issues.md
   ├── validation/
   │   └── quality_report.pdf
   └── README.md
   ```

2. Data Dictionary Template (数据字典)
   For each layer, generate a table:
   | Field | Type | Length | Description (CN) | Description (EN) |
   | Constraints | Valid Values | Example |

3. CRS Documentation
   - Source CRS, target CRS, transformation method, accuracy

4. Validation Report Summary
   - Record count per layer
   - Geometry validity check results
   - Topology check results
   - Attribute completeness percentage
   - CRS verification

5. README Template
   - Package overview, layer inventory, usage instructions
   - Contact information, license terms

Output: {{language}} Markdown. Include file naming conventions.
```

#### Variables to Customize

- `{{project_name}}`, `{{project_code}}`, `{{layer_count}}`
- `{{study_area}}`, `{{target_crs}}`
- `{{output_formats}}` -- e.g., "GeoPackage, GeoParquet, Shapefile"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A complete delivery package specification with directory tree, data dictionary template, CRS docs, validation summary template, and README -- ready to populate with actual data.

#### Validation Checklist

- [ ] Directory structure covers data, metadata, docs, validation
- [ ] Data dictionary includes both Chinese and English descriptions
- [ ] CRS documentation specifies transformation method
- [ ] Validation report template includes geometry and attribute checks
- [ ] README contains contact info and license terms

#### Cost Optimization

Generate the data dictionary template once, then use a script to populate it from database schema. Avoid asking the LLM to describe each of 47 layers individually -- batch by category instead.

#### Dark Arts Tip

Include a "known_issues.md" file proactively. Documenting minor issues yourself (e.g., "3 parcels in Dongchuan district lack elevation values due to cloud cover") signals thoroughness and preempts the client from discovering them as defects during acceptance review.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [GIS Tools Overview](../tools/)

#### Extensibility

Feed layer list into **P5 (Metadata Generator)** and validation results into **P7 (Data Quality Certification)**. Use with **P22 (Acceptance Testing)** to verify the package meets criteria.

---

### Prompt 5 -- Metadata Generator (元数据生成)

#### Scenario (实际场景)

The provincial geographic information center requires ISO 19115-compliant metadata for every spatial dataset submitted to their SDI catalog. You have 30+ layers with no existing metadata. Manually writing XML would take days. You need to auto-generate metadata from dataset descriptions and refine.

#### Roles

**GIS Analyst** provides dataset descriptions; **Data Engineer** generates metadata; **Client SDI Administrator (甲方信息中心)** validates compliance.

#### Prompt Template

```text
You are a Spatial Data Infrastructure specialist generating ISO 19115
compliant metadata for a GIS dataset.

Dataset information:
- Title: {{dataset_title}}
- Abstract: {{abstract}}
- Purpose: {{purpose}}
- Spatial extent: {{bbox}} (West, South, East, North)
- Temporal extent: {{temporal_start}} to {{temporal_end}}
- CRS: {{crs}}
- Lineage: {{processing_steps}}
- Source data: {{source_datasets}}
- Responsible party: {{organization}}
- Contact: {{contact_email}}
- Distribution format: {{format}}
- Access constraints: {{access_constraints}}
- Use constraints: {{use_constraints}}

Generate:

1. ISO 19115 XML Metadata
   - Identification (title, abstract, purpose, keywords)
   - Spatial representation (vector/raster, geometry type)
   - Reference system (CRS with EPSG code)
   - Data quality (completeness, positional accuracy)
   - Lineage (source data, processing steps)
   - Distribution (format, online resource URL)
   - Constraints (access, use, security classification)
   - Maintenance (update frequency, next update date)

2. FGDC-compatible summary (if {{fgdc_required}} is true)

3. Human-readable metadata card (Markdown)
   - One-page summary suitable for a data catalog entry

Keywords to include: {{keywords}}
Metadata standard: ISO 19115:2014 / ISO 19139 encoding
Output language: {{language}}
```

#### Variables to Customize

- `{{dataset_title}}`, `{{abstract}}`, `{{purpose}}`
- `{{bbox}}` -- bounding box coordinates
- `{{temporal_start}}`, `{{temporal_end}}`
- `{{crs}}`, `{{processing_steps}}`, `{{source_datasets}}`
- `{{organization}}`, `{{contact_email}}`
- `{{format}}`, `{{access_constraints}}`, `{{use_constraints}}`
- `{{keywords}}`, `{{fgdc_required}}`, `{{language}}`

#### Expected Output

A valid ISO 19115/19139 XML metadata file, optional FGDC summary, and a human-readable Markdown metadata card. The XML should pass validation against the ISO 19139 schema.

#### Validation Checklist

- [ ] XML is well-formed and follows ISO 19139 encoding
- [ ] All mandatory ISO 19115 elements are present
- [ ] Bounding box coordinates are valid and consistent with CRS
- [ ] Lineage section documents all processing steps
- [ ] Keywords include both English and Chinese terms

#### Cost Optimization

Batch multiple datasets in a single prompt by passing a JSON array of dataset descriptions. The model can generate metadata for 5-8 datasets in one call, reducing per-dataset cost by 60%.

#### Dark Arts Tip

Pre-populate the "Data Quality" section with generic but technically correct statements like "Positional accuracy assessed through comparison with control points; RMSE within specification." Most SDI validators check for presence, not substance, of quality elements.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [GIS Tools Overview](../tools/)

#### Extensibility

Integrate into an automated pipeline: extract schema from GeoPackage, pass to this prompt, write XML to metadata directory. Chain with **P4 (Spatial Data Package)** for complete delivery.

---

### Prompt 6 -- Data Dictionary & Schema Documentation

#### Scenario (实际场景)

A smart city project has accumulated 85 database tables across three PostgreSQL schemas with inconsistent naming (mixed Chinese and English column names, undocumented foreign keys). The client demands a bilingual data dictionary before accepting the database handover. You have one week.

#### Roles

**Database Administrator** extracts schema info; **GIS Analyst** writes field descriptions; **Client Technical Reviewer (甲方技术评审)** checks completeness.

#### Prompt Template

```text
You are a GIS Database Specialist creating a bilingual Data Dictionary
and Schema Documentation for client delivery.

Database context:
- Database: {{db_name}} ({{db_engine}})
- Schema(s): {{schema_list}}
- Table count: {{table_count}}
- Domain: {{project_domain}}

For the following table, generate a complete data dictionary entry:

Table: {{table_name}}
Schema: {{schema_name}}
Description (CN): {{table_desc_cn}}
Description (EN): {{table_desc_en}}
Raw columns (from information_schema):
{{column_info_json}}

Generate:

1. Table Overview
   - Purpose, record count estimate, update frequency
   - Primary key, spatial column, geometry type, SRID

2. Field Dictionary (字段字典)
   | # | Field Name | Data Type | Length | Nullable |
   | Description (CN) | Description (EN) | Constraints |
   | Valid Values / Range | Default | Example |

3. Relationships
   - Foreign keys to other tables (with cardinality)
   - Spatial joins or implicit relationships

4. Indexes
   - Existing indexes (spatial, B-tree, GiST)
   - Recommended additional indexes for common queries

5. Sample Queries
   - 2-3 common query patterns with SQL examples

Output formats:
- Markdown (for documentation site)
- Excel-compatible CSV (pipe-delimited for easy import)

Language: {{language}}
```

#### Variables to Customize

- `{{db_name}}`, `{{db_engine}}` -- e.g., "smartcity_db", "PostgreSQL 15 + PostGIS 3.4"
- `{{schema_list}}`, `{{table_count}}`, `{{project_domain}}`
- `{{table_name}}`, `{{schema_name}}`, `{{table_desc_cn}}`, `{{table_desc_en}}`
- `{{column_info_json}}` -- output from `information_schema.columns` query
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A complete data dictionary entry for one table in both Markdown and CSV formats, with field descriptions in Chinese and English, relationship documentation, index recommendations, and sample queries.

#### Validation Checklist

- [ ] Every column from the schema is documented
- [ ] Chinese and English descriptions are provided for all fields
- [ ] Primary key and foreign keys are identified
- [ ] At least 2 sample queries are included
- [ ] Valid values/ranges are specified for coded fields

#### Cost Optimization

Extract `information_schema` metadata programmatically and pass as JSON. The LLM adds human-readable descriptions and documentation structure. This is 80% automation, 20% LLM enhancement.

#### Dark Arts Tip

For undocumented legacy columns, generate plausible descriptions based on the column name, data type, and sample values. Mark them with "(inferred)" so you are technically transparent, but most reviewers will not re-verify inferred descriptions if they look reasonable.

#### Related Pages

- [GIS Tools Overview](../tools/)
- [Data Sources Catalog](../data-sources/)

#### Extensibility

Loop this prompt across all tables using a script. Combine outputs into a single data dictionary document for **P4 (Spatial Data Package)**. Feed relationship info into **P16 (Technical Documentation)** database schema section.

---

### Prompt 7 -- Data Quality Certification (数据质量认证报告)

#### Scenario (实际场景)

Before final acceptance of a cadastral mapping project, the client requires a formal data quality report conforming to ISO 19157. Your team has run automated checks but needs to compile results into a structured certification document with statistical summaries and pass/fail verdicts per quality element.

#### Roles

**QA/QC Specialist** compiles the report; **Project Manager** signs off; **Client Quality Inspector (甲方质检员)** reviews against contract specifications.

#### Prompt Template

```text
You are a GIS Quality Assurance specialist producing a Data Quality
Certification Report per ISO 19157 for client acceptance.

Project: {{project_name}}
Dataset: {{dataset_name}}
Specification reference: {{quality_spec}}
Inspection method: {{inspection_method}}
Sample size: {{sample_size}} ({{sampling_method}})

Quality check results (input as JSON):
{{quality_results_json}}

Generate a formal report with:

1. Executive Summary (概述)
   - Overall quality verdict: PASS / CONDITIONAL PASS / FAIL
   - Key statistics in a summary table

2. Quality Elements (per ISO 19157)
   a. Completeness (完整性)
      - Commission errors: {{commission_rate}}%
      - Omission errors: {{omission_rate}}%
      - Threshold: {{completeness_threshold}}%
      - Verdict: PASS/FAIL

   b. Positional Accuracy (位置精度)
      - Method: {{accuracy_method}}
      - RMSE: {{rmse_value}} meters
      - CE90: {{ce90_value}} meters
      - Threshold: {{accuracy_threshold}} meters
      - Verdict: PASS/FAIL

   c. Attribute Accuracy (属性精度)
      - Error rate by field category
      - Threshold: {{attribute_threshold}}%
      - Verdict: PASS/FAIL

   d. Logical Consistency (逻辑一致性)
      - Topology errors: count and type
      - Domain violations: count
      - Format consistency: PASS/FAIL

   e. Temporal Quality (时间质量)
      - Currency assessment
      - Temporal consistency check

3. Non-Conformance Register
   | ID | Layer | Quality Element | Description | Severity | Status |

4. Recommendations
   - Items requiring remediation before re-inspection
   - Items acceptable with documented limitations

Output: {{language}} formal report format.
```

#### Variables to Customize

- `{{project_name}}`, `{{dataset_name}}`, `{{quality_spec}}`
- `{{inspection_method}}`, `{{sample_size}}`, `{{sampling_method}}`
- `{{quality_results_json}}` -- automated check results
- Threshold values for each quality element
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A formal ISO 19157-structured quality certification report with per-element verdicts, non-conformance register, and remediation recommendations. Suitable for attachment to acceptance documentation.

#### Validation Checklist

- [ ] All five ISO 19157 quality elements are assessed
- [ ] Each element has a quantitative measure and pass/fail verdict
- [ ] Non-conformance register lists all failures with severity
- [ ] Executive summary verdict is consistent with element verdicts
- [ ] Thresholds match contract specification values

#### Cost Optimization

Feed automated QC tool output (e.g., from QGIS sketcher or PostGIS validation queries) directly as JSON. The LLM structures and narrates but does not invent numbers.

#### Dark Arts Tip

Set your internal QC thresholds 5% tighter than the contract specification. When you report "PASS" against the contract threshold, the client sees comfortable margins. This also gives you a buffer if the client re-tests with slightly different methodology.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [GIS Tools Overview](../tools/)

#### Extensibility

Attach to **P4 (Spatial Data Package)** validation directory. Reference from **P22 (Acceptance Testing Plan)** as evidence. Feed non-conformances into issue tracking for remediation.

---

## 3. Map & Visualization Deliverables

---

### Prompt 8 -- Professional Map Series (专题图集)

#### Scenario (实际场景)

A provincial land department requires a printed atlas of 24 thematic maps (land use, soil type, slope, population density, etc.) for their annual report. Maps must use standardized symbology, include bilingual legends, and be print-ready at 300 DPI. Consistency across all maps is critical.

#### Roles

**Cartographer/GIS Analyst** produces maps; **Project Manager** reviews compliance; **Client Department Head (甲方处长)** approves the atlas.

#### Prompt Template

```text
You are a GIS Cartographer creating a Professional Map Series
(专题图集) specification and automation scripts.

Map series context:
- Project: {{project_name}}
- Map count: {{map_count}} thematic maps
- Study area: {{study_area}}
- Print size: {{paper_size}} (e.g., A1, A3)
- Resolution: 300 DPI minimum
- Style guide: {{style_guide_ref}}

For each map theme below, generate:
{{map_themes_list}}

1. Map Specification Sheet
   - Title (CN + EN), subtitle, map number
   - Data layers with symbology rules (color codes, line weights)
   - Classification method (natural breaks, equal interval, manual)
   - Legend layout (bilingual labels)
   - Scale bar, north arrow, graticule settings
   - Inset map requirements (location map, detail area)
   - Data source citation text
   - Map frame dimensions and margins

2. QGIS Python Script (PyQGIS)
   - Load layers from {{data_source_path}}
   - Apply symbology from QML/SLD files
   - Set print layout with all map elements
   - Export to PDF at 300 DPI
   - Batch process all {{map_count}} maps

3. Quality Checklist per Map
   - Symbology matches specification
   - All labels readable at print size
   - No overlapping labels
   - Scale bar accurate
   - CRS annotation correct
   - Data source citation present

Output: {{language}} with Python code blocks.
```

#### Variables to Customize

- `{{project_name}}`, `{{map_count}}`, `{{study_area}}`
- `{{paper_size}}`, `{{style_guide_ref}}`
- `{{map_themes_list}}` -- e.g., "land use, soil type, slope, population density"
- `{{data_source_path}}` -- path to GeoPackage or PostGIS connection
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

Map specification sheets for each theme, a batch PyQGIS script for automated production, and a per-map quality checklist. The script should produce print-ready PDFs with minimal manual adjustment.

#### Validation Checklist

- [ ] Symbology rules include specific color hex codes
- [ ] Classification method is specified for each quantitative map
- [ ] PyQGIS script handles batch export for all maps
- [ ] Bilingual legends are included in layout specification
- [ ] Output resolution is 300 DPI or higher

#### Cost Optimization

Generate the specification for one "template map" first, then ask the LLM to adapt it for each theme. This reuses 70% of the layout specification and reduces total tokens.

#### Dark Arts Tip

Create two versions of each map: one with standard government-approved colors, one with a modern aesthetic. Present the modern version first in meetings. Clients will feel progressive choosing it. If they hesitate, the standard version is your "safe" fallback that still meets requirements.

#### Related Pages

- [Desktop GIS Tools](../tools/desktop-gis.md)
- [Python GIS Libraries](../tools/python-libraries.md)

#### Extensibility

Feed map specifications into **P10 (Dashboard Design)** for web adaptation. Use the PyQGIS scripts as a starting point for **P9 (Interactive Web Map)** layer styling.

---

### Prompt 9 -- Interactive Web Map (在线地图应用)

#### Scenario (实际场景)

A city planning bureau wants a public-facing web map showing approved development projects, zoning boundaries, and public facilities. The application must work on mobile devices, load under 3 seconds, and support Chinese-language search. No proprietary map services -- open-source only.

#### Roles

**Web GIS Developer** builds the application; **UX Designer** reviews usability; **Client Planning Officer (甲方规划师)** defines content requirements.

#### Prompt Template

```text
You are a Senior Web GIS Developer creating an Interactive Web Map
application specification and starter code.

Application requirements:
- Project: {{project_name}}
- Purpose: {{app_purpose}}
- Target users: {{user_profile}}
- Data layers: {{layer_list}}
- Basemap: {{basemap_provider}} (e.g., Tianditu, OSM)
- Search: {{search_requirements}}
- Mobile support: required, responsive design

Generate:

1. Application Architecture
   - Frontend: MapLibre GL JS + {{frontend_framework}}
   - Backend: {{backend_stack}} (if needed)
   - Data serving: PMTiles / Vector Tiles / GeoJSON
   - Hosting: {{hosting_env}}

2. Layer Configuration
   For each layer in {{layer_list}}:
   - Source type (vector tiles, GeoJSON, WMS)
   - Style specification (MapLibre style JSON)
   - Popup template (HTML with field bindings)
   - Filter controls (if applicable)
   - Min/max zoom levels

3. UI Components
   - Map controls (zoom, locate, basemap switcher)
   - Layer panel with toggle switches
   - Search bar with geocoding
   - Feature info panel / popup design
   - Legend (auto-generated from style)
   - Responsive layout breakpoints

4. Performance Optimization
   - Tile caching strategy
   - Lazy loading for off-screen layers
   - Viewport-based data fetching
   - Target: first meaningful paint < 3 seconds

5. Starter Code
   - index.html with MapLibre initialization
   - map-config.js with layer definitions
   - styles.css with responsive layout

Output: {{language}} documentation + JavaScript/HTML code blocks.
```

#### Variables to Customize

- `{{project_name}}`, `{{app_purpose}}`, `{{user_profile}}`
- `{{layer_list}}` -- e.g., "development projects (polygon), zoning (polygon), facilities (point)"
- `{{basemap_provider}}` -- e.g., "Tianditu WMTS"
- `{{search_requirements}}` -- e.g., "search by project name, address, parcel ID"
- `{{frontend_framework}}` -- e.g., "Vue 3", "React", "vanilla JS"
- `{{backend_stack}}`, `{{hosting_env}}`
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

Complete application architecture document, layer configurations with MapLibre style JSON, UI component specifications, performance optimization plan, and working starter code.

#### Validation Checklist

- [ ] Architecture uses only open-source components
- [ ] All specified layers have style and popup configurations
- [ ] Responsive breakpoints cover mobile (360px+) and desktop
- [ ] Performance section addresses < 3 second load target
- [ ] Starter code is runnable without modification

#### Cost Optimization

Provide the MapLibre style spec documentation URL as context rather than expecting the model to recall syntax from training data. This improves code accuracy and reduces back-and-forth corrections.

#### Dark Arts Tip

Add a subtle loading animation with a progress bar during initial tile fetch. Users perceive applications with progress indicators as 40% faster than identical applications without them, even when actual load times are the same.

#### Related Pages

- [2D Web Mapping](../js-bindbox/2d-mapping.md)
- [Framework Integration](../js-bindbox/framework-integration.md)

#### Extensibility

Extend with **P10 (Dashboard Design)** for analytical dashboards. Feed layer styles into **P11 (3D Visualization)** for CesiumJS adaptation.

---

### Prompt 10 -- Dashboard Design Specification

#### Scenario (实际场景)

An environmental protection bureau needs a real-time monitoring dashboard showing air quality stations, water quality readings, and pollution source locations on a map, with time-series charts and summary statistics. The dashboard will be displayed on a 4K command center screen and individual officer laptops.

#### Roles

**Dashboard Designer** creates the specification; **Frontend Developer** implements; **Client Operations Director (甲方运营主管)** defines KPIs and layout preferences.

#### Prompt Template

```text
You are a GIS Dashboard Designer creating a comprehensive Dashboard
Design Specification for a spatial monitoring system.

Dashboard context:
- Project: {{project_name}}
- Data sources: {{data_sources_list}}
- Update frequency: {{update_frequency}}
- Display targets: {{display_targets}}
- Key metrics: {{kpi_list}}

Generate:

1. Layout Wireframe (ASCII)
   - Grid-based layout showing component placement
   - Responsive variants: command center (3840x2160) and laptop (1920x1080)
   - Component sizing ratios

2. Map Component
   - Center map, basemap, default zoom
   - Layer list with real-time data binding
   - Color-coded markers (thresholds: {{threshold_config}})
   - Click interaction: popup with latest readings + trend spark

3. Chart Components
   For each KPI in {{kpi_list}}:
   - Chart type (line, bar, gauge, heatmap calendar)
   - Data binding (field, aggregation, time window)
   - Threshold lines and alert coloring
   - Library: {{chart_library}} (e.g., ECharts, Observable Plot)

4. Filter & Control Panel
   - Time range selector (preset + custom)
   - Spatial filter (draw polygon, admin boundary select)
   - Category filter (station type, pollutant type)
   - Auto-refresh toggle with interval setting

5. Data Flow Architecture
   - Source -> API -> Frontend state -> Components
   - WebSocket vs polling for real-time updates
   - Caching strategy for historical data

6. Alert System
   - Threshold breach notification (visual + optional audio)
   - Alert history panel

Output: {{language}} with ASCII wireframes and code snippets.
```

#### Variables to Customize

- `{{project_name}}`, `{{data_sources_list}}`, `{{update_frequency}}`
- `{{display_targets}}` -- e.g., "4K command center, laptop, tablet"
- `{{kpi_list}}` -- e.g., "AQI, PM2.5, water pH, COD, noise dB"
- `{{threshold_config}}` -- e.g., "AQI: green<50, yellow<100, orange<150, red>=150"
- `{{chart_library}}` -- e.g., "ECharts 5"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A complete dashboard design specification with ASCII wireframes, component-level configurations, data flow architecture, and alert system design. Ready for frontend development without further design iteration.

#### Validation Checklist

- [ ] Wireframe covers both command center and laptop layouts
- [ ] Every KPI has a dedicated chart component with threshold coloring
- [ ] Map component includes real-time marker updates
- [ ] Data flow supports the specified update frequency
- [ ] Alert system covers threshold breach visualization

#### Cost Optimization

Use the LLM to generate the specification document only. Actual wireframes should be created in Figma or Excalidraw for pixel-perfect fidelity. ASCII wireframes are sufficient for developer handoff.

#### Dark Arts Tip

Default the time range to "last 24 hours" rather than "last 7 days." Shorter windows show fewer anomalies, making the system look healthier during demos. The client can always switch to longer ranges, but first impressions are formed from defaults.

#### Related Pages

- [Charting Integration](../js-bindbox/charting-integration.md)
- [2D Web Mapping](../js-bindbox/2d-mapping.md)
- [Framework Integration](../js-bindbox/framework-integration.md)

#### Extensibility

Combine with **P9 (Interactive Web Map)** for the map component. Feed KPI definitions into **P19 (Client Presentation)** for executive reporting.

---

### Prompt 11 -- 3D Visualization & Digital Twin

#### Scenario (实际场景)

A new urban district development authority wants a 3D digital twin of their 15 km2 area showing existing buildings (LOD2), planned buildings (from BIM models), terrain, underground utilities, and real-time IoT sensor overlays. The system must run in a browser using CesiumJS with 3D Tiles.

#### Roles

**3D GIS Developer** designs the system; **BIM Specialist** provides model conversion guidance; **Client Smart City Director (甲方智慧城市主管)** defines visualization priorities.

#### Prompt Template

```text
You are a 3D GIS Specialist designing a Digital Twin visualization
system using CesiumJS and 3D Tiles.

Project context:
- Project: {{project_name}}
- Area: {{area_km2}} km2
- Data sources:
  - Terrain: {{terrain_source}}
  - Buildings: {{building_source}} (LOD level: {{lod_level}})
  - BIM models: {{bim_format}} (e.g., IFC, Revit)
  - Underground: {{underground_data}}
  - IoT sensors: {{iot_source}}

Generate:

1. Data Pipeline
   - Terrain: source -> quantized mesh tiles
   - Buildings: source -> 3D Tiles (b3dm/glTF)
     Conversion tool: {{conversion_tool}} (e.g., py3dtiles, FME)
   - BIM: IFC -> glTF -> 3D Tiles with metadata preservation
   - Underground: cross-section generation approach

2. CesiumJS Application Architecture
   - Tileset loading strategy (LOD switching, screen-space error)
   - Terrain provider configuration
   - Imagery layer management
   - Entity/Primitive API usage for dynamic data
   - Camera flight presets for key viewpoints

3. Interaction Design
   - Building click: info panel with attributes + BIM link
   - Sensor overlay: real-time value display, history chart
   - Underground mode: terrain transparency, cross-section view
   - Time slider: show construction phases
   - Measurement tools: distance, area, height

4. Performance Optimization
   - Tile LOD thresholds for {{area_km2}} km2 area
   - Maximum simultaneous tile requests
   - GPU memory management for large tilesets
   - Progressive loading priority queue

5. Starter Code
   - CesiumJS viewer initialization
   - 3D Tileset loading with style-by-attribute
   - IoT sensor entity with real-time update

Output: {{language}} with JavaScript code blocks.
```

#### Variables to Customize

- `{{project_name}}`, `{{area_km2}}`
- `{{terrain_source}}` -- e.g., "ASTGTM2 30m DEM"
- `{{building_source}}`, `{{lod_level}}` -- e.g., "CityGML LOD2"
- `{{bim_format}}`, `{{underground_data}}`, `{{iot_source}}`
- `{{conversion_tool}}` -- e.g., "py3dtiles", "Cesium Ion", "FME"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

Complete 3D digital twin design document with data pipeline, CesiumJS architecture, interaction specifications, performance optimization plan, and starter code. Sufficient for a development team to begin implementation.

#### Validation Checklist

- [ ] Data pipeline covers all five source types
- [ ] BIM to 3D Tiles conversion preserves metadata
- [ ] Performance section addresses LOD and GPU memory
- [ ] Underground visualization approach is specified
- [ ] Starter code initializes viewer with at least one tileset

#### Cost Optimization

Split this into two prompts: (1) data pipeline and conversion specs, (2) frontend architecture and code. The pipeline prompt can use a cheaper model since it is mostly procedural knowledge.

#### Dark Arts Tip

Pre-render a fly-through video of the 3D scene for the first client demo. Browser-based 3D performance varies wildly across client hardware. A pre-rendered video guarantees a smooth first impression. Transition to live demo only after confirming the client's hardware can handle it.

#### Related Pages

- [3D Web Mapping](../js-bindbox/3d-mapping.md)
- [3D Visualization Tools](../tools/3d-visualization.md)

#### Extensibility

Integrate **P10 (Dashboard Design)** charts as overlay panels. Use **P9 (Interactive Web Map)** patterns for 2D fallback mode. Feed into **P16 (Technical Documentation)** for system architecture docs.

---

## 4. Analysis Reports

---

### Prompt 12 -- Site Suitability Analysis (选址分析报告)

#### Scenario (实际场景)

A logistics company needs to identify optimal locations for three new distribution centers within a metropolitan area. The analysis must consider proximity to highways, distance from residential areas, land cost, flood risk, and existing infrastructure. The client expects a ranked list of candidate sites with quantitative justification.

#### Roles

**GIS Analyst** performs the analysis; **Project Manager** frames the business context; **Client Logistics VP (甲方物流副总)** makes the final site decision.

#### Prompt Template

```text
You are a GIS Spatial Analyst producing a Site Suitability Analysis
Report (选址分析报告) using multi-criteria evaluation.

Project context:
- Objective: {{site_objective}}
- Study area: {{study_area}}
- Number of sites needed: {{site_count}}
- Candidate areas: {{candidate_areas}} (or "entire study area")

Criteria and weights (from AHP or client input):
{{criteria_table}}
| Criterion | Weight | Data Source | Favorable Condition |

Constraints (hard exclusions):
{{constraints_list}}

Generate:

1. Methodology
   - Multi-Criteria Decision Analysis (MCDA) framework
   - AHP consistency ratio validation (if weights from AHP)
   - Raster reclassification and overlay workflow
   - Tools: {{analysis_tools}}

2. Data Preparation
   - For each criterion: source, preprocessing, reclassification
   - Standardization method (min-max, z-score, value function)
   - Cell size and extent alignment

3. Suitability Model
   - Weighted overlay formula
   - Constraint masking
   - Result classification (5-class suitability scale)

4. Results
   - Suitability map description
   - Top {{site_count * 3}} candidate sites ranked by score
   | Rank | Site ID | Location | Score | Area (ha) |
   | Key advantages | Key disadvantages |

5. Sensitivity Analysis
   - Weight variation: +/-10% on top 3 criteria
   - Impact on top-ranked sites
   - Robustness assessment

6. Recommendations
   - Recommended sites with justification
   - Site visit priorities
   - Additional data needs for final decision

Output: {{language}} formal report with methodology transparency.
```

#### Variables to Customize

- `{{site_objective}}` -- e.g., "logistics distribution center location"
- `{{study_area}}`, `{{site_count}}`, `{{candidate_areas}}`
- `{{criteria_table}}` -- criteria with weights and data sources
- `{{constraints_list}}` -- e.g., "no sites in flood zone, >500m from schools"
- `{{analysis_tools}}` -- e.g., "Python (rasterio, numpy), QGIS"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A complete suitability analysis report with methodology, data preparation steps, ranked candidate sites, sensitivity analysis, and actionable recommendations. Reproducible by another analyst given the same data.

#### Validation Checklist

- [ ] AHP consistency ratio < 0.10 (if applicable)
- [ ] All criteria have explicit reclassification rules
- [ ] Constraint masking eliminates prohibited areas
- [ ] Sensitivity analysis tests at least 3 weight scenarios
- [ ] Recommendations are ranked with quantitative scores

#### Cost Optimization

Use the LLM for report structure and narrative. Perform actual spatial analysis in Python/QGIS and feed results as tables. Do not ask the LLM to compute spatial operations.

#### Dark Arts Tip

Always include the client's intuitively preferred location in your candidate list, even if it ranks lower. If your analysis excludes their preferred site, they will question the methodology. If it is included but ranked lower, they can make an informed choice -- and usually accept the data-driven recommendation.

#### Related Pages

- [Python GIS Libraries](../tools/python-libraries.md)
- [Data Sources Catalog](../data-sources/)

#### Extensibility

Feed top candidates into **P13 (Environmental Impact Assessment)** for environmental screening. Use site boundaries in **P8 (Map Series)** for presentation maps.

---

### Prompt 13 -- Environmental Impact Assessment GIS (环评GIS)

#### Scenario (实际场景)

A highway expansion project requires a GIS-based environmental impact assessment identifying sensitive receptors within buffer zones, analyzing viewshed impact from elevated sections, and modeling noise propagation. The environmental agency requires spatial evidence for each impact category before issuing permits.

#### Roles

**Environmental GIS Analyst** performs spatial analysis; **Environmental Scientist** interprets results; **Client Environmental Officer (甲方环保官员)** reviews compliance with regulations.

#### Prompt Template

```text
You are an Environmental GIS Specialist producing a GIS-based
Environmental Impact Assessment (环评GIS分析报告).

Project context:
- Project: {{project_name}} ({{project_type}})
- Project footprint: {{footprint_description}}
- Study area buffer: {{buffer_distance}} km from centerline
- Regulatory framework: {{regulations}}

Generate analysis specifications for each impact category:

1. Sensitive Receptor Analysis
   - Buffer zones: {{buffer_distances}} (e.g., 50m, 100m, 300m, 500m)
   - Receptor types: schools, hospitals, residences, protected areas
   - Data sources: {{receptor_data_sources}}
   - Output: receptor count per buffer zone table

2. Ecological Impact
   - Habitat mapping within study area
   - Protected species occurrence records
   - Ecological corridor connectivity analysis
   - Vegetation cover change (before/after)

3. Noise Modeling (if applicable)
   - Source: traffic volume projections
   - Propagation model: ISO 9613-2 parameters
   - Barrier/terrain attenuation
   - Contour map at 5 dB intervals

4. Air Quality Dispersion (if applicable)
   - Source: emission factors for {{project_type}}
   - Meteorological data: wind rose, stability class
   - Receptor grid resolution
   - Concentration contour maps

5. Visual Impact / Viewshed
   - Observer points: {{observer_locations}}
   - Viewshed from project infrastructure
   - Visual magnitude assessment
   - Photo-simulations needed: yes/no

6. Water Resources
   - Watershed delineation
   - Drainage path analysis
   - Flood risk within project area
   - Sediment and runoff modeling approach

7. Mitigation Mapping
   - Spatial mitigation measures per impact category
   - Green buffer zones, noise barriers, drainage design

Output: {{language}} with methodology specifications and Python/QGIS
workflow descriptions.
```

#### Variables to Customize

- `{{project_name}}`, `{{project_type}}` -- e.g., "G56 Highway Expansion", "linear infrastructure"
- `{{footprint_description}}`, `{{buffer_distance}}`
- `{{buffer_distances}}` -- multiple buffer rings
- `{{regulations}}` -- e.g., "EIA Law of PRC, GB 3096-2008 (noise)"
- `{{receptor_data_sources}}` -- reference [../data-sources/](../data-sources/)
- `{{observer_locations}}` -- key viewpoints for viewshed
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A comprehensive EIA-GIS analysis specification covering 7 impact categories, each with methodology, data requirements, analysis workflow, and output format. Sufficient for a GIS team to execute the analyses.

#### Validation Checklist

- [ ] All applicable impact categories are addressed
- [ ] Buffer distances match regulatory requirements
- [ ] Noise/air quality models reference appropriate standards
- [ ] Mitigation mapping is spatially explicit
- [ ] Data sources are identified for each analysis

#### Cost Optimization

Generate the specification document with the LLM. Actual spatial analysis (buffering, viewshed, noise modeling) must be done in GIS software. Use the spec as a task list for the analysis team.

#### Dark Arts Tip

Generate viewshed analysis from the worst-case scenario (maximum infrastructure height) first. If the visual impact is acceptable at maximum, you do not need to run additional scenarios. If it is not, negotiate design modifications before submitting -- never present a failing result without a mitigation proposal.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [Python GIS Libraries](../tools/python-libraries.md)

#### Extensibility

Feed results into **P8 (Map Series)** for EIA atlas production. Compile findings into **P19 (Client Presentation)** for stakeholder review. Link to **P22 (Acceptance Testing)** for regulatory submission.

---

### Prompt 14 -- Urban Planning Analysis (城市规划分析报告)

#### Scenario (实际场景)

A district government is preparing a comprehensive plan revision and needs spatial analysis of current population distribution, land use efficiency, public facility coverage, and infrastructure capacity. The analysis must support evidence-based zoning adjustments and identify underserved areas. Results will be presented to the planning committee.

#### Roles

**Urban GIS Analyst** performs spatial analysis; **Urban Planner** interprets planning implications; **Client Planning Committee (甲方规划委员会)** uses findings for decision-making.

#### Prompt Template

```text
You are an Urban Planning GIS Analyst producing a comprehensive
Urban Planning Analysis Report (城市规划分析报告).

Planning context:
- Area: {{planning_area}}
- Plan type: {{plan_type}} (comprehensive / detailed / special)
- Planning horizon: {{planning_year}}
- Current population: {{current_pop}}, projected: {{projected_pop}}
- Key concerns: {{planning_concerns}}

Generate analysis specifications and report structure:

1. Population & Demographics
   - Population density mapping (grid-based, {{grid_size}}m)
   - Age structure spatial distribution
   - Population projection by sub-district
   - Growth hotspot identification (kernel density)

2. Land Use Analysis
   - Current land use classification and area statistics
   - Land use efficiency metrics (FAR, building density)
   - Vacant/underutilized land identification
   - Land use compatibility matrix assessment
   - Comparison with previous plan targets

3. Public Facility Accessibility
   - Service area analysis for: {{facility_types}}
   - Walking/driving isochrones (5/10/15/20 min)
   - Coverage gap identification by residential block
   - Per-capita facility provision vs national standards

4. Transportation & Connectivity
   - Road network density by district
   - Public transit coverage (300m/500m walk buffer)
   - Commute pattern analysis (if OD data available)
   - Pedestrian/cycling network completeness

5. Infrastructure Capacity
   - Water supply/drainage network coverage
   - Power grid capacity vs demand mapping
   - Telecommunications coverage assessment
   - Green space per capita by neighborhood

6. Synthesis & Recommendations
   - Composite suitability map for development
   - Priority development/redevelopment areas
   - Infrastructure investment priorities
   - Zoning adjustment recommendations with spatial evidence

Data sources: reference {{data_catalog}}
Analysis tools: {{tools}}
Output: {{language}} formal report structure.
```

#### Variables to Customize

- `{{planning_area}}` -- e.g., "Guandu District, Kunming"
- `{{plan_type}}`, `{{planning_year}}`
- `{{current_pop}}`, `{{projected_pop}}`, `{{grid_size}}`
- `{{planning_concerns}}` -- e.g., "urban sprawl, aging population, flood risk"
- `{{facility_types}}` -- e.g., "schools, hospitals, parks, markets"
- `{{data_catalog}}` -- reference [../data-sources/urban-planning-smart-cities.md](../data-sources/urban-planning-smart-cities.md)
- `{{tools}}` -- e.g., "Python, QGIS, PostGIS, NetworkX"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A structured urban planning analysis report with six analytical sections, each containing methodology, data requirements, analysis workflow, and output descriptions. Includes synthesis with actionable zoning and investment recommendations.

#### Validation Checklist

- [ ] All six analysis domains are covered with methodology
- [ ] Accessibility analysis uses network-based (not Euclidean) distances
- [ ] National standards are cited for facility provision benchmarks
- [ ] Synthesis map integrates findings from multiple sections
- [ ] Recommendations are spatially explicit (mapped, not just listed)

#### Cost Optimization

Reuse standard analysis templates across projects. Only customize the context variables and area-specific data references. The methodology sections are 80% reusable between urban planning projects.

#### Dark Arts Tip

Present population projections using three scenarios (low/medium/high growth). Frame your recommended scenario as "medium" even if it is the one that best supports the client's development ambitions. Decision-makers gravitate toward the middle option and will feel they made a balanced choice.

#### Related Pages

- [Urban Planning & Smart Cities Data](../data-sources/urban-planning-smart-cities.md)
- [Python GIS Libraries](../tools/python-libraries.md)
- [Geocoding & Routing](../tools/geocoding-routing.md)

#### Extensibility

Feed accessibility gaps into **P12 (Site Suitability)** for facility location optimization. Use population grids in **P8 (Map Series)** for atlas production. Present findings with **P19 (Client Presentation)**.

---

### Prompt 15 -- Transportation Analysis (交通分析报告)

#### Scenario (实际场景)

A municipal transportation bureau needs a GIS-based traffic analysis to evaluate the impact of a proposed BRT corridor. The analysis must include current network performance, service area changes, origin-destination patterns, and accessibility improvements for underserved communities. Results feed into the transportation master plan revision.

#### Roles

**Transportation GIS Analyst** performs network analysis; **Traffic Engineer** validates modeling assumptions; **Client Transportation Director (甲方交通局长)** uses findings for corridor selection.

#### Prompt Template

```text
You are a Transportation GIS Analyst producing a Transportation
Analysis Report (交通分析报告) with network modeling.

Project context:
- Study area: {{study_area}}
- Analysis purpose: {{analysis_purpose}}
- Transport mode(s): {{transport_modes}}
- Proposed infrastructure: {{proposed_infra}}
- Travel demand data: {{demand_data_source}}

Generate:

1. Network Data Preparation
   - Road network source: {{network_source}} (e.g., OSM, NavInfo)
   - Network topology construction and validation
   - Speed/impedance assignment by road class
   - Turn restrictions and one-way streets

2. Current Network Analysis
   - Service area analysis from {{origin_points}}
   - Drive-time isochrones: {{time_thresholds}} minutes
   - Network connectivity assessment
   - Bottleneck identification (betweenness centrality)

3. Origin-Destination Analysis
   - OD matrix construction from {{demand_data_source}}
   - Desire line visualization
   - Flow aggregation by corridor
   - Mode split estimation (if multi-modal)

4. Proposed Infrastructure Impact
   - Before/after service area comparison
   - Accessibility change maps (time savings per zone)
   - Affected population by improvement category
   - Cost-effectiveness: population served per km of new route

5. Equity Analysis
   - Low-income area accessibility improvement
   - Elderly/disabled population service coverage
   - Comparison with equity targets

6. Routing Optimization (if applicable)
   - Optimal route alignment using least-cost path
   - Stop/station location optimization
   - Catchment area maximization

Tools: {{analysis_tools}}
Output: {{language}} with methodology and result table templates.
```

#### Variables to Customize

- `{{study_area}}`, `{{analysis_purpose}}`
- `{{transport_modes}}` -- e.g., "BRT, bus, private vehicle, cycling"
- `{{proposed_infra}}` -- e.g., "12 km BRT corridor with 18 stations"
- `{{demand_data_source}}` -- e.g., "mobile phone signaling OD data"
- `{{network_source}}` -- e.g., "OpenStreetMap + field survey"
- `{{origin_points}}` -- e.g., "proposed BRT stations"
- `{{time_thresholds}}` -- e.g., "5, 10, 15, 20, 30"
- `{{analysis_tools}}` -- e.g., "Python (NetworkX, OSMnx), QGIS Network Analysis"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A transportation analysis report covering network preparation, current conditions, OD patterns, before/after impact assessment, equity analysis, and optimization recommendations. Includes table templates for quantitative results.

#### Validation Checklist

- [ ] Network topology is validated (connected components check)
- [ ] Isochrones use network distance, not Euclidean buffers
- [ ] Before/after comparison uses identical network parameters except the proposed change
- [ ] Equity analysis includes at least one disadvantaged population group
- [ ] Cost-effectiveness metric is computed (population per km or similar)

#### Cost Optimization

Use OSMnx to extract and prepare the road network programmatically. Pass summary statistics (not raw geometry) to the LLM for report narrative generation.

#### Dark Arts Tip

When presenting before/after isochrone maps, use the same color ramp but ensure the "after" map has visibly more green (accessible) area. Choose isochrone thresholds that maximize visual difference rather than evenly spaced intervals. A 12-minute isochrone that captures a major residential area is more impactful than a 15-minute one that adds mostly farmland.

#### Related Pages

- [Geocoding & Routing Tools](../tools/geocoding-routing.md)
- [Data Sources Catalog](../data-sources/)

#### Extensibility

Feed accessibility results into **P14 (Urban Planning Analysis)** transportation section. Use OD visualizations in **P10 (Dashboard Design)** for real-time monitoring. Present with **P19 (Client Presentation)**.

---

## 5. Project Documentation

---

### Prompt 16 -- Technical Documentation (技术文档)

#### Scenario (实际场景)

Your team is handing over a city-level spatial data infrastructure (SDI) platform to the client's IT department. They need comprehensive technical documentation covering system architecture, API specifications, database design, deployment procedures, and maintenance guides. The documentation must be bilingual because the client plans to engage an international consultant for future enhancements.

#### Roles

**Technical Lead** writes architecture and API docs; **DBA** writes database docs; **Client IT Operations (甲方运维团队)** uses the documentation for ongoing maintenance.

#### Prompt Template

```text
You are a GIS Technical Writer producing comprehensive Technical
Documentation (技术文档) for system handover.

System context:
- System name: {{system_name}}
- Architecture: {{architecture_type}} (e.g., microservices, monolith)
- Tech stack: {{tech_stack}}
- Database: {{database_info}}
- Deployment: {{deployment_env}}
- User count: {{user_count}} (concurrent: {{concurrent_users}})

Generate documentation structure and content for:

1. System Architecture (系统架构)
   - Architecture diagram description (component + flow)
   - Component inventory: name, purpose, technology, port
   - Inter-component communication (REST, gRPC, message queue)
   - External integrations and dependencies

2. API Documentation (接口文档)
   - For each API endpoint: method, path, parameters, response
   - Authentication mechanism
   - Rate limiting and pagination
   - Error code reference table
   - Example requests (curl) and responses (JSON)

3. Database Design (数据库设计)
   - ER diagram description
   - Table inventory with row count estimates
   - Spatial index strategy
   - Partitioning scheme (if applicable)
   - Backup and recovery procedures

4. Deployment Guide (部署指南)
   - Environment requirements (OS, runtime, dependencies)
   - Step-by-step deployment procedure
   - Configuration file reference (with environment variables)
   - Health check endpoints
   - SSL/TLS certificate setup

5. Operations & Maintenance (运维手册)
   - Monitoring: metrics to track, alerting thresholds
   - Log management: locations, rotation, analysis
   - Common issues and troubleshooting flowchart
   - Scaling procedures (horizontal/vertical)
   - Security patch update process

6. Version History & Changelog
   - Version numbering scheme
   - Current version features
   - Known issues and workarounds

Output: {{language}} Markdown documentation. Use code blocks for
configurations and API examples.
```

#### Variables to Customize

- `{{system_name}}`, `{{architecture_type}}`, `{{tech_stack}}`
- `{{database_info}}` -- e.g., "PostgreSQL 15 + PostGIS 3.4, 500GB"
- `{{deployment_env}}` -- e.g., "Docker Compose on Alibaba Cloud ECS"
- `{{user_count}}`, `{{concurrent_users}}`
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A structured technical documentation package covering all six sections, with placeholder diagrams described in text, API examples in code blocks, and step-by-step procedures. Bilingual section headers and key terms.

#### Validation Checklist

- [ ] All six documentation sections have substantive content
- [ ] API documentation includes request/response examples
- [ ] Deployment guide is step-by-step reproducible
- [ ] Troubleshooting section covers at least 5 common issues
- [ ] Configuration file reference lists all environment variables

#### Cost Optimization

Extract API specs from OpenAPI/Swagger files and database schema from SQL dumps. Feed these as structured input. The LLM adds narrative, troubleshooting, and operations guidance around the raw specs.

#### Dark Arts Tip

Include a "Quick Start" section at the very beginning with a 5-step deployment. Technical reviewers often only verify the quick start works. If it does, they assume the rest of the documentation is equally reliable. Invest disproportionate effort in making those 5 steps bulletproof.

#### Related Pages

- [GIS Tools Overview](../tools/)
- [JavaScript Bindbox](../js-bindbox/)

#### Extensibility

Feed API documentation into **P9 (Interactive Web Map)** and **P10 (Dashboard)** for frontend integration. Use database design in **P6 (Data Dictionary)**. Reference from **P17 (User Manual)** for technical appendix.

---

### Prompt 17 -- User Manual (用户手册)

#### Scenario (实际场景)

The city planning bureau has 45 staff members who will use the new GIS platform daily. Most are planners, not IT professionals. The user manual must explain every feature in plain language, include screenshot placeholders with captions, and provide a troubleshooting FAQ. The client has explicitly requested Chinese as the primary language with English subtitles for technical terms.

#### Roles

**Technical Writer** creates the manual; **UX Designer** reviews for clarity; **Client End Users (甲方业务人员)** are the target audience.

#### Prompt Template

```text
You are a Technical Writer creating a User Manual (用户手册) for a
GIS application, targeting non-technical government staff.

Application context:
- Application name: {{app_name}}
- Primary functions: {{function_list}}
- User roles: {{user_roles}}
- Platform: {{platform}} (web / desktop / mobile)
- Language: Chinese primary, English technical terms in parentheses

Generate:

1. Getting Started (快速入门)
   - System requirements and browser/device support
   - Login and first-time setup
   - Interface overview with labeled screenshot placeholders
     [Screenshot: 主界面总览 / Main Interface Overview]
   - Navigation quick reference card

2. Feature Guides (功能指南)
   For each function in {{function_list}}:
   - Purpose: one-sentence description
   - Step-by-step instructions (numbered, max 8 steps)
   - Screenshot placeholder at key steps
     [Screenshot: {{function_name}} 操作步骤{{step_number}}]
   - Expected result description
   - Common mistakes and how to avoid them

3. Data Operations (数据操作)
   - Searching and filtering spatial data
   - Viewing feature attributes
   - Exporting data (supported formats)
   - Printing maps

4. Account & Settings (账户与设置)
   - Password change
   - Personal preferences (language, default map view)
   - Notification settings

5. FAQ & Troubleshooting (常见问题)
   - Top 10 user questions with answers
   - "Map not loading" troubleshooting tree
   - "Data export failed" resolution steps
   - Who to contact for different issue types

6. Glossary (术语表)
   | Chinese Term | English Term | Definition |

Output: Chinese Markdown with English in parentheses for technical
terms. Include [Screenshot: description] placeholders.
```

#### Variables to Customize

- `{{app_name}}`, `{{function_list}}`
- `{{user_roles}}` -- e.g., "planner (规划师), reviewer (审批员), admin (管理员)"
- `{{platform}}` -- web, desktop, or mobile

#### Expected Output

A complete user manual in Chinese with English technical terms, numbered step-by-step instructions, screenshot placeholders, FAQ section, and bilingual glossary. Written at a level appropriate for non-technical government staff.

#### Validation Checklist

- [ ] Every feature has step-by-step instructions with 8 or fewer steps
- [ ] Screenshot placeholders are included at key interaction points
- [ ] FAQ covers at least 10 common questions
- [ ] Glossary includes all technical terms used in the manual
- [ ] Language is non-technical with Chinese as primary

#### Cost Optimization

Generate the manual structure and one feature guide as a template. Have a junior writer fill in the remaining feature guides following the established pattern, using the LLM only for the FAQ and troubleshooting sections.

#### Dark Arts Tip

Put the FAQ section before the detailed feature guides in the table of contents. Most users skip to FAQ anyway. If they find their answer there, they perceive the manual as "easy to use." If the FAQ is buried at the end, users who cannot find their answer quickly will call the helpdesk and complain the manual is useless.

#### Related Pages

- [GIS Tools Overview](../tools/)

#### Extensibility

Pair with **P18 (Training Materials)** for instructor-led sessions. Reference **P16 (Technical Documentation)** for admin-level appendix. Link FAQ items to helpdesk ticket templates.

---

### Prompt 18 -- Training Materials (培训材料)

#### Scenario (实际场景)

As part of the project handover, you must conduct a three-day training program for 30 client staff members. The audience ranges from GIS beginners to intermediate users. You need slide deck outlines, hands-on exercise worksheets, and assessment quizzes. Training must cover the delivered GIS platform and basic spatial analysis concepts.

#### Roles

**Training Lead** designs the curriculum; **GIS Analyst** develops exercises; **Client HR/Training Coordinator (甲方培训协调员)** schedules and provides venue.

#### Prompt Template

```text
You are a GIS Training Specialist designing a training program for
client staff handover.

Training context:
- Program: {{training_title}}
- Duration: {{duration_days}} days ({{hours_per_day}} hours/day)
- Audience: {{audience_profile}}
- Skill levels: {{skill_levels}}
- Platform/tools covered: {{tools_covered}}
- Training environment: {{training_env}}

Generate:

1. Training Schedule (培训日程)
   | Day | Time | Topic | Type (lecture/hands-on/Q&A) | Duration |
   Balance: 40% lecture, 50% hands-on, 10% Q&A

2. Slide Deck Outlines (课件大纲)
   For each lecture topic:
   - Learning objectives (2-3 per topic)
   - Slide-by-slide outline (title + bullet points, 15-25 slides)
   - Demonstration script (what to show, in what order)
   - Transition to hands-on exercise

3. Hands-on Exercises (实操练习)
   For each exercise:
   - Exercise title and objective
   - Prerequisites (data loaded, software open)
   - Step-by-step instructions (numbered, max 15 steps)
   - Expected result (with description of correct output)
   - Challenge extension for advanced participants
   - Troubleshooting hints for common errors

4. Assessment (考核)
   - Day-end quiz: 10 multiple-choice questions per day
   - Final practical assessment: complete a mini-project
   - Grading rubric

5. Reference Materials
   - Quick reference cards (laminated handout format)
   - Keyboard shortcut cheat sheet
   - Recommended further learning resources

Output: {{language}} with slide outlines and exercise worksheets
as separate sections.
```

#### Variables to Customize

- `{{training_title}}` -- e.g., "GIS Platform Operations Training"
- `{{duration_days}}`, `{{hours_per_day}}`
- `{{audience_profile}}` -- e.g., "city planning bureau staff, age 25-55"
- `{{skill_levels}}` -- e.g., "beginner (60%), intermediate (30%), advanced (10%)"
- `{{tools_covered}}` -- e.g., "Web GIS platform, QGIS basic operations"
- `{{training_env}}` -- e.g., "computer lab with pre-installed software"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A complete training program package with daily schedule, slide deck outlines for each topic, hands-on exercise worksheets, assessment quizzes, and reference handouts.

#### Validation Checklist

- [ ] Schedule balances lecture and hands-on (approximately 40/50/10)
- [ ] Every lecture topic has learning objectives
- [ ] Exercises have step-by-step instructions with expected results
- [ ] Assessment includes both knowledge (quiz) and skills (practical)
- [ ] Difficulty progression accommodates mixed skill levels

#### Cost Optimization

Generate the training outline and one complete exercise with the LLM. Create remaining exercises by adapting the template manually. Use the LLM for quiz question generation (high quality-to-cost ratio for MCQs).

#### Dark Arts Tip

Start Day 1 with the most visually impressive feature of the platform (e.g., 3D flythrough, animated time series). First impressions set the tone for the entire training. If participants are impressed in the first 30 minutes, they remain engaged for three days. If you start with login procedures and settings, you lose them by lunch.

#### Related Pages

- [GIS Tools Overview](../tools/)
- [Data Sources Catalog](../data-sources/)

#### Extensibility

Reference **P17 (User Manual)** as the post-training reference document. Use exercise data from **P4 (Spatial Data Package)** for realistic training datasets. Record exercises as video tutorials for ongoing onboarding.

---

## 6. Presentation & Communication

---

### Prompt 19 -- Client Presentation (汇报PPT大纲)

#### Scenario (实际场景)

You have completed a six-month land use change monitoring project and must present findings to the client's leadership team (bureau director, division heads, and invited experts). The presentation must distill complex GIS analysis into clear, decision-relevant insights. You have 30 minutes plus 15 minutes Q&A. The audience includes both technical and non-technical members.

#### Roles

**Project Manager** presents; **GIS Analyst** prepares maps and data; **Client Bureau Director (甲方局长)** and **Expert Panel (评审专家)** evaluate.

#### Prompt Template

```text
You are a GIS Project Manager preparing a client presentation
(汇报PPT) for a project milestone review.

Presentation context:
- Project: {{project_name}}
- Milestone: {{milestone}} (kickoff / mid-term / final)
- Duration: {{duration_minutes}} minutes + {{qa_minutes}} Q&A
- Audience: {{audience_profile}}
- Key results: {{key_findings}}

Generate a slide deck outline:

1. Title Slide
   - Project name (CN + EN), date, presenting organization

2. Agenda (1 slide)
   - 5-6 bullet points mapping to presentation sections

3. Project Overview (2-3 slides)
   - Background and objectives (simplified from proposal)
   - Study area map
   - Timeline with current milestone highlighted

4. Methodology Summary (2-3 slides)
   - Simplified workflow diagram (input -> process -> output)
   - Data sources used (logos/icons, not text lists)
   - Key technical innovations (1 per slide, with visual)
   NOTE: Avoid jargon. Use analogies for technical concepts.

5. Key Findings (4-6 slides)
   For each finding in {{key_findings}}:
   - One slide per finding
   - Headline: insight statement (not description)
   - Visual: map / chart / comparison (described)
   - Supporting data point (one number, prominent)
   - Implication for client decision-making

6. Deliverables Status (1-2 slides)
   - Deliverables table: item, status, acceptance criteria
   - Demo or screenshot of key deliverables

7. Issues & Risks (1 slide)
   - Resolved issues (brief, shows competence)
   - Open risks with mitigation plan

8. Recommendations & Next Steps (1-2 slides)
   - Actionable recommendations (max 5)
   - Proposed next steps with timeline
   - Resources needed from client

9. Q&A Slide
   - Contact information
   - "Thank you" in appropriate style

Total: {{slide_count}} slides for {{duration_minutes}} minutes
Rule of thumb: 2 minutes per slide.

Output: {{language}} slide outline with speaker notes.
```

#### Variables to Customize

- `{{project_name}}`, `{{milestone}}`
- `{{duration_minutes}}`, `{{qa_minutes}}`
- `{{audience_profile}}` -- e.g., "bureau director (non-technical), division heads (semi-technical), 3 invited GIS experts"
- `{{key_findings}}` -- 3-5 major results
- `{{slide_count}}` -- calculated from duration
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A complete slide deck outline with slide-by-slide content, visual descriptions, speaker notes, and timing guidance. Designed for mixed technical/non-technical audience.

#### Validation Checklist

- [ ] Slide count is appropriate for presentation duration (2 min/slide)
- [ ] Key findings use insight headlines, not descriptive titles
- [ ] Methodology is simplified for non-technical audience
- [ ] Recommendations are actionable and numbered
- [ ] Speaker notes are included for each slide

#### Cost Optimization

Use the LLM to generate the outline and speaker notes. Create actual slides in PowerPoint/Keynote. Do not ask the LLM to generate slide visuals -- describe them textually for a designer.

#### Dark Arts Tip

Place your single most impressive finding on the slide immediately after the methodology section. Decision-makers pay most attention in the first third of a presentation. By the time you reach "Issues & Risks," they have already formed a positive impression that colors how they interpret problems.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [GIS Tools Overview](../tools/)

#### Extensibility

Pull findings from **P12-P15 (Analysis Reports)**. Use maps from **P8 (Map Series)**. Link deliverables to **P22 (Acceptance Testing)** criteria.

---

### Prompt 20 -- Meeting Minutes (会议纪要)

#### Scenario (实际场景)

You just completed a two-hour project coordination meeting with the client, covering data delivery issues, a change request for additional analysis layers, and a timeline adjustment. Three departments were present. You need to distribute formal meeting minutes within 24 hours, in Chinese, with clear action items and deadlines.

#### Roles

**Project Coordinator** drafts minutes; **Project Manager** reviews; **All Participants** receive and confirm within 3 business days.

#### Prompt Template

```text
You are a GIS Project Coordinator drafting formal Meeting Minutes
(会议纪要) for a client project meeting.

Meeting information:
- Project: {{project_name}}
- Meeting type: {{meeting_type}} (kickoff / progress / technical / review)
- Date: {{meeting_date}}
- Location: {{location}}
- Duration: {{duration}}
- Attendees:
  {{attendee_list}}
  (Name | Organization | Role | Present/Absent)

Meeting notes (raw):
{{raw_notes}}

Generate formal meeting minutes:

1. Header (抬头)
   - Meeting title, project name, date, location, duration
   - Attendee list (present and absent, with affiliations)
   - Minutes recorder

2. Agenda Items Discussed (议程)
   For each topic:
   - Topic title and presenter
   - Key points discussed (3-5 bullets)
   - Decisions made (numbered, explicit)
   - Dissenting opinions (if any, attributed)

3. Action Items (待办事项)
   | # | Action | Responsible | Deadline | Priority |
   Number sequentially across all topics.
   Use SMART criteria for each action.

4. Technical Issues Log (技术问题)
   | # | Issue | Status | Owner | Target Date |
   Open / In Progress / Resolved

5. Change Requests (变更申请)
   | # | Description | Impact | Approval Status |
   If approved, note impact on timeline and budget.

6. Next Meeting
   - Proposed date, time, location
   - Preliminary agenda items

7. Approval
   - Signature lines for key participants
   - Confirmation deadline: {{confirmation_deadline}}

Output: {{language}} formal document format.
Tone: objective, factual, no editorializing.
```

#### Variables to Customize

- `{{project_name}}`, `{{meeting_type}}`, `{{meeting_date}}`
- `{{location}}`, `{{duration}}`
- `{{attendee_list}}` -- structured list
- `{{raw_notes}}` -- unstructured meeting notes
- `{{confirmation_deadline}}` -- e.g., "3 business days"
- `{{language}}` -- typically Chinese for government clients

#### Expected Output

Formal meeting minutes with structured decisions, numbered action items with owners and deadlines, technical issue log, and change request documentation. Ready for distribution and confirmation.

#### Validation Checklist

- [ ] Every action item has a named owner and deadline
- [ ] Decisions are explicitly stated (not implied)
- [ ] Change requests include impact assessment
- [ ] Next meeting date and preliminary agenda are set
- [ ] Attendee list matches actual participants

#### Cost Optimization

Record meetings (with consent) and use speech-to-text for raw notes. Pass the transcript to the LLM for structuring. This eliminates manual note-taking errors and costs less than having a dedicated minute-taker.

#### Dark Arts Tip

Distribute draft minutes within 4 hours of the meeting, not 24. The faster you send them, the more likely participants will accept your framing of ambiguous discussions. After 24 hours, people reconstruct their own version of events. First documented version becomes the official record.

#### Related Pages

- [GIS Tools Overview](../tools/)

#### Extensibility

Action items feed into project management tools (Jira, Feishu). Technical issues feed into **P22 (Acceptance Testing)**. Change requests trigger updates to **P1 (Technical Proposal)** and **P3 (Cost Estimation)**.

---

### Prompt 21 -- Email Templates

#### Scenario (实际场景)

Throughout a 12-month GIS project, you need to send dozens of formal emails to the client covering project kickoff, data requests, progress updates, issue escalation, deliverable handoff, and project closure. Each email must strike the right tone for a Chinese government client -- respectful, concise, and action-oriented.

#### Roles

**Project Manager** sends all formal correspondence; **Client Project Liaison (甲方项目对接人)** receives and acts on emails.

#### Prompt Template

```text
You are a GIS Project Manager generating professional email templates
for client communication throughout the project lifecycle.

Project context:
- Project: {{project_name}}
- Client organization: {{client_org}}
- Client contact: {{client_name}}, {{client_title}}
- Your organization: {{your_org}}
- Your name/title: {{your_name}}, {{your_title}}

Generate email templates for each scenario below.
Each template must include: subject line, salutation, body, sign-off.

1. Project Kickoff Notification (项目启动通知)
   - Confirm project start date
   - Attach kickoff meeting agenda
   - List initial data/access requests
   - Propose first meeting date

2. Data Request (数据需求函)
   - Specify datasets needed: {{data_request_list}}
   - Format and CRS requirements
   - Deadline for data provision
   - Contact for technical coordination

3. Progress Update (项目进展报告)
   - Period: {{reporting_period}}
   - Completed tasks, in-progress items
   - Upcoming milestones
   - Issues requiring client attention (if any)

4. Issue Escalation (问题上报)
   - Issue description: {{issue_description}}
   - Impact on timeline/quality
   - Proposed resolution options (A/B)
   - Requested client decision by {{decision_deadline}}

5. Deliverable Handoff (成果提交)
   - Deliverable: {{deliverable_name}}
   - Summary of contents
   - Acceptance review request
   - Review period: {{review_period}} business days

6. Project Closure (项目结项通知)
   - Confirm all deliverables accepted
   - Summary of project achievements
   - Warranty/support period details
   - Thank you and future cooperation

Output: {{language}} (Chinese for government clients).
Tone: formal, respectful, concise. Use 您 (formal "you").
```

#### Variables to Customize

- `{{project_name}}`, `{{client_org}}`, `{{client_name}}`, `{{client_title}}`
- `{{your_org}}`, `{{your_name}}`, `{{your_title}}`
- `{{data_request_list}}`, `{{reporting_period}}`
- `{{issue_description}}`, `{{decision_deadline}}`
- `{{deliverable_name}}`, `{{review_period}}`
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

Six professional email templates covering the full project lifecycle, each with subject line, formal salutation, structured body, and appropriate sign-off. Chinese versions use proper government correspondence conventions.

#### Validation Checklist

- [ ] All six lifecycle stages have complete email templates
- [ ] Subject lines are specific and include project name
- [ ] Issue escalation includes options and decision deadline
- [ ] Tone is appropriately formal for government correspondence
- [ ] Chinese versions use 您 and proper salutation conventions

#### Cost Optimization

Generate all six templates in a single prompt call. Store as reusable templates in your project management system. Customize per-project with simple find-and-replace on variables.

#### Dark Arts Tip

In the progress update email, lead with completed items (even minor ones) before mentioning issues. Recipients read the first two lines and form an impression. "Completed 3 milestones on schedule. One issue requires your input below." reads very differently from "We have an issue" followed by a list of completed work.

#### Related Pages

- [GIS Tools Overview](../tools/)

#### Extensibility

Attach deliverable emails to **P4 (Spatial Data Package)** handover. Link issue escalation to **P20 (Meeting Minutes)** action items. Use closure email with **P23 (Project Closure Report)**.

---

## 7. Quality & Acceptance

---

### Prompt 22 -- Acceptance Testing Plan (验收测试方案)

#### Scenario (实际场景)

Your GIS platform project is entering the acceptance phase. The client has assembled a review committee of five experts who will spend two days testing the system. You must prepare a structured acceptance testing plan with specific test cases, pass/fail criteria, and scoring methodology. The contract specifies that 85% overall pass rate is required for acceptance.

#### Roles

**QA Manager** designs the test plan; **Project Manager** coordinates the acceptance event; **Client Review Committee (甲方验收委员会)** executes tests and scores.

#### Prompt Template

```text
You are a GIS QA Manager designing an Acceptance Testing Plan
(验收测试方案) for project final delivery.

Project context:
- Project: {{project_name}}
- Deliverables under test:
  {{deliverables_list}}
- Contract quality requirements: {{quality_requirements}}
- Pass threshold: {{pass_threshold}}% overall score
- Testing duration: {{test_duration}} days
- Review committee: {{committee_size}} members

Generate:

1. Test Plan Overview (测试概述)
   - Scope: what is and is not being tested
   - Testing methodology: demonstration + hands-on + inspection
   - Scoring system: {{scoring_method}} (e.g., pass/fail, 1-5 scale)
   - Acceptance criteria: overall and per-category thresholds

2. Test Categories & Cases
   a. Data Quality (数据质量) -- weight: {{data_weight}}%
      | TC# | Test Case | Method | Expected Result | Pass Criteria |
      Cover: completeness, accuracy, consistency, format compliance

   b. Map Products (地图成果) -- weight: {{map_weight}}%
      | TC# | Test Case | Method | Expected Result | Pass Criteria |
      Cover: symbology, labeling, scale accuracy, print quality

   c. Application Functionality (应用功能) -- weight: {{app_weight}}%
      | TC# | Test Case | Method | Expected Result | Pass Criteria |
      Cover: core features, search, export, user management

   d. Performance (系统性能) -- weight: {{perf_weight}}%
      | TC# | Test Case | Method | Expected Result | Pass Criteria |
      Cover: page load time, concurrent users, data query speed

   e. Documentation (文档资料) -- weight: {{doc_weight}}%
      | TC# | Test Case | Method | Expected Result | Pass Criteria |
      Cover: completeness, accuracy, bilingual quality

3. Testing Schedule (测试日程)
   | Day | Time | Category | Test Cases | Facilitator |

4. Scoring Sheet Template
   | TC# | Category | Max Score | Actual Score | Remarks |
   Summary row with weighted total

5. Issue Classification
   - Critical: blocks acceptance (must fix before re-test)
   - Major: requires remediation within {{remediation_days}} days
   - Minor: documented, fix in maintenance phase

6. Re-test Procedure
   - Trigger: overall score < {{pass_threshold}}%
   - Re-test scope: failed categories only
   - Timeline: within {{retest_days}} days

Output: {{language}} formal test plan document.
```

#### Variables to Customize

- `{{project_name}}`, `{{deliverables_list}}`
- `{{quality_requirements}}` -- from contract
- `{{pass_threshold}}` -- e.g., 85
- `{{test_duration}}`, `{{committee_size}}`
- `{{scoring_method}}` -- pass/fail or numeric scale
- Weight percentages for each category (must sum to 100%)
- `{{remediation_days}}`, `{{retest_days}}`
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A complete acceptance testing plan with categorized test cases, pass/fail criteria, testing schedule, scoring sheet template, and issue classification system. Ready for the review committee to execute.

#### Validation Checklist

- [ ] Category weights sum to 100%
- [ ] Every test case has an explicit pass/fail criterion
- [ ] Testing schedule fits within the allocated duration
- [ ] Scoring sheet template is ready to print and use
- [ ] Re-test procedure has clear trigger and timeline

#### Cost Optimization

Generate the test plan structure with the LLM. Populate specific test cases from your project's requirements traceability matrix. The LLM excels at structuring and standardizing, not at inventing project-specific test cases.

#### Dark Arts Tip

Offer to "help the committee prepare" by providing pre-filled scoring sheets with test case descriptions. Committees that use your scoring sheets follow your test sequence and focus on your chosen criteria. If they design their own, they test unpredictably and may focus on areas you are less confident about.

#### Related Pages

- [GIS Tools Overview](../tools/)
- [Data Sources Catalog](../data-sources/)

#### Extensibility

Reference **P7 (Data Quality Certification)** as evidence for data quality test cases. Link **P16 (Technical Documentation)** completeness to documentation test cases. Feed results into **P23 (Project Closure Report)**.

---

### Prompt 23 -- Project Closure Report (项目结项报告)

#### Scenario (实际场景)

After 18 months, your comprehensive GIS platform project for the municipal natural resources bureau has passed acceptance testing. You must prepare a formal project closure report documenting achievements, deliverables, financial summary, lessons learned, and recommendations for future phases. This report becomes the official project record and is filed with both organizations.

#### Roles

**Project Manager** compiles the report; **Team Leads** contribute section inputs; **Client Project Office (甲方项目办)** co-signs and archives.

#### Prompt Template

```text
You are a GIS Project Manager preparing a Project Closure Report
(项目结项报告) for formal project completion.

Project summary:
- Project: {{project_name}}
- Contract number: {{contract_number}}
- Duration: {{start_date}} to {{end_date}} ({{actual_months}} months)
- Original budget: {{original_budget}} CNY
- Final cost: {{final_cost}} CNY
- Client: {{client_org}}
- Contractor: {{contractor_org}}

Generate:

1. Executive Summary (项目概述)
   - Project background (2-3 sentences)
   - Objectives achieved (bullet list with status)
   - Overall assessment: {{overall_rating}}

2. Objectives & Achievement (目标达成)
   | # | Objective | Target | Actual | Status | Evidence |
   For each objective from the original proposal

3. Deliverables Register (交付清单)
   | # | Deliverable | Format | Acceptance Date | Status |
   | Storage Location | Responsible Person |
   All deliverables with acceptance confirmation

4. Financial Summary (经费决算)
   | Category | Budgeted | Actual | Variance | Explanation |
   By major cost category from P3
   Overall variance analysis

5. Timeline Review (进度回顾)
   - Planned vs actual timeline (milestone comparison)
   - Delays and causes
   - Recovery actions taken

6. Team Performance (团队绩效)
   - Team members and contributions
   - Person-months planned vs actual
   - Subcontractor performance (if applicable)

7. Lessons Learned (经验总结)
   | # | Category | Lesson | Recommendation |
   Categories: technical, management, client relations, data, tools
   Minimum 8 lessons

8. Risk Register Closeout (风险关闭)
   | # | Risk | Status | Outcome |
   All risks from original register with final status

9. Recommendations (建议)
   - System maintenance recommendations
   - Data update schedule
   - Potential Phase 2 / follow-on projects
   - Technology upgrade roadmap

10. Appendices
    - Acceptance test report reference
    - Key correspondence log
    - Change order summary
    - IP and license transfer records

Output: {{language}} formal report. Professional, objective tone.
```

#### Variables to Customize

- `{{project_name}}`, `{{contract_number}}`
- `{{start_date}}`, `{{end_date}}`, `{{actual_months}}`
- `{{original_budget}}`, `{{final_cost}}`
- `{{client_org}}`, `{{contractor_org}}`
- `{{overall_rating}}` -- e.g., "Successful with minor deviations"
- `{{language}}` -- Chinese, English, or Bilingual

#### Expected Output

A comprehensive project closure report with 10 sections covering all aspects of project completion. Formal enough for official filing, detailed enough to serve as institutional knowledge for future projects.

#### Validation Checklist

- [ ] All deliverables are listed with acceptance dates
- [ ] Financial variance is explained for each category
- [ ] Lessons learned cover at least 5 categories with 8+ items
- [ ] Recommendations include specific next steps, not vague suggestions
- [ ] Appendix references are complete

#### Cost Optimization

Compile inputs from existing project artifacts (acceptance report, financial ledger, team timesheets) and pass as structured data. The LLM synthesizes and narrates. This avoids generating fictional project data.

#### Dark Arts Tip

In the lessons learned section, frame challenges as "opportunities identified for process improvement" rather than failures. Write them in third person ("the project team found that...") rather than first person. This depersonalizes issues and positions your team as reflective professionals rather than individuals who made mistakes.

#### Related Pages

- [Data Sources Catalog](../data-sources/)
- [GIS Tools Overview](../tools/)

#### Extensibility

The recommendations section seeds the proposal for Phase 2 -- feed directly into **P1 (Technical Proposal)** for follow-on work. Lessons learned contribute to organizational knowledge base. Financial summary informs **P3 (Cost Estimation)** for future projects.

---

## Quick Reference: Prompt Index

| # | Prompt Name | Section | Primary Output |
|---|-------------|---------|----------------|
| P1 | Technical Proposal (技术方案) | Proposals | Project proposal document |
| P2 | Technology Selection (技术选型) | Proposals | Comparison report |
| P3 | Cost Estimation | Proposals | Budget breakdown |
| P4 | Spatial Data Package (数据交付包) | Data | Delivery specification |
| P5 | Metadata Generator (元数据) | Data | ISO 19115 XML |
| P6 | Data Dictionary (数据字典) | Data | Schema documentation |
| P7 | Quality Certification (质量认证) | Data | ISO 19157 report |
| P8 | Map Series (专题图集) | Maps | Print-ready maps |
| P9 | Web Map (在线地图) | Maps | Web application |
| P10 | Dashboard Design | Maps | Dashboard specification |
| P11 | 3D / Digital Twin | Maps | CesiumJS application |
| P12 | Site Suitability (选址分析) | Analysis | Ranked site report |
| P13 | Environmental Impact (环评) | Analysis | EIA-GIS report |
| P14 | Urban Planning (城市规划) | Analysis | Planning analysis |
| P15 | Transportation (交通分析) | Analysis | Network analysis |
| P16 | Technical Docs (技术文档) | Docs | System documentation |
| P17 | User Manual (用户手册) | Docs | End-user guide |
| P18 | Training Materials (培训材料) | Docs | Training package |
| P19 | Presentation (汇报PPT) | Communication | Slide deck outline |
| P20 | Meeting Minutes (会议纪要) | Communication | Formal minutes |
| P21 | Email Templates | Communication | Lifecycle emails |
| P22 | Acceptance Testing (验收测试) | Quality | Test plan |
| P23 | Closure Report (结项报告) | Quality | Project closure |

---

[Back to AI Prompts](README.md) | [Back to Main README](../README.md)
