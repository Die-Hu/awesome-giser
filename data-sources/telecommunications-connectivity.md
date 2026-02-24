# Telecommunications & Connectivity Data

Geospatial data sources for cell towers, broadband coverage, submarine cables, fiber networks, spectrum allocation, internet exchange points, satellite communications, 5G deployment, digital divide metrics, and emergency communication infrastructure.

> **Quick Picks**
> - **SOTA**: [Ookla Open Data](https://github.com/teamookla/ookla-open-data) -- quarterly global network performance tiles (fixed + mobile) at ~610m resolution, free on AWS
> - **Free Best**: [OpenCelliD](https://opencellid.org) -- world's largest open cell tower database with 2B+ measurements, CC BY-SA 4.0
> - **Fastest Setup**: [TeleGeography Submarine Cable Map](https://www.submarinecablemap.com) -- interactive map with downloadable GeoJSON/KML, no registration

## Cell Tower & Base Station Locations

Crowdsourced and regulatory databases of cell tower positions, antenna characteristics, and radio access technologies.

### OpenCelliD

The world's largest open database of cell towers, maintained by Unwired Labs with crowdsourced contributions. Contains Cell ID (CID), Location Area Code (LAC), Mobile Country Code (MCC), Mobile Network Code (MNC), radio type (GSM, UMTS, LTE, 5G NR), and estimated lat/lon coordinates.

| Dataset | Coverage | Contents | Format | Access | Update | Label |
|---------|----------|----------|--------|--------|--------|-------|
| OpenCelliD Full Database | Global (~40M+ cells) | Cell ID, LAC, MCC, MNC, radio type, lat/lon, signal range, samples count | CSV (gzip) | [Download](https://opencellid.org/downloads.php) -- API key required (free registration) | Daily | Free / CC BY-SA 4.0 |
| OpenCelliD API | Global | Single-cell lookup, geolocation by cell parameters | JSON | [API Docs](https://wiki.opencellid.org/wiki/API) -- API key required | Real-time | Free (rate-limited) |
| OpenCelliD on AWS | Global | Full database mirror | CSV | [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-ljzfb55am2rla) | Daily | Free |

> **Note**: Download data is limited to the last 18 months. API access covers the full historical database.

### Mozilla Location Service (MLS)

| Dataset | Status | Notes |
|---------|--------|-------|
| Mozilla Location Service | **Retired (June 2024)** | MLS was shut down in stages between March-June 2024. No new API keys, no data submissions, all downloads deleted. Use OpenCelliD or commercial providers (Google Geolocation API, HERE Positioning) as alternatives. |

### FCC Antenna Structure Registration (ASR)

The FCC's registry of all antenna structures in the United States that require registration under Part 17 rules. Contains tower location (lat/lon), height, structure type, marking/lighting, owner information, and FAA study numbers.

| Dataset | Coverage | Contents | Format | Access | URL |
|---------|----------|----------|--------|--------|-----|
| ASR Registration Search | United States | Tower location, height, type, marking, lighting, ownership | Web search / downloadable records | Free | [FCC ASR](https://wireless2.fcc.gov/UlsApp/AsrSearch/asrRegistrationSearch.jsp) |
| ASR Advanced Search | United States | Radius-based search, state/county filtering | Web search | Free | [Advanced Search](https://wireless2.fcc.gov/UlsApp/AsrSearch/asrAdvancedSearch.jsp) |

### National Regulators (Select)

| Country | Agency | Contents | Format | URL | Label |
|---------|--------|----------|--------|-----|-------|
| Germany | Bundesnetzagentur | Base station locations, frequencies, power levels, EMF measurements | Web map / CSV | [EMF-Datenbank](https://www.bundesnetzagentur.de/emf) | Free |
| France | ANFR | All radio transmitter sites including cellular, broadcast, Wi-Fi | Web map / CSV | [Cartoradio](https://www.cartoradio.fr) | Free |
| United Kingdom | Ofcom | Mobile base station locations by operator | CSV | [Ofcom Sitefinder](https://www.ofcom.org.uk) | Free |
| Australia | ACMA | All licensed radiocommunications sites | CSV / API | [ACMA](https://www.acma.gov.au/licences/radiocomms-licence-data) | Free |
| Canada | ISED | Antenna site and licence data | CSV | [ISED](https://www.ic.gc.ca/eic/site/smtgst.nsf/eng/h_sf01841.html) | Free |

## Broadband & Internet Coverage

Maps and datasets showing where fixed and mobile broadband services are available, at what speeds, and with what quality.

### FCC National Broadband Map

The FCC's authoritative map of broadband availability in the United States, based on the Broadband Data Collection (BDC) system. ISPs report coverage at the level of individual Broadband Serviceable Locations (BSLs) rather than census blocks.

| Dataset | Coverage | Contents | Format | Access | Update | Label |
|---------|----------|----------|--------|--------|--------|-------|
| BDC Fixed Broadband Data | United States | ISP-reported fixed broadband availability per location, technology type, max speeds | CSV | [Data Download](https://broadbandmap.fcc.gov/data-download) | Semi-annual (Jan & Jul) | Free |
| BDC Mobile Broadband Data | United States | Mobile broadband coverage polygons by provider and technology (4G/5G) | SHP, CSV | [Data Download](https://broadbandmap.fcc.gov/data-download) | Semi-annual | Free |
| FCC Broadband Map (Interactive) | United States | Location-level search with speed, provider, technology | Web map | [broadbandmap.fcc.gov](https://broadbandmap.fcc.gov/) | Semi-annual | Free |

### Ofcom Connected Nations

| Dataset | Coverage | Contents | Format | Access | URL |
|---------|----------|----------|--------|--------|-----|
| Connected Nations 2025 Data | United Kingdom | Fixed broadband by postcode/premises, mobile coverage by operator (2G/3G/4G/5G), local authority summaries | CSV, XLSX | Free | [Ofcom Data Downloads](https://www.ofcom.org.uk/phones-and-broadband/coverage-and-speeds/connected-nations-20252/data-downloads-2025) |
| Interactive Report | United Kingdom | Dashboard with geographic drill-down | Web | Free | [Interactive Report](https://www.ofcom.org.uk/phones-and-broadband/coverage-and-speeds/connected-nations-interactive-report-2025) |

### Ookla Open Data (Speedtest)

Global network performance data aggregated into map tiles from Speedtest by Ookla. Contains download/upload speeds and latency measured from real user tests on mobile and fixed networks.

| Dataset | Coverage | Resolution | Contents | Format | Access | Update | Label |
|---------|----------|------------|----------|--------|--------|--------|-------|
| Fixed Performance Tiles | Global | Zoom 16 (~610m) | Avg download/upload speed, latency, test count | SHP, Parquet | [GitHub](https://github.com/teamookla/ookla-open-data), [AWS](https://registry.opendata.aws/speedtest-global-performance/) | Quarterly (since Q1 2019) | Free / CC BY-NC-SA 4.0 |
| Mobile Performance Tiles | Global | Zoom 16 (~610m) | Avg download/upload speed, latency, test count, connection type (4G/5G) | SHP, Parquet | [GitHub](https://github.com/teamookla/ookla-open-data), [AWS](https://registry.opendata.aws/speedtest-global-performance/) | Quarterly (since Q1 2019) | Free / CC BY-NC-SA 4.0 |
| Ookla on Google Earth Engine | Global | Zoom 16 tiles | Same as above, as ImageCollection | GEE Asset | [GEE Community Catalog](https://gee-community-catalog.org/projects/speedtest/) | Quarterly | Free |

### M-Lab (Measurement Lab)

The world's largest open internet measurement platform, run by a consortium including Google, academic institutions, and civil society organizations.

| Dataset | Coverage | Contents | Format | Access | Update | Label |
|---------|----------|----------|--------|--------|--------|-------|
| M-Lab NDT | Global | Download/upload speed, latency, packet loss, client IP, geo-located | BigQuery (SQL queryable) | [measurementlab.net](https://www.measurementlab.net/data/) -- Google Cloud account required | Daily (~24h delay) | Free |
| M-Lab Raw Data | Global | Full test results from all M-Lab tools | JSON / raw archives | [Google Cloud Storage](https://www.measurementlab.net/data/) | Daily | Free |

> **Note**: M-Lab uses MaxMind GeoLite2 for IP geolocation. Accuracy may be limited for mobile, VPN, or NAT users.

### GSMA Mobile Coverage Maps

| Dataset | Coverage | Contents | Format | Access | Label |
|---------|----------|----------|--------|--------|-------|
| GSMA Coverage Maps (Interactive) | Global (196 countries) | Mobile network coverage by technology (2G/3G/4G/5G), operator-sourced | Web map | [gsma.com/coverage](https://www.gsma.com/coverage/) | Free (interactive) |
| GSMA Coverage Maps (Data) | Global (196 countries) | Raster coverage data by technology and country | Raster / proprietary | [GSMA Intelligence](https://www.gsmaintelligence.com/) | Commercial (subscription) |

## Submarine Cable Networks

Datasets mapping the undersea fiber optic cables that carry 99%+ of intercontinental internet traffic.

### TeleGeography Submarine Cable Map

The industry-standard reference for global submarine cable infrastructure. The 2025 edition depicts 597 cable systems and 1,712 landing stations.

| Dataset | Contents | Format | Access | Label |
|---------|----------|--------|--------|-------|
| Submarine Cable Map (Interactive) | Cable routes, landing points, owners, RFS date, length, cable details | Web map | [submarinecablemap.com](https://www.submarinecablemap.com/) | Free |
| Cable Route Data | Cable polyline geometries with metadata | GeoJSON | [cable-geo.json](https://www.submarinecablemap.com/api/v3/cable/cable-geo.json) | Free |
| Landing Point Data | Landing station point geometries | GeoJSON | [landing-point-geo.json](https://www.submarinecablemap.com/api/v3/landing-point/landing-point-geo.json) | Free |
| All Cables Metadata | Full cable metadata (owners, RFS, length) | JSON | [cable/all.json](https://www.submarinecablemap.com/api/v3/cable/all.json) | Free |

### Infrapedia

| Dataset | Coverage | Contents | Format | Access | Label |
|---------|----------|----------|--------|--------|-------|
| Infrastructure Atlas | Global | 4,000+ data centers, 15,000+ networks, submarine & terrestrial fiber routes | Web map (interactive) | [infrapedia.com](https://www.infrapedia.com/) | Free (basic) / Commercial (full) |

## Fiber Optic Networks

| Dataset | Coverage | Contents | Format | Access | Label |
|---------|----------|----------|--------|--------|-------|
| NTIA NBAM Platform | United States (38 states) | Multi-source broadband availability, fiber coverage, unserved/underserved areas | GIS platform (web) | [nbam.ntia.gov](https://nbam.ntia.gov/) | Free (authorized users) |
| BroadbandUSA Data | United States | Broadband investment, deployment, and availability data | Web portal / reports | [broadbandusa.ntia.gov](https://broadbandusa.ntia.gov/resources/data-and-mapping) | Free |
| USAC Connect America Fund Map | United States | Subsidized broadband deployment areas, CAF recipients | Web map | [USAC Open Data](https://data.usac.org/publicreports/caf-map/) | Free |
| EU Broadband Coverage Mapping | EU member states | Broadband technology availability by household | Web dashboard | [EC Digital Strategy](https://digital-strategy.ec.europa.eu) | Free |

## Spectrum Allocation & Licensing

### FCC Universal Licensing System (ULS)

The FCC's comprehensive database of all wireless licenses in the United States: frequency assignments, geographic service areas, licensee information, and technical parameters.

| Dataset | Contents | Format | Access | Update | Label |
|---------|----------|--------|--------|--------|-------|
| ULS License Search | All wireless licenses: call sign, licensee, frequencies, service area, technical specs | Web search | [License Search](https://wireless2.fcc.gov/UlsApp/UlsSearch/searchLicense.jsp) | Continuous | Free |
| ULS Public Access Downloads | Full database dumps: applications, licenses, frequencies, locations | Pipe-delimited flat files (ZIP) | [Database Downloads](https://www.fcc.gov/wireless/data/public-access-files-database-downloads) | Weekly full + daily incremental | Free |

### ITU Spectrum & Satellite Filings

| Dataset | Coverage | Contents | Format | Access | Label |
|---------|----------|----------|--------|--------|-------|
| BR IFIC (Space Services) | Global | Satellite network filings: orbital parameters, frequencies, service areas, coordination status | MDB database (via BRsoft) | [ITU BR IFIC](https://www.itu.int/en/ITU-R/space/Pages/brificDatabase.aspx) | Free (ITU members) / Subscription |
| ITU SpaceExplorer | Global | Interactive dashboard of satellite network filings, spectrum occupancy, orbit analysis | Web dashboard | [ITU](https://www.itu.int/en/ITU-R/space/Pages/default.aspx) | Free |
| ITU Radio Regulations | Global | International Table of Frequency Allocations (9 kHz - 275 GHz) | PDF / Database | [ITU Publications](https://www.itu.int/pub/R-REG-RR) | Commercial |

## Internet Exchange Points

Physical locations where internet networks interconnect to exchange traffic.

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| PeeringDB | Global | IXPs, data centers, networks: members, traffic, peering policy, locations, contacts | Web / JSON REST API | [peeringdb.com](https://www.peeringdb.com/) -- API key required since Jul 2025 | Free |
| Euro-IX IXPDB | Global | IXP architecture, membership, hardware, capacity, utilization, coordinates. Auto-collected from IXP management systems. | Web / API | [ixpdb.euro-ix.net](https://ixpdb.euro-ix.net/en/) | Free |
| PCH IXP Directory | Global | IXP locations, peering subnets, equipment, membership, lat/lon. Extensive historical data. | JSON, CSV | [pch.net/ixp/dir](https://www.pch.net/ixp/dir) | Free |
| Internet Exchange Map | Global | Interactive map of IXPs worldwide | Web map | [internetexchangemap.com](https://www.internetexchangemap.com/) | Free |

## Satellite Communication

Coverage maps, constellation tracking, and regulatory filings for satellite internet systems.

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| Starlink Availability Map | Global (115+ countries) | Service availability by address, residential/business/maritime/aviation | Web map | [starlink.com/map](https://www.starlink.com/map) | Free (view only) |
| OneWeb Coverage Map | Global | LEO constellation coverage footprint (~634 satellites, merged with Eutelsat 2023) | Web map | [groundcontrol.com](https://www.groundcontrol.com/knowledge/calculators-and-maps/oneweb-coverage-map/) | Free (view only) |
| UCS Satellite Database | Global | All active satellites: operator, purpose, orbit type, mass, launch date, country. Updated quarterly. | Excel / CSV | [ucsusa.org](https://www.ucsusa.org/resources/satellite-database) | Free |
| CelesTrak | Global | Satellite TLE/OMM data for 20,000+ objects, conjunction assessments, decay predictions | TLE, XML, JSON, CSV | [celestrak.org](https://celestrak.org/) | Free (daily updates) |
| SatBeams | Global | GEO satellite footprints, beam patterns, EIRP contour maps | Web map | [satbeams.com](https://www.satbeams.com/) | Free |
| N2YO Satellite Tracker | Global | Real-time tracking of 20,000+ satellites with pass predictions | Web / API | [n2yo.com](https://www.n2yo.com/) | Free |

## 5G Deployment

### Ookla 5G Map

Ookla's comprehensive tracker of 5G deployments worldwide: 233+ providers with 145,000+ deployments across 142 countries as of 2025.

| Dataset | Coverage | Contents | Format | Access | Label |
|---------|----------|----------|--------|--------|-------|
| 5G Map (Interactive) | Global (142 countries) | 5G deployment locations by operator, technology (sub-6/mmWave), commercial/limited status | Web map | [speedtest.net/ookla-5g-map](https://www.speedtest.net/ookla-5g-map) | Free |
| 5G Data on GEE | Global | 5G deployment points, operator, technology type | GEE Asset | [GEE Community Catalog](https://gee-community-catalog.org/projects/ookla_5g/) | Free |

### GSMA 5G Data

| Dataset | Coverage | Contents | Format | Access | Label |
|---------|----------|----------|--------|--------|-------|
| 5G Overview Dashboard | Global | 5G connections, adoption, coverage, spectrum assignments, network launches by country | Web dashboard | [GSMA Intelligence](https://www.gsmaintelligence.com/) | Free (summary) / Commercial (full) |
| EU 5G Observatory | EU/EEA | 5G coverage %, spectrum auction results, deployment by member state | PDF / Web | [EC Digital Strategy](https://digital-strategy.ec.europa.eu/en/policies/5g-observatory-2025) | Free |

### Carrier Coverage Maps (Select)

| Provider | Country | Technologies | URL |
|----------|---------|-------------|-----|
| T-Mobile | US | 5G Extended Range, Ultra Capacity, 4G LTE | [t-mobile.com/coverage](https://www.t-mobile.com/coverage/coverage-map) |
| AT&T | US | 5G, 5G+, LTE, FirstNet | [att.com/maps](https://www.att.com/maps/wireless-coverage.html) |
| Verizon | US | 5G Ultra Wideband, 5G Nationwide, 4G LTE | [verizon.com/coverage-map](https://www.verizon.com/coverage-map/) |
| Deutsche Telekom | Germany | 5G, LTE, 3G | [telekom.de/netzausbau](https://www.telekom.de/netzausbau) |

## Digital Divide & Connectivity Metrics

Indices, surveys, and datasets measuring who is connected, how well, and what barriers remain.

| Source | Coverage | Contents | Format | Access | Label |
|--------|----------|----------|--------|--------|-------|
| ITU ICT Development Index 2025 | Global (200+ economies) | Composite index score (access, use, skills sub-indices), country rankings. Confirms 2.2B people remain offline. | PDF / Web | [ITU IDI 2025](https://www.itu.int/itu-d/reports/statistics/idi2025/) | Free |
| ITU DataHub | Global | 200+ ICT indicators per country, time series | Online database | [datahub.itu.int](https://datahub.itu.int/) | Free (basic) / Commercial (full) |
| World Bank ICT Indicators | Global | Internet users, broadband subs, mobile subs, ICT goods trade, by country | CSV, API, Excel | [Data360](https://data360.worldbank.org/en/dataset/ITU_DH) | Free |
| GSMA SOMIC 2025 | Global (LMICs focus) | Mobile internet adoption, usage gap vs coverage gap, barriers (affordability, skills, relevance) | PDF / Web | [gsma.com/somic](https://www.gsma.com/somic/) | Free |
| GSMA Mobile Connectivity Index | Global | Composite index: infrastructure, affordability, consumer readiness, content & services | Web interactive | [mobileconnectivityindex.com](https://www.mobileconnectivityindex.com/) | Free |
| OECD Broadband Statistics | OECD countries | Fixed/mobile broadband subscriptions, speeds, prices, by country | Excel / Web | [OECD Broadband Portal](https://www.oecd.org/sti/broadband/broadband-statistics/) | Free |
| Internet Society Pulse | Global | Internet shutdowns, resilience, accessibility metrics | Web dashboard | [pulse.internetsociety.org](https://pulse.internetsociety.org/) | Free |
| Inclusive Internet Index | Global (100 countries) | Availability, affordability, relevance, readiness indices | Web / PDF | [theinclusiveinternet.eiu.com](https://theinclusiveinternet.eiu.com/) | Free |

## Emergency Communication Infrastructure

Networks and systems for public safety and first-responder communication.

### FirstNet (US)

The US nationwide public safety broadband network built with AT&T. Covers 2.99+ million square miles with 7+ million public safety connections as of 2025.

| Dataset | Contents | Format | Access | Label |
|---------|----------|--------|--------|-------|
| FirstNet Authority Network Info | Network buildout status, site deployments, coverage expansion plans | Web / reports | [firstnet.gov/network](https://www.firstnet.gov/network) | Free (reports) |
| AT&T Coverage Map (FirstNet layer) | Combined AT&T + FirstNet coverage including Band 14 | Web map | [att.com/maps](https://www.att.com/maps/wireless-coverage.html) | Free (view) |

### TETRA Networks (Europe)

TETRA (Terrestrial Trunked Radio) is the primary public safety radio standard in Europe.

| Network | Country | Operator | Status |
|---------|---------|----------|--------|
| BOS-Digitalfunk | Germany | BDBOS | Operational (world's largest TETRA network, ~1M subscribers) |
| Airwave / ESN | United Kingdom | EE (transitioning) | TETRA being replaced by ESN (4G/5G) |
| VIRVE | Finland | Erillisverkot | Operational since 2002 |
| C2000 | Netherlands | KPN | Operational |
| ASTRID | Belgium | Astrid SA | Operational |
| Reseau Radio du Futur | France | -- | Replacing TETRAPOL with 4G/5G |

### Other Emergency Communication

| Source | Coverage | Contents | URL | Label |
|--------|----------|----------|-----|-------|
| EUCCS Map | EU | Interactive map of critical communications deployments across EU | [euccs.eu/map](https://euccs.eu/map/) | Free |
| RadioReference | US / Global | Public safety radio frequencies, trunked systems, sites, talk groups | [radioreference.com](https://www.radioreference.com/) | Free (basic) / Premium |
| CISA Emergency Communications | US | Priority telecommunications, GETS, WPS, nationwide interoperability resources | [cisa.gov](https://www.cisa.gov/topics/emergency-communications) | Free |

## Processing Notes

- **Google Earth Engine**: Ookla, M-Lab, and OpenCelliD data available as GEE assets for cloud-based analysis.
- **DuckDB + GeoParquet**: Query Ookla Parquet tiles locally with SQL for fast analysis without cloud infrastructure.
- **BigQuery**: M-Lab, FCC, and other large telecom datasets queryable via Google BigQuery (free tier: 1TB/month).
- **QGIS**: Open-source desktop GIS for loading and analyzing any of the above datasets.
- **Data Currency**: Telecommunications infrastructure data changes rapidly. URLs verified as of early 2026.
