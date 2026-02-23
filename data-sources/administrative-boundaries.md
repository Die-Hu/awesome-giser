# Administrative Boundaries

Authoritative and community-sourced boundary datasets for countries, regions, and administrative subdivisions at all levels.

> **Quick Picks**
> - **SOTA**: [geoBoundaries](https://www.geoboundaries.org) -- the most comprehensive open-license global boundary dataset (CC BY 4.0)
> - **Free Best**: [GADM v4.1](https://gadm.org) -- deepest subdivision levels (up to level 5), widely used in research
> - **Fastest Setup**: [Natural Earth](https://naturalearthdata.com) -- instant download, multiple scales, public domain

## Global Datasets

Worldwide administrative boundary datasets covering multiple levels of subdivisions.

| Dataset | Coverage | Levels | Format | License | URL | Label |
|---------|----------|--------|--------|---------|-----|-------|
| GADM v4.1 | Global | 0-5 (country to village) | SHP, GPKG, GeoJSON, KMZ, R (sp/sf) | Non-commercial only | [gadm.org](https://gadm.org) | Free / Practical |
| Natural Earth | Global | 0-1 (country, province/state) | SHP, GeoJSON, GeoPackage | Public Domain | [naturalearthdata.com](https://naturalearthdata.com) | Free / Practical |
| geoBoundaries v6 | Global | 0-5 (varies by country) | SHP, GeoJSON | CC BY 4.0 (open) | [geoboundaries.org](https://www.geoboundaries.org) | Free / SOTA |
| Overture Maps Divisions | Global | Administrative hierarchies | GeoParquet | Various (ODbL base) | [overturemaps.org](https://overturemaps.org) | Free / SOTA |
| OpenStreetMap Boundaries | Global | Varies (admin_level 2-11) | PBF, GeoJSON | ODbL | [osm-boundaries.com](https://osm-boundaries.com) | Free |
| UN SALB (Second Admin Level Boundaries) | Global | 0-2 | SHP | Free for non-commercial | [salb.un.org](https://salb.un.org) | Free |
| LSIB (US State Dept Large Scale Intl Boundaries) | Global | 0 (country only) | SHP | Public Domain | [geonode.state.gov](https://geonode.state.gov) | Free |
| World Bank Boundaries | Global | 0-2 | SHP, GeoJSON | CC BY 4.0 | [datacatalog.worldbank.org](https://datacatalog.worldbank.org) | Free |
| OCHA Admin Boundaries (CODs) | Humanitarian focus (180+ countries) | 0-4 | SHP, GeoJSON | Various (HDX) | [data.humdata.org](https://data.humdata.org) | Free / Practical |
| FAO GAUL (via GEE) | Global | 0-2 | FeatureCollection | Open | [GEE Catalog](https://developers.google.com/earth-engine/datasets/catalog/FAO_GAUL_2015_level0) | Free |

## Country-Specific Portals

National mapping agencies and statistical offices providing authoritative boundary data.

| Country | Agency | Levels Available | Format | License | URL |
|---------|--------|-----------------|--------|---------|-----|
| USA | US Census Bureau | State, County, Tract, Block Group, Block | SHP, GeoJSON, KML | Public Domain | [census.gov/geographies](https://www.census.gov/geographies) |
| UK | ONS / Ordnance Survey | Country, Region, LA, Ward, LSOA, MSOA | GeoPackage, SHP, GeoJSON | OGL v3 | [geoportal.statistics.gov.uk](https://geoportal.statistics.gov.uk) |
| Canada | Statistics Canada | Province, Census Division, CSD, DA | SHP, GeoJSON | Open Government License | [statcan.gc.ca](https://www12.statcan.gc.ca) |
| Australia | ABS | State, SA4, SA3, SA2, SA1, LGA | SHP, GeoJSON, GeoPackage | CC BY 4.0 | [abs.gov.au](https://www.abs.gov.au) |
| China | NGCC / MNR | Province, City, County | SHP | Government License | [ngcc.cn](https://www.ngcc.cn), [webmap.cn](https://www.webmap.cn) |
| India | Survey of India | State, District, Taluk, Village | SHP | Government (restricted) | [surveyofindia.gov.in](https://surveyofindia.gov.in) |
| Japan | GSI (Geospatial Info Authority) | Prefecture, Municipality | SHP, GeoJSON, Vector Tiles | CC BY 4.0 | [gsi.go.jp](https://www.gsi.go.jp) |
| Germany | BKG (Federal Cartography Agency) | Land, Kreis, Gemeinde | SHP, GML, GeoPackage | dl-de/by-2-0 | [gdz.bkg.bund.de](https://gdz.bkg.bund.de) |
| France | IGN (Admin Express) | Region, Departement, Commune | SHP, GeoJSON, GeoPackage | Open License 2.0 | [ign.fr](https://www.ign.fr) |
| Brazil | IBGE | State, Municipality, Census Tract | SHP, GeoJSON | Free | [ibge.gov.br](https://www.ibge.gov.br) |
| South Korea | KOSTAT / NGII | Province, City/County, Eup/Myeon/Dong | SHP | Government | [kostat.go.kr](https://kostat.go.kr) |
| Mexico | INEGI | State, Municipality, Locality | SHP | Free | [inegi.org.mx](https://www.inegi.org.mx) |

## Standards & Formats

Common standards, coding schemes, and best practices for working with administrative boundary data.

- **ISO 3166**: International standard for country codes (3166-1 alpha-2/3) and subdivision codes (3166-2). Use for unambiguous identification.
- **HASC Codes**: Hierarchical Administrative Subdivision Codes for multi-level identification across countries.
- **NUTS**: Nomenclature of Territorial Units for Statistics used by EU/Eurostat (NUTS 0-3 levels). Standard for EU spatial analysis.
- **GeoJSON-T**: Temporal extensions for boundary change tracking -- useful for historical analysis.
- **Topology**: Use TopoJSON or GeoPackage with topology rules for shared borders to reduce file sizes and ensure consistency.
- **Generalization**: Choose appropriate simplification levels for your map scale. Use Mapshaper (`mapshaper.org`) for interactive simplification.
- **Coordinate Reference Systems**: Ensure boundary data aligns with your project CRS. Administrative boundaries are typically in WGS-84 (EPSG:4326).
- **GeoParquet**: For large boundary datasets, GeoParquet offers fast analytical queries. Overture Maps Divisions already uses this format.
- **STAC for Boundaries**: Some providers (e.g., Microsoft Planetary Computer) serve boundary data via STAC catalogs for programmatic discovery.
