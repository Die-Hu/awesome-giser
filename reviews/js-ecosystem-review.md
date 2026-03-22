# JS 生态评审报告

> 审阅日期：2026-03-21
> 审阅人：JS 生态专家 (js-ecosystem-expert)

## 审阅范围

1. `js-bindbox/2d-mapping.md`
2. `js-bindbox/3d-mapping.md`
3. `js-bindbox/tile-servers.md`
4. `js-bindbox/spatial-analysis.md`
5. `js-bindbox/data-formats-loading.md`
6. `js-bindbox/charting-integration.md`
7. `js-bindbox/framework-integration.md`
8. `js-bindbox/performance-optimization.md`
9. `js-bindbox/realtime-offline-advanced.md`
10. `js-bindbox/README.md`
11. `web-dev/frontend-integration.md`

---

## 工具评估清单

### 2d-mapping.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | TS 支持 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|---------|------|
| Leaflet | ~44.5K (文档写~42K) | ~2.8M | 活跃 | @types/leaflet（有缺口） | **保留** -- 数据需更新 | 行业标杆，stars/downloads 均上涨，文档中 42K stars 偏低应更正为 ~44K |
| MapTiler SDK | N/A | ~93K | 活跃（v3.11） | 内置 | **保留** | 作为 MapLibre 封装有其价值，定位准确 |
| MapLibre GL JS 4.x | ~9.9K (文档写~10.2K) | ~2.1M | 非常活跃（v5.21.0） | 内置 | **保留** -- 版本号需更新 | 文档标注 4.x，实际已到 v5.x，需更正版本号 |
| OpenLayers | ~12.3K (文档写~12K) | ~650K (ol 包) | 活跃 | 内置 | **保留** | 数据基本准确，OGC 领域无可替代 |

**数据准确性问题：**
- Leaflet stars 从 42K 涨至 ~44.5K，需更新
- MapLibre GL JS 文档标注为 "4.x"，但实际最新版为 **v5.21.0**，这是一个重大版本号错误
- MapLibre stars 文档写 10.2K，实际 ~9.9K，轻微偏差

---

### 3d-mapping.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | TS 支持 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|---------|------|
| MapLibre 3D (2.5D) | 同 MapLibre | 同上 | 同上 | 同上 | **保留** | 不是独立库，是 MapLibre 功能子集，定位正确 |
| Globe.gl | ~2.6K | ~43K | 活跃（v2.44） | 有限 | **保留** | 文档定位准确（原型/演示用途），社区确实较小 |
| deck.gl 9.x | ~13.9K | ~172K | 活跃（v9.2.9） | 内置 | **保留** | 生态健康，Uber/vis.gl 维护，数据可视化标准 |
| CesiumJS | ~14.9K (文档写~13K) | ~50K(估) | 活跃（v1.139.1） | 内置 | **保留** -- 数据需更新 | stars 从 13K 涨至 ~14.9K，需更正 |
| Three.js + geo | ~110.7K | ~5.5M | 非常活跃（v0.183） | 内置 | **保留** | 作为通用 3D 引擎地位无可撼动，文档中"geo 用途是 niche"的定位正确 |

**数据准确性问题：**
- CesiumJS stars 文档写 ~13K，实际已达 ~14.9K
- deck.gl 文档未给出具体 stars 数（写"Part of vis.gl ecosystem"），实际 ~13.9K

---

### tile-servers.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|------|
| PMTiles (Protomaps) | ~2.6K | N/A (JS客户端) | 活跃（v4.4.0） | **保留** | 革命性工具，serverless 瓦片方案，生态增长中 |
| MapTiler / Stadia / Mapbox | 商业服务 | N/A | 持续运营 | **保留** | 云瓦片服务对比表有实用价值 |
| Martin | ~2.5K(估) | N/A (Rust) | 非常活跃 | **保留** | MapLibre 官方瓦片服务器，Rust 性能优异 |
| pg_tileserv | ~500(估) | N/A (Go) | 维护中 | **保留，标注降级** | 文档已正确定位为原型工具，Martin 更优 |
| TileServer GL | ~2.2K(估) | N/A (Node) | 维护中 | **保留** | WMTS 栅格输出仍有 niche 价值 |
| t-rex | ~600(估) | N/A (Rust) | **已废弃** | **标记过时** | 项目已标注"不再维护"，推荐迁移至 bbox-tile-server，文档需更新 |
| GeoServer 2.25 | ~4K | N/A (Java) | 活跃 | **保留** | OGC 合规无替代品，文档定位准确 |

**数据准确性问题：**
- t-rex 文档写"Largely superseded by Martin"，实际情况更严重：**项目已正式废弃**，不再维护，推荐 bbox-tile-server 替代。文档应更新措辞为"已废弃"而非"被超越"

---

### spatial-analysis.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | TS 支持 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|---------|------|
| Turf.js (@turf/*) | ~10.2K | ~934K (@turf/turf) | 活跃（v7.3.4） | 内置 | **保留** | 客户端空间分析标准，文档中"millions/week"表述需具体化 |
| Proj4js (proj4) | ~2.2K | ~654K | 稳定 | @types/proj4 | **保留** | CRS 转换无替代品，生态健康 |
| Flatbush | ~1.2K | ~500K(估) | 稳定 | 内置 | **保留** | mourner 作品，极致性能，行业标准 |
| rbush | ~4K | ~1.5M(估) | 稳定 | @types/rbush | **保留** | 动态空间索引标准选择 |
| kdbush | ~700(估) | ~300K(估) | 稳定 | 内置 | **保留** | 点数据专用索引，文档定位正确 |
| H3-js | ~5.3K (uber/h3-js) | ~100K(估) | 活跃 | 内置 | **保留** | Uber 生态，六边形索引标准 |
| JSTS | ~2K(估) | ~100K(估) | 维护中 | 有限 | **保留** | JTS 移植版，拓扑正确性无替代 |
| geotiff.js | ~1K | ~223K | 活跃（v3.0.5） | 内置 | **保留** | COG 浏览器读取唯一选择 |
| loam | ~200(估) | 极低 | **已废弃（2023年停更）** | 无 | **标记过时** | 文档已标注"effectively abandoned"，评估一致 |
| gdal3.js | ~415 | ~226 | 维护中 | 有限 | **保留，降级标注** | 极 niche，WASM 方案仍有特定价值 |

**数据准确性问题：**
- Turf.js 文档写 "npm downloads: Millions/week combined"，实际 @turf/turf 单包约 934K/周。"Millions" 如果指所有 @turf/* 子包合计可能成立，但措辞应更精确
- loam 废弃状态文档已正确标注

---

### data-formats-loading.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | TS 支持 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|---------|------|
| GeoJSON (原生) | N/A | N/A | N/A | N/A | **保留** | 通用格式，文档分析全面 |
| TopoJSON (topojson-client) | ~1.5K(估) | ~719K | 稳定 | @types | **保留** | 共享边界压缩仍有价值 |
| CSV + Papa Parse | ~12K | ~5.5M | 稳定（v5.5.3） | @types | **保留** | CSV 解析事实标准 |
| @tmcw/togeojson | ~400(估) | ~15K(估) | 维护中 | 内置 | **保留** | GPX/KML 转换唯一可靠方案 |
| FlatGeobuf | ~785 | ~7.8K | 活跃（v4.3.1） | 内置 | **保留** | 空间过滤 + 静态托管，独特价值 |
| shapefile (mbostock) | ~400(估) | ~144K | **8年未更新** | @types | **保留，标注稳定/遗留** | v0.6.6 已 8 年未发布新版，但作为 Shapefile 读取仍可用。文档定位为"Legacy Import"准确 |
| GeoParquet + DuckDB-WASM | ~4K (duckdb-wasm) | N/A | 非常活跃（v1.33） | 内置 | **保留** | 前沿技术，分析场景革命性工具 |
| Apache Arrow / GeoArrow | N/A | N/A | 活跃 | N/A | **保留，降级标注** | 生态不成熟的标注准确 |

**数据准确性问题：**
- shapefile.js 最后发版确实约 8 年前，文档应明确标注这一事实（目前文档中仅暗示为 legacy）
- DuckDB-WASM 文档提到 "npm supply chain risk -- September 2025 compromise" -- 这条安全事件信息有价值，应保留

---

### charting-integration.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | TS 支持 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|---------|------|
| Observable Plot | ~4.8K (文档写~5.2K) | ~50K(估) | 活跃 | 内置 | **保留** -- 数据需更新 | stars 数偏差，文档写 5.2K，实际 ~4.8K |
| Apache ECharts 5.x | ~65.4K (文档写~66K) | ~1.1M | 非常活跃（**v6.0.0**） | 内置 | **保留** -- 版本号需更新 | 文档标注 5.x，实际已升级到 **v6.0.0** |
| Plotly.js | ~17.7K (文档写~18K) | ~235K (文档写~3M) | 活跃（v3.3.1） | @types | **保留** -- 下载量数据偏差大 | 文档写 npm ~3M/week，实际 plotly.js 本体 ~235K。3M 可能包含 plotly.js-dist 等衍生包，需澄清 |
| Vega-Lite | ~5.1K (文档写~5.2K) | ~285K | 活跃（v6.4.2） | 内置 | **保留** | 数据基本准确 |
| D3-geo | ~1K (d3-geo单包) | ~500K(估) | **稳定（~2年未更新）** | @types | **保留** | 文档标注"not actively developed since June 2024"基本准确（最新版 v3.1.1 发布约 2 年前） |
| AntV L7 | ~4K | ~35K/周(估) | 活跃 | 内置 | **保留** | 中国市场定位准确，国际价值有限 |
| Highcharts Maps | 商业 | N/A | 活跃 | 内置 | **保留** | 商业方案，文档定位准确 |

**数据准确性问题：**
- ECharts 文档标注 "5.x"，实际已发布 **v6.0.0**，这是一个需要更正的版本号错误
- Plotly.js npm 下载量文档写 ~3M/week，与实际 ~235K 偏差巨大。可能因为文档统计了 plotly.js-dist（主要使用的分发包），需要澄清
- Observable Plot stars 文档写 5.2K，实际 ~4.8K，轻微偏高
- ECharts 文档提到 "High open issue count (1,758)"，这类数据变化快，应注明为"截至某日期"或改为"高 issue 数"

---

### framework-integration.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | TS 支持 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|---------|------|
| svelte-maplibre | ~486 (文档写~502) | ~713 | 活跃 | 内置 | **保留** | 小生态但 DX 优秀，文档评估准确 |
| react-map-gl v7 | ~8.4K | ~1.13M | 活跃（**v8.1.0**） | 内置 | **保留** -- 版本号需更新 | 文档标注 v7，实际已到 **v8.1.0** |
| vue-maplibre-gl | ~121 | 低（数据未取到） | 维护中 | 内置 | **保留** | 文档中"生态碎片化"评估准确 |
| Resium | ~600(估) | ~29K | 维护中 | 内置 | **保留** | 单维护者风险标注准确 |
| Zustand | ~52K(估) | ~8M(估) | 非常活跃 | 内置 | **保留** | React 状态管理标准选择 |
| Pinia | ~14K(估) | ~3M(估) | 非常活跃 | 内置 | **保留** | Vue 官方状态管理 |

**数据准确性问题：**
- react-map-gl 文档标注 "v7"，实际最新版为 **v8.1.0**。这是重大版本号错误
- svelte-maplibre stars 文档写 502，实际 ~486，轻微偏差

---

### performance-optimization.md

该文件主要是技术方案指南，不包含独立工具推荐。涉及的库（PMTiles、MapLibre、deck.gl、Flatbush、DuckDB-WASM、PostGIS）在其他文件中已评估。

| 涉及工具 | 评估结论 | 理由 |
|---------|---------|------|
| tippecanoe | **保留** | 瓦片生成标准工具，无替代品 |
| Workbox (SW缓存) | **保留** | PWA 缓存标准方案 |
| PostGIS 调优指南 | **保留** | 实用性高，数据库优化建议准确 |

**技术准确性：** 渲染性能基准表（Leaflet/MapLibre/deck.gl/CesiumJS 对比）数据合理，符合实际经验。

---

### realtime-offline-advanced.md

| 工具名 | GitHub Stars | npm 周下载 | 最近更新 | TS 支持 | 评估结论 | 理由 |
|--------|-------------|-----------|---------|---------|---------|------|
| SSE (原生) | N/A | N/A | 浏览器内置 | N/A | **保留** | 最简实时方案，文档推荐合理 |
| Supabase Realtime | ~80K(supabase总) | ~大量 | 非常活跃 | 内置 | **保留** | 管理型实时方案，文档评估准确 |
| Socket.io | ~63K | ~5.7M-9.7M | 活跃（v4.8.1） | 内置 | **保留** | 实时通信事实标准 |
| MQTT (mqtt.js) | ~9K | ~1.4M | 活跃 | 内置 | **保留** | IoT 标准，文档定位准确 |
| Dexie.js | ~14.1K | ~1.1M | 活跃（v4.3.0） | 内置 | **保留** | IndexedDB 封装标准选择 |
| Yjs | ~20.9K (文档写~21.5K) | ~900K | 活跃 | 内置 | **保留** | CRDT 协作编辑标准，文档评估全面 |
| Debezium | ~11K(估) | N/A (Java) | 活跃 | N/A | **保留** | 企业 CDC，文档定位准确 |
| Vitest | ~14K(估) | ~10M(估) | 非常活跃 | 内置 | **保留** | 测试框架标准选择 |
| Playwright | ~72K(估) | ~4M(估) | 非常活跃 | 内置 | **保留** | E2E 测试标准 |

**数据准确性问题：**
- Yjs stars 文档写 21.5K，实际 ~20.9K，轻微偏差

---

### README.md

README 文件是导航索引，不包含独立工具推荐。结构清晰，小项目/大项目双路径设计有效。

**问题：**
- 引用的版本号问题会传递至 README（如 "MapLibre GL JS 4.x" 应为 5.x，"ECharts 5.x" 应为 6.x）
- "deck.gl 9.x" 当前版本 9.2.9，仍在 9.x 范围内，准确

---

### web-dev/frontend-integration.md

该文件与 `framework-integration.md` 有大量内容重叠，但以企业视角组织。

| 工具名 | 评估结论 | 理由 |
|---------|---------|------|
| react-map-gl v7 + MapLibre GL JS 4.x | **保留** -- 版本号需更新 | 同上：react-map-gl 已 v8，MapLibre 已 v5 |
| deck.gl 9.x | **保留** | 数据准确 |
| vue-maplibre-gl | **保留** | 同上评估 |
| svelte-maplibre | **保留** | 同上评估 |
| Resium | **保留** | 同上评估 |

**内容重复问题：** 此文件与 `js-bindbox/framework-integration.md` 约 60-70% 内容重复。建议在重组时将其中一个改为引用另一个，避免维护两份相同内容。

---

## 标记为过时/低价值的项目

### 需标记为"已废弃"

| 工具 | 文件位置 | 当前状态 | 建议 |
|------|---------|---------|------|
| **loam** | spatial-analysis.md | 2023年11月停更 | 文档已标注，评估一致。建议更强调废弃状态 |
| **t-rex** | tile-servers.md | 项目已正式标注"不再维护" | 文档仅写"Largely superseded by Martin"，**需更新为"已废弃，推荐 bbox-tile-server"** |

### 需更正版本号的项目（非过时，但文档数据滞后）

| 工具 | 文档标注 | 实际版本 | 影响 |
|------|---------|---------|------|
| **MapLibre GL JS** | 4.x | **5.21.0** | 高 -- 主版本号错误，影响读者判断 |
| **react-map-gl** | v7 | **v8.1.0** | 高 -- 主版本号错误 |
| **Apache ECharts** | 5.x | **6.0.0** | 高 -- 主版本号错误 |

### 需更正统计数据的项目

| 工具 | 文档数据 | 实际数据 | 偏差程度 |
|------|---------|---------|---------|
| Leaflet stars | ~42K | ~44.5K | 低（6%偏差） |
| CesiumJS stars | ~13K | ~14.9K | 中（15%偏差） |
| Plotly.js npm/week | ~3M | ~235K (plotly.js 本体) | **极高** -- 可能统计口径不同 |
| Observable Plot stars | ~5.2K | ~4.8K | 低（8%偏差） |

### 低价值但应保留（niche 用途）的项目

| 工具 | 理由 |
|------|------|
| gdal3.js | 极 niche（226 周下载），但浏览器端 GDAL 无其他选择 |
| GeoArrow | 生态不成熟，但作为前沿方向有参考价值 |
| Highcharts Maps | 商业且功能有限，但企业客户可能已有许可 |

---

## 自我反思

### 本报告的不足与盲点

1. **npm 下载量并非万能指标。** 许多 GIS 专业工具（Martin、tippecanoe、GeoServer）不通过 npm 分发，或以 CLI/服务器形式部署。用 npm 周下载量评估这些工具会严重低估其实际采用度。本报告在这些工具上主要依赖 GitHub stars 和社区活跃度作为替代指标。

2. **中国生态可能被低估。** AntV L7、ECharts 的中国市场采用度远超 GitHub stars 和 npm 数据所能反映的。百度地图 API、高德地图 JS API 等中国特有工具未被本文档覆盖，但对面向中国市场的 GISer 可能至关重要。

3. **专业领域工具不应仅以"热度"衡量。** JSTS（拓扑正确性）、proj4（CRS 转换）、geotiff.js（COG 读取）等工具的周下载量可能只有几万到几十万，但在各自领域是**不可替代的**。以 npm 热度来建议"降级"这些工具是错误的。

4. **版本号时效性问题。** 本报告指出了多个版本号错误（MapLibre 4.x->5.x, react-map-gl v7->v8, ECharts 5.x->6.x），但 GIS 库版本迭代快，任何静态文档都会面临同样的问题。建议在文档中使用 "MapLibre GL JS" 而非 "MapLibre GL JS 4.x" 的写法，除非版本号对内容有实质影响。

5. **Tree-shaking 和包体积评估不够深入。** 报告未对每个库进行实际的 bundle 分析（如 bundlephobia 数据）。文档中的 bundle size 数据大多是近似值，实际项目中的 tree-shaking 效果取决于导入方式。

6. **TypeScript 支持质量评估粗糙。** 仅标注"内置"或"@types"不足以反映实际 DX。一些"内置"类型的库（如早期 Leaflet 插件）类型定义质量很差，而一些 @types 包（如 @types/d3-geo）质量很高。

7. **安全审计缺失。** 除了提到 DuckDB-WASM 的 npm 供应链事件外，未对其他库进行安全漏洞审查。Leaflet 和 OpenLayers 等老牌项目可能有已知 CVE。

---

## 对重组的生态建议

### 1. 版本号策略

**建议：** 在文件标题和正文中移除具体主版本号（如 "MapLibre GL JS 4.x"），改为 "MapLibre GL JS"。仅在代码示例的 `import` 语句或安装命令中标注具体版本。理由：主版本号变化快（MapLibre 一年内从 4 升到 5），静态文档无法跟上。

### 2. 精简内容重复

- `web-dev/frontend-integration.md` 与 `js-bindbox/framework-integration.md` 内容高度重叠。建议 `frontend-integration.md` 聚焦企业架构层面（SSR 策略、状态管理、性能优化），将框架 wrapper 的 API 细节和代码示例集中在 `framework-integration.md`，`frontend-integration.md` 通过链接引用。

### 3. 快速检索优化

建议在 `README.md` 增加一个**统一工具索引表**，包含所有工具的一句话定位和当前状态标签（活跃/稳定/废弃），方便开发者快速扫描。格式示例：

```
| 工具 | 一句话定位 | 状态 | 详见 |
|------|-----------|------|------|
| MapLibre GL JS | 生产级 WebGL 地图引擎 | 活跃 | 2d-mapping.md |
| Leaflet | 最快上手的轻量地图库 | 活跃 | 2d-mapping.md |
| t-rex | Rust 瓦片服务器 | 已废弃 | tile-servers.md |
```

### 4. "最后验证日期"机制

建议在每个文件顶部添加 `> 数据验证日期：YYYY-MM-DD`，提醒读者数据的时效性。npm 下载量和 GitHub stars 变化快，明确标注验证日期比追求"最新数据"更诚实。

### 5. 被遗漏的重要工具

以下工具在当前文档中未被覆盖，但对 GISer 有潜在价值：

| 工具 | 用途 | 是否建议添加 |
|------|------|------------|
| **Mapbox GL JS v3** (商业版) | 商业地图渲染 | 可在 2d-mapping 中简要提及（作为 MapLibre 的商业分支） |
| **Protobuf / FlatBuffers** | 高效二进制序列化 | 可在 data-formats 中简要提及 |
| **TiTiler** | 动态 COG 瓦片服务（Python） | 可在 tile-servers 中提及作为 COG 服务端方案 |
| **bbox-tile-server** | t-rex 的继任者 | 应在 tile-servers.md 中替换 t-rex |

### 6. 总体评价

文档整体质量**很高**。工具选择全面、排序合理（小项目优先）、caveats 部分极为实用。主要问题集中在：
- 三个主版本号错误（MapLibre、react-map-gl、ECharts）
- 一个废弃状态未更新（t-rex）
- 少量 GitHub stars/npm 数据滞后（正常现象）

建议的改进方向是**减少硬编码数据、增加时效标注、消除文件间重复**，而非增删工具。当前的工具覆盖度和深度对 GISer 来说已经足够。

---

## Phase 2 互审补充

> 基于阅读架构评审专家 (arch-reviewer) 和生产就绪专家 (production-reviewer) 报告后的补充意见。

### 一、对自身报告不足的反思

阅读另外两位同事的报告后，我意识到自己 Phase 1 报告有以下盲点：

**1. 过度聚焦 npm 数据，忽视非 JS 工具的生态评估。**
架构专家对 FastAPI + GeoAlchemy2、TiTiler、OSRM、Valhalla、pygeoapi 等后端工具有深入评估，这些工具虽然不在 JS 生态内，但与 JS 前端库有密切的上下游关系。我的报告完全忽略了这一层。例如：Martin 的生态健康度不能只看 GitHub stars，还要看它作为 PostGIS -> MapLibre 管道中间件的不可替代性。

**2. 未评估工具间的版本兼容性风险。**
生产专家提到"工具组合的生产可靠性"未被评估。从 JS 生态角度，这一点非常重要：
- react-map-gl v8 是否兼容 MapLibre v5？（需验证）
- deck.gl 9.x 的 MapboxOverlay 是否与 MapLibre v5 兼容？（有 breaking changes 风险）
- Svelte 5 runes 迁移是否影响 svelte-maplibre？（文档已标注但我未深挖）

这些版本兼容性问题比单个库的 stars 数更影响开发者的实际选型。

**3. 未充分关注供应链安全。**
我只提到了 DuckDB-WASM 的 npm 供应链事件。但从 JS 生态安全角度，应系统性地关注：
- **Leaflet 700+ 插件中的安全风险**：许多插件 3+ 年未更新，可能包含已知漏洞的依赖
- **MapLibre 的 npm 发布流程**：作为关键基础设施，其 npm 发布账户的安全性需关注
- **CesiumJS 的 Cesium ion 依赖**：API key 泄露风险在文档中提到但未归入安全专题

**4. 对"内容重叠"问题描述不够精准。**
生产专家明确指出了 12 个跨目录重复的工具，并判断"这不是从不同角度看同一工具，而是相同的 caveats、相同的代码示例、相同的评级"。我在 Phase 1 只指出了 frontend-integration.md 与 framework-integration.md 的重叠，遗漏了更大范围的重复问题。

---

### 二、对架构评审专家报告的专业意见

#### 认同的判断

**1. t-rex 应移除：完全认同。**
我的 Phase 1 已通过 WebSearch 确认 t-rex 项目已正式废弃。架构专家建议移除，生产专家也建议删除。三方一致。

**2. Pelias 应移除或极度精简：认同。**
从 JS 生态角度补充：Pelias 的 npm 生态确实几乎不存在。其 JS 客户端 `@mapzen/mapzen.js` 已废弃，现有 Pelias API 客户端下载量极低。Geocode.earth（Pelias 的商业托管版）是更实际的建议。

**3. pg_tileserv / pg_featureserv 降级至 Tier 3：认同。**
Martin + FastAPI 确实完全覆盖了它们的功能。但建议保留一句话提及（而非完全删除），因为它们的"零配置自动发现"特性对快速原型有独特价值。

**4. 增加"迁移路径"章节：强烈认同。**
从 JS 生态角度补充三条关键迁移路径：
- **Mapbox GL JS v1 -> MapLibre GL JS**：这是目前最常见的迁移需求（因 Mapbox 许可变更），涉及 npm 包替换、样式 URL 更改、API key 处理
- **Leaflet -> MapLibre GL JS**：当项目超越 5K 要素限制时的自然迁移路径
- **react-map-gl (Mapbox) -> react-map-gl (MapLibre)**：只需更改导入路径从 `react-map-gl` 到 `react-map-gl/maplibre`

#### 需要纠正/补充的判断

**1. 架构专家写"MapLibre GL JS 4.x 活跃迭代"——版本号错误。**
MapLibre 当前版本为 v5.21.0，不是 4.x。架构专家的报告中多处引用了文档中的 "4.x" 标注而未验证。这进一步印证了我 Phase 1 的建议：移除硬编码版本号。

**2. 架构专家写"react-map-gl v7"——版本号错误。**
同理，react-map-gl 当前为 v8.1.0。架构专家的工具评估表沿用了文档中的过时版本号。

**3. 架构专家对 SvelteKit 的评估缺乏 JS 生态细节。**
报告只说"招聘难度"，但从 JS 生态角度，Svelte 5 的 runes 迁移是一个更紧迫的风险：
- svelte-maplibre 当前仍基于 Svelte 4 语法
- Svelte 5 的 breaking changes 可能导致 wrapper 需要重写
- MIERUNE 团队已推出了一个独立的 `svelte-maplibre-gl`（Svelte 5 原生），可能与 dimfeld 的 `svelte-maplibre` 产生社区分裂

**4. 架构专家建议"增加 ADR 模板"——方向正确但超出文档定位。**
当前文档是工具选型指南，不是项目管理模板。ADR 模板虽有价值，但放在这里会偏离核心目标（精简、快速检索）。建议改为在 README 中链接到外部 ADR 资源。

---

### 三、对生产就绪专家报告的专业意见

#### 认同的判断

**1. DuckDB-WASM 生产就绪度被轻微美化：认同。**
从 JS 生态角度补充：
- DuckDB-WASM 的 npm 供应链事件（2025 年 9 月）虽已修复，但暴露了 WASM 包分发的固有风险
- 4-8MB WASM 包在移动端网络环境下的加载时间问题不仅影响性能，还影响用户留存
- 空间扩展 (`INSTALL spatial`) 需要额外网络请求，离线场景不可用

但我**不认同将评级大幅下调**。DuckDB-WASM 在分析仪表盘场景下确实是革命性的——GeoParquet + DuckDB-WASM + deck.gl 这条管道正在快速被采纳。建议保持 3/5 评级，但在文档中更明确地区分"分析场景"和"交互式地图场景"的适用性。

**2. Yjs CRDT 的生产运维成本被低估：部分认同。**
从 JS 生态角度：
- Yjs 周下载量 ~900K，GitHub stars ~20.9K，生态健康度很高
- y-websocket、y-redis、y-indexeddb 等配套包活跃维护
- **但**生产专家指出的 CRDT 文档膨胀问题是真实的

我的建议是**保持 4/5 评级**但增加明确的生产前提条件：
> "4/5 的评级基于使用 y-redis + 定期文档压缩的前提。仅使用默认 y-websocket 服务器时降为 2/5。"

这比简单调低评级更有指导意义。

**3. 两个目录间的内容重叠是核心问题：强烈认同。**
生产专家列出了 12 个重复工具。从 JS 生态视角，最应优先去重的是：
- **PostGIS 调优**：纯后端内容，不应出现在 js-bindbox
- **Vitest/Playwright/Testcontainers**：测试工具应集中在 web-dev/testing-and-security.md
- **Debezium**：企业 CDC 工具，不应出现在 js-bindbox

而以下重复是合理的（不同深度、不同受众）：
- PMTiles 在 tile-servers.md（详细）和 performance-optimization.md（使用角度）
- Socket.io 在 realtime-offline-advanced.md（JS 用法）和 web-dev/realtime-and-collaboration.md（架构角度）

**4. PostGIS 空间 RLS 性能未量化：认同但超出我的审阅范围。**
这是后端性能问题，我无法从 JS 生态角度补充。但我认同这是一个读者会踩的坑。

#### 需要纠正/补充的判断

**1. 生产专家对 Mosquitto HA 的建议不完整。**
从 JS 生态角度补充：浏览器端 MQTT 客户端 (mqtt.js) 通过 WebSocket 连接到 broker。如果使用 EMQX 替代 Mosquitto 实现 HA，需注意 EMQX 的 WebSocket 端口配置与 Mosquitto 不同（默认 8083 vs 8884）。mqtt.js 客户端代码不需要修改，只需更改连接 URL。

**2. 生产专家未关注 JS 前端库的 breaking changes 风险。**
生产报告关注了后端工具的生产可靠性，但遗漏了前端 JS 库的 breaking changes：
- MapLibre v4 -> v5 改变了事件处理和 source/layer 管理
- react-map-gl v7 -> v8 可能有 API 变更
- ECharts v5 -> v6 是主版本升级，必定有 breaking changes

这些 breaking changes 直接影响生产环境中应用的升级策略。文档应在每个库的 caveats 中标注"最近一次 breaking major version"和"升级指南 URL"。

**3. 生产专家的 GDPR 代码示例问题——我从 JS 生态角度认同。**
`cleanupOldData` 函数中使用字符串插值而非参数化查询确实与文档自身的安全最佳实践矛盾。虽然 `retentionDays` 是数字类型（注入风险极低），但代码示例应该以身作则。

---

### 四、三方统一修正意见

以下是三位专家**一致同意或至少两位同意**的修正建议：

| 优先级 | 修正项 | 三方共识度 | 具体操作 |
|--------|--------|-----------|---------|
| **P0** | 移除硬编码版本号 | 3/3 同意 | MapLibre "4.x"->移除, react-map-gl "v7"->移除, ECharts "5.x"->移除 |
| **P0** | 标记 t-rex 为已废弃 | 3/3 同意 | 更新为"已废弃，替代品为 bbox-tile-server 和 Martin" |
| **P1** | 消除跨目录内容重叠 | 3/3 同意 | js-bindbox 中的 PostGIS 调优、测试工具、Debezium 等改为简短摘要 + 链接 |
| **P1** | 合并 frontend-integration.md 和 framework-integration.md | 2/3 同意（JS生态+生产） | js-bindbox 保留 API 细节，web-dev 聚焦架构层面 |
| **P1** | 降级 pg_tileserv 至 Lower Priority | 2/3 同意（架构+JS生态） | 与 pg_featureserv 一起移至文件底部 |
| **P2** | 移除或极度精简 Pelias | 2/3 同意（架构+JS生态） | 一句话提及即可 |
| **P2** | 增加迁移路径章节 | 2/3 同意（架构+JS生态） | Mapbox->MapLibre, Leaflet->MapLibre, react-map-gl 切换 |
| **P2** | 增加"数据验证日期"机制 | 2/3 同意（JS生态+生产） | 每个文件顶部添加验证日期 |
| **P2** | DuckDB-WASM 明确区分适用场景 | 3/3 同意 | 分析仪表盘: 推荐; 交互式地图: 不推荐 |
| **P2** | Yjs 评级增加前提条件 | 2/3 同意（JS生态+生产） | "4/5 基于使用 y-redis + 定期压缩" |
| **P3** | 增加 Resium 单维护者风险突出标注 | 2/3 同意（架构+生产） | 在摘要级别标注，而非埋在 caveats 中 |
| **P3** | 更新 GitHub stars / npm 数据 | 1/3 建议（JS生态） | 低优先级，数据偏差 <15% 不影响决策 |
