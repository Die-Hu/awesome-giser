# Phase 4 JS 生态校验报告

> 校验人：js-ecosystem-expert
> 日期：2026-03-21
> 目标：验证实践者重写后的文件是否正确执行了三方共识

---

## 校验结果总览

| 文件 | 校验项 | 状态 | 说明 |
|------|--------|------|------|
| README.md | 统一工具索引表 | PASS | 新增 `## Unified Tool Index`，按类别列出所有工具，含一句话描述和状态标签 |
| README.md | 数据验证日期 | PASS | 顶部添加 `> Data verified: 2026-03-21` |
| README.md | t-rex 标记废弃 | PASS | 索引表中标注 `**Deprecated**` |
| README.md | loam 标记废弃 | PASS | 索引表中标注 `**Deprecated**` |
| README.md | Resium 风险标注 | PASS | 索引表中标注 `**single maintainer risk**` |
| README.md | DuckDB-WASM 场景限定 | PASS | 标注 `analysis dashboards, not interactive maps` |
| README.md | Yjs 生产前提 | PASS | 标注 `requires y-redis + compression for production` |
| README.md | pg_tileserv 标注降级 | PASS | 标注 `prototype only` |
| README.md | shapefile.js 遗留标注 | PASS | 标注 `Stable (8yr no release)` |
| README.md | 版本号移除 | PASS | 索引表中无硬编码版本号（MapLibre/react-map-gl/ECharts 均无版本号后缀） |
| 2d-mapping.md | MapLibre 版本号 | **FAIL** | 第128行仍为 `### 3. MapLibre GL JS 4.x`，硬编码版本号未移除 |
| charting-integration.md | ECharts 版本号 | **FAIL** | 第12行 `Apache ECharts 5.x`、第77行 `### 2. Apache ECharts 5.x`，硬编码版本号未移除 |
| charting-integration.md | Plotly.js npm 数据 | **FAIL** | 第138行仍为 `npm downloads: ~3M/week`，未修正（实际 plotly.js 本体 ~235K/week） |
| framework-integration.md | react-map-gl 版本号 | **FAIL** | 第71行仍为 `### 2. react-map-gl v7 (React/Next.js)`，硬编码版本号未移除 |
| tile-servers.md | t-rex 废弃标注 | **FAIL** | 第257-282行仍用旧措辞 `Largely superseded by Martin`，未更新为"已废弃"。无 `Deprecated` 标签 |
| tile-servers.md | pg_tileserv 降级 | **PARTIAL** | pg_tileserv 仍在 Quick Picks 表中（第14行）、仍在 `## Recommended Order` 第4位。文档正文已注明"原型用途"，但层级未调整为 Lower Priority |

---

## 详细分析

### PASS 项：README.md 重写质量很高

README.md 的重写执行了绝大部分三方共识：
- "I Need..." 快速决策格式替代了旧的双路径设计，更直接
- Unified Tool Index 覆盖了所有 7 个类别、~40 个工具
- 状态标签（Active / Stable / Deprecated / Experimental / Active (niche)）分类合理
- 部署成本快速参考（$0-5/mo vs $20-40/mo）非常实用
- 版本号已在 README 层面完全移除

### FAIL 项：子文件中的硬编码版本号未修正

三方 P0 共识"移除硬编码版本号"在 README 层面已执行，但在以下子文件中**未执行**：

1. **2d-mapping.md:128** -- `### 3. MapLibre GL JS 4.x`
   - 应改为 `### 3. MapLibre GL JS`
   - 当前 MapLibre 实际版本为 v5.21.0，"4.x" 已过时

2. **charting-integration.md:12,77** -- `Apache ECharts 5.x`
   - 应改为 `Apache ECharts`
   - 当前 ECharts 实际版本为 v6.0.0，"5.x" 已过时

3. **framework-integration.md:71** -- `react-map-gl v7`
   - 应改为 `react-map-gl`
   - 当前 react-map-gl 实际版本为 v8.1.0，"v7" 已过时

### FAIL 项：tile-servers.md 中 t-rex 废弃标注不充分

tile-servers.md 第257行仍用原文 `**Largely superseded by Martin.**`，但实际 t-rex 项目已**正式废弃**（GitHub 标注 "not maintained anymore"），替代品为 bbox-tile-server。

当前文档措辞给读者的印象是"Martin 更好但 t-rex 仍可用"，实际情况是"t-rex 不再维护，不应选择"。

虽然 README.md 索引表中已正确标注 `**Deprecated**`，但 tile-servers.md 正文——读者实际阅读工具细节的地方——仍未反映这一事实。

### FAIL 项：Plotly.js npm 数据未修正

charting-integration.md 第138行仍为 `npm downloads: ~3M/week`。根据 Phase 1 验证，plotly.js 本体包周下载量约 235K。~3M 可能包含 plotly.js-dist 等衍生包，但文档应澄清统计口径或修正为准确数字。

### PARTIAL 项：pg_tileserv 层级未调整

pg_tileserv 在 tile-servers.md 中仍位于 `## Recommended Order` 第4位（与 PMTiles、Martin 并列为推荐级别）。三方共识建议将其降级至 Lower Priority 或至少标注为"仅限原型"。

README 索引表已标注 `prototype only`，但正文中 pg_tileserv 与其他推荐工具处于同一层级，可能误导读者认为它是生产可用的选择。

---

## 修正建议

以下 5 项修正均为简单文本替换，不影响内容结构：

| # | 文件 | 行号 | 当前文本 | 建议修正 |
|---|------|------|---------|---------|
| 1 | 2d-mapping.md | 128 | `### 3. MapLibre GL JS 4.x` | `### 3. MapLibre GL JS` |
| 2 | charting-integration.md | 12 | `Apache ECharts 5.x` | `Apache ECharts` |
| 3 | charting-integration.md | 77 | `### 2. Apache ECharts 5.x` | `### 2. Apache ECharts` |
| 4 | charting-integration.md | 138 | `~3M/week` | `~235K/week (plotly.js-dist ~1M+)` |
| 5 | framework-integration.md | 71 | `### 2. react-map-gl v7 (React/Next.js)` | `### 2. react-map-gl (React/Next.js)` |

tile-servers.md 中 t-rex 和 pg_tileserv 的层级调整需要更大的结构改动，建议在 Phase 5 整合时处理。
