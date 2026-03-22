# Phase 4 架构校验报告

**校验人**: arch-reviewer
**日期**: 2026-03-21
**目的**: 验证 Phase 3 实践者重写是否正确执行了三方共识

---

## 校验结果总览

| 文件 | 校验状态 | 说明 |
|------|---------|------|
| `backend-services.md` | **未修改** | 三方共识的修改均未执行 |
| `frontend-integration.md` | **未修改** | 版本号硬编码仍在，Resium 风险提示未强化 |
| `fullstack-architecture.md` | **未修改** | 架构模式本身合理，但版本号引用仍含 "Next.js 15" |

---

## 逐文件校验

### 1. backend-services.md

#### P0: t-rex 是否已移除或标记废弃？

**结果: 未执行**

t-rex 仍保留完整的 Tier 3 条目（第 490-506 行），措辞仍为 "Superseded by Martin"，未更正为"已废弃（Deprecated）"。三方共识（架构/JS生态/生产）均建议移除或标记废弃，但实践者未做任何修改。

当前内容：
```
### t-rex -- Rust Vector Tile Server
⚠️ **Superseded by Martin.** Was competitive before Martin matured.
```

应为（极简保留方案）：
```
### t-rex -- Deprecated
⚠️ **已废弃。** 项目不再维护，推荐迁移至 Martin。
```

或直接移除整个条目。

#### P1: pg_tileserv / pg_featureserv 是否已降级至 Tier 3？

**结果: 未执行**

两者仍在 "Tier 2 -- Solid, With Caveats" 章节下（第 350 行、第 374 行）。三方共识建议降级至 Tier 3，但未做任何调整。

#### P2: Pelias 是否已精简？

**结果: 未执行**

Pelias 仍保留完整的 Tier 3 条目（第 445-464 行），包括详细的 caveats 和大小项目说明。三方共识建议极度精简或移除，但未做修改。此外，Enterprise Quick Picks 仍推荐 Pelias 作为地理编码方案（第 12 行），这与 Tier 3 定位矛盾。

---

### 2. frontend-integration.md

#### P0: 版本号硬编码是否已移除？

**结果: 未执行**

以下硬编码版本号仍存在：
- 第 6 行: `react-map-gl v7 + MapLibre GL JS 4.x`
- 第 19 行: `### react-map-gl v7 + MapLibre GL JS 4.x`

JS 生态专家确认实际版本为 react-map-gl v8 和 MapLibre GL JS v5.x。三方共识建议移除正文中的硬编码主版本号。

#### Resium 单人维护者风险是否在摘要级别突出？

**结果: 部分已有，无需大改**

Enterprise Quick Picks 第 8 行已有 "-- but understand the cost" 的风险提示。正文第 188 行和第 196 行已明确标注单人维护者风险。这部分原始内容已基本到位，但摘要中未直接提及"单人维护者"这个具体风险。

建议 Quick Picks 改为：
```
> - **React + 3D globe:** Resium (CesiumJS for React) -- single maintainer risk, 5MB+ bundle
```

---

### 3. fullstack-architecture.md

#### 架构模式是否合理？

**结果: 合理，无需修改**

微服务 vs 模块化单体的决策框架、反模式警告、大小项目分级建议——均维持 Phase 1 评审时的质量。架构模式部分本身不在三方共识的修改范围内。

#### 版本号引用

Enterprise Quick Picks 第 6 行包含 "Next.js 15"。Next.js 15 是当前版本，未过时，保留合理。但 "MapLibre" 在此文件中未附带版本号，符合共识。

---

## 校验结论

**Phase 3 实践者未对 web-dev/ 下的这三个文件执行任何三方共识修改。** 所有 P0/P1/P2 级修正项均未落地：

| 优先级 | 修正项 | 状态 |
|--------|--------|------|
| P0 | t-rex 移除或标记废弃 | 未执行 |
| P0 | 版本号硬编码移除 (frontend-integration.md) | 未执行 |
| P1 | pg_tileserv/pg_featureserv 降级至 Tier 3 | 未执行 |
| P2 | Pelias 精简 | 未执行 |
| P2 | Resium 风险摘要强化 | 未执行（原始内容部分覆盖） |

**建议**: 这些修改应在 Phase 5（内容整合编辑）中统一执行，或由 team-lead 指派专人处理。修改量不大——每个文件仅需数行调整。
