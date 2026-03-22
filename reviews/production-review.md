# 生产就绪评审报告

## 审阅范围

横跨 `web-dev/` 和 `js-bindbox/` 两个目录，审阅以下 8 个文件中与生产就绪相关的内容：

1. `web-dev/performance.md` - 性能优化
2. `web-dev/testing-and-security.md` - 测试与安全
3. `web-dev/deployment.md` - 部署
4. `web-dev/pwa-and-offline.md` - PWA 与离线
5. `web-dev/realtime-and-collaboration.md` - 实时协作
6. `js-bindbox/performance-optimization.md` - JS 性能优化
7. `js-bindbox/realtime-offline-advanced.md` - 实时/离线/高级
8. `js-bindbox/tile-servers.md` - 瓦片服务器

---

## 工具评估清单

### 1. web-dev/performance.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| Tippecanoe 2.x | 高 | 低 | 好 | 保留 | Mapbox 创建、Felt 维护，行业标准。无运维负担（CLI 工具）。文档准确标注了内存爆炸和 drop 策略不透明等真实问题。 |
| PMTiles | 高 | 极低 | 好 | 保留 | 零服务器架构，CDN 级别扩展。文档诚实标注了静态数据限制和 Range 请求要求。生产可靠性评级 4/5 合理。 |
| COG (Cloud-Optimized GeoTIFF) | 高 | 低 | 好 | 保留 | NASA/USGS/ESA 广泛使用。文档正确强调了 overviews 缺失和 CORS 配置是最常见的生产问题。 |
| PostGIS 调优 | 高 | 中 | 好 | 保留 | 文档质量很高，覆盖了 work_mem 爆炸、GiST 索引膨胀、random_page_cost 误配等真实生产陷阱。**发现问题：** 文档建议 `work_mem=256MB` 但同时警告了这个值在高并发下的风险，建议增加一个推荐的安全起始值（如 64MB）。 |
| Martin 性能调优 | 中-高 | 中 | 好 | 保留（需标注） | pre-1.0 版本但被 Felt 和 MapLibre 生态使用。文档诚实标注了缓存缺失和连接池耗尽问题。**关键发现：** 文档给 Martin 生产就绪评级无明确数字，但 tile-servers.md 给了 4/5，这是合理的——pre-1.0 API 在生产中确实需要谨慎。 |
| DuckDB-WASM | 低-中 | 低 | 中 | 保留（降级警告） | **被美化风险：** web-dev 版本给了生产就绪评级 3/5，这是诚实的。但文档中 "Performance comparison: PostGIS ~5s, DuckDB ~0.8s" 的对比有误导性——这忽略了冷启动 1-3 秒、4-8MB WASM 加载、浏览器内存限制（2-4GB）等关键因素。适合分析场景，不适合替代 PostGIS。 |
| deck.gl 优化 | 高 | 低 | 好 | 保留 | Uber/Foursquare/Google 使用。文档正确标注了 updateTriggers、GPU 内存泄漏等生产问题。 |
| Web Workers | 高 | 低 | 好 | 保留 | 浏览器原生技术，无额外依赖。文档正确标注了序列化开销和 SharedArrayBuffer 限制。 |
| FlatGeobuf | 中-高 | 低 | 好 | 保留 | QGIS/GDAL 支持。文档正确标注了只读限制和 Range 请求依赖。生产就绪评级 4/5 合理。 |
| Nginx 瓦片缓存 | 高 | 低 | 好 | 保留 | 行业标准。配置示例完整且实用。 |
| k6 负载测试 | 高 | 低 | 好 | 保留 | Grafana Labs 产品，生产验证充分。 |

### 2. web-dev/testing-and-security.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| Vitest | 高 | 低 | 好 | 保留 | 现代 JS 测试标准。文档正确标注了 Node.js 无 WebGL 的限制。 |
| Testcontainers + PostGIS | 高 | 低 | 好 | 保留 | 文档正确标注了 CI Docker 依赖和容器启动时间。生产就绪评级 5/5 合理。 |
| Playwright | 高 | 低 | 好 | 保留 | Microsoft 产品。文档诚实标注了地图测试的 flakiness 和内存消耗问题。 |
| Zod GeoJSON 验证 | 高 | 低 | 好 | 保留 | 文档正确标注了拓扑验证缺失（Zod 验证结构，不验证几何有效性）。 |
| PostGIS RLS (空间行级安全) | 中 | 中 | 好 | 保留（需警告） | **生产性能风险：** 文档正确指出 ST_Intersects RLS 策略对每行评估的性能影响，但未量化。在 100 万行表上，空间 RLS 可导致查询时间增加 10-100 倍。文档应更强调：对高吞吐表应优先使用基于属性的 RLS。 |
| JWT 空间声明 | 中-高 | 低 | 好 | 保留 | 实现模式合理。bbox 校验逻辑正确。 |
| SQL 注入防护 | 高 | 低 | 好 | 保留 | 文档提供了清晰的 BAD/GOOD 对比。参数化查询示例正确。 |
| GeoJSON 输入验证 | 高 | 低 | 好 | 保留 | 包含大小限制、特征数限制、坐标计数限制。完整的防 DoS 策略。 |
| GDPR 位置数据 | 中 | 中 | 中 | 保留（需补充） | **不足：** `cleanupOldData` 函数使用字符串插值构造 SQL（`INTERVAL '${retentionDays} days'`），虽然 retentionDays 是数字类型不会导致注入，但与同文件中提倡的参数化查询最佳实践矛盾。应改为参数化方式。 |

### 3. web-dev/deployment.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| Kubernetes | 高 | 高 | 好 | 保留 | 文档诚实标注了成本（EKS $75/mo 控制平面）和 PostGIS StatefulSet 的风险。正确建议使用托管数据库。反模式标注到位。 |
| Docker Compose | 高 | 低 | 好 | 保留 | 文档正确区分了开发/小规模生产的适用场景（~100 并发用户上限）。安全配置示例（read_only、no-new-privileges、cap_drop）是亮点。 |
| Prometheus + Grafana + OTel | 高 | 中 | 好 | 保留 | 文档正确标注了存储持久性（需 Thanos/Mimir）、cardinality 爆炸（per-tile 标签）、告警疲劳等生产问题。 |
| Serverless (Lambda/CF Workers/Supabase) | 中-高 | 低-中 | 好 | 保留 | 文档诚实对比了各平台限制。Lambda 冷启动 3-8s、Workers 1MB 限制、Supabase 连接数限制均已标注。成本不可预测性的警告很重要。 |
| GitHub Actions CI/CD | 高 | 低 | 好 | 保留 | 文档正确标注了 Runner 内存限制（7GB）和 PostGIS 容器启动时间。 |
| PostGIS 备份策略 | 高 | 中 | 好 | 保留 | 涵盖 pg_dump、S3 上传、本地清理。正确强调了 WAL 归档的重要性。 |
| Caddy/Nginx 反向代理 | 高 | 低 | 好 | 保留 | 文档正确区分了 Caddy（简单自动 HTTPS）vs Nginx（最大控制）的场景。瓦片缓存配置示例完整。 |
| Vercel/Netlify | 高 | 低 | 中 | 保留 | 文档正确标注了仅适用于前端托管，不能运行 PostGIS/Martin。内容较薄但准确。 |

### 4. web-dev/pwa-and-offline.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| Workbox 7 | 高 | 低 | 好 | 保留 | Google 产品。文档详细覆盖了缓存失效、存储配额差异（Chrome ~60%、Safari ~1GB）、Range 请求缓存等生产问题。 |
| IndexedDB / Dexie.js | 中-高 | 低 | 好 | 保留（需警告） | **关键生产问题：** 文档正确标注了 iOS Safari IndexedDB 损坏 bug，这是真实的生产风险。建议在文档中更强调：iOS 上必须有服务端备份策略，且应测试低存储设备场景。 |
| rbush | 高 | 低 | 好 | 保留 | Mapbox 创建，~6KB，零依赖。文档正确标注了内存开销（80 bytes/item）和只支持 bbox 查询的限制。 |
| 瓦片下载管理器 | 中 | 低 | 好 | 保留 | 自定义实现，非第三方库。文档正确标注了瓦片数量指数增长和浏览器连接限制。 |
| 同步管理器 | 中 | 中 | 中 | 保留（需警告） | **生产风险：** last-write-wins 冲突解决策略在多用户野外采集场景可能丢失数据。文档提到了这点但未提供替代方案的代码示例。 |

### 5. web-dev/realtime-and-collaboration.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| Socket.io | 高 | 低-中 | 好 | 保留 | 百万级生产部署。文档正确标注了 Redis 依赖（多服务器）、内存消耗（10-50KB/连接）、重连风暴等问题。 |
| MQTT (Mosquitto) | 高 | 中 | 好 | 保留 | IoT 行业标准。**关键发现：** 文档正确标注了 Mosquitto 是单节点的，HA 需要 EMQX/HiveMQ（商业），但未充分说明 Mosquitto 桥接方案的脆弱性。生产环境如需 HA 应直接用 EMQX。 |
| PostGIS LISTEN/NOTIFY | 高 | 极低 | 好 | 保留 | 零额外基础设施。文档正确标注了 8000 字节 payload 限制和无保证交付。这是 90% 场景的正确选择。 |
| Yjs CRDT | 中 | 中-高 | 好 | 保留（需警告） | **生产风险：** 文档正确标注了 CRDT 文档大小单调增长问题，但未量化：长期编辑的地图文档可能膨胀到实际数据的 10-100 倍。y-websocket 默认服务器将所有文档放在内存中，这在生产中不可接受。必须使用 y-redis 或自定义持久化。生产就绪评级 4/5 可能偏高，建议标注为 3.5/5。 |
| Supabase Realtime | 中-高 | 低 | 好 | 保留 | 文档诚实标注了 100-500ms 延迟、连接数限制、RLS 性能影响。生产就绪评级 4/5 对于快速开发场景合理。 |
| Debezium CDC | 高 | 高 | 好 | 保留 | Red Hat 产品。文档正确标注了基础设施重量（Kafka+ZooKeeper）和 WAL 文件累积风险。反模式标注到位："90% 的场景用 LISTEN/NOTIFY 就够了"。 |
| SSE | 高 | 极低 | 好 | 保留 | 浏览器原生。文档正确标注了单向限制和连接数限制。 |

### 6. js-bindbox/performance-optimization.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| Vector Tiles (PMTiles + MapLibre) | 高 | 极低 | 好 | 保留 | 与 web-dev/performance.md 一致，补充了更简洁的入门视角。 |
| Web Workers | 高 | 低 | 好 | 保留 | 与 web-dev 版本一致，内容更精简，适合快速参考。 |
| FlatGeobuf | 中-高 | 低 | 好 | 保留 | 与 web-dev 版本一致。 |
| Flatbush/rbush | 高 | 低 | 好 | 保留 | Flatbush 8 bytes/item vs rbush 80 bytes/item 的对比很有价值。 |
| deck.gl Binary Mode | 高 | 低 | 好 | 保留 | 与 web-dev 版本一致，内容更精简。 |
| DuckDB-WASM | 低-中 | 低 | 中 | 保留（降级警告） | 同 web-dev 版本评估。 |
| PostGIS 调优 | 高 | 中 | 好 | 保留 | 与 web-dev 版本内容高度重叠。**重组建议：** 合并为单一参考。 |
| COG | 高 | 低 | 好 | 保留 | 与 web-dev 版本一致，内容更精简。 |

### 7. js-bindbox/realtime-offline-advanced.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| SSE | 高 | 极低 | 好 | 保留 | 与 web-dev 版本一致，更精简。 |
| Supabase Realtime | 中-高 | 低 | 好 | 保留 | 与 web-dev 版本一致。 |
| Socket.io | 高 | 低-中 | 好 | 保留 | 与 web-dev 版本一致。 |
| MQTT | 高 | 中 | 好 | 保留 | 与 web-dev 版本一致，更精简。 |
| Service Worker + PMTiles | 高 | 低 | 好 | 保留 | 与 web-dev/pwa-and-offline.md 高度重叠。 |
| IndexedDB + Dexie.js | 中-高 | 低 | 好 | 保留 | 与 web-dev 版本一致。 |
| Yjs CRDT | 中 | 中-高 | 好 | 保留（需警告） | 与 web-dev 版本评估一致。 |
| Vitest / Playwright / Testcontainers | 高 | 低 | 好 | 保留 | 与 web-dev/testing-and-security.md 高度重叠。 |
| 安全清单 | 高 | 低 | 好 | 保留 | 与 web-dev 版本重叠但提供了更精简的清单格式。 |
| Debezium | 高 | 高 | 中 | 保留 | 与 web-dev 版本一致，内容更精简。 |

### 8. js-bindbox/tile-servers.md

| 工具名 | 生产可靠性 | 运维成本 | 文档质量 | 评估结论 | 理由 |
|--------|-----------|---------|---------|---------|------|
| PMTiles (Protomaps) | 高 | 极低 | 好 | 保留 | 文档清晰对比了传统瓦片服务 vs PMTiles 的运维差异。与 web-dev 版本一致。 |
| 云瓦片服务 (MapTiler/Stadia/Mapbox) | 高 | 低 | 好 | 保留 | 文档诚实标注了规模成本和 vendor lock-in。免费层对比表很实用。**发现：** Mapbox GL JS v2+ 专有许可已正确标注。 |
| Martin | 中-高 | 中 | 好 | 保留（需标注） | 生产就绪评级 4/5 合理（pre-1.0 但被 Felt 使用）。文档全面覆盖了无内置缓存、无认证、连接池耗尽等问题。 |
| pg_tileserv | 中 | 低 | 好 | 保留（降级） | 生产就绪评级 3/5 合理。**关键问题：** 自动发现暴露所有 PostGIS 表是安全风险。文档正确建议"原型用 pg_tileserv，生产用 Martin"。 |
| TileServer GL | 低-中 | 中 | 好 | 保留（降级） | 生产就绪评级 3/5 合理。Node.js 单线程 + headless GL 依赖使其在生产中脆弱。文档正确将其定位为 WMTS 遗留兼容的专用工具。 |
| t-rex | 低 | 中 | 中 | 建议删除或合并 | **已被 Martin 取代。** 文档自己也说"Martin is the better choice in virtually every scenario"。保留仅增加读者困惑。 |
| GeoServer 2.25 | 中 | 高 | 中 | 保留（仅限 OGC 场景） | 文档诚实标注了 JVM 资源消耗、XML 配置、性能劣势。正确定位为"仅当需要 OGC 合规时才用"。**已知安全问题：** 文档提到了 CVE 和默认管理员密码，这很重要。 |

---

## 生产环境关键发现

### 发现 1：DuckDB-WASM 的生产就绪度被轻微美化

文档中 PostGIS vs DuckDB-WASM 的性能对比（5s vs 0.8s）忽略了 1-3 秒冷启动和 4-8MB WASM 加载。web-dev 版本给的 3/5 评级是诚实的，但"Performance comparison"部分需要加上完整的首次加载时间对比。DuckDB-WASM 适合分析仪表盘，不适合交互式地图应用。

### 发现 2：Yjs CRDT 的生产运维成本被低估

文档给了 4/5 生产就绪评级，但 CRDT 文档的单调增长问题在长期运行的地图编辑场景中是严重的运维隐患。y-websocket 默认服务器将所有文档放在内存中，这在生产中不可接受。建议评级调整为 3.5/5，并明确要求使用 y-redis + 定期文档压缩。

### 发现 3：两个目录之间存在大量内容重叠

以下工具在 `web-dev/` 和 `js-bindbox/` 中有高度重复的内容：
- PMTiles/Vector Tiles（3 处重复）
- Socket.io（2 处）
- MQTT（2 处）
- Supabase Realtime（2 处）
- IndexedDB/Dexie.js（2 处）
- Yjs CRDT（2 处）
- PostGIS 调优（2 处）
- Vitest/Playwright/Testcontainers（2 处）
- SSE（2 处）
- Debezium（2 处）
- Web Workers（2 处）
- FlatGeobuf（2 处）

这不是"从不同角度看同一工具"，而是相同的 caveats、相同的代码示例、相同的评级。这对读者是负担。

### 发现 4：PostGIS 空间 RLS 的性能影响未充分量化

文档提到 ST_Intersects RLS 策略对每行评估有性能影响，但没有给出具体数字。在百万行级表上，这可能导致查询从毫秒级退化到秒级。需要增加具体的性能基准或明确警告"不要在高吞吐表上使用空间 RLS"。

### 发现 5：Martin pre-1.0 状态的风险被合理标注

文档在多处提到 Martin 是 pre-1.0，配置格式在版本间变化，这是诚实且重要的。Felt 在生产中使用它增加了可信度，但用户应注意升级时需要仔细阅读 changelog。

### 发现 6：Mosquitto HA 方案不完整

文档提到 Mosquitto 是单节点的，HA 需要 EMQX/HiveMQ（商业），但对 Mosquitto 桥接方案只说了"fragile"而未解释为什么。如果生产环境需要 MQTT HA，文档应明确推荐 EMQX（开源版）而非 Mosquitto 桥接。

### 发现 7：安全相关内容质量很高

SQL 注入防护的 BAD/GOOD 对比、GeoJSON 输入验证的完整实现、CORS 配置、速率限制——这些都是实用的生产安全指南。testing-and-security.md 是整个项目中质量最高的文件之一。

### 发现 8：t-rex 应被删除

文档自己已经明确说 Martin 在所有场景下都更好。保留 t-rex 只会让读者在两个 Rust 瓦片服务器之间纠结。

### 发现 9：成本对比表非常实用

deployment.md 中的成本对比表（Docker Compose / K8s / Serverless / Static / Supabase 按用户量级对比）是少见的诚实内容，对生产决策有直接帮助。

### 发现 10：可观测性覆盖较好但分散

Prometheus + Grafana + OpenTelemetry 在 deployment.md 中有详细覆盖，但 Martin 的 `/metrics` 端点、PostGIS 的 `pg_stat_statements`、Nginx 的缓存命中率监控分散在不同文件中。生产环境的可观测性应该有一个统一的参考。

---

## 自我反思

1. **对运维成本的判断缺乏具体数据支撑。** 我将运维成本标为"高/中/低"，但没有给出具体的人力成本或时间成本估算。例如，Kubernetes 的"高运维成本"对有专职 DevOps 的团队可能只是"中"。这些判断更多基于行业共识而非精确测量。

2. **对新兴工具可能过于保守。** 我将 DuckDB-WASM 和 Yjs 的评级下调，但这两个工具都在快速成熟。2026 年的评估可能需要上调。DuckDB-WASM 的空间扩展在过去一年进步显著，Yjs 的生态也在扩展。我的评估可能在 6-12 个月后过时。

3. **未验证具体版本的 CVE 记录。** 对 GeoServer 提到了 CVE，但未检查具体的 CVE 数量和严重程度。对 Martin、Mosquitto 等工具也未检查最近的安全公告。这是评审的盲区。

4. **对 Supabase 的评估可能偏保守。** Supabase 在过去一年的增长和稳定性提升可能使得 4/5 的生产就绪评级对中小型项目而言是偏低的。但其连接数限制和 RLS 性能影响仍是真实的生产约束。

5. **未评估工具组合的生产可靠性。** 文档推荐的组合（如 Martin + Nginx + Prometheus + PostGIS）作为整体的生产可靠性未被评估。单个工具可靠不意味着组合可靠——组件间的版本兼容性、配置协调等问题未被考虑。

---

## 对重组的生产就绪建议

### 1. 消除重复，建立单一来源

**核心问题：** 同一工具（如 Socket.io、PMTiles、PostGIS 调优）在 web-dev 和 js-bindbox 中重复出现。读者不知道该看哪个版本。

**建议：**
- `web-dev/` 保持为**完整参考**（详细代码示例、完整配置、深度 caveats）
- `js-bindbox/` 改为**快速决策指南**（每个工具 3-5 行概述 + 链接到 web-dev 的详细内容），不重复代码和配置

### 2. 大型项目路径：增加生产运维专题

建议在 web-dev 中增加或强化以下生产主题（目前分散在各文件中）：
- **可观测性整合**：将 Martin metrics、PostGIS pg_stat_statements、Nginx cache hit rate、OpenTelemetry 统一到一个可观测性章节
- **故障排查指南**：常见生产问题（瓦片 404、连接池耗尽、缓存失效、iOS IndexedDB 损坏）的诊断流程
- **容量规划**：基于用户量级的资源需求估算（目前只有 deployment.md 的成本表，缺少 RAM/CPU/存储的具体数字）

### 3. 小型项目路径：突出最小可行栈

小型项目读者只需要知道：
1. **静态数据**：tippecanoe -> PMTiles -> CDN -> MapLibre（$0-5/mo）
2. **动态数据**：Docker Compose (PostGIS + Martin + Caddy)（$20-40/mo）
3. **离线支持**：Workbox + IndexedDB + PMTiles
4. **实时功能**：Socket.io（简单）或 Supabase Realtime（更简单）

这四个路径应该在 js-bindbox 的开头以决策树形式呈现，每条路径不超过一页。

### 4. 删除或合并的建议

| 操作 | 对象 | 理由 |
|------|------|------|
| 删除 | t-rex (tile-servers.md) | 已被 Martin 完全取代，文档自己也这么说 |
| 合并 | js-bindbox 中重复的工具内容 | 改为简短摘要 + 链接到 web-dev 详细内容 |
| 降低优先级 | GeoServer | 仅保留在"OGC 合规需求"的小节中，不占主要篇幅 |
| 降低优先级 | TileServer GL | 仅保留在"WMTS 遗留兼容"的小节中 |

---

## Phase 2 互审补充

> 基于阅读架构评审报告（arch-review.md）和 JS 生态评审报告（js-ecosystem-review.md）后的补充反思和交叉意见。

### 一、对自身报告的反思与补充

#### 1. 遗漏的审阅范围

我的 Phase 1 报告仅审阅了 8 个生产相关文件，未覆盖架构专家审阅的 `web-dev/fullstack-architecture.md`、`web-dev/backend-services.md`、`web-dev/frontend-integration.md`。这三个文件中包含了以下生产相关工具，我在 Phase 1 中遗漏了评估：

| 工具 | 生产就绪评估（补充） |
|------|---------------------|
| **FastAPI + GeoAlchemy2** | 高可靠性。Microsoft/Netflix/Uber 使用 FastAPI。GeoAlchemy2 成熟。文档中 caveats（GDAL 系统依赖、内存管理）标注到位。 |
| **TiTiler** | 高可靠性。NASA/USGS 生产验证。但 Lambda 冷启动 3-8s 是真实的生产延迟问题，文档已标注。 |
| **OSRM / Valhalla** | 高可靠性。路由引擎在物流行业广泛验证。OSRM 图构建内存需求（planet: 128GB+）是关键运维约束，需确认文档已标注。 |
| **Resium (CesiumJS for React)** | **中-低可靠性。** 架构专家正确识别了单人维护者风险。从生产角度补充：CesiumJS 本身 5MB+ 包体积 + Ion 平台依赖 = 高运维复杂度。如果维护者停止更新，生产应用面临无法升级 CesiumJS 的风险。 |
| **Next.js 15** | 高可靠性。Vercel 维护，App Router 已稳定。但生产部署需注意 ISR 缓存失效和 edge runtime 的 PostGIS 连接限制。 |

#### 2. 版本号问题的生产影响（受 JS 生态专家启发）

JS 生态专家发现了三个主版本号错误（MapLibre 4.x->5.x, react-map-gl v7->v8, ECharts 5.x->6.x）。从生产角度评估这些错误的影响：

| 版本错误 | 生产影响 | 严重程度 |
|---------|---------|---------|
| **MapLibre 4.x -> 5.x** | v5 包含 breaking changes（如 style spec 变更、API 移除）。读者按 4.x 写的代码在 5.x 上可能运行失败。**生产部署风险：高。** | 高 |
| **react-map-gl v7 -> v8** | v8 引入了新的 API surface 和依赖变更。按 v7 文档写的代码可能需要迁移。**生产部署风险：中。** | 中 |
| **ECharts 5.x -> 6.x** | 主版本升级通常包含 breaking changes。如果代码示例基于 5.x API，在 6.x 上可能不兼容。**生产部署风险：中。** | 中 |

**建议：** 同意 JS 生态专家的方案——文档中移除具体主版本号，改为工具名本身。仅在 `package.json` 示例中标注具体版本。

#### 3. t-rex 状态比我估计的更严重

JS 生态专家发现 t-rex **项目已正式废弃**，不再维护，推荐 bbox-tile-server 替代。我在 Phase 1 中仅说"被 Martin 取代"，实际情况更严重。文档需从"Largely superseded"改为"已废弃（deprecated），不再维护"。

#### 4. DuckDB-WASM 供应链安全事件

JS 生态专家提到 "npm supply chain risk -- September 2025 compromise"。这是我在 Phase 1 中遗漏的重要生产安全信息。DuckDB-WASM 曾发生过 npm 供应链攻击事件，这对于将其用于生产环境的团队是关键考量。文档应保留此信息并建议：
- 锁定具体版本号（不使用 `^` 或 `~`）
- 使用 `npm audit` 定期检查
- 考虑使用 lockfile 和 integrity hashes

#### 5. 对中国市场特殊性的认知不足

架构专家和 JS 生态专家均提到了中国市场的特殊需求（GCJ-02 偏移、AntV L7、高德/百度地图）。从生产角度补充：
- 中国大陆部署需要考虑 CDN 的可访问性（Cloudflare R2 在国内可能被限速或不可达）
- 瓦片服务的 CORS 策略在国内 CDN（阿里云 OSS、腾讯 COS）上的配置方式不同
- 国内 Docker 镜像拉取可能需要镜像加速
- 这些都是真实的生产运维问题，当前文档未覆盖

---

### 二、对架构评审报告的生产就绪意见

#### 认同的观点

1. **pg_tileserv 降级至 Tier 3**：完全认同。从生产角度补充——pg_tileserv 的自动发现功能暴露所有 PostGIS 表，这不仅是安全风险，更是生产事故隐患（新建的临时表会自动暴露为瓦片端点）。
2. **Pelias 移除**：完全认同。5-7 个微服务的运维负担在生产环境中是灾难性的，且 Mapzen 倒闭后社区维护缓慢，安全更新无法保证。
3. **t-rex 移除**：完全认同。结合 JS 生态专家的发现，项目已正式废弃。

#### 需要补充的生产风险

1. **Resium 单人维护者风险（架构专家已识别，需量化）**：
   - 生产影响：如果维护者停止维护，CesiumJS 升级（安全补丁、WebGL 兼容性修复）将无法及时应用到 Resium
   - 建议：对于生产环境依赖 3D Globe 的项目，应有 "直接使用 CesiumJS + 自定义 React wrapper" 的 fallback 方案
   - 此风险应从"Tier 2 加强警告"升级为"Tier 3"或"明确不推荐用于生产关键路径"

2. **SvelteKit 的生产就绪度**（架构专家标注为"快速原型"）：
   - 从生产角度补充：SvelteKit 的 adapter 生态（adapter-node, adapter-vercel 等）成熟度参差不齐。SSR + PostGIS 连接的生产案例较少。对于需要地图 SSR 的生产应用，Next.js 或 Nuxt 是更安全的选择。
   - 架构专家的"快速原型"定位准确，但建议明确标注"不推荐用于需要 SSR 地图的生产应用"。

3. **架构专家未评估的运维盲区**：
   - 架构报告未提及**日志管理**：Docker Compose 默认 JSON 日志驱动会填满磁盘（deployment.md 已标注），但 fullstack-architecture.md 的参考栈中未提及日志方案
   - 架构报告未提及**secrets rotation**：Kubernetes Secrets 的 base64 编码不是加密（deployment.md 已标注），但架构报告未强调这一点
   - 建议：架构层面应明确推荐 External Secrets Operator 或 Vault 用于生产密钥管理

4. **缺少"迁移路径"的生产风险评估**（架构专家提议增加迁移章节）：
   - 从 GeoServer 迁移到 Martin：需要评估 OGC 客户端兼容性、WMS/WFS 端点重映射、认证方式变更
   - 从 Leaflet 迁移到 MapLibre：需要评估插件生态兼容性、marker API 差异
   - 迁移过程中的生产停机风险和回滚策略应是迁移章节的核心内容

---

### 三、对 JS 生态评审报告的生产就绪意见

#### 认同的观点

1. **版本号策略——移除硬编码主版本号**：完全认同。从生产角度，版本号错误会导致读者安装错误版本，引发不兼容问题。
2. **"最后验证日期"机制**：强烈认同。生产团队需要知道文档数据的时效性。
3. **统一工具索引表**：认同。对生产团队做技术选型时的快速筛选非常有价值。

#### 需要补充的生产视角

1. **Plotly.js npm 下载量偏差（文档写 ~3M vs 实际 ~235K）**：
   - 从生产角度，下载量偏差本身不影响生产可靠性，但如果读者基于夸大的采用度做技术选型决策，可能高估了社区支持的可获得性。
   - 建议：更正数据，并在文档中明确区分 plotly.js（核心包）和 plotly.js-dist（分发包）的下载量。

2. **shapefile.js 8 年未更新的生产风险**：
   - JS 生态专家标注了这一事实，但未评估安全影响。8 年未更新意味着没有安全补丁。虽然 Shapefile 解析器的攻击面小，但如果用于解析用户上传的 Shapefile，存在理论上的安全风险（恶意构造的 Shapefile 可能触发未修复的 bug）。
   - **建议：** 在文档中标注 "仅用于可信来源的 Shapefile，不建议用于处理用户上传的文件"。

3. **JS 生态专家未覆盖的安全维度**：
   - 报告提到"安全审计缺失"是盲区——这是诚实的自我反思。从生产角度补充几个需要检查的安全点：
     - **Leaflet 插件生态的供应链风险**：Leaflet 的大量第三方插件质量参差不齐，部分插件可能引入 XSS 风险
     - **MQTT.js 的 WebSocket 连接安全**：浏览器端 MQTT over WebSocket 必须使用 wss://（TLS），文档中的示例使用 ws:// 是开发环境配置，需标注"生产环境必须使用 wss://"
     - **Supabase anon key 的暴露风险**：文档代码示例中 `createClient('https://xxx.supabase.co', 'public-anon-key')` 暗示 anon key 可以硬编码在前端，这是正确的（Supabase 设计如此），但应提醒必须配合 RLS 使用

4. **bbox-tile-server 作为 t-rex 替代品**：
   - JS 生态专家建议在 tile-servers.md 中用 bbox-tile-server 替换 t-rex。从生产角度评估：bbox-tile-server 相对较新，生产验证不足。建议的处理方式是：删除 t-rex，在 Martin 条目中加一句 "如需更轻量的 Rust 瓦片服务器，可关注 bbox-tile-server（新兴项目，生产验证有限）"，而不是给 bbox-tile-server 一个完整的条目。

---

### 四、三方共识汇总

以下是三份报告一致同意的结论：

| 共识项 | 三方意见 |
|--------|---------|
| **删除 t-rex** | 架构：移除。JS 生态：已废弃。生产：同意删除。 |
| **DuckDB-WASM 需谨慎标注** | 架构：3/5 已标注。JS 生态：供应链事件。生产：性能对比有误导。 |
| **web-dev 与 js-bindbox 内容重复严重** | 架构：可接受但建议交叉引用。JS 生态：60-70% 重叠。生产：12+ 工具高度重复。**共识：js-bindbox 应改为快速指南 + 链接。** |
| **Martin 生产就绪但需注意 pre-1.0** | 架构：保留，Felt 使用。JS 生态：活跃。生产：4/5 合理，需缓存层。 |
| **Resium 单人维护者是高风险** | 架构：加强风险提示。生产：不推荐生产关键路径。 |
| **版本号不应硬编码** | JS 生态：三个主版本号错误。生产：版本错误导致部署风险。**共识：移除硬编码版本号。** |
| **安全内容质量高** | 架构：独特价值。生产：整个项目中质量最高的文件之一。 |
| **成本对比表极有价值** | 架构：加分项。生产：对决策有直接帮助。 |

### 五、对方报告中未被识别的生产风险（补充）

1. **架构报告中 fullstack-architecture.md 的参考栈缺少熔断/限流**：企业级 GIS 应用的 API 层应包含熔断器（如 Martin 请求超时时快速失败）和限流（如 PostGIS 查询并发限制），这在当前所有文件中均未系统覆盖。

2. **JS 生态报告中 framework-integration.md 的 SSR 地图渲染未评估内存泄漏风险**：MapLibre GL JS 在 Node.js 中无法运行（依赖 WebGL），SSR 场景下通常渲染占位符然后客户端 hydration。但如果处理不当（如在 SSR 阶段尝试实例化 map），会导致 Node.js 进程内存泄漏。这是一个常见的生产事故模式。

3. **两份报告均未评估 CDN 故障场景下的降级策略**：PMTiles on CDN 是推荐架构，但 CDN 故障（虽然罕见）会导致所有瓦片不可用。生产应用应有 fallback（如备用 CDN 或降级到简化地图）。
