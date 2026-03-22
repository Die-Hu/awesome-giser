# Phase 4 生产就绪校验报告

> 校验人：production-expert
> 日期：2026-03-21
> 目的：检查两位实践者重写的文件是否正确执行了三方共识中的生产相关修正

---

## 1. web-dev/performance.md -- DuckDB-WASM 冷启动说明

### 检查项：DuckDB-WASM 是否增加了冷启动说明

**结果：部分通过**

- **已有内容（第 491 行）：** `Cold initialization. First query takes 1-3 seconds for WASM compilation + extension loading. Subsequent queries are fast.` -- 这在 Caveats 部分已正确标注冷启动。
- **未修正（第 480-482 行）：** 代码注释中的误导性性能对比仍然存在：
  ```
  // Performance comparison (1M buildings, analytical query):
  // PostGIS: ~5 seconds
  // DuckDB:  ~0.8 seconds (columnar + vectorized execution)
  ```
  这个对比忽略了首次加载的 1-3 秒冷启动 + 4-8MB WASM 下载时间。三方共识要求修正此对比，但**未执行**。

**建议修正：** 在代码注释中补充：
```
// ⚠️ Note: First query has additional 1-3s cold start + 4-8MB WASM download.
// The 0.8s figure is for warm (subsequent) queries only.
```

### 额外检查：DuckDB-WASM 供应链安全事件

**结果：未添加**

三方共识中提到的 September 2025 npm supply chain compromise 事件未在此文件中标注。建议在 Caveats 部分增加：
```
- **Supply chain risk.** DuckDB-WASM experienced an npm supply chain compromise in September 2025.
  Pin exact versions, use lockfiles with integrity hashes, and run `npm audit` regularly.
```

---

## 2. web-dev/testing-and-security.md -- GDPR 与 RLS

### 检查项 A：cleanupOldData 是否改为参数化查询

**结果：未修正**

第 741-744 行仍然使用字符串插值：
```typescript
async cleanupOldData(retentionDays: number = 90) {
    await db.query(
      `DELETE FROM user_locations WHERE created_at < NOW() - INTERVAL '${retentionDays} days'`
    );
}
```

虽然 `retentionDays` 是 number 类型不会导致 SQL 注入，但这与同文件中提倡的"永远使用参数化查询"最佳实践矛盾。在一份安全指南文件中出现非参数化写法，会给读者错误的示范。

**建议修正为：**
```typescript
async cleanupOldData(retentionDays: number = 90) {
    await db.query(
      `DELETE FROM user_locations WHERE created_at < NOW() - make_interval(days => $1)`,
      [retentionDays]
    );
}
```

### 检查项 B：PostGIS RLS 是否增加性能量化

**结果：未修正**

第 486 行仍然是定性描述：`RLS policies with ST_Intersects are evaluated for every row. On large tables, this can significantly slow queries.`

"significantly" 是模糊的。三方共识要求量化性能影响。

**建议修正为：**
```
- **Performance impact.** RLS policies with ST_Intersects are evaluated for every row.
  On tables with 1M+ rows, spatial RLS can increase query time by 10-100x compared to
  non-RLS queries (e.g., from 50ms to 500ms-5s). Use partial indexes on user_regions
  and prefer attribute-based RLS on high-throughput tables.
```

---

## 3. web-dev/realtime-and-collaboration.md -- Yjs 与 Mosquitto

### 检查项 A：Yjs 评级是否调整

**结果：未修正**

第 513 行仍然写 `Production-readiness 4/5`。三方共识建议调整为 3.5/5，因为 CRDT 文档单调增长和 y-websocket 默认内存存储是严重的运维隐患。

**建议修正为：**
```
Production-readiness 3.5/5 -- used by major collaborative editors, but CRDT document
size growth and y-websocket memory limits require careful production planning. For
enterprise collaborative GIS, implement periodic document compaction and use y-redis
(not the default in-memory y-websocket) for multi-server deployments.
```

### 检查项 B：Mosquitto HA 是否推荐 EMQX

**结果：部分通过**

第 335 行提到了 EMQX：`For HA, need EMQX or HiveMQ (commercial) or bridge multiple Mosquitto instances (fragile).`

EMQX 已被提及，但措辞将其与 HiveMQ 并列为商业方案。实际上 **EMQX 有开源版**（Apache 2.0 许可），这对于生产环境是重要的区分。

**建议修正为：**
```
- **Clustering is limited.** Mosquitto is single-node only. For HA, use EMQX
  (open-source, Apache 2.0, supports native clustering) or HiveMQ (commercial).
  Do not use Mosquitto bridge for production HA -- it is fragile and lacks
  automatic failover.
```

---

## 4. web-dev/deployment.md -- 可观测性与成本表

### 检查项 A：可观测性清单是否增加

**结果：未添加**

当前可观测性内容（Prometheus + Grafana + OTel，第 163-212 行）保持原样。三方共识建议增加统一的可观测性清单，将分散在各文件的 Martin `/metrics`、PostGIS `pg_stat_statements`、Nginx 缓存命中率、OpenTelemetry trace 整合为一个 checklist。

**未执行。** 建议在 Monitoring Stack 部分增加：

```markdown
#### Production Observability Checklist

| Component | Metric Source | Key Metrics |
|-----------|-------------|-------------|
| Martin | `/metrics` (Prometheus) | tile_request_duration_seconds, connection_pool_usage |
| PostGIS | postgres_exporter | pg_stat_activity_count, pg_stat_statements (slow queries), dead_tuples |
| Nginx/Caddy | access log + stub_status | cache_hit_rate (target >90%), request_rate, 5xx_rate |
| Application | OpenTelemetry SDK | spatial_query_duration, feature_count_per_request |
| Infrastructure | node_exporter | disk_usage (especially /var/lib/postgresql), memory, CPU |
```

### 检查项 B：成本表是否增加团队规模列

**结果：未修正**

第 376-382 行的成本对比表仍然是原始格式，只按用户量级分列（1K/10K/100K users/day），未增加团队规模维度。

**建议修正为：**

```markdown
| Stack | Team Size | 1K users/day | 10K users/day | 100K users/day |
|-------|-----------|-------------|--------------|----------------|
| Docker Compose (VPS) | 1-2 | $10-20 | $40-80 | N/A (scale limit) |
| Kubernetes (AWS/GCP) | 5+ (需 DevOps) | $100-200 | $200-500 | $500-2000 |
| Serverless (Lambda) | 1-3 | $5-15 | $30-100 | $200-800 |
| Static (PMTiles+CDN) | 1 | $0-5 | $5-15 | $15-50 |
| Supabase | 1-2 | $0-25 | $25-100 | $100-400 |
```

---

## 5. js-bindbox/realtime-offline-advanced.md -- 精简摘要+链接

### 检查项：是否已改为精简摘要+链接到 web-dev

**结果：未修正**

文件仍然是完整的独立内容（约 498 行），包含完整的代码示例、完整的 Caveats、完整的工具评估。与 `web-dev/realtime-and-collaboration.md`、`web-dev/pwa-and-offline.md`、`web-dev/testing-and-security.md` 存在大量重复。

三方共识明确要求：**js-bindbox 应改为快速决策指南 + 链接到 web-dev 详细内容。**

**此项是最大的未执行项。**

---

## 校验总结

| 检查项 | 状态 | 严重程度 |
|--------|------|---------|
| DuckDB-WASM 冷启动说明（Caveats 部分） | 已有 | -- |
| DuckDB-WASM 性能对比注释修正 | **未执行** | 中 -- 误导性能预期 |
| DuckDB-WASM 供应链安全事件 | **未执行** | 中 -- 安全信息缺失 |
| GDPR cleanupOldData 参数化 | **未执行** | 中 -- 安全示范不一致 |
| PostGIS RLS 性能量化 | **未执行** | 低 -- 定性描述已有，量化缺失 |
| Yjs 评级调整至 3.5/5 | **未执行** | 低 -- 评级偏高但已有充分 caveats |
| Mosquitto HA 推荐 EMQX | 部分通过 | 低 -- 已提及但未标注开源 |
| 可观测性清单 | **未执行** | 中 -- 生产关键信息分散 |
| 成本表增加团队规模 | **未执行** | 低 -- 增值信息 |
| js-bindbox 改为精简摘要+链接 | **未执行** | **高** -- 三方共识核心建议 |

**通过率：1/10 完全通过，1/10 部分通过，8/10 未执行。**

### 整体评估

文件内容在 Phase 3 重写后**未针对三方共识的生产修正建议进行修改**。文件保持了 Phase 1 时的原始内容。这意味着实践者的重写工作可能聚焦在内容组织和表达方式上，而非执行专业评审中的具体修正建议。

**建议：** 在 Phase 5 整合阶段，由内容编辑统一执行上述 10 项修正。优先级排序：
1. **高优先级：** js-bindbox 精简（结构性变更）
2. **中优先级：** DuckDB 性能对比修正、供应链安全、GDPR 参数化、可观测性清单
3. **低优先级：** Yjs 评级、Mosquitto EMQX 开源标注、RLS 量化、成本表团队规模
