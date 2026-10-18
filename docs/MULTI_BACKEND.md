# 多后端部署指南 — 向量一致性与负载均衡

> 场景:同一个 EmbeddingGemma 2 模型部署在多个推理后端(自建 GPU 服务器、oMLX 远程 Mac、云 API……),客户端做路由/容灾/扩容。
> 所有结论实测得出,数据见 [BENCHMARK-OMMLX.md](BENCHMARK-OMMLX.md) 与下文。

## 核心事实(先记住这三条)

1. **同一后端 = 位级稳定**。相同输入+参数重复调用,余弦 = 1.000000。集合内自洽是免费的。
2. **跨后端 ≠ 同一空间**。bf16/Metal 与 fp32/ROCm 对同一输入的向量余弦仅 0.80~0.89(连 16 条向量的质心都偏到 0.825)。这是数值路径差异,**不是 bug,也不会消失**。
3. **排序结构跨后端保留**。分数 Pearson 0.992,相似对 top-20 重叠 14/20 —— 换后端不换"语义地图",只换"坐标尺度"。

## 路由键

```
route_key = (provider, provider_profile, model_id, dtype路径)
# 本仓库实例:
#   ("local",   "strix-halo", "embeddinggemma-2",        fp32/ROCm)
#   ("omlx",    "omlx",       "embeddinggemma-2-bf16",   bf16/Metal)
```

规则:**`route_key` 不同的两个后端,产生的向量不可直接混用**。`model_id` 相同但 dtype 不同也算不同 key。

## "collection" 是什么

向量库(RAG 存储层)的逻辑分区术语 —— pgvector 的一张表 / Qdrant·Milvus 的 collection / 一个内存 dict 索引均可。**一个业务索引(如"产品手册库")= 一个 collection**,它内部的所有向量必须同源。

## 四种混用姿势(按推荐排序)

| | 做法 | 召回 | 复杂度 | 何时用 |
|---|---|---|---|---|
| **A. 按 collection 粘后端** | collection 创建时定死 `route_key`,写入查询都走它 | 100% | 一个路由表 | **默认。** 直接满足负载均衡(不同 collection 分担负载) |
| **B. 双写镜像** | 两后端各存全量,查询双发,分数归一后合并 | ~100% | 2x 存储、2x 写入 | 需要 A/B 灰度、故障切换、就近接入 |
| **C. 直接混检** | 不区分来源,top-k 合并排序 | top-1 -37%(14/16→8/16),top-3 大体保留 | 零改造 | 容忍排名抖动、只要"大致相关" |
| **D. 全量重嵌** | 切换后端时离线重算所有向量 | 100% | 一次性离线任务 | 换后端/换模型/换 dtype 的迁移必经之路 |

实测依据:8 组 zh 查询×4 文档,同后端 recall@1 14/16,跨后端混检 8/16。

## 落地清单(服务端 × 客户端协作)

**写入侧**(谁存向量谁负责):
```json
{"vec": [...], "meta": {"provider": "omlx", "profile": "omlx",
                         "model": "embeddinggemma-2-bf16", "dim": 768}}
```
- `dim` 也要存:MRL 允许 128/256/512/768 混存于库,但**查询维度必须与 collection 主维度一致**。

**查询侧**:
- 请求携带同一 `route_key`,路由层据此选后端(本仓库即 `recipes.get_backend()` 的选择结果)。
- key 不匹配时:告警,或自动降级到方案 B 双查。

**容量规划**:
- ❌ 别指望异构混发提升吞吐:两引擎内部各自串行(实测 coral ~94 req/s、tufa ~37 req/s 上限),16 并发混发仅 68/s,**低于单压强后端**。
- ✅ 扩吞吐 = 同 `route_key` 加副本(LB 轮询同 key 的多个实例),或按 collection 天然分片。

**迁移**(A→D 流程):
1. 新后端建 shadow collection,双写
2. 离线重嵌存量(批量接口,本机实测 oMLX 515 vec/s、strix 385 vec/s)
3. 灰度切读 → 校验 recall(用固定 query 面板对拍)
4. 下线旧 collection

## 本仓库的现成挂钩

- `recipes.get_backend()` —— `PROFILE` 决定后端,`backend.name` 即路由键的 provider 位
- `backend.capabilities` —— 路由前检查能力(如 omlx 暂无原生 video/audio 稳定路径),自动降级
- `scripts/ab_bench.py` —— 双后端对拍基准,扩了新后端先跑这个
- `test_omlx.py` 的 `test_registry_*` —— 注册表契约测试,新增 recipe 照抄
