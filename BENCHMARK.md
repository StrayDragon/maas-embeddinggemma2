# 性能报告 — embeddinggemma2-serve

环境: AMD Ryzen AI MAX+ 395 / Radeon 8060S (gfx1151, ROCm 7.2) · torch 2.14.1+rocm7.2 · **fp32** · 16 核 CPU / 122G 统一内存
测量层: **HTTP 端到端**(含 JSON/base64 传输 + tokenize + 推理), 单客户端顺序, 服务端单飞锁。
复现: `cd sdk-test && bun run bench`（`--quick` 快速档）

## 文本（768d）

| 场景 | p50 | p95 | 吞吐(均值) |
|---|---|---|---|
| 单条 | **24 ms** | 26 ms | ~42/s |
| batch ×8 | 29 ms | 31 ms | ~275/s |
| batch ×32 | 64 ms | 65 ms | ~500/s |
| batch ×64 | 120 ms | 121 ms | **~530/s** |
| batch ×32 @256d(MRL) | 49 ms | 50 ms | ~650/s |
| batch ×32 +`Document`前缀 | 82 ms | 84 ms | ~390/s |

要点: 批量越大单位成本越低;**256d 截断反而最快**(池化后切片+归一,省投影?主因是后续向量运算变小);任务前缀约 +25%(前缀 token 进序列)。

## 多模态（单条 → 批量）

| 场景 | p50 | 批量摊薄 |
|---|---|---|
| 图片 224px | 644 ms | ×8 → 627 ms/张 |
| 音频 16s wav | 224 ms | ×8 → 196 ms/条 |
| 视频 2s(16帧@8fps) | 547 ms | ×4 → 538 ms/条 |
| 混合 文本+图+视频 | — | ×4 → 1194 ms/条 |

要点: 视觉塔是时延大头(644ms/图),音频/视频解码便宜(torchcodec),混合输入近似各模态之和。

## 并发

8 客户端×10 轮单条: wall 1.83s → **44 req/s**（服务端锁串行,单卡推理本就全局吞吐最优;客户端侧重试/退避即可)。

## CPU fallback（无 GPU 参考,进程内直测）

| batch | 耗时 |
|---|---|
| ×1 | 59 ms |
| ×8 | 77 ms |
| ×32 | 137 ms |

CPU 文本路径完全可用(~230/s@32);多模态未测,预期视觉塔慢 3-5 倍。

## 冷启动 / 资源

- 模型加载+首次编码就绪: ~3 s(页缓存热;冷页首次到 10 s 级)
- 常驻显存(GTT): fp32 权重 ~3 GiB + 激活,远低于 122G 统一内存上限
- 批 64 文本峰值 < 1 GiB 激活

## 已知优化空间（未实施）

1. **图片 645ms 偏高**: 疑 CPU 侧 resize/normalize 单线程;可试 `DTYPE=bfloat16`(注意本机 bf16 注意力偶发 NaN,见 README)、vision 预处理批量化、或 torchvision GPU resize。
2. 长视频(>30s)按 1fps 采样,时延线性;可用 `video_metadata` 预降采样。
3. 多并发吞吐受单飞锁限制;如需多租户可改信号量+批聚合(dynamic batching)。

> oMLX 远程后端(`ommlx` profile)的同口径对比数据见 [BENCHMARK-OMMLX.md](BENCHMARK-OMMLX.md)。

## 结论

单卡 APU 上: 文本检索级 500+ vec/s,单条 24ms;多模态单条亚秒。对 RAG(百级 chunk 批量嵌)与实时查询(24ms)均充裕。
