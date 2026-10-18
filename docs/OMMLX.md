# oMLX 后端(ommlx profile)

在 Mac 上用 [oMLX](https://github.com/jundot/omlx) 直接 serve EmbeddingGemma 2,本仓库作为它的客户端/测试台。本机不再装 torch —— 推理与权重都在 oMLX 侧。

## 快速开始

```bash
just setup ommlx          # 轻量 venv(仅 requests/numpy/pydantic,见 requirements-ommlx.txt)
# 编辑 .env(已由 setup 写好 PROFILE=ommlx,补两行):
#   OMLX_BASE_URL=http://coral:11534
#   OMLX_API_KEY=<你的 key>(勿提交;.env 已 gitignore)
just ommlx-config         # 校验配置与连通性
just test-ommlx           # 22 项活体断言(维度/MRL/跨模态语义/鉴权/容错)
```

SDK 侧回归(`test-sdk/client/mm/real/bench`)会经 `scripts/with_target_env.sh` 自动拿到远端地址与模型 id,同一套命令在本地/ommlx 两种 profile 下都能跑:

```bash
just test-client          # 30/30 PASS(远端模式自动切换断言)
just test-real            # 真实素材 zh 检索 top1 4/4
just recall               # Python 跨模态 sanity 3/3
```

## 实测能力矩阵(oMLX 0.7.1rc1 + `embeddinggemma-2-bf16`)

`scripts/probe_omlx.sh` 可随时重测。结论:

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| 文本(单条/批量) | ✅ | `input: str \| str[]` |
| OpenAI content parts | ✅ | `{"type":"text"\|"image_url"\|"input_audio"}` 折叠为一个多模态输入 |
| `items[]` 结构化 | ✅ | 每项 `{text?, image?, audio?}`,可混合批量 |
| 图片 | ✅ | 仅接受 `data:image/...;base64,` data URI(URL/路径会被 400) |
| 文+图融合 | ✅ | 单输入内 text+image → 一个 768 维向量 |
| MRL 截断 | ✅ | `dimensions: 128/256/512/768`,服务端截断后**重新归一化**(实测 norm=1.0000) |
| usage 计数 | ✅ | 图片计 280 tokens(与模型卡一致) |
| 音频 | ⚠️ 服务端 500 | 开关 `embedding_audio_enabled` 已在 Web UI 打开(tower 常驻 ~1.5 GB,错误从 "not supported" 变 500 可证);但 0.7.1rc1 的音频推理对**所有**编码/时长/形状都返回 500(wav 8-48k 0.25-10s 含上游测试同款 fixture、mp3/ogg/opus、items 与 content-parts 两种形状、预热/重载后均复现)→ 需 coral 服务端日志定位,客户端无进一步空间;`test_ommlx` 条件跳过 |
| 视频 | ⚠️ 客户端抽帧 | `/v1/embeddings` 无 video 字段;`recipes.ommlx_backend.embed_video()` 已封装(fps 抽帧→逐帧 embed→均值归一),实测 zh 查询对双视频判别 2/2 top1 正确,与最佳单帧差距 ~0.01 |
| 任务前缀 | ✅ 手动 | 服务端不注入 prompt 模板,`task: ... \| query: ...` 直接写进文本(实测 steered 0.778 / plain 0.827 vs doc,均远超随机) |
| `max_model_len` | 262144(声明) | 实际 embedding 上下文 8192,超长输入服务端静默截断,不报错 |
| 多模型 | ✅ | 同服 bge-m3(dim 1024)与 eg2(dim 768)各自成空间,勿跨模型比相似度 |

### 与本地 server(`server.py`)的语义差异

| | 本地 `server.py` | oMLX `/v1/embeddings` |
| --- | --- | --- |
| 任务前缀 | `prompt_name` 预设(15+) | 客户端自行拼前缀 |
| 图片 | path / URL / b64 均可 | 仅 data URI |
| 多图/视频/交错 | 完整 | 单图;视频需客户端抽帧 |
| DashScope 兼容层 | 有(`/api/v1/...`) | 无(用 items 模拟,语义一致) |
| `/metadata` `/health` 详情 | 有 | 仅 `/health`(无需鉴权) |

## 后端抽象(`recipes/`)

所有客户端代码经统一注册表选路,`PROFILE` 决定 recipe,调用侧零分支:

```python
from recipes import get_backend
b = get_backend()            # PROFILE=ommlx -> oMLX;cpu/strix-halo -> 本地 server
b.embed_text("q", task="search", dim=256)   # 任务前缀差异被 recipe 吸收
b.embed_file("clip.mp4")                    # ommlx 下自动走抽帧;local 走原生 video
b.embed_item({"text": "t", "image": "p.png", "audio": "a.wav"})
b.capabilities                               # 按能力降级而非 try/except
```

- `recipes/local_backend.py` -- 本仓库 server(全能力,`prompt_name` 预设)
- `recipes/ommlx_backend.py` -- oMLX(能力矩阵见上;video 抽帧/任务前缀注文本都在这层吸收)
- `omlx_client.py` 保留为兼容 facade,实现已全部移入 recipes

## 客户端用法(`omlx_client.py`,零 torch 依赖)

```python
from omlx_client import embed, cosine

embed("hello")                                        # -> [[...768]]
embed(["a", "b"])                                     # 批量
embed({"text": "红方块", "image": "poster.png"})       # path 自动转 data URI
embed([{"type": "text", "text": "t"},
       {"type": "image_url", "image_url": {"url": data_uri}}])  # OpenAI parts
embed("q", dimensions=256)                            # MRL,已归一化
```

错误统一 `OmmlxError(status, body)`:401(鉴权)、400(形状/媒体)、404(模型名)、0(不可达,提示检查 `OMLX_BASE_URL`)。

## oMLX 侧模型设置备忘(源码结论,来自 ../omlx)

- `embedding_audio_enabled`(默认 off):加载 audio tower,`items[].audio` 才可用;`embedding_audio_max_seconds` 控制截断(模型上下文内约 327 s)。
- 音频错误文案即开关提示:`... does not support audio inputs (enable embedding_audio_enabled in its model settings)`。
- admin API(`PUT /admin/api/models/{id}/settings`)只认主密钥 Bearer 或 Web UI session;子密钥仅用于 `/v1/*` 推理。
- eg2 的图片预处理有一个上游 PIL channel-layout 补丁(`omlx/patches/embedding_gemma2_image_layout.py`),1px/3px 高的图会静默降质,0.7.1rc1 已修。
- 改完模型设置会触发 engine 重载(`embedding_audio_enabled` 在 runtime signature 里),首次调用会有 ~15 s 冷启动。

## 跨后端一致性 与 负载均衡可行性(实测)

「相同输入+参数,两个后端返回同一向量吗?」—— **否,但受控**:

| 层面 | 实测结论 |
| --- | --- |
| 单后端确定性 | ✅ cos=1.000000(位稳定,同输入重复调用不变) |
| 跨后端同一输入 | cos ≈ 0.80-0.89(bf16/Metal vs fp32/ROCm 的数值路径差异,含 centroid 偏移 cos 0.825) |
| 排序结构 | ✅ 保持:跨后端 score Pearson 0.992,top-20 相似对重叠 14/20 |
| 混用向量库 | ⚠️ 召回损失实证:8 组 zh 查询/4 文档,same-backend recall@1 14/16,**跨后端混检 8/16** |

**负载均衡结论**:
- ✅ 可行且推荐:**按 collection 分流** —— 索引和查询始终落在同一后端(一致性哈希/路由表),单后端位稳定保证集合内自洽。
- ⚠️ 谨慎:**同一 collection 混双后端写入** —— 跨后端混检 recall@1 掉 37.5%(14/16→8/16);必须双写+分别检索,或全量重嵌。
- ❌ 无意义:纯吞吐型 round-robin —— 两引擎内部各自串行(coral ~94/s 上限,tufa ~37/s),混发 16 并发仅 68/s,反而低于单压 coral。要扩吞吐应加 coral 侧副本,而非异构混发。

> 数据:16 样本云质心分析 + 8 组检索面板;复测脚本思路见 `scripts/ab_bench.py`(请求构造同款)。

## 已知坑

1. **图片必须 data URI** —— 客户端 `omlx_client` 已自动转换;裸 URL 会被 oMLX 拒绝(与本地 server 不同)。
2. **音频 500** —— 开关已开(见上);0.7.1rc1 服务端推理持续 500,`test-ommlx` 条件跳过。修复需 coral 侧 `omlx` 服务日志(stderr)定位 mlx_vlm 音频链路异常。
3. **无视频入参** —— 需要视频语义时客户端抽帧(ffmpeg/av)转图片批量。
4. **8192 才是真实上下文** —— `/v1/models` 的 262144 是 RoPE 上限;超长文本会被截断而非报错,重要内容放前面。
5. **两个模型两套空间** —— bge-m3 与 eg2 的向量不可混用;`dim` 字段可作断言。
