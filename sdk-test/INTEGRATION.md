# TS 生态接入 embeddinggemma2-serve —— 实测经验报告

> 基于 ai@7.0.138 + @ai-sdk/openai@4.0.92 + bun 1.4 / node 26 实测。
> 服务端两个入口: `POST /v1/embeddings`(OpenAI 兼容) 与 `POST /embed`(全模态原生)。

## 1. AI SDK 实测结论

### 即插即用的部分

```ts
const openai = createOpenAI({ baseURL: "http://host:8000/v1", apiKey: "anything" });
const { embedding, usage } = await embed({ model: openai.textEmbeddingModel("embeddinggemma-2"), value: "..." });
```

- `baseURL` 覆盖干净利落,自建服务零适配;`embedMany` 批量、`usage.tokens`(服务端真实 tokenizer 计数)都正常。
- `providerOptions: { openai: { dimensions: 256 } }` → 服务端 MRL 截断,类型提示完好。
- 错误是 `APICallError`,`statusCode` 可靠;但 `message` 只有 "Bad Request"——**detail 在响应体里,SDK 不透传**,排错要看 `error.responseBody` 或服务端日志。

### 协议级局限(改不了的,不是 SDK bug)

| 局限 | 影响 | 我们的对策 |
| --- | --- | --- |
| `embed()` 的 `value: string` —— OpenAI embeddings 协议**没有多模态** | 图片/音频/视频无法走 SDK | 原生 `POST /embed`,任意 HTTP 客户端直连(见下) |
| 无"任务前缀"概念(EmbeddingGemma 的 prompt 工程是质量关键) | 检索/聚类不对称场景拿不到最优质量 | **模型名后缀约定**: `embeddinggemma-2:Document`、`:SearchQuery`、`:STS`… 对 SDK 完全透明 |
| 不暴露 `encoding_format`/normalize 开关 | 想要 base64 或非归一化向量不行 | 服务端恒归一(点积=cos,通常正是想要的);base64 走裸 fetch |
| 每请求无法注入 prompt 文本 | — | 同上,后缀或裸调 `/embed` 的 `prompt` 字段 |

### 与自建服务的并发模型

SDK `embedMany` 一次请求批量发;我们服务端单飞锁串行(CPU/GPU 推理吞吐优先,单租户场景正确)。高并发客户端自行限制并发或加大 `batch_size`。

## 2. 什么时候不用 ai-sdk(替代方案)

| 需求 | 推荐 | 备注 |
| --- | --- | --- |
| RAG/Agent 管线,LLM+embedding 统一 | **ai + @ai-sdk/openai**(现状) | 体验最好,类型完整 |
| 协议级控制: base64、dimensions、usage 明细、token 数组输入 | **`openai` 官方 npm 包** | `OpenAI({baseURL})` 同样即插即用;我们全兼容 |
| **多模态**(图文音视 embedding/跨模态检索) | 裸 `fetch` / ky / ofetch → `POST /embed` | 生态内无多模态 embed 标准,这是现状而非缺陷 |
| 端侧/浏览器本地推理 | `@huggingface/transformers` | 740M 模型端侧偏重;embeddinggemma2 的 transformers.js 支持需查版本;服务端远程调用通常更优 |
| 向量入库 | pgvector / qdrant / lancedb | 与 embedding 层解耦,归一化向量直接点积 |

**结论:SDK 走 `/v1/embeddings`(文本),多模态走 `/embed`,两者共用同一模型实例与语义空间(同输入同向量,已验证 parity)。**

## 3. 多模态接入(原生 API)

```ts
const emb = await fetch("http://host:8000/embed", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: {
      text: "product photo  <",          // 占位符按顺序消耗右侧列表
      image: [{ b64: base64OrDataUri }, "/path/on/server.jpg", "https://..."],
      video: { b64: mp4Bytes },          // 或 path/url
      audio: { array: [...], sampling_rate: 44100 },  // PCM 直传,自动重采样 16k
    },
    prompt_name: "Document",             // 可选;truncate_dim 可选
  }),
}).then((r) => r.json());
```

- 图片: b64/dataURI/服务器路径/URL;音频: 任意容器(mp3/m4a/ogg/flac/wav...)或 PCM;视频: mp4/mov/webm/mkv...
- 8K 共享上下文,媒体 token 预算见模型卡(图 280/视频帧 140/音频 25/s)。

## 4. 踩坑记录(真实遇到)

1. **PCM 位宽**: 16-bit PCM 按 float32 解释 → 值域错乱。Int16 除以 32768;或直接传 `Float32Array`(服务端 `_audio_from_array` 两者都收)。
2. **SDK 吞错误详情**: 断言/日志用 `error.statusCode`,别 grep message。
3. **模型名后缀是我们的约定**: `:Document` 等;写错后缀 → 400 列出合法值。
4. **确定性**: 同输入两次编码 cos=1.0000(fp32);若自建 pipeline 发现抖动,先查是否混用 bf16(本机 ROCM 栈 bf16 批量注意力偶发 NaN,服务已默认 fp32)。
5. **node 跑 TS**: node ≥24 `--experimental-strip-types` 可直接跑这些 `.ts`;bun 更省事。

## 5. 验证入口

```bash
bun run test        # SDK 语义层 6 项
bun run test:mm     # 全模态语义 8 项(合成素材,含颜色/音高/跨模态断言)
bun run client      # 全接口×全选项×3SDK(ai/openai/dashscope形状)×一致性 41 项
bun run real        # 真实素材: 丢进 assets/ 或 bun run real /path/*(结构+确定性+自检索)
```
