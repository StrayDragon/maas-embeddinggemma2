# TS / 前端集成指南 —— embeddinggemma2-serve

> 实测版本: ai@7.0.138 · @ai-sdk/openai@4.0.92 · openai@7.32 · bun 1.4 / node 26
> 服务三个入口,同一模型实例、**同一语义空间**(交叉验证 parity cos≥0.999999,见 `client.ts` F 段):

| 入口 | 路由 | 形状 | 适用 |
| --- | --- | --- | --- |
| OpenAI 兼容 | `POST /v1/embeddings` | OpenAI embeddings 协议 | 文本 RAG / 主流 SDK |
| 原生全模态 | `POST /embed/st`(`POST /embed` 别名) | 自定义 JSON,图/音/视/交错 | 多模态与跨模态检索 |
| DashScope 兼容 | `POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`(别名 `/embed/dashscope`) | 阿里云百炼形状 | 阿里云系迁移、contents 风格 |

## 0. 选型决策树（先看这个）

```text
你的输入是纯文本？
├─ 是 → 用 SDK
│   ├─ 要 LLM+embedding 统一管线(RAG/Agent)      → ai-sdk   POST /v1/embeddings
│   ├─ 要协议级控制(base64/usage/token 数组)      → openai包 POST /v1 embeddings
│   └─ 在任务前缀质量敏感(检索/聚类不对称)        → 上述任一 + 模型名后缀 :Document/:SearchQuery/:STS
└─ 否(图/音/视频/交错/跨模态)
    ├� 从阿里云百炼迁移、习惯 contents 数组        → POST /api/v1/.../multimodal-embedding(形状一致)
    └─ 其它                                      → fetch POST /embed/st(能力最全: audio PCM/占位符交错/prompt_name)
```

规则细则：

1. **文本走 SDK、多模态走 fetch** —— OpenAI embeddings 协议与 ai-sdk 的 `value: string` 都没有多模态，这是生态现状。
2. **任务前缀三种给法**（效果等同，已验证）：SDK 里用模型名后缀 `embeddinggemma-2:Document`；`/embed/st` 用 `prompt_name`/`prompt` 字段；DashScope 层用 `parameters.instruct`。**同一检索系统里 query 与 document 必须用各自的前缀且前后一致**（如 query=`SearchQuery`、doc=`Document`）。
3. **不要混用维度**：query 与库内向量 `truncate_dim`/`dimensions` 必须一致（128/256/512/768）。
4. **向量恒归一**：点积即余弦；入库前无需再归一化。
5. **浏览器直连**：CORS 全开(`*`)，可直接从 Next/Nuxt 客户端调；生产建议经自己后端转发以便控密与限流。

## 1. AI SDK(文本 RAG 首选)

```ts
import { createOpenAI } from "@ai-sdk/openai";
import { embed, embedMany } from "ai";

const openai = createOpenAI({ baseURL: "http://host:50233/v1", apiKey: "local" });

await embed({ model: openai.textEmbeddingModel("embeddinggemma-2"), value: "hello" });
await embedMany({
  model: openai.textEmbeddingModel("embeddinggemma-2:Document"), // 任务前缀
  values: ["doc one", "doc two"],
});
await embed({
  model: openai.textEmbeddingModel("embeddinggemma-2"),
  value: "q",
  providerOptions: { openai: { dimensions: 256 } }, // MRL 截断
});
```

实测结论：

- `baseURL` 覆盖零适配;`embedMany`/`usage.tokens`(真实 tokenizer 计数)正常。
- 错误是 `APICallError`,`statusCode` 可靠;但 `message` 只有 "Bad Request"——**detail 不透传**,排错看 `error.responseBody` 或服务端日志。
- 局限:无多模态、无 prompt 注入、无 encoding_format 选择(服务恒归一;base64 需裸 fetch)。
- 并发:SDK 批量一次发;服务端单飞锁串行,客户端自行限并发。

## 2. 官方 `openai` 包(协议级控制)

```ts
import OpenAI from "openai";
const openai = new OpenAI({ baseURL: "http://host:50233/v1", apiKey: "local", maxRetries: 1 });

const r = await openai.embeddings.create({ model: "embeddinggemma-2:Document", input: ["a", "b"] });
const b64 = await openai.embeddings.create({ model: "embeddinggemma-2", input: "x", encoding_format: "base64" });
// base64 = float32 little-endian; new Float32Array(Buffer.from(b64.data[0].embedding as string, "base64").buffer)
```

## 3. 多模态/跨模态(`POST /embed/st`)

```ts
const emb = await fetch("http://host:50233/embed/st", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    input: {
      text: "product photo  <",          // 占位符按顺序消耗右侧列表
      image: [{ b64: dataUri }, "/path/on/server.jpg", "https://..."],
      video: { b64: mp4Bytes },          // 或 path/url
      audio: { array: [...], sampling_rate: 44100 },  // PCM 直传,自动重采样 16k
    },
    prompt_name: "Document",             // 或 prompt: "自定义前缀"
    truncate_dim: 512,
  }),
}).then((r) => r.json());
// => { embeddings: [[...768]], dimension, count, normalized, took_ms }
```

- 图片: b64/dataURI/服务器路径/URL;音频: 任意容器或 PCM;视频: mp4/mov/webm/mkv...
- 8K 共享上下文;媒体 token 预算: 图 280 / 视频帧 140 / 音频 25/s。
- 典型跨模态检索: 库端 `{"image":...}`+`Document`,查询端文本+`SearchQuery`,余弦点积排序。

## 4. DashScope 形状(阿里云迁移零改造)

```ts
// 只改 host;路径与百炼一致 → 现有 dashscope 封装/fetch 层直接换 base
const DS = "http://host:50233/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding";
await fetch(DS, {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: "Bearer anything" },
  body: JSON.stringify({
    model: "embeddinggemma-2",
    input: { contents: [{ text: "一只猫" }, { image: dataUri }, { multi_images: [url1, url2] }] },
    parameters: { dimension: 256, enable_fusion: false, instruct: "find the pet" },
  }),
});
// output.embeddings[{index,embedding,type}]; 单 content 多模态=>type=fused; enable_fusion=>全部融合 type=fusion
```

与本服务的映射差异（文档级约定）：

- `enable_fusion=true`：全部 contents 融合为 1 条 `type=fusion`（对应 qwen3 语义）
- 单 content 内多模态（如 `{text,image}`）：融合为 1 条 `type=fused`（对应 2026-03-06 语义）
- `multi_images` 独立模式下每图一条向量；`instruct` 映射为 EmbeddingGemma 指令前缀；视觉 token 为估计值

## 5. 踩坑记录（真实遇到）

1. **PCM 位宽**: 16-bit PCM 按 float32 解释 → 值域错乱。`Int16Array` 除以 32768,或直接传 `Float32Array`。
2. **SDK 吞错误详情**: 断言/日志用 `error.statusCode`,别 grep message。
3. **模型名后缀是约定**: `:Document` 等;写错 → 400 并列出合法值。
4. **确定性**: 服务启动时已对各模态 warmup,同输入重复编码 cos=1.0;若自建 pipeline 抖动,先查是否混用 bf16(本机 ROCM 栈 bf16 批量注意力偶发 NaN,服务默认 fp32)。
5. **node 跑 TS**: node ≥24 `--experimental-strip-types`;bun 更省事。
6. **`(null): No such file...`**: ROCm 用户态初始化噪音,无害,忽略。

## 6. 验证入口

```bash
bun run test        # AI SDK 语义层 6 项
bun run test:mm     # 全模态语义 8 项(合成素材,颜色/音高/跨模态断言)
bun run client      # 全接口×全选项×3 SDK 形状×三方一致性 41 项
bun run real        # 真实素材: 丢 assets/ 或 bun run real /path/*(结构+确定性+自检索)
bun run bench       # 性能基线(见根目录 BENCHMARK.md)
```
