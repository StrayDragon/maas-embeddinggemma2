# API 参考

> 部署 / 配置 / profile 见根目录 [README](../README.md);本文只覆盖 HTTP 接口与验证。
> **前端 / TS 集成必读**:[`sdk-test/INTEGRATION.md`](../sdk-test/INTEGRATION.md) —— 三入口选型决策树、ai-sdk/openai/DashScope 三种形状的实测结论与坑。
> 一句话:文本 RAG 用 SDK 打 `/v1/embeddings`;多模态与跨模态用 fetch 打 `/embed/st`;阿里云迁移走 DashScope 兼容路径;三者同语义空间。

## POST /embed/st(sentence-transformers 风格, `/embed` 为别名)

```jsonc
{
  "input": "text"                                     // 单条文本
        | {"text":..., "image":..., "audio":..., "video":...}  // 多模态单条
        | [ 混合列表 ],                                // 批量, 字符串/对象混排
  "prompt_name": "SearchQuery",   // 可选; 全部预设见 GET /metadata
  "prompt": "自定义前缀",          // 可选, 与 prompt_name 互斥
  "truncate_dim": 512,            // MRL 截断: 768/512/256/128
  "normalize_embeddings": true,   // 默认 true, 截断后自动重归一化
  "batch_size": 32                // 可选
}
```

媒体字段取值(image / audio / video 通用):

| 写法 | 说明 |
| --- | --- |
| `"/path/a.jpg"` | 服务器本地路径 |
| `"https://..../a.jpg"` | URL, 服务端代抓 |
| `{"b64": "..."}` | base64 或 dataURI |
| `{"path":...}` / `{"url":...}` | 显式对象 |
| `{"array":[...],"sampling_rate":44100}` | 仅 audio: 原始 PCM, 自动转 16k 单声道 |

交错模态: 在 `text` 中放 ` IMAGE_TOKEN ` / ` VIDEO_TOKEN ` 占位符, 按顺序消耗 `image`/`video` 列表(共享 8192 token 上下文)。

响应: `{"embeddings": [[...]], "dimension", "count", "normalized", "took_ms"}` — 单位向量, 直接内积即余弦相似度。

## DashScope 兼容层(阿里云百炼多模态向量形状)

`POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`(别名 `/embed/dashscope`)

- `input.contents[]`: `{text|image|video|multi_images}` 或纯字符串; image 支持 URL/base64 dataURI/服务器路径
- 单 content 多模态 → 融合向量(`type=fused`);`parameters.enable_fusion=true` → 全部 contents 融合为 1 条(`type=fusion`);默认独立向量
- `parameters.dimension` → MRL 截断;`parameters.instruct` → 按 EmbeddingGemma 指令格式 `task:...|query:` 前缀
- 响应/错误/`request_id`/`usage` 均按 DashScope 形状;视觉 token 为估计值(280/图, 1120/视频)
- 所有向量与 `/embed/st` 同一语义空间, 可直接混用于跨模态检索

```bash
curl -s localhost:50233/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding \
  -H 'Content-Type: application/json' \
  -d '{"model":"embeddinggemma-2","input":{"contents":[{"text":"一只猫"},{"image":"data:image/png;base64,..."}]},"parameters":{"dimension":256}}'
```

## GET /health · GET /metadata

健康状态(含生效 profile);prompt 预设 / 支持维度 / 模态清单。

## OpenAI 兼容层(TypeScript AI SDK 生态直连)

- `POST /v1/embeddings`(别名 `/embeddings`)+ `GET /v1/models` + 全量 CORS —— `ai` / `@ai-sdk/openai` / `openai` npm 包 / langchain 零适配直连。
- 模型名可带任务后缀选择 prompt 前缀:`embeddinggemma-2:Document`、`:SearchQuery`、`:STS`、`:Clustering`……(列表见 `/v1/models` 的 `task_suffixes`)
- `dimensions` → MRL 截断;`encoding_format: base64` → float32 LE;usage 返回真实 token 数。

```ts
import { createOpenAI } from "@ai-sdk/openai";
import { embed, embedMany } from "ai";

const openai = createOpenAI({ baseURL: "http://localhost:50233/v1", apiKey: "local" });

const { embedding } = await embed({
  model: openai.textEmbeddingModel("embeddinggemma-2"),
  value: "hello",
});

const { embeddings } = await embedMany({
  model: openai.textEmbeddingModel("embeddinggemma-2:Document"),   // 任务前缀
  values: ["doc one", "doc two"],
});

await embed({
  model: openai.textEmbeddingModel("embeddinggemma-2"),
  value: "hello",
  providerOptions: { openai: { dimensions: 256 } },   // MRL 截断
});
```

验证(sdk-test/, bun 或 node 24+):

```bash
cd sdk-test && bun install
../.venv/bin/python make_assets.py  # 一次性生成测试媒体
bun run test        # AI SDK 语义层: embed/embedMany/dimensions/task后缀/错误路径  (6 项)
bun run test:mm     # 全模态端到端: 图/音/视频/交错/混排批次/base64 parity + 语义断言 (8 项)
```

多模态语义断言包括: 蓝底图↔"blue background"文本互检索优于红底;440Hz↔880Hz 可分;蓝色视频↔蓝色图对齐;交错向量介于文本与图像之间且方向正确;OpenAI base64 与 float 编码 parity。

## 示例

```bash
# 文本, 256 维
curl -s localhost:50233/embed -H 'Content-Type: application/json' \
  -d '{"input":"hello","prompt_name":"SearchQuery","truncate_dim":256}'

# 图文混合
curl -s localhost:50233/embed -H 'Content-Type: application/json' \
  -d '{"input":{"text":"a red mug  IMAGE_TOKEN ","image":{"b64":"<base64>"}}}'

# 批量: 文本+多模态混排
curl -s localhost:50233/embed -H 'Content-Type: application/json' \
  -d '{"input":["plain text", {"text":"doc", "image":["/tmp/a.jpg"]}], "prompt_name":"Document"}'
```

Python:

```python
import requests
r = requests.post("http://localhost:50233/embed", json={
    "input": [{"text": "q", "image": {"path": "/tmp/a.jpg"}}],
    "prompt_name": "SearchQuery",
})
vec = r.json()["embeddings"][0]
```

## 测试

```bash
.venv/bin/python test_api.py    # HTTP 全模态 + 批量 + 截断 + 错误路径
cd sdk-test && bun run client    # TS 全接口/双SDK(openai+ai)/一致性 34 项
```

## 备注

- bf16 在 gfx1151 + torch 2.14.1+rocm7.2 下批量注意力偶发 NaN(单条正常), 服务默认 fp32(~3GB 显存, 批量 12 条 ~45ms)。
- `attn_implementation` 已固定 `sdpa`。
- 音频解码: torchcodec(cpu) / 录制 PCM 直传; 视频解码: torchcodec; 图片: PIL。
