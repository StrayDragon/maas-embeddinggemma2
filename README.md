# embeddinggemma2-serve

EmbeddingGemma 2 (740M) 多模态 embedding 服务:文本 / 图片 / 音频 / 视频 / 交错输入 → 统一 768 维向量(MRL 768/512/256/128)。无状态单进程,机器差异全部收敛到 profile。API 详情见 [docs/API.md](docs/API.md)。

除自建推理外,也支持 **oMLX 远程后端**(`ommlx` profile):Mac 上 oMLX 直接 serve 本模型,本仓库退化为其客户端/测试台,详见 [docs/OMMLX.md](docs/OMMLX.md)。

## 快速开始

```bash
just setup strix-halo   # 或 just setup cpu:建 venv + 装对应 torch 栈 + 写 .env(PROFILE=<name>)
just setup ommlx        # 或:远端 oMLX 后端,轻量 venv(无 torch),配置 OMLX_BASE_URL/OMLX_API_KEY 即用
just serve              # 另: just dev 热重载;just config 查看生效配置(ommlx 下打印远端地址)
just check              # 一键回归(首次素材: just assets);ommlx 下自动切远端断言
just test-ommlx         # 仅 ommlx:22 项远端活体断言
```

## API 一览

| 入口 | 路径 | 适用场景 |
| --- | --- | --- |
| REST 全模态 | `POST /embed/st`(别名 `/embed`) | 多模态 / 跨模态 / 交错输入,完整参数(prompt 预设、MRL 截断、批量) |
| OpenAI 兼容 | `POST /v1/embeddings` · `GET /v1/models` | TypeScript `ai` / `@ai-sdk/openai` / `openai` / langchain 零适配直连 |
| DashScope 兼容 | `POST /api/v1/services/embeddings/multimodal-embedding/multimodal-embedding` | 阿里云百炼迁移,请求/响应形状 1:1 |
| 运维 | `GET /health` · `GET /metadata` | 探活(含生效 profile)/ prompt 预设 / 支持维度与模态 |

字段与示例:[docs/API.md](docs/API.md)。三个入口同一语义空间,向量可直接混用。
**TypeScript 技术栈集成实测与选型决策树:[sdk-test/INTEGRATION.md](sdk-test/INTEGRATION.md)**(ai-sdk / openai / DashScope 三种形状的结论与坑,配套 `sdk-test/*.ts` 可跑断言)。

客户端地址发现:测试与辅助脚本自动从 `.env`(profile 感知)解析服务地址,`EMBG2_BASE_URL=http://host:port` 可覆盖;均不可达时报错并给出启动提示(见 `endpoint.py` / `sdk-test/base.ts`)。

## 配置

- 唯一真源 [`config.py`](config.py)(pydantic-settings);`.env.example` 由它生成、含**全部 profile 的全部变量**:`just env-example`。
- `.env` 中 `PROFILE=auto | cpu | strix-halo | ommlx` 选择 profile(`auto` 启动时探测:有 GPU → strix-halo,否则 cpu;建议写死以跳过探测)。
- 覆盖项必须带 profile 前缀,一台机器一个块、互不干扰:`CPU_DEVICE=...`、`STRIX_HALO_PORT=...`。真实环境变量 > `.env` > profile 默认。
- 运行期 device/dtype 细节(auto 回退、ROCm bf16 批量 NaN → fp32)见 `server.py` 的 `_resolve_device/_resolve_dtype`。

## Profile

| profile | 目标机器 | env 前缀 | torch 栈 |
| --- | --- | --- | --- |
| `cpu` | 纯 CPU | `CPU_` | cpu wheels |
| `strix-halo` | AMD Radeon 8060S(gfx1151, ROCm 7.2) | `STRIX_HALO_` | rocm7.2 wheels,fp32 服务默认 |
| `ommlx` | 远程/本机 [oMLX](https://github.com/jundot/omlx) 已 serve 本模型 | `OMLX_` | 无(纯 HTTP 客户端,`requirements-ommlx.txt`) |

新增机器三步:`config.py` 加一个 `XxxSettings` 子类(定前缀与默认值)→ `scripts/setup_env.sh` 加一个 case(装机配方)→ `just env-example`。

## 性能

见 [BENCHMARK.md](BENCHMARK.md)(strix-halo 实测;CPU fallback 数据在同文件)。
