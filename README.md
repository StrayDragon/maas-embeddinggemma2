# embeddinggemma2-serve

EmbeddingGemma 2 (740M) 多模态 embedding 服务:文本 / 图片 / 音频 / 视频 / 交错输入 → 统一 768 维向量(MRL 768/512/256/128)。无状态单进程,机器差异全部收敛到 profile。API 详情见 [docs/API.md](docs/API.md)。

## 快速开始

```bash
just setup strix-halo   # 或 just setup cpu:建 venv + 装对应 torch 栈 + 写 .env(PROFILE=<name>)
just serve              # 另: just dev 热重载;just config 查看生效配置
just check              # 一键回归(首次素材: just assets)
```

## 配置

- 唯一真源 [`config.py`](config.py)(pydantic-settings);`.env.example` 由它生成、含**全部 profile 的全部变量**:`just env-example`。
- `.env` 中 `PROFILE=auto | cpu | strix-halo` 选择 profile(`auto` 启动时探测:有 GPU → strix-halo,否则 cpu;建议写死以跳过探测)。
- 覆盖项必须带 profile 前缀,一台机器一个块、互不干扰:`CPU_DEVICE=...`、`STRIX_HALO_PORT=...`。真实环境变量 > `.env` > profile 默认。
- 运行期 device/dtype 细节(auto 回退、ROCm bf16 批量 NaN → fp32)见 `server.py` 的 `_resolve_device/_resolve_dtype`。

## Profile

| profile | 目标机器 | env 前缀 | torch 栈 |
| --- | --- | --- | --- |
| `cpu` | 纯 CPU | `CPU_` | cpu wheels |
| `strix-halo` | AMD Radeon 8060S(gfx1151, ROCm 7.2) | `STRIX_HALO_` | rocm7.2 wheels,fp32 服务默认 |

新增机器三步:`config.py` 加一个 `XxxSettings` 子类(定前缀与默认值)→ `scripts/setup_env.sh` 加一个 case(装机配方)→ `just env-example`。

## 性能

见 [BENCHMARK.md](BENCHMARK.md)(strix-halo 实测;CPU fallback 数据在同文件)。
