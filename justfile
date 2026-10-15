# embeddinggemma2-serve — 常用操作
# 机器差异收敛到 profile：.env 的 PROFILE 选择 cpu|strix-halo（auto 探测），
# 覆盖项用对应前缀 CPU_*/STRIX_HALO_*（全表见 .env.example，由 config.py 生成）

set shell := ["bash", "-uc"]
py := ".venv/bin/python"
uvbin := "uv"
# 服务地址: EMBG2_BASE_URL > .env(profile 感知) > 探测;见 endpoint.py
resolve := f'{{py}} -c "from endpoint import resolve_base_url; print(resolve_base_url())"'

# 默认：列出所有命令
default:
    @just --list

# ---------------------------------------------------------------- 服务
# 前台启动服务（Ctrl-C 停止）；配置见 .env（cp .env.example .env）
serve:
    ./run.sh

# 开发模式（改代码自动重载）
dev:
    RELOAD=1 ./run.sh

# 健康检查 / 全量元信息(地址自动发现,失败给提示)
health:
    url=$({{resolve}}) && curl -sf "$url/health" && echo
meta:
    url=$({{resolve}}) && curl -s "$url/metadata" | {{py}} -m json.tool

# 单条文本冒烟:断言维度/归一化(可换 prompt 与维度)
try text:
    {{py}} scripts/try_embed.py {{quote(text)}}

# ---------------------------------------------------------------- �验证
up-guard:
    @url=$({{resolve}}) && curl -sf -m 2 "$url/health" >/dev/null || { echo "service unreachable — 先 just serve,或 EMBG2_BASE_URL=http://host:port 指定"; exit 1; }

# Python HTTP 全模态测试
test-api: (up-guard)
    {{py}} test_api.py

# TS：AI SDK 语义层（6 项）
test-sdk: (up-guard)
    cd sdk-test && bun run test

# TS：全接口 / 双 SDK / 三方一致性（34 项）
test-client: (up-guard)
    cd sdk-test && bun run client

# TS：多模态语义（合成素材 8 项，首次先 just assets）
test-mm: (up-guard)
    cd sdk-test && bun run test:mm

# TS：真实素材结构验证（传文件路径，或默认扫 sdk-test/assets）
test-real *paths: (up-guard)
    cd sdk-test && bun run real {{paths}}

# 一键回归：api + sdk + client + mm
check: test-api test-sdk test-client test-mm
    @echo "== all green"

# 跨模态检索 sanity（真实素材 assets/images，中文 query）
recall: (up-guard)
    {{py}} retrieval_check.py

# 性能基准（可加 --quick）
bench *args: (up-guard)
    cd sdk-test && bun run bench {{args}}

# 从 config.py 重新生成 .env.example（配置元信息的唯一来源）
env-example:
    {{py}} scripts/gen_env_example.py

# 生成合成测试素材
assets:
    cd sdk-test && ../{{py}} make_assets.py

# ---------------------------------------------------------------- 环境
# 按 profile 从零装环境（venv + 对应 torch 栈 + 锁定依赖 + 写入 .env 的 PROFILE）
setup profile:
    ./scripts/setup_env.sh {{profile}}

# 打印当前生效配置（profile / 模型 / 绑定地址 / device·dtype）
config:
    {{py}} -c 'from config import ACTIVE_PROFILE, settings; print("profile =", ACTIVE_PROFILE); print("model   =", settings.model_path); print("bind    = %s:%d" % (settings.host, settings.port)); print("device  = %s / %s" % (settings.device, settings.dtype))'

# 按 requirements 全量对齐已有 venv
sync:
    {{uvbin}} pip sync --python {{py}} requirements.txt

# ---------------------------------------------------------------- 仓库
clean:
    rm -rf __pycache__ sdk-test/__pycache__ server.log

push:
    git push

tag version:
    git tag {{version}}
    git push origin {{version}}
