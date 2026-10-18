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

# 单条文本冒烟:断言维度/归一化(经 recipes 后端;可换 task 与维度)
try text:
    {{py}} scripts/try_embed.py {{quote(text)}}

# 同上,但指定任务前缀与维度
try-task text task dim="256":
    {{py}} scripts/try_embed.py {{quote(text)}} --task {{task}} --dim {{dim}}

# ---------------------------------------------------------------- �验证
# ---------------------------------------------------------------- oMLX (remote engine)
# 远程 oMLX 后端(如 coral:11534,见 docs/OMMLX.md):.env PROFILE=omlx + OMLX_* 配置
omlx-config:
    {{py}} -c 'from config import settings; print("profile  = omlx"); print("base_url =", settings.resolved_base_url()); print("model    =", settings.model_path); print("api_key  =", (settings.api_key[:6] + "...") if settings.api_key else "(unset)")'

# oMLX 端点能力矩阵实测(不改任何状态,只读+推理)
probe-omlx:
    ./scripts/probe_omlx.sh

# oMLX 活体测试(19 项;服务器不可达时给出配置提示)
test-omlx: (up-guard-omlx)
    {{py}} test_omlx.py

up-guard-omlx:
    @{{py}} -c "import sys; sys.exit(0 if __import__('omlx_client').models() else 1)" 2>/dev/null || { echo 'oMLX 不可达 -- 检查 .env OMLX_BASE_URL / OMLX_API_KEY(见 .env.example omlx 块)'; exit 1; }

up-guard:
    @url=$({{resolve}}) && curl -sf -m 2 "$url/health" >/dev/null || { echo "service unreachable — 先 just serve,或 EMBG2_BASE_URL=http://host:port 指定"; exit 1; }

# Python HTTP 全模态测试
test-api: (up-guard)
    {{py}} test_api.py

# TS：全接口 / 双 SDK / 三方一致性（34 项）；地址与模型 id 由 endpoint.py/profile 决定，
# 远端 oMLX 时自动带 OMLX_API_KEY 与 EMBG2_MODEL（见 sdk-test/base.ts）
test-client: (up-guard)
    #!/usr/bin/env bash
    set -euo pipefail
    url=$( {{py}} -c "from endpoint import resolve_base_url; print(resolve_base_url())" )
    export EMBG2_BASE_URL="$url"
    if {{py}} -c "import sys; from config import ACTIVE_PROFILE; sys.exit(0 if ACTIVE_PROFILE=='omlx' else 1)" 2>/dev/null; then
      {{py}} -c 'from config import settings; print("export EMBG2_MODEL=%s" % settings.model_path)' > /tmp/embg2_env
      source /tmp/embg2_env
    fi
    cd sdk-test && ../scripts/with_target_env.sh bun run client

# TS：AI SDK 语义层（6 项）-- 同样的远端感知
test-sdk: (up-guard)
    #!/usr/bin/env bash
    set -euo pipefail
    url=$( {{py}} -c "from endpoint import resolve_base_url; print(resolve_base_url())" )
    export EMBG2_BASE_URL="$url"
    if {{py}} -c "import sys; from config import ACTIVE_PROFILE; sys.exit(0 if ACTIVE_PROFILE=='omlx' else 1)" 2>/dev/null; then
      {{py}} -c 'from config import settings; print("export EMBG2_MODEL=%s" % settings.model_path)' > /tmp/embg2_env
      source /tmp/embg2_env
    fi
    cd sdk-test && ../scripts/with_target_env.sh bun run test

# TS：多模态语义（合成素材 8 项，首次先 just assets）-- 远端感知同上
test-mm: (up-guard)
    #!/usr/bin/env bash
    set -euo pipefail
    url=$( {{py}} -c "from endpoint import resolve_base_url; print(resolve_base_url())" )
    export EMBG2_BASE_URL="$url"
    if {{py}} -c "import sys; from config import ACTIVE_PROFILE; sys.exit(0 if ACTIVE_PROFILE=='omlx' else 1)" 2>/dev/null; then
      {{py}} -c 'from config import settings; print("export EMBG2_MODEL=%s" % settings.model_path)' > /tmp/embg2_env
      source /tmp/embg2_env
    fi
    cd sdk-test && ../scripts/with_target_env.sh bun run test:mm

# TS：真实素材结构+语义检索（默认扫 ../assets 真实素材 + ./assets 合成素材）-- 远端感知同上
test-real *paths: (up-guard)
    #!/usr/bin/env bash
    set -euo pipefail
    url=$( {{py}} -c "from endpoint import resolve_base_url; print(resolve_base_url())" )
    export EMBG2_BASE_URL="$url"
    if {{py}} -c "import sys; from config import ACTIVE_PROFILE; sys.exit(0 if ACTIVE_PROFILE=='omlx' else 1)" 2>/dev/null; then
      {{py}} -c 'from config import settings; print("export EMBG2_MODEL=%s" % settings.model_path)' > /tmp/embg2_env
      source /tmp/embg2_env
    fi
    cd sdk-test && ../scripts/with_target_env.sh bun run real {{paths}}

# 一键回归：api + sdk + client + mm + real
check: test-api test-sdk test-client test-mm test-real
    @echo "== all green"

# 跨模态检索 sanity（真实素材 assets/images，中文 query）
recall: (up-guard)
    {{py}} retrieval_check.py

# 性能基准（可加 --quick）-- 远端感知同上
bench *args: (up-guard)
    #!/usr/bin/env bash
    set -euo pipefail
    url=$( {{py}} -c "from endpoint import resolve_base_url; print(resolve_base_url())" )
    export EMBG2_BASE_URL="$url"
    if {{py}} -c "import sys; from config import ACTIVE_PROFILE; sys.exit(0 if ACTIVE_PROFILE=='omlx' else 1)" 2>/dev/null; then
      {{py}} -c 'from config import settings; print("export EMBG2_MODEL=%s" % settings.model_path)' > /tmp/embg2_env
      source /tmp/embg2_env
    fi
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

dbg-env:
    #!/usr/bin/env bash
    echo "model from settings: $(.venv/bin/python -c 'from config import settings; print(settings.model_path)')"
    {{py}} -c 'from config import settings; print("export EMBG2_MODEL=%s" % settings.model_path)' > /tmp/embg2_env
    source /tmp/embg2_env
    echo "env now: [$EMBG2_MODEL]"

# 双后端 A/B 基准(strix-halo vs omlx,同请求 x3 取中位)
bench-ab:
    bash scripts/ab_bench.py
