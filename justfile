# embeddinggemma2-serve — 常用操作
# 服务前台运行（Ctrl-C 停止）；默认 cuda/fp32，环境变量可覆盖：DEVICE DTYPE MODEL_PATH PORT HOST

set shell := ["bash", "-uc"]
py := ".venv/bin/python"
uvbin := "uv"
port := env_var_or_default("PORT", "50233")
host := env_var_or_default("HOST", "0.0.0.0")

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

# 健康检查 / 全量元信息
health host="127.0.0.1" port=port:
    curl -sf http://{{host}}:{{port}}/health && echo
meta host="127.0.0.1" port=port:
    curl -s http://{{host}}:{{port}}/metadata | {{py}} -m json.tool

# 单条文本冒烟：断言维度/归一化（可换 prompt 与维度）
try text host="127.0.0.1" port=port:
    {{py}} scripts/try_embed.py {{quote(text)}} --url "http://{{host}}:{{port}}/embed"

# ---------------------------------------------------------------- �验证
up-guard port=port:
    @curl -sf -m 2 "http://127.0.0.1:{{port}}/health" >/dev/null || (echo "service not running — 先 just serve"; exit 1)

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
# 从零重建 venv（ROCm/gfx1151 组合，详见 README）；纯 CPU 机器按 README 换 cpu 索引
setup:
    {{uvbin}} venv --python 3.12 .venv
    {{uvbin}} pip install --python {{py}} \
      --index-url https://download.pytorch.org/whl/rocm7.2 \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match \
      "torch==2.14.1+rocm7.2" "torchvision==0.29.1+rocm7.2" "torchcodec==0.17.0+cpu"
    grep -vE '^(#|torch==|torchvision==|torchcodec==)' requirements.txt | grep -v '^$' > /tmp/req-rest.txt
    {{uvbin}} pip install --python {{py}} -r /tmp/req-rest.txt
    cp -n .env.example .env || true
    @echo "ok — 编辑 .env 后 just serve；just check 验证"

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
