#!/usr/bin/env bash
# Bootstrap one machine profile: venv + matching torch stack + locked deps + .env seed.
# Usage: scripts/setup_env.sh cpu|strix-halo      (or: just setup cpu)
set -euo pipefail
cd "$(dirname "$0")/.."
py=".venv/bin/python"

profile="${1:-}"
case "$profile" in
  cpu)
    torch_pkgs=("torch==2.14.1+cpu" "torchvision==0.29.1+cpu" "torchcodec==0.17.0+cpu")
    index_args=(--index-url https://download.pytorch.org/whl/cpu)
    ;;
  strix-halo)
    # ROCm/gfx1151 组合为本机验证过的版本;torchcodec 无 rocm 轮,用 cpu 版
    torch_pkgs=("torch==2.14.1+rocm7.2" "torchvision==0.29.1+rocm7.2" "torchcodec==0.17.0+cpu")
    index_args=(--index-url https://download.pytorch.org/whl/rocm7.2
                --extra-index-url https://download.pytorch.org/whl/cpu
                --index-strategy unsafe-best-match)
    ;;
  omlx)
    # 远程 oMLX 服务客户端:只装 requests(见 requirements-omlx.txt),
    # 不装 torch 栈;权重在服务端(coral),本机仅作 API 客户端/测试。
    torch_pkgs=()
    index_args=()
    ;;
  *)
    echo "usage: $0 cpu|strix-halo" >&2
    exit 2
    ;;
esac
prefix="$(echo "$profile" | tr 'a-z-' 'A-Z_')"

echo "== setup profile: ${profile} =="
uv venv --python 3.12 .venv
if [ ${#torch_pkgs[@]} -gt 0 ]; then
  uv pip install --python "$py" ${index_args+"${index_args[@]}"} "${torch_pkgs[@]}"
  grep -vE '^(#|torch==|torchvision==|torchcodec==)' requirements.txt | grep -v '^$' > /tmp/req-rest.$$.txt
  uv pip install --python "$py" -r /tmp/req-rest.$$.txt
  rm -f /tmp/req-rest.$$.txt
else
  uv pip install --python "$py" -r requirements-omlx.txt
fi

if [ ! -f .env ]; then
  printf 'PROFILE=%s\n# overrides: see .env.example (prefix %s*)\n' "$profile" "$prefix" > .env
  echo "wrote .env (PROFILE=${profile})"
elif ! grep -q '^PROFILE=' .env; then
  echo "PROFILE=${profile}" >> .env
  echo "appended PROFILE=${profile} to existing .env"
else
  echo ".env already pins PROFILE -- left untouched"
fi

echo "ok -- just serve 启动;just check 验证(首次素材: just assets)"
