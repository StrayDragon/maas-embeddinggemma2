#!/usr/bin/env bash
# Probe the oMLX server's embedding API surface: which shapes / modalities /
# limits it accepts for the eg2 model, and what it errors on.
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && . ./.env && set +a

BASE="${OMLX_BASE_URL:-http://127.0.0.1:11534}"
AUTH="Authorization: Bearer ${OMLX_API_KEY:-}"
MODEL="${OMLX_MODEL:-embeddinggemma-2-bf16}"
CT="Content-Type: application/json"
T="${OMLX_TIMEOUT_S:-180}"

say() { printf '\n===== %s =====\n' "$*"; }
post() { # $1 label, $2 json body -- prints compact result
  printf -- '-- %s\n' "$1"
  curl -s -m "$T" -o /tmp/omlx_probe.out -w 'http=%{http_code} t=%{time_total}s\n' \
    -H "$AUTH" -H "$CT" -d "$2" "$BASE/v1/embeddings"
  head -c 300 /tmp/omlx_probe.out; echo
}

say "meta"
curl -s -m 10 -H "$AUTH" "$BASE/v1/models" | head -c 400; echo

post "plain text"            "{\"model\":\"$MODEL\",\"input\":\"hello\"}"
post "text array"            "{\"model\":\"$MODEL\",\"input\":[\"a\",\"b\"]}"
post "content parts text"    "{\"model\":\"$MODEL\",\"input\":[{\"type\":\"text\",\"text\":\"parts hello\"}]}"
post "items text"            "{\"model\":\"$MODEL\",\"items\":[{\"text\":\"items hello\"}]}"
post "task steer (prefix)"   "{\"model\":\"$MODEL\",\"input\":\"task: search result | query: northern lights\"}"
say "image via data URI (1x1 png)"
PNG_B64=$(python3 - <<'EOF'
import base64, io
from PIL import Image
b = io.BytesIO(); Image.new("RGB", (4, 4), (200, 30, 30)).save(b, "PNG")
print(base64.b64encode(b.getvalue()).decode())
EOF
)
post "items image"  "{\"model\":\"$MODEL\",\"items\":[{\"image\":\"data:image/png;base64,$PNG_B64\"}]}"
post "parts image_url" "{\"model\":\"$MODEL\",\"input\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$PNG_B64\"}}]}"

say "audio via data URI (0.5s 16k sine wav)"
WAV_B64=$(python3 - <<'EOF'
import base64, io, math, struct, wave
buf = io.BytesIO()
with wave.open(buf, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", int(8000*math.sin(2*math.pi*440*i/16000))) for i in range(8000)))
print(base64.b64encode(buf.getvalue()).decode())
EOF
)
post "items audio"  "{\"model\":\"$MODEL\",\"items\":[{\"audio\":\"data:audio/wav;base64,$WAV_B64\"}]}"
post "parts input_audio" "{\"model\":\"$MODEL\",\"input\":[{\"type\":\"input_audio\",\"input_audio\":{\"data\":\"$WAV_B64\",\"format\":\"wav\"}}]}"

say "audio setting toggles"
curl -s -m 10 -H "$AUTH" "$BASE/admin/models" | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: print('(no admin models json)'); sys.exit()
models=d if isinstance(d,list) else d.get('models',d.get('data',[]))
for m in models:
    if isinstance(m,dict) and 'embeddinggemma' in str(m.get('id',m.get('name',''))):
        print(json.dumps(m)[:600])
"

say "video? (items video key)"
post "items video key" "{\"model\":\"$MODEL\",\"items\":[{\"video\":\"x\"}]}" 2>/dev/null | head -3
