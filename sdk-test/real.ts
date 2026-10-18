/**
 * Real-asset verification: encodes media through the native /embed API and
 * checks structure + determinism + retrieval SEMANTICS on the repo's real
 * assets/ set (zh images/audio/video), plus any extras in sdk-test/assets.
 *
 *   bun run real                      # scans ./assets AND ../assets (repo root)
 *   bun run real /path/a.jpg dir/ ... # explicit files; directories scan recursively
 */
import { readFile, readdir, stat } from "node:fs/promises";
import { basename, isAbsolute, join, resolve } from "node:path";
import { API_KEY, BASE, MODEL, REMOTE, modelId, preflight } from "./base";

await preflight();
const OWN = new URL("./assets", import.meta.url).pathname; // synthetic set (make_assets.py)
const ROOT_ASSETS = new URL("../assets", import.meta.url).pathname; // real media (git-tracked)

const SYNTH = new Set(["blue_bg.png", "blue_clip.mp4", "red_bg.png", "tone440.wav", "tone880.wav"]);

const KIND: Record<string, string> = {
  png: "image", jpg: "image", jpeg: "image", webp: "image", gif: "image", bmp: "image",
  tif: "image", tiff: "image", heic: "image", avif: "image",
  wav: "audio", mp3: "audio", m4a: "audio", aac: "audio", flac: "audio", ogg: "audio",
  oga: "audio", opus: "audio", weba: "audio", wma: "audio",
  mp4: "video", mov: "video", mkv: "video", webm: "video", avi: "video", m4v: "video",
  mpg: "video", mpeg: "video", ts: "video",
};

let failures = 0;
const check = (name: string, ok: boolean, info: string) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(34)} ${info}`);
  if (!ok) failures++;
};

const MIME: Record<string, string> = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", webp: "image/webp", wav: "audio/wav", mp3: "audio/mpeg", m4a: "audio/mp4", flac: "audio/flg" };
const dataUri = (raw: Buffer, ext: string): string => {
  const mime = MIME[ext] ?? (["wav","mp3","m4a","flac","ogg","opus","aac"].includes(ext) ? "audio/" + ext : "image/" + ext);
  return `data:${mime};base64,${raw.toString("base64")}`;
}

async function embedRemote(item: Record<string, unknown>, dims?: number): Promise<number[]> {
  // remote: {image|audio|text: local path} -> read bytes -> data URI item
  const body: Record<string, unknown> = { model: MODEL };
  const it: Record<string, unknown> = {};
  if (typeof item.text === "string") it.text = item.text;
  for (const k of ["image", "audio", "video"] as const) {
    const p = item[k];
    if (typeof p === "string") {
      const ext = (p.split(".").pop() ?? "").toLowerCase();
      it[k === "video" ? "image" : k] = dataUri(Buffer.from(await readFile(p)), ext); // video -> first-frame proxy not possible; see note
    }
  }
  if (!Object.keys(it).length) throw new Error("empty remote item");
  body.items = [it];
  if (dims) body.dimensions = dims;
  const res = await fetch(`${BASE}/v1/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(API_KEY && API_KEY !== "local" ? { Authorization: `Bearer ${API_KEY}` } : {}) },
    body: JSON.stringify(body),
  });
  const json: any = await res.json();
  if (!res.ok) throw new Error(`${res.status} ${JSON.stringify(json).slice(0, 160)}`);
  return json.data[0].embedding;
}

async function embed(input: unknown, extra: Record<string, unknown> = {}): Promise<number[]> {
  if (REMOTE) {
    if (typeof input === "string") {
      const res = await fetch(`${BASE}/v1/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...(API_KEY && API_KEY !== "local" ? { Authorization: `Bearer ${API_KEY}` } : {}) },
        body: JSON.stringify({ model: MODEL, input, ...(extra.truncate_dim ? { dimensions: extra.truncate_dim } : {}) }),
      });
      const json: any = await res.json();
      if (!res.ok) throw new Error(`${res.status} ${JSON.stringify(json).slice(0, 160)}`);
      return json.data[0].embedding;
    }
    const item = (Array.isArray(input) ? input[0] : input) as Record<string, unknown>;
    return embedRemote(item, extra.truncate_dim as number | undefined);
  }
  const res = await fetch(`${BASE}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input, ...extra }),
  });
  const json = await res.json();
  if (!res.ok) throw new Error(`${res.status} ${JSON.stringify(json).slice(0, 160)}`);
  return (json as { embeddings: number[][] }).embeddings[0];
}

function cosine(a: number[], b: number[]): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / Math.sqrt(na * nb);
}

/** expand CLI args: a file stays, a directory becomes its known-media files (recursive) */
async function expandPaths(args: string[]): Promise<string[]> {
  const out: string[] = [];
  for (const p of args) {
    const st = await stat(p).catch(() => null);
    if (st?.isDirectory()) {
      const entries = await readdir(p, { recursive: true });
      out.push(...entries.filter((f) => KIND[(f.split(".").pop() ?? "").toLowerCase()]).map((f) => join(p, f)));
    } else {
      out.push(p);
    }
  }
  return out;
}

const args = process.argv.slice(2).map((p) => (isAbsolute(p) ? p : resolve(p)));
let files: string[] = args.length ? await expandPaths(args) : [];
if (files.length === 0) {
  // default: repo-root real assets first, then synthetic extras from ./assets
  for (const dir of [ROOT_ASSETS, OWN]) {
    const entries = await readdir(dir, { recursive: true }).catch(() => []);
    files.push(...entries.filter((f) => !f.includes("node_modules")).map((f) => join(dir, f)));
  }
}
files = files.filter((f) => KIND[(basename(f).split(".").pop() ?? "").toLowerCase()]);
files = [...new Set(files)].sort((a, b) => a.localeCompare(b, "zh"));

if (files.length === 0) {
  console.log("no media found — pass paths, or populate ../assets and ./assets; rerun `bun run real`");
  process.exit(0);
}

const vecs = new Map<string, number[]>();

for (const path of files) {
  const name = basename(path);
  const ext = (name.split(".").pop() ?? "").toLowerCase();
  const kind = KIND[ext];
  const synthetic = SYNTH.has(name);

  const t0 = performance.now();
  if (REMOTE && kind === "video") {
    console.log(`SKIP  ${name}  (remote: no video field -- extract frames to images client-side)`);
    continue;
  }
  let v: number[], v2: number[];
  try {
    const input: unknown = { [kind]: path }; // server-local path
    v = await embed(input);
    v2 = await embed(input);
  } catch (e) {
    const msg = String(e);
    if (REMOTE && kind === "audio") {
      // flag is on server-side; 0.7.1rc1 audio inference 500s on every
      // encoding (documented in docs/OMMLX.md + recipes/omlx_backend)
      console.log(`SKIP  ${name}  (remote audio: 0.7.1rc1 server 500 -- see docs/OMMLX.md)`);
      continue;
    }
    check(name, false, msg.slice(0, 140));
    continue;
  }
  const ms = Math.round(performance.now() - t0);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const det = cosine(v, v2);
  const okStruct = v.length === 768 && v.every(Number.isFinite) && Math.abs(norm - 1) < 1e-3;

  vecs.set(path, v);
  const tag = synthetic ? " [synthetic]" : "";
  check(name + tag, okStruct && det > 0.999, `${v.length}d norm=${norm.toFixed(4)} det=${det.toFixed(4)} ${ms}ms`);
}

// ---- self-retrieval: every vector's top1 is itself (identity sanity)
if (vecs.size > 1) {
  let allSelf = true;
  const detail: string[] = [];
  for (const [p, vp] of vecs) {
    let best = "", bc = -1;
    for (const [q, vq] of vecs) {
      const c = cosine(vp, vq);
      if (c > bc) { bc = c; best = q; }
    }
    if (best !== p) { allSelf = false; detail.push(`${basename(p)} -> ${basename(best)} (${bc.toFixed(3)})`); }
  }
  check("self-retrieval argmax", allSelf, detail.length ? detail.join("; ") : `${vecs.size} items, each top1 = itself`);
}

// ---- real-image semantic retrieval: zh query -> expected image ranks top1
const REAL_QUERIES: [string, string][] = [
  // [query, image filename substring] — assets/images, all 384px photos with zh names
  ["一只猫", "猫在窗台上"],
  ["狗和小孩在公园玩球", "小狗和小孩玩球在公园"],
  ["窗户前一杯热咖啡旁边有书", "窗户前有一杯热咖啡和书"],
  ["窗台上一个猫形状的玩偶旁边有书", "窗户前有一个猫形状的玩偶和书"],
];
{
  const realImgs = files.filter((f) => f.startsWith(ROOT_ASSETS) && f.endsWith(".png"));
  if (realImgs.length >= 4) {
    const docVecs = new Map<string, number[]>();
    for (const p of realImgs) docVecs.set(p, vecs.get(p)!);
    let hits = 0;
    const detail: string[] = [];
    for (const [q, substr] of REAL_QUERIES) {
      const qv = REMOTE
        ? await embed(`task: search result | query: ${q}`)
        : await embed(q, { prompt_name: "SearchQuery" });
      const rank = [...docVecs.entries()]
        .map(([p, v], i) => ({ i, c: cosine(qv, v), p }))
        .sort((a, b) => b.c - a.c);
      const top = rank[0];
      const hit = basename(top.p).includes(substr);
      if (hit) hits++;
      else detail.push(`"${q}" -> ${basename(top.p)} (want ~${substr})`);
      console.log(`      top1 "${q}" = ${basename(top.p)} cos=${top.c.toFixed(3)}${hit ? " ✓" : " ✗"}`);
    }
    check("zh text->image retrieval top1", hits === REAL_QUERIES.length, `${hits}/${REAL_QUERIES.length} hit${detail.length ? "; " + detail.join("; ") : ""}`);
  }
}

const real = files.filter((f) => !SYNTH.has(basename(f))).length;
console.log(`\n${vecs.size} file(s) encoded (${real} real, ${vecs.size - real} synthetic), ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
