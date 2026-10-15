/**
 * Real-asset verification: encodes media files through the native /embed API,
 * checks structure + determinism + cross-file self-retrieval.
 *
 *   bun run real                      # scans ./assets (skips the synthetic set)
 *   bun run real /path/a.jpg ...      # or explicit paths anywhere
 */
import { readFile, readdir, stat } from "node:fs/promises";
import { basename, isAbsolute, join, resolve } from "node:path";
import { BASE, preflight } from "./base";

await preflight();
const ASSETS = new URL("./assets", import.meta.url).pathname;
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
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(30)} ${info}`);
  if (!ok) failures++;
};

async function embed(input: unknown): Promise<number[]> {
  const res = await fetch(`${BASE}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ input }),
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

const args = process.argv.slice(2).map((p) => (isAbsolute(p) ? p : resolve(p)));
let files: string[] = args;
if (files.length === 0) {
  const entries = await readdir(ASSETS, { recursive: true });
  files = entries
    .filter((f) => !f.includes("node_modules") && !SYNTH.has(basename(f)))
    .map((f) => join(ASSETS, f));
}
files = files.filter((f) => basename(f) !== "real.ts" && basename(f) !== "test.ts" && basename(f) !== "multimodal.ts");

if (files.length === 0) {
  console.log("no assets found — drop files into sdk-test/assets/ or pass paths; rerun `bun run real`");
  process.exit(0);
}

const vecs = new Map<string, number[]>();

for (const path of files) {
  const name = basename(path);
  const ext = (name.split(".").pop() ?? "").toLowerCase();

  const t0 = performance.now();
  let input: unknown;
  let v: number[], v2: number[];
  try {
    if (ext === "txt" || ext === "md") {
      input = (await readFile(path, "utf8")).slice(0, 6000);
    } else {
      const kind = KIND[ext];
      if (!kind) {
        console.log(`SKIP  ${name} (unknown ext .${ext})`);
        continue;
      }
      await stat(path); // exists?
      input = { [kind]: path }; // server-local path
    }
    v = await embed(input);
    v2 = await embed(input);
  } catch (e) {
    check(name, false, String(e).slice(0, 140));
    continue;
  }
  const ms = Math.round(performance.now() - t0);
  const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const det = cosine(v, v2);
  const okStruct = v.length === 768 && v.every(Number.isFinite) && Math.abs(norm - 1) < 1e-3;

  vecs.set(path, v);
  check(name, okStruct && det > 0.999, `${v.length}d norm=${norm.toFixed(4)} det=${det.toFixed(4)} ${ms}ms`);
}

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

console.log(`\n${vecs.size} file(s) encoded, ${failures} failure(s)`);
process.exit(failures ? 1 : 0);
