/**
 * HTTP-level benchmark. Usage: bun run bench [--quick]
 * Measures end-to-end latency (network + tokenize + GPU) against the running server.
 */
const BASE = "http://127.0.0.1:8000";
const A = new URL("./assets/", import.meta.url).pathname;
const QUICK = process.argv.includes("--quick");

async function embed(body: object): Promise<{ ms: number; dim: number; n: number }> {
  const t0 = performance.now();
  const res = await fetch(`${BASE}/embed`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 200)}`);
  const j = (await res.json()) as { embeddings: number[][]; dimension: number };
  return { ms: performance.now() - t0, dim: j.dimension, n: j.embeddings.length };
}

const pct = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b);
  const q = (p: number) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { p50: q(0.5), p95: q(0.95), mean: s.reduce((a, b) => a + b, 0) / s.length };
};

const b64 = async (f: string) => Buffer.from(await Bun.file(A + f).arrayBuffer()).toString("base64");

type Row = { name: string; n: number; p50: number; p95: number; mean: number; tput: string };
const rows: Row[] = [];

async function scenario(name: string, times: number, make: () => object) {
  for (let i = 0; i < 3; i++) await embed(make()); // warm
  const xs: number[] = [];
  let items = 0;
  for (let i = 0; i < times; i++) {
    const r = await embed(make());
    xs.push(r.ms);
    items += r.n;
  }
  const p = pct(xs);
  rows.push({ name, n: times, ...p, tput: `${((items / (p.mean * times)) * 1000 * times).toFixed(0)}/s` });
  console.log(`${name.padEnd(34)} p50=${p.p50.toFixed(0).padStart(5)}ms p95=${p.p95.toFixed(0).padStart(5)}ms mean=${p.mean.toFixed(0).padStart(5)}ms`);
}

// ---- assets
const png = await b64("blue_bg.png");
const wav16 = await Bun.file(A + "long16s.wav").arrayBuffer().then((b) => Buffer.from(b).toString("base64"));
const mp4 = await b64("blue_clip.mp4"); // 2 s
const imgItem = { image: { b64: png } };
const audItem = { audio: { b64: wav16 } };
const vidItem = { video: { b64: mp4 } };
const mixItem = { text: "product photo  <", ...imgItem, ...vidItem };

console.log(`benchmark ${QUICK ? "(quick)" : ""} against ${BASE}\n`);
const R = QUICK ? 0.4 : 1;

// text scaling
await scenario("text   x1   @768d", Math.round(50 * R), () => ({ input: "The northern lights are caused by charged particles." }));
await scenario("text   x8   @768d", Math.round(25 * R), () => ({ input: Array.from({ length: 8 }, (_, i) => `sample number ${i} text`) }));
await scenario("text   x32  @768d", Math.round(12 * R), () => ({ input: Array.from({ length: 32 }, (_, i) => `sample number ${i} text`) }));
await scenario("text   x64  @768d", Math.round(8 * R), () => ({ input: Array.from({ length: 64 }, (_, i) => `sample number ${i} text`) }));
await scenario("text   x32  @256d", Math.round(12 * R), () => ({ input: Array.from({ length: 32 }, (_, i) => `s ${i}`), truncate_dim: 256 }));
await scenario("text   x32  prompt=Document", Math.round(12 * R), () => ({ input: Array.from({ length: 32 }, (_, i) => `doc ${i}`), prompt_name: "Document" }));

// multimodal
await scenario("image  x1        (224px png)", Math.round(20 * R), () => ({ input: imgItem }));
await scenario("image  x8", Math.round(6 * R), () => ({ input: Array.from({ length: 8 }, () => imgItem) }));
await scenario("audio  x1        (16 s wav)", Math.round(20 * R), () => ({ input: audItem }));
await scenario("audio  x8        (16 s wav)", Math.round(5 * R), () => ({ input: Array.from({ length: 8 }, () => audItem) }));
await scenario("video  x1        (2 s clip)", Math.round(15 * R), () => ({ input: vidItem }));
await scenario("video  x4        (2 s clip)", Math.round(5 * R), () => ({ input: Array.from({ length: 4 }, () => vidItem) }));
await scenario("mixed  x4  text+img+video", Math.round(6 * R), () => ({ input: Array.from({ length: 4 }, () => mixItem) }));

// concurrency: 8 parallel clients x R*10 rounds of single text
{
  const clients = 8, rounds = Math.round(10 * (QUICK ? 0.5 : 1));
  for (let i = 0; i < 3; i++) await embed({ input: "warm" });
  const t0 = performance.now();
  let total = 0;
  await Promise.all(
    Array.from({ length: clients }, async () => {
      for (let r = 0; r < rounds; r++) { await embed({ input: `concurrent ${r}` }); total++; }
    }),
  );
  const wall = performance.now() - t0;
  console.log(`\nconcurrency 8 clients x${rounds}: wall=${wall.toFixed(0)}ms eff=${((total / wall) * 1000).toFixed(0)} req/s (serialized by server lock)`);
}

console.log("\ndone");
