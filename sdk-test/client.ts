/**
 * Full-surface client check: every endpoint, every option, both SDKs, cross-consistency.
 *   bun run client
 */
import OpenAI from "openai";
import { createOpenAI } from "@ai-sdk/openai";
import { embed as aiEmbed, embedMany as aiEmbedMany } from "ai";

const BASE = "http://127.0.0.1:50233";
const V1 = `${BASE}/v1`;

let failures = 0;
const total = { n: 0 };
function check(name: string, ok: boolean, info = "") {
  total.n++;
  console.log(`${ok ? "PASS" : "FAIL"}  ${name.padEnd(40)} ${info}`);
  if (!ok) failures++;
}

const cosine = (a: number[], b: number[]) => {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return d / Math.sqrt(na * nb);
};

async function jfetch(path: string, body?: object): Promise<{ status: number; json: unknown }> {
  const res = await fetch(`${BASE}${path}`, {
    method: body ? "POST" : "GET",
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  let json: unknown = null;
  try { json = await res.json(); } catch { /* empty */ }
  return { status: res.status, json };
}
const emb = async (input: unknown, extra: object = {}): Promise<number[]> => {
  const { json } = await jfetch("/embed", { input, ...extra });
  return (json as { embeddings: number[][] }).embeddings[0];
};

// ============================================================ A. service surface
{
  const h = await (await fetch(`${BASE}/health`)).json();
  check("health", h.status === "ok" && h.device === "cuda" && typeof h.gpu === "string", `${h.device}/${h.gpu}`);

  const m = await (await fetch(`${BASE}/metadata`)).json();
  check("metadata", m.dimension === 768 && Array.isArray(m.prompt_names) && m.prompt_names.length >= 15, `${m.prompt_names?.length ?? 0} prompts`);

  const models = await (await fetch(`${V1}/models`)).json();
  const ids: string[] = models.data.map((x: { id: string }) => x.id);
  check("GET /v1/models", ids.includes("embeddinggemma-2") && ids.some((x: string) => x.endsWith(":STS")), `${ids.length} entries`);
}

// ============================================================ B. /embed option matrix
{
  const v = await emb("hello world");
  check("/embed basic", v.length === 768 && v.every(Number.isFinite));

  for (const d of [768, 512, 256, 128, 64]) {
    const t = await emb("truncate me", { truncate_dim: d });
    check(`/embed truncate_dim=${d}`, t.length === d);
  }
  const bad = await jfetch("/embed", { input: "x", truncate_dim: 9999 });
  check("/embed truncate 9999 -> 400", bad.status === 400);

  const pn = await emb("q", { prompt_name: "SearchQuery" });
  const raw = await emb("task: search result | query: q");
  check("/embed prompt_name ~= manual prefix", cosine(pn, raw) > 0.99, `cos=${cosine(pn, raw).toFixed(4)}`);

  const custom = await emb("q", { prompt: "task: search result | query: " });
  check("/embed prompt custom == prompt_name", cosine(custom, pn) > 0.999);

  const both = await jfetch("/embed", { input: "x", prompt: "a", prompt_name: "STS" });
  check("/embed prompt+name -> 400", both.status === 400);

  const bs = await jfetch("/embed", { input: Array.from({ length: 16 }, (_, i) => `n ${i}`), batch_size: 4 });
  check("/embed batch_size param", bs.status === 200 && (bs.json as { count: number }).count === 16);

  const norm = await emb("norm check", { normalize_embeddings: false });
  const normT = await emb("norm check", { normalize_embeddings: true });
  check("/embed normalize flag accepted", norm.length === 768 && cosine(norm, normT) > 0.99);
}

// ============================================================ C. /embed error surface
{
  const cases: Array<[string, object, number]> = [
    ["unknown key", { input: { foo: 1 } }, 400],
    ["scalar input", { input: 3.14 }, 422],
    ["empty list", { input: [] }, 400],
    ["bad b64", { input: { image: { b64: "!!!!" } } }, 400],
    ["unknown prompt_name", { input: "x", prompt_name: "Nope" }, 400],
    ["batch_size 0", { input: ["x"], batch_size: 0 }, 400],
  ];
  for (const [name, body, want] of cases) {
    const r = await jfetch("/embed", body);
    check(`/embed ${name} -> ${want}`, r.status === want, `got ${r.status}`);
  }
}

// ============================================================ D. official `openai` pkg
{
  const openai = new OpenAI({ baseURL: V1, apiKey: "local", maxRetries: 1 });

  const r1 = await openai.embeddings.create({ model: "embeddinggemma-2", input: "hello" });
  check("openai.embeddings float", r1.data[0].embedding.length === 768 && r1.model === "embeddinggemma-2" && r1.usage.total_tokens > 0, `tokens=${r1.usage.total_tokens}`);

  const r2 = await openai.embeddings.create({ model: "embeddinggemma-2:Document", input: ["a", "b"] });
  check("openai task-suffix + batch", r2.data.length === 2 && r2.model === "embeddinggemma-2:Document");

  const r3 = await openai.embeddings.create({ model: "embeddinggemma-2", input: "d256", dimensions: 256 });
  check("openai dimensions=256", r3.data[0].embedding.length === 256 && Array.isArray(r3.data[0].embedding));

  const r4 = await openai.embeddings.create({ model: "embeddinggemma-2", input: "b64", encoding_format: "base64" });
  const dec = Array.from(new Float32Array(Buffer.from(r4.data[0].embedding as string, "base64").buffer));
  const flt = await openai.embeddings.create({ model: "embeddinggemma-2", input: "b64" });
  check("openai base64==float", dec.length === 768 && Math.abs(cosine(dec, flt.data[0].embedding as number[]) - 1) < 1e-5);

  const list = await openai.models.list();
  check("openai.models.list", list.data.length > 1);

  try {
    await openai.embeddings.create({ model: "embeddinggemma-2:Nope", input: "x" });
    check("openai bad suffix -> throws", false);
  } catch (e) {
    check("openai bad suffix -> throws", e instanceof OpenAI.APIError && (e as { status?: number }).status === 400, `status=${(e as { status?: number }).status}`);
  }

  const alias = await jfetch("/embeddings", { model: "embeddinggemma-2", input: "alias" }); // no /v1 alias
  check("POST /embeddings (no /v1) alias", alias.status === 200);
}

// ============================================================ E. ai-sdk deep pass
{
  const ai = createOpenAI({ baseURL: V1, apiKey: "local" });
  const model = (id: string) => (ai as unknown as Record<string, (i: string) => unknown>).textEmbeddingModel!(id);

  const big = await aiEmbedMany({ model: model("embeddinggemma-2") as never, values: Array.from({ length: 64 }, (_, i) => `doc ${i} content`) });
  check("ai-sdk embedMany x64", big.embeddings.length === 64 && big.embeddings[0].length === 768);

  const d128 = await aiEmbed({ model: model("embeddinggemma-2") as never, value: "d128", providerOptions: { openai: { dimensions: 128 } } });
  check("ai-sdk providerOptions 128d", d128.embedding.length === 128);

  const long = "long text ".repeat(400); // ~4k chars
  const lr = await aiEmbed({ model: model("embeddinggemma-2") as never, value: long });
  check("ai-sdk long input 4k chars", lr.embedding.length === 768 && lr.usage?.tokens > 100, `tokens=${lr.usage?.tokens}`);

  const sts = await aiEmbed({ model: model("embeddinggemma-2:STS") as never, value: "sym" });
  const plain = await aiEmbed({ model: model("embeddinggemma-2") as never, value: "sym" });
  check("ai-sdk suffix affects vector", cosine(sts.embedding, plain.embedding) < 0.999);
}

// ============================================================ F. cross-provider consistency
{
  const text = "cross provider consistency check";
  const raw = await emb(text);
  const openai = new OpenAI({ baseURL: V1, apiKey: "local" });
  const viaOpenai = (await openai.embeddings.create({ model: "embeddinggemma-2", input: text })).data[0].embedding as number[];
  const ai = createOpenAI({ baseURL: V1, apiKey: "local" });
  const viaAi = (await aiEmbed({ model: (ai as any).textEmbeddingModel("embeddinggemma-2"), value: text })).embedding;
  const c1 = cosine(raw, viaOpenai), c2 = cosine(raw, viaAi), c3 = cosine(viaAi, viaOpenai);
  check("3-way parity raw/openai/ai-sdk", [c1, c2, c3].every((c) => c > 0.9999), `min=${Math.min(c1, c2, c3).toFixed(6)}`);
}

// ============================================================ G. multimodal still fine (quick)
{
  const png = Buffer.from(await Bun.file(new URL("./assets/blue_bg.png", import.meta.url).pathname).arrayBuffer()).toString("base64");
  const v = await emb({ text: "a blue background  <", image: [{ b64: png }] });
  check("/embed multimodal interleave", v.length === 768 && v.every(Number.isFinite));
}

// ============================================================ H. DashScope-compatible API
{
  const DS = `${BASE}/api/v1/services/embeddings/multimodal-embedding/multimodal-embedding`;
  const post = async (body: object, path = DS) => {
    const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    return { status: r.status, json: (await r.json()) as Record<string, unknown> };
  };
  const png = Buffer.from(await Bun.file(new URL("./assets/blue_bg.png", import.meta.url).pathname).arrayBuffer()).toString("base64");
  const dataUri = `data:image/png;base64,${png}`;

  const r1 = await post({
    model: "qwen3-vl-embedding",
    input: { contents: [{ text: "一只猫" }, { image: dataUri }, { image: new URL("./assets/red_bg.png", import.meta.url).pathname }] },
  });
  const e1 = (r1.json as any).output?.embeddings;
  check(
    "dashscope independent",
    r1.status === 200 && e1?.length === 3 && e1[0].type === "text" && e1[1].type === "image" && e1.every((x: any) => x.embedding.length === 768),
    `n=3 types=${e1?.map((x: any) => x.type).join(",")}`,
  );

  const r2 = await post({
    model: "x",
    input: { contents: [{ text: " fusion ", image: dataUri }] },
    parameters: { dimension: 256 },
  });
  const e2 = (r2.json as any).output?.embeddings;
  check("dashscope fused element", r2.status === 200 && e2?.length === 1 && e2[0].type === "fused" && e2[0].embedding.length === 256);

  const r3 = await post({
    model: "x",
    input: { contents: [{ text: "a" }, { image: dataUri }, { video: new URL("./assets/blue_clip.mp4", import.meta.url).pathname }] },
    parameters: { enable_fusion: true },
  });
  const e3 = (r3.json as any).output?.embeddings;
  const u3 = (r3.json as any).usage;
  check("dashscope enable_fusion", r3.status === 200 && e3?.length === 1 && e3[0].type === "fusion" && u3?.total_tokens > 0, `dim=${e3?.[0]?.embedding?.length}`);

  const r4 = await post({ input: { contents: [{ multiImages: undefined, multi_images: [new URL("./assets/blue_bg.png", import.meta.url).pathname, new URL("./assets/red_bg.png", import.meta.url).pathname] }] } });
  const e4 = (r4.json as any).output?.embeddings;
  check("dashscope multi_images", r4.status === 200 && e4?.length === 2 && e4.every((x: any) => x.type === "image"));

  const r5 = await post({ input: { contents: [] } });
  check(
    "dashscope empty contents -> 400",
    r5.status === 400 && (r5.json as any).code === "InvalidParameter" && typeof (r5.json as any).request_id === "string",
    (r5.json as any).message,
  );

  const alias = await post({ input: { contents: ["hi"] } }, `${BASE}/embed/dashscope`);
  check("dashscope alias /embed/dashscope", alias.status === 200);

  const st = await jfetch("/embed/st", { input: "route rename check" });
  check("/embed/st (sentence-transformers style)", st.status === 200);
}

console.log(`\n${total.n - failures}/${total.n} PASS` + (failures ? `, ${failures} FAIL` : ""));
process.exit(failures ? 1 : 0);
