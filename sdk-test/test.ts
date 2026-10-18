/**
 * Verify embeddinggemma2-serve against the Vercel AI SDK (`ai` + `@ai-sdk/openai`).
 *   bun i && bun run test
 */
import { createOpenAI } from "@ai-sdk/openai";
import { embed, embedMany } from "ai";
import { API_KEY, BASE as ROOT, modelId, REMOTE, preflight } from "./base";

await preflight();
const BASE = `${ROOT}/v1`;

const openai = createOpenAI({ baseURL: BASE, apiKey: API_KEY });
// ai<=4 provider name vs newer: prefer textEmbeddingModel, fall back to embedding
const provider = openai as unknown as Record<string, (id: string) => unknown>;
const model = (id: string) =>
  typeof provider.textEmbeddingModel === "function"
    ? (provider.textEmbeddingModel as (id: string) => unknown)(id)
    : (provider.embedding as (id: string) => unknown)(id);

let failures = 0;
function check(name: string, ok: boolean, info: string) {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${info}`);
  if (!ok) failures++;
}

function cosine(a: number[], b: number[]): number {
  let d = 0,
    na = 0,
    nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return d / (Math.sqrt(na) * Math.sqrt(nb));
}

// 1) single embed, native 768d
{
  const { embedding, value, usage } = await embed({
    model: model(modelId()) as never,
    value: "The northern lights are caused by charged particles from the sun.",
  });
  const finite = embedding.every((x) => Number.isFinite(x));
  check(
    "embed/single",
    embedding.length === 768 && finite && usage?.tokens > 0,
    `dim=${embedding.length} tokens=${usage?.tokens} value="${value.slice(0, 24)}..."`,
  );
}

// 2) embedMany batch
{
  const { embeddings } = await embedMany({
    model: model(modelId()) as never,
    values: [
      "A red mug on the table",
      "Coffee in a red cup",
      "The stock market fell sharply",
    ],
  });
  const [a, b, c] = embeddings;
  const ab = cosine(a, b);
  const ac = cosine(a, c);
  check(
    "embedMany/batch",
    embeddings.length === 3 && a.length === 768 && ab > ac,
    `n=3 dim=768 cos(mug,mug)=${ab.toFixed(3)} > cos(mug,stocks)=${ac.toFixed(3)}`,
  );
}

// 3) MRL truncation via providerOptions.openai.dimensions
{
  const { embedding } = await embed({
    model: model(modelId()) as never,
    value: "dimension reduction test",
    providerOptions: { openai: { dimensions: 256 } },
  });
  check("providerOptions/dimensions=256", embedding.length === 256, `dim=${embedding.length}`);
}

// 4) task-steered model id suffix (`:STS`) changes the embedding vs bare id.
// Remote servers (oMLX) have no :Task ids -- the bare id is the same model, so
// cos == 1 there and the suffix-error check cannot apply either.
{
  const plain = await embed({ model: model(modelId()) as never, value: "sunny day" });
  const sts = await embed({ model: model(modelId("STS")) as never, value: "sunny day" });
  const cos = cosine(plain.embedding, sts.embedding);
  check("task-suffix/:STS", REMOTE ? Math.abs(cos - 1) < 1e-6 : cos < 0.999, `cos(plain,STS)=${cos.toFixed(4)}`);
}

// 5) task suffix via embedMany too + symmetric-task sanity
{
  const { embeddings } = await embedMany({
    model: model(modelId("Clustering")) as never,
    values: ["apple", "banana", "CPU fan noise"],
  });
  const fruit = cosine(embeddings[0], embeddings[1]);
  const mixed = cosine(embeddings[0], embeddings[2]);
  check("embedMany/:Clustering", fruit > mixed, `cos(apple,banana)=${fruit.toFixed(3)} > cos(apple,cpu)=${mixed.toFixed(3)}`);
}

// 6) invalid task suffix -> readable error from the SDK (local only: remotes
// accept any model id string and 404/400 differently)
{
  if (REMOTE) {
    console.log(`SKIP  error/unknown-suffix  (remote server: no :Task ids)`);
  } else {
    try {
      await embed({ model: model(modelId("Nope")) as never, value: "x" });
      check("error/unknown-suffix", false, "no error thrown");
    } catch (e) {
      const anyErr = e as Record<string, unknown>;
      const code = Number(anyErr.statusCode ?? 0);
      const msg = String(anyErr.message ?? e);
      check("error/unknown-suffix", code === 400 || /400|task suffix/.test(msg), `status=${code} ${msg.slice(0, 60)}`);
    }
  }
}

// 7) regression: CJK task-prefix forms (client-side prefix; former NaN->500)
{
  const { embeddings } = await embedMany({
    model: model(modelId()) as never,
    values: [
      "task: search result | query: 对数组按分数排序并取前 K 个的函数",
      "task: search result | query: 按分数从高到低取前若干个结果",
      "对数组按分数排序并取前 K 个的函数", // bare control
    ],
  });
  const finite = embeddings.every((e) => e.every((x) => Number.isFinite(x)));
  check(
    "regression/cjk-task-prefix",
    embeddings.length === 3 && finite,
    `n=${embeddings.length} finite=${finite}`,
  );
}

// 8) regression: trigger strings via task-steered model ids (:SearchQuery/:Document)
{
  const q = await embed({
    model: model(modelId("SearchQuery")) as never,
    value: "对数组按分数排序并取前 K 个的函数",
  });
  const d = await embed({
    model: model(modelId("Document")) as never,
    value: "按分数从高到低取前若干个结果",
  });
  const ok = q.embedding.every(Number.isFinite) && d.embedding.every(Number.isFinite);
  check("regression/suffix-steer-cjk", q.embedding.length === 768 && ok, `finite=${ok}`);
}

console.log(failures === 0 ? "\nALL PASS" : `\n${failures} FAILURES`);
process.exit(failures === 0 ? 0 : 1);
