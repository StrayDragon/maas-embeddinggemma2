/**
 * Shared service address for every check script.
 * EMBG2_BASE_URL env overrides; otherwise the repo default (server port lives in ../.env, profile-aware).
 * EMBG2_API_KEY / OMLX_API_KEY bearer for remotes that require one (oMLX); local server ignores it.
 * EMBG2_MODEL  model id sent to /v1/embeddings (oMLX serves "embeddinggemma-2-bf16";
 *              the local server id is "embeddinggemma-2" and also accepts ":<Task>" suffixes).
 * EMBG2_REMOTE=1 marks a remote that lacks the local task-suffix ids.
 */
export const BASE = (process.env.EMBG2_BASE_URL ?? "http://127.0.0.1:50233").replace(/\/+$/, "");
export const API_KEY = process.env.EMBG2_API_KEY ?? process.env.OMLX_API_KEY ?? "local";
export const MODEL = process.env.EMBG2_MODEL ?? "embeddinggemma-2";
export const REMOTE = process.env.EMBG2_REMOTE === "1" || BASE.includes("11534");

/** Task-suffix ids only exist on the local server; remotes get the bare id. */
export const modelId = (task?: string): string =>
  !task || REMOTE ? MODEL : `${MODEL}:${task}`;

/** auth header for raw fetch calls */
export const authHeaders: Record<string, string> = API_KEY
  ? { Authorization: `Bearer ${API_KEY}` }
  : {};

/** Fail fast with a friendly hint when the service is not reachable. */
export async function preflight(): Promise<void> {
  try {
    const r = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(3000) });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    await r.json();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    console.error(`✗ embedding 服务不可达: ${BASE} (${msg})`);
    console.error("  本机启动:    just serve            # 仓库根目录");
    console.error("  远程/自定义: EMBG2_BASE_URL=http://host:port 后重试");
    process.exit(1);
  }
}
