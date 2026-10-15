/**
 * Shared service address for every check script.
 * EMBG2_BASE_URL env overrides; otherwise the repo default (server port lives in ../.env, profile-aware).
 */
export const BASE = (process.env.EMBG2_BASE_URL ?? "http://127.0.0.1:50233").replace(/\/+$/, "");

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
