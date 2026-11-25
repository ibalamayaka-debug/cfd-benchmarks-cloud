export interface Env {
  CFD_BENCH_KV: KVNamespace;
  RATE_LIMIT_KV: KVNamespace;
  COMPUTE_BASE_URL?: string;
  COMPUTE_AUTH_TOKEN?: string;
  ALLOWED_ORIGIN: string;
  MAX_RUNS_PER_IP_PER_HOUR: string;
  ASSETS?: Fetcher;
}

type BenchId = 'cavity' | 'step' | 'pipe';
const VALID_BENCHES = new Set<BenchId>(['cavity', 'step', 'pipe']);
const VALID_FILES = new Set(['field.vtp', 'wall.vtp', 'streamlines.vtp', 'metrics.json']);

function corsHeaders(env: Env): Record<string, string> {
  return {
    'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN || '*',
    'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  };
}

function jsonResp(data: unknown, status: number, env: Env): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { 'Content-Type': 'application/json; charset=utf-8', ...corsHeaders(env) },
  });
}

async function sha16(canonical: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(canonical));
  return Array.from(new Uint8Array(digest))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
    .slice(0, 16);
}

function nearest(val: number, choices: readonly number[]): number {
  return choices.reduce((best, cur) => (Math.abs(cur - val) < Math.abs(best - val) ? cur : best), choices[0]);
}

async function normalizeAndKey(bench: BenchId, raw: Record<string, unknown>) {
  if (bench === 'cavity') {
    const re = nearest(Number(raw.re ?? 1000), [100, 400, 1000, 3200]);
    return { params: { re }, key: await sha16(`bench|cavity|re=${re}`) };
  }
  if (bench === 'step') {
    const re = nearest(Number(raw.re ?? 37500), [400, 5000, 28000, 37500]);
    return { params: { re }, key: await sha16(`bench|step|re=${re}`) };
  }
  const diameter_mm = nearest(Number(raw.diameter_mm ?? 50), [50, 80, 100]);
  const velocity_ms = nearest(Number(raw.velocity_ms ?? 2.0), [0.03, 0.8, 2.0, 4.0]);
  return {
    params: { diameter_mm, velocity_ms },
    key: await sha16(`bench|pipe|d=${diameter_mm.toFixed(0)}|u=${velocity_ms.toFixed(2)}`),
  };
}

function buildUrls(bench: BenchId, caseKey: string, hasWall: boolean) {
  return {
    field: `/api/results/${bench}/${caseKey}/field.vtp`,
    streamlines: `/api/results/${bench}/${caseKey}/streamlines.vtp`,
    wall: hasWall ? `/api/results/${bench}/${caseKey}/wall.vtp` : undefined,
    metrics: `/api/results/${bench}/${caseKey}/metrics.json`,
  };
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, { status: 204, headers: corsHeaders(env) });
    }

    const url = new URL(request.url);
    const path = url.pathname;

    if (!path.startsWith('/api/')) {
      if (env.ASSETS) return env.ASSETS.fetch(request);
      return new Response('Not Found', { status: 404 });
    }

    if (path === '/api/health') {
      return jsonResp(
        {
          status: 'ok',
          benchmarks: ['cavity', 'step', 'pipe'],
          compute_configured: Boolean(env.COMPUTE_BASE_URL),
        },
        200,
        env
      );
    }

    const casesMatch = path.match(/^\/api\/cases\/(cavity|step|pipe)$/);
    if (casesMatch && request.method === 'GET') {
      const bench = casesMatch[1] as BenchId;
      const raw = await env.CFD_BENCH_KV.get(`index:${bench}`);
      return jsonResp({ bench, cases: raw ? JSON.parse(raw) : [] }, 200, env);
    }

    const resMatch = path.match(/^\/api\/results\/(cavity|step|pipe)\/([a-f0-9]{16})\/([a-z0-9_.-]+)$/);
    if (resMatch && (request.method === 'GET' || request.method === 'HEAD')) {
      const [, bench, caseKey, fname] = resMatch;
      if (!VALID_FILES.has(fname)) return jsonResp({ error: 'Invalid file' }, 400, env);
      const val = await env.CFD_BENCH_KV.get(`case:${bench}:${caseKey}:${fname}`);
      if (!val) return jsonResp({ error: 'Not found in KV' }, 404, env);
      const ct = fname.endsWith('.json') ? 'application/json; charset=utf-8' : 'application/xml; charset=utf-8';
      return new Response(request.method === 'HEAD' ? null : val, {
        status: 200,
        headers: {
          'Content-Type': ct,
          'Cache-Control': 'public, max-age=31536000, immutable',
          ...corsHeaders(env),
        },
      });
    }

    const simMatch = path.match(/^\/api\/simulate\/(cavity|step|pipe)$/);
    if (simMatch && request.method === 'POST') {
      const bench = simMatch[1] as BenchId;
      if (!VALID_BENCHES.has(bench)) return jsonResp({ error: 'Invalid benchmark' }, 400, env);

      let body: Record<string, unknown> = {};
      try {
        body = (await request.json()) as Record<string, unknown>;
      } catch {
        return jsonResp({ error: 'Invalid JSON body' }, 400, env);
      }

      const { params, key: caseKey } = await normalizeAndKey(bench, body);
      const force = Boolean(body.force);

      if (!force) {
        const cachedRaw = await env.CFD_BENCH_KV.get(`case:${bench}:${caseKey}:metrics.json`);
        if (cachedRaw) {
          const parsed = JSON.parse(cachedRaw) as { file_sizes_kb?: Record<string, number> };
          return jsonResp(
            {
              status: 'ok',
              cached: true,
              backend: 'cloudflare-kv-cache',
              bench,
              case_key: caseKey,
              metrics: parsed,
              urls: buildUrls(bench, caseKey, Boolean(parsed.file_sizes_kb?.['wall.vtp'] ?? bench === 'pipe')),
            },
            200,
            env
          );
        }
      }

      if (!env.COMPUTE_BASE_URL) {
        return jsonResp({ error: 'COMPUTE_BASE_URL not configured' }, 503, env);
      }

      const ip = request.headers.get('CF-Connecting-IP') || 'anon';
      const hourBucket = Math.floor(Date.now() / 3600000);
      const rlKey = `rl:cfdbench:${ip}:${hourBucket}`;
      const limit = Math.max(1, parseInt(env.MAX_RUNS_PER_IP_PER_HOUR || '30', 10));
      const used = parseInt((await env.RATE_LIMIT_KV.get(rlKey)) || '0', 10);
      if (used >= limit) {
        return jsonResp({ error: `Hourly cloud CFD quota reached (${limit}/hr).` }, 429, env);
      }

      const base = env.COMPUTE_BASE_URL.replace(/\/$/, '');
      const headers: Record<string, string> = { 'Content-Type': 'application/json' };
      if (env.COMPUTE_AUTH_TOKEN) headers['Authorization'] = `Bearer ${env.COMPUTE_AUTH_TOKEN}`;

      const modalRes = await fetch(`${base}/simulate/${bench}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(params),
      });
      if (!modalRes.ok) {
        const errText = await modalRes.text();
        return jsonResp(
          { error: `Cloud OpenFOAM (${bench}) failed (${modalRes.status}): ${errText.slice(0, 600)}` },
          502,
          env
        );
      }

      const payload = (await modalRes.json()) as {
        metrics: Record<string, unknown>;
        files_content: Record<string, string>;
      };

      const writes: Promise<void>[] = [
        env.CFD_BENCH_KV.put(`case:${bench}:${caseKey}:metrics.json`, JSON.stringify(payload.metrics)),
      ];
      for (const [fname, content] of Object.entries(payload.files_content || {})) {
        if (VALID_FILES.has(fname) && content) {
          writes.push(env.CFD_BENCH_KV.put(`case:${bench}:${caseKey}:${fname}`, content));
        }
      }
      await Promise.all(writes);

      const idxRaw = await env.CFD_BENCH_KV.get(`index:${bench}`);
      const idxList: Array<{ case_key: string }> = idxRaw ? JSON.parse(idxRaw) : [];
      const updated = [payload.metrics as { case_key: string }, ...idxList.filter((x) => x.case_key !== caseKey)].slice(0, 20);
      await env.CFD_BENCH_KV.put(`index:${bench}`, JSON.stringify(updated));
      await env.RATE_LIMIT_KV.put(rlKey, String(used + 1), { expirationTtl: 3900 });

      return jsonResp(
        {
          status: 'ok',
          cached: false,
          backend: 'modal-openfoam-v2512',
          bench,
          case_key: caseKey,
          remaining_quota: Math.max(0, limit - used - 1),
          metrics: payload.metrics,
          urls: buildUrls(bench, caseKey, Boolean(payload.files_content?.['wall.vtp'])),
        },
        200,
        env
      );
    }

    return jsonResp({ error: 'Not found' }, 404, env);
  },
};
