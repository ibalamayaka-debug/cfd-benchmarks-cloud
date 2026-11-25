import type { BenchId, BenchMetrics, BenchSimResponse } from './types';

export async function fetchBenchCases(bench: BenchId): Promise<BenchMetrics[]> {
  const res = await fetch(`/api/cases/${bench}`);
  if (!res.ok) return [];
  const data = (await res.json()) as { cases?: BenchMetrics[] };
  return data.cases ?? [];
}

export async function runBenchSimulation(
  bench: BenchId,
  params: Record<string, unknown>
): Promise<BenchSimResponse> {
  const res = await fetch(`/api/simulate/${bench}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  const data = (await res.json()) as BenchSimResponse & { error?: string };
  if (!res.ok) throw new Error(data.error || `Simulation failed (${res.status})`);
  return data;
}

export async function fetchVtpBuffer(url: string): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch VTP (${res.status})`);
  return res.arrayBuffer();
}
