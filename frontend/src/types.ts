export type BenchId = 'cavity' | 'step' | 'pipe';
export type FieldId = 'U_mag' | 'u_x' | 'u_y' | 'p_kin' | 'p_pa' | 'cp' | 'vort_z' | 'k_turb' | 'stream_fn';
export type PaletteId = 'rainbow' | 'thermal';

export interface BenchMetrics {
  case_key: string;
  bench: BenchId;
  params: Record<string, number>;
  kpis: Record<string, any>;
  charts: Record<string, any>;
  timing_sec: {
    cfd_openfoam: number;
    total: number;
  };
  file_sizes_kb: Record<string, number>;
  computed_at: string;
}

export interface BenchSimResponse {
  status: string;
  cached: boolean;
  backend: string;
  bench: BenchId;
  case_key: string;
  metrics: BenchMetrics;
  urls: {
    field: string;
    streamlines: string;
    wall?: string;
    metrics: string;
  };
}
