import React, { useCallback, useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  BarChart3,
  Database,
  Eye,
  Flame,
  Gauge,
  Layers,
  PanelLeftClose,
  PanelLeftOpen,
  Play,
  Timer,
  Wind,
} from 'lucide-react';
import { fetchBenchCases, runBenchSimulation } from './api';
import { BenchCharts } from './components/BenchCharts';
import { BenchViewer } from './components/BenchViewer';
import { detectInitialLang, TRANSLATIONS, type Lang } from './i18n';
import type { BenchId, BenchMetrics, BenchSimResponse, FieldId, PaletteId } from './types';

function detectInitialBench(): BenchId {
  try {
    const q = new URLSearchParams(window.location.search).get('bench');
    if (q === 'cavity' || q === 'step' || q === 'pipe') return q;
  } catch { /* ignore */ }
  return 'step';
}

const DEFAULT_PARAMS: Record<BenchId, Record<string, number>> = {
  cavity: { re: 1000 },
  step: { re: 37500 },
  pipe: { diameter_mm: 50, velocity_ms: 2.0 },
};

function makeCachedResp(bench: BenchId, m: BenchMetrics): BenchSimResponse {
  return {
    status: 'ok',
    cached: true,
    backend: 'cloudflare-kv-cache',
    bench,
    case_key: m.case_key,
    metrics: m,
    urls: {
      field: `/api/results/${bench}/${m.case_key}/field.vtp`,
      streamlines: `/api/results/${bench}/${m.case_key}/streamlines.vtp`,
      wall: bench === 'pipe' ? `/api/results/${bench}/${m.case_key}/wall.vtp` : undefined,
      metrics: `/api/results/${bench}/${m.case_key}/metrics.json`,
    },
  };
}

export const App: React.FC = () => {
  const [lang, setLang] = useState<Lang>(detectInitialLang);
  const [bench, setBench] = useState<BenchId>(detectInitialBench);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  const [paramsMap, setParamsMap] = useState<Record<BenchId, Record<string, number>>>(DEFAULT_PARAMS);
  const [results, setResults] = useState<Partial<Record<BenchId, BenchSimResponse>>>({});
  const [cachedMap, setCachedMap] = useState<Record<BenchId, BenchMetrics[]>>({ cavity: [], step: [], pipe: [] });

  const [field, setField] = useState<FieldId>('U_mag');
  const [palette, setPalette] = useState<PaletteId>('rainbow');
  const [showStreams, setShowStreams] = useState(true);
  const [showWall, setShowWall] = useState(true);
  const [showEdges, setShowEdges] = useState(false);
  const [showChart, setShowChart] = useState(true);
  const [playing, setPlaying] = useState(true);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const t = TRANSLATIONS[lang];
  const curParams = paramsMap[bench];
  const curResult = results[bench] ?? null;
  const curCached = cachedMap[bench] ?? [];

  useEffect(() => {
    document.title = t.pageTitle;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    try { localStorage.setItem('cfdbench_lang', lang); } catch { /* ignore */ }
  }, [lang, t.pageTitle]);

  const loadCached = useCallback(async (target: BenchId) => {
    try {
      const list = await fetchBenchCases(target);
      setCachedMap((prev) => ({ ...prev, [target]: list }));
      if (list.length > 0) {
        setResults((prev) => (prev[target] ? prev : { ...prev, [target]: makeCachedResp(target, list[0]) }));
        setParamsMap((prev) => ({ ...prev, [target]: list[0].params }));
      }
    } catch { /* ignore */ }
  }, []);

  useEffect(() => {
    loadCached(bench);
  }, [bench, loadCached]);

  const handleRun = async (override?: Record<string, number>) => {
    const p = override ?? curParams;
    setLoading(true);
    setError('');
    try {
      const resp = await runBenchSimulation(bench, p);
      setResults((prev) => ({ ...prev, [bench]: resp }));
      setParamsMap((prev) => ({ ...prev, [bench]: resp.metrics.params }));
      const list = await fetchBenchCases(bench);
      setCachedMap((prev) => ({ ...prev, [bench]: list }));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const setParam = (key: string, val: number) => {
    setParamsMap((prev) => ({ ...prev, [bench]: { ...prev[bench], [key]: val } }));
  };

  const k = curResult?.metrics?.kpis;
  const tm = curResult?.metrics?.timing_sec;

  const availFields: FieldId[] =
    bench === 'cavity'
      ? ['U_mag', 'u_x', 'u_y', 'p_kin', 'vort_z', 'stream_fn']
      : bench === 'step'
        ? ['U_mag', 'u_x', 'u_y', 'cp', 'p_kin', 'k_turb']
        : ['U_mag', 'u_x', 'p_pa', 'k_turb'];

  useEffect(() => {
    if (!availFields.includes(field)) setField('U_mag');
  }, [bench, field]);

  return (
    <div className={`app-layout ${sidebarOpen ? '' : 'sidebar-collapsed'}`}>
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-top">
            <div className="brand-badge">{t.badge}</div>
            <div className="lang-switch">
              <button type="button" className={lang === 'zh' ? 'active' : ''} onClick={() => setLang('zh')}>中文</button>
              <button type="button" className={lang === 'en' ? 'active' : ''} onClick={() => setLang('en')}>EN</button>
            </div>
          </div>
          <h1>{t.title}</h1>
          <p>{t.subtitle}</p>
        </div>

        <div className="lab-tabs">
          {(['cavity', 'step', 'pipe'] as BenchId[]).map((id) => (
            <button
              key={id}
              type="button"
              className={`lab-tab ${bench === id ? 'active' : ''}`}
              onClick={() => { setBench(id); setError(''); }}
            >
              {t.tabs[id]}
            </button>
          ))}
        </div>

        <div className="goal-banner">{t.goals[bench]}</div>

        <section className="panel-section">
          <div className="section-title"><Wind size={14} /> {t.sectionParams}</div>

          {bench === 'cavity' && (
            <label className="select-row">
              <span>{t.cavityRe}</span>
              <select value={curParams.re} onChange={(e) => setParam('re', Number(e.target.value))}>
                <option value={100}>Re = 100 (Stokes-Dominated Vortex)</option>
                <option value={400}>Re = 400 (Shifted Core Vortex)</option>
                <option value={1000}>Re = 1,000 (Ghia Classic Benchmark)</option>
                <option value={3200}>Re = 3,200 (Strong Secondary Eddies)</option>
              </select>
            </label>
          )}

          {bench === 'step' && (
            <label className="select-row">
              <span>{t.stepRe}</span>
              <select value={curParams.re} onChange={(e) => setParam('re', Number(e.target.value))}>
                <option value={400}>{t.stepReOptions[400]}</option>
                <option value={5000}>{t.stepReOptions[5000]}</option>
                <option value={28000}>{t.stepReOptions[28000]}</option>
                <option value={37500}>{t.stepReOptions[37500]}</option>
              </select>
            </label>
          )}

          {bench === 'pipe' && (
            <>
              <label className="select-row">
                <span>{t.pipeDia}</span>
                <select value={curParams.diameter_mm} onChange={(e) => setParam('diameter_mm', Number(e.target.value))}>
                  <option value={50}>D = 50 mm (DN50 · L = 1.60 m)</option>
                  <option value={80}>D = 80 mm (DN80 · L = 2.56 m)</option>
                  <option value={100}>D = 100 mm (DN100 · L = 3.20 m)</option>
                </select>
              </label>
              <label className="select-row">
                <span>{t.pipeVel}</span>
                <select value={curParams.velocity_ms} onChange={(e) => setParam('velocity_ms', Number(e.target.value))}>
                  <option value={0.03}>{t.pipeVelOptions['0.03']}</option>
                  <option value={0.8}>{t.pipeVelOptions['0.8']}</option>
                  <option value={2.0}>{t.pipeVelOptions['2']}</option>
                  <option value={4.0}>{t.pipeVelOptions['4']}</option>
                </select>
              </label>
            </>
          )}

          <button type="button" className="run-btn" disabled={loading} onClick={() => handleRun()}>
            <Play size={15} /> {loading ? t.runningBtn : t.runBtn}
          </button>
        </section>

        <section className="panel-section">
          <div className="section-title"><Eye size={14} /> {t.sectionVisual}</div>

          <label className="select-row">
            <span>{t.fieldLabel}</span>
            <select value={field} onChange={(e) => setField(e.target.value as FieldId)}>
              {availFields.map((fid) => (
                <option key={fid} value={fid}>{t.fields[fid]}</option>
              ))}
            </select>
          </label>

          <div className="palette-switch">
            <button type="button" className={palette === 'rainbow' ? 'active' : ''} onClick={() => setPalette('rainbow')}>{t.palRainbow}</button>
            <button type="button" className={palette === 'thermal' ? 'active' : ''} onClick={() => setPalette('thermal')}>{t.palThermal}</button>
          </div>

          <div className="toggle-list">
            <label className="checkbox-row"><input type="checkbox" checked={showStreams} onChange={(e) => setShowStreams(e.target.checked)} /><span>{t.showStreams}</span></label>
            {bench === 'pipe' && (
              <label className="checkbox-row"><input type="checkbox" checked={showWall} onChange={(e) => setShowWall(e.target.checked)} /><span>{t.showWall}</span></label>
            )}
            <label className="checkbox-row"><input type="checkbox" checked={showEdges} onChange={(e) => setShowEdges(e.target.checked)} /><span>{t.showMeshEdges}</span></label>
            <label className="checkbox-row"><input type="checkbox" checked={showChart} onChange={(e) => setShowChart(e.target.checked)} /><span>{t.showChart}</span></label>
          </div>
        </section>

        <div className="sidebar-footer"><Flame size={13} /><span>OpenFOAM v2512 · Modal Cloud + Cloudflare KV</span></div>
      </aside>

      <main className="workspace">
        <header className="kpi-bar">
          <button type="button" className="collapse-btn" onClick={() => setSidebarOpen((v) => !v)} title={sidebarOpen ? t.hideSidebar : t.showSidebar}>
            {sidebarOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
          </button>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-label"><Gauge size={14} /> {t.kpis.re}</div>
              <div className="kpi-value gold">
                {k ? `Re ${Number(k.reynolds).toLocaleString()}${k.regime ? ` (${k.regime})` : ''}` : '--'}
              </div>
            </div>

            {bench === 'cavity' && (
              <>
                <div className="kpi-card">
                  <div className="kpi-label"><Activity size={14} /> {t.kpis.cavityCenter}</div>
                  <div className="kpi-value coral">{k ? `(${k.vortex_center_x}, ${k.vortex_center_y})` : '--'}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label"><BarChart3 size={14} /> {t.kpis.cavityGhia}</div>
                  <div className="kpi-value sage">{k ? `RMSE ${k.ghia_rmse}` : '--'}</div>
                </div>
              </>
            )}

            {bench === 'step' && (
              <>
                <div className="kpi-card">
                  <div className="kpi-label"><Activity size={14} /> {t.kpis.stepXr}</div>
                  <div className="kpi-value coral">{k ? `xR/H = ${k.reattachment_xr_h}` : '--'}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label"><BarChart3 size={14} /> {t.kpis.stepRev}</div>
                  <div className="kpi-value sage">{k ? `${k.u_rev_min_ratio} U_in · Cp ${k.cp_recovery_max}` : '--'}</div>
                </div>
              </>
            )}

            {bench === 'pipe' && (
              <>
                <div className="kpi-card">
                  <div className="kpi-label"><Activity size={14} /> {t.kpis.pipeDp}</div>
                  <div className="kpi-value coral">{k ? `${k.dp_total_pa} Pa (${k.dp_dx_pam} Pa/m)` : '--'}</div>
                </div>
                <div className="kpi-card">
                  <div className="kpi-label"><BarChart3 size={14} /> {t.kpis.pipeF}</div>
                  <div className="kpi-value sage">{k ? `${k.darcy_f_cfd} vs ${k.darcy_f_analytic} · U/Um=${k.umax_umean_ratio}` : '--'}</div>
                </div>
              </>
            )}

            <div className="kpi-card">
              <div className="kpi-label"><Layers size={14} /> {t.kpis.cells}</div>
              <div className="kpi-value">{k ? `${Number(k.n_cells).toLocaleString()} hex` : '--'}</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-label">{curResult?.cached ? <Database size={14} /> : <Timer size={14} />} {t.kpis.time}</div>
              <div className={`kpi-value ${curResult?.cached ? 'sage' : 'gold'}`}>
                {tm ? `${tm.cfd_openfoam}s (${curResult?.cached ? 'KV' : 'Live'})` : '--'}
              </div>
            </div>
          </div>
        </header>

        {error && <div className="error-banner"><AlertCircle size={15} /><span>{error}</span></div>}

        <div className="viewer-wrapper">
          <BenchViewer
            lang={lang}
            bench={bench}
            result={curResult}
            field={field}
            palette={palette}
            showStreams={showStreams}
            showWall={showWall}
            showEdges={showEdges}
            playing={playing}
            onTogglePlay={() => setPlaying((v) => !v)}
            loading={loading}
          />
        </div>

        {showChart && curResult?.metrics && (
          <div className="charts-dock">
            <BenchCharts lang={lang} metrics={curResult.metrics} />
          </div>
        )}

        {curCached.length > 0 && (
          <footer className="presets-strip">
            <span className="strip-label"><Database size={13} /> {t.cachedStrip}</span>
            <div className="strip-items">
              {curCached.map((c) => {
                const label =
                  bench === 'cavity'
                    ? `Re=${c.params.re} · Ghia RMSE=${c.kpis.ghia_rmse}`
                    : bench === 'step'
                      ? `Re=${c.params.re} · xR/H=${c.kpis.reattachment_xr_h}`
                      : `D=${c.params.diameter_mm}mm · U=${c.params.velocity_ms}m/s · ΔP=${c.kpis.dp_total_pa}Pa`;
                return (
                  <button
                    key={c.case_key}
                    type="button"
                    className={`case-chip ${c.case_key === curResult?.case_key ? 'active' : ''}`}
                    onClick={() => {
                      setParamsMap((prev) => ({ ...prev, [bench]: c.params }));
                      setResults((prev) => ({ ...prev, [bench]: makeCachedResp(bench, c) }));
                    }}
                  >
                    {label}
                  </button>
                );
              })}
            </div>
          </footer>
        )}
      </main>
    </div>
  );
};
