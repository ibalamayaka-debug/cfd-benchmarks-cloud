import React from 'react';
import { TRANSLATIONS, type Lang } from '../i18n';
import type { BenchMetrics } from '../types';

interface BenchChartsProps {
  lang: Lang;
  metrics: BenchMetrics;
}

function buildPath(xs: number[], ys: number[], xMin: number, xMax: number, yMin: number, yMax: number, w: number, h: number, pad: number): string {
  const sx = (x: number) => pad + ((x - xMin) / Math.max(xMax - xMin, 1e-9)) * (w - 2 * pad);
  const sy = (y: number) => h - pad - ((y - yMin) / Math.max(yMax - yMin, 1e-9)) * (h - 2 * pad);
  return xs
    .map((x, i) => `${i === 0 ? 'M' : 'L'}${sx(x).toFixed(1)},${sy(ys[i]).toFixed(1)}`)
    .join(' ');
}

export const BenchCharts: React.FC<BenchChartsProps> = ({ lang, metrics }) => {
  const t = TRANSLATIONS[lang].charts;
  const w = 460;
  const h = 164;
  const pad = 28;

  if (metrics.bench === 'cavity') {
    const cu = metrics.charts?.centerline_u;
    if (!cu) return null;
    const xMin = -0.45;
    const xMax = 1.05;
    const yMin = 0.0;
    const yMax = 1.0;
    const sx = (x: number) => pad + ((x - xMin) / (xMax - xMin)) * (w - 2 * pad);
    const sy = (y: number) => h - pad - ((y - yMin) / (yMax - yMin)) * (h - 2 * pad);
    const pathCfd = buildPath(cu.cfd_ux, cu.y, xMin, xMax, yMin, yMax, w, h, pad);

    return (
      <div className="charts-grid one">
        <div className="chart-card">
          <div className="chart-head">
            <strong>{t.cavityTitle}</strong>
            <span>{t.cavitySub}</span>
          </div>
          <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
            <line x1={sx(0)} y1={pad} x2={sx(0)} y2={h - pad} stroke="rgba(255,237,213,0.18)" strokeDasharray="3 3" />
            <line x1={pad} y1={sy(0.5)} x2={w - pad} y2={sy(0.5)} stroke="rgba(255,237,213,0.18)" strokeDasharray="3 3" />
            <path d={pathCfd} fill="none" stroke="#38bdf8" strokeWidth="2.4" />
            {(cu.ghia_y as number[]).map((gy, i) => (
              <circle key={i} cx={sx(cu.ghia_ux[i])} cy={sy(gy)} r="3.4" fill="#f97316" stroke="#16110d" strokeWidth="1" />
            ))}
            <text x={sx(0)} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="middle">u_x / U = 0</text>
            <text x={sx(1)} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="middle">1.0</text>
            <text x={sx(-0.4)} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="middle">-0.4</text>
            <text x={8} y={pad + 6} fontSize="9.5" fill="#cbb8a5">y=1</text>
            <text x={8} y={h - pad} fontSize="9.5" fill="#cbb8a5">y=0</text>
          </svg>
        </div>
      </div>
    );
  }

  if (metrics.bench === 'step') {
    const wc = metrics.charts?.wall_curve;
    if (!wc) return null;
    const xh = wc.x_h as number[];
    const uw = wc.u_wall_ratio as number[];
    const cp = wc.cp_wall as number[];
    const xr = Number(wc.xr_h ?? 6.2);
    const xMin = 0;
    const xMax = Math.max(...xh, 18);
    const yVals = [...uw, ...cp];
    const yMin = Math.min(-0.25, ...yVals);
    const yMax = Math.max(0.55, ...yVals);
    const sx = (x: number) => pad + ((x - xMin) / Math.max(xMax - xMin, 1)) * (w - 2 * pad);
    const sy = (y: number) => h - pad - ((y - yMin) / Math.max(yMax - yMin, 1e-6)) * (h - 2 * pad);
    const pathU = buildPath(xh, uw, xMin, xMax, yMin, yMax, w, h, pad);
    const pathCp = buildPath(xh, cp, xMin, xMax, yMin, yMax, w, h, pad);

    return (
      <div className="charts-grid one">
        <div className="chart-card">
          <div className="chart-head">
            <strong>{t.stepTitle}</strong>
            <span>{t.stepSub}</span>
          </div>
          <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
            <rect x={sx(0)} y={sy(0)} width={Math.max(0, sx(xr) - sx(0))} height={Math.max(0, sy(yMin) - sy(0))} fill="rgba(251, 113, 133, 0.14)" />
            <line x1={pad} y1={sy(0)} x2={w - pad} y2={sy(0)} stroke="rgba(255,237,213,0.28)" />
            <path d={pathCp} fill="none" stroke="#a78bfa" strokeWidth="1.6" strokeDasharray="4 3" />
            <path d={pathU} fill="none" stroke="#22d3ee" strokeWidth="2.4" />
            <circle cx={sx(xr)} cy={sy(0)} r="4.5" fill="#fbbf24" stroke="#16110d" strokeWidth="1.4" />
            <text x={sx(xr) + 6} y={sy(0) - 6} fontSize="10" fontWeight="700" fill="#fbbf24">
              xR/H = {xr.toFixed(2)}
            </text>
            <text x={sx(xr * 0.45)} y={sy(yMin * 0.65)} fontSize="9.5" fill="#fda4af" textAnchor="middle">
              Recirculation (u &lt; 0)
            </text>
            <text x={sx(0)} y={h - 8} fontSize="9.5" fill="#cbb8a5">0</text>
            <text x={sx(10)} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="middle">x/H = 10</text>
            <text x={sx(xMax)} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="end">{xMax.toFixed(0)}H</text>
          </svg>
        </div>
      </div>
    );
  }

  const rp = metrics.charts?.radial_profile;
  const ap = metrics.charts?.axial_pressure;
  if (!rp || !ap) return null;
  const rr = rp.r_over_r as number[];
  const pR = buildPath(rr, rp.u_norm_cfd, -1, 1, 0, 1.05, w, h, pad);
  const p7 = buildPath(rr, rp.u_power7, -1, 1, 0, 1.05, w, h, pad);
  const pL = buildPath(rr, rp.u_poiseuille, -1, 1, 0, 1.05, w, h, pad);

  const xd = ap.x_over_d as number[];
  const ppa = ap.p_pa as number[];
  const pMin = Math.min(...ppa);
  const pMax = Math.max(...ppa, pMin + 1);
  const pAx = buildPath(xd, ppa, Math.min(...xd), Math.max(...xd), pMin, pMax, w, h, pad);

  return (
    <div className="charts-grid two">
      <div className="chart-card">
        <div className="chart-head">
          <strong>{t.pipeRadTitle}</strong>
          <span>{t.pipeRadSub}</span>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
          <path d={pL} fill="none" stroke="rgba(251,113,133,0.75)" strokeWidth="1.5" strokeDasharray="2 3" />
          <path d={p7} fill="none" stroke="#fbbf24" strokeWidth="1.8" strokeDasharray="5 3" />
          <path d={pR} fill="none" stroke="#22d3ee" strokeWidth="2.5" />
          <text x={pad} y={h - 8} fontSize="9.5" fill="#cbb8a5">r/R = -1</text>
          <text x={w * 0.5} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="middle">0 (Center)</text>
          <text x={w - pad} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="end">+1</text>
        </svg>
      </div>
      <div className="chart-card">
        <div className="chart-head">
          <strong>{t.pipePressTitle}</strong>
          <span>{t.pipePressSub}</span>
        </div>
        <svg viewBox={`0 0 ${w} ${h}`} className="chart-svg">
          <path d={pAx} fill="none" stroke="#f97316" strokeWidth="2.5" />
          <text x={pad} y={pad + 8} fontSize="9.5" fill="#fed7aa">{pMax.toFixed(0)} Pa</text>
          <text x={w - pad} y={h - pad - 4} fontSize="9.5" fill="#fed7aa" textAnchor="end">{pMin.toFixed(0)} Pa</text>
          <text x={pad} y={h - 8} fontSize="9.5" fill="#cbb8a5">x/D = 0</text>
          <text x={w - pad} y={h - 8} fontSize="9.5" fill="#cbb8a5" textAnchor="end">x/D = 32</text>
        </svg>
      </div>
    </div>
  );
};
