import React, { useCallback, useEffect, useRef, useState } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkXMLPolyDataReader from '@kitware/vtk.js/IO/XML/XMLPolyDataReader';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkPoints from '@kitware/vtk.js/Common/Core/Points';
import vtkCellArray from '@kitware/vtk.js/Common/Core/CellArray';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import { ColorMode, ScalarMode } from '@kitware/vtk.js/Rendering/Core/Mapper/Constants';
import { Box, Crosshair, Maximize2, Pause, Play } from 'lucide-react';
import { fetchVtpBuffer } from '../api';
import { TRANSLATIONS, type Lang } from '../i18n';
import type { BenchId, BenchSimResponse, FieldId, PaletteId } from '../types';

interface BenchViewerProps {
  lang: Lang;
  bench: BenchId;
  result: BenchSimResponse | null;
  field: FieldId;
  palette: PaletteId;
  showStreams: boolean;
  showWall: boolean;
  showEdges: boolean;
  playing: boolean;
  onTogglePlay: () => void;
  loading: boolean;
}

type RGB = [number, number, number];

const RAINBOW_STOPS: Array<[number, RGB]> = [
  [0.0, [20, 62, 204]],
  [0.24, [0, 176, 245]],
  [0.48, [34, 204, 102]],
  [0.72, [250, 212, 24]],
  [0.88, [249, 115, 22]],
  [1.0, [220, 28, 38]],
];

const THERMAL_STOPS: Array<[number, RGB]> = [
  [0.0, [26, 20, 56]],
  [0.25, [110, 28, 124]],
  [0.52, [214, 68, 54]],
  [0.78, [249, 152, 24]],
  [1.0, [254, 248, 178]],
];

const RAINBOW_CSS =
  'linear-gradient(90deg, #143ecc 0%, #00b0f5 24%, #22cc66 48%, #fad418 72%, #f97316 88%, #dc1c26 100%)';
const THERMAL_CSS =
  'linear-gradient(90deg, #1a1438 0%, #6e1c7c 25%, #d64436 52%, #f99818 78%, #fef8b2 100%)';

const STEPS = 54;
const SIDES = 5;

interface StreamLine {
  pts: Float32Array;
  u: Float32Array;
  nrm: Float32Array;
  bnm: Float32Array;
  phase: number;
  waves: number;
}

interface AnimStream {
  lines: StreamLine[];
  poly: ReturnType<typeof vtkPolyData.newInstance>;
  pts: Float32Array;
  cols: Uint8Array;
  colArr: ReturnType<typeof vtkDataArray.newInstance>;
}

function sampleStops(stops: Array<[number, RGB]>, tRaw: number): RGB {
  const t = Math.min(Math.max(tRaw, 0), 1);
  let s0 = stops[0];
  let s1 = stops[stops.length - 1];
  for (let i = 0; i < stops.length - 1; i++) {
    if (t >= stops[i][0] && t <= stops[i + 1][0]) {
      s0 = stops[i];
      s1 = stops[i + 1];
      break;
    }
  }
  const f = (t - s0[0]) / Math.max(s1[0] - s0[0], 1e-6);
  return [
    s0[1][0] + (s1[1][0] - s0[1][0]) * f,
    s0[1][1] + (s1[1][1] - s0[1][1]) * f,
    s0[1][2] + (s1[1][2] - s0[1][2]) * f,
  ];
}

function applyStops(lut: ReturnType<typeof vtkColorTransferFunction.newInstance>, stops: Array<[number, RGB]>, lo: number, hi: number) {
  lut.removeAllPoints();
  const span = Math.max(hi - lo, 1e-5);
  for (const [pos, [r, g, b]] of stops) {
    lut.addRGBPoint(lo + pos * span, r / 255, g / 255, b / 255);
  }
  lut.setMappingRange(lo, hi);
  lut.updateRange();
}

function parseVtp(buf: ArrayBuffer) {
  const r = vtkXMLPolyDataReader.newInstance();
  r.parseAsArrayBuffer(buf);
  return r.getOutputData(0);
}

function activeFieldRange(poly: ReturnType<typeof vtkPolyData.newInstance>, field: string): [number, number] {
  const arr = poly.getPointData().getArrayByName(field) || poly.getPointData().getArrayByName('U_mag');
  if (!arr) return [0.01, 1];
  const data = arr.getData() as Float32Array;
  const sorted = Array.from(data).sort((a, b) => a - b);
  const n = sorted.length;
  if (n < 10) return [sorted[0] ?? 0, sorted[n - 1] ?? 1];
  if (field === 'u_x' || field === 'u_y' || field === 'cp' || field === 'vort_z' || field === 'p_kin' || field === 'stream_fn') {
    const lo = sorted[Math.floor(n * 0.015)];
    const hi = sorted[Math.floor(n * 0.985)];
    return hi > lo ? [lo, hi] : [lo, lo + 1e-3];
  }
  const hi = Math.max(sorted[Math.floor(n * 0.99)], 1e-4);
  const thr = hi * 0.012;
  let lo = sorted[0];
  for (let i = 0; i < n; i++) {
    if (sorted[i] > thr) {
      lo = sorted[i];
      break;
    }
  }
  if (lo >= hi * 0.9) lo = hi * 0.04;
  return [lo, hi];
}

function buildAnimStreams(rawPoly: ReturnType<typeof parseVtp>, tubeRadius: number): AnimStream {
  const rawPts = rawPoly.getPoints().getData() as Float32Array | Float64Array;
  const rawLines = rawPoly.getLines().getData() as Uint32Array | Int32Array;
  const rawUArr = rawPoly.getPointData().getArrayByName('U_mag');
  const rawU = rawUArr ? (rawUArr.getData() as Float32Array | Float64Array) : null;

  const lines: StreamLine[] = [];
  let cursor = 0;
  let lc = 0;

  while (cursor < rawLines.length) {
    const n = rawLines[cursor++];
    if (n < 4) {
      cursor += n;
      continue;
    }
    const sCum = new Float32Array(n);
    for (let k = 1; k < n; k++) {
      const i0 = rawLines[cursor + k - 1];
      const i1 = rawLines[cursor + k];
      const ds = Math.hypot(
        rawPts[i1 * 3] - rawPts[i0 * 3],
        rawPts[i1 * 3 + 1] - rawPts[i0 * 3 + 1],
        rawPts[i1 * 3 + 2] - rawPts[i0 * 3 + 2]
      );
      sCum[k] = sCum[k - 1] + ds;
    }
    const tot = sCum[n - 1];
    if (tot <= 1e-5) {
      cursor += n;
      continue;
    }
    const pts = new Float32Array(STEPS * 3);
    const uVals = new Float32Array(STEPS);
    let seg = 0;
    for (let s = 0; s < STEPS; s++) {
      const tt = (s / (STEPS - 1)) * tot;
      while (seg < n - 2 && sCum[seg + 1] < tt) seg++;
      const f = Math.min(Math.max((tt - sCum[seg]) / Math.max(sCum[seg + 1] - sCum[seg], 1e-8), 0), 1);
      const i0 = rawLines[cursor + seg];
      const i1 = rawLines[cursor + seg + 1];
      pts[s * 3 + 0] = rawPts[i0 * 3 + 0] + (rawPts[i1 * 3 + 0] - rawPts[i0 * 3 + 0]) * f;
      pts[s * 3 + 1] = rawPts[i0 * 3 + 1] + (rawPts[i1 * 3 + 1] - rawPts[i0 * 3 + 1]) * f;
      pts[s * 3 + 2] = rawPts[i0 * 3 + 2] + (rawPts[i1 * 3 + 2] - rawPts[i0 * 3 + 2]) * f;
      const u0 = rawU ? rawU[i0] : 1.0;
      const u1 = rawU ? rawU[i1] : 1.0;
      uVals[s] = u0 + (u1 - u0) * f;
    }

    const nrm = new Float32Array(STEPS * 3);
    const bnm = new Float32Array(STEPS * 3);
    let nx = 0, ny = 0, nz = 1;
    for (let s = 0; s < STEPS; s++) {
      const sp = Math.max(0, s - 1);
      const sn = Math.min(STEPS - 1, s + 1);
      let tx = pts[sn * 3] - pts[sp * 3];
      let ty = pts[sn * 3 + 1] - pts[sp * 3 + 1];
      let tz = pts[sn * 3 + 2] - pts[sp * 3 + 2];
      const tl = Math.hypot(tx, ty, tz) || 1;
      tx /= tl; ty /= tl; tz /= tl;
      if (s === 0) {
        const uy = Math.abs(tz) > 0.9 ? 1 : 0;
        const uz = Math.abs(tz) > 0.9 ? 0 : 1;
        const d = uy * ty + uz * tz;
        nx = -d * tx; ny = uy - d * ty; nz = uz - d * tz;
      } else {
        const d = nx * tx + ny * ty + nz * tz;
        nx -= d * tx; ny -= d * ty; nz -= d * tz;
      }
      const nl = Math.hypot(nx, ny, nz) || 1;
      nx /= nl; ny /= nl; nz /= nl;
      nrm[s * 3] = nx; nrm[s * 3 + 1] = ny; nrm[s * 3 + 2] = nz;
      bnm[s * 3] = ty * nz - tz * ny;
      bnm[s * 3 + 1] = tz * nx - tx * nz;
      bnm[s * 3 + 2] = tx * ny - ty * nx;
    }
    const h = Math.sin((lc + 1) * 12.9898) * 43758.5453;
    lines.push({ pts, u: uVals, nrm, bnm, phase: h - Math.floor(h), waves: 2.4 + (lc % 3) * 0.25 });
    lc++;
    cursor += n;
  }

  const nL = lines.length;
  const vpl = STEPS * SIDES;
  const ptsBuf = new Float32Array(nL * vpl * 3);
  const colsBuf = new Uint8Array(nL * vpl * 4);
  const polys = new Uint32Array(nL * (STEPS - 1) * SIDES * 2 * 4);
  let pi = 0;
  for (let l = 0; l < nL; l++) {
    const bv = l * vpl;
    for (let s = 0; s < STEPS - 1; s++) {
      const r0 = bv + s * SIDES;
      const r1 = bv + (s + 1) * SIDES;
      for (let k = 0; k < SIDES; k++) {
        const kn = (k + 1) % SIDES;
        polys[pi++] = 3; polys[pi++] = r0 + k; polys[pi++] = r0 + kn; polys[pi++] = r1 + kn;
        polys[pi++] = 3; polys[pi++] = r0 + k; polys[pi++] = r1 + kn; polys[pi++] = r1 + k;
      }
    }
  }
  const ct = new Float32Array(SIDES);
  const stbl = new Float32Array(SIDES);
  for (let k = 0; k < SIDES; k++) {
    const a = (k / SIDES) * Math.PI * 2;
    ct[k] = Math.cos(a) * tubeRadius;
    stbl[k] = Math.sin(a) * tubeRadius;
  }
  let off = 0;
  for (const l of lines) {
    for (let s = 0; s < STEPS; s++) {
      const p3 = s * 3;
      for (let k = 0; k < SIDES; k++) {
        const c = ct[k], sn = stbl[k];
        ptsBuf[off++] = l.pts[p3] + l.nrm[p3] * c + l.bnm[p3] * sn;
        ptsBuf[off++] = l.pts[p3 + 1] + l.nrm[p3 + 1] * c + l.bnm[p3 + 1] * sn;
        ptsBuf[off++] = l.pts[p3 + 2] + l.nrm[p3 + 2] * c + l.bnm[p3 + 2] * sn;
      }
    }
  }

  const poly = vtkPolyData.newInstance();
  const vpts = vtkPoints.newInstance();
  vpts.setData(ptsBuf, 3);
  poly.setPoints(vpts);
  const vpol = vtkCellArray.newInstance();
  vpol.setData(polys);
  poly.setPolys(vpol);
  const colArr = vtkDataArray.newInstance({ name: 'StreamRGBA', values: colsBuf, numberOfComponents: 4 });
  poly.getPointData().setScalars(colArr);

  return { lines, poly, pts: ptsBuf, cols: colsBuf, colArr };
}

function updateStreamColors(st: AnimStream, tSec: number, uMin: number, uMax: number, pal: PaletteId) {
  const stops = pal === 'rainbow' ? RAINBOW_STOPS : THERMAL_STOPS;
  const span = Math.max(uMax - uMin, 1e-4);
  let off = 0;
  for (const l of st.lines) {
    for (let s = 0; s < STEPS; s++) {
      const tf = s / (STEPS - 1);
      const [br, bg, bb] = sampleStops(stops, (l.u[s] - uMin) / span);
      const rp = tf * l.waves - tSec * 0.55 + l.phase;
      const cyc = rp - Math.floor(rp);
      const pulse = cyc > 0.34 ? (cyc < 0.9 ? Math.pow((cyc - 0.34) / 0.56, 2) : 1 - ((cyc - 0.9) / 0.1) * 0.2) : 0;
      const edge = Math.min(tf * 12, (1 - tf) * 10, 1);
      const r = Math.min(255, (br + (255 - br) * pulse * 0.45) | 0);
      const g = Math.min(255, (bg + (255 - bg) * pulse * 0.45) | 0);
      const b = Math.min(255, (bb + (255 - bb) * pulse * 0.45) | 0);
      const a = Math.min(255, ((72 + pulse * 183) * edge) | 0);
      for (let k = 0; k < SIDES; k++) {
        st.cols[off++] = r; st.cols[off++] = g; st.cols[off++] = b; st.cols[off++] = a;
      }
    }
  }
  st.colArr.modified();
}

const vtpCache = new Map<string, ArrayBuffer>();
async function getVtp(url: string) {
  const hit = vtpCache.get(url);
  if (hit) return hit;
  const buf = await fetchVtpBuffer(url);
  vtpCache.set(url, buf);
  return buf;
}

export const BenchViewer: React.FC<BenchViewerProps> = ({
  lang,
  bench,
  result,
  field,
  palette,
  showStreams,
  showWall,
  showEdges,
  playing,
  onTogglePlay,
  loading,
}) => {
  const t = TRANSLATIONS[lang];
  const langRef = useRef(lang);
  langRef.current = lang;
  const containerRef = useRef<HTMLDivElement | null>(null);
  const clockRef = useRef(0);
  const stateRef = useRef({ palette, showStreams, playing });
  stateRef.current = { palette, showStreams, playing };

  const [meshStatus, setMeshStatus] = useState('');
  const [range, setRange] = useState<[number, number]>([0.05, 1.0]);

  const sceneRef = useRef<{
    genericRW: ReturnType<typeof vtkGenericRenderWindow.newInstance>;
    fieldActor: ReturnType<typeof vtkActor.newInstance>;
    fieldMapper: ReturnType<typeof vtkMapper.newInstance>;
    wallActor: ReturnType<typeof vtkActor.newInstance>;
    wallMapper: ReturnType<typeof vtkMapper.newInstance>;
    streamActor: ReturnType<typeof vtkActor.newInstance>;
    streamMapper: ReturnType<typeof vtkMapper.newInstance>;
    lut: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    wallLut: ReturnType<typeof vtkColorTransferFunction.newInstance>;
    fieldPoly: ReturnType<typeof vtkPolyData.newInstance> | null;
    animStream: AnimStream | null;
    uBounds: [number, number];
    loaded: boolean;
  } | null>(null);

  const setCam = useCallback((mode: 'fit' | 'iso' | 'focus', res: BenchSimResponse | null) => {
    const s = sceneRef.current;
    if (!s) return;
    const ren = s.genericRW.getRenderer();
    const cam = ren.getActiveCamera();
    const b = res?.bench ?? bench;

    if (b === 'cavity') {
      cam.setFocalPoint(0.5, 0.5, 0.05);
      if (mode === 'iso') {
        cam.setPosition(0.95, 0.15, 1.75);
      } else if (mode === 'focus') {
        cam.setPosition(0.5, 0.52, 1.35);
      } else {
        cam.setPosition(0.5, 0.5, 2.05);
      }
      cam.setViewUp(0, 1, 0);
    } else if (b === 'step') {
      if (mode === 'focus') {
        cam.setFocalPoint(3.4, 0.85, 0.05);
        cam.setPosition(3.4, 0.85, 7.6);
      } else if (mode === 'iso') {
        cam.setFocalPoint(6.5, 1.0, 0.05);
        cam.setPosition(6.5, -5.0, 16.5);
      } else {
        cam.setFocalPoint(7.0, 1.0, 0.05);
        cam.setPosition(7.0, 1.0, 19.5);
      }
      cam.setViewUp(0, 1, 0);
    } else {
      const L = Number(res?.metrics?.kpis?.pipe_length_m ?? 1.6);
      const cx = L * 0.45;
      if (mode === 'focus') {
        cam.setFocalPoint(L * 0.25, 0.0, 0.0);
        cam.setPosition(L * 0.25, -L * 0.28, L * 0.22);
      } else if (mode === 'iso') {
        cam.setFocalPoint(cx, 0.0, 0.0);
        cam.setPosition(cx - L * 0.3, -L * 0.55, L * 0.48);
      } else {
        cam.setFocalPoint(cx, 0.0, 0.0);
        cam.setPosition(cx, 0.0, L * 0.92);
      }
      cam.setViewUp(0, 1, 0);
    }
    ren.resetCameraClippingRange();
    s.genericRW.getRenderWindow().render();
  }, [bench]);

  const syncVisuals = useCallback(() => {
    const s = sceneRef.current;
    if (!s || !s.loaded || !s.fieldPoly) return;
    const avail = s.fieldPoly.getPointData().getArrayByName(field) ? field : 'U_mag';
    const [lo, hi] = activeFieldRange(s.fieldPoly, avail);
    setRange([lo, hi]);
    const stops = palette === 'rainbow' ? RAINBOW_STOPS : THERMAL_STOPS;
    applyStops(s.lut, stops, lo, hi);
    s.fieldMapper.setColorByArrayName(avail);
    s.fieldMapper.setScalarRange(lo, hi);
    s.fieldActor.getProperty().setEdgeVisibility(showEdges);
    s.wallActor.setVisibility(Boolean(result?.urls?.wall) && showWall);

    if (s.animStream) {
      updateStreamColors(s.animStream, clockRef.current, s.uBounds[0], s.uBounds[1], palette);
    }
    s.streamActor.setVisibility(showStreams);
    s.genericRW.getRenderWindow().render();
  }, [field, palette, showStreams, showWall, showEdges, result]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let disposed = false;

    const genericRW = vtkGenericRenderWindow.newInstance({ background: [0.102, 0.08, 0.065] });
    genericRW.setContainer(el);
    genericRW.resize();
    const renderer = genericRW.getRenderer();

    const lut = vtkColorTransferFunction.newInstance();
    applyStops(lut, RAINBOW_STOPS, 0.05, 1.0);
    const wallLut = vtkColorTransferFunction.newInstance();
    applyStops(wallLut, RAINBOW_STOPS, 0.0, 100.0);

    const fieldMapper = vtkMapper.newInstance();
    fieldMapper.setLookupTable(lut);
    fieldMapper.setColorMode(ColorMode.MAP_SCALARS);
    fieldMapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    fieldMapper.setScalarVisibility(true);
    const fieldActor = vtkActor.newInstance();
    fieldActor.setMapper(fieldMapper);
    fieldActor.setVisibility(false);
    fieldActor.getProperty().setLighting(false);
    fieldActor.getProperty().setEdgeColor(0.14, 0.1, 0.08);
    renderer.addActor(fieldActor);

    const wallMapper = vtkMapper.newInstance();
    wallMapper.setLookupTable(wallLut);
    wallMapper.setColorMode(ColorMode.MAP_SCALARS);
    wallMapper.setScalarMode(ScalarMode.USE_POINT_FIELD_DATA);
    wallMapper.setColorByArrayName('p_pa');
    wallMapper.setScalarVisibility(true);
    const wallActor = vtkActor.newInstance();
    wallActor.setMapper(wallMapper);
    wallActor.setVisibility(false);
    wallActor.getProperty().setOpacity(0.26);
    renderer.addActor(wallActor);

    const streamMapper = vtkMapper.newInstance();
    streamMapper.setColorMode(ColorMode.DIRECT_SCALARS);
    streamMapper.setScalarMode(ScalarMode.USE_POINT_DATA);
    streamMapper.setScalarVisibility(true);
    const streamActor = vtkActor.newInstance();
    streamActor.setMapper(streamMapper);
    streamActor.setVisibility(false);
    streamActor.getProperty().setLighting(false);
    renderer.addActor(streamActor);

    sceneRef.current = {
      genericRW, fieldActor, fieldMapper, wallActor, wallMapper, streamActor, streamMapper,
      lut, wallLut, fieldPoly: null, animStream: null, uBounds: [0.05, 1.0], loaded: false,
    };

    const ro = new ResizeObserver(() => { if (!disposed) genericRW.resize(); });
    ro.observe(el);

    let rafId = 0;
    let lastTs = performance.now();
    const tick = (ts: number) => {
      if (disposed) return;
      const dt = Math.min((ts - lastTs) / 1000, 0.05);
      lastTs = ts;
      const s = sceneRef.current;
      const st = stateRef.current;
      if (s && s.loaded && st.playing && st.showStreams && s.animStream) {
        clockRef.current += dt;
        updateStreamColors(s.animStream, clockRef.current, s.uBounds[0], s.uBounds[1], st.palette);
        s.genericRW.getRenderWindow().render();
      }
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);

    return () => {
      disposed = true;
      cancelAnimationFrame(rafId);
      ro.disconnect();
      try { genericRW.delete(); } catch { /* ignore */ }
      sceneRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!result || !sceneRef.current) return;
    let cancelled = false;
    (async () => {
      setMeshStatus(TRANSLATIONS[langRef.current].loadingVtp);
      try {
        const bufs = await Promise.all([
          getVtp(result.urls.field),
          getVtp(result.urls.streamlines),
          result.urls.wall ? getVtp(result.urls.wall) : Promise.resolve(null),
        ]);
        if (cancelled || !sceneRef.current) return;
        const s = sceneRef.current;
        const fieldPoly = parseVtp(bufs[0]);
        s.fieldPoly = fieldPoly;
        s.fieldMapper.setInputData(fieldPoly);
        s.fieldActor.setVisibility(true);

        if (bufs[2]) {
          const wallPoly = parseVtp(bufs[2]);
          const [wlo, whi] = activeFieldRange(wallPoly, 'p_pa');
          applyStops(s.wallLut, RAINBOW_STOPS, wlo, whi);
          s.wallMapper.setInputData(wallPoly);
          s.wallMapper.setScalarRange(wlo, whi);
        }

        const [uLo, uHi] = activeFieldRange(fieldPoly, 'U_mag');
        s.uBounds = [uLo, uHi];
        const radius =
          result.bench === 'cavity'
            ? 0.0045
            : result.bench === 'step'
              ? 0.032
              : Number(result.metrics.params.diameter_mm ?? 50) * 1e-3 * 0.025;
        const anim = buildAnimStreams(parseVtp(bufs[1]), radius);
        s.animStream = anim;
        s.streamMapper.setInputData(anim.poly);
        s.loaded = true;

        syncVisuals();
        setCam(result.bench === 'pipe' ? 'iso' : 'fit', result);
        setMeshStatus('');
      } catch (e) {
        if (!cancelled) setMeshStatus((e as Error).message);
      }
    })();
    return () => { cancelled = true; };
  }, [result, setCam, syncVisuals]);

  useEffect(() => {
    if (result && sceneRef.current?.loaded) syncVisuals();
  }, [result, field, palette, showStreams, showWall, showEdges, syncVisuals]);

  const palCss = palette === 'rainbow' ? RAINBOW_CSS : THERMAL_CSS;
  const dec = Math.abs(range[1] - range[0]) < 1 ? 3 : 2;

  return (
    <div className="viewer-shell">
      <div ref={containerRef} className="vtk-canvas" />
      <div className="viewer-toolbar">
        <button type="button" className={playing ? 'active' : ''} onClick={onTogglePlay}>
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {playing ? t.pause : t.play}
        </button>
        <button type="button" onClick={() => setCam('fit', result)}><Maximize2 size={14} /> {t.camFit}</button>
        <button type="button" onClick={() => setCam('iso', result)}><Box size={14} /> {t.camIso}</button>
        <button type="button" onClick={() => setCam('focus', result)}><Crosshair size={14} /> {t.camZoom}</button>
      </div>

      {result && (
        <div className="colorbars-stack">
          <div className="colorbar-card">
            <div className="colorbar-header">
              <span>{t.fields[field] ?? field}</span>
            </div>
            <div className="colorbar-strip" style={{ background: palCss }} />
            <div className="colorbar-ticks">
              {[0, 0.25, 0.5, 0.75, 1].map((f) => (
                <span key={f}>{(range[0] + (range[1] - range[0]) * f).toFixed(dec)}</span>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="viewer-hint">{t.hint}</div>

      {(loading || meshStatus || !result) && (
        <div className="viewer-overlay">
          {loading ? (
            <div className="spinner-card">
              <div className="spinner" />
              <h3>{t.loadingCloud}</h3>
              <p>{t.loadingCloudDesc}</p>
            </div>
          ) : meshStatus ? (
            <div className="spinner-card">
              <div className="spinner" />
              <h3>{meshStatus}</h3>
            </div>
          ) : (
            <div className="spinner-card"><h3>{t.emptyTitle}</h3></div>
          )}
        </div>
      )}
    </div>
  );
};
