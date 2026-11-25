"""
Three Classic OpenFOAM v2512 Industrial CFD Benchmark Solvers for Modal Cloud:
  1. cavity : Lid-Driven Cavity (Re = 100..3200) -> flow familiarity & Ghia et al. centerline benchmark
  2. step   : Backward-Facing Step (Re_H = 400..37500) -> separation shear layer & recirculation x_R/H
  3. pipe   : Turbulent Pipe Flow (Re = 1,500..250,000) -> pressure drop dP/dx, Darcy f & radial velocity u(r)
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

OPENFOAM_BASHRC_CANDIDATES = [
    os.environ.get("OPENFOAM_BASHRC", ""),
    "/usr/lib/openfoam/openfoam2512/etc/bashrc",
    "/usr/lib/openfoam/openfoam2412/etc/bashrc",
    "/opt/openfoam2512/etc/bashrc",
]

N_PROCS = min(4, os.cpu_count() or 1)

# Ghia, Ghia & Shin (1982) benchmark u_x(x=0.5, y) values
GHIA_Y = [
    1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172, 0.5000,
    0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000,
]
GHIA_U: Dict[int, List[float]] = {
    100: [
        1.0000, 0.8412, 0.7887, 0.7372, 0.6872, 0.2315, 0.0033, -0.1364, -0.2058,
        -0.2109, -0.1566, -0.1015, -0.0643, -0.0478, -0.0419, -0.0372, 0.0000,
    ],
    400: [
        1.0000, 0.7584, 0.6844, 0.6176, 0.5589, 0.2909, 0.1626, 0.0214, -0.1148,
        -0.1712, -0.3273, -0.2430, -0.1461, -0.1034, -0.0927, -0.0819, 0.0000,
    ],
    1000: [
        1.0000, 0.6593, 0.5749, 0.5112, 0.4660, 0.3330, 0.1872, 0.0570, -0.0608,
        -0.1065, -0.2781, -0.3829, -0.2973, -0.2222, -0.2020, -0.1811, 0.0000,
    ],
    3200: [
        1.0000, 0.5324, 0.4829, 0.4655, 0.4610, 0.3468, 0.1979, 0.0716, -0.0427,
        -0.0866, -0.2443, -0.3432, -0.4193, -0.3783, -0.3534, -0.3241, 0.0000,
    ],
}


def _find_bashrc() -> Optional[str]:
    for p in OPENFOAM_BASHRC_CANDIDATES:
        if p and os.path.isfile(p):
            return p
    return None


def _run_foam(case_dir: Path, cmd: str, timeout: int = 240) -> str:
    bashrc = _find_bashrc()
    full_cmd = f"source '{bashrc}' >/dev/null 2>&1 || true; {cmd}" if bashrc else cmd
    proc = subprocess.run(
        ["bash", "-lc", full_cmd],
        cwd=str(case_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-28:])
        raise RuntimeError(f"OpenFOAM failed ({cmd}):\n{tail}")
    return proc.stdout


def _poly_to_ascii(poly) -> str:
    import vtk

    w = vtk.vtkXMLPolyDataWriter()
    w.SetWriteToOutputString(True)
    w.SetInputData(poly)
    w.SetDataModeToAscii()
    if w.Write() != 1:
        raise RuntimeError("Failed serializing VTP")
    return w.GetOutputString()


def _keep_only(poly, keep: Tuple[str, ...]):
    import vtk

    out = vtk.vtkPolyData()
    out.DeepCopy(poly)
    out.GetCellData().Initialize()
    out.GetFieldData().Initialize()
    pd = out.GetPointData()
    for n in [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]:
        if n not in keep:
            pd.RemoveArray(n)
    if keep and pd.GetArray(keep[0]) is not None:
        pd.SetActiveScalars(keep[0])
    return out


def _read_internal_vtu_points(case_dir: Path):
    import vtk

    vtk_dir = case_dir / "VTK"
    int_vtu = sorted(vtk_dir.rglob("internal.vtu"), key=lambda p: p.stat().st_mtime)
    if not int_vtu:
        raise RuntimeError(f"No internal.vtu found in {vtk_dir}")
    r = vtk.vtkXMLUnstructuredGridReader()
    r.SetFileName(str(int_vtu[-1]))
    r.Update()
    ug = r.GetOutput()
    c2p = vtk.vtkCellDataToPointData()
    c2p.SetInputData(ug)
    c2p.PassCellDataOff()
    c2p.Update()
    return ug, c2p.GetOutput()


def _sample_2d_image(ug_pt, nx: int, ny: int, x0: float, x1: float, y0: float, y1: float, z_mid: float):
    """Sample a 2D OpenFOAM case onto a 3-slice vtkImageData slab so 2D RK45 streamlines stay strictly in-plane."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    resampler = vtk.vtkResampleToImage()
    resampler.SetInputDataObject(ug_pt)
    resampler.UseInputBoundsOff()
    resampler.SetSamplingBounds(x0, x1, y0, y1, z_mid, z_mid)
    resampler.SetSamplingDimensions(nx, ny, 1)
    resampler.Update()
    sl = resampler.GetOutput()

    u_raw = vtk_to_numpy(sl.GetPointData().GetArray("U")).reshape(ny, nx, 3)
    p_raw = vtk_to_numpy(sl.GetPointData().GetArray("p")).reshape(ny, nx)
    k_arr = sl.GetPointData().GetArray("k")
    k_raw = vtk_to_numpy(k_arr).reshape(ny, nx) if k_arr is not None else np.zeros((ny, nx), dtype=np.float32)

    u_raw[:, :, 2] = 0.0
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    dx = (x1 - x0) / max(nx - 1, 1)
    dy = (y1 - y0) / max(ny - 1, 1)

    duy_dx = np.gradient(u_raw[:, :, 1], dx, axis=1)
    dux_dy = np.gradient(u_raw[:, :, 0], dy, axis=0)
    vort_z = (duy_dx - dux_dy).astype(np.float32)
    u_mag = np.linalg.norm(u_raw, axis=2).astype(np.float32)

    # Build 3-layer vtkImageData (z = z_mid - 0.05, z_mid, z_mid + 0.05) for foolproof 2D RK45 tracing
    img = vtk.vtkImageData()
    img.SetDimensions(nx, ny, 3)
    img.SetOrigin(x0, y0, z_mid - 0.05)
    img.SetSpacing(dx, dy, 0.05)

    u_3 = np.repeat(u_raw[None, :, :, :], 3, axis=0).reshape(-1, 3).astype(np.float32)
    u_mag_3 = np.repeat(u_mag[None, :, :], 3, axis=0).ravel()

    ipd = img.GetPointData()
    uv = numpy_to_vtk(u_3, deep=True)
    uv.SetName("U")
    ipd.SetVectors(uv)
    umv = numpy_to_vtk(u_mag_3, deep=True)
    umv.SetName("U_mag")
    ipd.AddArray(umv)
    ipd.SetActiveVectors("U")

    # Build crisp 2D surface vtkPolyData at z = z_mid
    sg = vtk.vtkStructuredGrid()
    sg.SetDimensions(nx, ny, 1)
    gx, gy = np.meshgrid(xs, ys)
    pts_np = np.column_stack([gx.ravel(), gy.ravel(), np.full(nx * ny, z_mid)]).astype(np.float32)
    pts_v = vtk.vtkPoints()
    pts_v.SetData(numpy_to_vtk(pts_np, deep=True))
    sg.SetPoints(pts_v)
    spd = sg.GetPointData()
    for name, arr in (
        ("U_mag", u_mag.ravel()),
        ("u_x", u_raw[:, :, 0].ravel().astype(np.float32)),
        ("u_y", u_raw[:, :, 1].ravel().astype(np.float32)),
        ("p_kin", p_raw.ravel().astype(np.float32)),
        ("vort_z", vort_z.ravel()),
        ("k_turb", k_raw.ravel().astype(np.float32)),
    ):
        va = numpy_to_vtk(arr, deep=True)
        va.SetName(name)
        spd.AddArray(va)

    gf = vtk.vtkStructuredGridGeometryFilter()
    gf.SetInputData(sg)
    gf.Update()

    return img, gf.GetOutput(), xs, ys, u_raw, p_raw, vort_z, k_raw


def _trace_streamlines(img, seed_points: List[Tuple[float, float, float]], u_array_name: str, max_prop: float, init_step: float, max_step: float, both_dirs: bool = True, max_pts_per_line: int = 70):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    seeds = vtk.vtkPoints()
    for x, y, z in seed_points:
        seeds.InsertNextPoint(float(x), float(y), float(z))
    seed_poly = vtk.vtkPolyData()
    seed_poly.SetPoints(seeds)

    tracer = vtk.vtkStreamTracer()
    tracer.SetInputData(img)
    tracer.SetSourceData(seed_poly)
    tracer.SetMaximumPropagation(max_prop)
    tracer.SetInitialIntegrationStep(init_step)
    tracer.SetMaximumIntegrationStep(max_step)
    if both_dirs:
        tracer.SetIntegrationDirectionToBoth()
    else:
        tracer.SetIntegrationDirectionToForward()
    tracer.SetIntegratorTypeToRungeKutta45()
    tracer.SetInputArrayToProcess(0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, "U")
    tracer.Update()

    raw_s = tracer.GetOutput()
    out_poly = vtk.vtkPolyData()
    if raw_s.GetNumberOfPoints() == 0 or raw_s.GetNumberOfLines() == 0:
        return out_poly

    s_pts = vtk_to_numpy(raw_s.GetPoints().GetData())
    u_sc = raw_s.GetPointData().GetArray(u_array_name)
    s_u = vtk_to_numpy(u_sc) if u_sc is not None else np.ones(len(s_pts), dtype=np.float32)

    out_pts, out_lines, out_u = vtk.vtkPoints(), vtk.vtkCellArray(), []
    lines = raw_s.GetLines()
    lines.InitTraversal()
    ids = vtk.vtkIdList()
    while lines.GetNextCell(ids):
        n = ids.GetNumberOfIds()
        if n < 5:
            continue
        step = max(1, n // max_pts_per_line)
        idxs = list(range(0, n, step))
        if idxs[-1] != n - 1:
            idxs.append(n - 1)
        new_ids = []
        for k in idxs:
            oid = ids.GetId(k)
            c = s_pts[oid]
            new_ids.append(out_pts.InsertNextPoint(round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)))
            out_u.append(round(float(s_u[oid]), 4))
        out_lines.InsertNextCell(len(new_ids))
        for nid in new_ids:
            out_lines.InsertCellPoint(nid)

    out_poly.SetPoints(out_pts)
    out_poly.SetLines(out_lines)
    if out_u:
        au = numpy_to_vtk(np.array(out_u, dtype=np.float32), deep=True)
        au.SetName("U_mag")
        out_poly.GetPointData().SetScalars(au)
    return out_poly


# ==============================================================================
# BENCHMARK 1: LID-DRIVEN CAVITY (OpenFOAM simpleFoam, laminar, Ghia et al. 1982)
# ==============================================================================

@dataclass(frozen=True)
class CavityParams:
    re: int = 1000  # 100 | 400 | 1000 | 3200

    def normalized(self) -> "CavityParams":
        allowed = (100, 400, 1000, 3200)
        re = min(allowed, key=lambda x: abs(x - int(self.re)))
        return CavityParams(re=re)

    def case_key(self) -> str:
        p = self.normalized()
        return hashlib.sha256(f"bench|cavity|re={p.re}".encode()).hexdigest()[:16]


def run_cavity_benchmark(raw_params: CavityParams) -> Dict[str, Any]:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    t0 = time.perf_counter()
    params = raw_params.normalized()
    case_key = params.case_key()
    nu = 1.0 / float(params.re)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"of_cavity_{case_key}_"))
    try:
        for sub in ("0", "constant", "system"):
            (tmp_dir / sub).mkdir(parents=True, exist_ok=True)

        (tmp_dir / "system" / "blockMeshDict").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object blockMeshDict;
}
scale 1;
vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 1 0.1)
    (0 1 0.1)
);
blocks
(
    hex (0 1 2 3 4 5 6 7) (90 90 1) simpleGrading (1 1 1)
);
edges ();
boundary
(
    movingWall { type wall; faces ((3 7 6 2)); }
    fixedWalls { type wall; faces ((0 4 7 3) (2 6 5 1) (1 5 4 0)); }
    frontAndBack { type empty; faces ((0 3 2 1) (4 5 6 7)); }
);
mergePatchPairs ();
""")

        (tmp_dir / "system" / "controlDict").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1800;
deltaT          1;
writeControl    timeStep;
writeInterval   1800;
purgeWrite      1;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")

        u_scheme = "bounded Gauss upwind" if params.re >= 3200 else "bounded Gauss linearUpwind grad(U)"
        (tmp_dir / "system" / "fvSchemes").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object fvSchemes;
}}
ddtSchemes {{ default steadyState; }}
gradSchemes {{ default Gauss linear; grad(U) cellLimited Gauss linear 1; }}
divSchemes
{{
    default none;
    div(phi,U) {u_scheme};
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}
laplacianSchemes {{ default Gauss linear orthogonal; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default orthogonal; }}
""")

        (tmp_dir / "system" / "fvSolution").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSolution;
}
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-7; relTol 0.05; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    pRefCell        0;
    pRefValue       0;
    residualControl
    {
        p 1e-6;
        U 1e-6;
    }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; }
}
""")

        (tmp_dir / "constant" / "transportProperties").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object transportProperties;
}}
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {nu:.8e};
""")

        (tmp_dir / "constant" / "turbulenceProperties").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object turbulenceProperties;
}
simulationType laminar;
""")

        (tmp_dir / "0" / "U").write_text("""FoamFile
{
    version 2.0; format ascii; class volVectorField; object U;
}
dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    movingWall   { type fixedValue; value uniform (1 0 0); }
    fixedWalls   { type noSlip; }
    frontAndBack { type empty; }
}
""")

        (tmp_dir / "0" / "p").write_text("""FoamFile
{
    version 2.0; format ascii; class volScalarField; object p;
}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    movingWall   { type zeroGradient; }
    fixedWalls   { type zeroGradient; }
    frontAndBack { type empty; }
}
""")

        t_cfd0 = time.perf_counter()
        _run_foam(tmp_dir, "blockMesh", timeout=60)
        _run_foam(tmp_dir, "simpleFoam", timeout=180)
        _run_foam(tmp_dir, "foamToVTK -latestTime -noFaceZones", timeout=90)
        cfd_sec = time.perf_counter() - t_cfd0

        ug, ug_pt = _read_internal_vtu_points(tmp_dir)
        img, surf_poly, xs, ys, u_raw, p_raw, vort_z, _ = _sample_2d_image(
            ug_pt, 91, 91, 0.0, 1.0, 0.0, 1.0, 0.05
        )

        # Add scaled velocity/pressure fields nicely
        pd = surf_poly.GetPointData()
        u_mag_flat = np.linalg.norm(u_raw, axis=2).ravel()
        stream_fn = np.cumsum(u_raw[:, :, 0], axis=0) * (ys[1] - ys[0])
        sf_v = numpy_to_vtk(stream_fn.ravel().astype(np.float32), deep=True)
        sf_v.SetName("stream_fn")
        pd.AddArray(sf_v)

        # Primary vortex center in interior [0.3..0.85, 0.3..0.85]
        i0, i1 = np.searchsorted(xs, 0.3), np.searchsorted(xs, 0.85)
        j0, j1 = np.searchsorted(ys, 0.3), np.searchsorted(ys, 0.85)
        sub_umag = np.linalg.norm(u_raw[j0:j1, i0:i1], axis=2)
        min_j, min_i = np.unravel_index(np.argmin(sub_umag), sub_umag.shape)
        vc_x = float(xs[i0 + min_i])
        vc_y = float(ys[j0 + min_j])

        # Ghia centerline extraction: u_x(x=0.5, y) & u_y(x, y=0.5)
        mid_i = int(np.argmin(np.abs(xs - 0.5)))
        mid_j = int(np.argmin(np.abs(ys - 0.5)))
        cfd_u_vert = u_raw[:, mid_i, 0]
        cfd_v_horiz = u_raw[mid_j, :, 1]

        ghia_u_ref = GHIA_U[params.re]
        cfd_at_ghia = np.interp(GHIA_Y, ys, cfd_u_vert)
        rmse = float(np.sqrt(np.mean((cfd_at_ghia - np.array(ghia_u_ref)) ** 2)))

        # Build crisp cavity streamlines covering core + Moffatt corner eddies
        seeds = []
        for y in (0.12, 0.24, 0.36, 0.48, 0.60, 0.72, 0.82, 0.90, 0.96):
            seeds.append((0.50, y, 0.05))
        # Secondary bottom-left & bottom-right eddies + upper-left eddy (for high Re)
        for sx, sy in ((0.06, 0.06), (0.11, 0.04), (0.94, 0.06), (0.87, 0.08), (0.05, 0.93)):
            seeds.append((sx, sy, 0.05))
        streams_poly = _trace_streamlines(img, seeds, "U_mag", max_prop=4.5, init_step=0.008, max_step=0.018, both_dirs=True)

        files_content = {
            "field.vtp": _poly_to_ascii(_keep_only(surf_poly, ("U_mag", "u_x", "u_y", "p_kin", "vort_z", "stream_fn"))),
            "streamlines.vtp": _poly_to_ascii(streams_poly),
        }

        total_sec = time.perf_counter() - t0
        metrics = {
            "case_key": case_key,
            "bench": "cavity",
            "params": asdict(params),
            "kpis": {
                "reynolds": params.re,
                "n_cells": int(ug.GetNumberOfCells()),
                "vortex_center_x": round(vc_x, 3),
                "vortex_center_y": round(vc_y, 3),
                "u_min_centerline": round(float(np.min(cfd_u_vert)), 3),
                "v_max_centerline": round(float(np.max(cfd_v_horiz)), 3),
                "v_min_centerline": round(float(np.min(cfd_v_horiz)), 3),
                "max_vorticity_z": round(float(np.percentile(np.abs(vort_z), 99.5)), 2),
                "ghia_rmse": round(rmse, 4),
            },
            "charts": {
                "centerline_u": {
                    "y": [round(float(y), 3) for y in ys[::2]],
                    "cfd_ux": [round(float(v), 4) for v in cfd_u_vert[::2]],
                    "ghia_y": GHIA_Y,
                    "ghia_ux": ghia_u_ref,
                },
                "centerline_v": {
                    "x": [round(float(x), 3) for x in xs[::2]],
                    "cfd_uy": [round(float(v), 4) for v in cfd_v_horiz[::2]],
                },
            },
            "timing_sec": {"cfd_openfoam": round(cfd_sec, 2), "total": round(total_sec, 2)},
            "file_sizes_kb": {k: round(len(v.encode("utf-8")) / 1024.0, 1) for k, v in files_content.items()},
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return {"metrics": metrics, "files_content": files_content}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ==============================================================================
# BENCHMARK 2: BACKWARD-FACING STEP (Separation & Recirculation Bubble x_R/H)
# ==============================================================================

@dataclass(frozen=True)
class StepParams:
    re: int = 37500  # 400 (Laminar) | 5000 | 28000 | 37500 (Driver & Seegmiller)

    def normalized(self) -> "StepParams":
        allowed = (400, 5000, 28000, 37500)
        re = min(allowed, key=lambda x: abs(x - int(self.re)))
        return StepParams(re=re)

    def case_key(self) -> str:
        p = self.normalized()
        return hashlib.sha256(f"bench|step|re={p.re}".encode()).hexdigest()[:16]


def run_step_benchmark(raw_params: StepParams) -> Dict[str, Any]:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    t0 = time.perf_counter()
    params = raw_params.normalized()
    case_key = params.case_key()
    u_in = 10.0
    H = 1.0
    rho = 1.225
    nu = (u_in * H) / float(params.re)
    is_turb = params.re >= 2300

    i_turb = 0.04
    k_in = max(1.5 * (u_in * i_turb) ** 2, 1e-4)
    omega_in = max(math.sqrt(k_in) / (0.09 ** 0.25 * 0.07 * H), 1.0)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"of_step_{case_key}_"))
    try:
        for sub in ("0", "constant", "system"):
            (tmp_dir / sub).mkdir(parents=True, exist_ok=True)

        # Conformal 3-block step mesh: inlet x:[-4,0] y:[1,2], step wake x:[0,22] y:[0,2], ER=2.0
        (tmp_dir / "system" / "blockMeshDict").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object blockMeshDict;
}
scale 1;
vertices
(
    (-4 1 0)   // 0
    (-4 2 0)   // 1
    ( 0 0 0)   // 2
    ( 0 1 0)   // 3
    ( 0 2 0)   // 4
    (22 0 0)   // 5
    (22 1 0)   // 6
    (22 2 0)   // 7

    (-4 1 0.1) // 8
    (-4 2 0.1) // 9
    ( 0 0 0.1) // 10
    ( 0 1 0.1) // 11
    ( 0 2 0.1) // 12
    (22 0 0.1) // 13
    (22 1 0.1) // 14
    (22 2 0.1) // 15
);
blocks
(
    // Block 0: inlet duct x:[-4,0], y:[1,2]
    hex (0 3 4 1  8 11 12 9) (36 28 1) simpleGrading (0.35 ((0.5 0.5 3.0) (0.5 0.5 0.333)) 1)
    // Block 1: upper downstream x:[0,22], y:[1,2]
    hex (3 6 7 4  11 14 15 12) (130 28 1) simpleGrading (4.0 ((0.5 0.5 3.0) (0.5 0.5 0.333)) 1)
    // Block 2: lower step wake x:[0,22], y:[0,1]
    hex (2 5 6 3  10 13 14 11) (130 28 1) simpleGrading (4.0 ((0.5 0.5 3.5) (0.5 0.5 0.285)) 1)
);
edges ();
boundary
(
    inlet      { type patch; faces ((0 1 9 8)); }
    outlet     { type patch; faces ((5 13 14 6) (6 14 15 7)); }
    upperWall  { type wall;  faces ((1 4 12 9) (4 7 15 12)); }
    lowerWall  { type wall;  faces ((0 8 11 3) (3 11 10 2) (2 10 13 5)); }
    frontAndBack { type empty; faces ((0 3 4 1) (3 6 7 4) (2 5 6 3) (8 9 12 11) (11 12 15 14) (10 11 14 13)); }
);
mergePatchPairs ();
""")

        (tmp_dir / "system" / "controlDict").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         800;
deltaT          1;
writeControl    timeStep;
writeInterval   800;
purgeWrite      1;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
""")

        (tmp_dir / "system" / "fvSchemes").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
wallDist { method meshWave; }
""")

        (tmp_dir / "system" / "fvSolution").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSolution;
}
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl { p 1e-5; U 1e-5; "(k|omega)" 1e-5; }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; k 0.6; omega 0.6; }
}
""")

        (tmp_dir / "constant" / "transportProperties").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object transportProperties;
}}
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {nu:.8e};
""")

        if is_turb:
            turb_text = """FoamFile
{
    version 2.0; format ascii; class dictionary; object turbulenceProperties;
}
simulationType RAS;
RAS { RASModel kOmegaSST; turbulence on; printCoeffs off; }
"""
        else:
            turb_text = """FoamFile
{
    version 2.0; format ascii; class dictionary; object turbulenceProperties;
}
simulationType laminar;
"""
        (tmp_dir / "constant" / "turbulenceProperties").write_text(turb_text)

        (tmp_dir / "0" / "U").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volVectorField; object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({u_in} 0 0);
boundaryField
{{
    inlet        {{ type fixedValue; value uniform ({u_in} 0 0); }}
    outlet       {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({u_in*0.5} 0 0); }}
    upperWall    {{ type noSlip; }}
    lowerWall    {{ type noSlip; }}
    frontAndBack {{ type empty; }}
}}
""")

        (tmp_dir / "0" / "p").write_text("""FoamFile
{
    version 2.0; format ascii; class volScalarField; object p;
}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet        { type zeroGradient; }
    outlet       { type fixedValue; value uniform 0; }
    upperWall    { type zeroGradient; }
    lowerWall    { type zeroGradient; }
    frontAndBack { type empty; }
}
""")

        if is_turb:
            (tmp_dir / "0" / "k").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; object k;
}}
dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_in:.6e};
boundaryField
{{
    inlet        {{ type fixedValue; value uniform {k_in:.6e}; }}
    outlet       {{ type inletOutlet; inletValue uniform {k_in:.6e}; value uniform {k_in:.6e}; }}
    upperWall    {{ type kqRWallFunction; value uniform {k_in:.6e}; }}
    lowerWall    {{ type kqRWallFunction; value uniform {k_in:.6e}; }}
    frontAndBack {{ type empty; }}
}}
""")
            (tmp_dir / "0" / "omega").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; object omega;
}}
dimensions [0 0 -1 0 0 0 0];
internalField uniform {omega_in:.6e};
boundaryField
{{
    inlet        {{ type fixedValue; value uniform {omega_in:.6e}; }}
    outlet       {{ type inletOutlet; inletValue uniform {omega_in:.6e}; value uniform {omega_in:.6e}; }}
    upperWall    {{ type omegaWallFunction; value uniform {omega_in:.6e}; }}
    lowerWall    {{ type omegaWallFunction; value uniform {omega_in:.6e}; }}
    frontAndBack {{ type empty; }}
}}
""")
            (tmp_dir / "0" / "nut").write_text("""FoamFile
{
    version 2.0; format ascii; class volScalarField; object nut;
}
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet        { type calculated; value uniform 0; }
    outlet       { type calculated; value uniform 0; }
    upperWall    { type nutkWallFunction; value uniform 0; }
    lowerWall    { type nutkWallFunction; value uniform 0; }
    frontAndBack { type empty; }
}
""")

        t_cfd0 = time.perf_counter()
        _run_foam(tmp_dir, "blockMesh", timeout=60)
        _run_foam(tmp_dir, "simpleFoam", timeout=180)
        _run_foam(tmp_dir, "foamToVTK -latestTime -noFaceZones", timeout=90)
        cfd_sec = time.perf_counter() - t_cfd0

        ug, ug_pt = _read_internal_vtu_points(tmp_dir)

        # Clip surface directly from actual step geometry via slice + sample 2D image for streamlines & curves
        # Direct GeometryFilter/Cutter on ug_pt at z=0.05 preserves the exact crisp step cutout [-4,0]x[0,1]!
        pl = vtk.vtkPlane()
        pl.SetOrigin(0.0, 0.0, 0.05)
        pl.SetNormal(0.0, 0.0, 1.0)
        cut = vtk.vtkCutter()
        cut.SetCutFunction(pl)
        cut.SetInputData(ug_pt)
        cut.Update()
        tf = vtk.vtkTriangleFilter()
        tf.SetInputConnection(cut.GetOutputPort())
        tf.Update()
        step_surf = tf.GetOutput()

        from vtk.util.numpy_support import vtk_to_numpy
        spd = step_surf.GetPointData()
        u_surf_np = vtk_to_numpy(spd.GetArray("U"))
        p_surf_np = vtk_to_numpy(spd.GetArray("p"))
        k_surf_arr = spd.GetArray("k")
        k_surf_np = vtk_to_numpy(k_surf_arr) if k_surf_arr is not None else np.zeros(len(p_surf_np), dtype=np.float32)

        q_in = 0.5 * u_in * u_in
        for name, arr in (
            ("U_mag", np.linalg.norm(u_surf_np, axis=1).astype(np.float32)),
            ("u_x", u_surf_np[:, 0].astype(np.float32)),
            ("u_y", u_surf_np[:, 1].astype(np.float32)),
            ("p_kin", (p_surf_np * rho).astype(np.float32)),
            ("cp", (p_surf_np / max(q_in, 1e-6)).astype(np.float32)),
            ("k_turb", k_surf_np.astype(np.float32)),
        ):
            va = numpy_to_vtk(arr, deep=True)
            va.SetName(name)
            spd.AddArray(va)

        # Sample downstream & full box for quantitative curves + streamlines
        img, _, xs, ys, u_raw, p_raw, vort_z, _ = _sample_2d_image(
            ug_pt, 180, 61, -3.9, 21.9, 0.0, 2.0, 0.05
        )
        # Mask upstream solid step region (-3.9 <= x < 0, y < 1.0) in u_raw to 0
        solid_x = xs < 0.0
        solid_y = ys < 1.0
        u_raw[np.ix_(solid_y, solid_x)] = 0.0

        # Extract near-bottom-wall velocity along y ~ 0.04H for x >= 0.1 to pinpoint exact reattachment x_R/H!
        j_wall = int(np.argmin(np.abs(ys - 0.04)))
        mask_ds = xs >= 0.05
        xs_ds = xs[mask_ds]
        u_wall_ds = u_raw[j_wall, mask_ds, 0] / u_in
        # Near-wall Cp along bottom wall (y ~ 0.04) relative to p(x=-0.5, y=1.5)
        i_ref = int(np.argmin(np.abs(xs - (-0.5))))
        j_ref = int(np.argmin(np.abs(ys - 1.5)))
        p_ref = float(p_raw[j_ref, i_ref])
        cp_wall_ds = (p_raw[j_wall, mask_ds] - p_ref) / q_in

        # Find primary reattachment x_R/H where u_wall crosses from <0 to >=0 after x=0.5
        xr_h = 6.2
        start_idx = int(np.searchsorted(xs_ds, 0.5))
        for idx in range(max(1, start_idx), len(xs_ds)):
            if u_wall_ds[idx - 1] < 0.0 <= u_wall_ds[idx]:
                denom = max(u_wall_ds[idx] - u_wall_ds[idx - 1], 1e-9)
                frac = -u_wall_ds[idx - 1] / denom
                xr_h = float(xs_ds[idx - 1] + frac * (xs_ds[idx] - xs_ds[idx - 1]))
                break

        u_rev_min = float(np.min(u_raw[:, mask_ds, 0]) / u_in)

        # Cross-sectional profiles at x/H = 1, 4, 8, 14
        profiles = {}
        for x_loc in (1.0, 4.0, 8.0, 14.0):
            ix = int(np.argmin(np.abs(xs - x_loc)))
            profiles[f"x{int(x_loc)}"] = [round(float(v / u_in), 4) for v in u_raw[:, ix, 0][::2]]

        # Streamlines: incoming freestream + closed recirculation bubble seeds!
        seeds = []
        for y in (1.06, 1.15, 1.28, 1.42, 1.58, 1.74, 1.88):
            seeds.append((-3.6, y, 0.05))
        # Explicit recirculation bubble seeds inside backflow & secondary corner eddy
        for sx in (0.6, 1.4, 2.4, 3.5, 4.8):
            for sy in (0.16, 0.42, 0.72):
                seeds.append((sx, sy, 0.05))
        streams_poly = _trace_streamlines(img, seeds, "U_mag", max_prop=26.0, init_step=0.04, max_step=0.10, both_dirs=True)

        files_content = {
            "field.vtp": _poly_to_ascii(_keep_only(step_surf, ("U_mag", "u_x", "u_y", "p_kin", "cp", "k_turb"))),
            "streamlines.vtp": _poly_to_ascii(streams_poly),
        }

        total_sec = time.perf_counter() - t0
        metrics = {
            "case_key": case_key,
            "bench": "step",
            "params": asdict(params),
            "kpis": {
                "reynolds": params.re,
                "regime": "Turbulent (k-ω SST)" if is_turb else "Laminar",
                "n_cells": int(ug.GetNumberOfCells()),
                "reattachment_xr_h": round(xr_h, 2),
                "u_rev_min_ratio": round(u_rev_min, 3),
                "cp_recovery_max": round(float(np.max(cp_wall_ds)), 3),
                "u_max_ms": round(float(np.max(np.linalg.norm(u_surf_np, axis=1))), 2),
            },
            "charts": {
                "wall_curve": {
                    "x_h": [round(float(x), 2) for x in xs_ds[::3]],
                    "u_wall_ratio": [round(float(v), 4) for v in u_wall_ds[::3]],
                    "cp_wall": [round(float(v), 4) for v in cp_wall_ds[::3]],
                    "xr_h": round(xr_h, 2),
                },
                "profiles": {
                    "y_h": [round(float(y), 3) for y in ys[::2]],
                    **profiles,
                },
            },
            "timing_sec": {"cfd_openfoam": round(cfd_sec, 2), "total": round(total_sec, 2)},
            "file_sizes_kb": {k: round(len(v.encode("utf-8")) / 1024.0, 1) for k, v in files_content.items()},
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return {"metrics": metrics, "files_content": files_content}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ==============================================================================
# BENCHMARK 3: TURBULENT PIPE FLOW (Pressure Drop dP/dx, Darcy f & u(r) Profile)
# ==============================================================================

@dataclass(frozen=True)
class PipeBenchParams:
    diameter_mm: float = 50.0   # 50 | 80 | 100
    velocity_ms: float = 2.0    # 0.03 (laminar) | 0.8 | 2.0 | 4.0

    def normalized(self) -> "PipeBenchParams":
        d = min((50.0, 80.0, 100.0), key=lambda x: abs(x - float(self.diameter_mm)))
        u = min((0.03, 0.8, 2.0, 4.0), key=lambda x: abs(x - float(self.velocity_ms)))
        return PipeBenchParams(diameter_mm=d, velocity_ms=u)

    def case_key(self) -> str:
        p = self.normalized()
        return hashlib.sha256(f"bench|pipe|d={p.diameter_mm:.0f}|u={p.velocity_ms:.2f}".encode()).hexdigest()[:16]


def run_pipe_benchmark(raw_params: PipeBenchParams) -> Dict[str, Any]:
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    t0 = time.perf_counter()
    params = raw_params.normalized()
    case_key = params.case_key()

    D = params.diameter_mm * 1e-3
    R = 0.5 * D
    L = 32.0 * D
    U = params.velocity_ms
    nu = 1.0e-6
    rho = 998.2
    Re = U * D / nu
    is_turb = Re >= 2300.0

    i_turb = 0.16 * (max(Re, 2300.0) ** (-0.125))
    k_in = max(1.5 * (U * i_turb) ** 2, 1e-6)
    omega_in = max(math.sqrt(k_in) / (0.09 ** 0.25 * 0.07 * D), 0.1)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"of_pipe_{case_key}_"))
    try:
        for sub in ("0", "constant", "system"):
            (tmp_dir / sub).mkdir(parents=True, exist_ok=True)

        # True 5-block O-grid straight circular pipe along +X from x=0 to x=L=32*D
        a = 0.36 * R
        c = R / math.sqrt(2.0)
        pts2d = [(-a, -a), (+a, -a), (+a, +a), (-a, +a), (-c, -c), (+c, -c), (+c, +c), (-c, +c)]
        v_lines = []
        for s_idx, xv in enumerate((0.0, L)):
            for idx, (yv, zv) in enumerate(pts2d):
                v_lines.append(f"    ({xv:.7g} {yv:.7g} {zv:.7g}) // {s_idx*8+idx}")

        arcs = [(4, 5, 0.0, -R), (5, 6, +R, 0.0), (6, 7, 0.0, +R), (7, 4, -R, 0.0)]
        e_lines = []
        for s_idx, xv in enumerate((0.0, L)):
            b = s_idx * 8
            for i0, i1, ym, zm in arcs:
                e_lines.append(f"    arc {b+i0} {b+i1} ({xv:.7g} {ym:.7g} {zm:.7g})")

        # 5 right-handed hex blocks along +x: (y->+y, z->+z, x->+x) -> (0 3 2 1) is -x? Wait:
        # Check cross((3)-(0), (1)-(0))? No, right-handed hex in OpenFOAM:
        # v0=(-a,-a), v1=(+a,-a), v2=(+a,+a), v3=(-a,+a) in (y,z):
        # e_x = v8-v0 = (+L,0,0), e_y = v3-v0 = (0,+2a,0), e_z = v1-v0 = (0,0,+2a) => Wait! (+x) x (+y) = +z?
        # Wait: (0->3) is +z=(-a,+a)-(-a,-a)=(0,+2a), (0->1) is +y=(+a,-a)-(-a,-a)=(+2a,0).
        # OpenFOAM hex(p0 p1 p2 p3 p4 p5 p6 p7): local x = p0->p1, local y = p0->p3, local z = p0->p4!
        # Let's check exact pipe_solver.py convention: in local (u, v, x):
        # If local x = u (along +y), local y = v (along +z), local z = +x (along pipe axis),
        # then u x v = (+y) x (+z) = +x! So det(u, v, +x) = +1 (right-handed!!)!
        # Let's verify core block (0 1 2 3  8 9 10 11):
        # 0->1 = (+2a, 0) in (y,z) = +y; 0->3 = (0, +2a) = +z; 0->8 = +x! (+y) x (+z) = +x > 0! RIGHT-HANDED!
        # Bottom ring (4 5 1 0): 4->5 = +y, 4->0 = (+y,+z) radially inward => (+y) x (+z) = +x > 0! RIGHT-HANDED!
        # Outer-right ring (1 5 6 2): 1->5 = radially outward (+y,-z), 1->2 = +z => outward_y x (+z) = +x! RIGHT-HANDED!
        # Top ring (3 2 6 7): 3->2 = +y, 3->7 = +z => +y x +z = +x! RIGHT-HANDED!
        # Left ring (4 0 3 7): 4->0 = +y, 4->7 = +z => +y x +z = +x! RIGHT-HANDED!
        nc, nr, nx = 7, 7, 84
        (tmp_dir / "system" / "blockMeshDict").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object blockMeshDict;
}}
scale 1;
vertices
(
{chr(10).join(v_lines)}
);
blocks
(
    hex (0 1 2 3  8  9 10 11) ({nc} {nc} {nx}) simpleGrading (1 1 1)
    hex (4 5 1 0  12 13 9  8) ({nc} {nr} {nx}) simpleGrading (1 4.0 1)
    hex (1 5 6 2  9 13 14 10) ({nr} {nc} {nx}) simpleGrading (0.25 1 1)
    hex (3 2 6 7  11 10 14 15) ({nc} {nr} {nx}) simpleGrading (1 0.25 1)
    hex (4 0 3 7  12 8 11 15) ({nr} {nc} {nx}) simpleGrading (4.0 1 1)
);
edges
(
{chr(10).join(e_lines)}
);
boundary
(
    inlet
    {{
        type patch;
        faces ( (0 3 2 1) (4 0 1 5) (1 2 6 5) (3 7 6 2) (4 7 3 0) );
    }}
    outlet
    {{
        type patch;
        faces ( (8 9 10 11) (12 13 9 8) (9 13 14 10) (11 10 14 15) (12 8 11 15) );
    }}
    walls
    {{
        type wall;
        faces ( (4 5 13 12) (5 6 14 13) (6 7 15 14) (7 4 12 15) );
    }}
);
mergePatchPairs ();
""")

        (tmp_dir / "system" / "controlDict").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object controlDict;
}
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         600;
deltaT          1;
writeControl    timeStep;
writeInterval   600;
purgeWrite      1;
writeFormat     binary;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
functions
{
    inletPressure
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        writeControl writeTime; writeFields false;
        regionType patch; name inlet; operation areaAverage; fields (p);
    }
    outletPressure
    {
        type surfaceFieldValue; libs (fieldFunctionObjects);
        writeControl writeTime; writeFields false;
        regionType patch; name outlet; operation areaAverage; fields (p);
    }
}
""")

        (tmp_dir / "system" / "decomposeParDict").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object decomposeParDict;
}}
numberOfSubdomains {N_PROCS};
method scotch;
""")

        (tmp_dir / "system" / "fvSchemes").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSchemes;
}
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; grad(U) cellLimited Gauss linear 1; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,k) bounded Gauss upwind;
    div(phi,omega) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
wallDist { method meshWave; }
""")

        (tmp_dir / "system" / "fvSolution").write_text("""FoamFile
{
    version 2.0; format ascii; class dictionary; object fvSolution;
}
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-6; relTol 0.05; }
    "(U|k|omega)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-6; relTol 0.1; }
}
SIMPLE
{
    nNonOrthogonalCorrectors 0;
    consistent yes;
    residualControl { p 1e-5; U 1e-5; "(k|omega)" 1e-5; }
}
relaxationFactors
{
    fields { p 0.3; }
    equations { U 0.7; k 0.7; omega 0.7; }
}
""")

        (tmp_dir / "constant" / "transportProperties").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class dictionary; object transportProperties;
}}
transportModel Newtonian;
nu [0 2 -1 0 0 0 0] {nu:.6e};
""")

        (tmp_dir / "constant" / "turbulenceProperties").write_text(
            """FoamFile
{
    version 2.0; format ascii; class dictionary; object turbulenceProperties;
}
simulationType RAS;
RAS { RASModel kOmegaSST; turbulence on; printCoeffs off; }
"""
            if is_turb
            else """FoamFile
{
    version 2.0; format ascii; class dictionary; object turbulenceProperties;
}
simulationType laminar;
"""
        )

        (tmp_dir / "0" / "U").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volVectorField; object U;
}}
dimensions [0 1 -1 0 0 0 0];
internalField uniform ({U:.6g} 0 0);
boundaryField
{{
    inlet  {{ type fixedValue; value uniform ({U:.6g} 0 0); }}
    outlet {{ type zeroGradient; }}
    walls  {{ type noSlip; }}
}}
""")

        (tmp_dir / "0" / "p").write_text("""FoamFile
{
    version 2.0; format ascii; class volScalarField; object p;
}
dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls  { type zeroGradient; }
}
""")

        if is_turb:
            (tmp_dir / "0" / "k").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; object k;
}}
dimensions [0 2 -2 0 0 0 0];
internalField uniform {k_in:.6e};
boundaryField
{{
    inlet  {{ type fixedValue; value uniform {k_in:.6e}; }}
    outlet {{ type zeroGradient; }}
    walls  {{ type kqRWallFunction; value uniform {k_in:.6e}; }}
}}
""")
            (tmp_dir / "0" / "omega").write_text(f"""FoamFile
{{
    version 2.0; format ascii; class volScalarField; object omega;
}}
dimensions [0 0 -1 0 0 0 0];
internalField uniform {omega_in:.6e};
boundaryField
{{
    inlet  {{ type fixedValue; value uniform {omega_in:.6e}; }}
    outlet {{ type zeroGradient; }}
    walls  {{ type omegaWallFunction; value uniform {omega_in:.6e}; }}
}}
""")
            (tmp_dir / "0" / "nut").write_text("""FoamFile
{
    version 2.0; format ascii; class volScalarField; object nut;
}
dimensions [0 2 -1 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet  { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    walls  { type nutkWallFunction; value uniform 0; }
}
""")

        t_cfd0 = time.perf_counter()
        _run_foam(tmp_dir, "blockMesh", timeout=60)
        if N_PROCS > 1:
            _run_foam(tmp_dir, "decomposePar -force", timeout=60)
            _run_foam(
                tmp_dir,
                f"OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 mpirun --allow-run-as-root --oversubscribe -np {N_PROCS} simpleFoam -parallel",
                timeout=180,
            )
            _run_foam(tmp_dir, "reconstructPar -latestTime", timeout=90)
        else:
            _run_foam(tmp_dir, "simpleFoam", timeout=180)
        _run_foam(tmp_dir, "foamToVTK -latestTime -noFaceZones", timeout=90)
        cfd_sec = time.perf_counter() - t_cfd0

        ug, ug_pt = _read_internal_vtu_points(tmp_dir)

        # Cut longitudinal center slice (z=0 plane -> shows axial pressure drop & radial profile!)
        pl = vtk.vtkPlane()
        pl.SetOrigin(0.0, 0.0, 0.0)
        pl.SetNormal(0.0, 0.0, 1.0)
        cut = vtk.vtkCutter()
        cut.SetCutFunction(pl)
        cut.SetInputData(ug_pt)
        cut.Update()
        tf = vtk.vtkTriangleFilter()
        tf.SetInputConnection(cut.GetOutputPort())
        tf.Update()
        slice_poly = tf.GetOutput()

        spd = slice_poly.GetPointData()
        u_sl = vtk_to_numpy(spd.GetArray("U"))
        p_sl = vtk_to_numpy(spd.GetArray("p"))
        k_sl_arr = spd.GetArray("k")
        k_sl = vtk_to_numpy(k_sl_arr) if k_sl_arr is not None else np.zeros(len(p_sl), dtype=np.float32)
        for name, arr in (
            ("U_mag", np.linalg.norm(u_sl, axis=1).astype(np.float32)),
            ("u_x", u_sl[:, 0].astype(np.float32)),
            ("p_pa", (p_sl * rho).astype(np.float32)),
            ("k_turb", k_sl.astype(np.float32)),
        ):
            va = numpy_to_vtk(arr, deep=True)
            va.SetName(name)
            spd.AddArray(va)

        # Pipe wall VTP (outer cylinder wall) for translucent 3D context
        walls_vtp = sorted((tmp_dir / "VTK").rglob("walls.vtp"), key=lambda p: p.stat().st_mtime)
        wall_poly = vtk.vtkPolyData()
        if walls_vtp:
            rw = vtk.vtkXMLPolyDataReader()
            rw.SetFileName(str(walls_vtp[-1]))
            rw.Update()
            c2p_w = vtk.vtkCellDataToPointData()
            c2p_w.SetInputData(rw.GetOutput())
            c2p_w.Update()
            wall_poly = c2p_w.GetOutput()
            wp = vtk_to_numpy(wall_poly.GetPointData().GetArray("p")) * rho
            wpv = numpy_to_vtk(wp.astype(np.float32), deep=True)
            wpv.SetName("p_pa")
            wall_poly.GetPointData().AddArray(wpv)

        # Sample radial velocity profile u(r) at fully-developed outlet x = 29*D and axial p(x) along centerline y=0,z=0
        def probe_line(pts_list):
            pp = vtk.vtkPolyData()
            vp = vtk.vtkPoints()
            for x, y, z in pts_list:
                vp.InsertNextPoint(float(x), float(y), float(z))
            pp.SetPoints(vp)
            pr = vtk.vtkProbeFilter()
            pr.SetInputData(pp)
            pr.SetSourceData(ug_pt)
            pr.Update()
            return pr.GetOutput().GetPointData()

        n_rad = 41
        r_norm = np.linspace(-0.98, 0.98, n_rad)
        rad_pd = probe_line([(29.0 * D, float(rn * R), 0.0) for rn in r_norm])
        u_rad = vtk_to_numpy(rad_pd.GetArray("U"))[:, 0]
        u_rad = np.maximum(u_rad, 0.0)
        u_max_dev = max(float(np.max(u_rad)), 1e-6)

        # Analytical velocity profiles normalized by U_max:
        # Turbulent 1/7th power law: (1 - |r/R|)^(1/7); Laminar Poiseuille: (1 - (r/R)^2)
        u_power7 = [(1.0 - abs(float(rn))) ** (1.0 / 7.0) for rn in r_norm]
        u_poiseuille = [1.0 - float(rn) ** 2 for rn in r_norm]

        n_ax = 33
        x_norm = np.linspace(0.02 * L, 0.98 * L, n_ax)
        ax_pd = probe_line([(float(xv), 0.0, 0.0) for xv in x_norm])
        p_ax_pa = vtk_to_numpy(ax_pd.GetArray("p")) * rho

        # Developed pressure gradient over x in [12D, 30D]
        m_dev = (x_norm >= 12.0 * D) & (x_norm <= 30.0 * D)
        slope, _ = np.polyfit(x_norm[m_dev], p_ax_pa[m_dev], 1)
        dp_dx = max(-float(slope), 0.0)
        dp_total = max(float(p_ax_pa[0] - p_ax_pa[-1]), 0.0)
        f_cfd = (dp_dx * D) / max(0.5 * rho * U * U, 1e-9)
        f_analytic = (0.3164 * (Re ** -0.25)) if is_turb else (64.0 / max(Re, 1.0))

        # 3D Streamlines along pipe interior (x=0.02L)
        img_res = vtk.vtkResampleToImage()
        img_res.SetInputDataObject(ug_pt)
        img_res.UseInputBoundsOff()
        img_res.SetSamplingBounds(0.0, L, -R * 1.02, R * 1.02, -R * 1.02, R * 1.02)
        img_res.SetSamplingDimensions(90, 26, 26)
        img_res.Update()
        img_3d = img_res.GetOutput()
        u_i = vtk_to_numpy(img_3d.GetPointData().GetArray("U"))
        um_i = numpy_to_vtk(np.linalg.norm(u_i, axis=1).astype(np.float32), deep=True)
        um_i.SetName("U_mag")
        img_3d.GetPointData().AddArray(um_i)
        img_3d.GetPointData().SetActiveVectors("U")

        seeds_3d = [(0.02 * L, 0.0, 0.0)]
        for r_frac in (0.32, 0.68, 0.88):
            for ang in range(0, 360, 60):
                rad = math.radians(ang)
                seeds_3d.append((0.02 * L, r_frac * R * math.cos(rad), r_frac * R * math.sin(rad)))
        streams_poly = _trace_streamlines(
            img_3d, seeds_3d, "U_mag", max_prop=L * 1.05, init_step=0.05 * D, max_step=0.25 * D, both_dirs=False
        )

        # Normalize spatial coordinates into [-1, 1] x-display or keep meters (VTK auto-frames cleanly)
        files_content = {
            "field.vtp": _poly_to_ascii(_keep_only(slice_poly, ("U_mag", "u_x", "p_pa", "k_turb"))),
            "wall.vtp": _poly_to_ascii(_keep_only(wall_poly, ("p_pa",))),
            "streamlines.vtp": _poly_to_ascii(streams_poly),
        }

        total_sec = time.perf_counter() - t0
        metrics = {
            "case_key": case_key,
            "bench": "pipe",
            "params": asdict(params),
            "kpis": {
                "reynolds": round(Re, 0),
                "regime": "Turbulent (k-ω SST)" if is_turb else "Laminar",
                "n_cells": int(ug.GetNumberOfCells()),
                "dp_total_pa": round(dp_total, 2),
                "dp_dx_pam": round(dp_dx, 2),
                "darcy_f_cfd": round(f_cfd, 4),
                "darcy_f_analytic": round(f_analytic, 4),
                "umax_umean_ratio": round(u_max_dev / max(U, 1e-6), 3),
                "u_max_ms": round(u_max_dev, 3),
                "pipe_length_m": round(L, 3),
            },
            "charts": {
                "radial_profile": {
                    "r_over_r": [round(float(rn), 3) for rn in r_norm],
                    "u_norm_cfd": [round(float(v / u_max_dev), 4) for v in u_rad],
                    "u_power7": [round(v, 4) for v in u_power7],
                    "u_poiseuille": [round(v, 4) for v in u_poiseuille],
                },
                "axial_pressure": {
                    "x_over_d": [round(float(xv / D), 2) for xv in x_norm],
                    "p_pa": [round(float(pv), 2) for pv in p_ax_pa],
                },
            },
            "timing_sec": {"cfd_openfoam": round(cfd_sec, 2), "total": round(total_sec, 2)},
            "file_sizes_kb": {k: round(len(v.encode("utf-8")) / 1024.0, 1) for k, v in files_content.items()},
            "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        return {"metrics": metrics, "files_content": files_content}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
