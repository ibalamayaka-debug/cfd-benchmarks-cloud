"""
Modal Serverless OpenFOAM v2512 Cloud Compute App for 3 Classic Industrial CFD Benchmarks:
  1. /simulate/cavity : Lid-Driven Cavity (Ghia et al. benchmark)
  2. /simulate/step   : Backward-Facing Step (Separation & Recirculation x_R/H)
  3. /simulate/pipe   : Turbulent Pipe Flow (Pressure Drop & Radial Velocity Profile)
"""

import os

import modal

app = modal.App("cfd-benchmarks-openfoam")

image = (
    modal.Image.from_registry("opencfd/openfoam-run:2512", add_python="3.11")
    .apt_install("libglu1-mesa", "libgl1", "libgomp1")
    .pip_install("numpy", "vtk", "fastapi[standard]")
    .add_local_file("compute/benchmark_solvers.py", "/app/benchmark_solvers.py")
)


@app.function(
    image=image,
    cpu=4.0,
    memory=4096,
    timeout=300,
    scaledown_window=30,
    secrets=[modal.Secret.from_name("cfd-bench-secrets")],
)
@modal.asgi_app()
def fastapi_app():
    import sys

    sys.path.insert(0, "/app")
    from benchmark_solvers import (
        CavityParams,
        PipeBenchParams,
        StepParams,
        run_cavity_benchmark,
        run_pipe_benchmark,
        run_step_benchmark,
    )
    from fastapi import Body, FastAPI, Header, HTTPException

    web_app = FastAPI(title="OpenFOAM v2512 Classic CFD Benchmarks Engine")

    def _auth(authorization: str | None):
        expected = os.environ.get("COMPUTE_AUTH_TOKEN", "")
        if expected and (not authorization or authorization != f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @web_app.get("/health")
    def health():
        return {"status": "ok", "engine": "openfoam2512", "benchmarks": ["cavity", "step", "pipe"]}

    @web_app.post("/simulate/cavity")
    def sim_cavity(payload: dict = Body(...), authorization: str | None = Header(default=None)):
        _auth(authorization)
        try:
            return run_cavity_benchmark(CavityParams(re=int(payload.get("re", 1000))))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @web_app.post("/simulate/step")
    def sim_step(payload: dict = Body(...), authorization: str | None = Header(default=None)):
        _auth(authorization)
        try:
            return run_step_benchmark(StepParams(re=int(payload.get("re", 37500))))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    @web_app.post("/simulate/pipe")
    def sim_pipe(payload: dict = Body(...), authorization: str | None = Header(default=None)):
        _auth(authorization)
        try:
            return run_pipe_benchmark(
                PipeBenchParams(
                    diameter_mm=float(payload.get("diameter_mm", 50.0)),
                    velocity_ms=float(payload.get("velocity_ms", 2.0)),
                )
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    return web_app
