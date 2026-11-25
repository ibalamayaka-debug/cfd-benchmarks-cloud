import { bindings, defineConfig } from "cf/config";
import * as entrypoint from "./worker/src/index.ts" with { type: "cf-worker" };

export default defineConfig({
  worker: {
    name: "cfd-benchmarks-cloud",
    compatibilityDate: "2026-11-17",
    entrypoint,
    assets: {
      notFoundHandling: "single-page-application",
      runWorkerFirst: ["/api/*"],
    },
    env: {
      CFD_BENCH_KV: bindings.kv({ id: "a29146a0d225426f9c0baf7692a61d43" }),
      RATE_LIMIT_KV: bindings.kv({ id: "881e01edca014d0c8236b3c61688a937" }),
      COMPUTE_BASE_URL: bindings.text(
        "https://ibalamayaka-debug--cfd-benchmarks-openfoam-fastapi-app.us-east.modal.run"
      ),
      COMPUTE_AUTH_TOKEN: bindings.secret(),
      ALLOWED_ORIGIN: bindings.text("*"),
      MAX_RUNS_PER_IP_PER_HOUR: bindings.text("30"),
      ASSETS: bindings.assets(),
    },
  },
});
