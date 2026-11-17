# fmesh2 — working notes for Claude

Visual jigsaw-based FESOM2 mesh generator: browser viewer (three.js WebGL2),
resolution-layer generation, local cavity refinement, journaled node editing.
Delivered per `docs/plans/completed/20260710-fmesh2.md` (M1 viewer, M2
generation, M3 local refinement, M4 node editing; acceptance records in
`docs/acceptance/`).

## Environment & commands

- Python: micromamba env **meshgen** (py3.12; jigsaw binary from conda-forge).
  Editable install already done. Node v26 at `/opt/homebrew/bin`.
- Tests: `micromamba run -n meshgen python -m pytest` (repo root).
- Frontend: `npm --prefix frontend run test` (vitest),
  `npm --prefix frontend run build` (tsc strict + vite → `src/fmesh2/server/static/`,
  shipped in the wheel; no server restart needed after rebuild).
- e2e: `make e2e` (Playwright smoke against a real server + core2;
  once: `make e2e-setup`).
- Serve for manual checks with ABSOLUTE mesh paths:
  `fmesh2 /abs/path --no-browser --port 8017` (`/?mesh=<id|name>` picks).
- Reference data (siblings of the repo): `../core2` (127k nodes, fast),
  `../NG5_mesh` (7.4M nodes, perf only), `../fmesh/topo/RTopo-*.nc` (3.7 GB —
  **keep it**; parity + coastal-remesh tests skip without it).
  `FMESH2_TEST_DATA` overrides the data root. Legacy env `fmesh_legacy`
  exists only for re-capturing golden fixtures (`experiments/fmesh_golden_capture/`).

## Architecture map

- `src/fmesh2/core/` — `Mesh` dataclass, FESOM ASCII I/O, topology/metrics,
  stereographic projection (pure numpy).
- `src/fmesh2/data/` — coastlines (Natural Earth cache), lazy windowed topo
  reader (RTopo/GEBCO-style NetCDF).
- `src/fmesh2/hfun/` — resolution-field grid + ordered layer compose
  (base/mercator/coast/bathy/region/field), size estimator (fmesh parity).
- `src/fmesh2/generate/` — jigsaw driver (ellipsoid-grid hfun, subprocess in
  its own process group) + vectorized postprocess (land cut, loose elements,
  lake removal via connected_components, depths, coast flags).
- `src/fmesh2/edit/` — M3/M4: `cavity.py` (polygon selection → boundary
  loops), `local_remesh.py` (frozen-ring jigsaw cavity remesh + stitch),
  `ops.py` (move/smooth with inversion safeguards), `journal.py` (JSONL
  undo/redo log, deterministic replay, `mesh_hash`), `redepth.py` (topo
  re-sampling + smooth_seafloor.f90 smoothing, new nodes only or global;
  CLI `fmesh2 redepth`).
- `src/fmesh2/server/` — FastAPI app, `state.py` registry, `jobs.py`
  (worker-process jobs + WS progress), `edit.py` (per-mesh edit sessions).
- `src/fmesh2/viewer/buffers.py` — GPU buffer builder (disk-cached).
- `frontend/src/` — `scene.ts` orchestrates; `layers/` (mesh with shared
  morph uniform, coast, hfun overlay), `modes/` (draw/refine/edit),
  `panels/layers.ts`, `picking.ts` (GPU id pass), `selection.ts`, `geo.ts`
  (pure math — put testable logic here or in other pure modules).

## Conventions & invariants (violating these breaks tests on purpose)

- **Bit-exact FESOM ASCII I/O**: writer emits shortest round-trippable repr,
  reader uses `float_precision="round_trip"`. The M3 outside-untouched
  guarantee and M4 journal-replay hashes depend on it — do not "optimize".
- Core I/O never normalizes lon and `read_fesom` returns writable arrays
  (edit ops mutate in place; pandas CoW views are read-only).
- Depth sign follows the HOST mesh: core2 stores depths negative-down; the
  M2 generator writes positive-down. Edit/remesh paths match the host.
- jigsaw (euclidean cavity recipe, Task 20 gate): explicit `hfun_hmax=inf`,
  `geom_feat=False` for frozen rings, ring vertices as init mesh, no spaces
  in workdir paths, subprocess cwd pinned. See `docs/notes/cavity-spike.md`.
- Reproducibility anchors: raw-jigsaw npz caches (global + cavity) — jigsaw
  itself is not assumed bit-deterministic; edit journals replay by applying
  stored after-state (never recomputing smoothing).
- Viewer buffers: unit-sphere Y = north, plane = (lon_rad, lat_rad); dateline
  triangles get +360° duplicate vertices, `dupsrc` maps them to source nodes
  (cache format v2). Scalars computed on true topology, copied to duplicates.
- Frontend layers share one morph uniform so A/B swaps stay in lockstep;
  interaction modes (draw/refine/edit) are mutually exclusive and use
  capture-phase listeners + stopPropagation to preempt OrbitControls.
- Process: one commit per task, plan checkboxes ticked in the same commit;
  tests green before moving on (pytest + vitest + build; e2e for frontend
  tasks). Keep new tests in the suites listed above.
