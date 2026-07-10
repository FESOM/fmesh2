# Next session — start Milestone 3 (local refinement)

Paste-ready prompt:

---

Continue implementing fmesh2 (visual jigsaw-based FESOM2 mesh generator) at
`/Users/nkolduno/PYTHON/mesh_fesom/fmesh2`.

Read `docs/plans/20260710-fmesh2.md` first — it is the source of truth and its
checkboxes are current. M1 (viewer, `v0.1.0`) and M2 (generation, Tasks 11–19,
fmesh parity verified — see `docs/acceptance/m2.md`) are complete; repo is
tagged `v0.2.0` with one commit per task on `main`.

Continue with **Milestone 3, Task 20 — the ⚠️ GATE cavity spike**
(`experiments/cavity_spike.py`): on core2, extract a polygon cavity + boundary
ring, project to a local stereographic plane, and try jigsaw `euclidean-mesh`
with the ring as frozen geometry (`geom_feat`, `mesh_top1`, init mesh with
ring nodes) at 3× finer target. Hard assertions: ring vertices unmoved within
1e-12, interior sizing follows hfun. Repeat for a coastal polygon. Record the
verdict + parameters in `docs/notes/cavity-spike.md` — it decides whether
Tasks 21–22 use jigsaw or the Shewchuk `triangle` fallback (NOT in the meshgen
env, non-commercial license — record packaging implications if chosen).
Then Tasks 21–23, one task at a time: tests green before each next task
(`micromamba run -n meshgen python -m pytest`; for frontend tasks
`npm --prefix frontend run test` + `build`, `make e2e` for Playwright smoke),
tick plan checkboxes, commit per task.

Key facts:
- Python env: micromamba `meshgen` (py3.12, editable install done). Node v26.
- Reference meshes: `/Users/nkolduno/PYTHON/mesh_fesom/{core2,NG5_mesh}`.
- Parity tests (`tests/test_parity_golden.py`) skip without
  `../fmesh/topo/RTopo-*.nc` (3.7 GB, downloaded — keep it). Legacy env
  `fmesh_legacy` exists for golden-capture reruns (pins: `openssl>=3`,
  `fastkml=0.12`; capture harness: `experiments/fmesh_golden_capture/`).
- Serve for manual checks with ABSOLUTE mesh paths as a managed background
  process (`fmesh2 <abs path> --no-browser --port 8017`); frontend rebuilds
  need no server restart.
- M2 pieces to reuse in M3: `core/projection.py` (spherical stereographic +
  haversine, pure numpy), `hfun/region_layer.py` (ring loading/densify,
  point-in-polygon in the projected plane), `generate/postprocess.py`
  (land cut / loose elements / boundary coast flags — shared pieces for
  cavity re-land-cut), `generate/jigsaw_driver.py` (binary discovery, msh
  I/O patterns). The refine endpoint (Task 23) should reuse the Task 17
  `JobManager` and the draw-mode polygon UX from Task 18.
- jigsaw litters its cwd (`opts.log`) — pin subprocess cwd. xarray caches
  netCDF handles — tests writing multiple .nc files need unique names.

State: 141 pytest + 28 vitest + 1 Playwright test green.

---

After M3: M4 node editing (Tasks 24–26), then `v1.0.0`.
