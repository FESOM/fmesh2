# Next session — start Milestone 4 (node editing)

Paste-ready prompt:

---

Continue implementing fmesh2 (visual jigsaw-based FESOM2 mesh generator) at
`/Users/nkolduno/PYTHON/mesh_fesom/fmesh2`.

Read `docs/plans/20260710-fmesh2.md` first — it is the source of truth and its
checkboxes are current. M1 (viewer, `v0.1.0`), M2 (generation, `v0.2.0`,
fmesh parity verified) and M3 (local refinement, Tasks 20–23, `v0.3.0` —
see `docs/acceptance/m3.md` and `docs/notes/cavity-spike.md`) are complete;
one commit per task on `main`.

Continue with **Milestone 4, Tasks 24–26** (node editing + release), one task
at a time: tests green before each next task
(`micromamba run -n meshgen python -m pytest`; for frontend tasks
`npm --prefix frontend run test` + `build`, `make e2e` for Playwright smoke),
tick plan checkboxes, commit per task.

Key facts:
- Python env: micromamba `meshgen` (py3.12, editable install done). Node v26.
- Reference meshes: `/Users/nkolduno/PYTHON/mesh_fesom/{core2,NG5_mesh}`.
  ⚠️ core2 stores aux3d depths **negative-down** (edit paths match the host
  mesh's sign convention; the M2 generator writes positive-down).
- Parity tests skip without `../fmesh/topo/RTopo-*.nc` (3.7 GB, downloaded —
  keep it); the coastal remesh test is gated the same way. Legacy env
  `fmesh_legacy` exists for golden-capture reruns.
- FESOM ASCII I/O is bit-exact for arbitrary float64 (shortest-repr writer +
  `float_precision="round_trip"` reader) — M4's journal-replay hash
  guarantee depends on it; don't "optimize" it away.
- M3 pieces to reuse in M4: `edit/cavity.py` (selection/loops),
  `edit/local_remesh.py` patterns, the shared-morph mesh swap +
  `installMesh` in `frontend/src/scene.ts` (in-place buffer updates for
  drags will extend `MeshLayer`), Task 17/23 `JobManager`, and the
  refine-panel UX (`modes/refine.ts`) as the template for edit mode.
- jigsaw gotchas (euclidean): set `hfun_hmax=inf` explicitly, keep
  `geom_feat=False` for frozen rings, no spaces in workdir paths, pin
  subprocess cwd (litters `.log`). See `docs/notes/cavity-spike.md`.
- Serve for manual checks with ABSOLUTE mesh paths
  (`fmesh2 <abs path> --no-browser --port 8017`); frontend rebuilds need no
  server restart.

State: 165 pytest + 28 vitest + 1 Playwright test green.

---

After M4: `v1.0.0`, move the plan to `docs/plans/completed/`.
