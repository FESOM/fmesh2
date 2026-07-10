# Next session — start Milestone 2 (generation pipeline)

Paste-ready prompt:

---

Continue implementing fmesh2 (visual jigsaw-based FESOM2 mesh generator) at
`/Users/nkolduno/PYTHON/mesh_fesom/fmesh2`.

Read `docs/plans/20260710-fmesh2.md` first — it is the source of truth and its
checkboxes are current. Milestone 1 (viewer) is complete and user-accepted,
tagged `v0.1.0`. Continue with **Milestone 2: Task 11 (project model)** and work
through Task 19, one task at a time, updating plan checkboxes and committing per
task. Tests must pass before moving to the next task
(`micromamba run -n meshgen python -m pytest` and, for frontend tasks,
`npm --prefix frontend run test` + `npm --prefix frontend run build`).

Key facts:
- Python env: micromamba `meshgen` (py3.12; numpy/scipy/pandas/xarray/netcdf4/
  shapely/pyproj/pyyaml/fastapi/uvicorn/jigsaw/jigsawpy/pytest/httpx installed).
  Editable install already done. Node v26 + npm for the Vite/three.js frontend.
- Reference meshes: `/Users/nkolduno/PYTHON/mesh_fesom/{core2,NG5_mesh}`;
  legacy fmesh (algorithm reference to port/vectorize): `../fmesh/fmesh.py`,
  `../fmesh/utils.py`, `../fmesh/configure.yaml`.
- Early in M2 (flagged in Task 19, do it before it blocks): create the legacy
  fmesh env (py<=3.9, `../fmesh/requirements.yml`), run fmesh once, commit
  golden fixtures to `tests/fixtures/fmesh_golden/`. Note `../fmesh/kml/` only
  ships Baffin_Bay/Nares/Peabody KMLs and `../fmesh/topo/` is EMPTY — RTopo must
  be downloaded via `../fmesh/download_data.sh` (AWI nextcloud) for the golden
  run; the fmesh2 test suite itself must rely only on the synthetic topo fixture
  (Task 14).
- Serve for manual checks:
  `micromamba run -n meshgen fmesh2 serve <abs mesh paths> --no-browser --port 8017`
  — always absolute mesh paths, always as a managed background process (shell-`&`
  detached servers die when the tool call ends). Static files are re-read per
  request: rebuilding the frontend does NOT need a server restart.
- Lessons already learned: pandas `sep=r"\s+"` hits the fast C path; keep lon
  raw in core I/O (bit-exact roundtrip is tested and guaranteed); viewer-side
  lon normalization only in `viewer/buffers.py`; OrbitControls minDistance is
  target-relative — the zoom floor is a per-frame radial clamp
  (`viewMode.radialClampFactor`).

State: 35 Python + 19 frontend tests green; commits are one-per-task on `main`.

---

After M2, the M3 gate is Task 20 (jigsaw cavity-with-fixed-boundary spike)
before any local-refinement code.
