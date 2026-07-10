# Legacy fmesh golden fixtures

Captured 2026-07-10 from a one-time run of legacy fmesh (py3.9 env
`fmesh_legacy`, fastkml 0.12) on `configure.yaml` in this directory —
see `experiments/fmesh_golden_capture/` for the exact procedure.
Task 19 parity tests compare against these **static** files; legacy fmesh is
never executed by the fmesh2 test suite.

| file | contents |
|------|----------|
| `configure.yaml` | the parity config the run used (all hfun stages enabled) |
| `hfun_field.npz` | `hfun` (361x721 km field), `latitudes`, `longitudes`, per-region inner/outer ring coordinates from fmesh's `from_kml` |
| `coast_points.npz` | smoothed coastline points fed to `refine_along_coastlines` (`min_length=100km`, `averaging=50km`): flat `lon`/`lat` + per-segment `lengths` |
| `nod2d.out`, `elem2d.out`, `aux3d.out` | the generated coarse FESOM2 mesh (24549 nodes / 44443 elements) |
| `jigsaw_raw_counts.txt` | raw jigsaw mesh size before the land cut (41753 verts / 83502 tris) |
| `run.log` | full stdout incl. the estimator printout: "estimated # of elements ... is 100925" |
