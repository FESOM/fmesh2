# fmesh2

Visual, jigsaw-based mesh generator for [FESOM2](https://fesom.de). Successor to
[fmesh](https://github.com/FESOM/fmesh): draw refinement regions on an interactive
globe, generate meshes with [JIGSAW](https://github.com/dengwirda/jigsaw), inspect
results at 10M+ node scale in the browser, locally refine and hand-edit existing meshes.

## Feature tour

- **Viewer** — open any FESOM2 mesh on a 3D globe or Plate Carrée map
  (animated morph, `[G]`); resolution/depth coloring, wireframe, coastline
  overlay, GPU-picked node inspection (hover + click-to-pin). NG5 scale
  (7.4M nodes, 22M-edge wireframe) navigates at 60 fps; warm reopen is
  instant thanks to a binary buffer cache.
- **Generation** — an ordered stack of resolution layers (base, mercator,
  coastline distance, bathymetry, drawn regions, imported lon/lat fields)
  composed live with an on-globe hfun preview (`[H]`) and node estimates;
  jigsaw + vectorized post-processing produce a ready-to-run FESOM2 mesh,
  visually or headless (`fmesh2 build`). Verified fmesh-parity.
- **Local refinement** (`[V]`) — draw a polygon on a *loaded* mesh, set a
  target resolution, and re-mesh only that cavity: the boundary ring stays
  bit-frozen (watertight seam), every outside node keeps its exact
  lon/lat/depth, and an `old_id,new_id` mapping table is exported. A/B
  compare (`[B]`) flips between original and refined mesh in place.
- **Node editing** (`[E]`) — select nodes by click, lasso, or radius brush;
  drag the selection in the tangent plane with live preview, or Laplacian-
  smooth it (coastline anchored). Ops that would invert elements are
  rejected and the offending triangles flash red. Every op lands in a JSONL
  journal with full undo/redo (⌘Z/⇧⌘Z); save-as writes the FESOM2 triplet
  plus `edits.jsonl`, and replaying the journal on the pristine mesh
  reproduces the edited mesh bit-exact.

## Install (development)

The jigsaw binary comes from conda-forge, so a conda/micromamba env is the supported route:

```bash
micromamba create -n meshgen -c conda-forge python=3.12 numpy scipy pandas xarray \
    netcdf4 shapely pyproj pyyaml fastapi uvicorn jigsaw jigsawpy pytest httpx pytest-timeout
micromamba run -n meshgen python -m pip install -e . --no-deps
```

Frontend (dev only, users get a prebuilt bundle):

```bash
npm --prefix frontend install
make frontend-build
```

## Use

```bash
fmesh2 /path/to/mesh_dir        # open mesh(es) in the browser viewer
fmesh2 build project.yaml       # headless mesh generation (reproducible)
fmesh2 convert MESH_DIR OUT     # FESOM2 ASCII -> binary viewer buffers
```

A FESOM2 mesh directory contains `nod2d.out`, `elem2d.out`, `aux3d.out`.

## Generating meshes (M2)

A mesh is described by a relocatable project directory with a `project.yaml`
holding an ordered stack of resolution layers (base, mercator, coastline
distance, bathymetry, drawn/KML region polygons, imported fields) — see
`docs/project-format.md`. Work visually or headless:

- **Viewer**: open a project in the panel (or `?project=/path` URL), edit
  layers with live hfun preview ([H]) and node estimates, draw refinement
  regions directly on the globe ([X]), and launch builds with streamed
  progress.
- **CLI**: `fmesh2 build project.yaml` runs compose → jigsaw → postprocess
  (land cut, loose-element cleanup, lake removal, depths, coast flags) and
  writes the FESOM2 triplet to the project's `out/`. The raw jigsaw mesh is
  cached by hfun hash; `--only-postprocess` reuses it for bit-reproducible
  reruns.

Coming from fmesh? `docs/migration-from-fmesh.md` maps every `configure.yaml`
knob; `examples/fmesh-parity/` is a fully migrated config. Parity vs legacy
fmesh is verified against golden fixtures (`docs/acceptance/m2.md`: hfun
median diff 0.000%, element count +1.9%).

## Editing existing meshes (M3/M4)

Refinement and hand-editing work on any opened mesh — no project needed.

**Deep refinements are iterative.** The boundary of a refined region stays
bit-frozen, so the resolution must transition from the surrounding mesh to
your target over a band of ~3× the local edge length. Inside a polygon
small relative to that band a very fine target is unreachable — the panel
warns and reports the finest reachable size. To go e.g. 100 km → 1 km,
either draw a region much larger than the transition band, or refine in
steps (100 → 30 → 10 → 3 → 1 km): each pass makes the seam finer, which
shrinks the next pass's band.
Both are journaled/reproducible: refinements cache the raw jigsaw cavity
(bit-identical reruns), and hand edits replay from `edits.jsonl`
(`fmesh2.edit.journal.apply_journal`). Acceptance walkthroughs with figures:
`docs/acceptance/m3.md` (refine + A/B) and `docs/acceptance/m4.md`
(full refine-then-edit workflow on the Denmark Strait).

## Development

- Implementation plan (completed): `docs/plans/completed/20260710-fmesh2.md`;
  per-milestone acceptance records: `docs/acceptance/`
- Contributor/agent notes: `CLAUDE.md`
- Render spike that validated the architecture: `experiments/spike_render/`
- Tests: `make test` (Python), `make frontend-test` (vitest), `make e2e`
  (Playwright smoke; once: `make e2e-setup`)
- Golden fixtures from legacy fmesh: `tests/fixtures/fmesh_golden/`
  (capture procedure: `experiments/fmesh_golden_capture/`)
