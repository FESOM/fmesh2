# fmesh2

Visual, jigsaw-based mesh generator for [FESOM2](https://fesom.de). Successor to
[fmesh](https://github.com/FESOM/fmesh): draw refinement regions on an interactive
globe, generate meshes with [JIGSAW](https://github.com/dengwirda/jigsaw), inspect
results at 10M+ node scale in the browser, locally refine and hand-edit existing meshes.

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

## Development

- Implementation plan: `docs/plans/20260710-fmesh2.md`
- Render spike that validated the architecture: `experiments/spike_render/`
- Tests: `make test` (Python), `make frontend-test` (vitest), `make e2e`
  (Playwright smoke; once: `make e2e-setup`)
- Golden fixtures from legacy fmesh: `tests/fixtures/fmesh_golden/`
  (capture procedure: `experiments/fmesh_golden_capture/`)
