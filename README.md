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
fmesh2 convert MESH_DIR OUT     # FESOM2 ASCII -> binary viewer buffers
```

A FESOM2 mesh directory contains `nod2d.out`, `elem2d.out`, `aux3d.out`.

## Development

- Implementation plan: `docs/plans/20260710-fmesh2.md`
- Render spike that validated the architecture: `experiments/spike_render/`
- Tests: `make test` (Python), `make frontend-test` (vitest)
