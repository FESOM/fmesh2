# Migrating from fmesh

fmesh2 reproduces every fmesh refinement stage (verified against golden
fixtures — see `docs/acceptance/m2.md`), replaces the KML + ParaView loop
with a live viewer, and runs the hfun composition ~1000× faster.

## Config mapping: `configure.yaml` → `project.yaml`

fmesh applied stages in a fixed order, each *overwriting* the field. In
fmesh2 the stages are an ordered **layer list**; use `mode: replace` on each
migrated layer to keep fmesh's overwrite semantics (the default `mode: min`
instead keeps the finest value of any layer — usually what you want for new
projects).

| fmesh `configure.yaml` | fmesh2 `project.yaml` |
|---|---|
| `topo_path` | `topo` |
| `n_latitudes`, `n_longitudes` | `grid.n_lat`, `grid.n_lon` (omit to auto-size) |
| `base_resolution` | layer `- type: base` / `resolution` |
| `mercator_resolution.do_mercator_refinement: true` | layer `- type: mercator, mode: replace` |
| `…norhtern_boundary` (sic) | `northern_boundary` |
| `…norhtern_lowest_resolution` | `northern_lowest_resolution` |
| `refine_along_coastlines.*` | layer `- type: coast, mode: replace` (`min_resolution`, `max_distance`, `min_length`, `averaging` keep their meanings) |
| `refine_over_bathymetry.*` | layer `- type: bathy, mode: replace` (`min_depth`, `min_resolution`, `max_depth`) |
| `regions: [{name, path, resolution}]` | one `- type: region` layer per region, `path` pointing at the imported GeoJSON |
| `minimum_depth` | `minimum_depth` |
| `levels` | `levels` (fmesh's 69-level set is the default — omit if unchanged) |
| `use_existed_refinements` / `use_existed_jigsaw_mesh` | automatic: hfun-keyed jigsaw cache in `cache/`; `fmesh2 build --only-postprocess` forces reuse |
| `plot_region` | gone — inspect live in the viewer (`fmesh2 serve`, [H] hfun preview) |

`examples/fmesh-parity/project.yaml` is a complete migrated config.

## KML regions

Import each Google Earth two-ring KML once:

```python
from fmesh2.io.kml import kml_to_geojson
kml_to_geojson("Baffin_Bay.kml", "myproject/regions/Baffin_Bay.geojson")
```

Inner/outer rings are detected by containment (file order does not matter)
and stored with `role` tags. New regions are better drawn directly in the
viewer ([X] draw region) — those use a single ring plus `transition_km`.

## Coastlines

fmesh used a preprocessed OSM coastline pickle; fmesh2 downloads the Natural
Earth 10m coastline on first use (cached in `~/.cache/fmesh2/`) and applies
the same `min_length` filter and `averaging` smoothing. Coastal ramps
therefore differ in local detail, not in structure. A custom coastline
GeoJSON can be passed via the coast layer's `source`.

## Running

```bash
fmesh2 build myproject/project.yaml     # headless, like `python fmesh.py`
fmesh2 serve mesh_out_dir               # inspect any FESOM2 mesh
fmesh2 --help
```

The build writes `nod2d.out`, `elem2d.out`, `aux3d.out` into the project's
`out/` directory. Node depths, minimum-depth clamping, land cut, loose-element
cleanup and lake removal follow fmesh's rules (`docs/acceptance/m2.md`
quantifies the parity).
