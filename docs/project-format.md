# fmesh2 project format

A mesh-generation project is a directory containing a `project.yaml` plus
data subdirectories. All relative paths in the YAML resolve against the
project directory, so the folder is fully relocatable.

```
myproject/
    project.yaml    # everything below
    regions/        # *.geojson region polygons (drawn in the UI or imported from KML)
    fields/         # imported lon/lat resolution grids
    out/            # generated FESOM2 meshes (nod2d.out, elem2d.out, aux3d.out)
    cache/          # composed hfun grids, raw jigsaw meshes (the determinism anchor)
```

## Top level

```yaml
name: my-mesh                 # required, display name
topo: topo/RTopo-2.0.1.nc     # topography NetCDF (bathy layer, land cut, node depths)
grid:                         # hfun lat/lon grid density, fmesh convention:
  n_lat: 360                  #   n_lat intervals -> n_lat+1 latitudes (-90..90)
  n_lon: 720                  #   omit both to auto-suggest from the finest layer
layers: []                    # ordered resolution layers, see below
levels: [0.0, -5.0, ...]      # vertical levels, strictly decreasing, negative down
                              # (default: fmesh's proven 69-level set)
minimum_depth: 30             # m, positive; shallower ocean nodes are clamped to this
output_dir: out               # where generated meshes are written
seeds:                        # lon/lat points known to be ocean; the connected
  - [0.0, 0.0]                # component containing the first seed survives
                              # lake elimination
```

## Layers

`layers` is an **ordered** list; each entry is applied on top of the field
composed so far. Every layer shares these fields:

| field     | default | meaning                                                        |
|-----------|---------|----------------------------------------------------------------|
| `type`    | —       | one of `base`, `mercator`, `coast`, `bathy`, `region`, `field` |
| `name`    | `""`    | optional display name                                          |
| `enabled` | `true`  | disabled layers are skipped (UI toggle)                        |
| `mode`    | `min`   | `min` keeps the finer value, `replace` overwrites              |

Resolutions are km, depths are positive metres.

### `base` — constant background

```yaml
- type: base
  resolution: 300
```

### `mercator` — cos(latitude) scaling with polar caps

Matches fmesh's `mercator_resolution`: `resolution * cos(lat)` between the
boundaries, constant caps poleward of them.

```yaml
- type: mercator
  resolution: 300                  # km at the equator
  northern_boundary: 60            # default 60
  northern_lowest_resolution: 25   # km poleward of northern_boundary
  southern_boundary: -60           # default -60
  southern_lowest_resolution: 25
```

### `coast` — distance-to-coastline ramp

`min_resolution` at the coast, blending linearly into the surrounding field
over `max_distance`. Matches fmesh's `refine_along_coastlines`.

```yaml
- type: coast
  min_resolution: 10   # km at the coast
  max_distance: 100    # km, ramp width
  min_length: 100      # km, ignore coastlines shorter than this (default 100)
  averaging: 50        # km, coastline smoothing (default 50)
  source: null         # optional custom coastline file; null = bundled source
```

### `bathy` — depth ramp

`min_resolution` where depth ≤ `min_depth`, blending into the surrounding
field at `max_depth`; land untouched. Matches fmesh's `refine_over_bathymetry`.
Requires `topo`.

```yaml
- type: bathy
  min_resolution: 5
  min_depth: 50        # m (default 50)
  max_depth: 1500      # m (default 1500)
```

### `region` — polygon refinement

Target resolution inside a polygon. Two variants:

```yaml
# drawn polygon + transition band (primary UX)
- type: region
  path: regions/baffin.geojson
  resolution: 25
  transition_km: 150   # blend width outside the polygon

# fmesh-compatible two-ring: GeoJSON holds inner AND outer rings,
# the blend spans the gap between them
- type: region
  path: regions/nares.geojson
  resolution: 10       # no transition_km
```

KML region files (Google Earth, fmesh style) are converted to GeoJSON in
`regions/` on import.

### `field` — imported resolution grid

Lon/lat resolution field from NetCDF (or 3-column text), regridded onto the
hfun grid.

```yaml
- type: field
  path: fields/atlantic.nc
  variable: resolution   # optional if the file has a single data variable
  mode: replace          # often used with replace
```

## Python API

```python
from fmesh2.project import Project

proj = Project.load("myproject")        # dir or explicit project.yaml path
proj.layers[0].resolution = 250
proj.save()                             # back to myproject/project.yaml
proj.ensure_layout()                    # create regions/ fields/ out/ cache/
proj.topo_path                          # absolute Path, resolved vs project dir
```

Validation errors raise `fmesh2.project.ProjectError` with the offending
field spelled out; unknown layer `type` values list the valid ones.
