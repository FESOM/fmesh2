# Task 20 gate — cavity remesh with a frozen boundary ring

**Verdict: GATE PASSED — jigsaw. Tasks 21–22 use jigsaw `euclidean-mesh`;
the Shewchuk `triangle` fallback is NOT needed** (no new dependency, no
non-commercial-license packaging implications for conda-forge).

Spike: `experiments/cavity_spike.py` (hard-asserting; exits non-zero on any
regression). Run:

```bash
micromamba run -n meshgen python experiments/cavity_spike.py [--plot] [--workdir DIR]
```

## The recipe (parameters Tasks 21–22 must use)

Cavity extracted from core2, boundary ring projected to a polygon-centered
stereographic plane (km), then jigsaw `euclidean-mesh` with:

- **geom**: `euclidean-mesh` PSLG — ring vertices as `vert2`, ring edges as
  `edge2` (plain zero IDtags; per-edge tags unnecessary);
- **init**: the same ring vertices as an init mesh (`opts.init_file`) —
  jigsaw keeps init vertices **bit-exact** (measured displacement 0.0);
  init edges are unnecessary (identical results with/without);
- **hfun**: `euclidean-grid` over the padded ring bbox, cell = target/3, with
  the seam-protecting ramp below;
- **opts**: `hfun_scal="absolute"`, **`hfun_hmax=inf`** (critical, see
  gotcha 1), `mesh_dims=2`, `optm_qlim=0.95`, `optm_iter=32`, `mesh_top1=True`,
  **`geom_feat=False`** (critical, see gotcha 2), default zip/div.

**Seam-protecting hfun ramp** (`ring_h` in the spike): for each evaluation
point, find the nearest ring *segment* (not vertex); with `L` = that
segment's length, `d` = distance to it:

```
h(x) = target + (1.2 * L - target) * clip(1 - d / (3 * L), 0, 1)
```

i.e. h at the ring equals 1.2× the local seam-edge length and relaxes to the
target over a **local** band of 3× that edge's length. Both localities are
load-bearing:

- distance to the nearest *vertex* sags below `L` at the midpoint of long
  edges → jigsaw splits them;
- a *global* band scaled to the median spacing decays too fast around the
  longest edges (coastal case: band/L_max = 1.35 failed with midpoint splits;
  ≥ 2.1 passed everywhere). Band factor 3 with headroom 1.2 passed all cases.

## Results (core2, target = median seam spacing / 3)

| case | cavity | ring | ring moved | boundary == ring | sizing med (edge/h) | deep med vs target | holes/outside tris |
|---|---|---|---|---|---|---|---|
| open ocean (mid-Atlantic box) | 241 elems | 49 nodes, 1 loop | 0.0 | yes | 0.99 [0.89, 1.09] | 1.00 | 0 / 0 |
| coastal (NW-Africa box, ring 47% coast edges) | 140 elems | 38 nodes, 1 loop | 0.0 | yes | 0.98 [0.69, 1.14] | n/a (band spans cavity) | 0 / 0 |
| island (Iceland enclosed) | 1421 elems | 201 nodes, 2 loops | 0.0 | yes | 0.99 [0.91, 1.07] | 1.00 | 0 / 0 |

Negative control: the same runs with `geom_feat=True` split 7–40 ring edges
(open ocean, island); every configuration with `geom_feat=False` passed.
Figures: `figures/cavity-{open_ocean,coastal,island}.png`.

- "Ring moved 0.0" is exact: jigsaw writes `.msh` with 17 significant digits,
  so float64 coordinates round-trip bit-exact through geom/init input and
  mesh output; every ring vertex reappears at KDTree distance exactly 0.0.
- Watertightness is topological: the output mesh's boundary-edge set equals
  the input ring-edge set exactly (no splits, no merges) — stitching in
  Task 22 can match seam nodes by position and rebuild triangles directly.
- Multi-loop PSLGs need no hole handling: jigsaw excludes the island
  interior by nesting parity (0 triangles inside the hole loop).

## jigsaw gotchas discovered (euclidean mode)

1. **`opts.hfun_hmax` must be set explicitly (use `inf`)** whenever a
   file-based hfun is used with `hfun_scal="absolute"` on euclidean
   geometry: left unset, jigsaw applies its *relative*-mode default (0.02),
   capping h at ~20 m and refining without bound (runaway to millions of
   points, minutes of CPU). The ellipsoid path in `generate/jigsaw_driver.py`
   is unaffected (the relative default never binds at ocean scales), but any
   euclidean driver in `edit/` must set it.
2. **`geom_feat=True` splits frozen edges at sharp ring corners.** Feature
   detection flags ring vertices with sharp interior angles (measured on
   lost edges: 48–118°) and protects them Ruppert-style by inserting points
   on the adjacent edges (new boundary points at distance ~1e-14 km from the
   ring line = exactly on it). With the ring supplied as init vertices,
   feature detection adds nothing — keep it off. (`mesh_top1=True` is
   harmless and kept for parity with the global driver.)
3. **Paths in `.jig` config files must not contain spaces** — jigsaw's
   parser reports `**parse error` for `GEOM_FILE=/tmp/dir with spaces/...`.
   Keep workdirs space-free (project cache dirs are; watch out in tests).
4. jigsaw still litters its cwd (`*.log`) in euclidean mode — keep pinning
   the subprocess `cwd` to the workdir.

## Notes for Task 21 (extraction machinery)

- Element selection: centroid-in-polygon in the stereographic plane, then
  largest edge-connected component, then iterative pinch cleanup (drop
  cavity elements incident to any ring node with >2 ring edges, re-take the
  largest component). On all three cases this converged in ≤1 iteration and
  produced clean simple loops.
- `topology._edge_keys` concatenates `[edge01 of all tris, edge12 of all,
  edge20 of all]` — element-id companion arrays must use `np.tile`, not
  `np.repeat` (a `repeat` bug here silently produced garbage components and
  single-triangle holes in early spike runs).
- Ring ordering: walk the 2-regular adjacency; multiple loops fall out
  naturally (outer seam + island coast rings).

## Notes for Task 22 (local hfun + stitch)

- The user-facing `transition_km` should act as a *minimum*: the effective
  per-segment band must stay ≥ ~2× (we use 3×) the local seam-edge length or
  jigsaw will split the seam. Guard-assert the output boundary equals the
  ring after every remesh (cheap, catches any regression loudly).
- Sizing tracks the hfun closely (median edge/h ≈ 0.98–1.00), so achieved-vs
  -target validation can use tight-ish tolerances (±20% on medians is
  comfortable).
- jigsaw runs are not assumed bit-deterministic: reproducibility of a refine
  operation is anchored the same way as global builds — cache the raw cavity
  mesh (npz keyed by ring+hfun hash) and keep postprocess/stitch pure numpy.
