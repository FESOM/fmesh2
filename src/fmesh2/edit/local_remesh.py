"""Local cavity remesh with a frozen seam.

Remeshes the interior of a `Cavity` at a new target resolution while keeping
every boundary-ring vertex bit-frozen, then stitches the result back into
the untouched outside mesh. jigsaw recipe and hfun shape are exactly the
Task 20 gate parameters (`docs/notes/cavity-spike.md`):

* geometry = ring vertices + edges, `geom_feat=False` (feature detection
  splits frozen edges at sharp corners), ring vertices also given as the
  init mesh (jigsaw preserves init vertices bit-exact);
* `hfun_scal="absolute"` with an explicit `hfun_hmax=inf` (unset, jigsaw
  applies its relative-mode default and refines without bound);
* hfun ramps from `RING_HEADROOM x` the local seam-edge length at the ring
  to the target over a per-segment band of `BAND_FACTOR x` that edge's
  length (narrower bands make jigsaw split the longest frozen edges) —
  SEAM segments only: coast segments border land, nothing stitches to
  them, so jigsaw may subdivide the coast polyline to meet the target.
  With an explicit transition width the ramp is gentle and the headroom
  is `WIDE_HEADROOM` (1.0) instead, so no ring of coarser elements is
  left along the seam.

Reproducibility follows the M2 contract: the raw jigsaw cavity is cached
(npz keyed by the exact inputs) and everything around it is pure numpy, so
same cache -> bit-identical stitched mesh.
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jigsawpy
import numpy as np
import shapely
from scipy.spatial import cKDTree

from fmesh2.core.mesh import Mesh
from fmesh2.core.projection import stereo_inverse
from fmesh2.core.topology import _edge_keys, boundary_nodes, unique_edges
from fmesh2.data.topo import Topo
from fmesh2.generate.jigsaw_driver import JigsawError, _find_jigsaw
from fmesh2.generate.postprocess import _neighbor_and_valence

from .cavity import Cavity

RING_HEADROOM = 1.2   # hfun at the ring = headroom x local seam-edge length
WIDE_HEADROOM = 1.0   # ... when an explicit transition_km makes the ramp gentle
BAND_FACTOR = 3.0     # per-segment transition band = factor x edge length
SEAM_MATCH_TOL_KM = 1e-6  # ring vertices must reappear this close (measured: 0.0)
MAX_HFUN_GRID = 4096  # per-axis cap on the local hfun grid


class RemeshError(RuntimeError):
    pass


@dataclass
class RemeshResult:
    """Stitched mesh, old->new node id mapping (-1 = removed), report."""

    mesh: Mesh
    mapping: np.ndarray
    report: dict


# --------------------------------------------------------------------- hfun

def seam_hfun_at(
    points_xy,
    cavity: Cavity,
    target,
    transition_km: float | None = None,
    headroom: float | None = None,
):
    """The seam-protecting ramp evaluated at plane points (km).

    `target` is a scalar or a per-point array. The band is per SEAM segment:
    `max(BAND_FACTOR * segment_length, transition_km)`. The ramp starts at
    `headroom` x the nearest seam segment's length; by default
    `seam_headroom(transition_km)`.

    Only seam segments (shared with the outside mesh) are protected — those
    edges must survive jigsaw untouched for the stitch. Coast segments are
    exempt: land borders nothing to stitch to, so jigsaw is free to
    subdivide the frozen coast polyline (identical shoreline, more
    vertices). Without this exemption, coast-bound regions — straits,
    shelves — could never be refined below ~the old coastline spacing.
    Returns (h, distance to the nearest seam segment; inf when the ring is
    all coast, e.g. a fully enclosed sea).
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    target = np.broadcast_to(
        np.asarray(target, dtype=np.float64), (len(points_xy),)
    )
    seam = ~cavity.edge_is_coast
    if not seam.any():
        return target.copy(), np.full(len(points_xy), np.inf)
    if headroom is None:
        headroom = seam_headroom(transition_km)
    seam_len = cavity.edge_len_km[seam]
    segs = shapely.linestrings(cavity.ring_xy[cavity.ring_edges[seam]])
    pgeo = shapely.points(points_xy)
    inp, near = shapely.STRtree(segs).query_nearest(pgeo, all_matches=False)
    d = np.empty(len(points_xy))
    hseam = np.empty(len(points_xy))
    band = np.empty(len(points_xy))
    d[inp] = shapely.distance(pgeo[inp], segs[near])
    hseam[inp] = headroom * seam_len[near]
    band[inp] = np.maximum(BAND_FACTOR * seam_len[near], transition_km or 0.0)
    return target + (hseam - target) * np.clip(1.0 - d / band, 0.0, 1.0), d


def seam_headroom(transition_km: float | None) -> float:
    """Headroom of the seam ramp.

    `RING_HEADROOM` (1.2) for the automatic band of `BAND_FACTOR` seam
    edges: that ramp is steep, and without headroom the sizing just inside
    the seam drops below the frozen edge length and jigsaw splits seam
    edges. `WIDE_HEADROOM` (1.0) with an explicit `transition_km`: the ramp
    is gentle, and 1.2 would only leave a ring of ~5% coarser elements
    along the seam.
    """
    return RING_HEADROOM if transition_km is None else WIDE_HEADROOM


def build_local_hfun(
    cavity: Cavity,
    target_km: float,
    transition_km: float | None = None,
    target_field: Callable | None = None,
):
    """euclidean-grid hfun (xg, yg, values) covering the padded cavity.

    `target_field(lon, lat) -> km`, when given, replaces the constant target
    away from the seam (NaNs fall back to `target_km`).
    """
    if target_km <= 0.0:
        raise RemeshError("target resolution must be positive")
    pad = max(
        BAND_FACTOR * float(cavity.edge_len_km.max()), transition_km or 0.0
    )
    finest = min(target_km, float(cavity.edge_len_km.min()))
    x0, x1 = cavity.ring_xy[:, 0].min() - pad, cavity.ring_xy[:, 0].max() + pad
    y0, y1 = cavity.ring_xy[:, 1].min() - pad, cavity.ring_xy[:, 1].max() + pad
    # cell: finest/3 when it fits, else floor at the grid cap — the hfun only
    # has to resolve the seam ramp (band >= 3x seam edge) and the target
    # plateau, so cells up to `finest` still sample both adequately; only a
    # cell coarser than the finest scale would blur the sizing
    cell = max(finest / 3.0, max(x1 - x0, y1 - y0) / (MAX_HFUN_GRID - 1))
    if cell > finest:
        raise RemeshError(
            f"region needs a {cell:.2f} km hfun cell to fit the "
            f"{MAX_HFUN_GRID}-per-axis grid, coarser than the finest scale "
            f"({finest:.2f} km) — split the region or coarsen the target"
        )
    nx = int(np.ceil((x1 - x0) / cell)) + 1
    ny = int(np.ceil((y1 - y0) / cell)) + 1
    xg = np.linspace(x0, x1, nx)
    yg = np.linspace(y0, y1, ny)
    xx, yy = np.meshgrid(xg, yg)
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    target = np.full(len(pts), float(target_km))
    if target_field is not None:
        lon, lat = stereo_inverse(pts[:, 0], pts[:, 1], cavity.lon0, cavity.lat0)
        vals = np.asarray(target_field(lon, lat), dtype=np.float64)
        good = np.isfinite(vals) & (vals > 0.0)
        target[good] = vals[good]
    h, _ = seam_hfun_at(pts, cavity, target, transition_km)
    return xg, yg, h.reshape(ny, nx)


# ------------------------------------------------------------------- jigsaw

#: Task 20 gate parameter set — do not "improve" without re-running the spike
_JIGSAW_OPTS = {
    "hfun_scal": "absolute",
    "hfun_hmax": float("inf"),
    "mesh_dims": 2,
    "optm_qlim": 0.95,
    "optm_iter": 32,
    "mesh_top1": True,
    "geom_feat": False,
}


def _cavity_hash(cavity: Cavity, xg, yg, hval) -> str:
    h = hashlib.sha256()
    for arr in (cavity.ring_xy, cavity.ring_edges, xg, yg, hval):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(repr(sorted(_JIGSAW_OPTS.items())).encode())
    return h.hexdigest()[:16]


def run_jigsaw_cavity(
    cavity: Cavity,
    xg,
    yg,
    hval,
    workdir=None,
    cache_dir=None,
    progress: Callable[[str], None] | None = None,
    process_callback: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Plane triangulation of the cavity with the ring frozen.

    Returns (points_xy, triangles); the ring vertices are guaranteed present
    bit-exact. Cached as npz keyed by the exact inputs (the determinism
    anchor for reproducible refinements). `progress` streams raw jigsaw
    stdout lines; `process_callback` receives the Popen right after spawn
    (jigsaw runs in its own process group, so big jobs can be cancelled).
    """
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"cavity_{_cavity_hash(cavity, xg, yg, hval)}.npz"
        if cache_file.exists():
            d = np.load(cache_file)
            if progress:
                progress(f"reusing cached cavity mesh {cache_file.name}")
            return d["pts"], d["tri"]

    if workdir is None:
        workdir = Path(cache_dir) if cache_dir is not None else Path(
            tempfile.mkdtemp(prefix="fmesh2_cavity_")
        )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if " " in str(workdir):
        raise JigsawError(
            f"jigsaw cannot parse paths containing spaces: {workdir}"
        )

    T = jigsawpy.jigsaw_msh_t
    opts = jigsawpy.jigsaw_jig_t()
    opts.geom_file = str(workdir / "cavity_geom.msh")
    opts.init_file = str(workdir / "cavity_init.msh")
    opts.hfun_file = str(workdir / "cavity_hfun.msh")
    opts.mesh_file = str(workdir / "cavity_mesh.msh")
    opts.jcfg_file = str(workdir / "cavity_opts.jig")

    geom = T()
    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.zeros(len(cavity.ring_xy), dtype=T.VERT2_t)
    geom.vert2["coord"] = cavity.ring_xy
    geom.edge2 = np.zeros(len(cavity.ring_edges), dtype=T.EDGE2_t)
    geom.edge2["index"] = cavity.ring_edges
    jigsawpy.savemsh(opts.geom_file, geom)

    init = T()
    init.mshID = "euclidean-mesh"
    init.ndims = +2
    init.vert2 = np.zeros(len(cavity.ring_xy), dtype=T.VERT2_t)
    init.vert2["coord"] = cavity.ring_xy
    jigsawpy.savemsh(opts.init_file, init)

    hmat = T()
    hmat.mshID = "euclidean-grid"
    hmat.ndims = +2
    hmat.xgrid = np.asarray(xg, dtype=T.REALS_t)
    hmat.ygrid = np.asarray(yg, dtype=T.REALS_t)
    hmat.value = np.asarray(hval, dtype=T.REALS_t)
    jigsawpy.savemsh(opts.hfun_file, hmat)

    for key, value in _JIGSAW_OPTS.items():
        setattr(opts, key, value)
    jigsawpy.savejig(opts.jcfg_file, opts)

    proc = subprocess.Popen(
        [_find_jigsaw(), opts.jcfg_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=workdir,  # jigsaw writes .log files into its cwd
        start_new_session=True,  # own process group: killable as a unit
    )
    if process_callback:
        process_callback(proc)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-25:]
        if progress and line:
            progress(line)
    code = proc.wait()
    if code != 0:
        raise JigsawError(
            f"jigsaw cavity remesh failed ({code}):\n" + "\n".join(tail)
        )

    out = T()
    jigsawpy.loadmsh(opts.mesh_file, out)
    pts = np.asarray(out.vert2["coord"], dtype=np.float64)
    tri = np.asarray(out.tria3["index"], dtype=np.int32)
    if cache_file is not None:
        np.savez_compressed(cache_file, pts=pts, tri=tri)
    return pts, tri


#: weld tolerance for jigsaw output vertices (1 m — far below any mesh scale)
WELD_TOL_KM = 1e-3


def _weld_cavity(
    pts: np.ndarray, tri: np.ndarray, ring_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Merge near-coincident jigsaw output vertices; drop collapsed triangles.

    At continent scale jigsaw occasionally emits clusters of vertices within
    ~1e-5 km of each other (observed: whole-Arctic run, 794 zero-area
    micro-triangles, two of them wedged onto seam edges — breaking the
    watertight stitch). Each cluster is welded to one representative; a
    member sitting bit-exact on a ring vertex wins, so the frozen-ring
    guarantee is untouched. Positions are never averaged.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    pairs = cKDTree(pts).query_pairs(WELD_TOL_KM, output_type="ndarray")
    if not len(pairs):
        return pts, tri
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(pts),) * 2
    )
    _, labels = connected_components(graph, directed=False)
    ring_pos = {p.tobytes() for p in np.ascontiguousarray(ring_xy)}
    pts = pts.copy()
    mapping = np.arange(len(pts))
    involved = np.unique(pairs)
    for lab in np.unique(labels[involved]):
        members = np.flatnonzero(labels == lab)
        on_ring = [m for m in members if np.ascontiguousarray(pts[m]).tobytes() in ring_pos]
        distinct = {np.ascontiguousarray(pts[m]).tobytes() for m in on_ring}
        if len(distinct) > 1:
            raise RemeshError(
                "two frozen ring vertices coincide within the weld tolerance — "
                "the region boundary pinches to a point; adjust the polygon"
            )
        # jigsaw lists init (ring) vertices first: the smallest index at the
        # ring position is the true frozen vertex
        rep = int(min(on_ring)) if on_ring else int(members.min())
        mapping[members] = rep
        # park orphaned duplicates far away so the exact-position ring match
        # can never pick one of them over the representative
        orphans = members[members != rep]
        pts[orphans] = 1e12
    tri = mapping[tri]
    ok = (tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 2] != tri[:, 0])
    return pts, tri[ok]


def _repair_split_seams(
    pts: np.ndarray,
    tri: np.ndarray,
    cavity: Cavity,
    ring_new: np.ndarray,
    progress: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Un-split frozen seam edges that jigsaw bisected.

    jigsaw's encroachment rule bisects a boundary edge whenever another
    vertex falls inside its diametral circle — independent of the hfun, so
    the seam band cannot prevent it where a (finely subdivided) coastline
    passes close to a seam edge (observed: 2 of 194 seam edges on the
    whole-Arctic run). Repair: delete each bisection vertex (it lies
    bit-on the segment) and re-triangulate its link — the fill restores
    the frozen edge exactly.
    """
    from .topo_ops import _ear_clip, _link_chain

    seam = cavity.ring_edges[~cavity.edge_is_coast]
    ge = np.sort(ring_new[seam], axis=1).astype(np.uint64)
    for a, b in ge:
        for _ in range(4):  # a cascade may have split the edge more than once
            has_a = (tri == int(a)).any(axis=1)
            has_b = (tri == int(b)).any(axis=1)
            if (has_a & has_b).any():
                break
            # bisection vertices lie ON the segment a-b (interior)
            pa, pb = pts[int(a)], pts[int(b)]
            seg = pb - pa
            L2 = float(seg @ seg)
            rel = pts - pa
            t_par = (rel @ seg) / L2
            d = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / np.sqrt(L2)
            cand = np.flatnonzero((d < 1e-6) & (t_par > 1e-9) & (t_par < 1 - 1e-9))
            cand = cand[(cand != a) & (cand != b)]
            if not cand.size:
                raise RemeshError(
                    "a frozen seam edge is missing from the jigsaw output and "
                    "no bisection vertex was found to repair it"
                )
            m = int(cand[np.argmin(np.abs(t_par[cand] - 0.5))])
            star, chain, closed = _link_chain(tri.astype(np.int64), m)
            fill = np.empty((0, 3), dtype=tri.dtype)
            if len(chain) >= 3:
                local, _ = _ear_clip(pts[chain], m)
                fill = np.asarray(chain, dtype=tri.dtype)[local]
            tri = np.vstack([np.delete(tri, star, axis=0), fill])
            if progress:
                progress(f"restored a bisected seam edge (removed split vertex {m})")
    return tri


# ---------------------------------------------------------- land cut inside

def _protected_land_cut(
    pts, tri, seam_new_idx: np.ndarray, lon, lat, topo: Topo
) -> np.ndarray:
    """Land-cut the cavity interior without ever touching the seam.

    Elements containing a seam vertex are exempt from every removal rule:
    the seam strip must stay compatible with the outside mesh, which was
    built with the *original* coastline — the new topography may disagree
    with it right at the ring (observed with RTopo vs core2). Elsewhere,
    elements are dropped per the fmesh rule (any land node), loose elements
    are cleaned, and only components attached to the seam survive.
    """
    elev = topo.elevation(lon, lat)
    land = elev >= 0.0
    protected = np.zeros(len(pts), dtype=bool)
    protected[seam_new_idx] = True
    land &= ~protected
    seam_elem = protected[tri].any(axis=1)
    tri = tri[seam_elem | ~land[tri].any(axis=1)]
    if not tri.size:
        raise RemeshError("no cavity elements survive the land cut")

    # loose-element rules with seam nodes and seam-adjacent elements exempt
    while tri.size:
        neighbors, valence = _neighbor_and_valence(tri, len(pts))
        v, d = valence, neighbors + 1
        kill = (
            ((v == 1) & (d == 3)) | ((v == 2) & (d == 5)) | ((v >= 3) & (v <= 6) & (d >= v + 3))
        ) & ~protected
        if not kill.any():
            break
        seam_elem = protected[tri].any(axis=1)
        drop = kill[tri].any(axis=1) & ~seam_elem
        if not drop.any():
            break
        tri = tri[~drop]

    # keep only components attached to the seam (drops carved-off ponds)
    if seam_new_idx.size and tri.size:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        graph = coo_matrix(
            (np.ones(edges.shape[0], dtype=np.int8), (edges[:, 0], edges[:, 1])),
            shape=(len(pts), len(pts)),
        )
        _, labels = connected_components(graph, directed=False)
        keep_labels = np.unique(labels[seam_new_idx])
        tri = tri[np.isin(labels[tri[:, 0]], keep_labels)]
    if not tri.size:
        raise RemeshError("land cut disconnected the cavity from its seam")
    return tri


# -------------------------------------------------------------------- stitch

def remesh_cavity(
    mesh: Mesh,
    cavity: Cavity,
    target_km: float,
    *,
    transition_km: float | None = None,
    target_field: Callable | None = None,
    topo: Topo | None = None,
    minimum_depth: float = 5.0,
    workdir=None,
    cache_dir=None,
    progress: Callable[[str], None] | None = None,
    process_callback: Callable[[subprocess.Popen], None] | None = None,
) -> RemeshResult:
    """Remesh the cavity at `target_km` and stitch it back bit-exact.

    Outside nodes keep their exact lon/lat/depth; node ids are renumbered
    contiguously with the old->new mapping returned (`-1` = removed). With
    `topo` given, the land cut is re-run inside the cavity (finer coastline
    detail) with the seam protected. `progress`/`process_callback` stream
    jigsaw output and expose its process for cancellation of big jobs.
    """
    xg, yg, hval = build_local_hfun(cavity, target_km, transition_km, target_field)
    reachable_min = _reachable_min_km(mesh, cavity, target_km, transition_km)
    if progress:
        progress(
            f"local hfun grid {len(xg)}x{len(yg)} "
            f"(cell {float(xg[1] - xg[0]):.2f} km)"
        )
        progress(f"expecting roughly {_estimate_elements(mesh, cavity, target_km, transition_km):,} elements")
    pts, tri = run_jigsaw_cavity(
        cavity, xg, yg, hval, workdir, cache_dir, progress, process_callback
    )
    n_raw = len(tri)
    pts, tri = _weld_cavity(pts, tri, cavity.ring_xy)
    if progress and len(tri) != n_raw:
        progress(f"welded {n_raw - len(tri)} degenerate micro-triangles away")

    # identify the frozen ring in the jigsaw output (gate: exact positions)
    dist, ring_new = cKDTree(pts).query(cavity.ring_xy, workers=-1)
    if dist.max() > SEAM_MATCH_TOL_KM or np.unique(ring_new).size != len(ring_new):
        raise RemeshError(
            f"frozen ring not preserved by jigsaw (max displacement {dist.max():.3e} km)"
        )
    tri = _repair_split_seams(pts, tri, cavity, ring_new, progress)

    seam_local = np.unique(cavity.ring_edges[~cavity.edge_is_coast])
    if topo is not None:
        lon_new, lat_new = stereo_inverse(pts[:, 0], pts[:, 1], cavity.lon0, cavity.lat0)
        tri = _protected_land_cut(
            pts, tri, ring_new[seam_local], lon_new, lat_new, topo
        )

    used = np.zeros(len(pts), dtype=bool)
    used[tri.ravel()] = True
    if not used[ring_new[seam_local]].all():
        raise RemeshError("a seam vertex lost all its cavity elements")

    # --- assemble the stitched node table --------------------------------
    is_ring_new = np.zeros(len(pts), dtype=bool)
    is_ring_new[ring_new] = True
    ring_old_of_new = np.full(len(pts), -1, dtype=np.int64)
    ring_old_of_new[ring_new] = cavity.ring_nodes

    outside_tri = mesh.tri[~cavity.elem_mask]
    keep_old = np.zeros(mesh.n_nodes, dtype=bool)
    keep_old[outside_tri.ravel()] = True
    # ring nodes still used by the new cavity stay even if coast-only
    keep_old[ring_old_of_new[is_ring_new & used]] = True

    mapping = np.full(mesh.n_nodes, -1, dtype=np.int64)
    mapping[keep_old] = np.arange(int(keep_old.sum()))
    n_kept = int(keep_old.sum())

    interior_new = used & ~is_ring_new
    interior_idx = np.flatnonzero(interior_new)
    new_id_of = np.full(len(pts), -1, dtype=np.int64)
    new_id_of[interior_idx] = n_kept + np.arange(interior_idx.size)
    new_id_of[ring_new] = mapping[cavity.ring_nodes]

    lon_i, lat_i = stereo_inverse(
        pts[interior_idx, 0], pts[interior_idx, 1], cavity.lon0, cavity.lat0
    )
    lon = np.concatenate([mesh.lon[keep_old], lon_i])
    lat = np.concatenate([mesh.lat[keep_old], lat_i])

    depth = None
    if mesh.depth is not None or topo is not None:
        if topo is not None:
            elev_i = topo.elevation(lon_i, lat_i)
            depth_i = np.abs(elev_i)
            depth_i[elev_i >= -minimum_depth] = minimum_depth
            # match the host mesh's sign convention (core2 stores depths
            # negative-down; the M2 pipeline writes positive-down)
            if mesh.depth is not None and np.nanmedian(mesh.depth) < 0:
                depth_i = -depth_i
        else:
            cav_nodes = np.unique(mesh.tri[cavity.elem_mask])
            if mesh.depth is None:
                raise RemeshError("mesh has no depths and no topo was given")
            from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

            from fmesh2.core.projection import stereo_xy

            cx, cy = stereo_xy(
                mesh.lon[cav_nodes], mesh.lat[cav_nodes], cavity.lon0, cavity.lat0
            )
            src = np.column_stack([cx, cy])
            vals = mesh.depth[cav_nodes]
            depth_i = LinearNDInterpolator(src, vals)(pts[interior_idx])
            bad = ~np.isfinite(depth_i)
            if bad.any():
                depth_i[bad] = NearestNDInterpolator(src, vals)(pts[interior_idx][bad])
        old_depth = (
            mesh.depth[keep_old]
            if mesh.depth is not None
            else np.full(n_kept, np.nan)
        )
        depth = np.concatenate([old_depth, depth_i])

    tri_new_part = new_id_of[tri]
    tri_stitched = np.vstack([mapping[outside_tri], tri_new_part]).astype(np.int32)

    coast = None
    if mesh.coast_flag is not None:
        on_boundary = boundary_nodes(tri_stitched, lon.size)
        coast = np.concatenate(
            [mesh.coast_flag[keep_old], on_boundary[n_kept:].astype(np.int8)]
        )

    stitched = Mesh(
        lon=lon, lat=lat, tri=tri_stitched, depth=depth, coast_flag=coast,
        levels=mesh.levels,
    )

    report = _validate(mesh, cavity, stitched, mapping, n_kept, target_km)
    report["reachable_min_km"] = reachable_min
    report["target_reached"] = bool(reachable_min <= 1.5 * target_km)
    return RemeshResult(mesh=stitched, mapping=mapping, report=report)


def _estimate_elements(
    mesh: Mesh, cavity: Cavity, target_km: float, transition_km: float | None
) -> int:
    """Rough new-element count: old element areas / equilateral area at the
    local sizing (evaluated at the old centroids) — sets expectations for
    big jobs before jigsaw starts."""
    from fmesh2.core.projection import stereo_xy

    cav_tri = mesh.tri[cavity.elem_mask]
    px, py = stereo_xy(mesh.lon, mesh.lat, cavity.lon0, cavity.lat0)
    p = np.stack([px[cav_tri], py[cav_tri]], axis=-1)  # (m, 3, 2)
    u = p[:, 1] - p[:, 0]
    v = p[:, 2] - p[:, 0]
    area = 0.5 * np.abs(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0])
    cx = p.mean(axis=1)
    h, _ = seam_hfun_at(cx, cavity, float(target_km), transition_km)
    return int(np.sum(area / (np.sqrt(3.0) / 4.0 * h**2)))


def _reachable_min_km(
    mesh: Mesh, cavity: Cavity, target_km: float, transition_km: float | None
) -> float:
    """Finest sizing the seam-protecting band allows INSIDE the cavity.

    The band ramps from the (frozen) seam edge length to the target over
    ~BAND_FACTOR x the local edge length — in a polygon small relative to
    that band the target is simply never reached, which reads as "my input
    was ignored". Sampled at the old cavity nodes and element centroids
    (the band is smooth, and those points are inside by construction);
    scalar-target estimate even when a target_field is given.
    """
    from fmesh2.core.projection import stereo_xy

    cav_tri = mesh.tri[cavity.elem_mask]
    nodes = np.unique(cav_tri)
    px, py = stereo_xy(mesh.lon[nodes], mesh.lat[nodes], cavity.lon0, cavity.lat0)
    cx, cy = stereo_xy(
        mesh.lon[cav_tri].mean(axis=1),
        mesh.lat[cav_tri].mean(axis=1),
        cavity.lon0,
        cavity.lat0,
    )
    pts = np.column_stack([np.concatenate([px, cx]), np.concatenate([py, cy])])
    h, _ = seam_hfun_at(pts, cavity, float(target_km), transition_km)
    return float(h.min())


def _validate(mesh, cavity, stitched, mapping, n_kept, target_km) -> dict:
    """Watertightness + quality + sizing report for the stitched mesh."""
    from fmesh2.core.metrics import local_resolution_km, triangle_min_angle_deg

    keys, counts = np.unique(
        _edge_keys(stitched.tri, stitched.n_nodes), return_counts=True
    )
    seam = cavity.ring_edges[~cavity.edge_is_coast]
    ge = np.sort(mapping[cavity.ring_nodes[seam]], axis=1)
    seam_keys = ge[:, 0].astype(np.uint64) * np.uint64(stitched.n_nodes) + ge[:, 1].astype(np.uint64)
    pos = np.searchsorted(keys, seam_keys)
    found = (pos < keys.size) & (keys[np.minimum(pos, keys.size - 1)] == seam_keys)
    watertight = bool(found.all() and (counts[pos[found]] == 2).all())

    res = local_resolution_km(stitched)
    interior = np.arange(n_kept, stitched.n_nodes)
    angles = triangle_min_angle_deg(stitched)
    n_outside = mesh.n_tri - int(cavity.elem_mask.sum())
    new_tris = np.arange(n_outside, stitched.n_tri)

    return {
        "watertight_seam": watertight,
        "seam_edges": int(seam_keys.size),
        "n_nodes_before": mesh.n_nodes,
        "n_nodes_after": stitched.n_nodes,
        "n_tri_before": mesh.n_tri,
        "n_tri_after": stitched.n_tri,
        "n_new_interior_nodes": int(stitched.n_nodes - n_kept),
        "interior_resolution_km": (
            float(np.median(res[interior])) if interior.size else None
        ),
        "target_km": float(target_km),
        "min_angle_deg": float(angles[new_tris].min()) if new_tris.size else None,
        "median_min_angle_deg": (
            float(np.median(angles[new_tris])) if new_tris.size else None
        ),
    }


def write_mapping_csv(path, mapping: np.ndarray) -> None:
    """old_id,new_id table (1-based FESOM ids; -1 = node removed)."""
    old = np.arange(1, mapping.size + 1)
    new = np.where(mapping >= 0, mapping + 1, -1)
    np.savetxt(
        path,
        np.column_stack([old, new]),
        fmt="%d",
        delimiter=",",
        header="old_id,new_id",
        comments="",
    )
