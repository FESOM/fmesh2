#!/usr/bin/env python
"""Task 20 GATE spike: remesh a cavity of an existing FESOM2 mesh with jigsaw
while keeping the cavity's boundary ring bit-frozen.

For each test polygon on core2:

1. select cavity elements (centroid-in-polygon in a polygon-centered
   stereographic plane), clean pinch nodes, keep the largest component;
2. extract the boundary ring as ordered closed loops and project it to the
   local stereographic plane (km);
3. build a euclidean-grid hfun that equals the measured seam spacing at the
   ring and ramps to a 3x finer target over a transition band inward;
4. run jigsaw euclidean-mesh with the ring as geometry edges and as the init
   mesh, under several option configurations;
5. check hard gate criteria per configuration:
   - every ring vertex exists in the output within RING_TOL_KM (1e-12 km),
     matched one-to-one;
   - the output mesh boundary is EXACTLY the input ring (no split/merged
     seam edges — watertight stitch guaranteed);
   - interior sizing follows the hfun (median incident-edge/hfun ratio
     within SIZING_MED);
   - no triangles outside the outer loop or inside hole loops.

The script exits non-zero unless at least one configuration passes every
required case; the passing configuration is the parameter set Tasks 21-22
must use. Run: `micromamba run -n meshgen python experiments/cavity_spike.py`
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jigsawpy
import numpy as np
import shapely
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from fmesh2.core.io_fesom import read_fesom
from fmesh2.core.projection import ring_center, stereo_xy
from fmesh2.core.topology import _edge_keys, boundary_edges, unique_edges
from fmesh2.generate.jigsaw_driver import _find_jigsaw
from fmesh2.hfun.region_layer import _close, _densify

RING_TOL_KM = 1e-12          # max allowed ring-vertex displacement
SIZING_MED = (0.70, 1.30)    # allowed median(edge_len / hfun) for interior nodes
TARGET_FACTOR = 3.0          # remesh at seam_spacing / TARGET_FACTOR
BAND_FACTOR = 3.0            # transition band = BAND_FACTOR * local seam-edge length

#: lon/lat polygons on core2 (drawn as boxes; densified before projection)
CASES = {
    "open_ocean": {
        "ring": [(-42.0, 28.0), (-28.0, 28.0), (-28.0, 36.0), (-42.0, 36.0)],
        "required": True,   # mid-North-Atlantic, no coast inside
    },
    "coastal": {
        "ring": [(-21.0, 15.0), (-13.0, 15.0), (-13.0, 23.0), (-21.0, 23.0)],
        "required": True,   # NW-Africa box crossing the coastline
    },
    "island": {
        "ring": [(-27.0, 61.5), (-11.0, 61.5), (-11.0, 67.5), (-27.0, 67.5)],
        "required": False,  # Iceland inside -> multi-loop ring, if resolved
    },
}

#: option configurations, simplest first — the first one passing all
#: required cases becomes the Task 21/22 parameter set.
#: geom_feat=True protects sharp ring corners with Ruppert-style balls and
#: SPLITS the adjacent frozen edges (measured: every lost seam edge touched a
#: 48-118 degree corner), so the frozen-ring recipe needs feat off — ring
#: vertices are supplied via the init mesh instead.
CONFIGS = [
    ("no-feat", dict(geom_feat=False, edge_tags=False, init_edges=False, no_zipdiv=False)),
    ("no-feat+init-edges", dict(geom_feat=False, edge_tags=False, init_edges=True, no_zipdiv=False)),
    ("no-feat+no-zipdiv", dict(geom_feat=False, edge_tags=False, init_edges=False, no_zipdiv=True)),
    ("feat", dict(geom_feat=True, edge_tags=False, init_edges=False, no_zipdiv=False)),  # negative control
]


# --------------------------------------------------------------- extraction

def _largest_component(tri: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep only the largest edge-connected component of the masked elements."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return mask
    keys = _edge_keys(tri[idx], int(tri.max()) + 1)
    # _edge_keys concatenates [edge01 of all, edge12 of all, edge20 of all]
    elem = np.tile(np.arange(idx.size), 3)
    order = np.argsort(keys, kind="stable")
    k, e = keys[order], elem[order]
    same = k[1:] == k[:-1]
    a, b = e[:-1][same], e[1:][same]
    graph = coo_matrix((np.ones(a.size), (a, b)), shape=(idx.size, idx.size))
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp > 1:
        keep = labels == np.bincount(labels).argmax()
        out = np.zeros_like(mask)
        out[idx[keep]] = True
        return out
    return mask


def select_cavity(mesh, ring_lonlat) -> tuple[np.ndarray, tuple[float, float]]:
    """Element mask of the cavity + stereographic projection center."""
    ring = _densify(_close(np.asarray(ring_lonlat, dtype=np.float64)))
    lon0, lat0 = ring_center(ring[:, 0], ring[:, 1])
    poly = shapely.Polygon(
        np.column_stack(stereo_xy(ring[:, 0], ring[:, 1], lon0, lat0))
    )
    px, py = stereo_xy(mesh.lon, mesh.lat, lon0, lat0)
    cx, cy = px[mesh.tri].mean(axis=1), py[mesh.tri].mean(axis=1)
    mask = shapely.contains(poly, shapely.points(cx, cy))
    mask = _largest_component(mesh.tri, mask)
    # pinch cleanup: a ring node with >2 incident ring edges (cavity touching
    # itself at a vertex) cannot be walked into simple loops — drop its
    # cavity elements and re-take the largest component until clean
    for _ in range(20):
        ring_e = boundary_edges(mesh.tri[mask], mesh.n_nodes)
        deg = np.bincount(ring_e.ravel(), minlength=mesh.n_nodes)
        bad = np.flatnonzero(deg > 2)
        if bad.size == 0:
            break
        mask &= ~np.isin(mesh.tri, bad).any(axis=1)
        mask = _largest_component(mesh.tri, mask)
    else:
        raise RuntimeError("pinch cleanup did not converge")
    if mask.sum() < 20:
        raise RuntimeError(f"cavity too small: {mask.sum()} elements")
    return mask, (lon0, lat0)


def order_loops(ring_edges: np.ndarray) -> list[list[int]]:
    """Ordered closed loops (as node-id lists, not repeated at the end)."""
    nbr: dict[int, list[int]] = {}
    for a, b in ring_edges.tolist():
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    for node, ns in nbr.items():
        if len(ns) != 2:
            raise RuntimeError(f"ring node {node} has {len(ns)} ring edges")
    unvisited = set(nbr)
    loops = []
    while unvisited:
        start = min(unvisited)
        loop, prev, cur = [start], -1, start
        while True:
            a, b = nbr[cur]
            nxt = b if a == prev else a
            if nxt == start:
                break
            loop.append(nxt)
            prev, cur = cur, nxt
        loops.append(loop)
        unvisited -= set(loop)
    return loops


def extract_ring(mesh, cav_mask, lon0, lat0):
    """Projected ring points, local edge list, loops, per-vertex spacing."""
    ring_e = boundary_edges(mesh.tri[cav_mask], mesh.n_nodes)
    loops = order_loops(ring_e)
    ring_nodes = np.concatenate([np.asarray(l, dtype=np.int64) for l in loops])
    edges_local, off = [], 0
    for loop in loops:
        m = len(loop)
        i = np.arange(m)
        edges_local.append(np.column_stack([off + i, off + (i + 1) % m]))
        off += m
    edges_local = np.vstack(edges_local)
    x, y = stereo_xy(mesh.lon[ring_nodes], mesh.lat[ring_nodes], lon0, lat0)
    pts = np.column_stack([x, y])
    seg = np.hypot(*(pts[edges_local[:, 0]] - pts[edges_local[:, 1]]).T)
    ssum = np.zeros(len(pts))
    scnt = np.zeros(len(pts))
    for col in (0, 1):
        np.add.at(ssum, edges_local[:, col], seg)
        np.add.at(scnt, edges_local[:, col], 1.0)
    spacing = ssum / scnt
    return pts, edges_local, loops, spacing


# --------------------------------------------------------------------- hfun

#: hfun at the ring = RING_HEADROOM x local seam-edge length, so jigsaw never
#: sees a frozen edge as oversized (splitting it would break the seam)
RING_HEADROOM = 1.2


def ring_h(points, ring_pts, ring_edges, target, band_factor):
    """(h, distance) of the seam-spacing->target ramp at arbitrary points.

    Distance is to the nearest ring SEGMENT; the near-ring value is that
    segment's own length (a per-node ramp would sag below the edge length at
    the midpoint of long edges) and the transition band is LOCAL,
    band_factor x that segment's length: with a global band scaled to the
    median spacing, the hfun around the longest seam edges decays too fast
    and jigsaw splits them (measured on the coastal case: band/L_max = 1.35
    failed, >= 2.1 passed).
    """
    segs = shapely.linestrings(ring_pts[ring_edges])
    seg_len = np.hypot(*(ring_pts[ring_edges[:, 0]] - ring_pts[ring_edges[:, 1]]).T)
    pgeo = shapely.points(points)
    inp, near = shapely.STRtree(segs).query_nearest(pgeo, all_matches=False)
    d = np.empty(len(points))
    hseam = np.empty(len(points))
    band = np.empty(len(points))
    d[inp] = shapely.distance(pgeo[inp], segs[near])
    hseam[inp] = RING_HEADROOM * seg_len[near]
    band[inp] = band_factor * seg_len[near]
    h = target + (hseam - target) * np.clip(1.0 - d / band, 0.0, 1.0)
    return h, d


def build_hfun(pts, edges, target, band_factor):
    """euclidean-grid hfun: local seam spacing at the ring, target inward."""
    cell = target / 3.0
    seg_len = np.hypot(*(pts[edges[:, 0]] - pts[edges[:, 1]]).T)
    pad = band_factor * float(seg_len.max())
    x0, x1 = pts[:, 0].min() - pad, pts[:, 0].max() + pad
    y0, y1 = pts[:, 1].min() - pad, pts[:, 1].max() + pad
    xg = np.linspace(x0, x1, int(np.ceil((x1 - x0) / cell)) + 1)
    yg = np.linspace(y0, y1, int(np.ceil((y1 - y0) / cell)) + 1)
    xx, yy = np.meshgrid(xg, yg)
    h, _ = ring_h(np.column_stack([xx.ravel(), yy.ravel()]), pts, edges, target, band_factor)
    return xg, yg, h.reshape(yg.size, xg.size)


# ------------------------------------------------------------------- jigsaw

def run_cavity(workdir: Path, tag: str, pts, edges, xg, yg, hval, cfg):
    """One jigsaw euclidean-mesh run; returns (out_pts, out_tri, seconds)."""
    workdir.mkdir(parents=True, exist_ok=True)
    T = jigsawpy.jigsaw_msh_t
    opts = jigsawpy.jigsaw_jig_t()
    opts.geom_file = str(workdir / f"{tag}_geom.msh")
    opts.init_file = str(workdir / f"{tag}_init.msh")
    opts.hfun_file = str(workdir / f"{tag}_hfun.msh")
    opts.mesh_file = str(workdir / f"{tag}_mesh.msh")
    opts.jcfg_file = str(workdir / f"{tag}_opts.jig")

    geom = T()
    geom.mshID = "euclidean-mesh"
    geom.ndims = +2
    geom.vert2 = np.zeros(len(pts), dtype=T.VERT2_t)
    geom.vert2["coord"] = pts
    geom.edge2 = np.zeros(len(edges), dtype=T.EDGE2_t)
    geom.edge2["index"] = edges
    if cfg["edge_tags"]:
        # a unique IDtag per seam edge makes every ring vertex a topological
        # feature (junction of differently-tagged parts) => pinned by jigsaw
        geom.edge2["IDtag"] = np.arange(1, len(edges) + 1)
    jigsawpy.savemsh(opts.geom_file, geom)

    init = T()
    init.mshID = "euclidean-mesh"
    init.ndims = +2
    init.vert2 = np.zeros(len(pts), dtype=T.VERT2_t)
    init.vert2["coord"] = pts
    if cfg["init_edges"]:
        init.edge2 = np.zeros(len(edges), dtype=T.EDGE2_t)
        init.edge2["index"] = edges
        if cfg["edge_tags"]:
            init.edge2["IDtag"] = np.arange(1, len(edges) + 1)
    jigsawpy.savemsh(opts.init_file, init)

    hmat = T()
    hmat.mshID = "euclidean-grid"
    hmat.ndims = +2
    hmat.xgrid = np.asarray(xg, dtype=T.REALS_t)
    hmat.ygrid = np.asarray(yg, dtype=T.REALS_t)
    hmat.value = np.asarray(hval, dtype=T.REALS_t)
    jigsawpy.savemsh(opts.hfun_file, hmat)

    opts.hfun_scal = "absolute"
    # CRITICAL: without an explicit hmax, jigsaw applies its relative-mode
    # default (0.02) even under absolute scaling on euclidean geometry,
    # capping h at ~20 m and refining without bound
    opts.hfun_hmax = float("inf")
    opts.mesh_dims = +2
    opts.optm_qlim = 0.95
    opts.optm_iter = 32
    opts.mesh_top1 = True
    opts.geom_feat = cfg["geom_feat"]
    if cfg["no_zipdiv"]:
        opts.optm_zip_ = False
        opts.optm_div_ = False
    jigsawpy.savejig(opts.jcfg_file, opts)

    t0 = time.perf_counter()
    proc = subprocess.run(
        [_find_jigsaw(), opts.jcfg_file],
        cwd=workdir,  # jigsaw writes .log files into its cwd
        capture_output=True,
        text=True,
        timeout=600,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = "\n".join(proc.stdout.splitlines()[-25:])
        raise RuntimeError(f"jigsaw failed ({proc.returncode}):\n{tail}")
    out = T()
    jigsawpy.loadmsh(opts.mesh_file, out)
    return np.asarray(out.vert2["coord"], float), np.asarray(out.tria3["index"], np.int32), dt


# --------------------------------------------------------------- evaluation

def evaluate(pts, edges, loops, target, band_factor, out_pts, out_tri):
    res = {"n_out_nodes": len(out_pts), "n_out_tris": len(out_tri)}

    # 1. frozen ring: one-to-one match within RING_TOL_KM
    d, idx = cKDTree(out_pts).query(pts, workers=-1)
    res["ring_max_move_km"] = float(d.max())
    res["ring_one_to_one"] = bool(np.unique(idx).size == len(pts))

    # 2. watertight seam: output boundary == exactly the input ring edges
    out_bnd = boundary_edges(out_tri, len(out_pts))
    out_set = {tuple(sorted(e)) for e in out_bnd.tolist()}
    ring_set = {tuple(sorted((int(idx[a]), int(idx[b])))) for a, b in edges.tolist()}
    res["extra_boundary_edges"] = len(out_set - ring_set)
    res["lost_ring_edges"] = len(ring_set - out_set)
    res["boundary_equals_ring"] = out_set == ring_set

    # 3. containment: no triangles outside the shell loop / inside holes
    starts = np.cumsum([0] + [len(l) for l in loops])
    loop_polys = [
        shapely.Polygon(pts[starts[k]:starts[k + 1]]) for k in range(len(loops))
    ]
    shell = int(np.argmax([abs(p.area) for p in loop_polys]))
    cen = shapely.points(
        out_pts[out_tri, 0].mean(axis=1), out_pts[out_tri, 1].mean(axis=1)
    )
    res["tris_outside_shell"] = int((~shapely.contains(loop_polys[shell], cen)).sum())
    res["tris_in_holes"] = int(
        sum(
            shapely.contains(loop_polys[k], cen).sum()
            for k in range(len(loops))
            if k != shell
        )
    )

    # 4. sizing: mean incident edge length vs the hfun ramp, interior nodes
    eo = unique_edges(out_tri, len(out_pts))
    seg = np.hypot(*(out_pts[eo[:, 0]] - out_pts[eo[:, 1]]).T)
    esum = np.zeros(len(out_pts))
    ecnt = np.zeros(len(out_pts))
    for col in (0, 1):
        np.add.at(esum, eo[:, col], seg)
        np.add.at(ecnt, eo[:, col], 1.0)
    mean_edge = esum / np.maximum(ecnt, 1.0)
    interior = np.ones(len(out_pts), dtype=bool)
    interior[idx] = False
    h_true, d_ring = ring_h(out_pts, pts, edges, target, band_factor)
    seg_len = np.hypot(*(pts[edges[:, 0]] - pts[edges[:, 1]]).T)
    band = band_factor * float(seg_len.max())  # deep-interior cutoff
    ratio = mean_edge[interior] / h_true[interior]
    res["n_interior"] = int(interior.sum())
    if interior.any():
        res["sizing_median"] = float(np.median(ratio))
        res["sizing_p10"] = float(np.percentile(ratio, 10))
        res["sizing_p90"] = float(np.percentile(ratio, 90))
    deep = interior & (d_ring > band)
    res["deep_median_vs_target"] = (
        float(np.median(mean_edge[deep]) / target) if deep.any() else None
    )

    res["pass"] = bool(
        res["ring_max_move_km"] <= RING_TOL_KM
        and res["ring_one_to_one"]
        and res["boundary_equals_ring"]
        and res["tris_outside_shell"] == 0
        and res["tris_in_holes"] == 0
        and res["n_interior"] > 0
        and SIZING_MED[0] <= res["sizing_median"] <= SIZING_MED[1]
        and (
            res["deep_median_vs_target"] is None
            or SIZING_MED[0] <= res["deep_median_vs_target"] <= SIZING_MED[1]
        )
    )
    return res


# --------------------------------------------------------------------- main

def plot_case(name, mesh, cav_mask, lon0, lat0, pts, out_pts, out_tri, out_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    px, py = stereo_xy(mesh.lon, mesh.lat, lon0, lat0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    axes[0].triplot(px, py, mesh.tri[cav_mask], lw=0.4, color="0.4")
    axes[0].set_title(f"{name}: original cavity ({cav_mask.sum()} elements)")
    axes[1].triplot(out_pts[:, 0], out_pts[:, 1], out_tri, lw=0.3, color="0.4")
    axes[1].set_title(f"remeshed ({len(out_tri)} elements)")
    for ax in axes:
        ax.plot(
            np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]),
            ".", ms=2, color="crimson",
        )
        ax.set_aspect("equal")
        ax.set_xlabel("km")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"cavity-{name}.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mesh", default=str(repo.parent / "core2"))
    ap.add_argument("--workdir", default=None, help="keep jigsaw files here")
    ap.add_argument("--plot", action="store_true", help="write docs/notes/figures/*.png")
    args = ap.parse_args()

    workdir = Path(args.workdir) if args.workdir else Path(
        tempfile.mkdtemp(prefix="cavity_spike_")
    )
    print(f"mesh: {args.mesh}\njigsaw files: {workdir}\n")
    mesh = read_fesom(args.mesh)

    results: dict[str, dict[str, dict]] = {}
    for name, case in CASES.items():
        try:
            cav_mask, (lon0, lat0) = select_cavity(mesh, case["ring"])
            pts, edges, loops, spacing = extract_ring(mesh, cav_mask, lon0, lat0)
        except RuntimeError as exc:
            if case["required"]:
                raise
            print(f"[{name}] skipped: {exc}")
            continue
        if name == "island" and len(loops) < 2:
            print(f"[{name}] skipped: island not resolved as a hole in core2")
            continue
        s_med = float(np.median(spacing))
        target = s_med / TARGET_FACTOR
        print(
            f"[{name}] cavity: {int(cav_mask.sum())} elems, ring: {len(pts)} nodes "
            f"in {len(loops)} loop(s), spacing median {s_med:.1f} km "
            f"-> target {target:.1f} km, band {BAND_FACTOR:.1f}x local edge"
        )
        xg, yg, hval = build_hfun(pts, edges, target, BAND_FACTOR)
        results[name] = {}
        for cfg_name, cfg in CONFIGS:
            tag = f"{name}_{cfg_name}".replace("+", "_")
            try:
                out_pts, out_tri, dt = run_cavity(
                    workdir, tag, pts, edges, xg, yg, hval, cfg
                )
                r = evaluate(pts, edges, loops, target, BAND_FACTOR, out_pts, out_tri)
                r["seconds"] = dt
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                r = {"pass": False, "error": str(exc).splitlines()[0][:80]}
            results[name][cfg_name] = r
            flag = "PASS" if r["pass"] else "fail"
            detail = (
                f"move {r['ring_max_move_km']:.2e} km, "
                f"bnd {'ok' if r['boundary_equals_ring'] else f'+{r['extra_boundary_edges']}/-{r['lost_ring_edges']}'}, "
                f"sizing med {r.get('sizing_median', float('nan')):.2f} "
                f"[{r.get('sizing_p10', float('nan')):.2f},{r.get('sizing_p90', float('nan')):.2f}], "
                f"deep {r['deep_median_vs_target'] and f'{r['deep_median_vs_target']:.2f}' or 'n/a'}, "
                f"holes {r['tris_in_holes']}, outside {r['tris_outside_shell']}, "
                f"{r['n_out_tris']} tris, {r['seconds']:.1f}s"
                if "error" not in r
                else r["error"]
            )
            print(f"  {cfg_name:24s} {flag}  {detail}")
            if args.plot and r["pass"] and cfg_name == CONFIGS[0][0]:
                plot_case(
                    name, mesh, cav_mask, lon0, lat0, pts, out_pts, out_tri,
                    repo / "docs" / "notes" / "figures",
                )

    required = [n for n, c in CASES.items() if c["required"] and n in results]
    ran = list(results)
    recommended = next(
        (
            cfg_name
            for cfg_name, _ in CONFIGS
            if all(results[case][cfg_name]["pass"] for case in ran)
        ),
        None,
    )

    print()
    if recommended is None:
        # fall back: a config passing at least the required cases
        recommended = next(
            (
                cfg_name
                for cfg_name, _ in CONFIGS
                if all(results[case][cfg_name]["pass"] for case in required)
            ),
            None,
        )
    assert recommended is not None, (
        "GATE FAILED: no jigsaw configuration froze the ring — "
        "Tasks 21-22 need the Shewchuk `triangle` fallback"
    )
    # hard gate assertions on the recommended configuration
    for case in required:
        r = results[case][recommended]
        assert r["ring_max_move_km"] <= RING_TOL_KM, (case, r["ring_max_move_km"])
        assert r["ring_one_to_one"] and r["boundary_equals_ring"], (case, r)
        assert SIZING_MED[0] <= r["sizing_median"] <= SIZING_MED[1], (case, r)
    print(
        f"GATE PASSED — use jigsaw euclidean-mesh with config '{recommended}' "
        f"(cases: {', '.join(ran)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
