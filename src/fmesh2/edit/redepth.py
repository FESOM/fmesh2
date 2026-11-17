"""Re-assign node depths from a topography file and smooth them ("redepth").

Two modes, both: sample the topography at the nodes, raise anything
shallower than min_depth to min_depth, then smooth with the neighbour-average sweep of
FESOM's `smooth_seafloor.f90` (Q. Wang, 2011):

* local  — only the nodes a refinement ADDED are sampled and smoothed;
  every old node keeps its depth bit-identical and acts as a fixed
  neighbour value. New nodes are those whose exact lon/lat does not occur
  in the original mesh (refinement keeps old positions bit-identical), so
  any number of refinement passes works without mapping tables.
* global — every node is sampled and smoothed: the whole bathymetry
  re-derived on the mesh.

No separate seam treatment: refinement grows the elements gradually from
the old seam size down to the target, so new nodes next to the seam have
roughly the old spacing and the smoother acts on them over roughly the
same distance as it did on the original mesh. That only holds if sampling,
clamp and smoothing settings match the pipeline that produced the original
mesh — the defaults below are the values in `smooth_seafloor.f90`, except
that no maximum depth is imposed.

Depth sign follows the host mesh. Recomputed depths are rounded to whole
metres, as `smooth_seafloor.f90` writes them (format f8.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix

from fmesh2.core.mesh import Mesh
from fmesh2.core.topology import unique_edges
from fmesh2.data.topo import Topo

# settings of smooth_seafloor.f90 — must match how the original mesh was made
MIN_DEPTH = 30.0       # m, set_min_depth / min_water_depth
MAX_DEPTH = None       # no cap: the topography data sets the deepest value
ITERATIONS = 1         # num_iteration
CENTER_WEIGHT = 2.0    # relative_center_weight


class RedepthError(ValueError):
    """Invalid input for redepth (caught by the CLI for a one-line message)."""


@dataclass
class RedepthResult:
    mesh: Mesh            # input mesh with depths replaced
    is_new: np.ndarray    # per node: depth was recomputed
    report: dict


def match_original_nodes(original: Mesh, refined: Mesh) -> np.ndarray:
    """Index of each refined node in the original mesh, -1 for new nodes.

    Exact lon/lat equality: refinement and the FESOM ASCII round trip are
    both bit-exact, so a node without an exact match did not exist before.
    """
    key_o = original.lon + 1j * original.lat
    key_r = refined.lon + 1j * refined.lat
    order = np.argsort(key_o, kind="stable")
    sorted_o = key_o[order]
    pos = np.clip(np.searchsorted(sorted_o, key_r), 0, sorted_o.size - 1)
    found = sorted_o[pos] == key_r
    return np.where(found, order[pos], -1).astype(np.int64)


def neighbour_matrix(tri: np.ndarray, n_nodes: int):
    """Mesh edges and, per node, the nodes it shares an edge with.

    Returns (edges as (e, 2), sparse matrix whose row i holds the
    neighbours of node i — the smoother walks these rows).
    """
    edges = unique_edges(tri, n_nodes).astype(np.int64)
    both = np.concatenate([edges, edges[:, ::-1]])
    nbrs = coo_matrix(
        (np.ones(len(both), dtype=np.int8), (both[:, 0], both[:, 1])),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    nbrs.sort_indices()
    return edges, nbrs


def sample_depth(topo: Topo, lon, lat, min_depth: float, max_depth: float | None = None):
    """Topography at the points as positive-down depth, clamped.

    Anything shallower than `min_depth` (including points that are land in
    the topography) is set to `min_depth`; with `max_depth` given, anything
    deeper is set to it (default: no cap). Returns (depth, number of land
    points).
    """
    elev = topo.elevation(lon, lat)
    return np.clip(-elev, min_depth, max_depth), int(np.count_nonzero(elev >= 0.0))


def smooth_depths(
    h: np.ndarray,
    nbrs,
    active: np.ndarray,
    iterations: int = ITERATIONS,
    center_weight: float = CENTER_WEIGHT,
) -> np.ndarray:
    """`smooth_seafloor.f90` sweep over the `active` nodes.

    Per iteration the active nodes are visited from shallowest to deepest
    (order fixed at the start of the iteration) and updated in place, so
    later nodes already see their smoothed neighbours:

        h[m] = (sum(h[neighbours]) + (cw*(n-1) - 1) * h[m]) / ((n-1) * (1+cw))

    with n the neighbour count and cw the relative centre weight.
    Inactive nodes are read but never written.
    """
    if center_weight < 1.0:
        raise RedepthError("center weight must be >= 1")
    if iterations < 0:
        raise RedepthError("iterations must be >= 0")
    idx = np.flatnonzero(active)
    indptr = nbrs.indptr.tolist()
    indices = nbrs.indices.tolist()
    hl = np.asarray(h, dtype=np.float64).tolist()
    cw = float(center_weight)
    for _ in range(int(iterations)):
        cur = np.asarray(hl)
        for m in idx[np.argsort(cur[idx], kind="stable")].tolist():
            s, e = indptr[m], indptr[m + 1]
            n = e - s
            if n < 2:
                continue
            total = 0.0
            for k in range(s, e):
                total += hl[indices[k]]
            hl[m] = (total + (cw * (n - 1) - 1.0) * hl[m]) / ((n - 1) * (1.0 + cw))
    return np.asarray(hl, dtype=np.float64)


def round_whole_metres(h: np.ndarray) -> np.ndarray:
    """Round positive depths to whole metres, halves away from zero — the
    rounding of smooth_seafloor.f90's `write(13,'(1f8.0)')`."""
    return np.floor(np.asarray(h, dtype=np.float64) + 0.5)


def edge_steepness(h: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Per-edge bathymetry steepness r = |h1 - h2| / (h1 + h2), h positive."""
    a, b = h[edges[:, 0]], h[edges[:, 1]]
    return np.abs(a - b) / np.maximum(a + b, 1e-12)


def _stats(r: np.ndarray) -> dict | None:
    if not r.size:
        return None
    return {
        "edges": int(r.size),
        "median": float(np.median(r)),
        "p99": float(np.percentile(r, 99)),
        "max": float(r.max()),
    }


def _check_settings(min_depth, max_depth) -> None:
    if min_depth < 0.0 or (max_depth is not None and not min_depth < max_depth):
        raise RedepthError(f"invalid depth range [{min_depth}, {max_depth}]")


def _host_sign(mesh: Mesh) -> float:
    if mesh.depth is None:
        raise RedepthError("mesh has no depths (aux3d.out)")
    return -1.0 if np.nanmedian(mesh.depth) < 0 else 1.0


def redepth_local(
    original: Mesh,
    refined: Mesh,
    topo: Topo,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float | None = MAX_DEPTH,
    iterations: int = ITERATIONS,
    center_weight: float = CENTER_WEIGHT,
) -> RedepthResult:
    """Sample + smooth the nodes `refined` has and `original` has not."""
    _check_settings(min_depth, max_depth)
    sign = _host_sign(refined)
    _host_sign(original)
    orig_index = match_original_nodes(original, refined)
    is_new = orig_index < 0
    n_new = int(is_new.sum())
    if n_new == 0:
        raise RedepthError("no new nodes: both meshes have identical node positions")
    if n_new == refined.n_nodes:
        raise RedepthError(
            "no node matches the original mesh — is it the mesh the refinement started from?"
        )
    old = ~is_new
    n_old_changed = int(
        np.count_nonzero(refined.depth[old] != original.depth[orig_index[old]])
    )

    edges, nbrs = neighbour_matrix(refined.tri, refined.n_nodes)
    h = np.abs(refined.depth).astype(np.float64)
    h[is_new], n_land = sample_depth(
        topo, refined.lon[is_new], refined.lat[is_new], min_depth, max_depth
    )
    h = smooth_depths(h, nbrs, is_new, iterations, center_weight)

    depth = refined.depth.copy()
    h[is_new] = round_whole_metres(h[is_new])
    depth[is_new] = sign * h[is_new]  # old entries stay bit-identical

    e_new = is_new[edges]
    r = edge_steepness(h, edges)
    report = {
        "mode": "local",
        "n_nodes": refined.n_nodes,
        "n_new": n_new,
        "n_old": refined.n_nodes - n_new,
        "n_new_on_topo_land": n_land,
        "n_old_depth_differs_from_original": n_old_changed,
        "settings": {"min_depth": min_depth, "max_depth": max_depth,
                     "iterations": iterations, "center_weight": center_weight},
        "steepness": {
            "untouched": _stats(r[~e_new[:, 0] & ~e_new[:, 1]]),
            "seam": _stats(r[e_new[:, 0] != e_new[:, 1]]),
            "new": _stats(r[e_new[:, 0] & e_new[:, 1]]),
        },
    }
    mesh = Mesh(lon=refined.lon, lat=refined.lat, tri=refined.tri, depth=depth,
                coast_flag=refined.coast_flag, levels=refined.levels)
    return RedepthResult(mesh=mesh, is_new=is_new, report=report)


def redepth_global(
    mesh: Mesh,
    topo: Topo,
    *,
    min_depth: float = MIN_DEPTH,
    max_depth: float | None = MAX_DEPTH,
    iterations: int = ITERATIONS,
    center_weight: float = CENTER_WEIGHT,
) -> RedepthResult:
    """Sample + smooth every node of `mesh`."""
    _check_settings(min_depth, max_depth)
    sign = _host_sign(mesh)
    edges, nbrs = neighbour_matrix(mesh.tri, mesh.n_nodes)
    everyone = np.ones(mesh.n_nodes, dtype=bool)
    h, n_land = sample_depth(topo, mesh.lon, mesh.lat, min_depth, max_depth)
    h = round_whole_metres(smooth_depths(h, nbrs, everyone, iterations, center_weight))
    report = {
        "mode": "global",
        "n_nodes": mesh.n_nodes,
        "n_on_topo_land": n_land,
        "settings": {"min_depth": min_depth, "max_depth": max_depth,
                     "iterations": iterations, "center_weight": center_weight},
        "steepness": {"all": _stats(edge_steepness(h, edges))},
    }
    out = Mesh(lon=mesh.lon, lat=mesh.lat, tri=mesh.tri, depth=sign * h,
               coast_flag=mesh.coast_flag, levels=mesh.levels)
    return RedepthResult(mesh=out, is_new=everyone, report=report)
