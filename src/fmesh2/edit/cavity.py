"""Cavity extraction for local remeshing.

A *cavity* is the set of mesh elements selected by a drawn polygon, together
with its boundary ring: the closed loops of edges separating it from the
untouched outside mesh (seam edges) and/or from land (coast edges). The ring
is what a local remesh must keep bit-frozen so the result stitches back
watertight — algorithm and jigsaw parameters validated by the Task 20 gate
spike (`experiments/cavity_spike.py`, see `docs/notes/cavity-spike.md`).

All plane coordinates are km in a polygon-centered spherical stereographic
projection (`core.projection`): conformal, so point-in-polygon selection in
the plane is topologically exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely

from fmesh2.core.mesh import Mesh
from fmesh2.core.projection import ring_center, stereo_xy
from fmesh2.core.topology import _edge_keys, boundary_edges
from fmesh2.hfun.region_layer import _close, _densify

#: pinch cleanup is strictly monotone (each pass removes elements), the cap
#: only guards against a logic regression looping forever
_MAX_CLEAN_PASSES = 50


class CavityError(ValueError):
    """The polygon does not select a remeshable cavity."""


@dataclass
class Cavity:
    """A cavity and its frozen boundary ring.

    `ring_nodes` holds global node ids of all boundary loops concatenated;
    `loop_starts` are offsets delimiting each loop (`n_loops + 1` entries).
    `ring_edges` are (r, 2) indices INTO `ring_nodes`, one edge per ring node
    (each loop is closed). `ring_xy` are stereographic plane coordinates (km)
    centered at (`lon0`, `lat0`). `polygon_xy` is the (densified, closed)
    polygon that selected the cavity, in the same plane — None for a
    cavity selected by an element mask.
    """

    elem_mask: np.ndarray
    lon0: float
    lat0: float
    ring_nodes: np.ndarray
    loop_starts: np.ndarray
    ring_xy: np.ndarray
    ring_edges: np.ndarray
    edge_len_km: np.ndarray
    edge_is_coast: np.ndarray
    polygon_xy: np.ndarray | None = None

    @property
    def n_elems(self) -> int:
        return int(self.elem_mask.sum())

    @property
    def n_loops(self) -> int:
        return int(self.loop_starts.size - 1)

    @property
    def loops(self) -> list[np.ndarray]:
        """Global node ids per boundary loop."""
        return [
            self.ring_nodes[self.loop_starts[k]:self.loop_starts[k + 1]]
            for k in range(self.n_loops)
        ]


def _polygon_plane(ring_lonlat: np.ndarray) -> tuple[shapely.Polygon, float, float]:
    """Densified drawn polygon in its own stereographic plane."""
    ring = _densify(_close(np.asarray(ring_lonlat, dtype=np.float64)))
    lon0, lat0 = ring_center(ring[:, 0], ring[:, 1])
    xy = np.column_stack(stereo_xy(ring[:, 0], ring[:, 1], lon0, lat0))
    # pole-encircling rings close at lon +-180: the first and last projected
    # points coincide only to ~1e-13, which shapely reads as a degenerate
    # self-intersecting closure segment — drop the duplicate and let the
    # polygon close itself exactly
    if np.hypot(*(xy[0] - xy[-1])) < 1e-6:
        xy = xy[:-1]
    poly = shapely.Polygon(xy)
    if not poly.is_valid:
        raise CavityError("polygon is self-intersecting")
    return poly, lon0, lat0


def _largest_component(tri: np.ndarray, mask: np.ndarray, n_nodes: int) -> np.ndarray:
    """Keep only the largest edge-connected component of the masked elements."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return mask
    keys = _edge_keys(tri[idx], n_nodes)
    # _edge_keys lays out [edge01 of all, edge12 of all, edge20 of all]
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


def clean_selection(tri: np.ndarray, mask: np.ndarray, n_nodes: int) -> np.ndarray:
    """Reduce an element selection to one component with simple ring loops.

    Keeps the largest edge-connected component, then repeatedly removes the
    elements incident to any *pinch node* (a boundary node with more than two
    boundary edges — the cavity touching itself at a vertex, which cannot be
    walked into simple loops).
    """
    mask = _largest_component(tri, mask, n_nodes)
    for _ in range(_MAX_CLEAN_PASSES):
        if not mask.any():
            raise CavityError("selection vanished during pinch cleanup")
        ring_e = boundary_edges(tri[mask], n_nodes)
        deg = np.bincount(ring_e.ravel(), minlength=n_nodes)
        bad = np.flatnonzero(deg > 2)
        if bad.size == 0:
            return mask
        mask = mask & ~np.isin(tri, bad).any(axis=1)
        mask = _largest_component(tri, mask, n_nodes)
    raise CavityError("pinch cleanup did not converge")


def select_elements(mesh: Mesh, ring_lonlat) -> np.ndarray:
    """Boolean element mask of the cavity selected by a lon/lat polygon.

    Elements are picked by centroid-in-polygon. A polygon drawn snugly
    around individual node(s) — smaller than any element, so it contains
    no centroid — falls back to selecting the enclosed nodes' full element
    stars: the nodes become cavity-interior, so a remesh replaces (or,
    with a coarser target, deletes) exactly those nodes.
    """
    poly, lon0, lat0 = _polygon_plane(ring_lonlat)
    px, py = stereo_xy(mesh.lon, mesh.lat, lon0, lat0)
    shapely.prepare(poly)
    mask = shapely.contains_xy(
        poly, px[mesh.tri].mean(axis=1), py[mesh.tri].mean(axis=1)
    )
    if not mask.any():
        inside = shapely.contains_xy(poly, px, py)
        if inside.any():
            mask = inside[mesh.tri].any(axis=1)
    if not mask.any():
        raise CavityError(
            "polygon selects no elements and encloses no nodes (is it over land?)"
        )
    return clean_selection(mesh.tri, mask, mesh.n_nodes)


def order_loops(ring_edges: np.ndarray) -> list[np.ndarray]:
    """Ordered closed loops from boundary edges (2 edges per node required).

    Deterministic: loops are discovered from their smallest node id and
    walked toward its smaller-id neighbour first.
    """
    nbr: dict[int, list[int]] = {}
    for a, b in ring_edges.tolist():
        nbr.setdefault(a, []).append(b)
        nbr.setdefault(b, []).append(a)
    for node, ns in nbr.items():
        if len(ns) != 2:
            raise CavityError(
                f"boundary node {node} has {len(ns)} boundary edges — "
                "selection was not cleaned"
            )
    unvisited = set(nbr)
    loops = []
    while unvisited:
        start = min(unvisited)
        loop, prev, cur = [start], -1, start
        cur = min(nbr[start])  # deterministic walk direction
        loop.append(cur)
        prev = start
        while cur != start:
            a, b = nbr[cur]
            prev, cur = cur, (b if a == prev else a)
            loop.append(cur)
        loop.pop()  # closing node repeats the start
        loops.append(np.asarray(loop, dtype=np.int64))
        unvisited -= set(loop)
    return loops


def extract_cavity(
    mesh: Mesh, ring_lonlat=None, *, elem_mask: np.ndarray | None = None
) -> Cavity:
    """Extract the cavity selected by a polygon (or an explicit element mask).

    Raises `CavityError` for empty or whole-mesh selections and for
    selections that cannot be reduced to simple boundary loops.
    """
    if elem_mask is None:
        if ring_lonlat is None:
            raise CavityError("either a polygon or an element mask is required")
        elem_mask = select_elements(mesh, ring_lonlat)
    else:
        elem_mask = np.asarray(elem_mask, dtype=bool)
        if elem_mask.shape != (mesh.n_tri,):
            raise CavityError("element mask must have one entry per element")
        if not elem_mask.any():
            raise CavityError("element mask selects no elements")
        elem_mask = clean_selection(mesh.tri, elem_mask, mesh.n_nodes)
    if elem_mask.all():
        raise CavityError(
            "selection covers the entire mesh — nothing to stitch to; "
            "use a global rebuild instead"
        )

    ring_e = boundary_edges(mesh.tri[elem_mask], mesh.n_nodes)
    loops = order_loops(ring_e)
    ring_nodes = np.concatenate(loops)
    loop_starts = np.cumsum([0] + [loop.size for loop in loops])

    edges_local = []
    for k, loop in enumerate(loops):
        i = np.arange(loop.size)
        edges_local.append(
            np.column_stack([loop_starts[k] + i, loop_starts[k] + (i + 1) % loop.size])
        )
    ring_edges = np.vstack(edges_local).astype(np.int64)

    polygon_xy = None
    if ring_lonlat is not None:
        poly, lon0, lat0 = _polygon_plane(ring_lonlat)
        polygon_xy = np.asarray(poly.exterior.coords, dtype=np.float64)
    else:
        lon0, lat0 = ring_center(mesh.lon[ring_nodes], mesh.lat[ring_nodes])
    x, y = stereo_xy(mesh.lon[ring_nodes], mesh.lat[ring_nodes], lon0, lat0)
    ring_xy = np.column_stack([x, y])
    edge_len = np.hypot(*(ring_xy[ring_edges[:, 0]] - ring_xy[ring_edges[:, 1]]).T)

    # coast edges = ring edges that are boundary edges of the FULL mesh
    keys, counts = np.unique(_edge_keys(mesh.tri, mesh.n_nodes), return_counts=True)
    coast_keys = keys[counts == 1]
    ge = np.sort(ring_nodes[ring_edges], axis=1).astype(np.int64)
    ring_keys = ge[:, 0].astype(np.uint64) * np.uint64(mesh.n_nodes) + ge[:, 1].astype(
        np.uint64
    )
    edge_is_coast = np.isin(ring_keys, coast_keys)

    return Cavity(
        elem_mask=elem_mask,
        lon0=float(lon0),
        lat0=float(lat0),
        ring_nodes=ring_nodes,
        loop_starts=np.asarray(loop_starts, dtype=np.int64),
        ring_xy=ring_xy,
        ring_edges=ring_edges,
        edge_len_km=edge_len,
        edge_is_coast=edge_is_coast,
        polygon_xy=polygon_xy,
    )
