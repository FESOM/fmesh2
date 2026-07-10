"""Vectorized post-processing of a raw jigsaw mesh into a FESOM2 ocean mesh.

Ports fmesh `cut_land` without Python dict loops:

1. sample topography at nodes -> land mask; drop every element touching land
2. iterative loose-element cleanup (fmesh valence rules, evaluated per pass)
3. lake elimination: `connected_components` on the node graph, keeping the
   union of components containing the ocean seeds (fmesh: BFS from ~0E 0N)
4. drop unused nodes, renumber contiguously
5. node depths = |elevation| clamped to `minimum_depth`
6. coast flags from topological boundary edges

Pure numpy end-to-end: the same raw mesh always yields a bit-identical
result (the reproducibility contract).
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from fmesh2.core.mesh import Mesh
from fmesh2.core.topology import SphereKDTree, boundary_nodes
from fmesh2.data.topo import Topo

from .jigsaw_driver import RawMesh


def _neighbor_and_valence(tri: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-node: number of distinct neighbor nodes, and incident-element count."""
    valence = np.bincount(tri.ravel(), minlength=n_nodes)
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    pairs = np.concatenate([edges, edges[:, ::-1]])  # both directions
    keys = pairs[:, 0].astype(np.uint64) * np.uint64(n_nodes) + pairs[:, 1].astype(np.uint64)
    uniq = np.unique(keys)
    neighbors = np.bincount((uniq // np.uint64(n_nodes)).astype(np.int64), minlength=n_nodes)
    return neighbors, valence


def drop_loose_elements(tri: np.ndarray, n_nodes: int, max_valence: int = 6) -> np.ndarray:
    """fmesh loose-element rules, applied simultaneously per pass until stable.

    A node is deleted (with its incident elements) when its incident-element
    count `v` and distinct-node count `d` (neighbors + itself) satisfy:
    ``(v == 1 and d == 3) or (v == 2 and d == 5) or (v >= 3 and d >= v + 3)``.
    """
    tri = tri.copy()
    while tri.size:
        neighbors, valence = _neighbor_and_valence(tri, n_nodes)
        distinct = neighbors + 1  # incident nodes include the node itself
        v, d = valence, distinct
        kill = (
            ((v == 1) & (d == 3))
            | ((v == 2) & (d == 5))
            | ((v >= 3) & (v <= max_valence) & (d >= v + 3))
        )
        if not kill.any():
            break
        tri = tri[~kill[tri].any(axis=1)]
    return tri


def keep_seed_components(
    tri: np.ndarray, lon: np.ndarray, lat: np.ndarray, seeds
) -> np.ndarray:
    """Drop elements not node-connected to any ocean seed (lake elimination)."""
    if not tri.size:
        raise ValueError("mesh is empty before lake elimination")
    n = lon.size
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(edges.shape[0], dtype=np.int8), (edges[:, 0], edges[:, 1])), shape=(n, n)
    )
    _, labels = connected_components(graph, directed=False)

    used = np.zeros(n, dtype=bool)
    used[tri.ravel()] = True
    used_idx = np.flatnonzero(used)
    tree = SphereKDTree(lon[used_idx], lat[used_idx])
    seed_lon = np.array([s[0] for s in seeds], dtype=np.float64)
    seed_lat = np.array([s[1] for s in seeds], dtype=np.float64)
    _, nearest = tree.nearest_km(seed_lon, seed_lat)
    keep_labels = np.unique(labels[used_idx[nearest]])
    return tri[np.isin(labels[tri[:, 0]], keep_labels)]


def postprocess(
    raw: RawMesh,
    topo: Topo,
    minimum_depth: float,
    seeds=((0.0, 0.0),),
    levels=None,
) -> Mesh:
    elev = topo.elevation(raw.lon, raw.lat)

    land = elev >= 0.0  # fmesh: depth >= 0 is land
    tri = raw.tri[~land[raw.tri].any(axis=1)]
    if not tri.size:
        raise ValueError("no ocean elements survive the land cut — check the topography")

    tri = drop_loose_elements(tri, raw.n_nodes)
    tri = keep_seed_components(tri, raw.lon, raw.lat, seeds)
    tri = drop_loose_elements(tri, raw.n_nodes)  # lakes may expose new loose nodes

    keep = np.zeros(raw.n_nodes, dtype=bool)
    keep[tri.ravel()] = True
    new_index = np.cumsum(keep) - 1
    tri_new = new_index[tri].astype(np.int32)
    lon_new = raw.lon[keep]
    lat_new = raw.lat[keep]

    depth = np.abs(elev[keep])
    depth[elev[keep] >= -minimum_depth] = minimum_depth

    coast = boundary_nodes(tri_new, int(keep.sum())).astype(np.int8)
    return Mesh(
        lon=lon_new,
        lat=lat_new,
        tri=tri_new,
        depth=depth,
        coast_flag=coast,
        levels=None if levels is None else np.asarray(levels, dtype=np.float64),
    )
