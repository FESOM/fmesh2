"""Vectorized mesh topology: edges, adjacency, boundary, spatial index."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

EARTH_RADIUS_KM = 6371.0


def sphere_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Unit-sphere cartesian coordinates, (n, 3) float64."""
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    out = np.empty((lon.shape[0], 3), np.float64)
    coslat = np.cos(lat)
    out[:, 0] = coslat * np.cos(lon)
    out[:, 1] = coslat * np.sin(lon)
    out[:, 2] = np.sin(lat)
    return out


def chord_to_arc_km(chord: np.ndarray) -> np.ndarray:
    """Great-circle distance from unit-sphere chord length."""
    return 2.0 * np.arcsin(np.clip(chord / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KM


def _edge_keys(tri: np.ndarray, n_nodes: int) -> np.ndarray:
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).astype(np.int64)
    e.sort(axis=1)
    return e[:, 0].astype(np.uint64) * np.uint64(n_nodes) + e[:, 1].astype(np.uint64)


def _decode_keys(keys: np.ndarray, n_nodes: int) -> np.ndarray:
    e0 = (keys // np.uint64(n_nodes)).astype(np.int64)
    e1 = (keys % np.uint64(n_nodes)).astype(np.int64)
    return np.column_stack([e0, e1])


def unique_edges(tri: np.ndarray, n_nodes: int) -> np.ndarray:
    """Unique undirected edges as (e, 2) int64, each row sorted ascending."""
    return _decode_keys(np.unique(_edge_keys(tri, n_nodes)), n_nodes)


def boundary_edges(tri: np.ndarray, n_nodes: int) -> np.ndarray:
    """Edges adjacent to exactly one element (mesh boundary, i.e. coast)."""
    keys, counts = np.unique(_edge_keys(tri, n_nodes), return_counts=True)
    return _decode_keys(keys[counts == 1], n_nodes)


def boundary_nodes(tri: np.ndarray, n_nodes: int) -> np.ndarray:
    """Boolean mask of nodes lying on the mesh boundary."""
    mask = np.zeros(n_nodes, dtype=bool)
    be = boundary_edges(tri, n_nodes)
    mask[be.ravel()] = True
    return mask


def node_elem_csr(tri: np.ndarray, n_nodes: int) -> tuple[np.ndarray, np.ndarray]:
    """CSR arrays (indptr, elem_indices): elements incident to each node."""
    nodes = tri.ravel()
    elems = np.repeat(np.arange(tri.shape[0], dtype=np.int64), 3)
    order = np.argsort(nodes, kind="stable")
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(np.bincount(nodes, minlength=n_nodes), out=indptr[1:])
    return indptr, elems[order]


class SphereKDTree:
    """KDTree over unit-sphere points; distances in km along great circles."""

    def __init__(self, lon_deg: np.ndarray, lat_deg: np.ndarray):
        self._tree = cKDTree(sphere_xyz(lon_deg, lat_deg))

    def nearest_km(self, lon_deg, lat_deg) -> tuple[np.ndarray, np.ndarray]:
        """(distance_km, index) of the nearest tree point for each query."""
        q = sphere_xyz(np.atleast_1d(lon_deg), np.atleast_1d(lat_deg))
        chord, idx = self._tree.query(q, workers=-1)
        return chord_to_arc_km(chord), idx

    def query_radius(self, lon_deg, lat_deg, radius_km: float) -> list[np.ndarray]:
        q = sphere_xyz(np.atleast_1d(lon_deg), np.atleast_1d(lat_deg))
        chord = 2.0 * np.sin(min(radius_km / EARTH_RADIUS_KM, np.pi) / 2.0)
        return [np.asarray(r) for r in self._tree.query_ball_point(q, chord)]
