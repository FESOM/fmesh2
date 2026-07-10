"""Per-node and per-element mesh metrics."""
from __future__ import annotations

import numpy as np

from fmesh2.core.mesh import Mesh
from fmesh2.core.topology import chord_to_arc_km, sphere_xyz, unique_edges


def edge_lengths_km(xyz: np.ndarray, edges: np.ndarray) -> np.ndarray:
    chord = np.linalg.norm(xyz[edges[:, 0]] - xyz[edges[:, 1]], axis=1)
    return chord_to_arc_km(chord)


def local_resolution_km(mesh: Mesh) -> np.ndarray:
    """Mean length (km) of the edges incident to each node.

    Computed on the true mesh topology; nodes outside any element get 0.
    """
    xyz = sphere_xyz(mesh.lon, mesh.lat)
    edges = unique_edges(mesh.tri, mesh.n_nodes)
    length = edge_lengths_km(xyz, edges)
    sums = np.bincount(edges[:, 0], weights=length, minlength=mesh.n_nodes)
    sums += np.bincount(edges[:, 1], weights=length, minlength=mesh.n_nodes)
    cnts = np.bincount(edges[:, 0], minlength=mesh.n_nodes)
    cnts += np.bincount(edges[:, 1], minlength=mesh.n_nodes)
    return (sums / np.maximum(cnts, 1)).astype(np.float64)


def min_angle_deg(xyz: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Smallest internal angle per element, computed from 3-D chords."""
    p = xyz[tri]  # (m, 3, 3)
    angles = np.empty((tri.shape[0], 3), np.float64)
    for i in range(3):
        a = p[:, (i + 1) % 3] - p[:, i]
        b = p[:, (i + 2) % 3] - p[:, i]
        cosang = np.einsum("ij,ij->i", a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )
        angles[:, i] = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return angles.min(axis=1)


def signed_orientation(xyz: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Positive where the element is CCW seen from outside the sphere.

    Cross product of two element edges dotted with the outward normal at the
    element centroid; used to detect inverted elements after node moves.
    """
    p = xyz[tri]
    normal = p.mean(axis=1)
    cross = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    return np.einsum("ij,ij->i", cross, normal)


def triangle_min_angle_deg(mesh: Mesh) -> np.ndarray:
    """`min_angle_deg` over the whole mesh."""
    return min_angle_deg(sphere_xyz(mesh.lon, mesh.lat), mesh.tri)


def triangle_signed_orientation(mesh: Mesh) -> np.ndarray:
    """`signed_orientation` over the whole mesh."""
    return signed_orientation(sphere_xyz(mesh.lon, mesh.lat), mesh.tri)
