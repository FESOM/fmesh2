"""Node-level editing operations: move and Laplacian smoothing (M4).

Each operation validates the candidate positions against element inversion
(signed orientation in the tangent frame must not flip for any incident
element), mutates the mesh in place, and returns an `EditOp` carrying the
inverse (previous positions/depths) for undo plus a quality-delta report.
All math is pure numpy, so ops are deterministic — the journal-replay
reproducibility guarantee (`edit.journal`) rests on that.

Depths of moved nodes are re-sampled from `topo` when one is given, using
the same `minimum_depth` clamp as the generation postprocess and matching
the host mesh's sign convention (core2 stores depths negative-down; the M2
pipeline writes positive-down).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmesh2.core.mesh import Mesh
from fmesh2.core.metrics import min_angle_deg, signed_orientation
from fmesh2.core.topology import (
    boundary_nodes,
    chord_to_arc_km,
    sphere_xyz,
    unique_edges,
)


class EditError(ValueError):
    """The operation is invalid or would corrupt the mesh.

    `element_ids` holds the offending global element indices when the
    rejection is an inversion safeguard.
    """

    def __init__(self, message: str, element_ids: np.ndarray | None = None):
        super().__init__(message)
        self.element_ids = (
            np.asarray(element_ids, dtype=np.int64)
            if element_ids is not None
            else np.empty(0, dtype=np.int64)
        )


@dataclass
class EditOp:
    """One applied edit: the moved nodes, their inverse, and a report.

    `before`/`after` are (n, 2) lon/lat degrees for `ids`; depths are per
    moved node or None when the mesh has no depth array. `after` is
    authoritative for journal replay — smoothing is never recomputed.
    """

    kind: str
    ids: np.ndarray
    before: np.ndarray
    after: np.ndarray
    before_depth: np.ndarray | None
    after_depth: np.ndarray | None
    params: dict
    report: dict


def _validate_ids(mesh: Mesh, ids) -> np.ndarray:
    ids = np.asarray(ids, dtype=np.int64).ravel()
    if ids.size == 0:
        raise EditError("no nodes selected")
    if np.unique(ids).size != ids.size:
        raise EditError("duplicate node ids in selection")
    if ids.min() < 0 or ids.max() >= mesh.n_nodes:
        raise EditError(
            f"node ids out of range 0..{mesh.n_nodes - 1}: "
            f"[{ids.min()}, {ids.max()}]"
        )
    return ids


def _affected_elements(mesh: Mesh, ids: np.ndarray) -> np.ndarray:
    moved = np.zeros(mesh.n_nodes, dtype=bool)
    moved[ids] = True
    return np.flatnonzero(moved[mesh.tri].any(axis=1))


def _lonlat_from_xyz(xyz: np.ndarray, host_lon: np.ndarray) -> np.ndarray:
    lon = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(xyz[:, 2], -1.0, 1.0)))
    if host_lon.size and host_lon.min() >= 0.0 and host_lon.max() > 180.0:
        lon %= 360.0  # host mesh uses the [0, 360) convention
    return np.column_stack([lon, lat])


def _resampled_depth(mesh: Mesh, new_lonlat, topo, minimum_depth: float):
    """Topo-sampled depths at the new positions, in the host sign convention."""
    elev = topo.elevation(new_lonlat[:, 0], new_lonlat[:, 1])
    depth = np.abs(elev)
    depth[elev >= -minimum_depth] = minimum_depth
    if np.nanmedian(mesh.depth) < 0:
        depth = -depth
    return depth


def _apply(
    mesh: Mesh,
    ids: np.ndarray,
    new_lonlat: np.ndarray,
    kind: str,
    params: dict,
    topo,
    minimum_depth: float,
) -> EditOp:
    """Inversion safeguard, quality report, depth resample, then mutate."""
    aff = _affected_elements(mesh, ids)
    tri_aff = mesh.tri[aff]
    xyz = sphere_xyz(mesh.lon, mesh.lat)
    old_xyz = xyz[ids].copy()
    orient_before = signed_orientation(xyz, tri_aff)
    angle_before = min_angle_deg(xyz, tri_aff)

    xyz[ids] = sphere_xyz(new_lonlat[:, 0], new_lonlat[:, 1])
    orient_after = signed_orientation(xyz, tri_aff)
    angle_after = min_angle_deg(xyz, tri_aff)

    sign = np.sign(orient_before)
    # healthy elements must keep their orientation strictly; elements that
    # were already degenerate only need to stay non-degenerate
    inverted = aff[np.where(sign == 0.0, orient_after == 0.0, sign * orient_after <= 0.0)]
    if inverted.size:
        shown = ", ".join(map(str, inverted[:10]))
        more = "..." if inverted.size > 10 else ""
        raise EditError(
            f"{kind} would invert {inverted.size} element(s): {shown}{more}",
            element_ids=inverted,
        )

    before = np.column_stack([mesh.lon[ids].copy(), mesh.lat[ids].copy()])
    before_depth = after_depth = None
    if mesh.depth is not None and topo is not None:
        before_depth = mesh.depth[ids].copy()
        after_depth = _resampled_depth(mesh, new_lonlat, topo, minimum_depth)

    mesh.lon[ids] = new_lonlat[:, 0]
    mesh.lat[ids] = new_lonlat[:, 1]
    if after_depth is not None:
        mesh.depth[ids] = after_depth

    disp_km = chord_to_arc_km(np.linalg.norm(xyz[ids] - old_xyz, axis=1))
    report = {
        "n_nodes": int(ids.size),
        "n_elements": int(aff.size),
        "max_displacement_km": float(disp_km.max()) if disp_km.size else 0.0,
        "min_angle_before_deg": float(angle_before.min()) if aff.size else None,
        "min_angle_after_deg": float(angle_after.min()) if aff.size else None,
        "median_min_angle_before_deg": float(np.median(angle_before)) if aff.size else None,
        "median_min_angle_after_deg": float(np.median(angle_after)) if aff.size else None,
    }
    return EditOp(
        kind=kind,
        ids=ids,
        before=before,
        after=new_lonlat.copy(),
        before_depth=before_depth,
        after_depth=after_depth,
        params=params,
        report=report,
    )


def move_nodes(
    mesh: Mesh,
    ids,
    new_lonlat,
    *,
    topo=None,
    minimum_depth: float = 5.0,
) -> EditOp:
    """Move nodes to explicit lon/lat targets (degrees), in place.

    Rejects (leaving the mesh untouched) when any incident element would
    invert or degenerate. Coast nodes may be moved — hand-fixing coastal
    nodes is a primary use case.
    """
    ids = _validate_ids(mesh, ids)
    new_lonlat = np.asarray(new_lonlat, dtype=np.float64)
    if new_lonlat.shape != (ids.size, 2):
        raise EditError(
            f"new_lonlat must have shape ({ids.size}, 2), got {new_lonlat.shape}"
        )
    if not np.isfinite(new_lonlat).all():
        raise EditError("target positions must be finite")
    if np.abs(new_lonlat[:, 1]).max() > 90.0:
        raise EditError("target latitude out of range [-90, 90]")
    return _apply(mesh, ids, new_lonlat, "move", {}, topo, minimum_depth)


def smooth_patch(
    mesh: Mesh,
    ids,
    iterations: int = 1,
    strength: float = 0.5,
    *,
    topo=None,
    minimum_depth: float = 5.0,
) -> EditOp:
    """Laplacian-smooth the selection interior on the unit sphere, in place.

    Mesh-boundary (coast) nodes in the selection are anchored — smoothing
    must never reshape the coastline. Each iteration pulls every remaining
    selected node toward the centroid of ALL its neighbours (selected or
    not) by `strength`, re-projected onto the sphere; unselected neighbours
    act as fixed anchors, so the patch blends into its surroundings.
    """
    ids = _validate_ids(mesh, ids)
    if not 0.0 < strength <= 1.0:
        raise EditError(f"strength must be in (0, 1], got {strength}")
    if iterations < 1:
        raise EditError(f"iterations must be >= 1, got {iterations}")

    edges = unique_edges(mesh.tri, mesh.n_nodes)
    deg = np.bincount(edges.ravel(), minlength=mesh.n_nodes)
    anchored = boundary_nodes(mesh.tri, mesh.n_nodes)
    move = ids[~anchored[ids] & (deg[ids] > 0)]
    if move.size == 0:
        raise EditError(
            "selection has no smoothable interior "
            "(all nodes on the mesh boundary)"
        )

    xyz = sphere_xyz(mesh.lon, mesh.lat)
    for _ in range(int(iterations)):
        sums = np.zeros_like(xyz)
        for k in range(3):
            sums[:, k] = np.bincount(
                edges[:, 0], weights=xyz[edges[:, 1], k], minlength=mesh.n_nodes
            )
            sums[:, k] += np.bincount(
                edges[:, 1], weights=xyz[edges[:, 0], k], minlength=mesh.n_nodes
            )
        centroid = sums[move] / deg[move, None]
        p = xyz[move] + strength * (centroid - xyz[move])
        xyz[move] = p / np.linalg.norm(p, axis=1, keepdims=True)

    new_lonlat = _lonlat_from_xyz(xyz[move], mesh.lon)
    params = {
        "iterations": int(iterations),
        "strength": float(strength),
        "n_requested": int(ids.size),
    }
    return _apply(mesh, move, new_lonlat, "smooth", params, topo, minimum_depth)


def patch_roughness_km(mesh: Mesh, ids) -> float:
    """RMS distance (km) of each selected node from its neighbours' centroid.

    The metric Laplacian smoothing minimizes; used to verify that smoothing
    actually smooths and as a UI readout.
    """
    ids = _validate_ids(mesh, ids)
    edges = unique_edges(mesh.tri, mesh.n_nodes)
    xyz = sphere_xyz(mesh.lon, mesh.lat)
    sums = np.zeros_like(xyz)
    for k in range(3):
        sums[:, k] = np.bincount(
            edges[:, 0], weights=xyz[edges[:, 1], k], minlength=mesh.n_nodes
        )
        sums[:, k] += np.bincount(
            edges[:, 1], weights=xyz[edges[:, 0], k], minlength=mesh.n_nodes
        )
    deg = np.bincount(edges.ravel(), minlength=mesh.n_nodes)
    ids = ids[deg[ids] > 0]
    centroid = sums[ids] / deg[ids, None]
    chord = np.linalg.norm(xyz[ids] - centroid, axis=1)
    return float(np.sqrt(np.mean(chord_to_arc_km(chord) ** 2)))
