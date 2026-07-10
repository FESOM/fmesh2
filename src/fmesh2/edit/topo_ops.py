"""Topology-changing hand edits: delete and add nodes (v1.1).

Unlike move/smooth these change node and element counts. Deleting a node
removes its element star and re-triangulates the hole (for a boundary/coast
node the coastline is locally straightened to the chord between its two
boundary neighbours — the classic hand-widening of straits). Adding a node
either splits the containing element in three, or — clicked just outside
the mesh — attaches a new triangle to the nearest boundary edge ("digging"
ocean into land).

Every op is a list of sequential `TopologyStep` deltas (one node each),
recorded in the numbering of the state it was applied to, with enough
payload to undo bit-exact: the removed node rows, the removed triangle
rows WITH their row positions, the appended triangle rows, and coast-flag
changes of affected surviving nodes. `apply_step` / `undo_step` are the
single source of truth for the mutation — live edits, journal replay,
undo and redo all run the same code, so `mesh_hash` reproducibility holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fmesh2.core.mesh import Mesh
from fmesh2.core.projection import stereo_xy
from fmesh2.core.topology import boundary_nodes, sphere_xyz

from .ops import EditError, _validate_ids

#: reject fill triangles thinner than this (degrees, in the tangent plane)
MIN_FILL_ANGLE_DEG = 0.5


@dataclass
class TopologyStep:
    """One node's worth of topology change, invertible bit-exact."""

    kind: str  # "delete" | "add"
    node_ids: np.ndarray  # node-table rows removed (delete) / appended (add)
    node_lonlat: np.ndarray
    node_depth: np.ndarray | None
    node_coast: np.ndarray | None
    tri_removed_idx: np.ndarray  # ascending row positions in the pre-step table
    tri_removed: np.ndarray  # (r, 3) pre-step node ids
    tri_added: np.ndarray  # (a, 3) appended rows, pre-step node ids
    # coast flags of surviving nodes that flip (pre-step ids); None until the
    # step is first applied live, then recorded for deterministic replay
    coast_ids: np.ndarray | None = None
    coast_before: np.ndarray | None = None
    coast_after: np.ndarray | None = None


@dataclass
class TopologyOp:
    """A journaled delete/add operation (sequential steps)."""

    kind: str
    steps: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


# ------------------------------------------------------------- step engine

def _reinsert_rows(arr: np.ndarray, idx: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Rebuild `arr` with `rows` restored at ascending row positions `idx`."""
    out = np.empty((arr.shape[0] + len(idx), *arr.shape[1:]), dtype=arr.dtype)
    mask = np.ones(out.shape[0], dtype=bool)
    mask[idx] = False
    out[idx] = rows
    out[mask] = arr
    return out


def _reinsert_values(arr: np.ndarray, idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.empty(arr.shape[0] + len(idx), dtype=arr.dtype)
    mask = np.ones(out.shape[0], dtype=bool)
    mask[idx] = False
    out[idx] = values
    out[mask] = arr
    return out


def apply_step(mesh: Mesh, step: TopologyStep) -> None:
    """Apply one recorded delta; fills in the coast delta on first (live) use."""
    tri = np.delete(np.asarray(mesh.tri, dtype=np.int64), step.tri_removed_idx, axis=0)
    if step.tri_added.size:
        tri = np.vstack([tri, step.tri_added.astype(np.int64)])

    if step.kind == "add":
        mesh.lon = np.append(mesh.lon, step.node_lonlat[:, 0])
        mesh.lat = np.append(mesh.lat, step.node_lonlat[:, 1])
        if mesh.depth is not None:
            mesh.depth = np.append(mesh.depth, step.node_depth)
        if mesh.coast_flag is not None:
            mesh.coast_flag = np.append(mesh.coast_flag, step.node_coast)
        coast_map = None  # ids unchanged by an append
    else:
        keep = np.ones(mesh.n_nodes, dtype=bool)
        keep[step.node_ids] = False
        mapping = np.cumsum(keep) - 1  # old id -> new id (valid where kept)
        tri = mapping[tri]
        mesh.lon = mesh.lon[keep]
        mesh.lat = mesh.lat[keep]
        if mesh.depth is not None:
            mesh.depth = mesh.depth[keep]
        if mesh.coast_flag is not None:
            mesh.coast_flag = mesh.coast_flag[keep]
        coast_map = mapping
    mesh.tri = tri.astype(np.int32)

    if mesh.coast_flag is None:
        return
    if step.coast_ids is None:
        # live first application: recompute flags of the affected nodes and
        # record the delta so replay never has to recompute
        affected = np.unique(
            np.concatenate([step.tri_removed.ravel(), step.tri_added.ravel()])
        )
        affected = affected[~np.isin(affected, step.node_ids)]
        now_ids = coast_map[affected] if coast_map is not None else affected
        on_b = boundary_nodes(mesh.tri, mesh.n_nodes)[now_ids].astype(np.int8)
        changed = mesh.coast_flag[now_ids] != on_b
        step.coast_ids = affected[changed]
        step.coast_before = mesh.coast_flag[now_ids[changed]].copy()
        step.coast_after = on_b[changed]
        mesh.coast_flag[now_ids[changed]] = on_b[changed]
    elif step.coast_ids.size:
        now_ids = coast_map[step.coast_ids] if coast_map is not None else step.coast_ids
        mesh.coast_flag[now_ids] = step.coast_after


def undo_step(mesh: Mesh, step: TopologyStep) -> None:
    """Exact inverse of `apply_step` (only valid as the latest applied step)."""
    tri = np.asarray(mesh.tri, dtype=np.int64)
    if step.kind == "add":
        if step.tri_added.size:
            tri = tri[: -len(step.tri_added)]
        tri = _reinsert_rows(tri, step.tri_removed_idx, step.tri_removed.astype(np.int64))
        k = len(step.node_ids)
        mesh.lon = mesh.lon[:-k]
        mesh.lat = mesh.lat[:-k]
        if mesh.depth is not None:
            mesh.depth = mesh.depth[:-k]
        if mesh.coast_flag is not None:
            mesh.coast_flag = mesh.coast_flag[:-k]
    else:
        # restore node rows at their original positions, un-renumber, then
        # swap the fill rows back for the original star rows
        old_of_new = np.setdiff1d(
            np.arange(mesh.n_nodes + len(step.node_ids)), step.node_ids
        )
        tri = old_of_new[tri]
        if step.tri_added.size:
            tri = tri[: -len(step.tri_added)]
        tri = _reinsert_rows(tri, step.tri_removed_idx, step.tri_removed.astype(np.int64))
        mesh.lon = _reinsert_values(mesh.lon, step.node_ids, step.node_lonlat[:, 0])
        mesh.lat = _reinsert_values(mesh.lat, step.node_ids, step.node_lonlat[:, 1])
        if mesh.depth is not None:
            mesh.depth = _reinsert_values(mesh.depth, step.node_ids, step.node_depth)
        if mesh.coast_flag is not None:
            mesh.coast_flag = _reinsert_values(
                mesh.coast_flag, step.node_ids, step.node_coast
            )
    mesh.tri = tri.astype(np.int32)
    if mesh.coast_flag is not None and step.coast_ids is not None and step.coast_ids.size:
        mesh.coast_flag[step.coast_ids] = step.coast_before


# ---------------------------------------------------------------- geometry

def _min_angle_2d(p0, p1, p2) -> float:
    ang = []
    for a, b, c in ((p0, p1, p2), (p1, p2, p0), (p2, p0, p1)):
        u, v = b - a, c - a
        cosv = np.dot(u, v) / max(np.linalg.norm(u) * np.linalg.norm(v), 1e-300)
        ang.append(np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0))))
    return float(min(ang))


def _ear_clip(poly_xy: np.ndarray, node_label: int) -> tuple[np.ndarray, float]:
    """Best-quality-first ear clipping of a simple polygon.

    Returns (triangles as indices into poly_xy, min fill angle). Triangle
    orientation follows the polygon's own orientation, which callers derive
    from the mesh triangles — so fills stay consistent with the mesh.
    """
    m = poly_xy.shape[0]
    idx = list(range(m))
    # polygon orientation sign (shoelace)
    x, y = poly_xy[:, 0], poly_xy[:, 1]
    orient = np.sign(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    if orient == 0:
        raise EditError(f"degenerate hole around node {node_label} — cannot re-triangulate")
    tris: list[tuple[int, int, int]] = []
    worst = 180.0
    while len(idx) > 3:
        best, best_q = -1, -1.0
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            p0, p1, p2 = poly_xy[i0], poly_xy[i1], poly_xy[i2]
            cross = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0])
            if cross * orient <= 0:
                continue  # reflex corner
            # no other active vertex inside the candidate ear
            others = [j for j in idx if j not in (i0, i1, i2)]
            if others and _any_inside(poly_xy[others], p0, p1, p2):
                continue
            q = _min_angle_2d(p0, p1, p2)
            if q > best_q:
                best, best_q = k, q
        if best < 0:
            raise EditError(
                f"cannot re-triangulate the hole around node {node_label} "
                "(non-simple neighbourhood)"
            )
        k = best
        tris.append((idx[k - 1], idx[k], idx[(k + 1) % len(idx)]))
        worst = min(worst, best_q)
        del idx[k]
    p0, p1, p2 = poly_xy[idx[0]], poly_xy[idx[1]], poly_xy[idx[2]]
    tris.append((idx[0], idx[1], idx[2]))
    worst = min(worst, _min_angle_2d(p0, p1, p2))
    if worst < MIN_FILL_ANGLE_DEG:
        raise EditError(
            f"re-triangulating around node {node_label} would create a "
            f"degenerate element (min angle {worst:.2f} deg)"
        )
    return np.asarray(tris, dtype=np.int64), worst


def _any_inside(pts: np.ndarray, p0, p1, p2) -> bool:
    d = np.empty((len(pts), 3))
    for k, (a, b) in enumerate(((p0, p1), (p1, p2), (p2, p0))):
        d[:, k] = (b[0] - a[0]) * (pts[:, 1] - a[1]) - (b[1] - a[1]) * (pts[:, 0] - a[0])
    return bool(((d > 1e-14).all(axis=1) | (d < -1e-14).all(axis=1)).any())


def _link_chain(tri: np.ndarray, n: int) -> tuple[np.ndarray, list[int], bool]:
    """Star rows and the ordered link of node n.

    Returns (star row indices, ordered link node ids, closed?). The link is
    directed by the star triangles' own orientation, so fills inherit the
    mesh convention. Raises for pinched/non-manifold neighbourhoods.
    """
    star = np.flatnonzero((tri == n).any(axis=1))
    directed: dict[int, int] = {}
    for t in tri[star]:
        a, b, c = int(t[0]), int(t[1]), int(t[2])
        if a == n:
            e = (b, c)
        elif b == n:
            e = (c, a)
        else:
            e = (a, b)
        if e[0] in directed:
            raise EditError(f"node {n} has a non-manifold neighbourhood")
        directed[e[0]] = e[1]
    if not directed:
        return star, [], True  # isolated node: nothing to re-triangulate
    heads = set(directed) - set(directed.values())
    if len(heads) > 1:
        raise EditError(
            f"deleting node {n} would pinch the mesh apart "
            "(its elements touch it in more than one fan)"
        )
    closed = not heads
    start = next(iter(heads)) if heads else next(iter(directed))
    chain = [start]
    cur = start
    while cur in directed:
        cur = directed.pop(cur)
        if cur == start:
            break
        chain.append(cur)
    if directed:
        raise EditError(
            f"deleting node {n} would disconnect the mesh "
            "(its element fan is not a single piece)"
        )
    return star, chain, closed


# --------------------------------------------------------------------- ops

def delete_nodes(mesh: Mesh, ids, *, allow_coast: bool = False) -> TopologyOp:
    """Delete nodes: remove each element star and re-triangulate the hole.

    Boundary (coast) nodes need `allow_coast=True`; deleting one replaces
    its stretch of coastline with the chord between its boundary
    neighbours. Applied sequentially per node (descending id, so earlier
    steps don't shift later targets); a failure rolls the whole op back.
    """
    ids = _validate_ids(mesh, ids)
    if ids.size >= mesh.n_nodes - 2:
        raise EditError("cannot delete (nearly) the whole mesh")
    if not allow_coast:
        on_b = boundary_nodes(mesh.tri, mesh.n_nodes)
        bad = ids[on_b[ids]]
        if bad.size:
            shown = ", ".join(map(str, bad[:8]))
            raise EditError(
                f"{bad.size} selected node(s) lie on the coastline ({shown}"
                f"{'...' if bad.size > 8 else ''}) — deleting them reshapes "
                "the coast; retry with allow_coast"
            )

    op = TopologyOp(
        kind="delete",
        params={"n_requested": int(ids.size), "allow_coast": bool(allow_coast)},
    )
    worst_fill = 180.0
    try:
        for n in sorted((int(i) for i in ids), reverse=True):
            star, chain, closed = _link_chain(mesh.tri, n)
            fill = np.empty((0, 3), dtype=np.int64)
            if len(chain) >= 3:
                x, y = stereo_xy(
                    mesh.lon[chain], mesh.lat[chain],
                    float(mesh.lon[n]), float(mesh.lat[n]),
                )
                local, q = _ear_clip(np.column_stack([x, y]), n)
                worst_fill = min(worst_fill, q)
                fill = np.asarray(chain, dtype=np.int64)[local]
            step = TopologyStep(
                kind="delete",
                node_ids=np.array([n], dtype=np.int64),
                node_lonlat=np.array([[mesh.lon[n], mesh.lat[n]]], dtype=np.float64),
                node_depth=(
                    np.array([mesh.depth[n]]) if mesh.depth is not None else None
                ),
                node_coast=(
                    np.array([mesh.coast_flag[n]], dtype=np.int8)
                    if mesh.coast_flag is not None
                    else None
                ),
                tri_removed_idx=star.astype(np.int64),
                tri_removed=mesh.tri[star].astype(np.int64),
                tri_added=fill,
            )
            apply_step(mesh, step)
            op.steps.append(step)
    except EditError:
        for step in reversed(op.steps):
            undo_step(mesh, step)
        raise
    op.report = {
        "n_deleted": int(ids.size),
        "n_tri_removed": int(sum(len(s.tri_removed) for s in op.steps)),
        "n_tri_added": int(sum(len(s.tri_added) for s in op.steps)),
        "min_fill_angle_deg": None if worst_fill == 180.0 else float(worst_fill),
        "n_nodes": mesh.n_nodes,
        "n_tri": mesh.n_tri,
    }
    return op


def add_node(
    mesh: Mesh,
    lon: float,
    lat: float,
    *,
    topo=None,
    minimum_depth: float = 5.0,
) -> TopologyOp:
    """Insert a node at (lon, lat), in place.

    Inside an element: split it in three. Outside the mesh but within two
    edge-lengths of a boundary edge: attach a new coast triangle to that
    edge (extends the ocean into land). Depth comes from `topo` when the
    session has one, else from interpolation (split) / the edge mean
    (attach).
    """
    from scipy.spatial import cKDTree

    lon, lat = float(lon), float(lat)
    if not (np.isfinite(lon) and np.isfinite(lat) and abs(lat) <= 90.0):
        raise EditError("invalid position for the new node")
    px, py = stereo_xy(mesh.lon, mesh.lat, lon, lat)
    tree = cKDTree(sphere_xyz(mesh.lon, mesh.lat))
    _, near = tree.query(sphere_xyz(np.array([lon]), np.array([lat]))[0],
                         k=min(24, mesh.n_nodes))
    near = np.atleast_1d(near)
    if np.hypot(px[near[0]], py[near[0]]) < 1e-9:
        raise EditError("a node already exists at this position")

    cand = np.flatnonzero(np.isin(mesh.tri, near).any(axis=1))
    split_elem = -1
    bary = None
    for e in cand:
        a, b, c = mesh.tri[e]
        d = np.array([
            px[b] * py[c] - px[c] * py[b],
            px[c] * py[a] - px[a] * py[c],
            px[a] * py[b] - px[b] * py[a],
        ])  # sub-areas vs the origin (= the click point)
        total = d.sum()
        if abs(total) < 1e-300:
            continue
        w = d / total
        if (w >= -1e-12).all():
            split_elem, bary = int(e), np.clip(w, 0.0, None)
            break

    new_id = mesh.n_nodes
    if split_elem >= 0:
        t = mesh.tri[split_elem].astype(np.int64)
        depth = None
        if mesh.depth is not None:
            depth = float(np.dot(bary / bary.sum(), mesh.depth[t]))
        tri_added = np.array(
            [[t[0], t[1], new_id], [t[1], t[2], new_id], [t[2], t[0], new_id]],
            dtype=np.int64,
        )
        step_kwargs = dict(
            tri_removed_idx=np.array([split_elem], dtype=np.int64),
            tri_removed=mesh.tri[[split_elem]].astype(np.int64),
            tri_added=tri_added,
        )
        coast_val, mode = 0, "split"
        detail = {"element": split_elem}
    else:
        # outside the mesh: attach to the nearest boundary edge
        from fmesh2.core.topology import boundary_edges

        be = boundary_edges(mesh.tri, mesh.n_nodes)
        be = be[np.isin(be, near).any(axis=1)]
        if not be.size:
            raise EditError("click is neither inside the mesh nor near its boundary")
        p0 = np.column_stack([px[be[:, 0]], py[be[:, 0]]])
        p1 = np.column_stack([px[be[:, 1]], py[be[:, 1]]])
        seg = p1 - p0
        t_par = np.clip(
            -np.einsum("ij,ij->i", p0, seg) / np.maximum(
                np.einsum("ij,ij->i", seg, seg), 1e-300
            ),
            0.0,
            1.0,
        )
        dist = np.linalg.norm(p0 + t_par[:, None] * seg, axis=1)
        k = int(np.argmin(dist))
        edge_len = float(np.linalg.norm(seg[k]))
        if dist[k] > 2.0 * edge_len:
            raise EditError(
                "click is neither inside the mesh nor near its boundary "
                f"(nearest coast edge is {dist[k]:.0f} km away)"
            )
        u, v = int(be[k, 0]), int(be[k, 1])
        owner = np.flatnonzero(
            (mesh.tri == u).any(axis=1) & (mesh.tri == v).any(axis=1)
        )[0]
        t = mesh.tri[owner]
        # directed (u -> v) as the owner uses it; the new triangle reverses it
        pairs = [(int(t[0]), int(t[1])), (int(t[1]), int(t[2])), (int(t[2]), int(t[0]))]
        if (u, v) not in pairs:
            u, v = v, u
        depth = None
        if mesh.depth is not None:
            depth = float((mesh.depth[u] + mesh.depth[v]) / 2.0)
        step_kwargs = dict(
            tri_removed_idx=np.empty(0, dtype=np.int64),
            tri_removed=np.empty((0, 3), dtype=np.int64),
            tri_added=np.array([[v, u, new_id]], dtype=np.int64),
        )
        coast_val, mode = 1, "attach"
        detail = {"edge": [u, v]}

    if topo is not None and mesh.depth is not None:
        from .ops import _resampled_depth

        depth = float(
            _resampled_depth(mesh, np.array([[lon, lat]]), topo, minimum_depth)[0]
        )

    step = TopologyStep(
        kind="add",
        node_ids=np.array([new_id], dtype=np.int64),
        node_lonlat=np.array([[lon, lat]], dtype=np.float64),
        node_depth=(np.array([depth]) if mesh.depth is not None else None),
        node_coast=(
            np.array([coast_val], dtype=np.int8) if mesh.coast_flag is not None else None
        ),
        **step_kwargs,
    )
    apply_step(mesh, step)
    op = TopologyOp(
        kind="add",
        steps=[step],
        params={"lon": lon, "lat": lat},
        report={
            "mode": mode,
            "new_node": new_id,
            "n_nodes": mesh.n_nodes,
            "n_tri": mesh.n_tri,
            **detail,
        },
    )
    return op
