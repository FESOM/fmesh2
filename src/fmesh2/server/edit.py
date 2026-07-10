"""Server-side edit sessions (M4).

An `EditSession` pairs the live `Mesh` of an opened viewer entry with its
edit `Journal` (kept in a temp dir until save-as copies it next to the
written FESOM triplet), a lazily rebuilt KDTree for brush selection, and
the duplicate-vertex map from the viewer buffers so position updates can
be expressed directly in GPU vertex indices (source node + its dateline
duplicate, with the +360-degree plane shift applied server-side — one
source of truth for the buffer conventions).
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import shapely

from fmesh2.core.io_fesom import read_fesom, write_fesom
from fmesh2.core.projection import stereo_xy
from fmesh2.core.topology import SphereKDTree
from fmesh2.edit.cavity import CavityError, _polygon_plane
from fmesh2.edit.journal import Journal, mesh_hash
from fmesh2.edit.ops import EditError, EditOp, move_nodes, smooth_patch
from fmesh2.edit.topo_ops import TopologyOp, add_node, delete_nodes
from fmesh2.viewer.buffers import ViewerBuffers, build_buffers


class EditSession:
    """Journaled editing of one opened mesh."""

    def __init__(
        self,
        mesh_dir: str | Path,
        buffers: ViewerBuffers,
        topo_path: str | Path | None = None,
    ):
        self.mesh_dir = Path(mesh_dir)
        self.mesh = read_fesom(self.mesh_dir)
        self._tmp = Path(tempfile.mkdtemp(prefix="fmesh2_edit_"))
        self.journal_path = self._tmp / "edits.jsonl"
        self.journal = Journal(self.journal_path, self.mesh)
        self.set_topo(topo_path)
        self._tree: SphereKDTree | None = None
        self._set_dup_map(buffers)

    def _set_dup_map(self, buffers: ViewerBuffers) -> None:
        n = self.mesh.n_nodes
        dup_src = np.asarray(buffers.dupsrc, dtype=np.int64)
        self._dup_vert_of = np.full(n, -1, dtype=np.int64)
        self._dup_vert_of[dup_src] = n + np.arange(dup_src.size)

    def rebuild_buffers(self, name: str) -> ViewerBuffers:
        """Fresh viewer buffers after a topology change (delete/add/undo)."""
        buffers = build_buffers(self.mesh, name=name)
        self._set_dup_map(buffers)
        self._tree = None
        return buffers

    def set_topo(self, topo_path: str | Path | None) -> None:
        """Set/replace the topography used to re-sample depths of moved nodes."""
        if topo_path is None:
            self.topo = None
            return
        from fmesh2.data.topo import Topo

        self.topo = Topo(topo_path)

    # ------------------------------------------------------------ selection

    def select_lasso(self, ring_lonlat) -> np.ndarray:
        """Node ids inside a lasso polygon (stereographic point-in-polygon)."""
        try:
            poly, lon0, lat0 = _polygon_plane(np.asarray(ring_lonlat, dtype=np.float64))
        except CavityError as exc:
            raise EditError(str(exc)) from exc
        px, py = stereo_xy(self.mesh.lon, self.mesh.lat, lon0, lat0)
        shapely.prepare(poly)
        return np.flatnonzero(shapely.contains_xy(poly, px, py))

    def select_brush(self, lon: float, lat: float, radius_km: float) -> np.ndarray:
        """Node ids within `radius_km` of a point (KDTree on live positions)."""
        if self._tree is None:
            self._tree = SphereKDTree(self.mesh.lon, self.mesh.lat)
        return np.sort(self._tree.query_radius(lon, lat, radius_km)[0])

    # ----------------------------------------------------------------- ops

    def move(self, ids, lonlat) -> EditOp:
        op = move_nodes(self.mesh, ids, lonlat, topo=self.topo)
        self.journal.record(op)
        self._tree = None
        return op

    def smooth(self, ids, iterations: int, strength: float) -> EditOp:
        op = smooth_patch(
            self.mesh, ids, iterations=iterations, strength=strength, topo=self.topo
        )
        self.journal.record(op)
        self._tree = None
        return op

    def delete(self, ids, allow_coast: bool) -> TopologyOp:
        op = delete_nodes(self.mesh, ids, allow_coast=allow_coast)
        self.journal.record(op)
        self._tree = None
        return op

    def add(self, lon: float, lat: float) -> TopologyOp:
        op = add_node(self.mesh, lon, lat, topo=self.topo)
        self.journal.record(op)
        self._tree = None
        return op

    def undo(self) -> EditOp | TopologyOp | None:
        op = self.journal.undo(self.mesh)
        if op is not None:
            self._tree = None
        return op

    def redo(self) -> EditOp | TopologyOp | None:
        op = self.journal.redo(self.mesh)
        if op is not None:
            self._tree = None
        return op

    # ------------------------------------------------------------- updates

    def vertex_updates(self, ids) -> dict:
        """GPU buffer updates for nodes `ids` at their CURRENT positions.

        Vertex indices include each node's dateline duplicate; positions
        follow the viewer conventions (sphere Y = north; plane lon shifted
        +360 deg for duplicates).
        """
        ids = np.asarray(ids, dtype=np.int64)
        lon = (self.mesh.lon[ids] + 180.0) % 360.0 - 180.0
        lat = self.mesh.lat[ids]

        dvert = self._dup_vert_of[ids]
        hasd = dvert >= 0
        verts = np.concatenate([ids, dvert[hasd]])
        lon_v = np.concatenate([lon, lon[hasd] + 360.0])
        lat_v = np.concatenate([lat, lat[hasd]])

        lon_r = np.deg2rad(lon_v)
        lat_r = np.deg2rad(lat_v)
        coslat = np.cos(lat_r)
        pos = np.column_stack(
            [coslat * np.cos(lon_r), np.sin(lat_r), -coslat * np.sin(lon_r)]
        )
        pos2 = np.column_stack([lon_r, lat_r, np.zeros_like(lon_r)])

        dep = None
        if self.mesh.depth is not None:
            d = np.abs(self.mesh.depth[ids])
            dep = np.concatenate([d, d[hasd]]).tolist()
        return {
            "verts": verts.tolist(),
            "pos": pos.tolist(),
            "pos2": pos2.tolist(),
            "dep": dep,
        }

    def journal_state(self) -> dict:
        return {
            "can_undo": self.journal.can_undo,
            "can_redo": self.journal.can_redo,
            "n_ops": len(self.journal.ops),
        }

    # ---------------------------------------------------------------- save

    def save(self, out_dir: str | Path) -> tuple[Path, str]:
        """Write the edited FESOM triplet + journal; returns (dir, hash)."""
        out = Path(out_dir).expanduser()
        if out.resolve() == self.mesh_dir.resolve():
            raise EditError(
                "refusing to overwrite the source mesh — save to a new directory"
            )
        write_fesom(self.mesh, out)
        shutil.copy2(self.journal_path, out / "edits.jsonl")
        return out, mesh_hash(self.mesh)
