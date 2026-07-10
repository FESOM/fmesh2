"""Append-only JSONL edit journal with undo/redo and deterministic replay.

The journal is the reproducibility record for hand edits (M4): a project's
`edits.jsonl` plus the pristine base mesh deterministically reproduce the
edited mesh, byte for byte (`apply_journal` + `mesh_hash`).

File format — one JSON object per line, never rewritten:

* header (first line): ``{"kind": "base", "hash": <mesh_hash>, ...}``
* op records: kind ``move``/``smooth`` with node ids, before/after lon/lat
  (and depths when a topo resample happened) and the op's params/report;
* markers: ``{"kind": "undo"}`` / ``{"kind": "redo"}``.

Replay scans the log with stack semantics (op pushes; undo pops onto the
redo stack; redo pushes back; a new op clears the redo stack) and applies
the surviving ops' stored `after` state in order — smoothing is never
recomputed, so replays survive future algorithm changes. Positions travel
through JSON bit-exact: `json` serializes floats with `repr` (shortest
round-trippable decimal) and parses back to the nearest double.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fmesh2.core.mesh import Mesh

from .ops import EditError, EditOp
from .topo_ops import TopologyOp, TopologyStep, apply_step, undo_step

_FORMAT_VERSION = 1

#: journal record kinds that change topology (node/element counts)
TOPOLOGY_KINDS = ("delete", "add")


def mesh_hash(mesh: Mesh) -> str:
    """sha256 over the mesh arrays (dtype/shape tagged), hex string.

    Matches read->write->read through the bit-exact FESOM ASCII I/O: the
    hash of a replayed mesh equals the hash of the live-edited one.
    """
    h = hashlib.sha256()
    for name in ("lon", "lat", "tri", "depth", "coast_flag", "levels"):
        arr = getattr(mesh, name)
        if arr is None:
            h.update(f"{name}:none;".encode())
            continue
        arr = np.ascontiguousarray(arr)
        h.update(f"{name}:{arr.dtype.str}:{arr.shape};".encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _step_to_dict(step: TopologyStep) -> dict:
    d = {
        "kind": step.kind,
        "node_ids": step.node_ids.tolist(),
        "node_lonlat": step.node_lonlat.tolist(),
        "tri_removed_idx": step.tri_removed_idx.tolist(),
        "tri_removed": step.tri_removed.tolist(),
        "tri_added": step.tri_added.tolist(),
    }
    if step.node_depth is not None:
        d["node_depth"] = step.node_depth.tolist()
    if step.node_coast is not None:
        d["node_coast"] = step.node_coast.tolist()
    if step.coast_ids is not None:
        d["coast_ids"] = step.coast_ids.tolist()
        d["coast_before"] = step.coast_before.tolist()
        d["coast_after"] = step.coast_after.tolist()
    return d


def _step_from_dict(d: dict) -> TopologyStep:
    def arr(key, dtype, shape2=None):
        if key not in d:
            return None
        a = np.asarray(d[key], dtype=dtype)
        if shape2 and a.size == 0:
            a = a.reshape(0, shape2)
        return a

    return TopologyStep(
        kind=d["kind"],
        node_ids=np.asarray(d["node_ids"], dtype=np.int64),
        node_lonlat=np.asarray(d["node_lonlat"], dtype=np.float64).reshape(-1, 2),
        node_depth=arr("node_depth", np.float64),
        node_coast=arr("node_coast", np.int8),
        tri_removed_idx=np.asarray(d["tri_removed_idx"], dtype=np.int64),
        tri_removed=arr("tri_removed", np.int64, 3),
        tri_added=arr("tri_added", np.int64, 3),
        coast_ids=arr("coast_ids", np.int64),
        coast_before=arr("coast_before", np.int8),
        coast_after=arr("coast_after", np.int8),
    )


def _op_to_record(op: EditOp | TopologyOp) -> dict:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if isinstance(op, TopologyOp):
        return {
            "kind": op.kind,
            "steps": [_step_to_dict(s) for s in op.steps],
            "params": op.params,
            "report": op.report,
            "t": stamp,
        }
    rec = {
        "kind": op.kind,
        "ids": op.ids.tolist(),
        "before": op.before.tolist(),
        "after": op.after.tolist(),
        "params": op.params,
        "report": op.report,
        "t": stamp,
    }
    if op.after_depth is not None:
        rec["before_depth"] = op.before_depth.tolist()
        rec["after_depth"] = op.after_depth.tolist()
    return rec


def _op_from_record(rec: dict) -> EditOp | TopologyOp:
    if rec["kind"] in TOPOLOGY_KINDS:
        return TopologyOp(
            kind=rec["kind"],
            steps=[_step_from_dict(s) for s in rec["steps"]],
            params=rec.get("params", {}),
            report=rec.get("report", {}),
        )
    return EditOp(
        kind=rec["kind"],
        ids=np.asarray(rec["ids"], dtype=np.int64),
        before=np.asarray(rec["before"], dtype=np.float64),
        after=np.asarray(rec["after"], dtype=np.float64),
        before_depth=(
            np.asarray(rec["before_depth"], dtype=np.float64)
            if "before_depth" in rec
            else None
        ),
        after_depth=(
            np.asarray(rec["after_depth"], dtype=np.float64)
            if "after_depth" in rec
            else None
        ),
        params=rec.get("params", {}),
        report=rec.get("report", {}),
    )


def _set_state(mesh: Mesh, op: EditOp, lonlat: np.ndarray, depth) -> None:
    if op.ids.size and op.ids.max() >= mesh.n_nodes:
        raise EditError(
            f"journal op touches node {op.ids.max()} but the mesh has "
            f"{mesh.n_nodes} nodes — wrong base mesh?"
        )
    mesh.lon[op.ids] = lonlat[:, 0]
    mesh.lat[op.ids] = lonlat[:, 1]
    if depth is not None:
        if mesh.depth is None:
            raise EditError("journal op carries depths but the mesh has none")
        mesh.depth[op.ids] = depth


def _apply_op(mesh: Mesh, op: EditOp | TopologyOp) -> None:
    if isinstance(op, TopologyOp):
        for step in op.steps:
            apply_step(mesh, step)
    else:
        _set_state(mesh, op, op.after, op.after_depth)


def _unapply_op(mesh: Mesh, op: EditOp | TopologyOp) -> None:
    if isinstance(op, TopologyOp):
        for step in reversed(op.steps):
            undo_step(mesh, step)
    else:
        _set_state(mesh, op, op.before, op.before_depth)


class Journal:
    """Undo/redo stacks persisted as an append-only JSONL file.

    `Journal(path, mesh)` opens (or creates) the journal at `path`; the
    mesh is only used to stamp the base-hash header on creation. Existing
    files are loaded with their full undo/redo state. Every mutation
    (`record`, `undo`, `redo`) appends one line and flushes.
    """

    def __init__(self, path: str | Path, mesh: Mesh | None = None):
        self.path = Path(path)
        self.base_hash: str | None = None
        self._undo: list[EditOp] = []
        self._redo: list[EditOp] = []
        if self.path.exists():
            self._load()
        elif mesh is not None:
            header = {
                "kind": "base",
                "v": _FORMAT_VERSION,
                "hash": mesh_hash(mesh),
                "n_nodes": mesh.n_nodes,
                "n_tri": mesh.n_tri,
            }
            self.base_hash = header["hash"]
            self._append_line(header)

    def _load(self) -> None:
        with open(self.path) as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EditError(f"{self.path}:{lineno}: corrupt journal line") from exc
                kind = rec.get("kind")
                if kind == "base":
                    self.base_hash = rec.get("hash")
                elif kind == "undo":
                    if not self._undo:
                        raise EditError(f"{self.path}:{lineno}: undo with empty stack")
                    self._redo.append(self._undo.pop())
                elif kind == "redo":
                    if not self._redo:
                        raise EditError(f"{self.path}:{lineno}: redo with empty stack")
                    self._undo.append(self._redo.pop())
                elif kind in ("move", "smooth", *TOPOLOGY_KINDS):
                    self._undo.append(_op_from_record(rec))
                    self._redo.clear()
                else:
                    raise EditError(f"{self.path}:{lineno}: unknown record kind {kind!r}")

    def _append_line(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    @property
    def ops(self) -> list[EditOp | TopologyOp]:
        """The active (applied, not undone) ops, oldest first."""
        return list(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def record(self, op: EditOp | TopologyOp) -> None:
        """Log an op already applied to the mesh; invalidates the redo stack."""
        self._undo.append(op)
        self._redo.clear()
        self._append_line(_op_to_record(op))

    def undo(self, mesh: Mesh) -> EditOp | TopologyOp | None:
        """Revert the latest op on `mesh`; None when there is nothing to undo."""
        if not self._undo:
            return None
        op = self._undo.pop()
        _unapply_op(mesh, op)
        self._redo.append(op)
        self._append_line({"kind": "undo"})
        return op

    def redo(self, mesh: Mesh) -> EditOp | TopologyOp | None:
        """Re-apply the latest undone op; None when there is nothing to redo."""
        if not self._redo:
            return None
        op = self._redo.pop()
        _apply_op(mesh, op)
        self._undo.append(op)
        self._append_line({"kind": "redo"})
        return op


def apply_journal(mesh: Mesh, journal: Journal | str | Path) -> Mesh:
    """Replay a journal onto its pristine base mesh, in place.

    Applies the stored `after` state of every active op in order and
    returns the mesh; `mesh_hash` of the result equals the hash of the
    live-edited mesh. Raises `EditError` when the journal was recorded
    against a different base mesh.
    """
    if not isinstance(journal, Journal):
        path = Path(journal)
        if not path.exists():
            raise EditError(f"journal not found: {path}")
        journal = Journal(path)
    if journal.base_hash is not None and journal.base_hash != mesh_hash(mesh):
        raise EditError(
            "journal was recorded against a different base mesh "
            f"(expected hash {journal.base_hash[:12]}...)"
        )
    for op in journal.ops:
        _apply_op(mesh, op)
    return mesh
