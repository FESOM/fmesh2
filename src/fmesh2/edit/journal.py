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

_FORMAT_VERSION = 1


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


def _op_to_record(op: EditOp) -> dict:
    rec = {
        "kind": op.kind,
        "ids": op.ids.tolist(),
        "before": op.before.tolist(),
        "after": op.after.tolist(),
        "params": op.params,
        "report": op.report,
        "t": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if op.after_depth is not None:
        rec["before_depth"] = op.before_depth.tolist()
        rec["after_depth"] = op.after_depth.tolist()
    return rec


def _op_from_record(rec: dict) -> EditOp:
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
                elif kind in ("move", "smooth"):
                    self._undo.append(_op_from_record(rec))
                    self._redo.clear()
                else:
                    raise EditError(f"{self.path}:{lineno}: unknown record kind {kind!r}")

    def _append_line(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()

    @property
    def ops(self) -> list[EditOp]:
        """The active (applied, not undone) ops, oldest first."""
        return list(self._undo)

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def record(self, op: EditOp) -> None:
        """Log an op already applied to the mesh; invalidates the redo stack."""
        self._undo.append(op)
        self._redo.clear()
        self._append_line(_op_to_record(op))

    def undo(self, mesh: Mesh) -> EditOp | None:
        """Revert the latest op on `mesh`; None when there is nothing to undo."""
        if not self._undo:
            return None
        op = self._undo.pop()
        _set_state(mesh, op, op.before, op.before_depth)
        self._redo.append(op)
        self._append_line({"kind": "undo"})
        return op

    def redo(self, mesh: Mesh) -> EditOp | None:
        """Re-apply the latest undone op; None when there is nothing to redo."""
        if not self._redo:
            return None
        op = self._redo.pop()
        _set_state(mesh, op, op.after, op.after_depth)
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
        _set_state(mesh, op, op.after, op.after_depth)
    return mesh
