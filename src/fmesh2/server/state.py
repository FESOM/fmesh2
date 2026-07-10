from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fmesh2.data import coastlines
from fmesh2.viewer import ViewerBuffers, load_or_build


@dataclass
class MeshEntry:
    id: str
    path: Path
    buffers: ViewerBuffers


@dataclass
class AppState:
    cache_dir: Path | None = None
    coastline_source: Path | None = None
    meshes: dict[str, MeshEntry] = field(default_factory=dict)
    project: object | None = None  # active fmesh2.project.Project
    edit_sessions: dict = field(default_factory=dict)  # mesh id -> EditSession
    _counter: int = 0
    _coast: dict[str, np.ndarray] | None = None

    def open_mesh(self, path: str | Path, reload: bool = False) -> MeshEntry:
        path = Path(path).expanduser().resolve()
        for entry in list(self.meshes.values()):
            if entry.path == path:
                if not reload:
                    return entry
                del self.meshes[entry.id]  # rebuilt mesh: drop the stale entry
        if not (path / "nod2d.out").exists():
            raise FileNotFoundError(f"{path} is not a FESOM2 mesh directory (no nod2d.out)")
        buffers = load_or_build(path, cache_dir=self.cache_dir)
        self._counter += 1
        entry = MeshEntry(id=f"m{self._counter}", path=path, buffers=buffers)
        self.meshes[entry.id] = entry
        return entry

    def coastline(self) -> dict[str, np.ndarray]:
        if self._coast is None:
            self._coast = coastlines.coastline_segments(self.coastline_source)
        return self._coast
