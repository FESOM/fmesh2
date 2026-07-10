"""GPU-ready binary buffers for the WebGL viewer.

Layout (validated by the 2026-07-10 render spike):
- pos:  float32 (v, 3) unit-sphere xyz, Y = north pole (three.js Y-up)
- pos2: float32 (v, 3) Plate Carree, x = lon_rad, y = lat_rad, z = 0
- tri:  uint32 (m, 3) with dateline-crossing triangles remapped to duplicated
        seam vertices (west corners shifted +360 deg in the plane)
- edge: uint32 (e, 2) unique undirected edges of the remapped triangles
- res/dep: float32 per vertex; computed on the TRUE topology before seam
        duplication, then copied to duplicates.
- dupsrc: uint32 (verts - nodes,) source node id of each duplicated seam
        vertex (vertex n + i duplicates node dupsrc[i]) — edit mode (M4)
        uses it to keep duplicates in sync during live node drags.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fmesh2.core.io_fesom import read_fesom
from fmesh2.core.mesh import Mesh
from fmesh2.core.metrics import local_resolution_km
from fmesh2.core.topology import unique_edges

BUFFER_NAMES = ("pos", "pos2", "tri", "edge", "res", "dep", "dupsrc")


@dataclass
class ViewerBuffers:
    pos: np.ndarray
    pos2: np.ndarray
    tri: np.ndarray
    edge: np.ndarray
    res: np.ndarray
    dep: np.ndarray
    dupsrc: np.ndarray
    meta: dict = field(default_factory=dict)

    def buffer(self, name: str) -> np.ndarray:
        if name not in BUFFER_NAMES:
            raise KeyError(name)
        return getattr(self, name)


def _percentile_range(values: np.ndarray, lo=2, hi=98) -> list[float]:
    if values.size == 0:
        return [0.0, 1.0]
    a, b = np.percentile(values, [lo, hi])
    if a == b:
        b = a + 1.0
    return [float(a), float(b)]


def build_buffers(mesh: Mesh, name: str = "mesh") -> ViewerBuffers:
    n = mesh.n_nodes

    # scalars on the true topology, before any seam duplication
    res = local_resolution_km(mesh).astype(np.float32)
    dep = (
        np.asarray(mesh.depth, dtype=np.float32)
        if mesh.depth is not None
        else np.zeros(n, np.float32)
    )
    dep = np.abs(dep)

    # duplicate west-side vertices of dateline-crossing triangles (+360 deg)
    lon_norm = (np.asarray(mesh.lon, dtype=np.float64) + 180.0) % 360.0 - 180.0
    lat = np.asarray(mesh.lat, dtype=np.float64)
    tri = mesh.tri.astype(np.int64).copy()
    tl = lon_norm[tri]
    crossing = (tl.max(axis=1) - tl.min(axis=1)) > 180.0
    west = (tl < 0.0) & crossing[:, None]
    dup_src = np.unique(tri[west])
    remap = np.full(n, -1, np.int64)
    remap[dup_src] = n + np.arange(dup_src.shape[0])
    tri[west] = remap[tri[west]]

    lon_v = np.concatenate([lon_norm, lon_norm[dup_src] + 360.0])
    lat_v = np.concatenate([lat, lat[dup_src]])
    nv = lon_v.shape[0]

    lon_r = np.deg2rad(lon_v)
    lat_r = np.deg2rad(lat_v)
    pos = np.empty((nv, 3), np.float32)
    coslat = np.cos(lat_r)
    pos[:, 0] = coslat * np.cos(lon_r)
    pos[:, 1] = np.sin(lat_r)
    pos[:, 2] = -coslat * np.sin(lon_r)

    pos2 = np.zeros((nv, 3), np.float32)
    pos2[:, 0] = lon_r
    pos2[:, 1] = lat_r

    edge = unique_edges(tri, nv).astype(np.uint32)
    res_v = np.concatenate([res, res[dup_src]])
    dep_v = np.concatenate([dep, dep[dup_src]])

    meta = {
        "name": name,
        "nodes": int(n),
        "verts": int(nv),
        "triangles": int(mesh.n_tri),
        "edges": int(edge.shape[0]),
        "res_range": _percentile_range(res),
        "dep_range": [0.0, _percentile_range(dep)[1]],
    }
    buffers = ViewerBuffers(
        pos=pos,
        pos2=pos2,
        tri=tri.astype(np.uint32),
        edge=edge,
        res=res_v.astype(np.float32),
        dep=dep_v.astype(np.float32),
        dupsrc=dup_src.astype(np.uint32),
        meta=meta,
    )
    meta["bytes"] = {b: int(buffers.buffer(b).nbytes) for b in BUFFER_NAMES}
    return buffers


def _cache_key(mesh_dir: Path) -> str:
    h = hashlib.sha1()
    h.update(b"format:v2;")  # v2: added the dupsrc buffer
    for fname in ("nod2d.out", "elem2d.out", "aux3d.out"):
        p = mesh_dir / fname
        if p.exists():
            st = p.stat()
            h.update(f"{fname}:{st.st_size}:{st.st_mtime_ns};".encode())
    return h.hexdigest()[:16]


_DTYPES = {"pos": np.float32, "pos2": np.float32, "tri": np.uint32,
           "edge": np.uint32, "res": np.float32, "dep": np.float32,
           "dupsrc": np.uint32}
_SHAPES = {"pos": 3, "pos2": 3, "tri": 3, "edge": 2, "res": 1, "dep": 1,
           "dupsrc": 1}


def load_or_build(mesh_dir: str | Path, cache_dir: str | Path | None = None,
                  name: str | None = None) -> ViewerBuffers:
    """Build viewer buffers for a FESOM2 mesh directory, using a disk cache."""
    mesh_dir = Path(mesh_dir)
    name = name or mesh_dir.name
    cache = None
    if cache_dir is not None:
        cache = Path(cache_dir) / _cache_key(mesh_dir)
        meta_file = cache / "meta.json"
        if meta_file.exists() and all((cache / f"{b}.bin").exists() for b in BUFFER_NAMES):
            meta = json.loads(meta_file.read_text())
            arrays = {}
            for b in BUFFER_NAMES:
                flat = np.fromfile(cache / f"{b}.bin", dtype=_DTYPES[b])
                arrays[b] = flat.reshape(-1, _SHAPES[b]) if _SHAPES[b] > 1 else flat
            return ViewerBuffers(**arrays, meta=meta)

    buffers = build_buffers(read_fesom(mesh_dir), name=name)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        for b in BUFFER_NAMES:
            buffers.buffer(b).tofile(cache / f"{b}.bin")
        (cache / "meta.json").write_text(json.dumps(buffers.meta))
    return buffers
