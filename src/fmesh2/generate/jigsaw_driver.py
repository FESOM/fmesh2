"""jigsaw driver: HfunGrid -> raw global triangulation on the sphere.

Mirrors fmesh `triangulation`: ellipsoid geometry (R = 6371 km),
`ellipsoid-grid` hfun in radians with absolute scaling, and the proven
option set. The jigsaw binary runs in its own process group so a
cancellation can kill it without hunting grandchildren, and stdout is
streamed line-by-line to a progress callback.

The raw mesh is cached as npz keyed by a hash of the hfun field and the
options — the determinism anchor for reproducible builds: postprocessing is
pure numpy, so same cached raw mesh -> bit-identical output mesh.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jigsawpy
import numpy as np

from fmesh2.hfun.grid import HfunGrid

EARTH_RADIUS_KM = 6371.0

#: fmesh-proven option set
JIGSAW_OPTS = {
    "hfun_scal": "absolute",
    "mesh_dims": 2,
    "optm_qlim": 0.95,
    "optm_iter": 32,
    "mesh_top1": True,
    "geom_feat": True,
}


@dataclass
class RawMesh:
    """Raw jigsaw output on the sphere (degrees, 0-based triangles)."""

    lon: np.ndarray
    lat: np.ndarray
    tri: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.lon.shape[0])


class JigsawError(RuntimeError):
    pass


def hfun_hash(grid: HfunGrid) -> str:
    """Digest of everything that determines the jigsaw input."""
    h = hashlib.sha256()
    h.update(grid.lat.tobytes())
    h.update(grid.lon.tobytes())
    h.update(np.asarray(grid.values, dtype=np.float32).tobytes())
    h.update(repr(sorted(JIGSAW_OPTS.items())).encode())
    h.update(str(EARTH_RADIUS_KM).encode())
    return h.hexdigest()[:16]


def _find_jigsaw() -> str:
    bundled = Path(jigsawpy.__file__).resolve().parent / "_bin" / (
        "jigsaw.exe" if os.name == "nt" else "jigsaw"
    )
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("jigsaw")
    if found:
        return found
    raise JigsawError(
        "jigsaw executable not found (neither bundled with jigsawpy nor on PATH); "
        "install it with `conda install -c conda-forge jigsaw`"
    )


def _xyz_to_lonlat(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arcsin(np.clip(z / np.linalg.norm(coords, axis=1), -1.0, 1.0)))
    return lon, lat


def run_jigsaw(
    grid: HfunGrid,
    cache_dir: str | Path | None = None,
    workdir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
    process_callback: Callable[[subprocess.Popen], None] | None = None,
) -> RawMesh:
    """Triangulate; returns the cached raw mesh when the hfun is unchanged.

    `progress` receives raw jigsaw stdout lines; `process_callback` receives
    the Popen right after spawn (for cancellation via its process group).
    """
    cache_file = None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / f"jigsaw_{hfun_hash(grid)}.npz"
        if cache_file.exists():
            d = np.load(cache_file)
            if progress:
                progress(f"reusing cached jigsaw mesh {cache_file.name}")
            return RawMesh(lon=d["lon"], lat=d["lat"], tri=d["tri"])

    if workdir is None:
        if cache_dir is not None:
            workdir = Path(cache_dir)
        else:  # never litter the caller's cwd with .msh/.jig files
            import tempfile

            workdir = Path(tempfile.mkdtemp(prefix="fmesh2_jigsaw_"))
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    opts = jigsawpy.jigsaw_jig_t()
    opts.geom_file = str(workdir / "geom.msh")
    opts.jcfg_file = str(workdir / "opts.jig")
    opts.mesh_file = str(workdir / "mesh.msh")
    opts.hfun_file = str(workdir / "hfun.msh")
    for key, value in JIGSAW_OPTS.items():
        setattr(opts, key, value)

    geom = jigsawpy.jigsaw_msh_t()
    geom.mshID = "ellipsoid-mesh"
    geom.radii = EARTH_RADIUS_KM * np.ones(3)
    geom.ndims = +2
    jigsawpy.savemsh(opts.geom_file, geom)

    hmat = jigsawpy.jigsaw_msh_t()
    hmat.mshID = "ellipsoid-grid"
    hmat.ndims = +2
    hmat.radii = geom.radii
    hmat.xgrid = np.deg2rad(grid.lon).astype(hmat.REALS_t)
    hmat.ygrid = np.deg2rad(grid.lat).astype(hmat.REALS_t)
    hmat.value = np.asarray(grid.values, dtype=hmat.REALS_t)
    jigsawpy.savemsh(opts.hfun_file, hmat)

    jigsawpy.savejig(opts.jcfg_file, opts)
    proc = subprocess.Popen(
        [_find_jigsaw(), opts.jcfg_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=workdir,  # jigsaw writes a .log next to its cwd
        start_new_session=True,  # own process group: killable as a unit
    )
    if process_callback:
        process_callback(proc)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-30:]
        if progress and line:
            progress(line)
    code = proc.wait()
    if code != 0:
        raise JigsawError(
            f"jigsaw exited with code {code}:\n" + "\n".join(tail)
        )

    mesh = jigsawpy.jigsaw_msh_t()
    jigsawpy.loadmsh(opts.mesh_file, mesh)
    lon, lat = _xyz_to_lonlat(np.asarray(mesh.vert3["coord"], dtype=np.float64))
    tri = np.asarray(mesh.tria3["index"], dtype=np.int32)
    raw = RawMesh(lon=lon, lat=lat, tri=tri)

    if cache_file is not None:
        np.savez_compressed(cache_file, lon=raw.lon, lat=raw.lat, tri=raw.tri)
    return raw
