"""Coastline vector data for viewer overlay and hfun algorithms.

Default source is Natural Earth 10m coastline (GeoJSON), downloaded once into
the user cache; any GeoJSON with (Multi)LineString features can be substituted.
"""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np

NE_10M_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/"
    "master/geojson/ne_10m_coastline.geojson"
)


def cache_dir() -> Path:
    return Path(os.environ.get("FMESH2_CACHE", "~/.cache/fmesh2")).expanduser()


def default_coastline_path(download: bool = True) -> Path:
    """Path to the cached Natural Earth coastline, downloading if needed."""
    path = cache_dir() / "ne_10m_coastline.geojson"
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"coastline not cached at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        urllib.request.urlretrieve(NE_10M_URL, tmp)
    except OSError as exc:
        raise RuntimeError(
            f"could not download coastline from {NE_10M_URL} and no cached copy "
            f"exists at {path}; download it manually or pass a GeoJSON path"
        ) from exc
    tmp.rename(path)
    return path


def read_polylines(source: str | Path) -> list[np.ndarray]:
    """(Multi)LineString features of a GeoJSON file as (k, 2) lon/lat arrays."""
    gj = json.loads(Path(source).read_text())
    lines: list[np.ndarray] = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "LineString":
            parts = [geom["coordinates"]]
        elif geom["type"] == "MultiLineString":
            parts = geom["coordinates"]
        else:
            continue
        for part in parts:
            arr = np.asarray(part, dtype=np.float64)[:, :2]
            if arr.shape[0] >= 2:
                lines.append(arr)
    return lines


def coastline_segments(source: str | Path | None = None) -> dict[str, np.ndarray]:
    """Viewer line-segment buffers: {'pos','pos2'} float32 arrays, 2 verts/segment.

    Dateline-crossing segments get their west endpoint shifted +360 deg in the
    plane coordinates (same convention as the mesh seam duplication).
    """
    if source is None:
        source = default_coastline_path()
    lines = read_polylines(source)

    a = np.concatenate([ln[:-1] for ln in lines])
    b = np.concatenate([ln[1:] for ln in lines])
    for arr in (a, b):
        arr[:, 0] = (arr[:, 0] + 180.0) % 360.0 - 180.0
    crossing = np.abs(a[:, 0] - b[:, 0]) > 180.0
    for arr in (a, b):
        west = crossing & (arr[:, 0] < 0.0)
        arr[west, 0] += 360.0

    pts = np.empty((a.shape[0] * 2, 2), np.float64)
    pts[0::2] = a
    pts[1::2] = b
    lon = np.deg2rad(pts[:, 0])
    lat = np.deg2rad(pts[:, 1])

    pos = np.empty((pts.shape[0], 3), np.float32)
    coslat = np.cos(lat)
    pos[:, 0] = coslat * np.cos(lon)
    pos[:, 1] = np.sin(lat)
    pos[:, 2] = -coslat * np.sin(lon)

    pos2 = np.zeros_like(pos)
    pos2[:, 0] = lon
    pos2[:, 1] = lat
    return {"pos": pos, "pos2": pos2}
