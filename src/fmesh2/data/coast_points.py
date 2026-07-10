"""Coastline point preparation for the hfun coast layer.

Ports fmesh `read_coastlines2` semantics, vectorized: drop coastlines shorter
than `min_length` km, then thin each one by averaging runs of points ~
`averaging` km long. Works on any (Multi)LineString GeoJSON; the default
source is the cached Natural Earth 10m coastline.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from fmesh2.core.topology import chord_to_arc_km, sphere_xyz

from .coastlines import default_coastline_path, read_polylines


@dataclass(frozen=True)
class CoastPoints:
    lon: np.ndarray  # flat, degrees in [-180, 180]
    lat: np.ndarray
    lengths: np.ndarray  # points per coastline, sum == lon.size


def _step_km(lonlat: np.ndarray) -> np.ndarray:
    """Great-circle length of each polyline segment, km."""
    xyz = sphere_xyz(lonlat[:, 0], lonlat[:, 1])
    return chord_to_arc_km(np.linalg.norm(np.diff(xyz, axis=0), axis=1))


def polyline_length_km(lonlat: np.ndarray) -> float:
    return float(_step_km(lonlat).sum())


def smooth_polyline(lonlat: np.ndarray, averaging_km: float) -> np.ndarray:
    """fmesh-style thinning: average runs of consecutive points spanning
    ~`averaging_km` of coastline, close the ring. Longitudes are unwrapped
    before averaging so dateline-crossing coastlines produce no artifacts
    (replaces fmesh's post-hoc >1 deg outlier deletion)."""
    lon = np.rad2deg(np.unwrap(np.deg2rad(lonlat[:, 0])))
    lat = lonlat[:, 1]
    if averaging_km > 0:
        along = np.concatenate([[0.0], np.cumsum(_step_km(lonlat))])
        bins = np.unique((along / averaging_km).astype(np.int64), return_inverse=True)[1]
        counts = np.bincount(bins)
        lon = np.bincount(bins, weights=lon) / counts
        lat = np.bincount(bins, weights=lat) / counts
    if lon.size < 3:  # fmesh fallback: tiny ring becomes a 3-point mean triangle
        unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lonlat[:, 0])))
        thirds = [t for t in np.array_split(np.arange(lonlat.shape[0]), 3) if t.size]
        lon = np.array([unwrapped[t].mean() for t in thirds])
        lat = np.array([lonlat[t, 1].mean() for t in thirds])
    lon = (lon + 180.0) % 360.0 - 180.0
    out = np.column_stack([lon, lat])
    return np.vstack([out, out[:1]])  # fmesh closes every coastline


def prepare_coast_points(
    source: str | Path | None = None,
    min_length: float = 100.0,
    averaging: float = 50.0,
) -> CoastPoints:
    """Coast points for distance fields; results are memoized per arguments."""
    if source is None:
        source = default_coastline_path()
    return _prepare_cached(str(source), float(min_length), float(averaging))


@lru_cache(maxsize=8)
def _prepare_cached(source: str, min_length: float, averaging: float) -> CoastPoints:
    kept = [
        smooth_polyline(line, averaging)
        for line in read_polylines(source)
        if polyline_length_km(line) >= min_length
    ]
    if not kept:
        raise ValueError(
            f"no coastline in {source} is at least {min_length} km long — "
            "lower the coast layer's min_length"
        )
    return CoastPoints(
        lon=np.concatenate([k[:, 0] for k in kept]),
        lat=np.concatenate([k[:, 1] for k in kept]),
        lengths=np.array([k.shape[0] for k in kept], dtype=np.int64),
    )
