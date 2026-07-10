"""Regular lat/lon grid holding the hfun (target-resolution) field, plus the
mesh-size estimator.

Conventions follow fmesh: `n_lat`/`n_lon` count *intervals*, so the grid has
one more point per axis; lon spans -180..180 inclusive (the duplicate dateline
column is deliberate — jigsaw's ellipsoid-grid wants a closed longitude ring
and the fmesh estimator sums over it too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: km per degree of latitude, fmesh convention (kept for estimator parity)
KM_PER_DEG = 111.0

#: hard cap on auto-suggested intervals: 1/12 deg (~9 km cells at the equator)
#: keeps the float64 compose grid under ~100 MB
MAX_AUTO_INTERVALS = 2160


@dataclass
class HfunGrid:
    lat: np.ndarray  # (n_lat+1,) degrees, ascending, -90..90
    lon: np.ndarray  # (n_lon+1,) degrees, ascending, -180..180
    values: np.ndarray  # (lat.size, lon.size) target resolution, km

    def __post_init__(self) -> None:
        self.lat = np.asarray(self.lat, dtype=np.float64)
        self.lon = np.asarray(self.lon, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.lat.ndim != 1 or self.lon.ndim != 1:
            raise ValueError("lat and lon must be 1-D")
        if np.any(np.diff(self.lat) <= 0) or np.any(np.diff(self.lon) <= 0):
            raise ValueError("lat and lon must be strictly ascending")
        if self.values.shape != (self.lat.size, self.lon.size):
            raise ValueError(
                f"values shape {self.values.shape} != (lat, lon) sizes "
                f"({self.lat.size}, {self.lon.size})"
            )

    @classmethod
    def regular(cls, n_lat: int, n_lon: int, fill: float = np.inf) -> "HfunGrid":
        """Global grid with `n_lat` x `n_lon` intervals (fmesh convention)."""
        lat = np.linspace(-90.0, 90.0, int(n_lat) + 1)
        lon = np.linspace(-180.0, 180.0, int(n_lon) + 1)
        values = np.full((lat.size, lon.size), fill, dtype=np.float64)
        return cls(lat=lat, lon=lon, values=values)

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def cell_km(self) -> tuple[float, float]:
        """Nominal (d_lat, d_lon) cell size in km at the equator."""
        dlat = float(self.lat[1] - self.lat[0])
        dlon = float(self.lon[1] - self.lon[0])
        return dlat * KM_PER_DEG, dlon * KM_PER_DEG


def suggest_intervals(finest_km: float) -> tuple[int, int]:
    """Grid intervals so a cell (at the equator) is no larger than the finest
    target resolution — i.e. >=4 cells across a transition band of ~4 cells.
    Capped at MAX_AUTO_INTERVALS; compose() warns when the cap bites."""
    if not np.isfinite(finest_km) or finest_km <= 0:
        raise ValueError(f"finest_km must be positive and finite, got {finest_km}")
    n_lat = min(math.ceil(180.0 * KM_PER_DEG / finest_km), MAX_AUTO_INTERVALS)
    return n_lat, 2 * n_lat


@dataclass
class SizeEstimate:
    """fmesh's estimator counts jigsaw *elements including land* (despite the
    legacy function name); ocean node count is derived (~2 elements per node,
    times the ocean area fraction)."""

    elements_including_land: int
    ocean_nodes: int
    ocean_fraction: float


def estimate_size(grid: HfunGrid, ocean_fraction: float = 0.71) -> SizeEstimate:
    """Vectorized port of fmesh `estimate_number_of_nodes` (exact same sum,
    including the duplicate dateline column and 111 km/deg convention)."""
    dlat = abs(float(grid.lat[1] - grid.lat[0]))
    dlon = abs(float(grid.lon[1] - grid.lon[0]))
    area = dlat * KM_PER_DEG * dlon * KM_PER_DEG * np.cos(np.deg2rad(grid.lat))
    element_area = 0.7 * grid.values**2 / 2.0
    total = float(np.sum(area[:, None] / element_area))
    elements = int(np.round(total))
    # derived from the rounded (displayed) element count so the two shown
    # numbers stay mutually consistent
    return SizeEstimate(
        elements_including_land=elements,
        ocean_nodes=int(np.round(elements / 2.0 * ocean_fraction)),
        ocean_fraction=ocean_fraction,
    )
