from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    """FESOM2 surface mesh.

    lon/lat are kept exactly as read (degrees, not normalized) so that
    read -> write roundtrips are bit-identical; any normalization is a
    viewer/algorithm concern. `tri` is 0-based. `depth` is positive down (m).
    """

    lon: np.ndarray
    lat: np.ndarray
    tri: np.ndarray
    depth: np.ndarray | None = None
    coast_flag: np.ndarray | None = None
    levels: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.lon = np.asarray(self.lon, dtype=np.float64)
        self.lat = np.asarray(self.lat, dtype=np.float64)
        self.tri = np.asarray(self.tri)
        if self.lon.shape != self.lat.shape or self.lon.ndim != 1:
            raise ValueError("lon and lat must be 1-D arrays of equal length")
        if self.tri.ndim != 2 or self.tri.shape[1] != 3:
            raise ValueError("tri must have shape (n_tri, 3)")
        if self.tri.size and (self.tri.min() < 0 or self.tri.max() >= self.lon.shape[0]):
            raise ValueError("tri contains node indices out of range")
        for name in ("depth", "coast_flag"):
            arr = getattr(self, name)
            if arr is not None and np.asarray(arr).shape != self.lon.shape:
                raise ValueError(f"{name} must have one value per node")

    @property
    def n_nodes(self) -> int:
        return int(self.lon.shape[0])

    @property
    def n_tri(self) -> int:
        return int(self.tri.shape[0])
