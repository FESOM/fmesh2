"""Coastline-distance layer evaluator.

fmesh `refine_along_coastlines` semantics: within `max_distance` km of the
coast the resolution ramps linearly from `min_resolution` (at the coast) to
the field composed so far (at `max_distance`); rows poleward of |85| deg are
left untouched. The per-gridpoint multiprocessing scan is replaced by one
KDTree nearest-neighbour query on unit-sphere coast points.

The distance field depends only on the coast source parameters and the grid,
so it is cached in-process: tweaking ramp parameters or layers above/below
recomposes in milliseconds (live-preview budget).
"""

from __future__ import annotations

import numpy as np

from fmesh2.core.topology import SphereKDTree
from fmesh2.data.coast_points import CoastPoints, prepare_coast_points
from fmesh2.project import CoastLayer, Project

from .compose import register
from .grid import HfunGrid

#: fmesh leaves grid rows with |lat| >= this untouched (its lon search window
#: degenerates near the poles; kept for parity)
LAT_LIMIT = 85.0

_distance_cache: dict[tuple, np.ndarray] = {}
_DISTANCE_CACHE_MAX = 4


def coast_distance_km(points: CoastPoints, grid: HfunGrid, cache_key: tuple | None = None) -> np.ndarray:
    """Great-circle distance (km) from every grid point to the nearest coast
    point; +inf in the untouched polar rows."""
    if cache_key is not None:
        key = (cache_key, grid.lat.tobytes(), grid.lon.tobytes())
        cached = _distance_cache.get(key)
        if cached is not None:
            return cached
    inside = (grid.lat > -LAT_LIMIT) & (grid.lat < LAT_LIMIT)
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat[inside])
    tree = SphereKDTree(points.lon, points.lat)
    d, _ = tree.nearest_km(lon2d.ravel(), lat2d.ravel())
    dist = np.full(grid.shape, np.inf)
    dist[inside] = d.reshape(lon2d.shape)
    if cache_key is not None:
        if len(_distance_cache) >= _DISTANCE_CACHE_MAX:
            _distance_cache.pop(next(iter(_distance_cache)))
        _distance_cache[key] = dist
    return dist


@register("coast")
def coast_field(layer: CoastLayer, grid: HfunGrid, project: Project) -> np.ndarray:
    source = layer.source
    if source is not None:
        source = project.resolve_path(source)
    points = prepare_coast_points(source, layer.min_length, layer.averaging)
    dist = coast_distance_km(
        points, grid, cache_key=(str(source), layer.min_length, layer.averaging)
    )
    proposal = np.full(grid.shape, np.nan)
    within = dist < layer.max_distance  # strict <, fmesh parity
    current = grid.values[within]
    proposal[within] = (
        layer.min_resolution
        + (current - layer.min_resolution) * dist[within] / layer.max_distance
    )
    return proposal
