"""Bathymetry-ramp layer evaluator.

fmesh `bathymetry_adjustment` semantics, branch-for-branch (topo is elevation,
negative down):

* land (elevation > 0): untouched
* 0 >= elevation >= -min_depth: `min_resolution`
* -min_depth > elevation >= -max_depth: linear ramp from `min_resolution`
  back to the field composed so far
* deeper: untouched
"""

from __future__ import annotations

import numpy as np

from fmesh2.data.topo import open_topo
from fmesh2.project import BathyLayer, Project

from .compose import register
from .grid import HfunGrid


@register("bathy")
def bathy_field(layer: BathyLayer, grid: HfunGrid, project: Project) -> np.ndarray:
    if project.topo is None:
        raise ValueError(
            "a bathy layer needs project.topo to point at a topography NetCDF"
        )
    topo = open_topo(str(project.topo_path), var=project.topo_var)
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
    elev = topo.elevation(lon2d, lat2d)

    proposal = np.full(grid.shape, np.nan)
    shallow = (elev <= 0) & (elev >= -layer.min_depth)
    proposal[shallow] = layer.min_resolution
    ramp = (elev < -layer.min_depth) & (elev >= -layer.max_depth)
    current = grid.values[ramp]
    proposal[ramp] = layer.min_resolution + (current - layer.min_resolution) * (
        np.abs(elev[ramp]) - layer.min_depth
    ) / (layer.max_depth - layer.min_depth)
    return proposal
