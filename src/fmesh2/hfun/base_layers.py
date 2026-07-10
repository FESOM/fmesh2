"""Evaluators for the `base` and `mercator` layers.

Mercator matches fmesh `define_resolutions`: `resolution * cos(lat)` between
the boundaries, constant caps poleward of them (fmesh applies the caps with
`>=`/`<=`, kept here for parity). Note fmesh *replaces* the field with the
mercator profile, so a migrated project uses `mode: replace` on this layer.
"""

from __future__ import annotations

import numpy as np

from fmesh2.project import BaseLayer, MercatorLayer, Project

from .compose import register
from .grid import HfunGrid


@register("base")
def base_field(layer: BaseLayer, grid: HfunGrid, project: Project) -> np.ndarray:
    return np.full(grid.shape, float(layer.resolution))


@register("mercator")
def mercator_field(layer: MercatorLayer, grid: HfunGrid, project: Project) -> np.ndarray:
    profile = layer.resolution * np.cos(np.deg2rad(grid.lat))
    profile = np.where(grid.lat >= layer.northern_boundary, layer.northern_lowest_resolution, profile)
    profile = np.where(grid.lat <= layer.southern_boundary, layer.southern_lowest_resolution, profile)
    return np.repeat(profile[:, None], grid.lon.size, axis=1)
