"""Ordered composition of resolution layers into an hfun field.

Each layer type registers an evaluator ``fn(layer, grid, project) -> ndarray``
returning a full-shape proposal in km, with NaN where the layer is undefined.
Evaluators may read ``grid.values`` (the field composed so far) to blend into
the surrounding resolution. The layer's ``mode`` then combines proposal and
current field: ``min`` keeps the finer value, ``replace`` overwrites where the
proposal is defined.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from fmesh2.project import Project

from .grid import KM_PER_DEG, MAX_AUTO_INTERVALS, HfunGrid, suggest_intervals

Evaluator = Callable[..., np.ndarray]

_EVALUATORS: dict[str, Evaluator] = {}


def register(layer_type: str):
    """Decorator: register an evaluator for a layer type."""

    def deco(fn: Evaluator) -> Evaluator:
        _EVALUATORS[layer_type] = fn
        return fn

    return deco


@dataclass
class HfunWarning:
    layer: str  # layer type or name, "" for project-level warnings
    message: str
    suggested_n_lat: int | None = None
    suggested_n_lon: int | None = None


@dataclass
class ComposeResult:
    grid: HfunGrid
    warnings: list[HfunWarning]


def finest_scale(layer) -> float:
    """Smallest resolution a layer can introduce, for grid-density decisions."""
    if layer.type == "mercator":
        return min(
            layer.resolution,
            layer.northern_lowest_resolution,
            layer.southern_lowest_resolution,
        )
    for attr in ("min_resolution", "resolution"):
        value = getattr(layer, attr, None)
        if value is not None:
            return float(value)
    return np.inf  # e.g. field layer: unknown until its data is loaded


def _make_grid(project: Project, layers: list) -> tuple[HfunGrid, list[HfunWarning]]:
    warnings: list[HfunWarning] = []
    n_lat, n_lon = project.grid.n_lat, project.grid.n_lon
    if n_lat is None or n_lon is None:
        finest = min((finest_scale(l) for l in layers), default=np.inf)
        if not np.isfinite(finest):
            raise ValueError(
                "cannot auto-suggest grid density: no enabled layer declares a "
                "resolution; set grid.n_lat/n_lon explicitly"
            )
        s_lat, s_lon = suggest_intervals(finest)
        n_lat, n_lon = n_lat or s_lat, n_lon or s_lon
        if s_lat >= MAX_AUTO_INTERVALS:
            warnings.append(
                HfunWarning(
                    layer="",
                    message=(
                        f"auto-suggested grid capped at {MAX_AUTO_INTERVALS} intervals; "
                        f"finest layer resolution {finest:g} km would need more — set "
                        "grid.n_lat/n_lon explicitly if you really want a denser grid"
                    ),
                )
            )
    return HfunGrid.regular(n_lat, n_lon), warnings


def compose(project: Project, grid: HfunGrid | None = None) -> ComposeResult:
    """Compose all enabled layers of `project` onto a (possibly auto-sized) grid."""
    layers = [l for l in project.layers if l.enabled]
    warnings: list[HfunWarning] = []
    if grid is None:
        grid, warnings = _make_grid(project, layers)

    for layer in layers:
        fn = _EVALUATORS.get(layer.type)
        if fn is None:
            raise NotImplementedError(f"no evaluator registered for layer type {layer.type!r}")
        proposal = fn(layer, grid, project)
        if layer.mode == "min":
            # fmin: NaN in the proposal keeps the current value
            grid.values = np.fmin(grid.values, proposal)
        else:  # replace
            defined = ~np.isnan(proposal)
            grid.values[defined] = proposal[defined]

    undefined = ~np.isfinite(grid.values)
    if undefined.any():
        warnings.append(
            HfunWarning(
                layer="",
                message=(
                    f"hfun field is undefined in {int(undefined.sum())} of "
                    f"{grid.values.size} cells — add a base layer"
                ),
            )
        )
    else:
        # fmesh README gotcha: a grid cell larger than the finest resolution
        # cannot represent that refinement for jigsaw
        min_res = float(grid.values.min())
        cell = max(grid.cell_km)
        if cell > min_res:
            n_lat, n_lon = suggest_intervals(min_res)
            warnings.append(
                HfunWarning(
                    layer="",
                    message=(
                        f"hfun grid too coarse: cell size {cell:.0f} km exceeds the "
                        f"finest composed resolution {min_res:g} km; refinements will "
                        f"be lost — use at least n_lat={n_lat}, n_lon={n_lon}"
                    ),
                    suggested_n_lat=n_lat,
                    suggested_n_lon=n_lon,
                )
            )
    return ComposeResult(grid=grid, warnings=warnings)
