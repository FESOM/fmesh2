"""Imported lon/lat resolution-field layer evaluator.

Reads a resolution grid (km) from NetCDF or whitespace 3-column text
(lon lat value on a regular grid), regrids it onto the hfun grid with
bilinear `RegularGridInterpolator`, and leaves cells outside the field's
domain untouched (NaN proposal). Values pass through unchanged — no unit
conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from fmesh2.data.topo import LAT_NAMES, LON_NAMES, _find_name
from fmesh2.project import FieldLayer, Project

from .compose import register
from .grid import HfunGrid

VAR_CANDIDATES = ("resolution", "res", "hfun", "values")


def load_field(path: str | Path, variable: str | None = None):
    """(lat_vec, lon_vec, values) from NetCDF or 3-column text."""
    path = Path(path)
    if path.suffix.lower() in (".nc", ".nc4", ".netcdf"):
        ds = xr.open_dataset(path)
        var = variable or _find_name(ds, VAR_CANDIDATES)
        if var is None:
            two_d = [n for n, v in ds.data_vars.items() if v.ndim == 2]
            if len(two_d) == 1:
                var = two_d[0]
            else:
                raise ValueError(
                    f"cannot identify the resolution variable in {path}; found "
                    f"{sorted(ds.data_vars)} — set the layer's variable"
                )
        elif var not in ds.variables:
            raise ValueError(f"variable {var!r} not in {path}; found {sorted(ds.data_vars)}")
        da = ds[var]
        lon_name = _find_name(ds, LON_NAMES)
        lat_name = _find_name(ds, LAT_NAMES)
        if lon_name is None or lat_name is None:
            raise ValueError(f"cannot identify lon/lat coordinates in {path}")
        lon = np.asarray(ds[lon_name].values, dtype=np.float64)
        lat = np.asarray(ds[lat_name].values, dtype=np.float64)
        values = np.asarray(da.transpose(ds[lat_name].dims[0], ds[lon_name].dims[0]).values)
    else:
        data = np.loadtxt(path)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(f"{path}: text fields need 3 columns (lon lat value)")
        lon = np.unique(data[:, 0])
        lat = np.unique(data[:, 1])
        if lon.size * lat.size != data.shape[0]:
            raise ValueError(f"{path}: points do not form a regular lon/lat grid")
        order = np.lexsort((data[:, 0], data[:, 1]))
        values = data[order, 2].reshape(lat.size, lon.size)
    if lat[0] > lat[-1]:
        lat, values = lat[::-1], values[::-1]
    return lat, lon, values


@register("field")
def field_field(layer: FieldLayer, grid: HfunGrid, project: Project) -> np.ndarray:
    lat, lon, values = load_field(project.resolve_path(layer.path), layer.variable)
    rgi = RegularGridInterpolator(
        (lat, lon), values, method="linear", bounds_error=False, fill_value=np.nan
    )
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
    return rgi((lat2d, lon2d))
