"""Topography (bathymetry) NetCDF access.

Opens RTopo/GEBCO-style files lazily via xarray, tolerating the common naming
variants (`topo`/`elevation`/`z`/..., `lon`/`longitude`/`x`, 1-D or RTopo's
2-D coordinate variables) and both lat orientations. `elevation()` bilinearly
samples arbitrary lon/lat arrays — vectorized, dateline-wrapping, pole rows
clamped — loading only the grid window the query needs (a global query loads
the variable once; regional queries stay cheap).

Convention: returned values are metres of elevation, negative below sea level.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import xarray as xr

VAR_CANDIDATES = ("topo", "elevation", "z", "bathymetry", "bed")
LON_NAMES = ("lon", "longitude", "x")
LAT_NAMES = ("lat", "latitude", "y")


class TopoError(ValueError):
    """Unusable topography file (missing/ambiguous variables or coords)."""


def _collapse_coord(values: np.ndarray, name: str) -> np.ndarray:
    """RTopo stores lon/lat as (n, 2) with the real vector in one column and
    zeros in the other; pick the strictly monotonic one."""
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        cols = values.T if values.shape[0] > values.shape[1] else values
        for col in cols:
            d = np.diff(col)
            if np.all(d > 0) or np.all(d < 0):
                return col
    raise TopoError(f"coordinate {name!r} has no strictly monotonic 1-D vector")


def _find_name(ds: xr.Dataset, candidates: tuple[str, ...]) -> str | None:
    names = {n.lower(): n for n in list(ds.variables)}
    for cand in candidates:
        if cand in names:
            return names[cand]
    return None


class Topo:
    def __init__(self, path: str | Path, var: str | None = None):
        self._ds = xr.open_dataset(path)
        self.path = str(path)

        if var is None:
            var = _find_name(self._ds, VAR_CANDIDATES)
            if var is None:
                two_d = [n for n, v in self._ds.data_vars.items() if v.ndim == 2]
                if len(two_d) == 1:
                    var = two_d[0]
                else:
                    raise TopoError(
                        f"cannot identify the elevation variable in {path}; "
                        f"found variables {sorted(self._ds.data_vars)} — pass var= explicitly"
                    )
        elif var not in self._ds.variables:
            raise TopoError(
                f"variable {var!r} not in {path}; found {sorted(self._ds.data_vars)}"
            )
        self.var_name = var
        da = self._ds[var]
        if da.ndim != 2:
            raise TopoError(f"elevation variable {var!r} must be 2-D, has dims {da.dims}")

        lon_name = _find_name(self._ds, LON_NAMES)
        lat_name = _find_name(self._ds, LAT_NAMES)
        if lon_name is None or lat_name is None:
            raise TopoError(
                f"cannot identify lon/lat coordinates in {path}; "
                f"found variables {sorted(self._ds.variables)}"
            )
        lon = _collapse_coord(np.asarray(self._ds[lon_name].values, dtype=np.float64), lon_name)
        lat = _collapse_coord(np.asarray(self._ds[lat_name].values, dtype=np.float64), lat_name)

        # which axis of the variable is latitude?
        lat_dims = set(self._ds[lat_name].dims) & set(da.dims)
        if lat_dims:
            lat_axis = da.dims.index(next(iter(lat_dims)))
        elif da.shape[0] == lat.size and da.shape[1] != lat.size:
            lat_axis = 0
        elif da.shape[1] == lat.size and da.shape[0] != lat.size:
            lat_axis = 1
        else:
            raise TopoError(f"cannot match lat axis: var shape {da.shape}, lat size {lat.size}")
        if lat_axis != 0:
            da = da.transpose(da.dims[1], da.dims[0])
        if da.shape != (lat.size, lon.size):
            raise TopoError(
                f"coordinate sizes ({lat.size}, {lon.size}) do not match "
                f"variable shape {da.shape}"
            )

        self._flip_lat = bool(lat[0] > lat[-1])
        self.lat = lat[::-1].copy() if self._flip_lat else lat
        if np.any(np.diff(lon) <= 0):
            raise TopoError("longitude vector must be strictly ascending")
        self.lon = lon
        self._da = da

        # cached loaded window: values[rows r0:r1, all columns]
        self._win: np.ndarray | None = None
        self._win_r0 = 0
        self._win_r1 = 0

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.size, self.lon.size

    def _rows(self, r0: int, r1: int) -> np.ndarray:
        """Grid rows [r0, r1) in ascending-lat space, all columns, loading and
        growing the cached window as needed."""
        if self._win is None or r0 < self._win_r0 or r1 > self._win_r1:
            n = self.lat.size
            load0 = min(r0, self._win_r0) if self._win is not None else r0
            load1 = max(r1, self._win_r1) if self._win is not None else r1
            if self._flip_lat:
                block = self._da.isel({self._da.dims[0]: slice(n - load1, n - load0)}).values
                block = block[::-1]
            else:
                block = self._da.isel({self._da.dims[0]: slice(load0, load1)}).values
            self._win, self._win_r0, self._win_r1 = np.asarray(block), load0, load1
        return self._win[r0 - self._win_r0 : r1 - self._win_r0]

    def elevation(self, lon, lat) -> np.ndarray:
        """Bilinear elevation (m, negative down) at arbitrary points."""
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        out_shape = np.broadcast_shapes(lon.shape, lat.shape)
        lon = np.broadcast_to(lon, out_shape).ravel()
        lat = np.broadcast_to(lat, out_shape).ravel()
        lonv, latv = self.lon, self.lat
        n, m = lonv.size, latv.size

        q = lonv[0] + (lon - lonv[0]) % 360.0  # into [lon0, lon0+360)
        i0 = np.clip(np.searchsorted(lonv, q, side="right") - 1, 0, n - 1)
        i1 = (i0 + 1) % n  # column n-1 wraps to 0 (grids without a duplicate seam)
        dlon = (lonv[i1] - lonv[i0]) % 360.0
        dlon[dlon == 0.0] = 360.0  # degenerate wrap on duplicate-seam grids
        wx = np.clip((q - lonv[i0]) / dlon, 0.0, 1.0)

        ql = np.clip(lat, latv[0], latv[-1])  # pole rows clamp to edge
        j0 = np.clip(np.searchsorted(latv, ql, side="right") - 1, 0, m - 2)
        wy = np.clip((ql - latv[j0]) / (latv[j0 + 1] - latv[j0]), 0.0, 1.0)

        rows = self._rows(int(j0.min()), int(j0.max()) + 2)
        j = j0 - int(j0.min())
        v00 = rows[j, i0].astype(np.float64)
        v01 = rows[j, i1].astype(np.float64)
        v10 = rows[j + 1, i0].astype(np.float64)
        v11 = rows[j + 1, i1].astype(np.float64)
        vals = (
            v00 * (1 - wx) * (1 - wy)
            + v01 * wx * (1 - wy)
            + v10 * (1 - wx) * wy
            + v11 * wx * wy
        )
        return vals.reshape(out_shape)


@lru_cache(maxsize=4)
def open_topo(path: str, var: str | None = None) -> Topo:
    """Shared Topo instances so the loaded window is reused across layers."""
    return Topo(path, var=var)
