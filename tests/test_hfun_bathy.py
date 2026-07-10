from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fmesh2.hfun import compose
from fmesh2.project import BaseLayer, BathyLayer, GridSettings, Project

FIXTURES = Path(__file__).parent / "fixtures"

BASE = 300.0
MIN_RES = 50.0
MIN_DEPTH = 50.0
MAX_DEPTH = 1500.0


def banded_topo(tmp_path):
    """Constant-per-longitude-band elevations covering every fmesh branch."""
    lon = np.array([-180.0, -120.0, -60.0, 0.0, 60.0, 120.0])
    lat = np.array([-90.0, 0.0, 90.0])
    bands = np.array([100.0, -30.0, -50.0, -775.0, -1500.0, -3000.0])
    values = np.repeat(bands[None, :], lat.size, axis=0)
    path = tmp_path / "bands.nc"
    xr.Dataset({"topo": (("lat", "lon"), values)}, coords={"lat": lat, "lon": lon}).to_netcdf(path)
    return path, bands


def bathy_project(topo_path, mode="min", n_lat=6, n_lon=12):
    return Project(
        name="t",
        topo=str(topo_path),
        grid=GridSettings(n_lat=n_lat, n_lon=n_lon),
        layers=[
            BaseLayer(type="base", resolution=BASE),
            BathyLayer(
                type="bathy",
                mode=mode,
                min_resolution=MIN_RES,
                min_depth=MIN_DEPTH,
                max_depth=MAX_DEPTH,
            ),
        ],
    )


def expected_for(elev, current=BASE):
    """fmesh bathymetry_adjustment, branch for branch."""
    if elev > 0:
        return current
    if elev >= -MIN_DEPTH:
        return MIN_RES
    if elev >= -MAX_DEPTH:
        return MIN_RES + (current - MIN_RES) * (abs(elev) - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
    return current


def test_ramp_matches_fmesh_branches(tmp_path):
    topo_path, bands = banded_topo(tmp_path)
    result = compose(bathy_project(topo_path))
    g = result.grid
    for lon_c, elev in zip([-180.0, -120.0, -60.0, 0.0, 60.0, 120.0], bands):
        i = np.searchsorted(g.lon, lon_c)
        got = g.values[1, i]  # equator row, cell centred on the band
        assert got == pytest.approx(expected_for(elev), rel=1e-12), f"band elev={elev}"
    # spot values: land 300, shallow 50, exact -50 -> 50, mid-ramp -775 -> 175,
    # exact -1500 -> 300, deep 300
    assert expected_for(-775.0) == 175.0


def test_zero_elevation_is_shallow(tmp_path):
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    lat = np.array([-90.0, 0.0, 90.0])
    path = tmp_path / "z.nc"
    xr.Dataset(
        {"topo": (("lat", "lon"), np.zeros((3, 4)))}, coords={"lat": lat, "lon": lon}
    ).to_netcdf(path)
    result = compose(bathy_project(path))
    assert np.all(result.grid.values == MIN_RES)  # fmesh: depth 0 hits the shallow branch


def test_mode_min_vs_replace_when_coarsening(tmp_path):
    """fmesh replaces (can coarsen a finer field); mode=min cannot."""
    topo_path, _ = banded_topo(tmp_path)
    fine = 30.0  # finer than the bathy layer's min_resolution

    def with_base(mode):
        proj = bathy_project(topo_path, mode=mode)
        proj.layers[0].resolution = fine
        return compose(proj).grid

    g_replace = with_base("replace")
    g_min = with_base("min")
    i_shallow = np.searchsorted(g_replace.lon, -60.0)
    assert g_replace.values[1, i_shallow] == MIN_RES  # coarsened, fmesh semantics
    assert g_min.values[1, i_shallow] == fine  # min keeps the finer field


def test_bathy_requires_topo():
    proj = Project(
        name="t",
        grid=GridSettings(n_lat=4, n_lon=8),
        layers=[
            BaseLayer(type="base", resolution=300),
            BathyLayer(type="bathy", min_resolution=50),
        ],
    )
    with pytest.raises(ValueError, match="project.topo"):
        compose(proj)


def test_compose_on_synthetic_global(tmp_path):
    proj = Project(
        name="t",
        topo=str(FIXTURES / "topo_synthetic.nc"),
        grid=GridSettings(n_lat=90, n_lon=180),
        layers=[
            BaseLayer(type="base", resolution=BASE),
            BathyLayer(type="bathy", min_resolution=MIN_RES),
        ],
    )
    g = compose(proj).grid
    lon2d, lat2d = np.meshgrid(g.lon, g.lat)
    from fmesh2.data.topo import open_topo

    elev = open_topo(str(FIXTURES / "topo_synthetic.nc")).elevation(lon2d, lat2d)
    assert np.all(g.values[elev > 0] == BASE)  # land untouched
    shelf = (elev <= 0) & (elev >= -MIN_DEPTH)
    assert shelf.any() and np.all(g.values[shelf] == MIN_RES)
    deep = elev < -MAX_DEPTH
    assert np.all(g.values[deep] == BASE)
    mid = (elev < -MIN_DEPTH) & (elev >= -MAX_DEPTH)
    assert mid.any()
    assert np.all((g.values[mid] > MIN_RES) & (g.values[mid] < BASE))
