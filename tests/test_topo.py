import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fmesh2.data.topo import Topo, TopoError, open_topo

FIXTURES = Path(__file__).parent / "fixtures"


def tiny_ds(var="topo", lon_name="lon", lat_name="lat", descending_lat=False):
    """4x4 grid, no duplicate seam column: lon -180..90, lat -60..60."""
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    lat = np.array([-60.0, -20.0, 20.0, 60.0])
    values = lat[:, None] * 10.0 + lon[None, :] / 10.0  # distinct, linear
    if descending_lat:
        lat, values = lat[::-1], values[::-1]
    return xr.Dataset(
        {var: ((lat_name, lon_name), values)},
        coords={lat_name: lat, lon_name: lon},
    )


_counter = iter(range(10**6))


def make_topo(tmp_path, ds, **kw) -> Topo:
    path = tmp_path / f"t{next(_counter)}.nc"  # unique: xarray caches open handles
    ds.to_netcdf(path)
    return Topo(path, **kw)


def test_exact_at_grid_points(tmp_path):
    t = make_topo(tmp_path, tiny_ds())
    got = t.elevation(np.array([-90.0, 0.0]), np.array([-20.0, 60.0]))
    np.testing.assert_allclose(got, [-20 * 10 - 9.0, 60 * 10 + 0.0], rtol=1e-6)


def test_bilinear_midpoints(tmp_path):
    t = make_topo(tmp_path, tiny_ds())
    # midpoint between (0,-20) and (90,20): mean of 4 corners
    corners = np.array([-200 + 0, -200 + 9, 200 + 0, 200 + 9], dtype=float)
    got = t.elevation(np.array([45.0]), np.array([0.0]))
    assert got[0] == pytest.approx(corners.mean(), rel=1e-6)


def test_dateline_wrap(tmp_path):
    t = make_topo(tmp_path, tiny_ds())
    # 135E..180..-180: between last column (90) and first (-180, i.e. 180)
    v90 = t.elevation(np.array([90.0]), np.array([20.0]))[0]
    v180 = t.elevation(np.array([180.0]), np.array([20.0]))[0]
    vm180 = t.elevation(np.array([-180.0]), np.array([20.0]))[0]
    assert v180 == pytest.approx(vm180)  # same physical meridian
    mid = t.elevation(np.array([135.0]), np.array([20.0]))[0]
    assert mid == pytest.approx((v90 + v180) / 2, rel=1e-6)
    assert t.elevation(np.array([540.0]), np.array([20.0]))[0] == pytest.approx(
        v180
    )  # lon normalization beyond +360


def test_pole_rows_clamped(tmp_path):
    t = make_topo(tmp_path, tiny_ds())  # lat only reaches +-60
    edge = t.elevation(np.array([0.0]), np.array([60.0]))[0]
    assert t.elevation(np.array([0.0]), np.array([90.0]))[0] == pytest.approx(edge)
    assert t.elevation(np.array([0.0]), np.array([-90.0]))[0] == pytest.approx(
        t.elevation(np.array([0.0]), np.array([-60.0]))[0]
    )


def test_descending_lat_normalized(tmp_path):
    t_up = make_topo(tmp_path, tiny_ds())
    ds_down = tiny_ds(descending_lat=True)
    path = tmp_path / "down.nc"
    ds_down.to_netcdf(path)
    t_down = Topo(path)
    lons = np.array([-120.0, 30.0, 170.0])
    lats = np.array([-45.0, 10.0, 55.0])
    np.testing.assert_allclose(t_down.elevation(lons, lats), t_up.elevation(lons, lats), rtol=1e-6)


def test_rtopo_style_2d_coords(tmp_path):
    base = tiny_ds()
    lat2d = np.zeros((4, 2), np.float64)
    lat2d[:, 1] = base.lat.values  # real vector in column 1 (RTopo style)
    lon2d = np.zeros((4, 2), np.float64)
    lon2d[:, 0] = base.lon.values  # real vector in column 0
    ds = xr.Dataset(
        {
            "topo": (("DimLat", "DimLon"), base.topo.values),
            "lat": (("DimLat", "nax"), lat2d),
            "lon": (("DimLon", "nax"), lon2d),
        }
    )
    t = make_topo(tmp_path, ds)
    np.testing.assert_allclose(
        t.elevation(np.array([-90.0]), np.array([20.0])),
        base.topo.values[2, 1],
        rtol=1e-6,
    )


def test_variable_detection_and_errors(tmp_path):
    assert make_topo(tmp_path, tiny_ds(var="elevation")).var_name == "elevation"
    # single 2-D var with unknown name is accepted
    assert make_topo(tmp_path, tiny_ds(var="height_above_geoid")).var_name == "height_above_geoid"

    ds = tiny_ds(var="height_a")
    ds["height_b"] = ds["height_a"].copy()
    with pytest.raises(TopoError) as exc:
        make_topo(tmp_path, ds)
    assert "height_a" in str(exc.value) and "height_b" in str(exc.value)
    assert make_topo(tmp_path, ds, var="height_b").var_name == "height_b"
    with pytest.raises(TopoError, match="nope"):
        make_topo(tmp_path, ds, var="nope")

    bad = tiny_ds(lon_name="col", lat_name="row")
    with pytest.raises(TopoError, match="lon/lat"):
        make_topo(tmp_path, bad)


def test_open_topo_cached(tmp_path):
    path = tmp_path / "t.nc"
    tiny_ds().to_netcdf(path)
    assert open_topo(str(path)) is open_topo(str(path))


# ------------------------------------------------- committed global fixture


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "make_topo_synthetic", FIXTURES / "make_topo_synthetic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_synthetic_fixture_matches_generator():
    gen = _load_generator()
    ds = xr.open_dataset(FIXTURES / "topo_synthetic.nc")
    fresh = gen.build_dataset()
    assert ds.topo.shape == fresh.topo.shape == (361, 720)
    np.testing.assert_allclose(ds.topo.values, fresh.topo.values, atol=0.5)


def test_synthetic_fixture_features():
    t = Topo(FIXTURES / "topo_synthetic.nc")

    def at(lon, lat):
        return t.elevation(np.array([float(lon)]), np.array([float(lat)]))[0]

    assert at(-40, -10) < -3500  # deep ocean
    assert at(30, 15) > 0  # big continent land ring
    assert -1500 < at(20, 15) < 0  # enclosed lake below sea level
    assert at(180, 5) > 0 and at(-178, 5) > 0  # dateline continent, both sides
    assert at(0, -90) > 0  # polar cap is land
    assert at(0, 90) < 0  # north pole is ocean
