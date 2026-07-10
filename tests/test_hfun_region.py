import json
from pathlib import Path

import numpy as np
import pytest

from fmesh2.core.projection import haversine_km
from fmesh2.hfun import compose
from fmesh2.project import BaseLayer, FieldLayer, GridSettings, Project, RegionLayer

FIXTURES = Path(__file__).parent / "fixtures"
BASE = 300.0
RES = 25.0


def write_region(path, *rings, roles=None):
    features = []
    for k, ring in enumerate(rings):
        props = {}
        if roles:
            props["role"] = roles[k]
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in ring]]},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def square(lon0, lon1, lat0, lat1):
    return [
        [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0],
    ]


def region_project(path, n_lat=180, n_lon=360, **layer_kw):
    kw = dict(resolution=RES)
    kw.update(layer_kw)
    return Project(
        name="t",
        grid=GridSettings(n_lat=n_lat, n_lon=n_lon),
        layers=[
            BaseLayer(type="base", resolution=BASE),
            RegionLayer(type="region", path=str(path), **kw),
        ],
    )


def grid_value(g, lon, lat):
    return g.values[np.searchsorted(g.lat, lat), np.searchsorted(g.lon, lon)]


def test_mid_atlantic_square_single_ring(tmp_path):
    path = write_region(tmp_path / "r.geojson", square(-40, -20, 10, 30))
    g = compose(region_project(path, transition_km=500)).grid
    assert grid_value(g, -30, 20) == RES  # deep inside
    assert grid_value(g, -39, 11) == RES  # inside near corner
    v_band = grid_value(g, -18, 20)  # ~208 km outside the -20 edge
    d = haversine_km(-18.0, 20.0, -20.0, 20.0)
    expected = RES + (BASE - RES) * d / 500.0
    assert v_band == pytest.approx(expected, rel=0.02)
    assert grid_value(g, -10, 20) == BASE  # beyond the band
    assert grid_value(g, 60, -40) == BASE  # far field untouched


def test_dateline_straddling_polygon(tmp_path):
    path = write_region(tmp_path / "r.geojson", square(170, 190, -10, 10))
    g = compose(region_project(path, transition_km=300)).grid
    assert grid_value(g, 179, 0) == RES
    assert grid_value(g, -179, 0) == RES  # west of the dateline, inside
    assert grid_value(g, 175, 5) == RES
    assert grid_value(g, 150, 0) == BASE  # far outside
    v_band = grid_value(g, -168, 0)  # ~222 km east of the 190(-170) edge
    assert RES < v_band < BASE


def test_polar_cap_polygon(tmp_path):
    ring = [[lon, 70.0] for lon in range(-180, 181, 20)]
    path = write_region(tmp_path / "r.geojson", ring)
    g = compose(region_project(path, transition_km=300)).grid
    assert grid_value(g, 0, 80) == RES  # pole side is "inside"
    assert grid_value(g, 90, 85) == RES
    assert grid_value(g, 0, 89) == RES
    assert grid_value(g, 0, 50) == BASE
    band = grid_value(g, 0, 68)
    assert RES < band < BASE


def test_two_ring_matches_single_plus_width_convex(tmp_path):
    """Square inner ring; outer = same square widened by ~W on each side."""
    lat_c = 20.0
    w_deg_lat = 3.0  # ~333 km
    w_km = haversine_km(0.0, lat_c, 0.0, lat_c + w_deg_lat)
    w_deg_lon = w_deg_lat / np.cos(np.deg2rad(lat_c))
    inner = square(-40, -20, lat_c - 5, lat_c + 5)
    outer = square(-40 - w_deg_lon, -20 + w_deg_lon, lat_c - 5 - w_deg_lat, lat_c + 5 + w_deg_lat)

    p_two = write_region(tmp_path / "two.geojson", inner, outer, roles=["inner", "outer"])
    p_one = write_region(tmp_path / "one.geojson", inner)
    g_two = compose(region_project(p_two)).grid
    g_one = compose(region_project(p_one, transition_km=float(w_km))).grid

    assert grid_value(g_two, -30, 20) == RES == grid_value(g_one, -30, 20)
    # compare along the pure-east direction where the band width is exactly w
    for lon in (-19.0, -18.0, -17.5):
        a, b = grid_value(g_two, lon, lat_c), grid_value(g_one, lon, lat_c)
        assert a == pytest.approx(b, rel=0.05), f"lon={lon}: two-ring {a} vs single {b}"
    far = grid_value(g_two, -5, 20)
    assert far == BASE == grid_value(g_one, -5, 20)


def test_region_replace_vs_min(tmp_path):
    path = write_region(tmp_path / "r.geojson", square(-40, -20, 10, 30))
    proj = region_project(path, transition_km=300, mode="replace")
    proj.layers[0].resolution = 10.0  # base finer than the region target
    g = compose(proj).grid
    assert grid_value(g, -30, 20) == RES  # replace coarsens, fmesh semantics
    proj2 = region_project(path, transition_km=300, mode="min")
    proj2.layers[0].resolution = 10.0
    g2 = compose(proj2).grid
    assert grid_value(g2, -30, 20) == 10.0  # min keeps the finer base


def test_region_config_errors(tmp_path):
    p_one = write_region(tmp_path / "one.geojson", square(-40, -20, 10, 30))
    with pytest.raises(ValueError, match="transition_km"):
        compose(region_project(p_one))  # single ring without width
    p_two = write_region(
        tmp_path / "two.geojson",
        square(-40, -20, 10, 30),
        square(-45, -15, 5, 35),
        roles=["inner", "outer"],
    )
    with pytest.raises(ValueError, match="ambiguous"):
        compose(region_project(p_two, transition_km=100))
    p_three = write_region(
        tmp_path / "three.geojson",
        square(-40, -20, 10, 30),
        square(-45, -15, 5, 35),
        square(-50, -10, 0, 40),
    )
    with pytest.raises(ValueError, match="1 .*or 2"):
        compose(region_project(p_three, transition_km=100))


# --------------------------------------------------------------- field layer


def test_field_layer_netcdf_regrid():
    proj = Project(
        name="t",
        grid=GridSettings(n_lat=90, n_lon=180),  # 2 deg cells, coincide with 5 deg? no: sample analytically
        layers=[
            BaseLayer(type="base", resolution=BASE),
            FieldLayer(type="field", path=str(FIXTURES / "res_field.nc"), mode="replace"),
        ],
    )
    g = compose(proj).grid

    def expected(lon, lat):
        return 15.0 + lat * 0.5 + (lon + 60.0) * 0.1  # the fixture's analytic law

    for lon, lat in [(-30.0, 20.0), (-58.0, 2.0), (-16.0, 34.0)]:  # on the 2-deg grid
        got = grid_value(g, lon, lat)
        assert got == pytest.approx(expected(lon, lat), rel=1e-9), (lon, lat)
    assert grid_value(g, 100.0, 0.0) == BASE  # outside the field: untouched
    assert grid_value(g, -30.0, -20.0) == BASE


def test_field_layer_text(tmp_path):
    lon, lat = np.meshgrid(np.arange(-10.0, 11.0, 5.0), np.arange(0.0, 21.0, 5.0))
    vals = 30.0 + lat * 1.0
    txt = tmp_path / "field.txt"
    np.savetxt(txt, np.column_stack([lon.ravel(), lat.ravel(), vals.ravel()]))
    proj = Project(
        name="t",
        grid=GridSettings(n_lat=90, n_lon=180),
        layers=[
            BaseLayer(type="base", resolution=BASE),
            FieldLayer(type="field", path=str(txt)),
        ],
    )
    g = compose(proj).grid
    assert grid_value(g, 0.0, 10.0) == pytest.approx(40.0, rel=1e-9)
    assert grid_value(g, 100.0, 10.0) == BASE


def test_field_layer_errors(tmp_path):
    import xarray as xr

    ds = xr.Dataset(
        {
            "a": (("lat", "lon"), np.ones((3, 4))),
            "b": (("lat", "lon"), np.ones((3, 4))),
        },
        coords={"lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0, 3.0]},
    )
    path = tmp_path / "amb.nc"
    ds.to_netcdf(path)
    proj = Project(
        name="t",
        grid=GridSettings(n_lat=10, n_lon=20),
        layers=[BaseLayer(type="base", resolution=BASE), FieldLayer(type="field", path=str(path))],
    )
    with pytest.raises(ValueError) as exc:
        compose(proj)
    assert "'a'" in str(exc.value) and "'b'" in str(exc.value)
    proj.layers[1].variable = "b"
    assert compose(proj).grid is not None  # explicit variable resolves it

    bad = tmp_path / "bad.txt"
    np.savetxt(bad, np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0], [2.0, 0.0, 3.0]]))
    proj.layers[1] = FieldLayer(type="field", path=str(bad))
    with pytest.raises(ValueError, match="regular"):
        compose(proj)
