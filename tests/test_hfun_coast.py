import json
import time

import numpy as np
import pytest

from fmesh2.data.coast_points import (
    CoastPoints,
    polyline_length_km,
    prepare_coast_points,
    smooth_polyline,
)
from fmesh2.hfun import compose
from fmesh2.hfun.coast_layer import _distance_cache, coast_distance_km
from fmesh2.project import BaseLayer, CoastLayer, GridSettings, Project

KM_PER_DEG_ARC = np.pi / 180.0 * 6371.0  # exact great-circle km per degree


def geojson(path, *lines):
    features = [
        {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "LineString", "coordinates": [[x, y] for x, y in line]},
        }
        for line in lines
    ]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def meridian(lon=0.0, lat0=-10.0, lat1=10.0, step=0.01):
    lats = np.arange(lat0, lat1 + step, step)
    return [(lon, la) for la in lats]


def coast_project(coast_path, n_lat=180, n_lon=360, **coast_kw):
    kw = dict(min_resolution=50, max_distance=300, min_length=0, averaging=0)
    kw.update(coast_kw)
    return Project(
        name="t",
        grid=GridSettings(n_lat=n_lat, n_lon=n_lon),
        layers=[
            BaseLayer(type="base", resolution=300),
            CoastLayer(type="coast", source=str(coast_path), **kw),
        ],
    )


# ------------------------------------------------------------- preprocessing


def test_polyline_length_analytic():
    line = np.array([[0.0, 0.0], [0.0, 10.0]])  # 10 deg meridian arc
    assert polyline_length_km(line) == pytest.approx(10 * KM_PER_DEG_ARC, rel=1e-12)


def test_short_island_excluded(tmp_path):
    big = meridian(lon=0.0, lat0=-10, lat1=10)  # ~2224 km
    tiny = [(50.0, 0.0), (50.3, 0.0), (50.3, 0.3), (50.0, 0.0)]  # ~100 km ring
    src = geojson(tmp_path / "c.geojson", big, tiny)
    pts = prepare_coast_points(src, min_length=500, averaging=0)
    assert pts.lengths.shape == (1,)  # only the long coastline survives
    assert np.all(np.abs(pts.lon) < 1)  # no points near the tiny island
    both = prepare_coast_points(src, min_length=50, averaging=0)
    assert both.lengths.shape == (2,)


def test_no_coastline_long_enough(tmp_path):
    src = geojson(tmp_path / "c.geojson", [(0, 0), (0.1, 0)])
    with pytest.raises(ValueError, match="min_length"):
        prepare_coast_points(src, min_length=500, averaging=0)


def test_smoothing_thins_and_stays_on_line():
    line = np.column_stack([np.zeros(4001), np.linspace(-10, 10, 4001)])  # 0.005 deg steps
    sm = smooth_polyline(line, averaging_km=50)
    assert sm.shape[0] < line.shape[0] / 40  # heavily thinned
    assert np.allclose(sm[:, 0], 0.0)  # stays on the meridian
    assert np.all(np.diff(sm[:-1, 1]) > 0)  # still ordered along the line
    step = np.diff(sm[:-1, 1]) * KM_PER_DEG_ARC
    assert np.all(step < 2 * 50) and step.mean() == pytest.approx(50, rel=0.2)
    assert np.array_equal(sm[0], sm[-1])  # ring closed (fmesh convention)


def test_smoothing_across_dateline_no_artifacts():
    lon = np.concatenate([np.linspace(178, 180, 500), np.linspace(-180, -178, 500)[1:]])
    line = np.column_stack([lon, np.zeros(lon.size)])
    sm = smooth_polyline(line, averaging_km=50)
    assert np.all((sm[:, 0] <= 180) & (sm[:, 0] >= -180))
    # averaged points stay near the dateline: nothing snaps to ~0 deg
    assert np.all((np.abs(sm[:, 0]) > 177) | np.isclose(np.abs(sm[:, 0]), 180))
    assert np.allclose(sm[:, 1], 0.0)


def test_tiny_ring_fallback():
    line = np.array([[10.0, 0.0], [10.05, 0.0], [10.05, 0.05], [10.0, 0.05]])
    sm = smooth_polyline(line, averaging_km=50)  # all points in one bin
    assert sm.shape[0] == 4  # 3-point triangle + closure
    assert np.array_equal(sm[0], sm[-1])
    eps = 1e-9  # lon wrap adds float noise
    assert np.all((sm[:, 0] >= 10.0 - eps) & (sm[:, 0] <= 10.05 + eps))


# ------------------------------------------------------------ distance field


def test_straight_coast_analytic_ramp(tmp_path):
    src = geojson(tmp_path / "m.geojson", meridian())
    result = compose(coast_project(src, n_lat=360, n_lon=720))
    g = result.grid
    equator = g.values[np.searchsorted(g.lat, 0.0)]

    def at_lon(x):
        return equator[np.searchsorted(g.lon, x)]

    for lon in (0.5, 1.0, 1.5, 2.0, 2.5):
        d = lon * KM_PER_DEG_ARC  # exact arc distance on the equator row
        expected = 50 + (300 - 50) * d / 300.0
        assert at_lon(lon) == pytest.approx(expected, rel=0.01)
        assert at_lon(-lon) == pytest.approx(expected, rel=0.01)
    assert at_lon(0.0) == pytest.approx(50, rel=0.01)
    assert at_lon(3.0) == 300.0  # beyond max_distance: base untouched
    assert at_lon(120.0) == 300.0


def test_polar_rows_untouched(tmp_path):
    src = geojson(tmp_path / "m.geojson", meridian(lat0=80, lat1=90))
    result = compose(coast_project(src, n_lat=180, n_lon=360))
    g = result.grid
    polar = np.abs(g.lat) >= 85
    assert np.all(g.values[polar] == 300.0)  # fmesh parity: |lat| >= 85 untouched
    near = (g.lat > 80) & (g.lat < 85)
    assert g.values[near].min() < 100  # refinement clearly active below the limit


def test_distance_field_cache(tmp_path):
    src = geojson(tmp_path / "m.geojson", meridian())
    _distance_cache.clear()
    proj = coast_project(src)
    r1 = compose(proj)
    assert len(_distance_cache) == 1
    proj.layers[1].max_distance = 150  # ramp param change must reuse distances
    r2 = compose(proj)
    assert len(_distance_cache) == 1
    assert not np.array_equal(r1.grid.values, r2.grid.values)


def test_coast_distance_direct():
    pts = CoastPoints(lon=np.array([0.0]), lat=np.array([0.0]), lengths=np.array([1]))
    from fmesh2.hfun.grid import HfunGrid

    grid = HfunGrid.regular(36, 72)
    d = coast_distance_km(pts, grid)
    j = np.searchsorted(grid.lat, 0.0)
    i = np.searchsorted(grid.lon, 10.0)
    assert d[j, i] == pytest.approx(10 * KM_PER_DEG_ARC, rel=1e-6)
    assert np.isinf(d[0, 0])  # polar row


# -------------------------------------------------------------------- perf


def test_perf_smoke_ne10m_half_degree():
    """Full compose (base + NE-10m coast) on a global 0.5 deg grid < 10 s."""
    try:
        from fmesh2.data.coastlines import default_coastline_path

        src = default_coastline_path(download=True)
    except (RuntimeError, OSError) as exc:  # offline and not cached
        pytest.skip(f"NE-10m coastline unavailable: {exc}")
    proj = Project(
        name="perf",
        grid=GridSettings(n_lat=360, n_lon=720),
        layers=[
            BaseLayer(type="base", resolution=300),
            CoastLayer(
                type="coast",
                source=str(src),
                min_resolution=10,
                max_distance=100,
                min_length=100,
                averaging=50,
            ),
        ],
    )
    _distance_cache.clear()
    t0 = time.perf_counter()
    result = compose(proj)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"cold compose took {elapsed:.1f}s"
    v = result.grid.values
    assert v.min() >= 10 and v.max() == 300
    assert (v < 100).sum() > 1000  # plenty of coastal refinement happened
