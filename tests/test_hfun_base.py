import numpy as np
import pytest

from fmesh2.hfun import (
    HfunGrid,
    compose,
    estimate_size,
    finest_scale,
    suggest_intervals,
)
from fmesh2.hfun.grid import KM_PER_DEG, MAX_AUTO_INTERVALS
from fmesh2.project import BaseLayer, CoastLayer, GridSettings, MercatorLayer, Project


def project_with(layers, n_lat=90, n_lon=180) -> Project:
    return Project(name="t", grid=GridSettings(n_lat=n_lat, n_lon=n_lon), layers=layers)


def base(res, **kw) -> BaseLayer:
    return BaseLayer(type="base", resolution=res, **kw)


MERC = dict(
    type="mercator",
    resolution=300,
    northern_boundary=60,
    northern_lowest_resolution=100,
    southern_boundary=-60,
    southern_lowest_resolution=80,
)


# ---------------------------------------------------------------------- grid


def test_regular_grid():
    g = HfunGrid.regular(180, 360)
    assert g.shape == (181, 361)
    assert g.lat[0] == -90 and g.lat[-1] == 90
    assert g.lon[0] == -180 and g.lon[-1] == 180
    assert np.all(np.isinf(g.values))
    dlat, dlon = g.cell_km
    assert dlat == pytest.approx(KM_PER_DEG)
    assert dlon == pytest.approx(KM_PER_DEG)


def test_grid_validation():
    with pytest.raises(ValueError, match="ascending"):
        HfunGrid(lat=np.array([0.0, 0.0]), lon=np.array([0.0, 1.0]), values=np.zeros((2, 2)))
    with pytest.raises(ValueError, match="shape"):
        HfunGrid(lat=np.array([0.0, 1.0]), lon=np.array([0.0, 1.0]), values=np.zeros((3, 2)))


def test_suggest_intervals():
    n_lat, n_lon = suggest_intervals(300)
    assert n_lon == 2 * n_lat
    assert 180 * KM_PER_DEG / n_lat <= 300  # cell no larger than the target
    assert suggest_intervals(1.0) == (MAX_AUTO_INTERVALS, 2 * MAX_AUTO_INTERVALS)
    with pytest.raises(ValueError):
        suggest_intervals(0)
    with pytest.raises(ValueError):
        suggest_intervals(np.inf)


# -------------------------------------------------------------------- layers


def test_base_layer():
    result = compose(project_with([base(300)]))
    assert result.grid.values.shape == (91, 181)
    assert np.all(result.grid.values == 300.0)
    assert result.warnings == []


def test_mercator_analytic_values():
    result = compose(project_with([base(300), MercatorLayer(**MERC)], n_lat=180, n_lon=360))
    g = result.grid
    v = g.values

    def row(lat):
        j = np.searchsorted(g.lat, lat)
        assert g.lat[j] == lat  # exact grid row
        return v[j]

    assert np.all(row(0) == 300.0)
    np.testing.assert_allclose(row(45), 300 * np.cos(np.deg2rad(45)))
    np.testing.assert_allclose(row(-45), 300 * np.cos(np.deg2rad(45)))
    assert np.all(row(60) == 100.0)  # fmesh cap uses >= at the boundary
    assert np.all(row(90) == 100.0)
    assert np.all(row(-60) == 80.0)
    assert np.all(row(-90) == 80.0)


def test_mercator_matches_fmesh_loop():
    """Exact parity with fmesh define_resolutions' mercator branch."""
    result = compose(
        project_with([MercatorLayer(**{**MERC, "mode": "replace"})], n_lat=37, n_lon=73)
    )
    g = result.grid
    expected = np.full(g.shape, np.nan)
    for j, lat in enumerate(g.lat):  # fmesh loop, verbatim semantics
        expected[j, :] = 300 * np.cos(np.deg2rad(lat))
        if lat >= 60:
            expected[j, :] = 100
        elif lat <= -60:
            expected[j, :] = 80
    np.testing.assert_allclose(g.values, expected, rtol=1e-14)


def test_min_vs_replace_on_overlapping_layers():
    v = compose(project_with([base(300), base(100)])).grid.values
    assert np.all(v == 100.0)
    v = compose(project_with([base(300), base(100), base(400)])).grid.values
    assert np.all(v == 100.0)  # min keeps the finer field
    v = compose(project_with([base(300), base(100), base(400, mode="replace")])).grid.values
    assert np.all(v == 400.0)  # replace overwrites


def test_disabled_layer_skipped():
    v = compose(project_with([base(300), base(50, enabled=False)])).grid.values
    assert np.all(v == 300.0)


def test_unimplemented_layer_type_raises():
    proj = project_with([base(300), CoastLayer(type="coast", min_resolution=10, max_distance=100)])
    with pytest.raises(NotImplementedError, match="coast"):
        compose(proj)


def test_finest_scale():
    assert finest_scale(base(300)) == 300
    assert finest_scale(MercatorLayer(**MERC)) == 80
    assert finest_scale(CoastLayer(type="coast", min_resolution=10, max_distance=100)) == 10


# ----------------------------------------------------------------- estimator


def test_estimator_uniform_closed_form():
    """Uniform field: total elements ~ sphere area / element area, within 2%."""
    result = compose(project_with([base(300)], n_lat=180, n_lon=360))
    est = estimate_size(result.grid)
    sphere_area = 4 * np.pi * 6371.0**2
    element_area = 0.7 * 300.0**2 / 2
    assert est.elements_including_land == pytest.approx(sphere_area / element_area, rel=0.02)


def test_estimator_matches_fmesh_loop_exactly():
    result = compose(project_with([base(300), MercatorLayer(**MERC)], n_lat=36, n_lon=72))
    g = result.grid
    total = 0.0  # fmesh estimate_number_of_nodes, verbatim
    dlat = abs(g.lat[1] - g.lat[0])
    dlon = abs(g.lon[1] - g.lon[0])
    for j, lat in enumerate(g.lat):
        for i, _ in enumerate(g.lon):
            area = dlat * 111 * dlon * 111 * np.cos(np.deg2rad(lat))
            element_area = 0.7 * (g.values[j, i] ** 2) / 2
            total += area / element_area
    est = estimate_size(g)
    assert est.elements_including_land == int(np.round(total))


def test_estimator_ocean_nodes_derivation():
    result = compose(project_with([base(300)]))
    est = estimate_size(result.grid, ocean_fraction=0.71)
    assert est.ocean_nodes == int(np.round(est.elements_including_land / 2 * 0.71))
    assert est.ocean_fraction == 0.71


def test_estimator_on_golden_hfun():
    """Vectorized estimator reproduces the legacy printout on the captured field."""
    from pathlib import Path

    d = np.load(Path(__file__).parent / "fixtures" / "fmesh_golden" / "hfun_field.npz")
    grid = HfunGrid(lat=d["latitudes"], lon=d["longitudes"], values=d["hfun"])
    est = estimate_size(grid)
    assert est.elements_including_land == 100925  # from run.log


# ------------------------------------------------------------------ warnings


def test_too_coarse_warning():
    result = compose(project_with([base(50)], n_lat=90, n_lon=180))  # 222 km cells
    assert len(result.warnings) == 1
    w = result.warnings[0]
    assert "too coarse" in w.message
    assert w.suggested_n_lat >= 180 * KM_PER_DEG / 50
    assert w.suggested_n_lon == 2 * w.suggested_n_lat


def test_undefined_cells_warning():
    proj = Project(name="t", grid=GridSettings(n_lat=10, n_lon=20), layers=[])
    result = compose(proj)
    assert any("add a base layer" in w.message for w in result.warnings)


def test_auto_grid_from_finest_layer():
    proj = Project(name="t", layers=[base(300)])  # no explicit grid
    result = compose(proj)
    n_lat = result.grid.lat.size - 1
    assert 180 * KM_PER_DEG / n_lat <= 300
    assert result.grid.lon.size - 1 == 2 * n_lat
    assert result.warnings == []  # fine grid, no cap hit


def test_auto_grid_without_scales_raises():
    proj = Project(name="t", layers=[])
    with pytest.raises(ValueError, match="auto-suggest"):
        compose(proj)
