"""Parity vs the legacy fmesh golden fixtures (tests/fixtures/fmesh_golden/).

The hfun/element tests feed fmesh2 the exact inputs of the golden run — the
captured smoothed coast points and region rings — so the only remaining
differences are algorithmic (spherical vs geopy distances, simultaneous vs
order-dependent band blending, bilinear vs gradapproach topo sampling).
They need the local RTopo NetCDF (fmesh checkout next to fmesh2) and are
skipped when it is absent; the estimator test runs anywhere.

Measured 2026-07-10 (see docs/acceptance/m2.md): hfun median rel diff 0.000%,
99.66% of cells within 5%; estimator +0.02%; elements +1.94%.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fmesh2.hfun import HfunGrid, compose, estimate_size
from fmesh2.project import (
    BaseLayer,
    BathyLayer,
    CoastLayer,
    GridSettings,
    MercatorLayer,
    Project,
    RegionLayer,
)

FIXTURES = Path(__file__).parent / "fixtures" / "fmesh_golden"
RTOPO = (
    Path(__file__).resolve().parents[2]
    / "fmesh" / "topo" / "RTopo-2.0.1_30sec_bos_fix_lowres_D3.nc"
)

needs_rtopo = pytest.mark.skipif(
    not RTOPO.exists(), reason=f"RTopo not found at {RTOPO} (run fmesh download_data.sh)"
)


@pytest.fixture(scope="module")
def golden():
    return np.load(FIXTURES / "hfun_field.npz")


def write_golden_inputs(workdir: Path, golden) -> list:
    """Golden coast points -> GeoJSON, golden rings -> region GeoJSONs."""
    coast = np.load(FIXTURES / "coast_points.npz")
    offsets = np.concatenate([[0], np.cumsum(coast["lengths"])])
    features = []
    for k in range(len(coast["lengths"])):
        seg = np.column_stack(
            [coast["lon"][offsets[k] : offsets[k + 1]], coast["lat"][offsets[k] : offsets[k + 1]]]
        )
        if seg.shape[0] >= 2:
            features.append(
                {
                    "type": "Feature",
                    "properties": {},
                    "geometry": {"type": "LineString", "coordinates": seg.tolist()},
                }
            )
    (workdir / "golden_coast.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features})
    )

    layers = []
    for name, res in (("Baffin_Bay", 30), ("Nares", 25), ("Peabody", 25)):
        feats = []
        for role in ("inner", "outer"):
            ring = np.column_stack(
                [golden[f"region_{name}_{role}_lon"], golden[f"region_{name}_{role}_lat"]]
            )
            feats.append(
                {
                    "type": "Feature",
                    "properties": {"role": role},
                    "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
                }
            )
        p = workdir / f"{name}.geojson"
        p.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
        layers.append(
            RegionLayer(type="region", name=name, mode="replace", path=str(p), resolution=res)
        )
    return layers


def parity_project(workdir: Path, golden) -> Project:
    regions = write_golden_inputs(workdir, golden)
    return Project(
        name="parity",
        topo=str(RTOPO),
        grid=GridSettings(n_lat=360, n_lon=720),
        layers=[
            BaseLayer(type="base", resolution=300),
            MercatorLayer(
                type="mercator", mode="replace", resolution=300,
                northern_boundary=60, northern_lowest_resolution=100,
                southern_boundary=-60, southern_lowest_resolution=100,
            ),
            CoastLayer(
                type="coast", mode="replace", min_resolution=50, max_distance=300,
                min_length=0, averaging=0,  # golden points are already fmesh-smoothed
                source=str(workdir / "golden_coast.geojson"),
            ),
            *regions,
            BathyLayer(
                type="bathy", mode="replace", min_resolution=50, min_depth=50, max_depth=1500
            ),
        ],
    )


def test_estimator_matches_fmesh_printout(golden):
    """Same semantics as the legacy 'estimated # of elements' console line."""
    grid = HfunGrid(lat=golden["latitudes"], lon=golden["longitudes"], values=golden["hfun"])
    est = estimate_size(grid)
    assert est.elements_including_land == 100925  # verbatim from run.log


@needs_rtopo
@pytest.mark.timeout(600)
def test_hfun_field_parity(tmp_path, golden):
    result = compose(parity_project(tmp_path, golden))
    mine = result.grid.values
    ref = golden["hfun"]
    rel = np.abs(mine - ref) / ref
    assert np.median(rel) < 0.005
    assert (rel < 0.05).mean() > 0.98  # measured: 99.66% of cells within 5%
    assert np.abs(mine - ref).mean() < 1.0  # km; measured 0.18

    est = estimate_size(result.grid)
    assert est.elements_including_land == pytest.approx(100925, rel=0.01)  # measured +0.02%


@needs_rtopo
@pytest.mark.timeout(900)
def test_element_count_parity(tmp_path, golden):
    """Full pipeline on the parity hfun: raw + final counts within 10% of the
    golden fmesh mesh (measured: raw -0.13%, final elements +1.94%)."""
    from fmesh2.data.topo import open_topo
    from fmesh2.generate import postprocess, run_jigsaw

    result = compose(parity_project(tmp_path, golden))
    raw = run_jigsaw(result.grid, cache_dir=tmp_path)
    assert raw.tri.shape[0] == pytest.approx(83502, rel=0.05)  # golden raw tria3

    mesh = postprocess(raw, open_topo(str(RTOPO)), minimum_depth=30.0, seeds=[(0.0, 0.0)])
    assert mesh.n_tri == pytest.approx(44443, rel=0.10)  # golden elem2d count
    assert mesh.n_nodes == pytest.approx(24549, rel=0.10)  # golden nod2d count
