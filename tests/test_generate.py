from pathlib import Path

import numpy as np
import pytest

from fmesh2.core.io_fesom import read_fesom, write_fesom
from fmesh2.core.topology import boundary_edges
from fmesh2.generate import build_project, hfun_hash, run_jigsaw
from fmesh2.hfun import HfunGrid, compose
from fmesh2.project import BaseLayer, GridSettings, Project, RegionLayer

FIXTURES = Path(__file__).parent / "fixtures"
TOPO = str((FIXTURES / "topo_synthetic.nc").resolve())


def coarse_project(tmp_path, base=500.0, extra_layers=(), **kw) -> Project:
    proj = Project(
        name="e2e",
        topo=TOPO,
        grid=GridSettings(n_lat=90, n_lon=180),
        layers=[BaseLayer(type="base", resolution=base), *extra_layers],
        **kw,
    )
    proj.save(tmp_path / "proj")
    proj.ensure_layout()
    return proj


def test_hfun_hash_sensitivity():
    g1 = HfunGrid.regular(10, 20, fill=300.0)
    g2 = HfunGrid.regular(10, 20, fill=300.0)
    assert hfun_hash(g1) == hfun_hash(g2)
    g2.values[3, 4] = 299.0
    assert hfun_hash(g1) != hfun_hash(g2)
    g3 = HfunGrid.regular(20, 40, fill=300.0)
    assert hfun_hash(g1) != hfun_hash(g3)


@pytest.mark.timeout(300)
def test_jigsaw_cache_roundtrip(tmp_path):
    grid = compose(
        Project(
            name="c",
            grid=GridSettings(n_lat=45, n_lon=90),
            layers=[BaseLayer(type="base", resolution=1000)],
        )
    ).grid
    lines = []
    raw1 = run_jigsaw(grid, cache_dir=tmp_path, progress=lines.append)
    assert (tmp_path / f"jigsaw_{hfun_hash(grid)}.npz").exists()
    assert any("mesh" in ln.lower() or "jigsaw" in ln.lower() for ln in lines)
    cached_lines = []
    raw2 = run_jigsaw(grid, cache_dir=tmp_path, progress=cached_lines.append)
    assert any("cached" in ln for ln in cached_lines)
    np.testing.assert_array_equal(raw1.tri, raw2.tri)
    np.testing.assert_array_equal(raw1.lon, raw2.lon)


@pytest.mark.timeout(300)
def test_e2e_coarse_global_build(tmp_path):
    proj = coarse_project(tmp_path, base=300.0)
    mesh, result = build_project(proj)

    # output is FESOM2-valid: write, re-read, compare
    out = proj.out_dir
    write_fesom(mesh, out)
    back = read_fesom(out)
    np.testing.assert_array_equal(back.tri, mesh.tri)
    assert back.n_nodes == mesh.n_nodes

    # depths respect the minimum and stay positive-down
    assert mesh.depth.min() >= proj.minimum_depth
    assert mesh.depth.max() <= 4100.0  # synthetic floor is -4000

    # no degenerate elements: 3 distinct nodes each
    tri = mesh.tri
    assert np.all(tri[:, 0] != tri[:, 1])
    assert np.all(tri[:, 1] != tri[:, 2])
    assert np.all(tri[:, 0] != tri[:, 2])

    # single connected component (the enclosed lake was eliminated)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    graph = coo_matrix(
        (np.ones(edges.shape[0], np.int8), (edges[:, 0], edges[:, 1])),
        shape=(mesh.n_nodes, mesh.n_nodes),
    )
    n_comp, _ = connected_components(graph, directed=False)
    assert n_comp == 1

    # the lake interior (~20E 15N, below sea level in the fixture) has no nodes
    in_lake = (np.abs(mesh.lon - 20.0) < 2.0) & (np.abs(mesh.lat - 15.0) < 1.5)
    assert not in_lake.any()

    # land interiors have no nodes either (sample continent B's center)
    on_land = (np.abs(mesh.lon - 120.0) < 5.0) & (np.abs(mesh.lat + 25.0) < 5.0)
    assert not on_land.any()

    # coast flags = topological boundary
    b_nodes = np.zeros(mesh.n_nodes, bool)
    b_nodes[boundary_edges(tri, mesh.n_nodes).ravel()] = True
    np.testing.assert_array_equal(mesh.coast_flag.astype(bool), b_nodes)


@pytest.mark.timeout(300)
def test_e2e_region_refinement_shows_up(tmp_path):
    import json

    region = tmp_path / "proj" / "regions" / "patch.geojson"
    ring = [[-60, -20], [-30, -20], [-30, 5], [-60, 5], [-60, -20]]  # open ocean
    proj = coarse_project(
        tmp_path,
        base=500.0,
        extra_layers=(
            RegionLayer(type="region", path="regions/patch.geojson", resolution=200, transition_km=500),
        ),
    )
    region.parent.mkdir(parents=True, exist_ok=True)
    region.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {},
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                    }
                ],
            }
        )
    )
    mesh, _ = build_project(proj)
    from fmesh2.core.metrics import local_resolution_km

    res = local_resolution_km(mesh)
    inside = (
        (mesh.lon > -55) & (mesh.lon < -35) & (mesh.lat > -15) & (mesh.lat < 0)
    )
    outside = (
        (mesh.lon > 60) & (mesh.lon < 90) & (mesh.lat > -10) & (mesh.lat < 10)
    )
    assert inside.sum() > 10 and outside.sum() > 10
    assert np.median(res[inside]) < 0.55 * np.median(res[outside])
