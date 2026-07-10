from pathlib import Path

import numpy as np
import pytest

from fmesh2.core.metrics import local_resolution_km, triangle_min_angle_deg
from fmesh2.core.topology import boundary_edges
from fmesh2.edit.cavity import extract_cavity
from fmesh2.edit.local_remesh import (
    RemeshError,
    build_local_hfun,
    remesh_cavity,
    seam_hfun_at,
    write_mapping_csv,
)

OPEN_OCEAN = [(-42.0, 28.0), (-28.0, 28.0), (-28.0, 36.0), (-42.0, 36.0)]
COASTAL = [(-21.0, 15.0), (-13.0, 15.0), (-13.0, 23.0), (-21.0, 23.0)]

RTOPO = (
    Path(__file__).resolve().parents[2]
    / "fmesh" / "topo" / "RTopo-2.0.1_30sec_bos_fix_lowres_D3.nc"
)
needs_rtopo = pytest.mark.skipif(
    not RTOPO.exists(), reason=f"RTopo not found at {RTOPO} (run fmesh download_data.sh)"
)


@pytest.fixture(scope="module")
def core2(core2_dir):
    from fmesh2.core.io_fesom import read_fesom

    return read_fesom(core2_dir)


@pytest.fixture(scope="module")
def refined(core2, tmp_path_factory):
    """One shared open-ocean 3x refinement (jigsaw run reused across tests)."""
    cav = extract_cavity(core2, OPEN_OCEAN)
    target = float(np.median(cav.edge_len_km)) / 3.0
    work = tmp_path_factory.mktemp("remesh_open")
    res = remesh_cavity(core2, cav, target, cache_dir=work)
    return cav, target, res


def _outside_nodes(mesh, cav):
    return np.unique(mesh.tri[~cav.elem_mask])


def test_outside_bit_identical(core2, refined):
    cav, _, res = refined
    outside = _outside_nodes(core2, cav)
    m = res.mapping[outside]
    assert (m >= 0).all()
    assert np.array_equal(res.mesh.lon[m], core2.lon[outside])
    assert np.array_equal(res.mesh.lat[m], core2.lat[outside])
    assert np.array_equal(res.mesh.depth[m], core2.depth[outside])
    assert np.array_equal(res.mesh.coast_flag[m], core2.coast_flag[outside])
    # dropped nodes are exactly the old cavity-interior nodes
    cavity_only = np.setdiff1d(np.unique(core2.tri[cav.elem_mask]), outside)
    assert set(np.flatnonzero(res.mapping == -1)) == set(cavity_only.tolist())


def test_seam_watertight(core2, refined):
    cav, _, res = refined
    assert res.report["watertight_seam"]
    assert res.report["seam_edges"] == len(cav.ring_edges)  # open ocean: all seam
    # no boundary change anywhere: same boundary edge set as the original
    def bset(mesh, ids=None):
        be = boundary_edges(mesh.tri, mesh.n_nodes)
        return {tuple(e) for e in np.sort(be, axis=1).tolist()}

    orig = {
        tuple(sorted((int(res.mapping[a]), int(res.mapping[b]))))
        for a, b in boundary_edges(core2.tri, core2.n_nodes).tolist()
    }
    assert bset(res.mesh) == orig


def test_interior_resolution_and_quality(core2, refined):
    cav, target, res = refined
    n_new = res.report["n_new_interior_nodes"]
    assert n_new > 0
    interior = np.arange(res.mesh.n_nodes - n_new, res.mesh.n_nodes)
    # deep interior (beyond every seam band) reaches the target within 20%
    from fmesh2.core.projection import stereo_xy

    x, y = stereo_xy(
        res.mesh.lon[interior], res.mesh.lat[interior], cav.lon0, cav.lat0
    )
    _, d = seam_hfun_at(np.column_stack([x, y]), cav, target)
    deep = d > 3.0 * float(cav.edge_len_km.max())
    assert deep.any()
    r = local_resolution_km(res.mesh)[interior]
    assert abs(np.median(r[deep]) / target - 1.0) < 0.20
    assert res.report["interior_resolution_km"] < 0.6 * float(
        np.median(cav.edge_len_km)
    )  # clearly finer than before
    # element quality of the new region
    assert res.report["min_angle_deg"] > 15.0
    assert res.report["median_min_angle_deg"] > 40.0


def test_mapping_csv_and_fesom_roundtrip(core2, refined, tmp_path):
    cav, _, res = refined
    write_mapping_csv(tmp_path / "mapping.csv", res.mapping)
    lines = (tmp_path / "mapping.csv").read_text().splitlines()
    assert lines[0] == "old_id,new_id"
    assert len(lines) == core2.n_nodes + 1
    table = np.loadtxt(tmp_path / "mapping.csv", delimiter=",", skiprows=1, dtype=np.int64)
    assert (table[:, 0] == np.arange(1, core2.n_nodes + 1)).all()
    assert (table[:, 1] == np.where(res.mapping >= 0, res.mapping + 1, -1)).all()

    from fmesh2.core.io_fesom import read_fesom, write_fesom

    write_fesom(res.mesh, tmp_path / "mesh")
    back = read_fesom(tmp_path / "mesh")
    assert np.array_equal(back.tri, res.mesh.tri)
    assert np.array_equal(back.lon, res.mesh.lon)
    assert np.array_equal(back.depth, res.mesh.depth)


def test_reproducible_via_cache(core2, refined, tmp_path):
    cav, target, res = refined
    res2 = remesh_cavity(core2, cav, target, cache_dir=tmp_path)
    res3 = remesh_cavity(core2, cav, target, cache_dir=tmp_path)  # cache hit
    assert np.array_equal(res2.mesh.lon, res3.mesh.lon)
    assert np.array_equal(res2.mesh.lat, res3.mesh.lat)
    assert np.array_equal(res2.mesh.tri, res3.mesh.tri)
    assert np.array_equal(res2.mesh.depth, res3.mesh.depth)


def test_coarsening_runs(core2, tmp_path):
    cav = extract_cavity(core2, OPEN_OCEAN)
    target = 2.0 * float(np.median(cav.edge_len_km))
    res = remesh_cavity(core2, cav, target, cache_dir=tmp_path)
    assert res.report["watertight_seam"]
    outside = _outside_nodes(core2, cav)
    assert np.array_equal(res.mesh.lon[res.mapping[outside]], core2.lon[outside])
    # coarser: fewer elements than the original cavity had
    n_new_tris = res.mesh.n_tri - (core2.n_tri - int(cav.elem_mask.sum()))
    assert n_new_tris < int(cav.elem_mask.sum())


def test_interpolated_depths_plausible(core2, refined):
    cav, _, res = refined
    n_new = res.report["n_new_interior_nodes"]
    new_depth = res.mesh.depth[res.mesh.n_nodes - n_new:]
    old_cav_depth = core2.depth[np.unique(core2.tri[cav.elem_mask])]
    assert np.isfinite(new_depth).all()
    assert new_depth.min() >= old_cav_depth.min() - 1e-9
    assert new_depth.max() <= old_cav_depth.max() + 1e-9


def test_target_field_overrides_constant(core2):
    cav = extract_cavity(core2, OPEN_OCEAN)
    target = float(np.median(cav.edge_len_km)) / 3.0
    xg, yg, h_const = build_local_hfun(cav, target)
    xg2, yg2, h_field = build_local_hfun(
        cav, target, target_field=lambda lon, lat: np.full(lon.shape, 2.0 * target)
    )
    assert np.array_equal(xg, xg2) and np.array_equal(yg, yg2)
    # far from the seam the field value wins; the seam value stays protected
    xx, yy = np.meshgrid(xg, yg)
    _, d = seam_hfun_at(
        np.column_stack([xx.ravel(), yy.ravel()]), cav, target
    )
    deep = (d > 3.0 * float(cav.edge_len_km.max())).reshape(h_const.shape)
    assert np.allclose(h_const[deep], target)
    assert np.allclose(h_field[deep], 2.0 * target)
    near = (d < 1e-6).reshape(h_const.shape)
    if near.any():
        assert np.allclose(h_field[near], h_const[near])


def test_bad_targets(core2):
    cav = extract_cavity(core2, OPEN_OCEAN)
    with pytest.raises(RemeshError, match="positive"):
        build_local_hfun(cav, -5.0)
    with pytest.raises(RemeshError, match="grid"):
        build_local_hfun(cav, 0.02)  # absurdly fine -> grid cap


@needs_rtopo
def test_refine_coastal_with_topo(core2, tmp_path):
    from fmesh2.data.topo import Topo

    cav = extract_cavity(core2, COASTAL)
    target = float(np.median(cav.edge_len_km)) / 3.0
    topo = Topo(RTOPO)
    res = remesh_cavity(
        core2, cav, target, topo=topo, minimum_depth=5.0, cache_dir=tmp_path
    )
    assert res.report["watertight_seam"]
    outside = _outside_nodes(core2, cav)
    assert np.array_equal(res.mesh.lon[res.mapping[outside]], core2.lon[outside])
    assert np.array_equal(res.mesh.depth[res.mapping[outside]], core2.depth[outside])
    # no land elements away from the seam strip (seam-adjacent elements are
    # exempt from the land cut: RTopo may disagree with core2's coastline)
    n_new = res.report["n_new_interior_nodes"]
    interior = np.arange(res.mesh.n_nodes - n_new, res.mesh.n_nodes)
    n_outside_tris = core2.n_tri - int(cav.elem_mask.sum())
    new_tris = res.mesh.tri[n_outside_tris:]
    seam_new = res.mapping[
        cav.ring_nodes[np.unique(cav.ring_edges[~cav.edge_is_coast])]
    ]
    seam_adjacent = np.isin(new_tris, seam_new).any(axis=1)
    in_seam_strip = np.zeros(res.mesh.n_nodes, dtype=bool)
    in_seam_strip[new_tris[seam_adjacent].ravel()] = True
    free = interior[~in_seam_strip[interior]]
    elev = topo.elevation(res.mesh.lon[free], res.mesh.lat[free])
    assert (elev < 0.0).all()
    # depths follow the host convention (core2: negative down), clamped
    assert (res.mesh.depth[interior] <= -5.0).all()
    # coastline detail improves: strictly more boundary edges than before
    n_bnd_before = len(boundary_edges(core2.tri, core2.n_nodes))
    n_bnd_after = len(boundary_edges(res.mesh.tri, res.mesh.n_nodes))
    assert n_bnd_after > n_bnd_before


def test_unreachable_target_is_reported(core2, refined, tmp_path):
    """A target far below what the seam band allows must be flagged.

    Small open-ocean polygon (all-seam ring — coast edges carry no band
    since they are subdividable) + 1 km target on a ~110 km mesh: the
    frozen-seam transition band (~3x seam edge) fills the cavity, so the
    sizing never approaches the target — the report must say so instead
    of silently delivering ~the old resolution.
    """
    tight = [(-35.8, 29.2), (-34.2, 29.2), (-34.2, 30.8), (-35.8, 30.8)]
    cav = extract_cavity(core2, tight)
    assert not cav.edge_is_coast.any()  # fully band-limited ring
    res = remesh_cavity(core2, cav, 1.0, cache_dir=tmp_path)
    rep = res.report
    assert rep["target_reached"] is False
    assert rep["reachable_min_km"] > 10 * rep["target_km"]
    assert rep["watertight_seam"]  # still a valid (just barely refined) mesh
    # an achievable target on the big open-ocean cavity reports reached
    big = refined[2].report
    assert big["target_reached"] is True
    assert big["reachable_min_km"] == pytest.approx(big["target_km"])


def test_single_node_delete_via_star_remesh(core2, tmp_path):
    """Tight polygon around one node + coarser target = delete that node."""
    from fmesh2.core.topology import SphereKDTree

    _, idx = SphereKDTree(core2.lon, core2.lat).nearest_km(-35.0, 30.0)
    n = int(np.atleast_1d(idx)[0])
    lon0, lat0 = float(core2.lon[n]), float(core2.lat[n])
    box = [(lon0 - 0.15, lat0 - 0.15), (lon0 + 0.15, lat0 - 0.15),
           (lon0 + 0.15, lat0 + 0.15), (lon0 - 0.15, lat0 + 0.15)]
    cav = extract_cavity(core2, box)
    target = 1.5 * float(np.median(cav.edge_len_km))
    res = remesh_cavity(core2, cav, target, cache_dir=tmp_path)
    assert res.mapping[n] == -1  # the node is gone
    assert res.report["watertight_seam"]
    assert res.mesh.n_nodes == core2.n_nodes - 1 + res.report["n_new_interior_nodes"]


GIBRALTAR = [(-8.0, 34.5), (-3.0, 34.5), (-3.0, 37.5), (-8.0, 37.5)]


def _strait_stats(m):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    s = (m.lon > -6.2) & (m.lon < -4.8) & (m.lat > 35.6) & (m.lat < 36.4)
    e = np.concatenate([m.tri[:, [0, 1]], m.tri[:, [1, 2]], m.tri[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(m.n_nodes, m.n_nodes))
    res = local_resolution_km(m)
    return float(np.median(res[s])), connected_components(g, directed=False)[0]


def test_strait_refines_along_frozen_shoreline(core2, tmp_path):
    """Coast segments carry no seam band: a coast-bound strait CAN refine.

    Without topo the shoreline SHAPE is preserved exactly — jigsaw only
    subdivides the frozen coast polyline — and Gibraltar (1-2 elements
    wide in core2) actually reaches ~the target instead of staying pinned
    to the old coastline spacing.
    """
    import shapely

    from fmesh2.core.projection import stereo_xy
    from fmesh2.core.topology import boundary_edges

    cav = extract_cavity(core2, GIBRALTAR)
    res = remesh_cavity(core2, cav, 12.0, cache_dir=tmp_path)
    assert res.report["watertight_seam"]
    med, n_comp = _strait_stats(res.mesh)
    assert n_comp == 1  # the strait stays open
    assert med < 25.0  # was pinned at ~50 km before the coast exemption

    # every new boundary vertex lies ON the old coast polyline (bit-frozen shape)
    coast_lines = shapely.multilinestrings(
        [cav.ring_xy[e] for e in cav.ring_edges[cav.edge_is_coast]]
    )
    n_kept = int((res.mapping >= 0).sum())
    be = boundary_edges(res.mesh.tri, res.mesh.n_nodes)
    new_bnodes = np.unique(be[be >= n_kept])
    assert new_bnodes.size > 0  # the coastline did get subdivided
    x, y = stereo_xy(
        res.mesh.lon[new_bnodes], res.mesh.lat[new_bnodes], cav.lon0, cav.lat0
    )
    d = shapely.distance(shapely.points(np.column_stack([x, y])), coast_lines)
    assert d.max() < 1e-9  # km


@needs_rtopo
def test_strait_carved_open_with_topo(core2, tmp_path):
    """RTopo land cut at a target finer than the real strait keeps it open."""
    from fmesh2.data.topo import Topo

    cav = extract_cavity(core2, GIBRALTAR)
    res = remesh_cavity(core2, cav, 5.0, topo=Topo(RTOPO), cache_dir=tmp_path)
    assert res.report["watertight_seam"]
    med, n_comp = _strait_stats(res.mesh)
    assert n_comp == 1  # Atlantic and Mediterranean stay connected
    assert med < 12.0  # real ~14 km channel resolved by ~5-8 km elements
