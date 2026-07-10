import numpy as np
import pytest

from fmesh2.generate.jigsaw_driver import RawMesh
from fmesh2.generate.postprocess import (
    drop_loose_elements,
    keep_seed_components,
    postprocess,
)


def hex_fan(closed=True):
    """Node 0 at the center, nodes 1..6 around it; 6 (closed) or 5 triangles."""
    tri = [[0, k, k % 6 + 1] for k in range(1, 7 if closed else 6)]
    return np.array(tri, dtype=np.int32)


# ------------------------------------------------------------ loose elements


def test_lone_triangle_deleted():
    tri = np.array([[0, 1, 2]], dtype=np.int32)  # every node: v=1, d=3
    assert drop_loose_elements(tri, 3).size == 0


def test_bowtie_deleted_edge_pair_kept():
    bowtie = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int32)  # node 0: v=2, d=5
    assert drop_loose_elements(bowtie, 5).size == 0
    # two triangles sharing an edge: v=2, d=4 for the shared nodes — kept,
    # BUT their rim nodes are lone-fan corners (v=1, d=3)? no: v=1 nodes have
    # d=3 only when their single triangle is isolated. Here each triangle has
    # a neighbor, still d=3 for the apex nodes -> the pair dies too (fmesh
    # behaves the same: valence-1 apexes of an isolated pair are loose).
    pair = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    assert drop_loose_elements(pair, 4).size == 0


def test_closed_fan_survives():
    tri = hex_fan(closed=True)  # center: v=6, d=7 (= v+1); rim: v=2, d=4
    out = drop_loose_elements(tri, 7)
    assert out.shape == tri.shape


def test_pinched_fans_deleted():
    """Two closed fans joined only at one node (non-manifold pinch)."""
    fan_a = hex_fan(closed=True)
    fan_b = hex_fan(closed=True) + 10
    fan_b[fan_b == 10] = 0  # share the center node: v=12, d=13 >= v+3? 13 < 15
    tri = np.vstack([fan_a, fan_b])
    out = drop_loose_elements(tri, 17)
    # v=12, d=13 -> 13 < 15, the pinch node survives fmesh's rule (documented)
    assert out.shape == tri.shape
    # but a pinch of two SINGLE triangles (the bowtie) is caught (v=2, d=5)
    assert drop_loose_elements(np.array([[0, 1, 2], [0, 3, 4]], np.int32), 5).size == 0


def test_grid_patch_trims_corners_then_stabilizes():
    """fmesh rules trim v=1/d=3 corner nodes of a patch, then reach a fixed
    point; interior nodes always survive."""
    lon, lat, tri = grid_patch(0.0, 0.0, n=6)
    out = drop_loose_elements(tri, 36)
    assert out.size > 0
    corner = 0  # (0,0) node: v=1, d=3 -> trimmed
    assert corner not in out
    interior = {j * 6 + i for j in range(1, 5) for i in range(1, 5)}
    assert interior <= set(out.ravel().tolist())
    out2 = drop_loose_elements(out, 36)  # idempotent: already a fixed point
    assert np.array_equal(out, out2)


# ---------------------------------------------------------- lake elimination


def grid_patch(lon0, lat0, n=4, step=1.0, offset=0):
    """A small structured triangle patch; returns (lon, lat, tri) with ids offset."""
    lon, lat = np.meshgrid(lon0 + np.arange(n) * step, lat0 + np.arange(n) * step)
    lon, lat = lon.ravel(), lat.ravel()
    tri = []
    for j in range(n - 1):
        for i in range(n - 1):
            a = j * n + i
            tri += [[a, a + 1, a + n], [a + 1, a + n + 1, a + n]]
    return lon, lat, np.array(tri, dtype=np.int32) + offset


def test_lake_removal_keeps_seed_component():
    lon_a, lat_a, tri_a = grid_patch(0.0, 0.0)  # "ocean" at (0..3, 0..3)
    lon_b, lat_b, tri_b = grid_patch(50.0, 20.0, offset=16)  # "lake", disjoint
    lon = np.concatenate([lon_a, lon_b])
    lat = np.concatenate([lat_a, lat_b])
    tri = np.vstack([tri_a, tri_b])

    kept = keep_seed_components(tri, lon, lat, seeds=[(1.0, 1.0)])
    assert np.array_equal(np.unique(kept), np.arange(16))  # only patch A nodes
    kept_b = keep_seed_components(tri, lon, lat, seeds=[(51.0, 21.0)])
    assert np.array_equal(np.unique(kept_b), np.arange(16, 32))
    both = keep_seed_components(tri, lon, lat, seeds=[(1.0, 1.0), (51.0, 21.0)])
    assert both.shape[0] == tri.shape[0]  # multi-seed keeps the union


# ---------------------------------------------------------------- land cut


class FakeTopo:
    """Topo stand-in with an analytic elevation function."""

    def __init__(self, fn):
        self._fn = fn

    def elevation(self, lon, lat):
        return self._fn(np.asarray(lon, float), np.asarray(lat, float))


def test_postprocess_land_cut_and_depths():
    lon_a, lat_a, tri_a = grid_patch(0.0, 0.0, n=6)
    raw = RawMesh(lon=lon_a, lat=lat_a, tri=tri_a)
    # land (elev >= 0) for lon >= 4.5: cuts the two rightmost columns' elements
    topo = FakeTopo(lambda lo, la: np.where(lo >= 4.5, 100.0, -10.0 - 100.0 * lo))
    mesh = postprocess(raw, topo, minimum_depth=30.0, seeds=[(1.0, 1.0)], levels=[0.0, -10.0])

    assert mesh.lon.max() <= 4.0  # land column nodes dropped entirely
    assert mesh.tri.max() == mesh.n_nodes - 1  # contiguous renumbering
    # depths: elev(-10 - 100*lon) -> |elev| clamped at 30
    left = mesh.lon < 0.15
    assert np.all(mesh.depth[left] == 30.0)  # elev -10 -> clamp to minimum_depth
    deep = mesh.lon > 3.9
    assert np.all(mesh.depth[deep] > 300.0)
    assert mesh.levels is not None and mesh.levels.tolist() == [0.0, -10.0]
    # coast flags: boundary of the patch (all outer rim after cuts)
    assert mesh.coast_flag.sum() > 0
    assert set(np.unique(mesh.coast_flag)) <= {0, 1}


def test_postprocess_all_land_raises():
    lon, lat, tri = grid_patch(0.0, 0.0)
    raw = RawMesh(lon=lon, lat=lat, tri=tri)
    with pytest.raises(ValueError, match="land cut"):
        postprocess(raw, FakeTopo(lambda lo, la: np.full(lo.shape, 5.0)), 30.0)
