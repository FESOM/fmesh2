import numpy as np

from fmesh2.viewer import build_buffers, load_or_build


def test_tiny_dateline_duplication(tiny_mesh):
    b = build_buffers(tiny_mesh)
    # node 4 (-175E) is the only west corner of the crossing triangle (3,4,5)
    assert b.meta["nodes"] == 6
    assert b.meta["verts"] == 7
    # duplicate keeps the source scalars
    np.testing.assert_array_equal(b.res[6], b.res[4])
    np.testing.assert_array_equal(b.dep[6], b.dep[4])
    # no zero-resolution vertices (orphan-scalar quirk fixed)
    assert (b.res > 0).all()
    # in plane coordinates no triangle spans more than 180 deg
    lon_plane = np.degrees(b.pos2[:, 0])
    tl = lon_plane[b.tri]
    assert (tl.max(axis=1) - tl.min(axis=1)).max() < 180.0
    # duplicated vertex coincides with its source on the sphere
    np.testing.assert_allclose(b.pos[6], b.pos[4], atol=1e-7)
    # dupsrc maps the duplicate back to its source node
    np.testing.assert_array_equal(b.dupsrc, [4])


def test_core2_counts_match_spike(core2_dir):
    from fmesh2.core import read_fesom

    b = build_buffers(read_fesom(core2_dir))
    # reference values measured by the validated render spike
    assert b.meta["verts"] == 127235          # 126858 + 377 duplicates
    assert b.meta["edges"] == 372018
    assert b.meta["triangles"] == 244659
    assert b.tri.dtype == np.uint32 and b.pos.dtype == np.float32
    assert b.meta["res_range"][0] > 10
    assert b.dupsrc.shape == (377,) and b.dupsrc.dtype == np.uint32
    assert (b.dupsrc < b.meta["nodes"]).all()


def test_cache_roundtrip(tiny_mesh, tmp_path):
    from fmesh2.core import write_fesom

    mesh_dir = tmp_path / "mesh"
    write_fesom(tiny_mesh, mesh_dir)

    b1 = load_or_build(mesh_dir, cache_dir=tmp_path / "cache")
    b2 = load_or_build(mesh_dir, cache_dir=tmp_path / "cache")  # cache hit
    for name in ("pos", "pos2", "tri", "edge", "res", "dep", "dupsrc"):
        np.testing.assert_array_equal(b1.buffer(name), b2.buffer(name))
    assert b1.meta == b2.meta
    assert (tmp_path / "cache").exists()
