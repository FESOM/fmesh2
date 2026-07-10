import numpy as np
import pytest

from fmesh2.core import Mesh, read_fesom, write_fesom


def test_core2_roundtrip(core2_dir, tmp_path):
    m1 = read_fesom(core2_dir)
    assert m1.n_nodes == 126858
    assert m1.n_tri == 244659
    assert m1.levels is not None and len(m1.levels) == 48
    assert m1.depth is not None

    write_fesom(m1, tmp_path)
    m2 = read_fesom(tmp_path)

    np.testing.assert_array_equal(m1.lon, m2.lon)
    np.testing.assert_array_equal(m1.lat, m2.lat)
    np.testing.assert_array_equal(m1.tri, m2.tri)
    np.testing.assert_array_equal(m1.depth, m2.depth)
    np.testing.assert_array_equal(m1.coast_flag, m2.coast_flag)
    np.testing.assert_array_equal(m1.levels, m2.levels)

    # headers carry the counts
    assert (tmp_path / "nod2d.out").read_text().splitlines()[0] == "126858"
    assert (tmp_path / "elem2d.out").read_text().splitlines()[0] == "244659"


def test_tiny_roundtrip(tiny_mesh, tmp_path):
    write_fesom(tiny_mesh, tmp_path)
    m2 = read_fesom(tmp_path)
    np.testing.assert_array_equal(tiny_mesh.lon, m2.lon)
    np.testing.assert_array_equal(tiny_mesh.tri, m2.tri)
    np.testing.assert_array_equal(tiny_mesh.depth, m2.depth)


def test_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="nod2d.out"):
        read_fesom(tmp_path)


def test_truncated_elem2d(tiny_mesh, tmp_path):
    write_fesom(tiny_mesh, tmp_path)
    lines = (tmp_path / "elem2d.out").read_text().splitlines()
    (tmp_path / "elem2d.out").write_text("\n".join([lines[0], lines[1]]) + "\n")
    with pytest.raises(ValueError, match="promises 2 rows"):
        read_fesom(tmp_path)


def test_elem2d_out_of_range(tiny_mesh, tmp_path):
    write_fesom(tiny_mesh, tmp_path)
    (tmp_path / "elem2d.out").write_text("1\n1 2 99\n")
    with pytest.raises(ValueError, match="out of range"):
        read_fesom(tmp_path)


def test_aux3d_levels_only(tiny_mesh, tmp_path):
    write_fesom(tiny_mesh, tmp_path)
    (tmp_path / "aux3d.out").write_text("3\n0.0\n-5.0\n-10.0\n")
    m = read_fesom(tmp_path)
    assert m.depth is None
    assert len(m.levels) == 3


def test_mesh_validation():
    with pytest.raises(ValueError, match="out of range"):
        Mesh(lon=np.zeros(2), lat=np.zeros(2), tri=np.array([[0, 1, 2]]))
    with pytest.raises(ValueError, match="equal length"):
        Mesh(lon=np.zeros(2), lat=np.zeros(3), tri=np.zeros((0, 3), dtype=int))
