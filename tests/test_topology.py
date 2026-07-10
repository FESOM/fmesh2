import numpy as np

from fmesh2.core.topology import (
    SphereKDTree,
    boundary_edges,
    boundary_nodes,
    node_elem_csr,
    sphere_xyz,
    unique_edges,
)


def test_unique_edges_tiny(tiny_mesh):
    edges = unique_edges(tiny_mesh.tri, tiny_mesh.n_nodes)
    expected = {(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)}
    assert {tuple(e) for e in edges} == expected


def test_boundary_tiny(tiny_mesh):
    # every edge of two lone triangles is a boundary edge
    be = boundary_edges(tiny_mesh.tri, tiny_mesh.n_nodes)
    assert len(be) == 6
    assert boundary_nodes(tiny_mesh.tri, tiny_mesh.n_nodes).all()


def test_shared_edge_not_boundary():
    tri = np.array([[0, 1, 2], [0, 2, 3]])
    be = boundary_edges(tri, 4)
    assert (0, 2) not in {tuple(e) for e in be}
    assert len(be) == 4


def test_node_elem_csr(tiny_mesh):
    indptr, elems = node_elem_csr(tiny_mesh.tri, tiny_mesh.n_nodes)
    for node in range(6):
        incident = elems[indptr[node] : indptr[node + 1]]
        assert list(incident) == ([0] if node < 3 else [1])


def test_sphere_xyz_units():
    xyz = sphere_xyz(np.array([0.0, 90.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(xyz[0], [1, 0, 0], atol=1e-15)
    np.testing.assert_allclose(xyz[1], [0, 1, 0], atol=1e-15)
    np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), 1.0)


def test_kdtree_nearest_km():
    # one degree of longitude at the equator is ~111.19 km
    tree = SphereKDTree(np.array([0.0]), np.array([0.0]))
    d, i = tree.nearest_km(1.0, 0.0)
    np.testing.assert_allclose(d[0], 111.19, atol=0.05)
    assert i[0] == 0


def test_kdtree_query_radius():
    tree = SphereKDTree(np.array([0.0, 1.0, 10.0]), np.array([0.0, 0.0, 0.0]))
    (hits,) = tree.query_radius(0.0, 0.0, 200.0)
    assert set(hits) == {0, 1}
