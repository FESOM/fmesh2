import numpy as np
import pytest

from fmesh2.core.mesh import Mesh
from fmesh2.core.projection import haversine_km, stereo_inverse, stereo_xy
from fmesh2.edit.cavity import (
    CavityError,
    clean_selection,
    extract_cavity,
    order_loops,
    select_elements,
)

# polygons validated by the Task 20 spike (experiments/cavity_spike.py)
OPEN_OCEAN = [(-42.0, 28.0), (-28.0, 28.0), (-28.0, 36.0), (-42.0, 36.0)]
COASTAL = [(-21.0, 15.0), (-13.0, 15.0), (-13.0, 23.0), (-21.0, 23.0)]
ISLAND = [(-27.0, 61.5), (-11.0, 61.5), (-11.0, 67.5), (-27.0, 67.5)]
SAHARA = [(2.0, 20.0), (10.0, 20.0), (10.0, 26.0), (2.0, 26.0)]


@pytest.fixture(scope="module")
def core2(core2_dir):
    from fmesh2.core.io_fesom import read_fesom

    return read_fesom(core2_dir)


def test_open_ocean_single_loop(core2):
    cav = extract_cavity(core2, OPEN_OCEAN)
    assert cav.n_loops == 1
    assert cav.n_elems == 241  # spike-pinned regression value
    assert len(cav.ring_nodes) == 49
    deg = np.bincount(cav.ring_edges.ravel(), minlength=len(cav.ring_nodes))
    assert (deg == 2).all()
    assert np.unique(cav.ring_nodes).size == cav.ring_nodes.size
    # mid-ocean: no coast edges on the ring
    assert cav.edge_is_coast.sum() == 0
    # consecutive ring nodes are actual mesh edges of cavity elements
    from fmesh2.core.topology import boundary_edges

    expect = {
        tuple(e)
        for e in boundary_edges(core2.tri[cav.elem_mask], core2.n_nodes).tolist()
    }
    got = {
        tuple(sorted(pair))
        for pair in cav.ring_nodes[cav.ring_edges].tolist()
    }
    assert got == expect
    # plane edge lengths match great-circle lengths (small local distortion)
    ge = cav.ring_nodes[cav.ring_edges]
    d_true = haversine_km(
        core2.lon[ge[:, 0]], core2.lat[ge[:, 0]], core2.lon[ge[:, 1]], core2.lat[ge[:, 1]]
    )
    assert np.abs(cav.edge_len_km / d_true - 1.0).max() < 0.01


def test_projection_roundtrip(core2):
    cav = extract_cavity(core2, OPEN_OCEAN)
    nodes = np.unique(core2.tri[cav.elem_mask])
    lon, lat = core2.lon[nodes], core2.lat[nodes]
    x, y = stereo_xy(lon, lat, cav.lon0, cav.lat0)
    lon2, lat2 = stereo_inverse(x, y, cav.lon0, cav.lat0)
    dlon = (lon2 - lon + 180.0) % 360.0 - 180.0
    assert np.abs(dlon).max() < 1e-9
    assert np.abs(lat2 - lat).max() < 1e-9


def test_coastal_single_loop_with_coast_edges(core2):
    cav = extract_cavity(core2, COASTAL)
    assert cav.n_loops == 1
    assert len(cav.ring_nodes) == 38  # spike-pinned
    assert cav.edge_is_coast.sum() == 18  # ring is ~half coastline here


def test_island_two_loops(core2):
    cav = extract_cavity(core2, ISLAND)
    assert cav.n_loops == 2
    loops = cav.loops
    per_loop_coast = [
        cav.edge_is_coast[cav.loop_starts[k]:cav.loop_starts[k + 1]]
        for k in range(2)
    ]
    kinds = sorted(
        (loop.size, coast.all(), coast.any())
        for loop, coast in zip(loops, per_loop_coast)
    )
    # island (Iceland) loop: all coast edges; outer seam loop: none
    assert kinds[0] == (66, True, True)
    assert kinds[1] == (135, False, False)


def test_empty_selection_raises(core2):
    with pytest.raises(CavityError, match="no elements"):
        extract_cavity(core2, SAHARA)


def test_full_mesh_selection_raises(core2):
    with pytest.raises(CavityError, match="entire mesh"):
        extract_cavity(core2, elem_mask=np.ones(core2.n_tri, dtype=bool))


def test_bad_inputs(core2):
    with pytest.raises(CavityError, match="polygon or an element mask"):
        extract_cavity(core2)
    with pytest.raises(CavityError, match="one entry per element"):
        extract_cavity(core2, elem_mask=np.ones(5, dtype=bool))
    with pytest.raises(CavityError, match="no elements"):
        extract_cavity(core2, elem_mask=np.zeros(core2.n_tri, dtype=bool))


def _quad_grid_mesh(n=4):
    """(n x n)-node structured triangulation on a small lon/lat patch."""
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    lon, lat = i.ravel().astype(float), j.ravel().astype(float)
    tris = []
    for jj in range(n - 1):
        for ii in range(n - 1):
            a, b = jj * n + ii, jj * n + ii + 1
            c, d = (jj + 1) * n + ii + 1, (jj + 1) * n + ii
            tris.append([a, b, c])
            tris.append([a, c, d])
    return Mesh(lon=lon, lat=lat, tri=np.asarray(tris, dtype=np.int32))


def test_pinch_cleanup_synthetic():
    """Center quad + corner quad removed -> their shared node pinches the ring."""
    mesh = _quad_grid_mesh(4)  # 9 quads = 18 triangles
    mask = np.ones(mesh.n_tri, dtype=bool)
    quad_of = np.repeat(np.arange(9), 2)
    mask[quad_of == 0] = False  # corner quad (nodes 0,1,5,4)
    mask[quad_of == 4] = False  # center quad (nodes 5,6,10,9)
    # node 5 has four boundary edges now: a pinch
    from fmesh2.core.topology import boundary_edges

    deg = np.bincount(
        boundary_edges(mesh.tri[mask], mesh.n_nodes).ravel(), minlength=mesh.n_nodes
    )
    assert deg[5] == 4
    cleaned = clean_selection(mesh.tri, mask, mesh.n_nodes)
    # cleanup removed the elements pinned at node 5, selection stays connected
    assert cleaned.sum() < mask.sum()
    cav = extract_cavity(mesh, elem_mask=cleaned)
    deg2 = np.bincount(cav.ring_edges.ravel(), minlength=len(cav.ring_nodes))
    assert (deg2 == 2).all()


def test_largest_component_synthetic(tiny_mesh):
    # two disjoint triangles: selection keeps only one
    mask = np.ones(tiny_mesh.n_tri, dtype=bool)
    cleaned = clean_selection(tiny_mesh.tri, mask, tiny_mesh.n_nodes)
    assert cleaned.sum() == 1


def test_order_loops_deterministic():
    edges = np.array([[1, 2], [0, 1], [0, 2], [4, 5], [3, 4], [3, 5]])
    loops = order_loops(edges)
    assert [loop.tolist() for loop in loops] == [[0, 1, 2], [3, 4, 5]]


def test_order_loops_rejects_pinch():
    edges = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]])
    with pytest.raises(CavityError, match="boundary edges"):
        order_loops(edges)


def test_single_node_polygon_selects_star(core2):
    """A polygon smaller than any element falls back to the node's star.

    This is the delete/replace-one-node UX: the enclosed node becomes
    cavity-interior instead of erroring with "selects no elements".
    """
    from fmesh2.core.topology import SphereKDTree

    _, idx = SphereKDTree(core2.lon, core2.lat).nearest_km(-35.0, 30.0)
    n = int(np.atleast_1d(idx)[0])
    lon0, lat0 = float(core2.lon[n]), float(core2.lat[n])
    box = [(lon0 - 0.15, lat0 - 0.15), (lon0 + 0.15, lat0 - 0.15),
           (lon0 + 0.15, lat0 + 0.15), (lon0 - 0.15, lat0 + 0.15)]
    cav = extract_cavity(core2, box)
    star = np.flatnonzero((core2.tri == n).any(axis=1))
    assert set(np.flatnonzero(cav.elem_mask)) == set(star)
    assert n not in cav.ring_nodes  # cavity-interior -> replaced by a remesh
