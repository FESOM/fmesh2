"""Unit tests for the FESOM all-boundary-triangle hygiene cleanup."""
import numpy as np

from fmesh2.edit.local_remesh import _drop_all_boundary_triangles


def _fan(n=6):
    """Hexagon fan: node 0 interior, 1..n on the rim - every element has
    the interior node, so nothing may be removed."""
    return np.array([[0, 1 + i, 1 + (i + 1) % n] for i in range(n)])


def test_clean_fan_untouched():
    tri = _fan()
    out, keep, removed = _drop_all_boundary_triangles(tri, 7)
    assert removed == 0
    assert out.shape == tri.shape
    assert keep.all()


def test_detached_strip_removed_iteratively():
    # fan + detached two-triangle strip (nodes 7..10): all four strip
    # nodes are boundary, removal of one triangle exposes the other
    strip = np.array([[7, 8, 9], [8, 9, 10]])
    tri = np.vstack([_fan(), strip])
    out, keep, removed = _drop_all_boundary_triangles(tri, 11)
    assert removed == 2
    assert len(out) == 6
    assert keep[:7].all() and not keep[7:].any()


def test_dangling_bay_removed():
    # a single-triangle "bay" glued to the fan rim by one edge (1,2):
    # its vertices 1, 2, 7 are all boundary nodes -> must be removed,
    # node 7 orphaned; the fan itself must survive
    bay = np.array([[1, 2, 7]])
    tri = np.vstack([_fan(), bay])
    out, keep, removed = _drop_all_boundary_triangles(tri, 8)
    assert removed == 1
    assert len(out) == 6
    assert keep[:7].all() and not keep[7]
