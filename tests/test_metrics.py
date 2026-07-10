import numpy as np
import pytest

from fmesh2.core import Mesh, read_fesom
from fmesh2.core.metrics import (
    local_resolution_km,
    triangle_min_angle_deg,
    triangle_signed_orientation,
)
from fmesh2.core.topology import chord_to_arc_km, sphere_xyz


def _dist_km(lon1, lat1, lon2, lat2):
    a, b = sphere_xyz(np.array([lon1, lon2]), np.array([lat1, lat2]))
    return chord_to_arc_km(np.linalg.norm(a - b))


def test_local_resolution_tiny(tiny_mesh):
    res = local_resolution_km(tiny_mesh)
    d01 = _dist_km(0, 0, 10, 0)
    d02 = _dist_km(0, 0, 5, 8)
    np.testing.assert_allclose(res[0], (d01 + d02) / 2)
    assert (res > 0).all()


def test_min_angle_equilateral_vs_degenerate():
    equil = Mesh(
        lon=np.array([0.0, 1.0, 0.5]),
        lat=np.array([0.0, 0.0, np.degrees(np.radians(1.0) * np.sqrt(3) / 2)]),
        tri=np.array([[0, 1, 2]]),
    )
    assert triangle_min_angle_deg(equil)[0] == pytest.approx(60.0, abs=0.2)

    sliver = Mesh(
        lon=np.array([0.0, 1.0, 0.5]),
        lat=np.array([0.0, 0.0, 0.001]),
        tri=np.array([[0, 1, 2]]),
    )
    assert triangle_min_angle_deg(sliver)[0] < 0.5


def test_orientation_sign_flips():
    ccw = Mesh(lon=np.array([0.0, 1.0, 0.5]), lat=np.array([0.0, 0.0, 1.0]),
               tri=np.array([[0, 1, 2]]))
    cw = Mesh(lon=ccw.lon, lat=ccw.lat, tri=np.array([[0, 2, 1]]))
    assert triangle_signed_orientation(ccw)[0] > 0
    assert triangle_signed_orientation(cw)[0] < 0


def test_core2_resolution_sane(core2_dir):
    mesh = read_fesom(core2_dir)
    res = local_resolution_km(mesh)
    lo, hi = np.percentile(res, [2, 98])
    # spike-measured reference: [20.28, 120.15] km
    assert lo == pytest.approx(20.28, abs=1.0)
    assert hi == pytest.approx(120.15, abs=1.5)
    coast_frac = (
        mesh.coast_flag.astype(bool).sum() / mesh.n_nodes
    )
    assert 0 < coast_frac < 0.2
