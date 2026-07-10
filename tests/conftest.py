import os
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_root() -> Path:
    return Path(os.environ.get("FMESH2_TEST_DATA", REPO_ROOT.parent))


@pytest.fixture(scope="session")
def core2_dir() -> Path:
    d = data_root() / "core2"
    if not (d / "nod2d.out").exists():
        pytest.skip(f"core2 reference mesh not found at {d} (set FMESH2_TEST_DATA)")
    return d


@pytest.fixture()
def tiny_mesh():
    """Two disjoint triangle patches; the second one crosses the dateline.

    nodes:  0 (0,0)   1 (10,0)   2 (5,8)      patch A, tri (0,1,2)
            3 (175,0) 4 (-175,0) 5 (178,8)    patch B, tri (3,4,5) crosses 180
    """
    from fmesh2.core.mesh import Mesh

    lon = np.array([0.0, 10.0, 5.0, 175.0, -175.0, 178.0])
    lat = np.array([0.0, 0.0, 8.0, 0.0, 0.0, 8.0])
    tri = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    depth = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    coast = np.ones(6, dtype=np.int8)
    levels = np.array([0.0, -5.0, -10.0])
    return Mesh(lon=lon, lat=lat, tri=tri, depth=depth, coast_flag=coast, levels=levels)
