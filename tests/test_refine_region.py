"""Refinement regions from a file: region.geojson written by every
refinement, read back by the viewer's region-file field, single ring only,
and the seam headroom (1.0 with a transition width, 1.2 without).

Synthetic mesh (a jittered ~1 degree Delaunay grid): no reference data.
"""

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial import Delaunay

from fmesh2.core.io_fesom import write_fesom
from fmesh2.core.mesh import Mesh
from fmesh2.edit.cavity import extract_cavity
from fmesh2.edit.local_remesh import (
    RING_HEADROOM,
    WIDE_HEADROOM,
    seam_headroom,
    seam_hfun_at,
)
from fmesh2.io.refine_region import RegionFileError, read_refine_region, write_refine_region
from fmesh2.server.app import create_app
from fmesh2.server.state import AppState

FIXTURE_COAST = Path(__file__).parent / "fixtures" / "mini_coast.geojson"
SQUARE = [[15.0, 15.0], [25.0, 15.0], [25.0, 25.0], [15.0, 25.0]]


def grid_mesh() -> Mesh:
    rng = np.random.default_rng(0)
    g = np.arange(0.0, 40.01, 1.0)
    lon, lat = (a.ravel() for a in np.meshgrid(g, g))
    lon = lon + rng.uniform(-0.05, 0.05, lon.size)
    lat = lat + rng.uniform(-0.05, 0.05, lat.size)
    tri = Delaunay(np.column_stack([lon, lat])).simplices.astype(np.int32)
    return Mesh(lon=lon, lat=lat, tri=tri, depth=np.full(lon.size, -3000.0),
                coast_flag=np.zeros(lon.size, np.int8), levels=np.array([0.0, -10.0, -6000.0]))


def feature(ring, properties=None):
    return {"type": "Feature", "properties": properties or {},
            "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]}}


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(AppState(cache_dir=tmp_path / "cache", coastline_source=FIXTURE_COAST)))


def test_region_file_roundtrip(tmp_path):
    settings = {"target_km": 5.0, "transition_km": 150.0, "topo": "/data/RTopo.nc",
                "minimum_depth": 30.0, "source_mesh": "/meshes/dars2"}
    path = write_refine_region(tmp_path / "region.geojson", SQUARE, settings)
    ring, got = read_refine_region(path)
    assert np.array_equal(ring[0], ring[-1])                 # closed
    assert np.array_equal(ring[:-1], np.asarray(SQUARE))     # same vertices
    assert got == settings
    gj = json.loads(path.read_text())
    assert len(gj["features"]) == 1 and gj["features"][0]["geometry"]["type"] == "Polygon"


def test_plain_region_file_has_no_settings(tmp_path):
    path = tmp_path / "drawn.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature(SQUARE)]}))
    ring, settings = read_refine_region(path)
    assert len(ring) == 5 and settings == {}


def test_two_ring_file_is_rejected(tmp_path):
    outer = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
    path = tmp_path / "two.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        feature(SQUARE, {"role": "inner"}), feature(outer, {"role": "outer"})]}))
    with pytest.raises(RegionFileError, match="two rings"):
        read_refine_region(path)


def test_region_endpoint(client, tmp_path):
    path = write_refine_region(tmp_path / "r.geojson", SQUARE, {"target_km": 25.0})
    r = client.get("/api/region", params={"path": str(path)})
    assert r.status_code == 200
    assert r.json()["ring"][:4] == SQUARE and r.json()["settings"]["target_km"] == 25.0
    assert client.get("/api/region", params={"path": str(tmp_path / "nope.geojson")}).status_code == 400
    two = tmp_path / "two.geojson"
    two.write_text(json.dumps({"type": "FeatureCollection", "features": [
        feature(SQUARE), feature([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])]}))
    r = client.get("/api/region", params={"path": str(two)})
    assert r.status_code == 400 and "two rings" in r.json()["detail"]


def test_seam_headroom():
    assert seam_headroom(None) == RING_HEADROOM == 1.2
    assert seam_headroom(150.0) == WIDE_HEADROOM == 1.0
    cav = extract_cavity(grid_mesh(), SQUARE)
    seam = np.flatnonzero(~cav.edge_is_coast)
    k = seam[0]
    a, b = cav.ring_edges[k]
    mid = 0.5 * (cav.ring_xy[a] + cav.ring_xy[b])        # on the seam: d = 0
    length = cav.edge_len_km[k]
    h_auto, d = seam_hfun_at(mid[None], cav, 10.0)
    h_wide, _ = seam_hfun_at(mid[None], cav, 10.0, transition_km=300.0)
    assert d[0] < 1e-9
    assert np.isclose(h_auto[0], 1.2 * length) and np.isclose(h_wide[0], length)


def _wait(client, job_id, timeout=300.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error", "cancelled"):
            return snap
        time.sleep(0.25)
    raise TimeoutError(job_id)


def test_refine_writes_region_file_and_it_repeats(client, tmp_path):
    write_fesom(grid_mesh(), tmp_path / "mesh")
    mesh_id = client.post("/api/meshes/open", json={"path": str(tmp_path / "mesh")}).json()["id"]
    body = {"polygon": SQUARE, "target_km": 40.0, "transition_km": 200.0,
            "out_dir": str(tmp_path / "refined")}
    snap = _wait(client, client.post(f"/api/mesh/{mesh_id}/refine", json=body).json()["job_id"])
    assert snap["status"] == "done", snap
    region_file = Path(snap["result"]["region_file"])
    assert region_file == tmp_path / "refined" / "region.geojson"

    loaded = client.get("/api/region", params={"path": str(region_file)}).json()
    assert loaded["ring"][:4] == SQUARE
    s = loaded["settings"]
    assert (s["target_km"], s["transition_km"], s["topo"]) == (40.0, 200.0, None)
    assert s["source_mesh"] == str((tmp_path / "mesh").resolve())

    # the saved region refines the same area again, with the same settings
    again = {"polygon": loaded["ring"][:-1], "target_km": s["target_km"],
             "transition_km": s["transition_km"], "out_dir": str(tmp_path / "again")}
    snap2 = _wait(client, client.post(f"/api/mesh/{mesh_id}/refine", json=again).json()["job_id"])
    assert snap2["status"] == "done", snap2
    assert snap2["result"]["n_nodes"] == snap["result"]["n_nodes"]
    assert json.loads((tmp_path / "again" / "region.geojson").read_text())["features"][0]["geometry"] == \
        json.loads(region_file.read_text())["features"][0]["geometry"]
