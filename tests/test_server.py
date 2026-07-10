from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fmesh2.server.app import create_app
from fmesh2.server.state import AppState

FIXTURE_COAST = Path(__file__).parent / "fixtures" / "mini_coast.geojson"


@pytest.fixture()
def client(tmp_path):
    state = AppState(cache_dir=tmp_path / "cache", coastline_source=FIXTURE_COAST)
    return TestClient(create_app(state))


def test_open_and_meta(client, core2_dir):
    r = client.post("/api/meshes/open", json={"path": str(core2_dir)})
    assert r.status_code == 200
    mesh_id = r.json()["id"]
    meta = r.json()["meta"]
    assert meta["nodes"] == 126858 and meta["verts"] == 127235

    assert client.get("/api/meshes").json()[0]["id"] == mesh_id
    assert client.get(f"/api/mesh/{mesh_id}/meta").json() == meta

    # reopening the same path returns the same entry
    r2 = client.post("/api/meshes/open", json={"path": str(core2_dir)})
    assert r2.json()["id"] == mesh_id


def test_buffer_bytes(client, core2_dir):
    mesh_id = client.post("/api/meshes/open", json={"path": str(core2_dir)}).json()["id"]
    meta = client.get(f"/api/mesh/{mesh_id}/meta").json()
    for name, itemsize, width in [("pos", 4, 3), ("tri", 4, 3), ("edge", 4, 2), ("res", 4, 1)]:
        r = client.get(f"/api/mesh/{mesh_id}/buffer/{name}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        count = {"pos": meta["verts"], "res": meta["verts"],
                 "tri": meta["triangles"], "edge": meta["edges"]}[name]
        assert len(r.content) == count * itemsize * width

    pos = np.frombuffer(
        client.get(f"/api/mesh/{mesh_id}/buffer/pos").content, dtype=np.float32
    ).reshape(-1, 3)
    np.testing.assert_allclose(np.linalg.norm(pos, axis=1), 1.0, atol=1e-5)


def test_errors(client, tmp_path):
    assert client.get("/api/mesh/nope/meta").status_code == 404
    r = client.post("/api/meshes/open", json={"path": str(tmp_path)})
    assert r.status_code == 400
    assert "nod2d" in r.json()["detail"]


def test_unknown_buffer(client, core2_dir):
    mesh_id = client.post("/api/meshes/open", json={"path": str(core2_dir)}).json()["id"]
    assert client.get(f"/api/mesh/{mesh_id}/buffer/bogus").status_code == 404


def test_coastline_endpoint(client):
    r = client.get("/api/coastline/pos")
    assert r.status_code == 200
    assert len(r.content) == 8 * 3 * 4  # mini fixture: 4 segments -> 8 verts
    assert client.get("/api/coastline/bogus").status_code == 404


def test_node_info(client, core2_dir):
    from fmesh2.core import read_fesom
    from fmesh2.core.metrics import local_resolution_km

    mesh_id = client.post("/api/meshes/open", json={"path": str(core2_dir)}).json()["id"]
    info = client.get(f"/api/mesh/{mesh_id}/node/0").json()

    mesh = read_fesom(core2_dir)
    assert info["lon"] == pytest.approx(mesh.lon[0], abs=1e-4)
    assert info["lat"] == pytest.approx(mesh.lat[0], abs=1e-4)
    assert info["res_km"] == pytest.approx(local_resolution_km(mesh)[0], rel=1e-4)
    assert info["depth_m"] == pytest.approx(abs(mesh.depth[0]), abs=0.01)
    assert 3 <= len(info["neighbors"]) <= 10
    assert all(isinstance(n, int) for n in info["neighbors"])

    assert client.get(f"/api/mesh/{mesh_id}/node/999999999").status_code == 404


def test_placeholder_page(client):
    r = client.get("/")
    assert r.status_code == 200


def test_cli_convert(core2_dir, tmp_path, capsys):
    from fmesh2.cli import main

    assert main(["convert", str(core2_dir), str(tmp_path / "out")]) == 0
    out = tmp_path / "out"
    assert (out / "meta.json").exists()
    assert (out / "pos.bin").stat().st_size == 127235 * 12
    assert "MB of buffers" in capsys.readouterr().out
