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
    for name, itemsize, width in [("pos", 4, 3), ("tri", 4, 3), ("edge", 4, 2),
                                  ("res", 4, 1), ("dupsrc", 4, 1)]:
        r = client.get(f"/api/mesh/{mesh_id}/buffer/{name}")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"
        count = {"pos": meta["verts"], "res": meta["verts"],
                 "tri": meta["triangles"], "edge": meta["edges"],
                 "dupsrc": meta["verts"] - meta["nodes"]}[name]
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


# ----------------------------------------------------- project CRUD (task 18)


FIXTURES = Path(__file__).parent / "fixtures"


def open_project(client, tmp_path):
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir(exist_ok=True)
    r = client.post("/api/project/open", json={"path": str(proj_dir)})
    assert r.status_code == 200
    return r.json()


def test_project_open_creates_and_reloads(client, tmp_path):
    pj = open_project(client, tmp_path)
    assert pj["name"] == "proj" and pj["layers"] == []
    assert (tmp_path / "proj" / "project.yaml").exists()
    assert (tmp_path / "proj" / "regions").is_dir()
    # reopen: loads the saved file
    pj2 = client.post("/api/project/open", json={"path": str(tmp_path / "proj")}).json()
    assert pj2["name"] == "proj"
    assert client.get("/api/project").json()["name"] == "proj"


def test_project_put_layer_crud(client, tmp_path):
    pj = open_project(client, tmp_path)
    pj["layers"] = [
        {"type": "base", "resolution": 300},
        {"type": "mercator", "resolution": 300, "mode": "replace"},
    ]
    pj["grid"] = {"n_lat": 36, "n_lon": 72}
    r = client.put("/api/project", json=pj)
    assert r.status_code == 200
    assert [l["type"] for l in r.json()["layers"]] == ["base", "mercator"]

    # reorder + toggle survives a save/load roundtrip on disk
    pj = r.json()
    pj["layers"] = [pj["layers"][1], {**pj["layers"][0], "enabled": False}]
    r = client.put("/api/project", json=pj)
    got = client.get("/api/project").json()
    assert [l["type"] for l in got["layers"]] == ["mercator", "base"]
    assert got["layers"][1]["enabled"] is False

    bad = dict(pj)
    bad["layers"] = [{"type": "wavelet", "resolution": 10}]
    assert client.put("/api/project", json=bad).status_code == 422


def test_project_region_roundtrip(client, tmp_path):
    open_project(client, tmp_path)
    ring = [[-40, 10], [-20, 10], [-20, 30], [-40, 30], [-40, 10]]
    gj = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [ring]}}
        ],
    }
    r = client.post("/api/project/regions/patch", json=gj)
    assert r.status_code == 200 and r.json()["path"] == "regions/patch.geojson"
    assert client.get("/api/project/regions/patch").json() == gj
    assert client.delete("/api/project/regions/patch").status_code == 200
    assert client.get("/api/project/regions/patch").status_code == 404

    assert client.post("/api/project/regions/bad name!", json=gj).status_code in (400, 404, 422)
    r = client.post("/api/project/regions/broken", json={"type": "FeatureCollection", "features": []})
    assert r.status_code == 422
    assert not (tmp_path / "proj" / "regions" / "broken.geojson").exists()


def test_project_hfun_and_estimate(client, tmp_path):
    pj = open_project(client, tmp_path)
    pj["layers"] = [{"type": "base", "resolution": 300}]
    pj["grid"] = {"n_lat": 72, "n_lon": 144}  # 277 km cells: fine enough for 300 km
    client.put("/api/project", json=pj)

    r = client.get("/api/project/hfun")
    assert r.status_code == 200
    n_lat, n_lon = int(r.headers["x-hfun-nlat"]), int(r.headers["x-hfun-nlon"])
    assert (n_lat, n_lon) == (73, 145)
    values = np.frombuffer(r.content, dtype=np.float32).reshape(n_lat, n_lon)
    assert np.all(values == 300.0)
    assert float(r.headers["x-hfun-min"]) == 300.0

    est = client.get("/api/project/estimate").json()
    assert est["elements_including_land"] > 10000
    assert est["ocean_nodes"] == int(round(est["elements_including_land"] / 2 * 0.71))
    assert est["warnings"] == []

    # a fine layer on the coarse grid produces a structured warning
    pj["layers"].append({"type": "base", "resolution": 25})
    client.put("/api/project", json=pj)
    est = client.get("/api/project/estimate").json()
    assert any("too coarse" in w for w in est["warnings"])


def test_project_endpoints_without_project(client):
    assert client.get("/api/project").status_code == 404
    assert client.get("/api/project/hfun").status_code == 404
    assert client.get("/api/project/estimate").status_code == 404


# ------------------------------------------------------- local refinement (M3)


def _wait_job(client, job_id: str, timeout: float = 300.0) -> dict:
    import time

    t0 = time.time()
    snap: dict = {}
    while time.time() - t0 < timeout:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error", "cancelled"):
            return snap
        time.sleep(0.25)
    raise TimeoutError(f"job never finished: {snap}")


def test_refine_end_to_end(client, core2_dir, tmp_path):
    mesh_id = client.post("/api/meshes/open", json={"path": str(core2_dir)}).json()["id"]
    r = client.post(
        f"/api/mesh/{mesh_id}/refine",
        json={
            "polygon": [[-42, 28], [-28, 28], [-28, 36], [-42, 36]],
            "target_km": 60.0,
            "out_dir": str(tmp_path / "refined"),
        },
    )
    assert r.status_code == 200, r.text
    snap = _wait_job(client, r.json()["job_id"])
    assert snap["status"] == "done", snap
    result = snap["result"]
    assert result["report"]["watertight_seam"] is True
    assert result["n_nodes"] > 0

    # the refined mesh auto-registered in the viewer
    new_id = result["mesh_id"]
    assert new_id != mesh_id
    assert any(m["id"] == new_id for m in client.get("/api/meshes").json())
    assert client.get(f"/api/mesh/{new_id}/meta").status_code == 200

    # mapping table downloadable from the refined mesh, absent on the source
    mapping = client.get(f"/api/mesh/{new_id}/mapping")
    assert mapping.status_code == 200
    lines = mapping.text.splitlines()
    assert lines[0] == "old_id,new_id"
    assert len(lines) == 126858 + 1  # one row per source node
    assert client.get(f"/api/mesh/{mesh_id}/mapping").status_code == 404


def test_refine_validation_and_failure(client, core2_dir, tmp_path):
    assert (
        client.post("/api/mesh/nope/refine", json={"polygon": [[0, 0]] * 3, "target_km": 50}).status_code
        == 404
    )
    mesh_id = client.post("/api/meshes/open", json={"path": str(core2_dir)}).json()["id"]
    # schema validation: short polygon / non-positive target
    bad = client.post(f"/api/mesh/{mesh_id}/refine", json={"polygon": [[0, 0], [1, 1]], "target_km": 50})
    assert bad.status_code == 422
    bad = client.post(f"/api/mesh/{mesh_id}/refine", json={"polygon": [[0, 0]] * 3, "target_km": 0})
    assert bad.status_code == 422
    bad = client.post(
        f"/api/mesh/{mesh_id}/refine",
        json={"polygon": [[0, 0]] * 3, "target_km": 50, "topo": str(tmp_path / "nope.nc")},
    )
    assert bad.status_code == 400 and "topo" in bad.json()["detail"]
    # a land polygon fails in the worker and surfaces as a job error
    r = client.post(
        f"/api/mesh/{mesh_id}/refine",
        json={
            "polygon": [[2, 20], [10, 20], [10, 26], [2, 26]],
            "target_km": 50.0,
            "out_dir": str(tmp_path / "land"),
        },
    )
    snap = _wait_job(client, r.json()["job_id"])
    assert snap["status"] == "error"
    assert "no elements" in snap["error"]


# ------------------------------------------------------------ node editing (M4)


def _open_edit(client, core2_dir) -> str:
    mesh_id = client.post("/api/meshes/open", json={"path": str(core2_dir)}).json()["id"]
    r = client.post(f"/api/mesh/{mesh_id}/edit/open", json={})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_nodes"] == 126858
    assert body["journal"] == {"can_undo": False, "can_redo": False, "n_ops": 0}
    return mesh_id


def test_edit_requires_open_session(client, core2_dir):
    mesh_id = client.post("/api/meshes/open", json={"path": str(core2_dir)}).json()["id"]
    r = client.post(f"/api/mesh/{mesh_id}/edit/select", json={"center": [0, 0], "radius_km": 100})
    assert r.status_code == 404 and "edit/open" in r.json()["detail"]
    assert client.post("/api/mesh/nope/edit/open", json={}).status_code == 404


def test_edit_select_lasso_and_brush(client, core2_dir):
    from fmesh2.core.io_fesom import read_fesom
    from fmesh2.core.topology import SphereKDTree

    mesh_id = _open_edit(client, core2_dir)
    mesh = read_fesom(core2_dir)

    # lasso: mid-Atlantic square; every returned node lies inside its bbox
    ring = [[-40, 25], [-30, 25], [-30, 33], [-40, 33]]
    r = client.post(f"/api/mesh/{mesh_id}/edit/select", json={"lasso": ring})
    assert r.status_code == 200
    ids = r.json()["ids"]
    assert len(ids) > 10
    assert all(-40 <= mesh.lon[i] <= -30 and 25 <= mesh.lat[i] <= 33 for i in ids)

    # brush: matches a direct KDTree radius query
    r = client.post(
        f"/api/mesh/{mesh_id}/edit/select",
        json={"center": [-35, 29], "radius_km": 300},
    )
    expect = sorted(SphereKDTree(mesh.lon, mesh.lat).query_radius(-35.0, 29.0, 300.0)[0])
    assert r.json()["ids"] == [int(i) for i in expect]

    # validation: nothing given / short lasso
    assert client.post(f"/api/mesh/{mesh_id}/edit/select", json={}).status_code == 422
    assert (
        client.post(
            f"/api/mesh/{mesh_id}/edit/select", json={"lasso": [[0, 0], [1, 1]]}
        ).status_code
        == 422
    )


def test_edit_move_undo_redo_save_api(client, core2_dir, tmp_path):
    from fmesh2.core.io_fesom import read_fesom, write_fesom
    from fmesh2.edit.journal import mesh_hash

    mesh_id = _open_edit(client, core2_dir)
    pristine = read_fesom(core2_dir)
    ids = client.post(
        f"/api/mesh/{mesh_id}/edit/select",
        json={"center": [-35, 29], "radius_km": 200},
    ).json()["ids"][:2]

    target = [[float(pristine.lon[i]) + 0.01, float(pristine.lat[i]) - 0.01] for i in ids]
    r = client.post(f"/api/mesh/{mesh_id}/edit/move", json={"ids": ids, "lonlat": target})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["journal"]["can_undo"] and body["journal"]["n_ops"] == 1
    assert body["report"]["n_nodes"] == 2
    upd = body["update"]
    assert upd["verts"][: len(ids)] == ids  # duplicates (if any) appended after
    assert upd["pos2"][0][0] == pytest.approx(np.deg2rad(target[0][0]))
    assert len(upd["dep"]) == len(upd["verts"])

    # undo restores the original positions in the returned buffer update
    r = client.post(f"/api/mesh/{mesh_id}/edit/undo")
    upd = r.json()["update"]
    assert r.json()["journal"] == {"can_undo": False, "can_redo": True, "n_ops": 0}
    assert upd["pos2"][0][0] == pytest.approx(np.deg2rad(pristine.lon[ids[0]]), abs=1e-12)

    # nothing left to undo
    r = client.post(f"/api/mesh/{mesh_id}/edit/undo")
    assert r.json()["update"] is None

    # redo re-applies, then save-as writes triplet + journal and registers
    client.post(f"/api/mesh/{mesh_id}/edit/redo")
    out = tmp_path / "edited"
    r = client.post(f"/api/mesh/{mesh_id}/edit/save", json={"out_dir": str(out)})
    assert r.status_code == 200, r.text
    saved = read_fesom(out)
    assert saved.lon[ids[0]] == target[0][0]
    assert (out / "edits.jsonl").exists()
    assert mesh_hash(saved) == r.json()["hash"]
    assert any(m["id"] == r.json()["mesh_id"] for m in client.get("/api/meshes").json())

    # journal replay on the pristine mesh reproduces the saved hash
    from fmesh2.edit.journal import apply_journal

    replayed = apply_journal(pristine, out / "edits.jsonl")
    assert mesh_hash(replayed) == r.json()["hash"]

    # saving over the source mesh is refused
    r = client.post(f"/api/mesh/{mesh_id}/edit/save", json={"out_dir": str(core2_dir)})
    assert r.status_code == 400 and "refusing" in r.json()["detail"]


def test_edit_move_inversion_and_smooth_api(client, core2_dir):
    mesh_id = _open_edit(client, core2_dir)
    ids = client.post(
        f"/api/mesh/{mesh_id}/edit/select",
        json={"center": [-35, 29], "radius_km": 300},
    ).json()["ids"]

    # dragging one node across its neighbours is rejected with element ids
    r = client.post(
        f"/api/mesh/{mesh_id}/edit/move",
        json={"ids": ids[:1], "lonlat": [[-25.0, 29.0]]},
    )
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "invert" in detail["message"] and len(detail["element_ids"]) > 0

    # smoothing the patch through the API
    r = client.post(
        f"/api/mesh/{mesh_id}/edit/smooth",
        json={"ids": ids, "iterations": 2, "strength": 0.5},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report"]["n_nodes"] > 0
    assert len(body["update"]["verts"]) >= body["report"]["n_nodes"]
    assert body["journal"]["n_ops"] == 1


def test_edit_delete_and_add_api(client, core2_dir):
    from fmesh2.core.io_fesom import read_fesom
    from fmesh2.core.topology import boundary_nodes

    mesh_id = _open_edit(client, core2_dir)
    mesh = read_fesom(core2_dir)
    coast = boundary_nodes(mesh.tri, mesh.n_nodes)

    # one interior open-ocean node
    ids = client.post(
        f"/api/mesh/{mesh_id}/edit/select",
        json={"center": [-35, 29], "radius_km": 150},
    ).json()["ids"]
    interior = [i for i in ids if not coast[i]]

    # deleting a coastal node without allow_coast is refused with a message
    coastal = int(np.flatnonzero(coast)[0])
    r = client.post(
        f"/api/mesh/{mesh_id}/edit/delete", json={"ids": [coastal]}
    )
    assert r.status_code == 422 and "coastline" in r.json()["detail"]["message"]

    # interior delete: topology response + rebuilt buffers
    r = client.post(
        f"/api/mesh/{mesh_id}/edit/delete", json={"ids": interior[:1]}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["topology_changed"] is True
    assert body["meta"]["nodes"] == 126857
    assert body["report"]["n_deleted"] == 1
    assert client.get(f"/api/mesh/{mesh_id}/meta").json()["nodes"] == 126857
    pos = client.get(f"/api/mesh/{mesh_id}/buffer/pos").content
    assert len(pos) == body["meta"]["verts"] * 12

    # undo restores the topology (and says so)
    r = client.post(f"/api/mesh/{mesh_id}/edit/undo")
    assert r.json()["topology_changed"] is True
    assert r.json()["meta"]["nodes"] == 126858

    # add a node in open ocean: element split
    r = client.post(
        f"/api/mesh/{mesh_id}/edit/add", json={"lon": -35.2, "lat": 29.1}
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["report"]["mode"] == "split"
    assert body["meta"]["nodes"] == 126859

    # far from any mesh point -> rejected  (mid-Sahara)
    r = client.post(f"/api/mesh/{mesh_id}/edit/add", json={"lon": 10, "lat": 23})
    assert r.status_code == 422
