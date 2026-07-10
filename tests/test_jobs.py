import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fmesh2.project import BaseLayer, GridSettings, Project
from fmesh2.server.app import create_app

FIXTURES = Path(__file__).parent / "fixtures"
TOPO = str((FIXTURES / "topo_synthetic.nc").resolve())


@pytest.fixture()
def client():
    with TestClient(create_app()) as c:
        yield c


def make_project(tmp_path, base=500.0, topo=TOPO, n_lat=90) -> Path:
    proj = Project(
        name="job-test",
        topo=topo,
        grid=GridSettings(n_lat=n_lat, n_lon=2 * n_lat),
        layers=[BaseLayer(type="base", resolution=base)],
    )
    return proj.save(tmp_path / "proj")


def wait_for(client, job_id, predicate, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if predicate(snap):
            return snap
        time.sleep(0.1)
    raise AssertionError(f"timeout waiting for job {job_id}; last: {snap}")


@pytest.mark.timeout(300)
def test_build_job_ws_ordered_stages(client, tmp_path):
    path = make_project(tmp_path)
    job_id = client.post("/api/project/build", json={"path": str(path)}).json()["job_id"]

    events = []
    with client.websocket_connect(f"/api/jobs/{job_id}") as ws:
        while True:
            ev = ws.receive_json()
            events.append(ev)
            if ev["type"] in ("status", "done", "error", "cancelled"):
                if ev["type"] == "status" or ev["type"] != "done":
                    break

    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["compose", "jigsaw", "postprocess", "write", "register"]
    done = [e for e in events if e["type"] == "done"]
    assert done and done[0]["n_nodes"] > 500
    assert "mesh_id" in done[0]

    # polling fallback agrees and the mesh is registered in viewer state
    snap = client.get(f"/api/jobs/{job_id}").json()
    assert snap["status"] == "done" and snap["fraction"] == 1.0
    meshes = client.get("/api/meshes").json()
    assert any(m["id"] == done[0]["mesh_id"] for m in meshes)
    # jigsaw log lines made it into the tail at some point
    assert isinstance(snap["log_tail"], list)


@pytest.mark.timeout(300)
def test_cancel_mid_jigsaw_kills_subprocess(client, tmp_path):
    # fine enough that jigsaw runs for a while
    path = make_project(tmp_path, base=50.0, n_lat=360)
    job_id = client.post("/api/project/build", json={"path": str(path)}).json()["job_id"]

    manager = client.app.state.jobs
    job = manager.jobs[job_id]
    t0 = time.time()
    while job.jigsaw_pid is None and time.time() - t0 < 60:
        assert job.status == "running", f"job ended early: {job.snapshot()}"
        time.sleep(0.05)
    pid = job.jigsaw_pid
    assert pid is not None, "jigsaw never started"

    snap = client.post(f"/api/jobs/{job_id}/cancel").json()
    assert snap["status"] == "cancelled"

    t0 = time.time()
    while time.time() - t0 < 10:
        try:
            os.kill(pid, 0)  # raises when the process is gone
            time.sleep(0.1)
        except ProcessLookupError:
            break
    else:
        os.kill(pid, 9)  # clean up before failing
        pytest.fail("jigsaw subprocess still alive 10 s after cancel")

    proc = manager._procs[job_id]
    proc.join(timeout=10)
    assert not proc.is_alive()


@pytest.mark.timeout(300)
def test_failure_surfaces_error(client, tmp_path):
    path = make_project(tmp_path, topo=str(tmp_path / "missing_topo.nc"))
    job_id = client.post("/api/project/build", json={"path": str(path)}).json()["job_id"]
    snap = wait_for(client, job_id, lambda s: s["status"] in ("done", "error"))
    assert snap["status"] == "error"
    assert "missing_topo.nc" in snap["error"] or "No such file" in snap["error"]
    assert any("Traceback" in line for line in snap["log_tail"])


def test_build_request_validation(client, tmp_path):
    r = client.post("/api/project/build", json={"path": str(tmp_path / "nope")})
    assert r.status_code == 400 and "not found" in r.json()["detail"]

    no_topo = Project(name="x", layers=[BaseLayer(type="base", resolution=300)])
    p = no_topo.save(tmp_path / "nt")
    r = client.post("/api/project/build", json={"path": str(p)})
    assert r.status_code == 400 and "topo" in r.json()["detail"]

    assert client.get("/api/jobs/nope").status_code == 404
    assert client.post("/api/jobs/nope/cancel").status_code == 404
