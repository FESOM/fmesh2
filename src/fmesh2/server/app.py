from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fmesh2.server.jobs import TERMINAL, JobManager
from fmesh2.server.state import AppState

STATIC_DIR = Path(__file__).parent / "static"


class OpenMeshRequest(BaseModel):
    path: str


class BuildRequest(BaseModel):
    path: str  # project.yaml or project directory


def create_app(state: AppState | None = None) -> FastAPI:
    state = state or AppState()
    app = FastAPI(title="fmesh2")
    app.state.fmesh2 = state

    @app.get("/api/meshes")
    def list_meshes():
        return [{"id": e.id, "meta": e.buffers.meta} for e in state.meshes.values()]

    @app.post("/api/meshes/open")
    def open_mesh(req: OpenMeshRequest):
        try:
            entry = state.open_mesh(req.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"id": entry.id, "meta": entry.buffers.meta}

    def _entry(mesh_id: str):
        entry = state.meshes.get(mesh_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown mesh id {mesh_id!r}")
        return entry

    @app.get("/api/mesh/{mesh_id}/meta")
    def mesh_meta(mesh_id: str):
        return _entry(mesh_id).buffers.meta

    @app.get("/api/mesh/{mesh_id}/buffer/{name}")
    def mesh_buffer(mesh_id: str, name: str):
        entry = _entry(mesh_id)
        try:
            arr = entry.buffers.buffer(name)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown buffer {name!r}") from None
        return Response(content=arr.tobytes(), media_type="application/octet-stream")

    @app.get("/api/mesh/{mesh_id}/node/{idx}")
    def node_info(mesh_id: str, idx: int):
        import numpy as np

        b = _entry(mesh_id).buffers
        nv = b.meta["verts"]
        if not 0 <= idx < nv:
            raise HTTPException(status_code=404, detail=f"node {idx} out of range 0..{nv - 1}")
        lon = (float(np.degrees(b.pos2[idx, 0])) + 180.0) % 360.0 - 180.0
        mask = (b.edge[:, 0] == idx) | (b.edge[:, 1] == idx)
        e = b.edge[mask]
        neighbors = np.where(e[:, 0] == idx, e[:, 1], e[:, 0])
        return {
            "index": idx,
            "lon": lon,
            "lat": float(np.degrees(b.pos2[idx, 1])),
            "res_km": float(b.res[idx]),
            "depth_m": float(b.dep[idx]),
            "neighbors": [int(x) for x in neighbors],
        }

    @app.get("/api/coastline/{name}")
    def coastline_buffer(name: str):
        coast = state.coastline()
        if name not in coast:
            raise HTTPException(status_code=404, detail=f"unknown buffer {name!r}")
        return Response(content=coast[name].tobytes(), media_type="application/octet-stream")

    # ------------------------------------------------------- generation jobs

    def _register_built_mesh(out_dir: str) -> dict:
        entry = state.open_mesh(out_dir, reload=True)
        return {"mesh_id": entry.id}

    jobs = JobManager(register=_register_built_mesh)
    app.state.jobs = jobs

    @app.post("/api/project/build")
    def build_project_endpoint(req: BuildRequest):
        from fmesh2.project import Project, ProjectError

        try:
            project = Project.load(req.path)
        except ProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if project.topo is None:
            raise HTTPException(
                status_code=400, detail="project.topo is required to build a mesh"
            )
        job = jobs.start_build(str(Path(req.path).expanduser().resolve()))
        return {"job_id": job.id}

    def _job(job_id: str):
        job = jobs.jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"unknown job id {job_id!r}")
        return job

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        return _job(job_id).snapshot()

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str):
        _job(job_id)
        return jobs.cancel(job_id).snapshot()

    @app.websocket("/api/jobs/{job_id}")
    async def job_events(ws: WebSocket, job_id: str):
        job = jobs.jobs.get(job_id)
        if job is None:
            await ws.close(code=4404)
            return
        await ws.accept()
        cursor = 0
        try:
            while True:
                while cursor < len(job.events):
                    await ws.send_json(job.events[cursor])
                    cursor += 1
                if job.status in TERMINAL:
                    await ws.send_json({"type": "status", **job.snapshot()})
                    break
                await asyncio.sleep(0.05)
        finally:
            try:
                await ws.close()
            except RuntimeError:
                pass  # client already gone

    if (STATIC_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    else:

        @app.get("/", response_class=HTMLResponse)
        def placeholder():
            return (
                "<h1>fmesh2</h1><p>Frontend not built. "
                "Run <code>make frontend-build</code> and restart.</p>"
            )

    return app
