from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from fmesh2.server.state import AppState

STATIC_DIR = Path(__file__).parent / "static"


class OpenMeshRequest(BaseModel):
    path: str


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

    @app.get("/api/coastline/{name}")
    def coastline_buffer(name: str):
        coast = state.coastline()
        if name not in coast:
            raise HTTPException(status_code=404, detail=f"unknown buffer {name!r}")
        return Response(content=coast[name].tobytes(), media_type="application/octet-stream")

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
