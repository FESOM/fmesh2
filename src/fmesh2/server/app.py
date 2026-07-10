from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fmesh2.server.jobs import TERMINAL, JobManager
from fmesh2.server.state import AppState

STATIC_DIR = Path(__file__).parent / "static"


class OpenMeshRequest(BaseModel):
    path: str


class BuildRequest(BaseModel):
    path: str  # project.yaml or project directory


class RefineRequest(BaseModel):
    polygon: list[tuple[float, float]] = Field(min_length=3)  # (lon, lat) ring
    target_km: float = Field(gt=0)
    transition_km: float | None = Field(default=None, gt=0)
    topo: str | None = None  # NetCDF path enables the coastal land cut
    minimum_depth: float = 5.0
    out_dir: str | None = None  # default: sibling '<mesh>_refined[-N]'


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

    # ------------------------------------------------------- active project

    def _project():
        if state.project is None:
            raise HTTPException(status_code=404, detail="no project open")
        return state.project

    def _project_json():
        proj = _project()
        return {"path": str(proj.project_dir), **proj.model_dump(mode="json")}

    @app.post("/api/project/open")
    def project_open(req: OpenMeshRequest):
        from fmesh2.project import Project, ProjectError

        path = Path(req.path).expanduser()
        try:
            if path.is_dir() and not (path / "project.yaml").exists():
                # empty directory: start a fresh project there
                proj = Project(name=path.name or "project")
                proj.save(path)
            else:
                proj = Project.load(path)
        except ProjectError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        proj.ensure_layout()
        state.project = proj
        return _project_json()

    @app.get("/api/project")
    def project_get():
        return _project_json()

    @app.put("/api/project")
    def project_put(body: dict):
        from pydantic import ValidationError

        from fmesh2.project import Project

        proj = _project()
        body.pop("path", None)
        try:
            updated = Project.model_validate(body)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        updated._dir = proj.project_dir
        updated.save()
        state.project = updated
        return _project_json()

    @app.post("/api/project/regions/{name}")
    def region_save(name: str, body: dict):
        proj = _project()
        if not name.replace("-", "").replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail=f"invalid region name {name!r}")
        import json as _json

        from fmesh2.hfun.region_layer import load_region_rings

        path = proj.regions_dir / f"{name}.geojson"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(body))
        try:
            load_region_rings(path)  # validate before anyone composes with it
        except (ValueError, KeyError) as exc:
            path.unlink()
            raise HTTPException(status_code=422, detail=f"invalid region: {exc}") from exc
        return {"path": f"regions/{name}.geojson"}

    @app.get("/api/project/regions/{name}")
    def region_get(name: str):
        import json as _json

        proj = _project()
        path = proj.regions_dir / f"{name}.geojson"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"no region {name!r}")
        return _json.loads(path.read_text())

    @app.delete("/api/project/regions/{name}")
    def region_delete(name: str):
        proj = _project()
        path = proj.regions_dir / f"{name}.geojson"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"no region {name!r}")
        path.unlink()
        return {"deleted": f"regions/{name}.geojson"}

    def _compose_active():
        import numpy as np

        from fmesh2.hfun import compose

        proj = _project()
        try:
            result = compose(proj)
        except (ValueError, NotImplementedError, FileNotFoundError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        finite = result.grid.values[np.isfinite(result.grid.values)]
        return result, finite

    @app.get("/api/project/hfun")
    def project_hfun():
        import numpy as np

        result, finite = _compose_active()
        g = result.grid
        values = np.asarray(g.values, dtype=np.float32)
        return Response(
            content=values.tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Hfun-Nlat": str(g.lat.size),
                "X-Hfun-Nlon": str(g.lon.size),
                "X-Hfun-Min": str(float(finite.min()) if finite.size else 0.0),
                "X-Hfun-Max": str(float(finite.max()) if finite.size else 0.0),
            },
        )

    @app.get("/api/project/estimate")
    def project_estimate():
        from fmesh2.hfun import estimate_size

        result, finite = _compose_active()
        if not finite.size:
            raise HTTPException(status_code=422, detail="hfun field is empty")
        est = estimate_size(result.grid)
        return {
            "elements_including_land": est.elements_including_land,
            "ocean_nodes": est.ocean_nodes,
            "ocean_fraction": est.ocean_fraction,
            "warnings": [w.message for w in result.warnings],
        }

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

    # ---------------------------------------------------- local refinement

    @app.post("/api/mesh/{mesh_id}/refine")
    def refine_mesh(mesh_id: str, req: RefineRequest):
        entry = _entry(mesh_id)
        if req.topo is not None and not Path(req.topo).expanduser().exists():
            raise HTTPException(status_code=400, detail=f"topo file not found: {req.topo}")
        if req.out_dir is not None:
            out = Path(req.out_dir).expanduser()
        else:
            out = entry.path.parent / f"{entry.path.name}_refined"
            k = 1
            while out.exists():
                k += 1
                out = entry.path.parent / f"{entry.path.name}_refined-{k}"
        job = jobs.start_refine(
            {
                "mesh_dir": str(entry.path),
                "polygon": [list(p) for p in req.polygon],
                "target_km": req.target_km,
                "transition_km": req.transition_km,
                "topo": req.topo,
                "minimum_depth": req.minimum_depth,
                "out_dir": str(out),
            }
        )
        return {"job_id": job.id, "out_dir": str(out)}

    @app.get("/api/mesh/{mesh_id}/mapping")
    def mesh_mapping(mesh_id: str):
        entry = _entry(mesh_id)
        path = entry.path / "mapping.csv"
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"mesh {mesh_id!r} has no old->new mapping table (not a refined mesh?)",
            )
        return FileResponse(path, media_type="text/csv", filename="mapping.csv")

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
