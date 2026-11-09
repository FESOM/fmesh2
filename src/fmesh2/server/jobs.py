"""Generation jobs: the build pipeline runs in a worker *process*, streaming
staged progress events back over a multiprocessing queue.

Process tree: server -> worker (spawn) -> jigsaw binary (own process group).
Cancellation therefore kills the jigsaw process group directly (its pid is
reported in an event as soon as it spawns) and then terminates the worker —
no orphaned grandchildren.

A supervisor thread per job drains the queue into the Job's event history;
WebSocket/polling handlers only ever read that state, so they need no
cross-thread wiring.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import traceback
from collections import deque
from dataclasses import dataclass, field
from queue import Empty
from typing import Callable

#: nominal progress fraction when a stage begins
STAGE_FRACTIONS = {
    "compose": 0.05,
    "jigsaw": 0.25,
    "postprocess": 0.80,
    "write": 0.90,
    "register": 0.95,
}
REFINE_FRACTIONS = {"extract": 0.05, "remesh": 0.30, "write": 0.85}
TERMINAL = ("done", "error", "cancelled")
MAX_EVENTS = 2000


def _run_build(project_path: str, queue) -> None:
    """Worker-process entry point (top-level for spawn picklability)."""
    try:
        from fmesh2.core.io_fesom import write_fesom
        from fmesh2.data.topo import open_topo
        from fmesh2.generate import postprocess, run_jigsaw
        from fmesh2.hfun import compose
        from fmesh2.project import Project

        def stage(name):
            queue.put({"type": "stage", "stage": name, "fraction": STAGE_FRACTIONS[name]})

        def log(line):
            queue.put({"type": "log", "line": line})

        project = Project.load(project_path)
        project.ensure_layout()

        stage("compose")
        result = compose(project)
        for w in result.warnings:
            log(f"warning: {w.message}")

        stage("jigsaw")
        raw = run_jigsaw(
            result.grid,
            cache_dir=project.cache_dir,
            progress=log,
            process_callback=lambda p: queue.put({"type": "jigsaw_pid", "pid": p.pid}),
        )
        log(f"raw mesh: {raw.n_nodes} nodes, {raw.tri.shape[0]} triangles")

        stage("postprocess")
        topo = open_topo(str(project.topo_path), var=project.topo_var)
        mesh = postprocess(
            raw,
            topo,
            minimum_depth=project.minimum_depth,
            seeds=project.seeds,
            levels=project.levels,
        )
        log(f"ocean mesh: {mesh.n_nodes} nodes, {mesh.n_tri} triangles")

        stage("write")
        write_fesom(mesh, project.out_dir)
        queue.put(
            {
                "type": "done",
                "out_dir": str(project.out_dir),
                "n_nodes": mesh.n_nodes,
                "n_tri": mesh.n_tri,
            }
        )
    except Exception as exc:  # noqa: BLE001 — everything must surface in the job
        queue.put(
            {"type": "error", "message": str(exc), "traceback": traceback.format_exc()}
        )


def _run_refine(args: dict, queue) -> None:
    """Worker-process entry point for local refinement (M3)."""
    try:
        from pathlib import Path

        from fmesh2.core.io_fesom import read_fesom, write_fesom
        from fmesh2.edit.cavity import extract_cavity
        from fmesh2.edit.local_remesh import remesh_cavity, write_mapping_csv

        def stage(name):
            queue.put({"type": "stage", "stage": name, "fraction": REFINE_FRACTIONS[name]})

        def log(line):
            queue.put({"type": "log", "line": line})

        out_dir = Path(args["out_dir"])

        stage("extract")
        mesh = read_fesom(args["mesh_dir"])
        cavity = extract_cavity(mesh, args["polygon"])
        log(
            f"cavity: {cavity.n_elems} elements, ring {len(cavity.ring_nodes)} "
            f"nodes in {cavity.n_loops} loop(s)"
        )

        stage("remesh")
        topo = None
        if args.get("topo"):
            from fmesh2.data.topo import open_topo

            topo = open_topo(args["topo"])
        res = remesh_cavity(
            mesh,
            cavity,
            args["target_km"],
            transition_km=args.get("transition_km"),
            topo=topo,
            minimum_depth=args.get("minimum_depth", 5.0),
            cache_dir=out_dir / "cache",
            progress=log,
            process_callback=lambda p: queue.put({"type": "jigsaw_pid", "pid": p.pid}),
        )
        rep = res.report
        achieved = rep["interior_resolution_km"]
        log(
            f"stitched: {rep['n_nodes_after']} nodes "
            f"({rep['n_new_interior_nodes']} new interior), "
            f"watertight={rep['watertight_seam']}"
            + (f", median new resolution {achieved:.1f} km" if achieved else "")
        )

        stage("write")
        write_fesom(res.mesh, out_dir)
        write_mapping_csv(out_dir / "mapping.csv", res.mapping)
        from fmesh2.io.refine_region import REGION_FILENAME, write_refine_region

        region_file = write_refine_region(
            out_dir / REGION_FILENAME,
            args["polygon"],
            {
                "target_km": args["target_km"],
                "transition_km": args.get("transition_km"),
                "topo": args.get("topo"),
                "minimum_depth": args.get("minimum_depth", 5.0),
                "source_mesh": str(Path(args["mesh_dir"]).resolve()),
            },
        )
        queue.put(
            {
                "type": "done",
                "out_dir": str(out_dir),
                "region_file": str(region_file),
                "n_nodes": res.mesh.n_nodes,
                "n_tri": res.mesh.n_tri,
                "report": rep,
            }
        )
    except Exception as exc:  # noqa: BLE001 — everything must surface in the job
        queue.put(
            {"type": "error", "message": str(exc), "traceback": traceback.format_exc()}
        )


@dataclass
class Job:
    id: str
    project_path: str
    status: str = "running"
    stage: str = "starting"
    fraction: float = 0.0
    log: deque = field(default_factory=lambda: deque(maxlen=80))
    events: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    jigsaw_pid: int | None = None

    def snapshot(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "fraction": self.fraction,
            "log_tail": list(self.log)[-15:],
            "result": self.result,
            "error": self.error,
        }

    def _push(self, event: dict) -> None:
        if len(self.events) < MAX_EVENTS:
            self.events.append(event)


class JobManager:
    """Owns build jobs; `register` is called with the output dir on success
    and returns metadata merged into the job result (e.g. the viewer mesh id)."""

    def __init__(self, register: Callable[[str], dict] | None = None):
        self.jobs: dict[str, Job] = {}
        self._register = register
        self._counter = 0
        self._ctx = mp.get_context("spawn")
        self._procs: dict[str, mp.Process] = {}
        self._lock = threading.Lock()

    def start_build(self, project_path: str) -> Job:
        return self._start(_run_build, (str(project_path),), label=str(project_path))

    def start_refine(self, args: dict) -> Job:
        """`args` are the `_run_refine` inputs (mesh_dir, polygon, target_km, ...)."""
        return self._start(_run_refine, (args,), label=args["mesh_dir"])

    def _start(self, target, args: tuple, label: str) -> Job:
        with self._lock:
            self._counter += 1
            job = Job(id=f"j{self._counter}", project_path=label)
            self.jobs[job.id] = job
        queue = self._ctx.Queue()
        proc = self._ctx.Process(target=target, args=(*args, queue), daemon=True)
        proc.start()
        self._procs[job.id] = proc
        threading.Thread(
            target=self._supervise, args=(job, proc, queue), daemon=True, name=f"job-{job.id}"
        ).start()
        return job

    def _supervise(self, job: Job, proc, queue) -> None:
        while True:
            if job.status in TERMINAL:
                break
            try:
                event = queue.get(timeout=0.2)
            except Empty:
                if not proc.is_alive():
                    if job.status == "running":
                        job.error = "worker process died unexpectedly"
                        job.status = "error"
                        job._push({"type": "error", "message": job.error})
                    break
                continue
            kind = event.get("type")
            if kind == "stage":
                job.stage = event["stage"]
                job.fraction = event["fraction"]
            elif kind == "log":
                job.log.append(event["line"])
            elif kind == "jigsaw_pid":
                job.jigsaw_pid = event["pid"]
            elif kind == "done":
                job.stage = "register"
                job.fraction = STAGE_FRACTIONS["register"]
                job._push({"type": "stage", "stage": "register", "fraction": job.fraction})
                result = {k: v for k, v in event.items() if k != "type"}
                if self._register is not None:
                    try:
                        result.update(self._register(event["out_dir"]))
                    except Exception as exc:  # noqa: BLE001
                        job.error = f"mesh registration failed: {exc}"
                        job.status = "error"
                        job._push({"type": "error", "message": job.error})
                        break
                job.result = result
                job.fraction = 1.0
                job.status = "done"
                job._push({"type": "done", **result})
                break
            elif kind == "error":
                job.error = event["message"]
                job.log.append(event.get("traceback", ""))
                job.status = "error"
            job._push(event)
            if job.status in TERMINAL:
                break
        proc.join(timeout=5)

    def cancel(self, job_id: str) -> Job:
        job = self.jobs[job_id]
        if job.status in TERMINAL:
            return job
        job.status = "cancelled"
        job._push({"type": "cancelled"})
        if job.jigsaw_pid is not None:
            try:  # jigsaw runs in its own session: pid == pgid
                os.killpg(job.jigsaw_pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        proc = self._procs.get(job_id)
        if proc is not None and proc.is_alive():
            proc.terminate()
        return job
