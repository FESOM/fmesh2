"""Mesh generation pipeline: hfun compose -> jigsaw -> postprocess."""

from __future__ import annotations

from typing import Callable

from fmesh2.core.mesh import Mesh
from fmesh2.data.topo import open_topo
from fmesh2.hfun import ComposeResult, compose
from fmesh2.project import Project

from .jigsaw_driver import JigsawError, RawMesh, hfun_hash, run_jigsaw
from .postprocess import postprocess

__all__ = [
    "JigsawError",
    "RawMesh",
    "build_project",
    "hfun_hash",
    "postprocess",
    "run_jigsaw",
]


def build_project(
    project: Project,
    progress: Callable[[str], None] | None = None,
    process_callback=None,
    raw: RawMesh | None = None,
) -> tuple[Mesh, ComposeResult]:
    """Full pipeline for a project; jigsaw results are cached in the project's
    cache dir. Pass `raw` to postprocess an existing raw mesh (--only-postprocess)."""
    if project.topo is None:
        raise ValueError("project.topo is required to build a mesh (land cut needs it)")
    result = compose(project)
    for w in result.warnings:
        if progress:
            progress(f"warning: {w.message}")
    if raw is None:
        cache_dir = project.cache_dir if project.project_dir is not None else None
        raw = run_jigsaw(
            result.grid,
            cache_dir=cache_dir,
            progress=progress,
            process_callback=process_callback,
        )
    if progress:
        progress(f"raw mesh: {raw.n_nodes} nodes, {raw.tri.shape[0]} triangles")
    topo = open_topo(str(project.topo_path), var=project.topo_var)
    mesh = postprocess(
        raw,
        topo,
        minimum_depth=project.minimum_depth,
        seeds=project.seeds,
        levels=project.levels,
    )
    if progress:
        progress(f"ocean mesh: {mesh.n_nodes} nodes, {mesh.n_tri} triangles")
    return mesh, result
