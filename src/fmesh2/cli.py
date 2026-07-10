from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from fmesh2.data.coastlines import cache_dir
    from fmesh2.server.app import create_app
    from fmesh2.server.state import AppState

    state = AppState(cache_dir=cache_dir() / "buffers")
    for path in args.mesh:
        entry = state.open_mesh(path)
        print(f"opened {entry.path} as {entry.id} "
              f"({entry.buffers.meta['nodes']:,} nodes)")

    url = f"http://{args.host}:{args.port}/"
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    print(f"fmesh2 viewer at {url}")
    uvicorn.run(create_app(state), host=args.host, port=args.port, log_level="warning")
    return 0


def _cmd_convert(args: argparse.Namespace) -> int:
    import json

    from fmesh2.viewer import load_or_build

    buffers = load_or_build(args.mesh_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("pos", "pos2", "tri", "edge", "res", "dep"):
        buffers.buffer(name).tofile(out / f"{name}.bin")
    (out / "meta.json").write_text(json.dumps(buffers.meta, indent=1))
    total = sum(buffers.meta["bytes"].values())
    print(f"wrote {total/1e6:.0f} MB of buffers to {out}")
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    """Headless reproducible build: project.yaml -> FESOM2 mesh in out/.

    Exit codes: 0 ok, 1 build error, 2 bad project file.
    """
    from fmesh2.core.io_fesom import write_fesom
    from fmesh2.generate import JigsawError, build_project
    from fmesh2.generate.jigsaw_driver import RawMesh, hfun_hash
    from fmesh2.hfun import compose
    from fmesh2.project import Project, ProjectError

    try:
        project = Project.load(args.project)
    except ProjectError as exc:
        print(f"error: {exc}")
        return 2
    if project.topo is None:
        print("error: project.topo is required to build a mesh")
        return 2
    project.ensure_layout()

    raw = None
    if args.only_postprocess:
        import numpy as np

        cache_file = project.cache_dir / f"jigsaw_{hfun_hash(compose(project).grid)}.npz"
        if not cache_file.exists():
            print(f"error: --only-postprocess needs a cached jigsaw mesh at {cache_file}")
            return 1
        d = np.load(cache_file)
        raw = RawMesh(lon=d["lon"], lat=d["lat"], tri=d["tri"])
        print(f"reusing cached jigsaw mesh {cache_file.name}")

    try:
        mesh, _ = build_project(project, progress=print, raw=raw)
    except (JigsawError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}")
        return 1
    write_fesom(mesh, project.out_dir)
    print(f"wrote {mesh.n_nodes:,} nodes / {mesh.n_tri:,} triangles to {project.out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fmesh2", description="FESOM2 mesh tool")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="open the viewer (default command)")
    serve.add_argument("mesh", nargs="*", help="FESOM2 mesh directories to preload")
    serve.add_argument("--port", type=int, default=8017)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(func=_cmd_serve)

    conv = sub.add_parser("convert", help="FESOM2 ASCII mesh -> binary viewer buffers")
    conv.add_argument("mesh_dir")
    conv.add_argument("out_dir")
    conv.set_defaults(func=_cmd_convert)

    build = sub.add_parser("build", help="headless mesh generation from a project.yaml")
    build.add_argument("project", help="project.yaml or project directory")
    build.add_argument(
        "--only-postprocess",
        action="store_true",
        help="skip jigsaw and reuse the cached raw mesh (reproducibility path)",
    )
    build.set_defaults(func=_cmd_build)

    if argv is None:
        import sys

        argv = sys.argv[1:]
    # `fmesh2 [paths...]` == `fmesh2 serve [paths...]`
    if not argv or argv[0] not in {"serve", "convert", "build", "-h", "--help"}:
        argv = ["serve", *argv]
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
