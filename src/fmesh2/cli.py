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


def _cmd_redepth(args: argparse.Namespace) -> int:
    """Re-sample depths from a topography file and smooth them.

    local: only the nodes a refinement added (old nodes unchanged);
    global: every node. Exit codes: 0 ok, 1 processing error, 2 bad arguments.
    """
    import json

    from fmesh2.core.io_fesom import read_fesom, write_fesom
    from fmesh2.data.topo import Topo, TopoError
    from fmesh2.edit.redepth import RedepthError, redepth_global, redepth_local

    local = args.mode == "local"
    if local and args.original is None:
        print("error: --mode local needs --original (the mesh the refinement started from)")
        return 2
    if not local and args.original is not None:
        print("error: --original only applies to --mode local")
        return 2

    mesh_dir = Path(args.mesh_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    inputs = {mesh_dir.resolve()}
    if local:
        inputs.add(Path(args.original).expanduser().resolve())
    if out_dir.resolve() in inputs:
        print("error: OUT_DIR must differ from the input mesh directories")
        return 2
    try:
        mesh = read_fesom(mesh_dir)
        original = read_fesom(Path(args.original).expanduser()) if local else None
        topo = Topo(Path(args.topo).expanduser(), var=args.topo_var)
    except (FileNotFoundError, ValueError, TopoError) as exc:
        print(f"error: {exc}")
        return 2

    settings = dict(min_depth=args.min_depth, max_depth=args.max_depth,
                    iterations=args.iterations, center_weight=args.center_weight)
    try:
        if local:
            res = redepth_local(original, mesh, topo, **settings)
        else:
            res = redepth_global(mesh, topo, **settings)
    except RedepthError as exc:
        print(f"error: {exc}")
        return 1
    rep = res.report
    write_fesom(res.mesh, out_dir)
    (out_dir / "redepth_report.json").write_text(json.dumps(rep, indent=1))

    cap = "no maximum" if args.max_depth is None else f"maximum {args.max_depth:g} m"
    print(f"minimum depth {args.min_depth:g} m, {cap}, "
          f"{args.iterations} smoothing iteration(s), centre weight {args.center_weight:g}")
    if local:
        print(f"{rep['n_new']:,} new nodes re-sampled and smoothed, "
              f"{rep['n_old']:,} old nodes unchanged")
        land = rep["n_new_on_topo_land"]
        if rep["n_old_depth_differs_from_original"]:
            print(f"note: {rep['n_old_depth_differs_from_original']:,} old nodes already "
                  "differ in depth from the original mesh (left as they are)")
        rows = (("untouched", "untouched mesh"), ("seam", "seam (old-new)"),
                ("new", "new nodes"))
    else:
        print(f"all {rep['n_nodes']:,} nodes re-sampled and smoothed")
        land = rep["n_on_topo_land"]
        rows = (("all", "all edges"),)
    if land:
        print(f"note: {land:,} nodes are land in the topography (set to the minimum depth)")
    print("edge steepness |h1-h2|/(h1+h2):   median      p99      max")
    for key, label in rows:
        st = rep["steepness"][key]
        if st:
            print(f"  {label:<28} {st['median']:8.4f} {st['p99']:8.4f} {st['max']:8.4f}")
    print(f"wrote {out_dir}")
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

    red = sub.add_parser(
        "redepth",
        help="re-sample depths from a topography file and smooth them "
             "(nodes added by refinement, or all nodes)",
    )
    red.add_argument("mesh_dir", help="FESOM2 mesh directory")
    red.add_argument("out_dir", help="output mesh directory (must differ from the inputs)")
    red.add_argument("--topo", required=True, help="topography NetCDF (RTopo/GEBCO style)")
    red.add_argument("--topo-var", default=None, help="elevation variable, if not auto-detected")
    red.add_argument("--mode", choices=("local", "global"), default="local",
                     help="local: only nodes added by refinement, old nodes unchanged "
                          "(default); global: every node")
    red.add_argument("--original", default=None,
                     help="local mode: mesh directory the refinement started from")
    red.add_argument("--min-depth", type=float, default=30.0,
                     help="minimum depth in m (default 30, as smooth_seafloor.f90)")
    red.add_argument("--max-depth", type=float, default=None,
                     help="maximum depth in m (default: none, the topography sets it)")
    red.add_argument("--iterations", type=int, default=1,
                     help="smoothing iterations (default 1, as smooth_seafloor.f90)")
    red.add_argument("--center-weight", type=float, default=2.0,
                     help="relative centre weight of the smoother, >= 1 "
                          "(default 2, as smooth_seafloor.f90)")
    red.set_defaults(func=_cmd_redepth)

    if argv is None:
        import sys

        argv = sys.argv[1:]
    # `fmesh2 [paths...]` == `fmesh2 serve [paths...]`
    if not argv or argv[0] not in {"serve", "convert", "build", "redepth", "-h", "--help"}:
        argv = ["serve", *argv]
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
