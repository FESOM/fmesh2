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

    if argv is None:
        import sys

        argv = sys.argv[1:]
    # `fmesh2 [paths...]` == `fmesh2 serve [paths...]`
    if not argv or argv[0] not in {"serve", "convert", "-h", "--help"}:
        argv = ["serve", *argv]
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
