"""FESOM2 ASCII mesh format (nod2d.out / elem2d.out / aux3d.out).

Readers use pandas' C engine (`sep=r"\\s+"` hits the whitespace fast path);
NG5 (7.4M nodes) parses in a few seconds. Writers use pandas' default float
formatting (shortest roundtrip repr), so read -> write -> read is bit-exact.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from fmesh2.core.mesh import Mesh


def _read_table(path: Path, n_cols_min: int) -> tuple[int, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"missing mesh file: {path}")
    with open(path) as f:
        header = f.readline().split()
    try:
        count = int(header[0])
    except (IndexError, ValueError):
        raise ValueError(f"{path}: first line must be the record count") from None
    # round_trip: the C engine's default float parser loses 1 ulp on long
    # decimals — bit-exact reads underpin the M3 outside-untouched stitch
    # guarantee and M4 journal-replay hashes (costs ~2.8x on float parsing)
    df = pd.read_csv(
        path, sep=r"\s+", header=None, skiprows=1, float_precision="round_trip"
    )
    if len(df) < count:
        raise ValueError(f"{path}: header promises {count} rows, found {len(df)}")
    if df.shape[1] < n_cols_min:
        raise ValueError(f"{path}: expected at least {n_cols_min} columns, found {df.shape[1]}")
    return count, df.iloc[:count]


def read_fesom(mesh_dir: str | Path) -> Mesh:
    mesh_dir = Path(mesh_dir)

    n_nodes, nod = _read_table(mesh_dir / "nod2d.out", 4)
    lon = nod.iloc[:, 1].to_numpy(dtype=np.float64)
    lat = nod.iloc[:, 2].to_numpy(dtype=np.float64)
    coast = nod.iloc[:, 3].to_numpy(dtype=np.int8)

    _, elem = _read_table(mesh_dir / "elem2d.out", 3)
    tri = elem.iloc[:, :3].to_numpy(dtype=np.int64) - 1
    if tri.min() < 0 or tri.max() >= n_nodes:
        raise ValueError(f"{mesh_dir}/elem2d.out: node ids out of range 1..{n_nodes}")
    tri = tri.astype(np.int32)

    levels = None
    depth = None
    aux = mesh_dir / "aux3d.out"
    if aux.exists():
        vals = (
            pd.read_csv(aux, header=None, dtype=np.float64, float_precision="round_trip")
            .to_numpy()
            .ravel()
        )
        nlev = int(vals[0])
        if 1 + nlev > len(vals):
            raise ValueError(f"{aux}: promises {nlev} levels, file too short")
        levels = vals[1 : 1 + nlev]
        node_vals = vals[1 + nlev :]
        if len(node_vals) >= n_nodes:
            depth = node_vals[:n_nodes]

    return Mesh(lon=lon, lat=lat, tri=tri, depth=depth, coast_flag=coast, levels=levels)


def write_fesom(mesh: Mesh, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    coast = mesh.coast_flag
    if coast is None:
        coast = np.zeros(mesh.n_nodes, dtype=np.int8)
    nod = pd.DataFrame(
        {
            "id": np.arange(1, mesh.n_nodes + 1),
            "lon": mesh.lon,
            "lat": mesh.lat,
            "flag": coast.astype(np.int64),
        }
    )
    # shortest round-trippable decimal per float: pandas' default float writer
    # keeps only 16 significant digits, losing 1 ulp on long-decimal values
    fmt = lambda x: repr(float(x))  # noqa: E731

    with open(out_dir / "nod2d.out", "w") as f:
        f.write(f"{mesh.n_nodes}\n")
        nod.to_csv(f, sep=" ", header=False, index=False, float_format=fmt)

    elem = pd.DataFrame(np.asarray(mesh.tri, dtype=np.int64) + 1)
    with open(out_dir / "elem2d.out", "w") as f:
        f.write(f"{mesh.n_tri}\n")
        elem.to_csv(f, sep=" ", header=False, index=False)

    if mesh.levels is not None or mesh.depth is not None:
        levels = mesh.levels if mesh.levels is not None else np.array([])
        with open(out_dir / "aux3d.out", "w") as f:
            f.write(f"{len(levels)}\n")
            if len(levels):
                pd.Series(levels).to_csv(f, header=False, index=False, float_format=fmt)
            if mesh.depth is not None:
                pd.Series(mesh.depth).to_csv(f, header=False, index=False, float_format=fmt)
