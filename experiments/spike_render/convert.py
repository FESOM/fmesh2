"""Convert a FESOM2 mesh (nod2d.out / elem2d.out / aux3d.out) to binary GPU buffers.

Usage: python convert.py /path/to/mesh_dir /path/to/out_dir [name]

Writes: pos.bin (float32 xyz on unit sphere, Y = north pole),
        tri.bin (uint32 triangle indices), edge.bin (uint32 unique edge indices),
        res.bin (float32 mean edge length per node, km),
        dep.bin (float32 bottom depth per node, m), meta.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    mesh_dir = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    name = sys.argv[3] if len(sys.argv) > 3 else mesh_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    nod = pd.read_csv(mesh_dir / "nod2d.out", sep=r"\s+", header=None, skiprows=1).to_numpy()
    n = nod.shape[0]
    print(f"[{time.time()-t0:6.1f}s] nod2d: {n:,} nodes")

    tri = pd.read_csv(mesh_dir / "elem2d.out", sep=r"\s+", header=None, skiprows=1).to_numpy()
    tri = (tri[:, :3].astype(np.int64) - 1)
    ntri = tri.shape[0]
    print(f"[{time.time()-t0:6.1f}s] elem2d: {ntri:,} triangles")

    # duplicate vertices of dateline-crossing triangles (shift west side +360)
    # so the mesh renders correctly in Plate Carree as well as on the sphere
    lon_deg = (nod[:, 1] + 180.0) % 360.0 - 180.0
    lat_deg = nod[:, 2]
    tl = lon_deg[tri]
    cross = (tl.max(axis=1) - tl.min(axis=1)) > 180.0
    west = (tl < 0.0) & cross[:, None]
    dup_src = np.unique(tri[west])
    remap = np.full(n, -1, np.int64)
    remap[dup_src] = n + np.arange(dup_src.shape[0])
    tri[west] = remap[tri[west]]
    lon2 = np.concatenate([lon_deg, lon_deg[dup_src] + 360.0])
    lat2 = np.concatenate([lat_deg, lat_deg[dup_src]])
    nv = lon2.shape[0]
    print(f"[{time.time()-t0:6.1f}s] dateline: {cross.sum():,} crossing tris, "
          f"{dup_src.shape[0]:,} duplicated verts")

    lon = np.deg2rad(lon2)
    lat = np.deg2rad(lat2)
    pos = np.empty((nv, 3), np.float32)
    coslat = np.cos(lat)
    pos[:, 0] = coslat * np.cos(lon)
    pos[:, 1] = np.sin(lat)              # north pole up (three.js Y-up)
    pos[:, 2] = -coslat * np.sin(lon)    # right-handed, east positive

    # Plate Carree positions for the globe<->map morph (same scale: equator = 2*pi)
    pos2 = np.zeros((nv, 3), np.float32)
    pos2[:, 0] = lon
    pos2[:, 1] = lat

    # unique undirected edges via 64-bit keys
    e = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    e.sort(axis=1)
    key = np.unique(e[:, 0].astype(np.uint64) * np.uint64(nv) + e[:, 1].astype(np.uint64))
    e0 = (key // np.uint64(nv)).astype(np.uint32)
    e1 = (key % np.uint64(nv)).astype(np.uint32)
    nedge = e0.shape[0]
    print(f"[{time.time()-t0:6.1f}s] edges: {nedge:,} unique")

    # local resolution: mean length of edges touching each node, in km
    chord = np.linalg.norm(pos[e0].astype(np.float64) - pos[e1].astype(np.float64), axis=1)
    length_km = 2.0 * np.arcsin(np.clip(chord / 2.0, 0, 1)) * 6371.0
    sums = np.bincount(e0, weights=length_km, minlength=nv) + np.bincount(e1, weights=length_km, minlength=nv)
    cnts = np.bincount(e0, minlength=nv) + np.bincount(e1, minlength=nv)
    res = (sums / np.maximum(cnts, 1)).astype(np.float32)
    print(f"[{time.time()-t0:6.1f}s] resolution: {np.nanmin(res):.2f} .. {np.nanmax(res):.2f} km")

    # depths from aux3d: nlev, levels, then one depth per node
    dep = np.zeros(nv, np.float32)
    aux = mesh_dir / "aux3d.out"
    if aux.exists():
        vals = pd.read_csv(aux, header=None, sep=r"\s+").to_numpy().ravel()
        nlev = int(vals[0])
        node_vals = vals[1 + nlev: 1 + nlev + n]
        if node_vals.shape[0] == n:
            d = np.abs(node_vals)
            dep = np.concatenate([d, d[dup_src]]).astype(np.float32)
            print(f"[{time.time()-t0:6.1f}s] aux3d: {nlev} levels, depth {dep.min():.0f} .. {dep.max():.0f} m")
        else:
            print(f"[{time.time()-t0:6.1f}s] aux3d: no per-node depths found ({node_vals.shape[0]} != {n})")

    pos.tofile(out_dir / "pos.bin")
    pos2.tofile(out_dir / "pos2.bin")
    tri.astype(np.uint32).tofile(out_dir / "tri.bin")
    np.column_stack([e0, e1]).tofile(out_dir / "edge.bin")
    res.tofile(out_dir / "res.bin")
    dep.tofile(out_dir / "dep.bin")

    meta = {
        "name": name,
        "nodes": int(n),
        "verts": int(nv),
        "triangles": int(ntri),
        "edges": int(nedge),
        "res_range": [float(np.percentile(res, 2)), float(np.percentile(res, 98))],
        "dep_range": [0.0, float(np.percentile(dep, 98))],
        "bytes": {f: (out_dir / f).stat().st_size for f in
                  ["pos.bin", "pos2.bin", "tri.bin", "edge.bin", "res.bin", "dep.bin"]},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[{time.time()-t0:6.1f}s] wrote {sum(meta['bytes'].values())/1e6:.0f} MB to {out_dir}")


if __name__ == "__main__":
    main()
