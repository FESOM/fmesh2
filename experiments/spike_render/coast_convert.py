"""Convert a coastline GeoJSON to line-segment buffers (sphere + Plate Carree).

Usage: python coast_convert.py ne_10m_coastline.geojson out_dir
Writes coast_pos.bin / coast_pos2.bin: float32 xyz pairs, one segment per vertex pair.
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    src = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    gj = json.loads(src.read_text())

    seg_a, seg_b = [], []
    for feat in gj["features"]:
        geom = feat["geometry"]
        lines = geom["coordinates"] if geom["type"] == "MultiLineString" else [geom["coordinates"]]
        for line in lines:
            arr = np.asarray(line, dtype=np.float64)[:, :2]
            if arr.shape[0] < 2:
                continue
            seg_a.append(arr[:-1])
            seg_b.append(arr[1:])

    a = np.concatenate(seg_a)
    b = np.concatenate(seg_b)

    # normalize lon, then shift the negative end of dateline-crossing segments +360
    for arr in (a, b):
        arr[:, 0] = (arr[:, 0] + 180.0) % 360.0 - 180.0
    crossing = np.abs(a[:, 0] - b[:, 0]) > 180.0
    for arr in (a, b):
        west = crossing & (arr[:, 0] < 0.0)
        arr[west, 0] += 360.0

    pts = np.empty((a.shape[0] * 2, 2), np.float64)
    pts[0::2] = a
    pts[1::2] = b
    lon = np.deg2rad(pts[:, 0])
    lat = np.deg2rad(pts[:, 1])

    pos = np.empty((pts.shape[0], 3), np.float32)
    coslat = np.cos(lat)
    pos[:, 0] = coslat * np.cos(lon)
    pos[:, 1] = np.sin(lat)
    pos[:, 2] = -coslat * np.sin(lon)

    pos2 = np.zeros_like(pos)
    pos2[:, 0] = lon
    pos2[:, 1] = lat

    pos.tofile(out_dir / "coast_pos.bin")
    pos2.tofile(out_dir / "coast_pos2.bin")
    print(f"{a.shape[0]:,} segments ({crossing.sum():,} crossed dateline), "
          f"{pos.nbytes/1e6:.1f} MB per buffer -> {out_dir}")


if __name__ == "__main__":
    main()
