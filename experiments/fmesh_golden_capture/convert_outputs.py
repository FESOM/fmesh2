"""Convert legacy fmesh run artifacts into committed golden fixtures.

Runs INSIDE the fmesh_legacy env (py3.9) from the capture work dir, right
after ``python fmesh.py`` finished there. Produces plain npz/text files so
the fmesh2 test suite (py3.12) never needs to unpickle legacy objects.

Usage: python convert_outputs.py <workdir> <out_fixture_dir>
"""

import pickle
import shutil
import sys
from pathlib import Path

import numpy as np

workdir = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

# hfun field + grid vectors + per-region rings from define_resolutions()
with open(workdir / "_result_temp.pkl", "rb") as f:
    regions, result, latitudes, longitudes = pickle.load(f)

arrays = {
    "hfun": np.asarray(result, dtype=np.float64),
    "latitudes": np.asarray(latitudes, dtype=np.float64),
    "longitudes": np.asarray(longitudes, dtype=np.float64),
}
for reg in regions:
    name = reg["name"]
    for ring_key, tag in (("Polygon inside", "inner"), ("Polygon outside", "outer")):
        lon, lat = reg[ring_key]
        arrays[f"region_{name}_{tag}_lon"] = np.asarray(lon, dtype=np.float64)
        arrays[f"region_{name}_{tag}_lat"] = np.asarray(lat, dtype=np.float64)
np.savez_compressed(out / "hfun_field.npz", **arrays)

# smoothed coastline points actually used by refine_along_coastlines
coast_pkl = workdir / "coastlines" / "min_length=100km_averaging=50km.pkl"
with open(coast_pkl, "rb") as f:
    lon_coast, lat_coast = pickle.load(f)
np.savez_compressed(
    out / "coast_points.npz",
    lon=np.asarray(lon_coast, dtype=np.float64),
    lat=np.asarray(lat_coast, dtype=np.float64),
)

# raw jigsaw mesh counts (pre land-cut) for driver-level comparison
with open(workdir / "_mesh_temp.pkl", "rb") as f:
    mesh = pickle.load(f)
with open(out / "jigsaw_raw_counts.txt", "w") as f:
    f.write(f"vert3 {len(mesh.vert3)}\n")
    f.write(f"tria3 {len(mesh.tria3)}\n")

# final FESOM2 mesh triplet
for name in ("nod2d.out", "elem2d.out", "aux3d.out"):
    shutil.copy2(workdir / name, out / name)

print("golden fixtures written to", out)
