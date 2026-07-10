"""Generate the committed synthetic global topography fixture.

An analytic, deterministic idealized world on a 0.5 deg grid: smooth Gaussian
continents with wide shelf gradients, one continent straddling the dateline,
a polar (Antarctica-like) cap, and an enclosed below-sea-level lake inside the
big continent (for lake-elimination tests). Deep ocean is -4000 m.

Regenerate with:  python tests/fixtures/make_topo_synthetic.py [out.nc]
The e2e tests (Tasks 16/19) read the committed `topo_synthetic.nc`; a test
guards that the file matches this code.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

#: (center lon, center lat, sigma lon, sigma lat, peak height above the -4000 floor)
CONTINENTS = [
    (20.0, 15.0, 25.0, 18.0, 5000.0),  # big continent, hosts the lake
    (120.0, -25.0, 18.0, 12.0, 4800.0),
    (-90.0, 45.0, 20.0, 14.0, 4700.0),
    (180.0, 5.0, 12.0, 10.0, 4600.0),  # straddles the dateline
]
LAKE = (20.0, 15.0, 4.0, 3.0, 1400.0)  # dip carved into the big continent's top
POLAR_CAP = (10.0, 5000.0)  # (sigma lat, height): land poleward of ~-83
FLOOR = -4000.0


def synthetic_elevation(lon, lat) -> np.ndarray:
    """Elevation (m, negative down) at lon/lat arrays (broadcast together)."""
    lon, lat = np.broadcast_arrays(np.asarray(lon, float), np.asarray(lat, float))

    def bump(lon0, lat0, slon, slat, height):
        dlon = (lon - lon0 + 180.0) % 360.0 - 180.0  # wrapped lon distance
        return height * np.exp(-0.5 * ((dlon / slon) ** 2 + ((lat - lat0) / slat) ** 2))

    elev = np.full(lon.shape, FLOOR)
    for c in CONTINENTS:
        elev += bump(*c)
    elev -= bump(*LAKE)
    sig, h = POLAR_CAP
    elev += h * np.exp(-0.5 * ((lat + 90.0) / sig) ** 2)
    return elev


def build_dataset(step: float = 0.5) -> xr.Dataset:
    lon = np.arange(-180.0, 180.0, step)  # no duplicate seam column
    lat = np.linspace(-90.0, 90.0, int(round(180.0 / step)) + 1)
    elev = synthetic_elevation(lon[None, :], lat[:, None]).astype(np.float32)
    return xr.Dataset(
        {"topo": (("lat", "lon"), elev)},
        coords={"lat": lat, "lon": lon},
        attrs={
            "title": "fmesh2 synthetic global topography (test fixture)",
            "source": "tests/fixtures/make_topo_synthetic.py",
        },
    )


def main(out: str | None = None) -> Path:
    path = Path(out) if out else Path(__file__).parent / "topo_synthetic.nc"
    ds = build_dataset()
    ds.to_netcdf(path, encoding={"topo": {"zlib": True, "complevel": 6}})
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")
    return path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
