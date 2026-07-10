"""Lightweight KML region import — no fastkml dependency.

fmesh regions are Google Earth files with two Placemarks tracing an inner and
an outer ring (any geometry type that carries `<coordinates>`). Rings are
extracted namespace-agnostically, closed, and ordered by containment like
fmesh `from_kml` (which ring lies inside the other), independent of file
order. Imported regions are stored as GeoJSON with `role` properties.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import shapely

from fmesh2.core.projection import ring_center, stereo_xy


class KMLError(ValueError):
    """Unusable KML region file."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def read_kml_rings(path: str | Path) -> list[tuple[str, np.ndarray]]:
    """All Placemark rings as (name, closed (n, 2) lon/lat array)."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise KMLError(f"{path} is not valid XML: {e}") from e
    rings: list[tuple[str, np.ndarray]] = []
    for placemark in (el for el in root.iter() if _local(el.tag) == "Placemark"):
        name = ""
        coords: list[np.ndarray] = []
        for el in placemark.iter():
            if _local(el.tag) == "name" and el.text:
                name = name or el.text.strip()
            elif _local(el.tag) == "coordinates" and el.text:
                triples = [t.split(",") for t in el.text.split()]
                arr = np.array([[float(t[0]), float(t[1])] for t in triples])
                if arr.shape[0] >= 3:
                    coords.append(arr)
        for arr in coords:
            if not np.array_equal(arr[0], arr[-1]):
                arr = np.vstack([arr, arr[:1]])  # close, like fmesh from_kml
            rings.append((name, arr))
    if not rings:
        raise KMLError(f"no Placemark with <coordinates> found in {path}")
    return rings


def order_rings(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(inner, outer) by containment, tested in a shared stereographic plane."""
    lon0, lat0 = ring_center(
        np.concatenate([a[:, 0], b[:, 0]]), np.concatenate([a[:, 1], b[:, 1]])
    )
    poly_a = shapely.Polygon(np.column_stack(stereo_xy(a[:, 0], a[:, 1], lon0, lat0)))
    poly_b = shapely.Polygon(np.column_stack(stereo_xy(b[:, 0], b[:, 1], lon0, lat0)))
    # full-polygon containment: a representative point of the outer ring could
    # itself lie inside the inner ring
    if poly_b.contains(poly_a):
        return a, b
    if poly_a.contains(poly_b):
        return b, a
    raise KMLError("region rings do not nest: neither ring contains the other")


def kml_to_geojson(kml_path: str | Path, out_path: str | Path) -> Path:
    """Convert an fmesh two-ring region KML to a role-tagged GeoJSON file."""
    rings = read_kml_rings(kml_path)
    if len(rings) != 2:
        raise KMLError(
            f"{kml_path} has {len(rings)} rings; fmesh region KMLs contain exactly "
            "2 (inner + outer)"
        )
    inner, outer = order_rings(rings[0][1], rings[1][1])
    features = [
        {
            "type": "Feature",
            "properties": {"role": role, "name": rings[0][0].rsplit("_", 1)[0]},
            "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
        }
        for role, ring in (("inner", inner), ("outer", outer))
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return out_path
