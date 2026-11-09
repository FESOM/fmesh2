"""Refinement region files (GeoJSON) — repeatable local refinements.

Every local refinement writes the polygon it used, together with its
settings, as `region.geojson` next to the refined mesh. Loading that file
in the viewer's refine panel (or any single-ring region GeoJSON, e.g. one
drawn for a project) refines exactly the same area on another mesh.

Format: a FeatureCollection with ONE Polygon feature; the settings are its
properties (`target_km`, `transition_km` (null = automatic band), `topo`,
`minimum_depth`, `source_mesh`). Files with more than one ring (fmesh
inner + outer regions) are rejected: refinement takes a single polygon and
makes its own transition band.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fmesh2.hfun.region_layer import load_region_rings

REGION_FILENAME = "region.geojson"
SETTING_KEYS = ("target_km", "transition_km", "topo", "minimum_depth", "source_mesh")


class RegionFileError(ValueError):
    """Unusable refinement region file."""


def write_refine_region(path: str | Path, polygon, settings: dict) -> Path:
    """Write the (lon, lat) polygon and the refinement settings to `path`."""
    ring = np.asarray(polygon, dtype=np.float64)[:, :2]
    if not np.array_equal(ring[0], ring[-1]):
        ring = np.vstack([ring, ring[:1]])
    props = {"fmesh2": "refine-region"}
    props.update({k: settings.get(k) for k in SETTING_KEYS})
    feature = {
        "type": "Feature",
        "properties": props,
        "geometry": {"type": "Polygon", "coordinates": [ring.tolist()]},
    }
    path = Path(path)
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=1))
    return path


def read_refine_region(path: str | Path) -> tuple[np.ndarray, dict]:
    """(closed (n, 2) lon/lat ring, settings) of a single-ring region file.

    `settings` holds the keys of `SETTING_KEYS` that the file defines; a
    region GeoJSON without refinement settings gives an empty dict.
    """
    path = Path(path)
    try:
        inner, outer = load_region_rings(path)
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise RegionFileError(f"cannot read region file {path}: {exc}") from exc
    if outer is not None:
        raise RegionFileError(
            f"{path} contains two rings (fmesh inner + outer); refinement takes a "
            "single polygon — use one ring and set the transition km instead"
        )
    gj = json.loads(path.read_text())
    features = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    props = next(
        (f.get("properties") or {} for f in features if isinstance(f, dict)), {}
    )
    settings = {k: props[k] for k in SETTING_KEYS if k in props}
    return inner, settings
