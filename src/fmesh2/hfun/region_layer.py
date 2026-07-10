"""Region polygon layer evaluator.

Two variants, both computed in a polygon-centered stereographic plane
(conformal — containment and nearest-point picks are exact; distances are
then measured as true great circles), so dateline- and pole-spanning
polygons need no special casing:

* single ring + `transition_km` (primary UX): `resolution` inside, blending
  to the local surrounding field across the band outside the ring;
* two rings (fmesh KML compatibility): `resolution` inside the inner ring,
  and between the rings the fmesh `refine` formula
  ``res + (base - res) * d_in / (d_in + d_out)`` with `base` sampled from the
  field at the grid cell nearest to the outer-ring exit point.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import shapely

from fmesh2.core.projection import (
    EARTH_RADIUS_KM,
    haversine_km,
    ring_center,
    stereo_inverse,
    stereo_xy,
)
from fmesh2.core.topology import sphere_xyz
from fmesh2.project import Project, RegionLayer

from .compose import register
from .grid import HfunGrid


def _close(ring: np.ndarray) -> np.ndarray:
    return ring if np.array_equal(ring[0], ring[-1]) else np.vstack([ring, ring[:1]])


def _densify(ring: np.ndarray, max_step_deg: float = 0.2) -> np.ndarray:
    """Subdivide ring edges linearly in (unwrapped) lon/lat so that projected
    edges follow the drawn lon/lat lines instead of long stereographic chords."""
    lon = np.rad2deg(np.unwrap(np.deg2rad(ring[:, 0])))
    lat = ring[:, 1]
    pieces = []
    for k in range(lon.size - 1):
        n = max(1, int(np.ceil(max(abs(lon[k + 1] - lon[k]), abs(lat[k + 1] - lat[k])) / max_step_deg)))
        t = np.linspace(0.0, 1.0, n, endpoint=False)
        pieces.append(np.column_stack([lon[k] + t * (lon[k + 1] - lon[k]), lat[k] + t * (lat[k + 1] - lat[k])]))
    pieces.append(np.array([[lon[-1], lat[-1]]]))
    return np.concatenate(pieces)


def load_region_rings(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """(inner, outer-or-None) closed lon/lat rings from a region GeoJSON."""
    gj = json.loads(Path(path).read_text())
    features = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    rings: list[np.ndarray] = []
    roles: list[str | None] = []
    for feat in features:
        geom = feat.get("geometry", feat)
        if geom["type"] == "Polygon":
            coords = geom["coordinates"][0]
        elif geom["type"] == "LineString":
            coords = geom["coordinates"]
        else:
            continue
        rings.append(_close(np.asarray(coords, dtype=np.float64)[:, :2]))
        roles.append(feat.get("properties", {}).get("role") if "properties" in feat else None)
    if len(rings) == 1:
        return rings[0], None
    if len(rings) == 2:
        if set(roles) == {"inner", "outer"}:
            return rings[roles.index("inner")], rings[roles.index("outer")]
        from fmesh2.io.kml import order_rings  # containment-based fallback

        return order_rings(rings[0], rings[1])
    raise ValueError(
        f"region file {path} must contain 1 (drawn) or 2 (fmesh inner+outer) rings, "
        f"found {len(rings)}"
    )


def _nearest_index(vec: np.ndarray, vals: np.ndarray) -> np.ndarray:
    i = np.searchsorted(vec, vals)
    i0 = np.clip(i - 1, 0, vec.size - 1)
    i1 = np.clip(i, 0, vec.size - 1)
    return np.where(np.abs(vals - vec[i0]) <= np.abs(vals - vec[i1]), i0, i1)


def _ring_endpoint_geo(cells_xy, boundary, lon0, lat0):
    """Nearest boundary point of each cell in the plane -> its lon/lat."""
    lines = shapely.shortest_line(cells_xy, boundary)
    coords = shapely.get_coordinates(lines)
    end_x, end_y = coords[1::2, 0], coords[1::2, 1]
    return stereo_inverse(end_x, end_y, lon0, lat0)


@register("region")
def region_field(layer: RegionLayer, grid: HfunGrid, project: Project) -> np.ndarray:
    inner, outer = load_region_rings(project.resolve_path(layer.path))
    inner = _densify(inner)
    if outer is not None:
        outer = _densify(outer)
    if outer is not None and layer.transition_km is not None:
        raise ValueError(
            f"region {layer.path}: has two rings AND transition_km — ambiguous; "
            "drop one of them"
        )
    if outer is None and layer.transition_km is None:
        raise ValueError(
            f"region {layer.path}: a single-ring region needs transition_km"
        )

    boundary_ring = inner if outer is None else outer
    all_pts = np.vstack([inner] + ([outer] if outer is not None else []))
    lon0, lat0 = ring_center(all_pts[:, 0], all_pts[:, 1])

    # cheap candidate preselection by angular distance from the region center
    center = sphere_xyz(np.array([lon0]), np.array([lat0]))[0]
    ring_cos = sphere_xyz(all_pts[:, 0], all_pts[:, 1]) @ center
    theta_ring = float(np.arccos(np.clip(ring_cos.min(), -1.0, 1.0)))
    margin_km = (layer.transition_km or 0.0) + 2.0 * max(grid.cell_km)
    theta_lim = min(theta_ring * 1.05 + margin_km / EARTH_RADIUS_KM, np.pi * 0.9)
    lon2d, lat2d = np.meshgrid(grid.lon, grid.lat)
    cand = (
        sphere_xyz(lon2d.ravel(), lat2d.ravel()) @ center >= np.cos(theta_lim)
    ).reshape(grid.shape)

    proposal = np.full(grid.shape, np.nan)
    if not cand.any():
        return proposal
    clon, clat = lon2d[cand], lat2d[cand]
    cx, cy = stereo_xy(clon, clat, lon0, lat0)
    cells = shapely.points(cx, cy)

    poly_in = shapely.Polygon(np.column_stack(stereo_xy(inner[:, 0], inner[:, 1], lon0, lat0)))
    inside = shapely.contains(poly_in, cells)
    values = np.full(clon.shape, np.nan)
    values[inside] = layer.resolution

    if outer is not None:
        poly_out = shapely.Polygon(
            np.column_stack(stereo_xy(outer[:, 0], outer[:, 1], lon0, lat0))
        )
        band = shapely.contains(poly_out, cells) & ~inside
        if band.any():
            in_lon, in_lat = _ring_endpoint_geo(cells[band], poly_in.exterior, lon0, lat0)
            out_lon, out_lat = _ring_endpoint_geo(cells[band], poly_out.exterior, lon0, lat0)
            d_in = haversine_km(clon[band], clat[band], in_lon, in_lat)
            d_out = haversine_km(clon[band], clat[band], out_lon, out_lat)
            # fmesh: the ramp targets the field where the band exits the outer ring
            jj = _nearest_index(grid.lat, out_lat)
            ii = _nearest_index(grid.lon, out_lon)
            base = grid.values[jj, ii]
            values[band] = layer.resolution + (base - layer.resolution) * (
                d_in / np.maximum(d_in + d_out, 1e-12)
            )
    else:
        near = ~inside
        if near.any():
            in_lon, in_lat = _ring_endpoint_geo(cells[near], poly_in.exterior, lon0, lat0)
            d_in = haversine_km(clon[near], clat[near], in_lon, in_lat)
            band_vals = np.full(d_in.shape, np.nan)
            in_band = d_in < layer.transition_km
            if in_band.any():
                base = grid.values[cand][near][in_band]  # local surrounding field
                band_vals[in_band] = layer.resolution + (base - layer.resolution) * (
                    d_in[in_band] / layer.transition_km
                )
            values[near] = band_vals

    proposal[cand] = values
    return proposal
