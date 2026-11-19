"""Refinement regions from a file: region.geojson written by every
refinement, read back by the viewer's region-file field, single ring only;
the seam headroom (1.0 with a transition width, 1.2 without) and the
smooth transition sizing (no rays from per-seam-edge ramps).

Synthetic mesh (a jittered ~1 degree Delaunay grid): no reference data.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest
import shapely
from fastapi.testclient import TestClient
from scipy.spatial import Delaunay

from fmesh2.core.io_fesom import write_fesom
from fmesh2.core.projection import stereo_xy
from fmesh2.core.mesh import Mesh
from fmesh2.edit.cavity import extract_cavity
from fmesh2.edit.local_remesh import (
    RING_HEADROOM,
    SEAM_FLOOR,
    WIDE_HEADROOM,
    seam_headroom,
    seam_hfun_at,
)
from fmesh2.io.refine_region import RegionFileError, read_refine_region, write_refine_region
from fmesh2.server.app import create_app
from fmesh2.server.state import AppState

FIXTURE_COAST = Path(__file__).parent / "fixtures" / "mini_coast.geojson"
SQUARE = [[15.0, 15.0], [25.0, 15.0], [25.0, 25.0], [15.0, 25.0]]


def grid_mesh() -> Mesh:
    rng = np.random.default_rng(0)
    g = np.arange(0.0, 40.01, 1.0)
    lon, lat = (a.ravel() for a in np.meshgrid(g, g))
    lon = lon + rng.uniform(-0.05, 0.05, lon.size)
    lat = lat + rng.uniform(-0.05, 0.05, lat.size)
    tri = Delaunay(np.column_stack([lon, lat])).simplices.astype(np.int32)
    return Mesh(lon=lon, lat=lat, tri=tri, depth=np.full(lon.size, -3000.0),
                coast_flag=np.zeros(lon.size, np.int8), levels=np.array([0.0, -10.0, -6000.0]))


def fine_mesh() -> Mesh:
    """~27 km jittered grid (seam spacing like a real 30 km mesh)."""
    rng = np.random.default_rng(0)
    g = np.arange(5.0, 35.01, 0.25)
    lon, lat = (a.ravel() for a in np.meshgrid(g, g))
    lon = lon + rng.uniform(-0.03, 0.03, lon.size)
    lat = lat + rng.uniform(-0.03, 0.03, lat.size)
    tri = Delaunay(np.column_stack([lon, lat])).simplices.astype(np.int32)
    return Mesh(lon=lon, lat=lat, tri=tri, depth=np.full(lon.size, -3000.0))


FINE_SQUARE = [[12.0, 12.0], [28.0, 12.0], [28.0, 28.0], [12.0, 28.0]]


def partly_fine_mesh() -> Mesh:
    """fine_mesh with a ~5.5 km patch where the square's east side runs,
    like a seam through an area refined before."""
    base = fine_mesh()
    centre, radius = np.array([28.0, 20.0]), 2.5
    keep = np.hypot(base.lon - centre[0], base.lat - centre[1]) > radius
    f = np.arange(-radius, radius + 1e-9, 0.05)
    flon, flat = (a.ravel() + c for a, c in zip(np.meshgrid(f, f), centre))
    disk = np.hypot(flon - centre[0], flat - centre[1]) <= radius
    lon = np.concatenate([base.lon[keep], flon[disk]])
    lat = np.concatenate([base.lat[keep], flat[disk]])
    tri = Delaunay(np.column_stack([lon, lat])).simplices.astype(np.int32)
    return Mesh(lon=lon, lat=lat, tri=tri, depth=np.full(lon.size, -3000.0))


def feature(ring, properties=None):
    return {"type": "Feature", "properties": properties or {},
            "geometry": {"type": "Polygon", "coordinates": [ring + ring[:1]]}}


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(AppState(cache_dir=tmp_path / "cache", coastline_source=FIXTURE_COAST)))


def test_region_file_roundtrip(tmp_path):
    settings = {"target_km": 5.0, "transition_km": 150.0, "topo": "/data/RTopo.nc",
                "minimum_depth": 30.0, "source_mesh": "/meshes/dars2"}
    path = write_refine_region(tmp_path / "region.geojson", SQUARE, settings)
    ring, got = read_refine_region(path)
    assert np.array_equal(ring[0], ring[-1])                 # closed
    assert np.array_equal(ring[:-1], np.asarray(SQUARE))     # same vertices
    assert got == settings
    gj = json.loads(path.read_text())
    assert len(gj["features"]) == 1 and gj["features"][0]["geometry"]["type"] == "Polygon"


def test_plain_region_file_has_no_settings(tmp_path):
    path = tmp_path / "drawn.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [feature(SQUARE)]}))
    ring, settings = read_refine_region(path)
    assert len(ring) == 5 and settings == {}


def test_two_ring_file_is_rejected(tmp_path):
    outer = [[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]]
    path = tmp_path / "two.geojson"
    path.write_text(json.dumps({"type": "FeatureCollection", "features": [
        feature(SQUARE, {"role": "inner"}), feature(outer, {"role": "outer"})]}))
    with pytest.raises(RegionFileError, match="two rings"):
        read_refine_region(path)


def test_region_endpoint(client, tmp_path):
    path = write_refine_region(tmp_path / "r.geojson", SQUARE, {"target_km": 25.0})
    r = client.get("/api/region", params={"path": str(path)})
    assert r.status_code == 200
    assert r.json()["ring"][:4] == SQUARE and r.json()["settings"]["target_km"] == 25.0
    assert client.get("/api/region", params={"path": str(tmp_path / "nope.geojson")}).status_code == 400
    two = tmp_path / "two.geojson"
    two.write_text(json.dumps({"type": "FeatureCollection", "features": [
        feature(SQUARE), feature([[10.0, 10.0], [30.0, 10.0], [30.0, 30.0], [10.0, 30.0]])]}))
    r = client.get("/api/region", params={"path": str(two)})
    assert r.status_code == 400 and "two rings" in r.json()["detail"]


def test_seam_headroom():
    assert seam_headroom(None) == RING_HEADROOM == 1.2
    assert seam_headroom(150.0) == WIDE_HEADROOM == 1.0
    cav = extract_cavity(grid_mesh(), SQUARE)
    seam = np.flatnonzero(~cav.edge_is_coast)
    k = seam[0]
    a, b = cav.ring_edges[k]
    mid = 0.5 * (cav.ring_xy[a] + cav.ring_xy[b])        # on the seam: d = 0
    length = cav.edge_len_km[k]
    h_auto, d = seam_hfun_at(mid[None], cav, 10.0)
    h_wide, _ = seam_hfun_at(mid[None], cav, 10.0, transition_km=300.0)
    assert d[0] < 1e-9
    assert np.isclose(h_auto[0], 1.2 * length) and np.isclose(h_wide[0], length)


def _wait(client, job_id, timeout=300.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        snap = client.get(f"/api/jobs/{job_id}").json()
        if snap["status"] in ("done", "error", "cancelled"):
            return snap
        time.sleep(0.25)
    raise TimeoutError(job_id)


def test_refine_writes_region_file_and_it_repeats(client, tmp_path):
    write_fesom(grid_mesh(), tmp_path / "mesh")
    mesh_id = client.post("/api/meshes/open", json={"path": str(tmp_path / "mesh")}).json()["id"]
    body = {"polygon": SQUARE, "target_km": 40.0, "transition_km": 200.0,
            "out_dir": str(tmp_path / "refined")}
    snap = _wait(client, client.post(f"/api/mesh/{mesh_id}/refine", json=body).json()["job_id"])
    assert snap["status"] == "done", snap
    region_file = Path(snap["result"]["region_file"])
    assert region_file == tmp_path / "refined" / "region.geojson"

    loaded = client.get("/api/region", params={"path": str(region_file)}).json()
    assert loaded["ring"][:4] == SQUARE
    s = loaded["settings"]
    assert (s["target_km"], s["transition_km"], s["topo"]) == (40.0, 200.0, None)
    assert s["source_mesh"] == str((tmp_path / "mesh").resolve())

    # the saved region refines the same area again, with the same settings
    again = {"polygon": loaded["ring"][:-1], "target_km": s["target_km"],
             "transition_km": s["transition_km"], "out_dir": str(tmp_path / "again")}
    snap2 = _wait(client, client.post(f"/api/mesh/{mesh_id}/refine", json=again).json()["job_id"])
    assert snap2["status"] == "done", snap2
    assert snap2["result"]["n_nodes"] == snap["result"]["n_nodes"]
    assert json.loads((tmp_path / "again" / "region.geojson").read_text())["features"][0]["geometry"] == \
        json.loads(region_file.read_text())["features"][0]["geometry"]


def test_cavity_keeps_the_drawn_polygon():
    mesh = grid_mesh()
    cav = extract_cavity(mesh, SQUARE)
    assert cav.polygon_xy is not None and np.array_equal(cav.polygon_xy[0], cav.polygon_xy[-1])
    assert shapely.Polygon(cav.polygon_xy).area > 0
    assert extract_cavity(mesh, elem_mask=cav.elem_mask).polygon_xy is None


def _max_jump_along(cav, polygon_xy, dist_km, transition_km):
    """Largest relative sizing jump between neighbouring points 0.5 km apart
    on a path `dist_km` inside the polygon and parallel to it."""
    poly = shapely.Polygon(polygon_xy)
    path = np.asarray(shapely.segmentize(poly.buffer(-dist_km).exterior, 0.5).coords)
    h, _ = seam_hfun_at(path, cav, 5.0, transition_km)
    return float(np.max(np.abs(np.diff(h)) / h[:-1]))


def test_transition_has_no_rays_away_from_the_seam():
    """Per-seam-edge ramps jump where the nearest seam edge changes (rays);
    the smooth transition does not, beyond the seam-protection zone."""
    cav = extract_cavity(fine_mesh(), FINE_SQUARE)
    per_edge = dataclasses.replace(cav, polygon_xy=None)  # sizing without the polygon
    for dist in (150.0, 220.0):
        assert _max_jump_along(cav, cav.polygon_xy, dist, 300.0) < 0.08
        assert _max_jump_along(per_edge, cav.polygon_xy, dist, 300.0) > 0.25


def test_sizing_next_to_the_seam_is_unchanged():
    """At the seam: exactly the frozen edge length; within SEAM_FLOOR[0] edge
    lengths: never below the per-edge ramp; without a transition width:
    identical to the per-edge ramp everywhere."""
    cav = extract_cavity(fine_mesh(), FINE_SQUARE)
    per_edge = dataclasses.replace(cav, polygon_xy=None)
    seam = np.flatnonzero(~cav.edge_is_coast)
    a, b = cav.ring_edges[seam].T
    mids = 0.5 * (cav.ring_xy[a] + cav.ring_xy[b])
    h, _ = seam_hfun_at(mids, cav, 5.0, 300.0)
    assert np.allclose(h, cav.edge_len_km[seam])

    rng = np.random.default_rng(1)
    lo, hi = cav.ring_xy.min(axis=0), cav.ring_xy.max(axis=0)
    pts = rng.uniform(lo, hi, size=(20000, 2))
    h_new, d = seam_hfun_at(pts, cav, 5.0, 300.0)
    h_edge, _ = seam_hfun_at(pts, per_edge, 5.0, 300.0)
    segs = shapely.linestrings(cav.ring_xy[cav.ring_edges[seam]])
    _, nearest = shapely.STRtree(segs).query_nearest(shapely.points(pts), all_matches=False)
    near = d < SEAM_FLOOR[0] * cav.edge_len_km[seam][nearest]  # per point: its nearest seam edge
    assert near.any() and (h_new[near] >= h_edge[near] - 1e-9).all()

    auto, _ = seam_hfun_at(pts, cav, 5.0, None)
    auto_edge, _ = seam_hfun_at(pts, per_edge, 5.0, None)
    assert np.array_equal(auto, auto_edge)


def test_no_coarse_band_where_the_seam_is_already_fine():
    """A seam running through an already fine area: the transition next to
    that part must not start coarser than the seam there (averaging along
    the seam would pull it towards the coarse rest of the seam)."""
    cav = extract_cavity(partly_fine_mesh(), FINE_SQUARE)
    x0, y0 = stereo_xy(28.0, 20.0, cav.lon0, cav.lat0)      # on the fine seam part
    cx, cy = shapely.Polygon(cav.polygon_xy).centroid.coords[0]
    u = np.array([cx - x0, cy - y0]) / np.hypot(cx - x0, cy - y0)
    d = np.arange(0.0, 161.0, 2.0)
    line = np.array([x0, y0]) + d[:, None] * u            # straight inwards
    h, _ = seam_hfun_at(line, cav, 5.0, 300.0)
    fine_seam = float(np.median(seam_hfun_at(line[:3], cav, 5.0, 300.0)[0]))
    assert fine_seam < 8.0                                   # the seam there is fine
    assert h[d >= 20].max() <= 1.1 * fine_seam               # no coarser band inside it
