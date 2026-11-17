"""`fmesh2 redepth`: topography re-sampling + smooth_seafloor.f90 smoothing,
for the nodes added by a refinement (local) or for every node (global).

Synthetic meshes only (no jigsaw, no reference data): the "refined" mesh is
a Delaunay re-triangulation that keeps the outer original nodes bit-exact,
drops the inner ones and adds a finer block — the node bookkeeping of a
local refinement (kept nodes first, new nodes appended).
"""

import json

import numpy as np
import pytest
import xarray as xr
from scipy.spatial import Delaunay

from fmesh2.cli import main
from fmesh2.core.io_fesom import read_fesom, write_fesom
from fmesh2.core.mesh import Mesh
from fmesh2.data.topo import Topo
from fmesh2.edit.redepth import (
    RedepthError,
    edge_steepness,
    match_original_nodes,
    neighbour_matrix,
    redepth_global,
    redepth_local,
    round_whole_metres,
    smooth_depths,
)

LEVELS = np.array([0.0, -10.0, -100.0, -6000.0])


def plane_depth(lon, lat):
    """Positive-down depth, linear in lon/lat (bilinear sampling is exact)."""
    return 1000.0 + 50.0 * np.asarray(lon) + 20.0 * np.asarray(lat)


def make_meshes(sign=-1.0):
    rng = np.random.default_rng(1)
    g = np.arange(0.0, 20.5, 1.0)
    lon, lat = (a.ravel() for a in np.meshgrid(g, g))
    lon = lon + rng.uniform(-0.05, 0.05, lon.size)  # no cocircular ties
    lat = lat + rng.uniform(-0.05, 0.05, lat.size)
    tri = Delaunay(np.column_stack([lon, lat])).simplices.astype(np.int32)
    original = Mesh(lon=lon, lat=lat, tri=tri, depth=sign * plane_depth(lon, lat),
                    coast_flag=np.zeros(lon.size, dtype=np.int8), levels=LEVELS)

    keep = ~((lon > 5.5) & (lon < 14.5) & (lat > 5.5) & (lat < 14.5))
    f = np.arange(6.0, 14.01, 0.25)
    nlon, nlat = (a.ravel() for a in np.meshgrid(f, f))
    nlon = nlon + rng.uniform(-0.02, 0.02, nlon.size)
    nlat = nlat + rng.uniform(-0.02, 0.02, nlat.size)
    rlon = np.concatenate([lon[keep], nlon])
    rlat = np.concatenate([lat[keep], nlat])
    rtri = Delaunay(np.column_stack([rlon, rlat])).simplices.astype(np.int32)
    rdepth = np.concatenate([original.depth[keep], np.full(nlon.size, sign * 1.0)])
    refined = Mesh(lon=rlon, lat=rlat, tri=rtri, depth=rdepth,
                   coast_flag=np.zeros(rlon.size, dtype=np.int8), levels=LEVELS)
    return original, refined, keep


def write_topo(tmp_path, values_fn, name="topo.nc", step=0.5) -> Topo:
    """Topography NetCDF with elevation = -values_fn(lon, lat)."""
    lon = np.arange(-2.0, 22.5, step)
    lat = np.arange(-2.0, 22.5, step)
    LON, LAT = np.meshgrid(lon, lat)
    path = tmp_path / name
    xr.Dataset({"topo": (("lat", "lon"), -values_fn(LON, LAT))},
               coords={"lat": lat, "lon": lon}).to_netcdf(path)
    return Topo(path)


def plane_topo(tmp_path, offset=0.0, name="topo.nc") -> Topo:
    return write_topo(tmp_path, lambda x, y: plane_depth(x, y) + offset, name)


def fortran_reference(h, tri, active, iterations, cw):
    """Line-by-line port of smooth_seafloor.f90: negative depths, maxloc
    ordering, in-place update — restricted to the `active` nodes."""
    n = h.size
    nbrs = [set() for _ in range(n)]
    for a, b, c in tri:
        nbrs[a].update((b, c))
        nbrs[b].update((a, c))
        nbrs[c].update((a, b))
    dep = -np.asarray(h, dtype=np.float64)
    for _ in range(iterations):
        dep_old = dep.copy()
        dep_old[~active] = -1e30
        for _ in range(int(active.sum())):
            m = int(np.argmax(dep_old))  # maxloc: shallowest, first on ties
            dep_old[m] = -1e30
            cnt = len(nbrs[m])
            dsum = sum(dep[k] for k in sorted(nbrs[m]))
            dsum += (cw * (cnt - 1.0) - 1.0) * dep[m]
            dep[m] = dsum / (cnt - 1 + (cnt - 1) * cw)
    return -dep


def test_match_finds_exactly_the_kept_nodes():
    original, refined, keep = make_meshes()
    idx = match_original_nodes(original, refined)
    n_kept = int(keep.sum())
    assert np.array_equal(idx[:n_kept], np.flatnonzero(keep))
    assert (idx[n_kept:] == -1).all()


def test_smoother_matches_fortran_on_a_subset():
    original, refined, _ = make_meshes()
    new = match_original_nodes(original, refined) < 0
    h = np.random.default_rng(7).uniform(30.0, 5000.0, refined.n_nodes)
    _, nbrs = neighbour_matrix(refined.tri, refined.n_nodes)
    for iters, cw in ((1, 2.0), (3, 1.0), (2, 3.5)):
        got = smooth_depths(h, nbrs, new, iters, cw)
        assert np.allclose(got, fortran_reference(h, refined.tri, new, iters, cw),
                           rtol=1e-12, atol=1e-9)
        assert np.array_equal(got[~new], h[~new])  # inactive nodes never written


def test_smoother_matches_fortran_on_the_whole_mesh():
    original, _, _ = make_meshes()
    h = np.random.default_rng(3).uniform(30.0, 5000.0, original.n_nodes)
    everyone = np.ones(original.n_nodes, dtype=bool)
    _, nbrs = neighbour_matrix(original.tri, original.n_nodes)
    got = smooth_depths(h, nbrs, everyone, 1, 2.0)
    assert np.allclose(got, fortran_reference(h, original.tri, everyone, 1, 2.0),
                       rtol=1e-12, atol=1e-9)


def test_local_old_nodes_bit_identical_and_sign_follows_mesh(tmp_path):
    for sign in (-1.0, 1.0):
        original, refined, keep = make_meshes(sign)
        res = redepth_local(original, refined, plane_topo(tmp_path, 300.0, f"t{sign}.nc"))
        old = ~res.is_new
        assert np.array_equal(res.mesh.depth[old], original.depth[keep])
        assert (np.sign(res.mesh.depth[res.is_new]) == sign).all()
        assert np.array_equal(res.mesh.lon, refined.lon)
        assert np.array_equal(res.mesh.tri, refined.tri)
        assert np.array_equal(res.mesh.levels, LEVELS)
        assert res.report["n_old_depth_differs_from_original"] == 0


def test_local_is_sample_then_smooth_on_new_nodes(tmp_path):
    """Local result == clamped topography on the new nodes, old depths
    elsewhere, then the Fortran sweep over the new nodes only."""
    original, refined, _ = make_meshes()
    topo = write_topo(tmp_path, lambda x, y: plane_depth(x, y) + 400 * np.sin(3 * x) * np.cos(2 * y))
    new = match_original_nodes(original, refined) < 0
    h = np.abs(refined.depth).astype(float)
    h[new] = np.clip(-topo.elevation(refined.lon[new], refined.lat[new]), 30.0, None)

    raw = redepth_local(original, refined, topo, iterations=0)
    expect = h.copy()
    expect[new] = round_whole_metres(h[new])
    assert np.array_equal(np.abs(raw.mesh.depth), expect)
    res = redepth_local(original, refined, topo, iterations=2, center_weight=2.0)
    ref = fortran_reference(h, refined.tri, new, 2, 2.0)
    ref[new] = round_whole_metres(ref[new])
    assert np.array_equal(np.abs(res.mesh.depth), ref)
    assert (res.mesh.depth[new] == np.round(res.mesh.depth[new])).all()  # whole metres


def test_global_is_sample_then_smooth_on_all_nodes(tmp_path):
    for sign in (-1.0, 1.0):
        _, refined, _ = make_meshes(sign)
        topo = write_topo(tmp_path, lambda x, y: plane_depth(x, y) + 300 * np.cos(4 * x),
                          name=f"g{sign}.nc")
        h = np.clip(-topo.elevation(refined.lon, refined.lat), 30.0, None)
        everyone = np.ones(refined.n_nodes, dtype=bool)

        raw = redepth_global(refined, topo, iterations=0)
        assert np.array_equal(np.abs(raw.mesh.depth), round_whole_metres(h))
        res = redepth_global(refined, topo)
        ref = round_whole_metres(fortran_reference(h, refined.tri, everyone, 1, 2.0))
        assert np.array_equal(np.abs(res.mesh.depth), ref)
        assert (np.sign(res.mesh.depth) == sign).all()
        assert np.array_equal(res.mesh.tri, refined.tri)
        assert res.report["mode"] == "global"


def test_clamp_land_and_limits(tmp_path):
    original, refined, _ = make_meshes()
    land = write_topo(tmp_path, lambda x, y: np.full_like(x, -25.0), "land.nc")  # +25 m
    res = redepth_global(refined, land, iterations=0)
    assert np.allclose(np.abs(res.mesh.depth), 30.0)
    assert res.report["n_on_topo_land"] == refined.n_nodes
    deep = write_topo(tmp_path, lambda x, y: np.full_like(x, 9000.0), "deep.nc")
    res = redepth_local(original, refined, deep, iterations=0)  # no cap by default
    assert np.allclose(np.abs(res.mesh.depth[res.is_new]), 9000.0)
    res = redepth_local(original, refined, deep, iterations=0, max_depth=6000.0)
    assert np.allclose(np.abs(res.mesh.depth[res.is_new]), 6000.0)


def test_rejects_bad_input(tmp_path):
    original, refined, _ = make_meshes()
    topo = plane_topo(tmp_path)
    with pytest.raises(RedepthError, match="no new nodes"):
        redepth_local(original, original, topo)
    shifted = Mesh(lon=original.lon + 0.3, lat=original.lat, tri=original.tri,
                   depth=original.depth)
    with pytest.raises(RedepthError, match="matches the original"):
        redepth_local(shifted, refined, topo)
    with pytest.raises(RedepthError, match="no depths"):
        redepth_global(Mesh(lon=original.lon, lat=original.lat, tri=original.tri), topo)
    with pytest.raises(RedepthError, match="depth range"):
        redepth_global(original, topo, min_depth=100.0, max_depth=50.0)
    with pytest.raises(RedepthError, match="center weight"):
        redepth_global(original, topo, center_weight=0.5)


def test_edge_steepness_definition():
    r = edge_steepness(np.array([100.0, 300.0, 300.0]), np.array([[0, 1], [1, 2]]))
    assert np.allclose(r, [0.5, 0.0])


def test_cli_both_modes(tmp_path, capsys):
    original, refined, keep = make_meshes()
    plane_topo(tmp_path, offset=200.0)
    topo = str(tmp_path / "topo.nc")
    odir, rdir = tmp_path / "orig", tmp_path / "refined"
    write_fesom(original, odir)
    write_fesom(refined, rdir)
    n_kept = int(keep.sum())

    out = tmp_path / "out_local"
    assert main(["redepth", str(rdir), str(out), "--original", str(odir), "--topo", topo]) == 0
    assert "old nodes unchanged" in capsys.readouterr().out
    got, back = read_fesom(out), read_fesom(rdir)
    assert np.array_equal(got.depth[:n_kept], back.depth[:n_kept])
    assert not np.array_equal(got.depth[n_kept:], back.depth[n_kept:])
    for name in ("nod2d.out", "elem2d.out"):  # byte-identical copies
        assert (out / name).read_bytes() == (rdir / name).read_bytes()
    rep = json.loads((out / "redepth_report.json").read_text())
    assert rep["mode"] == "local" and rep["n_new"] == refined.n_nodes - n_kept

    out = tmp_path / "out_global"
    assert main(["redepth", str(rdir), str(out), "--mode", "global", "--topo", topo,
                 "--iterations", "0"]) == 0
    assert "all" in capsys.readouterr().out
    expect = round_whole_metres(np.clip(plane_depth(refined.lon, refined.lat) + 200.0, 30.0, None))
    assert np.array_equal(np.abs(read_fesom(out).depth), expect)

    # argument checks -> exit code 2
    assert main(["redepth", str(rdir), str(out), "--topo", topo]) == 2          # local, no --original
    assert main(["redepth", str(rdir), str(out), "--mode", "global", "--topo", topo,
                 "--original", str(odir)]) == 2                                  # global + --original
    assert main(["redepth", str(rdir), str(rdir), "--mode", "global", "--topo", topo]) == 2
    assert main(["redepth", str(tmp_path / "nope"), str(out), "--mode", "global",
                 "--topo", topo]) == 2
