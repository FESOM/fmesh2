"""Tests for M4 node editing: ops safeguards, journal undo/redo, replay hash."""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from fmesh2.core.io_fesom import read_fesom, write_fesom
from fmesh2.core.mesh import Mesh
from fmesh2.core.topology import SphereKDTree, boundary_nodes
from fmesh2.data.topo import Topo
from fmesh2.edit.journal import Journal, apply_journal, mesh_hash
from fmesh2.edit.ops import (
    EditError,
    move_nodes,
    patch_roughness_km,
    smooth_patch,
)


def grid_mesh(n=9, depth_value=100.0) -> Mesh:
    """Structured n x n triangulated patch near the equator (lon/lat 0..n-1)."""
    ii, jj = np.meshgrid(np.arange(n, dtype=np.float64), np.arange(n, dtype=np.float64))
    lon, lat = ii.ravel(), jj.ravel()
    tris = []
    for j in range(n - 1):
        for i in range(n - 1):
            k = j * n + i
            tris.append([k, k + 1, k + n + 1])
            tris.append([k, k + n + 1, k + n])
    depth = np.full(n * n, depth_value)
    return Mesh(lon=lon, lat=lat, tri=np.asarray(tris, dtype=np.int32), depth=depth)


def hex_mesh() -> Mesh:
    """Six-triangle fan around a center node at (0, 0), ring radius 1 deg."""
    ang = np.deg2rad(np.arange(6) * 60.0)
    lon = np.concatenate([[0.0], np.cos(ang)])
    lat = np.concatenate([[0.0], np.sin(ang)])
    tri = np.array([[0, 1 + i, 1 + (i + 1) % 6] for i in range(6)], dtype=np.int32)
    return Mesh(lon=lon, lat=lat, tri=tri, depth=np.full(7, 50.0))


def copy_mesh(mesh: Mesh) -> Mesh:
    return Mesh(
        lon=mesh.lon.copy(),
        lat=mesh.lat.copy(),
        tri=mesh.tri.copy(),
        depth=None if mesh.depth is None else mesh.depth.copy(),
        coast_flag=None if mesh.coast_flag is None else mesh.coast_flag.copy(),
        levels=None if mesh.levels is None else mesh.levels.copy(),
    )


def flat_topo(tmp_path, elevation: float) -> Topo:
    """Constant-elevation global topo (bilinear anywhere == elevation)."""
    lon = np.array([-180.0, -90.0, 0.0, 90.0])
    lat = np.array([-60.0, -20.0, 20.0, 60.0])
    ds = xr.Dataset(
        {"topo": (("lat", "lon"), np.full((4, 4), elevation))},
        coords={"lat": lat, "lon": lon},
    )
    path = tmp_path / f"flat_{elevation}.nc"
    ds.to_netcdf(path)
    return Topo(path)


# ------------------------------------------------------------------ move ops


def test_move_apply_and_undo_identity(tmp_path):
    mesh = grid_mesh()
    h0 = mesh_hash(mesh)
    journal = Journal(tmp_path / "edits.jsonl", mesh)

    ids = np.array([30, 31, 40])
    target = np.column_stack([mesh.lon[ids] + 0.2, mesh.lat[ids] - 0.1])
    op = move_nodes(mesh, ids, target)
    journal.record(op)

    assert np.array_equal(mesh.lon[ids], target[:, 0])
    assert np.array_equal(mesh.lat[ids], target[:, 1])
    assert mesh_hash(mesh) != h0
    assert op.report["n_nodes"] == 3
    assert op.report["max_displacement_km"] > 0

    assert journal.undo(mesh) is op
    assert mesh_hash(mesh) == h0  # bit-exact identity


def test_move_rejects_inversion_and_leaves_mesh_untouched():
    mesh = hex_mesh()
    h0 = mesh_hash(mesh)
    with pytest.raises(EditError) as exc:
        move_nodes(mesh, [0], [[3.0, 0.0]])  # far outside the ring
    assert exc.value.element_ids.size > 0
    assert set(exc.value.element_ids.tolist()) <= set(range(6))
    assert mesh_hash(mesh) == h0


def test_move_inside_hexagon_is_allowed():
    mesh = hex_mesh()
    op = move_nodes(mesh, [0], [[0.5, 0.0]])
    assert op.kind == "move"
    assert op.report["n_elements"] == 6
    # ~0.5 deg along the equator
    assert op.report["max_displacement_km"] == pytest.approx(55.6, rel=0.01)


def test_move_validation_errors():
    mesh = grid_mesh()
    with pytest.raises(EditError, match="no nodes"):
        move_nodes(mesh, [], np.empty((0, 2)))
    with pytest.raises(EditError, match="duplicate"):
        move_nodes(mesh, [3, 3], [[0.0, 0.0], [0.0, 0.0]])
    with pytest.raises(EditError, match="out of range"):
        move_nodes(mesh, [999], [[0.0, 0.0]])
    with pytest.raises(EditError, match="shape"):
        move_nodes(mesh, [3], [[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(EditError, match="finite"):
        move_nodes(mesh, [30], [[np.nan, 0.0]])
    with pytest.raises(EditError, match="latitude"):
        move_nodes(mesh, [30], [[0.0, 91.0]])


def test_move_resamples_depth_from_topo(tmp_path):
    mesh = grid_mesh(depth_value=100.0)  # positive-down host
    topo = flat_topo(tmp_path, -1000.0)
    op = move_nodes(mesh, [30], [[3.5, 3.5]], topo=topo)
    assert mesh.depth[30] == pytest.approx(1000.0)
    assert op.before_depth[0] == 100.0
    assert op.after_depth[0] == pytest.approx(1000.0)

    neg = grid_mesh(depth_value=-100.0)  # negative-down host (core2 style)
    move_nodes(neg, [30], [[3.5, 3.5]], topo=topo)
    assert neg.depth[30] == pytest.approx(-1000.0)

    shallow = grid_mesh(depth_value=100.0)
    move_nodes(shallow, [30], [[3.5, 3.5]], topo=flat_topo(tmp_path, -3.0),
               minimum_depth=5.0)
    assert shallow.depth[30] == 5.0  # clamped

    # without topo depths are untouched and the op stores none
    plain = grid_mesh(depth_value=100.0)
    op = move_nodes(plain, [30], [[3.5, 3.5]])
    assert plain.depth[30] == 100.0
    assert op.before_depth is None and op.after_depth is None


def test_move_depth_undo_restores_bits(tmp_path):
    mesh = grid_mesh(depth_value=100.0)
    h0 = mesh_hash(mesh)
    journal = Journal(tmp_path / "edits.jsonl", mesh)
    journal.record(move_nodes(mesh, [30], [[3.5, 3.5]], topo=flat_topo(tmp_path, -1000.0)))
    assert mesh.depth[30] == pytest.approx(1000.0)
    journal.undo(mesh)
    assert mesh_hash(mesh) == h0


# ------------------------------------------------------------------- smooth


def test_smooth_reduces_roughness_and_anchors_boundary():
    mesh = grid_mesh(n=11)
    interior = np.flatnonzero(~boundary_nodes(mesh.tri, mesh.n_nodes))
    rng = np.random.default_rng(42)
    mesh.lon[interior] += rng.uniform(-0.3, 0.3, interior.size)
    mesh.lat[interior] += rng.uniform(-0.3, 0.3, interior.size)

    boundary_lon = mesh.lon[boundary_nodes(mesh.tri, mesh.n_nodes)].copy()
    rough_before = patch_roughness_km(mesh, interior)

    ids = np.arange(mesh.n_nodes)  # select everything: boundary must anchor
    op = smooth_patch(mesh, ids, iterations=3, strength=0.5)

    assert op.kind == "smooth"
    assert set(op.ids.tolist()) == set(interior.tolist())  # boundary excluded
    assert np.array_equal(
        mesh.lon[boundary_nodes(mesh.tri, mesh.n_nodes)], boundary_lon
    )
    assert patch_roughness_km(mesh, interior) < 0.5 * rough_before
    assert op.params["iterations"] == 3


def test_smooth_validation_errors():
    mesh = grid_mesh()
    with pytest.raises(EditError, match="strength"):
        smooth_patch(mesh, [30], strength=0.0)
    with pytest.raises(EditError, match="strength"):
        smooth_patch(mesh, [30], strength=1.5)
    with pytest.raises(EditError, match="iterations"):
        smooth_patch(mesh, [30], iterations=0)
    corner = np.array([0])  # grid corner is a boundary node
    with pytest.raises(EditError, match="interior"):
        smooth_patch(mesh, corner)


def test_smooth_undo_identity(tmp_path):
    mesh = grid_mesh(n=7)
    interior = np.flatnonzero(~boundary_nodes(mesh.tri, mesh.n_nodes))
    rng = np.random.default_rng(1)
    mesh.lon[interior] += rng.uniform(-0.2, 0.2, interior.size)
    h0 = mesh_hash(mesh)
    journal = Journal(tmp_path / "edits.jsonl", mesh)
    journal.record(smooth_patch(mesh, interior, iterations=2, strength=0.7))
    assert mesh_hash(mesh) != h0
    journal.undo(mesh)
    assert mesh_hash(mesh) == h0


# ------------------------------------------------------------------ journal


def test_journal_undo_redo_stacks(tmp_path):
    mesh = grid_mesh()
    journal = Journal(tmp_path / "edits.jsonl", mesh)
    assert journal.undo(mesh) is None
    assert journal.redo(mesh) is None
    assert not journal.can_undo and not journal.can_redo

    journal.record(move_nodes(mesh, [30], [[3.3, 3.3]]))
    journal.record(move_nodes(mesh, [40], [[4.4, 4.4]]))
    h_both = mesh_hash(mesh)

    journal.undo(mesh)
    assert journal.can_redo
    journal.redo(mesh)
    assert mesh_hash(mesh) == h_both

    # a new op after an undo invalidates the redo stack
    journal.undo(mesh)
    journal.record(move_nodes(mesh, [50], [[5.5, 5.5]]))
    assert not journal.can_redo
    assert len(journal.ops) == 2


def test_journal_reload_from_disk(tmp_path):
    path = tmp_path / "edits.jsonl"
    mesh = grid_mesh()
    live = Journal(path, mesh)
    live.record(move_nodes(mesh, [30], [[3.3, 3.3]]))
    live.record(move_nodes(mesh, [40], [[4.4, 4.4]]))
    h_both = mesh_hash(mesh)
    live.undo(mesh)

    reloaded = Journal(path)
    assert reloaded.base_hash == live.base_hash
    assert len(reloaded.ops) == 1
    assert reloaded.can_redo

    # redo keeps working on the reloaded journal and restores the state
    reloaded.redo(mesh)
    assert mesh_hash(mesh) == h_both
    assert len(reloaded.ops) == 2


def test_journal_corrupt_and_unknown_records(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"kind": "base", "hash": "x"}\nnot json\n')
    with pytest.raises(EditError, match="corrupt"):
        Journal(path)
    path.write_text('{"kind": "teleport"}\n')
    with pytest.raises(EditError, match="unknown record kind"):
        Journal(path)
    path.write_text('{"kind": "undo"}\n')
    with pytest.raises(EditError, match="empty stack"):
        Journal(path)


def test_apply_journal_wrong_base_rejected(tmp_path):
    mesh = grid_mesh()
    journal = Journal(tmp_path / "edits.jsonl", mesh)
    journal.record(move_nodes(mesh, [30], [[3.3, 3.3]]))

    other = grid_mesh()
    other.lon[0] += 1.0  # not the base mesh anymore
    with pytest.raises(EditError, match="different base mesh"):
        apply_journal(other, journal)
    with pytest.raises(EditError, match="not found"):
        apply_journal(other, tmp_path / "missing.jsonl")


def test_apply_journal_replay_reproduces_hash(tmp_path):
    path = tmp_path / "edits.jsonl"
    live = grid_mesh(n=11)
    interior = np.flatnonzero(~boundary_nodes(live.tri, live.n_nodes))
    pristine = copy_mesh(live)

    journal = Journal(path, live)
    # small relative offsets (no inversion); long decimals exercise
    # bit-exact JSON round-tripping
    journal.record(
        move_nodes(
            live,
            [30],
            [[np.nextafter(live.lon[30] + 0.05, 99.0), live.lat[30] + 0.0123456789012345]],
        )
    )
    journal.record(smooth_patch(live, interior, iterations=2, strength=0.6))
    journal.record(move_nodes(live, [40], [[live.lon[40] + 0.25, live.lat[40] + 0.25]]))
    journal.undo(live)
    journal.record(  # kills the redo
        move_nodes(live, [50], [[live.lon[50] + 0.125, live.lat[50] + 0.125]])
    )

    replayed = apply_journal(copy_mesh(pristine), path)
    assert mesh_hash(replayed) == mesh_hash(live)
    assert not np.array_equal(replayed.lon, pristine.lon)  # really edited


# ------------------------------------------------------------ core2 workflow


def test_core2_edit_replay_and_io_roundtrip(core2_dir, tmp_path):
    mesh = read_fesom(core2_dir)
    journal = Journal(tmp_path / "edits.jsonl", mesh)

    tree = SphereKDTree(mesh.lon, mesh.lat)
    patch = np.sort(tree.query_radius(-30.0, 30.0, 400.0)[0])  # mid-Atlantic
    assert patch.size >= 10

    ids = patch[:3]
    target = np.column_stack([mesh.lon[ids] + 0.01, mesh.lat[ids] - 0.01])
    journal.record(move_nodes(mesh, ids, target))
    journal.record(smooth_patch(mesh, patch, iterations=2, strength=0.5))
    n4, n5 = patch[4], patch[5]
    journal.record(
        move_nodes(mesh, [n4], [[mesh.lon[n4] + 0.0123456789012345, mesh.lat[n4]]])
    )
    journal.undo(mesh)
    journal.record(
        move_nodes(mesh, [n5], [[mesh.lon[n5] - 0.01, mesh.lat[n5] + 0.01]])
    )

    # replay onto a pristine re-read reproduces the live hash bit-exact
    replayed = apply_journal(read_fesom(core2_dir), tmp_path / "edits.jsonl")
    assert mesh_hash(replayed) == mesh_hash(mesh)

    # and the hash survives the FESOM ASCII round trip (bit-exact I/O)
    out = tmp_path / "edited_mesh"
    write_fesom(mesh, out)
    assert mesh_hash(read_fesom(out)) == mesh_hash(mesh)
