import hashlib
from pathlib import Path

import pytest

from fmesh2.cli import main
from fmesh2.core.io_fesom import read_fesom
from fmesh2.project import BaseLayer, GridSettings, Project

FIXTURES = Path(__file__).parent / "fixtures"
TOPO = str((FIXTURES / "topo_synthetic.nc").resolve())


def make_project(tmp_path) -> Path:
    proj = Project(
        name="cli",
        topo=TOPO,
        grid=GridSettings(n_lat=90, n_lon=180),
        layers=[BaseLayer(type="base", resolution=500)],
    )
    return proj.save(tmp_path / "proj")


def mesh_hash(out_dir: Path) -> str:
    h = hashlib.sha256()
    for name in ("nod2d.out", "elem2d.out", "aux3d.out"):
        h.update((out_dir / name).read_bytes())
    return h.hexdigest()


@pytest.mark.timeout(300)
def test_cli_build_and_reproducibility(tmp_path, capsys):
    path = make_project(tmp_path)
    out_dir = tmp_path / "proj" / "out"

    assert main(["build", str(path)]) == 0
    mesh = read_fesom(out_dir)
    assert mesh.n_nodes > 500
    assert mesh.depth.min() >= 30.0
    first = mesh_hash(out_dir)
    assert any((tmp_path / "proj" / "cache").glob("jigsaw_*.npz"))  # anchor cached

    # reproducibility contract: postprocess from the cached raw mesh is
    # bit-deterministic — two runs give identical FESOM files
    assert main(["build", str(path), "--only-postprocess"]) == 0
    assert "cached" in capsys.readouterr().out
    assert mesh_hash(out_dir) == first
    assert main(["build", str(path), "--only-postprocess"]) == 0
    assert mesh_hash(out_dir) == first


def test_cli_build_exit_codes(tmp_path):
    assert main(["build", str(tmp_path / "nope")]) == 2  # unreadable project

    no_topo = Project(name="x", layers=[BaseLayer(type="base", resolution=500)])
    p = no_topo.save(tmp_path / "nt")
    assert main(["build", str(p)]) == 2  # no topo configured

    proj = Project(
        name="x",
        topo=str(tmp_path / "missing.nc"),
        grid=GridSettings(n_lat=45, n_lon=90),
        layers=[BaseLayer(type="base", resolution=1000)],
    )
    p = proj.save(tmp_path / "bt")
    assert main(["build", str(p)]) == 1  # build fails: topo file missing

    p2 = make_project(tmp_path)
    assert main(["build", str(p2), "--only-postprocess"]) == 1  # no cache yet
