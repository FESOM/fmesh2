from pathlib import Path

import pytest
import yaml

from fmesh2.project import (
    DEFAULT_LEVELS,
    LAYER_TYPES,
    BaseLayer,
    BathyLayer,
    CoastLayer,
    FieldLayer,
    GridSettings,
    MercatorLayer,
    Project,
    ProjectError,
    RegionLayer,
)

FIXTURE = Path(__file__).parent / "fixtures" / "project_minimal"


def full_project() -> Project:
    return Project(
        name="full",
        topo="topo/rtopo.nc",
        grid=GridSettings(n_lat=360, n_lon=720),
        layers=[
            BaseLayer(type="base", resolution=300),
            MercatorLayer(
                type="mercator",
                resolution=300,
                northern_boundary=60,
                northern_lowest_resolution=100,
                southern_boundary=-60,
                southern_lowest_resolution=100,
            ),
            CoastLayer(type="coast", min_resolution=50, max_distance=300),
            BathyLayer(type="bathy", min_resolution=50),
            RegionLayer(
                type="region", path="regions/baffin.geojson", resolution=25, transition_km=100
            ),
            FieldLayer(type="field", path="fields/res.nc", variable="resolution", mode="replace"),
        ],
        minimum_depth=30,
        seeds=[(0.0, 0.0), (-30.0, 40.0)],
    )


# ------------------------------------------------------------------ fixture


def test_minimal_fixture_validates():
    proj = Project.load(FIXTURE)
    assert proj.name == "minimal"
    assert len(proj.layers) == 1
    assert proj.layers[0].type == "base"
    assert proj.layers[0].resolution == 300
    assert proj.project_dir == FIXTURE.resolve()


def test_load_accepts_file_or_dir():
    by_dir = Project.load(FIXTURE)
    by_file = Project.load(FIXTURE / "project.yaml")
    assert by_dir == by_file


# ----------------------------------------------------------------- defaults


def test_defaults():
    proj = Project(name="d")
    assert proj.levels == DEFAULT_LEVELS
    assert proj.minimum_depth == 30.0
    assert proj.grid.n_lat is None and proj.grid.n_lon is None
    assert proj.seeds == [(0.0, 0.0)]
    assert proj.output_dir == "out"
    assert proj.topo is None


def test_layer_defaults():
    layer = CoastLayer(type="coast", min_resolution=10, max_distance=100)
    assert layer.enabled is True
    assert layer.mode == "min"
    assert layer.min_length == 100.0
    assert layer.averaging == 50.0
    assert layer.source is None


# ---------------------------------------------------------------- roundtrip


def test_roundtrip_fidelity(tmp_path):
    original = full_project()
    saved = original.save(tmp_path / "proj")
    assert saved == tmp_path / "proj" / "project.yaml"
    loaded = Project.load(saved)
    assert loaded == original


def test_roundtrip_preserves_layer_order(tmp_path):
    proj = full_project()
    proj.save(tmp_path)
    raw = yaml.safe_load((tmp_path / "project.yaml").read_text())
    assert [l["type"] for l in raw["layers"]] == [
        "base", "mercator", "coast", "bathy", "region", "field",
    ]
    # reversed order survives a save/load cycle too
    proj.layers = list(reversed(proj.layers))
    proj.save(tmp_path)
    reloaded = Project.load(tmp_path)
    assert [l.type for l in reloaded.layers] == [
        "field", "region", "bathy", "coast", "mercator", "base",
    ]


# ------------------------------------------------------------------- errors


def test_unknown_layer_type_clear_error(tmp_path):
    (tmp_path / "project.yaml").write_text(
        "name: bad\nlayers:\n  - type: wavelet\n    resolution: 10\n"
    )
    with pytest.raises(ProjectError) as exc:
        Project.load(tmp_path)
    msg = str(exc.value)
    assert "wavelet" in msg
    for t in LAYER_TYPES:
        assert f"'{t}'" in msg


def test_missing_file_and_bad_yaml(tmp_path):
    with pytest.raises(ProjectError, match="not found"):
        Project.load(tmp_path / "nope")
    (tmp_path / "project.yaml").write_text("name: [unclosed\n")
    with pytest.raises(ProjectError, match="not valid YAML"):
        Project.load(tmp_path)
    (tmp_path / "project.yaml").write_text("- just\n- a list\n")
    with pytest.raises(ProjectError, match="mapping"):
        Project.load(tmp_path)


def test_unknown_top_level_key_rejected(tmp_path):
    (tmp_path / "project.yaml").write_text("name: x\nresolutoin: 300\n")
    with pytest.raises(ProjectError, match="resolutoin"):
        Project.load(tmp_path)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(minimum_depth=0),
        dict(minimum_depth=-5),
        dict(levels=[0.0]),
        dict(levels=[0.0, -10.0, -10.0]),
        dict(levels=[0.0, -10.0, -5.0]),
        dict(seeds=[]),
        dict(seeds=[(0.0, 95.0)]),
    ],
)
def test_invalid_project_values(kwargs):
    with pytest.raises(ValueError):
        Project(name="x", **kwargs)


def test_invalid_layer_values():
    with pytest.raises(ValueError):
        BaseLayer(type="base", resolution=0)
    with pytest.raises(ValueError, match="max_depth must exceed"):
        BathyLayer(type="bathy", min_resolution=10, min_depth=500, max_depth=100)
    with pytest.raises(ValueError, match="southern_boundary must be south"):
        MercatorLayer(
            type="mercator", resolution=100, southern_boundary=70, northern_boundary=60
        )
    with pytest.raises(ValueError):
        RegionLayer(type="region", path="r.geojson", resolution=10, transition_km=0)


# -------------------------------------------------------- path resolution


def test_relative_paths_resolve_against_project_dir(tmp_path):
    proj = full_project()
    proj.save(tmp_path)
    assert proj.topo_path == (tmp_path / "topo" / "rtopo.nc").resolve()
    assert proj.resolve_path("regions/baffin.geojson") == (
        tmp_path / "regions" / "baffin.geojson"
    ).resolve()
    assert proj.out_dir == (tmp_path / "out").resolve()
    assert proj.cache_dir == (tmp_path / "cache").resolve()


def test_absolute_path_passes_through(tmp_path):
    proj = Project(name="x", topo=str(tmp_path / "abs.nc"))
    assert proj.topo_path == tmp_path / "abs.nc"  # no project dir needed


def test_relative_path_without_dir_raises():
    proj = Project(name="x", topo="topo.nc")
    with pytest.raises(ProjectError, match="not bound"):
        _ = proj.topo_path
    with pytest.raises(ProjectError, match="pass a path"):
        proj.save()


def test_ensure_layout(tmp_path):
    proj = Project(name="x", output_dir="meshes")
    proj.save(tmp_path)
    proj.ensure_layout()
    for sub in ("regions", "fields", "cache", "meshes"):
        assert (tmp_path / sub).is_dir()
