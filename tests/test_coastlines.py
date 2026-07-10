from pathlib import Path

import numpy as np
import pytest

from fmesh2.data import coastlines

FIXTURE = Path(__file__).parent / "fixtures" / "mini_coast.geojson"


def test_read_polylines():
    lines = coastlines.read_polylines(FIXTURE)
    # LineString(3 pts) + two MultiLineString parts; Point ignored
    assert [len(l) for l in lines] == [3, 2, 2]


def test_segments_and_dateline():
    buf = coastlines.coastline_segments(FIXTURE)
    # 2 + 1 + 1 segments, two vertices each
    assert buf["pos"].shape == (8, 3)
    assert buf["pos2"].shape == (8, 3)
    lon_plane = np.degrees(buf["pos2"][:, 0])
    # crossing segment (179 -> -179) becomes (179 -> 181)
    assert lon_plane.max() == pytest.approx(181.0)
    assert (np.abs(np.diff(lon_plane.reshape(-1, 2), axis=1)) < 180).all()
    # sphere positions are unit vectors
    np.testing.assert_allclose(np.linalg.norm(buf["pos"], axis=1), 1.0, atol=1e-6)


def test_cached_source_no_network(tmp_path, monkeypatch):
    monkeypatch.setenv("FMESH2_CACHE", str(tmp_path))
    cached = tmp_path / "ne_10m_coastline.geojson"
    cached.write_text(FIXTURE.read_text())

    def boom(*a, **k):
        raise AssertionError("network access attempted despite cache")

    monkeypatch.setattr(coastlines.urllib.request, "urlretrieve", boom)
    assert coastlines.default_coastline_path() == cached
    buf = coastlines.coastline_segments()
    assert buf["pos"].shape[0] == 8


def test_offline_without_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FMESH2_CACHE", str(tmp_path))

    def offline(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(coastlines.urllib.request, "urlretrieve", offline)
    with pytest.raises(RuntimeError, match="download it manually"):
        coastlines.default_coastline_path()
    with pytest.raises(FileNotFoundError):
        coastlines.default_coastline_path(download=False)
