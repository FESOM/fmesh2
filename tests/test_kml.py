import json
from pathlib import Path

import numpy as np
import pytest

from fmesh2.io.kml import KMLError, kml_to_geojson, order_rings, read_kml_rings

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_parses_to_two_closed_rings():
    rings = read_kml_rings(FIXTURES / "region.kml")
    assert [name for name, _ in rings] == ["Test_external", "Test_internal"]
    for _, ring in rings:
        assert np.array_equal(ring[0], ring[-1])  # closed
    outer = rings[0][1]
    assert outer.shape == (5, 2)
    np.testing.assert_array_equal(outer[0], [-34.0, 16.0])


def test_order_rings_by_containment():
    rings = read_kml_rings(FIXTURES / "region.kml")
    # file order is external first; detection must fix it either way round
    inner, outer = order_rings(rings[0][1], rings[1][1])
    assert inner[:, 0].min() == -32 and outer[:, 0].min() == -34
    inner2, outer2 = order_rings(rings[1][1], rings[0][1])
    np.testing.assert_array_equal(inner, inner2)
    np.testing.assert_array_equal(outer, outer2)


def test_order_rings_rejects_disjoint():
    a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    b = a + 10.0
    with pytest.raises(KMLError, match="nest"):
        order_rings(a, b)


def test_kml_to_geojson_roundtrip(tmp_path):
    out = kml_to_geojson(FIXTURES / "region.kml", tmp_path / "regions" / "test.geojson")
    gj = json.loads(out.read_text())
    roles = {f["properties"]["role"]: f for f in gj["features"]}
    assert set(roles) == {"inner", "outer"}
    inner_ring = np.asarray(roles["inner"]["geometry"]["coordinates"][0])
    assert inner_ring[:, 0].min() == -32  # correctly ordered despite file order
    assert roles["inner"]["properties"]["name"] == "Test"


def test_shipped_fmesh_kmls_match_golden_rings():
    """Our importer reproduces fmesh from_kml on the real region files."""
    golden = np.load(FIXTURES / "fmesh_golden" / "hfun_field.npz")
    for name in ("Baffin_Bay", "Nares", "Peabody"):
        rings = read_kml_rings(FIXTURES / "kml" / f"{name}.kml")
        assert len(rings) == 2
        inner, outer = order_rings(rings[0][1], rings[1][1])
        np.testing.assert_allclose(inner[:, 0], golden[f"region_{name}_inner_lon"], atol=1e-9)
        np.testing.assert_allclose(inner[:, 1], golden[f"region_{name}_inner_lat"], atol=1e-9)
        np.testing.assert_allclose(outer[:, 0], golden[f"region_{name}_outer_lon"], atol=1e-9)
        np.testing.assert_allclose(outer[:, 1], golden[f"region_{name}_outer_lat"], atol=1e-9)


def test_errors(tmp_path):
    bad = tmp_path / "bad.kml"
    bad.write_text("<kml><Document></Document></kml>")
    with pytest.raises(KMLError, match="no Placemark"):
        read_kml_rings(bad)
    bad.write_text("not xml at all <<<")
    with pytest.raises(KMLError, match="not valid XML"):
        read_kml_rings(bad)
    one_ring = tmp_path / "one.kml"
    one_ring.write_text(
        "<kml><Placemark><coordinates>0,0 1,0 1,1</coordinates></Placemark></kml>"
    )
    with pytest.raises(KMLError, match="exactly"):
        kml_to_geojson(one_ring, tmp_path / "out.geojson")
