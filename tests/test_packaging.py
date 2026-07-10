from importlib import resources


def test_frontend_bundle_shipped():
    static = resources.files("fmesh2.server") / "static"
    index = static / "index.html"
    assert index.is_file(), "frontend bundle missing — run `make frontend-build`"
    html = index.read_text()
    assert "fmesh2" in html
    # the built bundle references hashed asset files that must ship alongside
    assets = static / "assets"
    assert any(f.name.endswith(".js") for f in assets.iterdir())
