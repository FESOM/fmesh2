import fmesh2


def test_version():
    assert isinstance(fmesh2.__version__, str)
    assert fmesh2.__version__
