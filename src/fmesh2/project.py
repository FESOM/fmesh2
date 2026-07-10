"""Project model: `project.yaml` schema, load/save, directory layout.

A project directory looks like::

    myproject/
        project.yaml      # this schema
        regions/          # *.geojson region polygons (drawn or KML-imported)
        fields/           # imported lon/lat resolution fields
        out/              # generated FESOM2 meshes
        cache/            # raw jigsaw meshes, composed hfun grids

All relative paths inside `project.yaml` are resolved against the project
directory, so a project folder is fully relocatable. See
`docs/project-format.md` for the user-facing format description.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

PROJECT_FILENAME = "project.yaml"

#: fmesh's proven 69-level vertical discretization (0 at surface, negative down).
DEFAULT_LEVELS = [
    0.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0,
    -40.0, -45.0, -50.0, -55.0, -60.0, -65.0, -70.0, -75.0,
    -80.0, -85.0, -90.0, -95.0, -100.0, -110.0, -120.0,
    -130.0, -140.0, -150.0, -160.0, -170.0, -180.0, -190.0,
    -200.0, -220.0, -240.0, -260.0, -280.0, -300.0, -340.0,
    -380.0, -420.0, -460.0, -500.0, -540.0, -580.0, -620.0,
    -660.0, -760.0, -860.0, -1040.0, -1180.0, -1380.0,
    -1500.0, -1700.0, -1920.0, -2150.0, -2400.0, -2650.0,
    -2900.0, -3150.0, -3400.0, -3650.0, -3900.0, -4150.0,
    -4400.0, -4650.0, -4900.0, -5150.0, -5400.0, -5650.0,
    -6000.0, -6350.0,
]


class ProjectError(ValueError):
    """Invalid, unreadable, or inconsistent project data."""


class _LayerBase(BaseModel):
    """Fields shared by all resolution layers."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = ""
    enabled: bool = True
    #: how the layer combines with the field composed so far:
    #: "min" keeps the finer of the two, "replace" overwrites where defined.
    mode: Literal["min", "replace"] = "min"


class BaseLayer(_LayerBase):
    """Constant background resolution everywhere."""

    type: Literal["base"]
    resolution: float = Field(gt=0, description="km")


class MercatorLayer(_LayerBase):
    """cos(lat)-scaled resolution with constant caps poleward of the boundaries."""

    type: Literal["mercator"]
    resolution: float = Field(gt=0, description="km at the equator")
    northern_boundary: float = Field(default=60.0, ge=-90, le=90)
    northern_lowest_resolution: float = Field(default=25.0, gt=0)
    southern_boundary: float = Field(default=-60.0, ge=-90, le=90)
    southern_lowest_resolution: float = Field(default=25.0, gt=0)

    @model_validator(mode="after")
    def _boundaries_ordered(self) -> "MercatorLayer":
        if self.southern_boundary >= self.northern_boundary:
            raise ValueError(
                "southern_boundary must be south of northern_boundary "
                f"(got {self.southern_boundary} >= {self.northern_boundary})"
            )
        return self


class CoastLayer(_LayerBase):
    """Distance-to-coastline ramp: min_resolution at the coast, blending into
    the surrounding field over max_distance."""

    type: Literal["coast"]
    min_resolution: float = Field(gt=0, description="km at the coast")
    max_distance: float = Field(gt=0, description="km, ramp width from the coast")
    min_length: float = Field(default=100.0, ge=0, description="km, ignore shorter coastlines")
    averaging: float = Field(default=50.0, ge=0, description="km, coastline smoothing")
    #: path to a custom coastline file; None uses the bundled source
    source: str | None = None


class BathyLayer(_LayerBase):
    """Depth ramp: min_resolution at depths <= min_depth, blending into the
    surrounding field at max_depth. Land is untouched."""

    type: Literal["bathy"]
    min_resolution: float = Field(gt=0, description="km over the shallowest water")
    min_depth: float = Field(default=50.0, gt=0, description="m, full refinement above this depth")
    max_depth: float = Field(default=1500.0, gt=0, description="m, no refinement below this depth")

    @model_validator(mode="after")
    def _depths_ordered(self) -> "BathyLayer":
        if self.max_depth <= self.min_depth:
            raise ValueError(
                f"max_depth must exceed min_depth (got {self.max_depth} <= {self.min_depth})"
            )
        return self


class RegionLayer(_LayerBase):
    """Target resolution inside a polygon, blending outward.

    With `transition_km` set, the GeoJSON needs a single polygon and the blend
    happens across that band. Without it, the GeoJSON must contain two rings
    (inner + outer, fmesh KML style) and the blend spans the gap between them.
    """

    type: Literal["region"]
    path: str = Field(description="GeoJSON path, usually under regions/")
    resolution: float = Field(gt=0, description="km inside the polygon")
    transition_km: float | None = Field(default=None, gt=0)


class FieldLayer(_LayerBase):
    """Resolution field imported from a lon/lat grid (NetCDF or 3-column text)."""

    type: Literal["field"]
    path: str = Field(description="data path, usually under fields/")
    variable: str | None = None


Layer = Annotated[
    Union[BaseLayer, MercatorLayer, CoastLayer, BathyLayer, RegionLayer, FieldLayer],
    Field(discriminator="type"),
]

LAYER_TYPES = ("base", "mercator", "coast", "bathy", "region", "field")


class GridSettings(BaseModel):
    """Hfun lat/lon grid density as interval counts (fmesh convention:
    n_lat intervals produce n_lat+1 latitudes). None means auto-suggest
    from the finest layer resolution at composition time."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    n_lat: int | None = Field(default=None, ge=2)
    n_lon: int | None = Field(default=None, ge=2)


class Project(BaseModel):
    """Everything needed to reproduce a mesh, minus bulky data (topo, caches)."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = Field(min_length=1)
    topo: str | None = Field(default=None, description="topography NetCDF path")
    topo_var: str | None = Field(
        default=None, description="elevation variable name, if auto-detection fails"
    )
    grid: GridSettings = Field(default_factory=GridSettings)
    layers: list[Layer] = Field(default_factory=list)
    levels: list[float] = Field(default_factory=lambda: list(DEFAULT_LEVELS))
    minimum_depth: float = Field(default=30.0, gt=0, description="m, positive")
    output_dir: str = "out"
    #: lon/lat points known to be ocean; the mesh component containing the
    #: first seed survives lake elimination.
    seeds: list[tuple[float, float]] = Field(default_factory=lambda: [(0.0, 0.0)])

    _dir: Path | None = PrivateAttr(default=None)

    @field_validator("levels")
    @classmethod
    def _levels_decreasing(cls, v: list[float]) -> list[float]:
        if len(v) < 2:
            raise ValueError("levels needs at least 2 entries")
        if any(b >= a for a, b in zip(v, v[1:])):
            raise ValueError("levels must be strictly decreasing (0 at surface, negative down)")
        return v

    @field_validator("seeds")
    @classmethod
    def _seeds_valid(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not v:
            raise ValueError("seeds must contain at least one (lon, lat) point")
        for lon, lat in v:
            if not -90.0 <= lat <= 90.0:
                raise ValueError(f"seed latitude {lat} outside [-90, 90]")
        return v

    # ---------------------------------------------------------------- paths

    @property
    def project_dir(self) -> Path | None:
        """Directory this project was loaded from / saved to, if any."""
        return self._dir

    def resolve_path(self, p: str | Path) -> Path:
        """Absolute paths pass through; relative ones resolve against the
        project directory (requires the project to be bound to one)."""
        p = Path(p).expanduser()
        if p.is_absolute():
            return p
        if self._dir is None:
            raise ProjectError(
                f"cannot resolve relative path {str(p)!r}: project is not bound to a "
                "directory (load it from disk or save it first)"
            )
        return (self._dir / p).resolve()

    @property
    def regions_dir(self) -> Path:
        return self.resolve_path("regions")

    @property
    def fields_dir(self) -> Path:
        return self.resolve_path("fields")

    @property
    def cache_dir(self) -> Path:
        return self.resolve_path("cache")

    @property
    def out_dir(self) -> Path:
        return self.resolve_path(self.output_dir)

    @property
    def topo_path(self) -> Path | None:
        return None if self.topo is None else self.resolve_path(self.topo)

    def ensure_layout(self) -> None:
        """Create the standard project subdirectories."""
        for d in (self.regions_dir, self.fields_dir, self.cache_dir, self.out_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ load/save

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        """Load from a `project.yaml` file or a directory containing one."""
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / PROJECT_FILENAME
        if not path.is_file():
            raise ProjectError(f"project file not found: {path}")
        try:
            raw = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            raise ProjectError(f"{path} is not valid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise ProjectError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
        try:
            project = cls.model_validate(raw)
        except ValidationError as e:
            raise ProjectError(f"invalid project {path}:\n{e}") from e
        project._dir = path.parent.resolve()
        return project

    def save(self, path: str | Path | None = None) -> Path:
        """Write `project.yaml`, binding the project to its directory.

        `path` may be a yaml file or a directory; omitted, it saves back to
        the bound project directory. Layer order is preserved as-is.
        """
        if path is None:
            if self._dir is None:
                raise ProjectError("project is not bound to a directory; pass a path to save()")
            path = self._dir / PROJECT_FILENAME
        else:
            path = Path(path).expanduser()
            if path.is_dir() or path.suffix not in (".yaml", ".yml"):
                path = path / PROJECT_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())
        self._dir = path.parent.resolve()
        return path

    def to_yaml(self) -> str:
        data = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
