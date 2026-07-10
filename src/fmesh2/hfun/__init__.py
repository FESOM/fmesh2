"""Resolution-field (hfun) grid, layers, and composition."""

from .compose import ComposeResult, HfunWarning, compose, finest_scale, register
from .grid import HfunGrid, SizeEstimate, estimate_size, suggest_intervals

from . import base_layers  # noqa: E402,F401  (registers base + mercator evaluators)
from . import bathy_layer  # noqa: E402,F401  (registers bathy evaluator)
from . import coast_layer  # noqa: E402,F401  (registers coast evaluator)

__all__ = [
    "ComposeResult",
    "HfunGrid",
    "HfunWarning",
    "SizeEstimate",
    "compose",
    "estimate_size",
    "finest_scale",
    "register",
    "suggest_intervals",
]
