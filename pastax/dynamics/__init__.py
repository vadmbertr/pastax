"""
This module provides dynamics examples to be used with [`pastax.simulator.BaseSimulator`][].
"""

from ._linear_uv import linear_uv, LinearUV
from ._smagorinsky_diffusion import (
    DeterministicSmagorinskyDiffusion,
    SmagorinskyDiffusion,
    StochasticSmagorinskyDiffusion,
)


__all__ = [
    "linear_uv",
    "LinearUV",
    "DeterministicSmagorinskyDiffusion",
    "StochasticSmagorinskyDiffusion",
    "SmagorinskyDiffusion",
]
