"""
This module provides dynamics examples to be used with [`pastax.simulator.BaseSimulator`][].
"""

from ._linear_uv import linear_uv, LinearUV
from ._smagorinsky_diffusion import SmagorinskyDiffusion


__all__ = ["linear_uv", "LinearUV", "SmagorinskyDiffusion"]