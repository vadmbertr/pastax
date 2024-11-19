"""
This module provides dynamics examples to be used with [`pastax.simulator.Simulator`][].
"""

from ._linear_uv import linear_uv, LinearUV
from .smagorinsky_diffusion import SmagorinskyDiffusion


__all__ = ["linear_uv", "LinearUV", "SmagorinskyDiffusion"]
