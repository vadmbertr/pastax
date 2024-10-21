"""
This module provides base and example simulator classes for trajectory reconstruction in JAX.
"""


from ._simulator import Simulator
from ._diffrax_simulator import DiffraxSimulator, DeterministicDiffrax, StochasticDiffrax, SDEControl
from .simulators import LinearSSC, TrainableLinearSSC, SmagorinskyDiffusion
from .simulators import __all__ as simulators_all


__all__ = [
    "Simulator", 
    "DiffraxSimulator", "DeterministicDiffrax", "StochasticDiffrax", "SDEControl",
    "LinearSSC", "TrainableLinearSSC", "SmagorinskyDiffusion"
] + simulators_all
