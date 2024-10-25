"""
This module provides base and example [`pastax.Simulator`][] classes for [`pastax.Trajectory`][] 
and [`pastax.TrajectoryEnsemble`][] simulation in JAX.
"""


from ._simulator import Simulator
from ._diffrax_simulator import DiffraxSimulator, DeterministicDiffrax, StochasticDiffrax, SDEControl
from .simulators import IdentitySSC, LinearSSC, SmagorinskyDiffusion
from .simulators import __all__ as simulators_all


__all__ = [
    "Simulator", 
    "DiffraxSimulator", "DeterministicDiffrax", "StochasticDiffrax", "SDEControl",
] + simulators_all
