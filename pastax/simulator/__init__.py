"""
This module provides base and example [`pastax.simulator.Simulator`][] classes for [`pastax.trajectory.Trajectory`][] 
and [`pastax.trajectory.TrajectoryEnsemble`][] simulation in JAX.
"""


from ._simulator import Simulator
from ._diffrax_simulator import DiffraxSimulator, DeterministicDiffrax, StochasticDiffrax, SDEControl
from .simulators import IdentitySimulator, LinearSimulator, LinearRHS, SmagorinskySimulator, SmagorinskyRHS
from .simulators import __all__ as simulators_all


__all__ = [
    "Simulator", 
    "DiffraxSimulator", "DeterministicDiffrax", "StochasticDiffrax", "SDEControl",
] + simulators_all
