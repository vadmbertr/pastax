"""
This module provides base [`pastax.simulator.Simulator`][] classes for [`pastax.trajectory.Trajectory`][] 
and [`pastax.trajectory.TrajectoryEnsemble`][] simulation in JAX.
"""


from ._simulator import Simulator
from ._diffrax_simulator import DiffraxSimulator
from .simulator import DeterministicSimulator, StochasticSimulator, SDEControl
# from .dynamics import linear_uv, LinearUV, SmagorinskyDiffusion
from .dynamics import __all__ as dynamics_all


__all__ = [
    "Simulator", 
    "DiffraxSimulator", 
    "DeterministicSimulator", "StochasticSimulator", "SDEControl",
] + dynamics_all
