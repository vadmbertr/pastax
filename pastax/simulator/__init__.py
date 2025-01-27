"""
This module provides base [`pastax.simulator.BaseSimulator`][] classes for [`pastax.trajectory.Trajectory`][]
and [`pastax.trajectory.TrajectoryEnsemble`][] simulation in JAX.
"""

from ._base_simulator import BaseSimulator
from ._diffrax_simulator import (
    DiffraxSimulator,
    DeterministicSimulator,
    SDEControl,
    StochasticSimulator,
)


__all__ = [
    "BaseSimulator",
    "DiffraxSimulator",
    "DeterministicSimulator",
    "StochasticSimulator",
    "SDEControl",
]
