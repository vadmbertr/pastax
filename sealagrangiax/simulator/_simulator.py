from typing import Callable, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float, Int, PyTree, Scalar

from ..trajectory import Displacement, Location, Trajectory, TrajectoryEnsemble
from ..utils.unit import UNIT


class Simulator(eqx.Module):
    """
    Base class for defining differentiable trajectories or ensembles of trajectories simulators.

    Attributes
    ----------
    id : str
        The identifier for the simulator.

    Methods
    -------
    get_domain(x0, t0, ts)
        Computes the minimum and maximum time and location bounds of the simulation space-time domain.
    __call__(args, x0, ts, n_samples=None, key=None)
        Simulates a trajectory or ensemble of trajectories based on the initial location and time steps (including t0).
    """

    id: str

    @staticmethod
    def get_domain(
        x0: Location,
        ts: Float[Array, "time"]
    ) -> Tuple[Float[Array, ""], Float[Array, ""], Location, Location]:
        """
        Computes the minimum and maximum time and location bounds of the simulation space-time domain.

        Parameters
        ----------
        x0 : Location
            The initial location.
        ts : Float[Array, "time"]
            The time steps for the simulation.

        Returns
        -------
        Tuple[Float[Array, ""], Float[Array, ""], Location, Location]
            The minimum time, maximum time, minimum location, and maximum location bounds.
        """
        one_day = 60 * 60 * 24
        min_time = ts[0] - one_day
        max_time = ts[-1] + one_day  # this way we can always interpolate in time
        n_days = (max_time - min_time) / one_day

        max_travel_distance = .5  # in °/day ; inferred from data
        max_travel_distance *= (n_days - 2)  # in °
        max_travel_distance = Displacement(
            jnp.full(2, max_travel_distance, dtype=float), unit=UNIT["°"]
        )

        min_corner = Location(x0 - max_travel_distance)
        max_corner = Location(x0 + max_travel_distance)

        return min_time, max_time, min_corner, max_corner

    def __call__(
        self,
        args: PyTree,
        x0: Location,
        ts: Float[Array, "time"],
        dt0: Float[Scalar, ""],
        solver: Callable = None,
        n_samples: Int[Scalar, ""] = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory | TrajectoryEnsemble:
        """
        Simulates a trajectory or ensemble of trajectories based on the initial location and time steps (including t0).

        This method must be implemented by its subclasses.

        Parameters
        ----------
        args : PyTree
            Any PyTree of argument(s) used by the simulator.
            Could be for example one or several `sealagrangiax.Dataset` of gridded physical fields (SSC, SSH, SST, etc.).
        x0 : Location
            The initial location.
        ts : Float[Array, "time"]
            The time steps for the simulation outputs (including t0).
        dt0 : Float[Scalar, ""], optional
            The initial time step of the solver, in seconds.
        solver : Callable, optional
            The solver function to use for the simulation (default is None).
        n_samples : Int[Scalar, ""], optional
            The number of samples to generate (default is None, meaning a single trajectory).
        key : jrd.PRNGKey, optional
            The random key for sampling (default is None).

        Returns
        -------
        Trajectory | TrajectoryEnsemble
            The simulated trajectory or ensemble of trajectories, including the initial conditions (x0, t0).

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError()
