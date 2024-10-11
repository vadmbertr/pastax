from typing import Callable, Dict, Tuple

import equinox as eqx
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Int

from ..grid import Dataset
from ..timeseries import Displacement, Location, Trajectory, TrajectoryEnsemble
from ..utils import UNIT


class Simulator(eqx.Module):
    """
    A class used to simulate trajectories from various physical fields.

    Attributes
    ----------
    datasets : Dict[str, Dataset] | Dataset
        One or several datasets of gridded physical fields (SSC, SSH, SST, etc.) used by the simulator.
    id : str
        The identifier for the simulator.

    Methods
    -------
    get_domain(x0, t0, ts)
        Computes the minimum and maximum time and location bounds of the simulation space-time domain.
    __call__(x0, t0, ts, n_samples=None, key=None)
        Simulates trajectories based on the initial location, time, and time steps.
    """

    datasets: Dict[str, Dataset]
    id: str

    @staticmethod
    @eqx.filter_jit
    def get_domain(
            x0: Location,
            t0: Int[Array, ""],
            ts: Int[Array, "time-1"]
    ) -> Tuple[Int[Array, ""], Int[Array, ""], Location, Location]:
        """
        Computes the minimum and maximum time and location bounds of the simulation space-time domain.

        Parameters
        ----------
        x0 : Location
            The initial location.
        t0 : Int[Array, ""]
            The initial time.
        ts : Int[Array, "time-1"]
            The time steps for the simulation.

        Returns
        -------
        Tuple[Int[Array, ""], Int[Array, ""], Location, Location]
            The minimum time, maximum time, minimum location, and maximum location bounds.
        """
        one_day = 60 * 60 * 24
        min_time = t0 - one_day
        max_time = ts[-1] + one_day  # this way we can always interpolate in time
        n_days = (max_time - min_time) / one_day

        max_travel_distance = .5  # in °/day ; inferred from data
        max_travel_distance *= (n_days - 2)  # in °
        max_travel_distance = Displacement(
            jnp.full(2, max_travel_distance, dtype=float), unit=UNIT.degrees
        )

        min_corner = Location(x0 - max_travel_distance)
        max_corner = Location(x0 + max_travel_distance)

        return min_time, max_time, min_corner, max_corner

    def __call__(
        self,
        x0: Location,
        t0: Int[Array, ""],
        ts: Int[Array, "time-1"],
        dt0: int = 30 * 60,  # 30 minutes in seconds
        solver: Callable = None,
        n_samples: int = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory | TrajectoryEnsemble:
        """
        Simulates the trajectory based on the initial location, time, and time steps.

        This method must be implemented by its subclasses.

        Parameters
        ----------
        x0 : Location
            The initial location.
        t0 : Int[Array, ""]
            The initial time.
        ts : Int[Array, "time-1"]
            The time steps for the simulation outputs.
        dt0 : int, optional
            The initial time step in seconds (default is 30 * 60 seconds).
        solver : Callable, optional
            The solver function to use for the simulation (default is None).
        n_samples : int, optional
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
            If the method is not implemented.
        """
        # must return the full trajectory, including (x0, t0)
        raise NotImplementedError()
