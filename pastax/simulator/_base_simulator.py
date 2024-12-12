from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, Scalar

from ..trajectory import Displacement, Location, Trajectory, TrajectoryEnsemble
from ..utils._unit import UNIT


class BaseSimulator(eqx.Module):
    """
    Base class for defining differentiable [`pastax.trajectory.Trajectory`][] or
    [`pastax.trajectory.TrajectoryEnsemble`][] simulators.

    Attributes
    ----------
    id : str | None
        The identifier for the simulator.

    Methods
    -------
    get_domain(x0, t0, ts)
        Computes the minimum and maximum time and location bounds of the simulation space-time domain.
    __call__(dynamics, args, x0, ts, dt0, solver, n_samples=None, key=None)
        Simulates a [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][]
        following the prescribe drift `dynamics` and physical field(s) `args`,
        from the initial [`pastax.trajectory.Location`][] `x0` at time steps (including t0) `ts`,
        using a given `solver`.
    """

    id: str | None = None

    @staticmethod
    def get_domain(  # TODO: to be removed?
        x0: Location, ts: Float[Array, "time"]
    ) -> tuple[Float[Array, ""], Float[Array, ""], Location, Location]:
        """
        Computes the minimum and maximum time and location bounds of the simulation space-time domain.

        Parameters
        ----------
        x0 : Location
            The initial [`pastax.trajectory.Location`][].
        ts : Float[Array, "time"]
            The time steps for the simulation.

        Returns
        -------
        tuple[Float[Array, ""], Float[Array, ""], Location, Location]
            The minimum time, maximum time, minimum location, and maximum location bounds.
        """
        one_day = 60 * 60 * 24
        min_time = ts[0] - one_day
        max_time = ts[-1] + one_day  # this way we can always interpolate in time
        n_days = (max_time - min_time) / one_day

        max_travel_distance = 0.5  # in °/day ; inferred from data
        max_travel_distance *= n_days - 2  # in °
        max_travel_distance = Displacement(jnp.full(2, max_travel_distance, dtype=float), unit=UNIT["°"])

        min_corner = Location(x0.value - max_travel_distance)
        max_corner = Location(x0.value + max_travel_distance)

        return min_time, max_time, min_corner, max_corner

    def __call__(
        self,
        dynamics: Callable[[int, Float[Array, "2"], PyTree], PyTree],
        args: PyTree,
        x0: Location,
        ts: Float[Array, "time"],
        dt0: Float[Scalar, ""],
        solver: Callable | None = None,
        n_samples: Int[Scalar, ""] | None = None,
        key: Array | None = None,
    ) -> Trajectory | TrajectoryEnsemble:
        r"""
        Simulates a [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][]
        following the prescribe drift `dynamics` and physical field(s) `args`,
        from the initial [`pastax.trajectory.Location`][] `x0` at time steps (including t0) `ts`,
        using a given `solver`.

        This method must be implemented by its subclasses.

        Parameters
        ----------
        dynamics: Callable[[int, Float[Array, "2"], PyTree], PyTree]
            A Callable (including an [`equinox.Module`][] with a __call__ method) describing the dynamics of the
            right-hand-side of the solved Differential Equation.

            !!! example

                Formulating the displacement at time $t$ from the position $\mathbf{X}(t)$ as:

                $$
                d\mathbf{X}(t) = f(t, \mathbf{X}(t), \text{args}) dt
                $$

                `dynamics` is here the function $f$ returning the displacement speed.
                In the simpliest case, $f$ is the function interpolating a velocity field $\mathbf{u}$
                in space and time.

        args : PyTree
            The PyTree of argument(s) required to compute the `dynamics`.
            Could be for example one or several [`pastax.gridded.Gridded`][] of gridded physical fields
            (SSC, SSH, SST, etc...).
        x0 : Location
            The initial [`pastax.trajectory.Location`][].
        ts : Float[Array, "time"]
            The time steps for the simulation outputs (including t0).
        dt0 : Float[Scalar, ""], optional
            The initial time step of the solver, in seconds.
        solver : Callable | None, optional
            The solver to use for the simulation, defaults to None.
        n_samples : Int[Scalar, ""] | None, optional
            The number of samples to generate, default to None, meaning a single [`pastax.trajectory.Trajectory`][].
        key : Array | None, optional
            The random key for sampling, defaults to None.

        Returns
        -------
        Trajectory | TrajectoryEnsemble
            The simulated [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][],
            including the initial conditions (x0, t0).

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError()
