from typing import Any, Callable, Literal

import equinox as eqx
from jaxtyping import Array, Int, Key, PyTree, Real

from ..trajectory import Location, Trajectory, TrajectoryEnsemble
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

    def __call__(
        self,
        dynamics: Callable[[Real[Any, ""], PyTree, PyTree], PyTree],
        args: PyTree,
        x0: Location,
        ts: Real[Array, "time"],
        dt0: Real[Any, ""],
        solver: Callable,
        ad: Literal["forward", "reverse", None] = "forward",
        n_samples: Int[Any, ""] | None = None,
        key: Key[Array, ""] | None = None,
    ) -> Trajectory | TrajectoryEnsemble:
        r"""
        Simulates a [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][]
        following the prescribe drift `dynamics` and physical field(s) `args`,
        from the initial [`pastax.trajectory.Location`][] `x0` at time steps (including t0) `ts`,
        using a given `solver`.

        This method must be implemented by its subclasses.

        Parameters
        ----------
        dynamics: Callable[[Real[Any, ""], PyTree, PyTree], PyTree]
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
        ts : Real[Any, "time"]
            The time steps for the simulation outputs (including t0).
        dt0 : Real[Any, ""]
            The initial time step of the solver, in seconds.
        solver : Callable
            The solver to use for the simulation.
        ad: Literal["forward", "reverse", None], optional
            The mode used for differentiating through the solve, defaults to "forward".
        n_samples : Int[Any, ""] | None, optional
            The number of samples to generate, defaults to None (not used for deterministic simulators).
        key : Key[Array, ""] | None, optional
            The random key for sampling, defaults to None (not used for deterministic simulators).

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
