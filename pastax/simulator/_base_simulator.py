from typing import Any, Callable

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
    __call__(dynamics, args, x0, ts, solver, dt0)
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
        n_samples: Int[Any, ""] | None,
        key: Key[Array, ""] | None,
        solver: Callable,
        dt0: Real[Any, ""],
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
            The time steps for the simulation outputs, including $t_0$, unit should be the same as for `dt0`.
        solver : Callable
            The solver to use for the simulation.
        dt0 : Real[Any, ""]
            The initial time step of the solver, unit should be the same as for `ts`.

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
