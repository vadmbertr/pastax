from __future__ import annotations
from typing import Callable

import diffrax as dfx
import jax.random as jrd
from jaxtyping import Array, Float, Int, PyTree, Scalar

from ..trajectory import Location, Trajectory, TrajectoryEnsemble
from ._simulator import Simulator


class DiffraxSimulator(Simulator):
    """
    Base class for defining differentiable [`pastax.trajectory.Trajectory`][] or 
    [`pastax.trajectory.TrajectoryEnsemble`][] simulators using `diffrax` library.

    Methods
    -------
    __call__(dynamics, args, x0, ts, dt0, solver, n_samples, key)
        Simulates a [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][] 
        following the prescribe drift `dynamics` and physical field(s) `args`, 
        from the initial [`pastax.trajectory.Location`][] `x0` at time steps (including t0) `ts`,
        using a given [`diffrax.AbstractSolver`][].
    """

    def __call__(
        self,
        dynamics: Callable[[int, Float[Array, "2"], PyTree], PyTree],
        args: PyTree,
        x0: Location,
        ts: Float[Array, "time"],
        dt0: Float[Scalar, ""],
        solver: dfx.AbstractSolver = dfx.Heun(),
        n_samples: Int[Scalar, ""] = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory | TrajectoryEnsemble:
        r"""
        Simulates a [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][] 
        following the prescribe drift `dynamics` and physical field(s) `args`, 
        from the initial [`pastax.trajectory.Location`][] `x0` at time steps (including t0) `ts`,
        using a given [`diffrax.AbstractSolver`][].

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
        
            Parameters
            ----------
            t : int
                The current time.
            y : Float[Array, "2"]
                The current state (latitude and longitude in degrees).
            args : PyTree
                Any PyTree of argument(s) used by the simulator.
                Could be for example one or several [`pastax.grid.Dataset`][] of gridded physical fields 
                (SSC, SSH, SST, etc...).

            Returns
            -------
            PyTree
                The drift dynamics.

        args : PyTree
            The PyTree of argument(s) required to compute the `dynamics`.
            Could be for example one or several [`pastax.grid.Dataset`][] of gridded physical fields 
            (SSC, SSH, SST, etc...).
        x0 : Location
            The initial [`pastax.trajectory.Location`][].
        ts : Float[Array, "time"]
            The time steps for the simulation outputs.
        dt0 : Float[Scalar, ""]
            The initial time step of the solver, in seconds.
        solver : dfx.AbstractSolver, optional
            The [`diffrax.AbstractSolver`][] to use for the simulation, defaults to [`diffrax.Heun`][].
        n_samples : Int[Scalar, ""], optional
            The number of samples to generate, defaults to None, meaning a single trajectory.
        key : jrd.PRNGKey, optional
            The random key for sampling, defaults to None, useless for the deterministic simulator.

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
        raise NotImplementedError
