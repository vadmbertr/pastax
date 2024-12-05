from __future__ import annotations
from typing import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float, Int, PyTree, Scalar

from ..trajectory import Location, Trajectory, TrajectoryEnsemble
from ._base_simulator import BaseSimulator


class DiffraxSimulator(BaseSimulator):
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
                Could be for example one or several [`pastax.gridded.Gridded`][] of gridded physical fields 
                (SSC, SSH, SST, etc...).

            Returns
            -------
            PyTree
                The drift dynamics.

        args : PyTree
            The PyTree of argument(s) required to compute the `dynamics`.
            Could be for example one or several [`pastax.gridded.Gridded`][] of gridded physical fields 
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


class DeterministicSimulator(DiffraxSimulator):
    """
    Class defining deterministic differentiable [`pastax.trajectory.Trajectory`][] simulators.

    Methods
    -------
    __call__(dynamics, args, x0, ts, dt0, solver, n_samples, key)
        Simulates a [`pastax.trajectory.Trajectory`][] following the prescribe drift `dynamics` 
        and physical field(s) `args`, from the initial [`pastax.trajectory.Location`][] `x0` 
        at time steps (including t0) `ts`, using a given [`diffrax.AbstractSolver`][].
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
    ) -> Trajectory:
        r"""
        Simulates a [`pastax.trajectory.Trajectory`][] following the prescribe drift `dynamics` 
        and physical field(s) `args`, from the initial [`pastax.trajectory.Location`][] `x0` 
        at time steps (including t0) `ts`, using a given [`diffrax.AbstractSolver`][].

        Parameters
        ----------
        dynamics : Callable[[int, Float[Array, "2"], PyTree], PyTree]
            A Callable (including an [`equinox.Module`][] with a `__call__` method) describing the dynamics of the 
            right-hand-side of the solved Ordinary Differential Equation.
            
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
                The PyTree of argument(s) required to compute the `dynamics`.
                Could be for example one or several [`pastax.gridded.Gridded`][] of gridded physical fields 
                (SSC, SSH, SST, etc...).

            Returns
            -------
            PyTree
                The drift dynamics.
        
        args : PyTree
            The PyTree of argument(s) required to compute the `dynamics`.
            Could be for example one or several [`pastax.gridded.Gridded`][] of gridded physical fields 
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
            The number of samples to generate, defaults to None, not use with deterministic simulators.
        key : jrd.PRNGKey, optional
            The random key for sampling, default to None, not use with deterministic simulators.

        Returns
        -------
        Trajectory
            The simulated [`pastax.trajectory.Trajectory`][], including the initial conditions (x0, t0).
        """
        t0 = ts[0]
        t1 = ts[-1]

        ys = dfx.diffeqsolve(
            dfx.ODETerm(dynamics),
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=x0.value,
            args=args,
            saveat=dfx.SaveAt(ts=ts[1:])
        ).ys

        return Trajectory.from_array(jnp.concat((x0.value[None], ys), axis=0), ts, unit=x0.unit)


class SDEControl(dfx.AbstractPath):
    """
    Class representing a [`diffrax.AbstractPath`][] for Stochastic Differential Equation.

    Attributes
    ----------
    t0 : Float[Scalar, ""]
        The initial time.
    t1 : Float[Scalar, ""]
        The final time.
    brownian_motion : dfx.VirtualBrownianTree
        The [`diffrax.VirtualBrownianTree`][] used in the Stochastic Differential Equation.

    Methods
    -------
    evaluate(t0, t1=None, left=True, use_levy=False)
        Evaluates the path at the given time points.
    """

    t0 = None
    t1 = None
    brownian_motion: dfx.VirtualBrownianTree

    def evaluate(
        self,
        t0: Float[Scalar, ""], 
        t1: Float[Scalar, ""] = None, 
        left: bool = True, 
        use_levy: bool = False
    ) -> Float[Array, "x+1"]:
        """
        Evaluates the path at the given time points.

        Parameters
        ----------
        t0 : Float[Scalar, ""]
            The initial time.
        t1 : Float[Scalar, ""], optional
            The final time, defaults to None.
        left : bool, optional
            Whether to use the left limit, defaults to True.
        use_levy : bool, optional
            Whether to use the Levy area, defaults to False.

        Returns
        -------
        Float[Array, "x+1"]
            The evaluated control.
        """
        return jnp.concatenate(
            [
                jnp.asarray([t1 - t0], dtype=float), 
                self.brownian_motion.evaluate(t0=t0, t1=t1, left=left, use_levy=use_levy)
            ]
        )


class StochasticSimulator(DiffraxSimulator):
    """
    Class defining stochastic differentiable [`pastax.trajectory.TrajectoryEnsemble`][] simulators.

    Methods
    -------
    __call__(dynamics, args, x0, ts, dt0, solver, n_samples, key)
        Simulates a [`pastax.trajectory.TrajectoryEnsemble`][] of `n_samples` [`pastax.trajectory.Trajectory`][] 
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
        n_samples: Int[Scalar, ""] = jnp.asarray(100),
        key: jrd.PRNGKey = jrd.key(0)
    ) -> TrajectoryEnsemble:
        r"""
        Simulates a [`pastax.trajectory.TrajectoryEnsemble`][] based on the initial [`pastax.trajectory.Location`][] 
        and time steps (including t0).

        Parameters
        ----------
        dynamics : Callable[[int, Float[Array, "2"], PyTree], PyTree]
            A Callable (including an [`equinox.Module`][] with a `__call__` method) describing the dynamics of the 
            right-hand-side of the solved Stochastic Differential Equation.
            
            !!! example

                Formulating a displacement at time $t$ from the position $\mathbf{X}(t)$ as:

                $$
                d\mathbf{X}(t) = f(t, \mathbf{X}(t), \text{args}) \cdot [dt, d\mathbf{W}(t)]
                $$

                `dynamics` is here the function $f$ returning the displacement speed and diffusion as a 2*3 matrix.
        
            Parameters
            ----------
            t : int
                The current time.
            y : Float[Array, "2"]
                The current state (latitude and longitude in degrees).
            args : PyTree
                The PyTree of argument(s) required to compute the `dynamics`.
                Could be for example one or several [`pastax.gridded.Gridded`][] of gridded physical fields 
                (SSC, SSH, SST, etc...).

            Returns
            -------
            PyTree
                The drift dynamics.
        
        args : PyTree
            The PyTree of argument(s) required to compute the `dynamics`.
            Could be for example one or several [`pastax.gridded.Gridded`][] of gridded physical fields 
            (SSC, SSH, SST, etc...).
        x0 : Location
            The initial [`pastax.trajectory.Location`][].
        ts : Float[Array, "time"]
            The time steps for the simulation outputs (including t0).
        dt0 : Float[Scalar, ""]
            The initial time step of the solver, in seconds.
        solver : dfx.AbstractSolver, optional
            The [`diffrax.AbstractSolver`][] to use for the simulation, defaults to [`diffrax.Heun`][].
        n_samples : Int[Scalar, ""], optional
            The number of samples to generate, defaults to `jnp.asarray(100)`.
        key : jrd.PRNGKey, optional
            The random key for sampling, defaults to `jrd.key(0)`.

        Returns
        -------
        TrajectoryEnsemble
            The simulated [`pastax.trajectory.TrajectoryEnsemble`][].
        """
        t0 = ts[0]  
        t1 = ts[-1]

        keys = jrd.split(key, n_samples)

        @jax.vmap
        def solve(subkey: jrd.PRNGKey) -> Float[Array, "time 2"]:
            eps = 1e-3
            brownian_motion = dfx.VirtualBrownianTree(t0, t1 + eps, tol=eps, shape=(2,), key=subkey)
            sde_control = SDEControl(brownian_motion=brownian_motion)
            sde_term = dfx.ControlTerm(dynamics, sde_control)

            return dfx.diffeqsolve(
                sde_term,
                solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=x0.value,
                args=args,
                saveat=dfx.SaveAt(ts=ts[1:])
            ).ys

        ys = solve(keys)

        y0 = jnp.tile(x0.value, (n_samples, 1, 1))
        return TrajectoryEnsemble.from_array(jnp.concat((y0, ys), axis=1), ts, unit=x0.unit)

