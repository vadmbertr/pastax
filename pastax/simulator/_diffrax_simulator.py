from __future__ import annotations

from typing import Any, Callable, Literal

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float, Int, Key, PyTree, Real

from ..trajectory import Location, Trajectory, TrajectoryEnsemble
from ._base_simulator import BaseSimulator


class DiffraxSimulator(BaseSimulator):
    """
    Base class for defining differentiable [`pastax.trajectory.Trajectory`][] or
    [`pastax.trajectory.TrajectoryEnsemble`][] simulators using `diffrax` library.

    Methods
    -------
    __call__(dynamics, args, x0, ts, solver, dt0)
        Simulates a [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][]
        following the prescribe drift `dynamics` and physical field(s) `args`,
        from the initial [`pastax.trajectory.Location`][] `x0` at time steps (including t0) `ts`,
        using a given [`diffrax.AbstractSolver`][].
    """

    @classmethod
    def _get_diffeqsolve_best_args(
        cls,
        ts: Real[Any, "time"],
        dt0: Real[Any, ""],
        constant_step_size: bool = True,
        n_steps: Int[Any, ""] = None,
        ad_mode: Literal["forward", "reverse"] = "forward",
    ) -> tuple[
        Real[Any, ""],
        dfx.SaveAt,
        dfx.StepsizeController,
        dfx.Adjoint,
        Int[Any, ""],
        Callable[[tuple[int, ...], Key[Array, ""]], dfx.AbstractBrownianPath],
    ]:
        t0 = ts[0]
        t1 = ts[-1]

        if constant_step_size and n_steps is not None:
            brownian_motion: Callable[[tuple[int, ...], Key[Array, ""]], dfx.AbstractBrownianPath] = (
                lambda shape, key: PrecomputedBrownianMotion(n_steps=n_steps, dt=dt0, shape=shape, key=key)
            )
        else:
            brownian_motion: Callable[[tuple[int, ...], Key[Array, ""]], dfx.AbstractBrownianPath] = (
                lambda shape, key: dfx.VirtualBrownianTree(t0, t1, tol=dt0, shape=shape, key=key)
            )

        if n_steps is not None:
            dt0 = None
            saveat = dfx.SaveAt(steps=True)
            stepsize_controller = dfx.StepTo(ts=jnp.linspace(t0, t1, n_steps + 1))
        else:
            saveat = dfx.SaveAt(ts=ts)
            stepsize_controller = dfx.ConstantStepSize()

        if ad_mode == "reverse":
            adjoint = dfx.RecursiveCheckpointAdjoint()
        else:
            adjoint = dfx.ForwardMode()

        return dt0, saveat, stepsize_controller, adjoint, n_steps, brownian_motion

    def __call__(
        self,
        dynamics: Callable[[Real[Any, ""], PyTree, PyTree], PyTree],
        args: PyTree,
        x0: Location,
        ts: Real[Any, "time"],
        solver: dfx.AbstractSolver = dfx.Heun(),
        dt0: Real[Any, ""] = None,
        saveat: dfx.SaveAt | None = None,
        stepsize_controller: dfx.StepsizeController = dfx.ConstantStepSize(),
        adjoint: dfx.Adjoint = dfx.ForwardMode(),
        max_steps: Int[Any, ""] = 4096,
    ) -> Trajectory | TrajectoryEnsemble:
        r"""
        Simulates a [`pastax.trajectory.Trajectory`][] or [`pastax.trajectory.TrajectoryEnsemble`][]
        following the prescribe drift `dynamics` and physical field(s) `args`,
        from the initial [`pastax.trajectory.Location`][] `x0` at time steps (including t0) `ts`,
        using a given [`diffrax.AbstractSolver`][].

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

            Parameters
            ----------
            t : Real[Array, ""]
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
        ts : Real[Any, "time"]
            The time steps for the simulation outputs, including $t_0$, unit should be the same as for `dt0`.
        solver : dfx.AbstractSolver, optional
            The [`diffrax.AbstractSolver`][] to use for the simulation, defaults to [`diffrax.Heun`][].
        dt0 : Real[Any, ""], optional
            The initial time step of the solver, unit should be the same as for `ts`, defaults to `None`.
        saveat : dfx.SaveAt, optional
            The [`diffrax.SaveAt`][] object to use for saving the solution, defaults to `SaveAt(ts=ts)`.
        stepsize_controller : dfx.StepsizeController, optional
            The [`diffrax.StepsizeController`][] to use for controlling the stepsize,
            defaults to [`diffrax.ConstantStepSize`][].
        adjoint : dfx.Adjoint, optional
            The [`diffrax.Adjoint`][] object to use for the adjoint method, defaults to [`diffrax.ForwardMode`][].
            [`diffrax.ForwardMode`][] should be used when computing the gradient in forward automtic differentiation
            mode with respect to few (<50) parameters, while [`diffrax.RecursiveCheckpointAdjoint`][] should be used
            when computing the gradient in reverse automatic differentiation mode with respect to many (>50) parameters.
        max_steps : Int[Any, ""], optional
            The maximum number of steps to take, defaults to `4096`.

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
        dynamics: Callable[[Real[Any, ""], PyTree, PyTree], PyTree],
        args: PyTree,
        x0: Location,
        ts: Real[Any, "time"],
        solver: dfx.AbstractSolver = dfx.Heun(),
        dt0: Real[Any, ""] = None,
        saveat: dfx.SaveAt | None = None,
        stepsize_controller: dfx.StepsizeController = dfx.ConstantStepSize(),
        adjoint: dfx.Adjoint = dfx.ForwardMode(),
        max_steps: Int[Any, ""] = 4096,
    ) -> Trajectory:
        r"""
        Simulates a [`pastax.trajectory.Trajectory`][] following the prescribe drift `dynamics`
        and physical field(s) `args`, from the initial [`pastax.trajectory.Location`][] `x0`
        at time steps (including t0) `ts`, using a given [`diffrax.AbstractSolver`][].

        Parameters
        ----------
        dynamics : Callable[[Real[Any, ""], PyTree, PyTree], PyTree]
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
            t : Real[Any, ""]
                The current time.
            y : PyTree
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
        ts : Real[Any, "time"]
            The time steps for the simulation outputs, including $t_0$, unit should be the same as for `dt0`.
        solver : dfx.AbstractSolver, optional
            The [`diffrax.AbstractSolver`][] to use for the simulation, defaults to [`diffrax.Heun`][].
        dt0 : Real[Any, ""], optional
            The initial time step of the solver, unit should be the same as for `ts`, defaults to `None`.
        saveat : dfx.SaveAt, optional
            The [`diffrax.SaveAt`][] object to use for saving the solution, defaults to `SaveAt(ts=ts)`.
        stepsize_controller : dfx.StepsizeController, optional
            The [`diffrax.StepsizeController`][] to use for controlling the stepsize,
            defaults to [`diffrax.ConstantStepSize`][].
        adjoint : dfx.Adjoint, optional
            The [`diffrax.Adjoint`][] object to use for the adjoint method, defaults to [`diffrax.ForwardMode`][].
            [`diffrax.ForwardMode`][] should be used when computing the gradient in forward automtic differentiation
            mode with respect to few (<50) parameters, while [`diffrax.RecursiveCheckpointAdjoint`][] should be used
            when computing the gradient in reverse automatic differentiation mode with respect to many (>50) parameters.
        max_steps : Int[Any, ""], optional
            The maximum number of steps to take, defaults to `4096`.

        Returns
        -------
        Trajectory
            The simulated [`pastax.trajectory.Trajectory`][], including the initial conditions (x0, t0).
        """
        t0, t1 = ts[0], ts[-1]

        if saveat is None:
            saveat = dfx.SaveAt(ts=ts)

        ys = dfx.diffeqsolve(
            dfx.ODETerm(dynamics),
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=x0.value,
            args=args,
            saveat=saveat,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            max_steps=max_steps,
        ).ys

        return Trajectory.from_array(ys, ts, unit=x0.unit)  # type: ignore


class PrecomputedBrownianMotion(dfx.AbstractBrownianPath):
    """
    Class representing a precomputed Brownian motion for use with Stochastic Differential Equation.
    It requires that the (maximum) number of time steps is known in advance
    and that the integration time step is constant.

    Attributes
    ----------
    dWs : Float[Any, "nt x"]
        The precomputed Brownian increments.
    counter : int
        The current index for the Brownian increments.

    Methods
    -------
    evaluate(t0, t1=None, left=True, use_levy=False)
        Returns the next Brownian increment.
    """

    dWs: Float[Any, "nt x"]
    counter: int = 0

    def __init__(
        self,
        n_steps: Real[Any, ""],
        dt: Real[Any, ""],
        shape: tuple[Real[Any, ""], ...],
        key: Key[Array, ""],
    ):
        self.dWs = jrd.normal(key, (n_steps, *shape)) * jnp.sqrt(dt)

    def evaluate(
        self,
        t0: Real[Any, ""],
        t1: Real[Any, ""],
        left: bool = True,
        use_levy: bool = False,
    ) -> tuple[Real[Any, ""], Float[Any, "x"]]:
        """
        Returns the next Brownian increment.
        Parameters are not used here as the Brownian motion is precomputed.

        Parameters
        ----------
        t0 : Real[Any, ""]
            The initial time.
        t1 : Real[Any, ""]
            The final time.
        left : bool, optional
            Whether to use the left limit, defaults to True.
        use_levy : bool, optional
            Whether to use the Levy area, defaults to False.

        Returns
        -------
        Float[Any, "x"]
            The Brownian increment.
        """
        dW = self.dWs[self.counter]
        self.counter += 1
        return dW


class SDEControl(dfx.AbstractPath):
    """
    Class representing a [`diffrax.AbstractPath`][] for Stochastic Differential Equation.

    Attributes
    ----------
    t0 : Real[Any, ""]
        The initial time.
    t1 : Real[Any, ""]
        The final time.
    brownian_motion : dfx.AbstractBrownianPath
        The [`diffrax.AbstractBrownianPath`][] used in the Stochastic Differential Equation.

    Methods
    -------
    evaluate(t0, t1=None, left=True, use_levy=False)
        Evaluates the path at the given time points.
    """

    t0: Real[Any, ""]
    t1: Real[Any, ""]
    brownian_motion: dfx.AbstractBrownianPath

    def evaluate(
        self,
        t0: Real[Any, ""],
        t1: Real[Any, ""] | None = None,
        left: bool = True,
        use_levy: bool = False,
    ) -> tuple[Real[Any, ""], Float[Any, "x"]]:
        """
        Evaluates the path at the given time points.

        Parameters
        ----------
        t0 : Real[Any, ""]
            The initial time.
        t1 : Real[Any, ""], optional
            The final time, defaults to None.
        left : bool, optional
            Whether to use the left limit, defaults to True.
        use_levy : bool, optional
            Whether to use the Levy area, defaults to False.

        Returns
        -------
        (Real[Any, ""], Float[Any, "x"])
            The evaluated control.
        """
        return t1 - t0, self.brownian_motion.evaluate(t0=t0, t1=t1, left=left, use_levy=use_levy)


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
        dynamics: Callable[[Real[Any, ""], PyTree, PyTree], PyTree],
        args: PyTree,
        x0: Location,
        ts: Real[Any, "time"],
        solver: dfx.AbstractSolver = dfx.Heun(),
        dt0: Real[Any, ""] = None,
        saveat: dfx.SaveAt | None = None,
        stepsize_controller: dfx.StepsizeController = dfx.ConstantStepSize(),
        adjoint: dfx.Adjoint = dfx.ForwardMode(),
        max_steps: Int[Any, ""] = 4096,
        n_samples: Int[Any, ""] = 100,
        key: Key[Array, ""] = jrd.key(0),
        brownian_path: Callable[[tuple[int, ...], Key[Array, ""]], dfx.AbstractBrownianPath] | None = None,
    ) -> TrajectoryEnsemble:
        r"""
        Simulates a [`pastax.trajectory.TrajectoryEnsemble`][] based on the initial [`pastax.trajectory.Location`][]
        and time steps (including t0).

        Parameters
        ----------
        dynamics : Callable[[Real[Any, ""], PyTree, PyTree], PyTree]
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
            t : Real[Any, ""]
                The current time.
            y : PyTree
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
        ts : Real[Any, "time"]
            The time steps for the simulation outputs, including $t_0$, unit should be the same as for `dt0`.
        solver : dfx.AbstractSolver, optional
            The [`diffrax.AbstractSolver`][] to use for the simulation, defaults to [`diffrax.Heun`][].
        dt0 : Real[Any, ""], optional
            The initial time step of the solver, unit should be the same as for `ts`, defaults to `None`.
        saveat : dfx.SaveAt, optional
            The [`diffrax.SaveAt`][] object to use for saving the solution, defaults to `SaveAt(ts=ts)`.
        stepsize_controller : dfx.StepsizeController, optional
            The [`diffrax.StepsizeController`][] to use for controlling the stepsize,
            defaults to [`diffrax.ConstantStepSize`][].
        adjoint : dfx.Adjoint, optional
            The [`diffrax.Adjoint`][] object to use for the adjoint method, defaults to [`diffrax.ForwardMode`][].
            [`diffrax.ForwardMode`][] should be used when computing the gradient in forward automtic differentiation
            mode with respect to few (<50) parameters, while [`diffrax.RecursiveCheckpointAdjoint`][] should be used
            when computing the gradient in reverse automatic differentiation mode with respect to many (>50) parameters.
        max_steps : Int[Any, ""], optional
            The maximum number of steps to take, defaults to `4096`.
        n_samples : Int[Any, ""], optional
            The number of samples to generate, defaults to `100`.
        key : Key[Array, ""], optional
            The random key for sampling, defaults to `jrd.key(0)`.
        brownian_path : Callable[[tuple[int, ...], Key[Array, ""]], dfx.AbstractBrownianPath] | None, optional
            The [`diffrax.AbstractBrownianPath`][] to use for the simulation, defaults to `None`.

        Returns
        -------
        TrajectoryEnsemble
            The simulated [`pastax.trajectory.TrajectoryEnsemble`][].
        """
        t0, t1 = ts[0], ts[-1]

        if saveat is None:
            saveat = dfx.SaveAt(ts=ts)

        if brownian_path is None:
            brownian_path = lambda shape, key: dfx.VirtualBrownianTree(t0, t1, tol=dt0, shape=shape, key=key)

        keys = jrd.split(key, n_samples)

        @jax.vmap
        def solve(subkey: Array) -> Float[Array, "time 2"]:
            sde_control = SDEControl(t0=t0, t1=t1, brownian_motion=brownian_path((2,), subkey))
            sde_term = dfx.ControlTerm(dynamics, sde_control)

            ys = dfx.diffeqsolve(
                sde_term,
                solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=x0.value,
                args=args,
                saveat=saveat,
                stepsize_controller=stepsize_controller,
                adjoint=adjoint,
                max_steps=max_steps,
            ).ys

            return ys  # type: ignore

        ys = solve(keys)

        return TrajectoryEnsemble.from_array(ys, ts, unit=x0.unit)
