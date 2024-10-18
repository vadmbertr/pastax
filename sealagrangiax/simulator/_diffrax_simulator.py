from __future__ import annotations
from typing import Callable

import diffrax as dfx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float, Int, PyTree

from ..trajectory import Location, Trajectory, TrajectoryEnsemble
from ._simulator import Simulator


class DiffraxSimulator(Simulator):
    """
    Base class for defining differentiable trajectories or ensembles of trajectories simulators using diffrax library.

    Methods
    -------
    __call__(x0, ts, dt0, solver=dfx.Heun(), n_samples=None, key=None)
        Simulates trajectories or ensembles of trajectories based on the initial location and time steps (including t0).
    """

    def __call__(
        self,
        args: PyTree,
        x0: Location,
        ts: Int[Array, "time"],
        dt0: int,
        solver: dfx.AbstractSolver = dfx.Heun(),
        n_samples: int = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory:
        """
        Simulates trajectories or ensembles of trajectories based on the initial location and time steps (including t0).

        Parameters
        ----------
        args : PyTree
            Any PyTree of argument(s) used by the simulator.
            Could be for example one or several `sealagrangiax.Dataset` of gridded physical fields (SSC, SSH, SST, etc.).
        x0 : Location
            The initial location.
        ts : Int[Array, "time"]
            The time steps for the simulation outputs.
        dt0 : int
            The initial time step of the solver, in seconds.
        solver : dfx.AbstractSolver, optional
            The solver function to use for the simulation (default is dfx.Heun()).
        n_samples : int, optional
            The number of samples to generate (default is None, meaning a single trajectory).
        key : jrd.PRNGKey, optional
            The random key for sampling (default is None, useless for the deterministic simulator).

        Returns
        -------
        Trajectory
            The simulated trajectory or ensemble of trajectories, including the initial conditions (x0, t0).

        Raises
        ------
        NotImplementedError
            If the method is not implemented by the subclass.
        """
        raise NotImplementedError


class DeterministicDiffrax(DiffraxSimulator):
    """
    Base class for defining deterministic differentiable trajectories simulators using diffrax library.

    Attributes
    ----------
    ode_vf : Callable[[int, Float[Array, "2"], PyTree], PyTree]
        Any Callable (including another Equinox module with a __call__ method) that returns the drift term 
        of the solved Ordinary Differential Equation.
        
        Parameters
        ----------
        t : int
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        args : PyTree
            Any PyTree of argument(s) used by the simulator.
            Could be for example one or several `sealagrangiax.Dataset` of gridded physical fields (SSC, SSH, SST, etc.).

        Returns
        -------
        PyTree
            The stacked drift and diffusion terms.

    Methods
    -------
    drift_term(t, y, args)
        Computes the drift term of the solved Ordinary Differential Equation.
    solve(x0, t0, ts, dt0, solver=dfx.Heun())
        Solves an Ordinary Differential Equation simulating the trajectory.
    __call__(x0, ts, dt0, solver=dfx.Heun(), n_samples=None, key=None)
        Simulates the trajectory based on the initial location and time steps (including t0).
    """

    ode_vf: Callable[[int, Float[Array, "2"], PyTree], PyTree]

    def __call__(
        self,
        args: PyTree,
        x0: Location,
        ts: Int[Array, "time"],
        dt0: int,
        solver: dfx.AbstractSolver = dfx.Heun(),
        n_samples: int = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory:
        """
        Simulates the trajectory based on the initial location and time steps (including t0).

        Parameters
        ----------
        args : PyTree
            Any PyTree of argument(s) used by the simulator.
            Could be for example one or several `sealagrangiax.Dataset` of gridded physical fields (SSC, SSH, SST, etc.).
        x0 : Location
            The initial location.
        ts : Int[Array, "time"]
            The time steps for the simulation outputs.
        dt0 : int
            The initial time step of the solver, in seconds.
        solver : dfx.AbstractSolver, optional
            The solver function to use for the simulation (default is dfx.Heun()).
        n_samples : int, optional
            The number of samples to generate (default is None, meaning a single trajectory).
        key : jrd.PRNGKey, optional
            The random key for sampling (default is None, useless for the deterministic simulator).

        Returns
        -------
        Trajectory
            The simulated trajectory, including the initial conditions (x0, t0).
        """

        t0 = ts[0]
        t1 = ts[-1]

        ys = dfx.diffeqsolve(
            dfx.ODETerm(self.ode_vf),
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=x0.value,
            args=args,
            saveat=dfx.SaveAt(ts=ts)
        ).ys

        return Trajectory.from_array(ys, ts)


class SDEControl(dfx.AbstractPath):
    """
    Class representing a diffrax control for Stochastic Differential Equation.

    Attributes
    ----------
    t0 : int
        The initial time.
    t1 : int
        The final time.
    brownian_motion : dfx.VirtualBrownianTree
        The Brownian motion used in the Stochastic Differential Equation.

    Methods
    -------
    evaluate(t0, t1=None, left=True, use_levy=False)
        Evaluates the control at the given time points.
    """

    t0 = None
    t1 = None
    brownian_motion: dfx.VirtualBrownianTree

    def evaluate(self, t0: int, t1: int = None, left: bool = True, use_levy: bool = False) -> Float[Array, "x+1"]:
        """
        Evaluates the control at the given time points.

        Parameters
        ----------
        t0 : int
            The initial time.
        t1 : int, optional
            The final time (default is None).
        left : bool, optional
            Whether to use the left limit (default is True).
        use_levy : bool, optional
            Whether to use the Levy area (default is False).

        Returns
        -------
        Float[Array, "x+1"]
            The evaluated control.
        """
        return jnp.concatenate(
            [jnp.asarray([t1 - t0]), self.brownian_motion.evaluate(t0=t0, t1=t1, left=left, use_levy=use_levy)]
        )


class StochasticDiffrax(DiffraxSimulator):
    """
    Base class for defining stochastic differentiable trajectory ensembles simulators using diffrax library.

    Attributes
    ----------
    sde_cvf : Callable[[int, Float[Array, "2"], PyTree], lnx.PyTreeLinearOperator]
        Any Callable (including another Equinox module with a __call__ method) that stacks and returns 
        the drift and diffusion terms of the solved Stochastic Differential Equation.
        
        Parameters
        ----------
        t : int
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        args : Dataset
            Any PyTree of argument(s) used by the simulator.
            Could be for example one or several `sealagrangiax.Dataset` of gridded physical fields (SSC, SSH, SST, etc.).

        Returns
        -------
        PyTree
            The stacked drift and diffusion terms.

    Methods
    -------
    __call__(args, x0, ts, dt0, solver=dfx.Heun(), n_samples=100, key=jrd.PRNGKey(0))
        Simulates the trajectory ensemble based on the initial location and time steps (including t0).
    """

    sde_cvf: Callable[[int, Float[Array, "2"], PyTree], PyTree]

    def __call__(
        self,
        args: PyTree,
        x0: Location,
        ts: Int[Array, "time"],
        dt0: int,
        solver: dfx.AbstractSolver = dfx.Heun(),
        n_samples: int = 100,
        key: jrd.PRNGKey = jrd.PRNGKey(0)
    ) -> TrajectoryEnsemble:
        """
        Simulates the trajectory ensemble based on the initial location and time steps (including t0).

        Parameters
        ----------
        args : PyTree
            Any PyTree of argument(s) used by the simulator.
            Could be for example one or several `sealagrangiax.Dataset` of gridded physical fields (SSC, SSH, SST, etc.).
        x0 : Location
            The initial location.
        ts : Int[Array, "time"]
            The time steps for the simulation outputs (including t0).
        dt0 : int
            The initial time step of the solver, in seconds.
        solver : dfx.AbstractSolver, optional
            The solver function to use for the simulation (default is dfx.Heun()).
        n_samples : int, optional
            The number of samples to generate (default is 100).
        key : jrd.PRNGKey, optional
            The random key for sampling (default is jrd.PRNGKey(0)).

        Returns
        -------
        TrajectoryEnsemble
            The simulated ensemble of trajectories.
        """

        t0 = ts[0]  
        t1 = ts[-1]

        keys = jrd.split(key, n_samples)

        @jax.vmap
        def solve(subkey: jrd.PRNGKey) -> Float[Array, "time 2"]:
            eps = 1e-3
            brownian_motion = dfx.VirtualBrownianTree(t0, t1 + eps, tol=eps, shape=(2,), key=subkey)
            sde_control = SDEControl(brownian_motion=brownian_motion)
            sde_term = dfx.ControlTerm(self.sde_cvf, sde_control)

            return dfx.diffeqsolve(
                sde_term,
                solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=x0.value,
                args=args,
                saveat=dfx.SaveAt(ts=ts)
            ).ys

        ys = solve(keys)

        return TrajectoryEnsemble.from_array(ys, ts)
