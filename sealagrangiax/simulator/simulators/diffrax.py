from __future__ import annotations
from typing import Dict, Tuple

import diffrax as dfx
import equinox as eqx
import interpax as inx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float, Int
import lineax as lnx

from ...grid import Dataset, spatial_derivative
from ...timeseries import Displacement, Location, Trajectory, TrajectoryEnsemble
from ...utils import UNIT
from .._simulator import Simulator


class Diffrax(Simulator):
    """
    A class used to simulate trajectories using diffrax library.

    Methods
    -------
    _transform_times(t0, ts)
        Transforms the initial time and time steps for the simulation.
    """

    @staticmethod
    @jax.jit
    def _transform_times(
            t0: Int[Array, ""], ts: Int[Array, "time-1"]
    ) -> Tuple[Float[Array, "time"], Int[Array, ""]]:
        """
        Transforms the initial time and time steps for the simulation.

        Parameters
        ----------
        t0 : Int[Array, ""]
            The initial time.
        ts : Int[Array, "time-1"]
            The time steps for the simulation.

        Returns
        -------
        Tuple[Float[Array, "time"], Int[Array, ""]]
            The transformed time steps and the final time.
        """
        ts = jnp.pad(ts, (1, 0), constant_values=t0)
        t1 = ts[-1]

        return ts, t1


class DeterministicDiffrax(Diffrax):
    """
    A class used to simulate deterministic trajectories using diffrax library.

    Methods
    -------
    drift_term(t, y, args)
        Computes the drift term for the differential equation.
    solve(x0, t0, ts, dt0=30*60, solver=dfx.Heun())
        Solves the differential equation to simulate the trajectory.
    __call__(x0, t0, ts, dt0=30*60, solver=dfx.Heun(), n_samples=None, key=None)
        Simulates the trajectory based on the initial location, time, and time steps.
    """

    @staticmethod
    @eqx.filter_jit
    def drift_term(t: int, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
        """
        Computes the drift term for the differential equation.

        Parameters
        ----------
        t : int
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Dataset
            The dataset containing the physical fields (only u and v here).

        Returns
        -------
        Float[Array, "2"]
            The drift term (change in latitude and longitude).
        """
        t = jnp.asarray(t)
        dataset = args
        x = Location(y)

        u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=x.latitude, longitude=x.longitude)  # m/s
        vu = jnp.asarray([v, u])  # scalars

        dlatlon = Displacement(vu, UNIT.meters)  # m/s
        dlatlon = dlatlon.convert_to(UNIT.degrees, x.latitude)  # °/s

        return dlatlon

    @eqx.filter_jit
    def solve(
        self,
        x0: Location,
        t0: Int[Array, ""],
        ts: Int[Array, "time-1"],
        dt0: int,  # 30 minutes in seconds
        solver: dfx.AbstractSolver
    ) -> Float[Array, "time 2"]:
        """
        Solves the differential equation to simulate the trajectory.

        Parameters
        ----------
        x0 : Location
            The initial location.
        t0 : Int[Array, ""]
            The initial time.
        ts : Int[Array, "time-1"]
            The time steps for the simulation outputs.
        dt0 : int
            The initial time step in seconds.
        solver : dfx.AbstractSolver
            The solver function to use for the simulation.

        Returns
        -------
        Float[Array, "time 2"]
            The simulated trajectory (latitude and longitude) at each time step.
        """
        ts, t1 = self._transform_times(t0, ts)

        return dfx.diffeqsolve(
            dfx.ODETerm(self.drift_term),
            solver,
            t0=t0,
            t1=t1,
            dt0=dt0,
            y0=x0.value,
            args=self.datasets,
            saveat=dfx.SaveAt(ts=ts)
        ).ys

    def __call__(
        self,
        x0: Location,
        t0: Int[Array, ""],
        ts: Int[Array, "time-1"],
        dt0: int = 30 * 60,  # 30 minutes in seconds
        solver: dfx.AbstractSolver = dfx.Heun(),
        n_samples: int = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory:
        """
        Simulates the trajectory based on the initial location, time, and time steps.

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
        ys = self.solve(x0, t0, ts, dt0, solver)
        return Trajectory(ys, ts)


class SDEControl(dfx.AbstractPath):
    """
    A class used to represent a diffrax control for Stochastic Differential Equation (SDE).

    Attributes
    ----------
    t0 : int
        The initial time.
    t1 : int
        The final time.
    brownian_motion : dfx.VirtualBrownianTree
        The Brownian motion used in the SDE simulation.

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


class StochasticDiffrax(Diffrax):
    """
    A class used to simulate stochastic ensembles of trajectories using diffrax library.

    Methods
    -------
    drift_and_diffusion_term(t, y, args)
        Computes the drift and diffusion terms for the stochastic differential equation.
    solve(x0, t0, ts, dt0, solver, n_samples, key)
        Solves the stochastic differential equation to simulate the trajectory.
    _neighborhood(t, x, dataset)
        Restricts the dataset to the neighborhood around the given location and time.
    __call__(x0, t0, ts, dt0=30*60, solver=dfx.Heun(), n_samples=100, key=jrd.PRNGKey(0))
        Simulates the trajectory based on the initial location, time, and time steps.
    """

    @staticmethod
    @eqx.filter_jit
    def drift_and_diffusion_term(
        t: int, 
        y: Float[Array, "2"], 
        args: Dict[str, Dataset] | Dataset
    ) -> lnx.PyTreeLinearOperator:
        """
        Computes the drift and diffusion terms of the SDE.

        Parameters
        ----------
        t : int
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Dataset
            The dataset(s) containing the physical fields.

        Returns
        -------
        lnx.PyTreeLinearOperator
            The drift and diffusion terms.
        """
        raise NotImplementedError

    @eqx.filter_jit
    def solve(
        self,
        x0: Location,
        t0: Int[Array, ""],
        ts: Int[Array, "time-1"],
        dt0: int,  # 30 minutes in seconds
        solver: dfx.AbstractSolver,
        n_samples: int,
        key: jrd.PRNGKey
    ) -> Float[Array, "member time 2"]:
        """
        Solves the stochastic differential equation to simulate the trajectory.

        Parameters
        ----------
        x0 : Location
            The initial location.
        t0 : Int[Array, ""]
            The initial time.
        ts : Int[Array, "time-1"]
            The time steps for the simulation outputs.
        dt0 : int
            The initial time step in seconds.
        solver : dfx.AbstractSolver
            The solver function to use for the simulation.
        n_samples : int
            The number of samples to generate.
        key : jrd.PRNGKey
            The random key for sampling.

        Returns
        -------
        Float[Array, "member time 2"]
            The simulated trajectory (latitude and longitude) at each time step.
        """
        y0 = x0.value
        ts, t1 = self._transform_times(t0, ts)

        keys = jrd.split(key, n_samples)

        @jax.vmap
        def _solve(subkey: jrd.PRNGKey) -> Float[Array, "time 2"]:
            eps = 1e-3
            brownian_motion = dfx.VirtualBrownianTree(t0, t1 + eps, tol=eps, shape=(2,), key=subkey)
            sde_control = SDEControl(brownian_motion=brownian_motion)
            sde_term = dfx.ControlTerm(self.drift_and_diffusion_term, sde_control)

            return dfx.diffeqsolve(
                sde_term,
                solver,
                t0=t0,
                t1=t1,
                dt0=dt0,
                y0=y0,
                args=self.datasets,
                saveat=dfx.SaveAt(ts=ts)
            ).ys

        return _solve(keys)

    @staticmethod
    @eqx.filter_jit
    def _neighborhood(t: Int[Array, ""], x: Location, dataset: Dataset) -> Dataset:
        """
        Restricts the dataset to the neighborhood around the given location and time.

        Parameters
        ----------
        t : Int[Array, ""]
            The current time.
        x : Location
            The current location.
        dataset : Dataset
            The dataset containing the physical fields.

        Returns
        -------
        Dataset
            The neighborhood dataset.
        """
        # restrict dataset to the neighborhood around X(t)
        neighborhood = dataset.neighborhood(
            "u", "v",
            time=t, latitude=x.latitude, longitude=x.longitude,
            t_width=2, x_width=7
        )  # "x_width x_width"

        return neighborhood

    def __call__(
        self,
        x0: Location,
        t0: Int[Array, ""],
        ts: Int[Array, "time-1"],
        dt0: int = 30 * 60,  # 30 minutes in seconds
        solver: dfx.AbstractSolver = dfx.Heun(),
        n_samples: int = 100,
        key: jrd.PRNGKey = jrd.PRNGKey(0)
    ) -> TrajectoryEnsemble:
        """
        Simulates the trajectory based on the initial location, time, and time steps.

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
        ys = self.solve(x0, t0, ts, dt0, solver, n_samples, key)

        return TrajectoryEnsemble(ys, ts)


class SmagorinskyDiffusion(StochasticDiffrax):
    """
    A class used to simulate stochastic trajectories using the Smagorinsky model with the Diffrax library.

    Attributes
    ----------
    id : str
        The identifier for the SmagorinskyDiffrax model.

    Methods
    -------
    drift_and_diffusion_term(t, y, args)
        Computes the drift and diffusion terms of the SDE.
    _drift_term(t, y, neighborhood, smag_ds)
        Computes the drift term of the SDE.
    _diffusion_term(y, smag_ds)
        Computes the diffusion term of the SDE.
    _smagorinsky_coefficients(t, neighborhood)
        Computes the Smagorinsky coefficients.
    """

    id: str = "smagorinsky_diffusion"

    @staticmethod
    @eqx.filter_jit
    def drift_and_diffusion_term(t: int, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2 3"]:
        """
        Computes the drift and diffusion terms of the SDE.

        Parameters
        ----------
        t : int
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Dataset
            The dataset containing the physical fields.

        Returns
        -------
        Float[Array, "2 3"]
            The drift and diffusion terms.
        """
        t = jnp.asarray(t)
        dataset = args
        x = Location(y)

        neighborhood = SmagorinskyDiffusion._neighborhood(t, x, dataset)
        smag_ds = SmagorinskyDiffusion._smagorinsky_coefficients(t, neighborhood)  # m^2/s "1 x_width-2 x_width-2"

        dlatlon_drift = SmagorinskyDiffusion._drift_term(t, y, neighborhood, smag_ds)
        dlatlon_diffusion = SmagorinskyDiffusion._diffusion_term(y, smag_ds)

        dlatlon = jnp.column_stack([dlatlon_drift, dlatlon_diffusion])

        return dlatlon

    @staticmethod
    @eqx.filter_jit
    def _drift_term(
        t: Float[Array, ""],
        y: Float[Array, "2"],
        neighborhood: Dataset,
        smag_ds: Dataset
    ) -> Float[Array, "2"]:
        """
        Computes the drift term of the SDE.

        Parameters
        ----------
        t : Float[Array, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        neighborhood : Dataset
            The dataset containing the physical fields in the neighborhood.
        smag_ds : Dataset
            The dataset containing the Smagorinsky coefficients for the given fields.

        Returns
        -------
        Float[Array, "2"]
            The drift term (change in latitude and longitude).
        """
        x = Location(y)

        smag_k = jnp.squeeze(smag_ds.variables["smag_k"].values)  # "x_width-2 x_width-2"

        # $\mathbf{u}(t, \mathbf{X}(t))$ term
        u, v = neighborhood.interp_spatiotemporal("u", "v", time=t, latitude=x.latitude, longitude=x.longitude)  # m/s
        vu = jnp.asarray([v, u])  # "2"

        # $(\nabla \cdot \mathbf{K})(t, \mathbf{X}(t))$ term
        dkdx, dkdy = spatial_derivative(
            smag_k, dx=smag_ds.dx, dy=smag_ds.dy, is_land=smag_ds.is_land
        )  # m/s - "x_width-4 x_width-4"
        dkdx = inx.interp2d(
            x.latitude, x.longitude + 180,
            smag_ds.coordinates.latitude[1:-1], smag_ds.coordinates.longitude[1:-1],
            dkdx,
            method="linear"
        )
        dkdy = inx.interp2d(
            x.latitude, x.longitude + 180,
            smag_ds.coordinates.latitude[1:-1], smag_ds.coordinates.longitude[1:-1],
            dkdy,
            method="linear"
        )
        gradk = jnp.asarray([dkdy, dkdx])  # "2"

        dlatlon = Displacement(vu + gradk, UNIT.meters)  # m/s
        dlatlon = dlatlon.convert_to(UNIT.degrees, x.latitude)  # °/s

        return dlatlon

    @staticmethod
    @eqx.filter_jit
    def _diffusion_term(y: Float[Array, "2"], smag_ds: Dataset) -> Float[Array, "2 2"]:
        """
        Computes the diffusion term of the SDE.

        Parameters
        ----------
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        smag_ds : Dataset
            The dataset containing the Smagorinsky coefficients.

        Returns
        -------
        Float[Array, "2 2"]
            The diffusion term (change in latitude and longitude).
        """
        x = Location(y)

        smag_k = smag_ds.interp_spatial("smag_k", latitude=x.latitude, longitude=x.longitude)[0]  # m^2/s
        smag_k = jnp.squeeze(smag_k)  # scalar
        smag_k = (2 * smag_k) ** (1 / 2)  # m/s^(1/2)

        dlatlon = Displacement(jnp.full(2, smag_k), UNIT.meters)
        dlatlon = dlatlon.convert_to(UNIT.degrees, x.latitude)  # °/s^(1/2)

        return jnp.eye(2) * dlatlon

    @staticmethod
    @eqx.filter_jit
    def _smagorinsky_coefficients(t: Int[Array, ""], neighborhood: Dataset) -> Dataset:
        """
        Computes the Smagorinsky coefficients for the given fields.

        Parameters
        ----------
        t : Int[Array, ""]
            The simulation time.
        neighborhood : Dataset
            The dataset containing the physical fields.

        Returns
        -------
        Dataset
            The dataset containing the Smagorinsky coefficients.

        Notes
        -----
        The physical fields are first interpolated in time and then spatial derivatives are computed using finite central difference.
        """
        u, v = neighborhood.interp_temporal("u", "v", time=t)  # "x_width x_width"
        dudx, dudy, dvdx, dvdy = spatial_derivative(
            u, v, dx=neighborhood.dx, dy=neighborhood.dy, is_land=neighborhood.is_land
        )  # "x_width-2 x_width-2"

        # computes Smagorinsky coefficients
        smag_c = .1
        cell_area = neighborhood.cell_area[1:-1, 1:-1]  # "x_width-2 x_width-2"
        smag_k = smag_c * cell_area * ((dudx ** 2 + dvdy ** 2 + 0.5 * (dudy + dvdx) ** 2) ** (1 / 2))

        smag_ds = Dataset.from_arrays(
            {"smag_k": smag_k[None, ...]},
            time=t[None],
            latitude=neighborhood.coordinates.latitude.values[1:-1],
            longitude=neighborhood.coordinates.longitude.values[1:-1],
            interpolation_method="linear"
        )

        return smag_ds
