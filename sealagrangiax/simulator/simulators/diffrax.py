from __future__ import annotations
from typing import Callable, Dict

import diffrax as dfx
import equinox as eqx
import interpax as inx
import jax
import jax.numpy as jnp
import jax.random as jrd
from jaxtyping import Array, Float, Int
import lineax as lnx

from ...input import Field
from ...gridded import Dataset, derivative_spatial
from ...trajectory import Displacement, Location, Trajectory, TrajectoryEnsemble
from ...utils import UNIT
from .._simulator import Simulator


def wrap_solver(solver_class: Callable, aux_fn: Callable):
    class WrappedSolver(solver_class):
        def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
            aux = aux_fn(t0, y0, args)
            y, y_error, dense_info, solver_state, solver_result = super().step(
                terms,
                t0,
                t1,
                (y0, aux),
                args,
                solver_state,
                made_jump
            )  # maybe y needs to be of the form (y0, aux)
            return y, y_error, dense_info, solver_state, solver_result

    return WrappedSolver


class Diffrax(Simulator):
    dataset_variables: Dict[str, str]
    dataset_coordinates: Dict[str, str]
    dataset_interpolation_method: str
    solver: Callable

    def __init__(
        self,
        fields: Dict[str, Field],
        simulator_id: str,
        dataset_variables: Dict[str, str],
        dataset_coordinates: Dict[str, str],
        dataset_interpolation_method: str,
        solver: Callable
    ):
        super().__init__(fields, simulator_id)
        self.dataset_variables = dataset_variables
        self.dataset_coordinates = dataset_coordinates
        self.dataset_interpolation_method = dataset_interpolation_method
        self.solver = solver

    def _get_dataset(self, x0: Location, t0: Int[Array, ""], ts: Int[Array, "time-1"]) -> Dataset:
        min_time, max_time, min_point, max_point = self._get_minmax(x0, t0, ts)

        self.fields["ssc"].load_data(
            min_time.item(), max_time.item(),
            min_point.latitude.item(), max_point.latitude.item(),
            min_point.longitude.item(), max_point.longitude.item()
        )

        dataset = Dataset.from_xarray(
            self.fields["ssc"].dataset,
            variables=self.dataset_variables,
            coordinates=self.dataset_coordinates,
            interpolation_method=self.dataset_interpolation_method
        )

        return dataset

    @staticmethod
    @jax.jit
    def _transform_times(
            t0: Int[Array, ""], ts: Int[Array, "time-1"]
    ) -> (Float[Array, "time"], Int[Array, ""]):
        ts = jnp.pad(ts, (1, 0), constant_values=t0)
        t1 = ts[-1]

        return ts, t1


class DeterministicDiffrax(Diffrax):
    def __init__(
        self,
        fields: Dict[str, Field],
        simulator_id: str,
        dataset_variables: Dict[str, str],
        dataset_coordinates: Dict[str, str],
        dataset_interpolation_method: str,
        solver: Callable = dfx.Tsit5
    ):
        super().__init__(
            fields,
            simulator_id,
            dataset_variables,
            dataset_coordinates,
            dataset_interpolation_method,
            solver
        )

    @staticmethod
    @eqx.filter_jit
    def drift_term(t: int, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
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
        t1: Int[Array, ""],
        ts: Int[Array, "time"],
        dataset: Dataset
    ) -> Float[Array, "time"]:
        print("Simulating trajectory...")

        return dfx.diffeqsolve(
            dfx.ODETerm(self.drift_term),  # noqa
            self.solver(),
            t0=t0,
            t1=t1,
            dt0=5 * 60,  # 5 minutes in seconds
            y0=x0.value,
            args=dataset,
            saveat=dfx.SaveAt(ts=ts)
        ).ys

    def __call__(
        self,
        x0: Location,
        t0: Int[Array, ""],
        ts: Int[Array, "time-1"],
        n_samples: int = None,
        key: jrd.PRNGKey = None
    ) -> Trajectory:
        ts, t1 = self._transform_times(t0, ts)
        dataset = self._get_dataset(x0, t0, ts)

        ys = self.solve(x0, t0, t1, ts, dataset)

        return Trajectory(ys, ts)


class StochasticDiffrax(Diffrax):
    def __init__(
        self,
        fields: Dict[str, Field],
        simulator_id: str,
        dataset_variables: Dict[str, str],
        dataset_coordinates: Dict[str, str],
        dataset_interpolation_method: str,
        solver: Callable = dfx.StratonovichMilstein
    ):
        super().__init__(
            fields,
            simulator_id,
            dataset_variables,
            dataset_coordinates,
            dataset_interpolation_method,
            solver
        )

    @staticmethod
    @eqx.filter_jit
    def diffusion_term(t: int, y: Float[Array, "2"], args: Dataset) -> Float[Array, "4"] | lnx.DiagonalLinearOperator:
        raise NotImplementedError

    @staticmethod
    @eqx.filter_jit
    def drift_term(t: int, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
        raise NotImplementedError

    @eqx.filter_jit
    def solve(
        self,
        x0: Location,
        t0: Int[Array, ""],
        t1: Int[Array, ""],
        ts: Int[Array, "time"],
        n_samples: int,
        key: jrd.PRNGKey,
        dataset: Dataset
    ) -> Float[Array, "member time"]:
        print("Simulating trajectories...")

        y0 = x0.value
        keys = jrd.split(key, n_samples)

        @jax.vmap
        def _solve(subkey: jrd.PRNGKey) -> Float[Array, "time 2"]:
            eps = 1e-3
            brownian_motion = dfx.VirtualBrownianTree(t0, t1 + eps, tol=eps, shape=(2,), key=subkey)

            return dfx.diffeqsolve(
                dfx.MultiTerm(dfx.ODETerm(self.drift_term), dfx.ControlTerm(self.diffusion_term, brownian_motion)),  # noqa
                self.solver(),
                t0=t0,
                t1=t1,
                dt0=5 * 60,  # 5 minutes in seconds
                y0=y0,
                args=dataset,
                saveat=dfx.SaveAt(ts=ts)
            ).ys

        return _solve(keys)

    @staticmethod
    @eqx.filter_jit
    def _neighborhood(t: Int[Array, ""], x: Location, dataset: Dataset) -> Dataset:
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
        n_samples: int = 1,
        key: jrd.PRNGKey = None
    ) -> TrajectoryEnsemble:
        ts, t1 = self._transform_times(t0, ts)
        dataset = self._get_dataset(x0, t0, ts)

        ys = self.solve(x0, t0, t1, ts, n_samples, key, dataset)

        return TrajectoryEnsemble(ys, ts)


class SmagorinskyDiffrax(StochasticDiffrax):
    def __init__(
        self,
        fields: Dict[str, Field],
        simulator_id: str = "smagorinsky_diffrax",
        dataset_variables: Dict[str, str] = {"u": "u", "v": "v"},  # noqa
        dataset_coordinates: Dict[str, str] = {"time": "time", "latitude": "lat_t", "longitude": "lon_t"},  # noqa
        dataset_interpolation_method: str = "linear",
        solver: Callable = dfx.StratonovichMilstein
    ):
        super().__init__(
            fields,
            simulator_id,
            dataset_variables,
            dataset_coordinates,
            dataset_interpolation_method,
            wrap_solver(solver, SmagorinskyDiffrax.aux_fn)
        )

    @staticmethod
    @eqx.filter_jit
    def aux_fn(t: int, y: Float[Array, "2"], args: Dataset) -> (Dataset, Dataset):
        t = jnp.asarray(t)
        dataset = args
        x = Location(y)

        neighborhood = SmagorinskyDiffrax._neighborhood(t, x, dataset)
        smag_ds = SmagorinskyDiffrax._smagorinsky_coefficients(t, neighborhood)  # m^2/s "1 x_width-2 x_width-2"

        return neighborhood, smag_ds

    @staticmethod
    @eqx.filter_jit
    def diffusion_term(t: int, y: Float[Array, "2"], args: Dataset) -> lnx.DiagonalLinearOperator:
        y, (neighborhood, smag_ds) = y
        x = Location(y)

        smag_k = smag_ds.interp_spatial("smag_k", latitude=x.latitude, longitude=x.longitude)[0]  # m^2/s
        smag_k = jnp.squeeze(smag_k)  # scalar
        smag_k = (2 * smag_k) ** (1 / 2)  # m/s

        dlatlon = Displacement(jnp.full(2, smag_k), UNIT.meters)
        dlatlon = dlatlon.convert_to(UNIT.degrees, x.latitude)  # °/s

        return lnx.DiagonalLinearOperator(dlatlon)

    @staticmethod
    @eqx.filter_jit
    def drift_term(t: int, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
        t = jnp.asarray(t)
        y, (neighborhood, smag_ds) = y
        x = Location(y)

        smag_k = jnp.squeeze(smag_ds.variables["smag_k"].values)  # "x_width-2 x_width-2"

        # $\mathbf{u}(t, \mathbf{X}(t))$ term
        u, v = neighborhood.interp_spatiotemporal("u", "v", time=t, latitude=x.latitude, longitude=x.longitude)  # m/s
        vu = jnp.asarray([v, u])  # "2"

        # $(\nabla \cdot \mathbf{K})(t, \mathbf{X}(t))$ term
        dkdx, dkdy = derivative_spatial(
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

    # TODO: compute the derivatives "within the cell" using interpolated values?
    @staticmethod
    @eqx.filter_jit
    def _smagorinsky_coefficients(t: Int[Array, ""], neighborhood: Dataset) -> Dataset:
        u, v = neighborhood.interp_temporal("u", "v", time=t)  # "x_width x_width"
        dudx, dudy, dvdx, dvdy = derivative_spatial(
            u, v, dx=neighborhood.dx, dy=neighborhood.dy, is_land=neighborhood.is_land
        )  # "x_width-2 x_width-2"

        # computes Smagorinsky coefficients
        smag_c = .1
        cell_area = neighborhood.cell_area[1:-1, 1:-1]  # "x_width-2 x_width-2"
        smag_k = smag_c * cell_area * ((dudx ** 2 + dvdy ** 2 + 0.5 * (dudy + dvdx) ** 2) ** (1 / 2))

        smag_ds = Dataset.from_arrays(
            {"smag_k": smag_k[None, ...]},
            time=t[None],
            latitude=neighborhood.coordinates.latitude[1:-1],
            longitude=neighborhood.coordinates.longitude[1:-1] - 180,
            interpolation_method="linear"
        )

        return smag_ds
