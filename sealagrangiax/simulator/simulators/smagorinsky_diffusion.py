from __future__ import annotations

import equinox as eqx
import interpax as inx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ...grid import Dataset, spatial_derivative
from ...trajectory import Displacement, Location
from ...utils import UNIT
from .._diffrax_simulator import StochasticDiffrax


class SmagorinskyDiffusion(StochasticDiffrax):
    """
    Stochastic simulator using Smagorinsky diffusion.

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
