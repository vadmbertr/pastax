from __future__ import annotations
from typing import List

import equinox as eqx
import interpax as ipx
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from ...grid import Dataset, spatial_derivative
from ...trajectory import Location
from .._diffrax_simulator import StochasticDiffrax


class SmagorinskyDiffusionCVF(eqx.Module):
    """
    Attributes
    ----------
    cs : Float[Scalar, ""], optional
        The Smagorinsky constant, defaults to 0.1.

    Methods
    -------
    _neighborhood(t, x, dataset, *variables)
        Restricts the dataset to a neighborhood around the given location and time.
    _smagorinsky_coefficients(t, neighborhood)
        Computes the Smagorinsky coefficients.
    _drift_term(t, y, neighborhood, smag_ds)
        Computes the drift term of the Stochastic Differential Equation.
    _diffusion_term(y, smag_ds)
        Computes the diffusion term of the Stochastic Differential Equation.
    __call__(t, y, args)
        Computes the drift and diffusion terms of the Stochastic Differential Equation.

    Notes
    -----
    As the class inherits from `eqx.Module`, its `cs` attribute can be treated as a trainable parameter.
    """

    cs: Float[Scalar, ""] = eqx.field(static=True, default_factory=lambda: 0.1)

    @staticmethod
    def _neighborhood(t: Float[Scalar, ""], x: Location, dataset: Dataset, *variables: List[str]) -> Dataset:
        """
        Restricts the dataset to a neighborhood around the given location and time.

        Parameters
        ----------
        t : Float[Scalar, ""]
            The current time.
        x : Location
            The current location.
        dataset : Dataset
            The dataset containing the physical fields.
        *variables : List[str]
            The variables to retain in the neighborhood.

        Returns
        -------
        Dataset
            The neighborhood dataset.
        """
        # restrict dataset to the neighborhood around X(t)
        neighborhood = dataset.neighborhood(
            *variables,
            time=t, latitude=x.latitude.value, longitude=x.longitude.value,
            t_width=2, x_width=7
        )  # "x_width x_width"

        return neighborhood

    def _smagorinsky_coefficients(self, t: Float[Scalar, ""], neighborhood: Dataset) -> Dataset:
        """
        Computes the Smagorinsky coefficients for the given fields.

        Parameters
        ----------
        t : Float[Scalar, ""]
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
        cell_area = neighborhood.cell_area[1:-1, 1:-1]  # "x_width-2 x_width-2"
        smag_k = self.cs * cell_area * ((dudx ** 2 + dvdy ** 2 + 0.5 * (dudy + dvdx) ** 2) ** (1 / 2))

        smag_ds = Dataset.from_arrays(
            {"smag_k": smag_k[None, ...]},
            time=t[None],
            latitude=neighborhood.coordinates.latitude.values[1:-1],
            longitude=neighborhood.coordinates.longitude.values[1:-1],
            interpolation_method="linear",
            convert_uv_to_dps=False
        )

        return smag_ds

    @staticmethod
    def _drift_term(
        t: Float[Scalar, ""],
        y: Float[Array, "2"],
        neighborhood: Dataset,
        smag_ds: Dataset
    ) -> Float[Array, "2"]:
        """
        Computes the drift term of the Stochastic Differential Equation.

        Parameters
        ----------
        t : Float[Scalar, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        neighborhood : Dataset
            The dataset containing the physical fields in the neighborhood.
        smag_ds : Dataset
            The dataset containing the Smagorinsky coefficients for the given fields.

        Returns
        -------
        Float[Array, "2"]
            The drift term (change in latitude and longitude in degrees).
        """
        latitude, longitude = y[0], y[1]

        smag_k = jnp.squeeze(smag_ds.variables["smag_k"].values)  # "x_width-2 x_width-2"

        # $\mathbf{u}(t, \mathbf{X}(t))$ term
        u, v = neighborhood.interp_spatiotemporal("u", "v", time=t, latitude=latitude, longitude=longitude)  # °/s
        vu = jnp.asarray([v, u])  # "2"

        # $(\nabla \cdot \mathbf{K})(t, \mathbf{X}(t))$ term
        dkdx, dkdy = spatial_derivative(
            smag_k, dx=smag_ds.dx, dy=smag_ds.dy, is_land=smag_ds.is_land
        )  # °/s - "x_width-4 x_width-4"
        dkdx = ipx.interp2d(
            latitude, longitude + 180,
            smag_ds.coordinates.latitude[1:-1], smag_ds.coordinates.longitude.values[1:-1],
            dkdx,
            method="linear",
            extrap=True
        )
        dkdy = ipx.interp2d(
            latitude, longitude + 180,
            smag_ds.coordinates.latitude[1:-1], smag_ds.coordinates.longitude[1:-1],
            dkdy,
            method="linear",
            extrap=True
        )
        gradk = jnp.asarray([dkdy, dkdx])  # "2"

        return vu + gradk  # °/s

    @staticmethod
    def _diffusion_term(y: Float[Array, "2"], smag_ds: Dataset) -> Float[Array, "2 2"]:
        """
        Computes the diffusion term of the Stochastic Differential Equation.

        Parameters
        ----------
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        smag_ds : Dataset
            The dataset containing the Smagorinsky coefficients.

        Returns
        -------
        Float[Array, "2 2"]
            The diffusion term.
        """
        latitude, longitude = y[0], y[1]

        smag_k = smag_ds.interp_spatial("smag_k", latitude=latitude, longitude=longitude)[0]  # °^2/s
        smag_k = jnp.squeeze(smag_k)  # scalar
        smag_k = (2 * smag_k) ** (1 / 2)  # °/s^(1/2)

        return jnp.eye(2) * smag_k

    def __call__(self, t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2 3"]:
        """
        Computes the drift and diffusion terms of the Stochastic Differential Equation.

        Parameters
        ----------
        t : float
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        args : Dataset
            The dataset containing the velocity fields.

        Returns
        -------
        Float[Array, "2 3"]
            The stacked drift and diffusion terms.
        """
        t = jnp.asarray(t)
        dataset = args

        neighborhood = self._neighborhood(t, Location(y), dataset, "u", "v")
        smag_ds = self._smagorinsky_coefficients(t, neighborhood)  # °^2/s "1 x_width-2 x_width-2"

        dlatlon_drift = self._drift_term(t, y, neighborhood, smag_ds)
        dlatlon_diffusion = self._diffusion_term(y, smag_ds)

        dlatlon = jnp.column_stack([dlatlon_drift, dlatlon_diffusion])

        return dlatlon  # °/s


class SmagorinskyDiffusion(StochasticDiffrax):
    """
    Stochastic simulator using Smagorinsky diffusion.
 
    Attributes
    ----------
    sde_cvf : SmagorinskyDiffusionCVF
        Computes the drift and diffusion terms of the Smagorinsky diffusion SDE.
    id : str
        The identifier for the SmagorinskyDiffrax model (set to "smagorinsky_diffusion").

    Notes
    -----
    In this example, the `sde_cvf` attribute is an `eqx.Module` with the Smagorinsky constant as attribute, allowing to treat it as a trainable parameter.
    """

    sde_cvf: SmagorinskyDiffusionCVF = SmagorinskyDiffusionCVF()
    id: str = eqx.field(static=True, default_factory=lambda: "smagorinsky_diffusion")
