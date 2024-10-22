from __future__ import annotations
from typing import List

import equinox as eqx
import interpax as ipx
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from ...grid import Dataset, spatial_derivative
from ...utils import meters_to_degrees
from .._diffrax_simulator import StochasticDiffrax


class SmagorinskyDiffusionCVF(eqx.Module):
    """
    Attributes
    ----------
    cs : Float[Scalar, ""], optional
        The Smagorinsky constant, defaults to 0.1.

    Methods
    -------
    _neighborhood(*variables, t, y, dataset)
        Restricts the dataset to a neighborhood around the given location and time.
    _smagorinsky_coefficients(t, y, dataset)
        Computes the Smagorinsky coefficients.
    _drift_term(t, y, dataset, smag_ds)
        Computes the drift term of the Stochastic Differential Equation.
    _diffusion_term(y, smag_ds)
        Computes the diffusion term of the Stochastic Differential Equation.
    __call__(t, y, args)
        Computes the drift and diffusion terms of the Stochastic Differential Equation.

    Notes
    -----
    As the class inherits from `eqx.Module`, its `cs` attribute can be treated as a trainable parameter.
    """

    cs: Float[Scalar, ""] = eqx.field(converter=lambda x: jnp.asarray(x, dtype=float), default_factory=lambda: 0.1)

    @staticmethod
    def _neighborhood(*variables: List[str], t: Float[Scalar, ""], y: Float[Array, "2"], dataset: Dataset) -> Dataset:
        """
        Restricts the dataset to a neighborhood around the given location and time.

        Parameters
        ----------
        *variables : List[str]
            The variables to retain in the neighborhood.
        t : Float[Scalar, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        dataset : Dataset
            The dataset containing the physical fields.

        Returns
        -------
        Dataset
            The neighborhood dataset.
        """
        # restrict dataset to the neighborhood around X(t)
        neighborhood = dataset.neighborhood(
            *variables,
            time=t, latitude=y[0], longitude=y[1],
            t_width=3, x_width=7
        )  # "x_width x_width"

        return neighborhood

    def _smagorinsky_coefficients(self, t: Float[Scalar, ""], y: Float[Array, "2"], dataset: Dataset) -> Dataset:
        """
        Computes the Smagorinsky coefficients.

        Parameters
        ----------
        t : Float[Scalar, ""]
            The simulation time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        dataset : Dataset
            The dataset containing the physical fields.

        Returns
        -------
        Dataset
            The dataset containing the Smagorinsky coefficients.

        Notes
        -----
        The physical fields are first restricted to a small neighborhood, then interpolated in time and finally spatial derivatives are computed using finite central difference.
        """
        neighborhood = self._neighborhood("u", "v", t=t, y=y, dataset=dataset)

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
            is_spherical_mesh=neighborhood.is_spherical_mesh,
            is_uv_mps=neighborhood.is_uv_mps
        )

        return smag_ds

    @staticmethod
    def _drift_term(
        t: Float[Scalar, ""],
        y: Float[Array, "2"],
        dataset: Dataset,
        smag_ds: Dataset
    ) -> Float[Array, "2"]:
        """
        Computes the drift term of the Stochastic Differential Equation.

        Parameters
        ----------
        t : Float[Scalar, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        dataset : Dataset
            The dataset containing the physical fields.
        smag_ds : Dataset
            The dataset containing the Smagorinsky coefficients for the given fields.

        Returns
        -------
        Float[Array, "2"]
            The drift term (change in latitude and longitude).
        """
        latitude, longitude = y[0], y[1]

        smag_k = jnp.squeeze(smag_ds.variables["smag_k"].values)  # "x_width-2 x_width-2"

        # $\mathbf{u}(t, \mathbf{X}(t))$ term
        u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=latitude, longitude=longitude)
        vu = jnp.asarray([v, u], dtype=float)  # "2"

        if smag_ds.is_spherical_mesh:
            longitude += 180

        # $(\nabla \cdot \mathbf{K})(t, \mathbf{X}(t))$ term
        dkdx, dkdy = spatial_derivative(
            smag_k, dx=smag_ds.dx, dy=smag_ds.dy, is_land=smag_ds.is_land
        )  # "x_width-4 x_width-4"
        dkdx = ipx.interp2d(
            latitude, longitude,
            smag_ds.coordinates.latitude[1:-1], smag_ds.coordinates.longitude.values[1:-1],
            dkdx,
            method="linear",
            extrap=True
        )
        dkdy = ipx.interp2d(
            latitude, longitude,
            smag_ds.coordinates.latitude[1:-1], smag_ds.coordinates.longitude[1:-1],
            dkdy,
            method="linear",
            extrap=True
        )
        gradk = jnp.asarray([dkdy, dkdx], dtype=float)  # "2"

        return vu + gradk

    @staticmethod
    def _diffusion_term(y: Float[Array, "2"], smag_ds: Dataset) -> Float[Array, "2 2"]:
        """
        Computes the diffusion term of the Stochastic Differential Equation.

        Parameters
        ----------
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        smag_ds : Dataset
            The dataset containing the Smagorinsky coefficients.

        Returns
        -------
        Float[Array, "2 2"]
            The diffusion term.
        """
        latitude, longitude = y[0], y[1]

        smag_k = smag_ds.interp_spatial("smag_k", latitude=latitude, longitude=longitude)[0]
        smag_k = jnp.squeeze(smag_k)  # scalar
        smag_k = (2 * smag_k) ** (1 / 2)

        return jnp.eye(2) * smag_k

    def __call__(self, t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2 3"]:
        """
        Computes the drift and diffusion terms of the Stochastic Differential Equation.

        Parameters
        ----------
        t : float
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Dataset
            The dataset containing the velocity fields.

        Returns
        -------
        Float[Array, "2 3"]
            The stacked drift and diffusion terms.
        """
        t = jnp.asarray(t)
        dataset = args

        neighborhood = self._neighborhood("u", "v", t=t, y=y, dataset=dataset)
        smag_ds = self._smagorinsky_coefficients(t, y, dataset)  # "1 x_width-2 x_width-2"

        dlatlon_drift = self._drift_term(t, y, dataset, smag_ds)
        dlatlon_diffusion = self._diffusion_term(y, smag_ds)

        dlatlon = jnp.column_stack([dlatlon_drift, dlatlon_diffusion])

        if dataset.is_spherical_mesh and dataset.is_uv_mps:
            dlatlon = meters_to_degrees(dlatlon.T, latitude=y[0]).T

        return dlatlon


class SmagorinskyDiffusion(StochasticDiffrax):
    """
    Stochastic simulator using Smagorinsky diffusion.
 
    Attributes
    ----------
    sde_cvf : SmagorinskyDiffusionCVF
        Computes the drift and diffusion terms of the Smagorinsky diffusion SDE.
    id : str
        The identifier for the SmagorinskyDiffrax model (set to "smagorinsky_diffusion").

    Methods
    -------
    from_param(cs)
        Creates a SmagorinskyDiffusion simulator with the given Smagorinsky constant.

    Notes
    -----
    In this example, the `sde_cvf` attribute is an `eqx.Module` with the Smagorinsky constant as attribute, allowing to treat it as a trainable parameter.
    """

    id: str = eqx.field(static=True, default_factory=lambda: "smagorinsky_diffusion")
    sde_cvf: SmagorinskyDiffusionCVF = SmagorinskyDiffusionCVF()

    @classmethod
    def from_param(cls, cs: Float[Array, ""] = None, id: str = None) -> SmagorinskyDiffusion:
        """
        Creates a SmagorinskyDiffusion simulator with the given Smagorinsky constant.

        Parameters
        ----------
        cs : Float[Array, ""], optional
            The Smagorinsky constant, defaults to None.
        id : str, optional
            The identifier for the simulator, defaults to None.

        Returns
        -------
        SmagorinskyDiffusion
            The SmagorinskyDiffusion simulator.

        Notes
        -----
        If any of the parameters is None, its default value is used.
        """
        sde_cvf_kwargs = {}
        if cs is not None:
            sde_cvf_kwargs["cs"] = cs

        self_kwargs = {}
        if id is not None:
            self_kwargs["id"] = id

        return cls(sde_cvf=SmagorinskyDiffusionCVF(**sde_cvf_kwargs), **self_kwargs)
