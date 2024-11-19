from __future__ import annotations

import equinox as eqx
import interpax as ipx
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from ...grid import Dataset, spatial_derivative
from ...utils import meters_to_degrees


class SmagorinskyDiffusion(eqx.Module):
    r"""
    Trainable Smagorinsky diffusion dynamics.

    !!! example "Formulation"

        This dynamics allows to formulate a displacement at time $t$ from the position $\mathbf{X}(t)$ as:

        $$
        d\mathbf{X}(t) = (\mathbf{u} + \nabla K)(t, \mathbf{X}(t)) dt + V(t, \mathbf{X}(t)) d\mathbf{W}(t)
        $$

        where $V = \sqrt{2 K}$ and $K$ is the Smagorinsky diffusion:
        
        $$
        K = C_s \Delta x \Delta y \sqrt{\left(\frac{\partial u}{\partial x} \right)^2 + \left(\frac{\partial v}{\partial y} \right)^2 + \frac{1}{2} \left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right)^2}
        $$

        where $C_s$ is the ***trainable*** Smagorinsky constant, $\Delta x \Delta y$ a spatial scaling factor, and the rest of the expression represents the horizontal diffusion.

    Attributes
    ----------
    cs : Float[Scalar, ""], optional
        The Smagorinsky constant, defaults to `jnp.asarray(0.1)`.

    Methods
    -------
    _neighborhood(*variables, t, y, dataset)
        Restricts the [`pastax.grid.Dataset`][] to a neighborhood around the given location and time.
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
    As the class inherits from [`equinox.Module`][], its `cs` attribute can be treated as a trainable parameter.
    """

    cs: Float[Scalar, ""] = eqx.field(converter=lambda x: jnp.asarray(x, dtype=float), default_factory=lambda: 0.1)

    @staticmethod
    def _neighborhood(*variables: list[str], t: Float[Scalar, ""], y: Float[Array, "2"], dataset: Dataset) -> Dataset:
        """
        Restricts the [`pastax.grid.Dataset`][] to a neighborhood around the given location and time.

        Parameters
        ----------
        *variables : list[str]
            The variables to retain in the neighborhood.
        t : Float[Scalar, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        dataset : Dataset
            The [`pastax.grid.Dataset`][] containing the physical fields.

        Returns
        -------
        Dataset
            The neighborhood [`pastax.grid.Dataset`][].
        """
        # restrict dataset to the neighborhood around X(t)
        neighborhood = dataset.neighborhood(
            *variables,
            time=t, latitude=y[0], longitude=y[1],
            t_width=3, x_width=7
        )  # "x_width x_width"

        return neighborhood

    def _smagorinsky_diffusion(self, t: Float[Scalar, ""], y: Float[Array, "2"], dataset: Dataset) -> Dataset:
        r"""
        Computes the Smagorinsky diffusion:

        $$
        K = C_s \Delta x \Delta y \sqrt{\left(\frac{\partial u}{\partial x} \right)^2 + \left(\frac{\partial v}{\partial y} \right)^2 + \frac{1}{2} \left(\frac{\partial u}{\partial y} + \frac{\partial v}{\partial x} \right)^2}
        $$

        where $C_s$ is the ***trainable*** Smagorinsky constant, $\Delta x \Delta y$ a spatial scaling factor, 
        and the rest of the expression represents the horizontal diffusion.

        Parameters
        ----------
        t : Float[Scalar, ""]
            The simulation time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        dataset : Dataset
            The [`pastax.grid.Dataset`][] containing the physical fields.

        Returns
        -------
        Dataset
            The [`pastax.grid.Dataset`][] containing the Smagorinsky coefficients.

        Notes
        -----
        The physical fields are first restricted to a small neighborhood, then interpolated in time and 
        finally spatial derivatives are computed using finite central difference.
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
            use_degrees=neighborhood.use_degrees,
            is_uv_mps=False  # no uv anyway...
        )

        return smag_ds

    @staticmethod
    def _deterministic_dynamics(
        t: Float[Scalar, ""],
        y: Float[Array, "2"],
        dataset: Dataset,
        smag_ds: Dataset
    ) -> Float[Array, "2"]:
        r"""
        Computes the deterministic part of the dynamics (i.e. the Lagrangian velocity): 
        $(\mathbf{u} + \nabla K)(t, \mathbf{X}(t))$.

        Parameters
        ----------
        t : Float[Scalar, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        dataset : Dataset
            The [`pastax.grid.Dataset`][] containing the physical fields.
        smag_ds : Dataset
            The [`pastax.grid.Dataset`][] containing the Smagorinsky coefficients for the given fields.

        Returns
        -------
        Float[Array, "2"]
            The deterministic part of the dynamics (i.e. the Lagrangian velocity).
        """
        latitude, longitude = y[0], y[1]

        smag_k = jnp.squeeze(smag_ds.variables["smag_k"].values)  # "x_width-2 x_width-2"

        # $\mathbf{u}(t, \mathbf{X}(t))$ term
        u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=latitude, longitude=longitude)
        vu = jnp.asarray([v, u], dtype=float)  # "2"

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
    def _stochastic_dynamics(y: Float[Array, "2"], smag_ds: Dataset) -> Float[Array, "2 2"]:
        r"""
        Computes the stochastic part of the dynamics (i.e. the diffusion): 
        $V(t, \mathbf{X}(t)) = \sqrt{2 K(t, \mathbf{X}(t))}$.

        Parameters
        ----------
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        smag_ds : Dataset
            The [`pastax.grid.Dataset`][] containing the Smagorinsky coefficients.

        Returns
        -------
        Float[Array, "2 2"]
            The stochastic part of the dynamics (i.e. the diffusion).
        """
        latitude, longitude = y[0], y[1]

        smag_k = smag_ds.interp_spatial("smag_k", latitude=latitude, longitude=longitude)[0]
        smag_k = jnp.squeeze(smag_k)  # scalar
        smag_k = (2 * smag_k) ** (1 / 2)

        return jnp.eye(2) * smag_k

    def __call__(self, t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2 3"]:
        r"""
        Computes the determinist (i.e. Lagrangian velocity): $(\mathbf{u} + \nabla K)(t, \mathbf{X}(t))$ 
        and stochastic (i.e. diffusion): $V(t, \mathbf{X}(t)) = \sqrt{2 K(t, \mathbf{X}(t))}$ parts of the dynamics.

        Parameters
        ----------
        t : float
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Dataset
            The [`pastax.grid.Dataset`][] containing the velocity fields.

        Returns
        -------
        Float[Array, "2 3"]
            The stacked determinist and stochastic parts of the dynamics.
        """
        t = jnp.asarray(t)
        dataset = args

        smag_ds = self._smagorinsky_diffusion(t, y, dataset)  # "1 x_width-2 x_width-2"

        dlatlon_drift = self._deterministic_dynamics(t, y, dataset, smag_ds)
        dlatlon_diffusion = self._stochastic_dynamics(y, smag_ds)

        dlatlon = jnp.column_stack([dlatlon_drift, dlatlon_diffusion])

        if dataset.is_spherical_mesh and not dataset.use_degrees:
            dlatlon = meters_to_degrees(dlatlon.T, latitude=y[0]).T

        return dlatlon
