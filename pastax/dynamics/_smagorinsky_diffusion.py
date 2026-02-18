from __future__ import annotations

from typing import Any

import equinox as eqx
import interpax as ipx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Real
import lineax as lx

from ..gridded import Gridded, spatial_derivative
from ..utils import meters_to_degrees


def _to_cs(x: Real[Any, ""]) -> Real[Array, ""]:
    return jnp.exp(x)


def _from_cs(x: Real[Any, ""]) -> Real[Array, ""]:
    return jnp.log(jnp.clip(x, min=1e-4))


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
        K = C_s \Delta x \Delta y \sqrt{\left(\frac{\partial u}{\partial x} \right)^2 +
        \left(\frac{\partial v}{\partial y} \right)^2 + \frac{1}{2} \left(\frac{\partial u}{\partial y} +
        \frac{\partial v}{\partial x} \right)^2}
        $$

        where $C_s$ is the ***trainable*** Smagorinsky constant, $\Delta x \Delta y$ a spatial scaling factor, and the
        rest of the expression represents the horizontal diffusion.

    Methods
    -------
    cs
        Returns the Smagorinsky constant.
    _neighborhood(*fields, t, y, gridded)
        Restricts the [`pastax.gridded.Gridded`][] to a neighborhood around the given location and time.
    _smagorinsky_coefficients(t, y, gridded)
        Computes the Smagorinsky diffusion:

        $$
        K = C_s \Delta x \Delta y \sqrt{\left(\frac{\partial u}{\partial x} \right)^2 +
        \left(\frac{\partial v}{\partial y} \right)^2 + \frac{1}{2} \left(\frac{\partial u}{\partial y} +
        \frac{\partial v}{\partial x} \right)^2}
        $$

        where $C_s$ is the ***trainable*** Smagorinsky constant, $\Delta x \Delta y$ a spatial scaling factor,
        and the rest of the expression represents the horizontal diffusion.
    _deterministic_dynamics(t, y, gridded, smag_ds)
        Computes the deterministic part of the dynamics: $(\mathbf{u} + \nabla K)(t, \mathbf{X}(t))$.
    _stochastic_dynamics(y, smag_ds)
        Computes the stochastic part of the dynamics: $V(t, \mathbf{X}(t)) = \sqrt{2 K(t, \mathbf{X}(t))}$.

    Notes
    -----
    As the class inherits from [`equinox.Module`][], its `cs` attribute can be treated as a trainable parameter.
    """

    _cs: Real[Array, ""] = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=float),
        default_factory=lambda: _from_cs(0.1),
    )

    @property
    def cs(self) -> Float[Array, ""]:
        """
        Returns the Smagorinsky constant.
        """
        return _to_cs(self._cs)

    @staticmethod
    def _neighborhood(*fields: str, t: Real[Array, ""], y: Float[Array, "2"], gridded: Gridded) -> Gridded:
        """
        Restricts the [`pastax.gridded.Gridded`][] to a neighborhood around the given location and time.

        Parameters
        ----------
        *fields : list[str]
            The fields to retain in the neighborhood.
        t : Real[Array, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        gridded : Gridded
            The [`pastax.gridded.Gridded`][] containing the physical fields.

        Returns
        -------
        Gridded
            The neighborhood [`pastax.gridded.Gridded`][].
        """
        # restrict gridded to the neighborhood around X(t)
        neighborhood = gridded.neighborhood(
            *fields, time=t, latitude=y[0], longitude=y[1], t_width=3, x_width=7
        )  # "x_width x_width"

        return neighborhood

    def _smagorinsky_diffusion(self, t: Real[Array, ""], y: Float[Array, "2"], gridded: Gridded) -> Gridded:
        r"""
        Computes the Smagorinsky diffusion:

        $$
        K = C_s \Delta x \Delta y \sqrt{\left(\frac{\partial u}{\partial x} \right)^2 +
        \left(\frac{\partial v}{\partial y} \right)^2 + \frac{1}{2} \left(\frac{\partial u}{\partial y} +
        \frac{\partial v}{\partial x} \right)^2}
        $$

        where $C_s$ is the ***trainable*** Smagorinsky constant, $\Delta x \Delta y$ a spatial scaling factor,
        and the rest of the expression represents the horizontal diffusion.

        Parameters
        ----------
        t : Real[Array, ""]
            The simulation time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        gridded : Gridded
            The [`pastax.gridded.Gridded`][] containing the physical fields.

        Returns
        -------
        Gridded
            The [`pastax.gridded.Gridded`][] containing the Smagorinsky coefficients.

        Notes
        -----
        The physical fields are first restricted to a small neighborhood, then interpolated in time and
        finally spatial derivatives are computed using finite central difference.
        """
        neighborhood = self._neighborhood("u", "v", t=t, y=y, gridded=gridded)

        fields = neighborhood.interp("u", "v", time=t)  # "x_width x_width"
        (dudx, dudy), (dvdx, dvdy) = spatial_derivative(
            fields["u"], fields["v"], dx=neighborhood.dx, dy=neighborhood.dy, is_masked=jnp.isnan(fields["u"])
        )  # "x_width x_width"

        # computes Smagorinsky coefficients
        cell_area = neighborhood.cell_area  # "x_width x_width"
        smag_k = self.cs * cell_area * ((dudx**2 + dvdy**2 + 0.5 * (dudy + dvdx) ** 2) ** (1 / 2))

        smag_ds = Gridded.from_array(
            {"smag_k": smag_k[None, ...]},
            time=t[None],
            latitude=neighborhood.coordinates["latitude"].values,
            longitude=neighborhood.coordinates["longitude"].values,
            interpolation_method="linear",
            is_spherical_mesh=neighborhood.is_spherical_mesh,
            use_degrees=neighborhood.use_degrees,
            is_uv_mps=False,  # no uv anyway...
        )

        return smag_ds

    @staticmethod
    def _deterministic_dynamics(
        t: Real[Array, ""], y: Float[Array, "2"], gridded: Gridded, smag_ds: Gridded
    ) -> Float[Array, "2"]:
        r"""
        Computes the deterministic part of the dynamics: $(\mathbf{u} + \nabla K)(t, \mathbf{X}(t))$.

        Parameters
        ----------
        t : Real[Array, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        gridded : Gridded
            The [`pastax.gridded.Gridded`][] containing the physical fields.
        smag_ds : Gridded
            The [`pastax.gridded.Gridded`][] containing the Smagorinsky coefficients for the given fields.

        Returns
        -------
        Float[Array, "2"]
            The deterministic part of the dynamics.
        """
        latitude, longitude = y[0], y[1]

        smag_k = jnp.squeeze(smag_ds.fields["smag_k"].values)  # "x_width x_width"

        # $\mathbf{u}(t, \mathbf{X}(t))$ term
        scalar_values = gridded.interp("u", "v", time=t, latitude=latitude, longitude=longitude)
        vu = jnp.asarray([scalar_values["v"], scalar_values["u"]])  # "2"

        # $(\nabla \cdot \mathbf{K})(t, \mathbf{X}(t))$ term
        ((dkdx, dkdy),) = spatial_derivative(
            smag_k, dx=smag_ds.dx, dy=smag_ds.dy, is_masked=jnp.isnan(smag_k)
        )  # "x_width x_width"
        dkdx = ipx.interp2d(
            latitude,
            longitude,
            smag_ds.coordinates["latitude"].values,
            smag_ds.coordinates["longitude"].values,
            dkdx,
            method="linear",
            extrap=True,
        )
        dkdy = ipx.interp2d(
            latitude,
            longitude,
            smag_ds.coordinates["latitude"].values,
            smag_ds.coordinates["longitude"].values,
            dkdy,
            method="linear",
            extrap=True,
        )
        gradk = jnp.asarray([dkdy, dkdx])  # "2"

        return vu + gradk

    @staticmethod
    def _stochastic_dynamics(y: Float[Array, "2"], smag_ds: Gridded) -> Float[Array, "2 2"]:
        r"""
        Computes the stochastic part of the dynamics: $V(t, \mathbf{X}(t)) = \sqrt{2 K(t, \mathbf{X}(t))}$.

        Parameters
        ----------
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        smag_ds : Gridded
            The [`pastax.gridded.Gridded`][] containing the Smagorinsky coefficients.

        Returns
        -------
        Float[Array, "2 2"]
            The stochastic part of the dynamics.
        """
        latitude, longitude = y[0], y[1]

        scalar_value = smag_ds.interp("smag_k", latitude=latitude, longitude=longitude)
        smag_k = jnp.squeeze(scalar_value["smag_k"])  # scalar
        smag_k = (2 * smag_k) ** (1 / 2)

        return jnp.eye(2) * smag_k


class StochasticSmagorinskyDiffusion(SmagorinskyDiffusion):
    r"""
    Trainable stochastic Smagorinsky diffusion dynamics.

    !!! example "Formulation"

        This dynamics allows to formulate a displacement at time $t$ from the position $\mathbf{X}(t)$ as:

        $$
        d\mathbf{X}(t) = (\mathbf{u} + \nabla K)(t, \mathbf{X}(t)) dt + V(t, \mathbf{X}(t)) d\mathbf{W}(t)
        $$

        where $V = \sqrt{2 K}$ and $K$ is the Smagorinsky diffusion:

        $$
        K = C_s \Delta x \Delta y \sqrt{\left(\frac{\partial u}{\partial x} \right)^2 +
        \left(\frac{\partial v}{\partial y} \right)^2 + \frac{1}{2} \left(\frac{\partial u}{\partial y} +
        \frac{\partial v}{\partial x} \right)^2}
        $$

        where $C_s$ is the ***trainable*** Smagorinsky constant, $\Delta x \Delta y$ a spatial scaling factor, and the
        rest of the expression represents the horizontal diffusion.

    Methods
    -------
    __call__(t, y, args)
        Computes the deterministic and stochastic terms of the dynamics and returns them as lineax.PyTreeLinearOperator.
    """

    def __call__(self, t: Real[Array, ""], y: Float[Array, "2"], args: Gridded) -> lx.PyTreeLinearOperator:
        r"""
        Computes the deterministic and stochastic terms of the dynamics and returns them as [`lineax.PyTreeLinearOperator`][].

        Parameters
        ----------
        t : Real[Array, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Gridded
            The [`pastax.gridded.Gridded`][] containing the velocity fields.

        Returns
        -------
        lx.PyTreeLinearOperator
            The stacked deterministic and stochastic parts of the dynamics.
        """
        gridded = args

        smag_ds = self._smagorinsky_diffusion(t, y, gridded)  # "1 x_width x_width"

        dlatlon_deter = self._deterministic_dynamics(t, y, gridded, smag_ds)
        dlatlon_stoch = self._stochastic_dynamics(y, smag_ds)

        if gridded.is_spherical_mesh and not gridded.use_degrees:
            dlatlon_deter = meters_to_degrees(dlatlon_deter, latitude=y[0])
            dlatlon_stoch = meters_to_degrees(dlatlon_stoch, latitude=y[0])

        return lx.PyTreeLinearOperator((dlatlon_deter, dlatlon_stoch), jax.ShapeDtypeStruct((2,), float))

    @classmethod
    def from_cs(cls, cs: Real[Any, ""] = 0.1):
        """
        Initializes the stochastic Smagorinsky diffusion with the given Smagorinsky constant.

        Parameters
        ----------
        cs : Real[Any, ""], optional
            The Smagorinsky constant, defaults to `jnp.asarray(0.1, dtype=float)`.

        Returns
        -------
        StochasticSmagorinskyDiffusion
            The [`pastax.dynamics.StochasticSmagorinskyDiffusion`][] initialized with the given Smagorinsky constant.
        """
        return cls(_cs=_from_cs(cs))


class DeterministicSmagorinskyDiffusion(SmagorinskyDiffusion):
    r"""
    Trainable deterministic Smagorinsky diffusion dynamics.

    !!! example "Formulation"

        This dynamics allows to formulate a displacement at time $t$ from the position $\mathbf{X}(t)$ as:

        $$
        d\mathbf{X}(t) = (\mathbf{u} + \nabla K)(t, \mathbf{X}(t)) dt + V(t, \mathbf{X}(t)) d\mathbf{W}(t)
        $$

        where $V = \sqrt{2 K}$ and $K$ is the Smagorinsky diffusion:

        $$
        K = C_s \Delta x \Delta y \sqrt{\left(\frac{\partial u}{\partial x} \right)^2 +
        \left(\frac{\partial v}{\partial y} \right)^2 + \frac{1}{2} \left(\frac{\partial u}{\partial y} +
        \frac{\partial v}{\partial x} \right)^2}
        $$

        where $C_s$ is the ***trainable*** Smagorinsky constant, $\Delta x \Delta y$ a spatial scaling factor, and the
        rest of the expression represents the horizontal diffusion.

    Methods
    -------
    __call__(t, y, args)
        Computes the deterministic term of the dynamics.
    """

    def __call__(self, t: Real[Array, ""], y: Float[Array, "2"], args: Gridded) -> Float[Array, "2"]:
        r"""
        Computes the deterministic term of the dynamics.

        Parameters
        ----------
        t : Real[Array, ""]
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude).
        args : Gridded
            The [`pastax.gridded.Gridded`][] containing the velocity fields.

        Returns
        -------
        Float[Array, "2 3"]
            The deterministic part of the dynamics.
        """
        gridded = args

        smag_ds = self._smagorinsky_diffusion(t, y, gridded)  # "1 x_width-2 x_width-2"
        dlatlon = self._deterministic_dynamics(t, y, gridded, smag_ds)

        if gridded.is_spherical_mesh and not gridded.use_degrees:
            dlatlon = meters_to_degrees(dlatlon, latitude=y[0])

        return dlatlon

    @classmethod
    def from_cs(cls, cs: Real[Any, ""] = 0.1):
        """
        Initializes the deterministic Smagorinsky diffusion with the given Smagorinsky constant.

        Parameters
        ----------
        cs : Real[Any, ""], optional
            The Smagorinsky constant, defaults to `jnp.asarray(0.1, dtype=float)`.

        Returns
        -------
        DeterministicSmagorinskyDiffusion
            The [`pastax.dynamics.DeterministicSmagorinskyDiffusion`][] initialized with the given Smagorinsky constant.
        """
        return cls(_cs=_from_cs(cs))
