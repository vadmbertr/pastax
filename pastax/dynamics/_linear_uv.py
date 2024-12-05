from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils import meters_to_degrees
from ..gridded import Gridded


def _linear_uv(t: float, y: Float[Array, "2"], args: Gridded) -> Float[Array, "2"]:
    t = jnp.asarray(t, dtype=float)
    dataset = args
    latitude, longitude = y[0], y[1]

    u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=latitude, longitude=longitude)
    dlatlon = jnp.asarray([v, u], dtype=float)

    return dlatlon


def linear_uv(t: float, y: Float[Array, "2"], args: Gridded) -> Float[Array, "2"]:
    """
    Computes the Lagrangian drift velocity by interpolating in space and time the velocity fields.

    Parameters
    ----------
    t : float
        The current time.
    y : Float[Array, "2"]
        The current state (latitude and longitude in degrees).
    args : Dataset
        The [`pastax.gridded.Gridded`][] containing the physical fields (only u and v here).

    Returns
    -------
    Float[Array, "2"]
        The Lagrangian drift velocity.
    """
    dlatlon = _linear_uv(t, y, args)

    dataset = args
    if dataset.is_spherical_mesh and not dataset.use_degrees:
        dlatlon = meters_to_degrees(dlatlon, latitude=y[0])

    return dlatlon


class LinearUV(eqx.Module):
    """
    Trainable linear transformation of the Lagrangian drift velocity 
    computed by interpolating in space and time the velocity fields.

    Attributes
    ----------
    intercept : Float[Array, ""] | Float[Array, "2"], optional
        The intercept of the linear relation, defaults to `jnp.asarray([0., 0.])`.
    slope : Float[Array, ""] | Float[Array, "2"], optional
        The slope of the linear relation, defaults to `jnp.asarray([1., 1.])`.

    Methods
    -------
    __call__(t, y, args)
        Computes the Lagrangian drift velocity.

    Notes
    -----
    As the class inherits from [`equinox.Module`][], its `intercept` and `slope` attributes can be treated as 
    trainable parameters.
    """

    intercept: Float[Array, ""] | Float[Array, "2"] = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=float), default_factory=lambda: [0, 0]
    )
    slope: Float[Array, ""] | Float[Array, "2"] = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=float), default_factory=lambda: [1, 1]
    )

    def __call__(self, t: float, y: Float[Array, "2"], args: Gridded) -> Float[Array, "2"]:
        """
        Computes the Lagrangian drift velocity as the linear relation `intercept + slope * [v, u]`.

        Parameters
        ----------
        t : float
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        args : Dataset
            The [`pastax.gridded.Gridded`][] containing the physical fields (only u and v here).

        Returns
        -------
        Float[Array, "2"]
            The Lagrangian drift velocity.
        """
        vu = _linear_uv(t, y, args)

        dlatlon = self.intercept + self.slope * vu

        dataset = args
        if dataset.is_spherical_mesh and not dataset.use_degrees:
            dlatlon = meters_to_degrees(dlatlon, latitude=y[0])

        return dlatlon
