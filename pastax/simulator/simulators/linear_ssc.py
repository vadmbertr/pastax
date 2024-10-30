from __future__ import annotations
from typing import Callable

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Scalar

from .._diffrax_simulator import DeterministicDiffrax
from ...utils import meters_to_degrees
from ...grid import Dataset


def _linear_ssc_rhs(t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
    t = jnp.asarray(t, dtype=float)
    dataset = args
    latitude, longitude = y[0], y[1]

    u, v = dataset.interp_spatiotemporal("u", "v", time=t, latitude=latitude, longitude=longitude)
    dlatlon = jnp.asarray([v, u], dtype=float)

    return dlatlon


def linear_ssc_rhs(t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
    """
    Computes the drift term of the solved Ordinary Differential Equation by interpolating in space and 
    time the velocity fields.

    Parameters
    ----------
    t : float
        The current time.
    y : Float[Array, "2"]
        The current state (latitude and longitude in degrees).
    args : Dataset
        The [`pastax.grid.Dataset`][] containing the physical fields (only u and v here).

    Returns
    -------
    Float[Array, "2"]
        The drift term (change in latitude and longitude in degrees).
    """
    dlatlon = _linear_ssc_rhs(t, y, args)

    dataset = args
    if dataset.is_spherical_mesh and not dataset.use_degrees:
        dlatlon = meters_to_degrees(dlatlon, latitude=y[0])

    return dlatlon


class IdentitySSC(DeterministicDiffrax):
    """
    Deterministic simulator considering only Sea Surface Currents.

    Attributes
    ----------
    id : str
        The identifier for the IdentitySSC model, defaults to "identity_ssc".
        
    Methods
    -------
    rhs(t, y, args)
        Computes the drift term of the solved Ordinary Differential Equation.

    Notes
    -----
    In this example, the `rhs` attribute is simply a function as the simulator does not have parameters to optimize.
    """

    id: str = eqx.field(static=True, default_factory=lambda: "identity_ssc")
    rhs: Callable[[Float[Scalar, ""], Float[Array, "2"], Dataset], Float[Array, "2"]] = linear_ssc_rhs


class LinearRHS(eqx.Module):
    """
    Attributes
    ----------
    intercept : Float[Array, ""] | Float[Array, "2"], optional
        The intercept of the linear relation, defaults to [0, 0].
    slope : Float[Array, ""] | Float[Array, "2"], optional
        The slope of the linear relation, defaults to [1, 1].

    Methods
    -------
    __call__(t, y, args)
        Computes the drift term of the solved Ordinary Differential Equation.

    Notes
    -----
    As the class inherits from `eqx.Module`, its `intercept` and `slope` attributes can be treated as a trainable parameters.
    """

    intercept: Float[Array, ""] | Float[Array, "2"] = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=float), default_factory=lambda: [0, 0]
    )
    slope: Float[Array, ""] | Float[Array, "2"] = eqx.field(
        converter=lambda x: jnp.asarray(x, dtype=float), default_factory=lambda: [1, 1]
    )

    def __call__(self, t: float, y: Float[Array, "2"], args: Dataset) -> Float[Array, "2"]:
        """
        Computes the drift term of the solved Ordinary Differential Equation as the linear relation 
        `intercept` + `slope` * `[v, u]`.

        Parameters
        ----------
        t : float
            The current time.
        y : Float[Array, "2"]
            The current state (latitude and longitude in degrees).
        args : Dataset
            The [`pastax.grid.Dataset`][] containing the physical fields (only u and v here).

        Returns
        -------
        Float[Array, "2"]
            The drift term (change in latitude and longitude in degrees).
        """
        vu = _linear_ssc_rhs(t, y, args)

        dlatlon = self.intercept + self.slope * vu

        dataset = args
        if dataset.is_spherical_mesh and not dataset.use_degrees:
            dlatlon = meters_to_degrees(dlatlon, latitude=y[0])

        return dlatlon


class LinearSSC(DeterministicDiffrax):
    """
    Trainable (intercept and slope) linear deterministic simulator considering only Sea Surface Currents.

    Attributes
    ----------
    rhs : _RHS
        Computes the drift term of the solved Ordinary Differential Equation.
    id : str
        The identifier for the `pastax.LinearSSC` simulator, defaults to `"linear_ssc"`.

    Methods
    -------
    from_param(intercept=None, slope=None, id=None)
        Creates a `pastax.LinearSSC` simulator with the given intercept, slope, and id.

    Notes
    -----
    In this example, the `rhs` attribute is an `eqx.Module` with intercept and slope as attributes, allowing to treat them as a trainable parameters.
    """

    id: str = eqx.field(static=True, default_factory=lambda: "linear_ssc")
    rhs: LinearRHS = LinearRHS()
    
    @classmethod
    def from_param(
        cls, 
        intercept: Float[Array, ""] | Float[Array, "2"] = None, 
        slope: Float[Array, ""] | Float[Array, "2"] = None, 
        id: str = None
    ) -> LinearSSC:
        """
        Creates a `pastax.LinearSSC` simulator with the given intercept and slope parameters for the linear relation.

        Parameters
        ----------
        intercept : Float[Array, ""] | Float[Array, "2"], optional
            The intercept of the linear relation, defaults to None.
        slope : Float[Array, ""] | Float[Array, "2"], optional
            The slope of the linear relation, defaults to None.
        id : str, optional
            The identifier for the simulator, defaults to None.

        Returns
        -------
        LinearSSC
            The `pastax.LinearSSC` simulator.

        Notes
        -----
        If any of the parameters is None, its default value is used.
        """
        rhs_kwargs = {}
        if intercept is not None:
            rhs_kwargs["intercept"] = intercept
        if slope is not None:
            rhs_kwargs["slope"] = slope

        self_kwargs = {}
        if id is not None:
            self_kwargs["id"] = id

        return cls(rhs=LinearRHS(**rhs_kwargs), **self_kwargs)
