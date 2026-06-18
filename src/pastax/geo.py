"""Geographic utilities: unit conversions and Haversine distance."""

import jax.numpy as jnp

from ._safe_math import safe_sqrt
from ._types import Array, Float

__all__ = [
    "EARTH_RADIUS",
    "haversine",
    "meters_to_degrees",
    "degrees_to_meters",
]

EARTH_RADIUS: float = 6_371_008.8
"""Mean Earth radius in metres (IUGG 2015 mean radius)."""


def haversine(
    y1: Float[Array, "... 2"],
    y2: Float[Array, "... 2"],
) -> Float[Array, "..."]:
    r"""Great-circle distance between ``[lat, lon]`` points.

    Uses the spherical haversine formula with :data:`EARTH_RADIUS` as the
    sphere radius :math:`R`:

    .. math::

        a = \sin^2\!\left(\tfrac{\Delta\varphi}{2}\right)
        + \cos\varphi_1 \cos\varphi_2 \sin^2\!\left(\tfrac{\Delta\lambda}{2}\right)

    .. math::

        d = 2 R \arcsin\!\sqrt{a}

    where :math:`\varphi` is latitude and :math:`\lambda` is longitude (in
    radians). The last axis of each input must have size 2 (lat, lon); leading
    axes broadcast under standard NumPy/JAX rules.

    Args:
        y1: First point(s) ``[lat, lon]`` in degrees, shape ``(..., 2)``.
        y2: Second point(s) ``[lat, lon]`` in degrees, shape ``(..., 2)``.

    Returns:
        Great-circle distance in metres, with shape matching the broadcast of
        the leading axes of ``y1`` and ``y2``.
    """
    lat1 = jnp.radians(y1[..., 0])
    lat2 = jnp.radians(y2[..., 0])
    d = jnp.radians(y1 - y2)
    a = jnp.sin(d[..., 0] / 2) ** 2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(d[..., 1] / 2) ** 2
    c = 2.0 * jnp.arctan2(safe_sqrt(a), safe_sqrt(1.0 - a))
    return EARTH_RADIUS * c


def meters_to_degrees(
    disp_m: Float[Array, "... 2"],
    lat_deg: Float[Array, ""],
) -> Float[Array, "... 2"]:
    r"""Convert a ``[north, east]`` displacement in metres to ``[dlat, dlon]`` in degrees.

    Uses a flat-Earth approximation around ``lat_deg``: the meridional
    component is converted via :data:`EARTH_RADIUS`; the zonal component is
    additionally divided by :math:`\cos(\mathrm{lat})` to account for shrinking
    longitude circles toward the poles.

    Args:
        disp_m: Displacement(s) ``[north, east]`` in **metres**. The last axis
            must have size 2; leading axes are passed through unchanged.
        lat_deg: Reference latitude in **degrees**, used for the longitude
            scaling.

    Returns:
        Same shape as ``disp_m``, but expressed as ``[dlat, dlon]`` in **degrees**.
    """
    rad = disp_m / EARTH_RADIUS
    deg = jnp.degrees(rad)
    lon_scale = jnp.cos(jnp.radians(lat_deg))
    return deg.at[..., 1].divide(lon_scale)


def degrees_to_meters(
    disp_deg: Float[Array, "... 2"],
    lat_deg: Float[Array, ""],
) -> Float[Array, "... 2"]:
    """Convert a ``[dlat, dlon]`` displacement in degrees to ``[north, east]`` in metres.

    Inverse of :func:`meters_to_degrees`. Uses a flat-Earth approximation
    around ``lat_deg``.

    Args:
        disp_deg: Displacement(s) ``[dlat, dlon]`` in **degrees**. The last
            axis must have size 2; leading axes are passed through unchanged.
        lat_deg: Reference latitude in **degrees**, used for the longitude
            scaling.

    Returns:
        Same shape as ``disp_deg``, but expressed as ``[north, east]`` in **metres**.
    """
    rad = jnp.radians(disp_deg)
    meters = rad * EARTH_RADIUS
    lon_scale = jnp.cos(jnp.radians(lat_deg))
    return meters.at[..., 1].multiply(lon_scale)
