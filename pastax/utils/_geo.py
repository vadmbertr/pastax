import jax.numpy as jnp
from jaxtyping import Array, Float


EARTH_RADIUS = 6371008.8
"""
float: The radius of the Earth in meters.
"""


def distance_on_earth(latlon1: Float[Array, "... 2"], latlon2: Float[Array, "... 2"]) -> Array:
    """
    Calculates the distance in meters between two points on the Earth's surface.

    This function uses the Haversine formula to compute the distance between two (array of) points
    specified by their latitude and longitude coordinates.

    Parameters
    ----------
    latlon1 : Float[Array, "... 2"]
        A 2-element(s) array containing the latitude and longitude in degrees of the first point(s).
    latlon2 : Float[Array, "... 2"]
        A 2-element(s) array containing the latitude and longitude in degrees of the second point(s).

    Returns
    -------
    Array
        The distance between the two (array of) points in meters.
    """

    def safe_for_grad_sqrt(x):
        # grad(sqrt(x)) is not defined at x=0, here we are happy if it evaluates to 0 when x is a
        y = jnp.sqrt(jnp.where(x != 0.0, x, 1.0))  # type: ignore
        return jnp.where(x != 0.0, y, 0.0)

    lat1_rad = jnp.radians(latlon1[..., 0])
    lat2_rad = jnp.radians(latlon2[..., 0])
    d_rad = jnp.radians(latlon1 - latlon2)

    a = jnp.sin(d_rad[..., 0] / 2) ** 2 + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * jnp.sin(d_rad[..., 1] / 2) ** 2
    c = 2 * jnp.atan2(safe_for_grad_sqrt(a), jnp.sqrt(1 - a))
    d = EARTH_RADIUS * c

    return d


def longitude_in_180_180_degrees(longitude: Array) -> Array:
    """
    Adjusts an array of longitudes to be within the range of -180 to 180 degrees.

    Parameters
    ----------
    longitude : Array
        An array of longitudes in degrees.

    Returns
    -------
    Array
        The input longitudes adjusted to be within the range of -180 to 180 degrees.

    Notes
    -----
    This function acts as the identity for longitudes that are already within the range of -180 to 180 degrees.
    """
    return (longitude + 180) % 360 - 180
