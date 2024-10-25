import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


EARTH_RADIUS = 6371008.8
"""
float: The radius of the Earth in meters.
"""


def distance_on_earth(latlon1: Float[Array, "... 2"], latlon2: Float[Array, "... 2"]) -> ArrayLike:
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
    State
        The distance between the two (array of) points in meters.
    """
    lat1_rad = jnp.radians(latlon1[..., 0])
    lat2_rad = jnp.radians(latlon2[..., 0])
    d_rad = jnp.radians(latlon1 - latlon2)

    a = jnp.sin(d_rad[..., 0] / 2) ** 2 + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * jnp.sin(d_rad[..., 1] / 2) ** 2
    c = 2 * jnp.atan2(jnp.sqrt(a), jnp.sqrt(1 - a))
    d = EARTH_RADIUS * c

    return d


def longitude_in_180_180_degrees(longitude: ArrayLike) -> ArrayLike:
    """
    Adjusts an array of longitudes to be within the range of -180 to 180 degrees.

    Parameters
    ----------
    longitude : ArrayLike
        An array of longitudes in degrees.

    Returns
    -------
    Float[Scalar, ""]
        The input longitudes adjusted to be within the range of -180 to 180 degrees.

    Notes
    -----
    This function acts as the identy for longitudes that are already within the range of -180 to 180 degrees.
    """
    return (longitude + 180) % 360 - 180
