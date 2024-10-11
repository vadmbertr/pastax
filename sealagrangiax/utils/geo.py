import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


EARTH_RADIUS = 6371008.8
"""
float: The radius of the Earth in meters.
"""


@jax.jit
def earth_distance(latlon1: Float[Array, "2"], latlon2: Float[Array, "2"]) -> Float[Array, ""]:
    """
    Calculate the great-circle distance between two points on the Earth's surface.

    This function uses the haversine formula to compute the distance between two points
    specified by their latitude and longitude coordinates.

    Parameters
    ----------
    latlon1 : Float[Array, "2"]
        A 2-element array containing the latitude and longitude of the first point.
    latlon2 : Float[Array, "2"]
        A 2-element array containing the latitude and longitude of the second point.

    Returns
    -------
    Float[Array, ""]
        The distance between the two points in meters.
    """
    lat1_rad = jnp.radians(latlon1[..., 0])
    lat2_rad = jnp.radians(latlon2[..., 0])
    dlat_rad = jnp.radians(latlon1 - latlon2)

    a = jnp.sin(dlat_rad[..., 0] / 2) ** 2 + jnp.cos(lat1_rad) * jnp.cos(lat2_rad) * jnp.sin(dlat_rad[..., 1] / 2) ** 2
    c = 2 * jnp.atan2(jnp.sqrt(a), jnp.sqrt(1 - a))
    distance = EARTH_RADIUS * c

    return distance
