import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int     
import numpy as np

from .geo import earth_distance, EARTH_RADIUS


class UNIT(eqx.Enumeration):
    """
    JAX-compatible enumeration class representing various units of measurement.

    Attributes
    ----------
    dimensionless : UNIT
        Represents a dimensionless quantity.
    degrees : UNIT
        Represents degrees (°).
    kilometers : UNIT
        Represents kilometers (km).
    meters : UNIT
        Represents meters (m).
    square_kilometers : UNIT
        Represents square kilometers (km^2).
    square_meters : UNIT
        Represents square meters (m^2).
    days : UNIT
        Represents days (d).
    hours : UNIT
        Represents hours (h).
    seconds : UNIT
        Represents seconds (s).
    """
    dimensionless = ""

    degrees = "°"
    kilometers = "km"
    meters = "m"

    square_kilometers = "km^2"
    square_meters = "m^2"

    days = "d"
    hours = "h"
    seconds = "s"


@jax.jit
def longitude_in_180_180_degrees(longitude: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Adjusts an array of longitudes to be within the range of -180 to 180 degrees.

    Parameters
    ----------
    longitude : Float[Array, "..."]
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


@jax.jit
def degrees_to_meters(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    """
    Converts an array of latitude/longitude distances from degrees to meters.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of latitude/longitude distances in degrees.
    latitude : Float[Array, ""]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "2"]
        An array of latitude/longitude distances in meters.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    dlat, dlon = arr[..., 0], arr[..., 1]
    
    lat1 = latitude
    lat2 = lat1 + dlat
    lon2 = dlon

    dy = earth_distance(
        jnp.stack([lat1, jnp.zeros_like(lat1)], axis=-1), jnp.stack([lat2, jnp.zeros_like(lat2)], axis=-1)
        )
    dx = earth_distance(jnp.stack([lat1, jnp.zeros_like(lat1)], axis=-1), jnp.stack([lat1, lon2], axis=-1))

    return jnp.stack([dx, dy], axis=-1)


@jax.jit
def degrees_to_kilometers(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    """
    Converts an array of latitude/longitude distances from degrees to kilometers.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of latitude/longitude distances in degrees.
    latitude : Float[Array, ""]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "2"]
        An array of latitude/longitude distances in kilometers.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    return meters_to_kilometers(degrees_to_meters(arr, latitude))


@jax.jit
def meters_to_degrees(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    """
    Converts an array of latitude/longitude distances from meters to degrees.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of latitude/longitude distances in meters.
    latitude : Float[Array, ""]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "2"]
        An array of latitude/longitude distances in degrees.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    dx, dy = arr[..., 1], arr[..., 0]
    lat1_rad = jnp.radians(latitude)

    distance = jnp.sqrt(dx**2 + dy**2)
    bearing = jnp.atan2(dx, dy)

    # using Haversine formula
    lat2_rad = jnp.asin(
        jnp.sin(lat1_rad) * jnp.cos(distance / EARTH_RADIUS) + 
        jnp.cos(lat1_rad) * jnp.sin(distance / EARTH_RADIUS) * jnp.cos(bearing)
    )
    
    dlon = jnp.atan2(
        jnp.sin(bearing) * jnp.sin(distance / EARTH_RADIUS) * jnp.cos(lat1_rad),
        jnp.cos(distance / EARTH_RADIUS) - jnp.sin(lat1_rad) * jnp.sin(lat2_rad)
    )
    
    dlat = jnp.degrees(lat2_rad - lat1_rad)
    dlon = jnp.degrees(dlon)

    return jnp.stack([dlat, dlon], axis=-1)


@jax.jit
def meters_to_kilometers(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    """
    Converts an array of distances from meters to kilometers.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of distances in meters.

    Returns
    -------
    Float[Array, "2"]
        An array of distances in kilometers.
    """
    return arr / 1000


@jax.jit
def kilometers_to_degrees(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    """
    Converts an array of latitude/longitude distances from kilometers to degrees.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of latitude/longitude distances in kilometers.
    latitude : Float[Array, ""]
        The latitude in degrees at which the conversion is to be performed.

    Returns
    -------
    Float[Array, "2"]
        An array of latitude/longitude distances in degrees.

    Notes
    -----
    This function uses the Haversine formula for accurate conversion of distances.
    """
    return meters_to_degrees(kilometers_to_meters(arr), latitude)


@jax.jit
def kilometers_to_meters(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    """
    Converts an array of distances from kilometers to meters.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of distances in kilometers.

    Returns
    -------
    Float[Array, "2"]
        An array of distances in meters.
    """
    return arr * 1000


@jax.jit
def sq_kilometers_to_sq_meters(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    """
    Converts an array from squared kilometers to squared meters.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of squared kilometers.

    Returns
    -------
    Float[Array, "2"]
        An array of squared meters.
    """
    return kilometers_to_meters(kilometers_to_meters(arr))


@jax.jit
def sq_meters_to_sq_kilometers(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    """
    Converts an array from squared meters to squared kilometers.

    Parameters
    ----------
    arr : Float[Array, "2"]
        An array of squared meters.

    Returns
    -------
    Float[Array, "2"]
        An array of squared kilometers.
    """
    return meters_to_kilometers(meters_to_kilometers(arr))


def time_in_seconds(arr: Int[Array, ""]) -> Int[Array, ""]:
    """
    Converts an array of datetime64 values to seconds since the Unix epoch.

    Parameters
    ----------
    arr : Int[Array, ""]
        An array of datetime64 values or a single datetime64 value.

    Returns
    -------
    Int[Array, ""]
        An array of integers representing the number of seconds since the Unix epoch.
    """
    if (isinstance(arr, np.datetime64) or (isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.datetime64))):
        arr = arr.astype("datetime64[s]").astype(int)

    return arr


def seconds_to_days(arr: Int[Array, ""]) -> Int[Array, ""]:
    """
    Converts an array of time durations from seconds to days.

    Parameters
    ----------
    arr : Int[Array, ""]
        An array of time durations in seconds.

    Returns
    -------
    Int[Array, ""]
        An array of time durations in days.
    """
    return arr / (60 * 60 * 24)
