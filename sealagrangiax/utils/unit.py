import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
import numpy as np


class UNIT(eqx.Enumeration):
    dimensionless = ""

    degrees = "°"
    kilometers = "km"
    meters = "m"

    square_kilometers = "km^2"
    square_meters = "m^2"

    days = "d"
    hours = "h"
    seconds = "s"


EARTH_RADIUS = 6371008.8


@jax.jit
def longitude_in_0_360_degrees(latlon: Float[Array, "2"]):
    def adjust_longitude(lon: Float[Array, ""]) -> Float[Array, ""]:
        lon += 180
        lon %= 360
        lon -= 180
        return lon

    return latlon.at[1].set(adjust_longitude(latlon[1]))


@jax.jit
def degrees_to_radians(arr: Float[Array, "..."]) -> Float[Array, "..."]:
    return arr * jnp.pi / 180.


@jax.jit
def degrees_to_meters(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    arr = degrees_to_radians(arr) * EARTH_RADIUS
    arr = arr.at[1].multiply(jnp.cos(degrees_to_radians(latitude)))

    return arr


@jax.jit
def degrees_to_kilometers(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    return meters_to_kilometers(degrees_to_meters(arr, latitude))


@jax.jit
def meters_to_degrees(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    arr = (arr / EARTH_RADIUS) * (180. / jnp.pi)
    arr = arr.at[1].divide(jnp.cos(degrees_to_radians(latitude)))

    return arr


@jax.jit
def meters_to_kilometers(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    return arr / 1000


@jax.jit
def kilometers_to_degrees(arr: Float[Array, "2"], latitude: Float[Array, ""]) -> Float[Array, "2"]:
    return meters_to_degrees(kilometers_to_meters(arr), latitude)


@jax.jit
def kilometers_to_meters(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    return arr * 1000


@jax.jit
def sq_kilometers_to_sq_meters(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    return kilometers_to_meters(kilometers_to_meters(arr))


@jax.jit
def sq_meters_to_sq_kilometers(arr: Float[Array, "2"]) -> Float[Array, "2"]:
    return meters_to_kilometers(meters_to_kilometers(arr))


def time_in_seconds(arr: Int[Array, ""]) -> Int[Array, ""]:
    if (isinstance(arr, np.datetime64) or
            (isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.datetime64))):
        arr = arr.astype("datetime64[s]").astype(int)

    return arr


def seconds_to_days(arr: Int[Array, ""]) -> Int[Array, ""]:
    return arr / (60 * 60 * 24)
