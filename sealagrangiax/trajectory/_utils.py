import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils.unit import degrees_to_radians, EARTH_RADIUS  # noqa


@jax.jit
def earth_distance(latlon1: Float[Array, "2"], latlon2: Float[Array, "2"]) -> Float[Array, ""]:
    # haversine formula
    self_phi = degrees_to_radians(latlon1[0])
    other_phi = degrees_to_radians(latlon2[0])
    delta = degrees_to_radians(latlon1 - latlon2)

    a = jnp.sin(delta[0] / 2) ** 2 + jnp.cos(self_phi) * jnp.cos(other_phi) * jnp.sin(delta[1] / 2) ** 2
    c = 2 * jnp.atan2(jnp.sqrt(a), jnp.sqrt(1 - a))
    distance = EARTH_RADIUS * c

    return distance
