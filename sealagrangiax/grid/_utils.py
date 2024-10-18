from typing import Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def spatial_derivative(
    *fields: Tuple[Float[Array, "(time) lat lon"], ...],
    dx: Float[Array, "lat lon-1"],
    dy: Float[Array, "lat-1 lon"],
    is_land: Bool[Array, "lat lon"]
) -> Tuple[Float[Array, "(time) lat lon"], ...]:
    """
    Computes spatial derivatives for given fields using central finite differences.

    This function calculates the spatial derivatives of the provided fields, taking into account the presence of land
    and the grid spacing in both latitude and longitude directions.
    It uses central finite differences for the computation and leverages JAX for efficient computation and automatic 
    differentiation.

    Parameters
    ----------
    fields (Tuple[Float[Array, "(time) lat lon"], ...])
        Variable number of fields for which the spatial derivatives are to be computed. 
        Each field is a 2D or 3D array with dimensions (latitude, longitude) or (time, latitude, longitude).
    dx (Float[Array, "lat lon-1"])
        Grid spacing in the longitude direction.
    dy (Float[Array, "lat-1 lon"])
        Grid spacing in the latitude direction.
    is_land (Bool[Array, "lat lon"])
        Boolean array indicating the presence of land at each grid point.

    Returns
    -------
    Tuple[Float[Array, "(time) lat lon"], ...]
        A tuple containing the spatial derivatives of the input fields. 
        Each derivative is a 2D or 3D array with dimensions (latitude, longitude) or (time, latitude, longitude).
        For field f1 and f2, returns (df1_x, df1_y, df2_x, df2_y).
    """
    def central_finite_difference(
            _f: Float[Array, "(time) lat lon"], _axis: int
    ) -> Tuple[Float[Array, "(time) lat-2 lon-2"], ...]:
        def _axis1(_dxy: Float[Array, "(time) lat(-1) lon(-1)"]) -> Tuple[Float[Array, "(time) lat-2 lon-2"], ...]:
            _f_l = _f[..., :-2, 1:-1]
            _f_r = _f[..., 2:, 1:-1]

            _is_land_l = is_land[:-2, 1:-1]
            _is_land_r = is_land[2:, 1:-1]

            _dx_l = _dxy[:-1, 1:-1]
            _dx_r = _dxy[1:, 1:-1]

            return _f_l, _f_r, _is_land_l, _is_land_r, _dx_l, _dx_r

        def _axis2(_dxy: Float[Array, "(time) lat(-1) lon(-1)"]) -> Tuple[Float[Array, "(time) lat-2 lon-2"], ...]:
            _f_l = _f[..., 1:-1, :-2]
            _f_r = _f[..., 1:-1, 2:]

            _is_land_l = is_land[1:-1, :-2]
            _is_land_r = is_land[1:-1, 2:]

            _dx_l = _dxy[1:-1, :-1]
            _dx_r = _dxy[1:-1, 1:]

            return _f_l, _f_r, _is_land_l, _is_land_r, _dx_l, _dx_r

        f_l, f_r, is_land_l, is_land_r, dx_l, dx_r = jax.lax.cond(
            _axis == -1,
            lambda: _axis2(dx),
            lambda: _axis1(dy)
        )

        f_c = _f[..., 1:-1, 1:-1]
        f_l = jnp.where(is_land_l, f_c, f_l)
        f_r = jnp.where(is_land_r, f_c, f_r)

        return (f_r - f_l) / (dx_r + dx_l)

    derivatives = tuple(
        central_finite_difference(f, axis) for f in fields for axis in (-1, -2)
    )

    return derivatives
