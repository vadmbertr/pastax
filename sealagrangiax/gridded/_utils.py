import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


@jax.jit
def derivative_spatial(
    *fields: (Float[Array, "(time) lat lon"], ...),
    dx: Float[Array, "lat lon-1"],
    dy: Float[Array, "lat-1 lon"],
    is_land: Bool[Array, "lat lon"]
) -> (Float[Array, "..."], ...):
    def central_finite_difference(
            _f: Float[Array, "(time) lat lon"], _axis: int
    ) -> (Float[Array, "(time) lat-2 lon-2"], ...):
        def _axis1(_dxy: Float[Array, "(time) lat(-1) lon(-1)"]) -> (Float[Array, "(time) lat-2 lon-2"], ...):
            _f_l = _f[..., :-2, 1:-1]
            _f_r = _f[..., 2:, 1:-1]

            _is_land_l = is_land[:-2, 1:-1]
            _is_land_r = is_land[2:, 1:-1]

            _dx_l = _dxy[:-1, 1:-1]
            _dx_r = _dxy[1:, 1:-1]

            return _f_l, _f_r, _is_land_l, _is_land_r, _dx_l, _dx_r

        def _axis2(_dxy: Float[Array, "(time) lat(-1) lon(-1)"]) -> (Float[Array, "(time) lat-2 lon-2"], ...):
            _f_l = _f[..., 1:-1, :-2]
            _f_r = _f[..., 1:-1, 2:]

            _is_land_l = is_land[1:-1, :-2]
            _is_land_r = is_land[1:-1, 2:]

            _dx_l = _dxy[1:-1, :-1]
            _dx_r = _dxy[1:-1, 1:]

            return _f_l, _f_r, _is_land_l, _is_land_r, _dx_l, _dx_r

        f_l, f_r, is_land_l, is_land_r, dx_l, dx_r = jax.lax.cond(
            _axis == 1,
            lambda: _axis1(dy),
            lambda: _axis2(dx)
        )

        f_c = _f[..., 1:-1, 1:-1]
        f_l = jnp.where(is_land_l, f_c, f_l)
        f_r = jnp.where(is_land_r, f_c, f_r)

        return (f_r - f_l) / (dx_r + dx_l)

    derivatives = tuple(
        central_finite_difference(f, axis) for f in fields for axis in (2, 1)
    )

    return derivatives
