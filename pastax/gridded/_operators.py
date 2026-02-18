import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def spatial_derivative(
    *fields: Float[Array, "(time) lat lon"],
    dx: Float[Array, "lat lon-1"],
    dy: Float[Array, "lat-1 lon"],
    is_masked: Bool[Array, "lat lon"],
) -> tuple[tuple[Float[Array, "(time) lat lon"], Float[Array, "(time) lat lon"]], ...]:
    """
    Computes spatial derivatives for given fields using central finite differences.

    This function calculates the spatial derivatives of the provided fields, taking into account the presence of mask
    and the grid spacing in both latitude and longitude directions.
    It uses central finite differences for the computation and leverages JAX for efficient computation and automatic
    differentiation.

    Parameters
    ----------
    *fields : Float[Array, "(time) lat lon"]
        Variable number of fields for which the spatial derivatives are to be computed.
        Each field is a 2D or 3D array with dimensions (latitude, longitude) or (time, latitude, longitude).
    dx : Float[Array, "lat lon-1"]
        Gridded spacing in the longitude direction.
    dy : Float[Array, "lat-1 lon"]
        Gridded spacing in the latitude direction.
    is_masked : Bool[Array, "lat lon"]
        Boolean array indicating whether a grid point should be masked (`True` means masked, `False` not masked).

    Returns
    -------
    tuple[tuple[Float[Array, "(time) lat lon"], Float[Array, "(time) lat lon"]], ...]
        A tuple containing the spatial derivatives of the input fields.
        Each derivative is a 2D or 3D array with dimensions (latitude, longitude) or (time, latitude, longitude).
        For field f1 and f2, returns ((df1_x, df1_y), (df2_x, df2_y)).
    """

    def central_finite_difference(
        field: Float[Array, "(time) lat lon"],
    ) -> tuple[Float[Array, "(time) lat lon"], Float[Array, "(time) lat lon"]]:
        dx_left = jnp.pad(dx, ((0, 0), (1, 0)), mode="edge")
        dx_right = jnp.pad(dx, ((0, 0), (0, 1)), mode="edge")
        dy_left = jnp.pad(dy, ((1, 0), (0, 0)), mode="edge")
        dy_right = jnp.pad(dy, ((0, 1), (0, 0)), mode="edge")
        dx_cfd = dx_right
        dx_cfd = dx_cfd.at[..., 1:-1].set((dx_right[..., 1:-1] + dx_right[..., 2:]) / 2)
        dx_cfd = dx_cfd.at[..., -1].set(dx_left[..., -1])
        dy_cfd = dy_right
        dy_cfd = dy_cfd.at[..., 1:-1, :].set((dy_right[..., 1:-1, :] + dy_right[..., 2:, :]) / 2)
        dy_cfd = dy_cfd.at[..., -1, :].set(dy_left[..., -1, :])

        f_x, f_y = jnp.gradient(field, axis=(-1, -2))

        # handle masked values and normalize by grid spacing
        is_not_right_border_x = jnp.full_like(f_x, True).at[..., -1].set(False).astype(bool)
        is_not_left_border_x = jnp.full_like(f_x, True).at[..., 0].set(False).astype(bool)
        f_x = jnp.where(is_masked, jnp.nan, f_x / dx_cfd)
        f_x = jnp.where(
            ~is_masked & jnp.isnan(f_x) & is_not_right_border_x,
            jnp.diff(field, axis=-1, append=field[..., -1:]) / dx_right,
            f_x,
        )
        f_x = jnp.where(
            ~is_masked & jnp.isnan(f_x) & is_not_left_border_x,
            -jnp.diff(field[..., ::-1], axis=-1, append=field[..., :1])[..., ::-1] / dx_left,
            f_x,
        )
        is_not_right_border_y = jnp.full_like(f_y, True).at[..., -1, :].set(False).astype(bool)
        is_not_left_border_y = jnp.full_like(f_y, True).at[..., 0, :].set(False).astype(bool)
        f_y = jnp.where(is_masked, jnp.nan, f_y / dy_cfd)
        f_y = jnp.where(
            ~is_masked & jnp.isnan(f_y) & is_not_right_border_y,
            jnp.diff(field, axis=-2, append=field[..., -1:, :]) / dy_right,
            f_y,
        )
        f_y = jnp.where(
            ~is_masked & jnp.isnan(f_y) & is_not_left_border_y,
            -jnp.diff(field[..., ::-1, :], axis=-2, append=field[..., :1, :])[..., ::-1, :] / dy_left,
            f_y,
        )

        return f_x, f_y

    derivatives = tuple(central_finite_difference(field) for field in fields)
    return derivatives
