import jax
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
        field: Float[Array, "(time) lat lon"], axis: int
    ) -> Float[Array, "(time) lat-2 lon-2"]:
        def _axis1(dxy: Float[Array, "lat-1 lon"]) -> tuple[Float[Array, "(time) lat-2 lon-2"], ...]:
            field_start = field[..., :-2, 1:-1]
            field_end = field[..., 2:, 1:-1]

            is_masked_start = is_masked[:-2, 1:-1]
            is_masked_end = is_masked[2:, 1:-1]

            dx_start = dxy[:-1, 1:-1]
            dx_end = dxy[1:, 1:-1]

            return field_start, field_end, is_masked_start, is_masked_end, dx_start, dx_end

        def _axis2(dxy: Float[Array, "lat lon-1"]) -> tuple[Float[Array, "(time) lat-2 lon-2"], ...]:
            field_start = field[..., 1:-1, :-2]
            field_end = field[..., 1:-1, 2:]

            is_masked_start = is_masked[1:-1, :-2]
            is_masked_end = is_masked[1:-1, 2:]

            dx_start = dxy[1:-1, :-1]
            dx_end = dxy[1:-1, 1:]

            return field_start, field_end, is_masked_start, is_masked_end, dx_start, dx_end

        field_start, field_end, is_masked_start, is_masked_end, dx_start, dx_end = jax.lax.cond(
            axis == -1, lambda: _axis2(dx), lambda: _axis1(dy)
        )

        field_center = field[..., 1:-1, 1:-1]
        field_start = jnp.where(is_masked_start, field_center, field_start)
        field_end = jnp.where(is_masked_end, field_center, field_end)

        return (field_end - field_start) / (dx_end + dx_start)  # type: ignore

    derivatives = tuple(tuple(central_finite_difference(field, axis) for axis in (-1, -2)) for field in fields)

    return derivatives  # type: ignore
