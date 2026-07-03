"""Grid metadata for Arakawa A- and C-grid forcing layouts.

The :class:`Grid` object describes the *centre* (tracer) coordinates of a
forcing dataset together with its stagger type, and — for C-grids — the
staggered U/V-face coordinates. It is the single owner of coordinates: it is
held on :class:`pastax.Dataset` and every :class:`pastax.Field` references it,
reading the coordinates appropriate to its stagger role via
:meth:`Grid.coords_for`.

Each :class:`Field` therefore carries no coordinates of its own; interpolation
reads them from the shared :class:`Grid`. This keeps the ``Field.interp`` path
(bilinear on equally-spaced coords) valid for both A- and C-grid fields, since
shifting an equally-spaced grid by half a cell preserves equal spacing.

Curvilinear (2-D) coordinate arrays are accepted structurally for forward
compatibility but not yet supported by interpolation.
"""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ._types import Array, Float, Int

__all__ = ["Grid"]


class Grid(eqx.Module):
    """Grid metadata: centre coordinates plus stagger and topology type.

    Attributes:
        t_coords: 1-D time coordinates in seconds, equally spaced.
        lat_coords: Latitude of cell centres. 1-D for rectilinear grids, 2-D
            ``(lat, lon)`` for curvilinear grids.
        lon_coords: Longitude of cell centres. 1-D for rectilinear grids,
            2-D ``(lat, lon)`` for curvilinear grids.
        grid_type: ``"rectilinear"`` (default) or ``"curvilinear"``.
            Curvilinear grids are accepted structurally but not yet supported
            by ``Field.interp``.
        stagger_type: ``"A"`` (default — all variables at cell centres) or
            ``"C"`` (NEMO-convention Arakawa C-grid: U on east faces, V on
            north faces).
        lon_period: If set (e.g. ``360.0``), longitude is treated as periodic
            with that period. The centre grid is assumed to span exactly one
            period.
        t_origin: Epoch offset of the time axis, in seconds since the Unix
            epoch, stored as a 0-d ``int64`` array. ``t_coords`` are relative
            to this instant, i.e. the absolute time of ``t_coords[i]`` is
            ``t_origin + t_coords[i]``. Loaders set it when converting
            ``datetime64`` input (which is rebased to the first timestamp so
            the stored float32 coordinates stay small and precise); it is
            ``0`` for plain numeric time input. Solver times (``t0`` and
            interpolation queries) live in the same rebased frame.

            ``t_origin`` is a pytree *leaf*, not static metadata: swapping
            forcing datasets with different origins does not retrigger jit
            compilation. Integer seconds are exact in int64; read the value
            host-side with ``int(grid.t_origin)``. Inside ``jax.jit`` (without
            ``jax_enable_x64``) the leaf is canonicalized to int32, which
            remains exact for epoch seconds until 2038.
        u_lat_coords, u_lon_coords: U-face coordinates (NEMO C-grid). Populated
            by the C-grid loaders (derived from the centre grid via
            :meth:`u_face_coords` or supplied explicitly); ``None`` on A-grids.
        v_lat_coords, v_lon_coords: V-face coordinates (NEMO C-grid). Populated
            by the C-grid loaders (derived via :meth:`v_face_coords` or supplied
            explicitly); ``None`` on A-grids.
    """

    t_coords: Float[Array, "time"]
    lat_coords: Float[Array, "..."]
    lon_coords: Float[Array, "..."]
    grid_type: Literal["rectilinear", "curvilinear"] = eqx.field(
        static=True, default="rectilinear"
    )
    stagger_type: Literal["A", "C"] = eqx.field(static=True, default="A")
    lon_period: float | None = eqx.field(static=True, default=None)
    t_origin: Int[Array, ""] = eqx.field(
        default_factory=lambda: np.array(0, dtype=np.int64)
    )
    u_lat_coords: Float[Array, "..."] | None = None
    u_lon_coords: Float[Array, "..."] | None = None
    v_lat_coords: Float[Array, "..."] | None = None
    v_lon_coords: Float[Array, "..."] | None = None

    def u_face_coords(self) -> tuple[Float[Array, "lat"], Float[Array, "lon_u"]]:
        r"""Return ``(lat_u, lon_u)`` — coordinates of U-face centres (NEMO C-grid).

        For a centre grid of size ``nlon`` in longitude, U lives on the
        ``nlon - 1`` east faces between adjacent cells:

        .. math::

            \mathrm{lon}_u[i] = \tfrac{1}{2}\left(\mathrm{lon}_c[i] + \mathrm{lon}_c[i+1]\right)

        Latitude is unchanged: :math:`\mathrm{lat}_u = \mathrm{lat}_c`.

        Raises:
            NotImplementedError: For curvilinear grids.
        """
        self._check_rectilinear("u_face_coords")
        lon_c = self.lon_coords
        lon_u = 0.5 * (lon_c[:-1] + lon_c[1:])
        return self.lat_coords, lon_u

    def v_face_coords(self) -> tuple[Float[Array, "lat_v"], Float[Array, "lon"]]:
        r"""Return ``(lat_v, lon_v)`` — coordinates of V-face centres (NEMO C-grid).

        For a centre grid of size ``nlat`` in latitude, V lives on the
        ``nlat - 1`` north faces between adjacent cells:

        .. math::

            \mathrm{lat}_v[j] = \tfrac{1}{2}\left(\mathrm{lat}_c[j] + \mathrm{lat}_c[j+1]\right)

        Longitude is unchanged: :math:`\mathrm{lon}_v = \mathrm{lon}_c`.

        Raises:
            NotImplementedError: For curvilinear grids.
        """
        self._check_rectilinear("v_face_coords")
        lat_c = self.lat_coords
        lat_v = 0.5 * (lat_c[:-1] + lat_c[1:])
        return lat_v, self.lon_coords

    def coords_for(
        self, stagger: Literal["center", "u_face", "v_face"]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        """Return the ``(lat, lon)`` coordinates a field at ``stagger`` reads.

        ``"center"`` returns the centre (tracer) coordinates; ``"u_face"`` and
        ``"v_face"`` return the stored staggered coordinates (populated by the
        C-grid loaders). This is the single source of coordinates consulted by
        :meth:`pastax.Field.interp`.

        Raises:
            ValueError: For ``"u_face"`` / ``"v_face"`` on a grid that carries
                no staggered coordinates, or for an unknown ``stagger``.
        """
        if stagger == "center":
            return self.lat_coords, self.lon_coords
        if stagger == "u_face":
            if self.u_lat_coords is None or self.u_lon_coords is None:
                raise ValueError(
                    "Grid has no U-face coordinates; build it via a C-grid "
                    "loader (Dataset.from_arrays_cgrid / from_xarray_cgrid)."
                )
            return self.u_lat_coords, self.u_lon_coords
        if stagger == "v_face":
            if self.v_lat_coords is None or self.v_lon_coords is None:
                raise ValueError(
                    "Grid has no V-face coordinates; build it via a C-grid "
                    "loader (Dataset.from_arrays_cgrid / from_xarray_cgrid)."
                )
            return self.v_lat_coords, self.v_lon_coords
        raise ValueError(
            f"Unknown stagger {stagger!r}; expected 'center', 'u_face' or "
            "'v_face'."
        )

    def period_for(
        self, stagger: Literal["center", "u_face", "v_face"]
    ) -> float | None:
        """Return the longitude period a field at ``stagger`` should use.

        Centre (tracer) fields inherit :attr:`lon_period`. U/V faces always get
        ``None``: their coordinate arrays no longer span a full period, so
        periodic wrapping would be ill-defined at first order.

        Consequence for global (periodic) C-grids: the U face between
        ``lon_c[-1]`` and ``lon_c[0] + lon_period`` — the seam face — is not
        represented (there are ``nlon - 1`` U faces, not ``nlon``), and face
        fields *extrapolate* rather than wrap across the seam. Queries near
        the seam meridian therefore lose the C-grid's coastal guarantees; if
        trajectories cross the seam, prefer A-grid forcing or rotate the grid
        so the seam sits over land.
        """
        return self.lon_period if stagger == "center" else None

    def _check_rectilinear(self, method: str) -> None:
        if self.grid_type != "rectilinear":
            raise NotImplementedError(
                f"Grid.{method} is only implemented for rectilinear grids "
                f"(got grid_type={self.grid_type!r})."
            )
        if jnp.ndim(self.lat_coords) != 1 or jnp.ndim(self.lon_coords) != 1:
            raise ValueError(
                f"Rectilinear Grid expected 1-D lat/lon coords, got "
                f"lat ndim={jnp.ndim(self.lat_coords)}, "
                f"lon ndim={jnp.ndim(self.lon_coords)}."
            )
