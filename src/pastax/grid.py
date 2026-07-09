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

from ._types import Array, Float

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
            with that period. The grid then either spans exactly one period
            (a global grid — see :attr:`lon_closed`) or is a seam-crossing
            regional subset.
        lon_closed: Whether a periodic (``lon_period`` set) grid spans exactly
            one full period. ``True`` (default) is a **closed** global grid:
            the cell past the last centre is identified with the first and
            interpolation wraps (``(i+1) % n``). ``False`` is an **open**
            seam-crossing regional slab (e.g. ``170°…-170°``): longitude is
            still circular for query folding, but the grid does not wrap — it
            behaves like a bounded grid with ``nlon - 1`` interior faces and
            extrapolates past its two ends. Derived at load from
            ``nlon * dlon`` vs ``lon_period``; irrelevant when ``lon_period``
            is ``None``.
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
    lon_closed: bool = eqx.field(static=True, default=True)
    u_lat_coords: Float[Array, "..."] | None = None
    u_lon_coords: Float[Array, "..."] | None = None
    v_lat_coords: Float[Array, "..."] | None = None
    v_lon_coords: Float[Array, "..."] | None = None

    def u_face_coords(self) -> tuple[Float[Array, "lat"], Float[Array, "lon_u"]]:
        r"""Return ``(lat_u, lon_u)`` — coordinates of U-face centres (NEMO C-grid).

        For a **bounded** centre grid of size ``nlon`` in longitude, U lives on
        the ``nlon - 1`` interior east faces between adjacent cells:

        .. math::

            \mathrm{lon}_u[i] = \tfrac{1}{2}\left(\mathrm{lon}_c[i] + \mathrm{lon}_c[i+1]\right)

        When :attr:`lon_period` is set **and** the grid is closed
        (:attr:`lon_closed`), the grid is a global periodic grid, so every
        centre cell has an east face — including the seam face between the last
        centre and the first-plus-a-period. There are then ``nlon`` U faces,
        each a half-cell east of its centre:

        .. math::

            \mathrm{lon}_u[i] = \mathrm{lon}_c[i] + \tfrac{1}{2}\,\Delta\lambda

        This ``nlon``-face axis spans exactly one period, so U interpolation
        wraps across the seam with the same first-order periodic scheme used for
        centre fields (see :meth:`period_for`).

        An **open** seam-crossing regional grid (``lon_period`` set but
        ``lon_closed`` false) has no wrap and therefore no seam face: like a
        bounded grid it has ``nlon - 1`` interior east faces at the centre
        midpoints. The midpoints are correct because the centre longitudes are
        stored unwrapped (strictly ascending) by the loaders.

        Latitude is unchanged: :math:`\mathrm{lat}_u = \mathrm{lat}_c`.

        Raises:
            NotImplementedError: For curvilinear grids.
        """
        self._check_rectilinear("u_face_coords")
        lon_c = self.lon_coords
        if self.lon_period is None or not self.lon_closed:
            lon_u = 0.5 * (lon_c[:-1] + lon_c[1:])
        else:
            dlon = lon_c[1] - lon_c[0]
            lon_u = lon_c + 0.5 * dlon
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

        Every stagger role inherits :attr:`lon_period`: centre longitudes span
        one period by construction, V-face longitudes equal the centre
        longitudes, and periodic U faces are built seam-inclusive (``nlon``
        faces spanning one period; see :meth:`u_face_coords`). When
        :attr:`lon_period` is ``None`` all roles get ``None`` (bounded grid, no
        wrapping). Latitude is never periodic, so this concerns only the
        longitude axis.
        """
        return self.lon_period

    def closed_for(
        self, stagger: Literal["center", "u_face", "v_face"]
    ) -> bool:
        """Return whether the longitude axis a field at ``stagger`` reads is closed.

        Every stagger role inherits the centre grid's :attr:`lon_closed`: a
        U/V face of an open (seam-crossing) centre grid is itself an open
        interior axis, and the faces of a closed global grid are closed. Only
        meaningful together with a non-``None`` :meth:`period_for`; on a bounded
        grid the value is ignored.
        """
        return self.lon_closed

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
