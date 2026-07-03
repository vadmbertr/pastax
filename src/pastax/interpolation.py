"""Bilinear interpolation on equally-spaced rectilinear A-grids."""

import jax.numpy as jnp

from ._safe_math import safe_divide
from ._types import Array, Bool, Float, Int

__all__ = [
    "linear_interp_1d",
    "bilinear_interp_2d",
    "spatiotemporal_interp",
    "bilinear_velocity_partialslip_2d",
    "spatiotemporal_velocity_partialslip",
]


def _index_and_weight(
    coords: Float[Array, "n"], x: Float[Array, ""]
) -> tuple[Int[Array, ""], Float[Array, ""]]:
    """Floor index and linear weight for a point on an equally-spaced 1-D grid.

    For grid spacing dx = coords[1] - coords[0] and origin x0 = coords[0]:
        i = clip(floor((x - x0) / dx), 0, n-2)
        w = (x - x0) / dx - i          (= fractional position within cell [i, i+1])

    w is in [0, 1) for in-range x; outside that range it extrapolates linearly.
    """
    x0 = coords[0]
    dx = coords[1] - coords[0]
    u = (x - x0) / dx
    i = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, coords.shape[0] - 2)
    w = u - i  # equivalent to (x - (x0 + i*dx)) / dx
    return i, w


def _periodic_index_and_weight(
    coords: Float[Array, "n"], x: Float[Array, ""], period: float
) -> tuple[Int[Array, ""], Float[Array, ""]]:
    """Floor index and linear weight on a periodic equally-spaced 1-D grid.

    The grid is assumed to span exactly one period: ``n * dx == period`` and
    ``coords[n] == coords[0] + period``, with ``coords[n]`` not stored. The
    returned index ``i`` is in ``[0, n-1]``; the right neighbour is
    ``(i + 1) % n``.

    ``coords`` must be **ascending** (``dx > 0``): the ``% period`` fold maps
    the query into ``[0, period)`` above ``coords[0]``, which is inconsistent
    with a negative ``dx``. (The non-periodic :func:`_index_and_weight`
    handles descending coordinates fine — the sign cancels in ``(x - x0)/dx``.)
    The loaders enforce this when ``lon_period`` is set.
    """
    x0 = coords[0]
    dx = coords[1] - coords[0]
    n = coords.shape[0]
    u = ((x - x0) % period) / dx  # u in [0, n)
    floor_u = jnp.floor(u)
    i = floor_u.astype(jnp.int32) % n
    w = u - floor_u
    return i, w


def linear_interp_1d(
    values: Float[Array, "n"],
    coords: Float[Array, "n"],
    x: Float[Array, ""],
) -> Float[Array, ""]:
    """Linearly interpolate a 1-D field on an equally-spaced grid.

    Args:
        values: Field values at each grid node, shape ``(n,)``.
        coords: Equally-spaced 1-D grid coordinates, shape ``(n,)``.
        x: Query coordinate.

    Returns:
        Interpolated scalar value at ``x``. For ``x`` outside the grid the
        result is the linear extrapolation from the nearest cell.
    """
    i, w = _index_and_weight(coords, x)
    return values[i] * (1.0 - w) + values[i + 1] * w


def bilinear_interp_2d(
    values: Float[Array, "lat lon"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
    lon_period: float | None = None,
    mask: Bool[Array, "lat lon"] | None = None,
) -> Float[Array, ""]:
    r"""Bilinearly interpolate a 2-D field on an equally-spaced rectilinear grid.

    Args:
        values: Field values, shape ``(n_lat, n_lon)``.
        lat_coords: Equally-spaced latitude coordinates in degrees, shape ``(n_lat,)``.
        lon_coords: Equally-spaced longitude coordinates in degrees, shape ``(n_lon,)``.
        lat: Query latitude in degrees.
        lon: Query longitude in degrees.
        lon_period: If given (e.g. ``360.0``), the longitude axis is treated
            as periodic with that period; the grid is assumed to span exactly
            one period (``n_lon * dlon == lon_period``) and the cell at
            ``lon_coords[-1] + dlon`` is identified with ``lon_coords[0]``.
            Periodic longitudes must be **ascending** (see
            :func:`_periodic_index_and_weight`). ``None`` (default) reproduces
            the non-wrapping behaviour with linear extrapolation past the
            boundary.
        mask: Optional 2-D boolean land mask, same shape as ``values``.
            ``True`` marks a land cell. Behaviour by mixed-corner count:

            * All four corners ocean → standard bilinear (bit-exact identical
              to the ``mask=None`` path).
            * Mixed corners → inverse-distance partial-cell weighting on the
              ocean corners in normalised cell coordinates
              (:math:`d^2 = \alpha^2 + \beta^2 + \varepsilon` where
              :math:`\alpha`, :math:`\beta` are the fractional distances along
              the cell axes); land corners are dropped.
            * All four corners land → returns ``0`` (zero velocity for
              fully-grounded cells).

            The :math:`\varepsilon` floor and :func:`safe_divide` keep both
            forward and backward passes finite for queries on or near a corner.

    Returns:
        Interpolated scalar value at ``(lat, lon)``.
    """
    il, wl = _index_and_weight(lat_coords, lat)
    if lon_period is None:
        jl, wj = _index_and_weight(lon_coords, lon)
        jl1 = jl + 1
    else:
        nlon = lon_coords.shape[0]
        jl, wj = _periodic_index_and_weight(lon_coords, lon, lon_period)
        jl1 = (jl + 1) % nlon

    v00 = values[il,     jl ]
    v10 = values[il + 1, jl ]
    v01 = values[il,     jl1]
    v11 = values[il + 1, jl1]
    naive = (
        v00 * (1.0 - wl) * (1.0 - wj)
        + v10 * wl         * (1.0 - wj)
        + v01 * (1.0 - wl) * wj
        + v11 * wl         * wj
    )
    if mask is None:
        return naive

    o00 = ~mask[il,     jl ]
    o10 = ~mask[il + 1, jl ]
    o01 = ~mask[il,     jl1]
    o11 = ~mask[il + 1, jl1]
    n_ocean = (
        o00.astype(jnp.int32) + o10.astype(jnp.int32)
        + o01.astype(jnp.int32) + o11.astype(jnp.int32)
    )

    eps = jnp.asarray(1e-7, dtype=naive.dtype)
    wl_lo = wl
    wl_hi = 1.0 - wl
    wj_lo = wj
    wj_hi = 1.0 - wj
    d00 = wl_lo * wl_lo + wj_lo * wj_lo + eps
    d10 = wl_hi * wl_hi + wj_lo * wj_lo + eps
    d01 = wl_lo * wl_lo + wj_hi * wj_hi + eps
    d11 = wl_hi * wl_hi + wj_hi * wj_hi + eps

    one = jnp.asarray(1.0, dtype=naive.dtype)
    zero = jnp.asarray(0.0, dtype=naive.dtype)
    iw00 = jnp.where(o00, one / d00, zero)
    iw10 = jnp.where(o10, one / d10, zero)
    iw01 = jnp.where(o01, one / d01, zero)
    iw11 = jnp.where(o11, one / d11, zero)
    weighted = v00 * iw00 + v10 * iw10 + v01 * iw01 + v11 * iw11
    total = iw00 + iw10 + iw01 + iw11
    invdist = safe_divide(weighted, total)

    all_ocean = n_ocean == 4
    all_land  = n_ocean == 0
    return jnp.where(all_ocean, naive, jnp.where(all_land, zero, invdist))


def spatiotemporal_interp(
    values: Float[Array, "time lat lon"],
    t_coords: Float[Array, "time"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    t: Float[Array, ""],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
    lon_period: float | None = None,
    mask: Bool[Array, "lat lon"] | None = None,
) -> Float[Array, ""]:
    """Trilinearly interpolate a field in time and space on an A-grid.

    Performs :func:`bilinear_interp_2d` at the two bounding time slices, then
    linearly blends the two results in time.

    Args:
        values: Field values, shape ``(n_time, n_lat, n_lon)``.
        t_coords: Equally-spaced time coordinates in seconds, shape ``(n_time,)``.
        lat_coords: Equally-spaced latitude coordinates in degrees.
        lon_coords: Equally-spaced longitude coordinates in degrees.
        t: Query time in seconds.
        lat: Query latitude in degrees.
        lon: Query longitude in degrees.
        lon_period: If given, treat the longitude axis as periodic with this
            period (see :func:`bilinear_interp_2d`).
        mask: Optional 2-D land mask shared across time (see
            :func:`bilinear_interp_2d`).

    Returns:
        Interpolated scalar value at ``(t, lat, lon)``.
    """
    it, wt = _index_and_weight(t_coords, t)
    v0 = bilinear_interp_2d(
        values[it],     lat_coords, lon_coords, lat, lon,
        lon_period=lon_period, mask=mask,
    )
    v1 = bilinear_interp_2d(
        values[it + 1], lat_coords, lon_coords, lat, lon,
        lon_period=lon_period, mask=mask,
    )
    return v0 * (1.0 - wt) + v1 * wt


def bilinear_velocity_partialslip_2d(
    u_values: Float[Array, "lat lon"],
    v_values: Float[Array, "lat lon"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
    mask: Bool[Array, "lat lon"],
    slip_a: float = 0.5,
    slip_b: float = 0.5,
    lon_period: float | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    r"""Bilinear A-grid velocity interpolation with partial-slip wall correction.

    Replaces the standard bilinear weights for ``U`` (tangential to a
    latitudinal coast) and ``V`` (tangential to a longitudinal coast) with
    a wall-slip-aware formula whenever an entire cell edge is land.

    For a south-edge-fully-land cell with the north edge in the ocean, the
    naive bilinear for ``U`` is :math:`w_l\,U_{\mathrm{along}\,N}` where

    .. math::

        U_{\mathrm{along}\,N} = (1 - w_j)\,U_{NW} + w_j\,U_{NE}

    is the linear interpolation of ``U`` along the north (ocean) edge. The
    partial-slip correction replaces this with
    :math:`(a + b\,w_l)\,U_{\mathrm{along}\,N}`: at the coast
    (:math:`w_l = 0`) the tangential velocity is
    :math:`a\,U_{\mathrm{along}\,N}` rather than :math:`0` (which would trap
    the particle). The default :math:`a = b = 0.5` gives a half-slip wall;
    :math:`a = 1,\ b = 0` recovers full free-slip; :math:`a = 0,\ b = 1`
    recovers the no-slip naive bilinear.

    Symmetrically for north-edge-land cells (using :math:`U_{\mathrm{along}\,S}`
    and :math:`a + b\,(1 - w_l)`) and for ``V`` across east/west land edges.
    Cells without a fully-land edge fall back to standard bilinear.

    Args:
        u_values: U-component values on the A-grid, shape ``(n_lat, n_lon)``.
        v_values: V-component values, same shape as ``u_values``.
        lat_coords: Equally-spaced latitudes, shape ``(n_lat,)``.
        lon_coords: Equally-spaced longitudes, shape ``(n_lon,)``.
        lat: Query latitude.
        lon: Query longitude.
        mask: Joint U/V land mask, ``True`` = land. Typically built as
            ``u_mask & v_mask`` so a corner is considered land only when
            both components are masked there.
        slip_a: Slip coefficient at the wall. Default 0.5.
        slip_b: Slip coefficient gradient. Default 0.5.
        lon_period: Periodic longitude period in degrees, or ``None``.

    Returns:
        Tuple ``(u, v)`` of interpolated A-grid velocity components.
    """
    il, wl = _index_and_weight(lat_coords, lat)
    if lon_period is None:
        jl, wj = _index_and_weight(lon_coords, lon)
        jl1 = jl + 1
    else:
        nlon = lon_coords.shape[0]
        jl, wj = _periodic_index_and_weight(lon_coords, lon, lon_period)
        jl1 = (jl + 1) % nlon

    u_sw = u_values[il,     jl ]
    u_nw = u_values[il + 1, jl ]
    u_se = u_values[il,     jl1]
    u_ne = u_values[il + 1, jl1]
    v_sw = v_values[il,     jl ]
    v_nw = v_values[il + 1, jl ]
    v_se = v_values[il,     jl1]
    v_ne = v_values[il + 1, jl1]

    m_sw = mask[il,     jl ]
    m_nw = mask[il + 1, jl ]
    m_se = mask[il,     jl1]
    m_ne = mask[il + 1, jl1]

    w_sw = (1.0 - wl) * (1.0 - wj)
    w_nw = wl         * (1.0 - wj)
    w_se = (1.0 - wl) * wj
    w_ne = wl         * wj
    u_naive = u_sw * w_sw + u_nw * w_nw + u_se * w_se + u_ne * w_ne
    v_naive = v_sw * w_sw + v_nw * w_nw + v_se * w_se + v_ne * w_ne

    south_land = m_sw & m_se
    north_land = m_nw & m_ne
    west_land  = m_sw & m_nw
    east_land  = m_se & m_ne

    a = slip_a
    b = slip_b

    # U: corrected when a latitudinal (S or N) edge is fully land.
    # `u_along_N` and `u_along_S` are linear interps of U along the ocean
    # edge (computed unconditionally — they are polynomial and safe).
    u_along_N = (1.0 - wj) * u_nw + wj * u_ne
    u_along_S = (1.0 - wj) * u_sw + wj * u_se
    u_south_corr = (a + b * wl)         * u_along_N
    u_north_corr = (a + b * (1.0 - wl)) * u_along_S
    u_corrected = jnp.where(
        south_land & ~north_land, u_south_corr,
        jnp.where(north_land & ~south_land, u_north_corr, u_naive),
    )

    # V: corrected when a longitudinal (E or W) edge is fully land.
    v_along_E = (1.0 - wl) * v_se + wl * v_ne
    v_along_W = (1.0 - wl) * v_sw + wl * v_nw
    v_west_corr = (a + b * wj)         * v_along_E
    v_east_corr = (a + b * (1.0 - wj)) * v_along_W
    v_corrected = jnp.where(
        west_land & ~east_land, v_west_corr,
        jnp.where(east_land & ~west_land, v_east_corr, v_naive),
    )

    return u_corrected, v_corrected


def spatiotemporal_velocity_partialslip(
    u_values: Float[Array, "time lat lon"],
    v_values: Float[Array, "time lat lon"],
    t_coords: Float[Array, "time"],
    lat_coords: Float[Array, "lat"],
    lon_coords: Float[Array, "lon"],
    t: Float[Array, ""],
    lat: Float[Array, ""],
    lon: Float[Array, ""],
    mask: Bool[Array, "lat lon"],
    slip_a: float = 0.5,
    slip_b: float = 0.5,
    lon_period: float | None = None,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Trilinear A-grid velocity interpolation with partial-slip wall correction.

    Applies :func:`bilinear_velocity_partialslip_2d` at the two bounding
    time slabs and blends linearly in time.

    Returns:
        Tuple ``(u, v)`` of trilinearly-interpolated A-grid velocities at
        ``(t, lat, lon)``.
    """
    it, wt = _index_and_weight(t_coords, t)
    u0, v0 = bilinear_velocity_partialslip_2d(
        u_values[it], v_values[it], lat_coords, lon_coords, lat, lon,
        mask, slip_a=slip_a, slip_b=slip_b, lon_period=lon_period,
    )
    u1, v1 = bilinear_velocity_partialslip_2d(
        u_values[it + 1], v_values[it + 1], lat_coords, lon_coords, lat, lon,
        mask, slip_a=slip_a, slip_b=slip_b, lon_period=lon_period,
    )
    return (u0 * (1.0 - wt) + u1 * wt, v0 * (1.0 - wt) + v1 * wt)
