"""Forcing field representation and loading from xarray datasets or plain arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

from ._types import Array, Bool, Float
from .grid import Grid
from .interpolation import spatiotemporal_interp, spatiotemporal_velocity_partialslip

if TYPE_CHECKING:
    import xarray as xr
    from jax.typing import DTypeLike

__all__ = ["Field", "Dataset"]


def _coerce_time_to_seconds(t: Array) -> Array:
    """Convert a datetime64 time coordinate to relative seconds; pass-through otherwise.

    Both ``Dataset.from_arrays`` and ``Dataset.from_xarray`` route through this
    helper, so users may pass NumPy ``datetime64`` arrays (any unit) directly:
    they are converted to seconds since the Unix epoch.

    Plain numeric arrays are returned unchanged.

    JAX arrays (concrete or traced) are passed straight through: JAX has no
    ``datetime64`` dtype, so there is nothing to coerce, and this keeps the
    helper safe to call under ``jax.vmap`` / ``jax.jit`` when ``t`` is a tracer
    (``np.asarray`` on a tracer would raise).

    Returns:
        ``t_seconds`` the time coordinate array.
    """
    import numpy as np

    if isinstance(t, jax.Array):
        return t

    # The ``Array`` annotation aliases ``jax.Array``, so type checkers flag the
    # lines below as unreachable. They are not: at runtime ``t`` is often a
    # NumPy ``datetime64`` / numeric array (e.g. from ``from_xarray``), which is
    # exactly what this NumPy path handles.
    t_arr = np.asarray(t)
    if t_arr.dtype.kind == "M":
        secs = t_arr.astype("datetime64[s]").astype(np.int64)
        return secs
    return t_arr


def _asarray_time_int64(t: Array) -> Array:
    """Store integer-second time coordinates as a JAX ``int64`` array.

    Time coordinates are kept as ``int64`` seconds so epoch-scale counts
    (``datetime64`` input converts to seconds since the Unix epoch, ~1.7e9 s)
    stay exact. When ``jax_enable_x64`` is disabled JAX silently truncates the
    request to ``int32``, which represents second counts exactly only while
    ``|t| < 2**31`` (dates before 2038-01-19); later timestamps overflow and
    corrupt time interpolation.

    JAX flags this with a generic ``int64 ... truncated to int32`` message that
    does not mention the consequence for time. This helper suppresses that
    message and emits a pastax-specific one instead, spelling out the overflow
    risk and how to avoid it. Tracers are passed straight through (no host-side
    config read while tracing).
    """
    import warnings

    if isinstance(t, jax.Array) or jax.config.jax_enable_x64:
        return jnp.asarray(t, dtype=jnp.int64)

    with warnings.catch_warnings():
        # Swallow JAX's generic downcast warning for this call; we re-raise a
        # more informative one below.
        warnings.filterwarnings(
            "ignore",
            message="Explicitly requested dtype int64",
            category=UserWarning,
        )
        t_arr = jnp.asarray(t, dtype=jnp.int64)

    warnings.warn(
        "pastax stores time coordinates as int64 seconds, but jax_enable_x64 "
        "is disabled so they were truncated to int32. int32 seconds are exact "
        "only for |t| < 2**31 (dates before 2038-01-19); later timestamps "
        "overflow and corrupt time interpolation. Enable 64-bit precision with "
        'jax.config.update("jax_enable_x64", True) (or set JAX_ENABLE_X64=1) '
        "before building the Dataset, or pass numeric time offsets relative to "
        "a reference near your data instead of absolute epoch seconds.",
        UserWarning,
        stacklevel=3,
    )
    return t_arr


def _nearest_idx(coords: Float[Array, "n"], x: Float[Array, ""], n: int) -> Array:
    """Nearest-neighbour index on an equally-spaced 1-D grid, clamped to [0, n-1]."""
    x0 = coords[0]
    dx = coords[1] - coords[0]
    return jnp.clip(jnp.round((x - x0) / dx).astype(jnp.int32), 0, n - 1)


def _nearest_idx_periodic(
    coords: Float[Array, "n"], x: Float[Array, ""], n: int, period: float
) -> Array:
    """Nearest-neighbour index on a periodic equally-spaced 1-D grid (mod n)."""
    x0 = coords[0]
    dx = coords[1] - coords[0]
    return jnp.round(((x - x0) % period) / dx).astype(jnp.int32) % n


class Field(eqx.Module):
    """A single scalar forcing field on a (time, lat, lon) rectilinear grid.

    A ``Field`` carries no coordinates of its own: it holds a reference to its
    parent :class:`Grid` and reads the coordinates appropriate to its stagger
    role from it (xarray-like — the grid is the shared source of coordinates).
    The coordinate attributes (:attr:`t_coords`, :attr:`lat_coords`,
    :attr:`lon_coords`, :attr:`lon_period`) are available as read-only
    properties that delegate to the grid.

    Attributes:
        values: Field values, shape ``(time, lat, lon)``.
        grid: The parent :class:`Grid`; the single source of coordinates for
            this field, indexed by :attr:`stagger`.
        stagger: Position of this field on the parent grid. ``"center"``
            (default) is the A-grid / tracer position; ``"u_face"`` and
            ``"v_face"`` mark the eastern and northern velocity faces of a
            NEMO-convention Arakawa C-grid. The grid serves the coordinates
            for the stagger position (so a ``stagger="u_face"`` field reads
            the grid's half-cell-shifted U-face longitudes); ``Field.interp``
            itself is the same bilinear scheme regardless.
        mask: Optional 2-D boolean land mask aligned with ``(lat, lon)``;
            ``True`` marks a land cell, ``False`` marks ocean. Assumed
            time-invariant (wet-and-dry is out of scope). ``None`` (default)
            means no land logic — ``Field.interp`` is plain bilinear. When
            a mask is present, ``Field.interp`` switches to inverse-distance
            partial-cell weighting that consults it to drop land corners.
    """

    values: Float[Array, "time lat lon"]
    grid: Grid
    stagger: Literal["center", "u_face", "v_face"] = eqx.field(
        static=True, default="center"
    )
    mask: Bool[Array, "lat lon"] | None = None

    @property
    def t_coords(self) -> Float[Array, "time"]:
        """1-D time coordinates in seconds (from the parent grid)."""
        return self.grid.t_coords

    @property
    def lat_coords(self) -> Float[Array, "lat"]:
        """1-D latitude coordinates in degrees for this field's stagger role."""
        return self.grid.coords_for(self.stagger)[0]

    @property
    def lon_coords(self) -> Float[Array, "lon"]:
        """1-D longitude coordinates in degrees for this field's stagger role."""
        return self.grid.coords_for(self.stagger)[1]

    @property
    def lon_period(self) -> float | None:
        """Longitude period for this field's stagger role (``None`` on faces)."""
        return self.grid.period_for(self.stagger)

    @classmethod
    def standalone(
        cls,
        values: Float[Array, "time lat lon"],
        t_coords: Float[Array, "time"],
        lat_coords: Float[Array, "lat"],
        lon_coords: Float[Array, "lon"],
        lon_period: float | None = None,
        stagger: Literal["center", "u_face", "v_face"] = "center",
        mask: Bool[Array, "lat lon"] | None = None,
    ) -> Field:
        """Build a self-contained ``Field`` backed by a private one-field grid.

        Convenience for constructing a ``Field`` outside a :class:`Dataset`
        (e.g. in tests): the given coordinates are stored on a private
        :class:`Grid` in the slot matching ``stagger``, and the returned field
        reads them back through the usual grid-backed properties.

        With ``lon_period`` set, the passed ``lon_coords`` are taken to span
        exactly one period and the field wraps in longitude — for a face
        stagger this means the coordinates must be the seam-inclusive face
        axis (``nlon`` points; see :meth:`pastax.Grid.u_face_coords`), not the
        ``nlon - 1`` interior faces.
        """
        if stagger == "center":
            grid = Grid(
                t_coords=t_coords, lat_coords=lat_coords, lon_coords=lon_coords,
                lon_period=lon_period,
            )
        elif stagger == "u_face":
            grid = Grid(
                t_coords=t_coords, lat_coords=lat_coords, lon_coords=lon_coords,
                stagger_type="C", lon_period=lon_period,
                u_lat_coords=lat_coords, u_lon_coords=lon_coords,
            )
        elif stagger == "v_face":
            grid = Grid(
                t_coords=t_coords, lat_coords=lat_coords, lon_coords=lon_coords,
                stagger_type="C", lon_period=lon_period,
                v_lat_coords=lat_coords, v_lon_coords=lon_coords,
            )
        else:
            raise ValueError(
                f"Unknown stagger {stagger!r}; expected 'center', 'u_face' or "
                "'v_face'."
            )
        return cls(values=values, grid=grid, stagger=stagger, mask=mask)

    def interp(
        self,
        t: Float[Array, ""],
        lon: Float[Array, ""],
        lat: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Trilinearly interpolate the field at a single ``(t, lat, lon)`` point.

        Args:
            t: Query time in seconds.
            lon: Query longitude in degrees.
            lat: Query latitude in degrees.

        Returns:
            Interpolated scalar value at the query point. Outside the grid the
            interpolation extrapolates linearly (clamping to grid boundaries
            beyond one cell). When ``lon_period`` is set, longitude wraps
            instead of extrapolating. When ``self.mask`` is set, coastal
            cells use inverse-distance partial-cell weighting and fully
            land-bound cells return ``0`` (see
            :func:`pastax.interpolation.bilinear_interp_2d`) — the right
            answer for velocities, but an arbitrary fill value for masked
            tracers (an SST of ``0`` over land is a value, not a gap).
        """
        lat_coords, lon_coords = self.grid.coords_for(self.stagger)
        return spatiotemporal_interp(
            self.values, self.grid.t_coords, lat_coords, lon_coords,
            t, lat, lon, lon_period=self.grid.period_for(self.stagger),
            mask=self.mask,
        )

    def neighborhood(
        self,
        t: Float[Array, ""],
        lon: Float[Array, ""],
        lat: Float[Array, ""],
        t_window: int = 1,
        lat_window: int = 1,
        lon_window: int = 1,
    ) -> Float[Array, "wt wlat wlon"]:
        """Extract a window of raw grid values centred on the nearest grid point.

        Args:
            t: Query time in seconds.
            lon: Query longitude in degrees.
            lat: Query latitude in degrees.
            t_window: Half-width along the time axis (window size = 2*t_window+1).
            lat_window: Half-width along the latitude axis.
            lon_window: Half-width along the longitude axis.

        Returns:
            Array of shape (2*t_window+1, 2*lat_window+1, 2*lon_window+1).
            Time and latitude windows are clamped to the grid boundary near the
            edges. The longitude window wraps modulo ``lon_period`` when that
            attribute is set, otherwise it is clamped like the others.
        """
        nt   = self.t_coords.shape[0]
        nlat = self.lat_coords.shape[0]
        nlon = self.lon_coords.shape[0]

        wt   = 2 * t_window   + 1
        wlat = 2 * lat_window + 1
        wlon = 2 * lon_window + 1

        it   = _nearest_idx(self.t_coords,   t,   nt)
        ilat = _nearest_idx(self.lat_coords, lat, nlat)

        it_start   = jnp.clip(it   - t_window,   0, nt   - wt)
        ilat_start = jnp.clip(ilat - lat_window, 0, nlat - wlat)

        if self.lon_period is None:
            ilon = _nearest_idx(self.lon_coords, lon, nlon)
            ilon_start = jnp.clip(ilon - lon_window, 0, nlon - wlon)
            return jax.lax.dynamic_slice(
                self.values,
                (it_start, ilat_start, ilon_start),
                (wt, wlat, wlon),
            )

        ilon = _nearest_idx_periodic(self.lon_coords, lon, nlon, self.lon_period)
        block = jax.lax.dynamic_slice(
            self.values,
            (it_start, ilat_start, jnp.astype(0, jnp.int32)),
            (wt, wlat, nlon),
        )
        lon_idx = (ilon - lon_window + jnp.arange(wlon)) % nlon
        return block[:, :, lon_idx]


class Dataset(eqx.Module):
    """Collection of named :class:`Field` instances sharing a common grid.

    Attributes:
        fields: Mapping ``{field_name: Field}``. For A-grid datasets every
            field lives at cell centres; for C-grid datasets velocity
            fields live on their respective faces (see :attr:`Field.stagger`).
        grid: The shared :class:`Grid` owning the coordinates of every field
            (centre coordinates plus, for C-grids, the staggered U/V-face
            coordinates) and the stagger type of the underlying ocean grid.
            All loaders populate it — A-grid datasets carry a ``stagger_type="A"``
            grid, C-grid datasets a ``stagger_type="C"`` grid. ``None`` is
            accepted when constructing a :class:`Dataset` directly.
    """

    fields: dict[str, Field]
    grid: Grid | None = None

    def __getitem__(self, name: str) -> Field:
        """Return the :class:`Field` registered under ``name``.

        Args:
            name: Field name as registered in ``fields``.

        Returns:
            The :class:`Field` instance.
        """
        return self.fields[name]

    def neighborhood(
        self,
        t: Float[Array, ""],
        lon: Float[Array, ""],
        lat: Float[Array, ""],
        t_window: int = 1,
        lat_window: int = 1,
        lon_window: int = 1,
    ) -> dict[str, Float[Array, "wt wlat wlon"]]:
        """Extract a neighbourhood patch from every field at one query point.

        Equivalent to calling :meth:`Field.neighborhood` on every field with
        the same query and window arguments. Useful for SDE terms that need
        local spatial gradients (e.g. Smagorinsky-style diffusion).

        Args:
            t: Query time in seconds.
            lon: Query longitude in degrees.
            lat: Query latitude in degrees.
            t_window: Half-width along the time axis (window size = ``2*t_window+1``).
            lat_window: Half-width along the latitude axis.
            lon_window: Half-width along the longitude axis.

        Returns:
            Mapping ``{field_name: array}`` where each array has shape
            ``(2*t_window+1, 2*lat_window+1, 2*lon_window+1)``.
        """
        return {
            name: field.neighborhood(t, lon, lat, t_window, lat_window, lon_window)
            for name, field in self.fields.items()
        }

    def velocity_interp(
        self,
        t: Float[Array, ""],
        lon: Float[Array, ""],
        lat: Float[Array, ""],
        *,
        scheme: Literal["default", "partialslip"] = "default",
        u_name: str = "u",
        v_name: str = "v",
        slip_a: float = 0.5,
        slip_b: float = 0.5,
    ) -> Float[Array, "2"]:
        r"""Interpolate the ``(U, V)`` velocity vector at a single point.

        Returns ``[u_value, v_value]``, which already matches the
        :math:`[\mathrm{d}lon/\mathrm{d}t,\ \mathrm{d}lat/\mathrm{d}t]` ordering
        of a ``[lon, lat]`` solver term — feed it straight through the usual
        metres-to-degrees conversion with no component swap.

        Args:
            t: Query time in seconds.
            lon: Query longitude in degrees.
            lat: Query latitude in degrees.
            scheme: Coastal interpolation scheme.

                * ``"default"`` (default) — composes per-field
                  :meth:`Field.interp` for ``V`` and ``U``. Each field uses
                  its own scheme as configured by its mask: bilinear with
                  inverse-distance partial-cell weighting when a mask is
                  present, plain bilinear otherwise.
                * ``"partialslip"`` — A-grid only. Reads ``U`` and ``V``
                  together with their joint land mask (the AND of both
                  fields' masks) and applies a wall-slip correction
                  whenever a full cell edge is land:
                  ``U`` is rescaled by
                  :math:`(\mathrm{slip\_a} + \mathrm{slip\_b}\,w_l)` near a
                  latitudinal coast and ``V`` by
                  :math:`(\mathrm{slip\_a} + \mathrm{slip\_b}\,w_j)`
                  near a longitudinal coast. The default :math:`a = b = 0.5`
                  gives a half-slip wall; :math:`a = 1,\ b = 0` recovers
                  full free-slip. Requires both U and V to carry a mask;
                  raises ``ValueError`` otherwise. Raises
                  ``NotImplementedError`` on Arakawa C-grid datasets.

                  The joint mask is the **AND** of the U and V masks: a
                  corner counts as land only when *both* components are
                  masked there. If the two masks differ, a corner masked
                  in just one component is treated as ocean and that
                  component's stored value — ``0`` after NaN filling —
                  is blended into the interpolation, biasing the
                  velocity toward zero. On A-grid data U and V normally
                  share one land mask, which makes the AND exact; supply
                  identical masks (or fix the discrepancy upstream) when
                  they differ.

            u_name: Name of the U-component Field in ``self.fields``.
            v_name: Name of the V-component Field in ``self.fields``.
            slip_a: Wall slip coefficient (partial-slip only).
            slip_b: Wall slip gradient coefficient (partial-slip only).

        Returns:
            ``[u, v]`` velocity vector of shape ``(2,)``.
        """
        u_field = self.fields[u_name]
        v_field = self.fields[v_name]

        if scheme == "default":
            u = u_field.interp(t, lon, lat)
            v = v_field.interp(t, lon, lat)
            return jnp.stack([u, v])

        if scheme == "partialslip":
            # Check the fields' stagger roles, not just the dataset-level
            # grid: a Dataset built directly (grid=None) around face-staggered
            # fields must not fall through to the A-grid maths below, which
            # would read V values on U-face coordinates.
            if (
                (self.grid is not None and self.grid.stagger_type == "C")
                or u_field.stagger != "center"
                or v_field.stagger != "center"
            ):
                raise NotImplementedError(
                    "scheme='partialslip' is not implemented for Arakawa "
                    "C-grid datasets. Use scheme='default' (per-Field "
                    "inverse-distance) or convert to A-grid forcing."
                )
            if u_field.mask is None or v_field.mask is None:
                raise ValueError(
                    "scheme='partialslip' requires both U and V fields to "
                    "carry a 2-D land mask. Provide masks via the loader "
                    "(NaN-inferred or explicit)."
                )
            joint_mask = u_field.mask & v_field.mask
            u, v = spatiotemporal_velocity_partialslip(
                u_field.values, v_field.values,
                u_field.t_coords, u_field.lat_coords, u_field.lon_coords,
                t, lat, lon, joint_mask,
                slip_a=slip_a, slip_b=slip_b,
                lon_period=u_field.lon_period,
            )
            return jnp.stack([u, v])

        raise ValueError(
            f"Unknown velocity_interp scheme {scheme!r}; "
            "expected 'default' or 'partialslip'."
        )

    @staticmethod
    def from_arrays(
        fields: dict[str, Array],
        t: Array,
        lat: Array,
        lon: Array,
        dtype: DTypeLike = jnp.float32,
        lon_period: float | None = None,
        masks: dict[str, Array] | None = None,
    ) -> Dataset:
        """Build a Dataset from numpy or JAX arrays.

        Args:
            fields: Mapping {field_name: array of shape (time, lat, lon)}.
            t: 1-D time coordinate array. Either equally-spaced numeric values
                (seconds since the Unix epoch) or a NumPy ``datetime64``
                array (any unit); the latter is auto-converted to seconds
                relative to the Unix epoch.
                Numeric input is used as-is.
                Double precision should be used to avoid truncation errors.
            lat: 1-D latitude coordinate array (degrees), equally spaced.
            lon: 1-D longitude coordinate array (degrees), equally spaced.
            dtype: JAX dtype for all arrays, except for the time coordinate 
                (default float32).
            lon_period: If set (e.g. ``360.0``), all fields are constructed
                with periodic longitude wrapping. The grid must span exactly
                one period.
            masks: Optional ``{field_name: 2-D bool array of shape (lat, lon)}``
                land masks. ``True`` marks a land cell. When a field appears
                in ``masks``, that mask is used. Otherwise a mask is inferred
                from NaN locations in the values array (collapsed across the
                time axis). Fields with neither user-supplied nor inferred
                NaN entries carry ``mask=None``. NaN values
                in the input are always replaced with 0 in the stored
                ``values`` so no NaN can leak into interpolation.

        Returns:
            Dataset with all fields on the given grid.
        """
        t = _coerce_time_to_seconds(t)
        t_arr   = _asarray_time_int64(t)
        lat_arr = jnp.asarray(lat, dtype=dtype)
        lon_arr = jnp.asarray(lon, dtype=dtype)
        _check_periodic_lon_ascending(lon_arr, lon_period)
        nt   = int(t_arr.shape[0])
        nlat = int(lat_arr.shape[0])
        nlon = int(lon_arr.shape[0])
        grid = Grid(
            t_coords=t_arr,
            lat_coords=lat_arr,
            lon_coords=lon_arr,
            grid_type="rectilinear",
            stagger_type="A",
            lon_period=lon_period,
        )
        masks = masks or {}
        unknown = set(masks) - set(fields)
        if unknown:
            raise ValueError(
                f"masks contains keys {sorted(unknown)!r} that match no field; "
                f"known fields are {sorted(fields)!r}."
            )
        loaded: dict[str, Field] = {}
        for name, v in fields.items():
            v_arr = jnp.asarray(v, dtype=dtype)
            if v_arr.shape != (nt, nlat, nlon):
                raise ValueError(
                    f"fields[{name!r}]: expected shape (time, lat, lon) = "
                    f"{(nt, nlat, nlon)} matching the coordinate arrays, got "
                    f"{v_arr.shape}. Check the axis order — data stored as "
                    "(time, lon, lat) must be transposed."
                )
            clean, mask = _resolve_mask(
                v_arr, masks.get(name),
                expected_mask_shape=(nlat, nlon),
                field_name=name,
            )
            loaded[name] = Field(
                values=clean,
                grid=grid,
                stagger="center",
                mask=mask,
            )
        return Dataset(fields=loaded, grid=grid)

    @staticmethod
    def from_xarray(
        ds: xr.Dataset,
        fields: dict[str, str],
        coordinates: dict[str, str],
        dtype: DTypeLike = jnp.float32,
        lon_period: float | None = None,
        masks: dict[str, Array] | None = None,
    ) -> Dataset:
        """Load a Dataset from an xarray Dataset (zarr or netCDF backed).

        Args:
            ds: Source xarray Dataset.
            fields: Mapping {internal_name: xarray_variable_name}.
            coordinates: Mapping with keys "time", "lat", "lon" → xarray coord names.
            dtype: JAX dtype for all arrays (default float32).
            lon_period: If set (e.g. ``360.0``), all fields are constructed
                with periodic longitude wrapping. The grid must span exactly
                one period.
            masks: Optional land masks keyed by internal field name; see
                :meth:`from_arrays` for semantics. If omitted, masks are
                inferred from NaN — which matches the CMEMS / CF
                ``_FillValue`` convention.

        Returns:
            Dataset with all fields loaded into host memory as JAX arrays.
        """
        field_arrays = {
            internal: ds[xr_name].values for internal, xr_name in fields.items()
        }
        return Dataset.from_arrays(
            field_arrays,
            t=ds[coordinates["time"]].values,
            lat=ds[coordinates["lat"]].values,
            lon=ds[coordinates["lon"]].values,
            dtype=dtype,
            lon_period=lon_period,
            masks=masks,
        )

    @staticmethod
    def from_arrays_cgrid(
        t: Array,
        center_lat: Array,
        center_lon: Array,
        vectors: dict[str, dict[Literal["u", "v"], tuple[str, Array]]],
        tracers: dict[str, Array] | None = None,
        *,
        u_lat: Array | None = None,
        u_lon: Array | None = None,
        v_lat: Array | None = None,
        v_lon: Array | None = None,
        dtype: DTypeLike = jnp.float32,
        lon_period: float | None = None,
        masks: dict[str, Array] | None = None,
    ) -> Dataset:
        """Build a Dataset on a NEMO-convention Arakawa C-grid.

        The centre grid ``(center_lat, center_lon)`` carries any tracer
        fields. Each vector field has its U component on the east faces
        of the centre cells (one fewer longitude column) and its V
        component on the north faces (one fewer latitude row). Several
        vector fields can share the same C-grid — e.g. surface current
        and 10-m wind, or geostrophic / Ekman / Stokes velocity
        components — by registering additional entries in ``vectors``.
        When the staggered coordinate arrays are omitted they are
        auto-derived from the centre grid as half-cell shifts (see
        :meth:`Grid.u_face_coords` / :meth:`Grid.v_face_coords`) and
        shared by every registered vector.

        Args:
            t: 1-D time coordinates (seconds or NumPy ``datetime64``).
            center_lat: 1-D centre latitudes (degrees), equally spaced.
            center_lon: 1-D centre longitudes (degrees), equally spaced.
            vectors: Mapping ``{group_name: {"u": (field_name, u_array),
                "v": (field_name, v_array)}}``. The outer key is a free-form
                label for the vector pair (e.g. ``"current"``, ``"wind"``,
                ``"geostrophic"``) and is used only in error messages. The
                inner ``(field_name, array)`` tuples give the names under
                which each component is registered in ``Dataset.fields``
                (and how :meth:`velocity_interp` finds them via ``u_name`` /
                ``v_name``) and the corresponding values. U arrays have
                shape ``(time, nlat, nlon - 1)`` on a bounded grid, or
                ``(time, nlat, nlon)`` when ``lon_period`` is set (the seam
                east face is then represented); V arrays have shape
                ``(time, nlat - 1, nlon)``. At least one vector must be
                supplied; field names must be unique across all vectors
                and tracers.
            tracers: Optional mapping ``{name: array of shape (time, nlat, nlon)}``
                for additional fields at cell centres.
            u_lat: Override for U latitudes (defaults to ``center_lat``).
                Shared by every registered U field.
            u_lon: Override for U longitudes (defaults to centre lons shifted
                east by half a cell: length ``nlon - 1`` on a bounded grid, or
                ``nlon`` when ``lon_period`` is set — the seam face included).
                Shared by every registered U field.
            v_lat: Override for V latitudes (defaults to centre lats shifted
                north by half a cell, length ``nlat - 1``). Shared by every
                registered V field.
            v_lon: Override for V longitudes (defaults to ``center_lon``).
                Shared by every registered V field.
            dtype: JAX dtype for all arrays (default float32).
            lon_period: If set (e.g. ``360.0``), the centre grid is treated
                as periodic in longitude and every field wraps across the
                seam. Tracer and V-face fields keep their ``nlon`` longitude
                columns; U-face fields gain the seam east face, so U is
                ``nlon``-wide (one face per centre cell) rather than
                ``nlon - 1``, and its default longitudes are the centre lons
                shifted east by half a cell (see :meth:`Grid.u_face_coords`).
                All roles share the centre period (see
                :meth:`pastax.Grid.period_for`).
            masks: Optional land masks keyed by the field names declared in
                ``vectors`` and ``tracers``. Each mask is a 2-D bool array;
                the expected shape per field is ``(nlat, nlon - 1)`` (or
                ``(nlat, nlon)`` when ``lon_period`` is set) for a U-face
                field, ``(nlat - 1, nlon)`` for a V-face field, and
                ``(nlat, nlon)`` for a tracer. When a field is absent from
                ``masks``, a mask is inferred from NaN locations in that
                field's values array. NaN values are always replaced with
                0 in the stored ``values``.

        Returns:
            Dataset with one ``Field(stagger="u_face")`` and one
            ``Field(stagger="v_face")`` per registered vector (plus any
            tracers) and a C-grid :class:`Grid` metadata object.
        """
        if not vectors:
            raise ValueError(
                "from_arrays_cgrid requires at least one vector field; got "
                "an empty 'vectors' mapping."
            )

        t = _coerce_time_to_seconds(t)
        t_arr   = _asarray_time_int64(t)
        lat_arr = jnp.asarray(center_lat,  dtype=dtype)
        lon_arr = jnp.asarray(center_lon,  dtype=dtype)
        _check_periodic_lon_ascending(lon_arr, lon_period)

        nt   = int(t_arr.shape[0])
        nlat = int(lat_arr.shape[0])
        nlon = int(lon_arr.shape[0])

        centre_grid = Grid(
            t_coords=t_arr,
            lat_coords=lat_arr,
            lon_coords=lon_arr,
            grid_type="rectilinear",
            stagger_type="C",
            lon_period=lon_period,
        )
        derived_u_lat, derived_u_lon = centre_grid.u_face_coords()
        derived_v_lat, derived_v_lon = centre_grid.v_face_coords()

        # A periodic centre grid has one U face per centre cell (the seam face
        # included), so U is nlon-wide; a bounded grid has nlon - 1 interior
        # faces. V is unaffected (latitude is never periodic).
        nlon_u = nlon if lon_period is not None else nlon - 1

        u_lat_arr = jnp.asarray(u_lat, dtype=dtype) if u_lat is not None else derived_u_lat
        u_lon_arr = jnp.asarray(u_lon, dtype=dtype) if u_lon is not None else derived_u_lon
        v_lat_arr = jnp.asarray(v_lat, dtype=dtype) if v_lat is not None else derived_v_lat
        v_lon_arr = jnp.asarray(v_lon, dtype=dtype) if v_lon is not None else derived_v_lon
        _check_cgrid_shape("u_lat", u_lat_arr.shape, (nlat,))
        _check_cgrid_shape("u_lon", u_lon_arr.shape, (nlon_u,))
        _check_cgrid_shape("v_lat", v_lat_arr.shape, (nlat - 1,))
        _check_cgrid_shape("v_lon", v_lon_arr.shape, (nlon,))

        # The grid is the single owner of coordinates: store the resolved
        # staggered coords on it so every field can read them by stagger role.
        grid = Grid(
            t_coords=t_arr,
            lat_coords=lat_arr,
            lon_coords=lon_arr,
            grid_type="rectilinear",
            stagger_type="C",
            lon_period=lon_period,
            u_lat_coords=u_lat_arr,
            u_lon_coords=u_lon_arr,
            v_lat_coords=v_lat_arr,
            v_lon_coords=v_lon_arr,
        )

        masks = masks or {}
        loaded: dict[str, Field] = {}

        for group, entry in vectors.items():
            if set(entry.keys()) != {"u", "v"}:
                raise ValueError(
                    f"vectors[{group!r}] must declare exactly the keys "
                    f"{{'u', 'v'}}; got {sorted(entry.keys())!r}."
                )
            u_field_name, u_values = entry["u"]
            v_field_name, v_values = entry["v"]
            if u_field_name in loaded:
                raise ValueError(
                    f"Duplicate field name {u_field_name!r} declared by "
                    f"vectors[{group!r}]['u']."
                )
            if v_field_name in loaded:
                raise ValueError(
                    f"Duplicate field name {v_field_name!r} declared by "
                    f"vectors[{group!r}]['v']."
                )

            u_arr = jnp.asarray(u_values, dtype=dtype)
            v_arr = jnp.asarray(v_values, dtype=dtype)
            _check_cgrid_shape(
                f"vectors[{group!r}]['u'] ({u_field_name!r})",
                u_arr.shape, (nt, nlat, nlon_u),
            )
            _check_cgrid_shape(
                f"vectors[{group!r}]['v'] ({v_field_name!r})",
                v_arr.shape, (nt, nlat - 1, nlon),
            )

            u_clean, u_mask = _resolve_mask(
                u_arr, masks.get(u_field_name),
                expected_mask_shape=(nlat, nlon_u), field_name=u_field_name,
            )
            v_clean, v_mask = _resolve_mask(
                v_arr, masks.get(v_field_name),
                expected_mask_shape=(nlat - 1, nlon), field_name=v_field_name,
            )
            loaded[u_field_name] = Field(
                values=u_clean, grid=grid, stagger="u_face", mask=u_mask,
            )
            loaded[v_field_name] = Field(
                values=v_clean, grid=grid, stagger="v_face", mask=v_mask,
            )

        if tracers:
            for name, arr in tracers.items():
                if name in loaded:
                    raise ValueError(
                        f"Duplicate field name {name!r}: tracer collides "
                        "with a vector component."
                    )
                a = jnp.asarray(arr, dtype=dtype)
                _check_cgrid_shape(f"tracers[{name!r}]", a.shape, (nt, nlat, nlon))
                tr_clean, tr_mask = _resolve_mask(
                    a, masks.get(name),
                    expected_mask_shape=(nlat, nlon), field_name=name,
                )
                loaded[name] = Field(
                    values=tr_clean, grid=grid, stagger="center", mask=tr_mask,
                )

        unknown = set(masks) - set(loaded)
        if unknown:
            raise ValueError(
                f"masks contains keys {sorted(unknown)!r} that match no field; "
                f"known fields are {sorted(loaded)!r}."
            )
        return Dataset(fields=loaded, grid=grid)

    @staticmethod
    def from_xarray_cgrid(
        ds: xr.Dataset,
        *,
        vectors: dict[str, dict[Literal["u", "v"], tuple[str, str]]],
        coordinates: dict[str, str],
        tracers: dict[str, str] | None = None,
        staggered_coordinates: dict[str, str] | None = None,
        dtype: DTypeLike = jnp.float32,
        lon_period: float | None = None,
        masks: dict[str, Array] | None = None,
    ) -> Dataset:
        """Load a C-grid Dataset from an xarray Dataset.

        Centre coordinates (used for time and tracer fields) come from
        ``coordinates``; staggered U/V coordinates are auto-derived from
        the centre grid as half-cell shifts unless overridden via
        ``staggered_coordinates``. Multiple vector fields living on the
        same C-grid (e.g. surface current and 10-m wind) are declared as
        separate entries in ``vectors``.

        Args:
            ds: Source xarray Dataset.
            vectors: Mapping ``{group_name: {"u": (field_name, xarray_var_name),
                "v": (field_name, xarray_var_name)}}``. The outer key is a
                free-form label for the vector pair (e.g. ``"current"``,
                ``"wind"``). Each inner ``(field_name, xarray_var_name)``
                tuple says which xarray variable holds the values and under
                which name to register the resulting Field in
                ``Dataset.fields``. U variables have shape
                ``(time, nlat, nlon - 1)`` — or ``(time, nlat, nlon)`` when
                ``lon_period`` is set (the seam east face is represented);
                V variables have shape ``(time, nlat - 1, nlon)``.
            coordinates: Mapping with keys ``"time"``, ``"lat"``, ``"lon"``
                → xarray coord names for the centre grid.
            tracers: Optional ``{internal_name: xarray_variable_name}`` for
                extra centre-grid fields.
            staggered_coordinates: Optional override mapping with any
                subset of keys ``"u_lat"``, ``"u_lon"``, ``"v_lat"``,
                ``"v_lon"`` → xarray coord names. Unspecified keys are
                auto-derived. The overrides are shared by every registered
                vector.
            dtype: JAX dtype for all arrays (default float32).
            lon_period: Forwarded to :meth:`from_arrays_cgrid`.
            masks: Forwarded to :meth:`from_arrays_cgrid`. Keys must match
                the field names declared in ``vectors`` and ``tracers``.

        Returns:
            Dataset with C-grid stagger and :class:`Grid` metadata.
        """
        stag = staggered_coordinates or {}
        arr_vectors: dict[str, dict[Literal["u", "v"], tuple[str, Array]]] = {}
        for group, entry in vectors.items():
            if set(entry.keys()) != {"u", "v"}:
                raise ValueError(
                    f"vectors[{group!r}] must declare exactly the keys "
                    f"{{'u', 'v'}}; got {sorted(entry.keys())!r}."
                )
            u_field_name, u_xr = entry["u"]
            v_field_name, v_xr = entry["v"]
            arr_vectors[group] = {
                "u": (u_field_name, ds[u_xr].values),
                "v": (v_field_name, ds[v_xr].values),
            }
        return Dataset.from_arrays_cgrid(
            t=ds[coordinates["time"]].values,
            center_lat=ds[coordinates["lat"]].values,
            center_lon=ds[coordinates["lon"]].values,
            vectors=arr_vectors,
            tracers={
                internal: ds[xr_name].values
                for internal, xr_name in (tracers or {}).items()
            } or None,
            u_lat=ds[stag["u_lat"]].values if "u_lat" in stag else None,
            u_lon=ds[stag["u_lon"]].values if "u_lon" in stag else None,
            v_lat=ds[stag["v_lat"]].values if "v_lat" in stag else None,
            v_lon=ds[stag["v_lon"]].values if "v_lon" in stag else None,
            dtype=dtype,
            lon_period=lon_period,
            masks=masks,
        )


def _is_traced(x: Array) -> bool:
    """True when ``x`` is an abstract JAX tracer (inside ``jit`` / ``vmap`` / ``grad``).

    Host-side validation that must read concrete values (a ``bool(...)`` on an
    array) is skipped when this returns ``True`` so the loaders stay callable
    from within a traced region; the caller then owns coordinate correctness.
    """
    return isinstance(x, jax.core.Tracer)


def _check_periodic_lon_ascending(
    lon_arr: Float[Array, "lon"], lon_period: float | None
) -> None:
    """Reject descending longitudes when periodic wrapping is requested.

    The periodic index arithmetic folds queries with ``% lon_period`` above
    ``lon_coords[0]``, which assumes an ascending axis; a descending axis
    silently produces wrong indices and weights. (Without ``lon_period``,
    descending coordinates work — the spacing sign cancels.)

    The check reads concrete values, so it is a no-op when ``lon_arr`` is a
    tracer (the loader was called from inside ``jit`` / ``vmap``): correctness
    of the coordinate axis is then the caller's responsibility.
    """
    if lon_period is None:
        return
    if _is_traced(lon_arr):
        return
    if not bool(jnp.all(jnp.diff(lon_arr) > 0)):
        raise ValueError(
            "lon_period requires strictly ascending longitude coordinates; "
            "the periodic wrap arithmetic is undefined on a descending axis. "
            "Flip the longitude axis (and the field values) before loading."
        )


def _check_cgrid_shape(name: str, got: tuple[int, ...], expected: tuple[int, ...]) -> None:
    if got != expected:
        raise ValueError(
            f"C-grid shape mismatch for {name}: expected {expected}, got {got}. "
            f"NEMO convention requires U at shape (time, nlat, nlon-1) — or "
            f"(time, nlat, nlon) on a periodic grid (lon_period set) — and "
            f"V at shape (time, nlat-1, nlon)."
        )


def _resolve_mask(
    values: Float[Array, "time lat lon"],
    user_mask: Array | None,
    *,
    expected_mask_shape: tuple[int, int],
    field_name: str,
) -> tuple[Float[Array, "time lat lon"], Bool[Array, "lat lon"] | None]:
    """Replace NaN in ``values`` with 0 and resolve the field's land mask.

    Rules (in order):

    1. NaN values are always replaced with 0 in the returned values array.
    2. If ``user_mask`` is provided, it is validated against
       ``expected_mask_shape`` and used as-is, and the values under the mask
       are zeroed. Datasets that flag land with a fill value (e.g. ``1e20``)
       instead of NaN would otherwise leak those fill values into the
       interpolation paths that still read land corners (the partial-slip
       naive fallback and its all-corners-land case). Time-varying (3-D)
       masks are rejected.
    3. Otherwise, if NaN was present in ``values``, infer a 2-D mask from
       ``isnan(values).any(axis=0)``. Cells that are NaN at *some but not
       all* time steps are almost certainly missing data (sensor gaps)
       rather than land — land is time-invariant. Such cells still end up
       masked (and zero-filled), but a ``UserWarning`` is emitted so the
       gap does not silently become a permanent wall; pass an explicit
       mask to silence it.
    4. If no user mask and no NaN, return ``mask=None`` — the resulting
       Field is bit-exact identical (PyTree structure included) to one
       built before the mask feature was added.

    NaN-mask inference (step 3) reads concrete values *and* would make the
    returned PyTree structure data-dependent (``mask=None`` vs a bool array),
    which is illegal under tracing. It is therefore skipped when ``values`` is
    a tracer: the NaN→0 replacement still happens (a safe traced op), but the
    mask is left ``None`` unless an explicit ``user_mask`` is given. Pass
    ``masks=`` when building a Dataset from inside ``jit`` and you need coasts.

    Returns:
        ``(clean_values, mask)`` where ``mask`` is either a 2-D bool array
        or ``None``.
    """
    nan_locs = jnp.isnan(values)
    clean_values = jnp.where(nan_locs, jnp.asarray(0.0, dtype=values.dtype), values)

    if user_mask is not None:
        mask = jnp.asarray(user_mask, dtype=jnp.bool_)
        if mask.shape != expected_mask_shape:
            raise ValueError(
                f"masks[{field_name!r}]: expected 2-D bool array of shape "
                f"{expected_mask_shape}, got {mask.shape}. Time-varying masks "
                "(wet-and-dry) are not supported."
            )
        clean_values = jnp.where(
            mask, jnp.asarray(0.0, dtype=clean_values.dtype), clean_values
        )
        return clean_values, mask

    if _is_traced(values):
        return clean_values, None

    if bool(nan_locs.any()):
        inferred = nan_locs.any(axis=0)
        transient = inferred & ~nan_locs.all(axis=0)
        if bool(transient.any()):
            import warnings

            warnings.warn(
                f"Field {field_name!r}: {int(transient.sum())} cell(s) are NaN "
                "at some but not all time steps. Land is time-invariant, so "
                "these look like missing data (sensor gaps), yet they will be "
                "masked as land for the whole simulation (values zero-filled). "
                "Fill the gaps or pass an explicit mask to silence this "
                "warning.",
                UserWarning,
                stacklevel=2,
            )
        return clean_values, inferred

    return clean_values, None
