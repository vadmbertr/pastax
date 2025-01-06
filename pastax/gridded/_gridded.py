from __future__ import annotations

from typing import Callable, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Bool, Float, Int, Scalar

from ..utils._unit import degrees_to_meters, meters_to_degrees
from ._coordinate import Coordinate, LongitudeCoordinate
from ._field import SpatialField, SpatioTemporalField


class Gridded(eqx.Module):
    """
    Class providing some routines for handling gridded spatiotemporal data in JAX.

    Attributes
    ----------
    cell_area : Float[Array, "lat lon"]
        Array of cell areas in square meters.
    coordinates : Coordinates
        Coordinates object containing time, latitude, and longitude.
    dx : Float[Array, "lat lon-1"]
        Array of longitudinal distances in meters.
    dy : Float[Array, "lat-1 lon"]
        Array of latitudinal distances in meters.
    fields : dict[str, SpatioTemporalField]
        Dictionary of spatiotemporal fields.
    is_masked : SpatialField
        Spatial field used for masking (`True` means masked, `False` not masked).
    is_spherical_mesh: bool
        Boolean indicating whether the mesh uses spherical coordinates.
    interpolation_method: Literal["nearest", "linear", "cubic", "cubic2", "catmull-rom", "cardinal", "monotonic", "monotonic-0", "akima"]
        String indicating the interpolation method used when interpolating the fields.
        For details, see [`interpax` documentation](https://interpax.readthedocs.io/en/latest/index.html).
    use_degrees : bool
        Boolean indicating whether distance units are degrees.

    Methods
    -------
    indices(time, latitude, longitude)
        Gets nearest indices of the spatio-temporal point `time`, `latitude`, `longitude`.
    interp(*fields, **coordinates)
        Interpolates the given fields at the given coordinates.
    neighborhood(*fields, time, latitude, longitude, t_width, x_width)
        Extracts a neighborhood of data around a specified point in time and space.
    to_xarray()
        Returns the [`pastax.gridded.Gridded`][] object as a `xarray.Dataset`.
    from_array(fields, time, latitude, longitude, interpolation_method="linear", is_spherical_mesh=True,
            use_degrees=False, is_uv_mps=True)
        Constructs a [`pastax.gridded.Gridded`][] object from arrays of fields and coordinates `time`, `latitude`,
        `longitude`.
    from_xarray(dataset, fields, coordinates, interpolation_method="linear", is_spherical_mesh=True, use_degrees=False,
            is_uv_mps=True)
        Constructs a [`pastax.gridded.Gridded`][] object from a `xarray.Dataset`.
    xarray_to_array(dataset, fields, coordinates, transform_fn=lambda x: jnp.asarray(x, dtype=float))
        Converts an `xarray.Dataset` to arrays of fields and coordinates.
    """

    coordinates: dict[str, Coordinate | LongitudeCoordinate]
    dx: Float[Array, "lat lon-1"]
    dy: Float[Array, "lat-1 lon"]
    cell_area: Float[Array, "lat lon"]
    fields: dict[str, SpatioTemporalField]
    is_masked: SpatialField
    is_spherical_mesh: bool
    interpolation_method: Literal[
        "nearest", "linear", "cubic", "cubic2", "catmull-rom", "cardinal", "monotonic", "monotonic-0", "akima"
    ]
    use_degrees: bool

    def indices(self, **coordinates: Int[Array, "Nq"] | Float[Array, "Nq"]) -> tuple[Int[Array, "Nq"], ...]:
        """
        Gets the nearest indices of the N-dimensional point specified by the given coordinates.

        Parameters
        ----------
        **coordinates : Int[Array, "Nq"] | Float[Array, "Nq"]
            The N-dimensional point to get the nearest indices.

        Returns
        -------
        tuple[Int[Array, "Nq"], ...]
            A tuple of arrays containing the nearest indices of the N-dimensional point.
        """
        return tuple(self.coordinates[k].index(v) for k, v in coordinates.items())

    def interp(
        self, *fields: str, **coordinates: Int[Array, "Nq"] | Float[Array, "Nq"]
    ) -> dict[str, Bool[Array, "Nq ..."] | Float[Array, "Nq ..."] | Int[Array, "Nq ..."]]:
        """
        Interpolates the given fields at the given coordinates.

        Parameters
        ----------
        *fields: str
            Fields names to be interpolated.
        **coordinates : Int[Array, "Nq"] | Float[Array, "Nq"]
            The N-dimensional points to interpolate to.

        Returns
        -------
        dict[str, Bool[Array, "Nq ..."] | Float[Array, "Nq ..."] | Int[Array, "Nq ..."]]
            A dict of arrays containing the interpolated values for each field.
        """
        mask = self.is_masked.interp(**coordinates)

        interpolated_fields = {}
        for field_name in fields:
            field = self.fields[field_name]
            interpolated_field = field.interp(**coordinates)
            interpolated_field = jnp.where(mask, 0, interpolated_field)
            interpolated_fields[field_name] = interpolated_field

        return interpolated_fields

    def neighborhood(
        self,
        *fields: str,
        time: Int[Scalar, ""],
        latitude: Float[Scalar, ""],
        longitude: Float[Scalar, ""],
        t_width: int,
        x_width: int,
    ) -> Gridded:
        """
        Extracts a neighborhood of data around a specified point in time and space.

        Parameters
        ----------
        *fields : tuple[str, ...]
            Fields names to extract from the dataset.
        time : Int[Scalar, ""]
            The time coordinate for the center of the neighborhood.
        latitude : Float[Scalar, ""]
            The latitude coordinate for the center of the neighborhood.
        longitude : Float[Scalar, ""]
            The longitude coordinate for the center of the neighborhood.
        t_width : int
            The width of the neighborhood in the time dimension.
        x_width : int
            The width of the neighborhood in the spatial dimensions (latitude and longitude).

        Returns
        -------
        Dataset
            A [`pastax.gridded.Gridded`][] object restricted to the neighborhing data.
        """
        t_i, lat_i, lon_i = self.indices(time=time, latitude=latitude, longitude=longitude)

        from_t_i = t_i - t_width // 2
        from_lat_i = lat_i - x_width // 2
        from_lon_i = lon_i - x_width // 2

        t_neighborhood = jax.lax.dynamic_slice_in_dim(self.coordinates["time"].values, from_t_i, t_width)
        lat_neighborhood = jax.lax.dynamic_slice_in_dim(self.coordinates["latitude"].values, from_lat_i, x_width)

        def no_edge_cases():
            lon_neighborhood = jax.lax.dynamic_slice_in_dim(self.coordinates["longitude"].values, from_lon_i, x_width)

            fields_neighborhood = dict(
                (
                    field_name,
                    jax.lax.dynamic_slice(
                        self.fields[field_name].values, (from_t_i, from_lat_i, from_lon_i), (t_width, x_width, x_width)
                    ),
                )
                for field_name in fields
            )

            is_masked_neighborhood = jax.lax.dynamic_slice(
                self.is_masked.values, (from_lat_i, from_lon_i), (x_width, x_width)
            )

            return lon_neighborhood, fields_neighborhood, is_masked_neighborhood

        def edge_cases():
            dx = jnp.linspace(-(x_width // 2), x_width // 2, x_width) * self.dx[lat_i, lon_i]
            lon = jnp.full(x_width, longitude) + dx
            lon_indices = self.indices(longitude=lon)[0]

            lon_neighborhood = self.coordinates["longitude"][lon_indices]

            fields_neighborhood = dict(
                (
                    field_name,
                    jax.lax.dynamic_slice(
                        self.fields[field_name].values,
                        (from_t_i, from_lat_i, 0),
                        (t_width, x_width, self.coordinates["longitude"].values.size),
                    )[..., lon_indices],
                )
                for field_name in fields
            )

            is_masked_neighborhood = jax.lax.dynamic_slice_in_dim(self.is_masked.values, from_lat_i, x_width)[
                ..., lon_indices
            ]

            return lon_neighborhood, fields_neighborhood, is_masked_neighborhood

        lon_neighborhood, fields_neighborhood, is_masked_neighborhood = jax.lax.cond(
            (self.is_spherical_mesh and (self.indices(longitude=self.coordinates["longitude"][-1] + self.dx[-1]) == 0))
            and ((from_lon_i < 0) or (from_lon_i + x_width > self.coordinates["longitude"].values.size)),
            edge_cases,
            no_edge_cases,
        )

        return Gridded.from_array(
            fields_neighborhood,
            t_neighborhood,
            lat_neighborhood,
            lon_neighborhood,
            is_masked=is_masked_neighborhood,
            interpolation_method=self.interpolation_method,
            is_spherical_mesh=self.is_spherical_mesh,
            use_degrees=self.use_degrees,
        )

    def to_xarray(self) -> xr.Dataset:
        """
        Converts the [`pastax.gridded.Gridded`][] to a `xarray.Dataset`.

        This method constructs an xarray Dataset from the object's fields and coordinates.
        The fields are added as data variables with coordinates ["time", "latitude", "longitude"].
        The coordinates are added as coordinate variables.

        Returns
        -------
        xr.Dataset
            The corresponding `xarray.Dataset`.
        """
        dataset = xr.Dataset(
            data_vars=dict(
                (var_name, (["time", "latitude", "longitude"], var.values)) for var_name, var in self.fields.items()
            ),
            coords=dict(
                time=np.asarray(self.coordinates["time"].values, dtype="datetime64[s]"),
                latitude=self.coordinates["latitude"].values,
                longitude=self.coordinates["longitude"].values,
            ),
        )

        return dataset

    @classmethod
    def from_array(
        cls,
        fields: dict[str, Float[Array, "time lat lon"]],
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        is_masked: Bool[Array, "lat lon"] | None = None,
        interpolation_method: Literal[
            "nearest", "linear", "cubic", "cubic2", "catmull-rom", "cardinal", "monotonic", "monotonic-0", "akima"
        ] = "linear",
        is_spherical_mesh: bool = True,
        use_degrees: bool = False,
        is_uv_mps: bool = True,
    ) -> Gridded:
        """
        Create a [`pastax.gridded.Gridded`][] object from arrays of fields, time, latitude, and longitude.

        Parameters
        ----------
        fields : dict[str, Float[Array, "time lat lon"]]
            A dictionary where keys are fields names and values are 3D arrays representing
            the field data over time, latitude, and longitude.
        time : Int[Array, "time"]
            A 1D array representing the time dimension.
        latitude : Float[Array, "lat"]
            A 1D array representing the latitude dimension.
        longitude : Float[Array, "lon"]
            A 1D array representing the longitude dimension.
        is_masked : Bool[Array, "lat lon"], optional
            2D array used for masking fields (`True` means masked, `False` not masked).
        interpolation_method : Literal["nearest", "linear", "cubic", "cubic2", "catmull-rom", "cardinal", "monotonic", "monotonic-0", "akima"], optional
            String indicating the interpolation method used when interpolating the fields, defaults to `"linear"`.
            For details, see [`interpax` documentation](https://interpax.readthedocs.io/en/latest/index.html).
        is_spherical_mesh : bool, optional
            Whether the mesh uses spherical coordinate, defaults to `True`.
        use_degrees : bool, optional
            Whether distance units should be degrees rather than meters, defaults to `False`.
        is_uv_mps : bool, optional
            Whether the velocity data is in m/s, defaults to `True`.

        Returns
        -------
        Dataset
            The corresponding [`pastax.gridded.Gridded`][].
        """

        def compute_cell_dlatlon(dright: Float[Array, "latlon-1"], axis: int) -> Float[Array, "latlon"]:
            if axis == 0:
                dcentered = (dright[1:, :] + dright[:-1, :]) / 2
                dstart = ((dright[0, :] - dcentered[0, :] / 2) * 2)[None, :]
                dend = ((dright[-1, :] - dcentered[-1, :] / 2) * 2)[None, :]
            else:
                dcentered = (dright[:, 1:] + dright[:, :-1]) / 2
                dstart = ((dright[:, 0] - dcentered[:, 0] / 2) * 2)[:, None]
                dend = ((dright[:, -1] - dcentered[:, -1] / 2) * 2)[:, None]
            return jnp.concat((dstart, dcentered, dend), axis=axis)

        use_degrees = use_degrees & is_spherical_mesh  # if flat mesh, no reason to use degrees

        time_coord = Coordinate.from_array(time, extrap=True)
        latitude_coord = Coordinate.from_array(latitude, extrap=True)
        longitude_coord = LongitudeCoordinate.from_array(longitude, is_spherical=is_spherical_mesh, extrap=True)

        # compute grid spacings and cells area
        dlat = jnp.diff(latitude)
        dlon = jnp.diff(longitude)

        if is_spherical_mesh and not use_degrees:
            dlatlon = degrees_to_meters(
                jnp.stack([dlat, jnp.zeros_like(dlat)], axis=-1),
                (latitude[:-1] + latitude[1:]) / 2,
            )
            dlat = dlatlon[:, 0]
            _, dlat = jnp.meshgrid(longitude, dlat)

            dlatlon = jax.vmap(
                lambda lat: jax.vmap(
                    lambda _dlon: degrees_to_meters(jnp.stack([jnp.zeros_like(_dlon), _dlon], axis=-1), lat)
                )(dlon)
            )(latitude)
            dlon = dlatlon[:, :, 1]
        else:
            _, dlat = jnp.meshgrid(longitude, dlat)
            dlon, _ = jnp.meshgrid(dlon, latitude)

        cell_dlat = compute_cell_dlatlon(dlat, axis=0)
        cell_dlon = compute_cell_dlatlon(dlon, axis=1)
        cell_area = cell_dlat * cell_dlon

        # compute mask
        if is_masked is None:
            is_masked_arr = jnp.zeros((latitude.size, longitude.size), dtype=bool)
            for field in fields.values():
                _is_masked = jnp.isnan(field).sum(axis=0, dtype=bool)
                is_masked_arr = jnp.logical_or(is_masked_arr, _is_masked)
        else:
            is_masked_arr = is_masked
        is_masked_field = SpatialField.from_array(
            is_masked_arr, latitude_coord.values, longitude_coord.values, interpolation_method="nearest"
        )

        # apply it
        for field_name in fields:
            field = fields[field_name]
            field = jnp.where(is_masked_arr, 0, field)
            fields[field_name] = field

        # if required, convert uv from m/s to °/s
        if use_degrees and is_uv_mps:
            vu = jnp.stack((fields["v"], fields["u"]), axis=-1)
            original_shape = vu.shape
            vu = vu.reshape(vu.shape[0], -1, 2)

            _, lat_grid = jnp.meshgrid(longitude, latitude)
            lat_grid = lat_grid.ravel()

            vu = eqx.filter_vmap(lambda x: meters_to_degrees(x, lat_grid))(vu)
            vu = vu.reshape(original_shape)

            fields["v"] = vu[..., 0]
            fields["u"] = vu[..., 1]

            is_uv_mps = False

        fields_ = dict(
            (
                field_name,
                SpatioTemporalField.from_array(
                    values,
                    time_coord.values,
                    latitude_coord.values,
                    longitude_coord.values,
                    interpolation_method=interpolation_method,
                ),
            )
            for field_name, values in fields.items()
        )

        return cls(
            cell_area=cell_area,
            coordinates={"time": time_coord, "latitude": latitude_coord, "longitude": longitude_coord},
            dx=dlon,
            dy=dlat,
            fields=fields_,
            is_masked=is_masked_field,
            is_spherical_mesh=is_spherical_mesh,
            interpolation_method=interpolation_method,
            use_degrees=use_degrees,
        )

    @classmethod
    def from_xarray(
        cls,
        dataset: xr.Dataset,
        fields: dict[str, str],
        coordinates: dict[str, str],
        interpolation_method: Literal[
            "nearest", "linear", "cubic", "cubic2", "catmull-rom", "cardinal", "monotonic", "monotonic-0", "akima"
        ] = "linear",
        is_spherical_mesh: bool = True,
        is_uv_mps: bool = True,
        use_degrees: bool = False,
    ) -> Gridded:
        """
        Create a [`pastax.gridded.Gridded`][] object from a `xarray.Dataset`.

        Parameters
        ----------
        dataset : xr.Dataset
            The `xarray.Dataset` containing the data.
        fields : dict[str, str]
            A dictionary mapping the target field names (keys) to the source variable names in the dataset (values).
        coordinates : dict[str, str]
            A dictionary mapping the coordinate names ('time', 'latitude', 'longitude') to their corresponding names in
            the dataset.
        interpolation_method: Literal["nearest", "linear", "cubic", "cubic2", "catmull-rom", "cardinal", "monotonic", "monotonic-0", "akima"], optional
            String indicating the interpolation method used when interpolating the fields, defaults to `"linear"`.
            For details, see [`interpax` documentation](https://interpax.readthedocs.io/en/latest/index.html).
        is_spherical_mesh : bool, optional
            Whether the mesh uses spherical coordinate, defaults to `True`.
        is_uv_mps : bool, optional
            Whether the velocity data is in m/s, defaults to `True`.
        use_degrees : bool, optional
            Whether distance unit should be degrees rather than meters, defaults to `False`.

        Returns
        -------
        Dataset
            The corresponding [`pastax.gridded.Gridded`][].
        """
        fields_, t, lat, lon = cls.xarray_to_array(dataset, fields, coordinates)

        return cls.from_array(
            fields_,
            t,
            lat,
            lon,
            interpolation_method=interpolation_method,
            is_spherical_mesh=is_spherical_mesh,
            use_degrees=use_degrees,
            is_uv_mps=is_uv_mps,
        )

    @staticmethod
    def xarray_to_array(
        dataset: xr.Dataset,
        fields: dict[str, str],  # to -> from
        coordinates: dict[str, str],  # to -> from
        transform_fn: Callable[[Array], Array] = lambda x: jnp.asarray(x, dtype=float),
    ) -> tuple[
        dict[str, Float[Array, "..."]],
        Float[Array, "..."],
        Float[Array, "..."],
        Float[Array, "..."],
    ]:
        """
        Converts an `xarray.Dataset` to arrays of fields and coordinates.

        Parameters
        ----------
        dataset : xr.Dataset
            The `xarray.Dataset` to convert.
        fields : dict[str, str]
            A dictionary mapping the target field names to the source variable names in the dataset.
        coordinates : dict[str, str]
            A dictionary mapping the target coordinate names to the source coordinate names in the dataset.
        transform_fn : Callable[[Array], Array], optional
            Function converting dataarrays to JAX (or numpy) arrays,
            defaults to `lambda x: jnp.asarray(x, dtype=float)`.

        Returns
        -------
        tuple[dict[str, Float[Array, "..."]], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]
            A tuple containing:
            - A dictionary of converted fields.
            - The time coordinate array.
            - The latitude coordinate array.
            - The longitude coordinate array.
        """
        fields_ = dict((to_name, transform_fn(dataset[from_name].data)) for to_name, from_name in fields.items())

        t = transform_fn(dataset[coordinates["time"]].data.astype("datetime64[s]").astype(int))
        lat = transform_fn(dataset[coordinates["latitude"]].data)
        lon = transform_fn(dataset[coordinates["longitude"]].data)

        return fields_, t, lat, lon
