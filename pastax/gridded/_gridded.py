from __future__ import annotations

from typing import Callable, Dict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jaxtyping import Array, Bool, Float, Int, Scalar

from ..utils._unit import degrees_to_meters, meters_to_degrees
from ._coordinate import Coordinates
from ._field import SpatioTemporalField


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
    fields : Dict[str, SpatioTemporalField]
        Dictionary of spatiotemporal fields.
    is_land : Bool[Array, "lat lon"]
        Boolean array indicating land presence.
    is_spherical_mesh: bool
        Boolean indicating whether the mesh uses spherical coordinates.
    use_degrees : bool
        Boolean indicating whether distance units are degrees.

    Methods
    -------
    indices(time, latitude, longitude)
        Gets nearest indices of the spatio-temporal point `time`, `latitude`, `longitude`.
    interp_temporal(*fields, time)
        Interpolates the specified fields temporally at the given time.
    interp_spatial(*fields, latitude, longitude)
        Interpolates the specified fields spatially at the given latitude and longitude.
    interp_spatiotemporal(*fields, time, latitude, longitude)
        Interpolates the specified fields spatiotemporally at the given time, latitude, and longitude.
    neighborhood(*fields, time, latitude, longitude, t_width, x_width)
        Gets a neighborhood `t_width`*`x_width`*`x_width` around the spatio-temporal point `time`, `latitude`,
        `longitude`.
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

    coordinates: Coordinates
    dx: Float[Array, "lat lon-1"]
    dy: Float[Array, "lat-1 lon"]
    cell_area: Float[Array, "lat lon"]
    fields: Dict[str, SpatioTemporalField]
    is_land: Bool[Array, "lat lon"]
    is_spherical_mesh: bool
    use_degrees: bool

    def indices(
        self,
        time: Int[Scalar, ""],
        latitude: Float[Scalar, ""],
        longitude: Float[Scalar, ""],
    ) -> tuple[Int[Scalar, ""], Int[Scalar, ""], Int[Scalar, ""]]:
        """
        Get nearest indices of the spatio-temporal point `time`, `latitude`, `longitude`.

        Parameters
        ----------
        time : Int[Scalar, ""]
            The nearest time index.
        latitude : Float[Scalar, ""]
            The nearest time index.
        longitude : Float[Scalar, ""]
            The nearest time index.

        Returns
        -------
        tuple[Int[Scalar, ""], Int[Scalar, ""], Int[Scalar, ""]]
            A tuple of arrays containing the nearest indices of the spatio-temporal point.
        """
        return self.coordinates.indices(time, latitude, longitude)

    def interp_temporal(self, *fields: str, time: Float[Array, "..."]) -> tuple[Float[Array, "... ... ..."], ...]:
        """
        Interpolates the given fields in time.

        Parameters
        ----------
        fields : tuple[str, ...]
            Fields names to be interpolated.
        time : Float[Array, "..."]
            The time points at which to interpolate the fields.

        Returns
        -------
        tuple[Float[Array, "... ... ..."], ...]
            A tuple of arrays containing the interpolated values for each field.
        """
        return tuple(self.fields[field_name].interp_temporal(time) for field_name in fields)

    def interp_spatial(
        self,
        *fields: str,
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."],
    ) -> tuple[Float[Array, "... ... ..."], ...]:
        """
        Interpolates the given fields in space.

        Parameters
        ----------
        *fields : tuple[str, ...]
            Fields names to be interpolated.
        latitude : Float[Array, "..."]
            The latitude values at which to interpolate the fields.
        longitude : Float[Array, "..."]
            The latitude values at which to interpolate the fields.

        Returns
        -------
        tuple[Float[Array, "... ... ..."], ...]
            A tuple of arrays containing interpolated values for each field.
        """
        return tuple(self.fields[field_name].interp_spatial(latitude, longitude) for field_name in fields)

    def interp_spatiotemporal(
        self,
        *fields: str,
        time: Float[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."],
    ) -> tuple[Float[Array, "... ... ..."], ...]:
        """
        Interpolates the specified fields spatiotemporally at the given time, latitude, and longitude.

        Parameters
        ----------
        *fields : tuple[str, ...]
            Field names to interpolate.
        time : Float[Array, "..."]
            The time values at which to interpolate the fields.
        latitude : Float[Array, "..."]
            The latitude values at which to interpolate the fields.
        longitude : Float[Array, "..."]
            The latitude values at which to interpolate the fields.

        Returns
        -------
        tuple[Float[Array, "... ... ..."], ...]
            A tuple of arrays containing the interpolated values for each field.
        """
        return tuple(self.fields[field_name].interp_spatiotemporal(time, latitude, longitude) for field_name in fields)

    def neighborhood(  # TODO: handle edge cases
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
        t_i, lat_i, lon_i = self.indices(time, latitude, longitude)

        from_t_i = t_i - t_width // 2
        from_lat_i = lat_i - x_width // 2
        from_lon_i = lon_i - x_width // 2

        fields_ = dict(
            (
                field_name,
                jax.lax.dynamic_slice(
                    self.fields[field_name].values,
                    (from_t_i, from_lat_i, from_lon_i),
                    (t_width, x_width, x_width),
                ),
            )
            for field_name in fields
        )

        t = jax.lax.dynamic_slice_in_dim(self.coordinates.time.values, from_t_i, t_width)
        lat = jax.lax.dynamic_slice_in_dim(self.coordinates.latitude.values, from_lat_i, x_width)
        lon = jax.lax.dynamic_slice_in_dim(self.coordinates.longitude.values, from_lon_i, x_width)

        return Gridded.from_array(
            fields_,
            t,
            lat,
            lon,
            interpolation_method="linear",
            is_spherical_mesh=self.is_spherical_mesh,
            use_degrees=self.use_degrees,
        )

    def to_xarray(self) -> xr.Dataset:
        """
        Converts the [`pastax.gridded.Gridded`][] to a `xarray.Dataset`.

        This method constructs an xarray Dataset from the object's fields and coordinates.
        The fields are added as data variables with dimensions ["time", "latitude", "longitude"].
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
                time=np.asarray(self.coordinates.time.values, dtype="datetime64[s]"),
                latitude=self.coordinates.latitude.values,
                longitude=self.coordinates.longitude.values,
            ),
        )

        return dataset

    @classmethod
    def from_array(
        cls,
        fields: Dict[str, Float[Array, "time lat lon"]],
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str = "linear",
        is_spherical_mesh: bool = True,
        use_degrees: bool = False,
        is_uv_mps: bool = True,
    ) -> Gridded:
        """
        Create a [`pastax.gridded.Gridded`][] object from arrays of fields, time, latitude, and longitude.

        Parameters
        ----------
        fields : Dict[str, Float[Array, "time lat lon"]]
            A dictionary where keys are fields names and values are 3D arrays representing
            the field data over time, latitude, and longitude.
        time : Int[Array, "time"]
            A 1D array representing the time dimension.
        latitude : Float[Array, "lat"]
            A 1D array representing the latitude dimension.
        longitude : Float[Array, "lon"]
            A 1D array representing the longitude dimension.
        interpolation_method : str, optional
            The method to use for latter possible interpolation of the fields, defaults to `"linear"`.
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

        coordinates = Coordinates.from_array(time, latitude, longitude, is_spherical_mesh)

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

        # compute land mask
        is_land = jnp.zeros((latitude.size, longitude.size), dtype=bool)
        for field in fields.values():
            _is_land = jnp.isnan(field).sum(axis=0, dtype=bool)
            is_land = jnp.logical_or(is_land, _is_land)

        # apply it
        for field_name in fields:
            field = fields[field_name]
            field = jnp.where(is_land, 0, field)
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
                    coordinates.time.values,
                    coordinates.latitude.values,
                    coordinates.longitude.values,
                    interpolation_method=interpolation_method,
                ),
            )
            for field_name, values in fields.items()
        )

        return cls(
            cell_area=cell_area,
            coordinates=coordinates,
            dx=dlon,
            dy=dlat,
            fields=fields_,
            is_land=is_land,
            is_spherical_mesh=is_spherical_mesh,
            use_degrees=use_degrees,
        )

    @classmethod
    def from_xarray(
        cls,
        dataset: xr.Dataset,
        fields: Dict[str, str],
        coordinates: Dict[str, str],
        interpolation_method: str = "linear",
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
        fields : Dict[str, str]
            A dictionary mapping the target field names (keys) to the source variable names in the dataset (values).
        coordinates : Dict[str, str]
            A dictionary mapping the coordinate names ('time', 'latitude', 'longitude') to their corresponding names in
            the dataset.
        interpolation_method : str, optional
            The method to use for latter possible interpolation of the fields, defaults to `"linear"`.
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
            interpolation_method,
            is_spherical_mesh,
            use_degrees,
            is_uv_mps,
        )

    @staticmethod
    def xarray_to_array(
        dataset: xr.Dataset,
        fields: Dict[str, str],  # to -> from
        coordinates: Dict[str, str],  # to -> from
        transform_fn: Callable[[Array], Array] = lambda x: jnp.asarray(x, dtype=float),
    ) -> tuple[
        Dict[str, Float[Array, "..."]],
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
        fields : Dict[str, str]
            A dictionary mapping the target field names to the source variable names in the dataset.
        coordinates : Dict[str, str]
            A dictionary mapping the target coordinate names to the source coordinate names in the dataset.
        transform_fn : Callable[[Array], Array], optional
            Function converting dataarrays to JAX (or numpy) arrays,
            defaults to `lambda x: jnp.asarray(x, dtype=float)`.

        Returns
        -------
        tuple[Dict[str, Float[Array, "..."]], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]
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
