from __future__ import annotations
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar
import numpy as np
import xarray as xr

from ..utils.unit import degrees_to_meters, meters_to_degrees
from ._coordinates import Coordinates
from ._grid import SpatioTemporalField


class Dataset(eqx.Module):
    """
    Class that provides some routines for handling gridded spatiotemporal data in JAX.

    Attributes
    ----------
    is_spherical_mesh: bool
        Boolean indicating whether the mesh uses spherical coordinates.
    use_degrees : bool
        Boolean indicating whether distance units are degrees.
    coordinates : Coordinates
        Coordinates object containing time, latitude, and longitude.
    dx : Float[Array, "lat lon-1"]
        Array of longitudinal distances in meters.
    dy : Float[Array, "lat-1 lon"]
        Array of latitudinal distances in meters.
    cell_area : Bool[Array, "lat lon"]
        Array of cell areas in square meters.
    variables : Dict[str, Spatiotemporal]
        Dictionary of spatiotemporal variables.
    is_land : Bool[Array, "lat lon"]
        Boolean array indicating land presence.

    Methods
    -------
    indices(time, latitude, longitude)
        Gets nearest indices of the spatio-temporal point `time`, `latitude`, `longitude`.
    interp_temporal(*variables, time)
        Interpolates the specified variables temporally at the given time.
    interp_spatial(*variables, latitude, longitude)
        Interpolates the specified variables spatially at the given latitude and longitude.
    interp_spatiotemporal(*variables, time, latitude, longitude)
        Interpolates the specified variables spatiotemporally at the given time, latitude, and longitude.
    neighborhood(*variables, time, latitude, longitude, t_width, x_width)
        Gets a neighborhood `t_width`*`x_width`*`x_width` around the spatio-temporal point `time`, `latitude`, 
        `longitude`.
    from_arrays(variables, time, latitude, longitude, interpolation_method="linear", is_spherical_mesh=True, use_degrees=False, is_uv_mps=True)
        Constructs a `Dataset` object from arrays of variables and coordinates `time`, `latitude`, `longitude`.
    from_xarray(dataset, variables, coordinates, interpolation_method="linear", is_spherical_mesh=True, use_degrees=False, is_uv_mps=True)
        Constructs a `Dataset` object from an `xarray.Dataset`.
    to_xarray()
        Returns the `Dataset` object as an `xarray.Dataset`.
    """
    is_spherical_mesh: bool
    use_degrees: bool
    coordinates: Coordinates
    dx: Float[Array, "lat lon-1"]
    dy: Float[Array, "lat-1 lon"]
    cell_area: Bool[Array, "lat lon"]
    variables: Dict[str, SpatioTemporalField]
    is_land: Bool[Array, "lat lon"]

    def indices(
        self,
        time: Int[Scalar, ""],
        latitude: Float[Scalar, ""],
        longitude: Float[Scalar, ""]
    ) -> Tuple[Int[Scalar, ""], Int[Scalar, ""], Int[Scalar, ""]]:
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
        Tuple[Int[Scalar, ""], Int[Scalar, ""], Int[Scalar, ""]]
        """
        return self.coordinates.indices(time, latitude, longitude)

    def interp_temporal(
        self,
        *variables: Tuple[str, ...],
        time: Float[Array, "..."]
    ) -> Tuple[Float[Array, "... ... ..."], ...]:
        """
        Interpolates the given variables in time.

        Parameters
        ----------
        variables : Tuple[str, ...]
            Variable names to be interpolated.
        time : Float[Array, "..."]
            The time points at which to interpolate the variables.

        Returns
        -------
        Tuple[Float[Array, "... ... ..."], ...]
            A tuple containing the interpolated values for each variable.
        """
        return tuple(self.variables[var_name].interp_temporal(time) for var_name in variables)

    def interp_spatial(
        self,
        *variables: Tuple[str, ...],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Tuple[Float[Array, "... ... ..."], ...]:
        """
        Interpolates the given variables in space.

        Parameters
        ----------
        *variables : Tuple[str, ...]
            Variable names to be interpolated.
        latitude : Float[Array, "..."]
            The latitude values at which to interpolate the variables.
        longitude : Float[Array, "..."]
            The latitude values at which to interpolate the variables.

        Returns
        -------
        Tuple[Float[Array, "... ... ..."], ...]
            A tuple containing interpolated values for each variable.
        """
        return tuple(
            self.variables[var_name].interp_spatial(latitude, longitude)
            for var_name in variables
        )
    
    def interp_spatiotemporal(
        self,
        *variables: Tuple[str, ...],
        time: Float[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Tuple[Float[Array, "... ... ..."], ...]:
        return tuple(
            self.variables[var_name].interp_spatiotemporal(time, latitude, longitude)
            for var_name in variables
        )

    def neighborhood(
        self,
        *variables: Tuple[str, ...],
        time: Int[Scalar, ""],
        latitude: Float[Scalar, ""],
        longitude: Float[Scalar, ""],
        t_width: int,
        x_width: int
    ) -> Dataset:
        """
        Extracts a neighborhood of data around a specified point in time and space.

        Parameters
        ----------
        *variables : Tuple[str, ...]
            Variable names to extract from the dataset.
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
        Fields
            A Fields object containing the extracted neighborhood data.
        """
        t_i, lat_i, lon_i = self.indices(time, latitude, longitude)

        # TODO: handle edge cases
        from_t_i = t_i - t_width // 2
        from_lat_i = lat_i - x_width // 2
        from_lon_i = lon_i - x_width // 2

        variables = dict(
            (
                var_name,
                jax.lax.dynamic_slice(
                    self.variables[var_name].values,
                    (from_t_i, from_lat_i, from_lon_i),
                    (t_width, x_width, x_width)
                )
            )
            for var_name in variables
        )

        t = jax.lax.dynamic_slice_in_dim(self.coordinates.time.values, from_t_i, t_width)
        lat = jax.lax.dynamic_slice_in_dim(self.coordinates.latitude.values, from_lat_i, x_width)
        lon = jax.lax.dynamic_slice_in_dim(self.coordinates.longitude.values, from_lon_i, x_width)

        return Dataset.from_arrays(
            variables, 
            t, lat, lon, 
            interpolation_method="linear", 
            is_spherical_mesh=self.is_spherical_mesh, 
            use_degrees=self.use_degrees
        )

    @classmethod
    def from_arrays(
        cls,
        variables: Dict[str, Float[Array, "time lat lon"]],
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str = "linear",
        is_spherical_mesh: bool = True,
        use_degrees: bool = False,
        is_uv_mps: bool = True
    ) -> Dataset:
        """
        Create a Dataset object from arrays of variables, time, latitude, and longitude.

        Parameters
        ----------
        variables : Dict[str, Float[Array, "time lat lon"]]
            A dictionary where keys are variable names and values are 3D arrays representing 
            the variable data over time, latitude, and longitude.
        time : Int[Array, "time"]
            A 1D array representing the time dimension.
        latitude : Float[Array, "lat"]
            A 1D array representing the latitude dimension.
        longitude : Float[Array, "lon"]
            A 1D array representing the longitude dimension.
        interpolation_method : str, optional
            The method to use for latter possible interpolation of the variables, defaults to "linear".
        is_spherical_mesh : bool, optional
            Whether the mesh uses spherical coordinate, defaults to True.
        use_degrees : bool, optional
            Whether distance units should be degrees rather than meters, defaults to False.
        is_uv_mps : bool, optional
            Whether the velocity data is in m/s, defaults to True.

        Returns
        -------
        Dataset
            A Dataset object containing the processed variables, land mask, grid spacing in 
            meters, cell area in square meters, and coordinates.
        """
        def compute_cell_dlatlon(dright: Float[Array, "latlon-1"], axis: int) -> Float[Array, "latlon"]:
            if axis == 0:
                dcentered = (dright[1:, :] + dright[:-1, :]) / 2
                dstart =((dright[0, :] - dcentered[0, :] / 2) * 2)[None, :]
                dend = ((dright[-1, :] - dcentered[-1, :] / 2) * 2)[None, :]
            else:
                dcentered = (dright[:, 1:] + dright[:, :-1]) / 2
                dstart = ((dright[:, 0] - dcentered[:, 0] / 2) * 2)[:, None]
                dend = ((dright[:, -1] - dcentered[:, -1] / 2) * 2)[:, None]
            return jnp.concat((dstart, dcentered, dend), axis=axis)
        
        use_degrees = use_degrees & is_spherical_mesh  # if flat mesh, no reason to use degrees

        coordinates = Coordinates.from_arrays(time, latitude, longitude, is_spherical_mesh)

        # compute grid spacings and cells area
        dlat = jnp.diff(latitude)
        dlon = jnp.diff(longitude)

        if is_spherical_mesh and not use_degrees:
            dlatlon = degrees_to_meters(
                jnp.stack([dlat, jnp.zeros_like(dlat)], axis=-1), (latitude[:-1] + latitude[1:]) / 2
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
        for variable in variables.values():
            _is_land = jnp.isnan(variable).sum(axis=0, dtype=bool)
            is_land = jnp.logical_or(is_land, _is_land)

        # apply it
        for var_name in variables:
            variable = variables[var_name]
            variable = jnp.where(is_land, 0, variable)
            variables[var_name] = variable

        # if required, convert uv from m/s to °/s
        if use_degrees and is_uv_mps:
            vu = jnp.stack((variables["v"], variables["u"]), axis=-1)
            original_shape = vu.shape
            vu = vu.reshape(vu.shape[0], -1, 2)

            _, lat_grid = jnp.meshgrid(longitude, latitude)
            lat_grid = lat_grid.ravel()
             
            vu = eqx.filter_vmap(lambda x: meters_to_degrees(x, lat_grid))(vu)
            vu = vu.reshape(original_shape)

            variables["v"] = vu[..., 0]
            variables["u"] = vu[..., 1]

            is_uv_mps = False

        variables = dict(
            (
                variable_name,
                SpatioTemporalField.from_array(
                    values,
                    coordinates.time.values, coordinates.latitude.values, coordinates.longitude.values,
                    interpolation_method=interpolation_method
                )
            ) for variable_name, values in variables.items()
        )

        return cls(
            is_spherical_mesh=is_spherical_mesh,
            use_degrees=use_degrees,
            coordinates=coordinates,
            dx=dlon,
            dy=dlat,
            cell_area=cell_area,
            variables=variables,
            is_land=is_land
        )

    @classmethod
    def from_xarray(
        cls,
        dataset: xr.Dataset,
        variables: Dict[str, str],
        coordinates: Dict[str, str],
        interpolation_method: str = "linear",
        is_spherical_mesh: bool = True,
        is_uv_mps: bool = True,
        use_degrees: bool = False
    ) -> Dataset:
        """
        Create a Fields object from an xarray Dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            The xarray Dataset containing the data.
        variables : Dict[str, str]
            A dictionary mapping the target variable names (keys) to the source variable names in the dataset (values).
            We expect at least "u" and "v" to be provided as target variables names.
        coordinates : Dict[str, str]
            A dictionary mapping the coordinate names ('time', 'latitude', 'longitude') to their corresponding names in
            the dataset.
        interpolation_method : str, optional
            The method to use for latter possible interpolation of the variables (default is "linear").
        is_spherical_mesh : bool, optional
            Whether the mesh uses spherical coordinate, defaults to True.
        is_uv_mps : bool, optional
            Whether the velocity data is in m/s, defaults to True.
        use_dps : bool, optional
            Whether distance unit should be degrees rather than meters, defaults to False.

        Returns
        -------
        Fields
            An instance of the Fields class created from the provided xarray Dataset.
        """
        variables, t, lat, lon = cls.to_arrays(dataset, variables, coordinates, to_jax=True)

        return cls.from_arrays(variables, t, lat, lon, interpolation_method, is_spherical_mesh, use_degrees, is_uv_mps)

    @staticmethod
    def to_arrays(
        dataset: xr.Dataset,
        variables: Dict[str, str],  # to -> from
        coordinates: Dict[str, str],  # to -> from
        to_jax: bool = True
    ) -> Tuple[Dict[str, Float[Array, "..."]], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]:
        """
        Converts an xarray Dataset to arrays for specified variables and coordinates.

        Parameters
        ----------
        dataset : xr.Dataset
            The xarray Dataset to convert.
        variables : Dict[str, str]
            A dictionary mapping the target variable names to the source variable names in the dataset.
        coordinates : Dict[str, str]
            A dictionary mapping the target coordinate names to the source coordinate names in the dataset.
        to_jax : bool, optional
            Whether to convert the arrays to JAX arrays (default is True).

        Returns
        -------
        Tuple[Dict[str, Float[Array, "..."]], Float[Array, "..."], Float[Array, "..."], Float[Array, "..."]]
            A tuple containing:
            - A dictionary of converted variables.
            - The time coordinate array.
            - The latitude coordinate array.
            - The longitude coordinate array.
        """
        if to_jax:
            transform_fn = lambda arr: jnp.asarray(arr, dtype=float)
        else:
            transform_fn = lambda arr: np.asarray(arr, dtype=float)

        variables = dict(
            (to_name, transform_fn(dataset[from_name].data)) for to_name, from_name in variables.items()
        )

        t = transform_fn(dataset[coordinates["time"]].data.astype("datetime64[s]").astype(int))
        lat = transform_fn(dataset[coordinates["latitude"]].data)
        lon = transform_fn(dataset[coordinates["longitude"]].data)

        return variables, t, lat, lon

    def to_xarray(self) -> xr.Dataset:
        """
        Converts the current object to an xarray Dataset.

        This method constructs an xarray Dataset from the object's variables and coordinates.
        The variables are added as data variables with dimensions ["time", "latitude", "longitude"].
        The coordinates are added as coordinate variables.

        Returns
        -------
            xr.Dataset: The constructed xarray Dataset containing the data variables and coordinates.
        """
        dataset = xr.Dataset(
            data_vars=dict(
                (var_name, (["time", "latitude", "longitude"], var.values)) for var_name, var in self.variables.items()
            ),
            coords=dict(
                time=np.asarray(self.coordinates.time.values, dtype="datetime64[s]"),
                latitude=self.coordinates.latitude.values,
                longitude=self.coordinates.longitude.values
            )
        )

        return dataset
