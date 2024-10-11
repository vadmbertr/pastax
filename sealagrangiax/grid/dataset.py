from __future__ import annotations
from typing import Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Scalar
import numpy as np
import xarray as xr

from ..utils.geo import EARTH_RADIUS
from ._coordinates import Coordinates
from ._grid import SpatioTemporal


class Dataset(eqx.Module):
    """
    Class that provides some routines for handling gridded spatiotemporal data in JAX.

    Attributes
    ----------
    variables : Dict[str, Spatiotemporal]
        Dictionary of spatiotemporal variables.
    is_land : Bool[Array, "lat lon"]
        Boolean array indicating land presence.
    dx : Float[Array, "lat lon-1"]
        Array of longitudinal distances in meters.
    dy : Float[Array, "lat-1 lon"]
        Array of latitudinal distances in meters.
    cell_area : Bool[Array, "lat lon"]
        Array of cell areas in square meters.
    coordinates : Coordinates
        Coordinates object containing time, latitude, and longitude.

    Methods
    -------
    indices(time: Int[Array, ""], latitude: Float[Array, ""], longitude: Float[Array, ""]) -> Tuple[Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]]
        Gets nearest indices of the spatio-temporal point `time`, `latitude`, `longitude`.
    interp_temporal(*variables: Tuple[str, ...], time: Float[Array, "..."]) -> Tuple[Float[Array, "... ... ..."], ...]
        Interpolates the specified variables temporally at the given time.
    interp_spatial(*variables: Tuple[str, ...], latitude: Float[Array, "..."], longitude: Float[Array, "..."]) -> Tuple[Float[Array, "... ... ..."], ...]
        Interpolates the specified variables spatially at the given latitude and longitude.
    interp_spatiotemporal(*variables: Tuple[str, ...], time: Float[Array, "..."], latitude: Float[Array, "..."], longitude: Float[Array, "..."]) -> Tuple[Float[Array, "... ... ..."], ...]
        Interpolates the specified variables spatiotemporally at the given time, latitude, and longitude.
    neighborhood(*variables: Tuple[str, ...], time: Int[Array, ""], latitude: Float[Array, ""], longitude: Float[Array, ""], t_width: int = 2, x_width: int = 7) -> Fields
        Gets a neighborhood `t_width`*`x_width`*`x_width` around the spatio-temporal point `time`, `latitude`, 
        `longitude`.
    from_arrays(variables: Dict[str, Float[Array, "time lat lon"]], time: Int[Array, "time"], latitude: Float[Array, "lat"], longitude: Float[Array, "lon"], interpolation_method: str) -> Fields
        Constructs a `Fields` object from arrays of variables and coordinates `time`, `latitude`, `longitude`.
    from_xarray(dataset: xr.Dataset, variables: Dict[str, str], coordinates: Dict[str, str], interpolation_method: str) -> Fields
        Constructs a `Fields` object from an `xarray.Dataset`.
    to_xarray() -> xr.Dataset
        Returns the `Fields` object as an `xarray.Dataset`.
    """
    variables: Dict[str, SpatioTemporal]
    is_land: Bool[Array, "lat lon"]
    dx: Float[Array, "lat lon-1"]  # m
    dy: Float[Array, "lat-1 lon"]  # m
    cell_area: Bool[Array, "lat lon"]  # m^2
    coordinates: Coordinates

    @eqx.filter_jit
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

    @eqx.filter_jit
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

    @eqx.filter_jit
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
    
    @eqx.filter_jit
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

    @eqx.filter_jit
    def neighborhood(
        self,
        *variables: Tuple[str, ...],
        time: Int[Scalar, ""],
        latitude: Float[Scalar, ""],
        longitude: Float[Scalar, ""],
        t_width: int = 2,
        x_width: int = 7
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
        t_width : int, optional
            The width of the neighborhood in the time dimension (default is 2).
        x_width : int, optional
            The width of the neighborhood in the spatial dimensions (latitude and longitude) (default is 7).

        Returns
        -------
        Fields
            A Fields object containing the extracted neighborhood data.
        """
        t_i, lat_i, lon_i = self.indices(time, latitude, longitude)

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

        return Dataset.from_arrays(variables, t, lat, lon, interpolation_method="linear")

    # construct a `Dataset` from `Array`s of `variables` and coordinates `time`, `latitude`, `longitude`
    @staticmethod
    def from_arrays(
        variables: Dict[str, Float[Array, "time lat lon"]],
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str
    ) -> Dataset:
        """
        Create a Fields object from arrays of variables, time, latitude, and longitude.

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
        interpolation_method : str
            The method to use for latter possible interpolation of the variables.

        Returns
        -------
        Fields
            A Fields object containing the processed variables, land mask, grid spacing in 
            meters, cell area in square meters, and coordinates.
        """
        @jax.jit
        def compute_dlatlon(latlon: Float[Array, "latlon"]) -> Float[Array, "latlon-1"]:
            return latlon[1:] - latlon[:-1]

        @jax.jit
        def compute_cell_dlatlon(dright: Float[Array, "latlon-1"]) -> Float[Array, "latlon"]:
            dcentered = (dright[1:] + dright[:-1]) / 2
            dlatlon = jnp.pad(
                dcentered,
                1,
                constant_values=((dright[0] - dcentered[0] / 2) * 2, (dright[-1] - dcentered[-1] / 2) * 2)
            )
            return dlatlon

        @jax.jit
        def dlat_to_meters(_dlat: Float[Array, "lat-1"]) -> Float[Array, "lat-1 lon"]:
            return jnp.einsum("i,j->ij", jnp.radians(_dlat), jnp.full_like(longitude, EARTH_RADIUS))

        @jax.jit
        def dlon_to_meters(_dlon: Float[Array, "lon-1"]) -> Float[Array, "lat lon-1"]:
            return jnp.einsum(
                "i,j->ij",
                jnp.cos(jnp.radians(latitude)) * EARTH_RADIUS, jnp.radians(_dlon)
            )

        longitude = (longitude + 180) % 360 - 180  # force conversion to [-180, 180]

        is_land = jnp.zeros((latitude.size, longitude.size), dtype=bool)

        for variable in variables.values():
            _is_land = jnp.isnan(variable).sum(axis=0, dtype=bool)
            is_land = jnp.logical_or(is_land, _is_land)

        for var_name in variables:
            variable = variables[var_name]
            variable = jnp.where(is_land, 0, variable)
            variables[var_name] = variable

        coordinates = Coordinates.from_arrays(time, latitude, longitude)

        variables = dict(
            (
                variable_name,
                SpatioTemporal.from_array(
                    values,
                    coordinates.time.values, coordinates.latitude.values, coordinates.longitude.values,
                    interpolation_method=interpolation_method
                )
            ) for variable_name, values in variables.items()
        )

        dlat = compute_dlatlon(latitude)  # degrees
        dlon = compute_dlatlon(longitude)  # degrees

        cell_dlat = compute_cell_dlatlon(dlat)  # degrees
        cell_dlon = compute_cell_dlatlon(dlon)  # degrees

        dlat = dlat_to_meters(dlat)  # meters
        dlon = dlon_to_meters(dlon)  # meters

        cell_dlat = dlat_to_meters(cell_dlat)  # meters
        cell_dlon = dlon_to_meters(cell_dlon)  # meters
        cell_area = cell_dlat * cell_dlon  # m^2

        return Dataset(
            variables=variables, 
            is_land=is_land, 
            dx=dlon, 
            dy=dlat, 
            cell_area=cell_area, 
            coordinates=coordinates
        )

    @staticmethod
    def from_xarray(
            dataset: xr.Dataset,
            variables: Dict[str, str],
            coordinates: Dict[str, str],
            interpolation_method: str
    ) -> Dataset:
        """
        Create a Fields object from an xarray Dataset.

        Parameters
        ----------
        dataset : xr.Dataset
            The xarray Dataset containing the data.
        variables : Dict[str, str]
            A dictionary mapping the target variable names (keys) to the source variable names in the dataset (values).
        coordinates : Dict[str, str]
            A dictionary mapping the coordinate names ('time', 'latitude', 'longitude') to their corresponding names in
            the dataset.
        interpolation_method : str
            The method to use for latter possible interpolation of the data.

        Returns
        -------
        Fields
            An instance of the Fields class created from the provided xarray Dataset.
        """
        variables, t, lat, lon = Dataset.to_arrays(dataset, variables, coordinates, to_jax=True)

        return Dataset.from_arrays(variables, t, lat, lon, interpolation_method)

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
            transform_fn = lambda arr: jnp.asarray(arr)
        else:
            transform_fn = lambda arr: np.asarray(arr)

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
