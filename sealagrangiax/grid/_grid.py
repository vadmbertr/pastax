from __future__ import annotations
from typing import Any

import equinox as eqx
import interpax as inx
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


class Grid(eqx.Module):
    """
    Class that represents a grid of values, which can be either floating-point or integer arrays.

    Attributes
    ----------
    values : Float[Array, "..."] | Int[Array, "..."]
        The grid values stored as either floating-point or integer arrays.

    Methods
    -------
    __getitem__(item: Any) -> Float[Array, "..."] | Int[Array, "..."]
        Retrieves the value(s) at the specified index or slice from the grid.
    """
    values: Float[Array, "..."] | Int[Array, "..."]

    def __getitem__(self, item: Any) -> Float[Array, "..."] | Int[Array, "..."]:
        """
        Retrieve an item from the values array.

        Parameters
        ----------
        item : Any
            The index or slice used to retrieve the item from the values array.

        Returns
        -------
        Float[Array, "..."] | Int[Array, "..."]
            The item retrieved from the values array, which can be either a float or an integer array.
        """
        return self.values.__getitem__(item)


class Coordinate(Grid):
    """
    Class for handling 1D coordinates (i.e. of rectilinear grids).

    Attributes
    ----------
    values : Float[Array, "dim"]
        1D array of coordinate values.
    indices : inx.Interpolator1D
        Interpolator for nearest index interpolation.
    is_circular : boolean, default=False
        Indicates if the coordinate system is circular.

    Methods
    -------
    index(query: Float[Array, "..."]) -> Int[Array, "..."]
        Returns the nearest index for the given query coordinates.
        
    from_array(values: Float[Array, "coord"], is_circular: bool = False, **interpolator_kwargs: Any) -> Coordinate
        Creates a Coordinate instance from an array of values.
    """
    _values: Float[Array, "dim"]  # only handles 1D coordinates, i.e. rectilinear grids
    indices: inx.Interpolator1D
    is_circular: bool = False

    @property
    def values(self) -> Float[Array, "dim"]:
        """
        Returns the coordinate values.

        Returns
        -------
        Float[Array, "dim"]
            The coordinate values.
        """
        values = self._values
        if self.is_circular:
            values -= 180

        return values

    @eqx.filter_jit
    def index(self, query: Float[Array, "..."]) -> Int[Array, "..."]:
        """
        Returns the nearest index interpolation for the given query.

        Parameters
        ----------
        query : Float[Array, "..."]
            The query array for which the nearest indices are to be found.

        Returns
        -------
        Int[Array, "..."]
            An array of integers representing the nearest indices.
        """
        if self.is_circular:
            query += 180
        
        return self.indices(query).astype(int)

    @staticmethod
    @eqx.filter_jit
    def from_array(
        values: Float[Array, "dim"],
        is_circular: bool = False,
        **interpolator_kwargs: Any
    ) -> Coordinate:
        """
        Create a Coordinate object from an array of values.

        This method initializes a Coordinate object using the provided array of values.
        It uses a 1D interpolator to generate indices from values, with the interpolation method set to "nearest".

        Parameters
        ----------
        values : Float[Array, "dim"]
            An array of coordinate values.
        is_circular : bool, optional
            Indicates if the coordinate system is circular, by default False.
        **interpolator_kwargs : Any
            Additional keyword arguments for the interpolator.

        Returns
        -------
        Coordinate
            A Coordinate object containing the provided values and corresponding indices interpolator.
        """
        if is_circular:
            values += 180

        interpolator_kwargs["method"] = "nearest"
        indices = inx.Interpolator1D(values, jnp.arange(values.size), **interpolator_kwargs)

        return Coordinate(_values=values, indices=indices, is_circular=is_circular)


class Spatial(Grid):
    """
    Class representing a spatial field with interpolation capabilities.

    Attributes
    ----------
    values : Float[Array, "lat lon"]
        The gridded data values.
    spatial_field : inx.Interpolator2D
        The interpolator for spatial data.

    Methods
    -------
    interp_spatial(latitude: Float[Array, "..."], longitude: Float[Array, "..."]) -> Float[Array, "... ..."]:
        Interpolates the spatial data at the given latitude and longitude.
        
    from_array(values: Float[Array, "lat lon"], latitude: Float[Array, "lat"], longitude: Float[Array, "lon"], interpolation_method: str) -> Spatial:
        Creates a Spatial instance from the given array of values, latitude, and longitude using the specified
        interpolation method.
    """
    values: Float[Array, "lat lon"]
    spatial_field: inx.Interpolator2D

    @eqx.filter_jit
    def interp_spatial(
        self,
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Float[Array, "... ..."]:
        """
        Interpolates spatial data based on given latitude and longitude arrays.

        Parameters
        ----------
        latitude : Float[Array, "..."]
            Array of latitude values.
        longitude : Float[Array, "..."]
            Array of longitude values.

        Returns
        -------
        Float[Array, "... ..."]
            Interpolated spatial data array.
        """
        longitude += 180  # circular domain

        return self.spatial_field(latitude, longitude)

    @staticmethod
    @eqx.filter_jit
    def from_array(
        values: Float[Array, "lat lon"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str
    ) -> Spatial:
        """
        Create a Spatial object from given arrays of values, latitude, and longitude.

        Parameters
        ----------
        values : Float[Array, "lat lon"]
            A 2D array of values representing the spatial data.
        latitude : Float[Array, "lat"]
            A 1D array of latitude values.
        longitude : Float[Array, "lon"]
            A 1D array of longitude values.
        interpolation_method : str
            The method to use for interpolation.

        Returns
        -------
        Spatial
            A Spatial object containing the values and the interpolated spatial field.
        """
        spatial_field = inx.Interpolator2D(
            latitude, longitude + 180,  # circular domain
            values,
            method=interpolation_method, extrap=True, period=(None, 360)
        )

        return Spatial(values=values, spatial_field=spatial_field)


class SpatioTemporal(Grid):
    """
    Class representing a spatiotemporal field with interpolation capabilities.

    Attributes
    ----------
    values : Float[Array, "time lat lon"]
        The values of the spatiotemporal field.
    temporal_field : inx.Interpolator1D
        Interpolator for the temporal dimension.
    spatial_field : inx.Interpolator2D
        Interpolator for the spatial dimensions (latitude and longitude).
    spatiotemporal_field : inx.Interpolator3D
        Interpolator for the spatiotemporal dimensions.

    Methods
    -------
    interp_temporal(tq: Float[Array, "..."]) -> Float[Array, "... ... ..."]
        Interpolates the spatiotemporal field at the given times.
    
    interp_spatial(latitude: Float[Array, "..."], longitude: Float[Array, "..."]) -> Float[Array, "... ... ..."]
        Interpolates the spatiotemporal field at the given latitudes and longitudes.
    
    interp_spatiotemporal(time: Float[Array, "..."], latitude: Float[Array, "..."], longitude: Float[Array, "..."]) -> Float[Array, "... ... ..."]
        Interpolates the spatiotemporal field at the given times, latitudes, and longitudes.
    
    from_array(values: Float[Array, "time lat lon"], time: Float[Array, "time"], latitude: Float[Array, "lat"], longitude: Float[Array, "lon"], interpolation_method: str) -> SpatioTemporal
        Creates a SpatioTemporal instance from the given array of values and coordinates.
    """
    values: Float[Array, "time lat lon"]
    temporal_field: inx.Interpolator1D
    spatial_field: inx.Interpolator2D
    spatiotemporal_field: inx.Interpolator3D

    @eqx.filter_jit
    def interp_temporal(self, tq: Float[Array, "..."]) -> Float[Array, "... ... ..."]:
        """
        Interpolates the spatiotemporal field at the given time points.

        Parameters
        ----------
        tq : Float[Array, "..."]
            An array of time points at which to interpolate the spatiotemporal field.

        Returns
        -------
        Float[Array, "... ... ..."]
            Interpolated values at the given time points.
        """
        return self.temporal_field(tq)

    @eqx.filter_jit
    def interp_spatial(
        self,
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Float[Array, "... ... ..."]:
        """
        Interpolates the spatiotemporal field at the given latitude/longitude points.

        Parameters
        ----------
        latitude : Float[Array, "..."]
            Array of latitude values.
        longitude : Float[Array, "..."]
            Array of longitude values.

        Returns
        -------
        Float[Array, "... ... ..."]
            Interpolated values at the given latitude/longitude points.
        """
        longitude += 180  # circular domain

        return jnp.moveaxis(self.spatial_field(latitude, longitude), -1, 0)

    @eqx.filter_jit
    def interp_spatiotemporal(
        self,
        time: Float[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Float[Array, "... ... ..."]:
        """
        Interpolates the spatiotemporal field at the given time/latitude/longitude points.

        Parameters
        ----------
        time : Float[Array, "..."]
            Array of time values.
        latitude : Float[Array, "..."]
            Array of latitude values.
        longitude : Float[Array, "..."]
            Array of longitude values.

        Returns
        -------
        Float[Array, "... ... ..."]
            Interpolated values at the given time/latitude/longitude points.
        """
        longitude += 180  # circular domain

        return self.spatiotemporal_field(time, latitude, longitude)

    @staticmethod
    @eqx.filter_jit
    def from_array(
        values: Float[Array, "time lat lon"],
        time: Float[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str
    ) -> SpatioTemporal:
        """
        Create a SpatioTemporal object from given arrays of values, time, latitude, and longitude.

        Parameters
        ----------
        values : Float[Array, "time lat lon"]
            The array of values representing the data over time, latitude, and longitude.
        time : Float[Array, "time"]
            The array of time points.
        latitude : Float[Array, "lat"]
            The array of latitude points.
        longitude : Float[Array, "lon"]
            The array of longitude points.
        interpolation_method : str
            The method to be used for interpolation (e.g., 'linear', 'nearest', ...).

        Returns
        -------
        SpatioTemporal
            An object containing the original values and temporal, spatial, and spatiotemporal interpolators.
        """
        temporal_field = inx.Interpolator1D(
            time,
            values,
            method=interpolation_method, extrap=True
        )
        spatial_field = inx.Interpolator2D(
            latitude, longitude + 180,  # circular domain
            jnp.moveaxis(values, 0, -1),
            method=interpolation_method, extrap=True, period=(None, 360)
        )
        spatiotemporal_field = inx.Interpolator3D(
            time, latitude, longitude + 180,  # circular domain
            values,
            method=interpolation_method, extrap=True, period=(None, None, 360)
        )

        return SpatioTemporal(
            values=values,                            
            temporal_field=temporal_field,            
            spatial_field=spatial_field,              
            spatiotemporal_field=spatiotemporal_field,
        )
