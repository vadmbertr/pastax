from __future__ import annotations
from typing import Any

import equinox as eqx
import interpax as ipx
import jax.numpy as jnp
from jaxtyping import Float, Int, Array


class Grid(eqx.Module):
    """
    Base class for representing a grid of values, which can be either floating-point or integer arrays.

    Attributes
    ----------
    _values : Float[Array, "..."] | Int[Array, "..."]
        The grid values stored as either floating-point or integer arrays.

    Methods
    -------
    __getitem__(item:)
        Retrieves the value(s) at the specified index or slice from the grid.
    """
    _values: Float[Array, "..."] | Int[Array, "..."]

    @property
    def values(self) -> Float[Array, "dim"]:
        """
        Returns the grid values.
# 
        Returns
        -------
        Float[Array, "dim"]
            The grid values.
        """
        return self._values

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
    _values : Float[Array, "dim"]
        1D array of coordinate values.
    indices : ipx.Interpolator1D
        Interpolator for nearest index interpolation.

    Methods
    -------
    index(query)
        Returns the nearest index for the given query coordinates.
        
    from_array(values, **interpolator_kwargs)
        Creates a `pastax.grid.Coordinate` instance from an array of values.
    """
    _values: Float[Array, "dim"]  # only handles 1D coordinates, i.e. rectilinear grids
    indices: ipx.Interpolator1D

    @property
    def values(self) -> Float[Array, "dim"]:
        """
        Returns the coordinate values.
# 
        Returns
        -------
        Float[Array, "dim"]
            The coordinate values.
        """
        return self._values

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
        return self.indices(query).astype(int)

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "dim"],
        **interpolator_kwargs: Any
    ) -> Coordinate:
        """
        Create a `pastax.grid.Coordinate` object from an array of values.

        This method initializes a `pastax.grid.Coordinate` object using the provided array of values.
        It uses a 1D interpolator to generate indices from values, with the interpolation method set to `"nearest"`.

        Parameters
        ----------
        values : Float[Array, "dim"]
            An array of coordinate values.
        **interpolator_kwargs : Any
            Additional keyword arguments for the interpolator.

        Returns
        -------
        Coordinate
            A  `pastax.grid.Coordinate` object containing the provided values and corresponding indices interpolator.
        """
        interpolator_kwargs["method"] = "nearest"
        indices = ipx.Interpolator1D(values, jnp.arange(values.size), **interpolator_kwargs)

        return cls(_values=values, indices=indices)

class LongitudeCoordinate(Coordinate):
    """
    Class for handling 1D longitude coordinates (i.e. of rectilinear grids). 
    This class handles the circular nature of longitudes coordinates.

    Attributes
    ----------
    _values : Float[Array, "dim"]
        1D array of longitude coordinate values.
    indices : ipx.Interpolator1D
        Interpolator for nearest index interpolation.

    Methods
    -------
    index(query)
        Returns the nearest index for the given query coordinates.
        
    from_array(values, **interpolator_kwargs)
        Creates a `pastax.LongitudeCoordinate` instance from an array of values.
    """
    _values: Float[Array, "dim"]  # only handles 1D coordinates, i.e. rectilinear grids
    indices: ipx.Interpolator1D

    @property
    def values(self) -> Float[Array, "dim"]:
        """
        Returns the coordinate values.
# 
        Returns
        -------
        Float[Array, "dim"]
            The coordinate values.
        """
        return self._values - 180

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
        return self.indices(query + 180).astype(int)

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "dim"],
        **interpolator_kwargs: Any
    ) -> LongitudeCoordinate:
        """
        Create a LongitudeCoordinate object from an array of values.

        This method initializes a LongitudeCoordinate object using the provided array of values.
        It uses a 1D interpolator to generate indices from values, with the interpolation method set to "nearest".

        Parameters
        ----------
        values : Float[Array, "dim"]
            An array of coordinate values.
        **interpolator_kwargs : Any
            Additional keyword arguments for the interpolator.

        Returns
        -------
        LongitudeCoordinate
            A `pastax.LongitudeCoordinate` object containing the provided values and corresponding indices interpolator.
        """
        values += 180

        interpolator_kwargs["method"] = "nearest"
        indices = ipx.Interpolator1D(values, jnp.arange(values.size), **interpolator_kwargs)

        return cls(_values=values, indices=indices)


class SpatialField(Grid):
    """
    Class representing a spatial field with interpolation capabilities.

    Attributes
    ----------
    _values : Float[Array, "lat lon"]
        The gridded data values.
    spatial_field : ipx.Interpolator2D
        The interpolator for spatial data.

    Methods
    -------
    interp_spatial(latitude, longitude)
        Interpolates the spatial data at the given latitude and longitude.
        
    from_array(values, latitude, longitude, interpolation_method)
        Creates a `pastax.SpatialField` instance from the given array of values, latitude, and longitude 
        using the specified interpolation method.
    """
    _values: Float[Array, "lat lon"]
    spatial_field: ipx.Interpolator2D

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

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "lat lon"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str
    ) -> SpatialField:
        """
        Create a `pastax.SpatialField` object from given arrays of values, latitude, and longitude.

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
        SpatialField
            A `pastax.SpatialField` object containing the values and the interpolated spatial field.
        """
        spatial_field = ipx.Interpolator2D(
            latitude, longitude + 180,  # circular domain
            values,
            method=interpolation_method, extrap=True, period=(None, 360)
        )

        return cls(_values=values, spatial_field=spatial_field)


class SpatioTemporalField(Grid):
    """
    Class representing a spatiotemporal field with interpolation capabilities.

    Attributes
    ----------
    values : Float[Array, "time lat lon"]
        The values of the spatiotemporal field.
    temporal_field : ipx.Interpolator1D
        Interpolator for the temporal dimension.
    spatial_field : ipx.Interpolator2D
        Interpolator for the spatial dimensions (latitude and longitude).
    spatiotemporal_field : ipx.Interpolator3D
        Interpolator for the spatiotemporal dimensions.

    Methods
    -------
    interp_temporal(tq)
        Interpolates the spatiotemporal field at the given times.
    
    interp_spatial(latitude, longitude)
        Interpolates the spatiotemporal field at the given latitudes and longitudes.
    
    interp_spatiotemporal(time, latitude, longitude)
        Interpolates the spatiotemporal field at the given times, latitudes, and longitudes.
    
    from_array(values, time, latitude, longitude, interpolation_method)
        Creates a `pastax.SpatioTemporalField` instance from the given array of values and coordinates.
    """
    _values: Float[Array, "time lat lon"]
    temporal_field: ipx.Interpolator1D
    spatial_field: ipx.Interpolator2D
    spatiotemporal_field: ipx.Interpolator3D

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

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "time lat lon"],
        time: Float[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str
    ) -> SpatioTemporalField:
        """
        Create a `pastax.SpatioTemporalField` object from given arrays of values, time, latitude, and longitude.

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
        SpatioTemporalField
            A `pastax.SpatioTemporalField` object containing the original values and temporal, spatial, 
            and spatiotemporal interpolators.
        """
        temporal_field = ipx.Interpolator1D(
            time,
            values,
            method=interpolation_method, extrap=True
        )
        spatial_field = ipx.Interpolator2D(
            latitude, longitude + 180,  # circular domain
            jnp.moveaxis(values, 0, -1),
            method=interpolation_method, extrap=True, period=(None, 360)
        )
        spatiotemporal_field = ipx.Interpolator3D(
            time, latitude, longitude + 180,  # circular domain
            values,
            method=interpolation_method, extrap=True, period=(None, None, 360)
        )

        return cls(
            _values=values,                            
            temporal_field=temporal_field,            
            spatial_field=spatial_field,              
            spatiotemporal_field=spatiotemporal_field,
        )
