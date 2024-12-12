from __future__ import annotations

from typing import Any

import equinox as eqx
import interpax as ipx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class Field(eqx.Module):
    """
    Base class for representing a field on a grid.

    Attributes
    ----------
    _values : Float[Array, "..."] | Int[Array, "..."]
        The gridded values of the field.

    Methods
    -------
    __getitem__(item:)
        Retrieves the value(s) at the specified index or slice of the grid.
    """

    _values: Float[Array, "..."] | Int[Array, "..."]

    @property
    def values(self) -> Float[Array, "dim"]:
        """
        Returns the gridded values.

        Returns
        -------
        Float[Array, "dim"]
            The gridded values.
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
            The item retrieved from the values array.
        """
        return self.values.__getitem__(item)


class SpatialField(Field):
    """
    Class representing a spatial (2D) field with interpolation capabilities.

    Attributes
    ----------
    _values : Float[Array, "lat lon"]
        The gridded data values.
    _fx : ipx.Interpolator2D
        The interpolator for spatial data.

    Methods
    -------
    interp_spatial(latitude, longitude)
        Interpolates the spatial data at the given latitude and longitude.

    from_array(values, latitude, longitude, interpolation_method)
        Creates a [`pastax.gridded.SpatialField`][] instance from the given array of values, latitude, and longitude
        using the specified interpolation method.
    """

    _values: Float[Array, "lat lon"]
    _fx: ipx.Interpolator2D

    def interp_spatial(self, latitude: Float[Array, "..."], longitude: Float[Array, "..."]) -> Float[Array, "... ..."]:
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

        return self._fx(latitude, longitude)

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "lat lon"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str,
    ) -> SpatialField:
        """
        Create a [`pastax.gridded.SpatialField`][] object from given arrays of values, latitude, and longitude.

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
            A [`pastax.gridded.SpatialField`][] object containing the values and the interpolated spatial field.
        """
        _fx = ipx.Interpolator2D(
            latitude,
            longitude + 180,  # circular domain
            values,
            method=interpolation_method,
            extrap=True,
            period=(None, 360),
        )

        return cls(_values=values, _fx=_fx)


class SpatioTemporalField(Field):
    """
    Class representing a spatiotemporal (3D) field with interpolation capabilities.

    Attributes
    ----------
    values : Float[Array, "time lat lon"]
        The values of the spatiotemporal field.
    _ft : ipx.Interpolator1D
        Interpolator for the temporal dimension.
    _fx : ipx.Interpolator2D
        Interpolator for the spatial dimensions (latitude and longitude).
    _ftx : ipx.Interpolator3D
        Interpolator for the spatiotemporal dimensions.

    Methods
    -------
    interp_temporal(time)
        Interpolates the spatiotemporal field at the given times.
    interp_spatial(latitude, longitude)
        Interpolates the spatiotemporal field at the given latitudes and longitudes.
    interp_spatiotemporal(time, latitude, longitude)
        Interpolates the spatiotemporal field at the given times, latitudes, and longitudes.
    from_array(values, time, latitude, longitude, interpolation_method)
        Creates a [`pastax.gridded.SpatioTemporalField`][] instance from the given array of values and coordinates.
    """

    _values: Float[Array, "time lat lon"]
    _ft: ipx.Interpolator1D
    _fx: ipx.Interpolator2D
    _ftx: ipx.Interpolator3D

    def interp_temporal(self, time: Float[Array, "..."]) -> Float[Array, "... ... ..."]:
        """
        Interpolates the spatiotemporal field at the given time points.

        Parameters
        ----------
        time : Float[Array, "..."]
            An array of time points at which to interpolate the spatiotemporal field.

        Returns
        -------
        Float[Array, "... ... ..."]
            Interpolated values at the given time points.
        """
        return self._ft(time)

    def interp_spatial(
        self, latitude: Float[Array, "..."], longitude: Float[Array, "..."]
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

        return jnp.moveaxis(self._fx(latitude, longitude), -1, 0)  # time dim is moved back to the first axis

    def interp_spatiotemporal(
        self,
        time: Float[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."],
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

        return self._ftx(time, latitude, longitude)

    @classmethod
    def from_array(
        cls,
        values: Float[Array, "time lat lon"],
        time: Float[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str,
    ) -> SpatioTemporalField:
        """
        Create a [`pastax.gridded.SpatioTemporalField`][] object from given arrays of values, time, latitude, and
        longitude.

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
            A [`pastax.gridded.SpatioTemporalField`][] object containing the original values and temporal, spatial,
            and spatiotemporal interpolators.
        """
        _ft = ipx.Interpolator1D(time, values, method=interpolation_method, extrap=True)
        _fx = ipx.Interpolator2D(
            latitude,
            longitude + 180,  # circular domain
            jnp.moveaxis(values, 0, -1),  # time dim is moved to the last axis as it is not interpolated
            method=interpolation_method,
            extrap=True,
            period=(None, 360),
        )
        _ftx = ipx.Interpolator3D(
            time,
            latitude,
            longitude + 180,  # circular domain
            values,
            method=interpolation_method,
            extrap=True,
            period=(None, None, 360),
        )

        return cls(
            _values=values,
            _ft=_ft,
            _fx=_fx,
            _ftx=_ftx,
        )
