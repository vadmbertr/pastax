from __future__ import annotations

from typing import Any

import equinox as eqx
import interpax as ipx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..utils._geo import longitude_in_180_180_degrees


class Field(eqx.Module):
    """
    Base class for representing a field on a grid.

    Methods
    -------
    values
        Returns the field values.
    __getitem__(item)
        Retrieves the value(s) at the specified index or slice of the grid.
    """

    _values: Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]

    @property
    def values(self) -> Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]:
        """
        Returns the field values.

        Returns
        -------
        Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]
            The gridded values.
        """
        return self._values

    def __getitem__(self, item: Any) -> Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]:
        """
        Retrieve an item from the values array.

        Parameters
        ----------
        item : Any
            The index or slice used to retrieve the item from the values array.

        Returns
        -------
        Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]
            The item retrieved from the values array.
        """
        return self.values.__getitem__(item)


class SpatialField(Field):
    """
    Class representing a spatial (2D) field with interpolation capabilities.

    Methods
    -------
    interp(**coordinates)
        Interpolates the field at the given coordinates.
    from_array(values, latitude, longitude, interpolation_method)
        Creates a [`pastax.gridded.SpatialField`][] instance from the given array of values, latitude, and longitude
        using the specified interpolation method.
    """

    _values: Bool[Array, "lat lon"] | Float[Array, "lat lon"] | Int[Array, "lat lon"]
    _fx: ipx.Interpolator2D

    def interp(self, **coordinates: Float[Array, "Nq"]) -> Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]:
        """
        Interpolates the field at the given coordinates.

        Parameters
        ----------
        **coordinates : Float[Array, "Nq"]
            The 2-dimensional points to interpolate to.

        Returns
        -------
        Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]
            Interpolated values at the given coordinates.
        """
        if "latitude" in coordinates and "longitude" in coordinates:
            return self._interp_spatial(
                latitude=coordinates["latitude"],
                longitude=coordinates["longitude"],
            )
        else:
            return self.values

    def _interp_spatial(
        self, latitude: Float[Array, "Nq"], longitude: Float[Array, "Nq"]
    ) -> Bool[Array, "Nq ..."] | Float[Array, "Nq ..."] | Int[Array, "Nq ..."]:
        """
        Interpolates spatial data based on given latitude and longitude arrays.

        Parameters
        ----------
        latitude : Float[Array, "Nq"]
            Array of latitude values.
        longitude : Float[Array, "Nq"]
            Array of longitude values.

        Returns
        -------
        Bool[Array, "Nq ..."] | Float[Array, "Nq ..."] | Int[Array, "Nq ..."]
            Interpolated spatial data array.
        """
        longitude = longitude_in_180_180_degrees(longitude)  # force to be in -180 to 180 degrees
        longitude += 180  # shift back to 0 to 360 degrees

        return self._fx(latitude, longitude)

    @classmethod
    def from_array(
        cls,
        values: Bool[Array, "lat lon"] | Float[Array, "lat lon"] | Int[Array, "lat lon"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        interpolation_method: str,
    ) -> SpatialField:
        """
        Create a [`pastax.gridded.SpatialField`][] object from given arrays of values, latitude, and longitude.

        Parameters
        ----------
        values : Bool[Array, "lat lon"] | Float[Array, "lat lon"] | Int[Array, "lat lon"]
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
        longitude = longitude_in_180_180_degrees(longitude)  # force to be in -180 to 180 degrees
        longitude += 180  # shift back to 0 to 360 degrees

        _fx = ipx.Interpolator2D(
            latitude,
            longitude,  # periodic domain
            values,
            method=interpolation_method,
            extrap=True,
            period=(None, 360),
        )

        return cls(_values=values, _fx=_fx)


class SpatioTemporalField(Field):
    """
    Class representing a spatiotemporal (3D) field with interpolation capabilities.

    Methods
    -------
    interp(**coordinates)
        Interpolates the field at the given coordinates.
    from_array(values, time, latitude, longitude, interpolation_method)
        Creates a [`pastax.gridded.SpatioTemporalField`][] instance from the given array of values and coordinates.
    """

    _values: Bool[Array, "time lat lon"] | Float[Array, "time lat lon"] | Int[Array, "time lat lon"]
    _ft: ipx.Interpolator1D
    _fx: ipx.Interpolator2D
    _ftx: ipx.Interpolator3D

    def interp(self, **coordinates: Float[Array, "Nq"]) -> Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]:
        """
        Interpolates the field at the given coordinates.

        Parameters
        ----------
        **coordinates : Float[Array, "Nq"]
            The N-dimensional points to interpolate to.

        Returns
        -------
        Bool[Array, "..."] | Float[Array, "..."] | Int[Array, "..."]
            Interpolated values at the given coordinates.
        """
        if "time" in coordinates and "latitude" in coordinates and "longitude" in coordinates:
            return self._interp_spatiotemporal(
                time=coordinates["time"],
                latitude=coordinates["latitude"],
                longitude=coordinates["longitude"],
            )
        elif "latitude" in coordinates and "longitude" in coordinates:
            return self._interp_spatial(
                latitude=coordinates["latitude"],
                longitude=coordinates["longitude"],
            )
        elif "time" in coordinates:
            return self._interp_temporal(time=coordinates["time"])
        else:
            return self.values

    def _interp_temporal(
        self, time: Float[Array, "Nq"]
    ) -> Bool[Array, "Nq lat lon"] | Float[Array, "Nq lat lon"] | Int[Array, "Nq lat lon"]:
        """
        Interpolates the spatiotemporal field at the given time points.

        Parameters
        ----------
        time : Float[Array, Nq"]
            An array of time points at which to interpolate the spatiotemporal field.

        Returns
        -------
        Bool[Array, "Nq lat lon"] | Float[Array, "Nq lat lon"] | Int[Array, "Nq lat lon"]
            Interpolated values at the given time points.
        """
        return self._ft(time)

    def _interp_spatial(
        self, latitude: Float[Array, "Nq"], longitude: Float[Array, "Nq"]
    ) -> Bool[Array, "Nq time"] | Float[Array, "Nq time"] | Int[Array, "Nq time"]:
        """
        Interpolates the spatiotemporal field at the given latitude/longitude points.

        Parameters
        ----------
        latitude : Float[Array, "Nq"]
            Array of latitude values.
        longitude : Float[Array, "Nq"]
            Array of longitude values.

        Returns
        -------
        Bool[Array, "Nq time"] | Float[Array, "Nq time"] | Int[Array, "Nq time"]
            Interpolated values at the given latitude/longitude points.
        """
        longitude = longitude_in_180_180_degrees(longitude)  # force to be in -180 to 180 degrees
        longitude += 180  # shift back to 0 to 360 degrees

        return self._fx(latitude, longitude)

    def _interp_spatiotemporal(
        self,
        time: Float[Array, "Nq"],
        latitude: Float[Array, "Nq"],
        longitude: Float[Array, "Nq"],
    ) -> Bool[Array, "Nq"] | Float[Array, "Nq"] | Int[Array, "Nq"]:
        """
        Interpolates the spatiotemporal field at the given time/latitude/longitude points.

        Parameters
        ----------
        time : Float[Array, "Nq"]
            Array of time values.
        latitude : Float[Array, "Nq"]
            Array of latitude values.
        longitude : Float[Array, "Nq"]
            Array of longitude values.

        Returns
        -------
        Bool[Array, "Nq"] | Float[Array, "Nq"] | Int[Array, "Nq"]
            Interpolated values at the given time/latitude/longitude points.
        """
        longitude = longitude_in_180_180_degrees(longitude)  # force to be in -180 to 180 degrees
        longitude += 180  # shift back to 0 to 360 degrees

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
        longitude = longitude_in_180_180_degrees(longitude)  # force to be in -180 to 180 degrees
        longitude += 180  # shift back to 0 to 360 degrees

        _ft = ipx.Interpolator1D(time, values, method=interpolation_method, extrap=True)
        _fx = ipx.Interpolator2D(
            latitude,
            longitude,
            jnp.moveaxis(values, 0, -1),  # time dim is moved to the last axis as it is not interpolated
            method=interpolation_method,
            extrap=True,
            period=(None, 360),
        )
        _ftx = ipx.Interpolator3D(
            time,
            latitude,
            longitude,
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
