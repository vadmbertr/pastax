from __future__ import annotations
from typing import Any

import equinox as eqx
import interpax as ipx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from ..utils._geo import longitude_in_180_180_degrees


class Coordinate(eqx.Module):
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
        Creates a [`pastax.grid.Coordinate`][] instance from an array of values.
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
        Create a [`pastax.grid.Coordinate`][] object from an array of values.

        This method initializes a [`pastax.grid.Coordinate`][] object using the provided array of values.
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
            A  [`pastax.grid.Coordinate`][] object containing the provided values and corresponding indices interpolator.
        """
        interpolator_kwargs["method"] = "nearest"
        indices = ipx.Interpolator1D(values, jnp.arange(values.size), **interpolator_kwargs)

        return cls(_values=values, indices=indices)

    def __getitem__(self, item: Any) -> Float[Array, "..."]:
        """
        Retrieve an item from the coordinate array.

        Parameters
        ----------
        item : Any
            The index or slice used to retrieve the item from the values array.

        Returns
        -------
        Float[Array, "..."] | Int[Array, "..."]
            The item retrieved from the coordinate array.
        """
        return self.values.__getitem__(item)


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
        Creates a [`pastax.grid.LongitudeCoordinate`][] instance from an array of values.
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
            A [`pastax.grid.LongitudeCoordinate`][] object containing the provided values and corresponding indices interpolator.
        """
        values += 180

        interpolator_kwargs["method"] = "nearest"
        indices = ipx.Interpolator1D(values, jnp.arange(values.size), **interpolator_kwargs)

        return cls(_values=values, indices=indices)


class Coordinates(eqx.Module):
    """
    Class for handling time, latitude, and longitude coordinates.

    Attributes
    ----------
    time : Coordinate
        The time [`pastax.grid.Coordinate`][].
    latitude : Coordinate
        The latitude [`pastax.grid.Coordinate`][].
    longitude : Coordinate
        The longitude [`pastax.grid.Coordinate`][].

    Methods
    -------        
    indices(time, latitude, longitude)
        Returns the indices of the given time, latitude, and longitude arrays.
        
    from_array(time, latitude, longitude, is_spherical_mesh)
        Creates a [`pastax.grid.Coordinates`][] object from arrays of time, latitude, and longitude.
    """
    time: Coordinate
    latitude: Coordinate
    longitude: LongitudeCoordinate

    def indices(
        self,
        time: Int[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> tuple[Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]]:
        """
        Returns the nearest indices for the given time, latitude, and longitude.

        Parameters
        ----------
        time : Int[Array, "..."]
            The time array for which to get the nearest indices.
        latitude : Float[Array, "..."]
            The latitude array for which to get the nearest indices.
        longitude : Float[Array, "..."]
            The longitude array for which to get the nearest indices.

        Returns
        -------
        tuple[Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]]
            A tuple containing the indices for time, latitude, and longitude respectively.
        """
        return self.time.index(time), self.latitude.index(latitude), self.longitude.index(longitude)

    @classmethod
    def from_array(
        cls,
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        is_spherical_mesh: bool
    ) -> Coordinates:
        """
        Create a [`pastax.grid.Coordinates`][] object from arrays of time, latitude, and longitude.

        Parameters
        ----------
        time : Int[Array, "time"]
            Array of time values.
        latitude : Float[Array, "lat"]
            Array of latitude values.
        longitude : Float[Array, "lon"]
            Array of longitude values.
        is_spherical_mesh : bool
            Whether the mesh is spherical (or flat).

        Returns
        -------
        Coordinates
            A [`pastax.grid.Coordinates`][] object with time, latitude, and longitude.
        """
        t = Coordinate.from_array(time, extrap=True)
        lat = Coordinate.from_array(latitude, extrap=True)

        if is_spherical_mesh:
            longitude = longitude_in_180_180_degrees(longitude)
            lon = LongitudeCoordinate.from_array(longitude, extrap=True, period=360)
        else:
            lon = Coordinate.from_array(longitude, extrap=True)

        return cls(time=t, latitude=lat, longitude=lon)
