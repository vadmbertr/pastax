from __future__ import annotations
from typing import Tuple

import equinox as eqx
from jaxtyping import Array, Float, Int

from ._grid import Coordinate


class Coordinates(eqx.Module):
    """
    Class for handling time, latitude, and longitude coordinates.

    Attributes
    ----------
    time : Coordinate
        The time coordinate.
    latitude : Coordinate
        The latitude coordinate.
    longitude : Coordinate
        The longitude coordinate.

    Methods
    -------        
    indices(time: Int[Array, "..."], latitude: Float[Array, "..."], longitude: Float[Array, "..."]) -> Tuple[Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]]
        Returns the indices of the given time, latitude, and longitude arrays.
        
    from_arrays(time: Int[Array, "time"], latitude: Float[Array, "lat"], longitude: Float[Array, "lon"]) -> Coordinates
        Creates a Coordinates object from arrays of time, latitude, and longitude.
    """
    time: Coordinate
    latitude: Coordinate
    longitude: Coordinate

    @eqx.filter_jit
    def indices(
        self,
        time: Int[Array, "..."],
        latitude: Float[Array, "..."],
        longitude: Float[Array, "..."]
    ) -> Tuple[Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]]:
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
        Tuple[Int[Array, "..."], Int[Array, "..."], Int[Array, "..."]]
            A tuple containing the indices for time, latitude, and longitude respectively.
        """
        return self.time.index(time), self.latitude.index(latitude), self.longitude.index(longitude)

    @staticmethod
    @eqx.filter_jit
    def from_arrays(
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"]
    ) -> Coordinates:
        """
        Create a Coordinates object from arrays of time, latitude, and longitude.

        Parameters
        ----------
        time : Int[Array, "time"]
            Array of time values.
        latitude : Float[Array, "lat"]
            Array of latitude values.
        longitude : Float[Array, "lon"]
            Array of longitude values.

        Returns
        -------
        Coordinates
            A Coordinates object with time, latitude, and longitude.
        """
        t = Coordinate.from_array(time, extrap=True)
        lat = Coordinate.from_array(latitude, extrap=True)
        lon = Coordinate.from_array(longitude, is_circular=True, extrap=True, period=360)

        return Coordinates(time=t, latitude=lat, longitude=lon)