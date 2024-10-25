from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Float, Int

from ..utils.geo import longitude_in_180_180_degrees
from ._grid import Coordinate, LongitudeCoordinate


class Coordinates(eqx.Module):
    """
    Class for handling time, latitude, and longitude coordinates.

    Attributes
    ----------
    time : Coordinate
        The time [`pastax.Coordinate`][].
    latitude : Coordinate
        The latitude [`pastax.Coordinate`][].
    longitude : Coordinate
        The longitude [`pastax.Coordinate`][].

    Methods
    -------        
    indices(time, latitude, longitude)
        Returns the indices of the given time, latitude, and longitude arrays.
        
    from_arrays(time, latitude, longitude, is_spherical_mesh)
        Creates a `pastax.Coordinates` object from arrays of time, latitude, and longitude.
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
    def from_arrays(
        cls,
        time: Int[Array, "time"],
        latitude: Float[Array, "lat"],
        longitude: Float[Array, "lon"],
        is_spherical_mesh: bool
    ) -> Coordinates:
        """
        Create a `pastax.Coordinates` object from arrays of time, latitude, and longitude.

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
            A `pastax.Coordinates` object with time, latitude, and longitude.
        """
        t = Coordinate.from_array(time, extrap=True)
        lat = Coordinate.from_array(latitude, extrap=True)

        if is_spherical_mesh:
            longitude = longitude_in_180_180_degrees(longitude)
            lon = LongitudeCoordinate.from_array(longitude, extrap=True, period=360)
        else:
            lon = Coordinate.from_array(longitude, extrap=True)

        return cls(time=t, latitude=lat, longitude=lon)
