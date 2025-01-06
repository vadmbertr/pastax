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
    indices : ipx.Interpolator1D
        Interpolator for nearest index interpolation.

    Methods
    -------
    values
        Returns the coordinate values.
    index(query)
        Returns the nearest index for the given query coordinates.
    from_array(values, **interpolator_kwargs)
        Creates a [`pastax.gridded.Coordinate`][] instance from an array of values.
    __getitem__(item:)
        Retrieve an item from the coordinate array.
    """

    _values: Float[Array, "dim"]  # only handles 1D coordinates, i.e. rectilinear grids
    indices: ipx.Interpolator1D

    @property
    def values(self) -> Float[Array, "dim"]:
        """
        Returns the coordinate values.

        Returns
        -------
        Float[Array, "dim"]
            The coordinate values.
        """
        return self._values

    def index(self, query: Float[Array, "Nq"]) -> Int[Array, "Nq"]:
        """
        Returns the nearest index interpolation for the given query.

        Parameters
        ----------
        query : Float[Array, "Nq"]
            The query array for which the nearest indices are to be found.

        Returns
        -------
        Int[Array, "Nq"]
            An array of integers representing the nearest indices.
        """
        return self.indices(query).astype(int)

    @classmethod
    def from_array(cls, values: Float[Array, "dim"], **interpolator_kwargs: Any) -> Coordinate:
        """
        Create a [`pastax.gridded.Coordinate`][] object from an array of values.

        This method initializes a [`pastax.gridded.Coordinate`][] object using the provided array of values.
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
            A  [`pastax.gridded.Coordinate`][] object containing the provided values and corresponding indices
            interpolator.
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
    indices : ipx.Interpolator1D
        Interpolator for nearest index interpolation.
    is_spherical : bool
        Whether the mesh uses spherical coordinate.

    Methods
    -------
    values
        Returns the coordinate values.
    index(query)
        Returns the nearest index for the given query coordinates.
    from_array(values, **interpolator_kwargs)
        Creates a [`pastax.gridded.LongitudeCoordinate`][] instance from an array of values.
    """

    _values: Float[Array, "dim"]  # only handles 1D coordinates, i.e. rectilinear grids
    indices: ipx.Interpolator1D
    is_spherical: bool

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
        if self.is_spherical:
            values -= 180

        return values

    def index(self, query: Float[Array, "Nq"]) -> Int[Array, "Nq"]:
        """
        Returns the nearest index interpolation for the given query.

        Parameters
        ----------
        query : Float[Array, "Nq"]
            The query array for which the nearest indices are to be found.

        Returns
        -------
        Int[Array, "Nq"]
            An array of integers representing the nearest indices.
        """
        if self.is_spherical:
            query = longitude_in_180_180_degrees(query)  # force to be in -180 to 180 degrees
            query += 180  # shift back to 0 to 360 degrees

        return self.indices(query).astype(int)

    @classmethod
    def from_array(
        cls, values: Float[Array, "dim"], is_spherical: bool = True, **interpolator_kwargs: Any
    ) -> LongitudeCoordinate:
        """
        Create a LongitudeCoordinate object from an array of values.

        This method initializes a LongitudeCoordinate object using the provided array of values.
        It uses a 1D interpolator to generate indices from values, with the interpolation method set to "nearest".

        Parameters
        ----------
        values : Float[Array, "dim"]
            An array of coordinate values.
        is_spherical : bool, optional
            Whether the mesh uses spherical coordinate, defaults to `True`.
        **interpolator_kwargs : Any
            Additional keyword arguments for the interpolator.

        Returns
        -------
        LongitudeCoordinate
            A [`pastax.gridded.LongitudeCoordinate`][] object containing the provided values and corresponding indices
            interpolator.
        """
        if is_spherical:
            values = longitude_in_180_180_degrees(values)  # force to be in -180 to 180 degrees
            values += 180  # shift back to 0 to 360 degrees
            interpolator_kwargs["period"] = 360

        interpolator_kwargs["method"] = "nearest"
        indices = ipx.Interpolator1D(values, jnp.arange(values.size), **interpolator_kwargs)

        return cls(_values=values, indices=indices, is_spherical=is_spherical)
