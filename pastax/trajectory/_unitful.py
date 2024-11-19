from __future__ import annotations
from typing import Dict, Type

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..utils.unit import compose_units, Unit


def unit_converter(x: Unit | Dict[Unit, int]) -> Dict[Unit, int]:
    """
    Converts a [`pastax.utils.Unit`][] to a dictionary with [`pastax.utils.Unit`][] as keys and their exponents as values.

    Parameters
    ----------
    x : Unit or Dict[Unit, int]
        A [`pastax.utils.Unit`][] or a dictionary with [`pastax.utils.Unit`][] as keys and their exponents as values.

    Returns
    -------
    Dict[Unit, int]
        A dictionary with [`pastax.utils.Unit`][] as keys and their exponents as values.
    """
    if isinstance(x, Unit):
        return {x: 1}
    else:
        return x


class Unitful(eqx.Module):
    """
    Class representing a quantity and its associated dictionary of [`pastax.utils.Unit`][].

    Attributes
    ----------
    _value : Float[Array, "quantity"]
        The value of the quantity.
    _unit : Dict[Unit, int | float], optional
        The dictionary of [`pastax.utils.Unit`][] of the quantity, defaults to an empty dictionary (i.e. no unit).

    Methods
    -------
    __init__(value, unit={}, name=None)
        Initializes the [`pastax.utils.Unitful`][] with given value, and unit.
    value
        Returns the value of the quantity.
    cumsum(axis=None)
        Computes the cumulative sum of the quantity along the specified axis.
    euclidean_distance(other)
        Computes the Euclidean distance between this quantity and another quantity.
    mean(axis=None)
        Computes the mean of the quantity along the specified axis.
    sqrt()
        Computes the square root of the quantity.
    sum(axis=None)
        Computes the sum of the quantity along the specified axis.
    __add__(other)
        Adds another quantity or array like to this quantity.
    __mul__(other)
        Multiplies this quantity by another quantity or array like.
    __pow__(pow)
        Raises this quantity to the power of pow.
    __truediv__(other)
        Divides this quantity by another quantity or array like.
    __sub__(other)
        Subtracts another quantity or array like from this quantity.
    """
    
    _value: ArrayLike = eqx.field(converter=lambda x: jnp.asarray(x, dtype=float))
    _unit: Dict[Unit, int | float] = eqx.field(static=True, converter=unit_converter)

    def __init__ (self, value: ArrayLike = jnp.nan, unit: Unit | Dict[Unit, int | float] = {}):
        """
        Initializes the [`pastax.utils.Unitful`][] with given value and unit.

        Parameters
        ----------
        value : ArrayLike
            The value of the quantity.
        unit : Unit | Dict[Unit, int | float], optional
            The unit of the quantity, defaults to an empty Dict.
        """
        self._value = value
        self._unit = unit

    @property
    def value(self) -> ArrayLike:
        """
        Returns the value of the quantity.

        Returns
        -------
        ArrayLike
            The value of the quantity.
        """
        return self._value

    @property
    def unit(self) -> Dict[Unit, int | float]:
        """
        Returns the unit of the quantity.

        Returns
        -------
        Dict[Unit, int | float]
            The unit of the quantity.
        """
        return self._unit
    
    def cumsum(self, axis: ArrayLike = None) -> Unitful:
        """
        Computes the cumulative sum of the quantity.

        Parameters
        ----------
        axis : ArrayLike, optional
            Axis along which the cumulative sum to be computed. If None (default), the cumulative sum is computed over the flattened array.

        Returns
        -------
        Unitful
            The cumulative sum of the quantity.
        """
        return Unitful(self.value.cumsum(axis), self.unit)

    def euclidean_distance(self, other: Unitful | ArrayLike) -> Unitful:
        """
        Computes the Euclidean distance between this quantity and another quantity.

        Parameters
        ----------
        other : Unitful | ArrayLike
            The other quantity to compute the distance to.

        Returns
        -------
        Unitful
            The Euclidean distance between the two quantities.
        """
        return ((self - other) ** 2).sum().sqrt()
    
    def mean(self, axis: ArrayLike = None) -> Unitful:
        """
        Computes the mean of the quantity along the specified axis.

        Parameters
        ----------
        axis : ArrayLike, optional
            The axis along which to compute the mean, defaults to None, meaning along all axes.

        Returns
        -------
        Unitful
            The mean of the quantity along the specified axis.
        """
        return Unitful(self.value.mean(axis=axis), self.unit)

    def sqrt(self) -> Unitful:
        """
        Computes the square root of the quantity.

        Returns
        -------
        Unitful
            The square root of the quantity.
        """
        return self**(1/2)
    
    def sum(self, axis: ArrayLike = None) -> Unitful:
        """
        Computes the sum of the quantity.

        Parameters
        ----------
        axis : ArrayLike, optional
            Axis along which the sum to be computed. If None (default), the sum is computed along all the axes.

        Returns
        -------
        Unitful
            The sum of the quantity.
        """
        return Unitful(self.value.sum(axis), self.unit)

    def __extract_from_other(
        self, 
        other: Unitful | ArrayLike, 
        additive_op: bool
    ) -> tuple[ArrayLike, str | None, Type]:
        """
        Extract value and unit from other operand if it is a [`pastax.utils.Unitful`][] instance.

        Parameters
        ----------
        other : Unitful or ArrayLike
            The other operand to extract value and unit from if it is a Unitful.
        additive : bool
            Type of operation performed with self and other.

        Returns
        -------
        tuple[ArrayLike, str | None, Type]
            The value and unit of the other operand, and the class of the result of the operation type.

        Raises
        ------
        ValueError
            If the units of the quantities are different and additive is True.
        """
        other_value = other
        other_unit = None

        if isinstance(other, Unitful):
            if additive_op:
                if self.unit != other.unit:
                    raise ValueError(
                        "Cannot perform additive operation between quantities with incompatible units: "
                        f"{self.unit} and {other.unit}"
                    )
            other_value = other.value
            other_unit = other.unit

        return other_value, other_unit

    def __add__(self, other: Unitful | ArrayLike) -> Unitful:
        """
        Adds another quantity or array like to this quantity.

        Parameters
        ----------
        other : Unitful | ArrayLike
            The other operand to add.

        Returns
        -------
        Unitful
            The addition of the two operands.

        Notes
        -----
        If the other operand is a [`pastax.utils.Unitful`], the unit must match.
        """
        other_value, _ = self.__extract_from_other(other, additive_op=True)
        return Unitful(self.value + other_value, self.unit)

    def __mul__(self, other: Unitful | ArrayLike) -> Unitful:
        """
        Multiplies this quantity by another quantity or array like.

        Parameters
        ----------
        other : Unitful | ArrayLike
            The other operand to multiply.

        Returns
        -------
        Unitful
            The multiplication of the two operands.
        """
        other_value, other_unit = self.__extract_from_other(other, additive_op=False)
        return Unitful(self.value * other_value, compose_units(self.unit, other_unit, 1))

    def __pow__(self, pow: int | float) -> Unitful:
        """
        Raises this quantity to the power of pow.

        Parameters
        ----------
        power : Number
            The power to apply.

        Returns
        -------
        Unitful
            The power pow of the quantity.
        """
        unit = {k: v * pow for k, v in self.unit.items()}
        return Unitful(self.value ** pow, unit)

    def __truediv__(self, other: Unitful | ArrayLike) -> Unitful:
        """
        Divides this quantity by another quantity or array like.

        Parameters
        ----------
        other : Unitful | ArrayLike
            The other operand to divide by.

        Returns
        -------
        Unitful
            The division of the two operands.
        """
        other_value, other_unit = self.__extract_from_other(other, additive_op=False)
        return Unitful(self.value / other_value, compose_units(self.unit, other_unit, -1))

    def __sub__(self, other: Unitful | ArrayLike) -> Unitful:
        """
        Subtracts another quantity or array like from this quantity.

        Parameters
        ----------
        other : Unitful | ArrayLike
            The other operand to subtract.

        Returns
        -------
        Unitful
            The subtraction of the two operands.

        Notes
        -----
        If the other operand is a [`pastax.utils.Unitful`], the units must match.
        """
        other_value, _ = self.__extract_from_other(other, additive_op=True)
        return Unitful(self.value - other_value, self.unit)
