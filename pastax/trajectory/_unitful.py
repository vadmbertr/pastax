from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..utils._unit import compose_units, Unit


def unit_converter(x: Unit | dict[Unit, int]) -> dict[Unit, int]:
    """
    Converts a [`pastax.utils.Unit`][] to a dictionary with [`pastax.utils.Unit`][] as keys and their exponents as
    values.

    Parameters
    ----------
    x : Unit or dict[Unit, int]
        A [`pastax.utils.Unit`][] or a dictionary with [`pastax.utils.Unit`][] as keys and their exponents as values.

    Returns
    -------
    dict[Unit, int]
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
    _unit : dict[Unit, int | float], optional
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

    _value: Array = eqx.field(converter=lambda x: jnp.asarray(x, dtype=float))
    _unit: dict[Unit, int | float] = eqx.field(static=True, converter=unit_converter)

    def __init__(self, value: Array = jnp.asarray(jnp.nan, dtype=float), unit: dict[Unit, int | float] = {}):
        """
        Initializes the [`pastax.utils.Unitful`][] with given value and unit.

        Parameters
        ----------
        value : Array
            The value of the quantity.
        unit : dict[Unit, int | float], optional
            The unit of the quantity, defaults to an empty dict.
        """
        self._value = value
        self._unit = unit

    @property
    def value(self) -> Array:
        """
        Returns the value of the quantity.

        Returns
        -------
        Array
            The value of the quantity.
        """
        return self._value

    @property
    def unit(self) -> dict[Unit, int | float]:
        """
        Returns the unit of the quantity.

        Returns
        -------
        dict[Unit, int | float]
            The unit of the quantity.
        """
        return self._unit

    def cumsum(self, axis: ArrayLike | None = None) -> Unitful:
        """
        Computes the cumulative sum of the quantity.

        Parameters
        ----------
        axis : ArrayLike | None, optional
            Axis along which the cumulative sum to be computed. If None (default), the cumulative sum is computed over
            the flattened array.

        Returns
        -------
        Unitful
            The cumulative sum of the quantity.
        """
        return Unitful(self.value.cumsum(axis), self.unit)  # type: ignore

    def euclidean_distance(self, other: Unitful | Array) -> Unitful:
        """
        Computes the Euclidean distance between this quantity and another quantity.

        Parameters
        ----------
        other : Unitful | Array
            The other quantity to compute the distance to.

        Returns
        -------
        Unitful
            The Euclidean distance between the two quantities.
        """
        return ((self - other) ** 2).sum().sqrt()

    def mean(self, axis: ArrayLike | None = None) -> Unitful:
        """
        Computes the mean of the quantity along the specified axis.

        Parameters
        ----------
        axis : ArrayLike | None, optional
            The axis along which to compute the mean, defaults to None, meaning along all axes.

        Returns
        -------
        Unitful
            The mean of the quantity along the specified axis.
        """
        return Unitful(self.value.mean(axis=axis), self.unit)  # type: ignore

    def sqrt(self) -> Unitful:
        """
        Computes the square root of the quantity.

        Returns
        -------
        Unitful
            The square root of the quantity.
        """
        return self ** (1 / 2)

    def sum(self, axis: ArrayLike | None = None) -> Unitful:
        """
        Computes the sum of the quantity.

        Parameters
        ----------
        axis : ArrayLike | None, optional
            Axis along which the sum to be computed. If None (default), the sum is computed along all the axes.

        Returns
        -------
        Unitful
            The sum of the quantity.
        """
        return Unitful(self.value.sum(axis), self.unit)  # type: ignore

    def __extract_from_other(
        self, other: Unitful | ArrayLike, additive_op: bool
    ) -> tuple[ArrayLike, dict[Unit, int | float] | None]:
        """
        Extract value and unit from other operand if it is a [`pastax.utils.Unitful`][] instance.

        Parameters
        ----------
        other : Unitful or ArrayLike
            The other operand to extract value and unit from if it is a [`pastax.utils.Unitful`][].
        additive : bool
            Type of operation performed with self and other.

        Returns
        -------
        tuple[ArrayLike, dict[Unit, int | float] | None]
            The value and unit of the other operand.

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

        return other_value, other_unit  # type: ignore

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
        other : Unitful | Array
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
        return Unitful(self.value**pow, unit)

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
