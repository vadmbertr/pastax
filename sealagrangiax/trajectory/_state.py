from __future__ import annotations
from numbers import Number

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float

from ..utils import UNIT, WHAT


class State(eqx.Module):
    """
    Class representing a state with a value, type, and unit.

    Attributes
    ----------
    value : Float[Array, "state"]
        The value of the state.
    what : WHAT
        The type of the state.
    unit : UNIT
        The unit of the state.

    Methods
    -------
    __init__(value, what=WHAT.unknown, unit=UNIT.dimensionless)
        Initializes the State with given value, type, and unit.
    euclidean_distance(other)
        Computes the Euclidean distance between this state and another state.
    _get_other_numeric(other)
        Checks and converts the type of the other operand.
    __add__(other)
        Adds another state, array, or number to this state.
    __mul__(other)
        Multiplies this state by another state, array, or number.
    __sub__(other)
        Subtracts another state, array, or number from this state.
    __truediv__(other)
        Divides this state by another state, array, or number.
    __eq__(other)
        Checks if this state is equal to another state.
    """
    
    value: Float[Array, "state"]
    what: WHAT
    unit: UNIT

    def __init__ (self, value: Float[Array, "state"], what: WHAT = WHAT.unknown, unit: UNIT = UNIT.dimensionless):
        """
        Initializes the State with given value, type, and unit.

        Parameters
        ----------
        value : Float[Array, "state"]
            The value of the state.
        what : WHAT, optional
            The type of the state (default is WHAT.unknown).
        unit : UNIT, optional
            The unit of the state (default is UNIT.dimensionless).
        """
        self.value = value
        self.what = what
        self.unit = unit

    @eqx.filter_jit
    def euclidean_distance(self, other: State) -> Float[Array, ""]:
        """
        Computes the Euclidean distance between this state and another state.

        Parameters
        ----------
        other : State
            The other state to compute the distance to.

        Returns
        -------
        Float[Array, ""]
            The Euclidean distance between the two states.
        """
        return jnp.sqrt(((self.value - other.value) ** 2).sum())

    @eqx.filter_jit
    def _get_other_numeric(self, other: State | Array | Number, check_type: bool) -> State | Array | Number:
        """
        Converts the other operand to a numeric type if it is a State instance.

        Parameters
        ----------
        other : State or Array or Number
            The other operand to check and convert.
        check_type : bool
            Whether to check the unit type of the other operand.

        Returns
        -------
        State or Array or Number
            The converted operand.

        Raises
        ------
        ValueError
            If the units of the states are different and check_type is True.
        """
        if isinstance(other, State):
            if check_type and other.unit != UNIT.dimensionless and self.unit != other.unit:
                raise ValueError(
                    f"Cannot perform operation between states with different units: {self.unit} and {other.unit}"
                )
            other = other.value

        return other

    @eqx.filter_jit
    def __add__(self, other: State | Array | Number) -> Float[Array, "state"]:
        """
        Adds another state, array, or number to this state.

        Parameters
        ----------
        other : State | Array | Number
            The other operand to add.

        Returns
        -------
        Float[Array, "state"]
            The result of the addition.

        Notes
        -----
        If the other operand is a state, the units must match or the other state must be dimensionless.
        """
        other = self._get_other_numeric(other, check_type=True)

        return self.value + other

    @eqx.filter_jit
    def __mul__(self, other: State | Array | Number) -> Float[Array, "state"]:
        """
        Multiplies this state by another state, array, or number.

        Parameters
        ----------
        other : State | Array | Number
            The other operand to multiply.

        Returns
        -------
        Float[Array, "state"]
            The result of the multiplication.
        """
        other = self._get_other_numeric(other, check_type=False)

        return self.value * other

    @eqx.filter_jit
    def __sub__(self, other: State | Array | Number) -> Float[Array, "state"]:
        """
        Subtracts another state, array, or number from this state.

        Parameters
        ----------
        other : State | Array | Number
            The other operand to subtract.

        Returns
        -------
        Float[Array, "state"]
            The result of the subtraction.

        Notes
        -----
        If the other operand is a state, the units must match or the other state must be dimensionless.
        """
        other = self._get_other_numeric(other, check_type=True)

        return self.value - other

    @eqx.filter_jit
    def __truediv__(self, other: State | Array | Number) -> Float[Array, "state"]:
        """
        Divides this state by another state, array, or number.

        Parameters
        ----------
        other : State | Array | Number
            The other operand to divide by.

        Returns
        -------
        Float[Array, "state"]
            The result of the division.
        """
        other = self._get_other_numeric(other, check_type=False)

        return self.value / other
