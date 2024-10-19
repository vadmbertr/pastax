from __future__ import annotations
from typing import Dict

import equinox as eqx
from jaxtyping import ArrayLike

from ..utils.unit import Unit
from ._unitful import Unitful


class State(Unitful):
    """
    Class representing a state with a value, unit, and name.

    Attributes
    ----------
    _value : Float[Array, "state"]
        The value of the state.
    _unit : Dict[Unit, int | float], optional
        The unit of the state (default is an empty Dict).
    name : str, optional
        The name of the state (default is None).

    Methods
    -------
    __init__(value, unit={}, name=None)
        Initializes the State with given value, unit and name.
    attach_name(name)
        Attaches a name to the state.
    """
    
    name: str = eqx.field(static=True)

    def __init__ (self, value: ArrayLike, unit: Unit | Dict[Unit, int | float] = {}, name: str = None):
        """
        Initializes the State with given value, unit and name.

        Parameters
        ----------
        value : ArrayLike
            The value of the state.
        unit : Unit | Dict[Unit, int | float], optional
            The unit of the state (default is an empty Dict).
        name : str, optional
            The type of the state (default is None).
        """
        super().__init__(value, unit)
        self.name = name
    
    def attach_name(self, name: str) -> State:
        """
        Attaches a name to the state.

        Parameters
        ----------
        name : str
            The name to attach to the state.

        Returns
        -------
        State
            A new state with the attached name.
        """
        return self.__class__(self.value, self.unit, name)
    