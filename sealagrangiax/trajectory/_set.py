from __future__ import annotations
from typing import Mapping, Sequence

import equinox as eqx
import jax


class Set(eqx.Module):
    """
    Base class representing a set of Pytrees.

    Attributes
    ----------
    _members : jax.Pytree or Mapping[str, jax.Pytree] or Sequence[jax.Pytree]
        The members of the set.
    size : int
        The number of members in the set.
    """
    _members: jax.Pytree | Mapping[str, jax.Pytree] | Sequence[jax.Pytree]
    size: int
