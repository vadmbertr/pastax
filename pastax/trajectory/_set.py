from __future__ import annotations

from typing import Mapping, Sequence

import equinox as eqx
from jaxtyping import PyTree


class Set(eqx.Module):
    """
    Base class representing a set of PyTrees.

    Attributes
    ----------
    size : int
        The number of members in the set.
    """

    _members: PyTree | Mapping[str, PyTree] | Sequence[PyTree]
    size: int
