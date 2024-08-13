from __future__ import annotations
from typing import Dict

import equinox as eqx
import jax


class Batch(eqx.Module):
    _members: jax.Pytree | Dict[str, jax.Pytree] | [jax.Pytree, ...]
    size: int
