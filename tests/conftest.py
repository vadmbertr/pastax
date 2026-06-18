import jax
import jax.numpy as jnp
import pytest

# Use CPU and 64-bit floats for test precision
jax.config.update("jax_platform_name", "cpu")
