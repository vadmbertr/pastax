import jax

# Run the test suite on CPU
jax.config.update("jax_platform_name", "cpu")
