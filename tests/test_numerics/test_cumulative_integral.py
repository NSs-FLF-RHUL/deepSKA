import jax.numpy as jnp

from deepska.glitch.numerics.cumulative_integral import cumulative_integral


def test_cumulative_integral_constant() -> None:
    y = jnp.array(3 * [1])
    dt = 0.1
    expected_output = jnp.array([0.1, 0.2, 0.3])
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)


def test_cumulative_integral_linear() -> None:
    y = jnp.array([1, 2, 3])
    dt = 0.1
    expected_output = jnp.array([0.1, 0.3, 0.6])
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)
