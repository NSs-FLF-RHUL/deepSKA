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


def test_cumulative_integral_zero_dt() -> None:
    y = jnp.array([3.5, 3.9, 4.1])
    dt = 0.
    expected_output = jnp.array([0., 0., 0.])
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)


def test_cumulative_integral_zero_y() -> None:
    y = jnp.array([0.])
    dt = 0.5
    expected_output = jnp.array([0.])
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)


def test_cumulative_integral_2d() -> None:
    y = jnp.array([3 * [1], [1, 2, 3]])
    dt = 0.1
    expected_output = jnp.array([[0.1, 0.2, 0.3], [0.1, 0.3, 0.6]])
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)


def test_cumulative_integral_random() -> None:
    y = jnp.array([0.2, 1.4, 3.9])
    dt = 0.1
    expected_output = jnp.array([0.02, 0.16, 0.55])
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)


def test_cumulative_integral_1d_dt() -> None:
    y = jnp.array([1, 2, 3])
    dt = jnp.array([0.1, 1., 10.])
    expected_output = jnp.array([0.1, 2.1, 32.1])
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)
