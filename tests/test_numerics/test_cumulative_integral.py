import jax.numpy as jnp

import pytest

from deepska.glitch.numerics.cumulative_integral import cumulative_integral


@pytest.mark.parametrize(
    ("y", "dt", "expected_output"),
    [
        pytest.param(
            2.5, 0.1, jnp.array(0.25), id="floats"
        ),
        pytest.param(
            jnp.array(3 * [1]),
            0.1,
            jnp.array([0.1, 0.2, 0.3]),
            id="constant_y"
        ),
        pytest.param(
            jnp.array([1, 2, 3]),
            0.1,
            jnp.array([0.1, 0.3, 0.6]),
            id="linear_y"
        ),
        pytest.param(
            jnp.array([3.5, 3.9, 4.1]),
            0.,
            jnp.array([0., 0., 0.]),
            id="zero_dt"
        ),
        pytest.param(
            jnp.array([0.]),
            0.5,
            jnp.array([0.]),
            id="zero_y"
        ),
        pytest.param(
            jnp.array([3 * [1], [1, 2, 3]]),
            0.1,
            jnp.array([[0.1, 0.2, 0.3], [0.1, 0.3, 0.6]]),
            id="2d_y"
        ),
        pytest.param(
            0.1,
            jnp.array([3 * [1], [1, 2, 3]]),
            jnp.array([[0.1, 0.2, 0.3], [0.1, 0.3, 0.6]]),
            id="2d_dt"
        ),
        pytest.param(
            jnp.array([0.2, 1.4, 3.9]),
            0.1,
            jnp.array([0.02, 0.16, 0.55]),
            id="random_y"
        ),
        pytest.param(
            jnp.array([1, 2, 3]),
            jnp.array([0.1, 1., 10.]),
            jnp.array([0.1, 2.1, 32.1]),
            id="1ds"
        )
    ]
)
def test_cumulative_integral_all(y, dt, expected_output) -> None:
    output = cumulative_integral(y, dt)
    assert jnp.allclose(output, expected_output)
