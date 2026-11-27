import jax.numpy as jnp
import pytest
from jax import Array

from deepska.glitch.numerics.flat_grid import flat_grid


@pytest.mark.parametrize(
    ("input_array", "expected_shape", "expected_draws"),
    [
        pytest.param(
            {
                "a": jnp.array([1, 2]),
                "b": jnp.array([3, 4]),
                "c": jnp.array([5, 6]),
            },
            (2, 2, 2),
            {
                "a": jnp.array([1] * 4 + [2] * 4),
                "b": jnp.array([3, 3, 4, 4] * 2),
                "c": jnp.array([5, 6] * 4),
            },
            id="3 arrays",
        ),
        pytest.param(
            {
                "b": jnp.array([1, 2, 3, 4]),
                "a": jnp.array([5, 6, 7]),
            },
            (4, 3),
            {
                "a": jnp.array([5, 6, 7] * 4),
                "b": jnp.array([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3),
            },
            id="Array order is dict order",
        ),
        pytest.param(
            {
                "a": jnp.array([0]),
                "b": jnp.array([4, 5, 6, 7]),
            },
            (1, 4),
            {
                "a": jnp.array([0] * 4),
                "b": jnp.array([4, 5, 6, 7]),
            },
            id="Trivial array",
        ),
        pytest.param(
            {
                "a": jnp.array([1, 2, 3, 4]),
                "b": jnp.array([5, 6, 7]),
            },
            (4, 3),
            {
                "a": jnp.array([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3),
                "b": jnp.array([5, 6, 7] * 4),
            },
            id="(4,3) output shape",
        ),
        pytest.param(
            {
                "a": jnp.array([1, 2, 3]),
                "b": jnp.array([4, 5, 6, 7]),
            },
            (3, 4),
            {
                "a": jnp.array([1] * 4 + [2] * 4 + [3] * 4),
                "b": jnp.array([4, 5, 6, 7] * 3),
            },
            id="(3,4) output shape",
        ),
    ],
)
def test_flat_grid(
    input_array: Array,
    expected_shape: tuple[int, ...],
    expected_draws: dict[str, Array],
) -> None:
    draws, shape = flat_grid(input_array)

    assert expected_shape == shape
    # JAX doesn't implement == on arrays and return a
    # scalar, which is what dictionary comparison wants to
    # do.
    assert set(expected_draws.keys()) == set(draws.keys())
    for key, expect_array in expected_draws.items():
        assert jnp.all(expect_array == draws[key])
