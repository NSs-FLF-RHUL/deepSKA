from collections.abc import Callable

import jax.numpy as jnp
import pytest
from jax import Array

from deepska.glitch.kinds import Num, Star
from deepska.glitch.numerics.every_pair import every_pair


@pytest.fixture
def run() -> Callable:
    """Test functional argument, returns a scalar."""

    def _inner(i: Num, o: Star, a1: Num = 1.0, a2: Num = 0.0) -> Num:
        return a1 * i * o - a2

    return _inner


@pytest.fixture
def run_return_tuple() -> Callable:
    """Test functional argument, returns an array."""

    def _inner(i: Num, o: Star, a1: Num, a2: Num) -> Array:
        return jnp.array([a1 * i, a2 * o])

    return _inner


@pytest.mark.parametrize(
    ("i", "o", "args", "expected"),
    [
        pytest.param(
            jnp.array([1, 2, 3]),
            jnp.array([4, 5, 6]),
            [1, 0],
            jnp.array([[4, 8, 12], [5, 10, 15], [6, 12, 18]]),
            id="Basic integer test",
        ),
        pytest.param(
            jnp.array([]), jnp.array([3, 4, 5]), [], jnp.array([]), id="One input empty"
        ),
    ],
)
def test_numerical_vals(
    i: Num, o: Star, run: Callable, expected: Array, args: tuple[object, ...]
) -> None:
    computed = every_pair(run, i, o, args)
    assert jnp.allclose(expected, computed)


@pytest.mark.parametrize(
    ("i", "o", "args", "expected"),
    [
        pytest.param(
            jnp.array([1, 2]),
            jnp.array([3, 4]),
            [1, 0],
            jnp.array([[[1, 0], [2, 0]], [[1, 0], [2, 0]]]),
        ),
    ],
)
def test_return_tuple(
    i: Num,
    o: Star,
    run_return_tuple: Callable,
    args: tuple[object, ...],
    expected: Array,
) -> None:
    computed = every_pair(run_return_tuple, i, o, args)
    for j, a in enumerate(computed):
        assert jnp.allclose(a, expected[j])
