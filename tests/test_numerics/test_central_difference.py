import jax.numpy as jnp
import pytest

from deepska.glitch.kinds import Arr, Num
from deepska.glitch.numerics.central_difference import central_difference


@pytest.mark.parametrize(
    ("y", "dx", "expect_dy"),
    [
        pytest.param(
            jnp.array([1.0] * 4),
            0.5,
            jnp.array([0] * 4),
            id="constant y",
        ),
        pytest.param(
            jnp.array([1.0, 2.0, 3.0, 4.0]),
            0.5,
            jnp.array([0.0, 2.0, 2.0, 0.0]),
            id="linear y",
        ),
        pytest.param(
            jnp.array([1.0, 4.0, 9.0, 16.0]),
            1.0,
            jnp.array([0.0, 4.0, 6.0, 0.0]),
            id="Quadratic y",
        ),
        pytest.param(
            jnp.array([0.0, 0.0, 0.0]),
            0.5,
            jnp.array([0.0, 0.0, 0.0]),
            id="trivial null",
        ),
        pytest.param(jnp.array([0.0]), 0.5, jnp.array([0.0]), id="len(y) = 1"),
        pytest.param(
            jnp.array([0.0, 0.5]), 0.5, jnp.array([0.0, 0.0]), id="len(y) = 2"
        ),
    ],
)
def test_central_difference_all(y: Arr, dx: Num, expect_dy: Arr) -> None:
    dy = central_difference(y, dx)
    assert jnp.all(expect_dy == dy)
