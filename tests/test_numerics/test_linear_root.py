import jax.numpy as jnp
import pytest
from jax import Array

from deepska.glitch.kinds import Num
from deepska.glitch.numerics.linear_root import linear_root


@pytest.mark.parametrize(
    ("xy0", "xy1", "y", "expected_x"),
    [
        pytest.param(jnp.array([0, 0]), jnp.array([1, 1]), 0.5, 0.5, id="simple"),
        pytest.param(jnp.array([-1, -1]), jnp.array([1, 1]), 0, 0, id="root"),
        pytest.param(
            jnp.array([5, 4]), jnp.array([-3, 5]), 4.125, 4, id="negative big change"
        ),
    ],
)
def test_linear_root(xy0: Array, xy1: Array, y: Num, expected_x: Num) -> None:
    x0, y0 = xy0
    x1, y1 = xy1
    assert expected_x == linear_root(x0, x1, y0, y1, y)
