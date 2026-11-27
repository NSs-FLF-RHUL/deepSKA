from deepska.glitch.numerics.every_pair import every_pair
from deepska.glitch.kinds import Fn, Num, Star
import jax.numpy as jnp
import jax
from jax import Array
import pytest

@pytest.fixture
def run():
    def _inner(i, o, a1=1, a2=0):
        return a1*i*o - a2
    return _inner

@pytest.mark.parametrize(
        ("i", "o", "args", "expected"),
    [
        pytest.param(
            jnp.array([1,2,3]),
            jnp.array([4,5,6]),
            [1, 0],
            jnp.array([[4,8,12],
            [5,10,15],
            [6,12,18]],),
            id="Basic integer test"),

        pytest.param(
            jnp.array([]),
            jnp.array([3,4,5]),
            [],
            jnp.array([]),
            id="One input empty"
        )
    ]
)     
def test_numerical_vals(i,o,run, expected, args):
    computed = every_pair(run, i, o, args)
    assert jnp.allclose(expected, computed)
