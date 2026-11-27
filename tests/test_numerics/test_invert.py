import jax.numpy as jnp
import pytest
from jax import Array

from deepska.glitch.kinds import Fn, Num
from deepska.glitch.numerics.invert import bis, bisection, invert


@pytest.mark.parametrize(
    ("carry", "f", "target", "up", "target_new_carry"),
    [
        pytest.param(
            (jnp.array(0.0), jnp.array(1.0)),
            lambda x: x,
            0.5,
            True,
            (jnp.array(0.0), jnp.array(0.5)),
            id="simple up",
        ),
        pytest.param(
            (jnp.array(2.0), jnp.array(0.0)),
            lambda x: x**3,
            1.5,
            False,
            (jnp.array(2), jnp.array(1.0)),
            id="up is false case",
        ),
        pytest.param(
            (jnp.array(0), jnp.array(2)),
            lambda x: x,
            1,
            True,
            (jnp.array(0), jnp.array(1.0)),
            id="mid point lies on edge of bracket",
        ),
    ],
)
def test_bis(
    carry: tuple[Array, Array],
    f: Fn,
    target: Num,
    up: Num,
    target_new_carry: tuple[Array, Array],
) -> None:
    new_carry, voidoutput = bis(carry, f, target, up)
    low, high = new_carry
    test_low, test_high = target_new_carry
    assert voidoutput == 0.0
    assert test_low == low
    assert test_high == high


@pytest.mark.parametrize(
    ("carry", "f", "target", "up", "n", "target_final_range"),
    [
        pytest.param(
            (jnp.array(0.0), jnp.array(1.0)),
            lambda x: x,
            1 / (2**8),
            True,
            8,
            (jnp.array(0.0), jnp.array(1 / (2**8))),
            id="identity function with 8 halvings",
        ),
        pytest.param(
            (jnp.array(1), jnp.array(0.0)),
            lambda x: x,
            1 / (2**8),
            False,
            8,
            (jnp.array(1 / (2**8)), jnp.array(0)),
            id="identity function with 8 halvings, downwards",
        ),
    ],
)
def test_bisection(
    carry: tuple[Array, Array],
    f: Fn,
    target: Num,
    up: Num,
    n: int,
    target_final_range: tuple[Array, Array],
) -> None:
    new_carry = bisection(f, target, carry, up, n)
    low, high = new_carry
    final_low, final_high = target_final_range
    assert final_low == low
    assert final_high == high


@pytest.mark.parametrize(
    ("f", "target", "low", "high", "n", "target_new_x"),
    [
        pytest.param(
            lambda x: 2 * x,
            1,
            0,
            1,
            1,
            0.5,
            id="simple linear function",
        ),
    ],
)
def test_invert(
    f: Fn, target: Num, low: Num, high: Num, n: int, target_new_x: Array
) -> None:
    new_x = invert(f, target, low, high, n)
    assert target_new_x == new_x
