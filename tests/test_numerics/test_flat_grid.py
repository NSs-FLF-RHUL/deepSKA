import jax.numpy as jnp

from deepska.glitch.numerics.flat_grid import flat_grid


def test_flat_grid_3_by_4() -> None:
    input = {
        "a": jnp.array([1, 2, 3]),
        "b": jnp.array([4, 5, 6, 7]),
    }
    draws, shape = flat_grid(input)
    expect_shape = (3, 4)
    expect_draws = {
        "a": jnp.array([1] * 4 + [2] * 4 + [3] * 4),
        "b": jnp.array([4, 5, 6, 7] * 3),
    }

    assert expect_shape == shape
    # JAX doesn't implement == on arrays and return a
    # scalar, which is what dictionary comparison wants to
    # do.
    assert set(expect_draws.keys()) == set(draws.keys())
    for key, expect_array in expect_draws.items():
        assert jnp.all(expect_array == draws[key])


def test_flat_grid_4_by_3() -> None:
    input = {
        "a": jnp.array([1, 2, 3, 4]),
        "b": jnp.array([5, 6, 7]),
    }
    draws, shape = flat_grid(input)
    expect_shape = (4, 3)
    expect_draws = {
        "a": jnp.array([1] * 3 + [2] * 3 + [3] * 3 + [4] * 3),
        "b": jnp.array([5, 6, 7] * 4),
    }

    assert expect_shape == shape
    # JAX doesn't implement == on arrays and return a
    # scalar, which is what dictionary comparison wants to
    # do.
    assert set(expect_draws.keys()) == set(draws.keys())
    for key, expect_array in expect_draws.items():
        assert jnp.all(expect_array == draws[key])


def test_flat_grid_trivial_array() -> None:
    input = {
        "a": jnp.array([0]),
        "b": jnp.array([4, 5, 6, 7]),
    }
    draws, shape = flat_grid(input)
    expect_shape = (1, 4)
    expect_draws = {
        "a": jnp.array([0] * 4),
        "b": jnp.array([4, 5, 6, 7]),
    }

    assert expect_shape == shape
    # JAX doesn't implement == on arrays and return a
    # scalar, which is what dictionary comparison wants to
    # do.
    assert set(expect_draws.keys()) == set(draws.keys())
    for key, expect_array in expect_draws.items():
        assert jnp.all(expect_array == draws[key])
