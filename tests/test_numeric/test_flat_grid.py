import jax.numpy as jnp
from deepska.glitch.numerics.flat_grid import flat_grid

def test_flat_grid_3_by_4() -> None:
    """ test flat grid with 3 values of one input and 4 of another """
    input = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6, 7])}
    draws, shape = flat_grid(input)
    expect_shape = (3, 4)
    expect_draws = {
        "a": jnp.array([1]*4 + [2]*4 + [3]*4),
        "b": jnp.array([4, 5, 6, 7]*3),
    }
    assert expect_shape == shape
    assert set(expect_draws.keys()) == set(draws.keys())

    for key in expect_draws:
        
        assert jnp.all(expect_draws[key] == draws[key])

    pass