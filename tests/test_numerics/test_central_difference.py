import pytest
import jax.numpy as jnp
from deepska.glitch.numerics.central_difference import central_difference

def test_constant() -> None:
    dx = 0.5
    y = jnp.array([1.]*4)
    dy = central_difference(y,dx)
    expect_dy = jnp.array([0]*4)
    assert jnp.all(expect_dy==dy)

def test_linear() -> None:
    dx = 0.5
    y=  jnp.array([1.0, 2.0, 3.0, 4.0])
    
    dy = central_difference(y,dx)
    expect_dy = jnp.array([0.0, 2.0, 2.0, 0.0])
    assert jnp.all(expect_dy == dy)

def test_quadratic() -> None:
    dx = 0.5
    x = jnp.arange(0.0,5.0,dx,dtype=jnp.float64)
    y=  x**2
    
    dy = central_difference(y,dx)
    expect_dy = 2 * x
    expect_dy = expect_dy.at[len(y)-1].set(0.0)

    assert jnp.all(expect_dy == dy)

def test_trivial() -> None:
    dx = 0.5
    y = jnp.array([0.0, 0.0, 0.0])
    dy = central_difference(y,dx)
    expect_dy = jnp.array([0.0, 0.0, 0.0])
    assert jnp.all(expect_dy == dy)

def test_1_by_1() -> None:
    dx = 0.5
    y = jnp.array([0.0])
    dy = central_difference(y,dx)
    expect_dy = jnp.array([0.0])
    assert jnp.all(expect_dy == dy) 

def test_2_by_1() -> None:
    dx = 0.5
    y = jnp.array([1.0, 2.0])
    dy = central_difference(y,dx)
    expect_dy = jnp.array([0.0, 0.0])
    assert jnp.all(expect_dy == dy)        

