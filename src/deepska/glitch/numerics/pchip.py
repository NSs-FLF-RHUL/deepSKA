# Copyright (C) 2026 Isaac Dodds, Royal Holloway University of London
"""
Pchip spline, used as it keeps the shape of the data.

No overshoot between knots like a normal cubic spline.

pchip_gap: The gaps between pchip knots.
pchip_interior: Slope at the pchip knots that arent on an edge.
pchip_edge: Slope at the two end knots of a pchip. End knots only have a neighbour on
one side so the interior formula doesnt work.
pchip_build: The spline params from the points.
pad_increasing: Fixing up padding rows so pchip can take them.
pchip_interval: Which interval between knots a point falls in.
pchip_eval: The spline itself, from the params pchip_build gives.
pchip_area: Integral of one whole pchip interval.
pchip_part: Integral of part of a pchip interval.
pchip_integral: Integral of a pchip between any two points. Exact as its just cubics.
loglog_pchip_build: Pchip in log x and log y.
loglog_pchip_eval: Reading a log-log pchip back off.

"""

import jax
import jax.numpy as jnp
from jax import Array

from deepska.glitch.kinds import Arr, Num

MIN_KNOTS = 3  # two ends and at least one middle point


def pchip_gap(i: Num, x: Arr, y: Arr) -> tuple[Array, Array]:
    """
    Calculate the width of a gap between knots and the gradient over it.

    :param i: Which gap.
    :param x: Knot positions.
    :param y: Value at each knot.
    :returns h_i: Width of the gap.
    :returns delta_i: Gradient over the entire width.
    """
    h_i = x[i + 1] - x[i]
    delta_i = (y[i + 1] - y[i]) / h_i
    return h_i, delta_i


def pchip_interior(i: Num, h: Arr, delta: Arr) -> Array:
    """
    Calculate the slope at a knot that isnt on an edge.

    :param i: Which knot, from 1 to n-2.
    :param h: Width of every gap.
    :param delta: Gradient over every gap.
    :returns: The slope at knot i, 0 where the gaps either side slope opposite ways.
    """
    w1 = 2.0 * h[i] + h[i - 1]  # weights, wider gap counts for more
    w2 = h[i] + 2.0 * h[i - 1]
    return jnp.where(
        delta[i - 1] * delta[i] <= 0.0,
        0.0,
        # knots need to be same sign as whole area slope
        (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]),
    )


def pchip_edge(h_a: Num, h_b: Num, d_a: Num, d_b: Num) -> Array:
    """
    Calculate the slope at an end knot of a pchip.

    One sided estimate, then clamped.

    :param h_a: Width of the gap next to the end.
    :param h_b: Width of the gap after that.
    :param d_a: Gradient over the gap next to the end.
    :param d_b: Gradient over the gap after that.
    :returns: The slope at the end knot.
    """
    m = ((2.0 * h_a + h_b) * d_a - h_a * d_b) / (h_a + h_b)
    # points the wrong way, flatten it
    m = jnp.where(jnp.sign(m) != jnp.sign(d_a), 0.0, m)
    # the two gaps disagree, cap at 3x the first slope
    return jnp.where(
        (jnp.sign(d_a) != jnp.sign(d_b)) & (jnp.abs(m) > 3.0 * jnp.abs(d_a)),
        3.0 * d_a,
        m,
    )


def pchip_build(
    x: Arr, y: Num, n_valid: Num = None
) -> tuple[Array, Array, Array, Array]:
    """
    Build a pchip spline through the points.

    Gives the params and not a function, as cant return a function and vmap it.

    :param x: Knot positions, at least 3 and always increasing.
    :param y: Value at each knot.
    :param n_valid: Number of real points if the arrays have padding on the end.
    :returns: The params (x, y, h, m), h the gap widths and m the knot slopes.
    """
    n = x.shape[0]
    if n < MIN_KNOTS:  # as needs two end points and middle points to work
        msg = "pchip needs at least 3 points"
        raise ValueError(msg)

    # needs order of array of x to always be increasing, nan if not
    ok = jnp.all(x[1:] > x[:-1])
    y = jnp.where(ok, y, jnp.nan)

    h, delta = jax.vmap(pchip_gap, in_axes=(0, None, None))(jnp.arange(n - 1), x, y)

    # n-2 interior slopes
    m_int = jax.vmap(pchip_interior, in_axes=(0, None, None))(
        jnp.arange(1, n - 1), h, delta
    )

    m_0 = pchip_edge(h[0], h[1], delta[0], delta[1])
    m_n1 = pchip_edge(h[-1], h[-2], delta[-1], delta[-2])
    m = jnp.concatenate([m_0[None], m_int, m_n1[None]])  # one array

    if n_valid is not None:
        k = n_valid - 1  # last real point
        # its really an end point, everything after it is padding
        m = m.at[k].set(pchip_edge(h[k - 1], h[k - 2], delta[k - 1], delta[k - 2]))
    # return param for actual function as cant return a function and vmap it
    return (x, y, h, m)


def pad_increasing(x: Arr, n_valid: Num) -> Array:
    """
    Overwrite the padding rows of x so it keeps increasing.

    The padding rows repeat the last row and pchip cant take repeated x. Cant chop them
    off as jit needs every array the same length.

    :param x: Values that increase up to row n_valid.
    :param n_valid: Number of real rows before the padding.
    :returns: x with the padding rows stepping up.
    """
    i = jnp.arange(x.shape[0])
    last = x[n_valid - 1]
    # padding rows step up by |last| each row
    return jnp.where(i < n_valid, x, last + (i - n_valid + 1) * jnp.abs(last))


def pchip_interval(x: Arr, q: Num) -> Array:
    """
    Find which interval between knots a point falls in.

    Clipped so a point past the last knot uses the last interval.

    :param x: Knot positions.
    :param q: The point.
    :returns: The i with x[i] <= q < x[i + 1].
    """
    n = x.shape[0]
    return jnp.clip(jnp.sum(x <= q) - 1, 0, n - 2)


def pchip_eval(params: tuple[Array, ...], xq: Num) -> Array:
    """
    Evaluate the spline at one point.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param xq: Where to evaluate, one value, vmap it for more.
    :returns: The value of the spline at xq.
    """
    x, y, h, m = params
    i = pchip_interval(x, xq)  # find what range xq falls in
    t = (xq - x[i]) / h[i]
    return (
        y[i] * (1 - 3 * t**2 + 2 * t**3)
        + y[i + 1] * (3 * t**2 - 2 * t**3)
        # hermite cubic, values and slopes at both ends
        + h[i] * m[i] * (t - 2 * t**2 + t**3)
        + h[i] * m[i + 1] * (t**3 - t**2)
    )


def pchip_area(params: tuple[Array, ...], i: Num) -> Array:
    """
    Calculate the integral of the spline over one whole interval.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param i: Which interval, from 0 to n-2.
    :returns: The area under the spline over interval i.
    """
    _x, y, h, m = params
    return h[i] * (y[i] + y[i + 1]) / 2.0 + h[i] ** 2 * (m[i] - m[i + 1]) / 12.0


def pchip_part(params: tuple[Array, ...], i: Num, xq: Num) -> Array:
    """
    Calculate the integral of the spline over part of one interval.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param i: Which interval.
    :param xq: Where to stop, inside interval i.
    :returns: The area under the spline from knot x[i] up to xq.
    """
    x, y, h, m = params
    t = (xq - x[i]) / h[i]
    return h[i] * (
        y[i] * (t - t**3 + t**4 / 2)
        + y[i + 1] * (t**3 - t**4 / 2)
        + h[i] * m[i] * (t**2 / 2 - 2 * t**3 / 3 + t**4 / 4)
        # hermite cubic integrated 0 to t, times the width
        + h[i] * m[i + 1] * (t**4 / 4 - t**3 / 3)
    )


def pchip_integral(params: tuple[Array, ...], a: Num, b: Num) -> Array:
    """
    Integrate the spline between two points.

    Whole intervals in the middle plus the partial bits at each end.

    :param params: Spline params (x, y, h, m) from pchip_build.
    :param a: Lower limit, doesnt need to be on a knot.
    :param b: Upper limit, doesnt need to be on a knot.
    :returns: The integral from a to b.
    """
    x = params[0]
    n = x.shape[0]
    j = pchip_interval(x, a)  # interval a belongs to
    i = pchip_interval(x, b)  # interval b belongs to
    k = jnp.arange(n - 1)
    # whole intervals strictly between
    inside = (k >= j + 1) & (k < i)
    # vmap over all, mask picks j+1 .. i-1
    middle = jnp.sum(jax.vmap(pchip_area, in_axes=(None, 0))(params, k) * inside)
    # interval j minus the bit below a
    a_piece = pchip_area(params, j) - pchip_part(params, j, a)
    b_piece = pchip_part(params, i, b)  # bit of interval i above x_i
    same = j == i
    # same interval would double count
    return jnp.where(
        same,
        pchip_part(params, i, b) - pchip_part(params, j, a),
        a_piece + middle + b_piece,
    )


def loglog_pchip_build(x: Num, y: Num) -> Array:
    """
    Build a pchip spline in log x and log y.

    :param x: Knot positions, positive and increasing.
    :param y: Value at each knot, positive.
    :returns: The spline params (x, y, h, m), in logs.
    """
    return pchip_build(jnp.log(x), jnp.log(y))


def loglog_pchip_eval(params: tuple[Array, ...], xq: Num) -> Array:
    """
    Evaluate a log-log pchip spline.

    :param params: Spline params from loglog_pchip_build.
    :param xq: Where to evaluate, one value or an array.
    :returns: The value of the spline at xq, out of logs.
    """
    # below the first knot, hold the first knots value
    xq = jnp.maximum(xq, jnp.exp(params[0][0]))

    def log_y_at(x: Num) -> Array:
        return pchip_eval(params, jnp.log(x))

    # atleast_1d so a single value works too
    return jnp.exp(jax.vmap(log_y_at)(jnp.atleast_1d(xq)))
