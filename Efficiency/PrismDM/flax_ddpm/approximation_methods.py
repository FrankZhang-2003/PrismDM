import jax.numpy as jnp
from flax.linen.linear import Array
from typing import Any, Callable, Dict, Optional, Tuple, Union

def transform_x(x, x_min=-14, x_max=0):
    return 2 * (x - x_min) / (x_max - x_min) - 1

def chexp_PDM(x):
    x2 = x*x
    x3 = x2*x

    # Supplementary material, Eq. (2): split at -3.07.
    judge = x >= -3.07

    ex1=0.00014351*x3+0.00426741*x2+0.04082211*x+0.12514002
    ex2=0.04096367*x3+0.31576001*x2+0.89263596*x+0.98513184

    return (1-judge)*ex1+judge*ex2

def approximate_softmax_PDM(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = chexp_PDM(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor

def approximate_silu_PDM(x: Array) -> Array:
    b0 = x < -6.0
    b1 = x < -1.86
    b2 = x > 6.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    a_coeffs = jnp.array(
        [-0.32222088, -0.05728559]
    )
    b_coeffs = jnp.array(
        [
            0.04439874,
            0.54418372,
            0.18299721,
            -0.01862711,
        ]
    )
    x2 = jnp.square(x)
    x3 = x2 * x
    seg1 = a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[3] * x3
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret

def approximate_mish_PDM(x: Array) -> Array:
    b0 = x < -6.0
    b1 = x < -2.22
    b2 = x > 6.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    a_coeffs = jnp.array(
        [-0.30983219, -0.05448525]
    )
    b_coeffs = jnp.array(
        [
            0.09984283,
            0.60460080,
            0.17227525,
            -0.01895927,
        ]
    )
    x2 = jnp.square(x)
    x3 = x2 * x
    seg1 = a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[3] * x3
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret
