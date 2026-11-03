import jax.numpy as jnp
from flax.linen.linear import Array
from typing import Any, Callable, Dict, Optional, Tuple, Union

def transform_x(x, x_min=-14, x_max=0):
    return 2 * (x - x_min) / (x_max - x_min) - 1

def chexp_CDM(x):
    x = transform_x(x)
    x2 = x*x
    x3 = x2*x
    x4 = x3*x
    x5 = x4*x
    x6 = x5*x
    x7 = x6*x
    t0 = 1
    t1 = x
    t2 = 2*x2 -1
    t3 = 4*x3 -3*x
    t4 = 8*x4 -8*x2 +1
    t5 = 16*x5 - 20*x3 +5*x
    t6 = 32*x6 - 48*x4 +18*x2 -1
    t7 = 64*x7 -112*x5 +56*x3 -7*x
    #ex= 0.14021878*t0 + 0.27541278*t1+ 0.22122865*t2 + 0.14934221*t3 + 0.0907736*t4 + 0.04369614*t5 + 0.02087868*t6 + 0.00996535*t7 original CipherDM
    ex= 0.15373774*t0 + 0.28457847*t1+ 0.22616736*t2 + 0.15533998*t3 + 0.09301880*t4 + 0.04903278*t5 + 0.02297197*t6 + 0.00965226*t7
    return ex

def approximate_softmax_CDM(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = chexp_CDM(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor

def approximate_silu_CDM(x: Array) -> Array:
    b0 = x < -6.0
    b1 = x < -2.0
    b2 = x > 6.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    a_coeffs = jnp.array(
        [-0.52212664, -0.16910363, -0.01420163]
    )
    b_coeffs = jnp.array(
        [
            0.03453821,
            0.49379432,
            0.19784596,
            -0.00602401,
            0.00008032,
        ]
    )
    x2 = jnp.square(x)
    x4 = jnp.square(x2)
    x6 = x2 * x4
    seg1 = a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[4] * x6
        + b_coeffs[3] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret

def approximate_mish_CDM(x: Array) -> Array:
    b0 = x < -6.0
    b1 = x < -2.0
    b2 = x > 6.0
    b3 = b1 ^ b2 ^ True  # x in [-4.0, 4.0)
    b4 = b0 ^ b1  # x in [-8.0, -4.0)
    a_coeffs = jnp.array(
        [-0.55684445, -0.18375535, -0.01572019]
    )
    b_coeffs = jnp.array(
        [
            0.07559242,
            0.54902050,
            0.20152583,
            -0.00735309,
            0.00010786,
        ]
    )
    x2 = jnp.square(x)
    x4 = jnp.square(x2)
    x6 = x2 * x4
    seg1 = a_coeffs[2] * x2 + a_coeffs[1] * x + a_coeffs[0]
    seg2 = (
        b_coeffs[4] * x6
        + b_coeffs[3] * x4
        + b_coeffs[2] * x2
        + b_coeffs[1] * x
        + b_coeffs[0]
    )
    ret = b2 * x + b4 * seg1 + b3 * seg2
    return ret

def chexp_CTA(x):
    x2 = x*x
    x3 = x2*x

    # Supplementary material, Eq. (2): split at -3.07.
    judge = x >= -3.07

    ex1=0.00014351*x3+0.00426741*x2+0.04082211*x+0.12514002
    ex2=0.04096367*x3+0.31576001*x2+0.89263596*x+0.98513184

    return (1-judge)*ex1+judge*ex2

def approximate_softmax_CTA(x: Array,
            axis: Optional[Union[int, Tuple[int, ...]]] = -1,
            where: Optional[Array] = None,
            initial: Optional[Array] = None) -> Array:

    x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
    x = x - x_max

    # exp on large negative is clipped to zero
    b = x > -14
    nexp = chexp_CTA(x) * b

    divisor = jnp.sum(nexp, axis, where=where, keepdims=True)

    return nexp / divisor

def approximate_silu_CTA(x: Array) -> Array:
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

def approximate_mish_CTA(x: Array) -> Array:
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
