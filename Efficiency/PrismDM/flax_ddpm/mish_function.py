import jax
from jax._src.typing import Array, ArrayLike
import jax.numpy as jnp

@jax.jit
def mish(x: ArrayLike) -> Array:
    return x * jnp.tanh(jnp.log1p(jnp.exp(x)))
