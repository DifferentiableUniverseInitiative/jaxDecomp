from functools import partial
from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit
from jax._src.typing import Array, ArrayLike
from jax.lib import xla_client

from jaxdecomp._src import pfft as _pfft

Shape = Sequence[int]

__all__ = [
    "pfft3d",
    "pifft3d",
]


def _fft_norm(s: Array, func_name: str, norm: str) -> Array:
  if norm == "backward":
    return 1 / jnp.prod(s) if func_name.startswith("i") else jnp.array(1)
  elif norm == "ortho":
    return (1 / jnp.sqrt(jnp.prod(s)) if func_name.startswith("i") else 1 /
            jnp.sqrt(jnp.prod(s)))
  elif norm == "forward":
    return jnp.prod(s) if func_name.startswith("i") else 1 / jnp.prod(s)**2
  raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                   '"ortho" or "forward".')


# Has to be jitted here because _fft_norm will act on non fully addressable global array
# Which means this should be jit wrapped
@partial(jit, static_argnums=(0, 1, 3, 4))
def _do_pfft(
    func_name: str,
    fft_type: xla_client.FftType,
    arr: ArrayLike,
    norm: Optional[str],
) -> Array:
  # this is not allowed in a multi host setup
  # arr = jnp.asarray(a)
  transformed = _pfft(arr, fft_type)
  transformed *= _fft_norm(
      jnp.array(arr.shape, dtype=transformed.dtype), func_name, norm)
  return transformed


def pfft3d(a: ArrayLike, norm: Optional[str] = "backward") -> Array:
  return _do_pfft("fft", xla_client.FftType.FFT, a, norm=norm)


def pifft3d(a: ArrayLike, norm: Optional[str] = "backward") -> Array:
  return _do_pfft("ifft", xla_client.FftType.IFFT, a, norm=norm)
