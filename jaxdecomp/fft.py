import jax.numpy as jnp
from jax._src.numpy.fft import _fft_norm
from jaxdecomp._src import pfft as _pfft
from typing import Optional, Sequence, Union, List
from jax._src.typing import Array, ArrayLike
from jax.lib import xla_client

Shape = Sequence[int]

__all__ = [
    "pfft3d",
    "pifft3d",
    "pirfft3d",
    "prfft3d",
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


def _do_pfft(
    func_name: str,
    fft_type: xla_client.FftType,
    a: ArrayLike,
    pdims,
    global_shape,
    norm: Optional[str],
) -> Array:
  arr = jnp.asarray(a)
  transformed = _pfft(arr, fft_type, pdims=pdims, global_shape=global_shape)
  transformed *= _fft_norm(
      jnp.array(global_shape, dtype=transformed.dtype), func_name, norm)
  return transformed


def pfft3d(a: ArrayLike,
           pdims,
           global_shape,
           norm: Optional[str] = "backward") -> Array:
  return _do_pfft(
      "fft",
      xla_client.FftType.FFT,
      a,
      pdims=pdims,
      global_shape=global_shape,
      norm=norm,
  )


def pifft3d(a: ArrayLike,
            pdims,
            global_shape,
            norm: Optional[str] = "backward") -> Array:
  return _do_pfft(
      "ifft",
      xla_client.FftType.IFFT,
      a,
      pdims=pdims,
      global_shape=global_shape,
      norm=norm,
  )


def prfft3d(a: ArrayLike,
            pdims,
            global_shape,
            norm: Optional[str] = "backward") -> Array:
  return _do_pfft(
      "rfft",
      xla_client.FftType.RFFT,
      a,
      pdims=pdims,
      global_shape=global_shape,
      norm=norm,
  )


def pirfft3d(a: ArrayLike,
             pdims,
             global_shape,
             norm: Optional[str] = "backward") -> Array:
  return _do_pfft(
      "irfft",
      xla_client.FftType.IRFFT,
      a,
      pdims=pdims,
      global_shape=global_shape,
      norm=norm,
  )
