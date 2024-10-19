from functools import partial
from typing import Optional, Sequence

import jax.numpy as jnp
from jax import jit
from jax._src.numpy.util import promote_dtypes_inexact
from jax._src.typing import Array, ArrayLike
from jax.lib import xla_client

import jaxdecomp
from jaxdecomp._src.cudecomp.fft import pfft as _cudecomp_pfft
from jaxdecomp._src.jax import fftfreq as _fftfreq
from jaxdecomp._src.jax.fft import pfft as _jax_pfft

Shape = Sequence[int]

__all__ = [
    "pfft3d",
    "pifft3d",
]

FftType = xla_client.FftType


def _str_to_fft_type(s: str) -> xla_client.FftType | int:
  """
  Convert a string to an FFT type enum.

  Parameters
  ----------
  s : str
    String representation of FFT type.

  Returns
  -------
  xla_client.FftType
    Corresponding FFT type enum.

  Raises
  ------
  ValueError
    If the string `s` does not match known FFT types.
  """
  if s in ("fft", "FFT"):
    return xla_client.FftType.FFT
  elif s in ("ifft", "IFFT"):
    return xla_client.FftType.IFFT
  elif s in ("rfft", "RFFT"):
    return xla_client.FftType.RFFT
  elif s in ("irfft", "IRFFT"):
    return xla_client.FftType.IRFFT
  else:
    raise ValueError(f"Unknown FFT type '{s}'")


def _fft_norm(s: Array, func_name: str, norm: Optional[str]) -> Array:
  """
  Compute the normalization factor for FFT operations.

  Parameters
  ----------
  s : Array
    Shape of the input array.
  func_name : str
    Name of the FFT function ("fft" or "ifft").
  norm : str
    Type of normalization ("backward", "ortho", or "forward").

  Returns
  -------
  Array
    Normalization factor.

  Raises
  ------
  ValueError
    If an invalid norm value is provided.
  """
  if norm == "backward":
    return 1 / jnp.prod(s) if func_name.startswith("i") else jnp.array(1)
  elif norm == "ortho":
    return (1 / jnp.sqrt(jnp.prod(s)))
  elif norm == "forward":
    return jnp.array(1) if func_name.startswith("i") else 1 / jnp.prod(s)
  raise ValueError(f'Invalid norm value {norm}; should be "backward",'
                   '"ortho" or "forward".')


# Has to be jitted here because _fft_norm will act on non fully addressable global array
# Which means this should be jit wrapped
@partial(jit, static_argnums=(0, 1, 3, 4))
def _do_pfft(func_name: str,
             fft_type: xla_client.FftType,
             arr: Array,
             norm: Optional[str],
             backend: str = "JAX") -> Array:
  """
  Perform 3D FFT or inverse 3D FFT on the input array.

  Parameters
  ----------
  func_name : str
    Name of the FFT function ("fft" or "ifft").
  fft_type : xla_client.FftType
    Type of FFT operation.
  arr : ArrayLike
    Input array to transform.
  norm : Optional[str]
    Type of normalization ("backward", "ortho", or "forward").

  Returns
  -------
  Array
    Transformed array after FFT or inverse FFT.
  """
  if isinstance(fft_type, str):
    typ = _str_to_fft_type(fft_type)
  elif isinstance(fft_type, xla_client.FftType):
    typ = fft_type
  else:
    raise TypeError(f"Unknown FFT type value '{fft_type}'")
  if backend.lower() == "cudecomp":
    transformed = _cudecomp_pfft(arr, typ)
  elif backend.lower() == "jax":
    transformed = _jax_pfft(arr, typ)
  else:
    raise ValueError(f"Unknown backend value '{backend}'")

  transformed *= _fft_norm(
      jnp.array(arr.shape, dtype=transformed.dtype), func_name, norm)
  return transformed


def pfft3d(a: ArrayLike,
           norm: Optional[str] = "backward",
           backend: str = "JAX") -> Array:
  """
  Perform 3D FFT on the input array.

  Parameters
  ----------
  a : ArrayLike
    Input array to transform.
  norm : Optional[str], optional
    Type of normalization ("backward", "ortho", or "forward"), by default "backward".

  Returns
  -------
  Array
    Transformed array after 3D FFT.
  """
  return _do_pfft("fft", xla_client.FftType.FFT, a, norm=norm, backend=backend)


def pifft3d(a: ArrayLike,
            norm: Optional[str] = "backward",
            backend: str = "JAX") -> Array:
  """
  Perform inverse 3D FFT on the input array.

  Parameters
  ----------
  a : ArrayLike
    Input array to transform.
  norm : Optional[str], optional
    Type of normalization ("backward", "ortho", or "forward"), by default "backward".

  Returns
  -------
  Array
    Transformed array after inverse 3D FFT.
  """
  return _do_pfft(
      "ifft", xla_client.FftType.IFFT, a, norm=norm, backend=backend)


def fftfreq3d(array, d=1.0, dtype=None):
  return _fftfreq.fftfreq3d(array, d=d, dtype=dtype)


def rfftfreq3d(array, d=1.0, dtype=None):
  return _fftfreq.rfftfreq3d(array, d=d, dtype=dtype)


def fftfreq3d_shard(array, d=1.0, dtype=None):
  return _fftfreq.fftfreq3d_shard(array, d=d, dtype=dtype)
