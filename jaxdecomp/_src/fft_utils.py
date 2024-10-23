from jax import numpy as jnp
from jax.lib import xla_client

FftType = xla_client.FftType
FORWARD_FFTs = {FftType.FFT, FftType.RFFT}
INVERSE_FFTs = {FftType.IFFT, FftType.IRFFT}

from math import prod
from typing import Tuple

from jaxtyping import Array


def ADJOINT(fft_type: FftType) -> FftType:
  match fft_type:
    case FftType.FFT:
      return FftType.IFFT
    case FftType.IFFT:
      return FftType.FFT
    case FftType.RFFT:
      return FftType.IRFFT
    case FftType.IRFFT:
      return FftType.RFFT
    case _:
      raise ValueError(f"Unknown FFT type '{fft_type}'")


def COMPLEX(fft_type: FftType) -> FftType:
  match fft_type:
    case FftType.RFFT | FftType.FFT:
      return FftType.FFT
    case FftType.IRFFT | FftType.IFFT:
      return FftType.IFFT
    case _:
      raise ValueError(f"Unknown FFT type '{fft_type}'")


def _un_normalize_fft(s: Tuple, fft_type: FftType) -> Array:
  if fft_type in FORWARD_FFTs:
    return jnp.array(1)
  else:
    return jnp.array(prod(s))


def fftn(a: Array, fft_type: FftType, adjoint: bool) -> Array:
  if fft_type in FORWARD_FFTs:
    axes = tuple(range(0, 3))
  else:
    axes = tuple(range(2, -1, -1))

  if adjoint:
    fft_type = ADJOINT(fft_type)

  if fft_type == FftType.FFT:
    a = jnp.fft.fftn(a, axes=axes)
  elif fft_type == FftType.IFFT:
    a = jnp.fft.ifftn(a, axes=axes)
  elif fft_type == FftType.RFFT:
    a = jnp.fft.rfftn(a, axes=axes)
  elif fft_type == FftType.IRFFT:
    a = jnp.fft.irfftn(a, axes=axes)
  else:
    raise ValueError(f"Unknown FFT type '{fft_type}'")

  s = a.shape
  a *= _un_normalize_fft(s, fft_type)

  return a


def fft(a: Array, fft_type: FftType, axis: int, adjoint: bool) -> Array:

  if adjoint:
    fft_type = ADJOINT(fft_type)

  if fft_type == FftType.FFT:
    a = jnp.fft.fft(a, axis=axis)
  elif fft_type == FftType.IFFT:
    a = jnp.fft.ifft(a, axis=axis)
  elif fft_type == FftType.RFFT:
    a = jnp.fft.rfft(a, axis=axis)
  elif fft_type == FftType.IRFFT:
    a = jnp.fft.irfft(a, axis=axis)
  else:
    raise ValueError(f"Unknown FFT type '{fft_type}'")

  s = (a.shape[axis],)
  a *= _un_normalize_fft(s, fft_type)

  return a


def fft2(a: Array, fft_type: FftType, axes: Tuple[int, int],
         adjoint: bool) -> Array:

  if adjoint:
    fft_type = ADJOINT(fft_type)

  if fft_type == FftType.FFT:
    a = jnp.fft.fft2(a, axes=axes)
  elif fft_type == FftType.IFFT:
    a = jnp.fft.ifft2(a, axes=axes)
  elif fft_type == FftType.RFFT:
    a = jnp.fft.rfft2(a, axes=axes)
  elif fft_type == FftType.IRFFT:
    a = jnp.fft.irfft2(a, axes=axes)
  else:
    raise ValueError(f"Unknown FFT type '{fft_type}'")

  s = tuple(a.shape[i] for i in axes)
  a *= _un_normalize_fft(s, fft_type)

  return a
