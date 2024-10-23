from functools import partial

import jax
from jax import numpy as jnp
from jax.experimental.shard_alike import shard_alike
from jax.lib import xla_client

import jaxdecomp

FftType = xla_client.FftType


@partial(jax.jit, static_argnums=(1,))
def fftfreq3d(k_array, d=1.0):

  # in frequency space, the order is Z pencil
  # X pencil is Z Y X
  # Z pencil is Y X Z

  if jnp.iscomplexobj(k_array):
    dtype = jnp.float32 if k_array.dtype == jnp.complex64 else jnp.float64
  else:
    dtype = k_array.dtype

  ky = jnp.fft.fftfreq(k_array.shape[0], d=d, dtype=dtype) * 2 * jnp.pi
  kx = jnp.fft.fftfreq(k_array.shape[1], d=d, dtype=dtype) * 2 * jnp.pi
  kz = jnp.fft.fftfreq(k_array.shape[2], d=d, dtype=dtype) * 2 * jnp.pi

  ky, _ = shard_alike(ky, k_array[:, 0, 0])
  kx, _ = shard_alike(kx, k_array[0, :, 0])
  kz, _ = shard_alike(kz, k_array[0, 0, :])

  ky = ky.reshape([-1, 1, 1])
  kx = kx.reshape([1, -1, 1])
  kz = kz.reshape([1, 1, -1])

  return kz, ky, kx


@partial(jax.jit, static_argnums=(1,))
def rfftfreq3d(k_array, d=1.0):

  # in frequency space, the order is Z pencil
  # X pencil is Z Y X
  # Z pencil is Y X Z
  # the first axis to be FFT'd is X so it is the real one

  if jnp.iscomplexobj(k_array):
    dtype = jnp.float32 if k_array.dtype == jnp.complex64 else jnp.float64
  else:
    dtype = k_array.dtype

  ky = jnp.fft.fftfreq(k_array.shape[0], d=d, dtype=dtype) * 2 * jnp.pi
  kx = jnp.fft.rfftfreq(k_array.shape[1], d=d, dtype=dtype) * 2 * jnp.pi
  kz = jnp.fft.fftfreq(k_array.shape[2], d=d, dtype=dtype) * 2 * jnp.pi

  ky, _ = shard_alike(ky, k_array[:, 0, 0])
  kx, _ = shard_alike(kx, k_array[0, :, 0])
  kz, _ = shard_alike(kz, k_array[0, 0, :])

  ky = ky.reshape([-1, 1, 1])
  kx = kx.reshape([1, -1, 1])
  kz = kz.reshape([1, 1, -1])

  return kz, ky, kx
