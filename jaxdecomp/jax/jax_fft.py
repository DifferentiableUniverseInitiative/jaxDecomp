from jax import lax
from jax import numpy as jnp
from jax._src import mesh as mesh_lib
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.spmd_ops import autoshmap, get_pencil_type


def _fft_slab_xy(operand):
  # pdims are (Py=1,Pz=N)
  # input is (Z / Pz , Y , X) with specs P('z', 'y')
  # First FFT ont XY
  operand = jnp.fft.fft2(operand, axes=(2, 1))
  if jaxdecomp.config.transpose_axis_contiguous:
    # transpose to (Y , X/Pz , Z) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, 'z', 2, 0, tiled=True).transpose([1, 2, 0])
    operand = jnp.fft.fft(operand, axis=-1)
  else:
    # transpose to (Z , Y , X/Pz) with specs P(None, 'y', 'z')
    operand = lax.all_to_all(operand, 'z', 2, 0, tiled=True)
    operand = jnp.fft.fft(operand, axis=0)
  return operand


def _fft_slab_yz(operand):
  # pdims are (Py=N,Pz=1)
  # input is (Z , Y / Py, X) with specs P('z', 'y')
  # First FFT on X
  operand = jnp.fft.fft(operand, axis=-1)
  if jaxdecomp.config.transpose_axis_contiguous:
    # transpose to (X / py, Z , Y) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, 'y', 2, 1, tiled=True).transpose([2, 0, 1])
    # FFT on YZ plane
    operand = jnp.fft.fft2(operand, axes=(2, 1))
  else:
    # transpose to (Z , Y , X / py) with specs P('z', None, 'y')
    operand = lax.all_to_all(operand, 'y', 2, 1, tiled=True)
    # FFT on YZ plane
    operand = jnp.fft.fft2(operand, axes=(1, 0))

  return operand


def _fft_pencils(operand):
  # pdims are (Py=N,Pz=N)
  # input is (Z / Pz , Y / Py  , X) with specs P('z', 'y')
  # First FFT on X
  operand = jnp.fft.fft(operand, axis=-1)
  if jaxdecomp.config.transpose_axis_contiguous:
    # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, 'y', 2, 1, tiled=True).transpose([2, 0, 1])
    # FFT on the Y axis
    operand = jnp.fft.fft(operand, axis=-1)
    # transpose to (Y / pz, X / py, Z) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, 'z', 2, 1, tiled=True).transppose([2, 0, 1])
    # FFT on the Z axis
    operand = jnp.fft.fft(operand, axis=-1)
  else:
    # transpose to (Z / Pz , Y , X / py) with specs P('z', None, 'y')
    operand = lax.all_to_all(operand, 'y', 2, 1, tiled=True)
    # FFT on the Y axis
    operand = jnp.fft.fft(operand, axis=1)
    # transpose to (Z , Y / pz, X / Py) with specs P(None , 'z', 'y')
    operand = lax.all_to_all(operand, 'z', 1, 0, tiled=True)
    # FFT on the Z axis
    operand = jnp.fft.fft(operand, axis=0)
  return operand


def _ifft_slab_xy(operand):
  # pdims are (Py=1,Pz=N)
  if jaxdecomp.config.transpose_axis_contiguous:
    # input is (Y , X/Pz , Z) with specs P('y', 'z')
    # First IFFT on Z
    operand = jnp.fft.ifft(operand, axis=-1)
    # transpose to (Z / Pz , Y , X) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, 'z', 2, 1, tiled=True).transpose([2, 0, 1])
    # IFFT on XY plane
    operand = jnp.fft.ifft2(operand, axes=(2, 1))
  else:
    # input is (Z , Y , X/Pz) with specs P(None, 'y', 'z')
    # First IFFT on Z
    operand = jnp.fft.ifft(operand, axis=0)
    # transpose to (Z/Pz , Y , X) with specs P('z', 'y')
    operand = lax.all_to_all(operand, 'z', 0, 2, tiled=True)
    # IFFT on XY plane
    operand = jnp.fft.ifft2(operand, axes=(2, 1))
  return operand


def _ifft_slab_yz(operand):
  # pdims are (Py=N,Pz=1)
  if jaxdecomp.config.transpose_axis_contiguous:
    # input is (X / py, Z , Y) with specs P('y', 'z')
    # First IFFT
    operand = jnp.fft.ifft2(operand, axes=(2, 1))
    # transpose to (Z , Y / Py, X) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, 'y', 2, 0, tiled=True).transpose([1, 2, 0])
    # IFFT on X axis
    operand = jnp.fft.ifft(operand, axis=-1)
  else:
    # input is (Z , Y , X / py) with specs P('z', None, 'y')
    # First IFFT on Y
    operand = jnp.fft.ifft2(operand, axes=(1, 0))
    # transpose to (Z , Y / py, X) with specs P('z', 'y')
    operand = lax.all_to_all(operand, 'y', 1, 2, tiled=True)
    # IFFT on X axis
    operand = jnp.fft.ifft(operand, axis=-1)

  return operand


def _ifft_pencils(operand):
  # pdims are (Py=N,Pz=N)
  if jaxdecomp.config.transpose_axis_contiguous:
    # input is (Y / pz, X / py, Z) with specs P('z', 'y')
    # First IFFT on Z
    operand = jnp.fft.ifft(operand, axis=-1)
    # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, 'z', 2, 0, tiled=True).transpose([1, 2, 0])
    # IFFT on the Y axis
    operand = jnp.fft.ifft(operand, axis=-1)
    # transpose to (Z / Pz , Y / Py  , X) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, 'y', 2, 0, tiled=True).transpose([1, 2, 0])
    # IFFT on the X axis
    operand = jnp.fft.ifft(operand, axis=-1)
  else:
    # input is (Z / Pz , Y / Py  , X) with specs P('z', 'y')
    # First IFFT on X
    operand = jnp.fft.ifft(operand, axis=0)
    # transpose to (Y / pz, X / py, Z) with specs P('z', 'y')
    operand = lax.all_to_all(operand, 'z', 0, 1, tiled=True)
    # IFFT on the Z axis
    operand = jnp.fft.ifft(operand, axis=0)
    # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
    operand = lax.all_to_all(operand, 'y', 1, 2, tiled=True)
    # IFFT on the Y axis
    operand = jnp.fft.ifft(operand, axis=-1)
  return operand


def jax_pfft3d(operand):
  if jaxdecomp.config.transpose_axis_contiguous:
    match get_pencil_type():
      case _jaxdecomp.NO_DECOMP:
        return jnp.fft.fftn(operand)
      case _jaxdecomp.SLAB_XY:
        return autoshmap(_fft_slab_xy, P('z', 'y'), P('y', 'z'))(operand)
      case _jaxdecomp.SLAB_YZ:
        return autoshmap(_fft_slab_yz, P('z', 'y'), P('y', 'z'))(operand)
      case _jaxdecomp.PENCILS:
        return autoshmap(_fft_pencils, P('z', 'y'), P('z', 'y'))(operand)
  else:
    match get_pencil_type():
      case _jaxdecomp.NO_DECOMP:
        return jnp.fft.fftn(operand)
      case _jaxdecomp.SLAB_XY:
        return autoshmap(_fft_slab_xy, P('z', 'y'), P(None, 'y', 'z'))(operand)
      case _jaxdecomp.SLAB_YZ:
        return autoshmap(_fft_slab_yz, P('z', 'y'), P('z', None, 'y'))(operand)
      case _jaxdecomp.PENCILS:
        return autoshmap(_fft_pencils, P('z', 'y'), P(None, 'z', 'y'))(operand)


def jax_ifft3d(operand):
  if jaxdecomp.config.transpose_axis_contiguous:
    match get_pencil_type():
      case _jaxdecomp.NO_DECOMP:
        return jnp.fft.ifftn(operand)
      case _jaxdecomp.SLAB_XY:
        return autoshmap(_ifft_slab_xy, P('y', 'z'), P('z', 'y'))(operand)
      case _jaxdecomp.SLAB_YZ:
        return autoshmap(_ifft_slab_yz, P('y', 'z'), P('z', 'y'))(operand)
      case _jaxdecomp.PENCILS:
        return autoshmap(_ifft_pencils, P('z', 'y'), P('z', 'y'))(operand)
      case _:
        raise ValueError(f"Unsupported pencil type {get_pencil_type()}")
  else:
    match get_pencil_type():
      case _jaxdecomp.NO_DECOMP:
        return jnp.fft.ifftn(operand)
      case _jaxdecomp.SLAB_XY:
        return autoshmap(_ifft_slab_xy, P(None, 'y', 'z'), P('z', 'y'))(operand)
      case _jaxdecomp.SLAB_YZ:
        return autoshmap(_ifft_slab_yz, P('z', None, 'y'), P('z', 'y'))(operand)
      case _jaxdecomp.PENCILS:
        return autoshmap(_ifft_pencils, P(None, 'z', 'y'), P('z', 'y'))(operand)
      case _:
        raise ValueError(f"Unsupported pencil type {get_pencil_type()}")
