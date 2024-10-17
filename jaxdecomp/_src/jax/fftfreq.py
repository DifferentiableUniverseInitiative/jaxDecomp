from functools import partial
from typing import Tuple

import jax
from jax import lax
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ShapedArray
from jax._src.typing import Array
from jax.lib import xla_client
from jax.numpy.fft import fftfreq as jnp_fftfreq
from jax.numpy.fft import rfftfreq as jnp_rfftfreq
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.pencil_utils import get_transpose_order
from jaxdecomp._src.spmd_ops import (CustomParPrimitive, get_pencil_type,
                                     register_primitive)

FftType = xla_client.FftType


class FFTFreqPrimitive(CustomParPrimitive):
  name = 'jax_fftfreq'
  multiple_results = True
  impl_static_args: Tuple[int, ...] = (0, 1, 2)
  outer_primitive = None

  @staticmethod
  def impl(shape, d, dtype):

    assert isinstance(shape, tuple), "shape must be a tuple"
    assert len(shape) == 3, "Only 3D FFTFreq is supported"

    kz, ky, kx = [jnp_fftfreq(n, d, dtype=dtype) for n in shape]

    return (kz.reshape([-1, 1, 1]),
            ky.reshape([1, -1, 1]),
            kx.reshape([1, 1, -1])) # yapf: disable

  @staticmethod
  def per_shard_impl(shape: Tuple[int, int, int],
                     d: float,
                     dtype,
                     mesh,
                     x_axis_name=None,
                     y_axis_name=None) -> Tuple[Array, Array, Array]:
    assert x_axis_name is not None and y_axis_name is not None

    pencil_type = get_pencil_type()
    match pencil_type:
      case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
        kx_sharding = NamedSharding(mesh, P(x_axis_name))
        ky_sharding = NamedSharding(mesh, P(None, y_axis_name))
        kz_sharding = NamedSharding(mesh, P(None, None, None))
      case _jaxdecomp.SLAB_YZ:
        kx_sharding = NamedSharding(mesh, P(y_axis_name))
        ky_sharding = NamedSharding(mesh, P(None, x_axis_name))
        kz_sharding = NamedSharding(mesh, P(None, None, None))
      case _:
        raise ValueError(f"Unsupported pencil type {pencil_type}")

    transpose_order = get_transpose_order(FftType.FFT)
    fftfreq_shape = tuple(shape[i] for i in transpose_order)
    kx = jnp_fftfreq(fftfreq_shape[0], d, dtype=dtype, device=kx_sharding)
    ky = jnp_fftfreq(fftfreq_shape[1], d, dtype=dtype, device=ky_sharding)
    kz = jnp_fftfreq(fftfreq_shape[2], d, dtype=dtype, device=kz_sharding)

    return kx, ky, kz

  @staticmethod
  def infer_sharding_from_operands(shape, d, dtype, mesh: Mesh,
                                   arg_infos: Tuple[ShapeDtypeStruct],
                                   result_infos: Tuple[ShapedArray]):

    del shape, d, dtype, arg_infos, result_infos

    x_axis_name, y_axis_name = mesh.axis_names

    pencil_type = get_pencil_type()
    match pencil_type:
      case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
        kx_sharding = NamedSharding(mesh, P(x_axis_name))
        ky_sharding = NamedSharding(mesh, P(None, y_axis_name))
        kz_sharding = NamedSharding(mesh, P(None, None, None))
      case _jaxdecomp.SLAB_YZ:
        kx_sharding = NamedSharding(mesh, P(y_axis_name))
        ky_sharding = NamedSharding(mesh, P(None, x_axis_name))
        kz_sharding = NamedSharding(mesh, P(None, None, None))
      case _:
        raise ValueError(f"Unsupported pencil type {pencil_type}")

    return kx_sharding, ky_sharding, kz_sharding

  @staticmethod
  def partition(shape, d, dtype, mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):

    x_axis_name, y_axis_name = mesh.axis_names

    input_sharding = NamedSharding(mesh, P(*arg_infos[0].sharding.spec))
    output_sharding = NamedSharding(mesh, P(*result_infos.sharding.spec))

    impl = partial(
        FFTFreqPrimitive.per_shard_impl,
        shape=shape,
        d=d,
        dtype=dtype,
        mesh=mesh,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name)

    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(FFTFreqPrimitive)


@partial(jax.jit, static_argnums=(0, 1, 2))
def fftfreq_custompar(shape, d=1.0, dtype=None) -> Array:
  return FFTFreqPrimitive.outer_lowering(shape, d, dtype)


def fftfreq_mesh(shape,
                 d=1.0,
                 dtype=None,
                 mesh: Mesh = None) -> Tuple[Array, Array, Array]:
  x_axis_name, y_axis_name = mesh.axis_names

  pencil_type = get_pencil_type(mesh)
  match pencil_type:
    case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
      kx_sharding = NamedSharding(mesh, P(x_axis_name))
      ky_sharding = NamedSharding(mesh, P(y_axis_name))
      kz_sharding = NamedSharding(mesh, P(None))
    case _jaxdecomp.SLAB_YZ:
      kx_sharding = NamedSharding(mesh, P(y_axis_name))
      ky_sharding = NamedSharding(mesh, P(x_axis_name))
      kz_sharding = NamedSharding(mesh, P(None))
    case _:
      raise ValueError(f"Unsupported pencil type {pencil_type}")

  transpose_order = get_transpose_order(FftType.FFT, mesh)
  fftfreq_shape = tuple(shape[i] for i in transpose_order)
  kx = jnp_fftfreq(fftfreq_shape[0], d, dtype=dtype, device=kx_sharding)
  ky = jnp_fftfreq(fftfreq_shape[1], d, dtype=dtype, device=ky_sharding)
  kz = jnp_fftfreq(fftfreq_shape[2], d, dtype=dtype, device=kz_sharding)

  return kx, ky, kz


def fftfreq_impl(shape, d, dtype, reality, mesh: Mesh | None):
  last_axis_fftfreq = jnp_rfftfreq if reality else jnp_fftfreq

  if mesh is None:
    kx = jnp_fftfreq(shape[0], d, dtype=dtype)
    ky = jnp_fftfreq(shape[1], d, dtype=dtype)
    kz = last_axis_fftfreq(shape[2], d, dtype=dtype)
    return kx, ky, kz

  x_axis_name, y_axis_name = mesh.axis_names

  pencil_type = get_pencil_type(mesh)
  match pencil_type:
    case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
      kx_sharding = NamedSharding(mesh, P(x_axis_name))
      ky_sharding = NamedSharding(mesh, P(None, y_axis_name))
      kz_sharding = NamedSharding(mesh, P(None))
    case _jaxdecomp.SLAB_YZ:
      kx_sharding = NamedSharding(mesh, P(y_axis_name))
      ky_sharding = NamedSharding(mesh, P(None, x_axis_name))
      kz_sharding = NamedSharding(mesh, P(None))
    case _:
      raise ValueError(f"Unsupported pencil type {pencil_type}")

  transpose_order = get_transpose_order(FftType.FFT, mesh)
  fftfreq_shape = tuple(shape[i] for i in transpose_order)
  kx = jnp_fftfreq(fftfreq_shape[0], d, dtype=dtype).reshape([-1, 1, 1])
  ky = jnp_fftfreq(fftfreq_shape[1], d, dtype=dtype).reshape([1, -1, 1])
  kz = last_axis_fftfreq(fftfreq_shape[2], d, dtype=dtype).reshape([1, 1, -1])

  kx = lax.with_sharding_constraint(kx, kx_sharding)
  ky = lax.with_sharding_constraint(ky, ky_sharding)
  kz = lax.with_sharding_constraint(kz, kz_sharding)

  return kx, ky, kz


def fftfreq(shape, d=1.0, dtype=None, mesh: Mesh | None = None):
  return fftfreq_impl(shape, d, dtype, reality=False, mesh=mesh)


def rfftfreq(shape, d=1.0, dtype=None, mesh: Mesh | None = None):
  return fftfreq_impl(shape, d, dtype, reality=True, mesh=mesh)
