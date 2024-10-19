from functools import partial
from typing import List, Tuple

import jax
from jax import lax
from jax import numpy as jnp
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ShapedArray
from jax._src.typing import Array
from jax.lib import xla_client
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.pencil_utils import get_transpose_order
from jaxdecomp._src.spmd_ops import (CustomParPrimitive, get_pencil_type,
                                     register_primitive)

FftType = xla_client.FftType

# TODO Custom partionning for FFTFreq is to be removed
# the only working implementation is fftfreq3d_shard


class FFTFreqPrimitive(CustomParPrimitive):
  name = 'jax_fftfreq'
  multiple_results = True
  impl_static_args: Tuple[int, ...] = ()
  outer_primitive = None

  @staticmethod
  def impl(array, d) -> Tuple[Array, Array, Array]:

    assert array.ndim == 3, "Only 3D FFTFreq is supported"

    kx = jnp.fft.fftfreq(array.shape[0], d=d, dtype=array.dtype)
    ky = jnp.fft.fftfreq(array.shape[1], d=d, dtype=array.dtype)
    kz = jnp.fft.fftfreq(array.shape[2], d=d, dtype=array.dtype)

    assert len(kx) == array.shape[0], "kx must have the same size as array"
    assert len(ky) == array.shape[1], "ky must have the same size as array"
    assert len(kz) == array.shape[2], "kz must have the same size as array"

    kvec = (kx, ky, kz)
    transpose_order = get_transpose_order(FftType.FFT)
    kx, ky, kz = kvec[transpose_order[0]], kvec[transpose_order[1]], kvec[
        transpose_order[2]]

    kx , ky , kz =  (kx.reshape([-1, 1, 1]),
                     ky.reshape([1, -1, 1]),
                     kz.reshape([1, 1, -1])) # yapf: disable

    print(f"IMPL shape of kx {kx.shape} ky {ky.shape} kz {kz.shape}")
    return kx, ky, kz

  @staticmethod
  def per_shard_impl(a: Array, kx: Array, ky: Array, kz: Array,
                     mesh) -> Tuple[Array, Array, Array]:

    x_axis_name, y_axis_name = mesh.axis_names
    assert x_axis_name is not None and y_axis_name is not None
    transpose_order = get_transpose_order(FftType.FFT, mesh)
    kvec = (kx, ky, kz)
    kx = kvec[transpose_order[0]]
    ky = kvec[transpose_order[1]]
    kz = kvec[transpose_order[2]]

    return kx, ky, kz

  @staticmethod
  def infer_sharding_from_operands(mesh: Mesh,
                                   arg_infos: Tuple[ShapeDtypeStruct],
                                   result_infos: Tuple[ShapedArray]):
    del mesh

    input_mesh = arg_infos[0].sharding.mesh
    x_axis_name, y_axis_name = input_mesh.axis_names

    pencil_type = get_pencil_type(input_mesh)
    match pencil_type:
      case _jaxdecomp.SLAB_XY | _jaxdecomp.PENCILS:
        kx_sharding = NamedSharding(input_mesh, P(x_axis_name))
        ky_sharding = NamedSharding(input_mesh, P(None, y_axis_name))
        kz_sharding = NamedSharding(input_mesh, P(None, None, None))
      case _jaxdecomp.SLAB_YZ:
        kx_sharding = NamedSharding(input_mesh, P(y_axis_name))
        ky_sharding = NamedSharding(input_mesh, P(None, x_axis_name))
        kz_sharding = NamedSharding(input_mesh, P(None, None, None))
      case _:
        raise ValueError(f"Unsupported pencil type {pencil_type}")

    return (kx_sharding, ky_sharding, kz_sharding)

  @staticmethod
  def partition(mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):

    # assert isinstance(arg_infos, tuple) and len(
    #     arg_infos) == 2, "Arg info must be a tuple of two sharding"
    # assert isinstance(result_infos, tuple) and len(
    #     result_infos) == 3, "Result info must be a tuple of three sharding"

    print(f"arg_infos size {len(arg_infos)} and {arg_infos}")

    input_sharding = arg_infos[0].sharding
    print(f"args_infos[1] sharding {arg_infos[1].sharding}")
    input_mesh = input_sharding.mesh
    kvec_sharding = (NamedSharding(input_mesh, P(None)),) * 3
    input_shardings = (input_sharding, *kvec_sharding)
    print(f"input_shardings {input_shardings}")
    print(f"len(input_shardings) {len(input_shardings)}")
    print(f"input_mesh {input_mesh}")

    output_sharding = tuple([
        NamedSharding(input_mesh, P(*result_infos[i].sharding.spec))
        for i in range(3)
    ])

    impl = partial(FFTFreqPrimitive.per_shard_impl, mesh=input_mesh)

    return mesh, impl, output_sharding, input_shardings


register_primitive(FFTFreqPrimitive)


@jax.jit
def fftfreq_impl(array, kx, ky, kz) -> Array:
  return FFTFreqPrimitive.outer_lowering(array, kx, ky, kz)


@partial(jax.jit, static_argnums=(1, 2))
def fftfreq3d(array, d=1.0, dtype=None):

  kx = jnp.fft.fftfreq(array.shape[0], d=d, dtype=dtype)
  ky = jnp.fft.fftfreq(array.shape[1], d=d, dtype=dtype)
  kz = jnp.fft.fftfreq(array.shape[2], d=d, dtype=dtype)

  print(f"global shape of kx {kx.shape} ky {ky.shape} kz {kz.shape}")

  return fftfreq_impl(array, kx, ky, kz)


@partial(jax.jit, static_argnums=(1, 2))
def rfftfreq3d(array, d=1.0, dtype=None):

  kx = jnp.fft.fftfreq(array.shape[0], d=d, dtype=dtype)
  ky = jnp.fft.fftfreq(array.shape[1], d=d, dtype=dtype)
  kz = jnp.fft.rfftfreq(array.shape[2], d=d, dtype=dtype)

  return fftfreq_impl(array, kx, ky, kz)


from jax.experimental.shard_alike import shard_alike


@partial(jax.jit, static_argnums=(1, 2))
def fftfreq3d_shard(array, d=1.0, dtype=None):

  kx = jnp.fft.fftfreq(array.shape[0], d=d, dtype=dtype)
  ky = jnp.fft.fftfreq(array.shape[1], d=d, dtype=dtype)
  kz = jnp.fft.fftfreq(array.shape[2], d=d, dtype=dtype)

  kx, _ = shard_alike(kx, array[:, 0, 0])
  ky, _ = shard_alike(ky, array[0, :, 0])
  kz, _ = shard_alike(kz, array[0, 0, :])

  return kx, ky, kz
