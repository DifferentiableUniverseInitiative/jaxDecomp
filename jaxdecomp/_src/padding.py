from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import jit
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ShapedArray
from jax._src.typing import Array, ArrayLike
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxdecomp._src.spmd_ops import CustomParPrimitive, register_primitive


## Padding Custom SPMD lowering
class SlicePaddingPrimitive(CustomParPrimitive):

  name = "slice_pad"
  multiple_results = False
  impl_static_args = (1, 2, 3)
  outer_pritimive = None

  # Global array implementation is used purely for its abstract eval
  # at jit time, the shape of the global array output is infered from this function
  # it is padding_width * pdims since we pad each slice
  # the pdim is needing for the global array and not the slice
  # This function is lowered, but never executed
  @staticmethod
  def impl(arr: ArrayLike,
           padding_width: int | tuple[int],
           pdims: tuple[int],
           mode: str = 'constant') -> Array:

    assert arr.ndim == 3, "Only 3D arrays are supported"
    assert len(pdims) == 2, "Only 2D pdims are supported"

    # If padding width is an integer then unpad the entire array
    if isinstance(padding_width, int):
      padding_width = ((padding_width, padding_width),) * arr.ndim
    elif isinstance(padding_width, tuple):
      # TODO(wassim) : support single value padding width (low and high are the same)
      # Padding width (if more than one) has to be equal to the number of dimensions
      assert len(padding_width) == arr.ndim
      padding_width = padding_width

    for dim, local_padding in enumerate(padding_width):

      if isinstance(local_padding, int):
        first, last = local_padding, local_padding
      elif isinstance(local_padding, tuple):
        first, last = local_padding
      else:
        raise ValueError(
            "Padding width must be an integer or a tuple of integers")

      if first == 0 and last == 0:
        continue

      match dim:
      # X dimension
        case 0:
          slices = jnp.array_split(arr, pdims[1], axis=0)
          arr = jnp.concatenate([
              jnp.pad(s, ((first, last), (0, 0), (0, 0)), mode=mode)
              for s in slices
          ],
                                axis=0)
        case 1:
          slices = jnp.array_split(arr, pdims[0], axis=1)
          arr = jnp.concatenate([
              jnp.pad(s, ((0, 0), (first, last), (0, 0)), mode=mode)
              for s in slices
          ],
                                axis=1)
        case 2:
          # no distributed padding in the z dimension
          arr = jnp.pad(arr, ((0, 0), (0, 0), (first, last)), mode=mode)
        case _:
          raise ValueError("Only 3D arrays are supported")

    return arr

  # Actual per slice implementation of the primitive
  @staticmethod
  def per_shard_impl(arr: ArrayLike,
                     padding_width: int | tuple[int],
                     mode: str = 'constant') -> Array:
    return jnp.pad(arr, padding_width, mode=mode)

  @staticmethod
  def infer_sharding_from_operands(padding_width: int | tuple[int],
                                   pdims: tuple[int], mode: str, mesh: Mesh,
                                   arg_infos: Tuple[ShapeDtypeStruct],
                                   result_infos: Tuple[ShapedArray]):
    input_sharding = arg_infos[0].sharding
    return NamedSharding(input_sharding.mesh, P(*input_sharding.spec))

  @staticmethod
  def partition(padding_width: int | tuple[int], pdims: tuple[int], mode: str,
                mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):

    # Only one non static input and one output
    input_sharding = NamedSharding(mesh, P(*arg_infos[0].sharding.spec))
    output_sharding = NamedSharding(mesh, P(*result_infos.sharding.spec))
    impl = partial(
        SlicePaddingPrimitive.per_shard_impl,
        padding_width=padding_width,
        mode=mode)
    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(SlicePaddingPrimitive)


@partial(jit, static_argnums=(1, 2, 3))
def slice_pad(x: ArrayLike,
              padding_width: int | tuple[int],
              pdims: tuple[int],
              mode: str = 'constant') -> Array:
  return SlicePaddingPrimitive.outer_lowering(x, padding_width, pdims, mode)


## Unpadding Custom SPMD lowering


class SliceUnPaddingPrimitive(CustomParPrimitive):

  name = "slice_unpad"
  multiple_results = False
  impl_static_args = (1, 2)
  outer_pritimive = None

  # Same as padding, the global array implementation is used purely for its abstract eval
  @staticmethod
  def impl(arr: ArrayLike, padding_width: int | tuple[int],
           pdims: tuple[int]) -> Array:

    # If padding width is an integer then unpad the entire array
    if isinstance(padding_width, int):
      unpadding_width = ((padding_width, padding_width),) * arr.ndim
    elif isinstance(padding_width, tuple):
      # Unpadding width (if more than one) has to be equal to the number of dimensions
      assert len(padding_width) == arr.ndim
      unpadding_width = padding_width

    for dim, local_padding in enumerate(padding_width):
      if isinstance(local_padding, int):
        first, last = local_padding, local_padding
      elif isinstance(local_padding, tuple):
        first, last = local_padding
      else:
        raise ValueError(
            "Padding width must be an integer or a tuple of integers")

      if first == 0 and last == 0:
        continue

      match dim:
      # X dimension
        case 0:
          slices = jnp.array_split(arr, pdims[1], axis=0)
          arr = jnp.concatenate([arr[first:-last] for arr in slices], axis=0)
        case 1:
          slices = jnp.array_split(arr, pdims[0], axis=1)
          arr = jnp.concatenate([arr[:, first:-last] for arr in slices], axis=1)
        case 2:
          # no distributed padding in the z dimension
          arr = arr[:, :, first:-last]
        case _:
          raise ValueError("Only 3D arrays are supported")

    return arr

  # Actual per slice implementation of the primitive
  @staticmethod
  def per_shard_impl(arr: ArrayLike, padding_width: int | tuple[int]) -> Array:
    # If padding width is an integer then unpad the entire array
    if isinstance(padding_width, int):
      unpadding_width = ((padding_width, padding_width),) * arr.ndim
    elif isinstance(padding_width, tuple):
      # Unpadding width (if more than one) has to be equal to the number of dimensions
      assert len(padding_width) == arr.ndim
      unpadding_width = padding_width

    first_x, last_x = unpadding_width[0]
    first_y, last_y = unpadding_width[1]
    first_z, last_z = unpadding_width[2]
    last_x = arr.shape[0] - last_x
    last_y = arr.shape[1] - last_y
    last_z = arr.shape[2] - last_z

    return arr[first_x:last_x, first_y:last_y, first_z:last_z]

  @staticmethod
  def infer_sharding_from_operands(padding_width: int | tuple[int],
                                   pdims: tuple[int], mesh: Mesh,
                                   arg_infos: Tuple[ShapeDtypeStruct],
                                   result_infos: Tuple[ShapedArray]):
    input_sharding = arg_infos[0].sharding
    return NamedSharding(input_sharding.mesh, P(*input_sharding.spec))

  @staticmethod
  def partition(padding_width: int | tuple[int], pdims: tuple[int], mesh: Mesh,
                arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):

    # Only one non static input and one output
    input_sharding = NamedSharding(mesh, P(*arg_infos[0].sharding.spec))
    output_sharding = NamedSharding(mesh, P(*result_infos.sharding.spec))
    impl = partial(
        SliceUnPaddingPrimitive.per_shard_impl, padding_width=padding_width)
    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(SliceUnPaddingPrimitive)


@partial(jit, static_argnums=(1, 2))
def slice_unpad(arr: ArrayLike, padding_width: int | tuple[int],
                pdims: tuple[int]) -> Array:
  return SliceUnPaddingPrimitive.outer_lowering(arr, padding_width, pdims)
