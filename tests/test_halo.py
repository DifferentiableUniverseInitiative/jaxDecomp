from functools import partial

import jax
import pytest

jax.config.update("jax_enable_x64", True)
from math import prod

import jax.numpy as jnp
from jax._src.distributed import global_state  # This may break in the future
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose, assert_array_equal

import jaxdecomp
from jaxdecomp import slice_pad, slice_unpad

# Initialize jax distributed to instruct jax local process which GPU to use
jax.distributed.initialize()
rank = global_state.process_id
size = global_state.num_processes

# Initialize cuDecomp


# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

  assert (len(global_shape) == 3)
  assert (len(pdims) == 2)
  assert (prod(pdims) == size
         ), "The product of pdims must be equal to the number of MPI processes"

  local_array = jnp.ones(shape=[
      global_shape[0] // pdims[1], global_shape[1] // pdims[0], global_shape[2]
  ])
  # Remap to the global array from the local slice
  devices = mesh_utils.create_device_mesh(pdims[::-1])
  mesh = Mesh(devices, axis_names=('z', 'y'))
  global_array = multihost_utils.host_local_array_to_global_array(
      local_array, mesh, P('z', 'y'))

  return global_array, mesh


pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100

parameters = [((1, size), "X_halo"), ((1, size), "Y_halo"),
              ((1, size), "XY_halo"), ((size, 1), "X_halo"),
              ((size, 1), "Y_halo"), ((size, 1), "XY_halo"),
              (pencil_1, "X_halo"), (pencil_1, "Y_halo"), (pencil_1, "XY_halo"),
              (pencil_2, "X_halo"), (pencil_2, "Y_halo"), (pencil_2, "XY_halo")]


@pytest.mark.parametrize("pdims , halo",
                         parameters)  # Test with Slab and Pencil decompositions
def test_empty_halo(pdims, halo):

  print("*" * 80)
  print(f"Testing with pdims {pdims} with {halo} halo")

  global_shape = (8, 8, 8)
  global_array, mesh = create_spmd_array(global_shape, pdims)

  halo_size = 2

  halo_tuple = (halo_size, halo_size)
  if halo == "XY_halo":
    padding = (halo_tuple, halo_tuple, (0, 0))
    halo_extents = (halo_size, halo_size, 0)
  elif halo == "X_halo":
    padding = (halo_tuple, (0, 0), (0, 0))
    halo_extents = (halo_size, 0, 0)
  elif halo == "Y_halo":
    padding = ((0, 0), halo_tuple, (0, 0))
    halo_extents = (0, halo_size, 0)

  with mesh:
    # perform halo exchange
    padded_array = slice_pad(global_array, padding, pdims)
    exchanged_array = jaxdecomp.halo_exchange(
        padded_array,
        halo_extents=halo_extents,
        halo_periods=(True, True, True))
    # Remove the padding
    stripped_exchanged_array = slice_unpad(exchanged_array, padding, pdims)
    unpadded_array = slice_unpad(padded_array, padding, pdims)

  # Gather array from all processes
  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)
  gathered_stripped_exchanged_array = multihost_utils.process_allgather(
      stripped_exchanged_array, tiled=True)
  gathered_unpadded_array = multihost_utils.process_allgather(
      unpadded_array, tiled=True)

  assert_allclose(
      gathered_array, gathered_unpadded_array, rtol=1e-10, atol=1e-10)
  assert_allclose(
      gathered_array, gathered_stripped_exchanged_array, rtol=1e-10, atol=1e-10)

  print(f"pdims {pdims} with {halo} OK!")


@pytest.mark.parametrize(
    "pdims", [(1, 4)])  # For simplicity, only test Z axis decomposition
def test_full_halo(pdims):

  print("*" * 80)
  print(f"Testing with pdims {pdims}")

  global_shape = (16, 16, 16
                 )  # These sizes are prime numbers x size of the pmesh
  global_array, mesh = create_spmd_array(global_shape, pdims)
  halo_size = 2
  padding = ((halo_size, halo_size), (0, 0), (0, 0))

  # Function that adds one and multiplies by the rank for each shard
  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  def sharded_add_multiply(arr):
    return (arr) * (rank + 1)

  with mesh:
    # perform halo exchange
    padded_array = slice_pad(global_array, padding, pdims)
    updated_array = sharded_add_multiply(padded_array)
    periodic_exchanged_array = jaxdecomp.halo_exchange(
        updated_array,
        halo_extents=(halo_size, 0, 0),
        halo_periods=(True, True, True))
    exchanged_array = jaxdecomp.halo_exchange(
        updated_array,
        halo_extents=(halo_size, 0, 0),
        halo_periods=(False, False, False))
    exchanged_reduced_array = jaxdecomp.halo_exchange(
        updated_array,
        halo_extents=(halo_size, 0, 0),
        halo_periods=(True, True, True),
        reduce_halo=True)
    unpadded_reduced_array = slice_unpad(exchanged_reduced_array, padding,
                                         pdims)
    unpadded_updated_array = slice_unpad(updated_array, padding, pdims)

  # Gather array from all processes
  # gathered_array = multihost_utils.process_allgather(global_array,tiled=True)
  exchanged_gathered_array = multihost_utils.process_allgather(
      exchanged_array, tiled=True)
  periodic_exchanged_gathered_array = multihost_utils.process_allgather(
      periodic_exchanged_array, tiled=True)
  updated_gathered_array = multihost_utils.process_allgather(
      updated_array, tiled=True)
  gathered_exchanged_reduced_array = multihost_utils.process_allgather(
      unpadded_reduced_array, tiled=True)
  gathered_unpadded_updated_array = multihost_utils.process_allgather(
      unpadded_updated_array, tiled=True)
  # Get the slices using array_split
  gathered_array_slices = jnp.array_split(
      exchanged_gathered_array, size, axis=0)
  gathered_updated_slices = jnp.array_split(
      updated_gathered_array, size, axis=0)
  gathered_periodic_exchanged_slices = jnp.array_split(
      periodic_exchanged_gathered_array, size, axis=0)
  gathered_reduced_exchange_slices = jnp.array_split(
      gathered_exchanged_reduced_array, size, axis=0)
  gathered_unpadded_updated_slices = jnp.array_split(
      gathered_unpadded_updated_array, size, axis=0)

  gathered_arrays = zip(gathered_periodic_exchanged_slices, gathered_array_slices\
                      , gathered_reduced_exchange_slices, gathered_updated_slices \
                      , gathered_unpadded_updated_slices)

  for slice_indx, (periodic_exchanged_slice, exchanged_slice, reduced_slice,
                   original_slice,
                   unpadded_slice) in enumerate(gathered_arrays):

    next_indx = slice_indx + 1
    prev_indx = slice_indx - 1
    if slice_indx == len(gathered_array_slices) - 1:
      next_indx = 0
    if slice_indx == 0:
      prev_indx = len(gathered_array_slices) - 1

    next_slice = gathered_array_slices[next_indx]
    prev_slice = gathered_array_slices[prev_indx]

    # The center of the array (no halo) should be the same
    assert_array_equal(exchanged_slice[halo_size:-halo_size],
                       original_slice[halo_size:-halo_size])
    assert_array_equal(periodic_exchanged_slice[halo_size:-halo_size],
                       original_slice[halo_size:-halo_size])
    ## The upper padding should be equal to the lower center of the previous slice
    assert_array_equal(periodic_exchanged_slice[:halo_size],
                       prev_slice[-2 * halo_size:-halo_size])
    ## The lower padding should be equal to the upper center of the next slice
    assert_array_equal(periodic_exchanged_slice[-halo_size:],
                       next_slice[halo_size:2 * halo_size])
    ## First slice, the non periodic exchange upper padding should sum to zero
    if slice_indx == 0:
      assert_array_equal(exchanged_slice[:halo_size].sum(), 0)
    # Last slice, the non periodic exchange lower padding should sum to zero
    elif slice_indx == len(gathered_array_slices) - 1:
      assert_array_equal(exchanged_slice[-halo_size:].sum(), 0)
    # Else, it should be the same as the periodic exchange
    else:
      assert_array_equal(exchanged_slice[:halo_size],
                         periodic_exchanged_slice[:halo_size])
      assert_array_equal(exchanged_slice[-halo_size:],
                         periodic_exchanged_slice[-halo_size:])

    # Test reduced halo

    # Lower center of the previous slice
    previous_halo_extension = prev_slice[-2 * halo_size:-halo_size]
    # Upper center of the next slice
    next_halo_extension = next_slice[halo_size:2 * halo_size]
    # Upper and lower center of the reduced slice
    upper_halo_reduced = reduced_slice[:halo_size]
    lower_halo_reduced = reduced_slice[-halo_size:]
    # Upper and lower center of the original slice (after update no exchange and halo reduction)
    upper_halo_original = unpadded_slice[:halo_size]
    lower_halo_original = unpadded_slice[-halo_size:]

    # Upper slice should be equal to original upper slice + lower center of the previous slice
    assert_array_equal(upper_halo_reduced,
                       (previous_halo_extension + upper_halo_original))
    # Lower slice should be equal to original lower slice + upper center of the next slice
    assert_array_equal(lower_halo_reduced,
                       (next_halo_extension + lower_halo_original))


def test_end():
  # fake test to finalize the MPI processes
  jaxdecomp.finalize()
  jax.distributed.shutdown()
