from conftest import (compare_sharding, create_ones_spmd_array,
                      create_spmd_array, device_arange, initialize_distributed,
                      is_on_cluster)

initialize_distributed()
import jax

size = jax.device_count()
from functools import partial
from math import prod

import jax.numpy as jnp
import pytest
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose, assert_array_equal

import jaxdecomp
from jaxdecomp import slice_pad, slice_unpad
from jaxdecomp._src.spmd_ops import get_pencil_type

pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100
pdims = [(1, size), (size, 1), pencil_1, pencil_2]


@pytest.mark.skip(reason="Need to fix the test")
@pytest.mark.parametrize(
    "pdims", [(1, 4)])  # For simplicity, only test Z axis decomposition
def test_full_halo(pdims):

  print("*" * 80)
  print(f"Testing with pdims {pdims}")

  global_shape = (16, 16, 16)
  # These sizes are prime numbers x size of the pmesh
  global_array, mesh = create_ones_spmd_array(global_shape, pdims)
  halo_size = 2
  devices = device_arange(pdims)
  # Function that adds one and multiplies by the rank for each shard
  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  def sharded_add_multiply(arr, devices):
    return (arr) + (devices + 10)

  if pdims[0] == 1:
    padding = ((halo_size, halo_size), (0, 0), (0, 0))
  elif pdims[1] == 1:
    padding = ((0, 0), (halo_size, halo_size), (0, 0))
  else:
    padding = ((halo_size, halo_size), (halo_size, halo_size), (0, 0))

  with mesh:
    # perform halo exchange
    padded_array = slice_pad(global_array, padding, pdims)
    updated_array = sharded_add_multiply(padded_array, devices)
    periodic_exchanged_array = jaxdecomp.halo_exchange(
        updated_array, halo_extent=halo_size, periodic=True)

  original_gathered = process_allgather(global_array, tiled=True)
  updated_gathered = process_allgather(updated_array, tiled=True)
  periodic_gathered = process_allgather(periodic_exchanged_array, tiled=True)

  print(f"original array \n{original_gathered[:, :, 0]}")
  print(f"updated array \n{updated_gathered[:, :, 0]}")
  print(f"periodic exchanged array \n{periodic_gathered[:, :, 0]}")


@pytest.mark.parametrize("pdims",
                         pdims)  # Test with Slab and Pencil decompositions
def test_ppermute_halo(pdims):

  jnp.set_printoptions(linewidth=200)

  print("*" * 80)
  print(f"Testing with pdims {pdims}")

  global_shape = (16, 16, 16)
  global_array, mesh = create_spmd_array(global_shape, pdims)

  # @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  # def sharded_add_multiply(arr, devices):
  #   return (arr) + (devices + 10)

  halo_size = 2

  halo_x = (halo_size, halo_size) if pdims[1] > 1 else (0, 0)
  halo_y = (halo_size, halo_size) if pdims[0] > 1 else (0, 0)
  halo_extents = (halo_x[0], halo_y[0], 0)
  padding = (halo_x, halo_y, (0, 0))

  @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
  def pad(arr):
    return jnp.pad(arr, padding, mode='linear_ramp', end_values=20)

  with mesh:
    # perform halo exchange
    updated_array = pad(global_array)
    jax_exchanged = jaxdecomp.halo_exchange(
        updated_array, halo_extent=halo_size, periodic=True, backend="JAX")
    cudecomp_exchanged = jaxdecomp.halo_exchange(
        updated_array, halo_extent=halo_size, periodic=True, backend="CUDECOMP")

  g_array = process_allgather(updated_array, tiled=True)
  g_jax_exchanged = process_allgather(jax_exchanged, tiled=True)
  g_cudecomp_exchanged = process_allgather(cudecomp_exchanged, tiled=True)
  print(f"Original \n{g_array[:,:,0]}")
  print(f"exchanged cudecomp \n{g_jax_exchanged[:,:,0]}")
  print(f"exchanged jax \n{g_cudecomp_exchanged[:,:,0]}")

  assert_array_equal(g_jax_exchanged, g_cudecomp_exchanged)
