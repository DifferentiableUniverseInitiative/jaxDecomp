from functools import partial

import jax
import pytest

jax.config.update("jax_enable_x64", False)
from itertools import permutations
from math import prod

import jax.numpy as jnp
import numpy as np
from conftest import initialize_distributed
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose, assert_array_equal

import jaxdecomp
from jaxdecomp import (transposeXtoY, transposeYtoX, transposeYtoZ,
                       transposeZtoY)

initialize_distributed()
rank = jax.process_index()
size = jax.process_count()


# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

  assert (len(global_shape) == 3)
  assert (len(pdims) == 2)
  assert (prod(pdims) == size
         ), "The product of pdims must be equal to the number of MPI processes"
  local_array = jnp.arange((global_shape[0] // pdims[1]) *
                           (global_shape[1] // pdims[0]) * global_shape[2])

  local_array = local_array.reshape(global_shape[0] // pdims[1],
                                    global_shape[1] // pdims[0],
                                    global_shape[2])
  local_array = local_array + (100**rank)
  local_array = jnp.array(local_array, dtype=jnp.float32)

  # Remap to the global array from the local slice
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('y', 'z'))
  global_array = multihost_utils.host_local_array_to_global_array(
      local_array, mesh, P('z', 'y'))

  return global_array, mesh


pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100

decomp = [(size, 1), (1, size), pencil_1, pencil_2]
global_shapes = [(4, 8, 16), (4, 4, 4), (29 * size, 19 * size, 17 * size)
                ]  # Cubes, non-cubes and primes


# Cartesian product tests
@pytest.mark.parametrize("pdims",
                         decomp)  # Test with Slab and Pencil decompositions
@pytest.mark.parametrize("global_shape",
                         global_shapes)  # Test cubes, non-cubes and primes
def test_tranpose(pdims, global_shape):
  """ Goes from an array of shape [z,y,x] # What we call an x pencil
    to [x,z,y] # what we call a y pencil
    """
  print("*" * 80)
  print(f"Testing with pdims {pdims} and global shape {global_shape}")

  global_array, mesh = create_spmd_array(global_shape, pdims)

  with mesh:
    jd_tranposed_xy = transposeXtoY(global_array)
    jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
    jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
    jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)

  print(f"jd_tranposed_xy shape {jd_tranposed_xy.shape}")
  print(f"jd_tranposed_yz shape {jd_tranposed_yz.shape}")
  print(f"jd_tranposed_zy shape {jd_tranposed_zy.shape}")
  print(f"jd_tranposed_yx shape {jd_tranposed_yx.shape}")

  if pdims[1] == 1:
    original_sharding = P(None, 'y')
    transposed_sharding = P('y',)
  elif pdims[0] == 1:
    original_sharding = P('z',)
    transposed_sharding = P(None, 'z')
  else:
    original_sharding = P('z', 'y')
    transposed_sharding = P('y', 'z')

  print(f"Original sharding {original_sharding}")
  print(f"Tansposed sharding {transposed_sharding}")

  print(f"JD tranposed yz sharding {jd_tranposed_yz.sharding.spec}")
  print(f"JD tranposed xy sharding {jd_tranposed_xy.sharding.spec}")
  print(f"JD tranposed zy sharding {jd_tranposed_zy.sharding.spec}")
  print(f"JD tranposed yx sharding {jd_tranposed_yx.sharding.spec}")

  assert global_array.sharding.spec == P('z', 'y')
  assert jd_tranposed_xy.sharding.spec == transposed_sharding
  assert jd_tranposed_yz.sharding.spec == original_sharding
  assert jd_tranposed_zy.sharding.spec == transposed_sharding
  assert jd_tranposed_yx.sharding.spec == original_sharding

  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)

  gathered_jd_xy = multihost_utils.process_allgather(
      jd_tranposed_xy, tiled=True)
  gathered_jd_yz = multihost_utils.process_allgather(
      jd_tranposed_yz, tiled=True)
  gathered_jd_zy = multihost_utils.process_allgather(
      jd_tranposed_zy, tiled=True)
  gathered_jd_yx = multihost_utils.process_allgather(
      jd_tranposed_yx, tiled=True)

  # Explanation :
  # Tranposing forward is a shift axis to the right so ZYX to XZY to YXZ (2 0 1)
  # Tranposing backward is a shift axis to the left so YXZ to XZY to ZYX (1 2 0)
  # Double Tranposing from ZYX to YXZ is double (2 0 1) so  (1 2 0)

  forward_tranpose = [2, 0, 1]
  backward_tranpose = [1, 2, 0]
  double_forward = [1, 2, 0]

  #
  # Test X to Y transpose
  # It tranposes ZYX to XZY so from 0 1 2 to 2 0 1
  assert_array_equal(gathered_array.transpose(forward_tranpose), gathered_jd_xy)
  # *********************************************
  # Test Y to Z transpose
  # It tranposes XZY to YXZ so from 0 1 2 to 2 0 1 again
  assert_array_equal(gathered_jd_xy.transpose(forward_tranpose), gathered_jd_yz)
  # and from the global array ZYX to YXZ so from 0 1 2 to 1 2 0
  assert_array_equal(gathered_array.transpose(double_forward), gathered_jd_yz)
  # *********************************************
  # Test Z to Y transpose
  # It tranposes YXZ to XZY so from 0 1 2 to 1 2 0
  assert_array_equal(
      gathered_jd_yz.transpose(backward_tranpose), gathered_jd_zy)
  # The Y pencils should match in forward and backward transposes (despite the inverted grid)
  # assert_array_equal(gathered_jd_zy, gathered_jd_xy)
  # *********************************************
  # Test Y to X transpose
  # It tranposes XZY to ZYX so from 0 1 2 to 1 2 0
  assert_array_equal(
      gathered_jd_zy.transpose(backward_tranpose), gathered_jd_yx)
  # The X pencils should match in forward and backward transposes (original array)
  assert_array_equal(gathered_jd_yx, gathered_array)

  print(f"Pdims {pdims} are ok!")


# Cartesian product tests
@pytest.mark.parametrize("pdims",
                         decomp)  # Test with Slab and Pencil decompositions
@pytest.mark.parametrize("global_shape",
                         global_shapes)  # Test cubes, non-cubes and primes
def test_tranpose_grad(pdims, global_shape):

  global_array, mesh = create_spmd_array(global_shape, pdims)

  @jax.jit
  def jaxdecomp_transpose(global_array):
    jd_tranposed_xy = transposeXtoY(global_array)
    jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
    jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
    jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)
    y = (jd_tranposed_yx * jnp.conjugate(jd_tranposed_yx)).real.sum()
    return y

  @jax.jit
  def jax_transpose(global_array):
    jax_transposed_xy = global_array.transpose([0, 2, 1])
    jax_transposed_yz = jax_transposed_xy.transpose([2, 1, 0])
    jax_transposed_zy = jax_transposed_yz.transpose([2, 1, 0])
    jax_transposed_yx = jax_transposed_zy.transpose([0, 2, 1])
    y = (jax_transposed_yx * jnp.conjugate(jax_transposed_yx)).real.sum()
    return y

  with mesh:
    array_grad = jax.grad(jaxdecomp_transpose)(global_array)
    print("Here is the gradient I'm getting", array_grad.shape)

  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)
  gathered_grads = multihost_utils.process_allgather(array_grad, tiled=True)
  jax_grad = jax.grad(jax_transpose)(gathered_array)

  print(f"Shape of JAX array {jax_grad.shape}")
  # Check the gradients
  assert_allclose(jax_grad, gathered_grads, rtol=1e-5, atol=1e-5)
