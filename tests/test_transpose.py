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
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose, assert_array_equal

import jaxdecomp
from jaxdecomp import (transposeXtoY, transposeYtoX, transposeYtoZ,
                       transposeZtoY)
from jaxdecomp._src.spmd_ops import get_pdims_from_sharding
from jaxdecomp.jax import (jax_transpose_XtoY, jax_transpose_XtoZ,
                           jax_transpose_YtoX, jax_transpose_YtoZ,
                           jax_transpose_ZtoX, jax_transpose_ZtoY)

initialize_distributed()
rank = jax.process_index()
size = jax.process_count()


def compare_sharding(sharding1, sharding2):
  pdims1 = get_pdims_from_sharding(sharding1)
  pdims2 = get_pdims_from_sharding(sharding2)
  pdims1 = pdims1 + (1,) * (3 - len(pdims1))
  pdims2 = pdims2 + (1,) * (3 - len(pdims2))
  return pdims1 == pdims2


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
  mesh = Mesh(devices.T, axis_names=('z', 'y'))
  global_array = multihost_utils.host_local_array_to_global_array(
      local_array, mesh, P('z', 'y'))

  return global_array, mesh


pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100

decomp = [(size, 1), (1, size), pencil_1, pencil_2]
global_shapes = [(4, 8, 16), (4, 4, 4), (29 * size, 19 * size, 17 * size)
                ]  # Cubes, non-cubes and primes
local_transpose = [False, True]


# Cartesian product tests
@pytest.mark.parametrize("pdims",
                         decomp)  # Test with Slab and Pencil decompositions
@pytest.mark.parametrize("global_shape",
                         global_shapes)  # Test cubes, non-cubes and primes
@pytest.mark.parametrize("local_transpose", local_transpose)
def test_tranpose(pdims, global_shape, local_transpose):
  """ Goes from an array of shape [z,y,x] # What we call an x pencil
    to [x,z,y] # what we call a y pencil
    """
  print("*" * 80)
  print(
      f"Testing with pdims {pdims} and global shape {global_shape} with local transpose {local_transpose}"
  )
  jaxdecomp.config.update("transpose_axis_contiguous", local_transpose)
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

  if local_transpose:
    original_sharding = NamedSharding(mesh, P('z', 'y'))
    y_pencil_sharding = NamedSharding(mesh, P('y', 'z'))
    z_pencil_sharding = NamedSharding(mesh, P('z', 'y'))
  else:
    original_sharding = NamedSharding(mesh, P('z', 'y'))
    y_pencil_sharding = NamedSharding(mesh, P('z', None, 'y'))
    z_pencil_sharding = NamedSharding(mesh, P(None, 'z', 'y'))

  print(f"Original sharding {original_sharding}")
  print(f"y pencil sharding {y_pencil_sharding}")
  print(f"z pencil sharding {z_pencil_sharding}")

  print(f"JD tranposed xy sharding {jd_tranposed_xy.sharding.spec}")
  print(f"JD tranposed yz sharding {jd_tranposed_yz.sharding.spec}")
  print(f"JD tranposed zy sharding {jd_tranposed_zy.sharding.spec}")
  print(f"JD tranposed yx sharding {jd_tranposed_yx.sharding.spec}")

  assert compare_sharding(global_array.sharding, original_sharding)
  assert compare_sharding(jd_tranposed_xy.sharding, y_pencil_sharding)
  assert compare_sharding(jd_tranposed_yz.sharding, z_pencil_sharding)
  assert compare_sharding(jd_tranposed_zy.sharding, y_pencil_sharding)
  assert compare_sharding(jd_tranposed_yx.sharding, original_sharding)

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

  if local_transpose:
    forward_tranpose = [2, 0, 1]
    backward_tranpose = [1, 2, 0]
    double_forward = [1, 2, 0]
  else:
    forward_tranpose = [0, 1, 2]
    backward_tranpose = [0, 1, 2]
    double_forward = [0, 1, 2]

  print(
      f"For local_transpose {local_transpose} forward_tranpose {forward_tranpose} backward_tranpose {backward_tranpose}"
  )
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

  print(f"Pdims {pdims} with local_transpose {local_transpose} is ok!!")


# Cartesian product tests
@pytest.mark.parametrize("pdims",
                         decomp)  # Test with Slab and Pencil decompositions
@pytest.mark.parametrize("global_shape",
                         global_shapes)  # Test cubes, non-cubes and primes
@pytest.mark.parametrize("local_transpose", local_transpose)
def test_tranpose_grad(pdims, global_shape, local_transpose):

  jaxdecomp.config.update("transpose_axis_contiguous", local_transpose)

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


@pytest.mark.parametrize("pdims",
                         decomp)  # Test with Slab and Pencil decompositions
@pytest.mark.parametrize("global_shape",
                         global_shapes)  # Test cubes, non-cubes and primes
@pytest.mark.parametrize("local_transpose", local_transpose)
def test_jax_transposes(pdims, global_shape, local_transpose):

  jaxdecomp.config.update("transpose_axis_contiguous", local_transpose)
  global_array, mesh = create_spmd_array(global_shape, pdims)
  sharding = global_array.sharding
  print(f"Sharding of the global array {sharding.spec}")

  with mesh:

    jax_transposed_xy = jax_transpose_XtoY(global_array)
    jax_transposed_yz = jax_transpose_YtoZ(jax_transposed_xy)
    jax_transposed_zy = jax_transpose_ZtoY(jax_transposed_yz)
    jax_transposed_yx = jax_transpose_YtoX(jax_transposed_zy)
    jax_transposed_xz = jax_transpose_XtoZ(global_array)
    jax_transposed_zx = jax_transpose_ZtoX(jax_transposed_xz)

  with mesh:
    jd_tranposed_xy = transposeXtoY(global_array)
    jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy)
    jd_tranposed_zy = transposeZtoY(jd_tranposed_yz)
    jd_tranposed_yx = transposeYtoX(jd_tranposed_zy)

  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)
  # gather JAX transposed
  g_jax_transposed_xy = multihost_utils.process_allgather(
      jax_transposed_xy, tiled=True)
  g_jax_transposed_yz = multihost_utils.process_allgather(
      jax_transposed_yz, tiled=True)
  g_jax_transposed_zy = multihost_utils.process_allgather(
      jax_transposed_zy, tiled=True)
  g_jax_transposed_yx = multihost_utils.process_allgather(
      jax_transposed_yx, tiled=True)
  g_jax_transposed_xz = multihost_utils.process_allgather(
      jax_transposed_xz, tiled=True)
  g_jax_transposed_zx = multihost_utils.process_allgather(
      jax_transposed_zx, tiled=True)
  # gather JD transposed
  jd_gathered_xy = multihost_utils.process_allgather(
      jd_tranposed_xy, tiled=True)
  jd_gathered_yz = multihost_utils.process_allgather(
      jd_tranposed_yz, tiled=True)
  jd_gathered_zy = multihost_utils.process_allgather(
      jd_tranposed_zy, tiled=True)
  jd_gathered_yx = multihost_utils.process_allgather(
      jd_tranposed_yx, tiled=True)

  assert compare_sharding(jax_transposed_xy.sharding, jd_tranposed_xy.sharding)
  assert compare_sharding(jax_transposed_yz.sharding, jd_tranposed_yz.sharding)
  assert compare_sharding(jax_transposed_zy.sharding, jd_tranposed_zy.sharding)
  assert compare_sharding(jax_transposed_yx.sharding, jd_tranposed_yx.sharding)

  assert_array_equal(gathered_array, jd_gathered_xy)
  assert_array_equal(gathered_array, jd_gathered_yz)
  assert_array_equal(gathered_array, jd_gathered_zy)
  assert_array_equal(gathered_array, jd_gathered_yx)

  if local_transpose:
    forward_tranpose = [2, 0, 1]
    backward_tranpose = [1, 2, 0]
    double_forward = [1, 2, 0]
  else:
    forward_tranpose = [0, 1, 2]
    backward_tranpose = [0, 1, 2]
    double_forward = [0, 1, 2]

  # Test X to Y transpose
  # It tranposes ZYX to XZY so from 0 1 2 to 2 0 1
  assert_array_equal(
      gathered_array.transpose(forward_tranpose), g_jax_transposed_xy)
  # *********************************************
  # Test Y to Z transpose
  # It tranposes XZY to YXZ so from 0 1 2 to 2 0 1 again
  assert_array_equal(
      g_jax_transposed_xy.transpose(forward_tranpose), g_jax_transposed_yz)
  # and from the global array ZYX to YXZ so from 0 1 2 to 1 2 0
  assert_array_equal(
      gathered_array.transpose(double_forward), g_jax_transposed_yz)
  # *********************************************
  # Test Z to Y transpose
  # It tranposes YXZ to XZY so from 0 1 2 to 1 2 0
  assert_array_equal(
      g_jax_transposed_yz.transpose(backward_tranpose), g_jax_transposed_zy)
  # The Y pencils should match in forward and backward transposes (despite the inverted grid)
  # assert_array_equal(gathered_jd_zy, gathered_jd_xy)
  # *********************************************
  # Test Y to X transpose
  # It tranposes XZY to ZYX so from 0 1 2 to 1 2 0
  assert_array_equal(
      g_jax_transposed_zy.transpose(backward_tranpose), g_jax_transposed_yx)
  # The X pencils should match in forward and backward transposes (original array)
  assert_array_equal(g_jax_transposed_yx, gathered_array)

  # Testing XZ and ZX transposes
  assert_array_equal(
      gathered_array.transpose(backward_tranpose), g_jax_transposed_xz)
  assert_array_equal(
      g_jax_transposed_xz.transpose(forward_tranpose), g_jax_transposed_zx)
  assert_array_equal(gathered_array, g_jax_transposed_zx)

  print(f"""
      JAX transposes {"contiguous" if local_transpose else "non-contiguous"}
      for pdims {pdims} with global shape {global_shape} are ok!!
      """)
