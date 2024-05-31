import jax

jax.config.update("jax_enable_x64", True)
from functools import partial
from math import prod

import jax.numpy as jnp
import pytest
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from numpy.testing import assert_allclose

import jaxdecomp

# Initialize cuDecomp
jaxdecomp.init()
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()


# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

  assert (len(global_shape) == 3)
  assert (len(pdims) == 2)
  assert (prod(pdims) == size
         ), "The product of pdims must be equal to the number of MPI processes"

  local_array = jax.random.normal(
      shape=[
          global_shape[0] // pdims[1], global_shape[1] // pdims[0],
          global_shape[2]
      ],
      key=jax.random.PRNGKey(rank))
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
def test_fft(pdims, global_shape):

  print("*" * 80)
  print(f"Testing with pdims {pdims} and global shape {global_shape}")

  global_array, mesh = create_spmd_array(global_shape, pdims)

  # Perform distributed FFT
  with mesh:
    karray = jaxdecomp.fft.pfft3d(global_array)
    # Perform inverse FFT
    rec_array = jaxdecomp.fft.pifft3d(karray)

  # Check the forward FFT
  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)
  gathered_karray = multihost_utils.process_allgather(karray, tiled=True)
  gathered_rec_array = multihost_utils.process_allgather(rec_array, tiled=True)
  jax_karray = jnp.fft.fftn(gathered_array)

  # Check reconstructed array
  assert_allclose(
      gathered_array.real, gathered_rec_array.real, rtol=1e-7, atol=1e-7)
  assert_allclose(
      gathered_array.imag, gathered_rec_array.imag, rtol=1e-7, atol=1e-7)

  # Check the forward FFT
  transpose_back = [2, 0, 1]
  jax_karray_transposed = jax_karray.transpose(transpose_back)
  assert_allclose(
      gathered_karray.real, jax_karray_transposed.real, rtol=1e-7, atol=1e-7)
  assert_allclose(
      gathered_karray.imag, jax_karray_transposed.imag, rtol=1e-7, atol=1e-7)


# Cartesian product tests
@pytest.mark.parametrize("pdims",
                         decomp)  # Test with Slab and Pencil decompositions
@pytest.mark.parametrize("global_shape",
                         global_shapes)  # Test cubes, non-cubes and primes
def test_grad(pdims, global_shape):

  transpose_back = [2, 0, 1]

  print("*" * 80)
  print(f"Testing with pdims {pdims} and global shape {global_shape}")

  global_array, mesh = create_spmd_array(global_shape, pdims)

  print("-" * 40)
  print(f"Testing fwd grad")

  @jax.jit
  def spmd_grad(arr):
    y = jaxdecomp.fft.pfft3d(arr)
    y = (y * jnp.conjugate(y)).real.sum()
    return y

    # Perform local FFT
  @jax.jit
  def local_grad(arr):
    y = jnp.fft.fftn(arr).transpose(transpose_back)
    y = (y * jnp.conjugate(y)).real.sum()
    return y

  with mesh:
    # Perform distributed FFT
    array_grad = jax.grad(spmd_grad)(global_array)
    print("Here is the gradient I'm getting", array_grad.shape)

  gathered_array = multihost_utils.process_allgather(global_array, tiled=True)
  gathered_grads = multihost_utils.process_allgather(array_grad, tiled=True)
  jax_grad = jax.grad(local_grad)(gathered_array)

  print(f"Shape of JAX array {jax_grad.shape}")
  # Check the gradients
  assert_allclose(jax_grad, gathered_grads, rtol=1e-5, atol=1e-5)

  print("-" * 40)
  print(f"Testing backward grad")

  @jax.jit
  def inv_spmd_grad(arr):
    y = jaxdecomp.fft.pifft3d(arr)
    y = (y * jnp.conjugate(y)).real.sum()
    return y

  @jax.jit
  def inv_local_grad(arr):
    y = jnp.fft.ifftn(arr).transpose([2, 0, 1])
    return (y * jnp.conjugate(y)).real.sum()

  with mesh:
    # Perform distributed FFT
    ifft_array_grad = jax.grad(inv_spmd_grad)(global_array)
    print("Here is the gradient I'm getting", array_grad.shape)

  ifft_gathered_grads = multihost_utils.process_allgather(
      ifft_array_grad, tiled=True)
  ifft_jax_grad = jax.grad(inv_local_grad)(gathered_array)

  print(f"Shape of JAX array {ifft_jax_grad.shape}")

  # Check the gradients
  assert_allclose(ifft_jax_grad, ifft_gathered_grads, rtol=1e-5, atol=1e-5)


@pytest.mark.skip(reason="vmap is not yet implemented for the 3D FFT")
def test_vmap():

  global_shape = (29 * size, 19 * size, 17 * size
                 )  # These sizes are prime numbers x size of the pmesh
  pdims = (1, size)

  x = jax.random.normal(
      shape=[
          128, global_shape[0] // pdims[1], global_shape[1] // pdims[0],
          global_shape[2]
      ],
      key=jax.random.PRNGKey(0))
  # Local value of the array
  array = x + rank
  # Global array
  global_array = jnp.concatenate([x + i for i in range(size)], axis=1)
  # Compute global FFT locally, and account for the transpose
  global_karray = jax.vmap(jnp.fft.fftn)(global_array).transpose([0, 2, 3, 1])

  # Perform distributed FFT
  karray = jax.vmap(lambda x: jaxdecomp.fft.pfft3d(
      x, pdims=pdims, global_shape=global_shape))(
          array)

  # Perform inverse FFT
  rec_array = jax.vmap(lambda x: jaxdecomp.fft.pifft3d(
      x, pdims=pdims, global_shape=global_shape))(
          karray)

  # Check the forward FFT
  assert_allclose(
      global_karray[:, rank * global_shape[1] // pdims[1]:(rank + 1) *
                    global_shape[1] // pdims[1]].real,
      karray.real,
      atol=1e-10)
  assert_allclose(
      global_karray[:, rank * global_shape[1] // pdims[1]:(rank + 1) *
                    global_shape[1] // pdims[1]].imag,
      karray.imag,
      atol=1e-10)
  # Check the reverse FFT
  assert_allclose(array, rec_array, rtol=1e-10, atol=1e-10)


# find a way to finalize pytest
def test_end():
  # Make sure that it is cleaned up
  # This has to be this way because pytest runs the "global code" before running the tests
  # There are other solutions https://stackoverflow.com/questions/41871278/pytest-run-a-function-at-the-end-of-the-tests
  # but this require the least amount of work
  jaxdecomp.finalize()
  jax.distributed.shutdown()
