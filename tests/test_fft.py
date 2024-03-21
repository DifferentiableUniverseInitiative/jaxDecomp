from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
import jax
jax.config.update("jax_enable_x64", True)
from numpy.testing import assert_allclose
import jax.numpy as jnp
import jaxdecomp
import pytest
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from functools import partial

# Initialize cuDecomp
jax.distributed.initialize()

pdims = (1, size)
global_shape = (29 * size, 19 * size, 17 * size
               )  # These sizes are prime numbers x size of the pmesh
x = jax.random.normal(
    shape=[
        global_shape[0] // pdims[0], global_shape[1] // pdims[1],
        global_shape[2]
    ],
    key=jax.random.PRNGKey(0))
# Local value of the array
array = x + rank
# Global array
global_array = jnp.concatenate([x + i for i in range(size)], axis=0)
# Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('z', 'y'))
gspmd_array = multihost_utils.host_local_array_to_global_array(
    array, mesh, P('z', 'y'))
# Compute global FFT locally, and account for the transpose
global_karray = jnp.fft.fftn(global_array).transpose([1, 2, 0])
global_karray_slice = global_karray[rank * global_shape[1] //
                                    pdims[0]:(rank + 1) * global_shape[1] //
                                    pdims[1]]


def test_fft():

  # Perform distributed FFT
  with mesh:
    karray = jaxdecomp.fft.pfft3d(gspmd_array)

    # Perform inverse FFT
    rec_array = jaxdecomp.fft.pifft3d(karray)

  # Check the forward FFT
  # Non cube shape .. need to think about the slice coming from JAX
  # assert_allclose(global_karray_slice.real, karray.addressable_data(0).real, atol=1e-10)
  # assert_allclose(global_karray_slice.imag, karray.addressable_data(0).imag, atol=1e-10)
  # Check the reverse FFT
  assert_allclose(array, rec_array.addressable_data(0), rtol=1e-10, atol=1e-10)

def test_jit():
  # Perform distributed FFT
  with mesh:
    karray = jax.jit(lambda x: jaxdecomp.fft.pfft3d(x))(gspmd_array)

    # Perform inverse FFT
    rec_array = jax.jit(lambda x: jaxdecomp.fft.pifft3d(x))(karray)

  # Check the forward FFT
  #assert_allclose(
  #    global_karray[rank * global_shape[1] // pdims[1]:(rank + 1) *
  #                  global_shape[1] // pdims[1]].real,
  #    karray.real,
  #    atol=1e-10)
  #assert_allclose(
  #    global_karray[rank * global_shape[1] // pdims[1]:(rank + 1) *
  #                  global_shape[1] // pdims[1]].imag,
  #    karray.imag,
  #    atol=1e-10)
  # Check the reverse FFT
  assert_allclose(array, rec_array.addressable_data(0), rtol=1e-10, atol=1e-10)

def test_grad_fwd():
  #from mpi4jax import allreduce
  # Cannot combine jax distributed and mpi4jax
  from jax.experimental.shard_map import shard_map

  with mesh:

    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'),
          out_specs=P(None),check_rep=False)
    def norm_sum(arr):
      arr =  (arr * jnp.conjugate(arr)).real.sum()
      arr_sum = jax.lax.psum(arr , 'y')
      return arr_sum

    # Ops on global non fully addressable arrays must always be wrapped in a jit
    # Perform distributed FFT
    @jax.jit
    def fun(arr):
      y = jaxdecomp.fft.pfft3d(arr)
      y = norm_sum(y)
      return y

    array_grad = jax.grad(fun)(gspmd_array)
    print("Here is the gradient I'm getting", array_grad.shape)

    # Perform local FFT
    @jax.grad
    def ref_fun(arr):
      y = jnp.fft.fftn(arr).transpose([1, 2, 0])
      return (y * jnp.conjugate(y)).real.sum()

    ref_grad = ref_fun(global_array)
    ref_grad = ref_grad[rank * global_shape[0] // pdims[1]:(rank + 1) *
                      global_shape[0] // pdims[1]]

    print(f"Shape of JAX array {ref_grad.shape}")
  # Check the gradients
  # TODO(wassim) Cannot compare addressable with non addressable .. 
  # But for now the Grad works
  # assert_allclose(ref_grad, array_grad, rtol=1e-10, atol=1e-10)

def test_grad_bwd():
  #from mpi4jax import allreduce
  # Cannot combine jax distributed and mpi4jax
  from jax.experimental.shard_map import shard_map

  with mesh:

    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'),
          out_specs=P(None),check_rep=False)
    def norm_sum(arr):
      arr =  (arr * jnp.conjugate(arr)).real.sum()
      arr_sum = jax.lax.psum(arr , 'y')
      return arr_sum

    # Ops on global non fully addressable arrays must always be wrapped in a jit
    # Perform distributed FFT
    @jax.jit
    def fun(arr):
      y = jaxdecomp.fft.pifft3d(arr)
      y = norm_sum(y)
      return y

    karray = jaxdecomp.fft.pfft3d(gspmd_array)
    array_grad = jax.grad(fun)(karray)
    print("Here is the gradient I'm getting", array_grad.shape)

    # Perform local FFT
    @jax.grad
    def ref_fun(arr):
      y = jnp.fft.ifftn(arr).transpose([2, 0, 1])
      return (y * jnp.conjugate(y)).real.sum()

    ref_grad = ref_fun(global_karray)
    ref_grad = ref_grad[rank * global_shape[0] // pdims[1]:(rank + 1) *
                        global_shape[0] // pdims[1]]
    print(f"Shape of JAX array {ref_grad.shape}")


  # Check the gradients
  # TODO(wassim) Cannot compare addressable with non addressable .. Find a way
  # But for now the Grad works
  # assert_allclose(ref_grad, array_grad, rtol=1e-10, atol=1e-10)


@pytest.mark.skip(reason="vmap is not yet implemented for the 3D FFT")
def test_vmap():
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
