from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Manually set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (rank + 1)

from jax.config import config

config.update("jax_enable_x64", True)
from numpy.testing import assert_allclose
import jax
import jax.numpy as jnp
import jaxdecomp
import pytest

# Initialize cuDecomp
jaxdecomp.init()

pdims = (1, size)
global_shape = (29 * size, 19 * size, 17 * size
               )  # These sizes are prime numbers x size of the pmesh
x = jax.random.normal(
    shape=[
        global_shape[0] // pdims[1], global_shape[1] // pdims[0],
        global_shape[2]
    ],
    key=jax.random.PRNGKey(0))
# Local value of the array
array = x + rank
# Global array
global_array = jnp.concatenate([x + i for i in range(size)], axis=0)
# Compute global FFT locally, and account for the transpose
global_karray = jnp.fft.fftn(global_array).transpose([1, 2, 0])
global_karray_slice = global_karray[rank * global_shape[1] //
                                    pdims[1]:(rank + 1) * global_shape[1] //
                                    pdims[1]]


def test_fft():

  # Perform distributed FFT
  karray = jaxdecomp.fft.pfft3d(array, pdims=pdims, global_shape=global_shape)

  # Perform inverse FFT
  rec_array = jaxdecomp.fft.pifft3d(
      karray, pdims=pdims, global_shape=global_shape)

  # Check the forward FFT
  assert_allclose(global_karray_slice.real, karray.real, atol=1e-10)
  assert_allclose(global_karray_slice.imag, karray.imag, atol=1e-10)
  # Check the reverse FFT
  assert_allclose(array, rec_array, rtol=1e-10, atol=1e-10)


def test_jit():
  # Perform distributed FFT
  karray = jax.jit(lambda x: jaxdecomp.fft.pfft3d(
      x, pdims=pdims, global_shape=global_shape))(
          array)

  # Perform inverse FFT
  rec_array = jax.jit(lambda x: jaxdecomp.fft.pifft3d(
      x, pdims=pdims, global_shape=global_shape))(
          karray)

  # Check the forward FFT
  assert_allclose(
      global_karray[rank * global_shape[1] // pdims[1]:(rank + 1) *
                    global_shape[1] // pdims[1]].real,
      karray.real,
      atol=1e-10)
  assert_allclose(
      global_karray[rank * global_shape[1] // pdims[1]:(rank + 1) *
                    global_shape[1] // pdims[1]].imag,
      karray.imag,
      atol=1e-10)
  # Check the reverse FFT
  assert_allclose(array, rec_array, rtol=1e-10, atol=1e-10)


def test_wrong_array_size():
  # Let's create an array that doesnt have the size of the local slice
  array = jnp.ones([16, 16, 16])

  with pytest.raises(AssertionError) as excinfo:
    karray = jaxdecomp.fft.pfft3d(array, pdims=pdims, global_shape=global_shape)

  assert "Only array dimensions divisible" in str(excinfo.value)


def test_grad_fwd():
  from mpi4jax import allreduce

  # Perform distributed FFT
  def fun(arr):
    y = jaxdecomp.fft.pfft3d(arr, pdims=pdims, global_shape=global_shape)
    y = (y * jnp.conjugate(y)).real.sum()
    return allreduce(y, op=MPI.SUM)[0].sum()

  array_grad = jax.grad(fun)(array)
  print("Here is the gradient I'm getting", array_grad.shape)

  # Perform local FFT
  @jax.grad
  def ref_fun(arr):
    y = jnp.fft.fftn(arr).transpose([1, 2, 0])
    return (y * jnp.conjugate(y)).real.sum()

  ref_grad = ref_fun(global_array)
  ref_grad = ref_grad[rank * global_shape[0] // pdims[1]:(rank + 1) *
                      global_shape[0] // pdims[1]]

  # Check the gradients
  assert_allclose(ref_grad, array_grad, rtol=1e-10, atol=1e-10)


def test_grad_bwd():
  from mpi4jax import allreduce

  # Perform distributed FFT
  def fun(arr):
    y = jaxdecomp.fft.pifft3d(arr, pdims=pdims, global_shape=global_shape)
    y = (y * jnp.conjugate(y)).real.sum()
    return allreduce(y, op=MPI.SUM)[0].sum()

  array_grad = jax.grad(fun)(global_karray_slice)
  print("Here is the gradient I'm getting", array_grad.shape)

  # Perform local FFT
  @jax.grad
  def ref_fun(arr):
    y = jnp.fft.ifftn(arr).transpose([2, 0, 1])
    return (y * jnp.conjugate(y)).real.sum()

  ref_grad = ref_fun(global_karray)
  ref_grad = ref_grad[rank * global_shape[0] // pdims[1]:(rank + 1) *
                      global_shape[0] // pdims[1]]

  # Check the gradients
  assert_allclose(ref_grad, array_grad, rtol=1e-10, atol=1e-10)


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
