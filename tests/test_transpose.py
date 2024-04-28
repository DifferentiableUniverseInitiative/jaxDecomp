import jax
import jax.numpy as jnp
import pytest
from numpy.testing import assert_allclose

import jaxdecomp

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

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


@pytest.mark.skip(reason="transpose functions are still in development")
def test_x_y():
  """ Goes from an array of shape [z,y,x] # What we call an x pencil
    to [x,z,y] # what we call a y pencil
    """
  array_transposed = jaxdecomp.transposeXtoY(
      array, pdims=pdims, global_shape=global_shape)
  global_array_transposed = global_array.transpose([2, 0, 1])

  assert_allclose(
      array_transposed,
      global_array_transposed[:, rank * global_shape[0] // pdims[1]:(rank + 1) *
                              global_shape[0] // pdims[1]],
      rtol=1e-10,
      atol=1e-10)


@pytest.mark.skip(reason="transpose functions are still in development")
def test_y_z():
  """ Goes from an array of shape [x,z,y] # what we call a y pencil
    to [y,x,z] # what we call a z pencil
    """
  # This is to create the Y pencil, tested in the previous call
  array_transposed = jaxdecomp.transposeXtoY(
      array, pdims=pdims, global_shape=global_shape)

  array_transposed = jaxdecomp.transposeYtoZ(
      array_transposed, pdims=pdims, global_shape=global_shape)
  global_array_transposed = global_array.transpose([1, 2, 0])

  assert_allclose(
      array_transposed,
      global_array_transposed[rank * global_shape[1] // pdims[1]:(rank + 1) *
                              global_shape[1] // pdims[1]],
      rtol=1e-10,
      atol=1e-10)


@pytest.mark.skip(reason="transpose functions are still in development")
def test_z_y():
  """ Goes from an array of shape [y,x,z] # what we call a z pencil
    to [x,z,y] # what we call a y pencil
    """
  # This is to create the Y pencil, tested in the previous call
  array_transposed = jaxdecomp.transposeXtoY(
      array, pdims=pdims, global_shape=global_shape)
  array_transposed = jaxdecomp.transposeYtoZ(
      array_transposed, pdims=pdims, global_shape=global_shape)
  array_transposed = jaxdecomp.transposeZtoY(
      array_transposed, pdims=pdims, global_shape=global_shape)

  global_array_transposed = global_array.transpose([2, 0, 1])

  assert_allclose(
      array_transposed,
      global_array_transposed[:, rank * global_shape[0] // pdims[1]:(rank + 1) *
                              global_shape[0] // pdims[1]],
      rtol=1e-10,
      atol=1e-10)


@pytest.mark.skip(reason="transpose functions are still in development")
def test_y_x():
  """ Goes from an array of shape [x,z,y] # what we call a y pencil
    to [z,y,x] # What we call an x pencil
    """
  array_transposed = jaxdecomp.transposeXtoY(
      array, pdims=pdims, global_shape=global_shape)
  array_transposed = jaxdecomp.transposeYtoX(
      array_transposed, pdims=pdims, global_shape=global_shape)
  global_array_transposed = global_array

  assert_allclose(
      array_transposed,
      global_array_transposed[rank * global_shape[0] // pdims[1]:(rank + 1) *
                              global_shape[0] // pdims[1]],
      rtol=1e-10,
      atol=1e-10)
