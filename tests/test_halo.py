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

# Initialize jax distributed to instruct jax local process which GPU to use
jax.distributed.initialize()
# Initialize cuDecomp

pdims = (1, size)
global_shape = (29 * size, 19 * size, 17 * size
               )  # These sizes are prime numbers x size of the pmesh
local_array = jax.random.normal(
      shape=[
          global_shape[0] // pdims[0], global_shape[1] // pdims[1],
          global_shape[2]
      ],
      key=jax.random.PRNGKey(rank))
# Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('z', 'y'))
global_array = multihost_utils.host_local_array_to_global_array(
      array, mesh, P('z', 'y'))


def test_empty_halo():
  jitted_pad = jax.jit(jnp.pad)

  padded_array = jitted_pad(x_global, [(32, 32), (32, 32), (32, 32)])

  # perform halo exchange
  exchanged_array = jaxdecomp.halo_exchange(
      padded_array,
      halo_extents=(32, 32, 32),
      halo_periods=(True, True, True),
      pdims=pdims,
      global_shape=global_shape)

  # Remove the padding
  exchanged_array = exchanged_array[32:-32, 32:-32, 32:-32]

  assert_allclose(array, exchanged_array, rtol=1e-10, atol=1e-10)


def test_full_halo():
  padded_array = jnp.pad(
      array, [(global_shape[0] // pdims[1], global_shape[0] // pdims[1]),
              (0, 0), (0, 0)])

  # perform halo exchange
  exchanged_array = jaxdecomp.halo_exchange(
      padded_array,
      halo_extents=(global_shape[0] // pdims[1], 0, 0),
      halo_periods=(True, True, True),
      pdims=pdims,
      global_shape=global_shape)

  # Remove the padding
  upper_slice = exchanged_array[-global_shape[0] // pdims[1]:]

  next_rank = (rank + 1) % size
  # Check that we indeed got the expected slice from adjacent process
  assert_allclose(
      upper_slice,
      global_array[next_rank * global_shape[0] // pdims[1]:(next_rank + 1) *
                   global_shape[0] // pdims[1]],
      rtol=1e-10,
      atol=1e-10)
