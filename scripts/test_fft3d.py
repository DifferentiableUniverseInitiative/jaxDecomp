from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import jax
import jax.numpy as jnp
import jaxdecomp
import time
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

#jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_P2P_PL)
pdims = (2, 2)
global_shape = (1024, 1024, 1024)

# Initialize an array with the expected gobal size
array = jax.random.normal(
    shape=[
        global_shape[0] // pdims[1], global_shape[1] // pdims[0],
        global_shape[2]
    ],
    key=jax.random.PRNGKey(0)) + rank

# Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims[::-1])
mesh = Mesh(devices, axis_names=('z', 'y'))
global_array = multihost_utils.host_local_array_to_global_array(
    array, mesh, P('z', 'y'))

@jax.jit
def do_fft(x):
  return jaxdecomp.fft.pfft3d(global_array)


do_fft(array)
before = time.time()
karray = do_fft(array).block_until_ready()
after = time.time()
print(rank, 'took', after - before, 's')

# And now, let's do the inverse FFT
rec_array = jaxdecomp.fft.pifft3d(
    karray)

if rank == 0:
  # Let's test if things are like we expect
  diff = rec_array - array
  print('maximum reconstruction difference', jnp.abs(diff).max())

jaxdecomp.finalize()
