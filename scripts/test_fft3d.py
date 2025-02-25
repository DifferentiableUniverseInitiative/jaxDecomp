import time

import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

import jaxdecomp

# Initialize jax distributed to instruct jax local process which GPU to use
jaxdecomp.init()
jax.distributed.initialize()
rank = jax.process_index()

# jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_P2P_PL)
pdims = (2, 2)
global_shape = (1024, 1024, 1024)

# Initialize a local slice of the global array
array = jax.random.normal(
    shape=[global_shape[0] // pdims[1], global_shape[1] // pdims[0], global_shape[2]],
    key=jax.random.PRNGKey(0),
)

# Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims[::-1])
mesh = Mesh(devices, axis_names=('z', 'y'))
global_array = multihost_utils.host_local_array_to_global_array(array, mesh, P('z', 'y'))


@jax.jit
def do_fft(x):
    return jaxdecomp.fft.pfft3d(x)


with mesh:
    do_fft(global_array)
    before = time.time()
    karray = do_fft(global_array).block_until_ready()
    after = time.time()
    print(rank, 'took', after - before, 's')

    # And now, let's do the inverse FFT
    rec_array = jaxdecomp.fft.pifft3d(karray)

    diff = jax.jit(lambda x, y: abs(x - y).max())(rec_array, global_array)

# Let's test if things are like we expect
if rank == 0:
    print('maximum reconstruction difference', diff)

jaxdecomp.finalize()
jax.distributed.shutdown()
