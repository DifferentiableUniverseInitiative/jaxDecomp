from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Manually set the GPU to use
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (rank + 1)

import jax

jax.config.update('jax_array', True)
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

import jaxdecomp

print(rank, "Setup", jax.devices())
jaxdecomp.init()

jax.distributed.initialize(
    coordinator_address="dappce88:8880", num_processes=size, process_id=rank)

print(rank, "Initialized")

# Sharding definition
sharding = PositionalSharding(mesh_utils.create_device_mesh((size, 1, 1)))

# Let's build the local slice
x = jax.make_array_from_single_device_arrays([8, 8, 8], sharding,
                                             [jnp.ones([8 // 2, 8, 8])])

# Let's build a config from what we know about x
config = jaxdecomp.make_config()
config.pdims = x.sharding.shape[:2]
config.gdims = x.shape
config.halo_comm_backend = jaxdecomp._src._jaxdecomp.HALO_COMM_MPI
config.transpose_comm_backend = jaxdecomp._src._jaxdecomp.TRANSPOSE_COMM_MPI_P2P

print("I have a config", config)

pencil_info = jaxdecomp.get_pencil_info(config)
print(rank, pencil_info.lo, pencil_info.hi, pencil_info.shape)

arr = jnp.zeros(pencil_info.shape) + rank

print(rank, arr)
if rank == 0:
  print('--------------------------')

arrt = jaxdecomp.transposeXtoY(arr, pdims=[2, 1], global_shape=[4, 4, 4])

print(rank, arrt)

jaxdecomp.finalize()
