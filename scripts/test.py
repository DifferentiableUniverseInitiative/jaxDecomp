import os
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
# Manually set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"]="%d"%(rank+1)
import jax
import jax.numpy as jnp
import jaxdecomp
import time

print(rank, "Setup", jax.devices())
jaxdecomp.init()
print(rank, "Initialized")

# Let's build a 
config = jaxdecomp.make_config()
config.pdims = [2,1]
config.gdims = [4,4,4]
config.halo_comm_backend = jaxdecomp._src._jaxdecomp.HALO_COMM_MPI
config.transpose_comm_backend = jaxdecomp._src._jaxdecomp.TRANSPOSE_COMM_MPI_P2P

print("I have a config", config)

pencil_info = jaxdecomp.get_pencil_info(config)
print(rank, pencil_info.lo, pencil_info.hi, pencil_info.shape)

arr = jnp.zeros(pencil_info.shape)+rank

print(rank, arr)
if rank == 0:
    print('--------------------------')
time.sleep(1)

arrt = jaxdecomp.transposeXtoY(arr, 
                               pdims=[2,1],
                               global_shape=[4,4,4])

print(rank, arrt)

jaxdecomp.finalize()

