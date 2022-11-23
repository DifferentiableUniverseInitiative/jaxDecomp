# again, this only works on startup!
from jax.config import config
config.update("jax_enable_x64", True)

from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# Manually set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % (rank + 1)

import jax
import numpy as np
import jax.numpy as jnp
import jaxdecomp
import matplotlib.pyplot as plt
import time

# Can we remove this?
jaxdecomp.init()
pdims= (1,2)
global_shape=(32,32,32)

# Initialize an array with the expected gobal size
array = jax.random.normal(shape=[32//2,
                                 32,
                                 32], key=jax.random.PRNGKey(0)) + rank

array = array.astype('complex128')

@jax.jit
def do_fft(x):
    return jaxdecomp.fft.pfft3d(x, pdims=pdims, global_shape=global_shape)

do_fft(array)
before = time.time()
karray = do_fft(array)
after = time.time()
print(rank, 'took', after - before, 's')

# And now, let's do the inverse FFT
rec_array = jaxdecomp.fft.pifft3d(karray, pdims=pdims, global_shape=global_shape)

from jaxdecomp._src import _jaxdecomp
config = _jaxdecomp.GridConfig()
config.pdims = pdims
config.gdims = global_shape
config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P
pencil = jaxdecomp.get_pencil_info(config, 0)
print(rank, pencil.shape, pencil.lo, pencil.hi, pencil.order)
pencil = jaxdecomp.get_pencil_info(config, 2)
print(rank, pencil.shape, pencil.lo, pencil.hi, pencil.order)


if rank ==0:
    # Let's test if things are like we expect
    global_array = jnp.concatenate([array, array+1], axis=0)
    global_array = jnp.fft.fftn(global_array)

    # So... if I understand correctly, the array should have shape [y,z,x]
    global_array = global_array.transpose([1,2,0])

    diff = global_array[:16,:,:] - karray
    print('maximum difference', jnp.abs(diff).max())
    plt.subplot(131)
    plt.imshow(jnp.abs(diff).mean(axis=0)) 
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(jnp.abs(diff).mean(axis=1)) 
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(jnp.abs(diff).mean(axis=2))    
    plt.colorbar()
    plt.savefig("test_forward_fft.png")

    diff = rec_array - array
    print('maximum reconstruction difference', jnp.abs(diff).max())
    plt.figure()
    plt.subplot(131)
    plt.imshow(jnp.abs(diff).mean(axis=0)) 
    plt.colorbar()
    plt.subplot(132)
    plt.imshow(jnp.abs(diff).mean(axis=1)) 
    plt.colorbar()
    plt.subplot(133)
    plt.imshow(jnp.abs(diff).mean(axis=2))    
    plt.colorbar()
    plt.savefig("test_rec_fft.png")

jaxdecomp.finalize()
