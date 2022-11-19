# jaxDecomp
JAX bindings for the cuDecomp NVIDIA library, to allow for efficient parallel FFTs and halo exchanges directly in low level NCCL/CUDA-Aware MPI from your JAX code :-)

https://nvidia.github.io/cuDecomp/index.html

## Design 

The idea of this project is to provide a few additional JAX ops, that will allow for efficient 3D FFTs and halo operations, directly through cuDecomp instead of letting JAX handles them. This is particularly important because NCCL is not necessarily well supported on all machines, and that JAX is currently not able to natively perform 3D FFTs without replicating the data on all processes.


Once core guiding principle is to make high-level JAX primitives compatible with the `jax.Array` API of JAX v0.4 (documented [here](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)) making it completely transparent to the user.


Here is a prototype of what we are aiming for (still aspirational):
```python
# We need to initialize MPI as it is required by cuDecomp
from mpi4py import MPI
comm = MPI.COMM_WORLD
from jaxdecomp import pfft3

import jax
import jax.numpy as jnp

# For now, the Array API needs to be explicitely activated
jax.config.update('jax_array', True)

from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding

# Let's create a Sharding object which we will use to distribute
# a value across devices.
sharding = PositionalSharding(mesh_utils.create_device_mesh((2,4,1)))

# Create a global array
def local_array_init(index):
    return 2*index + 1 # Or whatever code that knows how to compute the array's global value at 'index'
y = jax.make_array_from_single_device_arrays(shape=[512,512,512],
                                             sharding=sharding,
                                             data_callback=local_array_init)
# If we could inspect the distribution of y, we would see that it is sliced in 2 along x, and 4 along y

# This could also be part of a jitted function, no problem
z = pfft3(y)

# And z remains at all times distributed.
```

#### Backend configuration

We can set the default communication backend to use for cuDecomp operations either through a `config` module, or environment variables. This will allow the users to choose at startup (although can be changed afterwards) the communication backend, making it possible to use CUDA-aware MPI or NVSHMEM as preferred.

Here is how it would like:
```python
jaxdecomp.config.update('transpose_comm_backend', 'NCCL')
# We could for instance time how long it takes to execute in this mode
%timeit pfft3(y)

# And then update the backend 
jaxdecomp.config.update('transpose_comm_backend', 'MPI')
# And measure again
%timeit pfft3(y)
```

#### Autotune computational mesh

We can also make things fancier, since cuDecomp is able to autotune, we could use it to tell us what is the best way to partition the data given the available GPUs, something like this:
```python
automesh = jaxdecomp.autotune(shape=[512,512,512]) 
# This is a JAX Sharding spec object, optimized for the given GPUs 
# and shape of the tensor
sharding = PositionalSharding(automesh)
```
