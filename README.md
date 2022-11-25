# jaxDecomp: JAX Library for 3D Domain Decomposition
JAX bindings for NVIDIA's [cuDecomp](https://nvidia.github.io/cuDecomp/index.html) library [(Romero et al. 2022)](https://dl.acm.org/doi/abs/10.1145/3539781.3539797), allowing for efficient **multi-node parallel FFTs and halo exchanges** directly in low level NCCL/CUDA-Aware MPI from your JAX code :tada:

The idea of this project is to provide a few additional JAX ops, that will allow for efficient 3D FFTs and halo operations, directly through cuDecomp instead of letting JAX handling them. This is particularly important because NCCL is not necessarily well supported on all machines, and that JAX is currently not able to natively perform 3D FFTs without replicating the data on all processes.

## Usage

The API is still under development, so it doesn't look very streamlined, but you
can already do the following:
```python
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import jax
import jax.numpy as jnp
import jaxdecomp

# Initialise the library, and optionally selects a communication backend (defaults to NCCL)
jaxdecomp.init()
jaxdecomp.config.update('halo_comm_backend', jaxdecomp.HALO_COMM_MPI)
jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_A2A)

# Setup a processor mesh (should be same size as "size")
pdims= [2,4]
global_shape=[1024,1024,1024]

# Initialize an array with the expected gobal size
array = jax.random.normal(shape=[1024//pdims[1], 
                                 1024//pdims[0], 
                                 1024], 
            key=jax.random.PRNGKey(rank)).astype('complex64')

# Forward FFT, note that the output FFT is transposed
karray = jaxdecomp.pfft3d(array, 
                global_shape=global_shape, pdims=pdims)

# Reverse FFT
recarray = jaxdecomp.ipfft3d(karray, 
        global_shape=global_shape, pdims=pdims)
        
# Add halo regions to our array
padded_array = jnp.pad(array, [(32,32),(32,32),(32,32)])
# Perform a halo exchange
padded_array = jaxdecomp.halo_exchange(padded_array,
                                       halo_extents=(32,32,32),
                                       halo_periods=(True,True,True),
                                       pdims=pdims,
                                       global_shape=global_shape)
```
*Note*: All these functions are jittable and have well defined derivatives

This script would have to be run on 8 GPUs in total with something like
```bash
$ mpirun -n 8 python demo.py
```

## Install

Start by cloning this repository locally on your cluster:
```bash
$ git clone --recurse-submodules https://github.com/DifferentiableUniverseInitiative/jaxDecomp
```

#### Requirements

This install procedure assumes that the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) is available in your environment. You can either install it from the NVIDIA website, or better yet, it may be available as a module on your cluster.

Make sure all environment variables relative to the SDK are properly set.

### Step I: Building cuDecomp

Start by following the instructions in the `third_party/cuDecomp/README.md` to compile 
cuDecomp for your environment/machine. 
Note that there are configuration files in `third_party/cuDecomp/configs` for particular systems.

For instance, on NERSC's Perlmutter do the following:
```bash
$ cd third_party/cuDecomp
$ make -j CONFIGFILE=configs/nvhpcsdk_pm.conf
```

### Step II: Building jaxDecomp

This step is easier :-) From this directory, just run the following
```bash
$ pip install --user .
```
If CMake complains of not finding the NVHPC SDK, you can manually specify the location
of the sdk's cmake files like so:
```
$ export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$NVCOMPILERS/$NVARCH/22.9/cmake
$ pip install --user .
```

## Design (still aspirational)

Ideally we will want to make high-level JAX primitives compatible with the `jax.Array` API of JAX v0.4 (documented [here](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)) making it completely transparent to the user.


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
