# jaxDecomp: JAX Library for 3D Domain Decomposition and Parallel FFTs
[![Code Formatting](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/formatting.yml/badge.svg)](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/formatting.yml)

JAX bindings for NVIDIA's [cuDecomp](https://nvidia.github.io/cuDecomp/index.html) library [(Romero et al. 2022)](https://dl.acm.org/doi/abs/10.1145/3539781.3539797), allowing for efficient **multi-node parallel FFTs and halo exchanges** directly in low level NCCL/CUDA-Aware MPI from your JAX code :tada:

## Usage

Here is an example of how to use `jaxDecomp` to perform a 3D FFT on a 3D array distributed across multiple GPUs. This example also includes a halo exchange operation, which is a common operation in many scientific computing applications.

```python
import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
import jaxdecomp

# Initialise the library, and optionally selects a communication backend (defaults to NCCL)
jaxdecomp.config.update('halo_comm_backend', jaxdecomp.HALO_COMM_MPI)
jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_A2A)

# Initialize jax distributed to instruct jax local process which GPU to use
jax.distributed.initialize()
rank = jax.process_index()

# Setup a processor mesh (should be same size as "size")
pdims= (1,4)
global_shape=[1024,1024,1024]

# Initialize an array with the expected gobal size
local_array = jax.random.normal(
    shape=[
        global_shape[0] // pdims[1], global_shape[1] // pdims[0],
        global_shape[2]
    ],
    key=jax.random.PRNGKey(rank))

 # Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims[::-1])
mesh = Mesh(devices, axis_names=('z', 'y'))
global_array = multihost_utils.host_local_array_to_global_array(
    local_array, mesh, P('z', 'y'))

# Forward FFT, note that the output FFT is transposed
@jax.jit
def modify_array(array):
    return 2 * array + 1

with mesh:
    # Forward FFT
    karray = jaxdecomp.fft.pfft3d(global_array)
    # Do some operation on your array
    karray = modify_array(karray)
    # Reverse FFT
    recarray = jaxdecomp.fft.pifft3d(karray).astype('float32')
    # Add halo regions to our array
    padding_width = ((32,32),(32,32),(32,32)) # Has to a tuple of tuples
    padded_array = jaxdecomp.slice_pad(recarray, padding_width , pdims)
    # Perform a halo exchange + reduce
    exchanged_reduced = jaxdecomp.halo_exchange(padded_array,
                                           halo_extents=(32,32,32),
                                           halo_periods=(True,True,True))
    # Remove the halo regions
    recarray = jaxdecomp.slice_unpad(exchanged_reduced, padding_width, pdims)

    # Gather the results (only if it fits on CPU memory)
    gathered_array = multihost_utils.process_allgather(recarray, tiled=True)

# Finalize the library
jaxdecomp.finalize()
jax.distributed.shutdown()
```
**Note**: All these functions are jittable and have well defined derivatives!

This script would have to be run on 8 GPUs in total with something like

```bash
$ mpirun -n 8 python demo.py
```

On an HPC cluster like Jean Zay you should do this

```bash
$ srun python demo.py
```

Check the slurm [README](slurms/README.md) and [template](slurms/template.slurm) for more information on how to run on a Jean Zay.

### Caveats

The code presented above should work, but there are a few caveats mentioned in [this issue](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/issues/1). If you need a functionality that is not currently implemented, feel free to mention it on that issue.

## Install

Start by cloning this repository locally on your cluster:
```bash
$ git clone --recurse-submodules https://github.com/DifferentiableUniverseInitiative/jaxDecomp
```

#### Requirements

This install procedure assumes that the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) is available in your environment. You can either install it from the NVIDIA website, or better yet, it may be available as a module on your cluster.

Make sure all environment variables relative to the SDK are properly set.

### Building jaxDecomp

From this directory, install & build jaxDecomp via pip
```bash
$ pip install --user .
```
If CMake complains of not finding the NVHPC SDK, you can manually specify the location
of the sdk's cmake files like so:
```
$ export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$NVCOMPILERS/$NVARCH/22.9/cmake
$ pip install --user .
```

### Specific Install Notes for Specific Machines

#### IDRIS [Jean Zay](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html) HPE SGI 8600 supercomputer

As of April. 2024, the following works:

You need to load modules in that order exactly.
```bash
# Load NVHPC 23.9 because it has cuda 12.2
module load nvidia-compilers/23.9 cuda/12.2.0 cudnn/8.9.7.29-cuda  openmpi/4.1.5-cuda nccl/2.18.5-1-cuda cmake
# Installing mpi4py
CFLAGS=-noswitcherror pip install mpi4py
# Installing jax
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Installing jaxdecomp
export CMAKE_PREFIX_PATH=$NVHPC_ROOT/cmake
pip install .
```

#### NERSC [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/) HPE Cray EX supercomputer

As of Nov. 2022, the following works:
```bash
module load PrgEnv-nvhpc python
export CRAY_ACCEL_TARGET=nvidia80
# Installing mpi4py
MPICC="cc -target-accel=nvidia80 -shared" CC=nvc CFLAGS="-noswitcherror" pip install --force --no-cache-dir --no-binary=mpi4py mpi4py
# Installing jax
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Installing jaxdecomp
export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cmake
pip install .
```

## Design

Here is  what works now :

```python
from jaxdecomp.fft import pfft3, ipfft3

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils

# Initialize jax distributed to instruct jax local process which GPU to use
jax.distributed.initialize()

pdims = (2 , 4)
global_shape = (512 , 512 , 512 )

local_array = jax.random.normal(shape=[global_shape[0]//pdims[0],
                                        global_shape[1]//pdims[1],
                                        global_shape[2]], key=jax.random.PRNGKey(0))

# remap to global array (this is a free call no communications are happening)

devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices, axis_names=('z', 'y'))
global_array = multihost_utils.host_local_array_to_global_array(
    array, mesh, P('z', 'y'))


with mesh
    z = pfft3(global_array)

    # If we could inspect the distribution of y, we would see that it is sliced in 2 along x, and 4 along y

    # This could also be part of a jitted function, no problem
    z_rec = ipfft3(z)

# And z remains at all times distributed.

jaxdecomp.finalize()
jax.distributed.shutdown()

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
