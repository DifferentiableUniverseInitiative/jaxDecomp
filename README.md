# jaxDecomp: JAX Library for 3D Domain Decomposition and Parallel FFTs
[![Code Formatting](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/formatting.yml/badge.svg)](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/formatting.yml)
[![Tests](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/tests.yml/badge.svg)](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/tests.yml/badge.svg)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


> [!IMPORTANT]
> Version `0.2.0` has a pure JAX backend and no longer requires MPI .. MPI and NCCL backends are still available through cuDecomp


JAX bindings for NVIDIA's [cuDecomp](https://nvidia.github.io/cuDecomp/index.html) library [(Romero et al. 2022)](https://dl.acm.org/doi/abs/10.1145/3539781.3539797), allowing for efficient **multi-node parallel FFTs and halo exchanges** directly in low level NCCL/CUDA-Aware MPI from your JAX code :tada:

## Usage

Here is an example of how to use `jaxDecomp` to perform a 3D FFT on a 3D array distributed across multiple GPUs. This example also includes a halo exchange operation, which is a common operation in many scientific computing applications.

```python
import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import numpy as jnp
import jaxdecomp
from functools import partial
# Initialize jax distributed to instruct jax local process which GPU to use
jax.distributed.initialize()
rank = jax.process_index()

# Setup a processor mesh (should be same size as "size")
pdims = (2, 4)
global_shape = (1024, 1024, 1024)

# Initialize an array with the expected gobal size
local_shape = (global_shape[0] // pdims[1], global_shape[1] // pdims[0],
               global_shape[2])
# Remap to the global array from the local slice
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices.T, axis_names=('x', 'y'))
sharding = NamedSharding(mesh, P('x', 'y'))
global_array = jax.make_array_from_callback(
    global_shape,
    sharding,
    data_callback=lambda _: jax.random.normal(
        jax.random.PRNGKey(rank), local_shape))

padding_width = ((32, 32), (32, 32), (0, 0))  # Has to a tuple of tuples


@partial(
    shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def pad(arr, padding):
  return jnp.pad(arr, padding)


@partial(
    shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def reduce_halo(x, pad_width):

  halo_x , _ = pad_width[0]
  halo_y , _ = pad_width[1]
  # Apply corrections along x
  x = x.at[halo_x:halo_x + halo_x // 2].add(x[:halo_x // 2])
  x = x.at[-(halo_x + halo_x // 2):-halo_x].add(x[-halo_x // 2:])
  # Apply corrections along y
  x = x.at[:, halo_y:halo_y + halo_y // 2].add(x[:, :halo_y // 2])
  x = x.at[:, -(halo_y + halo_y // 2):-halo_y].add(x[:, -halo_y // 2:])

  return x[halo_x:-halo_x, halo_y:-halo_y]


@jax.jit
def modify_array(array):
  return 2 * array + 1


# Forward FFT
karray = jaxdecomp.fft.pfft3d(global_array)
# Do some operation on your array
karray = modify_array(karray)
kvec = jaxdecomp.fft.fftfreq3d(karray)
# Do a gradient in the X axis
karray_gradient = 1j * kvec[0] * karray
# Reverse FFT
recarray = jaxdecomp.fft.pifft3d(karray_gradient).real
# Add halo regions to our array
padded_array = pad(recarray, padding_width)
# Perform a halo exchange
exchanged_array = jaxdecomp.halo_exchange(
    padded_array, halo_extents=(16, 16), halo_periods=(True, True))
# Reduce the halo regions and remove the padding
reduced_array = reduce_halo(exchanged_array, padding_width)

# Gather the results (only if it fits on CPU memory)
gathered_array = multihost_utils.process_allgather(recarray, tiled=True)

# Finalize the distributed JAX
jax.distributed.shutdown()
```
**Note**: All these functions are jittable and have well defined derivatives!

This script would have to be run on 8 GPUs in total with something like

```bash
mpirun -n 8 python demo.py
```
or on a slurm cluster like Jean jean-zay

```bash
srun -n 8 python demo.py
```

## Using cuDecomp (MPI and NCCL)

You can also use the cuDecomp backend by compiling the library with the right flag (check the [installation instructions](#install)) and setting the backend to use MPI or NCCL. Here is how you can do it:

```python
import jaxdecomp
# Initialise the library, and optionally selects a communication backend (defaults to NCCL)
jaxdecomp.config.update('halo_comm_backend', jaxdecomp.HALO_COMM_MPI)
jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_A2A)

# and then call the functions with the cuDecomp backends
karray = jaxdecomp.fft.pfft3d(global_array , backend='cudecomp')
recarray = jaxdecomp.fft.pifft3d(karray , backend='cudecomp')
exchanged_array = jaxdecomp.halo_exchange(
    padded_array, halo_extents=(16, 16), halo_periods=(True, True), backend='cudecomp')

```

please check the tests in [tests](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/tree/main/tests) folder for more examples.

On an HPC cluster like Jean Zay you should do this

```bash
$ srun python demo.py
```

Check the slurm [README](slurms/README.md) and [template](slurms/template.slurm) for more information on how to run on a Jean Zay.

## Install

### Installing the pure JAX version (Easy)

jaxDecomp is available on pypi and can be installed via pip:

First install desired JAX version

For GPU
```bash
pip install -U jax[cuda12]
```
For CPU
```bash
pip install -U jax[cpu]
```
Then you can pip install jaxdecomp

```bash
pip install jaxdecomp
```

### Installing JAX and cuDecomp (Advanced)

You need to install from this github after installing or loading the correct modules

This install procedure assumes that the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) is available in your environment. You can either install it from the NVIDIA website, or better yet, it may be available as a module on your cluster.

### Building jaxDecomp

```bash
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -CCmake.define.JD_CUDECOMP_BACKEND=ON
```

If CMake complains of not finding the NVHPC SDK, you can manually specify the location
of the sdk's cmake files like so:
```
$ export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$NVCOMPILERS/$NVARCH/22.9/cmake
$ pip install --user .
```

### Specific Install Notes for Specific Machines

#### IDRIS [Jean Zay](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html) HPE SGI 8600 supercomputer

As of October. 2024, the following works:

You need to load modules in that order exactly.
```bash
# Load NVHPC 23.9 because it has cuda 12.2
module load nvidia-compilers/23.9 cuda/12.2.0 cudnn/8.9.7.29-cuda  openmpi/4.1.5-cuda nccl/2.18.5-1-cuda cmake
# Installing jax
pip install --upgrade "jax[cuda12]"
# Installing jaxdecomp
export CMAKE_PREFIX_PATH=$NVHPC_ROOT/cmake # Not always needed
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -CCmake.define.JD_CUDECOMP_BACKEND=ON
```

__Note__: This is needed **only** if you want to use the cuDecomp backend. If you are using the pure JAX backend, you can skip the NVHPC SDK installation and just `pip install jaxdecomp` **after** installing the correct JAX version for your hardware.

#### NERSC [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/) HPE Cray EX supercomputer

As of Nov. 2022, the following works:
```bash
module load PrgEnv-nvhpc python
export CRAY_ACCEL_TARGET=nvidia80
# Installing jax
pip install --upgrade "jax[cuda12]"
# Installing jaxdecomp
export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cmake
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -CCmake.define.JD_CUDECOMP_BACKEND=ON
```
## Backend configuration (Only for cuDecomp)

__Note__: For the JAX backend, only NCCL is available.

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

## Autotune computational mesh (Only for cuDecomp)

We can also make things fancier, since cuDecomp is able to autotune, we could use it to tell us what is the best way to partition the data given the available GPUs, something like this:
```python
automesh = jaxdecomp.autotune(shape=[512,512,512])
# This is a JAX Sharding spec object, optimized for the given GPUs
# and shape of the tensor
sharding = PositionalSharding(automesh)
```
