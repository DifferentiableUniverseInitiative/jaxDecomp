
# jaxDecomp: JAX Library for 3D Domain Decomposition and Parallel FFTs

[![Build](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/github-deploy.yml/badge.svg)](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/github-deploy.yml)
[![Code Formatting](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/formatting.yml/badge.svg)](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/formatting.yml)
[![Tests](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/tests.yml/badge.svg)](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/actions/workflows/tests.yml)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://readthedocs.org/projects/jaxdecomp/badge/?version=latest)](https://jaxdecomp.readthedocs.io/en/latest/)

> **Important**
> Version `0.2.0` includes a **pure JAX backend** that **no longer requires MPI**. For multi-node runs, MPI and NCCL backends are still available through **cuDecomp**.

JAX reimplementation and bindings for NVIDIA's [cuDecomp](https://nvidia.github.io/cuDecomp/index.html) library [(Romero et al. 2022)](https://dl.acm.org/doi/abs/10.1145/3539781.3539797), enabling **multi-node parallel FFTs and halo exchanges** directly in low-level NCCL/CUDA-Aware MPI from your JAX code.

> **Important**
> Starting from version **0.2.8**, jaxDecomp supports JAX's Shardy partitioner, which can be activated via `jax.config.update('jax_use_shardy_partitioner', True)`. This partitioner is enabled by default in JAX 0.7.x and later versions.
> Shardy support is an **internal implementation change** and users should not expect any behavioral differences outside of what the JAX sharding mechanism provides, as explained in the [JAX Shardy migration documentation](https://docs.jax.dev/en/latest/shardy_jax_migration.html).


---

## Usage

Below is a simple code snippet illustrating how to perform a **3D FFT** on a distributed 3D array, followed by a halo exchange. For demonstration purposes, we force 8 CPU devices via environment variables:

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
import jaxdecomp

# Create a 2x4 mesh of devices on CPU
pdims = (2, 4)
mesh = jax.make_mesh(pdims, axis_names=('x', 'y'))
sharding = NamedSharding(mesh, P('x', 'y'))

# Create a random 3D array and enforce sharding
a = jax.random.normal(jax.random.PRNGKey(0), (1024, 1024, 1024))
a = jax.lax.with_sharding_constraint(a, sharding)

# Parallel FFTs
k_array = jaxdecomp.fft.pfft3d(a)
rec_array = jaxdecomp.fft.pifft3d(a)

# Parallel halo exchange
exchanged = jaxdecomp.halo_exchange(a, halo_extents=(16, 16), halo_periods=(True, True))
```

All these functions are **JIT**-compatible and support **automatic differentiation** (with [some caveats](https://jaxdecomp.readthedocs.io/en/latest/06-caveats.html)).

See also:
- [Basic Usage](https://jaxdecomp.readthedocs.io/en/latest/01-basic_usage.html)
- [Distributed LPT Example](examples/lpt_nbody_demo.py)

> **Important**
> Multi-node FFTs work with both JAX and cuDecomp backends\
> For CPU with JAX, Multi-node is supported starting JAX v0.5.1 (with `gloo` backend)

---

## Running on an HPC Cluster

On HPC clusters (e.g., Jean Zay, Perlmutter), you typically launch your script with:
```bash
srun python your_script.py
```
or
```bash
mpirun -n 8 python your_script.py
```

See the Slurm [README](slurms/README.md) and [template script](slurms/template.slurm) for more details.


---

## Using cuDecomp (MPI and NCCL)

For other features, compile and install with cuDecomp enabled as described in [install](#2-jax--cudecomp-backend-advanced):

```python
import jaxdecomp

# Optionally select communication backends (defaults to NCCL)
jaxdecomp.config.update('halo_comm_backend', jaxdecomp.HALO_COMM_MPI)
jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_A2A)

# Then specify 'backend="cudecomp"' in your FFT or halo calls:
karray = jaxdecomp.fft.pfft3d(global_array, backend='cudecomp')
recarray = jaxdecomp.fft.pifft3d(karray, backend='cudecomp')
exchanged_array = jaxdecomp.halo_exchange(
    padded_array, halo_extents=(16, 16), halo_periods=(True, True), backend='cudecomp'
)
```

## Install

### 1. Pure JAX Version (Easy / Recommended)

`jaxDecomp` is on PyPI:

1. **Install the appropriate JAX wheel**:
   - **GPU**:
     ```bash
     pip install --upgrade "jax[cuda]"
     ```
   - **CPU**:
     ```bash
     pip install --upgrade "jax[cpu]"
     ```
2. **Install `jaxdecomp`**:
   ```bash
   pip install jaxdecomp
   ```

This setup uses the pure-JAX backend—**no** MPI required.

### 2. JAX + cuDecomp Backend (Advanced)

If you need to use `MPI` instead of `NCCL` for `GPU`, you can build from GitHub with cuDecomp enabled. This requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk). Ensure `nvc`, `nvc++`, and `nvcc` are in your `PATH`, `CUDA`, `MPI`, and `NCCL` shared libraries are on `LD_LIBRARY_PATH`, and set `CC=nvc` and `CXX=nvc++` before building.

```bash
pip install -U pip
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -Ccmake.define.JD_CUDECOMP_BACKEND=ON
```

Alternatively, clone the repository locally and install from your checkout:

```bash
git clone https://github.com/DifferentiableUniverseInitiative/jaxDecomp.git --recursive
cd jaxDecomp
pip install -U pip
pip install . -Ccmake.define.JD_CUDECOMP_BACKEND=ON
```

- If CMake cannot find NVHPC, set:
  ```bash
  export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$NVCOMPILERS/$NVARCH/22.9/cmake
  ```
  and then install again.

---

## Machine-Specific Notes

### IDRIS [Jean Zay](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html) HPE SGI 8600 supercomputer


As of February 2025, loading modules **in this exact order** works:

```bash
module load nvidia-compilers/23.9 cuda/12.2.0 cudnn/8.9.7.29-cuda openmpi/4.1.5-cuda nccl/2.18.5-1-cuda cmake

# Install JAX
pip install --upgrade "jax[cuda]"

# Install jaxDecomp with cuDecomp
export CMAKE_PREFIX_PATH=$NVHPC_ROOT/cmake # sometimes needed
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -Ccmake.define.JD_CUDECOMP_BACKEND=ON
```

**Note**: If using only the pure-JAX backend, you do not need NVHPC.

#### NERSC [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/) HPE Cray EX supercomputer

As of November 2022:

```bash
module load PrgEnv-nvhpc python
export CRAY_ACCEL_TARGET=nvidia80

# Install JAX
pip install --upgrade "jax[cuda]"

# Install jaxDecomp w/ cuDecomp
export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cmake
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -CCmake.define.JD_CUDECOMP_BACKEND=ON
```

---

## Backend Configuration (cuDecomp Only)

By default, cuDecomp uses NCCL for inter-device communication. You can customize this at runtime:

```python
import jaxdecomp

# Choose MPI or NVSHMEM for halo and transpose ops
jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_A2A)
jaxdecomp.config.update('halo_comm_backend', jaxdecomp.HALO_COMM_MPI)
```

This can also be managed via environment variables, as described in the [docs](https://github.com/DifferentiableUniverseInitiative/jaxDecomp/tree/main/docs).

---

## Autotune Computational Mesh (cuDecomp Only)

The cuDecomp library can **autotune** the partition layout to maximize performance:

```python
automesh = jaxdecomp.autotune(shape=[512,512,512])
# 'automesh' is an optimized partition layout.
# You can then create a JAX Sharding spec from this:
from jax.sharding import PositionalSharding
sharding = PositionalSharding(automesh)
```

---

**License**: This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

For more details, see the [examples](examples/) directory and the [documentation](https://jaxdecomp.readthedocs.io/en/latest). Contributions and issues are welcome!
