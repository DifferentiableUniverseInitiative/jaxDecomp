# Installation

## 1. Pure JAX Version (Easy / Recommended)

The easiest way to get started with `jaxDecomp` is via PyPI using the pure JAX backend—**no MPI or GPU-specific setup required**.

### ➤ Step-by-step

1. **Install the appropriate JAX wheel**:
   - **GPU**:
     ```bash
     pip install --upgrade "jax[cuda]"
     ```
   - **CPU**:
     ```bash
     pip install --upgrade "jax[cpu]"
     ```

2. **Install `jaxDecomp`**:
   ```bash
   pip install jaxdecomp
   ```

This setup uses the JAX backend by default and is ideal for experimentation, development, and most common research workflows.

---

## 2. cuDecomp Backend (Advanced / HPC)

If you're working on an HPC cluster and need **MPI-based communication** for large-scale GPU or CPU FFTs, you can build from source with cuDecomp enabled.

### ➤ Install with cuDecomp

Make sure your environment provides a **CUDA-aware MPI toolchain**, such as the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk).

```bash
pip install -U pip
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -Ccmake.define.JD_CUDECOMP_BACKEND=ON
```

If CMake cannot find the NVHPC toolchain, set:

```bash
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$NVHPC_ROOT/cmake
```

Then re-run the installation.

### Troubleshooting

If JAX complains about incompatibility with cuSparse or any other library, the easiest solution is to install JAX locally using the `cuda-local` option:

```bash
pip install --upgrade "jax[cuda-local]"
```

Then proceed with installing `jaxDecomp` with cuDecomp support.

> ℹ️ You can read more about cuDecomp setup and tuning at the official [cuDecomp GitHub repo](https://github.com/NVIDIA/cuDecomp).

---

## Machine-Specific Installation Notes

### IDRIS [Jean Zay](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html) HPE SGI 8600 supercomputer

As of February 2025, loading modules **in this exact order** works:

```bash
module load nvidia-compilers/25.1 cuda/12.6.3 openmpi/4.1.6-cuda nccl/2.26.2-1-cuda cudnn  cmake
# Install JAX
pip install --upgrade "jax[cuda-local]"

# Install jaxDecomp with cuDecomp
export CMAKE_PREFIX_PATH=$NVHPC_ROOT/cmake # sometimes needed
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -Ccmake.define.JD_CUDECOMP_BACKEND=ON
```

**Note**: If using only the pure-JAX backend, you do not need NVHPC.

> **Important for JeanZay users**
> Make sure to load the correct architecture module before loading the `nvidia-compilers` module.
> For example for A100 you need to load `module load arch/a100` first.
> You also need to set the CXXFLAGS to `export CXXFLAGS="-tp=zen2 -noswitcherror"` if you are using the H100 or A100 partition or if you are using AMD CPUs in general.
> More info in [Jean Zay documentation](http://www.idris.fr/eng/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm-eng.html#a100_partition_gpu_p5).


### NERSC [Perlmutter](https://docs.nersc.gov/systems/perlmutter/architecture/) HPE Cray EX supercomputer

As of November 2022:

```bash
module load PrgEnv-nvhpc python
export CRAY_ACCEL_TARGET=nvidia80

# Install JAX
pip install --upgrade "jax[cuda]"

# Install jaxDecomp w/ cuDecomp
export CMAKE_PREFIX_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.5/cmake
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -Ccmake.define.JD_CUDECOMP_BACKEND=ON
```

---

## Backend Selection at Runtime

Most functions in `jaxDecomp` support dynamic backend selection via a `backend` keyword argument. For example:

```python
from jaxdecomp.fft import pfft3d

# Use the default (pure JAX)
k_array = pfft3d(x)

# Use cuDecomp (if compiled and available)
k_array = pfft3d(x, backend="cudecomp")
```

This applies to:

* `jaxdecomp.fft.pfft3d`
* `jaxdecomp.fft.pifft3d`
* `jaxdecomp.halo_exchange`
* (and other `jaxdecomp.fft.*` and transposition routines)

---

## cuDecomp Transpose Communication Backends

If you're using the cuDecomp backend, you can also **manually choose the transpose communication strategy**, which may significantly affect performance depending on your cluster hardware and MPI configuration.

Available options:

```python
from jaxdecomp import (
    TRANSPOSE_COMM_NCCL,
    TRANSPOSE_COMM_MPI_A2A,
    TRANSPOSE_COMM_MPI_P2P,
)

# Set transpose communication backend (default is NCCL)
jaxdecomp.config.update('transpose_comm_backend', TRANSPOSE_COMM_NCCL)
jaxdecomp.config.update('transpose_comm_backend', TRANSPOSE_COMM_MPI_P2P)
jaxdecomp.config.update('transpose_comm_backend', TRANSPOSE_COMM_MPI_A2A)
```

> ℹ️ These options are described in more detail in the [cuDecomp GitHub documentation](https://github.com/NVIDIA/cuDecomp#transpose-communication-backends).

---

## Notes on Performance

Backend performance varies widely depending on your cluster setup (e.g., interconnect type, topology, NCCL version, MPI implementation). We recommend benchmarking both backends on your target workload to determine the best configuration.
