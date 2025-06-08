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
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:$NVCOMPILERS/$NVARCH/22.9/cmake
```

Then re-run the installation.

> ℹ️ You can read more about cuDecomp setup and tuning at the official [cuDecomp GitHub repo](https://github.com/NVIDIA/cuDecomp).

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
