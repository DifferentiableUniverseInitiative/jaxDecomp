## Project Overview

This project, `jaxDecomp`, is a JAX library for performing 3D domain decomposition and parallel Fast Fourier Transforms (FFTs). It provides a high-level API for these operations, with two backend implementations: a pure JAX backend and a high-performance backend that uses NVIDIA's cuDecomp library. The pure JAX backend is recommended for ease of use and does not require MPI, while the cuDecomp backend offers better performance on multi-node GPU systems.

The core functionalities of the library include:
- `pfft3d`: a function for performing a 3D parallel FFT on a distributed 3D array.
- `pifft3d`: a function for performing the inverse of `pfft3d`.
- `halo_exchange`: a function for performing a halo exchange on a distributed array.
- a set of transpose functions for changing the data layout of a distributed array.

The library is designed to be JIT-compatible and supports automatic differentiation. It also supports JAX's Shardy partitioner for more flexible sharding of arrays.

## Building and Running

### Installation

The project can be installed from PyPI. The installation process depends on the desired backend.

#### Pure JAX Version (Easy / Recommended)

1.  **Install the appropriate JAX wheel**:
    *   **GPU**:
        ```bash
        pip install --upgrade "jax[cuda]"
        ```
    *   **CPU**:
        ```bash
        pip install --upgrade "jax[cpu]"
        ```
2.  **Install `jaxdecomp`**:
    ```bash
    pip install jaxdecomp
    ```

This setup uses the pure-JAX backend and does not require MPI.

#### JAX + cuDecomp Backend (Advanced)

This requires the [NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk).

```bash
pip install -U pip
pip install git+https://github.com/DifferentiableUniverseInitiative/jaxDecomp -Ccmake.define.JD_CUDECOMP_BACKEND=ON
```

### Running Tests

The project uses `pytest` for testing. To run the tests, first install the test dependencies:

```bash
pip install -e ".[test]"
```

Then run pytest:

```bash
pytest
```

## Development Conventions

The project uses `ruff` for linting and formatting. The configuration can be found in the `pyproject.toml` file.

- **Linting**: `ruff check .`
- **Formatting**: `ruff format .`

The project also uses `mypy` for static type checking.

- **Type Checking**: `mypy src/jaxdecomp`
