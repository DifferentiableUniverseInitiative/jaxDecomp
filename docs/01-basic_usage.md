# Basic Usage

This example demonstrates how to run a JAX-based script in a **distributed** setup with `jaxDecomp`.
You can launch this script with **8 processes** (and thus 8 GPUs) via:
```bash
mpirun -n 8 python demo.py
```
or on a Slurm-based cluster (e.g., Jean Zay) using:
```bash
srun -n 8 python demo.py
```

Below is a full example script illustrating:

1. **Initializing JAX distributed** across multiple GPUs
2. **Creating a globally sharded 3D array**
3. **Performing a parallel FFT**
4. **Applying a halo exchange**
5. **Gathering results** back to a single process

```python
import jax
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import numpy as jnp
import jaxdecomp
from functools import partial

# -----------------------------
# 1. Initialize JAX distributed
# -----------------------------
# This instructs JAX which GPU to use per process.
jax.distributed.initialize()
rank = jax.process_index()

# -----------------------------
# 2. Create a globally sharded array
# -----------------------------
# Suppose we have 8 total processes. We'll create a processor mesh
# of shape (2,4). Adjust these as needed for your environment.
pdims = (2, 4)
global_shape = (1024, 1024, 1024)

# Compute local slice sizes
local_shape = (
    global_shape[0] // pdims[1],
    global_shape[1] // pdims[0],
    global_shape[2]
)

# Create a mesh of devices based on pdims
devices = mesh_utils.create_device_mesh(pdims)
mesh = Mesh(devices.T, axis_names=('x', 'y'))

# Define the sharding spec
sharding = NamedSharding(mesh, P('x', 'y'))

# Create a distributed global array
global_array = jax.make_array_from_callback(
    global_shape,
    sharding,
    data_callback=lambda _: jax.random.normal(
        jax.random.PRNGKey(rank), local_shape)
)

# -----------------------------
# 3. Perform a parallel FFT
# -----------------------------
# We will also demonstrate applying a halo exchange afterwards.

padding_width = ((32, 32), (32, 32), (0, 0))  # must be a tuple of tuples

# Shard-map helper to pad an array
@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def pad(arr, padding):
  return jnp.pad(arr, padding)

# Shard-map helper to remove the padded halo
@partial(shard_map, mesh=mesh, in_specs=(P('x', 'y'), P()), out_specs=P('x', 'y'))
def reduce_halo(x, pad_width):
  halo_x, _ = pad_width[0]
  halo_y, _ = pad_width[1]
  # Apply corrections along x
  x = x.at[halo_x:halo_x + halo_x // 2].add(x[:halo_x // 2])
  x = x.at[-(halo_x + halo_x // 2):-halo_x].add(x[-halo_x // 2:])
  # Apply corrections along y
  x = x.at[:, halo_y:halo_y + halo_y // 2].add(x[:, :halo_y // 2])
  x = x.at[:, -(halo_y + halo_y // 2):-halo_y].add(x[:, -halo_y // 2:])
  return x[halo_x:-halo_x, halo_y:-halo_y]

# A simple JITed function to modify an array
@jax.jit
def modify_array(array):
  return 2 * array + 1

# Forward FFT
karray = jaxdecomp.fft.pfft3d(global_array)

# Apply some operation (e.g., scale + offset)
karray = modify_array(karray)

# Obtain frequency grid
kvec = jaxdecomp.fft.fftfreq3d(karray)

# Demonstration: compute a gradient in the x-axis in Fourier space
karray_gradient = 1j * kvec[0] * karray

# Inverse FFT
recarray = jaxdecomp.fft.pifft3d(karray_gradient).real

# -----------------------------
# 4. Perform a halo exchange
# -----------------------------
# Example: pad the array, exchange halos, then remove the padding
padded_array = pad(recarray, padding_width)

# Exchange halo across processes
exchanged_array = jaxdecomp.halo_exchange(
    padded_array,
    halo_extents=(16, 16),
    halo_periods=(True, True)
)

# Remove the halo paddings after exchange
reduced_array = reduce_halo(exchanged_array, padding_width)

# -----------------------------
# 5. Gather results (optional)
# -----------------------------
# Only do this if the final array can fit in CPU memory.
gathered_array = multihost_utils.process_allgather(recarray, tiled=True)

# -----------------------------
# Finalize distributed JAX
# -----------------------------
jax.distributed.shutdown()
```

When you run this script, each MPI process (or Slurm task) will create its local slice of the global array. The FFT and halo operations are orchestrated in parallel using JAX and `jaxDecomp`.
