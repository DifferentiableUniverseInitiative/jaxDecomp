# XLA Sharding Configuration Guide

This guide covers XLA sharding configuration when working with jaxDecomp, including Shardy partitioner settings, sharding spec/mesh compatibility requirements, and explicit vs auto axis types.

## Shardy Partitioner Configuration

### Activating/Deactivating Shardy

The Shardy partitioner is the default in JAX 0.7.0+. You can control it via JAX configuration:

```python
import jax

# Activate Shardy partitioner (default in JAX 0.7.0+)
jax.config.update('jax_use_shardy_partitioner', True)

# Deactivate Shardy partitioner (use legacy GSPMD)
jax.config.update('jax_use_shardy_partitioner', False)
```

## Sharding Spec Must Match Mesh

### Warning: PartitionSpec Must Directly Use Mesh Axis Names

When creating a sharding, the `PartitionSpec` must directly correspond to the mesh axes. Using `None` for an axis that has size > 1 in the mesh is **NOT** valid.

**Example - Correct vs Incorrect:**

```python
import os
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

pdims = (2, 4)
pdim_x, pdim_y = pdims

mesh_2d = jax.make_mesh((pdims), ('x', 'y'))

# OK - spec directly uses mesh axis names
sharding = NamedSharding(mesh_2d, P('x', 'y'))

# NOT OK - 'x' has size 2, cannot use None
sharding = NamedSharding(mesh_2d, P(None, 'y'))
```

### Workaround: Create a Specific Mesh for Partial Sharding

If you need to use `None` in your spec, create a mesh where that axis has size 1:

```python
# Create a mesh with size 1 on the first axis
mesh_1 = jax.make_mesh((1, pdim_y), ('UNUSED', 'y'), devices=jax.devices()[::pdim_x])

# OK now - UNUSED axis has size 1
sharding = NamedSharding(mesh_1, P(None, 'y'))
```

### Validating Spec/Mesh Compatibility

Use `jaxdecomp.validate_spec_matches_mesh` to check compatibility:

```python
from jaxdecomp import validate_spec_matches_mesh
from jax.sharding import PartitionSpec as P

def check(spec, mesh, name):
    try:
        validate_spec_matches_mesh(spec, mesh)
        print(f"Spec {spec} is VALID for mesh {name}")
    except Exception as e:
        print(f"Spec {spec} is INVALID for mesh {name}: {e}")

check(P('x', 'y'), mesh_2d, "mesh_2d")   # VALID
check(P(None, 'y'), mesh_2d, "mesh_2d")  # INVALID
check(P(None, 'y'), mesh_1, "mesh_1")    # VALID
```

## Auto vs Explicit Axis Types

### jaxDecomp Compatibility Warning

```{warning}
**Important (February 2026):** jaxDecomp's `custom_partitioning` primitives are **not directly compatible** with `AxisType.Explicit` mesh axes. This is due to a limitation in JAX's `custom_partitioning` mechanism where callbacks receive meshes with axis types converted to `Auto`, but XLA's SPMD partitioner still makes decisions based on the original `Explicit` types.
```

### Option 1: Use Auto Axis Types (Recommended)

The simplest approach is to use `AxisType.Auto` when working with jaxDecomp:

```python
from jax.sharding import AxisType

# Use Auto axis types
mesh_auto = jax.make_mesh(pdims, ('x', 'y'),
                           axis_types=(AxisType.Auto, AxisType.Auto))
```

### Option 2: Use `auto_axes` Wrapper for Explicit Meshes

If you must use `AxisType.Explicit` (e.g., for compatibility with other parts of your codebase), you can use JAX's `auto_axes` decorator to wrap jaxDecomp functions:

```python
import jax
from jax.experimental import mesh_utils
from jax.sharding import AxisType, auto_axes, reshard
from jax.sharding import PartitionSpec as P
import jaxdecomp as jd

pdims = (2, 4)

# Create an Explicit mesh
mesh_explicit = jax.make_mesh(pdims, ('x', 'y'),
                               axis_types=(AxisType.Explicit, AxisType.Explicit))

# Set up array with explicit sharding
arr = jax.random.normal(jax.random.PRNGKey(0), (8, 8, 8))
jax.set_mesh(mesh_explicit)
arr = reshard(arr, P('x', 'y'))

# Get the expected output sharding for the FFT
out_sharding = jd.get_fft_output_sharding(arr.sharding)

# Wrap jaxDecomp function with auto_axes
@auto_axes
def pfft3d_explicit_safe(x, out_sharding=out_sharding):
    return jd.fft.pfft3d(x)

# Now it works with Explicit mesh
result = pfft3d_explicit_safe(arr, out_sharding=out_sharding)
```

```{note}
The `auto_axes` decorator temporarily converts the mesh to `AxisType.Auto` for the duration of the wrapped function, allowing `custom_partitioning` to work correctly.
```

### cuDecomp Backend: Transposed Mesh Required

When using the cuDecomp backend, you must create a **transposed mesh**:

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, AxisType

pdims = (2, 4)
devices = mesh_utils.create_device_mesh(pdims)

# cuDecomp backend requires transposed mesh with ('y', 'x') axis names
mesh = Mesh(devices.T, ('y', 'x'), axis_types=(AxisType.Auto, AxisType.Auto))

# Note: axis_types defaults to Auto when using the Mesh constructor
mesh = Mesh(devices.T, ('y', 'x'))  # Auto by default
```
