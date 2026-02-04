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
**Important (February 2026):** jaxDecomp is currently **not compatible** with `AxisType.Explicit` mesh axes. This is due to a limitation in JAX's `custom_partitioning` mechanism where XLA's SPMD partitioner modifies shardings differently for explicit axes.
```

### Use Auto Axis Types

Always use `AxisType.Auto` when working with jaxDecomp:

```python
from jax.sharding import AxisType

# Will NOT work with jaxDecomp (even with auto_axes wrapper)
mesh_explicit = jax.make_mesh(pdims, ('x', 'y'),
                               axis_types=(AxisType.Explicit, AxisType.Explicit))

# Use Auto axis types instead
mesh_auto = jax.make_mesh(pdims, ('x', 'y'),
                           axis_types=(AxisType.Auto, AxisType.Auto))
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
