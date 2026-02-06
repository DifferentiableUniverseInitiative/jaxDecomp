### Description


When using `custom_partitioning` with a mesh that has `AxisType.Explicit` axes, the callbacks (`partition`, `sharding_rule`, `infer_sharding_from_operands`) receive a mesh where axis types have been converted to `AxisType.Auto`. However, Shardy/XLA still makes partitioning decisions based on the original `Explicit` axis types, leading to inconsistent behavior and errors.

## Reproduction

```python
import os
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P, AxisType

def my_transpose_impl(x):
    return x.T

my_op = custom_partitioning(my_transpose_impl, static_argnums=())

def partition_fn(mesh, arg_shapes, result_shape):
    print(f'mesh.axis_types: {mesh.axis_types}')  # Always (Auto, Auto)

    arg_shardings = jax.tree.map(lambda s: s.sharding, arg_shapes)
    result_sharding = result_shape.sharding

    print(f'arg_shardings[0]: {arg_shardings[0]}')
    print(f'result_sharding: {result_sharding}')

    def lower_fn(x):
        x = jax.lax.all_to_all(x, axis_name='p', split_axis=1, concat_axis=0, tiled=True)
        return x.T

    return mesh, lower_fn, result_sharding, arg_shardings

def shardy_rule(mesh, arg_shapes, result_infos):
    print(f'shardy_rule mesh.axis_types: {mesh.axis_types}')  # Always (Auto, Auto)
    return 'p n -> p n'

my_op.def_partition(partition=partition_fn, sharding_rule=shardy_rule)

# Test with Auto (works)
mesh_auto = jax.make_mesh((2, 2), ('p', 'n'), axis_types=(AxisType.Auto, AxisType.Auto))
x = jnp.arange(64).reshape(8, 8).astype(jnp.float32)
x_sharded = jax.device_put(x, NamedSharding(mesh_auto, P('p', None)))

@jax.jit
def fn(x):
    return my_op(x)

result = fn(x_sharded)  # Works

# Test with Explicit (fails)
mesh_explicit = jax.make_mesh((2, 2), ('p', 'n'), axis_types=(AxisType.Explicit, AxisType.Explicit))
x_sharded = jax.device_put(x, NamedSharding(mesh_explicit, P('p', None)))
result = fn(x_sharded)  # Fails with shape mismatch
```

## Observed Behavior

**With `AxisType.Auto`:**
```
mesh.axis_types: (Auto, Auto)
arg_shardings[0]: NamedSharding(mesh=..., spec=PartitionSpec('p', None))
result_sharding: NamedSharding(mesh=..., spec=PartitionSpec('p', None))
```
Result: Works correctly.

**With `AxisType.Explicit`:**
```
mesh.axis_types: (Auto, Auto)  # <-- Converted to Auto!
arg_shardings[0]: NamedSharding(mesh=..., spec=PartitionSpec('p', None))
result_sharding: NamedSharding(mesh=..., spec=PartitionSpec(None, 'p'))  # <-- Different!
```
Result: `ValueError: Mismatch in result shapes. [ShapedArray(float32[4,8])] vs [ShapedArray(float32[8,4])]`

## Root Cause

1. The mesh passed to `custom_partitioning` callbacks has its `axis_types` converted from `Explicit` to `Auto`
2. However, Shardy/XLA's SPMD partitioner still uses the original `Explicit` axis types when deciding the `result_sharding`
3. This causes the `result_sharding` to differ between Auto and Explicit cases
4. The partition callback cannot adapt because it doesn't see the true axis types

## Expected Behavior

Either:
1. The callbacks should receive the mesh with the **original** `axis_types` preserved, OR
2. The result sharding inferred by Shardy should be consistent regardless of axis type

## Impact

This makes `custom_partitioning` incompatible with `AxisType.Explicit` meshes. Libraries like [jaxDecomp](https://github.com/DifferentiableUniverseInitiative/jaxDecomp) that rely on `custom_partitioning` for distributed operations cannot support explicit axis sharding.

## Environment

- JAX version: 0.9.0
- Shardy enabled: True
- Python: 3.11
- Platform: CPU (reproduced with `--xla_force_host_platform_device_count=4`)

