# Understanding Domain Decomposition in `jaxDecomp`

`jaxDecomp` supports both **slab** and **pencil** domain decompositions through the `pdims` argument. This determines how your 3D array is partitioned across devices.

## What is `pdims`?

The `pdims` parameter defines the decomposition of your domain:

- `pdims=(1, N)` or `(N, 1)` → **Slab decomposition**
- `pdims=(M, N)` where both M, N > 1 → **Pencil decomposition**

## Slab vs Pencil: Tradeoffs

| Feature             | Slab Decomposition      | Pencil Decomposition     |
|---------------------|-------------------------|---------------------------|
| Faster per FFT call | ✅ Often faster         | ❌ Slightly slower        |
| Accuracy            | ⚠️ Can be slightly lower| ✅ Slightly better        |


## Slab Decomposition

Slab decomposition is typically faster per FFT and easier to configure. It works well for:

- Prototyping
- Small-to-medium simulations
- GPU-limited environments

```python
pdims = (1, 8)  # 8-GPU slab decomposition along the second axis
```

---

## Pencil Decomposition

Pencil decomposition enables better load balancing and scalability. It is ideal for:

* Large-scale simulations (e.g., 2048³ grids)
* Production workloads
* Higher accuracy in tightly coupled FFT pipelines

```python
pdims = (2, 4)  # 8-GPU pencil decomposition
```


## Dynamically Generating `pdims`

You can programmatically compute `pdims` based on the number of devices available:

```python
import jax

device_count = jax.device_count()  # Total number of GPUs
assert device_count % 2 == 0, "Need an even number of devices for 2D mesh"

# Example: 2 rows, N/2 columns
pdims = (2, device_count // 2)
```

You can experiment with other factorizations depending on your topology. The goal is to create a `pdims = (Px, Py)` such that `Px * Py == jax.device_count()`.


## Recommendation

There is no universal "best" choice — we recommend trying both. For most scientific simulations, the accuracy difference is small, but pencil decompositions are often more scalable in the long run.

## Creating the JAX Mesh and Sharding

Once `pdims` is defined, use it to create a JAX mesh and sharding spec:

```python
from jax.sharding import NamedSharding, PartitionSpec as P

mesh = jax.make_mesh(pdims, axis_names=('x', 'y'))

sharding = NamedSharding(mesh, P('x', 'y'))
```

This sharding object can then be used with distributed arrays, FFTs, halo exchanges, and more.


## TL;DR

* Use `pdims = (1, N)` for simpler and faster setups.
* Use `pdims = (M, N)` (M, N > 1) for large simulations that need scalability.
* Pencil decompositions require more transposes but enable more parallelism.
* Choose `pdims` based on your hardware and workload—there’s no one-size-fits-all.
