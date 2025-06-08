
# Halo Exchange in `jaxDecomp`

In distributed simulations, each subdomain typically only has access to its own portion of the global array. However, many numerical methods—such as stencil operations, particle-mesh solvers, and finite-difference PDEs—require values from neighboring subdomains at the domain boundaries.

To handle this, `jaxDecomp` provides a high-performance, differentiable **halo exchange** operator that synchronizes "ghost zones" between neighboring slices. This is essential for maintaining correctness across device boundaries in domain-decomposed arrays.


## Halo Exchange Process

For each axis, `jaxDecomp` performs a **bidirectional halo exchange**, meaning that every subdomain sends and receives a slice of width equal to the **halo extent** from its neighbors in both directions.

This process is illustrated below:

![Visualization of the distributed halo exchange process in `jaxDecomp`](assets/halo-exchange.svg)

### Index Ranges

The exchanged regions follow a simple, symmetric pattern:

| Direction            | Sent Range (from current slice)         | Received Range (into current slice)         |
|----------------------|------------------------------------------|---------------------------------------------|
| To next neighbor     | $[S - 2h : S - h]$                       | $[0 : h]$ (from previous neighbor)          |
| To previous neighbor | $[h : 2h]$                               | $[S - h : S]$ (from next neighbor)          |

Where:
- $S$ is the local array size along the axis
- $h$ is the **halo extent**


## Boundary Conditions

`jaxDecomp` supports both:

- **Periodic boundaries**: Values wrap around the global array edges.
- **Non-periodic boundaries**: Halo slices are zeroed out (or ignored) when no neighbor is present.

You can control this with the `halo_periods` argument:

```python
halo_extents = (16, 16)
halo_periods = (True, False)

z = jaxdecomp.halo_exchange(z, halo_extents=halo_extents, halo_periods=halo_periods)
````

---

## Padding and Halo Exchange

If you have padded your array (e.g. for convolution buffers or guard zones), the **halo region is taken from inside the unpadded array** and sent to the appropriate location in the padded version.

This ensures that halo data is correctly aligned and avoids contamination from padding.

---

## Implementation Details

Under the hood, `jaxDecomp` uses `jax.lax.ppermute` to perform efficient device-to-device communication in parallel. This enables halo exchange to work seamlessly with JAX transformations like `jit`, 

## Summary

* Halo exchange enables boundary-aware computations in distributed arrays.
* `jaxDecomp` supports periodic and non-periodic boundaries.
* Halo slices are taken from inside the array, not from padding zones.
* Internally built on `lax.ppermute`, compatible with JAX transformations.

> 🔎 See the [Basic Usage](01-basic_usage.md) example to learn how to combine halo exchange with FFTs in practice.

