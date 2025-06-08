# Distributed 3D FFTs in `jaxDecomp`

`jaxDecomp` implements distributed 3D Fast Fourier Transforms (FFTs) by chaining local 1D FFTs along each axis of a 3D array, interleaved with global data transpositions. This allows large 3D grids to be split across multiple GPUs, enabling high-performance simulation workloads to scale well beyond single-device memory limits—all while staying differentiable and compatible with JAX's transformation system.

This page describes how the distributed FFT algorithm works in `jaxDecomp`, including the use of slab and pencil decomposition strategies, global transpositions, and layout transformations.

---

## Overview of the FFT Algorithm

The distributed 3D FFT is performed in three steps:

1. Apply a 1D FFT along the **Z-axis** (which is initially undistributed).
2. **Transpose** the array so that the Y-axis becomes undistributed.
3. Apply a 1D FFT along the **Y-axis**.
4. **Transpose** again to make the X-axis undistributed.
5. Apply a 1D FFT along the **X-axis**.

This ensures that each 1D FFT is applied to a locally contiguous, undistributed axis, while transpositions handle the redistribution of data between steps.

> 🔄 This same logic applies for the inverse FFT (`pifft3d`) but in reverse order.

![Visualization of the 3D FFT algorithm in `jaxDecomp`, including forward and backward passes via axis-aligned transpositions and local 1D FFTs.](../joss-paper/assets/fft.svg)

---

## Transpositions Between Axes

Transpositions are needed to reshuffle the distributed dimensions so the next axis is undistributed. Here's the typical forward FFT sequence in pencil decomposition:

| Step            | Operation Description                                      |
|------------------|------------------------------------------------------------|
| FFT along Z      | Batched 1D FFT on the Z-axis (undistributed).             |
| Transpose Z→Y    | Redistribute to align Y-axis as undistributed.            |
| FFT along Y      | Batched 1D FFT on the Y-axis.                             |
| Transpose Y→X    | Redistribute to align X-axis as undistributed.            |
| FFT along X      | Batched 1D FFT on the X-axis.                             |

These transpositions are global communications across devices, handled via cuDecomp using NCCL, MPI, or NVSHMEM.

---

## Pencil Decomposition Strategy

With pencil decomposition (`pdims=(Px, Py)` where both `Px > 1`, `Py > 1`), all three FFTs are performed in sequence with two transpositions between them.

| Step             | Origin Shape                                  | Target Shape                                   |
|------------------|------------------------------------------------|------------------------------------------------|
| Transpose Z→Y    | $\frac{X}{P_x} \times \frac{Y}{P_y} \times Z$ | $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$ |
| Transpose Y→X    | $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$ | $\frac{Y}{P_x} \times \frac{Z}{P_y} \times X$ |
| Transpose X→Y    | $\frac{Y}{P_x} \times \frac{Z}{P_y} \times X$ | $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$ |
| Transpose Y→Z    | $\frac{Z}{P_y} \times \frac{X}{P_x} \times Y$ | $\frac{X}{P_x} \times \frac{Y}{P_y} \times Z$ |

This layout supports large-scale parallelism and is most useful for big grids on many GPUs.

---

## Slab Decomposition Strategy

Slab decomposition uses a single-axis split (`pdims=(1, N)` or `(N, 1)`), enabling a simpler transposition scheme. It often requires just one transposition and supports hybrid 1D/2D FFTs.

Example decomposition:
| Step            | Shape                                          | FFT Feasibility             |
|-----------------|------------------------------------------------|-----------------------------|
| Initial         | $X \times \frac{Y}{P_y} \times Z$              | 1D FFT on Z                 |
| Transpose Z→Y   | $\frac{Z}{P_y} \times X \times Y$              | 2D FFT on YX                |

For large-scale slab use cases, we often apply a coordinate transformation to reduce the number of transpose steps.

---

### Coordinate Transformation in Slab Mode

In some slab decompositions (e.g., `pdims=(N,1)`), `jaxDecomp` reinterprets the axes to simplify the FFT steps. For example:

| Step            | Shape                                             | Interpretation                            |
|-----------------|--------------------------------------------------|-------------------------------------------|
| Initial         | $\frac{Z}{P_x} \times X \times Y$                | Equivalent to $X \times Y \times \frac{Z}{P_x}$ |
| Transpose Y→Z   | $X \times \frac{Y}{P_x} \times Z$                | Enables 1D FFT on X                       |
| Transpose Z→Y   | $\frac{Z}{P_x} \times X \times Y$                | Restore original layout                   |

This minimizes communication steps and improves performance in slab-based runs.

---

## Non-Contiguous Global Transpositions

In many workflows (e.g., halo exchanges), it’s useful to preserve the axis order across devices. `jaxDecomp` supports **non-contiguous transposes** that avoid changing the logical axis names.

it can be set to `False` by doing : 

```python
jaxdecomp.config.update('transpose_axis_contiguous', False)
```

Example transpositions (keeping axis order as `X, Y, Z`):

| Step          | Input Shape                                | Output Shape                               |
|---------------|---------------------------------------------|---------------------------------------------|
| Transpose Z→Y | $\frac{X}{P_x} \times \frac{Y}{P_y} \times Z$ | $\frac{X}{P_x} \times Y \times \frac{Z}{P_y}$ |
| Transpose Y→X | $\frac{X}{P_x} \times Y \times \frac{Z}{P_y}$ | $X \times \frac{Y}{P_x} \times \frac{Z}{P_y}$ |
| ...           | ...                                         | ...                                         |

> 🧪 Benchmark Note: We observed no major performance difference between contiguous and non-contiguous layouts, so we recommend using whichever simplifies your pipeline.

---

## Summary

- `jaxDecomp` performs 3D FFTs by combining local 1D FFTs with global data transpositions.
- Pencil decompositions require two global transposes; slabs typically require one.
- Coordinate transformations can reduce communication cost in slab layouts.
- Non-contiguous transposes are supported and often easier to work with.
- All operations are compatible with JAX transformations (`jit`, `grad`, etc.) and support multiple backends (NCCL, MPI, NVSHMEM).

For more on choosing decomposition layouts, see [Domain Decomposition](02-decomposition.md).
