# Transpositions in `jaxDecomp`

Transpositions are a core operation in `jaxDecomp`, enabling distributed 3D FFTs by realigning data across devices so that each axis can be processed locally. These are **global transposes**: they reshuffle slices of data between GPUs according to the domain decomposition layout.

---

## What is a Global Transpose?

In a distributed 3D FFT, the algorithm applies a series of 1D FFTs along different axes. Between each FFT, the array must be transposed so that the next axis becomes undistributed and locally contiguous.

For example:

```text
Start → FFT along Z
Transpose Z → Y
FFT along Y
Transpose Y → X
FFT along X
```

These transpositions change the mapping of the distributed axes while preserving the global data shape.


## Visual Illustration

The animation below shows how distributed pencils are rotated during a round-trip FFT. Each step reorients the domain decomposition for the next FFT axis.

![Animation of distributed transpositions in jaxDecomp](../joss-paper/assets/decomp2d.gif)



## Contiguous vs Non-Contiguous Transpositions

`jaxDecomp` supports two modes of transposition:

* **Contiguous**: The layout is physically reshuffled (e.g., changing from `ZXY` to `YZX`).
* **Non-contiguous**: The global axis order is preserved, but the device mapping changes.

In most cases, both perform similarly. Non-contiguous transposes are useful when the logical layout (e.g., for halo exchange or diagnostics) should remain unchanged.

it can be set to `False` by doing :

```python
jaxdecomp.config.update('transpose_axis_contiguous', False)
```


---

## API Example

```python
# Manually transpose a distributed array
y_pencil = jaxdecomp.transposeXtoY(x_pencil)
z_pencil = jaxdecomp.transposeYtoZ(y_pencil)
```

> **Note:** These functions are already called internally by `pfft3d` and `pifft3d`.
> You only need to use them directly for custom workflows—such as I/O reordering, diagnostics, or algorithms requiring specific axis alignments.

---

## Summary

* Transpositions are required to align each axis for local 1D FFTs in a distributed array.
* `jaxDecomp` provides high-level primitives for axis-aligned transpositions.
* Both contiguous and non-contiguous modes are supported and efficient.
* The transpose API is fully differentiable and JAX-compatible.

> 🔄 See [Distributed FFT](03-distributed_ffts.md) for how these transposes are used in `pfft3d`.
> 🧱 See [Domain Decomposition](02-decomposition.md) to understand how arrays are partitioned across GPUs.
