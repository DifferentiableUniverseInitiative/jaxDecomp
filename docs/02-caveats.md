
# Caveats and Workarounds: Autodiff + SPMD Sharding with `jaxDecomp`

This page explains some **known caveats** when using JAX’s automatic differentiation (AD) with the distributed FFT routines in `jaxDecomp`. Specifically, you may encounter errors when combining **SPMD sharding** and **AD transforms** such as `jax.grad`, `jax.jacfwd`, or `jax.jacrev`. Below, we show how to annotate your code to avoid these issues.

---

## 1. Background

- **SPMD Sharding in JAX**: When you run JAX on multiple devices (e.g., multiple GPUs or CPU devices), you can specify how arrays should be partitioned across those devices using a mesh and a sharding specification (`NamedSharding`, `PartitionSpec`, etc.).
- **AD Transforms**: JAX’s `jax.grad`, `jax.jacfwd`, and `jax.jacrev` automatically compute derivatives of your functions. Under the hood, JAX sometimes rewrites your function into a new function that can cause changes to sharding or lead to “unsharded” arrays.

In certain scenarios, JAX’s AD transformations might **lose** the sharding specification if the function’s first operation is a parallel operation (like `pfft3d`). This can trigger errors like:

```
Input sharding was found to be None while lowering the SPMD rule.
You are likely calling jacfwd with pfft as the first function.
```

---

## 2. `jacfwd` with Parallel FFT

### Problem

Consider the following function, which calls `pfft3d` immediately:

```python
def forward(a):
    return jaxdecomp.fft.pfft3d(a).real
```

If we attempt:

```python
jax.jacfwd(forward)(a)
```

we will encouter this error:
```
Input sharding was found to be None while lowering the SPMD rule.
You are likely calling jacfwd with pfft as the first function.
due to a bug in JAX, the sharding is not correctly passed to the SPMD rule.
```

### Workaround

By **annotating** the input array’s sharding *explicitly* within the function we differentiate, we ensure JAX does not lose the sharding information. For instance:

```python
import jax
import jax.numpy as jnp
from jax import lax
import jaxdecomp

# Suppose we have a sharding object named `sharding`.
# In your real code, you might do something like:
#    mesh = jax.make_mesh((1, 8), axis_names=('x','y'))
#    sharding = NamedSharding(mesh, P('x', 'y'))

def annotated_forward(a):
    # explicitly ensure 'a' is recognized as sharded
    a = lax.with_sharding_constraint(a, sharding)
    return jaxdecomp.fft.pfft3d(a).real

# Now jacfwd works without losing the sharding:
jax.jacfwd(annotated_forward)(a)
```

---

## 3. `jacrev` with Parallel FFT

### Problem

When computing reverse-mode Jacobians (`jax.jacrev`), a similar issue can arise. If our function is:

```python
def forward(a):
   return jaxdecomp.fft.pfft3d(a).real
```

Then:

```python
jax.jacrev(forward)(a)
```

can cause JAX to replicate the array or fail the sharding constraint. We might see an unexpected result like a fully replicated array (`SingleDeviceSharding`), or an error about “Input sharding was found to be None ...”.

### Workaround

Again, we can **annotate** the function:

```python
def annotated_forward(a):
    a = lax.with_sharding_constraint(a, sharding)
    return jaxdecomp.fft.pfft3d(a).real

# Now jacrev retains correct sharding
rev_jac = jax.jacrev(annotated_forward)(a)
```

You can verify the resulting array’s sharding with:
```python
print(rev_jac.sharding)
```

---

## 4. `grad` of a Scalar-Reduced FFT

### Problem

When your function returns a scalar (e.g., via `jnp.sum` of the FFT output), the gradient pipeline might fail with the same “Input sharding was found to be None” error. For example:

```python
def fft_reduce(a):
    return jaxdecomp.fft.pfft3d(a).real.sum()

jax.grad(fft_reduce)(a)
```

can fail for the same reason: the initial pfft step is ambiguous to JAX’s SPMD rule.

### Workaround

1. **Perform `pfft3d`**,
2. **Annotate** the output array’s new sharding,
3. Then reduce.

Example:

```python
def fft_reduce_with_annotation(a):
    # Perform FFT
    res = jaxdecomp.fft.pfft3d(a).real
    # Annotate the resulting array with the sharding that pfft3d produces:
    out_sharding = jaxdecomp.get_fft_output_sharding(sharding)
    res = lax.with_sharding_constraint(res, out_sharding)
    # Now reduce to scalar
    return res.sum()

# This will now run successfully
grad_val = jax.grad(fft_reduce_with_annotation)(a)
```

---

## 5. Summary of Best Practices

1. **Annotate Inputs**
   If your function starts with `pfft3d(...)`, insert a `lax.with_sharding_constraint(input_array, sharding)` to ensure JAX retains the correct distribution info during AD transforms.

2. **Annotate Outputs**
   For scalar-reduction patterns (`.sum()`, `.mean()`, etc.), or any time the output shape differs significantly from the input, use `lax.with_sharding_constraint(output_array, new_sharding)` to ensure the partial derivatives keep correct partitioning.

3. **Check Sharding**
   Inspect the `.sharding` attribute of returned arrays after `jax.jacrev`, `jax.jacfwd`, or `jax.grad` to confirm that the output is still sharded the way you intend.

---

## 6. Conclusion

Due to a **bug** in how JAX’s AD transforms currently interact with SPMD partitioning, you may need to explicitly annotate sharding constraints around FFT calls. By applying `lax.with_sharding_constraint` or by retrieving the FFT’s “expected” output sharding (via `jaxdecomp.get_fft_output_sharding`), you can ensure that your distributed computations remain partitioned as expected.

Feel free to open an issue on GitHub if you encounter other scenarios where sharding + AD transforms produce unexpected results!
