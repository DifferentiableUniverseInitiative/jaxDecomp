import numpy as np
import jaxlib.mlir.ir as ir
from jaxlib.mhlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax.interpreters import xla
from jax.interpreters import mlir
from jaxdecomp._src import _jaxdecomp
from jax import jit
from jax.lib import xla_client

from jax.interpreters import ad
from jax.interpreters import batching

from jax import lax
from typing import Union
from jax._src.api import jit, linear_transpose, ShapeDtypeStruct
from jax._src.numpy.util import _promote_dtypes_complex, _promote_dtypes_inexact

FftType = xla_client.FftType

_complex_dtype = lambda dtype: (np.zeros((), dtype) + np.zeros(
    (), np.complex64)).dtype
_real_dtype = lambda dtype: np.finfo(dtype).dtype
_is_even = lambda x: x % 2 == 0


def pfft_mhlo(a, dtype, *, fft_type: FftType, pdims, global_shape):
  """cuDecomp FFT kernel.
    This implementation is adapted from the logic in
    https://github.com/google/jax/blob/61aa4153567c96cfc2e2187773153d4a206c0639/jaxlib/ducc_fft.py#L110
    """
  a_type = ir.RankedTensorType(a.type)
  n = len(a_type.shape)

  # Figure out which fft we want
  forward = fft_type in (FftType.FFT, FftType.RFFT)
  real = fft_type in (FftType.RFFT, FftType.IRFFT)
  is_double = np.finfo(dtype).dtype == np.float64

  if fft_type == FftType.RFFT:
    assert dtype in (np.float32, np.float64), dtype
    out_dtype = np.dtype(np.complex64 if dtype == np.float32 else np.complex128)
    out_shape = list(a_type.shape)
    out_shape[-1] = out_shape[-1] // 2 + 1
  elif fft_type == FftType.IRFFT:
    assert np.issubdtype(dtype, np.complexfloating), dtype
    out_dtype = np.dtype(np.float32 if dtype == np.complex64 else np.float64)
    out_shape = list(a_type.shape)
    out_shape[-1] = global_shape[-1]
  else:
    assert np.issubdtype(dtype, np.complexfloating), dtype
    out_dtype = dtype
    out_shape = list(a_type.shape)

  if out_dtype == np.float32:
    out_type = ir.F32Type.get()
  elif out_dtype == np.float64:
    out_type = ir.F64Type.get()
  elif out_dtype == np.complex64:
    out_type = ir.ComplexType.get(ir.F32Type.get())
  elif out_dtype == np.complex128:
    out_type = ir.ComplexType.get(ir.F64Type.get())
  else:
    raise ValueError(f"Unknown output type {out_dtype}")

  config = _jaxdecomp.GridConfig()
  config.pdims = pdims
  config.gdims = global_shape
  config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
  config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P
  opaque = _jaxdecomp.build_fft_descriptor(config, forward, real, is_double)
  layout = tuple(range(n - 1, -1, -1))
  return custom_call(
      "pfft3d",
      [ir.RankedTensorType.get(out_shape, out_type)],
      operands=[a],
      operand_layouts=[layout],
      result_layouts=[layout],
      has_side_effect=True,
      operand_output_aliases= {} if real else {0: 0}, # In the real case, we don't reuse the input array as they don't have exactly the same size
      backend_config=opaque,
  )


def _str_to_fft_type(s: str) -> xla_client.FftType:
  if s in ("fft", "FFT"):
    return xla_client.FftType.FFT
  elif s in ("ifft", "IFFT"):
    return xla_client.FftType.IFFT
  elif s in ("rfft", "RFFT"):
    return xla_client.FftType.RFFT
  elif s in ("irfft", "IRFFT"):
    return xla_client.FftType.IRFFT
  else:
    raise ValueError(f"Unknown FFT type '{s}'")


@partial(jit, static_argnums=(1, 2, 3))
def pfft(x, fft_type: Union[xla_client.FftType, str], pdims, global_shape):
  if isinstance(fft_type, str):
    typ = _str_to_fft_type(fft_type)
  elif isinstance(fft_type, xla_client.FftType):
    typ = fft_type
  else:
    raise TypeError(f"Unknown FFT type value '{fft_type}'")

  if typ == xla_client.FftType.RFFT:
    if np.iscomplexobj(x):
      raise ValueError("only real valued inputs supported for rfft")
    (x,) = _promote_dtypes_inexact(x)
  else:
    (x,) = _promote_dtypes_complex(x)

  return pfft_p.bind(x, fft_type=typ, pdims=pdims, global_shape=global_shape)


def pfft_abstract_eval(x, fft_type, pdims, global_shape):
  if not _is_even(global_shape[-1]):
    raise ValueError(
        f"Only even arrays on the last dimension are currently supported")
  if fft_type == xla_client.FftType.RFFT:
    shape = (x.shape[:-1] + (global_shape[-1] // 2 + 1,))
    dtype = _complex_dtype(x.dtype)
  elif fft_type == xla_client.FftType.IRFFT:
    shape = x.shape[:-1] + global_shape[-1:]
    dtype = _real_dtype(x.dtype)
  else:
    shape = x.shape
    dtype = x.dtype
  # The results of the forward FFT are transposed actually
  # shape = (shape[1], shape[2], shape[0]) # TODO: figure this out!
  return x.update(shape=shape, dtype=dtype)


def pfft_lowering(ctx, x, *, fft_type, pdims, global_shape):
  (x_aval,) = ctx.avals_in
  return [
      pfft_mhlo(
          x,
          x_aval.dtype,
          fft_type=fft_type,
          pdims=pdims,
          global_shape=global_shape)
  ]


def _naive_rfft(x, pdims, global_shape):
  y = pfft(x, xla_client.FftType.FFT, pdims, global_shape)
  n = global_shape[-1]
  return y[..., :n // 2 + 1]


@partial(jit, static_argnums=1)
def _rfft_transpose(t, pdims, global_shape):
  # The transpose of RFFT can't be expressed only in terms of irfft. Instead of
  # manually building up larger twiddle matrices (which would increase the
  # asymptotic complexity and is also rather complicated), we rely JAX to
  # transpose a naive RFFT implementation.
  dummy_shape = t.shape[:-len(global_shape)] + global_shape
  dummy_primal = ShapeDtypeStruct(dummy_shape, _real_dtype(t.dtype))
  transpose = linear_transpose(
      partial(_naive_rfft, pdims, global_shape), dummy_primal)
  (result,) = transpose(t)
  assert result.dtype == _real_dtype(t.dtype), (result.dtype, t.dtype)
  return result


def _irfft_transpose(t, pdims, global_shape):
  # The transpose of IRFFT is the RFFT of the cotangent times a scaling
  # factor and a mask. The mask scales the cotangent for the Hermitian
  # symmetric components of the RFFT by a factor of two, since these components
  # are de-duplicated in the RFFT.
  x = pfft(t, xla_client.FftType.RFFT, pdims, global_shape)
  n = x.shape[-1]
  full = partial(lax.full_like, t, dtype=x.dtype)
  mask = lax.concatenate(
      [full(1.0, shape=(1,)),
       full(2.0, shape=(n - 2,)),
       full(1.0, shape=(1,))],
      dimension=0,
  )
  scale = 1 / np.prod(global_shape)
  out = scale * lax.expand_dims(mask, range(x.ndim - 1)) * x
  assert out.dtype == _complex_dtype(t.dtype), (out.dtype, t.dtype)
  # Use JAX's convention for complex gradients
  # https://github.com/google/jax/issues/6223#issuecomment-807740707
  return lax.conj(out)


def _fft_transpose_rule(t, operand, fft_type, pdims, global_shape):
  if fft_type == xla_client.FftType.RFFT:
    result = _rfft_transpose(t, pdims, global_shape)
  elif fft_type == xla_client.FftType.IRFFT:
    result = _irfft_transpose(t, pdims, global_shape)
  else:
    result = pfft(t, fft_type, pdims, global_shape)
  return (result,)


def _fft_batching_rule(batched_args, batch_dims, fft_type, pdims, global_shape):
  (x,) = batched_args
  (bd,) = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return pfft(x, pdims, fft_type, pdims, global_shape), 0


pfft_p = Primitive("pfft")
pfft_p.def_impl(partial(xla.apply_primitive, pfft_p))
pfft_p.def_abstract_eval(pfft_abstract_eval)
ad.deflinear2(pfft_p, _fft_transpose_rule)
mlir.register_lowering(pfft_p, pfft_lowering, platform="gpu")
batching.primitive_batchers[pfft_p] = _fft_batching_rule
