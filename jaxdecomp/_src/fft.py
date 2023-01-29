import numpy as np
import jaxlib.mlir.ir as ir
from jaxlib.hlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax.interpreters import xla
from jax.interpreters import mlir
import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jax import jit
from jax.lib import xla_client

from jax._src.lib.mlir.dialects import mhlo
import jax
from jax.interpreters import ad

from typing import Union
from jax._src.api import jit
from jax._src.numpy.util import _promote_dtypes_complex

FftType = xla_client.FftType


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


@partial(jit, static_argnums=(1, 2, 3, 4))
def pfft(x,
         fft_type: Union[xla_client.FftType, str],
         pdims,
         global_shape,
         adjoint=False):
  if isinstance(fft_type, str):
    typ = _str_to_fft_type(fft_type)
  elif isinstance(fft_type, xla_client.FftType):
    typ = fft_type
  else:
    raise TypeError(f"Unknown FFT type value '{fft_type}'")

  if typ in [xla_client.FftType.RFFT, xla_client.FftType.IRFFT]:
    raise TypeError("only complex FFTs are currently supported through pfft.")

  (x,) = _promote_dtypes_complex(x)

  return pfft_p.bind(
      x, fft_type=typ, pdims=pdims, global_shape=global_shape, adjoint=adjoint)


def pfft_abstract_eval(x, fft_type, pdims, global_shape, adjoint):

  out_global_shape = global_shape

  # Figure out what is the pencil decomposition at the output
  axis = 0
  if fft_type in [xla_client.FftType.RFFT, xla_client.FftType.FFT]:
    axis = 2
  config = _jaxdecomp.GridConfig()
  config.pdims = pdims
  # Dimensions are actually in reverse order due to Fortran indexing at the cuDecomp level
  config.gdims = out_global_shape[::-1]
  config.halo_comm_backend = jaxdecomp.config.halo_comm_backend
  config.transpose_comm_backend = jaxdecomp.config.transpose_comm_backend
  pencil = _jaxdecomp.get_pencil_info(config, axis)
  # Dimensions are actually in reverse order due to Fortran indexing at the cuDecomp level
  shape = pencil.shape[::-1]
  assert np.prod(shape) == np.prod(
      x.shape
  ), "Only array dimensions divisible by the process mesh size are currently supported. The current configuration leads to local slices of varying sizes between forward and reverse FFT."

  return x.update(shape=shape, dtype=x.dtype)


def pfft_lowering(ctx, a, *, fft_type, pdims, global_shape, adjoint):
  (x_aval,) = ctx.avals_in
  (aval_out,) = ctx.avals_out
  dtype = x_aval.dtype
  a_type = ir.RankedTensorType(a.type)
  n = len(a_type.shape)

  # We currently only support complex FFTs through this interface, so let's check the fft type
  assert fft_type in (FftType.FFT,
                      FftType.IFFT), "Only complex FFTs are currently supported"

  # Figure out which fft we want
  forward = fft_type in (FftType.FFT,)
  is_double = np.finfo(dtype).dtype == np.float64

  # Compute the descriptor for our FFT
  config = _jaxdecomp.GridConfig()
  config.pdims = pdims
  config.gdims = global_shape[::-1]
  config.halo_comm_backend = jaxdecomp.config.halo_comm_backend
  config.transpose_comm_backend = jaxdecomp.config.transpose_comm_backend
  workspace_size, opaque = _jaxdecomp.build_fft_descriptor(
      config, forward, is_double, adjoint)
  layout = tuple(range(n - 1, -1, -1))

  # We ask XLA to allocate a workspace for this operation.
  # TODO: check that the memory is not used all the time, just when needed
  workspace = mlir.full_like_aval(
      0, jax.core.ShapedArray(shape=[workspace_size], dtype=np.byte))

  # Run the custom op with same input and output shape, so that we can perform operations
  # inplace.
  result = custom_call(
      "pfft3d",
      [a_type],
      operands=[a, workspace],
      operand_layouts=[layout, (0,)],
      result_layouts=[layout],
      has_side_effect=True,
      operand_output_aliases={0: 0},
      backend_config=opaque,
  )

  # Finally we reshape the arry to the expected shape.
  return mhlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results


def _fft_transpose_rule(x, operand, fft_type, pdims, global_shape, adjoint):
  assert fft_type in [FftType.FFT, FftType.IFFT]
  if fft_type == FftType.FFT:
    result = pfft(x, FftType.IFFT, pdims, global_shape, ~adjoint)
  elif fft_type == FftType.IFFT:
    result = pfft(x, FftType.FFT, pdims, global_shape, ~adjoint)
  else:
    raise NotImplementedError

  return (result,)


pfft_p = Primitive("pfft")
pfft_p.def_impl(partial(xla.apply_primitive, pfft_p))
pfft_p.def_abstract_eval(pfft_abstract_eval)
ad.deflinear2(pfft_p, _fft_transpose_rule)
mlir.register_lowering(pfft_p, pfft_lowering, platform="gpu")
