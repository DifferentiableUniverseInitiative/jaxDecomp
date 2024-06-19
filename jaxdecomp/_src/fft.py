from functools import partial
from typing import Tuple, Union

import jax
import jaxlib.mlir.ir as ir
import numpy as np
from jax import ShapeDtypeStruct
from jax._src.interpreters import mlir
from jax._src.lib.mlir.dialects import hlo
from jax._src.numpy.util import promote_dtypes_complex
from jax.core import Primitive, ShapedArray
from jax.interpreters import ad, mlir, xla
from jax.interpreters.mlir import hlo
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

import jaxdecomp
from jaxdecomp._src import _jaxdecomp

FftType = xla_client.FftType
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from jaxdecomp._src.spmd_ops import (BasePrimitive, get_axis_size,
                                     register_primitive)


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


class FFTPrimitive(BasePrimitive):

  name = "fft"
  multiple_results = False
  impl_static_args = (1,)
  inner_primitive = None
  outer_primitive = None

  @staticmethod
  def abstract(x, fft_type, pdims, global_shape, adjoint):

    if global_shape == x.shape:
      return FFTPrimitive.outer_abstract(x, fft_type=fft_type, adjoint=adjoint)

    match fft_type:
      case xla_client.FftType.FFT:
        # FFT is X to Y to Z so Z-Pencil is returned
        # Except if we are doing a YZ slab in which case we return a Y-Pencil
        transpose_shape = (1, 2, 0)
        transposed_pdims = pdims
      case xla_client.FftType.IFFT:
        # IFFT is Z to X to Y so X-Pencil is returned
        # In YZ slab case we only need one transposition back to get the X-Pencil
        transpose_shape = (2, 0, 1)
        transposed_pdims = pdims
      case _:
        raise TypeError(
            "only complex FFTs are currently supported through pfft.")

    output_shape = (global_shape[transpose_shape[0]] // transposed_pdims[1],
                    global_shape[transpose_shape[1]] // transposed_pdims[0],
                    global_shape[transpose_shape[2]])

    return ShapedArray(output_shape, x.dtype)

  @staticmethod
  def outer_abstract(x, fft_type, adjoint):

    match fft_type:
      case xla_client.FftType.FFT:
        # FFT is X to Y to Z so Z-Pencil is returned
        # Except if we are doing a YZ slab in which case we return a Y-Pencil
        transpose_shape = (1, 2, 0)
      case xla_client.FftType.IFFT:
        # IFFT is Z to X to Y so X-Pencil is returned
        # In YZ slab case we only need one transposition back to get the X-Pencil
        transpose_shape = (2, 0, 1)
      case _:
        raise TypeError(
            "only complex FFTs are currently supported through pfft.")

    output_shape = tuple([x.shape[i] for i in transpose_shape])
    return ShapedArray(output_shape, x.dtype)

  @staticmethod
  def lowering(ctx, a, *, fft_type, pdims, global_shape, adjoint):
    (x_aval,) = ctx.avals_in
    (aval_out,) = ctx.avals_out
    dtype = x_aval.dtype
    a_type = ir.RankedTensorType(a.type)
    # We currently only support complex FFTs through this interface, so let's check the fft type
    assert fft_type in (
        FftType.FFT, FftType.IFFT), "Only complex FFTs are currently supported"

    # Figure out which fft we want
    forward = fft_type in (FftType.FFT,)
    is_double = np.finfo(dtype).dtype == np.float64

    # Get original global shape
    match fft_type:
      case xla_client.FftType.FFT:
        transpose_back_shape = (0, 1, 2)
      case xla_client.FftType.IFFT:
        transpose_back_shape = (2, 0, 1)
      case _:
        raise TypeError(
            "only complex FFTs are currently supported through pfft.")
    # Make sure to get back the original shape of the X-Pencil
    global_shape = tuple([global_shape[i] for i in transpose_back_shape])
    # Compute the descriptor for our FFT
    config = _jaxdecomp.GridConfig()

    config.pdims = pdims
    config.gdims = global_shape[::-1]
    config.halo_comm_backend = jaxdecomp.config.halo_comm_backend
    config.transpose_comm_backend = jaxdecomp.config.transpose_comm_backend
    workspace_size, opaque = _jaxdecomp.build_fft_descriptor(
        config, forward, is_double, adjoint)

    n = len(a_type.shape)
    layout = tuple(range(n - 1, -1, -1))

    # We ask XLA to allocate a workspace for this operation.
    # TODO: check that the memory is not used all the time, just when needed
    workspace = mlir.full_like_aval(
        ctx, 0, jax.core.ShapedArray(shape=[workspace_size], dtype=np.byte))

    # Run the custom op with same input and output shape, so that we can perform operations
    # inplace.
    result = custom_call(
        "pfft3d",
        result_types=[a_type],
        operands=[a, workspace],
        operand_layouts=[layout, (0,)],
        result_layouts=[layout],
        has_side_effect=True,
        operand_output_aliases={0: 0},
        backend_config=opaque,
    )

    # Finally we reshape the arry to the expected shape.
    return hlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results

  # Note : This must no longer be jitted because it will have single device abstract shapes
  # The actual jit is done in the pfft function in lower_fn
  # This means that sfft should never be lowered as is and only be lowered in the context of pfft
  @staticmethod
  def impl(x, fft_type, adjoint):

    if isinstance(fft_type, str):
      typ = _str_to_fft_type(fft_type)
    elif isinstance(fft_type, xla_client.FftType):
      typ = fft_type
    else:
      raise TypeError(f"Unknown FFT type value '{fft_type}'")

    if typ in [xla_client.FftType.RFFT, xla_client.FftType.IRFFT]:
      raise TypeError("only complex FFTs are currently supported through pfft.")

    (x,) = promote_dtypes_complex(x)

    pdims = (1, jax.device_count())
    global_shape = x.shape

    return FFTPrimitive.inner_primitive.bind(
        x,
        fft_type=typ,
        pdims=pdims,
        global_shape=global_shape,
        adjoint=adjoint)

  @staticmethod
  def per_shard_impl(x, kind, pdims, global_shape, adjoint):
    return FFTPrimitive.inner_primitive.bind(
        x, kind=kind, pdims=pdims, global_shape=global_shape, adjoint=adjoint)

  @staticmethod
  def infer_sharding_from_operands(fft_type, adjoint, mesh: Mesh,
                                   arg_infos: Tuple[ShapeDtypeStruct],
                                   result_infos: Tuple[ShapedArray]):
    """
    Tell XLA how to infer the sharding of the output from the input sharding.

    Args:
        mesh (Mesh): The contextual mesh

        arg_shapes (tuple): A tuple of ShapeDtypeStruct that contains the shape and the sharding of each input operand

        result_shape (ShapedArray) : a single ShapedArray reprsenting a single output without the sharding information

    Returns:

        result_sharding (XLACompatibleSharding): The sharding result for example a NamedSharding.

    """
    input_sharding = arg_infos[0].sharding
    return NamedSharding(mesh, P(*input_sharding.spec))

  @staticmethod
  def partition(fft_type, adjoint, mesh, arg_shapes, result_shape):
    """
      Tells XLA how to partition the primitive

      Args:
          mesh (Mesh): The contextual mesh

          arg_shapes (tuple): A tuple of ShapeDtypeStruct that contains the shape and the sharding of each input operand

          result_shape (ShapeDtypeStruct) : a ShapeDtypeStruct reprsenting a single output

      Returns:
          Mesh (Mesh) : The mesh.

          function: The lowered function, to allow the user to redefine how the primitive is called in a context of a specific sharding

          result_sharding (XLACompatibleSharding): The sharding result for example a NamedSharding.

          arg_shardings (tuple): a tuple of all XLACompatibleSharding of the input operands
      """

    # pfft only has one operand
    input_sharding = NamedSharding(mesh, P(*arg_shapes[0].sharding.spec))
    output_sharding = NamedSharding(mesh, P(*result_shape.sharding.spec))

    pdims = (get_axis_size(input_sharding, 1), get_axis_size(input_sharding, 0))
    global_shape = arg_shapes[0].shape

    impl = partial(
        FFTPrimitive.per_shard_impl,
        fft_type=fft_type,
        pdims=pdims,
        global_shape=global_shape)

    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(FFTPrimitive)


def pfft_p_lower(x, fft_type, adjoint=False):
  return FFTPrimitive.outer_primitive.bind(
      x, fft_type=fft_type, adjoint=adjoint)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def pfft(x, fft_type, adjoint):
  output, _ = _pfft_fwd_rule(x, fft_type=fft_type, adjoint=adjoint)
  return output


def _pfft_fwd_rule(x, fft_type: str, adjoint: bool = False):
  # Linear function has no residuals
  return pfft_p_lower(x, fft_type=fft_type, adjoint=adjoint), None


def _pfft_bwd_rule(fft_type, adjoint, ctx, g):

  assert fft_type in [FftType.FFT, FftType.IFFT]
  if fft_type == FftType.FFT:
    fft_type = FftType.IFFT
  elif fft_type == FftType.IFFT:
    fft_type = FftType.FFT

  return pfft_p_lower(g, fft_type, ~adjoint),


pfft.defvjp(_pfft_fwd_rule, _pfft_bwd_rule)
