from functools import partial
from typing import Tuple

import jax
import jaxlib.mlir.ir as ir
import numpy as np
from jax import ShapeDtypeStruct
from jax._src.interpreters import mlir
from jax._src.lib.mlir.dialects import hlo
from jax._src.numpy.util import promote_dtypes_complex
from jax._src.typing import Array
from jax.core import Primitive, ShapedArray
from jax.lib import xla_client
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib.hlo_helpers import custom_call

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.fft_utils import fftn
from jaxdecomp._src.pencil_utils import (get_lowering_args, get_output_specs,
                                         get_pencil_type, get_transpose_order)
from jaxdecomp._src.spmd_ops import (BasePrimitive, get_pdims_from_sharding,
                                     register_primitive)

FftType = xla_client.FftType
GdimsType = Tuple[int, int, int]
# pdims is only two integers
# but in some cases we need to represent ('x' , None , 'y') as (Nx , 1 , Ny)
PdimsType = Tuple[int, int, int]


class FFTPrimitive(BasePrimitive):
  """
  Custom primitive for FFT operations.
  """

  name = "fft"
  multiple_results = False
  impl_static_args = (1, 2)
  inner_primitive = None
  outer_primitive = None

  @staticmethod
  def abstract(x: Array, fft_type: FftType, pdims: PdimsType,
               global_shape: GdimsType, adjoint: bool,
               mesh: Mesh) -> ShapedArray:
    """
    Abstract function to compute the shape of FFT output.

    Parameters
    ----------
    x : Array
      Input array.
    fft_type : xla_client.FftType
      Type of FFT operation.
    pdims : Tuple[int, int]
      Parallel dimensions.
    global_shape : Tuple[int, int, int]
      Global shape of the array.
    adjoint : bool
      Whether to compute the adjoint FFT.

    Returns
    -------
    ShapedArray
      Shape of the output array.
    """
    if global_shape == x.shape:
      return FFTPrimitive.outer_abstract(x, fft_type=fft_type, adjoint=adjoint)

    transpose_shape = get_transpose_order(fft_type)

    output_shape = (global_shape[transpose_shape[0]] // pdims[0],
                    global_shape[transpose_shape[1]] // pdims[1],
                    global_shape[transpose_shape[2]] // pdims[2])

    return ShapedArray(output_shape, x.dtype)

  @staticmethod
  def outer_abstract(x: Array, fft_type: FftType, adjoint: bool) -> ShapedArray:
    """
    Abstract function for outer FFT operation.

    Parameters
    ----------
    x : Array
      Input array.
    fft_type : xla_client.FftType
      Type of FFT operation.
    adjoint : bool
      Whether to compute the adjoint FFT.

    Returns
    -------
    ShapedArray
      Shape of the output array.
    """
    del adjoint

    transpose_shape = get_transpose_order(fft_type)

    output_shape = tuple([x.shape[i] for i in transpose_shape])
    return ShapedArray(output_shape, x.dtype)

  @staticmethod
  def lowering(ctx, a: Array, *, fft_type: xla_client.FftType, pdims: PdimsType,
               global_shape: GdimsType, adjoint: bool, mesh: Mesh):
    """
    Lowering function for FFT primitive.

    Parameters
    ----------
    ctx
      Context.
    a : Primitive
      Input primitive.
    fft_type : xla_client.FftType
      Type of FFT operation.
    pdims : Tuple[int, int]
      Parallel dimensions.
    global_shape : Tuple[int, int, int]
      Global shape of the array.
    adjoint : bool
      Whether to compute the adjoint FFT.

    Returns
    -------
    list
      List of results from the operation.
    """
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
    pencil_type = get_pencil_type(mesh)
    pdims, global_shape = get_lowering_args(fft_type, global_shape, mesh)
    print(f"lowering with pdims {pdims} and globalshape {global_shape}")
    local_transpose = jaxdecomp.config.transpose_axis_contiguous
    # Compute the descriptor for our FFT
    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = global_shape[::-1]
    config.halo_comm_backend = jaxdecomp.config.halo_comm_backend
    config.transpose_comm_backend = jaxdecomp.config.transpose_comm_backend
    workspace_size, opaque = _jaxdecomp.build_fft_descriptor(
        config, forward, is_double, adjoint, local_transpose, pencil_type)

    n = len(a_type.shape)
    layout = tuple(range(n - 1, -1, -1))

    # We ask XLA to allocate a workspace for this operation.
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

  @staticmethod
  def impl(x: Array, fft_type: FftType, adjoint: bool) -> Array:
    """
    Implementation function for FFT primitive.

    Parameters
    ----------
    x : Array
      Input array.
    fft_type : Union[str, xla_client.FftType]
      Type of FFT operation.
    adjoint : bool
      Whether to compute the adjoint FFT.

    Returns
    -------
    Primitive
      Result of the operation.
    """
    assert isinstance(fft_type, xla_client.FftType)

    transpose_order = get_transpose_order(fft_type)

    return fftn(
        x, fft_type=fft_type, adjoint=adjoint).transpose(transpose_order)

  @staticmethod
  def per_shard_impl(x: Array,
                     fft_type: FftType,
                     pdims: PdimsType,
                     global_shape: GdimsType,
                     adjoint: bool,
                     mesh : Mesh): # yapf: disable
    """
    Implementation function for per-shard FFT primitive.

    Parameters
    ----------
    x : Array
      Input array.
    fft_type : xla_client.FftType
      Type of FFT operation.
    pdims : Tuple[int, int]
      Parallel dimensions.
    global_shape : Tuple[int, int, int]
      Global shape of the array.
    adjoint : bool
      Whether to compute the adjoint FFT.

    Returns
    -------
    Primitive
      Result of the operation.
    """

    # TODO transpose_axis_contiguous_2 must be removed after benchmarking
    if fft_type == FftType.IFFT and pdims[0] == 1:
      if jaxdecomp.config.transpose_axis_contiguous_2 and jaxdecomp.config.transpose_axis_contiguous:
        x = x.transpose([1, 2, 0])

    out = FFTPrimitive.inner_primitive.bind(
        x,
        fft_type=fft_type,
        pdims=pdims,
        global_shape=global_shape,
        adjoint=adjoint,
        mesh=mesh)

    return out

  @staticmethod
  def infer_sharding_from_operands(
      fft_type: xla_client.FftType, adjoint: bool, mesh: Mesh,
      arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]) -> NamedSharding:
    """
    Infer sharding for FFT primitive based on operands.

    Parameters
    ----------
    fft_type : xla_client.FftType
      Type of FFT operation.
    adjoint : bool
      Whether to compute the adjoint FFT.
    mesh : Mesh
      Contextual mesh for sharding.
    arg_infos : Tuple[ShapeDtypeStruct]
      Shape and sharding information of input operands.
    result_infos : Tuple[ShapedArray]
      Shape information of output.

    Returns
    -------
    NamedSharding
      Sharding information for the result.
    """
    del adjoint, mesh, result_infos

    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    spec = input_sharding.spec
    input_mesh = input_sharding.mesh
    transposed_specs = get_output_specs(fft_type, spec, input_mesh, 'cudecomp')
    return NamedSharding(input_mesh, P(*transposed_specs))

  @staticmethod
  def partition(
      fft_type: FftType, adjoint: bool, mesh: Mesh,
      arg_shapes: Tuple[ShapeDtypeStruct], result_shape: ShapeDtypeStruct
  ) -> Tuple[Mesh, partial, NamedSharding, Tuple[NamedSharding]]:
    """
    Partition the FFT primitive for XLA.

    Parameters
    ----------
    fft_type : xla_client.FftType
      Type of FFT operation.
    adjoint : bool
      Whether to compute the adjoint FFT.
    mesh : Mesh
      Contextual mesh for sharding.
    arg_shapes : Tuple[ShapeDtypeStruct]
      Shape and sharding information of input operands.
    result_shape : ShapeDtypeStruct
      Shape and sharding information of output.

    Returns
    -------
    Tuple[Mesh, partial, NamedSharding, Tuple[NamedSharding]]
      Mesh, lowered function, output sharding, and input operand sharding.
    """
    input_mesh = arg_shapes[0].sharding.mesh
    input_sharding = NamedSharding(input_mesh, P(*arg_shapes[0].sharding.spec))
    output_sharding = NamedSharding(input_mesh, P(*result_shape.sharding.spec))
    global_shape = arg_shapes[0].shape
    pdims = get_pdims_from_sharding(output_sharding)

    impl = partial(
        FFTPrimitive.per_shard_impl,
        fft_type=fft_type,
        pdims=pdims,
        global_shape=global_shape,
        adjoint=adjoint,
        mesh=input_mesh)

    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(FFTPrimitive)


@partial(jax.jit, static_argnums=(1, 2))
def pfft_impl(x: Array, fft_type: FftType, adjoint: bool) -> Primitive:
  """
  Lowering function for pfft primitive.

  Parameters
  ----------
  x : Array
    Input array.
  fft_type : Union[str, xla_client.FftType]
    Type of FFT operation.
  adjoint : bool
    Whether to compute the adjoint FFT.

  Returns
  -------
  Primitive
    Result of the operation.
  """
  (x,) = promote_dtypes_complex(x)

  return FFTPrimitive.outer_primitive.bind(
      x, fft_type=fft_type, adjoint=adjoint)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def pfft(x: Array, fft_type: FftType, adjoint: bool = False) -> Primitive:
  """
  Custom VJP definition for pfft.

  Parameters
  ----------
  x : Array
    Input array.
  fft_type : Union[str, xla_client.FftType]
    Type of FFT operation.
  adjoint : bool, optional
    Whether to compute the adjoint FFT. Defaults to False.

  Returns
  -------
  Primitive
    Result of the operation.
  """
  output, _ = _pfft_fwd_rule(x, fft_type=fft_type, adjoint=adjoint)
  return output


def _pfft_fwd_rule(x: Array, fft_type: FftType,
                   adjoint: bool) -> Tuple[Primitive, None]:
  """
  Forward rule for pfft.

  Parameters
  ----------
  x : Array
    Input array.
  fft_type : Union[str, xla_client.FftType]
    Type of FFT operation.
  adjoint : bool, optional
    Whether to compute the adjoint FFT. Defaults to False.

  Returns
  -------
  Tuple[Primitive, None]
    Result of the operation and None (no residuals).
  """
  return pfft_impl(x, fft_type=fft_type, adjoint=adjoint), None


def _pfft_bwd_rule(fft_type: FftType, adjoint: bool, _,
                   g: Primitive) -> Tuple[Primitive]:
  """
  Backward rule for pfft.

  Parameters
  ----------
  fft_type : Union[str, xla_client.FftType]
    Type of FFT operation.
  adjoint : bool
    Whether to compute the adjoint FFT.
  ctx
    Context.
  g : Primitive
    Gradient value.

  Returns
  -------
  Tuple[Primitive]
    Result of the operation.
  """
  assert fft_type in [FftType.FFT, FftType.IFFT]
  if fft_type == FftType.FFT:
    fft_type = FftType.IFFT
  elif fft_type == FftType.IFFT:
    fft_type = FftType.FFT

  return pfft_impl(g, fft_type, not adjoint),


pfft.defvjp(_pfft_fwd_rule, _pfft_bwd_rule)
