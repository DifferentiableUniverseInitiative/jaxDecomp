from functools import partial
from typing import Union

import jax
import jaxlib.mlir.ir as ir
import numpy as np
from jax import jit
from jax._src.api import jit
from jax._src.interpreters import mlir
from jax._src.lib.mlir.dialects import hlo
from jax._src.numpy.util import promote_dtypes_complex
from jax.core import Primitive
from jax.interpreters import ad, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

import jaxdecomp
from jaxdecomp._src import _jaxdecomp

FftType = xla_client.FftType
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


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


# Note : This must no longer be jitted because it will have single device abstract shapes
# The actual jit is done in the pfft function in lower_fn
# This means that sfft should never be lowered as is and only be lowered in the context of pfft
#@partial(jit, static_argnums=(1, 2, 3, 4))
def sfft(x,
         fft_type: Union[xla_client.FftType, str],
         adjoint=False,
         pdims=[1, 1],
         global_shape=[1024, 1024, 1024]):

  #TODO(wassim) : find a way to prevent user from using the primitive directly

  if isinstance(fft_type, str):
    typ = _str_to_fft_type(fft_type)
  elif isinstance(fft_type, xla_client.FftType):
    typ = fft_type
  else:
    raise TypeError(f"Unknown FFT type value '{fft_type}'")

  if typ in [xla_client.FftType.RFFT, xla_client.FftType.IRFFT]:
    raise TypeError("only complex FFTs are currently supported through pfft.")

  (x,) = promote_dtypes_complex(x)

  return sfft_p.bind(
      x, fft_type=typ, pdims=pdims, global_shape=global_shape, adjoint=adjoint)


def sfft_abstract_eval(x, fft_type, pdims, global_shape, adjoint):

  # TODO(Wassim) : this only handles cube shapes
  # This function is called twice once with the global array and once with the local slice shape

  # Figure out what is the pencil decomposition at the output
  axis = 0
  if fft_type in [xla_client.FftType.RFFT, xla_client.FftType.FFT]:
    axis = 2

  output_shape = None
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
      raise TypeError("only complex FFTs are currently supported through pfft.")

  # Are we operating on the global array?
  # This is called when the abstract_eval of the custom partitioning is called _custom_partitioning_abstract_eval in  https://github.com/google/jax/blob/main/jax/experimental/custom_partitioning.py#L223
  if x.shape == global_shape:
    shape = tuple([global_shape[i] for i in transpose_shape])
    output_shape = shape
  # Or are we operating on a local slice?
  # this is called JAX calls make_jaxpr(lower_fn) in https://github.com/google/jax/blob/main/jax/experimental/custom_partitioning.py#L142C5-L142C35
  else:
    output_shape = (global_shape[transpose_shape[0]] // transposed_pdims[1],
                    global_shape[transpose_shape[1]] // transposed_pdims[0],
                    global_shape[transpose_shape[2]])

  # Sanity check
  assert (output_shape is not None)
  return x.update(shape=output_shape, dtype=x.dtype)


def sfft_lowering(ctx, a, *, fft_type, pdims, global_shape, adjoint):
  (x_aval,) = ctx.avals_in
  (aval_out,) = ctx.avals_out
  dtype = x_aval.dtype
  a_type = ir.RankedTensorType(a.type)
  # We currently only support complex FFTs through this interface, so let's check the fft type
  assert fft_type in (FftType.FFT,
                      FftType.IFFT), "Only complex FFTs are currently supported"

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
      raise TypeError("only complex FFTs are currently supported through pfft.")
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
  out_type = ir.RankedTensorType.get(aval_out.shape, a_type.element_type)
  return hlo.ReshapeOp(out_type, result).results


def _fft_transpose_rule(x, operand, fft_type, pdims, global_shape, adjoint):
  assert fft_type in [FftType.FFT, FftType.IFFT]
  if fft_type == FftType.FFT:
    result = sfft(x, FftType.IFFT, ~adjoint, pdims, global_shape)
  elif fft_type == FftType.IFFT:
    result = sfft(x, FftType.FFT, ~adjoint, pdims, global_shape)
  else:
    raise NotImplementedError

  return (result,)


def get_axis_size(sharding, index):
  axis_name = sharding.spec[index]
  if axis_name == None:
    return 1
  else:
    return sharding.mesh.shape[sharding.spec[index]]


# Only named sharding have a spec
# this function is actually useless because positional sharding do not have a spec
# in case the user does not use a context mesh this will fail
# this is a placeholder function for the future
# the spec needs to be carried by a custom object that we create ourselfs
# to get inspired : https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuFFTMp/JAX_FFT/src/xfft/xfft.py#L20
def to_named_sharding(sharding):
  return NamedSharding(sharding.mesh, P(*sharding.spec))


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
  input_sharding = arg_shapes[0].sharding

  def lower_fn(operand):
    # Operand is a local slice and arg_shapes contains the global shape
    # No need to retranpose in the relowered function because abstract eval understands sliced input
    # and in the original lowering we use aval.out
    # it cannot work any other way because custom partition compares the output of the lower_fn with the abs eval (after comparing the global one)
    # this means that the abs eval should handle both global shapes and slice shape

    global_shape = arg_shapes[0].shape
    pdims = (get_axis_size(input_sharding, 1), get_axis_size(input_sharding, 0))

    output = sfft(operand, fft_type, adjoint, pdims, global_shape)

    # This is supposed to let us avoid making an extra transpose in the YZ case
    # it does not work
    # # In case of YZ slab the cuda code tranposes only once
    # # We transpose again to give back the Z-Pencil to the user in case of FFT and the X-Pencil in case of IFFT
    # # this transposition is supposed to compiled out by XLA when doing a gradient (forward followed by backward)
    # if get_axis_size(input_sharding, 0) == 1:
    #   if fft_type == FftType.FFT:
    #     output = output.transpose((1, 2, 0))
    #   elif fft_type == FftType.IFFT:
    #     output = output.transpose((2, 0, 1))
    return output

  return mesh, lower_fn,  \
      to_named_sharding(result_shape.sharding), \
      (to_named_sharding(arg_shapes[0].sharding),)


def infer_sharding_from_operands(fft_type, adjoint, mesh, arg_shapes,
                                 result_shape):
  # Static arguments fft_type  adjoint are carried along
  """
    Tell XLA how to infer the sharding of the output from the input sharding.

    Args:
        mesh (Mesh): The contextual mesh

        arg_shapes (tuple): A tuple of ShapeDtypeStruct that contains the shape and the sharding of each input operand

        result_shape (ShapedArray) : a single ShapedArray reprsenting a single output without the sharding information

    Returns:

        result_sharding (XLACompatibleSharding): The sharding result for example a NamedSharding.

    """
  # only one operand is used in pfft
  input_sharding = arg_shapes[0].sharding
  return NamedSharding(mesh, P(*input_sharding.spec))


@partial(custom_partitioning, static_argnums=(1, 2))
def pfft_p_lower(x, fft_type, adjoint=False):
  # the product of the fake dim has to be equal to the product of the global shape
  # Fake dims and shape values are irrelevant because they are never used as concrete values only as Traced values
  # their shapes however are used in the abstract eval of the custom partitioning

  size = jax.device_count()
  # The pdims product must be equal to the number of devices because this is checked both in the abstract eval and in cudecomp
  dummy_pdims = (1, size)
  dummy_global = x.shape
  return sfft(x, fft_type, adjoint, dummy_pdims, dummy_global)


sfft_p = Primitive("pfft")
sfft_p.def_impl(partial(xla.apply_primitive, sfft_p))
sfft_p.def_abstract_eval(sfft_abstract_eval)
ad.deflinear2(sfft_p, _fft_transpose_rule)
mlir.register_lowering(sfft_p, sfft_lowering, platform="gpu")

# Define the partitioning for the primitive
pfft_p_lower.def_partition(
    partition=partition,
    infer_sharding_from_operands=infer_sharding_from_operands)

# declaring a differentiable SPMD primitive
# Inspired from
# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/jax/cpp_extensions.py#L188
# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/jax/cpp_extensions.py#L694
# https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/jax/layernorm.py#L49
# Note TE does a cleaner way of defining the primitive by using register_primitive(cls): that declares two primitives
# An inner which is represented here by sfft
# And an outer which by analogy shoud be represented here by pfft


# Do not jit this
# the jit is happening in jaxdecomp/fft.py: _do_pfft
@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def pfft(x, fft_type, adjoint=False):
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
