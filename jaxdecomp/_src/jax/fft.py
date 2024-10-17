from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax
from jax import numpy as jnp
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ShapedArray
from jax._src.typing import Array, ArrayLike
from jax.lib import xla_client
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.fft_utils import COMPLEX, FORWARD_FFTs, fft, fft2, fftn
from jaxdecomp._src.pencil_utils import get_output_specs, get_transpose_order
from jaxdecomp._src.spmd_ops import (CustomParPrimitive, get_pencil_type,
                                     register_primitive)

FftType = xla_client.FftType


def _fft_slab_xy(operand, fft_type, adjoint, x_axis_name):
  # pdims are (Py=1,Pz=N)
  # input is (Z / Pz , Y , X) with specs P('z', 'y')
  # First FFT ont XY
  print(f"fft type {fft_type} , x_axis_name {x_axis_name} adjoint {adjoint} ")
  operand = fft2(operand, fft_type, axes=(2, 1), adjoint=adjoint)
  if jaxdecomp.config.transpose_axis_contiguous:
    # transpose to (Y , X/Pz , Z) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
    operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
  else:
    # transpose to (Z , Y , X/Pz) with specs P(None, 'y', 'z')
    operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True)
    operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)

  return operand


def _fft_slab_yz(operand, fft_type, adjoint, y_axis_name):
  # pdims are (Py=N,Pz=1)
  # input is (Z , Y / Py, X) with specs P('z', 'y')
  # First FFT on X
  print(f"fft type {fft_type} , y_axis_name {y_axis_name} adjoint {adjoint} ")
  operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
  if jaxdecomp.config.transpose_axis_contiguous:
    # transpose to (X / py, Z , Y) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, y_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
    # FFT on YZ plane
    operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)
  else:
    # transpose to (Z , Y , X / py) with specs P('z', None, 'y')
    operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True)
    # FFT on YZ plane
    operand = fft2(operand, COMPLEX(fft_type), axes=(1, 0), adjoint=adjoint)

  return operand


def _fft_pencils(operand, fft_type, adjoint, x_axis_name, y_axis_name):
  # pdims are (Py=N,Pz=N)
  # input is (Z / Pz , Y / Py  , X) with specs P('z', 'y')
  # First FFT on X
  operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
  if jaxdecomp.config.transpose_axis_contiguous:
    # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, y_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
    # FFT on the Y axis
    operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
    # transpose to (Y / pz, X / py, Z) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
    # FFT on the Z axis
    operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
  else:
    # transpose to (Z / Pz , Y , X / py) with specs P('z', None, 'y')
    operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True)
    # FFT on the Y axis
    operand = fft(operand, COMPLEX(fft_type), axis=1, adjoint=adjoint)
    # transpose to (Z , Y / pz, X / Py) with specs P(None , 'z', 'y')
    operand = lax.all_to_all(operand, x_axis_name, 1, 0, tiled=True)
    # FFT on the Z axis
    operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
  return operand


def _ifft_slab_xy(operand, fft_type, adjoint, x_axis_name):

  print(f"fft type {fft_type} , x_axis_name {x_axis_name} adjoint {adjoint}")

  # pdims are (Py=1,Pz=N)
  if jaxdecomp.config.transpose_axis_contiguous:
    # input is (Y , X/Pz , Z) with specs P('y', 'z')
    # First IFFT on Z
    operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
    # transpose to (Z / Pz , Y , X) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
    # IFFT on XY plane
    operand = fft2(operand, fft_type, axes=(2, 1), adjoint=adjoint)
  else:
    # input is (Z , Y , X/Pz) with specs P(None, 'y', 'z')
    # First IFFT on Z
    operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
    # transpose to (Z/Pz , Y , X) with specs P('z', 'y')
    operand = lax.all_to_all(operand, x_axis_name, 0, 2, tiled=True)
    # IFFT on XY plane
    operand = fft2(operand, fft_type, axes=(2, 1), adjoint=adjoint)
  return operand


def _ifft_slab_yz(operand, fft_type, adjoint, y_axis_name):

  print(f"fft type {fft_type} , y_axis_name {y_axis_name} adjoint {adjoint}")
  # pdims are (Py=N,Pz=1)
  if jaxdecomp.config.transpose_axis_contiguous:
    # input is (X / py, Z , Y) with specs P('y', 'z')
    # First IFFT
    operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)
    # transpose to (Z , Y / Py, X) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, y_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
    print(f"passage")
    # IFFT on X axis
    operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
  else:
    # input is (Z , Y , X / py) with specs P('z', None, 'y')
    # First IFFT on Y
    operand = fft2(operand, COMPLEX(fft_type), axes=(1, 0), adjoint=adjoint)
    # transpose to (Z , Y / py, X) with specs P('z', 'y')
    operand = lax.all_to_all(operand, y_axis_name, 1, 2, tiled=True)
    # IFFT on X axis
    operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)

  return operand


def _ifft_pencils(operand, fft_type, adjoint, x_axis_name, y_axis_name):
  # pdims are (Py=N,Pz=N)
  if jaxdecomp.config.transpose_axis_contiguous:
    # input is (Y / pz, X / py, Z) with specs P('z', 'y')
    # First IFFT on Z
    operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
    # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
    # IFFT on the Y axis
    operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
    # transpose to (Z / Pz , Y / Py  , X) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, y_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
    # IFFT on the X axis
    operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
  else:
    # input is (Z / Pz , Y / Py  , X) with specs P('z', 'y')
    # First IFFT on X
    operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
    # transpose to (Y / pz, X / py, Z) with specs P('z', 'y')
    operand = lax.all_to_all(operand, x_axis_name, 0, 1, tiled=True)
    # IFFT on the Z axis
    operand = fft(operand, COMPLEX(fft_type), axis=1, adjoint=adjoint)
    # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
    operand = lax.all_to_all(operand, y_axis_name, 1, 2, tiled=True)
    # IFFT on the Y axis
    operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
  return operand


class JAXFFTPrimitive(CustomParPrimitive):
  name = 'jax_fft'
  multiple_results = False
  impl_static_args: Tuple[int, int] = (1, 2)
  outer_primitive = None

  @staticmethod
  def impl(x, fft_type, adjoint):

    assert isinstance(fft_type, FftType)

    transpose_order = get_transpose_order(fft_type)
    return fftn(x, fft_type, adjoint=adjoint).transpose(transpose_order)

  @staticmethod
  def per_shard_impl(x: ArrayLike,
                     fft_type: int | tuple[int],
                     adjoint,
                     x_axis_name=None,
                     y_axis_name=None) -> Array:

    assert isinstance(fft_type, FftType)
    assert (x_axis_name is not None) or (y_axis_name is not None)
    pencil_type = get_pencil_type()
    if fft_type in FORWARD_FFTs:
      match pencil_type:
        case _jaxdecomp.SLAB_XY:
          return _fft_slab_xy(x, fft_type, adjoint, x_axis_name)
        case _jaxdecomp.SLAB_YZ:
          return _fft_slab_yz(x, fft_type, adjoint, y_axis_name)
        case _jaxdecomp.PENCILS:
          return _fft_pencils(x, fft_type, adjoint, x_axis_name, y_axis_name)
        case _:
          raise ValueError(f"Unsupported pencil type {pencil_type}")
    else:
      match pencil_type:
        case _jaxdecomp.SLAB_XY:
          return _ifft_slab_xy(x, fft_type, adjoint, x_axis_name)
        case _jaxdecomp.SLAB_YZ:
          return _ifft_slab_yz(x, fft_type, adjoint, y_axis_name)
        case _jaxdecomp.PENCILS:
          return _ifft_pencils(x, fft_type, adjoint, x_axis_name, y_axis_name)
        case _:
          raise ValueError(f"Unsupported pencil type {pencil_type}")

  @staticmethod
  def infer_sharding_from_operands(fft_type: FftType, adjoint: bool, mesh: Mesh,
                                   arg_infos: Tuple[ShapeDtypeStruct],
                                   result_infos: Tuple[ShapedArray]):

    del adjoint, result_infos

    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    spec = input_sharding.spec
    transposed_specs = get_output_specs(fft_type, spec, 'jax')

    return NamedSharding(mesh, P(*transposed_specs))

  @staticmethod
  def partition(fft_type: FftType, adjoint: bool, mesh: Mesh,
                arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):

    x_axis_name, y_axis_name = mesh.axis_names

    input_sharding = NamedSharding(mesh, P(*arg_infos[0].sharding.spec))
    output_sharding = NamedSharding(mesh, P(*result_infos.sharding.spec))

    impl = partial(
        JAXFFTPrimitive.per_shard_impl,
        fft_type=fft_type,
        adjoint=adjoint,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name)

    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(JAXFFTPrimitive)


@partial(jax.jit, static_argnums=(1, 2))
def pfft_impl(x: Array, fft_type: FftType, adjoint: bool) -> Array:
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

  return JAXFFTPrimitive.outer_lowering(x, fft_type=fft_type, adjoint=adjoint)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def pfft(x: Array, fft_type: FftType, adjoint: bool = False) -> Array:
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
  print(f"GETTING CALLED")
  output, _ = _pfft_fwd_rule(x, fft_type=fft_type, adjoint=adjoint)
  return output


def _pfft_fwd_rule(x: Array, fft_type: FftType,
                   adjoint: bool) -> Tuple[Array, None]:
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
                   g: Array) -> Tuple[Array]:
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
