from functools import partial
from typing import Tuple

import jax
from jax import lax
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ShapedArray
from jax._src.typing import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomplib import _jaxdecomp

import jaxdecomp
from jaxdecomp._src.fft_utils import COMPLEX  # yapf: disable
from jaxdecomp._src.fft_utils import FftType  # yapf: disable
from jaxdecomp._src.fft_utils import ADJOINT, FORWARD_FFTs, fft, fft2, fftn
from jaxdecomp._src.pencil_utils import get_output_specs, get_transpose_order
from jaxdecomp._src.spmd_ops import CustomParPrimitive  # yapf: disable
from jaxdecomp._src.spmd_ops import get_pencil_type, register_primitive


def _fft_slab_xy(operand: Array, fft_type: FftType, adjoint: bool,
                 x_axis_name: str) -> Array:
  """
  Performs the FFT slab XY operation.

  Parameters
  ----------
  operand : Array
      Input array for the FFT operation.
  fft_type : FftType
      Type of FFT to perform.
  adjoint : bool
      Whether to compute the adjoint FFT.
  x_axis_name : str
      Axis name for the X axis.

  Returns
  -------
  Array
      Resulting array after the FFT slab XY operation.
  """
  operand = fft2(operand, fft_type, axes=(2, 1), adjoint=adjoint)
  if jaxdecomp.config.transpose_axis_contiguous:
    operand = lax.all_to_all(
        operand, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
    operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
  else:
    operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True)
    operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
  return operand


def _fft_slab_yz(operand: Array, fft_type: FftType, adjoint: bool,
                 y_axis_name: str) -> Array:
  """
  Performs the FFT slab YZ operation.

  Parameters
  ----------
  operand : Array
      Input array for the FFT operation.
  fft_type : FftType
      Type of FFT to perform.
  adjoint : bool
      Whether to compute the adjoint FFT.
  y_axis_name : str
      Axis name for the Y axis.

  Returns
  -------
  Array
      Resulting array after the FFT slab YZ operation.
  """
  operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
  if jaxdecomp.config.transpose_axis_contiguous:
    # transpose to (X / py, Z , Y) with specs P('y', 'z')
    operand = lax.all_to_all(
        operand, y_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
    # FFT on YZ plane
    operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)

    if jaxdecomp.config.transpose_axis_contiguous_2:
      operand = operand.transpose([2, 0, 1])
  else:
    # transpose to (Z , Y , X / py) with specs P('z', None, 'y')
    operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True)
    # FFT on YZ plane
    operand = fft2(operand, COMPLEX(fft_type), axes=(1, 0), adjoint=adjoint)

  return operand


def _fft_pencils(operand: Array, fft_type: FftType, adjoint: bool,
                 x_axis_name: str, y_axis_name: str) -> Array:
  """
  Performs the FFT pencils operation.

  Parameters
  ----------
  operand : Array
      Input array for the FFT operation.
  fft_type : FftType
      Type of FFT to perform.
  adjoint : bool
      Whether to compute the adjoint FFT.
  x_axis_name : str
      Axis name for the X axis.
  y_axis_name : str
      Axis name for the Y axis.

  Returns
  -------
  Array
      Resulting array after the FFT pencils operation.
  """
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


def _ifft_slab_xy(operand: Array, fft_type: FftType, adjoint: bool,
                  x_axis_name: str) -> Array:
  """
  Performs the IFFT slab XY operation.

  Parameters
  ----------
  operand : Array
      Input array for the IFFT operation.
  fft_type : FftType
      Type of IFFT to perform.
  adjoint : bool
      Whether to compute the adjoint IFFT.
  x_axis_name : str
      Axis name for the X axis.

  Returns
  -------
  Array
      Resulting array after the IFFT slab XY operation.
  """
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


def _ifft_slab_yz(operand: Array, fft_type: FftType, adjoint: bool,
                  y_axis_name: str) -> Array:
  """
  Performs the IFFT slab YZ operation.

  Parameters
  ----------
  operand : Array
      Input array for the IFFT operation.
  fft_type : FftType
      Type of IFFT to perform.
  adjoint : bool
      Whether to compute the adjoint IFFT.
  y_axis_name : str
      Axis name for the Y axis.

  Returns
  -------
  Array
      Resulting array after the IFFT slab YZ operation.
  """
  if jaxdecomp.config.transpose_axis_contiguous:
    if jaxdecomp.config.transpose_axis_contiguous_2:
      operand = operand.transpose([1, 2, 0])
    # input is (X / py, Z , Y) with specs P('y', 'z')
    # First IFFT
    operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)
    # transpose to (Z , Y / Py, X) with specs P('z', 'y')
    operand = lax.all_to_all(
        operand, y_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
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


def _ifft_pencils(operand: Array, fft_type: FftType, adjoint: bool,
                  x_axis_name: str, y_axis_name: str) -> Array:
  """
  Performs the IFFT pencils operation.

  Parameters
  ----------
  operand : Array
      Input array for the IFFT operation.
  fft_type : FftType
      Type of IFFT to perform.
  adjoint : bool
      Whether to compute the adjoint IFFT.
  x_axis_name : str
      Axis name for the X axis.
  y_axis_name : str
      Axis name for the Y axis.

  Returns
  -------
  Array
      Resulting array after the IFFT pencils operation.
  """
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
  """
  JAX Custom FFT primitive for performing FFT operations on distributed data.
  """

  name = 'jax_fft'
  multiple_results = False
  impl_static_args: Tuple[int, ...] = (1, 2)
  outer_primitive = None

  @staticmethod
  def impl(x: Array, fft_type: FftType, adjoint: bool) -> Array:
    """
    Implementation of the FFT operation using the FFT primitive.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.

    Returns
    -------
    Array
        Resulting array after the FFT operation.
    """
    transpose_order = get_transpose_order(fft_type)
    return fftn(x, fft_type, adjoint=adjoint).transpose(transpose_order)

  @staticmethod
  def per_shard_impl(x: Array, fft_type: FftType, adjoint: bool,
                     mesh: Mesh) -> Array:
    """
    Per-shard implementation of the FFT operation.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    mesh : Mesh
        Mesh configuration for the distributed FFT.

    Returns
    -------
    Array
        Resulting array after the per-shard FFT operation.
    """
    x_axis_name, y_axis_name = mesh.axis_names
    assert isinstance(fft_type, FftType)  # type: ignore
    assert (x_axis_name is not None) or (y_axis_name is not None)
    pencil_type = get_pencil_type(mesh)
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
  def infer_sharding_from_operands(
      fft_type: FftType, adjoint: bool, mesh: Mesh,
      arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]) -> NamedSharding:
    """
    Infers the sharding for the result based on the input operands.

    Parameters
    ----------
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    mesh : Mesh
        Mesh configuration for the distributed FFT.
    arg_infos : Tuple[ShapeDtypeStruct]
        Shape and dtype information of the input operands.
    result_infos : Tuple[ShapedArray]
        Shape and dtype information of the result.

    Returns
    -------
    NamedSharding
        Sharding information for the result.
    """
    del mesh, result_infos, adjoint
    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    input_mesh: Mesh = input_sharding.mesh  # type: ignore
    spec = input_sharding.spec
    transposed_specs = get_output_specs(
        fft_type, spec, mesh=input_mesh, backend='jax')
    return NamedSharding(input_sharding.mesh, P(*transposed_specs))

  @staticmethod
  def partition(fft_type: FftType, adjoint: bool, mesh: Mesh,
                arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):
    """
    Partitions the FFT operation for distributed execution.

    Parameters
    ----------
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    mesh : Mesh
        Mesh configuration for the distributed FFT.
    arg_infos : Tuple[ShapeDtypeStruct]
        Shape and dtype information of the input operands.
    result_infos : Tuple[ShapedArray]
        Shape and dtype information of the result.

    Returns
    -------
    Tuple
        Mesh configuration, implementation function, output sharding, and input sharding.
    """
    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    output_sharding: NamedSharding = result_infos.sharding  # type: ignore
    input_mesh: Mesh = arg_infos[0].sharding.mesh  # type: ignore

    impl = partial(
        JAXFFTPrimitive.per_shard_impl,
        fft_type=fft_type,
        adjoint=adjoint,
        mesh=input_mesh)

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
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.

    Returns
    -------
    Array
        Resulting array after the pfft operation.
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
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool, optional
        Whether to compute the adjoint FFT. Defaults to False.

    Returns
    -------
    Array
        Resulting array after the pfft operation.
    """
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
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.

    Returns
    -------
    Tuple[Array, None]
        Resulting array after the forward pfft operation and None (no residuals).
    """
  return pfft_impl(x, fft_type=fft_type, adjoint=adjoint), None


def _pfft_bwd_rule(fft_type: FftType, adjoint: bool, _,
                   g: Array) -> Tuple[Array]:
  """
    Backward rule for pfft.

    Parameters
    ----------
    fft_type : FftType
        Type of FFT operation.
    adjoint : bool
        Whether to compute the adjoint FFT.
    g : Array
        Gradient of the result array.

    Returns
    -------
    Tuple[Array]
        Resulting array after the backward pfft operation.
    """
  return pfft_impl(g, ADJOINT(fft_type), not adjoint),


pfft.defvjp(_pfft_fwd_rule, _pfft_bwd_rule)
