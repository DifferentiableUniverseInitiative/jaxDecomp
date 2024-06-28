from functools import partial
from typing import Tuple

import jax
import jaxlib.mlir.ir as ir
import numpy as np
from jax import ShapeDtypeStruct
from jax._src.interpreters import mlir
from jax._src.lib.mlir.dialects import hlo
from jax._src.typing import Array, ArrayLike
from jax.core import ShapedArray
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib.hlo_helpers import custom_call

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.spmd_ops import (BasePrimitive, get_axis_size,
                                     register_primitive)


class TransposePrimitive(BasePrimitive):
  """
    JAX primitive for transposing arrays with different partitioning strategies.

    Attributes
    ----------
    name : str
        Name of the primitive ("transpose").
    multiple_results : bool
        Boolean indicating if the primitive returns multiple results (False).
    impl_static_args : tuple
        Static arguments for the implementation (tuple containing (1,)).
    inner_primitive : object
        Inner core.Primitive object for the primitive.
    outer_primitive : object
        Outer core.Primitive object for the primitive.
    """

  name: str = "transpose"
  multiple_results: bool = False
  impl_static_args: Tuple[int] = (1,)
  inner_primitive: object = None
  outer_primitive: object = None

  @staticmethod
  def abstract(x: ArrayLike, kind: str, pdims: Tuple[int],
               global_shape: Tuple[int]) -> ShapedArray:
    """
        Abstract method to describe the shape of the output array after transposition.

        Parameters
        ----------
        x : ArrayLike
            Input array.
        kind : str
            Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
        pdims : tuple[int]
            Partition dimensions.
        global_shape : tuple[int]
            Global shape of the input array.

        Returns
        -------
        ShapedArray
            Abstract shape of the output array after transposition.
        """
    if global_shape == x.shape:
      return TransposePrimitive.outer_abstract(x, kind)
    # Make sure that global_shape is divisible by pdims and equals to slice

    assert kind in ['x_y', 'y_z', 'z_y', 'y_x']
    match kind:
    # From X to Y the axis are rolled by 1 and pdims are swapped wrt to the input pdims
      case 'x_y':
        transpose_shape = (2, 0, 1)
        transpose_pdims = pdims[::-1]
      case 'y_z':
        transpose_shape = (2, 0, 1)
        transpose_pdims = pdims
      case 'z_y':
        transpose_shape = (1, 2, 0)
        transpose_pdims = pdims[::-1]
      case 'y_x':
        transpose_shape = (1, 2, 0)
        transpose_pdims = pdims

    shape = (global_shape[transpose_shape[0]] // transpose_pdims[1],
             global_shape[transpose_shape[1]] // transpose_pdims[0],
             global_shape[transpose_shape[2]])

    return ShapedArray(shape, x.dtype)

  @staticmethod
  def outer_abstract(x: ArrayLike, kind: str) -> ShapedArray:
    """
        Abstract method for transposition that does not require knowledge of global shape.

        Parameters
        ----------
        x : ArrayLike
            Input array.
        kind : str
            Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').

        Returns
        -------
        ShapedArray
            Abstract shape of the output array after transposition.
        """
    assert kind in ['x_y', 'y_z', 'z_y', 'y_x']
    match kind:
    # From X to Y the axis are rolled by 1 and pdims are swapped wrt to the input pdims
      case 'x_y' | 'y_z':
        transpose_shape = (2, 0, 1)
      case 'y_x' | 'z_y':
        transpose_shape = (1, 2, 0)

    shape = (x.shape[transpose_shape[0]], x.shape[transpose_shape[1]],
             x.shape[transpose_shape[2]])

    return ShapedArray(shape, x.dtype)

  @staticmethod
  def lowering(ctx, x: ArrayLike, *, kind: str, pdims: Tuple[int],
               global_shape: Tuple[int]):
    """
        Method to lower the transposition operation to MLIR.

        Parameters
        ----------
        ctx : object
            Context for the operation.
        x : ArrayLike
            Input array.
        kind : str
            Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
        pdims : tuple[int]
            Partition dimensions.
        global_shape : tuple[int]
            Global shape of the input array.

        Returns
        -------
        List
            List of lowered results.
        """
    assert kind in ['x_y', 'y_z', 'z_y', 'y_x']
    (aval_in,) = ctx.avals_in
    (aval_out,) = ctx.avals_out
    dtype = aval_in.dtype
    x_type = ir.RankedTensorType(x.type)
    is_double = dtype == np.float64

    layout = tuple(range(len(x_type.shape) - 1, -1, -1))

    # Recover original global shape
    match kind:
      case 'x_y':
        transpose_shape = (0, 1, 2)
        transpose_type = _jaxdecomp.TRANSPOSE_XY
      case 'y_z':
        transpose_shape = (1, 2, 0)
        transpose_type = _jaxdecomp.TRANSPOSE_YZ
      case 'z_y':
        transpose_shape = (2, 0, 1)
        transpose_type = _jaxdecomp.TRANSPOSE_ZY
      case 'y_x':
        transpose_shape = (1, 2, 0)
        transpose_type = _jaxdecomp.TRANSPOSE_YX
      case _:
        raise ValueError("Invalid kind")

    # Make sure to get back the original shape of the X-Pencil
    global_shape = tuple([global_shape[i] for i in transpose_shape])

    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = global_shape[::-1]
    config.halo_comm_backend = jaxdecomp.config.halo_comm_backend
    config.transpose_comm_backend = jaxdecomp.config.transpose_comm_backend

    workspace_size, opaque = _jaxdecomp.build_transpose_descriptor(
        config, transpose_type, is_double)

    workspace = mlir.full_like_aval(
        ctx, 0, jax.core.ShapedArray(shape=[workspace_size], dtype=np.byte))

    result = custom_call(
        "transpose",
        result_types=[x_type],
        operands=[x, workspace],
        operand_layouts=[layout, (0,)],
        result_layouts=[layout],
        has_side_effect=True,
        operand_output_aliases={0: 0},
        backend_config=opaque,
    )
    # Finally we reshape the arry to the expected shape.
    return hlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results

  @staticmethod
  def impl(x: ArrayLike, kind: str):
    """
        Implementation method for the transposition primitive.

        Parameters
        ----------
        x : ArrayLike
            Input array.
        kind : str
            Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').

        Returns
        -------
        Object
            Result of binding the inner primitive with input arguments.
        """
    size = jax.device_count()
    pdims = (size, 1)  # pdims product must be equal to the number of devices
    global_shape = x.shape
    return TransposePrimitive.inner_primitive.bind(
        x, kind=kind, pdims=pdims, global_shape=global_shape)

  @staticmethod
  def per_shard_impl(x: ArrayLike, kind: str, pdims: Tuple[int],
                     global_shape: Tuple[int]):
    """
        Per-shard implementation method for the transposition primitive.

        Parameters
        ----------
        x : ArrayLike
            Input array.
        kind : str
            Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
        pdims : tuple[int]
            Partition dimensions.
        global_shape : tuple[int]
            Global shape of the input array.

        Returns
        -------
        Object
            Result of binding the inner primitive with input arguments.
        """
    return TransposePrimitive.inner_primitive.bind(
        x, kind=kind, pdims=pdims, global_shape=global_shape)

  @staticmethod
  def infer_sharding_from_operands(
      kind: str, mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]) -> NamedSharding:
    """
        Method to infer sharding information from operands for custom partitioning.

        Parameters
        ----------
        kind : str
            Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
        mesh : Mesh
            Sharding mesh information.
        arg_infos : Tuple[ShapeDtypeStruct]
            Tuple of ShapeDtypeStruct for input operands.
        result_infos : Tuple[ShapedArray]
            Tuple of ShapedArray for result information.

        Returns
        -------
        NamedSharding
            Named sharding information.
        """
    input_sharding = arg_infos[0].sharding

    tranposed_pdims = (input_sharding.spec[1], input_sharding.spec[0], None)

    return NamedSharding(input_sharding.mesh, P(*tranposed_pdims))

  @staticmethod
  def partition(
      kind: str, mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]
  ) -> Tuple[Mesh, callable, NamedSharding, Tuple[NamedSharding]]:
    """
        Method to partition the transposition operation for custom partitioning.

        Parameters
        ----------
        kind : str
            Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
        mesh : Mesh
            Sharding mesh information.
        arg_infos : Tuple[ShapeDtypeStruct]
            Tuple of ShapeDtypeStruct for input operands.
        result_infos : Tuple[ShapedArray]
            Tuple of ShapedArray for result information.

        Returns
        -------
        Tuple
            Tuple containing mesh, implementation function, output sharding, and input sharding.
        """
    input_sharding = NamedSharding(mesh, P(*arg_infos[0].sharding.spec))
    output_sharding = NamedSharding(mesh, P(*result_infos.sharding.spec))
    global_shape = arg_infos[0].shape
    match kind:
      case 'x_y' | 'z_y':
        pdims = (get_axis_size(input_sharding,
                               1), get_axis_size(input_sharding, 0))
      case 'y_z' | 'y_x':
        pdims = (get_axis_size(input_sharding,
                               0), get_axis_size(input_sharding, 1))

    impl = partial(
        TransposePrimitive.per_shard_impl,
        kind=kind,
        pdims=pdims,
        global_shape=global_shape)

    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(TransposePrimitive)


@partial(jax.jit, static_argnums=(1,))
def transpose(x: ArrayLike, kind: str) -> Array:
  """
    JIT-compiled function for performing transposition using the outer primitive.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').

    Returns
    -------
    Array
        Transposed array.
    """
  return TransposePrimitive.outer_primitive.bind(x, kind=kind)


# Custom transposition functions


@jax.custom_vjp
def transposeXtoY(x: ArrayLike) -> Array:
  """
    Custom JAX transposition function for X to Y.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
  return transpose(x, 'x_y')


def transposeXtoY_fwd(x):
  return transpose(x, 'x_y'), None


def transposeXtoY_bwd(_, g):
  return transpose(g, 'y_x'),


@jax.custom_vjp
def transposeYtoZ(x: ArrayLike) -> Array:
  """
    Custom JAX transposition function for Y to Z.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
  return transpose(x, 'y_z')


def transposeYtoZ_fwd(x):
  return transpose(x, 'y_z'), None


def transposeYtoZ_bwd(_, g):
  return transpose(g, 'z_y'),


@jax.custom_vjp
def transposeZtoY(x: ArrayLike) -> Array:
  """
    Custom JAX transposition function for Z to Y.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
  return transpose(x, 'z_y')


def transposeZtoY_fwd(x):
  return transpose(x, 'z_y'), None


def transposeZtoY_bwd(_, g):
  return transpose(g, 'y_z'),


@jax.custom_vjp
def transposeYtoX(x: ArrayLike) -> Array:
  """
    Custom JAX transposition function for Y to X.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
  return transpose(x, 'y_x')


def transposeYtoX_fwd(x):
  return transpose(x, 'y_x'), None


def transposeYtoX_bwd(_, g):
  return transpose(g, 'x_y'),


# Define VJPs for custom transposition functions
transposeXtoY.defvjp(transposeXtoY_fwd, transposeXtoY_bwd)
transposeYtoZ.defvjp(transposeYtoZ_fwd, transposeYtoZ_bwd)
transposeZtoY.defvjp(transposeZtoY_fwd, transposeZtoY_bwd)
transposeYtoX.defvjp(transposeYtoX_fwd, transposeYtoX_bwd)
