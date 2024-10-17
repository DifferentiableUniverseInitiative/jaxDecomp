from functools import partial
from typing import Tuple

import jax
import jaxlib.mlir.ir as ir
import numpy as np
from jax import ShapeDtypeStruct
from jax._src.interpreters import mlir
from jax._src.typing import Array
from jax.core import Primitive, ShapedArray
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P, Mesh
from jaxlib.hlo_helpers import custom_call

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.spmd_ops import (BasePrimitive, get_axis_size,
                                     get_pencil_type, register_primitive)

GdimsType = Tuple[int, int, int]
PdimsType = Tuple[int, int]


class HaloPrimitive(BasePrimitive):
  """
    Custom primitive for performing halo exchange operation.
    """

  name = "halo_exchange"
  multiple_results = False
  impl_static_args = (1, 2)
  inner_primitive = None
  outer_primitive = None

  @staticmethod
  def abstract(x: Array, halo_extent: int, periodic: bool, pdims: PdimsType,
               global_shape: GdimsType) -> Array:
    """
    Abstract function for determining the shape and dtype after the halo exchange operation.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.
    pdims : Tuple[int, int]
        Processor dimensions.
    global_shape : Tuple[int, int, int]
        Global shape of the array.

    Returns
    -------
    Array
        Abstract array after the halo exchange operation.
    """
    del halo_extent, periodic, pdims, global_shape
    return x.update(shape=x.shape, dtype=x.dtype)

  @staticmethod
  def outer_abstract(x: Array, halo_extent: int, periodic: bool) -> Array:
    """
        Abstract function for determining the shape and dtype without considering inner details.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.

        Returns
        -------
        Array
            Abstract array after the halo exchange operation.
        """
    del halo_extent, periodic
    return x.update(shape=x.shape, dtype=x.dtype)

  @staticmethod
  def lowering(ctx, x: Array, halo_extent: int, periodic: bool,
               pdims: PdimsType, global_shape: GdimsType) -> Array:
    """
        Lowering function to generate the MLIR representation for halo exchange.

        Parameters
        ----------
        ctx
            Context for the operation.
        x : Array
            Input array.
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.
        pdims : Tuple[int, int]
            Processor dimensions.
        global_shape : Tuple[int, int, int]
            Global shape of the array.

        Returns
        -------
        Array
            Resulting array after the halo exchange operation.
        """
    (x_aval,) = ctx.avals_in
    x_type = ir.RankedTensorType(x.type)
    n = len(x_type.shape)

    is_double = np.finfo(x_aval.dtype).dtype == np.float64

    # Compute the descriptor for the halo exchange operation
    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = global_shape[::-1]
    config.halo_comm_backend = jaxdecomp.config.halo_comm_backend
    config.transpose_comm_backend = jaxdecomp.config.transpose_comm_backend

    halo_periods = (periodic, periodic, periodic)
    pencil_type = get_pencil_type()
    match pencil_type:
      case _jaxdecomp.SLAB_XY:
        halo_extents = (halo_extent, 0, 0)
      case _jaxdecomp.SLAB_YZ:
        halo_extents = (0, halo_extent, 0)
      case _jaxdecomp.PENCILS:
        halo_extents = (halo_extent, halo_extent, 0)
      case _:
        raise ValueError("Invalid pencil type")

    workspace_size, opaque = _jaxdecomp.build_halo_descriptor(
        config, is_double, halo_extents[::-1], halo_periods[::-1], 0)
    layout = tuple(range(n - 1, -1, -1))

    workspace = mlir.full_like_aval(
        ctx, 0, jax.core.ShapedArray(shape=[workspace_size], dtype=np.byte))

    # Perform custom call for halo exchange
    out = custom_call(
        "halo",
        result_types=[x_type],
        operands=[x, workspace],
        operand_layouts=[layout, (0,)],
        result_layouts=[layout],
        has_side_effect=True,
        operand_output_aliases={0: 0},
        backend_config=opaque,
    )
    return out.results

  @staticmethod
  def impl(x: Array, halo_extent: int, periodic: bool) -> Primitive:
    """
        Implementation function for performing halo exchange.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.

        Returns
        -------
        Primitive
            Inner primitive bound with input parameters.
        """
    pdims = (1, jax.device_count())
    global_shape = x.shape

    return HaloPrimitive.inner_primitive.bind(
        x,
        halo_extent=halo_extent,
        periodic=periodic,
        pdims=pdims,
        global_shape=global_shape,
    )

  @staticmethod
  def per_shard_impl(x: Array, halo_extent: int, periodic: bool,
                     pdims: PdimsType, global_shape: GdimsType) -> Array:
    """
        Implementation function for performing halo exchange per shard.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.
        pdims : Tuple[int, int]
            Processor dimensions.
        global_shape : Tuple[int, int, int]
            Global shape of the array.

        Returns
        -------
        Array
            Resulting array after the halo exchange operation.
        """
    output = HaloPrimitive.inner_primitive.bind(
        x,
        halo_extent=halo_extent,
        periodic=periodic,
        pdims=pdims,
        global_shape=global_shape,
    )
    return output

  @staticmethod
  def infer_sharding_from_operands(
      halo_extent: int, periodic: bool, mesh: NamedSharding,
      arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]) -> NamedSharding:
    """
        Infer sharding information for halo exchange operation.

        Parameters
        ----------
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.
        mesh : NamedSharding
            Mesh object for sharding.
        arg_shapes : Tuple[ir.ShapeDtypeStruct]
            Shapes and dtypes of input operands.
        result_shape : ir.ShapedArray
            Shape and dtype of the output result.

        Returns
        -------
        NamedSharding
            Sharding information for halo exchange operation.
        """
    halo_exchange_sharding = arg_infos[0].sharding
    return NamedSharding(mesh, P(*halo_exchange_sharding.spec))

  @staticmethod
  def partition(halo_extent: int, periodic: bool, mesh: Mesh,
                arg_shapes: Tuple[ShapeDtypeStruct],
                result_shape: ShapeDtypeStruct):
    """
        Partition function for halo exchange operation.

        Parameters
        ----------
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.
        mesh : NamedSharding
            Mesh object for sharding.
        arg_shapes : Tuple[ir.ShapeDtypeStruct]
            Shapes and dtypes of input operands.
        result_shape : ir.ShapedArray
            Shape and dtype of the output result.

        Returns
        -------
        Tuple[NamedSharding, partial]
            Mesh object, implementation function, sharding information, and its tuple.
        """
    halo_exchange_sharding = NamedSharding(mesh,
                                           P(*arg_shapes[0].sharding.spec))
    global_shape = arg_shapes[0].shape
    pdims = (get_axis_size(halo_exchange_sharding,
                           1), get_axis_size(halo_exchange_sharding, 0))

    pencil_type = get_pencil_type()
    match pencil_type:
      case _jaxdecomp.SLAB_XY:
        halo_extents = (halo_extent, 0, 0)
      case _jaxdecomp.SLAB_YZ:
        halo_extents = (0, halo_extent, 0)
      case _jaxdecomp.PENCILS:
        halo_extents = (halo_extent, halo_extent, 0)
      case _:
        raise ValueError("Invalid pencil type")

    shape_without_halo = (global_shape[0] - 2 * pdims[1] * halo_extents[0],
                          global_shape[1] - 2 * pdims[0] * halo_extents[1],
                          global_shape[2] - 2 * halo_extents[2])

    impl = partial(
        HaloPrimitive.per_shard_impl,
        halo_extent=halo_extent,
        periodic=periodic,
        pdims=pdims,
        global_shape=shape_without_halo)

    return mesh, impl, halo_exchange_sharding, (halo_exchange_sharding,)


register_primitive(HaloPrimitive)


@partial(jax.jit, static_argnums=(1, 2))
def halo_p_lower(x: Array, halo_extent: int, periodic: bool) -> Array:
  """
    Lowering function for the halo exchange operation.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Primitive
        Inner primitive bound with input parameters.
    """
  return HaloPrimitive.outer_primitive.bind(
      x,
      halo_extent=halo_extent,
      periodic=periodic,
  )


# Custom Partitioning
@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def halo_exchange(x: Array, halo_extent: int, periodic: bool) -> Array:
  """
    Halo exchange operation with custom VJP.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Array
        Output array after the halo exchange operation.
    """
  output, _ = _halo_fwd_rule(x, halo_extent, periodic)
  return output


def _halo_fwd_rule(x: Array, halo_extent: int,
                   periodic: bool) -> Tuple[Array, None]:
  """
    Forward rule for the halo exchange operation.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Tuple[Array, None]
        Output array after the halo exchange operation and None for no residuals.
    """
  return halo_p_lower(x, halo_extent, periodic), None


def _halo_bwd_rule(halo_extent: int, periodic: bool, _,
                   g: Array) -> Tuple[Array]:
  """
    Backward rule for the halo exchange operation.

    Parameters
    ----------
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.
    ctx
        Context for the operation.
    g : Array
        Gradient array.

    Returns
    -------
    Tuple[Array]
        Gradient array after the halo exchange operation.
    """
  return halo_p_lower(g, halo_extent, periodic),


# Define VJP for custom halo_exchange operation
halo_exchange.defvjp(_halo_fwd_rule, _halo_bwd_rule)
