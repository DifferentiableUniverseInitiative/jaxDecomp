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
from jax.sharding import PartitionSpec as P
from jaxlib.hlo_helpers import custom_call

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.spmd_ops import (BasePrimitive, get_axis_size,
                                     register_primitive)


class HaloPrimitive(BasePrimitive):
  """
    Custom primitive for performing halo exchange operation.
    """

  name = "halo_exchange"
  multiple_results = False
  impl_static_args = (1, 2, 3)
  inner_primitive = None
  outer_primitive = None

  @staticmethod
  def abstract(x: Array, halo_extents: Tuple[int, int, int],
               halo_periods: Tuple[bool, bool, bool], reduce_halo: bool,
               pdims: Tuple[int, int], global_shape: Tuple[int, int,
                                                           int]) -> Array:
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
        reduce_halo : bool
            Flag indicating whether to reduce the halo.
        pdims : Tuple[int, int]
            Processor dimensions.
        global_shape : Tuple[int, int, int]
            Global shape of the array.

        Returns
        -------
        Array
            Abstract array after the halo exchange operation.
        """
    return x.update(shape=x.shape, dtype=x.dtype)

  @staticmethod
  def outer_abstract(x: Array, halo_extents: Tuple[int, int, int],
                     halo_periods: Tuple[bool, bool,
                                         bool], reduce_halo: bool) -> Array:
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
        reduce_halo : bool
            Flag indicating whether to reduce the halo.

        Returns
        -------
        Array
            Abstract array after the halo exchange operation.
        """
    return x.update(shape=x.shape, dtype=x.dtype)

  @staticmethod
  def lowering(ctx, x: Array, halo_extents: Tuple[int, int, int],
               halo_periods: Tuple[bool, bool, bool], reduce_halo: bool,
               pdims: Tuple[int, int], global_shape: Tuple[int, int,
                                                           int]) -> Array:
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
        reduce_halo : bool
            Flag indicating whether to reduce the halo.
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
  def impl(x: Array, halo_extents: Tuple[int, int, int],
           halo_periods: Tuple[bool, bool,
                               bool], reduce_halo: bool) -> Primitive:
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
        reduce_halo : bool
            Flag indicating whether to reduce the halo.

        Returns
        -------
        Primitive
            Inner primitive bound with input parameters.
        """
    pdims = (1, jax.device_count())
    global_shape = x.shape

    return HaloPrimitive.inner_primitive.bind(
        x,
        halo_extents=halo_extents,
        halo_periods=halo_periods,
        reduce_halo=reduce_halo,
        pdims=pdims,
        global_shape=global_shape,
    )

  @staticmethod
  def per_shard_impl(x: Array, halo_extents: Tuple[int, int, int],
                     halo_periods: Tuple[bool, bool, bool], reduce_halo: bool,
                     pdims: Tuple[int, int], global_shape: Tuple[int, int,
                                                                 int]) -> Array:
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
        reduce_halo : bool
            Flag indicating whether to reduce the halo.
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
        halo_extents=halo_extents,
        halo_periods=halo_periods,
        reduce_halo=reduce_halo,
        pdims=pdims,
        global_shape=global_shape,
    )

    if reduce_halo:
      # Padding is usally halo_size and the halo_exchange extents are halo_size // 2
      # So the reduction is done on half of the halo_size
      halo_x, halo_y, halo_z = [extent * 2 for extent in halo_extents]

      # Apply corrections along x
      if halo_x > 0:
        output = output.at[halo_x:halo_x + halo_x // 2].add(output[:halo_x //
                                                                   2])
        output = output.at[-(halo_x + halo_x // 2):-halo_x].add(
            output[-halo_x // 2:])
      # Apply corrections along y
      if halo_y > 0:
        output = output.at[:, halo_y:halo_y + halo_y // 2].add(
            output[:, :halo_y // 2])
        output = output.at[:, -(halo_y + halo_y // 2):-halo_y].add(
            output[:, -halo_y // 2:])
      # Apply corrections along z
      if halo_z > 0:
        output = output.at[:, :, halo_z:halo_z + halo_z // 2].add(
            output[:, :, :halo_z // 2])
        output = output.at[:, :, -(halo_z + halo_z // 2):-halo_z].add(
            output[:, :, -halo_z // 2:])

    return output

  @staticmethod
  def infer_sharding_from_operands(
      halo_extents: Tuple[int, int,
                          int], halo_periods: Tuple[bool, bool,
                                                    bool], reduce_halo: bool,
      mesh: NamedSharding, arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]) -> NamedSharding:
    """
        Infer sharding information for halo exchange operation.

        Parameters
        ----------
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.
        reduce_halo : bool
            Flag indicating whether to reduce the halo.
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
  def partition(
      halo_extents: Tuple[int, int,
                          int], halo_periods: Tuple[bool, bool,
                                                    bool], reduce_halo: bool,
      mesh: NamedSharding, arg_shapes: Tuple[ShapeDtypeStruct],
      result_shape: ShapeDtypeStruct) -> Tuple[NamedSharding, partial]:
    """
        Partition function for halo exchange operation.

        Parameters
        ----------
        halo_extents : Tuple[int, int, int]
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Tuple[bool, bool, bool]
            Periodicity of the halo in x, y, and z dimensions.
        reduce_halo : bool
            Flag indicating whether to reduce the halo.
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

    shape_without_halo = (global_shape[0] - 2 * pdims[1] * halo_extents[0],
                          global_shape[1] - 2 * pdims[0] * halo_extents[1],
                          global_shape[2] - 2 * halo_extents[2])

    impl = partial(
        HaloPrimitive.per_shard_impl,
        halo_extents=halo_extents,
        halo_periods=halo_periods,
        reduce_halo=reduce_halo,
        pdims=pdims,
        global_shape=shape_without_halo)

    return mesh, impl, halo_exchange_sharding, (halo_exchange_sharding,)


register_primitive(HaloPrimitive)


def halo_p_lower(x: Array, halo_extents: Tuple[int, int, int],
                 halo_periods: Tuple[bool, bool,
                                     bool], reduce_halo: bool) -> Primitive:
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
    reduce_halo : bool
        Flag indicating whether to reduce the halo.

    Returns
    -------
    Primitive
        Inner primitive bound with input parameters.
    """
  return HaloPrimitive.outer_primitive.bind(
      x,
      halo_extents=halo_extents,
      halo_periods=halo_periods,
      reduce_halo=reduce_halo,
  )


# Custom Partitioning
@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def halo_exchange(x: Array,
                  halo_extents: Tuple[int, int, int],
                  halo_periods: Tuple[bool, bool, bool],
                  reduce_halo: bool = False) -> Array:
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
    reduce_halo : bool, optional
        Flag indicating whether to reduce the halo. Default is False.

    Returns
    -------
    Array
        Output array after the halo exchange operation.
    """
  output, _ = _halo_fwd_rule(x, halo_extents, halo_periods, reduce_halo)
  return output


def _halo_fwd_rule(x: Array, halo_extents: Tuple[int, int, int],
                   halo_periods: Tuple[bool, bool, bool],
                   reduce_halo: bool) -> Tuple[Array, None]:
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
    reduce_halo : bool
        Flag indicating whether to reduce the halo.

    Returns
    -------
    Tuple[Array, None]
        Output array after the halo exchange operation and None for no residuals.
    """
  return halo_p_lower(x, halo_extents, halo_periods, reduce_halo), None


def _halo_bwd_rule(halo_extents: Tuple[int, int, int],
                   halo_periods: Tuple[bool, bool, bool], reduce_halo: bool,
                   ctx, g: Array) -> Tuple[Array]:
  """
    Backward rule for the halo exchange operation.

    Parameters
    ----------
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.
    reduce_halo : bool
        Flag indicating whether to reduce the halo.
    ctx
        Context for the operation.
    g : Array
        Gradient array.

    Returns
    -------
    Tuple[Array]
        Gradient array after the halo exchange operation.
    """
  return halo_p_lower(g, halo_extents, halo_periods, reduce_halo),


# Define VJP for custom halo_exchange operation
halo_exchange.defvjp(_halo_fwd_rule, _halo_bwd_rule)

# JIT compile the halo_exchange operation
halo_exchange = jax.jit(halo_exchange, static_argnums=(1, 2, 3))
