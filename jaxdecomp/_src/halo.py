from functools import partial
from typing import Tuple

import jax
import jaxlib.mlir.ir as ir
import numpy as np
from jax._src.interpreters import mlir
from jax.core import Primitive
from jax.experimental.custom_partitioning import custom_partitioning
from jax.interpreters import ad, xla
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jaxlib.hlo_helpers import custom_call

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.spmd_ops import (BasePrimitive, get_axis_size,
                                     register_primitive)


class HaloPrimitive(BasePrimitive):

  name = "halo_exchange"
  multiple_results = False
  impl_static_args = (1, 2, 3)
  inner_primitive = None

  @staticmethod
  def abstract(x, halo_extents, halo_periods, reduce_halo, pdims, global_shape):
    return x.update(shape=x.shape, dtype=x.dtype)

  @staticmethod
  def outer_abstract(x, halo_extents, halo_periods, reduce_halo, pdims,
                     global_shape):
    return x.update(shape=x.shape, dtype=x.dtype)

  @staticmethod
  def lowering(ctx, x, halo_extents, halo_periods, reduce_halo, pdims,
               global_shape):
    (x_aval,) = ctx.avals_in
    x_type = ir.RankedTensorType(x.type)
    n = len(x_type.shape)

    is_double = np.finfo(x_aval.dtype).dtype == np.float64

    # Compute the descriptor for our FFT
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

    # reduce_halo is not used in the inner primitive
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
  def impl(x, halo_extents, halo_periods, reduce_halo):

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
  def per_shard_impl(x, halo_extents, halo_periods, reduce_halo, pdims,
                     global_shape):
    output = HaloPrimitive.inner_primitive.bind(
        x,
        halo_extents=halo_extents,
        halo_periods=halo_periods,
        reduce_halo=reduce_halo,
        pdims=pdims,
        global_shape=global_shape,
    )

    if reduce_halo:

      halo_x, halo_y, halo_z = halo_extents

      ## Apply correction along x
      if halo_x > 0:
        output = output.at[halo_x:halo_x + halo_x // 2].add(output[:halo_x //
                                                                   2])
        output = output.at[-(halo_x + halo_x // 2):-halo_x].add(
            output[-halo_x // 2:])
      ## Apply correction along y
      if halo_y > 0:
        output = output.at[:, halo_y:halo_y + halo_y // 2].add(
            output[:, :halo_y // 2])
        output = output.at[:, -(halo_y + halo_y // 2):-halo_y].add(
            output[:, -halo_y // 2:])
      ## Apply correction along z
      if halo_z > 0:
        output = output.at[:, :, halo_z:halo_z + halo_z // 2].add(
            output[:, :, :halo_z // 2])
        output = output.at[:, :, -(halo_z + halo_z // 2):-halo_z].add(
            output[:, :, -halo_z // 2:])

    return output

  @staticmethod
  def infer_sharding_from_operands(halo_extents, halo_periods, reduce_halo,
                                   mesh, arg_shapes, result_shape):
    # Sharding is the same here aswell because halo_exchange is a pointwise operation
    halo_exchange_sharding = arg_shapes[0].sharding
    return NamedSharding(mesh, P(*halo_exchange_sharding.spec))

  @staticmethod
  def partition(halo_extents, halo_periods, reduce_halo, mesh, arg_shapes,
                result_shape):

    halo_exchange_sharding = NamedSharding(mesh,
                                           P(*arg_shapes[0].sharding))
    global_shape = arg_shapes[0].shape
    pdims = (get_axis_size(halo_exchange_sharding,
                           1), get_axis_size(halo_exchange_sharding, 0))

    shape_without_halo = (global_shape[0] - 2 * pdims[1] * halo_extents[0],\
                          global_shape[1] - 2 * pdims[0] * halo_extents[1],\
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


def halo_p_lower(x, halo_extents, halo_periods, reduce_halo):

  return HaloPrimitive.outer_primitive.bind(
      x,
      halo_extents=halo_extents,
      halo_periods=halo_periods,
      reduce_halo=reduce_halo,
  )


# declare primitive


# Custom Partitioning
@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def halo_exchange(x, halo_extents, halo_periods, reduce_halo=False):
  output, _ = _halo_fwd_rule(x, halo_extents, halo_periods, reduce_halo)
  return output


def _halo_fwd_rule(x, halo_extents, halo_periods, reduce_halo):
  # Linear function has no residuals
  return halo_p_lower(x, halo_extents, halo_periods, reduce_halo), None


def _halo_bwd_rule(halo_extents, halo_periods, reduce_halo, ctx, g):
  return halo_p_lower(g, halo_extents, halo_periods, reduce_halo),


halo_exchange.defvjp(_halo_fwd_rule, _halo_bwd_rule)

halo_exchange = jax.jit(halo_exchange, static_argnums=(1, 2, 3))
