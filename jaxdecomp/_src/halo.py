import numpy as np
import jaxlib.mlir.ir as ir
from jaxlib.hlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax.interpreters import xla
from jax._src.interpreters import mlir
import jaxdecomp
from jaxdecomp._src import _jaxdecomp
import jax
from jax.interpreters import ad

from typing import Tuple
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

# This is the inner primitive and it should not be used directly
def inner_halo_exchange(x, *, halo_extents: Tuple[int, int, int],
                  halo_periods: Tuple[bool, bool, bool],
                  reduce_halo: bool = True,
                  pdims: Tuple[int, int]=(1, 1),
                  global_shape: Tuple[int, int, int]=[1024, 1024, 1024]):
  # TODO: check float or real
  return inner_halo_p.bind(
      x,
      halo_extents=halo_extents,
      halo_periods=halo_periods,
      reduce_halo=reduce_halo,
      pdims=pdims,
      global_shape=global_shape)


def inner_halo_abstract_eval(x, halo_extents, halo_periods,reduce_halo, pdims, global_shape):
  # The return shape is equal to the global shape for the inner primitive (the one that is not exposed)
  # The return shape is equal to the slice shape for the outer primitive (the one that is exposed)
  # in all cases it is x.shape
  return x.update(shape=x.shape, dtype=x.dtype)


def inner_halo_lowering(ctx, x, *, halo_extents, halo_periods, reduce_halo, pdims, global_shape):
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


def inner_halo_transpose_rule(x, operand, halo_extents, halo_periods, pdims,
                        global_shape):
  result = halo_exchange(x, halo_extents, halo_periods, pdims, global_shape)
  return (result,)

# Custom Primitive

def get_axis_size(sharding , index):
  axis_name = sharding.spec[index]
  if axis_name == None:
    return 1
  else:
    return sharding.mesh.shape[sharding.spec[index]]

def partition(halo_extents, halo_periods,reduce_halo, mesh, arg_shapes, result_shape):

  # halo_exchange has three operands 
  # x, halo_extents, halo_periods
  # (sanity check)
  #assert len(arg_shapes) == 3 , "halo_exchange must have only 3 operands in the partitioning lower function"
  # only x is sharded the other are fully replicated
  halo_exchange_sharding = arg_shapes[0].sharding

  def lower_fn(operand):

    global_shape = arg_shapes[0].shape
    pdims = (get_axis_size(halo_exchange_sharding,1), get_axis_size(halo_exchange_sharding,0))

    shape_without_halo = (global_shape[0] - 2 * pdims[1] * halo_extents[0],\
                          global_shape[1] - 2 * pdims[0] * halo_extents[1],\
                          global_shape[2] - 2 * halo_extents[2])

    output = inner_halo_exchange(operand, halo_extents=halo_extents, \
      halo_periods=halo_periods, pdims=pdims, global_shape=shape_without_halo)

    if reduce_halo:
      ## Apply correction along x
      output = output.at[halo_extents[0]:2 * halo_extents[0]].add(output[ :halo_extents[0]])
      output = output.at[-2 * halo_extents[0]:-halo_extents[0]].add(output[-halo_extents[0]:])
      ## Apply correction along y
      output = output.at[:, halo_extents[1]:2 * halo_extents[1]].add(output[:, :halo_extents[1]])
      output = output.at[:, -2 * halo_extents[1]:-halo_extents[1]].add(output[:, -halo_extents[1]:])

    return output

  return mesh, lower_fn,  \
      result_shape.sharding, \
      (halo_exchange_sharding,)

def infer_sharding_from_operands(halo_extents, halo_periods,reduce_halo, mesh, arg_shapes, result_shape):
  # Sharding is the same here aswell because halo_exchange is a pointwise operation
  halo_exchange_sharding = arg_shapes[0].sharding
  return halo_exchange_sharding


@partial(custom_partitioning, static_argnums=(1, 2, 3))
def halo_p_lower(x, halo_extents, halo_periods, reduce_halo):

  size = jax.device_count()
  # The pdims product must be equal to the number of devices because this is checked both in the abstract eval and in cudecomp
  dummy_pdims = (1, size)
  dummy_global = x.shape
  return inner_halo_exchange(x, halo_extents=halo_extents, halo_periods=halo_periods,reduce_halo = reduce_halo,\
     pdims=dummy_pdims, global_shape=dummy_global)

# declare primitive

inner_halo_p = Primitive("halo_exchange")
inner_halo_p.def_impl(partial(xla.apply_primitive, inner_halo_p))
inner_halo_p.def_abstract_eval(inner_halo_abstract_eval)
ad.deflinear2(inner_halo_p, inner_halo_transpose_rule)
mlir.register_lowering(inner_halo_p, inner_halo_lowering, platform="gpu")

# Define the partitioning for the primitive
halo_p_lower.def_partition(
    partition=partition,
    infer_sharding_from_operands=infer_sharding_from_operands)

# Custom Partitioning
@partial(jax.custom_vjp, nondiff_argnums=(1, 2, 3))
def halo_exchange(x, halo_extents, halo_periods, reduce_halo = False):
  output , _ =  _halo_fwd_rule(x, halo_extents, halo_periods,reduce_halo)
  return output

def _halo_fwd_rule(x, halo_extents, halo_periods, reduce_halo):
  # Linear function has no residuals
  return halo_p_lower(x, halo_extents, halo_periods,reduce_halo), None

def _halo_bwd_rule(halo_extents, halo_periods, reduce_halo, ctx, g):
  return halo_p_lower(g, halo_extents, halo_periods,reduce_halo),

halo_exchange.defvjp(_halo_fwd_rule, _halo_bwd_rule)

halo_exchange = jax.jit(halo_exchange,static_argnums=(1 , 2, 3))