import numpy as np
import jaxlib.mlir.ir as ir
from jaxlib.mhlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax import abstract_arrays
from jax.interpreters import xla
from jax.interpreters import mlir
from jax._src.lib.mlir.dialects import mhlo

from jaxdecomp._src import _jaxdecomp

_out_axes = {'x_y': 1, 'y_z': 2, 'z_y': 1, 'y_x': 0}

def transpose_abstract_eval(x, *, kind, pdims, global_shape):
  assert kind in ['x_y', 'y_z', 'z_y', 'y_x']
  config = _jaxdecomp.GridConfig()
  config.pdims = pdims
  config.gdims = global_shape[::-1]
  config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
  config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P

  pencil = _jaxdecomp.get_pencil_info(config, _out_axes[kind])
  shape = pencil.shape[::-1]
  assert np.prod(shape) == np.prod(x.shape), "Only array dimensions divisible by the process mesh size are currently supported. The current configuration leads to local slices of varying sizes between forward and reverse FFT."
  return abstract_arrays.ShapedArray(shape, x.dtype)

def transpose_lowering(ctx, x, *, kind, pdims, global_shape):
  assert kind in ['x_y', 'y_z', 'z_y', 'y_x']
  (aval_out, ) = ctx.avals_out
  x_type = ir.RankedTensorType(x.type)
  layout = tuple(range(len(x_type.shape) - 1, -1, -1))

  config = _jaxdecomp.GridConfig()
  config.pdims = pdims
  config.gdims = global_shape[::-1]
  config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
  config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P

  opaque = _jaxdecomp.build_transpose_descriptor(config)

  result = custom_call(
          "transpose_"+kind,
          [x_type],
          operands=[x],
          operand_layouts=[layout],
          result_layouts=[layout],
          has_side_effect=True,
          operand_output_aliases={0: 0},
          backend_config=opaque,
      )
  # Finally we reshape the arry to the expected shape.
  return mhlo.ReshapeOp(mlir.aval_to_ir_type(aval_out), result).results



def transposeXtoY(x, *, pdims, global_shape):
  """Transposes distributed array"""
  return transposeXtoY_p.bind(x, kind="x_y", pdims=pdims, global_shape=global_shape)

transposeXtoY_p = Primitive("transposeXtoY")
transposeXtoY_p.def_impl(partial(xla.apply_primitive, transposeXtoY_p))
transposeXtoY_p.def_abstract_eval(transpose_abstract_eval)
mlir.register_lowering(transposeXtoY_p, transpose_lowering, platform="gpu")


def transposeYtoZ(x, *, pdims, global_shape):
  """Transposes distributed array"""
  return transposeYtoZ_p.bind(x, kind="y_z", pdims=pdims, global_shape=global_shape)

transposeYtoZ_p = Primitive("transposeYtoZ")
transposeYtoZ_p.def_impl(partial(xla.apply_primitive, transposeYtoZ_p))
transposeYtoZ_p.def_abstract_eval(transpose_abstract_eval)
mlir.register_lowering(transposeYtoZ_p, transpose_lowering, platform="gpu")


def transposeZtoY(x, *, pdims, global_shape):
  """Transposes distributed array"""
  return transposeZtoY_p.bind(x, kind="z_y", pdims=pdims, global_shape=global_shape)

transposeZtoY_p = Primitive("transposeZtoY")
transposeZtoY_p.def_impl(partial(xla.apply_primitive, transposeZtoY_p))
transposeZtoY_p.def_abstract_eval(transpose_abstract_eval)
mlir.register_lowering(transposeZtoY_p, transpose_lowering, platform="gpu")


def transposeYtoX(x, *, pdims, global_shape):
  """Transposes distributed array"""
  return transposeYtoX_p.bind(x, kind="y_x", pdims=pdims, global_shape=global_shape)

transposeYtoX_p = Primitive("transposeYtoX")
transposeYtoX_p.def_impl(partial(xla.apply_primitive, transposeYtoX_p))
transposeYtoX_p.def_abstract_eval(transpose_abstract_eval)
mlir.register_lowering(transposeYtoX_p, transpose_lowering, platform="gpu")
