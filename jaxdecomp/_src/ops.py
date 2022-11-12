import jaxlib.mlir.ir as ir
from jaxlib.mhlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax import abstract_arrays
from jax.interpreters import xla
from jax.interpreters import mlir

from jaxdecomp._src import _jaxdecomp

def transposeXtoY(x, global_shape):
    """Transposes distributed array
    """
    nx, ny, nz = global_shape
    return transposeXtoY_p.bind(x, nx, ny, nz)

def transposeXtoY_abstract_eval(x, nx, ny, nz):
    return abstract_arrays.ShapedArray(x.shape, x.dtype)

def transposeXtoY_lowering(ctx, x, nx, ny, nz):
    dtype = ir.RankedTensorType(x.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))
    descriptor = _jaxdecomp.build_decomp_descriptor(4, 
                                                    4, 
                                                    4)

    return [custom_call(
        b'transpose_x_y',
      [dtype],
      [x],
      backend_config=descriptor,
      operand_layouts=[layout],
      result_layouts=[layout],
      has_side_effect=True)]

transposeXtoY_p = Primitive("transposeXtoY")
transposeXtoY_p.def_impl(partial(xla.apply_primitive, transposeXtoY_p))
transposeXtoY_p.def_abstract_eval(transposeXtoY_abstract_eval)
mlir.register_lowering(transposeXtoY_p, 
                       transposeXtoY_lowering, 
                       platform='gpu')

print("ALL GOOD")