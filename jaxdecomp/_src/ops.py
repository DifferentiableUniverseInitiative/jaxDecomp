import jax.numpy as jnp
import numpy as np
import jaxlib.mlir.ir as ir
from jaxlib.mhlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax import abstract_arrays
from jax.interpreters import xla
from jax.interpreters import mlir

from jaxdecomp._src import _jaxdecomp

def transposeXtoY(x, pdims, global_shape):
    """Transposes distributed array
    """
    return transposeXtoY_p.bind(x, 
                        pdims = tuple(pdims),
                        global_shape=tuple(global_shape))

def transposeXtoY_abstract_eval(x, *, pdims, global_shape):
    return abstract_arrays.ShapedArray(x.shape, x.dtype)

def transposeXtoY_lowering(ctx, x, *, pdims, global_shape):
    dtype = ir.RankedTensorType(x.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = global_shape
    config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
    config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P

    opaque = _jaxdecomp.build_grid_config_descriptor(config)

    return [custom_call(
        'transpose_x_y',
        [dtype],
        operands=[x],
        operand_layouts=[layout],
        result_layouts=[layout],
        has_side_effect=True, 
        operand_output_aliases={0:0},
        backend_config=opaque)]

transposeXtoY_p = Primitive("transposeXtoY")
transposeXtoY_p.def_impl(partial(xla.apply_primitive, transposeXtoY_p))
transposeXtoY_p.def_abstract_eval(transposeXtoY_abstract_eval)
mlir.register_lowering(transposeXtoY_p, 
                       transposeXtoY_lowering, 
                       platform='gpu')

print("ALL GOOD")