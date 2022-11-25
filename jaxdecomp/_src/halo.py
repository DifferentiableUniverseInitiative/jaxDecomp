import numpy as np
import jaxlib.mlir.ir as ir
from jaxlib.mhlo_helpers import custom_call
from functools import partial
from jax.core import Primitive
from jax.interpreters import xla
from jax.interpreters import mlir
from jaxdecomp._src import _jaxdecomp
import jax
from jax.interpreters import ad


def halo_exchange(x, *, halo_extents, halo_periods, pdims, global_shape):
    # TODO: check float or real
    return halo_p.bind(x, halo_extents=halo_extents, halo_periods=halo_periods, pdims=pdims, global_shape=global_shape)


def halo_abstract_eval(x, halo_extents, halo_periods, pdims, global_shape):
    return x.update(shape=x.shape, dtype=x.dtype)

def halo_lowering(ctx, x, *, halo_extents, halo_periods, pdims, global_shape):
    (x_aval,) = ctx.avals_in
    x_type = ir.RankedTensorType(x.type)
    n = len(x_type.shape)

    is_double = np.finfo(x_aval.dtype).dtype == np.float64

    # Compute the descriptor for our FFT
    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = global_shape[::-1]
    config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
    config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P

    workspace_size, opaque = _jaxdecomp.build_halo_descriptor(config, is_double,
                                    halo_extents[::-1], halo_periods[::-1], 0)
    layout = tuple(range(n - 1, -1, -1))

    workspace = mlir.full_like_aval(0, jax.core.ShapedArray(shape=[workspace_size],
                                  dtype=np.byte));
    return [custom_call(
                "halo",
                [x_type],
                operands=[x, workspace],
                operand_layouts=[layout, (0,)],
                result_layouts=[layout],
                has_side_effect=True,
                operand_output_aliases= {0:0}, 
                backend_config=opaque,
            )
            ]

def halo_transpose_rule(x, operand, halo_extents, halo_periods, pdims, global_shape):
    result = halo_exchange(x, halo_extents, halo_periods, pdims, global_shape)
    return (result,)

halo_p = Primitive("halo_exchange")
halo_p.def_impl(partial(xla.apply_primitive, halo_p))
halo_p.def_abstract_eval(halo_abstract_eval)
ad.deflinear2(halo_p, halo_transpose_rule)
mlir.register_lowering(halo_p, halo_lowering, platform="gpu")

