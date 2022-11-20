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


def pfft3d(x, global_shape, pdims):
    return pfft3d_p.bind(x,
                         global_shape=tuple(global_shape),
                         pdims=tuple(pdims)) 

def pfft3d_abstract_eval(x, *, global_shape, pdims):
    return abstract_arrays.ShapedArray(x.shape, x.dtype)


def pfft3d_lowering(ctx, x, *, global_shape, pdims):
    dtype = ir.RankedTensorType(x.type)
    layout = tuple([0,1,2]) # TODO: fix to avoid the memory transpose

    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = global_shape
    config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
    config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P

    opaque = _jaxdecomp.build_grid_config_descriptor(config)

    return [
        custom_call(
            'pfft3d', [dtype],
            operands=[x],
            operand_layouts=[layout],
            result_layouts=[layout],
            has_side_effect=True,
            operand_output_aliases={0: 0},
            backend_config=opaque)
    ]

pfft3d_p = Primitive("pfft3d")
pfft3d_p.def_impl(partial(xla.apply_primitive, pfft3d_p))
pfft3d_p.def_abstract_eval(pfft3d_abstract_eval)
mlir.register_lowering(pfft3d_p, pfft3d_lowering, platform='gpu')

def ipfft3d(x, global_shape, pdims):
    return ipfft3d_p.bind(x,
                         global_shape=tuple(global_shape),
                         pdims=tuple(pdims)) / np.prod(global_shape)

def ipfft3d_abstract_eval(x, *, global_shape, pdims):
    return abstract_arrays.ShapedArray(x.shape, x.dtype)


def ipfft3d_lowering(ctx, x, *, global_shape, pdims):
    dtype = ir.RankedTensorType(x.type)
    layout = tuple([0,1,2]) # TODO: fix to avoid the memory transpose

    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = global_shape
    config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
    config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P

    opaque = _jaxdecomp.build_grid_config_descriptor(config)

    return [
        custom_call(
            'ipfft3d', [dtype],
            operands=[x],
            operand_layouts=[layout],
            result_layouts=[layout],
            has_side_effect=True,
            operand_output_aliases={0: 0},
            backend_config=opaque)
    ]

ipfft3d_p = Primitive("ipfft3d")
ipfft3d_p.def_impl(partial(xla.apply_primitive, ipfft3d_p))
ipfft3d_p.def_abstract_eval(ipfft3d_abstract_eval)
mlir.register_lowering(ipfft3d_p, ipfft3d_lowering, platform='gpu')


def transposeXtoY(x):
    """Transposes distributed array
    """
    pdims = x.sharding.shape[:2] # TODO: fix this durty hack
    return transposeXtoY_p.bind(x,
                                pdims=pdims)

def transposeXtoY_abstract_eval(x, *, pdims):
    return abstract_arrays.ShapedArray(x.shape, x.dtype)


def transposeXtoY_lowering(ctx, x, *, pdims):
    dtype = ir.RankedTensorType(x.type)
    dims = dtype.shape
    layout = tuple(range(len(dims) - 1, -1, -1))

    config = _jaxdecomp.GridConfig()
    config.pdims = pdims
    config.gdims = x.shape
    config.halo_comm_backend = _jaxdecomp.HALO_COMM_MPI
    config.transpose_comm_backend = _jaxdecomp.TRANSPOSE_COMM_MPI_P2P

    opaque = _jaxdecomp.build_grid_config_descriptor(config)

    return [
        custom_call(
            'transpose_x_y', [dtype],
            operands=[x],
            operand_layouts=[layout],
            result_layouts=[layout],
            has_side_effect=True,
            operand_output_aliases={0: 0},
            backend_config=opaque)
    ]


transposeXtoY_p = Primitive("transposeXtoY")
transposeXtoY_p.def_impl(partial(xla.apply_primitive, transposeXtoY_p))
transposeXtoY_p.def_abstract_eval(transposeXtoY_abstract_eval)
mlir.register_lowering(transposeXtoY_p, transposeXtoY_lowering, platform='gpu')
