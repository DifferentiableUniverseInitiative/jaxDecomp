from functools import partial
from typing import Any

import jax
import jax.ffi
import jaxlib.mlir.ir as ir
import numpy as np
from jax import ShapeDtypeStruct
from jax._src.interpreters import mlir
from jax._src.typing import Array
from jax.core import ShapedArray
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomplib import _jaxdecomp

import jaxdecomp
from jaxdecomp._src.error import error_during_jacfwd, error_during_jacrev
from jaxdecomp._src.pencil_utils import get_pdims_from_mesh, validate_spec_matches_mesh
from jaxdecomp._src.spmd_ops import (
    BasePrimitive,
    register_primitive,
)
from jaxdecomp.typing import GdimsType, HaloExtentType, PdimsType, Periodicity


class HaloPrimitive(BasePrimitive):
    """
    Custom primitive for performing halo exchange operation.
    """

    name: str = 'halo_exchange'
    multiple_results: bool = False
    impl_static_args: tuple[int, int] = (1, 2)
    inner_primitive: Any = None
    outer_primitive: Any = None

    @staticmethod
    def abstract(
        x: Array,
        halo_extents: HaloExtentType,
        halo_periods: Periodicity,
        pdims: PdimsType,
        global_shape: GdimsType,
    ) -> ShapedArray:
        """
        Abstract function for determining the shape and dtype after the halo exchange operation.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : HaloExtentType
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Periodicity
            Periodicity of the halo in x, y, and z dimensions.
        pdims : PdimsType
            Processor dimensions.
        global_shape : GdimsType
            Global shape of the array.

        Returns
        -------
        ShapedArray
            Output aval.
        """
        del halo_extents, halo_periods, pdims, global_shape
        return ShapedArray(x.shape, x.dtype)

    @staticmethod
    def outer_abstract(x: Array, halo_extents: HaloExtentType, halo_periods: Periodicity) -> ShapedArray:
        """
        Abstract function for determining the shape and dtype without considering inner details.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : HaloExtentType
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Periodicity
            Periodicity of the halo in x, y, and z dimensions.

        Returns
        -------
        Array
            Abstract array after the halo exchange operation.
        """
        del halo_extents, halo_periods
        return ShapedArray(x.shape, x.dtype)

    @staticmethod
    def lowering(
        ctx,
        x,
        halo_extents: HaloExtentType,
        halo_periods: Periodicity,
        pdims: PdimsType,
        global_shape: GdimsType,
    ):
        """
        Lowering function to generate the MLIR representation for halo exchange.

        Parameters
        ----------
        ctx : Any
            Context for the operation.
        x
            Input array.
        halo_extents : HaloExtentType
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Periodicity
            Periodicity of the halo in x, y, and z dimensions.
        pdims : PdimsType
            Processor dimensions.
        global_shape : GdimsType
            Global shape of the array.

        Returns
        -------
        list
            Resulting MLIR values after the halo exchange operation.
        """
        (x_aval,) = ctx.avals_in
        x_type = ir.RankedTensorType(x.type)
        n = len(x_type.shape)
        is_double = np.finfo(x_aval.dtype).dtype == np.float64
        ffi_name = 'halo_F64' if is_double else 'halo_F32'

        lowered_halo_extents = (*halo_extents, 0)
        lowered_halo_periods = (*halo_periods, True)

        workspace_size = _jaxdecomp.get_halo_workspace_size(
            gdims=list(global_shape[::-1]),
            pdims=list(pdims),
            transpose_comm_backend=int(jaxdecomp.config.transpose_comm_backend.value),
            halo_comm_backend=int(jaxdecomp.config.halo_comm_backend.value),
            halo_extents=list(lowered_halo_extents[::-1]),
            halo_periods=list(lowered_halo_periods[::-1]),
            axis=0,
            double_precision=is_double,
        )

        layout = tuple(range(n - 1, -1, -1))
        workspace = mlir.full_like_aval(ctx, 0, ShapedArray(shape=[workspace_size], dtype=np.byte))

        rule = jax.ffi.ffi_lowering(
            ffi_name,
            operand_layouts=[layout, (0,)],
            result_layouts=[layout],
            operand_output_aliases={0: 0},
        )
        return rule(
            ctx,
            x,
            workspace,
            gdims=np.array(global_shape[::-1], dtype=np.int64),
            pdims=np.array(pdims, dtype=np.int64),
            transpose_comm_backend=np.int64(jaxdecomp.config.transpose_comm_backend.value),
            halo_comm_backend=np.int64(jaxdecomp.config.halo_comm_backend.value),
            halo_extents=np.array(lowered_halo_extents[::-1], dtype=np.int64),
            halo_periods=np.array([int(p) for p in lowered_halo_periods[::-1]], dtype=np.int64),
            axis=np.int64(0),
        )

    @staticmethod
    def batching(batched_args: tuple[Array], batched_axis, halo_extents: HaloExtentType, halo_periods: Periodicity):
        raise NotImplementedError("""
            Batching not implemented for Halo primitive using cudecomp
            Please use the JAX backend for batching
            """)

    @staticmethod
    def impl(x: Array, halo_extents: HaloExtentType, halo_periods: Periodicity) -> Array:
        """
        Implementation function for performing halo exchange.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : HaloExtentType
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Periodicity
            Periodicity of the halo in x, y, and z dimensions.

        Returns
        -------
        Array
            Result of the operation.
        """
        del halo_extents, halo_periods
        return x

    @staticmethod
    def per_shard_impl(
        x: Array,
        halo_extents: HaloExtentType,
        halo_periods: Periodicity,
        pdims: PdimsType,
        global_shape: GdimsType,
    ) -> Array:
        """
        Implementation function for performing halo exchange per shard.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : HaloExtentType
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Periodicity
            Periodicity of the halo in x, y, and z dimensions.
        pdims : PdimsType
            Processor dimensions.
        global_shape : GdimsType
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
            pdims=pdims,
            global_shape=global_shape,
        )
        return output

    @staticmethod
    def infer_sharding_from_operands(
        halo_extents: HaloExtentType,
        halo_periods: Periodicity,
        mesh: Mesh,
        arg_infos: tuple[ShapeDtypeStruct],
        result_infos: tuple[ShapedArray],
    ) -> NamedSharding:
        """
        Infer sharding information for halo exchange operation.

        Parameters
        ----------
        halo_extents : HaloExtentType
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Periodicity
            Periodicity of the halo in x, y, and z dimensions.
        mesh : Mesh
            Mesh object for sharding.
        arg_infos : Tuple[ShapeDtypeStruct]
            Shapes and dtypes of input operands.
        result_infos : Tuple[ShapedArray]
            Shape and dtype of the output result.

        Returns
        -------
        NamedSharding
            Sharding information for halo exchange operation.
        """
        del halo_extents, halo_periods, result_infos, mesh
        halo_exchange_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
        if halo_exchange_sharding is None:
            error_during_jacfwd('Halo Exchange')

        if all([spec is None for spec in halo_exchange_sharding.spec]):
            error_during_jacrev('Halo Exchange')

        input_mesh: Mesh = halo_exchange_sharding.mesh  # type: ignore
        return NamedSharding(input_mesh, P(*halo_exchange_sharding.spec))

    @staticmethod
    def sharding_rule_producer(
        halo_extents: HaloExtentType,
        halo_periods: Periodicity,
        mesh: Mesh,
        arg_infos: tuple[ShapeDtypeStruct],
        result_infos: tuple[ShapedArray],
    ) -> str:
        """
        Produces sharding rule for halo exchange operation for Shardy partitioner.

        Parameters
        ----------
        halo_extents : HaloExtentType
            Extents of the halo in X and Y directions.
        halo_periods : Periodicity
            Periodicity of the halo in X and Y directions.
        mesh : Mesh
            Mesh configuration for the distributed halo exchange.
        arg_infos : Tuple[ShapeDtypeStruct]
            Information about input arguments.
        result_infos : Tuple[ShapedArray]
            Information about result.

        Returns
        -------
        str
            Einsum string describing the halo exchange operation (identity).
        """
        del result_infos, halo_extents, halo_periods, mesh

        operand = arg_infos[0]
        if operand.rank != 3:
            raise NotImplementedError(
                f'cuDecomp backend only supports 3D arrays, got rank {operand.rank}. ' 'Please use the JAX backend for batching/vmap support.'
            )

        spec = ('i', 'j', 'k')  # einsum spec for shardy
        einsum_spec = ' '.join(spec)

        # Halo exchange preserves sharding (input = output)
        return f'{einsum_spec}->{einsum_spec}'

    @staticmethod
    def partition(
        halo_extents: HaloExtentType,
        halo_periods: Periodicity,
        mesh: Mesh,
        arg_shapes: tuple[ShapeDtypeStruct],
        result_shape: ShapeDtypeStruct,
    ):
        """
        Partition function for halo exchange operation.

        Parameters
        ----------
        halo_extents : HaloExtentType
            Extents of the halo in x, y, and z dimensions.
        halo_periods : Periodicity
            Periodicity of the halo in x, y, and z dimensions.
        mesh : Mesh
            Mesh object for sharding.
        arg_shapes : Tuple[ShapeDtypeStruct]
            Shapes and dtypes of input operands.
        result_shape : ShapedArray
            Shape and dtype of the output result.

        Returns
        -------
        Tuple[Mesh, partial, NamedSharding, Tuple[NamedSharding]]
            Mesh object, implementation function, sharding information, and its tuple.
        """
        del result_shape
        halo_exchange_sharding = arg_shapes[0].sharding
        global_shape = arg_shapes[0].shape
        pdims = get_pdims_from_mesh(mesh)

        validate_spec_matches_mesh(halo_exchange_sharding.spec, mesh)

        shape_without_halo = (
            global_shape[0] - 2 * pdims[1] * halo_extents[0],
            global_shape[1] - 2 * pdims[0] * halo_extents[1],
            global_shape[2],
        )

        impl = partial(
            HaloPrimitive.per_shard_impl,
            halo_extents=halo_extents,
            halo_periods=halo_periods,
            pdims=pdims,
            global_shape=shape_without_halo,
        )

        return mesh, impl, halo_exchange_sharding, (halo_exchange_sharding,)


register_primitive(HaloPrimitive)


@partial(jax.jit, static_argnums=(1, 2))
def halo_p_lower(x: Array, halo_extents: HaloExtentType, halo_periods: Periodicity) -> Array:
    """
    Lowering function for the halo exchange operation.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : HaloExtentType
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Periodicity
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Array
        Result of the operation.
    """
    return HaloPrimitive.outer_primitive.bind(
        x,
        halo_extents=halo_extents,
        halo_periods=halo_periods,
    )


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def halo_exchange(x: Array, halo_extents: HaloExtentType, halo_periods: Periodicity) -> Array:
    """
    Halo exchange operation with custom VJP.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : HaloExtentType
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Periodicity
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Array
        Output array after the halo exchange operation.
    """
    output, _ = _halo_fwd_rule(x, halo_extents, halo_periods)
    return output


def _halo_fwd_rule(x: Array, halo_extents: HaloExtentType, halo_periods: Periodicity) -> tuple[Array, None]:
    """
    Forward rule for the halo exchange operation.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : HaloExtentType
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Periodicity
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Tuple[Array, None]
        Output array after the halo exchange operation and None for no residuals.
    """
    return halo_p_lower(x, halo_extents, halo_periods), None


def _halo_bwd_rule(halo_extents: HaloExtentType, halo_periods: Periodicity, _, g: Array) -> tuple[Array]:
    """
    Backward rule for the halo exchange operation.

    Parameters
    ----------
    halo_extents : HaloExtentType
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Periodicity
        Periodicity of the halo in x, y, and z dimensions.
    g : Array
        Gradient array.

    Returns
    -------
    Tuple[Array]
        Gradient array after the halo exchange operation.
    """
    return (halo_p_lower(g, halo_extents, halo_periods),)


# Define VJP for custom halo_exchange operation
halo_exchange.defvjp(_halo_fwd_rule, _halo_bwd_rule)
