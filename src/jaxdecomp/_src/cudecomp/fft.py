from functools import partial
from typing import Any

import jax
import jax.ffi
import jaxlib.mlir.ir as ir
import numpy as np
from jax import ShapeDtypeStruct
from jax._src.interpreters import mlir
from jax._src.numpy.util import promote_dtypes_complex
from jax.core import ShapedArray
from jax.extend.core import Primitive
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomplib import _jaxdecomp
from jaxtyping import Array

import jaxdecomp
from jaxdecomp._src.error import error_during_jacfwd, error_during_jacrev
from jaxdecomp._src.fft_utils import FftType, FORWARD_FFTs, fftn
from jaxdecomp._src.pencil_utils import (
    get_lowering_args,
    get_output_specs,
    get_pdims_from_sharding,
    get_pencil_type,
    get_transpose_order,
    validate_spec_matches_mesh,
)
from jaxdecomp._src.spmd_ops import (
    BasePrimitive,
    register_primitive,
)
from jaxdecomp.typing import GdimsType, TransposablePdimsType


class FFTPrimitive(BasePrimitive):
    """
    Custom primitive for FFT operations.
    """

    name: str = 'fft'
    multiple_results: bool = False
    impl_static_args: tuple[int, int] = (1, 2)
    inner_primitive: Any = None
    outer_primitive: Any = None

    @staticmethod
    def abstract(
        x: Array,
        fft_type: FftType,
        pdims: TransposablePdimsType,
        global_shape: GdimsType,
        adjoint: bool,
        mesh: Mesh,
    ) -> ShapedArray:
        """
        Abstract function to compute the shape of FFT output.

        Parameters
        ----------
        x : Array
            Input array.
        fft_type : FftType
            Type of FFT operation (forward or inverse).
        pdims : TransposablePdimsType
            Parallel dimensions.
        global_shape : GdimsType
            Global shape of the array.
        adjoint : bool
            Whether to compute the adjoint FFT.
        mesh : Mesh
            The device mesh.

        Returns
        -------
        ShapedArray
            Output aval.
        """
        del mesh

        transpose_shape = get_transpose_order(fft_type)
        output_shape = (
            global_shape[transpose_shape[0]] // pdims[0],
            global_shape[transpose_shape[1]] // pdims[1],
            global_shape[transpose_shape[2]] // pdims[2],
        )
        return ShapedArray(output_shape, x.dtype)

    @staticmethod
    def outer_abstract(x: Array, fft_type: FftType, adjoint: bool) -> ShapedArray:
        """
        Abstract function for outer FFT operation.

        Parameters
        ----------
        x : Array
            Input array.
        fft_type : FftType
            Type of FFT operation (forward or inverse).
        adjoint : bool
            Whether to compute the adjoint FFT.

        Returns
        -------
        ShapedArray
            Shape of the output array.
        """
        del adjoint

        transpose_shape = get_transpose_order(fft_type)

        output_shape = tuple([x.shape[i] for i in transpose_shape])
        return ShapedArray(output_shape, x.dtype)

    @staticmethod
    def batching(batched_args: tuple[Array], batched_axis, fft_type: FftType, adjoint: bool):
        raise NotImplementedError("""
            Batching not implemented for FFT primitive using cudecomp
            Please use the JAX backend for batching
            """)

    @staticmethod
    def lowering(
        ctx,
        a,
        *,
        fft_type: FftType,
        pdims: TransposablePdimsType,
        global_shape: GdimsType,
        adjoint: bool,
        mesh: Mesh,
    ):
        """
        Lowering function for FFT primitive.

        Parameters
        ----------
        ctx
            Context.
        a
            Input array.
        fft_type : FftType
            Type of FFT operation (forward or inverse).
        pdims : TransposablePdimsType
            Parallel dimensions.
        global_shape : GdimsType
            Global shape of the array.
        adjoint : bool
            Whether to compute the adjoint FFT.
        mesh : Mesh
            The device mesh.

        Returns
        -------
        list
            List of results from the operation.
        """
        del pdims
        (x_aval,) = ctx.avals_in
        (aval_out,) = ctx.avals_out
        dtype = x_aval.dtype
        a_type = ir.RankedTensorType(a.type)

        assert fft_type in (
            FftType.FFT,
            FftType.IFFT,
        ), 'Only complex FFTs are currently supported'

        forward = fft_type in (FftType.FFT,)
        is_double = np.finfo(dtype).dtype == np.float64
        ffi_name = 'pfft_C128' if is_double else 'pfft_C64'

        pencil_type = get_pencil_type(mesh)
        original_pdims, global_shape = get_lowering_args(fft_type, global_shape, mesh)
        local_transpose = jaxdecomp.config.transpose_axis_contiguous

        workspace_size = _jaxdecomp.get_fft_workspace_size(
            gdims=list(global_shape[::-1]),
            pdims=list(original_pdims),
            transpose_comm_backend=int(jaxdecomp.config.transpose_comm_backend.value),
            halo_comm_backend=int(jaxdecomp.config.halo_comm_backend.value),
            forward=forward,
            double_precision=is_double,
            adjoint=adjoint,
            contiguous=local_transpose,
            decomposition=int(pencil_type.value),
        )

        n = len(a_type.shape)
        layout = tuple(range(n - 1, -1, -1))
        workspace = mlir.full_like_aval(ctx, 0, ShapedArray(shape=[workspace_size], dtype=np.byte))

        rule = jax.ffi.ffi_lowering(
            ffi_name,
            operand_layouts=(layout, (0,)),
            result_layouts=(layout,),
            skip_ffi_layout_processing=True,
        )
        result = rule(
            ctx,
            a,
            workspace,
            gdims=np.array(global_shape[::-1], dtype=np.int64),
            pdims=np.array(original_pdims, dtype=np.int64),
            transpose_comm_backend=np.int64(jaxdecomp.config.transpose_comm_backend.value),
            halo_comm_backend=np.int64(jaxdecomp.config.halo_comm_backend.value),
            forward=forward,
            adjoint=adjoint,
            contiguous=local_transpose,
            decomposition=np.int64(pencil_type.value),
        )

        return result

    @staticmethod
    def impl(x: Array, fft_type: FftType, adjoint: bool) -> Array:
        """
        Implementation function for FFT primitive.

        Parameters
        ----------
        x : Array
            Input array.
        fft_type : FftType
            Type of FFT operation (forward or inverse).
        adjoint : bool
            Whether to compute the adjoint FFT.

        Returns
        -------
        Array
            Result of the FFT operation.
        """
        assert isinstance(fft_type, FftType)  # type: ignore

        transpose_order = get_transpose_order(fft_type)

        return fftn(x, fft_type=fft_type, adjoint=adjoint).transpose(transpose_order)

    @staticmethod
    def per_shard_impl(
        x: Array,
        fft_type: FftType,
        pdims: TransposablePdimsType,
        global_shape: GdimsType,
        adjoint: bool,
        mesh: Mesh,
    ) -> Array:
        """
        Implementation function for per-shard FFT primitive.

        Parameters
        ----------
        x : Array
            Input array.
        fft_type : FftType
            Type of FFT operation (forward or inverse).
        pdims : TransposablePdimsType
            Parallel dimensions.
        global_shape : GdimsType
            Global shape of the array.
        adjoint : bool
            Whether to compute the adjoint FFT.
        mesh : Mesh
            The device mesh.

        Returns
        -------
        Array
            Result of the per-shard FFT operation.
        """
        if fft_type == FftType.IFFT and pdims[0] == 1:
            if jaxdecomp.config.transpose_axis_contiguous:
                x = x.transpose([1, 2, 0])
        assert FFTPrimitive.inner_primitive is not None

        out = FFTPrimitive.inner_primitive.bind(
            x,
            fft_type=fft_type,
            pdims=pdims,
            global_shape=global_shape,
            adjoint=adjoint,
            mesh=mesh,
        )

        return out

    @staticmethod
    def infer_sharding_from_operands(
        fft_type: FftType,
        adjoint: bool,
        mesh: Mesh,
        arg_infos: tuple[ShapeDtypeStruct],
        result_infos: tuple[ShapedArray],
    ) -> NamedSharding:
        """
        Infer sharding for FFT primitive based on operands.

        Parameters
        ----------
        fft_type : FftType
            Type of FFT operation (forward or inverse).
        adjoint : bool
            Whether to compute the adjoint FFT.
        mesh : Mesh
            The device mesh.
        arg_infos : Tuple[ShapeDtypeStruct]
            Shape and sharding information of input operands.
        result_infos : Tuple[ShapedArray]
            Shape information of output.

        Returns
        -------
        NamedSharding
            Sharding information for the result.
        """
        del adjoint, mesh, result_infos

        input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
        if input_sharding is None:
            error_during_jacfwd('pfft')

        if all([spec is None for spec in input_sharding.spec]):
            error_during_jacrev('pfft')

        operand = arg_infos[0]
        if operand.ndim != 3:
            raise NotImplementedError(
                f'cuDecomp backend only supports 3D arrays, got {operand.ndim}D. ' 'Please use the JAX backend for batching/vmap support.'
            )

        spec = input_sharding.spec
        input_mesh: Mesh = input_sharding.mesh  # type: ignore
        pencil_type = get_pencil_type(input_mesh)
        transposed_specs = get_output_specs(fft_type, pencil_type, spec, 'cudecomp')
        return NamedSharding(input_mesh, P(*transposed_specs))

    @staticmethod
    def sharding_rule_producer(
        fft_type: FftType,
        adjoint: bool,
        mesh: Mesh,
        arg_infos: tuple[ShapeDtypeStruct],
        result_infos: tuple[ShapedArray],
    ) -> str:
        """
        Produces sharding rule for FFT operation for Shardy partitioner.

        Parameters
        ----------
        fft_type : FftType
            Type of FFT operation to perform.
        adjoint : bool
            Whether this is an adjoint (inverse) FFT operation.
        mesh : Mesh
            Mesh configuration for the distributed FFT.
        arg_infos : tuple[ShapeDtypeStruct]
            Information about input arguments.
        result_infos : tuple[ShapedArray]
            Information about result.

        Returns
        -------
        str
            Einsum string describing the FFT operation.
        """
        del adjoint, result_infos

        operand = arg_infos[0]
        if operand.rank != 3:
            raise NotImplementedError(
                f'cuDecomp backend only supports 3D arrays, got rank {operand.rank}. ' 'Please use the JAX backend for batching/vmap support.'
            )

        pencil_type = get_pencil_type(mesh)
        spec = ('i', 'j', 'k')  # einsum spec for shardy
        transposed_specs: tuple[str, ...] = get_output_specs(fft_type, pencil_type, spec, 'cudecomp')  # type: ignore
        einsum_in = ' '.join(spec)
        einsum_out = ' '.join(transposed_specs)
        return f'{einsum_in}->{einsum_out}'

    @staticmethod
    def partition(
        fft_type: FftType,
        adjoint: bool,
        mesh: Mesh,
        arg_shapes: tuple[ShapeDtypeStruct],
        result_shape: ShapeDtypeStruct,
    ) -> tuple[Mesh, partial, NamedSharding, tuple[NamedSharding]]:
        """
        Partition the FFT primitive for XLA.

        Parameters
        ----------
        fft_type : FftType
            Type of FFT operation (forward or inverse).
        adjoint : bool
            Whether to compute the adjoint FFT.
        mesh : Mesh
            The device mesh.
        arg_shapes : Tuple[ShapeDtypeStruct]
            Shape and sharding information of input operands.
        result_shape : ShapeDtypeStruct
            Shape and sharding information of output.

        Returns
        -------
        Tuple[Mesh, partial, NamedSharding, Tuple[NamedSharding]]
            Mesh, lowered function, output sharding, and input operand sharding.
        """
        input_sharding: NamedSharding = arg_shapes[0].sharding  # type: ignore
        output_sharding: NamedSharding = result_shape.sharding  # type: ignore
        input_mesh: Mesh = input_sharding.mesh  # type: ignore
        global_shape = arg_shapes[0].shape
        pdims = get_pdims_from_sharding(output_sharding)

        if fft_type in FORWARD_FFTs:
            validate_spec_matches_mesh(input_sharding.spec, input_mesh)

        impl = partial(
            FFTPrimitive.per_shard_impl,
            fft_type=fft_type,
            pdims=pdims,
            global_shape=global_shape,
            adjoint=adjoint,
            mesh=input_mesh,
        )

        return mesh, impl, output_sharding, (input_sharding,)


register_primitive(FFTPrimitive)


@partial(jax.jit, static_argnums=(1, 2))
def pfft_impl(x: Array, fft_type: FftType, adjoint: bool) -> Array:
    """
    Lowering function for pfft primitive.

    Parameters
    ----------
    x : Array
      Input array.
    fft_type : Union[str, lax.FftType]
      Type of FFT operation.
    adjoint : bool
      Whether to compute the adjoint FFT.

    Returns
    -------
    Primitive
      Result of the operation.
    """
    (x,) = promote_dtypes_complex(x)

    assert FFTPrimitive.outer_primitive is not None

    return FFTPrimitive.outer_primitive.bind(x, fft_type=fft_type, adjoint=adjoint)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def pfft(x: Array, fft_type: FftType, adjoint: bool = False) -> Primitive:
    """
    Custom VJP definition for pfft.

    Parameters
    ----------
    x : Array
      Input array.
    fft_type : Union[str, lax.FftType]
      Type of FFT operation.
    adjoint : bool, optional
      Whether to compute the adjoint FFT. Defaults to False.

    Returns
    -------
    Primitive
      Result of the operation.
    """
    output, _ = _pfft_fwd_rule(x, fft_type=fft_type, adjoint=adjoint)
    return output


def _pfft_fwd_rule(x: Array, fft_type: FftType, adjoint: bool) -> tuple[Primitive, None]:
    """
    Forward rule for pfft.

    Parameters
    ----------
    x : Array
      Input array.
    fft_type : Union[str, lax.FftType]
      Type of FFT operation.
    adjoint : bool, optional
      Whether to compute the adjoint FFT. Defaults to False.

    Returns
    -------
    Tuple[Primitive, None]
      Result of the operation and None (no residuals).
    """
    return pfft_impl(x, fft_type=fft_type, adjoint=adjoint), None


def _pfft_bwd_rule(fft_type: FftType, adjoint: bool, _, g: Primitive) -> tuple[Primitive]:
    """
    Backward rule for pfft.

    Parameters
    ----------
    fft_type : Union[str, lax.FftType]
      Type of FFT operation.
    adjoint : bool
      Whether to compute the adjoint FFT.
    ctx
      Context.
    g : Primitive
      Gradient value.

    Returns
    -------
    Tuple[Primitive]
      Result of the operation.
    """
    assert fft_type in [FftType.FFT, FftType.IFFT]
    if fft_type == FftType.FFT:
        fft_type = FftType.IFFT
    elif fft_type == FftType.IFFT:
        fft_type = FftType.FFT

    return (pfft_impl(g, fft_type, not adjoint),)


pfft.defvjp(_pfft_fwd_rule, _pfft_bwd_rule)
