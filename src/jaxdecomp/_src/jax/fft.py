from functools import partial

import jax
from jax import lax
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ShapedArray
from jax._src.interpreters import batching
from jax._src.typing import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomplib import _jaxdecomp

import jaxdecomp
from jaxdecomp._src.error import error_during_jacfwd, error_during_jacrev
from jaxdecomp._src.fft_utils import (
    ADJOINT,
    COMPLEX,  # yapf: disable
    FftType,  # yapf: disable
    FORWARD_FFTs,
    fft,
    fft2,
    fftn,
)
from jaxdecomp._src.pencil_utils import get_axis_names_from_mesh, get_output_specs, get_pencil_type_from_axis_names, get_transpose_order
from jaxdecomp._src.spmd_ops import custom_spmd_rule


def _fft_slab_xy(operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str) -> Array:
    """
    Performs the FFT slab XY operation.

    Parameters
    ----------
    operand : Array
        Input array for the FFT operation.
    fft_type : FftType
        Type of FFT to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    x_axis_name : str
        Axis name for the X axis.

    Returns
    -------
    Array
        Resulting array after the FFT slab XY operation.
    """
    operand = fft2(operand, fft_type, axes=(2, 1), adjoint=adjoint)
    if jaxdecomp.config.transpose_axis_contiguous:
        operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
    else:
        operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True)
        operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
    return operand


def _fft_slab_yz(operand: Array, fft_type: FftType, adjoint: bool, y_axis_name: str) -> Array:
    """
    Performs the FFT slab YZ operation.

    Parameters
    ----------
    operand : Array
        Input array for the FFT operation.
    fft_type : FftType
        Type of FFT to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    y_axis_name : str
        Axis name for the Y axis.

    Returns
    -------
    Array
        Resulting array after the FFT slab YZ operation.
    """
    operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
    if jaxdecomp.config.transpose_axis_contiguous:
        # transpose to (X / py, Z , Y) with specs P('y', 'z')
        operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
        # FFT on YZ plane
        operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)
        operand = operand.transpose([2, 0, 1])
    else:
        # transpose to (Z , Y , X / py) with specs P('z', None, 'y')
        operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True)
        # FFT on YZ plane
        operand = fft2(operand, COMPLEX(fft_type), axes=(1, 0), adjoint=adjoint)

    return operand


def _fft_pencils(operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str, y_axis_name: str) -> Array:
    """
    Performs the FFT pencils operation.

    Parameters
    ----------
    operand : Array
        Input array for the FFT operation.
    fft_type : FftType
        Type of FFT to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    x_axis_name : str
        Axis name for the X axis.
    y_axis_name : str
        Axis name for the Y axis.

    Returns
    -------
    Array
        Resulting array after the FFT pencils operation.
    """
    operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
    if jaxdecomp.config.transpose_axis_contiguous:
        # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
        operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
        # FFT on the Y axis
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
        # transpose to (Y / pz, X / py, Z) with specs P('z', 'y')
        operand = lax.all_to_all(operand, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
        # FFT on the Z axis
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
    else:
        # transpose to (Z / Pz , Y , X / py) with specs P('z', None, 'y')
        operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True)
        # FFT on the Y axis
        operand = fft(operand, COMPLEX(fft_type), axis=1, adjoint=adjoint)
        # transpose to (Z , Y / pz, X / Py) with specs P(None , 'z', 'y')
        operand = lax.all_to_all(operand, x_axis_name, 1, 0, tiled=True)
        # FFT on the Z axis
        operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
    return operand


def _ifft_slab_xy(operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str) -> Array:
    """
    Performs the IFFT slab XY operation.

    Parameters
    ----------
    operand : Array
        Input array for the IFFT operation.
    fft_type : FftType
        Type of IFFT to perform.
    adjoint : bool
        Whether to compute the adjoint IFFT.
    x_axis_name : str
        Axis name for the X axis.

    Returns
    -------
    Array
        Resulting array after the IFFT slab XY operation.
    """
    if jaxdecomp.config.transpose_axis_contiguous:
        # input is (Y , X/Pz , Z) with specs P('y', 'z')
        # First IFFT on Z
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
        # transpose to (Z / Pz , Y , X) with specs P('z', 'y')
        operand = lax.all_to_all(operand, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
        # IFFT on XY plane
        operand = fft2(operand, fft_type, axes=(2, 1), adjoint=adjoint)
    else:
        # input is (Z , Y , X/Pz) with specs P(None, 'y', 'z')
        # First IFFT on Z
        operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
        # transpose to (Z/Pz , Y , X) with specs P('z', 'y')
        operand = lax.all_to_all(operand, x_axis_name, 0, 2, tiled=True)
        # IFFT on XY plane
        operand = fft2(operand, fft_type, axes=(2, 1), adjoint=adjoint)
    return operand


def _ifft_slab_yz(operand: Array, fft_type: FftType, adjoint: bool, y_axis_name: str) -> Array:
    """
    Performs the IFFT slab YZ operation.

    Parameters
    ----------
    operand : Array
        Input array for the IFFT operation.
    fft_type : FftType
        Type of IFFT to perform.
    adjoint : bool
        Whether to compute the adjoint IFFT.
    y_axis_name : str
        Axis name for the Y axis.

    Returns
    -------
    Array
        Resulting array after the IFFT slab YZ operation.
    """
    if jaxdecomp.config.transpose_axis_contiguous:
        operand = operand.transpose([1, 2, 0])
        # input is (X / py, Z , Y) with specs P('y', 'z')
        # First IFFT
        operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)
        # transpose to (Z , Y / Py, X) with specs P('z', 'y')
        operand = lax.all_to_all(operand, y_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
        # IFFT on X axis
        operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
    else:
        # input is (Z , Y , X / py) with specs P('z', None, 'y')
        # First IFFT on Y
        operand = fft2(operand, COMPLEX(fft_type), axes=(1, 0), adjoint=adjoint)
        # transpose to (Z , Y / py, X) with specs P('z', 'y')
        operand = lax.all_to_all(operand, y_axis_name, 1, 2, tiled=True)
        # IFFT on X axis
        operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)

    return operand


def _ifft_pencils(operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str, y_axis_name: str) -> Array:
    """
    Performs the IFFT pencils operation.

    Parameters
    ----------
    operand : Array
        Input array for the IFFT operation.
    fft_type : FftType
        Type of IFFT to perform.
    adjoint : bool
        Whether to compute the adjoint IFFT.
    x_axis_name : str
        Axis name for the X axis.
    y_axis_name : str
        Axis name for the Y axis.

    Returns
    -------
    Array
        Resulting array after the IFFT pencils operation.
    """
    if jaxdecomp.config.transpose_axis_contiguous:
        # input is (Y / pz, X / py, Z) with specs P('z', 'y')
        # First IFFT on Z
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
        # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
        operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
        # IFFT on the Y axis
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
        # transpose to (Z / Pz , Y / Py  , X) with specs P('z', 'y')
        operand = lax.all_to_all(operand, y_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
        # IFFT on the X axis
        operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
    else:
        # input is (Z / Pz , Y / Py  , X) with specs P('z', 'y')
        # First IFFT on X
        operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
        # transpose to (Y / pz, X / py, Z) with specs P('z', 'y')
        operand = lax.all_to_all(operand, x_axis_name, 0, 1, tiled=True)
        # IFFT on the Z axis
        operand = fft(operand, COMPLEX(fft_type), axis=1, adjoint=adjoint)
        # transpose to (X / py, Z / pz , Y) with specs P('y', 'z')
        operand = lax.all_to_all(operand, y_axis_name, 1, 2, tiled=True)
        # IFFT on the Y axis
        operand = fft(operand, fft_type, axis=-1, adjoint=adjoint)
    return operand


def spmd_fft(x: Array, fft_type: FftType, adjoint: bool) -> Array:
    """
    Implementation of the FFT operation using the FFT primitive.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.

    Returns
    -------
    Array
        Resulting array after the FFT operation.
    """
    transpose_order = get_transpose_order(fft_type)

    def _impl(x):
        return fftn(x, fft_type, adjoint=adjoint).transpose(transpose_order)

    if x.ndim == 3:
        return _impl(x)
    if x.ndim == 4:
        return jax.vmap(_impl)(x)
    else:
        raise ValueError(f'Unsupported input shape {x.shape}')


def per_shard_impl(x: Array, fft_type: FftType, adjoint: bool, x_axis_name: str, y_axis_name: str) -> Array:
    """
    Per-shard implementation of the FFT operation.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    mesh : Mesh
        Mesh configuration for the distributed FFT.

    Returns
    -------
    Array
        Resulting array after the per-shard FFT operation.
    """
    assert isinstance(fft_type, FftType)  # type: ignore
    pencil_type = get_pencil_type_from_axis_names(x_axis_name, y_axis_name)

    def _impl(x):
        if fft_type in FORWARD_FFTs:
            match pencil_type:
                case _jaxdecomp.SLAB_XY:
                    assert x_axis_name is not None
                    return _fft_slab_xy(x, fft_type, adjoint, x_axis_name)
                case _jaxdecomp.SLAB_YZ:
                    assert y_axis_name is not None
                    return _fft_slab_yz(x, fft_type, adjoint, y_axis_name)
                case _jaxdecomp.PENCILS:
                    assert (x_axis_name is not None) and (y_axis_name is not None)
                    return _fft_pencils(x, fft_type, adjoint, x_axis_name, y_axis_name)
                case _:
                    raise ValueError(f'Unsupported pencil type {pencil_type}')
        else:
            match pencil_type:
                case _jaxdecomp.SLAB_XY:
                    assert x_axis_name is not None
                    return _ifft_slab_xy(x, fft_type, adjoint, x_axis_name)
                case _jaxdecomp.SLAB_YZ:
                    assert y_axis_name is not None
                    return _ifft_slab_yz(x, fft_type, adjoint, y_axis_name)
                case _jaxdecomp.PENCILS:
                    assert (x_axis_name is not None) and (y_axis_name is not None)
                    return _ifft_pencils(x, fft_type, adjoint, x_axis_name, y_axis_name)
                case _:
                    raise ValueError(f'Unsupported pencil type {pencil_type}')

    if x.ndim == 3:
        return _impl(x)
    if x.ndim == 4:
        return jax.vmap(_impl)(x)
    else:
        raise ValueError(f'Unsupported input shape {x.shape}')


spmd_fft_primitive = custom_spmd_rule(spmd_fft, static_argnums=(1, 2), multiple_results=False)


@spmd_fft_primitive.def_infer_sharding
def infer_sharding_from_operands(
    fft_type: FftType,
    adjoint: bool,
    mesh: Mesh,
    arg_infos: tuple[ShapeDtypeStruct],
    result_infos: tuple[ShapedArray],
) -> NamedSharding:
    """
    Infers the sharding for the result based on the input operands.

    Parameters
    ----------
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    mesh : Mesh
        Mesh configuration for the distributed FFT.
    arg_infos : Tuple[ShapeDtypeStruct]
        Shape and dtype information of the input operands.
    result_infos : Tuple[ShapedArray]
        Shape and dtype information of the result.

    Returns
    -------
    NamedSharding
        Sharding information for the result.
    """
    del mesh, result_infos
    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    if input_sharding is None:
        error_during_jacfwd('pfft')

    if all([spec is None for spec in input_sharding.spec]):
        error_during_jacrev('pfft')

    input_mesh: Mesh = input_sharding.mesh  # type: ignore
    operand = arg_infos[0]
    if operand.ndim == 3:
        spec = input_sharding.spec
        transposed_specs = get_output_specs(fft_type, spec, mesh=input_mesh, backend='jax')
    elif operand.ndim == 4:
        assert input_sharding.spec[0] is None
        spec = input_sharding.spec[1:]
        transposed_specs = get_output_specs(fft_type, spec, mesh=input_mesh, backend='jax')
        assert len(transposed_specs) == 3
        transposed_specs = (None,) + transposed_specs
    else:
        raise ValueError(f'Unsupported input shape {operand.shape}')

    return NamedSharding(input_sharding.mesh, P(*transposed_specs))


@spmd_fft_primitive.def_sharding_rule
def fft_sharding_rule_producer(
    fft_type: FftType,
    adjoint: bool,
    mesh: Mesh,
    arg_infos,
    result_infos,
) -> str:
    del result_infos

    spec = ('i', 'j', 'k')  # einsum spec for shardy
    transposed_specs: tuple[str] = get_output_specs(fft_type, spec, mesh=mesh, backend='jax')  # type: ignore
    einsum_in = ' '.join(spec)
    einsum_out = ' '.join(transposed_specs)

    operand = arg_infos[0]
    if operand.rank == 3:
        pass
    elif operand.rank == 4:
        einsum_in = 'b ' + einsum_in
        einsum_out = 'b ' + einsum_out
    else:
        raise ValueError(f'Unsupported input shape {operand.shape}')

    return f'{einsum_in}->{einsum_out}'


@spmd_fft_primitive.def_partition
def partition(
    fft_type: FftType,
    adjoint: bool,
    mesh: Mesh,
    arg_infos: tuple[ShapeDtypeStruct],
    result_infos: tuple[ShapedArray],
):
    """
    Partitions the FFT operation for distributed execution.

    Parameters
    ----------
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    mesh : Mesh
        Mesh configuration for the distributed FFT.
    arg_infos : Tuple[ShapeDtypeStruct]
        Shape and dtype information of the input operands.
    result_infos : Tuple[ShapedArray]
        Shape and dtype information of the result.

    Returns
    -------
    Tuple
        Mesh configuration, implementation function, output sharding, and input sharding.
    """
    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    output_sharding: NamedSharding = result_infos.sharding  # type: ignore
    input_mesh: Mesh = arg_infos[0].sharding.mesh  # type: ignore
    x_axis_name, y_axis_name = get_axis_names_from_mesh(input_mesh)

    impl = partial(
        per_shard_impl,
        fft_type=fft_type,
        adjoint=adjoint,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
    )

    return mesh, impl, output_sharding, (input_sharding,)


@spmd_fft_primitive.def_transpose_rule
def transpose_rule(cotangent: Array, x: Array, fft_type: FftType, adjoint: bool) -> tuple[Array]:
    """
    Transpose rule for the FFT operation.

    Parameters
    ----------
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.
    x : Array
        Input array.

    Returns
    -------
    Array
        Resulting array after the transpose operation.
    """
    return (spmd_fft_primitive(cotangent, fft_type=ADJOINT(fft_type), adjoint=not adjoint),)


@spmd_fft_primitive.def_batching_rule
def batching_rule(batched_args: tuple[Array], batched_axis, fft_type: FftType, adjoint: bool) -> Array:
    """
    Batching rule for the FFT operation.

    Parameters
    ----------
    batched_args : Tuple[Array]
        Batched input arrays.
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.

    Returns
    -------
    Array
        Resulting array after the FFT operation.
    """
    (x,) = batched_args
    (bd,) = batched_axis
    x = batching.moveaxis(x, bd, 0)
    return spmd_fft_primitive(x, fft_type=fft_type, adjoint=adjoint), 0


@partial(jax.jit, static_argnums=(1, 2))
def pfft_impl(x: Array, fft_type: FftType, adjoint: bool) -> Array:
    """
    Lowering function for pfft primitive.

    Parameters
    ----------
    x : Array
        Input array.
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool
        Whether to compute the adjoint FFT.

    Returns
    -------
    Array
        Resulting array after the pfft operation.
    """
    return spmd_fft_primitive(x, fft_type=fft_type, adjoint=adjoint)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2))
def pfft(x: Array, fft_type: FftType, adjoint: bool = False) -> Array:
    """
    Custom VJP definition for pfft.

    Parameters
    ----------
    x : Array
        Input array.
    fft_type : FftType
        Type of FFT operation to perform.
    adjoint : bool, optional
        Whether to compute the adjoint FFT. Defaults to False.

    Returns
    -------
    Array
        Resulting array after the pfft operation.
    """
    return pfft_impl(x, fft_type=fft_type, adjoint=adjoint)


@pfft.defjvp
def pfft_jvp(fft_type: FftType, adjoint: bool, primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (x,) = primals
    (x_dot,) = tangents

    primals_out = pfft_impl(x, fft_type=fft_type, adjoint=adjoint)

    tangents_out = pfft_impl(x_dot, fft_type=fft_type, adjoint=adjoint)

    return primals_out, tangents_out
