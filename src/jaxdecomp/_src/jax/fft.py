from functools import partial
from typing import Tuple, Type, Union

import jax
from jax import lax
from jax._src.api import ShapeDtypeStruct
from jax._src.core import ShapedArray
from jax._src.typing import Array
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomplib import _jaxdecomp

import jaxdecomp
from jaxdecomp._src.fft_utils import COMPLEX  # yapf: disable
from jaxdecomp._src.fft_utils import FftType  # yapf: disable
from jaxdecomp._src.fft_utils import ADJOINT, FORWARD_FFTs, fft, fft2, fftn
from jaxdecomp._src.pencil_utils import get_output_specs, get_transpose_order
from jaxdecomp._src.spmd_ops import custom_spmd_rule , get_pencil_type  # yapf: disable
from jaxdecomp._src.sharded_array import ShardedArray


def _fft_slab_xy(
    operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str
) -> Array:
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
        operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True).transpose(
            [1, 2, 0]
        )
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
    else:
        operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True)
        operand = fft(operand, COMPLEX(fft_type), axis=0, adjoint=adjoint)
    return operand


def _fft_slab_yz(
    operand: Array, fft_type: FftType, adjoint: bool, y_axis_name: str
) -> Array:
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
        operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True).transpose(
            [2, 0, 1]
        )
        # FFT on YZ plane
        operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)

        if jaxdecomp.config.transpose_axis_contiguous_2:
            operand = operand.transpose([2, 0, 1])
    else:
        # transpose to (Z , Y , X / py) with specs P('z', None, 'y')
        operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True)
        # FFT on YZ plane
        operand = fft2(operand, COMPLEX(fft_type), axes=(1, 0), adjoint=adjoint)

    return operand


def _fft_pencils(
    operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str, y_axis_name: str
) -> Array:
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
        operand = lax.all_to_all(operand, y_axis_name, 2, 1, tiled=True).transpose(
            [2, 0, 1]
        )
        # FFT on the Y axis
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
        # transpose to (Y / pz, X / py, Z) with specs P('z', 'y')
        operand = lax.all_to_all(operand, x_axis_name, 2, 1, tiled=True).transpose(
            [2, 0, 1]
        )
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


def _ifft_slab_xy(
    operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str
) -> Array:
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
        operand = lax.all_to_all(operand, x_axis_name, 2, 1, tiled=True).transpose(
            [2, 0, 1]
        )
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


def _ifft_slab_yz(
    operand: Array, fft_type: FftType, adjoint: bool, y_axis_name: str
) -> Array:
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
        if jaxdecomp.config.transpose_axis_contiguous_2:
            operand = operand.transpose([1, 2, 0])
        # input is (X / py, Z , Y) with specs P('y', 'z')
        # First IFFT
        operand = fft2(operand, COMPLEX(fft_type), axes=(2, 1), adjoint=adjoint)
        # transpose to (Z , Y / Py, X) with specs P('z', 'y')
        operand = lax.all_to_all(operand, y_axis_name, 2, 0, tiled=True).transpose(
            [1, 2, 0]
        )
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


def _ifft_pencils(
    operand: Array, fft_type: FftType, adjoint: bool, x_axis_name: str, y_axis_name: str
) -> Array:
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
        operand = lax.all_to_all(operand, x_axis_name, 2, 0, tiled=True).transpose(
            [1, 2, 0]
        )
        # IFFT on the Y axis
        operand = fft(operand, COMPLEX(fft_type), axis=-1, adjoint=adjoint)
        # transpose to (Z / Pz , Y / Py  , X) with specs P('z', 'y')
        operand = lax.all_to_all(operand, y_axis_name, 2, 0, tiled=True).transpose(
            [1, 2, 0]
        )
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
    return fftn(x, fft_type, adjoint=adjoint).transpose(transpose_order)


def per_shard_impl(x: Array, fft_type: FftType, adjoint: bool, mesh: Mesh) -> Array:
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
    assert (
        len(mesh.axis_names) <= 2
    ), "Only one or two-dimensional meshes are supported."
    axis_names: Tuple[str | None, ...] = mesh.axis_names
    x_axis_name, y_axis_name = (
        axis_names if len(axis_names) == 2 else (*axis_names, None)
    )
    assert isinstance(fft_type, FftType)  # type: ignore
    pencil_type = get_pencil_type(mesh)
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
                raise ValueError(f"Unsupported pencil type {pencil_type}")
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
                raise ValueError(f"Unsupported pencil type {pencil_type}")


spmd_fft_primitive = custom_spmd_rule(
    spmd_fft, static_argnums=(1, 2), multiple_results=False
)


@spmd_fft_primitive.def_infer_sharding
def infer_sharding_from_operands(
    fft_type: FftType,
    adjoint: bool,
    mesh: Mesh,
    arg_infos: Tuple[ShapeDtypeStruct],
    result_infos: Tuple[ShapedArray],
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
    input_mesh: Mesh = input_sharding.mesh  # type: ignore
    spec = input_sharding.spec
    transposed_specs = get_output_specs(fft_type, spec, mesh=input_mesh, backend="jax")

    return NamedSharding(input_sharding.mesh, P(*transposed_specs))


@spmd_fft_primitive.def_partition
def partition(
    fft_type: FftType,
    adjoint: bool,
    mesh: Mesh,
    arg_infos: Tuple[ShapeDtypeStruct],
    result_infos: Tuple[ShapedArray],
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

    impl = partial(per_shard_impl, fft_type=fft_type, adjoint=adjoint, mesh=input_mesh)

    return mesh, impl, output_sharding, (input_sharding,)


@spmd_fft_primitive.def_transpose_rule
def transpose_rule(
    cotangent: Array, x: Array, fft_type: FftType, adjoint: bool
) -> Tuple[Array]:
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
    return (
        spmd_fft_primitive(cotangent, fft_type=ADJOINT(fft_type), adjoint=not adjoint),
    )


@partial(jax.jit, static_argnums=(1, 2))
def pfft_impl(
    x: Type[Union[Array, ShardedArray]], fft_type: FftType, adjoint: bool
) -> Array:
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
    if isinstance(x, ShardedArray):
        if x.initial_sharding is None:
            return spmd_fft(x.data, fft_type=fft_type, adjoint=adjoint)
        else:
            input_mesh: Mesh = x.initial_sharding.mesh  # type: ignore
            forward_specs = x.initial_sharding.spec  # type: ignore
            backward_specs = get_output_specs(
                FftType.FFT, forward_specs, mesh=input_mesh, backend="jax"
            )

            in_specs = forward_specs if fft_type in FORWARD_FFTs else backward_specs
            out_specs = backward_specs if fft_type in FORWARD_FFTs else forward_specs

            in_specs = P(*in_specs)
            out_specs = P(*out_specs)

            pper_shard_impl = partial(
                per_shard_impl, fft_type=fft_type, adjoint=adjoint, mesh=input_mesh
            )
            return shard_map(
                pper_shard_impl,
                mesh=input_mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_rep=False,
            )(x)

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
def pfft_jvp(
    fft_type: FftType, adjoint: bool, primals: Tuple[Array], tangents: Tuple[Array]
) -> Tuple[Array, Array]:
    (x,) = primals
    (x_dot,) = tangents

    primals_out = pfft_impl(x, fft_type=fft_type, adjoint=adjoint)

    tangents_out = pfft_impl(x_dot, fft_type=fft_type, adjoint=adjoint)

    return primals_out, tangents_out
