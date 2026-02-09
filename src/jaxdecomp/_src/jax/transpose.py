from functools import partial

import jax
from jax import ShapeDtypeStruct, lax
from jax._src.interpreters import batching
from jax._src.typing import Array, ArrayLike
from jax.core import ShapedArray
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src.error import error_during_jacfwd, error_during_jacrev
from jaxdecomp._src.pencil_utils import get_axis_names_from_mesh
from jaxdecomp._src.spmd_ops import custom_spmd_rule


def spmd_transpose(x: Array, kind: str) -> Array:
    """
    Implementation method for the transposition primitive.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').

    Returns
    -------
    Object
        Result of binding the inner primitive with input arguments.
    """
    if jaxdecomp.config.transpose_axis_contiguous:
        match kind:
            case 'x_y' | 'y_z' | 'z_x':
                transpose_order = (2, 0, 1)
            case 'y_x' | 'z_y' | 'x_z':
                transpose_order = (1, 2, 0)
            case _:
                raise ValueError('Invalid kind')
    else:
        transpose_order = (0, 1, 2)

    def _impl(x):
        return x.transpose(transpose_order)

    if x.ndim == 3:
        return _impl(x)
    if x.ndim == 4:
        return jax.vmap(_impl)(x)
    else:
        raise ValueError(f'Unsupported input shape {x.shape}')


def per_shard_impl(a: Array, kind: str, x_axis_name: str, y_axis_name: str) -> Array:
    """
    Per-shard implementation method for the transposition primitive.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
    pdims : tuple[int]
        Partition dimensions.
    global_shape : tuple[int]
        Global shape of the input array.

    Returns
    -------
    Object
        Result of binding the inner primitive with input arguments.
    """

    def _impl(a):
        if jaxdecomp.config.transpose_axis_contiguous:
            match kind:
                case 'x_y':
                    return lax.all_to_all(a, y_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
                case 'y_z':
                    return lax.all_to_all(a, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
                case 'z_y':
                    return lax.all_to_all(a, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
                case 'y_x':
                    return lax.all_to_all(a, y_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
                case 'x_z':
                    return lax.all_to_all(a, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
                case 'z_x':
                    return lax.all_to_all(a, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
                case _:
                    raise ValueError('Invalid kind')
        else:
            match kind:
                case 'x_y':
                    return lax.all_to_all(a, y_axis_name, 2, 1, tiled=True)
                case 'y_z':
                    return lax.all_to_all(a, x_axis_name, 1, 0, tiled=True)
                case 'z_y':
                    return lax.all_to_all(a, x_axis_name, 0, 1, tiled=True)
                case 'y_x':
                    return lax.all_to_all(a, y_axis_name, 1, 2, tiled=True)
                case 'x_z':
                    return lax.all_to_all(a, x_axis_name, 2, 0, tiled=True)
                case 'z_x':
                    return lax.all_to_all(a, x_axis_name, 0, 2, tiled=True)
                case _:
                    raise ValueError('Invalid kind')

    if a.ndim == 3:
        return _impl(a)
    if a.ndim == 4:
        return jax.vmap(_impl)(a)
    else:
        raise ValueError(f'Unsupported input shape {a.shape}')


spmd_transpose_primitive = custom_spmd_rule(spmd_transpose, static_argnums=(1,), multiple_results=False)


def get_output_specs(spec, kind):
    spec = spec + (None,) * (3 - len(spec))
    if jaxdecomp.config.transpose_axis_contiguous:
        transposed_specs = (spec[1], spec[0], spec[2])
    else:
        match kind:
            case 'x_y':
                transposed_specs = (spec[0], spec[2], spec[1])
            case 'y_z':
                transposed_specs = (spec[1], spec[0], spec[2])
            case 'z_y':
                transposed_specs = (spec[1], spec[0], spec[2])
            case 'y_x':
                transposed_specs = (spec[0], spec[2], spec[1])
            case _:
                raise ValueError('Invalid kind')

    return transposed_specs


@spmd_transpose_primitive.def_infer_sharding
def infer_sharding_from_operands(
    kind: str,
    mesh: Mesh,
    arg_infos: tuple[ShapeDtypeStruct],
    result_infos: tuple[ShapedArray],
) -> NamedSharding:
    """
    Method to infer sharding information from operands for custom partitioning.

    Parameters
    ----------
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
    mesh : Mesh
        Sharding mesh information.
    arg_infos : Tuple[ShapeDtypeStruct]
        Tuple of ShapeDtypeStruct for input operands.
    result_infos : Tuple[ShapedArray]
        Tuple of ShapedArray for result information.

    Returns
    -------
    NamedSharding
        Named sharding information.
    """
    del mesh, result_infos
    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    operand = arg_infos[0]

    if input_sharding is None:
        error_during_jacfwd(f'Transpose {kind}')

    if all([spec is None for spec in input_sharding.spec]):
        error_during_jacrev(f'Transpose {kind}')

    if operand.ndim == 3:
        spec = input_sharding.spec
        transposed_specs = get_output_specs(spec, kind)
    elif operand.ndim == 4:
        batch_spec = input_sharding.spec[0]
        spec = input_sharding.spec[1:]
        transposed_specs = get_output_specs(spec, kind)
        transposed_specs = (batch_spec,) + transposed_specs
    else:
        raise ValueError(f'Unsupported input shape {operand.shape}')

    return NamedSharding(input_sharding.mesh, P(*transposed_specs))


@spmd_transpose_primitive.def_sharding_rule
def transpose_sharding_rule_producer(
    kind: str,
    mesh: Mesh,
    arg_infos,
    result_infos,
) -> str:
    """
    Produces sharding rule for transpose operation for Shardy partitioner.

    Parameters
    ----------
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x', 'x_z', 'z_x').
    mesh : Mesh
        Mesh configuration for the distributed transpose.
    arg_infos : Tuple
        Information about input arguments.
    result_infos : Tuple
        Information about result.

    Returns
    -------
    str
        Einsum string describing the transpose operation.
    """
    del result_infos, mesh

    spec = ('i', 'j', 'k')  # einsum spec for shardy
    transposed_specs = get_output_specs(spec, kind)
    einsum_in = ' '.join(spec)
    einsum_out = ' '.join(transposed_specs)

    operand = arg_infos[0]
    if operand.rank == 3:
        pass
    elif operand.rank == 4:
        einsum_in = 'b ' + einsum_in
        einsum_out = 'b ' + einsum_out
    else:
        raise ValueError(f'Unsupported input shape rank {operand.rank}')

    return f'{einsum_in}->{einsum_out}'


@spmd_transpose_primitive.def_partition
def partition(
    kind: str,
    mesh: Mesh,
    arg_infos: tuple[ShapeDtypeStruct],
    result_infos: tuple[ShapedArray],
):
    """
    Method to partition the transposition operation for custom partitioning.

    Parameters
    ----------
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').
    mesh : Mesh
        Sharding mesh information.
    arg_infos : Tuple[ShapeDtypeStruct]
        Tuple of ShapeDtypeStruct for input operands.
    result_infos : Tuple[ShapedArray]
        Tuple of ShapedArray for result information.

    Returns
    -------
    Tuple
        Tuple containing mesh, implementation function, output sharding, and input sharding.
    """

    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    output_sharding: NamedSharding = result_infos.sharding  # type: ignore
    x_axis_name, y_axis_name = get_axis_names_from_mesh(mesh)

    impl = partial(per_shard_impl, kind=kind, x_axis_name=x_axis_name, y_axis_name=y_axis_name)

    return mesh, impl, output_sharding, (input_sharding,)


@spmd_transpose_primitive.def_transpose_rule
def vjp_transpose_rule(cotangent: Array, x: Array, kind: str) -> tuple[Array]:
    match kind:
        case 'x_y':
            return (spmd_transpose_primitive(cotangent, kind='y_x'),)
        case 'y_z':
            return (spmd_transpose_primitive(cotangent, kind='z_y'),)
        case 'z_y':
            return (spmd_transpose_primitive(cotangent, kind='y_z'),)
        case 'y_x':
            return (spmd_transpose_primitive(cotangent, kind='x_y'),)
        case 'x_z':
            return (spmd_transpose_primitive(cotangent, kind='z_x'),)
        case 'z_x':
            return (spmd_transpose_primitive(cotangent, kind='x_z'),)
        case _:
            raise ValueError('Invalid kind')


@spmd_transpose_primitive.def_batching_rule
def batching_rule(batched_args: tuple[Array], batched_axis: tuple[int | None, ...], kind: str) -> tuple[Array, int]:
    """
    Batching rule for the transpose operation.

    Parameters
    ----------
    batched_args : Tuple[Array]
        Batched input arrays.
    batched_axis : tuple[int | None, ...]
        Batch axis for each operand.
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').

    Returns
    -------
    tuple[Array, int]
        Resulting array and the output batch axis.
    """
    (x,) = batched_args
    (bd,) = batched_axis
    x = batching.moveaxis(x, bd, 0)
    return spmd_transpose_primitive(x, kind=kind), 0


@partial(jax.jit, static_argnums=(1,))
def transpose_impl(x: ArrayLike, kind: str) -> Array:
    """
    JIT-compiled function for performing transposition using the outer primitive.

    Parameters
    ----------
    x : ArrayLike
        Input array.
    kind : str
        Kind of transposition ('x_y', 'y_z', 'z_y', 'y_x').

    Returns
    -------
    Array
        Transposed array.
    """
    return spmd_transpose_primitive(x, kind=kind)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def transpose(x: ArrayLike, kind: str) -> Array:
    return transpose_impl(x, kind=kind)


@transpose.defjvp
def transpose_jvp(kind: str, primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
    (x,) = primals
    (t,) = tangents
    y = transpose(x, kind=kind)
    return y, transpose(t, kind=kind)


@partial(jax.custom_jvp, nondiff_argnums=(1, 2, 3))
def transpose_shard(x: ArrayLike, kind: str, mesh: Mesh, x_axis_name: str, y_axis_name: str) -> Array:
    return per_shard_impl(x, kind, x_axis_name, y_axis_name)


@transpose_shard.defjvp
def transpose_shard_jvp(
    kind: str,
    mesh: Mesh,
    x_axis_name: str,
    y_axis_name: str,
    primals: tuple[Array],
    tangents: tuple[Array],
) -> tuple[Array, Array]:
    (x,) = primals
    (t,) = tangents
    y = transpose_shard(x, kind=kind, mesh=mesh, x_axis_name=x_axis_name, y_axis_name=y_axis_name)
    return y, transpose_shard(t, kind=kind, mesh=mesh, x_axis_name=x_axis_name, y_axis_name=y_axis_name)


# Custom transposition functions


def transposeXtoY(x: ArrayLike) -> Array:
    """
    Custom JAX transposition function for X to Y.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
    return transpose(x, 'x_y')


def transposeYtoZ(x: ArrayLike) -> Array:
    """
    Custom JAX transposition function for Y to Z.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
    return transpose(x, 'y_z')


def transposeZtoY(x: ArrayLike) -> Array:
    """
    Custom JAX transposition function for Z to Y.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
    return transpose(x, 'z_y')


def transposeYtoX(x: ArrayLike) -> Array:
    """
    Custom JAX transposition function for Y to X.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
    return transpose(x, 'y_x')


def transposeXtoZ(x: ArrayLike) -> Array:
    """
      Custom JAX transposition function for X to Z.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
    return transpose(x, 'x_z')


def transposeZtoX(x: ArrayLike) -> Array:
    """
    Custom JAX transposition function for X to Z.

    Parameters
    ----------
    x : ArrayLike
        Input array.

    Returns
    -------
    Array
        Transposed array.
    """
    return transpose(x, 'z_x')
