from functools import partial
from typing import Tuple

import jax
from jax import ShapeDtypeStruct, lax
from jax._src.typing import Array, ArrayLike
from jax.core import ShapedArray
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src.sharded_array import ShardedArray
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
            case "x_y" | "y_z" | "z_x":
                transpose_order = (2, 0, 1)
            case "y_x" | "z_y" | "x_z":
                transpose_order = (1, 2, 0)
            case _:
                raise ValueError("Invalid kind")
    else:
        transpose_order = (0, 1, 2)

    return x.transpose(transpose_order)


def per_shard_impl(a: Array, kind: str, mesh: Mesh):
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

    assert (
        len(mesh.axis_names) <= 2
    ), "Only one or two-dimensional meshes are supported."
    axis_names: Tuple[str | None, ...] = mesh.axis_names
    x_axis_name, y_axis_name = (
        axis_names if len(axis_names) == 2 else (*axis_names, None)
    )
    if jaxdecomp.config.transpose_axis_contiguous:
        match kind:
            case "x_y":
                return lax.all_to_all(a, y_axis_name, 2, 1, tiled=True).transpose(
                    [2, 0, 1]
                )
            case "y_z":
                return lax.all_to_all(a, x_axis_name, 2, 1, tiled=True).transpose(
                    [2, 0, 1]
                )
            case "z_y":
                return lax.all_to_all(a, x_axis_name, 2, 0, tiled=True).transpose(
                    [1, 2, 0]
                )
            case "y_x":
                return lax.all_to_all(a, y_axis_name, 2, 0, tiled=True).transpose(
                    [1, 2, 0]
                )
            case "x_z":
                return lax.all_to_all(a, x_axis_name, 2, 0, tiled=True).transpose(
                    [1, 2, 0]
                )
            case "z_x":
                return lax.all_to_all(a, x_axis_name, 2, 1, tiled=True).transpose(
                    [2, 0, 1]
                )
            case _:
                raise ValueError("Invalid kind")
    else:
        match kind:
            case "x_y":
                return lax.all_to_all(a, y_axis_name, 2, 1, tiled=True)
            case "y_z":
                return lax.all_to_all(a, x_axis_name, 1, 0, tiled=True)
            case "z_y":
                return lax.all_to_all(a, x_axis_name, 0, 1, tiled=True)
            case "y_x":
                return lax.all_to_all(a, y_axis_name, 1, 2, tiled=True)
            case "x_z":
                return lax.all_to_all(a, x_axis_name, 2, 0, tiled=True)
            case "z_x":
                return lax.all_to_all(a, x_axis_name, 0, 2, tiled=True)
            case _:
                raise ValueError("Invalid kind")


spmd_transpose_primitive = custom_spmd_rule(
    spmd_transpose, static_argnums=(1,), multiple_results=False
)


def get_output_specs(spec, kind):
    spec = spec + (None,) * (3 - len(spec))
    if jaxdecomp.config.transpose_axis_contiguous:
        transposed_specs = (spec[1], spec[0], None)
    else:
        match kind:
            case "x_y":
                transposed_specs = (spec[0], None, spec[1])
            case "y_z":
                transposed_specs = (None, spec[0], spec[2])
            case "z_y":
                transposed_specs = (spec[1], None, spec[2])
            case "y_x":
                transposed_specs = (spec[0], spec[2], None)
            case _:
                raise ValueError("Invalid kind")

    return transposed_specs


def get_input_specs_from_origin(spec, kind):
    spec = spec + (None,) * (3 - len(spec))
    if jaxdecomp.config.transpose_axis_contiguous:
        match kind:
            case "x_y" | "z_y" | "x_z":
                transposed_specs = (spec[0], spec[1], None)
            case "y_z" | "y_x" | "z_x":
                transposed_specs = (spec[1], spec[0], None)
            case _:
                raise ValueError("Invalid kind")
    else:
        match kind:
            case "x_y":
                transposed_specs = (spec[0], spec[1], None)
            case "y_z":
                transposed_specs = (spec[0], None, spec[1])
            case "z_y":
                transposed_specs = (None, spec[0], spec[1])
            case "y_x":
                transposed_specs = (spec[0], None, spec[1])
            case "x_z":
                transposed_specs = (spec[0], spec[1], None)
            case "z_x":
                transposed_specs = (None, spec[1], spec[0])

            case _:
                raise ValueError("Invalid kind")

    return transposed_specs


@spmd_transpose_primitive.def_infer_sharding
def infer_sharding_from_operands(
    kind: str,
    mesh: Mesh,
    arg_infos: Tuple[ShapeDtypeStruct],
    result_infos: Tuple[ShapedArray],
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
    transposed_specs = get_output_specs(input_sharding.spec, kind)

    return NamedSharding(input_sharding.mesh, P(*transposed_specs))


@spmd_transpose_primitive.def_partition
def partition(
    kind: str,
    mesh: Mesh,
    arg_infos: Tuple[ShapeDtypeStruct],
    result_infos: Tuple[ShapedArray],
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
    input_mesh: Mesh = arg_infos[0].sharding.mesh  # type: ignore

    impl = partial(per_shard_impl, kind=kind, mesh=input_mesh)

    return mesh, impl, output_sharding, (input_sharding,)


@spmd_transpose_primitive.def_transpose_rule
def vjp_transpose_rule(cotangent: Array, x: Array, kind: str) -> Tuple[Array]:
    match kind:
        case "x_y":
            return (spmd_transpose_primitive(cotangent, kind="y_x"),)
        case "y_z":
            return (spmd_transpose_primitive(cotangent, kind="z_y"),)
        case "z_y":
            return (spmd_transpose_primitive(cotangent, kind="y_z"),)
        case "y_x":
            return (spmd_transpose_primitive(cotangent, kind="x_y"),)
        case "x_z":
            return (spmd_transpose_primitive(cotangent, kind="z_x"),)
        case "z_x":
            return (spmd_transpose_primitive(cotangent, kind="x_z"),)
        case _:
            raise ValueError("Invalid kind")


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
    if isinstance(x, ShardedArray):
        if x.initial_sharding is None:
            return spmd_transpose(x, kind=kind)
        else:
            input_mesh: Mesh = x.initial_sharding.mesh  # type: ignore
            forward_specs = x.initial_sharding.spec  # type: ignore

            in_specs = get_input_specs_from_origin(forward_specs, kind)
            out_specs = get_output_specs(in_specs, kind)
            in_specs = P(*in_specs)
            out_specs = P(*out_specs)
            pper_shard_impl = partial(per_shard_impl, kind=kind, mesh=input_mesh)
            return shard_map(
                pper_shard_impl, mesh=input_mesh, in_specs=in_specs, out_specs=out_specs
            )(x)
    return spmd_transpose_primitive(x, kind=kind)


@partial(jax.custom_jvp, nondiff_argnums=(1,))
def transpose(x: ArrayLike, kind: str) -> Array:
    return transpose_impl(x, kind=kind)


@transpose.defjvp
def transpose_jvp(
    kind: str, primals: Tuple[Array], tangents: Tuple[Array]
) -> Tuple[Array, Array]:
    (x,) = primals
    (t,) = tangents
    y = transpose(x, kind=kind)
    return y, transpose(t, kind=kind)


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
    return transpose(x, "x_y")


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
    return transpose(x, "y_z")


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
    return transpose(x, "z_y")


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
    return transpose(x, "y_x")


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
    return transpose(x, "x_z")


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
    return transpose(x, "z_x")
