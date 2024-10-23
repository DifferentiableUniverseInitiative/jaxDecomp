from functools import partial
from typing import Tuple

import jax
from jax import ShapeDtypeStruct, lax
from jax._src.typing import Array, ArrayLike
from jax.core import ShapedArray
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src.spmd_ops import CustomParPrimitive, register_primitive


class JAXTransposePrimitive(CustomParPrimitive):
  """
    jax primitive for transposing arrays with different partitioning strategies.

    attributes
    ----------
    name : str
        Name of the primitive ("transpose").
    multiple_results : bool
        Boolean indicating if the primitive returns multiple results (False).
    impl_static_args : tuple
        Static arguments for the implementation (tuple containing (1,)).
    inner_primitive : object
        Inner core.Primitive object for the primitive.
    outer_primitive : object
        Outer core.Primitive object for the primitive.
    """

  name: str = "jax_transpose"
  multiple_results: bool = False
  impl_static_args: Tuple[int] = (1,)
  outer_primitive: object = None

  @staticmethod
  def impl(x: Array, kind: str):
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
          raise ValueError("Invalid kind")
    else:
      transpose_order = (0, 1, 2)

    return x.transpose(transpose_order)

  @staticmethod
  def per_shard_impl(a: Array, kind: str, x_axis_name: str, y_axis_name: str):
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

    assert (x_axis_name is not None) or (y_axis_name is not None)
    if jaxdecomp.config.transpose_axis_contiguous:
      match kind:
        case 'x_y':
          return lax.all_to_all(
              a, y_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
        case 'y_z':
          return lax.all_to_all(
              a, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
        case 'z_y':
          return lax.all_to_all(
              a, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
        case 'y_x':
          return lax.all_to_all(
              a, y_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
        case 'x_z':
          return lax.all_to_all(
              a, x_axis_name, 2, 0, tiled=True).transpose([1, 2, 0])
        case 'z_x':
          return lax.all_to_all(
              a, x_axis_name, 2, 1, tiled=True).transpose([2, 0, 1])
        case _:
          raise ValueError("Invalid kind")
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
          raise ValueError("Invalid kind")

  @staticmethod
  def infer_sharding_from_operands(
      kind: str, mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]) -> NamedSharding:
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
    del mesh
    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    if jaxdecomp.config.transpose_axis_contiguous:
      transposed_pdims = (input_sharding.spec[1], input_sharding.spec[0], None)
    else:
      match kind:
        case 'x_y':
          transposed_pdims = (input_sharding.spec[0], None,
                              input_sharding.spec[1])
        case 'y_z':
          transposed_pdims = (None, input_sharding.spec[0],
                              input_sharding.spec[2])
        case 'z_y':
          transposed_pdims = (input_sharding.spec[1], None,
                              input_sharding.spec[2])
        case 'y_x':
          transposed_pdims = (input_sharding.spec[0], input_sharding.spec[2],
                              None)
        case _:
          raise ValueError("Invalid kind")

    return NamedSharding(input_sharding.mesh, P(*transposed_pdims))

  @staticmethod
  def partition(kind: str, mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):
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

    input_sharding = arg_infos[0].sharding
    input_mesh = input_sharding.mesh
    output_sharding = NamedSharding(input_mesh, P(*result_infos.sharding.spec))

    x_axis_name, y_axis_name = input_mesh.axis_names
    impl = partial(
        JAXTransposePrimitive.per_shard_impl,
        kind=kind,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name)

    return mesh, impl, output_sharding, (input_sharding,)


register_primitive(JAXTransposePrimitive)


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
  return JAXTransposePrimitive.outer_lowering(x, kind)


@partial(jax.custom_vjp, nondiff_argnums=(1,))
def transpose(x: ArrayLike, kind: str) -> Array:
  out, _ = transpose_fwd_rule(x, kind)
  return out


def transpose_fwd_rule(x: ArrayLike, kind: str):
  return transpose_impl(x, kind), None


def transpose_bwd_rule(kind: str, _, g):
  match kind:
    case 'x_y':
      return transpose_impl(g, 'y_x'),
    case 'y_z':
      return transpose_impl(g, 'z_y'),
    case 'z_y':
      return transpose_impl(g, 'y_z'),
    case 'y_x':
      return transpose_impl(g, 'x_y'),
    case 'x_z':
      return transpose_impl(g, 'z_x'),
    case 'z_x':
      return transpose_impl(g, 'x_z'),
    case _:
      raise ValueError("Invalid kind")


transpose.defvjp(transpose_fwd_rule, transpose_bwd_rule)

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
