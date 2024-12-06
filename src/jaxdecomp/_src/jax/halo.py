from functools import partial
from typing import Tuple

import jax
from jax import ShapeDtypeStruct, lax
from jax.core import ShapedArray
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxdecomplib import _jaxdecomp
from jaxtyping import Array

from jaxdecomp._src.spmd_ops import (CustomParPrimitive, get_pencil_type,
                                     register_primitive)
from jaxdecomp.typing import HaloExtentType, Periodicity


def _halo_slab_xy(operand: Array, halo_extents: HaloExtentType,
                  halo_periods: Periodicity, x_axis_name: str) -> Array:
  """
    Halo exchange for slab decomposition in the XY plane.

    Parameters
    ----------
    operand : Array
        Input array.
    halo_extents : Tuple[int, int]
        Extents of the halo in X and Y directions.
    halo_periods : Tuple[bool, bool]
        Periodicity in X and Y directions.
    x_axis_name : str
        The axis name for the X axis.

    Returns
    -------
    Array
        The array with halo exchange applied.
    """
  z_size = lax.psum(1, x_axis_name)
  halo_extent = halo_extents[0]
  periodic = halo_periods[0]

  upper_halo = operand[halo_extent:halo_extent + halo_extent]
  lower_halo = operand[-(halo_extent + halo_extent):-halo_extent]

  permutations = slice(None, None) if periodic else slice(None, -1)
  forward_indexing_z = [(j, (j + 1) % z_size) for j in range(z_size)
                       ][permutations]
  reverse_indexing_z = [((j + 1) % z_size, j) for j in range(z_size)
                       ][permutations]

  exchanged_upper_halo = lax.ppermute(
      upper_halo, axis_name=x_axis_name, perm=reverse_indexing_z)
  exchanged_lower_halo = lax.ppermute(
      lower_halo, axis_name=x_axis_name, perm=forward_indexing_z)

  operand = operand.at[:halo_extent].set(exchanged_lower_halo)
  operand = operand.at[-halo_extent:].set(exchanged_upper_halo)

  return operand


def _halo_slab_yz(operand: Array, halo_extents: HaloExtentType,
                  halo_periods: Periodicity, y_axis_name: str) -> Array:
  """
    Halo exchange for slab decomposition in the YZ plane.

    Parameters
    ----------
    operand : Array
        Input array.
    halo_extents : Tuple[int, int]
        Extents of the halo in X and Y directions.
    halo_periods : Tuple[bool, bool]
        Periodicity in X and Y directions.
    y_axis_name : str
        The axis name for the Y axis.

    Returns
    -------
    Array
        The array with halo exchange applied.
    """
  y_size = lax.psum(1, y_axis_name)
  halo_extent = halo_extents[1]
  periodic = halo_periods[1]

  right_halo = operand[:, halo_extent:halo_extent + halo_extent]
  left_halo = operand[:, -(halo_extent + halo_extent):-halo_extent]

  permutations = slice(None, None) if periodic else slice(None, -1)
  reverse_indexing_y = [((j + 1) % y_size, j) for j in range(y_size)
                       ][permutations]
  forward_indexing_y = [(j, (j + 1) % y_size) for j in range(y_size)
                       ][permutations]

  exchanged_right_halo = lax.ppermute(
      right_halo, axis_name=y_axis_name, perm=reverse_indexing_y)
  exchanged_left_halo = lax.ppermute(
      left_halo, axis_name=y_axis_name, perm=forward_indexing_y)

  operand = operand.at[:, :halo_extent].set(exchanged_left_halo)
  operand = operand.at[:, -halo_extent:].set(exchanged_right_halo)

  return operand


def _halo_pencils(operand: Array, halo_extents: HaloExtentType,
                  halo_periods: Periodicity, x_axis_name: str,
                  y_axis_name: str) -> Array:
  """
    Halo exchange for pencil decomposition.

    Parameters
    ----------
    operand : Array
        Input array.
    halo_extents : Tuple[int, int]
        Extents of the halo in X and Y directions.
    halo_periods : Tuple[bool, bool]
        Periodicity in X and Y directions.
    x_axis_name : str
        The axis name for the X axis.
    y_axis_name : str
        The axis name for the Y axis.

    Returns
    -------
    Array
        The array with halo exchange applied.
    """
  y_size = lax.psum(1, y_axis_name)
  z_size = lax.psum(1, x_axis_name)

  halo_x, halo_y = halo_extents
  periodic_x, periodic_y = halo_periods

  upper_halo = operand[halo_x:halo_x + halo_x]
  lower_halo = operand[-(halo_x + halo_x):-halo_x]

  right_halo = operand[:, halo_y:halo_y + halo_y]
  left_halo = operand[:, -(halo_y + halo_y):-halo_y]

  upper_right_corner = operand[halo_x:halo_x + halo_x, halo_y:halo_y + halo_y]
  upper_left_corner = operand[halo_x:halo_x + halo_x,
                              -(halo_y + halo_y):-halo_y]
  lower_right_corner = operand[-(halo_x + halo_x):-halo_x,
                               halo_y:halo_y + halo_y]
  lower_left_corner = operand[-(halo_x + halo_x):-halo_x,
                              -(halo_y + halo_y):-halo_y]

  permutations_x = slice(None, None) if periodic_x else slice(None, -1)
  permutations_y = slice(None, None) if periodic_y else slice(None, -1)
  reverse_indexing_z = [((j + 1) % z_size, j) for j in range(z_size)
                       ][permutations_x]
  forward_indexing_z = [(j, (j + 1) % z_size) for j in range(z_size)
                       ][permutations_x]
  reverse_indexing_y = [((j + 1) % y_size, j) for j in range(y_size)
                       ][permutations_y]
  forward_indexing_y = [(j, (j + 1) % y_size) for j in range(y_size)
                       ][permutations_y]

  exchanged_upper_halo = lax.ppermute(
      upper_halo, axis_name=x_axis_name, perm=reverse_indexing_z)
  exchanged_lower_halo = lax.ppermute(
      lower_halo, axis_name=x_axis_name, perm=forward_indexing_z)
  exchanged_right_halo = lax.ppermute(
      right_halo, axis_name=y_axis_name, perm=reverse_indexing_y)
  exchanged_left_halo = lax.ppermute(
      left_halo, axis_name=y_axis_name, perm=forward_indexing_y)

  exchanged_upper_right_corner = lax.ppermute(
      lax.ppermute(
          upper_right_corner, axis_name=x_axis_name, perm=reverse_indexing_z),
      axis_name=y_axis_name,
      perm=reverse_indexing_y)
  exchanged_upper_left_corner = lax.ppermute(
      lax.ppermute(
          upper_left_corner, axis_name=x_axis_name, perm=reverse_indexing_z),
      axis_name=y_axis_name,
      perm=forward_indexing_y)
  exchanged_lower_right_corner = lax.ppermute(
      lax.ppermute(
          lower_right_corner, axis_name=x_axis_name, perm=forward_indexing_z),
      axis_name=y_axis_name,
      perm=reverse_indexing_y)
  exchanged_lower_left_corner = lax.ppermute(
      lax.ppermute(
          lower_left_corner, axis_name=x_axis_name, perm=forward_indexing_z),
      axis_name=y_axis_name,
      perm=forward_indexing_y)

  operand = operand.at[:halo_x].set(exchanged_lower_halo)
  operand = operand.at[-halo_x:].set(exchanged_upper_halo)
  operand = operand.at[:, :halo_y].set(exchanged_left_halo)
  operand = operand.at[:, -halo_y:].set(exchanged_right_halo)

  operand = operand.at[:halo_x, :halo_y].set(exchanged_lower_left_corner)
  operand = operand.at[:halo_x, -halo_y:].set(exchanged_lower_right_corner)
  operand = operand.at[-halo_x:, :halo_y].set(exchanged_upper_left_corner)
  operand = operand.at[-halo_x:, -halo_y:].set(exchanged_upper_right_corner)

  return operand


class JAXHaloPrimitive(CustomParPrimitive):
  """
    Custom JAX primitive for halo exchange operation.

    Attributes
    ----------
    name : str
        Name of the primitive.
    multiple_results : bool
        Boolean indicating if the primitive returns multiple results (False).
    impl_static_args : Tuple[int, int]
        Static arguments for the implementation (halo extents and periods).
    outer_primitive : object
        The outer core primitive.
    """
  name = 'jax_halo'
  multiple_results = False
  impl_static_args: Tuple[int, int] = (1, 2)
  outer_primitive = None

  @staticmethod
  def impl(x: Array, halo_extents: HaloExtentType,
           halo_periods: Periodicity) -> Array:
    del halo_extents, halo_periods
    return x

  @staticmethod
  def per_shard_impl(x: Array, halo_extents: HaloExtentType,
                     halo_periods: Periodicity, mesh: Mesh) -> Array:
    """
        Per-shard implementation for the halo exchange.

        Parameters
        ----------
        x : Array
            Input array.
        halo_extents : HaloExtentType
            Extents of the halo in X and Y directions.
        halo_periods : Periodicity
            Periodicity in X and Y directions.
        mesh : Mesh
            Mesh containing axis information for sharding.

        Returns
        -------
        Array
            Array after the halo exchange operation.
        """
    x_axis_name, y_axis_name = mesh.axis_names
    assert (x_axis_name is not None) or (y_axis_name is not None)
    pencil_type = get_pencil_type(mesh)

    match pencil_type:
      case _jaxdecomp.SLAB_XY:
        return _halo_slab_xy(x, halo_extents, halo_periods, x_axis_name)
      case _jaxdecomp.SLAB_YZ:
        return _halo_slab_yz(x, halo_extents, halo_periods, y_axis_name)
      case _jaxdecomp.PENCILS:
        return _halo_pencils(x, halo_extents, halo_periods, x_axis_name,
                             y_axis_name)
      case _:
        raise ValueError(f"Unsupported pencil type {pencil_type}")

  @staticmethod
  def infer_sharding_from_operands(
      halo_extents: HaloExtentType, halo_periods: Periodicity, mesh: Mesh,
      arg_infos: Tuple[ShapeDtypeStruct],
      result_infos: Tuple[ShapedArray]) -> NamedSharding:
    """
        Infer sharding from the operands.

        Parameters
        ----------
        halo_extents : HaloExtentType
            Extents of the halo in X and Y directions.
        halo_periods : Periodicity
            Periodicity in X and Y directions.
        mesh : Mesh
            Mesh object for sharding.
        arg_infos : Tuple[ShapeDtypeStruct]
            Shape and dtype information for input operands.
        result_infos : Tuple[ShapedArray]
            Shape and dtype information for result operands.

        Returns
        -------
        NamedSharding
            Named sharding information.
        """
    del halo_periods, result_infos, halo_extents, mesh
    halo_exchange_sharding: NamedSharding = arg_infos[
        0].sharding  # type: ignore
    return NamedSharding(halo_exchange_sharding.mesh,
                         P(*halo_exchange_sharding.spec))

  @staticmethod
  def partition(
      halo_extents: HaloExtentType, halo_periods: Periodicity, mesh: Mesh,
      arg_infos: Tuple[ShapeDtypeStruct], result_infos: Tuple[ShapedArray]
  ) -> Tuple[Mesh, partial, NamedSharding, Tuple[NamedSharding]]:
    """
        Partition the halo exchange operation for custom partitioning.

        Parameters
        ----------
        halo_extents : HaloExtentType
            Extents of the halo in X and Y directions.
        halo_periods : Periodicity
            Periodicity in X and Y directions.
        mesh : Mesh
            Mesh object for sharding.
        arg_infos : Tuple[ShapeDtypeStruct]
            Shape and dtype information for input operands.
        result_infos : Tuple[ShapedArray]
            Shape and dtype information for result operands.

        Returns
        -------
        Tuple
            Tuple containing mesh, implementation function, output sharding, and input sharding.
        """
    input_sharding: NamedSharding = arg_infos[0].sharding  # type: ignore
    output_sharding: NamedSharding = result_infos.sharding  # type: ignore
    input_mesh: Mesh = arg_infos[0].sharding.mesh  # type: ignore

    impl = partial(
        JAXHaloPrimitive.per_shard_impl,
        halo_extents=halo_extents,
        halo_periods=halo_periods,
        mesh=input_mesh)

    return mesh, impl, input_sharding, (output_sharding,)


register_primitive(JAXHaloPrimitive)


@partial(jax.jit, static_argnums=(1, 2))
def halo_p_lower(x: Array, halo_extents: HaloExtentType,
                 halo_periods: Periodicity) -> Array:
  """
    Lowering function for the halo exchange operation.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : HaloExtentType
        Extents of the halo in X and Y directions.
    halo_periods : Periodicity
        Periodicity in X and Y directions.

    Returns
    -------
    Array
        The lowered array after the halo exchange.
    """
  return JAXHaloPrimitive.outer_lowering(
      x, halo_extents=halo_extents, halo_periods=halo_periods)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def halo_exchange(x: Array, halo_extents: HaloExtentType,
                  halo_periods: Periodicity) -> Array:
  """
    Custom VJP definition for halo exchange.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : HaloExtentType
        Extents of the halo in X and Y directions.
    halo_periods : Periodicity
        Periodicity in X and Y directions.

    Returns
    -------
    Array
        Output array after the halo exchange operation.
    """
  output, _ = _halo_fwd_rule(x, halo_extents, halo_periods)
  return output


def _halo_fwd_rule(x: Array, halo_extents: HaloExtentType,
                   halo_periods: Periodicity) -> Tuple[Array, None]:
  """
    Forward rule for halo exchange.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : HaloExtentType
        Extents of the halo in X and Y directions.
    halo_periods : Periodicity
        Periodicity in X and Y directions.

    Returns
    -------
    Tuple[Array, None]
        Forward result and None for residuals.
    """
  return halo_p_lower(x, halo_extents, halo_periods), None


def _halo_bwd_rule(halo_extents: HaloExtentType, halo_periods: Periodicity, _,
                   g: Array) -> Tuple[Array]:
  """
    Backward rule for halo exchange.

    Parameters
    ----------
    halo_extents : HaloExtentType
        Extents of the halo in X and Y directions.
    halo_periods : Periodicity
        Periodicity in X and Y directions.
    g : Array
        Gradient array.

    Returns
    -------
    Tuple[Array]
        Gradient array after applying the halo exchange.
    """
  return halo_p_lower(g, halo_extents, halo_periods),


# Define VJP for custom halo_exchange operation
halo_exchange.defvjp(_halo_fwd_rule, _halo_bwd_rule)
