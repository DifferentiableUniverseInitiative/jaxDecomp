from functools import partial
from typing import Tuple

import jax
from jax import ShapeDtypeStruct, lax
from jax.core import ShapedArray
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxtyping import Array

from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.spmd_ops import (CustomParPrimitive, get_pencil_type,
                                     register_primitive)

HaloExtentType = Tuple[int, int]
Periodicity = Tuple[bool, bool]


def _halo_slab_xy(operand, halo_extents: HaloExtentType,
                  halo_periods: Periodicity, x_axis_name):
  # pdims are (Py=1,Pz=N)
  # input is (Z / Pz , Y , X) with specs P(x_axis_name, None)
  z_size = lax.psum(1, x_axis_name)

  halo_extent = halo_extents[0]
  periodic = halo_periods[0]

  # Extract the halo halo
  upper_halo = operand[halo_extent:halo_extent + halo_extent]
  lower_halo = operand[-(halo_extent + halo_extent):-halo_extent]

  permutations = slice(None, None) if periodic else slice(None, -1)
  reverse_indexing_z = [(j, (j + 1) % z_size) for j in range(z_size)
                       ][permutations]
  forward_indexing_z = [((j + 1) % z_size, j) for j in range(z_size)
                       ][permutations]

  # circular shift to the next rank
  exchanged_upper_halo = lax.ppermute(
      upper_halo, axis_name=x_axis_name, perm=reverse_indexing_z)

  # circular shift to the previous rank
  exchanged_lower_halo = lax.ppermute(
      lower_halo, axis_name=x_axis_name, perm=forward_indexing_z)

  # Upper halo gets the lower halo of the previous rank
  operand = operand.at[:halo_extent].set(exchanged_lower_halo)
  # Lower halo gets the upper halo of the next rank
  operand = operand.at[-halo_extent:].set(exchanged_upper_halo)

  return operand


def _halo_slab_yz(operand, halo_extents: HaloExtentType,
                  halo_periods: Periodicity, y_axis_name):
  # pdims are (Py=N,Pz=1)
  # input is (Z , Y /Py, X) with specs P(None), y_axis_name)
  y_size = lax.psum(1, y_axis_name)

  halo_extent = halo_extents[1]
  periodic = halo_periods[1]
  # Extract the halo halo
  right_halo = operand[:, halo_extent:halo_extent + halo_extent]
  left_halo = operand[:, -(halo_extent + halo_extent):-halo_extent]

  permutations = slice(None, None) if periodic else slice(None, -1)
  reverse_indexing_y = [((j + 1) % y_size, j) for j in range(y_size)
                       ][permutations]
  forward_indexing_y = [(j, (j + 1) % y_size) for j in range(y_size)
                       ][permutations]

  # circular shift to the next rank
  exchanged_right_halo = lax.ppermute(
      right_halo, axis_name=y_axis_name, perm=reverse_indexing_y)

  # circular shift to the previous rank
  exchanged_left_halo = lax.ppermute(
      left_halo, axis_name=y_axis_name, perm=forward_indexing_y)

  # Right halo gets the left halo of the next rank
  operand = operand.at[:, :halo_extent].set(exchanged_left_halo)
  # Left halo gets the right halo of the previous rank
  operand = operand.at[:, -halo_extent:].set(exchanged_right_halo)

  return operand


def _halo_pencils(operand, halo_extents: HaloExtentType,
                  halo_periods: Periodicity, x_axis_name, y_axis_name):
  # pdims are (Py=N,Pz=N)
  # input is (Z/Pz , Y/Py , X) with specs P(None), y_axis_name, x_axis_name)
  y_size = lax.psum(1, y_axis_name)
  z_size = lax.psum(1, x_axis_name)

  halo_x = halo_extents[0]
  periodic_x = halo_periods[0]
  halo_y = halo_extents[1]
  periodic_y = halo_periods[1]

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
  reverse_indexing_z = [(j, (j + 1) % z_size) for j in range(z_size)
                       ][permutations_x]
  forward_indexing_z = [((j + 1) % z_size, j) for j in range(z_size)
                       ][permutations_x]
  reverse_indexing_y = [((j + 1) % y_size, j) for j in range(y_size)
                       ][permutations_y]
  forward_indexing_y = [(j, (j + 1) % y_size) for j in range(y_size)
                       ][permutations_y]

  # circular shift to the next rank
  exchanged_upper_halo = lax.ppermute(
      upper_halo, axis_name=x_axis_name, perm=reverse_indexing_z)

  # circular shift to the previous rank
  exchanged_lower_halo = lax.ppermute(
      lower_halo, axis_name=x_axis_name, perm=forward_indexing_z)

  # circular shift to the next rank
  exchanged_right_halo = lax.ppermute(
      right_halo, axis_name=y_axis_name, perm=reverse_indexing_y)

  # circular shift to the previous rank
  exchanged_left_halo = lax.ppermute(
      left_halo, axis_name=y_axis_name, perm=forward_indexing_y)

  # For corners we need to do two circular shifts

  # Handle upper right corners
  # circular shift to the next rank
  exchanged_upper_right_corner = lax.ppermute(
      upper_right_corner, axis_name=x_axis_name, perm=reverse_indexing_z)
  # second circular shift to the next rank
  exchanged_upper_right_corner = lax.ppermute(
      exchanged_upper_right_corner,
      axis_name=y_axis_name,
      perm=reverse_indexing_y)

  # Handle upper left corners
  # circular shift to the next rank
  exchanged_upper_left_corner = lax.ppermute(
      upper_left_corner, axis_name=x_axis_name, perm=reverse_indexing_z)
  # second circular shift to the previous rank
  exchanged_upper_left_corner = lax.ppermute(
      exchanged_upper_left_corner,
      axis_name=y_axis_name,
      perm=forward_indexing_y)

  # Handle lower right corners
  # circular shift to the previous rank
  exchanged_lower_right_corner = lax.ppermute(
      lower_right_corner, axis_name=x_axis_name, perm=forward_indexing_z)
  # second circular shift to the next rank
  exchanged_lower_right_corner = lax.ppermute(
      exchanged_lower_right_corner,
      axis_name=y_axis_name,
      perm=reverse_indexing_y)

  # Handle lower left corners
  # circular shift to the previous rank
  exchanged_lower_left_corner = lax.ppermute(
      lower_left_corner, axis_name=x_axis_name, perm=forward_indexing_z)
  # second circular shift to the previous rank
  exchanged_lower_left_corner = lax.ppermute(
      exchanged_lower_left_corner,
      axis_name=y_axis_name,
      perm=forward_indexing_y)

  # Upper halo gets the lower halo of the previous rank
  operand = operand.at[:halo_x].set(exchanged_lower_halo)
  # Lower halo gets the upper halo of the next rank
  operand = operand.at[-halo_x:].set(exchanged_upper_halo)
  # Right halo gets the left halo of the next rank
  operand = operand.at[:, :halo_y].set(exchanged_left_halo)
  # Left halo gets the right halo of the previous rank
  operand = operand.at[:, -halo_y:].set(exchanged_right_halo)
  # Handle corners
  # Upper right corner gets the lower left corner of the next rank
  operand = operand.at[:halo_x, :halo_y].set(exchanged_lower_left_corner)
  # Upper left corner gets the lower right corner of the next rank
  operand = operand.at[:halo_x, -halo_y:].set(exchanged_lower_right_corner)
  # Lower right corner gets the upper left corner of the next rank
  operand = operand.at[-halo_x:, :halo_y].set(exchanged_upper_left_corner)
  # Lower left corner gets the upper right corner of the next rank
  operand = operand.at[-halo_x:, -halo_y:].set(exchanged_upper_right_corner)

  return operand


class JAXHaloPrimitive(CustomParPrimitive):
  name = 'jax_halo'
  multiple_results = False
  impl_static_args: Tuple[int, int] = (1, 2)
  outer_primitive = None

  @staticmethod
  def impl(x, halo_extents: HaloExtentType, halo_periods: Periodicity):
    del halo_extents, halo_periods
    return x

  @staticmethod
  def per_shard_impl(x: Array, halo_extents: HaloExtentType,
                     halo_periods: Periodicity, mesh: Mesh) -> Array:

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
  def infer_sharding_from_operands(halo_extents: HaloExtentType,
                                   halo_periods: Periodicity, mesh: Mesh,
                                   arg_infos: Tuple[ShapeDtypeStruct],
                                   result_infos: Tuple[ShapedArray]):

    del halo_periods, result_infos, halo_extents, mesh
    halo_exchange_sharding = arg_infos[0].sharding
    return NamedSharding(halo_exchange_sharding.mesh,
                         P(*halo_exchange_sharding.spec))

  @staticmethod
  def partition(halo_extents: HaloExtentType, halo_periods: Periodicity,
                mesh: Mesh, arg_infos: Tuple[ShapeDtypeStruct],
                result_infos: Tuple[ShapedArray]):

    halo_exchange_sharding = NamedSharding(mesh, P(*arg_infos[0].sharding.spec))
    input_mesh: Mesh = halo_exchange_sharding.mesh

    impl = partial(
        JAXHaloPrimitive.per_shard_impl,
        halo_extents=halo_extents,
        halo_periods=halo_periods,
        mesh=input_mesh)

    return mesh, impl, halo_exchange_sharding, (halo_exchange_sharding,)


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
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Primitive
        Inner primitive bound with input parameters.
    """
  return JAXHaloPrimitive.outer_lowering(
      x, halo_extents=halo_extents, halo_periods=halo_periods)


@partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def halo_exchange(x: Array, halo_extents: HaloExtentType,
                  halo_periods: Periodicity) -> Array:
  """
    Halo exchange operation with custom VJP.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.

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
    Forward rule for the halo exchange operation.

    Parameters
    ----------
    x : Array
        Input array.
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.

    Returns
    -------
    Tuple[Array, None]
        Output array after the halo exchange operation and None for no residuals.
    """
  return halo_p_lower(x, halo_extents, halo_periods), None


def _halo_bwd_rule(halo_extents: HaloExtentType, halo_periods: Periodicity, _,
                   g: Array) -> Tuple[Array]:
  """
    Backward rule for the halo exchange operation.

    Parameters
    ----------
    halo_extents : Tuple[int, int, int]
        Extents of the halo in x, y, and z dimensions.
    halo_periods : Tuple[bool, bool, bool]
        Periodicity of the halo in x, y, and z dimensions.
    ctx
        Context for the operation.
    g : Array
        Gradient array.

    Returns
    -------
    Tuple[Array]
        Gradient array after the halo exchange operation.
    """
  return halo_p_lower(g, halo_extents, halo_periods),


# Define VJP for custom halo_exchange operation
halo_exchange.defvjp(_halo_fwd_rule, _halo_bwd_rule)
