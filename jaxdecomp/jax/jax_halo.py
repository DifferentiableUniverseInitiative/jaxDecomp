from functools import partial

from jax import lax
from jax._src import mesh as mesh_lib
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src import _jaxdecomp
from jaxdecomp._src.spmd_ops import autoshmap, get_pencil_type


def _halo_exchange_slab_xy(operand, halo_extent):
  # pdims are (Py=1,Pz=N)
  # input is (Z / Pz , Y , X) with specs P('z', None)
  z_size = lax.psum(1, 'z')

  # Extract the halo halo
  upper_halo = operand[halo_extent:halo_extent + halo_extent]
  lower_halo = operand[-(halo_extent + halo_extent):-halo_extent]

  reverse_indexing_z = [(j, (j + 1) % z_size) for j in range(z_size)]
  forward_indexing_z = [((j + 1) % z_size, j) for j in range(z_size)]

  # circular shift to the next rank
  exchanged_upper_halo = lax.ppermute(
      upper_halo, axis_name='z', perm=reverse_indexing_z)

  # circular shift to the previous rank
  exchanged_lower_halo = lax.ppermute(
      lower_halo, axis_name='z', perm=forward_indexing_z)

  # Upper halo gets the lower halo of the previous rank
  operand = operand.at[:halo_extent].set(exchanged_lower_halo)
  # Lower halo gets the upper halo of the next rank
  operand = operand.at[-halo_extent:].set(exchanged_upper_halo)

  return operand


def _halo_exchange_slab_yz(operand, halo_extent):
  # pdims are (Py=N,Pz=1)
  # input is (Z , Y /Py, X) with specs P(None), 'y')
  y_size = lax.psum(1, 'y')

  # Extract the halo halo
  right_halo = operand[:, halo_extent:halo_extent + halo_extent]
  left_halo = operand[:, -(halo_extent + halo_extent):-halo_extent]

  reverse_indexing_y = [((j + 1) % y_size, j) for j in range(y_size)]
  forward_indexing_y = [(j, (j + 1) % y_size) for j in range(y_size)]

  # circular shift to the next rank
  exchanged_right_halo = lax.ppermute(
      right_halo, axis_name='y', perm=reverse_indexing_y)

  # circular shift to the previous rank
  exchanged_left_halo = lax.ppermute(
      left_halo, axis_name='y', perm=forward_indexing_y)

  # Right halo gets the left halo of the next rank
  operand = operand.at[:, :halo_extent].set(exchanged_left_halo)
  # Left halo gets the right halo of the previous rank
  operand = operand.at[:, -halo_extent:].set(exchanged_right_halo)

  return operand


def _halo_exchange_pencils(operand, halo_extent):
  # pdims are (Py=N,Pz=N)
  # input is (Z/Pz , Y/Py , X) with specs P(None), 'y', 'z')
  y_size = lax.psum(1, 'y')
  z_size = lax.psum(1, 'z')

  upper_halo = operand[halo_extent:halo_extent + halo_extent]
  lower_halo = operand[-(halo_extent + halo_extent):-halo_extent]

  right_halo = operand[:, halo_extent:halo_extent + halo_extent]
  left_halo = operand[:, -(halo_extent + halo_extent):-halo_extent]

  upper_right_corner = operand[halo_extent:halo_extent + halo_extent,
                               halo_extent:halo_extent + halo_extent]
  upper_left_corner = operand[halo_extent:halo_extent + halo_extent,
                              -(halo_extent + halo_extent):-halo_extent]
  lower_right_corner = operand[-(halo_extent + halo_extent):-halo_extent,
                               halo_extent:halo_extent + halo_extent]
  lower_left_corner = operand[-(halo_extent + halo_extent):-halo_extent,
                              -(halo_extent + halo_extent):-halo_extent]

  reverse_indexing_z = [(j, (j + 1) % z_size) for j in range(z_size)]
  forward_indexing_z = [((j + 1) % z_size, j) for j in range(z_size)]
  reverse_indexing_y = [((j + 1) % y_size, j) for j in range(y_size)]
  forward_indexing_y = [(j, (j + 1) % y_size) for j in range(y_size)]

  # circular shift to the next rank
  exchanged_upper_halo = lax.ppermute(
      upper_halo, axis_name='z', perm=reverse_indexing_z)

  # circular shift to the previous rank
  exchanged_lower_halo = lax.ppermute(
      lower_halo, axis_name='z', perm=forward_indexing_z)

  # circular shift to the next rank
  exchanged_right_halo = lax.ppermute(
      right_halo, axis_name='y', perm=reverse_indexing_y)

  # circular shift to the previous rank
  exchanged_left_halo = lax.ppermute(
      left_halo, axis_name='y', perm=forward_indexing_y)

  # For corners we need to do two circular shifts

  # Handle upper right corners
  # circular shift to the next rank
  exchanged_upper_right_corner = lax.ppermute(
      upper_right_corner, axis_name='z', perm=reverse_indexing_z)
  # second circular shift to the next rank
  exchanged_upper_right_corner = lax.ppermute(
      exchanged_upper_right_corner, axis_name='y', perm=reverse_indexing_y)

  # Handle upper left corners
  # circular shift to the next rank
  exchanged_upper_left_corner = lax.ppermute(
      upper_left_corner, axis_name='z', perm=reverse_indexing_z)
  # second circular shift to the previous rank
  exchanged_upper_left_corner = lax.ppermute(
      exchanged_upper_left_corner, axis_name='y', perm=forward_indexing_y)

  # Handle lower right corners
  # circular shift to the previous rank
  exchanged_lower_right_corner = lax.ppermute(
      lower_right_corner, axis_name='z', perm=forward_indexing_z)
  # second circular shift to the next rank
  exchanged_lower_right_corner = lax.ppermute(
      exchanged_lower_right_corner, axis_name='y', perm=reverse_indexing_y)

  # Handle lower left corners
  # circular shift to the previous rank
  exchanged_lower_left_corner = lax.ppermute(
      lower_left_corner, axis_name='z', perm=forward_indexing_z)
  # second circular shift to the previous rank
  exchanged_lower_left_corner = lax.ppermute(
      exchanged_lower_left_corner, axis_name='y', perm=forward_indexing_y)

  # Upper halo gets the lower halo of the previous rank
  operand = operand.at[:halo_extent].set(exchanged_lower_halo)
  # Lower halo gets the upper halo of the next rank
  operand = operand.at[-halo_extent:].set(exchanged_upper_halo)
  # Right halo gets the left halo of the next rank
  operand = operand.at[:, :halo_extent].set(exchanged_left_halo)
  # Left halo gets the right halo of the previous rank
  operand = operand.at[:, -halo_extent:].set(exchanged_right_halo)
  # Handle corners
  # Upper right corner gets the lower left corner of the next rank
  operand = operand.at[:halo_extent, :halo_extent].set(
      exchanged_lower_left_corner)
  # Upper left corner gets the lower right corner of the next rank
  operand = operand.at[:halo_extent,
                       -halo_extent:].set(exchanged_lower_right_corner)
  # Lower right corner gets the upper left corner of the next rank
  operand = operand.at[-halo_extent:, :halo_extent].set(
      exchanged_upper_left_corner)
  # Lower left corner gets the upper right corner of the next rank
  operand = operand.at[-halo_extent:,
                       -halo_extent:].set(exchanged_upper_right_corner)

  return operand


def jax_halo_exchange(operand, halo_extent):
  match get_pencil_type():
    case _jaxdecomp.NO_DECOMP:
      return operand
    case _jaxdecomp.SLAB_XY:
      return autoshmap(
          partial(_halo_exchange_slab_xy, halo_extent=halo_extent), P('z', 'y'),
          P('z', 'y'))(
              operand)
    case _jaxdecomp.SLAB_YZ:
      return autoshmap(
          partial(_halo_exchange_slab_yz, halo_extent=halo_extent), P('z', 'y'),
          P('z', 'y'))(
              operand)
    case _jaxdecomp.PENCILS:
      return autoshmap(
          partial(_halo_exchange_pencils, halo_extent=halo_extent), P('z', 'y'),
          P('z', 'y'))(
              operand)
    case _:
      raise ValueError('Invalid pencil type')
