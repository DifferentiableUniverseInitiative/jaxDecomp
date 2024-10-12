from jax import lax
from jax._src import mesh as mesh_lib
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src.spmd_ops import autoshmap


def jax_transpose_XtoY(operand):
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    return operand.transpose(
        [2, 0, 1]) if jaxdecomp.config.transpose_axis_contiguous else operand
  else:
    if jaxdecomp.config.transpose_axis_contiguous:
      # For contiguous version, the output spec is `P('y', 'z')`
      fn = lambda x: lax.all_to_all(
          x, 'y', 2, 1, tiled=True).transpose([2, 0, 1])
      return autoshmap(fn, P('z', 'y'), P('y', 'z'))(operand)
    else:
      # Non-contiguous version
      fn = lambda x: lax.all_to_all(x, 'y', 2, 1, tiled=True)
      return autoshmap(fn, P('z', 'y'), P('z', None, 'y'))(operand)


def jax_transpose_YtoZ(operand):
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    return operand.transpose(
        [2, 0, 1]) if jaxdecomp.config.transpose_axis_contiguous else operand
  else:
    if jaxdecomp.config.transpose_axis_contiguous:
      # For contiguous version, the output spec is `P('z', 'y')`
      fn = lambda x: lax.all_to_all(
          x, 'z', 2, 1, tiled=True).transpose([2, 0, 1])
      return autoshmap(fn, P('y', 'z'), P('z', 'y'))(operand)
    else:
      # Non-contiguous version
      fn = lambda x: lax.all_to_all(x, 'z', 1, 0, tiled=True)
      return autoshmap(fn, P('z', None, 'y'), P(None, 'z', 'y'))(operand)


def jax_transpose_ZtoY(operand):
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    return operand.transpose(
        [1, 2, 0]) if jaxdecomp.config.transpose_axis_contiguous else operand
  else:
    if jaxdecomp.config.transpose_axis_contiguous:
      # For contiguous version, the output spec is `P('y', 'z')`
      fn = lambda x: lax.all_to_all(
          x, 'z', 2, 0, tiled=True).transpose([1, 2, 0])
      return autoshmap(fn, P('z', 'y'), P('y', 'z'))(operand)
    else:
      # Non-contiguous version
      fn = lambda x: lax.all_to_all(x, 'z', 0, 1, tiled=True)
      return autoshmap(fn, P(None, 'z', 'y'), P('z', None, 'y'))(operand)


def jax_transpose_YtoX(operand):
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    return operand.transpose(
        [1, 2, 0]) if jaxdecomp.config.transpose_axis_contiguous else operand
  else:
    if jaxdecomp.config.transpose_axis_contiguous:
      # For contiguous version, the output spec is `P('y', 'z')`
      fn = lambda x: lax.all_to_all(
          x, 'y', 2, 0, tiled=True).transpose([1, 2, 0])
      return autoshmap(fn, P('y', 'z'), P('z', 'y'))(operand)
    else:
      # Non-contiguous version
      fn = lambda x: lax.all_to_all(x, 'y', 1, 2, tiled=True)
      return autoshmap(fn, P('z', None, 'y'), P('z', 'y'))(operand)


def jax_transpose_XtoZ(operand):
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    return operand.transpose(
        [2, 0, 1]) if jaxdecomp.config.transpose_axis_contiguous else operand
  else:
    if jaxdecomp.config.transpose_axis_contiguous:
      # For contiguous version, the output spec is `P('y', 'z')`
      fn = lambda x: lax.all_to_all(
          x, 'z', 2, 0, tiled=True).transpose([1, 2, 0])
      return autoshmap(fn, P('z', 'y'), P('y', 'z'))(operand)
    else:
      # Non-contiguous version
      fn = lambda x: lax.all_to_all(x, 'z', 2, 0, tiled=True)
      return autoshmap(fn, P('z', 'y'), P(None, 'y', 'z'))(operand)


def jax_transpose_ZtoX(operand):
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    return operand.transpose(
        [2, 0, 1]) if jaxdecomp.config.transpose_axis_contiguous else operand
  else:
    if jaxdecomp.config.transpose_axis_contiguous:
      # For contiguous version, the output spec is `P('y', 'z')`
      fn = lambda x: lax.all_to_all(
          x, 'z', 2, 1, tiled=True).transpose([2, 0, 1])
      return autoshmap(fn, P('y', 'z'), P('z', 'y'))(operand)
    else:
      # Non-contiguous version
      fn = lambda x: lax.all_to_all(x, 'z', 0, 2, tiled=True)
      return autoshmap(fn, P(None, 'y', 'z'), P('z', 'y'))(operand)
