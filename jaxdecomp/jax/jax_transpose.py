from typing import Any, Callable, Hashable

Specs = Any
AxisName = Hashable

from functools import partial

from jax import lax
from jax._src import mesh as mesh_lib
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

import jaxdecomp


def autoshmap(f: Callable,
              in_specs: Specs,
              out_specs: Specs,
              check_rep: bool = True,
              auto: frozenset[AxisName] = frozenset(),
              in_fourrier_space=False) -> Callable:
  """Helper function to wrap the provided function in a shard map if
    the code is being executed in a mesh context."""
  mesh = mesh_lib.thread_resources.env.physical_mesh
  if mesh.empty:
    return f
  else:
    if in_fourrier_space and 1 in mesh.devices.shape:
      in_specs, out_specs = switch_specs((in_specs, out_specs))
    return shard_map(f, mesh, in_specs, out_specs, check_rep, auto)


def switch_specs(specs):
  if isinstance(specs, P):
    new_axes = tuple(
        'y' if ax == 'z' else 'z' if ax == 'y' else ax for ax in specs)
    return P(*new_axes)
  elif isinstance(specs, tuple):
    return tuple(switch_specs(sub_spec) for sub_spec in specs)
  else:
    raise TypeError("Element must be either a PartitionSpec or a tuple")


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
