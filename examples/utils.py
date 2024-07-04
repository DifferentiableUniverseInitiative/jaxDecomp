from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.lax import scan
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def generate_random_field(
    mesh_shape: Tuple[int, int, int], sharding: NamedSharding,
    key: jax.random.PRNGKey, local_mesh_shape: Tuple[int, int,
                                                     int]) -> jnp.ndarray:
  """
    Generate a random field using a normal distribution.

    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the full mesh.
    sharding : Any
        Sharding strategy for the array.
    key : jax.random.PRNGKey
        Random key for generating the noise.
    local_mesh_shape : tuple of int
        Shape of the local mesh.

    Returns
    -------
    jnp.ndarray
        Generated random field.
    """
  return jax.make_array_from_single_device_arrays(
      shape=mesh_shape,
      sharding=sharding,
      arrays=[jax.random.normal(key, local_mesh_shape, dtype='float32')])


def generate_initial_positions(mesh_shape: Tuple[int, int, int],
                               sharding: NamedSharding) -> jnp.ndarray:
  """
    Generate initial positions for particles on a mesh grid.

    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    sharding : Any
        Sharding strategy for the array.

    Returns
    -------
    jnp.ndarray
        Initial positions on the mesh grid.
    """
  pos = jax.make_array_from_callback(
      shape=tuple([*mesh_shape, 3]),
      sharding=sharding,
      data_callback=lambda x: jnp.stack(
          jnp.meshgrid(
              jnp.arange(mesh_shape[0])[x[0]],
              jnp.arange(mesh_shape[1])[x[1]],
              jnp.arange(mesh_shape[2]),
              indexing='ij'),
          axis=-1))
  return pos


def fttk(mesh_shape: Tuple[int, int, int], mesh: jax.sharding.Mesh) -> list:
  """
    Generate Fourier transform wave numbers for a given mesh.

    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    mesh : Any
        Mesh object for sharding.

    Returns
    -------
    list
        List of wave number arrays for each dimension.
    """
  kd = np.fft.fftfreq(mesh_shape[0]).astype('float32') * 2 * np.pi
  return [
      jax.make_array_from_callback(
          (mesh_shape[0], 1, 1),
          sharding=jax.sharding.NamedSharding(mesh, P('z')),
          data_callback=lambda x: kd.reshape([-1, 1, 1])[x]),
      jax.make_array_from_callback(
          (1, mesh_shape[1], 1),
          sharding=jax.sharding.NamedSharding(mesh, P(None, 'y')),
          data_callback=lambda x: kd.reshape([1, -1, 1])[x]),
      kd.reshape([1, 1, -1])
  ]
