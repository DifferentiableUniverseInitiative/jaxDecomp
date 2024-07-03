from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp
import jaxpm as jaxpm
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


def _chunk_split(ptcl_num: int, chunk_size: Optional[int], *arrays:
                 jnp.ndarray) -> Tuple[Optional[list], list]:
  """
    Split and reshape particle arrays into chunks and remainders.

    Parameters
    ----------
    ptcl_num : int
        Number of particles.
    chunk_size : int, optional
        Size of each chunk. If None, no chunking is performed.
    *arrays : jnp.ndarray
        Arrays to be split.

    Returns
    -------
    tuple
        A tuple containing the remainder and chunks of the arrays.
    """
  chunk_size = ptcl_num if chunk_size is None else min(chunk_size, ptcl_num)
  remainder_size = ptcl_num % chunk_size
  chunk_num = ptcl_num // chunk_size

  remainder = None
  chunks = arrays
  if remainder_size:
    remainder = [x[:remainder_size] if x.ndim != 0 else x for x in arrays]
    chunks = [x[remainder_size:] if x.ndim != 0 else x for x in arrays]

  # `scan` triggers errors in scatter and gather without the `full`
  chunks = [
      x.reshape(chunk_num, chunk_size, *x.shape[1:])
      if x.ndim != 0 else jnp.full(chunk_num, x) for x in chunks
  ]

  return remainder, chunks


def enmesh(i1: jnp.ndarray, d1: jnp.ndarray, a1: float,
           s1: Optional[jnp.ndarray], b12: float, a2: Optional[float],
           s2: Optional[jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """
    Multilinear enmeshing of indices and displacements.

    Parameters
    ----------
    i1 : jnp.ndarray
        Initial indices.
    d1 : jnp.ndarray
        Initial displacements.
    a1 : float
        Scaling factor for the initial grid.
    s1 : jnp.ndarray, optional
        Shape of the initial grid.
    b12 : float
        Displacement offset.
    a2 : float, optional
        Scaling factor for the target grid.
    s2 : jnp.ndarray, optional
        Shape of the target grid.

    Returns
    -------
    tuple
        A tuple containing the target indices and fractions.
    """
  i1 = jnp.asarray(i1)
  d1 = jnp.asarray(d1)
  a1 = jnp.float64(a1) if a2 is not None else jnp.array(a1, dtype=d1.dtype)
  if s1 is not None:
    s1 = jnp.array(s1, dtype=i1.dtype)
  b12 = jnp.float64(b12)
  if a2 is not None:
    a2 = jnp.float64(a2)
  if s2 is not None:
    s2 = jnp.array(s2, dtype=i1.dtype)

  dim = i1.shape[1]
  neighbors = (jnp.arange(2**dim, dtype=i1.dtype)[:, jnp.newaxis] >> jnp.arange(
      dim, dtype=i1.dtype)) & 1

  if a2 is not None:
    P = i1 * a1 + d1 - b12
    P = P[:, jnp.newaxis]  # insert neighbor axis
    i2 = P + neighbors * a2  # multilinear

    if s1 is not None:
      L = s1 * a1
      i2 %= L

    i2 //= a2
    d2 = P - i2 * a2

    if s1 is not None:
      d2 -= jnp.rint(d2 / L) * L  # also abs(d2) < a2 is expected

    i2 = i2.astype(i1.dtype)
    d2 = d2.astype(d1.dtype)
    a2 = a2.astype(d1.dtype)

    d2 /= a2
  else:
    i12, d12 = jnp.divmod(b12, a1)
    i1 -= i12.astype(i1.dtype)
    d1 -= d12.astype(d1.dtype)

    # insert neighbor axis
    i1 = i1[:, jnp.newaxis]
    d1 = d1[:, jnp.newaxis]

    # multilinear
    d1 /= a1
    i2 = jnp.floor(d1).astype(i1.dtype)
    i2 += neighbors
    d2 = d1 - i2
    i2 += i1

    if s1 is not None:
      i2 %= s1

  f2 = 1 - jnp.abs(d2)

  if s1 is None and s2 is not None:  # all i2 >= 0 if s1 is not None
    i2 = jnp.where(i2 < 0, s2, i2)

  f2 = f2.prod(axis=-1)

  return i2, f2


def _scatter_chunk(
    carry: Tuple[jnp.ndarray, float,
                 float], chunk: Tuple[jnp.ndarray, jnp.ndarray,
                                      jnp.ndarray], mesh_shape: Tuple[int, int,
                                                                      int]
) -> Tuple[Tuple[jnp.ndarray, float, float], None]:
  """
    Scatter chunk data onto the mesh.

    Parameters
    ----------
    carry : tuple
        A tuple containing the current state of the mesh, offset, and cell size.
    chunk : tuple
        A tuple containing particle midpoints, displacements, and values.
    mesh_shape : tuple of int
        Shape of the mesh grid.

    Returns
    -------
    tuple
        Updated carry tuple and None.
    """
  mesh, offset, cell_size = carry
  pmid, disp, val = chunk
  spatial_ndim = pmid.shape[1]
  spatial_shape = mesh.shape

  # multilinear mesh indices and fractions
  ind, frac = enmesh(pmid, disp, cell_size, mesh_shape, offset, cell_size,
                     spatial_shape)
  # scatter
  ind = tuple(ind[..., i] for i in range(spatial_ndim))
  mesh = mesh.at[ind].add(val * frac)

  carry = mesh, offset, cell_size
  return carry, None


def scatter(pmid: jnp.ndarray,
            disp: jnp.ndarray,
            mesh: jnp.ndarray,
            mesh_shape: Tuple[int, int, int],
            chunk_size: int = 2**24,
            val: float = 1.0,
            offset: float = 0.0,
            cell_size: float = 1.0) -> jnp.ndarray:
  """
    Scatter particle data onto a mesh grid.

    Parameters
    ----------
    pmid : jnp.ndarray
        Particle midpoints.
    disp : jnp.ndarray
        Particle displacements.
    mesh : jnp.ndarray
        Mesh grid to scatter onto.
    mesh_shape : tuple of int
        Shape of the mesh grid.
    chunk_size : int, optional
        Size of each chunk. Default is 2**24.
    val : float, optional
        Value to scatter. Default is 1.0.
    offset : float, optional
        Offset value. Default is 0.0.
    cell_size : float, optional
        Cell size of the grid. Default is 1.0.

    Returns
    -------
    jnp.ndarray
        Updated mesh grid with scattered values.
    """
  ptcl_num, spatial_ndim = pmid.shape
  val = jnp.asarray(val)
  mesh = jnp.asarray(mesh)

  remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)
  carry = mesh, offset, cell_size
  scatter_chunk_fn = partial(_scatter_chunk, mesh_shape=mesh_shape)
  if remainder is not None:
    carry = scatter_chunk_fn(carry, remainder)[0]
  carry = scan(scatter_chunk_fn, carry, chunks)[0]
  mesh = carry[0]
  return mesh
