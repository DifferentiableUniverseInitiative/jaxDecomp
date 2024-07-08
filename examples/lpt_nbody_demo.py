import argparse
import os
from functools import partial
from typing import Any, Callable, Hashable, Tuple

from jax._src import mesh as mesh_lib

Specs = Any
AxisName = Hashable

import jax

jax.config.update('jax_enable_x64', False)

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from scatter import scatter

import jaxdecomp


def shmap(f: Callable,
          in_specs: Specs,
          out_specs: Specs,
          check_rep: bool = True,
          auto: frozenset[AxisName] = frozenset()):
  """Helper function to create a shard_map function that extracts the mesh from the
    context."""
  # Extracts the mesh from the context
  mesh = mesh_lib.thread_resources.env.physical_mesh
  return shard_map(f, mesh, in_specs, out_specs, check_rep, auto)


def _global_to_local_size(mesh_shape):
  """ Utility function to compute the expected local size of a mesh
      given the global size and the sharding strategy.
  """
  pdims = mesh_lib.thread_resources.env.physical_mesh.devices.shape
  return [mesh_shape[0] // pdims[1], mesh_shape[1] // pdims[0], mesh_shape[2]]


def fttk(nc: int) -> list:
  """
    Generate Fourier transform wave numbers for a given mesh.

    Parameters
    ----------
    mesh_shape : int
        Shape of the mesh grid.
    sharding : Any
        Sharding strategy for the array.

    Returns
    -------
    list
        List of wave number arrays for each dimension.
  """
  kd = np.fft.fftfreq(nc) * 2 * np.pi

  @partial(
      shmap,
      in_specs=(P('z'), P('y'), P(None)),
      out_specs=(P('z'), P(None, 'y'), P(None)))
  def get_kvec(kx, ky, kz):
    return (kx.reshape([-1, 1, 1]), ky.reshape([1, -1,
                                                1]), kz.reshape([1, 1, -1]))

  return get_kvec(kd, kd, kd)


def gravity_kernel(kvec):
  """ Fourier kernel to compute gravitational forces from a Fourier space density field.

    Parameters
    ----------
    kvec : tuple of float
        Wave vector in Fourier space.

    Returns
    -------
    jnp.ndarray
        Gravitational kernel.
  """
  kx, ky, kz = kvec
  kk = kx**2 + ky**2 + kz**2
  laplace_kernel = jnp.where(kk == 0, 1., 1. / kk)
  grav_kernel = [
      laplace_kernel * 1j / 6.0 * (8 * jnp.sin(kx) - jnp.sin(2 * kx)),
      laplace_kernel * 1j / 6.0 * (8 * jnp.sin(ky) - jnp.sin(2 * ky)),
      laplace_kernel * 1j / 6.0 * (8 * jnp.sin(kz) - jnp.sin(2 * kz))
  ]
  return grav_kernel


def gaussian_field_and_forces(key, nc, box_size, power_spectrum):
  """
    Generate a Gaussian field with a given power spectrum, along with gravitational forces.

    Parameters
    ----------
    key : int
        key for the random number generator.
    nc : int
        Number of cells in the mesh.
    box_size : float
        Size of the box.
    power_spectrum : callable
        Power spectrum function.
    sharding : Any
        Sharding strategy for the array.

    Returns
    -------
    delta, forces : tuple of jnp.ndarray
        The generated Gaussian field and the gravitational forces.
    """
  mesh_shape = (nc,) * 3
  local_mesh_shape = _global_to_local_size(mesh_shape)

  # Create a distributed field drawn from a Gaussian distribution in real space
  @partial(shmap, in_specs=(), out_specs=P('z', 'y'))
  def _sample_gaussian():
    return jax.random.normal(key, local_mesh_shape, dtype='float32')

  delta = _sample_gaussian()

  # Compute the Fourier transform of the field
  delta_k = jaxdecomp.fft.pfft3d(delta.astype(jnp.complex64))

  # Compute the Fourier wavenumbers of the field
  kx, ky, kz = fttk(nc)
  kk = jnp.sqrt(kx**2 + ky**2 + kz**2) * (nc / box_size)**3

  # Apply power spectrum to Fourier modes
  delta_k *= (power_spectrum(kk) * (nc / box_size)**3)**0.5

  # Compute inverse Fourier transform to recover the initial conditions in real space
  delta = jaxdecomp.fft.pifft3d(delta_k).real

  # Compute gravitational forces associated with this field
  grav_kernel = gravity_kernel([kx, ky, kz])
  forces_k = [g * delta_k for g in grav_kernel]

  # Retrieve the forces in real space by inverse Fourier transforming
  forces = jnp.stack([jaxdecomp.fft.pifft3d(f).real for f in forces_k], axis=-1)

  return delta, forces


def cic_paint(displacement, halo_size):
  """ Paints particles on a mesh using Cloud-In-Cell interpolation.

    Parameters
    ----------
    displacement : jnp.ndarray
        Displacement field of particles.
    sharding : Any
        Sharding strategy for the array.
    halo_size : int
        Halo size for painting.

    Returns
    -------
    jnp.ndarray
        Density field.
  """
  local_mesh_shape = _global_to_local_size(displacement.shape)

  @partial(shmap, in_specs=(P('z', 'y'),), out_specs=P('z', 'y'))
  def cic_op(disp):
    """ CiC operation on each local slice of the mesh."""
    # Create a mesh to paint the particles on for the local slice
    mesh = jnp.zeros(disp.shape[:-1], dtype='float32')

    # Padding the output array along the two first dimensions
    mesh = jnp.pad(mesh,
                   [[halo_size, halo_size], [halo_size, halo_size], [0, 0]])

    a, b, c = jnp.meshgrid(
        jnp.arange(local_mesh_shape[0]),
        jnp.arange(local_mesh_shape[1]),
        jnp.arange(local_mesh_shape[2]),
        indexing='ij')

    # adding an offset of size halo size
    pmid = jnp.stack([a + halo_size, b + halo_size, c], axis=-1)
    return scatter(pmid.reshape([-1, 3]), disp.reshape([-1, 3]), mesh)

  # Performs painting on padded mesh
  field = cic_op(displacement)

  # Run halo exchange to get the correct values at the boundaries
  field = jaxdecomp.halo_exchange(
      field,
      halo_extents=(halo_size // 2, halo_size // 2, 0),
      halo_periods=(True, True, True))

  @partial(shmap, in_specs=(P('z', 'y'),), out_specs=P('z', 'y'))
  def unpad(x):
    """ Unpad the output array. """
    x = x.at[halo_size:halo_size + halo_size // 2].add(x[:halo_size // 2])
    x = x.at[-(halo_size + halo_size // 2):-halo_size].add(x[-halo_size // 2:])
    x = x.at[:, halo_size:halo_size + halo_size // 2].add(x[:, :halo_size // 2])
    x = x.at[:,
             -(halo_size + halo_size // 2):-halo_size].add(x[:,
                                                             -halo_size // 2:])
    return x[halo_size:-halo_size, halo_size:-halo_size, :]

  # Unpad the output array
  field = unpad(field)
  return field


@partial(jax.jit, static_argnames=('nc', 'box_size', 'halo_size'))
def simulation_fn(key, nc, box_size, halo_size, a=1.0):
  """
    Run a simulation to generate initial conditions and density field using LPT.

    Parameters
    ----------
    key : list of int
        Jax random key for the random number generator.
    nc : int
        Size of the mesh grid.
    box_size : float
        Size of the box.
    sharding : Any
        Sharding strategy for the simulation.
    halo_size: int
        Halo size for painting.
    a : float
        Scale factor of final field.

    Returns
    -------
    initial_conditions, field : tuple of jnp.ndarray
        Initial conditions and the density field.
  """
  # Build a default cosmology
  cosmology = jc.Planck15()

  # Create a small function to generate the linear matter power spectrum at arbitrary k
  k = jnp.logspace(-4, 1, 128)
  pk = jc.power.linear_matter_power(cosmology, k)
  pk_fn = jax.jit(lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).
                  reshape(x.shape))

  # Generate a Gaussian field and gravitational forces from a power spectrum
  intial_conditions, initial_forces = gaussian_field_and_forces(
      key=key, nc=nc, box_size=box_size, power_spectrum=pk_fn)

  # Compute the LPT displacement of that particles initialy placed on a regular grid
  # would experience at scale factor a, by simple Zeldovich approximation
  initial_displacement = jc.background.growth_factor(
      cosmology, jnp.atleast_1d(a)) * initial_forces

  # Paints the displaced particles on a mesh to obtain the density field
  final_field = cic_paint(initial_displacement, halo_size)

  return intial_conditions, final_field


def main(args):
  print(f"Running with arguments {args}")

  # Setting up distributed jax
  jax.distributed.initialize()
  rank = jax.process_index()
  size = jax.process_count()

  # Setting up distributed random numbers
  master_key = jax.random.PRNGKey(42)
  key = jax.random.split(master_key, size)[rank]

  # Create computing mesh and sharding information
  pdims = tuple(map(int, args.pdims.split('x')))
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('y', 'x'))

  with mesh:
    # Run the simulation on the compute mesh
    initial_conds, final_field = simulation_fn(
        key=key, nc=args.nc, box_size=args.box_size, halo_size=args.halo_size)

  # Create output directory to save the results
  output_dir = args.output
  os.makedirs(output_dir, exist_ok=True)
  np.save(f'{output_dir}/initial_conditions_{rank}.npy',
          initial_conds.addressable_data(0))
  np.save(f'{output_dir}/field_{rank}.npy', final_field.addressable_data(0))
  print(f"Finished saved to {output_dir}")

  # Closing distributed jax
  jax.distributed.shutdown()


if __name__ == '__main__':
  parser = argparse.ArgumentParser("LPT N-body simulation with JAX.")
  parser.add_argument(
      '--pdims', type=str, default='1x1', help="Processor grid dimensions")
  parser.add_argument(
      '--nc', type=int, default=256, help="Number of cells in the mesh")
  parser.add_argument(
      '--box_size',
      type=float,
      default=256.,
      help="Size of the simulation box in Mpc/h")
  parser.add_argument(
      '--halo_size', type=int, default=32, help="Halo size for painting")
  parser.add_argument('--output', type=str, default='out')
  args = parser.parse_args()

  main(args)
