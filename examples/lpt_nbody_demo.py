import argparse
import os
from functools import partial
from typing import Any, Callable, Hashable

Specs = Any
AxisName = Hashable

import jax

jax.config.update('jax_enable_x64', False)

import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from jax._src import mesh as mesh_lib
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
  mesh = mesh_lib.thread_resources.env.physical_mesh
  return shard_map(f, mesh, in_specs, out_specs, check_rep, auto)


def _global_to_local_size(nc: int):
  """ Helper function to get the local size of a mesh given the global size.
  """
  pdims = mesh_lib.thread_resources.env.physical_mesh.devices.shape
  return [nc // pdims[0], nc // pdims[1], nc]


def fttk(nc: int) -> list:
  """
    Generate Fourier transform wave numbers for a given mesh.

    Args:
        nc (int): Shape of the mesh grid.

    Returns:
        list: List of wave number arrays for each dimension in
        the order [kx, ky, kz].
  """
  kd = np.fft.fftfreq(nc) * 2 * np.pi

  @partial(
      shmap,
      in_specs=(P('x'), P('y'), P(None)),
      out_specs=(P('x'), P(None, 'y'), P(None)))
  def get_kvec(ky, kz, kx):
    return (ky.reshape([-1, 1, 1]),
            kz.reshape([1, -1, 1]),
            kx.reshape([1, 1, -1])) # yapf: disable
  ky, kz, kx = get_kvec(kd, kd, kd)  # The order of the output
  # corresponds to the order of dimensions in the transposed FFT
  # output
  return kx, ky, kz


def gravity_kernel(kx, ky, kz):
  """ Computes a Fourier kernel combining laplace and derivative
    operators to compute gravitational forces.

    Args:
        kvec (tuple of float): Wave numbers in Fourier space.

    Returns:
        tuple of jnp.ndarray: kernels for each dimension.
  """
  kk = kx**2 + ky**2 + kz**2
  laplace_kernel = jnp.where(kk == 0, 1., 1. / kk)

  grav_kernel = (laplace_kernel * 1j * kx,
                 laplace_kernel * 1j * ky,
                 laplace_kernel * 1j * kz) # yapf: disable
  return grav_kernel


def gaussian_field_and_forces(key, nc, box_size, power_spectrum):
  """
    Generate a Gaussian field with a given power spectrum, along with gravitational forces.

    Args:
        key (int): Key for the random number generator.
        nc (int): Number of cells in the mesh.
        box_size (float): Size of the box.
        power_spectrum (callable): Power spectrum function.

    Returns:
        tuple of jnp.ndarray: The generated Gaussian field and the gravitational forces.
  """
  local_mesh_shape = _global_to_local_size(nc)

  # Create a distributed field drawn from a Gaussian distribution in real space
  delta = shmap(
      partial(jax.random.normal, shape=local_mesh_shape, dtype='float32'),
      in_specs=P(None),
      out_specs=P('x', 'y'))(key)  # yapf: disable

  # Compute the Fourier transform of the field
  delta_k = jaxdecomp.fft.pfft3d(delta.astype(jnp.complex64))

  # Compute the Fourier wavenumbers of the field
  kx, ky, kz = fttk(nc)
  kk = jnp.sqrt(kx**2 + ky**2 + kz**2) * (nc / box_size)

  # Apply power spectrum to Fourier modes
  delta_k *= (power_spectrum(kk) * (nc / box_size)**3)**0.5

  # Compute inverse Fourier transform to recover the initial conditions in real space
  delta = jaxdecomp.fft.pifft3d(delta_k).real

  # Compute gravitational forces associated with this field
  grav_kernel = gravity_kernel(kx, ky, kz)
  forces_k = [g * delta_k for g in grav_kernel]

  # Retrieve the forces in real space by inverse Fourier transforming
  forces = jnp.stack([jaxdecomp.fft.pifft3d(f).real for f in forces_k], axis=-1)

  return delta, forces


def cic_paint(displacement, halo_size):
  """ Paints particles on a mesh using Cloud-In-Cell interpolation.

    Args:
        displacement (jnp.ndarray): Displacement of each particle.
        halo_size (int): Halo size for painting.

    Returns:
        jnp.ndarray: Density field.
  """
  local_mesh_shape = _global_to_local_size(displacement.shape[0])
  hs = halo_size

  @partial(shmap, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
  def cic_op(disp):
    """ CiC operation on each local slice of the mesh."""
    # Create a mesh to paint the particles on for the local slice
    mesh = jnp.zeros(disp.shape[:-1], dtype='float32')

    # Padding the mesh along the two first dimensions
    mesh = jnp.pad(mesh, [[hs, hs], [hs, hs], [0, 0]])

    # Compute the position of the particles on a regular grid
    pos_x, pos_y, pos_z = jnp.meshgrid(
        jnp.arange(local_mesh_shape[0]),
        jnp.arange(local_mesh_shape[1]),
        jnp.arange(local_mesh_shape[2]),
        indexing='ij')

    # adding an offset of size halo size
    pos = jnp.stack([pos_x + hs, pos_y + hs, pos_z], axis=-1)

    # Apply scatter operation to paint the particles on the local mesh
    field = scatter(pos.reshape([-1, 3]), disp.reshape([-1, 3]), mesh)

    return field

  # Performs painting on a padded mesh, with halos on the two first dimensions
  field = cic_op(displacement)

  # Run halo exchange to get the correct values at the boundaries
  field = jaxdecomp.halo_exchange(
      field,
      halo_extents=(hs // 2, hs // 2, 0),
      halo_periods=(True, True, True))

  @partial(shmap, in_specs=(P('x', 'y'),), out_specs=P('x', 'y'))
  def unpad(x):
    """ Removes the padding and reduce the halo regions"""
    x = x.at[hs:hs + hs // 2].add(x[:hs // 2])
    x = x.at[-(hs + hs // 2):-hs].add(x[-hs // 2:])
    x = x.at[:, hs:hs + hs // 2].add(x[:, :hs // 2])
    x = x.at[:, -(hs + hs // 2):-hs].add(x[:, -hs // 2:])
    return x[hs:-hs, hs:-hs, :]

  # Unpad the output array
  field = unpad(field)
  return field


@partial(jax.jit, static_argnames=('nc', 'box_size', 'halo_size'))
def simulation_fn(key, nc, box_size, halo_size, a=1.0):
  """
    Run a simulation to generate initial conditions and density field using LPT.

    Args:
        key (list of int): Jax random key for the random number generator.
        nc (int): Size of the mesh grid.
        box_size (float): Size of the box.
        halo_size (int): Halo size for painting.
        a (float): Scale factor of final field.

    Returns:
            tuple of jnp.ndarray: Initial conditions and final density field.
  """
  # Build a default cosmology
  cosmo = jc.Planck15()

  # Create a small function to generate the linear matter power spectrum at arbitrary k
  k = jnp.logspace(-4, 1, 128)
  pk = jc.power.linear_matter_power(cosmo, k)
  pk_fn = jax.jit(lambda x: jc.scipy.interpolate.interp(x.reshape([-1]), k, pk).
                  reshape(x.shape))

  # Generate a Gaussian field and gravitational forces from a power spectrum
  intial_conditions, initial_forces = gaussian_field_and_forces(
      key=key, nc=nc, box_size=box_size, power_spectrum=pk_fn)

  # Compute the LPT displacement that particles initialy placed on a regular grid
  # would experience at scale factor a, by simple Zeldovich approximation
  initial_displacement = jc.background.growth_factor(
      cosmo, jnp.atleast_1d(a)) * initial_forces

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
  mesh = Mesh(devices.T, axis_names=('x', 'y'))

  # Run the simulation on the compute mesh
  with mesh:
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
  parser = argparse.ArgumentParser("Distributed LPT N-body simulation.")
  parser.add_argument(
      '--pdims', type=str, default='1x1', help="Processor grid dimensions")
  parser.add_argument(
      '--nc', type=int, default=256, help="Number of cells in the mesh")
  parser.add_argument(
      '--box_size', type=float, default=512., help="Box size in Mpc/h")
  parser.add_argument(
      '--halo_size', type=int, default=32, help="Halo size for painting")
  parser.add_argument('--output', type=str, default='out')
  args = parser.parse_args()

  main(args)
