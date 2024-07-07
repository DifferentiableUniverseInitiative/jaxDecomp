import jax
import jaxdecomp

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

print(f"Started process {rank} of {size}")

import argparse
import os
import time
from functools import partial

import jax.lax as lax
import jax.numpy as jnp
import jax_cosmo as jc
import numpy as np
from growth import dGfa, growth_factor, growth_rate
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

def _global_to_local_size(mesh_shape, sharding):
    """ Utility function to compute the expected local size of a mesh
    given the global size and the sharding strategy. 
    """
    return mesh_shape # TODO: sort out how to get the information from sharding

def fttk(mesh_shape: Tuple[int, int, int], sharding) -> list:
    """
    Generate Fourier transform wave numbers for a given mesh.

    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh grid.
    sharding : Any
        Sharding strategy for the array.

    Returns
    -------
    list
        List of wave number arrays for each dimension.
    """
    kd = np.fft.fftfreq(mesh_shape[0]) * 2 * np.pi
    return [
      jax.make_array_from_callback(
          (mesh_shape[0], 1, 1),
          sharding=jax.sharding.NamedSharding(sharding.mesh, P('z')),
          data_callback=lambda x: kd.reshape([-1, 1, 1])[x]),
      jax.make_array_from_callback(
          (1, mesh_shape[1], 1),
          sharding=jax.sharding.NamedSharding(sharding.mesh, P(None, 'y')),
          data_callback=lambda x: kd.reshape([1, -1, 1])[x]),
      kd.reshape([1, 1, -1])
    ]

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
    kk = jnp.sqrt(kx**2 + ky**2 + kz**2)
    laplace_kernel = jnp.where(kk == 0, 1., 1. / (kx**2 + ky**2 + kz**2))
    grav_kernel = [laplace_kernel * 1j / 6.0 * (8 * jnp.sin(kx) - jnp.sin(2 * kx)),
                   laplace_kernel * 1j / 6.0 * (8 * jnp.sin(ky) - jnp.sin(2 * ky)),
                   laplace_kernel * 1j / 6.0 * (8 * jnp.sin(kz) - jnp.sin(2 * kz))]
    return grav_kernel

def gaussian_field_and_forces(mesh_shape, box_size, power_spectrum, seed, sharding):
    """
    Generate a Gaussian field with a given power spectrum, along with gravitational forces.
    
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh.
    box_size : float
        Size of the box.
    power_spectrum : callable
        Power spectrum function.
    seed : int
        Seed for the random number generator.
    sharding : Any
        Sharding strategy for the array.
    
    Returns
    -------
    delta, forces : tuple of jnp.ndarray
        The generated Gaussian field and the gravitational forces.
    """
    local_mesh_shape = _global_to_local_size(mesh_shape, sharding)

    # Create a distributed field drawn from a Gaussian distribution in real space
    delta = jax.make_array_from_single_device_arrays(shape=mesh_shape,
                                                     sharding=sharding,
                                                     arrays=[jax.random.normal(seed, local_mesh_shape, dtype='float32')])

    # Compute the Fourier transform of the field
    delta_k = jaxdecomp.fft.pfft3d(delta.astype(jnp.complex64))

    # Compute the Fourier wavenumbers of the field
    kx, ky, kz = fttk(mesh_shape, sharding)
    kk = jnp.sqrt((kx / box_size * mesh_shape[0])**2 +
                  (ky / box_size * mesh_shape[1])**2 +
                  (kz / box_size * mesh_shape[2])**2)
    
    # Apply power spectrum to Fourier modes
    delta_k *= power_spectrum(kk)**0.5 * jnp.prod(mesh_shape) / jnp.prod(box_size)
    
    # Compute inverse Fourier transform to recover the initial conditions in real space
    delta = jaxdecomp.fft.pifft3d(delta_k).real

    # Compute gravitational forces associated with this field
    grav_kernel = gravity_kernel([kx, ky, kz])
    forces_k = [g * delta_k for g in grav_kernel]

    # Retrieve the forces in real space by inverse Fourier transforming
    forces = jnp.stack([jaxdecomp.fft.pifft3d(f).real for f in forces_k], axis=-1)

    return delta, forces


def cic_paint(displacement, halo_size=32):
    original_shape = displacement.shape
    
    mesh = jnp.zeros(original_shape[:-1], dtype='float32')
    
    # Padding the output array along the two first dimensions
    mesh = jnp.pad(mesh, [[halo_size, halo_size], [halo_size, halo_size], [0, 0]])

    a,b,c = jnp.meshgrid(jnp.arange(local_mesh_shape[0]),
                         jnp.arange(local_mesh_shape[1]), 
                         jnp.arange(local_mesh_shape[2]))
    # adding an offset of size halo size
    pmid = jnp.stack([b+halo_size,a+halo_size,c], axis=-1)
    pmid = pmid.reshape([-1,3])
    
    painted_field = scatter(pmid, displacement.reshape([-1,3]), mesh)

    # Perform halo exchange to get the correct values at the boundaries
    painted_field = jaxdecomp.halo_exchange(field, 
                                    halo_extents=(halo_size//2, halo_size//2, 0), 
                                    halo_periods=(True, True, True), reduce_halo=False)
    
    # unpadding the output array
    field = unpad(painted_field)
    return field

def simulation_fn(cosmology, mesh_shape, box_size, seed, a, sharding):
    """
    Run a simulation to generate initial conditions and density field using LPT.
    
    Parameters
    ----------
    mesh_shape : tuple of int
        Shape of the mesh.
    box_size : float
        Size of the box.
    power_spectrum : callable
        Power spectrum function.
    seed : int
        Seed for the random number generator.
    a : float
        Scale factor.
    sharding : Any
        Sharding strategy for the array.
    
    Returns
    -------
    initial_conditions, field : tuple of jnp.ndarray
        Initial conditions and the density field.
    """
    # Define the power spectrum
    power_spectrum = lambda k: jc.power.linear_matter_power(cosmology, k)

    # Generate a Gaussian field and gravitational forces from a power spectrum
    intial_conditions, initial_forces = gaussian_field_and_forces(mesh_shape, box_size, power_spectrum, seed, sharding)

    # Compute the LPT displacement of that particles initialy placed on a regular grid
    # would experience at scale factor a, by simple Zeldovich approximation
    initial_displacement = jc.background.growth_factor(cosmology, a) * initial_forces
    
    # Paints the displaced particles on a mesh to obtain the density field
    final_field = cic_paint(initial_displacement, halo_size=32)

    return intial_conditions, final_field

if __name__ == '__main__':

  jax.config.update('jax_enable_x64', False)

  parser = argparse.ArgumentParser()

  parser.add_argument('-s', '--size', type=int, default=64)
  parser.add_argument('-p', '--pdims', type=str, default='1x1')
  parser.add_argument('-b', '--box_size', type=int, default=200)
  parser.add_argument('-hs', '--halo_size', type=int, default=32)
  parser.add_argument('-o', '--output', type=str, default='out')

  args = parser.parse_args()

  print(f"Running with arguments {args}")

  # *********************************
  # Setup
  # *********************************
  master_key = jax.random.PRNGKey(42)
  key = jax.random.split(master_key, size)[rank]
  # Read parameters
  pdims = tuple(map(int, args.pdims.split('x')))
  mesh_shape = (args.size, args.size, args.size)
  box_size = [float(args.box_size), float(args.box_size), float(args.box_size)]
  halo_size = args.halo_size

  output_dir = args.output
  # Create output directory recursively
  os.makedirs(output_dir, exist_ok=True)

  # Create computing mesh
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('y', 'z'))
  sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))
  replicate = jax.sharding.NamedSharding(mesh, P())

  print(f"Saving on folder {output_dir}")
  print(f"Created initial field {z.shape} and sharding {z.sharding}")
  print(f"Created painting mesh with shape {painting_mesh.shape}")
  print(f"And sharding  {painting_mesh.sharding}")
  print(f"Created positions {pos.shape} and sharding {pos.sharding}")
  print("Corrected positions for rank {rank}  --> ")
  print(
      f"pos shape {pos.shape} pos sharding = {pos.sharding} shape of local {pos.addressable_data(0).shape}"
  )

  with mesh:
    jit_start = time.perf_counter()
    initial_conds, field = intial_conditions(z, kvec, pos, painting_mesh, a=1.)
    field.block_until_ready()
    jit_end = time.perf_counter()

    print(f"JIT done in {jit_end - jit_start}")

    start = time.perf_counter()
    initial_conds, field = forward_fn(z, kvec, pos, painting_mesh, a=1.)
    field.block_until_ready()
    end = time.perf_counter()

    print(f"Execution done in {end - start}")

  with open(f"{output_dir}/log_{rank}.log", 'w') as log_file:
    log_file.write(f"JIT time: {jit_end - jit_start}\n")
    log_file.write(f"Execution time: {end - start}\n")

  # Saving results
  np.save(f'{output_dir}/initial_conditions_{rank}.npy',
          initial_conds.addressable_data(0))
  np.save(f'{output_dir}/field_{rank}.npy', field.addressable_data(0))

  print(f"Finished saved to {output_dir}")

jax.distributed.shutdown()
