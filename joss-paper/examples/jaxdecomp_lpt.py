import jax

import jaxdecomp

# Initialize the distributed computing environment
jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

print(f"Started process {rank} of {size}")

import argparse
import os
import time
from functools import partial

import jax.numpy as jnp
import jax_cosmo as jc
import jaxpm as jaxpm
import numpy as np
from growth import growth_factor
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from utils import *

## Usage  ##
# Using mpirun: mpirun -n 4 python jaxdecomp_lpt.py -s 64 -b 200 -p 2x2 -hs 32 -o out

if __name__ == '__main__':

  jax.config.update('jax_enable_x64', False)

  parser = argparse.ArgumentParser()

  parser.add_argument(
      '-s', '--size', type=int, default=64, help="Size of the mesh grid.")
  parser.add_argument(
      '-p',
      '--pdims',
      type=str,
      default='1x1',
      help="Processor grid dimensions.")
  parser.add_argument(
      '-b',
      '--box_size',
      type=int,
      default=200,
      help="Size of the simulation box.")
  parser.add_argument(
      '-hs',
      '--halo_size',
      type=int,
      default=32,
      help="Size of the halo region.")
  parser.add_argument(
      '-o',
      '--output',
      type=str,
      default='out',
      help="Output directory for results.")

  args = parser.parse_args()

  print(f"Running with arguments {args}")

  # Setup
  master_key = jax.random.PRNGKey(42)
  key = jax.random.split(master_key, size)[rank]

  pdims = tuple(map(int, args.pdims.split('x')))
  mesh_shape = (args.size, args.size, args.size)
  box_size = [float(args.box_size), float(args.box_size), float(args.box_size)]
  halo_size = args.halo_size

  output_dir = args.output
  os.makedirs(output_dir, exist_ok=True)

  # Create computing mesh
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('y', 'z'))
  sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))

  # Create all initial distributed tensors
  local_mesh_shape = [
      mesh_shape[0] // pdims[1], mesh_shape[1] // pdims[0], mesh_shape[2]
  ]

  # Create gaussian field distributed across the mesh
  z = generate_random_field(mesh_shape, sharding, key, local_mesh_shape)
  kvec = fttk(mesh_shape, mesh)

  print(f"Local mesh shape {local_mesh_shape}")
  print(f"Saving on folder {output_dir}")
  print(f"Created initial field {z.shape} and sharding {z.sharding}")

  @partial(shard_map, mesh=mesh, in_specs=(P('z', 'y'),), out_specs=P('z', 'y'))
  def cic_paint_sharded(displacement: jnp.ndarray) -> jnp.ndarray:
    """
        Distribute particles' displacements onto the mesh grid.

        Parameters
        ----------
        displacement : jnp.ndarray
            Particle displacements.
            shape : (x, y, z, 3)

        Returns
        -------
        jnp.ndarray
            Mesh with distributed particle data.
            shape : (x, y, z)
        """
    original_shape = displacement.shape
    mesh = jnp.zeros(original_shape[:-1], dtype='float32')

    # Padding the output array along the two first dimensions
    mesh = jnp.pad(mesh,
                   [[halo_size, halo_size], [halo_size, halo_size], [0, 0]])

    a, b, c = jnp.meshgrid(
        jnp.arange(local_mesh_shape[1]), jnp.arange(local_mesh_shape[0]),
        jnp.arange(local_mesh_shape[2]))

    # Adding an offset of size halo size
    pmid = jnp.stack([b + halo_size, a + halo_size, c], axis=-1)
    pmid = pmid.reshape([-1, 3])
    return scatter(pmid, displacement.reshape([-1, 3]), mesh, mesh_shape)

  def cic_paint(displacement: jnp.ndarray) -> jnp.ndarray:
    """
        Paint the particles onto the mesh grid and handle halo exchange.

        Parameters
        ----------
        displacement : jnp.ndarray
            Particle displacements .
            shape : (x , y, z, 3)

        Returns
        -------
        jnp.ndarray
            Final mesh with painted particles.
            shape : (x, y, z)
        """
    field = cic_paint_sharded(displacement)

    field = jaxdecomp.halo_exchange(
        field,
        halo_extents=(halo_size, halo_size, 0),
        halo_periods=(True, True, True))

    # Removing the padding
    field = jaxdecomp.slice_unpad(field, ((halo_size, halo_size),
                                          (halo_size, halo_size), (0, 0)),
                                  pdims)
    return field

  @partial(
      shard_map,
      mesh=mesh,
      in_specs=(P('z', 'y'), P('z', 'y')),
      out_specs=P('z', 'y'))
  def interpolate(kfield: jnp.ndarray, kk: jnp.ndarray) -> jnp.ndarray:
    """
        Interpolate the power spectrum values.

        Parameters
        ----------
        kfield : jnp.ndarray
            Fourier-transformed field.
        kk : jnp.ndarray
            Wave numbers.

        Returns
        -------
        jnp.ndarray
            Interpolated delta_k field.
        """
    k = jnp.logspace(-4, 2, 256)
    pk = jc.power.linear_matter_power(jc.Planck15(), k)
    pk = pk * (mesh_shape[0] / box_size[0]) * (mesh_shape[1] / box_size[1]) * (
        mesh_shape[2] / box_size[2])
    delta_k = kfield * jc.scipy.interpolate.interp(kk.flatten(), k, pk**
                                                   0.5).reshape(kfield.shape)
    return delta_k

  @jax.jit
  def forward_fn(z: jnp.ndarray, kvec: list,
                 a: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
        Perform the forward simulation step.

        Parameters
        ----------
        z : jnp.ndarray
            Initial field.
            global array with shape (x, y, z)
        kvec : list
            Fourier transform wave numbers.
            list of 3 arrays with shape (x, 1, 1), (1, y, 1), (1, 1, z) with only the third fully replicated
        a : float
            Scale factor.

        Returns
        -------
        tuple
            Tuple containing initial conditions and the field.
        """
    kfield = jaxdecomp.fft.pfft3d(z.astype(jnp.complex64))

    ky, kz, kx = kvec
    kk = jnp.sqrt((kx / box_size[0] * mesh_shape[0])**2 +
                  (ky / box_size[1] * mesh_shape[1])**2 +
                  (kz / box_size[1] * mesh_shape[1])**2)

    delta_k = interpolate(kfield, kk)

    # Inverse Fourier transform to generate the initial conditions
    initial_conditions = jaxdecomp.fft.pifft3d(delta_k).real

    # Compute LPT displacement
    cosmo = jc.Planck15()
    a = jnp.atleast_1d(a)

    kernel_lap = jnp.where(
        kk == 0, 1., 1. / (kx**2 + ky**2 + kz**2))  # Laplace kernel + longrange

    pot_k = delta_k * kernel_lap
    forces_k = jnp.stack([
        pot_k * 1j / 6.0 *
        (8 * jnp.sin(kx) - jnp.sin(2 * kx)), pot_k * 1j / 6.0 *
        (8 * jnp.sin(ky) - jnp.sin(2 * ky)), pot_k * 1j / 6.0 *
        (8 * jnp.sin(kz) - jnp.sin(2 * kz))
    ],
                         axis=-1)

    init_force = jnp.stack(
        [jaxdecomp.fft.pifft3d(forces_k[..., i]).real for i in range(3)],
        axis=-1)
    dx = growth_factor(cosmo, a) * init_force

    field = cic_paint(dx)
    return initial_conditions, field

  with mesh:
    jit_start = time.perf_counter()
    initial_conds, field = forward_fn(z, kvec, a=1.0)
    field.block_until_ready()
    jit_end = time.perf_counter()

    print(f"JIT done in {jit_end - jit_start}")

    start = time.perf_counter()
    initial_conds, field = forward_fn(z, kvec, a=1.0)
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

  print(f"Finished saving to {output_dir}")

jax.distributed.shutdown()
