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
from utils import *

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

  ### Create all initial distributed tensors ###
  local_mesh_shape = [
      mesh_shape[0] // pdims[1], mesh_shape[1] // pdims[0], mesh_shape[2]
  ]
  # Correction for positions relative to the local slice
  correct_y = -local_mesh_shape[1] * (rank // pdims[0])
  correct_z = -local_mesh_shape[0] * (rank % pdims[0])

  # Create gaussian field distributed across the mesh
  z = generate_random_field(mesh_shape, sharding, key, local_mesh_shape)

  kvec = fttk(mesh_shape, mesh)
  pos = generate_initial_positions(mesh_shape, sharding)
  painting_mesh = jnp.zeros_like(z, device=sharding)

  print(f"Local mesh shape {local_mesh_shape}")
  print(f"Saving on folder {output_dir}")
  print(f"Created initial field {z.shape} and sharding {z.sharding}")
  print(f"Created painting mesh with shape {painting_mesh.shape}")
  print(f"And sharding  {painting_mesh.sharding}")
  print(f"Created positions {pos.shape} and sharding {pos.sharding}")
  print("Corrected positions for rank {rank}  --> ")
  print(f" \tare Y: {correct_y} Z: {correct_z}")
  print(
      f"pos shape {pos.shape} pos sharding = {pos.sharding} shape of local {pos.addressable_data(0).shape}"
  )

  @partial(
      shard_map,
      mesh=mesh,
      in_specs=(P('z', 'y'), P('z', 'y')),
      out_specs=P('z', 'y'))
  def cic_paint_sharded(mesh: jnp.ndarray,
                        positions: jnp.ndarray) -> jnp.ndarray:
    """
        Distributed part of the CIC painting f

        Parameters
        ----------
        mesh : jnp.ndarray with shape ( X, Y, Z)
            The mesh onto which mass is painted.
        positions : jnp.ndarray with shape (X , Y, Z, 3)
            Positions of particles.

        Returns
        -------
        jnp.ndarray
            The mesh with painted mass.
        """
    # Get positions relative to the start of each slice
    positions = positions.at[:, :, :, 1].add(correct_y)
    positions = positions.at[:, :, :, 0].add(correct_z)
    positions = positions.reshape([-1, 3])

    mesh = jnp.pad(mesh, [(halo_size, halo_size), (halo_size, halo_size),
                          (0, 0)])
    positions += jnp.array([halo_size, halo_size, 0]).reshape([-1, 3])

    positions = jnp.expand_dims(positions, 1)
    floor = jnp.floor(positions)

    connection = jnp.array([[[0, 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1],
                             [1., 1, 0], [1., 0, 1], [0., 1, 1], [1., 1, 1]]])

    neighboor_coords = floor + connection
    kernel = 1. - jnp.abs(positions - neighboor_coords)
    kernel = kernel[..., 0] * kernel[..., 1] * kernel[..., 2]

    neighboor_coords_mod = jnp.mod(
        neighboor_coords.reshape([-1, 8, 3]).astype('int32'),
        jnp.array(mesh.shape))

    dnums = jax.lax.ScatterDimensionNumbers(
        update_window_dims=(),
        inserted_window_dims=(0, 1, 2),
        scatter_dims_to_operand_dims=(0, 1, 2))
    mesh = lax.scatter_add(mesh, neighboor_coords_mod, kernel.reshape([-1, 8]),
                           dnums)

    return mesh

  @jax.jit
  def cic_paint(mesh: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """
        Wrapper function to paint mass onto a mesh using CIC method and
        perform halo exchange.

        Parameters
        ----------
        mesh : jnp.ndarray with shape ( X, Y, Z)
            The mesh onto which mass is painted.
        positions : jnp.ndarray with shape (X , Y, Z, 3)
            Positions of particles.

        Returns
        -------
        jnp.ndarray
            The mesh with painted mass after halo exchange.
        """
    field = cic_paint_sharded(mesh, positions)

    field = jaxdecomp.halo_exchange(
        field,
        halo_extents=(halo_size // 2, halo_size // 2, 0),
        halo_periods=(True, True, True),
        reduce_halo=True)
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
        Interpolates the power spectrum onto the k-space field.

        Parameters
        ----------
        kfield : jnp.ndarray with shape ( X, Y, Z)
            The k-space field.
        kk : jnp.ndarray
            Magnitude of k-vectors.

        Returns
        -------
        jnp.ndarray with shape ( X, Y, Z)
            The interpolated k-space field.
        """
    k = jnp.logspace(-4, 2, 256)
    pk = jc.power.linear_matter_power(jc.Planck15(), k)
    pk = pk * (mesh_shape[0] / box_size[0]) * (mesh_shape[1] / box_size[1]) * (
        mesh_shape[2] / box_size[2])
    delta_k = kfield * jc.scipy.interpolate.interp(kk.flatten(), k, pk**
                                                   0.5).reshape(kfield.shape)

    return delta_k

  @jax.jit
  def forward_fn(z: jnp.ndarray, kvec: tuple, pos: jnp.ndarray,
                 painting_mesh: jnp.ndarray, a: float) -> tuple:
    """
        Computes initial conditions and density field using Lagrangian perturbation theory (LPT).

        Parameters
        ----------
        z : jnp.ndarray
            The initial Gaussian random field.
        kvec : tuple
            K-vectors for Fourier Transform.
        pos : jnp.ndarray
            Initial positions of particles.
        painting_mesh : jnp.ndarray
            The mesh for mass painting.
        a : float
            Scale factor.

        Returns
        -------
        tuple
            Initial conditions and the density field.
        """
    kfield = jaxdecomp.fft.pfft3d(z.astype(jnp.complex64))

    ky, kz, kx = kvec
    kk = jnp.sqrt((kx / box_size[0] * mesh_shape[0])**2 +
                  (ky / box_size[1] * mesh_shape[1])**2 +
                  (kz / box_size[1] * mesh_shape[1])**2)

    delta_k = interpolate(kfield, kk)

    # Inverse Fourier transform to generate the initial conditions
    initial_conditions = jaxdecomp.fft.pifft3d(delta_k).real

    ###  Compute LPT displacement
    cosmo = jc.Planck15()
    a = jnp.atleast_1d(a)

    kernel_lap = jnp.where(kk == 0, 1., 1. / -(kx**2 + ky**2 + kz**2))

    pot_k = delta_k * kernel_lap
    # Forces have to be a Z pencil because they are going to be IFFT back to X pencil
    forces_k = -jnp.stack([
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

    p = a**2 * growth_rate(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo,
                                                                   a)) * dx
    f = a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a)) * dGfa(cosmo,
                                                             a) * init_force

    field = cic_paint(painting_mesh, (pos + dx))

    return initial_conditions, field

  with mesh:
    jit_start = time.perf_counter()
    initial_conds, field = forward_fn(z, kvec, pos, painting_mesh, a=1.)
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
