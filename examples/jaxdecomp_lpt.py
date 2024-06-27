import diffrax
import jax
from diffrax import ODETerm, PIDController, SaveAt, diffeqsolve
from jax.experimental.ode import odeint

import jaxdecomp

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()

import argparse
import os
import time
from functools import partial

import jax.lax as lax
import jax.numpy as jnp
import jax_cosmo as jc
import jaxpm as jaxpm
import numpy as np
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
# Defining painting function "borrowed" from pmwd
from jax.lax import scan
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jaxpm.growth import growth_factor


def _chunk_split(ptcl_num, chunk_size, *arrays):
  """Split and reshape particle arrays into chunks and remainders, with the remainders
    preceding the chunks. 0D ones are duplicated as full arrays in the chunks."""
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


def enmesh(i1, d1, a1, s1, b12, a2, s2):
  """Multilinear enmeshing."""
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


def _scatter_chunk(carry, chunk):
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


def scatter(pmid, disp, mesh, chunk_size=2**24, val=1., offset=0, cell_size=1.):

  ptcl_num, spatial_ndim = pmid.shape
  val = jnp.asarray(val)
  mesh = jnp.asarray(mesh)

  remainder, chunks = _chunk_split(ptcl_num, chunk_size, pmid, disp, val)
  carry = mesh, offset, cell_size
  if remainder is not None:
    carry = _scatter_chunk(carry, remainder)[0]
  carry = scan(_scatter_chunk, carry, chunks)[0]
  mesh = carry[0]
  return mesh


def lpt2_source(lineark_laplace, kvec):
  ky, kz, kx = kvec
  source = jnp.zeros_like(lineark_laplace)

  D1 = [1, 2, 0]
  D2 = [2, 0, 1]
  grad_kernels = [
      1j / 6.0 * (8 * jnp.sin(kx) - jnp.sin(2 * kx)),
      1j / 6.0 * (8 * jnp.sin(ky) - jnp.sin(2 * ky)),
      1j / 6.0 * (8 * jnp.sin(kz) - jnp.sin(2 * kz))
  ]
  phi_ii = [
      jaxdecomp.fft.pifft3d(lineark_laplace * grad_kernels[i]**2).real
      for i in range(3)
  ]

  # Diagonal terms
  for d in range(3):
    source += phi_ii[D1[d]] * phi_ii[D2[d]]

  # off-diag terms
  for d in range(3):
    gradi = grad_kernels[D1[d]]
    gradj = grad_kernels[D2[d]]
    phi = jaxdecomp.fft.pifft3d(lineark_laplace * gradi * gradj).real
    source -= phi**2

  return jaxdecomp.fft.pfft3d(source * 3. / 7.)


if __name__ == '__main__':

  jax.config.update('jax_enable_x64', False)

  parser = argparse.ArgumentParser()

  parser.add_argument('-s', '--size', type=int, default=64)
  parser.add_argument('-p', '--pdims', type=str, default='1x1')
  parser.add_argument('-b', '--box_size', type=int, default=200)
  parser.add_argument('-hs', '--halo_size', type=int, default=0)
  parser.add_argument('-o', '--output', type=str, default='out')

  args = parser.parse_args()

  # Setup random keys
  master_key = jax.random.PRNGKey(42)
  key = jax.random.split(master_key, size)[rank]
  # Rea
  pdims = tuple(map(int, args.pdims.split('x')))
  mesh_shape = [args.size, args.size, args.size]
  box_size = [float(args.box_size), float(args.box_size), float(args.box_size)]
  halo_size = args.halo_size

  lpt2 = args.lpt2
  ode_str = args.ode
  output_dir = f"{args.output}"
  # Create output directory recursively
  os.makedirs(output_dir, exist_ok=True)
  snapshots = jnp.linspace(0.1, 1.0, 2)
  # Create computing mesgh
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices, axis_names=('y', 'z'))
  sharding = jax.sharding.NamedSharding(mesh, P('z', 'y'))

  ### Create all initial distributed tensors ###
  local_mesh_shape = [
      mesh_shape[0] // pdims[1], mesh_shape[1] // pdims[0], mesh_shape[2]
  ]

  # Create gaussian field distributed across the mesh
  z = jax.make_array_from_single_device_arrays(
      shape=mesh_shape,
      sharding=sharding,
      arrays=[jax.random.normal(key, local_mesh_shape, dtype='float32')])

  kd = np.fft.fftfreq(mesh_shape[0]).astype('float32') * 2 * np.pi
  kvec = [
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

  @partial(shard_map, mesh=mesh, in_specs=(P('z', 'y'),), out_specs=P('z', 'y'))
  def cic_paint_sharded(displacement):
    original_shape = displacement.shape

    mesh = jnp.zeros(original_shape[:-1], dtype='float32')

    # Padding the output array along the two first dimensions
    mesh = jnp.pad(mesh,
                   [[halo_size, halo_size], [halo_size, halo_size], [0, 0]])

    a, b, c = jnp.meshgrid(
        jnp.arange(local_mesh_shape[1]), jnp.arange(local_mesh_shape[0]),
        jnp.arange(local_mesh_shape[2]))
    # adding an offset of size halo size
    pmid = jnp.stack([b + halo_size, a + halo_size, c], axis=-1)
    pmid = pmid.reshape([-1, 3])
    return scatter(pmid, displacement.reshape([-1, 3]), mesh)

  @jax.jit
  def cic_paint(displacement):

    print(f"Painting displacement {displacement.shape}")

    field = cic_paint_sharded(displacement)

    if use_halo_exchange:
      field = jaxdecomp.halo_exchange(
          field,
          halo_extents=(halo_size // 2, halo_size // 2, 0),
          halo_periods=(True, True, True),
          reduce_halo=False)
      # Removing the padding
      field = unpad(field)

    return field

  def make_ode_fn(mesh_shape):

    def nbody_ode(state, a, args):
      """
            state is a tuple (position, velocities)
            """
      pos, vel = state
      cosmo, kvec = args

      ky, kz, kx = kvec
      kk = jnp.sqrt((kx / box_size[0] * mesh_shape[0])**2 +
                    (ky / box_size[1] * mesh_shape[1])**2 +
                    (kz / box_size[1] * mesh_shape[1])**2)

      mesh = cic_paint(pos)

      delta_k = jaxdecomp.fft.pfft3d(mesh)

      kernel_lap = jnp.where(kk == 0, 1.,
                             1. / (kx**2 + ky**2 + kz**2))  # Laplace kernel
      pot_k = delta_k * kernel_lap

      # Forces have to be a Z pencil because they are going to be IFFT back to X pencil
      forces_k = jnp.stack([
          pot_k * 1j / 6.0 *
          (8 * jnp.sin(kx) - jnp.sin(2 * kx)), pot_k * 1j / 6.0 *
          (8 * jnp.sin(ky) - jnp.sin(2 * ky)), pot_k * 1j / 6.0 *
          (8 * jnp.sin(kz) - jnp.sin(2 * kz))
      ],
                           axis=-1)
      forces = jnp.stack(
          [jaxdecomp.fft.pifft3d(forces_k[..., i]).real for i in range(3)],
          axis=-1)

      # Computes the update of position (drift)
      dpos = 1. / (a**3 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * vel

      # Computes the update of velocity (kick)
      dvel = 1. / (a**2 * jnp.sqrt(jc.background.Esqr(cosmo, a))) * forces

      return jnp.stack([dpos, dvel], axis=0)

    return nbody_ode

  @partial(shard_map, mesh=mesh, in_specs=(P('z', 'y')), out_specs=P('z', 'y'))
  def unpad(x):
    x = x.at[halo_size:halo_size + halo_size // 2].add(x[:halo_size // 2])
    x = x.at[-(halo_size + halo_size // 2):-halo_size].add(x[-halo_size // 2:])
    x = x.at[:, halo_size:halo_size + halo_size // 2].add(x[:, :halo_size // 2])
    x = x.at[:,
             -(halo_size + halo_size // 2):-halo_size].add(x[:,
                                                             -halo_size // 2:])
    return x[halo_size:-halo_size, halo_size:-halo_size, :]

  @partial(
      shard_map,
      mesh=mesh,
      in_specs=(P('z', 'y'), P('z', 'y')),
      out_specs=P('z', 'y'))
  def interpolate(kfield, kk):

    k = jnp.logspace(-4, 2, 256)  # I don't understand why 256?
    pk = jc.power.linear_matter_power(jc.Planck15(), k)
    pk = pk * (mesh_shape[0] / box_size[0]) * (mesh_shape[1] / box_size[1]) * (
        mesh_shape[2] / box_size[2])
    delta_k = kfield * jc.scipy.interpolate.interp(kk.flatten(), k, pk**
                                                   0.5).reshape(kfield.shape)

    return delta_k

  @jax.jit
  def forward_fn(z, kvec, a):
    kfield = jaxdecomp.fft.pfft3d(z.astype(jnp.complex64))

    ky, kz, kx = kvec
    kk = jnp.sqrt((kx / box_size[0] * mesh_shape[0])**2 +
                  (ky / box_size[1] * mesh_shape[1])**2 +
                  (kz / box_size[1] * mesh_shape[1])**2)

    delta_k = interpolate(kfield, kk)

    # Inverse fourier transform to generate the initial conditions
    initial_conditions = jaxdecomp.fft.pifft3d(delta_k).real

    ###  Compute LPT displacement
    cosmo = jc.Planck15()
    a = jnp.atleast_1d(a)

    kernel_lap = jnp.where(
        kk == 0, 1., 1. / (kx**2 + ky**2 + kz**2))  # Laplace kernel + longrange
    pot_k = delta_k * kernel_lap
    # Forces have to be a Z pencil because they are going to be IFFT back to X pencil
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

    p = a**2 * growth_factor(cosmo, a) * jnp.sqrt(jc.background.Esqr(cosmo,
                                                                     a)) * dx
    if lpt2:
      # Add second order lpt
      source_lpt2k = lpt2_source(pot_k, kvec)
      pot_k2 = source_lpt2k * kernel_lap

      forces_k2 = jnp.stack([
          pot_k2 * 1j / 6.0 *
          (8 * jnp.sin(kx) - jnp.sin(2 * kx)), pot_k2 * 1j / 6.0 *
          (8 * jnp.sin(ky) - jnp.sin(2 * ky)), pot_k2 * 1j / 6.0 *
          (8 * jnp.sin(kz) - jnp.sin(2 * kz))
      ],
                            axis=-1)
      # TODO: Use second order grotwh here, this will only work for a=1.
      dx2 = jnp.stack(
          [jaxdecomp.fft.pifft3d(forces_k2[..., i]).real for i in range(3)],
          axis=-1)
      dx += dx2

    field = cic_paint(dx)

    if args.ode == "lpt":
      return initial_conditions, field, None

    elif args.ode == "jax":

      ode_fn = jaxpm.pm.make_ode_fn(mesh_shape)
      res = odeint(
          ode_fn, [dx, p], jnp.array([0.1, 1.]), cosmo, rtol=1e-3, atol=1e-3)

      final_particles = [ode_field for ode_field in res[0]]

      return initial_conditions, field, final_particles

    elif args.ode == "diffrax":

      ode_fn = make_ode_fn(mesh_shape)

      term = ODETerm(lambda t, state, args: ode_fn(state, t, args))
      solver = diffrax.Euler()

      stepsize_controller = diffrax.ConstantStepSize()
      res = diffeqsolve(
          term,
          solver,
          t0=0.1,
          t1=1.,
          dt0=0.01,
          y0=jnp.stack([dx, p], axis=0),
          args=(cosmo, kvec),
          stepsize_controller=stepsize_controller)

      final_fields = [ode_field[0] for ode_field in res.ys]

      return initial_conditions, field, final_fields

    elif args.ode == "manual":
      ode_fn = make_ode_fn(mesh_shape)

      state = jnp.stack([dx, p], axis=0)
      final_fields = []
      min_step = 0.01
      max_step = 1.
      nb_steps = 10
      steps = jnp.linspace(min_step, max_step, nb_steps)
      snapshots = jnp.linspace(0.1, 1.0, 10)
      step_size = steps[1] - steps[0]
      for step in steps:
        state = state + ode_fn(state, step, [cosmo, kvec]) * step_size
        final_fields.append(state[0])
        print(f"Doing step {step}")

      return initial_conditions, field, final_fields

  with mesh:
    jit_start = time.perf_counter()
    initial_conds, field, final_fields = forward_fn(z, kvec, a=1.)
    field.block_until_ready()
    jit_end = time.perf_counter()
    # del initial_conds, field, final_fields, res

    start = time.perf_counter()
    initial_conds, field, final_fields = forward_fn(z, kvec, a=1.)
    field.block_until_ready()
    end = time.perf_counter()

    # hlo = jax.jit(forward_fn).lower(z, kvec, a=1.).compile().runtime_executable().hlo_modules()[0].to_string()
    # print(hlo)

  with open(f"{output_dir}/log_{rank}.log", 'w') as log_file:
    log_file.write(f"JIT time: {jit_end - jit_start}\n")
    log_file.write(f"Execution time: {end - start}\n")
    # log_file.write("Solver stat")
    # print(res.stats, file=log_file)
  # Saving results
  np.save(f'{output_dir}/initial_conditions_{rank}.npy',
          initial_conds.addressable_data(0))
  np.save(f'{output_dir}/field_{rank}.npy', field.addressable_data(0))

  print(f"final_fields {final_fields}")
  if final_fields is not None:
    for i, final_field in enumerate(final_fields):

      with mesh:
        final_field_mesh = cic_paint(final_field)
      np.save(f'{output_dir}/ode_solution_{i}_{rank}.npy',
              final_field_mesh.addressable_data(0))
      np.save(f'{output_dir}/final_field_{i}_{rank}.npy',
              final_field_mesh.addressable_data(0))

  print(f"Finished saved to {output_dir}")

# jaxdecomp.finalize()
jax.distributed.shutdown()
