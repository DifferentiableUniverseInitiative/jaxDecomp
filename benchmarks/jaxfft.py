import jax

jax.distributed.initialize()
rank = jax.process_index()
size = jax.process_count()
import argparse
import re
import time
from functools import partial

import jax.numpy as jnp
import jaxdecomp
import numpy as np
from jax.experimental import mesh_utils, multihost_utils
from jax.experimental.multihost_utils import sync_global_devices
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax_hpc_profiler import Timer


def run_benchmark(pdims, global_shape, backend, nb_nodes, precision, iterations, trace,contiguous,
                  output_path):

  pfft_backend = "cudecomp"
  if backend == "NCCL":
    jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_NCCL)
  elif backend == "NCCL_PL":
    jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_NCCL_PL)
  elif backend == "MPI_P2P":
    jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_P2P)
  elif backend == "MPI":
    jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_MPI_A2A)
  else:
    pfft_backend = "jax"
  # Initialize the local slice with the local slice shape
  array = jax.random.normal(
      shape=[global_shape[0] // pdims[0], global_shape[1] // pdims[1], global_shape[2]],
      key=jax.random.PRNGKey(rank))

  if contiguous:
    jaxdecomp.config.update('transpose_axis_contiguous', contiguous)

  # Remap to the global array from the local slice
  devices = mesh_utils.create_device_mesh(pdims)
  mesh = Mesh(devices.T, axis_names=('z', 'y'))
  global_array = multihost_utils.host_local_array_to_global_array(array, mesh, P('z', 'y'))

  if jax.process_index() == 0:
    print(f"Devices {jax.devices()}")
    print(
        f"Global dims {global_shape}, pdims {pdims} , Backend {backend} local array shape {array.shape} global array shape {global_array.shape}"
    )
    print("Sharding :")
    print(global_array.sharding)

  @jax.jit
  def do_fft(x):
    return jaxdecomp.fft.pfft3d(x , backend=pfft_backend)

  @jax.jit
  def do_ifft(x):
    return jaxdecomp.fft.pifft3d(x, backend=pfft_backend)

  @jax.jit
  def get_diff(arr1, arr2):
    return jnp.abs(arr1 - arr2).max()

  fft_chrono = Timer(save_jaxpr=False)
  ifft_chrono = Timer(save_jaxpr=False)


  # Warm start
  global_array = fft_chrono.chrono_jit(do_fft, global_array)
  global_array = ifft_chrono.chrono_jit(do_ifft, global_array)
  sync_global_devices("warmup")
  for i in range(iterations):
    global_array = fft_chrono.chrono_fun(do_fft, global_array)
    global_array = ifft_chrono.chrono_fun(do_ifft, global_array)
  
  cont_str = "-cont" if contiguous else "-noncont"
  fft_metadata = {
      'function': f"FFT{cont_str}",
      'precision': precision,
      'x': str(global_shape[0]),
      'px': str(pdims[0]),
      'py': str(pdims[1]),
      'backend': backend,
      'nodes': str(nb_nodes),
  }
  ifft_metadata = {
      'function': f"IFFT{cont_str}",
      'precision': precision,
      'x': str(global_shape[0]),
      'px': str(pdims[0]),
      'py': str(pdims[1]),
      'backend': backend,
      'nodes': str(nb_nodes),
  }

  fft_chrono.report(f"{output_path}/jaxfft.csv", **fft_metadata)
  ifft_chrono.report(f"{output_path}/jaxfft.csv", **ifft_metadata)
  print(f"Done")


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description='NBody Benchmark')
  parser.add_argument('-p', '--pdims', type=str, help='GPU grid', required=True)
  parser.add_argument('-l', '--local_shape', type=int, help='Local shape', default=None)
  parser.add_argument(
      '-g', '--global_shape', type=int, help='Global shape of the array', default=None)
  parser.add_argument(
      '-b',
      '--backend',
      type=str,
      help='Backend to use for transpose comm',
      choices=["NCCL", "NCCL_PL", "MPI_P2P", "MPI"],
      default="NCCL")
  parser.add_argument('-n', '--nb_nodes', type=int, help='Number of nodes', default=1)
  parser.add_argument('-o', '--output_path', type=str, help='Output path', default=".")
  parser.add_argument('-pr', '--precision', type=str, help='Precision', default="float32")
  parser.add_argument('-i', '--iterations', type=int, help='Number of iterations', default=10)
  parser.add_argument('-c', '--contiguous', type=bool, help='Contiguous', default=True)
  parser.add_argument('-t', '--trace', action='store_true', help='Profile using tensorboard')

  args = parser.parse_args()

  if args.local_shape is not None and args.global_shape is not None:
    print("Please provide either local_shape or global_shape")
    parser.print_help()
    exit(0)

  if args.local_shape is not None:
    global_shape = (args.local_shape * jax.device_count(), args.local_shape * jax.device_count(),
                    args.local_shape * jax.device_count())
  elif args.global_shape is not None:
    global_shape = (args.global_shape, args.global_shape, args.global_shape)
  else:
    print("Please provide either local_shape or global_shape")
    parser.print_help()
    exit(0)

  if args.precision == "float32":
    jax.config.update("jax_enable_x64", False)
  elif args.precision == "float64":
    jax.config.update("jax_enable_x64", True)
  else:
    print("Precision should be either float32 or float64")
    parser.print_help()
    exit(0)

  pdims = tuple(map(int, args.pdims.split("x")))

  backend = args.backend
  nb_nodes = args.nb_nodes
  output_path = args.output_path
  import os
  os.makedirs(output_path, exist_ok=True)

  for dim in global_shape:
    for pdim in pdims:
      if dim % pdim != 0:
        print(f"Global shape {global_shape} is not divisible by pdims {pdims}")
        exit(0)
        # Do not raise error for slurm jobs
        # raise ValueError(f"Global shape {global_shape} is not divisible by pdims {pdims}")

  run_benchmark(pdims, global_shape, backend, nb_nodes, args.precision, args.iterations, args.trace,args.contiguous,
                output_path)

jaxdecomp.finalize()
jax.distributed.shutdown()