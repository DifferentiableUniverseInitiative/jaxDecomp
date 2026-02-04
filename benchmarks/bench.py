import argparse
import os
import jax
import jax.numpy as jnp
import jaxdecomp
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax_hpc_profiler import Timer

# Initialize distributed context immediately
jax.distributed.initialize()

def run_benchmark(pdims, global_shape, backend, nb_nodes, precision, iterations, trace, contiguous, output_path):
    pfft_backend = backend
    if backend == 'cudecomp':
        jaxdecomp.config.update('transpose_comm_backend', jaxdecomp.TRANSPOSE_COMM_NCCL)
    
    rank = jax.process_index()
    
    if contiguous:
        jaxdecomp.config.update('transpose_axis_contiguous', contiguous)

    # create_device_mesh(pdims) expects a tuple/list of dimensions.
    # jaxfft.py used devices.T and axis_names=('z', 'y')
    devices = mesh_utils.create_device_mesh(pdims)
    mesh = Mesh(devices.T, axis_names=('z', 'y'))
    
    # Calculate local shape based on decomposition
    # Based on jaxfft.py logic:
    # array shape = [global_shape[0] // pdims[0], global_shape[1] // pdims[1], global_shape[2]]
    local_shape_0 = global_shape[0] // pdims[0]
    local_shape_1 = global_shape[1] // pdims[1]
    local_shape_2 = global_shape[2]
    
    # Initialize local array
    array = jax.random.normal(
        shape=[local_shape_0, local_shape_1, local_shape_2], 
        key=jax.random.PRNGKey(rank)
    )

    # Remap to global array
    # P('z', 'y') means axis 0 is sharded along mesh axis 'z', axis 1 along 'y'.
    global_array = multihost_utils.host_local_array_to_global_array(array, mesh, P('z', 'y'))

    if jax.process_index() == 0:
        print(f'Devices {jax.devices()}')
        print(f'Global dims {global_shape}, pdims {pdims}, Backend {backend}')
        print(f'Local array shape {array.shape}')
        print('Sharding:')
        print(global_array.sharding)

    @jax.jit
    def do_fft(x):
        return jaxdecomp.fft.pfft3d(x, backend=pfft_backend)

    @jax.jit
    def do_ifft(x):
        return jaxdecomp.fft.pifft3d(x, backend=pfft_backend)

    fft_chrono = Timer(save_jaxpr=False)
    ifft_chrono = Timer(save_jaxpr=False)

    # Warmup
    global_array = fft_chrono.chrono_jit(do_fft, global_array)
    global_array = ifft_chrono.chrono_jit(do_ifft, global_array)
    multihost_utils.sync_global_devices('warmup')
    
    # Benchmark loop
    for i in range(iterations):
        global_array = fft_chrono.chrono_fun(do_fft, global_array)
        global_array = ifft_chrono.chrono_fun(do_ifft, global_array)

    cont_str = '-cont' if contiguous else '-noncont'
    metadata = {
        'precision': precision,
        'x': str(global_shape[0]),
        'y': str(global_shape[1]),
        'z': str(global_shape[2]),
        'px': str(pdims[0]),
        'py': str(pdims[1]),
        'backend': backend,
        'nodes': str(nb_nodes),
    }

    fft_chrono.report(f'{output_path}/jaxfft.csv', function=f'FFT{cont_str}', **metadata)
    ifft_chrono.report(f'{output_path}/jaxfft.csv', function=f'IFFT{cont_str}', **metadata)
    print('Done')

def main():
    parser = argparse.ArgumentParser(description='JAXDecomp Benchmark')
    parser.add_argument('--pdims', nargs=2, type=int, required=True, help='Process mesh dimensions (e.g. 2 2)')
    parser.add_argument('--global_shape', nargs=3, type=int, help='Global array shape (e.g. 512 512 512)')
    parser.add_argument('--local_shape', nargs=3, type=int, help='Local array shape (e.g. 256 256 512)')
    parser.add_argument('-n', '--nb_nodes', type=int, default=1, help='Number of nodes')
    parser.add_argument('-o', '--output_path', type=str, default='.', help='Output directory path')
    parser.add_argument('-pr', '--precision', type=str, default='float32', choices=['float32', 'float64'], help='Precision')
    parser.add_argument('-i', '--iterations', type=int, default=10, help='Number of iterations')
    parser.add_argument('-c', '--contiguous', action='store_true', help='Use contiguous axes')
    parser.add_argument('-t', '--trace', action='store_true', help='Enable tracing')
    parser.add_argument('-b', '--backend', type=str, required=True, choices=['jax', 'cudecomp'], help='FFT Backend')

    args = parser.parse_args()

    if args.local_shape is not None and args.global_shape is not None:
        print('Please provide either local_shape or global_shape, not both.')
        exit(1)

    pdims = tuple(args.pdims)

    if args.local_shape is not None:
        # Reconstruct global shape from local shape and pdims
        # Assuming decomposition matches pdims[0] for axis 0 and pdims[1] for axis 1
        global_shape = (
            args.local_shape[0] * pdims[0],
            args.local_shape[1] * pdims[1],
            args.local_shape[2]
        )
    elif args.global_shape is not None:
        global_shape = tuple(args.global_shape)
    else:
        print('Please provide either local_shape or global_shape.')
        exit(1)

    if args.precision == 'float32':
        jax.config.update('jax_enable_x64', False)
    else:
        jax.config.update('jax_enable_x64', True)

    os.makedirs(args.output_path, exist_ok=True)
    
    # Validation
    for dim, pdim in zip(global_shape[:2], pdims):
        if dim % pdim != 0:
            print(f'Global shape {global_shape} is not divisible by pdims {pdims}')
            exit(1)

    run_benchmark(
        pdims, 
        global_shape, 
        args.backend, 
        args.nb_nodes, 
        args.precision, 
        args.iterations, 
        args.trace, 
        args.contiguous, 
        args.output_path
    )

if __name__ == '__main__':
    main()
