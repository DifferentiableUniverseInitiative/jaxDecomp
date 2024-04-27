from mpi4py import MPI
from functools import partial

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import jax
jax.config.update("jax_enable_x64", True)
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from math import prod
from numpy.testing import assert_array_equal
import pytest
# Initialize jax distributed to instruct jax local process which GPU to use
jax.distributed.initialize()
# Initialize cuDecomp


############################################################################################################
# This test is just to make sure that multihost_utils.process_allgather works as expected
############################################################################################################

# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

    assert(len(global_shape) == 3)
    assert(len(pdims) == 2)
    assert(prod(pdims) == size) , "The product of pdims must be equal to the number of MPI processes"

    local_array = jax.random.normal(
        shape=[
            global_shape[0] // pdims[1], global_shape[1] // pdims[0]
        ],
        key=jax.random.PRNGKey(rank))
    # Remap to the global array from the local slice
    devices = mesh_utils.create_device_mesh(pdims[::-1])
    mesh = Mesh(devices, axis_names=('z', 'y'))
    global_array = multihost_utils.host_local_array_to_global_array(
        local_array, mesh, P('z', 'y'))
    
    return global_array, mesh
  

@pytest.mark.parametrize("pdims", [(1, size), (size, 1) , (size // 2,size // 2)]) # Slabs and Pencils
def test_empty_halo(pdims):

    pdims = (2 , 2)
    global_shape =  (29 * size, 19 * size, 17 * size)  # These sizes are prime numbers x size of the pmesh

    global_array, mesh = create_spmd_array(global_shape, pdims)

    # All gather function
    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P() , check_rep=False)
    def sharded_allgather(arr):
        gathered_z_axis = jax.lax.all_gather(arr, axis_name='z', axis=0 , tiled=True)
        gathered = jax.lax.all_gather(gathered_z_axis, axis_name='y', axis=1 , tiled=True)
        return gathered


    gathered = sharded_allgather(global_array)
    process_allgather = multihost_utils.process_allgather(global_array , tiled=True)

    print(f"Shape of original array {global_array.shape}")
    print(f"Shape of gathered array using double lax gather {gathered.shape}")
    print(f"Shape of gathered array using process_allgather {process_allgather.shape}")

    assert_array_equal(gathered, process_allgather)

    
