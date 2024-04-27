import pytest
from functools import partial


from math import prod
import jax
jax.config.update("jax_enable_x64", True)
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P
from numpy.testing import assert_array_equal
import jax.numpy as jnp
import jaxdecomp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax._src.distributed import global_state  # This may break in the future
from jaxdecomp._src.padding import slice_pad, slice_unpad
# Initialize jax distributed to instruct jax local process which GPU to use
jax.distributed.initialize()
rank = global_state.process_id
size = global_state.num_processes
# Initialize cuDecomp

# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, pdims):

    assert(len(global_shape) == 3)
    assert(len(pdims) == 2)
    assert(prod(pdims) == size) , "The product of pdims must be equal to the number of MPI processes"

    local_array = jax.random.normal(
        shape=[
            global_shape[0] // pdims[1], global_shape[1] // pdims[0],
            global_shape[2]
        ],
        key=jax.random.PRNGKey(rank))
    # Remap to the global array from the local slice
    devices = mesh_utils.create_device_mesh(pdims[::-1])
    mesh = Mesh(devices, axis_names=('z', 'y'))
    global_array = multihost_utils.host_local_array_to_global_array(
        local_array, mesh, P('z', 'y'))
    
    return global_array, mesh

pencil_1 = (size // 2, size // (size // 2))
pencil_2 = (size // (size // 2), size // 2)

@pytest.mark.parametrize("pdims", [(1, size), (size, 1) , pencil_1 , pencil_2]) # Test with Slab and Pencil decompositions
def test_padding(pdims):

    print("*"*80)
    print(f"Testing with pdims {pdims}")

    global_shape = (29 * size, 19 * size, 17 * size)  # These sizes are prime numbers x size of the pmesh

    global_array, mesh = create_spmd_array(global_shape, pdims)

    padding = ((32 , 32),( 32 , 32), (0 , 0 ))

    # Reference implementation of per shard slicing
    @partial(shard_map, mesh=mesh, in_specs=(P('z', 'y'),P()),
            out_specs=P('z', 'y'))
    def sharded_pad(arr , padding):
        padded =  jnp.pad(arr,pad_width=padding)
        return padded

    @partial(shard_map, mesh=mesh, in_specs=(P('z', 'y')),
            out_specs=P('z', 'y'))
    def sharded_unpad(arr):
        x_unpading , y_unpading , z_unpading = padding[0] , padding[1] , padding[2]
        first_x, first_y, first_z = -x_unpading[0] , -y_unpading[0] , -z_unpading[0]
        last_x, last_y, last_z =   -x_unpading[1] ,  -y_unpading[1] , -z_unpading[1]

        return lax.pad(arr, padding_value=0.0, padding_config=[(first_x, last_x , 0), (first_y, last_y , 0), (first_z, last_z , 0)])

    # Test padding
    print("-"*40)
    print(f"Testing padding")

    with mesh:
        padded_array = jnp.pad(global_array, padding)
        padding_width = jnp.array(padding)
        sharded_padded = sharded_pad(global_array,padding_width)
        jaxdecomp_padded = slice_pad(global_array,padding,pdims)
    

    first_x , last_x = padding[0]
    first_y , last_y = padding[1]

    # using just jnp pad will pad the entire global array and not the slices
    expected_padded_shape = (global_shape[0] + first_x + last_x , global_shape[1] + first_y + last_y , global_shape[2])
    # Using a sharded jnp pad will pad the slices
    expected_sharded_pad_shape = (global_shape[0] + (first_x + last_x) * pdims[1] , global_shape[1] + (first_y + last_y) * pdims[0] , global_shape[2])

    print(f"Shape of global_array {global_array.shape}")
    print(f"Shape of padded_array {padded_array.shape} it should be {expected_padded_shape}")
    print(f"Shape of sharded_padded {sharded_padded.shape} it should be {expected_sharded_pad_shape}")
    print(f"Shape of jaxdecomp_padded {jaxdecomp_padded.shape} it should be {expected_sharded_pad_shape}")

    # Using pad on a global array will pad the array (uses communication)
    assert_array_equal(padded_array.shape,expected_padded_shape)
    # Test slice_pad agains reference sharded pad
    assert_array_equal(sharded_padded.shape,expected_sharded_pad_shape)
    assert_array_equal(jaxdecomp_padded.shape,expected_sharded_pad_shape)

    # Test unpadding
    print("-"*40)
    print(f"Testing unpadding")

    with mesh:
        unpadded_array = lax.pad(padded_array, padding_value=0.0, padding_config=((-32, -32 , 0), (-32, -32 , 0), (0, 0 , 0)))
        sharded_unpadded = sharded_unpad(sharded_padded)
        jaxdecomp_unpadded = slice_unpad(jaxdecomp_padded,padding,pdims)
    
    print(f"Shape of unpadded_array {unpadded_array.shape} should be {global_shape}")
    print(f"Shape of sharded_unpadded {sharded_unpadded.shape} should be {global_shape}")
    print(f"Shape of jaxdecomp_unpadded {jaxdecomp_unpadded.shape} should be {global_shape}")
    
    first_x , last_x = padding[0]
    first_y , last_y = padding[1]

    # Using pad on a global array will unpad the array (uses communication)
    assert_array_equal(unpadded_array.shape,global_shape)
    # Test slice_pad agains reference sharded pad
    assert_array_equal(sharded_unpadded.shape,global_shape)
    assert_array_equal(jaxdecomp_unpadded.shape,global_shape)

    gathered_original = multihost_utils.process_allgather(global_array,tiled=True)
    gathered_unpadded = multihost_utils.process_allgather(unpadded_array,tiled=True)
    # Make sure the unpadded arrays is equal to the original array
    assert_array_equal(gathered_original,gathered_unpadded)


def test_end():
    # fake test to finalize the MPI processes
    jaxdecomp.finalize()
    jax.distributed.shutdown()