from conftest import (
    assert_array_equal,
    create_ones_spmd_array,
    create_spmd_array,
    initialize_distributed,
    is_on_cluster,
)

initialize_distributed()
import jax

size = jax.device_count()
from functools import partial
from itertools import product
from math import prod

import jax.numpy as jnp
import pytest
from jax import lax
from jax.experimental.multihost_utils import process_allgather
from jax.experimental.shard_map import shard_map
from jax.sharding import PartitionSpec as P

import jaxdecomp
from jaxdecomp._src.spmd_ops import ALLOW_SHARDY_PARTITIONER

pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100
pdims = [(1, size), (size, 1), pencil_1, pencil_2]


def split_into_grid(array, pdims):
    """
    Splits the array into a 2D grid using vsplit and hsplit based on pdims.

    Args:
        array: The array to be split.
        pdims: A tuple (vertical_splits, horizontal_splits) defining how to split the array.

    Returns:
        A 2D list of lists where each element is a sub-array.
    """
    # First, vsplit into vertical splits (rows)
    vertical_splits = jax.tree.map(lambda array: jnp.vsplit(array, pdims[1]), array)
    vertical_splits = jax.tree.leaves(vertical_splits)
    # For each vertical split, hsplit into horizontal splits (columns)
    grid = [jnp.hsplit(vsplit, pdims[0]) for vsplit in vertical_splits]

    return grid


all_gather = partial(process_allgather, tiled=True)


@pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
@pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
@pytest.mark.parametrize('pdims', pdims)
# Test with Slab and Pencil decompositions
def test_halo_against_cudecomp(pdims, use_shardy):
    jnp.set_printoptions(linewidth=200)

    print('*' * 80)
    print(f'Testing with pdims {pdims} and use_shardy {use_shardy}')

    jax.config.update('jax_use_shardy_partitioner', use_shardy)

    if use_shardy and not ALLOW_SHARDY_PARTITIONER:
        pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

    global_shape = (16, 16, 16)
    global_array, mesh = create_spmd_array(global_shape, pdims)

    # @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
    # def sharded_add_multiply(arr, devices):
    #   return (arr) + (devices + 10)

    halo_size = 2

    halo_x = (halo_size, halo_size) if pdims[1] > 1 else (0, 0)
    halo_y = (halo_size, halo_size) if pdims[0] > 1 else (0, 0)
    halo_extents = (halo_x[0], halo_y[0])
    periodic = (True, True)
    padding = (halo_x, halo_y, (0, 0))

    @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
    def pad(arr):
        return jnp.pad(arr, padding, mode='linear_ramp', end_values=20)

    # perform halo exchange
    updated_array = pad(global_array)
    jax_exchanged = jaxdecomp.halo_exchange(updated_array, halo_extents=halo_extents, halo_periods=periodic, backend='JAX')
    cudecomp_exchanged = jaxdecomp.halo_exchange(
        updated_array,
        halo_extents=halo_extents,
        halo_periods=periodic,
        backend='CUDECOMP',
    )

    g_array = all_gather(updated_array)
    g_jax_exchanged = all_gather(jax_exchanged)
    g_cudecomp_exchanged = all_gather(cudecomp_exchanged)
    print(f'Original \n{g_array[:, :, 0]}')
    print(f'exchanged cudecomp \n{g_jax_exchanged[:, :, 0]}')
    print(f'exchanged jax \n{g_cudecomp_exchanged[:, :, 0]}')

    assert_array_equal(g_jax_exchanged, g_cudecomp_exchanged)


class TestHaloExchange:
    def run_test(self, global_shape, pdims, backend, use_shardy):
        print('*' * 80)
        print(f'Testing with pdims {pdims} and use_shardy {use_shardy}')

        jax.config.update('jax_use_shardy_partitioner', use_shardy)

        if use_shardy and not ALLOW_SHARDY_PARTITIONER:
            pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

        jnp.set_printoptions(linewidth=200)

        global_array, mesh = create_ones_spmd_array(global_shape, pdims)
        halo_size = 2

        halo_x = (halo_size, halo_size) if pdims[1] > 1 else (0, 0)
        halo_y = (halo_size, halo_size) if pdims[0] > 1 else (0, 0)
        halo_extents = (halo_x[0], halo_y[0])
        periodic = (True, True)
        padding = (halo_x, halo_y, (0, 0))

        @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
        def pad(arr):
            return jax.tree.map(
                lambda arr: jnp.pad(arr, padding, mode='linear_ramp', end_values=20),
                arr,
            )

        @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
        def multiply(arr):
            z_index = lax.axis_index('z') + 1
            y_index = lax.axis_index('y') + 1
            aranged = jnp.arange(prod(arr.shape)).reshape(arr.shape)
            arr *= z_index + y_index * pdims[0]

            arr += aranged

            return arr

        # perform halo exchange
        padded_array = multiply(global_array)
        padded_array = pad(padded_array)
        # periodic_exchanged_array = jaxdecomp.halo_exchange(
        #    padded_array,
        #    halo_extents=halo_extents,
        #    halo_periods=periodic,
        #    backend=backend,
        # )
        exchanged_array = jaxdecomp.halo_exchange(
            padded_array,
            halo_extents=halo_extents,
            halo_periods=periodic,
            backend=backend,
        )

        # Gather array from all processes
        # gathered_array = multihost_utils.process_allgather(global_array,tiled=True)
        exchanged_gathered_array = all_gather(exchanged_array, tiled=True)
        # periodic_exchanged_gathered_array = all_gather(
        #    periodic_exchanged_array, tiled=True
        # )
        padded_gathered_array = all_gather(padded_array, tiled=True)

        gathered_array_slices = split_into_grid(padded_gathered_array, pdims)
        gathered_exchanged_slices = split_into_grid(exchanged_gathered_array, pdims)
        # gathered_periodic_exchanged_slices = split_into_grid(
        #    periodic_exchanged_gathered_array, pdims
        # )

        print(f'len gathered array slices {len(gathered_array_slices)}')
        print(f'len Y gathered array slices {len(gathered_array_slices[0])}')

        def next_index(x, pdims):
            return x + 1 if x < pdims - 1 else 0

        def prev_index(x, pdims):
            return x - 1 if x > 0 else pdims - 1

        for z_slice, y_slice in product(range(pdims[1]), range(pdims[0])):
            print(f'z {z_slice} y {y_slice}')
            original_slice = gathered_array_slices[z_slice][y_slice]
            current_slice = gathered_exchanged_slices[z_slice][y_slice]
            next_z = next_index(z_slice, pdims[1])
            next_y = next_index(y_slice, pdims[0])
            prev_z = prev_index(z_slice, pdims[1])
            prev_y = prev_index(y_slice, pdims[0])
            # Get up and down slices
            up_slice = gathered_exchanged_slices[prev_z][y_slice]
            down_slice = gathered_exchanged_slices[next_z][y_slice]
            # Get left and right slices
            left_slice = gathered_exchanged_slices[z_slice][prev_y]
            right_slice = gathered_exchanged_slices[z_slice][next_y]
            # Get the upper corners
            upper_left_corner = gathered_exchanged_slices[prev_z][prev_y]
            upper_right_corner = gathered_exchanged_slices[prev_z][next_y]
            # Get the lower corners
            lower_left_corner = gathered_exchanged_slices[next_z][prev_y]
            lower_right_corner = gathered_exchanged_slices[next_z][next_y]

            print(f'z {z_slice} y {y_slice}')
            print(f'original slice \n{original_slice[:, :, 0]}')
            print(f'up slice \n{up_slice[:, :, 0]}')
            print(f'current slice \n{current_slice[:, :, 0]}')
            print(f'down slice \n{down_slice[:, :, 0]}')
            print('--' * 40)

            # if up down padding check the up down slices
            if pdims[1] > 1:
                # Check the upper padding
                assert_array_equal(current_slice[:halo_size], up_slice[-2 * halo_size : -halo_size])
                # Check the lower padding
                assert_array_equal(current_slice[-halo_size:], down_slice[halo_size : halo_size * 2])

            # if left right padding check the left right slices
            if pdims[0] > 1:
                # Check the left padding
                assert_array_equal(
                    current_slice[:, :halo_size],
                    left_slice[:, -2 * halo_size : -halo_size :],
                )
                # Check the right padding
                assert_array_equal(
                    current_slice[:, -halo_size:],
                    right_slice[:, halo_size : halo_size * 2],
                )
            # if both padded check the corners
            if pdims[0] > 1 and pdims[1] > 1:
                # Check the upper left corner
                assert_array_equal(
                    current_slice[:halo_size, :halo_size],
                    upper_left_corner[-2 * halo_size : -halo_size, -2 * halo_size : -halo_size],
                )
                # Check the upper right corner
                assert_array_equal(
                    current_slice[:halo_size, -halo_size:],
                    upper_right_corner[-2 * halo_size : -halo_size, halo_size : halo_size * 2],
                )
                # Check the lower left corner
                assert_array_equal(
                    current_slice[-halo_size:, :halo_size],
                    lower_left_corner[halo_size : halo_size * 2, -2 * halo_size : -halo_size],
                )
                # Check the lower right corner
                assert_array_equal(
                    current_slice[-halo_size:, -halo_size:],
                    lower_right_corner[halo_size : halo_size * 2, halo_size : halo_size * 2],
                )

    @pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('pdims', pdims)
    def test_cudecomp_halo(self, pdims, use_shardy):
        self.run_test((32, 32, 32), pdims, 'CUDECOMP', use_shardy)

    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('pdims', pdims)
    def test_jax_halo(
        self,
        pdims,
        use_shardy,
    ):
        self.run_test((16, 16, 16), pdims, 'JAX', use_shardy)


class TestHaloExchangeGrad:
    def run_test(self, global_shape, pdims, backend, use_shardy):
        print('*' * 80)
        print(f'Testing with pdims {pdims} and use_shardy {use_shardy}')

        jax.config.update('jax_use_shardy_partitioner', use_shardy)

        if use_shardy and not ALLOW_SHARDY_PARTITIONER:
            pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

        jnp.set_printoptions(linewidth=200)

        global_array, mesh = create_ones_spmd_array(global_shape, pdims)
        halo_size = 2

        halo_x = (halo_size, halo_size) if pdims[1] > 1 else (0, 0)
        halo_y = (halo_size, halo_size) if pdims[0] > 1 else (0, 0)
        halo_extents = (halo_x[0], halo_y[0])
        periodic = (True, True)
        padding = (halo_x, halo_y, (0, 0))

        @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
        def pad(arr):
            return jax.tree.map(
                lambda arr: jnp.pad(arr, padding, mode='linear_ramp', end_values=20),
                arr,
            )

        @partial(shard_map, mesh=mesh, in_specs=P('z', 'y'), out_specs=P('z', 'y'))
        def multiply(arr):
            z_index = lax.axis_index('z') + 1
            y_index = lax.axis_index('y') + 1
            aranged = jnp.arange(prod(arr.shape)).reshape(arr.shape)
            arr *= z_index + y_index * pdims[0]

            arr += aranged

            return arr

        @jax.jit
        def forward_model(array):
            padded_array = multiply(array)
            padded_array = pad(array)

            exchanged_array = jaxdecomp.halo_exchange(
                padded_array,
                halo_extents=halo_extents,
                halo_periods=periodic,
                backend=backend,
            )

            return exchanged_array

        @jax.grad
        @jax.jit
        def model(array, obs):
            padded_array = multiply(array)
            padded_array = pad(array)

            exchanged_array = jaxdecomp.halo_exchange(
                padded_array,
                halo_extents=halo_extents,
                halo_periods=periodic,
                backend=backend,
            )

            return jnp.sum((exchanged_array - obs) ** 2)

        obs = forward_model(global_array)

        assert_array_equal(model(global_array, obs), jnp.zeros_like(global_array))
        assert_array_equal(model(global_array * 2, obs), jnp.full_like(global_array, 3))

    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('pdims', pdims)
    def test_jax_halo(
        self,
        pdims,
        use_shardy,
    ):
        self.run_test((16, 16, 16), pdims, 'JAX', use_shardy)
