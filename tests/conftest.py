import os
from math import prod

import pytest

setup_done = False
on_cluster = False


def is_on_cluster():
    global on_cluster
    return on_cluster


def initialize_distributed():
    global setup_done
    global on_cluster
    if not setup_done:
        if 'SLURM_JOB_ID' in os.environ:
            on_cluster = True
            print('Running on cluster')
            import jax

            jax.distributed.initialize()
            setup_done = True
            on_cluster = True
        else:
            print('Running locally')
            setup_done = True
            on_cluster = False
            os.environ['JAX_PLATFORM_NAME'] = 'cpu'
            os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=8'
            import jax


@pytest.fixture(scope='session', autouse=True)
def setup_and_teardown_session():
    # Code to run at the start of the session
    print('Starting session...')
    initialize_distributed()
    # Setup code here
    # e.g., connecting to a database, initializing some resources, etc.

    yield

    import jax

    import jaxdecomp

    # Code to run at the end of the session
    print('Ending session...')
    jaxdecomp.finalize()
    jax.distributed.shutdown()

    # Teardown code here
    # e.g., closing connections, cleaning up resources, etc.


@pytest.fixture(params=['auto', 'explicit'])
def axis_type(request):
    return request.param


def compare_sharding(sharding1, sharding2):
    def get_axis_size(sharding, idx):
        axis_name = sharding.spec[idx]
        if axis_name is None:
            return 1
        else:
            return sharding.mesh.shape[sharding.spec[idx]]

    def get_pdims_from_sharding(sharding):
        return tuple([get_axis_size(sharding, i) for i in range(len(sharding.spec))])

    pdims1 = get_pdims_from_sharding(sharding1)
    pdims2 = get_pdims_from_sharding(sharding2)
    pdims1 = pdims1 + (1,) * (3 - len(pdims1))
    pdims2 = pdims2 + (1,) * (3 - len(pdims2))
    return pdims1 == pdims2


def replace_none_or_zero(value):
    # Replace None or 0 with 1
    return 0 if value is None else value


def process_slices(slices_tuple):
    start_product = 1
    stop_product = 1

    for s in slices_tuple:
        # Multiply the start and stop values, replacing None/0 with 1
        start_product *= replace_none_or_zero(s.start)
        stop_product *= replace_none_or_zero(s.stop)

    # Return the sum of the two products
    return int(start_product + stop_product)


def create_mesh(pdims, axis_type, transposed_devices=False):
    import jax
    from jax.experimental import mesh_utils
    from jax.sharding import AxisType

    size = jax.device_count()
    assert len(pdims) == 2
    if prod(pdims) != size:
        devices_id = jax.devices()[: prod(pdims)]
    else:
        devices_id = jax.devices()

    # Remap to the global array from the local slicei
    devices = mesh_utils.create_device_mesh(pdims, devices=devices_id)
    if transposed_devices:
        devices = devices.T

    if axis_type == 'explicit':
        axis_types = (AxisType.Explicit,) * len(pdims)
    else:
        axis_types = (AxisType.Auto,) * len(pdims)
    return jax.make_mesh(devices.shape, ('z', 'y'), devices=devices.flatten(), axis_types=axis_types)


def create_batched_mesh(batch_size, spatial_pdims, axis_type):
    import jax
    from jax.sharding import AxisType

    size = jax.device_count()
    assert len(spatial_pdims) == 2
    assert batch_size * prod(spatial_pdims) == size, 'batch_size * prod(spatial_pdims) must equal device count'

    mesh_shape = (batch_size, *spatial_pdims)
    if axis_type == 'explicit':
        axis_types = (AxisType.Explicit,) * len(mesh_shape)
    else:
        axis_types = (AxisType.Auto,) * len(mesh_shape)
    return jax.make_mesh(mesh_shape, ('c', 'z', 'y'), axis_types=axis_types)


def create_batched_spmd_array(global_shape, mesh):
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    assert len(global_shape) == 4

    local_shape = (
        global_shape[0] // mesh.shape['c'],
        global_shape[1] // mesh.shape['z'],
        global_shape[2] // mesh.shape['y'],
        global_shape[3],
    )

    sharding = NamedSharding(mesh, P('c', 'z', 'y'))
    global_array = jax.make_array_from_callback(
        global_shape,
        sharding,
        data_callback=lambda x: jax.random.normal(jax.random.PRNGKey(process_slices(x)), local_shape),
    )

    return global_array


def create_ones_spmd_array(global_shape, mesh):
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    assert len(global_shape) == 3

    local_shape = (
        global_shape[0] // mesh.shape['z'],
        global_shape[1] // mesh.shape['y'],
        global_shape[2],
    )

    sharding = NamedSharding(mesh, P('z', 'y'))
    global_array = jax.make_array_from_callback(global_shape, sharding, data_callback=lambda _: jax.numpy.ones(local_shape))

    return global_array


# Helper function to create a 3D array and remap it to the global array
def create_spmd_array(global_shape, mesh):
    import jax
    from jax.sharding import NamedSharding
    from jax.sharding import PartitionSpec as P

    assert len(global_shape) == 3

    local_shape = (
        global_shape[0] // mesh.shape['z'],
        global_shape[1] // mesh.shape['y'],
        global_shape[2],
    )

    sharding = NamedSharding(mesh, P('z', 'y'))
    global_array = jax.make_array_from_callback(
        global_shape,
        sharding,
        data_callback=lambda x: jax.random.normal(jax.random.PRNGKey(process_slices(x)), local_shape),
    )

    return global_array


def assert_allclose(x, y, rtol=1e-5, atol=1e-5):
    import jax

    x_leaves = jax.tree.leaves(x)
    y_leaves = jax.tree.leaves(y)

    return jax.tree.all(
        jax.tree.map(
            lambda x, y: jax.numpy.allclose(x, y, rtol=rtol, atol=atol),
            x_leaves,
            y_leaves,
        )
    )


def assert_array_equal(x, y):
    import jax

    x_leaves = jax.tree.leaves(x)
    y_leaves = jax.tree.leaves(y)
    return jax.tree.all(jax.tree.map(lambda x, y: jax.numpy.array_equal(x, y), x_leaves, y_leaves))
