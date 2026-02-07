from conftest import (
    assert_allclose,
    assert_array_equal,
    compare_sharding,
    create_batched_mesh,
    create_batched_spmd_array,
    create_mesh,
    create_spmd_array,
    initialize_distributed,
    is_on_cluster,
)

initialize_distributed()
import jax

size = jax.device_count()

from functools import partial

import jax.numpy as jnp
import pytest
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

try:
    from jax.sharding import auto_axes
except ImportError:
    auto_axes = None

import jaxdecomp
from jaxdecomp import (
    transposeXtoY,
    transposeYtoX,
    transposeYtoZ,
    transposeZtoY,
)
from jaxdecomp._src.spmd_ops import ALLOW_SHARDY_PARTITIONER

jax.config.update('jax_enable_x64', True)

all_gather = partial(process_allgather, tiled=True)

pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100

decomp = [(size, 1), (1, size), pencil_1, pencil_2]
global_shapes = [(8, 16, 32), (8, 8, 8), (29 * size, 19 * size, 17 * size)]
# Cubes, non-cubes and primes
local_transpose = [
    pytest.param(False, id='no_local_transpose'),
    pytest.param(True, id='local_transpose'),
]
use_shardy = [
    pytest.param(False, id='no_shardy'),
    pytest.param(True, id='shardy'),
]

# Parametrize over spatial decompositions that fit in 8 devices with batch_size=2
batched_decomp = [
    pytest.param((4, 1), id='SLAB_XY'),
    pytest.param((1, 4), id='SLAB_YZ'),
    pytest.param((2, 2), id='PENCILS'),
]


class TestTransposes:
    def run_test(self, pdims, global_shape, local_transpose, backend, use_shardy, axis_type):
        """Goes from an array of shape [z,y,x] # What we call an x pencil
        to [x,z,y] # what we call a y pencil
        """
        print('*' * 80)
        print(f'Testing with pdims {pdims} and global shape {global_shape} with local transpose {local_transpose} and use_shardy {use_shardy}')

        jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)
        jax.config.update('jax_use_shardy_partitioner', use_shardy)

        if use_shardy and not ALLOW_SHARDY_PARTITIONER:
            pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

        mesh = create_mesh(pdims, axis_type=axis_type)
        global_array = create_spmd_array(global_shape, mesh)

        if local_transpose:
            original_sharding = NamedSharding(mesh, P('z', 'y'))
            y_pencil_sharding = NamedSharding(mesh, P('y', 'z'))
            z_pencil_sharding = NamedSharding(mesh, P('z', 'y'))
        else:
            original_sharding = NamedSharding(mesh, P('z', 'y'))
            y_pencil_sharding = NamedSharding(mesh, P('z', None, 'y'))
            z_pencil_sharding = NamedSharding(mesh, P(None, 'z', 'y'))

        if axis_type == 'explicit':
            if auto_axes is None:
                pytest.skip(reason='auto_axes is not available in this JAX version, please upgrade to at least JAX 0.9.0')

            @auto_axes
            def transposeXtoY_safe(x, out_sharding=y_pencil_sharding):
                return jaxdecomp.transposeXtoY(x, backend=backend)

            @auto_axes
            def transposeYtoZ_safe(x, out_sharding=z_pencil_sharding):
                return jaxdecomp.transposeYtoZ(x, backend=backend)

            @auto_axes
            def transposeZtoY_safe(x, out_sharding=y_pencil_sharding):
                return jaxdecomp.transposeZtoY(x, backend=backend)

            @auto_axes
            def transposeYtoX_safe(x, out_sharding=original_sharding):
                return jaxdecomp.transposeYtoX(x, backend=backend)

            jd_tranposed_xy = transposeXtoY_safe(global_array, out_sharding=y_pencil_sharding)
            jd_tranposed_yz = transposeYtoZ_safe(jd_tranposed_xy, out_sharding=z_pencil_sharding)
            jd_tranposed_zy = transposeZtoY_safe(jd_tranposed_yz, out_sharding=y_pencil_sharding)
            jd_tranposed_yx = transposeYtoX_safe(jd_tranposed_zy, out_sharding=original_sharding)
        else:
            jd_tranposed_xy = transposeXtoY(global_array, backend=backend)
            jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy, backend=backend)
            jd_tranposed_zy = transposeZtoY(jd_tranposed_yz, backend=backend)
            jd_tranposed_yx = transposeYtoX(jd_tranposed_zy, backend=backend)

        print(f'jd_tranposed_xy shape {jd_tranposed_xy.shape}')
        print(f'jd_tranposed_yz shape {jd_tranposed_yz.shape}')
        print(f'jd_tranposed_zy shape {jd_tranposed_zy.shape}')
        print(f'jd_tranposed_yx shape {jd_tranposed_yx.shape}')

        print(f'Original sharding {original_sharding}')
        print(f'y pencil sharding {y_pencil_sharding}')
        print(f'z pencil sharding {z_pencil_sharding}')

        print(f'JD tranposed xy sharding {jd_tranposed_xy.sharding.spec}')
        print(f'JD tranposed yz sharding {jd_tranposed_yz.sharding.spec}')
        print(f'JD tranposed zy sharding {jd_tranposed_zy.sharding.spec}')
        print(f'JD tranposed yx sharding {jd_tranposed_yx.sharding.spec}')

        assert compare_sharding(global_array.sharding, original_sharding)
        assert compare_sharding(jd_tranposed_xy.sharding, y_pencil_sharding)
        assert compare_sharding(jd_tranposed_yz.sharding, z_pencil_sharding)
        assert compare_sharding(jd_tranposed_zy.sharding, y_pencil_sharding)
        assert compare_sharding(jd_tranposed_yx.sharding, original_sharding)

        gathered_array = all_gather(global_array)

        gathered_jd_xy = all_gather(jd_tranposed_xy)
        gathered_jd_yz = all_gather(jd_tranposed_yz)
        gathered_jd_zy = all_gather(jd_tranposed_zy)
        gathered_jd_yx = all_gather(jd_tranposed_yx)

        # Explanation :
        # Tranposing forward is a shift axis to the right so ZYX to XZY to YXZ (2 0 1)
        # Tranposing backward is a shift axis to the left so YXZ to XZY to ZYX (1 2 0)
        # Double Tranposing from ZYX to YXZ is double (2 0 1) so  (1 2 0)

        if local_transpose:
            forward_tranpose = [2, 0, 1]
            backward_tranpose = [1, 2, 0]
            double_forward = [1, 2, 0]
        else:
            forward_tranpose = [0, 1, 2]
            backward_tranpose = [0, 1, 2]
            double_forward = [0, 1, 2]

        print(f'For local_transpose {local_transpose} forward_tranpose {forward_tranpose} backward_tranpose {backward_tranpose}')
        #
        # Test X to Y transpose
        # It tranposes ZYX to XZY to from 0 1 2 to 2 0 1
        assert_array_equal(gathered_array.transpose(forward_tranpose), gathered_jd_xy)
        # *********************************************
        # Test Y to Z transpose
        # It tranposes XZY to YXZ to from 0 1 2 to 2 0 1 again
        assert_array_equal(gathered_jd_xy.transpose(forward_tranpose), gathered_jd_yz)
        # and from the global array ZYX to YXZ so from 0 1 2 to 1 2 0
        assert_array_equal(gathered_array.transpose(double_forward), gathered_jd_yz)
        # *********************************************
        # Test Z to Y transpose
        # It tranposes YXZ to XZY to from 0 1 2 to 1 2 0
        assert_array_equal(gathered_jd_yz.transpose(backward_tranpose), gathered_jd_zy)
        # The Y pencils should match in forward and backward transposes (despite the inverted grid)
        # assert_array_equal(gathered_jd_zy, gathered_jd_xy)
        # *********************************************
        # Test Y to X transpose
        # It tranposes XZY to ZYX to from 0 1 2 to 1 2 0
        assert_array_equal(gathered_jd_zy.transpose(backward_tranpose), gathered_jd_yx)
        # The X pencils should match in forward and backward transposes (original array)
        assert_array_equal(gathered_jd_yx, gathered_array)

        print(f'Pdims {pdims} with local_transpose {local_transpose} is ok!!')

        jax.clear_caches()

    @pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
    # Cartesian product tests
    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('use_shardy', use_shardy)  # Test with and without shardy
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_cudecomp_transpose(self, pdims, global_shape, local_transpose, use_shardy, axis_type):
        self.run_test(pdims, global_shape, local_transpose, backend='cuDecomp', use_shardy=use_shardy, axis_type=axis_type)

    # Cartesian product tests
    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('use_shardy', use_shardy)  # Test with and without shardy
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_jax_transpose(self, pdims, global_shape, local_transpose, use_shardy, axis_type):
        self.run_test(pdims, global_shape, local_transpose, backend='jax', use_shardy=use_shardy, axis_type=axis_type)


class TestTransposesGrad:
    def run_test(self, pdims, global_shape, local_transpose, backend, use_shardy, axis_type):
        """Goes from an array of shape [z,y,x] # What we call an x pencil
        to [x,z,y] # what we call a y pencil
        """
        print('*' * 80)
        print(f'Testing with pdims {pdims} and global shape {global_shape} with local transpose {local_transpose} and use_shardy {use_shardy}')

        jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)
        jax.config.update('jax_use_shardy_partitioner', use_shardy)

        if use_shardy and not ALLOW_SHARDY_PARTITIONER:
            pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

        if axis_type == 'explicit':
            pytest.skip(reason='Explicit axis type not yet supported for Transpose tests')

        mesh = create_mesh(pdims, axis_type=axis_type)
        global_array = create_spmd_array(global_shape, mesh)

        @jax.jit
        def jaxdecomp_transpose(global_array):
            jd_tranposed_xy = transposeXtoY(global_array, backend=backend)
            jd_tranposed_yz = transposeYtoZ(jd_tranposed_xy, backend=backend)
            jd_tranposed_zy = transposeZtoY(jd_tranposed_yz, backend=backend)
            jd_tranposed_yx = transposeYtoX(jd_tranposed_zy, backend=backend)
            y = (jd_tranposed_yx * jax.tree.map(jnp.conjugate, jd_tranposed_yx)).real.sum()
            return y

        @jax.jit
        def jax_transpose(global_array):
            jax_transposed_xy = global_array.transpose([0, 2, 1])
            jax_transposed_yz = jax_transposed_xy.transpose([2, 1, 0])
            jax_transposed_zy = jax_transposed_yz.transpose([2, 1, 0])
            jax_transposed_yx = jax_transposed_zy.transpose([0, 2, 1])
            y = (jax_transposed_yx * jax.tree.map(jnp.conjugate, jax_transposed_yx)).real.sum()
            return y

        array_grad = jax.grad(jaxdecomp_transpose)(global_array)
        print("Here is the gradient I'm getting", array_grad.shape)

        gathered_array = all_gather(global_array)
        gathered_grads = all_gather(array_grad)
        jax_grad = jax.grad(jax_transpose)(gathered_array)

        print(f'Shape of JAX array {jax_grad.shape}')
        # Check the gradients
        assert_allclose(jax_grad, gathered_grads, rtol=1e-5, atol=1e-5)

        jax.clear_caches()

    @pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
    # Cartesian product test
    @pytest.mark.parametrize('pdims', decomp)
    @pytest.mark.parametrize('global_shape', global_shapes)
    @pytest.mark.parametrize('local_transpose', local_transpose)
    @pytest.mark.parametrize('use_shardy', use_shardy)  # Test with and without shardy
    def test_cudecomp_transpose_grad(self, pdims, global_shape, local_transpose, use_shardy, axis_type):
        self.run_test(pdims, global_shape, local_transpose, backend='cuDecomp', use_shardy=use_shardy, axis_type=axis_type)

    # Cartesian product test
    @pytest.mark.parametrize('pdims', decomp)
    @pytest.mark.parametrize('global_shape', global_shapes)
    @pytest.mark.parametrize('local_transpose', local_transpose)
    @pytest.mark.parametrize('use_shardy', use_shardy)  # Test with and without shardy
    def test_jax_transpose_grad(self, pdims, global_shape, local_transpose, use_shardy, axis_type):
        self.run_test(pdims, global_shape, local_transpose, backend='jax', use_shardy=use_shardy, axis_type=axis_type)


@pytest.mark.parametrize('use_shardy', use_shardy)
@pytest.mark.parametrize('spatial_pdims', batched_decomp)
def test_sharded_vmap(spatial_pdims, use_shardy, axis_type):
    if use_shardy and not ALLOW_SHARDY_PARTITIONER:
        pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

    if axis_type == 'explicit':
        pytest.skip(reason='Explicit axis type not yet supported for sharded vmap tests')

    jax.config.update('jax_use_shardy_partitioner', use_shardy)
    jaxdecomp.config.update('transpose_axis_contiguous', True)

    batch_size = 2
    global_shape = (batch_size, 8, 8, 8)

    # Create 3D mesh: (batch, spatial_x, spatial_y)
    mesh = create_batched_mesh(batch_size, spatial_pdims, axis_type=axis_type)
    global_array = create_batched_spmd_array(global_shape, mesh)

    # Forward transpose X->Y
    v_transposeXtoY = jax.vmap(transposeXtoY)
    transposed_xy = v_transposeXtoY(global_array)

    assert transposed_xy.shape == global_shape

    # Roundtrip: Y->X should recover original
    v_transposeYtoX = jax.vmap(transposeYtoX)
    roundtrip = v_transposeYtoX(transposed_xy)

    assert roundtrip.shape == global_shape
    assert_array_equal(global_array, roundtrip)

    # Compare batch element [0] against individual transpose on a 2D mesh

    # Gather both to compare
    gathered_batched = all_gather(transposed_xy)
    gathered_batched_input = all_gather(global_array)

    # In contiguous mode, forward transpose permutation is [2, 0, 1]
    forward_transpose = [2, 0, 1]
    assert_array_equal(gathered_batched[0], gathered_batched_input[0].transpose(forward_transpose))

    jax.clear_caches()
