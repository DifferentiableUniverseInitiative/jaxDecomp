from conftest import (
    assert_allclose,
    create_spmd_array,
    initialize_distributed,
    is_on_cluster,
)

initialize_distributed()
import jax  # noqa: E402

size = jax.device_count()

jax.config.update('jax_enable_x64', True)

from functools import partial

import jax.numpy as jnp
import pytest
from jax.experimental.multihost_utils import process_allgather

import jaxdecomp
from jaxdecomp._src import PENCILS, SLAB_XY, SLAB_YZ
from jaxdecomp._src.spmd_ops import ALLOW_SHARDY_PARTITIONER

all_gather = partial(process_allgather, tiled=True)

pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100

decomp = [(size, 1), (1, size), pencil_1, pencil_2]
global_shapes = [
    (8, 16, 32),
    (8, 8, 8),
    (29 * size, 19 * size, 17 * size),
]  # Cubes, non-cubes and primes
local_transpose = [True, False]


class TestFFTs:
    def run_test(self, pdims, global_shape, local_transpose, backend, use_shardy):
        if use_shardy and not ALLOW_SHARDY_PARTITIONER:
            pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

        print('*' * 80)
        print(f'Testing with pdims {pdims} and global shape {global_shape} and local transpose {local_transpose} use shardy {use_shardy}')
        if pdims[0] == 1:
            penciltype = SLAB_XY
        elif pdims[1] == 1:
            penciltype = SLAB_YZ
        else:
            penciltype = PENCILS
        print(f'Decomposition type {penciltype}')

        jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)
        jax.config.update('jax_use_shardy_partitioner', use_shardy)

        global_array, mesh = create_spmd_array(global_shape, pdims)

        # Perform distributed FFT
        karray = jaxdecomp.fft.pfft3d(global_array, backend=backend)
        # Perform inverse FFT
        rec_array = jaxdecomp.fft.pifft3d(karray, backend=backend)

        print(f'orignal shard {global_array.sharding.spec}')
        print(f'sharding of karray {karray.sharding.spec}')
        print(f'sharding of rec_array {rec_array.sharding.spec}')
        # Check the forward FFT
        gathered_array = all_gather(global_array)
        gathered_karray = all_gather(karray)
        gathered_rec_array = all_gather(rec_array)

        jax_karray = jax.tree.map(jnp.fft.fftn, gathered_array)
        # Check the forward FFT
        if penciltype == SLAB_YZ:
            transpose_back = [2, 0, 1]
        else:
            transpose_back = [1, 2, 0]
        if not local_transpose:
            transpose_back = [0, 1, 2]
        else:
            transpose_back = [1, 2, 0]

        # Check reconstructed array
        assert_allclose(gathered_array.real, gathered_rec_array.real, rtol=1e-5, atol=1e-5)
        assert_allclose(gathered_array.imag, gathered_rec_array.imag, rtol=1e-5, atol=1e-5)

        print('Reconstruction check OK!')

        jax_karray_transposed = jax_karray.transpose(transpose_back)
        assert_allclose(gathered_karray.real, jax_karray_transposed.real, rtol=1e-5, atol=1e-5)
        assert_allclose(gathered_karray.imag, jax_karray_transposed.imag, rtol=1e-5, atol=1e-5)

        print('FFT with transpose check OK!')

        # Trigger rejit in case local transpose is switched
        jax.clear_caches()

    @pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
    # Cartesian product tests
    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_cudecomp_fft(self, pdims, global_shape, local_transpose, use_shardy):
        self.run_test(pdims, global_shape, local_transpose, backend='cuDeComp', use_shardy=use_shardy)

    # Cartesian product tests
    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_jax_fft(self, pdims, global_shape, local_transpose, use_shardy):
        self.run_test(pdims, global_shape, local_transpose, backend='jax', use_shardy=use_shardy)


class TestFFTsGrad:
    def run_test(self, pdims, global_shape, local_transpose, backend, use_shardy):
        if use_shardy and not ALLOW_SHARDY_PARTITIONER:
            pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

        if pdims[0] == 1:
            penciltype = SLAB_XY
        elif pdims[1] == 1:
            penciltype = SLAB_YZ
        else:
            penciltype = PENCILS

        # Check the forward FFT
        if penciltype == SLAB_YZ:
            transpose_back = [2, 0, 1]
        else:
            transpose_back = [1, 2, 0]
        if not local_transpose:
            transpose_back = [0, 1, 2]
        else:
            transpose_back = [1, 2, 0]

        print('*' * 80)
        # Cartesian product tests
        print(f"""
                Testing with pdims {pdims}
                            global shape {global_shape}
                            local transpose {local_transpose}
                            backend {backend}
                            use shardy {use_shardy}
                """)
        jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)
        jax.config.update('jax_use_shardy_partitioner', use_shardy)
        global_array, mesh = create_spmd_array(global_shape, pdims)

        print('-' * 40)
        print('Testing fwd grad')

        @jax.jit
        def spmd_grad(arr):
            y = jaxdecomp.fft.pfft3d(arr, backend=backend)
            return (y * jax.tree.map(jnp.conjugate, y)).real.sum()

        @jax.jit
        def local_grad(arr):
            y = jax.tree.map(jnp.fft.fftn, arr).transpose(transpose_back)
            return (y * jax.tree.map(jnp.conjugate, y)).real.sum()

        # Perform distributed FFT
        array_grad = jax.grad(spmd_grad)(global_array)
        print("Here is the gradient I'm getting", array_grad.shape)

        gathered_array = all_gather(global_array)
        gathered_grads = all_gather(array_grad)
        jax_grad = jax.grad(local_grad)(gathered_array)

        print(f'Shape of JAX array {jax_grad.shape}')
        # Check the gradients
        assert_allclose(jax_grad, gathered_grads, rtol=1e-5, atol=1e-5)

        print('-' * 40)
        print('Testing backward grad')

        @jax.jit
        def inv_spmd_grad(arr):
            y = jaxdecomp.fft.pifft3d(arr, backend=backend)
            return (y * jax.tree.map(jnp.conjugate, y)).real.sum()

        @jax.jit
        def inv_local_grad(arr):
            y = jax.tree.map(jnp.fft.ifftn, arr).transpose(transpose_back)
            return (y * jax.tree.map(jnp.conjugate, y)).real.sum()

        # Perform distributed FFT
        karray = jaxdecomp.fft.pfft3d(global_array, backend=backend)
        ifft_array_grad = jax.grad(inv_spmd_grad)(karray)
        print("Here is the gradient I'm getting", array_grad.shape)

        ifft_gathered_grads = all_gather(ifft_array_grad)
        jax_karray = jax.tree.map(jnp.fft.fftn, gathered_array).transpose(transpose_back)

        ifft_jax_grad = jax.grad(inv_local_grad)(jax_karray)

        print(f'Shape of JAX array {ifft_jax_grad.shape}')

        # Check the gradients
        assert_allclose(ifft_jax_grad, ifft_gathered_grads, rtol=1e-5, atol=1e-5)

        print('Grad check OK!')

        # Temporary solution because I need to find a way to retrigger the jit compile if the config changes
        jax.clear_caches()

    @pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_cudecomp_grad(self, pdims, global_shape, local_transpose, use_shardy):
        self.run_test(pdims, global_shape, local_transpose, backend='cuDecomp')

    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_jax_grad(self, pdims, global_shape, local_transpose, use_shardy):
        self.run_test(pdims, global_shape, local_transpose, backend='jax', use_shardy=use_shardy)


class TestFFTFreq:
    def run_test(self, pdims, global_shape, local_transpose, backend, use_shardy):
        if use_shardy and not ALLOW_SHARDY_PARTITIONER:
            pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

        print('*' * 80)
        print(f'Testing with pdims {pdims} and global shape {global_shape} and local transpose {local_transpose} use shardy {use_shardy}')

        jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)
        jax.config.update('jax_use_shardy_partitioner', use_shardy)
        if not local_transpose:
            pytest.skip(reason='Not implemented yet')

        global_array, mesh = create_spmd_array(global_shape, pdims)

        # Perform distributed gradient kernel
        karray = jaxdecomp.fft.pfft3d(global_array, backend=backend)
        kvec = jaxdecomp.fftfreq3d(karray)

        k_gradients = [k * karray for k in kvec]

        gradients = [jaxdecomp.fft.pifft3d(grad, backend=backend) for grad in k_gradients]

        gathered_gradients = [all_gather(grad) for grad in gradients]

        # perform local gradient kernel
        gathered_array = all_gather(global_array)
        jax_karray = jnp.fft.fftn(gathered_array)

        kz, ky, kx = [jnp.fft.fftfreq(jax_karray.shape[i]) * 2 * jnp.pi for i in range(3)]

        kz = kz.reshape(-1, 1, 1)
        ky = ky.reshape(1, -1, 1)
        kx = kx.reshape(1, 1, -1)

        kvec = [kz, ky, kx]

        jax_k_gradients = [k * jax_karray for k in kvec]
        jax_gradients = [jnp.fft.ifftn(grad) for grad in jax_k_gradients]

        # Check the gradients
        for i in range(3):
            assert_allclose(jax_gradients[i], gathered_gradients[i], rtol=1e-5, atol=1e-5)

        print('Gradient check OK!')

        # Trigger rejit in case local transpose is switched
        jax.clear_caches()

    @pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
    # Cartesian product tests
    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_cudecomp_fft(self, pdims, global_shape, local_transpose, use_shardy):
        self.run_test(pdims, global_shape, local_transpose, backend='cuDecomp' , use_shardy=use_shardy)

    # Cartesian product tests
    @pytest.mark.parametrize('local_transpose', local_transpose)  # Test with and without local transpose
    @pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
    @pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
    @pytest.mark.parametrize('global_shape', global_shapes)  # Test cubes, non-cubes and primes
    def test_jax_fft(self, pdims, global_shape, local_transpose, use_shardy):
        self.run_test(pdims, global_shape, local_transpose, backend='jax', use_shardy=use_shardy)


@pytest.mark.skipif(not is_on_cluster(), reason='Only run on cluster')
@pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
@pytest.mark.parametrize('pdims', decomp)
def test_huge_fft(pdims, use_shardy):
    if use_shardy and not ALLOW_SHARDY_PARTITIONER:
        pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

    with jax.experimental.disable_x64():
        jax.config.update('jax_use_shardy_partitioner', use_shardy)
        global_shape = (2048,) * 3  # Large cube to test integer overflow
        global_array, mesh = create_spmd_array(global_shape, pdims)
        # Perform distributed FFT
        karray = jaxdecomp.fft.pfft3d(global_array, backend='jax')
        # Perform inverse FFT
        rec_array = jaxdecomp.fft.pifft3d(karray, backend='jax')
        print('WORKED')
        # Check the reconstruction
        assert_allclose(global_array.real, rec_array.real, rtol=1e-5, atol=1e-5)
        assert_allclose(global_array.imag, rec_array.imag, rtol=1e-5, atol=1e-5)
        print('Reconstruction check OK!')


@pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
@pytest.mark.parametrize('pdims', decomp)
def test_vmap(pdims, use_shardy):
    if use_shardy and not ALLOW_SHARDY_PARTITIONER:
        pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

    jax.config.update('jax_use_shardy_partitioner', use_shardy)
    global_shape = (8, 8, 8)  # small shape because the shape in jacrev is (8 ,) * 6
    global_array, mesh = create_spmd_array(global_shape, pdims)

    fft_sharding = jaxdecomp.get_fft_output_sharding(global_array.sharding)

    batched = jnp.stack([global_array, global_array, global_array])

    v_pfft = jax.vmap(jaxdecomp.fft.pfft3d)

    batched_out = v_pfft(batched)

    assert batched_out.shape == (3, 8, 8, 8)
    assert batched_out[0].sharding.is_equivalent_to(fft_sharding, ndim=3)


@pytest.mark.parametrize('use_shardy', [False, True])  # Test with and without shardy
@pytest.mark.parametrize('pdims', decomp)  # Test with Slab and Pencil decompositions
def test_fwd_rev_grad(pdims, use_shardy):
    if use_shardy and not ALLOW_SHARDY_PARTITIONER:
        pytest.skip(reason='Shardy partitioner is not supported in this JAX version use at least JAX 0.7.0')

    jax.config.update('jax_use_shardy_partitioner', use_shardy)
    global_shape = (8, 8, 8)  # small shape because the shape in jacrev is (8 ,) * 6
    global_array, mesh = create_spmd_array(global_shape, pdims)

    # Fix with explicit sharding annotation
    in_sharding = global_array.sharding
    fft_sharding = jaxdecomp.get_fft_output_sharding(in_sharding)  # Assumes function exists

    def forward_with_annotation(array):
        array = jax.lax.with_sharding_constraint(array, in_sharding)
        return jaxdecomp.fft.pfft3d(array).real

    # Ensure jacfwd now runs
    try:
        fwd_grad = jax.jacfwd(forward_with_annotation)(global_array)
    except RuntimeError:
        pytest.fail('jacfwd still failed after annotating sharding!')

    # Ensure jacrev now runs
    try:
        rev_grad = jax.jacrev(forward_with_annotation)(global_array)
    except RuntimeError:
        pytest.fail('jacrev still failed after annotating sharding!')

    # 5. Fix grad with output sharding annotation
    def fft_reduce_with_annotation(array):
        array = jax.lax.with_sharding_constraint(array, in_sharding)
        res = jaxdecomp.fft.pfft3d(array).real
        res = jax.lax.with_sharding_constraint(res, fft_sharding)
        return res.sum()

    try:
        scalar_grad = jax.grad(fft_reduce_with_annotation)(global_array)
    except RuntimeError:
        pytest.fail('grad still failed after annotating output sharding!')

    assert fwd_grad.sharding.is_equivalent_to(fft_sharding, ndim=3)
    assert scalar_grad.sharding.is_equivalent_to(in_sharding, ndim=3)
    assert rev_grad[0, 0, 0, ...].sharding.is_equivalent_to(in_sharding, ndim=3)
