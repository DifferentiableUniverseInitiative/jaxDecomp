from conftest import (compare_sharding, create_spmd_array,
                      initialize_distributed, is_on_cluster)

initialize_distributed()
import jax

size = jax.device_count()

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import pytest
from jax.experimental.multihost_utils import process_allgather
from numpy.testing import assert_allclose

import jaxdecomp
from jaxdecomp._src import PENCILS, SLAB_XY, SLAB_YZ

all_gather = partial(process_allgather, tiled=True)

pencil_1 = (size // 2, size // (size // 2))  # 2x2 for V100 and 4x2 for A100
pencil_2 = (size // (size // 2), size // 2)  # 2x2 for V100 and 2x4 for A100

decomp = [(size, 1), (1, size), pencil_1, pencil_2]
global_shapes = [(8, 16, 32), (8, 8, 8), (29 * size, 19 * size, 17 * size)
                ]  # Cubes, non-cubes and primes
local_transpose = [True, False]


class TestFFTs:

  def run_test(self, pdims, global_shape, local_transpose, backend):

    print("*" * 80)
    print(
        f"Testing with pdims {pdims} and global shape {global_shape} and local transpose {local_transpose}"
    )
    if pdims[0] == 1:
      penciltype = SLAB_XY
    elif pdims[1] == 1:
      penciltype = SLAB_YZ
    else:
      penciltype = PENCILS
    print(f"Decomposition type {penciltype}")

    jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)

    global_array, mesh = create_spmd_array(global_shape, pdims)

    # Perform distributed FFT
    karray = jaxdecomp.fft.pfft3d(global_array, backend=backend)
    # Perform inverse FFT
    rec_array = jaxdecomp.fft.pifft3d(karray, backend=backend)

    print(f"orignal shard {global_array.sharding.spec}")
    print(f"sharding of karray {karray.sharding.spec}")
    print(f"sharding of rec_array {rec_array.sharding.spec}")

    # assert compare_sharding(karray.sharding, dist_jax_karray.sharding)
    # assert compare_sharding(rec_array.sharding, dist_jax_rec_array.sharding)
    # assert compare_sharding(global_array.sharding, dist_jax_rec_array.sharding)

    # Check the forward FFT
    gathered_array = all_gather(global_array)
    gathered_karray = all_gather(karray)
    gathered_rec_array = all_gather(rec_array)

    jax_karray = jnp.fft.fftn(gathered_array)
    jax_rec_array = jnp.fft.ifftn(jax_karray, norm='forward')
    # Check the forward FFT
    if penciltype == SLAB_YZ:
      transpose_back = [2, 0, 1]
    else:
      transpose_back = [1, 2, 0]
    if not local_transpose:
      transpose_back = [0, 1, 2]
    elif jaxdecomp.config.transpose_axis_contiguous_2:
      transpose_back = [1, 2, 0]

    # Check reconstructed array
    assert_allclose(
        gathered_array.real, gathered_rec_array.real, rtol=1e-5, atol=1e-5)
    assert_allclose(
        gathered_array.imag, gathered_rec_array.imag, rtol=1e-5, atol=1e-5)

    print(f"Reconstruction check OK!")

    jax_karray_transposed = jax_karray.transpose(transpose_back)
    assert_allclose(
        gathered_karray.real, jax_karray_transposed.real, rtol=1e-5, atol=1e-5)
    assert_allclose(
        gathered_karray.imag, jax_karray_transposed.imag, rtol=1e-5, atol=1e-5)

    print(f"FFT with transpose check OK!")

    # Trigger rejit in case local transpose is switched
    jax.clear_caches()

  @pytest.mark.skipif(not is_on_cluster(), reason="Only run on cluster")
  # Cartesian product tests
  @pytest.mark.parametrize(
      "local_transpose",
      local_transpose)  # Test with and without local transpose
  @pytest.mark.parametrize("pdims",
                           decomp)  # Test with Slab and Pencil decompositions
  @pytest.mark.parametrize("global_shape",
                           global_shapes)  # Test cubes, non-cubes and primes
  def test_cudecomp_fft(self, pdims, global_shape, local_transpose):
    self.run_test(pdims, global_shape, local_transpose, backend="cuDecomp")

  # Cartesian product tests
  @pytest.mark.parametrize(
      "local_transpose",
      local_transpose)  # Test with and without local transpose
  @pytest.mark.parametrize("pdims",
                           decomp)  # Test with Slab and Pencil decompositions
  @pytest.mark.parametrize("global_shape",
                           global_shapes)  # Test cubes, non-cubes and primes
  def test_jax_fft(self, pdims, global_shape, local_transpose):
    self.run_test(pdims, global_shape, local_transpose, backend="jax")


class TestFFTsGrad:

  def run_test(self, pdims, global_shape, local_transpose, backend):
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
    elif jaxdecomp.config.transpose_axis_contiguous_2:
      transpose_back = [1, 2, 0]

    print("*" * 80)
    # Cartesian product tests
    print(
        f"Testing with pdims {pdims} and global shape {global_shape} and local transpose {local_transpose}"
    )
    jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)
    global_array, mesh = create_spmd_array(global_shape, pdims)

    print("-" * 40)
    print(f"Testing fwd grad")

    @jax.jit
    def spmd_grad(arr):
      y = jaxdecomp.fft.pfft3d(arr, backend=backend)
      y = (y * jnp.conjugate(y)).real.sum()
      return y

      # Perform local FFT
    @jax.jit
    def local_grad(arr):
      y = jnp.fft.fftn(arr).transpose(transpose_back)
      y = (y * jnp.conjugate(y)).real.sum()
      return y

    # Perform distributed FFT
    array_grad = jax.grad(spmd_grad)(global_array)
    print("Here is the gradient I'm getting", array_grad.shape)

    gathered_array = all_gather(global_array)
    gathered_grads = all_gather(array_grad)
    jax_grad = jax.grad(local_grad)(gathered_array)

    print(f"Shape of JAX array {jax_grad.shape}")
    # Check the gradients
    assert_allclose(jax_grad, gathered_grads, rtol=1e-5, atol=1e-5)

    print("-" * 40)
    print(f"Testing backward grad")

    @jax.jit
    def inv_spmd_grad(arr):
      y = jaxdecomp.fft.pifft3d(arr, backend=backend)
      y = (y * jnp.conjugate(y)).real.sum()
      return y

    @jax.jit
    def inv_local_grad(arr):
      y = jnp.fft.ifftn(arr).transpose(transpose_back)
      return (y * jnp.conjugate(y)).real.sum()

    # Perform distributed FFT
    karray = jaxdecomp.fft.pfft3d(global_array, backend=backend)
    ifft_array_grad = jax.grad(inv_spmd_grad)(karray)
    print("Here is the gradient I'm getting", array_grad.shape)

    ifft_gathered_grads = all_gather(ifft_array_grad)
    jax_karray = jnp.fft.fftn(gathered_array).transpose(transpose_back)

    ifft_jax_grad = jax.grad(inv_local_grad)(jax_karray)

    print(f"Shape of JAX array {ifft_jax_grad.shape}")

    # Check the gradients
    assert_allclose(ifft_jax_grad, ifft_gathered_grads, rtol=1e-5, atol=1e-5)

    print("Grad check OK!")

    # Temporary solution because I need to find a way to retrigger the jit compile if the config changes
    jax.clear_caches()

  @pytest.mark.skipif(not is_on_cluster(), reason="Only run on cluster")
  @pytest.mark.parametrize(
      "local_transpose",
      local_transpose)  # Test with and without local transpose
  @pytest.mark.parametrize("pdims",
                           decomp)  # Test with Slab and Pencil decompositions
  @pytest.mark.parametrize("global_shape",
                           global_shapes)  # Test cubes, non-cubes and primes
  def test_cudecomp_grad(self, pdims, global_shape, local_transpose):
    self.run_test(pdims, global_shape, local_transpose, backend="cuDecomp")

  @pytest.mark.parametrize(
      "local_transpose",
      local_transpose)  # Test with and without local transpose
  @pytest.mark.parametrize("pdims",
                           decomp)  # Test with Slab and Pencil decompositions
  @pytest.mark.parametrize("global_shape",
                           global_shapes)  # Test cubes, non-cubes and primes
  def test_jax_grad(self, pdims, global_shape, local_transpose):
    self.run_test(pdims, global_shape, local_transpose, backend="jax")


class TestFFTFreq:

  def run_test(self, pdims, global_shape, local_transpose, backend):

    print("*" * 80)
    print(
        f"Testing with pdims {pdims} and global shape {global_shape} and local transpose {local_transpose}"
    )

    jaxdecomp.config.update('transpose_axis_contiguous', local_transpose)
    if not local_transpose:
      pytest.skip(reason="Not implemented yet")

    global_array, mesh = create_spmd_array(global_shape, pdims)

    # Perform distributed gradient kernel
    karray = jaxdecomp.fft.pfft3d(global_array, backend=backend)
    kvec = jaxdecomp.fftfreq3d(karray)

    k_gradients = [k * karray for k in kvec]

    gradients = [
        jaxdecomp.fft.pifft3d(grad, backend=backend) for grad in k_gradients
    ]

    gathered_gradients = [all_gather(grad) for grad in gradients]

    # perform local gradient kernel
    gathered_array = all_gather(global_array)
    jax_karray = jnp.fft.fftn(gathered_array)

    kz, ky, kx = [
        jnp.fft.fftfreq(jax_karray.shape[i]) * 2 * jnp.pi for i in range(3)
    ]

    kz = kz.reshape(-1, 1, 1)
    ky = ky.reshape(1, -1, 1)
    kx = kx.reshape(1, 1, -1)

    kvec = [kz, ky, kx]

    jax_k_gradients = [k * jax_karray for k in kvec]
    jax_gradients = [jnp.fft.ifftn(grad) for grad in jax_k_gradients]

    # Check the gradients
    for i in range(3):
      assert_allclose(
          jax_gradients[i], gathered_gradients[i], rtol=1e-5, atol=1e-5)

    print(f"Gradient check OK!")

    # Trigger rejit in case local transpose is switched
    jax.clear_caches()

  @pytest.mark.skipif(not is_on_cluster(), reason="Only run on cluster")
  # Cartesian product tests
  @pytest.mark.parametrize(
      "local_transpose",
      local_transpose)  # Test with and without local transpose
  @pytest.mark.parametrize("pdims",
                           decomp)  # Test with Slab and Pencil decompositions
  @pytest.mark.parametrize("global_shape",
                           global_shapes)  # Test cubes, non-cubes and primes
  def test_cudecomp_fft(self, pdims, global_shape, local_transpose):
    self.run_test(pdims, global_shape, local_transpose, backend="cuDecomp")

  # Cartesian product tests
  @pytest.mark.parametrize(
      "local_transpose",
      local_transpose)  # Test with and without local transpose
  @pytest.mark.parametrize("pdims",
                           decomp)  # Test with Slab and Pencil decompositions
  @pytest.mark.parametrize("global_shape",
                           global_shapes)  # Test cubes, non-cubes and primes
  def test_jax_fft(self, pdims, global_shape, local_transpose):
    self.run_test(pdims, global_shape, local_transpose, backend="jax")
