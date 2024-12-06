from jax.lib import xla_client
from jaxdecomplib import _jaxdecomp

init = _jaxdecomp.init
finalize = _jaxdecomp.finalize
get_pencil_info = _jaxdecomp.get_pencil_info
get_autotuned_config = _jaxdecomp.get_autotuned_config
make_config = _jaxdecomp.GridConfig

# Loading the comm configuration flags at the top level
from jaxdecomplib._jaxdecomp import HaloCommBackend  # yapf: disable
from jaxdecomplib._jaxdecomp import TransposeCommBackend  # yapf: disable
from jaxdecomplib._jaxdecomp import (  # dummy line to avoid yapf reformatting
    HALO_COMM_MPI, HALO_COMM_MPI_BLOCKING, HALO_COMM_NCCL, HALO_COMM_NVSHMEM,
    HALO_COMM_NVSHMEM_BLOCKING, NO_DECOMP, PENCILS, SLAB_XY, SLAB_YZ,
    TRANSPOSE_COMM_MPI_A2A, TRANSPOSE_COMM_MPI_P2P, TRANSPOSE_COMM_MPI_P2P_PL,
    TRANSPOSE_COMM_NCCL, TRANSPOSE_COMM_NCCL_PL, TRANSPOSE_COMM_NVSHMEM,
    TRANSPOSE_COMM_NVSHMEM_PL, TRANSPOSE_XY, TRANSPOSE_YX, TRANSPOSE_YZ,
    TRANSPOSE_ZY)

# Registering ops for XLA
for name, fn in _jaxdecomp.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")

__all__ = [
    "init",
    "finalize",
    "get_pencil_info",
    "get_autotuned_config",
    "make_config",
    "TRANSPOSE_XY",
    "TRANSPOSE_YX",
    "TRANSPOSE_YZ",
    "TRANSPOSE_ZY",
    "SLAB_XY",
    "SLAB_YZ",
    "PENCILS",
    "NO_DECOMP",
]
