from jax.lib import xla_client

from . import _jaxdecomp

init = _jaxdecomp.init
finalize = _jaxdecomp.finalize
get_pencil_info = _jaxdecomp.get_pencil_info
get_autotuned_config = _jaxdecomp.get_autotuned_config
make_config = _jaxdecomp.GridConfig

# Loading the comm configuration flags at the top level
from ._jaxdecomp import (HALO_COMM_MPI, HALO_COMM_MPI_BLOCKING, HALO_COMM_NCCL,
                         HALO_COMM_NVSHMEM, HALO_COMM_NVSHMEM_BLOCKING,
                         TRANSPOSE_COMM_MPI_A2A, TRANSPOSE_COMM_MPI_P2P,
                         TRANSPOSE_COMM_MPI_P2P_PL, TRANSPOSE_COMM_NCCL,
                         TRANSPOSE_COMM_NCCL_PL, TRANSPOSE_COMM_NVSHMEM,
                         TRANSPOSE_COMM_NVSHMEM_PL, TRANSPOSE_XY, TRANSPOSE_YX,
                         TRANSPOSE_YZ, TRANSPOSE_ZY, HaloCommBackend,
                         TransposeCommBackend)

# Registering ops for XLA
for name, fn in _jaxdecomp.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")

from .fft import pfft
from .halo import halo_exchange
from .padding import slice_pad, slice_unpad
from .transpose import (transposeXtoY, transposeYtoX, transposeYtoZ,
                        transposeZtoY)
