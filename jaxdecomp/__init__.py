from dataclasses import dataclass
from ._src import init, finalize, get_pencil_info, make_config, transposeXtoY, transposeYtoX, transposeYtoZ, transposeZtoY, halo_exchange
from ._src import _jaxdecomp
import jaxdecomp.fft as fft
from jaxdecomp.fft import pfft3d, pifft3d

__all__ = [
    "init",
    "finalize",
    "get_pencil_info",
    "make_config",
    "halo_exchange",
    "pfft3d",
    "pifft3d",
    "transposeXtoY",
    "transposeYtoZ",
    "transposeZtoY",
    "transposeYtoX",
]


@dataclass
class JAXDecompConfig:
    """Class for storing the configuration state of the library."""
    halo_comm_backend: _jaxdecomp.HaloCommBackend = _jaxdecomp.HALO_COMM_NCCL
    transpose_comm_backend: _jaxdecomp.TransposeCommBackend = _jaxdecomp.TRANSPOSE_COMM_NCCL
    
    def update(self, key, value):
      if hasattr(self, key):
        setattr(self, key, value)
      else:
        raise ValueError("key %s is not a valid configuration key"%key)

# Declare the global configuration object
config = JAXDecompConfig()
