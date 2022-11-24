from jax.lib import xla_client
from . import _jaxdecomp

init = _jaxdecomp.init
finalize = _jaxdecomp.finalize
get_pencil_info = _jaxdecomp.get_pencil_info
make_config = _jaxdecomp.GridConfig

from .transpose import transposeXtoY, transposeYtoX, transposeYtoZ, transposeZtoY
from .fft import pfft

# Registering ops for XLA
for name, fn in _jaxdecomp.registrations().items():
  xla_client.register_custom_call_target(name, fn, platform="gpu")
