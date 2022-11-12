from jax.lib import xla_client
from . import _jaxdecomp
init = _jaxdecomp.init
finalize = _jaxdecomp.finalize

from .ops import transposeXtoY

# Registering ops for XLA
for name, fn in _jaxdecomp.registrations().items():
    print("registering", name, fn)
    xla_client.register_custom_call_target(name, fn, platform="gpu")
