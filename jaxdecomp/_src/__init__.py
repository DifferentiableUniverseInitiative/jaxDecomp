from jax.lib import xla_client

from . import _jaxdecomp

# Registering ops for XLA
for name, fn in _jaxdecomp.registrations():
    xla_client.register_custom_call_target(name, fn, platform="cuda")
