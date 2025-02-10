
def error_during_jacfwd(function_name):
    raise ValueError(f"""
        Input sharding was found to be none while lowering the SPMD rule.
        You are likely calling jacfwd with pfft as the first function.
        due to a bug in JAX, the sharding is not correctly passed to the SPMD rule.
        You need to annotate the sharding before calling {function_name}.
        please check the caveat documentation, jacfwd section
        """)


def error_during_jacrev(function_name):
    raise ValueError(f"""
        Input sharding was found to be none while lowering the SPMD rule.
        You are likely calling jacrev with pfft as the first function.
        due to a bug in JAX, the sharding is not correctly passed to the SPMD rule.
        You need to annotate the sharding After calling {function_name}.
        please check the caveat documentation, grad section
        """)