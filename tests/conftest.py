import jax
import pytest

import jaxdecomp

setup_done = False


def initialize_distributed():
  global setup_done
  if not setup_done:
    jax.distributed.initialize()
    setup_done = True


@pytest.fixture(scope="session", autouse=True)
def setup_and_teardown_session():
  # Code to run at the start of the session
  print("Starting session...")
  initialize_distributed()
  # Setup code here
  # e.g., connecting to a database, initializing some resources, etc.

  yield

  # Code to run at the end of the session
  print("Ending session...")
  jaxdecomp.finalize()
  jax.distributed.shutdown()

  # Teardown code here
  # e.g., closing connections, cleaning up resources, etc.
