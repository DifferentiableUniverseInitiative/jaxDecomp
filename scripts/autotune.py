from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import jaxdecomp

pdims = (0, 0)
global_shape = (1024, 1024, 1024)

config = jaxdecomp.make_config()
config.pdims = pdims
config.gdims = global_shape[::-1]
config.halo_comm_backend = jaxdecomp.config.halo_comm_backend
config.transpose_comm_backend = jaxdecomp.config.transpose_comm_backend

# Run autotune
tuned_config = jaxdecomp.get_autotuned_config(config, False, False, True, True,
                                              (32, 32, 32), (True, True, True))

print(rank, "*** Results of optimization ***")
print(rank, "pdims", tuned_config.pdims)
print(rank, "halo_comm_backend", tuned_config.halo_comm_backend)
print(rank, "transpose_comm_backend", tuned_config.transpose_comm_backend)
