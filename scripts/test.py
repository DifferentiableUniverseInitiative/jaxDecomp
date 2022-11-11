from mpi4py import MPI
import python_example

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("Here we go!")


python_example.init()

print("cool, initialized")

python_example.finalize()

print("Done!")
