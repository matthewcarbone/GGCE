# Run with mpiexec -np 4 python3 petsc4py_hello_world.py

from petsc4py import PETSc

rank = PETSc.COMM_WORLD.Get_rank()
size = PETSc.COMM_WORLD.Get_size()

if rank == 0:
    print("Note: PETSc scalar type should be complex!")
    print("If not, you've installed petsc incorrectly.")
    print(PETSc.ScalarType)

print(f"On rank {rank}/{size}")

# Should see something like
# On rank 1/4
# On rank 3/4
# On rank 0/4
# On rank 2/4

A = PETSc.Mat()
A.create(PETSc.COMM_WORLD)
A.setSizes([100, 100])
