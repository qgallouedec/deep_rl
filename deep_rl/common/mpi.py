from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def mpi_mean(x):
    return mpi_sum(x) / size


def mpi_sum(x):
    return MPI.COMM_WORLD.allreduce(x, MPI.SUM)


def mpi_bcast(x):
    pass


def gather(x):
    return MPI.COMM_WORLD.allreduce(x, MPI.SUM)


if __name__ == "__main__":
    # print(mpi_mean(np.array([0, rank, rank+1])))

    # if rank == 0:
    #     x='helle'
    #     MPI.COMM_WORLD.bcast(x)
    # else:
    #     x = MPI.COMM_WORLD.bcast(None)
    # print(x)

    a = [0, rank, rank + 1]
    print(a)
    a = MPI.COMM_WORLD.allreduce(a, MPI.SUM)
    if rank == 0:
        b = np.mean(np.array(a))
        MPI.COMM_WORLD.bcast(b)
    else:
        b = MPI.COMM_WORLD.bcast(None)
    print(b)
    # else:
    #     MPI.COMM_WORLD.gather(a)
