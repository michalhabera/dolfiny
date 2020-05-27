from mpi4py import MPI


def pprint(str="", end="", flush=True, comm=MPI.COMM_WORLD):
    """Parallel print for MPI processes. Only rank==0 prints and flushes.

    Parameters
    ----------
    str: optional
        String to be printed
    end: optional
        Line ending
    flush: optional
        Flag for flushing output to stdout
    comm: optional
        MPI communicator

    """

    if comm.rank == 0:
        print(str, end, flush=True)
