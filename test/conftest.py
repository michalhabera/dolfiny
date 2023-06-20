import gc

import dolfinx
import pytest
from mpi4py import MPI


def pytest_runtest_teardown(item):
    """Collect garbage after every test to force calling
    destructors which might be collective"""

    # Do the normal teardown
    item.teardown()

    # Collect the garbage (call destructors collectively)
    del item
    # NOTE: How are we sure that 'item' does not hold references
    #       to temporaries and someone else does not hold a reference
    #       to 'item'?! Well, it seems that it works...
    gc.collect()
    comm = MPI.COMM_WORLD
    comm.Barrier()


def pytest_configure(config):
    config.addinivalue_line("markers", "convergence: mark as convergence test")
    config.addinivalue_line("markers", "postprocess: mark as postprocess step")


@pytest.fixture(scope="module")
def squaremesh_5():
    return dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, 5, 5)


@pytest.fixture(scope="module")
def V1(squaremesh_5):
    return dolfinx.fem.FunctionSpace(squaremesh_5, ("P", 1))


@pytest.fixture(scope="module")
def V2(squaremesh_5):
    return dolfinx.fem.FunctionSpace(squaremesh_5, ("P", 2))


@pytest.fixture(scope="module")
def vV1(squaremesh_5):
    return dolfinx.fem.VectorFunctionSpace(squaremesh_5, ("P", 1))


@pytest.fixture(scope="module")
def vV2(squaremesh_5):
    return dolfinx.fem.VectorFunctionSpace(squaremesh_5, ("P", 2))
