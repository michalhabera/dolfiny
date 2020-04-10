from mpi4py import MPI

import dolfinx
import pytest


@pytest.fixture(scope="module")
def squaremesh_5():
    return dolfinx.generation.UnitSquareMesh(MPI.COMM_WORLD, 5, 5)


@pytest.fixture(scope="module")
def V1(squaremesh_5):
    return dolfinx.FunctionSpace(squaremesh_5, ("P", 1))


@pytest.fixture(scope="module")
def V2(squaremesh_5):
    return dolfinx.FunctionSpace(squaremesh_5, ("P", 2))


@pytest.fixture(scope="module")
def vV1(squaremesh_5):
    return dolfinx.VectorFunctionSpace(squaremesh_5, ("P", 1))
