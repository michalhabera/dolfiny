import dolfin
import pytest


@pytest.fixture(scope="module")
def squaremesh_5():
    return dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)


@pytest.fixture(scope="module")
def V1(squaremesh_5):
    return dolfin.FunctionSpace(squaremesh_5, ("P", 1))


@pytest.fixture(scope="module")
def V2(squaremesh_5):
    return dolfin.FunctionSpace(squaremesh_5, ("P", 2))
