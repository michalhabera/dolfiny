import dolfin
import ufl
import dolfiny.function
import pytest


@pytest.fixture
def mesh():
    return dolfin.generation.UnitSquareMesh(dolfin.MPI.comm_world, 5, 5)


@pytest.fixture
def V1(mesh):
    return dolfin.FunctionSpace(mesh, ("P", 1))


@pytest.fixture
def V2(mesh):
    return dolfin.FunctionSpace(mesh, ("P", 2))


def test_ufl_block_extraction(V1, V2, mesh):
    u1, v1 = ufl.TrialFunction(V1), ufl.TestFunction(V1)
    u2, v2 = ufl.TrialFunction(V2), ufl.TestFunction(V2)

    a = u1 * v1 * ufl.dx(mesh) + ufl.grad(u1 * v2)[0] * ufl.dx(mesh)

    ablocks = dolfiny.function.extract_blocks(a, [v1, v2], [u1, u2])

    assert ablocks[1][1].empty()
    assert ablocks[0][1].empty()
