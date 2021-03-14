import time
import pytest
import numpy

from dolfinx.generation import UnitCubeMesh, UnitSquareMesh
from mpi4py import MPI
import ufl
from dolfinx import Function, FunctionSpace, TensorFunctionSpace
from dolfiny import interpolation, projection
from petsc4py import PETSc


N = 10
mesh = UnitCubeMesh(MPI.COMM_WORLD, N, N, N)

DG0 = FunctionSpace(mesh, ("DG", 0))
DG1 = FunctionSpace(mesh, ("DG", 1))
CG1 = FunctionSpace(mesh, ("CG", 1))
TCG1 = TensorFunctionSpace(mesh, ("CG", 1))
TDG0 = TensorFunctionSpace(mesh, ("DG", 0))
TDG1s = TensorFunctionSpace(mesh, ("DG", 1), symmetry=True)
TDG2s = TensorFunctionSpace(mesh, ("DG", 2), symmetry=True)

CG2 = FunctionSpace(mesh, ("CG", 2))
VCG1 = FunctionSpace(mesh, DG0.mesh.ufl_domain().ufl_coordinate_element())

f = Function(TCG1)
f.vector.set(1.0)
f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

x = ufl.SpatialCoordinate(mesh)
expr1 = f[0, 0] * ufl.grad(ufl.grad(ufl.sinh(x[0]) + x[1] ** 3))


@pytest.mark.parametrize("V", [TCG1, TDG0, TDG1s, TDG2s])
def test_expr(V):

    # Interpolate
    h_interp = Function(V)
    interpolation.interpolate(expr1, h_interp)

    # Project
    h_project = Function(V)
    projection.project(expr1, h_project)

    # Compare
    h_project.vector.axpy(-1.0, h_interp.vector)
    h_project.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    assert h_project.vector.norm() < 0.15


def test_diff():
    f = Function(CG1)
    f.vector.set(1.0)
    f.vector.ghostUpdate()

    x = ufl.SpatialCoordinate(mesh)
    expr = x[0] + x[1] + f

    h_project = Function(CG1)
    projection.project(expr, h_project)

    h_interp = Function(CG1)
    interpolation.interpolate(expr, h_interp)

    diff = Function(CG1)
    interpolation.interpolate(h_interp - h_project, diff)

    assert diff.vector.norm(3) < 1.0e-3


def test_perf():
    N = 500
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N)

    P1 = FunctionSpace(mesh, ("P", 1))
    P2 = FunctionSpace(mesh, ("P", 2))

    u1 = Function(P1)
    u2 = Function(P2)

    t0 = time.time()
    interpolation.interpolate(u1, u2)
    print("Cold", time.time() - t0)

    t0 = time.time()
    interpolation.interpolate(u1, u2)
    print("Hot", time.time() - t0)


def test_linear_combination():
    N = 100
    mesh = UnitSquareMesh(MPI.COMM_WORLD, N, N)

    P1 = FunctionSpace(mesh, ("P", 1))

    u1 = Function(P1)
    u2 = Function(P1)
    u3 = Function(P1)

    f = Function(P1)
    g = Function(P1)

    with u1.vector.localForm() as local:
        local.set(1.0)
    with u2.vector.localForm() as local:
        local.set(2.0)
    with u3.vector.localForm() as local:
        local.set(3.0)

    expr = 2.0 * (3.0 * u1 + u2) - u3 / 3.0 + 4.0 * (u1 + u3)

    t0 = time.time()
    interpolation.interpolate(expr, f)
    print(f"Linear combination {time.time() - t0}")

    t0 = time.time()
    compiled_expression = interpolation.CompiledExpression(expr, g.function_space.ufl_element())
    interpolation.interpolate_compiled(compiled_expression, g)
    print(f"Compiled interpolation {time.time() - t0}")

    assert numpy.isclose((g.vector - f.vector).norm(), 0.0)