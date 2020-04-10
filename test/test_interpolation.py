import time

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

CG2 = FunctionSpace(mesh, ("CG", 2))
VCG1 = FunctionSpace(mesh, DG0.mesh.ufl_domain().ufl_coordinate_element())


def test_expr():
    f = Function(CG1)
    f.vector.set(1.0)
    f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    x = ufl.SpatialCoordinate(mesh)
    expr1 = ufl.grad(ufl.grad(ufl.sinh(x[0]) + x[1] ** 3))

    # Interpolate
    h_interp = Function(TCG1)
    interpolation.interpolate(expr1, h_interp)

    h_project = Function(TCG1)
    projection.project(expr1, h_project)

    h_project.vector.axpy(-1.0, h_interp.vector)
    h_project.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    assert h_project.vector.norm() < 0.1

    expr2 = h_interp
    h_interp2 = Function(TDG0)
    interpolation.interpolate(expr2, h_interp2)

    h_project2 = Function(TDG0)
    projection.project(expr2, h_project2)

    h_project2.vector.axpy(-1.0, h_interp2.vector)
    h_project2.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    assert h_project2.vector.norm() < 0.1


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
