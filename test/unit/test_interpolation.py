import time

import dolfinx
import dolfiny
import numpy
import pytest
import ufl
from mpi4py import MPI
from petsc4py import PETSc

N = 10
mesh = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, N, N, N)

DG0 = dolfinx.fem.FunctionSpace(mesh, ("DP", 0))
DG1 = dolfinx.fem.FunctionSpace(mesh, ("DP", 1))
CG1 = dolfinx.fem.FunctionSpace(mesh, ("P", 1))
TCG1 = dolfinx.fem.TensorFunctionSpace(mesh, ("P", 1))
TDG0 = dolfinx.fem.TensorFunctionSpace(mesh, ("DP", 0))
TDG1s = dolfinx.fem.TensorFunctionSpace(mesh, ("DP", 1), symmetry=True)
TDG2s = dolfinx.fem.TensorFunctionSpace(mesh, ("DP", 2), symmetry=True)

CG2 = dolfinx.fem.FunctionSpace(mesh, ("P", 2))
VCG1 = dolfinx.fem.FunctionSpace(mesh, DG0.mesh.ufl_domain().ufl_coordinate_element())

f = dolfinx.fem.Function(TCG1)
f.vector.set(1.0)
f.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

x = ufl.SpatialCoordinate(mesh)
expr1 = f[0, 0] * ufl.grad(ufl.grad(ufl.sinh(x[0]) + x[1] ** 3))


@pytest.mark.parametrize("V", [TCG1, TDG0, TDG1s, TDG2s])
def test_expr(V):

    # Interpolate
    h_interp = dolfinx.fem.Function(V)
    dolfiny.interpolation.interpolate(expr1, h_interp)

    # Project
    h_project = dolfinx.fem.Function(V)
    dolfiny.projection.project(expr1, h_project)

    # Compare
    h_project.vector.axpy(-1.0, h_interp.vector)
    h_project.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    assert h_project.vector.norm() < 0.15


def test_diff():
    f = dolfinx.fem.Function(CG1)
    f.vector.set(1.0)
    f.vector.ghostUpdate()

    x = ufl.SpatialCoordinate(mesh)
    expr = x[0] + x[1] + f

    h_project = dolfinx.fem.Function(CG1)
    dolfiny.projection.project(expr, h_project)

    h_interp = dolfinx.fem.Function(CG1)
    dolfiny.interpolation.interpolate(expr, h_interp)

    diff = dolfinx.fem.Function(CG1)
    dolfiny.interpolation.interpolate(h_interp - h_project, diff)

    assert diff.vector.norm(3) < 1.0e-3


def test_perf():
    N = 500
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    P1 = dolfinx.fem.FunctionSpace(mesh, ("P", 1))
    P2 = dolfinx.fem.FunctionSpace(mesh, ("P", 2))

    u1 = dolfinx.fem.Function(P1)
    u2 = dolfinx.fem.Function(P2)

    t0 = time.time()
    dolfiny.interpolation.interpolate(u1, u2)
    print("Cold", time.time() - t0)

    t0 = time.time()
    dolfiny.interpolation.interpolate(u1, u2)
    print("Hot", time.time() - t0)


def test_linear_combination():
    N = 100
    mesh = dolfinx.mesh.create_unit_square(MPI.COMM_WORLD, N, N)

    P1 = dolfinx.fem.FunctionSpace(mesh, ("P", 1))

    u1 = dolfinx.fem.Function(P1)
    u2 = dolfinx.fem.Function(P1)
    u3 = dolfinx.fem.Function(P1)

    f = dolfinx.fem.Function(P1)
    g = dolfinx.fem.Function(P1)

    with u1.vector.localForm() as local:
        local.set(1.0)
    with u2.vector.localForm() as local:
        local.set(2.0)
    with u3.vector.localForm() as local:
        local.set(3.0)

    c1 = dolfinx.fem.Constant(mesh, 3.14)
    c2 = dolfinx.fem.Constant(mesh, 0.1)

    expr = (c1 + c2) * (2.0 * (3.0 * u1 + u2) - u3 / 3.0 + 4 * (u1 + c2 * u3))

    t0 = time.time()
    dolfiny.interpolation.interpolate(expr, f)
    print(f"Linear combination {time.time() - t0}")

    t0 = time.time()
    compiled_expression = dolfiny.interpolation.CompiledExpression(expr, g.function_space.ufl_element())
    dolfiny.interpolation.interpolate_compiled(compiled_expression, g)
    print(f"Compiled interpolation {time.time() - t0}")

    assert numpy.isclose((g.vector - f.vector).norm(), 0.0)
