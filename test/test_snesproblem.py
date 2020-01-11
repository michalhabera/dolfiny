from petsc4py import PETSc
import numpy as np
import dolfin
import ufl
import dolfiny.snesproblem
import pytest


def test_singleblock(V1, squaremesh_5):
    mesh = squaremesh_5
    V = V1

    u = dolfin.Function(V)
    v = ufl.TestFunction(V)

    Phi = (u - 0.25) ** 2 * ufl.dx(mesh)

    F = ufl.derivative(Phi, u, v)

    opts = PETSc.Options()

    opts.setValue('snes_type', 'newtonls')
    opts.setValue('snes_linesearch_type', 'basic')
    opts.setValue('snes_rtol', 1.0e-08)
    opts.setValue('snes_max_it', 12)

    opts.setValue('ksp_type', 'preonly')
    opts.setValue('pc_type', 'lu')
    opts.setValue('pc_factor_mat_solver_type', 'mumps')

    problem = dolfiny.snesproblem.SNESProblem(F, u, opts=opts)
    problem.snes.solve(None, problem.u.vector)
    assert np.isclose((problem.u.vector - 0.25).norm(), 0.0)


@pytest.mark.parametrize("nest", [True, False])
def test_block(V1, V2, squaremesh_5, nest):
    mesh = squaremesh_5

    u0 = dolfin.Function(V1, name="u0")
    u1 = dolfin.Function(V2, name="u1")

    v0 = ufl.TestFunction(V1)
    v1 = ufl.TestFunction(V2)

    Phi = (u0 - 0.25) ** 2 * ufl.dx(mesh) + (4.0 * u0 - u1) ** 2 * ufl.dx(mesh)

    F0 = ufl.derivative(Phi, u0, v0)
    F1 = ufl.derivative(Phi, u1, v1)

    F = [F0, F1]
    u = [u0, u1]

    opts = PETSc.Options()

    opts.setValue('snes_type', 'newtonls')
    opts.setValue('snes_linesearch_type', 'basic')
    opts.setValue('snes_rtol', 1.0e-08)
    opts.setValue('snes_max_it', 12)

    if nest:
        opts.setValue('ksp_type', 'cg')
        opts.setValue('pc_type', 'none')
        opts.setValue('ksp_rtol', 1.0e-10)
    else:
        opts.setValue('ksp_type', 'preonly')
        opts.setValue('pc_type', 'lu')
        opts.setValue('pc_factor_mat_solver_type', 'mumps')

    problem = dolfiny.snesproblem.SNESBlockProblem(F, u, opts=opts, nest=nest)
    sol = problem.solve()

    assert np.isclose((sol[0].vector - 0.25).norm(), 0.0)
    assert np.isclose((sol[1].vector - 1.0).norm(), 0.0)
