import numpy as np

import dolfinx
import dolfiny.snesblockproblem
import pytest
import ufl
from petsc4py import PETSc


def test_monolithic(V1, V2, squaremesh_5):
    mesh = squaremesh_5

    Wel = ufl.MixedElement([V1.ufl_element(), V2.ufl_element()])
    W = dolfinx.FunctionSpace(mesh, Wel)

    u = dolfinx.Function(W)
    u0, u1 = ufl.split(u)

    v = ufl.TestFunction(W)
    v0, v1 = ufl.split(v)

    Phi = (ufl.sin(u0) - 0.5) ** 2 * ufl.dx(mesh) + (4.0 * u0 - u1) ** 2 * ufl.dx(mesh)

    F = ufl.derivative(Phi, u, v)

    opts = PETSc.Options()

    opts.setValue('snes_type', 'newtonls')
    opts.setValue('snes_linesearch_type', 'basic')

    opts.setValue('snes_rtol', 1.0e-10)
    opts.setValue('snes_max_it', 20)

    opts.setValue('ksp_type', 'preonly')
    opts.setValue('pc_type', 'lu')
    opts.setValue('pc_factor_mat_solver_type', 'mumps')

    problem = dolfiny.snesblockproblem.SNESBlockProblem([F], [u], opts=opts)
    sol, = problem.solve()

    u0, u1 = sol.split()
    u0 = u0.collapse()
    u1 = u1.collapse()

    assert np.isclose((u0.vector - np.arcsin(0.5)).norm(), 0.0)
    assert np.isclose((u1.vector - 4.0 * np.arcsin(0.5)).norm(), 0.0)


@pytest.mark.parametrize("nest", [False])
def test_block(V1, V2, squaremesh_5, nest):
    mesh = squaremesh_5

    u0 = dolfinx.Function(V1, name="u0")
    u1 = dolfinx.Function(V2, name="u1")

    v0 = ufl.TestFunction(V1)
    v1 = ufl.TestFunction(V2)

    Phi = (ufl.sin(u0) - 0.5) ** 2 * ufl.dx(mesh) + (4.0 * u0 - u1) ** 2 * ufl.dx(mesh)

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
        opts.setValue('pc_type', 'fieldsplit')
        opts.setValue('fieldsplit_pc_type', 'lu')
        opts.setValue('ksp_rtol', 1.0e-10)
    else:
        opts.setValue('ksp_type', 'preonly')
        opts.setValue('pc_type', 'lu')
        opts.setValue('pc_factor_mat_solver_type', 'mumps')

    problem = dolfiny.snesblockproblem.SNESBlockProblem(F, u, opts=opts, nest=nest)
    sol = problem.solve()

    assert problem.snes.getConvergedReason() > 0
    assert np.isclose((sol[0].vector - np.arcsin(0.5)).norm(), 0.0)
    assert np.isclose((sol[1].vector - 4.0 * np.arcsin(0.5)).norm(), 0.0)
