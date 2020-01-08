from petsc4py import PETSc
import numpy as np
import dolfin
import ufl
import dolfiny.snesproblem


def test_singleblock(V1, squaremesh_5):
    mesh = squaremesh_5
    V = V1

    u = dolfin.Function(V)
    v = ufl.TestFunction(V)

    Phi = (u - 0.25) ** 2 * ufl.dx(mesh)

    F = ufl.derivative(Phi, u, v)

    problem = dolfiny.snesproblem.SNESProblem(F, u)

    def monitor(snes, it, norm):
        print("### SNES iteration %3d: |r| = %5.4e" % (it, norm))

    opts = PETSc.Options()

    opts.setValue('snes_type', 'newtonls')
    opts.setValue('snes_linesearch_type', 'basic')
    opts.setValue('snes_rtol', 1.0e-08)
    opts.setValue('snes_max_it', 12)

    opts.setValue('ksp_type', 'preonly')

    opts.setValue('pc_type', 'lu')
    opts.setValue('pc_factor_mat_solver_type', 'mumps')

    snes = PETSc.SNES().create(dolfin.MPI.comm_world)

    J = dolfin.fem.create_matrix(problem.J_form)
    F = dolfin.fem.create_vector(problem.F_form)

    snes.setFunction(problem.F, F)
    snes.setJacobian(problem.J, J)
    snes.setMonitor(monitor)

    snes.setFromOptions()
    snes.getKSP().setFromOptions()
    snes.getKSP().getPC().setFromOptions()

    snes.solve(None, problem.u.vector)
    assert np.isclose((problem.u.vector - 0.25).norm(), 0.0)
