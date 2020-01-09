import numpy as np
import ufl
import dolfin
from petsc4py import PETSc



class SNESProblem():
    def __init__(self, F_form, u, bcs=None, J_form=None, opts=None):
        self.F_form = F_form
        self.u = u

        if J_form is None:
            self.J_form = ufl.derivative(F_form, u, ufl.TrialFunction(u.function_space))
        else:
            self.J_form = J_form

        self.bcs = []
        self.opts = opts

        self.snes = PETSc.SNES().create(dolfin.MPI.comm_world)

        J = dolfin.fem.create_matrix(self.J_form)
        F = dolfin.fem.create_vector(self.F_form)

        self.snes.setFunction(self.F, F)
        self.snes.setJacobian(self.J, J)
        self.snes.setMonitor(self.monitor)

        self.snes.setFromOptions()
        self.snes.getKSP().setFromOptions()
        self.snes.getKSP().getPC().setFromOptions()

    def F(self, snes, u, F):
        """Assemble residual vector."""
        u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)

        dolfin.fem.assemble_vector(F, self.F_form)
        dolfin.fem.apply_lifting(F, [self.J_form], [self.bcs], [u], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfin.fem.set_bc(F, self.bcs, u)

    def J(self, snes, u, J, P):
        """Assemble Jacobi matrix."""
        J.zeroEntries()
        dolfin.fem.assemble_matrix(J, self.J_form, self.bcs)
        J.assemble()

    def monitor(self, snes, it, norm):
        print("### SNES iteration %3d: |r| = %5.4e" % (it, norm))



class SNESBlockProblem():
    def __init__(self, F_form, u, bcs=None, J_form=None, opts=None, nest=False):
        self.F_form = F_form
        self.u = u

        if J_form is None:
            self.J_form = [[None for i in range(len(u))] for j in range(len(u))]

            for i in range(len(u)):
                for j in range(len(u)):
                    self.J_form[i][j] = ufl.derivative(F_form[i], u[j], ufl.TrialFunction(u[j].function_space))
        else:
            self.J_form = J_form

        self.bcs = bcs
        self.opts = opts

        self.e_r = {}
        self.e_dx = {}
        self.e_x = {}

        self.snes = PETSc.SNES().create(dolfin.MPI.comm_world)

        if nest:
            J = dolfin.fem.create_matrix_nest(self.J_form)
            F = dolfin.fem.create_vector_nest(self.F_form)

            self.snes.setFunction(self.F_nest, F)
            self.snes.setJacobian(self.J_nest, J)
            self.snes.setMonitor(self.monitor_nest)
            self.snes.setConvergenceTest(self.converged_nest)

            self.snes.setFromOptions()
            self.snes.getKSP().setFromOptions()
            self.snes.getKSP().getPC().setFromOptions()

            self.x = dolfin.fem.create_vector_nest(self.F_form)

        else:
            J = dolfin.fem.create_matrix_block(self.J_form)
            F = dolfin.fem.create_vector_block(self.F_form)

            self.snes.setFunction(self.F_block, F)
            self.snes.setJacobian(self.J_block, J)
            self.snes.setMonitor(self.monitor_block)
            self.snes.setConvergenceTest(self.converged_block)

            self.snes.setFromOptions()
            self.snes.getKSP().setFromOptions()
            self.snes.getKSP().getPC().setFromOptions()

            self.x = dolfin.fem.create_vector_block(self.F_form)

    def F_block(self, snes, u, F):
        u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        # Update solution
        offset = 0
        u_array = u.getArray(readonly=True)
        for ui in self.u:
            size_local = ui.vector.getLocalSize()
            ui.vector.array[:] = u_array[offset:offset + size_local]
            ui.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            offset += size_local

        dolfin.fem.assemble_vector_block(F, self.F_form, self.J_form, self.bcs, x0=u, scale=-1.0)

    def F_nest(self, snes, u, F):
        # Update solution
        u = u.getNestSubVecs()
        for u_sub, ui_sub in zip(u, self.u):
            u_sub.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
            with u_sub.localForm() as _u, ui_sub.vector.localForm() as _ui:
                _ui[:] = _u

        bcs1 = dolfin.cpp.fem.bcs_cols(dolfin.fem.assemble._create_cpp_form(self.J_form), self.bcs)
        for L, F_sub, a, bc in zip(self.F_form, F.getNestSubVecs(), self.J_form, bcs1):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfin.fem.assemble_vector(F_sub, L)
            dolfin.fem.apply_lifting(F_sub, a, bc, x0=u, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = dolfin.cpp.fem.bcs_rows(dolfin.fem.assemble._create_cpp_form(self.F_form), self.bcs)
        for F_sub, bc, u_sub in zip(F.getNestSubVecs(), bcs0, u):
            dolfin.fem.set_bc(F_sub, bc, u_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def J_block(self, snes, u, J, P):
        J.zeroEntries()
        dolfin.fem.assemble_matrix_block(J, self.J_form, self.bcs, diagonal=1.0)
        J.assemble()

    def J_nest(self, snes, u, J, P):
        J.zeroEntries()
        dolfin.fem.assemble_matrix_nest(J, self.J_form, self.bcs, diagonal=1.0)
        J.assemble()

    def converged_block(self, snes, it, norms):
        self.compute_errors_block(snes)
        it = snes.getIterationNumber()

        atol_x = []
        rtol_x = []
        atol_dx = []
        rtol_dx = []
        atol_r = []
        rtol_r = []

        for i, ui in enumerate(self.u):
            atol_x.append(self.e_x[it][i] < snes.atol)
            atol_dx.append(self.e_dx[it][i] < snes.atol)
            atol_r.append(self.e_r[it][i] < snes.atol)

            rtol_x.append(self.e_x[it][i] < self.e_x[0][i] * snes.rtol)
            rtol_dx.append(self.e_dx[it][i] < self.e_dx[0][i] * snes.rtol)
            rtol_r.append(self.e_r[it][i] < self.e_r[0][i] * snes.rtol)

        if it == 0:
            return all(atol_r)
        else:
            return all(rtol_r) or all(atol_r) or all(rtol_dx) or all(atol_dx)

    def converged_nest(self, snes, it, norms):
        x = snes.getFunction()[0]

        converged_atol = []
        converged_rtol = []

        for subvec in x.getNestSubVecs():
            converged_atol.append(subvec.norm() < snes.atol)

        return False#x.norm() < snes.rtol

    def monitor_block(self, snes, it, norm):
        print("\n ### SNES iteration {}".format(it))

        self.compute_errors_block(snes)
        it = snes.getIterationNumber()

        for i, ui in enumerate(self.u):
            print("# sub {:2d} |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e} ({})".format(
                i, self.e_x[it][i], self.e_dx[it][i], self.e_r[it][i], ui.name))

        print("# all    |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e}".format(
                np.linalg.norm(np.asarray(self.e_x[it])),
                np.linalg.norm(np.asarray(self.e_dx[it])),
                np.linalg.norm(np.asarray(self.e_r[it]))))

    def monitor_nest(self, snes, it, norm):
        print("\n ### SNES iteration {}".format(it))

        for i, subvec in enumerate(snes.getFunction()[0].getNestSubVecs()):
            print("# Residual subvector {} norm {:1.3e} ({})".format(i, subvec.norm(), self.u[i].name))

    def compute_errors_block(self, snes):

        r = snes.getFunction()[0].getArray(readonly=True)
        dx = snes.getSolutionUpdate().getArray(readonly=True)
        x = snes.getSolution().getArray(readonly=True)

        ei_r = []
        ei_dx = []
        ei_x = []

        offset = 0
        for i, ui in enumerate(self.u):
            size_local = ui.vector.getLocalSize()
            subvec_r = r[offset:offset + size_local]
            subvec_dx = dx[offset:offset + size_local]
            subvec_x = x[offset:offset + size_local]

            ei_r.append(np.linalg.norm(subvec_r))
            ei_dx.append(np.linalg.norm(subvec_dx))
            ei_x.append(np.linalg.norm(subvec_x))

        it = snes.getIterationNumber()
        self.e_r[it] = ei_r
        self.e_dx[it] = ei_dx
        self.e_x[it] = ei_x


    def solution(self):
        sol = []

        if self.x.getType() == "nest":
            raise RuntimeError("Not implemented")
        else:
            offset = 0
            for i, ui in enumerate(self.u):
                u = dolfin.Function(self.u[i].function_space, name=self.u[i].name)
                size_local = ui.vector.getLocalSize()
                u.vector.array[:] = self.x.array[offset:offset + size_local]

                sol.append(u)

        return sol
