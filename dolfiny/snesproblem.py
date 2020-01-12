import typing
import numpy as np
import ufl
import dolfin
from petsc4py import PETSc


def functions_to_vec(u, x):
    """Copies functions into block vector"""
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            u[i].vector.copy(subvec)
            subvec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            x[offset:offset + size_local] = u[i].vector.array
            offset += size_local
            x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


def vec_to_functions(x, u):
    """Copies block vector into functions"""
    if x.getType() == "nest":
        for i, subvec in enumerate(x.getNestSubVecs()):
            subvec.copy(u[i].vector)
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
    else:
        offset = 0
        x = x.getArray(readonly=True)
        for i in range(len(u)):
            size_local = u[i].vector.getLocalSize()
            u[i].vector.array[:] = x[offset:offset + size_local]
            offset += size_local
            u[i].vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)


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

        r = snes.getFunctionNorm()
        dx = snes.getSolutionUpdate().norm()
        x = snes.getSolution().norm()

        if dolfin.MPI.comm_world.rank == 0:
            print("\n### SNES iteration {}".format(it))
            print("# |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e}".format(x, dx, r))


class SNESBlockProblem():
    def __init__(self, F_form: typing.List, u: typing.List, bcs=[], J_form=None, opts=None, nest=False):
        """SNES problem and solver wrapper

        Parameters
        ----------
        F_form
            Residual forms
        u
            Current solution functions
        bcs
        J_form
        opts
            PETSc options context
        nest: False
            True for 'matnest' data layout, False for 'aij'

        """
        self.F_form = F_form
        self.u = u

        if J_form is None:
            self.J_form = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    self.J_form[i][j] = ufl.derivative(
                        F_form[i], self.u[j], ufl.TrialFunction(self.u[j].function_space))
        else:
            self.J_form = J_form

        self.bcs = bcs
        self.opts = opts

        self.solution = []

        # Prepare empty functions on the corresponding sub-spaces
        # These store solution sub-functions
        for i, ui in enumerate(self.u):
            u = dolfin.Function(self.u[i].function_space, name=self.u[i].name)
            self.solution.append(u)

        self.norm_r = {}
        self.norm_dx = {}
        self.norm_x = {}

        self.snes = PETSc.SNES().create(dolfin.MPI.comm_world)

        if nest:
            self.J = dolfin.fem.create_matrix_nest(self.J_form)
            self.F = dolfin.fem.create_vector_nest(self.F_form)
            self.x = dolfin.fem.create_vector_nest(self.F_form)

            self.snes.setFunction(self._F_nest, self.F)
            self.snes.setJacobian(self._J_nest, self.J)
            self.snes.setMonitor(self._monitor_nest)
            self.snes.setConvergenceTest(self._converged)

            self.snes.setFromOptions()
            self.snes.getKSP().setFromOptions()
            self.snes.getKSP().getPC().setFromOptions()

        else:
            self.J = dolfin.fem.create_matrix_block(self.J_form)
            self.F = dolfin.fem.create_vector_block(self.F_form)
            self.x = dolfin.fem.create_vector_block(self.F_form)

            self.snes.setFunction(self._F_block, self.F)
            self.snes.setJacobian(self._J_block, self.J)
            self.snes.setMonitor(self._monitor_block)
            self.snes.setConvergenceTest(self._converged)

            self.snes.setFromOptions()
            self.snes.getKSP().setFromOptions()
            self.snes.getKSP().getPC().setFromOptions()

    def _F_block(self, snes, x, F):
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        with F.localForm() as f_local:
            f_local.set(0.0)

        # Update solution
        vec_to_functions(x, self.u)
        dolfin.fem.assemble_vector_block(F, self.F_form, self.J_form, self.bcs, x0=x, scale=-1.0)

    def _F_nest(self, snes, x, F):
        vec_to_functions(x, self.u)
        x = x.getNestSubVecs()

        bcs1 = dolfin.cpp.fem.bcs_cols(dolfin.fem.assemble._create_cpp_form(self.J_form), self.bcs)
        for L, F_sub, a, bc in zip(self.F_form, F.getNestSubVecs(), self.J_form, bcs1):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfin.fem.assemble_vector(F_sub, L)
            dolfin.fem.apply_lifting(F_sub, a, bc, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = dolfin.cpp.fem.bcs_rows(dolfin.fem.assemble._create_cpp_form(self.F_form), self.bcs)
        for F_sub, bc, u_sub in zip(F.getNestSubVecs(), bcs0, x):
            dolfin.fem.set_bc(F_sub, bc, u_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def _J_block(self, snes, u, J, P):
        J.zeroEntries()
        dolfin.fem.assemble_matrix_block(J, self.J_form, self.bcs, diagonal=1.0)
        J.assemble()

    def _J_nest(self, snes, u, J, P):
        J.zeroEntries()
        dolfin.fem.assemble_matrix_nest(J, self.J_form, self.bcs, diagonal=1.0)
        J.assemble()

    def _converged(self, snes, it, norms):
        it = snes.getIterationNumber()

        atol_x = []
        rtol_x = []
        atol_dx = []
        rtol_dx = []
        atol_r = []
        rtol_r = []

        for i, ui in enumerate(self.u):
            atol_x.append(self.norm_x[it][i] < snes.atol)
            atol_dx.append(self.norm_dx[it][i] < snes.atol)
            atol_r.append(self.norm_r[it][i] < snes.atol)

            # In some cases, 0th residual of a subfield could be 0.0
            # which would blow relative residual norm
            rtol_r0 = self.norm_r[0][i]
            if np.isclose(rtol_r0, 0.0):
                rtol_r0 = 1.0

            rtol_x.append(self.norm_x[it][i] < self.norm_x[0][i] * snes.rtol)
            rtol_dx.append(self.norm_dx[it][i] < self.norm_dx[0][i] * snes.rtol)
            rtol_r.append(self.norm_r[it][i] < rtol_r0 * snes.rtol)

        if it > snes.max_it:
            return -5
        elif all(atol_r) and it > 0:
            return 2
        elif all(rtol_r):
            return 3
        elif all(rtol_dx):
            return 4

    def _monitor_block(self, snes, it, norm):
        if dolfin.MPI.comm_world.rank == 0:
            print("\n### SNES iteration {}".format(it))
        self.compute_norms_block(snes)
        it = snes.getIterationNumber()
        self.print_norms(it)

    def _monitor_nest(self, snes, it, norm):
        if dolfin.MPI.comm_world.rank == 0:
            print("\n### SNES iteration {}".format(it))
        self.compute_norms_nest(snes)
        it = snes.getIterationNumber()
        self.print_norms(it)

    def print_norms(self, it):
        if dolfin.MPI.comm_world.rank == 0:
            for i, ui in enumerate(self.u):
                print("# sub {:2d} |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e} ({})".format(
                    i, self.norm_x[it][i], self.norm_dx[it][i], self.norm_r[it][i], ui.name))
            print("# all    |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e}".format(
                np.linalg.norm(np.asarray(self.norm_x[it])),
                np.linalg.norm(np.asarray(self.norm_dx[it])),
                np.linalg.norm(np.asarray(self.norm_r[it]))))

    def compute_norms_block(self, snes):
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

            offset += size_local

        it = snes.getIterationNumber()
        self.norm_r[it] = ei_r
        self.norm_dx[it] = ei_dx
        self.norm_x[it] = ei_x

    def compute_norms_nest(self, snes):
        r = snes.getFunction()[0].getNestSubVecs()
        dx = snes.getSolutionUpdate().getNestSubVecs()
        x = snes.getSolution().getNestSubVecs()

        ei_r = []
        ei_dx = []
        ei_x = []

        for i in range(len(self.u)):
            ei_r.append(r[i].norm())
            ei_dx.append(dx[i].norm())
            ei_x.append(x[i].norm())

        it = snes.getIterationNumber()
        self.norm_r[it] = ei_r
        self.norm_dx[it] = ei_dx
        self.norm_x[it] = ei_x

    def solve(self, u_init=None):

        if u_init is not None:
            functions_to_vec(u_init, self.x)

        self.snes.solve(None, self.x)
        vec_to_functions(self.x, self.solution)

        return self.solution
