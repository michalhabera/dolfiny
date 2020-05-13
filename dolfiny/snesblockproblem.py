import typing

import numpy as np
from mpi4py import MPI

import dolfinx
import ufl
from dolfiny.function import functions_to_vec, vec_to_functions
from dolfiny.utils import pprint
from petsc4py import PETSc


class SNESBlockProblem():
    def __init__(self, F_form: typing.List, u: typing.List, bcs=[], J_form=None,
                 opts=None, nest=False, restriction=None, prefix=None, comm=None):
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
        restriction: optional
            ``Restriction`` class used to provide information about degree-of-freedom
            indices for which this solver should solve.
        comm: optional
            MPI communicator

        """
        self.F_form = F_form
        self.u = u

        assert len(self.F_form) > 0, "List of residual forms is empty!"
        assert len(self.u) > 0, "List of current solution functions is empty!"
        assert isinstance(self.u[0], dolfinx.Function), "Provided solution function not of type dolfinx.Function!"

        if comm is None:
            self.comm = self.u[0].function_space.mesh.mpi_comm()
        else:
            self.comm = comm

        if J_form is None:
            self.J_form = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    self.J_form[i][j] = ufl.algorithms.expand_derivatives(ufl.derivative(
                        F_form[i], self.u[j], ufl.TrialFunction(self.u[j].function_space)))

                    # If the form happens to be empty replace with None
                    if self.J_form[i][j].empty():
                        self.J_form[i][j] = None
        else:
            self.J_form = J_form

        self.bcs = bcs
        self.opts = opts
        self.restriction = restriction

        self.solution = []

        # Prepare empty functions on the corresponding sub-spaces
        # These store solution sub-functions
        for i, ui in enumerate(self.u):
            u = dolfinx.Function(self.u[i].function_space, name=self.u[i].name)
            self.solution.append(u)

        self.norm_r = {}
        self.norm_dx = {}
        self.norm_x = {}

        self.snes = PETSc.SNES().create(self.comm)

        if nest:
            if restriction is not None:
                raise RuntimeError("Restriction for MATNEST not yet supported.")

            self.J = dolfinx.fem.create_matrix_nest(self.J_form)
            self.F = dolfinx.fem.create_vector_nest(self.F_form)
            self.x = self.F.copy()

            self.snes.setFunction(self._F_nest, self.F)
            self.snes.setJacobian(self._J_nest, self.J)
            self.snes.setMonitor(self._monitor_nest)

        else:
            self.J = dolfinx.fem.create_matrix_block(self.J_form)
            self.F = dolfinx.fem.create_vector_block(self.F_form)
            self.x = self.F.copy()

            if restriction is not None:
                # Need to create new global matrix for the restriction
                self._J = dolfinx.fem.create_matrix_block(self.J_form)
                self._J.assemble()

                self._x = self.x.copy()

                self.rJ = restriction.restrict_matrix(self._J)
                self.rF = restriction.restrict_vector(self.F)
                self.rx = restriction.restrict_vector(self._x)

                self.snes.setFunction(self._F_block, self.rF)
                self.snes.setJacobian(self._J_block, self.rJ)
            else:
                self.snes.setFunction(self._F_block, self.F)
                self.snes.setJacobian(self._J_block, self.J)

            self.snes.setMonitor(self._monitor_block)

        self.snes.setConvergenceTest(self._converged)
        self.snes.setOptionsPrefix(prefix)
        self.snes.setFromOptions()

    def _F_block(self, snes, x, F):
        with self.F.localForm() as f_local:
            f_local.set(0.0)

        # Update solution
        if self.restriction is not None:
            self.restriction.update_functions(self.u, x)
            functions_to_vec(self.u, self.x)
        else:
            vec_to_functions(x, self.u)
            x.copy(self.x)
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        dolfinx.fem.assemble_vector_block(self.F, self.F_form, self.J_form, self.bcs, x0=self.x, scale=-1.0)

        if self.restriction is not None:
            self.restriction.restrict_vector(self.F).copy(self.rF)

    def _F_nest(self, snes, x, F):
        vec_to_functions(x, self.u)
        x = x.getNestSubVecs()

        bcs1 = dolfinx.cpp.fem.bcs_cols(dolfinx.fem.assemble._create_cpp_form(self.J_form), self.bcs)
        for L, F_sub, a, bc in zip(self.F_form, F.getNestSubVecs(), self.J_form, bcs1):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfinx.fem.assemble_vector(F_sub, L)
            dolfinx.fem.apply_lifting(F_sub, a, bc, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = dolfinx.cpp.fem.bcs_rows(dolfinx.fem.assemble._create_cpp_form(self.F_form), self.bcs)
        for F_sub, bc, u_sub in zip(F.getNestSubVecs(), bcs0, x):
            dolfinx.fem.set_bc(F_sub, bc, u_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def _J_block(self, snes, u, J, P):
        self.J.zeroEntries()

        dolfinx.fem.assemble_matrix_block(self.J, self.J_form, self.bcs, diagonal=1.0)
        self.J.assemble()

        if self.restriction is not None:
            self.restriction.restrict_matrix(self.J).copy(self.rJ)

    def _J_nest(self, snes, u, J, P):
        self.J.zeroEntries()
        dolfinx.fem.assemble_matrix_nest(self.J, self.J_form, self.bcs, diagonal=1.0)
        self.J.assemble()

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
        pprint("\n### SNES iteration {}".format(it))
        self.compute_norms_block(snes)
        it = snes.getIterationNumber()
        self.print_norms(it)

    def _monitor_nest(self, snes, it, norm):
        pprint("\n### SNES iteration {}".format(it))
        self.compute_norms_nest(snes)
        it = snes.getIterationNumber()
        self.print_norms(it)

    def print_norms(self, it):
        for i, ui in enumerate(self.u):
            pprint("# sub {:2d} |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e} ({})".format(
                i, self.norm_x[it][i], self.norm_dx[it][i], self.norm_r[it][i], ui.name))
        pprint("# all    |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e}".format(
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

            # Need first apply square, only then sum over processes
            # i.e. norm is not a linear function
            ei_r.append(np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_r) ** 2, op=MPI.SUM)))
            ei_dx.append(np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_dx) ** 2, op=MPI.SUM)))
            ei_x.append(np.sqrt(self.comm.allreduce(np.linalg.norm(subvec_x) ** 2, op=MPI.SUM)))

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

        if self.restriction is not None:
            self.snes.solve(None, self.rx)
            self.restriction.update_functions(self.solution, self.rx)
        else:
            self.snes.solve(None, self.x)
            vec_to_functions(self.x, self.solution)

        return self.solution
