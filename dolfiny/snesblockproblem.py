import typing

import dolfinx
import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc

from dolfiny.function import functions_to_vec, vec_to_functions
import dolfiny
from dolfiny.utils import pprint


class SNESBlockProblem():
    def __init__(self, F_form: typing.List, u: typing.List, bcs=[], J_form=None,
                 nest=False, restriction=None, prefix=None, localsolver: dolfiny.localsolver.LocalSolver = None):
        """SNES problem and solver wrapper

        Parameters
        ----------
        F_form
            Residual forms
        u
            Current solution functions
        bcs
        J_form
        nest: False
            True for 'matnest' data layout, False for 'aij'
        restriction: optional
            ``Restriction`` class used to provide information about degree-of-freedom
            indices for which this solver should solve. With ``LocalSolver`` only
            degrees-of-freedom for global fields must be considered.
        localsolver: optional
            ``LocalSolver`` class providing context on elimination of local
            degrees-of-freedom.

        """
        self.u = u

        if not len(F_form) > 0:
            raise RuntimeError("List of provided residual forms is empty!")

        if not len(self.u) > 0:
            raise RuntimeError("List of provided solution functions is empty!")

        if not isinstance(self.u[0], dolfinx.fem.Function):
            raise RuntimeError("Provided solution function not of type dolfinx.Function!")

        self.comm = self.u[0].function_space.mesh.comm

        if J_form is None:
            J_form = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    J_form[i][j] = ufl.algorithms.expand_derivatives(ufl.derivative(
                        F_form[i], self.u[j], ufl.TrialFunction(self.u[j].function_space)))

                    # If the form happens to be empty replace with None
                    if J_form[i][j].empty():
                        J_form[i][j] = None
        else:
            self.J_form = J_form

        # Compile all forms
        self.F_form_all_ufc = dolfinx.fem.form(F_form)
        self.J_form_all_ufc = dolfinx.fem.form(J_form)

        # By default, copy all forms as the forms used in assemblers
        self.F_form = self.F_form_all_ufc.copy()
        self.J_form = self.J_form_all_ufc.copy()
        self.global_spaces_id = range(len(self.u))

        self.localsolver = localsolver
        if self.localsolver is not None:
            # Set UFL and compiled forms to localsolver
            self.localsolver.F_ufl = F_form.copy()
            self.localsolver.J_ufl = J_form.copy()
            self.localsolver.F_ufc = self.F_form_all_ufc.copy()
            self.localsolver.J_ufc = self.J_form_all_ufc.copy()

            # Stack Coefficients and Constants
            self.localsolver.stack_data()

            # Replace compiled forms with wrapped forms for local solver
            self.F_form = self.localsolver.reduced_F_forms()
            self.J_form = self.localsolver.reduced_J_forms()
            self.local_form = self.localsolver.local_form()
            self.global_spaces_id = self.localsolver.global_spaces_id

        self.bcs = bcs
        self.solution = []

        # Prepare empty functions on the corresponding sub-spaces
        # These store solution sub-functions
        for i, ui in enumerate(self.u):
            u = dolfinx.fem.Function(self.u[i].function_space, name=self.u[i].name)
            self.solution.append(u)

        self.norm_r = {}
        self.norm_dx = {}
        self.norm_x = {}

        self.snes = PETSc.SNES().create(self.comm)

        self.restriction = restriction
        if nest:
            if self.restriction is not None:
                raise RuntimeError("Restriction for MATNEST not yet supported.")

            if self.localsolver is not None:
                raise RuntimeError("LocalSolver for MATNEST not yet supported.")

            self.J = dolfinx.fem.petsc.create_matrix_nest(self.J_form)
            self.F = dolfinx.fem.petsc.create_vector_nest(self.F_form)
            self.x = self.F.copy()

            self.snes.setFunction(self._F_nest, self.F)
            self.snes.setJacobian(self._J_nest, self.J)
            self.snes.setMonitor(self._monitor_nest)

        else:
            if self.localsolver is not None:
                # Create global vector where all local fields are assembled into
                self.xloc = dolfinx.fem.petsc.create_vector_block(self.local_form)

            self.J = dolfinx.fem.petsc.create_matrix_block(self.J_form)
            self.F = dolfinx.fem.petsc.create_vector_block(self.F_form)
            self.x = self.F.copy()

            if self.restriction is not None:
                # Need to create new global matrix for the restriction
                self._J = dolfinx.fem.petsc.create_matrix_block(self.J_form)
                self._J.assemble()

                self._x = self.x.copy()

                self.rJ = self.restriction.restrict_matrix(self._J)
                self.rF = self.restriction.restrict_vector(self.F)
                self.rx = self.restriction.restrict_vector(self._x)

                self.snes.setFunction(self._F_block, self.rF)
                self.snes.setJacobian(self._J_block, self.rJ)
            else:
                self.snes.setFunction(self._F_block, self.F)
                self.snes.setJacobian(self._J_block, self.J)

            self.snes.setMonitor(self._monitor_block)

        self.snes.setConvergenceTest(self._converged)
        self.snes.setOptionsPrefix(prefix)
        self.snes.setFromOptions()

    def update_functions(self, x):
        """Update solution functions from the stored vector x"""

        if self.restriction is not None:
            self.restriction.update_functions([self.u[idx] for idx in self.global_spaces_id], x)
            functions_to_vec([self.u[idx] for idx in self.global_spaces_id], self.x)
        else:
            vec_to_functions(x, [self.u[idx] for idx in self.global_spaces_id])

            x.copy(self.x)
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

    def _F_block(self, snes, x, F):
        with self.F.localForm() as f_local:
            f_local.set(0.0)

        self.update_functions(x)

        if self.localsolver is not None:
            self.localsolver.local_update(self)

        dolfinx.fem.petsc.assemble_vector_block(self.F, self.F_form, self.J_form, self.bcs, x0=self.x, scale=-1.0)

        if self.restriction is not None:
            self.restriction.restrict_vector(self.F).copy(self.rF)
            self.rF.copy(F)
        else:
            self.F.copy(F)

    def _F_nest(self, snes, x, F):
        vec_to_functions(x, self.u)
        x = x.getNestSubVecs()

        bcs1 = dolfinx.fem.bcs.bcs_by_block(dolfinx.fem.forms.extract_function_spaces(self.J_form, 1), self.bcs)
        for L, F_sub, a in zip(self.F_form, F.getNestSubVecs(), self.J_form):
            with F_sub.localForm() as F_sub_local:
                F_sub_local.set(0.0)
            dolfinx.fem.petsc.assemble_vector(F_sub, L)
            dolfinx.fem.petsc.apply_lifting(F_sub, a, bcs1, x0=x, scale=-1.0)
            F_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Set bc value in RHS
        bcs0 = dolfinx.fem.bcs.bcs_by_block(dolfinx.fem.forms.extract_function_spaces(self.F_form), self.bcs)
        for F_sub, bc, u_sub in zip(F.getNestSubVecs(), bcs0, x):
            dolfinx.fem.petsc.set_bc(F_sub, bc, u_sub, -1.0)

        # Must assemble F here in the case of nest matrices
        F.assemble()

    def _J_block(self, snes, x, J, P):
        self.J.zeroEntries()
        self.update_functions(x)

        dolfinx.fem.petsc.assemble_matrix_block(self.J, self.J_form, self.bcs, diagonal=1.0)
        self.J.assemble()

        if self.restriction is not None:
            self.restriction.restrict_matrix(self.J).copy(self.rJ)

    def _J_nest(self, snes, u, J, P):
        self.J.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix_nest(self.J, self.J_form, self.bcs, diagonal=1.0)
        self.J.assemble()

    def _converged(self, snes, it, norms):
        it = snes.getIterationNumber()

        atol_x = []
        rtol_x = []
        atol_dx = []
        rtol_dx = []
        atol_r = []
        rtol_r = []

        for gi, i in enumerate(self.global_spaces_id):
            atol_x.append(self.norm_x[it][gi] < snes.atol)
            atol_dx.append(self.norm_dx[it][gi] < snes.atol)
            atol_r.append(self.norm_r[it][gi] < snes.atol)

            # In some cases, 0th residual of a subfield could be 0.0
            # which would blow relative residual norm
            rtol_r0 = self.norm_r[0][gi]
            if np.isclose(rtol_r0, 0.0):
                rtol_r0 = 1.0

            rtol_x.append(self.norm_x[it][gi] < self.norm_x[0][gi] * snes.rtol)
            rtol_dx.append(self.norm_dx[it][gi] < self.norm_dx[0][gi] * snes.rtol)
            rtol_r.append(self.norm_r[it][gi] < rtol_r0 * snes.rtol)

        if it > snes.max_it:
            return -5
        elif all(atol_r) and it > 0:
            return 2
        elif all(rtol_r):
            return 3
        elif all(rtol_dx):
            return 4

    def _monitor_block(self, snes, it, norm):
        self.compute_norms_block(snes)
        self.print_norms(it)

    def _monitor_nest(self, snes, it, norm):
        self.compute_norms_nest(snes)
        self.print_norms(it)

    def print_norms(self, it):
        pprint("\n### SNES iteration {}".format(it))
        for gi, i in enumerate(self.global_spaces_id):
            pprint("# sub {:2d} |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e} ({})".format(
                gi, self.norm_x[it][gi], self.norm_dx[it][gi], self.norm_r[it][gi], self.u[i].name))
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

        for i in range(len(self.global_spaces_id)):
            if self.restriction is not None:
                # In the restriction case local size if number of
                # owned restricted dofs
                size_local = self.restriction.bglobal_dofs_vec[i].shape[0]
            else:
                size_local = self.u[i].vector.getLocalSize()

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
            self.restriction.update_functions([self.solution[idx] for idx in self.global_spaces_id], self.rx)
        else:
            self.snes.solve(None, self.x)
            vec_to_functions(self.x, [self.solution[idx] for idx in self.global_spaces_id])

        if self.localsolver is not None:
            vec_to_functions(self.xloc, [self.solution[idx] for idx in self.localsolver.local_spaces_id])

        return self.solution
