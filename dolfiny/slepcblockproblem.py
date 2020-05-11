import typing

import dolfinx
import ufl
from dolfiny.function import vec_to_functions
from slepc4py import SLEPc


class SLEPcBlockProblem():
    def __init__(self, F_form: typing.List, u: typing.List, lmbda: dolfinx.Function,
                 A_form=None, B_form=None, prefix=None):
        """SLEPc problem and solver wrapper.

        Wrapper for a generalised eigenvalue problem obtained from UFL residual forms.
        """
        self.F_form = F_form
        self.u = u
        self.lmbda = lmbda
        self.comm = u[0].function_space.mesh.mpi_comm()

        self.ur = []
        self.ui = []
        for func in u:
            self.ur.append(dolfinx.function.Function(func.function_space, name=func.name))
            self.ui.append(dolfinx.function.Function(func.function_space, name=func.name))

        # Prepare tangent form M0 which has terms involving lambda
        self.M0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]
        for i in range(len(self.u)):
            for j in range(len(self.u)):
                self.M0[i][j] = ufl.algorithms.expand_derivatives(ufl.derivative(
                    F_form[i], self.u[j], ufl.TrialFunction(self.u[j].function_space)))

        if A_form is None:
            A0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    # If the form happens to be empty replace with None
                    if self.M0[i][j].empty():
                        A0[i][j] = None
                        continue

                    # Differentiate wrt. lambda and replace all remaining lambda with Zero
                    A0[i][j] = ufl.algorithms.expand_derivatives(ufl.diff(self.M0[i][j], lmbda))
                    A0[i][j] = ufl.replace(A0[i][j], {lmbda: ufl.zero()})
            self.A_form = A0
        else:
            self.A_form = A_form

        if B_form is None:
            B0 = [[None for i in range(len(self.u))] for j in range(len(self.u))]

            for i in range(len(self.u)):
                for j in range(len(self.u)):
                    B0[i][j] = ufl.replace(self.M0[i][j], {lmbda: ufl.zero()})

                    if B0[i][j].empty():
                        B0[i][j] = None
                        continue
            self.B_form = B0
        else:
            self.B_form = B_form

        self.eps = SLEPc.EPS().create(self.comm)
        self.eps.setOptionsPrefix(prefix)
        self.eps.setFromOptions()

        self.A = dolfinx.fem.create_matrix_block(self.A_form)

        self.B = None
        if not self.empty_B():
            self.B = dolfinx.fem.create_matrix_block(self.B_form)

    def solve(self):
        self.A.zeroEntries()
        dolfinx.fem.assemble_matrix_block(self.A, self.A_form, [])
        self.A.assemble()

        if not self.empty_B():
            self.B.zeroEntries()
            dolfinx.fem.assemble_matrix_block(self.B, self.B_form, [])
            self.B.assemble()

        self.eps.setOperators(self.A, self.B)
        self.eps.solve()

    def getEigenpair(self, i):
        xr, xi = self.A.getVecs()
        eigval = self.eps.getEigenpair(i, xr, xi)

        vec_to_functions(xr, self.ur)
        vec_to_functions(xi, self.ui)

        return (eigval, self.ur, self.ui)

    def empty_B(self):
        for i in range(len(self.B_form)):
            for j in range(len(self.B_form[i])):
                if self.B_form[i][j] is not None:
                    return False

        return True
