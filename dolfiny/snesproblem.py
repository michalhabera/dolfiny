import ufl
import dolfin
from petsc4py import PETSc


class SNESProblem():
    def __init__(self, F_form, u, bcs=None, J_form=None):
        self.F_form = F_form
        self.u = u

        if J_form is None:
            self.J_form = ufl.derivative(F_form, u, ufl.TrialFunction(u.function_space))
        else:
            self.J_form = J_form

        self.bcs = []

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
