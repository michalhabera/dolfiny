import ufl
import dolfinx
from petsc4py import PETSc


class SNESProblem():
    def __init__(self, F_form, u, bcs=None, J_form=None, opts=None):
        self.F_form = F_form
        self.u = u
        self.solution = dolfinx.Function(u.function_space)

        if J_form is None:
            self.J_form = ufl.derivative(F_form, u, ufl.TrialFunction(u.function_space))
        else:
            self.J_form = J_form

        self.bcs = []
        self.opts = opts

        self.snes = PETSc.SNES().create(dolfinx.MPI.comm_world)

        self.J = dolfinx.fem.create_matrix(self.J_form)
        self.F = dolfinx.fem.create_vector(self.F_form)
        self.x = dolfinx.fem.create_vector(self.F_form)

        self.snes.setFunction(self._F, self.F)
        self.snes.setJacobian(self._J, self.J)
        self.snes.setMonitor(self.monitor)

        self.snes.setFromOptions()
        self.snes.getKSP().setFromOptions()
        self.snes.getKSP().getPC().setFromOptions()

    def _F(self, snes, u, F):
        u.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        u.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        with F.localForm() as f_local:
            f_local.set(0.0)

        dolfinx.fem.assemble_vector(F, self.F_form)
        dolfinx.fem.apply_lifting(F, [self.J_form], [self.bcs], [u], -1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.set_bc(F, self.bcs, u)

    def _J(self, snes, u, J, P):
        J.zeroEntries()
        dolfinx.fem.assemble_matrix(J, self.J_form, self.bcs)
        J.assemble()

    def monitor(self, snes, it, norm):
        r = snes.getFunctionNorm()
        dx = snes.getSolutionUpdate().norm()
        x = snes.getSolution().norm()

        if dolfinx.MPI.comm_world.rank == 0:
            print("\n### SNES iteration {}".format(it))
            print("# |x|={:1.3e} |dx|={:1.3e} |r|={:1.3e}".format(x, dx, r))

    def solve(self, u_init=None):
        if u_init is not None:
            u_init.vector.copy(self.x)
            self.x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.snes.solve(None, self.x)

        self.x.copy(self.solution.vector)
        self.solution.vector.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        return self.solution
