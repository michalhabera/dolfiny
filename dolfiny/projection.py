from dolfinx.fem import assemble_vector, apply_lifting, set_bc, assemble_matrix
from dolfinx.function import Function
from dolfinx.la import solve
import ufl
from petsc4py import PETSc


def project(v, V=None, bcs=[], mesh=None, funct=None):
    # Ensure we have a mesh and attach to measure
    if mesh is None:
        mesh = V.mesh
    dx = ufl.dx(mesh)

    # Define variational problem for projection
    w = ufl.TestFunction(V)
    Pv = ufl.TrialFunction(V)
    a = ufl.inner(Pv, w) * dx
    L = ufl.inner(v, w) * dx

    # Assemble linear system
    A = assemble_matrix(a, bcs)
    A.assemble()
    b = assemble_vector(L)
    apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    # Solve linear system for projection
    if funct is None:
        funct = Function(V)
    solve(A, funct.vector, b)

    return funct
