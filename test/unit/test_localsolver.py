import dolfinx
import dolfiny
import numpy as np
import numba
import ufl
from petsc4py import PETSc

c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))


def test_linear(squaremesh_5):


    @numba.jit
    def sc_J(J, F, Jc):
        J01 = J[0][1]
        J00 = J[0][0]
        J10 = J[1][0]

        Jc[:, :] = - J10 @ np.linalg.solve(J00, J01)


    @numba.jit
    def sc_F_cell(J, F, Fc):
        J10 = J[1][0]
        J00 = J[0][0]
        F0 = F[0]
        F1 = F[1]

        Fc[:] = F1 - J10 @ np.linalg.solve(J00, F0)


    @numba.jit
    def sc_F_exterior_facet(J, F, Fc):
        F1 = F[1]

        Fc[:] = F1


    @numba.jit
    def solve_stress(J, F, u):
        J01 = J[0][1]
        J00 = J[0][0]

        F0 = F[0]
        F1 = F[1]

        u[:] = np.linalg.solve(J00, F0 - J01@F1)

    # Stress and displacement elements
    Se = ufl.TensorElement("DG", squaremesh_5.ufl_cell(), 1, symmetry=True)
    Ue = ufl.VectorElement("Lagrange", squaremesh_5.ufl_cell(), 2)

    S = dolfinx.fem.FunctionSpace(squaremesh_5, Se)
    U = dolfinx.fem.FunctionSpace(squaremesh_5, Ue)

    sigma0, tau = dolfinx.fem.Function(S, name="sigma"), ufl.TestFunction(S)
    u0, v = dolfinx.fem.Function(U, name="u"), ufl.TestFunction(U)

    # Locate all facets at the free end and assign them value 1. Sort the
    # facet indices (requirement for constructing MeshTags)
    free_end_facets = np.sort(dolfinx.mesh.locate_entities_boundary(squaremesh_5, 1, lambda x: np.isclose(x[0], 1.0)))
    mt = dolfinx.mesh.meshtags(squaremesh_5, 1, free_end_facets, 1)

    ds = ufl.Measure("ds", subdomain_data=mt)

    # Homogeneous boundary condition in displacement
    u_bc = dolfinx.fem.Function(U)
    u_bc.x.array[:] = 0.0

    # Displacement BC is applied to the left side
    left_facets = dolfinx.mesh.locate_entities_boundary(squaremesh_5, 1, lambda x: np.isclose(x[0], 0.0))
    bdofs = dolfinx.fem.locate_dofs_topological(U, 1, left_facets)
    bc = dolfinx.fem.dirichletbc(u_bc, bdofs)

    # Elastic stiffness tensor and Poisson ratio
    E, nu = 1.0, 1.0 / 3.0

    def sigma_u(u):
        """Consitutive relation for stress-strain. Assuming plane-stress in XY"""
        eps = 0.5 * (ufl.grad(u) + ufl.grad(u).T)
        sigma = E / (1. - nu ** 2) * ((1. - nu) * eps + nu * ufl.Identity(2) * ufl.tr(eps))
        return sigma

    f = ufl.as_vector([0.0, 1.0 / 16])
    F0 = ufl.inner(sigma0 - sigma_u(u0), tau) * ufl.dx
    F1 = ufl.inner(sigma0, ufl.grad(v)) * ufl.dx + ufl.inner(f, v) * ds(1) + \
        dolfinx.fem.Constant(squaremesh_5, 0.0) * ufl.inner(u0, v) * ufl.dx

    ls = dolfiny.localsolver.LocalSolver([S, U], local_spaces_id=[0],
                                         F_integrals=[{dolfinx.fem.IntegralType.cell:
                                                       ([(-1, sc_F_cell)], None),
                                                       dolfinx.fem.IntegralType.exterior_facet:
                                                       ([(1, sc_F_exterior_facet)], mt)}],
                                         J_integrals=[[{dolfinx.fem.IntegralType.cell: ([(-1, sc_J)], None)}]],
                                         local_integrals=[{dolfinx.fem.IntegralType.cell:
                                                           ([(-1, solve_stress)], None)}])

    opts = PETSc.Options("linear")

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08
    opts["snes_max_it"] = 5

    problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1], [sigma0, u0], [bc], prefix="linear", localsolver=ls)
    sigma1, u1 = problem.solve()

    print(sigma1.vector.norm())

    with dolfinx.io.XDMFFile(squaremesh_5.comm, "u.xdmf", "w", dolfinx.io.XDMFFile.Encoding.HDF5) as file:
        file.write_mesh(squaremesh_5)
        file.write_function(u1)
