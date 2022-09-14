import dolfinx
import time
import dolfiny
import numpy as np
import numba
import ufl
import pytest
from petsc4py import PETSc
from mpi4py import MPI
from dolfiny.function import vec_to_functions


c_signature = numba.types.void(
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.typeof(PETSc.ScalarType())),
    numba.types.CPointer(numba.types.double),
    numba.types.CPointer(numba.types.int32),
    numba.types.CPointer(numba.types.uint8))


def test_linear(squaremesh_5):
    mesh = squaremesh_5

    # Stress and displacement elements
    Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
    Ue = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)

    S = dolfinx.fem.FunctionSpace(mesh, Se)
    U = dolfinx.fem.FunctionSpace(mesh, Ue)

    sigma0, tau = dolfinx.fem.Function(S, name="sigma"), ufl.TestFunction(S)
    u0, v = dolfinx.fem.Function(U, name="u"), ufl.TestFunction(U)

    # Locate all facets at the free end and assign them value 1. Sort the
    # facet indices (requirement for constructing MeshTags)
    free_end_facets = np.sort(dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[0], 1.0)))
    mt = dolfinx.mesh.meshtags(mesh, 1, free_end_facets, 1)

    ds = ufl.Measure("ds", subdomain_data=mt)

    # Homogeneous boundary condition in displacement
    u_bc = dolfinx.fem.Function(U)
    u_bc.x.array[:] = 0.0

    # Displacement BC is applied to the left side
    left_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[0], 0.0))
    bdofs = dolfinx.fem.locate_dofs_topological(U, 1, left_facets)
    bc = dolfinx.fem.dirichletbc(u_bc, bdofs)

    # Elastic stiffness tensor and Poisson ratio
    E, nu = 1.0, 1.0 / 3.0

    def sigma_u(u):
        """Constitutive relation for stress-strain. Assuming plane-stress in XY"""
        eps = 0.5 * (ufl.grad(u) + ufl.grad(u).T)
        sigma = E / (1. - nu ** 2) * ((1. - nu) * eps + nu * ufl.Identity(2) * ufl.tr(eps))
        return sigma

    f = ufl.as_vector([0.0, 1.0 / 16])
    # Prepare increment function used to access increment in local solver
    du = dolfinx.fem.Function(U)

    F0 = ufl.inner(sigma0 - sigma_u(u0), tau) * ufl.dx
    F1 = ufl.inner(sigma0, ufl.grad(v)) * ufl.dx + ufl.inner(f, v) * ds(1) + \
        dolfinx.fem.Constant(mesh, 0.0) * ufl.inner(u0 + du, v) * ufl.dx

    sc_J = dolfiny.localsolver.UserKernel(
        name="sc_J",
        code=r"""
        template <typename T>
        void sc_J(T& A){
            A = J11.array - J10.array * J00.array.inverse() * J01.array;
        };
        """,
        required_J=[(1, 0), (0, 0), (0, 1), (1, 1)]
        )

    sc_F_cell = dolfiny.localsolver.UserKernel(
        name="sc_F_cell",
        code=r"""
        template <typename T>
        void sc_F_cell(T& A){
            A = F1.array - J10.array * J00.array.inverse() * F0.array;
        };
        """,
        required_J=[(1, 0), (0, 0)]
        )

    sc_F_exterior_facet = dolfiny.localsolver.UserKernel(
        name="sc_F_exterior_facet",
        code=r"""
        template <typename T>
        void sc_F_exterior_facet(T& A){
            A = F1.array;
        };
        """,
        required_J=[]
        )

    solve_stress = dolfiny.localsolver.UserKernel(
        name="solve_stress",
        code=r"""
        template <typename T>
        void solve_stress(T& A){
            auto du = Eigen::Map<const Eigen::Matrix<double, 12, 1>>(&F1.w[21]);
            auto sigma0 = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(&F1.w[0]);
            A = sigma0 - J00.array.inverse() * (F0.array - J01.array * du);
        };
        """,
        required_J=[(0, 1), (0, 0)]
        )

    def local_update(problem):
        dx = problem.snes.getSolutionUpdate()
        dx.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Fill the du function to be accessed from local kernel
        with dx.localForm() as dx_, du.vector.localForm() as du_:
            dx_.copy(du_)

        with problem.xloc.localForm() as x_local:
            x_local.set(0.0)
        # Assemble into local vector and scatter to functions
        dolfinx.fem.petsc.assemble_vector_block(
            problem.xloc, problem.local_form, problem.J_form, [], x0=problem.xloc, scale=-1.0)
        vec_to_functions(problem.xloc, [problem.u[idx] for idx in problem.localsolver.local_spaces_id])

    ls = dolfiny.localsolver.LocalSolver([S, U], local_spaces_id=[0],
                                         F_integrals=[{dolfinx.fem.IntegralType.cell:
                                                       ([(-1, sc_F_cell)], None),
                                                       dolfinx.fem.IntegralType.exterior_facet:
                                                       ([(1, sc_F_exterior_facet)], mt)}],
                                         J_integrals=[[{dolfinx.fem.IntegralType.cell: ([(-1, sc_J)], None)}]],
                                         local_integrals=[{dolfinx.fem.IntegralType.cell:
                                                           ([(-1, solve_stress)], None)}],
                                         local_update=local_update)

    opts = PETSc.Options("linear")

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08

    problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1], [sigma0, u0], [bc], prefix="linear", localsolver=ls)
    sigma1, u1 = problem.solve()

    assert np.isclose(u1.vector.norm(), 2.8002831339894887)
    assert np.isclose(sigma1.vector.norm(), 1.8884853905348435)


def test_nonlinear_elasticity_schur(squaremesh_5):

    mesh = squaremesh_5

    # Stress and displacement elements
    Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
    Ue = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)

    S = dolfinx.fem.FunctionSpace(mesh, Se)
    U = dolfinx.fem.FunctionSpace(mesh, Ue)

    sigma0, tau = dolfinx.fem.Function(S, name="sigma"), ufl.TestFunction(S)
    u0, v = dolfinx.fem.Function(U, name="u"), ufl.TestFunction(U)

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
    mu = E / (2 * (1 + nu))
    K = E / (3 * (1 - 2 * nu))
    C1, D1 = mu / 2, K / 2

    def F(u):
        return ufl.Identity(2) + ufl.grad(u)

    def sigma_u(u):
        """Consitutive relation for stress-strain. Assuming plane-stress in XY"""
        C = F(u).T * F(u)
        C = ufl.variable(C)
        J = ufl.sqrt(ufl.det(C))
        I1 = ufl.tr(C)
        W = C1 * (I1 - 2 - 2 * ufl.ln(J)) + D1 * (J - 1) ** 2
        PK2 = 2 * ufl.diff(W, C)

        return PK2

    f = ufl.as_vector([0.0, 1.0 / 16 / 2])
    du = dolfinx.fem.Function(U)
    F0 = ufl.inner(sigma0 - sigma_u(u0), tau) * ufl.dx
    F1 = ufl.inner(F(u0) * sigma0, ufl.grad(v)) * ufl.dx + ufl.inner(f, v) * ds(1) + \
        dolfinx.fem.Constant(squaremesh_5, 0.0) * ufl.inner(u0 + du, v) * ufl.dx

    sc_J = dolfiny.localsolver.UserKernel(
        name="sc_J",
        code=r"""
        template <typename T>
        void sc_J(T& A){
            A = J11.array - J10.array * J00.array.inverse() * J01.array;
        };
        """,
        required_J=[(1, 0), (0, 0), (0, 1), (1, 1)])

    sc_F_cell = dolfiny.localsolver.UserKernel(
        name="sc_F_cell",
        code=r"""
        template <typename T>
        void sc_F_cell(T& A){
            A = F1.array - J10.array * J00.array.inverse() * F0.array;
        };
        """,
        required_J=[(1, 0), (0, 0)])

    sc_F_exterior_facet = dolfiny.localsolver.UserKernel(
        name="sc_F_exterior_facet",
        code=r"""
        template <typename T>
        void sc_F_exterior_facet(T& A){
            A = F1.array;
        };
        """,
        required_J=[]
    )

    solve_stress = dolfiny.localsolver.UserKernel(
        name="solve_stress",
        code=r"""
        template <typename T>
        void solve_stress(T& A){
            auto du = Eigen::Map<const Eigen::Matrix<double, 12, 1>>(&F1.w[21]);
            auto sigma0 = Eigen::Map<const Eigen::Matrix<double, 9, 1>>(&F1.w[0]);
            A = sigma0 - J00.array.inverse() * (F0.array - J01.array * du);
        };
        """,
        required_J=[(0, 0), (0, 1)])

    def local_update(problem):
        dx = problem.snes.getSolutionUpdate()
        dx.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        # Fill the du function to be accessed from local kernel
        with dx.localForm() as dx_, du.vector.localForm() as du_:
            dx_.copy(du_)

        with problem.xloc.localForm() as x_local:
            x_local.set(0.0)
        # Assemble into local vector and scatter to functions
        dolfinx.fem.petsc.assemble_vector_block(
            problem.xloc, problem.local_form, problem.J_form, [], x0=problem.xloc, scale=-1.0)
        vec_to_functions(problem.xloc, [problem.u[idx] for idx in problem.localsolver.local_spaces_id])

    ls = dolfiny.localsolver.LocalSolver([S, U], local_spaces_id=[0],
                                         F_integrals=[{dolfinx.fem.IntegralType.cell:
                                                       ([(-1, sc_F_cell)], None),
                                                       dolfinx.fem.IntegralType.exterior_facet:
                                                       ([(1, sc_F_exterior_facet)], mt)}],
                                         J_integrals=[[{dolfinx.fem.IntegralType.cell: ([(-1, sc_J)], None)}]],
                                         local_integrals=[{dolfinx.fem.IntegralType.cell:
                                                           ([(-1, solve_stress)], None)}],
                                         local_update=local_update)

    opts = PETSc.Options("linear")

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08

    problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1], [sigma0, u0], [bc], prefix="linear", localsolver=ls)
    t0 = time.time()
    sigma1, u1 = problem.solve()
    print(time.time() - t0)

    assert np.isclose(u1.vector.norm(), 1.2419671416042748)
    assert np.isclose(sigma1.vector.norm(), 0.9835175347552177)


def test_nonlinear_elasticity_nonlinear(squaremesh_5):

    mesh = squaremesh_5
    import cffi
    ffi = cffi.FFI()

    # Stress and displacement elements
    Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
    Ue = ufl.VectorElement("Lagrange", mesh.ufl_cell(), 2)

    S = dolfinx.fem.FunctionSpace(mesh, Se)
    U = dolfinx.fem.FunctionSpace(mesh, Ue)

    sigma0, tau = dolfinx.fem.Function(S, name="sigma"), ufl.TestFunction(S)
    u0, v = dolfinx.fem.Function(U, name="u"), ufl.TestFunction(U)

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
    mu = E / (2 * (1 + nu))
    K = E / (3 * (1 - 2 * nu))
    C1, D1 = mu / 2, K / 2

    def F(u):
        return ufl.Identity(2) + ufl.grad(u)

    def sigma_u(u):
        """Consitutive relation for stress-strain. Assuming plane-stress in XY"""
        C = F(u).T * F(u)
        C = ufl.variable(C)
        J = ufl.sqrt(ufl.det(C))
        I1 = ufl.tr(C)
        W = C1 * (I1 - 2 - 2 * ufl.ln(J)) + D1 * (J - 1) ** 2
        PK2 = 2 * ufl.diff(W, C)

        return PK2

    f = ufl.as_vector([0.0, 1.0 / 16 / 2])
    F0 = ufl.inner(sigma0 - sigma_u(u0), tau) * ufl.dx
    F1 = ufl.inner(F(u0) * sigma0, ufl.grad(v)) * ufl.dx + ufl.inner(f, v) * ds(1) + \
        dolfinx.fem.Constant(squaremesh_5, 0.0) * ufl.inner(u0, v) * ufl.dx

    @numba.njit
    def sc_J(A, J, F):
        A[:] = J[1][1].array - J[1][0].array @ np.linalg.solve(J[0][0].array, J[0][1].array)

    @numba.njit
    def sc_F_cell(A, J, F):
        A[:] = F[1].array - J[1][0].array @ np.linalg.solve(J[0][0].array, F[0].array)

    sc_F_exterior_facet = dolfiny.localsolver.UserKernel(
        name="sc_F_exterior_facet",
        code=r"""
        template <typename T>
        void sc_F_exterior_facet(T& A){
            A = F1.array;
        };
        """,
        required_J=[]
    )

    solve_stress = dolfiny.localsolver.UserKernel(
        name="solve_stress",
        code=r"""
        template <typename T>
        void solve_stress(T& A){
            auto sigma0 = Eigen::Map<Eigen::Matrix<double, 9, 1>>(&F0.w[0]);

            for (int i = 0; i < 5; ++i){
                F0.array.setZero();
                F0.kernel(F0.array.data(), F0.w.data(), F0.c.data(), F0.coords.data(),
                          F0.entity_local_index.data(), F0.permutation.data());
                double err = F0.array.norm();

                if (err < 1e-12)
                    continue;

                auto dsigma = J00.array.inverse() * F0.array;
                sigma0 -= dsigma;
            }
        A = sigma0;
        }
        """,
        required_J=[(0, 0)]
        )

    def local_update(problem):
        with problem.xloc.localForm() as x_local:
            x_local.set(0.0)
        # Assemble into local vector and scatter to functions
        dolfinx.fem.petsc.assemble_vector_block(
            problem.xloc, problem.local_form, problem.J_form, [], x0=problem.xloc, scale=-1.0)
        vec_to_functions(problem.xloc, [problem.u[idx] for idx in problem.localsolver.local_spaces_id])

    ls = dolfiny.localsolver.LocalSolver([S, U], local_spaces_id=[0],
                                         F_integrals=[{dolfinx.fem.IntegralType.cell:
                                                       ([(-1, sc_F_cell)], None),
                                                       dolfinx.fem.IntegralType.exterior_facet:
                                                       ([(1, sc_F_exterior_facet)], mt)}],
                                         J_integrals=[[{dolfinx.fem.IntegralType.cell: ([(-1, sc_J)], None)}]],
                                         local_integrals=[{dolfinx.fem.IntegralType.cell:
                                                           ([(-1, solve_stress)], None)}],
                                         local_update=local_update)

    opts = PETSc.Options("linear")

    opts["snes_type"] = "newtonls"
    opts["snes_linesearch_type"] = "basic"
    opts["snes_rtol"] = 1.0e-08

    problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1], [sigma0, u0], [bc], prefix="linear", localsolver=ls)
    sigma1, u1 = problem.solve()

    assert np.isclose(u1.vector.norm(), 1.2419671416042748)
    assert np.isclose(sigma1.vector.norm(), 0.9835175347552177)
