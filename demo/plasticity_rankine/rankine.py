import numpy as np
import dolfinx
import ufl
import dolfiny
import mesh_notched
import cffi
import time
import numba
from mpi4py import MPI
from petsc4py import PETSc


name = "notched_vm"
gmsh_model, tdim = mesh_notched.mesh_notched(name, clscale=0.1)

# Create the mesh of the specimen with given dimensions and save as msh, then read into gmsh model
# mg.mesh_iso6892_gmshapi(name, l0, d0, nr, order=o, msh_file=f"{name}.msh")
# gmsh_model, tdim = dolfiny.mesh.msh_to_gmsh(f"{name}.msh")

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim, prune_z=True)

# Write mesh and meshtags to file
with dolfiny.io.XDMFFile(MPI.COMM_WORLD, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh, mts)

top_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.logical_and(np.isclose(x[0], 0.0), np.greater_equal(x[1], 0.5)))
bottom_facets = dolfinx.mesh.locate_entities_boundary(mesh, 1, lambda x: np.isclose(x[1], 0.0))

V = dolfinx.fem.VectorFunctionSpace(mesh, ("CG", 2))

quad_degree = 8
# Symmetric plastic strain space
Pe = ufl.TensorElement("Quadrature", mesh.ufl_cell(), degree=quad_degree, quad_scheme="default", symmetry=True)
P = dolfinx.fem.FunctionSpace(mesh, Pe)

# Scalar space for plastic multiplier
Le = ufl.FiniteElement("Quadrature", mesh.ufl_cell(), degree=quad_degree, quad_scheme="default")
L = dolfinx.fem.FunctionSpace(mesh, Le)

u0 = dolfinx.fem.Function(V, name="u0")
dP = dolfinx.fem.Function(P, name="dP")
dl = dolfinx.fem.Function(L, name="dl")

P0 = dolfinx.fem.Function(P, name="P0")
l0 = dolfinx.fem.Function(L, name="l0")

δu = ufl.TestFunction(V)
δdP = ufl.TestFunction(P)
δdl = ufl.TestFunction(L)


def f(sigma):
    """Square root of J2 invariant of tensor A"""
    J2 = 1 / 2 * ufl.inner(ufl.dev(sigma), ufl.dev(sigma))
    rJ2 = ufl.sqrt(J2)
    rJ2 = ufl.conditional(rJ2 < 1.0e-16, -0.0, rJ2)
    return ufl.sqrt(3) * rJ2 - Sy


def eigvals(A):
    eps = 3.0e-8
    i1 = ufl.tr(A)
    Δ = (A[0, 0] - A[1, 1])**2 + 4 * A[0, 1] * A[1, 0]  # = I1**2 - 4 * I2
    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    Δ += eps**2
    λ = (i1 + ufl.sqrt(Δ)) / 2, (i1 - ufl.sqrt(Δ)) / 2

    return λ


# def f(sigma):
#     max_eigtress = eigvals(sigma)[0]
#     return max_eigtress - Sy


# Strain measures
E = ufl.sym(ufl.grad(u0))
E_el = E - (P0 + dP)  # E_el = E(F) - P, elastic strain

mu = 100
la = 10
Sy = 0.3
sigma = 2 * mu * E_el + la * ufl.tr(E_el) * ufl.Identity(2)
young = 4*mu*(la + mu) / (la + 2*mu)
cn = 1 / young

sigma = ufl.variable(sigma)

dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": quad_degree})
F0 = ufl.inner(sigma, ufl.sym(ufl.grad(δu))) * dx  # Global momentum equilibrium
F1 = ufl.inner(dP - dl * ufl.diff(f(sigma), sigma), δdP) * dx  # Plastic flow rule
# F2 = ufl.inner(ufl.conditional(f(sigma) >= -1e-8, f(sigma), dl), δdl) * dx  # Lagrange multiplier
# F2 = ufl.inner(cn * dl - ufl.Max(0.0, cn*dl + f(sigma)), δdl) * dx
F2 = ufl.inner(ufl.Min(dl, -f(sigma)), δdl) * dx
# F2 = ufl.inner(ufl.Min(dl, -f(sigma)), δdl) * dx
# F2 = ufl.inner(ufl.sqrt(dl**2 + f(sigma)**2 + 2*1e-12) - dl + f(sigma), δdl) * dx

u_top = dolfinx.fem.Function(V, name="u_top")
u_bottom = dolfinx.fem.Function(V, name="u_bottom")


bcs = [dolfinx.fem.dirichletbc(u_top, dolfinx.fem.locate_dofs_topological(V, 1, top_facets)),
       dolfinx.fem.dirichletbc(u_bottom, dolfinx.fem.locate_dofs_topological(V, 1, bottom_facets))]


# @numba.njit
# def sc_J(A, J, F):
#     Jlg = np.concatenate((J[1][0].array, J[2][0].array), axis=0)
#     Jgl = np.concatenate((J[0][1].array, J[0][2].array), axis=1)
#     Jllrow0 = np.concatenate((J[1][1].array, J[1][2].array), axis=1)
#     Jllrow1 = np.concatenate((J[2][1].array, J[2][2].array), axis=1)
#     Jll = np.concatenate((Jllrow0, Jllrow1), axis=0)

#     A[:] = J[0][0].array - Jgl @ np.linalg.solve(Jll, Jlg)


sc_J = dolfiny.localsolver.UserKernel(
    name="sc_J",
    code=r"""
    template <typename T>
    void sc_J(T& A)
    {
        Eigen::MatrixXd Jllrow0(J11.array.rows(), J11.array.cols() + J12.array.cols());
        Jllrow0 << J11.array, J12.array;

        Eigen::MatrixXd Jllrow1(J21.array.rows(), J21.array.cols() + J22.array.cols());
        Jllrow1 << J21.array, J22.array;

        Eigen::MatrixXd Jll(J11.array.rows() + J21.array.rows(), J11.array.cols() + J22.array.cols());
        Jll << Jllrow0,
               Jllrow1;

        Eigen::MatrixXd Jlg(J10.array.rows() + J20.array.rows(), J10.array.cols());
        Jlg << J10.array,
               J20.array;

        auto J02 = Eigen::MatrixXd::Zero(J00.array.rows(), J12.array.cols());

        Eigen::MatrixXd Jgl(J01.array.rows(), J01.array.cols() + J02.cols());
        Jgl << J01.array, J02;

        A = J00.array - Jgl * Jll.partialPivLu().solve(Jlg);
    }
    """,
    required_J=[(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)])

sc_F_cell = dolfiny.localsolver.UserKernel(
    name="sc_F_cell",
    code=r"""
    template <typename T>
    void sc_F_cell(T& A)
    {
        A = F0.array;
    }
    """,
    required_J=[])

solve_body = r"""
    auto dP = Eigen::Map<Eigen::Matrix<double, 48, 1>>(&F1.w[12]);
    auto dl = Eigen::Map<Eigen::Matrix<double, 16, 1>>(&F1.w[60]);

    Eigen::Matrix<double, 48+16, 1> loc = Eigen::Matrix<double, 48+16, 1>::Zero();
    loc << dP,
           dl;
    Eigen::Matrix<double, 48+16, 1> dloc = Eigen::Matrix<double, 48+16, 1>::Zero();

    Eigen::MatrixXd Jllrow0(J11.array.rows(), J11.array.cols() + J12.array.cols());
    Eigen::MatrixXd Jllrow1(J21.array.rows(), J21.array.cols() + J22.array.cols());
    Eigen::MatrixXd Jll(J11.array.rows() + J21.array.rows(), J11.array.cols() + J22.array.cols());

    Eigen::Matrix<double, 48+16, 1> R = Eigen::Matrix<double, 48+16, 1>::Zero();

    const int N = 20;
    for (int i = 0; i < N; ++i)
    {
        F1.array.setZero();
        F1.kernel(F1.array.data(), F1.w.data(), F1.c.data(),
                    F1.coords.data(), F1.entity_local_index.data(),
                    F1.permutation.data());

        F2.array.setZero();
        F2.kernel(F2.array.data(), F2.w.data(), F2.c.data(),
                    F2.coords.data(), F2.entity_local_index.data(),
                    F2.permutation.data());

        R << F1.array,
             F2.array;
        double err = R.norm();
        double err0 = 0.0;
        if ((err < 1e-9 * err0) || (err < 1e-16))
            break;

        if (i > (N - 5))
            std::cout << "it=" << i << " error = " << err << std::endl;

        if (i == (N - 1))
            throw std::runtime_error("Failed to converge locally.");

        J11.array.setZero();
        J11.kernel(J11.array.data(), J11.w.data(), J11.c.data(),
                   J11.coords.data(), J11.entity_local_index.data(),
                   J11.permutation.data());

        J12.array.setZero();
        J12.kernel(J12.array.data(), J12.w.data(), J12.c.data(),
                   J12.coords.data(), J12.entity_local_index.data(),
                   J12.permutation.data());

        J21.array.setZero();
        J21.kernel(J21.array.data(), J21.w.data(), J21.c.data(),
                   J21.coords.data(), J21.entity_local_index.data(),
                   J21.permutation.data());

        J22.array.setZero();
        J22.kernel(J22.array.data(), J22.w.data(), J22.c.data(),
                   J22.coords.data(), J22.entity_local_index.data(),
                   J22.permutation.data());

        Jllrow0 << J11.array, J12.array;
        Jllrow1 << J21.array, J22.array;
        Jll << Jllrow0,
               Jllrow1;

        dloc = Jll.partialPivLu().solve(R);
        loc -= dloc;

        auto dP = loc(Eigen::seq(0, 47));
        auto dl = loc(Eigen::seq(48, 63));

        F1.w(Eigen::seq(12, 59)) = dP;
        F1.w(Eigen::seq(60, 75)) = dl;

        F2.w(Eigen::seq(12, 59)) = dP;
        F2.w(Eigen::seq(60, 75)) = dl;

        J11.w(Eigen::seq(12, 59)) = dP;
        J11.w(Eigen::seq(60, 75)) = dl;

        J21.w(Eigen::seq(12, 59)) = dP;
        J21.w(Eigen::seq(60, 75)) = dl;

        J22.w(Eigen::seq(12, 59)) = dP;
        J22.w(Eigen::seq(60, 75)) = dl;

        J12.w(Eigen::seq(12, 59)) = dP;
    }
"""

solve_dP = dolfiny.localsolver.UserKernel(
    name="solve_dP",
    code=f"""
    template <typename T>
    void solve_dP(T& A)
    {{
        {solve_body}
        A = dP;
    }}
    """,
    required_J=[(1, 1), (1, 2), (2, 1), (2, 2)])

solve_dl = dolfiny.localsolver.UserKernel(
    name="solve_dl",
    code=f"""
    template <typename T>
    void solve_dl(T& A)
    {{
        {solve_body}
        A = dl;
    }}
    """,
    required_J=[(1, 1), (1, 2), (2, 1), (2, 2)])

ffi = cffi.FFI()

@numba.njit
def solve_local(A, J, F, name):

    dP_idx = (12, 60)
    dl_idx = (60, 76)

    dP = F[1].w[dP_idx[0]:dP_idx[1]]
    dl = F[1].w[dl_idx[0]:dl_idx[1]]

    loc = np.concatenate((dP, dl)).copy()
    dloc = np.zeros_like(loc)

    N = 20
    for it in range(N):

        # Re-evaluate local residuals
        for i in [1, 2]:
            F[i].array[:] = 0.0
            F[i].kernel(ffi.from_buffer(F[i].array), ffi.from_buffer(F[i].w), ffi.from_buffer(
                F[i].c), ffi.from_buffer(F[i].coords), ffi.from_buffer(F[i].entity_local_index),
                ffi.from_buffer(F[i].permutation))

        R = np.concatenate((F[1].array, F[2].array))
        err = np.linalg.norm(R)

        if it == 0:
            err0 = err
        if err <= 1e-9 * err0 or err < 1e-16:
            break

        # if it > N - 25:
        print(it, err)

        for i in [1, 2]:
            for j in [1, 2]:
                J[i][j].array[:] = 0.0
                J[i][j].kernel(ffi.from_buffer(J[i][j].array), ffi.from_buffer(J[i][j].w), ffi.from_buffer(
                    J[i][j].c), ffi.from_buffer(J[i][j].coords), ffi.from_buffer(J[i][j].entity_local_index),
                    ffi.from_buffer(J[i][j].permutation))

        Jllrow0 = np.concatenate((J[1][1].array, J[1][2].array), axis=1)
        Jllrow1 = np.concatenate((J[2][1].array, J[2][2].array), axis=1)
        Jll = np.concatenate((Jllrow0, Jllrow1), axis=0)

        # Solve one NR iterate
        dloc[:] = 0.0
        dloc[:] = np.linalg.solve(Jll, R).flatten()
        loc -= dloc

        dP = loc[0:48]
        dl = loc[48:48+16]

        F[1].w[dP_idx[0]:dP_idx[1]] = dP
        F[1].w[dl_idx[0]:dl_idx[1]] = dl

        F[2].w[dP_idx[0]:dP_idx[1]] = dP
        F[2].w[dl_idx[0]:dl_idx[1]] = dl

        J[1][1].w[dP_idx[0]:dP_idx[1]] = dP
        J[1][1].w[dl_idx[0]:dl_idx[1]] = dl

        J[1][2].w[dP_idx[0]:dP_idx[1]] = dP

        J[2][1].w[dP_idx[0]:dP_idx[1]] = dP
        J[2][1].w[dl_idx[0]:dl_idx[1]] = dl

        J[2][2].w[dP_idx[0]:dP_idx[1]] = dP
        J[2][2].w[dl_idx[0]:dl_idx[1]] = dl

    return loc


# @numba.njit
# def solve_dP(A, J, F):
#     loc = solve_local(A, J, F, "dP")
#     A[:] = loc[0:48].reshape(48, 1)


# @numba.njit
# def solve_dl(A, J, F):
#     loc = solve_local(A, J, F, "dl")
#     A[:] = loc[48:48+16].reshape(16, 1)


def local_update(problem):
    with problem.xloc.localForm() as x_local:
        x_local.set(0.0)

    dolfiny.function.vec_to_functions(problem.xloc, [problem.u[idx] for idx in problem.localsolver.local_spaces_id])

    # Assemble into local vector and scatter to functions
    t0 = time.time()
    dolfinx.fem.petsc.assemble_vector_block(
        problem.xloc, problem.local_form, [[problem.J_form[0][0] for i in range(2)] for i in range(2)], [],
        x0=problem.xloc, scale=-1.0)
    print(f"Local update time: {time.time() - t0}")
    dolfiny.function.vec_to_functions(problem.xloc, [problem.u[idx] for idx in problem.localsolver.local_spaces_id])


ls = dolfiny.localsolver.LocalSolver([V, P, L], local_spaces_id=[1, 2],
                                     F_integrals=[{dolfinx.fem.IntegralType.cell:
                                                   ([(-1, sc_F_cell)], None)}],
                                     J_integrals=[[{dolfinx.fem.IntegralType.cell: ([(-1, sc_J)], None)}]],
                                     local_integrals=[{dolfinx.fem.IntegralType.cell:
                                                       ([(-1, solve_dP)], None)},
                                                      {dolfinx.fem.IntegralType.cell:
                                                       ([(-1, solve_dl)], None)}],
                                     local_update=local_update)


opts = PETSc.Options(name)

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-8
opts["snes_rtol"] = 1.0e-8
opts["snes_max_it"] = 35
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"

problem = dolfiny.snesblockproblem.SNESBlockProblem([F0, F1, F2], [u0, dP, dl], bcs=bcs,
                                                    prefix=name, localsolver=ls)

print(u0.vector.size)
print(u0.vector.size, dP.vector.size, dl.vector.size)


steps = 50
final = 0.005
stepsize = final / steps
t0 = time.time()
for i in range(steps):

    print(f"STEP ------ {i}")
    u_top.interpolate(lambda x: (np.zeros_like(x[0]), final / steps * i * np.ones_like(x[1])))

    # problem.solve(u_init=[u0, dP, dl])
    problem.solve()

    if problem.snes.getConvergedReason() < 0:
        raise RuntimeError(f"SNES failed {problem.snes.getConvergedReason()}")

    # Store primal states
    for source, target in zip([dP, dl], [P0, l0]):
        with source.vector.localForm() as locs, target.vector.localForm() as loct:
            loct.axpy(1.0, locs)
            # locs.set(0.0)

    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, f"{name}.xdmf", "a") as ofile:
        ofile.write_function(u0, float(i))

print(f"Simulation done in {time.time() - t0}")
