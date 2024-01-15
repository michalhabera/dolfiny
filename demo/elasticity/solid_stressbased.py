#!/usr/bin/env python3

from mpi4py import MPI
from petsc4py import PETSc
from slepc4py import SLEPc

import dolfinx
import ufl
import basix

import numpy as np

import dolfiny

name = "solid_stressbased"
comm = MPI.COMM_WORLD

# discretisation
e = 12
p = 1
c = dolfinx.cpp.mesh.CellType.hexahedron

# mesh
minx, maxx = -1.0, +1.0
mesh = dolfinx.mesh.create_box(comm, [[minx] * 3, [maxx] * 3], [e] * 3, cell_type=c)

# meshtags
mts = {}
for d in range(mesh.geometry.dim):
    for k, kx in enumerate((minx, maxx)):
        facets = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim - 1, lambda x: np.isclose(x[d], kx))
        mt = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, np.unique(facets), 2 * d + k)
        mts[f"face_x{d}={kx:+.2f}"] = mt

# merge meshtags, see `boundary_keys` for identifiers of outer faces
boundary, boundary_keys = dolfiny.mesh.merge_meshtags(mesh, mts, mesh.topology.dim - 1)
mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)

# stress
Se = basix.ufl.element("P", mesh.basix_cell(), p, shape=(3, 3), symmetry=True)
Sf = dolfinx.fem.functionspace(mesh, Se)
δS = ufl.TestFunction(Sf)
S = dolfinx.fem.Function(Sf, name='S')
S0 = dolfinx.fem.Function(Sf, name='S0')

# displacement
Ue = basix.ufl.element("P", mesh.basix_cell(), p + 1, shape=(3,))
Uf = dolfinx.fem.functionspace(mesh, Ue)
δu = ufl.TestFunction(Uf)
u = dolfinx.fem.Function(Uf, name='u')
u0 = dolfinx.fem.Function(Uf, name='u0')

# output / visualisation
vorder = mesh.geometry.cmap.degree
So = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ('P', vorder, (3, 3), True)), name="S")  # for output
uo = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ('P', vorder, (3,))), name="u")  # for output
so = dolfinx.fem.Function(dolfinx.fem.functionspace(mesh, ('P', vorder)), name="s")  # for output

# material
E = dolfinx.fem.Constant(mesh, 1.00)
nu = dolfinx.fem.Constant(mesh, 0.25)
la = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

# stabilisation, if needed by chosen stress form
psi_1 = dolfinx.fem.Constant(mesh, 0.31)  # 0.15/0.53, 0.25/0.31
psi_2 = dolfinx.fem.Constant(mesh, 0.05)  # 0.25/0.05


# strain
def strain_from_displm(u):
    return ufl.sym(ufl.grad(u))


# constitutive relations
def stress_from_strain(e):
    return 2 * mu * e + la * ufl.tr(e) * ufl.Identity(3)


def strain_from_stress(s):
    return 1 / (2 * mu) * s - la / (2 * mu * (3 * la + 2 * mu)) * ufl.tr(s) * ufl.Identity(3)


# spatial coordinates
x = ufl.SpatialCoordinate(mesh)

# normal
n = ufl.FacetNormal(mesh)

# manufactured solution
u0_expr = ufl.as_vector([ufl.sin(x[0] + x[1]), ufl.sin(x[1] + x[2]), ufl.sin(x[2] + x[0])])

E0_expr = strain_from_displm(u0_expr)
S0_expr = stress_from_strain(E0_expr)

# von Mises stress (output)
s = ufl.sqrt(3 / 2 * ufl.inner(ufl.dev(S), ufl.dev(S)))


def b(S):
    return -ufl.div(S)  # volume force


def t(S):
    return ufl.dot(S, n)  # boundary stress vector


def T(S):
    return (1 + nu) * ufl.dot(ufl.grad(S), n) + ufl.outer(ufl.grad(ufl.tr(S)), n) \
        + ufl.inner(ufl.div(S), n) * ufl.Identity(3)  # boundary stress tensor


def D(u):  # alternative form, based on given (extensional) boundary displacement
    return E / (1 - 2 * nu) * ufl.dot(ufl.grad(ufl.div(u)), n) * ufl.Identity(3)  # boundary stress tensor


# interpolate expression on function space
dolfiny.interpolation.interpolate(u0_expr, u0)
dolfiny.interpolation.interpolate(S0_expr, S0)

# measures
dx = ufl.Measure("dx", domain=mesh)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=boundary)

# boundaries (via mesh tags)
dirichlet = [0, 1, 2, 3, 4, 5]  # faces = all
# dirichlet = [0, 2, 4]  # faces = {x0 = xmin, x1 = xmin, x2 = xmin}
# dirichlet = [1, 3, 5]  # faces = {x0 = xmax, x1 = xmax, x2 = xmax}
neumann = list(set(boundary_keys.values()) - set(dirichlet))  # complement to dirichlet

# forms

# # form: stress-based, with stabilisation
# f_stress = ufl.inner(ufl.grad(δS), ufl.grad(S)) * dx \
#     + 1 / (1 + nu) * ufl.inner(ufl.div(δS), ufl.grad(ufl.tr(S))) * dx \
#     + 1 / (1 + nu) * ufl.inner(ufl.grad(ufl.tr(δS)), ufl.div(S)) * dx \
#     + (psi_1 + psi_2) * ufl.inner(ufl.div(δS), ufl.div(S)) * dx \
#     \
#     - (2 + psi_1) * ufl.inner(δS, ufl.sym(ufl.grad(b(S0_expr)))) * dx \
#     - (1 + nu**2) / (1 - nu**2) * ufl.tr(δS) * ufl.div(b(S0_expr)) * dx \
#     + psi_2 * ufl.inner(ufl.div(δS), b(S0_expr)) * dx

# form: stress-based
f_stress = (1 + nu) * ufl.inner(ufl.grad(δS), ufl.grad(S)) * dx \
    - (1 + nu) * 2 * ufl.inner(δS, ufl.sym(ufl.grad(b(S0_expr)))) * dx \
    + ufl.inner(ufl.div(δS), ufl.grad(ufl.tr(S))) * dx \
    \
    + ufl.inner(ufl.grad(ufl.tr(δS)), ufl.div(S)) * dx \
    - (1 + nu**2) / (1 - nu) * ufl.tr(δS) * ufl.div(b(S0_expr)) * dx \
    \
    - sum(ufl.inner(δS, T(S0_expr)) * ds(k) for k in neumann)
# - sum(0 * ufl.inner(δS, D(u0_expr)) * ds(k) for k in neumann) \
# CHECK: - (1 + nu) * ufl.inner(δS, ufl.outer(-b(S0_expr), n)) * ds - ufl.tr(δS) * ufl.dot(-b(S0_expr), n) * ds

# # form: stress-based, additional integration-by-parts on b-terms
# f_stress = (1 + nu) * ufl.inner(ufl.grad(δS), ufl.grad(S)) * dx \
#     + (1 + nu) * 2 * ufl.inner(ufl.div(δS), b(S0_expr)) * dx \
#     + ufl.inner(ufl.div(δS), ufl.grad(ufl.tr(S))) * dx \
#     \
#     + ufl.inner(ufl.grad(ufl.tr(δS)), ufl.div(S) + (1 + nu**2) / (1 - nu) * b(S0_expr)) * dx \
#     \
#     - sum(ufl.inner(δS, T(S0_expr)) * ds(k) for k in neumann) \
#     - sum((1 + nu) * 2 * ufl.inner(ufl.dot(δS, n), b(S0_expr)) * ds(k) for k in neumann) \
#     - sum((1 + nu**2) / (1 - nu) * ufl.tr(δS) * ufl.inner(b(S0_expr), n) * ds(k) for k in neumann)

# form: displacement-based
f_displm = ufl.inner(strain_from_displm(δu), stress_from_strain(strain_from_displm(u))) * dx \
    - ufl.inner(δu, b(S0_expr)) * dx \
    - sum(ufl.inner(δu, t(S0_expr)) * ds(k) for k in neumann)

# block form
F_stress = dolfiny.function.extract_blocks(f_stress, [δS])
F_displm = dolfiny.function.extract_blocks(f_displm, [δu])

# boundary conditions, stress
bcsdofs_Sf = dolfiny.mesh.locate_dofs_topological(Sf, boundary, dirichlet)
bcs_stress = [dolfinx.fem.dirichletbc(S0, bcsdofs_Sf)]

# boundary conditions, displm
bcsdofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, boundary, dirichlet)
bcs_displm = [dolfinx.fem.dirichletbc(u0, bcsdofs_Uf)]


# petsc options
opts = PETSc.Options(name)
opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-10
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 1
# direct solver
opts["ksp_type"] = "preonly"
opts["pc_type"] = "cholesky"
opts["pc_factor_mat_solver_type"] = "mumps"
# iterative solver
# opts["ksp_type"] = "cg"
# opts["ksp_atol"] = 1.0e-12
# opts["ksp_rtol"] = 1.0e-10
# opts["pc_type"] = "bjacobi"

# stress problem (nonlinear problem, for convenience)
problem_stress = dolfiny.snesblockproblem.SNESBlockProblem(F_stress, [S], prefix=name, bcs=bcs_stress)
problem_stress.solve()
problem_stress.status(verbose=True, error_on_failure=True)
# check symmetry
dolfiny.utils.pprint(f"(stress) discrete operator is symmetric = {problem_stress.J.isSymmetric(tol=1.0e-12)}")

# displm problem (nonlinear problem, for convenience)
problem_displm = dolfiny.snesblockproblem.SNESBlockProblem(F_displm, [u], prefix=name, bcs=bcs_displm)
problem_displm.solve()
problem_displm.status(verbose=True, error_on_failure=True)
# check symmetry
dolfiny.utils.pprint(f"(displm) discrete operator is symmetric = {problem_displm.J.isSymmetric(tol=1.0e-12)}")

# output
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
ofile.write_mesh_meshtags(mesh)
dolfiny.interpolation.interpolate(S, So)
ofile.write_function(So)
dolfiny.interpolation.interpolate(u, uo)
ofile.write_function(uo)
dolfiny.interpolation.interpolate(s, so)
ofile.write_function(so)
ofile.close()

# plot using pyvista
if comm.rank == 0:

    import pyvista

    class Xdmf3Reader(pyvista.XdmfReader):
        _vtk_module_name = "vtkIOXdmf3"
        _vtk_class_name = "vtkXdmf3Reader"

    reader = Xdmf3Reader(path=f"./{name}.xdmf")
    multiblock = reader.read()

    grid = multiblock[-1]
    grid.point_data["S"] = multiblock[0].point_data["S"]
    grid.point_data["u"] = multiblock[1].point_data["u"]
    grid.point_data["s"] = multiblock[2].point_data["s"]

    pixels = 2048
    plotter = pyvista.Plotter(off_screen=True, window_size=[pixels, pixels], image_scale=1)
    plotter.add_axes(labels_off=True)

    sargs = dict(height=0.05, width=0.8, position_x=0.1, position_y=0.90,
                 title="von Mises stress", font_family="courier", fmt="%1.2f", color="black",
                 title_font_size=pixels // 50, label_font_size=pixels // 50)

    grid_warped = grid.warp_by_vector("u", factor=1.0)

    if not grid.get_cell(0).is_linear:
        grid_warped = grid_warped.extract_surface(nonlinear_subdivision=3)

    plotter.add_mesh(grid_warped, scalars="s", scalar_bar_args=sargs, cmap="coolwarm",
                     specular=0.5, specular_power=20, smooth_shading=True, split_sharp_edges=True)

    plotter.add_mesh(grid_warped, style="wireframe", color="black", line_width=pixels // 500)

    plotter.zoom_camera(1.2)

    plotter.screenshot(f"{name}_solution.png", transparent_background=False)


# expressions
def norm(A):
    return ufl.sqrt(ufl.inner(A, A))


def error(Ah, Ar):
    return norm(Ah - Ar) / norm(Ar)  # relative error


def inc(A):
    trace_A = ufl.tr(A)
    div_A = ufl.div(A)
    Ddiv_A = ufl.grad(div_A)
    divdiv_A = ufl.div(div_A)
    laplace_A = ufl.div(ufl.grad(A))
    laplace_traceA = ufl.div(ufl.grad(trace_A))
    hessian_traceA = ufl.grad(ufl.grad(trace_A))

    return Ddiv_A + Ddiv_A.T - laplace_A - hessian_traceA + (laplace_traceA - divdiv_A) * ufl.Identity(3)


# Statistics
dolfiny.utils.pprint(f"SUMMARY :: cell = {mesh.topology.cell_name()}, e = {e:2d}")
dolfiny.utils.pprint(f"        :: p(S) = {p} [ndof = {S.vector.getSize()}]")
dolfiny.utils.pprint(f"        :: p(u) = {p + 1} [ndof = {u.vector.getSize()}]")
dolfiny.utils.pprint(f"        :: bc D = {dirichlet} = {[k for k, v in boundary_keys.items() for i in dirichlet if v == i]})")  # noqa: E501
dolfiny.utils.pprint(f"        :: bc N = {neumann} = {[k for k, v in boundary_keys.items() for i in neumann if v == i]})")  # noqa: E501

# Solution errors
dolfiny.utils.pprint("*** Errors ***")

S_vs_S0_error = dolfiny.expression.assemble(error(S, S0_expr), dx)
dolfiny.utils.pprint(f"relative S  error norm = {S_vs_S0_error:.3e}")

Su = stress_from_strain(strain_from_displm(u))
Su_vs_S0_error = dolfiny.expression.assemble(error(Su, S0_expr), dx)
dolfiny.utils.pprint(f"relative Su error norm = {Su_vs_S0_error:.3e}")

u_vs_u0_error = dolfiny.expression.assemble(error(u, u0_expr), dx)
dolfiny.utils.pprint(f"relative u  error norm = {u_vs_u0_error:.3e}")  # NOTE: could blow up (for zero-valued u0)

incE_norm = dolfiny.expression.assemble(norm(inc(strain_from_stress(S))), dx)
dolfiny.utils.pprint(f"inc(E(S )) norm = {incE_norm:.3e}")
incE_norm = dolfiny.expression.assemble(norm(inc(strain_from_stress(Su))), dx)
dolfiny.utils.pprint(f"inc(E(Su)) norm = {incE_norm:.3e}")
incE_norm = dolfiny.expression.assemble(norm(inc(strain_from_stress(S0_expr))), dx)
dolfiny.utils.pprint(f"inc(E(S0)) norm = {incE_norm:.3e}")

# Spectral properties
dolfiny.utils.pprint("*** Spectrum ***")

# dirichlet = []
bcsdofs_Sf = dolfiny.mesh.locate_dofs_topological(Sf, boundary, dirichlet)
bcs_stress = [dolfinx.fem.dirichletbc(S0, bcsdofs_Sf)]

dolfiny.utils.pprint(f"stress operator bc D = {dirichlet}")

A = dolfinx.fem.petsc.assemble_matrix_block(problem_stress.J_form, bcs=bcs_stress)
A.assemble()
assert A.isSymmetric(tol=1.0e-12)
nev = 20
opts["eps_type"] = "krylovschur"
opts["eps_hermitian"] = True
opts["eps_smallest_real"] = True
opts["eps_nev"] = nev
opts["eps_ncv"] = nev * 3
opts["eps_tol"] = 1.0e-12
opts["eps_max_it"] = 1000
eps = SLEPc.EPS().create()
eps.setOperators(A)
eps.setOptionsPrefix(name)
eps.setFromOptions()
eps.solve()
assert eps.getConverged() >= nev
eigenvalues = np.array([eps.getEigenvalue(i).real for i in range(nev)])
dolfiny.utils.pprint(eigenvalues[:nev])
dolfiny.utils.pprint(f"eps :: niter = {eps.getIterationNumber()}, reason = {eps.getConvergedReason()}")
exit()

# detailed error study

M = 11
S_errors = np.zeros((M, M))

p1 = np.logspace(-5, 5, M)
p2 = np.logspace(-5, 5, M)

for i, psi_1_value in enumerate(p1):
    for j, psi_2_value in enumerate(p2):
        psi_1.value = psi_1_value
        psi_2.value = psi_2_value

        Sh, = problem_stress.solve()

        S_errors[i, j] = dolfiny.expression.assemble(error(Sh, S0_expr), dx)

        dolfiny.utils.pprint(f"[{i}, {j}] e = {S_errors[i, j]:.4e}")


import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, dpi=400)

P1, P2 = np.meshgrid(p1, p2)
P1 = np.log10(P1)
P2 = np.log10(P2)
ax.plot_surface(P1, P2, S_errors,
                rstride=1, cstride=1, shade=True,
                linewidth=1, alpha=0.5, antialiased=False)

# ax.plot_wireframe(P1, P2, S_errors, rstride=1, cstride=1, linewidth=0.5)

ax.set_xlabel(r"$\log \psi_1$")
ax.set_ylabel(r"$\log \psi_2$")
ax.set_zlabel(r"$| S^a - S^h |_{2} / | S^ a |_{2}$")
ax.set_title("stress errors")
ax.grid(lw=0.1)

fig.tight_layout()
fig.savefig(f"{name}.png")
plt.close()
