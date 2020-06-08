#!/usr/bin/env python3

import numpy as np
from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
import ufl

import dolfiny.io
import dolfiny.mesh
import dolfiny.utils
import dolfiny.function
import dolfiny.expression
import dolfiny.snesblockproblem

import mesh_curve3d_gmshapi as mg


# POSTPROCESSING
def interpolate_on_mesh(mesh, q, u):

    # Extract mesh geometry nodal coordinates
    dm = mesh.geometry.dofmap
    oq = [0] + [*range(2, q + 1)] + [1]  # reorder lineX nodes: all ducks in a row...
    x0_idx = [[dm.links(i).tolist()[k] for k in oq] for i in range(dm.num_nodes)]
    x0_idx = [item for sublist in x0_idx for item in sublist]
    x0 = mesh.geometry.x[x0_idx]

    # Interpolate solution at mesh geometry nodes
    import dolfiny.interpolation
    Q = dolfinx.FunctionSpace(mesh, ("P", q))
    uf = dolfinx.Function(Q)

    if isinstance(u, list):
        ui = []
        for u_ in u:
            dolfiny.interpolation.interpolate(u_, uf)
            ui.append(uf.vector[x0_idx])
    else:
        dolfiny.interpolation.interpolate(u, uf)
        ui = uf.vector[x0_idx]

    return x0, ui


# Basic settings
name = "beam_static"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
L = 1.0  # beam length
N = 3 * 4  # number of nodes
p = 3  # physics: polynomial order
q = 3  # geometry: polynomial order

# Create the regular mesh of an annulus with given dimensions
gmsh_model, tdim = mg.mesh_curve3d_gmshapi(name, shape="f_arc", L=L, nL=N, order=q)

# # Create the regular mesh of an annulus with given dimensions and save as msh, then read into gmsh model
# mg.mesh_curve3d_gmshapi(name, shape="xline", L=L, nL=N, order=q, msh_file=f"{name}.msh")
# gmsh_model, tdim = dolfiny.mesh.msh_to_gmsh(f"{name}.msh")

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim)

# # Write mesh and meshtags to file
# with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
#     ofile.write_mesh_meshtags(mesh, mts)

# # Read mesh and meshtags from file
# with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "r") as ifile:
#     mesh, mts = ifile.read_mesh_meshtags()

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mts, tdim - 1)

# Define shorthands for labelled tags
beg = interfaces_keys["beg"]
end = interfaces_keys["end"]

# Structure: section geometry
b = 1.0  # [m]
h = L / 200  # [m]
A = b * h  # [m^2]
I = b * h**3 / 12  # [m^4]  # noqa: E741

# Structure: section geometry and material parameters
n = 0  # [-] Poisson ratio
E = 1.0e+8  # [N/m^2] elasticity modulus
G = E / 2. / (1 + n)  # [N/m^2] shear modulus

# Structure: shear correction factor, see Cowper (1966)
scf = 10 * (1 + n) / (12 + 11 * n)

# Structure: section stiffness quantities
EA = dolfinx.Constant(mesh, E * A)  # axial stiffness
EI = dolfinx.Constant(mesh, E * I)  # bending stiffness
GA = dolfinx.Constant(mesh, G * A * scf)  # shear stiffness

# Structure: load parameters
μ = dolfinx.Constant(mesh, 1.0)  # load factor

p_x = dolfinx.Constant(mesh, 1.0 * 0)
p_z = dolfinx.Constant(mesh, 1.0 * 0)
m_y = dolfinx.Constant(mesh, 1.0 * 0)
F_x = dolfinx.Constant(mesh, (2.0 * np.pi / L)**2 * EI.value * 0)
F_z = dolfinx.Constant(mesh, (0.5 * np.pi / L)**2 * EI.value * 0)
M_y = dolfinx.Constant(mesh, (2.0 * np.pi / L)**1 * EI.value * 2)
λsp = dolfinx.Constant(mesh, 1.0)  # prescribed axial stretch: 4/5, 2/3, 1/2, 2/5, 1/3 and 4/3, 2, 4
λξp = dolfinx.Constant(mesh, 1 / 2 * 0)  # prescribed shear stretch: 1/4, 1/2
κηp = dolfinx.Constant(mesh, -2 * np.pi * 0.0)  # prescribed curvature: κ0, ...

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)  # , metadata={"quadrature_degree": 4})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)  # , metadata={"quadrature_degree": 4})

# Function spaces
Ue = ufl.FiniteElement("CG", mesh.ufl_cell(), p)
We = ufl.FiniteElement("CG", mesh.ufl_cell(), p)
Re = ufl.FiniteElement("CG", mesh.ufl_cell(), p)

U = dolfinx.FunctionSpace(mesh, Ue)
W = dolfinx.FunctionSpace(mesh, We)
R = dolfinx.FunctionSpace(mesh, Re)

# Define functions
u = dolfinx.Function(U, name='u')
w = dolfinx.Function(W, name='w')
r = dolfinx.Function(R, name='r')

δu = ufl.TestFunction(U)
δw = ufl.TestFunction(W)
δr = ufl.TestFunction(R)

# Define state as (ordered) list of functions
m = [u, w, r]
δm = [δu, δw, δr]

# Coordinates of undeformed configuration
x0 = ufl.SpatialCoordinate(mesh)

# CURVATURE TENSOR
# Jacobi matrix of map reference -> undeformed
J0 = ufl.geometry.Jacobian(mesh)
# Metric tensor
G0 = ufl.dot(J0.T, J0)
# Contravariant basis
K0 = J0 * ufl.inv(G0)
# Tangent basis
g0 = J0[:, 0]
g1 = dolfinx.Constant(mesh, (0, 1, 0))  # unit vector e_y (assume curve in x-z plane)
g2 = ufl.cross(g0, g1)
# Unit tangent basis
g0 /= ufl.sqrt(ufl.dot(g0, g0))
g1 /= ufl.sqrt(ufl.dot(g1, g1))
g2 /= ufl.sqrt(ufl.dot(g2, g2))
# Curvature tensor
from ufl.differentiation import ReferenceGrad
B0 = ufl.dot(g2, ReferenceGrad(K0))  # == -ufl.dot(ReferenceGrad(g2).T, K0)

# DERIVATIVE with respect to straight reference configuration parameterised by arc-length
iJ0 = ufl.geometry.JacobianInverse(mesh)
detJ0 = ufl.geometry.JacobianDeterminant(mesh)
GRAD = lambda u: detJ0 * ufl.dot(iJ0[0, :], ufl.grad(u))  # noqa: E731

# FIX ufl definition of reference_grad (for using B0 later in the form)
from ufl.algorithms.apply_derivatives import GateauxDerivativeRuleset, GenericDerivativeRuleset
GateauxDerivativeRuleset.reference_grad = GenericDerivativeRuleset.independent_terminal

# Undeformed configuration: stretch (at the principal axis)
λ0 = ufl.sqrt(ufl.dot(GRAD(x0), GRAD(x0)))  # from geometry (!= 1)
# Undeformed configuration: curvature
κ0 = -B0[0, 0]  # from curvature tensor B0

# Deformed configuration: stretch components (at the principal axis)
λs = (1.0 + GRAD(x0[0]) * GRAD(u) + GRAD(x0[2]) * GRAD(w)) * ufl.cos(r) + \
     (      GRAD(x0[2]) * GRAD(u) - GRAD(x0[0]) * GRAD(w)) * ufl.sin(r)  # noqa: E201
λξ = (1.0 + GRAD(x0[0]) * GRAD(u) + GRAD(x0[2]) * GRAD(w)) * ufl.sin(r) - \
     (      GRAD(x0[2]) * GRAD(u) - GRAD(x0[0]) * GRAD(w)) * ufl.cos(r)  # noqa: E201
# Deformed configuration: curvature
κ = -GRAD(r)

# Green-Lagrange strains (prescribed)
e_presc = μ * 1 / 2 * (λsp**2 - λ0**2)  # prescribed axial strain, λsp denotes λ_effective = sqrt(λs^2 + λξ^2)
g_presc = μ * λξp  # prescribed shear strain
k_presc = μ * κηp  # prescribed bending strain

# Green-Lagrange strains (total): determined by deformation kinematics
e_total = 1 / 2 * (λs**2 + λξ**2 - λ0**2)
g_total = λξ
k_total = λs * κ + (λs - λ0) * κ0

# Green-Lagrange strains (elastic): e_total = e_elast + e_presc
e = e_elast = e_total - e_presc
g = g_elast = g_total - g_presc
k = k_elast = k_total - k_presc

# Variation of elastic Green-Lagrange strains
δe = dolfiny.expression.derivative(e, m, δm)
δg = dolfiny.expression.derivative(g, m, δm)
δk = dolfiny.expression.derivative(k, m, δm)

# Constitutive relations (Saint-Venant Kirchhoff)
N = EA * e
T = GA * g
M = EI * k

# Weak form: components (as one-form)
F = - δe * N * dx - δg * T * dx - δk * M * dx \
    + μ * (δu * p_x * dx + δw * p_z * dx + δr * m_y * dx) \
    + μ * (δu * F_x * ds(end) + δw * F_z * ds(end) + δr * M_y * ds(end))

# Optional: linearise weak form
# F = dolfiny.expression.linearise(F, m)  # linearise around zero state

# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
# ofile.write_mesh_meshtags(mesh, mts)

# Options for PETSc backend
opts = PETSc.Options()

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-08
opts["snes_stol"] = 1.0e-06
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_icntl_14"] = 500
opts["mat_mumps_icntl_24"] = 1

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, opts=opts)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
beg_dofs_U = dolfiny.mesh.locate_dofs_topological(U, interfaces, beg)
beg_dofs_W = dolfiny.mesh.locate_dofs_topological(W, interfaces, beg)
beg_dofs_R = dolfiny.mesh.locate_dofs_topological(R, interfaces, beg)

u_beg = dolfinx.Function(U)
w_beg = dolfinx.Function(W)
r_beg = dolfinx.Function(R)

import matplotlib.pyplot
fig, ax1 = matplotlib.pyplot.subplots()
ax1.set_title(r'finite deformation beam (displacement-based)', fontsize=12)
ax1.set_xlabel(r'coordinate $x$, displacement $[m]$', fontsize=12)
ax1.set_ylabel(r'coordinate $z$, displacement $[m]$', fontsize=12)
ax1.invert_yaxis()
ax1.axis('equal')
ax1.grid(linewidth=0.25)

fig.tight_layout()

# Process load steps
for factor in np.linspace(0, 1, num=20 + 1):

    # Set current time
    μ.value = factor

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.DirichletBC(u_beg, beg_dofs_U),  # u beg
        dolfinx.fem.DirichletBC(w_beg, beg_dofs_W),  # w beg
        dolfinx.fem.DirichletBC(r_beg, beg_dofs_R),  # r beg
    ]

    dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

    # Solve nonlinear problem
    m = problem.solve()

    # Assert convergence of nonlinear solver
    assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

    # print(f"L0 = {dolfinx.fem.assemble_scalar(ufl.sqrt(λ0**2) * dx):+7.5f}")
    # print(f"L  = {dolfinx.fem.assemble_scalar(ufl.sqrt(λs**2 + λξ**2) * dx):+7.5f}")

    # print(f"en(e) = {dolfinx.fem.assemble_scalar(0.5 * e * N * dx):+7.5f}")
    # print(f"en(g) = {dolfinx.fem.assemble_scalar(0.5 * g * T * dx):+7.5f}")
    # print(f"en(k) = {dolfinx.fem.assemble_scalar(0.5 * k * M * dx):+7.5f}")

    x0, (ui, wi, ri) = interpolate_on_mesh(mesh, q, m)

    color = (0.5 + μ.value * 0.5, 0.5 - μ.value * 0.5, 0.5 - μ.value * 0.5)
    endco = (0.5 - μ.value * 0.5, 0.5 - μ.value * 0.5, 0.5 + μ.value * 0.5)
    label = 'undeformed' if μ.value == 0.0 else 'deformed' if μ.value == 1.0 else None

    ax1.plot(x0[:, 0] + ui, x0[:, 2] + wi, '.-', linewidth=0.75, markersize=2.0, color=color, label=label)
    ax1.plot(x0[-1, 0] + ui[-1], x0[-1, 2] + wi[-1], 'o', markersize=3.0, color=endco)  # mark end

    ax1.legend(loc='upper left')
    fig.savefig(name + '_result.pdf')

# Extract solution
u_, w_, r_ = m

# u_.vector.view()
# w_.vector.view()
# r_.vector.view()

# Write output
# ofile.write_function(u_)
# ofile.write_function(w_)
# ofile.write_function(r_)

ofile.close()

# SANDBOX
# Push reference loads to global
# matT0= ufl.as_matrix([[ GRAD(x0[0]),-GRAD(x0[2]), 0],
#                       [ GRAD(x0[2]), GRAD(x0[0]), 0],
#                       [          0,            0, 1]])
# matT = ufl.as_matrix([[ ufl.cos(r), ufl.sin(r), 0],
#                       [-ufl.sin(r), ufl.cos(r), 0],
#                       [          0,          0, 1]])
# matZ = ufl.as_matrix([[λs, 0, κ + κ0],
#                       [λξ, 1,      0],
#                       [ 0, 0,     λs]])
# Fg = ufl.dot( ufl.dot(ufl.dot(matT0, matT), matZ), ufl.as_vector([F_x, F_z, M_y]) )
# F_x = Fg[0]
# F_z = Fg[1]
# M_y = Fg[2]
