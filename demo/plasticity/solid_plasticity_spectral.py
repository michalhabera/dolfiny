#!/usr/bin/env python3

from petsc4py import PETSc
from mpi4py import MPI

import dolfinx
import ufl

import dolfiny.io
import dolfiny.mesh
import dolfiny.utils
import dolfiny.odeint
import dolfiny.function
import dolfiny.snesblockproblem

import mesh_din50154_gmshapi as mg

import numpy as np

# Basic settings
name = "solid_plasticity_spectral"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
dx, dy, dz = 1.0, 0.1, 0.1
nx, ny, nz = 20, 4, 4

# Create the regular mesh of an annulus with given dimensions
gmsh_model, tdim = mg.mesh_din50154_gmshapi(name, dx, dy, dz, nx, ny, nz, px=1.0, py=1.0, pz=1.0, do_quads=False)

# # Create the regular mesh of an annulus with given dimensions and save as msh, then read into gmsh model
# mg.mesh_din50154_gmshapi(name, dx, dy, dz, nx, ny, nz, do_quads=False, msh_file=f"{name}.msh")
# gmsh_model, tdim = dolfiny.mesh.msh_to_gmsh(f"{name}.msh")

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim)

# Write mesh and meshtags to file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w") as ofile:
    ofile.write_mesh_meshtags(mesh, mts)

# Read mesh and meshtags from file
with dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "r") as ifile:
    mesh, mts = ifile.read_mesh_meshtags()

# Get merged MeshTags for each codimension
subdomains, subdomains_keys = dolfiny.mesh.merge_meshtags(mts, tdim - 0)
interfaces, interfaces_keys = dolfiny.mesh.merge_meshtags(mts, tdim - 1)

# Define shorthands for labelled tags
surface_left = interfaces_keys["surface_left"]
surface_right = interfaces_keys["surface_right"]

# Solid: material parameters
mu = dolfinx.Constant(mesh, 100)  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
la = dolfinx.Constant(mesh, 0)  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
# Sy = dolfinx.Constant(mesh, 0.3)  # initial yield stress
# bh = dolfinx.Constant(mesh, 20.)  # isotropic hardening: saturation rate   [-]
# qh = dolfinx.Constant(mesh, 0.1)  # isotropic hardening: saturation value [GPa]
# bb = dolfinx.Constant(mesh, 200)  # kinematic hardening: saturation rate   [-]
# qb = dolfinx.Constant(mesh, 0.1)  # kinematic hardening: saturation value [GPa] (includes factor 2/3)

# Solid: load parameters
μ = dolfinx.Constant(mesh, 1.0)  # load factor
t0 = μ * dolfinx.Constant(mesh, [0.0, 0.0, 0.0])  # [GPa]
u_bar = lambda x: μ.value * np.array([0.01 * x[0] / 1.0, 0.0 * x[1], 0.0 * x[2]])  # noqa: E731 [m]

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": 3})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces, metadata={"quadrature_degree": 3})

# Function spaces
Ue = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
# Pe = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
# He = ufl.FiniteElement("DG", mesh.ufl_cell(), 1)
# Be = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)

Uf = dolfinx.FunctionSpace(mesh, Ue)
# Pf = dolfinx.FunctionSpace(mesh, Pe)
# Hf = dolfinx.FunctionSpace(mesh, He)
# Bf = dolfinx.FunctionSpace(mesh, Be)

# Define functions
u = dolfinx.Function(Uf, name="u")
# P = dolfinx.Function(Pf, name="P")
# h = dolfinx.Function(Hf, name="h")
# B = dolfinx.Function(Bf, name="B")

# u0 = dolfinx.Function(Uf, name="u0")
# P0 = dolfinx.Function(Pf, name="P0")
# h0 = dolfinx.Function(Hf, name="h0")
# B0 = dolfinx.Function(Bf, name="B0")
# S0 = dolfinx.Function(Bf, name="S0")

δu = ufl.TestFunction(Uf)
# δP = ufl.TestFunction(Pf)
# δh = ufl.TestFunction(Hf)
# δB = ufl.TestFunction(Bf)

# Define state and rate as (ordered) list of functions
# m, δm = [u, P, h, B], [δu, δP, δh, δB]
m, δm = [u], [δu]

# def rJ2(A):
#     J2 = 1 / 2 * ufl.inner(A, A)
#     rJ2 = ufl.sqrt(J2)
#     return ufl.conditional(rJ2 < 1.0e-12, 0.0, rJ2)


# def f(S, h, B):
#     """
#     Yield function
#     """
#     f = ufl.sqrt(3) * rJ2(ufl.dev(S) - ufl.dev(B)) - (Sy + h)
#     return f


# def df(S, h, B):
#     """
#     Total differential of yield function
#     """
#     varS = ufl.variable(S)
#     varh = ufl.variable(h)
#     varB = ufl.variable(B)
#     f_ = f(varS, varh, varB)
#     return ufl.inner(ufl.diff(f_, varS), (S - S0)) \
#         + ufl.inner(ufl.diff(f_, varh), (h - h0)) \
#         + ufl.inner(ufl.diff(f_, varB), (B - B0))


# def g(S, h, B):
#     """
#     Plastic potential
#     """
#     return f(S, h, B)


# def dgdS(S, h, B):
#     """
#     Derivative of plastic potential wrt stress: dg / dS
#     """
#     varS = ufl.variable(S)
#     return ufl.diff(g(varS, h, B), varS)


# def ppos(f):
#     """
#     Macaulay bracket
#     """
#     return ufl.Max(f, 0)

def eigenstate(A):
    """Eigenvalues and eigenvectors of 3x3 (symmetric) tensor (https://doi.org/10.1145%2F355578.366316)"""
    # determine eigenvalues
    eps = 1.0e-10
    q = ufl.tr(A) / 3.0
    p1 = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
    p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2 * p1
    p = ufl.sqrt(p2 / 6)
    B = (1 / p) * (A - q * ufl.Identity(3))
    r = ufl.det(B) / 2
    r = ufl.Max(ufl.Min(r, 1.0), -1.0)
    phi = ufl.acos(r) / 3.0
    eig0 = ufl.conditional(p1 < eps, A[0, 0], q + 2 * p * ufl.cos(phi))
    eig2 = ufl.conditional(p1 < eps, A[1, 1], q + 2 * p * ufl.cos(phi + (2 * np.pi / 3)))
    eig1 = ufl.conditional(p1 < eps, A[2, 2], 3 * q - eig0 - eig2)  # since trace(A) = eig1 + eig2 + eig3
    # determine eigenvectors
    cd = A[1, 2] * A[2, 1] - A[1, 1] * A[2, 2]
    ca = A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2]
    cb = A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]
    d0 = cd + A[1, 1] * eig0 + A[2, 2] * eig0 - eig0**2  # TODO: check pathologies
    d1 = cd + A[1, 1] * eig1 + A[2, 2] * eig1 - eig1**2
    d2 = cd + A[1, 1] * eig2 + A[2, 2] * eig2 - eig2**2
    vec0 = ufl.as_vector([1, -(ca + A[1, 0] * eig0) / d0, -(cb + A[2, 0] * eig0) / d0])
    vec1 = ufl.as_vector([1, -(ca + A[1, 0] * eig1) / d1, -(cb + A[2, 0] * eig1) / d1])
    vec2 = ufl.as_vector([1, -(ca + A[1, 0] * eig2) / d2, -(cb + A[2, 0] * eig2) / d2])
    vec0 /= ufl.sqrt(ufl.dot(vec0, vec0))
    vec1 /= ufl.sqrt(ufl.dot(vec1, vec1))
    vec2 /= ufl.sqrt(ufl.dot(vec2, vec2))
    return \
        ufl.as_vector([eig0, eig1, eig2]), \
        ufl.as_matrix([[vec0[0], vec0[1], vec0[2]], [vec1[0], vec1[1], vec1[2]], [vec2[0], vec2[1], vec2[2]]])


def eig(A):
    """Eigenvalues of 3x3 tensor"""
    eps = 1.0e-10
    q = ufl.tr(A) / 3.0
    p1 = A[0, 1] ** 2 + A[0, 2] ** 2 + A[1, 2] ** 2
    p2 = (A[0, 0] - q) ** 2 + (A[1, 1] - q) ** 2 + (A[2, 2] - q) ** 2 + 2 * p1
    p = ufl.sqrt(p2 / 6)
    B = (1 / p) * (A - q * ufl.Identity(3))
    r = ufl.det(B) / 2
    r = ufl.Max(ufl.Min(r, 1.0), -1.0)
    phi = ufl.acos(r) / 3.0
    eig0 = ufl.conditional(p1 < eps, A[0, 0], q + 2 * p * ufl.cos(phi))
    eig2 = ufl.conditional(p1 < eps, A[1, 1], q + 2 * p * ufl.cos(phi + (2 * np.pi / 3)))
    eig1 = ufl.conditional(p1 < eps, A[2, 2], 3 * q - eig0 - eig2)  # since trace(A) = eig1 + eig2 + eig3
    return eig0, eig1, eig2


def W(F):
    # some helpers
    J = ufl.det(F)
    I = ufl.Identity(3)  # noqa: E741
    # multiplicative split: F = F_vol * F_iso
    F_vol = J**(1 / 3) * I
    F_iso = J**(-1 / 3) * F
    # additive split: E = E_vol + E_iso (as a result of the multiplicative F split)
    E_vol = 1 / 2 * (F_vol.T * F_vol - I)
    E_iso = 1 / 2 * (F_iso.T * F_iso - I) * J**(2 / 3)
    # SVK strain energy
    W = 0.5 * (3 * la + 2 * mu) * ufl.inner(E_vol, E_vol) + (mu) * ufl.inner(E_iso, E_iso)
    return W


def dWdF(F):
    # variable
    F = ufl.variable(F)
    # PK-I stress (as tensor)
    return ufl.diff(W(F), F)


def dWdl(l):  # noqa: E741
    # variable
    l = ufl.variable(l)  # noqa: E741
    # deformation gradient, spectral
    F = ufl.diag(l)
    # principal PK-I stress (as vector)
    return ufl.diff(W(F), l)


# Configuration gradient
I = ufl.Identity(u.geometric_dimension())  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient as function of displacement

# Natural stress, from strain energy function
P = dWdF(F)  # PK-I
S = dWdF(F) * 2 * ufl.inv(F.T + F)  # PK-II

# Natural strain
E = 1 / 2 * (F.T * F - I)  # for post-processing

# Principal stretches
l = ufl.as_vector([ufl.sqrt(z) for z in eig(F.T * F)])  # noqa: E741

# Principal stress, from strain energy function
p = dWdl(l)  # PK-I
s = ufl.as_vector([z / v for z, v in zip(dWdl(l), l)])  # PK-II

# Principal strains
e = ufl.as_vector([(z**2 - 1) / 2 for z in l])  # for post-processing

# Variation of principal strain
δl = dolfiny.expression.derivative(l, m, δm)

# Variation of natural strain
δF = dolfiny.expression.derivative(F, m, δm)

# # Plastic multiplier (J2 plasticity, closed-form solution for return-map)
# dλ = ppos(f(S, h, B))

# Variation of direction of principal strain
_, N = eigenstate(F.T * F)
δN = dolfiny.expression.derivative(N, m, δm)
lpδNN = sum([l[a] * p[a] * ufl.dot(δN[a, :], N[a, :]) for a in range(3)])

# Weak form (as one-form)
F = + ufl.inner(δl, p) * dx + lpδNN * dx - ufl.inner(δu, t0) * ds(surface_right)
# F = + ufl.inner(δF, P) * dx - ufl.inner(δu, t0) * ds(surface_right)

# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)

# Options for PETSc backend
opts = PETSc.Options(name)

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-08
opts["snes_rtol"] = 1.0e-06
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"

opts["pc_factor_mat_solver_type"] = "mumps"
# opts["pc_factor_mat_solver_type"] = "superlu_dist"
# opts["pc_factor_mat_solver_type"] = "umfpack"

opts_global = PETSc.Options()
opts_global["mat_mumps_icntl_7"] = 6
opts_global["mat_mumps_icntl_14"] = 1500
opts_global["mat_superlu_dist_rowperm"] = "norowperm"
opts_global["mat_superlu_dist_fact"] = "samepattern_samerowperm"

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, prefix=name)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_left_dofs_U = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_left)
surface_right_dofs_U = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_right)

u_prescribed = dolfinx.Function(Uf)

E_avg = []
S_avg = []
e_avg = []
s_avg = []

l_avg = []

# Set up load steps
K = 5  # number of steps per load phase
Z = 1  # number of cycles
load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
cycle = np.concatenate((load, unload, -load, -unload))
cycles = np.concatenate([cycle] * Z)

dedup = lambda a: np.r_[a[np.nonzero(np.diff(a))[0]], a[-1]]  # noqa: E731

# Process load steps
for step, factor in enumerate(dedup(cycles)):

    # Set current time
    μ.value = factor

    dolfiny.utils.pprint(f"\n+++ Processing load factor μ = {μ.value:5.4f}")

    u_prescribed.interpolate(u_bar)

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.DirichletBC(u_prescribed, surface_left_dofs_U),  # disp left (clamped, u=0)
        dolfinx.fem.DirichletBC(u_prescribed, surface_right_dofs_U),  # disp right (clamped, u=u_bar)
    ]

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

    # Post-process data
    V = dolfiny.expression.assemble(1.0, dx)
    e_avg.append(dolfiny.expression.assemble(e, dx) / V)
    s_avg.append(dolfiny.expression.assemble(s, dx) / V)
    E_avg.append(dolfiny.expression.assemble(E[0, 0], dx) / V)
    S_avg.append(dolfiny.expression.assemble(S[0, 0], dx) / V)

    l_avg.append(dolfiny.expression.assemble(l, dx) / V)

    # dλdf_avg = dolfiny.expression.assemble(dλ * df(S, h, B), dx) / V
    # print(f"(dλ * df)_avg = {dλdf_avg:4.3e}")
    # dλ_f_avg = dolfiny.expression.assemble(dλ * f(S, h, B), dx) / V
    # print(f"(dλ *  f)_avg = {dλ_f_avg:4.3e}")
    # Pvol_avg = dolfiny.expression.assemble(ufl.sqrt(ufl.tr(P)**2), dx) / V
    # print(f"( tr(P) )_avg = {Pvol_avg:4.3e}")

    # Write output
    ofile.write_function(u, step)
    # ofile.write_function(P, step)
    # ofile.write_function(h, step)
    # ofile.write_function(B, step)

    # # Store stress state
    # dolfiny.interpolation.interpolate(S, S0)

    # # Store primal states
    # for source, target in zip([u, P, h, B], [u0, P0, h0, B0]):
    #     with source.vector.localForm() as locs, target.vector.localForm() as loct:
    #         locs.copy(loct)

    # # Basic consistency checks
    # assert dλdf_avg < 1.e-5, "|| dλ*df || != 0.0"
    # assert dλ_f_avg < 1.e-5, "|| dλ*df || != 0.0"
    # assert Pvol_avg < 1.e-5, "|| tr(P) || != 0.0"

ofile.close()

# Post-process results

import matplotlib.pyplot

fig, ax1 = matplotlib.pyplot.subplots()
ax1.set_title("Rate-independent plasticity: $J_2$, spectral formulation, 3D", fontsize=12)
ax1.set_xlabel(r'volume-averaged principal strain $\frac{1}{V}\int e_i dV$ [%]', fontsize=12)
ax1.set_ylabel(r'volume-averaged principal stress $\frac{1}{V}\int s_i dV$ [GPa]', fontsize=12)
ax1.grid(linewidth=0.25)
fig.tight_layout()

E_avg = np.array(E_avg) * 100.0  # strain in percent
S_avg = np.array(S_avg)
e_avg = np.array(e_avg) * 100.0  # strain in percent
s_avg = np.array(s_avg)

print(np.array(l_avg))

# stress-strain curve
ax1.plot(E_avg, S_avg, color='tab:red', linestyle='-', linewidth=1.0, markersize=6.0, marker='h', label=r'$S_{00}-E_{00}$')  # noqa: E501
ax1.plot(e_avg[:, 0], s_avg[:, 0], color='tab:blue', linestyle='-', linewidth=1.0, markersize=4.0, marker='.', label=r'$s_1-e_1$ curve')  # noqa: E501
ax1.plot(e_avg[:, 1], s_avg[:, 1], color='tab:green', linestyle='-', linewidth=1.0, markersize=4.0, marker='o', label=r'$s_2-e_2$ curve')  # noqa: E501
ax1.plot(e_avg[:, 2], s_avg[:, 2], color='tab:orange', linestyle='-', linewidth=1.0, markersize=4.0, marker='x', label=r'$s_3-e_3$ curve')  # noqa: E501

ax1.legend(loc='lower right')
fig.savefig(f"{name}.pdf")
