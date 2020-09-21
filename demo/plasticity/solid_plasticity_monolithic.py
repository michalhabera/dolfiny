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
name = "solid_plasticity_monolithic"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
dx, dy, dz = 1.0, 0.1, 0.1
nx, ny, nz = 20, 2, 2

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
la = dolfinx.Constant(mesh,  10)  # [1e-9 * 1e+10 N/m^2 =  10 GPa]
Sy = dolfinx.Constant(mesh, 0.3)  # initial yield stress
bh = dolfinx.Constant(mesh,  20)  # isotropic hardening: saturation rate   [-]
qh = dolfinx.Constant(mesh, 0.1)  # isotropic hardening: saturation value [GPa]
bb = dolfinx.Constant(mesh,  20)  # kinematic hardening: saturation rate   [-]
qb = dolfinx.Constant(mesh, 0.2)  # kinematic hardening: saturation value [GPa] includes factor 2/3

# Solid: load parameters
μ = dolfinx.Constant(mesh, 1.0)  # load factor
t0 = μ * dolfinx.Constant(mesh, [0.0, 0.0, 0.0])  # [GPa]
u_bar = lambda x: μ.value * np.array([0.01 * x[0] / 1.0, 0.0 * x[1], 0.0 * x[2]])  # [m]

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": 4})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# Function spaces
Ue = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)
Pe = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=None)
Le = ufl.FiniteElement("DG", mesh.ufl_cell(), 1)

Uf = dolfinx.FunctionSpace(mesh, Ue)
Sf = dolfinx.FunctionSpace(mesh, Se)
Pf = dolfinx.FunctionSpace(mesh, Pe)
Lf = dolfinx.FunctionSpace(mesh, Le)

# Define functions
u = dolfinx.Function(Uf, name="u")
P = dolfinx.Function(Pf, name="P")
h = dolfinx.Function(Lf, name="h")
B = dolfinx.Function(Sf, name="B")

u0 = dolfinx.Function(Uf, name="u0")
P0 = dolfinx.Function(Pf, name="P0")
h0 = dolfinx.Function(Lf, name="h0")
B0 = dolfinx.Function(Sf, name="B0")
S0 = dolfinx.Function(Pf, name="S0")

δu = ufl.TestFunction(Uf)
δP = ufl.TestFunction(Pf)
δh = ufl.TestFunction(Lf)
δB = ufl.TestFunction(Sf)

# Define state and rate as (ordered) list of functions
m, δm = [u, P, h, B], [δu, δP, δh, δB]


def rJ2(A):
    J2 = 1 / 2 * ufl.inner(A, A)
    rJ2 = ufl.sqrt(J2)
    return ufl.conditional(rJ2 < 1.0e-12, 0.0, rJ2)

def f(S, h, B):
    """
    Yield function
    """
    f = ufl.sqrt(3) * rJ2(ufl.dev(S)-ufl.dev(B)) - (Sy + h)
    return f

def df(S, h, B):
    """
    Total differential of yield function
    """
    varS = ufl.variable(S)
    varh = ufl.variable(h)
    varB = ufl.variable(B)
    f_ = f(varS, varh, varB)
    return ufl.inner(ufl.diff(f_, varS), (S - S0)) \
            + ufl.inner(ufl.diff(f_, varh), (h - h0)) \
            + ufl.inner(ufl.diff(f_, varB), (B - B0))

def g(S, h, B):
    """
    Plastic potential
    """
    return f(S, h, B)

def dgdS(S, h, B):
    """
    Derivative of plastic potential wrt stress: dg / dS
    """
    varS = ufl.variable(S)
    return ufl.diff(g(varS, h, B), varS)

def ppos(f):
    """
    Macaulay bracket
    """
    return ufl.Max(f, 0)


# Configuration gradient
I = ufl.Identity(u.geometric_dimension())  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient as function of displacement

# Strain measures
# E = E(u), total strain
E = 1 / 2 * (ufl.grad(u) + ufl.grad(u).T)
# E_el = E(u) - P, elastic strain
E_el = E - P
# S = S(E_el), SVK
S = 2 * mu * E_el + la * ufl.tr(E_el) * I

# Variation of rate of Green-Lagrange strain
δE = dolfiny.expression.derivative(E, m, δm)

# Plastic multiplier (J2 plasticity, closed-for solution for return-map)
dλ = ppos(f(S,h,B))

# Weak form (as one-form)
F = + ufl.inner(δE, S) * dx \
    + ufl.inner(δP, (P - P0) - dλ * dgdS(S, h, B)) * dx \
    + ufl.inner(δh, (h - h0) - dλ * bh * (qh - h)) * dx \
    + ufl.inner(δB, (B - B0) - dλ * bb * (qb * dgdS(S, h, B) - B)) * dx \
    - ufl.inner(δu, t0) * ds(surface_right)

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
opts["snes_atol"] = 1.0e-08#12
opts["snes_rtol"] = 1.0e-06#09
opts["snes_max_it"] = 10
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"

# opts["pc_factor_mat_solver_type"] = "mumps"
# opts["pc_factor_mat_solver_type"] = "superlu_dist"
opts["pc_factor_mat_solver_type"] = "umfpack"

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
P_avg = []

# Set up load steps
K = 25
Z = 10
load, unload = np.linspace(0.0, 1.0, num=K + 1), np.linspace(1.0, 0.0, num=K + 1)
cycle = np.concatenate((load, unload, -load, -unload))
cycles = np.concatenate([cycle] * Z)

dedup = lambda a: np.r_[ a[np.nonzero(np.diff(a))[0]], a[-1] ]

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

    V = dolfiny.expression.assemble(1.0, dx)
    E_avg.append(dolfiny.expression.assemble(E[0,0], dx) / V)
    S_avg.append(dolfiny.expression.assemble(S[0,0], dx) / V)
    P_avg.append(dolfiny.expression.assemble(P[0,0], dx) / V)

    dλdf_avg = dolfiny.expression.assemble(dλ * df(S,h,B), dx) / V
    print(f"(dλ * df)_avg = {dλdf_avg:4.3e}")
    dλ_f_avg = dolfiny.expression.assemble(dλ * f(S,h,B), dx) / V
    print(f"(dλ *  f)_avg = {dλ_f_avg:4.3e}")
    Pvol_avg = dolfiny.expression.assemble(ufl.sqrt(ufl.tr(P)**2), dx) / V

    # Write output
    ofile.write_function(u, step)
    ofile.write_function(P, step)
    ofile.write_function(h, step)
    ofile.write_function(B, step)

    dolfiny.interpolation.interpolate(S, S0)

    for source, target in zip([u, P, h, B], [u0, P0, h0, B0]):
        with source.vector.localForm() as locs, target.vector.localForm() as loct:
            locs.copy(loct)

    assert dλdf_avg < 1.e-5, "|| dλ*df || != 0.0"
    assert dλ_f_avg < 1.e-5, "|| dλ*df || != 0.0"
    assert Pvol_avg < 1.e-5, "|| tr(P) || != 0.0"

ofile.close()

# Post-process results

import matplotlib.pyplot

fig, ax1 = matplotlib.pyplot.subplots()
ax1.set_title("Rate-independent plasticity: $J_2$, monolithic formulation, 3D", fontsize=12)
ax1.set_xlabel(r'volume-averaged strain $\frac{1}{V}\int E_{00} dV$ [%]', fontsize=12)
ax1.set_ylabel(r'volume-averaged stress $\frac{1}{V}\int S_{00} dV$ [GPa]', fontsize=12)
ax1.grid(linewidth=0.25)
fig.tight_layout()

E_avg = np.array(E_avg) * 100.0 # strain in percent
S_avg = np.array(S_avg)
P_avg = np.array(P_avg)

# stress-strain curve
ax1.plot(E_avg, S_avg, color='tab:blue', linestyle='-', linewidth=1.0, markersize=4.0, marker='.', label=r'$S-E$ curve')
# ax1.plot(E_avg, P_avg, color='tab:orange', linestyle='-', linewidth=1.0, markersize=4.0, marker='.', label=r'$P-E$ curve')

ax1.legend(loc='lower right')
fig.savefig(f"{name}.pdf")