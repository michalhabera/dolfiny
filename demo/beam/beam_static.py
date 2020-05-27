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
import dolfiny.snesblockproblem

import mesh_curve3d_gmshapi as mg

# Basic settings
name = "beam_static"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
L = 1.0  # beam length
N = 10  # number of elements
p = 1  # physics: polynomial order
g = 2  # geometry: polynomial order

# Create the regular mesh of an annulus with given dimensions
gmsh_model, tdim = mg.mesh_curve3d_gmshapi(name, shape="xline", L=L, nL=N, order=g)

# # Create the regular mesh of an annulus with given dimensions and save as msh, then read into gmsh model
# mg.mesh_curve3d_gmshapi(name, shape="xline", L=L, nL=N, order=g, msh_file=f"{name}.msh")
# gmsh_model, tdim = dolfiny.mesh.msh_to_gmsh(f"{name}.msh")

# Get mesh and meshtags
mesh, mts = dolfiny.mesh.gmsh_to_dolfin(gmsh_model, tdim, prune_z=True)  # TODO: prune z/y

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
left = interfaces_keys["left"]
right = interfaces_keys["right"]

# Structure material and load parameters
E = 1.0e+8  # [N/m^2]
G = E / 2.  # [N/m^2]
A = 1.0e-4  # [m^2]
I = 1.0e-6  # [m^4]
EA = dolfinx.Constant(mesh, E * A)  # axial stiffness
EI = dolfinx.Constant(mesh, E * I)  # bending stiffness
GA = dolfinx.Constant(mesh, G * A)  # shear stiffness

p_1 = dolfinx.Constant(mesh, 1.0 * 0)
p_3 = dolfinx.Constant(mesh, 1.0 * 0)
m_2 = dolfinx.Constant(mesh, 1.0 * 0)
F_1 = dolfinx.Constant(mesh, 1.0 * 0)
F_3 = dolfinx.Constant(mesh, 1.0 * 0)
M_2 = dolfinx.Constant(mesh, -np.pi * 2)

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains, metadata={"quadrature_degree": 4})
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces, metadata={"quadrature_degree": 4})

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

# Coordinates
x = ufl.SpatialCoordinate(mesh)

# Axial strain
ε = (1.0 + x[0].dx(0) * u.dx(0) + x[2].dx(0) * w.dx(0)) * ufl.cos(r) + \
    (      x[2].dx(0) * u.dx(0) - x[0].dx(0) * w.dx(0)) * ufl.sin(r) - 1.0
# Shear strain
γ = (1.0 + x[0].dx(0) * u.dx(0) + x[2].dx(0) * w.dx(0)) * ufl.sin(r) - \
    (      x[2].dx(0) * u.dx(0) - x[0].dx(0) * w.dx(0)) * ufl.cos(r)
# Bending strain
κ = r.dx(0)

# Virtual strains
δε = ufl.diff(ε, u) * δu + ufl.diff(ε, w) * δw + ufl.diff(ε, r) * δr
δγ = ufl.diff(γ, u) * δu + ufl.diff(γ, w) * δw + ufl.diff(γ, r) * δr
δκ = ufl.diff(κ, u) * δu + ufl.diff(κ, w) * δw + ufl.diff(κ, r) * δr

# Weak form: components (as one-form)
F = + δε * EA * ε * dx + δγ * GA * γ * dx + δκ * EI * κ * dx \
    + δu * p_1 * dx + δw * p_3 * dx + δr * m_2 * dx \
    + δu * F_1 * ds(right) + δw * F_3 * ds(right) + δr * M_2 * ds(right)

# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)

# Options for PETSc backend
opts = PETSc.Options()

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_rtol"] = 1.0e-08
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
opts["mat_mumps_icntl_14"] = 500
opts["mat_mumps_icntl_24"] = 1

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, opts=opts)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
left_dofs_U = dolfiny.mesh.locate_dofs_topological(U, interfaces, left)
left_dofs_W = dolfiny.mesh.locate_dofs_topological(W, interfaces, left)
left_dofs_R = dolfiny.mesh.locate_dofs_topological(R, interfaces, left)

u_left = dolfinx.Function(U)
w_left = dolfinx.Function(W)
r_left = dolfinx.Function(R)

# Set/update boundary conditions
problem.bcs = [
    dolfinx.fem.DirichletBC(u_left, left_dofs_U),  # disp0 left
    dolfinx.fem.DirichletBC(w_left, left_dofs_W),  # disp1 left
    dolfinx.fem.DirichletBC(r_left, left_dofs_R),  # rota2 left
]

# Solve nonlinear problem
m = problem.solve()

# Assert convergence of nonlinear solver
assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

# Extract solution
u_, w_, r_ = m

dolfiny.utils.pprint(u_.vector[:])
dolfiny.utils.pprint(w_.vector[:])
dolfiny.utils.pprint(r_.vector[:])

# Write output
ofile.write_function(u_)
ofile.write_function(w_)
ofile.write_function(r_)

ofile.close()
