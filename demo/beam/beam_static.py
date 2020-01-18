#!/usr/bin/env python3

# pip3 install -e . && export PYTHONDONTWRITEBYTECODE=1

import mesh_curve3d_gmshapi as mg
import dolfiny.odeint as oi
import dolfiny.snesblockproblem
import dolfiny.function
import prepare_output as po
from petsc4py import PETSc

import dolfin as df
import dolfin.io as dfio
import dolfin.log as dflog
import ufl as ufl

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Basic settings
name = "beam_static"

# Init files and folders
po.init(name)

# Geometry and mesh parameters
L = 1.0 # beam length
N = 10 # number of elements
p = 1 # ansatz polynomial order

# create the mesh of a curve with given dimensions
mesh, labels = mg.mesh_curve3d_gmshapi(name, shape="xline", L=L, nL=N)

# read mesh, subdomains and interfaces
with dfio.XDMFFile(df.MPI.comm_world, name + ".xdmf") as infile:
    mesh = infile.read_mesh(df.cpp.mesh.GhostMode.none)
with dfio.XDMFFile(df.MPI.comm_world, name + "_subdomains" + ".xdmf") as infile:
    msh_subdomains = mesh
    mvc_subdomains = infile.read_mvc_size_t(msh_subdomains)
    subdomains = df.cpp.mesh.MeshFunctionSizet(msh_subdomains, mvc_subdomains, 0)
with dfio.XDMFFile(df.MPI.comm_world, name + "_interfaces" + ".xdmf") as infile:
    msh_interfaces = mesh
    mvc_interfaces = infile.read_mvc_size_t(msh_interfaces)
    interfaces = df.cpp.mesh.MeshFunctionSizet(msh_interfaces, mvc_interfaces, 0)

beam = labels["subdomain_all"]
left = labels["interface_left"]
right = labels["interface_right"]

# Structure material and load parameters
EA = df.Constant(mesh, 1.0) # axial stiffness
EI = df.Constant(mesh, 1.0) # bending stiffness
GA = df.Constant(mesh, 1.0) # shear stiffness

p_1 = df.Constant(mesh, 1.0*0)
p_3 = df.Constant(mesh, 1.0*0)
m_2 = df.Constant(mesh, 1.0*0)
F_1 = df.Constant(mesh, 1.0*0)
F_3 = df.Constant(mesh, 1.0*0)
M_2 = df.Constant(mesh, -np.pi*2)

# Define measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# Check geometry data
int_dx_msh = df.fem.assemble_scalar(1 * dx)
int_dx_ana = L
print("int dx  = {:4.3e} (rel error = {:4.3e})".format(
      int_dx_msh,
      np.sqrt((int_dx_msh - int_dx_ana) ** 2 / int_dx_ana ** 2)))

# Function spaces
Ue = ufl.FiniteElement("CG", mesh.ufl_cell(), p)
We = ufl.FiniteElement("CG", mesh.ufl_cell(), p)
Re = ufl.FiniteElement("CG", mesh.ufl_cell(), p)

U = df.FunctionSpace(mesh, Ue)
W = df.FunctionSpace(mesh, We)
R = df.FunctionSpace(mesh, Re)

# Define functions
u = df.Function(U, name='u')
w = df.Function(W, name='w')
r = df.Function(R, name='r')

δu = ufl.TestFunction(U)
δw = ufl.TestFunction(W)
δr = ufl.TestFunction(R)

m = [u, w, r]
δm = [δu, δw, δr]

# Coordinates
x = ufl.SpatialCoordinate(mesh)

# Axial strain
ε = (1.0 + x[0].dx(0)*u.dx(0) + x[2].dx(0)*w.dx(0)) * ufl.cos(r) + (x[2].dx(0)*u.dx(0) - x[0].dx(0)*w.dx(0)) * ufl.sin(r) - 1.0
# Shear strain
γ = (1.0 + x[0].dx(0)*u.dx(0) + x[2].dx(0)*w.dx(0)) * ufl.sin(r) - (x[2].dx(0)*u.dx(0) - x[0].dx(0)*w.dx(0)) * ufl.cos(r)
# Bending strain
κ = r.dx(0)

# Virtual strains
δε = ufl.diff(ε, u) * δu + ufl.diff(ε, w) * δw + ufl.diff(ε, r) * δr
δγ = ufl.diff(γ, u) * δu + ufl.diff(γ, w) * δw + ufl.diff(γ, r) * δr
δκ = ufl.diff(κ, u) * δu + ufl.diff(κ, w) * δw + ufl.diff(κ, r) * δr

# Overall form
F = + δε * EA * ε * dx(beam) \
    + δγ * GA * γ * dx(beam) \
    + δκ * EI * κ * dx(beam) \
    + δu * p_1 * dx(beam)  \
    + δw * p_3 * dx(beam)  \
    + δr * m_2 * dx(beam)  \
    + δu * F_1 * ds(right) \
    + δw * F_3 * ds(right) \
    + δr * M_2 * ds(right)

# TEST form
F = δu * 1 * (u-x[0]) * dx + δw * 1 * (w-x[0]) * dx + δr * 1 * (r-x[0]) * dx

# Output files
ofile_u = dfio.XDMFFile(df.MPI.comm_world, name + "_disp1.xdmf")
ofile_w = dfio.XDMFFile(df.MPI.comm_world, name + "_disp3.xdmf")
ofile_r = dfio.XDMFFile(df.MPI.comm_world, name + "_rota2.xdmf")

# Options
opts = PETSc.Options()

opts.setValue('snes_type', 'newtonls')
opts.setValue('snes_linesearch_type', 'basic')
opts.setValue('snes_rtol', 1.0e-10)
opts.setValue('snes_max_it', 12)

opts.setValue('ksp_type', 'gmres')

opts.setValue('pc_type', 'lu')
opts.setValue('pc_factor_mat_solver_type', 'mumps')
opts.setValue('mat_mumps_icntl_14', 500)
opts.setValue('mat_mumps_icntl_24', 1)

# Obtain the list of forms from the monolithic form
F = dolfiny.function.extract_forms(F, δm)

# Set up the block problem
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, opts=opts)

u_left = df.Function(U)
w_left = df.Function(W)
r_left = df.Function(R)

u_left.interpolate(lambda x: np.zeros((1, x.shape[1])))
w_left.interpolate(lambda x: np.zeros((1, x.shape[1])))
r_left.interpolate(lambda x: np.zeros((1, x.shape[1])))

# Set/update boundary conditions
problem.bcs = [
    df.fem.DirichletBC(U, u_left, np.where(interfaces.values == left)[0]),  # u left
    df.fem.DirichletBC(W, w_left, np.where(interfaces.values == left)[0]),  # w left
    df.fem.DirichletBC(R, r_left, np.where(interfaces.values == left)[0]),  # r left
]

# Solve nonlinear problem
m = problem.solve()

# Extract solution
u_, w_, r_ = m

print(u_.vector[:])
print(w_.vector[:])
print(r_.vector[:])

# Write output
ofile_u.write_checkpoint(u_, 'disp1')
ofile_w.write_checkpoint(w_, 'disp3')
ofile_r.write_checkpoint(r_, 'rota2')

print('\nDone.')
