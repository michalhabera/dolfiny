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

import mesh_block3d_gmshapi as mg

# Basic settings
name = "solid_disp_tda"
comm = MPI.COMM_WORLD

# Geometry and mesh parameters
dx, dy, dz = 2.0, 0.01, 0.1
nx, ny, nz = 20, 2, 2

# Create the regular mesh of a block with given dimensions
gmsh_model, tdim = mg.mesh_block3d_gmshapi(name, dx, dy, dz, nx, ny, nz, px=1.0, py=1.0, pz=1.0, do_quads=False)

# # Create the regular mesh of a block with given dimensions and save as msh, then read into gmsh model
# mg.mesh_block3d_gmshapi(name, dx, dy, dz, nx, ny, nz, do_quads=False, msh_file=f"{name}.msh")
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

# Solid material parameters
rho = dolfinx.Constant(mesh, 1e-9 * 1e+4)  # [1e-9 * 1e+4 kg/m^3]
eta = dolfinx.Constant(mesh, 1e-9 * 0e+4)  # [1e-9 * 0e+4 kg/m^3/s]
mu = dolfinx.Constant(mesh, 1e-9 * 1e11)  # [1e-9 * 1e+11 N/m^2 = 100 GPa]
la = dolfinx.Constant(mesh, 1e-9 * 1e10)  # [1e-9 * 1e+10 N/m^2 =  10 GPa]

# Load
b = dolfinx.Constant(mesh, [0.0, -10, 0.0])  # [m/s^2]

# Reference values (Bernoulli cantilever, constant line load, free vibration frequency/period)
# import numpy
# u_t = (b.value[1] * rho.value * dz * dy * dx**4) / (8 * 2 * mu.value * dz * dy**3 / 12)
# print(f"u_t = {u_t} [m]")
# ω_0 = numpy.sqrt((2 * mu.value * dz * dy**3 / 12) / (rho.value * dz * dy * dx**4)) * numpy.array([1.875**2, 4.694**2])
# T_0 = 2 * numpy.pi / ω_0
# print(f"ω_0 = {ω_0} [rad/s]")
# print(f"T_0 = {T_0} [s]")

# Global time
time = dolfinx.Constant(mesh, 0.0)  # [s]
# Time step size
dt = dolfinx.Constant(mesh, 1e-2)  # [s]
# Number of time steps
nT = 200

# Define integration measures
dx = ufl.Measure("dx", domain=mesh, subdomain_data=subdomains)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=interfaces)

# Function spaces
Ue = ufl.VectorElement("CG", mesh.ufl_cell(), 2)

Uf = dolfinx.FunctionSpace(mesh, Ue)

# Define functions
u = dolfinx.Function(Uf, name="u")
ut = dolfinx.Function(Uf, name="ut")
utt = dolfinx.Function(Uf, name="utt")

u_ = dolfinx.Function(Uf, name="u_")  # boundary conditions

δu = ufl.TestFunction(Uf)

# Define state and rate as (ordered) list of functions
m, mt, mtt, δm = [u], [ut], [utt], [δu]

# Time integrator
odeint = dolfiny.odeint.ODEInt2(t=time, dt=dt, x=m, xt=mt, xtt=mtt, rho=0.95)

# Configuration gradient
I = ufl.Identity(u.geometric_dimension())  # noqa: E741
F = I + ufl.grad(u)  # deformation gradient as function of displacement

# Strain measures
# E = E(u) total strain
E = 1 / 2 * (F.T * F - I)
# S = S(E) stress
S = 2 * mu * E + la * ufl.tr(E) * I

# Variation of rate of Green-Lagrange strain
δE = dolfiny.expression.derivative(E, m, δm)

# Weak form (as one-form)
f = ufl.inner(δu, rho * utt) * dx + ufl.inner(δu, eta * ut) * dx \
    + ufl.inner(δE, S) * dx \
    - ufl.inner(δu, rho * b) * dx

# Optional: linearise weak form
# f = dolfiny.expression.linearise(f, m)  # linearise around zero state

# Overall form (as one-form)
F = odeint.discretise_in_time(f)
# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)
# Write initial state
ofile.write_function(u, time.value)

# Options for PETSc backend
opts = PETSc.Options(name)

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-09
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
# opts["pc_factor_mat_solver_type"] = "superlu_dist"

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, prefix=name)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_left_dofs_Uf = dolfiny.mesh.locate_dofs_topological(Uf, interfaces, surface_left)

# Process time steps
for time_step in range(1, nT + 1):

    dolfiny.utils.pprint(f"\n+++ Processing time instant = {time.value + dt.value:7.3f} in step {time_step:d}")

    # Stage next time step
    odeint.stage()

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.DirichletBC(u_, surface_left_dofs_Uf),  # disp left (clamped)
    ]

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

    # Update solution states for time integration
    odeint.update()

    # Write output
    ofile.write_function(u, time.value)

ofile.close()
