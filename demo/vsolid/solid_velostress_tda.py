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
name = "solid_velostress_tda"
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
Ve = ufl.VectorElement("CG", mesh.ufl_cell(), 2)
Se = ufl.TensorElement("DG", mesh.ufl_cell(), 1, symmetry=True)

Vf = dolfinx.FunctionSpace(mesh, Ve)
Sf = dolfinx.FunctionSpace(mesh, Se)

# Define functions
v = dolfinx.Function(Vf, name="v")
S = dolfinx.Function(Sf, name="S")

vt = dolfinx.Function(Vf, name="vt")
St = dolfinx.Function(Sf, name="St")

v_ = dolfinx.Function(Vf, name="u_")  # boundary conditions

δv = ufl.TestFunction(Vf)
δS = ufl.TestFunction(Sf)

# Define state and rate as (ordered) list of functions
m, mt, δm = [v, S], [vt, St], [δv, δS]

# Create other functions
u = dolfinx.Function(Vf, name="u")
d = dolfinx.Function(Vf, name="d")  # dummy

# Time integrator
odeint = dolfiny.odeint.ODEInt(t=time, dt=dt, x=m, xt=mt, rho=0.95)

# Expression for time-integrated quantities
u_expr = u + odeint.integral_dt(v)

# Configuration gradient
I = ufl.Identity(v.geometric_dimension())  # noqa: E741
F = I + ufl.grad(u_expr)  # deformation gradient as function of time-integrated velocity
dotF = ufl.grad(v)  # rate of deformation gradient as function of velocity

# Strain measures
# dot E = dot E(u,v) total strain rate
dotE = 1 / 2 * (dotF.T * F + F.T * dotF)
# dot E = dot E(S) elastic strain rate
dotEs = 1 / (2 * mu) * St - la / (2 * mu * (3 * la + 2 * mu)) * ufl.tr(St) * I

# Variation of rate of Green-Lagrange strain
δdotE = dolfiny.expression.derivative(dotE, m, δm)

# Weak form (as one-form)
f = ufl.inner(δv, rho * vt) * dx + ufl.inner(δv, eta * v) * dx \
    + ufl.inner(δdotE, S) * dx + ufl.inner(δS, dotEs - dotE) * dx \
    - ufl.inner(δv, rho * b) * dx

# Optional: linearise weak form
# f = dolfiny.expression.linearise(dolfiny.expression.evaluate(f, u_expr, u), m, [v, S, u])  # linearise

# Overall form (as one-form)
F = odeint.discretise_in_time(f)
# Overall form (as list of forms)
F = dolfiny.function.extract_blocks(F, δm)

# Create output xdmf file -- open in Paraview with Xdmf3ReaderT
ofile = dolfiny.io.XDMFFile(comm, f"{name}.xdmf", "w")
# Write mesh, meshtags
ofile.write_mesh_meshtags(mesh, mts)
# Write initial state
ofile.write_function(v, time.value)
ofile.write_function(S, time.value)
ofile.write_function(u, time.value)

# Options for PETSc backend
opts = PETSc.Options(name)

opts["snes_type"] = "newtonls"
opts["snes_linesearch_type"] = "basic"
opts["snes_atol"] = 1.0e-12
opts["snes_rtol"] = 1.0e-09
opts["snes_max_it"] = 12
opts["ksp_type"] = "preonly"
# opts["ksp_view"] = "::ascii_info_detail"
opts["pc_type"] = "lu"
opts["pc_factor_mat_solver_type"] = "mumps"
# opts["pc_factor_mat_solver_type"] = "superlu_dist"

opts_global = PETSc.Options()
opts_global["mat_mumps_icntl_14"] = 150
opts_global["mat_superlu_dist_rowperm"] = "norowperm"
opts_global["mat_superlu_dist_fact"] = "samepattern_samerowperm"

# Create nonlinear problem: SNES
problem = dolfiny.snesblockproblem.SNESBlockProblem(F, m, prefix=name)

# Identify dofs of function spaces associated with tagged interfaces/boundaries
surface_left_dofs_Vf = dolfiny.mesh.locate_dofs_topological(Vf, interfaces, surface_left)

# Process time steps
for time_step in range(1, nT + 1):

    dolfiny.utils.pprint(f"\n+++ Processing time instant = {time.value + dt.value:7.3f} in step {time_step:d}")

    # Stage next time step
    odeint.stage()

    # Set/update boundary conditions
    problem.bcs = [
        dolfinx.fem.DirichletBC(v_, surface_left_dofs_Vf),  # velocity left (clamped)
    ]

    # Solve nonlinear problem
    problem.solve()

    # Assert convergence of nonlinear solver
    assert problem.snes.getConvergedReason() > 0, "Nonlinear solver did not converge!"

    # Update solution states for time integration
    odeint.update()

    # Assign time-integrated quantities
    dolfiny.interpolation.interpolate(u_expr, d)
    dolfiny.interpolation.interpolate(d, u)

    # Write output
    ofile.write_function(v, time.value)
    ofile.write_function(S, time.value)
    ofile.write_function(u, time.value)

ofile.close()
